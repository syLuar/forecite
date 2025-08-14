import React, { useState, useEffect, useCallback } from 'react';
import { ArrowLeft, Scale, FileText, Save, Trash2, Eye, Search, PenTool, ScrollText, Gavel, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient, StreamingCallbacks } from '../../services/api';
import { ArgumentDraftRequest, ArgumentDraftResponse, SavedArgumentDraft, CaseFileResponse } from '../../types/api';
import SearchModal from '../shared/SearchModal';
import DraftArgumentModal from '../shared/DraftArgumentModal';
import SaveDraftModal from '../shared/SaveDraftModal';
import ConfirmModal from '../shared/ConfirmModal';
import SuccessModal from '../shared/SuccessModal';
import StreamingProgressModal, { StreamingStep } from '../shared/StreamingProgressModal';
import MootCourt from './MootCourt';
import MootCourtSessionsList from './MootCourtSessionsList';
import MootCourtSessionViewer from './MootCourtSessionViewer';
import AIEditModal from '../shared/AIEditModal';

interface CaseFileDetailProps {
  caseFileId: number;
  onBack: () => void;
}

const CaseFileDetail: React.FC<CaseFileDetailProps> = ({ caseFileId, onBack }) => {
  const [caseFile, setCaseFile] = useState<CaseFileResponse | null>(null);
  const [drafts, setDrafts] = useState<SavedArgumentDraft[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDrafting, setIsDrafting] = useState(false);
  const [draftedArgument, setDraftedArgument] = useState<ArgumentDraftResponse | null>(null);
  const [showArgumentDraft, setShowArgumentDraft] = useState(false);
  const [legalQuestion, setLegalQuestion] = useState('');
  const [additionalInstructions, setAdditionalInstructions] = useState('');
  const [viewingDraft, setViewingDraft] = useState<SavedArgumentDraft | null>(null);
  const [viewingDocument, setViewingDocument] = useState<any | null>(null);
  const [showSearchModal, setShowSearchModal] = useState(false);
  const [showDraftModal, setShowDraftModal] = useState(false);
  const [showSaveDraftModal, setShowSaveDraftModal] = useState(false);
  const [isSavingDraft, setIsSavingDraft] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [confirmAction, setConfirmAction] = useState<(() => void) | null>(null);
  const [confirmDetails, setConfirmDetails] = useState({ title: '', message: '', confirmText: '', isDestructive: false });
  const [isDeleting, setIsDeleting] = useState(false);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successDetails, setSuccessDetails] = useState({ title: '', message: '' });
  const [showMootCourt, setShowMootCourt] = useState(false);
  const [showMootCourtSessions, setShowMootCourtSessions] = useState(false);
  const [viewingMootCourtSession, setViewingMootCourtSession] = useState<number | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editForm, setEditForm] = useState({
    title: '',
    description: '',
    user_facts: '',
    party_represented: ''
  });
  const [isUpdating, setIsUpdating] = useState(false);

  // Draft editing states
  const [isEditingDraft, setIsEditingDraft] = useState(false);
  const [editedDraftContent, setEditedDraftContent] = useState('');
  const [showAIEditModal, setShowAIEditModal] = useState(false);
  const [isAIEditing, setIsAIEditing] = useState(false);
  const [editingDraftId, setEditingDraftId] = useState<number | null>(null);

  // Streaming state
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingSteps, setStreamingSteps] = useState<StreamingStep[]>([]);
  const [streamingError, setStreamingError] = useState<string | null>(null);
  const [showStreamingModal, setShowStreamingModal] = useState(false);
  const [streamingTitle, setStreamingTitle] = useState<string>('Processing');

  const loadCaseFileData = useCallback(async () => {
    try {
      setLoading(true);
      const [caseFileData, draftsList] = await Promise.all([
        apiClient.getCaseFile(caseFileId),
        apiClient.listDraftsForCaseFile(caseFileId)
      ]);
      
      setCaseFile(caseFileData);
      setDrafts(draftsList);
      
      // Pre-populate form if case file has existing facts/questions
      if (caseFileData.legal_question) {
        setLegalQuestion(caseFileData.legal_question);
      }
    } catch (error) {
      console.error('Failed to load case file data:', error);
    } finally {
      setLoading(false);
    }
  }, [caseFileId]);

  useEffect(() => {
    loadCaseFileData();
  }, [loadCaseFileData]);

  const processStreamingChunk = useCallback((chunk: any) => {
    if (chunk.stream_type === 'custom' && chunk.data?.brief_description) {
      // Use step_id from backend if available, otherwise generate one based on brief_description
      const stepId = chunk.data.step_id || chunk.data.brief_description.toLowerCase().replace(/\s+/g, '_');
      
      const newStep: StreamingStep = {
        id: stepId,
        brief_description: chunk.data.brief_description,
        description: chunk.data.description,
        status: chunk.data.status === 'in_progress' ? 'in_progress' : 
                chunk.data.status === 'complete' ? 'complete' : 'active',
        timestamp: new Date()
      };

      setStreamingSteps(prev => {
        // Check if this step already exists
        const existingIndex = prev.findIndex(step => step.id === stepId);
        
        if (existingIndex >= 0) {
          // Update existing step (for completion updates)
          const updated = [...prev];
          updated[existingIndex] = { ...updated[existingIndex], ...newStep };
          return updated;
        } else {
          // Add new step
          return [...prev, newStep];
        }
      });
    }
  }, []);

  const handleStreamingComplete = useCallback((finalResponse: any) => {
    // Mark final step as completed
    setStreamingSteps(prev => prev.map(step => ({ ...step, status: 'completed' as const })));
    
    setDraftedArgument(finalResponse);
    setShowArgumentDraft(true);
    setShowDraftModal(false);
    setIsStreaming(false);
    setIsDrafting(false);
  }, []);

  const handleStreamingError = useCallback((errorMessage: string) => {
    setStreamingError(errorMessage);
    setStreamingSteps(prev => prev.map(step => ({ 
      ...step, 
      status: step.status === 'active' ? 'error' as const : step.status 
    })));
    setIsStreaming(false);
    setIsDrafting(false);
    alert('Failed to draft argument. Please try again.');
  }, []);

  const handleDraftArgument = async (legalQuestion?: string, additionalInstructions?: string) => {
    if (!caseFile) {
      alert('Case file not found.');
      return;
    }

    if (!caseFile.user_facts || !caseFile.user_facts.trim()) {
      alert('Please add case facts to your case file before drafting an argument. Case facts are now stored as part of the case file.');
      return;
    }

    if (caseFile.documents.length === 0) {
      alert('Please add some documents to your case file before drafting an argument.');
      return;
    }

    setIsDrafting(true);
    setStreamingError(null);
    setStreamingSteps([]);

    try {
      const request: ArgumentDraftRequest = {
        case_file_id: caseFileId,
        legal_question: legalQuestion || '',
        additional_drafting_instructions: additionalInstructions || '',
      };

      // Check if streaming is enabled
      const streamingEnabled = process.env.REACT_APP_STREAMING === 'true';
      
      if (streamingEnabled) {
        setIsStreaming(true);
        setStreamingTitle('Drafting Legal Argument');
        setShowDraftModal(false);
        
        const streamingCallbacks: StreamingCallbacks = {
          onChunk: processStreamingChunk,
          onComplete: handleStreamingComplete,
          onError: handleStreamingError
        };

        await apiClient.draftArgument(request, streamingCallbacks);
      } else {
        // Fallback to non-streaming
        const response = await apiClient.draftArgument(request);
        setDraftedArgument(response);
        setShowArgumentDraft(true);
        setShowDraftModal(false);
        setIsDrafting(false);
      }
      
    } catch (error) {
      console.error('Argument drafting failed:', error);
      alert('Failed to draft argument. Please try again.');
      setIsDrafting(false);
      setIsStreaming(false);
    }
  };

  const handleSaveDraft = async () => {
    if (!draftedArgument) {
      alert('No draft to save.');
      return;
    }
    setShowSaveDraftModal(true);
  };

  const handleSaveDraftWithTitle = async (title: string) => {
    if (!draftedArgument) {
      return;
    }

    setIsSavingDraft(true);
    try {
      await apiClient.saveDraft(caseFileId, draftedArgument, title);
      await loadCaseFileData(); // Reload to show new draft
      setShowSaveDraftModal(false);
      setShowArgumentDraft(false);
      setDraftedArgument(null);
      // You could add a success toast here instead of alert
    } catch (error) {
      console.error('Failed to save draft:', error);
      alert('Failed to save draft. Please try again.');
    } finally {
      setIsSavingDraft(false);
    }
  };

  const handleViewDraft = async (draftId: number) => {
    try {
      const draft = await apiClient.getDraft(draftId);
      setViewingDraft(draft);
      setShowArgumentDraft(true);
    } catch (error) {
      console.error('Failed to load draft:', error);
      alert('Failed to load draft. Please try again.');
    }
  };

  const handleViewDocument = async (documentId: string) => {
    try {
      const document = await apiClient.getDocumentFromCaseFile(caseFileId, documentId);
      setViewingDocument(document);
      setShowArgumentDraft(true);
    } catch (error) {
      console.error('Failed to load document:', error);
      alert('Failed to load document. Please try again.');
    }
  };

  const showConfirmDialog = (title: string, message: string, confirmText: string, action: () => void, isDestructive: boolean = true) => {
    setConfirmDetails({ title, message, confirmText, isDestructive });
    setConfirmAction(() => action);
    setShowConfirmModal(true);
  };

  const handleConfirmAction = async () => {
    if (confirmAction) {
      setIsDeleting(true);
      try {
        await confirmAction();
        setShowConfirmModal(false);
        setConfirmAction(null);
      } catch (error) {
        // Error handling is done in the specific action functions
      } finally {
        setIsDeleting(false);
      }
    }
  };

  const handleRemoveDocument = async (documentId: string) => {
    const removeAction = async () => {
      try {
        await apiClient.removeDocumentFromCaseFile(caseFileId, documentId);
        await loadCaseFileData();
      } catch (error) {
        console.error('Failed to remove document:', error);
        alert('Failed to remove document. Please try again.');
        throw error;
      }
    };

    showConfirmDialog(
      'Remove Document',
      'Are you sure you want to remove this document from the case file?',
      'Remove Document',
      removeAction,
      true
    );
  };

  const handleDeleteDraft = async (draftId: number) => {
    const deleteAction = async () => {
      try {
        await apiClient.deleteDraft(draftId);
        await loadCaseFileData();
      } catch (error) {
        console.error('Failed to delete draft:', error);
        alert('Failed to delete draft. Please try again.');
        throw error;
      }
    };

    showConfirmDialog(
      'Delete Draft',
      'Are you sure you want to delete this draft? This action cannot be undone.',
      'Delete Draft',
      deleteAction,
      true
    );
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatScore = (score?: number) => {
    if (score === undefined || score === null) return 'N/A';
    return `${Math.round(score * 100)}%`;
  };

  const handleDocumentAdded = async () => {
    await loadCaseFileData();
    setSuccessDetails({
      title: 'Document Added Successfully!',
      message: 'The document has been added to your case file and is now available for use in your legal arguments.'
    });
    setShowSuccessModal(true);
  };

  const handleMootCourtClick = () => {
    setShowMootCourt(true);
  };

  const handleMootCourtSessionsClick = () => {
    setShowMootCourtSessions(true);
  };

  const handleViewMootCourtSession = (sessionId: number) => {
    if (sessionId === -1) {
      // -1 indicates new session
      setShowMootCourtSessions(false);
      setShowMootCourt(true);
    } else {
      setViewingMootCourtSession(sessionId);
      setShowMootCourtSessions(false);
    }
  };

  const handleBackFromMootCourtSessions = () => {
    setShowMootCourtSessions(false);
    setViewingMootCourtSession(null);
  };

  const handleBackFromMootCourtSession = () => {
    setViewingMootCourtSession(null);
    setShowMootCourtSessions(true);
  };

  // Missing function definitions
  const handleEditCaseFile = () => {
    setEditForm({
      title: caseFile?.title || '',
      description: caseFile?.description || '',
      user_facts: caseFile?.user_facts || '',
      party_represented: caseFile?.party_represented || ''
    });
    setShowEditModal(true);
  };

  const handleUpdateCaseFile = async () => {
    if (!caseFile) return;

    setIsUpdating(true);
    try {
      await apiClient.updateCaseFile(caseFileId, {
        title: editForm.title,
        description: editForm.description,
        user_facts: editForm.user_facts,
        party_represented: editForm.party_represented
      });
      
      // Reload case file data
      await loadCaseFileData();
      setShowEditModal(false);
      
      setSuccessDetails({
        title: 'Case File Updated',
        message: 'Your case file has been successfully updated.'
      });
      setShowSuccessModal(true);
    } catch (error) {
      console.error('Failed to update case file:', error);
      alert('Failed to update case file. Please try again.');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleStartEditDraft = (draft: SavedArgumentDraft) => {
    setEditedDraftContent(draft.drafted_argument);
    setEditingDraftId(draft.id);
    setIsEditingDraft(true);
  };

  const handleStartAIEdit = (draft: SavedArgumentDraft) => {
    setEditedDraftContent(draft.drafted_argument);
    setEditingDraftId(draft.id);
    setShowAIEditModal(true);
  };

  const handleSaveEditedDraft = async () => {
    if (!editingDraftId || !editedDraftContent.trim()) return;

    setIsUpdating(true);
    try {
      await apiClient.updateDraft(editingDraftId, editedDraftContent);
      
      // Reload drafts and update viewing draft if it's the same one
      await loadCaseFileData();
      if (viewingDraft && viewingDraft.id === editingDraftId) {
        const updatedDraft = await apiClient.getDraft(editingDraftId);
        setViewingDraft(updatedDraft);
      }
      
      setIsEditingDraft(false);
      setEditingDraftId(null);
      setEditedDraftContent('');
      
      setSuccessDetails({
        title: 'Draft Updated',
        message: 'Your draft has been successfully updated.'
      });
      setShowSuccessModal(true);
    } catch (error) {
      console.error('Failed to update draft:', error);
      alert('Failed to update draft. Please try again.');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleCancelEditDraft = () => {
    setIsEditingDraft(false);
    setEditingDraftId(null);
    setEditedDraftContent('');
  };

  const handleAIEdit = async (editInstructions: string) => {
    if (!editingDraftId) return;

    setIsAIEditing(true);
    setStreamingError(null);
    setStreamingSteps([]);

    try {
      // Check if streaming is enabled
      const streamingEnabled = process.env.REACT_APP_STREAMING === 'true';
      
      if (streamingEnabled) {
        setIsStreaming(true);
        setStreamingTitle('Editing Draft with AI');
        setShowAIEditModal(false);
        
        const streamingCallbacks: StreamingCallbacks = {
          onChunk: processStreamingChunk,
          onComplete: async (finalResponse: any) => {
            // Mark final step as completed
            setStreamingSteps(prev => prev.map(step => ({ ...step, status: 'completed' as const })));
            
            // Reload drafts and update viewing draft if it's the same one
            await loadCaseFileData();
            if (viewingDraft && viewingDraft.id === editingDraftId) {
              const updatedDraft = await apiClient.getDraft(editingDraftId);
              setViewingDraft(updatedDraft);
            }
            
            setEditingDraftId(null);
            setIsStreaming(false);
            setIsAIEditing(false);
            
            setSuccessDetails({
              title: 'AI Edit Completed',
              message: 'Your draft has been successfully edited with AI assistance.'
            });
            setShowSuccessModal(true);
          },
          onError: (errorMessage: string) => {
            setStreamingError(errorMessage);
            setStreamingSteps(prev => prev.map(step => ({ 
              ...step, 
              status: step.status === 'active' ? 'error' as const : step.status 
            })));
            setIsStreaming(false);
            setIsAIEditing(false);
            alert('Failed to edit draft with AI. Please try again.');
          }
        };

        await apiClient.editDraftWithAI(editingDraftId, editInstructions, streamingCallbacks);
      } else {
        // Fallback to non-streaming
        await apiClient.editDraftWithAI(editingDraftId, editInstructions);
        
        // Reload drafts and update viewing draft if it's the same one
        await loadCaseFileData();
        if (viewingDraft && viewingDraft.id === editingDraftId) {
          const updatedDraft = await apiClient.getDraft(editingDraftId);
          setViewingDraft(updatedDraft);
        }
        
        setShowAIEditModal(false);
        setEditingDraftId(null);
        setIsAIEditing(false);
        
        setSuccessDetails({
          title: 'AI Edit Completed',
          message: 'Your draft has been successfully edited with AI assistance.'
        });
        setShowSuccessModal(true);
      }
    } catch (error) {
      console.error('Failed to edit draft with AI:', error);
      alert('Failed to edit draft with AI. Please try again.');
      setIsAIEditing(false);
      setIsStreaming(false);
    }
  };

  // Show streaming modal when streaming starts
  useEffect(() => {
    if (isStreaming) {
      setShowStreamingModal(true);
    }
  }, [isStreaming]);

  const handleCloseStreamingModal = () => {
    // Only allow closing if not currently streaming or if there's an error
    if (!isStreaming || streamingError) {
      setShowStreamingModal(false);
      setStreamingSteps([]);
      setStreamingError(null);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-center py-20">
            <Scale className="h-24 w-24 text-primary animate-bounce mb-4" />
            <p className="text-lg text-gray-600 font-medium ml-4">Loading case file...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!caseFile) {
    return (
      <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
        <div className="max-w-6xl mx-auto">
          <div className="text-center py-20">
            <p className="text-lg text-gray-600">Case file not found.</p>
            <button
              onClick={onBack}
              className="mt-4 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Show MootCourt component
  if (showMootCourt) {
    return (
      <MootCourt
        caseFileId={caseFileId}
        caseFileTitle={caseFile.title}
        onBack={() => setShowMootCourt(false)}
      />
    );
  }

  // Show MootCourtSessionsList component
  if (showMootCourtSessions) {
    return (
      <MootCourtSessionsList
        caseFileId={caseFileId}
        caseFileTitle={caseFile.title}
        onBack={handleBackFromMootCourtSessions}
        onViewSession={handleViewMootCourtSession}
      />
    );
  }

  // Show MootCourtSessionViewer component
  if (viewingMootCourtSession) {
    return (
      <MootCourtSessionViewer
        sessionId={viewingMootCourtSession}
        caseFileTitle={caseFile.title}
        onBack={handleBackFromMootCourtSession}
      />
    );
  }
  
  // Show argument draft view or document view
  if (showArgumentDraft && (draftedArgument || viewingDraft || viewingDocument)) {
    const currentDraft = viewingDraft || draftedArgument;
    const currentDocument = viewingDocument;
    
    // If viewing a document, show document view
    if (currentDocument) {
      return (
        <div>
          <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
          <div className="max-w-6xl mx-auto">
            {/* Header with back button */}
            <div className="flex items-center justify-between mb-8">
              <button
                onClick={() => {
                  setShowArgumentDraft(false);
                  setViewingDocument(null);
                }}
                className="flex items-center text-primary hover:text-primary-700 font-medium"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Case File
              </button>
              <div className="text-right">
                <h1 className="text-2xl font-bold text-gray-900">{currentDocument.title}</h1>
                <p className="text-gray-600">{currentDocument.citation}</p>
              </div>
            </div>

            {/* Document Information */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Details</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Citation</h4>
                  <p className="text-gray-700">{currentDocument.citation}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Jurisdiction</h4>
                  <p className="text-gray-700">{currentDocument.jurisdiction}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Year</h4>
                  <p className="text-gray-700">{currentDocument.year}</p>
                </div>
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">Relevance Score</h4>
                  <p className="text-gray-700">{formatScore(currentDocument.relevance_score_percent / 100)}</p>
                </div>
              </div>
            </div>

            {/* Key Holdings */}
            {currentDocument.key_holdings && currentDocument.key_holdings.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Holdings</h3>
                <div className="space-y-3">
                  {currentDocument.key_holdings.map((holding: string, index: number) => (
                    <div key={index} className="border-l-4 border-primary pl-4">
                      <p className="text-gray-700">{holding}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Document Content */}
            {currentDocument.selected_chunks && currentDocument.selected_chunks.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Document Content</h3>
                <div className="space-y-6">
                  {currentDocument.selected_chunks.map((chunk: any, index: number) => (
                    <div key={index} className="border-b border-gray-200 pb-6 last:border-b-0">
                      {chunk.summary && (
                        <div className="mb-4">
                          <h4 className="font-medium text-gray-900 mb-2">Summary</h4>
                          <p className="text-gray-700 italic">{chunk.summary}</p>
                        </div>
                      )}
                      {chunk.text && (
                        <div className="mb-4">
                          <h4 className="font-medium text-gray-900 mb-2">Content</h4>
                          <div className="prose max-w-none text-gray-700 leading-relaxed bg-gray-50 p-4 rounded-lg">
                            {chunk.text.split('\n').map((paragraph: string, pIndex: number) => (
                              paragraph.trim() ? <p key={pIndex} className="mb-2">{paragraph}</p> : <br key={pIndex} />
                            ))}
                          </div>
                        </div>
                      )}
                      {/* Extracted entities */}
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                        {chunk.cases && chunk.cases.length > 0 && (
                          <div>
                            <h5 className="font-medium text-gray-900 mb-1">Cases</h5>
                            <div className="flex flex-wrap gap-1">
                              {chunk.cases.map((item: string, i: number) => (
                                <span key={i} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">{item}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {chunk.statutes && chunk.statutes.length > 0 && (
                          <div>
                            <h5 className="font-medium text-gray-900 mb-1">Statutes</h5>
                            <div className="flex flex-wrap gap-1">
                              {chunk.statutes.map((item: string, i: number) => (
                                <span key={i} className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">{item}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {chunk.concepts && chunk.concepts.length > 0 && (
                          <div>
                            <h5 className="font-medium text-gray-900 mb-1">Legal Concepts</h5>
                            <div className="flex flex-wrap gap-1">
                              {chunk.concepts.map((item: string, i: number) => (
                                <span key={i} className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs">{item}</span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* User Notes */}
            {currentDocument.user_notes && (
              <div className="bg-white rounded-lg shadow-md p-6 mt-8 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Notes</h3>
                <p className="text-gray-700">{currentDocument.user_notes}</p>
              </div>
            )}
          </div>
          </div>

          {/* Modals - Always rendered */}
          <DraftArgumentModal
            isOpen={showDraftModal}
            onClose={() => setShowDraftModal(false)}
            caseFileId={caseFileId}
            onDraft={handleDraftArgument}
            isDrafting={isDrafting}
            initialQuestion={legalQuestion}
            initialInstructions={additionalInstructions}
          />

          <SaveDraftModal
            isOpen={showSaveDraftModal}
            onClose={() => setShowSaveDraftModal(false)}
            onSave={handleSaveDraftWithTitle}
            isSaving={isSavingDraft}
          />

          <SearchModal
            isOpen={showSearchModal}
            onClose={() => setShowSearchModal(false)}
            caseFileId={caseFileId}
            onDocumentAdded={handleDocumentAdded}
          />

          <ConfirmModal
            isOpen={showConfirmModal}
            onClose={() => setShowConfirmModal(false)}
            onConfirm={handleConfirmAction}
            title={confirmDetails.title}
            message={confirmDetails.message}
            confirmText={confirmDetails.confirmText}
            isDestructive={confirmDetails.isDestructive}
            isLoading={isDeleting}
          />

          <SuccessModal
            isOpen={showSuccessModal}
            onClose={() => setShowSuccessModal(false)}
            title={successDetails.title}
            message={successDetails.message}
          />

          {/* AI Edit Modal */}
          <AIEditModal
            isOpen={showAIEditModal}
            onClose={() => setShowAIEditModal(false)}
            draftId={editingDraftId || 0}
            currentContent={editedDraftContent}
            onEdit={handleAIEdit}
            isEditing={isAIEditing}
          />

          {/* Streaming Progress Modal */}
          <StreamingProgressModal
            isOpen={showStreamingModal}
            onClose={handleCloseStreamingModal}
            steps={streamingSteps}
            isStreaming={isStreaming}
            error={streamingError || undefined}
            title={streamingTitle}
            allowClose={!!streamingError}
          />
        </div>
      );
    }
    
    // If viewing a draft, show draft view
    if (!currentDraft) return null;

    return (
      <>
        <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
          <div className="max-w-6xl mx-auto">
            {/* Header with back button */}
            <div className="flex items-center justify-between mb-8">
              <button
                onClick={() => {
                  setShowArgumentDraft(false);
                  setViewingDraft(null);
                  setDraftedArgument(null);
                  setViewingDocument(null);
                }}
                className="flex items-center text-primary hover:text-primary-700 font-medium"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Case File
              </button>
              <div className="text-right">
                <h1 className="text-2xl font-bold text-gray-900">
                  {viewingDraft ? viewingDraft.title : 'Legal Argument Draft'}
                </h1>
                <p className="text-gray-600">
                  {viewingDraft 
                    ? `Created: ${formatDate(viewingDraft.created_at)}`
                    : `Generated in ${draftedArgument?.execution_time?.toFixed(2)}s`
                  }
                </p>
              </div>
            </div>

            {/* Strategy Summary */}
            {currentDraft.strategy && (
              <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategy Overview</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Main Thesis</h4>
                    <p className="text-gray-700">{currentDraft.strategy.main_thesis}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Argument Type</h4>
                    <p className="text-gray-700 capitalize">{currentDraft.strategy.argument_type}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Strength Assessment</h4>
                    <p className="text-gray-700">{formatScore(currentDraft.argument_strength)}</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">Precedent Coverage</h4>
                    <p className="text-gray-700">{formatScore(currentDraft.precedent_coverage)}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Drafted Argument */}
            <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Legal Argument</h3>
                <div className="flex items-center gap-3">
                  {/* Edit buttons for saved drafts */}
                  {viewingDraft && !isEditingDraft && (
                    <>
                      <button
                        onClick={() => handleStartEditDraft(viewingDraft)}
                        className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                      >
                        <PenTool className="h-4 w-4 mr-2" />
                        Manual Edit
                      </button>
                      <button
                        onClick={() => handleStartAIEdit(viewingDraft)}
                        className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                      >
                        <Bot className="h-4 w-4 mr-2" />
                        AI Edit
                      </button>
                    </>
                  )}
                  
                  {/* Save/Cancel buttons for manual editing */}
                  {isEditingDraft && (
                    <>
                      <button
                        onClick={handleSaveEditedDraft}
                        className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                      >
                        <Save className="h-4 w-4 mr-2" />
                        Save Changes
                      </button>
                      <button
                        onClick={handleCancelEditDraft}
                        className="flex items-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                      >
                        Cancel
                      </button>
                    </>
                  )}
                  
                  {/* Save draft button for new drafts */}
                  {draftedArgument && !viewingDraft && (
                    <button
                      onClick={handleSaveDraft}
                      className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                    >
                      <Save className="h-4 w-4 mr-2" />
                      Save Draft
                    </button>
                  )}
                </div>
              </div>
              
              {/* Draft content - either editable or read-only */}
              {isEditingDraft ? (
                <div>
                  <textarea
                    value={editedDraftContent}
                    onChange={(e) => setEditedDraftContent(e.target.value)}
                    className="w-full h-96 px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-gray-900 text-base resize-vertical"
                    placeholder="Edit your legal argument..."
                  />
                  <p className="text-sm text-gray-600 mt-2">
                    {editedDraftContent.split(' ').length} words
                  </p>
                </div>
              ) : (
                <div className="prose max-w-none text-gray-700 leading-relaxed">
                  <ReactMarkdown 
                    components={{
                      h1: ({children, ...props}) => children ? <h1 className="text-2xl font-bold text-gray-900 mb-4" {...props}>{children}</h1> : null,
                      h2: ({children, ...props}) => children ? <h2 className="text-xl font-semibold text-gray-900 mb-3" {...props}>{children}</h2> : null,
                      h3: ({children, ...props}) => children ? <h3 className="text-lg font-medium text-gray-900 mb-2" {...props}>{children}</h3> : null,
                      p: ({...props}) => <p className="mb-4 text-gray-700 leading-relaxed" {...props} />,
                      ul: ({...props}) => <ul className="list-disc list-inside mb-4 space-y-1" {...props} />,
                      ol: ({...props}) => <ol className="list-decimal list-inside mb-4 space-y-1" {...props} />,
                      li: ({...props}) => <li className="text-gray-700" {...props} />,
                      blockquote: ({...props}) => <blockquote className="border-l-4 border-primary pl-4 italic text-gray-600 mb-4" {...props} />,
                      strong: ({...props}) => <strong className="font-semibold text-gray-900" {...props} />,
                      em: ({...props}) => <em className="italic" {...props} />,
                      code: ({...props}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props} />,
                    }}
                  >
                    {currentDraft.drafted_argument}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {/* Key Arguments */}
            {currentDraft.strategy?.key_arguments && (
              <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Arguments</h3>
                <div className="space-y-4">
                  {currentDraft.strategy.key_arguments.map((arg: any, index: number) => (
                    <div key={index} className="border-l-4 border-primary pl-4">
                      <h4 className="font-medium text-gray-900 mb-2">Argument {index + 1}</h4>
                      <p className="text-gray-700 mb-2">{arg.argument}</p>
                      <p className="text-sm text-gray-600"><strong>Authority:</strong> {arg.supporting_authority}</p>
                      <p className="text-sm text-gray-600"><strong>Factual Basis:</strong> {arg.factual_basis}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modals - Always rendered */}
        <DraftArgumentModal
          isOpen={showDraftModal}
          onClose={() => setShowDraftModal(false)}
          caseFileId={caseFileId}
          onDraft={handleDraftArgument}
          isDrafting={isDrafting}
          initialQuestion={legalQuestion}
          initialInstructions={additionalInstructions}
        />

        <SaveDraftModal
          isOpen={showSaveDraftModal}
          onClose={() => setShowSaveDraftModal(false)}
          onSave={handleSaveDraftWithTitle}
          isSaving={isSavingDraft}
        />

        <SearchModal
          isOpen={showSearchModal}
          onClose={() => setShowSearchModal(false)}
          caseFileId={caseFileId}
          onDocumentAdded={handleDocumentAdded}
        />

        <ConfirmModal
          isOpen={showConfirmModal}
          onClose={() => setShowConfirmModal(false)}
          onConfirm={handleConfirmAction}
          title={confirmDetails.title}
          message={confirmDetails.message}
          confirmText={confirmDetails.confirmText}
          isDestructive={confirmDetails.isDestructive}
          isLoading={isDeleting}
        />

        <SuccessModal
          isOpen={showSuccessModal}
          onClose={() => setShowSuccessModal(false)}
          title={successDetails.title}
          message={successDetails.message}
        />

        {/* AI Edit Modal */}
        <AIEditModal
          isOpen={showAIEditModal}
          onClose={() => setShowAIEditModal(false)}
          draftId={editingDraftId || 0}
          currentContent={editedDraftContent}
          onEdit={handleAIEdit}
          isEditing={isAIEditing}
        />

        {/* Streaming Progress Modal */}
        <StreamingProgressModal
          isOpen={showStreamingModal}
          onClose={handleCloseStreamingModal}
          steps={streamingSteps}
          isStreaming={isStreaming}
          error={streamingError || undefined}
          title={streamingTitle}
          allowClose={!!streamingError}
        />
      </>
    );
  }

  // Fallback main case file detail view (simplified placeholder to ensure component always returns)
  return (
    <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBack}
            className="flex items-center text-primary hover:text-primary-700 font-medium"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            Back to Strategy
          </button>
          <div className="text-right">
            <div className="flex items-center gap-4">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{caseFile.title}</h1>
                <p className="text-gray-600">
                  {caseFile.documents.length} documents • {drafts.length} drafts
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleEditCaseFile}
                  className="flex items-center px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                  title="Edit case file information"
                >
                  <PenTool className="h-4 w-4 mr-1.5" />
                  Edit
                </button>
                <button
                  onClick={handleMootCourtSessionsClick}
                  className="flex items-center px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                  title="View Moot Court Sessions"
                >
                  <Gavel className="h-4 w-4 mr-1.5" />
                  Sessions
                </button>
                <button
                  onClick={handleMootCourtClick}
                  className="flex items-center px-3 py-2 text-sm bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-lg hover:from-amber-600 hover:to-orange-600 transition-all duration-300 shadow-lg"
                  title="Open Moot Court - Practice your arguments"
                >
                  <Gavel className="h-4 w-4 mr-1.5" />
                  Moot Court
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Description */}
        {caseFile.description && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
            <p className="text-gray-700 whitespace-pre-line">{caseFile.description}</p>
          </div>
        )}

        {/* Facts */}
        {caseFile.user_facts && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Case Facts</h3>
            <p className="text-gray-700 whitespace-pre-line">{caseFile.user_facts}</p>
          </div>
        )}

        {/* Streaming Progress for Argument Drafting */}
        <StreamingProgressModal
          isOpen={showStreamingModal}
          onClose={handleCloseStreamingModal}
          steps={streamingSteps}
          isStreaming={isStreaming}
          error={streamingError || undefined}
          title={streamingTitle}
          allowClose={!!streamingError}
        />

        <div className="w-full">
          <div className="space-y-6">
            {/* Documents */}
            <div className="bg-white rounded-lg shadow-md border border-gray-200">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                    <FileText className="h-5 w-5 mr-2" />
                    Documents ({caseFile.documents.length})
                  </h3>
                  <button
                    onClick={() => setShowSearchModal(true)}
                    className="flex items-center px-3 py-1.5 text-sm bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors duration-200"
                    title="Search for documents to add to case file"
                  >
                    <Search className="h-4 w-4 mr-1.5" />
                    Search for Documents
                  </button>
                </div>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {caseFile.documents.length > 0 ? (
                  <div className="divide-y divide-gray-200">
                    {caseFile.documents.map((doc: any) => (
                      <div key={doc.document_id} className="p-4">
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-4 mb-2">
                              <h4 className="font-medium text-gray-900 text-base truncate">{doc.title}</h4>
                              {doc.relevance_score_percent !== undefined && (
                                <span className="px-3 py-1 text-sm font-medium bg-blue-100 text-blue-800 rounded-full whitespace-nowrap">
                                  {Math.round(doc.relevance_score_percent)}% relevant
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 truncate">{doc.citation}</p>
                          </div>
                          <div className="flex items-center gap-2 text-sm text-gray-500 whitespace-nowrap">
                            <span>{doc.jurisdiction}</span>
                            <span className="text-gray-400">•</span>
                            <span>{doc.year}</span>
                          </div>
                          <div className="flex items-center space-x-2 flex-shrink-0">
                            <button
                              onClick={() => handleViewDocument(doc.document_id)}
                              className="p-2 text-gray-400 hover:text-primary hover:bg-gray-100 rounded-lg transition-colors"
                              title="View document"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleRemoveDocument(doc.document_id)}
                              className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                              title="Remove document"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-6 text-center text-gray-500">
                    <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                    <p>No documents</p>
                    <p className="text-sm">Add documents from Search</p>
                  </div>
                )}
              </div>
            </div>

            {/* Saved Drafts */}
            <div className="bg-white rounded-lg shadow-md border border-gray-200">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                    <ScrollText className="h-5 w-5 mr-2" />
                    Saved Drafts ({drafts.length})
                  </h3>
                  <button
                    onClick={() => setShowDraftModal(true)}
                    className="flex items-center px-3 py-1.5 text-sm bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors duration-200"
                    title="Create a new legal argument"
                  >
                    <PenTool className="h-4 w-4 mr-1.5" />
                    Add Argument
                  </button>
                </div>
              </div>
              <div className="max-h-80 overflow-y-auto">
                {drafts.length > 0 ? (
                  <div className="divide-y divide-gray-200">
                    {drafts.map((draft) => (
                      <div key={draft.id} className="p-4">
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-4">
                              <h4 className="font-medium text-gray-900 text-base truncate">{draft.title}</h4>
                              <div className="flex items-center gap-3">
                                <span className={`px-3 py-1 text-sm font-medium rounded-full whitespace-nowrap ${
                                  (draft.argument_strength || 0) >= 0.8 ? 'bg-green-100 text-green-800' :
                                  (draft.argument_strength || 0) >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {formatScore(draft.argument_strength)} strength
                                </span>
                                <span className={`px-3 py-1 text-sm font-medium rounded-full whitespace-nowrap ${
                                  (draft.precedent_coverage || 0) >= 0.8 ? 'bg-purple-100 text-purple-800' :
                                  (draft.precedent_coverage || 0) >= 0.6 ? 'bg-indigo-100 text-indigo-800' :
                                  'bg-gray-100 text-gray-800'
                                }`}>
                                  {formatScore(draft.precedent_coverage)} coverage
                                </span>
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-gray-500 whitespace-nowrap">{formatDate(draft.created_at)}</p>
                          <div className="flex items-center space-x-2 flex-shrink-0">
                            <button
                              onClick={() => handleViewDraft(draft.id)}
                              className="p-2 text-gray-400 hover:text-primary hover:bg-gray-100 rounded-lg transition-colors"
                              title="View draft"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleDeleteDraft(draft.id)}
                              className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                              title="Delete draft"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-6 text-center text-gray-500">
                    <ScrollText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                    <p>No drafts saved</p>
                    <p className="text-sm">Create an argument to see it here</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Modals - Always rendered */}
      <DraftArgumentModal
        isOpen={showDraftModal}
        onClose={() => setShowDraftModal(false)}
        caseFileId={caseFileId}
        onDraft={handleDraftArgument}
        isDrafting={isDrafting}
        initialQuestion={legalQuestion}
        initialInstructions={additionalInstructions}
      />

      <SaveDraftModal
        isOpen={showSaveDraftModal}
        onClose={() => setShowSaveDraftModal(false)}
        onSave={handleSaveDraftWithTitle}
        isSaving={isSavingDraft}
      />

      <SearchModal
        isOpen={showSearchModal}
        onClose={() => setShowSearchModal(false)}
        caseFileId={caseFileId}
        onDocumentAdded={handleDocumentAdded}
      />

      <ConfirmModal
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        onConfirm={handleConfirmAction}
        title={confirmDetails.title}
        message={confirmDetails.message}
        confirmText={confirmDetails.confirmText}
        isDestructive={confirmDetails.isDestructive}
        isLoading={isDeleting}
      />

      <SuccessModal
        isOpen={showSuccessModal}
        onClose={() => setShowSuccessModal(false)}
        title={successDetails.title}
        message={successDetails.message}
      />

      {/* AI Edit Modal */}
      <AIEditModal
        isOpen={showAIEditModal}
        onClose={() => setShowAIEditModal(false)}
        draftId={editingDraftId || 0}
        currentContent={editedDraftContent}
        onEdit={handleAIEdit}
        isEditing={isAIEditing}
      />

      {/* Streaming Progress Modal */}
      <StreamingProgressModal
        isOpen={showStreamingModal}
        onClose={handleCloseStreamingModal}
        steps={streamingSteps}
        isStreaming={isStreaming}
        error={streamingError || undefined}
        title={streamingTitle}
        allowClose={!!streamingError}
      />

      {/* Edit Case File Modal */}
      {showEditModal && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
            onClick={() => setShowEditModal(false)}
          />
          {/* Modal */}
          <div className="flex min-h-full items-center justify-center p-4">
            <div className="relative bg-white rounded-lg shadow-xl w-full max-w-3xl">
              <div className="p-6 border-b">
                <h3 className="text-lg font-semibold text-gray-900">Edit Case File</h3>
              </div>
              <div className="p-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Title</label>
                    <input
                      type="text"
                      value={editForm.title}
                      onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
                      className="block w-full p-3 border rounded-lg focus:ring focus:ring-primary focus:outline-none"
                      placeholder="Enter case file title"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                    <textarea
                      value={editForm.description}
                      onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                      className="block w-full p-3 border rounded-lg focus:ring focus:ring-primary focus:outline-none"
                      rows={3}
                      placeholder="Enter case file description"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Facts</label>
                    <textarea
                      value={editForm.user_facts}
                      onChange={(e) => setEditForm({ ...editForm, user_facts: e.target.value })}
                      className="block w-full p-3 border rounded-lg focus:ring focus:ring-primary focus:outline-none"
                      rows={6}
                      placeholder="Enter case facts"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Party Represented</label>
                    <input
                      type="text"
                      value={editForm.party_represented}
                      onChange={(e) => setEditForm({ ...editForm, party_represented: e.target.value })}
                      className="block w-full p-3 border rounded-lg focus:ring focus:ring-primary focus:outline-none"
                      placeholder="Party you represent"
                    />
                  </div>
                </div>
              </div>
              <div className="p-6 border-t flex justify-end gap-4">
                <button
                  onClick={() => setShowEditModal(false)}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateCaseFile}
                  className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors"
                  disabled={isUpdating}
                >
                  {isUpdating ? 'Updating...' : 'Update Case File'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CaseFileDetail;
