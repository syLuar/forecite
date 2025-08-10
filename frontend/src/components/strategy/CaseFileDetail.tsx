import React, { useState, useEffect, useCallback } from 'react';
import { ArrowLeft, Scale, FileText, Save, Trash2, Eye, Search, PenTool, ScrollText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { apiClient } from '../../services/api';
import { ArgumentDraftRequest, ArgumentDraftResponse, SavedArgumentDraft, CaseFileResponse } from '../../types/api';
import SearchModal from '../shared/SearchModal';
import DraftArgumentModal from '../shared/DraftArgumentModal';
import SaveDraftModal from '../shared/SaveDraftModal';
import ConfirmModal from '../shared/ConfirmModal';
import SuccessModal from '../shared/SuccessModal';

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
  const [userFacts, setUserFacts] = useState('');
  const [legalQuestion, setLegalQuestion] = useState('');
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
      if (caseFileData.user_facts) {
        setUserFacts(caseFileData.user_facts);
      }
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

  const handleDraftArgument = async (facts?: string, question?: string) => {
    const factsToUse = facts || userFacts;
    const questionToUse = question || legalQuestion;

    if (!factsToUse.trim()) {
      alert('Please enter the case facts before drafting an argument.');
      return;
    }

    if (!caseFile || caseFile.documents.length === 0) {
      alert('Please add some documents to your case file before drafting an argument.');
      return;
    }

    setIsDrafting(true);
    try {
      // Update case file with current facts/questions
      await apiClient.updateCaseFile(caseFileId, {
        user_facts: factsToUse,
        legal_question: questionToUse
      });

      // Update local state
      setUserFacts(factsToUse);
      setLegalQuestion(questionToUse);

      const request: ArgumentDraftRequest = {
        user_facts: factsToUse,
        legal_question: questionToUse,
        case_file: {
          documents: caseFile.documents,
          total_documents: caseFile.total_documents,
          created_at: caseFile.created_at,
        },
      };

      const response = await apiClient.draftArgument(request);
      setDraftedArgument(response);
      setShowArgumentDraft(true);
      setShowDraftModal(false); // Close the modal
      
    } catch (error) {
      console.error('Argument drafting failed:', error);
      alert('Failed to draft argument. Please try again.');
    } finally {
      setIsDrafting(false);
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

  // Show argument draft view or document view
  if (showArgumentDraft && (draftedArgument || viewingDraft || viewingDocument)) {
    const currentDraft = viewingDraft || draftedArgument;
    const currentDocument = viewingDocument;
    
    // If viewing a document, show document view
    if (currentDocument) {
      return (
        <>
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
            initialFacts={userFacts}
            initialQuestion={legalQuestion}
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
        </>
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
            <div className="prose max-w-none text-gray-700 leading-relaxed">
              <ReactMarkdown 
                components={{
                  h1: ({...props}) => <h1 className="text-2xl font-bold text-gray-900 mb-4" {...props} />,
                  h2: ({...props}) => <h2 className="text-xl font-semibold text-gray-900 mb-3" {...props} />,
                  h3: ({...props}) => <h3 className="text-lg font-medium text-gray-900 mb-2" {...props} />,
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
          initialFacts={userFacts}
          initialQuestion={legalQuestion}
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
      </>
    );
  }

  return (
    <>
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
              <h1 className="text-2xl font-bold text-gray-900">{caseFile.title}</h1>
              <p className="text-gray-600">
                {caseFile.documents.length} documents • {drafts.length} drafts
              </p>
            </div>
          </div>

        {/* Case File Info */}
        {caseFile.description && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
            <p className="text-gray-700">{caseFile.description}</p>
          </div>
        )}

        <div className="max-w-md mx-auto">
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
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 text-sm mb-1">{doc.title}</h4>
                            <p className="text-xs text-gray-600 mb-1">{doc.citation}</p>
                            <p className="text-xs text-gray-500">{doc.jurisdiction} • {doc.year}</p>
                          </div>
                          <div className="flex items-center space-x-1 ml-2">
                            <button
                              onClick={() => handleViewDocument(doc.document_id)}
                              className="p-1 text-gray-400 hover:text-primary"
                              title="View document"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleRemoveDocument(doc.document_id)}
                              className="p-1 text-gray-400 hover:text-red-600"
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
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-medium text-gray-900 text-sm mb-1">{draft.title}</h4>
                            <p className="text-xs text-gray-500 mb-2">{formatDate(draft.created_at)}</p>
                            <div className="flex items-center space-x-2 text-xs">
                              <span>Strength: {formatScore(draft.argument_strength)}</span>
                              <span>•</span>
                              <span>Coverage: {formatScore(draft.precedent_coverage)}</span>
                            </div>
                          </div>
                          <div className="flex items-center space-x-1 ml-2">
                            <button
                              onClick={() => handleViewDraft(draft.id)}
                              className="p-1 text-gray-400 hover:text-primary"
                              title="View draft"
                            >
                              <Eye className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleDeleteDraft(draft.id)}
                              className="p-1 text-gray-400 hover:text-red-600"
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
      </div>

      {/* Modals - Always rendered */}
      <DraftArgumentModal
        isOpen={showDraftModal}
        onClose={() => setShowDraftModal(false)}
        caseFileId={caseFileId}
        onDraft={handleDraftArgument}
        isDrafting={isDrafting}
        initialFacts={userFacts}
        initialQuestion={legalQuestion}
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
    </>
  );
};

export default CaseFileDetail;
