import React, { useState, useEffect } from 'react';
import { Search, FolderOpen, Check, Plus, FileText } from 'lucide-react';
import Modal from './Modal';
import { LegalDocument } from '../../data/mockSearchData';
import { apiClient } from '../../services/api';
import { CaseFileListItem, CreateCaseFileRequest, AddDocumentToCaseFileRequest } from '../../types/api';

interface SelectCaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  document: LegalDocument;
  onAddToCase?: (caseItem: any, document: LegalDocument) => void;
}

const SelectCaseModal: React.FC<SelectCaseModalProps> = ({
  isOpen,
  onClose,
  document,
  onAddToCase
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);
  const [caseFiles, setCaseFiles] = useState<CaseFileListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createForm, setCreateForm] = useState<CreateCaseFileRequest>({
    title: '',
    description: '',
    user_facts: '',
    party_represented: ''
  });
  const [isCreating, setIsCreating] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  
  useEffect(() => {
    if (isOpen) {
      loadCaseFiles();
    }
  }, [isOpen]);

  const loadCaseFiles = async () => {
    try {
      setLoading(true);
      const files = await apiClient.listCaseFiles();
      setCaseFiles(files);
    } catch (error) {
      console.error('Failed to load case files:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredCases = searchQuery
    ? caseFiles.filter(caseFile => 
        caseFile.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (caseFile.description && caseFile.description.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : caseFiles;

  const handleCreateNewCaseFile = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    
    try {
      const result = await apiClient.createCaseFile(createForm);
      
      // Add the document to the new case file
      const originalData = document.originalApiData;
      
      const addDocRequest: AddDocumentToCaseFileRequest = {
        document_id: document.id,
        citation: document.citation,
        title: document.title,
        year: parseInt(document.date.split('-')[0]) || new Date().getFullYear(),
        jurisdiction: document.jurisdiction,
        relevance_score_percent: document.relevanceScore || 0,
        key_holdings: originalData?.holdings || [],
        selected_chunks: originalData ? [{
          chunk_id: originalData.chunk_id,
          text: originalData.text,
          summary: originalData.summary,
          score: originalData.score,
          statutes: originalData.statutes || [],
          courts: originalData.courts || [],
          cases: originalData.cases || [],
          concepts: originalData.concepts || [],
          judges: originalData.judges || [],
          holdings: originalData.holdings || [],
          facts: originalData.facts || [],
          legal_tests: originalData.legal_tests || []
        }] : []
      };
      
      await apiClient.addDocumentToCaseFile(result.case_file_id, addDocRequest);
      
      // Call the callback if provided
      if (onAddToCase) {
        onAddToCase({ id: result.case_file_id, title: createForm.title }, document);
      }
      
      onClose();
      setCreateForm({ title: '', description: '', user_facts: '', party_represented: '' });
      setShowCreateForm(false);
      
    } catch (error) {
      console.error('Failed to create case file:', error);
      alert('Failed to create case file. Please try again.');
    } finally {
      setIsCreating(false);
    }
  };

  const handleAddToExistingCase = async () => {
    if (!selectedCaseId) return;
    
    setIsAdding(true);
    
    try {
      // Use original API data if available, otherwise fall back to transformed data
      const originalData = document.originalApiData;
      
      const addDocRequest: AddDocumentToCaseFileRequest = {
        document_id: document.id,
        citation: document.citation,
        title: document.title,
        year: parseInt(document.date.split('-')[0]) || new Date().getFullYear(),
        jurisdiction: document.jurisdiction,
        relevance_score_percent: document.relevanceScore || 0,
        key_holdings: originalData?.holdings || [],
        selected_chunks: originalData ? [{
          chunk_id: originalData.chunk_id,
          text: originalData.text,
          summary: originalData.summary,
          score: originalData.score,
          statutes: originalData.statutes || [],
          courts: originalData.courts || [],
          cases: originalData.cases || [],
          concepts: originalData.concepts || [],
          judges: originalData.judges || [],
          holdings: originalData.holdings || [],
          facts: originalData.facts || [],
          legal_tests: originalData.legal_tests || []
        }] : []
      };
      
      await apiClient.addDocumentToCaseFile(selectedCaseId, addDocRequest);
      
      const selectedCase = caseFiles.find(c => c.id === selectedCaseId);
      if (onAddToCase && selectedCase) {
        onAddToCase(selectedCase, document);
      }
      
      onClose();
    } catch (error) {
      console.error('Failed to add document to case file:', error);
      alert('Failed to add document to case file. Please try again.');
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Add to Case File"
    >
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          Select which case file to add "{document.title}" to:
        </p>

        {/* New Case File Button */}
        <button
          onClick={() => setShowCreateForm(true)}
          className="w-full flex items-center justify-center p-4 border-2 border-dashed border-primary rounded-lg text-primary hover:bg-primary/5 transition-colors"
        >
          <Plus className="h-5 w-5 mr-2" />
          Create New Case File
        </button>

        {/* Create New Case File Form */}
        {showCreateForm && (
          <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
            <h4 className="font-medium text-gray-900 mb-3">Create New Case File</h4>
            <form onSubmit={handleCreateNewCaseFile} className="space-y-3">
              <div>
                <input
                  type="text"
                  placeholder="Case file title *"
                  required
                  value={createForm.title}
                  onChange={(e) => setCreateForm({ ...createForm, title: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div>
                <textarea
                  placeholder="Description (optional)"
                  value={createForm.description}
                  onChange={(e) => setCreateForm({ ...createForm, description: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                  rows={2}
                />
              </div>
              <div>
                <textarea
                  placeholder="Case facts - Describe the key facts of your case *"
                  required
                  value={createForm.user_facts}
                  onChange={(e) => setCreateForm({ ...createForm, user_facts: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                  rows={3}
                />
              </div>
              <div>
                <input
                  type="text"
                  placeholder="Party you represent"
                  value={createForm.party_represented}
                  onChange={(e) => setCreateForm({ ...createForm, party_represented: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={isCreating}
                  className="flex-1 px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-700 disabled:opacity-50 text-sm"
                >
                  {isCreating ? 'Creating...' : 'Create & Add Document'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 text-sm"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Existing Case Files */}
        {!showCreateForm && (
          <>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search your case files..."
                className="block w-full pl-10 pr-3 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
              />
            </div>
            
            <div className="max-h-80 overflow-y-auto">
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin h-6 w-6 border-b-2 border-primary rounded-full"></div>
                  <span className="ml-2 text-gray-600">Loading case files...</span>
                </div>
              ) : filteredCases.length > 0 ? (
                <div className="space-y-2">
                  {filteredCases.map((caseFile) => (
                    <div 
                      key={caseFile.id}
                      onClick={() => setSelectedCaseId(caseFile.id)}
                      className={`flex items-start p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedCaseId === caseFile.id 
                          ? 'border-primary bg-blue-50' 
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                    >
                      <div className="mr-3 mt-1">
                        <div className={`p-2 rounded-full ${
                          selectedCaseId === caseFile.id ? 'bg-primary text-white' : 'bg-gray-100'
                        }`}>
                          {selectedCaseId === caseFile.id ? (
                            <Check className="h-4 w-4" />
                          ) : (
                            <FolderOpen className="h-4 w-4" />
                          )}
                        </div>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-sm font-medium text-gray-900">{caseFile.title}</h4>
                        {caseFile.description && (
                          <p className="text-xs text-gray-600 mt-1 line-clamp-2">{caseFile.description}</p>
                        )}
                        <div className="flex items-center space-x-2 mt-1">
                          <span className="text-xs text-gray-500">
                            {caseFile.document_count} docs • {caseFile.draft_count} drafts
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <FileText className="h-12 w-12 text-gray-300 mx-auto mb-3" />
                  <p className="text-gray-500">No case files found</p>
                  <p className="text-sm text-gray-400 mt-1">Create a new case file to get started</p>
                </div>
              )}
            </div>
          </>
        )}
        
        {!showCreateForm && (
          <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={handleAddToExistingCase}
              disabled={!selectedCaseId || isAdding}
              className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAdding ? 'Adding...' : 'Add to Case File'}
            </button>
          </div>
        )}
      </div>
    </Modal>
  );
};
export default SelectCaseModal;