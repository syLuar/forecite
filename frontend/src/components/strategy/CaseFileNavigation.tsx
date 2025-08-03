import React, { useState, useEffect } from 'react';
import { FolderOpen, Plus, FileText, Trash2, Edit3 } from 'lucide-react';
import { apiClient } from '../../services/api';
import { CaseFileListItem, CreateCaseFileRequest } from '../../types/api';

interface CaseFileNavigationProps {
  onSelectCaseFile: (caseFileId: number) => void;
  selectedCaseFileId: number | null;
}

const CaseFileNavigation: React.FC<CaseFileNavigationProps> = ({
  onSelectCaseFile,
  selectedCaseFileId
}) => {
  const [caseFiles, setCaseFiles] = useState<CaseFileListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createForm, setCreateForm] = useState<CreateCaseFileRequest>({
    title: '',
    description: '',
    user_facts: '',
    legal_question: ''
  });

  useEffect(() => {
    loadCaseFiles();
  }, []);

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

  const handleCreateCaseFile = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const result = await apiClient.createCaseFile(createForm);
      await loadCaseFiles();
      onSelectCaseFile(result.case_file_id);
      setCreateForm({ title: '', description: '', user_facts: '', legal_question: '' });
      setShowCreateForm(false);
    } catch (error) {
      console.error('Failed to create case file:', error);
      alert('Failed to create case file. Please try again.');
    }
  };

  const handleDeleteCaseFile = async (caseFileId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this case file? This action cannot be undone.')) {
      try {
        await apiClient.deleteCaseFile(caseFileId);
        await loadCaseFiles();
        if (selectedCaseFileId === caseFileId) {
          onSelectCaseFile(0); // Deselect if deleted case was selected
        }
      } catch (error) {
        console.error('Failed to delete case file:', error);
        alert('Failed to delete case file. Please try again.');
      }
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin h-8 w-8 border-b-2 border-primary rounded-full"></div>
          <span className="ml-2">Loading case files...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center">
            <FolderOpen className="h-5 w-5 mr-2" />
            Case Files
          </h2>
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center px-3 py-1 text-sm bg-primary text-white rounded-md hover:bg-primary-700"
          >
            <Plus className="h-4 w-4 mr-1" />
            New Case
          </button>
        </div>
      </div>

      {showCreateForm && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <form onSubmit={handleCreateCaseFile} className="space-y-3">
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
            <div className="flex gap-2">
              <button
                type="submit"
                className="px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-700 text-sm"
              >
                Create
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

      <div className="max-h-64 overflow-y-auto">
        {caseFiles.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p>No case files yet</p>
            <p className="text-sm">Create your first case file to get started</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {caseFiles.map((caseFile) => (
              <div
                key={caseFile.id}
                onClick={() => onSelectCaseFile(caseFile.id)}
                className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                  selectedCaseFileId === caseFile.id ? 'bg-primary/10 border-r-4 border-primary' : ''
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900 mb-1">{caseFile.title}</h3>
                    {caseFile.description && (
                      <p className="text-sm text-gray-600 mb-2 line-clamp-2">{caseFile.description}</p>
                    )}
                    <div className="flex items-center text-xs text-gray-500 space-x-4">
                      <span>{formatDate(caseFile.created_at)}</span>
                      <span>{caseFile.document_count} docs</span>
                      <span>{caseFile.draft_count} drafts</span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteCaseFile(caseFile.id, e)}
                    className="ml-2 p-1 text-gray-400 hover:text-red-600 transition-colors"
                    title="Delete case file"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default CaseFileNavigation;
