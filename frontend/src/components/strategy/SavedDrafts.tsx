import React, { useState, useEffect } from 'react';
import { FileText, Clock, BarChart3, Trash2, Eye } from 'lucide-react';
import { apiClient } from '../../services/api';
import { ArgumentDraftListItem, SavedArgumentDraft } from '../../types/api';

interface SavedDraftsProps {
  caseFileId: number | null;
  onViewDraft: (draft: SavedArgumentDraft) => void;
}

const SavedDrafts: React.FC<SavedDraftsProps> = ({ caseFileId, onViewDraft }) => {
  const [drafts, setDrafts] = useState<ArgumentDraftListItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (caseFileId) {
      loadDrafts();
    } else {
      setDrafts([]);
    }
  }, [caseFileId]);

  const loadDrafts = async () => {
    if (!caseFileId) return;
    
    try {
      setLoading(true);
      const draftList = await apiClient.listDraftsForCaseFile(caseFileId);
      setDrafts(draftList);
    } catch (error) {
      console.error('Failed to load drafts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDraft = async (draftId: number) => {
    try {
      const draft = await apiClient.getDraft(draftId);
      onViewDraft(draft);
    } catch (error) {
      console.error('Failed to load draft:', error);
      alert('Failed to load draft. Please try again.');
    }
  };

  const handleDeleteDraft = async (draftId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this draft?')) {
      try {
        await apiClient.deleteDraft(draftId);
        await loadDrafts();
      } catch (error) {
        console.error('Failed to delete draft:', error);
        alert('Failed to delete draft. Please try again.');
      }
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const formatScore = (score?: number) => {
    if (score === undefined || score === null) return 'N/A';
    return `${Math.round(score * 100)}%`;
  };

  if (!caseFileId) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-center text-gray-500">
          <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
          <p>Select a case file to view saved drafts</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center">
          <div className="animate-spin h-8 w-8 border-b-2 border-primary rounded-full"></div>
          <span className="ml-2">Loading drafts...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <FileText className="h-5 w-5 mr-2" />
          Saved Drafts
        </h3>
      </div>

      <div className="max-h-64 overflow-y-auto">
        {drafts.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <FileText className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p>No drafts saved yet</p>
            <p className="text-sm">Generate an argument draft to see it here</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {drafts.map((draft) => (
              <div
                key={draft.id}
                className="p-4 hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 mb-1">{draft.title}</h4>
                    <div className="flex items-center text-xs text-gray-500 space-x-4 mb-2">
                      <span className="flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {formatDate(draft.created_at)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-4 text-xs">
                      <span className="flex items-center">
                        <BarChart3 className="h-3 w-3 mr-1 text-green-500" />
                        Strength: {formatScore(draft.argument_strength)}
                      </span>
                      <span className="flex items-center">
                        <BarChart3 className="h-3 w-3 mr-1 text-blue-500" />
                        Coverage: {formatScore(draft.precedent_coverage)}
                      </span>
                      <span className="flex items-center">
                        <BarChart3 className="h-3 w-3 mr-1 text-purple-500" />
                        Logic: {formatScore(draft.logical_coherence)}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={() => handleViewDraft(draft.id)}
                      className="p-1 text-gray-400 hover:text-primary transition-colors"
                      title="View draft"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    <button
                      onClick={(e) => handleDeleteDraft(draft.id, e)}
                      className="p-1 text-gray-400 hover:text-red-600 transition-colors"
                      title="Delete draft"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default SavedDrafts;
