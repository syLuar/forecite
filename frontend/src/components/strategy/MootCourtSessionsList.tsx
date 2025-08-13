import React, { useState, useEffect } from 'react';
import { Calendar, Eye, Trash2, Swords, User, ArrowLeft } from 'lucide-react';
import { apiClient } from '../../services/api';
import ConfirmModal from '../shared/ConfirmModal';

interface MootCourtSessionsListProps {
  caseFileId: number;
  caseFileTitle: string;
  onBack: () => void;
  onViewSession: (sessionId: number) => void;
}

interface MootCourtSession {
  id: number;
  title: string;
  created_at: string;
  draft_title?: string;
  counterargument_count: number;
  counterargument_strength?: number;
  research_comprehensiveness?: number;
}

const MootCourtSessionsList: React.FC<MootCourtSessionsListProps> = ({
  caseFileId,
  caseFileTitle,
  onBack,
  onViewSession,
}) => {
  const [sessions, setSessions] = useState<MootCourtSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteSessionId, setDeleteSessionId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);

  useEffect(() => {
    loadSessions();
  }, [caseFileId]);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const sessionsList = await apiClient.listMootCourtSessionsForCaseFile(caseFileId);
      setSessions(sessionsList);
    } catch (error) {
      console.error('Failed to load moot court sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRequestDeleteSession = (sessionId: number) => {
    setDeleteSessionId(sessionId);
    setShowConfirmModal(true);
  };

  const handleConfirmDelete = async () => {
    if (deleteSessionId == null) return;
    setDeleting(true);
    try {
      await apiClient.deleteMootCourtSession(deleteSessionId);
      setSessions(sessions.filter(session => session.id !== deleteSessionId));
      setDeleteSessionId(null);
      setShowConfirmModal(false);
    } catch (error) {
      console.error('Failed to delete moot court session:', error);
      alert('Failed to delete session. Please try again.');
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center mb-8">
            <button
              onClick={onBack}
              className="flex items-center text-primary hover:text-primary-700 font-medium"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back
            </button>
          </div>
          
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-white rounded-lg shadow-md p-6 animate-pulse">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="bg-gray-200 h-6 rounded w-3/4 mb-3"></div>
                    <div className="bg-gray-200 h-4 rounded w-1/2 mb-2"></div>
                    <div className="bg-gray-200 h-4 rounded w-1/3"></div>
                  </div>
                  <div className="bg-gray-200 h-8 w-16 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBack}
            className="flex items-center text-primary hover:text-primary-700 font-medium"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            Back to Case
          </button>
          <div className="text-right">
            <h1 className="text-2xl font-bold text-gray-900">Moot Court Sessions</h1>
            <p className="text-gray-600">{caseFileTitle}</p>
          </div>
        </div>

        {/* Sessions List */}
        {sessions.length === 0 ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <Swords className="h-16 w-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Moot Court Sessions</h3>
            <p className="text-gray-600 mb-6">
              You haven't saved any moot court practice sessions yet.
            </p>
            <button
              onClick={() => onViewSession(-1)} // -1 indicates new session
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-700 font-medium transition-colors"
            >
              Start New Session
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {sessions.map((session) => (
              <div key={session.id} className="bg-white rounded-lg shadow-md border border-gray-200 hover:shadow-lg transition-shadow">
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-3">
                        <h3 className="text-lg font-semibold text-gray-900 mr-4">
                          {session.title}
                        </h3>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => onViewSession(session.id)}
                            className="p-2 text-primary hover:text-primary-700 hover:bg-primary-50 rounded-lg transition-colors"
                            title="View session"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => handleRequestDeleteSession(session.id)}
                            disabled={deleting}
                            className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                            title="Delete session"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600">
                        <div className="flex items-center">
                          <Calendar className="h-4 w-4 mr-2" />
                          {formatDate(session.created_at)}
                        </div>
                        
                        {session.draft_title && (
                          <div className="flex items-center">
                            <User className="h-4 w-4 mr-2" />
                            Based on: {session.draft_title}
                          </div>
                        )}
                        
                        <div className="flex items-center">
                          <Swords className="h-4 w-4 mr-2" />
                          {session.counterargument_count} counterarguments
                        </div>
                      </div>

                      {/* Quality Metrics */}
                      {(session.counterargument_strength !== null || session.research_comprehensiveness !== null) && (
                        <div className="mt-4 pt-3 border-t border-gray-100">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                            {session.counterargument_strength !== null && (
                              <div>
                                <span className="text-gray-600">Argument Strength:</span>
                                <div className="flex items-center mt-1">
                                  <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                    <div
                                      className="bg-primary h-2 rounded-full"
                                      style={{ width: `${(session.counterargument_strength || 0) * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-gray-700 font-medium text-xs">
                                    {Math.round((session.counterargument_strength || 0) * 100)}%
                                  </span>
                                </div>
                              </div>
                            )}
                            
                            {session.research_comprehensiveness !== null && (
                              <div>
                                <span className="text-gray-600">Research Quality:</span>
                                <div className="flex items-center mt-1">
                                  <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                    <div
                                      className="bg-green-500 h-2 rounded-full"
                                      style={{ width: `${(session.research_comprehensiveness || 0) * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-gray-700 font-medium text-xs">
                                    {Math.round((session.research_comprehensiveness || 0) * 100)}%
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Add New Session Button */}
            <div className="pt-4">
              <button
                onClick={() => onViewSession(-1)} // -1 indicates new session
                className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-primary hover:text-primary hover:bg-primary-50 transition-colors"
              >
                <Swords className="h-6 w-6 mx-auto mb-2" />
                Start New Moot Court Session
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Confirm Delete Modal */}
      <ConfirmModal
        isOpen={showConfirmModal}
        onClose={() => { if (!deleting) { setShowConfirmModal(false); setDeleteSessionId(null); } }}
        onConfirm={handleConfirmDelete}
        title="Delete Session"
        message="Are you sure you want to delete this moot court session? This action cannot be undone."
        confirmText="Delete Session"
        isDestructive
        isLoading={deleting}
      />
    </div>
  );
};

export default MootCourtSessionsList;