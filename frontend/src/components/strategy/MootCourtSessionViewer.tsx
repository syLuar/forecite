import React, { useState, useEffect } from 'react';
import { ArrowLeft, Calendar, User, Swords, X } from 'lucide-react';
import { apiClient } from '../../services/api';

interface MootCourtSessionViewerProps {
  sessionId: number;
  caseFileTitle: string;
  onBack: () => void;
}

interface CounterArgument {
  title: string;
  argument: string;
  supporting_authority: string;
  factual_basis: string;
  strength_assessment?: number;
}

interface CounterArgumentRebuttal {
  title: string;
  content: string;
  authority: string;
}

interface SavedSession {
  id: number;
  title: string;
  created_at: string;
  counterarguments: CounterArgument[];
  rebuttals: CounterArgumentRebuttal[][];
  source_arguments?: any[];
  counterargument_strength?: number;
  research_comprehensiveness?: number;
  rebuttal_quality?: number;
}

const MootCourtSessionViewer: React.FC<MootCourtSessionViewerProps> = ({
  sessionId,
  caseFileTitle,
  onBack,
}) => {
  const [session, setSession] = useState<SavedSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [showRebuttalModal, setShowRebuttalModal] = useState(false);
  const [selectedOpponentArg, setSelectedOpponentArg] = useState<number | null>(null);

  useEffect(() => {
    loadSession();
  }, [sessionId]);

  const loadSession = async () => {
    try {
      setLoading(true);
      const sessionData = await apiClient.getMootCourtSession(sessionId);
      setSession(sessionData);
    } catch (error) {
      console.error('Failed to load moot court session:', error);
      alert('Failed to load session. Please try again.');
      onBack();
    } finally {
      setLoading(false);
    }
  };

  const handleRebuttalClick = (argumentIndex: number) => {
    setSelectedOpponentArg(argumentIndex);
    setShowRebuttalModal(true);
  };

  const closeRebuttalModal = () => {
    setShowRebuttalModal(false);
    setSelectedOpponentArg(null);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center mb-8">
            <button
              onClick={onBack}
              className="flex items-center text-primary hover:text-primary-700 font-medium"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back
            </button>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white rounded-lg shadow-md p-6 animate-pulse">
              <div className="bg-gray-200 h-6 rounded w-1/2 mb-4"></div>
              <div className="space-y-4">
                <div className="bg-gray-200 h-20 rounded"></div>
                <div className="bg-gray-200 h-20 rounded"></div>
                <div className="bg-gray-200 h-20 rounded"></div>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-md p-6 animate-pulse">
              <div className="bg-gray-200 h-6 rounded w-1/2 mb-4"></div>
              <div className="space-y-4">
                <div className="bg-gray-200 h-20 rounded"></div>
                <div className="bg-gray-200 h-20 rounded"></div>
                <div className="bg-gray-200 h-20 rounded"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!session) {
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
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <p className="text-gray-600">Session not found.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={onBack}
            className="flex items-center text-primary hover:text-primary-700 font-medium"
          >
            <ArrowLeft className="h-5 w-5 mr-2" />
            Back to Sessions
          </button>
          <div className="text-right">
            <h1 className="text-2xl font-bold text-gray-900">{session.title}</h1>
            <p className="text-gray-600">{caseFileTitle}</p>
          </div>
        </div>

        {/* Session Info */}
        <div className="bg-white rounded-lg shadow-md border border-gray-200 mb-8">
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-600 mb-4">
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-2" />
                {formatDate(session.created_at)}
              </div>
              <div className="flex items-center">
                <Swords className="h-4 w-4 mr-2" />
                {session.counterarguments.length} counterarguments
              </div>
              <div className="flex items-center">
                <User className="h-4 w-4 mr-2" />
                {session.source_arguments?.length || 0} source arguments
              </div>
            </div>

            {/* Quality Metrics */}
            {(session.counterargument_strength !== null || session.research_comprehensiveness !== null || session.rebuttal_quality !== null) && (
              <div className="pt-4 border-t border-gray-100">
                <h4 className="text-sm font-medium text-gray-900 mb-3">Session Quality Metrics</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  {session.counterargument_strength !== null && (
                    <div>
                      <span className="text-gray-600">Counterargument Strength:</span>
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

                  {session.rebuttal_quality !== null && (
                    <div>
                      <span className="text-gray-600">Rebuttal Quality:</span>
                      <div className="flex items-center mt-1">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${(session.rebuttal_quality || 0) * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-gray-700 font-medium text-xs">
                          {Math.round((session.rebuttal_quality || 0) * 100)}%
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Original Arguments */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Original Arguments</h2>
            </div>
            <div className="p-6">
              {session.source_arguments && session.source_arguments.length > 0 ? (
                <div className="space-y-6">
                  {session.source_arguments.map((arg: any, index: number) => (
                    <div key={index} className="border-l-4 border-primary pl-4 bg-gray-50 p-4 rounded-r-lg">
                      <h4 className="font-semibold text-gray-900 mb-2">Argument {index + 1}</h4>
                      <p className="text-gray-700 mb-3 leading-relaxed">{arg.argument}</p>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-900">Supporting Authority:</span>
                          <p className="text-gray-600 mt-1">{arg.supporting_authority}</p>
                        </div>
                        <div>
                          <span className="font-medium text-gray-900">Factual Basis:</span>
                          <p className="text-gray-600 mt-1">{arg.factual_basis}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <p>No source arguments recorded for this session.</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Counterarguments */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Counterarguments</h2>
            </div>
            <div className="p-6">
              <div className="space-y-6">
                {session.counterarguments.map((counterArg, index) => (
                  <div key={index} className="border-l-4 border-red-500 pl-4 bg-red-50 p-4 rounded-r-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900 mb-2">
                          Counter-Argument {index + 1}: {counterArg.title}
                        </h4>
                        <p className="text-gray-700 mb-3 leading-relaxed">
                          {counterArg.argument}
                        </p>
                        <div className="space-y-2 text-sm">
                          <div>
                            <span className="font-medium text-gray-900">Supporting Authority:</span>
                            <p className="text-gray-600 mt-1">{counterArg.supporting_authority}</p>
                          </div>
                          <div>
                            <span className="font-medium text-gray-900">Factual Basis:</span>
                            <p className="text-gray-600 mt-1">{counterArg.factual_basis}</p>
                          </div>
                          {counterArg.strength_assessment && (
                            <div>
                              <span className="font-medium text-gray-900">Strength:</span>
                              <div className="flex items-center mt-1">
                                <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2 max-w-24">
                                  <div
                                    className="bg-red-500 h-2 rounded-full"
                                    style={{ width: `${counterArg.strength_assessment * 100}%` }}
                                  ></div>
                                </div>
                                <span className="text-gray-700 font-medium text-xs">
                                  {Math.round(counterArg.strength_assessment * 100)}%
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => handleRebuttalClick(index)}
                        className="ml-4 w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors shadow-md hover:shadow-lg"
                        title="View rebuttals"
                      >
                        <Swords className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Rebuttal Modal */}
      {showRebuttalModal && selectedOpponentArg !== null && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">
                  Rebuttals to Counter-Argument {selectedOpponentArg + 1}
                </h2>
                <button
                  onClick={closeRebuttalModal}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-6">
                {selectedOpponentArg !== null && session.rebuttals[selectedOpponentArg]?.map((rebuttal: CounterArgumentRebuttal, index: number) => (
                  <div key={index} className="border-l-4 border-primary pl-4 bg-blue-50 p-4 rounded-r-lg">
                    <h3 className="font-semibold text-gray-900 mb-2">
                      Rebuttal {index + 1}: {rebuttal.title}
                    </h3>
                    <p className="text-gray-700 mb-3 leading-relaxed">
                      {rebuttal.content}
                    </p>
                    <div className="text-sm">
                      <span className="font-medium text-gray-900">Supporting Authority:</span>
                      <p className="text-gray-600 mt-1">{rebuttal.authority}</p>
                    </div>
                  </div>
                )) || (
                  <div className="text-center py-8 text-gray-500">
                    <p>No rebuttals available for this counterargument.</p>
                  </div>
                )}
              </div>
              
              <div className="mt-6 pt-4 border-t border-gray-200 flex justify-end">
                <button
                  onClick={closeRebuttalModal}
                  className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MootCourtSessionViewer;