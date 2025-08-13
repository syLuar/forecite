import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, ChevronDown, FileText, Swords, X, RefreshCw, Save } from 'lucide-react';
import { SavedArgumentDraft } from '../../types/api';
import { apiClient } from '../../services/api';

interface MootCourtProps {
  caseFileId: number;
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

const MootCourt: React.FC<MootCourtProps> = ({ caseFileId, caseFileTitle, onBack }) => {
  const [drafts, setDrafts] = useState<SavedArgumentDraft[]>([]);
  const [selectedDraft, setSelectedDraft] = useState<SavedArgumentDraft | null>(null);
  const [selectedDraftId, setSelectedDraftId] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [loadingDraft, setLoadingDraft] = useState(false);
  const [generatingCounterArgs, setGeneratingCounterArgs] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [showRebuttalModal, setShowRebuttalModal] = useState(false);
  const [selectedOpponentArg, setSelectedOpponentArg] = useState<number | null>(null);
  const [counterArguments, setCounterArguments] = useState<CounterArgument[]>([]);
  const [rebuttals, setRebuttals] = useState<CounterArgumentRebuttal[][]>([]);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [sessionTitle, setSessionTitle] = useState('');
  const [saving, setSaving] = useState(false);
  // Toast notifications state
  const [toasts, setToasts] = useState<{ id: number; type: 'success' | 'error' | 'info'; message: string }[]>([]);
  // Ref for draft selector dropdown
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  const showToast = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, type, message }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter(t => t.id !== id));
    }, 5000);
  };

  useEffect(() => {
    const loadDrafts = async () => {
      try {
        setLoading(true);
        const draftsList = await apiClient.listDraftsForCaseFile(caseFileId);
        setDrafts(draftsList);
      } catch (error) {
        console.error('Failed to load drafts:', error);
      } finally {
        setLoading(false);
      }
    };

    loadDrafts();
  }, [caseFileId]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleDraftSelection = async (draftId: string) => {
    if (!draftId) {
      setSelectedDraft(null);
      setSelectedDraftId('');
      setShowDropdown(false);
      return;
    }

    setLoadingDraft(true);
    try {
      const draft = await apiClient.getDraft(parseInt(draftId));
      setSelectedDraft(draft);
      setSelectedDraftId(draftId);
      setShowDropdown(false);
    } catch (error) {
      console.error('Failed to load draft:', error);
      showToast('error', 'Failed to load draft. Please try again.');
    } finally {
      setLoadingDraft(false);
    }
  };

  const getSelectedDraftTitle = () => {
    if (!selectedDraftId) return 'No draft selected';
    const draft = drafts.find(d => d.id.toString() === selectedDraftId);
    return draft ? draft.title : 'Unknown draft';
  };

  const generateCounterArguments = async () => {
    if (!selectedDraft) {
      showToast('info', 'Select a draft before generating counterarguments.');
      return;
    }

    setGeneratingCounterArgs(true);
    try {
      const response = await apiClient.generateCounterArguments(caseFileId, selectedDraft.id);
      setCounterArguments(response.counterarguments);
      setRebuttals(response.rebuttals);
      showToast('success', 'Counterarguments generated successfully.');
    } catch (error) {
      console.error('Failed to generate counterarguments:', error);
      showToast('error', 'Failed to generate counterarguments. Please try again.');
    } finally {
      setGeneratingCounterArgs(false);
    }
  };

  const hasGeneratedCounterArgs = counterArguments.length > 0;

  const handleRebuttalClick = (argumentIndex: number) => {
    setSelectedOpponentArg(argumentIndex);
    setShowRebuttalModal(true);
  };

  const closeRebuttalModal = () => {
    setShowRebuttalModal(false);
    setSelectedOpponentArg(null);
  };

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
            Back to Case
          </button>
          <div className="flex items-center space-x-4">
            {hasGeneratedCounterArgs && (
              <button
                onClick={() => setShowSaveModal(true)}
                className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 shadow-md hover:shadow-lg transition-colors"
              >
                <Save className="h-4 w-4 mr-2" />
                Save Session
              </button>
            )}
            <div className="text-right">
              <h1 className="text-2xl font-bold text-gray-900">Moot Court</h1>
              <p className="text-gray-600">{caseFileTitle}</p>
            </div>
          </div>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - My Arguments */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200 relative">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">My Arguments</h2>
                
                {/* Compact Draft Selector */}
                {loading ? (
                  <div className="animate-pulse bg-gray-200 h-8 w-8 rounded-lg"></div>
                ) : (
                  <div className="relative" ref={dropdownRef}>
                    <button
                      onClick={() => setShowDropdown(!showDropdown)}
                      className="flex items-center px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                      disabled={loadingDraft}
                    >
                      <FileText className="h-4 w-4 mr-2 text-gray-500" />
                      <span className="text-gray-700 max-w-32 truncate">
                        {getSelectedDraftTitle()}
                      </span>
                      <ChevronDown className={`h-4 w-4 ml-2 text-gray-400 transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
                    </button>
                    
                    {showDropdown && (
                      <div className="absolute right-0 top-full mt-1 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                        <div className="py-1">
                          <button
                            onClick={() => handleDraftSelection('')}
                            className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 focus:bg-gray-50"
                          >
                            No draft selected
                          </button>
                          {drafts.map((draft) => (
                            <button
                              key={draft.id}
                              onClick={() => handleDraftSelection(draft.id.toString())}
                              className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-50 focus:bg-gray-50 ${
                                selectedDraftId === draft.id.toString() ? 'bg-primary text-white hover:bg-primary-600' : 'text-gray-700'
                              }`}
                            >
                              <div className="truncate font-medium">{draft.title}</div>
                              <div className="text-xs opacity-75 truncate">
                                {new Date(draft.created_at).toLocaleDateString()}
                              </div>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Key Arguments Display */}
            <div className="p-6">
              {loadingDraft ? (
                <div className="space-y-4">
                  <div className="animate-pulse">
                    <div className="bg-gray-200 h-4 rounded w-3/4 mb-2"></div>
                    <div className="bg-gray-200 h-16 rounded"></div>
                  </div>
                  <div className="animate-pulse">
                    <div className="bg-gray-200 h-4 rounded w-3/4 mb-2"></div>
                    <div className="bg-gray-200 h-16 rounded"></div>
                  </div>
                  <div className="animate-pulse">
                    <div className="bg-gray-200 h-4 rounded w-3/4 mb-2"></div>
                    <div className="bg-gray-200 h-16 rounded"></div>
                  </div>
                </div>
              ) : selectedDraft && selectedDraft.strategy?.key_arguments ? (
                <div className="space-y-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Key Arguments</h3>
                  {selectedDraft.strategy.key_arguments.map((arg: any, index: number) => (
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
              ) : selectedDraft ? (
                <div className="text-center py-8 text-gray-500">
                  <p>This draft doesn't have structured key arguments.</p>
                  <p className="text-sm mt-2">Try selecting a different draft.</p>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <p className="text-lg mb-2">No draft selected</p>
                  <p className="text-sm">Choose a saved draft from the selector above to view your key arguments.</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Opponent Arguments */}
          <div className="bg-white rounded-lg shadow-md border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Opponent Arguments</h2>
                
                {/* Generate Button */}
                <button
                  onClick={generateCounterArguments}
                  disabled={!selectedDraft || generatingCounterArgs}
                  className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    !selectedDraft || generatingCounterArgs
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-primary text-white hover:bg-primary-700 shadow-md hover:shadow-lg'
                  }`}
                >
                  {generatingCounterArgs ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Swords className="h-4 w-4 mr-2" />
                  )}
                  {generatingCounterArgs ? 'Generating...' : 'Generate'}
                </button>
              </div>
            </div>
            <div className="p-6">
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Potential Counter-Arguments</h3>
                
                {!hasGeneratedCounterArgs && !generatingCounterArgs ? (
                  <div className="text-center py-12 text-gray-500">
                    <Swords className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p className="text-lg mb-2">No counterarguments generated yet</p>
                    <p className="text-sm">Select a draft and click "Generate" to create counterarguments using AI.</p>
                  </div>
                ) : generatingCounterArgs ? (
                  <div className="space-y-4">
                    <div className="animate-pulse">
                      <div className="bg-gray-200 h-6 rounded w-1/2 mb-3"></div>
                      <div className="bg-gray-200 h-20 rounded mb-2"></div>
                      <div className="bg-gray-200 h-4 rounded w-3/4 mb-1"></div>
                      <div className="bg-gray-200 h-4 rounded w-2/3"></div>
                    </div>
                    <div className="animate-pulse">
                      <div className="bg-gray-200 h-6 rounded w-1/2 mb-3"></div>
                      <div className="bg-gray-200 h-20 rounded mb-2"></div>
                      <div className="bg-gray-200 h-4 rounded w-3/4 mb-1"></div>
                      <div className="bg-gray-200 h-4 rounded w-2/3"></div>
                    </div>
                    <div className="animate-pulse">
                      <div className="bg-gray-200 h-6 rounded w-1/2 mb-3"></div>
                      <div className="bg-gray-200 h-20 rounded mb-2"></div>
                      <div className="bg-gray-200 h-4 rounded w-3/4 mb-1"></div>
                      <div className="bg-gray-200 h-4 rounded w-2/3"></div>
                    </div>
                  </div>
                ) : (
                  counterArguments.map((counterArg, index) => (
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
                  ))
                )}
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
                {selectedOpponentArg !== null && rebuttals[selectedOpponentArg]?.map((rebuttal: CounterArgumentRebuttal, index: number) => (
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

      {/* Save Modal (New) */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Save Session Title</h2>
                <button
                  onClick={() => setShowSaveModal(false)}
                  className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
            </div>
            
            <div className="p-6">
              <div className="space-y-4">
                <div>
                  <label htmlFor="sessionTitle" className="block text-sm font-medium text-gray-700">
                    Session Title
                  </label>
                  <input
                    id="sessionTitle"
                    value={sessionTitle}
                    onChange={(e) => setSessionTitle(e.target.value)}
                    className="mt-1 block w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                    placeholder="Enter a title for your session"
                  />
                </div>
                
                <div className="flex justify-end space-x-2">
                  <button
                    onClick={() => setShowSaveModal(false)}
                    className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={async () => {
                      if (!sessionTitle.trim()) {
                        showToast('info', 'Please enter a session title.');
                        return;
                      }
                      
                      setSaving(true);
                      try {
                        const saveRequest = {
                          case_file_id: caseFileId,
                          draft_id: selectedDraft?.id,
                          title: sessionTitle.trim(),
                          counterarguments: counterArguments,
                          rebuttals: rebuttals,
                          source_arguments: selectedDraft?.strategy?.key_arguments || [],
                          counterargument_strength: null,
                          research_comprehensiveness: null,
                          rebuttal_quality: null,
                          execution_time: null,
                        };
                        
                        const response = await apiClient.saveMootCourtSession(saveRequest);
                        showToast('success', `Session saved (ID: ${response.session_id}).`);
                        setShowSaveModal(false);
                        setSessionTitle('');
                      } catch (error) {
                        console.error('Failed to save moot court session:', error);
                        showToast('error', 'Failed to save session. Please try again.');
                      } finally {
                        setSaving(false);
                      }
                    }}
                    disabled={saving || !sessionTitle.trim()}
                    className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                      saving || !sessionTitle.trim()
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-primary text-white hover:bg-primary-700'
                    }`}
                  >
                    {saving ? (
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Save className="h-4 w-4 mr-2" />
                    )}
                    {saving ? 'Saving...' : 'Save Session'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Toast Notifications */}
      <div className="fixed bottom-4 right-4 z-50 space-y-3 w-80">
        {toasts.map(t => (
          <div
            key={t.id}
            className={`flex items-start p-4 rounded-lg shadow-lg text-sm animate-slide-in-left relative overflow-hidden ${
              t.type === 'success' ? 'bg-green-50 border border-green-200 text-green-800' :
              t.type === 'error' ? 'bg-red-50 border border-red-200 text-red-800' :
              'bg-blue-50 border border-blue-200 text-blue-800'
            }`}
          >
            <div className="flex-1 pr-6">{t.message}</div>
            <button
              onClick={() => setToasts(prev => prev.filter(x => x.id !== t.id))}
              className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
            <span
              className={`absolute bottom-0 left-0 h-1 animate-toast-bar ${
                t.type === 'success' ? 'bg-green-400' : t.type === 'error' ? 'bg-red-400' : 'bg-blue-400'
              }`}
              style={{ animationDuration: '5s', width: '100%' }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default MootCourt;
