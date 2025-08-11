import React, { useState, useEffect, useRef } from 'react';
import { ArrowLeft, ChevronDown, FileText, Swords, X } from 'lucide-react';
import { SavedArgumentDraft } from '../../types/api';
import { apiClient } from '../../services/api';

interface MootCourtProps {
  caseFileId: number;
  caseFileTitle: string;
  onBack: () => void;
}

const MootCourt: React.FC<MootCourtProps> = ({ caseFileId, caseFileTitle, onBack }) => {
  const [drafts, setDrafts] = useState<SavedArgumentDraft[]>([]);
  const [selectedDraft, setSelectedDraft] = useState<SavedArgumentDraft | null>(null);
  const [selectedDraftId, setSelectedDraftId] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [loadingDraft, setLoadingDraft] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [showRebuttalModal, setShowRebuttalModal] = useState(false);
  const [selectedOpponentArg, setSelectedOpponentArg] = useState<number | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

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
      alert('Failed to load draft. Please try again.');
    } finally {
      setLoadingDraft(false);
    }
  };

  const getSelectedDraftTitle = () => {
    if (!selectedDraftId) return 'No draft selected';
    const draft = drafts.find(d => d.id.toString() === selectedDraftId);
    return draft ? draft.title : 'Unknown draft';
  };

  // Hardcoded rebuttals for each opponent argument
  const rebuttals = {
    0: [ // Counter-Argument 1 rebuttals
      {
        title: "Precedent Supports Broad Interpretation",
        content: "Recent Supreme Court decisions in Williams v. State (2023) explicitly endorse a broader reading of similar statutory language, establishing that narrow interpretations fail to serve the statute's remedial purpose.",
        authority: "Williams v. State (2023), Thompson v. Federal Agency (2022)"
      },
      {
        title: "Legislative History Confirms Intent",
        content: "Committee reports and floor debates clearly demonstrate legislative intent to create a comprehensive framework, not the limited application suggested by opposing counsel.",
        authority: "House Committee Report 118-45, Senate Floor Debates March 2021"
      },
      {
        title: "Practical Application Supports Our Position",
        content: "Federal agencies have consistently applied this statute broadly for over a decade, creating established administrative practice that courts typically defer to under Chevron doctrine.",
        authority: "Chevron U.S.A. v. NRDC (1984), Agency Implementation Guidelines 2012-2024"
      }
    ],
    1: [ // Counter-Argument 2 rebuttals
      {
        title: "Evidence Meets All Daubert Requirements",
        content: "Our expert testimony is based on peer-reviewed methodology published in leading scientific journals and has been accepted by courts in similar cases nationwide.",
        authority: "Journal of Scientific Evidence (2023), Federal Evidence Review Vol. 45"
      },
      {
        title: "Multiple Independent Corroborating Sources",
        content: "The evidence is supported by three independent studies using different methodologies, all reaching consistent conclusions that strengthen reliability.",
        authority: "Anderson Study (2023), Miller Research Institute (2024), Federal Lab Report 2024"
      },
      {
        title: "Opposing Expert Lacks Relevant Experience",
        content: "Defense expert has never published in this field and lacks the specialized knowledge required to challenge our evidence under Federal Rule 702.",
        authority: "Federal Rule of Evidence 702, Kumho Tire Co. v. Carmichael (1999)"
      }
    ],
    2: [ // Counter-Argument 3 rebuttals
      {
        title: "Discovery Rule Applies",
        content: "Plaintiff could not reasonably have discovered the cause of action earlier due to defendant's active concealment of material facts, triggering the discovery rule exception.",
        authority: "State Civil Code Section 338(d), Martinez v. Corporate Defendant (2022)"
      },
      {
        title: "Equitable Estoppel Prevents Limitations Defense",
        content: "Defendant's affirmative misrepresentations and promises to resolve the matter prevented plaintiff from filing suit, creating equitable estoppel.",
        authority: "Batson v. Manufacturing Co. (2021), Equitable Estoppel Doctrine"
      },
      {
        title: "Continuing Violation Theory",
        content: "The complained-of conduct constitutes a continuing violation that was ongoing within the limitations period, making the entire claim timely filed.",
        authority: "Garcia v. Employer (2023), Continuing Violation Doctrine Applications"
      }
    ]
  };

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
          <div className="text-right">
            <h1 className="text-2xl font-bold text-gray-900">Moot Court</h1>
            <p className="text-gray-600">{caseFileTitle}</p>
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
                  {selectedDraft.strategy.key_arguments.slice(0, 3).map((arg: any, index: number) => (
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
              <h2 className="text-xl font-semibold text-gray-900">Opponent Arguments</h2>
            </div>
            <div className="p-6">
              <div className="space-y-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Potential Counter-Arguments</h3>
                
                {/* Opponent Argument 1 */}
                <div className="border-l-4 border-red-500 pl-4 bg-red-50 p-4 rounded-r-lg">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-2">Counter-Argument 1</h4>
                      <p className="text-gray-700 mb-3 leading-relaxed">
                        The plaintiff's interpretation of the statute is overly broad and would create an unworkable standard that conflicts with established precedent in similar cases.
                      </p>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-900">Supporting Authority:</span>
                          <p className="text-gray-600 mt-1">Smith v. Johnson (2020), Federal Circuit Court ruling on statutory interpretation</p>
                        </div>
                        <div>
                          <span className="font-medium text-gray-900">Factual Basis:</span>
                          <p className="text-gray-600 mt-1">Historical application of the statute shows narrow interpretation was intended by legislature</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleRebuttalClick(0)}
                      className="ml-4 w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors shadow-md hover:shadow-lg"
                      title="View rebuttals"
                    >
                      <Swords className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Opponent Argument 2 */}
                <div className="border-l-4 border-red-500 pl-4 bg-red-50 p-4 rounded-r-lg">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-2">Counter-Argument 2</h4>
                      <p className="text-gray-700 mb-3 leading-relaxed">
                        The evidence presented lacks sufficient foundation and reliability, failing to meet the standards required under the Federal Rules of Evidence.
                      </p>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-900">Supporting Authority:</span>
                          <p className="text-gray-600 mt-1">Daubert v. Merrell Dow Pharmaceuticals (1993), Evidence reliability standards</p>
                        </div>
                        <div>
                          <span className="font-medium text-gray-900">Factual Basis:</span>
                          <p className="text-gray-600 mt-1">Key evidence was obtained through unreliable methodology and lacks peer review</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleRebuttalClick(1)}
                      className="ml-4 w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors shadow-md hover:shadow-lg"
                      title="View rebuttals"
                    >
                      <Swords className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Opponent Argument 3 */}
                <div className="border-l-4 border-red-500 pl-4 bg-red-50 p-4 rounded-r-lg">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-2">Counter-Argument 3</h4>
                      <p className="text-gray-700 mb-3 leading-relaxed">
                        The plaintiff's claim is barred by the applicable statute of limitations, as the cause of action accrued more than two years prior to filing.
                      </p>
                      <div className="space-y-2 text-sm">
                        <div>
                          <span className="font-medium text-gray-900">Supporting Authority:</span>
                          <p className="text-gray-600 mt-1">State Civil Code Section 335.1, Discovery rule limitations</p>
                        </div>
                        <div>
                          <span className="font-medium text-gray-900">Factual Basis:</span>
                          <p className="text-gray-600 mt-1">Plaintiff had constructive notice of the claim as early as January 2022</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => handleRebuttalClick(2)}
                      className="ml-4 w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors shadow-md hover:shadow-lg"
                      title="View rebuttals"
                    >
                      <Swords className="h-4 w-4" />
                    </button>
                  </div>
                </div>
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
                {rebuttals[selectedOpponentArg as keyof typeof rebuttals]?.map((rebuttal, index) => (
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
                ))}
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

export default MootCourt;
