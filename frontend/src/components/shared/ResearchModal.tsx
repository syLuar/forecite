import React, { useState, useEffect } from 'react';
import { X, Bot, Plus, Minus } from 'lucide-react';

interface ResearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseFileId: number;
  onConductResearch: (legalIssues?: string[]) => void;
  isResearching: boolean;
}

const ResearchModal: React.FC<ResearchModalProps> = ({ 
  isOpen, 
  onClose, 
  caseFileId, 
  onConductResearch, 
  isResearching
}) => {
  const [legalIssues, setLegalIssues] = useState<string[]>(['']);
  const [useAIIdentification, setUseAIIdentification] = useState<boolean>(true);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setLegalIssues(['']);
      setUseAIIdentification(true);
    }
  }, [isOpen]);

  // Escape key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isResearching) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose, isResearching]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (useAIIdentification) {
      // Let AI identify legal issues from case facts
      onConductResearch();
    } else {
      // Filter out empty issues for manual specification
      const filteredIssues = legalIssues.filter(issue => issue.trim());
      
      if (filteredIssues.length === 0) {
        alert('Please enter at least one legal issue to research, or use AI identification.');
        return;
      }

      onConductResearch(filteredIssues);
    }
  };

  const addLegalIssue = () => {
    setLegalIssues([...legalIssues, '']);
  };

  const removeLegalIssue = (index: number) => {
    if (legalIssues.length > 1) {
      setLegalIssues(legalIssues.filter((_, i) => i !== index));
    }
  };

  const updateLegalIssue = (index: number, value: string) => {
    const updated = [...legalIssues];
    updated[index] = value;
    setLegalIssues(updated);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center">
            <Bot className="h-6 w-6 text-blue-600 mr-3" />
            <h2 className="text-xl font-semibold text-gray-900">Research with Forecite AI</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            disabled={isResearching}
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6">
          <div className="mb-6">
            <p className="text-gray-600 mb-4">
              Let Forecite AI conduct comprehensive legal research on your case. The AI will systematically 
              research each legal issue, find relevant authorities, and automatically add discovered documents 
              and research notes to your case file.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2">What the AI will do:</h4>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• {useAIIdentification ? 'Analyze case facts to identify legal issues' : 'Research your specified legal issues'}</li>
                <li>• Search for relevant cases with similar fact patterns</li>
                <li>• Find applicable statutes and regulations</li>
                <li>• Extract key legal holdings and principles</li>
                <li>• Analyze precedent chains and citation networks</li>
                <li>• Add relevant documents to your case file</li>
                <li>• Generate strategic research notes and insights</li>
              </ul>
            </div>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Legal Issue Identification
            </label>
            <div className="space-y-4">
              <div className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    id="ai-identification"
                    type="radio"
                    checked={useAIIdentification}
                    onChange={() => setUseAIIdentification(true)}
                    className="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300"
                    disabled={isResearching}
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="ai-identification" className="font-medium text-gray-700">
                    Let AI identify legal issues from case facts (Recommended)
                  </label>
                  <p className="text-gray-500">
                    The AI will analyze your case facts and automatically identify all relevant legal issues, 
                    claims, and potential causes of action.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="flex items-center h-5">
                  <input
                    id="manual-specification"
                    type="radio"
                    checked={!useAIIdentification}
                    onChange={() => setUseAIIdentification(false)}
                    className="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300"
                    disabled={isResearching}
                  />
                </div>
                <div className="ml-3 text-sm">
                  <label htmlFor="manual-specification" className="font-medium text-gray-700">
                    Specify legal issues manually
                  </label>
                  <p className="text-gray-500">
                    Provide specific legal issues you want the AI to focus on during research.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {!useAIIdentification && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Legal Issues to Research *
              </label>
              <div className="space-y-3">
                {legalIssues.map((issue, index) => (
                  <div key={index} className="flex items-center gap-3">
                    <div className="flex-1">
                      <input
                        type="text"
                        value={issue}
                        onChange={(e) => updateLegalIssue(index, e.target.value)}
                        placeholder={`Legal issue ${index + 1} (e.g., "breach of contract", "negligence", "damages")`}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        disabled={isResearching}
                      />
                    </div>
                    {legalIssues.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeLegalIssue(index)}
                        className="p-2 text-red-600 hover:text-red-800 transition-colors"
                        disabled={isResearching}
                        title="Remove this legal issue"
                      >
                        <Minus className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              
              <button
                type="button"
                onClick={addLegalIssue}
                className="mt-3 flex items-center text-sm text-blue-600 hover:text-blue-800 transition-colors"
                disabled={isResearching}
              >
                <Plus className="h-4 w-4 mr-1" />
                Add another legal issue
              </button>
            </div>
          )}

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 border border-gray-300 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-colors"
              disabled={isResearching}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={isResearching || (!useAIIdentification && legalIssues.every(issue => !issue.trim()))}
            >
              {isResearching ? 'Researching...' : 'Start Research'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ResearchModal;
