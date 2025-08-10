import React, { useState, useEffect } from 'react';
import { X, PenTool } from 'lucide-react';

interface DraftArgumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseFileId: number;
  onDraft: (userFacts: string, legalQuestion: string) => void;
  isDrafting: boolean;
  initialFacts?: string;
  initialQuestion?: string;
}

const DraftArgumentModal: React.FC<DraftArgumentModalProps> = ({ 
  isOpen, 
  onClose, 
  caseFileId, 
  onDraft, 
  isDrafting,
  initialFacts = '',
  initialQuestion = ''
}) => {
  const [userFacts, setUserFacts] = useState('');
  const [legalQuestion, setLegalQuestion] = useState('');

  // Initialize form with passed values
  useEffect(() => {
    if (isOpen) {
      setUserFacts(initialFacts);
      setLegalQuestion(initialQuestion);
    }
  }, [isOpen, initialFacts, initialQuestion]);

  // Escape key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
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
  }, [isOpen, onClose]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!userFacts.trim()) {
      alert('Please enter the case facts before drafting an argument.');
      return;
    }
    onDraft(userFacts, legalQuestion);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-lg shadow-xl w-full max-w-2xl">
          {/* Modal Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center">
              <PenTool className="h-6 w-6 text-primary mr-3" />
              <h2 className="text-2xl font-bold text-gray-900">Draft Legal Argument</h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>
          </div>

          {/* Modal Content */}
          <div className="p-6">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-lg font-medium text-gray-700 mb-3">
                  Case Facts *
                </label>
                <textarea
                  value={userFacts}
                  onChange={(e) => setUserFacts(e.target.value)}
                  placeholder="Describe the key facts of your case..."
                  className="w-full h-40 px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-gray-900 placeholder-gray-400 text-base"
                  required
                />
              </div>
              
              <div>
                <label className="block text-lg font-medium text-gray-700 mb-3">
                  Legal Question (Optional)
                </label>
                <textarea
                  value={legalQuestion}
                  onChange={(e) => setLegalQuestion(e.target.value)}
                  placeholder="What specific legal question should the argument address?"
                  className="w-full h-32 px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-gray-900 placeholder-gray-400 text-base"
                />
              </div>
              
              <div className="flex justify-end pt-6">
                <button
                  type="submit"
                  disabled={isDrafting || !userFacts.trim()}
                  className="flex items-center justify-center px-8 py-3 bg-primary text-white rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 font-medium text-base"
                >
                  {isDrafting ? (
                    <>
                      <PenTool className="h-5 w-5 mr-2 animate-spin" />
                      Drafting Argument...
                    </>
                  ) : (
                    <>
                      <PenTool className="h-5 w-5 mr-2" />
                      Draft Legal Argument
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DraftArgumentModal;
