import React, { useState, useEffect } from 'react';
import { X, Bot, Wand2 } from 'lucide-react';

interface AIEditModalProps {
  isOpen: boolean;
  onClose: () => void;
  draftId: number;
  currentContent: string;
  onEdit: (editInstructions: string) => void;
  isEditing: boolean;
}

const AIEditModal: React.FC<AIEditModalProps> = ({ 
  isOpen, 
  onClose, 
  draftId,
  currentContent,
  onEdit, 
  isEditing
}) => {
  const [editInstructions, setEditInstructions] = useState('');

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setEditInstructions('');
    }
  }, [isOpen]);

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
    if (editInstructions.trim()) {
      onEdit(editInstructions.trim());
    }
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
              <Bot className="h-6 w-6 text-primary mr-3" />
              <h2 className="text-2xl font-bold text-gray-900">AI-Assisted Draft Editing</h2>
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
            <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-400 rounded">
              <p className="text-sm text-blue-700">
                <strong>AI Editing:</strong> Describe the changes you want to make to your legal argument. 
                The AI will revise the draft while preserving the legal structure and citations.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-lg font-medium text-gray-700 mb-3">
                  Edit Instructions
                </label>
                <textarea
                  value={editInstructions}
                  onChange={(e) => setEditInstructions(e.target.value)}
                  placeholder="Describe what changes you want to make to the argument. For example: 'Make the introduction more concise', 'Add stronger emphasis on the precedent from Smith v. Jones', 'Reorganize the counterargument section', etc."
                  className="w-full h-40 px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-gray-900 placeholder-gray-400 text-base resize-none"
                  required
                />
              </div>

              {/* Preview of current content length */}
              <div className="text-sm text-gray-600">
                <p>Current draft: {currentContent.split(' ').length} words</p>
              </div>
              
              <div className="flex justify-end pt-6">
                <button
                  type="submit"
                  disabled={isEditing || !editInstructions.trim()}
                  className="flex items-center justify-center px-8 py-3 bg-primary text-white rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 font-medium text-base"
                >
                  {isEditing ? (
                    <>
                      <Wand2 className="h-5 w-5 mr-2 animate-spin" />
                      AI is editing...
                    </>
                  ) : (
                    <>
                      <Wand2 className="h-5 w-5 mr-2" />
                      Edit with AI
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

export default AIEditModal;