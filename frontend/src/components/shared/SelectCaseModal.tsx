import React, { useState } from 'react';
import { Search, FolderOpen, Check } from 'lucide-react';
import Modal from './Modal';
import { Case, mockCases } from '../../data/mockStrategyData';
import { LegalDocument } from '../../data/mockSearchData';

interface SelectCaseModalProps {
  isOpen: boolean;
  onClose: () => void;
  document: LegalDocument;
  onAddToCase: (caseItem: Case, document: LegalDocument) => void;
}

const SelectCaseModal: React.FC<SelectCaseModalProps> = ({
  isOpen,
  onClose,
  document,
  onAddToCase
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  
  const filteredCases = searchQuery
    ? mockCases.filter(caseItem => 
        caseItem.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        caseItem.caseNumber.toLowerCase().includes(searchQuery.toLowerCase()) ||
        caseItem.client.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : mockCases;

  const handleAddToCase = () => {
    if (!selectedCaseId) return;
    
    const selectedCase = mockCases.find(c => c.id === selectedCaseId);
    if (selectedCase) {
      onAddToCase(selectedCase, document);
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Add to Case"
    >
      <div className="space-y-4">
        <p className="text-sm text-gray-600">
          Select which case to add "{document.title}" to:
        </p>
        
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search your cases..."
            className="block w-full pl-10 pr-3 py-2 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
          />
        </div>
        
        <div className="max-h-80 overflow-y-auto">
          {filteredCases.length > 0 ? (
            <div className="space-y-2">
              {filteredCases.map((caseItem) => (
                <div 
                  key={caseItem.id}
                  onClick={() => setSelectedCaseId(caseItem.id)}
                  className={`flex items-start p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedCaseId === caseItem.id 
                      ? 'border-primary bg-blue-50' 
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <div className="mr-3 mt-1">
                    <div className={`p-2 rounded-full ${
                      selectedCaseId === caseItem.id ? 'bg-primary text-white' : 'bg-gray-100'
                    }`}>
                      {selectedCaseId === caseItem.id ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <FolderOpen className="h-4 w-4" />
                      )}
                    </div>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-sm font-medium text-gray-900">{caseItem.title}</h4>
                    <p className="text-xs text-gray-500 mt-1">Case #{caseItem.caseNumber}</p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`inline-block px-2 py-0.5 text-xs rounded-full ${
                        caseItem.status === 'active' ? 'bg-green-100 text-green-800' :
                        caseItem.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {caseItem.status.charAt(0).toUpperCase() + caseItem.status.slice(1)}
                      </span>
                      <span className={`inline-block px-2 py-0.5 text-xs rounded-full ${
                        caseItem.priority === 'high' ? 'bg-red-100 text-red-800' :
                        caseItem.priority === 'medium' ? 'bg-orange-100 text-orange-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {caseItem.priority.charAt(0).toUpperCase() + caseItem.priority.slice(1)} Priority
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500">No cases found</p>
            </div>
          )}
        </div>
        
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
          >
            Cancel
          </button>
          <button
            onClick={handleAddToCase}
            disabled={!selectedCaseId}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add to Case
          </button>
        </div>
      </div>
    </Modal>
  );
};

export default SelectCaseModal; 