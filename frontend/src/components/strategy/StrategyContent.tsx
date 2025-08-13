import React, { useState } from 'react';
import { FileText } from 'lucide-react';
import CaseFileNavigation from './CaseFileNavigation';
import CaseFileDetail from './CaseFileDetail';

const StrategyContent: React.FC = () => {
  // Case file management state
  const [selectedCaseFileId, setSelectedCaseFileId] = useState<number | null>(null);
  const [viewingCaseFileDetail, setViewingCaseFileDetail] = useState(false);

  const handleSelectCaseFile = (caseFileId: number) => {
    setSelectedCaseFileId(caseFileId);
    setViewingCaseFileDetail(true);
  };

  const handleBackToStrategy = () => {
    setViewingCaseFileDetail(false);
    setSelectedCaseFileId(null);
  };

  // Show case file detail view
  if (viewingCaseFileDetail && selectedCaseFileId) {
    return (
      <CaseFileDetail 
        caseFileId={selectedCaseFileId}
        onBack={handleBackToStrategy}
      />
    );
  }

  return (
    <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Legal Strategy</h2>
            <p className="text-gray-600">Build your case file and draft legal arguments</p>
          </div>
        </div>

        {/* Case File Navigation */}
        <div className="w-full">
          <CaseFileNavigation 
            onSelectCaseFile={handleSelectCaseFile}
            selectedCaseFileId={selectedCaseFileId}
          />
        </div>
      </div>
    </div>
  );
};

export default StrategyContent;