import React from 'react';
import { User, Building, FileText } from 'lucide-react';
import { Case } from '../../data/mockStrategyData';

interface CaseCardProps {
  case: Case;
  onClick: (caseId: string) => void;
}

const CaseCard: React.FC<CaseCardProps> = ({ case: caseData, onClick }) => {
  return (
    <div
      onClick={() => onClick(caseData.id)}
      className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 p-6 cursor-pointer border border-gray-200"
    >
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{caseData.title}</h3>
        <p className="text-sm text-gray-600 mb-3 leading-relaxed">{caseData.description}</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-gray-600">
        <div className="flex items-center">
          <User className="h-4 w-4 mr-2 text-gray-400" />
          <span><strong>Client:</strong> {caseData.client}</span>
        </div>
        <div className="flex items-center">
          <Building className="h-4 w-4 mr-2 text-gray-400" />
          <span><strong>Opposing:</strong> {caseData.opposingParty}</span>
        </div>
        <div className="flex items-center">
          <FileText className="h-4 w-4 mr-2 text-gray-400" />
          <span><strong>Case #:</strong> {caseData.caseNumber}</span>
        </div>
        <div className="flex items-center">
          <Building className="h-4 w-4 mr-2 text-gray-400" />
          <span><strong>Court:</strong> {caseData.court}</span>
        </div>
      </div>
    </div>
  );
};

export default CaseCard; 