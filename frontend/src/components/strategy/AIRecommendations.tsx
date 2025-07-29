import React, { useState } from 'react';
import { FileText, Lightbulb, HelpCircle, ExternalLink, Target } from 'lucide-react';
import { AIRecommendation } from '../../data/mockStrategyData';
import Modal from '../shared/Modal';

interface AIRecommendationsProps {
  recommendations: AIRecommendation[];
}

const AIRecommendations: React.FC<AIRecommendationsProps> = ({ recommendations }) => {
  const [selectedRecommendation, setSelectedRecommendation] = useState<AIRecommendation | null>(null);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'motion': return <FileText className="h-5 w-5" />;
      case 'discovery': return <FileText className="h-5 w-5" />;
      case 'settlement': return <FileText className="h-5 w-5" />;
      case 'brief': return <FileText className="h-5 w-5" />;
      case 'evidence': return <FileText className="h-5 w-5" />;
      default: return <FileText className="h-5 w-5" />;
    }
  };

  const formatWinProbability = (probability: number) => {
    return `${Math.round(probability * 100)}%`;
  };

  return (
    <>
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <Lightbulb className="h-6 w-6 text-yellow-500 mr-2" />
          <h3 className="text-xl font-semibold text-gray-900">AI-Recommended Actions</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {recommendations.map((recommendation) => (
            <div
              key={recommendation.id}
              onClick={() => setSelectedRecommendation(recommendation)}
              className="bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 p-6 cursor-pointer border border-gray-200 hover:border-primary group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="p-2 rounded-lg bg-blue-100 text-blue-800 border border-blue-200">
                  {getTypeIcon(recommendation.type)}
                </div>
                <div className="flex items-center bg-green-50 text-green-800 px-2.5 py-1 rounded-full border border-green-200">
                  <Target className="h-3 w-3 mr-1" />
                  <span className="text-xs font-medium">
                    {formatWinProbability(recommendation.winProbability)} win chance
                  </span>
                </div>
              </div>
              
              <h4 className="text-lg font-semibold text-gray-900 mb-2 group-hover:text-primary transition-colors">
                {recommendation.title}
              </h4>
              <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                {recommendation.description}
              </p>
              
              <div className="mt-4 flex items-center justify-between">
                <button className="text-sm text-primary hover:text-primary-700 font-medium flex items-center">
                  <HelpCircle className="h-4 w-4 mr-1" />
                  Why this action?
                </button>
                <div className="w-6 h-6 rounded-full bg-gray-100 group-hover:bg-primary group-hover:text-white transition-colors flex items-center justify-center">
                  <ExternalLink className="h-3 w-3" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommendation Explanation Modal */}
      {selectedRecommendation && (
        <Modal
          isOpen={!!selectedRecommendation}
          onClose={() => setSelectedRecommendation(null)}
          title={selectedRecommendation.title}
        >
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center bg-green-50 text-green-800 px-3 py-1.5 rounded-full border border-green-200">
                <Target className="h-4 w-4 mr-1" />
                <span className="text-sm font-medium">
                  {formatWinProbability(selectedRecommendation.winProbability)} chance to improve case outcome
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Type: {selectedRecommendation.type.charAt(0).toUpperCase() + selectedRecommendation.type.slice(1)}
              </div>
            </div>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="text-sm font-medium text-blue-900 mb-2">AI Analysis & Recommendation</h4>
              <p className="text-sm text-blue-800 leading-relaxed">{selectedRecommendation.explanation}</p>
            </div>
            
            <div className="flex justify-center pt-4 space-x-3">
              <button
                onClick={() => {
                  // TODO: Implement action execution
                  console.log('Starting action:', selectedRecommendation.title);
                }}
                className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors duration-200"
              >
                <ExternalLink className="h-4 w-4 mr-2" />
                Start This Action
              </button>
              <button
                onClick={() => {
                  // TODO: Implement more information
                  console.log('Getting more info for:', selectedRecommendation.title);
                }}
                className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors duration-200"
              >
                <HelpCircle className="h-4 w-4 mr-2" />
                Learn More
              </button>
            </div>
          </div>
        </Modal>
      )}
    </>
  );
};

export default AIRecommendations; 