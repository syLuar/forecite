import React, { useState } from 'react';
import { ArrowLeft, Plus, Scale } from 'lucide-react';
import { mockCases, mockTimelineEvents, mockAIRecommendations, TimelineEvent } from '../../data/mockStrategyData';
import CaseCard from './CaseCard';
import TimelineView from './TimelineView';
import AIRecommendations from './AIRecommendations';
import AddTimelineEventModal from './AddTimelineEventModal';

const StrategyContent: React.FC = () => {
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [isAddEventModalOpen, setIsAddEventModalOpen] = useState(false);
  const [timelineEvents, setTimelineEvents] = useState(mockTimelineEvents);
  const [isAIRecommendationsLoading, setIsAIRecommendationsLoading] = useState(false);

  const selectedCase = selectedCaseId ? mockCases.find(c => c.id === selectedCaseId) : null;
  const currentTimelineEvents = selectedCaseId ? timelineEvents[selectedCaseId] || [] : [];
  const aiRecommendations = selectedCaseId ? mockAIRecommendations[selectedCaseId] || [] : [];

  const handleAddTimelineEvent = (eventData: {
    actor: 'our_side' | 'opposing_counsel' | 'judge';
    date: string;
    time: string;
    action: string;
    description: string;
    documents: string[];
  }) => {
    if (!selectedCaseId) return;

    const newEvent: TimelineEvent = {
      id: `custom_${Date.now()}`,
      date: eventData.date,
      time: eventData.time,
      actor: eventData.actor,
      action: eventData.action,
      description: eventData.description,
      documents: eventData.documents.length > 0 ? eventData.documents : undefined
    };

    // Add the new event to the timeline (will be sorted by TimelineView)
    setTimelineEvents(prev => ({
      ...prev,
      [selectedCaseId]: [...(prev[selectedCaseId] || []), newEvent]
    }));

    // Trigger AI recommendations loading
    setIsAIRecommendationsLoading(true);
    
    // Show loading for 2.5 seconds, then show recommendations again
    setTimeout(() => {
      setIsAIRecommendationsLoading(false);
    }, 2500);
  };

  const LoadingSpinner = () => (
    <div className="flex flex-col items-center justify-center py-20">
      <Scale className="h-24 w-24 text-primary animate-bounce mb-4" />
      <p className="text-lg text-gray-600 font-medium">Analyzing case updates...</p>
      <p className="text-sm text-gray-500 mt-1">AI is reviewing the new timeline event and updating recommendations</p>
    </div>
  );

  if (selectedCaseId && selectedCase) {
    return (
      <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
        <div className="max-w-6xl mx-auto">
          {/* Case Detail Header */}
          <div className="flex items-center justify-between mb-8">
            <button
              onClick={() => setSelectedCaseId(null)}
              className="flex items-center text-primary hover:text-primary-700 font-medium"
            >
              <ArrowLeft className="h-5 w-5 mr-2" />
              Back to Cases
            </button>
            <div className="text-right">
              <h1 className="text-2xl font-bold text-gray-900">{selectedCase.title}</h1>
              <p className="text-gray-600">{selectedCase.caseNumber}</p>
            </div>
          </div>

          {/* Case Info Summary */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-2">Client</h3>
                <p className="text-gray-700">{selectedCase.client}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-2">Opposing Party</h3>
                <p className="text-gray-700">{selectedCase.opposingParty}</p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-900 mb-2">Court</h3>
                <p className="text-gray-700">{selectedCase.court}</p>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h3 className="text-sm font-medium text-gray-900 mb-2">Case Description</h3>
              <p className="text-gray-700 leading-relaxed">{selectedCase.description}</p>
            </div>
          </div>

          {/* Timeline */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-8 border border-gray-200">
            <TimelineView events={currentTimelineEvents} />
          </div>

          {/* AI Recommendations */}
          <div className="bg-gray-50 rounded-lg shadow-md p-6 border border-gray-200">
            {isAIRecommendationsLoading ? (
              <LoadingSpinner />
            ) : (
              <>
                <AIRecommendations recommendations={aiRecommendations} />
                
                {/* Add Event Button */}
                <div className="mt-8 pt-6 border-t border-gray-300">
                  <div className="text-center">
                    <button
                      onClick={() => setIsAddEventModalOpen(true)}
                      className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 font-medium"
                    >
                      <Plus className="h-5 w-5 mr-2" />
                      Add Timeline Event
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Add Timeline Event Modal */}
        <AddTimelineEventModal
          isOpen={isAddEventModalOpen}
          onClose={() => setIsAddEventModalOpen(false)}
          onSubmit={handleAddTimelineEvent}
        />
      </div>
    );
  }

  return (
    <div className="flex-1 p-4 md:p-6 pb-32 md:pb-12">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">Case Management</h2>
            <p className="text-gray-600">Manage your active legal cases and develop winning strategies</p>
          </div>
          <button className="flex items-center px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors duration-200">
            <Plus className="h-5 w-5 mr-2" />
            New Case
          </button>
        </div>

        {/* Cases Grid */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-gray-900">
              Your Cases ({mockCases.length})
            </h3>
          </div>
          
          <div className="grid grid-cols-1 gap-6">
            {mockCases.map((caseItem) => (
              <CaseCard
                key={caseItem.id}
                case={caseItem}
                onClick={setSelectedCaseId}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategyContent; 