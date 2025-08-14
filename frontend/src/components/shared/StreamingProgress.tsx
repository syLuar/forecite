import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

export interface StreamingStep {
  id: string;
  brief_description: string;
  description?: string;
  status: 'pending' | 'active' | 'completed' | 'error';
  timestamp?: Date;
}

interface StreamingProgressProps {
  steps: StreamingStep[];
  isStreaming: boolean;
  error?: string;
}

const StreamingProgress: React.FC<StreamingProgressProps> = ({ 
  steps, 
  isStreaming, 
  error 
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const toggleExpanded = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  const getStatusIcon = (status: StreamingStep['status']) => {
    switch (status) {
      case 'active':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <div className="h-4 w-4 rounded-full border-2 border-gray-300" />;
    }
  };

  const getStatusColor = (status: StreamingStep['status']) => {
    switch (status) {
      case 'active':
        return 'text-blue-700 bg-blue-50';
      case 'completed':
        return 'text-green-700 bg-green-50';
      case 'error':
        return 'text-red-700 bg-red-50';
      default:
        return 'text-gray-700 bg-gray-50';
    }
  };

  if (!isStreaming && steps.length === 0 && !error) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          {isStreaming ? 'Processing...' : error ? 'Process Failed' : 'Process Complete'}
        </h3>
        {isStreaming && (
          <Loader2 className="h-5 w-5 text-primary animate-spin" />
        )}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700 font-medium">Error occurred</span>
          </div>
          <p className="text-red-600 mt-1 text-sm">{error}</p>
        </div>
      )}

      <div className="space-y-3">
        {steps.map((step, index) => (
          <div key={step.id} className="relative">
            <div className={`rounded-lg border transition-colors ${
              step.status === 'active' ? 'border-blue-200' : 
              step.status === 'completed' ? 'border-green-200' : 
              step.status === 'error' ? 'border-red-200' : 
              'border-gray-200'
            }`}>
              <div 
                className={`p-3 rounded-lg cursor-pointer hover:bg-opacity-80 transition-colors ${getStatusColor(step.status)}`}
                onClick={() => step.description && toggleExpanded(step.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(step.status)}
                    <span className="font-medium">{step.brief_description}</span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {step.timestamp && (
                      <span className="text-xs text-gray-500">
                        {step.timestamp.toLocaleTimeString()}
                      </span>
                    )}
                    {step.description && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpanded(step.id);
                        }}
                        className="p-1 hover:bg-white hover:bg-opacity-50 rounded transition-colors"
                      >
                        {expandedSteps.has(step.id) ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </button>
                    )}
                  </div>
                </div>
                
                {step.description && expandedSteps.has(step.id) && (
                  <div className="mt-3 pt-3 border-t border-gray-200 border-opacity-50">
                    <p className="text-sm leading-relaxed">{step.description}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {isStreaming && steps.length > 0 && (
        <div className="mt-4 text-center">
          <div className="inline-flex items-center px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm">
            <Loader2 className="h-3 w-3 mr-2 animate-spin" />
            In progress...
          </div>
        </div>
      )}
    </div>
  );
};

export default StreamingProgress;