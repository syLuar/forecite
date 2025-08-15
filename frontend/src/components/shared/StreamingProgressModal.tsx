import React, { useState, useEffect, useMemo, useRef } from 'react';
import { ChevronDown, ChevronUp, Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';

export interface StreamingStep {
  id: string;
  brief_description: string;
  description?: string;
  status: 'pending' | 'active' | 'completed' | 'error' | 'in_progress' | 'complete';
  timestamp?: Date;
}

interface ProcessedStep extends StreamingStep {
  startTime?: Date;
  endTime?: Date;
  elapsedMs?: number;
}

interface StreamingProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  steps: StreamingStep[];
  isStreaming: boolean;
  error?: string;
  title?: string;
  allowClose?: boolean; // Whether to allow closing during streaming
  autoCloseDelay?: number; // Delay in ms before auto-closing on success (default: 2000)
}

const StreamingProgressModal: React.FC<StreamingProgressModalProps> = ({ 
  isOpen,
  onClose,
  steps, 
  isStreaming, 
  error,
  title = "Processing",
  allowClose = false,
  autoCloseDelay = 500
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const stepTimingsRef = useRef<Map<string, Date>>(new Map());

  // Process steps to handle status transitions and timing
  const processedSteps = useMemo(() => {
    const stepMap = new Map<string, ProcessedStep>();
    const timings = stepTimingsRef.current;

    for (const step of steps) {
      const processedStep: ProcessedStep = { ...step };

      if (step.status === 'in_progress') {
        // Track start time for in_progress steps
        if (!timings.has(step.id)) {
          timings.set(step.id, step.timestamp || new Date());
        }
        processedStep.startTime = timings.get(step.id);
        
        // Map active status for display
        processedStep.status = 'active';
        stepMap.set(step.id, processedStep);
      } else if (step.status === 'complete') {
        // Handle completion - replace the in_progress step
        const startTime = timings.get(step.id);
        const endTime = step.timestamp || new Date();
        
        processedStep.status = 'completed';
        processedStep.startTime = startTime;
        processedStep.endTime = endTime;
        
        if (startTime) {
          processedStep.elapsedMs = endTime.getTime() - startTime.getTime();
        }
        
        // Always replace any existing step with the same ID
        stepMap.set(step.id, processedStep);
      } else {
        // Handle other statuses - only add if not already present
        if (!stepMap.has(step.id)) {
          stepMap.set(step.id, processedStep);
        }
      }
    }

    // Convert map to array maintaining the order of first appearance
    const seenIds = new Set<string>();
    const orderedSteps: ProcessedStep[] = [];
    
    for (const step of steps) {
      if (!seenIds.has(step.id)) {
        const processedStep = stepMap.get(step.id);
        if (processedStep) {
          orderedSteps.push(processedStep);
          seenIds.add(step.id);
        }
      }
    }
    
    return orderedSteps;
  }, [steps]);

  // Handle escape key and body scroll
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && (allowClose || !isStreaming)) {
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
  }, [isOpen, onClose, allowClose, isStreaming]);

  // Auto-close modal when streaming completes successfully
  useEffect(() => {
    if (!isStreaming && !error && processedSteps.length > 0 && isOpen) {
      // Check if all steps are completed
      const allCompleted = processedSteps.every(step => step.status === 'completed');
      if (allCompleted) {
        const timer = setTimeout(() => {
          onClose();
        }, autoCloseDelay);

        return () => clearTimeout(timer);
      }
    }
  }, [isStreaming, error, processedSteps, isOpen, onClose, autoCloseDelay]);

  const toggleExpanded = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  const getStatusIcon = (status: ProcessedStep['status']) => {
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

  const getStatusColor = (status: ProcessedStep['status']) => {
    switch (status) {
      case 'active':
        return 'text-blue-700 bg-blue-50 border-blue-200';
      case 'completed':
        return 'text-green-700 bg-green-50 border-green-200';
      case 'error':
        return 'text-red-700 bg-red-50 border-red-200';
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  const formatElapsedTime = (elapsedMs: number) => {
    if (elapsedMs < 1000) {
      return `${elapsedMs}ms`;
    } else if (elapsedMs < 60000) {
      return `${(elapsedMs / 1000).toFixed(1)}s`;
    } else {
      const minutes = Math.floor(elapsedMs / 60000);
      const seconds = Math.floor((elapsedMs % 60000) / 1000);
      return `${minutes}m ${seconds}s`;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop with blur effect */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm transition-all duration-300"
        onClick={(allowClose || !isStreaming) ? onClose : undefined}
      />
      
      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-2xl transform transition-all duration-300 animate-in slide-in-from-bottom-4 fade-in-0">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              {isStreaming ? (
                <Loader2 className="h-6 w-6 text-primary animate-spin" />
              ) : error ? (
                <AlertCircle className="h-6 w-6 text-red-500" />
              ) : (
                <CheckCircle className="h-6 w-6 text-green-500" />
              )}
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  {isStreaming ? title : error ? 'Failed' : 'Complete'}
                </h2>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 max-h-96 overflow-y-auto">
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg animate-in slide-in-from-top-2 fade-in-0">
                <div className="flex items-center">
                  <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
                  <span className="text-red-700 font-medium">Error occurred</span>
                </div>
                <p className="text-red-600 mt-1 text-sm">{error}</p>
              </div>
            )}

            {processedSteps.length > 0 ? (
              <div className="space-y-3">
                {processedSteps.map((step, index) => (
                  <div 
                    key={step.id} 
                    className="transform transition-all duration-300 animate-in slide-in-from-left-1 fade-in-0"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className={`rounded-lg border transition-all duration-200 ${getStatusColor(step.status)}`}>
                      <div 
                        className={`p-4 rounded-lg cursor-pointer hover:bg-opacity-80 transition-all duration-200`}
                        onClick={() => step.description && toggleExpanded(step.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3">
                            {getStatusIcon(step.status)}
                            <span className="font-medium">{step.brief_description}</span>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {step.elapsedMs !== undefined && (
                              <div className="flex items-center space-x-1 text-xs text-gray-500 bg-white bg-opacity-50 px-2 py-1 rounded">
                                <Clock className="h-3 w-3" />
                                <span>{formatElapsedTime(step.elapsedMs)}</span>
                              </div>
                            )}
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
                                className="p-1 hover:bg-white hover:bg-opacity-50 rounded transition-all duration-200"
                              >
                                <div className="transform transition-transform duration-200">
                                  {expandedSteps.has(step.id) ? (
                                    <ChevronUp className="h-4 w-4" />
                                  ) : (
                                    <ChevronDown className="h-4 w-4" />
                                  )}
                                </div>
                              </button>
                            )}
                          </div>
                        </div>
                        
                        {step.description && expandedSteps.has(step.id) && (
                          <div className="mt-3 pt-3 border-t border-gray-200 border-opacity-50 animate-in slide-in-from-top-2 fade-in-0">
                            <p className="text-sm leading-relaxed">{step.description}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : !error && !isStreaming ? (
              <div className="text-center py-8">
                <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-3" />
                <p className="text-gray-600">Process completed successfully!</p>
              </div>
            ) : !error && isStreaming ? (
              <div className="text-center py-8">
                <Loader2 className="h-12 w-12 text-primary animate-spin mx-auto mb-3" />
                <p className="text-gray-600">Initializing process...</p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

export default StreamingProgressModal;