import React, { useState } from 'react';
import { Calendar, Clock, FileText, User, Gavel, Plus, X, ChevronDown } from 'lucide-react';
import Modal from '../shared/Modal';

interface AddTimelineEventModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (event: {
    actor: 'our_side' | 'opposing_counsel' | 'judge';
    date: string;
    time: string;
    action: string;
    description: string;
    documents: string[];
  }) => void;
}

const AddTimelineEventModal: React.FC<AddTimelineEventModalProps> = ({
  isOpen,
  onClose,
  onSubmit
}) => {
  const [selectedActor, setSelectedActor] = useState<'our_side' | 'opposing_counsel' | 'judge' | null>(null);
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [action, setAction] = useState('');
  const [description, setDescription] = useState('');
  const [documents, setDocuments] = useState<string[]>(['']);

  const actors = [
    {
      id: 'our_side' as const,
      label: 'Our Side',
      icon: User,
      color: 'bg-blue-50 border-blue-200 text-blue-800',
      selectedColor: 'bg-blue-100 border-blue-300'
    },
    {
      id: 'opposing_counsel' as const,
      label: 'Opposing Counsel',
      icon: User,
      color: 'bg-red-50 border-red-200 text-red-800',
      selectedColor: 'bg-red-100 border-red-300'
    },
    {
      id: 'judge' as const,
      label: 'Judge',
      icon: Gavel,
      color: 'bg-purple-50 border-purple-200 text-purple-800',
      selectedColor: 'bg-purple-100 border-purple-300'
    }
  ];

  const actionOptions = {
    our_side: [
      'Filed Motion for Summary Judgment',
      'Submitted Brief',
      'Served Notice of Deposition',
      'Conducted Deposition',
      'Filed Discovery Request',
      'Served Subpoena',
      'Filed Response to Motion',
      'Submitted Evidence',
      'Filed Appeal',
      'Requested Settlement Conference',
      'Filed Motion to Compel',
      'Served Interrogatories',
      'Other'
    ],
    opposing_counsel: [
      'Filed Response Brief',
      'Objected to Motion',
      'Served Counter-Notice',
      'Filed Motion to Compel',
      'Submitted Discovery Responses',
      'Filed Motion to Dismiss',
      'Served Counter-Claim',
      'Filed Opposition',
      'Requested Extension',
      'Filed Objection',
      'Served Notice of Appeal',
      'Filed Settlement Offer',
      'Other'
    ],
    judge: [
      'Issued Court Order',
      'Scheduled Hearing',
      'Granted Motion',
      'Denied Motion',
      'Case Management Order',
      'Set Trial Date',
      'Issued Ruling',
      'Ordered Mediation',
      'Continued Hearing',
      'Issued Sanctions',
      'Approved Settlement',
      'Dismissed Case',
      'Other'
    ]
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedActor || !date || !time || !action || !description) return;

    onSubmit({
      actor: selectedActor,
      date,
      time,
      action,
      description,
      documents: documents.filter(doc => doc.trim() !== '')
    });

    // Reset form
    setSelectedActor(null);
    setDate('');
    setTime('');
    setAction('');
    setDescription('');
    setDocuments(['']);
    onClose();
  };

  const addDocumentField = () => {
    setDocuments([...documents, '']);
  };

  const updateDocument = (index: number, value: string) => {
    const updated = [...documents];
    updated[index] = value;
    setDocuments(updated);
  };

  const removeDocument = (index: number) => {
    setDocuments(documents.filter((_, i) => i !== index));
  };

  const getTodayDate = () => {
    return new Date().toISOString().split('T')[0];
  };

  const getCurrentTime = () => {
    return new Date().toTimeString().slice(0, 5);
  };

  React.useEffect(() => {
    if (isOpen) {
      setDate(getTodayDate());
      setTime(getCurrentTime());
    }
  }, [isOpen]);

  // Reset action when actor changes
  React.useEffect(() => {
    setAction('');
  }, [selectedActor]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Add Timeline Event">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Actor Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-900 mb-3">Who is taking this action?</label>
          <div className="grid grid-cols-1 gap-3">
            {actors.map((actor) => {
              const Icon = actor.icon;
              const isSelected = selectedActor === actor.id;
              return (
                <button
                  key={actor.id}
                  type="button"
                  onClick={() => setSelectedActor(actor.id)}
                  className={`flex items-center p-4 border-2 rounded-lg transition-all duration-200 ${
                    isSelected ? actor.selectedColor : actor.color
                  } ${isSelected ? 'ring-2 ring-offset-2 ring-blue-500' : 'hover:bg-opacity-80'}`}
                >
                  <Icon className="h-5 w-5 mr-3" />
                  <span className="font-medium">{actor.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {selectedActor && (
          <>
            {/* Date and Time */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Date</label>
                <div className="relative">
                  <Calendar className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <input
                    type="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    required
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Time</label>
                <div className="relative">
                  <Clock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <input
                    type="time"
                    value={time}
                    onChange={(e) => setTime(e.target.value)}
                    required
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>

            {/* Action Dropdown */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Action</label>
              <div className="relative">
                <select
                  value={action}
                  onChange={(e) => setAction(e.target.value)}
                  required
                  className="block w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none bg-white"
                >
                  <option value="">Select an action...</option>
                  {actionOptions[selectedActor].map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 h-4 w-4 text-gray-400 pointer-events-none" />
              </div>
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Provide details about this action..."
                required
                rows={3}
                className="block w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              />
            </div>

            {/* Documents (Optional) */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Documents (Optional)</label>
              <div className="space-y-2">
                {documents.map((doc, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className="relative flex-1">
                      <FileText className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                      <input
                        type="text"
                        value={doc}
                        onChange={(e) => updateDocument(index, e.target.value)}
                        placeholder="Document name"
                        className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    {documents.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeDocument(index)}
                        className="p-2 text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                ))}
                <button
                  type="button"
                  onClick={addDocumentField}
                  className="flex items-center text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add Document
                </button>
              </div>
            </div>

            {/* Submit Buttons */}
            <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200"
              >
                Add Event
              </button>
            </div>
          </>
        )}
      </form>
    </Modal>
  );
};

export default AddTimelineEventModal; 