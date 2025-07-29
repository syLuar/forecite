import React from 'react';
import { Calendar, Clock, FileText, User, Scale } from 'lucide-react';
import { TimelineEvent } from '../../data/mockStrategyData';

interface TimelineViewProps {
  events: TimelineEvent[];
}

const TimelineView: React.FC<TimelineViewProps> = ({ events }) => {
  const getActorInfo = (actor: string) => {
    switch (actor) {
      case 'our_side':
        return {
          label: 'Our Side',
          color: 'bg-blue-500',
          bgColor: 'bg-blue-50',
          textColor: 'text-blue-800',
          icon: <User className="h-4 w-4 text-white" />
        };
      case 'opposing_counsel':
        return {
          label: 'Opposing Counsel',
          color: 'bg-red-500',
          bgColor: 'bg-red-50',
          textColor: 'text-red-800',
          icon: <User className="h-4 w-4 text-white" />
        };
      case 'judge':
        return {
          label: 'Judge',
          color: 'bg-gray-500',
          bgColor: 'bg-gray-50',
          textColor: 'text-gray-800',
          icon: <Scale className="h-4 w-4 text-white" />
        };
      default:
        return {
          label: 'Unknown',
          color: 'bg-gray-400',
          bgColor: 'bg-gray-50',
          textColor: 'text-gray-800',
          icon: <FileText className="h-4 w-4 text-white" />
        };
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const formatTime = (timeString: string) => {
    // Convert 24-hour format to 12-hour format with AM/PM
    const [hours, minutes] = timeString.split(':');
    const hour24 = parseInt(hours);
    const hour12 = hour24 === 0 ? 12 : hour24 > 12 ? hour24 - 12 : hour24;
    const ampm = hour24 >= 12 ? 'PM' : 'AM';
    return `${hour12}:${minutes} ${ampm}`;
  };

  // Sort events by date and time (oldest first, newest last)
  const sortedEvents = [...events].sort((a, b) => {
    const dateA = new Date(`${a.date} ${a.time}`);
    const dateB = new Date(`${b.date} ${b.time}`);
    return dateA.getTime() - dateB.getTime();
  });

  return (
    <div className="space-y-6">
      <h3 className="text-xl font-semibold text-gray-900 mb-4">Case Timeline</h3>
      
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200"></div>
        
        {sortedEvents.map((event, index) => {
          const actorInfo = getActorInfo(event.actor);
          
          return (
            <div key={event.id} className="relative flex items-start space-x-4 pb-8">
              {/* Timeline dot */}
              <div className={`relative z-0 flex items-center justify-center w-16 h-16 rounded-full ${actorInfo.color} shadow-lg`}>
                {actorInfo.icon}
              </div>
              
              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className={`p-4 rounded-lg border-l-4 ${actorInfo.bgColor} border-l-current`} style={{borderLeftColor: actorInfo.color.replace('bg-', '')}}>
                  <div className="flex items-center justify-between mb-2">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${actorInfo.textColor} bg-white`}>
                      {actorInfo.label}
                    </span>
                    <div className="flex items-center text-sm text-gray-500 space-x-3">
                      <div className="flex items-center">
                        <Calendar className="h-4 w-4 mr-1" />
                        {formatDate(event.date)}
                      </div>
                      <div className="flex items-center">
                        <Clock className="h-4 w-4 mr-1" />
                        {formatTime(event.time)}
                      </div>
                    </div>
                  </div>
                  
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">{event.action}</h4>
                  <p className="text-gray-700 mb-3 leading-relaxed">{event.description}</p>
                  
                  {event.documents && event.documents.length > 0 && (
                    <div className="mt-3">
                      <p className="text-sm font-medium text-gray-900 mb-2">Related Documents:</p>
                      <div className="flex flex-wrap gap-2">
                        {event.documents.map((doc, docIndex) => (
                          <span
                            key={docIndex}
                            className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                          >
                            <FileText className="h-3 w-3 mr-1" />
                            {doc}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        
        {events.length === 0 && (
          <div className="text-center py-8">
            <Clock className="h-12 w-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500">No timeline events yet.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TimelineView; 