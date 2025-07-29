import React from 'react';
import { Search, Gavel, Target } from 'lucide-react';

interface BottomNavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const BottomNavigation: React.FC<BottomNavigationProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'search', label: 'Search', icon: Search },
    { id: 'judge', label: 'Judge', icon: Gavel },
    { id: 'argue', label: 'Strategy', icon: Target },
  ];

  return (
    <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg">
      <div className="flex">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex-1 flex flex-col items-center py-3 px-2 transition-colors duration-200 ${
                activeTab === tab.id
                  ? 'text-primary bg-primary/10'
                  : 'text-gray-500 hover:text-primary hover:bg-gray-50'
              }`}
            >
              <Icon className="h-6 w-6 mb-1" />
              <span className="text-xs font-medium">{tab.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
};

export default BottomNavigation; 