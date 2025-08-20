import React, { useState } from 'react';
import Header from './components/Header';
import BottomNavigation from './components/BottomNavigation';
import SearchContent from './components/search/SearchContent';
import StrategyContent from './components/strategy/StrategyContent';
import { ErrorBoundary } from './components/shared/ErrorBoundary';

function App() {
  const [activeTab, setActiveTab] = useState('search');

  const handleTabChange = (tab: string) => {
    if (tab === 'guide') {
      // Open user guide in new tab and stay on current page
      window.open('/user-guide/', '_blank');
      return;
    }
    setActiveTab(tab);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'search':
        return <SearchContent />;
      case 'argue':
        return <StrategyContent />;
      default:
        return <SearchContent />;
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-offwhite flex flex-col">
        <Header />
        
        <main className="flex-1 pb-20 md:pb-6">
          {renderContent()}
        </main>
        
        <BottomNavigation 
          activeTab={activeTab} 
          onTabChange={handleTabChange} 
        />
      </div>
    </ErrorBoundary>
  );
}

export default App;
