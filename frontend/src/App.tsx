import React, { useState } from 'react';
import Header from './components/Header';
import BottomNavigation from './components/BottomNavigation';
import SearchContent from './components/search/SearchContent';
import JudgeContent from './components/JudgeContent';
import StrategyContent from './components/strategy/StrategyContent';

function App() {
  const [activeTab, setActiveTab] = useState('search');

  const renderContent = () => {
    switch (activeTab) {
      case 'search':
        return <SearchContent />;
      case 'judge':
        return <JudgeContent />;
      case 'argue':
        return <StrategyContent />;
      default:
        return <SearchContent />;
    }
  };

  return (
    <div className="min-h-screen bg-offwhite flex flex-col">
      <Header />
      
      <main className="flex-1 pb-20 md:pb-6">
        {renderContent()}
      </main>
      
      <BottomNavigation 
        activeTab={activeTab} 
        onTabChange={setActiveTab} 
      />
    </div>
  );
}

export default App;
