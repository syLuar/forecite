import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import BottomNavigation from './components/BottomNavigation';
import SearchContent from './components/search/SearchContent';
import StrategyContent from './components/strategy/StrategyContent';
import { ErrorBoundary } from './components/shared/ErrorBoundary';
import { ToastProvider, useToast } from './contexts/ToastContext';

function AppContent() {
  const [activeTab, setActiveTab] = useState('search');
  const [hasShownToast, setHasShownToast] = useState(false);
  const { showToast } = useToast();

  // Show demo notification on every page load (but only once per load)
  useEffect(() => {
    if (!hasShownToast) {
      // Show the notification after a short delay to ensure the app has loaded
      const timer = setTimeout(() => {
        showToast(
          'Demo Mode: No user management enabled. All users share the same case files and documents.',
          'info',
          10000 // Show for 10 seconds
        );
        setHasShownToast(true);
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [showToast, hasShownToast]);

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
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
