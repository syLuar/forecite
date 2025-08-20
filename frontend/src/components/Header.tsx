import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-primary text-white shadow-lg">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-center space-x-3">
          <img 
            src="/logo.png" 
            alt="Forecite Logo" 
            className="h-8 w-8 object-contain"
          />
          <h1 className="text-2xl md:text-3xl font-bold tracking-wide">
            Forecite
          </h1>
        </div>
      </div>
    </header>
  );
};

export default Header; 