import React, { useState, useEffect } from 'react';
import { X, Info } from 'lucide-react';

interface ToastProps {
  message: string;
  type?: 'info' | 'warning' | 'success' | 'error';
  duration?: number;
  onClose?: () => void;
  className?: string;
}

const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  duration = 8000, // 8 seconds default
  onClose,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    // Start entrance animation
    setIsAnimating(true);
    
    if (duration > 0) {
      const timer = setTimeout(() => {
        handleClose();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [duration]);

  const handleClose = () => {
    setIsAnimating(false);
    setTimeout(() => {
      setIsVisible(false);
      onClose?.();
    }, 300); // Match the exit animation duration
  };

  if (!isVisible) return null;

  const getToastStyles = () => {
    const baseStyles = "flex items-start gap-3 p-4 rounded-lg shadow-lg border backdrop-blur-sm";
    
    switch (type) {
      case 'info':
        return `${baseStyles} bg-blue-50/95 border-blue-200 text-blue-800`;
      case 'warning':
        return `${baseStyles} bg-amber-50/95 border-amber-200 text-amber-800`;
      case 'success':
        return `${baseStyles} bg-green-50/95 border-green-200 text-green-800`;
      case 'error':
        return `${baseStyles} bg-red-50/95 border-red-200 text-red-800`;
      default:
        return `${baseStyles} bg-blue-50/95 border-blue-200 text-blue-800`;
    }
  };

  const getIconColor = () => {
    switch (type) {
      case 'info':
        return 'text-blue-500';
      case 'warning':
        return 'text-amber-500';
      case 'success':
        return 'text-green-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-blue-500';
    }
  };

  return (
    <div 
      className={`
        fixed top-4 right-4 z-50 max-w-lg transition-all duration-300 ease-in-out
        ${isAnimating ? 'translate-x-0 opacity-100' : 'translate-x-full opacity-0'}
        ${className}
      `}
    >
      <div className={getToastStyles()}>
        <Info className={`w-5 h-5 mt-0.5 flex-shrink-0 ${getIconColor()}`} />
        
        <div className="flex-1 text-sm font-medium">
          {message}
        </div>
        
        <button
          onClick={handleClose}
          className="flex-shrink-0 ml-2 p-0.5 rounded-full hover:bg-black/10 transition-colors"
          aria-label="Close notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

export default Toast;
