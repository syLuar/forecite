import React, { createContext, useContext, useState, ReactNode } from 'react';
import Toast from '../components/shared/Toast';

interface ToastData {
  id: string;
  message: string;
  type?: 'info' | 'warning' | 'success' | 'error';
  duration?: number;
}

interface ToastContextType {
  showToast: (message: string, type?: 'info' | 'warning' | 'success' | 'error', duration?: number) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};

interface ToastProviderProps {
  children: ReactNode;
}

export const ToastProvider: React.FC<ToastProviderProps> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastData[]>([]);

  const showToast = (message: string, type: 'info' | 'warning' | 'success' | 'error' = 'info', duration: number = 8000) => {
    const id = Date.now().toString();
    const newToast: ToastData = {
      id,
      message,
      type,
      duration
    };
    
    setToasts(prev => [...prev, newToast]);
  };

  const removeToast = (id: string) => {
    setToasts(prev => prev.filter(toast => toast.id !== id));
  };

  return (
    <ToastContext.Provider value={{ showToast, removeToast }}>
      {children}
      
      {/* Render toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((toast, index) => (
          <div 
            key={toast.id}
            style={{ transform: `translateY(${index * 12}px)` }}
            className="transition-transform duration-200"
          >
            <Toast
              message={toast.message}
              type={toast.type}
              duration={toast.duration}
              onClose={() => removeToast(toast.id)}
            />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};
