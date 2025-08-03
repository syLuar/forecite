import React, { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="text-center py-12">
          <h2 className="text-lg font-medium text-gray-900 mb-2">
            Something went wrong
          </h2>
          <p className="text-gray-600 mb-4">
            Please try refreshing the page or contact support.
          </p>
          <button 
            onClick={() => this.setState({ hasError: false })}
            className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
