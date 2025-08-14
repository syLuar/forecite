import React, { useState, useEffect, useCallback } from 'react';
import { Search, Loader2, Send, X, Scale } from 'lucide-react';
import { LegalDocument } from '../../data/mockSearchData';
import { SearchFilters as SearchFiltersType } from '../../utils/searchUtils';
import { apiClient } from '../../services/api';
import { transformRetrievedDocToLegalDoc } from '../../services/dataTransforms';
import { ResearchQueryRequest } from '../../types/api';
import SearchResult from '../search/SearchResult';
import SearchFilters from '../search/SearchFilters';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  caseFileId?: number;
  onDocumentAdded?: () => void;
}

const SearchModal: React.FC<SearchModalProps> = ({ isOpen, onClose, caseFileId, onDocumentAdded }) => {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFiltersType>({ category: 'all' });
  const [results, setResults] = useState<LegalDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      // Reset state when opening
      setQuery('');
      setFilters({ category: 'all' });
      setResults([]);
      setIsLoading(false);
      setHasSearched(false);
      setError(null);
    }
  }, [isOpen]);

  // Escape key handler
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  const handleSearch = useCallback(async (searchQuery: string = query, showLoading: boolean = true) => {
    if (!searchQuery.trim()) {
      // If empty query, just clear results
      setResults([]);
      setHasSearched(true);
      return;
    }
    
    if (showLoading) {
      setIsLoading(true);
    }
    setError(null);
    setHasSearched(true);

    try {
      const request: ResearchQueryRequest = {
        query_text: searchQuery,
        max_results: 15,
      };

      // Add filters
      if (filters.jurisdiction) {
        request.jurisdiction = filters.jurisdiction;
      }
      if (filters.category && filters.category !== 'all') {
        request.document_type = filters.category === 'precedent' ? 'Case' : 'Statute';
      }

      const response = await apiClient.searchDocuments(request);
      
      // Transform API response to UI format
      const transformedResults = response.retrieved_docs.map(transformRetrievedDocToLegalDoc);
      setResults(transformedResults);
      
    } catch (err) {
      console.error('Search failed:', err);
      setError('Search failed. Please try again.');
      setResults([]);
    } finally {
      if (showLoading) {
        setIsLoading(false);
      }
    }
  }, [filters, query]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Always search, even with empty query to show all documents
    handleSearch(query, true); // Show loading for manual search
  };

  const handleClearSearch = () => {
    setQuery('');
    setResults([]);
    setHasSearched(false);
  };

  const handleFiltersChange = (newFilters: SearchFiltersType) => {
    setFilters(newFilters);
  };

  // Re-search when filters change (only if already searched) - NO LOADING
  useEffect(() => {
    if (hasSearched) {
      // Create a local version to avoid dependency issues
      const searchWithFilters = async () => {
        setError(null);

        try {
          const request: ResearchQueryRequest = {
            query_text: query,
            max_results: 15,
          };

          // Add filters
          if (filters.jurisdiction) {
            request.jurisdiction = filters.jurisdiction;
          }
          if (filters.category && filters.category !== 'all') {
            request.document_type = filters.category === 'precedent' ? 'Case' : 'Statute';
          }

          const response = await apiClient.searchDocuments(request);
          
          // Transform API response to UI format
          const transformedResults = response.retrieved_docs.map(transformRetrievedDocToLegalDoc);
          setResults(transformedResults);
          
        } catch (err) {
          console.error('Search failed:', err);
          setError('Search failed. Please try again.');
          setResults([]);
        }
      };

      searchWithFilters();
    }
  }, [filters.category, filters.jurisdiction]); // Removed hasSearched and query dependencies

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="flex min-h-full items-start justify-center p-4 pt-8">
        <div className="relative bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] overflow-hidden">
          {/* Modal Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Search Documents</h2>
              <p className="text-gray-600">Find documents to add to your case file</p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>
          </div>

          {/* Modal Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            {/* Search Form */}
            <div className="bg-gray-100 rounded-lg shadow-md p-6 mb-6 border border-gray-200">
              <form onSubmit={handleSubmit}>
                <div className="flex gap-3">
                  <div className="relative flex-1">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Search className="h-5 w-5 text-gray-400" />
                    </div>
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search legal documents, cases, statutes, regulations..."
                      className="block w-full pl-10 pr-10 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-gray-900 placeholder-gray-500"
                    />
                    {query && (
                      <button
                        type="button"
                        onClick={handleClearSearch}
                        className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    )}
                  </div>
                  
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="flex items-center justify-center px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 whitespace-nowrap"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Searching...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4 mr-2" />
                        Submit
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* Show filters only after search has been submitted */}
            {hasSearched && (
              <SearchFilters 
                filters={filters}
                onFiltersChange={handleFiltersChange}
                resultCount={results.length}
                currentResults={results}
              />
            )}

            {/* Loading State - Big floating law icon */}
            {isLoading && hasSearched && (
              <div className="flex flex-col items-center justify-center py-20">
                <Scale className="h-24 w-24 text-primary animate-bounce mb-4" />
                <p className="text-lg text-gray-600 font-medium">Searching legal database...</p>
              </div>
            )}

            {/* Error State */}
            {error && hasSearched && !isLoading && (
              <div className="text-center py-12">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md mx-auto">
                  <h3 className="text-lg font-medium text-red-800 mb-2">Search Error</h3>
                  <p className="text-red-600 mb-4">{error}</p>
                  <button 
                    onClick={() => handleSearch(query, true)}
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            )}

            {/* Search Results - show when not loading */}
            {hasSearched && !isLoading && !error && (
              <div className="space-y-4">
                {results.length > 0 ? (
                  <>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xl font-semibold text-gray-900">
                        Search Results
                      </h3>
                      <span className="text-sm text-gray-500">
                        Showing {results.length} {results.length === 1 ? 'result' : 'results'}
                        {query && ` for "${query}"`}
                      </span>
                    </div>
                    
                    {results.map((document) => (
                      <SearchResult
                        key={document.id}
                        document={document}
                        searchQuery={query}
                        onDocumentAdded={onDocumentAdded}
                      />
                    ))}
                  </>
                ) : (
                  <div className="text-center py-12">
                    <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No relevant documents!</h3>
                    <p className="text-gray-600 mb-4">
                      Try searching more general terms
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Pre-search state - show when no search has been performed and not loading */}
            {!hasSearched && !isLoading && (
              <div className="text-center py-16">
                <Search className="h-16 w-16 text-gray-300 mx-auto mb-6" />
                <h3 className="text-xl font-medium text-gray-900 mb-3">
                  Search through precedent cases, laws, and regulations
                </h3>
                <p className="text-gray-600 max-w-2xl mx-auto leading-relaxed">
                  Enter your relevant keywords to find helpful material to add to your case file
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchModal;
