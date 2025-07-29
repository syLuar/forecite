import React, { useState, useEffect } from 'react';
import { Search, Loader2, Send, X, Scale } from 'lucide-react';
import { mockLegalDocuments, LegalDocument } from '../data/mockData';
import { searchDocuments, SearchFilters as SearchFiltersType } from '../utils/searchUtils';
import SearchResult from './SearchResult';
import SearchFilters from './SearchFilters';

const SearchContent: React.FC = () => {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFiltersType>({ category: 'all' });
  const [results, setResults] = useState<LegalDocument[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (searchQuery: string = query, showLoading: boolean = true) => {
    if (showLoading) {
      setIsLoading(true);
    }
    setHasSearched(true);

    // Simulate 5s API delay for realistic feel only when showing loading
    if (showLoading) {
      await new Promise(resolve => setTimeout(resolve, 5000));
    }

    const searchResults = searchDocuments(mockLegalDocuments, searchQuery, filters);
    setResults(searchResults);
    
    if (showLoading) {
      setIsLoading(false);
    }
  };

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
      handleSearch(query, false); // No loading for filter changes
    }
  }, [filters]);

  return (
    <div className="flex-1 p-4 md:p-6 pb-24 md:pb-8">
      <div className="max-w-6xl mx-auto">
        {/* Search Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Legal Research</h2>
          <p className="text-gray-600">
            Tap into our rich and real-time legal database
          </p>
        </div>

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

        {/* Search Results - show when not loading */}
        {hasSearched && !isLoading && (
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
              Enter your relevant keywords to find helpful material
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchContent; 