import React from 'react';
import { Filter, Grid, Scale, FileText } from 'lucide-react';
import { legalCategories, LegalDocument } from '../data/mockData';
import { SearchFilters as SearchFiltersType } from '../utils/searchUtils';

interface SearchFiltersProps {
  filters: SearchFiltersType;
  onFiltersChange: (filters: SearchFiltersType) => void;
  resultCount: number;
  currentResults: LegalDocument[];
}

const SearchFilters: React.FC<SearchFiltersProps> = ({ 
  filters, 
  onFiltersChange, 
  resultCount,
  currentResults
}) => {
  const getCategoryIcon = (categoryId: string) => {
    switch (categoryId) {
      case 'all': return <Grid className="h-4 w-4" />;
      case 'precedent': return <Scale className="h-4 w-4" />;
      case 'laws': return <FileText className="h-4 w-4" />;
      default: return <Grid className="h-4 w-4" />;
    }
  };

  const getCategoryCount = (categoryId: string) => {
    if (categoryId === 'all') {
      return currentResults.length;
    }
    return currentResults.filter(doc => doc.category === categoryId).length;
  };

  const handleCategoryChange = (categoryId: string) => {
    onFiltersChange({
      ...filters,
      category: categoryId
    });
  };

  return (
    <div className="bg-gray-100 rounded-lg shadow-md p-4 mb-6 border border-gray-200">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <Filter className="h-5 w-5 text-gray-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Filter Results</h3>
        </div>
        <span className="text-sm text-gray-500">
          {resultCount} {resultCount === 1 ? 'result' : 'results'}
        </span>
      </div>

      <div className="flex flex-wrap gap-3">
        {legalCategories.map((category) => {
          const actualCount = getCategoryCount(category.id);
          return (
            <button
              key={category.id}
              onClick={() => handleCategoryChange(category.id)}
              className={`flex items-center px-4 py-2 rounded-lg border text-sm font-medium transition-colors duration-200 ${
                filters.category === category.id
                  ? 'bg-primary text-white border-primary shadow-sm'
                  : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50 hover:border-gray-300'
              }`}
            >
              {getCategoryIcon(category.id)}
              <span className="ml-2">{category.label}</span>
              <span className="ml-2 text-xs opacity-75">
                ({actualCount})
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default SearchFilters; 