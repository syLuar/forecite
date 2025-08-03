import React, { useState } from 'react';
import { Calendar, MapPin, Scale, FileText, HelpCircle, Plus, Check, Undo2, AlertCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { LegalDocument } from '../../data/mockSearchData';
import { highlightMatches } from '../../utils/searchUtils';
import Modal from '../shared/Modal';
import SelectCaseModal from '../shared/SelectCaseModal';

interface SearchResultProps {
  document: LegalDocument;
  searchQuery: string;
}

const SearchResult: React.FC<SearchResultProps> = ({ document, searchQuery }) => {
  const [showModal, setShowModal] = useState(false);
  const [showSelectCaseModal, setShowSelectCaseModal] = useState(false);
  const [explanation, setExplanation] = useState('');
  const [isAdded, setIsAdded] = useState(false);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);
  const [addedToCase, setAddedToCase] = useState<any | null>(null);

  const hasRelevance = document.relevanceScore !== undefined && document.relevanceScore > 0;

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'precedent': return <Scale className="h-4 w-4" />;
      case 'laws': return <FileText className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getCategoryLabel = (category: string) => {
    switch (category) {
      case 'precedent': return 'Precedent Case';
      case 'laws': return 'Law & Regulation';
      default: return 'Document';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'precedent': return 'bg-blue-100 text-blue-800';
      case 'laws': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const generateRelevanceExplanation = () => {
    if (!hasRelevance) {
      return `This document was included in results because it's part of our comprehensive legal database, though it doesn't contain direct matches for your search terms "${searchQuery}". It may still provide useful context or background information for your research.`;
    }
    
    const explanations = [
      `This ${document.category === 'precedent' ? 'case' : 'document'} is highly relevant because your search terms "${searchQuery}" directly match key legal concepts discussed in ${document.title}. The document addresses similar legal principles and contains matching terminology found in your query.`,
      
      `The relevance score reflects strong keyword matches between your search "${searchQuery}" and this document's content. Key terms from your query appear in the title, summary, and legal citations, indicating substantial topical alignment.`,
      
      `Your search query "${searchQuery}" aligns well with the core legal themes in ${document.title}. The document contains multiple references to concepts you're researching, making it a valuable resource for your legal analysis.`,
      
      `This document scores highly because it contains exact matches for your search terms "${searchQuery}" in critical sections including the case summary, key legal principles, and cited precedents. The overlapping terminology suggests strong conceptual relevance.`
    ];
    
    return explanations[Math.floor(Math.random() * explanations.length)];
  };

  const handleFindOutWhy = () => {
    setExplanation(generateRelevanceExplanation());
    setShowModal(true);
  };

  const handleAddReference = () => {
    if (isAdded) {
      // Reset added state - in real implementation, this might remove from the last selected case file
      setIsAdded(false);
      setAddedToCase(null);
      
      if (timeoutId) {
        clearTimeout(timeoutId);
        setTimeoutId(null);
      }
    } else {
      // Open the modal to select case file
      setShowSelectCaseModal(true);
    }
  };

  const handleAddToCase = (caseItem: any, legalDocument: LegalDocument) => {
    setIsAdded(true);
    setAddedToCase(caseItem);
    
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    
    const id = setTimeout(() => {
      setAddedToCase(null); // Remove the case reference but keep as added
    }, 10000);
    
    setTimeoutId(id);
  };

  const handleUndo = (e: React.MouseEvent) => {
    e.stopPropagation();
    
    // Reset the added state
    setIsAdded(false);
    setAddedToCase(null);

    if (timeoutId) {
      clearTimeout(timeoutId);
      setTimeoutId(null);
    }
  };

  return (
    <>
      <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 p-6 mb-4 border border-gray-200">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-3">
            <span className={`inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium ${getCategoryColor(document.category)}`}>
              {getCategoryIcon(document.category)}
              <span className="ml-1.5">{getCategoryLabel(document.category)}</span>
            </span>
            <div className="flex items-center space-x-3">
              {document.relevanceScore !== undefined && (
                <span className={`inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium ${hasRelevance ? 'bg-primary text-white' : 'bg-red-100 text-red-800'}`}>
                  {hasRelevance ? (
                    <>Relevance: {Math.round(document.relevanceScore)}%</>
                  ) : (
                    <>
                      <AlertCircle className="h-4 w-4 mr-1" />
                      General Result
                    </>
                  )}
                </span>
              )}
              <button
                onClick={handleFindOutWhy}
                className="text-sm text-primary hover:text-primary-700 underline flex items-center font-medium"
              >
                <HelpCircle className="h-4 w-4 mr-1" />
                Find out why
              </button>
            </div>
          </div>
        </div>

        <h3 
          className="text-lg font-semibold text-gray-900 mb-2"
          dangerouslySetInnerHTML={{ 
            __html: highlightMatches(document.title, searchQuery) 
          }}
        />

        <p className="text-sm text-gray-600 mb-2 font-mono">{document.citation}</p>

        {document.court && (
          <p className="text-sm text-gray-700 mb-2">
            <strong>Court:</strong> {document.court}
          </p>
        )}

        <div className="text-gray-700 mb-4 leading-relaxed prose prose-sm max-w-none">
          <ReactMarkdown>{document.summary}</ReactMarkdown>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          {document.keyTerms.slice(0, 5).map((term, index) => (
            <span 
              key={index}
              className="inline-block px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded-md"
              dangerouslySetInnerHTML={{ 
                __html: highlightMatches(term, searchQuery) 
              }}
            />
          ))}
          {document.keyTerms.length > 5 && (
            <span className="text-xs text-gray-500">
              +{document.keyTerms.length - 5} more
            </span>
          )}
        </div>

        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <MapPin className="h-4 w-4 mr-1" />
              {document.jurisdiction}
            </div>
            <div className="flex items-center">
              <Calendar className="h-4 w-4 mr-1" />
              {formatDate(document.date)}
            </div>
          </div>
          
          <div className="flex items-center">
            {isAdded ? (
              <>
                <span className="flex items-center text-green-600 font-medium mr-3">
                  <Check className="h-4 w-4 mr-1" />
                  Added to Case File
                </span>
                <button 
                  onClick={handleUndo}
                  className="flex items-center text-gray-500 hover:text-gray-700 font-medium"
                >
                  <Undo2 className="h-4 w-4 mr-1" />
                  Remove
                </button>
              </>
            ) : (
              <button 
                onClick={handleAddReference}
                className="flex items-center text-primary hover:text-primary-700 font-medium"
              >
                <Plus className="h-4 w-4 mr-1" />
                Add to Case File
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Relevance Explanation Modal */}
      <Modal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        title={document.title}
      >
        <div className="space-y-4">
          <div className="text-center">
            <span className={`inline-flex items-center px-4 py-2 rounded-full text-base font-medium ${hasRelevance ? 'bg-primary text-white' : 'bg-red-100 text-red-800'}`}>
              {hasRelevance ? (
                <>Relevance Score: {Math.round(document.relevanceScore || 0)}%</>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5 mr-2" />
                  General Result
                </>
              )}
            </span>
          </div>
          
          <div className={`border rounded-lg p-4 ${hasRelevance ? 'bg-blue-50 border-blue-200' : 'bg-red-50 border-red-200'}`}>
            <p className={`text-sm leading-relaxed ${hasRelevance ? 'text-blue-800' : 'text-red-800'}`}>
              {explanation}
            </p>
          </div>
        </div>
      </Modal>

      {/* Select Case Modal */}
      <SelectCaseModal
        isOpen={showSelectCaseModal}
        onClose={() => setShowSelectCaseModal(false)}
        document={document}
        onAddToCase={handleAddToCase}
      />
    </>
  );
};

export default SearchResult; 