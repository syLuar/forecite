import { LegalDocument } from '../data/mockData';

export interface SearchFilters {
  category: string;
  jurisdiction?: string;
  dateRange?: {
    start: string;
    end: string;
  };
}

export function tokenizeQuery(query: string): string[] {
  return query
    .toLowerCase()
    .replace(/[^\w\s]/g, '') // Remove punctuation
    .split(/\s+/)
    .filter(token => token.length > 2); // Remove very short words
}

export function calculateRelevanceScore(document: LegalDocument, tokens: string[]): number {
  if (tokens.length === 0) {
    return 50; // Default relevance when no search terms
  }

  let score = 0;
  const searchableText = [
    document.title,
    document.summary,
    document.citation,
    ...document.keyTerms,
    document.court || '',
    document.jurisdiction
  ].join(' ').toLowerCase();

  tokens.forEach(token => {
    // Title matches get highest score
    if (document.title.toLowerCase().includes(token)) {
      score += 10;
    }
    
    // Key terms matches get high score
    if (document.keyTerms.some(term => term.toLowerCase().includes(token))) {
      score += 8;
    }
    
    // Summary matches get medium score
    if (document.summary.toLowerCase().includes(token)) {
      score += 5;
    }
    
    // Citation matches get medium score
    if (document.citation.toLowerCase().includes(token)) {
      score += 5;
    }
    
    // Any other matches get low score
    if (searchableText.includes(token)) {
      score += 2;
    }
    
    // Exact phrase bonus
    if (searchableText.includes(token)) {
      const regex = new RegExp(`\\b${token}\\b`, 'gi');
      const matches = searchableText.match(regex);
      if (matches) {
        score += matches.length;
      }
    }
  });

  return Math.min(score, 100); // Cap at 100%
}

export function searchDocuments(
  documents: LegalDocument[],
  query: string,
  filters: SearchFilters
): LegalDocument[] {
  const tokens = tokenizeQuery(query);
  
  // Calculate relevance scores for ALL documents
  const scoredDocuments = documents.map(doc => ({
    ...doc,
    relevanceScore: calculateRelevanceScore(doc, tokens)
  }));

  // Apply filters but return ALL documents
  const filteredDocuments = applyFilters(scoredDocuments, filters);

  // Sort by relevance score (highest first) when there's a query
  if (query.trim()) {
    return filteredDocuments.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
  }
  
  // When no query, just return in original order with default relevance
  return filteredDocuments;
}

function applyFilters(documents: LegalDocument[], filters: SearchFilters): LegalDocument[] {
  let filtered = [...documents];

  // Category filter
  if (filters.category && filters.category !== 'all') {
    filtered = filtered.filter(doc => doc.category === filters.category);
  }

  // Jurisdiction filter (keeping this for future use)
  if (filters.jurisdiction && filters.jurisdiction !== 'all') {
    filtered = filtered.filter(doc => 
      doc.jurisdiction.toLowerCase() === filters.jurisdiction!.toLowerCase()
    );
  }

  // Date range filter (keeping this for future use)
  if (filters.dateRange && filters.dateRange.start && filters.dateRange.end) {
    filtered = filtered.filter(doc => {
      const docDate = new Date(doc.date);
      const startDate = new Date(filters.dateRange!.start);
      const endDate = new Date(filters.dateRange!.end);
      return docDate >= startDate && docDate <= endDate;
    });
  }

  return filtered;
}

export function highlightMatches(text: string, query: string): string {
  if (!query.trim()) return text;
  
  const tokens = tokenizeQuery(query);
  let highlightedText = text;
  
  tokens.forEach(token => {
    const regex = new RegExp(`(${token})`, 'gi');
    highlightedText = highlightedText.replace(regex, '<mark class="bg-yellow-200 px-1 rounded">$1</mark>');
  });
  
  return highlightedText;
} 