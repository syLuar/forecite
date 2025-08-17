import { RetrievedDocument } from '../types/api';

export interface LegalDocument {
  id: string;
  title: string;
  category: 'precedent' | 'laws';
  court?: string;
  jurisdiction: string;
  date: string;
  citation: string;
  parties: string[];
  summary: string;
  keyTerms: string[];
  relevanceScore?: number;
  originalApiData?: RetrievedDocument; // Store original API data for case file operations
}

export const legalCategories = [
  { id: 'all', label: 'All Categories', count: 0 },
  { id: 'precedent', label: 'Precedent Cases', count: 0 },
  { id: 'laws', label: 'Laws & Regulations', count: 0 },
];