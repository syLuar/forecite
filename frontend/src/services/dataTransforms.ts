import { RetrievedDocument } from '../types/api';
import { LegalDocument } from '../data/mockSearchData';

export function transformRetrievedDocToLegalDoc(doc: RetrievedDocument): LegalDocument & { originalApiData?: RetrievedDocument } {
  var title = extractTitleFromCitation(doc.document_citation || doc.document_source);
  return {
    id: doc.chunk_id || doc.document_citation || Math.random().toString(),
    title: title,
    citation: doc.parties?.join(' vs ') || title,
    parties: doc.parties || [],
    category: doc.document_type === 'Case' ? 'precedent' : 'laws',
    court: doc.courts?.[0],
    jurisdiction: doc.jurisdiction || 'Unknown',
    date: doc.document_year ? `${doc.document_year}-01-01` : '',
    summary: doc.summary || doc.text.substring(0, 200) + '...',
    keyTerms: [...(doc.concepts || []), ...(doc.legal_tests || [])],
    relevanceScore: doc.score,
    originalApiData: doc, // Preserve the original API data for case file operations
  };
}

function extractTitleFromCitation(citation: string): string {
  // Extract case name from citation format like "Smith v. Jones, 123 F.3d 456 (2000)"
  const match = citation.match(/^([^,]+)/);
  return match ? match[1].trim() : citation;
}
