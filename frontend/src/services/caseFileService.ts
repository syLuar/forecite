import { CaseFile, CaseFileDocument } from '../types/api';
import { LegalDocument } from '../data/mockSearchData';

class CaseFileService {
  private caseFile: CaseFile = {
    documents: [],
    total_documents: 0,
    created_at: new Date().toISOString(),
  };

  addDocument(document: LegalDocument): void {
    const caseFileDoc: CaseFileDocument = {
      document_id: document.id,
      citation: document.citation,
      title: document.title,
      year: parseInt(document.date.split('-')[0]) || new Date().getFullYear(),
      jurisdiction: document.jurisdiction,
      relevance_score_percent: document.relevanceScore || 0,
      key_holdings: [], // Could be extracted from summary
      selected_chunks: [],
    };

    // Avoid duplicates
    if (!this.caseFile.documents.find(d => d.document_id === document.id)) {
      this.caseFile.documents.push(caseFileDoc);
      this.caseFile.total_documents = this.caseFile.documents.length;
      this.caseFile.last_modified = new Date().toISOString();
    }
  }

  removeDocument(documentId: string): void {
    this.caseFile.documents = this.caseFile.documents.filter(
      d => d.document_id !== documentId
    );
    this.caseFile.total_documents = this.caseFile.documents.length;
    this.caseFile.last_modified = new Date().toISOString();
  }

  getCaseFile(): CaseFile {
    return { ...this.caseFile };
  }

  clearCaseFile(): void {
    this.caseFile = {
      documents: [],
      total_documents: 0,
      created_at: new Date().toISOString(),
    };
  }

  getDocumentCount(): number {
    return this.caseFile.total_documents;
  }

  hasDocument(documentId: string): boolean {
    return this.caseFile.documents.some(d => d.document_id === documentId);
  }
}

export const caseFileService = new CaseFileService();
