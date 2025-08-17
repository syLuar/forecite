// Research types
export interface ResearchQueryRequest {
  query_text: string;
  jurisdiction?: string;
  document_type?: string;
  date_range?: {
    from: number;
    to: number;
  };
  max_results?: number;
}

export interface RetrievedDocument {
  chunk_id?: string;
  text: string;
  parties: string[];
  summary?: string;
  document_source: string;
  document_citation?: string;
  document_year?: number;
  jurisdiction?: string;
  document_type?: string;
  court_level?: string;
  score?: number;
  statutes?: string[];
  courts?: string[];
  cases?: string[];
  concepts?: string[];
  judges?: string[];
  holdings?: string[];
  facts?: string[];
  legal_tests?: string[];
}

export interface ResearchQueryResponse {
  retrieved_docs: RetrievedDocument[];
  total_results: number;
  search_quality_score?: number;
  refinement_count: number;
  assessment_reason?: string;
  execution_time?: number;
  search_history?: any[];
}

// Drafting types
export interface CaseFileDocument {
  document_id: string;
  citation: string;
  title: string;
  year: number;
  jurisdiction: string;
  relevance_score_percent: number;
  key_holdings: string[];
  selected_chunks: any[];
  user_notes?: string;
}

export interface CaseFile {
  documents: CaseFileDocument[];
  total_documents: number;
  created_at?: string;
  last_modified?: string;
}

export interface ArgumentDraftRequest {
  case_file_id: number;
  legal_question?: string;
  additional_drafting_instructions?: string;
  argument_preferences?: any;
}

export interface LegalArgument {
  argument: string;
  supporting_authority: string;
  factual_basis: string;
  strength_assessment?: number;
}

export interface ArgumentStrategy {
  main_thesis: string;
  argument_type: string;
  primary_precedents: string[];
  legal_framework: string;
  key_arguments: LegalArgument[];
  anticipated_counterarguments: string[];
  counterargument_responses: string[];
  strength_assessment: number;
  risk_factors: string[];
  strategy_rationale: string;
}

export interface ArgumentDraftResponse {
  strategy: ArgumentStrategy;
  drafted_argument: string;
  argument_structure: any;
  citations_used: string[];
  argument_strength: number;
  precedent_coverage: number;
  logical_coherence: number;
  total_critique_cycles: number;
  revision_history?: any[];
  execution_time?: number;
}

// Case File Management Types
export interface CreateCaseFileRequest {
  title: string;
  description?: string;
  user_facts?: string;  // Only case facts - no legal question or instructions
  party_represented?: string;  // Which party the user represents
}

export interface UpdateCaseFileRequest {
  title?: string;
  description?: string;
  user_facts?: string;  // Only case facts - no legal question or instructions
  party_represented?: string;  // Which party the user represents
}

export interface CaseFileResponse {
  id: number;
  title: string;
  description?: string;
  user_facts?: string;  // Only case facts
  party_represented?: string;  // Which party the user represents
  created_at: string;
  updated_at?: string;
  documents: any[];
  notes: CaseFileNote[];
  total_documents: number;
}

export interface CaseFileNote {
  id: number;
  content: string;
  author_type: 'user' | 'ai';
  author_name?: string;
  note_type?: string;
  tags: string[];
  created_at: string;
  updated_at?: string;
}

export interface AddCaseFileNoteRequest {
  content: string;
  author_type: 'user' | 'ai';
  author_name?: string;
  note_type?: string;
  tags?: string[];
}

export interface UpdateCaseFileNoteRequest {
  content: string;
  note_type?: string;
  tags?: string[];
}

export interface CaseFileListItem {
  id: number;
  title: string;
  description?: string;
  party_represented?: string;  // Which party the user represents
  created_at: string;
  updated_at?: string;
  document_count: number;
  draft_count: number;
}

export interface AddDocumentToCaseFileRequest {
  document_id: string;
  citation: string;
  parties?: string[];
  title: string;
  year?: number;
  jurisdiction?: string;
  relevance_score_percent?: number;
  key_holdings?: string[];
  selected_chunks?: any[];
  user_notes?: string;
}

export interface SaveDraftRequest {
  case_file_id: number;
  title?: string;
}

export interface ArgumentDraftListItem {
  id: number;
  title: string;
  created_at: string;
  argument_strength?: number;
  precedent_coverage?: number;
  logical_coherence?: number;
}

export interface SavedArgumentDraft {
  id: number;
  case_file_id: number;
  title: string;
  drafted_argument: string;
  strategy?: any;
  argument_structure?: any;
  citations_used?: string[];
  argument_strength?: number;
  precedent_coverage?: number;
  logical_coherence?: number;
  total_critique_cycles?: number;
  execution_time?: number;
  revision_history?: any[];
  created_at: string;
}

// Utility types
export interface HealthResponse {
  status: string;
  timestamp: string;
  version?: string;
}

export interface PrecedentAnalysis {
  case_citation: string;
  total_citations: number;
  precedent_strength: number;
  avg_authority_weight?: number;
  citing_jurisdictions: string[];
  most_recent_citation?: number;
  earliest_citation?: number;
}
