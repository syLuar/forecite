// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// HTTP client with error handling
class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Research methods
  async searchDocuments(query: any): Promise<any> {
    return this.request('/api/v1/research/query', {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  async analyzePrecedent(citation: string): Promise<any> {
    return this.request(`/api/v1/research/precedent-analysis/${encodeURIComponent(citation)}`);
  }

  async getCitationNetwork(citation: string, direction: string = 'both'): Promise<any> {
    return this.request(`/api/v1/research/citation-network/${encodeURIComponent(citation)}?direction=${direction}`);
  }

  // Drafting methods
  async draftArgument(request: any): Promise<any> {
    return this.request('/api/v1/generation/draft-argument', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Case File Management methods
  async createCaseFile(request: any): Promise<{ case_file_id: number }> {
    return this.request('/api/v1/case-files', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async listCaseFiles(): Promise<any[]> {
    return this.request('/api/v1/case-files');
  }

  async getCaseFile(caseFileId: number): Promise<any> {
    return this.request(`/api/v1/case-files/${caseFileId}`);
  }

  async updateCaseFile(caseFileId: number, request: any): Promise<{ success: boolean }> {
    return this.request(`/api/v1/case-files/${caseFileId}`, {
      method: 'PUT',
      body: JSON.stringify(request),
    });
  }

  async deleteCaseFile(caseFileId: number): Promise<{ success: boolean }> {
    return this.request(`/api/v1/case-files/${caseFileId}`, {
      method: 'DELETE',
    });
  }

  async addDocumentToCaseFile(caseFileId: number, request: any): Promise<{ success: boolean }> {
    return this.request(`/api/v1/case-files/${caseFileId}/documents`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async removeDocumentFromCaseFile(caseFileId: number, documentId: string): Promise<{ success: boolean }> {
    return this.request(`/api/v1/case-files/${caseFileId}/documents/${encodeURIComponent(documentId)}`, {
      method: 'DELETE',
    });
  }

  async getDocumentFromCaseFile(caseFileId: number, documentId: string): Promise<any> {
    return this.request(`/api/v1/case-files/${caseFileId}/documents/${encodeURIComponent(documentId)}`);
  }

  // Argument Draft Management methods
  async saveDraft(caseFileId: number, draftResponse: any, title?: string): Promise<{ draft_id: number }> {
    return this.request('/api/v1/drafts/save', {
      method: 'POST',
      body: JSON.stringify({ 
        case_file_id: caseFileId, 
        title: title,
        draft_response: draftResponse 
      }),
    });
  }

  async listDraftsForCaseFile(caseFileId: number): Promise<any[]> {
    return this.request(`/api/v1/case-files/${caseFileId}/drafts`);
  }

  async getDraft(draftId: number): Promise<any> {
    return this.request(`/api/v1/drafts/${draftId}`);
  }

  async deleteDraft(draftId: number): Promise<{ success: boolean }> {
    return this.request(`/api/v1/drafts/${draftId}`, {
      method: 'DELETE',
    });
  }

  // Health check
  async healthCheck(): Promise<any> {
    return this.request('/health');
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
