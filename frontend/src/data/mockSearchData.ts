import { RetrievedDocument } from '../types/api';

export interface LegalDocument {
  id: string;
  title: string;
  category: 'precedent' | 'laws';
  court?: string;
  jurisdiction: string;
  date: string;
  citation: string;
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

export const mockLegalDocuments: LegalDocument[] = [
  // Precedent Cases
  {
    id: '1',
    title: 'Brown v. Board of Education',
    category: 'precedent',
    court: 'Supreme Court of the United States',
    jurisdiction: 'Federal',
    date: '1954-05-17',
    citation: '347 U.S. 483 (1954)',
    summary: 'Landmark case that declared state laws establishing separate public schools for black and white students to be unconstitutional.',
    keyTerms: ['education', 'segregation', 'equal protection', 'fourteenth amendment', 'civil rights']
  },
  {
    id: '2',
    title: 'Miranda v. Arizona',
    category: 'precedent',
    court: 'Supreme Court of the United States',
    jurisdiction: 'Federal',
    date: '1966-06-13',
    citation: '384 U.S. 436 (1966)',
    summary: 'Established the Miranda rights, requiring law enforcement to inform suspects of their rights before interrogation.',
    keyTerms: ['criminal law', 'fifth amendment', 'self-incrimination', 'miranda rights', 'interrogation']
  },
  {
    id: '3',
    title: 'Roe v. Wade',
    category: 'precedent',
    court: 'Supreme Court of the United States',
    jurisdiction: 'Federal',
    date: '1973-01-22',
    citation: '410 U.S. 113 (1973)',
    summary: 'Established constitutional right to abortion under the Due Process Clause of the Fourteenth Amendment.',
    keyTerms: ['abortion', 'privacy rights', 'due process', 'fourteenth amendment', 'reproductive rights']
  },
  {
    id: '4',
    title: 'Marbury v. Madison',
    category: 'precedent',
    court: 'Supreme Court of the United States',
    jurisdiction: 'Federal',
    date: '1803-02-24',
    citation: '5 U.S. 137 (1803)',
    summary: 'Established the principle of judicial review, giving the Supreme Court the power to declare acts of Congress unconstitutional.',
    keyTerms: ['judicial review', 'constitutional law', 'separation of powers', 'supreme court', 'marbury']
  },
  {
    id: '5',
    title: 'Gideon v. Wainwright',
    category: 'precedent',
    court: 'Supreme Court of the United States',
    jurisdiction: 'Federal',
    date: '1963-03-18',
    citation: '372 U.S. 335 (1963)',
    summary: 'Established that states are required to provide defense attorneys to criminal defendants who are unable to afford their own.',
    keyTerms: ['right to counsel', 'sixth amendment', 'criminal defense', 'indigent defendants', 'due process']
  },
  
  // Laws & Regulations
  {
    id: '6',
    title: 'Americans with Disabilities Act',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '1990-07-26',
    citation: '42 U.S.C. § 12101',
    summary: 'Civil rights law that prohibits discrimination based on disability in employment, public accommodations, and other areas.',
    keyTerms: ['disability', 'discrimination', 'accommodation', 'civil rights', 'employment']
  },
  {
    id: '7',
    title: 'Family and Medical Leave Act',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '1993-02-05',
    citation: '29 U.S.C. § 2601',
    summary: 'Provides eligible employees with unpaid, job-protected leave for specified family and medical reasons.',
    keyTerms: ['family leave', 'medical leave', 'employment', 'fmla', 'job protection']
  },
  {
    id: '8',
    title: 'OSHA Workplace Safety Standards',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '2021-03-15',
    citation: '29 CFR 1910',
    summary: 'Occupational Safety and Health Administration standards for workplace safety and health protection.',
    keyTerms: ['workplace safety', 'occupational health', 'osha', 'employee protection', 'safety standards']
  },
  {
    id: '9',
    title: 'First Amendment - Freedom of Speech',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '1791-12-15',
    citation: 'U.S. Constitution Amendment I',
    summary: 'Constitutional amendment protecting freedom of speech, religion, press, assembly, and petition.',
    keyTerms: ['first amendment', 'freedom of speech', 'religion', 'press', 'assembly', 'petition']
  },
  {
    id: '10',
    title: 'Fourth Amendment - Search and Seizure',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '1791-12-15',
    citation: 'U.S. Constitution Amendment IV',
    summary: 'Constitutional amendment protecting against unreasonable searches and seizures.',
    keyTerms: ['fourth amendment', 'search and seizure', 'warrant', 'probable cause', 'privacy']
  },
  {
    id: '11',
    title: 'Federal Rules of Civil Procedure',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '2023-12-01',
    citation: 'Fed. R. Civ. P.',
    summary: 'Rules governing civil procedure in United States district courts.',
    keyTerms: ['civil procedure', 'federal court', 'litigation', 'discovery', 'pleadings']
  },
  {
    id: '12',
    title: 'Federal Rules of Evidence',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '2023-12-01',
    citation: 'Fed. R. Evid.',
    summary: 'Rules governing the admission of evidence in federal court proceedings.',
    keyTerms: ['evidence', 'admissibility', 'hearsay', 'expert testimony', 'authentication']
  },
  {
    id: '13',
    title: 'Universal Declaration of Human Rights',
    category: 'laws',
    jurisdiction: 'International',
    date: '1948-12-10',
    citation: 'UN General Assembly Resolution 217 A',
    summary: 'International document that proclaims the inalienable rights of all human beings.',
    keyTerms: ['human rights', 'international law', 'united nations', 'civil rights', 'universal rights']
  },
  {
    id: '14',
    title: 'Geneva Conventions',
    category: 'laws',
    jurisdiction: 'International',
    date: '1949-08-12',
    citation: '75 UNTS 31',
    summary: 'International treaties that establish standards for humanitarian treatment in war.',
    keyTerms: ['war crimes', 'humanitarian law', 'geneva conventions', 'international conflict', 'prisoners of war']
  },
  {
    id: '15',
    title: 'Clean Air Act',
    category: 'laws',
    jurisdiction: 'Federal',
    date: '1990-11-15',
    citation: '42 U.S.C. § 7401',
    summary: 'Federal law designed to control air pollution on a national level and protect public health and the environment.',
    keyTerms: ['environmental law', 'air pollution', 'clean air', 'epa', 'emissions standards']
  }
];

// Update category counts
legalCategories.forEach(category => {
  if (category.id === 'all') {
    category.count = mockLegalDocuments.length;
  } else {
    category.count = mockLegalDocuments.filter(doc => doc.category === category.id).length;
  }
}); 