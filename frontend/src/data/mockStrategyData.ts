export interface Case {
  id: string;
  title: string;
  client: string;
  opposingParty: string;
  caseNumber: string;
  court: string;
  status: 'active' | 'pending' | 'settled' | 'closed';
  priority: 'high' | 'medium' | 'low';
  nextDeadline: string;
  lastActivity: string;
  description: string;
}

export interface TimelineEvent {
  id: string;
  date: string;
  time: string;
  actor: 'our_side' | 'opposing_counsel' | 'judge';
  action: string;
  description: string;
  documents?: string[];
}

export interface AIRecommendation {
  id: string;
  title: string;
  type: 'motion' | 'discovery' | 'settlement' | 'brief' | 'evidence';
  description: string;
  explanation: string;
  winProbability: number; // Softmax score representing likelihood of increasing win chances
}

export const mockCases: Case[] = [
  {
    id: '1',
    title: 'Johnson v. MetroCorp Insurance',
    client: 'Sarah Johnson',
    opposingParty: 'MetroCorp Insurance Ltd.',
    caseNumber: 'CV-2024-001234',
    court: 'Superior Court of California',
    status: 'active',
    priority: 'high',
    nextDeadline: '2024-02-15',
    lastActivity: '2024-01-28',
    description: 'Personal injury claim following automobile accident. Disputed liability and damages.'
  },
  {
    id: '2',
    title: 'TechStart Inc. v. Innovate Solutions',
    client: 'TechStart Inc.',
    opposingParty: 'Innovate Solutions LLC',
    caseNumber: 'CV-2024-002156',
    court: 'Federal District Court',
    status: 'active',
    priority: 'medium',
    nextDeadline: '2024-02-20',
    lastActivity: '2024-01-25',
    description: 'Patent infringement dispute over software algorithms and trade secrets.'
  },
  {
    id: '3',
    title: 'Williams Employment Dispute',
    client: 'Marcus Williams',
    opposingParty: 'GlobalTech Corporation',
    caseNumber: 'CV-2024-001892',
    court: 'Los Angeles County Superior Court',
    status: 'pending',
    priority: 'medium',
    nextDeadline: '2024-03-01',
    lastActivity: '2024-01-20',
    description: 'Wrongful termination and discrimination claims under state and federal employment law.'
  },
  {
    id: '4',
    title: 'Riverside Property Development',
    client: 'Riverside Holdings LLC',
    opposingParty: 'City of Riverside',
    caseNumber: 'CV-2023-008745',
    court: 'Riverside County Superior Court',
    status: 'active',
    priority: 'low',
    nextDeadline: '2024-02-28',
    lastActivity: '2024-01-15',
    description: 'Zoning dispute and environmental impact assessment challenge for commercial development.'
  }
];

export const mockTimelineEvents: { [caseId: string]: TimelineEvent[] } = {
  '1': [
    {
      id: 't1',
      date: '2024-01-28',
      time: '14:30',
      actor: 'our_side',
      action: 'Filed Motion for Summary Judgment',
      description: 'Submitted comprehensive motion arguing no genuine issue of material fact exists regarding defendant\'s liability.',
      documents: ['Motion for Summary Judgment', 'Supporting Declaration', 'Evidence Exhibits A-F']
    },
    {
      id: 't2',
      date: '2024-01-25',
      time: '11:15',
      actor: 'opposing_counsel',
      action: 'Discovery Response Filed',
      description: 'Opposing counsel submitted responses to our Request for Production of Documents, with several objections noted.',
      documents: ['Discovery Responses', 'Privilege Log']
    },
    {
      id: 't3',
      date: '2024-01-22',
      time: '16:45',
      actor: 'judge',
      action: 'Case Management Order',
      description: 'Court issued updated case management order extending discovery deadline by 30 days.',
      documents: ['Case Management Order']
    },
    {
      id: 't4',
      date: '2024-01-18',
      time: '10:00',
      actor: 'our_side',
      action: 'Deposition Notice Served',
      description: 'Served notice for deposition of defendant\'s claims adjuster and key witnesses.',
      documents: ['Deposition Notice', 'Document Request List']
    },
    {
      id: 't5',
      date: '2024-01-15',
      time: '15:20',
      actor: 'opposing_counsel',
      action: 'Motion to Compel Filed',
      description: 'Defense filed motion to compel further responses to their interrogatories.',
      documents: ['Motion to Compel', 'Meet and Confer Declaration']
    }
  ]
};

export const mockAIRecommendations: { [caseId: string]: AIRecommendation[] } = {
  '1': [
    {
      id: 'r1',
      title: 'File Motion to Compel Discovery',
      type: 'motion',
      description: 'Compel opposing counsel to provide complete medical records and claims file documentation.',
      explanation: 'Based on analysis of their discovery responses, several key documents are missing that are crucial to proving damages. The incomplete responses indicate they may be withholding material evidence. Filing this motion will ensure we have all necessary documentation to support our damage claims and liability arguments.',
      winProbability: 0.42
    },
    {
      id: 'r2',
      title: 'Obtain Expert Witness Testimony',
      type: 'discovery',
      description: 'Secure accident reconstruction expert and medical specialist to testify on liability and damages.',
      explanation: 'Expert testimony will significantly strengthen our position on both liability and damages. An accident reconstruction expert can definitively establish fault and counter defense arguments, while a medical expert can validate the extent of injuries and future medical needs, providing credible testimony that judges and juries find compelling.',
      winProbability: 0.35
    },
    {
      id: 'r3',
      title: 'Initiate Settlement Negotiations',
      type: 'settlement',
      description: 'Present comprehensive settlement demand based on strong liability position and documented damages.',
      explanation: 'With our current evidence and the strength of our liability case, initiating settlement discussions from a position of strength could result in favorable resolution without trial risks. Our documented damages and clear liability evidence give us significant negotiating leverage that could lead to an optimal outcome for the client.',
      winProbability: 0.23
    }
  ]
}; 