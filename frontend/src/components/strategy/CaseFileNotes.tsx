import React, { useState } from 'react';
import { Trash2, PenTool, Plus, User, Bot, Tag, Calendar, MessageSquare } from 'lucide-react';
import { CaseFileNote, AddCaseFileNoteRequest, UpdateCaseFileNoteRequest } from '../../types/api';
import ConfirmModal from '../shared/ConfirmModal';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';

interface CaseFileNotesProps {
  notes: CaseFileNote[];
  caseFileId: number;
  onAddNote: (request: AddCaseFileNoteRequest) => Promise<void>;
  onUpdateNote: (noteId: number, request: UpdateCaseFileNoteRequest) => Promise<void>;
  onDeleteNote: (noteId: number) => Promise<void>;
  isLoading?: boolean;
}

interface NoteFormData {
  content: string;
  note_type: string;
  tags: string[];
}

const CaseFileNotes: React.FC<CaseFileNotesProps> = ({
  notes,
  caseFileId,
  onAddNote,
  onUpdateNote,
  onDeleteNote,
  isLoading = false
}) => {
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingNoteId, setEditingNoteId] = useState<number | null>(null);
  const [formData, setFormData] = useState<NoteFormData>({
    content: '',
    note_type: '',
    tags: []
  });
  const [tagInput, setTagInput] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Confirm modal state
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [confirmAction, setConfirmAction] = useState<(() => void) | null>(null);
  const [confirmDetails, setConfirmDetails] = useState({
    title: '',
    message: '',
    confirmText: '',
    noteId: 0,
    authorType: ''
  });
  const [isDeleting, setIsDeleting] = useState(false);

  const noteTypes = [
    { value: '', label: 'General' },
    { value: 'research', label: 'Research' },
    { value: 'strategy', label: 'Strategy' },
    { value: 'fact', label: 'Fact' },
    { value: 'reminder', label: 'Reminder' },
    { value: 'observation', label: 'Observation' }
  ];

  const resetForm = () => {
    setFormData({ content: '', note_type: '', tags: [] });
    setTagInput('');
    setShowAddForm(false);
    setEditingNoteId(null);
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, tagInput.trim()]
      }));
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.content.trim()) {
      alert('Please enter note content');
      return;
    }

    setIsSubmitting(true);
    try {
      if (editingNoteId) {
        // Update existing note
        await onUpdateNote(editingNoteId, {
          content: formData.content,
          note_type: formData.note_type || undefined,
          tags: formData.tags.length > 0 ? formData.tags : undefined
        });
      } else {
        // Add new note
        await onAddNote({
          content: formData.content,
          author_type: 'user',
          author_name: 'User', // You could make this configurable
          note_type: formData.note_type || undefined,
          tags: formData.tags.length > 0 ? formData.tags : undefined
        });
      }
      resetForm();
    } catch (error) {
      console.error('Failed to save note:', error);
      alert('Failed to save note. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEdit = (note: CaseFileNote) => {
    if (note.author_type !== 'user') {
      alert('You can only edit your own notes');
      return;
    }
    
    setFormData({
      content: note.content,
      note_type: note.note_type || '',
      tags: note.tags || []
    });
    setEditingNoteId(note.id);
    setShowAddForm(true);
  };

  const handleDelete = async (noteId: number, authorType: string) => {
    const confirmMessage = authorType === 'user' 
      ? 'Are you sure you want to delete this note? This action cannot be undone.' 
      : 'Are you sure you want to delete this AI-generated note? This action cannot be undone.';

    const confirmTitle = authorType === 'user' ? 'Delete Note' : 'Delete AI Note';

    setConfirmDetails({
      title: confirmTitle,
      message: confirmMessage,
      confirmText: 'Delete Note',
      noteId,
      authorType
    });

    setConfirmAction(() => async () => {
      setIsDeleting(true);
      try {
        await onDeleteNote(noteId);
      } catch (error) {
        console.error('Failed to delete note:', error);
        alert('Failed to delete note. Please try again.');
        throw error;
      } finally {
        setIsDeleting(false);
      }
    });

    setShowConfirmModal(true);
  };

  const handleConfirmDelete = async () => {
    if (confirmAction) {
      try {
        await confirmAction();
        setShowConfirmModal(false);
        setConfirmAction(null);
      } catch (error) {
        // Error handling is done in the confirmAction
      }
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getAuthorIcon = (authorType: string) => {
    return authorType === 'user' ? (
      <User className="h-4 w-4 text-blue-600" />
    ) : (
      <Bot className="h-4 w-4 text-purple-600" />
    );
  };

  const getAuthorBadgeColor = (authorType: string) => {
    return authorType === 'user' 
      ? 'bg-blue-100 text-blue-800'
      : 'bg-purple-100 text-purple-800';
  };

  const getNoteTypeBadgeColor = (noteType?: string) => {
    const colors: Record<string, string> = {
      research: 'bg-green-100 text-green-800',
      strategy: 'bg-orange-100 text-orange-800',
      fact: 'bg-yellow-100 text-yellow-800',
      reminder: 'bg-red-100 text-red-800',
      observation: 'bg-indigo-100 text-indigo-800'
    };
    return colors[noteType || ''] || 'bg-gray-100 text-gray-800';
  };

  const sortedNotes = [...notes].sort((a, b) => 
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <MessageSquare className="h-5 w-5 mr-2" />
            Notes ({notes.length})
          </h3>
          {!showAddForm && (
            <button
              onClick={() => setShowAddForm(true)}
              className="flex items-center px-3 py-1.5 text-sm bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors duration-200"
              disabled={isLoading}
            >
              <Plus className="h-4 w-4 mr-1.5" />
              Add Note
            </button>
          )}
        </div>
      </div>

      {/* Add/Edit Note Form */}
      {showAddForm && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Note Content
              </label>
              <textarea
                value={formData.content}
                onChange={(e) => setFormData(prev => ({ ...prev, content: e.target.value }))}
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent resize-vertical"
                placeholder="Enter your note..."
                required
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Note Type
                </label>
                <select
                  value={formData.note_type}
                  onChange={(e) => setFormData(prev => ({ ...prev, note_type: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
                >
                  {noteTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Tags
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent text-sm"
                    placeholder="Add tag..."
                  />
                  <button
                    type="button"
                    onClick={handleAddTag}
                    className="px-3 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-sm"
                  >
                    Add
                  </button>
                </div>
                {formData.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {formData.tags.map(tag => (
                      <span
                        key={tag}
                        className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                      >
                        {tag}
                        <button
                          type="button"
                          onClick={() => handleRemoveTag(tag)}
                          className="ml-1 text-blue-600 hover:text-blue-800"
                        >
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={resetForm}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-700 transition-colors"
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Saving...' : editingNoteId ? 'Update Note' : 'Add Note'}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Notes List */}
      <div className="max-h-96 overflow-y-auto">
        {sortedNotes.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {sortedNotes.map((note) => (
              <div key={note.id} className="p-4 hover:bg-gray-50">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    {/* Note header */}
                    <div className="flex items-center gap-2 mb-2">
                      {getAuthorIcon(note.author_type)}
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getAuthorBadgeColor(note.author_type)}`}>
                        {note.author_type === 'user' ? (note.author_name || 'User') : (note.author_name || 'AI')}
                      </span>
                      {note.note_type && (
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getNoteTypeBadgeColor(note.note_type)}`}>
                          {noteTypes.find(t => t.value === note.note_type)?.label || note.note_type}
                        </span>
                      )}
                      <span className="text-xs text-gray-500 flex items-center">
                        <Calendar className="h-3 w-3 mr-1" />
                        {formatDate(note.created_at)}
                      </span>
                    </div>

                    {/* Note content */}
                    <div className="mb-2 prose prose-sm max-w-none text-gray-800">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeSanitize]}
                        components={{
                          a: ({ node, ...props }) => (
                            <a
                              {...props}
                              className="text-blue-600 underline"
                              target="_blank"
                              rel="noopener noreferrer"
                            />
                          ),
                          code: ({ inline, className, children, ...props }: any) =>
                            inline ? (
                              <code className="rounded bg-gray-100 px-1 py-0.5" {...props}>
                                {children}
                              </code>
                            ) : (
                              <pre className="rounded bg-gray-900 text-gray-100 p-3 overflow-x-auto">
                                <code {...props}>{children}</code>
                              </pre>
                            ),
                          ul: ({ node, ...props }) => <ul className="list-disc pl-5" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal pl-5" {...props} />,
                          blockquote: ({ node, ...props }) => (
                            <blockquote className="border-l-4 border-gray-300 pl-3 italic text-gray-600" {...props} />
                          ),
                          table: ({ node, ...props }) => (
                            <div className="overflow-x-auto">
                              <table className="min-w-full border border-gray-200" {...props} />
                            </div>
                          ),
                          th: ({ node, ...props }) => <th className="border px-2 py-1 bg-gray-50" {...props} />,
                          td: ({ node, ...props }) => <td className="border px-2 py-1" {...props} />
                        }}
                      >
                        {note.content ? note.content : ''}
                      </ReactMarkdown>
                    </div>

                    {/* Tags */}
                    {note.tags && note.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {note.tags.map(tag => (
                          <span
                            key={tag}
                            className="inline-flex items-center px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
                          >
                            <Tag className="h-3 w-3 mr-1" />
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Action buttons */}
                  <div className="flex items-center space-x-2 flex-shrink-0">
                    {/* Edit button - only for user notes */}
                    {note.author_type === 'user' && (
                      <button
                        onClick={() => handleEdit(note)}
                        className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                        title="Edit note"
                        disabled={isLoading}
                      >
                        <PenTool className="h-4 w-4" />
                      </button>
                    )}
                    {/* Delete button - for both user and AI notes */}
                    <button
                      onClick={() => handleDelete(note.id, note.author_type)}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                      title={note.author_type === 'user' ? 'Delete note' : 'Delete AI note'}
                      disabled={isLoading}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-6 text-center text-gray-500">
            <MessageSquare className="h-12 w-12 mx-auto mb-3 text-gray-300" />
            <p>No notes yet</p>
            <p className="text-sm">Add notes to keep track of important information about this case</p>
          </div>
        )}
      </div>

      {/* Confirm Modal */}
      <ConfirmModal
        isOpen={showConfirmModal}
        onClose={() => setShowConfirmModal(false)}
        onConfirm={handleConfirmDelete}
        title={confirmDetails.title}
        message={confirmDetails.message}
        confirmText={confirmDetails.confirmText}
        isDestructive={true}
        isLoading={isDeleting}
      />
    </div>
  );
};

export default CaseFileNotes;
