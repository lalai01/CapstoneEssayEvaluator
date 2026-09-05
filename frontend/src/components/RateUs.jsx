import React, { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { submitRating, getRatings, getRatingSummary, createComment, listComments, toggleReaction } from '../api';
import toast from 'react-hot-toast';

// Reaction emoji options
const REACTIONS = [
  { type: '👍', label: 'Like' },
  { type: '❤️', label: 'Love' },
  { type: '😂', label: 'Haha' },
  { type: '😢', label: 'Sad' },
  { type: '😡', label: 'Angry' }
];

// ------ CommentsSection Component ------
function CommentsSection({ ratingId }) {
  const { user } = useAuth();
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');
  const [replyTo, setReplyTo] = useState(null); // { id, userName }
  const [submitting, setSubmitting] = useState(false);

  const loadComments = useCallback(async () => {
    if (!ratingId) return;
    try {
      const data = await listComments(ratingId);
      setComments(data);
    } catch (err) {
      console.error('Failed to load comments:', err);
    }
  }, [ratingId]);

  useEffect(() => {
    loadComments();
  }, [loadComments]);

  const handleAddComment = async (parentId = null) => {
    const body = parentId ? replyTo?.body : newComment;
    if (!body?.trim()) {
      toast.error('Comment cannot be empty');
      return;
    }
    setSubmitting(true);
    try {
      await createComment(ratingId, parentId, body);
      toast.success('Comment added');
      setNewComment('');
      setReplyTo(null);
      loadComments();
    } catch (err) {
      toast.error('Failed to add comment');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReaction = async (commentId, reactionType) => {
    if (!user) {
      toast.error('Please log in to react');
      return;
    }
    try {
      await toggleReaction(commentId, reactionType);
      loadComments(); // refresh to update reaction counts
    } catch (err) {
      toast.error('Reaction failed');
    }
  };

  const renderComment = (comment, isReply = false) => {
    const { id, user_name, user_avatar, body, created_at, reactions, user_reactions } = comment;
    const isOwnComment = user?.id === comment.user_id;

    return (
      <div key={id} className={`mb-3 ${isReply ? 'ml-8 border-l-2 border-gray-200 pl-3' : ''}`}>
        <div className="flex items-start gap-2">
          {user_avatar ? (
            <img src={user_avatar} alt={user_name} className="w-6 h-6 rounded-full" />
          ) : (
            <div className="w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-xs font-bold">
              {user_name?.charAt(0).toUpperCase() || 'U'}
            </div>
          )}
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-sm">{user_name || 'Anonymous'}</span>
              <span className="text-xs text-gray-500">{new Date(created_at).toLocaleDateString()}</span>
            </div>
            <p className="text-sm text-gray-700 mt-1">{body}</p>
            <div className="flex items-center gap-2 mt-1">
              {/* Reaction buttons */}
              {REACTIONS.map(({ type, label }) => (
                <button
                  key={type}
                  onClick={() => handleReaction(id, type)}
                  className={`text-xs flex items-center gap-1 px-1.5 py-0.5 rounded-full border ${
                    user_reactions?.includes(type)
                      ? 'bg-blue-100 border-blue-300'
                      : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                  }`}
                  title={label}
                >
                  <span>{type}</span>
                  {reactions[type] > 0 && <span className="text-gray-600">{reactions[type]}</span>}
                </button>
              ))}
              {user && (
                <button
                  onClick={() => setReplyTo({ parentId: id, userName: user_name, body: '' })}
                  className="text-xs text-gray-500 hover:text-blue-600"
                >
                  Reply
                </button>
              )}
            </div>
          </div>
        </div>
        {/* Replies */}
        {comments.filter(c => c.parent_id === id).map(reply => renderComment(reply, true))}
      </div>
    );
  };

  return (
    <div className="mt-3 border-t pt-3">
      <h5 className="text-sm font-semibold text-gray-700 mb-2">Comments</h5>
      {/* Existing comments */}
      {comments.filter(c => c.parent_id === null).map(comment => renderComment(comment))}

      {/* New comment input */}
      {user ? (
        <div className="mt-3">
          {replyTo ? (
            <div className="mb-2 text-sm text-gray-600 flex items-center gap-1">
              Replying to <span className="font-medium">@{replyTo.userName}</span>
              <button onClick={() => setReplyTo(null)} className="text-red-500 text-xs ml-1">Cancel</button>
            </div>
          ) : null}
          <textarea
            value={replyTo ? replyTo.body : newComment}
            onChange={(e) => {
              if (replyTo) {
                setReplyTo({ ...replyTo, body: e.target.value });
              } else {
                setNewComment(e.target.value);
              }
            }}
            placeholder="Write a comment..."
            className="w-full border rounded-lg p-2 text-sm mb-2"
            rows="2"
          />
          <button
            onClick={() => handleAddComment(replyTo?.parentId)}
            disabled={submitting}
            className="bg-indigo-600 text-white px-3 py-1 rounded-lg text-sm hover:bg-indigo-700 disabled:opacity-50"
          >
            {submitting ? 'Posting...' : replyTo ? 'Reply' : 'Comment'}
          </button>
        </div>
      ) : (
        <p className="text-xs text-gray-500 mt-2">Log in to comment.</p>
      )}
    </div>
  );
}

// ------ Main RateUs Component ------
export default function RateUs() {
  const { user } = useAuth();
  const [ratings, setRatings] = useState([]);
  const [summary, setSummary] = useState({ average: 0, count: 0, distribution: {} });
  const [userRating, setUserRating] = useState(0);
  const [comment, setComment] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [ratingsData, summaryData] = await Promise.all([
        getRatings(),
        getRatingSummary()
      ]);
      setRatings(ratingsData);
      setSummary(summaryData);
    } catch (err) {
      toast.error('Failed to load ratings');
    }
  };

  const handleSubmit = async () => {
    if (userRating < 1) {
      toast.error('Please select a rating');
      return;
    }
    setSubmitting(true);
    try {
      await submitRating(userRating, comment);
      toast.success('Thank you for your feedback!');
      setShowForm(false);
      setUserRating(0);
      setComment('');
      loadData();
    } catch (err) {
      toast.error('Failed to submit rating');
    } finally {
      setSubmitting(false);
    }
  };

  const renderStars = (rating) => '⭐'.repeat(rating) + '☆'.repeat(5 - rating);

  return (
    <div className="glass-card rounded-2xl shadow-xl p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold text-indigo-700 mb-4">⭐ Rate Us</h2>

      {/* Summary */}
      <div className="flex items-center gap-6 mb-6">
        <div className="text-center">
          <div className="text-5xl font-bold text-indigo-600">{summary.average}</div>
          <div className="text-sm text-gray-600">{summary.count} reviews</div>
          <div className="text-yellow-500 text-xl">{renderStars(Math.round(summary.average))}</div>
        </div>
        <div className="flex-1">
          {[5, 4, 3, 2, 1].map(star => (
            <div key={star} className="flex items-center gap-2">
              <span className="w-6 text-sm">{star} ★</span>
              <div className="flex-1 h-2 bg-gray-200 rounded-full">
                <div
                  className="h-2 bg-yellow-400 rounded-full"
                  style={{ width: `${(summary.distribution[star] || 0) / (summary.count || 1) * 100}%` }}
                />
              </div>
              <span className="w-8 text-xs text-gray-500">{summary.distribution[star] || 0}</span>
            </div>
          ))}
        </div>
      </div>

      {/* User's own rating / form */}
      {user ? (
        !showForm ? (
          <button
            onClick={() => setShowForm(true)}
            className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
          >
            Write a Review
          </button>
        ) : (
          <div className="bg-gray-50 p-4 rounded-lg mb-6">
            <div className="flex items-center gap-2 mb-2">
              <span className="font-medium">Your Rating:</span>
              {[1, 2, 3, 4, 5].map(star => (
                <button
                  key={star}
                  onClick={() => setUserRating(star)}
                  className={`text-2xl ${userRating >= star ? 'text-yellow-500' : 'text-gray-300'}`}
                >
                  ★
                </button>
              ))}
            </div>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Share your experience (optional)"
              className="w-full border rounded-lg p-2 mb-2"
              rows="3"
            />
            <div className="flex gap-2">
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {submitting ? 'Submitting...' : 'Submit'}
              </button>
              <button
                onClick={() => setShowForm(false)}
                className="bg-gray-300 px-4 py-2 rounded-lg hover:bg-gray-400 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        )
      ) : (
        <p className="text-gray-600 mb-4">Please log in to leave a review.</p>
      )}

      {/* Reviews List with Comments */}
      <div className="space-y-6 max-h-[600px] overflow-y-auto pr-2">
        {ratings.map(r => (
          <div key={r.id} className="border-b pb-4">
            <div className="flex items-center gap-3">
              {r.user_avatar ? (
                <img
                  src={r.user_avatar}
                  alt={r.user_name}
                  className="w-10 h-10 rounded-full object-cover"
                />
              ) : (
                <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center text-indigo-600 font-bold text-lg">
                  {r.user_name?.charAt(0).toUpperCase() || 'U'}
                </div>
              )}
              <div className="flex-1">
                <div className="font-medium text-gray-800">{r.user_name || 'Anonymous'}</div>
                <div className="text-yellow-500">{renderStars(r.rating)}</div>
              </div>
              <div className="text-xs text-gray-400">
                {new Date(r.created_at).toLocaleDateString()}
              </div>
            </div>
            {r.comment && (
              <p className="mt-2 text-gray-700 pl-13">{r.comment}</p>
            )}
            {/* Comments section for this rating */}
            {r.id && <CommentsSection ratingId={r.id} />}
          </div>
        ))}
      </div>
    </div>
  );
}