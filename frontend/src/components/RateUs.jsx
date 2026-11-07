import React, { useState, useEffect, useCallback, useMemo } from 'react';
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
const STAR_LABELS = ['', 'Poor', 'Fair', 'Good', 'Very good', 'Excellent'];
const MAX_REVIEW = 1000;
const SORTS = {
  recent: { label: 'Most recent', cmp: (a, b) => new Date(b.created_at) - new Date(a.created_at) },
  highest: { label: 'Highest rated', cmp: (a, b) => b.rating - a.rating || new Date(b.created_at) - new Date(a.created_at) },
  lowest: { label: 'Lowest rated', cmp: (a, b) => a.rating - b.rating || new Date(b.created_at) - new Date(a.created_at) },
};

// "3 days ago", falling back to the date for anything older than a month.
function timeAgo(at) {
  const then = new Date(at);
  const seconds = (Date.now() - then.getTime()) / 1000;
  if (!Number.isFinite(seconds)) return '';
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return rtf.format(-Math.round(seconds / 60), 'minute');
  if (seconds < 86400) return rtf.format(-Math.round(seconds / 3600), 'hour');
  if (seconds < 86400 * 30) return rtf.format(-Math.round(seconds / 86400), 'day');
  return then.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

const STAR_PATH = 'M12 2.5l2.9 6 6.6.9-4.8 4.6 1.2 6.5L12 17.4l-5.9 3.1 1.2-6.5L2.5 9.4l6.6-.9z';

// Stars filled to `value` (fractions allowed, e.g. 4.3).
function Stars({ value, size = 'w-5 h-5' }) {
  return (
    <span className="inline-flex items-center gap-0.5" role="img" aria-label={`${value} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((n) => {
        const fill = Math.max(0, Math.min(1, value - (n - 1)));
        return (
          <span key={n} className={`relative inline-block ${size}`} aria-hidden="true">
            <svg viewBox="0 0 24 24" className={`absolute inset-0 ${size} text-gray-300`} fill="currentColor"><path d={STAR_PATH} /></svg>
            <span className="absolute inset-0 overflow-hidden" style={{ width: `${fill * 100}%` }}>
              <svg viewBox="0 0 24 24" className={`${size} text-amber-400`} fill="currentColor"><path d={STAR_PATH} /></svg>
            </span>
          </span>
        );
      })}
    </span>
  );
}

function Avatar({ name, src, size = 'w-10 h-10 text-base' }) {
  if (src) return <img src={src} alt="" className={`${size} shrink-0 rounded-full object-cover ring-2 ring-white`} />;
  return (
    <span className={`${size} shrink-0 rounded-full bg-gradient-to-br from-indigo-500 to-violet-500 text-white font-semibold flex items-center justify-center ring-2 ring-white`} aria-hidden="true">
      {(name || 'U').charAt(0).toUpperCase()}
    </span>
  );
}

// ------ CommentsSection Component ------
function CommentsSection({ ratingId }) {
  const { user } = useAuth();
  const [comments, setComments] = useState(null);
  const [newComment, setNewComment] = useState('');
  const [replyTo, setReplyTo] = useState(null); // { parentId, userName, body }
  const [submitting, setSubmitting] = useState(false);

  const loadComments = useCallback(async () => {
    try {
      setComments(await listComments(ratingId));
    } catch (err) {
      console.error('Failed to load comments:', err);
      setComments([]);
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
      await createComment(ratingId, parentId, body.trim());
      toast.success(parentId ? 'Reply posted' : 'Comment posted');
      setNewComment('');
      setReplyTo(null);
      loadComments();
    } catch {
      toast.error('Failed to add comment');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReaction = async (commentId, reactionType) => {
    if (!user) {
      toast.error('Please sign in to react');
      return;
    }
    try {
      await toggleReaction(commentId, reactionType);
      loadComments();
    } catch {
      toast.error('Reaction failed');
    }
  };

  const renderComment = (comment, isReply = false) => {
    const { id, user_name, user_avatar, body, created_at, reactions = {}, user_reactions } = comment;
    const replies = comments.filter((c) => c.parent_id === id);
    return (
      <li key={id} className={isReply ? 'mt-3' : 'mt-4 first:mt-0'}>
        <div className="flex items-start gap-2.5">
          <Avatar name={user_name} src={user_avatar} size="w-8 h-8 text-xs" />
          <div className="min-w-0 flex-1">
            <div className="rounded-2xl rounded-tl-sm bg-gray-50 px-3 py-2">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-sm font-semibold text-gray-900">{user_name || 'Anonymous'}</span>
                <span className="text-xs text-gray-500">{timeAgo(created_at)}</span>
              </div>
              <p className="mt-0.5 whitespace-pre-wrap text-sm text-gray-700">{body}</p>
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-1 pl-1">
              {REACTIONS.map(({ type, label }) => {
                const mine = user_reactions?.includes(type);
                const count = reactions[type] || 0;
                return (
                  <button key={type} type="button" onClick={() => handleReaction(id, type)} title={label}
                    aria-label={`${label}${count ? ` (${count})` : ''}`} aria-pressed={!!mine}
                    className={`flex items-center gap-1 rounded-full px-1.5 py-0.5 text-xs transition ${
                      mine ? 'bg-indigo-100 text-indigo-700 ring-1 ring-indigo-300' : count ? 'bg-gray-100 text-gray-600 hover:bg-gray-200' : 'text-gray-500 opacity-60 hover:bg-gray-100 hover:opacity-100'}`}>
                    <span>{type}</span>{count > 0 && <span>{count}</span>}
                  </button>
                );
              })}
              {user && !isReply && (
                <button type="button" onClick={() => setReplyTo({ parentId: id, userName: user_name, body: '' })}
                  className="ml-1 text-xs font-medium text-gray-500 hover:text-indigo-600">Reply</button>
              )}
            </div>
            {replies.length > 0 && (
              <ul className="mt-1 border-l-2 border-gray-100 pl-3">{replies.map((r) => renderComment(r, true))}</ul>
            )}
          </div>
        </div>
      </li>
    );
  };

  const composer = (
    <div className="mt-4">
      {replyTo && (
        <div className="mb-2 flex items-center gap-2 text-xs text-gray-600">
          Replying to <span className="font-semibold">{replyTo.userName || 'Anonymous'}</span>
          <button type="button" onClick={() => setReplyTo(null)} className="text-red-500 hover:underline">Cancel</button>
        </div>
      )}
      <div className="flex items-start gap-2">
        <textarea
          value={replyTo ? replyTo.body : newComment}
          onChange={(e) => (replyTo ? setReplyTo({ ...replyTo, body: e.target.value }) : setNewComment(e.target.value))}
          placeholder={replyTo ? 'Write a reply…' : 'Write a comment…'}
          aria-label={replyTo ? 'Reply' : 'Comment'}
          className="min-h-[2.5rem] flex-1 resize-y rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm focus:border-transparent focus:ring-2 focus:ring-indigo-500"
          rows={1}
        />
        <button type="button" onClick={() => handleAddComment(replyTo?.parentId)} disabled={submitting}
          className="rounded-xl bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
          {submitting ? 'Posting…' : replyTo ? 'Reply' : 'Post'}
        </button>
      </div>
    </div>
  );

  return (
    <div className="mt-4 border-t border-gray-100 pt-4">
      {comments === null ? (
        <p className="text-sm text-gray-500">Loading comments…</p>
      ) : comments.length === 0 ? (
        <p className="text-sm text-gray-500">No comments yet.</p>
      ) : (
        <ul>{comments.filter((c) => c.parent_id === null).map((c) => renderComment(c))}</ul>
      )}
      {user ? composer : <p className="mt-3 text-xs text-gray-500">Sign in to join the conversation.</p>}
    </div>
  );
}

function ReviewCard({ review, mine }) {
  const [showComments, setShowComments] = useState(false);
  return (
    <article className={`rounded-2xl border bg-white p-5 shadow-sm transition hover:shadow-md ${mine ? 'border-indigo-200 ring-1 ring-indigo-100' : 'border-gray-100'}`}>
      <header className="flex items-start gap-3">
        <Avatar name={review.user_name} src={review.user_avatar} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h4 className="font-semibold text-gray-900">{review.user_name || 'Anonymous'}</h4>
            {mine && <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700">Your review</span>}
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-2 text-sm">
            <Stars value={review.rating} size="w-4 h-4" />
            <span className="font-medium text-gray-700">{STAR_LABELS[review.rating]}</span>
            <span className="text-gray-300" aria-hidden="true">•</span>
            <time className="text-gray-500" dateTime={review.created_at} title={new Date(review.created_at).toLocaleString()}>
              {timeAgo(review.updated_at || review.created_at)}
            </time>
          </div>
        </div>
      </header>
      {review.comment
        ? <p className="mt-3 whitespace-pre-wrap leading-relaxed text-gray-700">{review.comment}</p>
        : <p className="mt-3 text-sm italic text-gray-400">Rated without a written review.</p>}
      <footer className="mt-4">
        <button type="button" onClick={() => setShowComments((v) => !v)} aria-expanded={showComments}
          className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 -ml-2 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:text-indigo-700">
          💬 {showComments ? 'Hide comments' : 'Comments'}
        </button>
        {showComments && <CommentsSection ratingId={review.id} />}
      </footer>
    </article>
  );
}

// ------ Main RateUs Component ------
export default function RateUs() {
  const { user } = useAuth();
  const [ratings, setRatings] = useState(null);
  const [summary, setSummary] = useState({ average: 0, count: 0, distribution: {} });
  const [userRating, setUserRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [sort, setSort] = useState('recent');
  const [starFilter, setStarFilter] = useState(0);

  const loadData = useCallback(async () => {
    try {
      const [ratingsData, summaryData] = await Promise.all([getRatings(), getRatingSummary()]);
      setRatings(Array.isArray(ratingsData) ? ratingsData : []);
      setSummary(summaryData);
    } catch {
      setRatings([]);
      toast.error('Failed to load ratings');
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const mine = useMemo(() => (user && ratings ? ratings.find((r) => r.user_id === user.id) : null), [user, ratings]);

  const openForm = () => {
    setUserRating(mine?.rating || 0);
    setComment(mine?.comment || '');
    setShowForm(true);
  };

  const handleSubmit = async () => {
    if (userRating < 1) {
      toast.error('Please choose a star rating');
      return;
    }
    setSubmitting(true);
    try {
      await submitRating(userRating, comment.trim());
      toast.success(mine ? 'Your review was updated' : 'Thank you for your feedback!');
      setShowForm(false);
      loadData();
    } catch {
      toast.error('Failed to submit rating');
    } finally {
      setSubmitting(false);
    }
  };

  const count = summary.count || 0;
  const shown = (ratings || [])
    .filter((r) => !starFilter || r.rating === starFilter)
    .sort(SORTS[sort].cmp);
  const active = hover || userRating;

  return (
    <div className="mx-auto max-w-6xl animate-fade-in space-y-6">
      {/* Header */}
      <div className="glass-card rounded-2xl bg-white/95 px-6 py-7 shadow-xl sm:px-8">
        <p className="text-sm font-semibold uppercase tracking-wider text-indigo-600">Rate Us</p>
        <h2 className="mt-1 text-3xl font-bold text-gray-900">Reviews from our users</h2>
        <p className="mt-2 max-w-2xl text-gray-600">
          Tell us how the AI Essay Evaluator worked for you. Reviews help us improve the scoring, the feedback and the experience for every teacher and student.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[22rem_1fr]">
        {/* Left: summary and your review */}
        <aside className="space-y-6 lg:sticky lg:top-24 lg:self-start">
          <section className="glass-card rounded-2xl bg-white/95 p-6 shadow-xl" aria-labelledby="rating-summary">
            <h3 id="rating-summary" className="text-sm font-semibold uppercase tracking-wider text-gray-500">Overall rating</h3>
            <div className="mt-3 flex items-end gap-3">
              <span className="text-5xl font-bold tabular-nums text-gray-900">{count ? Number(summary.average).toFixed(1) : '–'}</span>
              <span className="pb-1.5 text-gray-500">out of 5</span>
            </div>
            <div className="mt-2"><Stars value={Number(summary.average) || 0} size="w-6 h-6" /></div>
            <p className="mt-1 text-sm text-gray-600">Based on {count} {count === 1 ? 'review' : 'reviews'}</p>

            <div className="mt-5 space-y-1.5">
              {[5, 4, 3, 2, 1].map((star) => {
                const n = summary.distribution?.[star] || 0;
                const pct = count ? Math.round((n / count) * 100) : 0;
                const selected = starFilter === star;
                return (
                  <button key={star} type="button" onClick={() => setStarFilter(selected ? 0 : star)} aria-pressed={selected}
                    title={selected ? 'Show all reviews' : `Show ${star}-star reviews`}
                    className={`flex w-full items-center gap-3 rounded-lg px-2 py-1 text-sm transition ${selected ? 'bg-indigo-50 ring-1 ring-indigo-200' : 'hover:bg-gray-50'}`}>
                    <span className="w-12 shrink-0 text-left font-medium text-gray-700">{star} star</span>
                    <span className="h-2 flex-1 overflow-hidden rounded-full bg-gray-200">
                      <span className="block h-full rounded-full bg-amber-400" style={{ width: `${pct}%` }} />
                    </span>
                    <span className="w-10 shrink-0 text-right tabular-nums text-gray-500">{pct}%</span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="glass-card rounded-2xl bg-white/95 p-6 shadow-xl" aria-labelledby="your-review">
            <h3 id="your-review" className="text-lg font-semibold text-gray-900">{mine ? 'Your review' : 'Share your experience'}</h3>
            {!user ? (
              <p className="mt-2 text-sm text-gray-600">Sign in to rate the AI Essay Evaluator and write a review.</p>
            ) : showForm ? (
              <div className="mt-3">
                <div role="radiogroup" aria-label="Your rating" className="flex items-center gap-1" onMouseLeave={() => setHover(0)}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button key={star} type="button" role="radio" aria-checked={userRating === star} aria-label={`${star} star${star > 1 ? 's' : ''}: ${STAR_LABELS[star]}`}
                      onClick={() => setUserRating(star)} onMouseEnter={() => setHover(star)}
                      className="rounded-md p-0.5 transition hover:scale-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
                      <svg viewBox="0 0 24 24" className={`h-8 w-8 ${active >= star ? 'text-amber-400' : 'text-gray-300'}`} fill="currentColor" aria-hidden="true"><path d={STAR_PATH} /></svg>
                    </button>
                  ))}
                </div>
                <p className="mt-1 h-5 text-sm font-medium text-gray-700">{STAR_LABELS[active] || 'Choose a rating'}</p>
                <label htmlFor="review-text" className="mt-3 block text-sm font-medium text-gray-700">Your review <span className="font-normal text-gray-400">(optional)</span></label>
                <textarea id="review-text" value={comment} maxLength={MAX_REVIEW}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder="What did you like? What could be better?"
                  className="mt-1 w-full rounded-xl border border-gray-200 bg-white p-3 text-sm focus:border-transparent focus:ring-2 focus:ring-indigo-500"
                  rows={4} />
                <p className="text-right text-xs text-gray-400">{comment.length}/{MAX_REVIEW}</p>
                <div className="mt-2 flex gap-2">
                  <button type="button" onClick={handleSubmit} disabled={submitting}
                    className="flex-1 rounded-xl bg-indigo-600 px-4 py-2.5 font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50">
                    {submitting ? 'Submitting…' : mine ? 'Update review' : 'Submit review'}
                  </button>
                  <button type="button" onClick={() => setShowForm(false)}
                    className="rounded-xl border border-gray-200 px-4 py-2.5 font-medium text-gray-700 hover:bg-gray-50">Cancel</button>
                </div>
              </div>
            ) : mine ? (
              <div className="mt-3">
                <div className="flex items-center gap-2"><Stars value={mine.rating} /><span className="text-sm font-medium text-gray-700">{STAR_LABELS[mine.rating]}</span></div>
                {mine.comment && <p className="mt-2 line-clamp-4 text-sm text-gray-700">{mine.comment}</p>}
                <button type="button" onClick={openForm}
                  className="mt-4 w-full rounded-xl border border-indigo-200 px-4 py-2.5 font-medium text-indigo-700 hover:bg-indigo-50">Edit your review</button>
              </div>
            ) : (
              <>
                <p className="mt-2 text-sm text-gray-600">How was your experience evaluating essays with the app?</p>
                <button type="button" onClick={openForm}
                  className="mt-4 w-full rounded-xl bg-indigo-600 px-4 py-2.5 font-medium text-white shadow-sm hover:bg-indigo-700">Write a review</button>
              </>
            )}
          </section>
        </aside>

        {/* Right: reviews */}
        <section aria-labelledby="reviews-title" className="min-w-0">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl bg-white/95 px-5 py-3 shadow-xl glass-card">
            <div className="flex flex-wrap items-center gap-2">
              <h3 id="reviews-title" className="font-semibold text-gray-900">
                {starFilter ? `${starFilter}-star reviews` : 'All reviews'} <span className="font-normal text-gray-500">({shown.length})</span>
              </h3>
              {starFilter > 0 && (
                <button type="button" onClick={() => setStarFilter(0)} className="rounded-full bg-gray-100 px-2.5 py-0.5 text-xs text-gray-700 hover:bg-gray-200">✕ Clear filter</button>
              )}
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-600">
              Sort by
              <select value={sort} onChange={(e) => setSort(e.target.value)} className="rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm">
                {Object.entries(SORTS).map(([key, s]) => <option key={key} value={key}>{s.label}</option>)}
              </select>
            </label>
          </div>

          {ratings === null ? (
            <div className="flex justify-center py-16"><div className="spinner" /></div>
          ) : shown.length === 0 ? (
            <div className="glass-card rounded-2xl bg-white/95 p-10 text-center shadow-xl">
              <div className="text-4xl" aria-hidden="true">⭐</div>
              <p className="mt-2 font-medium text-gray-800">{starFilter ? 'No reviews with this rating yet.' : 'No reviews yet.'}</p>
              <p className="text-sm text-gray-500">{user ? 'Be the first to share your experience.' : 'Sign in to write the first review.'}</p>
            </div>
          ) : (
            <div className="space-y-4">
              {shown.map((r) => <ReviewCard key={r.id} review={r} mine={!!user && r.user_id === user.id} />)}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
