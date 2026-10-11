import React from 'react';

/**
 * Centered loading indicator — spinner + indeterminate progress bar.
 *
 * Usage:
 *   if (loading) return <LoadingIndicator message="Loading essays…" />;
 */
export default function LoadingIndicator({ message = 'Loading…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      {/* Spinner */}
      <div className="relative w-16 h-16 mb-6">
        <div className="absolute inset-0 rounded-full border-4 border-blue-100"></div>
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-600 animate-spin"></div>
      </div>

      {/* Message */}
      {message && (
        <p className="text-gray-700 font-medium mb-4">{message}</p>
      )}

      {/* Indeterminate progress bar */}
      <div className="w-64 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div className="h-full w-1/3 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full animate-loading-bar"></div>
      </div>
    </div>
  );
}