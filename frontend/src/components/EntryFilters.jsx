import React from 'react';

// Search + "which AI" chips + sort, shared by the Knowledge Base and the
// Learning Hub. `aiCounts` is [{ label, count }] for all (unfiltered) entries.
export default function EntryFilters({
  search, onSearch, placeholder,
  aiCounts, aiFilter, onAiFilter,
  sort, onSort,
  extraFilters = [], activeExtra, onExtra,
}) {
  const total = aiCounts.reduce((s, a) => s + a.count, 0);
  const chip = (active) =>
    `text-xs px-2.5 py-1 rounded-full border transition whitespace-nowrap ${
      active
        ? 'bg-indigo-600 text-white border-indigo-600'
        : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
    }`;

  return (
    <div className="space-y-2 mb-4">
      <div className="flex gap-2">
        <input
          type="search"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          placeholder={placeholder}
          aria-label="Search entries"
          className="flex-1 min-w-0 border border-gray-200 rounded-lg px-3 py-2 text-sm bg-white focus:ring-2 focus:ring-indigo-400"
        />
        <select
          value={sort}
          onChange={(e) => onSort(e.target.value)}
          aria-label="Sort entries"
          className="border border-gray-200 rounded-lg px-2 py-2 text-sm bg-white"
        >
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="title">Title A–Z</option>
        </select>
      </div>

      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by AI model">
        <span className="text-xs text-gray-500 mr-1">AI:</span>
        <button className={chip(aiFilter === 'all')} onClick={() => onAiFilter('all')}>
          All ({total})
        </button>
        {aiCounts.map(({ label, count }) => (
          <button key={label} className={chip(aiFilter === label)} onClick={() => onAiFilter(label)}>
            🤖 {label} ({count})
          </button>
        ))}
      </div>

      {extraFilters.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by status">
          <span className="text-xs text-gray-500 mr-1">Show:</span>
          {extraFilters.map(({ id, label }) => (
            <button key={id} className={chip(activeExtra === id)} onClick={() => onExtra(id)}>
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Renders **bold** and line breaks without injecting HTML.
export function RichText({ text }) {
  if (!text) return null;
  return (
    <>
      {String(text).split('\n').map((line, i) => (
        <React.Fragment key={i}>
          {i > 0 && <br />}
          {line.split(/(\*\*[^*]+\*\*)/g).map((part, j) =>
            part.startsWith('**') && part.endsWith('**')
              ? <strong key={j}>{part.slice(2, -2)}</strong>
              : <React.Fragment key={j}>{part}</React.Fragment>
          )}
        </React.Fragment>
      ))}
    </>
  );
}
