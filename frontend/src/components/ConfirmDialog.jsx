import React, { useEffect, useRef } from 'react';

// A small modal for confirmations and notices.
//   tone: 'warning' | 'error' | 'info'
//   confirmLabel: omit for a notice with a single button
export default function ConfirmDialog({
  open, title, children, tone = 'warning', icon,
  confirmLabel, cancelLabel = 'Cancel', onConfirm, onCancel,
}) {
  const primaryRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    primaryRef.current?.focus();
    const onKey = (e) => { if (e.key === 'Escape') onCancel?.(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onCancel]);

  if (!open) return null;

  const accent = {
    warning: { ring: 'bg-amber-100 text-amber-700', button: 'bg-amber-600 hover:bg-amber-700' },
    error: { ring: 'bg-red-100 text-red-700', button: 'bg-red-600 hover:bg-red-700' },
    info: { ring: 'bg-indigo-100 text-indigo-700', button: 'bg-indigo-600 hover:bg-indigo-700' },
  }[tone];

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 animate-fade-in" onMouseDown={onCancel}>
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby="confirm-dialog-body"
        className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-xl ${accent.ring}`} aria-hidden="true">
            {icon || (tone === 'error' ? '!' : tone === 'info' ? 'i' : '?')}
          </div>
          <div className="min-w-0">
            <h3 id="confirm-dialog-title" className="text-lg font-semibold text-gray-900">{title}</h3>
            <div id="confirm-dialog-body" className="mt-1 text-sm text-gray-600 space-y-2">{children}</div>
          </div>
        </div>
        <div className="mt-6 flex justify-end gap-3">
          {confirmLabel ? (
            <>
              <button type="button" onClick={onCancel} className="rounded-xl bg-gray-100 px-4 py-2 text-sm font-medium text-gray-800 hover:bg-gray-200">
                {cancelLabel}
              </button>
              <button ref={primaryRef} type="button" onClick={onConfirm} className={`rounded-xl px-4 py-2 text-sm font-semibold text-white ${accent.button}`}>
                {confirmLabel}
              </button>
            </>
          ) : (
            <button ref={primaryRef} type="button" onClick={onCancel} className={`rounded-xl px-4 py-2 text-sm font-semibold text-white ${accent.button}`}>
              {cancelLabel}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
