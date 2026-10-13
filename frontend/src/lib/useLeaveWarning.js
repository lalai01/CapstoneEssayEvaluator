import { useEffect } from 'react';

// While `active`, the browser asks for confirmation before a refresh, closing
// the tab, or pressing Back out of the app, so unsaved work isn't lost by
// accident. Browsers show their own wording ("Changes you made may not be
// saved"); for security they ignore any custom message.
export default function useLeaveWarning(active) {
  useEffect(() => {
    if (!active) return undefined;
    const warn = (e) => {
      e.preventDefault();
      e.returnValue = '';   // still required by some browsers
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [active]);
}
