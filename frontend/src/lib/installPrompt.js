import { useEffect, useState } from 'react';

// "Install app" support. Chrome and Edge fire `beforeinstallprompt` when the
// site can be installed (as a Windows, Android or ChromeOS app); keeping the
// event lets our own button show the install dialog. The listener is set up
// when this module loads, because the event can fire before React renders.
let deferred = null;
const listeners = new Set();
const notify = () => listeners.forEach((fn) => fn());

if (typeof window !== 'undefined') {
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();   // we show our own button instead of the browser's mini bar
    deferred = e;
    notify();
  });
  window.addEventListener('appinstalled', () => {
    deferred = null;
    notify();
  });
}

// Running as the installed app (its own window), not in a browser tab.
export const isInstalledApp = () =>
  typeof window !== 'undefined' &&
  (window.matchMedia?.('(display-mode: standalone)').matches || window.navigator.standalone === true);

export default function useInstallPrompt() {
  const [canInstall, setCanInstall] = useState(!!deferred);
  useEffect(() => {
    const update = () => setCanInstall(!!deferred);
    listeners.add(update);
    update();
    return () => listeners.delete(update);
  }, []);

  // Shows the browser's install dialog; resolves to true if the user installed.
  const install = async () => {
    if (!deferred) return false;
    const prompt = deferred;
    deferred = null;
    notify();
    prompt.prompt();
    const { outcome } = await prompt.userChoice;
    return outcome === 'accepted';
  };

  return { canInstall: canInstall && !isInstalledApp(), install };
}
