const CACHE_NAME = 'essay-evaluator-v2';
const urlsToCache = [
  '/',
  '/index.html',
  '/manifest.json'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  const url = event.request.url;

  // Do not intercept API calls – let browser handle normally
  if (url.includes('/api/') || url.includes('/ratings') || url.includes('/surveys') ||
      url.includes('/knowledge') || url.includes('/evaluate') || url.includes('/ocr') ||
      url.includes('/saved-essays') || url.includes('/override') || url.includes('/learning-kb')) {
    return;
  }

  // For navigation requests, always go network first (avoid stale index.html)
  if (event.request.mode === 'navigate') {
    event.respondWith(fetch(event.request));
    return;
  }

  // For static assets (JS, CSS, images), cache with network fallback
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(name => name !== CACHE_NAME).map(name => caches.delete(name))
      );
    })
  );
});