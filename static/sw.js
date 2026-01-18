/* Service Worker for AI Chat Playground */
const VERSION = new URL(self.location).searchParams.get('v') || 'dev';
const CACHE_NAME = `ai-chat-cache-${VERSION}`;
const RUNTIME = `ai-chat-runtime-${VERSION}`;

const PRECACHE_URLS = [];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.map((key) => {
      if (!key.includes(VERSION)) return caches.delete(key);
      return null;
    }))).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.pathname.startsWith('/api/')) return;

  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((res) => {
          const copy = res.clone();
          caches.open(RUNTIME).then((cache) => cache.put(event.request, copy));
          return res;
        }).catch(() => cached);
      })
    );
    return;
  }

  // CDN runtime cache (stale-while-revalidate)
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const networkFetch = fetch(event.request).then((res) => {
        const copy = res.clone();
        caches.open(RUNTIME).then((cache) => cache.put(event.request, copy));
        return res;
      }).catch(() => cached);
      return cached || networkFetch;
    })
  );
});
