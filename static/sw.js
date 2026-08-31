/* Service Worker for AI Chat Playground */
const VERSION = new URL(self.location).searchParams.get('v') || 'dev';
const CACHE_NAME = `ai-chat-cache-${VERSION}`;
const RUNTIME = `ai-chat-runtime-${VERSION}`;
const OFFLINE_FALLBACK_URL = '/static/offline.html';

const PRECACHE_URLS = [OFFLINE_FALLBACK_URL];

function isCacheable(res) {
  if (!res) return false;
  if (res.status === 200) return true;
  return res.type === 'opaque';
}

async function putRuntime(request, res) {
  if (!isCacheable(res)) return;
  const cache = await caches.open(RUNTIME);
  await cache.put(request, res.clone());
}

async function offlineFallback(request) {
  const exact = await caches.match(request);
  if (exact) return exact;
  const ignoreSearch = await caches.match(request, { ignoreSearch: true });
  if (ignoreSearch) return ignoreSearch;
  const root = await caches.match('/');
  if (root) return root;
  const offlinePage = await caches.match(OFFLINE_FALLBACK_URL, { ignoreSearch: true });
  if (offlinePage) return offlinePage;
  return new Response('Offline', { status: 503, statusText: 'Offline' });
}

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
  if (url.pathname.startsWith('/api/') ||
      url.pathname.startsWith('/auth/') ||
      url.pathname.startsWith('/login') ||
      url.pathname.startsWith('/verify-2fa') ||
      url.pathname.startsWith('/logout')) return;

  if (event.request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          const res = await fetch(event.request);
          putRuntime(event.request, res).catch(() => {});
          return res;
        } catch (e) {
          return offlineFallback(event.request);
        }
      })()
    );
    return;
  }

  if (url.origin === self.location.origin) {
    event.respondWith(
      (async () => {
        const cached = await caches.match(event.request);
        if (cached) return cached;
        try {
          const res = await fetch(event.request);
          putRuntime(event.request, res).catch(() => {});
          return res;
        } catch (e) {
          return offlineFallback(event.request);
        }
      })()
    );
    return;
  }

  // CDN runtime cache (stale-while-revalidate)
  event.respondWith(
    (async () => {
      const cached = await caches.match(event.request);
      const networkFetch = fetch(event.request).then((res) => {
        putRuntime(event.request, res).catch(() => {});
        return res;
      }).catch(() => cached);
      return cached || networkFetch;
    })()
  );
});
