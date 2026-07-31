// Temel PWA gereksinimi için boş servis sürücüsü
self.addEventListener('install', (e) => {
  self.skipWaiting();
});

self.addEventListener('fetch', (e) => {
  // Talepleri normal şekilde internetten çeker
});
