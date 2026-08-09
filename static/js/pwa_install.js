(() => {
  const BUTTON_SELECTOR = '[data-pwa-install-button]';
  const standaloneMedia = window.matchMedia
    ? window.matchMedia('(display-mode: standalone)')
    : null;
  const isStandalone = () => (
    standaloneMedia && standaloneMedia.matches
  ) || window.navigator.standalone === true;

  let deferredPrompt = null;
  let buttons = [];
  let orientationLockPending = false;

  const lockPwaOrientation = async () => {
    if (!isStandalone() || document.visibilityState === 'hidden' || orientationLockPending) return;
    const orientation = window.screen && window.screen.orientation;
    if (!orientation || typeof orientation.lock !== 'function') return;

    orientationLockPending = true;
    try {
      await orientation.lock('portrait-primary');
    } catch (_) {
      // Unsupported platforms (including iOS) fall back to the manifest setting.
    } finally {
      orientationLockPending = false;
    }
  };

  const hideButtons = () => {
    buttons.forEach((button) => button.classList.add('hidden'));
  };

  const showButtons = () => {
    buttons.forEach((button) => button.classList.remove('hidden'));
  };

  const refreshButtons = () => {
    if (!buttons.length) return;
    if (isStandalone()) {
      hideButtons();
      return;
    }
    if (deferredPrompt) {
      showButtons();
    } else {
      hideButtons();
    }
  };

  const registerServiceWorker = async () => {
    if (!('serviceWorker' in navigator) || !window.isSecureContext) return;
    // The chat screen owns its optional runtime-cache setting. Registering here
    // while that setting is off caused a register/unregister race on every reload.
    if (window.CHAT_CONFIG && window.CHAT_CONFIG.useSwCache !== true) return;
    const appVersion = window.APP_VERSION || '';
    const swUrl = appVersion ? `/sw.js?v=${encodeURIComponent(appVersion)}` : '/sw.js';
    try {
      await navigator.serviceWorker.register(swUrl);
    } catch (error) {
      console.warn('PWA service worker registration failed:', error);
    }
  };

  const handleInstallClick = async (event) => {
    event.preventDefault();
    if (!deferredPrompt) return;
    try {
      deferredPrompt.prompt();
      await deferredPrompt.userChoice;
    } catch (error) {
      console.warn('PWA install prompt failed:', error);
    } finally {
      deferredPrompt = null;
      refreshButtons();
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    buttons = Array.from(document.querySelectorAll(BUTTON_SELECTOR));
    buttons.forEach((button) => {
      button.addEventListener('click', handleInstallClick);
    });
    refreshButtons();
    registerServiceWorker();
    lockPwaOrientation();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') lockPwaOrientation();
  });

  window.addEventListener('pageshow', lockPwaOrientation);

  if (standaloneMedia && typeof standaloneMedia.addEventListener === 'function') {
    standaloneMedia.addEventListener('change', lockPwaOrientation);
  }

  window.addEventListener('beforeinstallprompt', (event) => {
    event.preventDefault();
    deferredPrompt = event;
    refreshButtons();
  });

  window.addEventListener('appinstalled', () => {
    deferredPrompt = null;
    hideButtons();
    lockPwaOrientation();
  });
})();
