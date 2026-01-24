
        window.__turnstileApiLoaded = false;
        window.onTurnstileLoad = () => {
            window.__turnstileApiLoaded = true;
            if (window.initTurnstileWidget) window.initTurnstileWidget();
        };
    