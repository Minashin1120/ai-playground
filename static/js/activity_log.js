/*
 * ログの収集を強化 (activity log).
 *
 * While the user turns it on (settings > フィードバック), this records the browser's own
 * operations to localStorage: clicks (element ids/classes, never text), setting switches,
 * requests (method, path without query, status, time), modals, toasts, page lifecycle,
 * errors and console warnings. It never records what was typed, prompts, answers or file
 * contents. Sending a feedback attaches entries within the configured retention window (`window.ActivityLog.recent()`).
 *
 * The log is not part of the site cache: clearing the cache keeps it, and it is removed
 * on expiry, by the "ログを削除" button, turning the setting off, or another account signing in.
 */
(function () {
    'use strict';
    if (window.ActivityLog) return;

    var POLICY = {"version":3,"events":["activity.created","activity.new_intent","activity_log.disabled","activity_log.enabled","app.crash","app.foreground","app.process_start","app.start","app_update.state","attach.stored","branch.cancelled","branch.done","branch.error","branch.start","change","click","console.error","console.warn","direct","direct.cancel","direct.error","direct.read_error","error","error.unhandled_rejection","feedback.chat_file","feedback.images","fetch","fetch.error","gemini.file_state","gemini.first_chunk","gemini.image_end","gemini.payload","gemini.stream_end","gemini.upload_cancelled","gemini.upload_done","gemini.upload_error","gemini.upload_sent","gemini.upload_session","gemini.upload_start","gen.after_answer_sync","gen.attachment_loaded","gen.cancelled","gen.conversation","gen.done","gen.error","gen.route","gen.saved_question","gen.start","gen.turns","heartbeat","html_preview.cancelled","html_preview.close","html_preview.done","html_preview.error","html_preview.start","http","http.cancel","http.error","http.read_error","image_split.cancelled","image_split.done","image_split.error","image_split.start","key","lyria.state","media.close","media.ended","media.error","media.pause","media.play","media.ready","media.start","media.waiting","nav.hashchange","nav.popstate","nav.pushState","nav.replaceState","net.offline","net.online","no_progress","page.hide","page.load","page.loaded","page.resize","page.show","page.visibility","perf.longtask","provider.ideogram.cancelled","provider.ideogram.done","provider.ideogram.error","provider.ideogram.start","realtime.state","resource.error","send.cancelled","send.done","send.error","send.event","send.rejoin","send.start","send.stop","send.stopped","send.stream.done","send.stream.error","send.stream.first","send.stream.start","server.client_log","server.request.end","server.request.start","server.stream.end","server.stream.error","server.stream.first","server.worker.end","server.worker.start","share.text_shape","stall.stacks","state","store.file_read","store.file_written","store.lock_wait","submit","sw.controllerchange","sync.attachment_loaded","sync.attachment_uploaded","sync.cancelled","sync.done","sync.engine_start","sync.error","sync.pending","sync.prepared","sync.pushed","sync.skipped","sync.start","thread.delete","thread.deleted","thread.fetch","thread.fetched","thread.open","thread.open_cancelled","thread.open_error","thread.opened","thread.refresh","toast","tool.create_file.cancelled","tool.create_file.done","tool.create_file.error","tool.create_file.start","tool.edit_file.cancelled","tool.edit_file.done","tool.edit_file.error","tool.edit_file.start","tool.python.cancelled","tool.python.done","tool.python.error","tool.python.start","translation.cancelled","translation.done","translation.error","translation.start","ui.drawer","ui.hide","ui.interaction","ui.panels","ui.press","ui.selection_windows","ui.settings_tab","ui.show","ui.tap","ui.tap_cancel","ui.viewer","ui.viewer.image","ui.viewer.tap","ui.viewer.zoom","xhr"],"numbers":["after_ms","arguments","attachment_bytes","attachments","away_ms","bytes","chars","chunks","clip_html_len","clip_items","col","consumed_after_ms","content_chars","content_width","count","dom_ms","down_consumed_after_ms","dpr","dropped","duration","error_code","failed_files","files","from","h","handle_changes","heap_mb","height","html_len","images","index","inline_bytes","input_bytes","length","line","load_ms","long_press_ms","max_mb","messages","moved_dp","moved_x","moved_y","ms","nodes","output_bytes","overlap","pending","pending_count","pid","pieces","press_ms","press_x","press_y","quiet_ms","reason_code","rows","scale","scroll_max","scroll_x","scrolled_px","sdk","selection_changes","selection_height","selection_menus","selection_width","seq","size","slop_dp","status","stored_bytes","t","text_chars","text_len","thread_id","threads","to","toolbar_creates","toolbar_destroys","toolbar_hides","toolbar_requests","turns","upload_completed","upload_count","uploaded","uploadedMessages","uploaded_files","used_mb","version_code","w","wait_ms","width","x","y"],"booleans":["accepted","action_mode_active","advanced","alpha","alt","api_key_prompt","attach_menu","attached","auth_busy","auth_error","banned","batch","batch_banner","batch_busy","batch_modal","branch","busy","cache_syncing","camera","cancelled","canvas","changelog","checked","clipped","closes","coding","compression","connection_message","ctrl","current","delete_thread","down_consumed","draft_empty","editing","failed","feedback_busy","feedback_modal","file_creation","file_view","focused","foreground","from_history","gem","gem_delete","gem_editor","gems_busy","has_older","hidden","history","item_changed","kept_running","legal","library","library_busy","library_failed","live_accepted","live_error","live_search","loaded","local_profile","locked","low_bandwidth","lyria","manual","maps","marker","mcp","mcp_busy","mcp_decision","meta","model_picker","new_thread","notice","offline","ok","older","on_image","online","open","pairing","persisted","prefs_busy","prompt_cache","python","quote","ready","realtime","redirected","rejected","rename","restored","retry","rich_paste","running","search","server_checking","server_error","serverless","settings","settings_modal","setup_required","shift","signed_in","standalone","starting","status","streaming","sync_message","syncing","system_prompt","temporary","thinking","thread_load_failed","thread_loading","thread_settings","token_modal","turnstile","up_consumed","uploading","url_context","vertex","vision_picker","window_focused","x_link_prompt"],"enums":{"method":["GET","POST","PUT","PATCH","DELETE","HEAD","OPTIONS"],"kind":["error","warning","success","info"],"state":["start","open","close","loaded","failed","loading","done","cancelled","visible","hidden","pending","ACTIVE","PROCESSING","FAILED"],"provider":["openai","gemini","google","xai","grok","deepseek","kimi","mistral","ideogram","zai","vertex_ai","unknown"],"type":["status","content","thought","done","error","accepted","message","file","image","audio","video","search","python","coding_diff","mcp","start","stop"],"key":["Enter","Escape","c","v","a","z","y","s","/",". ",","],"connection":["ONLINE","OFFLINE","CONNECTING","DEGRADED"],"source":["camera","share","picker","clipboard","library","unknown"],"phase":["start","done","error","cancelled"],"mime":["image/png","image/jpeg","image/webp","image/gif","image/svg+xml","video/mp4","video/webm","audio/wav","audio/mpeg","audio/mp4","audio/webm","application/pdf","application/json","text/plain","text/html","text/markdown","application/octet-stream"],"nav_type":["navigate","reload","back_forward","prerender"],"screen":["main","preview","translation","image_split"],"action":["ACTION_SEND","ACTION_SEND_MULTIPLE","ACTION_PROCESS_TEXT","android.intent.action.SEND","android.intent.action.SEND_MULTIPLE","android.intent.action.PROCESS_TEXT","android.intent.action.VIEW","android.intent.action.MAIN"],"tab":["general","model","display","prompt","account","security","data","feedback","mcp","android","connection","api","2fa"],"control":["account-dedupe-btn","account-dedupe-result","account-export-btn","account-export-download-btn","account-export-expiry","account-export-ready","account-export-ready-text","account-import-btn","account-import-categories","account-import-file","account-import-inplace","account-import-inplace-warning","account-import-result","account-import-settings-bypass","account-transfer-cancel-btn","account-transfer-progress","account-transfer-progress-bar","account-transfer-progress-detail","account-transfer-progress-percent","account-transfer-progress-text","active-gem-name","activity-log-feedback-status","activity-log-usage-text","add-gem-btn","add-gem-fixed-prompt-row","admin-api-key-mode-note","admin-api-key-mode-status","admin-api-key-mode-toggle","admin-enc-card","admin-enc-list","admin-enc-load","alpha-bar","alpha-info-modal","api-key-modal-cancel-btn","api-key-modal-desc","api-key-modal-fallback-btn","api-key-modal-input","api-key-modal-key-label","api-key-modal-model-name","api-key-modal-save-btn","api-key-required-modal","app-global-sys-prompt-preview","app-global-sys-prompt-preview-status","audio-gen-options","auth-container","auto-search-banner","auto-search-msg","auto-search-off-btn","auto-search-on-btn","auto-search-remember","back-to-chat","back-top-btn","background-blur-mode-status","ban-appeal-count","ban-appeal-list","ban-appeal-mark-read","ban-appeal-refresh","batch-count","batch-filter-tabs","batch-list","batch-manage-btn","batch-modal","batch-mode-container","batch-notification-banner","batch-notification-close","batch-notification-open","batch-notification-text","batch-refresh-btn","bot-admin-close","bot-admin-list","bot-admin-modal","bot-admin-open","bot-admin-refresh-btn","bot-admin-search","bot-admin-search-btn","bot-log-account-list","bot-log-close","bot-log-detail","bot-log-list-toolbar","bot-log-list-view","bot-log-modal","bot-log-open","bot-log-refresh-btn","bot-log-search","bot-log-search-btn","bot-speed-test-btn","bot-speed-test-result","bot-status","bot-unban-btn","bot-unban-linked-btn","bot-unban-result","bot-unban-username","br-date","br-delete-btn","br-fix-btn","br-id","br-model","br-name-input","br-save-name-btn","br-switch-btn","br-tokens","branch-detail-panel","branch-empty-panel","branch-manage-btn","branch-modal","branch-model-breakdown","branch-total-tokens","branch-tree-canvas","branch-tree-container","browser-fast-mode-cancel-btn","browser-fast-mode-container","browser-fast-mode-enable-btn","browser-fast-mode-ignore-row","browser-fast-mode-ignore-warning","browser-fast-mode-key-description","browser-fast-mode-modal","browser-fast-mode-warning","btn-tab-2fa","btn-tab-account","btn-tab-api","btn-tab-data","btn-tab-display","btn-tab-feedback","btn-tab-general","btn-tab-mcp","btn-tab-prompt","btn-tab-security","byok","camera-attach-btn","camera-canvas","camera-capture-btn","camera-capture-modal","camera-capture-preview-list","camera-clear-btn","camera-fallback-btn","camera-modal-close","camera-split-enabled","camera-status","camera-switch-btn","camera-video","cancel-edit-btn","canvas-block-count","canvas-block-list","canvas-block-shell","canvas-code-text","canvas-coding","canvas-mode-container","canvas-panel","canvas-panel-clear-btn","canvas-panel-close-btn","canvas-panel-copy-btn","canvas-panel-status","canvas-panel-tabs","canvas-panel-title","canvas-preview-empty","canvas-preview-frame","canvas-preview-lang","canvas-preview-shell","canvas-source-scroll","canvas-source-select","canvas-source-shell","changelog-container","chat-container","chat-stage","chat-transition-veil","chat-upload-status","clear-activity-log-btn","clear-coding-target-btn","clear-file-btn","clear-mask-btn","clear-prompt-cache-btn","clear-site-cache-btn","close-settings-btn","coding-mode-container","coding-target-bar","coding-target-text","compression-modal","compression-option","convenience","conversation-stage","delete-account-btn","disable-2fa-btn","drop-overlay","e2ee-card","easy-login-cancel","easy-login-code","easy-login-exp","easy-login-generate","easy-login-mins","easy-login-result","edit-bar","enable-batch-mode","enable-browser-fast-mode","enable-canvas-mode","enable-coding-mode","enable-compression","enable-file-creation","enable-maps","enable-mcp","enable-prompt-cache","enable-python","enable-search","enable-sys-prompt","enable-temporary-chat","enable-thinking","enable-url-context","enc-scan-all","enc-scan-result","enc-scan-thread","encryption-status-admin-actions","encryption-status-admin-toggle","encryption-status-body","encryption-status-modal","encryption-status-title","faq","fast-mode","fb-admin-list","fb-admin-panel","fb-attach-chat","fb-attach-chat-note","fb-images-add","fb-images-input","fb-images-list","fb-list","fb-message","fb-submit","fb-title","file-creation-container","file-input","file-name","file-preview","file-preview-thumbs","file-viewer","file-viewer-body","file-viewer-title","fixed-prompts-bar","flash-msg","form-totp","form-webauthn","g_id_onload","gem-active-indicator","gem-default-model","gem-desc","gem-fixed-prompts-container","gem-inst","gem-list","gem-modal","gem-modal-title","gem-name","gem-pull-indicator","gem-suggestions","gem-suggestions-list","gemini-backend-note","gemini-backend-status","gemini-backend-toggle","gemini-image-aspect","gemini-image-options","gemini-image-size","gemini-local-python-cancel","gemini-local-python-close","gemini-local-python-continue","gemini-local-python-dont-show","gemini-local-python-modal","gemini-music-options","gemini-vertex-settings","gemini-video-aspect","gemini-video-duration","gemini-video-duration-wrap","gemini-video-options","gemini-video-resolution","google-action-area","google-email-text","google-link-icon","google-link-status","google-link-text","gpt-image-compression","gpt-image-compression-wrap","gpt-image-format","gpt-image-options","gpt-image-quality","gpt-image-size","grok-image-aspect","grok-image-count","grok-image-format","grok-image-options","grok-image-quality","grok-image-resolution","grok-video-aspect","grok-video-duration","grok-video-options","grok-video-resolution","help-toc","help-toc-links","help-toc-title","history","history-modal","history-modal-btn","history-modal-thread-list-container","history-search-box","ideogram-image-aspect","ideogram-image-count","ideogram-image-magic","ideogram-image-negative","ideogram-image-note","ideogram-image-operation", "ideogram-image-options","ideogram-image-quality","ideogram-image-resolution","ideogram-image-seed","ideogram-image-speed","ideogram-image-style","image-input-limits","image-split-attach","image-split-cancel","image-split-compression-close","image-split-compression-enabled","image-split-compression-modal","image-split-compression-save","image-split-count","image-split-dimensions","image-split-downloads","image-split-input","image-split-mark-original","image-split-mark-pieces","image-split-modal","image-split-next","image-split-original","image-split-overlap","image-split-page","image-split-pieces","image-split-prev","image-split-preview","image-split-save","image-split-settings","image-split-status","image-viewer","image-viewer-img","image-viewer-meta","image-voice","import-files-cancel","import-files-close","import-files-confirm","import-files-grid","import-files-info","import-files-modal","import-files-none","import-files-select-all","import-files-summary","initial-logs-data","initial-theme-vars","input-container","input-row","landing-demo-chat","landing-demo-hub","landing-demo-hub-status","legal-content","legal-modal","legal-title","lib-attach-btn","lib-btn","lib-del-btn","lib-download-btn","lib-favorite-filter-btn","lib-grid","lib-load-more-btn","lib-modal","lib-rename-btn","lib-search","lib-sort","lib-total-count","lib-usage-btn","lib-usage-list","lib-usage-modal","lib-usage-title","loading-trigger","login-error","login-form","login-password","login-section","low-bandwidth-status-pill","low-bandwidth-toggle-btn","lyria-add-prompt-btn","lyria-apply-config-btn","lyria-apply-prompts-btn","lyria-bpm","lyria-bpm-label","lyria-brightness","lyria-brightness-label","lyria-density","lyria-density-label","lyria-elapsed","lyria-guidance","lyria-guidance-label","lyria-mode","lyria-mute-bass","lyria-mute-drums","lyria-only-bass-drums","lyria-open-studio-btn","lyria-pause-btn","lyria-play-btn","lyria-prompt-rows","lyria-realtime-studio-bar","lyria-reset-btn","lyria-save-btn","lyria-scale","lyria-status-dot","lyria-status-text","lyria-stop-btn","lyria-studio-close","lyria-studio-modal","lyria-studio-status","lyria-temperature","lyria-temperature-label","manual-theme-light-css","maps-grounding-container","marker-attach-original","marker-canvas","marker-clear","marker-color-picker","marker-crop-canvas","marker-crop-reset","marker-image","marker-modal","marker-modal-close","marker-opacity","marker-opacity-number","marker-opacity-value","marker-save","marker-size","marker-stage","marker-tool-crop","marker-tool-draw","marker-tool-hint","marker-tool-mosaic","marker-toolbar","marker-undo","marker-view-reset","marker-viewport","mask-btn","mask-input","mask-name","mask-preview","mcp-add-server-btn","mcp-container","mcp-custom-auth","mcp-custom-bearer","mcp-custom-bearer-wrap","mcp-custom-desc","mcp-custom-name","mcp-custom-status","mcp-custom-url","mcp-decision-allow","mcp-decision-args","mcp-decision-deny","mcp-decision-modal","mcp-decision-server","mcp-decision-tool","mcp-google-client-id","mcp-google-client-secret","mcp-google-client-state","mcp-google-oauth-card","mcp-save-google-client-btn","mcp-server-count","mcp-server-list","mcp-status-msg","menu-btn","mic-btn","mic-recording-indicator","mic-recording-text","mic-waveform","mig-progress-bar","mig-progress-text","mig-status-box","migration","minashin-action-area","minashin-email-text","minashin-g-login","minashin-g-setting","minashin-g-signup","minashin-link-icon","minashin-link-status","minashin-link-text","minashin-m-setting","minimal-options-backdrop","minimal-options-close-btn","minimal-options-items","minimal-options-model-body","minimal-options-model-section","minimal-options-panel","minimal-options-popup","mistral-ocr-options","mobile-chat-temporary-label","mobile-chat-title","mobile-chat-ttl","mobile-new-chat-btn","modal-gemini-image-aspect","modal-gemini-image-options","modal-gemini-image-size","modal-gpt-image-compression","modal-gpt-image-format","modal-gpt-image-options","modal-gpt-image-quality","modal-gpt-image-size","modal-grok-image-aspect","modal-grok-image-options","modal-grok-image-quality","modal-grok-image-resolution","modal-ideogram-image-aspect","modal-ideogram-image-count","modal-ideogram-image-magic","modal-ideogram-image-negative","modal-ideogram-image-operation", "modal-ideogram-image-options","modal-ideogram-image-quality","modal-ideogram-image-resolution","modal-ideogram-image-seed","modal-ideogram-image-speed","modal-ideogram-image-style","modal-mistral-ocr-options","modal-ocr-extract-footer","modal-ocr-extract-header","modal-ocr-include-blocks","modal-ocr-include-images","modal-ocr-pages","modal-ocr-table-format","model-api-key-apply-btn","model-api-key-input","model-api-key-list","model-api-key-model","model-api-keys-panel","model-list-container","model-modal","model-search","model-search-clear","model-select","model-selector-btn","model-selector-text","model-tag-bar","models","multimodal","music-instrumental","new-chat-btn","no-more","ocr-extract-footer","ocr-extract-header","ocr-include-blocks","ocr-include-images","ocr-pages","ocr-table-format","offline-banner","offline-banner-icon","offline-banner-text","overlay","passkey-count","passkey-error","passkey-list","passkey-login-btn","passkey-only-note","passkey-username","photo-input","prompt-cache-container","prompt-controls-row","prompt-controls-toggle-btn","prompt-controls-toggle-icon","prompt-controls-toggle-text","prompt-details-controls","prompt-input","prompt-primary-controls","prompt-token-estimate","python-container","python-exec-modal","python-exec-modal-body","python-exec-modal-title","quote-bar","quote-confirm-btn","quote-popover","quote-popover-container","quote-text-display","reasoning-effort","reasoning-effort-container","register-webauthn-btn","remember","reset-global-sys-prompt","reset-thread-sys-prompt","rich-paste-btn","rich-paste-cancel-btn","rich-paste-capture","rich-paste-clear-btn","rich-paste-close-btn","rich-paste-download-only","rich-paste-focus-btn","rich-paste-import-btn","rich-paste-modal","rich-paste-modal-close","rich-paste-preview-btn","rich-paste-progress-bar","rich-paste-progress-container","rich-paste-progress-label","rich-paste-progress-text","rich-paste-prompt","rich-paste-send-btn","rich-paste-send-server-btn","rich-paste-status","rich-paste-storage","rich-paste-use-default","safety-option","safety-setting","save-compression-settings-btn","save-gem-btn","save-settings-btn","save-thread-settings-btn","scroll-sentinel","scroll-to-bottom-btn","search-box","search-clear","search-container","search-input","search-status","security","send-btn","session-list","session-refresh-btn","session-revoke-all-btn","session-revoke-others-btn","set-activity-log","set-admin-api-key-mode","set-anthropic","set-apply-global-sys-prompt","set-auto-search-links","set-background-blur-mode","set-bot-detect","set-bot-detect-global","set-clear-cache-on-version-update","set-client-debug-log","set-compact-prompt-mode","set-deepseek","set-default-2fa-method","set-default-file-creation","set-default-maps","set-default-mcp","set-default-model","set-default-python","set-default-reasoning-effort","set-default-safety","set-default-search","set-default-sys-prompt","set-default-thinking","set-default-thinking-budget","set-default-thinking-level","set-default-url-context","set-default-vision-model","set-e2ee","set-enter-to-send","set-gemini","set-gemini-backend","set-gemini-local-python-dialog","set-gemini-vertex-location","set-gemini-vertex-project","set-global-sys-prompt-enabled","set-google-key","set-google-project","set-ideogram","set-kimi","set-latency-metrics","set-light-mode","set-liquid-glass","set-mic-transcribe-mode","set-minimal-prompt-mode","set-mistral","set-openai","set-passkey-only-login","set-password","set-prompt-bar-mode-normal","set-server-diagnostic-log","set-skip-2fa-google","set-stt-model","set-theme-color","set-theme-color-text","set-use-last-settings","set-use-sw-cache","set-username","set-voice-studio-ui","set-xai","set-zai","settings-btn","settings-confirmation-cancel","settings-confirmation-close","settings-confirmation-confirm","settings-confirmation-count","settings-confirmation-list","settings-confirmation-modal","settings-header-close","settings-modal","settings-search","settings-search-clear","settings-tabs","settings-tabs-arrow-left","settings-tabs-arrow-right","settings-tabs-wrap","setup-account-import-btn","setup-account-import-categories","setup-account-import-file","setup-account-import-result","setup-account-import-settings-bypass","setup-account-transfer-cancel-btn","setup-account-transfer-progress","setup-account-transfer-progress-bar","setup-account-transfer-progress-detail","setup-account-transfer-progress-percent","setup-account-transfer-progress-text","setup-form","setup-gemini-backend","setup-import-files-cancel","setup-import-files-close","setup-import-files-confirm","setup-import-files-grid","setup-import-files-info","setup-import-files-modal","setup-import-files-none","setup-import-files-select-all","setup-import-files-summary","setup-settings-confirmation-cancel","setup-settings-confirmation-close","setup-settings-confirmation-confirm","setup-settings-confirmation-count","setup-settings-confirmation-list","setup-settings-confirmation-modal","setup-vertex-fields","show-password-toggle","sidebar","sidebar-chat-temporary-label","sidebar-chat-title","sidebar-chat-ttl","sidebar-resize-handle","sidebar-toggle-btn","sidebar-toggle-icon","signup-link-section","signup-password","site-cache-usage-detail","site-cache-usage-refresh","site-cache-usage-text","slash-command-cancel-btn","slash-command-indicator","slash-command-list","slash-command-name","slash-command-suggestions","spinner","standard-chat-controls","step-1","step-2","step-3","step-num","storage-usage-bar","storage-usage-refresh","storage-usage-text","sts-auto-play","sts-auto-play-wrap","sts-auto-restart","sts-auto-send","sts-cancel-btn","sts-custom-vocab","sts-custom-vocab-wrap","sts-hint","sts-include-thoughts","sts-lang-wrap","sts-live-transcript","sts-mic-btn","sts-mode-label","sts-note","sts-options","sts-panel","sts-rate-in","sts-rate-out","sts-rate-wrap","sts-reasoning-effort","sts-reasoning-wrap","sts-settings","sts-settings-toggle","sts-silence-sec","sts-speed","sts-speed-label","sts-speed-wrap","sts-status","sts-target-lang","sts-thinking-level","sts-thinking-wrap","sts-transcribe-mode","sts-transcribe-wrap","sts-voice","sts-voice-wrap","success-screen","sys-prompt-option","sys-prompt-text","tab-2fa","tab-account","tab-api","tab-data","tab-display","tab-feedback","tab-general","tab-mcp","tab-prompt","tab-security","tab-totp","tab-webauthn","temporary-chat-container","temporary-chat-controls","temporary-chat-welcome-desc","theme-presets","theme-reset-btn","thinking","thinking-budget","thinking-budget-label","thinking-level","thinking-options","thinking-slide-bar","thinking-slide-close-btn","thinking-slide-inner","thinking-slide-scale","thinking-slide-value","thinking-slider","thread-app-global-sys-prompt-preview","thread-app-global-sys-prompt-preview-status","thread-apply-global-sys-prompt","thread-custom-instruction","thread-global-sys-prompt","thread-global-sys-prompt-enabled","thread-include-global-instruction","thread-list","thread-modal","thread-pull-indicator","toast-stack","toggle-model-api-keys-btn","token-detail-content","token-detail-encrypted","token-detail-in","token-detail-modal","token-detail-out","token-detail-thought","token-detail-title","token-detail-total","tools","top-model-bar","total-token-bar","total-token-count","total-token-count-all-branches","totp-error","totp-form","totp-input","totp-qr","totp-secret-disp","totp-section","totp-setup-area","totp-verify-code","tts-language","tts-language-wrap","tts-speed","tts-speed-label","tts-speed-note","tts-speed-wrap","tts-style","tts-style-wrap","tts-voice","tts-voice-custom","tts-voice-custom-wrap","turnstile-container","upload-btn","upload-camera-btn","upload-clear-btn","upload-dropzone","upload-lib-btn","upload-list","upload-modal","upload-modal-close","upload-modal-status-text","upload-modal-total-progress","upload-modal-total-progress-bar","upload-photo-btn","upload-select-btn","upload-split-btn","upload-total-progress","upload-total-progress-bar","url-context-container","version-display","version-update-clear-cache","version-update-dismiss","version-update-modal","version-update-reload","viewer-edit-image","vision-model-change-btn","vision-model-display","vision-model-info","voice-studio-close","voice-studio-file-host","voice-studio-modal","voice-studio-open-btn","voice-studio-panel-host","voice-studio-title","voice-studio-transcript","webauthn-form","webauthn-name","welcome-default-content","welcome-quick-start","welcome-screen","welcome-temporary-content","xai-chat-options","xai-frequency-penalty","xai-logprobs","xai-max-completion-tokens","xai-parallel-tool-calls","xai-presence-penalty","xai-response-format","xai-seed","xai-stop","xai-temperature","xai-tool-choice","xai-top-logprobs","xai-top-p"],"model":["antigravity-preview-05-2026","attach","batch-modal","bot-admin-modal","bot-log-modal","branch-modal","camera-capture-modal","canvas","claude-opus-4-6","claude-sonnet-4-6","coding","compress","compression-modal","deep-research-max-preview-04-2026","deep-research-preview-04-2026","deepseek-v4-flash","deepseek-v4-flash-0731","deepseek-v4-flash-vision-exp","deepseek-v4-pro","deepseek-v4.1-flash","encryption-status-modal","fast","file","gem-modal","gemini-2.5-computer-use-preview-10-2025","gemini-2.5-flash","gemini-2.5-flash-image","gemini-2.5-flash-lite","gemini-2.5-flash-native-audio-preview-12-2025","gemini-2.5-flash-preview-tts","gemini-2.5-pro","gemini-2.5-pro-preview-tts","gemini-3-flash-preview","gemini-3-pro-image","gemini-3-pro-image-preview","gemini-3-pro-preview","gemini-3.1-flash-image","gemini-3.1-flash-image-preview","gemini-3.1-flash-lite","gemini-3.1-flash-lite-image","gemini-3.1-flash-lite-preview","gemini-3.1-flash-live-preview","gemini-3.1-flash-tts-preview","gemini-3.1-pro-preview","gemini-3.5-flash","gemini-3.5-flash-lite","gemini-3.5-live-translate-preview","gemini-3.5-transcribe","gemini-3.5-transcribe-live","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash","gemini-3.8-flash-cyber","gemini-3.8-flash-lite-tts","gemini-3.8-flash-tts","gemini-3.8-live","gemini-3.8-live-extended-thinking","gemini-4-argon","gemini-embedding-2","gemini-nano-banana-2.1","gemini-omni-1.1-flash","gemini-omni-flash","gemini-robotics-er-2-preview","glm-4-32b-0414-128k","glm-4.5","glm-4.5-air","glm-4.5-airx","glm-4.5-flash","glm-4.5-x","glm-4.5v","glm-4.6","glm-4.6v","glm-4.6v-flash","glm-4.6v-flashx","glm-4.7","glm-4.7-flash","glm-4.7-flashx","glm-5","glm-5.1","glm-5.2","glm-5.3","glm-5.3-flash","glm-5.3-flashx","google-tts-neural","google-tts-studio","gpt-4o","gpt-4o-mini","gpt-4o-mini-tts","gpt-5-mini","gpt-5-search-api","gpt-5.1","gpt-5.2","gpt-5.4","gpt-5.4-mini","gpt-5.4-nano","gpt-5.4-pro","gpt-5.5","gpt-5.5-mini","gpt-5.5-nano","gpt-5.5-pro","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra","gpt-6-luna","gpt-6-sol","gpt-6.1-sol","gpt-6.1-terra","gpt-image-1","gpt-image-1-mini","gpt-image-1.5","gpt-image-2","gpt-image-2.5-flare","gpt-image-2.5-sunburst","gpt-live-1","gpt-live-transcribe","gpt-realtime","gpt-realtime-1.5","gpt-realtime-2","gpt-realtime-2.1","gpt-realtime-2.1-mini","gpt-realtime-mini","gpt-realtime-translate","gpt-realtime-whisper","gpt-transcribe","grok-4-1-fast-non-reasoning","grok-4-1-fast-reasoning","grok-4-fast-non-reasoning","grok-4-fast-reasoning","grok-4.20-0309-non-reasoning","grok-4.20-0309-reasoning","grok-4.20-multi-agent","grok-4.20-multi-agent-0309","grok-4.20-non-reasoning","grok-4.20-reasoning","grok-4.3","grok-4.5","grok-4.6","grok-build-0.1","grok-imagine-image","grok-imagine-image-2.0","grok-imagine-image-pro","grok-imagine-image-quality","grok-imagine-video","grok-imagine-video-1.5","grok-imagine-video-1.5-lite","grok-tts","grok-voice-3.0","grok-voice-agent","grok-voice-fast-1.0","grok-voice-latest","grok-voice-think-fast-1.0","grok-voice-think-fast-2.0","grok-voice-transcribe-2.0","grok-voice-transcribe-2.0-file","history-modal","ideogram-2.0","ideogram-2a","ideogram-3.0","ideogram-4.0","ideogram-4.5","kimi-k3","lib-modal","lyria-3-clip-preview","lyria-3-pro-preview","lyria-3.5","lyria-realtime-exp","maps","marker-modal","mcp","mistral-ocr-4-0","model-modal","options","paste","promptcache","python","python-exec-modal","rich-paste-modal","safety","search","settings","settings-modal","sysprompt","tempchat","thinking","thinking-high","thinking-low","thinking-mid","thinking-min","thinking-off","thread-modal","token-detail-modal","upload-modal","urls","veo-3.1-fast-generate-preview","veo-3.1-generate-preview","veo-3.1-lite-generate-preview","voice"],"vision_model":["antigravity-preview-05-2026","attach","batch-modal","bot-admin-modal","bot-log-modal","branch-modal","camera-capture-modal","canvas","claude-opus-4-6","claude-sonnet-4-6","coding","compress","compression-modal","deep-research-max-preview-04-2026","deep-research-preview-04-2026","deepseek-v4-flash","deepseek-v4-flash-0731","deepseek-v4-flash-vision-exp","deepseek-v4-pro","deepseek-v4.1-flash","encryption-status-modal","fast","file","gem-modal","gemini-2.5-computer-use-preview-10-2025","gemini-2.5-flash","gemini-2.5-flash-image","gemini-2.5-flash-lite","gemini-2.5-flash-native-audio-preview-12-2025","gemini-2.5-flash-preview-tts","gemini-2.5-pro","gemini-2.5-pro-preview-tts","gemini-3-flash-preview","gemini-3-pro-image","gemini-3-pro-image-preview","gemini-3-pro-preview","gemini-3.1-flash-image","gemini-3.1-flash-image-preview","gemini-3.1-flash-lite","gemini-3.1-flash-lite-image","gemini-3.1-flash-lite-preview","gemini-3.1-flash-live-preview","gemini-3.1-flash-tts-preview","gemini-3.1-pro-preview","gemini-3.5-flash","gemini-3.5-flash-lite","gemini-3.5-live-translate-preview","gemini-3.5-transcribe","gemini-3.5-transcribe-live","gemini-3.6-flash","gemini-3.7-flash","gemini-3.8-flash","gemini-3.8-flash-cyber","gemini-3.8-flash-lite-tts","gemini-3.8-flash-tts","gemini-3.8-live","gemini-3.8-live-extended-thinking","gemini-4-argon","gemini-embedding-2","gemini-nano-banana-2.1","gemini-omni-1.1-flash","gemini-omni-flash","gemini-robotics-er-2-preview","glm-4-32b-0414-128k","glm-4.5","glm-4.5-air","glm-4.5-airx","glm-4.5-flash","glm-4.5-x","glm-4.5v","glm-4.6","glm-4.6v","glm-4.6v-flash","glm-4.6v-flashx","glm-4.7","glm-4.7-flash","glm-4.7-flashx","glm-5","glm-5.1","glm-5.2","glm-5.3","glm-5.3-flash","glm-5.3-flashx","google-tts-neural","google-tts-studio","gpt-4o","gpt-4o-mini","gpt-4o-mini-tts","gpt-5-mini","gpt-5-search-api","gpt-5.1","gpt-5.2","gpt-5.4","gpt-5.4-mini","gpt-5.4-nano","gpt-5.4-pro","gpt-5.5","gpt-5.5-mini","gpt-5.5-nano","gpt-5.5-pro","gpt-5.6-luna","gpt-5.6-sol","gpt-5.6-terra","gpt-6-astra","gpt-6-luna","gpt-6-sol","gpt-6.1-sol","gpt-6.1-terra","gpt-image-1","gpt-image-1-mini","gpt-image-1.5","gpt-image-2","gpt-image-2.5-flare","gpt-image-2.5-sunburst","gpt-live-1","gpt-live-transcribe","gpt-realtime","gpt-realtime-1.5","gpt-realtime-2","gpt-realtime-2.1","gpt-realtime-2.1-mini","gpt-realtime-mini","gpt-realtime-translate","gpt-realtime-whisper","gpt-transcribe","grok-4-1-fast-non-reasoning","grok-4-1-fast-reasoning","grok-4-fast-non-reasoning","grok-4-fast-reasoning","grok-4.20-0309-non-reasoning","grok-4.20-0309-reasoning","grok-4.20-multi-agent","grok-4.20-multi-agent-0309","grok-4.20-non-reasoning","grok-4.20-reasoning","grok-4.3","grok-4.5","grok-4.6","grok-build-0.1","grok-imagine-image","grok-imagine-image-2.0","grok-imagine-image-pro","grok-imagine-image-quality","grok-imagine-video","grok-imagine-video-1.5","grok-imagine-video-1.5-lite","grok-tts","grok-voice-3.0","grok-voice-agent","grok-voice-fast-1.0","grok-voice-latest","grok-voice-think-fast-1.0","grok-voice-think-fast-2.0","grok-voice-transcribe-2.0","grok-voice-transcribe-2.0-file","history-modal","ideogram-2.0","ideogram-2a","ideogram-3.0","ideogram-4.0","ideogram-4.5","kimi-k3","lib-modal","lyria-3-clip-preview","lyria-3-pro-preview","lyria-3.5","lyria-realtime-exp","maps","marker-modal","mcp","mistral-ocr-4-0","model-modal","options","paste","promptcache","python","python-exec-modal","rich-paste-modal","safety","search","settings","settings-modal","sysprompt","tempchat","thinking","thinking-high","thinking-low","thinking-mid","thinking-min","thinking-off","thread-modal","token-detail-modal","upload-modal","urls","veo-3.1-fast-generate-preview","veo-3.1-generate-preview","veo-3.1-lite-generate-preview","voice"],"code_file":["AccountApi.kt","AccountTransfer.kt","ActivityLog.kt","AiSettings.kt","AnswerNotifications.kt","AnthropicDirect.kt","AppChangelogDialog.kt","AppChangelogViewModel.kt","AppUpdateChecker.kt","AppUpdateDialog.kt","AppUpdateDownloadManager.kt","AppUpdateDownloadService.kt","AppUpdateDownloader.kt","AppUpdateViewModel.kt","AttachmentExtractor.kt","BatchDialog.kt","BatchModels.kt","BatchNotifications.kt","BotTelemetry.kt","BranchManager.kt","BranchSnapshot.kt","BranchTree.kt","BrowserLoginPkce.kt","BubbleNotifications.kt","CanvasBlocks.kt","CanvasPanel.kt","CapturedPhotoOrientation.kt","ChatAutoScroll.kt","ChatBackend.kt","ChatCompletionsDirect.kt","ChatComponents.kt","ChatHeader.kt","ChatInstructionsDialog.kt","ChatNavigationHistory.kt","ChatViewModel.kt","CodingTargets.kt","ColorPickerDialog.kt","Composer.kt","ComposerRules.kt","CompressionDialog.kt","ConfirmDialogs.kt","ConnectionStatus.kt","Diagnostics.kt","DirectEngine.kt","DirectHttp.kt","DirectRouter.kt","EncryptedFileStore.kt","FileViewer.kt","GemDialog.kt","GeminiDirect.kt","GenerationOptions.kt","GenerationOptionsPanel.kt","GenerationService.kt","GenerationTracker.kt","GoogleAuthClient.kt","HtmlCodePreviewDialog.kt","ImageCompression.kt","ImageMarker.kt","ImageMarkerEditor.kt","ImageSplit.kt","ImageSplitCompression.kt","ImageSplitDialog.kt","ImageSplitOverlayActivity.kt","ImageSplitRenderer.kt","ImageSplitTileService.kt","ImportDialogs.kt","InAppCameraDialog.kt","InAppNoticeBanner.kt","InAppNoticeHost.kt","InAppNoticeState.kt","InlineAudioPlayer.kt","InteractionDiagnostics.kt","LatexText.kt","Library.kt","LibraryDialog.kt","LiveAnswer.kt","LocalChatBackend.kt","LocalChatStore.kt","LocalProfiles.kt","LocalSettingsStore.kt","LogPolicy.kt","LowBandwidth.kt","LyriaStudio.kt","MainActivity.kt","MarkdownText.kt","MathJaxRenderer.kt","MediaDirect.kt","MessageBubble.kt","MessageExtras.kt","MinimalOptions.kt","ModelPicker.kt","Models.kt","OfflineCacheStore.kt","OpenAiResponsesDirect.kt","PasskeyClient.kt","PlayIntegrityClient.kt","PlaygroundApi.kt","PlaygroundApplication.kt","PlaygroundDialog.kt","PlaygroundMotion.kt","PlaygroundScreen.kt","PlaygroundTheme.kt","ProcessTextActivity.kt","ProgressText.kt","QuoteSelection.kt","RealtimeModels.kt","RealtimeStudio.kt","ReauthDialog.kt","RemoteImage.kt","RichPasteDialog.kt","SelectionTranslationDialog.kt","SelectionTranslator.kt","ServerHistory.kt","ServerOrigin.kt","ServerlessSettings.kt","SettingsAccount.kt","SettingsDialog.kt","SettingsFields.kt","SettingsForm.kt","SettingsTabs.kt","SettingsTabsOther.kt","SharedTextMath.kt","SharedTextPreviewDialog.kt","Sidebar.kt","SlashCommands.kt","SseReader.kt","StartupSplash.kt","StartupSplashPolicy.kt","SyncEngine.kt","SystemPromptBuilder.kt","ThreadPdf.kt","TokenStore.kt","ToolbarNotification.kt","TranslateOverlayActivity.kt","TranslationSetting.kt","UploadSheet.kt","VertexAuth.kt","VibrationSetting.kt","WebComponents.kt","WebDialogs.kt","WebFonts.kt","WebModelCatalog.kt","WebTokens.kt","__init__.py","account_security.py","account_transfer.py","activity_log","agentic_media.py","app.py","app_settings_schema.py","background.py","chat_core","chat_state.py","create_file.py","crypto.py","diagnostic_logging.py","edit_file.py","external","feedback_chat_copy.py","feedback_images.py","ideogram.py","image_tools.py","logging_policy.py","lyria.py","mobile_auth.py","models.py","providers.py","realtime.py","request_hooks.py","request_identity.py","request_security.py","rich_paste_pdf.py","routes_account.py","routes_admin.py","routes_auth.py","routes_chat.py","routes_files.py","routes_media.py","routes_mobile.py","routes_mobile_account.py","routes_mobile_auth.py","routes_mobile_sync.py","routes_pages.py","routes_realtime.py","routes_settings.py","routes_threads_library.py","settings_ai.py","storage.py","temp_chat.py","thread_branches.py","token_utils.py","worker.py"],"error":["IOException","SocketException","SocketTimeoutException","TimeoutException","CancellationException","IllegalStateException","IllegalArgumentException","RuntimeException","OutOfMemoryError","TypeError","ReferenceError","SyntaxError","AbortError","NetworkError","Error"],"window":["PopupLayout","PopupDecorView","DecorView"],"realtime_state":["connecting","streaming","paused","stopped","error","closed","ready","recording","stopping"],"update_phase":["Checking","UpToDate","Available","Downloading","Ready","AwaitingInstallPermission","Installing","Error"],"ui_element":["message","math_block","inline_math"],"interaction_stage":["press","settled","focus","request","created","shown","hidden","destroyed","action","dispose","render","scroll"],"ui_action":["copy","quote","select_all"]},"routes":["/","/.well-known/assetlinks.json","/admin-bot-logs","/admin-bots","/android/auth/callback","/android/auth/google/callback","/android/auth/google/start","/android/auth/minashin/start","/android/connect","/android/integrity/turnstile","/android/integrity/turnstile/verify","/android/link/<provider>","/android/release-notes.md","/api/2fa/totp/enable","/api/2fa/totp/setup","/api/2fa/webauthn/register/options","/api/2fa/webauthn/register/verify","/api/2fa/webauthn/remove","/api/account/dedupe/execute","/api/account/dedupe/preview","/api/account/delete","/api/account/export","/api/account/export/<job_id>/download","/api/account/export/latest","/api/account/import","/api/account/import/upload/<upload_id>","/api/account/import/upload/<upload_id>/chunk","/api/account/import/upload/<upload_id>/complete","/api/account/import/upload/start","/api/account/transfer/<job_id>","/api/account/transfer/<job_id>/cancel","/api/account/unlink_google","/api/account/unlink_minashin","/api/admin/threads","/api/admin/threads/<thread_id>/encryption","/api/assets/fonts/japanese.ttf","/api/ban/appeal/status","/api/ban/appeals","/api/ban/appeals/mark_read","/api/ban/appeals/summary","/api/ban/appeals/update","/api/batch/jobs","/api/batch/jobs/<job_id>","/api/batch/jobs/<job_id>/cancel","/api/bot-telemetry","/api/bot/account/clear-identifiers","/api/bot/account/clear-lock","/api/bot/evidence","/api/bot/evidence/accounts","/api/bot/evidence/delete","/api/bot/lock","/api/bot/lock-status","/api/bot/turnstile-verify","/api/bot/unban","/api/bot/update","/api/bot/users","/api/browser_fast_mode/bootstrap","/api/browser_fast_mode/save","/api/changelogs","/api/changelogs/search","/api/client_log","/api/csrf_token","/api/debug/client_log","/api/easy_login","/api/encryption_scan","/api/feedback","/api/feedback/<fid>","/api/feedback/<fid>/chat_files","/api/feedback/<fid>/images","/api/feedback/<fid>/replies","/api/feedback/<fid>/replies/<int:reply_id>/images","/api/feedback/<fid>/update","/api/files","/api/files/delete","/api/files/favorite","/api/files/rename","/api/files/usage","/api/gemini/batch/status","/api/gemini/music/cancel","/api/gemini/music/command","/api/gemini/music/save","/api/gemini/music/start","/api/gemini/music/stream","/api/gemini/save_sts","/api/gemini/session","/api/gems","/api/gems/<string:gem_uuid>","/api/generate_title","/api/maintenance","/api/mcp/audit","/api/mcp/chat/<job_id>/decision","/api/mcp/oauth-client","/api/mcp/oauth/callback","/api/mcp/servers","/api/mcp/servers/<int:server_id>","/api/mcp/servers/<int:server_id>/auth/disconnect","/api/mcp/servers/<int:server_id>/auth/start","/api/mcp/servers/<int:server_id>/test","/api/mcp/servers/<int:server_id>/tools","/api/mcp/servers/<int:server_id>/tools/<path:tool_name>/permission","/api/messages/<int:mid>","/api/metrics/first_token","/api/mobile/v1/account/credentials","/api/mobile/v1/account/delete","/api/mobile/v1/account/easy-login","/api/mobile/v1/account/link/<provider>/start","/api/mobile/v1/auth/2fa/webauthn/options","/api/mobile/v1/auth/2fa/webauthn/verify","/api/mobile/v1/auth/exchange","/api/mobile/v1/auth/google","/api/mobile/v1/auth/login","/api/mobile/v1/auth/passkey/options","/api/mobile/v1/auth/passkey/verify","/api/mobile/v1/auth/signup","/api/mobile/v1/auth/totp","/api/mobile/v1/ban","/api/mobile/v1/ban/appeal","/api/mobile/v1/config","/api/mobile/v1/device","/api/mobile/v1/diagnostics","/api/mobile/v1/me","/api/mobile/v1/preferences","/api/mobile/v1/reauth","/api/mobile/v1/reauth/options","/api/mobile/v1/revoke","/api/mobile/v1/secrets/export","/api/mobile/v1/security","/api/mobile/v1/security/2fa/disable","/api/mobile/v1/security/e2ee","/api/mobile/v1/security/passkeys/options","/api/mobile/v1/security/passkeys/remove","/api/mobile/v1/security/passkeys/verify","/api/mobile/v1/security/preferences","/api/mobile/v1/security/totp/disable","/api/mobile/v1/security/totp/enable","/api/mobile/v1/security/totp/setup","/api/mobile/v1/security/turnstile","/api/mobile/v1/security/turnstile/complete","/api/mobile/v1/sessions","/api/mobile/v1/sessions/revoke","/api/mobile/v1/sessions/revoke_all","/api/mobile/v1/sessions/revoke_others","/api/mobile/v1/setup","/api/mobile/v1/sync/changes","/api/mobile/v1/sync/push","/api/mobile/v1/token","/api/realtime/audio","/api/realtime/cancel","/api/realtime/commit","/api/realtime/save","/api/realtime/start","/api/realtime/stream","/api/rich-paste/pdf","/api/sessions","/api/sessions/revoke","/api/sessions/revoke_all","/api/sessions/revoke_others","/api/settings","/api/settings/apply-ai-prompt","/api/speedtest/download","/api/speedtest/ping","/api/speedtest/upload","/api/stop_chat","/api/storage","/api/temporary_chat/heartbeat","/api/threads","/api/threads/<thread_id>","/api/threads/<thread_id>/bookmark","/api/threads/<thread_id>/branch","/api/threads/<thread_id>/settings","/api/threads/<thread_id>/title","/api/token_estimate","/api/version","/auth/minashin/callback","/ban/appeal","/banned","/batch","/branch","/c/<thread_id>","/c/<thread_id>/pdf","/camera","/changelog","/chat-settings","/chat_stream","/chat_stream_resume","/compression","/edit-image","/encryption-status","/favicon.ico","/files/<path:filename>","/files/thumb/<path:filename>","/gem","/help","/history","/library","/login","/login/google","/login/google/callback","/login/google/one-tap","/login/minashin","/login/passkey/options","/login/passkey/verify","/logout","/manifest.webmanifest","/model","/paste","/python-execution","/robots.txt","/settings","/setup","/signup","/static/<path:filename>","/sts","/sw.js","/synthesize","/token-details","/transcribe","/upload","/upload/chunk","/upload/complete","/upload/init","/verify-2fa","/verify-2fa/webauthn/options","/verify-2fa/webauthn/verify"],"identifiers":{"op":"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})","job":"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})|job_[0-9]{10,13}_[0-9]{1,20}_[0-9a-f]{16}","thread_id":"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})|[A-Za-z0-9_-]{43}|[0-9]{1,20}","parent_op":"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})","gesture":"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"},"version_pattern":"(?:[vV]?[0-9]{1,4}\\.[0-9]{1,4}\\.[0-9]{1,8}|[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{3})"};
    var ENABLED_KEY = 'aip_activity_log_enabled';
    var DATA_KEY = 'aip_activity_log_v1';
    var OWNER_KEY = 'aip_activity_log_owner';
    var RETENTION_KEY = 'aip_activity_log_retention_hours';
    var KEEP_MS = 60 * 60 * 1000;
    function setRetentionHours(value) {
        var hours = Number(value);
        if (!Number.isInteger(hours) || hours < 1 || hours > 8760) return false;
        // Remove already expired entries before increasing the window.
        expire();
        KEEP_MS = hours * 3600000;
        writeStorage(RETENTION_KEY, String(hours));
        expire();
        notify();
        return true;
    }
    var MAX_STORED_CHARS = 1500000;
    var MAX_FIELD = 500;
    var FLUSH_MS = 2000;

    var enabled = false;
    var buffer = [];
    var flushTimer = null;
    var seq = 0;
    var listeners = [];

    function readStorage(key) {
        try { return window.localStorage.getItem(key); } catch (e) { return null; }
    }
    function writeStorage(key, value) {
        try { window.localStorage.setItem(key, value); return true; } catch (e) { return false; }
    }
    function removeStorage(key) {
        try { window.localStorage.removeItem(key); } catch (e) {}
    }

    enabled = readStorage(ENABLED_KEY) === '1';
    var savedHours = Number(readStorage(RETENTION_KEY));
    if (Number.isInteger(savedHours) && savedHours >= 1 && savedHours <= 8760) KEEP_MS = savedHours * 3600000;

    function clip(value, max) {
        var text = String(value == null ? '' : value);
        var limit = max || MAX_FIELD;
        return text.length > limit ? text.slice(0, limit) + '…(' + text.length + ')' : text;
    }

    function readStored() {
        var raw = readStorage(DATA_KEY);
        if (!raw) return [];
        try {
            var parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    }

    function prune(entries) {
        var cutoff = Date.now() - KEEP_MS;
        var kept = entries.filter(function (entry) { return entry && entry.t > cutoff && entry.t <= Date.now(); });
        var text = JSON.stringify(kept);
        while (text.length > MAX_STORED_CHARS && kept.length > 1) {
            kept = kept.slice(Math.ceil(kept.length / 4));
            text = JSON.stringify(kept);
        }
        return { entries: kept, text: text };
    }

    function flush() {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        if (!buffer.length) return;
        var pending = buffer;
        buffer = [];
        if (!enabled) return;
        var result = prune(readStored().concat(pending).map(safeEntry).filter(Boolean));
        if (!writeStorage(DATA_KEY, result.text)) {
            // The site's storage is full: keep the newest half so recording can continue.
            var half = result.entries.slice(Math.floor(result.entries.length / 2));
            writeStorage(DATA_KEY, JSON.stringify(half));
        }
    }

    function scheduleFlush() {
        if (!flushTimer) flushTimer = setTimeout(flush, FLUSH_MS);
    }

    function safeRoute(url) {
        if (POLICY.routes.indexOf(url) >= 0) return url;
        try {
            var parsed = new URL(String(url), window.location.href);
            if (parsed.origin !== window.location.origin) return undefined;
            var path = parsed.pathname;
            return POLICY.routes.find(function (route) {
                var pattern = route.split(/<[^>]+>/).map(function (part) { return part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }).join('[^/]+');
                return new RegExp('^' + pattern + '$').test(path);
            });
        } catch (_) { return undefined; }
    }

    function providerOf(url) {
        try {
            var host = new URL(String(url), window.location.href).hostname;
            return ({ 'api.openai.com': 'openai', 'generativelanguage.googleapis.com': 'gemini',
                'aiplatform.googleapis.com': 'vertex_ai', 'api.x.ai': 'xai', 'api.deepseek.com': 'deepseek',
                'api.mistral.ai': 'mistral', 'api.ideogram.ai': 'ideogram', 'api.moonshot.ai': 'kimi',
                'api.z.ai': 'zai' })[host];
        } catch (_) { return undefined; }
    }

    function safeEntry(raw) {
        if (!raw || POLICY.events.indexOf(raw.ev) < 0) return null;
        var out = { ev: raw.ev };
        Object.keys(raw).forEach(function (key) {
            var value = raw[key];
            if (POLICY.numbers.indexOf(key) >= 0 && typeof value === 'number' && Number.isFinite(value) && Math.abs(value) <= 9007199254740992) out[key] = value;
            else if (POLICY.booleans.indexOf(key) >= 0 && typeof value === 'boolean') out[key] = value;
            else if (key === 'version' && typeof value === 'string' && new RegExp('^(?:' + POLICY.version_pattern + ')$').test(value)) out[key] = value;
            else if (POLICY.enums[key] && POLICY.enums[key].indexOf(value) >= 0) out[key] = value;
            else if ((key === 'route' || key === 'path') && POLICY.routes.indexOf(value) >= 0) out[key] = value;
            else if (POLICY.identifiers[key] && typeof value === 'string' && new RegExp('^(?:' + POLICY.identifiers[key] + ')$').test(value)) out[key] = value;
        });
        return out;
    }

    function log(ev, fields) {
        if (!enabled) return;
        try {
            var raw = Object.assign({}, fields || {}, { t: Date.now(), ev: ev, seq: ++seq });
            if (raw.path) raw.path = safeRoute(raw.path);
            var entry = safeEntry(raw);
            if (!entry) return;
            buffer.push(entry);
            scheduleFlush();
        } catch (_) {}
    }

    // Lifecycle-only operations: never pass arguments, return values or exception messages.
    function operation(name, work) {
        var started = Date.now();
        var op = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID().replace(/-/g, '') : undefined;
        log(name + '.start', { op: op });
        return Promise.resolve().then(work).then(function (value) {
            log(name + '.done', { op: op, ms: Date.now() - started }); return value;
        }, function (error) {
            log(name + '.error', { op: op, ms: Date.now() - started }); throw error;
        });
    }

    function expire() {
        if (!enabled) { removeStorage(DATA_KEY); return; }
        if (!readStorage(DATA_KEY)) return;
        var entries = prune(readStored()).entries.map(safeEntry).filter(Boolean);
        writeStorage(DATA_KEY, JSON.stringify(entries));
    }

    function recent(windowMs, maxChars) {
        flush();
        var cutoff = Date.now() - Math.min(windowMs || KEEP_MS, KEEP_MS);
        expire();
        var entries = readStored().filter(function (entry) { return entry && entry.t > cutoff && entry.t <= Date.now(); }).map(safeEntry).filter(Boolean);
        var limit = maxChars || 3000000;
        var size = JSON.stringify(entries).length;
        while (size > limit && entries.length > 1) {
            entries = entries.slice(Math.ceil(entries.length / 10));
            size = JSON.stringify(entries).length;
        }
        return entries;
    }

    function stats() {
        expire();
        var raw = readStorage(DATA_KEY) || '';
        return { count: readStored().length + buffer.length, chars: raw.length };
    }

    function clear() {
        buffer = [];
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        removeStorage(DATA_KEY);
        notify();
    }

    function setEnabled(value) {
        var next = !!value;
        if (next === enabled) return;
        if (!next) log('activity_log.disabled');
        flush();
        enabled = next;
        writeStorage(ENABLED_KEY, next ? '1' : '0');
        if (next) {
            log('activity_log.enabled', pageInfo());
        } else {
            clear();
        }
        notify();
    }

    function onChange(callback) {
        if (typeof callback === 'function') listeners.push(callback);
    }
    function notify() {
        listeners.slice().forEach(function (callback) { try { callback(enabled); } catch (e) {} });
    }

    // --- Descriptions that never include user text -------------------------------------

    function pathOnly(url) { return safeRoute(url); }

    function describe(el) {
        return el && POLICY.enums.control.indexOf(el.id) >= 0 ? el.id : undefined;
    }

    function actionable(target) {
        if (!target || !target.closest) return target;
        return target.closest('button, a, [role="button"], [onclick], input, select, label, summary, [data-action], [data-tab]') || target;
    }

    function isSecretField(el) {
        var type = String((el && el.type) || '').toLowerCase();
        return type === 'password' || type === 'text' || type === 'search' || type === 'email' || type === 'url'
            || type === 'textarea' || type === 'number' || type === 'tel' || el.isContentEditable;
    }

    function pageInfo() {
        var nav = window.navigator || {};
        var info = {
            path: window.location.pathname,
            version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || undefined,
            ua: clip(nav.userAgent || '', 300),
            lang: nav.language,
            online: nav.onLine,
            w: window.innerWidth, h: window.innerHeight,
            dpr: window.devicePixelRatio,
            standalone: !!(window.matchMedia && window.matchMedia('(display-mode: standalone)').matches),
            visibility: document.visibilityState,
        };
        try { info.sw = !!(nav.serviceWorker && nav.serviceWorker.controller); } catch (e) {}
        return info;
    }

    // --- Account ownership: another account's log is never sent ------------------------

    function checkOwner() {
        var owner = window.CHAT_CONFIG ? window.CHAT_CONFIG.currentUsername : undefined;
        if (owner === undefined || owner === null) return;
        var key = String(owner);
        var stored = readStorage(OWNER_KEY);
        if (stored !== null && stored !== key) removeStorage(DATA_KEY);
        writeStorage(OWNER_KEY, key);
    }

    // --- Hooks ---------------------------------------------------------------------------

    function hookFetch() {
        if (typeof window.fetch !== 'function') return;
        var original = window.fetch;
        window.fetch = function (input, init) {
            if (!enabled) return original.apply(this, arguments);
            var started = Date.now();
            var method = 'GET';
            var url = '';
            try {
                url = typeof input === 'string' ? input : (input && input.url) || String(input);
                method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
            } catch (e) {}
            var path = pathOnly(url);
            var promise = original.apply(this, arguments);
            try {
                promise.then(function (res) {
                    log('fetch', { provider: providerOf(url), method: method, path: path, status: res.status, ms: Date.now() - started,
                        op: res.headers && res.headers.get('X-Diagnostic-ID'),
                        redirected: res.redirected || undefined });
                }, function (err) {
                    log('fetch.error', { provider: providerOf(url), method: method, path: path, ms: Date.now() - started,
                        error: err && err.name, failed: true });
                });
            } catch (e) {}
            return promise;
        };
    }

    function hookXhr() {
        if (!window.XMLHttpRequest) return;
        var proto = window.XMLHttpRequest.prototype;
        var open = proto.open;
        var send = proto.send;
        proto.open = function (method, url) {
            try { this.__aipLog = { method: String(method || 'GET').toUpperCase(), path: pathOnly(url) }; } catch (e) {}
            return open.apply(this, arguments);
        };
        proto.send = function () {
            var info = this.__aipLog;
            if (enabled && info) {
                var started = Date.now();
                var xhr = this;
                xhr.addEventListener('loadend', function () {
                    log('xhr', { method: info.method, path: info.path, status: xhr.status, ms: Date.now() - started });
                });
            }
            return send.apply(this, arguments);
        };
    }

    function hookHistory() {
        ['pushState', 'replaceState'].forEach(function (name) {
            var original = window.history && window.history[name];
            if (typeof original !== 'function') return;
            window.history[name] = function (state, title, url) {
                var result = original.apply(this, arguments);
                if (enabled && url !== undefined && url !== null) log('nav.' + name, { path: pathOnly(url) });
                return result;
            };
        });
    }

    function hookConsole() {
        ['error', 'warn'].forEach(function (level) {
            var original = window.console && window.console[level];
            if (typeof original !== 'function') return;
            window.console[level] = function () {
                log('console.' + level, { arguments: arguments.length });
                return original.apply(this, arguments);
            };
        });
    }

    function watchToasts() {
        // chat_core calls its own showToast, so the toast stack is watched instead.
        var start = function () {
            var stack = document.getElementById && document.getElementById('toast-stack');
            if (!stack || !window.MutationObserver) return;
            new MutationObserver(function (mutations) {
                if (!enabled) return;
                mutations.forEach(function (mutation) {
                    Array.prototype.forEach.call(mutation.addedNodes || [], function (node) {
                        if (node.nodeType !== 1) return;
                        var match = /\b(error|success|warning|info)\b/.exec(String(node.className));
                        var kind = match ? match[1] : undefined;
                        log('toast', { kind: kind, count: 1 });
                    });
                });
            }).observe(stack, { childList: true });
        };
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
        else start();
    }

    function watchPanels() {
        if (!window.MutationObserver) return;
        var pattern = /modal|overlay|dialog|panel|popup|popover|sidebar|menu|banner|drawer/i;
        var observer = new MutationObserver(function (mutations) {
            if (!enabled) return;
            mutations.forEach(function (mutation) {
                var el = mutation.target;
                if (!el || !el.id || !pattern.test(el.id)) return;
                var wasHidden = /(^|\s)hidden(\s|$)/.test(mutation.oldValue || '');
                var isHidden = el.classList.contains('hidden');
                if (wasHidden !== isHidden) log(isHidden ? 'ui.hide' : 'ui.show', { control: describe(el) });
            });
        });
        var start = function () {
            if (document.body) observer.observe(document.body, { subtree: true, attributes: true, attributeFilter: ['class'], attributeOldValue: true });
        };
        if (document.body) start(); else document.addEventListener('DOMContentLoaded', start, { once: true });
    }

    function watchLongTasks() {
        try {
            if (!window.PerformanceObserver) return;
            var observer = new PerformanceObserver(function (list) {
                if (!enabled) return;
                list.getEntries().forEach(function (entry) {
                    if (entry.duration >= 200) log('perf.longtask', { ms: Math.round(entry.duration) });
                });
            });
            observer.observe({ type: 'longtask', buffered: false });
        } catch (e) {}
    }

    function heartbeat() {
        setInterval(function () {
            if (!enabled || document.visibilityState !== 'visible') return;
            var fields = { nodes: document.getElementsByTagName('*').length };
            try {
                var memory = window.performance && window.performance.memory;
                if (memory) fields.heap_mb = Math.round(memory.usedJSHeapSize / 1048576);
            } catch (e) {}
            log('heartbeat', fields);
        }, 60000);
    }

    function listen() {
        ['play', 'pause', 'ended', 'waiting', 'loadedmetadata', 'error'].forEach(function (name) {
            document.addEventListener(name, function (event) {
                var media = event.target;
                if (!media || (media.tagName !== 'AUDIO' && media.tagName !== 'VIDEO')) return;
                log('media.' + (name === 'loadedmetadata' ? 'ready' : name), {
                    type: media.tagName === 'AUDIO' ? 'audio' : 'video',
                    duration: Number.isFinite(media.duration) ? Math.round(media.duration * 1000) : undefined,
                    error_code: media.error && media.error.code,
                });
            }, true);
        });
        document.addEventListener('click', function (event) {
            if (!enabled) return;
            var el = actionable(event.target);
            log('click', {
                control: describe(el),
            });
        }, true);
        document.addEventListener('change', function (event) {
            if (!enabled) return;
            var el = event.target;
            if (!el || !el.tagName) return;
            var fields = { control: describe(el) };
            var type = String(el.type || '').toLowerCase();
            if (type === 'checkbox' || type === 'radio') fields.checked = !!el.checked;
            else if (el.tagName === 'SELECT') fields.checked = el.selectedIndex >= 0;
            else if (type === 'file') fields.files = el.files ? el.files.length : 0;
            else if (type === 'range' || type === 'color') fields.value = clip(el.value, 40);
            else if (isSecretField(el) || el.tagName === 'TEXTAREA') fields.length = String(el.value || '').length;
            log('change', fields);
        }, true);
        document.addEventListener('keydown', function (event) {
            if (!enabled) return;
            var key = event.key;
            if (key !== 'Enter' && key !== 'Escape' && !(event.ctrlKey || event.metaKey || event.altKey)) return;
            if ((event.ctrlKey || event.metaKey || event.altKey) && String(key).length === 1 && !/[a-z0-9\/\.,]/i.test(key)) return;
            log('key', { key: clip(key, 20), ctrl: event.ctrlKey || undefined, meta: event.metaKey || undefined,
                alt: event.altKey || undefined, shift: event.shiftKey || undefined, target: describe(event.target) });
        }, true);
        document.addEventListener('submit', function (event) {
            log('submit', { target: describe(event.target) });
        }, true);
        document.addEventListener('visibilitychange', function () {
            expire();
            log('page.visibility', { state: document.visibilityState });
            if (document.visibilityState === 'hidden') flush();
        });
        window.addEventListener('pagehide', function (event) {
            log('page.hide', { persisted: !!event.persisted });
            flush();
        });
        window.addEventListener('pageshow', function (event) {
            if (event.persisted) log('page.show', { persisted: true });
        });
        window.addEventListener('popstate', function () { log('nav.popstate', { path: window.location.pathname }); });
        window.addEventListener('hashchange', function () { log('nav.hashchange', { path: window.location.pathname }); });
        window.addEventListener('online', function () { log('net.online'); });
        window.addEventListener('offline', function () { log('net.offline'); });
        window.addEventListener('resize', (function () {
            var timer = null;
            return function () {
                if (!enabled) return;
                clearTimeout(timer);
                timer = setTimeout(function () { log('page.resize', { viewport: window.innerWidth + 'x' + window.innerHeight }); }, 500);
            };
        })());
        window.addEventListener('error', function (event) {
            if (!enabled) return;
            if (event && event.target && event.target !== window && event.target.tagName) {
                log('resource.error', { control: describe(event.target) });
                return;
            }
            log('error', { line: event && event.lineno, col: event && event.colno });
        }, true);
        window.addEventListener('unhandledrejection', function () { log('error.unhandled_rejection'); });
        try {
            if (navigator.serviceWorker) {
                navigator.serviceWorker.addEventListener('controllerchange', function () { log('sw.controllerchange'); });
            }
        } catch (e) {}
        window.addEventListener('storage', function (event) {
            if (event.key === RETENTION_KEY) {
                var hours = Number(event.newValue);
                KEEP_MS = (Number.isInteger(hours) && hours >= 1 && hours <= 8760 ? hours : 1) * 3600000;
                expire();
                var input = document.getElementById && document.getElementById('set-activity-log-retention-hours');
                if (input) input.value = String(KEEP_MS / 3600000);
                notify();
                return;
            }
            if (event.key !== ENABLED_KEY) return;
            enabled = event.newValue === '1';
            if (!enabled) buffer = [];
            notify();
        });
    }

    checkOwner();
    hookFetch();
    hookXhr();
    hookHistory();
    hookConsole();
    watchToasts();
    watchPanels();
    watchLongTasks();
    heartbeat();
    listen();
    expire();
    setInterval(expire, 30000);
    log('page.load', pageInfo());
    window.addEventListener('load', function () {
        var timing = {};
        try {
            var nav = window.performance.getEntriesByType('navigation')[0];
            if (nav) {
                timing.dom_ms = Math.round(nav.domContentLoadedEventEnd);
                timing.load_ms = Math.round(nav.loadEventStart);
                timing.nav_type = nav.type;
            }
        } catch (e) {}
        log('page.loaded', timing);
    });

    function canAttach() {
        var server = document.getElementById && document.getElementById('set-server-diagnostic-log');
        return enabled || !!(server && server.checked);
    }

    function feedbackPayload() {
        if (!canAttach()) return null;
        return {
            client: 'web',
            version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || '',
            window_seconds: Math.round(KEEP_MS / 1000),
            entries: enabled ? recent(KEEP_MS) : [],
        };
    }

    /** Browser storage entries whose key names [threadId] (branch names, the pinned branch, …). */
    function storageFor(storage, threadId) {
        var found = {};
        try {
            for (var i = 0; i < storage.length; i++) {
                var key = storage.key(i);
                if (key && key.indexOf(threadId) !== -1) found[key] = storage.getItem(key);
            }
        } catch (e) {}
        return found;
    }

    /** `chat_copy` of a feedback: the server decrypts the chat [threadId]; the browser adds its own storage of it. */
    function chatCopyPayload(threadId) {
        var id = String(threadId);
        return {
            client: 'web',
            version: (window.CHAT_CONFIG && window.CHAT_CONFIG.appVersion) || '',
            thread_id: id,
            client_state: {
                local_storage: storageFor(window.localStorage || { length: 0 }, id),
                session_storage: storageFor(window.sessionStorage || { length: 0 }, id),
                url: String(window.location && window.location.pathname || ''),
            },
        };
    }

    /**
     * Shows the result of a feedback sent with [feedbackPayload] (null when the log is off) and, when
     * [chatCopy], a copy of the open chat; false keeps the form (the request failed).
     */
    function reportFeedback(res, payload, chatCopy, images) {
        var toast = typeof window.showToast === 'function' ? window.showToast : function () {};
        return res.json().catch(function () { return {}; }).then(function (data) {
            if (!res.ok) {
                toast(data.error || 'フィードバックの送信に失敗しました', 'error', true);
                return false;
            }
            var failed = [];
            if (payload && data.logs_saved === false) failed.push('ログ');
            if (chatCopy && data.chat_copy_saved !== true) failed.push('チャットのコピー');
            if (images && images.count && !images.saved) failed.push('画像');
            if (failed.length) {
                toast('フィードバックを送信しました（' + failed.join('と') + 'は保存できませんでした）', 'error', true);
                return true;
            }
            var sent = ['フィードバック'];
            if (payload) sent.push('保存期間内のログ（' + (typeof data.logs_count === 'number' ? data.logs_count : payload.entries.length) + '件）');
            if (chatCopy) sent.push('チャットのコピー');
            if (images && images.count) sent.push('画像（' + images.count + '枚）');
            toast(sent.join(sent.length > 2 ? '、' : 'と') + 'を送信しました', 'success');
            return true;
        });
    }

    // --- Settings: the switch (feedback tab) and the delete button (data tab cache card) ---

    function usageText() {
        var info = stats();
        return '操作ログ: ' + info.count + '件 (' + (info.chars * 2 / (1024 * 1024)).toFixed(1) + 'MB)';
    }

    function refreshUi() {
        if (typeof document.getElementById !== 'function') return;
        var toggle = document.getElementById('set-activity-log');
        if (toggle) toggle.checked = enabled;
        var status = document.getElementById('activity-log-feedback-status');
        if (status) {
            status.textContent = enabled
                ? '記録中です。フィードバック・返信でログの添付を選択した場合に送信します。（' + usageText() + '）'
                : '記録していません。';
        }
        var usage = document.getElementById('activity-log-usage-text');
        if (usage) usage.textContent = usageText();
    }

    function bindUi() {
        try { bindControls(); } catch (e) {}
    }

    function bindControls() {
        if (typeof document.getElementById !== 'function') return;
        var toggle = document.getElementById('set-activity-log');
        if (toggle) toggle.addEventListener('change', function () { setEnabled(!!toggle.checked); });
        var retention = document.getElementById('set-activity-log-retention-hours');
        if (retention) {
            retention.value = String(KEEP_MS / 3600000);
            retention.addEventListener('change', function () {
                if (!setRetentionHours(retention.value)) retention.value = String(KEEP_MS / 3600000);
            });
        }
        var button = document.getElementById('clear-activity-log-btn');
        if (button) {
            button.addEventListener('click', function () {
                if (!window.confirm('操作ログを削除しますか？')) return;
                clear();
                if (typeof window.showToast === 'function') window.showToast('操作ログを削除しました', 'success');
            });
        }
        // Read the size again whenever the settings modal or one of the two tabs is shown.
        if (window.MutationObserver) {
            var observer = new MutationObserver(refreshUi);
            ['settings-modal', 'tab-feedback', 'tab-data'].forEach(function (id) {
                var el = document.getElementById(id);
                if (el) observer.observe(el, { attributes: true, attributeFilter: ['class'] });
            });
        }
        onChange(refreshUi);
        refreshUi();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bindUi, { once: true });
    else bindUi();

    window.ActivityLog = {
        isEnabled: function () { return enabled; },
        setEnabled: setEnabled,
        log: log,
        operation: operation,
        recent: recent,
        stats: stats,
        clear: clear,
        flush: flush,
        onChange: onChange,
        feedbackPayload: feedbackPayload,
        canAttach: canAttach,
        chatCopyPayload: chatCopyPayload,
        reportFeedback: reportFeedback,
        setRetentionHours: setRetentionHours,
        get WINDOW_MS() { return KEEP_MS; },
    };
})();
