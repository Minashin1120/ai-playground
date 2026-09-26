var hi=Object.defineProperty;var o=(e,t)=>hi(e,"name",{value:t,configurable:!0});const get=o(e=>document.getElementById(e),"get"),nativeConsoleLog=typeof console.log=="function"?console.
log.bind(console):function(){},nativeConsoleInfo=typeof console.info=="function"?console.info.bind(console):
nativeConsoleLog;let settingsModalLoaded=!1;const setSettingsSaveEnabled=o(e=>{const t=get("save-set\
tings-btn");t&&(t.disabled=!e,t.classList.toggle("opacity-60",!e),t.classList.toggle("cursor-not-all\
owed",!e),t.setAttribute("title",e?"":"\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u5B8C\u4E86\u5F8C\u306B\u4FDD\u5B58\u3067\u304D\u307E\u3059"))},
"setSettingsSaveEnabled");(function(){const t=o(l=>/(\/files\/thumb\/|\/files\/)/.test(String(l||"")),
"isFileUrl"),n=o(l=>fetch(l,{method:"GET",headers:{Range:"bytes=0-0"},cache:"no-store"}).then(c=>c.status).
catch(()=>-1),"fileUrlStatus");document.addEventListener("load",l=>{const c=l.target;if(!c||c.tagName!==
"IMG"||!c.classList.contains("chat-image"))return;const u=c.closest(".chat-image-frame");u&&(u.dataset.
chatImageState="loaded",u.removeAttribute("aria-busy"))},!0);const i=o((l,c)=>{const u=document.createElement(
"div");return u.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",c?u.innerHTML=
'<i class="fas fa-key" style="font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd\
34d;font-weight:700;line-height:1.3">\u6697\u53F7\u30AD\u30FC\u304C\u4E00\u81F4\u3057\u306A\u3044\u305F\u3081<br>\u95B2\u89A7\u3067\u304D\u307E\u305B\u3093</div>':
u.innerHTML='<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9\
px;color:#9ca3af;font-weight:700">\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093</div>',
l&&u.setAttribute("data-file-name",String(l)),u},"buildWarning"),a=o(l=>{const c=document.createElement(
"div");return c.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",c.innerHTML='\
<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;\
color:#bfdbfe;font-weight:700;line-height:1.3">\u4E00\u6642\u7684\u306B\u6DF7\u96D1\u3057\u3066\u3044\u307E\u3059<br>\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u304F\u3060\u3055\u3044</div>',
l&&c.setAttribute("data-file-name",String(l)),c},"buildBusyWarning"),r=o(l=>String(l||"").split("?")[0].
replace("/files/thumb/","/files/"),"fullFileUrl");document.addEventListener("error",l=>{const c=l.target;
if(!c||c.tagName!=="IMG")return;const u=c.currentSrc||c.src||"";if(!t(u)){const _=c.closest&&c.closest(
".chat-image-frame");if(_){_.dataset.chatImageState="error",_.removeAttribute("aria-busy");const C=_.
querySelector(".chat-image-loading"),L=C&&C.querySelector("span");L&&(L.textContent="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F");
const E=C&&C.querySelector("i");E&&(E.className="fas fa-image")}return}l.stopImmediatePropagation(),
l.preventDefault();const f=String(u).split("?")[0],g=c.getAttribute("data-viewer-filename")||f.split(
"/").pop(),y=o(_=>{const C=c.closest&&c.closest(".chat-image-frame");C&&(C.dataset.chatImageState="f\
ailed",C.removeAttribute("aria-busy"));const L=i(g,!!_);try{c.replaceWith(L)}catch{}},"showWarning"),
w=o(()=>{const _=c.closest&&c.closest(".chat-image-frame");_&&(_.dataset.chatImageState="busy",_.removeAttribute(
"aria-busy"));const C=a(g);try{c.replaceWith(C)}catch{}},"showBusyWarning"),v=o((_,C)=>{const L=c.cloneNode(
!1);L.setAttribute("data-file-retry",String(C));const E=_+(_.includes("?")?"&":"?")+"retry="+Date.now()+
"_"+C;L.setAttribute("src",E);try{c.replaceWith(L)}catch{}},"retryLoad"),k=o(_=>{if(_===429||_===503){
w();return}if(_===409){y(!0);return}if(_===404||_===410||_===403){y(!1);return}const C=parseInt(c.getAttribute&&
c.getAttribute("data-file-retry")||"0",10);if(C<2){v(u,C+1);return}if(u.includes("/files/thumb/")&&!c.
getAttribute("data-file-fallback")){c.setAttribute("data-file-fallback","1"),v(r(u),0);return}y(!1)},
"handleStatus");n(u).then(k).catch(()=>{const _=parseInt(c.getAttribute&&c.getAttribute("data-file-r\
etry")||"0",10);if(_<2){v(u,_+1);return}if(u.includes("/files/thumb/")){w();return}y(!1)})},!0)})();
function buildChatImageHtml(e,t={}){const n=String(e||""),i=String(t.alt||""),a=String(t.title||""),
r=String(t.viewerSrc||n),l=String(t.filename||"");if(n.startsWith("sandbox:"))return`<span class="te\
xt-xs text-gray-500" title="${escapeHtml(n)}">${escapeHtml(i)||"\uFF08\u753B\u50CF\u30C7\u30FC\u30BF\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09"}\
</span>`;const c=a?` title="${escapeHtml(a)}"`:"",u=l?` data-viewer-filename="${escapeHtml(l)}"`:"";
return`<span class="chat-image-frame" data-chat-image-state="loading" aria-busy="true"><span class="\
chat-image-loading" role="status" aria-label="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D"><i class="fas fa-spinner fa-spin" aria-hidde\
n="true"></i><span>\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</span></span><img src="${escapeHtml(
n)}" data-viewer-src="${escapeHtml(r)}" alt="${escapeHtml(i)}"${c}${u} class="chat-image" loading="l\
azy" decoding="async" width="320" height="320"></span>`}o(buildChatImageHtml,"buildChatImageHtml");const isAdminSidebarDebugEnabled=o(
()=>{try{const e=window.CHAT_CONFIG||{};return!!(e.botConfig&&e.botConfig.isAdmin)}catch{return!1}},
"isAdminSidebarDebugEnabled"),ADMIN_SIDEBAR_DEBUG_PREFIX="[admin-sidebar]",adminSidebarDebugEntries=[],
snapshotSidebarHistory=o(e=>{if(!isAdminSidebarDebugEnabled())return null;const t=get("thread-list"),
n=get("sidebar"),i=get("settings-modal"),a=get("history-modal"),r=t?window.getComputedStyle(t):null,
l=n?window.getComputedStyle(n):null,c=t?Array.from(t.querySelectorAll("[data-thread-id]")):[],u=c[0]||
null,f=u?window.getComputedStyle(u):null;let g=null;try{g=typeof threadLoading=="boolean"?threadLoading:
null}catch{g=null}const y={t:Date.now(),reason:String(e||""),path:location.pathname,vw:window.innerWidth,
liteHtml:document.documentElement.classList.contains("performance-lite-mode"),blurHtml:document.documentElement.
classList.contains("performance-blur-disabled"),liquidBody:!!(document.body&&document.body.classList.
contains("liquid-glass-mode")),blurMode:adaptiveBlurPreferenceMode,liteEnabled:adaptiveBlurLiteEnabled,
sidebarClass:n?n.className:null,sidebarDisplay:l?l.display:null,sidebarOpacity:l?l.opacity:null,sidebarVisibility:l?
l.visibility:null,compact:!!(n&&n.classList.contains("compact")),sidebarOpen:!!(n&&n.classList.contains(
"open")),listExists:!!t,listParent:t&&t.parentElement?t.parentElement.id||t.parentElement.className:
null,listClass:t?t.className:null,listChildCount:t?t.children.length:0,listItemCount:c.length,listDisplay:r?
r.display:null,listOpacity:r?r.opacity:null,listVisibility:r?r.visibility:null,listHeight:r?r.height:
null,hideCompact:!!(t&&t.classList.contains("hide-compact")),searchLen:(()=>{const w=get("search-box");
return w?String(w.value||"").length:0})(),firstItemText:u&&u.textContent?u.textContent.trim().slice(
0,40):null,firstItemOpacity:f?f.opacity:null,firstItemDisplay:f?f.display:null,firstItemVisibility:f?
f.visibility:null,firstItemClass:u?u.className:null,settingsHidden:i?i.classList.contains("hidden"):
null,settingsOpen:i?i.classList.contains("modal-open"):null,settingsDisplay:i&&i.style.display||null,
historyHidden:a?a.classList.contains("hidden"):null,threadLoading:g};adminSidebarDebugEntries.push(y),
adminSidebarDebugEntries.length>80&&adminSidebarDebugEntries.shift();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,
e,y)}catch{}return y},"snapshotSidebarHistory"),installAdminSidebarDebugObserver=o(()=>{if(!isAdminSidebarDebugEnabled())
return;const e=get("thread-list");if(!(!e||e.dataset.adminSidebarDebugObserved==="1")){e.dataset.adminSidebarDebugObserved=
"1";try{new MutationObserver(n=>{const i=n.reduce((r,l)=>r+Array.from(l.removedNodes||[]).filter(c=>c&&
c.nodeType===1&&c.getAttribute&&c.getAttribute("data-thread-id")).length,0),a=n.reduce((r,l)=>r+Array.
from(l.addedNodes||[]).filter(c=>c&&c.nodeType===1&&c.getAttribute&&c.getAttribute("data-thread-id")).
length,0);snapshotSidebarHistory(`thread-list-mutated added=${a} removed=${i}`)}).observe(e,{childList:!0,
attributes:!0,attributeFilter:["class","style"]})}catch{}}},"installAdminSidebarDebugObserver");window.
__adminSidebarDebugDump=()=>{if(!isAdminSidebarDebugEnabled())return[];const e=adminSidebarDebugEntries.
slice();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"dump",e)}catch{}return e},window.copyAdminSidebarDebug=
async()=>{if(!isAdminSidebarDebugEnabled())return!1;const e=JSON.stringify(adminSidebarDebugEntries,
null,2);try{return navigator.clipboard&&navigator.clipboard.writeText&&await navigator.clipboard.writeText(
e),nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copied",adminSidebarDebugEntries.length,"entries"),!0}catch{
try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copy-failed",e)}catch{}return!1}};const ADAPTIVE_BLUR_COOKIE="\
adaptive_blur_disabled",ADAPTIVE_LITE_COOKIE="adaptive_lite_mode",ADAPTIVE_BLUR_MODE_COOKIE="adaptiv\
e_blur_mode",readCookieValue=o(e=>{try{const t=document.cookie.split(";").map(n=>n.trim()).find(n=>n.
startsWith(`${e}=`));return t?decodeURIComponent(t.slice(e.length+1)):""}catch{return""}},"readCooki\
eValue"),normalizeAdaptiveBlurMode=o(e=>["enabled","disabled","lite"].includes(e)?e:"auto","normaliz\
eAdaptiveBlurMode"),writeAdaptiveBlurCookie=o((e,t,n=31536e3)=>{try{const i=window.location.protocol===
"https:"?"; Secure":"";document.cookie=`${e}=${encodeURIComponent(t)}; Path=/; Max-Age=${n}; SameSit\
e=Lax${i}`}catch{}},"writeAdaptiveBlurCookie"),adaptiveBlurInteractionCooldownMs=3e3;let adaptiveBlurPreferenceMode=normalizeAdaptiveBlurMode(
readCookieValue(ADAPTIVE_BLUR_MODE_COOKIE)),adaptiveBlurMeasurementActive=!1,adaptiveBlurMeasurementLastAt=0,
adaptiveBlurFallbackEnabled=document.documentElement.classList.contains("performance-blur-disabled"),
adaptiveBlurLiteEnabled=document.documentElement.classList.contains("performance-lite-mode");const syncAdaptiveBlurSettingsUi=o(
()=>{const e=get("set-background-blur-mode"),t=get("background-blur-mode-status");e&&(e.value=adaptiveBlurPreferenceMode),
t&&(adaptiveBlurPreferenceMode==="lite"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="enabled"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u5E38\u306B\u6709\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="disabled"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurLiteEnabled?t.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u8CA0\u8377\u304C\u975E\u5E38\u306B\u9AD8\u3044\u305F\u3081\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurFallbackEnabled?t.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u63CF\u753B\u8CA0\u8377\u3092\u691C\u51FA\u3057\u305F\u305F\u3081\u3001\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
t.textContent="\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u304C\u6709\u52B9\u3067\u3059\u3002\u64CD\u4F5C\u6642\u306E\u63CF\u753B\u304C\u91CD\u3044\u5834\u5408\u306F\u81EA\u52D5\u3067\u7121\u52B9\u5316\u3057\u307E\u3059\u3002")},
"syncAdaptiveBlurSettingsUi"),enableAdaptiveBlurFallback=o(()=>{adaptiveBlurPreferenceMode!=="auto"||
adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.classList.add(
"performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1"),syncAdaptiveBlurSettingsUi())},
"enableAdaptiveBlurFallback"),enableAdaptiveBlurLite=o(()=>{adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
(adaptiveBlurLiteEnabled=!0,adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.
classList.add("performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1")),document.
documentElement.classList.add("performance-lite-mode"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"lite-auto-enabled"),syncAdaptiveBlurSettingsUi(),showToast("\u63CF\u753B\u8CA0\u8377\u304C\u9AD8\u3044\u305F\u3081\u3001\u8EFD\u91CF\u8868\u793A\uFF08\u6700\u5C0F\u8CA0\u8377\uFF09\u3092\u81EA\u52D5\u9069\u7528\u3057\u307E\u3057\u305F\u3002\u30BF\u30C3\u30D7\u3067\u8A2D\u5B9A\u3092\u958B\u304F",
"info",!1,openAdaptiveBlurSettingsFromToast),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"1"))},"en\
ableAdaptiveBlurLite"),openAdaptiveBlurSettingsFromToast=o(()=>{typeof window.openSettingsModal=="fu\
nction"&&window.openSettingsModal();const e=get("set-background-blur-mode"),t=get("tab-display")||get(
"tab-general");if(!(!e||!t)){for(const n of t.children)if(n.contains(e)){jumpToSetting(t.id==="tab-d\
isplay"?"display":"general",n);return}}},"openAdaptiveBlurSettingsFromToast"),applyAdaptiveBlurPreference=o(
e=>{const t=normalizeAdaptiveBlurMode(e);t!==adaptiveBlurPreferenceMode&&(adaptiveBlurPreferenceMode=
t,adaptiveBlurMeasurementActive=!1,adaptiveBlurLiteEnabled=!1,writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,
"",0),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"",0),t==="auto"?writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,
"",0):writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,t),adaptiveBlurFallbackEnabled=t==="disabled"||
t==="lite",adaptiveBlurLiteEnabled=t==="lite",document.documentElement.classList.toggle("performance\
-blur-disabled",adaptiveBlurFallbackEnabled),document.documentElement.classList.toggle("performance-\
lite-mode",adaptiveBlurLiteEnabled),revealPersistentSidebarLists(),snapshotSidebarHistory("blur-pref\
erence-applied:"+t),syncAdaptiveBlurSettingsUi())},"applyAdaptiveBlurPreference"),isSettingsModalOpen=o(
()=>{const e=get("settings-modal");return e?e.classList.contains("modal-open")||e.classList.contains(
"modal-prep")?!0:e.classList.contains("hidden")?!1:e.style.display&&e.style.display!=="none":!1},"is\
SettingsModalOpen"),restoreThreadSearchValue=o((e,t)=>{const n=get("search-box");n&&n.value!==e&&(n.
value=e,clearTimeout(searchTimeout),snapshotSidebarHistory(t||"restored-search-box"))},"restoreThrea\
dSearchValue"),THREAD_SEARCH_INPUT_IDS=["search-box","history-search-box"],isUserInitiatedSearchInput=o(
e=>!!(e&&e.inputType),"isUserInitiatedSearchInput"),unlockThreadSearchInput=o(e=>{e&&e.hasAttribute(
"readonly")&&e.removeAttribute("readonly")},"unlockThreadSearchInput"),markThreadSearchUserEdited=o(
e=>{e&&(e.dataset.userEdited="1")},"markThreadSearchUserEdited"),discardAutofilledThreadSearch=o(e=>{
const t=get("search-box");if(!t||t.dataset.userEdited||!t.value)return;restoreThreadSearchValue("",e||
"cleared-autofill-search-box");const n=get("history-search-box");n&&!n.dataset.userEdited&&(n.value=
"")},"discardAutofilledThreadSearch"),hardenThreadSearchInputs=o(()=>{THREAD_SEARCH_INPUT_IDS.forEach(
e=>{const t=get(e);if(!t)return;const n=o(()=>unlockThreadSearchInput(t),"unlock");t.addEventListener(
"pointerdown",n),t.addEventListener("touchstart",n,{passive:!0}),t.addEventListener("keydown",n),t.addEventListener(
"focus",n)}),discardAutofilledThreadSearch("cleared-autofill-search-box-init"),[0,50,250,1e3].forEach(
e=>{setTimeout(()=>discardAutofilledThreadSearch("cleared-autofill-search-box-"+e+"ms"),e)})},"harde\
nThreadSearchInputs"),revealPersistentSidebarLists=o(()=>{document.querySelectorAll("#thread-list > \
[data-thread-id], #gem-list > .gem-item").forEach(e=>{e.classList.remove("model-list-animate","slide\
-in-animate","fade-in","opacity-0"),e.style.removeProperty("opacity"),e.style.removeProperty("transf\
orm"),e.style.removeProperty("animation"),e.style.removeProperty("animation-delay"),e.style.removeProperty(
"visibility")}),["thread-list","gem-list"].forEach(e=>{const t=get(e);t&&(t.style.removeProperty("op\
acity"),t.style.removeProperty("visibility"))}),snapshotSidebarHistory("reveal-sidebar-lists")},"rev\
ealPersistentSidebarLists"),adaptiveBlurIsBusy=o(()=>!!(activeStreamingBubbleId||document.querySelector(
".modal-overlay.modal-open, .modal-overlay.modal-prep, .modal-overlay.modal-close")),"adaptiveBlurIs\
Busy"),measureInteractionFrames=o((e=!1)=>{if(adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
adaptiveBlurMeasurementActive||document.visibilityState!=="visible")return;if(e)adaptiveBlurMeasurementLastAt=
Date.now();else{const a=Date.now();if(a-adaptiveBlurMeasurementLastAt<adaptiveBlurInteractionCooldownMs||
adaptiveBlurIsBusy())return;adaptiveBlurMeasurementLastAt=a}adaptiveBlurMeasurementActive=!0;const t=[];
let n=0;const i=o(a=>{if(document.visibilityState!=="visible"){adaptiveBlurMeasurementActive=!1;return}
if(n){const y=a-n;y<=200&&t.push(y)}if(n=a,t.length<30){requestAnimationFrame(i);return}adaptiveBlurMeasurementActive=
!1;const r=[...t].sort((y,w)=>y-w),l=Math.min(17.5,Math.max(7,r[Math.floor(r.length*.2)])),c=Math.max(
28,l*1.75),u=Math.max(44,l*2.7),f=t.filter(y=>y>=c).length,g=t.filter(y=>y>=u).length;(f>=5||f>=4&&g>=
2)&&(adaptiveBlurFallbackEnabled?enableAdaptiveBlurLite():enableAdaptiveBlurFallback())},"sampleFram\
e");requestAnimationFrame(i)},"measureInteractionFrames"),measureAdaptiveBlurAfterInteraction=o(()=>{
document.readyState!=="complete"||adaptiveBlurLiteEnabled||requestAnimationFrame(()=>{adaptiveBlurLiteEnabled||
measureInteractionFrames()})},"measureAdaptiveBlurAfterInteraction");document.addEventListener("clic\
k",e=>{const t=e.target instanceof Element?e.target:null;t&&t.closest('button, a, input, select, tex\
tarea, [role="button"], [tabindex]')&&measureAdaptiveBlurAfterInteraction()},!0);const externalScriptLoads=new Map,
loadExternalScript=o((e,t)=>{if(typeof t=="function"&&t())return Promise.resolve();if(externalScriptLoads.
has(e))return externalScriptLoads.get(e);const n=new Promise((i,a)=>{const r=document.createElement(
"script");r.src=e,r.async=!0,r.crossOrigin="anonymous",r.referrerPolicy="no-referrer",r.onload=()=>i(),
r.onerror=()=>a(new Error(`\u30E9\u30A4\u30D6\u30E9\u30EA\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F: ${e}`)),
document.head.appendChild(r)});return externalScriptLoads.set(e,n),n.catch(()=>externalScriptLoads.delete(
e)),n},"loadExternalScript"),ensurePdfLibraries=o(()=>Promise.all([loadExternalScript("/static/vendo\
r/html2canvas-pro-2.3.2.min.js",()=>typeof window.html2canvas=="function"),loadExternalScript("/stat\
ic/vendor/jspdf-2.5.1.umd.min.js",()=>!!(window.jspdf&&window.jspdf.jsPDF))]),"ensurePdfLibraries"),
ensureImageCompression=o(()=>loadExternalScript("https://cdn.jsdelivr.net/npm/browser-image-compress\
ion@2.0.2/dist/browser-image-compression.js",()=>typeof window.imageCompression=="function"),"ensure\
ImageCompression");let webauthnJsonLoad=null;const ensureWebAuthnJson=o(async()=>(window.webauthnJSON||
(webauthnJsonLoad||(webauthnJsonLoad=import("https://esm.sh/@github/webauthn-json@2.1.1").then(({create:e,
get:t})=>({create:e,get:t}))),window.webauthnJSON=await webauthnJsonLoad),window.webauthnJSON),"ensu\
reWebAuthnJson");window.DOMPurify&&window.DOMPurify.setConfig(window.CHAT_DOMPURIFY_CONFIG||{ADD_TAGS:[
"video","source"],ADD_ATTR:["controls","src","class","autoplay","loop","muted","poster","width","hei\
ght","start","type","reversed"],FORBID_TAGS:["iframe","object","embed"]});const THEME_DEFAULT="#0dd4\
bf",THEME_STORAGE_KEY="theme_color",INITIAL_THEME_COLOR=window.CHAT_CONFIG&&window.CHAT_CONFIG.initialThemeColor||
null,INITIAL_LIGHT_MODE_ENABLED=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.initialLightModeEnabled),INITIAL_LIQUID_GLASS_ENABLED=!!(window.
CHAT_CONFIG&&window.CHAT_CONFIG.initialLiquidGlassEnabled),RICH_PASTE_DEFAULT_PROMPT="\u3053\u306EPDF\u3092Markdown\
\u5F62\u5F0F\u306B\u5909\u63DB\u3057\u3001\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u306B\u66F8\u304D\u51FA\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
GEMINI_LOCAL_PY_DIALOG_KEY="gemini_local_py_dialog_enabled",COMPRESSION_SIZE_KEY="compression_max_si\
ze_mb",COMPRESSION_DIM_KEY="compression_max_dim",COMPRESSION_TYPE_KEY="compression_output_type",COMPRESSION_FORMAT_ONLY_KEY="\
compression_format_only",getCompressionMaxSizeMB=o(()=>parseFloat(localStorage.getItem(COMPRESSION_SIZE_KEY)||
"1.0"),"getCompressionMaxSizeMB"),getCompressionMaxDim=o(()=>parseInt(localStorage.getItem(COMPRESSION_DIM_KEY)||
"1920"),"getCompressionMaxDim"),getCompressionOutputType=o(()=>localStorage.getItem(COMPRESSION_TYPE_KEY)||
"original","getCompressionOutputType"),getCompressionFormatOnly=o(()=>localStorage.getItem(COMPRESSION_FORMAT_ONLY_KEY)===
"true","getCompressionFormatOnly"),IMAGE_EXTENSION_BY_MIME={"image/jpeg":".jpg","image/png":".png","\
image/webp":".webp"},imageFilenameForMime=o((e,t)=>{const n=IMAGE_EXTENSION_BY_MIME[String(t||"").toLowerCase()];
return n?`${String(e||"image").replace(/\.[^./\\]+$/,"")||"image"}${n}`:e||"image"},"imageFilenameFo\
rMime"),convertImageFormatOnly=o(async(e,t)=>{if(!e||!t||t==="original"||t===e.type)return e;await ensureImageCompression();
const n=await window.imageCompression.drawFileInCanvas(e,{fileType:t}),i=n&&n[0],a=n&&n[1];if(!a)throw new Error(
"Image conversion canvas is unavailable");let r;try{typeof a.convertToBlob=="function"?r=await a.convertToBlob(
{type:t,quality:1}):r=await new Promise((l,c)=>{a.toBlob(u=>u?l(u):c(new Error("Image conversion fai\
led")),t,1)})}finally{try{window.imageCompression.cleanupCanvasMemory(a)}catch{}try{i&&typeof i.close==
"function"&&i.close()}catch{}}return new File([r],imageFilenameForMime(e.name,t),{type:t,lastModified:e.
lastModified||Date.now()})},"convertImageFormatOnly"),setCompressionSettings=o((e,t,n,i)=>{localStorage.
setItem(COMPRESSION_SIZE_KEY,e),localStorage.setItem(COMPRESSION_DIM_KEY,t),localStorage.setItem(COMPRESSION_TYPE_KEY,
n),localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY,i)},"setCompressionSettings"),syncCompressionSettingsUi=o(
()=>{const e=get("compression-max-size"),t=get("compression-max-dim"),n=get("compression-output-type"),
i=get("compression-format-only");if(e&&(e.value=getCompressionMaxSizeMB()),t&&(t.value=getCompressionMaxDim()),
n&&(n.value=getCompressionOutputType()),i){i.checked=getCompressionFormatOnly();const g=i.checked;e&&
(e.disabled=g),t&&(t.disabled=g);const y=get("compression-size-wrap"),w=get("compression-dim-wrap");
y&&(y.style.opacity=g?"0.4":"1"),w&&(w.style.opacity=g?"0.4":"1")}const a=o((g,y)=>{get(g)&&get(y)&&
(get(y).value=get(g).value)},"sync");a("gpt-image-size","modal-gpt-image-size"),a("gpt-image-quality",
"modal-gpt-image-quality"),a("gpt-image-format","modal-gpt-image-format"),a("gpt-image-compression",
"modal-gpt-image-compression"),a("gemini-image-aspect","modal-gemini-image-aspect"),a("gemini-image-\
size","modal-gemini-image-size"),a("grok-image-aspect","modal-grok-image-aspect"),a("grok-image-reso\
lution","modal-grok-image-resolution"),a("grok-image-quality","modal-grok-image-quality"),a("ocr-tab\
le-format","modal-ocr-table-format"),a("ocr-pages","modal-ocr-pages");const r=o((g,y)=>{get(g)&&get(
y)&&(get(y).checked=get(g).checked)},"syncChk");r("ocr-extract-header","modal-ocr-extract-header"),r(
"ocr-extract-footer","modal-ocr-extract-footer"),r("ocr-include-blocks","modal-ocr-include-blocks"),
r("ocr-include-images","modal-ocr-include-images");const l=get("model-select").value,c=isGptImageModel(
l),u=isGeminiImageModel(l),f=isGrokImageModel(l);get("modal-gpt-image-options")&&get("modal-gpt-imag\
e-options").classList.toggle("hidden",!c),get("modal-gemini-image-options")&&get("modal-gemini-image\
-options").classList.toggle("hidden",!u),get("modal-grok-image-options")&&get("modal-grok-image-opti\
ons").classList.toggle("hidden",!f),get("modal-mistral-ocr-options")&&get("modal-mistral-ocr-options").
classList.toggle("hidden",!isMistralOcrModel(l))},"syncCompressionSettingsUi"),isGeminiLocalPyDialogEnabled=o(
()=>{const e=localStorage.getItem(GEMINI_LOCAL_PY_DIALOG_KEY);return e===null?!0:e==="1"||e==="true"},
"isGeminiLocalPyDialogEnabled"),setGeminiLocalPyDialogEnabled=o(e=>{localStorage.setItem(GEMINI_LOCAL_PY_DIALOG_KEY,
e?"1":"0")},"setGeminiLocalPyDialogEnabled"),syncGeminiLocalPyDialogSetting=o(()=>{const e=get("set-\
gemini-local-python-dialog");e&&(e.checked=isGeminiLocalPyDialogEnabled())},"syncGeminiLocalPyDialog\
Setting"),normalizeGeminiBackend=o(e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t===
"vertex_ai"||t==="vertex"||t==="vertexai"?"vertex_ai":"gemini_api"},"normalizeGeminiBackend"),normalizeAdminApiKeyMode=o(
e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t==="user_only"||t==="user"||
t==="settings"||t==="user_settings"?"user_only":"env_fallback"},"normalizeAdminApiKeyMode"),syncToggleButtons=o(
(e,t,n)=>{(e||[]).forEach(i=>{const a=i.getAttribute(n)===t;i.classList.toggle("border-cyan-400",a),
i.classList.toggle("bg-cyan-900/30",a),i.classList.toggle("text-white",a),i.classList.toggle("border\
-gray-600",!a),i.classList.toggle("bg-gray-800/70",!a)})},"syncToggleButtons"),syncAdminApiKeyModeUi=o(
()=>{const e=get("set-admin-api-key-mode"),t=get("admin-api-key-mode-note"),n=get("admin-api-key-mod\
e-status"),i=get("admin-api-key-mode-toggle");if(!e)return;const a=normalizeAdminApiKeyMode(e.value);
e.value=a,i&&!i.dataset.bound&&(i.dataset.bound="1",i.querySelectorAll("[data-admin-api-key-mode]").
forEach(r=>{r.addEventListener("click",()=>{e.value=normalizeAdminApiKeyMode(r.getAttribute("data-ad\
min-api-key-mode")),syncAdminApiKeyModeUi()})})),syncToggleButtons(i?i.querySelectorAll("[data-admin\
-api-key-mode]"):[],a,"data-admin-api-key-mode"),t&&(t.textContent=a==="user_only"?"\u901A\u5E38\u30E6\u30FC\u30B6\u30FC\u3068\u540C\u3058\u304F\u3001\u3053\u306E\u753B\u9762\u3067\
\u4FDD\u5B58\u3057\u305FAPI\u30AD\u30FC/Vertex\u8A2D\u5B9A\u306E\u307F\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002":
"\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u3068\u304D\u3060\u3051 .env \u3092\u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u5229\u7528\u3057\u307E\u3059\uFF08\u65E2\u5B9A\uFF09\u3002"),
n&&(n.textContent=a==="user_only"?"\u73FE\u5728: \u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u307F\uFF08\u63A8\u5968: \u8A2D\u5B9A\u5024\u3092\u660E\u793A\u7BA1\u7406\uFF09":
"\u73FE\u5728: .env \u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u6709\u52B9\uFF08\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306A\u3089 .env\uFF09")},
"syncAdminApiKeyModeUi"),ensureGeminiVertexCredentialsField=o(()=>{const e=get("gemini-vertex-settin\
gs");if(!e||get("set-gemini-vertex-credentials-json"))return;const t=document.createElement("div");t.
innerHTML=`
                <label class="text-xs text-gray-500 block">Vertex Service Account JSON (\u4EFB\u610F)</label>
                <textarea id="set-gemini-vertex-credentials-json" class="w-full h-28 bg-gray-800 bor\
der border-gray-600 rounded px-2 py-1 text-[11px] text-white font-mono" placeholder='{"type":"servic\
e_account", ...}'></textarea>
                <div class="text-[10px] text-gray-500 mt-1">\u672A\u5165\u529B\u6642\u306F\u30B5\u30FC\u30D0\u30FC\u5074ADC\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002\u5165\u529B\u3059\u308B\u3068\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u8A2D\u5B9A\u3060\u3051\u3067Ver\
tex\u8A8D\u8A3C\u3067\u304D\u307E\u3059\u3002</div>
            `,e.appendChild(t)},"ensureGeminiVertexCredentialsField"),syncGeminiBackendUi=o(()=>{const e=get(
"set-gemini-backend"),t=get("gemini-vertex-settings"),n=get("gemini-backend-note"),i=get("gemini-bac\
kend-status"),a=get("gemini-backend-toggle");if(!e)return;ensureGeminiVertexCredentialsField();const r=normalizeGeminiBackend(
e.value);e.value=r,a&&!a.dataset.bound&&(a.dataset.bound="1",a.querySelectorAll("[data-gemini-backen\
d]").forEach(l=>{l.addEventListener("click",()=>{e.value=normalizeGeminiBackend(l.getAttribute("data\
-gemini-backend")),syncGeminiBackendUi()})})),syncToggleButtons(a?a.querySelectorAll("[data-gemini-b\
ackend]"):[],r,"data-gemini-backend"),t&&t.classList.toggle("hidden",r!=="vertex_ai"),n&&(n.textContent=
r==="vertex_ai"?"Vertex AI \u3092\u5229\u7528\u3057\u307E\u3059\u3002Project ID / Location \u3092\u8A2D\u5B9A\u3057\u3001ADC \u307E\u305F\u306F Vertex Service Account JSON \u3092\u7528\u610F\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002":"Gemini API \u3092\u5229\u7528\u3057\u307E\u3059\u3002API Key \u3092\u8A2D\u5B9A\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
i&&(i.textContent=r==="vertex_ai"?"\u73FE\u5728: Vertex AI\uFF08Project ID / Location / \u8A8D\u8A3C\u60C5\u5831\u304C\u5FC5\u8981\uFF09":
"\u73FE\u5728: Gemini API\uFF08Gemini API Key \u3092\u4F7F\u7528\uFF09")},"syncGeminiBackendUi"),normalizeHex=o(
e=>{if(!e)return null;let t=String(e).trim();return!t||(t.startsWith("#")||(t=`#${t}`),t.length===4&&
(t=`#${t[1]}${t[1]}${t[2]}${t[2]}${t[3]}${t[3]}`),!/^#[0-9a-fA-F]{6}$/.test(t))?null:t.toLowerCase()},
"normalizeHex"),hexToRgb=o(e=>{const t=e.replace("#",""),n=parseInt(t.slice(0,2),16),i=parseInt(t.slice(
2,4),16),a=parseInt(t.slice(4,6),16);return[n,i,a]},"hexToRgb"),mix=o((e,t,n)=>Math.round(e+(t-e)*n),
"mix"),rgbToHex=o((e,t,n)=>`#${[e,t,n].map(i=>i.toString(16).padStart(2,"0")).join("")}`,"rgbToHex"),
deriveTheme=o(e=>{const[t,n,i]=hexToRgb(e),a=rgbToHex(mix(t,255,.45),mix(n,255,.45),mix(i,255,.45)),
r=rgbToHex(mix(t,255,.7),mix(n,255,.7),mix(i,255,.7)),l=rgbToHex(mix(t,0,.18),mix(n,0,.18),mix(i,0,.18)),
c=rgbToHex(mix(t,0,.32),mix(n,0,.32),mix(i,0,.32));return{base:e,light:a,lighter:r,dark:l,darker:c,rgb:`${t}\
, ${n}, ${i}`}},"deriveTheme"),applyThemeColor=o((e,t=!1)=>{const n=normalizeHex(e)||THEME_DEFAULT,i=deriveTheme(
n),a=document.documentElement;[["--theme-500",i.base],["--theme-600",i.dark],["--theme-700",i.darker],
["--theme-300",i.light],["--theme-200",i.lighter],["--theme-rgb",i.rgb]].forEach(([l,c])=>{a.style.getPropertyValue(
l).trim()!==String(c).trim()&&a.style.setProperty(l,c)}),t&&localStorage.setItem(THEME_STORAGE_KEY,n)},
"applyThemeColor"),applyLightMode=o(e=>{const t=!!e;let n=get("manual-theme-light-css");if(t&&!n){const i=window.
CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.manualLightTheme;if(!i)return;n=document.
createElement("link"),n.id="manual-theme-light-css",n.rel="stylesheet",n.href=i,document.head.appendChild(
n)}else!t&&n&&n.remove()},"applyLightMode"),syncThemeInputs=o(e=>{const t=normalizeHex(e)||THEME_DEFAULT,
n=get("set-theme-color"),i=get("set-theme-color-text");n&&(n.value=t),i&&(i.value=t),document.querySelectorAll(
"#theme-presets .theme-swatch").forEach(r=>{const l=normalizeHex(r.getAttribute("data-color"));r.classList.
toggle("active",l===t)})},"syncThemeInputs"),initThemeFromServer=o(()=>{INITIAL_LIGHT_MODE_ENABLED&&
applyLightMode(!0);const e=normalizeHex(INITIAL_THEME_COLOR);if(e){applyThemeColor(e,!1);return}const t=normalizeHex(
localStorage.getItem(THEME_STORAGE_KEY));applyThemeColor(t||THEME_DEFAULT,!1)},"initThemeFromServer"),
LIQUID_GLASS_SURFACE_SELECTOR=["#sidebar",".composer-dock","body > .flex-1 > header","#top-model-bar",
".modal-panel",".modal-glass-panel",".viewer-toolbar",".viewer-meta","#quote-bar","#slash-command-su\
ggestions","#gem-suggestions","#total-token-bar"].join(","),refreshLiquidGlassSurfaces=o(()=>{document.
querySelectorAll(LIQUID_GLASS_SURFACE_SELECTOR).forEach(e=>{e.classList.add("liquid-glass-surface"),
e.matches(".viewer-toolbar, .viewer-meta")&&e.classList.add("liquid-glass-clear");const t=e.matches(
'[data-liquid-glass-background="none"]')||!!e.closest(".liquid-glass-no-backdrop");e.classList.toggle(
"liquid-glass-no-background",t)})},"refreshLiquidGlassSurfaces"),applyLiquidGlassMode=o(e=>{document.
body&&(document.body.classList.toggle("liquid-glass-mode",!!e),e&&refreshLiquidGlassSurfaces())},"ap\
plyLiquidGlassMode");let pendingLiquidGlassPointer=null,liquidGlassPointerFrame=0,liquidGlassPointerPaintAt=0,
liquidGlassPointerSurface=null,liquidGlassPointerRect=null;const paintLiquidGlassPointer=o(e=>{if(!pendingLiquidGlassPointer||
!document.body||!document.body.classList.contains("liquid-glass-mode")){liquidGlassPointerFrame=0;return}
if(e-liquidGlassPointerPaintAt<30){liquidGlassPointerFrame=requestAnimationFrame(paintLiquidGlassPointer);
return}const t=pendingLiquidGlassPointer;pendingLiquidGlassPointer=null;const n=t.target&&t.target.closest?
t.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;if(!n){liquidGlassPointerFrame=0;return}(n!==liquidGlassPointerSurface||
!liquidGlassPointerRect)&&(liquidGlassPointerSurface=n,liquidGlassPointerRect=n.getBoundingClientRect());
const i=liquidGlassPointerRect;if(i.width&&i.height){const a=Math.max(0,Math.min(100,(t.clientX-i.left)/
i.width*100)),r=Math.max(0,Math.min(100,(t.clientY-i.top)/i.height*100));n.style.setProperty("--glas\
s-light-x",`${a.toFixed(1)}%`),n.style.setProperty("--glass-light-y",`${r.toFixed(1)}%`),liquidGlassPointerPaintAt=
e}liquidGlassPointerFrame=pendingLiquidGlassPointer?requestAnimationFrame(paintLiquidGlassPointer):0},
"paintLiquidGlassPointer");document.addEventListener("pointermove",e=>{!document.body||!document.body.
classList.contains("liquid-glass-mode")||(pendingLiquidGlassPointer={target:e.target,clientX:e.clientX,
clientY:e.clientY},liquidGlassPointerFrame||(liquidGlassPointerFrame=requestAnimationFrame(paintLiquidGlassPointer)))},
{passive:!0}),document.addEventListener("pointerout",e=>{const t=e.target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):
null;!t||e.relatedTarget&&t.contains(e.relatedTarget)||(pendingLiquidGlassPointer=null,t.style.removeProperty(
"--glass-light-x"),t.style.removeProperty("--glass-light-y"),t.classList.remove("liquid-glass-presse\
d"),t===liquidGlassPointerSurface&&(liquidGlassPointerSurface=null,liquidGlassPointerRect=null))},{passive:!0}),
document.addEventListener("pointerdown",e=>{if(!document.body||!document.body.classList.contains("li\
quid-glass-mode"))return;const t=e.target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;
t&&t.classList.add("liquid-glass-pressed")},{passive:!0});const releaseLiquidGlassPress=o(e=>{const t=e.
target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;t&&t.classList.remove("liquid-gl\
ass-pressed")},"releaseLiquidGlassPress");document.addEventListener("pointerup",releaseLiquidGlassPress,
{passive:!0}),document.addEventListener("pointercancel",releaseLiquidGlassPress,{passive:!0});let liquidGlassScrollTimer=0;
document.addEventListener("scroll",()=>{!document.body||!document.body.classList.contains("liquid-gl\
ass-mode")||(liquidGlassPointerRect=null,document.body.classList.add("liquid-glass-scrolling"),window.
clearTimeout(liquidGlassScrollTimer),liquidGlassScrollTimer=window.setTimeout(()=>{document.body&&document.
body.classList.remove("liquid-glass-scrolling")},140))},{passive:!0,capture:!0}),window.addEventListener(
"resize",()=>{liquidGlassPointerRect=null},{passive:!0});const MODAL_ANIM_MS=280,formatBytes=o(e=>{if(e==
null)return"0MB";const t=e/(1024*1024);return t<1024?`${t.toFixed(1)}MB`:`${(t/1024).toFixed(2)}GB`},
"formatBytes"),inspectSiteCacheStorage=o(async()=>{const e={cacheCount:0,entryCount:0,totalBytes:0,storageUsageBytes:null,
storageQuotaBytes:null};if("caches"in window)try{const t=await caches.keys();e.cacheCount=t.length;for(const n of t){
const i=await caches.open(n),a=await i.keys();e.entryCount+=a.length;for(const r of a)try{const l=await i.
match(r);if(!l)continue;const c=parseInt(l.headers.get("content-length")||"",10);if(Number.isFinite(
c)&&c>=0)e.totalBytes+=c;else{const u=await l.clone().blob();e.totalBytes+=u.size||0}}catch{}}}catch{}
if(navigator.storage&&navigator.storage.estimate)try{const t=await navigator.storage.estimate();e.storageUsageBytes=
Number(t.usage||0),e.storageQuotaBytes=Number(t.quota||0)}catch{}return e},"inspectSiteCacheStorage"),
loadSiteCacheUsage=o(async()=>{const e=get("site-cache-usage-text"),t=get("site-cache-usage-detail");
if(!(!e&&!t)){e&&(e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D..."),t&&(t.innerText="");try{const n=await inspectSiteCacheStorage(),
i=`\u30AD\u30E3\u30C3\u30B7\u30E5\u4F7F\u7528\u91CF: ${formatBytes(n.totalBytes)} (${n.cacheCount}\u30AD\u30E3\
\u30C3\u30B7\u30E5 / ${n.entryCount}\u4EF6)`;if(n.storageQuotaBytes){const a=Math.min(100,Math.round(
n.totalBytes/n.storageQuotaBytes*100));if(e&&(e.innerText=`${i} / \u4FDD\u5B58\u9818\u57DF\u4E0A\u9650 ${formatBytes(
n.storageQuotaBytes)} (${a}%)`),t){const r=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: \u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
t.innerText=`${r} / \u30D6\u30E9\u30A6\u30B6\u306E\u5B9F\u6E2C\u5024\u3067\u3059`}}else e&&(e.innerText=
i),t&&(t.innerText=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4E0A\u9650\u306F\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u3067\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093")}catch{
e&&(e.innerText="\u30AD\u30E3\u30C3\u30B7\u30E5\u5BB9\u91CF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),
t&&(t.innerText="")}}},"loadSiteCacheUsage");let versionUpdateCachePreferenceSavePromise=Promise.resolve();
const loadStorageUsage=o(async()=>{const e=get("storage-usage-text"),t=get("storage-usage-bar");if(!(!e||
!t)){e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D...";try{const n=await apiFetch("/api/storage",{cache:"\
no-store"});if(!n.ok)throw new Error("HTTP "+n.status);const i=await n.json(),a=Number(i.used_bytes||
0),r=Number(i.limit_bytes||0);if(i.is_unlimited||!r)e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(a)}\
 (\u7121\u5236\u9650)`,t.style.width="0%",t.style.opacity="0.5";else{const l=Math.min(100,Math.round(
a/r*100));e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(a)} / ${formatBytes(r)} (${l}%)`,t.style.width=
`${l}%`,t.style.opacity="1"}}catch{e.innerText="\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
t.style.width="0%",t.style.opacity="0.5"}}},"loadStorageUsage"),clearSiteCacheAndReload=o(async(e,t={})=>{
const{scanFirst:n=!0}=t||{},i=e?e.innerText:"";e&&(e.disabled=!0,e.innerText="\u524A\u9664\u4E2D...");
try{const a=n?await inspectSiteCacheStorage():null;await purgeCaches();const r=a?`\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5 ${formatBytes(
a.totalBytes)} \u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002`:"\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002";
showToast(`${r} \u518D\u8AAD\u307F\u8FBC\u307F\u3057\u307E\u3059\u3002`,"success"),window.setTimeout(
()=>location.reload(),900)}catch{showToast("\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u306E\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{e&&(e.disabled=!1,e.innerText=i||"\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664")}},
"clearSiteCacheAndReload"),syncVersionUpdateCachePreferenceUi=o(()=>{const e=get("version-update-cle\
ar-cache");e&&(e.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate))},"syn\
cVersionUpdateCachePreferenceUi"),saveVersionUpdateCachePreference=o(async e=>{window.CHAT_CONFIG&&(window.
CHAT_CONFIG.clearCacheOnVersionUpdate=!!e);try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clear_cache_on_version_update:!!e})})}catch{}},
"saveVersionUpdateCachePreference");initThemeFromServer(),applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED),
measureInteractionFrames(!0);const modalCloseTimers=new WeakMap,modalOpenFrames=new WeakMap,cancelModalTransitions=o(
e=>{const t=modalCloseTimers.get(e);t&&(clearTimeout(t),modalCloseTimers.delete(e));const n=modalOpenFrames.
get(e);n&&(cancelAnimationFrame(n.first),n.second&&cancelAnimationFrame(n.second),modalOpenFrames.delete(
e))},"cancelModalTransitions"),showModal=o(e=>{const t=get(e);if(!t||t.classList.contains("modal-ope\
n"))return;cancelModalTransitions(t),t.classList.remove("hidden"),t.style.display="flex",t.classList.
remove("modal-close"),t.classList.remove("modal-open"),t.classList.add("modal-prep");const n={first:0,
second:0};n.first=requestAnimationFrame(()=>{n.second=requestAnimationFrame(()=>{modalOpenFrames.delete(
t),t.classList.remove("modal-prep"),t.classList.add("modal-open")})}),modalOpenFrames.set(t,n)},"sho\
wModal");window.showModal=showModal;const hideModal=o((e,t={})=>{const n=get(e);if(!n)return;cancelModalTransitions(
n);const i=!!(t&&t.skipConfirm),a=!!(t&&t.skipReset);if(e==="camera-capture-modal"&&cameraCapturePendingFiles.
length>0&&!i&&!cameraCaptureBusy){attachCameraCapturedFiles();return}if(e==="rich-paste-modal"&&!i&&
hasRichPasteContent()&&!confirm("\u8CBC\u308A\u4ED8\u3051\u305F\u5185\u5BB9\u3092\u7834\u68C4\u3057\u3066\u9589\u3058\u307E\u3059\u304B\uFF1F"))
return;if(e==="marker-modal"&&(markerState.row=null),e==="camera-capture-modal"&&(a||resetCameraCapturePending(),
stopCameraCaptureStream()),!n.classList.contains("modal-open")){n.style.display="none",n.classList.remove(
"modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden");return}n.classList.remove(
"modal-open"),n.classList.add("modal-close");const r=setTimeout(()=>{n.style.display="none",n.classList.
remove("modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden"),modalCloseTimers.delete(
n)},MODAL_ANIM_MS);modalCloseTimers.set(n,r)},"hideModal");window.hideModal=hideModal;const RICH_PASTE_ALLOWED_TAGS=[
"a","abbr","address","article","b","blockquote","br","caption","cite","code","col","colgroup","dd","\
del","details","div","dl","dt","em","figcaption","figure","h1","h2","h3","h4","h5","h6","hr","i","im\
g","kbd","li","main","mark","ol","p","pre","q","s","samp","section","small","span","strong","sub","s\
ummary","sup","table","tbody","td","th","thead","tfoot","time","tr","u","ul","var"],RICH_PASTE_ALLOWED_ATTR=[
"align","alt","cellpadding","cellspacing","class","colspan","datetime","dir","headers","height","hre\
f","lang","open","rel","reversed","rowspan","scope","src","start","style","target","title","type","v\
alue","width"],RICH_PASTE_SAFE_STYLE_PROPS=new Set(["align-items","align-self","background","backgro\
und-color","background-image","border","border-block-color","border-block-style","border-block-width",
"border-bottom","border-bottom-color","border-bottom-left-radius","border-bottom-right-radius","bord\
er-bottom-style","border-bottom-width","border-collapse","border-color","border-image","border-inlin\
e-color","border-inline-style","border-inline-width","border-left","border-left-color","border-left-\
style","border-left-width","border-radius","border-right","border-right-color","border-right-style",
"border-right-width","border-spacing","border-style","border-top","border-top-color","border-top-lef\
t-radius","border-top-right-radius","border-top-style","border-top-width","border-width","box-shadow",
"box-sizing","break-after","break-before","break-inside","clear","clip-path","color","column-gap","d\
irection","display","flex","flex-basis","flex-direction","flex-grow","flex-shrink","flex-wrap","floa\
t","font","font-family","font-feature-settings","font-kerning","font-language-override","font-optica\
l-sizing","font-size","font-size-adjust","font-stretch","font-style","font-variant","font-variant-ca\
ps","font-variant-ligatures","font-variation-settings","font-weight","gap","grid","grid-auto-columns",
"grid-auto-flow","grid-auto-rows","grid-column","grid-column-end","grid-column-start","grid-row","gr\
id-row-end","grid-row-start","grid-template","grid-template-areas","grid-template-columns","grid-tem\
plate-rows","height","hyphens","justify-content","justify-items","justify-self","letter-spacing","li\
ne-break","line-height","list-style","list-style-position","list-style-type","margin","margin-block",
"margin-block-end","margin-block-start","margin-bottom","margin-inline","margin-inline-end","margin-\
inline-start","margin-left","margin-right","margin-top","max-height","max-width","min-height","min-w\
idth","object-fit","object-position","opacity","order","orphans","outline","outline-color","outline-\
offset","outline-style","outline-width","overflow","overflow-wrap","overflow-x","overflow-y","paddin\
g","padding-block","padding-block-end","padding-block-start","padding-bottom","padding-inline","padd\
ing-inline-end","padding-inline-start","padding-left","padding-right","padding-top","page-break-afte\
r","page-break-before","page-break-inside","row-gap","table-layout","text-align","text-decoration","\
text-decoration-color","text-decoration-line","text-decoration-style","text-decoration-thickness","t\
ext-indent","text-overflow","text-shadow","text-transform","text-underline-offset","vertical-align",
"visibility","white-space","widows","width","word-break","word-spacing","writing-mode","-webkit-text\
-stroke","-webkit-text-stroke-color","-webkit-text-stroke-width"]),RICH_PASTE_NOISE_TAGS=new Set(["s\
cript","style","link","meta","noscript","iframe","canvas","svg","object","embed"]);let userSettingsSnapshot=null,
userSettingsSnapshotPromise=null,richPastePromptSaveTimer=null,richPastePromptPreferenceSyncing=!1;const getRichPasteEditor=o(
()=>get("rich-paste-storage"),"getRichPasteEditor"),getRichPasteCapture=o(()=>get("rich-paste-captur\
e"),"getRichPasteCapture"),getRichPastePrompt=o(()=>get("rich-paste-prompt"),"getRichPastePrompt"),getRichPasteUseDefaultCheckbox=o(
()=>get("rich-paste-use-default"),"getRichPasteUseDefaultCheckbox"),getRichPasteStatus=o(()=>get("ri\
ch-paste-status"),"getRichPasteStatus"),downloadBlob=o((e,t)=>{const n=URL.createObjectURL(e),i=document.
createElement("a");i.href=n,i.download=t,document.body.appendChild(i),i.click(),setTimeout(()=>{document.
body.removeChild(i),URL.revokeObjectURL(n)},100)},"downloadBlob"),getRichPasteEffectivePrompt=o((e=null)=>{
if(e&&e.rich_paste_prompt_use_custom_default){const t=String(e.rich_paste_prompt_default||"").trim();
if(t)return t}return RICH_PASTE_DEFAULT_PROMPT},"getRichPasteEffectivePrompt"),syncRichPastePromptPreferencesUi=o(
(e=null,t={})=>{const n=!!t.preservePrompt,i=getRichPastePrompt(),a=getRichPasteUseDefaultCheckbox();
a&&(a.checked=!!(e&&e.rich_paste_prompt_use_custom_default)),i&&!richPastePromptPreferenceSyncing&&!n&&
(i.value=getRichPasteEffectivePrompt(e))},"syncRichPastePromptPreferencesUi"),cacheUserSettings=o((e,t={})=>(userSettingsSnapshot=
e||null,syncRichPastePromptPreferencesUi(userSettingsSnapshot,t),userSettingsSnapshot),"cacheUserSet\
tings"),SETTINGS_LOAD_TIMEOUT_MS=15e3,fetchSettingsSnapshot=o(async()=>{const e=new AbortController,
t=setTimeout(()=>e.abort(),SETTINGS_LOAD_TIMEOUT_MS);try{const n=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,
{cache:"no-store",signal:e.signal});if(!n.ok)throw new Error("HTTP "+n.status);const i=await n.json();
if(!i||typeof i!="object")throw new Error("Invalid settings response");return cacheUserSettings(i)}finally{
clearTimeout(t)}},"fetchSettingsSnapshot"),ensureUserSettingsSnapshot=o(async()=>userSettingsSnapshot||
(userSettingsSnapshotPromise||(userSettingsSnapshotPromise=fetchSettingsSnapshot().catch(()=>null).finally(
()=>{userSettingsSnapshotPromise=null})),await userSettingsSnapshotPromise),"ensureUserSettingsSnaps\
hot"),saveRichPastePromptPreferences=o(async()=>{const e=getRichPastePrompt(),t=getRichPasteUseDefaultCheckbox();
if(!e||!t)return;const n={rich_paste_prompt_default:e.value||"",rich_paste_prompt_use_custom_default:!!t.
checked};try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(n)}),cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},n),{preservePrompt:!0})}catch{}},"saveRichPastePromptPreferences"),queueRichPastePromptPreferenceSave=o(
()=>{richPastePromptSaveTimer&&clearTimeout(richPastePromptSaveTimer),richPastePromptSaveTimer=setTimeout(
()=>{richPastePromptSaveTimer=null,saveRichPastePromptPreferences()},500)},"queueRichPastePromptPref\
erenceSave"),hasRichPasteContent=o(()=>{const e=getRichPasteEditor();return e?(e.textContent||"").trim()?
!0:!!e.querySelector("img,table,ul,ol,blockquote,h1,h2,h3,h4,h5,h6,pre,code"):!1},"hasRichPasteConte\
nt"),updateRichPasteStatus=o(()=>{const e=getRichPasteEditor(),t=getRichPasteStatus();if(!t||!e)return;
const n=(e.innerText||"").trim();if(!n){t.textContent="\u307E\u3060\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002";
return}const i=e.querySelectorAll("img").length,a=e.querySelectorAll("table").length,r=e.querySelectorAll(
"a").length,l=e.querySelectorAll("h1,h2,h3,h4,h5,h6").length;t.textContent=`${n.length} \u6587\u5B57 / \u753B\u50CF ${i}\
 / \u8868 ${a} / \u30EA\u30F3\u30AF ${r} / \u898B\u51FA\u3057 ${l}`},"updateRichPasteStatus"),focusRichPasteEditor=o(
()=>{const e=getRichPasteCapture();if(!e)return;e.focus(),e.value=e.value||"",window.getSelection&&window.
getSelection()&&e.select&&e.select()},"focusRichPasteEditor"),clearRichPasteEditor=o((e=!0)=>{const t=getRichPasteEditor();
t&&(t.innerHTML="");const n=getRichPasteCapture();if(n&&(n.value=""),!e){const i=getRichPastePrompt();
i&&(i.value=RICH_PASTE_DEFAULT_PROMPT)}updateRichPasteStatus()},"clearRichPasteEditor"),sanitizeRichPasteStyle=o(
e=>{if(!e)return"";const t=[];return String(e).split(";").forEach(n=>{const i=n.trim();if(!i)return;
const a=i.indexOf(":");if(a<=0)return;const r=i.slice(0,a).trim().toLowerCase(),l=i.slice(a+1).trim();
if(!RICH_PASTE_SAFE_STYLE_PROPS.has(r)||!l||l.length>1e3)return;const c=l.toLowerCase();c.includes("\
url(")||c.includes("expression(")||c.includes("javascript:")||c.includes("@import")||c.includes("beh\
avior:")||c.includes("-moz-binding")||c.includes("var(")||c.includes("env(")||t.push(`${r}: ${l}`)}),
t.join("; ")},"sanitizeRichPasteStyle");let richPasteColorCanvasContext=null;const parseRichPasteCssColor=o(
e=>{const t=String(e||"").trim();if(!t||t==="inherit"||t==="currentcolor"||t==="transparent"||window.
CSS&&typeof window.CSS.supports=="function"&&!window.CSS.supports("color",t))return null;try{if(!richPasteColorCanvasContext){
const a=document.createElement("canvas");a.width=1,a.height=1,richPasteColorCanvasContext=a.getContext(
"2d",{willReadFrequently:!0})}const n=richPasteColorCanvasContext;if(!n)return null;n.clearRect(0,0,
1,1),n.fillStyle="rgba(1, 2, 3, 0.004)",n.fillStyle=t,n.fillRect(0,0,1,1);const i=n.getImageData(0,0,
1,1).data;return!i||i[3]===0?null:{r:i[0],g:i[1],b:i[2],a:i[3]/255}}catch{return null}},"parseRichPa\
steCssColor"),richPasteColorLuminance=o(e=>{if(!e)return 0;const t=o(n=>{const i=Math.max(0,Math.min(
255,Number(n)||0))/255;return i<=.04045?i/12.92:Math.pow((i+.055)/1.055,2.4)},"channel");return .2126*
t(e.r)+.7152*t(e.g)+.0722*t(e.b)},"richPasteColorLuminance"),richPasteColorContrast=o((e,t)=>{const n=richPasteColorLuminance(
e),i=richPasteColorLuminance(t);return(Math.max(n,i)+.05)/(Math.min(n,i)+.05)},"richPasteColorContra\
st"),richPasteColorCss=o(e=>e?`rgb(${Math.round(e.r)}, ${Math.round(e.g)}, ${Math.round(e.b)})`:"","\
richPasteColorCss"),makeRichPasteTheme=o((e,t)=>{const n=richPasteColorLuminance(e)<.32;let i=t;return(!i||
richPasteColorContrast(e,i)<3)&&(i=n?{r:244,g:244,b:245,a:1}:{r:17,g:24,b:39,a:1}),{mode:n?"dark":"l\
ight",background:richPasteColorCss(e),foreground:richPasteColorCss(i),muted:n?"rgb(161, 161, 170)":"\
rgb(100, 116, 139)",border:n?"rgb(63, 63, 70)":"rgb(203, 213, 225)",surface:n?"rgb(33, 33, 33)":"rgb\
(248, 250, 252)",quote:n?"rgb(39, 39, 42)":"rgb(255, 249, 235)",link:n?"rgb(125, 211, 252)":"rgb(15,\
 118, 110)"}},"makeRichPasteTheme"),detectRichPasteTheme=o(e=>{const t={r:255,g:255,b:255,a:1},n={r:17,
g:24,b:39,a:1},i=document.createElement("template");if(i.innerHTML=String(e||""),!i.content.querySelector(
"*"))return makeRichPasteTheme(t,n);const a=document.createElement("div");a.setAttribute("aria-hidde\
n","true"),a.style.position="fixed",a.style.left="-100000px",a.style.top="0",a.style.width="794px",a.
style.visibility="hidden",a.style.pointerEvents="none",a.style.color="#111827",a.style.background="t\
ransparent",a.appendChild(i.content.cloneNode(!0)),document.body.appendChild(a);try{const r=[a,...Array.
from(a.querySelectorAll("*")).slice(0,5e3)],l=[],c=new Map;let u=0;const f=o(k=>Array.from(k.childNodes||
[]).reduce((_,C)=>C&&C.nodeType===Node.TEXT_NODE?_+String(C.textContent||"").replace(/\s+/g," ").trim().
length:_,0),"directTextLength");r.forEach(k=>{if(!k||k===a||!k.style)return;const _=window.getComputedStyle(
k),C=f(k);if(C>0){const E=parseRichPasteCssColor(_.color);if(E&&E.a>=.5){const B=richPasteColorCss(E),
K=c.get(B)||{color:E,weight:0};K.weight+=C,c.set(B,K),u+=C}}if(!!(String(k.style.backgroundColor||"").
trim()||String(k.style.background||"").trim())){const E=parseRichPasteCssColor(_.backgroundColor);if(E&&
E.a>=.72){const B=String(k.textContent||"").replace(/\s+/g," ").trim().length;l.push({color:E,weight:Math.
max(1,B)})}}});const g=Array.from(c.values()).sort((k,_)=>_.weight-k.weight),y=g.length?g[0].color:null,
w=g.reduce((k,_)=>k+(richPasteColorLuminance(_.color)>=.6?_.weight:0),0);l.sort((k,_)=>_.weight-k.weight);
let v=l.length?l[0].color:null;return v||(v=u>0&&w/u>=.55?{r:11,g:11,b:12,a:1}:t),makeRichPasteTheme(
v,y||n)}catch{return makeRichPasteTheme(t,n)}finally{a.parentNode&&a.parentNode.removeChild(a)}},"de\
tectRichPasteTheme"),prepareRichPastePdfClone=o((e,t)=>{if(!e)return;const n=e.head||e.querySelector(
"head");n&&Array.from(n.querySelectorAll('link[rel="stylesheet"]')).forEach(i=>{try{i.remove()}catch{}}),
e.body&&(e.body.style.margin="0",e.body.style.background=t.background,e.body.style.color=t.foreground)},
"prepareRichPastePdfClone"),normalizeRichPasteTree=o(e=>{!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("*").forEach(t=>{if(!t||!t.getAttribute||!t.parentNode)return;const n=String(t.tagName||
"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(n)){t.remove();return}t.removeAttribute("class"),t.removeAttribute(
"id"),t.removeAttribute("role"),t.removeAttribute("aria-label"),n==="img"&&(t.setAttribute("loading",
"eager"),t.setAttribute("decoding","sync"),t.removeAttribute("srcset"),t.removeAttribute("sizes"));const i=t.
getAttribute("style");if(i){const a=sanitizeRichPasteStyle(i);a?t.setAttribute("style",a):t.removeAttribute(
"style")}})},"normalizeRichPasteTree"),extractRichPasteArticleHtml=o(e=>{const n=new DOMParser().parseFromString(
String(e||""),"text/html");if(!n.body)return"";const i=(n.body.textContent||"").replace(/\s+/g," ").
trim().length,a=n.body.querySelectorAll("*").length;if(i<1e3||a<120)return n.body.innerHTML;const l=[
...Array.from(n.body.querySelectorAll("article")),...Array.from(n.body.querySelectorAll("main")),...Array.
from(n.body.querySelectorAll('[role="main"],[role="article"]'))].filter(u=>(u.textContent||"").replace(
/\s+/g," ").trim().length>=i*.65);l.sort((u,f)=>{const g=+!!f.querySelector("h1")-+!!u.querySelector(
"h1");return g||u.querySelectorAll("*").length-f.querySelectorAll("*").length});const c=l[0]||null;return c?
c.outerHTML:n.body.innerHTML},"extractRichPasteArticleHtml"),sanitizeRichPasteHtml=o(e=>{if(!window.
DOMPurify||typeof window.DOMPurify.sanitize!="function"){const a=new DOMParser().parseFromString(String(
e||""),"text/html");return escapeHtml(a.body?a.body.textContent:"")}let t=extractRichPasteArticleHtml(
e),n=window.DOMPurify.sanitize(t||"",{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,
KEEP_CONTENT:!0});if((!n||n.trim()==="")&&e&&e.trim()!==""&&(n=window.DOMPurify.sanitize(e,{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,
ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,KEEP_CONTENT:!0})),!n)return"";const i=document.createElement("\
template");return i.innerHTML=n,normalizeRichPasteTree(i.content),i.innerHTML},"sanitizeRichPasteHtm\
l"),normalizeRichPastePrintHtml=o(e=>{const t=document.createElement("template");t.innerHTML=String(
e||"");const n=Array.from(t.content.querySelectorAll("*")),i=n.reduce((u,f)=>{const g=String(f.style&&
f.style.display||"").trim().toLowerCase();return u+(["flex","inline-flex","grid","inline-grid"].includes(
g)?1:0)},0),a=n.reduce((u,f)=>{if(!f||!f.style||!["article","div","main","section"].includes(String(
f.tagName||"").toLowerCase()))return u;const g=String(f.getAttribute("style")||""),y=Array.from(g.matchAll(
/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)).some(v=>Array.
from(v[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi)).some(k=>Math.abs(Number(k[1])||0)>=96)),w=Array.from(g.
matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)).some(v=>Math.abs(Number(v[1])||
0)>720);return u+(y||w?1:0)},0);if(n.length<=500&&i<=24&&a===0)return t.innerHTML;const r=new Set(["\
align-items","align-self","column-gap","flex","flex-basis","flex-direction","flex-grow","flex-shrink",
"flex-wrap","gap","grid","grid-auto-columns","grid-auto-flow","grid-auto-rows","grid-column","grid-c\
olumn-end","grid-column-start","grid-row","grid-row-end","grid-row-start","grid-template","grid-temp\
late-areas","grid-template-columns","grid-template-rows","justify-content","justify-items","justify-\
self","order","row-gap"]),l=new Set(["article","div","main","section"]),c=new Set(["padding","paddin\
g-left","padding-right","padding-inline","padding-inline-start","padding-inline-end"]);return n.forEach(
u=>{if(!u||!u.style)return;const f=String(u.tagName||"").toLowerCase(),g=[];String(u.getAttribute("s\
tyle")||"").split(";").forEach(y=>{if(!y||y.indexOf(":")<0)return;const w=y.indexOf(":"),v=y.slice(0,
w).trim().toLowerCase();let k=y.slice(w+1).trim();if(!(!v||!k||r.has(v))&&!["height","max-height","m\
in-height","overflow","overflow-x","overflow-y"].includes(v)&&!(["width","min-width"].includes(v)&&l.
has(f))){if(c.has(v)&&l.has(f)&&Array.from(k.matchAll(/(-?\d+(?:\.\d+)?)px/gi)).map(C=>Math.abs(Number(
C[1])||0)).some(C=>C>=96)&&(k="0px"),v==="display"){const _=k.toLowerCase();["flex","grid"].includes(
_)?k="block":["inline-flex","inline-grid"].includes(_)&&(k="inline-block")}g.push(`${v}: ${k}`)}}),g.
length?u.setAttribute("style",g.join("; ")):u.removeAttribute("style")}),t.innerHTML},"normalizeRich\
PastePrintHtml"),getRichPasteSelectionRange=o(e=>{const t=window.getSelection&&window.getSelection();
if(!t||!t.rangeCount)return null;const n=t.getRangeAt(0);if(e&&e.contains(n.commonAncestorContainer))
return n;const i=document.createRange();return i.selectNodeContents(e),i.collapse(!1),i},"getRichPas\
teSelectionRange"),insertNodeIntoRichPasteEditor=o(e=>{const t=getRichPasteEditor();!t||!e||(t.appendChild(
e),updateRichPasteStatus())},"insertNodeIntoRichPasteEditor"),insertHtmlIntoRichPasteEditor=o(e=>{const t=sanitizeRichPasteHtml(
e);if(!t||t.trim()==="")return!1;const n=document.createElement("template");n.innerHTML=t;const i=n.
content.cloneNode(!0);return insertNodeIntoRichPasteEditor(i),!0},"insertHtmlIntoRichPasteEditor"),insertTextIntoRichPasteEditor=o(
e=>{if(e==null)return;const t=document.createTextNode(String(e));insertNodeIntoRichPasteEditor(t)},"\
insertTextIntoRichPasteEditor"),blobToDataUrl=o(e=>new Promise((t,n)=>{const i=new FileReader;i.onload=
()=>t(String(i.result||"")),i.onerror=()=>n(i.error||new Error("clipboard_image_read_failed")),i.readAsDataURL(
e)}),"blobToDataUrl"),insertClipboardImageBlob=o(async(e,t="clipboard-image")=>{if(!e)return!1;const n=await blobToDataUrl(
e);return n?(insertHtmlIntoRichPasteEditor(`<p><img src="${escapeHtml(n)}" alt="${escapeHtml(t)}"></\
p>`),!0):!1},"insertClipboardImageBlob"),readClipboardRichContent=o(async()=>{if(!navigator.clipboard||
!navigator.clipboard.read)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30EA\u30C3\u30C1\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u8AAD\u307F\u53D6\u308A\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
const e=getRichPasteCapture();e&&(e.value="");const t=await navigator.clipboard.read();if(!t||!t.length)
return!1;let n=!1;for(const i of t){if(!i)continue;const a=Array.from(i.types||[]);let r=!1;if(a.includes(
"text/html")){const u=await(await i.getType("text/html")).text();u&&insertHtmlIntoRichPasteEditor(u)&&
(n=!0,r=!0)}if(!r&&a.includes("text/plain")){const u=await(await i.getType("text/plain")).text();u&&
(insertTextIntoRichPasteEditor(u),n=!0)}const l=a.find(c=>c&&c.startsWith("image/"));if(!r&&l){const c=await i.
getType(l);await insertClipboardImageBlob(c,"clipboard-image")&&(n=!0)}}return n},"readClipboardRich\
Content"),ingestRichPasteClipboardData=o(async e=>{if(!e)return!1;let t=!1;const n=e.getData&&e.getData(
"text/html"),i=e.getData&&e.getData("text/plain");let a=!1;n&&insertHtmlIntoRichPasteEditor(n)&&(t=!0,
a=!0),!a&&i&&(insertTextIntoRichPasteEditor(i),t=!0);const l=Array.from(e.items||[]).filter(c=>c&&c.
kind==="file").map(c=>c.getAsFile()).filter(c=>c&&c.type&&c.type.startsWith("image/"));if(!a&&l.length)
for(const c of l)try{await insertClipboardImageBlob(c,c.name||"clipboard-image")&&(t=!0)}catch{}return t},
"ingestRichPasteClipboardData"),buildRichPastePdfFilename=o(()=>{const e=new Date,t=o(n=>String(n).padStart(
2,"0"),"pad");return`clipboard_rich_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(
e.getMinutes())}${t(e.getSeconds())}.pdf`},"buildRichPastePdfFilename"),getRichPasteProgressElements=o(
()=>({container:get("rich-paste-progress-container"),bar:get("rich-paste-progress-bar"),text:get("ri\
ch-paste-progress-text")}),"getRichPasteProgressElements"),setRichPasteProgress=o((e,t=null)=>{const{
container:n,bar:i,text:a}=getRichPasteProgressElements(),r=Math.max(0,Math.min(100,Number(e)||0));if(n&&
(n.classList.remove("hidden"),n.style.setProperty("display","block","important")),i&&(i.style.width=
`${r}%`,i.style.transform="none"),a&&(a.textContent=`${Math.round(r)}%`),t&&n){const l=n.querySelector(
".text-amber-400");l&&(l.innerHTML=`<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(t)}`)}},"set\
RichPasteProgress"),hideRichPasteProgress=o(()=>{const{container:e,bar:t}=getRichPasteProgressElements();
t&&(t.style.transform="scaleX(0)"),e&&(e.classList.add("hidden"),e.style.display="none")},"hideRichP\
asteProgress"),inferRichPasteTitle=o(()=>{const e=getRichPasteEditor();if(!e)return"Clipboard Export";
const t=e.querySelector("h1, h2, h3, h4, h5, h6");if(t&&t.textContent&&t.textContent.trim())return t.
textContent.trim().slice(0,48);const n=(e.innerText||"").trim().replace(/\s+/g," ");return n?n.slice(
0,48):"Clipboard Export"},"inferRichPasteTitle"),waitForRichPasteMedia=o(async(e,t=2500)=>{if(!e)return;
const n=new Promise(a=>setTimeout(a,Math.max(0,t))),i=Promise.all(Array.from(e.querySelectorAll("img")||
[]).map(a=>!a||a.complete?Promise.resolve():new Promise(r=>{let l=!1;const c=o(()=>{l||(l=!0,r())},"\
finish");a.addEventListener("load",c,{once:!0}),a.addEventListener("error",c,{once:!0}),setTimeout(c,
Math.max(250,Math.min(t,2e3)))})));if(await Promise.race([i,n]),document.fonts&&document.fonts.ready)
try{await Promise.race([document.fonts.ready,n])}catch{}},"waitForRichPasteMedia"),normalizeRichPastePdfText=o(
e=>String(e||"").replace(/\u00a0/g," ").replace(/\r\n?/g,`
`).replace(/[ \t\f\v]+/g," ").replace(/\n[ \t]+/g,`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).trim(),"normalizeRichPastePdfText"),normalizeRichPastePdfCodeText=o(e=>String(e||"").replace(/\u00a0/g,
" ").replace(/\r\n?/g,`
`),"normalizeRichPastePdfCodeText"),collectRichPasteInlineSegments=o((e,t={})=>{if(!e)return[];const n=t.
allowLinks!==!1,i=[],a=o((r,l)=>{if(!r)return;if(r.nodeType===Node.TEXT_NODE){const f=r.textContent||
"";f&&i.push(Object.assign({},l,{text:f}));return}if(r.nodeType!==Node.ELEMENT_NODE)return;const c=String(
r.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(c))return;if(c==="br"){i.push({text:`
`});return}const u=Object.assign({},l);["b","strong"].includes(c)&&(u.bold=!0),["i","em"].includes(c)&&
(u.italic=!0),c==="a"&&n&&(u.link=String(r.getAttribute("href")||"").trim()),c==="code"&&(u.monospace=
!0),Array.from(r.childNodes||[]).forEach(f=>a(f,u))},"walk");return a(e,{bold:!!t.bold,italic:!!t.italic}),
i},"collectRichPasteInlineSegments"),collectRichPasteInlineText=o((e,t={})=>collectRichPasteInlineSegments(
e,t).map(i=>i.text).join(""),"collectRichPasteInlineText"),collectRichPasteTableRows=o(e=>{const t=[];
return Array.from(e.querySelectorAll("tr")||[]).forEach(n=>{n&&n.closest&&n.closest("table")===e&&t.
push(n)}),t},"collectRichPasteTableRows"),makeRichPasteTableMarkdown=o(e=>{const t=e&&e.querySelector?
e.querySelector("caption"):null,n=t?normalizeRichPastePdfText(collectRichPasteInlineText(t)):"",i=collectRichPasteTableRows(
e).map(u=>Array.from(u.children||[]).filter(g=>{const y=String(g.tagName||"").toLowerCase();return y===
"th"||y==="td"}).map(g=>normalizeRichPastePdfText(collectRichPasteInlineText(g))||" ")).filter(u=>u.
length);if(!i.length)return n||"[table]";const a=i.reduce((u,f)=>Math.max(u,f.length),0),r=i.map(u=>{
const f=u.slice(0,a);for(;f.length<a;)f.push(" ");return f}),l=`| ${Array(a).fill("---").join(" | ")}\
 |`,c=[];n&&(c.push(`Table: ${n}`),c.push("")),c.push(`| ${r[0].join(" | ")} |`),c.push(l);for(let u=1;u<
r.length;u+=1)c.push(`| ${r[u].join(" | ")} |`);return c.join(`
`)},"makeRichPasteTableMarkdown"),collectRichPasteListBlocks=o((e,t=!1,n=0)=>{const i=[],a=Array.from(
e.children||[]).filter(l=>String(l.tagName||"").toLowerCase()==="li");let r=1;return a.forEach(l=>{const c=l.
cloneNode(!0);Array.from(c.querySelectorAll("ul,ol")||[]).forEach(f=>{try{f.remove()}catch{}});const u=collectRichPasteInlineSegments(
c);u.length>0&&i.push({type:"list_item",ordered:t,depth:n,index:r,segments:u}),Array.from(l.children||
[]).forEach(f=>{const g=String(f.tagName||"").toLowerCase();(g==="ul"||g==="ol")&&i.push(...collectRichPasteListBlocks(
f,g==="ol",n+1))}),r+=1}),i},"collectRichPasteListBlocks"),collectRichPastePdfBlocks=o((e,t=0)=>{const n=[];
if(!e)return n;let i=[];const a=o(()=>{i.length!==0&&(n.push({type:"paragraph",segments:[...i]}),i=[])},
"flushBuffer");return Array.from(e.childNodes||[]).forEach(r=>{if(!r)return;if(r.nodeType===Node.TEXT_NODE){
const f=(r.textContent||"").replace(/\u00a0/g," ");f&&i.push({text:f});return}if(r.nodeType!==Node.ELEMENT_NODE)
return;const l=String(r.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(l))return;if(l==="br"){
i.push({text:`
`});return}if(/^h[1-6]$/.test(l)){a();const f=collectRichPasteInlineSegments(r);f.length>0&&n.push({
type:"heading",level:Number(l.slice(1))||1,segments:f});return}if(l==="p"){a();const f=collectRichPasteInlineSegments(
r);f.length>0&&n.push({type:"paragraph",segments:f});return}if(l==="blockquote"){a();const f=collectRichPasteInlineSegments(
r,{italic:!0});f.length>0&&n.push({type:"blockquote",segments:f});return}if(l==="pre"){a();const f=normalizeRichPastePdfCodeText(
r.innerText||r.textContent||"");f.trim()&&n.push({type:"code",text:f});return}if(l==="table"){a();const f=makeRichPasteTableMarkdown(
r);f&&n.push({type:"table",text:f});return}if(l==="ul"||l==="ol"){a(),n.push(...collectRichPasteListBlocks(
r,l==="ol",t));return}if(l==="hr"){a(),n.push({type:"hr"});return}if(l==="figure"){a();const f=r.querySelector(
"img");f&&n.push({type:"image",src:String(f.getAttribute("src")||"").trim(),alt:String(f.getAttribute(
"alt")||f.getAttribute("title")||"").trim(),title:String(f.getAttribute("title")||"").trim()});const g=r.
querySelector("figcaption");if(g){const y=collectRichPasteInlineSegments(g);y.length>0&&n.push({type:"\
paragraph",segments:y})}return}if(l==="img"){a(),n.push({type:"image",src:String(r.getAttribute("src")||
"").trim(),alt:String(r.getAttribute("alt")||r.getAttribute("title")||"").trim(),title:String(r.getAttribute(
"title")||"").trim()});return}if(l==="li"){a(),n.push(...collectRichPasteListBlocks(r,!1,t));return}
if(Array.from(r.children||[]).some(f=>{const g=String(f.tagName||"").toLowerCase();return/^h[1-6]$/.
test(g)||["p","div","section","article","main","blockquote","pre","table","ul","ol","hr","figure","i\
mg","li"].includes(g)})&&["div","section","article","main","figure"].includes(l)){a(),n.push(...collectRichPastePdfBlocks(
r,t+1));return}const u=collectRichPasteInlineSegments(r);u.length>0&&i.push(...u)}),a(),n},"collectR\
ichPastePdfBlocks"),detectImageMimeType=o(e=>{const t=String(e||"").match(/^data:(image\/[a-z0-9.+-]+);/i);
return t?t[1].toLowerCase():"image/png"},"detectImageMimeType"),loadRichPasteImageData=o(async(e,t=3e3)=>{
const n=String(e||"").trim();if(!n)return null;if(n.startsWith("data:image/"))return{dataUrl:n,mimeType:detectImageMimeType(
n)};let i=null;try{i=new URL(n,window.location.href)}catch{return null}if(!(i.origin===window.location.
origin))return null;const r=(async()=>{try{const l=await fetch(i.toString(),{credentials:"same-origi\
n",cache:"force-cache"});if(!l.ok)return null;const c=await l.blob(),u=await blobToDataUrl(c);return{
dataUrl:u,mimeType:c.type||detectImageMimeType(u)}}catch{return null}})();return await Promise.race(
[r,new Promise(l=>setTimeout(()=>l(null),Math.max(250,t)))])},"loadRichPasteImageData"),buildRichPastePreviewHtml=o(
(e="preview")=>{const t=getRichPasteEditor();if(!t)return"";const n=inferRichPasteTitle(),i=new Date().
toLocaleString("ja-JP"),a=sanitizeRichPasteHtml(t.innerHTML||""),r=detectRichPasteTheme(a),l=normalizeRichPastePrintHtml(
a),c=e==="pdf";return`<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(n)} - Preview</title>
  <style>
        :root {
          color-scheme: ${r.mode};
          --rp-background: ${r.background};
          --rp-foreground: ${r.foreground};
          --rp-muted: ${r.muted};
          --rp-border: ${r.border};
          --rp-surface: ${r.surface};
          --rp-quote: ${r.quote};
          --rp-link: ${r.link};
        }
	    body { margin: 0; background: ${c?"var(--rp-background)":"#eef2f7"}; color: var(--rp-foreground\
); font-family: "Noto Sans JP", system-ui, sans-serif; }
	    .page { max-width: ${c?"794px":"920px"}; margin: 0 auto; padding: ${c?"28px 30px 36px":"24px"};\
 }
	    .card { background: var(--rp-background); color: var(--rp-foreground); border: 1px solid var(--\
rp-border); border-radius: 18px; padding: 20px; box-shadow: ${c?"none":"0 18px 45px rgba(15,23,42,0.\
14)"}; }
	    .title { margin: 0; font-size: ${c?"22px":"24px"}; line-height: 1.35; color: var(--rp-foregroun\
d); }
	    .meta { margin-top: 8px; color: var(--rp-muted); font-size: 12px; }
	    .content { margin-top: 18px; color: var(--rp-foreground); font-size: 15px; line-height: 1.7; wo\
rd-break: break-word; overflow-wrap: anywhere; }
	    .content img, .content video, .content iframe, .content table, .content pre, .content blockquot\
e { max-width: 100%; }
	    .content table { display: block; overflow-x: auto; border-collapse: collapse; }
	    .content th, .content td { border: 1px solid var(--rp-border); padding: 8px 10px; }
	    .content th { background: var(--rp-surface); }
	    .content pre { padding: 14px 16px; border: 1px solid var(--rp-border); border-radius: 14px; bac\
kground: var(--rp-surface); color: var(--rp-foreground); overflow: auto; }
	    .content code { background: var(--rp-surface); color: var(--rp-foreground); }
	    .content pre code { background: transparent; }
	    .content blockquote { margin: 1em 0; padding: 12px 16px; border-left: 4px solid #f59e0b; backgr\
ound: var(--rp-quote); color: var(--rp-foreground); border-radius: 12px; }
	    .content a { color: var(--rp-link); }
    .toolbar { display:${c?"none":"flex"}; gap:10px; margin-top: 16px; flex-wrap: wrap; }
    .toolbar button { border: 1px solid var(--rp-border); background: var(--rp-surface); color: var(\
--rp-foreground); border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    ${c?".card { border-radius: 0; } .page { max-width: none; padding: 0; }":""}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1 class="title">${escapeHtml(n)}</h1>
      <div class="meta">Clipboard import | ${escapeHtml(i)} | \u672C\u6587\u78BA\u8A8D\u7528\u30D7\u30EC\u30D3\u30E5\u30FC</div>
      <div class="toolbar">
        <button onclick="window.close()">\u9589\u3058\u308B</button>
      </div>
      <div class="content">${l||"<p>\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</p>"}</di\
v>
    </div>
  </div>
</body>
</html>`},"buildRichPastePreviewHtml"),openSandboxedHtmlTab=o(e=>{const n=`<!doctype html><html><hea\
d><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:10\
0%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe i\
d="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referre\
r"></iframe><script>document.getElementById('preview').srcdoc=${JSON.stringify(String(e||"")).replace(
/</g,"\\u003c").replace(/\u2028/g,"\\u2028").replace(/\u2029/g,"\\u2029")};<\/script></body></html>`,
i=new Blob([n],{type:"text/html;charset=utf-8"}),a=URL.createObjectURL(i);return window.open(a,"_bla\
nk","noopener,noreferrer")?(setTimeout(()=>URL.revokeObjectURL(a),6e4),!0):(URL.revokeObjectURL(a),!1)},
"openSandboxedHtmlTab"),openRichPastePreviewTab=o(()=>{const e=buildRichPastePreviewHtml("preview");
if(!e){showToast("\u78BA\u8A8D\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093","warning",
!0);return}openSandboxedHtmlTab(e)||showToast("\u5225\u30BF\u30D6\u306E\u8868\u793A\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},"openRichPastePreviewTab"),renderRichPastePdfBlob=o(async()=>{const e=get("rich-paste-p\
rogress-container"),t=get("rich-paste-progress-bar"),n=get("rich-paste-progress-text"),i=o(w=>{const v=Math.
max(0,Math.min(100,Number(w)||0));t&&(t.style.width="100%",t.style.transformOrigin="left center",(!t.
style.transition||t.style.transition.indexOf("transform")===-1)&&(t.style.transition="transform 0.45\
s cubic-bezier(0.22, 1, 0.36, 1)"),t.style.transform=`scaleX(${v/100})`,t.style.willChange="transfor\
m"),n&&(n.innerText=`${Math.round(v)}%`)},"updateProgress");e&&(e.classList.remove("hidden"),e.style.
setProperty("display","block","important")),t&&(t.style.transition="none",t.style.width="100%",t.style.
transformOrigin="left center",t.style.transform="scaleX(0)",t.offsetHeight,t.style.transition="trans\
form 0.45s cubic-bezier(0.22, 1, 0.36, 1)"),i(0),await new Promise(w=>requestAnimationFrame(()=>setTimeout(
w,150)));const a=getRichPasteEditor();if(!a)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093");
const r=inferRichPasteTitle(),l=sanitizeRichPasteHtml(a.innerHTML||""),c=detectRichPasteTheme(l),u=normalizeRichPastePrintHtml(
l);await ensurePdfLibraries();const f=window.jspdf&&window.jspdf.jsPDF?window.jspdf.jsPDF:null;if(!f)
throw new Error("jsPDF \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
const g=window.html2canvas;if(typeof g!="function")throw new Error("html2canvas \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
i(5);const y=document.createElement("div");y.style.position="absolute",y.style.left="-10000px",y.style.
top="0",y.style.width="794px",y.style.background=c.background,y.style.color=c.foreground,y.style.boxSizing=
"border-box",y.style.fontFamily='"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif',y.
innerHTML=`
                <style>
                        :root {
                            color-scheme: ${c.mode};
                            --rp-background: ${c.background};
                            --rp-foreground: ${c.foreground};
                            --rp-muted: ${c.muted};
                            --rp-border: ${c.border};
                            --rp-surface: ${c.surface};
                            --rp-quote: ${c.quote};
                            --rp-link: ${c.link};
                        }
	                    .pdf-root-wrapper {
	                        background-color: var(--rp-background);
	                        color: var(--rp-foreground);
	                        padding: 40px;
	                        width: 794px;
	                        min-height: 1123px;
	                        box-sizing: border-box;
	                        color-scheme: ${c.mode};
	                        line-height: 1.6;
	                        font-size: 15px;
	                    }
	                    .pdf-root-wrapper * {
	                        box-sizing: border-box;
	                    }
                    .pdf-root-wrapper h1,
                    .pdf-root-wrapper h2,
                    .pdf-root-wrapper h3,
                    .pdf-root-wrapper h4,
                    .pdf-root-wrapper h5,
                    .pdf-root-wrapper h6 {
	                        line-height: 1.25;
	                        margin: 1.1em 0 0.55em 0;
	                    }
	                    .pdf-title {
	                        font-size: 26px;
	                        font-weight: bold;
	                        margin: 0 0 15px 0;
	                        border-bottom: 2px solid var(--rp-border);
	                        padding-bottom: 10px;
	                        line-height: 1.2;
                            color: var(--rp-foreground);
	                    }
	                    .pdf-meta {
	                        font-size: 12px;
	                        color: var(--rp-muted);
	                        margin-bottom: 30px;
	                    }
	                    .pdf-content {
	                        font-size: 15px;
	                        line-height: 1.6;
	                        color: inherit;
	                        overflow-wrap: anywhere;
	                    }
	                    .pdf-content p { margin: 0 0 1em 0; }
	                    .pdf-content img { max-width: 100%; height: auto; }
	                    .pdf-content video,
	                    .pdf-content iframe {
	                        max-width: 100%;
	                    }
	                    .pdf-content table { max-width: 100%; border-collapse: collapse; margin: 20px 0\
; border: 1px solid var(--rp-border); }
	                    .pdf-content th, .pdf-content td { border: 1px solid var(--rp-border); padding:\
 10px; text-align: left; word-break: break-word; vertical-align: top; }
	                    .pdf-content th { background-color: var(--rp-surface); color: var(--rp-foregrou\
nd); font-weight: bold; }
	                    .pdf-content pre {
	                        background-color: var(--rp-surface);
	                        color: var(--rp-foreground);
	                        border: 1px solid var(--rp-border);
	                        padding: 15px;
	                        border-radius: 5px;
	                        white-space: pre-wrap;
	                        word-break: break-word;
	                        font-family: "Noto Sans Mono", monospace;
	                        font-size: 13px;
	                        margin: 1.2em 0;
	                        line-height: 1.4;
	                        display: block;
	                        width: 100%;
	                        overflow-wrap: anywhere;
	                    }
	                    .pdf-content code {
	                        font-family: "Noto Sans Mono", monospace;
	                        background-color: var(--rp-surface);
	                        color: var(--rp-foreground);
	                        padding: 1px 4px;
	                        border-radius: 3px;
	                        font-size: 0.9em;
	                    }
	                    .pdf-content pre code {
	                        display: block;
	                        padding: 0;
	                        margin: 0;
	                        border-radius: 0;
	                        background: transparent;
	                        color: inherit;
	                        font-size: inherit;
	                        line-height: inherit;
	                        white-space: pre-wrap;
	                    }
	                    .pdf-content pre code * {
	                        background: transparent;
	                        color: inherit;
	                    }
	                    .pdf-content blockquote {
	                        border-left: 5px solid #f59e0b;
                            background: var(--rp-quote);
                            color: var(--rp-foreground);
	                        padding: 5px 0 5px 20px;
	                        margin: 1em 0;
	                        font-style: italic;
	                    }
	                    .pdf-content a {
	                        color: var(--rp-link);
	                        text-decoration: underline;
	                    }
	                    .pdf-content ul,
	                    .pdf-content ol {
	                        margin: 0 0 1em 0;
	                        padding-left: 1.5em;
	                    }
	                    .pdf-content li { margin-bottom: 0.4em; }
                </style>
                <div class="pdf-root-wrapper">
                    <div class="pdf-title">${escapeHtml(r)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString("ja-JP")}</div>
                    <div class="pdf-content">${u}</div>
                </div>
            `,document.body.appendChild(y),await waitForRichPasteMedia(y,4e3),i(15);try{const w=new f(
{unit:"mm",format:"a4",orientation:"portrait",compress:!0}),v=w.internal.pageSize.getWidth(),k=w.internal.
pageSize.getHeight(),_=794,C=Math.floor(k/v*_),L=y.scrollHeight||y.offsetHeight;let E=0,B=!0;const K=Math.
ceil(L/C);let Z=0;for(;E<L;){if(richPasteAbortController&&richPasteAbortController.signal.aborted)throw new DOMException(
"Aborted","AbortError");const O=Math.min(C,L-E),te=(await new Promise((ye,Me)=>{const oe=setTimeout(
()=>Me(new Error("PDF chunk rendering timed out")),12e4);g(y,{scale:1,useCORS:!0,allowTaint:!1,backgroundColor:c.
background,logging:!1,imageTimeout:5e3,x:0,y:E,width:_,height:O,windowWidth:_,scrollX:0,scrollY:0,signal:richPasteAbortController?
richPasteAbortController.signal:void 0,onclone:o(re=>{prepareRichPastePdfClone(re,c);const J=re.querySelector(
".pdf-root-wrapper");J&&(J.style.position="relative",J.style.left="0",J.style.top="0")},"onclone")}).
then(re=>{clearTimeout(oe),ye(re)}).catch(re=>{clearTimeout(oe),Me(re)})})).toDataURL("image/jpeg",.95),
be=w.getImageProperties(te),ue=Math.min(k,be.height*v/be.width);B||w.addPage(),w.addImage(te,"JPEG",
0,0,v,ue),B=!1,E+=O,Z++;const Le=Math.min(100,15+Math.round(Z/K*85));i(Le),await new Promise(ye=>setTimeout(
ye,100))}return i(100),{blob:w.output("blob"),fileName:buildRichPastePdfFilename()}}finally{e&&(e.classList.
add("hidden"),e.style.display="none"),y&&y.parentNode&&document.body.removeChild(y)}},"renderRichPas\
tePdfBlob"),createRichPastePdfBlob=o(async()=>await renderRichPastePdfBlob(),"createRichPastePdfBlob"),
buildRichPasteServerPayload=o(()=>{const e=getRichPasteEditor();if(!e)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\
\u3093");const t=String(e.innerHTML||"").trim(),n=String(e.textContent||"").trim(),i=t||(n?`<p>${escapeHtml(
n).replace(/\n/g,"<br/>")}</p>`:"");return{title:inferRichPasteTitle(),html:i,created_at:new Date().
toLocaleString("ja-JP"),theme:detectRichPasteTheme(sanitizeRichPasteHtml(i))}},"buildRichPasteServer\
Payload"),attachRichPastePdfAndSend=o(async(e,t,n,i)=>{const a=new Set(collectAttachmentItemsForSend().
map(g=>g.path)),r=new File([e],t,{type:"application/pdf",lastModified:Date.now()}),l=get("prompt-inp\
ut");if(l&&(l.value=n),await handleFiles([r],{openModal:!1}),!collectAttachmentItemsForSend().map(g=>g.
path).some(g=>!a.has(g)))throw l&&(l.value=i),new Error("PDF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const f=sendMessage();clearRichPasteEditor(!0),window.closeRichPasteModal(),showToast("PDF\u3092\u6DFB\u4ED8\u3057\u3066\u9001\u4FE1\u3092\u958B\u59CB\
\u3057\u307E\u3057\u305F","success"),f&&typeof f.catch=="function"&&f.catch(()=>{})},"attachRichPast\
ePdfAndSend"),openRichPasteModal=o(async()=>{await ensureUserSettingsSnapshot(),showModal("rich-past\
e-modal"),location.pathname!=="/paste"&&history.pushState({modal:"paste"},"","/paste");const e=getRichPastePrompt();
e&&(richPastePromptPreferenceSyncing=!0,e.value=getRichPasteEffectivePrompt(userSettingsSnapshot),richPastePromptPreferenceSyncing=
!1),updateRichPasteStatus(),setTimeout(()=>focusRichPasteEditor(),80)},"openRichPasteModal");window.
closeRichPasteModal=(e=!1)=>{hideModal("rich-paste-modal"),!e&&location.pathname==="/paste"&&history.
back()};const sendRichPasteToModel=o(async(e={})=>{const t=!!(e&&e.serverSide);if(abortController||richPasteAbortController){
showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u307E\u305F\u306FPDF\u5909\u63DB\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const n=getRichPasteEditor(),i=getRichPastePrompt(),a=get(t?"rich-paste-send-se\
rver-btn":"rich-paste-send-btn"),r=get("rich-paste-cancel-btn");if(!n||!n.innerText||!n.innerText.trim()){
showToast("\u8CBC\u308A\u4ED8\u3051\u308B\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}richPasteAbortController=new AbortController,r&&(r.onclick=()=>{richPasteAbortController&&
(richPasteAbortController.abort(),showToast("PDF\u5909\u63DB\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info"))});const l=i&&i.value&&i.value.trim()?i.value.trim():RICH_PASTE_DEFAULT_PROMPT,c=get("prompt\
-input")?get("prompt-input").value:"";a&&(a.disabled=!0);try{const u=get("toast-stack");if(u&&u.querySelectorAll(
".toast").forEach(f=>{(f.innerText.includes("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059")||
f.innerText.includes("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059"))&&
f.remove()}),t?(showToast("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...",
"info",!0),setRichPasteProgress(2,"\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...")):
showToast("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...","info",!0),t){if(!RICH_PASTE_PDF_SERVER_ROUTE)
throw new Error("\u30B5\u30FC\u30D0\u30FC\u5074PDF\u751F\u6210\u306EURL\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093");
const f=buildRichPasteServerPayload();setRichPasteProgress(10,"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u4FE1\u4E2D...");
const g=await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(f),signal:richPasteAbortController.signal});if(setRichPasteProgress(60,
"PDF\u3092\u53D7\u4FE1\u4E2D..."),!g.ok){let k="";try{const _=await g.json();k=_&&(_.message||_.error)?
String(_.message||_.error):""}catch{try{k=await g.text()}catch{k=""}}throw k==="missing_html"?new Error(
"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u308BHTML\u304C\u7A7A\u3067\u3059\u3002\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u5185\u5BB9\u306E\u53D6\u308A\u8FBC\u307F\u3092\u5148\u306B\u884C\u3063\u3066\u304F\u3060\u3055\u3044"):
new Error(k?`\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${k}`:
"\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}setRichPasteProgress(
75,"PDF\u3092\u6DFB\u4ED8\u4E2D...");const y=await g.blob(),w=g.headers.get("X-Rich-Paste-Filename")||
buildRichPastePdfFilename();!!(get("rich-paste-download-only")&&get("rich-paste-download-only").checked)?
(setRichPasteProgress(90,"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D..."),downloadBlob(y,w),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(y,w,l,c),setRichPasteProgress(100,"\u5B8C\u4E86"),
setTimeout(()=>hideRichPasteProgress(),400)}else{const f=await createRichPastePdfBlob();!!(get("rich\
-paste-download-only")&&get("rich-paste-download-only").checked)?(downloadBlob(f.blob,f.fileName),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(f.blob,f.fileName,l,c)}}catch(u){if(u.
name==="AbortError"){console.log("PDF generation aborted by user"),t&&(setRichPasteProgress(0,"\u30AD\u30E3\u30F3\u30BB\u30EB\
\u3055\u308C\u307E\u3057\u305F"),setTimeout(()=>hideRichPasteProgress(),800));return}get("prompt-inp\
ut")&&(get("prompt-input").value=c);const f=u&&u.message?u.message:"PDF\u5316\u3057\u3066\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
showToast(f,"error",!0),t&&(setRichPasteProgress(0,"\u5931\u6557\u3057\u307E\u3057\u305F"),setTimeout(
()=>hideRichPasteProgress(),1200))}finally{a&&(a.disabled=!1),richPasteAbortController=null}},"sendR\
ichPasteToModel");let csrfToken=document.querySelector('meta[name="csrf-token"]').content,csrfRefreshPromise=null;
const refreshCsrfToken=o(async()=>csrfRefreshPromise||(csrfRefreshPromise=(async()=>{const e=await fetch(
"/api/csrf_token",{method:"GET",credentials:"include",cache:"no-store",headers:{Accept:"application/\
json"}});if(!e.ok)return!1;const t=await e.json().catch(()=>({})),n=t&&typeof t.csrf_token=="string"?
t.csrf_token:"";if(!n)return!1;csrfToken=n;const i=document.querySelector('meta[name="csrf-token"]');
return i&&i.setAttribute("content",n),!0})().catch(()=>!1).finally(()=>{csrfRefreshPromise=null}),csrfRefreshPromise),
"refreshCsrfToken"),apiFetch=o(async(e,t={})=>{const n=(t.method||"GET").toUpperCase(),i=Object.assign(
{},t.headers||{}),a=!["GET","HEAD","OPTIONS"].includes(n);a&&(i["X-CSRF-Token"]=csrfToken);const r=t.
credentials||"include";let l=await fetch(e,Object.assign({},t,{headers:i,credentials:r}));if(a&&(l.status===
403||l.status===404)){let c=null;try{c=await l.clone().json()}catch{}const u=c&&c.error;if(u==="acco\
unt_locked")return!isAdminUser&&!document.getElementById("bot-lock-overlay")&&showBotLockOverlay(c.message||
"\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u3066\u3044\u307E\u3059\u3002",
c.remaining_seconds),l;if(u==="banned"||u==="turnstile_failed"||u==="rate_limit")return l;if(u==="tu\
rnstile_required"&&isBotDetectionActive())return botDetectionVerified=!1,await Promise.race([runBotDetectionGate(),
new Promise(y=>setTimeout(()=>y(!1),3e4))])&&(i["X-CSRF-Token"]=csrfToken,l=await fetch(e,Object.assign(
{},t,{headers:i,credentials:r}))),l;await refreshCsrfToken()&&(i["X-CSRF-Token"]=csrfToken,l=await fetch(
e,Object.assign({},t,{headers:i,credentials:r})))}return l},"apiFetch"),manualSpinnerRequestOptions=o(
e=>window.ProgressSpinner?window.ProgressSpinner.manualRequestOptions(e):e,"manualSpinnerRequestOpti\
ons");window.updateGoogleLinkUI=e=>{const t=get("google-link-text"),n=get("google-email-text"),i=get(
"google-action-area"),a=get("google-link-icon");!t||!i||(e.google_id?(t.innerText="\u9023\u643A\u6E08\u307F",
t.classList.replace("text-gray-200","text-green-400"),n.innerText=e.google_email||"\u9023\u643A\u4E2D\u306E Google \u30A2\u30AB\u30A6\u30F3\u30C8",
a.classList.replace("bg-gray-800","bg-green-900/30"),a.classList.add("text-green-400"),i.innerHTML='\
<button onclick="unlinkGoogleAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-\
400 border border-red-800 rounded text-xs font-bold transition btn-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):
(t.innerText="\u672A\u9023\u643A",t.classList.replace("text-green-400","text-gray-200"),n.innerText=
"Google \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
a.classList.replace("bg-green-900/30","bg-gray-800"),a.classList.remove("text-green-400"),i.innerHTML=
'<a href="/login/google" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white roun\
ded text-xs font-bold transition btn-hover">Google \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkGoogleAccount=
async()=>{if(confirm(`Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Google \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkGoogleAccount,{method:"POST"});if(e.ok)showToast("\
Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(t=>t.json()).then(t=>updateGoogleLinkUI(t));else{const t=await e.json();showToast(t.error||"\u89E3\u9664\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}},window.updateMinashinLinkUI=e=>{const t=get("minashin-link-text"),n=get("minashin-emai\
l-text"),i=get("minashin-action-area"),a=get("minashin-link-icon");!t||!i||(e.minashin_sub?(t.innerText=
"\u9023\u643A\u6E08\u307F",t.classList.replace("text-gray-200","text-green-400"),n.innerText=e.minashin_email||
"\u9023\u643A\u4E2D\u306E Minashin \u30A2\u30AB\u30A6\u30F3\u30C8",a.classList.replace("bg-gray-800",
"bg-green-900/30"),i.innerHTML='<button onclick="unlinkMinashinAccount()" class="px-4 py-2 bg-red-90\
0/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn\
-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):(t.innerText="\u672A\u9023\u643A",t.classList.replace(
"text-green-400","text-gray-200"),n.innerText="Minashin \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
a.classList.replace("bg-green-900/30","bg-gray-800"),i.innerHTML='<a href="/login/minashin" class="i\
nline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition \
btn-hover">Minashin \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkMinashinAccount=async()=>{if(confirm(
`Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Minashin \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkMinashinAccount,{method:"POST"});if(e.ok)showToast(
"Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(t=>t.json()).then(t=>updateMinashinLinkUI(t));else{const t=await e.json();showToast(t.error||"\u89E3\
\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}};let lastClientDebugEnabled=null;const isClientDebugLogEnabled=o(()=>{const e=get("set-\
client-debug-log");return!!(e&&e.checked)},"isClientDebugLogEnabled"),sendClientDebugLog=o((e,t)=>{if(!isClientDebugLogEnabled())
return;const n={level:String(e||"info"),message:String(t||"")};apiFetch("/api/debug/client_log",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(n)}).catch(()=>{})},"sendClien\
tDebugLog"),syncClientDebugLogToggle=o((e,t)=>{const n=get("set-client-debug-log");n&&(n.checked=!!e);
const i=!!e;i&&lastClientDebugEnabled!==!0&&sendClientDebugLog("info",`Client debug logging enabled \
(${t}).`),lastClientDebugEnabled=i},"syncClientDebugLogToggle"),nowPerfMs=o(()=>window.performance&&
typeof window.performance.now=="function"?window.performance.now():Date.now(),"nowPerfMs"),reportFirstTokenLatency=o(
e=>{if(enableLatencyMetrics)try{if(!e||typeof e!="object")return;const t=Number(e.latency_seconds);if(!Number.
isFinite(t)||t<0||t>600)return;const n=Number(e.latency_ms),i={latency_seconds:Number(t.toFixed(6)),
latency_ms:Number.isFinite(n)?Math.max(0,Math.round(n)):Math.round(t*1e3),thread_id:e.thread_id?String(
e.thread_id):null,job_id:e.job_id?String(e.job_id):null,model:e.model?String(e.model):null,first_event_type:e.
first_event_type?String(e.first_event_type):"content",client_sent_at_ms:Number.isFinite(Number(e.client_sent_at_ms))?
Math.round(Number(e.client_sent_at_ms)):Date.now(),is_total:!!e.is_total,client_done_at_ms:Number.isFinite(
Number(e.client_done_at_ms))?Math.round(Number(e.client_done_at_ms)):null};apiFetch("/api/metrics/fi\
rst_token",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(i)}).catch(
()=>{})}catch{}},"reportFirstTokenLatency");let currentThreadId=CHAT_CONFIG.initialThreadId;currentThreadId!=
null&&(currentThreadId=String(currentThreadId));const ATTACHMENT_MAX_FILES=Number(CHAT_CONFIG.attachmentMaxFiles)||
30,UPLOAD_CONCURRENCY=Math.max(1,Number(CHAT_CONFIG.uploadConcurrency)||3),TEMP_CHAT_TIMEOUT_MIN_SECONDS=10,
TEMP_CHAT_TIMEOUT_MAX_SECONDS=3600,TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS=90,TEMP_CHAT_HEARTBEAT_MIN_MS=4e3,
TEMP_CHAT_HEARTBEAT_MAX_MS=15e3;let activeGem=null,editingGemUuid=null,currentImageUrls=[],currentMaskImage=null,
abortController=null,richPasteAbortController=null,userAutoScroll=!0,searchTimeout,promptHistory=[],
historyIndex=-1,tempPrompt="";const markerAppliedUploads=new Set,attachmentSourceByPath=new Map,attachmentNameByPath=new Map,
BROWSER_FAST_IGNORE_WARNING_STORAGE="browser_fast_mode_ignore_warning",BROWSER_FAST_MAX_IMAGES=4,BROWSER_FAST_MAX_BYTES=12*
1024*1024,browserFastLocalFiles=new Map;let browserFastModeEnabled=!1,browserFastApiKey="",browserFastApiKeyModel="",
browserFastBootstrap=null,browserFastPreviousOptions=null,cameraCaptureStream=null,cameraCaptureFacingMode="\
environment",cameraCaptureBusy=!1,cameraCaptureSequence=0;const cameraCapturePendingFiles=[],cameraCapturePendingPreviewUrls=[];
let modalThreadId=null;const MARKER_HINT_TEXT="\u7DE8\u96C6\u6E08\u307F\u306E\u753B\u50CF\u3092\u898B\u3066\u304F\u3060\u3055\u3044\u3002",
MARKER_OPACITY_MIN_PCT=.1,MARKER_OPACITY_MAX_PCT=100,MARKER_OPACITY_MIN_ALPHA=MARKER_OPACITY_MIN_PCT/
100,markerState={row:null,filename:"",hasStroke:!1,naturalWidth:0,naturalHeight:0,colorHex:"#facc15",
opacity:.6,history:[],mode:"draw",cropRect:null,mosaicRects:[],mosaicPreviewRect:null,baseCanvas:null,
baseImageData:null},markerView={scale:1,offsetX:0,offsetY:0,minScale:1,maxScale:4},threadGemMap={};let pendingGemForNewThread=null,
loadedGems=[],currentJobId=null,currentThreadPending=null,currentVisionModel=null,activeStreamingBubbleId=null,
manualStopContext=null,manualStopSeq=0,isStopMode=!1;const suppressedPendingJobIds=new Set,pendingStreamReconnectJobs=new Set;
let editingMessageId=null;const messageStore={},lib={modal:get("lib-modal"),grid:get("lib-grid"),files:[],
selected:new Set,attachMode:!1,searchQuery:"",favoritesOnly:!1,totalCount:0,hasMore:!1,loading:!1,nextOffset:0},
LIBRARY_PAGE_SIZE=40,LIB_SORT_KEY="lib_sort_order",LIB_FAVORITES_ONLY_KEY="lib_favorites_only";let threadPage=1,
threadLoading=!1,hasMoreThreads=!0,threadObserver=null,currentQuote="",currentThreadTitle=null,temporaryChatEnabled=!1,
temporaryChatTimeoutSeconds=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS,tempChatExpiresAtMs=null,tempChatHeartbeatTimer=null,
tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=!1,tempChatHeaderTicker=null,enterToSend=CHAT_CONFIG.
enterToSend,autoSearchOnLinks=CHAT_CONFIG.autoSearchOnLinks,useSwCache=CHAT_CONFIG.useSwCache,compactPromptMode=CHAT_CONFIG.
compactPromptMode,minimalPromptMode=!!CHAT_CONFIG.minimalPromptMode,voiceStudioUiEnabled=!0;const CANVAS_MODE_STORAGE_KEY="\
canvas_mode_enabled_v1",CODING_MODE_STORAGE_KEY="coding_mode_enabled_v1";let canvasModeEnabled=!1,codingModeEnabled=!1,
codingModeEffective=!1,codingTargetSelection=null;const canvasPreviewState={blocks:[],rawText:"",renderText:"",
selectedIndex:-1,selectedKey:"",selectionMode:"auto",mobileView:"preview",sourceScrollTop:0,sourceScrollLeft:0,
frameScrollX:0,frameScrollY:0,frameRenderToken:0,panelAnimationToken:0,panelHideTimer:null,viewAnimationToken:0,
viewAnimationTimer:null,lastCanvasData:null};try{canvasModeEnabled=localStorage.getItem(CANVAS_MODE_STORAGE_KEY)===
"true"}catch{canvasModeEnabled=!1}try{codingModeEnabled=localStorage.getItem(CODING_MODE_STORAGE_KEY)===
"true"}catch{codingModeEnabled=!1}let enableLatencyMetrics=CHAT_CONFIG.enableLatencyMetrics,promptControlsExpanded=!1;
const appVersion=CHAT_CONFIG.appVersion,botConfig=CHAT_CONFIG.botConfig,isAdminUser=botConfig&&botConfig.
isAdmin,currentUsername=CHAT_CONFIG.currentUsername;let turnstileWidgetId=null,turnstileToken=null,turnstilePending=!1,
botDetectionVerified=!1,botDetectionGatePromise=null,botDetectionOverlayShown=!1,botDetectionDialogWidgetId=null,
sendButtonSpamTimestamps=[],chatDefaultsLoaded=!1,modelApiKeyMap={};const THREAD_INITIAL_MESSAGE_LIMIT=50,
THREAD_OLDER_PAGE_SIZE=50,LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT=40,LOW_BANDWIDTH_OLDER_PAGE_SIZE=60,LOW_BANDWIDTH_MODE_STORAGE_KEY="\
low_bandwidth_mode_pref_v1",LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD=.02,MATHJAX_SRC="https://c\
dn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js",HLJS_JS_SRC="https://cdnjs.cloudflare.com/ajax/libs/\
highlight.js/11.9.0/highlight.min.js",HLJS_CSS_SRC="https://cdnjs.cloudflare.com/ajax/libs/highlight\
.js/11.9.0/styles/atom-one-dark.min.css";let mathJaxLoadPromise=null,incrementalMathTypesetChain=Promise.
resolve(),highlightLoadPromise=null,lowBandwidthModePreference="auto",lowBandwidthModeAuto=!1,lowBandwidthMode=!1,
lowBandwidthModeReason="",lowBandwidthConnectionListenerAttached=!1,deferredDecorationObserver=null;
const deferredDecorationTextMap=new WeakMap;let threadHasOlderMessages=!1,oldestLoadedMessageId=null,
loadingOlderMessages=!1,threadLoadSequence=0,allMessages=[],currentLeafId=null,currentParentId=null;
function loadScriptOnce(e,t){const n=t?document.getElementById(t):null;return n?n.dataset.loaded==="\
1"?Promise.resolve(n):new Promise((i,a)=>{n.addEventListener("load",()=>i(n),{once:!0}),n.addEventListener(
"error",a,{once:!0})}):new Promise((i,a)=>{const r=document.createElement("script");t&&(r.id=t),r.src=
e,r.async=!0,r.onload=()=>{r.dataset.loaded="1",i(r)},r.onerror=a,document.head.appendChild(r)})}o(loadScriptOnce,
"loadScriptOnce");function loadStylesheetOnce(e,t){const n=t?document.getElementById(t):null;if(n)return Promise.
resolve(n);const i=Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find(a=>a.href===
e);return i?Promise.resolve(i):new Promise((a,r)=>{const l=document.createElement("link");t&&(l.id=t),
l.rel="stylesheet",l.href=e,l.onload=()=>a(l),l.onerror=r,document.head.appendChild(l)})}o(loadStylesheetOnce,
"loadStylesheetOnce");async function ensureMathJaxLoaded(){return window.MathJax&&typeof window.MathJax.
typesetPromise=="function"?window.MathJax:(mathJaxLoadPromise||(window.MathJax=window.MathJax||{tex:{
inlineMath:[["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{
ignoreHtmlClass:"tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},
startup:{typeset:!1}},mathJaxLoadPromise=loadScriptOnce(MATHJAX_SRC,"MathJax-script").catch(e=>{throw mathJaxLoadPromise=
null,e})),await mathJaxLoadPromise,window.MathJax||null)}o(ensureMathJaxLoaded,"ensureMathJaxLoaded");
async function ensureHighlightLoaded(){return window.hljs?window.hljs:(highlightLoadPromise||(highlightLoadPromise=
Promise.all([loadStylesheetOnce(HLJS_CSS_SRC,"hljs-theme-chat"),loadScriptOnce(HLJS_JS_SRC,"hljs-scr\
ipt")]).then(()=>window.hljs||null).catch(e=>{throw highlightLoadPromise=null,e})),await highlightLoadPromise)}
o(ensureHighlightLoaded,"ensureHighlightLoaded");function maybeNeedsMathJax(e){const t=String(e||"");
return t.includes("$$")||t.includes("\\(")||t.includes("\\[")||t.includes("\\begin{")?!0:/(?<!\$)\$(?!\$)(?=[\s\S]*?[A-Za-z\\^_{}])(?:[^$\n\\]|\\.)+?\$(?!\$)/.
test(t)}o(maybeNeedsMathJax,"maybeNeedsMathJax");function protectMathSegments(e){const t=String(e||""),
n=[],i=o(f=>{const g=`@@MATHJAX_BLOCK_${n.length}@@`;return n.push(f),g},"stash"),a=[],r=/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
let l=0,c;for(;(c=r.exec(t))!==null;){const f=c.index;f>l&&a.push({type:"text",value:t.slice(l,f)}),
a.push({type:"code",value:c[0]}),l=f+c[0].length}return l<t.length&&a.push({type:"text",value:t.slice(
l)}),a.length||a.push({type:"text",value:t}),{text:a.map(f=>{if(f.type==="code")return f.value;let g=f.
value;return g=g.replace(/\$\$([\s\S]+?)\$\$/g,i),g=g.replace(/\\\(([\s\S]+?)\\\)/g,i),g=g.replace(/\\\[([\s\S]+?)\\\]/g,
i),g=g.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g,i),g=g.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g,
i),g}).join(""),blocks:n}}o(protectMathSegments,"protectMathSegments");function getStreamMathSegmentKey(e,t){
const n=String(t||"");let i=2166136261;for(let a=0;a<n.length;a++)i^=n.charCodeAt(a),i=Math.imul(i,16777619);
return`${e}-${n.length}-${(i>>>0).toString(16)}`}o(getStreamMathSegmentKey,"getStreamMathSegmentKey");
function restoreMathSegments(e,t,n={}){return!t||!t.length?String(e||""):String(e||"").replace(/@@MATHJAX_BLOCK_(\d+)@@/g,
(i,a)=>{const r=t[Number(a)];if(r==null)return"";const l=String(r).replace(/&/g,"&amp;").replace(/</g,
"&lt;").replace(/>/g,"&gt;");return n.streamMathSegments?`<span class="stream-math-segment mathjax_p\
rocess" data-stream-math-key="${getStreamMathSegmentKey(Number(a),r)}">${l}</span>`:l})}o(restoreMathSegments,
"restoreMathSegments");function maybeNeedsHighlight(e,t=null){return String(e||"").includes("```")?!0:
!t||typeof t.querySelector!="function"?!1:!!t.querySelector("pre code")}o(maybeNeedsHighlight,"maybe\
NeedsHighlight");function queueMathTypeset(e,t="",n={}){lowBandwidthMode&&!n.force||!e||!maybeNeedsMathJax(
t)||ensureMathJaxLoaded().then(()=>{if(!(!window.MathJax||typeof window.MathJax.typesetPromise!="fun\
ction")){try{typeof window.MathJax.typesetClear=="function"&&window.MathJax.typesetClear([e])}catch{}
return window.MathJax.typesetPromise([e]).catch(()=>{})}}).catch(()=>{})}o(queueMathTypeset,"queueMa\
thTypeset");function queueIncrementalMathTypeset(e){const t=Array.from(e||[]).filter(n=>n&&n.isConnected&&
!n.getAttribute("data-stream-math-state"));!t.length||lowBandwidthMode||(t.forEach(n=>n.setAttribute(
"data-stream-math-state","queued")),incrementalMathTypesetChain=incrementalMathTypesetChain.catch(()=>{}).
then(async()=>{await ensureMathJaxLoaded();const n=t.filter(i=>i.isConnected&&i.getAttribute("data-s\
tream-math-state")==="queued");if(!(!n.length||!window.MathJax||typeof window.MathJax.typesetPromise!=
"function")){n.forEach(i=>i.setAttribute("data-stream-math-state","rendering"));try{await window.MathJax.
typesetPromise(n),n.forEach(i=>{i.isConnected&&i.setAttribute("data-stream-math-state","rendered")})}catch{
n.forEach(a=>a.removeAttribute("data-stream-math-state"))}}}).catch(()=>{t.forEach(n=>n.removeAttribute(
"data-stream-math-state"))}))}o(queueIncrementalMathTypeset,"queueIncrementalMathTypeset");function queueHighlight(e,t="",n={}){
lowBandwidthMode&&!n.force||!e||!maybeNeedsHighlight(t,e)||activeStreamingBubbleId&&e.closest(`#${activeStreamingBubbleId}`)||
ensureHighlightLoaded().then(()=>{window.hljs&&e.querySelectorAll("pre code").forEach(i=>{if(!(i.getAttribute(
"data-highlighted")==="true"&&!n.force))try{window.hljs.highlightElement(i)}catch{}})}).catch(()=>{})}
o(queueHighlight,"queueHighlight");function getNetworkConnectionInfo(){return navigator.connection||
navigator.mozConnection||navigator.webkitConnection||null}o(getNetworkConnectionInfo,"getNetworkConn\
ectionInfo");function detectLowBandwidthModeAuto(){const e=getNetworkConnectionInfo();if(!e)return{enabled:!1,
reason:""};const t=!!e.saveData,n=String(e.effectiveType||"").toLowerCase(),i=Number(e.downlink||0),
a=n==="slow-2g"||n==="2g"||n==="3g",r=Number.isFinite(i)&&i>0&&i<1.3,l=t||a||r,c=[];return t&&c.push(
"\u30C7\u30FC\u30BF\u7BC0\u7D04"),n&&c.push(`\u56DE\u7DDA:${n}`),r&&c.push(`\u4E0B\u308A:${i}Mbps`),
{enabled:l,reason:c.join(" / ")}}o(detectLowBandwidthModeAuto,"detectLowBandwidthModeAuto");function normalizeLowBandwidthModePreference(e){
const t=String(e||"").trim().toLowerCase();return t==="on"||t==="off"||t==="auto"?t:"auto"}o(normalizeLowBandwidthModePreference,
"normalizeLowBandwidthModePreference");function readLowBandwidthModePreference(){try{return normalizeLowBandwidthModePreference(
localStorage.getItem(LOW_BANDWIDTH_MODE_STORAGE_KEY)||"auto")}catch{return"auto"}}o(readLowBandwidthModePreference,
"readLowBandwidthModePreference");function persistLowBandwidthModePreference(e){const t=normalizeLowBandwidthModePreference(
e);lowBandwidthModePreference=t;try{t==="auto"?localStorage.removeItem(LOW_BANDWIDTH_MODE_STORAGE_KEY):
localStorage.setItem(LOW_BANDWIDTH_MODE_STORAGE_KEY,t)}catch{}}o(persistLowBandwidthModePreference,"\
persistLowBandwidthModePreference");function getEffectiveThreadInitialMessageLimit(){return lowBandwidthMode?
LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT:THREAD_INITIAL_MESSAGE_LIMIT}o(getEffectiveThreadInitialMessageLimit,
"getEffectiveThreadInitialMessageLimit");function getEffectiveThreadOlderPageSize(){return lowBandwidthMode?
LOW_BANDWIDTH_OLDER_PAGE_SIZE:THREAD_OLDER_PAGE_SIZE}o(getEffectiveThreadOlderPageSize,"getEffective\
ThreadOlderPageSize");function mergeBtnClasses(e,t=[],n=[]){e&&(n.forEach(i=>e.classList.remove(i)),
t.forEach(i=>e.classList.add(i)))}o(mergeBtnClasses,"mergeBtnClasses");function updateLowBandwidthModeUi(){
const e=get("low-bandwidth-toggle-btn"),t=get("low-bandwidth-status-pill"),n=lowBandwidthModePreference===
"auto"?"\u81EA\u52D5":lowBandwidthModePreference==="on"?"\u56FA\u5B9AON":"\u56FA\u5B9AOFF",i=lowBandwidthMode?
"ON":"OFF",a=lowBandwidthModeReason?` (${lowBandwidthModeReason})`:"";if(e&&(e.setAttribute("title",
`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9 ${i} / ${n}${a}`),e.setAttribute("aria-pressed",lowBandwidthMode?
"true":"false"),lowBandwidthMode?mergeBtnClasses(e,["text-amber-200","bg-amber-900/30","border","bor\
der-amber-600/40"],["text-gray-400"]):mergeBtnClasses(e,["text-gray-400"],["text-amber-200","bg-ambe\
r-900/30","border","border-amber-600/40"])),t)if(lowBandwidthMode){t.classList.remove("hidden");const r=lowBandwidthModePreference===
"auto"?" (\u81EA\u52D5)":" (\u624B\u52D5)";t.innerHTML=`<i class="fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9${r}${lowBandwidthModeReason?
`: ${escapeHtml(lowBandwidthModeReason)}`:""}`}else t.classList.add("hidden"),t.innerHTML='<i class=\
"fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9'}o(updateLowBandwidthModeUi,"updat\
eLowBandwidthModeUi");function refreshDecorationsForVisibleChat(){const e=get("chat-container");e&&(queueHighlight(
e,e.textContent||"",{force:!0}),queueMathTypeset(e,e.textContent||"",{force:!0}))}o(refreshDecorationsForVisibleChat,
"refreshDecorationsForVisibleChat");function applyLowBandwidthModeState(e,t={}){const n=lowBandwidthMode;
if(lowBandwidthMode=!!e,updateLowBandwidthModeUi(),n&&!lowBandwidthMode&&refreshDecorationsForVisibleChat(),
t.notify){const i=lowBandwidthModePreference==="auto"?"\u81EA\u52D5":"\u624B\u52D5",a=lowBandwidthModeReason?
` (${lowBandwidthModeReason})`:"";showToast(`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9\u3092${lowBandwidthMode?
"ON":"OFF"}\u306B\u3057\u307E\u3057\u305F [${i}]${a}`,"info",!1)}}o(applyLowBandwidthModeState,"appl\
yLowBandwidthModeState");function recomputeLowBandwidthMode(e={}){const t=detectLowBandwidthModeAuto();
lowBandwidthModeAuto=!!t.enabled,lowBandwidthModeReason=t.reason||"",applyLowBandwidthModeState(lowBandwidthModePreference===
"on"?!0:lowBandwidthModePreference==="off"?!1:lowBandwidthModeAuto,e)}o(recomputeLowBandwidthMode,"r\
ecomputeLowBandwidthMode");function cycleLowBandwidthModePreference(){const e=normalizeLowBandwidthModePreference(
lowBandwidthModePreference);persistLowBandwidthModePreference(e==="auto"?"on":e==="on"?"off":"auto"),
recomputeLowBandwidthMode({notify:!0})}o(cycleLowBandwidthModePreference,"cycleLowBandwidthModePrefe\
rence");function ensureDeferredDecorationObserver(){if(deferredDecorationObserver||typeof IntersectionObserver==
"undefined")return deferredDecorationObserver;const e=get("chat-container")||null;return deferredDecorationObserver=
new IntersectionObserver(t=>{t.forEach(n=>{!n.isIntersecting||!n.target||runDeferredDecorations(n.target)})},
{root:e,threshold:LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD}),deferredDecorationObserver}o(ensureDeferredDecorationObserver,
"ensureDeferredDecorationObserver");function runDeferredDecorations(e){if(!e)return;if(deferredDecorationObserver)
try{deferredDecorationObserver.unobserve(e)}catch{}const t=deferredDecorationTextMap.get(e)||"";queueHighlight(
e,t,{force:!0}),queueMathTypeset(e,t,{force:!0})}o(runDeferredDecorations,"runDeferredDecorations");
function queueMessageDecorations(e,t=""){if(!e)return;if(!lowBandwidthMode){queueHighlight(e,t),queueMathTypeset(
e,t);return}if(!maybeNeedsHighlight(t,e)&&!maybeNeedsMathJax(t))return;deferredDecorationTextMap.set(
e,String(t||""));const n=get("chat-container");if(n&&e===n){window.setTimeout(()=>runDeferredDecorations(
e),250);return}if(!e.isConnected)return;const i=ensureDeferredDecorationObserver();if(i){i.observe(e);
return}window.setTimeout(()=>runDeferredDecorations(e),250)}o(queueMessageDecorations,"queueMessageD\
ecorations");function initLowBandwidthMode(){lowBandwidthModePreference=readLowBandwidthModePreference(),
recomputeLowBandwidthMode({notify:!1});const e=get("low-bandwidth-toggle-btn");e&&!e.__lowBandwidthBound&&
(e.__lowBandwidthBound=!0,e.addEventListener("click",n=>{n&&n.preventDefault(),cycleLowBandwidthModePreference()}));
const t=getNetworkConnectionInfo();t&&typeof t.addEventListener=="function"&&!lowBandwidthConnectionListenerAttached&&
(lowBandwidthConnectionListenerAttached=!0,t.addEventListener("change",()=>{if(lowBandwidthModePreference===
"auto")recomputeLowBandwidthMode({notify:!0});else{const n=detectLowBandwidthModeAuto();lowBandwidthModeAuto=
!!n.enabled,lowBandwidthModeReason=n.reason||"",updateLowBandwidthModeUi()}}))}o(initLowBandwidthMode,
"initLowBandwidthMode");function escapeHtml(e){return e==null?"":String(e).replace(/&/g,"&amp;").replace(
/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;")}o(escapeHtml,"escape\
Html");const BLOCKED_SCRIPT_HOSTS=["polyfill.io","cdn.polyfill.io"];function isBlockedScriptSrc(e){if(!e)
return!1;const t=String(e).trim();if(!t)return!1;let n=t;t.startsWith("//")?n="https:"+t:!/^https?:\/\//i.
test(t)&&!t.startsWith("data:")&&!t.startsWith("blob:")&&(n="https://"+t);try{const a=(new URL(n,"ht\
tps://example.com").hostname||"").toLowerCase();return BLOCKED_SCRIPT_HOSTS.some(r=>a===r||a.endsWith(
"."+r))}catch{return/polyfill\.io/i.test(t)}}o(isBlockedScriptSrc,"isBlockedScriptSrc");function isPasswordPromptingScript(e){
if(!e)return!1;const t=String(e),n=t.toLowerCase();return!!(/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.
test(t)||/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(t)||
/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.
test(n)||/prompt\s*\(/.test(t)&&/(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.
test(t))}o(isPasswordPromptingScript,"isPasswordPromptingScript");function detectBlockedScriptsInCode(e){
if(!e)return!1;const t=String(e),n=/<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;let i;for(;(i=n.exec(
t))!==null;)if(isBlockedScriptSrc(i[1]))return!0;const a=/<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
for(;(i=a.exec(t))!==null;)if(isPasswordPromptingScript(i[1]))return!0;return!!(/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.
test(t)||/src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(t))}o(detectBlockedScriptsInCode,"detectBlockedS\
criptsInCode");function sanitizeHtmlForPreview(e){if(!e)return"";const t=detectBlockedScriptsInCode(
e);let n=String(e);try{const a=new DOMParser().parseFromString(n,"text/html");let r=!1;a.querySelectorAll(
"script").forEach(c=>{const u=c.getAttribute("src")||"";let f=!1;if(u&&isBlockedScriptSrc(u)){const g=a.
createElement("div");g.setAttribute("data-blocked-script","true"),g.style.cssText="background:#fee2e\
2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6p\
x 0;font-family:system-ui;";const y=u.length>70?u.slice(0,67)+"...":u;g.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: "+
y+" \uFF08polyfill.io \u306A\u3069\u306E\u5371\u967A\u30C9\u30E1\u30A4\u30F3\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u7121\u52B9\u5316\u3055\u308C\u307E\u3059\uFF09",
c.parentNode&&c.parentNode.replaceChild(g,c),r=!0,f=!0}else if(!u){const g=c.textContent||"";if(isPasswordPromptingScript(
g)){const y=a.createElement("div");y.setAttribute("data-blocked-script","true"),y.style.cssText="bac\
kground:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:\
12px;margin:6px 0;font-family:system-ui;",y.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: \u30D1\u30B9\u30EF\u30FC\u30C9\u5165\u529B\u8981\u6C42\u306A\u3069\u306E\u7591\u308F\u3057\u3044\u30A4\u30F3\u30E9\u30A4\u30F3\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\
\u305F",c.parentNode&&c.parentNode.replaceChild(y,c),r=!0,f=!0}}}),a.querySelectorAll('a[href^="java\
script:" i], area[href^="javascript:" i]').forEach(c=>{c.setAttribute("href","#"),c.setAttribute("ti\
tle",(c.getAttribute("title")||"")+" [javascript: disabled in preview]")});const l=a.head||a.querySelector(
"head");if(l&&!l.querySelector("base")){const c=a.createElement("base");c.setAttribute("href",`${window.
location.origin}/`),l.insertBefore(c,l.firstChild)}if(t||r){const c=a.body||a.documentElement;if(c){
const u=a.createElement("div");u.style.cssText="position:sticky;top:0;left:0;right:0;z-index:2147483\
647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:syst\
em-ui;border-bottom:1px solid #b91c1c;",u.innerHTML="\u26A0 <strong>\u5B89\u5168\u30D7\u30EC\u30D3\u30E5\u30FC</strong>: polyfill.io \u306A\u3069\u306E\u5371\u967A\u306A\u30B9\
\u30AF\u30EA\u30D7\u30C8\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002\u5B9F\u884C\u306F\u81EA\u5DF1\u8CAC\u4EFB\u3067\u3002",
c.firstChild?c.insertBefore(u,c.firstChild):c.appendChild(u)}}n=`<!DOCTYPE html>
`+(a.documentElement?a.documentElement.outerHTML:n)}catch{n=n.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
"<!-- blocked polyfill.io script for safety -->")}return n}o(sanitizeHtmlForPreview,"sanitizeHtmlFor\
Preview");function wrapTextWave(e){return e?e.split("").map((t,n)=>`<span class="wave-char" style="a\
nimation-delay: ${n*.028}s">${escapeHtml(t)}</span>`).join(""):""}o(wrapTextWave,"wrapTextWave");function getPendingSkeletonKind(e){
let t=String(e||"").toLowerCase();if(!t)try{t=String(get("model-select")&&get("model-select").value||
"").toLowerCase()}catch{t=""}return t.includes("video")?"video":t.includes("tts")||t.includes("trans\
cribe")||t.includes("realtime")||t.includes("voice")||t.includes("native-audio")||t.includes("live")&&
t.includes("gemini")?"audio":t.includes("gpt-image")||t.includes("imagine-image")||t.includes("image")&&
!t.includes("vision")||t.includes("gemini")&&(t.includes("image")||t.includes("nano"))?"image":t.includes(
"ocr")||t.includes("mistral-ocr")?"text":t.includes("build")||t.includes("code-fast")||t.includes("c\
oding")?"code":"text"}o(getPendingSkeletonKind,"getPendingSkeletonKind");function buildPendingSkeletonBody(e){
return e==="image"?'<div class="skeleton-media skeleton-image" aria-hidden="true"><div class="skelet\
on-media-icon"><i class="fas fa-image"></i></div></div>':e==="video"?'<div class="skeleton-media ske\
leton-video" aria-hidden="true"><div class="skeleton-media-icon"><i class="fas fa-play"></i></div><d\
iv class="skeleton-video-progress"></div></div>':e==="audio"?'<div class="skeleton-audio" aria-hidde\
n="true"><div class="skeleton-audio-disc"><i class="fas fa-volume-up"></i></div><div class="skeleton\
-wave"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><s\
pan></span></div></div>':e==="code"?'<div class="skeleton-code" aria-hidden="true"><div class="skele\
ton-code-header"><span class="skeleton-code-dot"></span><span class="skeleton-code-dot"></span><span\
 class="skeleton-code-dot"></span><div class="skeleton-code-title"></div></div><div class="skeleton-\
lines skeleton-code-lines"><div class="skeleton-line" style="width:72%"></div><div class="skeleton-l\
ine" style="width:88%"></div><div class="skeleton-line" style="width:54%"></div><div class="skeleton\
-line" style="width:76%"></div><div class="skeleton-line" style="width:41%"></div></div></div>':'<di\
v class="skeleton-lines" aria-hidden="true"><div class="skeleton-line" style="width:92%"></div><div \
class="skeleton-line" style="width:78%"></div><div class="skeleton-line" style="width:86%"></div><di\
v class="skeleton-line" style="width:64%"></div><div class="skeleton-line" style="width:48%"></div><\
/div>'}o(buildPendingSkeletonBody,"buildPendingSkeletonBody");function buildPendingSkeletonHtml(e,t){
const n=getPendingSkeletonKind(e),i=t==null||t===""?"\u56DE\u7B54\u3092\u751F\u6210\u4E2D...":String(
t);return`<div class="content-area pending-shimmer skeleton-pending" data-skeleton-kind="${escapeHtml(
n)}">${buildPendingSkeletonBody(n)}<div class="skeleton-status">${escapeHtml(i)}</div></div>`}o(buildPendingSkeletonHtml,
"buildPendingSkeletonHtml");function updatePendingSkeletonStatus(e,t,n){if(!e)return!1;const i=e.querySelector(
".content-area.skeleton-pending");if(!i)return!1;let a=i.querySelector(".skeleton-status");a||(a=document.
createElement("div"),a.className="skeleton-status",i.appendChild(a));const r=t==null?"":String(t),l=n==
null||n===""?"":String(n);return l?a.innerHTML=`${escapeHtml(r)}<span class="skeleton-status-sub">${escapeHtml(
l)}</span>`:a.textContent=r,!0}o(updatePendingSkeletonStatus,"updatePendingSkeletonStatus");function buildChatLoadingSkeletonHtml(){
return`<div class="chat-load-skeleton" role="status" aria-live="polite" aria-label="\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D">${[
{role:"user",widths:["62%","44%"]},{role:"ai",widths:["88%","76%","92%","58%"]},{role:"user",widths:[
"48%"]},{role:"ai",widths:["82%","70%","54%"]}].map((n,i)=>{const a=n.role==="user",r=a?"justify-end":
"justify-start",l=a?"message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4\
 rounded-2xl rounded-tr-none shadow-md relative":"message-bubble chat-load-skeleton-bubble chat-load\
-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative",c=n.widths.map(
(u,f)=>`<div class="skeleton-line" style="width:${u};animation-delay:${(i*.08+f*.06).toFixed(2)}s"><\
/div>`).join("");return`<div class="flex ${r} mb-4 chat-load-skeleton-row" style="animation-delay:${(i*
.07).toFixed(2)}s" aria-hidden="true"><div class="${l}"><div class="content-area pending-shimmer ske\
leton-pending chat-load-skeleton-body" data-skeleton-kind="text"><div class="skeleton-lines">${c}</d\
iv></div></div></div>`}).join("")}<div class="chat-load-skeleton-caption"><span class="chat-load-ske\
leton-caption-dot"></span>\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...</div></div>`}
o(buildChatLoadingSkeletonHtml,"buildChatLoadingSkeletonHtml");function showChatLoadError(e){const t=get(
"chat-container");if(!t)return;t.innerHTML='<div class="min-h-[45vh] flex items-center justify-cente\
r px-4"><div class="max-w-md w-full rounded-2xl border border-red-500/40 bg-red-950/30 p-5 text-cent\
er" role="alert"><i class="fas fa-triangle-exclamation text-red-300 text-xl mb-3"></i><p class="text\
-sm font-semibold text-red-100">\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F</p><p class="mt-2 text-xs text-red-200/80">\u901A\u4FE1\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\
\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p><button type="button" data-chat-load-retry class="mt-4 rounded-lg border border-red\
-300/40 px-4 py-2 text-sm text-red-100 hover:bg-red-500/20"><i class="fas fa-rotate-right mr-1"></i>\
\u518D\u8A66\u884C</button></div></div>';const n=t.querySelector("[data-chat-load-retry]");n&&n.addEventListener(
"click",()=>loadMessages(e))}o(showChatLoadError,"showChatLoadError");function hashString(e){let t=0;
if(!e)return"0";for(let n=0;n<e.length;n++)t=(t<<5)-t+e.charCodeAt(n),t|=0;return Math.abs(t).toString(
36)}o(hashString,"hashString");function decodeCodeButtonValue(e){if(!e)return"";try{return decodeURIComponent(
e)}catch{return""}}o(decodeCodeButtonValue,"decodeCodeButtonValue");function getCodingTargetFromButton(e){
if(!e)return null;const t=decodeCodeButtonValue(e.getAttribute("data-code")||"");if(!t)return null;const n=e.
closest(".code-wrapper"),i=e.closest(".message-group");return{code:t,language:String(e.getAttribute(
"data-coding-lang")||"text").trim().slice(0,40)||"text",key:String(e.getAttribute("data-code-key")||
(n==null?void 0:n.getAttribute("data-code-key"))||hashString(t)),message_id:i!=null&&i.id?i.id.replace(
/^msg-/,""):null,thread_id:currentThreadId?String(currentThreadId):null}}o(getCodingTargetFromButton,
"getCodingTargetFromButton");function findLatestCodingTarget(){const e=get("chat-container");if(!e)return null;
const t=Array.from(e.querySelectorAll(".message-group .coding-target-btn"));for(let n=t.length-1;n>=
0;n--){const i=getCodingTargetFromButton(t[n]);if(i)return i}return null}o(findLatestCodingTarget,"f\
indLatestCodingTarget");function extractPromptCodingTargets(e){const t=String(e||"").replace(/\r\n?/g,
`
`).split(`
`),n=[];let i=null;for(const a of t){if(!i){const c=a.match(/^\s*(`{3,}|~{3,})(.*)$/);if(!c)continue;
const u=String(c[2]||"").trim();i={markerChar:c[1][0],markerLength:c[1].length,language:(u.split(/\s+/)[0]||
"text").replace(/^\{?\.?/,"").replace(/\}$/,"")||"text",buffer:[]};continue}const r=String(a||"").trim();
if(new RegExp(`^\\${i.markerChar}{${i.markerLength},}\\s*$`).test(r)){const c=i.buffer.join(`
`);c.trim()&&n.push({code:c,language:i.language,key:hashString(`prompt\\n${i.language}\\n${c}`),candidate_id:`\
prompt-${n.length+1}`,prompt_index:n.length,message_id:null,thread_id:currentThreadId?String(currentThreadId):
null,prompt_source:!0}),i=null;continue}i.buffer.push(a)}return n}o(extractPromptCodingTargets,"extr\
actPromptCodingTargets");function extractLatestPromptCodingTarget(e){const t=extractPromptCodingTargets(
e);return t.length?t[t.length-1]:null}o(extractLatestPromptCodingTarget,"extractLatestPromptCodingTa\
rget");function collectCodingCandidates(e){if(codingTargetSelection){const r=codingTargetSelection.thread_id;
if(!r||!currentThreadId||String(r)===String(currentThreadId))return[{...codingTargetSelection,candidate_id:"\
selected-1",source:"history",explicit:!0}];codingTargetSelection=null}const t=extractPromptCodingTargets(
e),n=new Set(t.map(r=>`${r.language}
${r.code}`)),i=get("chat-container"),a=[];return i&&Array.from(i.querySelectorAll(".message-group .c\
oding-target-btn")).forEach(r=>{const l=getCodingTargetFromButton(r);if(!l)return;const c=`${l.language}\

${l.code}`;n.has(c)||(n.add(c),a.push(l))}),a.slice(-20).forEach((r,l)=>{t.push({...r,candidate_id:`\
history-${l+1}`,source:"history",explicit:!1})}),t}o(collectCodingCandidates,"collectCodingCandidate\
s");function resolveCodingTarget(e=null){var a;const t=String(e===null?((a=get("prompt-input"))==null?
void 0:a.value)||"":e||"");if(codingTargetSelection){const r=codingTargetSelection.thread_id;if(!r||
!currentThreadId||String(r)===String(currentThreadId))return{...codingTargetSelection,explicit:!0};codingTargetSelection=
null}const n=extractLatestPromptCodingTarget(t);if(n)return{...n,explicit:!1};const i=findLatestCodingTarget();
return i?{...i,explicit:!1}:null}o(resolveCodingTarget,"resolveCodingTarget");function syncCodingTargetButtons(e=document){
if(!e||typeof e.querySelectorAll!="function")return;const t=codingTargetSelection?String(codingTargetSelection.
key||""):"";e.querySelectorAll(".coding-target-btn").forEach(n=>{const i=!!t&&String(n.getAttribute(
"data-code-key")||"")===t;n.classList.toggle("coding-target-active",i),n.setAttribute("aria-pressed",
i?"true":"false"),n.innerHTML=i?'<i class="fas fa-thumbtack"></i>':'<i class="fas fa-quote-right"></\
i>',n.title=i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A",
n.setAttribute("aria-label",i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A")})}
o(syncCodingTargetButtons,"syncCodingTargetButtons");function syncCodingModeUi(e=codingModeEnabled,t={}){
var u;if(codingModeEnabled=!!e,t.persist!==!1)try{localStorage.setItem(CODING_MODE_STORAGE_KEY,codingModeEnabled?
"true":"false")}catch{}const n=get("enable-coding-mode");n&&n.checked!==codingModeEnabled&&(n.checked=
codingModeEnabled);const i=get("coding-target-bar"),a=get("coding-target-text"),r=get("clear-coding-\
target-btn");i&&i.classList.toggle("visible",codingModeEnabled);const l=resolveCodingTarget(),c=codingTargetSelection?
[l].filter(Boolean):collectCodingCandidates(String(((u=get("prompt-input"))==null?void 0:u.value)||""));
if(codingModeEffective=codingModeEnabled&&c.length>0,a)if(codingTargetSelection&&l)a.textContent=`\u7DE8\u96C6\
\u5BFE\u8C61: ${l.language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`;else if(c.length>1){
const f=c.filter(y=>y.prompt_source).length,g=c.length-f;a.textContent=`\u30E2\u30C7\u30EB\u304C\u7DE8\u96C6\u5BFE\u8C61\u3092\u5224\u65AD: \u5165\u529B${f}\
\u4EF6 / \u5C65\u6B74${g}\u4EF6`}else l&&l.prompt_source?a.textContent=`\u5165\u529B\u4E2D: ${l.language||
"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:l?a.textContent=`\u81EA\u52D5\u9078\u629E: \u6700\u65B0\u306E ${l.
language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:a.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u751F\u6210\u5F8C\u306B\u81EA\u52D5\u6709\u52B9\u5316";
r&&r.classList.toggle("hidden",!codingTargetSelection),syncCodingTargetButtons()}o(syncCodingModeUi,
"syncCodingModeUi");function activateDeferredCodingModeFromStream(e){if(!codingModeEnabled||codingModeEffective||
extractPromptCodingTargets(e).length===0)return!1;codingModeEffective=!0;const t=get("coding-target-\
text");return t&&(t.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u691C\u51FA: \u6B21\u306E\u9001\u4FE1\u304B\u3089\u6709\u52B9"),
!0}o(activateDeferredCodingModeFromStream,"activateDeferredCodingModeFromStream");function selectCodingTargetFromButton(e){
const t=getCodingTargetFromButton(e);if(!t){showToast("\u3053\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u7DE8\u96C6\u5BFE\u8C61\u306B\u3067\u304D\u307E\u305B\u3093",
"error",!0);return}codingTargetSelection=t,syncCodingModeUi(codingModeEnabled,{persist:!1}),codingModeEnabled?
showToast("Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u3057\u307E\u3057\u305F","suc\
cess"):showToast("\u7DE8\u96C6\u5BFE\u8C61\u3092\u9078\u629E\u3057\u307E\u3057\u305F\u3002\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306ECoding\u3092\u30AA\u30F3\u306B\u3059\u308B\u3068\u4F7F\u7528\u3057\u307E\u3059",
"info")}o(selectCodingTargetFromButton,"selectCodingTargetFromButton");function renderCodingDiffLines(e){
return String(e||"").split(`
`).map(t=>{let n="coding-diff-context";return t.startsWith("+++")||t.startsWith("---")?n="coding-dif\
f-file":t.startsWith("@@")?n="coding-diff-hunk":t.startsWith("+")?n="coding-diff-added":t.startsWith(
"-")&&(n="coding-diff-removed"),`<span class="${n}">${escapeHtml(t||" ")}</span>`}).join(`
`)}o(renderCodingDiffLines,"renderCodingDiffLines");function appendCodingLiveDiff(e,t){if(!e||!t||!t.
diff)return;let n=e.querySelector(".coding-live-diff");n||(n=document.createElement("div"),n.className=
"coding-live-diff",n.innerHTML='<div class="coding-live-diff-header"><span><i class="fas fa-code-bra\
nch"></i> Live Code Changes</span><span class="coding-live-diff-count">0 edits</span></div><div clas\
s="coding-live-diff-list"></div>',e.appendChild(n));const i=Math.max(0,Number(t.edit_index||0));if(i&&
n.querySelector(`[data-coding-edit-index="${i}"]`))return;const a=n.querySelector(".coding-live-diff\
-list"),r=document.createElement("div");r.className="coding-live-diff-edit",i&&r.setAttribute("data-\
coding-edit-index",String(i));const l=Number(t.repair_attempt||0)>0?` \xB7 Auto repair ${Number(t.repair_attempt)}`:
"";r.innerHTML=`<div class="coding-live-diff-meta">Edit ${i} \xB7 ${escapeHtml(t.language||"text")}${l}\
</div><pre>${renderCodingDiffLines(t.diff)}</pre>`,a&&a.appendChild(r);const c=n.querySelector(".cod\
ing-live-diff-count"),u=n.querySelectorAll(".coding-live-diff-edit").length;c&&(c.textContent=`${u} \
edit${u===1?"":"s"}`),n.scrollIntoView({block:"nearest",behavior:"smooth"})}o(appendCodingLiveDiff,"\
appendCodingLiveDiff");function isHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
return n==="html"||n==="htm"||n==="xhtml"?!0:n?!1:/<!doctype\s+html/i.test(t||"")}o(isHtmlPreviewCandidate,
"isHtmlPreviewCandidate");function openHtmlCodePreview(e){if(!e)return;let t="";try{t=decodeURIComponent(
e)}catch{showToast("HTML\u30D7\u30EC\u30D3\u30E5\u30FC\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}detectBlockedScriptsInCode(t)&&showToast("\u26A0 \u5371\u967A\u306A\u5916\u90E8\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u691C\u77E5 (polyfill.io \u306A\u3069)\u3002\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\
\u306F\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u958B\u304D\u307E\u3059\u3002","warning",!0);const i=sanitizeHtmlForPreview(
t);openSandboxedHtmlTab(i)}o(openHtmlCodePreview,"openHtmlCodePreview");function snapshotCodeCollapse(e){
if(!e)return[];const t=[];return e.querySelectorAll(".code-wrapper").forEach((n,i)=>{const a=String(
i),r=n.classList.contains("collapsed")||n.getAttribute("data-collapsed")==="true";t.push({key:a,collapsed:r})}),
t}o(snapshotCodeCollapse,"snapshotCodeCollapse");function applyCodeCollapse(e,t=[],n=!1){if(!e)return;
const i=new Map;t.forEach(a=>i.set(a.key,a.collapsed)),e.querySelectorAll(".code-wrapper").forEach((a,r)=>{
const l=String(r),c=i.has(l)?i.get(l):n;a.setAttribute("data-collapsed",c?"true":"false"),a.classList.
toggle("collapsed",!!c);const u=a.querySelector(".code-toggle");u&&(u.setAttribute("aria-expanded",c?
"false":"true"),u.innerHTML=c?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></\
i>',u.title=c?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",u.setAttribute("aria-label",c?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080"))})}o(applyCodeCollapse,"applyCodeCollapse");function snapshotCodeCollapseByMessage(e){
if(!e)return new Map;const t=new Map;return e.querySelectorAll(".message-group").forEach(n=>{const i=n.
getAttribute("id")||"";n.querySelectorAll(".code-wrapper").forEach((a,r)=>{const l=a.getAttribute("d\
ata-code-key")||String(r),c=a.classList.contains("collapsed")||a.getAttribute("data-collapsed")==="t\
rue";t.set(`${i}:${l}`,c)})}),t}o(snapshotCodeCollapseByMessage,"snapshotCodeCollapseByMessage");function applyCodeCollapseByMessage(e,t,n=!1){
e&&e.querySelectorAll(".message-group").forEach(i=>{const a=i.getAttribute("id")||"";i.querySelectorAll(
".code-wrapper").forEach((r,l)=>{const c=r.getAttribute("data-code-key")||String(l),u=`${a}:${c}`,f=t&&
t.has(u)?t.get(u):n;r.setAttribute("data-collapsed",f?"true":"false"),r.classList.toggle("collapsed",
!!f);const g=r.querySelector(".code-toggle");g&&(g.setAttribute("aria-expanded",f?"false":"true"),g.
innerHTML=f?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></i>',g.title=f?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080",g.setAttribute("aria-label",f?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080"))})})}
o(applyCodeCollapseByMessage,"applyCodeCollapseByMessage");function buildTokenTotals(e){const t={tokens_total:0,
tokens_in:0,tokens_out:0,tokens_content:0,tokens_thought:0};let n=!1,i=!1,a=!1,r=!1,l=!1;return(e||[]).
forEach(c=>{if(!c)return;let u=null;c.tokens!==null&&c.tokens!==void 0?u=Number(c.tokens||0):(c.tokens_in!==
null&&c.tokens_in!==void 0||c.tokens_out!==null&&c.tokens_out!==void 0)&&(u=Number(c.tokens_in||0)+Number(
c.tokens_out||0)),u!==null&&(t.tokens_total+=u,n=!0),c.tokens_in!==null&&c.tokens_in!==void 0&&(t.tokens_in+=
Number(c.tokens_in||0),i=!0),c.tokens_out!==null&&c.tokens_out!==void 0&&(t.tokens_out+=Number(c.tokens_out||
0),a=!0),c.tokens_content!==null&&c.tokens_content!==void 0&&(t.tokens_content+=Number(c.tokens_content||
0),r=!0),c.tokens_thought!==null&&c.tokens_thought!==void 0&&(t.tokens_thought+=Number(c.tokens_thought||
0),l=!0)}),{tokens_total:n?t.tokens_total:0,tokens_in:i?t.tokens_in:null,tokens_out:a?t.tokens_out:null,
tokens_content:r?t.tokens_content:null,tokens_thought:l?t.tokens_thought:null}}o(buildTokenTotals,"b\
uildTokenTotals");function updateTotalTokenBar(e,t=null,n=null){const i=get("total-token-bar"),a=get(
"total-token-count"),r=get("total-token-count-all-branches");if(!i||!a)return;const l=Number(e||0),c=Number(
n&&n.tokens_total||0);l>0||c>0?(i.classList.remove("hidden"),a.innerText=`Total: ${l} tokens`,t?(a.classList.
add("cursor-pointer","underline","decoration-dotted"),messageMeta.__total__={tokens_total:l,tokens_in:t.
tokens_in,tokens_out:t.tokens_out,tokens_content:t.tokens_content,tokens_thought:t.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation"},a.onclick=()=>openTokenDetail("__total__")):(a.classList.remove("\
cursor-pointer","underline","decoration-dotted"),a.onclick=null,delete messageMeta.__total__),r&&(n&&
c>0?(r.classList.remove("hidden"),r.classList.add("cursor-pointer","underline","decoration-dotted"),
r.innerText=`All branches: ${c} tokens`,messageMeta.__total_all_branches__={tokens_total:c,tokens_in:n.
tokens_in,tokens_out:n.tokens_out,tokens_content:n.tokens_content,tokens_thought:n.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation (All branches)"},r.onclick=()=>openTokenDetail("__total_all_branche\
s__")):(r.classList.add("hidden"),r.classList.remove("cursor-pointer","underline","decoration-dotted"),
r.innerText="All branches: 0 tokens",r.onclick=null,delete messageMeta.__total_all_branches__))):(i.
classList.add("hidden"),a.innerText="Total: 0 tokens",a.classList.remove("cursor-pointer","underline",
"decoration-dotted"),a.onclick=null,delete messageMeta.__total__,r&&(r.classList.add("hidden"),r.classList.
remove("cursor-pointer","underline","decoration-dotted"),r.innerText="All branches: 0 tokens",r.onclick=
null),delete messageMeta.__total_all_branches__)}o(updateTotalTokenBar,"updateTotalTokenBar");const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS=300;
let promptTokenEstimateTimer=null,promptTokenEstimateAbort=null,promptTokenEstimateSeq=0,promptTokenEstimateLastKey="",
promptTokenEstimateLastData=null;function setPromptTokenEstimateText(e,t="text-gray-400"){const n=get(
"prompt-token-estimate");if(n){if(!e){n.classList.add("hidden"),n.innerText="";return}n.className=`m\
t-1 px-1 text-[10px] ${t}`,n.classList.remove("hidden"),n.innerText=e}}o(setPromptTokenEstimateText,
"setPromptTokenEstimateText");function buildPromptTokenEstimatePayload(){return{model:get("model-sel\
ect")&&get("model-select").value?get("model-select").value:"",message:get("prompt-input")&&get("prom\
pt-input").value?get("prompt-input").value:"",quote_text:currentQuote||"",image_urls:collectImageUrlsForSend()}}
o(buildPromptTokenEstimatePayload,"buildPromptTokenEstimatePayload");function renderPromptTokenEstimate(e,t=null){
const n=t||buildPromptTokenEstimatePayload(),i=!!((n.message||"").trim()||(n.quote_text||"").trim()),
a=Array.isArray(n.image_urls)&&n.image_urls.length>0;if(!i&&!a){setPromptTokenEstimateText("");return}
if(e&&e.pending){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u4E2D...",
"text-gray-500");return}if(!e){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"text-red-300");return}if(!e.countable){setPromptTokenEstimateText("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u8868\u793A\u5BFE\u8C61\u5916\u3067\u3059",
"text-gray-500");return}const r=Number(e.tokens_total||0),l=Number(e.tokens_prompt||0),c=Number(e.tokens_files||
0),u=[];Number(e.files_non_text||0)>0&&u.push(`\u975E\u30C6\u30AD\u30B9\u30C8${e.files_non_text}\u4EF6\u306F0\u63DB\
\u7B97`),Number(e.files_missing||0)>0&&u.push(`\u672A\u691C\u51FA${e.files_missing}\u4EF6`),Number(e.
files_error||0)>0&&u.push(`\u5931\u6557${e.files_error}\u4EF6`);const f=u.length?` \u30FB ${u.join("\
 / ")}`:"";setPromptTokenEstimateText(`\u5165\u529B\u898B\u7A4D: ${r} tokens (\u672C\u6587 ${l} / \u30D5\u30A1\
\u30A4\u30EB ${c})${f}`,"text-cyan-300")}o(renderPromptTokenEstimate,"renderPromptTokenEstimate");function schedulePromptTokenEstimate(e=!1){
const t=buildPromptTokenEstimatePayload(),n=!!((t.message||"").trim()||(t.quote_text||"").trim()),i=Array.
isArray(t.image_urls)&&t.image_urls.length>0;if(!n&&!i){promptTokenEstimateLastKey="",promptTokenEstimateLastData=
null,promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=null),
promptTokenEstimateAbort&&(promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=null),renderPromptTokenEstimate(
null,t);return}const a=JSON.stringify([t.model||"",t.message||"",t.quote_text||"",t.image_urls||[]]);
if(a===promptTokenEstimateLastKey&&promptTokenEstimateLastData){renderPromptTokenEstimate(promptTokenEstimateLastData,
t);return}promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=
null);const r=o(async()=>{promptTokenEstimateAbort&&promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=
new AbortController;const l=++promptTokenEstimateSeq;renderPromptTokenEstimate({pending:!0},t);try{const c=await apiFetch(
CHAT_CONFIG.urls.estimatePromptTokensApi,{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(t),signal:promptTokenEstimateAbort.signal});if(!c.ok)throw new Error(`HTTP ${c.status}`);
const u=await c.json();if(l!==promptTokenEstimateSeq)return;promptTokenEstimateLastKey=a,promptTokenEstimateLastData=
u,renderPromptTokenEstimate(u,t)}catch(c){if(c&&c.name==="AbortError"||l!==promptTokenEstimateSeq)return;
promptTokenEstimateLastKey="",promptTokenEstimateLastData=null,renderPromptTokenEstimate(null,t)}},"\
run");e?r():promptTokenEstimateTimer=setTimeout(r,PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS)}o(schedulePromptTokenEstimate,
"schedulePromptTokenEstimate");function updatePromptPlaceholder(){const e=get("prompt-input");e&&(editingMessageId?
e.placeholder="\u7DE8\u96C6\u4E2D... (Enter\u9001\u4FE1\u306F\u8A2D\u5B9A\u306B\u5F93\u3044\u307E\u3059)":
enterToSend?e.placeholder="Enter \u3067\u9001\u4FE1 (Shift+Enter \u3067\u6539\u884C)":e.placeholder=
"Ctrl + Enter \u3067\u9001\u4FE1...")}o(updatePromptPlaceholder,"updatePromptPlaceholder");function readPromptBarModeFromForm(){
return get("set-minimal-prompt-mode")&&get("set-minimal-prompt-mode").checked?{compact_prompt_mode:!1,
minimal_prompt_mode:!0}:get("set-compact-prompt-mode")&&get("set-compact-prompt-mode").checked?{compact_prompt_mode:!0,
minimal_prompt_mode:!1}:{compact_prompt_mode:!1,minimal_prompt_mode:!1}}o(readPromptBarModeFromForm,
"readPromptBarModeFromForm");function writePromptBarModeToForm(e,t){const n=get("set-prompt-bar-mode\
-normal"),i=get("set-compact-prompt-mode"),a=get("set-minimal-prompt-mode");t&&a?a.checked=!0:e&&i?i.
checked=!0:n&&(n.checked=!0)}o(writePromptBarModeToForm,"writePromptBarModeToForm");function placeModelSelectorButton(){
const e=get("model-selector-btn"),t=get("top-model-bar"),n=get("prompt-primary-controls"),i=get("mod\
el-select");if(!(!e||!t||!n)){if(minimalPromptMode){e.parentElement!==t&&t.appendChild(e);return}if(i&&
i.parentElement===n){e.previousElementSibling!==i&&i.insertAdjacentElement("afterend",e);return}e.parentElement!==
n&&n.insertBefore(e,n.firstChild)}}o(placeModelSelectorButton,"placeModelSelectorButton");function applyMinimalPromptMode(){
const e=!!minimalPromptMode;document.body.classList.toggle("minimal-prompt-mode",e);const t=get("top\
-model-bar");t&&(t.classList.toggle("hidden",!e),t.classList.toggle("flex",e));const n=get("upload-b\
tn"),i=n?n.querySelector("i"):null;i&&(i.className=e?"fas fa-plus":"fas fa-paperclip"),n&&(n.title=e?
"\u30AA\u30D7\u30B7\u30E7\u30F3":"Upload"),e||(closeMinimalOptions(),hideThinkingSlider()),placeModelSelectorButton()}
o(applyMinimalPromptMode,"applyMinimalPromptMode");function applyPromptControlMode(){const e=get("pr\
ompt-details-controls"),t=get("prompt-controls-toggle-btn"),n=get("prompt-controls-toggle-text"),i=get(
"prompt-controls-toggle-icon"),a=get("prompt-controls-row");if(applyMinimalPromptMode(),!e||!t)return;
const r=compactPromptMode&&!minimalPromptMode,l=!r||promptControlsExpanded;a&&a.classList.toggle("co\
mpact-collapsed",r&&!l),r?l?(e.classList.remove("collapsed"),e.classList.add("expanded"),e.classList.
remove("hidden")):(e.classList.remove("expanded"),e.classList.add("collapsed")):(e.classList.remove(
"hidden"),e.classList.remove("collapsed"),e.classList.remove("expanded")),r?(t.classList.remove("hid\
den"),t.classList.add("inline-flex"),t.setAttribute("aria-expanded",l?"true":"false"),n&&(n.textContent=
l?"\u6298\u308A\u305F\u305F\u3080":"\u8A73\u7D30"),i&&(i.className=l?"fas fa-chevron-up text-[10px]":
"fas fa-chevron-down text-[10px]")):(t.classList.add("hidden"),t.classList.remove("inline-flex"),t.setAttribute(
"aria-expanded","true"),n&&(n.textContent="\u8A73\u7D30"),i&&(i.className="fas fa-chevron-down text-\
[10px]"))}o(applyPromptControlMode,"applyPromptControlMode");function setCompactPromptMode(e,t=!1){compactPromptMode=
!!e,compactPromptMode&&(minimalPromptMode=!1),compactPromptMode?t||(promptControlsExpanded=!1):promptControlsExpanded=
!0,applyPromptControlMode()}o(setCompactPromptMode,"setCompactPromptMode");function setMinimalPromptMode(e){
minimalPromptMode=!!e,minimalPromptMode&&(compactPromptMode=!1,promptControlsExpanded=!1),applyPromptControlMode()}
o(setMinimalPromptMode,"setMinimalPromptMode");function togglePromptControlDetails(){compactPromptMode&&
(promptControlsExpanded=!promptControlsExpanded,applyPromptControlMode())}o(togglePromptControlDetails,
"togglePromptControlDetails");const MINIMAL_MODEL_PANEL_IDS=["gpt-image-options","gemini-image-optio\
ns","grok-image-options","xai-chat-options","grok-video-options","mistral-ocr-options","image-input-\
limits","audio-gen-options"],THINKING_LEVELS=[{value:"minimal",label:"Min"},{value:"low",label:"Low"},
{value:"medium",label:"Mid"},{value:"high",label:"High"}],MINIMAL_POPUP_ITEMS=[{key:"attach",icon:"f\
a-paperclip",label:"\u30D5\u30A1\u30A4\u30EB\u3092\u6DFB\u4ED8",action:"upload"},{key:"voice-input",
icon:"fa-microphone",label:"Voice Input",action:"button",buttonId:"mic-btn"},{key:"rich-paste",icon:"\
fa-paste",label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051",action:"button",buttonId:"rich-paste-bt\
n"},{key:"canvas",icon:"fa-window-restore",label:"Canvas",checkboxId:"enable-canvas-mode",containerId:"\
canvas-mode-container"},{key:"coding",icon:"fa-code-branch",label:"Coding",checkboxId:"enable-coding\
-mode",containerId:"coding-mode-container"},{key:"fast",icon:"fa-bolt",label:"\u9AD8\u901F",checkboxId:"\
enable-browser-fast-mode",containerId:"browser-fast-mode-container"},{key:"batch",icon:"fa-layer-gro\
up",label:"Batch",checkboxId:"enable-batch-mode",containerId:"batch-mode-container"},{key:"search",icon:"\
fa-search",label:"Search",checkboxId:"enable-search",containerId:"search-container"},{key:"urls",icon:"\
fa-link",label:"URLs",checkboxId:"enable-url-context",containerId:"url-context-container"},{key:"map\
s",icon:"fa-map-location-dot",label:"Maps",checkboxId:"enable-maps",containerId:"maps-grounding-cont\
ainer"},{key:"python",icon:"fa-code",label:"Python",checkboxId:"enable-python",containerId:"python-c\
ontainer"},{key:"file",icon:"fa-file-lines",label:"File",checkboxId:"enable-file-creation",containerId:"\
file-creation-container"},{key:"mcp",icon:"fa-plug",label:"MCP",checkboxId:"enable-mcp",containerId:"\
mcp-container"},{key:"sysprompt",icon:"fa-terminal",label:"SysPrompt",checkboxId:"enable-sys-prompt",
containerId:"sys-prompt-option",gear:!0,gearAction:o(()=>{window.openThreadModal&&window.openThreadModal()},
"gearAction")},{key:"thinking",icon:"fa-brain",label:"Thinking",checkboxId:"enable-thinking",containerId:"\
thinking-options",special:"thinking"},{key:"effort",icon:"fa-sliders-h",label:"Effort",containerId:"\
reasoning-effort-container",selectId:"reasoning-effort"},{key:"safety",icon:"fa-shield-halved",label:"\
Safety",selectId:"safety-setting"},{key:"promptcache",icon:"fa-database",label:"PromptCache",checkboxId:"\
enable-prompt-cache",containerId:"prompt-cache-container"},{key:"compress",icon:"fa-compress-alt",label:"\
Compress",checkboxId:"enable-compression",containerId:"compression-option",gear:!0,gearAction:o(()=>{
window.openCompressionModal&&window.openCompressionModal()},"gearAction")},{key:"tempchat",icon:"fa-\
hourglass-half",label:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8",checkboxId:"enable-temporary-chat",containerId:"\
temporary-chat-container",gear:!0,gearAction:o(()=>openTemporaryChatSettings(),"gearAction")}];let minimalOptionsOpen=!1,
thinkingSliderOpen=!1,thinkingSliderTimer=null,thinkingSliderStartY=0,thinkingSliderStartX=0,thinkingSliderDragging=!1,
thinkingSliderAxis=null,popupSwipeStartY=0,popupSwipeStartX=0,popupSwipeDragging=!1,popupSwipeAtTop=!1,
popupSwipeAxis=null;const minimalPanelOrigins=new Map;function minimalOptionVisible(e){if(e.containerId){
const t=get(e.containerId);if(!t||t.classList.contains("hidden"))return!1}return!0}o(minimalOptionVisible,
"minimalOptionVisible");function minimalOptionDisabled(e){if(e.special==="thinking"){const t=get(e.containerId);
return!!(t&&t.classList.contains("pointer-events-none"))}if(e.checkboxId){const t=get(e.checkboxId);
if(t&&t.disabled)return!0}if(e.containerId){const t=get(e.containerId);if(t&&t.classList.contains("p\
ointer-events-none"))return!0}return!1}o(minimalOptionDisabled,"minimalOptionDisabled");function minimalOptionChecked(e){
if(!e.checkboxId)return!1;const t=get(e.checkboxId);return!!t&&t.checked}o(minimalOptionChecked,"min\
imalOptionChecked");function currentThinkingLevelLabel(){const e=get("thinking-level");if(!e)return THINKING_LEVELS[3].
label;const t=THINKING_LEVELS.find(n=>n.value===e.value);return t?t.label:e.selectedOptions[0]?e.selectedOptions[0].
textContent.trim():THINKING_LEVELS[3].label}o(currentThinkingLevelLabel,"currentThinkingLevelLabel");
function buildMinimalOptionItem(e){const t=document.createElement("div");t.className="minimal-option\
-item",t.dataset.key=e.key,e.action&&t.classList.add("action-"+e.action),minimalOptionChecked(e)?t.classList.
add("on"):t.classList.add("off"),minimalOptionDisabled(e)&&t.classList.add("disabled");const n=document.
createElement("i");n.className="fas "+e.icon+" minimal-option-icon",t.appendChild(n);const i=document.
createElement("span");if(i.className="minimal-option-label",i.textContent=e.label,t.appendChild(i),e.
special==="thinking"){const a=document.createElement("span");a.className="thinking-slide-value minim\
al-option-thinking-level",a.textContent=currentThinkingLevelLabel(),t.appendChild(a)}if(e.selectId){
const a=get(e.selectId);if(a){const r=a.cloneNode(!0);r.removeAttribute("id"),r.className="minimal-o\
ption-select",r.addEventListener("change",()=>{a.value=r.value,a.dispatchEvent(new Event("change",{bubbles:!0})),
refreshMinimalOptionItems()}),t.appendChild(r)}}if(e.gear){const a=document.createElement("button");
a.type="button",a.className="minimal-option-gear",a.title=e.label+"\u8A2D\u5B9A";const r=document.createElement(
"i");r.className="fas fa-cog",a.appendChild(r),a.addEventListener("click",l=>{l.stopPropagation(),closeMinimalOptions(),
typeof e.gearAction=="function"&&e.gearAction()}),t.appendChild(a)}return t.addEventListener("click",
()=>handleMinimalOptionClick(e)),t}o(buildMinimalOptionItem,"buildMinimalOptionItem");function renderMinimalOptionItems(){
const e=get("minimal-options-items");if(!e)return;const t=document.createDocumentFragment();MINIMAL_POPUP_ITEMS.
forEach(n=>{minimalOptionVisible(n)&&t.appendChild(buildMinimalOptionItem(n))}),e.innerHTML="",e.appendChild(
t)}o(renderMinimalOptionItems,"renderMinimalOptionItems");function refreshMinimalOptionItems(){const e=get(
"minimal-options-items");if(!e||!minimalOptionsOpen)return;const t=e.querySelectorAll(".minimal-opti\
on-item"),n={};t.forEach(i=>{n[i.dataset.key]=i}),MINIMAL_POPUP_ITEMS.forEach(i=>{const a=n[i.key];if(a){
if(!minimalOptionVisible(i)){a.classList.add("hidden");return}if(a.classList.remove("hidden"),a.classList.
toggle("on",minimalOptionChecked(i)),a.classList.toggle("off",!minimalOptionChecked(i)),a.classList.
toggle("disabled",minimalOptionDisabled(i)),i.special==="thinking"){const r=a.querySelector(".minima\
l-option-thinking-level");r&&(r.textContent=currentThinkingLevelLabel())}if(i.selectId){const r=get(
i.selectId),l=a.querySelector(".minimal-option-select");r&&l&&document.activeElement!==l&&l.value!==
r.value&&(l.value=r.value)}}})}o(refreshMinimalOptionItems,"refreshMinimalOptionItems");function handleMinimalOptionClick(e){
if(e.action==="upload"){closeMinimalOptions(),openUploadModal();return}if(e.action==="button"){const n=get(
e.buttonId);closeMinimalOptions(),n&&n.click();return}if(e.special==="thinking"){const n=get(e.checkboxId);
if(n&&!n.disabled){const i=!n.checked;n.checked=i,n.dispatchEvent(new Event("change",{bubbles:!0})),
i?(closeMinimalOptions(),showThinkingSlider()):hideThinkingSlider(),refreshMinimalOptionItems()}else
closeMinimalOptions(),showThinkingSlider();return}if(minimalOptionDisabled(e)||e.selectId)return;const t=get(
e.checkboxId);t&&(t.disabled||(t.checked=!t.checked,t.dispatchEvent(new Event("change",{bubbles:!0})),
refreshMinimalOptionItems(),e.key==="fast"?(closeMinimalOptions(),setTimeout(()=>refreshMinimalOptionItems(),
350)):e.key==="tempchat"&&setTimeout(()=>refreshMinimalOptionItems(),350)))}o(handleMinimalOptionClick,
"handleMinimalOptionClick");function moveModelPanelsIntoPopup(){const e=get("minimal-options-model-b\
ody");if(!e)return;let t=!1;MINIMAL_MODEL_PANEL_IDS.forEach(n=>{const i=get(n);if(i){if(i.parentElement===
e){i.classList.contains("hidden")||(t=!0);return}minimalPanelOrigins.has(i)||(minimalPanelOrigins.set(
i,{parent:i.parentElement,next:i.nextSibling}),e.appendChild(i),i.classList.contains("hidden")||(t=!0))}}),
refreshMinimalModelSection()}o(moveModelPanelsIntoPopup,"moveModelPanelsIntoPopup");function restoreModelPanelsFromPopup(){
get("minimal-options-model-body")&&(minimalPanelOrigins.forEach((t,n)=>{t.parent&&t.parent.contains(
n)&&(t.next&&t.next.parentNode===t.parent?t.parent.insertBefore(n,t.next):t.parent.appendChild(n))}),
minimalPanelOrigins.clear())}o(restoreModelPanelsFromPopup,"restoreModelPanelsFromPopup");function refreshMinimalModelSection(){
const e=get("minimal-options-model-body"),t=get("minimal-options-model-section");if(!e||!t)return;let n=!1;
Array.from(e.children).forEach(i=>{i.classList.contains("hidden")||(n=!0)}),t.classList.toggle("hidd\
en",!n)}o(refreshMinimalModelSection,"refreshMinimalModelSection");function openMinimalOptions(){if(minimalOptionsOpen||
!minimalPromptMode)return;hideThinkingSlider(),minimalOptionsOpen=!0,renderMinimalOptionItems(),moveModelPanelsIntoPopup();
const e=get("minimal-options-popup");if(!e)return;const t=get("minimal-options-panel");t&&(t.style.cssText=
""),e.classList.remove("minimal-options-closing","minimal-options-open"),e.classList.remove("hidden"),
e.setAttribute("aria-hidden","false"),e.offsetWidth,e.classList.add("minimal-options-open")}o(openMinimalOptions,
"openMinimalOptions");function closeMinimalOptions(){if(!minimalOptionsOpen)return;minimalOptionsOpen=
!1;const e=get("minimal-options-popup");e&&(e.classList.add("minimal-options-closing"),e.setAttribute(
"aria-hidden","true"),setTimeout(()=>{minimalOptionsOpen||(e.classList.remove("minimal-options-open",
"minimal-options-closing"),e.classList.add("hidden"))},560)),restoreModelPanelsFromPopup(),hideThinkingSlider()}
o(closeMinimalOptions,"closeMinimalOptions");function toggleMinimalOptions(){minimalOptionsOpen?closeMinimalOptions():
openMinimalOptions()}o(toggleMinimalOptions,"toggleMinimalOptions");function refreshMinimalOptionsIfOpen(){
minimalOptionsOpen&&(renderMinimalOptionItems(),refreshMinimalModelSection())}o(refreshMinimalOptionsIfOpen,
"refreshMinimalOptionsIfOpen");function allowedThinkingValues(){const e=get("thinking-level");return e?
Array.from(e.options).filter(n=>!n.disabled&&!n.classList.contains("hidden")).map(n=>n.value):THINKING_LEVELS.
map(n=>n.value)}o(allowedThinkingValues,"allowedThinkingValues");function thinkingIndexFromValue(e){
const t=THINKING_LEVELS.findIndex(n=>n.value===e);return t<0?3:t}o(thinkingIndexFromValue,"thinkingI\
ndexFromValue");function syncThinkingSliderUi(){const e=get("thinking-slider"),t=get("thinking-slide\
-value"),n=get("thinking-level"),i=thinkingIndexFromValue(n?n.value:"high");e&&(e.value=String(i)),t&&
(t.textContent=THINKING_LEVELS[i].label)}o(syncThinkingSliderUi,"syncThinkingSliderUi");function scheduleThinkingSliderHide(){
thinkingSliderTimer&&clearTimeout(thinkingSliderTimer),thinkingSliderTimer=setTimeout(()=>{thinkingSliderTimer=
null,hideThinkingSlider()},2500)}o(scheduleThinkingSliderHide,"scheduleThinkingSliderHide");function showThinkingSlider(){
if(thinkingSliderOpen){scheduleThinkingSliderHide();return}const e=get("thinking-slide-bar");if(!e)return;
const t=get("thinking-slide-inner");t&&(t.style.transform=""),thinkingSliderOpen=!0,e.classList.remove(
"hidden"),e.setAttribute("aria-hidden","false"),syncThinkingSliderUi(),e.offsetWidth,e.classList.add(
"thinking-slide-open"),scheduleThinkingSliderHide()}o(showThinkingSlider,"showThinkingSlider");function hideThinkingSlider(){
thinkingSliderTimer&&(clearTimeout(thinkingSliderTimer),thinkingSliderTimer=null);const e=get("think\
ing-slide-bar");e&&(thinkingSliderOpen=!1,e.classList.remove("thinking-slide-open"),e.setAttribute("\
aria-hidden","true"),setTimeout(()=>{thinkingSliderOpen||e.classList.add("hidden");const t=get("thin\
king-slide-inner");t&&(t.style.transform="")},360))}o(hideThinkingSlider,"hideThinkingSlider");function bindMinimalOptionsEvents(){
const e=get("minimal-options-backdrop"),t=get("minimal-options-close-btn"),n=get("minimal-options-po\
pup");n&&n.parentNode!==document.body&&document.body.appendChild(n),e&&e.addEventListener("click",()=>closeMinimalOptions()),
t&&t.addEventListener("click",()=>closeMinimalOptions()),document.addEventListener("keydown",c=>{if(c.
key==="Escape"){if(minimalOptionsOpen){closeMinimalOptions();return}thinkingSliderOpen&&hideThinkingSlider()}});
const i=get("thinking-slider");i&&i.addEventListener("input",()=>{const c=Number(i.value),u=allowedThinkingValues(),
f=get("thinking-level");if(u.length){const g=u.map(w=>thinkingIndexFromValue(w)),y=g.includes(c)?c:g.
reduce((w,v)=>Math.abs(v-c)<Math.abs(w-c)?v:w,g[0]);f&&(f.value=THINKING_LEVELS[y].value,f.dispatchEvent(
new Event("change",{bubbles:!0})))}syncThinkingSliderUi(),scheduleThinkingSliderHide()});const a=get(
"thinking-slide-close-btn");a&&a.addEventListener("click",c=>{c.stopPropagation(),hideThinkingSlider()});
const r=get("thinking-slide-bar");if(r){const c=get("thinking-slide-inner");r.addEventListener("touc\
hstart",u=>{thinkingSliderOpen&&(thinkingSliderDragging=!0,thinkingSliderStartY=u.touches[0].clientY,
thinkingSliderStartX=u.touches[0].clientX,thinkingSliderAxis=null,c&&c.classList.add("dragging"))},{
passive:!0}),r.addEventListener("touchmove",u=>{if(!thinkingSliderDragging)return;const f=u.touches[0].
clientX-thinkingSliderStartX,g=u.touches[0].clientY-thinkingSliderStartY;if(thinkingSliderAxis===null&&
(Math.abs(f)>8||Math.abs(g)>8)&&(thinkingSliderAxis=Math.abs(g)>Math.abs(f)?"v":"h"),thinkingSliderAxis===
"v")if(g>0){u.cancelable&&u.preventDefault();const y=Math.min((g-8)*.5,120);c&&(c.style.transform=y>
0?`translateY(${y}px)`:"")}else c&&(c.style.transform="")},{passive:!1}),r.addEventListener("touchen\
d",u=>{if(!thinkingSliderDragging)return;thinkingSliderDragging=!1;const f=u.changedTouches[0].clientY-
thinkingSliderStartY;c&&c.classList.remove("dragging"),thinkingSliderAxis==="v"&&f>100?(c&&(c.style.
transform=`translateY(${Math.max(f*.5,60)}px)`),hideThinkingSlider()):(c&&(c.style.transform=""),scheduleThinkingSliderHide())},
{passive:!0}),r.addEventListener("touchcancel",()=>{thinkingSliderDragging=!1,c&&(c.classList.remove(
"dragging"),c.style.transform=""),scheduleThinkingSliderHide()},{passive:!0})}const l=get("minimal-o\
ptions-panel");l&&(l.addEventListener("touchstart",c=>{if(!minimalOptionsOpen)return;popupSwipeDragging=
!0,popupSwipeStartY=c.touches[0].clientY,popupSwipeStartX=c.touches[0].clientX,popupSwipeAxis=null;let u=c.
target instanceof Element?c.target:null,f=!0;for(;u&&u!==l;){if(u.scrollTop>0){f=!1;break}u=u.parentElement}
popupSwipeAtTop=f,f&&l.classList.add("dragging")},{passive:!0}),l.addEventListener("touchmove",c=>{if(!popupSwipeDragging||
!popupSwipeAtTop||!minimalOptionsOpen)return;const u=c.touches[0].clientX-popupSwipeStartX,f=c.touches[0].
clientY-popupSwipeStartY;popupSwipeAxis===null&&(Math.abs(u)>8||Math.abs(f)>8)&&(popupSwipeAxis=Math.
abs(f)>Math.abs(u)?"v":"h"),popupSwipeAxis==="v"&&f>0&&(c.cancelable&&c.preventDefault(),l.style.transform=
`translateY(${Math.min(f*.6,140)}px)`)},{passive:!1}),l.addEventListener("touchend",c=>{if(!popupSwipeDragging)
return;popupSwipeDragging=!1;const u=c.changedTouches[0].clientY-popupSwipeStartY;l.classList.remove(
"dragging"),popupSwipeAtTop&&popupSwipeAxis!=="h"&&u>70?(l.style.transform=`translateY(${Math.max(u*
.6,100)}px)`,l.style.opacity="0",closeMinimalOptions()):l.style.transform=""},{passive:!0}),l.addEventListener(
"touchcancel",()=>{popupSwipeDragging=!1,l.classList.remove("dragging"),l.style.transform="",l.style.
opacity=""},{passive:!0}))}o(bindMinimalOptionsEvents,"bindMinimalOptionsEvents");function bindUploadButton(){
const e=get("upload-btn");e&&(e.onclick=()=>{minimalPromptMode?toggleMinimalOptions():openUploadModal()})}
o(bindUploadButton,"bindUploadButton");function applyChatDefaults(e){if(!e||(Object.prototype.hasOwnProperty.
call(e,"voice_studio_ui")&&(voiceStudioUiEnabled=e.voice_studio_ui!==!1),applyTemporaryChatTimeoutSeconds(
e.temp_chat_timeout_seconds),chatDefaultsLoaded))return;const n=!!e.use_last_chat_settings?{model:e.
last_model,enable_search:e.last_enable_search,enable_url_context:e.last_enable_url_context,enable_maps:e.
last_enable_maps,enable_python:e.last_enable_python,enable_file_creation:e.last_enable_file_creation,
enable_thinking:e.last_enable_thinking,thinking_level:e.last_thinking_level,thinking_budget:e.last_thinking_budget,
reasoning_effort:e.last_reasoning_effort,enable_system_prompt:e.last_enable_system_prompt,enable_mcp:e.
last_enable_mcp,safety_setting:e.last_safety_setting}:{model:e.default_model,enable_search:e.default_enable_search,
enable_url_context:e.default_enable_url_context,enable_maps:e.default_enable_maps,enable_python:e.default_enable_python,
enable_file_creation:e.default_enable_file_creation,enable_thinking:e.default_enable_thinking,thinking_level:e.
default_thinking_level,thinking_budget:e.default_thinking_budget,reasoning_effort:e.default_reasoning_effort,
enable_system_prompt:e.default_enable_system_prompt,enable_mcp:e.default_enable_mcp,safety_setting:e.
default_safety_setting},i=o((a,r)=>a==null||a===""?r:a,"s");n.model&&selectModelById(n.model),get("e\
nable-search")&&(get("enable-search").checked=!!i(n.enable_search,get("enable-search").checked)),get(
"enable-url-context")&&(get("enable-url-context").checked=!!i(n.enable_url_context,get("enable-url-c\
ontext").checked)),get("enable-maps")&&(get("enable-maps").checked=!!i(n.enable_maps,get("enable-map\
s").checked)),get("enable-python")&&(get("enable-python").checked=!!i(n.enable_python,get("enable-py\
thon").checked)),get("enable-file-creation")&&(get("enable-file-creation").checked=!!i(n.enable_file_creation,
get("enable-file-creation").checked)),get("enable-thinking")&&(get("enable-thinking").checked=!!i(n.
enable_thinking,get("enable-thinking").checked)),get("thinking-level")&&(get("thinking-level").value=
i(n.thinking_level,get("thinking-level").value||"high")),get("thinking-budget")&&(get("thinking-budg\
et").value=i(n.thinking_budget,get("thinking-budget").value||4096)),get("reasoning-effort")&&(get("r\
easoning-effort").value=i(n.reasoning_effort,get("reasoning-effort").value||"medium")),get("enable-s\
ys-prompt")&&(get("enable-sys-prompt").checked=!!i(n.enable_system_prompt,get("enable-sys-prompt").checked)),
get("enable-mcp")&&(get("enable-mcp").checked=!!i(n.enable_mcp,get("enable-mcp").checked)),get("safe\
ty-setting")&&(get("safety-setting").value=i(n.safety_setting,get("safety-setting").value||"default")),
chatDefaultsLoaded=!0,toggleOptions(),applyMcpPromptChipUi()}o(applyChatDefaults,"applyChatDefaults");
function setEditUi(e){const t=get("edit-bar");t&&(e?(t.classList.remove("hidden"),t.classList.add("f\
lex")):(t.classList.add("hidden"),t.classList.remove("flex")),updatePromptPlaceholder())}o(setEditUi,
"setEditUi");function cancelEdit(){editingMessageId=null,currentParentId=currentLeafId||null;const e=get(
"prompt-input");e&&(e.value="",e.style.height="auto"),currentImageUrls=[],get("file-preview").classList.
add("hidden"),get("file-input").value="",clearQuote(),setEditUi(!1)}o(cancelEdit,"cancelEdit");function beginEditMessage(e,t=!1){
const n=messageStore[e];if(n==null)return;const i=get("prompt-input");i.value=n||"",i.focus(),i.style.
height="auto",i.style.height=i.scrollHeight+"px";const a=allMessages.find(u=>u.id==e),r=messageMeta[e]||
{};a?currentParentId=a.parent_id===void 0?null:a.parent_id:r.parent_id!==void 0&&(currentParentId=r.
parent_id),editingMessageId=e,setEditUi(!0);const l=a?a.image_url:r.image_url;if(l)try{const u=JSON.
parse(l);Array.isArray(u)&&u.length?(currentImageUrls=u.map(f=>{let g="unknown",y=f;f&&typeof f=="ob\
ject"&&(g=normalizeAttachmentSource(f.source),y=f.filepath||f.path||f.url||f.file||"");const w=normalizeAttachmentPath(
y);return w&&setAttachmentSourceForPath(w,g),w}).filter(Boolean),get("file-preview").classList.remove(
"hidden"),get("file-name").innerText=`${currentImageUrls.length} files ready`):(currentImageUrls=[],
get("file-preview").classList.add("hidden"),get("file-input").value="")}catch{currentImageUrls=[],get(
"file-preview").classList.add("hidden"),get("file-input").value=""}else currentImageUrls=[],get("fil\
e-preview").classList.add("hidden"),get("file-input").value="";const c=a?a.quote_text:r.quote_text;c?
(currentQuote=c,get("quote-text-display").innerText=currentQuote,get("quote-bar").classList.add("vis\
ible")):clearQuote(),schedulePromptTokenEstimate(!0),t&&sendMessage()}o(beginEditMessage,"beginEditM\
essage");function playSendAnimation(){const e=get("send-btn");e&&(e.classList.remove("fly"),e.offsetWidth,
e.classList.add("fly"))}o(playSendAnimation,"playSendAnimation");function setSendBtnToStopMode(){const e=get(
"send-btn");if(!e)return;e.onclick=stopGeneration,isStopMode=!0,e.disabled=!1;const t=o(()=>{!e||!isStopMode||
(e.classList.add("stop-mode"),e.innerHTML='<span style="font-size:20px;line-height:1;color:#fff;">\u25A0<\
/span>',e.classList.add("btn-swap"),setTimeout(()=>e.classList.remove("btn-swap"),300))},"applyStopU\
i");if(e.classList.contains("fly")){const n=o(i=>{i.animationName==="sendBtnPop"&&(e.removeEventListener(
"animationend",n),t())},"onEnd");e.addEventListener("animationend",n),setTimeout(t,700)}else t()}o(setSendBtnToStopMode,
"setSendBtnToStopMode");function setSendBtnToSendMode(){const e=get("send-btn");e&&(e.classList.remove(
"stop-mode","fly","btn-swap"),e.innerHTML='<i class="fas fa-paper-plane"></i>',e.classList.add("btn-\
swap"),setTimeout(()=>e.classList.remove("btn-swap"),300),e.onclick=sendMessage,isStopMode=!1)}o(setSendBtnToSendMode,
"setSendBtnToSendMode");async function stopGeneration(){const e=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null,t=normalizeJobIdForUi(currentJobId),n=++manualStopSeq,i=captureStoppedPartialBubbleSnapshot(
getActiveStreamingBubbleElement());manualStopContext={seq:n,threadId:e,jobId:t,partialSnapshot:i},t&&
suppressPendingJob(t),abortController&&abortController.abort();try{if(t||e){const a={};t&&(a.job_id=
t),e&&(a.thread_id=e);const l=await(await apiFetch("/api/stop_chat",{method:"POST",headers:{"Content\
-Type":"application/json"},body:JSON.stringify(a)})).json().catch(()=>({})),c=normalizeJobIdForUi(l&&
l.job_id);c&&(suppressPendingJob(c),manualStopContext&&manualStopContext.seq===n&&(manualStopContext.
jobId=c))}manualStopContext&&manualStopContext.seq===n&&await syncThreadAfterAbortedStream(e,{retries:2,
retryDelayMs:180,notifyOnFailure:!0})&&manualStopContext.partialSnapshot&&appendStoppedPartialBubbleSnapshot(
manualStopContext.partialSnapshot,e)}finally{manualStopContext&&manualStopContext.seq===n&&(manualStopContext=
null),setSendBtnToSendMode(),updateFilePreview()}}o(stopGeneration,"stopGeneration");async function purgeCaches(){
if("caches"in window){const e=await caches.keys();await Promise.all(e.map(t=>caches.delete(t)))}if(navigator.
serviceWorker){const e=await navigator.serviceWorker.getRegistrations();await Promise.all(e.map(t=>t.
unregister()))}}o(purgeCaches,"purgeCaches");const SW_CACHE_MODE_STORAGE_KEY="ai_sw_cache_mode_v2";async function applyCacheMode(e,t={}){
if("serviceWorker"in navigator)if(e)try{await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(
appVersion)}`),localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY,"enabled")}catch{}else{const n=localStorage.
getItem(SW_CACHE_MODE_STORAGE_KEY);(!!t.forceCleanup||n!=="disabled")&&await purgeCaches(),localStorage.
setItem(SW_CACHE_MODE_STORAGE_KEY,"disabled")}}o(applyCacheMode,"applyCacheMode");function checkAndNotifyVersion(e){
!e||!appVersion||e===appVersion||(localStorage.getItem("version_notified")||"")===e||(localStorage.setItem(
"app_version",e),syncVersionUpdateCachePreferenceUi(),showModal("version-update-modal"))}o(checkAndNotifyVersion,
"checkAndNotifyVersion");async function checkVersion(){try{const e=await fetch("/api/version",{cache:"\
no-store"});if(!e.ok)return;const n=(await e.json()).version||"",i=localStorage.getItem("app_version")||
"";n&&!i&&localStorage.setItem("app_version",n),n&&i&&n!==i&&(await purgeCaches(),checkAndNotifyVersion(
n))}catch{}}o(checkVersion,"checkVersion");async function fetchChatStreamWithUnavailableRetry(e,t,n){
let i=0;for(;;){if(t.signal&&t.signal.aborted)throw new DOMException("Aborted","AbortError");try{const a=await apiFetch(
e,t),r=window.ConnectionMonitor.retryModeForResponse(a);let l=!1;if(a.status===425&&(l=(await a.clone().
json().catch(()=>({}))).code==="submission_in_progress"),!r&&!l)return window.ConnectionMonitor.markReachable(),
a;i+=1,r&&window.ConnectionMonitor.setUnavailable(r),updatePendingSkeletonStatus(n,r==="maintenance"?
"\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u7D42\u4E86\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...":"\u30B5\u30FC\u30D0\
\u30FC\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}catch(a){if(t.signal&&t.signal.aborted||a.name==="AbortError")throw a;i+=1,window.
ConnectionMonitor.setUnavailable("offline"),updatePendingSkeletonStatus(n,"\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u63A5\u7D9A\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}await window.ConnectionMonitor.waitForRetry(t.signal)}}o(fetchChatStreamWithUnavailableRetry,
"fetchChatStreamWithUnavailableRetry");function createClientRequestId(){return window.crypto&&typeof window.
crypto.randomUUID=="function"?window.crypto.randomUUID():`req-${window.crypto&&typeof window.crypto.
getRandomValues=="function"?Array.from(window.crypto.getRandomValues(new Uint32Array(4))).map(t=>t.toString(
16)).join(""):`${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`}`.slice(0,64)}o(createClientRequestId,
"createClientRequestId");async function reconnectPendingStreamUntilAvailable(e,t){const n=t!=null?String(
t):"",i=normalizeJobIdForUi(e&&e.job_id),a=i||`thread:${n}`;if(!n||pendingStreamReconnectJobs.has(a))
return;pendingStreamReconnectJobs.add(a);const r=new AbortController;let l=!1;abortController=r,currentJobId=
i,setSendBtnToStopMode();try{for(;!r.signal.aborted;){if(String(currentThreadId||"")!==n||i&&isPendingJobSuppressed(
i))return;const c=getActiveStreamingBubbleElement();if(updatePendingSkeletonStatus(c,"\u30B5\u30FC\u30D0\u30FC\u3078\u306E\u518D\u63A5\u7D9A\u3092\u5F85\u3063\u3066\u3044\
\u307E\u3059...","\u56DE\u7B54\u51E6\u7406\u306F\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u7D99\u7D9A\u3057\u3066\u3044\u307E\u3059"),
await window.ConnectionMonitor.waitForRetry(r.signal),!await loadMessages(n,{preserveDraft:!0,silent:!0,
skipHistory:!0})){window.ConnectionMonitor.probeNow();continue}const f=currentThreadPending;f&&f.job_id&&
!isPendingJobSuppressed(f.job_id)?(abortController===r&&(abortController=null),l=!0,resumePendingStream(
f)):window.ConnectionMonitor.markReachable();return}}catch(c){c.name!=="AbortError"&&sendClientDebugLog(
"error",`Stream reconnect failed: ${c.message}`)}finally{pendingStreamReconnectJobs.delete(a),abortController===
r&&(abortController=null),l||(currentJobId=null,setSendBtnToSendMode(),updateFilePreview())}}o(reconnectPendingStreamUntilAvailable,
"reconnectPendingStreamUntilAvailable"),window.initTurnstileWidget=()=>{if(!botConfig||!botConfig.turnstileSiteKey||
!window.turnstile||turnstileWidgetId!==null)return;const e=document.getElementById("turnstile-contai\
ner");e&&(e.classList.remove("hidden"),turnstileWidgetId=window.turnstile.render(e,{sitekey:botConfig.
turnstileSiteKey,size:"compact",appearance:"interaction-only",callback:o(t=>{turnstileToken=t,turnstilePending=
!1,verifyTurnstileOnServer(t)},"callback"),"expired-callback":o(()=>{turnstileToken=null,turnstilePending=
!1},"expired-callback"),"error-callback":o(()=>{turnstileToken=null,turnstilePending=!1},"error-call\
back")}),isBotDetectionActive()&&runBotDetectionGate())};async function getTurnstileToken(e=1500){if(!botConfig||
!botConfig.turnstileSiteKey)return null;if(turnstileToken)return turnstileToken;if(!window.turnstile)
return null;if(botDetectionOverlayShown&&botDetectionDialogWidgetId!==null)return turnstilePending=!0,
await new Promise(n=>{const i=turnstileToken,a=setTimeout(()=>n(null),Math.max(500,Number(e)||1500)),
r=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(a),clearInterval(r),n(turnstileToken))},
50)});if(turnstileWidgetId===null)return null;const t=document.getElementById("turnstile-container");
return t&&t.classList.remove("hidden"),turnstilePending=!0,await new Promise(n=>{const i=turnstileToken,
a=setTimeout(()=>n(null),Math.max(500,Number(e)||1500));try{window.turnstile.execute(turnstileWidgetId)}catch{
clearTimeout(a),n(null);return}const r=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(
a),clearInterval(r),verifyTurnstileOnServer(turnstileToken),n(turnstileToken))},50)})}o(getTurnstileToken,
"getTurnstileToken");function resetTurnstileToken(){if(turnstileToken=null,turnstilePending=!1,window.
turnstile&&turnstileWidgetId!==null)try{window.turnstile.reset(turnstileWidgetId)}catch{}if(window.turnstile&&
botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}}o(resetTurnstileToken,
"resetTurnstileToken");function isBotDetectionActive(){return!!(botConfig&&botConfig.globalEnabled&&
botConfig.accountEnabled&&!isAdminUser&&botConfig.turnstileSiteKey)}o(isBotDetectionActive,"isBotDet\
ectionActive");function renderBotDetectionDialogWidget(){if(botDetectionDialogWidgetId!==null||!botConfig||
!botConfig.turnstileSiteKey)return;const e=document.getElementById("bot-detection-widget-box");if(e){
if(!window.turnstile){setTimeout(renderBotDetectionDialogWidget,250);return}try{botDetectionDialogWidgetId=
window.turnstile.render(e,{sitekey:botConfig.turnstileSiteKey,theme:"dark",size:"flexible",callback:o(
t=>{turnstileToken=t,turnstilePending=!1,verifyTurnstileOnServer(t,!0,!0)},"callback"),"expired-call\
back":o(()=>{if(turnstileToken=null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.
turnstile.reset(botDetectionDialogWidgetId)}catch{}},"expired-callback"),"error-callback":o(()=>{if(turnstileToken=
null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}},
"error-callback")})}catch(t){console.error("bot-detection dialog widget error",t)}}}o(renderBotDetectionDialogWidget,
"renderBotDetectionDialogWidget");function showBotDetectionOverlay(e=""){let t=document.getElementById(
"bot-detection-overlay");if(t)t.style.display="flex";else{t=document.createElement("div"),t.id="bot-\
detection-overlay",t.style.cssText="position:fixed;inset:0;z-index:2147483000;background:rgba(3,7,18\
,0.92);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;";const i=document.
createElement("div");i.style.cssText="max-width:420px;width:100%;background:#0f172a;border:1px solid\
 #334155;border-radius:12px;padding:24px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.5);dis\
play:flex;flex-direction:column;align-items:stretch;gap:12px;";const a=document.createElement("div");
a.id="bot-detection-overlay-title",a.style.cssText="font-weight:700;font-size:15px;color:#f1f5f9;",a.
textContent=e||"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u4E2D...";const r=document.createElement("div");
r.style.cssText="font-size:12px;color:#94a3b8;line-height:1.6;",r.textContent="\u81EA\u52D5\u30A2\u30AF\u30BB\u30B9\u9632\u6B62\u306E\u305F\u3081\u3001\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u3066\u304F\u3060\
\u3055\u3044\u3002";const l=document.createElement("div");l.id="bot-detection-widget-box",l.style.cssText=
"margin-top:8px;min-height:65px;display:flex;justify-content:center;",i.appendChild(a),i.appendChild(
r),i.appendChild(l),t.appendChild(i),document.body.appendChild(t)}const n=document.getElementById("b\
ot-detection-overlay-title");e&&n&&(n.textContent=e),botDetectionOverlayShown=!0,renderBotDetectionDialogWidget()}
o(showBotDetectionOverlay,"showBotDetectionOverlay");function hideBotDetectionOverlay(){if(botDetectionOverlayShown=
!1,botDetectionDialogWidgetId!==null){try{window.turnstile.remove(botDetectionDialogWidgetId)}catch{}
botDetectionDialogWidgetId=null}const e=document.getElementById("bot-detection-widget-box");e&&e.replaceChildren();
const t=document.getElementById("bot-detection-overlay");t&&t.remove()}o(hideBotDetectionOverlay,"hi\
deBotDetectionOverlay");let botLockOverlay=null,botLockTimer=null;function showBotLockOverlay(e="\u9001\u4FE1\u64CD\
\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",t=600){
hideBotDetectionOverlay();let n=document.getElementById("bot-lock-overlay");if(n){n.style.display="f\
lex";const i=document.getElementById("bot-lock-overlay-message");i&&e&&(i.textContent=e)}else{n=document.
createElement("div"),n.id="bot-lock-overlay",n.style.cssText="position:fixed;inset:0;z-index:2147483\
000;background:rgba(3,7,18,0.94);display:flex;flex-direction:column;align-items:center;justify-conte\
nt:center;padding:24px;";const i=document.createElement("div");i.style.cssText="max-width:440px;widt\
h:100%;background:#0f172a;border:1px solid #f59e0b;border-radius:12px;padding:24px;text-align:center\
;box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;align-items:center;gap:12p\
x;";const a=document.createElement("div");a.style.cssText="font-size:26px;color:#fbbf24;",a.innerHTML=
'<i class="fas fa-lock"></i>';const r=document.createElement("div");r.id="bot-lock-overlay-title",r.
style.cssText="font-weight:700;font-size:16px;color:#fbbf24;",r.textContent="\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F";
const l=document.createElement("div");l.id="bot-lock-overlay-message",l.style.cssText="font-size:13p\
x;color:#f1f5f9;line-height:1.7;",l.textContent=e;const c=document.createElement("div");c.id="bot-lo\
ck-overlay-timer",c.style.cssText="font-size:12px;color:#94a3b8;margin-top:2px;";const u=document.createElement(
"div");u.style.cssText="font-size:11px;color:#94a3b8;line-height:1.6;",u.textContent="\u30ED\u30C3\u30AF\u89E3\u9664\u307E\u3067\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\
\u304F\u3060\u3055\u3044\u3002\u540C\u3058\u64CD\u4F5C\u3092\u7E70\u308A\u8FD4\u3059\u3068BAN\u3055\u308C\u308B\u5834\u5408\u304C\u3042\u308A\u307E\u3059\u3002",
i.appendChild(a),i.appendChild(r),i.appendChild(l),i.appendChild(c),i.appendChild(u),n.appendChild(i),
document.body.appendChild(n)}return botLockOverlay=n,updateBotLockTimer(t),n}o(showBotLockOverlay,"s\
howBotLockOverlay");function updateBotLockTimer(e){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const t=document.getElementById("bot-lock-overlay-timer");if(!t)return;const n=o(()=>{const i=Math.
max(0,Math.round(Number(e)||0)),a=Math.floor(i/60),r=String(i%60).padStart(2,"0");t.textContent=`\u30ED\u30C3\u30AF\
\u89E3\u9664\u307E\u3067: ${a}:${r}`},"render");n(),botLockTimer=setInterval(()=>{e-=1,n(),e<=0&&(botLockTimer&&
(clearInterval(botLockTimer),botLockTimer=null),location.reload())},1e3)}o(updateBotLockTimer,"updat\
eBotLockTimer");function hideBotLockOverlay(){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const e=document.getElementById("bot-lock-overlay");e&&e.remove(),botLockOverlay=null}o(hideBotLockOverlay,
"hideBotLockOverlay");async function applyBotLockFromServer(e){if(isAdminUser)return!0;let t=600;try{
const n=await apiFetch("/api/bot/lock",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({reason:e||""})});if(n.status===403){let a=null;try{a=await n.json()}catch{}if(a&&a.error===
"banned")return showToast("\u30ED\u30C3\u30AF\u304C\u7E70\u308A\u8FD4\u3055\u308C\u305F\u305F\u3081BAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800),!1}const i=await n.json().catch(()=>({}));
if(i&&(i.status==="skipped"||i.skipped))return!0;i&&typeof i.remaining_seconds=="number"&&(t=i.remaining_seconds)}catch{}
return showBotLockOverlay(e||"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",
t),!1}o(applyBotLockFromServer,"applyBotLockFromServer");const runBotDetectionGate=o(()=>botDetectionVerified||
!isBotDetectionActive()?Promise.resolve(!0):botDetectionGatePromise||(botDetectionGatePromise=(async()=>{
let e=0;for(;!botDetectionVerified;){if(!botDetectionOverlayShown){if(!window.__turnstileApiLoaded||
turnstileWidgetId===null){await new Promise(a=>setTimeout(a,1e3));continue}const n=await getTurnstileToken(
8e3);if(n&&await verifyTurnstileOnServer(n,!0,!1))break;e+=1;let i=!1;try{i=!!(botTelemetry&&botTelemetry.
looksSuspicious&&botTelemetry.looksSuspicious())}catch{}(e>=2||i)&&showBotDetectionOverlay();continue}
const t=await getTurnstileToken(25e3);if(t&&await verifyTurnstileOnServer(t,!0,!0))break;try{botTelemetry.
send(!0,{forceReport:!0})}catch{}await new Promise(n=>setTimeout(n,5e3))}return hideBotDetectionOverlay(),
!0})().finally(()=>{botDetectionGatePromise=null}),botDetectionGatePromise),"runBotDetectionGate");function registerSendButtonSpam(){
const e=performance.now();return sendButtonSpamTimestamps.push(e),sendButtonSpamTimestamps=sendButtonSpamTimestamps.
filter(t=>e-t<=3e3),sendButtonSpamTimestamps.length}o(registerSendButtonSpam,"registerSendButtonSpam");
function resetSendButtonSpam(){sendButtonSpamTimestamps=[]}o(resetSendButtonSpam,"resetSendButtonSpa\
m");async function runSendSpamVerification(){return isBotDetectionActive()?await applyBotLockFromServer(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002"):
!0}o(runSendSpamVerification,"runSendSpamVerification");let turnstileServerVerifiedAt=0,turnstileVerifyInFlight=null,
turnstileVerifyInFlightToken=null,turnstileLastSubmittedToken=null;async function verifyTurnstileOnServer(e,t=!1,n=null){
if(!e||!isBotDetectionActive()||botDetectionVerified)return!0;n===null&&(n=botDetectionOverlayShown);
const i=Date.now();if(!t&&i-turnstileServerVerifiedAt<60*1e3)return!0;if(turnstileVerifyInFlight&&turnstileVerifyInFlightToken===
e)return turnstileVerifyInFlight;if(turnstileLastSubmittedToken===e&&!t)return!!botDetectionVerified;
if(turnstileLastSubmittedToken===e)return turnstileVerifyInFlight&&turnstileVerifyInFlightToken===e?
turnstileVerifyInFlight:!!botDetectionVerified;turnstileLastSubmittedToken=e,turnstileVerifyInFlightToken=
e;const a=!!n;return turnstileVerifyInFlight=(async()=>{try{return(await apiFetch("/api/bot/turnstil\
e-verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({turnstile_token:e,
challenged:a})})).ok?(turnstileServerVerifiedAt=Date.now(),botDetectionVerified=!0,hideBotDetectionOverlay(),
!0):!1}catch{return!1}finally{turnstileVerifyInFlightToken===e&&(turnstileVerifyInFlight=null,turnstileVerifyInFlightToken=
null)}})(),turnstileVerifyInFlight}o(verifyTurnstileOnServer,"verifyTurnstileOnServer");function botTurnstileTokenForRequest(){
return isBotDetectionActive()?turnstileToken:null}o(botTurnstileTokenForRequest,"botTurnstileTokenFo\
rRequest");const botTelemetry=(()=>{const e={enabled:!1,windowStart:performance.now(),lastSend:0,clicks:0,
keys:0,moves:0,fastClicks:0,fastKeys:0,untrustedInput:!1,clickTimes:[],keyTimes:[],clickIntervals:[],
lastClickTs:0,lastKeyTs:0,lastMove:null,speedMax:0,speedSum:0,speedSamples:0,lastMoveSample:0},t=o(()=>{
e.enabled=!!(botConfig&&botConfig.globalEnabled&&botConfig.accountEnabled&&!isAdminUser)},"refreshEn\
abled"),n=o(()=>{e.windowStart=performance.now(),e.clicks=0,e.keys=0,e.moves=0,e.fastClicks=0,e.fastKeys=
0,e.untrustedInput=!1,e.clickTimes=[],e.keyTimes=[],e.clickIntervals=[],e.speedMax=0,e.speedSum=0,e.
speedSamples=0},"resetWindow"),i=o(w=>{const v=w&&w.target;return!v||typeof v.closest!="function"?!1:
!!v.closest("[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay")},
"isControlClick"),a=o(w=>{if(i(w))return;if(w&&w.isTrusted===!1){e.untrustedInput=!0,f(!0);return}const v=performance.
now();if(e.clicks+=1,e.lastClickTs){const k=v-e.lastClickTs;e.clickIntervals.push(k),e.clickIntervals.
length>10&&e.clickIntervals.shift(),k<120&&(e.fastClicks+=1)}e.lastClickTs=v,e.clickTimes.push(v),e.
clickTimes=e.clickTimes.filter(k=>v-k<=2e3),e.fastClicks>=4&&f(!0)},"recordClick"),r=o(w=>{if(w&&w.isTrusted===
!1){e.untrustedInput=!0,f(!0);return}const v=performance.now();e.keys+=1,e.lastKeyTs&&v-e.lastKeyTs<
50&&(e.fastKeys+=1),e.lastKeyTs=v,e.keyTimes.push(v),e.keyTimes=e.keyTimes.filter(k=>v-k<=2e3)},"rec\
ordKey"),l=o(w=>{const v=performance.now();if(!(v-e.lastMoveSample<80)){if(e.lastMoveSample=v,e.moves+=
1,e.lastMove){const k=w.clientX-e.lastMove.x,_=w.clientY-e.lastMove.y,C=v-e.lastMove.t;if(C>0){const L=Math.
sqrt(k*k+_*_)/(C/1e3);e.speedMax=Math.max(e.speedMax,L),e.speedSum+=L,e.speedSamples+=1}}e.lastMove=
{x:w.clientX,y:w.clientY,t:v}}},"recordMove"),c=o(()=>{const w=Math.max(1,performance.now()-e.windowStart),
v=e.clickTimes.length,k=e.keyTimes.length,_=e.speedSamples?e.speedSum/e.speedSamples:0;let C=0,L=1;if(e.
clickIntervals.length>=3){const E=e.clickIntervals.reduce((K,Z)=>K+Z,0)/e.clickIntervals.length,B=e.
clickIntervals.reduce((K,Z)=>K+Math.pow(Z-E,2),0)/e.clickIntervals.length;C=E,L=E>0?Math.sqrt(B)/E:1}
return{window_ms:Math.round(w),clicks:e.clicks,keys:e.keys,moves:e.moves,fast_clicks:e.fastClicks,fast_keys:e.
fastKeys,untrusted_input:!!e.untrustedInput,click_burst:v,key_burst:k,avg_click_ms:C,click_cv:L,event_rate:(e.
clicks+e.keys+e.moves)/(w/1e3),pointer_speed_max:e.speedMax,pointer_speed_avg:_}},"computeStats"),u=o(
w=>w.fast_clicks>=4||w.fast_keys>=8||w.click_burst>=8||w.key_burst>=14||w.event_rate>=20||w.avg_click_ms>
0&&w.avg_click_ms<160&&w.click_cv<.08,"isSuspicious"),f=o(async(w=!1,v={})=>{if(!e.enabled)return;const k=performance.
now();if(!w&&k-e.lastSend<3e3)return;e.lastSend=k;const _=c();if(!(!v.forceReport&&_.clicks+_.keys+_.
moves===0&&!_.untrusted_input)&&!(!w&&!_.untrusted_input&&!u(_))){_.turnstile_token=await getTurnstileToken(),
botConfig&&botConfig.turnstileSiteKey&&!_.turnstile_token&&!botDetectionVerified&&botDetectionOverlayShown&&
(_.turnstile_failed=!0,_.challenged=!0);try{const C=await apiFetch("/api/bot-telemetry",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(_)});if(C.status===403){let L=null;
try{L=await C.json()}catch{}if(L&&L.error==="banned"){showToast("\u30DC\u30C3\u30C8\u5224\u5B9A\u306B\u3088\u308ABAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800);return}}}catch{}resetTurnstileToken(),n()}},
"send");return{start:o(()=>{t(),e.enabled&&(typeof window.PointerEvent!="undefined"?document.addEventListener(
"pointerdown",a,!0):document.addEventListener("click",a,!0),document.addEventListener("keydown",r,!0),
document.addEventListener("wheel",()=>{e.moves+=1},{passive:!0}),document.addEventListener("mousemov\
e",l,!0),setInterval(()=>f(!1),4e3))},"start"),refreshEnabled:t,send:f,looksSuspicious:o(()=>{if(!e.
enabled)return!1;const w=c();return u(w)},"looksSuspicious")}})();function openFileViewer(e,t=""){if(!e)
return;const n=(t||e).split(".").pop().toLowerCase(),i=["png","jpg","jpeg","webp","gif"],a=["mp4","m\
ov","mkv","avi","m4v","webm"],r=["mp3","wav","m4a","ogg","flac"],l=["pdf","txt","md","csv","log","js\
on","docx"];if(i.includes(n)){openImageViewer(e);return}const c=get("file-viewer"),u=get("file-viewe\
r-body"),f=get("file-viewer-title");if(!(!c||!u||!f)){if(f.textContent=t||"File Preview",u.replaceChildren(),
a.includes(n)){const g=document.createElement("video");g.src=String(e),g.controls=!0,g.playsInline=!0,
g.preload="metadata",u.appendChild(g)}else if(r.includes(n)){const g=document.createElement("audio");
g.src=String(e),g.controls=!0,u.appendChild(g)}else if(l.includes(n)){const g=document.createElement(
"iframe");g.src=String(e),g.setAttribute("sandbox",""),g.referrerPolicy="no-referrer",u.appendChild(
g)}else{const g=document.createElement("div");g.className="fallback",g.appendChild(document.createTextNode(
"\u3053\u306E\u5F62\u5F0F\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u304D\u307E\u305B\u3093\u3002"));
const y=document.createElement("div");y.className="mt-3 flex justify-center gap-2";const w=document.
createElement("a");w.href=String(e),w.download="",w.className="px-3 py-1 bg-gray-800 text-white roun\
ded text-xs border border-gray-700",w.textContent="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9";const v=document.
createElement("a");v.href=String(e),v.target="_blank",v.rel="noopener noreferrer",v.className=w.className,
v.textContent="\u65B0\u3057\u3044\u30BF\u30D6\u3067\u958B\u304F",y.append(w,v),g.appendChild(y),u.appendChild(
g)}c.classList.add("visible")}}o(openFileViewer,"openFileViewer");function closeFileViewer(){const e=get(
"file-viewer"),t=get("file-viewer-body");!e||!t||(t.innerHTML="",e.classList.remove("visible"))}o(closeFileViewer,
"closeFileViewer");function showToast(e,t="error",n=!1,i=null){const a=get("toast-stack");if(!a)return;
for(;a.children.length>=3;)a.removeChild(a.firstChild);const r=document.createElement("div");return r.
className=`toast ${t}${i?" toast-clickable":""}`,r.innerHTML=`<i class="fas ${t==="error"?"fa-triang\
le-exclamation":"fa-circle-info"}"></i><span class="flex-1">${escapeHtml(e)}</span><button aria-labe\
l="close"><i class="fas fa-times"></i></button>`,r.querySelector("button").onclick=l=>{l.stopPropagation(),
r.remove()},i&&r.addEventListener("click",i),a.appendChild(r),n||setTimeout(()=>{r.parentNode&&r.remove()},
7e3),r}o(showToast,"showToast");function showProgressToast(e,t="info"){const n=get("toast-stack");if(!n)
return null;for(;n.children.length>=3;)n.removeChild(n.firstChild);const i=document.createElement("d\
iv");return i.className=`toast ${t} flex-col !items-start min-w-[240px]`,i.innerHTML=`
                <div class="flex items-center gap-2 w-full">
                    <i class="fas ${t==="error"?"fa-triangle-exclamation":"fa-circle-info"}"></i>
                    <span class="flex-1 font-bold">${escapeHtml(e)}</span>
                    <button aria-label="close" class="ml-auto opacity-50 hover:opacity-100"><i class\
="fas fa-times"></i></button>
                </div>
                <div class="w-full bg-white/10 h-1.5 rounded-full mt-2.5 overflow-hidden">
                    <div class="progress-bar h-full bg-blue-500 transition-all duration-300 shadow-[\
0_0_8px_rgba(59,130,246,0.5)]" style="width: 0%"></div>
                </div>
                <div class="w-full text-[10px] text-right mt-1.5 opacity-70 font-mono progress-text"\
>0%</div>
            `,i.querySelector("button").onclick=()=>i.remove(),n.appendChild(i),{update:o(a=>{const r=i.
querySelector(".progress-bar"),l=i.querySelector(".progress-text");r&&(r.style.width=`${Math.min(100,
Math.max(0,a))}%`),l&&(l.innerText=`${Math.round(a)}%`)},"update"),remove:o(()=>{i.parentNode&&i.remove()},
"remove")}}o(showProgressToast,"showProgressToast");let activeSettingsTab="general";const TAB_LABELS={
general:"\u4E00\u822C",api:"API\u30AD\u30FC",prompt:"\u30D7\u30ED\u30F3\u30D7\u30C8",display:"\u8868\u793A",
data:"\u30C7\u30FC\u30BF",account:"\u30A2\u30AB\u30A6\u30F3\u30C8",security:"\u30BB\u30AD\u30E5\u30EA\u30C6\u30A3",
"2fa":"2\u8981\u7D20\u8A8D\u8A3C",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",mcp:"MCP"},ALL_TABS=[
"general","api","prompt","display","data","account","security","2fa","feedback","mcp"];function getSectionHeading(e){
const t=e.querySelector("h3");if(t)return t.textContent.trim();const n=e.querySelector(".font-bold");
if(n&&!n.querySelector("input")&&!n.querySelector("select"))return n.textContent.trim();const i=e.querySelector(
"label");if(i){const a=i.textContent.trim().replace(/[：:].*$/,"").trim();if(a)return a}return""}o(
getSectionHeading,"getSectionHeading");function getSectionSnippet(e,t){const n=e.textContent,a=n.toLowerCase().
indexOf(t.toLowerCase());if(a===-1)return"";const r=Math.max(0,a-25),l=Math.min(n.length,a+t.length+
35);let c=n.substring(r,l).replace(/\s+/g," ").trim();return r>0&&(c="\u2026"+c),l<n.length&&(c=c+"\u2026"),
c}o(getSectionSnippet,"getSectionSnippet");function removeSearchOverlays(){ALL_TABS.forEach(e=>{const t=get(
"tab-"+e);if(!t)return;const n=t.querySelector(".settings-search-overlay");n&&n.remove(),Array.from(
t.children).forEach(i=>{i.classList.contains("settings-no-results")||(i.style.display="")})})}o(removeSearchOverlays,
"removeSearchOverlays");function filterSettings(){const e=get("settings-search");if(!e)return;const t=e.
value.trim().toLowerCase(),n=get("settings-search-clear");if(n&&n.classList.toggle("hidden",!t),removeSearchOverlays(),
!t){ALL_TABS.forEach(c=>{const u=get("btn-tab-"+c);if(u){const g=u.querySelector(".settings-search-b\
adge");g&&g.remove()}const f=get("tab-"+c);f&&f.classList.toggle("hidden",c!==activeSettingsTab)});return}
let i=[];ALL_TABS.forEach(c=>{const u=get("tab-"+c);u&&(u.classList.add("hidden"),Array.from(u.children).
forEach(f=>{if(!(f.classList.contains("settings-no-results")||f.classList.contains("settings-search-\
overlay"))&&f.textContent.toLowerCase().includes(t)){const g=getSectionHeading(f)||c,y=getSectionSnippet(
f,t);i.push({tabId:c,title:g,snippet:y,element:f})}}))});let a=activeSettingsTab;if(!i.some(c=>c.tabId===
a)){const c=i.find(u=>u.tabId);c&&(a=c.tabId)}const r=get("tab-"+a);if(!r)return;r.classList.remove(
"hidden"),Array.from(r.children).forEach(c=>{c.classList.contains("settings-no-results")||c.classList.
contains("settings-search-overlay")||(c.style.display="none")});const l=document.createElement("div");
if(l.className="settings-search-overlay",i.length===0){const c=document.createElement("div");c.className=
"settings-empty-state",c.innerHTML='<div class="settings-empty-icon"><i class="fas fa-search"></i></\
div><div class="settings-empty-title">\u4E00\u81F4\u3059\u308B\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093</div>';
const u=document.createElement("div");u.className="settings-empty-sub",u.textContent="\u300C"+t+"\u300D\u306B\u4E00\
\u81F4\u3059\u308B\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3002",c.appendChild(
u),l.appendChild(c)}else{const c=document.createElement("div");c.className="settings-search-count",c.
textContent=i.length+"\u4EF6\u306E\u4E00\u81F4",l.appendChild(c);let u=null;i.forEach((f,g)=>{if(f.tabId!==
u){if(u!==null){const C=document.createElement("div");C.className="border-t border-gray-700/50 my-1.\
5",l.appendChild(C)}if(f.tabId!==a){const C=document.createElement("div");C.className="text-[10px] t\
ext-gray-500 px-1 pb-1 font-bold",C.textContent="\u25BC "+(TAB_LABELS[f.tabId]||f.tabId),l.appendChild(
C)}u=f.tabId}const y=document.createElement("div");y.className="settings-search-result-item flex ite\
ms-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150",y.style.animation=
"fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both",y.style.animationDelay=g*30+"ms";const w=document.
createElement("span");w.className="settings-result-tab-badge shrink-0 mt-0.5",w.textContent=TAB_LABELS[f.
tabId]||f.tabId;const v=document.createElement("div");v.className="min-w-0 flex-1";const k=document.
createElement("div");k.className="text-sm font-bold text-white truncate",k.textContent=f.title;const _=document.
createElement("div");_.className="text-[11px] text-gray-400 truncate mt-0.5",_.textContent=f.snippet,
v.appendChild(k),v.appendChild(_),y.appendChild(w),y.appendChild(v),y.addEventListener("click",()=>jumpToSetting(
f.tabId,f.element)),l.appendChild(y)})}r.insertBefore(l,r.firstChild)}o(filterSettings,"filterSettin\
gs");function jumpToSetting(e,t){const n=get("settings-search");n&&(n.value=""),removeSearchOverlays(),
filterSettings(),e!==activeSettingsTab&&switchTab(e),setTimeout(()=>{t.scrollIntoView({behavior:"smo\
oth",block:"center"}),t.classList.add("settings-jump-highlight"),setTimeout(()=>t.classList.remove("\
settings-jump-highlight"),2e3)},260)}o(jumpToSetting,"jumpToSetting");function clickTab(e){const t=get(
"settings-search");t&&(t.value=""),switchTab(e)}o(clickTab,"clickTab");function switchTab(e){if(e===
activeSettingsTab||!ALL_TABS.includes(e))return;const t=get("tab-"+activeSettingsTab);t&&(t.classList.
remove("tab-enter"),t.classList.add("tab-exit"),setTimeout(()=>{t.classList.add("hidden"),t.classList.
remove("tab-exit")},170)),ALL_TABS.forEach(n=>{const i=get("btn-tab-"+n),a=get("tab-"+n);if(n===e){if(a&&
(a.classList.remove("hidden"),a.classList.remove("tab-exit"),a.classList.remove("tab-enter"),a.offsetWidth,
a.classList.add("tab-enter")),i){i.classList.add("is-active");try{i.scrollIntoView({inline:"nearest",
block:"nearest",behavior:"smooth"})}catch{}}}else i&&i.classList.remove("is-active")}),activeSettingsTab=
e,filterSettings(),refreshSettingsTabsScroll()}o(switchTab,"switchTab");function getSettingsTabsMaxScroll(e){
return e?Math.max(0,e.scrollWidth-e.clientWidth):0}o(getSettingsTabsMaxScroll,"getSettingsTabsMaxScr\
oll");function syncSettingsTabsOverflow(){const e=get("settings-tabs-wrap"),t=get("settings-tabs"),n=get(
"settings-tabs-arrow-left"),i=get("settings-tabs-arrow-right");if(!e||!t)return;const a=getSettingsTabsMaxScroll(
t),r=t.scrollLeft,l=a>2&&r>2,c=a>2&&r<a-2;e.classList.toggle("can-scroll",a>2),e.classList.toggle("c\
an-scroll-left",l),e.classList.toggle("can-scroll-right",c),n&&(n.disabled=!l,n.setAttribute("aria-h\
idden",l?"false":"true")),i&&(i.disabled=!c,i.setAttribute("aria-hidden",c?"false":"true"))}o(syncSettingsTabsOverflow,
"syncSettingsTabsOverflow");function refreshSettingsTabsScroll(){initSettingsTabsScroll(),syncSettingsTabsOverflow()}
o(refreshSettingsTabsScroll,"refreshSettingsTabsScroll");function initSettingsTabsScroll(){const e=get(
"settings-tabs-wrap"),t=get("settings-tabs"),n=get("settings-tabs-arrow-left"),i=get("settings-tabs-\
arrow-right");if(!e||!t||!n||!i)return;if(e.dataset.scrollBound==="1"){syncSettingsTabsOverflow();return}
e.dataset.scrollBound="1";const a=56;let r=0,l=0,c=0;const u=o(v=>{const k=e.getBoundingClientRect();
if(!k.width)return;const _=v-k.left;e.classList.toggle("is-edge-left",_>=0&&_<=a),e.classList.toggle(
"is-edge-right",_>=k.width-a&&_<=k.width)},"updateEdgeHover"),f=o(()=>{c||e.classList.remove("is-edg\
e-left","is-edge-right")},"clearEdgeHover"),g=o((v,k)=>{const _=getSettingsTabsMaxScroll(t);if(_<=0||
!v)return;const C=Math.max(0,Math.min(_,t.scrollLeft+v));k&&typeof t.scrollTo=="function"?t.scrollTo(
{left:C,behavior:"smooth"}):t.scrollLeft=C,syncSettingsTabsOverflow()},"scrollTabsBy"),y=o(()=>{c=0,
r&&(clearTimeout(r),r=0),l&&(cancelAnimationFrame(l),l=0)},"stopHold"),w=o(v=>{y(),c=v,e.classList.toggle(
"is-edge-left",v<0),e.classList.toggle("is-edge-right",v>0),g(v*Math.max(120,t.clientWidth*.55),!0),
r=setTimeout(()=>{const k=o(()=>{c&&(g(c*14,!1),l=requestAnimationFrame(k))},"step");l=requestAnimationFrame(
k)},280)},"startHold");if(e.addEventListener("pointermove",v=>{v.pointerType!=="touch"&&u(v.clientX)}),
e.addEventListener("pointerenter",v=>{v.pointerType!=="touch"&&u(v.clientX)}),e.addEventListener("po\
interleave",v=>{v.pointerType!=="touch"&&(y(),f())}),e.addEventListener("wheel",v=>{const k=getSettingsTabsMaxScroll(
t);if(k<=2)return;const C=Math.abs(v.deltaY)>=Math.abs(v.deltaX)?v.deltaY:v.deltaX;if(!C)return;const L=Math.
max(0,Math.min(k,t.scrollLeft+C));L!==t.scrollLeft&&(v.preventDefault(),t.scrollLeft=L,syncSettingsTabsOverflow())},
{passive:!1}),n.addEventListener("pointerdown",v=>{v.button!=null&&v.button!==0||(v.preventDefault(),
w(-1))}),i.addEventListener("pointerdown",v=>{v.button!=null&&v.button!==0||(v.preventDefault(),w(1))}),
n.addEventListener("click",v=>{v.preventDefault(),v.stopPropagation()}),i.addEventListener("click",v=>{
v.preventDefault(),v.stopPropagation()}),window.addEventListener("pointerup",y),window.addEventListener(
"pointercancel",y),window.addEventListener("blur",y),t.addEventListener("scroll",syncSettingsTabsOverflow,
{passive:!0}),window.addEventListener("resize",syncSettingsTabsOverflow),typeof ResizeObserver!="und\
efined")try{const v=new ResizeObserver(()=>syncSettingsTabsOverflow());v.observe(t),v.observe(e)}catch{}
syncSettingsTabsOverflow()}o(initSettingsTabsScroll,"initSettingsTabsScroll"),initSettingsTabsScroll();
const chatContainer=get("chat-container"),scrollToBottomBtn=get("scroll-to-bottom-btn"),CHAT_BOTTOM_THRESHOLD=64;
let chatAutoScrollFrame=0,chatTouchY=null,chatScrollbarDragging=!1,chatManualScrollPaused=!1,chatManualResumeArmed=!1,
chatManualPauseIntent=!1,chatPauseIntentTimer=0,chatLastScrollTop=chatContainer?chatContainer.scrollTop:
0;function isChatNearBottom(){return chatContainer?chatContainer.scrollHeight-chatContainer.scrollTop-
chatContainer.clientHeight<=CHAT_BOTTOM_THRESHOLD:!0}o(isChatNearBottom,"isChatNearBottom");function syncScrollToBottomButton(){
if(!scrollToBottomBtn)return;const e=!userAutoScroll&&!isChatNearBottom();scrollToBottomBtn.classList.
toggle("hidden",!e)}o(syncScrollToBottomButton,"syncScrollToBottomButton");function clearChatAutoScrollPauseIntent(){
chatManualPauseIntent=!1,chatPauseIntentTimer&&(clearTimeout(chatPauseIntentTimer),chatPauseIntentTimer=
0)}o(clearChatAutoScrollPauseIntent,"clearChatAutoScrollPauseIntent");function armChatAutoScrollPause(){
!chatContainer||chatManualScrollPaused||(chatManualPauseIntent=!0,chatPauseIntentTimer&&clearTimeout(
chatPauseIntentTimer),chatPauseIntentTimer=setTimeout(()=>{chatManualPauseIntent=!1,chatPauseIntentTimer=
0},500))}o(armChatAutoScrollPause,"armChatAutoScrollPause");function pauseChatAutoScroll(){chatContainer&&
(chatAutoScrollFrame&&(cancelAnimationFrame(chatAutoScrollFrame),chatAutoScrollFrame=0),clearChatAutoScrollPauseIntent(),
chatManualScrollPaused=!0,chatManualResumeArmed=!1,userAutoScroll=!1,syncScrollToBottomButton())}o(pauseChatAutoScroll,
"pauseChatAutoScroll");function resumeChatAutoScroll(e={}){clearChatAutoScrollPauseIntent(),chatManualScrollPaused=
!1,chatManualResumeArmed=!1,userAutoScroll=!0,chatContainer&&(e.scroll!==!1&&(chatContainer.scrollTop=
chatContainer.scrollHeight),chatLastScrollTop=chatContainer.scrollTop),e.scroll===!1?syncScrollToBottomButton():
scrollToBottom()}o(resumeChatAutoScroll,"resumeChatAutoScroll");function performChatAutoScroll(){chatAutoScrollFrame=
0,!(!chatContainer||!userAutoScroll)&&(chatContainer.scrollTop=chatContainer.scrollHeight,syncScrollToBottomButton())}
o(performChatAutoScroll,"performChatAutoScroll");function scrollToBottom(e=!1){if(chatContainer){if(e&&
(clearChatAutoScrollPauseIntent(),chatManualScrollPaused=!1,chatManualResumeArmed=!1,userAutoScroll=
!0),!userAutoScroll){syncScrollToBottomButton();return}chatAutoScrollFrame||(chatAutoScrollFrame=requestAnimationFrame(
performChatAutoScroll))}}if(o(scrollToBottom,"scrollToBottom"),chatContainer){chatContainer.addEventListener(
"scroll",()=>{const n=chatContainer.scrollTop;chatManualPauseIntent&&n<chatLastScrollTop-.5||chatScrollbarDragging&&
n<chatLastScrollTop-.5?pauseChatAutoScroll():chatScrollbarDragging&&chatManualScrollPaused&&n>chatLastScrollTop+
.5&&(chatManualResumeArmed=!0),chatManualScrollPaused?chatManualResumeArmed&&isChatNearBottom()?(chatManualScrollPaused=
!1,chatManualResumeArmed=!1,userAutoScroll=!0):userAutoScroll=!1:isChatNearBottom()&&(userAutoScroll=
!0),chatLastScrollTop=n,syncScrollToBottomButton()},{passive:!0}),chatContainer.addEventListener("wh\
eel",n=>{n.deltaY<0?armChatAutoScrollPause():n.deltaY>0&&chatManualScrollPaused&&(chatManualResumeArmed=
!0)},{passive:!0}),chatContainer.addEventListener("touchstart",n=>{chatTouchY=n.touches.length?n.touches[0].
clientY:null},{passive:!0}),chatContainer.addEventListener("touchmove",n=>{if(!n.touches.length)return;
const i=n.touches[0].clientY;chatTouchY!==null&&i>chatTouchY+2?armChatAutoScrollPause():chatTouchY!==
null&&i<chatTouchY-2&&chatManualScrollPaused&&(chatManualResumeArmed=!0),chatTouchY=i},{passive:!0}),
chatContainer.addEventListener("touchend",()=>{chatTouchY=null},{passive:!0}),chatContainer.addEventListener(
"pointerdown",n=>{const i=chatContainer.getBoundingClientRect().right-20;n.button===0&&n.clientX>=i&&
(chatScrollbarDragging=!0)},{passive:!0}),document.addEventListener("pointerup",()=>{chatScrollbarDragging=
!1},{passive:!0});const e=new ResizeObserver(()=>scrollToBottom());o(()=>{Array.from(chatContainer.children).
forEach(n=>e.observe(n))},"observeMessageSizes")(),new MutationObserver(n=>{n.forEach(i=>{i.addedNodes.
forEach(a=>{a.nodeType===Node.ELEMENT_NODE&&a.parentElement===chatContainer&&e.observe(a)})}),scrollToBottom()}).
observe(chatContainer,{childList:!0,subtree:!0,characterData:!0})}scrollToBottomBtn&&scrollToBottomBtn.
addEventListener("click",()=>scrollToBottom(!0)),document.addEventListener("keydown",e=>{const t=e.target,
n=t&&(t.matches("input, textarea, select")||t.isContentEditable);!n&&["ArrowUp","PageUp","Home"].includes(
e.key)?armChatAutoScrollPause():!n&&chatManualScrollPaused&&["ArrowDown","PageDown","End"].includes(
e.key)&&(chatManualResumeArmed=!0)});let viewerImages=[],viewerIndex=0,s=null,p=null,z=1,vX=0,vY=0,suppressViewerCloseClick=!1;
function applyViewerTransform(){const e=get("image-viewer-img");e&&(e.style.transform=`translate(${vX}\
px, ${vY}px) scale(${z})`)}o(applyViewerTransform,"applyViewerTransform");function resetViewerTransform(){
p=null,z=1,vX=vY=0}o(resetViewerTransform,"resetViewerTransform");function openImageViewer(e,t=".cha\
t-image"){const i=Array.from(document.querySelectorAll(t)).map(r=>({url:r.dataset.viewerSrc||r.currentSrc||
r.src,filename:r.dataset.viewerFilename||r.title||(r.dataset.viewerSrc||r.currentSrc||r.src).split("\
/").pop(),element:r})),a=i.findIndex(r=>r.url===e);if(a===-1){openViewerWithItems([{url:e,filename:e.
split("/").pop(),element:null}],0);return}openViewerWithItems(i,a)}o(openImageViewer,"openImageViewe\
r");function openViewerWithItems(e,t){viewerImages=e,viewerIndex=t>=0&&t<e.length?t:0,resetViewerTransform(),
clearViewerAdjacent(),updateViewerState(),get("image-viewer").classList.add("visible"),document.addEventListener(
"keydown",handleViewerKeydown)}o(openViewerWithItems,"openViewerWithItems");function closeImageViewer(){
get("image-viewer").classList.remove("visible"),document.removeEventListener("keydown",handleViewerKeydown),
clearViewerAdjacent(),viewerImages=[],viewerIndex=0,s=null,resetViewerTransform()}o(closeImageViewer,
"closeImageViewer");function clearViewerAdjacent(){const e=document.querySelector(".viewer-adjacent");
e&&e.remove()}o(clearViewerAdjacent,"clearViewerAdjacent");function renderViewerChrome(){if(!viewerImages.
length)return;const e=get("image-viewer-meta"),t=document.querySelector(".viewer-nav.prev"),n=document.
querySelector(".viewer-nav.next"),i=viewerImages[viewerIndex];if(e.innerText=`${viewerIndex+1} / ${viewerImages.
length} \u2022 ${i.filename}`,viewerIndex<viewerImages.length-1){const a=new Image;a.src=viewerImages[viewerIndex+
1].url}t.style.display=viewerImages.length>1?"flex":"none",n.style.display=viewerImages.length>1?"fl\
ex":"none",t.style.opacity=viewerIndex>0?"1":"0.3",n.style.opacity=viewerIndex<viewerImages.length-1?
"1":"0.3",t.style.pointerEvents=viewerIndex>0?"auto":"none",n.style.pointerEvents=viewerIndex<viewerImages.
length-1?"auto":"none"}o(renderViewerChrome,"renderViewerChrome");function updateViewerState(e){if(!viewerImages.
length)return;const t=get("image-viewer-img");if(!t)return;const n=viewerImages[viewerIndex],i=!e||e.
fade!==!1;resetViewerTransform(),renderViewerChrome(),t.style.transition="none",t.style.transform=i?
"scale(0.96)":"translate(0, 0) scale(1)",t.style.opacity=i?"0.35":"0";const a=o(()=>{t.style.transition=
i?"transform 0.28s var(--ease-out), opacity 0.28s var(--ease-out)":"none",t.style.opacity="1",t.style.
transform="scale(1)",i||clearViewerAdjacent()},"reveal");i?setTimeout(()=>{s&&s.active||(t.src=n.url,
t.onload=a,t.onerror=a,t.complete&&t.naturalWidth&&a())},140):(t.src=n.url,t.onload=a,t.onerror=a,t.
complete&&t.naturalWidth&&a())}o(updateViewerState,"updateViewerState");function navImage(e){const t=viewerIndex+
e;t>=0&&t<viewerImages.length&&(clearViewerAdjacent(),viewerIndex=t,updateViewerState())}o(navImage,
"navImage");function getViewerAdjacent(e){const t=document.querySelector(".viewer-content");if(!t)return null;
const n=viewerIndex+e;if(n<0||n>=viewerImages.length)return null;let i=t.querySelector(".viewer-adja\
cent");return i||(i=document.createElement("img"),i.className="viewer-adjacent",i.alt="",t.appendChild(
i)),i.src=viewerImages[n].url,i.dataset.dir=String(e),i}o(getViewerAdjacent,"getViewerAdjacent");function onViewerTouchStart(e){
if(!viewerImages.length)return;if(e.touches.length>=2){const n=e.touches[0],i=e.touches[1];p={d:Math.
hypot(n.clientX-i.clientX,n.clientY-i.clientY)||1,s:z,mx:(n.clientX+i.clientX)/2,my:(n.clientY+i.clientY)/
2,x:vX,y:vY},s=null,e.preventDefault();return}if(e.touches.length!==1)return;const t=e.touches[0];if(z>
1){s={startX:t.clientX,startY:t.clientY,dx:vX,dy:vY};return}s={startX:t.clientX,startY:t.clientY,lastX:t.
clientX,dx:0,dy:0,vx:0,dir:0,active:!1,resist:!1,adjacent:null,lastTime:Date.now()}}o(onViewerTouchStart,
"onViewerTouchStart");function onViewerTouchMove(e){if(e.touches.length>=2){if(!p){const E=e.touches[0],
B=e.touches[1];p={d:Math.hypot(E.clientX-B.clientX,E.clientY-B.clientY)||1,s:z,mx:(E.clientX+B.clientX)/
2,my:(E.clientY+B.clientY)/2,x:vX,y:vY},s=null}const w=e.touches[0],v=e.touches[1],k=Math.hypot(w.clientX-
v.clientX,w.clientY-v.clientY)||1,_=(w.clientX+v.clientX)/2,C=(w.clientY+v.clientY)/2;z=Math.min(6,Math.
max(1,p.s*k/p.d)),z<=1.0001?(z=1,vX=0,vY=0):(vX=p.x+_-p.mx,vY=p.y+C-p.my);const L=get("image-viewer-\
img");if(!L)return;e.preventDefault(),L.style.transition="none",applyViewerTransform();return}if(p)return;
if(z>1&&s&&!s.active){const w=e.touches[0];if(!w)return;e.preventDefault(),vX=s.dx+w.clientX-s.startX,
vY=s.dy+w.clientY-s.startY,applyViewerTransform();return}if(!s)return;const t=e.touches[0];if(!t)return;
const n=t.clientX-s.startX,i=t.clientY-s.startY,a=Date.now(),r=Math.max(a-s.lastTime,1),l=(t.clientX-
s.lastX)/r;if(s.vx=l*.6+s.vx*.4,s.lastX=t.clientX,s.lastTime=a,s.dx=n,!s.active){if(Math.abs(n)<10&&
Math.abs(i)<10)return;if(Math.abs(n)<Math.abs(i)*1.15){s=null;return}s.active=!0,s.dir=n>0?-1:1,s.adjacent=
getViewerAdjacent(s.dir),s.adjacent||(s.resist=!0)}e.preventDefault();const c=get("image-viewer-img");
if(!c)return;const u=document.querySelector(".viewer-content"),f=u?u.clientWidth:window.innerWidth,g=s.
resist?n*.3:n;c.style.transition="none",c.style.transform=`translateX(${g}px) scale(${1-Math.min(Math.
abs(g)/(f*4),.04)})`,c.style.opacity=String(Math.max(1-Math.min(Math.abs(g)/(f*.45),.55),.4));const y=s.
adjacent;if(y){const w=Number(y.dataset.dir)||0;y.style.transition="none",y.style.transform=`transla\
te(-50%, -50%) translateX(${w*f+n}px) scale(0.97)`,y.style.opacity=String(Math.min(Math.abs(n)/(f*.3),
1))}}o(onViewerTouchMove,"onViewerTouchMove");function onViewerTouchEnd(){if(p){p=null,s=null;return}
if(!s)return;const e=s;if(s=null,!e.active)return;suppressViewerCloseClick=!0,setTimeout(()=>{suppressViewerCloseClick=
!1},120);const t=get("image-viewer-img");if(!t)return;const n=document.querySelector(".viewer-conten\
t"),i=n?n.clientWidth:window.innerWidth,a=i*.22,r=e.dir||(e.dx>0?-1:1),l=window.matchMedia&&window.matchMedia(
"(prefers-reduced-motion: reduce)").matches,c=!e.resist&&(Math.abs(e.dx)>a||Math.abs(e.vx)>.45&&Math.
sign(e.dx)===r),u=e.adjacent;if(!c){if(t.style.transition="transform 0.32s var(--ease-out), opacity \
0.32s var(--ease-out)",t.style.transform="translateX(0) scale(1)",t.style.opacity="1",u){const g=u;u.
style.transition="transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)",u.style.transform=
`translate(-50%, -50%) translateX(${r*i}px) scale(0.97)`,u.style.opacity="0",setTimeout(()=>{g.isConnected&&
g.remove()},340)}return}if(l){finishSwipeNav(r);return}const f=r*i;t.style.transition="transform 0.3\
s var(--ease-out), opacity 0.3s var(--ease-out)",t.style.transform=`translateX(${f}px) scale(0.96)`,
t.style.opacity="0.2",u&&(u.style.transition="transform 0.3s var(--ease-out), opacity 0.3s var(--eas\
e-out)",u.style.transform="translate(-50%, -50%) translateX(0) scale(1)",u.style.opacity="1"),setTimeout(
()=>finishSwipeNav(r),300)}o(onViewerTouchEnd,"onViewerTouchEnd");function finishSwipeNav(e){if(!viewerImages.
length||s&&s.active)return;const t=get("image-viewer");if(!t||!t.classList.contains("visible")){clearViewerAdjacent();
return}const n=viewerIndex+e;n<0||n>=viewerImages.length||(viewerIndex=n,updateViewerState({fade:!1}))}
o(finishSwipeNav,"finishSwipeNav");function handleViewerKeydown(e){e.key==="ArrowLeft"&&navImage(-1),
e.key==="ArrowRight"&&navImage(1),e.key==="Escape"&&closeImageViewer()}o(handleViewerKeydown,"handle\
ViewerKeydown");function downloadCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex],
t=document.createElement("a");t.href=e.url,t.download=e.filename,document.body.appendChild(t),t.click(),
document.body.removeChild(t)}o(downloadCurrentImage,"downloadCurrentImage");function copyCurrentImageUrl(){
if(!viewerImages.length)return;const e=viewerImages[viewerIndex].url,t=new URL(e,window.location.origin).
href;copyToClipboard(t,()=>showToast("\u753B\u50CFURL\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F",
"success"),()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))}o(copyCurrentImageUrl,
"copyCurrentImageUrl");function reuseCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex];
let t=e.url;try{const n=new URL(t,window.location.origin);n.pathname.startsWith("/files/")&&(t=decodeURIComponent(
n.pathname.replace("/files/","")))}catch{}t&&(currentImageUrls.includes(t)?showToast("\u3053\u306E\u753B\u50CF\u306F\u65E2\u306B\u6DFB\u4ED8\u3055\u308C\u3066\u3044\u307E\
\u3059","info"):(currentImageUrls.push(t),setAttachmentNameForPath(t,e.filename||""),updateFilePreview(),
showToast("\u753B\u50CF\u3092\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F",
"success"),closeImageViewer()))}o(reuseCurrentImage,"reuseCurrentImage");async function copyToClipboard(e,t,n){
try{if(navigator.clipboard&&navigator.clipboard.writeText)await navigator.clipboard.writeText(e),t&&
t();else throw new Error("Clipboard API unavailable")}catch(i){try{const a=document.createElement("t\
extarea");a.value=e,a.style.position="fixed",a.style.left="-9999px",document.body.appendChild(a),a.focus(),
a.select();const r=document.execCommand("copy");document.body.removeChild(a),r?t&&t():n&&n(i)}catch(a){
n&&n(a)}}}o(copyToClipboard,"copyToClipboard");const isQuoteMobileLayout=o(()=>window.matchMedia("(m\
ax-width: 768px)").matches,"isQuoteMobileLayout");let quotePreviewText="";function showQuotePreview(e){
const t=get("quote-bar");quotePreviewText=e,t.classList.contains("preview")||(currentQuote="",t.classList.
add("preview")),get("quote-text-display").innerText=e,t.classList.add("visible"),schedulePromptTokenEstimate()}
o(showQuotePreview,"showQuotePreview");function handleQuotePopover(){const e=window.getSelection(),t=get(
"quote-popover");if(!t)return;const n=isQuoteMobileLayout();if(!e||e.rangeCount===0){t.style.display=
"none",t.classList.remove("show");return}const i=e.toString().trim();if(i.length>0&&get("chat-contai\
ner").contains(e.anchorNode)){if(n){showQuotePreview(i);return}const r=e.getRangeAt(0).getBoundingClientRect(),
l=t.style.display==="none"||!t.style.display||getComputedStyle(t).display==="none";t.style.display="\
block",t.style.top=r.top-40+"px",t.style.left=r.left+"px",l&&(t.classList.remove("show"),t.offsetWidth,
t.classList.add("show"))}else t.style.display="none",t.classList.remove("show")}o(handleQuotePopover,
"handleQuotePopover"),document.addEventListener("mouseup",handleQuotePopover),document.addEventListener(
"touchend",()=>setTimeout(handleQuotePopover,0),{passive:!0}),document.addEventListener("selectionch\
ange",()=>{window.getSelection&&window.getSelection().type==="Range"&&handleQuotePopover()}),get("qu\
ote-popover").onclick=()=>{currentQuote=window.getSelection().toString().trim(),currentQuote&&(get("\
quote-text-display").innerText=currentQuote,get("quote-bar").classList.add("visible"),get("prompt-in\
put").focus()),schedulePromptTokenEstimate();const e=get("quote-popover");e&&(e.style.display="none",
e.classList.remove("show"))},get("quote-confirm-btn").onclick=()=>{if(!quotePreviewText)return;currentQuote=
quotePreviewText,quotePreviewText="",get("quote-bar").classList.remove("preview"),get("prompt-input").
focus(),schedulePromptTokenEstimate()},window.clearQuote=()=>{currentQuote="",quotePreviewText="";const e=get(
"quote-bar");e.classList.remove("preview"),e.classList.remove("visible"),get("quote-text-display").innerText=
"",schedulePromptTokenEstimate()};const MODELS=[{category:"Gemini 3.8 / 3.7 / 3.6 / 3.5",icon:"fas f\
a-star text-yellow-400",description:"Google's latest multimodal models",items:[{id:"gemini-3.8-flash",
implementedAt:"2026-09-05",implementedRank:9160,quickEmoji:"\u26A1",name:"Gemini 3.8 Flash",desc:"Mo\
st intelligent Flash model for long-horizon software engineering, autonomous agents, and complex ent\
erprise workflows.",price:"In $0.75/1M, Out $3.75/1M (through 2026-12-31)",agenticView:!0},{id:"gemi\
ni-3.8-flash-cyber",implementedAt:"2026-09-20",implementedRank:9180,quickEmoji:"\u{1F6E1}\uFE0F",name:"\
Gemini 3.8 Flash Cyber",desc:"Gemini 3.8 Flash post-trained for cybersecurity workflows. Vertex AI o\
nly; access is allowlisted by Google.",price:"Google Cloud Standard PayGo / Flex PayGo / Priority Pa\
yGo",agenticView:!0},{id:"gemini-3.7-flash",implementedAt:"2026-08-14",implementedRank:8e3,quickEmoji:"\
\u26A1",name:"Gemini 3.7 Flash",desc:"Most capable Flash model for complex coding, agentic workflows\
, and multimodal tasks.",price:"In $0.75/1M, Out $3.75/1M (introductory)",agenticView:!0},{id:"gemin\
i-3.6-flash",implementedAt:"2026-07-30",implementedRank:6411,quickEmoji:"\u26A1",name:"Gemini 3.6 Fl\
ash",desc:"Latest Flash model for agentic, coding, and multimodal tasks.",price:"In $1.50/1M, Out $7\
.50/1M",agenticView:!0},{id:"gemini-3.5-flash",implementedAt:"2026-06-13",implementedRank:5900,quickEmoji:"\
\u2728",name:"Gemini 3.5 Flash",desc:"Most intelligent Gemini 3.5 model built for speed.",price:"In \
$1.50/1M, Out $9.00/1M",agenticView:!0},{id:"gemini-3.5-flash-lite",implementedAt:"2026-07-30",implementedRank:6410,
quickEmoji:"\u{1F680}",name:"Gemini 3.5 Flash-Lite",desc:"Fastest, lowest-cost Gemini 3.5 model for \
high-throughput execution.",price:"In $0.30/1M, Out $2.50/1M",agenticView:!0}]},{category:"Gemini 3.\
1 / Previous",icon:"fas fa-star text-yellow-400",description:"Previous Gemini 3.x generation models",
items:[{id:"gemini-3.1-flash-lite",implementedAt:"2026-07-30",implementedRank:6440,quickEmoji:"\u{1F4A8}",
name:"Gemini 3.1 Flash-Lite",desc:"Stable, cost-efficient model for high-volume lightweight tasks.",
price:"In $0.25/1M, Out $1.50/1M",agenticView:!0},{id:"gemini-3.1-pro-preview",implementedAt:"2026-0\
2-20",implementedRank:2430,name:"Gemini 3.1 Pro",desc:"Next-gen native multimodal model.",price:"In \
$2.00/1M, Out $12.00/1M (\u2264200k)"},{id:"gemini-3.1-flash-lite-preview",implementedAt:"2026-03-04",
implementedRank:3e3,name:"Gemini 3.1 Flash-Lite Preview",desc:"Retired preview model retained for ch\
at history compatibility.",price:"In $0.25/1M, Out $1.50/1M",deprecated:!0},{id:"gemini-3-flash-prev\
iew",implementedAt:"2026-06-13",implementedRank:5930,name:"Gemini 3.0 Flash",desc:"Fastest and most \
cost-efficient.",price:"In $0.50/1M, Out $3.00/1M"},{id:"gemini-3-pro-preview",implementedAt:"2026-0\
1-15",implementedRank:100,name:"Gemini 3.0 Pro",desc:"Shut down by Google (March 2026). Retained for\
 chat history compatibility.",price:"In $2.00/1M, Out $12.00/1M (\u2264200k)",deprecated:!0}]},{category:"\
Gemini 2.5",icon:"fas fa-history text-gray-400",description:"Gemini 2.5 generation models",items:[{id:"\
gemini-2.5-pro",implementedAt:"2026-08-25",implementedRank:8524,quickEmoji:"\u{1F9E0}",name:"Gemini \
2.5 Pro",desc:"Most advanced Gemini 2.5 model for complex reasoning, coding, and long-context analys\
is.",price:"In $1.25/1M (\u2264200k), Out $10.00/1M (\u2264200k)"},{id:"gemini-2.5-flash-lite",implementedAt:"\
2026-02-07",implementedRank:1530,name:"Gemini 2.5 Flash-Lite",desc:"Fastest and most cost-efficient \
Gemini 2.5 model.",price:"In $0.10/1M, Out $0.40/1M"},{id:"gemini-2.5-flash",implementedAt:"2026-02-\
07",implementedRank:1531,name:"Gemini 2.5 Flash",desc:"Balanced performance.",price:"In $0.30/1M, Ou\
t $2.50/1M"}]},{category:"Gemini Image (Banana)",icon:"fas fa-image text-pink-400",description:"Gemi\
ni image generation models",items:[{id:"gemini-2.5-flash-image",implementedAt:"2026-01-20",implementedRank:120,
quickEmoji:"\u{1F34C}",name:"Nano Banana",desc:"Fast image generation.",price:"In $0.30/1M, Out $0.0\
39/image"},{id:"gemini-3.1-flash-image",implementedAt:"2026-08-25",implementedRank:8526,quickEmoji:"\
\u{1F34C}",name:"Nano Banana 2",desc:"High-efficiency image generation and editing (stable).",price:"\
In $0.50/1M; Text/Thinking Out $3.00/1M; Image Out $60.00/1M ($0.067/1K image)"},{id:"gemini-3.1-fla\
sh-image-preview",implementedAt:"2026-02-26",implementedRank:2860,name:"Nano Banana 2 (Preview)",desc:"\
Retired preview retained for chat history compatibility. Use gemini-3.1-flash-image.",price:"In $0.5\
0/1M, Out $0.067/1K image ($60/1M img tokens)",deprecated:!0},{id:"gemini-3.1-flash-lite-image",implementedAt:"\
2026-07-01",implementedRank:6020,quickEmoji:"\u{1F34C}",name:"Nano Banana 2 Lite",desc:"Low-latency \
Gemini image generation and editing with 1K output.",price:"In $0.25/1M; Text/Thinking Out $1.50/1M;\
 Image Out $30/1M ($0.0336/1K image)"},{id:"gemini-3-pro-image",implementedAt:"2026-08-25",implementedRank:8525,
quickEmoji:"\u{1F34C}",name:"Nano Banana Pro",desc:"Professional image generation and editing with 4\
K output (stable).",price:"In $2.00/1M; Text/Thinking Out $12.00/1M; Image Out $120.00/1M ($0.134/1K\
-2K, $0.24/4K)"},{id:"gemini-3-pro-image-preview",implementedAt:"2026-01-25",implementedRank:130,name:"\
Nano Banana Pro (Preview)",desc:"Retired preview retained for chat history compatibility. Use gemini\
-3-pro-image.",price:"In $2.00/1M, Out $0.134 (1K/2K) or $0.24 (4K)",deprecated:!0}]},{category:"Gem\
ini Video Generation",icon:"fas fa-clapperboard text-cyan-400",description:"Gemini video generation \
models (Veo 3.1 / Omni Flash)",items:[{id:"gemini-omni-1.1-flash",implementedAt:"2026-09-02",implementedRank:9010,
quickEmoji:"\u{1F3AC}",name:"Gemini Omni 1.1 Flash",desc:"Fastest multimodal video generation and co\
nversational editing from text, images, video, and audio (native audio in output).",price:"In $1.50/\
1M (text/image/video/audio); Text Out $9.00/1M; Video $17.50/1M (\u2248$0.10/sec)"},{id:"gemini-omni\
-flash",implementedAt:"2026-08-25",implementedRank:8522,quickEmoji:"\u{1F3AC}",name:"Gemini Omni Fla\
sh",desc:"Fast conversational video generation and editing from text and images.",price:"In $1.50/1M\
; Text Out $9.00/1M; Video \u2248$0.10/sec"},{id:"veo-3.1-generate-preview",implementedAt:"2026-08-2\
5",implementedRank:8521,quickEmoji:"\u{1F3A5}",name:"Veo 3.1",desc:"Cinematic video generation with \
native audio and 4K output.",price:"$0.40/sec (720p/1080p), $0.60/sec (4K)"},{id:"veo-3.1-fast-gener\
ate-preview",implementedAt:"2026-08-25",implementedRank:8520,name:"Veo 3.1 Fast",desc:"Low-cost, fas\
t video generation from the Veo 3.1 family.",price:"$0.10/sec (720p), $0.12/sec (1080p)"},{id:"veo-3\
.1-lite-generate-preview",implementedAt:"2026-08-25",implementedRank:8519,name:"Veo 3.1 Lite",desc:"\
High-efficiency, developer-first video generation (no 4K).",price:"$0.05/sec (720p), $0.08/sec (1080\
p)"}]},{category:"Gemini Music Generation",icon:"fas fa-music text-fuchsia-400",description:"Lyria m\
usic generation models",items:[{id:"lyria-3.5",implementedAt:"2026-09-05",implementedRank:9050,quickEmoji:"\
\u{1F3BC}",name:"Lyria 3.5",desc:"Full-length song generation from text or images with vocals, lyric\
s, and structured arrangements.",price:"See Google AI pricing"},{id:"lyria-3-pro-preview",implementedAt:"\
2026-08-25",implementedRank:8518,quickEmoji:"\u{1F3B5}",name:"Lyria 3 Pro",desc:"Flagship music gene\
ration for full-length songs with structural coherence.",price:"$0.08 / song"},{id:"lyria-3-clip-pre\
view",implementedAt:"2026-08-25",implementedRank:8517,quickEmoji:"\u{1F3B6}",name:"Lyria 3 Clip",desc:"\
Short musical clips, loops, and previews (30 seconds).",price:"$0.04 / song"},{id:"lyria-realtime-ex\
p",implementedAt:"2026-08-25",implementedRank:8516,name:"Lyria RealTime",desc:"Experimental realtime\
 music generation with deep melodic control.",price:"Experimental (no vocals)"}]},{category:"Gemini \
Transcription",icon:"fas fa-microphone text-teal-400",description:"Gemini speech-to-text transcripti\
on models",items:[{id:"gemini-3.5-transcribe",implementedAt:"2026-08-27",implementedRank:8621,quickEmoji:"\
\u{1F399}\uFE0F",name:"Gemini 3.5 Transcribe",desc:"Audio-file speech-to-text with language detectio\
n, speaker diarization, word timestamps, and smart formatting (audio file up to 1 hour).",price:"In \
$2.00/1M (audio), Out $12.00/1M (text)"},{id:"gemini-3.5-transcribe-live",implementedAt:"2026-08-27",
implementedRank:8622,quickEmoji:"\u{1F534}",name:"Gemini 3.5 Transcribe Live",desc:"Real-time low-la\
tency streaming speech-to-text over the Live API (microphone input, sessions up to 10 minutes).",price:"\
In $3.50/1M (audio), Out $21.00/1M (text)"}]},{category:"OpenAI Image Gen",icon:"fas fa-paint-brush \
text-purple-400",description:"GPT Image models",items:[{id:"gpt-image-2.5-sunburst",implementedAt:"2\
026-09-09",implementedRank:9320,quickEmoji:"\u{1F31E}",name:"GPT-Image-2.5 Sunburst",desc:"Most capa\
ble image generation and editing with precision-focused quality.",price:"Text In $5/1M; Image In $8/\
1M; Image Out $30/1M"},{id:"gpt-image-2.5-flare",implementedAt:"2026-09-09",implementedRank:9321,quickEmoji:"\
\u{1F525}",name:"GPT-Image-2.5 Flare",desc:"Fast, high-quality everyday image generation and editing\
.",price:"Text In $5/1M; Image In $8/1M; Image Out $30/1M"},{id:"gpt-image-2",implementedAt:"2026-04\
-30",implementedRank:4680,name:"GPT Image 2",desc:"State-of-the-art image generation and editing.",price:"\
Text In $5/1M; Image In $8/1M; Image Out $30/1M"},{id:"gpt-image-1.5",implementedAt:"2026-03-13",implementedRank:3410,
name:"GPT Image 1.5",desc:"Previous-generation flagship image model.",price:"Text In $5/1M, Text Out\
 $10/1M; Image Out $32/1M"},{id:"gpt-image-1",implementedAt:"2026-03-13",implementedRank:3411,name:"\
GPT Image 1",desc:"Standard quality.",price:"Text In $5/1M; Image Out $40/1M"},{id:"gpt-image-1-mini",
implementedAt:"2026-03-13",implementedRank:3412,name:"GPT Image 1 Mini",desc:"Faster, lower resoluti\
on.",price:"Text In $2/1M; Image In $2.50/1M; Image Out $8/1M"}]},{category:"OpenAI GPT",icon:"fas f\
a-brain text-green-400",description:"OpenAI's flagship models",items:[{id:"gpt-5.6-sol",implementedAt:"\
2026-07-31",implementedRank:6550,quickEmoji:"\u2600\uFE0F",name:"GPT-5.6 Sol",desc:"Frontier reasoni\
ng model for complex professional work with 1.05M context.",price:"In $5.00/1M, Cached $0.50/1M, Out\
 $30.00/1M (over 272K: In $10.00, Out $45.00)"},{id:"gpt-5.6-terra",implementedAt:"2026-07-31",implementedRank:6560,
quickEmoji:"\u{1F30D}",name:"GPT-5.6 Terra",desc:"Balanced intelligence and cost for everyday work w\
ith 1.05M context.",price:"In $2.00/1M, Cached $0.20/1M, Out $12.00/1M (over 272K: In $4.00, Out $18\
.00)"},{id:"gpt-5.6-luna",implementedAt:"2026-07-31",implementedRank:6561,quickEmoji:"\u{1F319}",name:"\
GPT-5.6 Luna",desc:"Cost-efficient model for high-volume workloads with 1.05M context.",price:"In $0\
.20/1M, Cached $0.02/1M, Out $1.20/1M (over 272K: In $0.40, Out $1.80)"},{id:"gpt-4o",implementedAt:"\
2026-06-04",implementedRank:5820,name:"GPT-4o",desc:"Multimodal flagship model.",price:"In $2.50/1M,\
 Out $10.00/1M"},{id:"gpt-4o-mini",implementedAt:"2026-06-04",implementedRank:5821,name:"GPT-4o mini",
desc:"Fast, low-cost model.",price:"In $0.15/1M, Out $0.60/1M"},{id:"gpt-5.5",implementedAt:"2026-04\
-26",implementedRank:4500,name:"GPT-5.5",desc:"Experimental OpenAI model ID for accounts with access\
.",price:"In $5.00/1M, Out $30.00/1M"},{id:"gpt-5.5-mini",implementedAt:"2026-04-26",implementedRank:4501,
name:"GPT-5.5 mini",desc:"Smaller and more cost-efficient GPT-5.5 tier.",price:"Pricing not publicly\
 listed"},{id:"gpt-5.5-nano",implementedAt:"2026-04-26",implementedRank:4502,name:"GPT-5.5 nano",desc:"\
Smallest and fastest GPT-5.5 tier.",price:"Pricing not publicly listed"},{id:"gpt-5.5-pro",implementedAt:"\
2026-04-26",implementedRank:4503,name:"GPT-5.5 Pro",desc:"Higher-capacity GPT-5.5 tier for accounts \
with access.",price:"In $30.00/1M, Out $180.00/1M"},{id:"gpt-5.4",implementedAt:"2026-03-08",implementedRank:3150,
name:"GPT-5.4",desc:"Experimental OpenAI model ID for accounts with access.",price:"In $2.50/1M, Out\
 $15.00/1M"},{id:"gpt-5.4-mini",implementedAt:"2026-03-08",implementedRank:3151,name:"GPT-5.4 mini",
desc:"Smaller and more cost-efficient GPT-5.4 tier.",price:"In $0.75/1M, Out $4.50/1M"},{id:"gpt-5.4\
-nano",implementedAt:"2026-03-08",implementedRank:3152,name:"GPT-5.4 nano",desc:"Smallest and fastes\
t GPT-5.4 tier.",price:"In $0.20/1M, Out $1.25/1M"},{id:"gpt-5.4-pro",implementedAt:"2026-03-08",implementedRank:3153,
name:"GPT-5.4 Pro",desc:"Higher-capacity GPT-5.4 tier for accounts with access.",price:"In $30.00/1M\
, Out $180.00/1M"},{id:"gpt-5.2",implementedAt:"2026-02-15",implementedRank:200,name:"GPT-5.2 (Respo\
nses API)",desc:"Most capable reasoning model.",price:"In $1.75/1M, Out $14.00/1M"},{id:"gpt-5-searc\
h-api",implementedAt:"2026-02-02",implementedRank:740,name:"GPT-5 Search (API)",desc:"Search-optimiz\
ed model (Chat Completions).",price:"Model rates + Web search $10/1k calls"},{id:"gpt-5.1",implementedAt:"\
2026-02-05",implementedRank:200,name:"GPT-5.1",desc:"High intelligence.",price:"In $1.25/1M, Out $10\
.00/1M"},{id:"gpt-5-mini",implementedAt:"2026-02-02",implementedRank:770,name:"GPT-5 mini",desc:"Sma\
ll and efficient.",price:"In $0.25/1M, Out $2.00/1M"}]},{category:"DeepSeek V4.1 / V4",icon:"fas fa-\
bolt text-cyan-400",description:"DeepSeek's OpenAI-compatible V4.1 Flash and V4 Pro models",items:[{
id:"deepseek-v4.1-flash",implementedAt:"2026-09-13",implementedRank:9600,quickEmoji:"\u26A1",apiId:"\
deepseek-flash",name:"DeepSeek V4.1 Flash",desc:"V4.1 Flash with Vision, 1M context, 384K output, th\
inking, tools, and JSON.",price:"In $0.003 hit/$0.15 miss, Out $0.60 off-peak"},{id:"deepseek-v4-fla\
sh-vision-exp",implementedAt:"2026-08-23",implementedRank:8260,name:"DeepSeek V4 Flash Vision Exp",desc:"\
Retired; retained for history.",price:"Retired",deprecated:!0},{id:"deepseek-v4-flash-0731",implementedAt:"\
2026-07-31",implementedRank:6610,name:"DeepSeek V4 Flash",desc:"Retired; retained for history.",price:"\
Retired",deprecated:!0},{id:"deepseek-v4-flash",implementedAt:"2026-04-26",implementedRank:4510,name:"\
DeepSeek V4 Flash Preview",desc:"Retired; retained for history.",price:"Retired",deprecated:!0},{id:"\
deepseek-v4-pro",implementedAt:"2026-04-26",implementedRank:4511,name:"DeepSeek V4 Pro",desc:"V4 Pro\
 with 1M context, 384K output, thinking, tools, and JSON.",price:"In $0.022 hit/$0.66 miss, Out $1.9\
8 off-peak"}]},{category:"Kimi K3",icon:"fas fa-brain text-violet-400",description:"Moonshot AI's fl\
agship 2.8T-parameter model with 1M context and always-on thinking",items:[{id:"kimi-k3",implementedAt:"\
2026-07-30",implementedRank:6340,quickEmoji:"\u{1F9E0}",name:"Kimi K3",desc:"Always-reasoning flagsh\
ip model with 1M context, vision, tool calling.",price:"In $3.00/1M (miss), $0.30/1M (hit), Out $15.\
00/1M"}]},{category:"Mistral Document OCR",icon:"fas fa-file text-orange-300",description:"Document \
OCR (PDF / image / DOCX / PPTX). Not a chat completion model.",items:[{id:"mistral-ocr-4-0",implementedAt:"\
2026-08-15",implementedRank:8130,quickEmoji:"\u{1F4C4}",name:"Mistral OCR 4",desc:"Document AI OCR w\
ith markdown, tables, headers/footers, and paragraph bounding boxes. Chat history is not sent.",price:"\
$4 / 1,000 pages ($5 / 1,000 annotated pages)"}]},{category:"Anthropic Claude",icon:"fas fa-brain te\
xt-orange-400",description:"Anthropic's latest deep reasoning models",items:[{id:"claude-opus-4-6",implementedAt:"\
2026-05-01",implementedRank:480,name:"Claude Opus 4.6",desc:"Most capable model for deep reasoning a\
nd complex tasks.",price:"In $5.00/1M, Out $25.00/1M"},{id:"claude-sonnet-4-6",implementedAt:"2026-0\
5-01",implementedRank:481,name:"Claude Sonnet 4.6",desc:"Excellent balance of speed and intelligence\
 with adaptive thinking.",price:"In $3.00/1M, Out $15.00/1M"}]},{category:"Audio (TTS)",icon:"fas fa\
-microphone text-red-400",description:"Text-to-Speech models",items:[{id:"gemini-3.1-flash-tts-previ\
ew",implementedAt:"2026-04-17",implementedRank:4250,name:"Gemini 3.1 Flash TTS",desc:"Google TTS (Pr\
eview).",price:"Text In $1.00/1M, Audio Out $20.00/1M"},{id:"gpt-4o-mini-tts",implementedAt:"2026-03\
-01",implementedRank:250,name:"GPT-4o Mini TTS",desc:"OpenAI TTS.",price:"Text In $0.60/1M, Audio Ou\
t $12.00/1M"},{id:"gemini-2.5-flash-preview-tts",implementedAt:"2026-02-10",implementedRank:160,name:"\
Gemini 2.5 Flash TTS",desc:"Google TTS (Preview).",price:"Text In $0.50/1M, Audio Out $10.00/1M"},{id:"\
gemini-2.5-pro-preview-tts",implementedAt:"2026-02-10",implementedRank:161,name:"Gemini 2.5 Pro TTS",
desc:"Google TTS Pro (Preview).",price:"Text In $1.00/1M, Audio Out $20.00/1M"},{id:"google-tts-stud\
io",implementedAt:"2026-01-20",implementedRank:110,name:"Google TTS (Studio)",desc:"High fidelity st\
udio voices.",price:"$160 / 1M chars"},{id:"google-tts-neural",implementedAt:"2026-01-20",implementedRank:111,
name:"Google TTS (Neural2)",desc:"Standard neural voices.",price:"$16 / 1M chars"},{id:"grok-tts",implementedAt:"\
2026-05-27",implementedRank:5560,quickEmoji:"\u{1F50A}",name:"Grok TTS",desc:"xAI Text-to-Speech wit\
h expressive voices.",price:"$15.00 / 1M chars"}]},{category:"OpenAI Transcription",icon:"fas fa-clo\
sed-captioning text-emerald-400",description:"Speech-to-text models (audio in / text out)",items:[{id:"\
gpt-transcribe",implementedAt:"2026-07-29",implementedRank:6330,name:"GPT Transcribe",desc:"High-acc\
uracy file and committed-turn transcription.",price:"$0.0045 / minute"},{id:"gpt-live-transcribe",implementedAt:"\
2026-07-29",implementedRank:6331,name:"GPT Live Transcribe",desc:"Low-latency realtime transcription\
.",price:"$0.017 / minute"}]},{category:"Realtime Audio (STS)",icon:"fas fa-headset text-cyan-400",description:"\
Realtime voice models (audio in / audio out)",items:[{id:"gpt-realtime-2",implementedAt:"2026-05-11",
implementedRank:5080,name:"OpenAI Realtime 2",desc:"Most capable speech-to-speech reasoning model.",
price:"Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime-translate",implementedAt:"2026-05-11",implementedRank:5081,
name:"OpenAI Realtime Translate",desc:"Streaming speech-to-speech translation.",price:"$0.034 / minu\
te"},{id:"gpt-realtime-whisper",implementedAt:"2026-05-11",implementedRank:5082,name:"OpenAI Realtim\
e Whisper",desc:"Streaming speech-to-text (transcription).",price:"$0.017 / minute"},{id:"gpt-realti\
me-1.5",implementedAt:"2026-02-24",implementedRank:2530,name:"OpenAI Realtime 1.5",desc:"Latest Open\
AI speech-to-speech flagship model.",price:"Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime",implementedAt:"\
2026-02-24",implementedRank:2531,name:"OpenAI Realtime",desc:"OpenAI realtime speech-to-speech model\
.",price:"Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime-mini",implementedAt:"2026-02-24",implementedRank:2532,
name:"OpenAI Realtime Mini",desc:"Lower-latency, smaller realtime model.",price:"Audio In $10/1M, Au\
dio Out $20/1M"},{id:"gemini-2.5-flash-native-audio-preview-12-2025",implementedAt:"2026-01-15",implementedRank:90,
name:"Gemini 2.5 Flash Native Audio (Live)",desc:"Google Live native audio model.",price:"Audio In $\
3.00/1M, Audio Out $12.00/1M"},{id:"gemini-3.1-flash-live-preview",implementedAt:"2026-03-29",implementedRank:3870,
name:"Gemini 3.1 Flash Live",desc:"Google Live native audio model.",price:"Audio In $3.00/1M (~$0.00\
5/min), Out $12.00/1M"},{id:"gemini-3.8-live",implementedAt:"2026-09-20",implementedRank:9181,quickEmoji:"\
\u{1F534}",name:"Gemini 3.8 Flash Live",desc:"Low-latency audio-to-audio Live API model with interle\
aved reasoning and asynchronous function calling.",price:"See Gemini API pricing"},{id:"gemini-3.8-l\
ive-extended-thinking",implementedAt:"2026-09-20",implementedRank:9182,quickEmoji:"\u{1F9E0}",name:"\
Gemini 3.8 Live Extended Thinking",desc:"Live audio-to-audio model with configurable background reas\
oning for complex multi-step interactions.",price:"See Gemini API pricing"},{id:"gemini-3.5-live-tra\
nslate-preview",implementedAt:"2026-08-25",implementedRank:8523,quickEmoji:"\u{1F310}",name:"Gemini \
3.5 Live Translate",desc:"Low-latency real-time speech-to-speech translation supporting 70+ language\
s.",price:"Audio In $3.50/1M, Audio Out $21.00/1M"},{id:"grok-voice-think-fast-2.0",implementedAt:"2\
026-08-25",implementedRank:8502,quickEmoji:"\u{1F3A4}",name:"Grok Voice Think Fast 2.0",desc:"Curren\
t xAI speech-to-speech model.",price:"$0.08 / min ($4.80 / hr) audio + $0.004 / text input"},{id:"gr\
ok-voice-transcribe-2.0",implementedAt:"2026-09-24",implementedRank:10261,name:"Grok Voice Transcrib\
e 2.0 (Live)",desc:"xAI streaming speech-to-text.",price:"$0.20 / hr"},{id:"grok-voice-latest",implementedAt:"\
2026-05-27",implementedRank:5550,name:"Grok Voice Latest",desc:"Alias for the current flagship voice\
 model.",price:"$0.08 / min ($4.80 / hr) audio + $0.004 / text input"},{id:"grok-voice-think-fast-1.\
0",implementedAt:"2026-05-11",implementedRank:5140,name:"Grok Voice Think Fast 1.0",desc:"Deprecated\
 xAI realtime voice model retained for history compatibility.",price:"$0.05 / min ($3.00 / hr)",deprecated:!0},
{id:"grok-voice-fast-1.0",implementedAt:"2026-05-01",implementedRank:500,name:"Grok Voice Fast 1.0",
desc:"Legacy xAI realtime voice model retained for history compatibility.",price:"$0.05 / min ($3.00\
 / hr)",deprecated:!0},{id:"grok-voice-agent",implementedAt:"2026-04-01",implementedRank:380,name:"G\
rok Voice Agent",desc:"xAI realtime voice agent API.",price:"$0.05 / min (Realtime)",deprecated:!0}]},
{category:"Gemini Agent / Specialized",icon:"fas fa-robot text-indigo-400",description:"Gemini agent\
 and specialized models",items:[{id:"gemini-robotics-er-2-preview",implementedAt:"2026-08-25",implementedRank:8515,
name:"Gemini Robotics ER 2",desc:"Embodied reasoning model for robots with advanced video understand\
ing.",price:"In $2.00/1M, Out $8.00/1M"},{id:"deep-research-preview-04-2026",implementedAt:"2026-08-\
25",implementedRank:8514,quickEmoji:"\u{1F50E}",name:"Gemini Deep Research",desc:"Agentic multi-step\
 research producing comprehensive cited reports.",price:"Standard Gemini rates + tool usage fees"},{
id:"deep-research-max-preview-04-2026",implementedAt:"2026-08-25",implementedRank:8513,name:"Gemini \
Deep Research Max",desc:"Maximum-comprehension research agent over hundreds of sources.",price:"Stan\
dard Gemini rates + tool usage fees"},{id:"antigravity-preview-05-2026",implementedAt:"2026-08-25",implementedRank:8512,
name:"Antigravity Agent",desc:"Managed agent that plans, runs code, manages files, and browses the w\
eb in a sandbox.",price:"Standard Gemini rates (sandbox compute free during preview)"},{id:"gemini-2\
.5-computer-use-preview-10-2025",implementedAt:"2026-08-25",implementedRank:8511,name:"Gemini 2.5 Co\
mputer Use",desc:"Browser / desktop control agent model for UI automation.",price:"In $1.25/1M (\u2264200\
k), Out $10.00/1M (\u2264200k)"},{id:"gemini-embedding-2",implementedAt:"2026-08-25",implementedRank:8510,
name:"Gemini Embedding 2",desc:"Multimodal embedding model (text / image / audio / video / PDF).",price:"\
Text In $0.20/1M, Image $0.45/1M"}]},{category:"Grok Imagine",icon:"fas fa-magic text-blue-400",description:"\
Grok generation models",items:[{id:"grok-imagine-image-2.0",implementedAt:"2026-08-22",implementedRank:8250,
quickEmoji:"\u{1F3A8}",name:"Grok Imagine Image 2.0",desc:"Precise image generation and editing with\
 1K/2K output and low/medium quality control.",price:"from $0.04 / image"},{id:"grok-imagine-image-q\
uality",implementedAt:"2026-05-09",implementedRank:5020,name:"Grok Imagine Image Quality",desc:"Next\
-gen Grok image generation with 1K/2K support.",price:"$0.05 / image"},{id:"grok-imagine-image",implementedAt:"\
2026-01-30",implementedRank:520,name:"Grok Imagine Image",desc:"Latest Grok image generation.",price:"\
$0.02 / image"},{id:"grok-imagine-image-pro",implementedAt:"2026-02-01",implementedRank:530,name:"Gr\
ok Imagine Image Pro",desc:"Discontinued by xAI. Retained for chat history compatibility.",price:"$0\
.07 / image",deprecated:!0},{id:"grok-imagine-video-1.5",implementedAt:"2026-08-25",implementedRank:8501,
quickEmoji:"\u{1F3AC}",name:"Grok Imagine Video 1.5",desc:"Current xAI video generation model with 1\
080p text/image-to-video support.",price:"$0.080 / second"},{id:"grok-imagine-video",implementedAt:"\
2026-01-30",implementedRank:530,name:"Grok Imagine Video",desc:"Legacy Grok video generation.",price:"\
$0.05 / second"}]},{category:"xAI Grok",icon:"fas fa-rocket text-white",description:"Models by xAI",
items:[{id:"grok-4.6",implementedAt:"2026-08-19",implementedRank:8161,name:"Grok 4.6",desc:"Frontier\
 model for coding, agentic tasks, and knowledge work.",price:"In $2.00/1M, Out $6.00/1M"},{id:"grok-\
4.5",implementedAt:"2026-08-19",implementedRank:8160,name:"Grok 4.5",desc:"Intelligent coding model \
for agentic software and engineering tasks.",price:"In $2.00/1M, Out $6.00/1M"},{id:"grok-4.3",implementedAt:"\
2026-05-27",implementedRank:5530,name:"Grok 4.3",desc:"Most intelligent and fastest flagship model.",
price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-build-0.1",implementedAt:"2026-05-27",implementedRank:5520,
quickEmoji:"\u{1F6E0}\uFE0F",name:"Grok Build 0.1 (Coding)",desc:"Fast agentic coding model with vis\
ion and reasoning support.",price:"In $1.00/1M, Out $2.00/1M"},{id:"grok-4.20-0309-reasoning",implementedAt:"\
2026-08-25",implementedRank:8503,name:"Grok 4.20 (Reasoning, 0309)",desc:"Dated Grok 4.20 reasoning \
release.",price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-4.20-0309-non-reasoning",implementedAt:"2026-\
08-25",implementedRank:8504,name:"Grok 4.20 (Non-Reasoning, 0309)",desc:"Dated Grok 4.20 standard re\
lease.",price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-4.20-multi-agent-0309",implementedAt:"2026-08-2\
5",implementedRank:8505,name:"Grok 4.20 Multi-Agent (0309)",desc:"Dated Grok 4.20 multi-agent releas\
e.",price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-4.20-reasoning",implementedAt:"2026-04-09",implementedRank:4e3,
name:"Grok 4.20 (Reasoning)",desc:"Flagship reasoning model.",price:"In $1.25/1M, Out $2.50/1M"},{id:"\
grok-4.20-non-reasoning",implementedAt:"2026-04-09",implementedRank:4001,name:"Grok 4.20 (Non-Reason\
ing)",desc:"Flagship standard model.",price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-4.20-multi-agent",
implementedAt:"2026-04-09",implementedRank:4002,name:"Grok 4.20 Multi-Agent",desc:"Agentic flagship \
model.",price:"In $1.25/1M, Out $2.50/1M"},{id:"grok-4-1-fast-reasoning",implementedAt:"2026-03-01",
implementedRank:280,name:"Grok 4.1 Fast (Reasoning)",desc:"Fast with reasoning capabilities.",price:"\
In $0.20/1M, Out $0.50/1M",deprecated:!0},{id:"grok-4-1-fast-non-reasoning",implementedAt:"2026-03-0\
1",implementedRank:281,name:"Grok 4.1 Fast (Non-Reasoning)",desc:"Fast standard model.",price:"In $0\
.20/1M, Out $0.50/1M",deprecated:!0},{id:"grok-4-fast-reasoning",implementedAt:"2026-02-01",implementedRank:150,
name:"Grok 4 Fast (Reasoning)",desc:"Previous gen reasoning.",price:"In $0.20/1M, Out $0.50/1M",deprecated:!0},
{id:"grok-4-fast-non-reasoning",implementedAt:"2026-02-01",implementedRank:151,name:"Grok 4 Fast (No\
n-Reasoning)",desc:"Previous gen standard.",price:"In $0.20/1M, Out $0.50/1M",deprecated:!0}]}],WELCOME_QUICK_START_LIMIT=5,
listModelsFlat=o(()=>{const e=[];return MODELS.forEach(t=>{(t.items||[]).forEach(n=>{n&&n.id&&e.push(
n)})}),e},"listModelsFlat"),compareModelsByImplementedAt=o((e,t)=>{const n=String(e&&e.implementedAt||
""),i=String(t&&t.implementedAt||"");if(n!==i)return i.localeCompare(n);const a=Number(e&&e.implementedRank||
0),r=Number(t&&t.implementedRank||0);return a!==r?r-a:String(e&&e.id||"").localeCompare(String(t&&t.
id||""))},"compareModelsByImplementedAt"),getRecentModelsForQuickStart=o((e=WELCOME_QUICK_START_LIMIT)=>listModelsFlat().
filter(t=>t&&t.id&&!t.deprecated&&t.implementedAt).sort(compareModelsByImplementedAt).slice(0,Math.max(
0,Number(e)||0)),"getRecentModelsForQuickStart"),renderWelcomeQuickStart=o(()=>{const e=get("welcome\
-quick-start");if(!e)return;const t=getRecentModelsForQuickStart(WELCOME_QUICK_START_LIMIT);if(!t.length){
e.innerHTML="";return}e.innerHTML=t.map((n,i)=>{const a=(.1+i*.02).toFixed(2),r=n.quickEmoji?`${escapeHtml(
String(n.quickEmoji))} `:"",l=escapeHtml(String(n.name||n.id)),c=String(n.id).replace(/\\/g,"\\\\").
replace(/'/g,"\\'");return`<button type="button" class="welcome-btn p-3 rounded text-sm text-left tr\
ansition btn-hover slide-in-animate" style="animation-delay: ${a}s" onclick="quickStart('${c}')">${r}${l}\
</button>`}).join("")},"renderWelcomeQuickStart"),normalizeModelApiKeyMap=o(e=>{if(!e||typeof e!="ob\
ject")return{};const t={};return Object.entries(e).forEach(([n,i])=>{const a=String(n||"").trim(),r=String(
i||"").trim();!a||!r||(t[a]=r)}),t},"normalizeModelApiKeyMap"),MODEL_NAME_BY_ID=(()=>{const e=new Map;
return MODELS.forEach(t=>{(t.items||[]).forEach(n=>{const i=String(n.id||"").trim();!i||e.has(i)||e.
set(i,String(n.name||i))})}),e})(),getModelNameById=o(e=>{const t=String(e||"").trim();return t?MODEL_NAME_BY_ID.
get(t)||t:""},"getModelNameById"),maskApiKeyPreview=o(e=>{const t=String(e||"");return t?t.length<=8?
"********":`${t.slice(0,4)}...${t.slice(-4)}`:""},"maskApiKeyPreview"),getModelProviderInfo=o(e=>{const t=String(
e||"").toLowerCase().trim();return t?t.startsWith("gemini")||t.startsWith("veo-")||t.startsWith("lyr\
ia-")||t.startsWith("deep-research-")||t.startsWith("antigravity-")?{provider:"gemini",keyField:"gem\
ini_key",inputId:"set-gemini",label:"Gemini API Key"}:t.startsWith("gpt")||t.startsWith("o1")||t.startsWith(
"o3")?{provider:"openai",keyField:"openai_key",inputId:"set-openai",label:"OpenAI API Key"}:t.startsWith(
"deepseek")?{provider:"deepseek",keyField:"deepseek_key",inputId:"set-deepseek",label:"DeepSeek API \
Key"}:t.startsWith("kimi")?{provider:"kimi",keyField:"kimi_key",inputId:"set-kimi",label:"Kimi (Moon\
shot) API Key"}:t.startsWith("mistral")?{provider:"mistral",keyField:"mistral_key",inputId:"set-mist\
ral",label:"Mistral API Key"}:t.startsWith("claude")?{provider:"anthropic",keyField:"anthropic_key",
inputId:"set-anthropic",label:"Anthropic API Key"}:t.startsWith("grok")?{provider:"xai",keyField:"xa\
i_key",inputId:"set-xai",label:"xAI (Grok) API Key"}:t.startsWith("google")?{provider:"google",keyField:"\
google_key",inputId:"set-google-key",label:"Google API Key (TTS)"}:{provider:"openai",keyField:"open\
ai_key",inputId:"set-openai",label:"OpenAI API Key"}:null},"getModelProviderInfo"),setModelApiKeyPanelOpen=o(
e=>{const t=get("model-api-keys-panel"),n=get("toggle-model-api-keys-btn");if(!t||!n)return;const i=!!e;
t.classList.toggle("hidden",!i),n.innerText=i?"\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u8A2D\u5B9A\u3092\u9589\u3058\u308B":
"\u30E2\u30C7\u30EB\u5225\u306EAPI\u30AD\u30FC\u3092\u8A2D\u5B9A\u3059\u308B"},"setModelApiKeyPanelO\
pen"),syncModelApiKeyModelOptions=o(()=>{const e=get("model-api-key-model");if(!e)return;const t=e.value||
"";e.innerHTML="";const n=document.createElement("option");n.value="",n.textContent="\u30E2\u30C7\u30EB\u3092\u9078\u629E",
e.appendChild(n),MODELS.forEach(i=>{const a=Array.isArray(i.items)?i.items.filter(l=>!l.deprecated):
[];if(!a.length)return;const r=document.createElement("optgroup");r.label=String(i.category||"Models"),
a.forEach(l=>{const c=String(l.id||"").trim();if(!c)return;const u=document.createElement("option");
u.value=c,u.textContent=`${String(l.name||c)} (${c})`,r.appendChild(u)}),r.children.length>0&&e.appendChild(
r)}),t&&Array.from(e.options).some(a=>a.value===t)&&(e.value=t)},"syncModelApiKeyModelOptions"),renderModelApiKeyList=o(
()=>{const e=get("model-api-key-list");if(!e)return;modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap);
const t=Object.entries(modelApiKeyMap).sort((n,i)=>n[0].localeCompare(i[0]));if(e.innerHTML="",!t.length){
const n=document.createElement("div");n.className="text-[11px] text-gray-500",n.textContent="\u30E2\u30C7\u30EB\u5225\u30AD\u30FC\u306F\
\u672A\u8A2D\u5B9A\u3067\u3059\u3002",e.appendChild(n);return}t.forEach(([n,i])=>{const a=document.createElement(
"div");a.className="flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-9\
00/70 px-3 py-2";const r=document.createElement("div");r.className="min-w-0";const l=document.createElement(
"div");l.className="text-[11px] text-gray-200 truncate",l.textContent=`${getModelNameById(n)} (${n})`;
const c=document.createElement("div");c.className="text-[10px] text-cyan-300 font-mono",c.textContent=
maskApiKeyPreview(i),r.appendChild(l),r.appendChild(c);const u=document.createElement("button");u.type=
"button",u.className="text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-b\
old btn-hover shrink-0",u.textContent="\u524A\u9664",u.onclick=()=>{delete modelApiKeyMap[n],renderModelApiKeyList(),
showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u524A\u9664: ${n}`,"success")},a.appendChild(
r),a.appendChild(u),e.appendChild(a)})},"renderModelApiKeyList"),bindModelApiKeySettingsControls=o(()=>{
const e=get("toggle-model-api-keys-btn");e&&!e.dataset.bound&&(e.dataset.bound="1",e.addEventListener(
"click",()=>{const i=get("model-api-keys-panel");setModelApiKeyPanelOpen(i?i.classList.contains("hid\
den"):!0)}));const t=get("model-api-key-apply-btn");t&&!t.dataset.bound&&(t.dataset.bound="1",t.addEventListener(
"click",()=>{const i=get("model-api-key-model"),a=get("model-api-key-input"),r=i?String(i.value||"").
trim():"",l=a?String(a.value||"").trim():"";if(!r){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!l){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap),modelApiKeyMap[r]=l,a&&(a.
value=""),renderModelApiKeyList(),showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u8A2D\u5B9A: ${r}`,
"success")}));const n=get("model-api-key-input");n&&!n.dataset.bound&&(n.dataset.bound="1",n.addEventListener(
"keydown",i=>{if(i.key==="Enter"){i.preventDefault();const a=get("model-api-key-apply-btn");a&&a.click()}})),
syncModelApiKeyModelOptions(),renderModelApiKeyList(),setModelApiKeyPanelOpen(!1)},"bindModelApiKeyS\
ettingsControls");let activeModelTag="all";const MODEL_TAGS=["all","openai","gemini","anthropic","ki\
mi","deepseek","mistral","xai","image","video","audio","music","transcription","ocr","reasoning","fa\
st","agent","agentic view"],MINIMAL_SLASH_COMMANDS=[{id:"options",label:"/options",description:"\uFF0B\u30E1\u30CB\u30E5\
\u30FC\u3092\u958B\u304F",icon:"fa-plus",kind:"minimal",action:"options"},{id:"attach",label:"/attac\
h",description:"\u30D5\u30A1\u30A4\u30EB\u6DFB\u4ED8\u3092\u958B\u304F",icon:"fa-paperclip",kind:"mi\
nimal",itemKey:"attach"},{id:"voice",label:"/voice",description:"Voice Input\u3092\u958B\u59CB\u30FB\u505C\u6B62",
icon:"fa-microphone",kind:"minimal",itemKey:"voice-input"},{id:"paste",label:"/paste",description:"\u30EA\
\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u3092\u958B\u304F",icon:"fa-paste",kind:"minimal",itemKey:"rich\
-paste"},{id:"canvas",label:"/canvas",description:"Canvas\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-window-restore",kind:"minimal",itemKey:"canvas"},{id:"coding",label:"/coding",description:"\
Coding\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",icon:"fa-code-branch",kind:"minimal",
itemKey:"coding"},{id:"fast",label:"/fast",description:"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-bolt",kind:"minimal",itemKey:"fast"},{id:"search",label:"/search",description:"Search\u3092\u5207\u308A\u66FF\u3048\u308B\
\uFF08on / off\uFF09",icon:"fa-search",kind:"minimal",itemKey:"search"},{id:"urls",label:"/urls",description:"\
URLs\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",icon:"fa-link",kind:"minimal",itemKey:"\
urls"},{id:"maps",label:"/maps",description:"Maps\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-map-location-dot",kind:"minimal",itemKey:"maps"},{id:"python",label:"/python",description:"\
Python\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",icon:"fa-code",kind:"minimal",itemKey:"\
python"},{id:"file",label:"/file",description:"File\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-file-lines",kind:"minimal",itemKey:"file"},{id:"mcp",label:"/mcp",description:"MCP\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on\
 / off\uFF09",icon:"fa-plug",kind:"minimal",itemKey:"mcp"},{id:"sysprompt",label:"/sysprompt",description:"\
SysPrompt\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",icon:"fa-terminal",kind:"minimal",
itemKey:"sysprompt"},{id:"thinking",label:"/thinking",description:"Thinking\u306E\u5024\u3092\u9078\u629E\uFF08off / min / low / m\
id / high\uFF09",icon:"fa-brain",kind:"minimal",itemKey:"thinking",requiresArgument:!0,autocompleteArgument:!0},
{id:"thinking-off",label:"/thinking off",description:"Thinking\u3092OFF\u306B\u3059\u308B",icon:"fa-\
brain",kind:"minimal",itemKey:"thinking",presetArgument:"off"},{id:"thinking-min",label:"/thinking m\
in",description:"Thinking\u3092Min\u306B\u3059\u308B",icon:"fa-brain",kind:"minimal",itemKey:"thinki\
ng",presetArgument:"min"},{id:"thinking-low",label:"/thinking low",description:"Thinking\u3092Low\u306B\u3059\u308B",
icon:"fa-brain",kind:"minimal",itemKey:"thinking",presetArgument:"low"},{id:"thinking-mid",label:"/t\
hinking mid",description:"Thinking\u3092Mid\u306B\u3059\u308B",icon:"fa-brain",kind:"minimal",itemKey:"\
thinking",presetArgument:"mid"},{id:"thinking-high",label:"/thinking high",description:"Thinking\u3092Hig\
h\u306B\u3059\u308B",icon:"fa-brain",kind:"minimal",itemKey:"thinking",presetArgument:"high"},{id:"e\
ffort",label:"/effort",description:"Effort\u3092\u8ABF\u6574",icon:"fa-sliders-h",kind:"minimal",itemKey:"\
effort",requiresArgument:!0,argumentHint:"Effort\u3092\u5165\u529B\uFF08none / low / medium / high / xhigh / max\uFF09..."},
{id:"safety",label:"/safety",description:"Safety\u3092\u8ABF\u6574",icon:"fa-shield-halved",kind:"mi\
nimal",itemKey:"safety",requiresArgument:!0,argumentHint:"Safety\u3092\u5165\u529B\uFF08default / none\uFF09..."},
{id:"promptcache",label:"/promptcache",description:"PromptCache\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-database",kind:"minimal",itemKey:"promptcache"},{id:"compress",label:"/compress",description:"\
Compress\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",icon:"fa-compress-alt",kind:"minim\
al",itemKey:"compress"},{id:"tempchat",label:"/tempchat",description:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
icon:"fa-hourglass-half",kind:"minimal",itemKey:"tempchat"}],SLASH_COMMANDS=[{id:"settings",label:"/\
settings",description:"AI\u3067\u81EA\u7136\u8A00\u8A9E\u3092\u4F7F\u3063\u3066\u8A2D\u5B9A\u3092\u5909\u66F4\uFF08\u73FE\u5728\u9078\u629E\u4E2D\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\uFF09",
icon:"fa-cog",example:"\u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B"},
...MINIMAL_SLASH_COMMANDS];let slashSuggestionsVisible=!1,slashSelectedIndex=0,lastSlashFilter=null,
pendingSlashCommand=null;const AI_SETTINGS_CONVERSATION_KEY=`ai-settings-conversation:${typeof CHAT_CONFIG!=
"undefined"&&CHAT_CONFIG.currentUsername||"anonymous"}`;let aiSettingsConversation=[];function loadAiSettingsConversation(){
try{const e=sessionStorage.getItem(AI_SETTINGS_CONVERSATION_KEY),t=e?JSON.parse(e):[];return Array.isArray(
t)?t.filter(n=>n&&(n.role==="user"||n.role==="assistant")&&typeof n.content=="string").slice(-10).map(
n=>({role:n.role,content:n.content.slice(0,1600)})):[]}catch{return[]}}o(loadAiSettingsConversation,
"loadAiSettingsConversation");function persistAiSettingsConversation(){try{sessionStorage.setItem(AI_SETTINGS_CONVERSATION_KEY,
JSON.stringify(aiSettingsConversation.slice(-10)))}catch{}}o(persistAiSettingsConversation,"persistA\
iSettingsConversation");function clearAiSettingsConversation(){aiSettingsConversation=[];try{sessionStorage.
removeItem(AI_SETTINGS_CONVERSATION_KEY)}catch{}}o(clearAiSettingsConversation,"clearAiSettingsConve\
rsation");function appendAiSettingsConversation(e,t){const n=String(t||"").trim();n&&(aiSettingsConversation.
push({role:e,content:n.slice(0,1600)}),aiSettingsConversation=aiSettingsConversation.slice(-10),persistAiSettingsConversation())}
o(appendAiSettingsConversation,"appendAiSettingsConversation"),aiSettingsConversation=loadAiSettingsConversation();
function summarizeAiSettingsConversationValues(e,t){const n=Object.entries(e||{}),i=t==="inspect"?"\u73FE\
\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002":"\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002",
a=n.map(([r,l])=>`${r}: ${formatAiSettingValue(l).slice(0,180)}`).join(`
`);return`${i}${a?`
${a}`:""}`.slice(0,1600)}o(summarizeAiSettingsConversationValues,"summarizeAiSettingsConversationVal\
ues");let gemSuggestionsVisible=!1,gemSelectedIndex=0;const STS_MODELS=new Set(["gpt-transcribe","gp\
t-live-transcribe","gpt-realtime-2","gpt-realtime-translate","gpt-realtime-whisper","gpt-realtime-1.\
5","gpt-realtime","gpt-realtime-mini","gemini-2.5-flash-native-audio-preview-12-2025","gemini-3.1-fl\
ash-live-preview","gemini-3.8-live","gemini-3.8-live-extended-thinking","gemini-3.5-live-translate-p\
review","gemini-3.5-transcribe-live","grok-voice-think-fast-2.0","grok-voice-latest","grok-voice-thi\
nk-fast-1.0","grok-voice-fast-1.0","grok-voice-agent","grok-voice-transcribe-2.0"]),FILE_BASE_URL=CHAT_CONFIG.
urls.serveFileBase,FILE_THUMB_BASE_URL=CHAT_CONFIG.urls.serveFileThumbBase,RICH_PASTE_PDF_SERVER_ROUTE=CHAT_CONFIG.
urls.richPastePdfServer,IMAGE_EXTS=["png","jpg","jpeg","webp","gif","bmp","avif","heic","heif"],AUDIO_EXTS=[
"mp3","wav","aac","ogg","flac","aiff","aif","m4a","opus","oga","weba","webm"],VIDEO_EXTS=["mp4","mov",
"avi","mkv","m4v","webm","mpg","mpeg","wmv","3gp","3gpp","flv"],getFileExt=o(e=>{const t=typeof e=="\
string"?e:e==null?"":String(e);if(!t)return"";const n=t.lastIndexOf(".");return n===-1?"":t.slice(n+
1).toLowerCase()},"getFileExt"),normalizeAttachmentPath=o(e=>{if(!e)return"";let t="";if(typeof e=="\
string"?t=e:typeof e=="object"&&(t=String(e.path||e.url||e.name||e.filename||e.filepath||"")),!t)return"";
try{t.includes("://")&&(t=new URL(t,window.location.origin).pathname||"")}catch{}t.includes("?")&&(t=
t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=t.replace(/^\/+/,""),t.startsWith("files\
/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},"normalizeAttachmentPath"),isGeminiImageModelKey=o(
e=>{const t=(e||"").toLowerCase();return t.includes("gemini")&&(t.includes("image")||t.includes("nan\
o"))},"isGeminiImageModelKey"),isClaudeModelKey=o(e=>(e||"").toLowerCase().includes("claude"),"isCla\
udeModelKey"),getModelApiProvider=o(e=>{const t=String(e||"").toLowerCase().trim();return t?t.includes(
"claude")?"anthropic":t.includes("deepseek")?"deepseek":t.includes("grok")&&!t.includes("gpt")?"xai":
t.includes("google-tts")?"google":t.includes("gemini")||t.startsWith("veo-")||t.startsWith("lyria-")||
t.startsWith("deep-research-")||t.startsWith("antigravity-")?"gemini":"openai":null},"getModelApiPro\
vider"),PROVIDER_LABELS={openai:"OpenAI",gemini:"Gemini",anthropic:"Anthropic (Claude)",xai:"xAI (Gr\
ok)",deepseek:"DeepSeek",google:"Google Cloud"},isPromptCacheEnabled=o(()=>{const e=get("enable-prom\
pt-cache");return!!(e&&e.checked)},"isPromptCacheEnabled"),getPromptCacheLockedProvider=o(()=>{if(!isPromptCacheEnabled())
return null;const e=get("model-select");return getModelApiProvider(e?e.value:"")},"getPromptCacheLoc\
kedProvider"),updatePromptCacheUi=o(()=>{const e=get("prompt-cache-container"),t=get("enable-prompt-\
cache"),n=get("model-selector-btn");if(!t)return;const i=!!t.checked;e&&(e.classList.toggle("ring-1",
i),e.classList.toggle("ring-teal-500/50",i),e.classList.toggle("rounded",i),e.classList.toggle("px-1",
i)),n&&(i?(n.title="PromptCache\u6709\u52B9: \u540C\u4E00API\u30D7\u30ED\u30D0\u30A4\u30C0\u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\u629E\u53EF\u80FD",
n.classList.add("border-teal-500/60")):(n.title="",n.classList.remove("border-teal-500/60")))},"upda\
tePromptCacheUi"),bindPromptCacheControls=o(()=>{const e=get("enable-prompt-cache");!e||e.dataset.bound===
"1"||(e.dataset.bound="1",e.addEventListener("change",()=>{if(updatePromptCacheUi(),e.checked){const t=getModelApiProvider(
get("model-select")?get("model-select").value:""),n=PROVIDER_LABELS[t]||t||"\u73FE\u5728\u306EAPI";showToast(
`PromptCache \u3092\u6709\u52B9\u5316\u3057\u307E\u3057\u305F\u3002\u4EE5\u964D\u306F ${n} \u4EE5\u5916\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002`,"info",!0)}}))},"bindPromptCacheControls"),getModelMediaSupport=o(
e=>{const t=(e||"").toLowerCase();return t.includes("gemini")?t.includes("image")||t.includes("nano")||
t.includes("tts")||t.includes("native-audio")||t.includes("live")?{audio:!1,video:!1}:t.includes("em\
bedding")||t.startsWith("veo-")||t.includes("omni-flash")||t.includes("omni-1.1-flash")||t.startsWith(
"lyria-")?{audio:!1,video:!1}:{audio:!0,video:!0}:{audio:!1,video:!1}},"getModelMediaSupport"),supportsAudioInputModel=o(
()=>getModelMediaSupport(get("model-select").value).audio,"supportsAudioInputModel"),supportsVideoInputModel=o(
()=>getModelMediaSupport(get("model-select").value).video,"supportsVideoInputModel"),isImagePath=o(e=>IMAGE_EXTS.
includes(getFileExt(e||"")),"isImagePath"),isAudioPath=o(e=>AUDIO_EXTS.includes(getFileExt(e||"")),"\
isAudioPath"),isVideoPath=o(e=>VIDEO_EXTS.includes(getFileExt(e||"")),"isVideoPath"),OPENAI_TTS_VOICES=[
"alloy","ash","ballad","coral","echo","fable","nova","onyx","sage","shimmer","verse","marin","cedar"],
GEMINI_TTS_VOICES=["Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Auto\
noe","Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia",
"Achernar","Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadac\
hbia","Sadaltager","Sulafat"],OPENAI_STS_VOICES=["alloy","ash","ballad","coral","echo","sage","shimm\
er","verse","marin","cedar"],GROK_STS_VOICES=["Ara","Rex","Sal","Eve","Leo"],GROK_TTS_VOICES=["Eve",
"Ara","Rex","Sal","Leo"],GEMINI_STS_VOICES=["Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","\
Aoede","Callirrhoe","Autonoe","Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib",
"Rasalgethi","Laomedeia","Achernar","Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenu\
bi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"],GROK_PCM_RATES=[8e3,16e3,21050,24e3,32e3,44100,
48e3],isTtsModel=o(()=>get("model-select").value.includes("tts"),"isTtsModel"),isGptImageModel=o(()=>(get(
"model-select").value||"").includes("gpt-image"),"isGptImageModel"),isGeminiImageModel=o(()=>isGeminiImageModelKey(
get("model-select").value),"isGeminiImageModel"),isMistralOcrModel=o(e=>{const t=String(e!=null?e:get(
"model-select")&&get("model-select").value||"").toLowerCase();return t==="mistral-ocr-4-0"||t==="mis\
tral-ocr-latest"||t.startsWith("mistral-ocr")},"isMistralOcrModel"),isLlmModel=o(()=>{const e=(get("\
model-select").value||"").toLowerCase();return isMistralOcrModel(e)||e.includes("tts")||e.includes("\
transcribe")||e.includes("realtime")||e.includes("voice-agent")||e.includes("native-audio")||e.includes(
"live")||e.includes("image")||e.includes("video")||isGeminiVideoModelKey(e)||isGeminiMusicModelKey(e)||
isGeminiEmbeddingModelKey(e)||e.includes("gemini")&&(e.includes("image")||e.includes("nano"))?!1:e.includes(
"gpt")||e.includes("gemini")||e.includes("grok")||e.includes("deepseek")||e.startsWith("deep-researc\
h-")||e.startsWith("antigravity-")},"isLlmModel"),isGrokImageModel=o(()=>{const e=(get("model-select").
value||"").toLowerCase();return e.includes("grok")&&(e.includes("imagine")||e.includes("image"))&&!e.
includes("video")},"isGrokImageModel"),isGrokVideoModel=o(()=>{const e=(get("model-select").value||"").
toLowerCase();return e.includes("grok")&&e.includes("video")},"isGrokVideoModel"),isGeminiVideoModelKey=o(
e=>{const t=(e||"").toLowerCase();return t.startsWith("veo-")||t.includes("omni-flash")||t.includes(
"omni-1.1-flash")},"isGeminiVideoModelKey"),isGeminiVideoModel=o(()=>isGeminiVideoModelKey(get("mode\
l-select").value),"isGeminiVideoModel"),isGeminiMusicModelKey=o(e=>(e||"").toLowerCase().startsWith(
"lyria-"),"isGeminiMusicModelKey"),isGeminiMusicModel=o(()=>isGeminiMusicModelKey(get("model-select").
value),"isGeminiMusicModel"),isGeminiEmbeddingModelKey=o(e=>(e||"").toLowerCase().includes("gemini-e\
mbedding"),"isGeminiEmbeddingModelKey"),isGeminiEmbeddingModel=o(()=>isGeminiEmbeddingModelKey(get("\
model-select").value),"isGeminiEmbeddingModel"),isStsModel=o(()=>STS_MODELS.has(get("model-select").
value),"isStsModel"),isTranscriptionModel=o(()=>{const e=get("model-select")?get("model-select").value:
"";return e==="gpt-transcribe"||e==="gpt-live-transcribe"},"isTranscriptionModel"),isGeminiLiveModel=o(
()=>{const e=get("model-select").value;return e==="gemini-3.1-flash-live-preview"||e==="gemini-3.8-l\
ive"||e==="gemini-3.8-live-extended-thinking"||e==="gemini-3.5-live-translate-preview"||e==="gemini-\
3.5-transcribe-live"},"isGeminiLiveModel"),isGeminiLiveExtendedThinkingModel=o(()=>get("model-select").
value==="gemini-3.8-live-extended-thinking","isGeminiLiveExtendedThinkingModel"),isGeminiLiveTranslateModel=o(
()=>get("model-select").value==="gemini-3.5-live-translate-preview","isGeminiLiveTranslateModel"),isGeminiLiveTranscribeModel=o(
()=>get("model-select").value==="gemini-3.5-transcribe-live","isGeminiLiveTranscribeModel"),isXaiLiveTranscribeModel=o(
()=>!!get("model-select")&&get("model-select").value==="grok-voice-transcribe-2.0","isXaiLiveTranscr\
ibeModel"),isGeminiRealtimeMusicModel=o(()=>(get("model-select").value||"")==="lyria-realtime-exp","\
isGeminiRealtimeMusicModel"),isLyriaRealtimeModel=o(()=>isGeminiRealtimeMusicModel(),"isLyriaRealtim\
eModel"),isRealtimeSessionModel=o(()=>!(!isStsModel()||isGeminiLiveModel()||isTranscriptionModel()||
get("model-select")&&get("model-select").value==="gpt-realtime-whisper"),"isRealtimeSessionModel"),getStsProvider=o(
e=>{const t=(e||"").toLowerCase();return t.includes("gpt-realtime")||t==="gpt-transcribe"||t==="gpt-\
live-transcribe"?"openai":t.includes("grok-voice")?"xai":t.includes("gemini")&&(t.includes("native-a\
udio")||t.includes("live"))?"gemini":null},"getStsProvider");function setStsStatus(e,t=!1){const n=get(
"sts-status"),i=get("sts-mic-btn");n&&e&&(n.innerText=e),i&&(t?(i.classList.add("bg-red-600","animat\
e-pulse"),i.classList.remove("bg-cyan-600")):(i.classList.remove("bg-red-600","animate-pulse"),i.classList.
add("bg-cyan-600")))}o(setStsStatus,"setStsStatus");function updateStsUi(){const e=isStsModel(),t=e&&
voiceStudioUiEnabled!==!1,n=get("input-row"),i=get("sts-panel"),a=get("file-preview");e?(n&&n.classList.
add("hidden"),a&&a.classList.add("hidden"),i&&(i.classList.remove("hidden"),i.classList.toggle("voic\
e-dock",t)),!t&&window.VoiceStudio&&window.VoiceStudio.closeIfOpen(),window.VoiceStudio&&window.VoiceStudio.
syncDock(),setStsStatus("Tap to speak",!1)):(n&&n.classList.remove("hidden"),i&&i.classList.add("hid\
den"),window.VoiceStudio&&window.VoiceStudio.closeIfOpen())}o(updateStsUi,"updateStsUi");function updateStsOptions(){
if(!isStsModel())return;const e=get("model-select").value||"",t=getStsProvider(e),n=get("sts-voice"),
i=get("sts-speed-wrap"),a=get("sts-speed"),r=get("sts-speed-label"),l=get("sts-rate-wrap"),c=get("st\
s-rate-in"),u=get("sts-rate-out"),f=get("sts-thinking-wrap"),g=get("sts-note"),y=get("sts-voice-wrap"),
w=get("sts-auto-play-wrap"),v=get("sts-mode-label"),k=isTranscriptionModel()||isGeminiLiveTranscribeModel()||
isXaiLiveTranscribeModel(),_=get("sts-lang-wrap");if(k){v&&(v.textContent="Realtime Speech-to-Text"),
y&&y.classList.add("hidden"),w&&w.classList.add("hidden"),i&&i.classList.add("hidden"),l&&l.classList.
add("hidden"),f&&f.classList.add("hidden"),_&&_.classList.add("hidden");const C=get("sts-transcribe-\
wrap"),L=get("sts-custom-vocab-wrap");C&&C.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),
L&&L.classList.toggle("hidden",!isGeminiLiveTranscribeModel()&&!isXaiLiveTranscribeModel()),g&&(g.textContent=
isGeminiLiveTranscribeModel()?"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F4E\u9045\u5EF6\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM / \u6700\u592710\u5206\uFF09":
isXaiLiveTranscribeModel()?"xAI \u30B9\u30C8\u30EA\u30FC\u30DF\u30F3\u30B0\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM\uFF09":
e==="gpt-live-transcribe"?"\u4F4E\u9045\u5EF6\u30E9\u30A4\u30D6\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
"\u9AD8\u7CBE\u5EA6\u306A\u30B3\u30DF\u30C3\u30C8\u5358\u4F4D\u306E\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09")}else if(t===
"openai")v&&(v.textContent="Speech-to-Speech Live"),y&&y.classList.remove("hidden"),w&&w.classList.remove(
"hidden"),setSelectOptions(n,OPENAI_STS_VOICES,n.value||"alloy"),i&&i.classList.remove("hidden"),a&&
(a.min=.25,a.max=1.5,a.step=.05,a.value||(a.value=1),Number(a.value)<.25&&(a.value=.25),Number(a.value)>
1.5&&(a.value=1.5)),l&&l.classList.add("hidden"),f&&f.classList.add("hidden"),_&&_.classList.add("hi\
dden"),g&&(g.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A");else if(t==="xai")v&&(v.textContent=
"Speech-to-Speech Live"),y&&y.classList.remove("hidden"),w&&w.classList.remove("hidden"),setSelectOptions(
n,GROK_STS_VOICES,n.value||"Ara"),i&&i.classList.add("hidden"),l&&l.classList.remove("hidden"),f&&f.
classList.add("hidden"),_&&_.classList.add("hidden"),setSelectOptions(c,GROK_PCM_RATES,Number(c.value||
24e3)),setSelectOptions(u,GROK_PCM_RATES,Number(u.value||24e3)),g&&(g.textContent="xAI\u306FPCM\u30B5\u30F3\u30D7\u30EB\u30EC\u30FC\u30C8\u5909\u66F4\u53EF");else if(t===
"gemini"){if(v&&(v.textContent="Speech-to-Speech Live"),y&&y.classList.remove("hidden"),w&&w.classList.
remove("hidden"),setSelectOptions(n,GEMINI_STS_VOICES,n.value||"Kore"),i&&i.classList.add("hidden"),
l&&l.classList.add("hidden"),f&&f.classList.remove("hidden"),_&&_.classList.add("hidden"),g&&(g.textContent=
"Gemini Live\u306F\u97F3\u58F0\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC"),e==="gemini-3.8-live")f&&
f.classList.add("hidden"),g&&(g.textContent="Gemini 3.8 Flash Live\u306F\u56FA\u5B9A\u30EC\u30A4\u30C6\u30F3\u30B7\u306ELive API\u30E2\u30C7\u30EB\uFF08Thinking leve\
l\u975E\u5BFE\u5FDC\uFF09");else if(e==="gemini-3.8-live-extended-thinking"){g&&(g.textContent="Gemi\
ni 3.8 Live Extended Thinking\u306Flow / medium / high\u306E\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u63A8\u8AD6\u306B\u5BFE\u5FDC");
const C=get("sts-thinking-level");Array.from(C&&C.options||[]).forEach(L=>{L.disabled=L.value==="min\
imal"}),C&&!["low","medium","high"].includes(C.value)&&(C.value="medium")}e==="gemini-3.5-live-trans\
late-preview"&&(v&&(v.textContent="Realtime Translation"),f&&f.classList.add("hidden"),y&&y.classList.
add("hidden"),_&&_.classList.remove("hidden"),g&&(g.textContent="70\u4EE5\u4E0A\u306E\u8A00\u8A9E\u306B\u5BFE\u5FDC\u3059\u308B\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\uFF08Think\u975E\u5BFE\u5FDC\u30FB\u97F3\u58F0\u9078\
\u629E\u4E0D\u53EF\uFF09"))}i&&r&&a&&!i.classList.contains("hidden")&&(r.textContent=`${Number(a.value||
1).toFixed(2)}x`)}o(updateStsOptions,"updateStsOptions");function stsOpt(e){const t=get(e);return e===
"sts-auto-play"||e==="sts-auto-restart"?t?!!t.checked:!0:t?!!t.checked:!1}o(stsOpt,"stsOpt");function getStsSilenceMs(){
const e=get("sts-silence-sec");let t=e?parseFloat(e.value):1.5;return(isNaN(t)||t<.5)&&(t=.5),t>10&&
(t=10),Math.round(t*1e3)}o(getStsSilenceMs,"getStsSilenceMs");function getTtsProvider(e){if(!e)return null;
const t=e.toLowerCase();return t.includes("google-tts")?"google":t.includes("gemini")&&t.includes("t\
ts")?"gemini":t.includes("grok-tts")||t.includes("xai-tts")?"xai":t.includes("tts")?"openai":null}o(
getTtsProvider,"getTtsProvider");function setSelectOptions(e,t,n){e&&(e.innerHTML="",t.forEach(i=>{const a=document.
createElement("option");a.value=i.value||i,a.textContent=i.label||i,(i.value||i)===n&&(a.selected=!0),
e.appendChild(a)}))}o(setSelectOptions,"setSelectOptions");function updateTtsUi(){const e=get("model\
-select").value||"",t=getTtsProvider(e),n=get("audio-gen-options");if(!n)return;if(!t){n.classList.add(
"hidden");return}n.classList.remove("hidden");const i=get("tts-voice"),a=get("tts-voice-custom-wrap"),
r=get("tts-voice-custom"),l=get("tts-language-wrap"),c=get("tts-language"),u=get("tts-speed-wrap"),f=get(
"tts-speed"),g=get("tts-speed-label"),y=get("tts-speed-note");t==="openai"?(setSelectOptions(i,OPENAI_TTS_VOICES,
i.value||"alloy"),a.classList.add("hidden"),l.classList.add("hidden"),f&&(f.min=.25,f.max=4,f.step=.05,
f.value||(f.value=1),Number(f.value)<.25&&(f.value=.25),Number(f.value)>4&&(f.value=4),f.disabled=!1),
y&&(y.textContent="")):t==="gemini"?(setSelectOptions(i,GEMINI_TTS_VOICES,i.value||"Kore"),a.classList.
add("hidden"),l.classList.add("hidden"),f&&(f.disabled=!0),y&&(y.textContent="(Gemini TTS\u306F\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC)")):
t==="google"?(setSelectOptions(i,[{value:"auto",label:"Auto (Studio/Neural2)"},{value:"custom",label:"\
Custom Voice Name"}],i.value||"auto"),i.value==="custom"?a.classList.remove("hidden"):(a.classList.add(
"hidden"),r&&(r.value="")),l.classList.remove("hidden"),c&&!c.value&&(c.value="ja-JP"),f&&(f.min=.25,
f.max=2,f.step=.05,f.value||(f.value=1),Number(f.value)<.25&&(f.value=.25),Number(f.value)>2&&(f.value=
2),f.disabled=!1),y&&(y.textContent="")):t==="xai"&&(setSelectOptions(i,GROK_TTS_VOICES,i.value||"Ev\
e"),a.classList.remove("hidden"),l.classList.remove("hidden"),c&&!c.value&&(c.value="ja"),f&&(f.min=
.7,f.max=1.5,f.step=.05,f.value||(f.value=1),Number(f.value)<.7&&(f.value=.7),Number(f.value)>1.5&&(f.
value=1.5),f.disabled=!1),y&&(y.textContent="xAI TTS supports speed 0.7\u20131.5 and speech tags")),
f&&g&&(g.textContent=`${Number(f.value||1).toFixed(2)}x`)}o(updateTtsUi,"updateTtsUi");let mcpServers=[],
mcpLoaded=!1,mcpLoadPromise=null,mcpOauthPopups=[];const MCP_URLS={servers:o(()=>"/api/mcp/servers",
"servers"),server:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}`,"server"),test:o(e=>`/api/mcp/ser\
vers/${encodeURIComponent(e)}/test`,"test"),authStart:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}\
/auth/start`,"authStart"),authDisconnect:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}/auth/discon\
nect`,"authDisconnect"),tools:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}/tools`,"tools"),oauthClient:o(
()=>"/api/mcp/oauth-client","oauthClient"),permission:o((e,t)=>`/api/mcp/servers/${encodeURIComponent(
e)}/tools/${encodeURIComponent(t)}/permission`,"permission")},mcpGoogleProviderKey="google_workspace",
mcpEsc=o(e=>String(e==null?"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quo\
t;","'":"&#39;"})[t]),"mcpEsc"),mcpStatusMsg=o((e,t,n)=>{const i=get(e);i&&(i.textContent=t||"",i.style.
color=n?"#f87171":"#9ca3af")},"mcpStatusMsg");function mcpAuthStatusLabel(e){return e.auth_type==="n\
one"?"\u8A8D\u8A3C\u4E0D\u8981":e.auth_status==="connected"?"\u63A5\u7D9A\u6E08\u307F":e.auth_status===
"expired"?"\u671F\u9650\u5207\u308C\uFF08\u518D\u8A8D\u8A3C\uFF09":e.auth_status==="needs_auth"?"\u8A8D\u8A3C\u304C\
\u5FC5\u8981":"\u672A\u8A8D\u8A3C"}o(mcpAuthStatusLabel,"mcpAuthStatusLabel");function mcpConnectionStateLabel(e){
return e.connection_state==="error"?"\u30A8\u30E9\u30FC":e.connection_state==="connected"?"\u63A5\u7D9AOK":
e.connection_state==="needs_auth"?"\u8A8D\u8A3C\u5F85\u3061":"\u672A\u63A5\u7D9A"}o(mcpConnectionStateLabel,
"mcpConnectionStateLabel");function mcpBadgeClass(e){return e==="ok"||e==="connected"?"bg-emerald-70\
0/60 text-emerald-100":e==="error"||e==="expired"?"bg-red-700/60 text-red-100":e==="auth"?"bg-amber-\
600/50 text-amber-100":"bg-gray-700 text-gray-300"}o(mcpBadgeClass,"mcpBadgeClass");function mcpStateBadge(e){
const t=mcpAuthStatusLabel(e),n=e.auth_status==="connected"?"ok":e.auth_status==="expired"?"expired":
e.auth_status==="needs_auth"?"auth":"neutral";return`<span class="text-[9px] font-bold px-2 py-0.5 r\
ounded-full ${mcpBadgeClass(n)}">${mcpEsc(t)}</span>`}o(mcpStateBadge,"mcpStateBadge");function mcpOauthProviderLabel(e){
return e==="google_workspace"?"Google Workspace":e||"OAuth"}o(mcpOauthProviderLabel,"mcpOauthProvide\
rLabel");async function loadMcpServers(e){if(!get("mcp-server-list")||mcpLoadPromise&&(await mcpLoadPromise,
!e))return;if(!e&&mcpLoaded){renderMcpServers();return}mcpStatusMsg("mcp-status-msg","\u8AAD\u307F\u8FBC\u307F\u4E2D...",
!1);let n;n=(async()=>{try{const i=await apiFetch(MCP_URLS.servers());if(!i.ok){const r=await i.json().
catch(()=>({}));mcpStatusMsg("mcp-status-msg",r.error||"MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const a=await i.json();mcpServers=a&&Array.isArray(a.servers)?a.servers:[],mcpLoaded=!0,renderMcpServers(),
applyMcpPromptChipUi()}catch(i){mcpStatusMsg("mcp-status-msg","MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(i&&i.message?i.message:i),!0)}finally{mcpLoadPromise===n&&(mcpLoadPromise=null)}})(),mcpLoadPromise=
n,await n}o(loadMcpServers,"loadMcpServers");function mcpHasEnabledServer(){return(mcpServers||[]).some(
e=>!!e.enabled)}o(mcpHasEnabledServer,"mcpHasEnabledServer");function isMcpEnabledForSend(){const e=get(
"mcp-container");if(!e||e.classList.contains("hidden"))return!1;const t=get("enable-mcp");return!!t&&
t.checked}o(isMcpEnabledForSend,"isMcpEnabledForSend");function mcpModelSupported(){try{const e=String(
get("model-select")&&get("model-select").value||"").toLowerCase();return e?!!(e.includes("claude")||
e.startsWith("kimi")||typeof isLlmModel=="function"&&isLlmModel()):!1}catch{return!1}}o(mcpModelSupported,
"mcpModelSupported");function applyMcpPromptChipUi(){const e=get("mcp-container");if(!e)return;const t=mcpModelSupported()&&
mcpHasEnabledServer();if(e.classList.toggle("hidden",!t),syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}}o(applyMcpPromptChipUi,"applyMcpPromptChipUi");
function syncMcpAutoSysRows(){["set","thread"].forEach(e=>{const t=get(`${e}-auto-sys-mcp-enabled`);
t&&(t.disabled=!0,t.checked=isMcpEnabledForSend())})}o(syncMcpAutoSysRows,"syncMcpAutoSysRows");function renderMcpServers(){
const e=get("mcp-server-list"),t=get("mcp-server-count");if(!e)return;const n=mcpServers.length;if(t&&
(t.textContent=`${n}\u4EF6`),!n){e.innerHTML='<div class="text-[11px] text-gray-600 py-2">\u307E\u3060\u30B5\u30FC\u30D0\u30FC\u304C\u3042\u308A\u307E\
\u305B\u3093\u3002\u4E0A\u306E\u30AB\u30B9\u30BF\u30E0\u8FFD\u52A0\u30D5\u30A9\u30FC\u30E0\u304B\u3089\u767B\u9332\u3059\u308B\u304B\u3001Google Workspace \u306E\u8A8D\u8A3C\u3092\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>',
mcpStatusMsg("mcp-status-msg","");return}const i=mcpServers.map((a,r)=>mcpServerCard(a,r)).join("");
e.innerHTML=i,mcpStatusMsg("mcp-status-msg","")}o(renderMcpServers,"renderMcpServers");function mcpServerCard(e,t){
const n=!!e.is_preset,i=e.auth_type==="oauth",a=e.auth_type==="bearer",r=i||a,l=i&&!e.oauth_client_registered,
c=Number(e.tool_count||0),u=c>0?`${c}\u30C4\u30FC\u30EB`:"\u30C4\u30FC\u30EB\u672A\u53D6\u5F97",f=mcpStateBadge(
e),g=n?'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">\u30D7\u30EA\u30BB\u30C3\u30C8<\
/span>':'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">\u30AB\
\u30B9\u30BF\u30E0</span>',y=mcpAuthBlock(e),w=i?mcpOauthClientBlock(e):"";return`
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(e.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${e.enabled?"text-cyan-300":"text-gray-600"}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(e.name)}</span>
                ${g} ${f}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${r?mcpAuthActionButton(e):""}
            ${a&&e.auth_status!=="connected",""}
            ${n?"":`<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-da\
nger-btn" data-act="delete" data-id="${e.id}">\u524A\u9664</button>`}
            <label class="relative inline-flex items-center cursor-pointer ml-1" title="${e.enabled?
"\u7121\u52B9\u5316":"\u6709\u52B9\u5316"}">
                <input type="checkbox" class="sr-only peer mcp-enable-toggle" data-id="${e.id}" ${e.
enabled?"checked":""}>
                <div class="w-9 h-5 bg-gray-700 peer-focus:outline-none rounded-full peer-checked:af\
ter:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-whi\
te after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[var(--theme-600)]"><\
/div>
            </label>
        </div>
    </div>
    <div class="text-[10px] text-gray-500 mt-1 break-all">${mcpEsc(e.url)}</div>
    ${e.description?`<div class="text-[10px] text-gray-500 mt-0.5">${mcpEsc(e.description)}</div>`:""}\

    ${e.last_error?`<div class="text-[10px] text-red-400 mt-1">${mcpEsc(e.last_error)}</div>`:""}
    <div class="flex items-center justify-between gap-2 mt-2 flex-wrap">
        <div class="text-[10px] text-gray-500 flex items-center gap-2">
            <span class="${c>0?"text-emerald-300":"text-gray-500"}">${u}</span>
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="too\
ls" data-id="${e.id}">\u30C4\u30FC\u30EB\u4E00\u89A7</button>
        </div>
        <div class="flex items-center gap-1 flex-wrap">
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="tes\
t" data-id="${e.id}"><i class="fas fa-plug"></i> \u63A5\u7D9A\u30C6\u30B9\u30C8</button>
            <span class="text-[9px] text-gray-600">${mcpEsc(mcpConnectionStateLabel(e))}</span>
        </div>
    </div>
    ${c>0?`<div class="hidden mt-2" data-mcp-toolbox="${e.id}"></div>`:`<div class="hidden mt-2" dat\
a-mcp-toolbox="${e.id}"><div class="text-[10px] text-gray-600">\u63A5\u7D9A\u30C6\u30B9\u30C8\u5F8C\u306B\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u8868\u793A\u3055\u308C\u307E\u3059\u3002</div></div>`}\

    ${w}
    ${y}
</div>`}o(mcpServerCard,"mcpServerCard");function mcpAuthActionButton(e){return e.auth_type==="beare\
r"?"":e.auth_status==="connected"||e.auth_status==="expired"?`<button type="button" data-progress-no\
-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="reconnect" data-id="${e.id}"><i class="f\
as fa-sync"></i> \u518D\u8A8D\u8A3C</button>
                        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mc\
p-danger-btn" data-act="disconnect" data-id="${e.id}"><i class="fas fa-unlink"></i> \u63A5\u7D9A\u89E3\u9664</button>`:
`<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="a\
uth" data-id="${e.id}"><i class="fas fa-key"></i> \u8A8D\u8A3C\u3059\u308B</button>`}o(mcpAuthActionButton,
"mcpAuthActionButton");function mcpOauthClientBlock(e){const t=e.oauth_provider_key||e.slug||"",n=mcpOauthProviderLabel(
t);return e.oauth_client_registered?`
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 flex items-center justify-between">
        <span>OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\uFF08${mcpEsc(n)}\uFF09: ${mcpEsc(e.oauth_client_id_masked||
"\u767B\u9332\u6E08\u307F")}</span>
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="edit-oa\
uth" data-id="${e.id}">\u5909\u66F4</button>
    </div>
</div>`:`
<div class="mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2">
    <div class="text-[10px] text-amber-300 mb-1">${mcpEsc(n)} \u306E OAuth \u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\uFF08Client ID / Secret\uFF09\u304C\u5FC5\
\u8981\u3067\u3059\u3002</div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" data-oauth-pk="${mcpEsc(t)}" data-oauth-role="cid" placeholder="Client ID\
" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded \
px-2 py-1 text-xs text-white">
        <input type="password" data-oauth-pk="${mcpEsc(t)}" data-oauth-role="secret" placeholder="Cl\
ient Secret" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-7\
00 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oa\
uth" data-id="${e.id}" data-pk="${mcpEsc(t)}">\u4FDD\u5B58</button>
    </div>
</div>`}o(mcpOauthClientBlock,"mcpOauthClientBlock");function mcpAuthBlock(e){if(e.auth_type==="bear\
er")return`
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 mb-1">Bearer \u30C8\u30FC\u30AF\u30F3 ${!!e.auth_has_token?
'<span class="text-emerald-300">\uFF08\u4FDD\u5B58\u6E08\u307F\u30FB********\uFF09</span>':'<span cl\
ass="text-amber-300">\uFF08\u672A\u8A2D\u5B9A\uFF09</span>'}</div>
    <div class="flex gap-1">
        <input type="password" data-bearer-id="${e.id}" placeholder="Bearer \u30C8\u30FC\u30AF\u30F3" autocomplete="off"\
 data-1p-ignore="true" class="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs te\
xt-white">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data\
-act="save-bearer" data-id="${e.id}">\u4FDD\u5B58</button>
    </div>
</div>`;if(e.auth_type==="oauth"){const t=e.oauth_provider_key||e.slug||"";return`
<div class="text-[10px] text-gray-600 mt-1">${!e.oauth_client_registered?"OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3059\u308B\u3068\u300C\u8A8D\u8A3C\u3059\u308B\u300D\u304C\
\u4F7F\u3048\u307E\u3059\u3002":""}</div>`}return""}o(mcpAuthBlock,"mcpAuthBlock");async function mcpToggleEnabled(e,t){
mcpStatusMsg("mcp-status-msg",t?"\u6709\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...":"\u7121\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...",
!1);try{const n=await apiFetch(MCP_URLS.server(e),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({enabled:t})});if(!n.ok){const a=await n.json().catch(()=>({}));mcpStatusMsg(
"mcp-status-msg",a.error||"\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}const i=await n.
json();mcpStatusMsg("mcp-status-msg",t?"\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u30C1\u30E3\u30C3\u30C8\u306E\u30E2\u30C7\u30EB\u3078\u30C4\u30FC\u30EB\u304C\u516C\u958B\u3055\u308C\u307E\u3059\u3002":
"\u7121\u52B9\u306B\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(n){mcpStatusMsg("mcp\
-status-msg","\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(n&&n.message?n.message:n),!0)}}
o(mcpToggleEnabled,"mcpToggleEnabled");async function mcpOpenAuth(e){mcpStatusMsg("mcp-status-msg","\
\u8A8D\u53EFURL\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059...",!1);try{const t=await apiFetch(MCP_URLS.
authStart(e),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});if(!t.
ok){const a=await t.json().catch(()=>({}));a.requires_oauth_client?mcpStatusMsg("mcp-status-msg",a.error||
"OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u5148\u306B\u767B\u9332\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
!0):mcpStatusMsg("mcp-status-msg",a.error||"\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const n=await t.json();if(!n.url){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u304C\u8FD4\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
!0);return}const i=window.open(n.url,"_blank","width=520,height=680");if(i){mcpOauthPopups.push(i),mcpStatusMsg(
"mcp-status-msg","Google\u306E\u753B\u9762\u3067\u8A31\u53EF\u3057\u3066\u304F\u3060\u3055\u3044\u3002\u5B8C\u4E86\u5F8C\u3053\u306E\u30BF\u30D6\u306B\u53CD\u6620\u3055\u308C\u307E\u3059\u3002",
!1);const a=window.setInterval(()=>{(!i||i.closed)&&(window.clearInterval(a),loadMcpServers(!0))},1200)}else
mcpStatusMsg("mcp-status-msg","\u30DD\u30C3\u30D7\u30A2\u30C3\u30D7\u304C\u30D6\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F\u3002",
!0)}catch(t){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}o(mcpOpenAuth,"mcpOpenAuth");async function mcpDisconnect(e){if(window.
confirm("\u3053\u306E\u30B5\u30FC\u30D0\u30FC\u306E\u8A8D\u8A3C\u60C5\u5831\uFF08\u30C8\u30FC\u30AF\u30F3\uFF09\u3092\u524A\u9664\u3057\u3066\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const t=await apiFetch(MCP_URLS.authDisconnect(e),{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:"{}"});if(!t.ok){const n=await t.json().catch(()=>({}));mcpStatusMsg("mcp-status-m\
sg",n.error||"\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg(
"mcp-status-msg","\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(t){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}o(mcpDisconnect,"mcpDisconnect");async function mcpDeleteServer(e){if(window.
confirm("\u3053\u306E\u30AB\u30B9\u30BF\u30E0MCP\u30B5\u30FC\u30D0\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const t=await apiFetch(MCP_URLS.server(e),{method:"DELETE"});if(!t.ok){const n=await t.json().catch(
()=>({}));mcpStatusMsg("mcp-status-msg",n.error||"\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","\u524A\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(
!0)}catch(t){mcpStatusMsg("mcp-status-msg","\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}o(mcpDeleteServer,"mcpDeleteServer");async function mcpTestServer(e,t){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",!1);try{const n=await apiFetch(
MCP_URLS.test(e),{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}),i=await n.json().
catch(()=>({}));if(!n.ok){mcpStatusMsg("mcp-status-msg",i.error||"\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}i.probe&&i.probe.message&&mcpStatusMsg("mcp-status-msg",i.probe.message,!i.probe.ok),loadMcpServers(
!0)}catch(n){mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(n&&n.message?n.message:n),!0)}}o(mcpTestServer,"mcpTestServer");async function mcpLoadTools(e){const t=document.
querySelector(`[data-mcp-toolbox="${e}"]`);if(t){t.classList.remove("hidden"),t.innerHTML='<div clas\
s="text-[10px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const n=await apiFetch(MCP_URLS.
tools(e)),i=await n.json().catch(()=>({}));if(!n.ok){t.innerHTML=`<div class="text-[10px] text-red-4\
00">${mcpEsc(i.error||"\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}</div>`;return}const a=i&&
Array.isArray(i.tools)?i.tools:[];if(!a.length){t.innerHTML='<div class="text-[10px] text-gray-600">\
\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u300C\u63A5\u7D9A\u30C6\u30B9\u30C8\u300D\u3067\u53D6\u5F97\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>';
return}const r=a.map((l,c)=>`
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(l.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(l.description||"")}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${l.read_only?"bg-emerald-800/40 text-eme\
rald-200":"bg-amber-800/40 text-amber-200"}">${l.read_only?"\u8AAD\u307F\u53D6\u308A":"\u5909\u66F4"}\
</span>
</div>`).join("");t.innerHTML=`<div class="rounded border border-gray-800 bg-black/20 p-2">${r}</div\
>`}catch{t.innerHTML='<div class="text-[10px] text-red-400">\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}}
o(mcpLoadTools,"mcpLoadTools");async function mcpSaveOauthClient(e,t,n,i){mcpStatusMsg("mcp-status-m\
sg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);const a={provider_key:e,client_id:t,client_secret:n};
try{const r=await apiFetch(MCP_URLS.oauthClient(),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(a)}),l=await r.json().catch(()=>({}));if(!r.ok){mcpStatusMsg("mcp-status\
-msg",l.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg("mcp\
-status-msg","OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch(r){mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(r&&r.message?r.message:r),!0)}}o(mcpSaveOauthClient,"mcpSaveOauthClient");async function mcpAddCustomServer(){
const e=get("mcp-custom-name"),t=get("mcp-custom-url"),n=get("mcp-custom-auth"),i=get("mcp-custom-de\
sc"),a=get("mcp-custom-bearer"),r=get("mcp-custom-status"),l=get("mcp-add-server-btn");if(!e||!t||!n)
return;const c=(e.value||"").trim(),u=(t.value||"").trim(),f=n.value||"none",g=i?(i.value||"").trim():
"",y=(a&&a.value||"").trim();if(!c||!u){r&&(r.textContent="\u8868\u793A\u540D\u3068URL\u306F\u5FC5\u9808\u3067\u3059",
r.style.color="#f87171");return}l&&(l.disabled=!0),r&&(r.textContent="\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",
r.style.color="#9ca3af");const w={name:c,url:u,auth_type:f,description:g};f==="bearer"&&y&&(w.bearer_token=
y);try{const v=await apiFetch(MCP_URLS.servers(),{method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(w)}),k=await v.json().catch(()=>({}));if(!v.ok){r&&(r.textContent=k.error||
"\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",r.style.color="#f87171");return}r&&(r.textContent=
k.probe&&k.probe.message||"\u8FFD\u52A0\u3057\u307E\u3057\u305F",r.style.color=k.probe&&k.probe.ok?"\
#34d399":"#fbbf24"),e.value="",t.value="",i&&(i.value=""),a&&(a.value=""),mcpLoaded=!1,loadMcpServers(
!0)}catch(v){r&&(r.textContent="\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(v&&v.message?
v.message:v),r.style.color="#f87171")}finally{l&&(l.disabled=!1)}}o(mcpAddCustomServer,"mcpAddCustom\
Server");function bindMcpSettingsUi(){const e=get("mcp-server-list");if(!e)return;const t=get("mcp-a\
dd-server-btn");t&&t.addEventListener("click",mcpAddCustomServer);const n=get("mcp-custom-auth"),i=get(
"mcp-custom-bearer-wrap");if(n&&i){const r=o(()=>{i.classList.toggle("hidden",n.value!=="bearer")},"\
syncBearer");n.addEventListener("change",r),r()}const a=get("mcp-save-google-client-btn");a&&a.addEventListener(
"click",async()=>{const r=get("mcp-google-client-id"),l=get("mcp-google-client-secret"),c=get("mcp-g\
oogle-client-state"),u=r?r.value:"",f=l?l.value:"";if(!u&&!f){c&&(c.textContent="Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
c.style.color="#f87171");return}await mcpSaveOauthClient(mcpGoogleProviderKey,u||"********",f||"****\
****",null)}),e.addEventListener("click",async r=>{const l=r.target.closest("[data-act]");if(!l)return;
const c=l.getAttribute("data-act"),u=l.getAttribute("data-id");if(c==="test"){r.preventDefault(),mcpTestServer(
u);return}if(c==="tools"){r.preventDefault(),mcpLoadTools(u);return}if(c==="auth"||c==="reconnect"){
r.preventDefault(),mcpOpenAuth(u);return}if(c==="disconnect"){r.preventDefault(),mcpDisconnect(u);return}
if(c==="delete"){r.preventDefault(),mcpDeleteServer(u);return}if(c==="edit-oauth"){if(r.preventDefault(),
l.closest("[data-mcp-server]")){const g=l.getAttribute("data-oauth-pk")||"",y=mcpServers.find(k=>String(
k.id)===String(u)),w=document.createElement("div");w.className="mt-2 rounded border border-amber-700\
/50 bg-amber-950/20 p-2",w.innerHTML=`
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" placeholder="Client ID" autocomplete="off" data-1p-ignore="true" class="m\
cp-oauth-edit-cid w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white" va\
lue="">
        <input type="password" placeholder="Client Secret" autocomplete="off" data-1p-ignore="true" \
class="mcp-oauth-edit-sec w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-w\
hite">
    </div>
    <div class="flex justify-end mt-1 gap-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oa\
uth" data-id="${u}" data-pk="${mcpEsc(y&&(y.oauth_provider_key||y.slug)||"")}">\u4FDD\u5B58</button>
    </div>`;const v=l.closest("div");v.parentNode.insertBefore(w,v.nextSibling),l.remove()}return}if(c===
"save-oauth"){r.preventDefault();const f=l.getAttribute("data-pk")||"",g=l.closest("[data-mcp-server\
]")||document,y=g.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid'),w=g.querySelectorAll(
'[data-oauth-role="secret"], .mcp-oauth-edit-sec'),v=y.length?y[y.length-1].value:"",k=w.length?w[w.
length-1].value:"";if(!v&&!k){mcpStatusMsg("mcp-status-msg","Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpSaveOauthClient(f,v||"********",k||"********",u);return}if(c==="save-bearer"){r.preventDefault();
const f=document.querySelector(`[data-bearer-id="${u}"]`),g=f?f.value:"";if(!g||g.trim()===""){mcpStatusMsg(
"mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);try{const y=await apiFetch(
MCP_URLS.server(u),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({bearer_token:g})}),
w=await y.json().catch(()=>({}));if(!y.ok){mcpStatusMsg("mcp-status-msg",w.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch{mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}return}}),e.addEventListener("change",r=>{const l=r.target.closest(".mcp-enable-toggle");l&&mcpToggleEnabled(
l.getAttribute("data-id"),l.checked)})}o(bindMcpSettingsUi,"bindMcpSettingsUi");const initMcpUi=o(()=>{
try{bindMcpSettingsUi()}catch{}try{loadMcpServers()}catch{}},"initMcpUi");document.readyState==="loa\
ding"?document.addEventListener("DOMContentLoaded",initMcpUi,{once:!0}):initMcpUi();function bindMcpPromptToggle(){
const e=get("enable-mcp");e&&e.addEventListener("change",()=>{if(syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}})}if(o(bindMcpPromptToggle,"bindMcpPromptToggle"),
document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>{try{bindMcpPromptToggle()}catch{}});else
try{bindMcpPromptToggle()}catch{}function getModelTags(e,t){const n=[],i=(e.id||"").toLowerCase(),a=(e.
name||"").toLowerCase(),r=(e.desc||"").toLowerCase(),l=(t.category||"").toLowerCase();return(l.includes(
"gemini")||i.includes("gemini")||a.includes("gemini")||r.includes("gemini")||l.includes("banana")||a.
includes("banana"))&&n.push("gemini"),(l.includes("deepseek")||i.includes("deepseek")||a.includes("d\
eepseek")||r.includes("deepseek"))&&n.push("deepseek"),(l.includes("mistral")||i.includes("mistral")||
a.includes("mistral")||r.includes("mistral")||i.includes("ocr")||l.includes("ocr"))&&n.push("mistral"),
(l.includes("gpt")||l.includes("openai")||i.includes("gpt")||a.includes("gpt")||r.includes("openai"))&&
n.push("openai"),(l.includes("xai")||l.includes("grok")||i.includes("grok")||a.includes("grok")||r.includes(
"xai"))&&n.push("xai"),(l.includes("image")||i.includes("image")||a.includes("image")||r.includes("i\
mage"))&&n.push("image"),(l.includes("audio")||l.includes("music")||l.includes("transcription")||l.includes(
"speech")||i.includes("tts")||i.includes("transcri")||a.includes("tts")||a.includes("transcri")||a.includes(
"voice")||r.includes("tts")||i.includes("realtime")||i.includes("live")||i.includes("voice-agent")||
i.includes("native-audio")||a.includes("audio")||r.includes("audio")||r.includes("speech-to-text"))&&
n.push("audio"),(i.includes("reasoning")||a.includes("reasoning")||r.includes("reasoning"))&&n.push(
"reasoning"),(l.includes("deepseek")||i.includes("deepseek")||a.includes("deepseek"))&&!n.includes("\
reasoning")&&n.push("reasoning"),(i.includes("fast")||a.includes("fast")||r.includes("fast")||l.includes(
"fast"))&&n.push("fast"),(i.includes("deepseek-v4-flash")||l.includes("deepseek")&&a.includes("flash"))&&
!n.includes("fast")&&n.push("fast"),(l.includes("anthropic")||i.includes("claude")||a.includes("clau\
de")||r.includes("anthropic"))&&n.push("anthropic"),(l.includes("kimi")||i.includes("kimi")||a.includes(
"kimi")||r.includes("moonshot"))&&n.push("kimi"),(l.includes("video")||i.includes("video")||i.startsWith(
"veo-")||i.includes("omni-")||a.includes("video")||r.includes("video"))&&n.push("video"),(l.includes(
"music")||i.startsWith("lyria-")||a.includes("music")||r.includes("music")||r.includes("song"))&&n.push(
"music"),(l.includes("transcription")||i.includes("transcri")||a.includes("transcri")||r.includes("t\
ranscription")||r.includes("speech-to-text"))&&n.push("transcription"),(l.includes("ocr")||i.includes(
"ocr")||a.includes("ocr")||r.includes("ocr"))&&n.push("ocr"),(l.includes("agent")||i.includes("agent")||
a.includes("agent")||r.includes("agentic")||r.includes("computer use")||r.includes("deep research"))&&
n.push("agent"),e.agenticView&&n.push("agentic view"),n}o(getModelTags,"getModelTags");function updateModelTagUi(){
const e=get("model-tag-bar");if(!e)return;e.querySelectorAll(".model-tag-btn").forEach(n=>{const i=n.
innerText.trim().toLowerCase(),a=(i==="all"?"all":i)===activeModelTag;n.classList.toggle("is-active",
a)})}o(updateModelTagUi,"updateModelTagUi");function getModelCapabilitySearchTerms(e){const t=String(
e.id||"").toLowerCase(),n=[],i=t.includes("deepseek"),a=t.includes("tts"),r=t.startsWith("mistral-oc\
r"),l=a||r||t.includes("transcribe")||t.includes("realtime")||t.includes("voice-agent")||t.includes(
"native-audio")||t.includes("live")||t.includes("image")||t.includes("video")||t.startsWith("veo-")||
t.includes("omni-flash")||t.startsWith("lyria-")||t.includes("embedding"),c=!l&&(t.includes("gpt")||
t.includes("gemini")||t.includes("grok")||i||t.startsWith("deep-research-")||t.startsWith("antigravi\
ty-")),u=o((...g)=>g.forEach(y=>n.push(y,y.replace(/-/g," "))),"add");if((t.includes("gemini-3.1-fla\
sh-image")||t.includes("gemini-3-pro-image")||t.includes("gemini-2.5-flash-image"))&&u("image genera\
tion","image editing"),t==="gemini-3.1-flash-lite-image"||t==="gemini-3.1-flash-image"?u("thinking",
"\u601D\u8003","minimal","high","thinking level"):t.includes("gemini")&&!l&&(u("thinking","\u601D\u8003",
"thinking level"),t==="gemini-3.8-flash"||t==="gemini-3.7-flash"?u("low","medium","high"):t==="gemin\
i-3.6-flash"?u("medium","high"):t==="gemini-3.5-flash-lite"?u("minimal","medium","high"):t.includes(
"flash")?u("minimal","low","medium","high"):u("low","high")),i&&(u("thinking","\u601D\u8003","reason\
ing","\u63A8\u8AD6","reasoning effort","high"),t!=="deepseek-v4-pro"&&u("low"),t.includes("v4-flash")&&
u("none","max")),c&&(t.includes("gpt-5")||t.includes("o1")||t.includes("o3")||t.includes("grok-4.3")||
t.includes("grok-4.5")||t.includes("grok-4.6")||t.includes("grok-4.20-0309-reasoning")||t.includes("\
grok-build")||t.includes("multi-agent")||t.includes("gpt")&&!a)){u("reasoning","\u63A8\u8AD6","reaso\
ning effort","low","high");const g=t==="gpt-5.6"||t.startsWith("gpt-5.6-"),y=t.includes("grok-4.6"),
w=t.includes("grok-4.3")||t.includes("grok-4.5")||y||t.includes("grok-4.20-0309-reasoning")||t.includes(
"grok-build")||t.includes("multi-agent")||t.includes("gpt-5")||t.includes("o1")||t.includes("o3"),v=t.
includes("grok-4.3")||t.includes("grok-build")||t.includes("gpt-5")||i;w&&u("medium"),v&&u("none"),(g||
i)&&u("max"),(y||t.includes("multi-agent")||g)&&u("xhigh")}return t.includes("claude")&&u("thinking",
"\u601D\u8003","thinking budget","budget"),e.agenticView&&u("agentic view"),[...new Set(n)]}o(getModelCapabilitySearchTerms,
"getModelCapabilitySearchTerms");const modelListGroups=[];let modelListBanner=null,modelListEmpty=null,
modelListBuilt=!1,modelListAnimated=!1,modelListRenderFrame=0;function buildModelList(){const e=get(
"model-list-container");!e||modelListBuilt||(e.innerHTML="",modelListBanner=document.createElement("\
div"),modelListBanner.className="model-banner hidden",e.appendChild(modelListBanner),MODELS.forEach(
t=>{const n=t.items.filter(l=>!l.deprecated);if(!n.length)return;const i=document.createElement("sec\
tion");i.className="model-list-group",i.innerHTML=`
                    <div class="model-group-header">
                        <i class="${t.icon}"></i>
                        <div>
                            <h3 class="model-group-title">${t.category}</h3>
                            <p class="model-group-desc">${t.description}</p>
                        </div>
                    </div>
                    <div class="model-group-grid"></div>
                `;const a=i.querySelector(".model-group-grid"),r=n.map(l=>{const c=document.createElement(
"button"),u=String(l.apiId||l.id||"").trim(),f=l.agenticView?'<span class="inline-flex items-center \
gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold tex\
t-teal-200 whitespace-nowrap" title="Agentic View\u5BFE\u5FDC\uFF1A\u753B\u50CF\u3092\u30AF\u30ED\u30C3\u30D7\u3057\u3066\u518D\u89B3\u5BDF\u3057\u306A\u304C\u3089\u63A8\u8AD6\u3092\u7D99\u7D9A\u3067\u304D\u307E\u3059"><i class="fas fa-eye"\
 aria-hidden="true"></i>Agentic View</span>':"",g=u?`<div class="text-[10px] text-cyan-300/90 mt-1.5\
 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(u)}</\
div>`:"",y=l.price?`<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-\
1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${l.price}</span></div>`:"";
return c.type="button",c.className="model-card",c.dataset.selected="0",c.onclick=()=>selectModel(l.id,
l.name),c.innerHTML=`
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="model-name font-bold text-sm">${l.name}</span>
                                ${f}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle hidden shrink-0 mt-0.5\
"></i>
                        </div>
                        <span class="model-desc text-[10px]">${l.desc}</span>
                        ${g}
                        ${y}
                    `,a.appendChild(c),{model:l,button:c,searchText:`${l.name} ${l.id} ${u} ${l.agenticView?
"agentic view":""} ${t.category} ${getModelTags(l,t).join(" ")} ${getModelCapabilitySearchTerms(l).join(
" ")}`.toLowerCase(),provider:getModelApiProvider(l.id),tags:new Set(getModelTags(l,t))}});modelListGroups.
push({element:i,entries:r}),e.appendChild(i)}),modelListEmpty=document.createElement("div"),modelListEmpty.
className="model-list-empty hidden",e.appendChild(modelListEmpty),modelListBuilt=!0)}o(buildModelList,
"buildModelList");function updateModelButtonSelection(e,t){const n=t===e.model.id;if(e.button.dataset.
selected===(n?"1":"0"))return;e.button.dataset.selected=n?"1":"0",e.button.classList.toggle("is-sele\
cted",n);const i=e.button.querySelector(".model-selected-icon");i&&i.classList.toggle("hidden",!n)}o(
updateModelButtonSelection,"updateModelButtonSelection");function renderModelList(e="",t={}){const n=get(
"model-list-container");if(!n)return;buildModelList();const i=e.toLowerCase(),a=window._visionPickerActive?
null:getPromptCacheLockedProvider(),r=a?PROVIDER_LABELS[a]||a:"",l=get("model-select")?get("model-se\
lect").value:"";let c=0;modelListBanner.classList.toggle("hidden",!a),a&&(modelListBanner.innerHTML=
`<i class="fas fa-database mr-1.5"></i>PromptCache \u6709\u52B9\u4E2D: <strong>${r}</strong> \u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\
\u629E\u3067\u304D\u307E\u3059\uFF08\u4ED6API\u3078\u306E\u5207\u66FF\u306F\u4E0D\u53EF\uFF09`),modelListGroups.
forEach(u=>{let f=0;u.entries.forEach(g=>{const y=g.searchText.includes(i)&&(!a||g.provider===a)&&(activeModelTag===
"all"||g.tags.has(activeModelTag));g.button.classList.toggle("hidden",!y),updateModelButtonSelection(
g,l),y&&(f+=1)}),u.element.classList.toggle("hidden",f===0),c+=f}),modelListEmpty.classList.toggle("\
hidden",c!==0),c===0&&(modelListEmpty.textContent=a?`No ${r} models found.`:"No models found."),t.animate&&
!modelListAnimated&&(modelListAnimated=!0,n.classList.add("model-list-animate"))}o(renderModelList,"\
renderModelList");function scheduleModelListRender(e){modelListRenderFrame&&cancelAnimationFrame(modelListRenderFrame),
modelListRenderFrame=requestAnimationFrame(()=>{modelListRenderFrame=0,renderModelList(e)})}o(scheduleModelListRender,
"scheduleModelListRender");function animateModelCategoryChange(){const e=get("model-list-container");
e&&(e.classList.remove("model-category-enter"),e.offsetWidth,e.classList.add("model-category-enter"))}
o(animateModelCategoryChange,"animateModelCategoryChange");let modelListScrollFrame=0;function scrollSelectedModelIntoView(){
const e=get("model-list-container"),t=get("model-select")?get("model-select").value:"",n=modelListGroups.
flatMap(w=>w.entries).find(w=>w.model.id===t);if(!e||!n||n.button.classList.contains("hidden"))return;
const i=e.getBoundingClientRect(),a=n.button.getBoundingClientRect(),r=12;let l=e.scrollTop;if(a.top<
i.top+r?l+=a.top-i.top-r:a.bottom>i.bottom-r&&(l+=a.bottom-i.bottom+r),l=Math.max(0,Math.min(l,e.scrollHeight-
e.clientHeight)),Math.abs(l-e.scrollTop)<1)return;if(modelListScrollFrame&&cancelAnimationFrame(modelListScrollFrame),
window.matchMedia("(prefers-reduced-motion: reduce)").matches){e.scrollTop=l;return}const c=e.scrollTop,
u=l-c,f=performance.now(),g=160,y=o(w=>{const v=Math.min(1,(w-f)/g),k=1-Math.pow(1-v,3);e.scrollTop=
c+u*k,v<1?modelListScrollFrame=requestAnimationFrame(y):modelListScrollFrame=0},"step");modelListScrollFrame=
requestAnimationFrame(y)}o(scrollSelectedModelIntoView,"scrollSelectedModelIntoView");function openModelModal(){
location.pathname!=="/model"&&history.pushState({modal:"model"},"","/model");const e=get("model-sear\
ch");e&&(e.value=""),updateModelTagUi(),syncModelSearchClear(),renderModelList("",{animate:!0}),showModal(
"model-modal"),requestAnimationFrame(()=>requestAnimationFrame(scrollSelectedModelIntoView)),e&&window.
innerWidth>768&&requestAnimationFrame(()=>e.focus({preventScroll:!0}))}o(openModelModal,"openModelMo\
dal"),window.closeModelModal=(e=!1)=>{hideModal("model-modal"),!e&&location.pathname==="/model"&&history.
back()};function selectModel(e,t){if(window._visionPickerActive){currentVisionModel=e,window._visionPickerActive=
!1,window.closeModelModal(),_syncVisionModelDisplay();return}if(isPromptCacheEnabled()){const a=getModelApiProvider(
get("model-select")?get("model-select").value:""),r=getModelApiProvider(e);if(a&&r&&a!==r){const l=PROVIDER_LABELS[a]||
a,c=PROVIDER_LABELS[r]||r;showToast(`PromptCache \u6709\u52B9\u4E2D\u306F\u4ED6API\uFF08${c}\uFF09\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002\u73FE\u5728: ${l}`,"warning",!0);return}}const n=get("model-sel\
ect");n.value=e,get("model-selector-text").innerText=t,window.closeModelModal();const i=new Event("c\
hange");n.dispatchEvent(i)}o(selectModel,"selectModel");function selectModelById(e){let t=e;for(const n of MODELS){
const i=n.items.find(a=>a.id===e);if(i){t=i.name;break}}selectModel(e,t)}o(selectModelById,"selectMo\
delById");function populateAiSafeFormFields(e){if(e)try{get("set-default-model")&&(get("set-default-\
model").value=e.default_model||get("set-default-model").value),get("set-default-vision-model")&&(get(
"set-default-vision-model").value=e.default_vision_model||"gemini-3-flash-preview"),get("set-default\
-search")&&(get("set-default-search").checked=!!e.default_enable_search),get("set-default-url-contex\
t")&&(get("set-default-url-context").checked=!!e.default_enable_url_context),get("set-default-maps")&&
(get("set-default-maps").checked=!!e.default_enable_maps),get("set-default-python")&&(get("set-defau\
lt-python").checked=!!e.default_enable_python),get("set-default-file-creation")&&(get("set-default-f\
ile-creation").checked=!!e.default_enable_file_creation),get("set-default-thinking")&&(get("set-defa\
ult-thinking").checked=!!e.default_enable_thinking),get("set-default-sys-prompt")&&(get("set-default\
-sys-prompt").checked=!!e.default_enable_system_prompt),get("set-default-mcp")&&(get("set-default-mc\
p").checked=e.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-default-thinking\
-level").value=e.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get("set-defau\
lt-thinking-budget").value=e.default_thinking_budget||4096),get("set-default-reasoning-effort")&&(get(
"set-default-reasoning-effort").value=e.default_reasoning_effort||"medium"),get("set-default-safety")&&
(get("set-default-safety").value=e.default_safety_setting||"default"),get("sys-prompt-text")&&(get("\
sys-prompt-text").value=e.system_prompt||""),get("set-global-sys-prompt-enabled")&&(get("set-global-\
sys-prompt-enabled").checked=e.system_prompt_enabled!==!1),get("set-apply-global-sys-prompt")&&(get(
"set-apply-global-sys-prompt").checked=e.apply_global_system_prompt!==!1),get("set-apply-auto-sys-pr\
ompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=e.apply_auto_system_prompt_notices!==
!1),get("set-mic-transcribe-mode")&&(get("set-mic-transcribe-mode").value=e.mic_transcribe_mode||"st\
t_api"),get("set-stt-model")&&(get("set-stt-model").value=e.stt_model||"gpt-4o-mini-transcribe"),get(
"set-llm-transcribe-prompt")&&(get("set-llm-transcribe-prompt").value=e.llm_transcribe_prompt||""),get(
"set-enter-to-send")&&(get("set-enter-to-send").checked=!!e.enter_to_send),(get("set-compact-prompt-\
mode")||get("set-minimal-prompt-mode")||get("set-prompt-bar-mode-normal"))&&writePromptBarModeToForm(
!!e.compact_prompt_mode,!!e.minimal_prompt_mode),e.minimal_prompt_mode?setMinimalPromptMode(!0):(Object.
prototype.hasOwnProperty.call(e,"compact_prompt_mode")||Object.prototype.hasOwnProperty.call(e,"mini\
mal_prompt_mode"))&&setCompactPromptMode(!!e.compact_prompt_mode),get("set-use-sw-cache")&&(get("set\
-use-sw-cache").checked=!!e.use_sw_cache),get("set-liquid-glass")&&(get("set-liquid-glass").checked=
!!e.liquid_glass_enabled),applyLiquidGlassMode(!!e.liquid_glass_enabled),get("set-auto-search-links")&&
(get("set-auto-search-links").checked=e.auto_search_on_links!==!1),get("set-use-last-settings")&&(get(
"set-use-last-settings").checked=!!e.use_last_chat_settings),get("set-voice-studio-ui")&&(get("set-v\
oice-studio-ui").checked=e.voice_studio_ui!==!1),get("set-latency-metrics")&&(get("set-latency-metri\
cs").checked=!!e.enable_latency_metrics),get("set-client-debug-log")&&syncClientDebugLogToggle(!!e.enable_client_debug_log,
"ai-settings"),get("set-bot-detect")&&(get("set-bot-detect").checked=e.bot_detection_enabled!==!1),get(
"set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!e.skip_2fa_on_google_login),get("set-d\
efault-2fa-method")&&(get("set-default-2fa-method").value=e.default_2fa_method||"totp")}catch{}}o(populateAiSafeFormFields,
"populateAiSafeFormFields");function syncModelSearchClear(){const e=get("model-search"),t=get("model\
-search-clear");t&&t.classList.toggle("hidden",!e||!e.value)}o(syncModelSearchClear,"syncModelSearch\
Clear"),get("model-search")&&get("model-search").addEventListener("input",e=>{scheduleModelListRender(
e.target.value),syncModelSearchClear()}),get("model-search-clear")&&get("model-search-clear").addEventListener(
"click",()=>{const e=get("model-search");e&&(e.value="",syncModelSearchClear(),scheduleModelListRender(
""),e.focus())}),get("model-tag-bar")&&(get("model-tag-bar").addEventListener("click",e=>{const t=e.
target.closest(".model-tag-btn");if(!t)return;const n=t.innerText.trim().toLowerCase(),i=MODEL_TAGS.
includes(n)?n:"all";if(i===activeModelTag)return;activeModelTag=i,updateModelTagUi();const a=get("mo\
del-search");renderModelList(a?a.value:""),animateModelCategoryChange()}),updateModelTagUi()),window.
quickStart=e=>{selectModelById(e),get("welcome-screen").classList.add("hidden")};const BROWSER_FAST_DISABLED_OPTIONS=[
["enable-search","search-container"],["enable-url-context","url-context-container"],["enable-maps","\
maps-grounding-container"],["enable-sys-prompt","sys-prompt-option"],["enable-prompt-cache","prompt-\
cache-container"],["enable-mcp","mcp-container"],["enable-file-creation","file-creation-container"]];
function applyBrowserFastModeRestrictions(){if(!browserFastModeEnabled)return;browserFastPreviousOptions||
(browserFastPreviousOptions={checks:Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([n])=>[n,!!(get(
n)&&get(n).checked)])),coding:!!codingModeEnabled}),BROWSER_FAST_DISABLED_OPTIONS.forEach(([n,i])=>{
const a=get(n),r=get(i);a&&(a.checked=!1,a.disabled=!0),r&&r.classList.add("opacity-50","pointer-eve\
nts-none")}),codingModeEnabled&&syncCodingModeUi(!1,{persist:!1});const e=get("enable-coding-mode"),
t=get("coding-mode-container");e&&(e.disabled=!0),t&&t.classList.add("opacity-50","pointer-events-no\
ne"),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}o(applyBrowserFastModeRestrictions,
"applyBrowserFastModeRestrictions");function restoreBrowserFastModeOptions(){const e=browserFastPreviousOptions;
if(!e)return;BROWSER_FAST_DISABLED_OPTIONS.forEach(([i,a])=>{const r=get(i),l=get(a);r&&(r.disabled=
!1,e&&e.checks&&Object.prototype.hasOwnProperty.call(e.checks,i)&&(r.checked=!!e.checks[i])),l&&l.classList.
remove("opacity-50","pointer-events-none")});const t=get("enable-coding-mode"),n=get("coding-mode-co\
ntainer");t&&(t.disabled=!1),n&&n.classList.remove("opacity-50","pointer-events-none"),e&&e.coding&&
syncCodingModeUi(!0,{persist:!1}),browserFastPreviousOptions=null,typeof updatePromptCacheUi=="funct\
ion"&&updatePromptCacheUi(),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}
o(restoreBrowserFastModeOptions,"restoreBrowserFastModeOptions");function isBatchModelKey(e){const t=String(
e||"").trim().toLowerCase();return t.startsWith("gpt-")?!/(image|audio|tts|transcribe|realtime|search)/.
test(t):t.startsWith("grok-")?!/(image|video|voice|audio|tts|realtime)/.test(t):t.startsWith("gemini\
-")?!/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.
test(t):!1}o(isBatchModelKey,"isBatchModelKey");function updateBatchUi(e){const t=get("batch-mode-co\
ntainer"),n=get("enable-batch-mode");if(!t||!n)return;const i=isBatchModelKey(e);t.classList.toggle(
"hidden",!i),n.disabled=!i||browserFastModeEnabled,i||(n.checked=!1),t.classList.toggle("ring-1",i&&
n.checked),t.classList.toggle("ring-violet-300",i&&n.checked)}o(updateBatchUi,"updateBatchUi");function setBrowserFastModeEnabled(e,t={}){
browserFastModeEnabled=!!e;const n=get("enable-browser-fast-mode");n&&(n.checked=browserFastModeEnabled);
const i=get("browser-fast-mode-container");i&&(i.classList.toggle("ring-1",browserFastModeEnabled),i.
classList.toggle("ring-amber-300",browserFastModeEnabled)),!browserFastModeEnabled&&t.clearKey!==!1&&
(browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null),browserFastModeEnabled?applyBrowserFastModeRestrictions():
t.restoreOptions!==!1&&restoreBrowserFastModeOptions(),updateBatchUi(get("model-select")?get("model-\
select").value:"")}o(setBrowserFastModeEnabled,"setBrowserFastModeEnabled");function openBrowserFastModeModal(e=!0){
const t=get("browser-fast-mode-warning"),n=get("browser-fast-mode-ignore-row");t&&t.classList.toggle(
"hidden",!e),n&&n.classList.toggle("hidden",!e);const i=get("browser-fast-mode-key-description"),a=String(
get("model-select")?get("model-select").value:"Gemini");i&&(i.textContent=`${a} \u306E\u30E2\u30C7\u30EB\u5225\u30AD\u30FC \u2192 \u5171\u901AGemini\u30AD\u30FC\
\u306E\u9806\u306B\u3001\u30B5\u30FC\u30D0\u30FC\u304B\u3089\u81EA\u52D5\u53D6\u5F97\u3057\u307E\u3059\u3002`),
showModal("browser-fast-mode-modal")}o(openBrowserFastModeModal,"openBrowserFastModeModal");function browserFastBootstrapMatches(e,t,n,i){
return!e||e.model!==t||String(e.thread_id||"")!==String(n||"")?!1:String(e.parent_id||"")===String(i||
"")}o(browserFastBootstrapMatches,"browserFastBootstrapMatches");async function fetchBrowserFastBootstrap(e=!1){
const t=String(get("model-select")?get("model-select").value:"").trim(),n=currentThreadId||null,i=n&&
currentParentId||null;if(!e&&browserFastBootstrapMatches(browserFastBootstrap,t,n,i)&&browserFastApiKey)
return browserFastBootstrap;const a=await apiFetch("/api/browser_fast_mode/bootstrap",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({model:t,thread_id:n,parent_id:i})}),
r=await a.json().catch(()=>({}));if(!a.ok||!r.api_key)throw new Error(r.error||"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u306EGemini API\u30AD\
\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");return browserFastApiKey=
String(r.api_key),browserFastApiKeyModel=t,browserFastBootstrap=r,r}o(fetchBrowserFastBootstrap,"fet\
chBrowserFastBootstrap");async function requestBrowserFastModeEnable(){const e=String(get("model-sel\
ect")?get("model-select").value:"").toLowerCase();if(!e.startsWith("gemini-")||/(image|native-audio|tts|live)/.
test(e)){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306FGemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059",
"warning",!0),setBrowserFastModeEnabled(!1);return}if(currentImageUrls.length||uploadProgressState.active>
0||browserFastLocalFiles.size){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u308B\u524D\u306B\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u3092\u30AF\u30EA\u30A2\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0),setBrowserFastModeEnabled(!1);return}const t=(()=>{try{return localStorage.getItem(BROWSER_FAST_IGNORE_WARNING_STORAGE)===
"1"}catch{return!1}})();if(t){try{await fetchBrowserFastBootstrap(!0),setBrowserFastModeEnabled(!0,{
clearKey:!1}),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F",
"warning",!1)}catch(n){setBrowserFastModeEnabled(!1),showToast(n.message||"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u5316\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}return}openBrowserFastModeModal(!t)}o(requestBrowserFastModeEnable,"requestBrowserFastMo\
deEnable"),document.addEventListener("DOMContentLoaded",()=>{get("menu-btn")&&(get("menu-btn").onclick=
()=>{get("sidebar").classList.toggle("open"),get("overlay").classList.toggle("active")}),get("overla\
y")&&(get("overlay").onclick=()=>{get("sidebar").classList.remove("open"),get("overlay").classList.remove(
"active")})}),document.addEventListener("DOMContentLoaded",()=>{var Zn,ei;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const d=get("enable-batch-mode");d&&d.checked&&(d.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-batch-mode");t&&(t.onchange=()=>{if(t.checked){browserFastModeEnabled&&
setBrowserFastModeEnabled(!1);const d=get("enable-coding-mode");d&&d.checked&&(d.checked=!1,typeof syncCodingModeUi==
"function"&&syncCodingModeUi(!1),showToast("Batch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const n=get("model-\
select");n&&n.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const d=String(
n.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!d.startsWith("gemini-")||/(image|native-audio|tts|live|flash-cyber)/.test(d)?(setBrowserFastModeEnabled(
!1),n.dispatchEvent(new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const i=get("browser-fast-mode-enable-btn");
i&&(i.onclick=async()=>{const d=i.innerHTML;i.disabled=!0,i.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const m=get("browser-fast-mode-ignore-warning");if(m&&m.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(m){showToast(m.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{i.disabled=!1,i.innerHTML=d}});const a=get("browser-fast-mode-cancel-btn");a&&(a.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const r=document.getElementById(
"alpha-bar");setTimeout(()=>{if(r){const d=document.getElementById("version-display");if(d){const m=r.
getBoundingClientRect(),h=d.getBoundingClientRect(),b=h.left+h.width/2-(m.left+m.width/2),x=h.top+h.
height/2-(m.top+m.height/2);r.style.transform=`translate(${b}px, ${x}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const m=get("gpt-image-format"),h=get("gpt-image-compression-wrap");m&&h&&(m.value==="png"?h.
classList.add("hidden"):h.classList.remove("hidden"))}o(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const h=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{b&&(Array.from(b.options).
forEach(x=>{x.value!=="1K"&&(x.disabled=h)}),h&&b.value!=="1K"&&(b.value="1K"))})}o(c,"updateGeminiI\
mageUi");function u(){const d=get("grok-image-options");if(!d)return;const m=(get("model-select").value||
"").toLowerCase(),h=isGrokImageModel(),b=m==="grok-imagine-image-quality"||m==="grok-imagine-image-2\
.0",x=m==="grok-imagine-image-2.0";if(h){d.classList.remove("hidden");const T=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const M=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;M&&M.classList.toggle("hidden",!x)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const T=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const M=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;M&&M.classList.toggle(
"hidden",!x)}}o(u,"updateGrokImageUi");function f(){var b;const d=get("grok-video-options");if(!d)return;
const m=String(((b=get("model-select"))==null?void 0:b.value)||"").toLowerCase();isGrokVideoModel()?
d.classList.remove("hidden"):d.classList.add("hidden");const h=get("grok-video-resolution");if(h){const x=Array.
from(h.options).find(S=>S.value==="1080p");x&&(x.disabled=m!=="grok-imagine-video-1.5"),m!=="grok-im\
agine-video-1.5"&&h.value==="1080p"&&(h.value="720p")}}o(f,"updateGrokVideoUi");function g(){var x;const d=get(
"gemini-video-options");if(!d)return;const m=String(((x=get("model-select"))==null?void 0:x.value)||
"").toLowerCase();isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const h=get(
"gemini-video-resolution");if(h){const S=Array.from(h.options).find(M=>M.value==="4K"),T=m==="veo-3.\
1-lite-generate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";S&&(S.disabled=
T),T&&h.value==="4K"&&(h.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle(
"hidden",m==="gemini-omni-1.1-flash")}o(g,"updateGeminiVideoUi");function y(){const d=get("gemini-mu\
sic-options");if(!d)return;const m=isGeminiRealtimeMusicModel(),h=isGeminiMusicModel()&&!m;d.classList.
toggle("hidden",!h);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!m)}o(y,
"updateGeminiMusicUi");function w(){var T;const d=get("xai-chat-options");if(!d)return;const m=String(
((T=get("model-select"))==null?void 0:T.value)||"").toLowerCase(),h=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!h);const b=get("xai-logp\
robs"),x=get("xai-top-logprobs"),S=m.includes("grok-4.20");b&&(b.disabled=S,S&&(b.checked=!1)),x&&(x.
disabled=S,S&&(x.value=""))}o(w,"updateXaiChatUi");function v(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const h=get("modal-mistral-ocr-options");h&&
h.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const x=get(b);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(v,"updateMistralOcrUi");function k(){const d=get("image-input-limit\
s");if(!d)return;const m=(get("model-select").value||"").toLowerCase();let h="",b=!1;m.includes("gpt\
-image")?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):m==="deepseek-v4.1-flash"||m==="deepseek-v4-flash-vision-exp"?(b=!0,h=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):m.includes("deepseek")||(isGeminiImageModelKey(m)?(b=!0,m.includes("gemini-3.1-flash-lite-\
image")?h=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):m.includes("gemini-3.1-flash-image")?h=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):m.includes("gemini-2.5")&&m.includes("image")?h=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):h=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(m)?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):m.includes("grok")?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):m.includes("grok")&&m.includes("video")&&(b=!0,h=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(d.innerHTML=h,d.classList.remove("hidden")):(d.classList.add("hidden"),d.innerHTML="")}
o(k,"updateImageInputLimits");function _(){const d=get("model-select");if(!d)return;const m=d.value,
h=String(m||"").toLowerCase(),b=h.includes("deepseek"),x=get("thinking-options"),S=get("reasoning-ef\
fort-container"),T=get("enable-thinking"),M=get("thinking-level"),$=get("thinking-budget"),j=get("en\
able-search"),P=get("search-container"),H=get("url-context-container"),se=get("enable-maps"),q=get("\
maps-grounding-container"),N=get("enable-sys-prompt"),ee=get("sys-prompt-option"),Q=get("enable-pyth\
on"),xe=get("python-container"),W=get("prompt-cache-container"),le=get("enable-prompt-cache"),Ae=m===
"gpt-5-search-api",Y=m.includes("tts"),ce=isMistralOcrModel(m),Ce=h.includes("gemini-3.1-flash-lite-\
image"),$e=h.includes("gemini-3.1-flash-image")&&!Ce,st=isClaudeModelKey(m),We=h==="gemini-3.8-flash\
-cyber",lt=isLlmModel()&&!b&&!Y&&!h.includes("realtime")&&!h.includes("native-audio")&&!h.includes("\
live");W&&(lt?(W.classList.remove("hidden","opacity-50","pointer-events-none"),le&&(le.disabled=!1)):
(le&&(le.checked=!1,le.disabled=!0),W.classList.add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),
x&&x.classList.add("hidden"),S&&S.classList.add("hidden");const _t=get("vision-model-info");if(_t&&_t.
classList.add("hidden"),S){const fe=get("reasoning-effort");if(fe){Array.from(fe.options).forEach(Re=>{
const St=h==="gpt-5.6"||h.startsWith("gpt-5.6-"),Vt=h==="deepseek-v4.1-flash"||h==="deepseek-v4-flas\
h-0731"||h==="deepseek-v4-flash"||h==="deepseek-v4-flash-vision-exp",Ge=h==="deepseek-v4-pro",pt=h.includes(
"grok-4.5"),ln=h.includes("grok-4.6");Re.value==="max"?Re.classList.toggle("hidden",!St&&!Vt&&!Ge):Re.
value==="xhigh"?Re.classList.toggle("hidden",!ln&&!h.includes("multi-agent")&&!St):Re.value==="mediu\
m"?Re.classList.toggle("hidden",!(h.includes("grok-4.3")||pt||ln||h.includes("grok-4.20-0309-reasoni\
ng")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt-5")||h.includes("o1")||h.
includes("o3"))):Re.value==="none"?Re.classList.toggle("hidden",!h.includes("grok-4.3")&&!h.includes(
"grok-build")&&!h.includes("gpt-5")&&!Vt&&!Ge):Re.value==="low"&&Re.classList.toggle("hidden",Ge)});
const ke=fe.selectedOptions&&fe.selectedOptions[0];ke&&ke.classList.contains("hidden")&&(fe.value=b?
"high":"medium")}}H&&H.classList.add("hidden"),q&&q.classList.add("hidden"),T&&(T.disabled=!1),$&&($.
disabled=!0,$.classList.add("opacity-50"));const he=isGeminiImageModelKey(m);if(Y||ce)P&&(get("enabl\
e-search").checked=!1,P.classList.add("opacity-50","pointer-events-none")),H&&(get("enable-url-conte\
xt").checked=!1,H.classList.add("opacity-50","pointer-events-none")),q&&se&&(se.checked=!1,q.classList.
add("opacity-50","pointer-events-none")),xe&&(Q.checked=!1,xe.classList.add("opacity-50","pointer-ev\
ents-none")),N&&ee&&(N.checked=!1,N.disabled=!0,ee.classList.add("opacity-50"));else if($e||Ce)q&&se&&
(se.checked=!1,q.classList.add("hidden","opacity-50","pointer-events-none")),x.classList.remove("hid\
den"),Array.from(M.options).forEach(fe=>{["low","medium"].includes(fe.value)&&(fe.disabled=!0),["min\
imal","high"].includes(fe.value)&&(fe.disabled=!1)}),["minimal","high"].includes(M.value)||(M.value=
Ce?"minimal":"high"),T&&(T.disabled=!1),Ce&&(j&&(j.checked=!1,j.disabled=!0),P&&P.classList.add("opa\
city-50","pointer-events-none"));else if(he)q&&se&&(se.checked=!1,q.classList.add("hidden","opacity-\
50","pointer-events-none"));else if(st)x.classList.remove("hidden"),$&&($.disabled=!1,$.classList.remove(
"opacity-50")),Array.from(M.options).forEach(fe=>{fe.disabled=!0}),xe&&(Q.checked=!1,xe.classList.add(
"opacity-50","pointer-events-none"));else if(We){x&&x.classList.remove("hidden"),T&&(T.checked=!0,T.
disabled=!0),Array.from(M.options).forEach(ke=>{ke.disabled=!["low","medium","high"].includes(ke.value)}),
["low","medium","high"].includes(M.value)||(M.value="medium"),[P,H,q,xe].forEach(ke=>{ke&&ke.classList.
add("opacity-50","pointer-events-none")}),[j,se,Q].forEach(ke=>{ke&&(ke.checked=!1,ke.disabled=!0)});
const fe=get("enable-url-context");fe&&(fe.checked=!1,fe.disabled=!0),N&&ee&&(N.disabled=!1,ee.classList.
remove("opacity-50"))}else if(m.includes("gemini")&&!he){x.classList.remove("hidden"),H&&H.classList.
remove("hidden","opacity-50","pointer-events-none");const fe=m.includes("gemini-3");q&&(fe?q.classList.
remove("hidden","opacity-50","pointer-events-none"):(se&&(se.checked=!1),q.classList.add("hidden","o\
pacity-50","pointer-events-none")));const ke=m.includes("flash");Array.from(M.options).forEach(Re=>{
m==="gemini-3.8-flash"||m==="gemini-3.7-flash"?Re.disabled=!["low","medium","high"].includes(Re.value):
m==="gemini-3.6-flash"?Re.disabled=!["medium","high"].includes(Re.value):m==="gemini-3.5-flash-lite"?
Re.disabled=!["minimal","medium","high"].includes(Re.value):["minimal","medium"].includes(Re.value)?
Re.disabled=!ke:Re.disabled=!1}),(m==="gemini-3.8-flash"||m==="gemini-3.7-flash")&&!["low","medium",
"high"].includes(M.value)||m==="gemini-3.6-flash"&&!["medium","high"].includes(M.value)?M.value="med\
ium":m==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(M.value)?M.value="minimal":!ke&&
["minimal","medium"].includes(M.value)&&(M.value="high"),fe?T&&(T.checked=!0,T.disabled=!0):T&&(T.disabled=
!1),$&&m.includes("gemini-2.5")&&($.disabled=!1,$.classList.remove("opacity-50")),$&&!m.includes("ge\
mini-2.5")&&($.disabled=!0,$.classList.add("opacity-50"))}if(isLlmModel()&&(h.includes("gpt-5")||h.includes(
"o1")||h.includes("o3")||h.includes("grok-4.3")||h.includes("grok-4.5")||h.includes("grok-4.6")||h.includes(
"grok-4.20-0309-reasoning")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt")&&
!h.includes("tts")))S.classList.remove("hidden"),P&&P.classList.remove("opacity-50","pointer-events-\
none");else if(b){S.classList.remove("hidden");const fe=get("vision-model-info");if(fe&&fe.classList.
toggle("hidden",h==="deepseek-v4.1-flash"||h==="deepseek-v4-flash-vision-exp"),j&&(j.checked=!1,j.disabled=
!0),P&&P.classList.add("opacity-50","pointer-events-none"),H){const ke=get("enable-url-context");ke&&
(ke.checked=!1),H.classList.add("opacity-50","pointer-events-none")}q&&se&&(se.checked=!1,q.classList.
add("opacity-50","pointer-events-none"))}else ce||(P&&P.classList.remove("opacity-50","pointer-event\
s-none"),q&&se&&(se.checked=!1,q.classList.add("hidden","opacity-50","pointer-events-none")));if(Y?xe&&
xe.classList.add("opacity-50","pointer-events-none"):(xe&&xe.classList.remove("opacity-50","pointer-\
events-none"),(!he||$e)&&!m.includes("gpt-image")&&(N.disabled=!1,ee.classList.remove("opacity-50"))),
(he&&!$e||m.includes("gpt-image")||isGrokImageModel()||isGrokVideoModel()||ce)&&N&&ee&&(N.checked=!1,
N.disabled=!0,ee.classList.add("opacity-50")),xe&&(isLlmModel()?(xe.classList.remove("hidden"),Q.disabled=
!1):(Q.checked=!1,Q.disabled=!0,xe.classList.add("hidden"))),Ae?(j&&(j.checked=!0,j.disabled=!0),P&&
P.classList.add("opacity-50","pointer-events-none"),xe&&(Q.checked=!1,Q.disabled=!0,xe.classList.add(
"opacity-50","pointer-events-none"))):j&&!m.includes("tts")&&!ce&&!b&&!Ce&&(j.disabled=!1),We){[j,se,
Q].forEach(ke=>{ke&&(ke.checked=!1,ke.disabled=!0)});const fe=get("enable-url-context");fe&&(fe.checked=
!1,fe.disabled=!0),[P,H,q,xe].forEach(ke=>{ke&&ke.classList.add("opacity-50","pointer-events-none")})}
const je=get("mask-btn");je&&(isGptImageModel()?je.classList.remove("hidden"):(je.classList.add("hid\
den"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),l(),
c(),u(),f(),g(),y(),updateBatchUi(m),w(),v(),k(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}o(_,"toggleOptions"),get("model-select")&&(get("model-select").addEventListener(
"change",_),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
_(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const C=get("enable-canvas-mode");C&&(C.checked=canvasModeEnabled,C.addEventListener("change",()=>syncCanvasModeUi(
C.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const L=get("enable-codin\
g-mode");L&&(L.checked=codingModeEnabled,L.addEventListener("change",()=>syncCodingModeUi(L.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",d=>{const m=d.
target.closest("[data-canvas-block-index]");if(!m)return;const h=Number(m.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(h,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",d=>{if(d.target.value===
"")return;const m=Number(d.target.value);Number.isInteger(m)&&applyCanvasSelection(m,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",d=>{const m=d.target.closest(
"[data-canvas-panel-view]");if(!m)return;const h=m.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
h,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const d=getCanvasModeElements(),m=d&&d.code&&d.code.textContent||"";if(!m.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
m,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const E=get(
"prompt-controls-toggle-btn");E&&(E.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,m,h){const b=(m||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",S=(b||"").toLowerCase();
let T="";try{const N=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&x.length>2e4?T=escapeHtml(
x):T=hljs.highlight(x,{language:N}).value}catch{T=escapeHtml(x)}const M=encodeURIComponent(x).replace(
/'/g,"%27"),$=detectBlockedScriptsInCode(x),j=hashString(`${b||"TEXT"}
${x||""}`);let P="";if(canvasModeEnabled){const N=String(canvasPreviewState.selectedKey||"")===j,ee=N?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";P=`<button\
 class="canvas-preview-btn${N?" canvas-active":""}" data-code="${M}" data-code-key="${j}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${ee}" aria-label="${ee}" aria-pressed="${N?"true":"false"}"\
><i class="fas ${N?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,x)){const N=$?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
P=`<button class="html-preview-btn" data-code="${M}" ${$?'data-suspicious="1"':""} title="${N}" aria\
-label="${N}"><i class="fas ${$?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const H=`\
<button class="download-btn" data-code="${M}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,se=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${M}" data-code-key="${j}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,q=(b||"TEXT")+($?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\u305F\
">\u26A0</span>':"");return`<div class="code-wrapper collapsed" data-collapsed="true" data-code-key=\
"${j}"><div class="code-header"><span class="code-lang">${q}</span><div class="code-actions"><button\
 class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"\
></i></button>${se}${P}${H}<button class="copy-btn" data-code="${M}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i\
 class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language\
-${b}">${T}</code></pre></div></div>`},link(d,m,h){return`<a href="${d}" title="${m||""}" target="_b\
lank">${h}</a>`},image(d,m,h){return buildChatImageHtml(d,{alt:h,title:m})}},breaks:!0,gfm:!0}),threadObserver=
new IntersectionObserver(d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread\
-list"),threshold:.1}),threadObserver.observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),
(Zn=get("version-update-dismiss"))==null||Zn.addEventListener("click",()=>{const d=localStorage.getItem(
"app_version")||"";d&&localStorage.setItem("version_notified",d),hideModal("version-update-modal")});
const B=get("version-update-clear-cache");if(B&&(B.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.
clearCacheOnVersionUpdate),B.addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=
saveVersionUpdateCachePreference(B.checked)})),(ei=get("version-update-reload"))==null||ei.addEventListener(
"click",async()=>{var m;await versionUpdateCachePreferenceSavePromise.catch(()=>{}),!!((m=get("versi\
on-update-clear-cache"))!=null&&m.checked)?await clearSiteCacheAndReload(get("version-update-reload"),
{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.setVersionChangeHandler(
d=>{d&&d!==appVersion&&(localStorage.getItem("version_notified")||"")!==d&&(localStorage.setItem("ap\
p_version",d),purgeCaches().then(()=>checkAndNotifyVersion(d)))}),window.ConnectionMonitor.start(),window.
addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener("offline",
()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offline")}),window.
addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener("visibil\
itychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener("page\
hide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&botConfig.
lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(d){console.error(d)}try{runBotDetectionGate()}catch(d){console.error(
d)}}else{const d=get("turnstile-container");d&&d.classList.add("hidden")}const K=o(d=>{if(!d)return"\
\u4E0D\u660E";const m=new Date(d);return Number.isNaN(m.getTime())?d:m.toLocaleString()},"formatSess\
ionTime"),Z=o(d=>{const m=Array.isArray(d)?d:[],h=get("passkey-list"),b=get("passkey-count");if(b&&(b.
innerText=String(m.length)),!!h){if(!m.length){h.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
h.innerHTML="",m.forEach((x,S)=>{const T=x&&x.id?String(x.id):"",M=document.createElement("div");M.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const $=document.
createElement("div");$.className="min-w-0";const j=document.createElement("div");j.className="text-x\
s text-gray-200 truncate",j.innerText=x&&x.name?String(x.name):`Security Key ${S+1}`;const P=document.
createElement("div");P.className="text-[10px] text-gray-500 mt-1",P.innerText=x&&x.created_at?`\u767B\u9332\u65E5\u6642:\
 ${K(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",$.appendChild(j),$.appendChild(P),M.appendChild(
$);const H=document.createElement("button");H.type="button",H.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",H.innerText="\u524A\u9664",H.
disabled=!T,T&&(H.onclick=()=>window.removeWebAuthnCredential(T)),M.appendChild(H),h.appendChild(M)})}},
"renderPasskeyList"),Ie=o(d=>{const m=get("session-list");if(m){if(!d||!d.length){m.innerHTML='<div \
class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(h=>{const b=h.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=h.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!h.is_current&&!h.is_revoked?`<button \
data-session-id="${escapeHtml(h.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
T=(h.user_agent||"Unknown").slice(0,120),M=h.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${x}<div class="text-xs text-gray-200">${escapeHtml(
M)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(T)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(K(h.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(K(h.created_at))}</div></div>${S}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(h=>{h.onclick=async()=>{const b=h.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let S={};try{S=await x.json()}catch{}if(x.ok){if(S.logged_out){
location.href="/login";return}await O()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),O=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div c\
lass="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api/\
sessions");let h={};try{h=await m.json()}catch{}if(!m.ok){if(h&&h.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(h.sessions||[]).filter(x=>!x.is_revoked);Ie(b)},"loadSessions"),G=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>O());const m=get("session-revoke-others-btn");m&&(m.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await O():showToast("\u64CD\u4F5C\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F","error",!0)});const h=get("session-revoke-all-btn");h&&(h.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(d=>{d&&(currentVisionModel=d.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
d);try{loadMcpServers()}catch{}d&&d.theme_color&&applyThemeColor(d.theme_color,!0),d&&Object.prototype.
hasOwnProperty.call(d,"minimal_prompt_mode")&&d.minimal_prompt_mode?setMinimalPromptMode(!0):d&&Object.
prototype.hasOwnProperty.call(d,"compact_prompt_mode")&&setCompactPromptMode(!!d.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings sync");
const m=get("enable-sys-prompt");m&&d&&d.system_prompt&&String(d.system_prompt).trim()&&(!m.disabled&&
!d.default_enable_system_prompt&&!d.use_last_chat_settings&&(m.checked=!0),_())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const te=get("vision-model-change-btn");te&&(te.onclick=()=>_openVisionModelSelector());const be=get(
"compression-format-only");be&&(be.onchange=()=>{const d=be.checked,m=get("compression-max-size"),h=get(
"compression-max-dim");m&&(m.disabled=d),h&&(h.disabled=d);const b=get("compression-size-wrap"),x=get(
"compression-dim-wrap");b&&(b.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const ue=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");ue(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const Le=get("storage-u\
sage-refresh");Le&&(Le.onclick=()=>loadStorageUsage());let ye=null;const Me=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),oe=o((d={})=>{const m=get("account-transfer-progress"),h=get("account-tra\
nsfer-progress-bar"),b=get("account-transfer-progress-percent"),x=get("account-transfer-progress-tex\
t"),S=get("account-transfer-progress-detail"),T=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&
m.classList.remove("hidden"),h&&(h.style.width=`${T}%`),b&&(b.textContent=`${Math.round(T)}%`),x&&(x.
textContent=d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){
const $={queued:"\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\
\u30D5\u30A1\u30A4\u30EB\u3092\u66F8\u304D\u51FA\u3057\u4E2D",finalizing:"\u6700\u7D42\u51E6\u7406\u4E2D",
ready:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u6E96\u5099\u5B8C\u4E86",downloading:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D",
uploading:"ZIP\u3092\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D",validating:"ZIP\u3092\u691C\u8A3C\u4E2D",
validating_files:"\u30D5\u30A1\u30A4\u30EB\u60C5\u5831\u3092\u691C\u8A3C\u4E2D",reading_files:"\u30D5\u30A1\u30A4\u30EB\u3092\
\u8AAD\u307F\u8FBC\u307F\u4E2D",importing_settings:"\u8A2D\u5B9A\u3092\u53CD\u6620\u4E2D",importing_credentials:"\
\u8A8D\u8A3C\u60C5\u5831\u3092\u53CD\u6620\u4E2D",importing_gems:"Gem\u3092\u8FFD\u52A0\u4E2D",saving_files:"\
\u30D5\u30A1\u30A4\u30EB\u3092\u4FDD\u5B58\u4E2D",importing_chats:"\u30C1\u30E3\u30C3\u30C8\u5C65\u6B74\u3092\u8FFD\u52A0\u4E2D",
importing_feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u8FFD\u52A0\u4E2D",importing_diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF\u3092\u8FFD\u52A0\u4E2D",cancelling:"\u30AD\u30E3\u30F3\u30BB\u30EB\u51E6\u7406\u4E2D",
cancelled:"\u30AD\u30E3\u30F3\u30BB\u30EB\u6E08\u307F",expired:"\u4FDD\u5B58\u671F\u9650\u5207\u308C",
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=$[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const M=get("account-transfer-cancel-btn");M&&M.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),re=o(d=>{Te&&(Te.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const h=get("account-transfer-cancel-btn");
h&&(h.disabled=!d)},"setAccountTransferControls"),J=o((d={})=>{const m=get("account-export-ready"),h=get(
"account-export-ready-text"),b=get("account-export-expiry"),x=get("account-export-download-btn"),S=!!(d.
available&&d.download_url);if(m&&m.classList.toggle("hidden",!S),!S){x&&x.removeAttribute("href");return}
const T=Math.max(0,Number(d.size_bytes)||0),M=T>=1024*1024*1024?`${(T/(1024*1024*1024)).toFixed(2)} \
GB`:`${(T/(1024*1024)).toFixed(1)} MB`;if(h){const $=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";h.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${M}${$}`}
if(b){const $=d.expires_at?new Date(d.expires_at):null;b.textContent=$&&!Number.isNaN($.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${$.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),A=o(async d=>{for(;ye===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),h=await m.json().catch(()=>({}));if(m.ok&&(h.state!=="pending"&&oe(h),["ready","completed","\
failed","cancelled","expired"].includes(h.state)))return h}catch{}await new Promise(m=>setTimeout(m,
700))}return null},"pollAccountTransfer"),F=o((d,m,h=!0)=>{m&&(oe(m),J(m),h&&m.state==="ready"?showToast(
m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):h&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),X(d))},"handleFinishedAccountExport"),U=o(async()=>{try{const d=await apiFetch("/api/acc\
ount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(J(m),m.state==="ready"){oe(m);return}if(["failed","cancelled","expired"].includes(
m.state)){oe(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||ye&&ye.id===
m.job_id||ye)return;const h={id:m.job_id,type:"export",stopped:!1,restored:!0};ye=h,re(!0),oe(m);const b=await A(
h);b&&F(h,b,!0)}catch{}},"refreshLatestAccountExport"),X=o(d=>{ye===d&&(ye=null),d.stopped=!0,re(!1)},
"finishAccountTransfer"),ie=get("account-transfer-cancel-btn");ie&&(ie.onclick=async()=>{const d=ye;
if(!(!d||d.stopped)){d.cancelRequested=!0,ie.disabled=!0,oe({progress:0,phase:"cancelling",message:"\
\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/tra\
nsfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}d.controller&&d.controller.
abort(),oe({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&J({available:!1}),X(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const Te=get("account-export-btn");Te&&(Te.onclick=async()=>{if(ye)return;const d={id:Me(),
type:"export",stopped:!1};ye=d,re(!0),J({available:!1}),oe({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const m=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:d.id}),keepalive:!0})),h=await m.json().catch(()=>({}));if(m.status===409&&
h.error==="export_in_progress"&&h.job_id)d.id=h.job_id;else if(!m.ok)throw new Error(h.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
h.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
oe({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await A(d);!d.cancelRequested&&b&&F(d,b,!0)}catch(m){const h=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";oe({progress:0,phase:"failed",message:h}),
showToast(h,"error",!0),X(d)}});const ne=get("account-export-download-btn");ne&&ne.addEventListener(
"click",async d=>{const m=ne.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const h=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await h.json().catch(
()=>({}));h.ok&&b.available&&b.download_url?(ne.href=b.download_url,window.location.assign(b.download_url)):
(J(b),oe(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),U())}catch{window.location.assign(m)}}}),re(!1),U();const pe=get("import-files-grid"),
V=get("import-files-info"),we=get("import-files-summary"),ot=o(d=>{const m=Math.max(0,Number(d)||0);
return m>=1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(
1)} MB`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let Pe=null;const dt=o(()=>{
if(!Pe)return;const d=Pe.files,m=Pe.selection;let h=0;d.forEach(S=>{m.has(S.archive_path)&&(h+=Number(
S.size_bytes)||0)});const b=Number(Pe.available_bytes)||0,x=h>b;we&&(we.textContent=`\u9078\u629E\u4E2D: ${ot(
h)} / \u5229\u7528\u53EF\u80FD: ${ot(b)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,we.classList.
toggle("text-red-300",x)),V&&(V.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),bt=o(
()=>{if(!pe||!Pe)return;pe.innerHTML="";const d=Pe.files;if(!d.length){pe.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
dt();return}d.forEach(m=>{const h=document.createElement("label"),b=Pe.selection.has(m.archive_path);
h.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,h.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${ot(m.size_bytes)}</div></div>`;const x=h.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?Pe.selection.add(m.archive_path):
Pe.selection.delete(m.archive_path),h.classList.toggle("border-blue-500",x.checked),h.classList.toggle(
"border-gray-600",!x.checked),dt()}),pe.appendChild(h)}),dt()},"renderImportFileItems"),Ze=o(d=>new Promise(
m=>{if(Pe={files:d.files||[],selection:new Set((d.files||[]).map(h=>h.archive_path)),available_bytes:d.
available_bytes,resolve:m},bt(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),mt=o(d=>{if(hideModal("import-files-modal"),Pe){const m=Pe.resolve;
Pe=null,m(d)}},"closeImportFileSelection"),Tt=get("import-files-close");Tt&&(Tt.onclick=()=>mt(null));
const xt=get("import-files-cancel");xt&&(xt.onclick=()=>mt(null));const Ct=get("import-files-confirm");
Ct&&(Ct.onclick=()=>{if(!Pe)return;const d=Array.from(Pe.selection);mt(d.length?d.join(","):"__none_\
_")});const ft=get("import-files-select-all");ft&&(ft.onclick=()=>{Pe&&(Pe.files.forEach(d=>Pe.selection.
add(d.archive_path)),bt())});const Nt=get("import-files-none");Nt&&(Nt.onclick=()=>{Pe&&(Pe.selection.
clear(),bt())});const Rt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u4F7F\u7528",apply_global_system_prompt:"\
\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528",apply_auto_system_prompt_notices:"\
\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528",auto_system_prompt_notices_config:"\
\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u306E\u7A2E\u985E\u5225\u8A2D\u5B9A",
gemini_backend:"Gemini \u30D0\u30C3\u30AF\u30A8\u30F3\u30C9",gemini_vertex_location:"Vertex AI \u30ED\u30B1\u30FC\u30B7\u30E7\
\u30F3",mic_transcribe_mode:"\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u65B9\u5F0F",stt_model:"\
STT\u30E2\u30C7\u30EB",llm_transcribe_prompt:"LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8",
enter_to_send:"Enter\u30AD\u30FC\u3067\u9001\u4FE1",use_sw_cache:"Service Worker\u30AD\u30E3\u30C3\u30B7\u30E5",
clear_cache_on_version_update:"\u30D0\u30FC\u30B8\u30E7\u30F3\u66F4\u65B0\u6642\u30AD\u30E3\u30C3\u30B7\u30E5\u524A\u9664",
theme_color:"\u30C6\u30FC\u30DE\u30AB\u30E9\u30FC",liquid_glass_enabled:"Liquid Glass",light_mode_enabled:"\
\u30E9\u30A4\u30C8\u30E2\u30FC\u30C9",auto_search_on_links:"\u30EA\u30F3\u30AF\u3067\u81EA\u52D5\u691C\u7D22",
compact_prompt_mode:"\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u8868\u793A\uFF08\u30B3\u30F3\u30D1\u30AF\u30C8\uFF09",
minimal_prompt_mode:"\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u8868\u793A\uFF08\u30DF\u30CB\u30DE\u30EB\uFF09",
use_last_chat_settings:"\u76F4\u524D\u306E\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u3092\u4F7F\u7528",voice_studio_ui:"\
\u97F3\u58F0\u30B9\u30BF\u30B8\u30AAUI",temp_chat_timeout_seconds:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u306E\u6709\u52B9\u6642\u9593\uFF08\u79D2\uFF09",
default_model:"\u65E2\u5B9A\u306E\u30E2\u30C7\u30EB",default_enable_search:"\u65E2\u5B9A: Search",default_enable_url_context:"\
\u65E2\u5B9A: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",default_enable_maps:"\u65E2\u5B9A: Maps",default_enable_python:"\
\u65E2\u5B9A: Python",default_enable_file_creation:"\u65E2\u5B9A: File",default_enable_thinking:"\u65E2\u5B9A:\
 Thinking",default_thinking_level:"\u65E2\u5B9A: Thinking\u30EC\u30D9\u30EB",default_thinking_budget:"\
\u65E2\u5B9A: Thinking budget",default_reasoning_effort:"\u65E2\u5B9A: Reasoning effort",default_enable_system_prompt:"\
\u65E2\u5B9A: \u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",default_enable_mcp:"\u65E2\u5B9A: MCP",
default_safety_setting:"\u65E2\u5B9A: \u5B89\u5168\u8A2D\u5B9A",default_vision_model:"Vision Model",
rich_paste_prompt_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30D7\u30ED\u30F3\u30D7\u30C8",
rich_paste_prompt_use_custom_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30AB\u30B9\u30BF\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u65E2\u5B9A",
last_model:"\u76F4\u524D\u306E\u30E2\u30C7\u30EB",last_enable_search:"\u76F4\u524D: Search",last_enable_url_context:"\
\u76F4\u524D: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",last_enable_maps:"\u76F4\u524D: Maps",last_enable_python:"\
\u76F4\u524D: Python",last_enable_file_creation:"\u76F4\u524D: File",last_enable_thinking:"\u76F4\u524D: Think\
ing",last_thinking_level:"\u76F4\u524D: Thinking\u30EC\u30D9\u30EB",last_thinking_budget:"\u76F4\u524D: Thinki\
ng budget",last_reasoning_effort:"\u76F4\u524D: Reasoning effort",last_enable_system_prompt:"\u76F4\u524D: \u30B7\u30B9\u30C6\
\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",last_enable_mcp:"\u76F4\u524D: MCP",last_safety_setting:"\u76F4\u524D: \u5B89\
\u5168\u8A2D\u5B9A",enable_latency_metrics:"\u30EC\u30B9\u30DD\u30F3\u30B9\u901F\u5EA6\u306E\u8A08\u6E2C",
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},ut=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const m=String(
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let yt=null;const Lt=o(
d=>{if(yt){const m=yt;yt=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Mt=o(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}yt=m;const b=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(b.length?
x.innerHTML=b.map(T=>{const M=Rt[T.field]||T.field,$=ut(T.current),j=ut(T.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(M)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
$)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(j)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),vt=get("settings-confir\
mation-modal");vt&&vt.addEventListener("click",d=>{d.target===vt&&Lt(!1)});const Wt=get("settings-co\
nfirmation-close");Wt&&(Wt.onclick=()=>Lt(!1));const Jt=get("settings-confirmation-cancel");Jt&&(Jt.
onclick=()=>Lt(!1));const R=get("settings-confirmation-confirm");R&&(R.onclick=()=>Lt(!0));const de=get(
"account-import-btn"),Ee=get("account-import-inplace"),Ne=get("account-import-inplace-warning");if(Ee&&
Ne){const d=o(()=>Ne.classList.toggle("hidden",!Ee.checked),"syncInplaceWarn");Ee.addEventListener("\
change",d),d()}de&&(de.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:
null,h=get("account-import-categories"),b=h?Array.from(h.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(q=>q.value):[],x=get("account-import-inplace"),S=!!(x&&x.checked),T=get("account-import\
-settings-bypass"),M=!!(T&&T.checked);let $=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const j=h?Array.from(h.querySelectorAll('input[type="checkbox"]:checked')).map(q=>(q.
closest("label")&&q.closest("label").textContent||q.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\u5B58\u30C7\
\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${j.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const P={id:Me(),type:"import",stopped:!1,controller:new AbortController};
ye=P,re(!0),oe({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const H=get("account-import-result");let se=Promise.resolve(null);try{const N=Math.max(1,Math.ceil(m.
size/10485760)),ee=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:P.controller.
signal})),Q=await ee.json().catch(()=>({}));if(!ee.ok)throw new Error(Q.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
P.uploadId=Q.upload_id;const xe=Q.chunk_size||10485760;let W=0,le=0;const Ae=o(async()=>{for(;;){const he=le++;
if(he>=N)return;const ge=m.slice(he*xe,Math.min(m.size,(he+1)*xe)),je=new FormData;je.append("chunk",
ge,m.name),je.append("index",String(he));const fe=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
P.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:je,signal:P.controller.signal})),
ke=await fe.json().catch(()=>({}));if(!fe.ok)throw new Error(ke.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W++,oe({progress:Math.min(35,Math.round(W/N*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${W}\
/${N}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let Y=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),Y=!0);try{await Promise.
all([Ae(),Ae(),Ae()]);const he=await apiFetch(`/api/account/import/upload/${encodeURIComponent(P.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:P.controller.signal})),ge=await he.json().
catch(()=>({}));if(!he.ok)throw new Error(ge.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
oe({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
Y&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ce="",Ce=!1,$e=0;const st=o(
async()=>{let he=!1;const ge=o(()=>{he||(he=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const je=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),fe=await je.
json().catch(()=>null);if(!je.ok||!fe){ge();return}cacheUserSettings(fe);const ke=get("settings-moda\
l");if(ke&&ke.classList.contains("modal-open"))try{Rn(fe)}catch{}fe.theme_color&&applyThemeColor(fe.
theme_color,!0),Object.prototype.hasOwnProperty.call(fe,"minimal_prompt_mode")&&fe.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(fe,"compact_prompt_mode")&&setCompactPromptMode(
!!fe.compact_prompt_mode)}catch{}ge()},"refreshSettingsFormAfterImport"),We=o(he=>{const ge=he&&he.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";H&&(H.textContent=`\u5B8C\u4E86: ${ge}`,
H.classList.remove("hidden","text-red-300"),H.classList.add("text-emerald-300")),oe({progress:100,phase:"\
completed",message:ge}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&st()},"finishImportSuccess"),lt=o(async()=>{
try{const ge=await(await apiFetch(`/api/account/transfer/${P.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return ge&&ge.state?ge:null}catch{return null}},"fetchImportSta\
tus"),_t=o(async()=>{const he=await lt();if(!he)return{status:"unknown"};if(he.state==="completed")return We(
he),{status:"done"};if(["failed","cancelled","expired"].includes(he.state))throw new Error(he.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(he.state==="needs_sel\
ection"&&Array.isArray(he.files)){const ge=await Ze({files:he.files,available_bytes:he.available_bytes});
return ge===null?(oe({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
P.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(P.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ce=ge,{status:"reselect"})}if(he.state===
"needs_settings_confirmation"&&Array.isArray(he.settings_changes))return await Mt({settings_changes:he.
settings_changes})?($=!0,{status:"reselect"}):(oe({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\
\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),P.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
P.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(he.state==="running"){const ge=await Promise.race([se.catch(()=>null),new Promise(je=>setTimeout(
()=>je(null),6e4))]);if(ge&&ge.state==="completed")return We(ge),{status:"done"};throw ge&&["failed",
"cancelled","expired"].includes(ge.state)?new Error(ge.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ce;){P.stopped=!0,await se.catch(()=>null),
P.stopped=!1,se=A(P);let he;try{he=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:P.uploadId,
categories:b.join(","),job_id:P.id,selected_files:ce,restore_inplace:S,confirm_settings:$||M}),signal:P.
controller.signal}))}catch(Ge){if(P.cancelRequested||Ge&&Ge.name==="AbortError")throw Ge;const pt=await _t();
if(pt.status==="done"){Ce=!0;break}if(pt.status==="cancelled")return;if(pt.status==="reselect")continue;
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let ge=null;try{ge=await he.json()}catch{ge=null}if(ge===null){const Ge=await _t();if(Ge.status==="d\
one"){Ce=!0;break}if(Ge.status==="cancelled")return;if(Ge.status==="reselect")continue;if(he.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!he.ok&&ge.error==="storage_limit_files"&&ge.files){const Ge=await Ze(ge);if(Ge===null){oe({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
P.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(P.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ce=Ge;continue}if(ge&&ge.status==="settings_confirmation"&&
Array.isArray(ge.settings_changes)){if(!await Mt(ge)){oe({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\
\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),P.uploadId&&
apiFetch(`/api/account/import/upload/${encodeURIComponent(P.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}$=!0;continue}if(!he.ok)throw new Error(ge.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F");const je=ge.imported||{},fe=[`\u8A2D\u5B9A ${je.settings||0}\u4EF6`,
`API\u8A8D\u8A3C ${je.api_credentials||0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${je.chats||0}\u4EF6`,`Ge\
m ${je.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${je.files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${je.
feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${je.diagnostics||0}\u4EF6`].join(" / "),ke=ge.duplicates||
{},Re={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\
\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"},St=[];for(const Ge of Object.keys(Re)){const pt=Number(
ke[Ge])||0;pt>0&&St.push(`${Re[Ge]} ${pt}\u4EF6`)}const Vt=St.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${St.
join("\u3001")}\uFF09`:"";H&&(H.textContent=`\u5B8C\u4E86: ${fe}${Vt}`,H.classList.remove("hidden","\
text-red-300"),H.classList.add("text-emerald-300")),oe({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\
\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&st(),Ce=!0}}catch(q){if(P.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(P.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),P.cancelRequested||q&&q.name==="AbortError")return;const N=q&&q.message?q.
message:"",ee=N==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
N||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";oe({progress:0,phase:"\
failed",message:ee}),H&&(H.textContent=ee,H.classList.remove("hidden","text-emerald-300"),H.classList.
add("text-red-300")),showToast(ee,"error",!0)}finally{P.stopped=!0,await se.catch(()=>null),X(P)}});
const ze=get("account-dedupe-btn"),Je=get("account-dedupe-result"),rt=o((d,m=!1)=>{Je&&(Je.textContent=
d,Je.classList.remove("hidden"),Je.classList.toggle("text-red-300",!!m),Je.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");ze&&(ze.onclick=async()=>{const d=o(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),h=await m.json().catch(()=>null);if(!m.ok||!h)throw new Error(h&&h.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!h.has_duplicates){rt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const P of[
"chats","gems","files","feedback","diagnostics"]){const H=Number(h.duplicates&&h.duplicates[P])||0;H>
0&&b.push(`${x[P]} ${H}\u4EF6`)}const S=Number(h.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${h.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${h.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const T=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),M=await T.json().catch(()=>null);if(!T.ok||!M)throw new Error(
M&&M.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const $=[];for(const P of["chats","gems","files","feedback","diagnostics"]){const H=Number(M.removed&&
M.removed[P])||0;H>0&&$.push(`${x[P]} ${H}\u4EF6`)}const j=Number(M.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${M.kept_referenced_files}\u4EF6\uFF09`:"";rt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${$.join("\u3001")||"0\u4EF6"}${j}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!ze.disabled){ze.disabled=!0,rt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){rt(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{ze.disabled=!1}}});const Ve=get("site-cache-usage-refresh");Ve&&(Ve.onclick=()=>loadSiteCacheUsage());
const De=get("clear-site-cache-btn");De&&(De.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(De)});
const He=get("enc-scan-result"),Ke=o(async(d=null)=>{He&&(He.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const h=await apiFetch(
m,{cache:"no-store"}),b=await h.json();if(!h.ok){He&&(He.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=b.total||0,S=b.encrypted||0,T=b.unencrypted||0;let M=`Total: ${x} / Encrypted: ${S} /\
 Plain: ${T}`;if(b.samples&&b.samples.length){const $=b.samples.slice(0,8).map(j=>{const P=j.timestamp?
new Date(j.timestamp).toLocaleString():"";return`#${j.id} (${j.role||""}) ${P}`}).join(" / ");M+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${$}</div>`}He&&(He.innerHTML=M)}catch{He&&(He.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Kt=get("enc-scan-all");Kt&&(Kt.onclick=()=>Ke(
null));const Bt=get("enc-scan-thread");Bt&&(Bt.onclick=()=>currentThreadId?Ke(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const at=get("admin-enc-\
list");let Ft=null,Oe=!1;const qe=o(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThread\
EncryptedFromMessages"),tt=o(()=>{Ft=qe(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Xt=o(
async(d,m,{confirmPrompt:h=!0,reloadCurrent:b=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(h&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Oe=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
T=await S.json().catch(()=>({}));return S.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${T.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Ft=!!m,b&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),at&&await ht(),!0):
(showToast(T.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Oe=!1}},"setAdminThreadEncry\
ption"),cn=o(d=>{if(!at)return;const m=d.threads||[];if(!m.length){at.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
at.innerHTML=m.map(h=>{const b=h.encrypted_count>0?"enc":"plain",x=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",T=h.
updated_at?new Date(h.updated_at).toLocaleString():"",M=escapeHtml(String(h.thread_id)),$=currentThreadId&&
String(currentThreadId)===String(h.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(h.title||
"")}">${escapeHtml(h.title||"(\u7121\u984C)")}${$?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${T} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${h.
message_count} / \u6697\u53F7\u5316: ${h.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${M}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${M}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),ht=o(async()=>{if(at){at.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){at.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(cn(m),currentThreadId&&Array.isArray(m.threads)){const h=m.threads.find(b=>String(
b.thread_id)===String(currentThreadId));h&&(Ft=!!h.encrypted)}}catch{at.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>ht()),window.__loadAdminEncThreads=
ht,window.__refreshAdminThreadEncState=tt,window.__setAdminThreadEncryption=Xt;const me=get("encrypt\
ion-status-admin-toggle");me&&me.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),at&&(at.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const M=m.getAttribute("data-id");if(!M)return;typeof jt==
"function"?jt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(M)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const h=d.target.closest(".admin-enc-toggle");if(!h||Oe)return;const b=h.getAttribute("data-i\
d"),x=h.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;h.disabled=!0;
const T=h.textContent;h.textContent="\u51E6\u7406\u4E2D...";try{await Xt(b,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
h.disabled=!1,h.textContent=T,await ht()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const Se=o(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(h=>{const b=h.status||"new",x=h.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=h.created_at?new Date(h.created_at).toLocaleString():"",T=h.replied_at?new Date(h.replied_at).toLocaleString():
"",M=h.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${h.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(h.username||
"")}${x}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(S)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(b)}</di\
v>
                            <div class="text-xs text-gray-200 mt-2 whitespace-pre-wrap">${escapeHtml(
h.message||"")}</div>
                            <div class="text-[10px] text-gray-500 mt-2">BAN\u7406\u7531: ${escapeHtml(
h.ban_reason||"N/A")}</div>
                            ${h.evidence?`<details class="mt-2"><summary class="text-[10px] text-cya\
n-300 cursor-pointer">\u4E0D\u5BE9\u306A\u5C65\u6B74\uFF08\u8A18\u9332\uFF09\u3092\u8868\u793A</summary><pre class="mt-1 text-[10px] text-gray-300 whitespace-pr\
e-wrap bg-gray-950/70 border border-gray-700 rounded p-2 max-h-60 overflow-auto">${escapeHtml(h.evidence)}\
</pre></details>`:""}
                            <div class="mt-3">
                                <label class="text-[10px] text-gray-400">\u7BA1\u7406\u8005\u8FD4\u4FE1</label>
                                <textarea class="ban-appeal-reply w-full mt-1 bg-gray-800 border bor\
der-gray-700 rounded px-2 py-1 text-[11px] text-gray-100" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(
M)}</textarea>
                                ${M?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
T)}</div>`:""}
                            </div>
                            <div class="mt-2 flex flex-wrap gap-2">
                                <button class="ban-appeal-mark text-[10px] px-2 py-1 bg-gray-700 hov\
er:bg-gray-600 rounded" data-id="${h.id}">\u65E2\u8AAD</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-blue-700 h\
over:bg-blue-600 rounded" data-id="${h.id}" data-status="in_review">\u5BFE\u5FDC\u4E2D</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-green-700 \
hover:bg-green-600 rounded" data-id="${h.id}" data-status="resolved">\u5B8C\u4E86</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-red-700 ho\
ver:bg-red-600 rounded" data-id="${h.id}" data-status="rejected">\u5374\u4E0B</button>
                                <button class="ban-appeal-reply-send text-[10px] px-2 py-1 bg-sky-70\
0 hover:bg-sky-600 rounded" data-id="${h.id}">\u8FD4\u4FE1\u9001\u4FE1</button>
                                <button class="ban-appeal-block text-[10px] px-2 py-1 bg-rose-700 ho\
ver:bg-rose-600 rounded" data-id="${h.id}">\u7533\u3057\u7ACB\u3066\u30D6\u30ED\u30C3\u30AF</button>
                            </div>
                        </div>
                    `}).join("")}},"renderBanAppeals"),Xe=o(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const h=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!h.
ok)return;const x=(await h.json()).unread_count||0;m.textContent=String(x),d&&x>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${x}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Ye=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const h=await m.json();Se(h.items||[]),
await Xe(!1)}catch{}}},"loadBanAppeals"),nt=o(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await Ye()}catch{}},"markBanAppealsRead"),gt=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await Ye()}catch{}},"updateBanAppealStatus"),At=o(()=>{
const d=get("tab-general");if(!d||get("temp-chat-settings-card"))return;const m=document.createElement(
"div");m.id="temp-chat-settings-card",m.className="settings-card",m.innerHTML=`
                    <h3 class="settings-card-title">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">\u5207\u65AD\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\uFF08\u79D2\uFF09</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}\
" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rou\
nded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3067\u30DA\u30FC\u30B8\u306E\u8868\u793A/\u63A5\u7D9A\u304C\u9014\u5207\u308C\u305F\u72B6\u614B\u304C\u3053\u306E\u79D2\u6570\u3092\u8D85\u3048\u308B\u3068\u3001\u81EA\u52D5\u524A\
\u9664\u3055\u308C\u307E\u3059\u3002</div>
                    </div>
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),On=o(()=>{const d=get("set-st\
t-model");if(!d||get("set-llm-transcribe-prompt"))return;const m=d.closest(".space-y-2");if(!m)return;
const h=document.createElement("div");h.className="pt-2 border-t border-gray-700/60",h.innerHTML=`
                    <label class="text-xs text-gray-500 block">LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\uFF08LLM\u65B9\u5F0F\uFF09</label>
                    <textarea id="set-llm-transcribe-prompt" class="w-full h-24 bg-gray-800 border b\
order-gray-600 rounded px-2 py-2 text-xs text-white mt-1" placeholder=""></textarea>
                    <div class="flex items-center gap-2 mt-2">
                        <button type="button" id="reset-llm-transcribe-prompt" class="bg-gray-700 ho\
ver:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">\u65E2\u5B9A\u306B\u623B\u3059</button>
                        <div class="text-[10px] text-gray-500">LLM\u65B9\u5F0F\u306E\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u6642\u306E\u307F\u4F7F\u7528\u3002\u7A7A\u6B04\u3067\u4FDD\u5B58\u3059\u308B\u3068\u65E2\u5B9A\u6587\u9762\u3092\u4F7F\u3044\u307E\u3059\
\uFF08\u7121\u97F3\u6642\u306E\u5B89\u5168\u30AC\u30FC\u30C9\u306F\u5225\u9014\u81EA\u52D5\u4ED8\u4E0E\uFF09\u3002</div>
                    </div>
                `,m.appendChild(h);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const x=get(
"set-llm-transcribe-prompt");x&&(x.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),dn=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
{key:"gemini_local_python",label:"Gemini \u97F3\u58F0/\u52D5\u753B/PDF/DOCX + Python\uFF08\u30ED\u30FC\u30AB\u30EB\u5B9F\u884C\uFF09"},
{key:"grok_search",label:"Search\u88DC\u52A9\uFF08Grok\uFF09"},{key:"openai_search",label:"Search\u88DC\u52A9\uFF08\
OpenAI/xAI Responses\uFF09"},{key:"marker",label:"Marker\u7DE8\u96C6\u6642"},{key:"attachment_names",
label:"\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u540D\uFF08LLM\u5165\u529B\u6642\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{\
attachment_names}} / {{attachment_count}}"},{key:"mathjax",label:"MathJax\uFF08LaTeX\u6570\u5F0F\uFF09"},
{key:"image_analysis",label:"\u753B\u50CF\u89E3\u6790\uFF08Vision Model\u6307\u793A\u6587\uFF09"},{key:"\
mcp",label:"MCP\uFF08\u5916\u90E8\u30C4\u30FC\u30EB\u63A5\u7D9A\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{mcp_tools}}\uFF08\u63A5\
\u7D9A\u4E2D\u306EMCP\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u5165\u308A\u307E\u3059\uFF09",mcpLocked:!0}];
window.buildAutoSystemPromptRows=(d,m=!1)=>{const h=m?"w-full h-14 bg-gray-950 border border-gray-70\
0 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 rounded p-2\
 text-xs text-gray-200";return dn.map(b=>{const x=b.mcpLocked===!0,S=x?'<div class="text-[10px] text\
-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>':
"",T=x?`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${b.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${x?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${T}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${d}-auto-sys-${b.key}-text" class="${h}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${S}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{dn.forEach(h=>{const b=m&&
typeof m=="object"?m[h.key]||{}:{},x=get(`${d}-auto-sys-${h.key}-enabled`),S=get(`${d}-auto-sys-${h.
key}-text`);x&&(h.mcpLocked===!0?x.disabled=!0:x.checked=b.enabled!==!1),S&&(S.value=b.text||"",S.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Nn=o((d,m=null)=>{if(m){const h=get(m);h&&(h.checked=!0)}dn.forEach(h=>{const b=get(`${d}-auto\
-sys-${h.key}-enabled`),x=get(`${d}-auto-sys-${h.key}-text`);if(b&&(h.mcpLocked!==!0?b.checked=!0:b.
disabled=!0),x){const S=x.placeholder||"";x.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),ni=o(d=>{const m={};return dn.forEach(h=>{const b=get(`${d}\
-auto-sys-${h.key}-enabled`),x=get(`${d}-auto-sys-${h.key}-text`);m[h.key]={enabled:h.mcpLocked===!0?
!0:b?b.checked:!0,text:x?x.value:""}}),m},"collectAutoSystemPromptConfigFromForm");window.ensureAutoSystemPromptSettingsCard=
()=>{const d=get("set-global-sys-prompt-enabled"),m=d?d.closest(".space-y-4"):null;if(!m||get("auto-\
sys-prompt-settings"))return;const h=document.createElement("div");h.id="auto-sys-prompt-settings",h.
className="border-t border-gray-700 pt-3",h.innerHTML=`
                    <div class="flex items-center justify-between mb-2">
                        <label class="text-xs text-gray-500 block">\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\uFF08\u30E6\u30FC\u30B6\u30FC\u5358\u4F4D\uFF09</label>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-set-auto-sys-prompt-defaults" class="bg-\
gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">\u65E2\u5B9A\u306B\u623B\u3059</butt\
on>
                            <label class="flex items-center gap-1 text-[10px] text-gray-400">
                                <input type="checkbox" id="set-apply-auto-sys-prompt-notices" class=\
"accent-yellow-500 w-3 h-3">
                                <span>\u5168\u4F53\u9069\u7528</span>
                            </label>
                        </div>
                    </div>
                    <div id="set-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows(
"set",!1)}</div>
                    <div class="text-[10px] text-gray-500 mt-2">\u5404\u6587\u9762\u306F\u30E6\u30FC\u30B6\u30FC\u5358\u4F4D\u3067\u7DE8\u96C6\u3055\u308C\u307E\u3059\u3002\u7A7A\u6B04\u3067\u4FDD\u5B58\u3059\u308B\u3068\u65E2\u5B9A\u6587\u9762\u306B\u623B\u308A\u307E\u3059\u3002\
</div>
                `,m.appendChild(h)},window.ensureThreadAutoSystemPromptCard=()=>{const d=get("thread\
-global-sys-prompt"),m=d?d.closest(".space-y-3"):null;if(!m||get("thread-auto-sys-prompt-settings"))
return;const h=document.createElement("div");h.id="thread-auto-sys-prompt-settings",h.className="bor\
der-t border-gray-700 pt-3",h.innerHTML=`
                    <div class="flex items-center justify-between mb-2">
                        <div class="text-xs text-gray-400">\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\uFF08\u30E6\u30FC\u30B6\u30FC\u5358\u4F4D\uFF09</div>
                        <div class="flex items-center gap-2">
                            <button type="button" id="reset-thread-auto-sys-prompt-defaults" class="\
bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">\u65E2\u5B9A\u306B\u623B\u3059</b\
utton>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500">
                                <input type="checkbox" id="thread-apply-auto-sys-prompt-notices" cla\
ss="accent-yellow-500 w-3 h-3">
                                <span>\u5168\u4F53\u9069\u7528</span>
                            </label>
                        </div>
                    </div>
                    <div id="thread-auto-sys-prompt-items" class="space-y-2">${window.buildAutoSystemPromptRows(
"thread",!0)}</div>
                `,m.appendChild(h)},At(),On(),ue();const ii=o(()=>{const d=get("set-default-model");
if(!d)return;const m=d.value;d.innerHTML="",MODELS.forEach(b=>{const x=document.createElement("optgr\
oup");x.label=b.category,(b.items||[]).forEach(S=>{const T=document.createElement("option");T.value=
S.id,T.textContent=S.name,x.appendChild(T)}),d.appendChild(x)});const h=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";h&&Array.from(d.options).some(b=>b.value===h)&&(d.value=h)},"po\
pulateDefaultModelOptions"),si=o(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(b=>{const x=(b.items||[]).filter(T=>{const M=(T.id||"").toLowerCase();
return M.startsWith("gemini-")||M.startsWith("gpt-4o")||M.startsWith("claude-")||M.startsWith("grok-\
3")});if(x.length===0)return;const S=document.createElement("optgroup");S.label=b.category,x.forEach(
T=>{const M=document.createElement("option");M.value=T.id,M.textContent=T.name+" \u2605",S.appendChild(
M)}),d.appendChild(S)});const h=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
"gemini-3-flash-preview";h&&Array.from(d.options).some(b=>b.value===h)&&(d.value=h)},"populateDefaul\
tVisionModelOptions"),Rn=o(d=>{if(!d)return;cacheUserSettings(d);const m=get("app-global-sys-prompt-\
preview");m&&(m.value=d.global_system_prompt_effective||"");const h=get("app-global-sys-prompt-previ\
ew-status");h&&(d.global_system_prompt_enabled===!1?h.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
d.global_system_prompt_uses_time_fallback?h.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
h.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("sys-prompt-text")&&(get("sys-prompt-text").value=d.system_prompt||""),get("set-global-sys-promp\
t-enabled")&&(get("set-global-sys-prompt-enabled").checked=d.system_prompt_enabled!==!1),window.ensureAutoSystemPromptSettingsCard(),
get("set-apply-global-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=d.apply_global_system_prompt!==
!1),get("set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=d.apply_auto_system_prompt_notices!==
!1),window.applyAutoSystemPromptConfigToForm("set",d.auto_system_prompt_notices_config||{}),get("set\
-latency-metrics")&&(get("set-latency-metrics").checked=d.enable_latency_metrics===!0),get("set-clie\
nt-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings modal sync"),get("\
set-openai")&&(get("set-openai").value=d.openai_key||""),get("set-gemini")&&(get("set-gemini").value=
d.gemini_key||""),get("set-deepseek")&&(get("set-deepseek").value=d.deepseek_key||""),get("set-kimi")&&
(get("set-kimi").value=d.kimi_key||""),get("set-mistral")&&(get("set-mistral").value=d.mistral_key||
""),get("set-anthropic")&&(get("set-anthropic").value=d.anthropic_key||""),get("set-gemini-backend")&&
(get("set-gemini-backend").value=normalizeGeminiBackend(d.gemini_backend||"gemini_api")),get("set-ge\
mini-vertex-project")&&(get("set-gemini-vertex-project").value=d.gemini_vertex_project||""),get("set\
-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=d.gemini_vertex_location||"globa\
l"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-gemini\
-vertex-credentials-json").value=d.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),get("se\
t-admin-api-key-mode")&&(get("set-admin-api-key-mode").value=normalizeAdminApiKeyMode(d.admin_api_key_mode||
"env_fallback")),syncAdminApiKeyModeUi(),get("set-xai")&&(get("set-xai").value=d.xai_key||""),get("s\
et-google-key")&&(get("set-google-key").value=d.google_key||""),get("set-google-project")&&(get("set\
-google-project").value=d.google_project||""),modelApiKeyMap=normalizeModelApiKeyMap(d.model_api_keys||
{}),syncModelApiKeyModelOptions(),renderModelApiKeyList(),setModelApiKeyPanelOpen(!1),get("set-mic-t\
ranscribe-mode")&&(get("set-mic-transcribe-mode").value=d.mic_transcribe_mode||"stt_api"),get("set-s\
tt-model")&&(get("set-stt-model").value=d.stt_model||"gpt-4o-mini-transcribe"),get("set-llm-transcri\
be-prompt")&&(get("set-llm-transcribe-prompt").value=d.llm_transcribe_prompt||"",get("set-llm-transc\
ribe-prompt").placeholder=d.llm_transcribe_prompt_default||""),syncRichPastePromptPreferencesUi(d),updateGoogleLinkUI(
d),updateMinashinLinkUI(d),get("set-enter-to-send")&&(get("set-enter-to-send").checked=!!d.enter_to_send),
writePromptBarModeToForm(!!d.compact_prompt_mode,!!d.minimal_prompt_mode),get("set-use-sw-cache")&&(get(
"set-use-sw-cache").checked=!!d.use_sw_cache),get("set-clear-cache-on-version-update")&&(get("set-cl\
ear-cache-on-version-update").checked=!!d.clear_cache_on_version_update),get("set-liquid-glass")&&(get(
"set-liquid-glass").checked=!!d.liquid_glass_enabled),get("set-light-mode")&&(get("set-light-mode").
checked=!!d.light_mode_enabled),get("set-auto-search-links")&&(get("set-auto-search-links").checked=
d.auto_search_on_links!==!1),get("set-use-last-settings")&&(get("set-use-last-settings").checked=!!d.
use_last_chat_settings),get("set-default-model")&&(get("set-default-model").value=d.default_model||"\
gemini-3.6-flash"),get("set-default-vision-model")&&(get("set-default-vision-model").value=d.default_vision_model||
"gemini-3-flash-preview"),applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds),get("set-def\
ault-search")&&(get("set-default-search").checked=!!d.default_enable_search),get("set-default-url-co\
ntext")&&(get("set-default-url-context").checked=!!d.default_enable_url_context),get("set-default-ma\
ps")&&(get("set-default-maps").checked=!!d.default_enable_maps),get("set-default-python")&&(get("set\
-default-python").checked=!!d.default_enable_python),get("set-default-file-creation")&&(get("set-def\
ault-file-creation").checked=!!d.default_enable_file_creation),get("set-default-thinking")&&(get("se\
t-default-thinking").checked=!!d.default_enable_thinking),get("set-default-sys-prompt")&&(get("set-d\
efault-sys-prompt").checked=!!d.default_enable_system_prompt),get("set-default-mcp")&&(get("set-defa\
ult-mcp").checked=d.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-default-th\
inking-level").value=d.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get("set\
-default-thinking-budget").value=d.default_thinking_budget||4096),get("set-default-reasoning-effort")&&
(get("set-default-reasoning-effort").value=d.default_reasoning_effort||"medium"),get("set-default-sa\
fety")&&(get("set-default-safety").value=d.default_safety_setting||"default"),get("set-e2ee").checked=
d.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=d.bot_detection_enabled!==!1),get(
"set-bot-detect-global")&&(get("set-bot-detect-global").checked=d.bot_detection_global_enabled!==!1);
const b=get("bot-status");b&&(d.is_bot_banned?(b.textContent=`BAN\u4E2D: ${d.bot_ban_reason||"Bot de\
tection"}`,b.classList.remove("hidden"),b.classList.add("text-red-400")):b.classList.add("hidden")),
d&&d.theme_color?(applyThemeColor(d.theme_color,!0),syncThemeInputs(d.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=d.username);const x=get("2fa-badge"),S=get("disable-2fa-btn");d.is_2fa_enabled?
(x.innerText="ENABLED",x.classList.replace("bg-gray-700","bg-green-600"),x.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(x.innerText="DISABLED",x.classList.replace("bg\
-green-600","bg-gray-700"),x.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const T=get(
"set-passkey-only-login"),M=get("passkey-only-note"),$=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(Z($),T){T.checked=!!d.passkey_only_login;const q=$.length>0||!!d.has_webauthn;T.disabled=!q,q||
(T.checked=!1),M&&(q?M.classList.add("hidden"):M.classList.remove("hidden"))}const j=get("mig-status\
-box"),P=get("mig-progress-text"),H=get("mig-progress-bar");if((d.migration_status||"idle")==="proce\
ssing"){j.classList.remove("hidden");const q=(d.migration_progress||"").split("/");if(q.length===2){
const N=parseInt(q[0]||"0",10),ee=parseInt(q[1]||"0",10);P&&(P.innerText=`${N} / ${ee}`),H&&ee>0&&(H.
style.width=`${Math.min(100,Math.floor(N/ee*100))}%`)}}else j.classList.add("hidden"),H&&(H.style.width=
"0%"),P&&(P.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettingsFormFr\
omData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),snapshotSidebarHistory(
"settings-open-before");const d=await ensureUserSettingsSnapshot();d&&Rn(d);const m=get("search-box"),
h=m?m.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=""),filterSettings(),
ii(),si(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(()=>refreshSettingsTabsScroll()),
restoreThreadSearchValue(h,"restored-search-box-open"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(h,"rest\
ored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),U(),On(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),Xe(!0),Ye(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),Sn(),G(),O();try{loadMcpServers()}catch{}};const jt=o((d=!1)=>{snapshotSidebarHistory("\
settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),ai=o(()=>{const d=get("set-th\
eme-color"),m=get("set-theme-color-text"),h=get("theme-reset-btn"),b=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((S,T=!0)=>{const M=normalizeHex(S);M&&(applyThemeColor(M,T),syncThemeInputs(
M))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),m&&(m.addEventListener("chan\
ge",()=>{const S=normalizeHex(m.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(S,!0)}),m.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
m.blur())})),h&&(h.onclick=()=>x(THEME_DEFAULT,!0)),b.forEach(S=>{S.addEventListener("click",()=>x(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),oi=o(()=>{const d=get("reset-global-sys-prom\
pt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const m=get(
"reset-set-auto-sys-prompt-defaults");m&&(m.onclick=()=>{Nn("set","set-apply-auto-sys-prompt-notices"),
showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-thread-auto-sys-prompt-defaults");h&&(h.onclick=()=>{Nn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>jt();const Bn=get("settings-header-close");Bn&&(Bn.onclick=()=>jt());
const Dt=get("settings-search");Dt&&(Dt.addEventListener("input",filterSettings),Dt.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const h=m.querySelector(
":scope > .settings-match");h&&h.scrollIntoView({behavior:"smooth",block:"start"})}}));const Fn=get(
"settings-search-clear");Fn&&Fn.addEventListener("click",()=>{Dt&&(Dt.value="",filterSettings(),Dt.focus())}),
ai(),oi(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const vn=get("set-gemini-local-python-dialog");vn&&(vn.onchange=()=>setGeminiLocalPyDialogEnabled(vn.
checked));const jn=get("set-gemini-backend");jn&&(jn.onchange=()=>syncGeminiBackendUi());const Dn=get(
"set-admin-api-key-mode");Dn&&(Dn.onchange=()=>syncAdminApiKeyModeUi());const wn=get("set-temp-chat-\
timeout-seconds");wn&&(wn.onchange=()=>{applyTemporaryChatTimeoutSeconds(wn.value)});const Hn=get("s\
lash-command-cancel-btn");Hn&&(Hn.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const d=get("set-username"),m=get("set-password"),h=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:ni(
"set"),theme_color:normalizeHex(get("set-theme-color-text")?get("set-theme-color-text").value:"")||THEME_DEFAULT,
light_mode_enabled:get("set-light-mode")?get("set-light-mode").checked:!1,mic_transcribe_mode:get("s\
et-mic-transcribe-mode")?get("set-mic-transcribe-mode").value:"stt_api",stt_model:get("set-stt-model")?
get("set-stt-model").value:null,llm_transcribe_prompt:get("set-llm-transcribe-prompt")?get("set-llm-\
transcribe-prompt").value:"",enter_to_send:get("set-enter-to-send")?get("set-enter-to-send").checked:
!1,compact_prompt_mode:h.compact_prompt_mode,minimal_prompt_mode:h.minimal_prompt_mode,use_sw_cache:get(
"set-use-sw-cache")?get("set-use-sw-cache").checked:!1,clear_cache_on_version_update:get("set-clear-\
cache-on-version-update")?get("set-clear-cache-on-version-update").checked:!1,liquid_glass_enabled:get(
"set-liquid-glass")?get("set-liquid-glass").checked:!1,auto_search_on_links:get("set-auto-search-lin\
ks")?get("set-auto-search-links").checked:!0,use_last_chat_settings:get("set-use-last-settings")?get(
"set-use-last-settings").checked:!1,voice_studio_ui:get("set-voice-studio-ui")?get("set-voice-studio\
-ui").checked:!0,default_model:get("set-default-model")?get("set-default-model").value:null,default_vision_model:get(
"set-default-vision-model")?get("set-default-vision-model").value:null,temp_chat_timeout_seconds:normalizeTemporaryChatTimeoutSeconds(
get("set-temp-chat-timeout-seconds")?get("set-temp-chat-timeout-seconds").value:temporaryChatTimeoutSeconds),
default_enable_search:get("set-default-search")?get("set-default-search").checked:!1,default_enable_url_context:get(
"set-default-url-context")?get("set-default-url-context").checked:!1,default_enable_maps:get("set-de\
fault-maps")?get("set-default-maps").checked:!1,default_enable_python:get("set-default-python")?get(
"set-default-python").checked:!1,default_enable_file_creation:get("set-default-file-creation")?get("\
set-default-file-creation").checked:!1,default_enable_thinking:get("set-default-thinking")?get("set-\
default-thinking").checked:!1,default_enable_mcp:get("set-default-mcp")?get("set-default-mcp").checked:
!0,default_thinking_level:get("set-default-thinking-level")?get("set-default-thinking-level").value:
null,default_thinking_budget:get("set-default-thinking-budget")?get("set-default-thinking-budget").value:
null,default_reasoning_effort:get("set-default-reasoning-effort")?get("set-default-reasoning-effort").
value:null,default_enable_system_prompt:get("set-default-sys-prompt")?get("set-default-sys-prompt").
checked:!1,default_safety_setting:get("set-default-safety")?get("set-default-safety").value:null,enable_latency_metrics:get(
"set-latency-metrics")?get("set-latency-metrics").checked:!1,enable_client_debug_log:get("set-client\
-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-only-login")?
get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-google")?get("se\
t-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set-default-2fa\
-method").value:"totp",new_username:d?d.value:null,new_password:m?m.value:null},x=get("set-e2ee")?get(
"set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
x!==S&&(b.enable_e2ee=x),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
(b.gemini_key=get("set-gemini").value),get("set-deepseek")&&(b.deepseek_key=get("set-deepseek").value),
get("set-kimi")&&(b.kimi_key=get("set-kimi").value),get("set-mistral")&&(b.mistral_key=get("set-mist\
ral").value),get("set-anthropic")&&(b.anthropic_key=get("set-anthropic").value),b.model_api_keys=normalizeModelApiKeyMap(
modelApiKeyMap),get("set-gemini-backend")&&(b.gemini_backend=normalizeGeminiBackend(get("set-gemini-\
backend").value)),get("set-gemini-vertex-project")&&(b.gemini_vertex_project=get("set-gemini-vertex-\
project").value),get("set-gemini-vertex-location")&&(b.gemini_vertex_location=get("set-gemini-vertex\
-location").value),get("set-gemini-vertex-credentials-json")&&(b.gemini_vertex_credentials_json=get(
"set-gemini-vertex-credentials-json").value),get("set-xai")&&(b.xai_key=get("set-xai").value),get("s\
et-google-key")&&(b.google_key=get("set-google-key").value),get("set-google-project")&&(b.google_project=
get("set-google-project").value),get("set-admin-api-key-mode")&&(b.admin_api_key_mode=normalizeAdminApiKeyMode(
get("set-admin-api-key-mode").value)),get("set-bot-detect")&&(b.bot_detection_enabled=get("set-bot-d\
etect").checked),get("set-bot-detect-global")&&(b.bot_detection_global_enabled=get("set-bot-detect-g\
lobal").checked);const T=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(b)});if(T.ok){let M="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const H=await T.json();H&&H.message&&(M=H.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),jt();
const $=currentUsername,j=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;
const P=useSwCache;useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=
!!b.clear_cache_on_version_update),compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,
voiceStudioUiEnabled=b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,
applyThemeColor(b.theme_color,!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),
applyLiquidGlassMode(b.liquid_glass_enabled),applyAdaptiveBlurPreference(get("set-background-blur-mo\
de")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),minimalPromptMode?setMinimalPromptMode(
!0):setCompactPromptMode(compactPromptMode),updateStsUi(),P!==useSwCache&&applyCacheMode(useSwCache,
{forceCleanup:!useSwCache}),showToast(M,"success"),syncClientDebugLogToggle(b.enable_client_debug_log,
"settings saved"),b.new_username&&b.new_username!==$?setTimeout(()=>location.reload(),1e3):b.new_password&&
showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let M={};try{M=await T.json()}catch{}showToast(M.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const m=get("2fa-badge");m&&(m.innerText="D\
ISABLED",m.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const d=get("bot-unban-username"),m=d?d.value.trim():
"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"single"})}),b=await h.json(),x=get("bot-unban-result");if(h.
ok&&b&&b.status==="ok")x&&(x.textContent=`${m} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),b=await h.json(),x=get("bot-unban-result");if(h.
ok&&b&&b.status==="ok")x&&(x.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const h=o(W=>{m&&(m.textContent=W)},"setBox"),b=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((W,le)=>!W||!le||le<=0?0:W*8/(le/1e3)/1e3/1e3,"toMbps"),S=o(W=>Number.isFinite(
W)?`${W.toFixed(0)} ms`:"-","fmtMs"),T=o(W=>Number.isFinite(W)?`${W.toFixed(W>=100?0:1)} Mbps`:"-","\
fmtMbps"),M=o(async(W,le)=>{const Ae=await W.json().catch(()=>({}));return Ae&&Ae.error?Ae.error:le},
"parseErr"),$=[];h("\u6E2C\u5B9A\u4E2D... ping");for(let W=0;W<4;W++){const le=performance.now(),Ae=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),Y=performance.now();if(!Ae.ok)throw new Error(await M(
Ae,"ping_failed"));await Ae.json().catch(()=>({})),$.push(Y-le)}const j=$.reduce((W,le)=>W+le,0)/Math.
max(1,$.length),P=Math.min(...$),H=o(async W=>{const le=performance.now(),Ae=await apiFetch(`/api/sp\
eedtest/download?bytes=${W}&_=${b()}`,{cache:"no-store"});if(!Ae.ok)throw new Error(await M(Ae,"down\
load_failed"));const Y=await Ae.arrayBuffer(),ce=performance.now();return{bytes:Y.byteLength||W,ms:ce-
le,mbps:x(Y.byteLength||W,ce-le)}},"runDownload");h(`\u6E2C\u5B9A\u4E2D... ping ${S(j)}
\u6E2C\u5B9A\u4E2D... download`);const se=[];for(const W of[2*1024*1024,8*1024*1024])se.push(await H(
W)),h(`\u6E2C\u5B9A\u4E2D... ping ${S(j)}
download ${T(Math.max(...se.map(le=>le.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const q=Math.max(...se.map(W=>W.mbps)),N=o(async W=>{const le=new Uint8Array(
W),Ae=performance.now(),Y=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"C\
ontent-Type":"application/octet-stream"},body:le,cache:"no-store"}),ce=performance.now();if(!Y.ok)throw new Error(
await M(Y,"upload_failed"));const Ce=await Y.json().catch(()=>({})),$e=Number(Ce.bytes_received||W)||
W;return{bytes:$e,ms:ce-Ae,mbps:x($e,ce-Ae),serverMs:Number(Ce.server_elapsed_ms||0)||0}},"runUpload"),
ee=[];for(const W of[1*1024*1024,4*1024*1024])ee.push(await N(W));const Q=Math.max(...ee.map(W=>W.mbps)),
xe=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",`Ping (avg/m\
in): ${S(j)} / ${S(P)}`,`Download (best): ${T(q)}`,`Upload (best): ${T(Q)}`,`Download runs: ${se.map(
W=>`${Math.round(W.bytes/1024/1024)}MB=${T(W.mbps)}`).join(", ")}`,`Upload runs: ${ee.map(W=>`${Math.
round(W.bytes/1024/1024)}MB=${T(W.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\
\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];h(xe.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(h){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${h&&h.message?h.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Ye()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>nt()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const h=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){h&&await nt([Number(h)]);return}if(m.classList.contains(
"ban-appeal-status")){const b=m.getAttribute("data-status");h&&b&&await gt({id:Number(h),status:b});
return}if(m.classList.contains("ban-appeal-reply-send")){const b=m.closest("[data-appeal-id]"),x=b?b.
querySelector(".ban-appeal-reply"):null,S=x?x.value:"";h&&await gt({id:Number(h),admin_reply:S});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";h&&await gt({id:Number(
h),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const d=get("photo-input");d&&d.click()}),get("upload-clear-b\
tn")&&(get("upload-clear-btn").onclick=()=>{resetUploadState()}),get("marker-modal-close")&&(get("ma\
rker-modal-close").onclick=()=>{closeMarkerModal(),markerState.row=null}),get("marker-tool-draw")&&(get(
"marker-tool-draw").onclick=()=>setMarkerMode("draw")),get("marker-tool-mosaic")&&(get("marker-tool-\
mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool-crop")&&(get("marker-tool-crop").onclick=
()=>setMarkerMode("crop"));const xn=get("marker-color-picker");xn&&(xn.oninput=d=>setMarkerColor(d.target.
value),xn.onchange=d=>setMarkerColor(d.target.value));const kn=get("marker-opacity");kn&&(kn.oninput=
d=>setMarkerOpacity(d.target.value),kn.onchange=d=>setMarkerOpacity(d.target.value));const un=get("m\
arker-opacity-number");un&&(un.onchange=d=>setMarkerOpacity(d.target.value),un.onblur=d=>setMarkerOpacity(
d.target.value),un.onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(d.target.value),d.target.blur())}),
document.querySelectorAll("#marker-toolbar .marker-color-chip[data-marker-color]").forEach(d=>{d.onclick=
()=>setMarkerColor(d.getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-vie\
w-reset").onclick=()=>resetMarkerTransform()),get("marker-crop-reset")&&(get("marker-crop-reset").onclick=
()=>clearCropRect()),get("marker-undo")&&(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("ma\
rker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),get("marker-save")&&(get("marker\
-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),initCropCanvas(),
window.addEventListener("resize",()=>{const d=get("marker-modal");!d||d.classList.contains("hidden")||
(applyMarkerTransform(),renderCropOverlay())});const ri=o(()=>{const d=get("upload-modal");return!!(d&&
!d.classList.contains("hidden"))},"isUploadModalOpen"),Ht=get("drop-overlay");let Yt=0;const li=o(()=>{
ri()||Ht&&(Ht.classList.remove("hidden"),Ht.classList.add("flex"))},"showDropOverlay"),Qt=o(()=>{Yt=
0,Ht&&(Ht.classList.add("hidden"),Ht.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
Qt;const wt=get("upload-dropzone");wt&&(wt.addEventListener("dragover",d=>{d.preventDefault(),wt.classList.
add("dragover")}),wt.addEventListener("dragleave",()=>{wt.classList.remove("dragover")}),wt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),wt.classList.remove("dragover"),Qt();const m=d.dataTransfer?
d.dataTransfer.files:null;m&&m.length&&handleFiles(m)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(Yt+=1,li())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(Yt=Math.max(0,Yt-1),(Yt===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&Qt())}),window.addEventListener(
"dragend",()=>{Qt()}),window.addEventListener("drop",d=>{Qt(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(wt&&wt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const qn=get("bot-admin-modal"),ci=o(d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML=
"",!d||!d.length){m.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((h,b)=>{const x=!!h.is_bot_banned,S=h.bot_detection_enabled!==!1,T=document.createElement(
"div");T.className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs m\
odel-list-animate",T.style.animationDelay=`${Math.min(b,12)*.02}s`,T.innerHTML=`
                        <div class="flex-1">
                            <div class="text-gray-200 font-bold">${escapeHtml(h.username)}</div>
                            <div class="text-[10px] text-gray-500">${x?"BAN\u4E2D":"\u6B63\u5E38"} ${h.
bot_ban_reason?" / "+escapeHtml(h.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(h.username)}" data-enabled="${S?"1":"0"}">${S?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${x?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(h.username)}" data-bann\
ed="${x?"1":"0"}">${x?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${x?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(h.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-\
2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(h.username)}">\u524A\u9664</button>\

                    `,m.appendChild(T)})}},"renderBotUsers"),Zt=o(async(d="")=>{const m=get("bot-adm\
in-list");m&&(m.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const h=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),b=await h.json();h.ok&&b&&b.users?ci(b.users):(m&&(m.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{m&&(m.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),_n=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||qn))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await Zt(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=_n,window.closeBotAdminModal=(d=!1)=>{(get(
"bot-admin-modal")||qn)&&hideModal("bot-admin-modal"),!d&&location.pathname==="/admin-bots"&&history.
back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{_n()}),get("bot-admin-close")&&(get(
"bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-admin-search-btn")&&(get("bot-admin-se\
arch-btn").onclick=async()=>{await Zt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}),
get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=async()=>{await Zt("")}),get("bo\
t-admin-search")&&get("bot-admin-search").addEventListener("keydown",async d=>{d.key==="Enter"&&await Zt(
get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get("bot-admin-list").onclick=async d=>{
const m=d.target.closest("button");if(!m)return;const h=m.getAttribute("data-user");if(!h)return;let b;
if(m.classList.contains("bot-toggle-detect")){const x=m.getAttribute("data-enabled")!=="1";b=await apiFetch(
"/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,
action:"toggle_detection",enabled:x})})}else if(m.classList.contains("bot-toggle-ban"))if(m.getAttribute(
"data-banned")==="1")b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"appl\
ication/json"},body:JSON.stringify({username:h,action:"unban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))return;b=await apiFetch("/api/bot/update",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,action:"ban",reason:"Adm\
in ban"})})}else if(m.classList.contains("bot-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))return;b=await apiFetch("/a\
pi/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,
action:"unban_linked"})})}else if(m.classList.contains("bot-delete-account")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"delete_account"})})}if(b){if(b.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(b.ok){if(m.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),h===currentUsername)){location.href="/";return}}else{
let x={};try{x=await b.json()}catch{}showToast(x.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}await Zt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}});const qt={"\
/settings":{id:"settings-modal",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload\
-modal",open:o(()=>openUploadModal(),"open")},"/library":{id:"lib-modal",open:o(()=>{Yn(!1),showModal(
"lib-modal"),loadLibraryFiles()},"open")},"/history":{id:"history-modal",open:o(()=>window.showHistoryModal(),
"open")},"/branch":{id:"branch-modal",open:o(()=>window.showBranchModal(),"open")},"/batch":{id:"bat\
ch-modal",open:o(()=>window.showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:o(()=>openRichPasteModal(),
"open")},"/camera":{id:"camera-capture-modal",open:o(()=>openCameraCaptureModal(),"open")},"/edit-im\
age":{id:"marker-modal",open:o(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:o(()=>window.
openThreadModal(),"open")},"/model":{id:"model-modal",open:o(()=>openModelModal(),"open")},"/token-d\
etails":{id:"token-detail-modal",open:o(()=>showTokenDetailModal(),"open")},"/encryption-status":{id:"\
encryption-status-modal",open:o(()=>showEncryptionStatusModal(),"open")},"/python-execution":{id:"py\
thon-exec-modal",open:o(()=>showPythonExecDetailModal(),"open")},"/gem":{id:"gem-modal",open:o(()=>{
editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>C\
reate New Gem',showModal("gem-modal")},"open")},"/compression":{id:"compression-modal",open:o(()=>window.
openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:o(()=>_n(),"open")}},Gn=o((d,m=!1)=>{
switch(d){case"settings-modal":jt(m);break;case"upload-modal":closeUploadModal(m);break;case"camera-\
capture-modal":closeCameraCaptureModal(m?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(m);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(m);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(m);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(m);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
m);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(m);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(m);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(m);break;case"token-detail-modal":closeTokenDetail(m);break;case"encryption-s\
tatus-modal":closeEncryptionModal(m);break;case"python-exec-modal":closePythonExecDetail(m);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(m);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(m);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
m);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(d);break;case"\
version-update-modal":const h=localStorage.getItem("app_version")||"";h&&localStorage.setItem("versi\
on_notified",h),hideModal(d);break;default:hideModal(d);break}},"closeModalById");window.addEventListener(
"popstate",d=>{let m=!1;Object.values(qt).forEach(x=>{const S=get(x.id);S&&S.classList.contains("mod\
al-open")&&location.pathname!==Object.keys(qt).find(T=>qt[T].id===x.id)&&(Gn(x.id,!0),m=!0)});const h=location.
pathname.match(/^\/c\/(.+)$/);if(h){const x=decodeURIComponent(h[1]);String(currentThreadId)!==String(
x)&&loadMessages(x,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});
const b=qt[location.pathname];if(b){const x=get(b.id);x&&!x.classList.contains("modal-open")&&b.open()}});
const Un=location.pathname;qt[Un]&&(history.replaceState({},"","/"),setTimeout(()=>qt[Un].open(),500)),
get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const d=get("easy-login-mi\
ns"),m=d?parseInt(d.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${m}\u5206\u9593\u6709\u52B9\
\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const b=await(await apiFetch("/api/easy_login",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:m})})).json();b&&b.temp_password?
(get("easy-login-code").textContent=b.temp_password,get("easy-login-exp").textContent=b.expires_at||
"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const m=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(m&&m.cancelled){const h=get("easy-login-r\
esult");h&&h.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("fb-submit").onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-mess\
age").value.trim();if(!m){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}await apiFetch("/api/feedback",{method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({title:d,message:m})}),get("fb-title").value="",get("fb-message").value=
"",Sn()};async function Sn(){const m=await(await apiFetch("/api/feedback?all=1")).json(),h=get("fb-l\
ist");h.innerHTML="",(m.items||[]).filter(S=>!m.is_admin||S.user_id===void 0||S.user_id===null||!0).
forEach(S=>{if(m.is_admin)return;const T=document.createElement("div");T.className="p-2 rounded bord\
er border-gray-700 bg-gray-800/50",T.innerHTML=`<div class="text-[11px] text-gray-400">${S.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(S.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
S.status)}</div>${S.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(S.
admin_reply)}</div>`:""}`,h.appendChild(T)});const b=get("fb-admin-panel"),x=get("fb-admin-list");m.
is_admin?(b.classList.remove("hidden"),x.innerHTML="",(m.items||[]).forEach(S=>{const T=document.createElement(
"div");T.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",T.innerHTML=`
                            <div class="text-[11px] text-gray-400">#${S.id} / user:${S.user_id} / ${S.
created_at}</div>
                            <div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div>
                            <div class="flex items-center gap-2">
                                <select class="fb-status bg-gray-900 border border-gray-700 rounded \
px-2 py-1 text-xs text-white">
                                    <option value="new">new</option>
                                    <option value="in_review">in_review</option>
                                    <option value="replied">replied</option>
                                    <option value="rejected">rejected</option>
                                    <option value="resolved">resolved</option>
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3\
 py-1 rounded text-xs">\u4FDD\u5B58</button>
                            </div>
                            <textarea class="fb-reply w-full bg-gray-900 border border-gray-700 roun\
ded px-2 py-1 text-xs text-white" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(S.admin_reply||
"")}</textarea>
                        `,T.querySelector(".fb-status").value=S.status||"new",T.querySelector(".fb-s\
ave").onclick=async()=>{const M=T.querySelector(".fb-status").value,$=T.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${S.id}/update`,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({status:M,admin_reply:$})}),Sn()},x.appendChild(T)})):b.classList.add("hidd\
en")}if(o(Sn,"loadFeedback"),window.setupTOTP=async()=>{const m=await(await apiFetch("/api/2fa/totp/\
setup",{method:"POST"})).json();get("totp-qr").src=m.qr_image,get("totp-secret-disp").innerText=m.secret,
get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const d=get("totp-ver\
ify-code").value;if(!d)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),m=get("webauthn-nam\
e"),h=m?String(m.value||"").trim():"";try{d&&(d.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),x=await b.json();if(!b.ok){showToast(x.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const T=await(await ensureWebAuthnJson()).create({publicKey:x}),M=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},T,{name:h}))}),$=await M.json().catch(()=>({}));M.ok?(m&&(m.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast($.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(b){showToast(`WebAuthn Error: ${b}`,"error",!0)}finally{d&&(d.disabled=!1)}},window.
removeWebAuthnCredential=async d=>{if(!d||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const m=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:d})}),h=await m.json().catch(()=>({}));if(m.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(h.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let d;try{d=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(d.ok){location.href="/";
return}let m={};try{m=await d.json()}catch{}if(m&&m.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(m.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=d=>{if(d.isComposing)return;const m=get("prompt-input");
if(slashSuggestionsVisible){const h=get("slash-command-suggestions");if(d.key==="ArrowDown"){d.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(m.value),m.value));
return}if(d.key==="ArrowUp"){d.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(m.value),m.value));
return}if(d.key==="Enter"){d.preventDefault();const b=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(m.value),m.value));b[slashSelectedIndex]?selectSlashCommand(b[slashSelectedIndex].
id):b.length>0&&selectSlashCommand(b[0].id);return}if(d.key==="Escape"){d.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const h=m.value.trim();if(d.key==="ArrowDown"){d.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(h.substring(1));return}if(d.key==="ArrowUp"){
d.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(h.substring(1));
return}if(d.key==="Enter"){d.preventDefault();const b=h.substring(1).toLowerCase(),x=loadedGems.filter(
S=>S.name.toLowerCase().includes(b)||S.description&&S.description.toLowerCase().includes(b));x[gemSelectedIndex]?
selectGemSuggestion(x[gemSelectedIndex]):x.length>0&&selectGemSuggestion(x[0]);return}if(d.key==="Es\
cape"){d.preventDefault(),hideGemSuggestions();return}}if(d.key==="Escape"&&pendingSlashCommand){d.preventDefault(),
hidePendingSlashCommandIndicator();return}d.key==="ArrowUp"&&(m.selectionStart===0||d.ctrlKey)?promptHistory.
length>0&&(historyIndex===-1&&(tempPrompt=m.value),historyIndex<promptHistory.length-1&&(d.preventDefault(),
historyIndex++,m.value=promptHistory[historyIndex],m.dispatchEvent(new Event("input")))):d.key==="Ar\
rowDown"&&(m.selectionEnd===m.value.length||d.ctrlKey)&&historyIndex>-1&&(d.preventDefault(),historyIndex--,
historyIndex===-1?m.value=tempPrompt:m.value=promptHistory[historyIndex],m.dispatchEvent(new Event("\
input"))),enterToSend?d.key==="Enter"&&!d.shiftKey&&(d.preventDefault(),sendMessage()):(d.metaKey||d.
ctrlKey)&&d.key==="Enter"&&(d.preventDefault(),sendMessage())},get("prompt-input")&&(get("prompt-inp\
ut").addEventListener("input",function(){this.style.height="auto",this.style.height=this.scrollHeight+
"px",schedulePromptTokenEstimate(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const d=this.
value.trim();if(pendingSlashCommand)gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&
hideSlashCommandSuggestions(),lastSlashFilter=null;else if(d.startsWith("@")){const m=d.substring(1);
showGemSuggestions(m),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=null}else if(d.
startsWith("/")){const m=slashCommandSuggestionFilter(extractSlashCommandToken(d),this.value);(!slashSuggestionsVisible||
m!==lastSlashFilter)&&(lastSlashFilter=m,showSlashCommandSuggestions(m)),gemSuggestionsVisible&&hideGemSuggestions()}else
gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=
null}),get("prompt-input").addEventListener("blur",()=>{setTimeout(()=>{slashSuggestionsVisible&&hideSlashCommandSuggestions(),
gemSuggestionsVisible&&hideGemSuggestions()},150)})),get("cancel-edit-btn")&&(get("cancel-edit-btn").
onclick=cancelEdit),updatePromptPlaceholder(),aiSettingsConversation.length>0&&(pendingSlashCommand=
"settings",showPendingSlashCommandIndicator("settings")),get("search-box")&&(get("search-box").addEventListener(
"input",d=>{const m=get("search-box");if(m&&isUserInitiatedSearchInput(d))markThreadSearchUserEdited(
m);else if(m&&!m.dataset.userEdited){discardAutofilledThreadSearch("cleared-autofill-search-box-inpu\
t");return}if(isSettingsModalOpen()){snapshotSidebarHistory("ignore-search-input-settings-open");return}
clearTimeout(searchTimeout),searchTimeout=setTimeout(()=>{loadThreads(!1)},300)}),hardenThreadSearchInputs()),
get("mobile-new-chat-btn")&&(get("mobile-new-chat-btn").onclick=()=>startNewChat()),get("sts-mic-btn")&&
(get("sts-mic-btn").onclick=()=>{isStsModel()&&get("mic-btn").click()}),get("sts-cancel-btn")&&(get(
"sts-cancel-btn").onclick=()=>{isStsModel()&&ui()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async d=>{const m=(d.clipboardData||window.clipboardData).items,h=[];for(let b=0;b<m.length;b++)
if(m[b].kind==="file"){const x=m[b].getAsFile();x&&h.push(x)}h.length>0&&(d.preventDefault(),await handleFiles(
h,{openModal:!1}))}),get("rich-paste-btn")&&(get("rich-paste-btn").onclick=()=>openRichPasteModal()),
get("rich-paste-modal-close")&&(get("rich-paste-modal-close").onclick=()=>closeRichPasteModal()),get(
"rich-paste-close-btn")&&(get("rich-paste-close-btn").onclick=()=>closeRichPasteModal()),get("rich-p\
aste-focus-btn")&&(get("rich-paste-focus-btn").onclick=()=>focusRichPasteEditor()),get("rich-paste-c\
lear-btn")&&(get("rich-paste-clear-btn").onclick=()=>clearRichPasteEditor(!0)),get("rich-paste-previ\
ew-btn")&&(get("rich-paste-preview-btn").onclick=()=>openRichPastePreviewTab()),get("rich-paste-send\
-btn")&&(get("rich-paste-send-btn").onclick=()=>sendRichPasteToModel()),get("rich-paste-send-server-\
btn")&&(get("rich-paste-send-server-btn").onclick=()=>sendRichPasteToModel({serverSide:!0})),get("ri\
ch-paste-import-btn")&&(get("rich-paste-import-btn").onclick=async()=>{try{await readClipboardRichContent()||
showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u30EA\u30C3\u30C1\u30C6\u30AD\u30B9\u30C8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002Ctrl+V \u3067\u8CBC\u308A\u4ED8\u3051\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(d){const m=d&&d.message?d.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(m,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const d=get("rich-paste-capture");
d.addEventListener("paste",async m=>{const h=m.clipboardData||window.clipboardData;if(h){m.preventDefault();
try{await ingestRichPasteClipboardData(h)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),d.addEventListener("input",()=>{d.value=""})}get("chat-container").addEventListener(
"click",d=>{const m=d.target.closest("img.chat-image"),h=m?m.dataset.viewerSrc||m.currentSrc||m.src:
"";m&&h&&(d.preventDefault(),openImageViewer(h))});const en=document.querySelector(".viewer-content");
en&&(en.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),en.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),en.addEventListener("touchend",onViewerTouchEnd),en.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",d=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(d.target.id==="image-viewer"||d.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",d=>{d.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",d=>{d.key==="Escape"&&closeImageViewer()});
let it,Fe=null,pn=[],Tn=!1,mn=null,Gt=null,Et=null,tn=null,Cn=0,fn=!1,nn=null,Ut=null,kt=null,sn=null,
hn=null,$t=null,gn=null,bn=null;function zn(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
gn)&&gn.length)return gn;d.innerHTML="";const m=[];for(let h=0;h<24;h++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",m.push(b),d.appendChild(b)}return gn=m,m}o(zn,"ensureMicWaveformBars");function It(d,m="hidden"){
const h=get("mic-recording-indicator"),b=get("mic-recording-text");if(h){if(bn&&(clearTimeout(bn),bn=
null),m==="hidden"){h.classList.add("hidden");return}b&&d&&(b.innerText=d),h.classList.remove("hidde\
n"),m==="recording"?h.style.color="rgb(252 165 165)":m==="processing"?h.style.color="rgb(253 224 71)":
h.style.color="rgb(209 213 219)"}}o(It,"setMicRecordingIndicator");function Vn(){zn().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}o(Vn,"resetMicWaveformBars");function Pt(){if(hn&&(cancelAnimationFrame(
hn),hn=null),sn){try{sn.disconnect()}catch{}sn=null}if(Ut){try{Ut.close()}catch{}Ut=null}kt=null,$t=
null,Vn()}o(Pt,"stopMicWaveform");function Wn(d){Pt();const m=zn();if(!m.length)return;const h=window.
AudioContext||window.webkitAudioContext;if(!h)return;try{Ut=new h,kt=Ut.createAnalyser(),kt.fftSize=
256,kt.smoothingTimeConstant=0,sn=Ut.createMediaStreamSource(d),sn.connect(kt),$t=new Uint8Array(kt.
frequencyBinCount)}catch{Pt();return}const b=o(()=>{if(!kt||!$t)return;kt.getByteFrequencyData($t);const x=Math.
max(1,Math.floor($t.length/m.length));for(let S=0;S<m.length;S++){const M=($t[Math.min($t.length-1,S*
x)]||0)/255,$=Math.max(2,Math.round(2+M*10));m[S].style.height=`${$}px`,m[S].style.opacity=`${.35+M*
.65}`}hn=requestAnimationFrame(b)},"render");b()}o(Wn,"startMicWaveform");function yn(){if(mn&&(clearInterval(
mn),mn=null),tn){try{tn.disconnect()}catch{}tn=null}if(Gt){try{Gt.close()}catch{}Gt=null}Et=null}o(yn,
"stopSilenceMonitor");function Jn(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;yn();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;Gt=new m,Et=Gt.createAnalyser(),Et.fftSize=2048,
tn=Gt.createMediaStreamSource(d),tn.connect(Et);const h=new Uint8Array(Et.fftSize),b=getStsSilenceMs(),
x=.02;Cn=0,fn=!1,mn=setInterval(()=>{if(!Et)return;Et.getByteTimeDomainData(h);let S=0;for(let M=0;M<
h.length;M++){const $=(h[M]-128)/128;S+=$*$}if(Math.sqrt(S/h.length)>x){fn||(fn=!0),Cn=Date.now();return}
fn&&Date.now()-Cn>b&&it&&it.state==="recording"&&it.stop()},200)}o(Jn,"startSilenceMonitor");const An=class An{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null}async start(m,h,b,x={}){this.model=b,this.ws=new WebSocket(`${h}?access_token=${m}`),
this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Send\
ing setup...");const M=!!(x&&x.transcriptionConfig),$={setup:{model:`models/${b}`,generationConfig:{
responseModalities:M?["TEXT"]:["AUDIO"]},inputAudioTranscription:M?x.transcriptionConfig||{}:{},outputAudioTranscription:{}}};
x.speechConfig&&($.setup.generationConfig.speechConfig=x.speechConfig),x.thinkingConfig&&($.setup.generationConfig.
thinkingConfig=x.thinkingConfig),x.translationConfig&&($.setup.translationConfig=x.translationConfig),
console.log("Sending setup:",JSON.stringify($)),this.ws.send(JSON.stringify($))},this.ws.onmessage=M=>this.
_handleMessage(M),this.ws.onerror=M=>{console.error("Gemini Live WebSocket error:",M),this.onError&&
this.onError(M)},this.ws.onclose=M=>{console.log("Gemini Live WebSocket closed:",M.code,M.reason),this.
onClose&&this.onClose(M)},this.audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
this.stream=await navigator.mediaDevices.getUserMedia({audio:!0});const S=this.audioContext.createMediaStreamSource(
this.stream);this.processor=this.audioContext.createScriptProcessor(4096,1,1),this.userAudioChunks=[];
const T=new MediaRecorder(this.stream);T.ondataavailable=M=>{M.data.size>0&&this.userAudioChunks.push(
M.data)},T.start(500),this.backupRecorder=T,this.processor.onaudioprocess=M=>{if(!this.ws||this.ws.readyState!==
WebSocket.OPEN||!this.setupComplete)return;const $=M.inputBuffer.getChannelData(0),j=new Int16Array(
$.length);for(let P=0;P<$.length;P++)j[P]=Math.max(-1,Math.min(1,$[P]))*32767;this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(j.buffer))),mimeType:"\
audio/pcm;rate=16000"}}}))},S.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
const h=JSON.parse(m.data);if(console.log("Gemini Live raw message received:",h),h.setupComplete&&(console.
log("Gemini Live setup complete confirmed"),this.setupComplete=!0),h.serverContent){const b=h.serverContent;
b.modelTurn&&b.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log("Gemini thought delta:",
x.text),this.assistantThought+=x.text):(console.log("Gemini transcript delta (parts):",x.text),this.
assistantText+=x.text)),x.inlineData&&x.inlineData.data){const S=x.inlineData.data;console.log("Gemi\
ni audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);const T=atob(S),M=new Uint8Array(
T.length);for(let $=0;$<T.length;$++)M[$]=T.charCodeAt($);this.assistantAudioChunks.push(M)}}),b.outputTranscription&&
(console.log("Gemini output transcription delta:",b.outputTranscription.text),this.assistantText.includes(
b.outputTranscription.text)||(this.assistantText+=b.outputTranscription.text)),b.inputTranscription&&
(console.log("User input transcription delta:",b.inputTranscription.text),this.inputTranscript+=b.inputTranscription.
text,this.interimInputTranscript=""),b.interimInputTranscription&&(console.log("User interim transcr\
iption:",b.interimInputTranscription.text),this.interimInputTranscript=b.interimInputTranscription.text)}
this.onMessage&&this.onMessage(h)}stop(){this.ws&&this.ws.close(),this.processor&&this.processor.disconnect(),
this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().forEach(m=>m.stop()),
this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const m=new Blob(this.assistantAudioChunks),
h=await this._blobToBase64(m),b=new Blob(this.userAudioChunks),x=await this._blobToBase64(b);return{
user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.assistantThought,
audio_base64:h,user_audio_base64:x}}_blobToBase64(m){return new Promise(h=>{const b=new FileReader;b.
onloadend=()=>h(b.result.split(",")[1]),b.readAsDataURL(m)})}};o(An,"GeminiLiveClient");let Ln=An;const En=class En{constructor(m=24e3){
const h=window.AudioContext||window.webkitAudioContext;this.ctx=new h({sampleRate:m}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(m){if(!this.ctx)return;const h=atob(m),b=new Uint8Array(
h.length);for(let j=0;j<h.length;j++)b[j]=h.charCodeAt(j);const x=new Int16Array(b.buffer),S=new Float32Array(
x.length);for(let j=0;j<x.length;j++)S[j]=x[j]/32768;const T=this.ctx.createBuffer(1,S.length,this.ctx.
sampleRate);T.getChannelData(0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const M=this.
ctx.createBufferSource();M.buffer=T,M.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const $=Math.max(this.ctx.currentTime,this.nextStartTime);
M.start($),this.nextStartTime=$+T.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(En,
"RealTimeAudioPlayer");let an=En;const $n=class $n{constructor(){this.active=!1,this.capturing=!1,this.
sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=
null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript=
"",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=
null,this.saved=!1,this.saving=!1,this.stopping=!1}isActive(){return this.active}async start(){if(this.
active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const m=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const x=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=x.id!==null&&x.id!==void 0?String(x.id):x.id,setTemporaryChatUiState(!!(x&&x.
is_temporary)),setCurrentChatHeaderTitle(x&&x.title),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+x.id),get("welcome-screen").classList.add("hidden")}catch(b){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+b.message,"\
error",!0);return}const h={model:m,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,target_lang:isGeminiLiveTranslateModel()&&get("sts-target-lang")?get(
"sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(h.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const b=await apiFetch(
"/api/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
h)}),x=await b.json().catch(()=>({}));if(!b.ok)throw new Error(x.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=x.session_id,this.rateIn=x.rate_in||this.rateIn,this.rateOut=x.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null}catch(b){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
b.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(b){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
b.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const m="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),h=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(m,h).then(b=>{if(!b.ok)throw new Error("SSE stream failed ("+b.status+")");this.
reader=b.body.getReader(),this._readLoop()}).catch(b=>{b&&b.name==="AbortError"||(this.streamError=b&&
b.message?b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const m=new TextDecoder;let h="";try{for(;this.reader;){const{done:b,
value:x}=await this.reader.read();if(b)break;h+=m.decode(x,{stream:!0});let S;for(;(S=h.indexOf(`

`))>=0;){const T=h.slice(0,S);h=h.slice(S+2);for(const M of T.split(`
`)){if(!M.startsWith("data: "))continue;let $=null;try{$=JSON.parse(M.slice(6))}catch{continue}this.
_handleEvent($)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new an(this.rateOut||24e3),zt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(m.data));break;case"transcript":m.role==="user"?(m.cumulative?this.userTranscript=
m.delta:this.userTranscript+=m.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
m.role==="assistant"?(this.assistantTranscript+=m.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):m.role==="thought"&&(this.assistantThought+=m.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":m.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"error":this.streamError=m.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}zt=null}_startCapture(){const m=window.
AudioContext||window.webkitAudioContext;if(!m)throw new Error("AudioContext not supported");return this.
audioCtx=new m({sampleRate:this.rateIn||24e3}),navigator.mediaDevices.getUserMedia(Xn()).then(h=>{this.
stream=h;const b=this.audioCtx.createMediaStreamSource(h),x=this.rateIn||24e3,S=this.audioCtx.sampleRate,
T=4096;this.processor=this.audioCtx.createScriptProcessor(T,1,1),this.processor.onaudioprocess=M=>{if(!this.
active||!this.capturing)return;const $=M.inputBuffer.getChannelData(0),j=di($,S,x);!j||!j.byteLength||
this._sendAudio(j)},b.connect(this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(m){
if(!this.sessionId||!this.active)return;const h="/api/realtime/audio?session_id="+encodeURIComponent(
this.sessionId),b={method:"POST",credentials:"include",headers:{"X-CSRF-Token":csrfToken,"Content-Ty\
pe":"application/octet-stream"},body:m},x=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions==
"function"?window.ProgressSpinner.manualRequestOptions(b):b;fetch(h,x).catch(()=>{})}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}yn(),Pt()}async stop(){if(!this.active)return;this.active=
!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0);try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({session_id:this.sessionId})})}catch{}const m=Date.now(),h=this.responseDoneCount;
let b=this.lastAudioAt;for(;Date.now()-m<2e4&&!(this.responseDoneCount>h||(this.lastAudioAt>b&&(b=this.
lastAudioAt),!this.speechActive&&Date.now()-m>2e3&&Date.now()-b>2500));)await new Promise(x=>setTimeout(
x,250));await this._save()}async _save(){if(!this.saved){this.saved=!0,this.saving=!0;try{const m=await apiFetch(
"/api/realtime/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:this.sessionId,thread_id:currentThreadId})}),h=await m.json().catch(()=>({}));if(!m.ok)throw new Error(
h.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(m){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(m&&m.message?m.message:m),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this._stopCapture(),this._stopPlayback(),this.abortCtrl){try{this.abortCtrl.
abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const m=get("mic-btn");m&&(m.
classList.remove("bg-red-600","animate-pulse"),m.classList.add("bg-gray-700"))}};o($n,"RealtimeVoice\
Session");let Mn=$n;function di(d,m,h){let b=d;if(m!==h&&m>0&&h>0){const S=m/h,T=Math.floor(b.length/
S),M=new Float32Array(T);for(let $=0;$<T;$++)M[$]=b[Math.min(Math.floor($*S),b.length-1)];b=M}const x=new Int16Array(
b.length);for(let S=0;S<b.length;S++){const T=Math.max(-1,Math.min(1,b[S]));x[S]=T<0?T*32768:T*32767}
return x.buffer}o(di,"pcm16FromFloat32");const on=new Mn;(()=>{const h={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let b=null,x=null,S=!1,T=null,M=!1,$=0,j=0,P=null,
H="idle",se=!1,q=null;const N=o(I=>document.getElementById(I),"$"),ee=o(I=>{const D=Object.assign({},
I||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?
window.ProgressSpinner.manualRequestOptions(D):(D.progressSpinner=!1,D)},"noSpinner");function Q(I,D){
H=D;const ve=N("lyria-status-text"),ae=N("lyria-status-dot");ve&&(ve.textContent=I),ae&&(ae.className=
"w-2 h-2 rounded-full inline-block "+(h[D]||h.idle)),Ae(),Y()}o(Q,"setStatus");function xe(){const I=j?
Math.floor((Date.now()-j)/1e3):0,D=String(Math.floor(I/60)).padStart(2,"0"),ve=String(I%60).padStart(
2,"0");return`${D}:${ve}`}o(xe,"formatElapsed");function W(){j||(j=Date.now());const I=N("lyria-elap\
sed");I&&(I.textContent=xe()),P||(P=window.setInterval(()=>{const D=N("lyria-elapsed");D&&(D.textContent=
xe())},1e3))}o(W,"startElapsedTimer");function le(){P&&(window.clearInterval(P),P=null)}o(le,"stopEl\
apsedTimer");function Ae(){const I=N("lyria-play-btn"),D=N("lyria-pause-btn"),ve=N("lyria-stop-btn"),
ae=N("lyria-reset-btn"),_e=!!b,Be=H==="streaming"||H==="connecting";if(I){I.disabled=se;const Qe=I.querySelector(
"i");Qe&&(Qe.className="fas fa-play")}D&&(D.disabled=se||!Be),ve&&(ve.disabled=se||!_e||!Be),ae&&(ae.
disabled=se||!_e||!Be)}o(Ae,"updateTransportButtons");function Y(){const I=N("lyria-save-btn");if(!I)
return;const D=!!b&&H!=="idle"&&H!=="connecting"&&H!=="error";I.classList.toggle("hidden",!D)}o(Y,"u\
pdateSaveButton");function ce(I,D){const ve=N("lyria-prompt-rows");if(!ve)return;const ae=document.createElement(
"div");ae.className="flex items-center gap-2",ae.innerHTML=`
                        <input type="text" value="${escapeHtml(I||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof D=="num\
ber"?D:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof D=="number"?D:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const _e=ae.querySelector('input[type="range"]'),Be=ae.querySelector(".lyria-w\
eight-label");_e&&Be&&_e.addEventListener("input",()=>{Be.textContent=parseFloat(_e.value).toFixed(1)});
const Qe=ae.querySelector(".lyria-prompt-remove");Qe&&Qe.addEventListener("click",()=>{ve.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ae.remove()}),ae.classList.add("lyria-prompt-row-wrap"),ve.appendChild(
ae)}o(ce,"addPromptRow");function Ce(){const I=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),D=[];return I.forEach(ve=>{const ae=ve.querySelector('input[type="text"]'),_e=ve.querySelector(
'input[type="range"]'),Be=(ae?ae.value:"").trim();Be&&D.push({text:Be,weight:parseFloat(_e?_e.value:
1)||1})}),D}o(Ce,"collectPrompts");function $e(){const I={},D=o(fi=>{const Pn=N(fi);return Pn&&Pn.value!==
""?parseFloat(Pn.value):void 0},"num"),ve=D("lyria-bpm");ve!==void 0&&(I.bpm=Math.round(ve));const ae=D(
"lyria-guidance");ae!==void 0&&(I.guidance=ae);const _e=D("lyria-density");_e!==void 0&&(I.density=_e);
const Be=D("lyria-brightness");Be!==void 0&&(I.brightness=Be);const Qe=D("lyria-temperature");Qe!==void 0&&
(I.temperature=Qe);const Ue=N("lyria-scale");Ue&&Ue.value&&(I.scale=Ue.value);const et=N("lyria-mode");
et&&et.value&&(I.music_generation_mode=et.value);const ct=N("lyria-mute-bass"),Ot=N("lyria-mute-drum\
s"),ti=N("lyria-only-bass-drums");return ct&&(I.mute_bass=ct.checked),Ot&&(I.mute_drums=Ot.checked),
ti&&(I.only_bass_and_drums=ti.checked),I}o($e,"collectConfig");function st(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([D,ve])=>{
const ae=N(D),_e=N(ve);!ae||!_e||ae.addEventListener("input",()=>{const Be=parseFloat(ae.value);_e.textContent=
D==="lyria-bpm"?String(Math.round(Be)):Be.toFixed(1)})})}o(st,"bindRangeLabels");function We(){if(T){
try{T.close()}catch{}T=null}M=!1,$=0}o(We,"resetPlayback");function lt(){if(S=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(lt,"closeStream");async function _t(){lt(),x=new AbortController,
S=!0;try{const I=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(b)}`,ee({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!I.ok){const _e=await I.
json().catch(()=>({}));throw new Error(_e.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const D=I.body.getReader(),ve=new TextDecoder;let ae="";for(;S;){const{done:_e,value:Be}=await D.read();
if(_e)break;ae+=ve.decode(Be,{stream:!0});const Qe=ae.split(`

`);ae=Qe.pop();for(const Ue of Qe){const et=Ue.split(`
`).find(Ot=>Ot.startsWith("data: "));if(!et)continue;const ct=et.slice(6);try{const Ot=JSON.parse(ct);
he(Ot)}catch{}}}}catch(I){if(I&&I.name==="AbortError")return;S&&(Q("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&b&&_t()},1200))}finally{S=!1}}o(_t,"openStream");function he(I){
if(I&&I.snapshot){const D=I.status;if(D==="error"){Q("\u30A8\u30E9\u30FC","error"),le();return}if(D===
"closed"||D==="stopped"){Q("\u7D42\u4E86","closed"),le();return}Q(D==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",D==="paused"?"paused":"connecting");return}if(I&&I.audio){Q("\u518D\u751F\u4E2D...",
"streaming"),W(),ge(I.audio);return}if(I&&I.error){Q("\u30A8\u30E9\u30FC: "+I.error,"error"),le();return}
if(I&&I.final){Q("\u7D42\u4E86","closed"),le(),Ae();return}}o(he,"handleStreamMessage");function ge(I){
if(!I)return;if(!T){const Ue=window.AudioContext||window.webkitAudioContext;if(!Ue)return;T=new Ue({
sampleRate:48e3}),M=!1,$=0}let D;try{const Ue=atob(I);D=new Uint8Array(Ue.length);for(let et=0;et<Ue.
length;et++)D[et]=Ue.charCodeAt(et)}catch{return}const ve=new Int16Array(D.buffer),ae=Math.floor(ve.
length/2);if(ae<1)return;const _e=T.createBuffer(2,ae,48e3);for(let Ue=0;Ue<2;Ue++){const et=_e.getChannelData(
Ue);for(let ct=0;ct<ae;ct++)et[ct]=ve[ct*2+Ue]/32768}T.state==="suspended"&&T.resume();const Be=T.createBufferSource();
Be.buffer=_e,Be.connect(T.destination),M||($=T.currentTime+.08,M=!0);const Qe=Math.max(T.currentTime,
$);Be.start(Qe),$=Qe+_e.duration}o(ge,"playChunk");async function je(I,D){const ve=await fetch("/api\
/gemini/music/command",ee({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:I},D||{}))})),ae=await ve.json().catch(()=>({}));if(!ve.ok)throw new Error(
ae.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ae}
o(je,"apiCommand");async function fe(){if(se)return;const I=Ce();if(!I.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}se=!0,Ae(),Q("\u63A5\u7D9A\u4E2D...","connecting");try{
const D=await fetch("/api/gemini/music/start",ee({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:I,config:$e()})})),ve=await D.json().catch(()=>({}));if(!D.
ok)throw new Error(ve.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=ve.session_id,q=$e(),Q("\u63A5\u7D9A\u4E2D...","connecting"),_t()}catch(D){Q("\u30A8\u30E9\u30FC: "+
D.message,"error"),showToast("Lyria RealTime: "+D.message,"error",!0)}finally{se=!1,Ae()}}o(fe,"star\
tSession");async function ke(I){if(b){se=!0,Ae();try{await je("control",{action:I}),I==="PLAY"?Q("\u518D\u751F\
\u4E2D...","streaming"):I==="PAUSE"?Q("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):I==="STOP"?Q("\u505C\u6B62\u4E2D",
"stopped"):I==="RESET_CONTEXT"&&Q("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(D){showToast("Lyria RealTime: "+D.message,"error",!0),Q("\u30A8\u30E9\u30FC: "+D.
message,"error")}finally{se=!1,Ae()}}}o(ke,"control");async function Re(){if(!b)return;const I=Ce();
if(!I.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}se=!0;try{await je("prompts",{weighted_prompts:I}),Q("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
H==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(D){showToast("Lyria RealTime: "+D.message,"error",!0)}finally{se=!1,Ae()}}o(Re,"app\
lyPrompts");async function St(){if(!b)return;const I=$e(),D=q||{},ve=I.bpm!==void 0&&I.bpm!==D.bpm,ae=I.
scale!==void 0&&I.scale!==D.scale,_e=ve||ae;se=!0;try{await je("config",{config:I,reset_context:_e}),
q=I,Q(_e?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",H==="paused"?"paused":"streaming"),showToast(
_e?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Be){showToast("Lyria RealT\
ime: "+Be.message,"error",!0)}finally{se=!1,Ae()}}o(St,"applyConfig");async function Vt(){if(b){se=!0,
Q("\u4FDD\u5B58\u4E2D...","connecting"),Ae();try{const I=await fetch("/api/gemini/music/save",ee({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),D=await I.json().catch(()=>({}));if(!I.ok)throw new Error(D.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
Q("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),le(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),D.thread_id&&(currentThreadId=String(D.thread_id),history.pushState({},"","/c/"+D.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(D.thread_id||currentThreadId),ln(!0)}catch(I){
Q("\u30A8\u30E9\u30FC: "+I.message,"error"),showToast("Lyria RealTime: "+I.message,"error",!0)}finally{
se=!1,Ae()}}}o(Vt,"saveSession");async function Ge(){if(lt(),b)try{await fetch("/api/gemini/music/ca\
ncel",ee({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})}))}catch{}
b=null,le(),We(),Q("\u6E96\u5099\u5B8C\u4E86","idle")}o(Ge,"cancelSession");function pt(){const I=N(
"lyria-prompt-rows");I&&(I.innerHTML=""),ce("",1),q=null,j=0,["lyria-bpm","lyria-guidance","lyria-de\
nsity","lyria-brightness","lyria-temperature"].forEach(ae=>{const _e=N(ae);_e&&(_e.value=ae==="lyria\
-bpm"?"120":ae==="lyria-guidance"?"4":ae==="lyria-temperature"?"1.1":"0.5")});const D=N("lyria-scale");
D&&(D.value="");const ve=N("lyria-mode");ve&&(ve.value="QUALITY"),["lyria-mute-bass","lyria-mute-dru\
ms","lyria-only-bass-drums"].forEach(ae=>{const _e=N(ae);_e&&(_e.checked=!1)}),st()}o(pt,"resetContr\
ols");function ln(I){lt(),b&&fetch("/api/gemini/music/cancel",ee({method:"POST",headers:{"Content-Ty\
pe":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),b=null,S=!1,le(),We(),hideModal(
"lyria-studio-modal")}o(ln,"closeAndCleanup");function In(I){if(!isLyriaRealtimeModel()){showToast("\
Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const D=N("lyria-studio-modal");if(D&&D.classList.contains("modal-open")&&b){if(I&&
typeof I=="string"){const ae=N("lyria-prompt-rows");ae&&(ae.innerHTML=""),ce(I,1)}return}if(b&&Ge(),
pt(),I&&typeof I=="string"){const ae=N("lyria-prompt-rows");ae&&(ae.innerHTML=""),ce(I,1)}b=null,S=!1,
le(),We(),Q("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(In,"open");function mi(){
const I=N("lyria-open-studio-btn");I&&I.addEventListener("click",()=>In(""));const D=N("lyria-studio\
-close");D&&D.addEventListener("click",()=>ln(!1));const ve=N("lyria-play-btn");ve&&ve.addEventListener(
"click",()=>{if(!b){fe();return}ke("PLAY")});const ae=N("lyria-pause-btn");ae&&ae.addEventListener("\
click",()=>ke("PAUSE"));const _e=N("lyria-stop-btn");_e&&_e.addEventListener("click",()=>ke("STOP"));
const Be=N("lyria-reset-btn");Be&&Be.addEventListener("click",()=>ke("RESET_CONTEXT"));const Qe=N("l\
yria-add-prompt-btn");Qe&&Qe.addEventListener("click",()=>ce("",1));const Ue=N("lyria-apply-prompts-\
btn");Ue&&Ue.addEventListener("click",Re);const et=N("lyria-apply-config-btn");et&&et.addEventListener(
"click",St);const ct=N("lyria-save-btn");ct&&ct.addEventListener("click",Vt),st(),pt(),window.openLyriaStudio=
In}return o(mi,"init"),{init:mi,open:In}})().init(),(()=>{let d=null,m=null,h=null;const b="voiceDoc\
kSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=o(Y=>document.getElementById(Y),"$");function T(){return isStsModel()&&voiceStudioUiEnabled!==!1}o(
T,"isStudioMode");function M(){const Y=get("model-select")?get("model-select").value:"",ce=S("voice-\
studio-title");ce&&(Y==="gpt-transcribe"||Y==="gpt-live-transcribe"?ce.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":
Y==="gemini-3.5-live-translate-preview"?ce.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ce.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(M,"updateTitle");function $(){const Y=S("vo\
ice-studio-transcript");Y&&(Y.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gray-\
500">${x}</div>`);const ce=S("sts-live-transcript");ce&&(ce.innerHTML="",ce.classList.add("hidden"))}
o($,"resetTranscript");function j(Y,ce,Ce){const $e=Y.querySelectorAll(".voice-studio-line");let st=null;
for(let We=$e.length-1;We>=0;We--)if($e[We].dataset.role===ce){st=$e[We];break}if(st)st.innerHTML=Ce;else{
const We=Y.querySelector(".voice-studio-placeholder");We&&We.remove();const lt=document.createElement(
"div");lt.className="voice-studio-line",lt.dataset.role=ce,lt.innerHTML=Ce,Y.appendChild(lt)}Y.classList.
remove("hidden"),Y.scrollTop=Y.scrollHeight}o(j,"writeLine");function P(Y,ce){if(!ce||!String(ce).trim()||
!T())return;const st=`<span class="${Y==="user"?"text-cyan-300":"text-gray-100"} font-bold">${escapeHtml(
Y==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(ce)}</span>`;
[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(We=>j(We,Y,st))}o(P,
"log");function H(Y,ce=!0){const Ce=S("sts-panel"),$e=S("sts-settings-toggle");if(Ce&&Ce.classList.toggle(
"settings-open",!!Y),$e&&$e.setAttribute("aria-expanded",Y?"true":"false"),ce)try{localStorage.setItem(
b,Y?"1":"0")}catch{}}o(H,"setSettingsOpen");function se(){try{return localStorage.getItem(b)==="1"}catch{
return!1}}o(se,"readSettingsOpen");let q=null;function N(){const Y=get("model-select")?get("model-se\
lect").value:"";Y!==q&&(q=Y,$()),H(se(),!1),M()}o(N,"syncDock");function ee(){const Y=S("sts-panel"),
ce=S("voice-studio-panel-host");Y&&ce&&Y.parentNode!==ce&&(d=Y.parentNode,m=Y.nextSibling,ce.appendChild(
Y));const Ce=S("file-preview"),$e=S("voice-studio-file-host");Ce&&$e&&Ce.parentNode!==$e&&(h=Ce.parentNode,
$e.appendChild(Ce),$e.classList.remove("hidden"))}o(ee,"movePanelIntoModal");function Q(){const Y=S(
"sts-panel");Y&&d&&Y.parentNode!==d&&(m&&m.parentNode===d?d.insertBefore(Y,m):d.appendChild(Y));const ce=S(
"file-preview");ce&&h&&ce.parentNode!==h&&h.appendChild(ce);const Ce=S("voice-studio-file-host");Ce&&
Ce.classList.add("hidden"),d=null,m=null,h=null}o(Q,"movePanelBack");function xe(){if(!T())return;ee();
const Y=S("sts-panel");Y&&Y.classList.remove("hidden"),M(),window.VoiceStudioOpen=!0,showModal("voic\
e-studio-modal")}o(xe,"open");function W(){window.VoiceStudioOpen=!1,Q(),hideModal("voice-studio-mod\
al")}o(W,"close");function le(){window.VoiceStudioOpen&&W()}o(le,"closeIfOpen");function Ae(){window.
VoiceStudioOpen=!1;const Y=S("voice-studio-open-btn");Y&&Y.addEventListener("click",()=>xe());const ce=S(
"voice-studio-close");ce&&ce.addEventListener("click",()=>W());const Ce=S("sts-settings-toggle");Ce&&
Ce.addEventListener("click",()=>{const $e=S("sts-panel");H(!($e&&$e.classList.contains("settings-ope\
n")))}),window.VoiceStudio={open:xe,close:W,closeIfOpen:le,log:P,isStudioMode:T,syncDock:N},N()}return o(
Ae,"init"),{init:Ae}})().init();let zt=null;function Kn(){if(zt&&(zt.stop(),zt=null),nn){try{nn.pause()}catch{}
try{nn.src=""}catch{}nn=null}}o(Kn,"stopStsPlayback");async function yi(d){Kn();const m=new Audio;return m.
src=d,m.preload="auto",m.autoplay=!0,m.playsInline=!0,nn=m,await m.play(),new Promise(h=>{m.onended=
()=>h("ended"),m.onerror=()=>h("error")})}o(yi,"playStsAudio");function ui(){if(on.isActive()){on._cancel();
return}if(Fe){Fe.stop(),Fe=null,Kn(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap\
 to speak",!1),800),Pt();return}it&&it.state==="recording"&&(Tn=!0,it.stop())}o(ui,"cancelRecording");
function Xn(){if(isStsModel())return{audio:!0};const m=navigator.mediaDevices&&navigator.mediaDevices.
getSupportedConstraints?navigator.mediaDevices.getSupportedConstraints():{},h={channelCount:1};return m.
echoCancellation&&(h.echoCancellation=!1),m.noiseSuppression&&(h.noiseSuppression=!1),m.autoGainControl&&
(h.autoGainControl=!1),{audio:h}}o(Xn,"getMicCaptureConstraints"),get("mic-btn").onclick=async()=>{if(abortController){
showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Fe){setStsStatus("Processing...",!0);const d=Fe;Fe=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const m=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(m.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
m.assistant_text=(d.inputTranscript||"").trim(),m.assistant_thought="",!m.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){
const b=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=
String(b.id),history.pushState({},"","/c/"+b.id),get("welcome-screen").classList.add("hidden")}m.thread_id=
currentThreadId,m.model=get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error(
"Failed to save Gemini Live session:",m),setStsStatus("Error saving",!1)}return}if(on.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
on.stop();return}if(it&&it.state==="recording"){it.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||It("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const h=new Audio;
h.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",h.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const b={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(b.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const q=get("sts-custom\
-vocab").value.split(/[,、\n]/).map(N=>N.trim()).filter(Boolean);q.length&&(b.custom_vocabulary=q.slice(
0,1e3))}}else b.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(b.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",b.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(b.target_lang=get("sts-target-lang").value);const x=await apiFetch("/api/gemini\
/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});if(!x.
ok)throw new Error("Failed to get session token");const{token:S,url:T}=await x.json(),M=get("model-s\
elect").value,$=get("sts-voice")?get("sts-voice").value:"Kore",j=get("sts-thinking-level")?get("sts-\
thinking-level").value:"minimal",P=get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1;
if(Fe=new Ln,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Fe.rtPlayer=new an),isGeminiLiveTranscribeModel()){
const q=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",N={languageCodes:[]};
if((q==="SMART"||q==="VERBATIM")&&(N.mode=q),get("sts-custom-vocab")){const ee=get("sts-custom-vocab").
value.split(/[,、\n]/).map(Q=>Q.trim()).filter(Boolean);ee.length&&(N.customVocabulary=ee.slice(0,1e3))}
await Fe.start(S,T,M,{transcriptionConfig:N})}else if(isGeminiLiveTranslateModel()){const q=get("sts\
-target-lang")?get("sts-target-lang").value:"ja";await Fe.start(S,T,M,{translationConfig:{targetLanguageCode:q,
echoTargetLanguage:!0}})}else{const q={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:$}}}};
isGeminiLiveExtendedThinkingModel()&&(q.thinkingConfig={thinkingLevel:j,includeThoughts:P}),await Fe.
start(S,T,M,q)}it=Fe.backupRecorder,it.onstop=()=>{Fe&&get("mic-btn").click()};let H=!0,se="live-sts\
-"+Date.now();Fe.onMessage=q=>{if(q.serverContent){if(isGeminiLiveTranscribeModel()){const N=Fe.interimInputTranscript,
ee=Fe.inputTranscript,Q=ee+(N&&!ee.endsWith(N)?(ee?`
`:"")+N:""),xe=get("chat-messages");let W=document.getElementById(se);W||(W=document.createElement("\
div"),W.id=se,W.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg\
 border border-slate-700/50",W.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,xe.appendChild(W),xe.scrollTop=xe.scrollHeight);const le=W.
querySelector(".message-content");le.innerText=Q||"\u8074\u304D\u53D6\u308A\u4E2D...",xe.scrollTop=xe.
scrollHeight,window.VoiceStudio&&ee&&window.VoiceStudio.log("user",ee);return}if(q.serverContent.modelTurn){
H&&(setStsStatus("Gemini is speaking...",!1),H=!1);const N=get("chat-messages");let ee=document.getElementById(
se);ee||(ee=document.createElement("div"),ee.id=se,ee.className="flex flex-col gap-2 mb-4 assistant-\
message bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",ee.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,N.appendChild(ee),N.scrollTop=N.scrollHeight);const Q=ee.
querySelector(".thought-container"),xe=ee.querySelector(".message-content");Fe.assistantThought&&(Q.
classList.remove("hidden"),Q.innerText=Fe.assistantThought),xe.innerText=Fe.assistantText,N.scrollTop=
N.scrollHeight,window.VoiceStudio&&(Fe.inputTranscript&&window.VoiceStudio.log("user",Fe.inputTranscript),
Fe.assistantText&&window.VoiceStudio.log("assistant",Fe.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),Wn(Fe.stream),Jn(Fe.stream);return}catch(h){showToast("Gemini Live connection failed: "+
h.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){await on.start();
return}isStsModel()||(Vn(),It("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));const d=await navigator.
mediaDevices.getUserMedia(Xn());it=new MediaRecorder(d),pn=[],Tn=!1;const m=isStsModel();it.ondataavailable=
h=>pn.push(h.data),it.onstop=async()=>{if(Tn){pn=[],get("file-preview").classList.add("hidden"),d.getTracks().
forEach(T=>T.stop()),yn(),Pt(),m||(It("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),bn=setTimeout(()=>It("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const h=new Blob(pn,{type:"audio/web\
m"}),b=new File([h],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",b),get("fil\
e-preview").classList.remove("hidden");const S=m;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const Q=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=Q.id!==null&&Q.id!==void 0?String(Q.id):Q.id,setTemporaryChatUiState(!!(Q&&Q.
is_temporary)),setCurrentChatHeaderTitle(Q&&Q.title),applyTemporaryChatRuntimeMeta(Q||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+Q.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),x.append("model",get(
"model-select").value),x.append("thread_id",currentThreadId),get("sts-voice")&&x.append("sts_voice",
get("sts-voice").value||""),get("sts-speed")&&x.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&x.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&x.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&x.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&x.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const T=await apiFetch(
"/sts",{method:"POST",body:x});if(!T.ok){const ee=await T.json().catch(()=>({}));throw new Error(ee.
error||"Speech-to-speech failed")}const M=T.body.getReader(),$=new TextDecoder;let j="",P=null,H=null;
stsOpt("sts-auto-play")&&(H=new an,zt=H),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let se=!0,q="",N="";for(;;){const{done:ee,value:Q}=await M.read();if(ee)break;j+=
$.decode(Q,{stream:!0});const xe=j.split(`
`);j=xe.pop();for(const W of xe){if(!W.trim())continue;const le=JSON.parse(W);if(le.error)throw new Error(
le.error);le.audio_delta&&H&&(se&&(setStsStatus("Playing response...",!1),se=!1),await H.addChunk(le.
audio_delta)),le.input_delta&&(q+=le.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",q)),
le.transcript_delta&&(N+=le.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assistant",
N)),(le.final||le.audio_url)&&(P=le)}}window.VoiceStudio&&!q.trim()&&window.VoiceStudio.log("user","\
\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),P&&(P.audio_url||P.transcription_only)&&(stsOpt(
"sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-btn").click()},
500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const T=get("set-mic-\
transcribe-mode");if(!!(T&&T.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306FLLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\
\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093","e\
rror",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const j=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(j.transcript){const P=get("prompt-inp\
ut");P.value+=(P.value?" ":"")+j.transcript,P.style.height="auto",P.style.height=P.scrollHeight+"px"}else
showToast(j.error||"Transcription failed","error",!0)}}catch(T){showToast("Audio processing error: "+
T.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(T=>T.
stop()),yn(),Pt(),S||It("","hidden"),S&&setStsStatus("Tap to speak",!1)}},it.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(It("\u9332\u97F3\u4E2D\u2026","recording"),Wn(d)),Jn(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Pt(),isStsModel()||It("","hidden"),alert("Microphone access denied or not avail\
able.")}};const rn=o((d,m)=>{if(!d)return;const h=d.querySelector("span");h?h.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),h=get("lib-download-btn"),b=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(m&&(m.disabled=d===0,rn(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
h&&(h.disabled=d===0,rn(h,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=d===0,rn(b,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,rn(x,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=d!==1,rn(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const T=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",T&&d>0)}};function Yn(d){lib.
attachMode=!!d}o(Yn,"setLibAttachMode");const Qn=o((d=!1)=>{Yn(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>Qn(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>Qn(!0)),
get("lib-search")){let d=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),d&&clearTimeout(d),d=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const d=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=d,get("lib-sort").onchange=()=>{const m=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,m),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const pi=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>pi(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
m=get("gem-desc").value,h=get("gem-inst").value,b=collectGemFixedPrompts();if(d&&h){const x=editingGemUuid?
"PUT":"POST",S=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
S,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:h,
fixed_prompts:b,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=d,activeGem.instruction=
h,activeGem.fixed_prompts=b,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(d){if(d.target.closest(".edit-btn")){const h=d.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(h)}if(d.target.closest(".code-toggle")){
const m=d.target.closest(".code-toggle"),h=m.closest(".code-wrapper");if(!h)return;const b=h.classList.
toggle("collapsed");h.setAttribute("data-collapsed",b?"true":"false"),m.setAttribute("aria-expanded",
b?"false":"true"),m.innerHTML=b?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',m.title=b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",m.setAttribute("aria-label",b?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(d.target.closest(".download-btn")){const m=d.target.closest(".d\
ownload-btn"),h=m.getAttribute("data-code"),b=(m.getAttribute("data-lang")||"txt").toLowerCase();if(h)
try{const x=decodeURIComponent(h),S=new Blob([x],{type:"text/plain"}),T=URL.createObjectURL(S),M=document.
createElement("a");M.href=T;let j={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(j="txt");let P=`code.${j}`;b==="dockerfile"&&(P="Docke\
rfile"),b==="makefile"&&(P="Makefile"),M.download=P,document.body.appendChild(M),M.click(),document.
body.removeChild(M),URL.revokeObjectURL(T)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),h=m.getAttribute("data-code");h&&
window.copyCode(m,h)}if(d.target.closest(".html-preview-btn")){const h=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");h&&openHtmlCodePreview(h)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&Gn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){const e=get(
"file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-progress-bar"),
a=get("file-preview-thumbs"),r=get("upload-modal-status-text"),l=get("upload-modal-total-progress"),
c=get("upload-modal-total-progress-bar");if(!e||!t)return;if(a){const L=document.querySelectorAll("#\
upload-list .upload-row");a.innerHTML="",L.forEach((E,B)=>{const K=E.getAttribute("data-local-url"),
Z=E.getAttribute("data-filename"),Ie=E.querySelector("img.upload-preview")!==null;let O;if(Ie){let G=K;
if(!G&&Z){const te=Z.replace(/^\d+\//,"");G=buildAttachmentPreviewUrl(te)}G&&(O=document.createElement(
"img"),O.src=G,O.className="thumb-item shadow-sm",O.dataset.viewerSrc=G,O.dataset.viewerFilename=Z||
G.split("/").pop(),O.onclick=function(te){te.preventDefault(),openImageViewer(this.dataset.viewerSrc,
".thumb-item")},O.onerror=function(){this.parentElement.replaceChild(u("ERR"),this)})}O||(O=u("FILE")),
O.style.animationDelay=`${B*32}ms`,a.appendChild(O)}),L.length>0?a.classList.remove("hidden"):a.classList.
add("hidden")}function u(L){const E=document.createElement("div");return E.className="thumb-item bg-\
gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold",E.innerText=
L,E}o(u,"createFileThumb");const f=collectImageUrlsForSend(),g=uploadProgressState.total,y=uploadProgressState.
completed,w=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.classList.add("hidden"),
l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const v=get("send-btn"),k=get("mic-btn"),
_=get("mask-btn"),C=isStopMode;if(w>0?(v&&(v.disabled=!0),k&&(k.disabled=!0),_&&(_.disabled=!0)):C||
(v&&(v.disabled=!1),k&&(k.disabled=!1),_&&(_.disabled=!1)),w>0){const L=`Preparing... (${y}/${g})`;e.
classList.remove("hidden"),t.innerText=L,r&&(r.innerText=`(${y}/${g})`);let E=y*100,B=0;for(let Ie in uploadProgressState.
perFilePct)E+=uploadProgressState.perFilePct[Ie],B++;const K=g>0?E/(g*100)*100:0,Z=`${Math.min(100,K)}\
%`;n&&i&&(n.classList.remove("hidden"),i.style.width=Z),l&&c&&(l.classList.remove("hidden"),c.style.
width=Z)}else r&&(r.innerText=""),l&&l.classList.add("hidden"),f.length>0?(e.classList.remove("hidde\
n"),t.innerText=`${f.length} files ready`,n&&n.classList.add("hidden")):(e.classList.add("hidden"),t.
innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}o(updateFilePreview,"update\
FilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get("mask-name");!e||!t||(currentMaskImage?
(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.split("/").pop()}`):(e.classList.
add("hidden"),t.innerText=""))}o(updateMaskPreview,"updateMaskPreview");const markerToolHints={draw:"\
\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],a=t[3];return`#${n}${n}${i}${i}${a}${a}`}
return"#facc15"}o(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const t=normalizeMarkerHexColor(
e);return{r:parseInt(t.slice(1,3),16),g:parseInt(t.slice(3,5),16),b:parseInt(t.slice(5,7),16)}}o(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,t=60){const n=Number(e),i=Number.isFinite(n)?n:t;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,i))}o(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const t=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(t)?String(t):String(t).replace(/\.0$/,"")}o(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${t})`}o(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),a=get("ma\
rker-color-picker");a&&a.value!==e&&(a.value=e);const r=get("marker-opacity");r&&r.value!==i&&(r.value=
i);const l=get("marker-opacity-number");l&&l.value!==i&&(l.value=i);const c=get("marker-opacity-valu\
e");c&&(c.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(f=>{const g=normalizeMarkerHexColor(f.getAttribute("data-marker-color"));f.classList.
toggle("active",g===e)})}o(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}o(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}o(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const a=get("marker-tool-hi\
nt");a&&(a.textContent=markerToolHints[e]||"");const r=get("marker-crop-reset");r&&r.classList.toggle(
"hidden",e!=="crop");const l=get("marker-canvas");l&&(l.style.pointerEvents=e==="crop"?"none":"auto");
const c=get("marker-crop-canvas");c&&(c.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.
cropRect||markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}
o(setMarkerMode,"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}
o(clearCropRect,"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)
return;const t=Math.max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,
y:0,w:t,h:n})}o(resetCropRectToFull,"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.
scale<=markerView.minScale+1e-4){markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marke\
r-stage"),t=get("marker-viewport");if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(
1,e.clientHeight||0),a=Math.max(1,t.offsetWidth||t.clientWidth||0),r=Math.max(1,t.offsetHeight||t.clientHeight||
0);if(n<=1||i<=1||a<=1||r<=1)return;const l=(n-a)/2,c=(i-r)/2,u=a*markerView.scale,f=r*markerView.scale,
g=Math.min(n*.45,Math.max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),w=g-l-u,v=n-g-l,k=y-c-f,_=i-
y-c,C=o((L,E,B)=>Number.isFinite(L)?E>B?(E+B)/2:Math.min(B,Math.max(E,L)):0,"clampOffset");markerView.
offsetX=C(markerView.offsetX,w,v),markerView.offsetY=C(markerView.offsetY,k,_)}o(clampMarkerViewOffset,
"clampMarkerViewOffset");function applyMarkerTransform(){const e=get("marker-viewport");e&&(clampMarkerViewOffset(),
e.style.transform=`translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale}\
)`)}o(applyMarkerTransform,"applyMarkerTransform");function resetMarkerTransform(){markerView.scale=
1,markerView.offsetX=0,markerView.offsetY=0,applyMarkerTransform()}o(resetMarkerTransform,"resetMark\
erTransform");function getRowMarkerKey(e){return e&&(e.dataset.uploadId||e.getAttribute("data-filena\
me"))||null}o(getRowMarkerKey,"getRowMarkerKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(
e);n&&(t?markerAppliedUploads.add(n):markerAppliedUploads.delete(n));const i=e?e.querySelector(".upl\
oad-marker-tag"):null;i&&i.classList.toggle("hidden",!t)}o(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){
return markerAppliedUploads.size>0}o(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){
const t=String(e||"").trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t===
"uploaded"?"upload":"unknown"}o(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let t=String(e).replace(/\u0000/g,"");return t=t.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),t=t.trim(),!t||(t=t.split("/").pop().split("\\").pop().trim(),t=t.replace(/\s{2,}/g,
" "),t=t.replace(/[<>:"/\\|?*]+/g,"_"),!t||t==="."||t==="..")?"":(t.length>180&&(t=t.slice(0,180).trim()),
t)}o(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const t=normalizeAttachmentPath(e);return t?t.split("/").pop()||t:""}o(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,t){const n=normalizeAttachmentPath(
e);if(!n)return;const i=normalizeAttachmentDisplayName(t)||defaultAttachmentDisplayName(n);i&&attachmentNameByPath.
set(n,i)}o(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const t=normalizeAttachmentPath(e);if(!t)return"";const n=normalizeAttachmentDisplayName(attachmentNameByPath.
get(t));return n||defaultAttachmentDisplayName(t)}o(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,t){if(!e)return;const n=normalizeAttachmentDisplayName(t)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=n;const i=e.querySelector(".truncate");
i&&(i.textContent=n);const a=e.getAttribute("data-filename");a&&setAttachmentNameForPath(a,n)}o(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}o(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}o(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
o(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const a=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,a),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}o(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const a=e.querySelector(".truncate"),r=normalizeAttachmentDisplayName(
a?a.textContent:"");return r||getAttachmentNameForPath(t)}o(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,t){const n=normalizeAttachmentPath(e);if(!n)return;const i=normalizeAttachmentSource(
t);i!=="unknown"&&attachmentSourceByPath.set(n,i)}o(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const t=normalizeAttachmentPath(e);return t?normalizeAttachmentSource(
attachmentSourceByPath.get(t)):"unknown"}o(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,t){if(!e)return;const n=normalizeAttachmentSource(t);e.dataset.fileSource=
n;const i=e.getAttribute("data-filename");i&&setAttachmentSourceForPath(i,n)}o(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const t=normalizeAttachmentSource(
e.dataset.fileSource);if(t!=="unknown")return t;const n=e.getAttribute("data-filename");return getAttachmentSourceForPath(
n)}o(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const t=normalizeAttachmentSource(e.dataset.originalSource);if(t!=="unknown")return t;
const n=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(n)}o(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,t,n){const i=document.createElement(
"canvas");i.width=t,i.height=n;const a=i.getContext("2d");a?(a.drawImage(e,0,0,t,n),markerState.baseImageData=
a.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}o(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=o((l,c,u=null,f=!1)=>{if(!l)return;const g=Math.max(0,l.x),y=Math.max(0,l.y),w=Math.
max(1,l.w),v=Math.max(1,l.h);u&&(t.fillStyle=u,t.fillRect(g,y,w,v)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(g+.5,y+.5,Math.max(1,w-1),Math.max(1,v-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),u=Math.max(1,i.w),f=Math.max(1,i.h);t.clearRect(l,c,u,f),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((a,r,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const u=normalizeAttachmentSource(
r),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),g=t.get(c);if(g===void 0){const v=e.
length;t.set(c,v),e.push({path:c,source:u,name:f});return}const y=e[g];if(!y)return;const w=normalizeAttachmentSource(
y.source);(w==="unknown"&&u!=="unknown"||w==="library"&&u==="upload")&&(y.source=u),!normalizeAttachmentDisplayName(
y.name)&&f&&(y.name=f)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const r=a.getAttribute("data-filename");n(r,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const r=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const u=isAudioPath(
c),f=isVideoPath(c);if(u&&!t.audio||f&&!t.video){u&&(n+=1),f&&(i+=1);return}r.push(c)}),r.length!==currentImageUrls.
length&&(currentImageUrls=r)}const a=get("upload-list");if(a&&(a.querySelectorAll("[data-filename]").
forEach(r=>{const l=r.getAttribute("data-filename");l&&!currentImageUrls.includes(l)&&(isAudioPath(l)||
isVideoPath(l))&&(setRowMarkerState(r,!1),r.remove())}),a.children.length===0&&(a.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const r=[];n&&r.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&r.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${r.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}o(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const t=e.getAttribute("data-local-url");if(t)return t;const n=e.getAttribute("data-f\
ilename");return n?buildFileUrl(n):""}o(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const t=normalizeAttachmentPath(e);return t?FILE_BASE_URL+t:""}o(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const t=normalizeAttachmentPath(e);return t?isImagePath(t)?FILE_THUMB_BASE_URL+t:FILE_BASE_URL+t:""}
o(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e){
const t=getRowImageSource(e);if(!t){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e;const n=e?e.querySelector(".truncate"):null;markerState.filename=
n?n.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=[],markerState.naturalWidth=
0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=[],markerState.mosaicPreviewRect=
null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode("draw");const i=get("m\
arker-attach-original");i&&(i.checked=e.dataset.attachOriginal==="1");const a=get("marker-image"),r=get(
"marker-canvas"),l=get("marker-crop-canvas");if(r){const c=r.getContext("2d");c&&c.clearRect(0,0,r.width,
r.height)}if(l){const c=l.getContext("2d");c&&c.clearRect(0,0,l.width,l.height)}resetMarkerTransform(),
showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState({modal:"marker"},"","\
/edit-image"),a&&(a.onload=()=>{if(!get("marker-stage")||!r)return;const u=Math.max(1,Math.floor(a.clientWidth)),
f=Math.max(1,Math.floor(a.clientHeight));r.width=u,r.height=f,r.style.width=`${u}px`,r.style.height=
`${f}px`,r.style.left="0px",r.style.top="0px",l&&(l.width=u,l.height=f,l.style.width=`${u}px`,l.style.
height=`${f}px`,l.style.left="0px",l.style.top="0px"),markerState.naturalWidth=a.naturalWidth||u,markerState.
naturalHeight=a.naturalHeight||f;const g=r.getContext("2d");g&&g.clearRect(0,0,r.width,r.height),prepareMarkerBaseCanvas(
a,u,f),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},a.src=t)}o(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
o(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,u=c?c.getAttribute("data-local-url"):null;u&&URL.revokeObjectURL(
u)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.clear();const e=get(
"file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");t&&(t.innerHTML="",t.
classList.add("hidden")),updateFilePreview(),updateMaskPreview();const n=get("upload-list");n&&(n.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>');
const i=get("file-input");i&&(i.value="");const a=get("photo-input");a&&(a.value="");const r=get("ma\
sk-input");r&&(r.value="")}o(resetUploadState,"resetUploadState");async function uploadMaskFile(e){if(!e)
return;const t=new FormData;t.append("file",e);try{const n=await fetch(CHAT_CONFIG.urls.upload,{method:"\
POST",body:t}),i=await n.json();n.ok&&i.filename?(currentMaskImage=i.filename,updateMaskPreview()):showToast(
i.error||"Mask upload failed","error",!0)}catch{showToast("Mask upload failed","error",!0)}}o(uploadMaskFile,
"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){const n=get("camera-status");n&&(n.textContent=
e||"",n.classList.toggle("text-red-300",!!t),n.classList.toggle("text-gray-400",!t))}o(setCameraCaptureStatus,
"setCameraCaptureStatus");function updateCameraCapturePendingUi(){const e=cameraCapturePendingFiles.
length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,t.textContent=e?`\u6DFB\u4ED8 (${e}\
)`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=e===0||cameraCaptureBusy);const i=get(
"camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.forEach((a,r)=>{const l=document.
createElement("div");l.className="relative rounded overflow-hidden border border-gray-700 bg-black a\
spect-square",l.innerHTML=`
                        <img src="${a}" alt="capture ${r+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${r+1}</div>
                    `,i.appendChild(l)}),i.classList.toggle("hidden",e===0))}o(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const t=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(t)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}o(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(i=>{try{i.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const t=get("camera-capture-btn");t&&(t.disabled=!0);const n=get(
"camera-switch-btn");n&&(n.disabled=!0)}o(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const t=get("camera-video");if(!t)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const n=get(
"camera-switch-btn");n&&(n.disabled=!0);const i=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let a=null;for(const r of i)
try{const l=await navigator.mediaDevices.getUserMedia(r);cameraCaptureStream=l,t.srcObject=l;try{await t.
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],u=c&&c.getSettings?c.getSettings():{},
f=String(u.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
e;const g=get("camera-capture-btn");return g&&(g.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),l}catch(l){a=l}throw a||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
o(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const t=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(t,!0),showToast(t,"error",!0);const n=get("camera-capture-btn");n&&(n.disabled=
!0);const i=get("camera-attach-btn");i&&(i.disabled=!0)}}o(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const t=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!t&&location.pathname==="/camera"&&history.back()}o(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const t=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=t;try{await startCameraCaptureStream(t)}catch(n){const i=n&&
n.message?n.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
i,!0),showToast(i,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}o(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,t=o(a=>String(a).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}o(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const a=get("camera-attach-btn");a&&(a.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const r=t.getContext("2d");if(!r)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");r.drawImage(e,0,0,t.width,t.height);
const l=await new Promise((u,f)=>{t.toBlob(g=>{g?u(g):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),c=new File([l],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(c),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(l)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(r){
const l=r&&r.message?r.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
l,!0),showToast(l,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}o(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),a=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),a&&(a.disabled=!0);const r=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${r.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{await handleFiles(r,{openModal:!1}),showToast(`${r.length}\u679A\u306E\
\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(l){const c=l&&l.message?l.message:
"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";showToast(c,"\
error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.classList.contains(
"hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}o(attachCameraCapturedFiles,
"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay=="function"&&window.
hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.pathname!=="/upload"&&
history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-info");if(e){const n=(get(
"model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v4.1-flash"||n==="deepse\
ek-v4-flash-vision-exp",a=n.includes("deepseek")&&!i;e.classList.toggle("hidden",!a)}_syncVisionModelDisplay()}
o(openUploadModal,"openUploadModal");function _syncVisionModelDisplay(){const e=get("vision-model-di\
splay");if(!e)return;const t=currentVisionModel;if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(
a=>{a.id===t&&(n=a.name)})),e.textContent=n}else e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}
o(_syncVisionModelDisplay,"_syncVisionModelDisplay");function _openVisionModelSelector(){window._visionPickerActive=
!0,openModelModal(),setTimeout(()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},
50)}o(_openVisionModelSelector,"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.
hideDropOverlay=="function"&&window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname===
"/upload"&&history.back()}o(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){
const e=get("upload-list");if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(
n=>{const i=n.getAttribute("data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(
n,{source:getAttachmentSourceForPath(n),displayName:getAttachmentNameForPath(n)})}),e.children.length===
0&&(e.innerHTML='<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
o(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}o(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const a=e.split("/").pop()||e,r=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||a,l=(a.split(".").pop()||"").toLowerCase(),c=["png","jpg","jpeg","webp",
"gif"].includes(l),u=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):u,g=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=g,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=r,y.dataset.defaultDisplayName=r,y.dataset.sendNameCustomized="";const w=escapeHtml(
r),v=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",k=c?`<img src="${f}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${w}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${k}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${w}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${v}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;const _=y.querySelector(".upload-preview");_&&(_.onclick=()=>openFileViewer(u,getRowAttachmentName(
y)||r));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const L=y.
querySelector(".upload-remove");L&&(L.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const B=y.getAttribute("data-filename");B&&(currentImageUrls=currentImageUrls.
filter(K=>K!==B)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const E=y.querySelector(".upload-marker");return E&&(E.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,r),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:g}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const r=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const u=l&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let f="";l?(f=URL.createObjectURL(e),c=`<img src="${f}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${r}">`):(f=URL.createObjectURL(e),c=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${c}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${r}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">\u5F85\u6A5F\u4E2D</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${u}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `,f&&i.setAttribute("data-local-url",f);const g=i.querySelector(".upload-preview");g&&(g.
onclick=()=>{const k=i.getAttribute("data-filename"),_=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(_,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const _=i.
getAttribute("data-filename");_&&(currentImageUrls=currentImageUrls.filter(C=>C!==_)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const w=i.querySelector(".upload-marker");w&&(w.onclick=()=>openMarkerModalForRow(i));const v=i.querySelector(
".upload-send-name");return v&&(v.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(!e)
return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);try{const i=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.ok){const y=a&&a.error?a.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast(y,"error",!0),!1}const r=a.upload_id,l=a.chunk_size||10*1024*1024,c=Math.ceil(
e.size/l);for(let y=0;y<c;y++){const w=y*l,v=Math.min(e.size,w+l),k=e.slice(w,v);if(!await new Promise(
C=>{const L=new XMLHttpRequest;L.open("POST","/upload/chunk",!0),L.setRequestHeader("X-CSRF-Token",csrfToken),
L.upload.onprogress=B=>{if(B.lengthComputable&&t&&t.bar){const K=w+B.loaded,Z=Math.min(100,Math.floor(
K/e.size*100));t.bar.style.width=`${Z}%`,t.status&&(t.status.textContent=`${Z}%`),t.uploadId&&updateGlobalUploadProgress(
t.uploadId,Z)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},L.onload=()=>{L.status>=
200&&L.status<300?C(!0):C(!1)},L.onerror=()=>C(!1);const E=new FormData;E.append("upload_id",r),E.append(
"index",String(y)),E.append("total",String(c)),E.append("chunk",k,e.name),L.send(E)}))return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const u=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),f=await u.json();if(u.ok&&f&&f.filename){if(t&&t.row&&t.uploadId&&
uploadCancelTokens.has(t.uploadId))return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const v=t.
row.getAttribute("data-local-url");v&&URL.revokeObjectURL(v),t.row.removeAttribute("data-local-url");
const k=t.row.querySelector("img.upload-preview");if(k){const _=f.filename.replace(/^\d+\//,"");k.src=
buildAttachmentPreviewUrl(_)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(
y),t&&t.row&&(t.row.setAttribute("data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),
y)){const v=isRowAttachmentNameCustomized(t.row),k=defaultAttachmentDisplayName(y),_=v&&normalizeAttachmentDisplayName(
t.row.dataset.displayName)||k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,_)}return y&&
setAttachmentSourceForPath(y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(f.filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(v=>addLibraryFileFromPath(
v)),!0}const g=f&&f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(g,"error",!0),!1}catch{return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
i=!0);const a=o(()=>{i&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),i=!1)},
"finishUploadOp"),r=new XMLHttpRequest;r.open("POST",CHAT_CONFIG.urls.upload,!0),r.setRequestHeader(
"X-CSRF-Token",csrfToken),r.upload.onprogress=c=>{if(c.lengthComputable&&t&&t.bar){const u=Math.min(
100,Math.floor(c.loaded/c.total*100));t.bar.style.width=`${u}%`,t.status&&(t.status.textContent=`${u}\
%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,u)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},r.onload=()=>{let c={};try{c=JSON.parse(r.responseText||"{}")}catch{}if(r.status>=
200&&r.status<300&&c&&c.filename){if(t&&t.row&&t.uploadId&&uploadCancelTokens.has(t.uploadId)){t.row&&
t.row.parentNode&&t.row.remove(),a(),n(!1);return}if(t&&t.row){const g=t.row.getAttribute("data-loca\
l-url");g&&URL.revokeObjectURL(g),t.row.removeAttribute("data-local-url");const y=t.row.querySelector(
"img.upload-preview");if(y){const w=c.filename.replace(/^\d+\//,"");y.src=buildAttachmentPreviewUrl(
w)}}const u=normalizeAttachmentPath(c.filename);if(u&&currentImageUrls.push(u),t&&t.row&&(t.row.setAttribute(
"data-filename",u||c.filename),setRowAttachmentSource(t.row,"upload"),u)){const g=isRowAttachmentNameCustomized(
t.row),y=defaultAttachmentDisplayName(u),w=g&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
y;t.row.dataset.defaultDisplayName=y,setRowAttachmentName(t.row,w)}u&&setAttachmentSourceForPath(u,"\
upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(c.filenames)&&
c.filenames.length?c.filenames:[c.filename]).forEach(g=>addLibraryFileFromPath(g)),a(),n(!0)}else{const u=c&&
c.error?c.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(u,"error",!0),a(),n(!1)}},r.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),a(),n(!1)};const l=new FormData;l.append("file",e),r.send(l)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(f=>{n+=f.length});const i=new Float32Array(n);
let a=0;e.forEach(f=>{i.set(f,a),a+=f.length});const r=new ArrayBuffer(44+i.length*2),l=new DataView(
r),c=o((f,g)=>{for(let y=0;y<g.length;y++)l.setUint8(f+y,g.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let u=44;for(let f=0;f<i.length;f++){const g=Math.
max(-1,Math.min(1,i[f]));l.setInt16(u,g<0?g*32768:g*32767,!0),u+=2}return new Blob([l],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const r=e.row.getAttribute("data-local-url");r&&URL.revokeObjectURL(
r);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),u=escapeHtml(a),f=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${u}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',g=e.row.querySelector(".upload-preview");g&&(g.outerHTML=f);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const v=e.row.getAttribute("data-filename"),k=v?buildFileUrl(v):
e.row.getAttribute("data-local-url");openFileViewer(k,getRowAttachmentName(e.row)||a||v||"")});const w=e.
row.querySelector(".upload-marker");w&&w.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}o(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const t=e.getContext("2d");if(!t)return;const n=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[];markerState.history.push({imageData:t.
getImageData(0,0,e.width,e.height),mosaicRects:n}),markerState.history.length>40&&markerState.history.
shift()}o(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const t=e.getContext(
"2d");if(!t)return;const n=markerState.history[markerState.history.length-1];t.clearRect(0,0,e.width,
e.height),n&&n.imageData?(t.putImageData(n.imageData,0,0),markerState.mosaicRects=Array.isArray(n.mosaicRects)?
n.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[]):n?(t.putImageData(n,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}o(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const t=e.getContext("2d");t&&t.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}o(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let a=!1,r=0,
l=markerView.scale,c={x:0,y:0},u={x:0,y:0},f=[],g=16,y="",w=null,v=null,k=null,_=null,C=!1,L=null;const E=o(
A=>{const F=e.getBoundingClientRect(),U=(A.clientX-F.left)*(e.width/F.width),X=(A.clientY-F.top)*(e.
height/F.height);return{x:U,y:X}},"getPoint"),B=o((A,F)=>({x:(A.x+F.x)/2,y:(A.y+F.y)/2}),"getMid"),K=o(
(A,F)=>Math.hypot(A.x-F.x,A.y-F.y),"getDist");let Z=!1;const Ie=o(()=>{w||(w=document.createElement(
"canvas"),v=w.getContext("2d")),k||(k=document.createElement("canvas"),_=k.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),O=o(()=>{if(!t||!w||!k)return;const A=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(0,0,e.width,e.
height),t.drawImage(w,0,0),t.save(),t.globalAlpha=A,t.drawImage(k,0,0),t.restore()},"renderDrawPrevi\
ew"),G=o(()=>{_&&(_.strokeStyle=y,_.fillStyle=y,_.lineWidth=g,_.lineCap="round",_.lineJoin="round")},
"applyMarkerBrush"),te=o(A=>{if(!A)return!1;if(f.length===0)return f.push(A),!0;const F=f[f.length-1],
U=A.x-F.x,X=A.y-F.y,ie=Math.hypot(U,X),Te=Math.max(.35,g*.04);if(ie<Te)return!1;const ne=Math.max(1,
g*.25),pe=Math.max(1,Math.ceil(ie/ne));for(let V=1;V<=pe;V++){const we=V/pe;f.push({x:F.x+U*we,y:F.y+
X*we})}return!0},"appendStrokePoint"),be=o(()=>{if(_&&(_.clearRect(0,0,k.width,k.height),f.length!==
0)){if(G(),f.length===1){const A=f[0];_.beginPath(),_.arc(A.x,A.y,g/2,0,Math.PI*2),_.fill();return}if(_.
beginPath(),_.moveTo(f[0].x,f[0].y),f.length===2)_.lineTo(f[1].x,f[1].y);else{for(let U=1;U<f.length-
2;U++){const X=f[U],ie=f[U+1],Te=B(X,ie);_.quadraticCurveTo(X.x,X.y,Te.x,Te.y)}const A=f[f.length-2],
F=f[f.length-1];_.quadraticCurveTo(A.x,A.y,F.x,F.y)}_.stroke()}},"renderStrokeLayer"),ue=o((A,F)=>{if(!A||
!F)return null;const U=Math.min(A.x,F.x),X=Math.min(A.y,F.y),ie=Math.abs(A.x-F.x),Te=Math.abs(A.y-F.
y);return{x:U,y:X,w:ie,h:Te}},"normalizeMosaicRect"),Le=o(A=>{const F=n?Number(n.value||16):16,U=Math.
max(6,Math.floor(F)),X=Math.floor(U/2);return{x:A.x-X,y:A.y-X,w:U,h:U}},"buildMosaicRectFromPoint"),
ye=o(()=>{const A=document.createElement("canvas");A.width=e.width,A.height=e.height;const F=A.getContext(
"2d");if(!F)return null;markerState.baseCanvas&&F.drawImage(markerState.baseCanvas,0,0),F.drawImage(
e,0,0);try{return F.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),
Me=o(A=>{if(!t||!A)return!1;const F=ye();if(!F)return!1;const U=n?Number(n.value||16):16,X=Math.max(
4,Math.floor(U/2)),ie=Math.max(0,Math.floor(A.x)),Te=Math.max(0,Math.floor(A.y)),ne=Math.min(e.width,
Math.ceil(A.x+A.w)),pe=Math.min(e.height,Math.ceil(A.y+A.h));if(ne<=ie||pe<=Te)return!1;for(let V=Te;V<
pe;V+=X)for(let we=ie;we<ne;we+=X){const ot=Math.min(X,ne-we),Pe=Math.min(X,pe-V),dt=Math.min(e.width-
1,Math.max(0,we+Math.floor(ot/2))),Ze=(Math.min(e.height-1,Math.max(0,V+Math.floor(Pe/2)))*e.width+dt)*
4,mt=F.data[Ze],Tt=F.data[Ze+1],xt=F.data[Ze+2];t.fillStyle=`rgb(${mt},${Tt},${xt})`,t.fillRect(we,V,
ot,Pe)}return!0},"applyMosaicRect"),oe=o(A=>{if(!t)return;if(i.set(A.pointerId,{x:A.clientX,y:A.clientY}),
i.size>=2){const U=Array.from(i.values()),X=U[0],ie=U[1];a=!0,Z=!1,f=[],C=!1,L=null,markerState.mosaicPreviewRect=
null,r=K(X,ie)||1,l=markerView.scale,c={x:markerView.offsetX,y:markerView.offsetY},u=B(X,ie),renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(A.pointerId),A.preventDefault();return}if(a||markerState.mode===
"crop")return;Z=!0;const F=E(A);if(markerState.mode==="mosaic")C=!0,L=F,markerState.mosaicPreviewRect=
Le(F),renderCropOverlay();else{if(Ie(),!v||!_)return;v.clearRect(0,0,w.width,w.height),v.drawImage(e,
0,0),_.clearRect(0,0,k.width,k.height),g=n?Number(n.value||16):16,y=normalizeMarkerHexColor(markerState.
colorHex),f=[],te(F),be(),markerState.hasStroke=!0,O()}e.setPointerCapture&&e.setPointerCapture(A.pointerId),
A.preventDefault()},"start"),re=o(A=>{if(i.has(A.pointerId)&&i.set(A.pointerId,{x:A.clientX,y:A.clientY}),
a&&i.size>=2){const U=Array.from(i.values()),X=U[0],ie=U[1],Te=B(X,ie),ne=K(X,ie)||1,pe=l*(ne/r);markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,pe)),markerView.offsetX=c.x+(Te.x-u.
x),markerView.offsetY=c.y+(Te.y-u.y),applyMarkerTransform(),A.preventDefault();return}if(!Z||!t)return;
const F=E(A);if(markerState.mode==="mosaic"){if(!C||!L)return;markerState.mosaicPreviewRect=ue(L,F)||
Le(F),renderCropOverlay()}else te(F)&&(be(),O());A.preventDefault()},"move"),J=o(A=>{const F=Z;if(i.
delete(A.pointerId),i.size<2&&(a=!1),i.size===0){if(Z=!1,F&&t&&markerState.mode==="draw"&&f.length>0&&
(be(),O()),F&&markerState.mode==="mosaic"&&L){const U=E(A);let X=ue(L,U);(!X||X.w<2||X.h<2)&&(X=Le(L)),
Me(X)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(X))}f=[],C=!1,L=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),F&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(A.pointerId),
A.preventDefault()},"end");e.addEventListener("pointerdown",oe),e.addEventListener("pointermove",re),
e.addEventListener("pointerup",J),e.addEventListener("pointercancel",J)}o(initMarkerCanvas,"initMark\
erCanvas");function initCropCanvas(){const e=get("marker-crop-canvas");if(!e)return;const t=e.getContext(
"2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,u=0,f=markerView.scale,g={x:0,y:0},y={x:0,y:0};const w=8,
v=14,k=o((O,G,te)=>Math.min(te,Math.max(G,O)),"clamp"),_=o(O=>{const G=e.getBoundingClientRect(),te=(O.
clientX-G.left)*(e.width/G.width),be=(O.clientY-G.top)*(e.height/G.height);return{x:te,y:be}},"getPo\
int"),C=o((O,G)=>({x:(O.x+G.x)/2,y:(O.y+G.y)/2}),"getMid"),L=o((O,G)=>Math.hypot(O.x-G.x,O.y-G.y),"g\
etDist"),E=o(()=>(markerState.cropRect||resetCropRectToFull(),markerState.cropRect),"ensureCropRect"),
B=o((O,G)=>{if(!G)return"move";const te=G.x,be=G.y,ue=G.x+G.w,Le=G.y+G.h,ye=Math.abs(O.x-te)<=v,Me=Math.
abs(O.x-ue)<=v,oe=Math.abs(O.y-be)<=v,re=Math.abs(O.y-Le)<=v;if(ye&&oe)return"nw";if(Me&&oe)return"n\
e";if(ye&&re)return"sw";if(Me&&re)return"se";if(oe)return"n";if(re)return"s";if(ye)return"w";if(Me)return"\
e";if(O.x>te+v&&O.x<ue-v&&O.y>be+v&&O.y<Le-v)return"move";const A=O.x<te?"left":O.x>ue?"right":null,
F=O.y<be?"top":O.y>Le?"bottom":null;if(A&&F){if(A==="left"&&F==="top")return"nw";if(A==="right"&&F===
"top")return"ne";if(A==="left"&&F==="bottom")return"sw";if(A==="right"&&F==="bottom")return"se"}return A?
A==="left"?"w":"e":F?F==="top"?"n":"s":"move"},"hitTest"),K=o(O=>{if(markerState.mode!=="crop")return;
if(n.set(O.pointerId,{x:O.clientX,y:O.clientY}),n.size>=2){const be=Array.from(n.values()),ue=be[0],
Le=be[1];c=!0,i=!1,u=L(ue,Le)||1,f=markerView.scale,g={x:markerView.offsetX,y:markerView.offsetY},y=
C(ue,Le),e.setPointerCapture&&e.setPointerCapture(O.pointerId),O.preventDefault();return}if(c)return;
i=!0;const G=_(O),te=E();r=B(G,te),a=G,l=te?{x:te.x,y:te.y,w:te.w,h:te.h}:null,renderCropOverlay(),e.
setPointerCapture&&e.setPointerCapture(O.pointerId),O.preventDefault()},"start"),Z=o(O=>{if(markerState.
mode!=="crop")return;if(n.has(O.pointerId)&&n.set(O.pointerId,{x:O.clientX,y:O.clientY}),c&&n.size>=
2){const A=Array.from(n.values()),F=A[0],U=A[1],X=C(F,U),ie=L(F,U)||1,Te=f*(ie/u);markerView.scale=Math.
min(markerView.maxScale,Math.max(markerView.minScale,Te)),markerView.offsetX=g.x+(X.x-y.x),markerView.
offsetY=g.y+(X.y-y.y),applyMarkerTransform(),renderCropOverlay(),O.preventDefault();return}if(!i||!a||
!l)return;const G=_(O),te=e.width,be=e.height,ue={x:l.x,y:l.y,w:l.w,h:l.h},Le=l.x+l.w,ye=l.y+l.h,Me=o(
()=>{const A=k(G.x,0,Le-w);ue.x=A,ue.w=Le-A},"applyW"),oe=o(()=>{ue.w=k(G.x-l.x,w,te-l.x)},"applyE"),
re=o(()=>{const A=k(G.y,0,ye-w);ue.y=A,ue.h=ye-A},"applyN"),J=o(()=>{ue.h=k(G.y-l.y,w,be-l.y)},"appl\
yS");switch(r){case"move":{const A=G.x-a.x,F=G.y-a.y;ue.x=k(l.x+A,0,te-l.w),ue.y=k(l.y+F,0,be-l.h);break}case"\
w":Me();break;case"e":oe();break;case"n":re();break;case"s":J();break;case"nw":re(),Me();break;case"\
ne":re(),oe();break;case"sw":J(),Me();break;case"se":J(),oe();break;default:break}ue.x=k(ue.x,0,te-ue.
w),ue.y=k(ue.y,0,be-ue.h),markerState.cropRect=ue,renderCropOverlay(),O.preventDefault()},"move"),Ie=o(
O=>{n.delete(O.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),
e.releasePointerCapture&&e.releasePointerCapture(O.pointerId),O.preventDefault()},"end");e.addEventListener(
"pointerdown",K),e.addEventListener("pointermove",Z),e.addEventListener("pointerup",Ie),e.addEventListener(
"pointercancel",Ie),e.addEventListener("pointerleave",Ie)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const r=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;a.width=r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,0,r,l),c.drawImage(
n,0,0,r,l),markerState.cropRect){const C=r/n.width,L=l/n.height,E=Math.max(0,Math.floor(markerState.
cropRect.x*C)),B=Math.max(0,Math.floor(markerState.cropRect.y*L)),K=Math.min(r,Math.max(1,Math.floor(
markerState.cropRect.w*C))),Z=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.h*L))),Ie=document.
createElement("canvas");Ie.width=K,Ie.height=Z;const O=Ie.getContext("2d");O&&(O.drawImage(a,E,B,K,Z,
0,0,K,Z),a=Ie)}const u=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!u){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0);return}const g=(markerState.filename||"marked.png").replace(
/\.[^/.]+$/,""),y=new File([u],`${g}_marked.png`,{type:"image/png"}),w={row:e,uploadId:e.dataset.uploadId,
status:e.querySelector(".upload-status"),bar:e.querySelector(".upload-progress > div")};w.status&&(w.
status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),updateUploadRowFile(w,y);const v=e.getAttribute(
"data-filename"),k=getRowAttachmentSource(e);v&&!e.dataset.originalFilename&&(e.dataset.originalFilename=
v,e.dataset.originalSource=k,setAttachmentSourceForPath(v,k)),await uploadFileWithProgress(y,w)?(v&&
(currentImageUrls=currentImageUrls.filter(C=>C!==v)),setRowAttachmentSource(e,"upload"),setRowMarkerState(
e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}o(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,i.src=URL.createObjectURL(
e);let a=null,r=null,l=null,c=null,u=[],f=null;const g=o(()=>{f&&clearTimeout(f);try{URL.revokeObjectURL(
i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(w=>w.stop()),l)try{l.disconnect()}catch{}
if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(()=>{g(),n(null)},"fail");
i.onloadedmetadata=async()=>{try{a=i.captureStream();const w=a.getAudioTracks();if(!w||!w.length)return y();
r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),c=r.createMediaStreamSource(
new MediaStream(w)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=k=>{const _=k.inputBuffer.getChannelData(
0);u.push(new Float32Array(_))},c.connect(l),l.connect(r.destination);const v=isFinite(i.duration)?Math.
max(1,Math.ceil(i.duration*1e3)):0;v>0&&(f=setTimeout(()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,
""),_=encodeWav(u,r.sampleRate),C=new File([_],`${k}.audio.wav`,{type:"audio/wav"});g(),n(C)},v+250)),
await i.play(),i.onended=()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,""),_=encodeWav(u,r.sampleRate),
C=new File([_],`${k}.audio.wav`,{type:"audio/wav"});g(),n(C)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,t={}){if(!e||!e.length)return;const n=Array.from(
e).filter(Boolean);if(!n.length)return;const i=collectImageUrlsForSend().length+browserFastLocalFiles.
size+Math.max(0,Number(uploadProgressState.active)||0);let a=n;if(i+n.length>ATTACHMENT_MAX_FILES){const y=Math.
max(0,ATTACHMENT_MAX_FILES-i);if(y<=0){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059`,"error",!0);return}a=n.slice(0,y),showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u5148\u982D${y}\u4EF6\u306E\u307F\u8FFD\u52A0\u3057\u307E\u3059\u3002`,"war\
ning",!0)}t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=
a.length,uploadProgressState.active+=a.length,updateFilePreview();const r=!!(get("upload-audio-only")&&
get("upload-audio-only").checked),l=getModelMediaSupport(get("model-select").value),c=o(async y=>{let w=null;
try{if(isAudioFile(y)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(y)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!y.type||!y.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const v=addUploadRow(y);updateFilePreview(),w=v.uploadId,uploadProgressState.perFilePct[w]=
0;let k=y;if(r&&isVideoFile(y)){const _=await extractAudioFromVideo(y,v);_?(k=_,updateUploadRowFile(
v,_),v&&v.status&&(v.status.textContent="\u97F3\u58F0\u306E\u307F")):(v&&v.status&&(v.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&y.type.startsWith("image/"))try{const _=getCompressionOutputType();
if(getCompressionFormatOnly())k=await convertImageFormatOnly(y,_);else{const L={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};_&&_!=="original"&&(L.fileType=_),await ensureImageCompression();
const E=await window.imageCompression(y,L),B=new File([E],imageFilenameForMime(y.name,E.type||(_!=="\
original"?_:y.type)),{type:E.type||y.type,lastModified:y.lastModified||Date.now()});B.size>y.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
y.size)} -> ${formatBytes(B.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),k=y):k=B}k!==y&&updateUploadRowFile(v,k)}catch{}if(browserFastModeEnabled){const _=Array.from(
browserFastLocalFiles.values()).reduce((C,L)=>C+Number(L.file&&L.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||_+k.size>BROWSER_FAST_MAX_BYTES?(v&&v.status&&(v.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),v&&v.row&&v.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(v.uploadId,{file:k,rowObj:v}),v.status&&(v.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),v.bar&&(v.bar.style.width="100\
%"),v.row&&(v.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(k,v)}finally{
w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.perFilePct[w],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let u=0;const f=Math.min(UPLOAD_CONCURRENCY,a.length),g=Array.from({length:f}).map(async()=>{
for(;;){const y=u++;if(y>=a.length)break;await c(a[y])}});await Promise.all(g)}o(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){const n=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).replace(/\n/g,
"<br>");const i=protectMathSegments(n),a=window.marked.parse(i.text),r=restoreMathSegments(a,i.blocks,
t);return window.DOMPurify.sanitize(r)}o(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");function getCanvasModeElements(){
const e=get("canvas-panel");return e?{panel:e,stage:get("conversation-stage"),title:get("canvas-pane\
l-title"),status:get("canvas-panel-status"),blockCount:get("canvas-block-count"),blockList:get("canv\
as-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get("canvas-preview-lang"),sourceSelect:get(
"canvas-source-select"),frame:get("canvas-preview-frame"),empty:get("canvas-preview-empty"),sourceScroll:get(
"canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get("canvas-panel-copy-btn"),clearBtn:get(
"canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:null}o(getCanvasModeElements,"getC\
anvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(t||"");return/<!doctype\s+html/i.
test(i)||/<html[\s>]/i.test(i)}o(isCanvasHtmlPreviewCandidate,"isCanvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){
const n=String(e&&e.lang?e.lang:"").trim(),i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),a=!!(e&&
e.open);return{...e,index:t,lang:n,code:i,open:a,key:hashString(`${n||"TEXT"}
${i||""}`)}}o(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],a=[],r=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",u=[];for(const y of n){
if(!l){const k=y.match(r);if(k){l=k[2],c=String(k[3]||"").trim(),u=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const w=String(y||"").trim();if(w&&w.replace(/\s+/g,"")===l){const k=i[i.length-1];k&&(k.
code=u.join(`
`),k.open=!1),l=null,c="",u=[];continue}u.push(y);const v=i[i.length-1];v&&(v.code=u.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=u.join(`
`),y.open=!0)}const f=i.map((y,w)=>normalizeCanvasBlock(y,w)),g=selectCanvasPreviewBlock(f,t);return{
renderText:a.join(`
`),blocks:f,primaryBlock:g?g.block:null,primaryIndex:g?g.index:-1,rawText:t}}o(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const r=i[n];return{block:r,index:n,previewType:isCanvasHtmlPreviewCandidate(
r.lang,r.code)?"html":"code"}}if(i.length>0){const r=i.length-1,l=i[r];return{block:l,index:r,previewType:isCanvasHtmlPreviewCandidate(
l.lang,l.code)?"html":"code"}}const a=String(t||"");return isCanvasHtmlPreviewCandidate("",a)?{block:normalizeCanvasBlock(
{lang:"html",code:a,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}o(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}o(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),a=!!t&&t===i;n.classList.toggle("can\
vas-active",a),n.setAttribute("aria-pressed",a?"true":"false"),n.setAttribute("data-canvas-active",a?
"1":"0"),n.innerHTML=a?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
o(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}o(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
a={preview:0,blocks:1,source:2},r=i[n];if(!r||!(t in a)||!(n in a))return;canvasPreviewState.viewAnimationToken+=
1;const l=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(u=>{u&&u.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),r.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";r.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(r.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}o(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(u=>{const f=u.getAttribute("da\
ta-canvas-panel-view")===i;u.classList.toggle("active",f),u.setAttribute("aria-pressed",f?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}o(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const r=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",g=(String(i&&i.code?i.code:"").split(/\r?\n/).find(v=>v.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${r}`,w=`${y}\u3001${g}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(w)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
r)}</span><span class="canvas-block-chip-preview">${escapeHtml(g)}</span></span><span class="canvas-\
block-chip-state">${l?"\u8868\u793A\u4E2D":c}</span></button>`}).join("")}o(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((r,l)=>{const c=String(
r&&r.lang?r.lang:"text").trim()||"text";return`#${l+1} ${c}`}),a=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==a&&(e.sourceSelect.innerHTML=i.map((r,l)=>`<option value="${l}">${escapeHtml(
r)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=a),e.sourceSelect.value=String(
n)}o(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}o(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),a=String(e||""),r=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const l=new DOMParser().parseFromString(a,"text/html"),c=l.createElement("scr\
ipt");return c.setAttribute("data-canvas-scroll-bridge","true"),c.textContent=r,(l.body||l.documentElement).
appendChild(c),`<!DOCTYPE html>
`+l.documentElement.outerHTML}catch{return`${a}<script data-canvas-scroll-bridge>${r}<\/script>`}}o(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const t=e&&e.data?e.data:null;if(!t||t.type!=="canvas-preview-scroll")return;const n=getCanvasModeElements();
!n||!n.frame||e.source!==n.frame.contentWindow||t.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(t.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(t.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const t=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{t===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}o(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const t=getCanvasModeElements();if(t){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){t.panel.classList.add("hidden"),t.panel.classList.remove(
"canvas-panel-open","canvas-closing"),t.stage&&t.stage.classList.remove("canvas-enabled");return}t.panel.
classList.remove("canvas-panel-open"),t.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{t.panel.classList.add("hidden"),t.panel.classList.remove("canvas-closing"),t.
stage&&t.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}o(hideCanvasPreviewPanel,
"hideCanvasPreviewPanel");function resetCanvasPreviewPanel(e="Canvas\u3067\u8868\u793A\u4E2D"){const t=getCanvasModeElements();
t&&(canvasPreviewState.blocks=[],canvasPreviewState.rawText="",canvasPreviewState.renderText="",canvasPreviewState.
selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.selectionMode="auto",canvasPreviewState.
mobileView="preview",canvasPreviewState.lastCanvasData=null,resetCanvasScrollState(),showCanvasPreviewPanel(),
syncCanvasPanelViewUi("preview",{focus:!1}),t.title&&(t.title.textContent=e),t.status&&(t.status.textContent=
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D"),t.previewLang&&(t.previewLang.
textContent="idle"),t.sourceSelect&&(t.sourceSelect.innerHTML='<option value="">-</option>',t.sourceSelect.
disabled=!0,t.sourceSelect.dataset.canvasOptionsSignature=""),t.code&&(t.code.textContent=""),t.blockCount&&
(t.blockCount.textContent="0"),t.blockList&&(t.blockList.innerHTML='<div class="px-2 py-3 text-xs te\
xt-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>'),t.sourceScroll&&
(t.sourceScroll.scrollTop=0),t.frame&&(t.frame.removeAttribute("srcdoc"),t.frame.classList.add("hidd\
en")),t.empty&&t.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}o(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const t=e||canvasPreviewState.lastCanvasData;
if(!t)return null;canvasPreviewState.lastCanvasData=t,canvasPreviewState.blocks=Array.isArray(t.blocks)?
t.blocks.slice():[],canvasPreviewState.rawText=String(t.rawText||""),canvasPreviewState.renderText=String(
t.renderText||"");const n=canvasPreviewState.blocks,i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!n.length){const l=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return l&&l.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=l.block.
key||"",l.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let a=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?a=i:canvasPreviewState.selectionMode="auto";const r=n[a]||null;return canvasPreviewState.
selectedIndex=r?a:-1,canvasPreviewState.selectedKey=r&&r.key?r.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),r}o(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,a=n&&Number.isInteger(
n.index)?n.index:-1,r=!!i,l=String(i&&i.lang?i.lang:"").trim(),c=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),u=r?isCanvasHtmlPreviewCandidate(l,c):!1,f=r?u?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",g=r?u?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=g),e.status&&(e.status.textContent=
f),e.previewLang&&(e.previewLang.textContent=r?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,w=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=w,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),r){canvasPreviewState.frameRenderToken+=1;const v=canvasPreviewState.frameRenderToken,
k=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),v);e.frame&&(e.frame.srcdoc=k,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{v!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:v,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}o(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const a=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
a&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}o(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(a=>a&&a.key===n);return i===-1?!1:applyCanvasSelection(i)}
o(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
a=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:a}}o(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const c=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[c],selectedIndex:0,selectedKey:c.key||t.codeKey||""}}const a=[];let r=-1;if(i.forEach(
(c,u)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const g=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},u);a.push(g),r===-1&&f.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
r===-1&&(r=0);const l=a[r]||a[0]||null;return{blocks:a,selectedIndex:r,selectedKey:l&&l.key?l.key:t.
codeKey||""}}o(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
if(!e)return!1;const t=collectCanvasBlocksFromButton(e);if(!t||!t.blocks||!t.blocks.length)return!1;
const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[],i=n.findIndex(r=>r&&r.
key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(i);const a=t.blocks[t.selectedIndex]||
t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.rawText=a&&a.code!==void 0&&
a.code!==null?String(a.code):"",canvasPreviewState.renderText=canvasPreviewState.rawText,canvasPreviewState.
selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.selectedKey=t.selectedKey||
a&&a.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),canvasPreviewState.lastCanvasData=
{renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:a,primaryIndex:canvasPreviewState.
selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.mobileView="preview",syncCanvasPanelViewUi(
"preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}o(previewCanvasCodeFromButton,"previewCanvasCod\
eFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&e.code!==null?
e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(n,t))return sanitizeHtmlForPreview(
t);const a=n?`Canvas Preview: ${n}`:"Canvas Preview",r=escapeHtml(t||"");return`<!doctype html><html\
 lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-sc\
ale=1"><title>${escapeHtml(a)}</title><style>
                :root { color-scheme: dark; }
                html, body { margin: 0; min-height: 100%; background: #0b1220; color: #e5e7eb; font-\
family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
                body { box-sizing: border-box; padding: 16px; }
                .frame {
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94)\
);
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 14px;
                    padding: 14px;
                    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
                }
                .label {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: #67e8f9;
                    margin-bottom: 10px;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                    font-family: "JetBrains Mono", "Noto Sans Mono", ui-monospace, SFMono-Regular, M\
enlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #e2e8f0;
                }
                .muted { color: #94a3b8; }
            </style></head><body><div class="frame"><div class="label">${escapeHtml(a)}</div><pre>${r||
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}o(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,t={}){const n=t.persist!==
!1;if(canvasModeEnabled=!!e,n)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const i=get("enable-canvas-mode");if(i&&i.checked!==canvasModeEnabled&&(i.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(t.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!t.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}o(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}o(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,t,n){let i=normalizeMarkdownNewlines(
e);const a=normalizeMarkdownNewlines(n);if(!a&&a!=="")return i;const r=t?[String(t),""]:[""];for(const l of[
"`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);for(const f of r){const g=`${u}${f}
`,y=`
${u}`,w=g+a+y;i.includes(w)&&(i=i.split(w).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const r of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);
[`${r}${u}
${i}
${u}`,`${r}
${u}
${i}
${u}`,`
${r}${u}
${i}
${u}`,`
${r}
${u}
${i}
${u}`].forEach(g=>{n.includes(g)&&(n=n.split(g).join(`
`))})}return n}o(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}o(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}o(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[];if(!t)return{text:"",executions:n};const i=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let a=t.replace(i,(r,l,c)=>{const u=String(c||"").trim();try{const f=JSON.parse(u);n.push({code:f&&f.
code!=null?String(f.code):"",output:f&&f.output!=null?String(f.output):""})}catch{n.push({code:u,output:""})}
return`
`});return n.forEach(r=>{r.code&&(a=stripExactFencedBlock(a,"python",r.code),a=stripExactFencedBlock(
a,"py",r.code)),a=stripVisiblePythonOutputBlock(a,r.output)}),a=a.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:a,executions:n}}o(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(r=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(r)?n.push(r.trim()):
i.push(r)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}o(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}o(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",a=e&&e.output!=null?String(e.output):"";let r="";try{window.
hljs&&typeof window.hljs.highlight=="function"?r=window.hljs.highlight(i,{language:"python"}).value:
r=escapeHtml(i)}catch{r=escapeHtml(i)}const l=escapeHtml(a),c=encodeURIComponent(i).replace(/'/g,"%2\
7"),u=encodeURIComponent(a).replace(/'/g,"%27"),f=hashString(`pyexec-detail
${i}
${a}
${t}`),g=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",y=`<button class="download-btn" data-\
code="${c}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,w=`<button class="coding-target-btn" data-code="${c}" data-code-key="${f}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${f}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(g)}</span><div class="code-actions">${w}${y}<button class="copy-btn" data-copy="code" \
data-code="${c}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${u}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${r}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${l}</code></pre></div></div></div>`}o(buildPythonExecDetailBoxHtml,"buildP\
ythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/python-\
execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"","/py\
thon-execution")}showModal("python-exec-modal")}o(showPythonExecDetailModal,"showPythonExecDetailMod\
al");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("pytho\
n-exec-modal-body"),a=get("python-exec-modal-title");if(!n||!i)return;const r=t&&Array.isArray(t.python_executions)?
t.python_executions:[];if(!r.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(a){const l=r.length>1?`\uFF08${r.length}\u4EF6\uFF09`:"";a.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${l}`}i.innerHTML=r.map((l,c)=>buildPythonExecDetailBoxHtml(l,c,r.length)).join(
""),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(!0,{persist:!1})),showPythonExecDetailModal(
e)}o(openPythonExecDetail,"openPythonExecDetail"),window.openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){
get("python-exec-modal")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&
history.back())}o(closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;
function buildAiMarkdownHtml(e){const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(
t.text,t.notes),i=canvasModeEnabled?parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,
rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(i),refreshCanvasPreviewPanel());const a=document.
createElement("div");return a.className="prose prose-invert text-sm break-words",a.innerHTML=sanitizeMarkdownHtml(
i.renderText),wrapRenderedSvgBoxes(a),lowBandwidthMode||(maybeNeedsHighlight(i.renderText,a)&&ensureHighlightLoaded().
catch(()=>{}),maybeNeedsMathJax(i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),a.outerHTML}o(buildAiMarkdownHtml,
"buildAiMarkdownHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes),r=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,
blocks:[],primaryBlock:null,rawText:a};if(canvasModeEnabled&&(updateCanvasPreviewState(r),refreshCanvasPreviewPanel()),
n.incrementalMath){const l=document.createElement("template");l.innerHTML=sanitizeMarkdownHtml(r.renderText,
{streamMathSegments:!0});const c=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(f=>{const g=f.getAttribute("data-stream-math-key");g&&c.set(g,f)});const u=[];l.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(f=>{const g=c.get(f.getAttribute(
"data-stream-math-key"));g?f.replaceWith(g):u.push(f)}),e.replaceChildren(l.content),wrapRenderedSvgBoxes(
e),queueHighlight(e,r.renderText),queueIncrementalMathTypeset(u);return}e.innerHTML=sanitizeMarkdownHtml(
r.renderText),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,r.renderText)}o(renderAiMarkdownInto,
"renderAiMarkdownInto");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}o(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");function renderMessage(e,t,n,i,a,r,l=null,c=!0,u=null,f=null,g=null,y=null,w=null,v=null,k=null,_=null,C=!0,L=null,E=null,B=null){
const K=t==="user",Z=K?"bg-blue-600":"bg-gray-700",Ie=K?"justify-end":"justify-start";messageStore[e]=
n;const O=!K&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},G=K?n:O.text;let te=f;
if(te==null){const ne=g!=null?Number(g):0,pe=y!=null?Number(y):0;(g!=null||y!=null)&&(te=ne+pe)}messageMeta[e]=
{tokens_in:g,tokens_out:y,tokens_total:te,tokens_content:v,tokens_thought:k,is_encrypted:w,role:t,model:r,
parent_id:L,quote_text:u,image_url:i,gem_name:E,batch_job:B,python_executions:K?[]:O.executions||[]};
let be="";u&&(be=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-g\
ray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(u)}\
</div>`);let ue="";if(a&&!K){let ne="";try{ne=JSON.parse(a).text||""}catch{ne=a}ne&&(ue=`<div class=\
"thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brai\
n text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(ne)}\
</div></div>`)}let Le="";if(i)try{const ne=JSON.parse(i);if(ne.length){const pe=[];if(ne.forEach(V=>{
let we=V,ot="unknown";if(we&&typeof we=="object"&&(ot=normalizeAttachmentSource(we.source),we=we.filepath||
we.path||we.url||we.file||""),we=normalizeAttachmentPath(we)||we,!we)return;setAttachmentSourceForPath(
we,ot);const Pe=we.replace(/^\d+\//,""),dt=buildFileUrl(Pe),bt=buildAttachmentPreviewUrl(Pe),Ze=we.split(
"/").pop(),mt=Ze.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(mt)?pe.push(
buildChatImageHtml(bt,{viewerSrc:dt,alt:Ze,title:Ze,filename:Ze})):pe.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${dt}')" title="${Ze}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Ze}</span></div>`)}),pe.length>
0){let V="grid-multi";pe.length===1?V="grid-1":pe.length===2?V="grid-2":pe.length===3?V="grid-3":pe.
length===4&&(V="grid-4"),Le=`<div class="image-grid ${V}">${pe.join("")}</div>`}}}catch{}const ye=K?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,Me=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${K?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${ye}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
oe=[];!K&&r&&oe.push(escapeHtml(r)),E&&(K?oe.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml(E)}</span>`):oe.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml(E)}</span>`));const re=[];if(g!=null&&re.push(`In ${g}`),y!=null){let ne=`\
Out ${y}`;k!=null&&Number(k)>0&&(ne+=` (Thought ${k})`),re.push(ne)}if(re.length||f!=null){const ne=re.
length?re.join(" / "):`${f} tokens`;oe.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${ne}</button>`)}if(w!=null){const ne=w?"fa\
-lock":"fa-lock-open",pe=isAdminUser?w?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":w?"Encrypted":
"Plain",V=isAdminUser?w?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";oe.push(`<button class="${V}" title="${pe}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${ne}"></i></button>`)}if(!K&&O.executions&&O.executions.length){
const ne=O.executions.length,pe=ne>1?`Python \xD7${ne}`:"Python";oe.push(`<button type="button" clas\
s="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\
\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${pe}</span></button>`)}const J=oe.
length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${oe.
join(" \u2022 ")}</div>`:"";let A;const F=!K&&B?(()=>{const ne=String(B.state||"").toUpperCase(),pe=B.
status_text||(ne==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ne==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ne==="JOB_STATE_FAILED"||
ne==="JOB_STATE_CANCELLED"||ne==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ne==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(pe)}</div></div>`})():
"";K?A=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(A=F+(G&&String(G).trim()?buildAiMarkdownHtml(G):B?'<div class="content-area prose prose\
-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(G)),A.includes("content-area")||(A=A.replace("prose ","content-area prose ")));let U="";
if(l){const ne=l.siblings[l.current-2],pe=l.siblings[l.current];U=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ne}\
)" ${ne?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${pe}\
)" ${pe?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const X=c?"fade-in":"",ie=document.createElement("div");ie.className=`flex ${Ie} m\
b-4 ${X} relative message-group group`,ie.id=`msg-${e}`,ie.innerHTML=`<div class="message-bubble ${Z}\
 text-white p-4 rounded-2xl shadow-md relative">${Me}${be}${ue}${A}${Le}${U}${J}</div>`;const Te=_||
get("chat-container");return Te&&(Te.appendChild(ie),C&&scrollToBottom(),K||(queueMessageDecorations(
ie,G),syncCodingTargetButtons(ie),syncCodingModeUi(codingModeEnabled,{persist:!1}))),ie}o(renderMessage,
"renderMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const t={
modal:"token-details"};e!==null&&(t.messageId=e),history.pushState(t,"","/token-details")}showModal(
"token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];
if(!t||!get("token-detail-modal"))return;const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",r=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",u=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detai\
l-in").innerText=a,get("token-detail-out").innerText=r,get("token-detail-content").innerText=l,get("\
token-detail-thought").innerText=c,get("token-detail-encrypted").innerText=u;const f=t.model?`${t.model}\
 (${t.role})`:`${t.role}`;get("token-detail-title").innerText=f,showTokenDetailModal(e)}o(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){get("token-detail-modal")&&(hideModal("token-deta\
il-modal"),!e&&location.pathname==="/token-details"&&history.back())}o(closeTokenDetail,"closeTokenD\
etail");function openEncryptionSettings(e){const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}
o(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get("encrypti\
on-status-modal"))return;const n=get("encryption-status-title"),i=get("encryption-status-body"),a=get(
"encryption-status-admin-actions"),r=get("encryption-status-admin-toggle"),l=!!e;l?(n&&(n.innerText=
"\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\
\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
a&&r&&(!!(isAdminUser&&currentThreadId)?(a.classList.remove("hidden"),r.dataset.enable=l?"0":"1",r.disabled=
!1,r.textContent=l?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
r.className=l?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):a.classList.add("hidden")),showEncryptionStatusModal()}o(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}o(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const t=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${t?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const i=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,t,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=i}}o(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}o(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}o(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}o(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=o((e,t,n,i)=>{
const a=(e||"").toLowerCase();return!a.includes("gemini")||a.includes("image")||a.includes("nano")||
a.includes("tts")||a.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=o(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),a=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(r=>{let l=!1;function c(){n&&n.removeEventListener("cli\
ck",f),i&&i.removeEventListener("click",g),a&&a.removeEventListener("click",g),e.removeEventListener(
"click",y,!0)}o(c,"cleanup");function u(w){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(w)}o(u,"finalize");
function f(){u(!0)}o(f,"onOk");function g(){u(!1)}o(g,"onCancel");function y(w){w.target===e&&(w.preventDefault(),
w.stopImmediatePropagation(),g())}o(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",g),a&&a.addEventListener("click",g),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),u=`<div clas\
s="flex justify-start mb-4 ${r}"><div${l} class="message-bubble ai-pending-bubble bg-gray-700 text-w\
hite p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("chat-container");
if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",u);else{const g=document.
createElement("div");g.innerHTML=u;const y=g.firstElementChild;y&&f.appendChild(y)}n&&scrollToBottom()}}
o(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){if(!e||e.getAttribute(
"data-stream-transition")==="1")return;const t=e.querySelector(".content-area");t&&(t.classList.remove(
"pending-shimmer","skeleton-pending"),t.removeAttribute("data-skeleton-kind")),e.setAttribute("data-\
stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add("ai-stream-transitio\
n"),t&&(t.classList.add("ai-stream-content-transition"),setTimeout(()=>{t&&t.classList.remove("ai-st\
ream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-transition")},320)}
o(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){return e==
null||e===""?null:String(e)}o(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}o(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const t=Array.from(e.querySelectorAll(".prose")).some(c=>String(c.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(c=>!!String(c.textContent||
"").trim()&&c.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const a=e.parentElement;
if(!a)return null;const r=a.cloneNode(!0);r.setAttribute("data-local-stopped-partial","1"),r.classList.
remove("fade-in");const l=r.querySelector(".message-bubble");if(l&&(l.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),l.removeAttribute("data-stream-transition"),l.removeAttribute("id"),!r.
querySelector('[data-stopped-partial-note="1"]'))){const c=document.createElement("div");c.setAttribute(
"data-stopped-partial-note","1"),c.className="text-[10px] text-amber-200/90 mt-2 text-right",c.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",l.appendChild(c)}return{html:r.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}o(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const a=get("chat-\
container");return a?(a.querySelectorAll('[data-local-stopped-partial="1"]').forEach(r=>r.remove()),
a.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}o(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}o(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}o(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}o(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var c,u;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((u=t.retryDelayMs)!=
null?u:180)||0),a=!!t.notifyOnFailure,r=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!l||r&&l!==r)return!1;for(let f=0;f<=n;f++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==l?!1:(await loadMessages(l,{preserveDraft:!0,silent:!0}),
!0)}catch{f<n&&i>0&&await new Promise(y=>setTimeout(y,i))}return a&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===l&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}o(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(t){console.warn("Vi\
bration failed:",t)}}o(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const t=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(n=>n.kind==="minimal"&&!minimalPromptMode?!1:n.label.
toLowerCase().includes(t)||n.description.toLowerCase().includes(t))}o(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,t){if(String(e||"").toLowerCase()!=="thinking")
return e;const i=String(t||"").trimStart().match(/^\/thinking(\s+.*)$/i);return i?`thinking${i[1]}`.
toLowerCase():e}o(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const t=String(e||"").trim().toLowerCase();if(!t||t==="toggle"||t==="\u5207\u66FF"||t==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(t))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(t))return!1}o(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(r=>r.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(r=>r.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const a=String(n.presetArgument||t||"").trim();if(i.selectId){if(!a)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} ${n.id==="effort"?"none / low / medium / high / xhigh / max":"default\
 / none"}`,"info"),!1;const r=get(i.selectId),l=a.toLowerCase(),c=r?Array.from(r.options).find(u=>u.
value.toLowerCase()===l||u.textContent.trim().toLowerCase()===l):null;return!r||!c?(showToast(`${n.label}\
: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(r.value=c.value,r.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const r=a.toLowerCase(),
l={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},c=parseSlashToggleArgument(
a),u=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(l,r)){u&&!u.checked&&!u.disabled&&(u.
checked=!0,u.dispatchEvent(new Event("change",{bubbles:!0})));const f=get("thinking-level");return f&&
(f.value=l[r],f.dispatchEvent(new Event("change",{bubbles:!0}))),refreshMinimalOptionItems(),showToast(
`Thinking: ${r}`,"success"),!0}if(c===void 0)return showToast("\u4F7F\u3044\u65B9: /thinking on / off / min / low /\
 mid / high","info"),!1}if(i.checkboxId&&a){const r=parseSlashToggleArgument(a);if(r===void 0)return showToast(
`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const l=get(i.checkboxId);if(r!==null&&l&&l.checked===
r)return showToast(`${i.label}: ${r?"ON":"OFF"}`,"info"),!0}return handleMinimalOptionClick(i),!0}o(
executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){const t=String(
e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||"",i=n.
match(/^[a-z][\w-]*/i);return i?i[0]:n}o(extractSlashCommandToken,"extractSlashCommandToken");function hideSlashCommandSuggestions(){
const e=get("slash-command-suggestions");e&&e.classList.add("hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=
0}o(hideSlashCommandSuggestions,"hideSlashCommandSuggestions");function showPendingSlashCommandIndicator(e){
const t=get("slash-command-indicator"),n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.
find(r=>r.id===e);n.textContent=i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");
const a=get("prompt-input");a&&i&&(a.dataset.originalPlaceholder=a.placeholder,a.placeholder=i.argumentHint||
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}
o(showPendingSlashCommandIndicator,"showPendingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){
const e=get("slash-command-indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get(
"prompt-input");t&&t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.
dataset.originalPlaceholder);const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}
o(hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const a=visibleSlashCommands(e);if(a.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((w,v)=>{const k=document.createElement(
"div");k.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${v===
slashSelectedIndex?"bg-gray-700":""}`,k.innerHTML=`
                    <i class="fas ${w.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${w.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${w.description}</div>
                    </div>
                `;let _=!1;k.addEventListener("pointerdown",C=>{typeof C.button=="number"&&C.button!==
0||(C.preventDefault(),_=!0,selectSlashCommand(w.id))}),k.addEventListener("click",C=>{C.preventDefault(),
_||selectSlashCommand(w.id)}),k.onmouseenter=()=>{slashSelectedIndex=v,showSlashCommandSuggestions(e)},
n.appendChild(k)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,u=r.top,f=260,
g=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&u>c){const w=Math.min(f,u-g);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(f,c-g);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}o(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const l=String(n||"").trimStart();t.value=
l.substring(1+i.length).trimStart()}else{const l=n.lastIndexOf("/");l!==-1?t.value=n.substring(0,l).
trimEnd():t.value=""}hideSlashCommandSuggestions();const a=SLASH_COMMANDS.find(l=>l.id===e),r=t.value.
trim();if(a&&a.autocompleteArgument&&!r){t.value=`${a.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(a&&a.kind==="minimal"&&(!a.
requiresArgument||r)){t.value="",executeMinimalSlashCommand(e,r),t.dispatchEvent(new Event("input",{
bubbles:!0})),t.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),t.focus(),t.
dispatchEvent(new Event("input",{bubbles:!0}))}o(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
default_model:{label:"\u65E2\u5B9A\u306E\u30E2\u30C7\u30EB",tab:"general",control:"set-default-model"},
default_vision_model:{label:"Vision Model",tab:"general",control:"set-default-vision-model"},use_last_chat_settings:{
label:"\u524D\u56DE\u306E\u8A2D\u5B9A\u3092\u7D99\u7D9A",tab:"general",control:"set-use-last-setting\
s"},default_enable_search:{label:"\u65E2\u5B9A\u306ESearch",tab:"general",control:"set-default-searc\
h"},default_enable_url_context:{label:"\u65E2\u5B9A\u306EURLs",tab:"general",control:"set-default-ur\
l-context"},default_enable_maps:{label:"\u65E2\u5B9A\u306EMaps",tab:"general",control:"set-default-m\
aps"},default_enable_python:{label:"\u65E2\u5B9A\u306EPython",tab:"general",control:"set-default-pyt\
hon"},default_enable_file_creation:{label:"\u65E2\u5B9A\u306EFile",tab:"general",control:"set-defaul\
t-file-creation"},default_enable_thinking:{label:"\u65E2\u5B9A\u306EThinking",tab:"general",control:"\
set-default-thinking"},default_thinking_level:{label:"Thinking Level",tab:"general",control:"set-def\
ault-thinking-level"},default_thinking_budget:{label:"Thinking Budget",tab:"general",control:"set-de\
fault-thinking-budget"},default_reasoning_effort:{label:"Reasoning Effort",tab:"general",control:"se\
t-default-reasoning-effort"},default_enable_system_prompt:{label:"\u65E2\u5B9A\u306ESysPrompt",tab:"\
general",control:"set-default-sys-prompt"},default_enable_mcp:{label:"\u65E2\u5B9A\u306EMCP",tab:"ge\
neral",control:"set-default-mcp"},default_safety_setting:{label:"\u65E2\u5B9A\u306ESafety",tab:"gene\
ral",control:"set-default-safety"},auto_search_on_links:{label:"X\u30EA\u30F3\u30AF\u306E\u81EA\u52D5\u691C\u7D22",
tab:"general",control:"set-auto-search-links"},mic_transcribe_mode:{label:"\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u65B9\u5F0F",
tab:"general",control:"set-mic-transcribe-mode"},stt_model:{label:"STT\u30E2\u30C7\u30EB",tab:"gener\
al",control:"set-stt-model"},llm_transcribe_prompt:{label:"LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"general",control:"set-llm-transcribe-prompt"},enter_to_send:{label:"Enter\u3067\u9001\u4FE1",tab:"\
general",control:"set-enter-to-send"},compact_prompt_mode:{label:"\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u8868\u793A",
tab:"general",control:"set-compact-prompt-mode"},minimal_prompt_mode:{label:"\u30DF\u30CB\u30DE\u30EB\u8868\u793A",
tab:"general",control:"set-minimal-prompt-mode"},temp_chat_timeout_seconds:{label:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u4FDD\u6301\u6642\u9593",
tab:"general",control:"set-temp-chat-timeout-seconds"},system_prompt:{label:"\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"prompt",control:"sys-prompt-text"},system_prompt_enabled:{label:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"prompt",control:"set-global-sys-prompt-enabled"},apply_global_system_prompt:{label:"\u30E6\u30FC\u30B6\u30FC\u30D7\u30ED\u30F3\u30D7\u30C8\u306E\u9069\
\u7528",tab:"prompt",control:"set-apply-global-sys-prompt"},apply_auto_system_prompt_notices:{label:"\
\u81EA\u52D5\u6CE8\u5165\u30D7\u30ED\u30F3\u30D7\u30C8",tab:"prompt",control:"set-apply-auto-sys-pro\
mpt-notices"},auto_system_prompt_notices_config:{label:"\u81EA\u52D5\u6CE8\u5165\u30D7\u30ED\u30F3\u30D7\u30C8\u8A2D\u5B9A",
tab:"prompt",control:"auto-sys-prompt-settings"},theme_color:{label:"\u30C6\u30FC\u30DE\u30AB\u30E9\u30FC",
tab:"display",control:"set-theme-color"},liquid_glass_enabled:{label:"Liquid Glass",tab:"display",control:"\
set-liquid-glass"},use_sw_cache:{label:"\u9AD8\u901F\u30AD\u30E3\u30C3\u30B7\u30E5",tab:"data",control:"\
set-use-sw-cache"},enable_latency_metrics:{label:"\u30EC\u30B9\u30DD\u30F3\u30B9\u901F\u5EA6\u306E\u8A08\u6E2C",
tab:"data",control:"set-latency-metrics"},enable_client_debug_log:{label:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1",
tab:"data",control:"set-client-debug-log"},bot_detection_enabled:{label:"Bot Detection",tab:"securit\
y",control:"set-bot-detect"},skip_2fa_on_google_login:{label:"Google\u30ED\u30B0\u30A4\u30F3\u6642\u306E2FA",
tab:"2fa",control:"set-skip-2fa-google"},default_2fa_method:{label:"\u65E2\u5B9A\u306E2FA\u65B9\u5F0F",
tab:"2fa",control:"set-default-2fa-method"},rich_paste_prompt_default:{label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30D7\u30ED\u30F3\u30D7\u30C8",
modal:"rich-paste",control:"rich-paste-prompt"},rich_paste_prompt_use_custom_default:{label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\
\u306E\u65E2\u5B9A\u5024",modal:"rich-paste",control:"rich-paste-use-default"}};function formatAiSettingValue(e){
if(e===!0)return"ON";if(e===!1)return"OFF";if(e==="(\u66F4\u65B0)")return"\u66F4\u65B0\u6E08\u307F";
if(e==null||e==="")return"\u672A\u8A2D\u5B9A";if(typeof e=="object")try{return JSON.stringify(e)}catch{
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}o(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,t){
const n=get(`tab-${e}`);let i=get(t);if(!n||!i)return null;for(;i.parentElement&&i.parentElement!==n;)
i=i.parentElement;return i.parentElement===n?i:get(t)}o(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const t=AI_SETTING_JUMP_TARGETS[e];if(!t){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(t.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const n=get(t.control);n&&(n.scrollIntoView({behavior:"smooth",block:"center"}),n.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const n=findSettingsJumpElement(t.tab,t.control);n?jumpToSetting(t.tab,n):switchTab(t.tab||"general")},
320)}o(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const t=e.querySelector(".msg-controls");t&&t.remove()}o(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,t,n="update"){const i=Object.
entries(e||{}),a=`settings-result-${Date.now()}`,r=n==="inspect",l=i.length?r?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
r?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
c=renderMessage(a,"assistant",l,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!c)
return;removeEphemeralMessageControls(c);const u=c.querySelector(".message-bubble");if(!u||!i.length)
return;const f=document.createElement("div");f.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,w])=>{const v=AI_SETTING_JUMP_TARGETS[y]||{label:y},k=document.createElement("button");k.
type="button",k.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const _=document.createElement("span");_.className="min-w-0 flex-1";const C=document.createElement(
"span");C.className="block text-xs font-bold text-blue-200",C.textContent=v.label;const L=document.createElement(
"span");L.className="block mt-0.5 text-[11px] text-gray-300 break-words",L.textContent=formatAiSettingValue(
w);const E=document.createElement("i");E.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",_.appendChild(C),_.appendChild(L),k.appendChild(_),k.appendChild(E),k.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(k)});const g=u.querySelector(".message-footer-\
meta");g?u.insertBefore(f,g):u.appendChild(f),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const r=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${r}" class="flex justify-start mb-4 fade-in"><div cl\
ass="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-\
md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const u=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),f=get(r);if(f&&f.remove(),u&&u.status==="ok"&&u.mode==="inspect"&&u.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(u.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
u.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(u.current,t,"inspect");
return}if(u&&u.status==="ok"&&u.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
u.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(u.applied).length}\u9805\u76EE\uFF09`,"success");try{const w=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(v=>v.json());populateAiSafeFormFields(w),cacheUserSettings(w)}catch{}renderAiSettingsResultBubble(
u.applied,t);return}const g=u.message||u.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${g}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${g}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(g,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const u=get(r);u&&u.remove();const f=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(f),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
o(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}o(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const a=e.toLowerCase(),r=loadedGems.filter(v=>v.name.toLowerCase().includes(a)||v.description&&
v.description.toLowerCase().includes(a));if(r.length===0){hideGemSuggestions();return}gemSelectedIndex>=
r.length&&(gemSelectedIndex=0),n.innerHTML="",r.forEach((v,k)=>{const _=document.createElement("div");
_.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${k===gemSelectedIndex?
"bg-gray-700":""}`,_.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(v.name)}</div>
                        ${v.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
v.description)}</div>`:""}
                    </div>
                `,_.onclick=()=>selectGemSuggestion(v),_.onmouseenter=()=>{gemSelectedIndex=k,showGemSuggestions(
e)},n.appendChild(_)});const l=i.getBoundingClientRect(),c=window.innerHeight,u=c-l.bottom,f=l.top,g=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",u<180&&f>u){const v=Math.min(g,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(g,u-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${v}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}o(showGemSuggestions,
"showGemSuggestions");function selectGemSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideGemSuggestions(),
activateGem(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}o(selectGemSuggestion,"sel\
ectGemSuggestion");function browserFastModeIneligibility(e){const t=String(get("model-select")?get("\
model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!t.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(t))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeGem)return"Gems\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(currentQuote||editingMessageId)return"\u5F15\u7528\u30FB\u7DE8\u96C6\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(codingModeEnabled)return"Coding Mode\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(["enable-search","enable-url-context","enable-maps","enable-sys-prompt","enable-prompt-cache","en\
able-mcp"].some(l=>{const c=get(l);return!!(c&&c.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const i=get("thread-custom-instruction");if(i&&String(i.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const a=Array.from(browserFastLocalFiles.values());return a.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":a.reduce((l,c)=>l+Number(c.file&&c.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":a.some(l=>!l.
file||!String(l.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}o(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const a=String(i.result||""),r=a.indexOf(",");if(r<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
a.slice(r+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}o(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const a=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(r=>{r&&
a.push({thoughtSignature:String(r)})}),i.text&&a.push({text:String(i.text)});for(const r of Array.isArray(
i.images)?i.images:[])try{const l=await fetch(buildFileUrl(r.path),{credentials:"same-origin",cache:"\
no-store"});if(!l.ok)throw new Error(`HTTP ${l.status}`);const c=await l.blob();a.push({inlineData:{
mimeType:r.mime_type||c.type||"application/octet-stream",data:await fileToBase64Payload(c)}})}catch{
n++}a.length&&t.push({role:i.role,parts:a})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}o(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(){
const e=Array.from(browserFastLocalFiles.entries());for(const[t,n]of e){if(!n||!n.file||!n.rowObj)throw new Error(
"\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");
if(n.rowObj.status&&(n.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(n.file,n.rowObj))throw new Error(`${n.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);browserFastLocalFiles.delete(t)}}o(uploadBrowserFastLocalFiles,
"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){const t=get("enable-thinking");
if(!t||!t.checked)return null;const n=String(get("thinking-level")?get("thinking-level").value:"high").
toLowerCase();if(e.includes("2.5")){const a=Number(get("thinking-budget")?get("thinking-budget").value:
4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(a)?Math.max(0,Math.min(32768,Math.trunc(
a))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["MEDIUM","HIGH"].includes(i)&&(i="MEDIU\
M"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}
o(browserFastThinkingConfig,"browserFastThinkingConfig");function browserFastPythonBoxHtml(e){return`\
<div class="code-wrapper python-box collapsed" data-py-id="${e}" data-collapsed="true" data-code-key\
="${e}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Exec\
ution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" a\
ria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code"\
 data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class\
="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-alig\
n-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="pyth\
on-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div clas\
s="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext p\
ython-output"></code></pre></div></div></div>`}o(browserFastPythonBoxHtml,"browserFastPythonBoxHtml");
function updateBrowserFastPythonBox(e,t,n){if(e){if(t==="code"){const i=n==null?"":String(n),a=e.querySelector(
".python-code");a&&(a.textContent=i,a.removeAttribute("data-highlighted"),queueHighlight(e,i));const r=e.
querySelector('.copy-btn[data-copy="code"]');r&&r.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}else if(t==="output"){const i=n==null?"":String(n),a=e.querySelector(".python-output");
a&&(a.textContent=i);const r=e.querySelector('.copy-btn[data-copy="output"]');r&&r.setAttribute("dat\
a-code",encodeURIComponent(i).replace(/'/g,"%27"))}}}o(updateBrowserFastPythonBox,"updateBrowserFast\
PythonBox");async function sendBrowserFastMessage(e){const t=String(get("model-select").value||"").trim(),
n=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==t)throw new Error(
"\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const i=Array.from(browserFastLocalFiles.values()),a=[];for(const O of i)a.push({inlineData:{mimeType:O.
file.type,data:await fileToBase64Payload(O.file)}});a.push({text:e});const r={},l=browserFastThinkingConfig(
t.toLowerCase());l&&(r.thinkingConfig=l);const c={contents:[...await buildBrowserFastHistoryContents(
n.history),{role:"user",parts:a}],generationConfig:r};!!(get("enable-python")&&get("enable-python").
checked)&&(c.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden"),renderMessage(Date.now(),"user",
e,null,null,null,null,!0,null,null,null,null,null,null,null,null,!0);const f=`browser-fast-${Date.now()}`;
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${f}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const g=get(f);activeStreamingBubbleId=f,setSendBtnToStopMode(),resumeChatAutoScroll(),
abortController=new AbortController;let y="",w="";const v=[];let k=null,_=null,C=!1;const L={},E=[];
let B=null,K="";const Z=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Ie=!1;try{const O=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
t)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(c),signal:abortController.
signal}));if(!O.ok){const re=await O.json().catch(()=>({}));throw new Error(re&&re.error&&re.error.message?
re.error.message:`Gemini API HTTP ${O.status}`)}window.ConnectionMonitor&&(Ie=!0,window.ConnectionMonitor.
operationStarted()),Z&&Z.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const G=O.body.getReader(),te=new TextDecoder;let be="";const ue=o(re=>{const J=re.split(
/\r?\n/).filter(U=>U.startsWith("data:")).map(U=>U.slice(5).trim()).join("");if(!J||J==="[DONE]")return;
const A=JSON.parse(J);if(A.error)throw new Error(A.error.message||"Gemini API error");if((Array.isArray(
A.candidates)?A.candidates:[]).forEach(U=>{(U&&U.content&&Array.isArray(U.content.parts)?U.content.parts:
[]).forEach(ie=>{if(ie&&typeof ie.thoughtSignature=="string"&&!v.includes(ie.thoughtSignature)&&v.push(
ie.thoughtSignature),ie&&ie.executableCode&&typeof ie.executableCode.code=="string"){const ne=ie.executableCode.
code;y+=`
\`\`\`python
${ne}
\`\`\`
`,B=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,K=ne,L[B]||(g.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(B)),L[B]=g.querySelector(`[data-py-id="${B}"]`)),updateBrowserFastPythonBox(
L[B],"code",ne);return}if(ie&&ie.codeExecutionResult&&typeof ie.codeExecutionResult.output=="string"){
const ne=ie.codeExecutionResult.output;y+=`
**Output:**
\`\`\`
${ne}
\`\`\`
`;const pe=B||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;E.push({code:K||
"",output:ne}),L[pe]||(g.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(pe)),L[pe]=g.querySelector(
`[data-py-id="${pe}"]`)),updateBrowserFastPythonBox(L[pe],"output",ne);return}const Te=typeof ie.text==
"string"?ie.text:"";Te&&(ie.thought===!0?w+=Te:y+=Te)})}),!C&&(y||w)){beginPendingToStreamTransition(
g);const U=g.querySelector(".content-area");U&&U.remove(),C=!0}w&&(_||(g.insertAdjacentHTML("afterbe\
gin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></\
div>'),_=g.querySelector(".thought-content")),_.textContent=w),y&&(k||(k=document.createElement("div"),
k.className="content-area prose prose-invert text-sm break-words",g.appendChild(k)),renderAiMarkdownInto(
k,y,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:re,value:J}=await G.
read();if(re)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),Z&&Z.setPhase(
"receiving"),be+=te.decode(J,{stream:!0});const A=be.split(/\r?\n\r?\n/);be=A.pop()||"",A.forEach(ue)}
if(be+=te.decode(),be.trim()&&ue(be),!y.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
k&&renderAiMarkdownInto(k,y,{incrementalMath:!0}),_&&_.classList.add("collapsed"),E.length&&(y+=E.map(
re=>`
\`\`\`pyexec
${JSON.stringify(re)}
\`\`\`
`).join("")),i.length&&(Z&&Z.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),Z&&Z.setPhase("saving");const Le=collectImageUrlsForSend(),
ye=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:y,thought_content:w,model:t,image_urls:Le,temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:n.parent_id||null,thought_signatures:v,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}),g),Me=await ye.json().catch(()=>({}));if(!ye.ok||!Me.thread_id)throw new Error(
Me.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");const oe=!currentThreadId;currentThreadId=
String(Me.thread_id),currentParentId=Me.assistant_message_id||null,currentLeafId=Me.assistant_message_id||
null,resetUploadState(),browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,skipHistory:!oe}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\
\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F","success",!1)}catch(O){if(O.name!=="AbortError"){showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9: ${O.message}`,"error",!0),get("prompt-input").value||(get("prompt-i\
nput").value=e);const G=O.message||"\u30A8\u30E9\u30FC";g&&g.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
G));try{let te=y||"";E.length&&(te+=E.map(Me=>`
\`\`\`pyexec
${JSON.stringify(Me)}
\`\`\`
`).join(""));const be=buildChatErrorMarkdown(G,te),ue=i.length?[]:collectImageUrlsForSend(),Le=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:be,
thought_content:w||"",model:t,image_urls:ue,temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||
null,parent_id:n&&n.parent_id?n.parent_id:null,thought_signatures:v,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),g),ye=await Le.
json().catch(()=>({}));if(Le.ok&&ye.thread_id){const Me=!currentThreadId;currentThreadId=String(ye.thread_id),
currentParentId=ye.assistant_message_id||null,currentLeafId=ye.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!Me}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(te){sendClientDebugLog("error",`Browser fa\
st error persist failed: ${te&&te.message?te.message:te}`)}}}finally{Ie&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),Z&&Z(),setSendBtnToSendMode(),activeStreamingBubbleId===f&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Jt;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const R=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(R);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const t=get("prompt-inp\
ut").value;if(pendingSlashCommand){const R=pendingSlashCommand,de=t.trim(),Ee=get("model-select")?get(
"model-select").value:null;if(R==="settings"){if(!de){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!Ee){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
de,Ee)}else executeMinimalSlashCommand(R,de)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const n=t.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(n&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(R=>R.id===
n[1].toLowerCase())){executeMinimalSlashCommand(n[1].toLowerCase(),n[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const i=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(i&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(i)setBrowserFastModeEnabled(!1);else{const R=browserFastModeIneligibility(
t);if(!R){try{await sendBrowserFastMessage(t)}catch(de){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${de.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${R}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(de){showToast(
de.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}t.trim()&&(promptHistory.length===
0||promptHistory[0]!==t)&&(promptHistory.unshift(t),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const a=collectAttachmentItemsForSend(),r=a.map(R=>R.path),l=a.filter(R=>normalizeAttachmentSource(
R.source)==="upload").map(R=>R.path);if(r.length>ATTACHMENT_MAX_FILES){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u6DFB\u4ED8\u3092\u6E1B\u3089\u3057\u3066\u518D\u9001\u3057\u3066\u304F\u3060\u3055\u3044\u3002`,
"error",!0);return}const c=getModelMediaSupport(get("model-select").value),u=r.some(R=>isAudioPath(R)),
f=r.some(R=>isVideoPath(R)),g=(get("model-select").value||"").toLowerCase(),y=get("enable-python"),w=!!(y&&
y.checked);if(u&&!c.audio||f&&!c.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!t.trim()&&r.length===0)return;if(isMistralOcrModel(
g)){const R=/https?:\/\/\S+/i.test(t);if(r.filter(Ee=>isAudioPath(Ee)||isVideoPath(Ee)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!r.length&&!R){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const v=t.trim();if(/^\/settings(?:\s|$)/i.
test(v)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(v)){const R=v.replace(/^\/settings\s*/i,"").trim();
if(!R){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ee=get("prompt-input");Ee.value="/settings ";const Ne=extractSlashCommandToken(Ee.value);
lastSlashFilter=Ne,showSlashCommandSuggestions(Ne),Ee.focus();return}const de=get("model-select")?get(
"model-select").value:null;if(!de){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
R,de);return}if(isGeminiLocalPythonMode(g,u,f,w)&&!await confirmGeminiLocalPythonSwitch())return;let k=null,
_=[];if(codingModeEnabled){const R=collectCodingCandidates(t),de=R.filter(Ve=>Ve.prompt_source),Ee=R.
filter(Ve=>!Ve.prompt_source),Ne=de.reduce((Ve,De)=>Ve+String(De.code||"").length,0);if(Ne>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let ze=3e5-Ne;const Je=[];for(let Ve=Ee.length-1;Ve>=0;Ve--){const De=String(Ee[Ve].
code||"").length;De>ze||(Je.unshift(Ee[Ve]),ze-=De)}_=codingTargetSelection?Je.slice(-1):[...de,...Je];
const rt=de.length?de[de.length-1]:null;if(k=codingTargetSelection?_[0]:rt||_[_.length-1]||null,codingModeEffective=
!!(k&&String(k.code||"").trim()),codingModeEffective&&k.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const Ve=String(((Jt=get(
"model-select"))==null?void 0:Jt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(Ve)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const C=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${t.length}\
 attachments=${r.length} search=${get("enable-search").checked}`);const L=t,E=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&r.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const B=editingMessageId,K=currentParentId,Z=B!=null;B&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Ie=[],O=o(R=>{if(R==null)
return;let de=document.getElementById(`msg-${R}`);for(;de;)de.classList&&de.classList.contains("mess\
age-group")&&(Ie.push({node:de,prevDisplay:de.style.display}),de.style.display="none"),de=de.nextElementSibling},
"hideRenderedBranchFrom"),G=o(()=>{Ie.forEach(({node:R,prevDisplay:de})=>{R&&(R.style.display=de||"")}),
Ie.length=0},"restoreHiddenBranch");B&&O(B);const te=Date.now(),be=renderMessage(te,"user",L,JSON.stringify(
r),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,K,activeGem?activeGem.name:null);
let ue=!1;const Le=/(https?:\/\/)?(x\.com|twitter\.com)\//i,ye=Le.test(L||"")||Le.test(currentQuote||
""),Me="grok-4-fast-reasoning",oe=o(()=>{get("enable-search").checked=!0,get("model-select").value!==
Me&&selectModelById(Me)},"applyXLinkAuto");if(ye&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)oe();else{const R=get("auto-search-banner"),de=get("auto-search-on-btn"),Ee=get(
"auto-search-off-btn"),Ne=get("auto-search-remember");R&&de&&Ee&&(Ne&&(Ne.checked=!1),await new Promise(
ze=>{R.classList.remove("hidden");const Je=o(rt=>{R.classList.add("hidden"),de.onclick=null,Ee.onclick=
null,ze(rt)},"cleanup");de.onclick=()=>Je("enable"),Ee.onclick=()=>Je("disable")}).then(async ze=>{ze===
"enable"?(oe(),Ne&&Ne.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
ue=!0}))}const re=String(get("reasoning-effort").value||"").toLowerCase(),J=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&re==="none",A={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:L,model:get("model-select").value,image_urls:r,image_items:a,uploaded_image_urls:l,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:J?!1:get("enable-thin\
king").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:E,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&get("tts-voice-custom")?
get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-language")?get("tts-language").
value:null,tts_speed:isTtsModel()&&get("tts-speed")?get("tts-speed").value:null,image_size:isGptImageModel()&&
get("gpt-image-size")?get("gpt-image-size").value:null,image_quality:isGptImageModel()&&get("gpt-ima\
ge-quality")?get("gpt-image-quality").value:null,image_format:isGptImageModel()&&get("gpt-image-form\
at")?get("gpt-image-format").value:null,image_compression:isGptImageModel()&&get("gpt-image-compress\
ion")?get("gpt-image-compression").value:null,image_mask:isGptImageModel()?currentMaskImage:null,gemini_image_aspect:isGeminiImageModel()&&
get("gemini-image-aspect")?get("gemini-image-aspect").value:null,gemini_image_size:isGeminiImageModel()&&
get("gemini-image-size")?get("gemini-image-size").value:null,grok_image_aspect:isGrokImageModel()&&get(
"grok-image-aspect")?get("grok-image-aspect").value:null,grok_image_resolution:isGrokImageModel()&&get(
"grok-image-resolution")?get("grok-image-resolution").value:null,grok_image_quality:isGrokImageModel()&&
get("grok-image-quality")?get("grok-image-quality").value:null,grok_image_format:isGrokImageModel()&&
get("grok-image-format")?get("grok-image-format").value:null,grok_image_count:isGrokImageModel()&&get(
"grok-image-count")?get("grok-image-count").value:null,xai_temperature:get("xai-temperature")?get("x\
ai-temperature").value:null,xai_top_p:get("xai-top-p")?get("xai-top-p").value:null,xai_max_completion_tokens:get(
"xai-max-completion-tokens")?get("xai-max-completion-tokens").value:null,xai_seed:get("xai-seed")?get(
"xai-seed").value:null,xai_presence_penalty:get("xai-presence-penalty")?get("xai-presence-penalty").
value:null,xai_frequency_penalty:get("xai-frequency-penalty")?get("xai-frequency-penalty").value:null,
xai_stop:get("xai-stop")?get("xai-stop").value:null,xai_response_format:get("xai-response-format")?get(
"xai-response-format").value:null,xai_tool_choice:get("xai-tool-choice")?get("xai-tool-choice").value:
null,xai_parallel_tool_calls:get("xai-parallel-tool-calls")?get("xai-parallel-tool-calls").checked:!0,
xai_logprobs:get("xai-logprobs")?get("xai-logprobs").checked:!1,xai_top_logprobs:get("xai-top-logpro\
bs")?get("xai-top-logprobs").value:null,grok_video_duration:isGrokVideoModel()&&get("grok-video-dura\
tion")?get("grok-video-duration").value:null,grok_video_aspect:isGrokVideoModel()&&get("grok-video-a\
spect")?get("grok-video-aspect").value:null,grok_video_resolution:isGrokVideoModel()&&get("grok-vide\
o-resolution")?get("grok-video-resolution").value:null,gemini_video_duration:isGeminiVideoModel()&&get(
"gemini-video-duration")?get("gemini-video-duration").value:null,gemini_video_aspect:isGeminiVideoModel()&&
get("gemini-video-aspect")?get("gemini-video-aspect").value:null,gemini_video_resolution:isGeminiVideoModel()&&
get("gemini-video-resolution")?get("gemini-video-resolution").value:null,music_instrumental:isGeminiMusicModel()&&
get("music-instrumental")?get("music-instrumental").checked:!1,ocr_table_format:isMistralOcrModel()&&
get("ocr-table-format")?get("ocr-table-format").value:null,ocr_extract_header:isMistralOcrModel()&&get(
"ocr-extract-header")?get("ocr-extract-header").checked:!1,ocr_extract_footer:isMistralOcrModel()&&get(
"ocr-extract-footer")?get("ocr-extract-footer").checked:!1,ocr_include_blocks:isMistralOcrModel()&&get(
"ocr-include-blocks")?get("ocr-include-blocks").checked:!1,ocr_include_image_base64:isMistralOcrModel()&&
get("ocr-include-images")?get("ocr-include-images").checked:!0,ocr_pages:isMistralOcrModel()&&get("o\
cr-pages")?get("ocr-pages").value:null,transcription_language_codes:[],transcription_custom_vocabulary:[],
transcription_mode:"verbatim",transcription_diarization:!1,transcription_word_timestamps:!1,quote_text:currentQuote,
parent_id:K,parent_id_explicit:Z,disable_auto_search:ue,image_vision_model:currentVisionModel||null,
coding_mode:C,coding_target:C?{id:k.candidate_id,code:k.prompt_source?null:k.code,language:k.language||
"text",key:k.key||null,message_id:k.message_id||null,source:k.prompt_source?"prompt":"history",explicit:k.
explicit===!0}:null,coding_candidates:C?_.map(R=>({id:R.candidate_id,source:R.prompt_source?"prompt":
"history",prompt_index:R.prompt_source?R.prompt_index:null,code:R.prompt_source?null:R.code,language:R.
language||"text",explicit:R.explicit===!0})):[],batch_mode:i};e&&(A.turnstile_token=e);const F=get("\
thread-custom-instruction");F&&(A.thread_custom_instruction=F.value||""),activeGem?(A.system_prompt=
activeGem.instruction,A.enable_system_prompt=!0,A.gem_uuid=activeGem.uuid):A.gem_uuid=null,setSendBtnToStopMode();
const U="ai-"+Date.now(),X=String(A.model||"").toLowerCase(),ie=!!A.enable_thinking||!!re&&re!=="non\
e",Te=X.includes("gemini")||X.includes("o1")||X.includes("o3")||X.includes("gpt-5")||X.includes("rea\
soning")&&!X.includes("non-reasoning"),ne=ie&&Te;let pe=buildPendingSkeletonHtml(A.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${U}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${pe}</div></div>`),resumeChatAutoScroll();const V=get(U);activeStreamingBubbleId=
U,canvasModeEnabled&&resetCanvasPreviewPanel();let we=null;const ot=o(R=>!ne||!V?null:((!we||!V.contains(
we))&&(we=V.querySelector(".thought-content")),we||(V.insertAdjacentHTML("afterbegin",'<div class="t\
hought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i cla\
ss="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" \
data-placeholder="1"></div></div>'),we=V.querySelector(".thought-content")),we&&(we.setAttribute("da\
ta-placeholder","1"),we.textContent=R||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
we),"ensureThoughtPlaceholder");ne&&ot("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const Pe=currentThreadId,dt=nowPerfMs(),bt=Date.now();let Ze=!1,
mt=!1,Tt=!1,xt=null,Ct=null,ft=null,Nt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Rt=o((R,de)=>{if(!de||R==="status"&&Ze||R==="thought"&&mt||R==="content"&&Tt)return;const Ee=Math.
max(0,nowPerfMs()-dt);R==="status"?xt=Ee:R==="thought"?Ct=Ee:R==="content"&&(ft=Ee),reportFirstTokenLatency(
{latency_seconds:Ee/1e3,latency_ms:Ee,thread_id:Nt||currentThreadId,job_id:currentJobId,model:A.model,
first_event_type:R,client_sent_at_ms:bt}),R==="status"?Ze=!0:R==="thought"?mt=!0:R==="content"&&(Tt=
!0)},"maybeReportFirstEventLatency"),ut=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let yt=!1,Lt=!1,Mt=null,vt=null,Wt=!1;try{A.thread_id&&activeGem&&(threadGemMap[A.thread_id]=
activeGem,pendingGemForNewThread=null);const R=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(A),signal:abortController.signal}),V);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${R.status}`),!R.ok){const Oe=await R.json().catch(()=>({})),qe=new Error(Oe.error||
`HTTP ${R.status}`);throw qe.serverCode=Oe.code||null,qe.serverModel=Oe.model||A.model,qe.acceptedJobId=
Oe.job_id||null,qe.acceptedThreadId=Oe.thread_id||null,qe}yt=!0,window.ConnectionMonitor&&(Wt=!0,window.
ConnectionMonitor.operationStarted()),ut&&ut.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const de=o(()=>{if(!V)return;const Oe=V.querySelector(
".content-area");if(Oe&&Oe.getAttribute("data-api-accepted")!=="1"&&(Oe.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(V,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
Oe.outerHTML=buildPendingSkeletonHtml(A.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const qe=V.querySelector(".content-area");qe&&qe.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
V,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");de();
const Ee=R.body.getReader(),Ne=new TextDecoder;let ze="",Je="",rt="",Ve=!0,De=null,He=null,Ke=null,Kt=!1;
const Bt={};let at=0,Ft=!1;for(;!Ft;){const{done:Oe,value:qe}=await Ee.read();if(Oe)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ut&&ut.setPhase("receiving"),ze+=Ne.decode(qe,{stream:!0});
let tt=ze.split(`
`);ze=tt.pop();let Xt=!1,cn=!1;for(let ht of tt)if(ht.trim())try{const me=JSON.parse(ht);if(me.type===
"thread_id"){de();const Se=me.content!==null&&me.content!==void 0?String(me.content):me.content;Se&&
(Nt=Se,currentThreadId!==Se&&(currentThreadId=Se,history.pushState({},"","/c/"+Se)),activeGem&&(threadGemMap[Se]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(me.type==="job_\
id"){de(),currentJobId=me.content,i&&showToast("Batch\u767B\u9332","info");continue}if(me.type==="se\
arch_status"){me.content==="searching"&&!Ke?(V.insertAdjacentHTML("afterbegin",'<div class="search-b\
ox visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>'),Ke=V.querySelector(
".search-box")):me.content==="done"&&Ke&&(Ke.classList.remove("animate-pulse"),Ke.innerHTML='<i clas\
s="fas fa-check-circle text-green-400"></i> Search complete',setTimeout(()=>{Ke&&Ke.remove(),Ke=null},
2e3));continue}if(me.type==="mcp"){handleMcpStreamEvent(V,me.content||{});continue}if(me.type==="mcp\
_decision_request"){openMcpDecisionModal(me.content||{});continue}if(me.type==="status"){de();const Se=me.
content===null||me.content===void 0?"":String(me.content);if(Rt("status",!!Se),Ve&&V){const Xe=Se||"\
\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(V,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Ye=V.querySelector(".content-area");Ye&&(Ye.outerHTML=buildPendingSkeletonHtml(A.model,Xe),updatePendingSkeletonStatus(
V,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
ne&&ot(Se||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Ve){beginPendingToStreamTransition(
V);const Se=V.querySelector(".content-area");Se&&(Se.innerHTML=""),Ve=!1}if(me.type==="coding_diff")
appendCodingLiveDiff(V,me.content||{}),Rt("content",!0);else if(me.type==="thought"){if(De||(De=V.querySelector(
".thought-content")),rt+=me.content,Rt("thought",!!me.content),!De){const Se='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';Ke?Ke.insertAdjacentHTML(
"afterend",Se):V.insertAdjacentHTML("afterbegin",Se),De=V.querySelector(".thought-content")}if(De&&De.
getAttribute("data-placeholder")==="1"){if(De.textContent="",De.removeAttribute("data-placeholder"),
De){const Se=De.parentElement.querySelector(".thought-header");Se&&Se.classList.remove("thinking-shi\
mmer")}rt=me.content}De.classList.remove("collapsed"),cn=!0}else if(me.type==="image_analysis"){const Se=me.
content===null||me.content===void 0?"":String(me.content);if(!V)continue;let Xe=V.querySelector(".im\
age-analysis-box");if(!Xe){const nt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';Ke?Ke.insertAdjacentHTML("afterend",nt):V.insertAdjacentHTML("afterbegin",nt),Xe=V.querySelector(
".image-analysis-box")}const Ye=Xe.querySelector(".image-analysis-text");Ye&&(Ye.textContent=Se)}else if(me.
type==="python"){const Se=me.content||{},Xe=Se.id||`py_${Date.now()}`;if(!Bt[Xe]){const nt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Xe}" data-collapsed="true" data-code-key="${Xe}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;Ke?Ke.insertAdjacentHTML("afterend",nt):V.insertAdjacentHTML(
"afterbegin",nt),Bt[Xe]=V.querySelector(`[data-py-id="${Xe}"]`)}const Ye=Bt[Xe];if(Ye){if(Se.code!==
void 0){const nt=Se.code==null?"":String(Se.code),gt=Ye.querySelector(".python-code");gt&&(gt.textContent=
nt,gt.removeAttribute("data-highlighted"),queueHighlight(Ye,nt));const At=Ye.querySelector('.copy-bt\
n[data-copy="code"]');At&&At.setAttribute("data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}if(Se.
output!==void 0){const nt=Se.output==null?"":String(Se.output),gt=Ye.querySelector(".python-output");
gt&&(gt.textContent=nt);const At=Ye.querySelector('.copy-btn[data-copy="output"]');At&&At.setAttribute(
"data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}}}else if(me.type==="content"){const Se=me.content===
null||me.content===void 0?"":String(me.content);Je+=Se,/[`~]/.test(Se)&&activateDeferredCodingModeFromStream(
Je),He||(He=V.querySelector(".content-area")||document.createElement("div"),He.className="prose pros\
e-invert text-sm break-words",V.contains(He)||V.appendChild(He)),Xt=!0,Rt("content",!!Se)}else if(me.
type==="error"){Kt=!0,Ft=!0,V.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(me.content)),showToast(
me.content||"Unknown error","error",!0);break}}catch{}if(cn&&De&&(De.textContent=rt,userAutoScroll&&
(De.scrollTop=De.scrollHeight)),Xt&&He){const ht=Date.now();if(ht-at>100){const me=snapshotCodeCollapse(
He);renderAiMarkdownInto(He,Je,{incrementalMath:!0}),applyCodeCollapse(He,me,!0),at=ht}}scrollToBottom()}
if(ut&&ut(),He){const Oe=snapshotCodeCollapse(He);renderAiMarkdownInto(He,Je,{incrementalMath:!0}),applyCodeCollapse(
He,Oe,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),V)if(queueHighlight(V,Je),enableLatencyMetrics){
const Oe=nowPerfMs()-dt;reportFirstTokenLatency({is_total:!0,latency_seconds:Oe/1e3,latency_ms:Oe,thread_id:Nt||
currentThreadId,job_id:currentJobId,model:A.model,client_sent_at_ms:bt,client_done_at_ms:Date.now()});
let qe='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',tt=null;xt!==null&&(tt=xt),Ct!==null&&(tt===null||Ct<tt)&&(tt=
Ct),ft!==null&&(tt===null||ft<tt)&&(tt=ft),tt!==null&&(qe+=`<div>Initial: ${(tt/1e3).toFixed(2)}s</d\
iv>`),ft!==null&&ft!==tt&&(qe+=`<div>Content: ${(ft/1e3).toFixed(2)}s</div>`),qe+=`<div class="font-\
bold text-gray-300">Total: ${(Oe/1e3).toFixed(2)}s</div>`,currentJobId&&(qe+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),qe+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,qe+="</div>",V.insertAdjacentHTML("beforeend",qe)}else V.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),V&&V.querySelectorAll(".thought-co\
ntent").forEach(qe=>qe.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!B}),!Kt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(Oe=>Oe.json()).then(Oe=>{Oe.title&&
(document.title=Oe.title+" - AI Chat",setCurrentChatHeaderTitle(Oe.title),loadThreads())}):loadThreads(
!1)}catch(R){let de=!1;const Ee=R.name==="AbortError"&&isManualStopAbortForThread(Pe);if(R.name==="A\
bortError"&&!Ee&&(de=await syncThreadAfterAbortedStream(Pe,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${R.message}`),!yt){be&&be.remove();const Ne=V&&V.closest(
".fade-in");Ne&&Ne.remove(),delete messageStore[te],delete messageMeta[te]}if(R.serverCode==="reques\
t_already_accepted"&&R.acceptedJobId&&R.acceptedThreadId)yt=!0,Mt={job_id:R.acceptedJobId,thread_id:String(
R.acceptedThreadId),model:A.model},get("prompt-input").value="",get("prompt-input").style.height="au\
to",resetUploadState(),clearQuote();else if(yt&&!Ee)vt={job_id:normalizeJobIdForUi(currentJobId),thread_id:currentThreadId!=
null?String(currentThreadId):null,model:A.model},window.ConnectionMonitor.setUnavailable("offline"),
showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(R.serverCode==="turnstile_required"){const Ne=await getTurnstileToken();Ne?(await verifyTurnstileOnServer(
Ne,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(R.serverCode==="api_key_missing"){const Ne=R.serverModel||A.model,ze=await showApiKeyRequiredModalAsync(
Ne);ze==="set"?Lt=!0:ze==="switch"?showModal("model-modal"):showToast(R.message||`${getModelNameById(
Ne)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(R.
name!=="AbortError"){const Ne="Connection Error: "+R.message;showToast(Ne,"error",!0)}B&&!de&&G()}finally{
Wt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),ut&&ut(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===U&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(Mt){const R=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=Mt.thread_id,(R!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(Mt,currentThreadId)}
if(vt&&vt.thread_id)return reconnectPendingStreamUntilAvailable(vt,vt.thread_id);if(Lt)return sendMessage()}
o(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?
String(e.model):"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)
return;if(activeStreamingBubbleId=n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-\
area.skeleton-pending")){const G=a.querySelector(".content-area");G?G.outerHTML=buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const r=currentThreadId,
l=i.toLowerCase(),c=l.includes("gemini")||l.includes("o1")||l.includes("o3")||l.includes("gpt-5")||l.
includes("reasoning")&&!l.includes("non-reasoning");let u=null;const f=o(G=>!c||!a?null:((!u||!a.contains(
u))&&(u=a.querySelector(".thought-content")),u||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),u=a.querySelector(".thought-content")),u&&(u.setAttribute("data-pla\
ceholder","1"),u.textContent=G||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),u),
"ensureThoughtPlaceholder");c&&f("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let g="",y="",w="",v=!0,k=null,_=null,C=null,L=!1;const E={};let B=0,K=!1;const Z=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Ie=!1,O=!1;try{const G=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!G.ok)throw new Error(`Resume failed (${G.status})`);window.ConnectionMonitor&&
(O=!0,window.ConnectionMonitor.operationStarted()),Z&&Z.setPhase("waiting");const te=G.body.getReader(),
be=new TextDecoder;for(;!K;){const{done:ue,value:Le}=await te.read();if(ue)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),Z&&Z.setPhase("receiving"),g+=be.decode(Le,{stream:!0});let ye=g.
split(`
`);g=ye.pop();let Me=!1,oe=!1;for(let re of ye)if(re.trim())try{const J=JSON.parse(re);if(J.type==="\
job_id"){currentJobId=J.content||t;continue}if(J.type==="search_status"){J.content==="searching"&&!C?
(a.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),C=a.querySelector(".search-box")):J.content==="done"&&C&&(C.
classList.remove("animate-pulse"),C.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{C&&C.remove(),C=null},2e3));continue}if(J.type==="mcp"){handleMcpStreamEvent(
a,J.content||{});continue}if(J.type==="mcp_decision_request"){openMcpDecisionModal(J.content||{});continue}
if(J.type==="status"){const A=J.content===null||J.content===void 0?"":String(J.content);if(v&&a){const F=A||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(a,F,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const U=a.querySelector(".content-area");U&&(U.outerHTML=buildPendingSkeletonHtml(i,F),updatePendingSkeletonStatus(
a,F,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
c&&f(A||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(v){beginPendingToStreamTransition(
a);const A=a.querySelector(".content-area");A&&(A.innerHTML=""),v=!1}if(J.type==="coding_diff")appendCodingLiveDiff(
a,J.content||{});else if(J.type==="thought"){if(k||(k=a.querySelector(".thought-content")),w+=J.content,
!k){const A='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';C?C.insertAdjacentHTML("afterend",A):a.insertAdjacentHTML("afterbegin",A),k=a.querySelector(
".thought-content")}if(k&&k.getAttribute("data-placeholder")==="1"){if(k.textContent="",k.removeAttribute(
"data-placeholder"),k){const A=k.parentElement.querySelector(".thought-header");A&&A.classList.remove(
"thinking-shimmer")}w=J.content}k.classList.remove("collapsed"),oe=!0}else if(J.type==="image_analys\
is"){const A=J.content===null||J.content===void 0?"":String(J.content);if(!a)continue;let F=a.querySelector(
".image-analysis-box");if(!F){const X='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';C?C.insertAdjacentHTML("afterend",X):a.insertAdjacentHTML("afterbegin",X),F=a.querySelector(
".image-analysis-box")}const U=F.querySelector(".image-analysis-text");U&&(U.textContent=A)}else if(J.
type==="python"){const A=J.content||{},F=A.id||`py_${Date.now()}`;if(!E[F]){const X=`<div class="cod\
e-wrapper python-box collapsed" data-py-id="${F}" data-collapsed="true" data-code-key="${F}"><div cl\
ass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><di\
v class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B">\
<i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" t\
itle="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" dat\
a-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></b\
utton></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code<\
/div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-secti\
on"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"><\
/code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",X):a.insertAdjacentHTML("afterbegi\
n",X),E[F]=a.querySelector(`[data-py-id="${F}"]`)}const U=E[F];if(U){if(A.code!==void 0){const X=A.code==
null?"":String(A.code),ie=U.querySelector(".python-code");ie&&(ie.textContent=X,ie.removeAttribute("\
data-highlighted"),queueHighlight(U,X));const Te=U.querySelector('.copy-btn[data-copy="code"]');Te&&
Te.setAttribute("data-code",encodeURIComponent(X).replace(/'/g,"%27"))}if(A.output!==void 0){const X=A.
output==null?"":String(A.output),ie=U.querySelector(".python-output");ie&&(ie.textContent=X);const Te=U.
querySelector('.copy-btn[data-copy="output"]');Te&&Te.setAttribute("data-code",encodeURIComponent(X).
replace(/'/g,"%27"))}}}else if(J.type==="content"){const A=J.content===null||J.content===void 0?"":String(
J.content);y+=A,/[`~]/.test(A)&&activateDeferredCodingModeFromStream(y),_||(_=a.querySelector(".cont\
ent-area")||document.createElement("div"),_.className="prose prose-invert text-sm break-words",a.contains(
_)||a.appendChild(_)),Me=!0}else if(J.type==="error"){L=!0,K=!0,a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
J.content)),showToast(J.content||"Unknown error","error",!0);break}}catch{}if(oe&&k&&(k.textContent=
w,userAutoScroll&&(k.scrollTop=k.scrollHeight)),Me&&_){const re=Date.now();if(re-B>100){const J=snapshotCodeCollapse(
_);renderAiMarkdownInto(_,y,{incrementalMath:!0}),applyCodeCollapse(_,J,!0),B=re}}scrollToBottom()}if(Z&&
Z(),_){const ue=snapshotCodeCollapse(_);renderAiMarkdownInto(_,y,{incrementalMath:!0}),applyCodeCollapse(
_,ue,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,y),a&&a.querySelectorAll(".thought-content").
forEach(Le=>Le.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(G){const te=G.name==="AbortError"&&isManualStopAbortForThread(r);G.name==="Abo\
rtError"&&!te&&await syncThreadAfterAbortedStream(r,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
te||(Ie=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{O&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),Z&&Z(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Ie)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i},r)}o(resumePendingStream,"resumePendingStream");function updateThreadHighlighting(){
const e=get("thread-list");if(!e)return;e.querySelectorAll("[data-thread-id]").forEach(n=>{n.dataset.
threadId===String(currentThreadId)?n.classList.add("bg-gray-700/60","border-l-2","border-blue-500"):
n.classList.remove("bg-gray-700/60","border-l-2","border-blue-500")})}o(updateThreadHighlighting,"up\
dateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){snapshotSidebarHistory("\
loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory("loadThreads-s\
tart append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?t.value:"";
if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");return}const a=await(await apiFetch(
`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&page=${threadPage}`)).json(),r=get("th\
read-list");if(!r)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-r\
eplace-settings-open");return}const c=a&&Array.isArray(a.threads)?a.threads.length:-1,u=r.querySelectorAll(
"[data-thread-id]").length;if(c===0&&u>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const u=String(
c.id),f=document.createElement("div"),g=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",v=u===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${v}`,f.dataset.threadId=u,f.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${g} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${u}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(c.title||"No Title")}</spa\
n>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${u}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${u}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,f.onclick=k=>{k.target.closest("button")||k.target.closest("[dat\
a-thread-actions]")||loadMessages(u)},l?r.insertBefore(f,l):r.appendChild(f)}),hasMoreThreads=!!a.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+a.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error("Faile\
d to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,r=88,l=52,c=.5,u=8;let f=0,g=!1,y=0,w=null;
const v=o(()=>get(i),"indicatorEl"),k=o(()=>{const L=v();return L?L.querySelector(".ptr-pull-label"):
null},"labelEl"),_=o(L=>{const E=v();if(!E)return;E.style.height=Math.min(L,r)+"px",E.classList.toggle(
"active",L>2),E.classList.toggle("pull-ready",L>=a);const B=k();B&&(B.textContent=L>=a?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),C=o(()=>{const L=v();L&&(L.style.height=
"0px",L.classList.remove("active","pull-ready","refreshing"),L.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",L=>{if(w){g=!1;return}if(n.scrollTop>0){g=!1;return}const E=L.
touches[0];E&&(f=E.clientY,y=0,g=!0)},{passive:!0}),n.addEventListener("touchmove",L=>{if(!g||w)return;
if(n.scrollTop>0){g=!1;return}const E=L.touches[0];if(!E)return;const B=E.clientY-f;if(B<=0){y>0&&(y=
0,_(0)),g=!1;return}const K=v();K&&!K.classList.contains("dragging")&&K.classList.add("dragging"),y=
Math.min(B*c,r),_(y),B>=u&&L.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!g||
(g=!1,w))return;const L=v();L&&L.classList.remove("dragging");const E=y>=a;if(y=0,!E){C();return}let B;
try{B=t()}catch{B=null}const K=v();if(K){K.classList.add("refreshing"),K.style.height=l+"px";const Z=K.
querySelector(".ptr-pull-label");Z&&(Z.textContent="\u66F4\u65B0\u4E2D...")}B&&typeof B.then=="funct\
ion"?(w=B,B.catch(()=>{}).finally(()=>{w=null,C()})):(w=Promise.resolve(),setTimeout(()=>{w=null,C()},
400))}),n.addEventListener("touchcancel",()=>{g=!1,y=0,C()})}o(initPullToRefresh,"initPullToRefresh");
const initThreadPullToRefresh=o(()=>initPullToRefresh("thread-list",()=>loadThreads(!1)),"initThread\
PullToRefresh"),initGemPullToRefresh=o(()=>initPullToRefresh("gem-list",()=>loadGems()),"initGemPull\
ToRefresh"),initPullToRefreshAll=o(()=>{initThreadPullToRefresh(),initGemPullToRefresh()},"initPullT\
oRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;const mcpCardIdSelector=o(e=>"mcp_\
card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),mcpEscHtml=o(e=>String(e==null?
"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]),"mcpE\
scHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||"MCP")} / ${mcpEscHtml(e.tool_name||
e.internal_name||"")}`}o(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){if(!e)return null;
let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("div"),t.className=
"mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),e.appendChild(
t)),t}o(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){if(!e||!t||!t.
type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(e):null;if(n&&
!i)return;const a=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.querySelector(
'[data-mcp-card="'+a+'"]'))return;const r=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${a}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",r);return}if(t.type==="result"){let r=i.querySelector('[dat\
a-mcp-card="'+a+'"]');const l=t.summary||"";if(r)r.classList.remove("mcp-running"),r.classList.add("\
mcp-done"),r.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const c=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${a}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",c),r=i.querySelector('[data-mcp-card="'+a+'"]')}if(l){const c=document.
createElement("div");c.className="mcp-box-note",c.textContent=l.split(`
`)[0].slice(0,220),r&&r.appendChild(c)}return}if(t.type==="error"){let r=i.querySelector('[data-mcp-\
card="'+a+'"]');const l=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(r)r.classList.remove("mcp-running"),r.classList.add("mcp-error"),r.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const u=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",u),r=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
r&&r.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const r=get("mcp-decision-modal");if(r&&!r.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}o(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const u=JSON.parse(c);c=JSON.stringify(u,null,2)}catch{}a.textContent=c}const r=get(
"mcp-decision-allow"),l=get("mcp-decision-deny");r&&(r.onclick=()=>submitMcpDecision("allow")),l&&(l.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}o(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}o(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const a=e[0],r=a.
thread_id;n.textContent=e.length===1?`${a.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),r&&await loadMessages(r)});const l=get(
"batch-notification-close");l&&(l.onclick=()=>t.classList.add("hidden"))}o(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({}));(t.active||[]).some(a=>currentThreadId&&String(a.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const i=t.
completed||[];i.length&&(showGeminiBatchCompletionBanner(i),i.some(a=>String(a.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}o(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);let chatTransitionSequence=0,chatTransitionTimer=null;const CHAT_TRANSITION_DURATION_MS=460;function playChatTransition(e){
const t=get("chat-transition-veil");if(!t)return;const n=++chatTransitionSequence;if(chatTransitionTimer&&
(clearTimeout(chatTransitionTimer),chatTransitionTimer=null),window.matchMedia&&window.matchMedia("(\
prefers-reduced-motion: reduce)").matches){t.classList.remove("is-active"),t.removeAttribute("data-t\
ransition-kind");return}t.dataset.transitionKind=e||"history",t.classList.remove("is-active"),t.offsetWidth,
t.classList.add("is-active"),chatTransitionTimer=setTimeout(()=>{n===chatTransitionSequence&&(t.classList.
remove("is-active"),chatTransitionTimer=null)},CHAT_TRANSITION_DURATION_MS)}o(playChatTransition,"pl\
ayChatTransition");async function toggleBookmark(e,t){e&&e.stopPropagation(),await apiFetch(`/api/th\
reads/${t}/bookmark`,{method:"POST"}),loadThreads()}o(toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){
const n=++threadLoadSequence;window.closeHistoryModal&&window.closeHistoryModal();const i=!!t.preserveDraft,
a=!!t.silent;a||resumeChatAutoScroll({scroll:!1});const r=a?snapshotCodeCollapseByMessage(get("chat-\
container")):null;let l="",c="",u=[];if(i){const f=get("prompt-input");l=f?f.value:"",c=f?f.style.height:
"",u=currentImageUrls?currentImageUrls.slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();
a||playChatTransition("history"),currentThreadId=e!=null?String(e):e,t.skipHistory||history.pushState(
{},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),get("welcome-scree\
n").classList.add("hidden"),a||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());try{
const f=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);f.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const g=await apiFetch(f.toString());if(!g.
ok)throw new Error(`thread request failed (${g.status})`);const y=await g.json();if(!y||!Array.isArray(
y.messages))throw new Error("invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(
y&&y.title),allMessages=y.messages,threadHasOlderMessages=!!y.has_older_messages,oldestLoadedMessageId=
y.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const w=(allMessages||[]).filter(k=>k.
role==="user"&&k.content).map(k=>k.content);if(promptHistory=[...new Set(w.slice().reverse())],historyIndex=
-1,tempPrompt="",currentThreadPending=y.pending_job||null,setTemporaryChatUiState(!!(y&&y.is_temporary)),
applyTemporaryChatRuntimeMeta(y||{}),ensureTemporaryChatHeartbeat(!0),get("thread-custom-instruction")&&
(get("thread-custom-instruction").value=y.custom_instruction||""),y.last_model&&selectModelById(y.last_model),
get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!!y.enable_prompt_caching,updatePromptCacheUi()),
y.last_gem_uuid&&loadedGems.length>0){const k=loadedGems.find(_=>_.uuid===y.last_gem_uuid);k&&(threadGemMap[currentThreadId]=
k,applyActiveGem(k))}const v=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`);
if(v&&allMessages.find(k=>String(k.id)===String(v))?currentLeafId=v:allMessages.length>0?currentLeafId=
allMessages[allMessages.length-1].id:currentLeafId=null,renderThreadTree(a?{silent:a,keepScroll:a}:{
silent:a,keepScroll:a,animate:!0}),a&&r?applyCodeCollapseByMessage(get("chat-container"),r,!0):a||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!a&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),i){const k=get("prompt-input");k&&(k.value=l||"",
c?k.style.height=c:k.style.height="auto"),currentImageUrls=u,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(f){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",f),a||showChatLoadError(e),a||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}o(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const r=await(await apiFetch(i.toString())).json(),l=Array.isArray(r.messages)?r.messages:[];
if(l.length){const c=new Set(allMessages.map(f=>f.id)),u=l.filter(f=>!c.has(f.id));u.length&&(allMessages=
u.concat(allMessages))}if(threadHasOlderMessages=!!r.has_older_messages,oldestLoadedMessageId=r.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;let r=null;if(i&&(r=a.scrollTop),a.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const l={};allMessages.
forEach(v=>{l[v.id]=v,v.childrenIds=[]}),allMessages.forEach(v=>{v.parent_id&&l[v.parent_id]&&l[v.parent_id].
childrenIds.push(v.id)}),(!currentLeafId||!l[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const c=[];let u=l[currentLeafId];for(;u;)c.unshift(u),u=l[u.parent_id];const f=buildTokenTotals(
c),g=buildTokenTotals(allMessages),y=document.createDocumentFragment();if(threadHasOlderMessages){const v=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
k=loadingOlderMessages?"disabled":"",_=document.createElement("div");_.className="mb-3 text-center",
_.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${k}><i class="fas fa-clock-rotate-left mr-1"></i>${v}</button>`,y.appendChild(_)}
c.forEach(v=>{const k=v.parent_id?l[v.parent_id]:null,_=k?k.childrenIds:allMessages.filter(L=>!L.parent_id).
map(L=>L.id),C=_.length>1?{current:_.indexOf(v.id)+1,total:_.length,siblings:_}:null;renderMessage(v.
id,v.role,v.content,v.image_url,v.thought_data,v.model,C,n,v.quote_text,v.tokens,v.tokens_in,v.tokens_out,
v.is_encrypted,v.tokens_content,v.tokens_thought,y,!1,v.parent_id,v.gem_name,v.batch_job)});const w=currentThreadPending;
if(w&&!isPendingJobSuppressed(w.job_id)){const v=w.message_id,k=new Set(c.map(L=>L.id)),_=c.length?c[c.
length-1]:null;if(v&&k.has(v)&&currentLeafId===v||!v&&_&&_.role==="user"){const L=w.job_id?`pending-${w.
job_id}`:null;renderPendingMessage(y,n,!1,L,w.model||null)}}if(a.appendChild(y),updateTotalTokenBar(
f.tokens_total,f,g),currentParentId=currentLeafId,i&&r!==null?restoreThreadTreeScroll(a,r):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(a),c.length){
const v=c[c.length-1]&&c[c.length-1].content;queueMathTypeset(a,v)}}o(renderThreadTree,"renderThread\
Tree");function restoreThreadTreeScroll(e,t){if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&
!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=
e.scrollTop,syncScrollToBottomButton()}o(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){
currentLeafId=e;const t={};allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(
i=>{i.parent_id&&t[i.parent_id]&&t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=
allMessages.length>0?allMessages[allMessages.length-1].id:null,renderThreadTree({animate:!0});return}
for(;t[n]&&t[n].childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree(
{animate:!0})}o(switchVersion,"switchVersion");async function loadGems(){try{const t=await(await apiFetch(
CHAT_CONFIG.urls.handleGems)).json();loadedGems=t;const n=get("gem-list");if(!n)return;n.innerHTML='\
<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-do\
wn ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-l\
abel"></span></div>',Array.isArray(t)&&t.forEach(i=>{const a=document.createElement("div");a.className=
"gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between it\
ems-center group",a.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa\
-gem text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><div class="flex it\
ems-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group\
-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${i.uuid}')"><i class="fas fa-p\
encil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:o\
pacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${i.uuid}')"><i class\
="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=r=>{r.target.closest("button")||activateGem(
i)},n.appendChild(a)})}catch(e){console.error("Failed to load gems:",e)}}o(loadGems,"loadGems");function setGemDefaultModelSelect(e){
const t=get("gem-default-model");if(!t)return;t.innerHTML="";const n=document.createElement("option");
n.value="",n.textContent="Use current model",t.appendChild(n),MODELS.forEach(a=>{const r=(a.items||[]).
filter(c=>!c.deprecated);if(!r.length)return;const l=document.createElement("optgroup");l.label=a.category,
r.forEach(c=>{const u=document.createElement("option");u.value=c.id,u.textContent=c.name,l.appendChild(
u)}),t.appendChild(l)});const i=e||"";if(i&&!Array.from(t.options).some(a=>a.value===i)){const a=document.
createElement("option");a.value=i,a.textContent=MODEL_NAME_BY_ID[i]||i,t.appendChild(a)}t.value=i}o(
setGemDefaultModelSelect,"setGemDefaultModelSelect");async function openEditGemModal(e,t){e.stopPropagation(),
editingGemUuid=t;try{const i=await(await apiFetch(`/api/gems/${t}`)).json();get("gem-name").value=i.
name,get("gem-desc").value=i.description||"",get("gem-inst").value=i.instruction,setGemDefaultModelSelect(
i.default_model),renderGemFixedPromptsForEdit(i.fixed_prompts),get("gem-modal-title").innerHTML='<i \
class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-btn").innerText="Save Changes",showModal(
"gem-modal"),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")}catch{showToast(
"Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}o(openEditGemModal,"o\
penEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.urls.handleGems,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,instruction:t})}),loadGems()}
o(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get("fixed-prompts-bar");
if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),get("active-gem-n\
ame").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),t){t.innerHTML=
"";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}n.length>0?(t.
classList.remove("hidden"),n.forEach((i,a)=>{const r=document.createElement("button");r.className="f\
ixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover:bg-gray-600 t\
ext-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex items-center",r.style.
animationDelay=`${a*40}ms`,r.textContent=String(i.name||""),r.onclick=()=>{const l=get("prompt-input");
l&&(l.value=i.content,l.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(r)})):t.classList.
add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML="",t.classList.
add("hidden"));get("sys-prompt-option").style.opacity="1"}o(applyActiveGem,"applyActiveGem");function syncActiveGemForThread(e){
const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(t)}o(syncActiveGemForThread,"syncActi\
veGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_gem_uuid:t,thread_id:e})})}catch{}}
o(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){currentThreadId?(threadGemMap[currentThreadId]=
e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadGemUuid(currentThreadId,e?e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(
e),allMessages&&allMessages.length>0&&startNewChat({preserveGem:!0}))}o(activateGem,"activateGem");function clearActiveGem(){
currentThreadId&&(delete threadGemMap[currentThreadId],saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=
null,applyActiveGem(null)}o(clearActiveGem,"clearActiveGem");function addGemFixedPromptRow(e="",t=""){
const n=get("gem-fixed-prompts-container");if(!n)return;const i=document.createElement("div");i.className=
"flex gap-2 items-start gem-fixed-prompt-row ui-enter",i.innerHTML=`
                <input type="text" class="gem-fp-name bg-gray-900 border border-gray-600 rounded p-1\
.5 text-white text-[10px] w-24" placeholder="\u540D\u524D" value="${escapeHtml(e)}" autocomplete="of\
f" spellcheck="false">
                <textarea class="gem-fp-content flex-1 bg-gray-900 border border-gray-600 rounded p-\
1.5 text-white text-[10px] h-9 resize-none" placeholder="\u30D7\u30ED\u30F3\u30D7\u30C8\u5185\u5BB9" spellcheck="false">${escapeHtml(
t)}</textarea>
                <button type="button" class="text-gray-500 hover:text-red-400 p-1.5" onclick="this.p\
arentElement.remove()"><i class="fas fa-times"></i></button>
            `,n.appendChild(i)}o(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),t=[];return e.forEach(n=>{const i=n.querySelector(
".gem-fp-name").value.trim(),a=n.querySelector(".gem-fp-content").value.trim();i&&a&&t.push({name:i,
content:a})}),t.length>0?JSON.stringify(t):null}o(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const t=get("gem-fixed-prompts-container");if(t){t.innerHTML="";try{e&&JSON.parse(e).forEach(i=>addGemFixedPromptRow(
i.name,i.content))}catch{}}}o(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}o(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}o(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=[
"sidebar-chat-title","mobile-chat-title"],a=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],r=["sidebar-chat-ttl","mobile-chat-ttl"];i.forEach(l=>{const c=get(l);c&&(c.textContent=e)}),
a.forEach(l=>{const c=get(l);c&&c.classList.toggle("hidden",!n)}),r.forEach(l=>{const c=get(l);c&&(n&&
t?(c.textContent=t,c.classList.remove("hidden")):(c.textContent="",c.classList.add("hidden")))})}o(updateCurrentChatHeaderUi,
"updateCurrentChatHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="str\
ing"?e:null,updateCurrentChatHeaderUi()}o(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){
tempChatExpiresAtMs=null,updateCurrentChatHeaderUi()}o(resetTemporaryChatExpiresAt,"resetTemporaryCh\
atExpiresAt");function applyTemporaryChatRuntimeMeta(e){if(!e||typeof e!="object")return;Object.prototype.
hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(e.timeout_seconds);let t=null;
const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.floor(n*1e3);else{const i=Number(
e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=
t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=null),updateCurrentChatHeaderUi()}
o(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
o(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,t=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let n=Number(e);return Number.isFinite(n)||(n=Number(t)),Number.isFinite(n)||(n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
n=Math.trunc(n),n<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(n=TEMP_CHAT_TIMEOUT_MIN_SECONDS),n>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(n=TEMP_CHAT_TIMEOUT_MAX_SECONDS),n}o(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
n=get("temporary-chat-welcome-desc");n&&(n.textContent=t);const i=get("temporary-chat-container");i&&
(i.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
o(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const t=get(
"set-temp-chat-timeout-seconds");t&&(t.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}o(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,t))}o(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const t=get("enable-temporary-c\
hat");t&&t.checked!==temporaryChatEnabled&&(t.checked=temporaryChatEnabled);const n=get("welcome-def\
ault-content");n&&n.classList.toggle("hidden",temporaryChatEnabled);const i=get("welcome-temporary-c\
ontent");i&&i.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}o(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}o(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}o(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const t=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
n=await t.json().catch(()=>({}));t.ok&&n&&applyTemporaryChatRuntimeMeta(n),t.ok&&n&&n.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}o(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const t=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==t)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=t,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}o(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const t=!!e;if(setTemporaryChatUiState(
t),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const n=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:t})}),i=await n.json().catch(()=>({}));if(!n.ok)throw new Error(i&&i.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(i&&i.is_temporary)),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}o(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
if(playChatTransition("new"),threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),
resetUploadState(),stopTemporaryChatHeartbeat(),setTemporaryChatUiState(!1),currentThreadTitle=null,
tempChatExpiresAtMs=null,currentThreadId=null,allMessages=[],promptHistory=[],historyIndex=-1,tempPrompt=
"",threadHasOlderMessages=!1,oldestLoadedMessageId=null,loadingOlderMessages=!1,currentLeafId=null,currentParentId=
null,currentThreadPending=null,updateTotalTokenBar(0),typeof window.__refreshAdminThreadEncState=="f\
unction")try{window.__refreshAdminThreadEncState()}catch{}e.skipHistory||history.pushState({},"","/"),
get("chat-container").innerHTML="",get("welcome-screen").classList.remove("hidden"),updateCurrentChatHeaderUi(),
get("thread-custom-instruction")&&(get("thread-custom-instruction").value=""),get("enable-prompt-cac\
he")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(
activeGem):applyActiveGem(null),loadThreads(),window.innerWidth<768&&get("overlay").click()}o(startNewChat,
"startNewChat");let threadModalLoadSeq=0;window.openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=i.id!==null&&i.id!==void 0?
String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.is_temporary)),setCurrentChatHeaderTitle(i&&i.title),
applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),history.pushState({},"","/c/"+
i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,t=String(currentThreadId);modalThreadId=t,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[n,i]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${t}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==t)return;if(n.ok){const a=await n.
json(),r=get("thread-app-global-sys-prompt-preview");r&&(r.value=a.global_system_prompt_effective||"");
const l=get("thread-app-global-sys-prompt-preview-status");l&&(a.global_system_prompt_enabled===!1?l.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":a.global_system_prompt_uses_time_fallback?
l.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
l.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=a.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=a.system_prompt_enabled!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=a.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",a.auto_system_prompt_notices_config||{})}if(i.ok){const a=await i.json();if(e!==threadModalLoadSeq||
modalThreadId!==t)return;const r=get("thread-custom-instruction");r&&(r.value=a.custom_instruction||
"");const l=get("thread-include-global-instruction");l&&(l.checked=a.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),a=i?i.value:"",r=get("thread-include-global-instruction"),l=r?r.checked:
!0,c=get("thread-global-sys-prompt"),u=get("thread-global-sys-prompt-enabled");let f=null;try{f=c||u?
{system_prompt:c?c.value:"",system_prompt_enabled:u?u.checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:collectAutoSystemPromptConfigFromForm(
"thread")}:null}catch(g){sendClientDebugLog("error","Payload construction failed: "+g.message)}try{sendClientDebugLog(
"info","Starting PUT request for thread: "+e);const g=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+g.status);let y=!0;if(f){sendClientDebugLog(
"info","Starting POST request for user settings");const w=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});y=w.ok,sendClientDebugLog(
"info","POST request finished, status: "+w.status)}g.ok&&y?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\
\u308C\u307E\u3057\u305F","success")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(g){sendClientDebugLog("error","Save failed with error: "+g.message),showToast("\u30A8\u30E9\u30FC\
: "+g.message,"error",!0)}finally{t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=
()=>{syncCompressionSettingsUi(),showModal("compression-modal"),location.pathname!=="/compression"&&
history.pushState({modal:"compression"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal(
"compression-modal"),!e&&location.pathname==="/compression"&&history.back()},get("save-compression-s\
ettings-btn").onclick=()=>{const e=get("compression-max-size").value,t=get("compression-max-dim").value,
n=get("compression-output-type").value,i=get("compression-format-only").checked;setCompressionSettings(
e,t,n,i);const a=o((l,c)=>{get(l)&&get(c)&&(get(c).value=get(l).value)},"syncBack");a("modal-gpt-ima\
ge-size","gpt-image-size"),a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-forma\
t","gpt-image-format"),a("modal-gpt-image-compression","gpt-image-compression"),a("modal-gemini-imag\
e-aspect","gemini-image-aspect"),a("modal-gemini-image-size","gemini-image-size"),a("modal-grok-imag\
e-aspect","grok-image-aspect"),a("modal-grok-image-resolution","grok-image-resolution"),a("modal-gro\
k-image-quality","grok-image-quality"),a("modal-ocr-table-format","ocr-table-format"),a("modal-ocr-p\
ages","ocr-pages");const r=o((l,c)=>{get(l)&&get(c)&&(get(c).checked=get(l).checked)},"syncBackChk");
r("modal-ocr-extract-header","ocr-extract-header"),r("modal-ocr-extract-footer","ocr-extract-footer"),
r("modal-ocr-include-blocks","ocr-include-blocks"),r("modal-ocr-include-images","ocr-include-images"),
window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","s\
uccess")};async function deleteGem(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.
urls.handleGemItem.replace("0",t),{method:"DELETE"}),loadGems())}o(deleteGem,"deleteGem");async function renameThread(e,t){
e.stopPropagation();const n=prompt("Title:");if(n){const i=await apiFetch(CHAT_CONFIG.urls.updateTitle.
replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:n})}),
a=await i.json().catch(()=>({}));i.ok&&currentThreadId===String(t)&&setCurrentChatHeaderTitle(a&&a.title||
n),loadThreads()}}o(renameThread,"renameThread");async function deleteThread(e,t){e.stopPropagation(),
confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),
currentThreadId===t?startNewChat():loadThreads())}o(deleteThread,"deleteThread");async function deleteMessage(e,t){
!t&&!confirm("Delete this message and subsequent history?")||(await apiFetch(CHAT_CONFIG.urls.deleteMessage.
replace("0",e),{method:"DELETE"}),loadMessages(currentThreadId))}o(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=o(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=o(
e=>{if(!e)return"";try{const t=new Date(e);return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(t)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=o(e=>{if(!e)
return"";let t=String(e).trim();if(!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=o(e=>{const t=pdfNormalizeAttachmentPath(e);return t?
`${window.location.origin}/files/${encodeURI(t)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=o(
e=>{const t=pdfNormalizeAttachmentPath(e);return t?`${window.location.origin}/${PDF_IMAGE_EXTS.has((t.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(t)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=o(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const a=n&&n.filename?n.
filename:i.split("/").pop(),r=n&&n.source?String(n.source):"attachment",l=!!(n&&n.is_image),c=n&&n.url?
n.url:buildPdfAttachmentUrl(i),u=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:r,isImage:l,url:c,previewUrl:u}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(u=>maybeNeedsMathJax(u.content)||maybeNeedsMathJax(u.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",r=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(u=>{const f=u.
role==="user",g=u.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(u.quote_text)}\
</div>`:"",y=u.thought_text?`<div class="thought">${escapeHtml(u.thought_text)}</div>`:"",w=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(u.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(u.content||"")}</div>`,v=buildPdfMessageAttachments(u),k=v.length?`<div \
class="attachments">${v.map(L=>L.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(L.previewUrl)}\
" alt="${pdfEscapeAttr(L.filename)}"><div class="file-caption">${pdfEscapeAttr(L.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(L.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
L.filename)}</span><span class="file-source">${pdfEscapeAttr(L.source)}</span></span></a></div>`).join(
"")}</div>`:"",_=[];u.model&&!f&&_.push(u.model),u.tokens!==null&&u.tokens!==void 0&&_.push(`tokens:${u.
tokens}`),u.tokens_in!==null&&u.tokens_in!==void 0&&_.push(`in:${u.tokens_in}`),u.tokens_out!==null&&
u.tokens_out!==void 0&&_.push(`out:${u.tokens_out}`),u.tokens_thought!==null&&u.tokens_thought!==void 0&&
_.push(`thought:${u.tokens_thought}`),u.is_encrypted&&_.push("encrypted"),u.parent_id!==null&&u.parent_id!==
void 0&&_.push(`parent:#${u.parent_id}`);const C=_.length?`<div class="message-meta">${pdfEscapeAttr(
_.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(u.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${g}
                            ${w}
                            ${y}
                            ${k}
                            ${C}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(r)} - PDF Export</title>
        ${a}
        <style>
        :root { --ink:#0f172a; --muted:#475569; --line:#dbe3ee; --panel:#fff; --panel-soft:#f8fafc; \
--user:#0ea5e9; --ai:#10b981; --accent:#0f766e; }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body { font-family: "Noto Sans JP", system-ui, sans-serif; color: var(--ink); background: li\
near-gradient(180deg, #eef4fb 0%, #f8fbff 45%, #eef2f7 100%); }
        .page { max-width: 980px; margin: 0 auto; padding: 24px 18px 48px; }
        .cover { position: relative; overflow: hidden; border-radius: 26px; padding: 26px 24px; colo\
r: #eff6ff; background: linear-gradient(135deg, #0f172a 0%, #0b3b57 56%, #0f766e 100%); box-shadow: \
0 24px 48px rgba(15, 23, 42, 0.18); }
        .cover h1 { margin: 0 0 8px; font-size: 40px; line-height: 1.1; }
        .cover p { margin: 0; max-width: 72ch; color: rgba(226, 232, 240, 0.88); font-size: 14px; li\
ne-height: 1.8; }
        .meta-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; mar\
gin-top: 18px; }
        .meta-card { padding: 12px 14px; border-radius: 16px; background: rgba(15, 23, 42, 0.2); bor\
der: 1px solid rgba(255,255,255,0.15); }
        .meta-label { font-size: 11px; letter-spacing: 0.08em; color: rgba(226,232,240,0.68); margin\
-bottom: 6px; text-transform: uppercase; }
        .meta-value { font-size: 14px; font-weight: 700; word-break: break-word; }
        .message-list { margin-top: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message { border-radius: 22px; border: 1px solid var(--line); background: var(--panel); box\
-shadow: 0 14px 30px rgba(15, 23, 42, 0.06); overflow: hidden; break-inside: avoid; page-break-insid\
e: avoid; }
        .message.user { border-left: 6px solid var(--user); }
        .message.ai { border-left: 6px solid var(--ai); }
        .message-head { display:flex; gap:10px; justify-content:space-between; align-items:flex-star\
t; padding:14px 18px 0; }
        .message-role { display:inline-flex; align-items:center; gap:8px; font-weight:900; font-size\
:13px; }
        .message-role .dot { width:10px; height:10px; border-radius:50%; background: currentColor; }\

        .message-time { color: var(--muted); font-size: 11px; white-space: nowrap; }
        .message-body { padding: 12px 18px 18px; }
        .quote { margin:0 0 12px; padding:10px 12px; border-left:4px solid rgba(14,165,233,0.7); bac\
kground: var(--panel-soft); color: var(--muted); border-radius:12px; font-size:12px; line-height:1.7\
; }
        .thought { margin: 12px 0 0; padding: 12px 14px; border-radius: 14px; background: rgba(139, \
92, 246, 0.06); border: 1px solid rgba(139, 92, 246, 0.18); color: #4c1d95; font-size: 12px; line-he\
ight: 1.8; white-space: pre-wrap; }
        .content { font-size: 14px; line-height: 1.85; word-break: break-word; }
        .content pre { overflow:auto; padding:12px 14px; border-radius:14px; background:#0b1020; col\
or:#e2e8f0; border:1px solid rgba(15,23,42,0.18); }
        .content code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospac\
e; }
        .content blockquote { margin:12px 0; padding:8px 12px; border-left:4px solid rgba(14,165,233\
,0.65); background: rgba(14,165,233,0.06); border-radius:10px; color:#334155; }
        .content img { max-width:100%; height:auto; border-radius:14px; border:1px solid rgba(148,16\
3,184,0.28); margin:10px 0; }
        .attachments { margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minm\
ax(180px, 1fr)); gap: 12px; }
        .attachment { border-radius: 16px; border: 1px solid var(--line); background: #f8fafc; overf\
low: hidden; break-inside: avoid; }
        .attachment img { width: 100%; height: auto; display: block; }
        .attachment .file { display:flex; gap:10px; align-items:center; padding:12px 14px; color: va\
r(--ink); text-decoration:none; }
        .file-icon { font-size: 18px; color: var(--accent); }
        .file-name { display:block; font-weight:700; font-size:13px; word-break:break-word; }
        .file-source { display:block; color: var(--muted); font-size: 11px; margin-top: 3px; }
        .file-caption { padding:10px 12px; font-size:11px; color: var(--muted); }
        .message-meta { margin-top: 14px; text-align: right; color: var(--muted); font-size: 11px; l\
ine-height: 1.7; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        @media print { body { background:#fff; } .page { padding:0; max-width:none; } .cover, .messa\
ge { box-shadow:none; } .message, .attachment, .meta-card { break-inside: avoid; page-break-inside: \
avoid; } a { color: inherit; text-decoration: none; } }
        </style>
        </head>
        <body>
        <div class="page">
        <section class="cover">
        <h1>${pdfEscapeAttr(r)}</h1>
        <p>\u30B9\u30EC\u30C3\u30C9 ID: ${pdfEscapeAttr(t.public_id||"")}\u3002\u8868\u793A\u4E2D\u306E\u5C65\u6B74\u3092\u305D\u306E\u307E\u307E\u5370\u5237\u3067\u304D\u308B\u3088\u3046\u306B\u3001\u753B\u9762\u30AD\u30E3\u30D7\u30C1\
\u30E3\u3067\u306F\u306A\u304F\u5168\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u518D\u69CB\u6210\u3057\u3066\u51FA\u529B\u3057\u3066\u3044\u307E\u3059\u3002</p>
        <div class="meta-grid">
        ${l.map(u=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(u.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(u.value)}</div></div>`).join("")}
        </div>
        </section>
        <main id="pdf-message-list" class="message-list">${c||'<div class="meta-card" style="margin-\
top:20px;background:#fff;color:var(--muted);text-align:center;border:1px dashed rgba(148,163,184,0.5\
);padding:28px;border-radius:20px;">\u3053\u306E\u30B9\u30EC\u30C3\u30C9\u306B\u306F\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'}\
</main>
        </div>
        </body>
        </html>`},"buildPdfDocumentHtml");async function openThreadPdfPrintDialog(){if(!currentThreadId){
showToast("PDF\u5316\u3059\u308B\u30B9\u30EC\u30C3\u30C9\u3092\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}if(activePdfPrintFrame){showToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const e={isLock:!0};activePdfPrintFrame=e;const t=showProgressToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\
\u3059","info");t.update(5);try{const n=new URL(PDF_PRINT_ROUTE.replace("0",currentThreadId),window.
location.origin);currentLeafId!=null&&String(currentLeafId).trim()&&n.searchParams.set("leaf_id",String(
currentLeafId));const i=await apiFetch(n.toString(),{headers:{Accept:"application/json"}});if(t.update(
20),!i.ok){activePdfPrintFrame=null,t&&t.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const a=await i.json().catch(()=>null);if(t.update(30),!a){activePdfPrintFrame=null,
t&&t.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u89E3\u6790\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const r=document.createElement("iframe");activePdfPrintFrame=r,r.setAttribute("ar\
ia-hidden","true"),r.style.position="fixed",r.style.right="0",r.style.bottom="0",r.style.width="1px",
r.style.height="1px",r.style.opacity="0",r.style.pointerEvents="none",r.style.border="0";let l=null;
const c=o(()=>{l&&(clearTimeout(l),l=null),t&&t.remove(),(activePdfPrintFrame===r||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{r.parentNode&&r.parentNode.removeChild(r)}catch{}},"cleanup");l=setTimeout(
()=>{activePdfPrintFrame===r&&(console.log("PDF print cleanup fallback triggered"),c())},6e4),r.onload=
async()=>{try{const g=r.contentDocument,y=r.contentWindow;if(!g||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(L=>maybeNeedsMathJax(L.content)||maybeNeedsMathJax(L.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),g.fonts&&g.fonts.ready)try{await g.fonts.ready}catch{}t.update(60);const k=Array.from(g.
images||[]),_=Promise.all(k.map(L=>L.complete?Promise.resolve():new Promise(E=>{L.addEventListener("\
load",E,{once:!0}),L.addEventListener("error",E,{once:!0})})));if(await Promise.race([_,new Promise(
L=>setTimeout(L,5e3))]),t.update(80),g.getElementById("MathJax-script")){let L=0;for(;L<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(E=>setTimeout(E,50)),L++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(E){console.error(
"PDF MathJax typeset failed",E)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const u=buildPdfDocumentHtml(a),f=new Blob([u],{type:"text/html"});r.src=URL.createObjectURL(
f),document.body.appendChild(r)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=o((u,f)=>i.
compare(u.filename||"",f.filename||""),"nameAsc"),r=o((u,f)=>i.compare(f.filename||"",u.filename||""),
"nameDesc"),l=o((u,f)=>(Number(f.ts)||0)-(Number(u.ts)||0),"tsDesc"),c=o((u,f)=>(Number(u.ts)||0)-(Number(
f.ts)||0),"tsAsc");return t==="name_asc"?n.sort((u,f)=>a(u,f)||l(u,f)):t==="name_desc"?n.sort((u,f)=>r(
u,f)||l(u,f)):t==="oldest"?n.sort((u,f)=>c(u,f)||a(u,f)):n.sort((u,f)=>l(u,f)||a(u,f)),n}o(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}o(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}o(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}o(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}o(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const f=t.querySelector(".lib-empty-state");f&&f.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const f=get("lib-total-count");f&&(f.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),r=i.filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(
a)),l=get("lib-total-count");if(l){const f=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?l.innerText=`${lib.files.length} / ${f} files`:l.innerText=`${f} files`}if(!r.length){
if(n)return;const f=lib.favoritesOnly&&!a?"fa-star":"fa-search",g=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${f}"></i></div>\
<p class="lib-empty-title">${g}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(a)):r).forEach(f=>{
try{const g=renderLibraryItem(f,c++);t.appendChild(g)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(u=>fileNameForSearch(u).includes(n)):t).filter(u=>u.type==="image"),r=lib.favoritesOnly?
a.filter(u=>u.is_favorite):a;if(!r.length)return;const l=r.map(u=>({url:u.url,filename:u.filename||u.
original_filename||u.url.split("/").pop(),element:null}));let c=l.findIndex(u=>u.url===e.url);c===-1&&
(c=0),openViewerWithItems(l,c)}o(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}o(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,a=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),r=e.type==="\
image"?`<img src="${escapeHtml(i)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(a)}"></i></div><span class="lib-file-badge">${escapeHtml(a?a.toUpperCase():"FI\
LE")}</span></div>`,l=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",u=e.is_favorite?"fas fa-star":"far f\
a-star",f=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
g=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${f}\
" aria-label="${f}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${u}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${g}${y}`,n.
onclick=()=>{lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-se\
lected")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(_=>{_.onclick=C=>{C.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const v=n.querySelector(".lib-del-btn");v&&(v.onclick=async _=>{_.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const k=n.querySelector(".lib-favorite-btn");return k&&
(k.onclick=async _=>{_.stopPropagation(),k.disabled=!0;try{const C=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
L=await C.json().catch(()=>({}));if(!C.ok||typeof L.is_favorite!="boolean")throw new Error(L.error||
"favorite update failed");e.is_favorite=L.is_favorite,renderLibraryGrid(),showToast(L.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),k.disabled=!1}}),n}o(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}o(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",a=FILE_BASE_URL+e,r=i==="image"?FILE_THUMB_BASE_URL+
e:null,l={filename:t,original_filename:t,filepath:e,url:a,thumbnail_url:r,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(l),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
o(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(r=>r.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const a=(i||"").trim();if(!a){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const r=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:a})}),l=await r.json().catch(
()=>({}));if(!r.ok){showToast(l.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=l.filename||a,setAttachmentNameForPath(e,t.filename));const c=get(
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(u=>{u.getAttribute("data-filename")===
e&&setRowAttachmentName(u,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}o(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,t){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),t&&t.parentNode&&t.remove(),lib.files&&(lib.files=
lib.files.filter(n=>n.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}o(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}o(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],t=(lib.files||[]).find(a=>a&&a.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const a=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);a.searchParams.set("filepath",e);const r=await apiFetch(
a.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),l=await r.json().catch(()=>({}));
if(!r.ok)throw new Error(l.error||`HTTP ${r.status}`);const c=Array.isArray(l.chats)?l.chats:[];if(!c.
length){i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(i.innerHTML="",c.forEach(u=>{const f=document.createElement("div");f.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const g=u.updated_at?new Date(u.
updated_at).toLocaleString():"";f.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(u.title||"")}">${escapeHtml(u.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(g)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=f.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(u.id))}),i.appendChild(f)}),l.has_more){const u=document.
createElement("p");u.className="text-[11px] text-gray-500 text-center pt-2",u.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(u)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const u=getLibSortOrder(),f=getLibSearchQuery(),g=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(g),sort:u,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),w=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!w.ok)throw new Error("HTTP "+w.status);r=await w.json(),l=!0}catch(u){i=u}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const u=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,g=new Set(c.map(w=>w&&w.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(w=>{if(c.length>=LIBRARY_PAGE_SIZE||!w||g.has(w))return;const v=getAttachmentNameForPath(
w)||w.split("/").pop()||w,k=(v.split(".").pop()||"").toLowerCase(),_=["png","jpg","jpeg","webp","gif"].
includes(k)?"image":"file",C=_==="image"?f+w:null;c.unshift({filename:v,original_filename:v,filepath:w,
url:u+w,thumbnail_url:C,type:_,ext:k,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),g.add(w)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const u=c.filter(g=>g&&g.filepath&&
g.url);let f=[];if(e){const g=new Set(lib.files.map(y=>y.filepath));f=u.filter(y=>!g.has(y.filepath)),
lib.files.push(...f)}else lib.files=u;lib.files.forEach(g=>{g&&g.filepath&&setAttachmentNameForPath(
g.filepath,g.filename||g.original_filename||"")}),lib.fileSet=new Set(lib.files.map(g=>g.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
f:null)}catch(u){i=i||u}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}o(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}o(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(a=>{const r=isAudioPath(a),
l=isVideoPath(a);if(r&&!e.audio||l&&!e.video){r&&(t+=1),l&&(n+=1);return}const c=normalizeAttachmentPath(
a);if(!c)return;const u=(lib.files||[]).find(f=>f&&f.filepath===a);u&&u.filename&&setAttachmentNameForPath(
c,u.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const a=[];t&&a.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&a.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${a.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}o(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(t=>{const n=(lib.
files||[]).find(i=>i&&i.filepath===t);if(n&&n.url){const i=document.createElement("a");i.href=n.url,
i.download=n.filename||n.original_filename||t.split("/").pop()||"file",document.body.appendChild(i),
i.click(),document.body.removeChild(i)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}o(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const t=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=t,showModal("legal-modal");const n=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!n.ok)return;const i=await n.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
i)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,t)=>{const n=decodeURIComponent(t),i=o(()=>{const a=e.getAttribute("data-copy")||"";e.innerHTML=a===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},a=>{console.error(a),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(i,2e3)})},window.copyMessage=(e,t)=>{const n=messageStore[e]||
"";copyToClipboard(n,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i \
class="fas fa-copy"></i>',2e3)},i=>{console.error(i),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(
()=>t.innerHTML='<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;
t.classList.contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}o(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}o(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let t=0,n=e;const i={};for((allMessages||[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n];t+=a.tokens||
Number(a.tokens_in||0)+Number(a.tokens_out||0),n=a.parent_id}return t}o(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n],r=a.model||"Unknown";t[r]||(t[r]={total:0,in:0,out:0,
thought:0});const l=a.tokens||Number(a.tokens_in||0)+Number(a.tokens_out||0);t[r].total+=l,t[r].in+=
Number(a.tokens_in||0),t[r].out+=Number(a.tokens_out||0),t[r].thought+=Number(a.tokens_thought||0),n=
a.parent_id}return t}o(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(a=>t[a.id]={...a,children:[]}),
allMessages.forEach(a=>{a.parent_id&&t[a.parent_id]?t[a.parent_id].children.push(t[a.id]):a.parent_id||
n.push(t[a.id])});function i(a){const r=document.createElement("div");r.className="flex flex-col ite\
ms-center mt-4";const l=document.createElement("div"),c=String(a.id)===String(currentLeafId),u=a.id===
threadFixedBranchId,f=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),g=getCumulativeTokensForNode(
a.id);if(l.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,l.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(f)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(a.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${g}</span>
                    </div>
                    ${u?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${c?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,l.onclick=y=>{y.stopPropagation(),selectedBranchNodeId=a.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},r.appendChild(l),a.children.length>0){const y=document.createElement("div");
y.className="w-px h-4 bg-gray-700",r.appendChild(y);const w=document.createElement("div");w.className=
"flex gap-4 items-start",a.children.forEach(v=>w.appendChild(i(v))),r.appendChild(w)}return r}o(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}o(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}o(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(u=>u.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const r=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);r.innerHTML="",Object.entries(l).sort((u,f)=>f[1].
total-u[1].total).forEach(([u,f])=>{const g=document.createElement("div");g.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",g.innerHTML=`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${u}</span>
                        <span class="text-blue-400 shrink-0">${f.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${f.in}</div>
                        <div title="Output tokens">Out: ${f.out}</div>
                        <div title="Thought/Reasoning tokens">${f.thought>0?`Th: ${f.thought}`:""}</\
div>
                    </div>
                `,r.appendChild(g)}),get("br-name-input").value=branchLabelNames[n.id]||"";const c=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(c.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
c.classList.replace("bg-amber-600","bg-gray-600")):(c.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
c.classList.replace("bg-gray-600","bg-amber-600"))}o(updateBranchDetailPane,"updateBranchDetailPane"),
get("branch-manage-btn")&&(get("branch-manage-btn").onclick=showBranchModal),get("br-save-name-btn").
onclick=()=>{if(!selectedBranchNodeId)return;const e=get("br-name-input").value.trim();e?branchLabelNames[selectedBranchNodeId]=
e:delete branchLabelNames[selectedBranchNodeId],saveBranchData(),renderBranchTreeVisualization(),showToast(
"\u540D\u524D\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F")},get("br-switch-btn").onclick=()=>{selectedBranchNodeId&&
(switchVersion(selectedBranchNodeId),window.closeBranchModal(),showToast("\u30D6\u30E9\u30F3\u30C1\u3092\u5207\u308A\u66FF\u3048\u307E\u3057\u305F"))},
get("br-fix-btn").onclick=()=>{selectedBranchNodeId&&(threadFixedBranchId===selectedBranchNodeId?(threadFixedBranchId=
null,showToast("\u56FA\u5B9A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F")):(threadFixedBranchId=selectedBranchNodeId,
showToast("\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A\u3057\u307E\u3057\u305F")),saveBranchData(),
renderBranchTreeVisualization(),updateBranchDetailPane())},get("br-delete-btn").onclick=()=>{selectedBranchNodeId&&
confirm("\u3053\u306E\u30D6\u30E9\u30F3\u30C1\u3092\u524A\u9664\u3057\u3066\u3082\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F\uFF08\u305D\u306E\u5F8C\u306E\u5168\u3066\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u3082\u524A\u9664\u3055\u308C\u307E\u3059\uFF09")&&
(deleteMessage(selectedBranchNodeId,!0),selectedBranchNodeId=null,setTimeout(()=>{renderBranchTreeVisualization(),
updateBranchDetailPane()},500))};let batchJobsCache=[],batchFilterMode="all",batchListTimer=null;function batchProviderLabel(e){
return{gemini:"Gemini",openai:"OpenAI",xai:"xAI"}[String(e||"").toLowerCase()]||e||"Batch"}o(batchProviderLabel,
"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\u9001\u4FE1\u5F85\u3061",
JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",JOB_STATE_RUNNING:"\
\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\u5B8C\u4E86",
JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\u505C\u6B62",
JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}o(batchStateLabelShort,
"batchStateLabelShort");function batchStateTone(e){const t=String(e||"").toUpperCase();return t==="J\
OB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":t==="JOB_STATE_FAILED"?
"border-red-500/40 bg-red-900/20 text-red-200":t==="JOB_STATE_CANCELLED"||t==="JOB_STATE_EXPIRED"?"b\
order-gray-500/40 bg-gray-700/30 text-gray-300":t==="JOB_STATE_CANCELLING"?"border-amber-500/40 bg-a\
mber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}o(batchStateTone,
"batchStateTone");function batchFormatTime(e){if(!e)return"";let t=String(e);!/[zZ]$/.test(t)&&!/[+-]\d\d:?\d\d$/.
test(t)&&(t+="Z");const n=new Date(t);return isNaN(n.getTime())?String(e):n.toLocaleString("ja-JP",{
month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}o(batchFormatTime,"batchFormatTime");
function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove("batch-list-enter"),
e.offsetWidth,e.classList.add("batch-list-enter"))}o(playBatchListAnimation,"playBatchListAnimation");
function renderBatchJobs(e={}){const t=get("batch-list");if(!t)return;const n=batchJobsCache.filter(
a=>batchFilterMode==="active"?!!a.is_active:batchFilterMode==="done"?!a.is_active:!0),i=get("batch-c\
ount");if(i&&(i.textContent=`${n.length}\u4EF6`),t.innerHTML="",!n.length){t.innerHTML='<div class="\
batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}n.forEach(a=>{const r=document.createElement("div");r.className=
"batch-job-card";const l=escapeHtml(a.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),c=escapeHtml(
batchProviderLabel(a.provider)),u=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
g=escapeHtml(a.status_text||""),y=batchStateTone(a.state),w=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",v=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",k=a.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';r.innerHTML=`
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${l}">${l}\
</div>
                            <div class="batch-job-meta mt-1 flex flex-wrap items-center gap-2 text-[\
10px]">
                                <span class="inline-flex items-center gap-1"><i class="fas fa-layer-\
group"></i>${c}</span>
                                <span class="truncate max-w-[16rem]">${u}</span>
                                <span><i class="fas fa-history mr-1"></i>${f}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${y}">${escapeHtml(batchStateLabelShort(
a.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${g}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${w}${v}${k}
                    </div>`;const _=r.querySelector("[data-batch-open]");_&&(_.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const C=r.querySelector("[data-batch-cancel]");C&&(C.onclick=()=>cancelBatchJob(
a));const L=r.querySelector("[data-batch-delete]");L&&(L.onclick=()=>deleteBatchJob(a)),t.appendChild(
r)}),e.animate&&playBatchListAnimation()}o(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const t=await apiFetch("/api/batch/jobs");if(!t.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const n=await t.json().catch(()=>({}));batchJobsCache=Array.isArray(n.jobs)?n.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}o(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),n=await t.json().catch(()=>({}));if(!t.ok){showToast(n.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}o(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),n=await t.json().catch(()=>({}));
if(!t.ok){showToast(n.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}o(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(t=>{t.classList.toggle("is-active",t===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=o(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const a=i?get(i.inputId):null;get("api-key-modal-input").
value=a?a.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const r=get(
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),u=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=o(g=>{g.key==="Enter"&&(g.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),r.onclick=async()=>{
const g=get("api-key-modal-input").value.trim();if(!g){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=g);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:g})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=g)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",f),u(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",f),u(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",f),u(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const g=get("api-key-modal-input");g&&g.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const u=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,u)}catch{}finally{
a=!1}}o(r,"sendToServer"),console.log=function(...l){e.apply(console,l),r("log",l)},console.error=function(...l){
t.apply(console,l),r("error",l)},console.warn=function(...l){n.apply(console,l),r("warn",l)},console.
info=function(...l){i.apply(console,l),r("info",l)},window.addEventListener("error",function(l){r("e\
xception",[l.message,l.filename,l.lineno,l.colno,l.error])}),window.addEventListener("unhandledrejec\
tion",function(l){r("promise-rejection",[l.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();
