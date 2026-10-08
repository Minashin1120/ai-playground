var Ci=Object.defineProperty;var o=(e,t)=>Ci(e,"name",{value:t,configurable:!0});const get=o(e=>document.getElementById(e),"get"),nativeConsoleLog=typeof console.log=="function"?console.
log.bind(console):function(){},nativeConsoleInfo=typeof console.info=="function"?console.info.bind(console):
nativeConsoleLog;let settingsModalLoaded=!1;const setSettingsSaveEnabled=o(e=>{const t=get("save-set\
tings-btn");t&&(t.disabled=!e,t.classList.toggle("opacity-60",!e),t.classList.toggle("cursor-not-all\
owed",!e),t.setAttribute("title",e?"":"\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u5B8C\u4E86\u5F8C\u306B\u4FDD\u5B58\u3067\u304D\u307E\u3059"))},
"setSettingsSaveEnabled");(function(){const t=o(l=>/(\/files\/thumb\/|\/files\/)/.test(String(l||"")),
"isFileUrl"),n=o(l=>fetch(l,{method:"GET",headers:{Range:"bytes=0-0"},cache:"no-store"}).then(c=>c.status).
catch(()=>-1),"fileUrlStatus");document.addEventListener("load",l=>{const c=l.target;if(!c||c.tagName!==
"IMG"||!c.classList.contains("chat-image"))return;const m=c.closest(".chat-image-frame");m&&(m.dataset.
chatImageState="loaded",m.removeAttribute("aria-busy"))},!0);const i=o((l,c)=>{const m=document.createElement(
"div");return m.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",c?m.innerHTML=
'<i class="fas fa-key" style="font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd\
34d;font-weight:700;line-height:1.3">\u6697\u53F7\u30AD\u30FC\u304C\u4E00\u81F4\u3057\u306A\u3044\u305F\u3081<br>\u95B2\u89A7\u3067\u304D\u307E\u305B\u3093</div>':
m.innerHTML='<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9\
px;color:#9ca3af;font-weight:700">\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093</div>',
l&&m.setAttribute("data-file-name",String(l)),m},"buildWarning"),a=o(l=>{const c=document.createElement(
"div");return c.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",c.innerHTML='\
<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;\
color:#bfdbfe;font-weight:700;line-height:1.3">\u4E00\u6642\u7684\u306B\u6DF7\u96D1\u3057\u3066\u3044\u307E\u3059<br>\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u304F\u3060\u3055\u3044</div>',
l&&c.setAttribute("data-file-name",String(l)),c},"buildBusyWarning"),r=o(l=>String(l||"").split("?")[0].
replace("/files/thumb/","/files/"),"fullFileUrl");document.addEventListener("error",l=>{const c=l.target;
if(!c||c.tagName!=="IMG")return;const m=c.currentSrc||c.src||"";if(!t(m)){const _=c.closest&&c.closest(
".chat-image-frame");if(_){_.dataset.chatImageState="error",_.removeAttribute("aria-busy");const C=_.
querySelector(".chat-image-loading"),L=C&&C.querySelector("span");L&&(L.textContent="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F");
const $=C&&C.querySelector("i");$&&($.className="fas fa-image")}return}l.stopImmediatePropagation(),
l.preventDefault();const f=String(m).split("?")[0],b=c.getAttribute("data-viewer-filename")||f.split(
"/").pop(),y=o(_=>{const C=c.closest&&c.closest(".chat-image-frame");C&&(C.dataset.chatImageState="f\
ailed",C.removeAttribute("aria-busy"));const L=i(b,!!_);try{c.replaceWith(L)}catch{}},"showWarning"),
v=o(()=>{const _=c.closest&&c.closest(".chat-image-frame");_&&(_.dataset.chatImageState="busy",_.removeAttribute(
"aria-busy"));const C=a(b);try{c.replaceWith(C)}catch{}},"showBusyWarning"),w=o((_,C)=>{const L=c.cloneNode(
!1);L.setAttribute("data-file-retry",String(C));const $=_+(_.includes("?")?"&":"?")+"retry="+Date.now()+
"_"+C;L.setAttribute("src",$);try{c.replaceWith(L)}catch{}},"retryLoad"),k=o(_=>{if(_===429||_===503){
v();return}if(_===409){y(!0);return}if(_===404||_===410||_===403){y(!1);return}const C=parseInt(c.getAttribute&&
c.getAttribute("data-file-retry")||"0",10);if(C<2){w(m,C+1);return}if(m.includes("/files/thumb/")&&!c.
getAttribute("data-file-fallback")){c.setAttribute("data-file-fallback","1"),w(r(m),0);return}y(!1)},
"handleStatus");n(m).then(k).catch(()=>{const _=parseInt(c.getAttribute&&c.getAttribute("data-file-r\
etry")||"0",10);if(_<2){w(m,_+1);return}if(m.includes("/files/thumb/")){v();return}y(!1)})},!0)})();
function buildChatImageHtml(e,t={}){const n=String(e||""),i=String(t.alt||""),a=String(t.title||""),
r=String(t.viewerSrc||n),l=String(t.filename||"");if(n.startsWith("sandbox:"))return`<span class="te\
xt-xs text-gray-500" title="${escapeHtml(n)}">${escapeHtml(i)||"\uFF08\u753B\u50CF\u30C7\u30FC\u30BF\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09"}\
</span>`;const c=a?` title="${escapeHtml(a)}"`:"",m=l?` data-viewer-filename="${escapeHtml(l)}"`:"";
return`<span class="chat-image-frame" data-chat-image-state="loading" aria-busy="true"><span class="\
chat-image-loading" role="status" aria-label="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D"><i class="fas fa-spinner fa-spin" aria-hidde\
n="true"></i><span>\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</span></span><img src="${escapeHtml(
n)}" data-viewer-src="${escapeHtml(r)}" alt="${escapeHtml(i)}"${c}${m} class="chat-image" loading="l\
azy" decoding="async" width="320" height="320"></span>`}o(buildChatImageHtml,"buildChatImageHtml");const isAdminSidebarDebugEnabled=o(
()=>{try{const e=window.CHAT_CONFIG||{};return!!(e.botConfig&&e.botConfig.isAdmin)}catch{return!1}},
"isAdminSidebarDebugEnabled"),ADMIN_SIDEBAR_DEBUG_PREFIX="[admin-sidebar]",adminSidebarDebugEntries=[],
snapshotSidebarHistory=o(e=>{if(!isAdminSidebarDebugEnabled())return null;const t=get("thread-list"),
n=get("sidebar"),i=get("settings-modal"),a=get("history-modal"),r=t?window.getComputedStyle(t):null,
l=n?window.getComputedStyle(n):null,c=t?Array.from(t.querySelectorAll("[data-thread-id]")):[],m=c[0]||
null,f=m?window.getComputedStyle(m):null;let b=null;try{b=typeof threadLoading=="boolean"?threadLoading:
null}catch{b=null}const y={t:Date.now(),reason:String(e||""),path:location.pathname,vw:window.innerWidth,
liteHtml:document.documentElement.classList.contains("performance-lite-mode"),blurHtml:document.documentElement.
classList.contains("performance-blur-disabled"),liquidBody:!!(document.body&&document.body.classList.
contains("liquid-glass-mode")),blurMode:adaptiveBlurPreferenceMode,liteEnabled:adaptiveBlurLiteEnabled,
sidebarClass:n?n.className:null,sidebarDisplay:l?l.display:null,sidebarOpacity:l?l.opacity:null,sidebarVisibility:l?
l.visibility:null,compact:!!(n&&n.classList.contains("compact")),sidebarOpen:!!(n&&n.classList.contains(
"open")),listExists:!!t,listParent:t&&t.parentElement?t.parentElement.id||t.parentElement.className:
null,listClass:t?t.className:null,listChildCount:t?t.children.length:0,listItemCount:c.length,listDisplay:r?
r.display:null,listOpacity:r?r.opacity:null,listVisibility:r?r.visibility:null,listHeight:r?r.height:
null,hideCompact:!!(t&&t.classList.contains("hide-compact")),searchLen:(()=>{const v=get("search-box");
return v?String(v.value||"").length:0})(),firstItemText:m&&m.textContent?m.textContent.trim().slice(
0,40):null,firstItemOpacity:f?f.opacity:null,firstItemDisplay:f?f.display:null,firstItemVisibility:f?
f.visibility:null,firstItemClass:m?m.className:null,settingsHidden:i?i.classList.contains("hidden"):
null,settingsOpen:i?i.classList.contains("modal-open"):null,settingsDisplay:i&&i.style.display||null,
historyHidden:a?a.classList.contains("hidden"):null,threadLoading:b};adminSidebarDebugEntries.push(y),
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
!1;const r=[...t].sort((y,v)=>y-v),l=Math.min(17.5,Math.max(7,r[Math.floor(r.length*.2)])),c=Math.max(
28,l*1.75),m=Math.max(44,l*2.7),f=t.filter(y=>y>=c).length,b=t.filter(y=>y>=m).length;(f>=5||f>=4&&b>=
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
ght","start","type","reversed"],FORBID_TAGS:["iframe","object","embed"]});const AUDIO_PLAYER_SPEEDS=[
1,1.25,1.5,2,.75],formatAudioPlayerTime=o(e=>{const t=Number.isFinite(e)&&e>0?Math.floor(e):0;return`${Math.
floor(t/60)}:${String(t%60).padStart(2,"0")}`},"formatAudioPlayerTime"),upgradeAudioPlayer=o(e=>{if(!e||
e.dataset.aipReady==="1"||!e.hasAttribute("controls"))return;const t=e.parentNode;if(!t)return;e.dataset.
aipReady="1",e.removeAttribute("controls"),e.getAttribute("preload")||(e.preload="metadata");const n=document.
createElement("div");n.className="aip",n.setAttribute("role","group"),n.setAttribute("aria-label","\u97F3\
\u58F0\u30D7\u30EC\u30A4\u30E4\u30FC");const i=document.createElement("button");i.type="button",i.className=
"aip-toggle";const a=document.createElement("i");i.appendChild(a);const r=document.createElement("sp\
an");r.className="aip-time";const l=document.createElement("input");l.type="range",l.className="aip-\
seek",l.min="0",l.max="1000",l.step="1",l.value="0",l.setAttribute("aria-label","\u518D\u751F\u4F4D\u7F6E");
const c=document.createElement("button");c.type="button",c.className="aip-speed",c.setAttribute("ari\
a-label","\u518D\u751F\u901F\u5EA6");const m=document.createElement("span");m.className="aip-failed",
m.textContent="\u97F3\u58F0\u3092\u518D\u751F\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
n.append(i,r,l,c,m),t.insertBefore(n,e),n.appendChild(e);let f=!1;const b=o(()=>Number.isFinite(e.duration)&&
e.duration>0?e.duration:0,"duration"),y=o(()=>{const k=b(),_=f?Number(l.value)/1e3*k:e.currentTime;r.
textContent=`${formatAudioPlayerTime(_)} / ${formatAudioPlayerTime(k)}`,f||(l.value=k?String(Math.min(
1e3,e.currentTime/k*1e3)):"0"),l.disabled=!k,l.style.setProperty("--aip-progress",`${Number(l.value)/
10}%`)},"refresh"),v=o(()=>{const k=!e.paused&&!e.ended;a.className=n.classList.contains("is-loading")?
"fas fa-spinner fa-spin":k?"fas fa-pause":"fas fa-play";const _=k?"\u4E00\u6642\u505C\u6B62":"\u518D\u751F";
i.setAttribute("aria-label",_),i.title=_,c.textContent=`${e.playbackRate}x`},"syncState"),w=o(k=>{n.
classList.toggle("is-loading",k),v()},"setLoading");i.addEventListener("click",()=>{if(n.classList.contains(
"is-failed")&&(n.classList.remove("is-failed"),e.load()),e.paused||e.ended){e.readyState<3&&w(!0);const k=e.
play();k&&typeof k.catch=="function"&&k.catch(()=>{w(!1)})}else e.pause()}),c.addEventListener("clic\
k",()=>{const k=AUDIO_PLAYER_SPEEDS.indexOf(e.playbackRate);e.playbackRate=AUDIO_PLAYER_SPEEDS[(k+1)%
AUDIO_PLAYER_SPEEDS.length]}),l.addEventListener("input",()=>{f=!0,y()}),l.addEventListener("change",
()=>{const k=b();k&&(e.currentTime=Number(l.value)/1e3*k),f=!1,y()}),["timeupdate","durationchange",
"loadedmetadata","seeked","emptied"].forEach(k=>e.addEventListener(k,y)),["play","pause","ended","ra\
techange"].forEach(k=>e.addEventListener(k,v)),e.addEventListener("waiting",()=>w(!0)),["playing","p\
ause","canplay","ended"].forEach(k=>e.addEventListener(k,()=>w(!1))),e.addEventListener("loadeddata",
()=>n.classList.remove("is-failed")),e.addEventListener("error",()=>{n.classList.add("is-failed"),w(
!1)}),y(),v()},"upgradeAudioPlayer"),scanAudioPlayers=o(e=>{!e||e.nodeType!==1||(e.tagName==="AUDIO"?
upgradeAudioPlayer(e):e.querySelectorAll&&e.querySelectorAll("audio[controls]").forEach(upgradeAudioPlayer))},
"scanAudioPlayers"),initAudioPlayers=o(()=>{try{scanAudioPlayers(document.body),new MutationObserver(
e=>{e.forEach(t=>t.addedNodes.forEach(scanAudioPlayers))}).observe(document.body,{childList:!0,subtree:!0})}catch{}},
"initAudioPlayers");document.readyState==="loading"?document.addEventListener("DOMContentLoaded",initAudioPlayers,
{once:!0}):initAudioPlayers();const THEME_DEFAULT="#0dd4bf",THEME_STORAGE_KEY="theme_color",INITIAL_THEME_COLOR=window.
CHAT_CONFIG&&window.CHAT_CONFIG.initialThemeColor||null,INITIAL_LIGHT_MODE_ENABLED=!!(window.CHAT_CONFIG&&
window.CHAT_CONFIG.initialLightModeEnabled),INITIAL_LIQUID_GLASS_ENABLED=!!(window.CHAT_CONFIG&&window.
CHAT_CONFIG.initialLiquidGlassEnabled),RICH_PASTE_DEFAULT_PROMPT="\u3053\u306EPDF\u3092Markdown\u5F62\u5F0F\u306B\u5909\u63DB\u3057\u3001\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u306B\u66F8\u304D\u51FA\u3057\u3066\
\u304F\u3060\u3055\u3044\u3002",GEMINI_LOCAL_PY_DIALOG_KEY="gemini_local_py_dialog_enabled",COMPRESSION_SIZE_KEY="\
compression_max_size_mb",COMPRESSION_DIM_KEY="compression_max_dim",COMPRESSION_TYPE_KEY="compression\
_output_type",COMPRESSION_FORMAT_ONLY_KEY="compression_format_only",getCompressionMaxSizeMB=o(()=>parseFloat(
localStorage.getItem(COMPRESSION_SIZE_KEY)||"1.0"),"getCompressionMaxSizeMB"),getCompressionMaxDim=o(
()=>parseInt(localStorage.getItem(COMPRESSION_DIM_KEY)||"1920"),"getCompressionMaxDim"),getCompressionOutputType=o(
()=>localStorage.getItem(COMPRESSION_TYPE_KEY)||"original","getCompressionOutputType"),getCompressionFormatOnly=o(
()=>localStorage.getItem(COMPRESSION_FORMAT_ONLY_KEY)==="true","getCompressionFormatOnly"),IMAGE_EXTENSION_BY_MIME={
"image/jpeg":".jpg","image/png":".png","image/webp":".webp"},imageFilenameForMime=o((e,t)=>{const n=IMAGE_EXTENSION_BY_MIME[String(
t||"").toLowerCase()];return n?`${String(e||"image").replace(/\.[^./\\]+$/,"")||"image"}${n}`:e||"im\
age"},"imageFilenameForMime"),convertImageFormatOnly=o(async(e,t)=>{if(!e||!t||t==="original"||t===e.
type)return e;await ensureImageCompression();const n=await window.imageCompression.drawFileInCanvas(
e,{fileType:t}),i=n&&n[0],a=n&&n[1];if(!a)throw new Error("Image conversion canvas is unavailable");
let r;try{typeof a.convertToBlob=="function"?r=await a.convertToBlob({type:t,quality:1}):r=await new Promise(
(l,c)=>{a.toBlob(m=>m?l(m):c(new Error("Image conversion failed")),t,1)})}finally{try{window.imageCompression.
cleanupCanvasMemory(a)}catch{}try{i&&typeof i.close=="function"&&i.close()}catch{}}return new File([
r],imageFilenameForMime(e.name,t),{type:t,lastModified:e.lastModified||Date.now()})},"convertImageFo\
rmatOnly"),setCompressionSettings=o((e,t,n,i)=>{localStorage.setItem(COMPRESSION_SIZE_KEY,e),localStorage.
setItem(COMPRESSION_DIM_KEY,t),localStorage.setItem(COMPRESSION_TYPE_KEY,n),localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY,
i)},"setCompressionSettings"),syncCompressionSettingsUi=o(()=>{const e=get("compression-max-size"),t=get(
"compression-max-dim"),n=get("compression-output-type"),i=get("compression-format-only");if(e&&(e.value=
getCompressionMaxSizeMB()),t&&(t.value=getCompressionMaxDim()),n&&(n.value=getCompressionOutputType()),
i){i.checked=getCompressionFormatOnly();const b=i.checked;e&&(e.disabled=b),t&&(t.disabled=b);const y=get(
"compression-size-wrap"),v=get("compression-dim-wrap");y&&(y.style.opacity=b?"0.4":"1"),v&&(v.style.
opacity=b?"0.4":"1")}const a=o((b,y)=>{get(b)&&get(y)&&(get(y).value=get(b).value)},"sync");a("gpt-i\
mage-size","modal-gpt-image-size"),a("gpt-image-quality","modal-gpt-image-quality"),a("gpt-image-for\
mat","modal-gpt-image-format"),a("gpt-image-compression","modal-gpt-image-compression"),a("gemini-im\
age-aspect","modal-gemini-image-aspect"),a("gemini-image-size","modal-gemini-image-size"),a("grok-im\
age-aspect","modal-grok-image-aspect"),a("grok-image-resolution","modal-grok-image-resolution"),a("g\
rok-image-quality","modal-grok-image-quality"),IDEOGRAM_IMAGE_FIELDS.forEach(b=>a(`ideogram-image-${b}`,
`modal-ideogram-image-${b}`)),a("ocr-table-format","modal-ocr-table-format"),a("ocr-pages","modal-oc\
r-pages");const r=o((b,y)=>{get(b)&&get(y)&&(get(y).checked=get(b).checked)},"syncChk");r("ocr-extra\
ct-header","modal-ocr-extract-header"),r("ocr-extract-footer","modal-ocr-extract-footer"),r("ocr-inc\
lude-blocks","modal-ocr-include-blocks"),r("ocr-include-images","modal-ocr-include-images");const l=get(
"model-select").value,c=isGptImageModel(l),m=isGeminiImageModel(l),f=isGrokImageModel(l);get("modal-\
gpt-image-options")&&get("modal-gpt-image-options").classList.toggle("hidden",!c),get("modal-gemini-\
image-options")&&get("modal-gemini-image-options").classList.toggle("hidden",!m),get("modal-grok-ima\
ge-options")&&get("modal-grok-image-options").classList.toggle("hidden",!f),get("modal-ideogram-imag\
e-options")&&get("modal-ideogram-image-options").classList.toggle("hidden",!isIdeogramModel(l)),typeof updateIdeogramImageUi==
"function"&&updateIdeogramImageUi(),get("modal-mistral-ocr-options")&&get("modal-mistral-ocr-options").
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
c)&&c>=0)e.totalBytes+=c;else{const m=await l.clone().blob();e.totalBytes+=m.size||0}}catch{}}}catch{}
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
hot"),isUserSystemPromptActive=o(e=>!!(e&&String(e.system_prompt||"").trim()&&e.system_prompt_enabled!==
!1),"isUserSystemPromptActive");window.applySavedUserSystemPromptSettings=e=>{if(!e)return;const t=userSettingsSnapshot||
null,n=Object.assign({},t||{},e);cacheUserSettings(n);try{const i=document.getElementById("enable-sy\
s-prompt"),a=!t||String(t.system_prompt||"")!==String(n.system_prompt||"")||t.system_prompt_enabled!==
!1!=(n.system_prompt_enabled!==!1);if(i&&a){const r=isUserSystemPromptActive(n);i.disabled?r?i.dataset.
restoreChecked="1":delete i.dataset.restoreChecked:i.checked!==r&&(i.checked=r,i.dispatchEvent(new Event(
"change",{bubbles:!0})))}}catch{}fetchSettingsSnapshot().catch(()=>{})};const saveRichPastePromptPreferences=o(
async()=>{const e=getRichPastePrompt(),t=getRichPasteUseDefaultCheckbox();if(!e||!t)return;const n={
rich_paste_prompt_default:e.value||"",rich_paste_prompt_use_custom_default:!!t.checked};try{await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(n)}),cacheUserSettings(Object.assign({},userSettingsSnapshot||{},n),{preservePrompt:!0})}catch{}},
"saveRichPastePromptPreferences"),queueRichPastePromptPreferenceSave=o(()=>{richPastePromptSaveTimer&&
clearTimeout(richPastePromptSaveTimer),richPastePromptSaveTimer=setTimeout(()=>{richPastePromptSaveTimer=
null,saveRichPastePromptPreferences()},500)},"queueRichPastePromptPreferenceSave"),hasRichPasteContent=o(
()=>{const e=getRichPasteEditor();return e?(e.textContent||"").trim()?!0:!!e.querySelector("img,tabl\
e,ul,ol,blockquote,h1,h2,h3,h4,h5,h6,pre,code"):!1},"hasRichPasteContent"),updateRichPasteStatus=o(()=>{
const e=getRichPasteEditor(),t=getRichPasteStatus();if(!t||!e)return;const n=(e.innerText||"").trim();
if(!n){t.textContent="\u307E\u3060\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002";return}const i=e.
querySelectorAll("img").length,a=e.querySelectorAll("table").length,r=e.querySelectorAll("a").length,
l=e.querySelectorAll("h1,h2,h3,h4,h5,h6").length;t.textContent=`${n.length} \u6587\u5B57 / \u753B\u50CF ${i}\
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
from(a.querySelectorAll("*")).slice(0,5e3)],l=[],c=new Map;let m=0;const f=o(k=>Array.from(k.childNodes||
[]).reduce((_,C)=>C&&C.nodeType===Node.TEXT_NODE?_+String(C.textContent||"").replace(/\s+/g," ").trim().
length:_,0),"directTextLength");r.forEach(k=>{if(!k||k===a||!k.style)return;const _=window.getComputedStyle(
k),C=f(k);if(C>0){const $=parseRichPasteCssColor(_.color);if($&&$.a>=.5){const F=richPasteColorCss($),
X=c.get(F)||{color:$,weight:0};X.weight+=C,c.set(F,X),m+=C}}if(!!(String(k.style.backgroundColor||"").
trim()||String(k.style.background||"").trim())){const $=parseRichPasteCssColor(_.backgroundColor);if($&&
$.a>=.72){const F=String(k.textContent||"").replace(/\s+/g," ").trim().length;l.push({color:$,weight:Math.
max(1,F)})}}});const b=Array.from(c.values()).sort((k,_)=>_.weight-k.weight),y=b.length?b[0].color:null,
v=b.reduce((k,_)=>k+(richPasteColorLuminance(_.color)>=.6?_.weight:0),0);l.sort((k,_)=>_.weight-k.weight);
let w=l.length?l[0].color:null;return w||(w=m>0&&v/m>=.55?{r:11,g:11,b:12,a:1}:t),makeRichPasteTheme(
w,y||n)}catch{return makeRichPasteTheme(t,n)}finally{a.parentNode&&a.parentNode.removeChild(a)}},"de\
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
from(n.body.querySelectorAll('[role="main"],[role="article"]'))].filter(m=>(m.textContent||"").replace(
/\s+/g," ").trim().length>=i*.65);l.sort((m,f)=>{const b=+!!f.querySelector("h1")-+!!m.querySelector(
"h1");return b||m.querySelectorAll("*").length-f.querySelectorAll("*").length});const c=l[0]||null;return c?
c.outerHTML:n.body.innerHTML},"extractRichPasteArticleHtml"),sanitizeRichPasteHtml=o(e=>{if(!window.
DOMPurify||typeof window.DOMPurify.sanitize!="function"){const a=new DOMParser().parseFromString(String(
e||""),"text/html");return escapeHtml(a.body?a.body.textContent:"")}let t=extractRichPasteArticleHtml(
e),n=window.DOMPurify.sanitize(t||"",{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,
KEEP_CONTENT:!0});if((!n||n.trim()==="")&&e&&e.trim()!==""&&(n=window.DOMPurify.sanitize(e,{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,
ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,KEEP_CONTENT:!0})),!n)return"";const i=document.createElement("\
template");return i.innerHTML=n,normalizeRichPasteTree(i.content),i.innerHTML},"sanitizeRichPasteHtm\
l"),normalizeRichPastePrintHtml=o(e=>{const t=document.createElement("template");t.innerHTML=String(
e||"");const n=Array.from(t.content.querySelectorAll("*")),i=n.reduce((m,f)=>{const b=String(f.style&&
f.style.display||"").trim().toLowerCase();return m+(["flex","inline-flex","grid","inline-grid"].includes(
b)?1:0)},0),a=n.reduce((m,f)=>{if(!f||!f.style||!["article","div","main","section"].includes(String(
f.tagName||"").toLowerCase()))return m;const b=String(f.getAttribute("style")||""),y=Array.from(b.matchAll(
/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)).some(w=>Array.
from(w[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi)).some(k=>Math.abs(Number(k[1])||0)>=96)),v=Array.from(b.
matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)).some(w=>Math.abs(Number(w[1])||
0)>720);return m+(y||v?1:0)},0);if(n.length<=500&&i<=24&&a===0)return t.innerHTML;const r=new Set(["\
align-items","align-self","column-gap","flex","flex-basis","flex-direction","flex-grow","flex-shrink",
"flex-wrap","gap","grid","grid-auto-columns","grid-auto-flow","grid-auto-rows","grid-column","grid-c\
olumn-end","grid-column-start","grid-row","grid-row-end","grid-row-start","grid-template","grid-temp\
late-areas","grid-template-columns","grid-template-rows","justify-content","justify-items","justify-\
self","order","row-gap"]),l=new Set(["article","div","main","section"]),c=new Set(["padding","paddin\
g-left","padding-right","padding-inline","padding-inline-start","padding-inline-end"]);return n.forEach(
m=>{if(!m||!m.style)return;const f=String(m.tagName||"").toLowerCase(),b=[];String(m.getAttribute("s\
tyle")||"").split(";").forEach(y=>{if(!y||y.indexOf(":")<0)return;const v=y.indexOf(":"),w=y.slice(0,
v).trim().toLowerCase();let k=y.slice(v+1).trim();if(!(!w||!k||r.has(w))&&!["height","max-height","m\
in-height","overflow","overflow-x","overflow-y"].includes(w)&&!(["width","min-width"].includes(w)&&l.
has(f))){if(c.has(w)&&l.has(f)&&Array.from(k.matchAll(/(-?\d+(?:\.\d+)?)px/gi)).map(C=>Math.abs(Number(
C[1])||0)).some(C=>C>=96)&&(k="0px"),w==="display"){const _=k.toLowerCase();["flex","grid"].includes(
_)?k="block":["inline-flex","inline-grid"].includes(_)&&(k="inline-block")}b.push(`${w}: ${k}`)}}),b.
length?m.setAttribute("style",b.join("; ")):m.removeAttribute("style")}),t.innerHTML},"normalizeRich\
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
"text/html")){const m=await(await i.getType("text/html")).text();m&&insertHtmlIntoRichPasteEditor(m)&&
(n=!0,r=!0)}if(!r&&a.includes("text/plain")){const m=await(await i.getType("text/plain")).text();m&&
(insertTextIntoRichPasteEditor(m),n=!0)}const l=a.find(c=>c&&c.startsWith("image/"));if(!r&&l){const c=await i.
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
`});return}const m=Object.assign({},l);["b","strong"].includes(c)&&(m.bold=!0),["i","em"].includes(c)&&
(m.italic=!0),c==="a"&&n&&(m.link=String(r.getAttribute("href")||"").trim()),c==="code"&&(m.monospace=
!0),Array.from(r.childNodes||[]).forEach(f=>a(f,m))},"walk");return a(e,{bold:!!t.bold,italic:!!t.italic}),
i},"collectRichPasteInlineSegments"),collectRichPasteInlineText=o((e,t={})=>collectRichPasteInlineSegments(
e,t).map(i=>i.text).join(""),"collectRichPasteInlineText"),collectRichPasteTableRows=o(e=>{const t=[];
return Array.from(e.querySelectorAll("tr")||[]).forEach(n=>{n&&n.closest&&n.closest("table")===e&&t.
push(n)}),t},"collectRichPasteTableRows"),makeRichPasteTableMarkdown=o(e=>{const t=e&&e.querySelector?
e.querySelector("caption"):null,n=t?normalizeRichPastePdfText(collectRichPasteInlineText(t)):"",i=collectRichPasteTableRows(
e).map(m=>Array.from(m.children||[]).filter(b=>{const y=String(b.tagName||"").toLowerCase();return y===
"th"||y==="td"}).map(b=>normalizeRichPastePdfText(collectRichPasteInlineText(b))||" ")).filter(m=>m.
length);if(!i.length)return n||"[table]";const a=i.reduce((m,f)=>Math.max(m,f.length),0),r=i.map(m=>{
const f=m.slice(0,a);for(;f.length<a;)f.push(" ");return f}),l=`| ${Array(a).fill("---").join(" | ")}\
 |`,c=[];n&&(c.push(`Table: ${n}`),c.push("")),c.push(`| ${r[0].join(" | ")} |`),c.push(l);for(let m=1;m<
r.length;m+=1)c.push(`| ${r[m].join(" | ")} |`);return c.join(`
`)},"makeRichPasteTableMarkdown"),collectRichPasteListBlocks=o((e,t=!1,n=0)=>{const i=[],a=Array.from(
e.children||[]).filter(l=>String(l.tagName||"").toLowerCase()==="li");let r=1;return a.forEach(l=>{const c=l.
cloneNode(!0);Array.from(c.querySelectorAll("ul,ol")||[]).forEach(f=>{try{f.remove()}catch{}});const m=collectRichPasteInlineSegments(
c);m.length>0&&i.push({type:"list_item",ordered:t,depth:n,index:r,segments:m}),Array.from(l.children||
[]).forEach(f=>{const b=String(f.tagName||"").toLowerCase();(b==="ul"||b==="ol")&&i.push(...collectRichPasteListBlocks(
f,b==="ol",n+1))}),r+=1}),i},"collectRichPasteListBlocks"),collectRichPastePdfBlocks=o((e,t=0)=>{const n=[];
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
"alt")||f.getAttribute("title")||"").trim(),title:String(f.getAttribute("title")||"").trim()});const b=r.
querySelector("figcaption");if(b){const y=collectRichPasteInlineSegments(b);y.length>0&&n.push({type:"\
paragraph",segments:y})}return}if(l==="img"){a(),n.push({type:"image",src:String(r.getAttribute("src")||
"").trim(),alt:String(r.getAttribute("alt")||r.getAttribute("title")||"").trim(),title:String(r.getAttribute(
"title")||"").trim()});return}if(l==="li"){a(),n.push(...collectRichPasteListBlocks(r,!1,t));return}
if(Array.from(r.children||[]).some(f=>{const b=String(f.tagName||"").toLowerCase();return/^h[1-6]$/.
test(b)||["p","div","section","article","main","blockquote","pre","table","ul","ol","hr","figure","i\
mg","li"].includes(b)})&&["div","section","article","main","figure"].includes(l)){a(),n.push(...collectRichPastePdfBlocks(
r,t+1));return}const m=collectRichPasteInlineSegments(r);m.length>0&&i.push(...m)}),a(),n},"collectR\
ichPastePdfBlocks"),detectImageMimeType=o(e=>{const t=String(e||"").match(/^data:(image\/[a-z0-9.+-]+);/i);
return t?t[1].toLowerCase():"image/png"},"detectImageMimeType"),loadRichPasteImageData=o(async(e,t=3e3)=>{
const n=String(e||"").trim();if(!n)return null;if(n.startsWith("data:image/"))return{dataUrl:n,mimeType:detectImageMimeType(
n)};let i=null;try{i=new URL(n,window.location.href)}catch{return null}if(!(i.origin===window.location.
origin))return null;const r=(async()=>{try{const l=await fetch(i.toString(),{credentials:"same-origi\
n",cache:"force-cache"});if(!l.ok)return null;const c=await l.blob(),m=await blobToDataUrl(c);return{
dataUrl:m,mimeType:c.type||detectImageMimeType(m)}}catch{return null}})();return await Promise.race(
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
rogress-container"),t=get("rich-paste-progress-bar"),n=get("rich-paste-progress-text"),i=o(v=>{const w=Math.
max(0,Math.min(100,Number(v)||0));t&&(t.style.width="100%",t.style.transformOrigin="left center",(!t.
style.transition||t.style.transition.indexOf("transform")===-1)&&(t.style.transition="transform 0.45\
s cubic-bezier(0.22, 1, 0.36, 1)"),t.style.transform=`scaleX(${w/100})`,t.style.willChange="transfor\
m"),n&&(n.innerText=`${Math.round(w)}%`)},"updateProgress");e&&(e.classList.remove("hidden"),e.style.
setProperty("display","block","important")),t&&(t.style.transition="none",t.style.width="100%",t.style.
transformOrigin="left center",t.style.transform="scaleX(0)",t.offsetHeight,t.style.transition="trans\
form 0.45s cubic-bezier(0.22, 1, 0.36, 1)"),i(0),await new Promise(v=>requestAnimationFrame(()=>setTimeout(
v,150)));const a=getRichPasteEditor();if(!a)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093");
const r=inferRichPasteTitle(),l=sanitizeRichPasteHtml(a.innerHTML||""),c=detectRichPasteTheme(l),m=normalizeRichPastePrintHtml(
l);await ensurePdfLibraries();const f=window.jspdf&&window.jspdf.jsPDF?window.jspdf.jsPDF:null;if(!f)
throw new Error("jsPDF \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
const b=window.html2canvas;if(typeof b!="function")throw new Error("html2canvas \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
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
                    <div class="pdf-content">${m}</div>
                </div>
            `,document.body.appendChild(y),await waitForRichPasteMedia(y,4e3),i(15);try{const v=new f(
{unit:"mm",format:"a4",orientation:"portrait",compress:!0}),w=v.internal.pageSize.getWidth(),k=v.internal.
pageSize.getHeight(),_=794,C=Math.floor(k/w*_),L=y.scrollHeight||y.offsetHeight;let $=0,F=!0;const X=Math.
ceil(L/C);let U=0;for(;$<L;){if(richPasteAbortController&&richPasteAbortController.signal.aborted)throw new DOMException(
"Aborted","AbortError");const R=Math.min(C,L-$),ee=(await new Promise((Pe,Le)=>{const ge=setTimeout(
()=>Le(new Error("PDF chunk rendering timed out")),12e4);b(y,{scale:1,useCORS:!0,allowTaint:!1,backgroundColor:c.
background,logging:!1,imageTimeout:5e3,x:0,y:$,width:_,height:R,windowWidth:_,scrollX:0,scrollY:0,signal:richPasteAbortController?
richPasteAbortController.signal:void 0,onclone:o(de=>{prepareRichPastePdfClone(de,c);const H=de.querySelector(
".pdf-root-wrapper");H&&(H.style.position="relative",H.style.left="0",H.style.top="0")},"onclone")}).
then(de=>{clearTimeout(ge),Pe(de)}).catch(de=>{clearTimeout(ge),Le(de)})})).toDataURL("image/jpeg",.95),
we=v.getImageProperties(ee),ce=Math.min(k,we.height*w/we.width);F||v.addPage(),v.addImage(ee,"JPEG",
0,0,w,ce),F=!1,$+=R,U++;const Te=Math.min(100,15+Math.round(U/X*85));i(Te),await new Promise(Pe=>setTimeout(
Pe,100))}return i(100),{blob:v.output("blob"),fileName:buildRichPastePdfFilename()}}finally{e&&(e.classList.
add("hidden"),e.style.display="none"),y&&y.parentNode&&document.body.removeChild(y)}},"renderRichPas\
tePdfBlob"),createRichPastePdfBlob=o(async()=>await renderRichPastePdfBlob(),"createRichPastePdfBlob"),
buildRichPasteServerPayload=o(()=>{const e=getRichPasteEditor();if(!e)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\
\u3093");const t=String(e.innerHTML||"").trim(),n=String(e.textContent||"").trim(),i=t||(n?`<p>${escapeHtml(
n).replace(/\n/g,"<br/>")}</p>`:"");return{title:inferRichPasteTitle(),html:i,created_at:new Date().
toLocaleString("ja-JP"),theme:detectRichPasteTheme(sanitizeRichPasteHtml(i))}},"buildRichPasteServer\
Payload"),attachRichPastePdfAndSend=o(async(e,t,n,i)=>{const a=new Set(collectAttachmentItemsForSend().
map(b=>b.path)),r=new File([e],t,{type:"application/pdf",lastModified:Date.now()}),l=get("prompt-inp\
ut");if(l&&(l.value=n),await handleFiles([r],{openModal:!1}),!collectAttachmentItemsForSend().map(b=>b.
path).some(b=>!a.has(b)))throw l&&(l.value=i),new Error("PDF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const f=sendMessage();clearRichPasteEditor(!0),window.closeRichPasteModal(),showToast("PDF\u3092\u6DFB\u4ED8\u3057\u3066\u9001\u4FE1\u3092\u958B\u59CB\
\u3057\u307E\u3057\u305F","success"),f&&typeof f.catch=="function"&&f.catch(()=>{})},"attachRichPast\
ePdfAndSend"),openRichPasteModal=o(async()=>{await ensureUserSettingsSnapshot(),showModal("rich-past\
e-modal"),location.pathname!=="/paste"&&history.pushState({modal:"paste"},"","/paste");const e=getRichPastePrompt();
e&&(richPastePromptPreferenceSyncing=!0,e.value=getRichPasteEffectivePrompt(userSettingsSnapshot),richPastePromptPreferenceSyncing=
!1),updateRichPasteStatus(),setTimeout(()=>focusRichPasteEditor(),80)},"openRichPasteModal");window.
closeRichPasteModal=(e=!1)=>{hasRichPasteContent()&&!confirm("\u8CBC\u308A\u4ED8\u3051\u305F\u5185\u5BB9\u3092\u7834\u68C4\u3057\u3066\u9589\u3058\u307E\u3059\u304B\uFF1F")||
(hideModal("rich-paste-modal",{skipConfirm:!0}),!e&&location.pathname==="/paste"&&history.back())};const sendRichPasteToModel=o(
async(e={})=>{const t=!!(e&&e.serverSide);if(abortController||richPasteAbortController){showToast("\u56DE\
\u7B54\u751F\u6210\u4E2D\u307E\u305F\u306FPDF\u5909\u63DB\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const n=getRichPasteEditor(),i=getRichPastePrompt(),a=get(t?"rich-paste-send-se\
rver-btn":"rich-paste-send-btn"),r=get("rich-paste-cancel-btn");if(!n||!n.innerText||!n.innerText.trim()){
showToast("\u8CBC\u308A\u4ED8\u3051\u308B\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}richPasteAbortController=new AbortController,r&&(r.onclick=()=>{richPasteAbortController&&
(richPasteAbortController.abort(),showToast("PDF\u5909\u63DB\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info"))});const l=i&&i.value&&i.value.trim()?i.value.trim():RICH_PASTE_DEFAULT_PROMPT,c=get("prompt\
-input")?get("prompt-input").value:"";a&&(a.disabled=!0);try{const m=get("toast-stack");if(m&&m.querySelectorAll(
".toast").forEach(f=>{(f.innerText.includes("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059")||
f.innerText.includes("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059"))&&
f.remove()}),t?(showToast("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...",
"info",!0),setRichPasteProgress(2,"\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...")):
showToast("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...","info",!0),t){if(!RICH_PASTE_PDF_SERVER_ROUTE)
throw new Error("\u30B5\u30FC\u30D0\u30FC\u5074PDF\u751F\u6210\u306EURL\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093");
const f=buildRichPasteServerPayload();setRichPasteProgress(10,"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u4FE1\u4E2D...");
const b=await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(f),signal:richPasteAbortController.signal});if(setRichPasteProgress(60,
"PDF\u3092\u53D7\u4FE1\u4E2D..."),!b.ok){let k="";try{const _=await b.json();k=_&&(_.message||_.error)?
String(_.message||_.error):""}catch{try{k=await b.text()}catch{k=""}}throw k==="missing_html"?new Error(
"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u308BHTML\u304C\u7A7A\u3067\u3059\u3002\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u5185\u5BB9\u306E\u53D6\u308A\u8FBC\u307F\u3092\u5148\u306B\u884C\u3063\u3066\u304F\u3060\u3055\u3044"):
new Error(k?`\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${k}`:
"\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}setRichPasteProgress(
75,"PDF\u3092\u6DFB\u4ED8\u4E2D...");const y=await b.blob(),v=b.headers.get("X-Rich-Paste-Filename")||
buildRichPastePdfFilename();!!(get("rich-paste-download-only")&&get("rich-paste-download-only").checked)?
(setRichPasteProgress(90,"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D..."),downloadBlob(y,v),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(y,v,l,c),setRichPasteProgress(100,"\u5B8C\u4E86"),
setTimeout(()=>hideRichPasteProgress(),400)}else{const f=await createRichPastePdfBlob();!!(get("rich\
-paste-download-only")&&get("rich-paste-download-only").checked)?(downloadBlob(f.blob,f.fileName),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(f.blob,f.fileName,l,c)}}catch(m){if(m.
name==="AbortError"){console.log("PDF generation aborted by user"),t&&(setRichPasteProgress(0,"\u30AD\u30E3\u30F3\u30BB\u30EB\
\u3055\u308C\u307E\u3057\u305F"),setTimeout(()=>hideRichPasteProgress(),800));return}get("prompt-inp\
ut")&&(get("prompt-input").value=c);const f=m&&m.message?m.message:"PDF\u5316\u3057\u3066\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
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
403||l.status===404)){let c=null;try{c=await l.clone().json()}catch{}const m=c&&c.error;if(m==="acco\
unt_locked")return!isAdminUser&&!document.getElementById("bot-lock-overlay")&&showBotLockOverlay(c.message||
"\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u3066\u3044\u307E\u3059\u3002",
c.remaining_seconds),l;if(m==="banned"||m==="turnstile_failed"||m==="rate_limit")return l;if(m==="tu\
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
null&&(currentThreadId=String(currentThreadId));const UPLOAD_CONCURRENCY=Math.max(1,Number(CHAT_CONFIG.
uploadConcurrency)||3),TEMP_CHAT_TIMEOUT_MIN_SECONDS=10,TEMP_CHAT_TIMEOUT_MAX_SECONDS=3600,TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS=90,
TEMP_CHAT_HEARTBEAT_MIN_MS=4e3,TEMP_CHAT_HEARTBEAT_MAX_MS=15e3;let activeGem=null,editingGemUuid=null,
currentImageUrls=[],currentMaskImage=null,abortController=null,richPasteAbortController=null,userAutoScroll=!0,
searchTimeout,promptHistory=[],historyIndex=-1,tempPrompt="";const markerAppliedUploads=new Set,attachmentSourceByPath=new Map,
attachmentNameByPath=new Map,BROWSER_FAST_IGNORE_WARNING_STORAGE="browser_fast_mode_ignore_warning",
BROWSER_FAST_MAX_IMAGES=4,BROWSER_FAST_MAX_BYTES=12*1024*1024,browserFastLocalFiles=new Map;let browserFastModeEnabled=!1,
browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,browserFastPreviousOptions=null,
cameraCaptureStream=null,cameraCaptureFacingMode="environment",cameraCaptureBusy=!1,cameraCaptureSequence=0;
const cameraCapturePendingFiles=[],cameraCapturePendingPreviewUrls=[];let modalThreadId=null;const MARKER_HINT_TEXT="\
\u7DE8\u96C6\u6E08\u307F\u306E\u753B\u50CF\u3092\u898B\u3066\u304F\u3060\u3055\u3044\u3002",MARKER_OPACITY_MIN_PCT=.1,
MARKER_OPACITY_MAX_PCT=100,MARKER_OPACITY_MIN_ALPHA=MARKER_OPACITY_MIN_PCT/100,markerState={row:null,
filename:"",hasStroke:!1,naturalWidth:0,naturalHeight:0,colorHex:"#facc15",opacity:.6,history:[],mode:"\
draw",cropRect:null,mosaicRects:[],mosaicPreviewRect:null,baseCanvas:null,baseImageData:null},markerView={
scale:1,offsetX:0,offsetY:0,minScale:1,maxScale:4},threadGemMap={};let pendingGemForNewThread=null,loadedGems=[],
currentJobId=null,currentThreadPending=null,currentVisionModel=null,activeStreamingBubbleId=null,manualStopContext=null,
manualStopSeq=0,isStopMode=!1;const suppressedPendingJobIds=new Set,pendingStreamReconnectJobs=new Set;
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
n=[],i=o(f=>{const b=`@@MATHJAX_BLOCK_${n.length}@@`;return n.push(f),b},"stash"),a=[],r=/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
let l=0,c;for(;(c=r.exec(t))!==null;){const f=c.index;f>l&&a.push({type:"text",value:t.slice(l,f)}),
a.push({type:"code",value:c[0]}),l=f+c[0].length}return l<t.length&&a.push({type:"text",value:t.slice(
l)}),a.length||a.push({type:"text",value:t}),{text:a.map(f=>{if(f.type==="code")return f.value;let b=f.
value;return b=b.replace(/\$\$([\s\S]+?)\$\$/g,i),b=b.replace(/\\\(([\s\S]+?)\\\)/g,i),b=b.replace(/\\\[([\s\S]+?)\\\]/g,
i),b=b.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g,i),b=b.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g,
i),b}).join(""),blocks:n}}o(protectMathSegments,"protectMathSegments");function getStreamMathSegmentKey(e,t){
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
"script").forEach(c=>{const m=c.getAttribute("src")||"";let f=!1;if(m&&isBlockedScriptSrc(m)){const b=a.
createElement("div");b.setAttribute("data-blocked-script","true"),b.style.cssText="background:#fee2e\
2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6p\
x 0;font-family:system-ui;";const y=m.length>70?m.slice(0,67)+"...":m;b.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: "+
y+" \uFF08polyfill.io \u306A\u3069\u306E\u5371\u967A\u30C9\u30E1\u30A4\u30F3\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u7121\u52B9\u5316\u3055\u308C\u307E\u3059\uFF09",
c.parentNode&&c.parentNode.replaceChild(b,c),r=!0,f=!0}else if(!m){const b=c.textContent||"";if(isPasswordPromptingScript(
b)){const y=a.createElement("div");y.setAttribute("data-blocked-script","true"),y.style.cssText="bac\
kground:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:\
12px;margin:6px 0;font-family:system-ui;",y.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: \u30D1\u30B9\u30EF\u30FC\u30C9\u5165\u529B\u8981\u6C42\u306A\u3069\u306E\u7591\u308F\u3057\u3044\u30A4\u30F3\u30E9\u30A4\u30F3\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\
\u305F",c.parentNode&&c.parentNode.replaceChild(y,c),r=!0,f=!0}}}),a.querySelectorAll('a[href^="java\
script:" i], area[href^="javascript:" i]').forEach(c=>{c.setAttribute("href","#"),c.setAttribute("ti\
tle",(c.getAttribute("title")||"")+" [javascript: disabled in preview]")});const l=a.head||a.querySelector(
"head");if(l&&!l.querySelector("base")){const c=a.createElement("base");c.setAttribute("href",`${window.
location.origin}/`),l.insertBefore(c,l.firstChild)}if(t||r){const c=a.body||a.documentElement;if(c){
const m=a.createElement("div");m.style.cssText="position:sticky;top:0;left:0;right:0;z-index:2147483\
647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:syst\
em-ui;border-bottom:1px solid #b91c1c;",m.innerHTML="\u26A0 <strong>\u5B89\u5168\u30D7\u30EC\u30D3\u30E5\u30FC</strong>: polyfill.io \u306A\u3069\u306E\u5371\u967A\u306A\u30B9\
\u30AF\u30EA\u30D7\u30C8\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002\u5B9F\u884C\u306F\u81EA\u5DF1\u8CAC\u4EFB\u3067\u3002",
c.firstChild?c.insertBefore(m,c.firstChild):c.appendChild(m)}}n=`<!DOCTYPE html>
`+(a.documentElement?a.documentElement.outerHTML:n)}catch{n=n.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
"<!-- blocked polyfill.io script for safety -->")}return n}o(sanitizeHtmlForPreview,"sanitizeHtmlFor\
Preview");function wrapTextWave(e){return e?e.split("").map((t,n)=>`<span class="wave-char" style="a\
nimation-delay: ${n*.028}s">${escapeHtml(t)}</span>`).join(""):""}o(wrapTextWave,"wrapTextWave");function getPendingSkeletonKind(e){
let t=String(e||"").toLowerCase();if(!t)try{t=String(get("model-select")&&get("model-select").value||
"").toLowerCase()}catch{t=""}return t.includes("video")?"video":t.includes("tts")||t.includes("trans\
cribe")||t.includes("realtime")||t.includes("voice")||t.includes("native-audio")||t.includes("live")&&
t.includes("gemini")?"audio":t.includes("gpt-image")||t.includes("imagine-image")||t.startsWith("ide\
ogram-")||t.includes("image")&&!t.includes("vision")||t.includes("gemini")&&(t.includes("image")||t.
includes("nano"))?"image":t.includes("ocr")||t.includes("mistral-ocr")?"text":t.includes("build")||t.
includes("code-fast")||t.includes("coding")?"code":"text"}o(getPendingSkeletonKind,"getPendingSkelet\
onKind");function buildPendingSkeletonBody(e){return e==="image"?'<div class="skeleton-media skeleto\
n-image" aria-hidden="true"><div class="skeleton-media-icon"><i class="fas fa-image"></i></div></div\
>':e==="video"?'<div class="skeleton-media skeleton-video" aria-hidden="true"><div class="skeleton-m\
edia-icon"><i class="fas fa-play"></i></div><div class="skeleton-video-progress"></div></div>':e==="\
audio"?'<div class="skeleton-audio" aria-hidden="true"><div class="skeleton-audio-disc"><i class="fa\
s fa-volume-up"></i></div><div class="skeleton-wave"><span></span><span></span><span></span><span></\
span><span></span><span></span><span></span><span></span></div></div>':e==="code"?'<div class="skele\
ton-code" aria-hidden="true"><div class="skeleton-code-header"><span class="skeleton-code-dot"></spa\
n><span class="skeleton-code-dot"></span><span class="skeleton-code-dot"></span><div class="skeleton\
-code-title"></div></div><div class="skeleton-lines skeleton-code-lines"><div class="skeleton-line" \
style="width:72%"></div><div class="skeleton-line" style="width:88%"></div><div class="skeleton-line\
" style="width:54%"></div><div class="skeleton-line" style="width:76%"></div><div class="skeleton-li\
ne" style="width:41%"></div></div></div>':'<div class="skeleton-lines" aria-hidden="true"><div class\
="skeleton-line" style="width:92%"></div><div class="skeleton-line" style="width:78%"></div><div cla\
ss="skeleton-line" style="width:86%"></div><div class="skeleton-line" style="width:64%"></div><div c\
lass="skeleton-line" style="width:48%"></div></div>'}o(buildPendingSkeletonBody,"buildPendingSkeleto\
nBody");function buildPendingSkeletonHtml(e,t){const n=getPendingSkeletonKind(e),i=t==null||t===""?"\
\u56DE\u7B54\u3092\u751F\u6210\u4E2D...":String(t);return`<div class="content-area pending-shimmer s\
keleton-pending" data-skeleton-kind="${escapeHtml(n)}">${buildPendingSkeletonBody(n)}<div class="ske\
leton-status">${escapeHtml(i)}</div></div>`}o(buildPendingSkeletonHtml,"buildPendingSkeletonHtml");function updatePendingSkeletonStatus(e,t,n){
if(!e)return!1;const i=e.querySelector(".content-area.skeleton-pending");if(!i)return!1;let a=i.querySelector(
".skeleton-status");a||(a=document.createElement("div"),a.className="skeleton-status",i.appendChild(
a));const r=t==null?"":String(t),l=n==null||n===""?"":String(n);return l?a.innerHTML=`${escapeHtml(r)}\
<span class="skeleton-status-sub">${escapeHtml(l)}</span>`:a.textContent=r,!0}o(updatePendingSkeletonStatus,
"updatePendingSkeletonStatus");function buildChatLoadingSkeletonHtml(){return`<div class="chat-load-\
skeleton" role="status" aria-live="polite" aria-label="\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D">${[
{role:"user",widths:["62%","44%"]},{role:"ai",widths:["88%","76%","92%","58%"]},{role:"user",widths:[
"48%"]},{role:"ai",widths:["82%","70%","54%"]}].map((n,i)=>{const a=n.role==="user",r=a?"justify-end":
"justify-start",l=a?"message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4\
 rounded-2xl rounded-tr-none shadow-md relative":"message-bubble chat-load-skeleton-bubble chat-load\
-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative",c=n.widths.map(
(m,f)=>`<div class="skeleton-line" style="width:${m};animation-delay:${(i*.08+f*.06).toFixed(2)}s"><\
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
const m=String(c[2]||"").trim();i={markerChar:c[1][0],markerLength:c[1].length,language:(m.split(/\s+/)[0]||
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
var m;if(codingModeEnabled=!!e,t.persist!==!1)try{localStorage.setItem(CODING_MODE_STORAGE_KEY,codingModeEnabled?
"true":"false")}catch{}const n=get("enable-coding-mode");n&&n.checked!==codingModeEnabled&&(n.checked=
codingModeEnabled);const i=get("coding-target-bar"),a=get("coding-target-text"),r=get("clear-coding-\
target-btn");i&&i.classList.toggle("visible",codingModeEnabled);const l=resolveCodingTarget(),c=codingTargetSelection?
[l].filter(Boolean):collectCodingCandidates(String(((m=get("prompt-input"))==null?void 0:m.value)||""));
if(codingModeEffective=codingModeEnabled&&c.length>0,a)if(codingTargetSelection&&l)a.textContent=`\u7DE8\u96C6\
\u5BFE\u8C61: ${l.language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`;else if(c.length>1){
const f=c.filter(y=>y.prompt_source).length,b=c.length-f;a.textContent=`\u30E2\u30C7\u30EB\u304C\u7DE8\u96C6\u5BFE\u8C61\u3092\u5224\u65AD: \u5165\u529B${f}\
\u4EF6 / \u5C65\u6B74${b}\u4EF6`}else l&&l.prompt_source?a.textContent=`\u5165\u529B\u4E2D: ${l.language||
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
ing-live-diff-count"),m=n.querySelectorAll(".coding-live-diff-edit").length;c&&(c.textContent=`${m} \
edit${m===1?"":"s"}`),n.scrollIntoView({block:"nearest",behavior:"smooth"})}o(appendCodingLiveDiff,"\
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
toggle("collapsed",!!c);const m=a.querySelector(".code-toggle");m&&(m.setAttribute("aria-expanded",c?
"false":"true"),m.innerHTML=c?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></\
i>',m.title=c?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",m.setAttribute("aria-label",c?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080"))})}o(applyCodeCollapse,"applyCodeCollapse");function snapshotCodeCollapseByMessage(e){
if(!e)return new Map;const t=new Map;return e.querySelectorAll(".message-group").forEach(n=>{const i=n.
getAttribute("id")||"";n.querySelectorAll(".code-wrapper").forEach((a,r)=>{const l=a.getAttribute("d\
ata-code-key")||String(r),c=a.classList.contains("collapsed")||a.getAttribute("data-collapsed")==="t\
rue";t.set(`${i}:${l}`,c)})}),t}o(snapshotCodeCollapseByMessage,"snapshotCodeCollapseByMessage");function applyCodeCollapseByMessage(e,t,n=!1){
e&&e.querySelectorAll(".message-group").forEach(i=>{const a=i.getAttribute("id")||"";i.querySelectorAll(
".code-wrapper").forEach((r,l)=>{const c=r.getAttribute("data-code-key")||String(l),m=`${a}:${c}`,f=t&&
t.has(m)?t.get(m):n;r.setAttribute("data-collapsed",f?"true":"false"),r.classList.toggle("collapsed",
!!f);const b=r.querySelector(".code-toggle");b&&(b.setAttribute("aria-expanded",f?"false":"true"),b.
innerHTML=f?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></i>',b.title=f?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080",b.setAttribute("aria-label",f?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080"))})})}
o(applyCodeCollapseByMessage,"applyCodeCollapseByMessage");function buildTokenTotals(e){const t={tokens_total:0,
tokens_in:0,tokens_out:0,tokens_content:0,tokens_thought:0};let n=!1,i=!1,a=!1,r=!1,l=!1;return(e||[]).
forEach(c=>{if(!c)return;let m=null;c.tokens!==null&&c.tokens!==void 0?m=Number(c.tokens||0):(c.tokens_in!==
null&&c.tokens_in!==void 0||c.tokens_out!==null&&c.tokens_out!==void 0)&&(m=Number(c.tokens_in||0)+Number(
c.tokens_out||0)),m!==null&&(t.tokens_total+=m,n=!0),c.tokens_in!==null&&c.tokens_in!==void 0&&(t.tokens_in+=
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
0),m=[];Number(e.files_non_text||0)>0&&m.push(`\u975E\u30C6\u30AD\u30B9\u30C8${e.files_non_text}\u4EF6\u306F0\u63DB\
\u7B97`),Number(e.files_missing||0)>0&&m.push(`\u672A\u691C\u51FA${e.files_missing}\u4EF6`),Number(e.
files_error||0)>0&&m.push(`\u5931\u6557${e.files_error}\u4EF6`);const f=m.length?` \u30FB ${m.join("\
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
const m=await c.json();if(l!==promptTokenEstimateSeq)return;promptTokenEstimateLastKey=a,promptTokenEstimateLastData=
m,renderPromptTokenEstimate(m,t)}catch(c){if(c&&c.name==="AbortError"||l!==promptTokenEstimateSeq)return;
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
ns","grok-image-options","ideogram-image-options","xai-chat-options","grok-video-options","mistral-o\
cr-options","image-input-limits","audio-gen-options"],THINKING_LEVELS=[{value:"minimal",label:"Min"},
{value:"low",label:"Low"},{value:"medium",label:"Mid"},{value:"high",label:"High"}],MINIMAL_POPUP_ITEMS=[
{key:"attach",icon:"fa-paperclip",label:"\u30D5\u30A1\u30A4\u30EB\u3092\u6DFB\u4ED8",action:"upload"},
{key:"voice-input",icon:"fa-microphone",label:"Voice Input",action:"button",buttonId:"mic-btn"},{key:"\
rich-paste",icon:"fa-paste",label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051",action:"button",buttonId:"\
rich-paste-btn"},{key:"canvas",icon:"fa-window-restore",label:"Canvas",checkboxId:"enable-canvas-mod\
e",containerId:"canvas-mode-container"},{key:"coding",icon:"fa-code-branch",label:"Coding",checkboxId:"\
enable-coding-mode",containerId:"coding-mode-container"},{key:"fast",icon:"fa-bolt",label:"\u9AD8\u901F",
checkboxId:"enable-browser-fast-mode",containerId:"browser-fast-mode-container"},{key:"batch",icon:"\
fa-layer-group",label:"Batch",checkboxId:"enable-batch-mode",containerId:"batch-mode-container"},{key:"\
search",icon:"fa-search",label:"Search",checkboxId:"enable-search",containerId:"search-container"},{
key:"urls",icon:"fa-link",label:"URLs",checkboxId:"enable-url-context",containerId:"url-context-cont\
ainer"},{key:"maps",icon:"fa-map-location-dot",label:"Maps",checkboxId:"enable-maps",containerId:"ma\
ps-grounding-container"},{key:"python",icon:"fa-code",label:"Python",checkboxId:"enable-python",containerId:"\
python-container"},{key:"file",icon:"fa-file-lines",label:"File",checkboxId:"enable-file-creation",containerId:"\
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
const i=get("thinking-slider");i&&i.addEventListener("input",()=>{const c=Number(i.value),m=allowedThinkingValues(),
f=get("thinking-level");if(m.length){const b=m.map(v=>thinkingIndexFromValue(v)),y=b.includes(c)?c:b.
reduce((v,w)=>Math.abs(w-c)<Math.abs(v-c)?w:v,b[0]);f&&(f.value=THINKING_LEVELS[y].value,f.dispatchEvent(
new Event("change",{bubbles:!0})))}syncThinkingSliderUi(),scheduleThinkingSliderHide()});const a=get(
"thinking-slide-close-btn");a&&a.addEventListener("click",c=>{c.stopPropagation(),hideThinkingSlider()});
const r=get("thinking-slide-bar");if(r){const c=get("thinking-slide-inner");r.addEventListener("touc\
hstart",m=>{thinkingSliderOpen&&(thinkingSliderDragging=!0,thinkingSliderStartY=m.touches[0].clientY,
thinkingSliderStartX=m.touches[0].clientX,thinkingSliderAxis=null,c&&c.classList.add("dragging"))},{
passive:!0}),r.addEventListener("touchmove",m=>{if(!thinkingSliderDragging)return;const f=m.touches[0].
clientX-thinkingSliderStartX,b=m.touches[0].clientY-thinkingSliderStartY;if(thinkingSliderAxis===null&&
(Math.abs(f)>8||Math.abs(b)>8)&&(thinkingSliderAxis=Math.abs(b)>Math.abs(f)?"v":"h"),thinkingSliderAxis===
"v")if(b>0){m.cancelable&&m.preventDefault();const y=Math.min((b-8)*.5,120);c&&(c.style.transform=y>
0?`translateY(${y}px)`:"")}else c&&(c.style.transform="")},{passive:!1}),r.addEventListener("touchen\
d",m=>{if(!thinkingSliderDragging)return;thinkingSliderDragging=!1;const f=m.changedTouches[0].clientY-
thinkingSliderStartY;c&&c.classList.remove("dragging"),thinkingSliderAxis==="v"&&f>100?(c&&(c.style.
transform=`translateY(${Math.max(f*.5,60)}px)`),hideThinkingSlider()):(c&&(c.style.transform=""),scheduleThinkingSliderHide())},
{passive:!0}),r.addEventListener("touchcancel",()=>{thinkingSliderDragging=!1,c&&(c.classList.remove(
"dragging"),c.style.transform=""),scheduleThinkingSliderHide()},{passive:!0})}const l=get("minimal-o\
ptions-panel");l&&(l.addEventListener("touchstart",c=>{if(!minimalOptionsOpen)return;popupSwipeDragging=
!0,popupSwipeStartY=c.touches[0].clientY,popupSwipeStartX=c.touches[0].clientX,popupSwipeAxis=null;let m=c.
target instanceof Element?c.target:null,f=!0;for(;m&&m!==l;){if(m.scrollTop>0){f=!1;break}m=m.parentElement}
popupSwipeAtTop=f,f&&l.classList.add("dragging")},{passive:!0}),l.addEventListener("touchmove",c=>{if(!popupSwipeDragging||
!popupSwipeAtTop||!minimalOptionsOpen)return;const m=c.touches[0].clientX-popupSwipeStartX,f=c.touches[0].
clientY-popupSwipeStartY;popupSwipeAxis===null&&(Math.abs(m)>8||Math.abs(f)>8)&&(popupSwipeAxis=Math.
abs(f)>Math.abs(m)?"v":"h"),popupSwipeAxis==="v"&&f>0&&(c.cancelable&&c.preventDefault(),l.style.transform=
`translateY(${Math.min(f*.6,140)}px)`)},{passive:!1}),l.addEventListener("touchend",c=>{if(!popupSwipeDragging)
return;popupSwipeDragging=!1;const m=c.changedTouches[0].clientY-popupSwipeStartY;l.classList.remove(
"dragging"),popupSwipeAtTop&&popupSwipeAxis!=="h"&&m>70?(l.style.transform=`translateY(${Math.max(m*
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
chatDefaultsLoaded=!0,typeof window.toggleOptions=="function"&&window.toggleOptions(),applyMcpPromptChipUi()}
o(applyChatDefaults,"applyChatDefaults");function setEditUi(e){const t=get("edit-bar");t&&(e?(t.classList.
remove("hidden"),t.classList.add("flex")):(t.classList.add("hidden"),t.classList.remove("flex")),updatePromptPlaceholder())}
o(setEditUi,"setEditUi");function cancelEdit(){editingMessageId=null,currentParentId=currentLeafId||
null;const e=get("prompt-input");e&&(e.value="",e.style.height="auto"),currentImageUrls=[],get("file\
-preview").classList.add("hidden"),get("file-input").value="",clearQuote(),setEditUi(!1)}o(cancelEdit,
"cancelEdit");function beginEditMessage(e,t=!1){const n=messageStore[e];if(n==null)return;const i=get(
"prompt-input");i.value=n||"",i.focus(),i.style.height="auto",i.style.height=i.scrollHeight+"px";const a=allMessages.
find(m=>m.id==e),r=messageMeta[e]||{};a?currentParentId=a.parent_id===void 0?null:a.parent_id:r.parent_id!==
void 0&&(currentParentId=r.parent_id),editingMessageId=e,setEditUi(!0);const l=a?a.image_url:r.image_url;
if(l)try{const m=JSON.parse(l);Array.isArray(m)&&m.length?(currentImageUrls=m.map(f=>{let b="unknown",
y=f;f&&typeof f=="object"&&(b=normalizeAttachmentSource(f.source),y=f.filepath||f.path||f.url||f.file||
"");const v=normalizeAttachmentPath(y);return v&&setAttachmentSourceForPath(v,b),v}).filter(Boolean),
get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):(currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value=
"")}catch{currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value=""}else
currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value="";const c=a?
a.quote_text:r.quote_text;c?(currentQuote=c,get("quote-text-display").innerText=currentQuote,get("qu\
ote-bar").classList.add("visible")):clearQuote(),schedulePromptTokenEstimate(!0),t&&sendMessage()}o(
beginEditMessage,"beginEditMessage");function playSendAnimation(){const e=get("send-btn");e&&(e.classList.
remove("fly"),e.offsetWidth,e.classList.add("fly"))}o(playSendAnimation,"playSendAnimation");function setSendBtnToStopMode(){
const e=get("send-btn");if(!e)return;e.onclick=stopGeneration,isStopMode=!0,e.disabled=!1;const t=o(
()=>{!e||!isStopMode||(e.classList.add("stop-mode"),e.innerHTML='<span style="font-size:20px;line-he\
ight:1;color:#fff;">\u25A0</span>',e.classList.add("btn-swap"),setTimeout(()=>e.classList.remove("bt\
n-swap"),300))},"applyStopUi");if(e.classList.contains("fly")){const n=o(i=>{i.animationName==="send\
BtnPop"&&(e.removeEventListener("animationend",n),t())},"onEnd");e.addEventListener("animationend",n),
setTimeout(t,700)}else t()}o(setSendBtnToStopMode,"setSendBtnToStopMode");function setSendBtnToSendMode(){
const e=get("send-btn");e&&(e.classList.remove("stop-mode","fly","btn-swap"),e.innerHTML='<i class="\
fas fa-paper-plane"></i>',e.classList.add("btn-swap"),setTimeout(()=>e.classList.remove("btn-swap"),
300),e.onclick=sendMessage,isStopMode=!1)}o(setSendBtnToSendMode,"setSendBtnToSendMode");async function stopGeneration(){
const e=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,t=normalizeJobIdForUi(
currentJobId),n=++manualStopSeq,i=captureStoppedPartialBubbleSnapshot(getActiveStreamingBubbleElement());
manualStopContext={seq:n,threadId:e,jobId:t,partialSnapshot:i},t&&suppressPendingJob(t),abortController&&
abortController.abort();try{if(t||e){const a={};t&&(a.job_id=t),e&&(a.thread_id=e);const l=await(await apiFetch(
"/api/stop_chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(a)})).
json().catch(()=>({})),c=normalizeJobIdForUi(l&&l.job_id);c&&(suppressPendingJob(c),manualStopContext&&
manualStopContext.seq===n&&(manualStopContext.jobId=c))}manualStopContext&&manualStopContext.seq===n&&
await syncThreadAfterAbortedStream(e,{retries:2,retryDelayMs:180,notifyOnFailure:!0})&&manualStopContext.
partialSnapshot&&appendStoppedPartialBubbleSnapshot(manualStopContext.partialSnapshot,e)}finally{manualStopContext&&
manualStopContext.seq===n&&(manualStopContext=null),setSendBtnToSendMode(),updateFilePreview()}}o(stopGeneration,
"stopGeneration");async function purgeCaches(){if("caches"in window){const e=await caches.keys();await Promise.
all(e.map(t=>caches.delete(t)))}if(navigator.serviceWorker){const e=await navigator.serviceWorker.getRegistrations();
await Promise.all(e.map(t=>t.unregister()))}}o(purgeCaches,"purgeCaches");const SW_CACHE_MODE_STORAGE_KEY="\
ai_sw_cache_mode_v2";async function applyCacheMode(e,t={}){if("serviceWorker"in navigator)if(e)try{await navigator.
serviceWorker.register(`/sw.js?v=${encodeURIComponent(appVersion)}`),localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY,
"enabled")}catch{}else{const n=localStorage.getItem(SW_CACHE_MODE_STORAGE_KEY);(!!t.forceCleanup||n!==
"disabled")&&await purgeCaches(),localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY,"disabled")}}o(applyCacheMode,
"applyCacheMode");function checkAndNotifyVersion(e){!e||!appVersion||e===appVersion||(localStorage.getItem(
"version_notified")||"")===e||(localStorage.setItem("app_version",e),syncVersionUpdateCachePreferenceUi(),
showModal("version-update-modal"))}o(checkAndNotifyVersion,"checkAndNotifyVersion");async function checkVersion(){
try{const e=await fetch("/api/version",{cache:"no-store"});if(!e.ok)return;const n=(await e.json()).
version||"",i=localStorage.getItem("app_version")||"";n&&!i&&localStorage.setItem("app_version",n),n&&
i&&n!==i&&(await purgeCaches(),checkAndNotifyVersion(n))}catch{}}o(checkVersion,"checkVersion");async function fetchChatStreamWithUnavailableRetry(e,t,n){
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
ck-overlay-timer",c.style.cssText="font-size:12px;color:#94a3b8;margin-top:2px;";const m=document.createElement(
"div");m.style.cssText="font-size:11px;color:#94a3b8;line-height:1.6;",m.textContent="\u30ED\u30C3\u30AF\u89E3\u9664\u307E\u3067\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\
\u304F\u3060\u3055\u3044\u3002\u540C\u3058\u64CD\u4F5C\u3092\u7E70\u308A\u8FD4\u3059\u3068BAN\u3055\u308C\u308B\u5834\u5408\u304C\u3042\u308A\u307E\u3059\u3002",
i.appendChild(a),i.appendChild(r),i.appendChild(l),i.appendChild(c),i.appendChild(m),n.appendChild(i),
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
speedSamples=0},"resetWindow"),i=o(v=>{const w=v&&v.target;return!w||typeof w.closest!="function"?!1:
!!w.closest("[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay")},
"isControlClick"),a=o(v=>{if(i(v))return;if(v&&v.isTrusted===!1){e.untrustedInput=!0,f(!0);return}const w=performance.
now();if(e.clicks+=1,e.lastClickTs){const k=w-e.lastClickTs;e.clickIntervals.push(k),e.clickIntervals.
length>10&&e.clickIntervals.shift(),k<120&&(e.fastClicks+=1)}e.lastClickTs=w,e.clickTimes.push(w),e.
clickTimes=e.clickTimes.filter(k=>w-k<=2e3),e.fastClicks>=4&&f(!0)},"recordClick"),r=o(v=>{if(v&&v.isTrusted===
!1){e.untrustedInput=!0,f(!0);return}const w=performance.now();e.keys+=1,e.lastKeyTs&&w-e.lastKeyTs<
50&&(e.fastKeys+=1),e.lastKeyTs=w,e.keyTimes.push(w),e.keyTimes=e.keyTimes.filter(k=>w-k<=2e3)},"rec\
ordKey"),l=o(v=>{const w=performance.now();if(!(w-e.lastMoveSample<80)){if(e.lastMoveSample=w,e.moves+=
1,e.lastMove){const k=v.clientX-e.lastMove.x,_=v.clientY-e.lastMove.y,C=w-e.lastMove.t;if(C>0){const L=Math.
sqrt(k*k+_*_)/(C/1e3);e.speedMax=Math.max(e.speedMax,L),e.speedSum+=L,e.speedSamples+=1}}e.lastMove=
{x:v.clientX,y:v.clientY,t:w}}},"recordMove"),c=o(()=>{const v=Math.max(1,performance.now()-e.windowStart),
w=e.clickTimes.length,k=e.keyTimes.length,_=e.speedSamples?e.speedSum/e.speedSamples:0;let C=0,L=1;if(e.
clickIntervals.length>=3){const $=e.clickIntervals.reduce((X,U)=>X+U,0)/e.clickIntervals.length,F=e.
clickIntervals.reduce((X,U)=>X+Math.pow(U-$,2),0)/e.clickIntervals.length;C=$,L=$>0?Math.sqrt(F)/$:1}
return{window_ms:Math.round(v),clicks:e.clicks,keys:e.keys,moves:e.moves,fast_clicks:e.fastClicks,fast_keys:e.
fastKeys,untrusted_input:!!e.untrustedInput,click_burst:w,key_burst:k,avg_click_ms:C,click_cv:L,event_rate:(e.
clicks+e.keys+e.moves)/(v/1e3),pointer_speed_max:e.speedMax,pointer_speed_avg:_}},"computeStats"),m=o(
v=>v.fast_clicks>=4||v.fast_keys>=8||v.click_burst>=8||v.key_burst>=14||v.event_rate>=20||v.avg_click_ms>
0&&v.avg_click_ms<160&&v.click_cv<.08,"isSuspicious"),f=o(async(v=!1,w={})=>{if(!e.enabled)return;const k=performance.
now();if(!v&&k-e.lastSend<3e3)return;e.lastSend=k;const _=c();if(!(!w.forceReport&&_.clicks+_.keys+_.
moves===0&&!_.untrusted_input)&&!(!v&&!_.untrusted_input&&!m(_))){_.turnstile_token=await getTurnstileToken(),
botConfig&&botConfig.turnstileSiteKey&&!_.turnstile_token&&!botDetectionVerified&&botDetectionOverlayShown&&
(_.turnstile_failed=!0,_.challenged=!0);try{const C=await apiFetch("/api/bot-telemetry",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(_)});if(C.status===403){let L=null;
try{L=await C.json()}catch{}if(L&&L.error==="banned"){showToast("\u30DC\u30C3\u30C8\u5224\u5B9A\u306B\u3088\u308ABAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800);return}}}catch{}resetTurnstileToken(),n()}},
"send");return{start:o(()=>{t(),e.enabled&&(typeof window.PointerEvent!="undefined"?document.addEventListener(
"pointerdown",a,!0):document.addEventListener("click",a,!0),document.addEventListener("keydown",r,!0),
document.addEventListener("wheel",()=>{e.moves+=1},{passive:!0}),document.addEventListener("mousemov\
e",l,!0),setInterval(()=>f(!1),4e3))},"start"),refreshEnabled:t,send:f,looksSuspicious:o(()=>{if(!e.
enabled)return!1;const v=c();return m(v)},"looksSuspicious")}})();function openFileViewer(e,t=""){if(!e)
return;const n=(t||e).split(".").pop().toLowerCase(),i=["png","jpg","jpeg","webp","gif"],a=["mp4","m\
ov","mkv","avi","m4v","webm"],r=["mp3","wav","m4a","ogg","flac"],l=["pdf","txt","md","csv","log","js\
on","docx"];if(i.includes(n)){openImageViewer(e);return}const c=get("file-viewer"),m=get("file-viewe\
r-body"),f=get("file-viewer-title");if(!(!c||!m||!f)){if(f.textContent=t||"File Preview",m.replaceChildren(),
a.includes(n)){const b=document.createElement("video");b.src=String(e),b.controls=!0,b.playsInline=!0,
b.preload="metadata",m.appendChild(b)}else if(r.includes(n)){const b=document.createElement("audio");
b.src=String(e),b.controls=!0,m.appendChild(b)}else if(l.includes(n)){const b=document.createElement(
"iframe");b.src=String(e),b.setAttribute("sandbox",""),b.referrerPolicy="no-referrer",m.appendChild(
b)}else{const b=document.createElement("div");b.className="fallback",b.appendChild(document.createTextNode(
"\u3053\u306E\u5F62\u5F0F\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u304D\u307E\u305B\u3093\u3002"));
const y=document.createElement("div");y.className="mt-3 flex justify-center gap-2";const v=document.
createElement("a");v.href=String(e),v.download="",v.className="px-3 py-1 bg-gray-800 text-white roun\
ded text-xs border border-gray-700",v.textContent="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9";const w=document.
createElement("a");w.href=String(e),w.target="_blank",w.rel="noopener noreferrer",w.className=v.className,
w.textContent="\u65B0\u3057\u3044\u30BF\u30D6\u3067\u958B\u304F",y.append(v,w),b.appendChild(y),m.appendChild(
b)}c.classList.add("visible")}}o(openFileViewer,"openFileViewer");function closeFileViewer(){const e=get(
"file-viewer"),t=get("file-viewer-body");!e||!t||(t.innerHTML="",e.classList.remove("visible"))}o(closeFileViewer,
"closeFileViewer"),window.showToast=showToast;function showToast(e,t="error",n=!1,i=null){const a=get(
"toast-stack");if(!a)return;for(;a.children.length>=3;)a.removeChild(a.firstChild);const r=document.
createElement("div");return r.className=`toast ${t}${i?" toast-clickable":""}`,r.innerHTML=`<i class\
="fas ${t==="error"?"fa-triangle-exclamation":"fa-circle-info"}"></i><span class="flex-1">${escapeHtml(
e)}</span><button aria-label="close"><i class="fas fa-times"></i></button>`,r.querySelector("button").
onclick=l=>{l.stopPropagation(),r.remove()},i&&r.addEventListener("click",i),a.appendChild(r),n||setTimeout(
()=>{r.parentNode&&r.remove()},7e3),r}o(showToast,"showToast");function showProgressToast(e,t="info"){
const n=get("toast-stack");if(!n)return null;for(;n.children.length>=3;)n.removeChild(n.firstChild);
const i=document.createElement("div");return i.className=`toast ${t} flex-col !items-start min-w-[24\
0px]`,i.innerHTML=`
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
!t){ALL_TABS.forEach(c=>{const m=get("btn-tab-"+c);if(m){const b=m.querySelector(".settings-search-b\
adge");b&&b.remove()}const f=get("tab-"+c);f&&f.classList.toggle("hidden",c!==activeSettingsTab)});return}
let i=[];ALL_TABS.forEach(c=>{const m=get("tab-"+c);m&&(m.classList.add("hidden"),Array.from(m.children).
forEach(f=>{if(!(f.classList.contains("settings-no-results")||f.classList.contains("settings-search-\
overlay"))&&f.textContent.toLowerCase().includes(t)){const b=getSectionHeading(f)||c,y=getSectionSnippet(
f,t);i.push({tabId:c,title:b,snippet:y,element:f})}}))});let a=activeSettingsTab;if(!i.some(c=>c.tabId===
a)){const c=i.find(m=>m.tabId);c&&(a=c.tabId)}const r=get("tab-"+a);if(!r)return;r.classList.remove(
"hidden"),Array.from(r.children).forEach(c=>{c.classList.contains("settings-no-results")||c.classList.
contains("settings-search-overlay")||(c.style.display="none")});const l=document.createElement("div");
if(l.className="settings-search-overlay",i.length===0){const c=document.createElement("div");c.className=
"settings-empty-state",c.innerHTML='<div class="settings-empty-icon"><i class="fas fa-search"></i></\
div><div class="settings-empty-title">\u4E00\u81F4\u3059\u308B\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093</div>';
const m=document.createElement("div");m.className="settings-empty-sub",m.textContent="\u300C"+t+"\u300D\u306B\u4E00\
\u81F4\u3059\u308B\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3002",c.appendChild(
m),l.appendChild(c)}else{const c=document.createElement("div");c.className="settings-search-count",c.
textContent=i.length+"\u4EF6\u306E\u4E00\u81F4",l.appendChild(c);let m=null;i.forEach((f,b)=>{if(f.tabId!==
m){if(m!==null){const C=document.createElement("div");C.className="border-t border-gray-700/50 my-1.\
5",l.appendChild(C)}if(f.tabId!==a){const C=document.createElement("div");C.className="text-[10px] t\
ext-gray-500 px-1 pb-1 font-bold",C.textContent="\u25BC "+(TAB_LABELS[f.tabId]||f.tabId),l.appendChild(
C)}m=f.tabId}const y=document.createElement("div");y.className="settings-search-result-item flex ite\
ms-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150",y.style.animation=
"fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both",y.style.animationDelay=b*30+"ms";const v=document.
createElement("span");v.className="settings-result-tab-badge shrink-0 mt-0.5",v.textContent=TAB_LABELS[f.
tabId]||f.tabId;const w=document.createElement("div");w.className="min-w-0 flex-1";const k=document.
createElement("div");k.className="text-sm font-bold text-white truncate",k.textContent=f.title;const _=document.
createElement("div");_.className="text-[11px] text-gray-400 truncate mt-0.5",_.textContent=f.snippet,
w.appendChild(k),w.appendChild(_),y.appendChild(v),y.appendChild(w),y.addEventListener("click",()=>jumpToSetting(
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
e.dataset.scrollBound="1";const a=56;let r=0,l=0,c=0;const m=o(w=>{const k=e.getBoundingClientRect();
if(!k.width)return;const _=w-k.left;e.classList.toggle("is-edge-left",_>=0&&_<=a),e.classList.toggle(
"is-edge-right",_>=k.width-a&&_<=k.width)},"updateEdgeHover"),f=o(()=>{c||e.classList.remove("is-edg\
e-left","is-edge-right")},"clearEdgeHover"),b=o((w,k)=>{const _=getSettingsTabsMaxScroll(t);if(_<=0||
!w)return;const C=Math.max(0,Math.min(_,t.scrollLeft+w));k&&typeof t.scrollTo=="function"?t.scrollTo(
{left:C,behavior:"smooth"}):t.scrollLeft=C,syncSettingsTabsOverflow()},"scrollTabsBy"),y=o(()=>{c=0,
r&&(clearTimeout(r),r=0),l&&(cancelAnimationFrame(l),l=0)},"stopHold"),v=o(w=>{y(),c=w,e.classList.toggle(
"is-edge-left",w<0),e.classList.toggle("is-edge-right",w>0),b(w*Math.max(120,t.clientWidth*.55),!0),
r=setTimeout(()=>{const k=o(()=>{c&&(b(c*14,!1),l=requestAnimationFrame(k))},"step");l=requestAnimationFrame(
k)},280)},"startHold");if(e.addEventListener("pointermove",w=>{w.pointerType!=="touch"&&m(w.clientX)}),
e.addEventListener("pointerenter",w=>{w.pointerType!=="touch"&&m(w.clientX)}),e.addEventListener("po\
interleave",w=>{w.pointerType!=="touch"&&(y(),f())}),e.addEventListener("wheel",w=>{const k=getSettingsTabsMaxScroll(
t);if(k<=2)return;const C=Math.abs(w.deltaY)>=Math.abs(w.deltaX)?w.deltaY:w.deltaX;if(!C)return;const L=Math.
max(0,Math.min(k,t.scrollLeft+C));L!==t.scrollLeft&&(w.preventDefault(),t.scrollLeft=L,syncSettingsTabsOverflow())},
{passive:!1}),n.addEventListener("pointerdown",w=>{w.button!=null&&w.button!==0||(w.preventDefault(),
v(-1))}),i.addEventListener("pointerdown",w=>{w.button!=null&&w.button!==0||(w.preventDefault(),v(1))}),
n.addEventListener("click",w=>{w.preventDefault(),w.stopPropagation()}),i.addEventListener("click",w=>{
w.preventDefault(),w.stopPropagation()}),window.addEventListener("pointerup",y),window.addEventListener(
"pointercancel",y),window.addEventListener("blur",y),t.addEventListener("scroll",syncSettingsTabsOverflow,
{passive:!0}),window.addEventListener("resize",syncSettingsTabsOverflow),typeof ResizeObserver!="und\
efined")try{const w=new ResizeObserver(()=>syncSettingsTabsOverflow());w.observe(t),w.observe(e)}catch{}
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
"onViewerTouchStart");function onViewerTouchMove(e){if(e.touches.length>=2){if(!p){const $=e.touches[0],
F=e.touches[1];p={d:Math.hypot($.clientX-F.clientX,$.clientY-F.clientY)||1,s:z,mx:($.clientX+F.clientX)/
2,my:($.clientY+F.clientY)/2,x:vX,y:vY},s=null}const v=e.touches[0],w=e.touches[1],k=Math.hypot(v.clientX-
w.clientX,v.clientY-w.clientY)||1,_=(v.clientX+w.clientX)/2,C=(v.clientY+w.clientY)/2;z=Math.min(6,Math.
max(1,p.s*k/p.d)),z<=1.0001?(z=1,vX=0,vY=0):(vX=p.x+_-p.mx,vY=p.y+C-p.my);const L=get("image-viewer-\
img");if(!L)return;e.preventDefault(),L.style.transition="none",applyViewerTransform();return}if(p)return;
if(z>1&&s&&!s.active){const v=e.touches[0];if(!v)return;e.preventDefault(),vX=s.dx+v.clientX-s.startX,
vY=s.dy+v.clientY-s.startY,applyViewerTransform();return}if(!s)return;const t=e.touches[0];if(!t)return;
const n=t.clientX-s.startX,i=t.clientY-s.startY,a=Date.now(),r=Math.max(a-s.lastTime,1),l=(t.clientX-
s.lastX)/r;if(s.vx=l*.6+s.vx*.4,s.lastX=t.clientX,s.lastTime=a,s.dx=n,!s.active){if(Math.abs(n)<10&&
Math.abs(i)<10)return;if(Math.abs(n)<Math.abs(i)*1.15){s=null;return}s.active=!0,s.dir=n>0?-1:1,s.adjacent=
getViewerAdjacent(s.dir),s.adjacent||(s.resist=!0)}e.preventDefault();const c=get("image-viewer-img");
if(!c)return;const m=document.querySelector(".viewer-content"),f=m?m.clientWidth:window.innerWidth,b=s.
resist?n*.3:n;c.style.transition="none",c.style.transform=`translateX(${b}px) scale(${1-Math.min(Math.
abs(b)/(f*4),.04)})`,c.style.opacity=String(Math.max(1-Math.min(Math.abs(b)/(f*.45),.55),.4));const y=s.
adjacent;if(y){const v=Number(y.dataset.dir)||0;y.style.transition="none",y.style.transform=`transla\
te(-50%, -50%) translateX(${v*f+n}px) scale(0.97)`,y.style.opacity=String(Math.min(Math.abs(n)/(f*.3),
1))}}o(onViewerTouchMove,"onViewerTouchMove");function onViewerTouchEnd(){if(p){p=null,s=null;return}
if(!s)return;const e=s;if(s=null,!e.active)return;suppressViewerCloseClick=!0,setTimeout(()=>{suppressViewerCloseClick=
!1},120);const t=get("image-viewer-img");if(!t)return;const n=document.querySelector(".viewer-conten\
t"),i=n?n.clientWidth:window.innerWidth,a=i*.22,r=e.dir||(e.dx>0?-1:1),l=window.matchMedia&&window.matchMedia(
"(prefers-reduced-motion: reduce)").matches,c=!e.resist&&(Math.abs(e.dx)>a||Math.abs(e.vx)>.45&&Math.
sign(e.dx)===r),m=e.adjacent;if(!c){if(t.style.transition="transform 0.32s var(--ease-out), opacity \
0.32s var(--ease-out)",t.style.transform="translateX(0) scale(1)",t.style.opacity="1",m){const b=m;m.
style.transition="transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)",m.style.transform=
`translate(-50%, -50%) translateX(${r*i}px) scale(0.97)`,m.style.opacity="0",setTimeout(()=>{b.isConnected&&
b.remove()},340)}return}if(l){finishSwipeNav(r);return}const f=r*i;t.style.transition="transform 0.3\
s var(--ease-out), opacity 0.3s var(--ease-out)",t.style.transform=`translateX(${f}px) scale(0.96)`,
t.style.opacity="0.2",m&&(m.style.transition="transform 0.3s var(--ease-out), opacity 0.3s var(--eas\
e-out)",m.style.transform="translate(-50%, -50%) translateX(0) scale(1)",m.style.opacity="1"),setTimeout(
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
39/image"},{id:"gemini-nano-banana-2.1",implementedAt:"2026-10-06",implementedRank:9810,quickEmoji:"\
\u{1F34C}",name:"Nano Banana 2.1",desc:"High-efficiency image generation and editing with 1K/2K/4K o\
utput, video input, and up to 14 reference images.",price:"In $1.50/1M; Text/Thinking Out $7.50/1M; \
Image Out $30/1M ($0.0336/1K, $0.0504/2K, $0.113/4K image)"},{id:"gemini-3.1-flash-image",implementedAt:"\
2026-08-25",implementedRank:8526,quickEmoji:"\u{1F34C}",name:"Nano Banana 2",desc:"High-efficiency i\
mage generation and editing (stable).",price:"In $0.50/1M; Text/Thinking Out $3.00/1M; Image Out $60\
.00/1M ($0.067/1K image)"},{id:"gemini-3.1-flash-image-preview",implementedAt:"2026-02-26",implementedRank:2860,
name:"Nano Banana 2 (Preview)",desc:"Retired preview retained for chat history compatibility. Use ge\
mini-3.1-flash-image.",price:"In $0.50/1M, Out $0.067/1K image ($60/1M img tokens)",deprecated:!0},{
id:"gemini-3.1-flash-lite-image",implementedAt:"2026-07-01",implementedRank:6020,quickEmoji:"\u{1F34C}",
name:"Nano Banana 2 Lite",desc:"Low-latency Gemini image generation and editing with 1K output.",price:"\
In $0.25/1M; Text/Thinking Out $1.50/1M; Image Out $30/1M ($0.0336/1K image)"},{id:"gemini-3-pro-ima\
ge",implementedAt:"2026-08-25",implementedRank:8525,quickEmoji:"\u{1F34C}",name:"Nano Banana Pro",desc:"\
Professional image generation and editing with 4K output (stable).",price:"In $2.00/1M; Text/Thinkin\
g Out $12.00/1M; Image Out $120.00/1M ($0.134/1K-2K, $0.24/4K)"},{id:"gemini-3-pro-image-preview",implementedAt:"\
2026-01-25",implementedRank:130,name:"Nano Banana Pro (Preview)",desc:"Retired preview retained for \
chat history compatibility. Use gemini-3-pro-image.",price:"In $2.00/1M, Out $0.134 (1K/2K) or $0.24\
 (4K)",deprecated:!0}]},{category:"Gemini Video Generation",icon:"fas fa-clapperboard text-cyan-400",
description:"Gemini video generation models (Veo 3.1 / Omni Flash)",items:[{id:"gemini-omni-1.1-flas\
h",implementedAt:"2026-09-02",implementedRank:9010,quickEmoji:"\u{1F3AC}",name:"Gemini Omni 1.1 Flas\
h",desc:"Fastest multimodal video generation and conversational editing from text, images, video, an\
d audio (native audio in output).",price:"In $1.50/1M (text/image/video/audio); Text Out $9.00/1M; V\
ideo $17.50/1M (\u2248$0.10/sec)"},{id:"gemini-omni-flash",implementedAt:"2026-08-25",implementedRank:8522,
quickEmoji:"\u{1F3AC}",name:"Gemini Omni Flash",desc:"Fast conversational video generation and editi\
ng from text and images.",price:"In $1.50/1M; Text Out $9.00/1M; Video \u2248$0.10/sec"},{id:"veo-3.\
1-generate-preview",implementedAt:"2026-08-25",implementedRank:8521,quickEmoji:"\u{1F3A5}",name:"Veo\
 3.1",desc:"Cinematic video generation with native audio and 4K output.",price:"$0.40/sec (720p/1080\
p), $0.60/sec (4K)"},{id:"veo-3.1-fast-generate-preview",implementedAt:"2026-08-25",implementedRank:8520,
name:"Veo 3.1 Fast",desc:"Low-cost, fast video generation from the Veo 3.1 family.",price:"$0.10/sec\
 (720p), $0.12/sec (1080p)"},{id:"veo-3.1-lite-generate-preview",implementedAt:"2026-08-25",implementedRank:8519,
name:"Veo 3.1 Lite",desc:"High-efficiency, developer-first video generation (no 4K).",price:"$0.05/s\
ec (720p), $0.08/sec (1080p)"}]},{category:"Gemini Music Generation",icon:"fas fa-music text-fuchsia\
-400",description:"Lyria music generation models",items:[{id:"lyria-3.5",implementedAt:"2026-09-05",
implementedRank:9050,quickEmoji:"\u{1F3BC}",name:"Lyria 3.5",desc:"Full-length song generation from \
text or images with vocals, lyrics, and structured arrangements.",price:"See Google AI pricing"},{id:"\
lyria-3-pro-preview",implementedAt:"2026-08-25",implementedRank:8518,quickEmoji:"\u{1F3B5}",name:"Ly\
ria 3 Pro",desc:"Flagship music generation for full-length songs with structural coherence.",price:"\
$0.08 / song"},{id:"lyria-3-clip-preview",implementedAt:"2026-08-25",implementedRank:8517,quickEmoji:"\
\u{1F3B6}",name:"Lyria 3 Clip",desc:"Short musical clips, loops, and previews (30 seconds).",price:"\
$0.04 / song"},{id:"lyria-realtime-exp",implementedAt:"2026-08-25",implementedRank:8516,name:"Lyria \
RealTime",desc:"Experimental realtime music generation with deep melodic control.",price:"Experiment\
al (no vocals)"}]},{category:"Gemini Transcription",icon:"fas fa-microphone text-teal-400",description:"\
Gemini speech-to-text transcription models",items:[{id:"gemini-3.5-transcribe",implementedAt:"2026-0\
8-27",implementedRank:8621,quickEmoji:"\u{1F399}\uFE0F",name:"Gemini 3.5 Transcribe",desc:"Audio-fil\
e speech-to-text with language detection, speaker diarization, word timestamps, and smart formatting\
 (audio file up to 1 hour).",price:"In $2.00/1M (audio), Out $12.00/1M (text)"},{id:"gemini-3.5-tran\
scribe-live",implementedAt:"2026-08-27",implementedRank:8622,quickEmoji:"\u{1F534}",name:"Gemini 3.5\
 Transcribe Live",desc:"Real-time low-latency streaming speech-to-text over the Live API (microphone\
 input, sessions up to 10 minutes).",price:"In $3.50/1M (audio), Out $21.00/1M (text)"}]},{category:"\
OpenAI Image Gen",icon:"fas fa-paint-brush text-purple-400",description:"GPT Image models",items:[{id:"\
gpt-image-2.5-sunburst",implementedAt:"2026-09-09",implementedRank:9320,quickEmoji:"\u{1F31E}",name:"\
GPT-Image-2.5 Sunburst",desc:"Most capable image generation and editing with precision-focused quali\
ty.",price:"Text In $5/1M; Image In $8/1M; Image Out $30/1M"},{id:"gpt-image-2.5-flare",implementedAt:"\
2026-09-09",implementedRank:9321,quickEmoji:"\u{1F525}",name:"GPT-Image-2.5 Flare",desc:"Fast, high-\
quality everyday image generation and editing.",price:"Text In $5/1M; Image In $8/1M; Image Out $30/\
1M"},{id:"gpt-image-2",implementedAt:"2026-04-30",implementedRank:4680,name:"GPT Image 2",desc:"Stat\
e-of-the-art image generation and editing.",price:"Text In $5/1M; Image In $8/1M; Image Out $30/1M"},
{id:"gpt-image-1.5",implementedAt:"2026-03-13",implementedRank:3410,name:"GPT Image 1.5",desc:"Previ\
ous-generation flagship image model.",price:"Text In $5/1M, Text Out $10/1M; Image Out $32/1M"},{id:"\
gpt-image-1",implementedAt:"2026-03-13",implementedRank:3411,name:"GPT Image 1",desc:"Standard quali\
ty.",price:"Text In $5/1M; Image Out $40/1M"},{id:"gpt-image-1-mini",implementedAt:"2026-03-13",implementedRank:3412,
name:"GPT Image 1 Mini",desc:"Faster, lower resolution.",price:"Text In $2/1M; Image In $2.50/1M; Im\
age Out $8/1M"}]},{category:"OpenAI GPT",icon:"fas fa-brain text-green-400",description:"OpenAI's fl\
agship models",items:[{id:"gpt-6-astra",implementedAt:"2026-10-02",implementedRank:10443,quickEmoji:"\
\u{1F31F}",name:"GPT-6 Astra",desc:"Most capable model for demanding reasoning, coding, and professi\
onal work with 1.05M context.",price:"In $10.00/1M, Cached $1.00/1M, Out $50.00/1M (over 272K: In $2\
0.00, Out $75.00)"},{id:"gpt-6.1-sol",implementedAt:"2026-10-02",implementedRank:10442,quickEmoji:"\u{1F506}",
name:"GPT-6.1 Sol",desc:"Near-Astra performance for complex work at a lower cost, with 1.05M context\
.",price:"In $2.00/1M, Cached $0.10/1M, Out $10.00/1M (over 272K: In $4.00, Out $15.00)"},{id:"gpt-6\
-luna",implementedAt:"2026-10-02",implementedRank:10441,quickEmoji:"\u{1F315}",name:"GPT-6 Luna",desc:"\
Most efficient GPT-6 model for focused, high-volume tasks with 1.05M context.",price:"In $0.10/1M, C\
ached $0.01/1M, Out $0.50/1M (over 272K: In $0.20, Out $0.75)"},{id:"gpt-6-sol",implementedAt:"2026-\
10-02",implementedRank:10440,quickEmoji:"\u2600\uFE0F",name:"GPT-6 Sol",desc:"GPT-6 model for comple\
x coding and agentic workflows with 1.05M context.",price:"In $2.00/1M, Cached $0.20/1M, Out $10.00/\
1M (over 272K: In $4.00, Out $15.00)"},{id:"gpt-5.6-sol",implementedAt:"2026-07-31",implementedRank:6550,
quickEmoji:"\u2600\uFE0F",name:"GPT-5.6 Sol",desc:"Frontier reasoning model for complex professional\
 work with 1.05M context.",price:"In $5.00/1M, Cached $0.50/1M, Out $30.00/1M (over 272K: In $10.00,\
 Out $45.00)"},{id:"gpt-5.6-terra",implementedAt:"2026-07-31",implementedRank:6560,quickEmoji:"\u{1F30D}",
name:"GPT-5.6 Terra",desc:"Balanced intelligence and cost for everyday work with 1.05M context.",price:"\
In $2.00/1M, Cached $0.20/1M, Out $12.00/1M (over 272K: In $4.00, Out $18.00)"},{id:"gpt-5.6-luna",implementedAt:"\
2026-07-31",implementedRank:6561,quickEmoji:"\u{1F319}",name:"GPT-5.6 Luna",desc:"Cost-efficient mod\
el for high-volume workloads with 1.05M context.",price:"In $0.20/1M, Cached $0.02/1M, Out $1.20/1M \
(over 272K: In $0.40, Out $1.80)"},{id:"gpt-4o",implementedAt:"2026-06-04",implementedRank:5820,name:"\
GPT-4o",desc:"Multimodal flagship model.",price:"In $2.50/1M, Out $10.00/1M"},{id:"gpt-4o-mini",implementedAt:"\
2026-06-04",implementedRank:5821,name:"GPT-4o mini",desc:"Fast, low-cost model.",price:"In $0.15/1M,\
 Out $0.60/1M"},{id:"gpt-5.5",implementedAt:"2026-04-26",implementedRank:4500,name:"GPT-5.5",desc:"E\
xperimental OpenAI model ID for accounts with access.",price:"In $5.00/1M, Out $30.00/1M"},{id:"gpt-\
5.5-mini",implementedAt:"2026-04-26",implementedRank:4501,name:"GPT-5.5 mini",desc:"Smaller and more\
 cost-efficient GPT-5.5 tier.",price:"Pricing not publicly listed"},{id:"gpt-5.5-nano",implementedAt:"\
2026-04-26",implementedRank:4502,name:"GPT-5.5 nano",desc:"Smallest and fastest GPT-5.5 tier.",price:"\
Pricing not publicly listed"},{id:"gpt-5.5-pro",implementedAt:"2026-04-26",implementedRank:4503,name:"\
GPT-5.5 Pro",desc:"Higher-capacity GPT-5.5 tier for accounts with access.",price:"In $30.00/1M, Out \
$180.00/1M"},{id:"gpt-5.4",implementedAt:"2026-03-08",implementedRank:3150,name:"GPT-5.4",desc:"Expe\
rimental OpenAI model ID for accounts with access.",price:"In $2.50/1M, Out $15.00/1M"},{id:"gpt-5.4\
-mini",implementedAt:"2026-03-08",implementedRank:3151,name:"GPT-5.4 mini",desc:"Smaller and more co\
st-efficient GPT-5.4 tier.",price:"In $0.75/1M, Out $4.50/1M"},{id:"gpt-5.4-nano",implementedAt:"202\
6-03-08",implementedRank:3152,name:"GPT-5.4 nano",desc:"Smallest and fastest GPT-5.4 tier.",price:"I\
n $0.20/1M, Out $1.25/1M"},{id:"gpt-5.4-pro",implementedAt:"2026-03-08",implementedRank:3153,name:"G\
PT-5.4 Pro",desc:"Higher-capacity GPT-5.4 tier for accounts with access.",price:"In $30.00/1M, Out $\
180.00/1M"},{id:"gpt-5.2",implementedAt:"2026-02-15",implementedRank:200,name:"GPT-5.2 (Responses AP\
I)",desc:"Most capable reasoning model.",price:"In $1.75/1M, Out $14.00/1M"},{id:"gpt-5-search-api",
implementedAt:"2026-02-02",implementedRank:740,name:"GPT-5 Search (API)",desc:"Search-optimized mode\
l (Chat Completions).",price:"Model rates + Web search $10/1k calls"},{id:"gpt-5.1",implementedAt:"2\
026-02-05",implementedRank:200,name:"GPT-5.1",desc:"High intelligence.",price:"In $1.25/1M, Out $10.\
00/1M"},{id:"gpt-5-mini",implementedAt:"2026-02-02",implementedRank:770,name:"GPT-5 mini",desc:"Smal\
l and efficient.",price:"In $0.25/1M, Out $2.00/1M"}]},{category:"DeepSeek V4.1 / V4",icon:"fas fa-b\
olt text-cyan-400",description:"DeepSeek's OpenAI-compatible V4.1 Flash and V4 Pro models",items:[{id:"\
deepseek-v4.1-flash",implementedAt:"2026-09-13",implementedRank:9600,quickEmoji:"\u26A1",apiId:"deep\
seek-flash",name:"DeepSeek V4.1 Flash",desc:"V4.1 Flash with Vision, 1M context, 384K output, thinki\
ng, tools, and JSON.",price:"In $0.003 hit/$0.15 miss, Out $0.60 off-peak"},{id:"deepseek-v4-flash-v\
ision-exp",implementedAt:"2026-08-23",implementedRank:8260,name:"DeepSeek V4 Flash Vision Exp",desc:"\
Retired; retained for history.",price:"Retired",deprecated:!0},{id:"deepseek-v4-flash-0731",implementedAt:"\
2026-07-31",implementedRank:6610,name:"DeepSeek V4 Flash",desc:"Retired; retained for history.",price:"\
Retired",deprecated:!0},{id:"deepseek-v4-flash",implementedAt:"2026-04-26",implementedRank:4510,name:"\
DeepSeek V4 Flash Preview",desc:"Retired; retained for history.",price:"Retired",deprecated:!0},{id:"\
deepseek-v4-pro",implementedAt:"2026-04-26",implementedRank:4511,name:"DeepSeek V4 Pro",desc:"V4 Pro\
 with 1M context, 384K output, thinking, tools, and JSON.",price:"In $0.022 hit/$0.66 miss, Out $1.9\
8 off-peak"}]},{category:"Z.AI GLM",icon:"fas fa-brain text-emerald-400",description:"Z.AI chat and \
vision models (Chat Completions API)",items:[{id:"glm-5.3",implementedAt:"2026-09-29",implementedRank:10421,
quickEmoji:"\u{1F9E0}",name:"GLM-5.3",desc:"Flagship coding and agent model; 1M context.",price:"In \
$1.40/1M, Out $4.40/1M"},{id:"glm-5.3-flash",implementedAt:"2026-09-29",implementedRank:10422,name:"\
GLM-5.3-Flash",desc:"Fast multimodal model; image input, 1M context.",price:"In $0.15/1M, Out $0.50/\
1M"},{id:"glm-5.3-flashx",implementedAt:"2026-09-29",implementedRank:10423,name:"GLM-5.3-FlashX",desc:"\
Fast multimodal model; image input, 1M context.",price:"In $0.37/1M, Out $1.25/1M"},{id:"glm-5.2",implementedAt:"\
2026-09-29",implementedRank:10420,name:"GLM-5.2",desc:"Coding model; 1M context.",price:"In $1.40/1M\
, Out $4.40/1M"},{id:"glm-5.1",implementedAt:"2026-09-29",implementedRank:10419,name:"GLM-5.1",desc:"\
Long-running coding tasks; 200K context.",price:"In $1.40/1M, Out $4.40/1M"},{id:"glm-5",implementedAt:"\
2026-09-29",implementedRank:10418,name:"GLM-5",desc:"Reasoning and agentic coding; 200K context.",price:"\
In $1.00/1M, Out $3.20/1M"},{id:"glm-4.7",implementedAt:"2026-09-29",implementedRank:10417,name:"GLM\
-4.7",desc:"Agentic coding; 200K context.",price:"In $0.60/1M, Out $2.20/1M"},{id:"glm-4.7-flashx",implementedAt:"\
2026-09-29",implementedRank:10416,name:"GLM-4.7-FlashX",desc:"Fast agentic coding; 200K context.",price:"\
In $0.07/1M, Out $0.40/1M"},{id:"glm-4.7-flash",implementedAt:"2026-09-29",implementedRank:10415,name:"\
GLM-4.7-Flash",desc:"Free lightweight model; 200K context.",price:"Free"},{id:"glm-4.6",implementedAt:"\
2026-09-29",implementedRank:10414,name:"GLM-4.6",desc:"Coding and general use; 200K context.",price:"\
In $0.60/1M, Out $2.20/1M"},{id:"glm-4.5",implementedAt:"2026-09-29",implementedRank:10413,name:"GLM\
-4.5",desc:"Reasoning model; 128K context.",price:"In $0.60/1M, Out $2.20/1M"},{id:"glm-4.5-x",implementedAt:"\
2026-09-29",implementedRank:10412,name:"GLM-4.5-X",desc:"Fast reasoning model; 128K context.",price:"\
In $2.20/1M, Out $8.90/1M"},{id:"glm-4.5-air",implementedAt:"2026-09-29",implementedRank:10411,name:"\
GLM-4.5-Air",desc:"Cost-effective model; 128K context.",price:"In $0.20/1M, Out $1.10/1M"},{id:"glm-\
4.5-airx",implementedAt:"2026-09-29",implementedRank:10410,name:"GLM-4.5-AirX",desc:"Fast lightweigh\
t model; 128K context.",price:"In $1.10/1M, Out $4.50/1M"},{id:"glm-4.5-flash",implementedAt:"2026-0\
9-29",implementedRank:10409,name:"GLM-4.5-Flash",desc:"Free lightweight model; 200K context.",price:"\
Free"},{id:"glm-4-32b-0414-128k",implementedAt:"2026-09-29",implementedRank:10408,name:"GLM-4-32B-04\
14-128K",desc:"Compact general model; 128K context.",price:"In $0.10/1M, Out $0.10/1M"},{id:"glm-4.6\
v",implementedAt:"2026-09-29",implementedRank:10407,name:"GLM-4.6V",desc:"Image understanding with t\
ools; 128K context.",price:"In $0.30/1M, Out $0.90/1M"},{id:"glm-4.6v-flashx",implementedAt:"2026-09\
-29",implementedRank:10406,name:"GLM-4.6V-FlashX",desc:"Fast image understanding; 128K context.",price:"\
In $0.04/1M, Out $0.40/1M"},{id:"glm-4.6v-flash",implementedAt:"2026-09-29",implementedRank:10405,name:"\
GLM-4.6V-Flash",desc:"Free image understanding; 128K context.",price:"Free"},{id:"glm-4.5v",implementedAt:"\
2026-09-29",implementedRank:10404,name:"GLM-4.5V",desc:"Multimodal reasoning; 64K context.",price:"I\
n $0.60/1M, Out $1.80/1M"}]},{category:"Kimi K3",icon:"fas fa-brain text-violet-400",description:"Mo\
onshot AI's flagship 2.8T-parameter model with 1M context and always-on thinking",items:[{id:"kimi-k\
3",implementedAt:"2026-07-30",implementedRank:6340,quickEmoji:"\u{1F9E0}",name:"Kimi K3",desc:"Alway\
s-reasoning flagship model with 1M context, vision, tool calling.",price:"In $3.00/1M (miss), $0.30/\
1M (hit), Out $15.00/1M"}]},{category:"Mistral Document OCR",icon:"fas fa-file text-orange-300",description:"\
Document OCR (PDF / image / DOCX / PPTX). Not a chat completion model.",items:[{id:"mistral-ocr-4-0",
implementedAt:"2026-08-15",implementedRank:8130,quickEmoji:"\u{1F4C4}",name:"Mistral OCR 4",desc:"Do\
cument AI OCR with markdown, tables, headers/footers, and paragraph bounding boxes. Chat history is \
not sent.",price:"$4 / 1,000 pages ($5 / 1,000 annotated pages)"}]},{category:"Anthropic Claude",icon:"\
fas fa-brain text-orange-400",description:"Anthropic's latest deep reasoning models",items:[{id:"cla\
ude-opus-4-6",implementedAt:"2026-05-01",implementedRank:480,name:"Claude Opus 4.6",desc:"Most capab\
le model for deep reasoning and complex tasks.",price:"In $5.00/1M, Out $25.00/1M"},{id:"claude-sonn\
et-4-6",implementedAt:"2026-05-01",implementedRank:481,name:"Claude Sonnet 4.6",desc:"Excellent bala\
nce of speed and intelligence with adaptive thinking.",price:"In $3.00/1M, Out $15.00/1M"}]},{category:"\
Audio (TTS)",icon:"fas fa-microphone text-red-400",description:"Text-to-Speech models",items:[{id:"g\
emini-3.8-flash-tts",implementedAt:"2026-09-29",implementedRank:10423,quickEmoji:"\u{1F5E3}\uFE0F",name:"\
Gemini 3.8 Flash TTS",desc:"Studio-grade expressive Google TTS with long-form multi-speaker stabilit\
y.",price:"Text In $0.50/1M, Audio Out $9.00/1M"},{id:"gemini-3.8-flash-lite-tts",implementedAt:"202\
6-09-29",implementedRank:10424,name:"Gemini 3.8 Flash-Lite TTS",desc:"Fast, cost-efficient Google TT\
S for high-volume speech.",price:"Text In $0.50/1M, Audio Out $6.00/1M"},{id:"gemini-3.1-flash-tts-p\
review",implementedAt:"2026-04-17",implementedRank:4250,name:"Gemini 3.1 Flash TTS",desc:"Google TTS\
 (Preview).",price:"Text In $1.00/1M, Audio Out $20.00/1M"},{id:"gpt-4o-mini-tts",implementedAt:"202\
6-03-01",implementedRank:250,name:"GPT-4o Mini TTS",desc:"OpenAI TTS.",price:"Text In $0.60/1M, Audi\
o Out $12.00/1M"},{id:"gemini-2.5-flash-preview-tts",implementedAt:"2026-02-10",implementedRank:160,
name:"Gemini 2.5 Flash TTS",desc:"Google TTS (Preview).",price:"Text In $0.50/1M, Audio Out $10.00/1\
M"},{id:"gemini-2.5-pro-preview-tts",implementedAt:"2026-02-10",implementedRank:161,name:"Gemini 2.5\
 Pro TTS",desc:"Google TTS Pro (Preview).",price:"Text In $1.00/1M, Audio Out $20.00/1M"},{id:"googl\
e-tts-studio",implementedAt:"2026-01-20",implementedRank:110,name:"Google TTS (Studio)",desc:"High f\
idelity studio voices.",price:"$160 / 1M chars"},{id:"google-tts-neural",implementedAt:"2026-01-20",
implementedRank:111,name:"Google TTS (Neural2)",desc:"Standard neural voices.",price:"$16 / 1M chars"},
{id:"grok-tts",implementedAt:"2026-05-27",implementedRank:5560,quickEmoji:"\u{1F50A}",name:"Grok TTS",
desc:"xAI Text-to-Speech with expressive voices.",price:"$15.00 / 1M chars"}]},{category:"OpenAI Tra\
nscription",icon:"fas fa-closed-captioning text-emerald-400",description:"Speech-to-text models (aud\
io in / text out)",items:[{id:"gpt-transcribe",implementedAt:"2026-07-29",implementedRank:6330,name:"\
GPT Transcribe",desc:"High-accuracy file and committed-turn transcription.",price:"$0.0045 / minute"},
{id:"gpt-live-transcribe",implementedAt:"2026-07-29",implementedRank:6331,name:"GPT Live Transcribe",
desc:"Low-latency realtime transcription.",price:"$0.017 / minute"}]},{category:"Realtime Audio (STS\
)",icon:"fas fa-headset text-cyan-400",description:"Realtime voice models (audio in / audio out)",items:[
{id:"gpt-live-1",implementedAt:"2026-09-29",implementedRank:10425,quickEmoji:"\u{1F4DE}",name:"GPT-L\
ive 1",desc:"Full-duplex voice conversations with smooth interruption handling; reasoning is delegat\
ed to GPT-5.6 Luna.",price:"$0.05 / minute + backend usage"},{id:"gpt-realtime-2.1",implementedAt:"2\
026-09-29",implementedRank:10421,quickEmoji:"\u{1F399}\uFE0F",name:"OpenAI Realtime 2.1",desc:"Speec\
h-to-speech reasoning model with improved alphanumeric recognition, noise handling and interruptions\
.",price:"Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime-2.1-mini",implementedAt:"2026-09-29",
implementedRank:10422,name:"OpenAI Realtime 2.1 Mini",desc:"Faster, lower-cost realtime voice model.",
price:"Audio In $10/1M, Audio Out $20/1M"},{id:"gpt-realtime-2",implementedAt:"2026-05-11",implementedRank:5080,
name:"OpenAI Realtime 2",desc:"Previous-generation speech-to-speech reasoning model.",price:"Audio I\
n $32/1M, Audio Out $64/1M"},{id:"gpt-realtime-translate",implementedAt:"2026-05-11",implementedRank:5081,
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
e 2.0 (Live)",desc:"xAI streaming speech-to-text.",price:"$0.20 / hr"},{id:"grok-voice-transcribe-2.\
0-file",implementedAt:"2026-10-09",implementedRank:10721,name:"Grok Voice Transcribe 2.0",desc:"xAI \
speech-to-text for recorded clips and files.",price:"$0.10 / hr"},{id:"grok-voice-latest",implementedAt:"\
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
Text In $0.20/1M, Image $0.45/1M"}]},{category:"Ideogram",icon:"fas fa-paint-brush text-pink-400",description:"\
Ideogram image generation with strong text rendering. Ideogram 4.5 also supports multi-turn precise \
editing.",items:[{id:"ideogram-4.5",implementedAt:"2026-10-03",implementedRank:10474,quickEmoji:"\u{1F5BC}\uFE0F",
name:"Ideogram 4.5",desc:"Latest Ideogram model: native 2K generation and precise multi-turn editing\
 that keeps unedited pixels intact.",price:"from $0.008 / image (1K very low) to $0.22 / image (2K h\
igh)"},{id:"ideogram-4.0",implementedAt:"2026-10-03",implementedRank:10473,name:"Ideogram 4.0",desc:"\
Previous-generation Ideogram model with up to 2K output and structured prompts. Text-to-image only.",
price:"Priced per image by rendering speed"},{id:"ideogram-3.0",implementedAt:"2026-10-03",implementedRank:10472,
name:"Ideogram 3.0",desc:"Text-to-image with style types and negative prompts. Text-to-image only.",
price:"Priced per image by rendering speed"},{id:"ideogram-2a",implementedAt:"2026-10-03",implementedRank:10471,
name:"Ideogram 2a",desc:"Fast, lower-cost Ideogram 2 model. Text-to-image only.",price:"Priced per i\
mage by rendering speed"},{id:"ideogram-2.0",implementedAt:"2026-10-03",implementedRank:10470,name:"\
Ideogram 2.0",desc:"Ideogram 2.0 with style types and negative prompts. Text-to-image only.",price:"\
Priced per image by rendering speed"}]},{category:"Grok Imagine",icon:"fas fa-magic text-blue-400",description:"\
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
Key"}:t.startsWith("glm-")?{provider:"zai",keyField:"zai_key",inputId:"set-zai",label:"Z.AI API Key"}:
t.startsWith("kimi")?{provider:"kimi",keyField:"kimi_key",inputId:"set-kimi",label:"Kimi (Moonshot) \
API Key"}:t.startsWith("mistral")?{provider:"mistral",keyField:"mistral_key",inputId:"set-mistral",label:"\
Mistral API Key"}:t.startsWith("ideogram")?{provider:"ideogram",keyField:"ideogram_key",inputId:"set\
-ideogram",label:"Ideogram API Key"}:t.startsWith("claude")?{provider:"anthropic",keyField:"anthropi\
c_key",inputId:"set-anthropic",label:"Anthropic API Key"}:t.startsWith("grok")?{provider:"xai",keyField:"\
xai_key",inputId:"set-xai",label:"xAI (Grok) API Key"}:t.startsWith("google")?{provider:"google",keyField:"\
google_key",inputId:"set-google-key",label:"Google API Key (TTS)"}:{provider:"openai",keyField:"open\
ai_key",inputId:"set-openai",label:"OpenAI API Key"}:null},"getModelProviderInfo"),setModelApiKeyPanelOpen=o(
e=>{const t=get("model-api-keys-panel"),n=get("toggle-model-api-keys-btn");if(!t||!n)return;const i=!!e;
t.classList.toggle("hidden",!i),n.innerText=i?"\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u8A2D\u5B9A\u3092\u9589\u3058\u308B":
"\u30E2\u30C7\u30EB\u5225\u306EAPI\u30AD\u30FC\u3092\u8A2D\u5B9A\u3059\u308B"},"setModelApiKeyPanelO\
pen"),syncModelApiKeyModelOptions=o(()=>{const e=get("model-api-key-model");if(!e)return;const t=e.value||
"";e.innerHTML="";const n=document.createElement("option");n.value="",n.textContent="\u30E2\u30C7\u30EB\u3092\u9078\u629E",
e.appendChild(n),MODELS.forEach(i=>{const a=Array.isArray(i.items)?i.items.filter(l=>!l.deprecated):
[];if(!a.length)return;const r=document.createElement("optgroup");r.label=String(i.category||"Models"),
a.forEach(l=>{const c=String(l.id||"").trim();if(!c)return;const m=document.createElement("option");
m.value=c,m.textContent=`${String(l.name||c)} (${c})`,r.appendChild(m)}),r.children.length>0&&e.appendChild(
r)}),t&&Array.from(e.options).some(a=>a.value===t)&&(e.value=t)},"syncModelApiKeyModelOptions"),renderModelApiKeyList=o(
()=>{const e=get("model-api-key-list");if(!e)return;modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap);
const t=Object.entries(modelApiKeyMap).sort((n,i)=>n[0].localeCompare(i[0]));if(e.innerHTML="",!t.length){
const n=document.createElement("div");n.className="text-[11px] text-gray-500",n.textContent="\u30E2\u30C7\u30EB\u5225\u30AD\u30FC\u306F\
\u672A\u8A2D\u5B9A\u3067\u3059\u3002",e.appendChild(n);return}t.forEach(([n,i])=>{const a=document.createElement(
"div");a.className="flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-9\
00/70 px-3 py-2";const r=document.createElement("div");r.className="min-w-0";const l=document.createElement(
"div");l.className="text-[11px] text-gray-200 truncate",l.textContent=`${getModelNameById(n)} (${n})`;
const c=document.createElement("div");c.className="text-[10px] text-cyan-300 font-mono",c.textContent=
maskApiKeyPreview(i),r.appendChild(l),r.appendChild(c);const m=document.createElement("button");m.type=
"button",m.className="text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-b\
old btn-hover shrink-0",m.textContent="\u524A\u9664",m.onclick=()=>{delete modelApiKeyMap[n],renderModelApiKeyList(),
showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u524A\u9664: ${n}`,"success")},a.appendChild(
r),a.appendChild(m),e.appendChild(a)})},"renderModelApiKeyList"),bindModelApiKeySettingsControls=o(()=>{
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
mi","deepseek","zai","mistral","ideogram","xai","image","video","audio","music","transcription","ocr",
"reasoning","fast","agent","agentic view"],MINIMAL_SLASH_COMMANDS=[{id:"options",label:"/options",description:"\
\uFF0B\u30E1\u30CB\u30E5\u30FC\u3092\u958B\u304F",icon:"fa-plus",kind:"minimal",action:"options"},{id:"\
attach",label:"/attach",description:"\u30D5\u30A1\u30A4\u30EB\u6DFB\u4ED8\u3092\u958B\u304F",icon:"f\
a-paperclip",kind:"minimal",itemKey:"attach"},{id:"voice",label:"/voice",description:"Voice Input\u3092\u958B\u59CB\
\u30FB\u505C\u6B62",icon:"fa-microphone",kind:"minimal",itemKey:"voice-input"},{id:"paste",label:"/p\
aste",description:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u3092\u958B\u304F",icon:"fa-paste",kind:"\
minimal",itemKey:"rich-paste"},{id:"canvas",label:"/canvas",description:"Canvas\u3092\u5207\u308A\u66FF\u3048\u308B\uFF08on / off\uFF09",
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
t-live-transcribe","gpt-live-1","gpt-realtime-2.1","gpt-realtime-2.1-mini","gpt-realtime-2","gpt-rea\
ltime-translate","gpt-realtime-whisper","gpt-realtime-1.5","gpt-realtime","gpt-realtime-mini","gemin\
i-2.5-flash-native-audio-preview-12-2025","gemini-3.1-flash-live-preview","gemini-3.8-live","gemini-\
3.8-live-extended-thinking","gemini-3.5-live-translate-preview","gemini-3.5-transcribe-live","grok-v\
oice-think-fast-2.0","grok-voice-latest","grok-voice-think-fast-1.0","grok-voice-fast-1.0","grok-voi\
ce-agent","grok-voice-transcribe-2.0","grok-voice-transcribe-2.0-file"]),FILE_BASE_URL=CHAT_CONFIG.urls.
serveFileBase,FILE_THUMB_BASE_URL=CHAT_CONFIG.urls.serveFileThumbBase,RICH_PASTE_PDF_SERVER_ROUTE=CHAT_CONFIG.
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
"claude")?"anthropic":t.includes("deepseek")?"deepseek":t.startsWith("glm-")?"zai":t.includes("grok")&&
!t.includes("gpt")?"xai":t.includes("google-tts")?"google":t.includes("gemini")||t.startsWith("veo-")||
t.startsWith("lyria-")||t.startsWith("deep-research-")||t.startsWith("antigravity-")?"gemini":"opena\
i":null},"getModelApiProvider"),PROVIDER_LABELS={openai:"OpenAI",gemini:"Gemini",anthropic:"Anthropi\
c (Claude)",xai:"xAI (Grok)",deepseek:"DeepSeek",zai:"Z.AI (GLM)",google:"Google Cloud"},isPromptCacheEnabled=o(
()=>{const e=get("enable-prompt-cache");return!!(e&&e.checked)},"isPromptCacheEnabled"),getPromptCacheLockedProvider=o(
()=>{if(!isPromptCacheEnabled())return null;const e=get("model-select");return getModelApiProvider(e?
e.value:"")},"getPromptCacheLockedProvider"),updatePromptCacheUi=o(()=>{const e=get("prompt-cache-co\
ntainer"),t=get("enable-prompt-cache"),n=get("model-selector-btn");if(!t)return;const i=!!t.checked;
e&&(e.classList.toggle("ring-1",i),e.classList.toggle("ring-teal-500/50",i),e.classList.toggle("roun\
ded",i),e.classList.toggle("px-1",i)),n&&(i?(n.title="PromptCache\u6709\u52B9: \u540C\u4E00API\u30D7\u30ED\u30D0\u30A4\u30C0\u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\u629E\u53EF\u80FD",
n.classList.add("border-teal-500/60")):(n.title="",n.classList.remove("border-teal-500/60")))},"upda\
tePromptCacheUi"),bindPromptCacheControls=o(()=>{const e=get("enable-prompt-cache");!e||e.dataset.bound===
"1"||(e.dataset.bound="1",e.addEventListener("change",()=>{if(updatePromptCacheUi(),e.checked){const t=getModelApiProvider(
get("model-select")?get("model-select").value:""),n=PROVIDER_LABELS[t]||t||"\u73FE\u5728\u306EAPI";showToast(
`PromptCache \u3092\u6709\u52B9\u5316\u3057\u307E\u3057\u305F\u3002\u4EE5\u964D\u306F ${n} \u4EE5\u5916\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002`,"info",!0)}}))},"bindPromptCacheControls"),getModelMediaSupport=o(
e=>{const t=(e||"").toLowerCase();return t.includes("gemini")?t==="gemini-nano-banana-2.1"?{audio:!1,
video:!0}:t.includes("image")||t.includes("nano")||t.includes("tts")||t.includes("native-audio")||t.
includes("live")?{audio:!1,video:!1}:t.includes("embedding")||t.startsWith("veo-")||t.includes("omni\
-flash")||t.includes("omni-1.1-flash")||t.startsWith("lyria-")?{audio:!1,video:!1}:{audio:!0,video:!0}:
{audio:!1,video:!1}},"getModelMediaSupport"),supportsAudioInputModel=o(()=>getModelMediaSupport(get(
"model-select").value).audio,"supportsAudioInputModel"),supportsVideoInputModel=o(()=>getModelMediaSupport(
get("model-select").value).video,"supportsVideoInputModel"),isImagePath=o(e=>IMAGE_EXTS.includes(getFileExt(
e||"")),"isImagePath"),isAudioPath=o(e=>AUDIO_EXTS.includes(getFileExt(e||"")),"isAudioPath"),isVideoPath=o(
e=>VIDEO_EXTS.includes(getFileExt(e||"")),"isVideoPath"),OPENAI_TTS_VOICES=["alloy","ash","ballad","\
coral","echo","fable","nova","onyx","sage","shimmer","verse","marin","cedar"],GEMINI_TTS_VOICES=["Ze\
phyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe","Enceladus","Iape\
tus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar","Alnilam",
"Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","S\
ulafat"],OPENAI_STS_VOICES=["alloy","ash","ballad","coral","echo","sage","shimmer","verse","marin","\
cedar"],OPENAI_LIVE_VOICES=[...OPENAI_STS_VOICES,"quartz","ripple","vesper","willow","stone","gleam",
"meridian","bossa","tempo","beacon","delta","cinder"],OPENAI_LIVE_STS_MODELS=new Set(["gpt-live-1"]),
OPENAI_REASONING_STS_MODELS=new Set(["gpt-realtime-2","gpt-realtime-2.1","gpt-realtime-2.1-mini"]),GEMINI_INTERACTIONS_TTS_MODELS=new Set(
["gemini-3.8-flash-tts","gemini-3.8-flash-lite-tts"]),GROK_STS_VOICES=["Ara","Rex","Sal","Eve","Leo"],
GROK_TTS_VOICES=["Eve","Ara","Rex","Sal","Leo"],GEMINI_STS_VOICES=["Zephyr","Puck","Charon","Kore","\
Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe","Enceladus","Iapetus","Umbriel","Algieba","Desp\
ina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar","Alnilam","Schedar","Gacrux","Pulcherri\
ma","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"],GROK_PCM_RATES=[8e3,
16e3,22050,24e3,32e3,44100,48e3],isTtsModel=o(()=>get("model-select").value.includes("tts"),"isTtsMo\
del"),isGptImageModel=o(()=>(get("model-select").value||"").includes("gpt-image"),"isGptImageModel"),
isGeminiImageModel=o(()=>isGeminiImageModelKey(get("model-select").value),"isGeminiImageModel"),isMistralOcrModel=o(
e=>{const t=String(e!=null?e:get("model-select")&&get("model-select").value||"").toLowerCase();return t===
"mistral-ocr-4-0"||t==="mistral-ocr-latest"||t.startsWith("mistral-ocr")},"isMistralOcrModel"),isLlmModel=o(
()=>{const e=(get("model-select").value||"").toLowerCase();return isMistralOcrModel(e)||isIdeogramModelKey(
e)||e.includes("tts")||e.includes("transcribe")||e.includes("realtime")||e.includes("voice-agent")||
e.includes("native-audio")||e.includes("live")||e.includes("image")||e.includes("video")||isGeminiVideoModelKey(
e)||isGeminiMusicModelKey(e)||isGeminiEmbeddingModelKey(e)||e.includes("gemini")&&(e.includes("image")||
e.includes("nano"))?!1:e.includes("gpt")||e.includes("gemini")||e.includes("grok")||e.includes("deep\
seek")||e.startsWith("glm-")||e.startsWith("deep-research-")||e.startsWith("antigravity-")},"isLlmMo\
del"),isGrokImageModel=o(()=>{const e=(get("model-select").value||"").toLowerCase();return e.includes(
"grok")&&(e.includes("imagine")||e.includes("image"))&&!e.includes("video")},"isGrokImageModel"),isIdeogramModelKey=o(
e=>String(e||"").toLowerCase().startsWith("ideogram-"),"isIdeogramModelKey"),isIdeogramModel=o(e=>isIdeogramModelKey(
e!=null?e:get("model-select").value),"isIdeogramModel"),IDEOGRAM_IMAGE_FIELDS=["count","aspect","res\
olution","quality","speed","magic","style","negative","seed"],ideogramModelTraits=o(e=>{const t=String(
e||"").toLowerCase();return{edit:t==="ideogram-4.5",quality:t==="ideogram-4.5",speed:t!=="ideogram-4\
.5",resolution:t==="ideogram-4.5"||t==="ideogram-4.0",style:t==="ideogram-3.0"||t==="ideogram-2a"||t===
"ideogram-2.0",negative:t==="ideogram-3.0"||t==="ideogram-2.0",styleOptions:t==="ideogram-3.0"?["aut\
o","general","realistic","design","fiction","stylized"]:["auto","general","realistic","design","rend\
er_3d","anime"]}},"ideogramModelTraits"),isGrokVideoModel=o(()=>{const e=(get("model-select").value||
"").toLowerCase();return e.includes("grok")&&e.includes("video")},"isGrokVideoModel"),isGeminiVideoModelKey=o(
e=>{const t=(e||"").toLowerCase();return t.startsWith("veo-")||t.includes("omni-flash")||t.includes(
"omni-1.1-flash")},"isGeminiVideoModelKey"),isGeminiVideoModel=o(()=>isGeminiVideoModelKey(get("mode\
l-select").value),"isGeminiVideoModel"),isGeminiMusicModelKey=o(e=>(e||"").toLowerCase().startsWith(
"lyria-"),"isGeminiMusicModelKey"),isGeminiMusicModel=o(()=>isGeminiMusicModelKey(get("model-select").
value),"isGeminiMusicModel"),isGeminiEmbeddingModelKey=o(e=>(e||"").toLowerCase().includes("gemini-e\
mbedding"),"isGeminiEmbeddingModelKey"),isGeminiEmbeddingModel=o(()=>isGeminiEmbeddingModelKey(get("\
model-select").value),"isGeminiEmbeddingModel"),isStsModel=o(()=>STS_MODELS.has(get("model-select").
value),"isStsModel"),isTranscriptionModel=o(()=>{const e=get("model-select")?get("model-select").value:
"";return e==="gpt-transcribe"||e==="gpt-live-transcribe"||e==="gpt-realtime-whisper"||e==="grok-voi\
ce-transcribe-2.0-file"},"isTranscriptionModel"),isGeminiLiveModel=o(()=>{const e=get("model-select").
value;return e==="gemini-3.1-flash-live-preview"||e==="gemini-3.8-live"||e==="gemini-3.8-live-extend\
ed-thinking"||e==="gemini-3.5-live-translate-preview"||e==="gemini-3.5-transcribe-live"},"isGeminiLi\
veModel"),isGeminiLiveExtendedThinkingModel=o(()=>get("model-select").value==="gemini-3.8-live-exten\
ded-thinking","isGeminiLiveExtendedThinkingModel"),isGeminiLiveTranslateModel=o(()=>get("model-selec\
t").value==="gemini-3.5-live-translate-preview","isGeminiLiveTranslateModel"),isGeminiLiveTranscribeModel=o(
()=>get("model-select").value==="gemini-3.5-transcribe-live","isGeminiLiveTranscribeModel"),isXaiLiveTranscribeModel=o(
()=>!!get("model-select")&&get("model-select").value==="grok-voice-transcribe-2.0","isXaiLiveTranscr\
ibeModel"),isGeminiRealtimeMusicModel=o(()=>(get("model-select").value||"")==="lyria-realtime-exp","\
isGeminiRealtimeMusicModel"),isLyriaRealtimeModel=o(()=>isGeminiRealtimeMusicModel(),"isLyriaRealtim\
eModel"),isRealtimeSessionModel=o(()=>!(!isStsModel()||isGeminiLiveModel()||isTranscriptionModel()||
get("model-select")&&get("model-select").value==="gpt-realtime-whisper"),"isRealtimeSessionModel"),getStsProvider=o(
e=>{const t=(e||"").toLowerCase();return t.includes("gpt-realtime")||OPENAI_LIVE_STS_MODELS.has(t)||
t==="gpt-transcribe"||t==="gpt-live-transcribe"?"openai":t.includes("grok-voice")?"xai":t.includes("\
gemini")&&(t.includes("native-audio")||t.includes("live"))?"gemini":null},"getStsProvider");function setStsStatus(e,t=!1){
const n=get("sts-status"),i=get("sts-mic-btn");n&&e&&(n.innerText=e),i&&(t?(i.classList.add("bg-red-\
600","animate-pulse"),i.classList.remove("bg-cyan-600")):(i.classList.remove("bg-red-600","animate-p\
ulse"),i.classList.add("bg-cyan-600")))}o(setStsStatus,"setStsStatus");function updateStsUi(){const e=isStsModel(),
t=e&&voiceStudioUiEnabled!==!1,n=get("input-row"),i=get("sts-panel"),a=get("file-preview");e?(n&&n.classList.
add("hidden"),a&&a.classList.add("hidden"),i&&(i.classList.remove("hidden"),i.classList.toggle("voic\
e-dock",t)),!t&&window.VoiceStudio&&window.VoiceStudio.closeIfOpen(),window.VoiceStudio&&window.VoiceStudio.
syncDock(),setStsStatus("Tap to speak",!1)):(n&&n.classList.remove("hidden"),i&&i.classList.add("hid\
den"),window.VoiceStudio&&window.VoiceStudio.closeIfOpen())}o(updateStsUi,"updateStsUi");function updateStsOptions(){
if(!isStsModel())return;const e=get("model-select").value||"",t=getStsProvider(e),n=get("sts-voice"),
i=get("sts-speed-wrap"),a=get("sts-speed"),r=get("sts-speed-label"),l=get("sts-rate-wrap"),c=get("st\
s-rate-in"),m=get("sts-rate-out"),f=get("sts-thinking-wrap"),b=get("sts-note"),y=get("sts-voice-wrap"),
v=get("sts-auto-play-wrap"),w=get("sts-mode-label"),k=isTranscriptionModel()||isGeminiLiveTranscribeModel()||
isXaiLiveTranscribeModel(),_=get("sts-lang-wrap"),C=get("sts-reasoning-wrap");if(C&&C.classList.toggle(
"hidden",!OPENAI_REASONING_STS_MODELS.has(e)),k){w&&(w.textContent="Realtime Speech-to-Text"),y&&y.classList.
add("hidden"),v&&v.classList.add("hidden"),i&&i.classList.add("hidden"),l&&l.classList.add("hidden"),
f&&f.classList.add("hidden"),_&&_.classList.add("hidden");const L=get("sts-transcribe-wrap"),$=get("\
sts-custom-vocab-wrap");L&&L.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),$&&$.classList.
toggle("hidden",!isGeminiLiveTranscribeModel()&&!isXaiLiveTranscribeModel()),b&&(b.textContent=isGeminiLiveTranscribeModel()?
"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F4E\u9045\u5EF6\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM / \u6700\u592710\u5206\uFF09":
isXaiLiveTranscribeModel()?"xAI \u30B9\u30C8\u30EA\u30FC\u30DF\u30F3\u30B0\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM\uFF09":
e==="grok-voice-transcribe-2.0-file"?"xAI \u97F3\u58F0\u30D5\u30A1\u30A4\u30EB\u6587\u5B57\u8D77\u3053\u3057\uFF08\u9332\u97F3\u5358\u4F4D\uFF09":
e==="gpt-live-transcribe"?"\u4F4E\u9045\u5EF6\u30E9\u30A4\u30D6\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
e==="gpt-realtime-whisper"?"\u30B9\u30C8\u30EA\u30FC\u30DF\u30F3\u30B0\u97F3\u58F0\u8A8D\u8B58\u30E2\u30C7\u30EB\u306B\u3088\u308B\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
"\u9AD8\u7CBE\u5EA6\u306A\u30B3\u30DF\u30C3\u30C8\u5358\u4F4D\u306E\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09")}else if(t===
"openai")w&&(w.textContent="Speech-to-Speech Live"),y&&y.classList.remove("hidden"),v&&v.classList.remove(
"hidden"),setSelectOptions(n,OPENAI_STS_VOICES,n.value||"alloy"),i&&i.classList.remove("hidden"),a&&
(a.min=.25,a.max=1.5,a.step=.05,a.value||(a.value=1),Number(a.value)<.25&&(a.value=.25),Number(a.value)>
1.5&&(a.value=1.5)),l&&l.classList.add("hidden"),f&&f.classList.add("hidden"),_&&_.classList.add("hi\
dden"),b&&(b.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A"),OPENAI_REASONING_STS_MODELS.has(
e)&&b&&(b.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A\uFF08Reasoning\u3067\u63A8\u8AD6\u306E\u5F37\u3055\u3092\u6307\u5B9A\uFF09"),
OPENAI_LIVE_STS_MODELS.has(e)&&(setSelectOptions(n,OPENAI_LIVE_VOICES,OPENAI_LIVE_VOICES.includes(n.
value)?n.value:"marin"),i&&i.classList.add("hidden"),b&&(b.textContent="GPT-Live\u306F\u5168\u4E8C\u91CD\u306E\u97F3\u58F0\u4F1A\u8A71\uFF0824kHz PCM\u30FB\
\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC\u30FB\u63A8\u8AD6\u306Fgpt-5.6-luna\u306B\u59D4\u4EFB\uFF09")),
e==="gpt-realtime-translate"&&(w&&(w.textContent="Realtime Translation"),y&&y.classList.add("hidden"),
i&&i.classList.add("hidden"),_&&_.classList.remove("hidden"),b&&(b.textContent="\u8A71\u3057\u305F\u5185\u5BB9\u3092\u9078\u629E\u3057\u305F\u8A00\u8A9E\u3078\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\
\u97F3\u58F0\u7FFB\u8A33\uFF0824kHz PCM\u30FB\u97F3\u58F0\u9078\u629E\u4E0D\u53EF\uFF09"));else if(t===
"xai")w&&(w.textContent="Speech-to-Speech Live"),y&&y.classList.remove("hidden"),v&&v.classList.remove(
"hidden"),setSelectOptions(n,GROK_STS_VOICES,n.value||"Ara"),i&&i.classList.add("hidden"),l&&l.classList.
remove("hidden"),f&&f.classList.add("hidden"),_&&_.classList.add("hidden"),setSelectOptions(c,GROK_PCM_RATES,
Number(c.value||24e3)),setSelectOptions(m,GROK_PCM_RATES,Number(m.value||24e3)),b&&(b.textContent="x\
AI\u306FPCM\u30B5\u30F3\u30D7\u30EB\u30EC\u30FC\u30C8\u5909\u66F4\u53EF");else if(t==="gemini"){w&&(w.
textContent="Speech-to-Speech Live"),y&&y.classList.remove("hidden"),v&&v.classList.remove("hidden"),
setSelectOptions(n,GEMINI_STS_VOICES,n.value||"Kore"),i&&i.classList.add("hidden"),l&&l.classList.add(
"hidden"),f&&f.classList.remove("hidden"),_&&_.classList.add("hidden"),b&&(b.textContent="Gemini Liv\
e\u306F\u97F3\u58F0\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC");const L=get("sts-thinking-level");if(Array.
from(L&&L.options||[]).forEach($=>{$.disabled=!1}),e==="gemini-2.5-flash-native-audio-preview-12-202\
5")f&&f.classList.add("hidden");else if(e==="gemini-3.8-live")f&&f.classList.add("hidden"),b&&(b.textContent=
"Gemini 3.8 Flash Live\u306F\u56FA\u5B9A\u30EC\u30A4\u30C6\u30F3\u30B7\u306ELive API\u30E2\u30C7\u30EB\uFF08Thinking level\u975E\u5BFE\u5FDC\uFF09");else if(e===
"gemini-3.8-live-extended-thinking"){b&&(b.textContent="Gemini 3.8 Live Extended Thinking\u306Flow / medi\
um / high\u306E\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u63A8\u8AD6\u306B\u5BFE\u5FDC");const $=get(
"sts-thinking-level");Array.from($&&$.options||[]).forEach(F=>{F.disabled=F.value==="minimal"}),$&&![
"low","medium","high"].includes($.value)&&($.value="medium")}e==="gemini-3.5-live-translate-preview"&&
(w&&(w.textContent="Realtime Translation"),f&&f.classList.add("hidden"),y&&y.classList.add("hidden"),
_&&_.classList.remove("hidden"),b&&(b.textContent="70\u4EE5\u4E0A\u306E\u8A00\u8A9E\u306B\u5BFE\u5FDC\u3059\u308B\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\uFF08Think\u975E\u5BFE\u5FDC\u30FB\u97F3\u58F0\u9078\u629E\u4E0D\u53EF\uFF09"))}
i&&r&&a&&!i.classList.contains("hidden")&&(r.textContent=`${Number(a.value||1).toFixed(2)}x`)}o(updateStsOptions,
"updateStsOptions");function stsOpt(e){const t=get(e);return e==="sts-auto-play"||e==="sts-auto-rest\
art"?t?!!t.checked:!0:t?!!t.checked:!1}o(stsOpt,"stsOpt");function getStsSilenceMs(){const e=get("st\
s-silence-sec");let t=e?parseFloat(e.value):1.5;return(isNaN(t)||t<.5)&&(t=.5),t>10&&(t=10),Math.round(
t*1e3)}o(getStsSilenceMs,"getStsSilenceMs");function getTtsProvider(e){if(!e)return null;const t=e.toLowerCase();
return t.includes("google-tts")?"google":t.includes("gemini")&&t.includes("tts")?"gemini":t.includes(
"grok-tts")||t.includes("xai-tts")?"xai":t.includes("tts")?"openai":null}o(getTtsProvider,"getTtsPro\
vider");function setSelectOptions(e,t,n){e&&(e.innerHTML="",t.forEach(i=>{const a=document.createElement(
"option");a.value=i.value||i,a.textContent=i.label||i,(i.value||i)===n&&(a.selected=!0),e.appendChild(
a)}))}o(setSelectOptions,"setSelectOptions");function updateTtsUi(){const e=get("model-select").value||
"",t=getTtsProvider(e),n=get("audio-gen-options");if(!n)return;if(!t){n.classList.add("hidden");return}
n.classList.remove("hidden");const i=get("tts-voice"),a=get("tts-voice-custom-wrap"),r=get("tts-voic\
e-custom"),l=get("tts-language-wrap"),c=get("tts-language"),m=get("tts-speed-wrap"),f=get("tts-speed"),
b=get("tts-speed-label"),y=get("tts-speed-note"),v=get("tts-style-wrap"),w=t==="gemini"&&GEMINI_INTERACTIONS_TTS_MODELS.
has(e);v&&v.classList.toggle("hidden",!w),r&&(r.placeholder=w?"voice_... / voicekey_...":"e.g. en-US\
-Wavenet-D"),t==="openai"?(setSelectOptions(i,OPENAI_TTS_VOICES,i.value||"alloy"),a.classList.add("h\
idden"),l.classList.add("hidden"),f&&(f.min=.25,f.max=4,f.step=.05,f.value||(f.value=1),Number(f.value)<
.25&&(f.value=.25),Number(f.value)>4&&(f.value=4),f.disabled=!1),y&&(y.textContent="")):t==="gemini"?
(setSelectOptions(i,GEMINI_TTS_VOICES,i.value||"Kore"),a.classList.toggle("hidden",!w),!w&&r&&(r.value=
""),l.classList.add("hidden"),f&&(f.disabled=!0),y&&(y.textContent=w?"(\u672C\u6587\u306F\u305D\u306E\u307E\u307E\u8AAD\u307F\u4E0A\u3052\u3002\u8A71\u3057\u65B9\u306FStyle\u3067\u6307\u5B9A\u30FB\u901F\u5EA6\u306FS\
tyle\u3067\u8ABF\u6574)":"(Gemini TTS\u306F\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC)")):t==="googl\
e"?(setSelectOptions(i,[{value:"auto",label:"Auto (Studio/Neural2)"},{value:"custom",label:"Custom V\
oice Name"}],i.value||"auto"),i.value==="custom"?a.classList.remove("hidden"):(a.classList.add("hidd\
en"),r&&(r.value="")),l.classList.remove("hidden"),c&&!c.value&&(c.value="ja-JP"),f&&(f.min=.25,f.max=
2,f.step=.05,f.value||(f.value=1),Number(f.value)<.25&&(f.value=.25),Number(f.value)>2&&(f.value=2),
f.disabled=!1),y&&(y.textContent="")):t==="xai"&&(setSelectOptions(i,GROK_TTS_VOICES,i.value||"Eve"),
a.classList.remove("hidden"),l.classList.remove("hidden"),c&&!c.value&&(c.value="ja"),f&&(f.min=.7,f.
max=1.5,f.step=.05,f.value||(f.value=1),Number(f.value)<.7&&(f.value=.7),Number(f.value)>1.5&&(f.value=
1.5),f.disabled=!1),y&&(y.textContent="xAI TTS supports speed 0.7\u20131.5 and speech tags")),f&&b&&
(b.textContent=`${Number(f.value||1).toFixed(2)}x`)}o(updateTtsUi,"updateTtsUi");let mcpServers=[],mcpLoaded=!1,
mcpLoadPromise=null,mcpOauthPopups=[];const MCP_URLS={servers:o(()=>"/api/mcp/servers","servers"),server:o(
e=>`/api/mcp/servers/${encodeURIComponent(e)}`,"server"),test:o(e=>`/api/mcp/servers/${encodeURIComponent(
e)}/test`,"test"),authStart:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}/auth/start`,"authStart"),
authDisconnect:o(e=>`/api/mcp/servers/${encodeURIComponent(e)}/auth/disconnect`,"authDisconnect"),tools:o(
e=>`/api/mcp/servers/${encodeURIComponent(e)}/tools`,"tools"),oauthClient:o(()=>"/api/mcp/oauth-clie\
nt","oauthClient"),permission:o((e,t)=>`/api/mcp/servers/${encodeURIComponent(e)}/tools/${encodeURIComponent(
t)}/permission`,"permission")},mcpGoogleProviderKey="google_workspace",mcpEsc=o(e=>String(e==null?"":
e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]),"mcpEsc"),
mcpStatusMsg=o((e,t,n)=>{const i=get(e);i&&(i.textContent=t||"",i.style.color=n?"#f87171":"#9ca3af")},
"mcpStatusMsg");function mcpAuthStatusLabel(e){return e.auth_type==="none"?"\u8A8D\u8A3C\u4E0D\u8981":
e.auth_status==="connected"?"\u63A5\u7D9A\u6E08\u307F":e.auth_status==="expired"?"\u671F\u9650\u5207\u308C\uFF08\u518D\u8A8D\u8A3C\uFF09":
e.auth_status==="needs_auth"?"\u8A8D\u8A3C\u304C\u5FC5\u8981":"\u672A\u8A8D\u8A3C"}o(mcpAuthStatusLabel,
"mcpAuthStatusLabel");function mcpConnectionStateLabel(e){return e.connection_state==="error"?"\u30A8\u30E9\u30FC":
e.connection_state==="connected"?"\u63A5\u7D9AOK":e.connection_state==="needs_auth"?"\u8A8D\u8A3C\u5F85\u3061":
"\u672A\u63A5\u7D9A"}o(mcpConnectionStateLabel,"mcpConnectionStateLabel");function mcpBadgeClass(e){
return e==="ok"||e==="connected"?"bg-emerald-700/60 text-emerald-100":e==="error"||e==="expired"?"bg\
-red-700/60 text-red-100":e==="auth"?"bg-amber-600/50 text-amber-100":"bg-gray-700 text-gray-300"}o(
mcpBadgeClass,"mcpBadgeClass");function mcpStateBadge(e){const t=mcpAuthStatusLabel(e),n=e.auth_status===
"connected"?"ok":e.auth_status==="expired"?"expired":e.auth_status==="needs_auth"?"auth":"neutral";return`\
<span class="text-[9px] font-bold px-2 py-0.5 rounded-full ${mcpBadgeClass(n)}">${mcpEsc(t)}</span>`}
o(mcpStateBadge,"mcpStateBadge");function mcpOauthProviderLabel(e){return e==="google_workspace"?"Go\
ogle Workspace":e||"OAuth"}o(mcpOauthProviderLabel,"mcpOauthProviderLabel");async function loadMcpServers(e){
if(!get("mcp-server-list")||mcpLoadPromise&&(await mcpLoadPromise,!e))return;if(!e&&mcpLoaded){renderMcpServers();
return}mcpStatusMsg("mcp-status-msg","\u8AAD\u307F\u8FBC\u307F\u4E2D...",!1);let n;n=(async()=>{try{
const i=await apiFetch(MCP_URLS.servers());if(!i.ok){const r=await i.json().catch(()=>({}));mcpStatusMsg(
"mcp-status-msg",r.error||"MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const a=await i.json();mcpServers=a&&Array.isArray(a.servers)?a.servers:[],mcpLoaded=!0,renderMcpServers(),
applyMcpPromptChipUi()}catch(i){mcpStatusMsg("mcp-status-msg","MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(i&&i.message?i.message:i),!0)}finally{mcpLoadPromise===n&&(mcpLoadPromise=null)}})(),mcpLoadPromise=
n,await n}o(loadMcpServers,"loadMcpServers");function mcpHasEnabledServer(){return(mcpServers||[]).some(
e=>!!e.enabled)}o(mcpHasEnabledServer,"mcpHasEnabledServer");function isMcpEnabledForSend(){const e=get(
"mcp-container");if(!e||e.classList.contains("hidden"))return!1;const t=get("enable-mcp");return!!t&&
t.checked}o(isMcpEnabledForSend,"isMcpEnabledForSend");function mcpModelSupported(){try{const e=String(
get("model-select")&&get("model-select").value||"").toLowerCase();return!e||e.startsWith("glm-")?!1:
!!(e.includes("claude")||e.startsWith("kimi")||typeof isLlmModel=="function"&&isLlmModel())}catch{return!1}}
o(mcpModelSupported,"mcpModelSupported");function applyMcpPromptChipUi(){const e=get("mcp-container");
if(!e)return;const t=mcpModelSupported()&&mcpHasEnabledServer();if(e.classList.toggle("hidden",!t),syncMcpAutoSysRows(),
typeof refreshMinimalOptionsIfOpen=="function")try{refreshMinimalOptionsIfOpen()}catch{}}o(applyMcpPromptChipUi,
"applyMcpPromptChipUi");function syncMcpAutoSysRows(){["set","thread"].forEach(e=>{const t=get(`${e}\
-auto-sys-mcp-enabled`);t&&(t.disabled=!0,t.checked=isMcpEnabledForSend())})}o(syncMcpAutoSysRows,"s\
yncMcpAutoSysRows");function renderMcpServers(){const e=get("mcp-server-list"),t=get("mcp-server-cou\
nt");if(!e)return;const n=mcpServers.length;if(t&&(t.textContent=`${n}\u4EF6`),!n){e.innerHTML='<div\
 class="text-[11px] text-gray-600 py-2">\u307E\u3060\u30B5\u30FC\u30D0\u30FC\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u4E0A\u306E\u30AB\u30B9\u30BF\u30E0\u8FFD\u52A0\u30D5\u30A9\u30FC\u30E0\u304B\u3089\u767B\u9332\u3059\u308B\u304B\u3001Google Workspace \u306E\u8A8D\u8A3C\u3092\u3057\u3066\u304F\u3060\u3055\u3044\
\u3002</div>',mcpStatusMsg("mcp-status-msg","");return}const i=mcpServers.map((a,r)=>mcpServerCard(a,
r)).join("");e.innerHTML=i,mcpStatusMsg("mcp-status-msg","")}o(renderMcpServers,"renderMcpServers");
function mcpServerCard(e,t){const n=!!e.is_preset,i=e.auth_type==="oauth",a=e.auth_type==="bearer",r=i||
a,l=i&&!e.oauth_client_registered,c=Number(e.tool_count||0),m=c>0?`${c}\u30C4\u30FC\u30EB`:"\u30C4\u30FC\u30EB\u672A\u53D6\u5F97",
f=mcpStateBadge(e),b=n?'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-\
blue-100">\u30D7\u30EA\u30BB\u30C3\u30C8</span>':'<span class="text-[9px] font-bold px-1.5 py-0.5 ro\
unded bg-purple-700/50 text-purple-100">\u30AB\u30B9\u30BF\u30E0</span>',y=mcpAuthBlock(e),v=i?mcpOauthClientBlock(
e):"";return`
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(e.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${e.enabled?"text-cyan-300":"text-gray-600"}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(e.name)}</span>
                ${b} ${f}
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
            <span class="${c>0?"text-emerald-300":"text-gray-500"}">${m}</span>
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

    ${v}
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
return;const c=(e.value||"").trim(),m=(t.value||"").trim(),f=n.value||"none",b=i?(i.value||"").trim():
"",y=(a&&a.value||"").trim();if(!c||!m){r&&(r.textContent="\u8868\u793A\u540D\u3068URL\u306F\u5FC5\u9808\u3067\u3059",
r.style.color="#f87171");return}l&&(l.disabled=!0),r&&(r.textContent="\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",
r.style.color="#9ca3af");const v={name:c,url:m,auth_type:f,description:b};f==="bearer"&&y&&(v.bearer_token=
y);try{const w=await apiFetch(MCP_URLS.servers(),{method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(v)}),k=await w.json().catch(()=>({}));if(!w.ok){r&&(r.textContent=k.error||
"\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",r.style.color="#f87171");return}r&&(r.textContent=
k.probe&&k.probe.message||"\u8FFD\u52A0\u3057\u307E\u3057\u305F",r.style.color=k.probe&&k.probe.ok?"\
#34d399":"#fbbf24"),e.value="",t.value="",i&&(i.value=""),a&&(a.value=""),mcpLoaded=!1,loadMcpServers(
!0)}catch(w){r&&(r.textContent="\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(w&&w.message?
w.message:w),r.style.color="#f87171")}finally{l&&(l.disabled=!1)}}o(mcpAddCustomServer,"mcpAddCustom\
Server");function bindMcpSettingsUi(){const e=get("mcp-server-list");if(!e)return;const t=get("mcp-a\
dd-server-btn");t&&t.addEventListener("click",mcpAddCustomServer);const n=get("mcp-custom-auth"),i=get(
"mcp-custom-bearer-wrap");if(n&&i){const r=o(()=>{i.classList.toggle("hidden",n.value!=="bearer")},"\
syncBearer");n.addEventListener("change",r),r()}const a=get("mcp-save-google-client-btn");a&&a.addEventListener(
"click",async()=>{const r=get("mcp-google-client-id"),l=get("mcp-google-client-secret"),c=get("mcp-g\
oogle-client-state"),m=r?r.value:"",f=l?l.value:"";if(!m&&!f){c&&(c.textContent="Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
c.style.color="#f87171");return}await mcpSaveOauthClient(mcpGoogleProviderKey,m||"********",f||"****\
****",null)}),e.addEventListener("click",async r=>{const l=r.target.closest("[data-act]");if(!l)return;
const c=l.getAttribute("data-act"),m=l.getAttribute("data-id");if(c==="test"){r.preventDefault(),mcpTestServer(
m);return}if(c==="tools"){r.preventDefault(),mcpLoadTools(m);return}if(c==="auth"||c==="reconnect"){
r.preventDefault(),mcpOpenAuth(m);return}if(c==="disconnect"){r.preventDefault(),mcpDisconnect(m);return}
if(c==="delete"){r.preventDefault(),mcpDeleteServer(m);return}if(c==="edit-oauth"){if(r.preventDefault(),
l.closest("[data-mcp-server]")){const b=l.getAttribute("data-oauth-pk")||"",y=mcpServers.find(k=>String(
k.id)===String(m)),v=document.createElement("div");v.className="mt-2 rounded border border-amber-700\
/50 bg-amber-950/20 p-2",v.innerHTML=`
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
uth" data-id="${m}" data-pk="${mcpEsc(y&&(y.oauth_provider_key||y.slug)||"")}">\u4FDD\u5B58</button>
    </div>`;const w=l.closest("div");w.parentNode.insertBefore(v,w.nextSibling),l.remove()}return}if(c===
"save-oauth"){r.preventDefault();const f=l.getAttribute("data-pk")||"",b=l.closest("[data-mcp-server\
]")||document,y=b.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid'),v=b.querySelectorAll(
'[data-oauth-role="secret"], .mcp-oauth-edit-sec'),w=y.length?y[y.length-1].value:"",k=v.length?v[v.
length-1].value:"";if(!w&&!k){mcpStatusMsg("mcp-status-msg","Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpSaveOauthClient(f,w||"********",k||"********",m);return}if(c==="save-bearer"){r.preventDefault();
const f=document.querySelector(`[data-bearer-id="${m}"]`),b=f?f.value:"";if(!b||b.trim()===""){mcpStatusMsg(
"mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);try{const y=await apiFetch(
MCP_URLS.server(m),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({bearer_token:b})}),
v=await y.json().catch(()=>({}));if(!y.ok){mcpStatusMsg("mcp-status-msg",v.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
eepseek")||r.includes("deepseek"))&&n.push("deepseek"),(i.startsWith("glm-")||l.includes("z.ai"))&&n.
push("zai"),(l.includes("mistral")||i.includes("mistral")||a.includes("mistral")||r.includes("mistra\
l")||i.includes("ocr")||l.includes("ocr"))&&n.push("mistral"),(l.includes("ideogram")||i.includes("i\
deogram")||a.includes("ideogram"))&&n.push("ideogram"),(l.includes("gpt")||l.includes("openai")||i.includes(
"gpt")||a.includes("gpt")||r.includes("openai"))&&n.push("openai"),(l.includes("xai")||l.includes("g\
rok")||i.includes("grok")||a.includes("grok")||r.includes("xai"))&&n.push("xai"),(l.includes("image")||
i.includes("image")||a.includes("image")||r.includes("image"))&&n.push("image"),(l.includes("audio")||
l.includes("music")||l.includes("transcription")||l.includes("speech")||i.includes("tts")||i.includes(
"transcri")||a.includes("tts")||a.includes("transcri")||a.includes("voice")||r.includes("tts")||i.includes(
"realtime")||i.includes("live")||i.includes("voice-agent")||i.includes("native-audio")||a.includes("\
audio")||r.includes("audio")||r.includes("speech-to-text"))&&n.push("audio"),(i.includes("reasoning")||
a.includes("reasoning")||r.includes("reasoning"))&&n.push("reasoning"),(l.includes("deepseek")||i.includes(
"deepseek")||a.includes("deepseek"))&&!n.includes("reasoning")&&n.push("reasoning"),(i.includes("fas\
t")||a.includes("fast")||r.includes("fast")||l.includes("fast"))&&n.push("fast"),(i.includes("deepse\
ek-v4-flash")||l.includes("deepseek")&&a.includes("flash"))&&!n.includes("fast")&&n.push("fast"),(l.
includes("anthropic")||i.includes("claude")||a.includes("claude")||r.includes("anthropic"))&&n.push(
"anthropic"),(l.includes("kimi")||i.includes("kimi")||a.includes("kimi")||r.includes("moonshot"))&&n.
push("kimi"),(l.includes("video")||i.includes("video")||i.startsWith("veo-")||i.includes("omni-")||a.
includes("video")||r.includes("video"))&&n.push("video"),(l.includes("music")||i.startsWith("lyria-")||
a.includes("music")||r.includes("music")||r.includes("song"))&&n.push("music"),(l.includes("transcri\
ption")||i.includes("transcri")||a.includes("transcri")||r.includes("transcription")||r.includes("sp\
eech-to-text"))&&n.push("transcription"),(l.includes("ocr")||i.includes("ocr")||a.includes("ocr")||r.
includes("ocr"))&&n.push("ocr"),(l.includes("agent")||i.includes("agent")||a.includes("agent")||r.includes(
"agentic")||r.includes("computer use")||r.includes("deep research"))&&n.push("agent"),e.agenticView&&
n.push("agentic view"),n}o(getModelTags,"getModelTags");function updateModelTagUi(){const e=get("mod\
el-tag-bar");if(!e)return;e.querySelectorAll(".model-tag-btn").forEach(n=>{const i=n.innerText.trim().
toLowerCase(),a=(i==="all"?"all":i)===activeModelTag;n.classList.toggle("is-active",a)})}o(updateModelTagUi,
"updateModelTagUi");function getModelCapabilitySearchTerms(e){const t=String(e.id||"").toLowerCase(),
n=[],i=t.includes("deepseek"),a=t.includes("tts"),r=t.startsWith("mistral-ocr"),l=a||r||t.startsWith(
"ideogram-")||t.includes("transcribe")||t.includes("realtime")||t.includes("voice-agent")||t.includes(
"native-audio")||t.includes("live")||t.includes("image")||t.includes("video")||t.includes("gemini")&&
t.includes("nano")||t.startsWith("veo-")||t.includes("omni-flash")||t.startsWith("lyria-")||t.includes(
"embedding"),c=!l&&(t.includes("gpt")||t.includes("gemini")||t.includes("grok")||i||t.startsWith("gl\
m-")||t.startsWith("deep-research-")||t.startsWith("antigravity-")),m=o((...b)=>b.forEach(y=>n.push(
y,y.replace(/-/g," "))),"add");if((t==="gemini-nano-banana-2.1"||t.includes("gemini-3.1-flash-image")||
t.includes("gemini-3-pro-image")||t.includes("gemini-2.5-flash-image"))&&m("image generation","image\
 editing"),t==="gemini-nano-banana-2.1"?m("thinking","\u601D\u8003","minimal","medium","high","think\
ing level","video input"):t==="gemini-3.1-flash-lite-image"||t==="gemini-3.1-flash-image"?m("thinkin\
g","\u601D\u8003","minimal","high","thinking level"):t.includes("gemini")&&!l&&(m("thinking","\u601D\u8003",
"thinking level"),t==="gemini-3.8-flash"||t==="gemini-3.7-flash"?m("low","medium","high"):t==="gemin\
i-3.6-flash"?m("medium","high"):t==="gemini-3.5-flash-lite"?m("minimal","medium","high"):t.includes(
"flash")?m("minimal","low","medium","high"):m("low","high")),i&&(m("thinking","\u601D\u8003","reason\
ing","\u63A8\u8AD6","reasoning effort","high"),t!=="deepseek-v4-pro"&&m("low"),t.includes("v4-flash")&&
m("none","max")),c&&(t.includes("gpt-5")||t.includes("o1")||t.includes("o3")||t.includes("grok-4.3")||
t.includes("grok-4.5")||t.includes("grok-4.6")||t.includes("grok-4.20-0309-reasoning")||t.includes("\
grok-build")||t.includes("multi-agent")||t.includes("gpt")&&!a)){m("reasoning","\u63A8\u8AD6","reaso\
ning effort","low","high");const b=t==="gpt-5.6"||t.startsWith("gpt-5.6-"),y=t.startsWith("gpt-6"),v=t.
includes("grok-4.6"),w=t.includes("grok-4.3")||t.includes("grok-4.5")||v||t.includes("grok-4.20-0309\
-reasoning")||t.includes("grok-build")||t.includes("multi-agent")||t.includes("gpt-5")||y||t.includes(
"o1")||t.includes("o3"),k=t.includes("grok-4.3")||t.includes("grok-build")||t.includes("gpt-5")||t===
"gpt-6-sol"||t==="gpt-6-luna"||i;w&&m("medium"),k&&m("none"),(b||y||i)&&m("max"),(v||t.includes("mul\
ti-agent")||b||y)&&m("xhigh")}return t.includes("claude")&&m("thinking","\u601D\u8003","thinking bud\
get","budget"),e.agenticView&&m("agentic view"),[...new Set(n)]}o(getModelCapabilitySearchTerms,"get\
ModelCapabilitySearchTerms");const modelListGroups=[];let modelListBanner=null,modelListEmpty=null,modelListBuilt=!1,
modelListAnimated=!1,modelListRenderFrame=0;function buildModelList(){const e=get("model-list-contai\
ner");!e||modelListBuilt||(e.innerHTML="",modelListBanner=document.createElement("div"),modelListBanner.
className="model-banner hidden",e.appendChild(modelListBanner),MODELS.forEach(t=>{const n=t.items.filter(
l=>!l.deprecated);if(!n.length)return;const i=document.createElement("section");i.className="model-l\
ist-group",i.innerHTML=`
                    <div class="model-group-header">
                        <i class="${t.icon}"></i>
                        <div>
                            <h3 class="model-group-title">${t.category}</h3>
                            <p class="model-group-desc">${t.description}</p>
                        </div>
                    </div>
                    <div class="model-group-grid"></div>
                `;const a=i.querySelector(".model-group-grid"),r=n.map(l=>{const c=document.createElement(
"button"),m=String(l.apiId||l.id||"").trim(),f=l.agenticView?'<span class="inline-flex items-center \
gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold tex\
t-teal-200 whitespace-nowrap" title="Agentic View\u5BFE\u5FDC\uFF1A\u753B\u50CF\u3092\u30AF\u30ED\u30C3\u30D7\u3057\u3066\u518D\u89B3\u5BDF\u3057\u306A\u304C\u3089\u63A8\u8AD6\u3092\u7D99\u7D9A\u3067\u304D\u307E\u3059"><i class="fas fa-eye"\
 aria-hidden="true"></i>Agentic View</span>':"",b=m?`<div class="text-[10px] text-cyan-300/90 mt-1.5\
 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(m)}</\
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
                        ${b}
                        ${y}
                    `,a.appendChild(c),{model:l,button:c,searchText:`${l.name} ${l.id} ${m} ${l.agenticView?
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
forEach(m=>{let f=0;m.entries.forEach(b=>{const y=b.searchText.includes(i)&&(!a||b.provider===a)&&(activeModelTag===
"all"||b.tags.has(activeModelTag));b.button.classList.toggle("hidden",!y),updateModelButtonSelection(
b,l),y&&(f+=1)}),m.element.classList.toggle("hidden",f===0),c+=f}),modelListEmpty.classList.toggle("\
hidden",c!==0),c===0&&(modelListEmpty.textContent=a?`No ${r} models found.`:"No models found."),t.animate&&
!modelListAnimated&&(modelListAnimated=!0,n.classList.add("model-list-animate"))}o(renderModelList,"\
renderModelList");function scheduleModelListRender(e){modelListRenderFrame&&cancelAnimationFrame(modelListRenderFrame),
modelListRenderFrame=requestAnimationFrame(()=>{modelListRenderFrame=0,renderModelList(e)})}o(scheduleModelListRender,
"scheduleModelListRender");function animateModelCategoryChange(){const e=get("model-list-container");
e&&(e.classList.remove("model-category-enter"),e.offsetWidth,e.classList.add("model-category-enter"))}
o(animateModelCategoryChange,"animateModelCategoryChange");let modelListScrollFrame=0;function scrollSelectedModelIntoView(){
const e=get("model-list-container"),t=get("model-select")?get("model-select").value:"",n=modelListGroups.
flatMap(v=>v.entries).find(v=>v.model.id===t);if(!e||!n||n.button.classList.contains("hidden"))return;
const i=e.getBoundingClientRect(),a=n.button.getBoundingClientRect(),r=12;let l=e.scrollTop;if(a.top<
i.top+r?l+=a.top-i.top-r:a.bottom>i.bottom-r&&(l+=a.bottom-i.bottom+r),l=Math.max(0,Math.min(l,e.scrollHeight-
e.clientHeight)),Math.abs(l-e.scrollTop)<1)return;if(modelListScrollFrame&&cancelAnimationFrame(modelListScrollFrame),
window.matchMedia("(prefers-reduced-motion: reduce)").matches){e.scrollTop=l;return}const c=e.scrollTop,
m=l-c,f=performance.now(),b=160,y=o(v=>{const w=Math.min(1,(v-f)/b),k=1-Math.pow(1-w,3);e.scrollTop=
c+m*k,w<1?modelListScrollFrame=requestAnimationFrame(y):modelListScrollFrame=0},"step");modelListScrollFrame=
requestAnimationFrame(y)}o(scrollSelectedModelIntoView,"scrollSelectedModelIntoView");function openModelModal(){
location.pathname!=="/model"&&history.pushState({modal:"model"},"","/model");const e=get("model-sear\
ch");e&&(e.value=""),updateModelTagUi(),syncModelSearchClear(),renderModelList("",{animate:!0}),showModal(
"model-modal"),requestAnimationFrame(()=>requestAnimationFrame(scrollSelectedModelIntoView)),e&&window.
innerWidth>768&&requestAnimationFrame(()=>e.focus({preventScroll:!0}))}o(openModelModal,"openModelMo\
dal"),window.closeModelModal=(e=!1)=>{window._visionPickerActive=!1,hideModal("model-modal"),!e&&location.
pathname==="/model"&&history.back()};function selectModel(e,t){if(window._visionPickerActive){currentVisionModel=
e,window._visionPickerActive=!1,window.closeModelModal(),_syncVisionModelDisplay();return}if(isPromptCacheEnabled()){
const a=getModelApiProvider(get("model-select")?get("model-select").value:""),r=getModelApiProvider(
e);if(a&&r&&a!==r){const l=PROVIDER_LABELS[a]||a,c=PROVIDER_LABELS[r]||r;showToast(`PromptCache \u6709\u52B9\u4E2D\u306F\
\u4ED6API\uFF08${c}\uFF09\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\u3067\u304D\u307E\u305B\u3093\u3002\u73FE\u5728: ${l}`,
"warning",!0);return}}const n=get("model-select");n.value=e,get("model-selector-text").innerText=t,window.
closeModelModal();const i=new Event("change");n.dispatchEvent(i)}o(selectModel,"selectModel");function selectModelById(e){
let t=e;for(const n of MODELS){const i=n.items.find(a=>a.id===e);if(i){t=i.name;break}}selectModel(e,
t)}o(selectModelById,"selectModelById");function populateAiSafeFormFields(e){if(e)try{get("set-defau\
lt-model")&&(get("set-default-model").value=e.default_model||get("set-default-model").value),get("se\
t-default-vision-model")&&(get("set-default-vision-model").value=e.default_vision_model||"gemini-3-f\
lash-preview"),get("set-default-search")&&(get("set-default-search").checked=!!e.default_enable_search),
get("set-default-url-context")&&(get("set-default-url-context").checked=!!e.default_enable_url_context),
get("set-default-maps")&&(get("set-default-maps").checked=!!e.default_enable_maps),get("set-default-\
python")&&(get("set-default-python").checked=!!e.default_enable_python),get("set-default-file-creati\
on")&&(get("set-default-file-creation").checked=!!e.default_enable_file_creation),get("set-default-t\
hinking")&&(get("set-default-thinking").checked=!!e.default_enable_thinking),get("set-default-sys-pr\
ompt")&&(get("set-default-sys-prompt").checked=!!e.default_enable_system_prompt),get("set-default-mc\
p")&&(get("set-default-mcp").checked=e.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get(
"set-default-thinking-level").value=e.default_thinking_level||"high"),get("set-default-thinking-budg\
et")&&(get("set-default-thinking-budget").value=e.default_thinking_budget||4096),get("set-default-re\
asoning-effort")&&(get("set-default-reasoning-effort").value=e.default_reasoning_effort||"medium"),get(
"set-default-safety")&&(get("set-default-safety").value=e.default_safety_setting||"default"),get("sy\
s-prompt-text")&&(get("sys-prompt-text").value=e.system_prompt||""),get("set-global-sys-prompt-enabl\
ed")&&(get("set-global-sys-prompt-enabled").checked=e.system_prompt_enabled!==!1),get("set-apply-glo\
bal-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=e.apply_global_system_prompt!==!1),get(
"set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=e.apply_auto_system_prompt_notices!==
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
test(t):t.startsWith("grok-")?!/(image|video|voice|audio|tts|realtime)/.test(t):t.startsWith("glm-")?
!/(image|audio|tts|transcribe|realtime|video|embedding|ocr)/.test(t):t.startsWith("gemini-")?!/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.
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
"active")})});
