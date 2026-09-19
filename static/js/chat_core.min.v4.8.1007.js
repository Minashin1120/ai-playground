var pi=Object.defineProperty;var a=(e,t)=>pi(e,"name",{value:t,configurable:!0});const get=a(e=>document.getElementById(e),"get"),nativeConsoleLog=typeof console.log=="function"?console.
log.bind(console):function(){},nativeConsoleInfo=typeof console.info=="function"?console.info.bind(console):
nativeConsoleLog;let settingsModalLoaded=!1;const setSettingsSaveEnabled=a(e=>{const t=get("save-set\
tings-btn");t&&(t.disabled=!e,t.classList.toggle("opacity-60",!e),t.classList.toggle("cursor-not-all\
owed",!e),t.setAttribute("title",e?"":"\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u5B8C\u4E86\u5F8C\u306B\u4FDD\u5B58\u3067\u304D\u307E\u3059"))},
"setSettingsSaveEnabled");(function(){const t=a(r=>/(\/files\/thumb\/|\/files\/)/.test(String(r||"")),
"isFileUrl"),n=a(r=>fetch(r,{method:"GET",headers:{Range:"bytes=0-0"},cache:"no-store"}).then(l=>l.status).
catch(()=>-1),"fileUrlStatus");document.addEventListener("load",r=>{const l=r.target;if(!l||l.tagName!==
"IMG"||!l.classList.contains("chat-image"))return;const d=l.closest(".chat-image-frame");d&&(d.dataset.
chatImageState="loaded",d.removeAttribute("aria-busy"))},!0);const i=a((r,l)=>{const d=document.createElement(
"div");return d.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",l?d.innerHTML=
'<i class="fas fa-key" style="font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd\
34d;font-weight:700;line-height:1.3">\u6697\u53F7\u30AD\u30FC\u304C\u4E00\u81F4\u3057\u306A\u3044\u305F\u3081<br>\u95B2\u89A7\u3067\u304D\u307E\u305B\u3093</div>':
d.innerHTML='<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9\
px;color:#9ca3af;font-weight:700">\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093</div>',
r&&d.setAttribute("data-file-name",String(r)),d},"buildWarning"),s=a(r=>{const l=document.createElement(
"div");return l.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",l.innerHTML='\
<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;\
color:#bfdbfe;font-weight:700;line-height:1.3">\u4E00\u6642\u7684\u306B\u6DF7\u96D1\u3057\u3066\u3044\u307E\u3059<br>\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u304F\u3060\u3055\u3044</div>',
r&&l.setAttribute("data-file-name",String(r)),l},"buildBusyWarning"),o=a(r=>String(r||"").split("?")[0].
replace("/files/thumb/","/files/"),"fullFileUrl");document.addEventListener("error",r=>{const l=r.target;
if(!l||l.tagName!=="IMG")return;const d=l.currentSrc||l.src||"";if(!t(d)){const x=l.closest&&l.closest(
".chat-image-frame");if(x){x.dataset.chatImageState="error",x.removeAttribute("aria-busy");const S=x.
querySelector(".chat-image-loading"),T=S&&S.querySelector("span");T&&(T.textContent="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F");
const E=S&&S.querySelector("i");E&&(E.className="fas fa-image")}return}r.stopImmediatePropagation(),
r.preventDefault();const p=String(d).split("?")[0],h=l.getAttribute("data-viewer-filename")||p.split(
"/").pop(),g=a(x=>{const S=l.closest&&l.closest(".chat-image-frame");S&&(S.dataset.chatImageState="f\
ailed",S.removeAttribute("aria-busy"));const T=i(h,!!x);try{l.replaceWith(T)}catch{}},"showWarning"),
y=a(()=>{const x=l.closest&&l.closest(".chat-image-frame");x&&(x.dataset.chatImageState="busy",x.removeAttribute(
"aria-busy"));const S=s(h);try{l.replaceWith(S)}catch{}},"showBusyWarning"),b=a((x,S)=>{const T=l.cloneNode(
!1);T.setAttribute("data-file-retry",String(S));const E=x+(x.includes("?")?"&":"?")+"retry="+Date.now()+
"_"+S;T.setAttribute("src",E);try{l.replaceWith(T)}catch{}},"retryLoad"),w=a(x=>{if(x===429||x===503){
y();return}if(x===409){g(!0);return}if(x===404||x===410||x===403){g(!1);return}const S=parseInt(l.getAttribute&&
l.getAttribute("data-file-retry")||"0",10);if(S<2){b(d,S+1);return}if(d.includes("/files/thumb/")&&!l.
getAttribute("data-file-fallback")){l.setAttribute("data-file-fallback","1"),b(o(d),0);return}g(!1)},
"handleStatus");n(d).then(w).catch(()=>{const x=parseInt(l.getAttribute&&l.getAttribute("data-file-r\
etry")||"0",10);if(x<2){b(d,x+1);return}if(d.includes("/files/thumb/")){y();return}g(!1)})},!0)})();
function buildChatImageHtml(e,t={}){const n=String(e||""),i=String(t.alt||""),s=String(t.title||""),
o=String(t.viewerSrc||n),r=String(t.filename||"");if(n.startsWith("sandbox:"))return`<span class="te\
xt-xs text-gray-500" title="${escapeHtml(n)}">${escapeHtml(i)||"\uFF08\u753B\u50CF\u30C7\u30FC\u30BF\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09"}\
</span>`;const l=s?` title="${escapeHtml(s)}"`:"",d=r?` data-viewer-filename="${escapeHtml(r)}"`:"";
return`<span class="chat-image-frame" data-chat-image-state="loading" aria-busy="true"><span class="\
chat-image-loading" role="status" aria-label="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D"><i class="fas fa-spinner fa-spin" aria-hidde\
n="true"></i><span>\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</span></span><img src="${escapeHtml(
n)}" data-viewer-src="${escapeHtml(o)}" alt="${escapeHtml(i)}"${l}${d} class="chat-image" loading="l\
azy" decoding="async" width="320" height="320"></span>`}a(buildChatImageHtml,"buildChatImageHtml");const isAdminSidebarDebugEnabled=a(
()=>{try{const e=window.CHAT_CONFIG||{};return!!(e.botConfig&&e.botConfig.isAdmin)}catch{return!1}},
"isAdminSidebarDebugEnabled"),ADMIN_SIDEBAR_DEBUG_PREFIX="[admin-sidebar]",adminSidebarDebugEntries=[],
snapshotSidebarHistory=a(e=>{if(!isAdminSidebarDebugEnabled())return null;const t=get("thread-list"),
n=get("sidebar"),i=get("settings-modal"),s=get("history-modal"),o=t?window.getComputedStyle(t):null,
r=n?window.getComputedStyle(n):null,l=t?Array.from(t.querySelectorAll("[data-thread-id]")):[],d=l[0]||
null,p=d?window.getComputedStyle(d):null;let h=null;try{h=typeof threadLoading=="boolean"?threadLoading:
null}catch{h=null}const g={t:Date.now(),reason:String(e||""),path:location.pathname,vw:window.innerWidth,
liteHtml:document.documentElement.classList.contains("performance-lite-mode"),blurHtml:document.documentElement.
classList.contains("performance-blur-disabled"),liquidBody:!!(document.body&&document.body.classList.
contains("liquid-glass-mode")),blurMode:adaptiveBlurPreferenceMode,liteEnabled:adaptiveBlurLiteEnabled,
sidebarClass:n?n.className:null,sidebarDisplay:r?r.display:null,sidebarOpacity:r?r.opacity:null,sidebarVisibility:r?
r.visibility:null,compact:!!(n&&n.classList.contains("compact")),sidebarOpen:!!(n&&n.classList.contains(
"open")),listExists:!!t,listParent:t&&t.parentElement?t.parentElement.id||t.parentElement.className:
null,listClass:t?t.className:null,listChildCount:t?t.children.length:0,listItemCount:l.length,listDisplay:o?
o.display:null,listOpacity:o?o.opacity:null,listVisibility:o?o.visibility:null,listHeight:o?o.height:
null,hideCompact:!!(t&&t.classList.contains("hide-compact")),searchLen:(()=>{const y=get("search-box");
return y?String(y.value||"").length:0})(),firstItemText:d&&d.textContent?d.textContent.trim().slice(
0,40):null,firstItemOpacity:p?p.opacity:null,firstItemDisplay:p?p.display:null,firstItemVisibility:p?
p.visibility:null,firstItemClass:d?d.className:null,settingsHidden:i?i.classList.contains("hidden"):
null,settingsOpen:i?i.classList.contains("modal-open"):null,settingsDisplay:i&&i.style.display||null,
historyHidden:s?s.classList.contains("hidden"):null,threadLoading:h};adminSidebarDebugEntries.push(g),
adminSidebarDebugEntries.length>80&&adminSidebarDebugEntries.shift();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,
e,g)}catch{}return g},"snapshotSidebarHistory"),installAdminSidebarDebugObserver=a(()=>{if(!isAdminSidebarDebugEnabled())
return;const e=get("thread-list");if(!(!e||e.dataset.adminSidebarDebugObserved==="1")){e.dataset.adminSidebarDebugObserved=
"1";try{new MutationObserver(n=>{const i=n.reduce((o,r)=>o+Array.from(r.removedNodes||[]).filter(l=>l&&
l.nodeType===1&&l.getAttribute&&l.getAttribute("data-thread-id")).length,0),s=n.reduce((o,r)=>o+Array.
from(r.addedNodes||[]).filter(l=>l&&l.nodeType===1&&l.getAttribute&&l.getAttribute("data-thread-id")).
length,0);snapshotSidebarHistory(`thread-list-mutated added=${s} removed=${i}`)}).observe(e,{childList:!0,
attributes:!0,attributeFilter:["class","style"]})}catch{}}},"installAdminSidebarDebugObserver");window.
__adminSidebarDebugDump=()=>{if(!isAdminSidebarDebugEnabled())return[];const e=adminSidebarDebugEntries.
slice();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"dump",e)}catch{}return e},window.copyAdminSidebarDebug=
async()=>{if(!isAdminSidebarDebugEnabled())return!1;const e=JSON.stringify(adminSidebarDebugEntries,
null,2);try{return navigator.clipboard&&navigator.clipboard.writeText&&await navigator.clipboard.writeText(
e),nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copied",adminSidebarDebugEntries.length,"entries"),!0}catch{
try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copy-failed",e)}catch{}return!1}};const ADAPTIVE_BLUR_COOKIE="\
adaptive_blur_disabled",ADAPTIVE_LITE_COOKIE="adaptive_lite_mode",ADAPTIVE_BLUR_MODE_COOKIE="adaptiv\
e_blur_mode",readCookieValue=a(e=>{try{const t=document.cookie.split(";").map(n=>n.trim()).find(n=>n.
startsWith(`${e}=`));return t?decodeURIComponent(t.slice(e.length+1)):""}catch{return""}},"readCooki\
eValue"),normalizeAdaptiveBlurMode=a(e=>["enabled","disabled","lite"].includes(e)?e:"auto","normaliz\
eAdaptiveBlurMode"),writeAdaptiveBlurCookie=a((e,t,n=31536e3)=>{try{const i=window.location.protocol===
"https:"?"; Secure":"";document.cookie=`${e}=${encodeURIComponent(t)}; Path=/; Max-Age=${n}; SameSit\
e=Lax${i}`}catch{}},"writeAdaptiveBlurCookie"),adaptiveBlurInteractionCooldownMs=3e3;let adaptiveBlurPreferenceMode=normalizeAdaptiveBlurMode(
readCookieValue(ADAPTIVE_BLUR_MODE_COOKIE)),adaptiveBlurMeasurementActive=!1,adaptiveBlurMeasurementLastAt=0,
adaptiveBlurFallbackEnabled=document.documentElement.classList.contains("performance-blur-disabled"),
adaptiveBlurLiteEnabled=document.documentElement.classList.contains("performance-lite-mode");const syncAdaptiveBlurSettingsUi=a(
()=>{const e=get("set-background-blur-mode"),t=get("background-blur-mode-status");e&&(e.value=adaptiveBlurPreferenceMode),
t&&(adaptiveBlurPreferenceMode==="lite"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="enabled"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u5E38\u306B\u6709\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="disabled"?t.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurLiteEnabled?t.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u8CA0\u8377\u304C\u975E\u5E38\u306B\u9AD8\u3044\u305F\u3081\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurFallbackEnabled?t.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u63CF\u753B\u8CA0\u8377\u3092\u691C\u51FA\u3057\u305F\u305F\u3081\u3001\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
t.textContent="\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u304C\u6709\u52B9\u3067\u3059\u3002\u64CD\u4F5C\u6642\u306E\u63CF\u753B\u304C\u91CD\u3044\u5834\u5408\u306F\u81EA\u52D5\u3067\u7121\u52B9\u5316\u3057\u307E\u3059\u3002")},
"syncAdaptiveBlurSettingsUi"),enableAdaptiveBlurFallback=a(()=>{adaptiveBlurPreferenceMode!=="auto"||
adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.classList.add(
"performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1"),syncAdaptiveBlurSettingsUi())},
"enableAdaptiveBlurFallback"),enableAdaptiveBlurLite=a(()=>{adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
(adaptiveBlurLiteEnabled=!0,adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.
classList.add("performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1")),document.
documentElement.classList.add("performance-lite-mode"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"lite-auto-enabled"),syncAdaptiveBlurSettingsUi(),showToast("\u63CF\u753B\u8CA0\u8377\u304C\u9AD8\u3044\u305F\u3081\u3001\u8EFD\u91CF\u8868\u793A\uFF08\u6700\u5C0F\u8CA0\u8377\uFF09\u3092\u81EA\u52D5\u9069\u7528\u3057\u307E\u3057\u305F\u3002\u30BF\u30C3\u30D7\u3067\u8A2D\u5B9A\u3092\u958B\u304F",
"info",!1,openAdaptiveBlurSettingsFromToast),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"1"))},"en\
ableAdaptiveBlurLite"),openAdaptiveBlurSettingsFromToast=a(()=>{typeof window.openSettingsModal=="fu\
nction"&&window.openSettingsModal();const e=get("set-background-blur-mode"),t=get("tab-display")||get(
"tab-general");if(!(!e||!t)){for(const n of t.children)if(n.contains(e)){jumpToSetting(t.id==="tab-d\
isplay"?"display":"general",n);return}}},"openAdaptiveBlurSettingsFromToast"),applyAdaptiveBlurPreference=a(
e=>{const t=normalizeAdaptiveBlurMode(e);t!==adaptiveBlurPreferenceMode&&(adaptiveBlurPreferenceMode=
t,adaptiveBlurMeasurementActive=!1,adaptiveBlurLiteEnabled=!1,writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,
"",0),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"",0),t==="auto"?writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,
"",0):writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,t),adaptiveBlurFallbackEnabled=t==="disabled"||
t==="lite",adaptiveBlurLiteEnabled=t==="lite",document.documentElement.classList.toggle("performance\
-blur-disabled",adaptiveBlurFallbackEnabled),document.documentElement.classList.toggle("performance-\
lite-mode",adaptiveBlurLiteEnabled),revealPersistentSidebarLists(),snapshotSidebarHistory("blur-pref\
erence-applied:"+t),syncAdaptiveBlurSettingsUi())},"applyAdaptiveBlurPreference"),isSettingsModalOpen=a(
()=>{const e=get("settings-modal");return e?e.classList.contains("modal-open")||e.classList.contains(
"modal-prep")?!0:e.classList.contains("hidden")?!1:e.style.display&&e.style.display!=="none":!1},"is\
SettingsModalOpen"),restoreThreadSearchValue=a((e,t)=>{const n=get("search-box");n&&n.value!==e&&(n.
value=e,clearTimeout(searchTimeout),snapshotSidebarHistory(t||"restored-search-box"))},"restoreThrea\
dSearchValue"),THREAD_SEARCH_INPUT_IDS=["search-box","history-search-box"],isUserInitiatedSearchInput=a(
e=>!!(e&&e.inputType),"isUserInitiatedSearchInput"),unlockThreadSearchInput=a(e=>{e&&e.hasAttribute(
"readonly")&&e.removeAttribute("readonly")},"unlockThreadSearchInput"),markThreadSearchUserEdited=a(
e=>{e&&(e.dataset.userEdited="1")},"markThreadSearchUserEdited"),discardAutofilledThreadSearch=a(e=>{
const t=get("search-box");if(!t||t.dataset.userEdited||!t.value)return;restoreThreadSearchValue("",e||
"cleared-autofill-search-box");const n=get("history-search-box");n&&!n.dataset.userEdited&&(n.value=
"")},"discardAutofilledThreadSearch"),hardenThreadSearchInputs=a(()=>{THREAD_SEARCH_INPUT_IDS.forEach(
e=>{const t=get(e);if(!t)return;const n=a(()=>unlockThreadSearchInput(t),"unlock");t.addEventListener(
"pointerdown",n),t.addEventListener("touchstart",n,{passive:!0}),t.addEventListener("keydown",n),t.addEventListener(
"focus",n)}),discardAutofilledThreadSearch("cleared-autofill-search-box-init"),[0,50,250,1e3].forEach(
e=>{setTimeout(()=>discardAutofilledThreadSearch("cleared-autofill-search-box-"+e+"ms"),e)})},"harde\
nThreadSearchInputs"),revealPersistentSidebarLists=a(()=>{document.querySelectorAll("#thread-list > \
[data-thread-id], #gem-list > .gem-item").forEach(e=>{e.classList.remove("model-list-animate","slide\
-in-animate","fade-in","opacity-0"),e.style.removeProperty("opacity"),e.style.removeProperty("transf\
orm"),e.style.removeProperty("animation"),e.style.removeProperty("animation-delay"),e.style.removeProperty(
"visibility")}),["thread-list","gem-list"].forEach(e=>{const t=get(e);t&&(t.style.removeProperty("op\
acity"),t.style.removeProperty("visibility"))}),snapshotSidebarHistory("reveal-sidebar-lists")},"rev\
ealPersistentSidebarLists"),adaptiveBlurIsBusy=a(()=>!!(activeStreamingBubbleId||document.querySelector(
".modal-overlay.modal-open, .modal-overlay.modal-prep, .modal-overlay.modal-close")),"adaptiveBlurIs\
Busy"),measureInteractionFrames=a((e=!1)=>{if(adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
adaptiveBlurMeasurementActive||document.visibilityState!=="visible")return;if(e)adaptiveBlurMeasurementLastAt=
Date.now();else{const s=Date.now();if(s-adaptiveBlurMeasurementLastAt<adaptiveBlurInteractionCooldownMs||
adaptiveBlurIsBusy())return;adaptiveBlurMeasurementLastAt=s}adaptiveBlurMeasurementActive=!0;const t=[];
let n=0;const i=a(s=>{if(document.visibilityState!=="visible"){adaptiveBlurMeasurementActive=!1;return}
if(n){const g=s-n;g<=200&&t.push(g)}if(n=s,t.length<30){requestAnimationFrame(i);return}adaptiveBlurMeasurementActive=
!1;const o=[...t].sort((g,y)=>g-y),r=Math.min(17.5,Math.max(7,o[Math.floor(o.length*.2)])),l=Math.max(
28,r*1.75),d=Math.max(44,r*2.7),p=t.filter(g=>g>=l).length,h=t.filter(g=>g>=d).length;(p>=5||p>=4&&h>=
2)&&(adaptiveBlurFallbackEnabled?enableAdaptiveBlurLite():enableAdaptiveBlurFallback())},"sampleFram\
e");requestAnimationFrame(i)},"measureInteractionFrames"),measureAdaptiveBlurAfterInteraction=a(()=>{
document.readyState!=="complete"||adaptiveBlurLiteEnabled||requestAnimationFrame(()=>{adaptiveBlurLiteEnabled||
measureInteractionFrames()})},"measureAdaptiveBlurAfterInteraction");document.addEventListener("clic\
k",e=>{const t=e.target instanceof Element?e.target:null;t&&t.closest('button, a, input, select, tex\
tarea, [role="button"], [tabindex]')&&measureAdaptiveBlurAfterInteraction()},!0);const externalScriptLoads=new Map,
loadExternalScript=a((e,t)=>{if(typeof t=="function"&&t())return Promise.resolve();if(externalScriptLoads.
has(e))return externalScriptLoads.get(e);const n=new Promise((i,s)=>{const o=document.createElement(
"script");o.src=e,o.async=!0,o.crossOrigin="anonymous",o.referrerPolicy="no-referrer",o.onload=()=>i(),
o.onerror=()=>s(new Error(`\u30E9\u30A4\u30D6\u30E9\u30EA\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F: ${e}`)),
document.head.appendChild(o)});return externalScriptLoads.set(e,n),n.catch(()=>externalScriptLoads.delete(
e)),n},"loadExternalScript"),ensurePdfLibraries=a(()=>Promise.all([loadExternalScript("/static/vendo\
r/html2canvas-pro-2.3.2.min.js",()=>typeof window.html2canvas=="function"),loadExternalScript("/stat\
ic/vendor/jspdf-2.5.1.umd.min.js",()=>!!(window.jspdf&&window.jspdf.jsPDF))]),"ensurePdfLibraries"),
ensureImageCompression=a(()=>loadExternalScript("https://cdn.jsdelivr.net/npm/browser-image-compress\
ion@2.0.2/dist/browser-image-compression.js",()=>typeof window.imageCompression=="function"),"ensure\
ImageCompression");let webauthnJsonLoad=null;const ensureWebAuthnJson=a(async()=>(window.webauthnJSON||
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
compression_format_only",getCompressionMaxSizeMB=a(()=>parseFloat(localStorage.getItem(COMPRESSION_SIZE_KEY)||
"1.0"),"getCompressionMaxSizeMB"),getCompressionMaxDim=a(()=>parseInt(localStorage.getItem(COMPRESSION_DIM_KEY)||
"1920"),"getCompressionMaxDim"),getCompressionOutputType=a(()=>localStorage.getItem(COMPRESSION_TYPE_KEY)||
"original","getCompressionOutputType"),getCompressionFormatOnly=a(()=>localStorage.getItem(COMPRESSION_FORMAT_ONLY_KEY)===
"true","getCompressionFormatOnly"),IMAGE_EXTENSION_BY_MIME={"image/jpeg":".jpg","image/png":".png","\
image/webp":".webp"},imageFilenameForMime=a((e,t)=>{const n=IMAGE_EXTENSION_BY_MIME[String(t||"").toLowerCase()];
return n?`${String(e||"image").replace(/\.[^./\\]+$/,"")||"image"}${n}`:e||"image"},"imageFilenameFo\
rMime"),convertImageFormatOnly=a(async(e,t)=>{if(!e||!t||t==="original"||t===e.type)return e;await ensureImageCompression();
const n=await window.imageCompression.drawFileInCanvas(e,{fileType:t}),i=n&&n[0],s=n&&n[1];if(!s)throw new Error(
"Image conversion canvas is unavailable");let o;try{typeof s.convertToBlob=="function"?o=await s.convertToBlob(
{type:t,quality:1}):o=await new Promise((r,l)=>{s.toBlob(d=>d?r(d):l(new Error("Image conversion fai\
led")),t,1)})}finally{try{window.imageCompression.cleanupCanvasMemory(s)}catch{}try{i&&typeof i.close==
"function"&&i.close()}catch{}}return new File([o],imageFilenameForMime(e.name,t),{type:t,lastModified:e.
lastModified||Date.now()})},"convertImageFormatOnly"),setCompressionSettings=a((e,t,n,i)=>{localStorage.
setItem(COMPRESSION_SIZE_KEY,e),localStorage.setItem(COMPRESSION_DIM_KEY,t),localStorage.setItem(COMPRESSION_TYPE_KEY,
n),localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY,i)},"setCompressionSettings"),syncCompressionSettingsUi=a(
()=>{const e=get("compression-max-size"),t=get("compression-max-dim"),n=get("compression-output-type"),
i=get("compression-format-only");if(e&&(e.value=getCompressionMaxSizeMB()),t&&(t.value=getCompressionMaxDim()),
n&&(n.value=getCompressionOutputType()),i){i.checked=getCompressionFormatOnly();const h=i.checked;e&&
(e.disabled=h),t&&(t.disabled=h);const g=get("compression-size-wrap"),y=get("compression-dim-wrap");
g&&(g.style.opacity=h?"0.4":"1"),y&&(y.style.opacity=h?"0.4":"1")}const s=a((h,g)=>{get(h)&&get(g)&&
(get(g).value=get(h).value)},"sync");s("gpt-image-size","modal-gpt-image-size"),s("gpt-image-quality",
"modal-gpt-image-quality"),s("gpt-image-format","modal-gpt-image-format"),s("gpt-image-compression",
"modal-gpt-image-compression"),s("gemini-image-aspect","modal-gemini-image-aspect"),s("gemini-image-\
size","modal-gemini-image-size"),s("grok-image-aspect","modal-grok-image-aspect"),s("grok-image-reso\
lution","modal-grok-image-resolution"),s("grok-image-quality","modal-grok-image-quality"),s("ocr-tab\
le-format","modal-ocr-table-format"),s("ocr-pages","modal-ocr-pages");const o=a((h,g)=>{get(h)&&get(
g)&&(get(g).checked=get(h).checked)},"syncChk");o("ocr-extract-header","modal-ocr-extract-header"),o(
"ocr-extract-footer","modal-ocr-extract-footer"),o("ocr-include-blocks","modal-ocr-include-blocks"),
o("ocr-include-images","modal-ocr-include-images");const r=get("model-select").value,l=isGptImageModel(
r),d=isGeminiImageModel(r),p=isGrokImageModel(r);get("modal-gpt-image-options")&&get("modal-gpt-imag\
e-options").classList.toggle("hidden",!l),get("modal-gemini-image-options")&&get("modal-gemini-image\
-options").classList.toggle("hidden",!d),get("modal-grok-image-options")&&get("modal-grok-image-opti\
ons").classList.toggle("hidden",!p),get("modal-mistral-ocr-options")&&get("modal-mistral-ocr-options").
classList.toggle("hidden",!isMistralOcrModel(r))},"syncCompressionSettingsUi"),isGeminiLocalPyDialogEnabled=a(
()=>{const e=localStorage.getItem(GEMINI_LOCAL_PY_DIALOG_KEY);return e===null?!0:e==="1"||e==="true"},
"isGeminiLocalPyDialogEnabled"),setGeminiLocalPyDialogEnabled=a(e=>{localStorage.setItem(GEMINI_LOCAL_PY_DIALOG_KEY,
e?"1":"0")},"setGeminiLocalPyDialogEnabled"),syncGeminiLocalPyDialogSetting=a(()=>{const e=get("set-\
gemini-local-python-dialog");e&&(e.checked=isGeminiLocalPyDialogEnabled())},"syncGeminiLocalPyDialog\
Setting"),normalizeGeminiBackend=a(e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t===
"vertex_ai"||t==="vertex"||t==="vertexai"?"vertex_ai":"gemini_api"},"normalizeGeminiBackend"),normalizeAdminApiKeyMode=a(
e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t==="user_only"||t==="user"||
t==="settings"||t==="user_settings"?"user_only":"env_fallback"},"normalizeAdminApiKeyMode"),syncToggleButtons=a(
(e,t,n)=>{(e||[]).forEach(i=>{const s=i.getAttribute(n)===t;i.classList.toggle("border-cyan-400",s),
i.classList.toggle("bg-cyan-900/30",s),i.classList.toggle("text-white",s),i.classList.toggle("border\
-gray-600",!s),i.classList.toggle("bg-gray-800/70",!s)})},"syncToggleButtons"),syncAdminApiKeyModeUi=a(
()=>{const e=get("set-admin-api-key-mode"),t=get("admin-api-key-mode-note"),n=get("admin-api-key-mod\
e-status"),i=get("admin-api-key-mode-toggle");if(!e)return;const s=normalizeAdminApiKeyMode(e.value);
e.value=s,i&&!i.dataset.bound&&(i.dataset.bound="1",i.querySelectorAll("[data-admin-api-key-mode]").
forEach(o=>{o.addEventListener("click",()=>{e.value=normalizeAdminApiKeyMode(o.getAttribute("data-ad\
min-api-key-mode")),syncAdminApiKeyModeUi()})})),syncToggleButtons(i?i.querySelectorAll("[data-admin\
-api-key-mode]"):[],s,"data-admin-api-key-mode"),t&&(t.textContent=s==="user_only"?"\u901A\u5E38\u30E6\u30FC\u30B6\u30FC\u3068\u540C\u3058\u304F\u3001\u3053\u306E\u753B\u9762\u3067\
\u4FDD\u5B58\u3057\u305FAPI\u30AD\u30FC/Vertex\u8A2D\u5B9A\u306E\u307F\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002":
"\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u3068\u304D\u3060\u3051 .env \u3092\u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u5229\u7528\u3057\u307E\u3059\uFF08\u65E2\u5B9A\uFF09\u3002"),
n&&(n.textContent=s==="user_only"?"\u73FE\u5728: \u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u307F\uFF08\u63A8\u5968: \u8A2D\u5B9A\u5024\u3092\u660E\u793A\u7BA1\u7406\uFF09":
"\u73FE\u5728: .env \u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u6709\u52B9\uFF08\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306A\u3089 .env\uFF09")},
"syncAdminApiKeyModeUi"),ensureGeminiVertexCredentialsField=a(()=>{const e=get("gemini-vertex-settin\
gs");if(!e||get("set-gemini-vertex-credentials-json"))return;const t=document.createElement("div");t.
innerHTML=`
                <label class="text-xs text-gray-500 block">Vertex Service Account JSON (\u4EFB\u610F)</label>
                <textarea id="set-gemini-vertex-credentials-json" class="w-full h-28 bg-gray-800 bor\
der border-gray-600 rounded px-2 py-1 text-[11px] text-white font-mono" placeholder='{"type":"servic\
e_account", ...}'></textarea>
                <div class="text-[10px] text-gray-500 mt-1">\u672A\u5165\u529B\u6642\u306F\u30B5\u30FC\u30D0\u30FC\u5074ADC\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002\u5165\u529B\u3059\u308B\u3068\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u8A2D\u5B9A\u3060\u3051\u3067Ver\
tex\u8A8D\u8A3C\u3067\u304D\u307E\u3059\u3002</div>
            `,e.appendChild(t)},"ensureGeminiVertexCredentialsField"),syncGeminiBackendUi=a(()=>{const e=get(
"set-gemini-backend"),t=get("gemini-vertex-settings"),n=get("gemini-backend-note"),i=get("gemini-bac\
kend-status"),s=get("gemini-backend-toggle");if(!e)return;ensureGeminiVertexCredentialsField();const o=normalizeGeminiBackend(
e.value);e.value=o,s&&!s.dataset.bound&&(s.dataset.bound="1",s.querySelectorAll("[data-gemini-backen\
d]").forEach(r=>{r.addEventListener("click",()=>{e.value=normalizeGeminiBackend(r.getAttribute("data\
-gemini-backend")),syncGeminiBackendUi()})})),syncToggleButtons(s?s.querySelectorAll("[data-gemini-b\
ackend]"):[],o,"data-gemini-backend"),t&&t.classList.toggle("hidden",o!=="vertex_ai"),n&&(n.textContent=
o==="vertex_ai"?"Vertex AI \u3092\u5229\u7528\u3057\u307E\u3059\u3002Project ID / Location \u3092\u8A2D\u5B9A\u3057\u3001ADC \u307E\u305F\u306F Vertex Service Account JSON \u3092\u7528\u610F\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002":"Gemini API \u3092\u5229\u7528\u3057\u307E\u3059\u3002API Key \u3092\u8A2D\u5B9A\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
i&&(i.textContent=o==="vertex_ai"?"\u73FE\u5728: Vertex AI\uFF08Project ID / Location / \u8A8D\u8A3C\u60C5\u5831\u304C\u5FC5\u8981\uFF09":
"\u73FE\u5728: Gemini API\uFF08Gemini API Key \u3092\u4F7F\u7528\uFF09")},"syncGeminiBackendUi"),normalizeHex=a(
e=>{if(!e)return null;let t=String(e).trim();return!t||(t.startsWith("#")||(t=`#${t}`),t.length===4&&
(t=`#${t[1]}${t[1]}${t[2]}${t[2]}${t[3]}${t[3]}`),!/^#[0-9a-fA-F]{6}$/.test(t))?null:t.toLowerCase()},
"normalizeHex"),hexToRgb=a(e=>{const t=e.replace("#",""),n=parseInt(t.slice(0,2),16),i=parseInt(t.slice(
2,4),16),s=parseInt(t.slice(4,6),16);return[n,i,s]},"hexToRgb"),mix=a((e,t,n)=>Math.round(e+(t-e)*n),
"mix"),rgbToHex=a((e,t,n)=>`#${[e,t,n].map(i=>i.toString(16).padStart(2,"0")).join("")}`,"rgbToHex"),
deriveTheme=a(e=>{const[t,n,i]=hexToRgb(e),s=rgbToHex(mix(t,255,.45),mix(n,255,.45),mix(i,255,.45)),
o=rgbToHex(mix(t,255,.7),mix(n,255,.7),mix(i,255,.7)),r=rgbToHex(mix(t,0,.18),mix(n,0,.18),mix(i,0,.18)),
l=rgbToHex(mix(t,0,.32),mix(n,0,.32),mix(i,0,.32));return{base:e,light:s,lighter:o,dark:r,darker:l,rgb:`${t}\
, ${n}, ${i}`}},"deriveTheme"),applyThemeColor=a((e,t=!1)=>{const n=normalizeHex(e)||THEME_DEFAULT,i=deriveTheme(
n),s=document.documentElement;[["--theme-500",i.base],["--theme-600",i.dark],["--theme-700",i.darker],
["--theme-300",i.light],["--theme-200",i.lighter],["--theme-rgb",i.rgb]].forEach(([r,l])=>{s.style.getPropertyValue(
r).trim()!==String(l).trim()&&s.style.setProperty(r,l)}),t&&localStorage.setItem(THEME_STORAGE_KEY,n)},
"applyThemeColor"),applyLightMode=a(e=>{const t=!!e;let n=get("manual-theme-light-css");if(t&&!n){const i=window.
CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.manualLightTheme;if(!i)return;n=document.
createElement("link"),n.id="manual-theme-light-css",n.rel="stylesheet",n.href=i,document.head.appendChild(
n)}else!t&&n&&n.remove()},"applyLightMode"),syncThemeInputs=a(e=>{const t=normalizeHex(e)||THEME_DEFAULT,
n=get("set-theme-color"),i=get("set-theme-color-text");n&&(n.value=t),i&&(i.value=t),document.querySelectorAll(
"#theme-presets .theme-swatch").forEach(o=>{const r=normalizeHex(o.getAttribute("data-color"));o.classList.
toggle("active",r===t)})},"syncThemeInputs"),initThemeFromServer=a(()=>{INITIAL_LIGHT_MODE_ENABLED&&
applyLightMode(!0);const e=normalizeHex(INITIAL_THEME_COLOR);if(e){applyThemeColor(e,!1);return}const t=normalizeHex(
localStorage.getItem(THEME_STORAGE_KEY));applyThemeColor(t||THEME_DEFAULT,!1)},"initThemeFromServer"),
LIQUID_GLASS_SURFACE_SELECTOR=["#sidebar",".composer-dock","body > .flex-1 > header","#top-model-bar",
".modal-panel",".modal-glass-panel",".viewer-toolbar",".viewer-meta","#quote-bar","#slash-command-su\
ggestions","#gem-suggestions","#total-token-bar"].join(","),refreshLiquidGlassSurfaces=a(()=>{document.
querySelectorAll(LIQUID_GLASS_SURFACE_SELECTOR).forEach(e=>{e.classList.add("liquid-glass-surface"),
e.matches(".viewer-toolbar, .viewer-meta")&&e.classList.add("liquid-glass-clear");const t=e.matches(
'[data-liquid-glass-background="none"]')||!!e.closest(".liquid-glass-no-backdrop");e.classList.toggle(
"liquid-glass-no-background",t)})},"refreshLiquidGlassSurfaces"),applyLiquidGlassMode=a(e=>{document.
body&&(document.body.classList.toggle("liquid-glass-mode",!!e),e&&refreshLiquidGlassSurfaces())},"ap\
plyLiquidGlassMode");let pendingLiquidGlassPointer=null,liquidGlassPointerFrame=0,liquidGlassPointerPaintAt=0,
liquidGlassPointerSurface=null,liquidGlassPointerRect=null;const paintLiquidGlassPointer=a(e=>{if(!pendingLiquidGlassPointer||
!document.body||!document.body.classList.contains("liquid-glass-mode")){liquidGlassPointerFrame=0;return}
if(e-liquidGlassPointerPaintAt<30){liquidGlassPointerFrame=requestAnimationFrame(paintLiquidGlassPointer);
return}const t=pendingLiquidGlassPointer;pendingLiquidGlassPointer=null;const n=t.target&&t.target.closest?
t.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;if(!n){liquidGlassPointerFrame=0;return}(n!==liquidGlassPointerSurface||
!liquidGlassPointerRect)&&(liquidGlassPointerSurface=n,liquidGlassPointerRect=n.getBoundingClientRect());
const i=liquidGlassPointerRect;if(i.width&&i.height){const s=Math.max(0,Math.min(100,(t.clientX-i.left)/
i.width*100)),o=Math.max(0,Math.min(100,(t.clientY-i.top)/i.height*100));n.style.setProperty("--glas\
s-light-x",`${s.toFixed(1)}%`),n.style.setProperty("--glass-light-y",`${o.toFixed(1)}%`),liquidGlassPointerPaintAt=
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
t&&t.classList.add("liquid-glass-pressed")},{passive:!0});const releaseLiquidGlassPress=a(e=>{const t=e.
target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;t&&t.classList.remove("liquid-gl\
ass-pressed")},"releaseLiquidGlassPress");document.addEventListener("pointerup",releaseLiquidGlassPress,
{passive:!0}),document.addEventListener("pointercancel",releaseLiquidGlassPress,{passive:!0});let liquidGlassScrollTimer=0;
document.addEventListener("scroll",()=>{!document.body||!document.body.classList.contains("liquid-gl\
ass-mode")||(liquidGlassPointerRect=null,document.body.classList.add("liquid-glass-scrolling"),window.
clearTimeout(liquidGlassScrollTimer),liquidGlassScrollTimer=window.setTimeout(()=>{document.body&&document.
body.classList.remove("liquid-glass-scrolling")},140))},{passive:!0,capture:!0}),window.addEventListener(
"resize",()=>{liquidGlassPointerRect=null},{passive:!0});const MODAL_ANIM_MS=280,formatBytes=a(e=>{if(e==
null)return"0MB";const t=e/(1024*1024);return t<1024?`${t.toFixed(1)}MB`:`${(t/1024).toFixed(2)}GB`},
"formatBytes"),inspectSiteCacheStorage=a(async()=>{const e={cacheCount:0,entryCount:0,totalBytes:0,storageUsageBytes:null,
storageQuotaBytes:null};if("caches"in window)try{const t=await caches.keys();e.cacheCount=t.length;for(const n of t){
const i=await caches.open(n),s=await i.keys();e.entryCount+=s.length;for(const o of s)try{const r=await i.
match(o);if(!r)continue;const l=parseInt(r.headers.get("content-length")||"",10);if(Number.isFinite(
l)&&l>=0)e.totalBytes+=l;else{const d=await r.clone().blob();e.totalBytes+=d.size||0}}catch{}}}catch{}
if(navigator.storage&&navigator.storage.estimate)try{const t=await navigator.storage.estimate();e.storageUsageBytes=
Number(t.usage||0),e.storageQuotaBytes=Number(t.quota||0)}catch{}return e},"inspectSiteCacheStorage"),
loadSiteCacheUsage=a(async()=>{const e=get("site-cache-usage-text"),t=get("site-cache-usage-detail");
if(!(!e&&!t)){e&&(e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D..."),t&&(t.innerText="");try{const n=await inspectSiteCacheStorage(),
i=`\u30AD\u30E3\u30C3\u30B7\u30E5\u4F7F\u7528\u91CF: ${formatBytes(n.totalBytes)} (${n.cacheCount}\u30AD\u30E3\
\u30C3\u30B7\u30E5 / ${n.entryCount}\u4EF6)`;if(n.storageQuotaBytes){const s=Math.min(100,Math.round(
n.totalBytes/n.storageQuotaBytes*100));if(e&&(e.innerText=`${i} / \u4FDD\u5B58\u9818\u57DF\u4E0A\u9650 ${formatBytes(
n.storageQuotaBytes)} (${s}%)`),t){const o=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: \u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
t.innerText=`${o} / \u30D6\u30E9\u30A6\u30B6\u306E\u5B9F\u6E2C\u5024\u3067\u3059`}}else e&&(e.innerText=
i),t&&(t.innerText=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4E0A\u9650\u306F\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u3067\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093")}catch{
e&&(e.innerText="\u30AD\u30E3\u30C3\u30B7\u30E5\u5BB9\u91CF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),
t&&(t.innerText="")}}},"loadSiteCacheUsage");let versionUpdateCachePreferenceSavePromise=Promise.resolve();
const loadStorageUsage=a(async()=>{const e=get("storage-usage-text"),t=get("storage-usage-bar");if(!(!e||
!t)){e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D...";try{const n=await apiFetch("/api/storage",{cache:"\
no-store"});if(!n.ok)throw new Error("HTTP "+n.status);const i=await n.json(),s=Number(i.used_bytes||
0),o=Number(i.limit_bytes||0);if(i.is_unlimited||!o)e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(s)}\
 (\u7121\u5236\u9650)`,t.style.width="0%",t.style.opacity="0.5";else{const r=Math.min(100,Math.round(
s/o*100));e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(s)} / ${formatBytes(o)} (${r}%)`,t.style.width=
`${r}%`,t.style.opacity="1"}}catch{e.innerText="\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
t.style.width="0%",t.style.opacity="0.5"}}},"loadStorageUsage"),clearSiteCacheAndReload=a(async(e,t={})=>{
const{scanFirst:n=!0}=t||{},i=e?e.innerText:"";e&&(e.disabled=!0,e.innerText="\u524A\u9664\u4E2D...");
try{const s=n?await inspectSiteCacheStorage():null;await purgeCaches();const o=s?`\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5 ${formatBytes(
s.totalBytes)} \u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002`:"\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002";
showToast(`${o} \u518D\u8AAD\u307F\u8FBC\u307F\u3057\u307E\u3059\u3002`,"success"),window.setTimeout(
()=>location.reload(),900)}catch{showToast("\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u306E\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{e&&(e.disabled=!1,e.innerText=i||"\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664")}},
"clearSiteCacheAndReload"),syncVersionUpdateCachePreferenceUi=a(()=>{const e=get("version-update-cle\
ar-cache");e&&(e.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate))},"syn\
cVersionUpdateCachePreferenceUi"),saveVersionUpdateCachePreference=a(async e=>{window.CHAT_CONFIG&&(window.
CHAT_CONFIG.clearCacheOnVersionUpdate=!!e);try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clear_cache_on_version_update:!!e})})}catch{}},
"saveVersionUpdateCachePreference");initThemeFromServer(),applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED),
measureInteractionFrames(!0);const modalCloseTimers=new WeakMap,modalOpenFrames=new WeakMap,cancelModalTransitions=a(
e=>{const t=modalCloseTimers.get(e);t&&(clearTimeout(t),modalCloseTimers.delete(e));const n=modalOpenFrames.
get(e);n&&(cancelAnimationFrame(n.first),n.second&&cancelAnimationFrame(n.second),modalOpenFrames.delete(
e))},"cancelModalTransitions"),showModal=a(e=>{const t=get(e);if(!t||t.classList.contains("modal-ope\
n"))return;cancelModalTransitions(t),t.classList.remove("hidden"),t.style.display="flex",t.classList.
remove("modal-close"),t.classList.remove("modal-open"),t.classList.add("modal-prep");const n={first:0,
second:0};n.first=requestAnimationFrame(()=>{n.second=requestAnimationFrame(()=>{modalOpenFrames.delete(
t),t.classList.remove("modal-prep"),t.classList.add("modal-open")})}),modalOpenFrames.set(t,n)},"sho\
wModal");window.showModal=showModal;const hideModal=a((e,t={})=>{const n=get(e);if(!n)return;cancelModalTransitions(
n);const i=!!(t&&t.skipConfirm),s=!!(t&&t.skipReset);if(e==="camera-capture-modal"&&cameraCapturePendingFiles.
length>0&&!i&&!cameraCaptureBusy){attachCameraCapturedFiles();return}if(e==="rich-paste-modal"&&!i&&
hasRichPasteContent()&&!confirm("\u8CBC\u308A\u4ED8\u3051\u305F\u5185\u5BB9\u3092\u7834\u68C4\u3057\u3066\u9589\u3058\u307E\u3059\u304B\uFF1F"))
return;if(e==="marker-modal"&&(markerState.row=null),e==="camera-capture-modal"&&(s||resetCameraCapturePending(),
stopCameraCaptureStream()),!n.classList.contains("modal-open")){n.style.display="none",n.classList.remove(
"modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden");return}n.classList.remove(
"modal-open"),n.classList.add("modal-close");const o=setTimeout(()=>{n.style.display="none",n.classList.
remove("modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden"),modalCloseTimers.delete(
n)},MODAL_ANIM_MS);modalCloseTimers.set(n,o)},"hideModal");window.hideModal=hideModal;const RICH_PASTE_ALLOWED_TAGS=[
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
userSettingsSnapshotPromise=null,richPastePromptSaveTimer=null,richPastePromptPreferenceSyncing=!1;const getRichPasteEditor=a(
()=>get("rich-paste-storage"),"getRichPasteEditor"),getRichPasteCapture=a(()=>get("rich-paste-captur\
e"),"getRichPasteCapture"),getRichPastePrompt=a(()=>get("rich-paste-prompt"),"getRichPastePrompt"),getRichPasteUseDefaultCheckbox=a(
()=>get("rich-paste-use-default"),"getRichPasteUseDefaultCheckbox"),getRichPasteStatus=a(()=>get("ri\
ch-paste-status"),"getRichPasteStatus"),downloadBlob=a((e,t)=>{const n=URL.createObjectURL(e),i=document.
createElement("a");i.href=n,i.download=t,document.body.appendChild(i),i.click(),setTimeout(()=>{document.
body.removeChild(i),URL.revokeObjectURL(n)},100)},"downloadBlob"),getRichPasteEffectivePrompt=a((e=null)=>{
if(e&&e.rich_paste_prompt_use_custom_default){const t=String(e.rich_paste_prompt_default||"").trim();
if(t)return t}return RICH_PASTE_DEFAULT_PROMPT},"getRichPasteEffectivePrompt"),syncRichPastePromptPreferencesUi=a(
(e=null,t={})=>{const n=!!t.preservePrompt,i=getRichPastePrompt(),s=getRichPasteUseDefaultCheckbox();
s&&(s.checked=!!(e&&e.rich_paste_prompt_use_custom_default)),i&&!richPastePromptPreferenceSyncing&&!n&&
(i.value=getRichPasteEffectivePrompt(e))},"syncRichPastePromptPreferencesUi"),cacheUserSettings=a((e,t={})=>(userSettingsSnapshot=
e||null,syncRichPastePromptPreferencesUi(userSettingsSnapshot,t),userSettingsSnapshot),"cacheUserSet\
tings"),SETTINGS_LOAD_TIMEOUT_MS=15e3,fetchSettingsSnapshot=a(async()=>{const e=new AbortController,
t=setTimeout(()=>e.abort(),SETTINGS_LOAD_TIMEOUT_MS);try{const n=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,
{cache:"no-store",signal:e.signal});if(!n.ok)throw new Error("HTTP "+n.status);const i=await n.json();
if(!i||typeof i!="object")throw new Error("Invalid settings response");return cacheUserSettings(i)}finally{
clearTimeout(t)}},"fetchSettingsSnapshot"),ensureUserSettingsSnapshot=a(async()=>userSettingsSnapshot||
(userSettingsSnapshotPromise||(userSettingsSnapshotPromise=fetchSettingsSnapshot().catch(()=>null).finally(
()=>{userSettingsSnapshotPromise=null})),await userSettingsSnapshotPromise),"ensureUserSettingsSnaps\
hot"),saveRichPastePromptPreferences=a(async()=>{const e=getRichPastePrompt(),t=getRichPasteUseDefaultCheckbox();
if(!e||!t)return;const n={rich_paste_prompt_default:e.value||"",rich_paste_prompt_use_custom_default:!!t.
checked};try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(n)}),cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},n),{preservePrompt:!0})}catch{}},"saveRichPastePromptPreferences"),queueRichPastePromptPreferenceSave=a(
()=>{richPastePromptSaveTimer&&clearTimeout(richPastePromptSaveTimer),richPastePromptSaveTimer=setTimeout(
()=>{richPastePromptSaveTimer=null,saveRichPastePromptPreferences()},500)},"queueRichPastePromptPref\
erenceSave"),hasRichPasteContent=a(()=>{const e=getRichPasteEditor();return e?(e.textContent||"").trim()?
!0:!!e.querySelector("img,table,ul,ol,blockquote,h1,h2,h3,h4,h5,h6,pre,code"):!1},"hasRichPasteConte\
nt"),updateRichPasteStatus=a(()=>{const e=getRichPasteEditor(),t=getRichPasteStatus();if(!t||!e)return;
const n=(e.innerText||"").trim();if(!n){t.textContent="\u307E\u3060\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002";
return}const i=e.querySelectorAll("img").length,s=e.querySelectorAll("table").length,o=e.querySelectorAll(
"a").length,r=e.querySelectorAll("h1,h2,h3,h4,h5,h6").length;t.textContent=`${n.length} \u6587\u5B57 / \u753B\u50CF ${i}\
 / \u8868 ${s} / \u30EA\u30F3\u30AF ${o} / \u898B\u51FA\u3057 ${r}`},"updateRichPasteStatus"),focusRichPasteEditor=a(
()=>{const e=getRichPasteCapture();if(!e)return;e.focus(),e.value=e.value||"",window.getSelection&&window.
getSelection()&&e.select&&e.select()},"focusRichPasteEditor"),clearRichPasteEditor=a((e=!0)=>{const t=getRichPasteEditor();
t&&(t.innerHTML="");const n=getRichPasteCapture();if(n&&(n.value=""),!e){const i=getRichPastePrompt();
i&&(i.value=RICH_PASTE_DEFAULT_PROMPT)}updateRichPasteStatus()},"clearRichPasteEditor"),sanitizeRichPasteStyle=a(
e=>{if(!e)return"";const t=[];return String(e).split(";").forEach(n=>{const i=n.trim();if(!i)return;
const s=i.indexOf(":");if(s<=0)return;const o=i.slice(0,s).trim().toLowerCase(),r=i.slice(s+1).trim();
if(!RICH_PASTE_SAFE_STYLE_PROPS.has(o)||!r||r.length>1e3)return;const l=r.toLowerCase();l.includes("\
url(")||l.includes("expression(")||l.includes("javascript:")||l.includes("@import")||l.includes("beh\
avior:")||l.includes("-moz-binding")||l.includes("var(")||l.includes("env(")||t.push(`${o}: ${r}`)}),
t.join("; ")},"sanitizeRichPasteStyle");let richPasteColorCanvasContext=null;const parseRichPasteCssColor=a(
e=>{const t=String(e||"").trim();if(!t||t==="inherit"||t==="currentcolor"||t==="transparent"||window.
CSS&&typeof window.CSS.supports=="function"&&!window.CSS.supports("color",t))return null;try{if(!richPasteColorCanvasContext){
const s=document.createElement("canvas");s.width=1,s.height=1,richPasteColorCanvasContext=s.getContext(
"2d",{willReadFrequently:!0})}const n=richPasteColorCanvasContext;if(!n)return null;n.clearRect(0,0,
1,1),n.fillStyle="rgba(1, 2, 3, 0.004)",n.fillStyle=t,n.fillRect(0,0,1,1);const i=n.getImageData(0,0,
1,1).data;return!i||i[3]===0?null:{r:i[0],g:i[1],b:i[2],a:i[3]/255}}catch{return null}},"parseRichPa\
steCssColor"),richPasteColorLuminance=a(e=>{if(!e)return 0;const t=a(n=>{const i=Math.max(0,Math.min(
255,Number(n)||0))/255;return i<=.04045?i/12.92:Math.pow((i+.055)/1.055,2.4)},"channel");return .2126*
t(e.r)+.7152*t(e.g)+.0722*t(e.b)},"richPasteColorLuminance"),richPasteColorContrast=a((e,t)=>{const n=richPasteColorLuminance(
e),i=richPasteColorLuminance(t);return(Math.max(n,i)+.05)/(Math.min(n,i)+.05)},"richPasteColorContra\
st"),richPasteColorCss=a(e=>e?`rgb(${Math.round(e.r)}, ${Math.round(e.g)}, ${Math.round(e.b)})`:"","\
richPasteColorCss"),makeRichPasteTheme=a((e,t)=>{const n=richPasteColorLuminance(e)<.32;let i=t;return(!i||
richPasteColorContrast(e,i)<3)&&(i=n?{r:244,g:244,b:245,a:1}:{r:17,g:24,b:39,a:1}),{mode:n?"dark":"l\
ight",background:richPasteColorCss(e),foreground:richPasteColorCss(i),muted:n?"rgb(161, 161, 170)":"\
rgb(100, 116, 139)",border:n?"rgb(63, 63, 70)":"rgb(203, 213, 225)",surface:n?"rgb(33, 33, 33)":"rgb\
(248, 250, 252)",quote:n?"rgb(39, 39, 42)":"rgb(255, 249, 235)",link:n?"rgb(125, 211, 252)":"rgb(15,\
 118, 110)"}},"makeRichPasteTheme"),detectRichPasteTheme=a(e=>{const t={r:255,g:255,b:255,a:1},n={r:17,
g:24,b:39,a:1},i=document.createElement("template");if(i.innerHTML=String(e||""),!i.content.querySelector(
"*"))return makeRichPasteTheme(t,n);const s=document.createElement("div");s.setAttribute("aria-hidde\
n","true"),s.style.position="fixed",s.style.left="-100000px",s.style.top="0",s.style.width="794px",s.
style.visibility="hidden",s.style.pointerEvents="none",s.style.color="#111827",s.style.background="t\
ransparent",s.appendChild(i.content.cloneNode(!0)),document.body.appendChild(s);try{const o=[s,...Array.
from(s.querySelectorAll("*")).slice(0,5e3)],r=[],l=new Map;let d=0;const p=a(w=>Array.from(w.childNodes||
[]).reduce((x,S)=>S&&S.nodeType===Node.TEXT_NODE?x+String(S.textContent||"").replace(/\s+/g," ").trim().
length:x,0),"directTextLength");o.forEach(w=>{if(!w||w===s||!w.style)return;const x=window.getComputedStyle(
w),S=p(w);if(S>0){const E=parseRichPasteCssColor(x.color);if(E&&E.a>=.5){const F=richPasteColorCss(E),
J=l.get(F)||{color:E,weight:0};J.weight+=S,l.set(F,J),d+=S}}if(!!(String(w.style.backgroundColor||"").
trim()||String(w.style.background||"").trim())){const E=parseRichPasteCssColor(x.backgroundColor);if(E&&
E.a>=.72){const F=String(w.textContent||"").replace(/\s+/g," ").trim().length;r.push({color:E,weight:Math.
max(1,F)})}}});const h=Array.from(l.values()).sort((w,x)=>x.weight-w.weight),g=h.length?h[0].color:null,
y=h.reduce((w,x)=>w+(richPasteColorLuminance(x.color)>=.6?x.weight:0),0);r.sort((w,x)=>x.weight-w.weight);
let b=r.length?r[0].color:null;return b||(b=d>0&&y/d>=.55?{r:11,g:11,b:12,a:1}:t),makeRichPasteTheme(
b,g||n)}catch{return makeRichPasteTheme(t,n)}finally{s.parentNode&&s.parentNode.removeChild(s)}},"de\
tectRichPasteTheme"),prepareRichPastePdfClone=a((e,t)=>{if(!e)return;const n=e.head||e.querySelector(
"head");n&&Array.from(n.querySelectorAll('link[rel="stylesheet"]')).forEach(i=>{try{i.remove()}catch{}}),
e.body&&(e.body.style.margin="0",e.body.style.background=t.background,e.body.style.color=t.foreground)},
"prepareRichPastePdfClone"),normalizeRichPasteTree=a(e=>{!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("*").forEach(t=>{if(!t||!t.getAttribute||!t.parentNode)return;const n=String(t.tagName||
"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(n)){t.remove();return}t.removeAttribute("class"),t.removeAttribute(
"id"),t.removeAttribute("role"),t.removeAttribute("aria-label"),n==="img"&&(t.setAttribute("loading",
"eager"),t.setAttribute("decoding","sync"),t.removeAttribute("srcset"),t.removeAttribute("sizes"));const i=t.
getAttribute("style");if(i){const s=sanitizeRichPasteStyle(i);s?t.setAttribute("style",s):t.removeAttribute(
"style")}})},"normalizeRichPasteTree"),extractRichPasteArticleHtml=a(e=>{const n=new DOMParser().parseFromString(
String(e||""),"text/html");if(!n.body)return"";const i=(n.body.textContent||"").replace(/\s+/g," ").
trim().length,s=n.body.querySelectorAll("*").length;if(i<1e3||s<120)return n.body.innerHTML;const r=[
...Array.from(n.body.querySelectorAll("article")),...Array.from(n.body.querySelectorAll("main")),...Array.
from(n.body.querySelectorAll('[role="main"],[role="article"]'))].filter(d=>(d.textContent||"").replace(
/\s+/g," ").trim().length>=i*.65);r.sort((d,p)=>{const h=+!!p.querySelector("h1")-+!!d.querySelector(
"h1");return h||d.querySelectorAll("*").length-p.querySelectorAll("*").length});const l=r[0]||null;return l?
l.outerHTML:n.body.innerHTML},"extractRichPasteArticleHtml"),sanitizeRichPasteHtml=a(e=>{if(!window.
DOMPurify||typeof window.DOMPurify.sanitize!="function"){const s=new DOMParser().parseFromString(String(
e||""),"text/html");return escapeHtml(s.body?s.body.textContent:"")}let t=extractRichPasteArticleHtml(
e),n=window.DOMPurify.sanitize(t||"",{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,
KEEP_CONTENT:!0});if((!n||n.trim()==="")&&e&&e.trim()!==""&&(n=window.DOMPurify.sanitize(e,{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,
ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,KEEP_CONTENT:!0})),!n)return"";const i=document.createElement("\
template");return i.innerHTML=n,normalizeRichPasteTree(i.content),i.innerHTML},"sanitizeRichPasteHtm\
l"),normalizeRichPastePrintHtml=a(e=>{const t=document.createElement("template");t.innerHTML=String(
e||"");const n=Array.from(t.content.querySelectorAll("*")),i=n.reduce((d,p)=>{const h=String(p.style&&
p.style.display||"").trim().toLowerCase();return d+(["flex","inline-flex","grid","inline-grid"].includes(
h)?1:0)},0),s=n.reduce((d,p)=>{if(!p||!p.style||!["article","div","main","section"].includes(String(
p.tagName||"").toLowerCase()))return d;const h=String(p.getAttribute("style")||""),g=Array.from(h.matchAll(
/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)).some(b=>Array.
from(b[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi)).some(w=>Math.abs(Number(w[1])||0)>=96)),y=Array.from(h.
matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)).some(b=>Math.abs(Number(b[1])||
0)>720);return d+(g||y?1:0)},0);if(n.length<=500&&i<=24&&s===0)return t.innerHTML;const o=new Set(["\
align-items","align-self","column-gap","flex","flex-basis","flex-direction","flex-grow","flex-shrink",
"flex-wrap","gap","grid","grid-auto-columns","grid-auto-flow","grid-auto-rows","grid-column","grid-c\
olumn-end","grid-column-start","grid-row","grid-row-end","grid-row-start","grid-template","grid-temp\
late-areas","grid-template-columns","grid-template-rows","justify-content","justify-items","justify-\
self","order","row-gap"]),r=new Set(["article","div","main","section"]),l=new Set(["padding","paddin\
g-left","padding-right","padding-inline","padding-inline-start","padding-inline-end"]);return n.forEach(
d=>{if(!d||!d.style)return;const p=String(d.tagName||"").toLowerCase(),h=[];String(d.getAttribute("s\
tyle")||"").split(";").forEach(g=>{if(!g||g.indexOf(":")<0)return;const y=g.indexOf(":"),b=g.slice(0,
y).trim().toLowerCase();let w=g.slice(y+1).trim();if(!(!b||!w||o.has(b))&&!["height","max-height","m\
in-height","overflow","overflow-x","overflow-y"].includes(b)&&!(["width","min-width"].includes(b)&&r.
has(p))){if(l.has(b)&&r.has(p)&&Array.from(w.matchAll(/(-?\d+(?:\.\d+)?)px/gi)).map(S=>Math.abs(Number(
S[1])||0)).some(S=>S>=96)&&(w="0px"),b==="display"){const x=w.toLowerCase();["flex","grid"].includes(
x)?w="block":["inline-flex","inline-grid"].includes(x)&&(w="inline-block")}h.push(`${b}: ${w}`)}}),h.
length?d.setAttribute("style",h.join("; ")):d.removeAttribute("style")}),t.innerHTML},"normalizeRich\
PastePrintHtml"),getRichPasteSelectionRange=a(e=>{const t=window.getSelection&&window.getSelection();
if(!t||!t.rangeCount)return null;const n=t.getRangeAt(0);if(e&&e.contains(n.commonAncestorContainer))
return n;const i=document.createRange();return i.selectNodeContents(e),i.collapse(!1),i},"getRichPas\
teSelectionRange"),insertNodeIntoRichPasteEditor=a(e=>{const t=getRichPasteEditor();!t||!e||(t.appendChild(
e),updateRichPasteStatus())},"insertNodeIntoRichPasteEditor"),insertHtmlIntoRichPasteEditor=a(e=>{const t=sanitizeRichPasteHtml(
e);if(!t||t.trim()==="")return!1;const n=document.createElement("template");n.innerHTML=t;const i=n.
content.cloneNode(!0);return insertNodeIntoRichPasteEditor(i),!0},"insertHtmlIntoRichPasteEditor"),insertTextIntoRichPasteEditor=a(
e=>{if(e==null)return;const t=document.createTextNode(String(e));insertNodeIntoRichPasteEditor(t)},"\
insertTextIntoRichPasteEditor"),blobToDataUrl=a(e=>new Promise((t,n)=>{const i=new FileReader;i.onload=
()=>t(String(i.result||"")),i.onerror=()=>n(i.error||new Error("clipboard_image_read_failed")),i.readAsDataURL(
e)}),"blobToDataUrl"),insertClipboardImageBlob=a(async(e,t="clipboard-image")=>{if(!e)return!1;const n=await blobToDataUrl(
e);return n?(insertHtmlIntoRichPasteEditor(`<p><img src="${escapeHtml(n)}" alt="${escapeHtml(t)}"></\
p>`),!0):!1},"insertClipboardImageBlob"),readClipboardRichContent=a(async()=>{if(!navigator.clipboard||
!navigator.clipboard.read)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30EA\u30C3\u30C1\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u8AAD\u307F\u53D6\u308A\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
const e=getRichPasteCapture();e&&(e.value="");const t=await navigator.clipboard.read();if(!t||!t.length)
return!1;let n=!1;for(const i of t){if(!i)continue;const s=Array.from(i.types||[]);let o=!1;if(s.includes(
"text/html")){const d=await(await i.getType("text/html")).text();d&&insertHtmlIntoRichPasteEditor(d)&&
(n=!0,o=!0)}if(!o&&s.includes("text/plain")){const d=await(await i.getType("text/plain")).text();d&&
(insertTextIntoRichPasteEditor(d),n=!0)}const r=s.find(l=>l&&l.startsWith("image/"));if(!o&&r){const l=await i.
getType(r);await insertClipboardImageBlob(l,"clipboard-image")&&(n=!0)}}return n},"readClipboardRich\
Content"),ingestRichPasteClipboardData=a(async e=>{if(!e)return!1;let t=!1;const n=e.getData&&e.getData(
"text/html"),i=e.getData&&e.getData("text/plain");let s=!1;n&&insertHtmlIntoRichPasteEditor(n)&&(t=!0,
s=!0),!s&&i&&(insertTextIntoRichPasteEditor(i),t=!0);const r=Array.from(e.items||[]).filter(l=>l&&l.
kind==="file").map(l=>l.getAsFile()).filter(l=>l&&l.type&&l.type.startsWith("image/"));if(!s&&r.length)
for(const l of r)try{await insertClipboardImageBlob(l,l.name||"clipboard-image")&&(t=!0)}catch{}return t},
"ingestRichPasteClipboardData"),buildRichPastePdfFilename=a(()=>{const e=new Date,t=a(n=>String(n).padStart(
2,"0"),"pad");return`clipboard_rich_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(
e.getMinutes())}${t(e.getSeconds())}.pdf`},"buildRichPastePdfFilename"),getRichPasteProgressElements=a(
()=>({container:get("rich-paste-progress-container"),bar:get("rich-paste-progress-bar"),text:get("ri\
ch-paste-progress-text")}),"getRichPasteProgressElements"),setRichPasteProgress=a((e,t=null)=>{const{
container:n,bar:i,text:s}=getRichPasteProgressElements(),o=Math.max(0,Math.min(100,Number(e)||0));if(n&&
(n.classList.remove("hidden"),n.style.setProperty("display","block","important")),i&&(i.style.width=
`${o}%`,i.style.transform="none"),s&&(s.textContent=`${Math.round(o)}%`),t&&n){const r=n.querySelector(
".text-amber-400");r&&(r.innerHTML=`<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(t)}`)}},"set\
RichPasteProgress"),hideRichPasteProgress=a(()=>{const{container:e,bar:t}=getRichPasteProgressElements();
t&&(t.style.transform="scaleX(0)"),e&&(e.classList.add("hidden"),e.style.display="none")},"hideRichP\
asteProgress"),inferRichPasteTitle=a(()=>{const e=getRichPasteEditor();if(!e)return"Clipboard Export";
const t=e.querySelector("h1, h2, h3, h4, h5, h6");if(t&&t.textContent&&t.textContent.trim())return t.
textContent.trim().slice(0,48);const n=(e.innerText||"").trim().replace(/\s+/g," ");return n?n.slice(
0,48):"Clipboard Export"},"inferRichPasteTitle"),waitForRichPasteMedia=a(async(e,t=2500)=>{if(!e)return;
const n=new Promise(s=>setTimeout(s,Math.max(0,t))),i=Promise.all(Array.from(e.querySelectorAll("img")||
[]).map(s=>!s||s.complete?Promise.resolve():new Promise(o=>{let r=!1;const l=a(()=>{r||(r=!0,o())},"\
finish");s.addEventListener("load",l,{once:!0}),s.addEventListener("error",l,{once:!0}),setTimeout(l,
Math.max(250,Math.min(t,2e3)))})));if(await Promise.race([i,n]),document.fonts&&document.fonts.ready)
try{await Promise.race([document.fonts.ready,n])}catch{}},"waitForRichPasteMedia"),normalizeRichPastePdfText=a(
e=>String(e||"").replace(/\u00a0/g," ").replace(/\r\n?/g,`
`).replace(/[ \t\f\v]+/g," ").replace(/\n[ \t]+/g,`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).trim(),"normalizeRichPastePdfText"),normalizeRichPastePdfCodeText=a(e=>String(e||"").replace(/\u00a0/g,
" ").replace(/\r\n?/g,`
`),"normalizeRichPastePdfCodeText"),collectRichPasteInlineSegments=a((e,t={})=>{if(!e)return[];const n=t.
allowLinks!==!1,i=[],s=a((o,r)=>{if(!o)return;if(o.nodeType===Node.TEXT_NODE){const p=o.textContent||
"";p&&i.push(Object.assign({},r,{text:p}));return}if(o.nodeType!==Node.ELEMENT_NODE)return;const l=String(
o.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(l))return;if(l==="br"){i.push({text:`
`});return}const d=Object.assign({},r);["b","strong"].includes(l)&&(d.bold=!0),["i","em"].includes(l)&&
(d.italic=!0),l==="a"&&n&&(d.link=String(o.getAttribute("href")||"").trim()),l==="code"&&(d.monospace=
!0),Array.from(o.childNodes||[]).forEach(p=>s(p,d))},"walk");return s(e,{bold:!!t.bold,italic:!!t.italic}),
i},"collectRichPasteInlineSegments"),collectRichPasteInlineText=a((e,t={})=>collectRichPasteInlineSegments(
e,t).map(i=>i.text).join(""),"collectRichPasteInlineText"),collectRichPasteTableRows=a(e=>{const t=[];
return Array.from(e.querySelectorAll("tr")||[]).forEach(n=>{n&&n.closest&&n.closest("table")===e&&t.
push(n)}),t},"collectRichPasteTableRows"),makeRichPasteTableMarkdown=a(e=>{const t=e&&e.querySelector?
e.querySelector("caption"):null,n=t?normalizeRichPastePdfText(collectRichPasteInlineText(t)):"",i=collectRichPasteTableRows(
e).map(d=>Array.from(d.children||[]).filter(h=>{const g=String(h.tagName||"").toLowerCase();return g===
"th"||g==="td"}).map(h=>normalizeRichPastePdfText(collectRichPasteInlineText(h))||" ")).filter(d=>d.
length);if(!i.length)return n||"[table]";const s=i.reduce((d,p)=>Math.max(d,p.length),0),o=i.map(d=>{
const p=d.slice(0,s);for(;p.length<s;)p.push(" ");return p}),r=`| ${Array(s).fill("---").join(" | ")}\
 |`,l=[];n&&(l.push(`Table: ${n}`),l.push("")),l.push(`| ${o[0].join(" | ")} |`),l.push(r);for(let d=1;d<
o.length;d+=1)l.push(`| ${o[d].join(" | ")} |`);return l.join(`
`)},"makeRichPasteTableMarkdown"),collectRichPasteListBlocks=a((e,t=!1,n=0)=>{const i=[],s=Array.from(
e.children||[]).filter(r=>String(r.tagName||"").toLowerCase()==="li");let o=1;return s.forEach(r=>{const l=r.
cloneNode(!0);Array.from(l.querySelectorAll("ul,ol")||[]).forEach(p=>{try{p.remove()}catch{}});const d=collectRichPasteInlineSegments(
l);d.length>0&&i.push({type:"list_item",ordered:t,depth:n,index:o,segments:d}),Array.from(r.children||
[]).forEach(p=>{const h=String(p.tagName||"").toLowerCase();(h==="ul"||h==="ol")&&i.push(...collectRichPasteListBlocks(
p,h==="ol",n+1))}),o+=1}),i},"collectRichPasteListBlocks"),collectRichPastePdfBlocks=a((e,t=0)=>{const n=[];
if(!e)return n;let i=[];const s=a(()=>{i.length!==0&&(n.push({type:"paragraph",segments:[...i]}),i=[])},
"flushBuffer");return Array.from(e.childNodes||[]).forEach(o=>{if(!o)return;if(o.nodeType===Node.TEXT_NODE){
const p=(o.textContent||"").replace(/\u00a0/g," ");p&&i.push({text:p});return}if(o.nodeType!==Node.ELEMENT_NODE)
return;const r=String(o.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(r))return;if(r==="br"){
i.push({text:`
`});return}if(/^h[1-6]$/.test(r)){s();const p=collectRichPasteInlineSegments(o);p.length>0&&n.push({
type:"heading",level:Number(r.slice(1))||1,segments:p});return}if(r==="p"){s();const p=collectRichPasteInlineSegments(
o);p.length>0&&n.push({type:"paragraph",segments:p});return}if(r==="blockquote"){s();const p=collectRichPasteInlineSegments(
o,{italic:!0});p.length>0&&n.push({type:"blockquote",segments:p});return}if(r==="pre"){s();const p=normalizeRichPastePdfCodeText(
o.innerText||o.textContent||"");p.trim()&&n.push({type:"code",text:p});return}if(r==="table"){s();const p=makeRichPasteTableMarkdown(
o);p&&n.push({type:"table",text:p});return}if(r==="ul"||r==="ol"){s(),n.push(...collectRichPasteListBlocks(
o,r==="ol",t));return}if(r==="hr"){s(),n.push({type:"hr"});return}if(r==="figure"){s();const p=o.querySelector(
"img");p&&n.push({type:"image",src:String(p.getAttribute("src")||"").trim(),alt:String(p.getAttribute(
"alt")||p.getAttribute("title")||"").trim(),title:String(p.getAttribute("title")||"").trim()});const h=o.
querySelector("figcaption");if(h){const g=collectRichPasteInlineSegments(h);g.length>0&&n.push({type:"\
paragraph",segments:g})}return}if(r==="img"){s(),n.push({type:"image",src:String(o.getAttribute("src")||
"").trim(),alt:String(o.getAttribute("alt")||o.getAttribute("title")||"").trim(),title:String(o.getAttribute(
"title")||"").trim()});return}if(r==="li"){s(),n.push(...collectRichPasteListBlocks(o,!1,t));return}
if(Array.from(o.children||[]).some(p=>{const h=String(p.tagName||"").toLowerCase();return/^h[1-6]$/.
test(h)||["p","div","section","article","main","blockquote","pre","table","ul","ol","hr","figure","i\
mg","li"].includes(h)})&&["div","section","article","main","figure"].includes(r)){s(),n.push(...collectRichPastePdfBlocks(
o,t+1));return}const d=collectRichPasteInlineSegments(o);d.length>0&&i.push(...d)}),s(),n},"collectR\
ichPastePdfBlocks"),detectImageMimeType=a(e=>{const t=String(e||"").match(/^data:(image\/[a-z0-9.+-]+);/i);
return t?t[1].toLowerCase():"image/png"},"detectImageMimeType"),loadRichPasteImageData=a(async(e,t=3e3)=>{
const n=String(e||"").trim();if(!n)return null;if(n.startsWith("data:image/"))return{dataUrl:n,mimeType:detectImageMimeType(
n)};let i=null;try{i=new URL(n,window.location.href)}catch{return null}if(!(i.origin===window.location.
origin))return null;const o=(async()=>{try{const r=await fetch(i.toString(),{credentials:"same-origi\
n",cache:"force-cache"});if(!r.ok)return null;const l=await r.blob(),d=await blobToDataUrl(l);return{
dataUrl:d,mimeType:l.type||detectImageMimeType(d)}}catch{return null}})();return await Promise.race(
[o,new Promise(r=>setTimeout(()=>r(null),Math.max(250,t)))])},"loadRichPasteImageData"),buildRichPastePreviewHtml=a(
(e="preview")=>{const t=getRichPasteEditor();if(!t)return"";const n=inferRichPasteTitle(),i=new Date().
toLocaleString("ja-JP"),s=sanitizeRichPasteHtml(t.innerHTML||""),o=detectRichPasteTheme(s),r=normalizeRichPastePrintHtml(
s),l=e==="pdf";return`<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(n)} - Preview</title>
  <style>
        :root {
          color-scheme: ${o.mode};
          --rp-background: ${o.background};
          --rp-foreground: ${o.foreground};
          --rp-muted: ${o.muted};
          --rp-border: ${o.border};
          --rp-surface: ${o.surface};
          --rp-quote: ${o.quote};
          --rp-link: ${o.link};
        }
	    body { margin: 0; background: ${l?"var(--rp-background)":"#eef2f7"}; color: var(--rp-foreground\
); font-family: "Noto Sans JP", system-ui, sans-serif; }
	    .page { max-width: ${l?"794px":"920px"}; margin: 0 auto; padding: ${l?"28px 30px 36px":"24px"};\
 }
	    .card { background: var(--rp-background); color: var(--rp-foreground); border: 1px solid var(--\
rp-border); border-radius: 18px; padding: 20px; box-shadow: ${l?"none":"0 18px 45px rgba(15,23,42,0.\
14)"}; }
	    .title { margin: 0; font-size: ${l?"22px":"24px"}; line-height: 1.35; color: var(--rp-foregroun\
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
    .toolbar { display:${l?"none":"flex"}; gap:10px; margin-top: 16px; flex-wrap: wrap; }
    .toolbar button { border: 1px solid var(--rp-border); background: var(--rp-surface); color: var(\
--rp-foreground); border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    ${l?".card { border-radius: 0; } .page { max-width: none; padding: 0; }":""}
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
      <div class="content">${r||"<p>\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</p>"}</di\
v>
    </div>
  </div>
</body>
</html>`},"buildRichPastePreviewHtml"),openSandboxedHtmlTab=a(e=>{const n=`<!doctype html><html><hea\
d><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:10\
0%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe i\
d="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referre\
r"></iframe><script>document.getElementById('preview').srcdoc=${JSON.stringify(String(e||"")).replace(
/</g,"\\u003c").replace(/\u2028/g,"\\u2028").replace(/\u2029/g,"\\u2029")};<\/script></body></html>`,
i=new Blob([n],{type:"text/html;charset=utf-8"}),s=URL.createObjectURL(i);return window.open(s,"_bla\
nk","noopener,noreferrer")?(setTimeout(()=>URL.revokeObjectURL(s),6e4),!0):(URL.revokeObjectURL(s),!1)},
"openSandboxedHtmlTab"),openRichPastePreviewTab=a(()=>{const e=buildRichPastePreviewHtml("preview");
if(!e){showToast("\u78BA\u8A8D\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093","warning",
!0);return}openSandboxedHtmlTab(e)||showToast("\u5225\u30BF\u30D6\u306E\u8868\u793A\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},"openRichPastePreviewTab"),renderRichPastePdfBlob=a(async()=>{const e=get("rich-paste-p\
rogress-container"),t=get("rich-paste-progress-bar"),n=get("rich-paste-progress-text"),i=a(y=>{const b=Math.
max(0,Math.min(100,Number(y)||0));t&&(t.style.width="100%",t.style.transformOrigin="left center",(!t.
style.transition||t.style.transition.indexOf("transform")===-1)&&(t.style.transition="transform 0.45\
s cubic-bezier(0.22, 1, 0.36, 1)"),t.style.transform=`scaleX(${b/100})`,t.style.willChange="transfor\
m"),n&&(n.innerText=`${Math.round(b)}%`)},"updateProgress");e&&(e.classList.remove("hidden"),e.style.
setProperty("display","block","important")),t&&(t.style.transition="none",t.style.width="100%",t.style.
transformOrigin="left center",t.style.transform="scaleX(0)",t.offsetHeight,t.style.transition="trans\
form 0.45s cubic-bezier(0.22, 1, 0.36, 1)"),i(0),await new Promise(y=>requestAnimationFrame(()=>setTimeout(
y,150)));const s=getRichPasteEditor();if(!s)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093");
const o=inferRichPasteTitle(),r=sanitizeRichPasteHtml(s.innerHTML||""),l=detectRichPasteTheme(r),d=normalizeRichPastePrintHtml(
r);await ensurePdfLibraries();const p=window.jspdf&&window.jspdf.jsPDF?window.jspdf.jsPDF:null;if(!p)
throw new Error("jsPDF \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
const h=window.html2canvas;if(typeof h!="function")throw new Error("html2canvas \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
i(5);const g=document.createElement("div");g.style.position="absolute",g.style.left="-10000px",g.style.
top="0",g.style.width="794px",g.style.background=l.background,g.style.color=l.foreground,g.style.boxSizing=
"border-box",g.style.fontFamily='"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif',g.
innerHTML=`
                <style>
                        :root {
                            color-scheme: ${l.mode};
                            --rp-background: ${l.background};
                            --rp-foreground: ${l.foreground};
                            --rp-muted: ${l.muted};
                            --rp-border: ${l.border};
                            --rp-surface: ${l.surface};
                            --rp-quote: ${l.quote};
                            --rp-link: ${l.link};
                        }
	                    .pdf-root-wrapper {
	                        background-color: var(--rp-background);
	                        color: var(--rp-foreground);
	                        padding: 40px;
	                        width: 794px;
	                        min-height: 1123px;
	                        box-sizing: border-box;
	                        color-scheme: ${l.mode};
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
                    <div class="pdf-title">${escapeHtml(o)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString("ja-JP")}</div>
                    <div class="pdf-content">${d}</div>
                </div>
            `,document.body.appendChild(g),await waitForRichPasteMedia(g,4e3),i(15);try{const y=new p(
{unit:"mm",format:"a4",orientation:"portrait",compress:!0}),b=y.internal.pageSize.getWidth(),w=y.internal.
pageSize.getHeight(),x=794,S=Math.floor(w/b*x),T=g.scrollHeight||g.offsetHeight;let E=0,F=!0;const J=Math.
ceil(T/S);let X=0;for(;E<T;){if(richPasteAbortController&&richPasteAbortController.signal.aborted)throw new DOMException(
"Aborted","AbortError");const O=Math.min(S,T-E),Y=(await new Promise((me,_e)=>{const ne=setTimeout(()=>_e(
new Error("PDF chunk rendering timed out")),12e4);h(g,{scale:1,useCORS:!0,allowTaint:!1,backgroundColor:l.
background,logging:!1,imageTimeout:5e3,x:0,y:E,width:x,height:O,windowWidth:x,scrollX:0,scrollY:0,signal:richPasteAbortController?
richPasteAbortController.signal:void 0,onclone:a(se=>{prepareRichPastePdfClone(se,l);const W=se.querySelector(
".pdf-root-wrapper");W&&(W.style.position="relative",W.style.left="0",W.style.top="0")},"onclone")}).
then(se=>{clearTimeout(ne),me(se)}).catch(se=>{clearTimeout(ne),_e(se)})})).toDataURL("image/jpeg",.95),
pe=y.getImageProperties(Y),oe=Math.min(w,pe.height*b/pe.width);F||y.addPage(),y.addImage(Y,"JPEG",0,
0,b,oe),F=!1,E+=O,X++;const ke=Math.min(100,15+Math.round(X/J*85));i(ke),await new Promise(me=>setTimeout(
me,100))}return i(100),{blob:y.output("blob"),fileName:buildRichPastePdfFilename()}}finally{e&&(e.classList.
add("hidden"),e.style.display="none"),g&&g.parentNode&&document.body.removeChild(g)}},"renderRichPas\
tePdfBlob"),createRichPastePdfBlob=a(async()=>await renderRichPastePdfBlob(),"createRichPastePdfBlob"),
buildRichPasteServerPayload=a(()=>{const e=getRichPasteEditor();if(!e)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\
\u3093");const t=String(e.innerHTML||"").trim(),n=String(e.textContent||"").trim(),i=t||(n?`<p>${escapeHtml(
n).replace(/\n/g,"<br/>")}</p>`:"");return{title:inferRichPasteTitle(),html:i,created_at:new Date().
toLocaleString("ja-JP"),theme:detectRichPasteTheme(sanitizeRichPasteHtml(i))}},"buildRichPasteServer\
Payload"),attachRichPastePdfAndSend=a(async(e,t,n,i)=>{const s=new Set(collectAttachmentItemsForSend().
map(h=>h.path)),o=new File([e],t,{type:"application/pdf",lastModified:Date.now()}),r=get("prompt-inp\
ut");if(r&&(r.value=n),await handleFiles([o],{openModal:!1}),!collectAttachmentItemsForSend().map(h=>h.
path).some(h=>!s.has(h)))throw r&&(r.value=i),new Error("PDF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const p=sendMessage();clearRichPasteEditor(!0),window.closeRichPasteModal(),showToast("PDF\u3092\u6DFB\u4ED8\u3057\u3066\u9001\u4FE1\u3092\u958B\u59CB\
\u3057\u307E\u3057\u305F","success"),p&&typeof p.catch=="function"&&p.catch(()=>{})},"attachRichPast\
ePdfAndSend"),openRichPasteModal=a(async()=>{await ensureUserSettingsSnapshot(),showModal("rich-past\
e-modal"),location.pathname!=="/paste"&&history.pushState({modal:"paste"},"","/paste");const e=getRichPastePrompt();
e&&(richPastePromptPreferenceSyncing=!0,e.value=getRichPasteEffectivePrompt(userSettingsSnapshot),richPastePromptPreferenceSyncing=
!1),updateRichPasteStatus(),setTimeout(()=>focusRichPasteEditor(),80)},"openRichPasteModal");window.
closeRichPasteModal=(e=!1)=>{hideModal("rich-paste-modal"),!e&&location.pathname==="/paste"&&history.
back()};const sendRichPasteToModel=a(async(e={})=>{const t=!!(e&&e.serverSide);if(abortController||richPasteAbortController){
showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u307E\u305F\u306FPDF\u5909\u63DB\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const n=getRichPasteEditor(),i=getRichPastePrompt(),s=get(t?"rich-paste-send-se\
rver-btn":"rich-paste-send-btn"),o=get("rich-paste-cancel-btn");if(!n||!n.innerText||!n.innerText.trim()){
showToast("\u8CBC\u308A\u4ED8\u3051\u308B\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}richPasteAbortController=new AbortController,o&&(o.onclick=()=>{richPasteAbortController&&
(richPasteAbortController.abort(),showToast("PDF\u5909\u63DB\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info"))});const r=i&&i.value&&i.value.trim()?i.value.trim():RICH_PASTE_DEFAULT_PROMPT,l=get("prompt\
-input")?get("prompt-input").value:"";s&&(s.disabled=!0);try{const d=get("toast-stack");if(d&&d.querySelectorAll(
".toast").forEach(p=>{(p.innerText.includes("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059")||
p.innerText.includes("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059"))&&
p.remove()}),t?(showToast("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...",
"info",!0),setRichPasteProgress(2,"\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...")):
showToast("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...","info",!0),t){if(!RICH_PASTE_PDF_SERVER_ROUTE)
throw new Error("\u30B5\u30FC\u30D0\u30FC\u5074PDF\u751F\u6210\u306EURL\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093");
const p=buildRichPasteServerPayload();setRichPasteProgress(10,"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u4FE1\u4E2D...");
const h=await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(p),signal:richPasteAbortController.signal});if(setRichPasteProgress(60,
"PDF\u3092\u53D7\u4FE1\u4E2D..."),!h.ok){let w="";try{const x=await h.json();w=x&&(x.message||x.error)?
String(x.message||x.error):""}catch{try{w=await h.text()}catch{w=""}}throw w==="missing_html"?new Error(
"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u308BHTML\u304C\u7A7A\u3067\u3059\u3002\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u5185\u5BB9\u306E\u53D6\u308A\u8FBC\u307F\u3092\u5148\u306B\u884C\u3063\u3066\u304F\u3060\u3055\u3044"):
new Error(w?`\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${w}`:
"\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}setRichPasteProgress(
75,"PDF\u3092\u6DFB\u4ED8\u4E2D...");const g=await h.blob(),y=h.headers.get("X-Rich-Paste-Filename")||
buildRichPastePdfFilename();!!(get("rich-paste-download-only")&&get("rich-paste-download-only").checked)?
(setRichPasteProgress(90,"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D..."),downloadBlob(g,y),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(g,y,r,l),setRichPasteProgress(100,"\u5B8C\u4E86"),
setTimeout(()=>hideRichPasteProgress(),400)}else{const p=await createRichPastePdfBlob();!!(get("rich\
-paste-download-only")&&get("rich-paste-download-only").checked)?(downloadBlob(p.blob,p.fileName),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(p.blob,p.fileName,r,l)}}catch(d){if(d.
name==="AbortError"){console.log("PDF generation aborted by user"),t&&(setRichPasteProgress(0,"\u30AD\u30E3\u30F3\u30BB\u30EB\
\u3055\u308C\u307E\u3057\u305F"),setTimeout(()=>hideRichPasteProgress(),800));return}get("prompt-inp\
ut")&&(get("prompt-input").value=l);const p=d&&d.message?d.message:"PDF\u5316\u3057\u3066\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
showToast(p,"error",!0),t&&(setRichPasteProgress(0,"\u5931\u6557\u3057\u307E\u3057\u305F"),setTimeout(
()=>hideRichPasteProgress(),1200))}finally{s&&(s.disabled=!1),richPasteAbortController=null}},"sendR\
ichPasteToModel");let csrfToken=document.querySelector('meta[name="csrf-token"]').content,csrfRefreshPromise=null;
const refreshCsrfToken=a(async()=>csrfRefreshPromise||(csrfRefreshPromise=(async()=>{const e=await fetch(
"/api/csrf_token",{method:"GET",credentials:"include",cache:"no-store",headers:{Accept:"application/\
json"}});if(!e.ok)return!1;const t=await e.json().catch(()=>({})),n=t&&typeof t.csrf_token=="string"?
t.csrf_token:"";if(!n)return!1;csrfToken=n;const i=document.querySelector('meta[name="csrf-token"]');
return i&&i.setAttribute("content",n),!0})().catch(()=>!1).finally(()=>{csrfRefreshPromise=null}),csrfRefreshPromise),
"refreshCsrfToken"),apiFetch=a(async(e,t={})=>{const n=(t.method||"GET").toUpperCase(),i=Object.assign(
{},t.headers||{}),s=!["GET","HEAD","OPTIONS"].includes(n);s&&(i["X-CSRF-Token"]=csrfToken);const o=t.
credentials||"include";let r=await fetch(e,Object.assign({},t,{headers:i,credentials:o}));if(s&&(r.status===
403||r.status===404)){let l=null;try{l=await r.clone().json()}catch{}const d=l&&l.error;if(d==="acco\
unt_locked")return!isAdminUser&&!document.getElementById("bot-lock-overlay")&&showBotLockOverlay(l.message||
"\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u3066\u3044\u307E\u3059\u3002",
l.remaining_seconds),r;if(d==="banned"||d==="turnstile_failed"||d==="rate_limit")return r;if(d==="tu\
rnstile_required"&&isBotDetectionActive())return botDetectionVerified=!1,await Promise.race([runBotDetectionGate(),
new Promise(g=>setTimeout(()=>g(!1),3e4))])&&(i["X-CSRF-Token"]=csrfToken,r=await fetch(e,Object.assign(
{},t,{headers:i,credentials:o}))),r;await refreshCsrfToken()&&(i["X-CSRF-Token"]=csrfToken,r=await fetch(
e,Object.assign({},t,{headers:i,credentials:o})))}return r},"apiFetch"),manualSpinnerRequestOptions=a(
e=>window.ProgressSpinner?window.ProgressSpinner.manualRequestOptions(e):e,"manualSpinnerRequestOpti\
ons");window.updateGoogleLinkUI=e=>{const t=get("google-link-text"),n=get("google-email-text"),i=get(
"google-action-area"),s=get("google-link-icon");!t||!i||(e.google_id?(t.innerText="\u9023\u643A\u6E08\u307F",
t.classList.replace("text-gray-200","text-green-400"),n.innerText=e.google_email||"\u9023\u643A\u4E2D\u306E Google \u30A2\u30AB\u30A6\u30F3\u30C8",
s.classList.replace("bg-gray-800","bg-green-900/30"),s.classList.add("text-green-400"),i.innerHTML='\
<button onclick="unlinkGoogleAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-\
400 border border-red-800 rounded text-xs font-bold transition btn-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):
(t.innerText="\u672A\u9023\u643A",t.classList.replace("text-green-400","text-gray-200"),n.innerText=
"Google \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
s.classList.replace("bg-green-900/30","bg-gray-800"),s.classList.remove("text-green-400"),i.innerHTML=
'<a href="/login/google" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white roun\
ded text-xs font-bold transition btn-hover">Google \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkGoogleAccount=
async()=>{if(confirm(`Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Google \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkGoogleAccount,{method:"POST"});if(e.ok)showToast("\
Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(t=>t.json()).then(t=>updateGoogleLinkUI(t));else{const t=await e.json();showToast(t.error||"\u89E3\u9664\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}},window.updateMinashinLinkUI=e=>{const t=get("minashin-link-text"),n=get("minashin-emai\
l-text"),i=get("minashin-action-area"),s=get("minashin-link-icon");!t||!i||(e.minashin_sub?(t.innerText=
"\u9023\u643A\u6E08\u307F",t.classList.replace("text-gray-200","text-green-400"),n.innerText=e.minashin_email||
"\u9023\u643A\u4E2D\u306E Minashin \u30A2\u30AB\u30A6\u30F3\u30C8",s.classList.replace("bg-gray-800",
"bg-green-900/30"),i.innerHTML='<button onclick="unlinkMinashinAccount()" class="px-4 py-2 bg-red-90\
0/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn\
-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):(t.innerText="\u672A\u9023\u643A",t.classList.replace(
"text-green-400","text-gray-200"),n.innerText="Minashin \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
s.classList.replace("bg-green-900/30","bg-gray-800"),i.innerHTML='<a href="/login/minashin" class="i\
nline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition \
btn-hover">Minashin \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkMinashinAccount=async()=>{if(confirm(
`Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Minashin \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkMinashinAccount,{method:"POST"});if(e.ok)showToast(
"Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(t=>t.json()).then(t=>updateMinashinLinkUI(t));else{const t=await e.json();showToast(t.error||"\u89E3\
\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}};let lastClientDebugEnabled=null;const isClientDebugLogEnabled=a(()=>{const e=get("set-\
client-debug-log");return!!(e&&e.checked)},"isClientDebugLogEnabled"),sendClientDebugLog=a((e,t)=>{if(!isClientDebugLogEnabled())
return;const n={level:String(e||"info"),message:String(t||"")};apiFetch("/api/debug/client_log",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(n)}).catch(()=>{})},"sendClien\
tDebugLog"),syncClientDebugLogToggle=a((e,t)=>{const n=get("set-client-debug-log");n&&(n.checked=!!e);
const i=!!e;i&&lastClientDebugEnabled!==!0&&sendClientDebugLog("info",`Client debug logging enabled \
(${t}).`),lastClientDebugEnabled=i},"syncClientDebugLogToggle"),nowPerfMs=a(()=>window.performance&&
typeof window.performance.now=="function"?window.performance.now():Date.now(),"nowPerfMs"),reportFirstTokenLatency=a(
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
1"?Promise.resolve(n):new Promise((i,s)=>{n.addEventListener("load",()=>i(n),{once:!0}),n.addEventListener(
"error",s,{once:!0})}):new Promise((i,s)=>{const o=document.createElement("script");t&&(o.id=t),o.src=
e,o.async=!0,o.onload=()=>{o.dataset.loaded="1",i(o)},o.onerror=s,document.head.appendChild(o)})}a(loadScriptOnce,
"loadScriptOnce");function loadStylesheetOnce(e,t){const n=t?document.getElementById(t):null;if(n)return Promise.
resolve(n);const i=Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find(s=>s.href===
e);return i?Promise.resolve(i):new Promise((s,o)=>{const r=document.createElement("link");t&&(r.id=t),
r.rel="stylesheet",r.href=e,r.onload=()=>s(r),r.onerror=o,document.head.appendChild(r)})}a(loadStylesheetOnce,
"loadStylesheetOnce");async function ensureMathJaxLoaded(){return window.MathJax&&typeof window.MathJax.
typesetPromise=="function"?window.MathJax:(mathJaxLoadPromise||(window.MathJax=window.MathJax||{tex:{
inlineMath:[["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{
ignoreHtmlClass:"tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},
startup:{typeset:!1}},mathJaxLoadPromise=loadScriptOnce(MATHJAX_SRC,"MathJax-script").catch(e=>{throw mathJaxLoadPromise=
null,e})),await mathJaxLoadPromise,window.MathJax||null)}a(ensureMathJaxLoaded,"ensureMathJaxLoaded");
async function ensureHighlightLoaded(){return window.hljs?window.hljs:(highlightLoadPromise||(highlightLoadPromise=
Promise.all([loadStylesheetOnce(HLJS_CSS_SRC,"hljs-theme-chat"),loadScriptOnce(HLJS_JS_SRC,"hljs-scr\
ipt")]).then(()=>window.hljs||null).catch(e=>{throw highlightLoadPromise=null,e})),await highlightLoadPromise)}
a(ensureHighlightLoaded,"ensureHighlightLoaded");function maybeNeedsMathJax(e){const t=String(e||"");
return t.includes("$$")||t.includes("\\(")||t.includes("\\[")||t.includes("\\begin{")?!0:/(?<!\$)\$(?!\$)(?=[\s\S]*?[A-Za-z\\^_{}])(?:[^$\n\\]|\\.)+?\$(?!\$)/.
test(t)}a(maybeNeedsMathJax,"maybeNeedsMathJax");function protectMathSegments(e){const t=String(e||""),
n=[],i=a(p=>{const h=`@@MATHJAX_BLOCK_${n.length}@@`;return n.push(p),h},"stash"),s=[],o=/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
let r=0,l;for(;(l=o.exec(t))!==null;){const p=l.index;p>r&&s.push({type:"text",value:t.slice(r,p)}),
s.push({type:"code",value:l[0]}),r=p+l[0].length}return r<t.length&&s.push({type:"text",value:t.slice(
r)}),s.length||s.push({type:"text",value:t}),{text:s.map(p=>{if(p.type==="code")return p.value;let h=p.
value;return h=h.replace(/\$\$([\s\S]+?)\$\$/g,i),h=h.replace(/\\\(([\s\S]+?)\\\)/g,i),h=h.replace(/\\\[([\s\S]+?)\\\]/g,
i),h=h.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g,i),h=h.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g,
i),h}).join(""),blocks:n}}a(protectMathSegments,"protectMathSegments");function getStreamMathSegmentKey(e,t){
const n=String(t||"");let i=2166136261;for(let s=0;s<n.length;s++)i^=n.charCodeAt(s),i=Math.imul(i,16777619);
return`${e}-${n.length}-${(i>>>0).toString(16)}`}a(getStreamMathSegmentKey,"getStreamMathSegmentKey");
function restoreMathSegments(e,t,n={}){return!t||!t.length?String(e||""):String(e||"").replace(/@@MATHJAX_BLOCK_(\d+)@@/g,
(i,s)=>{const o=t[Number(s)];if(o==null)return"";const r=String(o).replace(/&/g,"&amp;").replace(/</g,
"&lt;").replace(/>/g,"&gt;");return n.streamMathSegments?`<span class="stream-math-segment mathjax_p\
rocess" data-stream-math-key="${getStreamMathSegmentKey(Number(s),o)}">${r}</span>`:r})}a(restoreMathSegments,
"restoreMathSegments");function maybeNeedsHighlight(e,t=null){return String(e||"").includes("```")?!0:
!t||typeof t.querySelector!="function"?!1:!!t.querySelector("pre code")}a(maybeNeedsHighlight,"maybe\
NeedsHighlight");function queueMathTypeset(e,t="",n={}){lowBandwidthMode&&!n.force||!e||!maybeNeedsMathJax(
t)||ensureMathJaxLoaded().then(()=>{if(!(!window.MathJax||typeof window.MathJax.typesetPromise!="fun\
ction")){try{typeof window.MathJax.typesetClear=="function"&&window.MathJax.typesetClear([e])}catch{}
return window.MathJax.typesetPromise([e]).catch(()=>{})}}).catch(()=>{})}a(queueMathTypeset,"queueMa\
thTypeset");function queueIncrementalMathTypeset(e){const t=Array.from(e||[]).filter(n=>n&&n.isConnected&&
!n.getAttribute("data-stream-math-state"));!t.length||lowBandwidthMode||(t.forEach(n=>n.setAttribute(
"data-stream-math-state","queued")),incrementalMathTypesetChain=incrementalMathTypesetChain.catch(()=>{}).
then(async()=>{await ensureMathJaxLoaded();const n=t.filter(i=>i.isConnected&&i.getAttribute("data-s\
tream-math-state")==="queued");if(!(!n.length||!window.MathJax||typeof window.MathJax.typesetPromise!=
"function")){n.forEach(i=>i.setAttribute("data-stream-math-state","rendering"));try{await window.MathJax.
typesetPromise(n),n.forEach(i=>{i.isConnected&&i.setAttribute("data-stream-math-state","rendered")})}catch{
n.forEach(s=>s.removeAttribute("data-stream-math-state"))}}}).catch(()=>{t.forEach(n=>n.removeAttribute(
"data-stream-math-state"))}))}a(queueIncrementalMathTypeset,"queueIncrementalMathTypeset");function queueHighlight(e,t="",n={}){
lowBandwidthMode&&!n.force||!e||!maybeNeedsHighlight(t,e)||activeStreamingBubbleId&&e.closest(`#${activeStreamingBubbleId}`)||
ensureHighlightLoaded().then(()=>{window.hljs&&e.querySelectorAll("pre code").forEach(i=>{if(!(i.getAttribute(
"data-highlighted")==="true"&&!n.force))try{window.hljs.highlightElement(i)}catch{}})}).catch(()=>{})}
a(queueHighlight,"queueHighlight");function getNetworkConnectionInfo(){return navigator.connection||
navigator.mozConnection||navigator.webkitConnection||null}a(getNetworkConnectionInfo,"getNetworkConn\
ectionInfo");function detectLowBandwidthModeAuto(){const e=getNetworkConnectionInfo();if(!e)return{enabled:!1,
reason:""};const t=!!e.saveData,n=String(e.effectiveType||"").toLowerCase(),i=Number(e.downlink||0),
s=n==="slow-2g"||n==="2g"||n==="3g",o=Number.isFinite(i)&&i>0&&i<1.3,r=t||s||o,l=[];return t&&l.push(
"\u30C7\u30FC\u30BF\u7BC0\u7D04"),n&&l.push(`\u56DE\u7DDA:${n}`),o&&l.push(`\u4E0B\u308A:${i}Mbps`),
{enabled:r,reason:l.join(" / ")}}a(detectLowBandwidthModeAuto,"detectLowBandwidthModeAuto");function normalizeLowBandwidthModePreference(e){
const t=String(e||"").trim().toLowerCase();return t==="on"||t==="off"||t==="auto"?t:"auto"}a(normalizeLowBandwidthModePreference,
"normalizeLowBandwidthModePreference");function readLowBandwidthModePreference(){try{return normalizeLowBandwidthModePreference(
localStorage.getItem(LOW_BANDWIDTH_MODE_STORAGE_KEY)||"auto")}catch{return"auto"}}a(readLowBandwidthModePreference,
"readLowBandwidthModePreference");function persistLowBandwidthModePreference(e){const t=normalizeLowBandwidthModePreference(
e);lowBandwidthModePreference=t;try{t==="auto"?localStorage.removeItem(LOW_BANDWIDTH_MODE_STORAGE_KEY):
localStorage.setItem(LOW_BANDWIDTH_MODE_STORAGE_KEY,t)}catch{}}a(persistLowBandwidthModePreference,"\
persistLowBandwidthModePreference");function getEffectiveThreadInitialMessageLimit(){return lowBandwidthMode?
LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT:THREAD_INITIAL_MESSAGE_LIMIT}a(getEffectiveThreadInitialMessageLimit,
"getEffectiveThreadInitialMessageLimit");function getEffectiveThreadOlderPageSize(){return lowBandwidthMode?
LOW_BANDWIDTH_OLDER_PAGE_SIZE:THREAD_OLDER_PAGE_SIZE}a(getEffectiveThreadOlderPageSize,"getEffective\
ThreadOlderPageSize");function mergeBtnClasses(e,t=[],n=[]){e&&(n.forEach(i=>e.classList.remove(i)),
t.forEach(i=>e.classList.add(i)))}a(mergeBtnClasses,"mergeBtnClasses");function updateLowBandwidthModeUi(){
const e=get("low-bandwidth-toggle-btn"),t=get("low-bandwidth-status-pill"),n=lowBandwidthModePreference===
"auto"?"\u81EA\u52D5":lowBandwidthModePreference==="on"?"\u56FA\u5B9AON":"\u56FA\u5B9AOFF",i=lowBandwidthMode?
"ON":"OFF",s=lowBandwidthModeReason?` (${lowBandwidthModeReason})`:"";if(e&&(e.setAttribute("title",
`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9 ${i} / ${n}${s}`),e.setAttribute("aria-pressed",lowBandwidthMode?
"true":"false"),lowBandwidthMode?mergeBtnClasses(e,["text-amber-200","bg-amber-900/30","border","bor\
der-amber-600/40"],["text-gray-400"]):mergeBtnClasses(e,["text-gray-400"],["text-amber-200","bg-ambe\
r-900/30","border","border-amber-600/40"])),t)if(lowBandwidthMode){t.classList.remove("hidden");const o=lowBandwidthModePreference===
"auto"?" (\u81EA\u52D5)":" (\u624B\u52D5)";t.innerHTML=`<i class="fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9${o}${lowBandwidthModeReason?
`: ${escapeHtml(lowBandwidthModeReason)}`:""}`}else t.classList.add("hidden"),t.innerHTML='<i class=\
"fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9'}a(updateLowBandwidthModeUi,"updat\
eLowBandwidthModeUi");function refreshDecorationsForVisibleChat(){const e=get("chat-container");e&&(queueHighlight(
e,e.textContent||"",{force:!0}),queueMathTypeset(e,e.textContent||"",{force:!0}))}a(refreshDecorationsForVisibleChat,
"refreshDecorationsForVisibleChat");function applyLowBandwidthModeState(e,t={}){const n=lowBandwidthMode;
if(lowBandwidthMode=!!e,updateLowBandwidthModeUi(),n&&!lowBandwidthMode&&refreshDecorationsForVisibleChat(),
t.notify){const i=lowBandwidthModePreference==="auto"?"\u81EA\u52D5":"\u624B\u52D5",s=lowBandwidthModeReason?
` (${lowBandwidthModeReason})`:"";showToast(`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9\u3092${lowBandwidthMode?
"ON":"OFF"}\u306B\u3057\u307E\u3057\u305F [${i}]${s}`,"info",!1)}}a(applyLowBandwidthModeState,"appl\
yLowBandwidthModeState");function recomputeLowBandwidthMode(e={}){const t=detectLowBandwidthModeAuto();
lowBandwidthModeAuto=!!t.enabled,lowBandwidthModeReason=t.reason||"",applyLowBandwidthModeState(lowBandwidthModePreference===
"on"?!0:lowBandwidthModePreference==="off"?!1:lowBandwidthModeAuto,e)}a(recomputeLowBandwidthMode,"r\
ecomputeLowBandwidthMode");function cycleLowBandwidthModePreference(){const e=normalizeLowBandwidthModePreference(
lowBandwidthModePreference);persistLowBandwidthModePreference(e==="auto"?"on":e==="on"?"off":"auto"),
recomputeLowBandwidthMode({notify:!0})}a(cycleLowBandwidthModePreference,"cycleLowBandwidthModePrefe\
rence");function ensureDeferredDecorationObserver(){if(deferredDecorationObserver||typeof IntersectionObserver==
"undefined")return deferredDecorationObserver;const e=get("chat-container")||null;return deferredDecorationObserver=
new IntersectionObserver(t=>{t.forEach(n=>{!n.isIntersecting||!n.target||runDeferredDecorations(n.target)})},
{root:e,threshold:LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD}),deferredDecorationObserver}a(ensureDeferredDecorationObserver,
"ensureDeferredDecorationObserver");function runDeferredDecorations(e){if(!e)return;if(deferredDecorationObserver)
try{deferredDecorationObserver.unobserve(e)}catch{}const t=deferredDecorationTextMap.get(e)||"";queueHighlight(
e,t,{force:!0}),queueMathTypeset(e,t,{force:!0})}a(runDeferredDecorations,"runDeferredDecorations");
function queueMessageDecorations(e,t=""){if(!e)return;if(!lowBandwidthMode){queueHighlight(e,t),queueMathTypeset(
e,t);return}if(!maybeNeedsHighlight(t,e)&&!maybeNeedsMathJax(t))return;deferredDecorationTextMap.set(
e,String(t||""));const n=get("chat-container");if(n&&e===n){window.setTimeout(()=>runDeferredDecorations(
e),250);return}if(!e.isConnected)return;const i=ensureDeferredDecorationObserver();if(i){i.observe(e);
return}window.setTimeout(()=>runDeferredDecorations(e),250)}a(queueMessageDecorations,"queueMessageD\
ecorations");function initLowBandwidthMode(){lowBandwidthModePreference=readLowBandwidthModePreference(),
recomputeLowBandwidthMode({notify:!1});const e=get("low-bandwidth-toggle-btn");e&&!e.__lowBandwidthBound&&
(e.__lowBandwidthBound=!0,e.addEventListener("click",n=>{n&&n.preventDefault(),cycleLowBandwidthModePreference()}));
const t=getNetworkConnectionInfo();t&&typeof t.addEventListener=="function"&&!lowBandwidthConnectionListenerAttached&&
(lowBandwidthConnectionListenerAttached=!0,t.addEventListener("change",()=>{if(lowBandwidthModePreference===
"auto")recomputeLowBandwidthMode({notify:!0});else{const n=detectLowBandwidthModeAuto();lowBandwidthModeAuto=
!!n.enabled,lowBandwidthModeReason=n.reason||"",updateLowBandwidthModeUi()}}))}a(initLowBandwidthMode,
"initLowBandwidthMode");function escapeHtml(e){return e==null?"":String(e).replace(/&/g,"&amp;").replace(
/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;")}a(escapeHtml,"escape\
Html");const BLOCKED_SCRIPT_HOSTS=["polyfill.io","cdn.polyfill.io"];function isBlockedScriptSrc(e){if(!e)
return!1;const t=String(e).trim();if(!t)return!1;let n=t;t.startsWith("//")?n="https:"+t:!/^https?:\/\//i.
test(t)&&!t.startsWith("data:")&&!t.startsWith("blob:")&&(n="https://"+t);try{const s=(new URL(n,"ht\
tps://example.com").hostname||"").toLowerCase();return BLOCKED_SCRIPT_HOSTS.some(o=>s===o||s.endsWith(
"."+o))}catch{return/polyfill\.io/i.test(t)}}a(isBlockedScriptSrc,"isBlockedScriptSrc");function isPasswordPromptingScript(e){
if(!e)return!1;const t=String(e),n=t.toLowerCase();return!!(/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.
test(t)||/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(t)||
/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.
test(n)||/prompt\s*\(/.test(t)&&/(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.
test(t))}a(isPasswordPromptingScript,"isPasswordPromptingScript");function detectBlockedScriptsInCode(e){
if(!e)return!1;const t=String(e),n=/<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;let i;for(;(i=n.exec(
t))!==null;)if(isBlockedScriptSrc(i[1]))return!0;const s=/<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
for(;(i=s.exec(t))!==null;)if(isPasswordPromptingScript(i[1]))return!0;return!!(/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.
test(t)||/src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(t))}a(detectBlockedScriptsInCode,"detectBlockedS\
criptsInCode");function sanitizeHtmlForPreview(e){if(!e)return"";const t=detectBlockedScriptsInCode(
e);let n=String(e);try{const s=new DOMParser().parseFromString(n,"text/html");let o=!1;s.querySelectorAll(
"script").forEach(l=>{const d=l.getAttribute("src")||"";let p=!1;if(d&&isBlockedScriptSrc(d)){const h=s.
createElement("div");h.setAttribute("data-blocked-script","true"),h.style.cssText="background:#fee2e\
2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6p\
x 0;font-family:system-ui;";const g=d.length>70?d.slice(0,67)+"...":d;h.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: "+
g+" \uFF08polyfill.io \u306A\u3069\u306E\u5371\u967A\u30C9\u30E1\u30A4\u30F3\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u7121\u52B9\u5316\u3055\u308C\u307E\u3059\uFF09",
l.parentNode&&l.parentNode.replaceChild(h,l),o=!0,p=!0}else if(!d){const h=l.textContent||"";if(isPasswordPromptingScript(
h)){const g=s.createElement("div");g.setAttribute("data-blocked-script","true"),g.style.cssText="bac\
kground:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:\
12px;margin:6px 0;font-family:system-ui;",g.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: \u30D1\u30B9\u30EF\u30FC\u30C9\u5165\u529B\u8981\u6C42\u306A\u3069\u306E\u7591\u308F\u3057\u3044\u30A4\u30F3\u30E9\u30A4\u30F3\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\
\u305F",l.parentNode&&l.parentNode.replaceChild(g,l),o=!0,p=!0}}}),s.querySelectorAll('a[href^="java\
script:" i], area[href^="javascript:" i]').forEach(l=>{l.setAttribute("href","#"),l.setAttribute("ti\
tle",(l.getAttribute("title")||"")+" [javascript: disabled in preview]")});const r=s.head||s.querySelector(
"head");if(r&&!r.querySelector("base")){const l=s.createElement("base");l.setAttribute("href",`${window.
location.origin}/`),r.insertBefore(l,r.firstChild)}if(t||o){const l=s.body||s.documentElement;if(l){
const d=s.createElement("div");d.style.cssText="position:sticky;top:0;left:0;right:0;z-index:2147483\
647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:syst\
em-ui;border-bottom:1px solid #b91c1c;",d.innerHTML="\u26A0 <strong>\u5B89\u5168\u30D7\u30EC\u30D3\u30E5\u30FC</strong>: polyfill.io \u306A\u3069\u306E\u5371\u967A\u306A\u30B9\
\u30AF\u30EA\u30D7\u30C8\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002\u5B9F\u884C\u306F\u81EA\u5DF1\u8CAC\u4EFB\u3067\u3002",
l.firstChild?l.insertBefore(d,l.firstChild):l.appendChild(d)}}n=`<!DOCTYPE html>
`+(s.documentElement?s.documentElement.outerHTML:n)}catch{n=n.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
"<!-- blocked polyfill.io script for safety -->")}return n}a(sanitizeHtmlForPreview,"sanitizeHtmlFor\
Preview");function wrapTextWave(e){return e?e.split("").map((t,n)=>`<span class="wave-char" style="a\
nimation-delay: ${n*.028}s">${escapeHtml(t)}</span>`).join(""):""}a(wrapTextWave,"wrapTextWave");function getPendingSkeletonKind(e){
let t=String(e||"").toLowerCase();if(!t)try{t=String(get("model-select")&&get("model-select").value||
"").toLowerCase()}catch{t=""}return t.includes("video")?"video":t.includes("tts")||t.includes("trans\
cribe")||t.includes("realtime")||t.includes("voice")||t.includes("native-audio")||t.includes("live")&&
t.includes("gemini")?"audio":t.includes("gpt-image")||t.includes("imagine-image")||t.includes("image")&&
!t.includes("vision")||t.includes("gemini")&&(t.includes("image")||t.includes("nano"))?"image":t.includes(
"ocr")||t.includes("mistral-ocr")?"text":t.includes("build")||t.includes("code-fast")||t.includes("c\
oding")?"code":"text"}a(getPendingSkeletonKind,"getPendingSkeletonKind");function buildPendingSkeletonBody(e){
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
/div>'}a(buildPendingSkeletonBody,"buildPendingSkeletonBody");function buildPendingSkeletonHtml(e,t){
const n=getPendingSkeletonKind(e),i=t==null||t===""?"\u56DE\u7B54\u3092\u751F\u6210\u4E2D...":String(
t);return`<div class="content-area pending-shimmer skeleton-pending" data-skeleton-kind="${escapeHtml(
n)}">${buildPendingSkeletonBody(n)}<div class="skeleton-status">${escapeHtml(i)}</div></div>`}a(buildPendingSkeletonHtml,
"buildPendingSkeletonHtml");function updatePendingSkeletonStatus(e,t,n){if(!e)return!1;const i=e.querySelector(
".content-area.skeleton-pending");if(!i)return!1;let s=i.querySelector(".skeleton-status");s||(s=document.
createElement("div"),s.className="skeleton-status",i.appendChild(s));const o=t==null?"":String(t),r=n==
null||n===""?"":String(n);return r?s.innerHTML=`${escapeHtml(o)}<span class="skeleton-status-sub">${escapeHtml(
r)}</span>`:s.textContent=o,!0}a(updatePendingSkeletonStatus,"updatePendingSkeletonStatus");function buildChatLoadingSkeletonHtml(){
return`<div class="chat-load-skeleton" role="status" aria-live="polite" aria-label="\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D">${[
{role:"user",widths:["62%","44%"]},{role:"ai",widths:["88%","76%","92%","58%"]},{role:"user",widths:[
"48%"]},{role:"ai",widths:["82%","70%","54%"]}].map((n,i)=>{const s=n.role==="user",o=s?"justify-end":
"justify-start",r=s?"message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4\
 rounded-2xl rounded-tr-none shadow-md relative":"message-bubble chat-load-skeleton-bubble chat-load\
-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative",l=n.widths.map(
(d,p)=>`<div class="skeleton-line" style="width:${d};animation-delay:${(i*.08+p*.06).toFixed(2)}s"><\
/div>`).join("");return`<div class="flex ${o} mb-4 chat-load-skeleton-row" style="animation-delay:${(i*
.07).toFixed(2)}s" aria-hidden="true"><div class="${r}"><div class="content-area pending-shimmer ske\
leton-pending chat-load-skeleton-body" data-skeleton-kind="text"><div class="skeleton-lines">${l}</d\
iv></div></div></div>`}).join("")}<div class="chat-load-skeleton-caption"><span class="chat-load-ske\
leton-caption-dot"></span>\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...</div></div>`}
a(buildChatLoadingSkeletonHtml,"buildChatLoadingSkeletonHtml");function showChatLoadError(e){const t=get(
"chat-container");if(!t)return;t.innerHTML='<div class="min-h-[45vh] flex items-center justify-cente\
r px-4"><div class="max-w-md w-full rounded-2xl border border-red-500/40 bg-red-950/30 p-5 text-cent\
er" role="alert"><i class="fas fa-triangle-exclamation text-red-300 text-xl mb-3"></i><p class="text\
-sm font-semibold text-red-100">\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F</p><p class="mt-2 text-xs text-red-200/80">\u901A\u4FE1\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\
\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p><button type="button" data-chat-load-retry class="mt-4 rounded-lg border border-red\
-300/40 px-4 py-2 text-sm text-red-100 hover:bg-red-500/20"><i class="fas fa-rotate-right mr-1"></i>\
\u518D\u8A66\u884C</button></div></div>';const n=t.querySelector("[data-chat-load-retry]");n&&n.addEventListener(
"click",()=>loadMessages(e))}a(showChatLoadError,"showChatLoadError");function hashString(e){let t=0;
if(!e)return"0";for(let n=0;n<e.length;n++)t=(t<<5)-t+e.charCodeAt(n),t|=0;return Math.abs(t).toString(
36)}a(hashString,"hashString");function decodeCodeButtonValue(e){if(!e)return"";try{return decodeURIComponent(
e)}catch{return""}}a(decodeCodeButtonValue,"decodeCodeButtonValue");function getCodingTargetFromButton(e){
if(!e)return null;const t=decodeCodeButtonValue(e.getAttribute("data-code")||"");if(!t)return null;const n=e.
closest(".code-wrapper"),i=e.closest(".message-group");return{code:t,language:String(e.getAttribute(
"data-coding-lang")||"text").trim().slice(0,40)||"text",key:String(e.getAttribute("data-code-key")||
(n==null?void 0:n.getAttribute("data-code-key"))||hashString(t)),message_id:i!=null&&i.id?i.id.replace(
/^msg-/,""):null,thread_id:currentThreadId?String(currentThreadId):null}}a(getCodingTargetFromButton,
"getCodingTargetFromButton");function findLatestCodingTarget(){const e=get("chat-container");if(!e)return null;
const t=Array.from(e.querySelectorAll(".message-group .coding-target-btn"));for(let n=t.length-1;n>=
0;n--){const i=getCodingTargetFromButton(t[n]);if(i)return i}return null}a(findLatestCodingTarget,"f\
indLatestCodingTarget");function extractPromptCodingTargets(e){const t=String(e||"").replace(/\r\n?/g,
`
`).split(`
`),n=[];let i=null;for(const s of t){if(!i){const l=s.match(/^\s*(`{3,}|~{3,})(.*)$/);if(!l)continue;
const d=String(l[2]||"").trim();i={markerChar:l[1][0],markerLength:l[1].length,language:(d.split(/\s+/)[0]||
"text").replace(/^\{?\.?/,"").replace(/\}$/,"")||"text",buffer:[]};continue}const o=String(s||"").trim();
if(new RegExp(`^\\${i.markerChar}{${i.markerLength},}\\s*$`).test(o)){const l=i.buffer.join(`
`);l.trim()&&n.push({code:l,language:i.language,key:hashString(`prompt\\n${i.language}\\n${l}`),candidate_id:`\
prompt-${n.length+1}`,prompt_index:n.length,message_id:null,thread_id:currentThreadId?String(currentThreadId):
null,prompt_source:!0}),i=null;continue}i.buffer.push(s)}return n}a(extractPromptCodingTargets,"extr\
actPromptCodingTargets");function extractLatestPromptCodingTarget(e){const t=extractPromptCodingTargets(
e);return t.length?t[t.length-1]:null}a(extractLatestPromptCodingTarget,"extractLatestPromptCodingTa\
rget");function collectCodingCandidates(e){if(codingTargetSelection){const o=codingTargetSelection.thread_id;
if(!o||!currentThreadId||String(o)===String(currentThreadId))return[{...codingTargetSelection,candidate_id:"\
selected-1",source:"history",explicit:!0}];codingTargetSelection=null}const t=extractPromptCodingTargets(
e),n=new Set(t.map(o=>`${o.language}
${o.code}`)),i=get("chat-container"),s=[];return i&&Array.from(i.querySelectorAll(".message-group .c\
oding-target-btn")).forEach(o=>{const r=getCodingTargetFromButton(o);if(!r)return;const l=`${r.language}\

${r.code}`;n.has(l)||(n.add(l),s.push(r))}),s.slice(-20).forEach((o,r)=>{t.push({...o,candidate_id:`\
history-${r+1}`,source:"history",explicit:!1})}),t}a(collectCodingCandidates,"collectCodingCandidate\
s");function resolveCodingTarget(e=null){var s;const t=String(e===null?((s=get("prompt-input"))==null?
void 0:s.value)||"":e||"");if(codingTargetSelection){const o=codingTargetSelection.thread_id;if(!o||
!currentThreadId||String(o)===String(currentThreadId))return{...codingTargetSelection,explicit:!0};codingTargetSelection=
null}const n=extractLatestPromptCodingTarget(t);if(n)return{...n,explicit:!1};const i=findLatestCodingTarget();
return i?{...i,explicit:!1}:null}a(resolveCodingTarget,"resolveCodingTarget");function syncCodingTargetButtons(e=document){
if(!e||typeof e.querySelectorAll!="function")return;const t=codingTargetSelection?String(codingTargetSelection.
key||""):"";e.querySelectorAll(".coding-target-btn").forEach(n=>{const i=!!t&&String(n.getAttribute(
"data-code-key")||"")===t;n.classList.toggle("coding-target-active",i),n.setAttribute("aria-pressed",
i?"true":"false"),n.innerHTML=i?'<i class="fas fa-thumbtack"></i>':'<i class="fas fa-quote-right"></\
i>',n.title=i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A",
n.setAttribute("aria-label",i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A")})}
a(syncCodingTargetButtons,"syncCodingTargetButtons");function syncCodingModeUi(e=codingModeEnabled,t={}){
var d;if(codingModeEnabled=!!e,t.persist!==!1)try{localStorage.setItem(CODING_MODE_STORAGE_KEY,codingModeEnabled?
"true":"false")}catch{}const n=get("enable-coding-mode");n&&n.checked!==codingModeEnabled&&(n.checked=
codingModeEnabled);const i=get("coding-target-bar"),s=get("coding-target-text"),o=get("clear-coding-\
target-btn");i&&i.classList.toggle("visible",codingModeEnabled);const r=resolveCodingTarget(),l=codingTargetSelection?
[r].filter(Boolean):collectCodingCandidates(String(((d=get("prompt-input"))==null?void 0:d.value)||""));
if(codingModeEffective=codingModeEnabled&&l.length>0,s)if(codingTargetSelection&&r)s.textContent=`\u7DE8\u96C6\
\u5BFE\u8C61: ${r.language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`;else if(l.length>1){
const p=l.filter(g=>g.prompt_source).length,h=l.length-p;s.textContent=`\u30E2\u30C7\u30EB\u304C\u7DE8\u96C6\u5BFE\u8C61\u3092\u5224\u65AD: \u5165\u529B${p}\
\u4EF6 / \u5C65\u6B74${h}\u4EF6`}else r&&r.prompt_source?s.textContent=`\u5165\u529B\u4E2D: ${r.language||
"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:r?s.textContent=`\u81EA\u52D5\u9078\u629E: \u6700\u65B0\u306E ${r.
language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:s.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u751F\u6210\u5F8C\u306B\u81EA\u52D5\u6709\u52B9\u5316";
o&&o.classList.toggle("hidden",!codingTargetSelection),syncCodingTargetButtons()}a(syncCodingModeUi,
"syncCodingModeUi");function activateDeferredCodingModeFromStream(e){if(!codingModeEnabled||codingModeEffective||
extractPromptCodingTargets(e).length===0)return!1;codingModeEffective=!0;const t=get("coding-target-\
text");return t&&(t.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u691C\u51FA: \u6B21\u306E\u9001\u4FE1\u304B\u3089\u6709\u52B9"),
!0}a(activateDeferredCodingModeFromStream,"activateDeferredCodingModeFromStream");function selectCodingTargetFromButton(e){
const t=getCodingTargetFromButton(e);if(!t){showToast("\u3053\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u7DE8\u96C6\u5BFE\u8C61\u306B\u3067\u304D\u307E\u305B\u3093",
"error",!0);return}codingTargetSelection=t,syncCodingModeUi(codingModeEnabled,{persist:!1}),codingModeEnabled?
showToast("Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u3057\u307E\u3057\u305F","suc\
cess"):showToast("\u7DE8\u96C6\u5BFE\u8C61\u3092\u9078\u629E\u3057\u307E\u3057\u305F\u3002\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306ECoding\u3092\u30AA\u30F3\u306B\u3059\u308B\u3068\u4F7F\u7528\u3057\u307E\u3059",
"info")}a(selectCodingTargetFromButton,"selectCodingTargetFromButton");function renderCodingDiffLines(e){
return String(e||"").split(`
`).map(t=>{let n="coding-diff-context";return t.startsWith("+++")||t.startsWith("---")?n="coding-dif\
f-file":t.startsWith("@@")?n="coding-diff-hunk":t.startsWith("+")?n="coding-diff-added":t.startsWith(
"-")&&(n="coding-diff-removed"),`<span class="${n}">${escapeHtml(t||" ")}</span>`}).join(`
`)}a(renderCodingDiffLines,"renderCodingDiffLines");function appendCodingLiveDiff(e,t){if(!e||!t||!t.
diff)return;let n=e.querySelector(".coding-live-diff");n||(n=document.createElement("div"),n.className=
"coding-live-diff",n.innerHTML='<div class="coding-live-diff-header"><span><i class="fas fa-code-bra\
nch"></i> Live Code Changes</span><span class="coding-live-diff-count">0 edits</span></div><div clas\
s="coding-live-diff-list"></div>',e.appendChild(n));const i=Math.max(0,Number(t.edit_index||0));if(i&&
n.querySelector(`[data-coding-edit-index="${i}"]`))return;const s=n.querySelector(".coding-live-diff\
-list"),o=document.createElement("div");o.className="coding-live-diff-edit",i&&o.setAttribute("data-\
coding-edit-index",String(i));const r=Number(t.repair_attempt||0)>0?` \xB7 Auto repair ${Number(t.repair_attempt)}`:
"";o.innerHTML=`<div class="coding-live-diff-meta">Edit ${i} \xB7 ${escapeHtml(t.language||"text")}${r}\
</div><pre>${renderCodingDiffLines(t.diff)}</pre>`,s&&s.appendChild(o);const l=n.querySelector(".cod\
ing-live-diff-count"),d=n.querySelectorAll(".coding-live-diff-edit").length;l&&(l.textContent=`${d} \
edit${d===1?"":"s"}`),n.scrollIntoView({block:"nearest",behavior:"smooth"})}a(appendCodingLiveDiff,"\
appendCodingLiveDiff");function isHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
return n==="html"||n==="htm"||n==="xhtml"?!0:n?!1:/<!doctype\s+html/i.test(t||"")}a(isHtmlPreviewCandidate,
"isHtmlPreviewCandidate");function openHtmlCodePreview(e){if(!e)return;let t="";try{t=decodeURIComponent(
e)}catch{showToast("HTML\u30D7\u30EC\u30D3\u30E5\u30FC\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}detectBlockedScriptsInCode(t)&&showToast("\u26A0 \u5371\u967A\u306A\u5916\u90E8\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u691C\u77E5 (polyfill.io \u306A\u3069)\u3002\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\
\u306F\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u958B\u304D\u307E\u3059\u3002","warning",!0);const i=sanitizeHtmlForPreview(
t);openSandboxedHtmlTab(i)}a(openHtmlCodePreview,"openHtmlCodePreview");function snapshotCodeCollapse(e){
if(!e)return[];const t=[];return e.querySelectorAll(".code-wrapper").forEach((n,i)=>{const s=String(
i),o=n.classList.contains("collapsed")||n.getAttribute("data-collapsed")==="true";t.push({key:s,collapsed:o})}),
t}a(snapshotCodeCollapse,"snapshotCodeCollapse");function applyCodeCollapse(e,t=[],n=!1){if(!e)return;
const i=new Map;t.forEach(s=>i.set(s.key,s.collapsed)),e.querySelectorAll(".code-wrapper").forEach((s,o)=>{
const r=String(o),l=i.has(r)?i.get(r):n;s.setAttribute("data-collapsed",l?"true":"false"),s.classList.
toggle("collapsed",!!l);const d=s.querySelector(".code-toggle");d&&(d.setAttribute("aria-expanded",l?
"false":"true"),d.innerHTML=l?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></\
i>',d.title=l?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",d.setAttribute("aria-label",l?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080"))})}a(applyCodeCollapse,"applyCodeCollapse");function snapshotCodeCollapseByMessage(e){
if(!e)return new Map;const t=new Map;return e.querySelectorAll(".message-group").forEach(n=>{const i=n.
getAttribute("id")||"";n.querySelectorAll(".code-wrapper").forEach((s,o)=>{const r=s.getAttribute("d\
ata-code-key")||String(o),l=s.classList.contains("collapsed")||s.getAttribute("data-collapsed")==="t\
rue";t.set(`${i}:${r}`,l)})}),t}a(snapshotCodeCollapseByMessage,"snapshotCodeCollapseByMessage");function applyCodeCollapseByMessage(e,t,n=!1){
e&&e.querySelectorAll(".message-group").forEach(i=>{const s=i.getAttribute("id")||"";i.querySelectorAll(
".code-wrapper").forEach((o,r)=>{const l=o.getAttribute("data-code-key")||String(r),d=`${s}:${l}`,p=t&&
t.has(d)?t.get(d):n;o.setAttribute("data-collapsed",p?"true":"false"),o.classList.toggle("collapsed",
!!p);const h=o.querySelector(".code-toggle");h&&(h.setAttribute("aria-expanded",p?"false":"true"),h.
innerHTML=p?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></i>',h.title=p?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080",h.setAttribute("aria-label",p?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080"))})})}
a(applyCodeCollapseByMessage,"applyCodeCollapseByMessage");function buildTokenTotals(e){const t={tokens_total:0,
tokens_in:0,tokens_out:0,tokens_content:0,tokens_thought:0};let n=!1,i=!1,s=!1,o=!1,r=!1;return(e||[]).
forEach(l=>{if(!l)return;let d=null;l.tokens!==null&&l.tokens!==void 0?d=Number(l.tokens||0):(l.tokens_in!==
null&&l.tokens_in!==void 0||l.tokens_out!==null&&l.tokens_out!==void 0)&&(d=Number(l.tokens_in||0)+Number(
l.tokens_out||0)),d!==null&&(t.tokens_total+=d,n=!0),l.tokens_in!==null&&l.tokens_in!==void 0&&(t.tokens_in+=
Number(l.tokens_in||0),i=!0),l.tokens_out!==null&&l.tokens_out!==void 0&&(t.tokens_out+=Number(l.tokens_out||
0),s=!0),l.tokens_content!==null&&l.tokens_content!==void 0&&(t.tokens_content+=Number(l.tokens_content||
0),o=!0),l.tokens_thought!==null&&l.tokens_thought!==void 0&&(t.tokens_thought+=Number(l.tokens_thought||
0),r=!0)}),{tokens_total:n?t.tokens_total:0,tokens_in:i?t.tokens_in:null,tokens_out:s?t.tokens_out:null,
tokens_content:o?t.tokens_content:null,tokens_thought:r?t.tokens_thought:null}}a(buildTokenTotals,"b\
uildTokenTotals");function updateTotalTokenBar(e,t=null,n=null){const i=get("total-token-bar"),s=get(
"total-token-count"),o=get("total-token-count-all-branches");if(!i||!s)return;const r=Number(e||0),l=Number(
n&&n.tokens_total||0);r>0||l>0?(i.classList.remove("hidden"),s.innerText=`Total: ${r} tokens`,t?(s.classList.
add("cursor-pointer","underline","decoration-dotted"),messageMeta.__total__={tokens_total:r,tokens_in:t.
tokens_in,tokens_out:t.tokens_out,tokens_content:t.tokens_content,tokens_thought:t.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation"},s.onclick=()=>openTokenDetail("__total__")):(s.classList.remove("\
cursor-pointer","underline","decoration-dotted"),s.onclick=null,delete messageMeta.__total__),o&&(n&&
l>0?(o.classList.remove("hidden"),o.classList.add("cursor-pointer","underline","decoration-dotted"),
o.innerText=`All branches: ${l} tokens`,messageMeta.__total_all_branches__={tokens_total:l,tokens_in:n.
tokens_in,tokens_out:n.tokens_out,tokens_content:n.tokens_content,tokens_thought:n.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation (All branches)"},o.onclick=()=>openTokenDetail("__total_all_branche\
s__")):(o.classList.add("hidden"),o.classList.remove("cursor-pointer","underline","decoration-dotted"),
o.innerText="All branches: 0 tokens",o.onclick=null,delete messageMeta.__total_all_branches__))):(i.
classList.add("hidden"),s.innerText="Total: 0 tokens",s.classList.remove("cursor-pointer","underline",
"decoration-dotted"),s.onclick=null,delete messageMeta.__total__,o&&(o.classList.add("hidden"),o.classList.
remove("cursor-pointer","underline","decoration-dotted"),o.innerText="All branches: 0 tokens",o.onclick=
null),delete messageMeta.__total_all_branches__)}a(updateTotalTokenBar,"updateTotalTokenBar");const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS=300;
let promptTokenEstimateTimer=null,promptTokenEstimateAbort=null,promptTokenEstimateSeq=0,promptTokenEstimateLastKey="",
promptTokenEstimateLastData=null;function setPromptTokenEstimateText(e,t="text-gray-400"){const n=get(
"prompt-token-estimate");if(n){if(!e){n.classList.add("hidden"),n.innerText="";return}n.className=`m\
t-1 px-1 text-[10px] ${t}`,n.classList.remove("hidden"),n.innerText=e}}a(setPromptTokenEstimateText,
"setPromptTokenEstimateText");function buildPromptTokenEstimatePayload(){return{model:get("model-sel\
ect")&&get("model-select").value?get("model-select").value:"",message:get("prompt-input")&&get("prom\
pt-input").value?get("prompt-input").value:"",quote_text:currentQuote||"",image_urls:collectImageUrlsForSend()}}
a(buildPromptTokenEstimatePayload,"buildPromptTokenEstimatePayload");function renderPromptTokenEstimate(e,t=null){
const n=t||buildPromptTokenEstimatePayload(),i=!!((n.message||"").trim()||(n.quote_text||"").trim()),
s=Array.isArray(n.image_urls)&&n.image_urls.length>0;if(!i&&!s){setPromptTokenEstimateText("");return}
if(e&&e.pending){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u4E2D...",
"text-gray-500");return}if(!e){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"text-red-300");return}if(!e.countable){setPromptTokenEstimateText("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u8868\u793A\u5BFE\u8C61\u5916\u3067\u3059",
"text-gray-500");return}const o=Number(e.tokens_total||0),r=Number(e.tokens_prompt||0),l=Number(e.tokens_files||
0),d=[];Number(e.files_non_text||0)>0&&d.push(`\u975E\u30C6\u30AD\u30B9\u30C8${e.files_non_text}\u4EF6\u306F0\u63DB\
\u7B97`),Number(e.files_missing||0)>0&&d.push(`\u672A\u691C\u51FA${e.files_missing}\u4EF6`),Number(e.
files_error||0)>0&&d.push(`\u5931\u6557${e.files_error}\u4EF6`);const p=d.length?` \u30FB ${d.join("\
 / ")}`:"";setPromptTokenEstimateText(`\u5165\u529B\u898B\u7A4D: ${o} tokens (\u672C\u6587 ${r} / \u30D5\u30A1\
\u30A4\u30EB ${l})${p}`,"text-cyan-300")}a(renderPromptTokenEstimate,"renderPromptTokenEstimate");function schedulePromptTokenEstimate(e=!1){
const t=buildPromptTokenEstimatePayload(),n=!!((t.message||"").trim()||(t.quote_text||"").trim()),i=Array.
isArray(t.image_urls)&&t.image_urls.length>0;if(!n&&!i){promptTokenEstimateLastKey="",promptTokenEstimateLastData=
null,promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=null),
promptTokenEstimateAbort&&(promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=null),renderPromptTokenEstimate(
null,t);return}const s=JSON.stringify([t.model||"",t.message||"",t.quote_text||"",t.image_urls||[]]);
if(s===promptTokenEstimateLastKey&&promptTokenEstimateLastData){renderPromptTokenEstimate(promptTokenEstimateLastData,
t);return}promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=
null);const o=a(async()=>{promptTokenEstimateAbort&&promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=
new AbortController;const r=++promptTokenEstimateSeq;renderPromptTokenEstimate({pending:!0},t);try{const l=await apiFetch(
CHAT_CONFIG.urls.estimatePromptTokensApi,{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(t),signal:promptTokenEstimateAbort.signal});if(!l.ok)throw new Error(`HTTP ${l.status}`);
const d=await l.json();if(r!==promptTokenEstimateSeq)return;promptTokenEstimateLastKey=s,promptTokenEstimateLastData=
d,renderPromptTokenEstimate(d,t)}catch(l){if(l&&l.name==="AbortError"||r!==promptTokenEstimateSeq)return;
promptTokenEstimateLastKey="",promptTokenEstimateLastData=null,renderPromptTokenEstimate(null,t)}},"\
run");e?o():promptTokenEstimateTimer=setTimeout(o,PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS)}a(schedulePromptTokenEstimate,
"schedulePromptTokenEstimate");function updatePromptPlaceholder(){const e=get("prompt-input");e&&(editingMessageId?
e.placeholder="\u7DE8\u96C6\u4E2D... (Enter\u9001\u4FE1\u306F\u8A2D\u5B9A\u306B\u5F93\u3044\u307E\u3059)":
enterToSend?e.placeholder="Enter \u3067\u9001\u4FE1 (Shift+Enter \u3067\u6539\u884C)":e.placeholder=
"Ctrl + Enter \u3067\u9001\u4FE1...")}a(updatePromptPlaceholder,"updatePromptPlaceholder");function readPromptBarModeFromForm(){
return get("set-minimal-prompt-mode")&&get("set-minimal-prompt-mode").checked?{compact_prompt_mode:!1,
minimal_prompt_mode:!0}:get("set-compact-prompt-mode")&&get("set-compact-prompt-mode").checked?{compact_prompt_mode:!0,
minimal_prompt_mode:!1}:{compact_prompt_mode:!1,minimal_prompt_mode:!1}}a(readPromptBarModeFromForm,
"readPromptBarModeFromForm");function writePromptBarModeToForm(e,t){const n=get("set-prompt-bar-mode\
-normal"),i=get("set-compact-prompt-mode"),s=get("set-minimal-prompt-mode");t&&s?s.checked=!0:e&&i?i.
checked=!0:n&&(n.checked=!0)}a(writePromptBarModeToForm,"writePromptBarModeToForm");function placeModelSelectorButton(){
const e=get("model-selector-btn"),t=get("top-model-bar"),n=get("prompt-primary-controls"),i=get("mod\
el-select");if(!(!e||!t||!n)){if(minimalPromptMode){e.parentElement!==t&&t.appendChild(e);return}if(i&&
i.parentElement===n){e.previousElementSibling!==i&&i.insertAdjacentElement("afterend",e);return}e.parentElement!==
n&&n.insertBefore(e,n.firstChild)}}a(placeModelSelectorButton,"placeModelSelectorButton");function applyMinimalPromptMode(){
const e=!!minimalPromptMode;document.body.classList.toggle("minimal-prompt-mode",e);const t=get("top\
-model-bar");t&&(t.classList.toggle("hidden",!e),t.classList.toggle("flex",e));const n=get("upload-b\
tn"),i=n?n.querySelector("i"):null;i&&(i.className=e?"fas fa-plus":"fas fa-paperclip"),n&&(n.title=e?
"\u30AA\u30D7\u30B7\u30E7\u30F3":"Upload"),e||(closeMinimalOptions(),hideThinkingSlider()),placeModelSelectorButton()}
a(applyMinimalPromptMode,"applyMinimalPromptMode");function applyPromptControlMode(){const e=get("pr\
ompt-details-controls"),t=get("prompt-controls-toggle-btn"),n=get("prompt-controls-toggle-text"),i=get(
"prompt-controls-toggle-icon"),s=get("prompt-controls-row");if(applyMinimalPromptMode(),!e||!t)return;
const o=compactPromptMode&&!minimalPromptMode,r=!o||promptControlsExpanded;s&&s.classList.toggle("co\
mpact-collapsed",o&&!r),o?r?(e.classList.remove("collapsed"),e.classList.add("expanded"),e.classList.
remove("hidden")):(e.classList.remove("expanded"),e.classList.add("collapsed")):(e.classList.remove(
"hidden"),e.classList.remove("collapsed"),e.classList.remove("expanded")),o?(t.classList.remove("hid\
den"),t.classList.add("inline-flex"),t.setAttribute("aria-expanded",r?"true":"false"),n&&(n.textContent=
r?"\u6298\u308A\u305F\u305F\u3080":"\u8A73\u7D30"),i&&(i.className=r?"fas fa-chevron-up text-[10px]":
"fas fa-chevron-down text-[10px]")):(t.classList.add("hidden"),t.classList.remove("inline-flex"),t.setAttribute(
"aria-expanded","true"),n&&(n.textContent="\u8A73\u7D30"),i&&(i.className="fas fa-chevron-down text-\
[10px]"))}a(applyPromptControlMode,"applyPromptControlMode");function setCompactPromptMode(e,t=!1){compactPromptMode=
!!e,compactPromptMode&&(minimalPromptMode=!1),compactPromptMode?t||(promptControlsExpanded=!1):promptControlsExpanded=
!0,applyPromptControlMode()}a(setCompactPromptMode,"setCompactPromptMode");function setMinimalPromptMode(e){
minimalPromptMode=!!e,minimalPromptMode&&(compactPromptMode=!1,promptControlsExpanded=!1),applyPromptControlMode()}
a(setMinimalPromptMode,"setMinimalPromptMode");function togglePromptControlDetails(){compactPromptMode&&
(promptControlsExpanded=!promptControlsExpanded,applyPromptControlMode())}a(togglePromptControlDetails,
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
containerId:"sys-prompt-option",gear:!0,gearAction:a(()=>{window.openThreadModal&&window.openThreadModal()},
"gearAction")},{key:"thinking",icon:"fa-brain",label:"Thinking",checkboxId:"enable-thinking",containerId:"\
thinking-options",special:"thinking"},{key:"effort",icon:"fa-sliders-h",label:"Effort",containerId:"\
reasoning-effort-container",selectId:"reasoning-effort"},{key:"safety",icon:"fa-shield-halved",label:"\
Safety",selectId:"safety-setting"},{key:"promptcache",icon:"fa-database",label:"PromptCache",checkboxId:"\
enable-prompt-cache",containerId:"prompt-cache-container"},{key:"compress",icon:"fa-compress-alt",label:"\
Compress",checkboxId:"enable-compression",containerId:"compression-option",gear:!0,gearAction:a(()=>{
window.openCompressionModal&&window.openCompressionModal()},"gearAction")},{key:"tempchat",icon:"fa-\
hourglass-half",label:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8",checkboxId:"enable-temporary-chat",containerId:"\
temporary-chat-container",gear:!0,gearAction:a(()=>openTemporaryChatSettings(),"gearAction")}];let minimalOptionsOpen=!1,
thinkingSliderOpen=!1,thinkingSliderTimer=null,thinkingSliderStartY=0,thinkingSliderStartX=0,thinkingSliderDragging=!1,
thinkingSliderAxis=null,popupSwipeStartY=0,popupSwipeStartX=0,popupSwipeDragging=!1,popupSwipeAtTop=!1,
popupSwipeAxis=null;const minimalPanelOrigins=new Map;function minimalOptionVisible(e){if(e.containerId){
const t=get(e.containerId);if(!t||t.classList.contains("hidden"))return!1}return!0}a(minimalOptionVisible,
"minimalOptionVisible");function minimalOptionDisabled(e){if(e.special==="thinking"){const t=get(e.containerId);
return!!(t&&t.classList.contains("pointer-events-none"))}if(e.checkboxId){const t=get(e.checkboxId);
if(t&&t.disabled)return!0}if(e.containerId){const t=get(e.containerId);if(t&&t.classList.contains("p\
ointer-events-none"))return!0}return!1}a(minimalOptionDisabled,"minimalOptionDisabled");function minimalOptionChecked(e){
if(!e.checkboxId)return!1;const t=get(e.checkboxId);return!!t&&t.checked}a(minimalOptionChecked,"min\
imalOptionChecked");function buildMinimalOptionItem(e){const t=document.createElement("div");t.className=
"minimal-option-item",t.dataset.key=e.key,e.action&&t.classList.add("action-"+e.action),minimalOptionChecked(
e)?t.classList.add("on"):t.classList.add("off"),minimalOptionDisabled(e)&&t.classList.add("disabled");
const n=document.createElement("i");n.className="fas "+e.icon+" minimal-option-icon",t.appendChild(n);
const i=document.createElement("span");if(i.className="minimal-option-label",i.textContent=e.label,t.
appendChild(i),e.selectId){const s=get(e.selectId);if(s){const o=s.cloneNode(!0);o.removeAttribute("\
id"),o.className="minimal-option-select",o.addEventListener("change",()=>{s.value=o.value,s.dispatchEvent(
new Event("change",{bubbles:!0})),refreshMinimalOptionItems()}),t.appendChild(o)}}if(e.gear){const s=document.
createElement("button");s.type="button",s.className="minimal-option-gear",s.title=e.label+"\u8A2D\u5B9A";
const o=document.createElement("i");o.className="fas fa-cog",s.appendChild(o),s.addEventListener("cl\
ick",r=>{r.stopPropagation(),closeMinimalOptions(),typeof e.gearAction=="function"&&e.gearAction()}),
t.appendChild(s)}return t.addEventListener("click",()=>handleMinimalOptionClick(e)),t}a(buildMinimalOptionItem,
"buildMinimalOptionItem");function renderMinimalOptionItems(){const e=get("minimal-options-items");if(!e)
return;const t=document.createDocumentFragment();MINIMAL_POPUP_ITEMS.forEach(n=>{minimalOptionVisible(
n)&&t.appendChild(buildMinimalOptionItem(n))}),e.innerHTML="",e.appendChild(t)}a(renderMinimalOptionItems,
"renderMinimalOptionItems");function refreshMinimalOptionItems(){const e=get("minimal-options-items");
if(!e||!minimalOptionsOpen)return;const t=e.querySelectorAll(".minimal-option-item"),n={};t.forEach(
i=>{n[i.dataset.key]=i}),MINIMAL_POPUP_ITEMS.forEach(i=>{const s=n[i.key];if(s){if(!minimalOptionVisible(
i)){s.classList.add("hidden");return}if(s.classList.remove("hidden"),s.classList.toggle("on",minimalOptionChecked(
i)),s.classList.toggle("off",!minimalOptionChecked(i)),s.classList.toggle("disabled",minimalOptionDisabled(
i)),i.selectId){const o=get(i.selectId),r=s.querySelector(".minimal-option-select");o&&r&&document.activeElement!==
r&&r.value!==o.value&&(r.value=o.value)}}})}a(refreshMinimalOptionItems,"refreshMinimalOptionItems");
function handleMinimalOptionClick(e){if(e.action==="upload"){closeMinimalOptions(),openUploadModal();
return}if(e.action==="button"){const n=get(e.buttonId);closeMinimalOptions(),n&&n.click();return}if(e.
special==="thinking"){const n=get(e.checkboxId);if(n&&!n.disabled){const i=!n.checked;n.checked=i,n.
dispatchEvent(new Event("change",{bubbles:!0})),i?(closeMinimalOptions(),showThinkingSlider()):hideThinkingSlider(),
refreshMinimalOptionItems()}else closeMinimalOptions(),showThinkingSlider();return}if(minimalOptionDisabled(
e)||e.selectId)return;const t=get(e.checkboxId);t&&(t.disabled||(t.checked=!t.checked,t.dispatchEvent(
new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),e.key==="fast"?(closeMinimalOptions(),
setTimeout(()=>refreshMinimalOptionItems(),350)):e.key==="tempchat"&&setTimeout(()=>refreshMinimalOptionItems(),
350)))}a(handleMinimalOptionClick,"handleMinimalOptionClick");function moveModelPanelsIntoPopup(){const e=get(
"minimal-options-model-body");if(!e)return;let t=!1;MINIMAL_MODEL_PANEL_IDS.forEach(n=>{const i=get(
n);if(i){if(i.parentElement===e){i.classList.contains("hidden")||(t=!0);return}minimalPanelOrigins.has(
i)||(minimalPanelOrigins.set(i,{parent:i.parentElement,next:i.nextSibling}),e.appendChild(i),i.classList.
contains("hidden")||(t=!0))}}),refreshMinimalModelSection()}a(moveModelPanelsIntoPopup,"moveModelPan\
elsIntoPopup");function restoreModelPanelsFromPopup(){get("minimal-options-model-body")&&(minimalPanelOrigins.
forEach((t,n)=>{t.parent&&t.parent.contains(n)&&(t.next&&t.next.parentNode===t.parent?t.parent.insertBefore(
n,t.next):t.parent.appendChild(n))}),minimalPanelOrigins.clear())}a(restoreModelPanelsFromPopup,"res\
toreModelPanelsFromPopup");function refreshMinimalModelSection(){const e=get("minimal-options-model-\
body"),t=get("minimal-options-model-section");if(!e||!t)return;let n=!1;Array.from(e.children).forEach(
i=>{i.classList.contains("hidden")||(n=!0)}),t.classList.toggle("hidden",!n)}a(refreshMinimalModelSection,
"refreshMinimalModelSection");function openMinimalOptions(){if(minimalOptionsOpen||!minimalPromptMode)
return;hideThinkingSlider(),minimalOptionsOpen=!0,renderMinimalOptionItems(),moveModelPanelsIntoPopup();
const e=get("minimal-options-popup");if(!e)return;const t=get("minimal-options-panel");t&&(t.style.cssText=
""),e.classList.remove("minimal-options-closing","minimal-options-open"),e.classList.remove("hidden"),
e.setAttribute("aria-hidden","false"),e.offsetWidth,e.classList.add("minimal-options-open")}a(openMinimalOptions,
"openMinimalOptions");function closeMinimalOptions(){if(!minimalOptionsOpen)return;minimalOptionsOpen=
!1;const e=get("minimal-options-popup");e&&(e.classList.add("minimal-options-closing"),e.setAttribute(
"aria-hidden","true"),setTimeout(()=>{minimalOptionsOpen||(e.classList.remove("minimal-options-open",
"minimal-options-closing"),e.classList.add("hidden"))},560)),restoreModelPanelsFromPopup(),hideThinkingSlider()}
a(closeMinimalOptions,"closeMinimalOptions");function toggleMinimalOptions(){minimalOptionsOpen?closeMinimalOptions():
openMinimalOptions()}a(toggleMinimalOptions,"toggleMinimalOptions");function refreshMinimalOptionsIfOpen(){
minimalOptionsOpen&&(renderMinimalOptionItems(),refreshMinimalModelSection())}a(refreshMinimalOptionsIfOpen,
"refreshMinimalOptionsIfOpen");function allowedThinkingValues(){const e=get("thinking-level");return e?
Array.from(e.options).filter(n=>!n.disabled&&!n.classList.contains("hidden")).map(n=>n.value):THINKING_LEVELS.
map(n=>n.value)}a(allowedThinkingValues,"allowedThinkingValues");function thinkingIndexFromValue(e){
const t=THINKING_LEVELS.findIndex(n=>n.value===e);return t<0?3:t}a(thinkingIndexFromValue,"thinkingI\
ndexFromValue");function syncThinkingSliderUi(){const e=get("thinking-slider"),t=get("thinking-slide\
-value"),n=get("thinking-level"),i=thinkingIndexFromValue(n?n.value:"high");e&&(e.value=String(i)),t&&
(t.textContent=THINKING_LEVELS[i].label)}a(syncThinkingSliderUi,"syncThinkingSliderUi");function scheduleThinkingSliderHide(){
thinkingSliderTimer&&clearTimeout(thinkingSliderTimer),thinkingSliderTimer=setTimeout(()=>{thinkingSliderTimer=
null,hideThinkingSlider()},2500)}a(scheduleThinkingSliderHide,"scheduleThinkingSliderHide");function showThinkingSlider(){
if(thinkingSliderOpen){scheduleThinkingSliderHide();return}const e=get("thinking-slide-bar");if(!e)return;
const t=get("thinking-slide-inner");t&&(t.style.transform=""),thinkingSliderOpen=!0,e.classList.remove(
"hidden"),e.setAttribute("aria-hidden","false"),syncThinkingSliderUi(),e.offsetWidth,e.classList.add(
"thinking-slide-open"),scheduleThinkingSliderHide()}a(showThinkingSlider,"showThinkingSlider");function hideThinkingSlider(){
thinkingSliderTimer&&(clearTimeout(thinkingSliderTimer),thinkingSliderTimer=null);const e=get("think\
ing-slide-bar");e&&(thinkingSliderOpen=!1,e.classList.remove("thinking-slide-open"),e.setAttribute("\
aria-hidden","true"),setTimeout(()=>{thinkingSliderOpen||e.classList.add("hidden");const t=get("thin\
king-slide-inner");t&&(t.style.transform="")},360))}a(hideThinkingSlider,"hideThinkingSlider");function bindMinimalOptionsEvents(){
const e=get("minimal-options-backdrop"),t=get("minimal-options-close-btn"),n=get("minimal-options-po\
pup");n&&n.parentNode!==document.body&&document.body.appendChild(n),e&&e.addEventListener("click",()=>closeMinimalOptions()),
t&&t.addEventListener("click",()=>closeMinimalOptions()),document.addEventListener("keydown",l=>{if(l.
key==="Escape"){if(minimalOptionsOpen){closeMinimalOptions();return}thinkingSliderOpen&&hideThinkingSlider()}});
const i=get("thinking-slider");i&&i.addEventListener("input",()=>{const l=Number(i.value),d=allowedThinkingValues(),
p=get("thinking-level");if(d.length){const h=d.map(y=>thinkingIndexFromValue(y)),g=h.includes(l)?l:h.
reduce((y,b)=>Math.abs(b-l)<Math.abs(y-l)?b:y,h[0]);p&&(p.value=THINKING_LEVELS[g].value,p.dispatchEvent(
new Event("change",{bubbles:!0})))}syncThinkingSliderUi(),scheduleThinkingSliderHide()});const s=get(
"thinking-slide-close-btn");s&&s.addEventListener("click",l=>{l.stopPropagation(),hideThinkingSlider()});
const o=get("thinking-slide-bar");if(o){const l=get("thinking-slide-inner");o.addEventListener("touc\
hstart",d=>{thinkingSliderOpen&&(thinkingSliderDragging=!0,thinkingSliderStartY=d.touches[0].clientY,
thinkingSliderStartX=d.touches[0].clientX,thinkingSliderAxis=null,l&&l.classList.add("dragging"))},{
passive:!0}),o.addEventListener("touchmove",d=>{if(!thinkingSliderDragging)return;const p=d.touches[0].
clientX-thinkingSliderStartX,h=d.touches[0].clientY-thinkingSliderStartY;if(thinkingSliderAxis===null&&
(Math.abs(p)>8||Math.abs(h)>8)&&(thinkingSliderAxis=Math.abs(h)>Math.abs(p)?"v":"h"),thinkingSliderAxis===
"v")if(h>0){d.cancelable&&d.preventDefault();const g=Math.min((h-8)*.5,120);l&&(l.style.transform=g>
0?`translateY(${g}px)`:"")}else l&&(l.style.transform="")},{passive:!1}),o.addEventListener("touchen\
d",d=>{if(!thinkingSliderDragging)return;thinkingSliderDragging=!1;const p=d.changedTouches[0].clientY-
thinkingSliderStartY;l&&l.classList.remove("dragging"),thinkingSliderAxis==="v"&&p>100?(l&&(l.style.
transform=`translateY(${Math.max(p*.5,60)}px)`),hideThinkingSlider()):(l&&(l.style.transform=""),scheduleThinkingSliderHide())},
{passive:!0}),o.addEventListener("touchcancel",()=>{thinkingSliderDragging=!1,l&&(l.classList.remove(
"dragging"),l.style.transform=""),scheduleThinkingSliderHide()},{passive:!0})}const r=get("minimal-o\
ptions-panel");r&&(r.addEventListener("touchstart",l=>{if(!minimalOptionsOpen)return;popupSwipeDragging=
!0,popupSwipeStartY=l.touches[0].clientY,popupSwipeStartX=l.touches[0].clientX,popupSwipeAxis=null;let d=l.
target instanceof Element?l.target:null,p=!0;for(;d&&d!==r;){if(d.scrollTop>0){p=!1;break}d=d.parentElement}
popupSwipeAtTop=p,p&&r.classList.add("dragging")},{passive:!0}),r.addEventListener("touchmove",l=>{if(!popupSwipeDragging||
!popupSwipeAtTop||!minimalOptionsOpen)return;const d=l.touches[0].clientX-popupSwipeStartX,p=l.touches[0].
clientY-popupSwipeStartY;popupSwipeAxis===null&&(Math.abs(d)>8||Math.abs(p)>8)&&(popupSwipeAxis=Math.
abs(p)>Math.abs(d)?"v":"h"),popupSwipeAxis==="v"&&p>0&&(l.cancelable&&l.preventDefault(),r.style.transform=
`translateY(${Math.min(p*.6,140)}px)`)},{passive:!1}),r.addEventListener("touchend",l=>{if(!popupSwipeDragging)
return;popupSwipeDragging=!1;const d=l.changedTouches[0].clientY-popupSwipeStartY;r.classList.remove(
"dragging"),popupSwipeAtTop&&popupSwipeAxis!=="h"&&d>70?(r.style.transform=`translateY(${Math.max(d*
.6,100)}px)`,r.style.opacity="0",closeMinimalOptions()):r.style.transform=""},{passive:!0}),r.addEventListener(
"touchcancel",()=>{popupSwipeDragging=!1,r.classList.remove("dragging"),r.style.transform="",r.style.
opacity=""},{passive:!0}))}a(bindMinimalOptionsEvents,"bindMinimalOptionsEvents");function bindUploadButton(){
const e=get("upload-btn");e&&(e.onclick=()=>{minimalPromptMode?toggleMinimalOptions():openUploadModal()})}
a(bindUploadButton,"bindUploadButton");function applyChatDefaults(e){if(!e||(Object.prototype.hasOwnProperty.
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
default_safety_setting},i=a((s,o)=>s==null||s===""?o:s,"s");n.model&&selectModelById(n.model),get("e\
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
chatDefaultsLoaded=!0,toggleOptions(),applyMcpPromptChipUi()}a(applyChatDefaults,"applyChatDefaults");
function setEditUi(e){const t=get("edit-bar");t&&(e?(t.classList.remove("hidden"),t.classList.add("f\
lex")):(t.classList.add("hidden"),t.classList.remove("flex")),updatePromptPlaceholder())}a(setEditUi,
"setEditUi");function cancelEdit(){editingMessageId=null,currentParentId=currentLeafId||null;const e=get(
"prompt-input");e&&(e.value="",e.style.height="auto"),currentImageUrls=[],get("file-preview").classList.
add("hidden"),get("file-input").value="",clearQuote(),setEditUi(!1)}a(cancelEdit,"cancelEdit");function beginEditMessage(e,t=!1){
const n=messageStore[e];if(n==null)return;const i=get("prompt-input");i.value=n||"",i.focus(),i.style.
height="auto",i.style.height=i.scrollHeight+"px";const s=allMessages.find(d=>d.id==e),o=messageMeta[e]||
{};s?currentParentId=s.parent_id===void 0?null:s.parent_id:o.parent_id!==void 0&&(currentParentId=o.
parent_id),editingMessageId=e,setEditUi(!0);const r=s?s.image_url:o.image_url;if(r)try{const d=JSON.
parse(r);Array.isArray(d)&&d.length?(currentImageUrls=d.map(p=>{let h="unknown",g=p;p&&typeof p=="ob\
ject"&&(h=normalizeAttachmentSource(p.source),g=p.filepath||p.path||p.url||p.file||"");const y=normalizeAttachmentPath(
g);return y&&setAttachmentSourceForPath(y,h),y}).filter(Boolean),get("file-preview").classList.remove(
"hidden"),get("file-name").innerText=`${currentImageUrls.length} files ready`):(currentImageUrls=[],
get("file-preview").classList.add("hidden"),get("file-input").value="")}catch{currentImageUrls=[],get(
"file-preview").classList.add("hidden"),get("file-input").value=""}else currentImageUrls=[],get("fil\
e-preview").classList.add("hidden"),get("file-input").value="";const l=s?s.quote_text:o.quote_text;l?
(currentQuote=l,get("quote-text-display").innerText=currentQuote,get("quote-bar").classList.add("vis\
ible")):clearQuote(),schedulePromptTokenEstimate(!0),t&&sendMessage()}a(beginEditMessage,"beginEditM\
essage");function playSendAnimation(){const e=get("send-btn");e&&(e.classList.remove("fly"),e.offsetWidth,
e.classList.add("fly"))}a(playSendAnimation,"playSendAnimation");function setSendBtnToStopMode(){const e=get(
"send-btn");if(!e)return;e.onclick=stopGeneration,isStopMode=!0,e.disabled=!1;const t=a(()=>{!e||!isStopMode||
(e.classList.add("stop-mode"),e.innerHTML='<span style="font-size:20px;line-height:1;color:#fff;">\u25A0<\
/span>',e.classList.add("btn-swap"),setTimeout(()=>e.classList.remove("btn-swap"),300))},"applyStopU\
i");if(e.classList.contains("fly")){const n=a(i=>{i.animationName==="sendBtnPop"&&(e.removeEventListener(
"animationend",n),t())},"onEnd");e.addEventListener("animationend",n),setTimeout(t,700)}else t()}a(setSendBtnToStopMode,
"setSendBtnToStopMode");function setSendBtnToSendMode(){const e=get("send-btn");e&&(e.classList.remove(
"stop-mode","fly","btn-swap"),e.innerHTML='<i class="fas fa-paper-plane"></i>',e.classList.add("btn-\
swap"),setTimeout(()=>e.classList.remove("btn-swap"),300),e.onclick=sendMessage,isStopMode=!1)}a(setSendBtnToSendMode,
"setSendBtnToSendMode");async function stopGeneration(){const e=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null,t=normalizeJobIdForUi(currentJobId),n=++manualStopSeq,i=captureStoppedPartialBubbleSnapshot(
getActiveStreamingBubbleElement());manualStopContext={seq:n,threadId:e,jobId:t,partialSnapshot:i},t&&
suppressPendingJob(t),abortController&&abortController.abort();try{if(t||e){const s={};t&&(s.job_id=
t),e&&(s.thread_id=e);const r=await(await apiFetch("/api/stop_chat",{method:"POST",headers:{"Content\
-Type":"application/json"},body:JSON.stringify(s)})).json().catch(()=>({})),l=normalizeJobIdForUi(r&&
r.job_id);l&&(suppressPendingJob(l),manualStopContext&&manualStopContext.seq===n&&(manualStopContext.
jobId=l))}manualStopContext&&manualStopContext.seq===n&&await syncThreadAfterAbortedStream(e,{retries:2,
retryDelayMs:180,notifyOnFailure:!0})&&manualStopContext.partialSnapshot&&appendStoppedPartialBubbleSnapshot(
manualStopContext.partialSnapshot,e)}finally{manualStopContext&&manualStopContext.seq===n&&(manualStopContext=
null),setSendBtnToSendMode(),updateFilePreview()}}a(stopGeneration,"stopGeneration");async function purgeCaches(){
if("caches"in window){const e=await caches.keys();await Promise.all(e.map(t=>caches.delete(t)))}if(navigator.
serviceWorker){const e=await navigator.serviceWorker.getRegistrations();await Promise.all(e.map(t=>t.
unregister()))}}a(purgeCaches,"purgeCaches");const SW_CACHE_MODE_STORAGE_KEY="ai_sw_cache_mode_v2";async function applyCacheMode(e,t={}){
if("serviceWorker"in navigator)if(e)try{await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(
appVersion)}`),localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY,"enabled")}catch{}else{const n=localStorage.
getItem(SW_CACHE_MODE_STORAGE_KEY);(!!t.forceCleanup||n!=="disabled")&&await purgeCaches(),localStorage.
setItem(SW_CACHE_MODE_STORAGE_KEY,"disabled")}}a(applyCacheMode,"applyCacheMode");function checkAndNotifyVersion(e){
!e||!appVersion||e===appVersion||(localStorage.getItem("version_notified")||"")===e||(localStorage.setItem(
"app_version",e),syncVersionUpdateCachePreferenceUi(),showModal("version-update-modal"))}a(checkAndNotifyVersion,
"checkAndNotifyVersion");async function checkVersion(){try{const e=await fetch("/api/version",{cache:"\
no-store"});if(!e.ok)return;const n=(await e.json()).version||"",i=localStorage.getItem("app_version")||
"";n&&!i&&localStorage.setItem("app_version",n),n&&i&&n!==i&&(await purgeCaches(),checkAndNotifyVersion(
n))}catch{}}a(checkVersion,"checkVersion");async function fetchChatStreamWithUnavailableRetry(e,t,n){
let i=0;for(;;){if(t.signal&&t.signal.aborted)throw new DOMException("Aborted","AbortError");try{const s=await apiFetch(
e,t),o=window.ConnectionMonitor.retryModeForResponse(s);let r=!1;if(s.status===425&&(r=(await s.clone().
json().catch(()=>({}))).code==="submission_in_progress"),!o&&!r)return window.ConnectionMonitor.markReachable(),
s;i+=1,o&&window.ConnectionMonitor.setUnavailable(o),updatePendingSkeletonStatus(n,o==="maintenance"?
"\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u7D42\u4E86\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...":"\u30B5\u30FC\u30D0\
\u30FC\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}catch(s){if(t.signal&&t.signal.aborted||s.name==="AbortError")throw s;i+=1,window.
ConnectionMonitor.setUnavailable("offline"),updatePendingSkeletonStatus(n,"\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u63A5\u7D9A\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}await window.ConnectionMonitor.waitForRetry(t.signal)}}a(fetchChatStreamWithUnavailableRetry,
"fetchChatStreamWithUnavailableRetry");function createClientRequestId(){return window.crypto&&typeof window.
crypto.randomUUID=="function"?window.crypto.randomUUID():`req-${window.crypto&&typeof window.crypto.
getRandomValues=="function"?Array.from(window.crypto.getRandomValues(new Uint32Array(4))).map(t=>t.toString(
16)).join(""):`${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`}`.slice(0,64)}a(createClientRequestId,
"createClientRequestId");async function reconnectPendingStreamUntilAvailable(e,t){const n=t!=null?String(
t):"",i=normalizeJobIdForUi(e&&e.job_id),s=i||`thread:${n}`;if(!n||pendingStreamReconnectJobs.has(s))
return;pendingStreamReconnectJobs.add(s);const o=new AbortController;let r=!1;abortController=o,currentJobId=
i,setSendBtnToStopMode();try{for(;!o.signal.aborted;){if(String(currentThreadId||"")!==n||i&&isPendingJobSuppressed(
i))return;const l=getActiveStreamingBubbleElement();if(updatePendingSkeletonStatus(l,"\u30B5\u30FC\u30D0\u30FC\u3078\u306E\u518D\u63A5\u7D9A\u3092\u5F85\u3063\u3066\u3044\
\u307E\u3059...","\u56DE\u7B54\u51E6\u7406\u306F\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u7D99\u7D9A\u3057\u3066\u3044\u307E\u3059"),
await window.ConnectionMonitor.waitForRetry(o.signal),!await loadMessages(n,{preserveDraft:!0,silent:!0,
skipHistory:!0})){window.ConnectionMonitor.probeNow();continue}const p=currentThreadPending;p&&p.job_id&&
!isPendingJobSuppressed(p.job_id)?(abortController===o&&(abortController=null),r=!0,resumePendingStream(
p)):window.ConnectionMonitor.markReachable();return}}catch(l){l.name!=="AbortError"&&sendClientDebugLog(
"error",`Stream reconnect failed: ${l.message}`)}finally{pendingStreamReconnectJobs.delete(s),abortController===
o&&(abortController=null),r||(currentJobId=null,setSendBtnToSendMode(),updateFilePreview())}}a(reconnectPendingStreamUntilAvailable,
"reconnectPendingStreamUntilAvailable"),window.initTurnstileWidget=()=>{if(!botConfig||!botConfig.turnstileSiteKey||
!window.turnstile||turnstileWidgetId!==null)return;const e=document.getElementById("turnstile-contai\
ner");e&&(e.classList.remove("hidden"),turnstileWidgetId=window.turnstile.render(e,{sitekey:botConfig.
turnstileSiteKey,size:"compact",appearance:"interaction-only",callback:a(t=>{turnstileToken=t,turnstilePending=
!1,verifyTurnstileOnServer(t)},"callback"),"expired-callback":a(()=>{turnstileToken=null,turnstilePending=
!1},"expired-callback"),"error-callback":a(()=>{turnstileToken=null,turnstilePending=!1},"error-call\
back")}),isBotDetectionActive()&&runBotDetectionGate())};async function getTurnstileToken(e=1500){if(!botConfig||
!botConfig.turnstileSiteKey)return null;if(turnstileToken)return turnstileToken;if(!window.turnstile)
return null;if(botDetectionOverlayShown&&botDetectionDialogWidgetId!==null)return turnstilePending=!0,
await new Promise(n=>{const i=turnstileToken,s=setTimeout(()=>n(null),Math.max(500,Number(e)||1500)),
o=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(s),clearInterval(o),n(turnstileToken))},
50)});if(turnstileWidgetId===null)return null;const t=document.getElementById("turnstile-container");
return t&&t.classList.remove("hidden"),turnstilePending=!0,await new Promise(n=>{const i=turnstileToken,
s=setTimeout(()=>n(null),Math.max(500,Number(e)||1500));try{window.turnstile.execute(turnstileWidgetId)}catch{
clearTimeout(s),n(null);return}const o=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(
s),clearInterval(o),verifyTurnstileOnServer(turnstileToken),n(turnstileToken))},50)})}a(getTurnstileToken,
"getTurnstileToken");function resetTurnstileToken(){if(turnstileToken=null,turnstilePending=!1,window.
turnstile&&turnstileWidgetId!==null)try{window.turnstile.reset(turnstileWidgetId)}catch{}if(window.turnstile&&
botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}}a(resetTurnstileToken,
"resetTurnstileToken");function isBotDetectionActive(){return!!(botConfig&&botConfig.globalEnabled&&
botConfig.accountEnabled&&!isAdminUser&&botConfig.turnstileSiteKey)}a(isBotDetectionActive,"isBotDet\
ectionActive");function renderBotDetectionDialogWidget(){if(botDetectionDialogWidgetId!==null||!botConfig||
!botConfig.turnstileSiteKey)return;const e=document.getElementById("bot-detection-widget-box");if(e){
if(!window.turnstile){setTimeout(renderBotDetectionDialogWidget,250);return}try{botDetectionDialogWidgetId=
window.turnstile.render(e,{sitekey:botConfig.turnstileSiteKey,theme:"dark",size:"flexible",callback:a(
t=>{turnstileToken=t,turnstilePending=!1,verifyTurnstileOnServer(t,!0,!0)},"callback"),"expired-call\
back":a(()=>{if(turnstileToken=null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.
turnstile.reset(botDetectionDialogWidgetId)}catch{}},"expired-callback"),"error-callback":a(()=>{if(turnstileToken=
null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}},
"error-callback")})}catch(t){console.error("bot-detection dialog widget error",t)}}}a(renderBotDetectionDialogWidget,
"renderBotDetectionDialogWidget");function showBotDetectionOverlay(e=""){let t=document.getElementById(
"bot-detection-overlay");if(t)t.style.display="flex";else{t=document.createElement("div"),t.id="bot-\
detection-overlay",t.style.cssText="position:fixed;inset:0;z-index:2147483000;background:rgba(3,7,18\
,0.92);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;";const i=document.
createElement("div");i.style.cssText="max-width:420px;width:100%;background:#0f172a;border:1px solid\
 #334155;border-radius:12px;padding:24px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.5);dis\
play:flex;flex-direction:column;align-items:stretch;gap:12px;";const s=document.createElement("div");
s.id="bot-detection-overlay-title",s.style.cssText="font-weight:700;font-size:15px;color:#f1f5f9;",s.
textContent=e||"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u4E2D...";const o=document.createElement("div");
o.style.cssText="font-size:12px;color:#94a3b8;line-height:1.6;",o.textContent="\u81EA\u52D5\u30A2\u30AF\u30BB\u30B9\u9632\u6B62\u306E\u305F\u3081\u3001\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u3066\u304F\u3060\
\u3055\u3044\u3002";const r=document.createElement("div");r.id="bot-detection-widget-box",r.style.cssText=
"margin-top:8px;min-height:65px;display:flex;justify-content:center;",i.appendChild(s),i.appendChild(
o),i.appendChild(r),t.appendChild(i),document.body.appendChild(t)}const n=document.getElementById("b\
ot-detection-overlay-title");e&&n&&(n.textContent=e),botDetectionOverlayShown=!0,renderBotDetectionDialogWidget()}
a(showBotDetectionOverlay,"showBotDetectionOverlay");function hideBotDetectionOverlay(){if(botDetectionOverlayShown=
!1,botDetectionDialogWidgetId!==null){try{window.turnstile.remove(botDetectionDialogWidgetId)}catch{}
botDetectionDialogWidgetId=null}const e=document.getElementById("bot-detection-widget-box");e&&e.replaceChildren();
const t=document.getElementById("bot-detection-overlay");t&&t.remove()}a(hideBotDetectionOverlay,"hi\
deBotDetectionOverlay");let botLockOverlay=null,botLockTimer=null;function showBotLockOverlay(e="\u9001\u4FE1\u64CD\
\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",t=600){
hideBotDetectionOverlay();let n=document.getElementById("bot-lock-overlay");if(n){n.style.display="f\
lex";const i=document.getElementById("bot-lock-overlay-message");i&&e&&(i.textContent=e)}else{n=document.
createElement("div"),n.id="bot-lock-overlay",n.style.cssText="position:fixed;inset:0;z-index:2147483\
000;background:rgba(3,7,18,0.94);display:flex;flex-direction:column;align-items:center;justify-conte\
nt:center;padding:24px;";const i=document.createElement("div");i.style.cssText="max-width:440px;widt\
h:100%;background:#0f172a;border:1px solid #f59e0b;border-radius:12px;padding:24px;text-align:center\
;box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;align-items:center;gap:12p\
x;";const s=document.createElement("div");s.style.cssText="font-size:26px;color:#fbbf24;",s.innerHTML=
'<i class="fas fa-lock"></i>';const o=document.createElement("div");o.id="bot-lock-overlay-title",o.
style.cssText="font-weight:700;font-size:16px;color:#fbbf24;",o.textContent="\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F";
const r=document.createElement("div");r.id="bot-lock-overlay-message",r.style.cssText="font-size:13p\
x;color:#f1f5f9;line-height:1.7;",r.textContent=e;const l=document.createElement("div");l.id="bot-lo\
ck-overlay-timer",l.style.cssText="font-size:12px;color:#94a3b8;margin-top:2px;";const d=document.createElement(
"div");d.style.cssText="font-size:11px;color:#94a3b8;line-height:1.6;",d.textContent="\u30ED\u30C3\u30AF\u89E3\u9664\u307E\u3067\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\
\u304F\u3060\u3055\u3044\u3002\u540C\u3058\u64CD\u4F5C\u3092\u7E70\u308A\u8FD4\u3059\u3068BAN\u3055\u308C\u308B\u5834\u5408\u304C\u3042\u308A\u307E\u3059\u3002",
i.appendChild(s),i.appendChild(o),i.appendChild(r),i.appendChild(l),i.appendChild(d),n.appendChild(i),
document.body.appendChild(n)}return botLockOverlay=n,updateBotLockTimer(t),n}a(showBotLockOverlay,"s\
howBotLockOverlay");function updateBotLockTimer(e){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const t=document.getElementById("bot-lock-overlay-timer");if(!t)return;const n=a(()=>{const i=Math.
max(0,Math.round(Number(e)||0)),s=Math.floor(i/60),o=String(i%60).padStart(2,"0");t.textContent=`\u30ED\u30C3\u30AF\
\u89E3\u9664\u307E\u3067: ${s}:${o}`},"render");n(),botLockTimer=setInterval(()=>{e-=1,n(),e<=0&&(botLockTimer&&
(clearInterval(botLockTimer),botLockTimer=null),location.reload())},1e3)}a(updateBotLockTimer,"updat\
eBotLockTimer");function hideBotLockOverlay(){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const e=document.getElementById("bot-lock-overlay");e&&e.remove(),botLockOverlay=null}a(hideBotLockOverlay,
"hideBotLockOverlay");async function applyBotLockFromServer(e){if(isAdminUser)return!0;let t=600;try{
const n=await apiFetch("/api/bot/lock",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({reason:e||""})});if(n.status===403){let s=null;try{s=await n.json()}catch{}if(s&&s.error===
"banned")return showToast("\u30ED\u30C3\u30AF\u304C\u7E70\u308A\u8FD4\u3055\u308C\u305F\u305F\u3081BAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800),!1}const i=await n.json().catch(()=>({}));
if(i&&(i.status==="skipped"||i.skipped))return!0;i&&typeof i.remaining_seconds=="number"&&(t=i.remaining_seconds)}catch{}
return showBotLockOverlay(e||"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",
t),!1}a(applyBotLockFromServer,"applyBotLockFromServer");const runBotDetectionGate=a(()=>botDetectionVerified||
!isBotDetectionActive()?Promise.resolve(!0):botDetectionGatePromise||(botDetectionGatePromise=(async()=>{
let e=0;for(;!botDetectionVerified;){if(!botDetectionOverlayShown){if(!window.__turnstileApiLoaded||
turnstileWidgetId===null){await new Promise(s=>setTimeout(s,1e3));continue}const n=await getTurnstileToken(
8e3);if(n&&await verifyTurnstileOnServer(n,!0,!1))break;e+=1;let i=!1;try{i=!!(botTelemetry&&botTelemetry.
looksSuspicious&&botTelemetry.looksSuspicious())}catch{}(e>=2||i)&&showBotDetectionOverlay();continue}
const t=await getTurnstileToken(25e3);if(t&&await verifyTurnstileOnServer(t,!0,!0))break;try{botTelemetry.
send(!0,{forceReport:!0})}catch{}await new Promise(n=>setTimeout(n,5e3))}return hideBotDetectionOverlay(),
!0})().finally(()=>{botDetectionGatePromise=null}),botDetectionGatePromise),"runBotDetectionGate");function registerSendButtonSpam(){
const e=performance.now();return sendButtonSpamTimestamps.push(e),sendButtonSpamTimestamps=sendButtonSpamTimestamps.
filter(t=>e-t<=3e3),sendButtonSpamTimestamps.length}a(registerSendButtonSpam,"registerSendButtonSpam");
function resetSendButtonSpam(){sendButtonSpamTimestamps=[]}a(resetSendButtonSpam,"resetSendButtonSpa\
m");async function runSendSpamVerification(){return isBotDetectionActive()?await applyBotLockFromServer(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002"):
!0}a(runSendSpamVerification,"runSendSpamVerification");let turnstileServerVerifiedAt=0,turnstileVerifyInFlight=null,
turnstileVerifyInFlightToken=null,turnstileLastSubmittedToken=null;async function verifyTurnstileOnServer(e,t=!1,n=null){
if(!e||!isBotDetectionActive()||botDetectionVerified)return!0;n===null&&(n=botDetectionOverlayShown);
const i=Date.now();if(!t&&i-turnstileServerVerifiedAt<60*1e3)return!0;if(turnstileVerifyInFlight&&turnstileVerifyInFlightToken===
e)return turnstileVerifyInFlight;if(turnstileLastSubmittedToken===e&&!t)return!!botDetectionVerified;
if(turnstileLastSubmittedToken===e)return turnstileVerifyInFlight&&turnstileVerifyInFlightToken===e?
turnstileVerifyInFlight:!!botDetectionVerified;turnstileLastSubmittedToken=e,turnstileVerifyInFlightToken=
e;const s=!!n;return turnstileVerifyInFlight=(async()=>{try{return(await apiFetch("/api/bot/turnstil\
e-verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({turnstile_token:e,
challenged:s})})).ok?(turnstileServerVerifiedAt=Date.now(),botDetectionVerified=!0,hideBotDetectionOverlay(),
!0):!1}catch{return!1}finally{turnstileVerifyInFlightToken===e&&(turnstileVerifyInFlight=null,turnstileVerifyInFlightToken=
null)}})(),turnstileVerifyInFlight}a(verifyTurnstileOnServer,"verifyTurnstileOnServer");function botTurnstileTokenForRequest(){
return isBotDetectionActive()?turnstileToken:null}a(botTurnstileTokenForRequest,"botTurnstileTokenFo\
rRequest");const botTelemetry=(()=>{const e={enabled:!1,windowStart:performance.now(),lastSend:0,clicks:0,
keys:0,moves:0,fastClicks:0,fastKeys:0,untrustedInput:!1,clickTimes:[],keyTimes:[],clickIntervals:[],
lastClickTs:0,lastKeyTs:0,lastMove:null,speedMax:0,speedSum:0,speedSamples:0,lastMoveSample:0},t=a(()=>{
e.enabled=!!(botConfig&&botConfig.globalEnabled&&botConfig.accountEnabled&&!isAdminUser)},"refreshEn\
abled"),n=a(()=>{e.windowStart=performance.now(),e.clicks=0,e.keys=0,e.moves=0,e.fastClicks=0,e.fastKeys=
0,e.untrustedInput=!1,e.clickTimes=[],e.keyTimes=[],e.clickIntervals=[],e.speedMax=0,e.speedSum=0,e.
speedSamples=0},"resetWindow"),i=a(y=>{const b=y&&y.target;return!b||typeof b.closest!="function"?!1:
!!b.closest("[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay")},
"isControlClick"),s=a(y=>{if(i(y))return;if(y&&y.isTrusted===!1){e.untrustedInput=!0,p(!0);return}const b=performance.
now();if(e.clicks+=1,e.lastClickTs){const w=b-e.lastClickTs;e.clickIntervals.push(w),e.clickIntervals.
length>10&&e.clickIntervals.shift(),w<120&&(e.fastClicks+=1)}e.lastClickTs=b,e.clickTimes.push(b),e.
clickTimes=e.clickTimes.filter(w=>b-w<=2e3),e.fastClicks>=4&&p(!0)},"recordClick"),o=a(y=>{if(y&&y.isTrusted===
!1){e.untrustedInput=!0,p(!0);return}const b=performance.now();e.keys+=1,e.lastKeyTs&&b-e.lastKeyTs<
50&&(e.fastKeys+=1),e.lastKeyTs=b,e.keyTimes.push(b),e.keyTimes=e.keyTimes.filter(w=>b-w<=2e3)},"rec\
ordKey"),r=a(y=>{const b=performance.now();if(!(b-e.lastMoveSample<80)){if(e.lastMoveSample=b,e.moves+=
1,e.lastMove){const w=y.clientX-e.lastMove.x,x=y.clientY-e.lastMove.y,S=b-e.lastMove.t;if(S>0){const T=Math.
sqrt(w*w+x*x)/(S/1e3);e.speedMax=Math.max(e.speedMax,T),e.speedSum+=T,e.speedSamples+=1}}e.lastMove=
{x:y.clientX,y:y.clientY,t:b}}},"recordMove"),l=a(()=>{const y=Math.max(1,performance.now()-e.windowStart),
b=e.clickTimes.length,w=e.keyTimes.length,x=e.speedSamples?e.speedSum/e.speedSamples:0;let S=0,T=1;if(e.
clickIntervals.length>=3){const E=e.clickIntervals.reduce((J,X)=>J+X,0)/e.clickIntervals.length,F=e.
clickIntervals.reduce((J,X)=>J+Math.pow(X-E,2),0)/e.clickIntervals.length;S=E,T=E>0?Math.sqrt(F)/E:1}
return{window_ms:Math.round(y),clicks:e.clicks,keys:e.keys,moves:e.moves,fast_clicks:e.fastClicks,fast_keys:e.
fastKeys,untrusted_input:!!e.untrustedInput,click_burst:b,key_burst:w,avg_click_ms:S,click_cv:T,event_rate:(e.
clicks+e.keys+e.moves)/(y/1e3),pointer_speed_max:e.speedMax,pointer_speed_avg:x}},"computeStats"),d=a(
y=>y.fast_clicks>=4||y.fast_keys>=8||y.click_burst>=8||y.key_burst>=14||y.event_rate>=20||y.avg_click_ms>
0&&y.avg_click_ms<160&&y.click_cv<.08,"isSuspicious"),p=a(async(y=!1,b={})=>{if(!e.enabled)return;const w=performance.
now();if(!y&&w-e.lastSend<3e3)return;e.lastSend=w;const x=l();if(!(!b.forceReport&&x.clicks+x.keys+x.
moves===0&&!x.untrusted_input)&&!(!y&&!x.untrusted_input&&!d(x))){x.turnstile_token=await getTurnstileToken(),
botConfig&&botConfig.turnstileSiteKey&&!x.turnstile_token&&!botDetectionVerified&&botDetectionOverlayShown&&
(x.turnstile_failed=!0,x.challenged=!0);try{const S=await apiFetch("/api/bot-telemetry",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(x)});if(S.status===403){let T=null;
try{T=await S.json()}catch{}if(T&&T.error==="banned"){showToast("\u30DC\u30C3\u30C8\u5224\u5B9A\u306B\u3088\u308ABAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800);return}}}catch{}resetTurnstileToken(),n()}},
"send");return{start:a(()=>{t(),e.enabled&&(typeof window.PointerEvent!="undefined"?document.addEventListener(
"pointerdown",s,!0):document.addEventListener("click",s,!0),document.addEventListener("keydown",o,!0),
document.addEventListener("wheel",()=>{e.moves+=1},{passive:!0}),document.addEventListener("mousemov\
e",r,!0),setInterval(()=>p(!1),4e3))},"start"),refreshEnabled:t,send:p,looksSuspicious:a(()=>{if(!e.
enabled)return!1;const y=l();return d(y)},"looksSuspicious")}})();function openFileViewer(e,t=""){if(!e)
return;const n=(t||e).split(".").pop().toLowerCase(),i=["png","jpg","jpeg","webp","gif"],s=["mp4","m\
ov","mkv","avi","m4v","webm"],o=["mp3","wav","m4a","ogg","flac"],r=["pdf","txt","md","csv","log","js\
on","docx"];if(i.includes(n)){openImageViewer(e);return}const l=get("file-viewer"),d=get("file-viewe\
r-body"),p=get("file-viewer-title");if(!(!l||!d||!p)){if(p.textContent=t||"File Preview",d.replaceChildren(),
s.includes(n)){const h=document.createElement("video");h.src=String(e),h.controls=!0,h.playsInline=!0,
h.preload="metadata",d.appendChild(h)}else if(o.includes(n)){const h=document.createElement("audio");
h.src=String(e),h.controls=!0,d.appendChild(h)}else if(r.includes(n)){const h=document.createElement(
"iframe");h.src=String(e),h.setAttribute("sandbox",""),h.referrerPolicy="no-referrer",d.appendChild(
h)}else{const h=document.createElement("div");h.className="fallback",h.appendChild(document.createTextNode(
"\u3053\u306E\u5F62\u5F0F\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u304D\u307E\u305B\u3093\u3002"));
const g=document.createElement("div");g.className="mt-3 flex justify-center gap-2";const y=document.
createElement("a");y.href=String(e),y.download="",y.className="px-3 py-1 bg-gray-800 text-white roun\
ded text-xs border border-gray-700",y.textContent="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9";const b=document.
createElement("a");b.href=String(e),b.target="_blank",b.rel="noopener noreferrer",b.className=y.className,
b.textContent="\u65B0\u3057\u3044\u30BF\u30D6\u3067\u958B\u304F",g.append(y,b),h.appendChild(g),d.appendChild(
h)}l.classList.add("visible")}}a(openFileViewer,"openFileViewer");function closeFileViewer(){const e=get(
"file-viewer"),t=get("file-viewer-body");!e||!t||(t.innerHTML="",e.classList.remove("visible"))}a(closeFileViewer,
"closeFileViewer");function showToast(e,t="error",n=!1,i=null){const s=get("toast-stack");if(!s)return;
for(;s.children.length>=3;)s.removeChild(s.firstChild);const o=document.createElement("div");return o.
className=`toast ${t}${i?" toast-clickable":""}`,o.innerHTML=`<i class="fas ${t==="error"?"fa-triang\
le-exclamation":"fa-circle-info"}"></i><span class="flex-1">${escapeHtml(e)}</span><button aria-labe\
l="close"><i class="fas fa-times"></i></button>`,o.querySelector("button").onclick=r=>{r.stopPropagation(),
o.remove()},i&&o.addEventListener("click",i),s.appendChild(o),n||setTimeout(()=>{o.parentNode&&o.remove()},
7e3),o}a(showToast,"showToast");function showProgressToast(e,t="info"){const n=get("toast-stack");if(!n)
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
            `,i.querySelector("button").onclick=()=>i.remove(),n.appendChild(i),{update:a(s=>{const o=i.
querySelector(".progress-bar"),r=i.querySelector(".progress-text");o&&(o.style.width=`${Math.min(100,
Math.max(0,s))}%`),r&&(r.innerText=`${Math.round(s)}%`)},"update"),remove:a(()=>{i.parentNode&&i.remove()},
"remove")}}a(showProgressToast,"showProgressToast");let activeSettingsTab="general";const TAB_LABELS={
general:"\u4E00\u822C",api:"API\u30AD\u30FC",prompt:"\u30D7\u30ED\u30F3\u30D7\u30C8",display:"\u8868\u793A",
data:"\u30C7\u30FC\u30BF",account:"\u30A2\u30AB\u30A6\u30F3\u30C8",security:"\u30BB\u30AD\u30E5\u30EA\u30C6\u30A3",
"2fa":"2\u8981\u7D20\u8A8D\u8A3C",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",mcp:"MCP"},ALL_TABS=[
"general","api","prompt","display","data","account","security","2fa","feedback","mcp"];function getSectionHeading(e){
const t=e.querySelector("h3");if(t)return t.textContent.trim();const n=e.querySelector(".font-bold");
if(n&&!n.querySelector("input")&&!n.querySelector("select"))return n.textContent.trim();const i=e.querySelector(
"label");if(i){const s=i.textContent.trim().replace(/[：:].*$/,"").trim();if(s)return s}return""}a(
getSectionHeading,"getSectionHeading");function getSectionSnippet(e,t){const n=e.textContent,s=n.toLowerCase().
indexOf(t.toLowerCase());if(s===-1)return"";const o=Math.max(0,s-25),r=Math.min(n.length,s+t.length+
35);let l=n.substring(o,r).replace(/\s+/g," ").trim();return o>0&&(l="\u2026"+l),r<n.length&&(l=l+"\u2026"),
l}a(getSectionSnippet,"getSectionSnippet");function removeSearchOverlays(){ALL_TABS.forEach(e=>{const t=get(
"tab-"+e);if(!t)return;const n=t.querySelector(".settings-search-overlay");n&&n.remove(),Array.from(
t.children).forEach(i=>{i.classList.contains("settings-no-results")||(i.style.display="")})})}a(removeSearchOverlays,
"removeSearchOverlays");function filterSettings(){const e=get("settings-search");if(!e)return;const t=e.
value.trim().toLowerCase(),n=get("settings-search-clear");if(n&&n.classList.toggle("hidden",!t),removeSearchOverlays(),
!t){ALL_TABS.forEach(l=>{const d=get("btn-tab-"+l);if(d){const h=d.querySelector(".settings-search-b\
adge");h&&h.remove()}const p=get("tab-"+l);p&&p.classList.toggle("hidden",l!==activeSettingsTab)});return}
let i=[];ALL_TABS.forEach(l=>{const d=get("tab-"+l);d&&(d.classList.add("hidden"),Array.from(d.children).
forEach(p=>{if(!(p.classList.contains("settings-no-results")||p.classList.contains("settings-search-\
overlay"))&&p.textContent.toLowerCase().includes(t)){const h=getSectionHeading(p)||l,g=getSectionSnippet(
p,t);i.push({tabId:l,title:h,snippet:g,element:p})}}))});let s=activeSettingsTab;if(!i.some(l=>l.tabId===
s)){const l=i.find(d=>d.tabId);l&&(s=l.tabId)}const o=get("tab-"+s);if(!o)return;o.classList.remove(
"hidden"),Array.from(o.children).forEach(l=>{l.classList.contains("settings-no-results")||l.classList.
contains("settings-search-overlay")||(l.style.display="none")});const r=document.createElement("div");
if(r.className="settings-search-overlay",i.length===0){const l=document.createElement("div");l.className=
"settings-empty-state",l.innerHTML='<div class="settings-empty-icon"><i class="fas fa-search"></i></\
div><div class="settings-empty-title">\u4E00\u81F4\u3059\u308B\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093</div>';
const d=document.createElement("div");d.className="settings-empty-sub",d.textContent="\u300C"+t+"\u300D\u306B\u4E00\
\u81F4\u3059\u308B\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3002",l.appendChild(
d),r.appendChild(l)}else{const l=document.createElement("div");l.className="settings-search-count",l.
textContent=i.length+"\u4EF6\u306E\u4E00\u81F4",r.appendChild(l);let d=null;i.forEach((p,h)=>{if(p.tabId!==
d){if(d!==null){const S=document.createElement("div");S.className="border-t border-gray-700/50 my-1.\
5",r.appendChild(S)}if(p.tabId!==s){const S=document.createElement("div");S.className="text-[10px] t\
ext-gray-500 px-1 pb-1 font-bold",S.textContent="\u25BC "+(TAB_LABELS[p.tabId]||p.tabId),r.appendChild(
S)}d=p.tabId}const g=document.createElement("div");g.className="settings-search-result-item flex ite\
ms-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150",g.style.animation=
"fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both",g.style.animationDelay=h*30+"ms";const y=document.
createElement("span");y.className="settings-result-tab-badge shrink-0 mt-0.5",y.textContent=TAB_LABELS[p.
tabId]||p.tabId;const b=document.createElement("div");b.className="min-w-0 flex-1";const w=document.
createElement("div");w.className="text-sm font-bold text-white truncate",w.textContent=p.title;const x=document.
createElement("div");x.className="text-[11px] text-gray-400 truncate mt-0.5",x.textContent=p.snippet,
b.appendChild(w),b.appendChild(x),g.appendChild(y),g.appendChild(b),g.addEventListener("click",()=>jumpToSetting(
p.tabId,p.element)),r.appendChild(g)})}o.insertBefore(r,o.firstChild)}a(filterSettings,"filterSettin\
gs");function jumpToSetting(e,t){const n=get("settings-search");n&&(n.value=""),removeSearchOverlays(),
filterSettings(),e!==activeSettingsTab&&switchTab(e),setTimeout(()=>{t.scrollIntoView({behavior:"smo\
oth",block:"center"}),t.classList.add("settings-jump-highlight"),setTimeout(()=>t.classList.remove("\
settings-jump-highlight"),2e3)},260)}a(jumpToSetting,"jumpToSetting");function clickTab(e){const t=get(
"settings-search");t&&(t.value=""),switchTab(e)}a(clickTab,"clickTab");function switchTab(e){if(e===
activeSettingsTab||!ALL_TABS.includes(e))return;const t=get("tab-"+activeSettingsTab);t&&(t.classList.
remove("tab-enter"),t.classList.add("tab-exit"),setTimeout(()=>{t.classList.add("hidden"),t.classList.
remove("tab-exit")},170)),ALL_TABS.forEach(n=>{const i=get("btn-tab-"+n),s=get("tab-"+n);if(n===e){if(s&&
(s.classList.remove("hidden"),s.classList.remove("tab-exit"),s.classList.remove("tab-enter"),s.offsetWidth,
s.classList.add("tab-enter")),i){i.classList.add("is-active");try{i.scrollIntoView({inline:"nearest",
block:"nearest",behavior:"smooth"})}catch{}}}else i&&i.classList.remove("is-active")}),activeSettingsTab=
e,filterSettings(),refreshSettingsTabsScroll()}a(switchTab,"switchTab");function getSettingsTabsMaxScroll(e){
return e?Math.max(0,e.scrollWidth-e.clientWidth):0}a(getSettingsTabsMaxScroll,"getSettingsTabsMaxScr\
oll");function syncSettingsTabsOverflow(){const e=get("settings-tabs-wrap"),t=get("settings-tabs"),n=get(
"settings-tabs-arrow-left"),i=get("settings-tabs-arrow-right");if(!e||!t)return;const s=getSettingsTabsMaxScroll(
t),o=t.scrollLeft,r=s>2&&o>2,l=s>2&&o<s-2;e.classList.toggle("can-scroll",s>2),e.classList.toggle("c\
an-scroll-left",r),e.classList.toggle("can-scroll-right",l),n&&(n.disabled=!r,n.setAttribute("aria-h\
idden",r?"false":"true")),i&&(i.disabled=!l,i.setAttribute("aria-hidden",l?"false":"true"))}a(syncSettingsTabsOverflow,
"syncSettingsTabsOverflow");function refreshSettingsTabsScroll(){initSettingsTabsScroll(),syncSettingsTabsOverflow()}
a(refreshSettingsTabsScroll,"refreshSettingsTabsScroll");function initSettingsTabsScroll(){const e=get(
"settings-tabs-wrap"),t=get("settings-tabs"),n=get("settings-tabs-arrow-left"),i=get("settings-tabs-\
arrow-right");if(!e||!t||!n||!i)return;if(e.dataset.scrollBound==="1"){syncSettingsTabsOverflow();return}
e.dataset.scrollBound="1";const s=56;let o=0,r=0,l=0;const d=a(b=>{const w=e.getBoundingClientRect();
if(!w.width)return;const x=b-w.left;e.classList.toggle("is-edge-left",x>=0&&x<=s),e.classList.toggle(
"is-edge-right",x>=w.width-s&&x<=w.width)},"updateEdgeHover"),p=a(()=>{l||e.classList.remove("is-edg\
e-left","is-edge-right")},"clearEdgeHover"),h=a((b,w)=>{const x=getSettingsTabsMaxScroll(t);if(x<=0||
!b)return;const S=Math.max(0,Math.min(x,t.scrollLeft+b));w&&typeof t.scrollTo=="function"?t.scrollTo(
{left:S,behavior:"smooth"}):t.scrollLeft=S,syncSettingsTabsOverflow()},"scrollTabsBy"),g=a(()=>{l=0,
o&&(clearTimeout(o),o=0),r&&(cancelAnimationFrame(r),r=0)},"stopHold"),y=a(b=>{g(),l=b,e.classList.toggle(
"is-edge-left",b<0),e.classList.toggle("is-edge-right",b>0),h(b*Math.max(120,t.clientWidth*.55),!0),
o=setTimeout(()=>{const w=a(()=>{l&&(h(l*14,!1),r=requestAnimationFrame(w))},"step");r=requestAnimationFrame(
w)},280)},"startHold");if(e.addEventListener("pointermove",b=>{b.pointerType!=="touch"&&d(b.clientX)}),
e.addEventListener("pointerenter",b=>{b.pointerType!=="touch"&&d(b.clientX)}),e.addEventListener("po\
interleave",b=>{b.pointerType!=="touch"&&(g(),p())}),e.addEventListener("wheel",b=>{const w=getSettingsTabsMaxScroll(
t);if(w<=2)return;const S=Math.abs(b.deltaY)>=Math.abs(b.deltaX)?b.deltaY:b.deltaX;if(!S)return;const T=Math.
max(0,Math.min(w,t.scrollLeft+S));T!==t.scrollLeft&&(b.preventDefault(),t.scrollLeft=T,syncSettingsTabsOverflow())},
{passive:!1}),n.addEventListener("pointerdown",b=>{b.button!=null&&b.button!==0||(b.preventDefault(),
y(-1))}),i.addEventListener("pointerdown",b=>{b.button!=null&&b.button!==0||(b.preventDefault(),y(1))}),
n.addEventListener("click",b=>{b.preventDefault(),b.stopPropagation()}),i.addEventListener("click",b=>{
b.preventDefault(),b.stopPropagation()}),window.addEventListener("pointerup",g),window.addEventListener(
"pointercancel",g),window.addEventListener("blur",g),t.addEventListener("scroll",syncSettingsTabsOverflow,
{passive:!0}),window.addEventListener("resize",syncSettingsTabsOverflow),typeof ResizeObserver!="und\
efined")try{const b=new ResizeObserver(()=>syncSettingsTabsOverflow());b.observe(t),b.observe(e)}catch{}
syncSettingsTabsOverflow()}a(initSettingsTabsScroll,"initSettingsTabsScroll"),initSettingsTabsScroll();
const chatContainer=get("chat-container"),scrollToBottomBtn=get("scroll-to-bottom-btn"),CHAT_BOTTOM_THRESHOLD=64;
let chatAutoScrollFrame=0,chatTouchY=null,chatScrollbarDragging=!1,chatManualScrollPaused=!1,chatManualResumeArmed=!1,
chatManualPauseIntent=!1,chatPauseIntentTimer=0,chatLastScrollTop=chatContainer?chatContainer.scrollTop:
0;function isChatNearBottom(){return chatContainer?chatContainer.scrollHeight-chatContainer.scrollTop-
chatContainer.clientHeight<=CHAT_BOTTOM_THRESHOLD:!0}a(isChatNearBottom,"isChatNearBottom");function syncScrollToBottomButton(){
if(!scrollToBottomBtn)return;const e=!userAutoScroll&&!isChatNearBottom();scrollToBottomBtn.classList.
toggle("hidden",!e)}a(syncScrollToBottomButton,"syncScrollToBottomButton");function clearChatAutoScrollPauseIntent(){
chatManualPauseIntent=!1,chatPauseIntentTimer&&(clearTimeout(chatPauseIntentTimer),chatPauseIntentTimer=
0)}a(clearChatAutoScrollPauseIntent,"clearChatAutoScrollPauseIntent");function armChatAutoScrollPause(){
!chatContainer||chatManualScrollPaused||(chatManualPauseIntent=!0,chatPauseIntentTimer&&clearTimeout(
chatPauseIntentTimer),chatPauseIntentTimer=setTimeout(()=>{chatManualPauseIntent=!1,chatPauseIntentTimer=
0},500))}a(armChatAutoScrollPause,"armChatAutoScrollPause");function pauseChatAutoScroll(){chatContainer&&
(chatAutoScrollFrame&&(cancelAnimationFrame(chatAutoScrollFrame),chatAutoScrollFrame=0),clearChatAutoScrollPauseIntent(),
chatManualScrollPaused=!0,chatManualResumeArmed=!1,userAutoScroll=!1,syncScrollToBottomButton())}a(pauseChatAutoScroll,
"pauseChatAutoScroll");function resumeChatAutoScroll(e={}){clearChatAutoScrollPauseIntent(),chatManualScrollPaused=
!1,chatManualResumeArmed=!1,userAutoScroll=!0,chatContainer&&(e.scroll!==!1&&(chatContainer.scrollTop=
chatContainer.scrollHeight),chatLastScrollTop=chatContainer.scrollTop),e.scroll===!1?syncScrollToBottomButton():
scrollToBottom()}a(resumeChatAutoScroll,"resumeChatAutoScroll");function performChatAutoScroll(){chatAutoScrollFrame=
0,!(!chatContainer||!userAutoScroll)&&(chatContainer.scrollTop=chatContainer.scrollHeight,syncScrollToBottomButton())}
a(performChatAutoScroll,"performChatAutoScroll");function scrollToBottom(e=!1){if(chatContainer){if(e&&
(clearChatAutoScrollPauseIntent(),chatManualScrollPaused=!1,chatManualResumeArmed=!1,userAutoScroll=
!0),!userAutoScroll){syncScrollToBottomButton();return}chatAutoScrollFrame||(chatAutoScrollFrame=requestAnimationFrame(
performChatAutoScroll))}}if(a(scrollToBottom,"scrollToBottom"),chatContainer){chatContainer.addEventListener(
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
!1},{passive:!0});const e=new ResizeObserver(()=>scrollToBottom());a(()=>{Array.from(chatContainer.children).
forEach(n=>e.observe(n))},"observeMessageSizes")(),new MutationObserver(n=>{n.forEach(i=>{i.addedNodes.
forEach(s=>{s.nodeType===Node.ELEMENT_NODE&&s.parentElement===chatContainer&&e.observe(s)})}),scrollToBottom()}).
observe(chatContainer,{childList:!0,subtree:!0,characterData:!0})}scrollToBottomBtn&&scrollToBottomBtn.
addEventListener("click",()=>scrollToBottom(!0)),document.addEventListener("keydown",e=>{const t=e.target,
n=t&&(t.matches("input, textarea, select")||t.isContentEditable);!n&&["ArrowUp","PageUp","Home"].includes(
e.key)?armChatAutoScrollPause():!n&&chatManualScrollPaused&&["ArrowDown","PageDown","End"].includes(
e.key)&&(chatManualResumeArmed=!0)});let viewerImages=[],viewerIndex=0,viewerSwipe=null,suppressViewerCloseClick=!1;
function openImageViewer(e,t=".chat-image"){const i=Array.from(document.querySelectorAll(t)).map(o=>({
url:o.dataset.viewerSrc||o.currentSrc||o.src,filename:o.dataset.viewerFilename||o.title||(o.dataset.
viewerSrc||o.currentSrc||o.src).split("/").pop(),element:o})),s=i.findIndex(o=>o.url===e);if(s===-1){
openViewerWithItems([{url:e,filename:e.split("/").pop(),element:null}],0);return}openViewerWithItems(
i,s)}a(openImageViewer,"openImageViewer");function openViewerWithItems(e,t){viewerImages=e,viewerIndex=
t>=0&&t<e.length?t:0,clearViewerAdjacent(),updateViewerState(),get("image-viewer").classList.add("vi\
sible"),document.addEventListener("keydown",handleViewerKeydown)}a(openViewerWithItems,"openViewerWi\
thItems");function closeImageViewer(){get("image-viewer").classList.remove("visible"),document.removeEventListener(
"keydown",handleViewerKeydown),clearViewerAdjacent(),viewerImages=[],viewerIndex=0,viewerSwipe=null}
a(closeImageViewer,"closeImageViewer");function clearViewerAdjacent(){const e=document.querySelector(
".viewer-adjacent");e&&e.remove()}a(clearViewerAdjacent,"clearViewerAdjacent");function renderViewerChrome(){
if(!viewerImages.length)return;const e=get("image-viewer-meta"),t=document.querySelector(".viewer-na\
v.prev"),n=document.querySelector(".viewer-nav.next"),i=viewerImages[viewerIndex];if(e.innerText=`${viewerIndex+
1} / ${viewerImages.length} \u2022 ${i.filename}`,viewerIndex<viewerImages.length-1){const s=new Image;
s.src=viewerImages[viewerIndex+1].url}t.style.display=viewerImages.length>1?"flex":"none",n.style.display=
viewerImages.length>1?"flex":"none",t.style.opacity=viewerIndex>0?"1":"0.3",n.style.opacity=viewerIndex<
viewerImages.length-1?"1":"0.3",t.style.pointerEvents=viewerIndex>0?"auto":"none",n.style.pointerEvents=
viewerIndex<viewerImages.length-1?"auto":"none"}a(renderViewerChrome,"renderViewerChrome");function updateViewerState(e){
if(!viewerImages.length)return;const t=get("image-viewer-img");if(!t)return;const n=viewerImages[viewerIndex],
i=!e||e.fade!==!1;renderViewerChrome(),t.style.transition="none",t.style.transform=i?"scale(0.96)":"\
translateX(0) scale(1)",t.style.opacity=i?"0.35":"0";const s=a(()=>{t.style.transition=i?"transform \
0.28s var(--ease-out), opacity 0.28s var(--ease-out)":"none",t.style.opacity="1",t.style.transform="\
scale(1)",i||clearViewerAdjacent()},"reveal");i?setTimeout(()=>{viewerSwipe&&viewerSwipe.active||(t.
src=n.url,t.onload=s,t.onerror=s,t.complete&&t.naturalWidth&&s())},140):(t.src=n.url,t.onload=s,t.onerror=
s,t.complete&&t.naturalWidth&&s())}a(updateViewerState,"updateViewerState");function navImage(e){const t=viewerIndex+
e;t>=0&&t<viewerImages.length&&(clearViewerAdjacent(),viewerIndex=t,updateViewerState())}a(navImage,
"navImage");function getViewerAdjacent(e){const t=document.querySelector(".viewer-content");if(!t)return null;
const n=viewerIndex+e;if(n<0||n>=viewerImages.length)return null;let i=t.querySelector(".viewer-adja\
cent");return i||(i=document.createElement("img"),i.className="viewer-adjacent",i.alt="",t.appendChild(
i)),i.src=viewerImages[n].url,i.dataset.dir=String(e),i}a(getViewerAdjacent,"getViewerAdjacent");function onViewerTouchStart(e){
if(!viewerImages.length||e.touches.length!==1)return;const t=e.touches[0];viewerSwipe={startX:t.clientX,
startY:t.clientY,lastX:t.clientX,lastY:t.clientY,dx:0,dy:0,vx:0,dir:0,active:!1,resist:!1,adjacent:null,
lastTime:Date.now()}}a(onViewerTouchStart,"onViewerTouchStart");function onViewerTouchMove(e){if(!viewerSwipe)
return;const t=e.touches[0],n=t.clientX-viewerSwipe.startX,i=t.clientY-viewerSwipe.startY,s=Date.now(),
o=Math.max(s-viewerSwipe.lastTime,1),r=(t.clientX-viewerSwipe.lastX)/o;if(viewerSwipe.vx=r*.6+viewerSwipe.
vx*.4,viewerSwipe.lastX=t.clientX,viewerSwipe.lastY=t.clientY,viewerSwipe.lastTime=s,viewerSwipe.dx=
n,viewerSwipe.dy=i,!viewerSwipe.active){if(Math.abs(n)<10&&Math.abs(i)<10)return;if(Math.abs(n)<Math.
abs(i)*1.15){viewerSwipe=null;return}viewerSwipe.active=!0,viewerSwipe.dir=n>0?-1:1,viewerSwipe.adjacent=
getViewerAdjacent(viewerSwipe.dir),viewerSwipe.adjacent||(viewerSwipe.resist=!0)}e.preventDefault();
const l=get("image-viewer-img");if(!l)return;const d=document.querySelector(".viewer-content"),p=d?d.
clientWidth:window.innerWidth,h=viewerSwipe.resist?n*.3:n;l.style.transition="none",l.style.transform=
`translateX(${h}px) scale(${1-Math.min(Math.abs(h)/(p*4),.04)})`,l.style.opacity=String(Math.max(1-Math.
min(Math.abs(h)/(p*.45),.55),.4));const g=viewerSwipe.adjacent;if(g){const y=Number(g.dataset.dir)||
0;g.style.transition="none",g.style.transform=`translate(-50%, -50%) translateX(${y*p+n}px) scale(0.\
97)`,g.style.opacity=String(Math.min(Math.abs(n)/(p*.3),1))}}a(onViewerTouchMove,"onViewerTouchMove");
function onViewerTouchEnd(){if(!viewerSwipe)return;const e=viewerSwipe;if(viewerSwipe=null,!e.active)
return;suppressViewerCloseClick=!0,setTimeout(()=>{suppressViewerCloseClick=!1},120);const t=get("im\
age-viewer-img");if(!t)return;const n=document.querySelector(".viewer-content"),i=n?n.clientWidth:window.
innerWidth,s=i*.22,o=e.dir||(e.dx>0?-1:1),r=window.matchMedia&&window.matchMedia("(prefers-reduced-m\
otion: reduce)").matches,l=!e.resist&&(Math.abs(e.dx)>s||Math.abs(e.vx)>.45&&Math.sign(e.dx)===o),d=e.
adjacent;if(!l){if(t.style.transition="transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out\
)",t.style.transform="translateX(0) scale(1)",t.style.opacity="1",d){const h=d;d.style.transition="t\
ransform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)",d.style.transform=`translate(-50%, -5\
0%) translateX(${o*i}px) scale(0.97)`,d.style.opacity="0",setTimeout(()=>{h.isConnected&&h.remove()},
340)}return}if(r){finishSwipeNav(o);return}const p=o*i;t.style.transition="transform 0.3s var(--ease\
-out), opacity 0.3s var(--ease-out)",t.style.transform=`translateX(${p}px) scale(0.96)`,t.style.opacity=
"0.2",d&&(d.style.transition="transform 0.3s var(--ease-out), opacity 0.3s var(--ease-out)",d.style.
transform="translate(-50%, -50%) translateX(0) scale(1)",d.style.opacity="1"),setTimeout(()=>finishSwipeNav(
o),300)}a(onViewerTouchEnd,"onViewerTouchEnd");function finishSwipeNav(e){if(!viewerImages.length||viewerSwipe&&
viewerSwipe.active)return;const t=get("image-viewer");if(!t||!t.classList.contains("visible")){clearViewerAdjacent();
return}const n=viewerIndex+e;n<0||n>=viewerImages.length||(viewerIndex=n,updateViewerState({fade:!1}))}
a(finishSwipeNav,"finishSwipeNav");function handleViewerKeydown(e){e.key==="ArrowLeft"&&navImage(-1),
e.key==="ArrowRight"&&navImage(1),e.key==="Escape"&&closeImageViewer()}a(handleViewerKeydown,"handle\
ViewerKeydown");function downloadCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex],
t=document.createElement("a");t.href=e.url,t.download=e.filename,document.body.appendChild(t),t.click(),
document.body.removeChild(t)}a(downloadCurrentImage,"downloadCurrentImage");function copyCurrentImageUrl(){
if(!viewerImages.length)return;const e=viewerImages[viewerIndex].url,t=new URL(e,window.location.origin).
href;copyToClipboard(t,()=>showToast("\u753B\u50CFURL\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F",
"success"),()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))}a(copyCurrentImageUrl,
"copyCurrentImageUrl");function reuseCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex];
let t=e.url;try{const n=new URL(t,window.location.origin);n.pathname.startsWith("/files/")&&(t=decodeURIComponent(
n.pathname.replace("/files/","")))}catch{}t&&(currentImageUrls.includes(t)?showToast("\u3053\u306E\u753B\u50CF\u306F\u65E2\u306B\u6DFB\u4ED8\u3055\u308C\u3066\u3044\u307E\
\u3059","info"):(currentImageUrls.push(t),setAttachmentNameForPath(t,e.filename||""),updateFilePreview(),
showToast("\u753B\u50CF\u3092\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F",
"success"),closeImageViewer()))}a(reuseCurrentImage,"reuseCurrentImage");async function copyToClipboard(e,t,n){
try{if(navigator.clipboard&&navigator.clipboard.writeText)await navigator.clipboard.writeText(e),t&&
t();else throw new Error("Clipboard API unavailable")}catch(i){try{const s=document.createElement("t\
extarea");s.value=e,s.style.position="fixed",s.style.left="-9999px",document.body.appendChild(s),s.focus(),
s.select();const o=document.execCommand("copy");document.body.removeChild(s),o?t&&t():n&&n(i)}catch(s){
n&&n(s)}}}a(copyToClipboard,"copyToClipboard");const isQuoteMobileLayout=a(()=>window.matchMedia("(m\
ax-width: 768px)").matches,"isQuoteMobileLayout");let quotePreviewText="";function showQuotePreview(e){
const t=get("quote-bar");quotePreviewText=e,t.classList.contains("preview")||(currentQuote="",t.classList.
add("preview")),get("quote-text-display").innerText=e,t.classList.add("visible"),schedulePromptTokenEstimate()}
a(showQuotePreview,"showQuotePreview");function handleQuotePopover(){const e=window.getSelection(),t=get(
"quote-popover");if(!t)return;const n=isQuoteMobileLayout();if(!e||e.rangeCount===0){t.style.display=
"none",t.classList.remove("show");return}const i=e.toString().trim();if(i.length>0&&get("chat-contai\
ner").contains(e.anchorNode)){if(n){showQuotePreview(i);return}const o=e.getRangeAt(0).getBoundingClientRect(),
r=t.style.display==="none"||!t.style.display||getComputedStyle(t).display==="none";t.style.display="\
block",t.style.top=o.top-40+"px",t.style.left=o.left+"px",r&&(t.classList.remove("show"),t.offsetWidth,
t.classList.add("show"))}else t.style.display="none",t.classList.remove("show")}a(handleQuotePopover,
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
ok-voice-latest",implementedAt:"2026-05-27",implementedRank:5550,name:"Grok Voice Latest",desc:"Alia\
s for the current flagship voice model.",price:"$0.08 / min ($4.80 / hr) audio + $0.004 / text input"},
{id:"grok-voice-think-fast-1.0",implementedAt:"2026-05-11",implementedRank:5140,name:"Grok Voice Thi\
nk Fast 1.0",desc:"Deprecated xAI realtime voice model retained for history compatibility.",price:"$\
0.05 / min ($3.00 / hr)",deprecated:!0},{id:"grok-voice-fast-1.0",implementedAt:"2026-05-01",implementedRank:500,
name:"Grok Voice Fast 1.0",desc:"Legacy xAI realtime voice model retained for history compatibility.",
price:"$0.05 / min ($3.00 / hr)",deprecated:!0},{id:"grok-voice-agent",implementedAt:"2026-04-01",implementedRank:380,
name:"Grok Voice Agent",desc:"xAI realtime voice agent API.",price:"$0.05 / min (Realtime)",deprecated:!0}]},
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
listModelsFlat=a(()=>{const e=[];return MODELS.forEach(t=>{(t.items||[]).forEach(n=>{n&&n.id&&e.push(
n)})}),e},"listModelsFlat"),compareModelsByImplementedAt=a((e,t)=>{const n=String(e&&e.implementedAt||
""),i=String(t&&t.implementedAt||"");if(n!==i)return i.localeCompare(n);const s=Number(e&&e.implementedRank||
0),o=Number(t&&t.implementedRank||0);return s!==o?o-s:String(e&&e.id||"").localeCompare(String(t&&t.
id||""))},"compareModelsByImplementedAt"),getRecentModelsForQuickStart=a((e=WELCOME_QUICK_START_LIMIT)=>listModelsFlat().
filter(t=>t&&t.id&&!t.deprecated&&t.implementedAt).sort(compareModelsByImplementedAt).slice(0,Math.max(
0,Number(e)||0)),"getRecentModelsForQuickStart"),renderWelcomeQuickStart=a(()=>{const e=get("welcome\
-quick-start");if(!e)return;const t=getRecentModelsForQuickStart(WELCOME_QUICK_START_LIMIT);if(!t.length){
e.innerHTML="";return}e.innerHTML=t.map((n,i)=>{const s=(.1+i*.02).toFixed(2),o=n.quickEmoji?`${escapeHtml(
String(n.quickEmoji))} `:"",r=escapeHtml(String(n.name||n.id)),l=String(n.id).replace(/\\/g,"\\\\").
replace(/'/g,"\\'");return`<button type="button" class="welcome-btn p-3 rounded text-sm text-left tr\
ansition btn-hover slide-in-animate" style="animation-delay: ${s}s" onclick="quickStart('${l}')">${o}${r}\
</button>`}).join("")},"renderWelcomeQuickStart"),normalizeModelApiKeyMap=a(e=>{if(!e||typeof e!="ob\
ject")return{};const t={};return Object.entries(e).forEach(([n,i])=>{const s=String(n||"").trim(),o=String(
i||"").trim();!s||!o||(t[s]=o)}),t},"normalizeModelApiKeyMap"),MODEL_NAME_BY_ID=(()=>{const e=new Map;
return MODELS.forEach(t=>{(t.items||[]).forEach(n=>{const i=String(n.id||"").trim();!i||e.has(i)||e.
set(i,String(n.name||i))})}),e})(),getModelNameById=a(e=>{const t=String(e||"").trim();return t?MODEL_NAME_BY_ID.
get(t)||t:""},"getModelNameById"),maskApiKeyPreview=a(e=>{const t=String(e||"");return t?t.length<=8?
"********":`${t.slice(0,4)}...${t.slice(-4)}`:""},"maskApiKeyPreview"),getModelProviderInfo=a(e=>{const t=String(
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
ai_key",inputId:"set-openai",label:"OpenAI API Key"}:null},"getModelProviderInfo"),setModelApiKeyPanelOpen=a(
e=>{const t=get("model-api-keys-panel"),n=get("toggle-model-api-keys-btn");if(!t||!n)return;const i=!!e;
t.classList.toggle("hidden",!i),n.innerText=i?"\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u8A2D\u5B9A\u3092\u9589\u3058\u308B":
"\u30E2\u30C7\u30EB\u5225\u306EAPI\u30AD\u30FC\u3092\u8A2D\u5B9A\u3059\u308B"},"setModelApiKeyPanelO\
pen"),syncModelApiKeyModelOptions=a(()=>{const e=get("model-api-key-model");if(!e)return;const t=e.value||
"";e.innerHTML="";const n=document.createElement("option");n.value="",n.textContent="\u30E2\u30C7\u30EB\u3092\u9078\u629E",
e.appendChild(n),MODELS.forEach(i=>{const s=Array.isArray(i.items)?i.items.filter(r=>!r.deprecated):
[];if(!s.length)return;const o=document.createElement("optgroup");o.label=String(i.category||"Models"),
s.forEach(r=>{const l=String(r.id||"").trim();if(!l)return;const d=document.createElement("option");
d.value=l,d.textContent=`${String(r.name||l)} (${l})`,o.appendChild(d)}),o.children.length>0&&e.appendChild(
o)}),t&&Array.from(e.options).some(s=>s.value===t)&&(e.value=t)},"syncModelApiKeyModelOptions"),renderModelApiKeyList=a(
()=>{const e=get("model-api-key-list");if(!e)return;modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap);
const t=Object.entries(modelApiKeyMap).sort((n,i)=>n[0].localeCompare(i[0]));if(e.innerHTML="",!t.length){
const n=document.createElement("div");n.className="text-[11px] text-gray-500",n.textContent="\u30E2\u30C7\u30EB\u5225\u30AD\u30FC\u306F\
\u672A\u8A2D\u5B9A\u3067\u3059\u3002",e.appendChild(n);return}t.forEach(([n,i])=>{const s=document.createElement(
"div");s.className="flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-9\
00/70 px-3 py-2";const o=document.createElement("div");o.className="min-w-0";const r=document.createElement(
"div");r.className="text-[11px] text-gray-200 truncate",r.textContent=`${getModelNameById(n)} (${n})`;
const l=document.createElement("div");l.className="text-[10px] text-cyan-300 font-mono",l.textContent=
maskApiKeyPreview(i),o.appendChild(r),o.appendChild(l);const d=document.createElement("button");d.type=
"button",d.className="text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-b\
old btn-hover shrink-0",d.textContent="\u524A\u9664",d.onclick=()=>{delete modelApiKeyMap[n],renderModelApiKeyList(),
showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u524A\u9664: ${n}`,"success")},s.appendChild(
o),s.appendChild(d),e.appendChild(s)})},"renderModelApiKeyList"),bindModelApiKeySettingsControls=a(()=>{
const e=get("toggle-model-api-keys-btn");e&&!e.dataset.bound&&(e.dataset.bound="1",e.addEventListener(
"click",()=>{const i=get("model-api-keys-panel");setModelApiKeyPanelOpen(i?i.classList.contains("hid\
den"):!0)}));const t=get("model-api-key-apply-btn");t&&!t.dataset.bound&&(t.dataset.bound="1",t.addEventListener(
"click",()=>{const i=get("model-api-key-model"),s=get("model-api-key-input"),o=i?String(i.value||"").
trim():"",r=s?String(s.value||"").trim():"";if(!o){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!r){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap),modelApiKeyMap[o]=r,s&&(s.
value=""),renderModelApiKeyList(),showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u8A2D\u5B9A: ${o}`,
"success")}));const n=get("model-api-key-input");n&&!n.dataset.bound&&(n.dataset.bound="1",n.addEventListener(
"keydown",i=>{if(i.key==="Enter"){i.preventDefault();const s=get("model-api-key-apply-btn");s&&s.click()}})),
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
n=>({role:n.role,content:n.content.slice(0,1600)})):[]}catch{return[]}}a(loadAiSettingsConversation,
"loadAiSettingsConversation");function persistAiSettingsConversation(){try{sessionStorage.setItem(AI_SETTINGS_CONVERSATION_KEY,
JSON.stringify(aiSettingsConversation.slice(-10)))}catch{}}a(persistAiSettingsConversation,"persistA\
iSettingsConversation");function clearAiSettingsConversation(){aiSettingsConversation=[];try{sessionStorage.
removeItem(AI_SETTINGS_CONVERSATION_KEY)}catch{}}a(clearAiSettingsConversation,"clearAiSettingsConve\
rsation");function appendAiSettingsConversation(e,t){const n=String(t||"").trim();n&&(aiSettingsConversation.
push({role:e,content:n.slice(0,1600)}),aiSettingsConversation=aiSettingsConversation.slice(-10),persistAiSettingsConversation())}
a(appendAiSettingsConversation,"appendAiSettingsConversation"),aiSettingsConversation=loadAiSettingsConversation();
function summarizeAiSettingsConversationValues(e,t){const n=Object.entries(e||{}),i=t==="inspect"?"\u73FE\
\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002":"\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002",
s=n.map(([o,r])=>`${o}: ${formatAiSettingValue(r).slice(0,180)}`).join(`
`);return`${i}${s?`
${s}`:""}`.slice(0,1600)}a(summarizeAiSettingsConversationValues,"summarizeAiSettingsConversationVal\
ues");let gemSuggestionsVisible=!1,gemSelectedIndex=0;const STS_MODELS=new Set(["gpt-transcribe","gp\
t-live-transcribe","gpt-realtime-2","gpt-realtime-translate","gpt-realtime-whisper","gpt-realtime-1.\
5","gpt-realtime","gpt-realtime-mini","gemini-2.5-flash-native-audio-preview-12-2025","gemini-3.1-fl\
ash-live-preview","gemini-3.8-live","gemini-3.8-live-extended-thinking","gemini-3.5-live-translate-p\
review","gemini-3.5-transcribe-live","grok-voice-think-fast-2.0","grok-voice-latest","grok-voice-thi\
nk-fast-1.0","grok-voice-fast-1.0","grok-voice-agent"]),FILE_BASE_URL=CHAT_CONFIG.urls.serveFileBase,
FILE_THUMB_BASE_URL=CHAT_CONFIG.urls.serveFileThumbBase,RICH_PASTE_PDF_SERVER_ROUTE=CHAT_CONFIG.urls.
richPastePdfServer,IMAGE_EXTS=["png","jpg","jpeg","webp","gif","bmp","avif","heic","heif"],AUDIO_EXTS=[
"mp3","wav","aac","ogg","flac","aiff","aif","m4a","opus","oga","weba","webm"],VIDEO_EXTS=["mp4","mov",
"avi","mkv","m4v","webm","mpg","mpeg","wmv","3gp","3gpp","flv"],getFileExt=a(e=>{const t=typeof e=="\
string"?e:e==null?"":String(e);if(!t)return"";const n=t.lastIndexOf(".");return n===-1?"":t.slice(n+
1).toLowerCase()},"getFileExt"),normalizeAttachmentPath=a(e=>{if(!e)return"";let t="";if(typeof e=="\
string"?t=e:typeof e=="object"&&(t=String(e.path||e.url||e.name||e.filename||e.filepath||"")),!t)return"";
try{t.includes("://")&&(t=new URL(t,window.location.origin).pathname||"")}catch{}t.includes("?")&&(t=
t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=t.replace(/^\/+/,""),t.startsWith("files\
/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},"normalizeAttachmentPath"),isGeminiImageModelKey=a(
e=>{const t=(e||"").toLowerCase();return t.includes("gemini")&&(t.includes("image")||t.includes("nan\
o"))},"isGeminiImageModelKey"),isClaudeModelKey=a(e=>(e||"").toLowerCase().includes("claude"),"isCla\
udeModelKey"),getModelApiProvider=a(e=>{const t=String(e||"").toLowerCase().trim();return t?t.includes(
"claude")?"anthropic":t.includes("deepseek")?"deepseek":t.includes("grok")&&!t.includes("gpt")?"xai":
t.includes("google-tts")?"google":t.includes("gemini")||t.startsWith("veo-")||t.startsWith("lyria-")||
t.startsWith("deep-research-")||t.startsWith("antigravity-")?"gemini":"openai":null},"getModelApiPro\
vider"),PROVIDER_LABELS={openai:"OpenAI",gemini:"Gemini",anthropic:"Anthropic (Claude)",xai:"xAI (Gr\
ok)",deepseek:"DeepSeek",google:"Google Cloud"},isPromptCacheEnabled=a(()=>{const e=get("enable-prom\
pt-cache");return!!(e&&e.checked)},"isPromptCacheEnabled"),getPromptCacheLockedProvider=a(()=>{if(!isPromptCacheEnabled())
return null;const e=get("model-select");return getModelApiProvider(e?e.value:"")},"getPromptCacheLoc\
kedProvider"),updatePromptCacheUi=a(()=>{const e=get("prompt-cache-container"),t=get("enable-prompt-\
cache"),n=get("model-selector-btn");if(!t)return;const i=!!t.checked;e&&(e.classList.toggle("ring-1",
i),e.classList.toggle("ring-teal-500/50",i),e.classList.toggle("rounded",i),e.classList.toggle("px-1",
i)),n&&(i?(n.title="PromptCache\u6709\u52B9: \u540C\u4E00API\u30D7\u30ED\u30D0\u30A4\u30C0\u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\u629E\u53EF\u80FD",
n.classList.add("border-teal-500/60")):(n.title="",n.classList.remove("border-teal-500/60")))},"upda\
tePromptCacheUi"),bindPromptCacheControls=a(()=>{const e=get("enable-prompt-cache");!e||e.dataset.bound===
"1"||(e.dataset.bound="1",e.addEventListener("change",()=>{if(updatePromptCacheUi(),e.checked){const t=getModelApiProvider(
get("model-select")?get("model-select").value:""),n=PROVIDER_LABELS[t]||t||"\u73FE\u5728\u306EAPI";showToast(
`PromptCache \u3092\u6709\u52B9\u5316\u3057\u307E\u3057\u305F\u3002\u4EE5\u964D\u306F ${n} \u4EE5\u5916\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002`,"info",!0)}}))},"bindPromptCacheControls"),getModelMediaSupport=a(
e=>{const t=(e||"").toLowerCase();return t.includes("gemini")?t.includes("image")||t.includes("nano")||
t.includes("tts")||t.includes("native-audio")||t.includes("live")?{audio:!1,video:!1}:t.includes("em\
bedding")||t.startsWith("veo-")||t.includes("omni-flash")||t.includes("omni-1.1-flash")||t.startsWith(
"lyria-")?{audio:!1,video:!1}:{audio:!0,video:!0}:{audio:!1,video:!1}},"getModelMediaSupport"),supportsAudioInputModel=a(
()=>getModelMediaSupport(get("model-select").value).audio,"supportsAudioInputModel"),supportsVideoInputModel=a(
()=>getModelMediaSupport(get("model-select").value).video,"supportsVideoInputModel"),isImagePath=a(e=>IMAGE_EXTS.
includes(getFileExt(e||"")),"isImagePath"),isAudioPath=a(e=>AUDIO_EXTS.includes(getFileExt(e||"")),"\
isAudioPath"),isVideoPath=a(e=>VIDEO_EXTS.includes(getFileExt(e||"")),"isVideoPath"),OPENAI_TTS_VOICES=[
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
48e3],isTtsModel=a(()=>get("model-select").value.includes("tts"),"isTtsModel"),isGptImageModel=a(()=>(get(
"model-select").value||"").includes("gpt-image"),"isGptImageModel"),isGeminiImageModel=a(()=>isGeminiImageModelKey(
get("model-select").value),"isGeminiImageModel"),isMistralOcrModel=a(e=>{const t=String(e!=null?e:get(
"model-select")&&get("model-select").value||"").toLowerCase();return t==="mistral-ocr-4-0"||t==="mis\
tral-ocr-latest"||t.startsWith("mistral-ocr")},"isMistralOcrModel"),isLlmModel=a(()=>{const e=(get("\
model-select").value||"").toLowerCase();return isMistralOcrModel(e)||e.includes("tts")||e.includes("\
transcribe")||e.includes("realtime")||e.includes("voice-agent")||e.includes("native-audio")||e.includes(
"live")||e.includes("image")||e.includes("video")||isGeminiVideoModelKey(e)||isGeminiMusicModelKey(e)||
isGeminiEmbeddingModelKey(e)||e.includes("gemini")&&(e.includes("image")||e.includes("nano"))?!1:e.includes(
"gpt")||e.includes("gemini")||e.includes("grok")||e.includes("deepseek")||e.startsWith("deep-researc\
h-")||e.startsWith("antigravity-")},"isLlmModel"),isGrokImageModel=a(()=>{const e=(get("model-select").
value||"").toLowerCase();return e.includes("grok")&&(e.includes("imagine")||e.includes("image"))&&!e.
includes("video")},"isGrokImageModel"),isGrokVideoModel=a(()=>{const e=(get("model-select").value||"").
toLowerCase();return e.includes("grok")&&e.includes("video")},"isGrokVideoModel"),isGeminiVideoModelKey=a(
e=>{const t=(e||"").toLowerCase();return t.startsWith("veo-")||t.includes("omni-flash")||t.includes(
"omni-1.1-flash")},"isGeminiVideoModelKey"),isGeminiVideoModel=a(()=>isGeminiVideoModelKey(get("mode\
l-select").value),"isGeminiVideoModel"),isGeminiMusicModelKey=a(e=>(e||"").toLowerCase().startsWith(
"lyria-"),"isGeminiMusicModelKey"),isGeminiMusicModel=a(()=>isGeminiMusicModelKey(get("model-select").
value),"isGeminiMusicModel"),isGeminiEmbeddingModelKey=a(e=>(e||"").toLowerCase().includes("gemini-e\
mbedding"),"isGeminiEmbeddingModelKey"),isGeminiEmbeddingModel=a(()=>isGeminiEmbeddingModelKey(get("\
model-select").value),"isGeminiEmbeddingModel"),isStsModel=a(()=>STS_MODELS.has(get("model-select").
value),"isStsModel"),isTranscriptionModel=a(()=>{const e=get("model-select")?get("model-select").value:
"";return e==="gpt-transcribe"||e==="gpt-live-transcribe"},"isTranscriptionModel"),isGeminiLiveModel=a(
()=>{const e=get("model-select").value;return e==="gemini-3.1-flash-live-preview"||e==="gemini-3.8-l\
ive"||e==="gemini-3.8-live-extended-thinking"||e==="gemini-3.5-live-translate-preview"||e==="gemini-\
3.5-transcribe-live"},"isGeminiLiveModel"),isGeminiLiveExtendedThinkingModel=a(()=>get("model-select").
value==="gemini-3.8-live-extended-thinking","isGeminiLiveExtendedThinkingModel"),isGeminiLiveTranslateModel=a(
()=>get("model-select").value==="gemini-3.5-live-translate-preview","isGeminiLiveTranslateModel"),isGeminiLiveTranscribeModel=a(
()=>get("model-select").value==="gemini-3.5-transcribe-live","isGeminiLiveTranscribeModel"),isGeminiRealtimeMusicModel=a(
()=>(get("model-select").value||"")==="lyria-realtime-exp","isGeminiRealtimeMusicModel"),isLyriaRealtimeModel=a(
()=>isGeminiRealtimeMusicModel(),"isLyriaRealtimeModel"),isRealtimeSessionModel=a(()=>!(!isStsModel()||
isGeminiLiveModel()||isTranscriptionModel()||get("model-select")&&get("model-select").value==="gpt-r\
ealtime-whisper"),"isRealtimeSessionModel"),getStsProvider=a(e=>{const t=(e||"").toLowerCase();return t.
includes("gpt-realtime")||t==="gpt-transcribe"||t==="gpt-live-transcribe"?"openai":t.includes("grok-\
voice")?"xai":t.includes("gemini")&&(t.includes("native-audio")||t.includes("live"))?"gemini":null},
"getStsProvider");function setStsStatus(e,t=!1){const n=get("sts-status"),i=get("sts-mic-btn");n&&e&&
(n.innerText=e),i&&(t?(i.classList.add("bg-red-600","animate-pulse"),i.classList.remove("bg-cyan-600")):
(i.classList.remove("bg-red-600","animate-pulse"),i.classList.add("bg-cyan-600")))}a(setStsStatus,"s\
etStsStatus");function updateStsUi(){const e=isStsModel(),t=e&&voiceStudioUiEnabled!==!1,n=get("inpu\
t-row"),i=get("sts-panel"),s=get("voice-studio-bar"),o=get("file-preview");e?(n&&n.classList.add("hi\
dden"),o&&o.classList.add("hidden"),t?(i&&(window.VoiceStudioOpen?i.classList.remove("hidden"):i.classList.
add("hidden")),s&&s.classList.remove("hidden")):(i&&i.classList.remove("hidden"),s&&s.classList.add(
"hidden"),window.VoiceStudio&&window.VoiceStudio.closeIfOpen()),setStsStatus("Tap to speak",!1)):(n&&
n.classList.remove("hidden"),i&&i.classList.add("hidden"),s&&s.classList.add("hidden"),window.VoiceStudio&&
window.VoiceStudio.closeIfOpen())}a(updateStsUi,"updateStsUi");function updateStsOptions(){if(!isStsModel())
return;const e=get("model-select").value||"",t=getStsProvider(e),n=get("sts-voice"),i=get("sts-speed\
-wrap"),s=get("sts-speed"),o=get("sts-speed-label"),r=get("sts-rate-wrap"),l=get("sts-rate-in"),d=get(
"sts-rate-out"),p=get("sts-thinking-wrap"),h=get("sts-note"),g=get("sts-voice-wrap"),y=get("sts-auto\
-play-wrap"),b=get("sts-mode-label"),w=isTranscriptionModel()||isGeminiLiveTranscribeModel(),x=get("\
sts-lang-wrap");if(w){b&&(b.textContent="Realtime Speech-to-Text"),g&&g.classList.add("hidden"),y&&y.
classList.add("hidden"),i&&i.classList.add("hidden"),r&&r.classList.add("hidden"),p&&p.classList.add(
"hidden"),x&&x.classList.add("hidden");const S=get("sts-transcribe-wrap"),T=get("sts-custom-vocab-wr\
ap");S&&S.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),T&&T.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),
h&&(h.textContent=isGeminiLiveTranscribeModel()?"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F4E\u9045\u5EF6\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM / \u6700\u592710\u5206\uFF09":
e==="gpt-live-transcribe"?"\u4F4E\u9045\u5EF6\u30E9\u30A4\u30D6\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
"\u9AD8\u7CBE\u5EA6\u306A\u30B3\u30DF\u30C3\u30C8\u5358\u4F4D\u306E\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09")}else if(t===
"openai")b&&(b.textContent="Speech-to-Speech Live"),g&&g.classList.remove("hidden"),y&&y.classList.remove(
"hidden"),setSelectOptions(n,OPENAI_STS_VOICES,n.value||"alloy"),i&&i.classList.remove("hidden"),s&&
(s.min=.25,s.max=1.5,s.step=.05,s.value||(s.value=1),Number(s.value)<.25&&(s.value=.25),Number(s.value)>
1.5&&(s.value=1.5)),r&&r.classList.add("hidden"),p&&p.classList.add("hidden"),x&&x.classList.add("hi\
dden"),h&&(h.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A");else if(t==="xai")b&&(b.textContent=
"Speech-to-Speech Live"),g&&g.classList.remove("hidden"),y&&y.classList.remove("hidden"),setSelectOptions(
n,GROK_STS_VOICES,n.value||"Ara"),i&&i.classList.add("hidden"),r&&r.classList.remove("hidden"),p&&p.
classList.add("hidden"),x&&x.classList.add("hidden"),setSelectOptions(l,GROK_PCM_RATES,Number(l.value||
24e3)),setSelectOptions(d,GROK_PCM_RATES,Number(d.value||24e3)),h&&(h.textContent="xAI\u306FPCM\u30B5\u30F3\u30D7\u30EB\u30EC\u30FC\u30C8\u5909\u66F4\u53EF");else if(t===
"gemini"){if(b&&(b.textContent="Speech-to-Speech Live"),g&&g.classList.remove("hidden"),y&&y.classList.
remove("hidden"),setSelectOptions(n,GEMINI_STS_VOICES,n.value||"Kore"),i&&i.classList.add("hidden"),
r&&r.classList.add("hidden"),p&&p.classList.remove("hidden"),x&&x.classList.add("hidden"),h&&(h.textContent=
"Gemini Live\u306F\u97F3\u58F0\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC"),e==="gemini-3.8-live")p&&
p.classList.add("hidden"),h&&(h.textContent="Gemini 3.8 Flash Live\u306F\u56FA\u5B9A\u30EC\u30A4\u30C6\u30F3\u30B7\u306ELive API\u30E2\u30C7\u30EB\uFF08Thinking leve\
l\u975E\u5BFE\u5FDC\uFF09");else if(e==="gemini-3.8-live-extended-thinking"){h&&(h.textContent="Gemi\
ni 3.8 Live Extended Thinking\u306Flow / medium / high\u306E\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u63A8\u8AD6\u306B\u5BFE\u5FDC");
const S=get("sts-thinking-level");Array.from(S&&S.options||[]).forEach(T=>{T.disabled=T.value==="min\
imal"}),S&&!["low","medium","high"].includes(S.value)&&(S.value="medium")}e==="gemini-3.5-live-trans\
late-preview"&&(b&&(b.textContent="Realtime Translation"),p&&p.classList.add("hidden"),g&&g.classList.
add("hidden"),x&&x.classList.remove("hidden"),h&&(h.textContent="70\u4EE5\u4E0A\u306E\u8A00\u8A9E\u306B\u5BFE\u5FDC\u3059\u308B\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\uFF08Think\u975E\u5BFE\u5FDC\u30FB\u97F3\u58F0\u9078\
\u629E\u4E0D\u53EF\uFF09"))}i&&o&&s&&!i.classList.contains("hidden")&&(o.textContent=`${Number(s.value||
1).toFixed(2)}x`)}a(updateStsOptions,"updateStsOptions");function stsOpt(e){const t=get(e);return e===
"sts-auto-play"||e==="sts-auto-restart"?t?!!t.checked:!0:t?!!t.checked:!1}a(stsOpt,"stsOpt");function getStsSilenceMs(){
const e=get("sts-silence-sec");let t=e?parseFloat(e.value):1.5;return(isNaN(t)||t<.5)&&(t=.5),t>10&&
(t=10),Math.round(t*1e3)}a(getStsSilenceMs,"getStsSilenceMs");function getTtsProvider(e){if(!e)return null;
const t=e.toLowerCase();return t.includes("google-tts")?"google":t.includes("gemini")&&t.includes("t\
ts")?"gemini":t.includes("grok-tts")||t.includes("xai-tts")?"xai":t.includes("tts")?"openai":null}a(
getTtsProvider,"getTtsProvider");function setSelectOptions(e,t,n){e&&(e.innerHTML="",t.forEach(i=>{const s=document.
createElement("option");s.value=i.value||i,s.textContent=i.label||i,(i.value||i)===n&&(s.selected=!0),
e.appendChild(s)}))}a(setSelectOptions,"setSelectOptions");function updateTtsUi(){const e=get("model\
-select").value||"",t=getTtsProvider(e),n=get("audio-gen-options");if(!n)return;if(!t){n.classList.add(
"hidden");return}n.classList.remove("hidden");const i=get("tts-voice"),s=get("tts-voice-custom-wrap"),
o=get("tts-voice-custom"),r=get("tts-language-wrap"),l=get("tts-language"),d=get("tts-speed-wrap"),p=get(
"tts-speed"),h=get("tts-speed-label"),g=get("tts-speed-note");t==="openai"?(setSelectOptions(i,OPENAI_TTS_VOICES,
i.value||"alloy"),s.classList.add("hidden"),r.classList.add("hidden"),p&&(p.min=.25,p.max=4,p.step=.05,
p.value||(p.value=1),Number(p.value)<.25&&(p.value=.25),Number(p.value)>4&&(p.value=4),p.disabled=!1),
g&&(g.textContent="")):t==="gemini"?(setSelectOptions(i,GEMINI_TTS_VOICES,i.value||"Kore"),s.classList.
add("hidden"),r.classList.add("hidden"),p&&(p.disabled=!0),g&&(g.textContent="(Gemini TTS\u306F\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC)")):
t==="google"?(setSelectOptions(i,[{value:"auto",label:"Auto (Studio/Neural2)"},{value:"custom",label:"\
Custom Voice Name"}],i.value||"auto"),i.value==="custom"?s.classList.remove("hidden"):(s.classList.add(
"hidden"),o&&(o.value="")),r.classList.remove("hidden"),l&&!l.value&&(l.value="ja-JP"),p&&(p.min=.25,
p.max=2,p.step=.05,p.value||(p.value=1),Number(p.value)<.25&&(p.value=.25),Number(p.value)>2&&(p.value=
2),p.disabled=!1),g&&(g.textContent="")):t==="xai"&&(setSelectOptions(i,GROK_TTS_VOICES,i.value||"Ev\
e"),s.classList.remove("hidden"),r.classList.remove("hidden"),l&&!l.value&&(l.value="ja"),p&&(p.min=
.7,p.max=1.5,p.step=.05,p.value||(p.value=1),Number(p.value)<.7&&(p.value=.7),Number(p.value)>1.5&&(p.
value=1.5),p.disabled=!1),g&&(g.textContent="xAI TTS supports speed 0.7\u20131.5 and speech tags")),
p&&h&&(h.textContent=`${Number(p.value||1).toFixed(2)}x`)}a(updateTtsUi,"updateTtsUi");let mcpServers=[],
mcpLoaded=!1,mcpLoadPromise=null,mcpOauthPopups=[];const MCP_URLS={servers:a(()=>"/api/mcp/servers",
"servers"),server:a(e=>`/api/mcp/servers/${encodeURIComponent(e)}`,"server"),test:a(e=>`/api/mcp/ser\
vers/${encodeURIComponent(e)}/test`,"test"),authStart:a(e=>`/api/mcp/servers/${encodeURIComponent(e)}\
/auth/start`,"authStart"),authDisconnect:a(e=>`/api/mcp/servers/${encodeURIComponent(e)}/auth/discon\
nect`,"authDisconnect"),tools:a(e=>`/api/mcp/servers/${encodeURIComponent(e)}/tools`,"tools"),oauthClient:a(
()=>"/api/mcp/oauth-client","oauthClient"),permission:a((e,t)=>`/api/mcp/servers/${encodeURIComponent(
e)}/tools/${encodeURIComponent(t)}/permission`,"permission")},mcpGoogleProviderKey="google_workspace",
mcpEsc=a(e=>String(e==null?"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quo\
t;","'":"&#39;"})[t]),"mcpEsc"),mcpStatusMsg=a((e,t,n)=>{const i=get(e);i&&(i.textContent=t||"",i.style.
color=n?"#f87171":"#9ca3af")},"mcpStatusMsg");function mcpAuthStatusLabel(e){return e.auth_type==="n\
one"?"\u8A8D\u8A3C\u4E0D\u8981":e.auth_status==="connected"?"\u63A5\u7D9A\u6E08\u307F":e.auth_status===
"expired"?"\u671F\u9650\u5207\u308C\uFF08\u518D\u8A8D\u8A3C\uFF09":e.auth_status==="needs_auth"?"\u8A8D\u8A3C\u304C\
\u5FC5\u8981":"\u672A\u8A8D\u8A3C"}a(mcpAuthStatusLabel,"mcpAuthStatusLabel");function mcpConnectionStateLabel(e){
return e.connection_state==="error"?"\u30A8\u30E9\u30FC":e.connection_state==="connected"?"\u63A5\u7D9AOK":
e.connection_state==="needs_auth"?"\u8A8D\u8A3C\u5F85\u3061":"\u672A\u63A5\u7D9A"}a(mcpConnectionStateLabel,
"mcpConnectionStateLabel");function mcpBadgeClass(e){return e==="ok"||e==="connected"?"bg-emerald-70\
0/60 text-emerald-100":e==="error"||e==="expired"?"bg-red-700/60 text-red-100":e==="auth"?"bg-amber-\
600/50 text-amber-100":"bg-gray-700 text-gray-300"}a(mcpBadgeClass,"mcpBadgeClass");function mcpStateBadge(e){
const t=mcpAuthStatusLabel(e),n=e.auth_status==="connected"?"ok":e.auth_status==="expired"?"expired":
e.auth_status==="needs_auth"?"auth":"neutral";return`<span class="text-[9px] font-bold px-2 py-0.5 r\
ounded-full ${mcpBadgeClass(n)}">${mcpEsc(t)}</span>`}a(mcpStateBadge,"mcpStateBadge");function mcpOauthProviderLabel(e){
return e==="google_workspace"?"Google Workspace":e||"OAuth"}a(mcpOauthProviderLabel,"mcpOauthProvide\
rLabel");async function loadMcpServers(e){if(!get("mcp-server-list")||mcpLoadPromise&&(await mcpLoadPromise,
!e))return;if(!e&&mcpLoaded){renderMcpServers();return}mcpStatusMsg("mcp-status-msg","\u8AAD\u307F\u8FBC\u307F\u4E2D...",
!1);let n;n=(async()=>{try{const i=await apiFetch(MCP_URLS.servers());if(!i.ok){const o=await i.json().
catch(()=>({}));mcpStatusMsg("mcp-status-msg",o.error||"MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const s=await i.json();mcpServers=s&&Array.isArray(s.servers)?s.servers:[],mcpLoaded=!0,renderMcpServers(),
applyMcpPromptChipUi()}catch(i){mcpStatusMsg("mcp-status-msg","MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(i&&i.message?i.message:i),!0)}finally{mcpLoadPromise===n&&(mcpLoadPromise=null)}})(),mcpLoadPromise=
n,await n}a(loadMcpServers,"loadMcpServers");function mcpHasEnabledServer(){return(mcpServers||[]).some(
e=>!!e.enabled)}a(mcpHasEnabledServer,"mcpHasEnabledServer");function isMcpEnabledForSend(){const e=get(
"mcp-container");if(!e||e.classList.contains("hidden"))return!1;const t=get("enable-mcp");return!!t&&
t.checked}a(isMcpEnabledForSend,"isMcpEnabledForSend");function mcpModelSupported(){try{const e=String(
get("model-select")&&get("model-select").value||"").toLowerCase();return e?!!(e.includes("claude")||
e.startsWith("kimi")||typeof isLlmModel=="function"&&isLlmModel()):!1}catch{return!1}}a(mcpModelSupported,
"mcpModelSupported");function applyMcpPromptChipUi(){const e=get("mcp-container");if(!e)return;const t=mcpModelSupported()&&
mcpHasEnabledServer();if(e.classList.toggle("hidden",!t),syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}}a(applyMcpPromptChipUi,"applyMcpPromptChipUi");
function syncMcpAutoSysRows(){["set","thread"].forEach(e=>{const t=get(`${e}-auto-sys-mcp-enabled`);
t&&(t.disabled=!0,t.checked=isMcpEnabledForSend())})}a(syncMcpAutoSysRows,"syncMcpAutoSysRows");function renderMcpServers(){
const e=get("mcp-server-list"),t=get("mcp-server-count");if(!e)return;const n=mcpServers.length;if(t&&
(t.textContent=`${n}\u4EF6`),!n){e.innerHTML='<div class="text-[11px] text-gray-600 py-2">\u307E\u3060\u30B5\u30FC\u30D0\u30FC\u304C\u3042\u308A\u307E\
\u305B\u3093\u3002\u4E0A\u306E\u30AB\u30B9\u30BF\u30E0\u8FFD\u52A0\u30D5\u30A9\u30FC\u30E0\u304B\u3089\u767B\u9332\u3059\u308B\u304B\u3001Google Workspace \u306E\u8A8D\u8A3C\u3092\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>',
mcpStatusMsg("mcp-status-msg","");return}const i=mcpServers.map((s,o)=>mcpServerCard(s,o)).join("");
e.innerHTML=i,mcpStatusMsg("mcp-status-msg","")}a(renderMcpServers,"renderMcpServers");function mcpServerCard(e,t){
const n=!!e.is_preset,i=e.auth_type==="oauth",s=e.auth_type==="bearer",o=i||s,r=i&&!e.oauth_client_registered,
l=Number(e.tool_count||0),d=l>0?`${l}\u30C4\u30FC\u30EB`:"\u30C4\u30FC\u30EB\u672A\u53D6\u5F97",p=mcpStateBadge(
e),h=n?'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">\u30D7\u30EA\u30BB\u30C3\u30C8<\
/span>':'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">\u30AB\
\u30B9\u30BF\u30E0</span>',g=mcpAuthBlock(e),y=i?mcpOauthClientBlock(e):"";return`
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(e.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${e.enabled?"text-cyan-300":"text-gray-600"}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(e.name)}</span>
                ${h} ${p}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${o?mcpAuthActionButton(e):""}
            ${s&&e.auth_status!=="connected",""}
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
            <span class="${l>0?"text-emerald-300":"text-gray-500"}">${d}</span>
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="too\
ls" data-id="${e.id}">\u30C4\u30FC\u30EB\u4E00\u89A7</button>
        </div>
        <div class="flex items-center gap-1 flex-wrap">
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="tes\
t" data-id="${e.id}"><i class="fas fa-plug"></i> \u63A5\u7D9A\u30C6\u30B9\u30C8</button>
            <span class="text-[9px] text-gray-600">${mcpEsc(mcpConnectionStateLabel(e))}</span>
        </div>
    </div>
    ${l>0?`<div class="hidden mt-2" data-mcp-toolbox="${e.id}"></div>`:`<div class="hidden mt-2" dat\
a-mcp-toolbox="${e.id}"><div class="text-[10px] text-gray-600">\u63A5\u7D9A\u30C6\u30B9\u30C8\u5F8C\u306B\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u8868\u793A\u3055\u308C\u307E\u3059\u3002</div></div>`}\

    ${y}
    ${g}
</div>`}a(mcpServerCard,"mcpServerCard");function mcpAuthActionButton(e){return e.auth_type==="beare\
r"?"":e.auth_status==="connected"||e.auth_status==="expired"?`<button type="button" data-progress-no\
-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="reconnect" data-id="${e.id}"><i class="f\
as fa-sync"></i> \u518D\u8A8D\u8A3C</button>
                        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mc\
p-danger-btn" data-act="disconnect" data-id="${e.id}"><i class="fas fa-unlink"></i> \u63A5\u7D9A\u89E3\u9664</button>`:
`<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="a\
uth" data-id="${e.id}"><i class="fas fa-key"></i> \u8A8D\u8A3C\u3059\u308B</button>`}a(mcpAuthActionButton,
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
</div>`}a(mcpOauthClientBlock,"mcpOauthClientBlock");function mcpAuthBlock(e){if(e.auth_type==="bear\
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
\u4F7F\u3048\u307E\u3059\u3002":""}</div>`}return""}a(mcpAuthBlock,"mcpAuthBlock");async function mcpToggleEnabled(e,t){
mcpStatusMsg("mcp-status-msg",t?"\u6709\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...":"\u7121\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...",
!1);try{const n=await apiFetch(MCP_URLS.server(e),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({enabled:t})});if(!n.ok){const s=await n.json().catch(()=>({}));mcpStatusMsg(
"mcp-status-msg",s.error||"\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}const i=await n.
json();mcpStatusMsg("mcp-status-msg",t?"\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u30C1\u30E3\u30C3\u30C8\u306E\u30E2\u30C7\u30EB\u3078\u30C4\u30FC\u30EB\u304C\u516C\u958B\u3055\u308C\u307E\u3059\u3002":
"\u7121\u52B9\u306B\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(n){mcpStatusMsg("mcp\
-status-msg","\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(n&&n.message?n.message:n),!0)}}
a(mcpToggleEnabled,"mcpToggleEnabled");async function mcpOpenAuth(e){mcpStatusMsg("mcp-status-msg","\
\u8A8D\u53EFURL\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059...",!1);try{const t=await apiFetch(MCP_URLS.
authStart(e),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});if(!t.
ok){const s=await t.json().catch(()=>({}));s.requires_oauth_client?mcpStatusMsg("mcp-status-msg",s.error||
"OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u5148\u306B\u767B\u9332\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
!0):mcpStatusMsg("mcp-status-msg",s.error||"\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const n=await t.json();if(!n.url){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u304C\u8FD4\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
!0);return}const i=window.open(n.url,"_blank","width=520,height=680");if(i){mcpOauthPopups.push(i),mcpStatusMsg(
"mcp-status-msg","Google\u306E\u753B\u9762\u3067\u8A31\u53EF\u3057\u3066\u304F\u3060\u3055\u3044\u3002\u5B8C\u4E86\u5F8C\u3053\u306E\u30BF\u30D6\u306B\u53CD\u6620\u3055\u308C\u307E\u3059\u3002",
!1);const s=window.setInterval(()=>{(!i||i.closed)&&(window.clearInterval(s),loadMcpServers(!0))},1200)}else
mcpStatusMsg("mcp-status-msg","\u30DD\u30C3\u30D7\u30A2\u30C3\u30D7\u304C\u30D6\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F\u3002",
!0)}catch(t){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}a(mcpOpenAuth,"mcpOpenAuth");async function mcpDisconnect(e){if(window.
confirm("\u3053\u306E\u30B5\u30FC\u30D0\u30FC\u306E\u8A8D\u8A3C\u60C5\u5831\uFF08\u30C8\u30FC\u30AF\u30F3\uFF09\u3092\u524A\u9664\u3057\u3066\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const t=await apiFetch(MCP_URLS.authDisconnect(e),{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:"{}"});if(!t.ok){const n=await t.json().catch(()=>({}));mcpStatusMsg("mcp-status-m\
sg",n.error||"\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg(
"mcp-status-msg","\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(t){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}a(mcpDisconnect,"mcpDisconnect");async function mcpDeleteServer(e){if(window.
confirm("\u3053\u306E\u30AB\u30B9\u30BF\u30E0MCP\u30B5\u30FC\u30D0\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const t=await apiFetch(MCP_URLS.server(e),{method:"DELETE"});if(!t.ok){const n=await t.json().catch(
()=>({}));mcpStatusMsg("mcp-status-msg",n.error||"\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","\u524A\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(
!0)}catch(t){mcpStatusMsg("mcp-status-msg","\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(t&&t.message?t.message:t),!0)}}a(mcpDeleteServer,"mcpDeleteServer");async function mcpTestServer(e,t){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",!1);try{const n=await apiFetch(
MCP_URLS.test(e),{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}),i=await n.json().
catch(()=>({}));if(!n.ok){mcpStatusMsg("mcp-status-msg",i.error||"\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}i.probe&&i.probe.message&&mcpStatusMsg("mcp-status-msg",i.probe.message,!i.probe.ok),loadMcpServers(
!0)}catch(n){mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(n&&n.message?n.message:n),!0)}}a(mcpTestServer,"mcpTestServer");async function mcpLoadTools(e){const t=document.
querySelector(`[data-mcp-toolbox="${e}"]`);if(t){t.classList.remove("hidden"),t.innerHTML='<div clas\
s="text-[10px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const n=await apiFetch(MCP_URLS.
tools(e)),i=await n.json().catch(()=>({}));if(!n.ok){t.innerHTML=`<div class="text-[10px] text-red-4\
00">${mcpEsc(i.error||"\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}</div>`;return}const s=i&&
Array.isArray(i.tools)?i.tools:[];if(!s.length){t.innerHTML='<div class="text-[10px] text-gray-600">\
\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u300C\u63A5\u7D9A\u30C6\u30B9\u30C8\u300D\u3067\u53D6\u5F97\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>';
return}const o=s.map((r,l)=>`
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(r.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(r.description||"")}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${r.read_only?"bg-emerald-800/40 text-eme\
rald-200":"bg-amber-800/40 text-amber-200"}">${r.read_only?"\u8AAD\u307F\u53D6\u308A":"\u5909\u66F4"}\
</span>
</div>`).join("");t.innerHTML=`<div class="rounded border border-gray-800 bg-black/20 p-2">${o}</div\
>`}catch{t.innerHTML='<div class="text-[10px] text-red-400">\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}}
a(mcpLoadTools,"mcpLoadTools");async function mcpSaveOauthClient(e,t,n,i){mcpStatusMsg("mcp-status-m\
sg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);const s={provider_key:e,client_id:t,client_secret:n};
try{const o=await apiFetch(MCP_URLS.oauthClient(),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(s)}),r=await o.json().catch(()=>({}));if(!o.ok){mcpStatusMsg("mcp-status\
-msg",r.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg("mcp\
-status-msg","OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch(o){mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(o&&o.message?o.message:o),!0)}}a(mcpSaveOauthClient,"mcpSaveOauthClient");async function mcpAddCustomServer(){
const e=get("mcp-custom-name"),t=get("mcp-custom-url"),n=get("mcp-custom-auth"),i=get("mcp-custom-de\
sc"),s=get("mcp-custom-bearer"),o=get("mcp-custom-status"),r=get("mcp-add-server-btn");if(!e||!t||!n)
return;const l=(e.value||"").trim(),d=(t.value||"").trim(),p=n.value||"none",h=i?(i.value||"").trim():
"",g=(s&&s.value||"").trim();if(!l||!d){o&&(o.textContent="\u8868\u793A\u540D\u3068URL\u306F\u5FC5\u9808\u3067\u3059",
o.style.color="#f87171");return}r&&(r.disabled=!0),o&&(o.textContent="\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",
o.style.color="#9ca3af");const y={name:l,url:d,auth_type:p,description:h};p==="bearer"&&g&&(y.bearer_token=
g);try{const b=await apiFetch(MCP_URLS.servers(),{method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(y)}),w=await b.json().catch(()=>({}));if(!b.ok){o&&(o.textContent=w.error||
"\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",o.style.color="#f87171");return}o&&(o.textContent=
w.probe&&w.probe.message||"\u8FFD\u52A0\u3057\u307E\u3057\u305F",o.style.color=w.probe&&w.probe.ok?"\
#34d399":"#fbbf24"),e.value="",t.value="",i&&(i.value=""),s&&(s.value=""),mcpLoaded=!1,loadMcpServers(
!0)}catch(b){o&&(o.textContent="\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(b&&b.message?
b.message:b),o.style.color="#f87171")}finally{r&&(r.disabled=!1)}}a(mcpAddCustomServer,"mcpAddCustom\
Server");function bindMcpSettingsUi(){const e=get("mcp-server-list");if(!e)return;const t=get("mcp-a\
dd-server-btn");t&&t.addEventListener("click",mcpAddCustomServer);const n=get("mcp-custom-auth"),i=get(
"mcp-custom-bearer-wrap");if(n&&i){const o=a(()=>{i.classList.toggle("hidden",n.value!=="bearer")},"\
syncBearer");n.addEventListener("change",o),o()}const s=get("mcp-save-google-client-btn");s&&s.addEventListener(
"click",async()=>{const o=get("mcp-google-client-id"),r=get("mcp-google-client-secret"),l=get("mcp-g\
oogle-client-state"),d=o?o.value:"",p=r?r.value:"";if(!d&&!p){l&&(l.textContent="Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
l.style.color="#f87171");return}await mcpSaveOauthClient(mcpGoogleProviderKey,d||"********",p||"****\
****",null)}),e.addEventListener("click",async o=>{const r=o.target.closest("[data-act]");if(!r)return;
const l=r.getAttribute("data-act"),d=r.getAttribute("data-id");if(l==="test"){o.preventDefault(),mcpTestServer(
d);return}if(l==="tools"){o.preventDefault(),mcpLoadTools(d);return}if(l==="auth"||l==="reconnect"){
o.preventDefault(),mcpOpenAuth(d);return}if(l==="disconnect"){o.preventDefault(),mcpDisconnect(d);return}
if(l==="delete"){o.preventDefault(),mcpDeleteServer(d);return}if(l==="edit-oauth"){if(o.preventDefault(),
r.closest("[data-mcp-server]")){const h=r.getAttribute("data-oauth-pk")||"",g=mcpServers.find(w=>String(
w.id)===String(d)),y=document.createElement("div");y.className="mt-2 rounded border border-amber-700\
/50 bg-amber-950/20 p-2",y.innerHTML=`
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
uth" data-id="${d}" data-pk="${mcpEsc(g&&(g.oauth_provider_key||g.slug)||"")}">\u4FDD\u5B58</button>
    </div>`;const b=r.closest("div");b.parentNode.insertBefore(y,b.nextSibling),r.remove()}return}if(l===
"save-oauth"){o.preventDefault();const p=r.getAttribute("data-pk")||"",h=r.closest("[data-mcp-server\
]")||document,g=h.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid'),y=h.querySelectorAll(
'[data-oauth-role="secret"], .mcp-oauth-edit-sec'),b=g.length?g[g.length-1].value:"",w=y.length?y[y.
length-1].value:"";if(!b&&!w){mcpStatusMsg("mcp-status-msg","Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpSaveOauthClient(p,b||"********",w||"********",d);return}if(l==="save-bearer"){o.preventDefault();
const p=document.querySelector(`[data-bearer-id="${d}"]`),h=p?p.value:"";if(!h||h.trim()===""){mcpStatusMsg(
"mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);try{const g=await apiFetch(
MCP_URLS.server(d),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({bearer_token:h})}),
y=await g.json().catch(()=>({}));if(!g.ok){mcpStatusMsg("mcp-status-msg",y.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch{mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}return}}),e.addEventListener("change",o=>{const r=o.target.closest(".mcp-enable-toggle");r&&mcpToggleEnabled(
r.getAttribute("data-id"),r.checked)})}a(bindMcpSettingsUi,"bindMcpSettingsUi");const initMcpUi=a(()=>{
try{bindMcpSettingsUi()}catch{}try{loadMcpServers()}catch{}},"initMcpUi");document.readyState==="loa\
ding"?document.addEventListener("DOMContentLoaded",initMcpUi,{once:!0}):initMcpUi();function bindMcpPromptToggle(){
const e=get("enable-mcp");e&&e.addEventListener("change",()=>{if(syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}})}if(a(bindMcpPromptToggle,"bindMcpPromptToggle"),
document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>{try{bindMcpPromptToggle()}catch{}});else
try{bindMcpPromptToggle()}catch{}function getModelTags(e,t){const n=[],i=(e.id||"").toLowerCase(),s=(e.
name||"").toLowerCase(),o=(e.desc||"").toLowerCase(),r=(t.category||"").toLowerCase();return(r.includes(
"gemini")||i.includes("gemini")||s.includes("gemini")||o.includes("gemini")||r.includes("banana")||s.
includes("banana"))&&n.push("gemini"),(r.includes("deepseek")||i.includes("deepseek")||s.includes("d\
eepseek")||o.includes("deepseek"))&&n.push("deepseek"),(r.includes("mistral")||i.includes("mistral")||
s.includes("mistral")||o.includes("mistral")||i.includes("ocr")||r.includes("ocr"))&&n.push("mistral"),
(r.includes("gpt")||r.includes("openai")||i.includes("gpt")||s.includes("gpt")||o.includes("openai"))&&
n.push("openai"),(r.includes("xai")||r.includes("grok")||i.includes("grok")||s.includes("grok")||o.includes(
"xai"))&&n.push("xai"),(r.includes("image")||i.includes("image")||s.includes("image")||o.includes("i\
mage"))&&n.push("image"),(r.includes("audio")||r.includes("music")||r.includes("transcription")||r.includes(
"speech")||i.includes("tts")||i.includes("transcri")||s.includes("tts")||s.includes("transcri")||s.includes(
"voice")||o.includes("tts")||i.includes("realtime")||i.includes("live")||i.includes("voice-agent")||
i.includes("native-audio")||s.includes("audio")||o.includes("audio")||o.includes("speech-to-text"))&&
n.push("audio"),(i.includes("reasoning")||s.includes("reasoning")||o.includes("reasoning"))&&n.push(
"reasoning"),(r.includes("deepseek")||i.includes("deepseek")||s.includes("deepseek"))&&!n.includes("\
reasoning")&&n.push("reasoning"),(i.includes("fast")||s.includes("fast")||o.includes("fast")||r.includes(
"fast"))&&n.push("fast"),(i.includes("deepseek-v4-flash")||r.includes("deepseek")&&s.includes("flash"))&&
!n.includes("fast")&&n.push("fast"),(r.includes("anthropic")||i.includes("claude")||s.includes("clau\
de")||o.includes("anthropic"))&&n.push("anthropic"),(r.includes("kimi")||i.includes("kimi")||s.includes(
"kimi")||o.includes("moonshot"))&&n.push("kimi"),(r.includes("video")||i.includes("video")||i.startsWith(
"veo-")||i.includes("omni-")||s.includes("video")||o.includes("video"))&&n.push("video"),(r.includes(
"music")||i.startsWith("lyria-")||s.includes("music")||o.includes("music")||o.includes("song"))&&n.push(
"music"),(r.includes("transcription")||i.includes("transcri")||s.includes("transcri")||o.includes("t\
ranscription")||o.includes("speech-to-text"))&&n.push("transcription"),(r.includes("ocr")||i.includes(
"ocr")||s.includes("ocr")||o.includes("ocr"))&&n.push("ocr"),(r.includes("agent")||i.includes("agent")||
s.includes("agent")||o.includes("agentic")||o.includes("computer use")||o.includes("deep research"))&&
n.push("agent"),e.agenticView&&n.push("agentic view"),n}a(getModelTags,"getModelTags");function updateModelTagUi(){
const e=get("model-tag-bar");if(!e)return;e.querySelectorAll(".model-tag-btn").forEach(n=>{const i=n.
innerText.trim().toLowerCase(),s=(i==="all"?"all":i)===activeModelTag;n.classList.toggle("is-active",
s)})}a(updateModelTagUi,"updateModelTagUi");function getModelCapabilitySearchTerms(e){const t=String(
e.id||"").toLowerCase(),n=[],i=t.includes("deepseek"),s=t.includes("tts"),o=t.startsWith("mistral-oc\
r"),r=s||o||t.includes("transcribe")||t.includes("realtime")||t.includes("voice-agent")||t.includes(
"native-audio")||t.includes("live")||t.includes("image")||t.includes("video")||t.startsWith("veo-")||
t.includes("omni-flash")||t.startsWith("lyria-")||t.includes("embedding"),l=!r&&(t.includes("gpt")||
t.includes("gemini")||t.includes("grok")||i||t.startsWith("deep-research-")||t.startsWith("antigravi\
ty-")),d=a((...h)=>h.forEach(g=>n.push(g,g.replace(/-/g," "))),"add");if((t.includes("gemini-3.1-fla\
sh-image")||t.includes("gemini-3-pro-image")||t.includes("gemini-2.5-flash-image"))&&d("image genera\
tion","image editing"),t==="gemini-3.1-flash-lite-image"||t==="gemini-3.1-flash-image"?d("thinking",
"\u601D\u8003","minimal","high","thinking level"):t.includes("gemini")&&!r&&(d("thinking","\u601D\u8003",
"thinking level"),t==="gemini-3.8-flash"||t==="gemini-3.7-flash"?d("low","medium","high"):t==="gemin\
i-3.6-flash"?d("medium","high"):t==="gemini-3.5-flash-lite"?d("minimal","medium","high"):t.includes(
"flash")?d("minimal","low","medium","high"):d("low","high")),i&&(d("thinking","\u601D\u8003","reason\
ing","\u63A8\u8AD6","reasoning effort","high"),t!=="deepseek-v4-pro"&&d("low"),t.includes("v4-flash")&&
d("none","max")),l&&(t.includes("gpt-5")||t.includes("o1")||t.includes("o3")||t.includes("grok-4.3")||
t.includes("grok-4.5")||t.includes("grok-4.6")||t.includes("grok-4.20-0309-reasoning")||t.includes("\
grok-build")||t.includes("multi-agent")||t.includes("gpt")&&!s)){d("reasoning","\u63A8\u8AD6","reaso\
ning effort","low","high");const h=t==="gpt-5.6"||t.startsWith("gpt-5.6-"),g=t.includes("grok-4.6"),
y=t.includes("grok-4.3")||t.includes("grok-4.5")||g||t.includes("grok-4.20-0309-reasoning")||t.includes(
"grok-build")||t.includes("multi-agent")||t.includes("gpt-5")||t.includes("o1")||t.includes("o3"),b=t.
includes("grok-4.3")||t.includes("grok-build")||t.includes("gpt-5")||i;y&&d("medium"),b&&d("none"),(h||
i)&&d("max"),(g||t.includes("multi-agent")||h)&&d("xhigh")}return t.includes("claude")&&d("thinking",
"\u601D\u8003","thinking budget","budget"),e.agenticView&&d("agentic view"),[...new Set(n)]}a(getModelCapabilitySearchTerms,
"getModelCapabilitySearchTerms");const modelListGroups=[];let modelListBanner=null,modelListEmpty=null,
modelListBuilt=!1,modelListAnimated=!1,modelListRenderFrame=0;function buildModelList(){const e=get(
"model-list-container");!e||modelListBuilt||(e.innerHTML="",modelListBanner=document.createElement("\
div"),modelListBanner.className="model-banner hidden",e.appendChild(modelListBanner),MODELS.forEach(
t=>{const n=t.items.filter(r=>!r.deprecated);if(!n.length)return;const i=document.createElement("sec\
tion");i.className="model-list-group",i.innerHTML=`
                    <div class="model-group-header">
                        <i class="${t.icon}"></i>
                        <div>
                            <h3 class="model-group-title">${t.category}</h3>
                            <p class="model-group-desc">${t.description}</p>
                        </div>
                    </div>
                    <div class="model-group-grid"></div>
                `;const s=i.querySelector(".model-group-grid"),o=n.map(r=>{const l=document.createElement(
"button"),d=String(r.apiId||r.id||"").trim(),p=r.agenticView?'<span class="inline-flex items-center \
gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold tex\
t-teal-200 whitespace-nowrap" title="Agentic View\u5BFE\u5FDC\uFF1A\u753B\u50CF\u3092\u30AF\u30ED\u30C3\u30D7\u3057\u3066\u518D\u89B3\u5BDF\u3057\u306A\u304C\u3089\u63A8\u8AD6\u3092\u7D99\u7D9A\u3067\u304D\u307E\u3059"><i class="fas fa-eye"\
 aria-hidden="true"></i>Agentic View</span>':"",h=d?`<div class="text-[10px] text-cyan-300/90 mt-1.5\
 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(d)}</\
div>`:"",g=r.price?`<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-\
1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${r.price}</span></div>`:"";
return l.type="button",l.className="model-card",l.dataset.selected="0",l.onclick=()=>selectModel(r.id,
r.name),l.innerHTML=`
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="model-name font-bold text-sm">${r.name}</span>
                                ${p}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle hidden shrink-0 mt-0.5\
"></i>
                        </div>
                        <span class="model-desc text-[10px]">${r.desc}</span>
                        ${h}
                        ${g}
                    `,s.appendChild(l),{model:r,button:l,searchText:`${r.name} ${r.id} ${d} ${r.agenticView?
"agentic view":""} ${t.category} ${getModelTags(r,t).join(" ")} ${getModelCapabilitySearchTerms(r).join(
" ")}`.toLowerCase(),provider:getModelApiProvider(r.id),tags:new Set(getModelTags(r,t))}});modelListGroups.
push({element:i,entries:o}),e.appendChild(i)}),modelListEmpty=document.createElement("div"),modelListEmpty.
className="model-list-empty hidden",e.appendChild(modelListEmpty),modelListBuilt=!0)}a(buildModelList,
"buildModelList");function updateModelButtonSelection(e,t){const n=t===e.model.id;if(e.button.dataset.
selected===(n?"1":"0"))return;e.button.dataset.selected=n?"1":"0",e.button.classList.toggle("is-sele\
cted",n);const i=e.button.querySelector(".model-selected-icon");i&&i.classList.toggle("hidden",!n)}a(
updateModelButtonSelection,"updateModelButtonSelection");function renderModelList(e="",t={}){const n=get(
"model-list-container");if(!n)return;buildModelList();const i=e.toLowerCase(),s=window._visionPickerActive?
null:getPromptCacheLockedProvider(),o=s?PROVIDER_LABELS[s]||s:"",r=get("model-select")?get("model-se\
lect").value:"";let l=0;modelListBanner.classList.toggle("hidden",!s),s&&(modelListBanner.innerHTML=
`<i class="fas fa-database mr-1.5"></i>PromptCache \u6709\u52B9\u4E2D: <strong>${o}</strong> \u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\
\u629E\u3067\u304D\u307E\u3059\uFF08\u4ED6API\u3078\u306E\u5207\u66FF\u306F\u4E0D\u53EF\uFF09`),modelListGroups.
forEach(d=>{let p=0;d.entries.forEach(h=>{const g=h.searchText.includes(i)&&(!s||h.provider===s)&&(activeModelTag===
"all"||h.tags.has(activeModelTag));h.button.classList.toggle("hidden",!g),updateModelButtonSelection(
h,r),g&&(p+=1)}),d.element.classList.toggle("hidden",p===0),l+=p}),modelListEmpty.classList.toggle("\
hidden",l!==0),l===0&&(modelListEmpty.textContent=s?`No ${o} models found.`:"No models found."),t.animate&&
!modelListAnimated&&(modelListAnimated=!0,n.classList.add("model-list-animate"))}a(renderModelList,"\
renderModelList");function scheduleModelListRender(e){modelListRenderFrame&&cancelAnimationFrame(modelListRenderFrame),
modelListRenderFrame=requestAnimationFrame(()=>{modelListRenderFrame=0,renderModelList(e)})}a(scheduleModelListRender,
"scheduleModelListRender");function animateModelCategoryChange(){const e=get("model-list-container");
e&&(e.classList.remove("model-category-enter"),e.offsetWidth,e.classList.add("model-category-enter"))}
a(animateModelCategoryChange,"animateModelCategoryChange");function openModelModal(){location.pathname!==
"/model"&&history.pushState({modal:"model"},"","/model");const e=get("model-search");e&&(e.value=""),
updateModelTagUi(),syncModelSearchClear(),renderModelList("",{animate:!0}),showModal("model-modal"),
e&&window.innerWidth>768&&requestAnimationFrame(()=>e.focus({preventScroll:!0}))}a(openModelModal,"o\
penModelModal"),window.closeModelModal=(e=!1)=>{hideModal("model-modal"),!e&&location.pathname==="/m\
odel"&&history.back()};function selectModel(e,t){if(window._visionPickerActive){currentVisionModel=e,
window._visionPickerActive=!1,window.closeModelModal(),_syncVisionModelDisplay();return}if(isPromptCacheEnabled()){
const s=getModelApiProvider(get("model-select")?get("model-select").value:""),o=getModelApiProvider(
e);if(s&&o&&s!==o){const r=PROVIDER_LABELS[s]||s,l=PROVIDER_LABELS[o]||o;showToast(`PromptCache \u6709\u52B9\u4E2D\u306F\
\u4ED6API\uFF08${l}\uFF09\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\u3067\u304D\u307E\u305B\u3093\u3002\u73FE\u5728: ${r}`,
"warning",!0);return}}const n=get("model-select");n.value=e,get("model-selector-text").innerText=t,window.
closeModelModal();const i=new Event("change");n.dispatchEvent(i)}a(selectModel,"selectModel");function selectModelById(e){
let t=e;for(const n of MODELS){const i=n.items.find(s=>s.id===e);if(i){t=i.name;break}}selectModel(e,
t)}a(selectModelById,"selectModelById");function populateAiSafeFormFields(e){if(e)try{get("set-defau\
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
efault-2fa-method")&&(get("set-default-2fa-method").value=e.default_2fa_method||"totp")}catch{}}a(populateAiSafeFormFields,
"populateAiSafeFormFields");function syncModelSearchClear(){const e=get("model-search"),t=get("model\
-search-clear");t&&t.classList.toggle("hidden",!e||!e.value)}a(syncModelSearchClear,"syncModelSearch\
Clear"),get("model-search")&&get("model-search").addEventListener("input",e=>{scheduleModelListRender(
e.target.value),syncModelSearchClear()}),get("model-search-clear")&&get("model-search-clear").addEventListener(
"click",()=>{const e=get("model-search");e&&(e.value="",syncModelSearchClear(),scheduleModelListRender(
""),e.focus())}),get("model-tag-bar")&&(get("model-tag-bar").addEventListener("click",e=>{const t=e.
target.closest(".model-tag-btn");if(!t)return;const n=t.innerText.trim().toLowerCase(),i=MODEL_TAGS.
includes(n)?n:"all";if(i===activeModelTag)return;activeModelTag=i,updateModelTagUi();const s=get("mo\
del-search");renderModelList(s?s.value:""),animateModelCategoryChange()}),updateModelTagUi()),window.
quickStart=e=>{selectModelById(e),get("welcome-screen").classList.add("hidden")};const BROWSER_FAST_DISABLED_OPTIONS=[
["enable-search","search-container"],["enable-url-context","url-context-container"],["enable-maps","\
maps-grounding-container"],["enable-sys-prompt","sys-prompt-option"],["enable-prompt-cache","prompt-\
cache-container"],["enable-mcp","mcp-container"],["enable-file-creation","file-creation-container"]];
function applyBrowserFastModeRestrictions(){if(!browserFastModeEnabled)return;browserFastPreviousOptions||
(browserFastPreviousOptions={checks:Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([n])=>[n,!!(get(
n)&&get(n).checked)])),coding:!!codingModeEnabled}),BROWSER_FAST_DISABLED_OPTIONS.forEach(([n,i])=>{
const s=get(n),o=get(i);s&&(s.checked=!1,s.disabled=!0),o&&o.classList.add("opacity-50","pointer-eve\
nts-none")}),codingModeEnabled&&syncCodingModeUi(!1,{persist:!1});const e=get("enable-coding-mode"),
t=get("coding-mode-container");e&&(e.disabled=!0),t&&t.classList.add("opacity-50","pointer-events-no\
ne"),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}a(applyBrowserFastModeRestrictions,
"applyBrowserFastModeRestrictions");function restoreBrowserFastModeOptions(){const e=browserFastPreviousOptions;
if(!e)return;BROWSER_FAST_DISABLED_OPTIONS.forEach(([i,s])=>{const o=get(i),r=get(s);o&&(o.disabled=
!1,e&&e.checks&&Object.prototype.hasOwnProperty.call(e.checks,i)&&(o.checked=!!e.checks[i])),r&&r.classList.
remove("opacity-50","pointer-events-none")});const t=get("enable-coding-mode"),n=get("coding-mode-co\
ntainer");t&&(t.disabled=!1),n&&n.classList.remove("opacity-50","pointer-events-none"),e&&e.coding&&
syncCodingModeUi(!0,{persist:!1}),browserFastPreviousOptions=null,typeof updatePromptCacheUi=="funct\
ion"&&updatePromptCacheUi(),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}
a(restoreBrowserFastModeOptions,"restoreBrowserFastModeOptions");function isBatchModelKey(e){const t=String(
e||"").trim().toLowerCase();return t.startsWith("gpt-")?!/(image|audio|tts|transcribe|realtime|search)/.
test(t):t.startsWith("grok-")?!/(image|video|voice|audio|tts|realtime)/.test(t):t.startsWith("gemini\
-")?!/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.
test(t):!1}a(isBatchModelKey,"isBatchModelKey");function updateBatchUi(e){const t=get("batch-mode-co\
ntainer"),n=get("enable-batch-mode");if(!t||!n)return;const i=isBatchModelKey(e);t.classList.toggle(
"hidden",!i),n.disabled=!i||browserFastModeEnabled,i||(n.checked=!1),t.classList.toggle("ring-1",i&&
n.checked),t.classList.toggle("ring-violet-300",i&&n.checked)}a(updateBatchUi,"updateBatchUi");function setBrowserFastModeEnabled(e,t={}){
browserFastModeEnabled=!!e;const n=get("enable-browser-fast-mode");n&&(n.checked=browserFastModeEnabled);
const i=get("browser-fast-mode-container");i&&(i.classList.toggle("ring-1",browserFastModeEnabled),i.
classList.toggle("ring-amber-300",browserFastModeEnabled)),!browserFastModeEnabled&&t.clearKey!==!1&&
(browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null),browserFastModeEnabled?applyBrowserFastModeRestrictions():
t.restoreOptions!==!1&&restoreBrowserFastModeOptions(),updateBatchUi(get("model-select")?get("model-\
select").value:"")}a(setBrowserFastModeEnabled,"setBrowserFastModeEnabled");function openBrowserFastModeModal(e=!0){
const t=get("browser-fast-mode-warning"),n=get("browser-fast-mode-ignore-row");t&&t.classList.toggle(
"hidden",!e),n&&n.classList.toggle("hidden",!e);const i=get("browser-fast-mode-key-description"),s=String(
get("model-select")?get("model-select").value:"Gemini");i&&(i.textContent=`${s} \u306E\u30E2\u30C7\u30EB\u5225\u30AD\u30FC \u2192 \u5171\u901AGemini\u30AD\u30FC\
\u306E\u9806\u306B\u3001\u30B5\u30FC\u30D0\u30FC\u304B\u3089\u81EA\u52D5\u53D6\u5F97\u3057\u307E\u3059\u3002`),
showModal("browser-fast-mode-modal")}a(openBrowserFastModeModal,"openBrowserFastModeModal");function browserFastBootstrapMatches(e,t,n,i){
return!e||e.model!==t||String(e.thread_id||"")!==String(n||"")?!1:String(e.parent_id||"")===String(i||
"")}a(browserFastBootstrapMatches,"browserFastBootstrapMatches");async function fetchBrowserFastBootstrap(e=!1){
const t=String(get("model-select")?get("model-select").value:"").trim(),n=currentThreadId||null,i=n&&
currentParentId||null;if(!e&&browserFastBootstrapMatches(browserFastBootstrap,t,n,i)&&browserFastApiKey)
return browserFastBootstrap;const s=await apiFetch("/api/browser_fast_mode/bootstrap",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({model:t,thread_id:n,parent_id:i})}),
o=await s.json().catch(()=>({}));if(!s.ok||!o.api_key)throw new Error(o.error||"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u306EGemini API\u30AD\
\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");return browserFastApiKey=
String(o.api_key),browserFastApiKeyModel=t,browserFastBootstrap=o,o}a(fetchBrowserFastBootstrap,"fet\
chBrowserFastBootstrap");async function requestBrowserFastModeEnable(){const e=String(get("model-sel\
ect")?get("model-select").value:"").toLowerCase();if(!e.startsWith("gemini-")||/(image|native-audio|tts|live)/.
test(e)){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306FGemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059",
"warning",!0),setBrowserFastModeEnabled(!1);return}if(currentImageUrls.length||uploadProgressState.active>
0||browserFastLocalFiles.size){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u308B\u524D\u306B\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u3092\u30AF\u30EA\u30A2\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0),setBrowserFastModeEnabled(!1);return}const t=(()=>{try{return localStorage.getItem(BROWSER_FAST_IGNORE_WARNING_STORAGE)===
"1"}catch{return!1}})();if(t){try{await fetchBrowserFastBootstrap(!0),setBrowserFastModeEnabled(!0,{
clearKey:!1}),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F",
"warning",!1)}catch(n){setBrowserFastModeEnabled(!1),showToast(n.message||"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u5316\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}return}openBrowserFastModeModal(!t)}a(requestBrowserFastModeEnable,"requestBrowserFastMo\
deEnable"),document.addEventListener("DOMContentLoaded",()=>{get("menu-btn")&&(get("menu-btn").onclick=
()=>{get("sidebar").classList.toggle("open"),get("overlay").classList.toggle("active")}),get("overla\
y")&&(get("overlay").onclick=()=>{get("sidebar").classList.remove("open"),get("overlay").classList.remove(
"active")})}),document.addEventListener("DOMContentLoaded",()=>{var Yn,Qn;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const c=get("enable-batch-mode");c&&c.checked&&(c.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-batch-mode");t&&(t.onchange=()=>{if(t.checked){browserFastModeEnabled&&
setBrowserFastModeEnabled(!1);const c=get("enable-coding-mode");c&&c.checked&&(c.checked=!1,typeof syncCodingModeUi==
"function"&&syncCodingModeUi(!1),showToast("Batch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const n=get("model-\
select");n&&n.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const c=String(
n.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!c.startsWith("gemini-")||/(image|native-audio|tts|live|flash-cyber)/.test(c)?(setBrowserFastModeEnabled(
!1),n.dispatchEvent(new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const i=get("browser-fast-mode-enable-btn");
i&&(i.onclick=async()=>{const c=i.innerHTML;i.disabled=!0,i.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const u=get("browser-fast-mode-ignore-warning");if(u&&u.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(u){showToast(u.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{i.disabled=!1,i.innerHTML=c}});const s=get("browser-fast-mode-cancel-btn");s&&(s.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const o=document.getElementById(
"alpha-bar");setTimeout(()=>{if(o){const c=document.getElementById("version-display");if(c){const u=o.
getBoundingClientRect(),m=c.getBoundingClientRect(),f=m.left+m.width/2-(u.left+u.width/2),v=m.top+m.
height/2-(u.top+u.height/2);o.style.transform=`translate(${f}px, ${v}px) scale(0.1)`,o.style.opacity=
"0",setTimeout(()=>{c.classList.add("pulse-target"),setTimeout(()=>c.classList.remove("pulse-target"),
2e3),o.remove()},800)}else o.style.opacity="0",setTimeout(()=>o.remove(),1e3)}},3e3);function r(){const c=get(
"gpt-image-options");if(!c)return;isGptImageModel()?c.classList.remove("hidden"):c.classList.add("hi\
dden");const u=get("gpt-image-format"),m=get("gpt-image-compression-wrap");u&&m&&(u.value==="png"?m.
classList.add("hidden"):m.classList.remove("hidden"))}a(r,"updateGptImageUi");function l(){const c=get(
"gemini-image-options");if(!c)return;isGeminiImageModel()?c.classList.remove("hidden"):c.classList.add(
"hidden");const m=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(f=>{f&&(Array.from(f.options).
forEach(v=>{v.value!=="1K"&&(v.disabled=m)}),m&&f.value!=="1K"&&(f.value="1K"))})}a(l,"updateGeminiI\
mageUi");function d(){const c=get("grok-image-options");if(!c)return;const u=(get("model-select").value||
"").toLowerCase(),m=isGrokImageModel(),f=u==="grok-imagine-image-quality"||u==="grok-imagine-image-2\
.0",v=u==="grok-imagine-image-2.0";if(m){c.classList.remove("hidden");const _=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;_&&_.classList.toggle("hidden",!f);const C=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;C&&C.classList.toggle("hidden",!v)}else
c.classList.add("hidden");if(get("modal-grok-image-options")){const _=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;_&&_.classList.toggle("hidden",!f);const C=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;C&&C.classList.toggle(
"hidden",!v)}}a(d,"updateGrokImageUi");function p(){var f;const c=get("grok-video-options");if(!c)return;
const u=String(((f=get("model-select"))==null?void 0:f.value)||"").toLowerCase();isGrokVideoModel()?
c.classList.remove("hidden"):c.classList.add("hidden");const m=get("grok-video-resolution");if(m){const v=Array.
from(m.options).find(k=>k.value==="1080p");v&&(v.disabled=u!=="grok-imagine-video-1.5"),u!=="grok-im\
agine-video-1.5"&&m.value==="1080p"&&(m.value="720p")}}a(p,"updateGrokVideoUi");function h(){var v;const c=get(
"gemini-video-options");if(!c)return;const u=String(((v=get("model-select"))==null?void 0:v.value)||
"").toLowerCase();isGeminiVideoModel()?c.classList.remove("hidden"):c.classList.add("hidden");const m=get(
"gemini-video-resolution");if(m){const k=Array.from(m.options).find(C=>C.value==="4K"),_=u==="veo-3.\
1-lite-generate-preview"||u==="veo-3.1-fast-generate-preview"||u==="gemini-omni-flash";k&&(k.disabled=
_),_&&m.value==="4K"&&(m.value="1080p")}const f=get("gemini-video-duration-wrap");f&&f.classList.toggle(
"hidden",u==="gemini-omni-1.1-flash")}a(h,"updateGeminiVideoUi");function g(){const c=get("gemini-mu\
sic-options");if(!c)return;const u=isGeminiRealtimeMusicModel(),m=isGeminiMusicModel()&&!u;c.classList.
toggle("hidden",!m);const f=get("lyria-realtime-studio-bar");f&&f.classList.toggle("hidden",!u)}a(g,
"updateGeminiMusicUi");function y(){var _;const c=get("xai-chat-options");if(!c)return;const u=String(
((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase(),m=u.startsWith("grok-")&&!isGrokImageModel(
u)&&!isGrokVideoModel(u)&&!u.includes("voice");c.classList.toggle("hidden",!m);const f=get("xai-logp\
robs"),v=get("xai-top-logprobs"),k=u.includes("grok-4.20");f&&(f.disabled=k,k&&(f.checked=!1)),v&&(v.
disabled=k,k&&(v.value=""))}a(y,"updateXaiChatUi");function b(){const c=isMistralOcrModel(),u=get("m\
istral-ocr-options");u&&u.classList.toggle("hidden",!c);const m=get("modal-mistral-ocr-options");m&&
m.classList.toggle("hidden",!c),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(f=>{const v=get(f);v&&(v.classList.toggle("opacity-50",c),v.classList.toggle("po\
inter-events-none",c))}),c&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}a(b,"updateMistralOcrUi");function w(){const c=get("image-input-limit\
s");if(!c)return;const u=(get("model-select").value||"").toLowerCase();let m="",f=!1;u.includes("gpt\
-image")?(f=!0,m=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):u==="deepseek-v4.1-flash"||u==="deepseek-v4-flash-vision-exp"?(f=!0,m=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):u.includes("deepseek")||(isGeminiImageModelKey(u)?(f=!0,u.includes("gemini-3.1-flash-lite-\
image")?m=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):u.includes("gemini-3.1-flash-image")?m=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):u.includes("gemini-2.5")&&u.includes("image")?m=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):m=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(u)?(f=!0,m=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):u.includes("grok")?(f=!0,m=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):u.includes("grok")&&u.includes("video")&&(f=!0,m=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),f?(c.innerHTML=m,c.classList.remove("hidden")):(c.classList.add("hidden"),c.innerHTML="")}
a(w,"updateImageInputLimits");function x(){const c=get("model-select");if(!c)return;const u=c.value,
m=String(u||"").toLowerCase(),f=m.includes("deepseek"),v=get("thinking-options"),k=get("reasoning-ef\
fort-container"),_=get("enable-thinking"),C=get("thinking-level"),A=get("thinking-budget"),B=get("en\
able-search"),$=get("search-container"),D=get("url-context-container"),ee=get("enable-maps"),P=get("\
maps-grounding-container"),M=get("enable-sys-prompt"),H=get("sys-prompt-option"),z=get("enable-pytho\
n"),ye=get("python-container"),V=get("prompt-cache-container"),ie=get("enable-prompt-cache"),xe=u===
"gpt-5-search-api",Ie=u.includes("tts"),Ne=isMistralOcrModel(u),Ke=m.includes("gemini-3.1-flash-lite\
-image"),Ze=m.includes("gemini-3.1-flash-image")&&!Ke,At=isClaudeModelKey(u),ht=m==="gemini-3.8-flas\
h-cyber",Et=isLlmModel()&&!f&&!Ie&&!m.includes("realtime")&&!m.includes("native-audio")&&!m.includes(
"live");V&&(Et?(V.classList.remove("hidden","opacity-50","pointer-events-none"),ie&&(ie.disabled=!1)):
(ie&&(ie.checked=!1,ie.disabled=!0),V.classList.add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),
v&&v.classList.add("hidden"),k&&k.classList.add("hidden");const yt=get("vision-model-info");if(yt&&yt.
classList.add("hidden"),k){const ce=get("reasoning-effort");if(ce){Array.from(ce.options).forEach(Ee=>{
const vt=m==="gpt-5.6"||m.startsWith("gpt-5.6-"),Ut=m==="deepseek-v4.1-flash"||m==="deepseek-v4-flas\
h-0731"||m==="deepseek-v4-flash"||m==="deepseek-v4-flash-vision-exp",Fe=m==="deepseek-v4-pro",ot=m.includes(
"grok-4.5"),an=m.includes("grok-4.6");Ee.value==="max"?Ee.classList.toggle("hidden",!vt&&!Ut&&!Fe):Ee.
value==="xhigh"?Ee.classList.toggle("hidden",!an&&!m.includes("multi-agent")&&!vt):Ee.value==="mediu\
m"?Ee.classList.toggle("hidden",!(m.includes("grok-4.3")||ot||an||m.includes("grok-4.20-0309-reasoni\
ng")||m.includes("grok-build")||m.includes("multi-agent")||m.includes("gpt-5")||m.includes("o1")||m.
includes("o3"))):Ee.value==="none"?Ee.classList.toggle("hidden",!m.includes("grok-4.3")&&!m.includes(
"grok-build")&&!m.includes("gpt-5")&&!Ut&&!Fe):Ee.value==="low"&&Ee.classList.toggle("hidden",Fe)});
const ge=ce.selectedOptions&&ce.selectedOptions[0];ge&&ge.classList.contains("hidden")&&(ce.value=f?
"high":"medium")}}D&&D.classList.add("hidden"),P&&P.classList.add("hidden"),_&&(_.disabled=!1),A&&(A.
disabled=!0,A.classList.add("opacity-50"));const de=isGeminiImageModelKey(u);if(Ie||Ne)$&&(get("enab\
le-search").checked=!1,$.classList.add("opacity-50","pointer-events-none")),D&&(get("enable-url-cont\
ext").checked=!1,D.classList.add("opacity-50","pointer-events-none")),P&&ee&&(ee.checked=!1,P.classList.
add("opacity-50","pointer-events-none")),ye&&(z.checked=!1,ye.classList.add("opacity-50","pointer-ev\
ents-none")),M&&H&&(M.checked=!1,M.disabled=!0,H.classList.add("opacity-50"));else if(Ze||Ke)P&&ee&&
(ee.checked=!1,P.classList.add("hidden","opacity-50","pointer-events-none")),v.classList.remove("hid\
den"),Array.from(C.options).forEach(ce=>{["low","medium"].includes(ce.value)&&(ce.disabled=!0),["min\
imal","high"].includes(ce.value)&&(ce.disabled=!1)}),["minimal","high"].includes(C.value)||(C.value=
Ke?"minimal":"high"),_&&(_.disabled=!1),Ke&&(B&&(B.checked=!1,B.disabled=!0),$&&$.classList.add("opa\
city-50","pointer-events-none"));else if(de)P&&ee&&(ee.checked=!1,P.classList.add("hidden","opacity-\
50","pointer-events-none"));else if(At)v.classList.remove("hidden"),A&&(A.disabled=!1,A.classList.remove(
"opacity-50")),Array.from(C.options).forEach(ce=>{ce.disabled=!0}),ye&&(z.checked=!1,ye.classList.add(
"opacity-50","pointer-events-none"));else if(ht){v&&v.classList.remove("hidden"),_&&(_.checked=!0,_.
disabled=!0),Array.from(C.options).forEach(ge=>{ge.disabled=!["low","medium","high"].includes(ge.value)}),
["low","medium","high"].includes(C.value)||(C.value="medium"),[$,D,P,ye].forEach(ge=>{ge&&ge.classList.
add("opacity-50","pointer-events-none")}),[B,ee,z].forEach(ge=>{ge&&(ge.checked=!1,ge.disabled=!0)});
const ce=get("enable-url-context");ce&&(ce.checked=!1,ce.disabled=!0),M&&H&&(M.disabled=!1,H.classList.
remove("opacity-50"))}else if(u.includes("gemini")&&!de){v.classList.remove("hidden"),D&&D.classList.
remove("hidden","opacity-50","pointer-events-none");const ce=u.includes("gemini-3");P&&(ce?P.classList.
remove("hidden","opacity-50","pointer-events-none"):(ee&&(ee.checked=!1),P.classList.add("hidden","o\
pacity-50","pointer-events-none")));const ge=u.includes("flash");Array.from(C.options).forEach(Ee=>{
u==="gemini-3.8-flash"||u==="gemini-3.7-flash"?Ee.disabled=!["low","medium","high"].includes(Ee.value):
u==="gemini-3.6-flash"?Ee.disabled=!["medium","high"].includes(Ee.value):u==="gemini-3.5-flash-lite"?
Ee.disabled=!["minimal","medium","high"].includes(Ee.value):["minimal","medium"].includes(Ee.value)?
Ee.disabled=!ge:Ee.disabled=!1}),(u==="gemini-3.8-flash"||u==="gemini-3.7-flash")&&!["low","medium",
"high"].includes(C.value)||u==="gemini-3.6-flash"&&!["medium","high"].includes(C.value)?C.value="med\
ium":u==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(C.value)?C.value="minimal":!ge&&
["minimal","medium"].includes(C.value)&&(C.value="high"),ce?_&&(_.checked=!0,_.disabled=!0):_&&(_.disabled=
!1),A&&u.includes("gemini-2.5")&&(A.disabled=!1,A.classList.remove("opacity-50")),A&&!u.includes("ge\
mini-2.5")&&(A.disabled=!0,A.classList.add("opacity-50"))}if(isLlmModel()&&(m.includes("gpt-5")||m.includes(
"o1")||m.includes("o3")||m.includes("grok-4.3")||m.includes("grok-4.5")||m.includes("grok-4.6")||m.includes(
"grok-4.20-0309-reasoning")||m.includes("grok-build")||m.includes("multi-agent")||m.includes("gpt")&&
!m.includes("tts")))k.classList.remove("hidden"),$&&$.classList.remove("opacity-50","pointer-events-\
none");else if(f){k.classList.remove("hidden");const ce=get("vision-model-info");if(ce&&ce.classList.
toggle("hidden",m==="deepseek-v4.1-flash"||m==="deepseek-v4-flash-vision-exp"),B&&(B.checked=!1,B.disabled=
!0),$&&$.classList.add("opacity-50","pointer-events-none"),D){const ge=get("enable-url-context");ge&&
(ge.checked=!1),D.classList.add("opacity-50","pointer-events-none")}P&&ee&&(ee.checked=!1,P.classList.
add("opacity-50","pointer-events-none"))}else Ne||($&&$.classList.remove("opacity-50","pointer-event\
s-none"),P&&ee&&(ee.checked=!1,P.classList.add("hidden","opacity-50","pointer-events-none")));if(Ie?
ye&&ye.classList.add("opacity-50","pointer-events-none"):(ye&&ye.classList.remove("opacity-50","poin\
ter-events-none"),(!de||Ze)&&!u.includes("gpt-image")&&(M.disabled=!1,H.classList.remove("opacity-50"))),
(de&&!Ze||u.includes("gpt-image")||isGrokImageModel()||isGrokVideoModel()||Ne)&&M&&H&&(M.checked=!1,
M.disabled=!0,H.classList.add("opacity-50")),ye&&(isLlmModel()?(ye.classList.remove("hidden"),z.disabled=
!1):(z.checked=!1,z.disabled=!0,ye.classList.add("hidden"))),xe?(B&&(B.checked=!0,B.disabled=!0),$&&
$.classList.add("opacity-50","pointer-events-none"),ye&&(z.checked=!1,z.disabled=!0,ye.classList.add(
"opacity-50","pointer-events-none"))):B&&!u.includes("tts")&&!Ne&&!f&&!Ke&&(B.disabled=!1),ht){[B,ee,
z].forEach(ge=>{ge&&(ge.checked=!1,ge.disabled=!0)});const ce=get("enable-url-context");ce&&(ce.checked=
!1,ce.disabled=!0),[$,D,P,ye].forEach(ge=>{ge&&ge.classList.add("opacity-50","pointer-events-none")})}
const Pe=get("mask-btn");Pe&&(isGptImageModel()?Pe.classList.remove("hidden"):(Pe.classList.add("hid\
den"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),r(),
l(),d(),p(),h(),g(),updateBatchUi(u),y(),b(),w(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}a(x,"toggleOptions"),get("model-select")&&(get("model-select").addEventListener(
"change",x),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
x(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const S=get("enable-canvas-mode");S&&(S.checked=canvasModeEnabled,S.addEventListener("change",()=>syncCanvasModeUi(
S.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const T=get("enable-codin\
g-mode");T&&(T.checked=codingModeEnabled,T.addEventListener("change",()=>syncCodingModeUi(T.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",c=>{const u=c.
target.closest("[data-canvas-block-index]");if(!u)return;const m=Number(u.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(m,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",c=>{if(c.target.value===
"")return;const u=Number(c.target.value);Number.isInteger(u)&&applyCanvasSelection(u,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",c=>{const u=c.target.closest(
"[data-canvas-panel-view]");if(!u)return;const m=u.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
m,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const c=getCanvasModeElements(),u=c&&c.code&&c.code.textContent||"";if(!u.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
u,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const E=get(
"prompt-controls-toggle-btn");E&&(E.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>r()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>l()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(c,u,m){const f=(u||"").match(/\S*/)[0];if(f==="py\
exec")return"";if(f==="chat_error")return buildChatErrorBubbleHtml(c||"");const v=c||"",k=(f||"").toLowerCase();
let _="";try{const M=hljs.getLanguage(f)?f:"plaintext";activeStreamingBubbleId&&v.length>2e4?_=escapeHtml(
v):_=hljs.highlight(v,{language:M}).value}catch{_=escapeHtml(v)}const C=encodeURIComponent(v).replace(
/'/g,"%27"),A=detectBlockedScriptsInCode(v),B=hashString(`${f||"TEXT"}
${v||""}`);let $="";if(canvasModeEnabled){const M=String(canvasPreviewState.selectedKey||"")===B,H=M?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";$=`<button\
 class="canvas-preview-btn${M?" canvas-active":""}" data-code="${C}" data-code-key="${B}" data-canva\
s-lang="${escapeHtml(f||"txt")}" title="${H}" aria-label="${H}" aria-pressed="${M?"true":"false"}"><\
i class="fas ${M?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
k,v)){const M=A?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
$=`<button class="html-preview-btn" data-code="${C}" ${A?'data-suspicious="1"':""} title="${M}" aria\
-label="${M}"><i class="fas ${A?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const D=`\
<button class="download-btn" data-code="${C}" data-lang="${f||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ee=k==="diff"?"":`<button class="codin\
g-target-btn" data-code="${C}" data-code-key="${B}" data-coding-lang="${escapeHtml(f||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,P=(f||"TEXT")+(A?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\u305F\
">\u26A0</span>':"");return`<div class="code-wrapper collapsed" data-collapsed="true" data-code-key=\
"${B}"><div class="code-header"><span class="code-lang">${P}</span><div class="code-actions"><button\
 class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"\
></i></button>${ee}${$}${D}<button class="copy-btn" data-code="${C}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i\
 class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language\
-${f}">${_}</code></pre></div></div>`},link(c,u,m){return`<a href="${c}" title="${u||""}" target="_b\
lank">${m}</a>`},image(c,u,m){return buildChatImageHtml(c,{alt:m,title:u})}},breaks:!0,gfm:!0}),threadObserver=
new IntersectionObserver(c=>{c[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread\
-list"),threshold:.1}),threadObserver.observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),
(Yn=get("version-update-dismiss"))==null||Yn.addEventListener("click",()=>{const c=localStorage.getItem(
"app_version")||"";c&&localStorage.setItem("version_notified",c),hideModal("version-update-modal")});
const F=get("version-update-clear-cache");if(F&&(F.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.
clearCacheOnVersionUpdate),F.addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=
saveVersionUpdateCachePreference(F.checked)})),(Qn=get("version-update-reload"))==null||Qn.addEventListener(
"click",async()=>{var u;await versionUpdateCachePreferenceSavePromise.catch(()=>{}),!!((u=get("versi\
on-update-clear-cache"))!=null&&u.checked)?await clearSiteCacheAndReload(get("version-update-reload"),
{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.setVersionChangeHandler(
c=>{c&&c!==appVersion&&(localStorage.getItem("version_notified")||"")!==c&&(localStorage.setItem("ap\
p_version",c),purgeCaches().then(()=>checkAndNotifyVersion(c)))}),window.ConnectionMonitor.start(),window.
addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener("offline",
()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offline")}),window.
addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener("visibil\
itychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener("page\
hide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&botConfig.
lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(c){console.error(c)}try{runBotDetectionGate()}catch(c){console.error(
c)}}else{const c=get("turnstile-container");c&&c.classList.add("hidden")}const J=a(c=>{if(!c)return"\
\u4E0D\u660E";const u=new Date(c);return Number.isNaN(u.getTime())?c:u.toLocaleString()},"formatSess\
ionTime"),X=a(c=>{const u=Array.isArray(c)?c:[],m=get("passkey-list"),f=get("passkey-count");if(f&&(f.
innerText=String(u.length)),!!m){if(!u.length){m.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML="",u.forEach((v,k)=>{const _=v&&v.id?String(v.id):"",C=document.createElement("div");C.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const A=document.
createElement("div");A.className="min-w-0";const B=document.createElement("div");B.className="text-x\
s text-gray-200 truncate",B.innerText=v&&v.name?String(v.name):`Security Key ${k+1}`;const $=document.
createElement("div");$.className="text-[10px] text-gray-500 mt-1",$.innerText=v&&v.created_at?`\u767B\u9332\u65E5\u6642:\
 ${J(v.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",A.appendChild(B),A.appendChild($),C.appendChild(
A);const D=document.createElement("button");D.type="button",D.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",D.innerText="\u524A\u9664",D.
disabled=!_,_&&(D.onclick=()=>window.removeWebAuthnCredential(_)),C.appendChild(D),m.appendChild(C)})}},
"renderPasskeyList"),Te=a(c=>{const u=get("session-list");if(u){if(!c||!c.length){u.innerHTML='<div \
class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}u.innerHTML=c.map(m=>{const f=m.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",v=m.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",k=!m.is_current&&!m.is_revoked?`<button \
data-session-id="${escapeHtml(m.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
_=(m.user_agent||"Unknown").slice(0,120),C=m.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${f}${v}<div class="text-xs text-gray-200">${escapeHtml(
C)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(_)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(J(m.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(J(m.created_at))}</div></div>${k}</div>`}).join(""),u.querySelectorAll(
".session-revoke-btn").forEach(m=>{m.onclick=async()=>{const f=m.getAttribute("data-session-id");if(!f||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const v=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:f})});let k={};try{k=await v.json()}catch{}if(v.ok){if(k.logged_out){
location.href="/login";return}await O()}else showToast(k&&k.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),O=a(async()=>{const c=get("session-list");c&&(c.innerHTML='<div c\
lass="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const u=await apiFetch("/api/\
sessions");let m={};try{m=await u.json()}catch{}if(!u.ok){if(m&&m.error==="session_revoked"){location.
href="/login";return}c&&(c.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const f=(m.sessions||[]).filter(v=>!v.is_revoked);Te(f)},"loadSessions"),q=a(()=>{const c=get(
"session-refresh-btn");c&&(c.onclick=()=>O());const u=get("session-revoke-others-btn");u&&(u.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await O():showToast("\u64CD\u4F5C\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F","error",!0)});const m=get("session-revoke-all-btn");m&&(m.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(c=>{c&&(currentVisionModel=c.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
c);try{loadMcpServers()}catch{}c&&c.theme_color&&applyThemeColor(c.theme_color,!0),c&&Object.prototype.
hasOwnProperty.call(c,"minimal_prompt_mode")&&c.minimal_prompt_mode?setMinimalPromptMode(!0):c&&Object.
prototype.hasOwnProperty.call(c,"compact_prompt_mode")&&setCompactPromptMode(!!c.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(c.enable_client_debug_log===!0,"settings sync");
const u=get("enable-sys-prompt");u&&c&&c.system_prompt&&String(c.system_prompt).trim()&&(!u.disabled&&
!c.default_enable_system_prompt&&!c.use_last_chat_settings&&(u.checked=!0),x())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const Y=get("vision-model-change-btn");Y&&(Y.onclick=()=>_openVisionModelSelector());const pe=get("c\
ompression-format-only");pe&&(pe.onchange=()=>{const c=pe.checked,u=get("compression-max-size"),m=get(
"compression-max-dim");u&&(u.disabled=c),m&&(m.disabled=c);const f=get("compression-size-wrap"),v=get(
"compression-dim-wrap");f&&(f.style.opacity=c?"0.4":"1"),v&&(v.style.opacity=c?"0.4":"1")});const oe=a(
()=>{const c=get("enable-temporary-chat");!c||c.dataset.bound==="1"||(c.dataset.bound="1",c.checked=
!!temporaryChatEnabled,c.onchange=async()=>{const u=temporaryChatEnabled;await applyTemporaryChatSetting(
c.checked)||(setTemporaryChatUiState(u),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");oe(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const ke=get("storage-u\
sage-refresh");ke&&(ke.onclick=()=>loadStorageUsage());let me=null;const _e=a(()=>{const c=new Uint8Array(
16);return window.crypto.getRandomValues(c),Array.from(c,u=>u.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),ne=a((c={})=>{const u=get("account-transfer-progress"),m=get("account-tra\
nsfer-progress-bar"),f=get("account-transfer-progress-percent"),v=get("account-transfer-progress-tex\
t"),k=get("account-transfer-progress-detail"),_=Math.max(0,Math.min(100,Number(c.progress)||0));if(u&&
u.classList.remove("hidden"),m&&(m.style.width=`${_}%`),f&&(f.textContent=`${Math.round(_)}%`),v&&(v.
textContent=c.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),k){
const A={queued:"\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};k.textContent=A[c.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const C=get("account-transfer-cancel-btn");C&&C.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(c.phase))},"renderAccountTransferProgress"),se=a(c=>{we&&(we.disabled=
!!c);const u=get("account-import-btn");u&&(u.disabled=!!c);const m=get("account-transfer-cancel-btn");
m&&(m.disabled=!c)},"setAccountTransferControls"),W=a((c={})=>{const u=get("account-export-ready"),m=get(
"account-export-ready-text"),f=get("account-export-expiry"),v=get("account-export-download-btn"),k=!!(c.
available&&c.download_url);if(u&&u.classList.toggle("hidden",!k),!k){v&&v.removeAttribute("href");return}
const _=Math.max(0,Number(c.size_bytes)||0),C=_>=1024*1024*1024?`${(_/(1024*1024*1024)).toFixed(2)} \
GB`:`${(_/(1024*1024)).toFixed(1)} MB`;if(m){const A=Number(c.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
c.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";m.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${C}${A}`}
if(f){const A=c.expires_at?new Date(c.expires_at):null;f.textContent=A&&!Number.isNaN(A.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${A.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
v&&(v.href=c.download_url)},"renderAccountExportAvailability"),L=a(async c=>{for(;me===c&&!c.stopped;){
try{const u=await apiFetch(`/api/account/transfer/${c.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),m=await u.json().catch(()=>({}));if(u.ok&&(m.state!=="pending"&&ne(m),["ready","completed","\
failed","cancelled","expired"].includes(m.state)))return m}catch{}await new Promise(u=>setTimeout(u,
700))}return null},"pollAccountTransfer"),R=a((c,u,m=!0)=>{u&&(ne(u),W(u),m&&u.state==="ready"?showToast(
u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(u.unreadable_count)>0?"warning":"success",Number(u.unreadable_count)>0):m&&u.state==="failed"&&
showToast(u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),K(c))},"handleFinishedAccountExport"),G=a(async()=>{try{const c=await apiFetch("/api/acc\
ount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),u=await c.json().catch(()=>({}));
if(!c.ok)return;if(W(u),u.state==="ready"){ne(u);return}if(["failed","cancelled","expired"].includes(
u.state)){ne(u);return}if(!["queued","running","cancelling"].includes(u.state)||!u.job_id||me&&me.id===
u.job_id||me)return;const m={id:u.job_id,type:"export",stopped:!1,restored:!0};me=m,se(!0),ne(u);const f=await L(
m);f&&R(m,f,!0)}catch{}},"refreshLatestAccountExport"),K=a(c=>{me===c&&(me=null),c.stopped=!0,se(!1)},
"finishAccountTransfer"),Z=get("account-transfer-cancel-btn");Z&&(Z.onclick=async()=>{const c=me;if(!(!c||
c.stopped)){c.cancelRequested=!0,Z.disabled=!0,ne({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${c.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
c.controller&&c.controller.abort(),ne({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
c.type==="export"&&W({available:!1}),K(c),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const we=get("account-export-btn");we&&(we.onclick=async()=>{if(me)return;const c={id:_e(),
type:"export",stopped:!1};me=c,se(!0),W({available:!1}),ne({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const u=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:c.id}),keepalive:!0})),m=await u.json().catch(()=>({}));if(u.status===409&&
m.error==="export_in_progress"&&m.job_id)c.id=m.job_id;else if(!u.ok)throw new Error(m.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
m.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
ne({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const f=await L(c);!c.cancelRequested&&f&&R(c,f,!0)}catch(u){const m=u&&u.message?u.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";ne({progress:0,phase:"failed",message:m}),
showToast(m,"error",!0),K(c)}});const Q=get("account-export-download-btn");Q&&Q.addEventListener("cl\
ick",async c=>{const u=Q.getAttribute("href");if(!(!u||u==="#")){c.preventDefault();try{const m=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await m.json().catch(
()=>({}));m.ok&&f.available&&f.download_url?(Q.href=f.download_url,window.location.assign(f.download_url)):
(W(f),ne(f),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),G())}catch{window.location.assign(u)}}}),se(!1),G();const re=get("import-files-grid"),
U=get("import-files-info"),he=get("import-files-summary"),tt=a(c=>{const u=Math.max(0,Number(c)||0);
return u>=1024*1024*1024?`${(u/(1024*1024*1024)).toFixed(2)} GB`:u>=1024*1024?`${(u/(1024*1024)).toFixed(
1)} MB`:u>=1024?`${Math.round(u/1024)} KB`:`${u} B`},"importFormatBytes");let Ce=null;const st=a(()=>{
if(!Ce)return;const c=Ce.files,u=Ce.selection;let m=0;c.forEach(k=>{u.has(k.archive_path)&&(m+=Number(
k.size_bytes)||0)});const f=Number(Ce.available_bytes)||0,v=m>f;he&&(he.textContent=`\u9078\u629E\u4E2D: ${tt(
m)} / \u5229\u7528\u53EF\u80FD: ${tt(f)}${v?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,he.classList.
toggle("text-red-300",v)),U&&(U.textContent=`${c.length} files`)},"updateImportFileSelectionUi"),ut=a(
()=>{if(!re||!Ce)return;re.innerHTML="";const c=Ce.files;if(!c.length){re.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
st();return}c.forEach(u=>{const m=document.createElement("label"),f=Ce.selection.has(u.archive_path);
m.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${f?"border-blue-500":"border-gray-600"}`,m.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${f?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(u.display_name)}">${escapeHtml(
u.display_name)}</div><div class="text-[10px] text-gray-500">${tt(u.size_bytes)}</div></div>`;const v=m.
querySelector(".import-file-check");v.addEventListener("change",()=>{v.checked?Ce.selection.add(u.archive_path):
Ce.selection.delete(u.archive_path),m.classList.toggle("border-blue-500",v.checked),m.classList.toggle(
"border-gray-600",!v.checked),st()}),re.appendChild(m)}),st()},"renderImportFileItems"),Je=a(c=>new Promise(
u=>{if(Ce={files:c.files||[],selection:new Set((c.files||[]).map(m=>m.archive_path)),available_bytes:c.
available_bytes,resolve:u},ut(),!get("import-files-modal")){u(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),rt=a(c=>{if(hideModal("import-files-modal"),Ce){const u=Ce.resolve;
Ce=null,u(c)}},"closeImportFileSelection"),wt=get("import-files-close");wt&&(wt.onclick=()=>rt(null));
const gt=get("import-files-cancel");gt&&(gt.onclick=()=>rt(null));const xt=get("import-files-confirm");
xt&&(xt.onclick=()=>{if(!Ce)return;const c=Array.from(Ce.selection);rt(c.length?c.join(","):"__none_\
_")});const lt=get("import-files-select-all");lt&&(lt.onclick=()=>{Ce&&(Ce.files.forEach(c=>Ce.selection.
add(c.archive_path)),ut())});const It=get("import-files-none");It&&(It.onclick=()=>{Ce&&(Ce.selection.
clear(),ut())});const Pt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},at=a(c=>{
if(c===!0)return"ON";if(c===!1)return"OFF";if(c==null||c==="")return"\u672A\u8A2D\u5B9A";const u=String(
c);return u.length>60?u.slice(0,60)+"\u2026":u},"formatAccountSettingValue");let pt=null;const kt=a(
c=>{if(pt){const u=pt;pt=null,hideModal("settings-confirmation-modal"),u(c)}},"resolveSettingsImport\
Confirmation"),_t=a(c=>new Promise(u=>{if(!get("settings-confirmation-modal")){u(!0);return}pt=u;const f=Array.
isArray(c&&c.settings_changes)?c.settings_changes:[],v=get("settings-confirmation-list");v&&(f.length?
v.innerHTML=f.map(_=>{const C=Pt[_.field]||_.field,A=at(_.current),B=at(_.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(C)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
A)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(B)}</d\
iv>
                            </div>`}).join(""):v.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const k=get("settin\
gs-confirmation-count");k&&(k.textContent=`${f.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),mt=get("settings-confir\
mation-modal");mt&&mt.addEventListener("click",c=>{c.target===mt&&kt(!1)});const zt=get("settings-co\
nfirmation-close");zt&&(zt.onclick=()=>kt(!1));const Vt=get("settings-confirmation-cancel");Vt&&(Vt.
onclick=()=>kt(!1));const N=get("settings-confirmation-confirm");N&&(N.onclick=()=>kt(!0));const ae=get(
"account-import-btn"),Se=get("account-import-inplace"),Me=get("account-import-inplace-warning");if(Se&&
Me){const c=a(()=>Me.classList.toggle("hidden",!Se.checked),"syncInplaceWarn");Se.addEventListener("\
change",c),c()}ae&&(ae.onclick=async()=>{const c=get("account-import-file"),u=c&&c.files?c.files[0]:
null,m=get("account-import-categories"),f=m?Array.from(m.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(P=>P.value):[],v=get("account-import-inplace"),k=!!(v&&v.checked),_=get("account-import\
-settings-bypass"),C=!!(_&&_.checked);let A=!1;if(!u){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!f.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const B=m?Array.from(m.querySelectorAll('input[type="checkbox"]:checked')).map(P=>(P.
closest("label")&&P.closest("label").textContent||P.value).trim()):f;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\u5B58\u30C7\
\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${B.join("\u3001")}${k?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const $={id:_e(),type:"import",stopped:!1,controller:new AbortController};
me=$,se(!0),ne({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const D=get("account-import-result");let ee=Promise.resolve(null);try{const M=Math.max(1,Math.ceil(u.
size/10485760)),H=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:u.size}),signal:$.controller.
signal})),z=await H.json().catch(()=>({}));if(!H.ok)throw new Error(z.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
$.uploadId=z.upload_id;const ye=z.chunk_size||10485760;let V=0,ie=0;const xe=a(async()=>{for(;;){const de=ie++;
if(de>=M)return;const ue=u.slice(de*ye,Math.min(u.size,(de+1)*ye)),Pe=new FormData;Pe.append("chunk",
ue,u.name),Pe.append("index",String(de));const ce=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
$.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Pe,signal:$.controller.signal})),
ge=await ce.json().catch(()=>({}));if(!ce.ok)throw new Error(ge.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
V++,ne({progress:Math.min(35,Math.round(V/M*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${V}\
/${M}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let Ie=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),Ie=!0);try{await Promise.
all([xe(),xe(),xe()]);const de=await apiFetch(`/api/account/import/upload/${encodeURIComponent($.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:$.controller.signal})),ue=await de.json().
catch(()=>({}));if(!de.ok)throw new Error(ue.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
ne({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
Ie&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let Ne="",Ke=!1,Ze=0;const At=a(
async()=>{let de=!1;const ue=a(()=>{de||(de=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Pe=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),ce=await Pe.
json().catch(()=>null);if(!Pe.ok||!ce){ue();return}cacheUserSettings(ce);const ge=get("settings-moda\
l");if(ge&&ge.classList.contains("modal-open"))try{Pn(ce)}catch{}ce.theme_color&&applyThemeColor(ce.
theme_color,!0),Object.prototype.hasOwnProperty.call(ce,"minimal_prompt_mode")&&ce.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(ce,"compact_prompt_mode")&&setCompactPromptMode(
!!ce.compact_prompt_mode)}catch{}ue()},"refreshSettingsFormAfterImport"),ht=a(de=>{const ue=de&&de.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";D&&(D.textContent=`\u5B8C\u4E86: ${ue}`,
D.classList.remove("hidden","text-red-300"),D.classList.add("text-emerald-300")),ne({progress:100,phase:"\
completed",message:ue}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),f.includes("chats")&&loadThreads(),f.includes("gems")&&loadGems(),f.includes("files")&&loadStorageUsage(),
(f.includes("settings")||f.includes("api_credentials"))&&At()},"finishImportSuccess"),Et=a(async()=>{
try{const ue=await(await apiFetch(`/api/account/transfer/${$.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return ue&&ue.state?ue:null}catch{return null}},"fetchImportSta\
tus"),yt=a(async()=>{const de=await Et();if(!de)return{status:"unknown"};if(de.state==="completed")return ht(
de),{status:"done"};if(["failed","cancelled","expired"].includes(de.state))throw new Error(de.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(de.state==="needs_sel\
ection"&&Array.isArray(de.files)){const ue=await Je({files:de.files,available_bytes:de.available_bytes});
return ue===null?(ne({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
$.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent($.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(Ne=ue,{status:"reselect"})}if(de.state===
"needs_settings_confirmation"&&Array.isArray(de.settings_changes))return await _t({settings_changes:de.
settings_changes})?(A=!0,{status:"reselect"}):(ne({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\
\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),$.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
$.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(de.state==="running"){const ue=await Promise.race([ee.catch(()=>null),new Promise(Pe=>setTimeout(
()=>Pe(null),6e4))]);if(ue&&ue.state==="completed")return ht(ue),{status:"done"};throw ue&&["failed",
"cancelled","expired"].includes(ue.state)?new Error(ue.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ke;){$.stopped=!0,await ee.catch(()=>null),
$.stopped=!1,ee=L($);let de;try{de=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:$.uploadId,
categories:f.join(","),job_id:$.id,selected_files:Ne,restore_inplace:k,confirm_settings:A||C}),signal:$.
controller.signal}))}catch(Fe){if($.cancelRequested||Fe&&Fe.name==="AbortError")throw Fe;const ot=await yt();
if(ot.status==="done"){Ke=!0;break}if(ot.status==="cancelled")return;if(ot.status==="reselect")continue;
if(Ze<2){Ze++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let ue=null;try{ue=await de.json()}catch{ue=null}if(ue===null){const Fe=await yt();if(Fe.status==="d\
one"){Ke=!0;break}if(Fe.status==="cancelled")return;if(Fe.status==="reselect")continue;if(de.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Ze<2){Ze++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!de.ok&&ue.error==="storage_limit_files"&&ue.files){const Fe=await Je(ue);if(Fe===null){ne({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
$.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent($.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}Ne=Fe;continue}if(ue&&ue.status==="settings_confirmation"&&
Array.isArray(ue.settings_changes)){if(!await _t(ue)){ne({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\
\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),$.uploadId&&
apiFetch(`/api/account/import/upload/${encodeURIComponent($.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}A=!0;continue}if(!de.ok)throw new Error(ue.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F");const Pe=ue.imported||{},ce=[`\u8A2D\u5B9A ${Pe.settings||0}\u4EF6`,
`API\u8A8D\u8A3C ${Pe.api_credentials||0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Pe.chats||0}\u4EF6`,`Ge\
m ${Pe.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Pe.files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Pe.
feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Pe.diagnostics||0}\u4EF6`].join(" / "),ge=ue.duplicates||
{},Ee={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\
\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"},vt=[];for(const Fe of Object.keys(Ee)){const ot=Number(
ge[Fe])||0;ot>0&&vt.push(`${Ee[Fe]} ${ot}\u4EF6`)}const Ut=vt.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${vt.
join("\u3001")}\uFF09`:"";D&&(D.textContent=`\u5B8C\u4E86: ${ce}${Ut}`,D.classList.remove("hidden","\
text-red-300"),D.classList.add("text-emerald-300")),ne({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\
\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),f.includes("chats")&&loadThreads(),f.includes("gems")&&loadGems(),f.includes("files")&&loadStorageUsage(),
(f.includes("settings")||f.includes("api_credentials"))&&At(),Ke=!0}}catch(P){if($.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent($.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),$.cancelRequested||P&&P.name==="AbortError")return;const M=P&&P.message?P.
message:"",H=M==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
M||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";ne({progress:0,phase:"\
failed",message:H}),D&&(D.textContent=H,D.classList.remove("hidden","text-emerald-300"),D.classList.
add("text-red-300")),showToast(H,"error",!0)}finally{$.stopped=!0,await ee.catch(()=>null),K($)}});const De=get(
"account-dedupe-btn"),qe=get("account-dedupe-result"),nt=a((c,u=!1)=>{qe&&(qe.textContent=c,qe.classList.
remove("hidden"),qe.classList.toggle("text-red-300",!!u),qe.classList.toggle("text-emerald-300",!u))},
"showDedupeResult");De&&(De.onclick=async()=>{const c=a(async()=>{const u=await apiFetch("/api/accou\
nt/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})}),
m=await u.json().catch(()=>null);if(!u.ok||!m)throw new Error(m&&m.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!m.has_duplicates){nt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const f=[],v={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const $ of[
"chats","gems","files","feedback","diagnostics"]){const D=Number(m.duplicates&&m.duplicates[$])||0;D>
0&&f.push(`${v[$]} ${D}\u4EF6`)}const k=Number(m.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${m.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${m.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${f.join("\u3001")}${k}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const _=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),C=await _.json().catch(()=>null);if(!_.ok||!C)throw new Error(
C&&C.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const A=[];for(const $ of["chats","gems","files","feedback","diagnostics"]){const D=Number(C.removed&&
C.removed[$])||0;D>0&&A.push(`${v[$]} ${D}\u4EF6`)}const B=Number(C.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${C.kept_referenced_files}\u4EF6\uFF09`:"";nt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${A.join("\u3001")||"0\u4EF6"}${B}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!De.disabled){De.disabled=!0,nt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await c()}catch(u){nt(u&&u.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{De.disabled=!1}}});const He=get("site-cache-usage-refresh");He&&(He.onclick=()=>loadSiteCacheUsage());
const Oe=get("clear-site-cache-btn");Oe&&(Oe.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Oe)});
const Re=get("enc-scan-result"),Ge=a(async(c=null)=>{Re&&(Re.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let u="/api/encryption_scan";c&&(u+=`?thread_id=${encodeURIComponent(c)}`);try{const m=await apiFetch(
u,{cache:"no-store"}),f=await m.json();if(!m.ok){Re&&(Re.textContent=f.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const v=f.total||0,k=f.encrypted||0,_=f.unencrypted||0;let C=`Total: ${v} / Encrypted: ${k} /\
 Plain: ${_}`;if(f.samples&&f.samples.length){const A=f.samples.slice(0,8).map(B=>{const $=B.timestamp?
new Date(B.timestamp).toLocaleString():"";return`#${B.id} (${B.role||""}) ${$}`}).join(" / ");C+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${A}</div>`}Re&&(Re.innerHTML=C)}catch{Re&&(Re.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Wt=get("enc-scan-all");Wt&&(Wt.onclick=()=>Ge(
null));const Ot=get("enc-scan-thread");Ot&&(Ot.onclick=()=>currentThreadId?Ge(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const et=get("admin-enc-\
list");let Nt=null,Le=!1;const Be=a(c=>!c||!c.length?null:c.some(u=>!!u.is_encrypted),"computeThread\
EncryptedFromMessages"),Ye=a(()=>{Nt=Be(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Jt=a(
async(c,u,{confirmPrompt:m=!0,reloadCurrent:f=!0}={})=>{if(!c)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const v=u?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(m&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${v}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Le=!0;try{const k=await apiFetch(`/api/admin/threads/${encodeURIComponent(
c)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:u})}),
_=await k.json().catch(()=>({}));return k.ok?(showToast(`${v}\u3057\u307E\u3057\u305F\uFF08${_.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Nt=!!u,f&&currentThreadId&&String(currentThreadId)===String(
c)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),et&&await ct(),!0):
(showToast(_.error||`${v}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${v}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Le=!1}},"setAdminThreadEncry\
ption"),on=a(c=>{if(!et)return;const u=c.threads||[];if(!u.length){et.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
et.innerHTML=u.map(m=>{const f=m.encrypted_count>0?"enc":"plain",v=f==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",k=f==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",_=m.
updated_at?new Date(m.updated_at).toLocaleString():"",C=escapeHtml(String(m.thread_id)),A=currentThreadId&&
String(currentThreadId)===String(m.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(m.title||
"")}">${escapeHtml(m.title||"(\u7121\u984C)")}${A?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${_} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${m.
message_count} / \u6697\u53F7\u5316: ${m.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${C}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${k} text-white px-2 py-1 roun\
ded shrink-0" data-id="${C}" data-enable="${f==="enc"?"0":"1"}" data-progress-expected-slow="true">${v}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),ct=a(async()=>{if(et){et.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const c=await apiFetch("/api/admin/threads",{cache:"no-store"}),u=await c.json().catch(()=>({}));
if(!c.ok){et.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(u.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(on(u),currentThreadId&&Array.isArray(u.threads)){const m=u.threads.find(f=>String(
f.thread_id)===String(currentThreadId));m&&(Nt=!!m.encrypted)}}catch{et.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>ct()),window.__loadAdminEncThreads=
ct,window.__refreshAdminThreadEncState=Ye,window.__setAdminThreadEncryption=Jt;const le=get("encrypt\
ion-status-admin-toggle");le&&le.addEventListener("click",c=>{c.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),et&&(et.onclick=async c=>{const u=c.target.closest("\
.admin-enc-open");if(u){c.preventDefault();const C=u.getAttribute("data-id");if(!C)return;typeof Rt==
"function"?Rt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(C)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const m=c.target.closest(".admin-enc-toggle");if(!m||Le)return;const f=m.getAttribute("data-i\
d"),v=m.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${v?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;m.disabled=!0;
const _=m.textContent;m.textContent="\u51E6\u7406\u4E2D...";try{await Jt(f,v,{confirmPrompt:!1,reloadCurrent:!0})}finally{
m.disabled=!1,m.textContent=_,await ct()}}),get("file-input").onchange=c=>{const u=Array.from(c.target.
files||[]);c.target.value="",u.length&&handleFiles(u)},get("photo-input")&&(get("photo-input").onchange=
c=>{const u=Array.from(c.target.files||[]);c.target.value="",u.length&&handleFiles(u)});const ve=a(c=>{
const u=get("ban-appeal-list");if(u){if(!c||!c.length){u.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
u.innerHTML=c.map(m=>{const f=m.status||"new",v=m.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
k=m.created_at?new Date(m.created_at).toLocaleString():"",_=m.replied_at?new Date(m.replied_at).toLocaleString():
"",C=m.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${m.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(m.username||
"")}${v}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(k)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(f)}</di\
v>
                            <div class="text-xs text-gray-200 mt-2 whitespace-pre-wrap">${escapeHtml(
m.message||"")}</div>
                            <div class="text-[10px] text-gray-500 mt-2">BAN\u7406\u7531: ${escapeHtml(
m.ban_reason||"N/A")}</div>
                            ${m.evidence?`<details class="mt-2"><summary class="text-[10px] text-cya\
n-300 cursor-pointer">\u4E0D\u5BE9\u306A\u5C65\u6B74\uFF08\u8A18\u9332\uFF09\u3092\u8868\u793A</summary><pre class="mt-1 text-[10px] text-gray-300 whitespace-pr\
e-wrap bg-gray-950/70 border border-gray-700 rounded p-2 max-h-60 overflow-auto">${escapeHtml(m.evidence)}\
</pre></details>`:""}
                            <div class="mt-3">
                                <label class="text-[10px] text-gray-400">\u7BA1\u7406\u8005\u8FD4\u4FE1</label>
                                <textarea class="ban-appeal-reply w-full mt-1 bg-gray-800 border bor\
der-gray-700 rounded px-2 py-1 text-[11px] text-gray-100" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(
C)}</textarea>
                                ${C?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
_)}</div>`:""}
                            </div>
                            <div class="mt-2 flex flex-wrap gap-2">
                                <button class="ban-appeal-mark text-[10px] px-2 py-1 bg-gray-700 hov\
er:bg-gray-600 rounded" data-id="${m.id}">\u65E2\u8AAD</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-blue-700 h\
over:bg-blue-600 rounded" data-id="${m.id}" data-status="in_review">\u5BFE\u5FDC\u4E2D</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-green-700 \
hover:bg-green-600 rounded" data-id="${m.id}" data-status="resolved">\u5B8C\u4E86</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-red-700 ho\
ver:bg-red-600 rounded" data-id="${m.id}" data-status="rejected">\u5374\u4E0B</button>
                                <button class="ban-appeal-reply-send text-[10px] px-2 py-1 bg-sky-70\
0 hover:bg-sky-600 rounded" data-id="${m.id}">\u8FD4\u4FE1\u9001\u4FE1</button>
                                <button class="ban-appeal-block text-[10px] px-2 py-1 bg-rose-700 ho\
ver:bg-rose-600 rounded" data-id="${m.id}">\u7533\u3057\u7ACB\u3066\u30D6\u30ED\u30C3\u30AF</button>
                            </div>
                        </div>
                    `}).join("")}},"renderBanAppeals"),Ue=a(async(c=!1)=>{if(!isAdminUser)return;const u=get(
"ban-appeal-count");if(u)try{const m=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!m.
ok)return;const v=(await m.json()).unread_count||0;u.textContent=String(v),c&&v>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${v}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),ze=a(async()=>{if(!isAdminUser)return;const c=get("ban-appeal-list");if(c){c.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const u=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!u.ok)return;const m=await u.json();ve(m.items||[]),
await Ue(!1)}catch{}}},"loadBanAppeals"),Qe=a(async(c=null)=>{if(!isAdminUser)return;const u=c?{ids:c}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(u)})).ok&&await ze()}catch{}},"markBanAppealsRead"),dt=a(async c=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(c)})).ok&&await ze()}catch{}},"updateBanAppealStatus"),St=a(()=>{
const c=get("tab-general");if(!c||get("temp-chat-settings-card"))return;const u=document.createElement(
"div");u.id="temp-chat-settings-card",u.className="settings-card",u.innerHTML=`
                    <h3 class="settings-card-title">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">\u5207\u65AD\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\uFF08\u79D2\uFF09</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}\
" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rou\
nded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3067\u30DA\u30FC\u30B8\u306E\u8868\u793A/\u63A5\u7D9A\u304C\u9014\u5207\u308C\u305F\u72B6\u614B\u304C\u3053\u306E\u79D2\u6570\u3092\u8D85\u3048\u308B\u3068\u3001\u81EA\u52D5\u524A\
\u9664\u3055\u308C\u307E\u3059\u3002</div>
                    </div>
                `,c.appendChild(u)},"ensureTemporaryChatSettingsCard"),$n=a(()=>{const c=get("set-st\
t-model");if(!c||get("set-llm-transcribe-prompt"))return;const u=c.closest(".space-y-2");if(!u)return;
const m=document.createElement("div");m.className="pt-2 border-t border-gray-700/60",m.innerHTML=`
                    <label class="text-xs text-gray-500 block">LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\uFF08LLM\u65B9\u5F0F\uFF09</label>
                    <textarea id="set-llm-transcribe-prompt" class="w-full h-24 bg-gray-800 border b\
order-gray-600 rounded px-2 py-2 text-xs text-white mt-1" placeholder=""></textarea>
                    <div class="flex items-center gap-2 mt-2">
                        <button type="button" id="reset-llm-transcribe-prompt" class="bg-gray-700 ho\
ver:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">\u65E2\u5B9A\u306B\u623B\u3059</button>
                        <div class="text-[10px] text-gray-500">LLM\u65B9\u5F0F\u306E\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u6642\u306E\u307F\u4F7F\u7528\u3002\u7A7A\u6B04\u3067\u4FDD\u5B58\u3059\u308B\u3068\u65E2\u5B9A\u6587\u9762\u3092\u4F7F\u3044\u307E\u3059\
\uFF08\u7121\u97F3\u6642\u306E\u5B89\u5168\u30AC\u30FC\u30C9\u306F\u5225\u9014\u81EA\u52D5\u4ED8\u4E0E\uFF09\u3002</div>
                    </div>
                `,u.appendChild(m);const f=get("reset-llm-transcribe-prompt");f&&(f.onclick=()=>{const v=get(
"set-llm-transcribe-prompt");v&&(v.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),rn=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
{key:"gemini_local_python",label:"Gemini \u97F3\u58F0/\u52D5\u753B/PDF/DOCX + Python\uFF08\u30ED\u30FC\u30AB\u30EB\u5B9F\u884C\uFF09"},
{key:"grok_search",label:"Search\u88DC\u52A9\uFF08Grok\uFF09"},{key:"openai_search",label:"Search\u88DC\u52A9\uFF08\
OpenAI/xAI Responses\uFF09"},{key:"marker",label:"Marker\u7DE8\u96C6\u6642"},{key:"attachment_names",
label:"\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u540D\uFF08LLM\u5165\u529B\u6642\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{\
attachment_names}} / {{attachment_count}}"},{key:"mathjax",label:"MathJax\uFF08LaTeX\u6570\u5F0F\uFF09"},
{key:"image_analysis",label:"\u753B\u50CF\u89E3\u6790\uFF08Vision Model\u6307\u793A\u6587\uFF09"},{key:"\
mcp",label:"MCP\uFF08\u5916\u90E8\u30C4\u30FC\u30EB\u63A5\u7D9A\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{mcp_tools}}\uFF08\u63A5\
\u7D9A\u4E2D\u306EMCP\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u5165\u308A\u307E\u3059\uFF09",mcpLocked:!0}];
window.buildAutoSystemPromptRows=(c,u=!1)=>{const m=u?"w-full h-14 bg-gray-950 border border-gray-70\
0 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 rounded p-2\
 text-xs text-gray-200";return rn.map(f=>{const v=f.mcpLocked===!0,k=v?'<div class="text-[10px] text\
-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>':
"",_=v?`<input type="checkbox" id="${c}-auto-sys-${f.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${c}-auto-sys-${f.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${f.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${v?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${_}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${c}-auto-sys-${f.key}-text" class="${m}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${f.hint?`<div class="text-[10px] text-gray-500 mt-1">${f.hint}</div>`:""}
                        ${k}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(c,u={})=>{rn.forEach(m=>{const f=u&&
typeof u=="object"?u[m.key]||{}:{},v=get(`${c}-auto-sys-${m.key}-enabled`),k=get(`${c}-auto-sys-${m.
key}-text`);v&&(m.mcpLocked===!0?v.disabled=!0:v.checked=f.enabled!==!1),k&&(k.value=f.text||"",k.placeholder=
f.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const In=a((c,u=null)=>{if(u){const m=get(u);m&&(m.checked=!0)}rn.forEach(m=>{const f=get(`${c}-auto\
-sys-${m.key}-enabled`),v=get(`${c}-auto-sys-${m.key}-text`);if(f&&(m.mcpLocked!==!0?f.checked=!0:f.
disabled=!0),v){const k=v.placeholder||"";v.value=k}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),ei=a(c=>{const u={};return rn.forEach(m=>{const f=get(`${c}\
-auto-sys-${m.key}-enabled`),v=get(`${c}-auto-sys-${m.key}-text`);u[m.key]={enabled:m.mcpLocked===!0?
!0:f?f.checked:!0,text:v?v.value:""}}),u},"collectAutoSystemPromptConfigFromForm");window.ensureAutoSystemPromptSettingsCard=
()=>{const c=get("set-global-sys-prompt-enabled"),u=c?c.closest(".space-y-4"):null;if(!u||get("auto-\
sys-prompt-settings"))return;const m=document.createElement("div");m.id="auto-sys-prompt-settings",m.
className="border-t border-gray-700 pt-3",m.innerHTML=`
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
                `,u.appendChild(m)},window.ensureThreadAutoSystemPromptCard=()=>{const c=get("thread\
-global-sys-prompt"),u=c?c.closest(".space-y-3"):null;if(!u||get("thread-auto-sys-prompt-settings"))
return;const m=document.createElement("div");m.id="thread-auto-sys-prompt-settings",m.className="bor\
der-t border-gray-700 pt-3",m.innerHTML=`
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
                `,u.appendChild(m)},St(),$n(),oe();const ti=a(()=>{const c=get("set-default-model");
if(!c)return;const u=c.value;c.innerHTML="",MODELS.forEach(f=>{const v=document.createElement("optgr\
oup");v.label=f.category,(f.items||[]).forEach(k=>{const _=document.createElement("option");_.value=
k.id,_.textContent=k.name,v.appendChild(_)}),c.appendChild(v)});const m=userSettingsSnapshot&&userSettingsSnapshot.
default_model||u||"gemini-3.6-flash";m&&Array.from(c.options).some(f=>f.value===m)&&(c.value=m)},"po\
pulateDefaultModelOptions"),ni=a(()=>{const c=get("set-default-vision-model");if(!c)return;const u=c.
value;c.innerHTML="",MODELS.forEach(f=>{const v=(f.items||[]).filter(_=>{const C=(_.id||"").toLowerCase();
return C.startsWith("gemini-")||C.startsWith("gpt-4o")||C.startsWith("claude-")||C.startsWith("grok-\
3")});if(v.length===0)return;const k=document.createElement("optgroup");k.label=f.category,v.forEach(
_=>{const C=document.createElement("option");C.value=_.id,C.textContent=_.name+" \u2605",k.appendChild(
C)}),c.appendChild(k)});const m=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||u||
"gemini-3-flash-preview";m&&Array.from(c.options).some(f=>f.value===m)&&(c.value=m)},"populateDefaul\
tVisionModelOptions"),Pn=a(c=>{if(!c)return;cacheUserSettings(c);const u=get("app-global-sys-prompt-\
preview");u&&(u.value=c.global_system_prompt_effective||"");const m=get("app-global-sys-prompt-previ\
ew-status");m&&(c.global_system_prompt_enabled===!1?m.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
c.global_system_prompt_uses_time_fallback?m.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
m.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("sys-prompt-text")&&(get("sys-prompt-text").value=c.system_prompt||""),get("set-global-sys-promp\
t-enabled")&&(get("set-global-sys-prompt-enabled").checked=c.system_prompt_enabled!==!1),window.ensureAutoSystemPromptSettingsCard(),
get("set-apply-global-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=c.apply_global_system_prompt!==
!1),get("set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=c.apply_auto_system_prompt_notices!==
!1),window.applyAutoSystemPromptConfigToForm("set",c.auto_system_prompt_notices_config||{}),get("set\
-latency-metrics")&&(get("set-latency-metrics").checked=c.enable_latency_metrics===!0),get("set-clie\
nt-debug-log")&&syncClientDebugLogToggle(c.enable_client_debug_log===!0,"settings modal sync"),get("\
set-openai")&&(get("set-openai").value=c.openai_key||""),get("set-gemini")&&(get("set-gemini").value=
c.gemini_key||""),get("set-deepseek")&&(get("set-deepseek").value=c.deepseek_key||""),get("set-kimi")&&
(get("set-kimi").value=c.kimi_key||""),get("set-mistral")&&(get("set-mistral").value=c.mistral_key||
""),get("set-anthropic")&&(get("set-anthropic").value=c.anthropic_key||""),get("set-gemini-backend")&&
(get("set-gemini-backend").value=normalizeGeminiBackend(c.gemini_backend||"gemini_api")),get("set-ge\
mini-vertex-project")&&(get("set-gemini-vertex-project").value=c.gemini_vertex_project||""),get("set\
-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=c.gemini_vertex_location||"globa\
l"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-gemini\
-vertex-credentials-json").value=c.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),get("se\
t-admin-api-key-mode")&&(get("set-admin-api-key-mode").value=normalizeAdminApiKeyMode(c.admin_api_key_mode||
"env_fallback")),syncAdminApiKeyModeUi(),get("set-xai")&&(get("set-xai").value=c.xai_key||""),get("s\
et-google-key")&&(get("set-google-key").value=c.google_key||""),get("set-google-project")&&(get("set\
-google-project").value=c.google_project||""),modelApiKeyMap=normalizeModelApiKeyMap(c.model_api_keys||
{}),syncModelApiKeyModelOptions(),renderModelApiKeyList(),setModelApiKeyPanelOpen(!1),get("set-mic-t\
ranscribe-mode")&&(get("set-mic-transcribe-mode").value=c.mic_transcribe_mode||"stt_api"),get("set-s\
tt-model")&&(get("set-stt-model").value=c.stt_model||"gpt-4o-mini-transcribe"),get("set-llm-transcri\
be-prompt")&&(get("set-llm-transcribe-prompt").value=c.llm_transcribe_prompt||"",get("set-llm-transc\
ribe-prompt").placeholder=c.llm_transcribe_prompt_default||""),syncRichPastePromptPreferencesUi(c),updateGoogleLinkUI(
c),updateMinashinLinkUI(c),get("set-enter-to-send")&&(get("set-enter-to-send").checked=!!c.enter_to_send),
writePromptBarModeToForm(!!c.compact_prompt_mode,!!c.minimal_prompt_mode),get("set-use-sw-cache")&&(get(
"set-use-sw-cache").checked=!!c.use_sw_cache),get("set-clear-cache-on-version-update")&&(get("set-cl\
ear-cache-on-version-update").checked=!!c.clear_cache_on_version_update),get("set-liquid-glass")&&(get(
"set-liquid-glass").checked=!!c.liquid_glass_enabled),get("set-light-mode")&&(get("set-light-mode").
checked=!!c.light_mode_enabled),get("set-auto-search-links")&&(get("set-auto-search-links").checked=
c.auto_search_on_links!==!1),get("set-use-last-settings")&&(get("set-use-last-settings").checked=!!c.
use_last_chat_settings),get("set-default-model")&&(get("set-default-model").value=c.default_model||"\
gemini-3.6-flash"),get("set-default-vision-model")&&(get("set-default-vision-model").value=c.default_vision_model||
"gemini-3-flash-preview"),applyTemporaryChatTimeoutSeconds(c.temp_chat_timeout_seconds),get("set-def\
ault-search")&&(get("set-default-search").checked=!!c.default_enable_search),get("set-default-url-co\
ntext")&&(get("set-default-url-context").checked=!!c.default_enable_url_context),get("set-default-ma\
ps")&&(get("set-default-maps").checked=!!c.default_enable_maps),get("set-default-python")&&(get("set\
-default-python").checked=!!c.default_enable_python),get("set-default-file-creation")&&(get("set-def\
ault-file-creation").checked=!!c.default_enable_file_creation),get("set-default-thinking")&&(get("se\
t-default-thinking").checked=!!c.default_enable_thinking),get("set-default-sys-prompt")&&(get("set-d\
efault-sys-prompt").checked=!!c.default_enable_system_prompt),get("set-default-mcp")&&(get("set-defa\
ult-mcp").checked=c.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-default-th\
inking-level").value=c.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get("set\
-default-thinking-budget").value=c.default_thinking_budget||4096),get("set-default-reasoning-effort")&&
(get("set-default-reasoning-effort").value=c.default_reasoning_effort||"medium"),get("set-default-sa\
fety")&&(get("set-default-safety").value=c.default_safety_setting||"default"),get("set-e2ee").checked=
c.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=c.bot_detection_enabled!==!1),get(
"set-bot-detect-global")&&(get("set-bot-detect-global").checked=c.bot_detection_global_enabled!==!1);
const f=get("bot-status");f&&(c.is_bot_banned?(f.textContent=`BAN\u4E2D: ${c.bot_ban_reason||"Bot de\
tection"}`,f.classList.remove("hidden"),f.classList.add("text-red-400")):f.classList.add("hidden")),
c&&c.theme_color?(applyThemeColor(c.theme_color,!0),syncThemeInputs(c.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=c.username);const v=get("2fa-badge"),k=get("disable-2fa-btn");c.is_2fa_enabled?
(v.innerText="ENABLED",v.classList.replace("bg-gray-700","bg-green-600"),v.classList.replace("text-g\
ray-400","text-white"),k.classList.remove("hidden")):(v.innerText="DISABLED",v.classList.replace("bg\
-green-600","bg-gray-700"),v.classList.replace("text-white","text-gray-400"),k.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!c.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=c.default_2fa_method||"totp");const _=get(
"set-passkey-only-login"),C=get("passkey-only-note"),A=Array.isArray(c.passkey_credentials)?c.passkey_credentials:
[];if(X(A),_){_.checked=!!c.passkey_only_login;const P=A.length>0||!!c.has_webauthn;_.disabled=!P,P||
(_.checked=!1),C&&(P?C.classList.add("hidden"):C.classList.remove("hidden"))}const B=get("mig-status\
-box"),$=get("mig-progress-text"),D=get("mig-progress-bar");if((c.migration_status||"idle")==="proce\
ssing"){B.classList.remove("hidden");const P=(c.migration_progress||"").split("/");if(P.length===2){
const M=parseInt(P[0]||"0",10),H=parseInt(P[1]||"0",10);$&&($.innerText=`${M} / ${H}`),D&&H>0&&(D.style.
width=`${Math.min(100,Math.floor(M/H*100))}%`)}}else B.classList.add("hidden"),D&&(D.style.width="0%"),
$&&($.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettingsFormFromData");
window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),snapshotSidebarHistory(
"settings-open-before");const c=await ensureUserSettingsSnapshot();c&&Pn(c);const u=get("search-box"),
m=u?u.value:"";clearTimeout(searchTimeout);const f=get("settings-search");if(f&&(f.value=""),filterSettings(),
ti(),ni(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(()=>refreshSettingsTabsScroll()),
restoreThreadSearchValue(m,"restored-search-box-open"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-open-after"),[50,200,400,800].forEach(v=>{setTimeout(()=>{restoreThreadSearchValue(m,"rest\
ored-search-box-"+v+"ms"),snapshotSidebarHistory("settings-open-later-"+v+"ms")},v)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),G(),$n(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),Ue(!0),ze(),c||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),xn(),q(),O();try{loadMcpServers()}catch{}};const Rt=a((c=!1)=>{snapshotSidebarHistory("\
settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!c&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),ii=a(()=>{const c=get("set-th\
eme-color"),u=get("set-theme-color-text"),m=get("theme-reset-btn"),f=document.querySelectorAll("#the\
me-presets .theme-swatch"),v=a((k,_=!0)=>{const C=normalizeHex(k);C&&(applyThemeColor(C,_),syncThemeInputs(
C))},"applyFromValue");c&&c.addEventListener("input",()=>v(c.value,!0)),u&&(u.addEventListener("chan\
ge",()=>{const k=normalizeHex(u.value);if(!k){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}v(k,!0)}),u.addEventListener("keydown",k=>{k.key==="Enter"&&(k.preventDefault(),
u.blur())})),m&&(m.onclick=()=>v(THEME_DEFAULT,!0)),f.forEach(k=>{k.addEventListener("click",()=>v(k.
getAttribute("data-color"),!0))})},"bindThemeControls"),si=a(()=>{const c=get("reset-global-sys-prom\
pt");c&&(c.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const u=get(
"reset-set-auto-sys-prompt-defaults");u&&(u.onclick=()=>{In("set","set-apply-auto-sys-prompt-notices"),
showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const m=get("reset-thread-auto-sys-prompt-defaults");m&&(m.onclick=()=>{In("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Rt();const On=get("settings-header-close");On&&(On.onclick=()=>Rt());
const Bt=get("settings-search");Bt&&(Bt.addEventListener("input",filterSettings),Bt.addEventListener(
"keydown",c=>{if(c.key==="Enter"){const u=get("tab-"+activeSettingsTab);if(!u)return;const m=u.querySelector(
":scope > .settings-match");m&&m.scrollIntoView({behavior:"smooth",block:"start"})}}));const Nn=get(
"settings-search-clear");Nn&&Nn.addEventListener("click",()=>{Bt&&(Bt.value="",filterSettings(),Bt.focus())}),
ii(),si(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const gn=get("set-gemini-local-python-dialog");gn&&(gn.onchange=()=>setGeminiLocalPyDialogEnabled(gn.
checked));const Rn=get("set-gemini-backend");Rn&&(Rn.onchange=()=>syncGeminiBackendUi());const Bn=get(
"set-admin-api-key-mode");Bn&&(Bn.onchange=()=>syncAdminApiKeyModeUi());const bn=get("set-temp-chat-\
timeout-seconds");bn&&(bn.onchange=()=>{applyTemporaryChatTimeoutSeconds(bn.value)});const Fn=get("s\
lash-command-cancel-btn");Fn&&(Fn.onclick=()=>{hidePendingSlashCommandIndicator();const c=get("promp\
t-input");c&&c.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const c=get("set-username"),u=get("set-password"),m=readPromptBarModeFromForm(),f={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:ei(
"set"),theme_color:normalizeHex(get("set-theme-color-text")?get("set-theme-color-text").value:"")||THEME_DEFAULT,
light_mode_enabled:get("set-light-mode")?get("set-light-mode").checked:!1,mic_transcribe_mode:get("s\
et-mic-transcribe-mode")?get("set-mic-transcribe-mode").value:"stt_api",stt_model:get("set-stt-model")?
get("set-stt-model").value:null,llm_transcribe_prompt:get("set-llm-transcribe-prompt")?get("set-llm-\
transcribe-prompt").value:"",enter_to_send:get("set-enter-to-send")?get("set-enter-to-send").checked:
!1,compact_prompt_mode:m.compact_prompt_mode,minimal_prompt_mode:m.minimal_prompt_mode,use_sw_cache:get(
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
-method").value:"totp",new_username:c?c.value:null,new_password:u?u.value:null},v=get("set-e2ee")?get(
"set-e2ee").checked:!1,k=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
v!==k&&(f.enable_e2ee=v),get("set-openai")&&(f.openai_key=get("set-openai").value),get("set-gemini")&&
(f.gemini_key=get("set-gemini").value),get("set-deepseek")&&(f.deepseek_key=get("set-deepseek").value),
get("set-kimi")&&(f.kimi_key=get("set-kimi").value),get("set-mistral")&&(f.mistral_key=get("set-mist\
ral").value),get("set-anthropic")&&(f.anthropic_key=get("set-anthropic").value),f.model_api_keys=normalizeModelApiKeyMap(
modelApiKeyMap),get("set-gemini-backend")&&(f.gemini_backend=normalizeGeminiBackend(get("set-gemini-\
backend").value)),get("set-gemini-vertex-project")&&(f.gemini_vertex_project=get("set-gemini-vertex-\
project").value),get("set-gemini-vertex-location")&&(f.gemini_vertex_location=get("set-gemini-vertex\
-location").value),get("set-gemini-vertex-credentials-json")&&(f.gemini_vertex_credentials_json=get(
"set-gemini-vertex-credentials-json").value),get("set-xai")&&(f.xai_key=get("set-xai").value),get("s\
et-google-key")&&(f.google_key=get("set-google-key").value),get("set-google-project")&&(f.google_project=
get("set-google-project").value),get("set-admin-api-key-mode")&&(f.admin_api_key_mode=normalizeAdminApiKeyMode(
get("set-admin-api-key-mode").value)),get("set-bot-detect")&&(f.bot_detection_enabled=get("set-bot-d\
etect").checked),get("set-bot-detect-global")&&(f.bot_detection_global_enabled=get("set-bot-detect-g\
lobal").checked);const _=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(f)});if(_.ok){let C="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const D=await _.json();D&&D.message&&(C=D.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!f.light_mode_enabled,liquid_glass_enabled:!!f.liquid_glass_enabled})),Rt();
const A=currentUsername,B=CHAT_CONFIG.enableE2EE;enterToSend=f.enter_to_send,autoSearchOnLinks=f.auto_search_on_links;
const $=useSwCache;useSwCache=f.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=
!!f.clear_cache_on_version_update),compactPromptMode=f.compact_prompt_mode,minimalPromptMode=f.minimal_prompt_mode,
voiceStudioUiEnabled=f.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=f.temp_chat_timeout_seconds,
applyThemeColor(f.theme_color,!0),syncThemeInputs(f.theme_color),applyLightMode(f.light_mode_enabled),
applyLiquidGlassMode(f.liquid_glass_enabled),applyAdaptiveBlurPreference(get("set-background-blur-mo\
de")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),minimalPromptMode?setMinimalPromptMode(
!0):setCompactPromptMode(compactPromptMode),updateStsUi(),$!==useSwCache&&applyCacheMode(useSwCache,
{forceCleanup:!useSwCache}),showToast(C,"success"),syncClientDebugLogToggle(f.enable_client_debug_log,
"settings saved"),f.new_username&&f.new_username!==A?setTimeout(()=>location.reload(),1e3):f.new_password&&
showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let C={};try{C=await _.json()}catch{}showToast(C.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const u=get("2fa-badge");u&&(u.innerText="D\
ISABLED",u.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const c=get("bot-unban-username"),u=c?c.value.trim():
"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const m=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"single"})}),f=await m.json(),v=get("bot-unban-result");if(m.
ok&&f&&f.status==="ok")v&&(v.textContent=`${u} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
v.classList.remove("hidden")),c&&(c.value="");else{const k=f&&f.error?f.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(k,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const c=get("bot-unban-username"),u=c?c.value.trim():"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const m=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"linked"})}),f=await m.json(),v=get("bot-unban-result");if(m.
ok&&f&&f.status==="ok")v&&(v.textContent=`${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
v.classList.remove("hidden")),c&&(c.value="");else{const k=f&&f.error?f.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(k,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const c=get(
"bot-speed-test-btn"),u=get("bot-speed-test-result");c&&(c.disabled=!0),c&&c.classList.add("opacity-\
60","cursor-not-allowed"),u&&(u.classList.remove("hidden"),u.textContent="\u5B9F\u884C\u4E2D...");try{
const m=a(V=>{u&&(u.textContent=V)},"setBox"),f=a(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),v=a((V,ie)=>!V||!ie||ie<=0?0:V*8/(ie/1e3)/1e3/1e3,"toMbps"),k=a(V=>Number.isFinite(
V)?`${V.toFixed(0)} ms`:"-","fmtMs"),_=a(V=>Number.isFinite(V)?`${V.toFixed(V>=100?0:1)} Mbps`:"-","\
fmtMbps"),C=a(async(V,ie)=>{const xe=await V.json().catch(()=>({}));return xe&&xe.error?xe.error:ie},
"parseErr"),A=[];m("\u6E2C\u5B9A\u4E2D... ping");for(let V=0;V<4;V++){const ie=performance.now(),xe=await apiFetch(
`/api/speedtest/ping?_=${f()}`,{cache:"no-store"}),Ie=performance.now();if(!xe.ok)throw new Error(await C(
xe,"ping_failed"));await xe.json().catch(()=>({})),A.push(Ie-ie)}const B=A.reduce((V,ie)=>V+ie,0)/Math.
max(1,A.length),$=Math.min(...A),D=a(async V=>{const ie=performance.now(),xe=await apiFetch(`/api/sp\
eedtest/download?bytes=${V}&_=${f()}`,{cache:"no-store"});if(!xe.ok)throw new Error(await C(xe,"down\
load_failed"));const Ie=await xe.arrayBuffer(),Ne=performance.now();return{bytes:Ie.byteLength||V,ms:Ne-
ie,mbps:v(Ie.byteLength||V,Ne-ie)}},"runDownload");m(`\u6E2C\u5B9A\u4E2D... ping ${k(B)}
\u6E2C\u5B9A\u4E2D... download`);const ee=[];for(const V of[2*1024*1024,8*1024*1024])ee.push(await D(
V)),m(`\u6E2C\u5B9A\u4E2D... ping ${k(B)}
download ${_(Math.max(...ee.map(ie=>ie.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const P=Math.max(...ee.map(V=>V.mbps)),M=a(async V=>{const ie=new Uint8Array(
V),xe=performance.now(),Ie=await apiFetch(`/api/speedtest/upload?_=${f()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ie,cache:"no-store"}),Ne=performance.now();if(!Ie.ok)
throw new Error(await C(Ie,"upload_failed"));const Ke=await Ie.json().catch(()=>({})),Ze=Number(Ke.bytes_received||
V)||V;return{bytes:Ze,ms:Ne-xe,mbps:v(Ze,Ne-xe),serverMs:Number(Ke.server_elapsed_ms||0)||0}},"runUp\
load"),H=[];for(const V of[1*1024*1024,4*1024*1024])H.push(await M(V));const z=Math.max(...H.map(V=>V.
mbps)),ye=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",`Ping\
 (avg/min): ${k(B)} / ${k($)}`,`Download (best): ${_(P)}`,`Upload (best): ${_(z)}`,`Download runs: ${ee.
map(V=>`${Math.round(V.bytes/1024/1024)}MB=${_(V.mbps)}`).join(", ")}`,`Upload runs: ${H.map(V=>`${Math.
round(V.bytes/1024/1024)}MB=${_(V.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\
\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];m(ye.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(m){u&&(u.textContent=`\u30A8\u30E9\u30FC: ${m&&m.message?m.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{c&&(c.disabled=!1,c.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>ze()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Qe()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async c=>{const u=c.target.closest("button");if(!u)return;const m=u.getAttribute("data-id");
if(u.classList.contains("ban-appeal-mark")){m&&await Qe([Number(m)]);return}if(u.classList.contains(
"ban-appeal-status")){const f=u.getAttribute("data-status");m&&f&&await dt({id:Number(m),status:f});
return}if(u.classList.contains("ban-appeal-reply-send")){const f=u.closest("[data-appeal-id]"),v=f?f.
querySelector(".ban-appeal-reply"):null,k=v?v.value:"";m&&await dt({id:Number(m),admin_reply:k});return}
if(u.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";m&&await dt({id:Number(
m),block_user:!0,block_reason:f});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const c=get("photo-input");c&&c.click()}),get("upload-clear-b\
tn")&&(get("upload-clear-btn").onclick=()=>{resetUploadState()}),get("marker-modal-close")&&(get("ma\
rker-modal-close").onclick=()=>{closeMarkerModal(),markerState.row=null}),get("marker-tool-draw")&&(get(
"marker-tool-draw").onclick=()=>setMarkerMode("draw")),get("marker-tool-mosaic")&&(get("marker-tool-\
mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool-crop")&&(get("marker-tool-crop").onclick=
()=>setMarkerMode("crop"));const yn=get("marker-color-picker");yn&&(yn.oninput=c=>setMarkerColor(c.target.
value),yn.onchange=c=>setMarkerColor(c.target.value));const vn=get("marker-opacity");vn&&(vn.oninput=
c=>setMarkerOpacity(c.target.value),vn.onchange=c=>setMarkerOpacity(c.target.value));const ln=get("m\
arker-opacity-number");ln&&(ln.onchange=c=>setMarkerOpacity(c.target.value),ln.onblur=c=>setMarkerOpacity(
c.target.value),ln.onkeydown=c=>{c.key==="Enter"&&(setMarkerOpacity(c.target.value),c.target.blur())}),
document.querySelectorAll("#marker-toolbar .marker-color-chip[data-marker-color]").forEach(c=>{c.onclick=
()=>setMarkerColor(c.getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-vie\
w-reset").onclick=()=>resetMarkerTransform()),get("marker-crop-reset")&&(get("marker-crop-reset").onclick=
()=>clearCropRect()),get("marker-undo")&&(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("ma\
rker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),get("marker-save")&&(get("marker\
-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),initCropCanvas(),
window.addEventListener("resize",()=>{const c=get("marker-modal");!c||c.classList.contains("hidden")||
(applyMarkerTransform(),renderCropOverlay())});const ai=a(()=>{const c=get("upload-modal");return!!(c&&
!c.classList.contains("hidden"))},"isUploadModalOpen"),Ft=get("drop-overlay");let Kt=0;const oi=a(()=>{
ai()||Ft&&(Ft.classList.remove("hidden"),Ft.classList.add("flex"))},"showDropOverlay"),Xt=a(()=>{Kt=
0,Ft&&(Ft.classList.add("hidden"),Ft.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
Xt;const ft=get("upload-dropzone");ft&&(ft.addEventListener("dragover",c=>{c.preventDefault(),ft.classList.
add("dragover")}),ft.addEventListener("dragleave",()=>{ft.classList.remove("dragover")}),ft.addEventListener(
"drop",c=>{c.preventDefault(),c.stopPropagation(),ft.classList.remove("dragover"),Xt();const u=c.dataTransfer?
c.dataTransfer.files:null;u&&u.length&&handleFiles(u)})),window.addEventListener("dragenter",c=>{!c.
dataTransfer||!c.dataTransfer.types||!c.dataTransfer.types.includes("Files")||(Kt+=1,oi())}),window.
addEventListener("dragover",c=>{!c.dataTransfer||!c.dataTransfer.types||!c.dataTransfer.types.includes(
"Files")||c.preventDefault()}),window.addEventListener("dragleave",c=>{!c.dataTransfer||!c.dataTransfer.
types||!c.dataTransfer.types.includes("Files")||(Kt=Math.max(0,Kt-1),(Kt===0||!c.relatedTarget||c.clientY<=
0||c.clientX<=0||c.clientX>=window.innerWidth||c.clientY>=window.innerHeight)&&Xt())}),window.addEventListener(
"dragend",()=>{Xt()}),window.addEventListener("drop",c=>{Xt(),!(!c.dataTransfer||!c.dataTransfer.files||
c.dataTransfer.files.length===0)&&(c.preventDefault(),!(ft&&ft.contains(c.target))&&handleFiles(c.dataTransfer.
files))});const jn=get("bot-admin-modal"),ri=a(c=>{const u=get("bot-admin-list");if(u){if(u.innerHTML=
"",!c||!c.length){u.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}c.forEach((m,f)=>{const v=!!m.is_bot_banned,k=m.bot_detection_enabled!==!1,_=document.createElement(
"div");_.className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs m\
odel-list-animate",_.style.animationDelay=`${Math.min(f,12)*.02}s`,_.innerHTML=`
                        <div class="flex-1">
                            <div class="text-gray-200 font-bold">${escapeHtml(m.username)}</div>
                            <div class="text-[10px] text-gray-500">${v?"BAN\u4E2D":"\u6B63\u5E38"} ${m.
bot_ban_reason?" / "+escapeHtml(m.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(m.username)}" data-enabled="${k?"1":"0"}">${k?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${v?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(m.username)}" data-bann\
ed="${v?"1":"0"}">${v?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${v?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(m.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-\
2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(m.username)}">\u524A\u9664</button>\

                    `,u.appendChild(_)})}},"renderBotUsers"),Yt=a(async(c="")=>{const u=get("bot-adm\
in-list");u&&(u.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const m=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
c)}`),f=await m.json();m.ok&&f&&f.users?ri(f.users):(u&&(u.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{u&&(u.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),wn=a(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||jn))return;const u=get(
"settings-modal");u&&(u.classList.contains("modal-open")||u.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await Yt(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=wn,window.closeBotAdminModal=(c=!1)=>{(get(
"bot-admin-modal")||jn)&&hideModal("bot-admin-modal"),!c&&location.pathname==="/admin-bots"&&history.
back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{wn()}),get("bot-admin-close")&&(get(
"bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-admin-search-btn")&&(get("bot-admin-se\
arch-btn").onclick=async()=>{await Yt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}),
get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=async()=>{await Yt("")}),get("bo\
t-admin-search")&&get("bot-admin-search").addEventListener("keydown",async c=>{c.key==="Enter"&&await Yt(
get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get("bot-admin-list").onclick=async c=>{
const u=c.target.closest("button");if(!u)return;const m=u.getAttribute("data-user");if(!m)return;let f;
if(u.classList.contains("bot-toggle-detect")){const v=u.getAttribute("data-enabled")!=="1";f=await apiFetch(
"/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:m,
action:"toggle_detection",enabled:v})})}else if(u.classList.contains("bot-toggle-ban"))if(u.getAttribute(
"data-banned")==="1")f=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"appl\
ication/json"},body:JSON.stringify({username:m,action:"unban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m}\
 \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))return;f=await apiFetch("/api/bot/update",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({username:m,action:"ban",reason:"Adm\
in ban"})})}else if(u.classList.contains("bot-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m}\
 \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))return;f=await apiFetch("/a\
pi/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:m,
action:"unban_linked"})})}else if(u.classList.contains("bot-delete-account")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m}\
 \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;f=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:m,action:"delete_account"})})}if(f){if(f.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${m}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(f.ok){if(u.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${m}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),m===currentUsername)){location.href="/";return}}else{
let v={};try{v=await f.json()}catch{}showToast(v.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}await Yt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}});const jt={"\
/settings":{id:"settings-modal",open:a(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload\
-modal",open:a(()=>openUploadModal(),"open")},"/library":{id:"lib-modal",open:a(()=>{Kn(!1),showModal(
"lib-modal"),loadLibraryFiles()},"open")},"/history":{id:"history-modal",open:a(()=>window.showHistoryModal(),
"open")},"/branch":{id:"branch-modal",open:a(()=>window.showBranchModal(),"open")},"/batch":{id:"bat\
ch-modal",open:a(()=>window.showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:a(()=>openRichPasteModal(),
"open")},"/camera":{id:"camera-capture-modal",open:a(()=>openCameraCaptureModal(),"open")},"/edit-im\
age":{id:"marker-modal",open:a(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:a(()=>window.
openThreadModal(),"open")},"/model":{id:"model-modal",open:a(()=>openModelModal(),"open")},"/token-d\
etails":{id:"token-detail-modal",open:a(()=>showTokenDetailModal(),"open")},"/encryption-status":{id:"\
encryption-status-modal",open:a(()=>showEncryptionStatusModal(),"open")},"/python-execution":{id:"py\
thon-exec-modal",open:a(()=>showPythonExecDetailModal(),"open")},"/gem":{id:"gem-modal",open:a(()=>{
editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>C\
reate New Gem',showModal("gem-modal")},"open")},"/compression":{id:"compression-modal",open:a(()=>window.
openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:a(()=>wn(),"open")}},Dn=a((c,u=!1)=>{
switch(c){case"settings-modal":Rt(u);break;case"upload-modal":closeUploadModal(u);break;case"camera-\
capture-modal":closeCameraCaptureModal(u?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(u);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(u);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(u);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(u);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
u);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(u);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(u);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(u);break;case"token-detail-modal":closeTokenDetail(u);break;case"encryption-s\
tatus-modal":closeEncryptionModal(u);break;case"python-exec-modal":closePythonExecDetail(u);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(u);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(u);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
u);break;case"version-update-modal":const m=localStorage.getItem("app_version")||"";m&&localStorage.
setItem("version_notified",m),hideModal(c);break;default:hideModal(c);break}},"closeModalById");window.
addEventListener("popstate",c=>{let u=!1;Object.values(jt).forEach(v=>{const k=get(v.id);k&&k.classList.
contains("modal-open")&&location.pathname!==Object.keys(jt).find(_=>jt[_].id===v.id)&&(Dn(v.id,!0),u=
!0)});const m=location.pathname.match(/^\/c\/(.+)$/);if(m){const v=decodeURIComponent(m[1]);String(currentThreadId)!==
String(v)&&loadMessages(v,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat(
{skipHistory:!0});const f=jt[location.pathname];if(f){const v=get(f.id);v&&!v.classList.contains("mo\
dal-open")&&f.open()}});const Hn=location.pathname;jt[Hn]&&(history.replaceState({},"","/"),setTimeout(
()=>jt[Hn].open(),500)),get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const c=get(
"easy-login-mins"),u=c?parseInt(c.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${u}\
\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const f=await(await apiFetch("\
/api/easy_login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:u})})).
json();f&&f.temp_password?(get("easy-login-code").textContent=f.temp_password,get("easy-login-exp").
textContent=f.expires_at||"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\
\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}),get("easy-login-cancel")&&(get(
"easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const u=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(u&&u.cancelled){const m=get("easy-login-r\
esult");m&&m.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("fb-submit").onclick=async()=>{const c=get("fb-title").value.trim(),u=get("fb-mess\
age").value.trim();if(!u){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}await apiFetch("/api/feedback",{method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({title:c,message:u})}),get("fb-title").value="",get("fb-message").value=
"",xn()};async function xn(){const u=await(await apiFetch("/api/feedback?all=1")).json(),m=get("fb-l\
ist");m.innerHTML="",(u.items||[]).filter(k=>!u.is_admin||k.user_id===void 0||k.user_id===null||!0).
forEach(k=>{if(u.is_admin)return;const _=document.createElement("div");_.className="p-2 rounded bord\
er border-gray-700 bg-gray-800/50",_.innerHTML=`<div class="text-[11px] text-gray-400">${k.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(k.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(k.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
k.status)}</div>${k.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(k.
admin_reply)}</div>`:""}`,m.appendChild(_)});const f=get("fb-admin-panel"),v=get("fb-admin-list");u.
is_admin?(f.classList.remove("hidden"),v.innerHTML="",(u.items||[]).forEach(k=>{const _=document.createElement(
"div");_.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",_.innerHTML=`
                            <div class="text-[11px] text-gray-400">#${k.id} / user:${k.user_id} / ${k.
created_at}</div>
                            <div class="font-bold text-sm">${escapeHtml(k.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(k.message)}</div>
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
ded px-2 py-1 text-xs text-white" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(k.admin_reply||
"")}</textarea>
                        `,_.querySelector(".fb-status").value=k.status||"new",_.querySelector(".fb-s\
ave").onclick=async()=>{const C=_.querySelector(".fb-status").value,A=_.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${k.id}/update`,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({status:C,admin_reply:A})}),xn()},v.appendChild(_)})):f.classList.add("hidd\
en")}if(a(xn,"loadFeedback"),window.setupTOTP=async()=>{const u=await(await apiFetch("/api/2fa/totp/\
setup",{method:"POST"})).json();get("totp-qr").src=u.qr_image,get("totp-secret-disp").innerText=u.secret,
get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const c=get("totp-ver\
ify-code").value;if(!c)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({code:c})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const c=get("register-webauthn-btn"),u=get("webauthn-nam\
e"),m=u?String(u.value||"").trim():"";try{c&&(c.disabled=!0);const f=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),v=await f.json();if(!f.ok){showToast(v.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const _=await(await ensureWebAuthnJson()).create({publicKey:v}),C=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},_,{name:m}))}),A=await C.json().catch(()=>({}));C.ok?(u&&(u.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(A.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(f){showToast(`WebAuthn Error: ${f}`,"error",!0)}finally{c&&(c.disabled=!1)}},window.
removeWebAuthnCredential=async c=>{if(!c||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const u=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:c})}),m=await u.json().catch(()=>({}));if(u.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(m.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let c;try{c=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(c.ok){location.href="/";
return}let u={};try{u=await c.json()}catch{}if(u&&u.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(u.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=c=>{if(c.isComposing)return;const u=get("prompt-input");
if(slashSuggestionsVisible){const m=get("slash-command-suggestions");if(c.key==="ArrowDown"){c.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(u.value),u.value));
return}if(c.key==="ArrowUp"){c.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(u.value),u.value));
return}if(c.key==="Enter"){c.preventDefault();const f=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(u.value),u.value));f[slashSelectedIndex]?selectSlashCommand(f[slashSelectedIndex].
id):f.length>0&&selectSlashCommand(f[0].id);return}if(c.key==="Escape"){c.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const m=u.value.trim();if(c.key==="ArrowDown"){c.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(m.substring(1));return}if(c.key==="ArrowUp"){
c.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(m.substring(1));
return}if(c.key==="Enter"){c.preventDefault();const f=m.substring(1).toLowerCase(),v=loadedGems.filter(
k=>k.name.toLowerCase().includes(f)||k.description&&k.description.toLowerCase().includes(f));v[gemSelectedIndex]?
selectGemSuggestion(v[gemSelectedIndex]):v.length>0&&selectGemSuggestion(v[0]);return}if(c.key==="Es\
cape"){c.preventDefault(),hideGemSuggestions();return}}if(c.key==="Escape"&&pendingSlashCommand){c.preventDefault(),
hidePendingSlashCommandIndicator();return}c.key==="ArrowUp"&&(u.selectionStart===0||c.ctrlKey)?promptHistory.
length>0&&(historyIndex===-1&&(tempPrompt=u.value),historyIndex<promptHistory.length-1&&(c.preventDefault(),
historyIndex++,u.value=promptHistory[historyIndex],u.dispatchEvent(new Event("input")))):c.key==="Ar\
rowDown"&&(u.selectionEnd===u.value.length||c.ctrlKey)&&historyIndex>-1&&(c.preventDefault(),historyIndex--,
historyIndex===-1?u.value=tempPrompt:u.value=promptHistory[historyIndex],u.dispatchEvent(new Event("\
input"))),enterToSend?c.key==="Enter"&&!c.shiftKey&&(c.preventDefault(),sendMessage()):(c.metaKey||c.
ctrlKey)&&c.key==="Enter"&&(c.preventDefault(),sendMessage())},get("prompt-input")&&(get("prompt-inp\
ut").addEventListener("input",function(){this.style.height="auto",this.style.height=this.scrollHeight+
"px",schedulePromptTokenEstimate(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const c=this.
value.trim();if(pendingSlashCommand)gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&
hideSlashCommandSuggestions(),lastSlashFilter=null;else if(c.startsWith("@")){const u=c.substring(1);
showGemSuggestions(u),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=null}else if(c.
startsWith("/")){const u=slashCommandSuggestionFilter(extractSlashCommandToken(c),this.value);(!slashSuggestionsVisible||
u!==lastSlashFilter)&&(lastSlashFilter=u,showSlashCommandSuggestions(u)),gemSuggestionsVisible&&hideGemSuggestions()}else
gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=
null}),get("prompt-input").addEventListener("blur",()=>{setTimeout(()=>{slashSuggestionsVisible&&hideSlashCommandSuggestions(),
gemSuggestionsVisible&&hideGemSuggestions()},150)})),get("cancel-edit-btn")&&(get("cancel-edit-btn").
onclick=cancelEdit),updatePromptPlaceholder(),aiSettingsConversation.length>0&&(pendingSlashCommand=
"settings",showPendingSlashCommandIndicator("settings")),get("search-box")&&(get("search-box").addEventListener(
"input",c=>{const u=get("search-box");if(u&&isUserInitiatedSearchInput(c))markThreadSearchUserEdited(
u);else if(u&&!u.dataset.userEdited){discardAutofilledThreadSearch("cleared-autofill-search-box-inpu\
t");return}if(isSettingsModalOpen()){snapshotSidebarHistory("ignore-search-input-settings-open");return}
clearTimeout(searchTimeout),searchTimeout=setTimeout(()=>{loadThreads(!1)},300)}),hardenThreadSearchInputs()),
get("mobile-new-chat-btn")&&(get("mobile-new-chat-btn").onclick=()=>startNewChat()),get("sts-mic-btn")&&
(get("sts-mic-btn").onclick=()=>{isStsModel()&&get("mic-btn").click()}),get("sts-cancel-btn")&&(get(
"sts-cancel-btn").onclick=()=>{isStsModel()&&Wn()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async c=>{const u=(c.clipboardData||window.clipboardData).items,m=[];for(let f=0;f<u.length;f++)
if(u[f].kind==="file"){const v=u[f].getAsFile();v&&m.push(v)}m.length>0&&(c.preventDefault(),await handleFiles(
m,{openModal:!1}))}),get("rich-paste-btn")&&(get("rich-paste-btn").onclick=()=>openRichPasteModal()),
get("rich-paste-modal-close")&&(get("rich-paste-modal-close").onclick=()=>closeRichPasteModal()),get(
"rich-paste-close-btn")&&(get("rich-paste-close-btn").onclick=()=>closeRichPasteModal()),get("rich-p\
aste-focus-btn")&&(get("rich-paste-focus-btn").onclick=()=>focusRichPasteEditor()),get("rich-paste-c\
lear-btn")&&(get("rich-paste-clear-btn").onclick=()=>clearRichPasteEditor(!0)),get("rich-paste-previ\
ew-btn")&&(get("rich-paste-preview-btn").onclick=()=>openRichPastePreviewTab()),get("rich-paste-send\
-btn")&&(get("rich-paste-send-btn").onclick=()=>sendRichPasteToModel()),get("rich-paste-send-server-\
btn")&&(get("rich-paste-send-server-btn").onclick=()=>sendRichPasteToModel({serverSide:!0})),get("ri\
ch-paste-import-btn")&&(get("rich-paste-import-btn").onclick=async()=>{try{await readClipboardRichContent()||
showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u30EA\u30C3\u30C1\u30C6\u30AD\u30B9\u30C8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002Ctrl+V \u3067\u8CBC\u308A\u4ED8\u3051\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(c){const u=c&&c.message?c.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(u,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const c=get("rich-paste-capture");
c.addEventListener("paste",async u=>{const m=u.clipboardData||window.clipboardData;if(m){u.preventDefault();
try{await ingestRichPasteClipboardData(m)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),c.addEventListener("input",()=>{c.value=""})}get("chat-container").addEventListener(
"click",c=>{const u=c.target.closest("img.chat-image"),m=u?u.dataset.viewerSrc||u.currentSrc||u.src:
"";u&&m&&(c.preventDefault(),openImageViewer(m))});const Qt=document.querySelector(".viewer-content");
Qt&&(Qt.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),Qt.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),Qt.addEventListener("touchend",onViewerTouchEnd),Qt.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",c=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(c.target.id==="image-viewer"||c.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",c=>{c.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",c=>{c.key==="Escape"&&closeImageViewer()});
let Ve,Ae=null,cn=[],kn=!1,dn=null,Dt=null,Tt=null,Zt=null,_n=0,un=!1,en=null,Ht=null,bt=null,tn=null,
pn=null,Ct=null,mn=null,fn=null;function qn(){const c=get("mic-waveform");if(!c)return[];if(Array.isArray(
mn)&&mn.length)return mn;c.innerHTML="";const u=[];for(let m=0;m<24;m++){const f=document.createElement(
"span");f.className="block rounded-full",f.style.background="rgba(252, 165, 165, 0.92)",f.style.width=
"2px",f.style.transition="height 75ms linear, opacity 75ms linear",f.style.height="2px",f.style.opacity=
"0.4",u.push(f),c.appendChild(f)}return mn=u,u}a(qn,"ensureMicWaveformBars");function Lt(c,u="hidden"){
const m=get("mic-recording-indicator"),f=get("mic-recording-text");if(m){if(fn&&(clearTimeout(fn),fn=
null),u==="hidden"){m.classList.add("hidden");return}f&&c&&(f.innerText=c),m.classList.remove("hidde\
n"),u==="recording"?m.style.color="rgb(252 165 165)":u==="processing"?m.style.color="rgb(253 224 71)":
m.style.color="rgb(209 213 219)"}}a(Lt,"setMicRecordingIndicator");function Gn(){qn().forEach(u=>{u.
style.height="2px",u.style.opacity="0.35"})}a(Gn,"resetMicWaveformBars");function Mt(){if(pn&&(cancelAnimationFrame(
pn),pn=null),tn){try{tn.disconnect()}catch{}tn=null}if(Ht){try{Ht.close()}catch{}Ht=null}bt=null,Ct=
null,Gn()}a(Mt,"stopMicWaveform");function Un(c){Mt();const u=qn();if(!u.length)return;const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;try{Ht=new m,bt=Ht.createAnalyser(),bt.fftSize=
256,bt.smoothingTimeConstant=0,tn=Ht.createMediaStreamSource(c),tn.connect(bt),Ct=new Uint8Array(bt.
frequencyBinCount)}catch{Mt();return}const f=a(()=>{if(!bt||!Ct)return;bt.getByteFrequencyData(Ct);const v=Math.
max(1,Math.floor(Ct.length/u.length));for(let k=0;k<u.length;k++){const C=(Ct[Math.min(Ct.length-1,k*
v)]||0)/255,A=Math.max(2,Math.round(2+C*10));u[k].style.height=`${A}px`,u[k].style.opacity=`${.35+C*
.65}`}pn=requestAnimationFrame(f)},"render");f()}a(Un,"startMicWaveform");function hn(){if(dn&&(clearInterval(
dn),dn=null),Zt){try{Zt.disconnect()}catch{}Zt=null}if(Dt){try{Dt.close()}catch{}Dt=null}Tt=null}a(hn,
"stopSilenceMonitor");function zn(c){if(!isStsModel()||!stsOpt("sts-auto-send"))return;hn();const u=window.
AudioContext||window.webkitAudioContext;if(!u)return;Dt=new u,Tt=Dt.createAnalyser(),Tt.fftSize=2048,
Zt=Dt.createMediaStreamSource(c),Zt.connect(Tt);const m=new Uint8Array(Tt.fftSize),f=getStsSilenceMs(),
v=.02;_n=0,un=!1,dn=setInterval(()=>{if(!Tt)return;Tt.getByteTimeDomainData(m);let k=0;for(let C=0;C<
m.length;C++){const A=(m[C]-128)/128;k+=A*A}if(Math.sqrt(k/m.length)>v){un||(un=!0),_n=Date.now();return}
un&&Date.now()-_n>f&&Ve&&Ve.state==="recording"&&Ve.stop()},200)}a(zn,"startSilenceMonitor");const Cn=class Cn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null}async start(u,m,f,v={}){this.model=f,this.ws=new WebSocket(`${m}?access_token=${u}`),
this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Send\
ing setup...");const C=!!(v&&v.transcriptionConfig),A={setup:{model:`models/${f}`,generationConfig:{
responseModalities:C?["TEXT"]:["AUDIO"]},inputAudioTranscription:C?v.transcriptionConfig||{}:{},outputAudioTranscription:{}}};
v.speechConfig&&(A.setup.generationConfig.speechConfig=v.speechConfig),v.thinkingConfig&&(A.setup.generationConfig.
thinkingConfig=v.thinkingConfig),v.translationConfig&&(A.setup.translationConfig=v.translationConfig),
console.log("Sending setup:",JSON.stringify(A)),this.ws.send(JSON.stringify(A))},this.ws.onmessage=C=>this.
_handleMessage(C),this.ws.onerror=C=>{console.error("Gemini Live WebSocket error:",C),this.onError&&
this.onError(C)},this.ws.onclose=C=>{console.log("Gemini Live WebSocket closed:",C.code,C.reason),this.
onClose&&this.onClose(C)},this.audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
this.stream=await navigator.mediaDevices.getUserMedia({audio:!0});const k=this.audioContext.createMediaStreamSource(
this.stream);this.processor=this.audioContext.createScriptProcessor(4096,1,1),this.userAudioChunks=[];
const _=new MediaRecorder(this.stream);_.ondataavailable=C=>{C.data.size>0&&this.userAudioChunks.push(
C.data)},_.start(500),this.backupRecorder=_,this.processor.onaudioprocess=C=>{if(!this.ws||this.ws.readyState!==
WebSocket.OPEN||!this.setupComplete)return;const A=C.inputBuffer.getChannelData(0),B=new Int16Array(
A.length);for(let $=0;$<A.length;$++)B[$]=Math.max(-1,Math.min(1,A[$]))*32767;this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(B.buffer))),mimeType:"\
audio/pcm;rate=16000"}}}))},k.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(u){
const m=JSON.parse(u.data);if(console.log("Gemini Live raw message received:",m),m.setupComplete&&(console.
log("Gemini Live setup complete confirmed"),this.setupComplete=!0),m.serverContent){const f=m.serverContent;
f.modelTurn&&f.modelTurn.parts.forEach(v=>{if(v.text&&(v.thought?(console.log("Gemini thought delta:",
v.text),this.assistantThought+=v.text):(console.log("Gemini transcript delta (parts):",v.text),this.
assistantText+=v.text)),v.inlineData&&v.inlineData.data){const k=v.inlineData.data;console.log("Gemi\
ni audio chunk received, size:",k.length),this.rtPlayer&&this.rtPlayer.addChunk(k);const _=atob(k),C=new Uint8Array(
_.length);for(let A=0;A<_.length;A++)C[A]=_.charCodeAt(A);this.assistantAudioChunks.push(C)}}),f.outputTranscription&&
(console.log("Gemini output transcription delta:",f.outputTranscription.text),this.assistantText.includes(
f.outputTranscription.text)||(this.assistantText+=f.outputTranscription.text)),f.inputTranscription&&
(console.log("User input transcription delta:",f.inputTranscription.text),this.inputTranscript+=f.inputTranscription.
text,this.interimInputTranscript=""),f.interimInputTranscription&&(console.log("User interim transcr\
iption:",f.interimInputTranscription.text),this.interimInputTranscript=f.interimInputTranscription.text)}
this.onMessage&&this.onMessage(m)}stop(){this.ws&&this.ws.close(),this.processor&&this.processor.disconnect(),
this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().forEach(u=>u.stop()),
this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const u=new Blob(this.assistantAudioChunks),
m=await this._blobToBase64(u),f=new Blob(this.userAudioChunks),v=await this._blobToBase64(f);return{
user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.assistantThought,
audio_base64:m,user_audio_base64:v}}_blobToBase64(u){return new Promise(m=>{const f=new FileReader;f.
onloadend=()=>m(f.result.split(",")[1]),f.readAsDataURL(u)})}};a(Cn,"GeminiLiveClient");let Sn=Cn;const Ln=class Ln{constructor(u=24e3){
const m=window.AudioContext||window.webkitAudioContext;this.ctx=new m({sampleRate:u}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(u){if(!this.ctx)return;const m=atob(u),f=new Uint8Array(
m.length);for(let B=0;B<m.length;B++)f[B]=m.charCodeAt(B);const v=new Int16Array(f.buffer),k=new Float32Array(
v.length);for(let B=0;B<v.length;B++)k[B]=v[B]/32768;const _=this.ctx.createBuffer(1,k.length,this.ctx.
sampleRate);_.getChannelData(0).set(k),this.ctx.state==="suspended"&&await this.ctx.resume();const C=this.
ctx.createBufferSource();C.buffer=_,C.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const A=Math.max(this.ctx.currentTime,this.nextStartTime);
C.start(A),this.nextStartTime=A+_.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};a(Ln,
"RealTimeAudioPlayer");let nn=Ln;const Mn=class Mn{constructor(){this.active=!1,this.capturing=!1,this.
sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=
null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript=
"",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=
null,this.saved=!1,this.saving=!1,this.stopping=!1}isActive(){return this.active}async start(){if(this.
active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const u=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const v=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=v.id!==null&&v.id!==void 0?String(v.id):v.id,setTemporaryChatUiState(!!(v&&v.
is_temporary)),setCurrentChatHeaderTitle(v&&v.title),applyTemporaryChatRuntimeMeta(v||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+v.id),get("welcome-screen").classList.add("hidden")}catch(f){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+f.message,"\
error",!0);return}const m={model:u,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,target_lang:isGeminiLiveTranslateModel()&&get("sts-target-lang")?get(
"sts-target-lang").value:""};setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const f=await apiFetch("/a\
pi/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)}),
v=await f.json().catch(()=>({}));if(!f.ok)throw new Error(v.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=v.session_id,this.rateIn=v.rate_in||this.rateIn,this.rateOut=v.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null}catch(f){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
f.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(f){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
f.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const u="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),m=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(u,m).then(f=>{if(!f.ok)throw new Error("SSE stream failed ("+f.status+")");this.
reader=f.body.getReader(),this._readLoop()}).catch(f=>{f&&f.name==="AbortError"||(this.streamError=f&&
f.message?f.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const u=new TextDecoder;let m="";try{for(;this.reader;){const{done:f,
value:v}=await this.reader.read();if(f)break;m+=u.decode(v,{stream:!0});let k;for(;(k=m.indexOf(`

`))>=0;){const _=m.slice(0,k);m=m.slice(k+2);for(const C of _.split(`
`)){if(!C.startsWith("data: "))continue;let A=null;try{A=JSON.parse(C.slice(6))}catch{continue}this.
_handleEvent(A)}}}}catch(f){if(f&&f.name==="AbortError")return;this.active&&(this.streamError=f&&f.message?
f.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(u){
if(u)switch(u.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new nn(this.rateOut||24e3),Gt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(u.data));break;case"transcript":u.role==="user"?(u.cumulative?this.userTranscript=
u.delta:this.userTranscript+=u.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
u.role==="assistant"?(this.assistantTranscript+=u.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):u.role==="thought"&&(this.assistantThought+=u.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":u.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"error":this.streamError=u.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Gt=null}_startCapture(){const u=window.
AudioContext||window.webkitAudioContext;if(!u)throw new Error("AudioContext not supported");return this.
audioCtx=new u({sampleRate:this.rateIn||24e3}),navigator.mediaDevices.getUserMedia(Jn()).then(m=>{this.
stream=m;const f=this.audioCtx.createMediaStreamSource(m),v=this.rateIn||24e3,k=this.audioCtx.sampleRate,
_=4096;this.processor=this.audioCtx.createScriptProcessor(_,1,1),this.processor.onaudioprocess=C=>{if(!this.
active||!this.capturing)return;const A=C.inputBuffer.getChannelData(0),B=li(A,k,v);!B||!B.byteLength||
this._sendAudio(B)},f.connect(this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(u){
if(!this.sessionId||!this.active)return;const m="/api/realtime/audio?session_id="+encodeURIComponent(
this.sessionId),f={method:"POST",credentials:"include",headers:{"X-CSRF-Token":csrfToken,"Content-Ty\
pe":"application/octet-stream"},body:u},v=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions==
"function"?window.ProgressSpinner.manualRequestOptions(f):f;fetch(m,v).catch(()=>{})}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(u=>u.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}hn(),Mt()}async stop(){if(!this.active)return;this.active=
!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0);try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({session_id:this.sessionId})})}catch{}const u=Date.now(),m=this.responseDoneCount;
let f=this.lastAudioAt;for(;Date.now()-u<2e4&&!(this.responseDoneCount>m||(this.lastAudioAt>f&&(f=this.
lastAudioAt),!this.speechActive&&Date.now()-u>2e3&&Date.now()-f>2500));)await new Promise(v=>setTimeout(
v,250));await this._save()}async _save(){if(!this.saved){this.saved=!0,this.saving=!0;try{const u=await apiFetch(
"/api/realtime/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:this.sessionId,thread_id:currentThreadId})}),m=await u.json().catch(()=>({}));if(!u.ok)throw new Error(
m.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(u){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(u&&u.message?u.message:u),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this._stopCapture(),this._stopPlayback(),this.abortCtrl){try{this.abortCtrl.
abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const u=get("mic-btn");u&&(u.
classList.remove("bg-red-600","animate-pulse"),u.classList.add("bg-gray-700"))}};a(Mn,"RealtimeVoice\
Session");let Tn=Mn;function li(c,u,m){let f=c;if(u!==m&&u>0&&m>0){const k=u/m,_=Math.floor(f.length/
k),C=new Float32Array(_);for(let A=0;A<_;A++)C[A]=f[Math.min(Math.floor(A*k),f.length-1)];f=C}const v=new Int16Array(
f.length);for(let k=0;k<f.length;k++){const _=Math.max(-1,Math.min(1,f[k]));v[k]=_<0?_*32768:_*32767}
return v.buffer}a(li,"pcm16FromFloat32");const qt=new Tn;(()=>{const m={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let f=null,v=null,k=!1,_=null,C=!1,A=0,B=0,$=null,
D="idle",ee=!1,P=null;const M=a(I=>document.getElementById(I),"$"),H=a(I=>{const j=Object.assign({},
I||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?
window.ProgressSpinner.manualRequestOptions(j):(j.progressSpinner=!1,j)},"noSpinner");function z(I,j){
D=j;const fe=M("lyria-status-text"),te=M("lyria-status-dot");fe&&(fe.textContent=I),te&&(te.className=
"w-2 h-2 rounded-full inline-block "+(m[j]||m.idle)),xe(),Ie()}a(z,"setStatus");function ye(){const I=B?
Math.floor((Date.now()-B)/1e3):0,j=String(Math.floor(I/60)).padStart(2,"0"),fe=String(I%60).padStart(
2,"0");return`${j}:${fe}`}a(ye,"formatElapsed");function V(){B||(B=Date.now());const I=M("lyria-elap\
sed");I&&(I.textContent=ye()),$||($=window.setInterval(()=>{const j=M("lyria-elapsed");j&&(j.textContent=
ye())},1e3))}a(V,"startElapsedTimer");function ie(){$&&(window.clearInterval($),$=null)}a(ie,"stopEl\
apsedTimer");function xe(){const I=M("lyria-play-btn"),j=M("lyria-pause-btn"),fe=M("lyria-stop-btn"),
te=M("lyria-reset-btn"),be=!!f,$e=D==="streaming"||D==="connecting";if(I){I.disabled=ee||!be;const We=I.
querySelector("i");We&&(We.className="fas fa-play")}j&&(j.disabled=ee||!$e),fe&&(fe.disabled=ee||!be||
!$e),te&&(te.disabled=ee||!be||!$e)}a(xe,"updateTransportButtons");function Ie(){const I=M("lyria-sa\
ve-btn");if(!I)return;const j=!!f&&D!=="idle"&&D!=="connecting"&&D!=="error";I.classList.toggle("hid\
den",!j)}a(Ie,"updateSaveButton");function Ne(I,j){const fe=M("lyria-prompt-rows");if(!fe)return;const te=document.
createElement("div");te.className="flex items-center gap-2",te.innerHTML=`
                        <input type="text" value="${escapeHtml(I||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof j=="num\
ber"?j:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof j=="number"?j:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const be=te.querySelector('input[type="range"]'),$e=te.querySelector(".lyria-w\
eight-label");be&&$e&&be.addEventListener("input",()=>{$e.textContent=parseFloat(be.value).toFixed(1)});
const We=te.querySelector(".lyria-prompt-remove");We&&We.addEventListener("click",()=>{fe.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||te.remove()}),te.classList.add("lyria-prompt-row-wrap"),fe.appendChild(
te)}a(Ne,"addPromptRow");function Ke(){const I=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),j=[];return I.forEach(fe=>{const te=fe.querySelector('input[type="text"]'),be=fe.querySelector(
'input[type="range"]'),$e=(te?te.value:"").trim();$e&&j.push({text:$e,weight:parseFloat(be?be.value:
1)||1})}),j}a(Ke,"collectPrompts");function Ze(){const I={},j=a(ui=>{const En=M(ui);return En&&En.value!==
""?parseFloat(En.value):void 0},"num"),fe=j("lyria-bpm");fe!==void 0&&(I.bpm=Math.round(fe));const te=j(
"lyria-guidance");te!==void 0&&(I.guidance=te);const be=j("lyria-density");be!==void 0&&(I.density=be);
const $e=j("lyria-brightness");$e!==void 0&&(I.brightness=$e);const We=j("lyria-temperature");We!==void 0&&
(I.temperature=We);const je=M("lyria-scale");je&&je.value&&(I.scale=je.value);const Xe=M("lyria-mode");
Xe&&Xe.value&&(I.music_generation_mode=Xe.value);const it=M("lyria-mute-bass"),$t=M("lyria-mute-drum\
s"),Zn=M("lyria-only-bass-drums");return it&&(I.mute_bass=it.checked),$t&&(I.mute_drums=$t.checked),
Zn&&(I.only_bass_and_drums=Zn.checked),I}a(Ze,"collectConfig");function At(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([j,fe])=>{
const te=M(j),be=M(fe);!te||!be||te.addEventListener("input",()=>{const $e=parseFloat(te.value);be.textContent=
j==="lyria-bpm"?String(Math.round($e)):$e.toFixed(1)})})}a(At,"bindRangeLabels");function ht(){if(_){
try{_.close()}catch{}_=null}C=!1,A=0}a(ht,"resetPlayback");function Et(){if(k=!1,v&&typeof v.abort==
"function")try{v.abort()}catch{}v=null}a(Et,"closeStream");async function yt(){Et(),v=new AbortController,
k=!0;try{const I=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(f)}`,H({method:"\
GET",signal:v.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!I.ok){const be=await I.
json().catch(()=>({}));throw new Error(be.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const j=I.body.getReader(),fe=new TextDecoder;let te="";for(;k;){const{done:be,value:$e}=await j.read();
if(be)break;te+=fe.decode($e,{stream:!0});const We=te.split(`

`);te=We.pop();for(const je of We){const Xe=je.split(`
`).find($t=>$t.startsWith("data: "));if(!Xe)continue;const it=Xe.slice(6);try{const $t=JSON.parse(it);
de($t)}catch{}}}}catch(I){if(I&&I.name==="AbortError")return;k&&(z("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{k&&f&&yt()},1200))}finally{k=!1}}a(yt,"openStream");function de(I){
if(I&&I.snapshot){const j=I.status;if(j==="error"){z("\u30A8\u30E9\u30FC","error"),ie();return}if(j===
"closed"||j==="stopped"){z("\u7D42\u4E86","closed"),ie();return}z(j==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",j==="paused"?"paused":"connecting");return}if(I&&I.audio){z("\u518D\u751F\u4E2D...",
"streaming"),V(),ue(I.audio);return}if(I&&I.error){z("\u30A8\u30E9\u30FC: "+I.error,"error"),ie();return}
if(I&&I.final){z("\u7D42\u4E86","closed"),ie(),xe();return}}a(de,"handleStreamMessage");function ue(I){
if(!I)return;if(!_){const je=window.AudioContext||window.webkitAudioContext;if(!je)return;_=new je({
sampleRate:48e3}),C=!1,A=0}let j;try{const je=atob(I);j=new Uint8Array(je.length);for(let Xe=0;Xe<je.
length;Xe++)j[Xe]=je.charCodeAt(Xe)}catch{return}const fe=new Int16Array(j.buffer),te=Math.floor(fe.
length/2);if(te<1)return;const be=_.createBuffer(2,te,48e3);for(let je=0;je<2;je++){const Xe=be.getChannelData(
je);for(let it=0;it<te;it++)Xe[it]=fe[it*2+je]/32768}_.state==="suspended"&&_.resume();const $e=_.createBufferSource();
$e.buffer=be,$e.connect(_.destination),C||(A=_.currentTime+.08,C=!0);const We=Math.max(_.currentTime,
A);$e.start(We),A=We+be.duration}a(ue,"playChunk");async function Pe(I,j){const fe=await fetch("/api\
/gemini/music/command",H({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:f,type:I},j||{}))})),te=await fe.json().catch(()=>({}));if(!fe.ok)throw new Error(
te.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return te}
a(Pe,"apiCommand");async function ce(){if(ee)return;const I=Ke();if(!I.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ee=!0,xe(),z("\u63A5\u7D9A\u4E2D...","connecting");try{
const j=await fetch("/api/gemini/music/start",H({method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({weighted_prompts:I,config:Ze()})})),fe=await j.json().catch(()=>({}));if(!j.
ok)throw new Error(fe.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
f=fe.session_id,P=Ze(),z("\u63A5\u7D9A\u4E2D...","connecting"),yt()}catch(j){z("\u30A8\u30E9\u30FC: "+
j.message,"error"),showToast("Lyria RealTime: "+j.message,"error",!0)}finally{ee=!1,xe()}}a(ce,"star\
tSession");async function ge(I){if(f){ee=!0,xe();try{await Pe("control",{action:I}),I==="PLAY"?z("\u518D\u751F\
\u4E2D...","streaming"):I==="PAUSE"?z("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):I==="STOP"?z("\u505C\u6B62\u4E2D",
"stopped"):I==="RESET_CONTEXT"&&z("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(j){showToast("Lyria RealTime: "+j.message,"error",!0),z("\u30A8\u30E9\u30FC: "+j.
message,"error")}finally{ee=!1,xe()}}}a(ge,"control");async function Ee(){if(!f)return;const I=Ke();
if(!I.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ee=!0;try{await Pe("prompts",{weighted_prompts:I}),z("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
D==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(j){showToast("Lyria RealTime: "+j.message,"error",!0)}finally{ee=!1,xe()}}a(Ee,"app\
lyPrompts");async function vt(){if(!f)return;const I=Ze(),j=P||{},fe=I.bpm!==void 0&&I.bpm!==j.bpm,te=I.
scale!==void 0&&I.scale!==j.scale,be=fe||te;ee=!0;try{await Pe("config",{config:I,reset_context:be}),
P=I,z(be?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",D==="paused"?"paused":"streaming"),showToast(
be?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch($e){showToast("Lyria RealT\
ime: "+$e.message,"error",!0)}finally{ee=!1,xe()}}a(vt,"applyConfig");async function Ut(){if(f){ee=!0,
z("\u4FDD\u5B58\u4E2D...","connecting"),xe();try{const I=await fetch("/api/gemini/music/save",H({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:f,thread_id:currentThreadId||
null})})),j=await I.json().catch(()=>({}));if(!I.ok)throw new Error(j.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
z("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ie(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),j.thread_id&&(currentThreadId=String(j.thread_id),history.pushState({},"","/c/"+j.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(j.thread_id||currentThreadId),an(!0)}catch(I){
z("\u30A8\u30E9\u30FC: "+I.message,"error"),showToast("Lyria RealTime: "+I.message,"error",!0)}finally{
ee=!1,xe()}}}a(Ut,"saveSession");async function Fe(){if(Et(),f)try{await fetch("/api/gemini/music/ca\
ncel",H({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:f})}))}catch{}
f=null,ie(),ht(),z("\u6E96\u5099\u5B8C\u4E86","idle")}a(Fe,"cancelSession");function ot(){const I=M(
"lyria-prompt-rows");I&&(I.innerHTML=""),Ne("",1),P=null,B=0,["lyria-bpm","lyria-guidance","lyria-de\
nsity","lyria-brightness","lyria-temperature"].forEach(te=>{const be=M(te);be&&(be.value=te==="lyria\
-bpm"?"120":te==="lyria-guidance"?"4":te==="lyria-temperature"?"1.1":"0.5")});const j=M("lyria-scale");
j&&(j.value="");const fe=M("lyria-mode");fe&&(fe.value="QUALITY"),["lyria-mute-bass","lyria-mute-dru\
ms","lyria-only-bass-drums"].forEach(te=>{const be=M(te);be&&(be.checked=!1)}),At()}a(ot,"resetContr\
ols");function an(I){Et(),f&&fetch("/api/gemini/music/cancel",H({method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify({session_id:f})})).catch(()=>{}),f=null,k=!1,ie(),ht(),hideModal(
"lyria-studio-modal")}a(an,"closeAndCleanup");function An(I){if(!isLyriaRealtimeModel()){showToast("\
Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const j=M("lyria-studio-modal");if(j&&j.classList.contains("modal-open")&&f){if(I&&
typeof I=="string"){const te=M("lyria-prompt-rows");te&&(te.innerHTML=""),Ne(I,1)}return}if(f&&Fe(),
ot(),I&&typeof I=="string"){const te=M("lyria-prompt-rows");te&&(te.innerHTML=""),Ne(I,1)}f=null,k=!1,
ie(),ht(),z("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}a(An,"open");function di(){
const I=M("lyria-open-studio-btn");I&&I.addEventListener("click",()=>An(""));const j=M("lyria-studio\
-close");j&&j.addEventListener("click",()=>an(!1));const fe=M("lyria-play-btn");fe&&fe.addEventListener(
"click",()=>{if(!f){ce();return}ge("PLAY")});const te=M("lyria-pause-btn");te&&te.addEventListener("\
click",()=>ge("PAUSE"));const be=M("lyria-stop-btn");be&&be.addEventListener("click",()=>ge("STOP"));
const $e=M("lyria-reset-btn");$e&&$e.addEventListener("click",()=>ge("RESET_CONTEXT"));const We=M("l\
yria-add-prompt-btn");We&&We.addEventListener("click",()=>Ne("",1));const je=M("lyria-apply-prompts-\
btn");je&&je.addEventListener("click",Ee);const Xe=M("lyria-apply-config-btn");Xe&&Xe.addEventListener(
"click",vt);const it=M("lyria-save-btn");it&&it.addEventListener("click",Ut),At(),ot(),window.openLyriaStudio=
An}return a(di,"init"),{init:di,open:An}})().init(),(()=>{let c=null,u=null;const m=a(P=>document.getElementById(
P),"$");function f(){return isStsModel()&&voiceStudioUiEnabled!==!1}a(f,"isStudioMode");function v(){
const P=get("model-select")?get("model-select").value:"",M=m("voice-studio-title");M&&(P==="gpt-tran\
scribe"||P==="gpt-live-transcribe"?M.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":
P==="gemini-3.5-live-translate-preview"?M.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
M.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}a(v,"updateTitle");function k(){const P=m("voi\
ce-studio-transcript");P&&(P.innerHTML='<div class="text-[10px] text-gray-500">\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</\
div>')}a(k,"resetTranscript");function _(P,M){if(!M||!String(M).trim())return;const H=m("voice-studi\
o-transcript");if(!H||!window.VoiceStudioOpen)return;const z=P==="user"?"\u3042\u306A\u305F":"AI",ye=P===
"user"?"text-cyan-300":"text-gray-100",V=H.querySelectorAll(".voice-studio-line");let ie=null;for(let Ie=V.
length-1;Ie>=0;Ie--)if(V[Ie].dataset.role===P){ie=V[Ie];break}const xe=`<span class="${ye} font-bold\
">${escapeHtml(z)}:</span> <span class="text-gray-200">${escapeHtml(M)}</span>`;if(ie)ie.innerHTML=xe;else{
const Ie=H.querySelector(".text-gray-500");Ie&&Ie.remove();const Ne=document.createElement("div");Ne.
className="voice-studio-line",Ne.dataset.role=P,Ne.innerHTML=xe,H.appendChild(Ne)}H.scrollTop=H.scrollHeight}
a(_,"log");function C(){const P=m("sts-panel"),M=m("voice-studio-panel-host");P&&M&&P.parentNode!==M&&
(c=P.parentNode,M.appendChild(P));const H=m("file-preview"),z=m("voice-studio-file-host");H&&z&&H.parentNode!==
z&&(u=H.parentNode,z.appendChild(H),z.classList.remove("hidden"))}a(C,"movePanelIntoModal");function A(){
const P=m("sts-panel");P&&c&&P.parentNode!==c&&c.appendChild(P);const M=m("file-preview");M&&u&&M.parentNode!==
u&&u.appendChild(M);const H=m("voice-studio-file-host");H&&H.classList.add("hidden"),c=null,u=null}a(
A,"movePanelBack");function B(){if(!f()){showToast("\u97F3\u58F0\u7CFB\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}C();const P=m("sts-panel");P&&P.classList.remove("hidden"),v(),k(),window.VoiceStudioOpen=
!0,showModal("voice-studio-modal")}a(B,"open");function $(){if(window.VoiceStudioOpen&&(Ae||Ve&&Ve.state===
"recording"||qt.isActive())&&Wn(),window.VoiceStudioOpen=!1,A(),hideModal("voice-studio-modal"),isStsModel()&&
voiceStudioUiEnabled!==!1){const P=m("sts-panel");P&&P.classList.add("hidden")}}a($,"close");function D(){
window.VoiceStudioOpen&&$()}a(D,"closeIfOpen");function ee(){window.VoiceStudioOpen=!1;const P=m("vo\
ice-studio-open-btn");P&&P.addEventListener("click",()=>B());const M=m("voice-studio-close");M&&M.addEventListener(
"click",()=>$()),window.VoiceStudio={open:B,close:$,closeIfOpen:D,log:_,isStudioMode:f}}return a(ee,
"init"),{init:ee,open:B,close:$,closeIfOpen:D,log:_,isStudioMode:f}})().init();let Gt=null;function Vn(){
if(Gt&&(Gt.stop(),Gt=null),en){try{en.pause()}catch{}try{en.src=""}catch{}en=null}}a(Vn,"stopStsPlay\
back");async function hi(c){Vn();const u=new Audio;return u.src=c,u.preload="auto",u.autoplay=!0,u.playsInline=
!0,en=u,await u.play(),new Promise(m=>{u.onended=()=>m("ended"),u.onerror=()=>m("error")})}a(hi,"pla\
yStsAudio");function Wn(){if(qt.isActive()){qt._cancel();return}if(Ae){Ae.stop(),Ae=null,Vn(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Mt();return}Ve&&Ve.state===
"recording"&&(kn=!0,Ve.stop())}a(Wn,"cancelRecording");function Jn(){if(isStsModel())return{audio:!0};
const u=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},m={channelCount:1};return u.echoCancellation&&(m.echoCancellation=!1),u.
noiseSuppression&&(m.noiseSuppression=!1),u.autoGainControl&&(m.autoGainControl=!1),{audio:m}}a(Jn,"\
getMicCaptureConstraints"),get("mic-btn").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\
\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Ae){setStsStatus("Processing...",!0);const c=Ae;Ae=null,c.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const u=await c.
getFinalData();if(isGeminiLiveTranscribeModel()&&(u.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
u.assistant_text=(c.inputTranscript||"").trim(),u.assistant_thought="",!u.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){
const f=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=
String(f.id),history.pushState({},"","/c/"+f.id),get("welcome-screen").classList.add("hidden")}u.thread_id=
currentThreadId,u.model=get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(u)}),setStsStatus("Saved",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(u){console.error(
"Failed to save Gemini Live session:",u),setStsStatus("Error saving",!1)}return}if(qt.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
qt.stop();return}if(Ve&&Ve.state==="recording"){Ve.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||Lt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const m=new Audio;
m.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",m.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const f={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(f.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const P=get("sts-custom\
-vocab").value.split(/[,、\n]/).map(M=>M.trim()).filter(Boolean);P.length&&(f.custom_vocabulary=P.slice(
0,1e3))}}else f.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(f.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",f.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(f.target_lang=get("sts-target-lang").value);const v=await apiFetch("/api/gemini\
/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});if(!v.
ok)throw new Error("Failed to get session token");const{token:k,url:_}=await v.json(),C=get("model-s\
elect").value,A=get("sts-voice")?get("sts-voice").value:"Kore",B=get("sts-thinking-level")?get("sts-\
thinking-level").value:"minimal",$=get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1;
if(Ae=new Sn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Ae.rtPlayer=new nn),isGeminiLiveTranscribeModel()){
const P=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",M={languageCodes:[]};
if((P==="SMART"||P==="VERBATIM")&&(M.mode=P),get("sts-custom-vocab")){const H=get("sts-custom-vocab").
value.split(/[,、\n]/).map(z=>z.trim()).filter(Boolean);H.length&&(M.customVocabulary=H.slice(0,1e3))}
await Ae.start(k,_,C,{transcriptionConfig:M})}else if(isGeminiLiveTranslateModel()){const P=get("sts\
-target-lang")?get("sts-target-lang").value:"ja";await Ae.start(k,_,C,{translationConfig:{targetLanguageCode:P,
echoTargetLanguage:!0}})}else{const P={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:A}}}};
isGeminiLiveExtendedThinkingModel()&&(P.thinkingConfig={thinkingLevel:B,includeThoughts:$}),await Ae.
start(k,_,C,P)}Ve=Ae.backupRecorder,Ve.onstop=()=>{Ae&&get("mic-btn").click()};let D=!0,ee="live-sts\
-"+Date.now();Ae.onMessage=P=>{if(P.serverContent){if(isGeminiLiveTranscribeModel()){const M=Ae.interimInputTranscript,
H=Ae.inputTranscript,z=H+(M&&!H.endsWith(M)?(H?`
`:"")+M:""),ye=get("chat-messages");let V=document.getElementById(ee);V||(V=document.createElement("\
div"),V.id=ee,V.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg\
 border border-slate-700/50",V.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ye.appendChild(V),ye.scrollTop=ye.scrollHeight);const ie=V.
querySelector(".message-content");ie.innerText=z||"\u8074\u304D\u53D6\u308A\u4E2D...",ye.scrollTop=ye.
scrollHeight,window.VoiceStudio&&H&&window.VoiceStudio.log("user",H);return}if(P.serverContent.modelTurn){
D&&(setStsStatus("Gemini is speaking...",!1),D=!1);const M=get("chat-messages");let H=document.getElementById(
ee);H||(H=document.createElement("div"),H.id=ee,H.className="flex flex-col gap-2 mb-4 assistant-mess\
age bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",H.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,M.appendChild(H),M.scrollTop=M.scrollHeight);const z=H.
querySelector(".thought-container"),ye=H.querySelector(".message-content");Ae.assistantThought&&(z.classList.
remove("hidden"),z.innerText=Ae.assistantThought),ye.innerText=Ae.assistantText,M.scrollTop=M.scrollHeight,
window.VoiceStudio&&(Ae.inputTranscript&&window.VoiceStudio.log("user",Ae.inputTranscript),Ae.assistantText&&
window.VoiceStudio.log("assistant",Ae.assistantText))}}},setStsStatus("Listening...",!0),get("mic-bt\
n").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),Un(Ae.
stream),zn(Ae.stream);return}catch(m){showToast("Gemini Live connection failed: "+m.message,"error",
!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){await qt.start();return}isStsModel()||
(Gn(),Lt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));const c=await navigator.mediaDevices.
getUserMedia(Jn());Ve=new MediaRecorder(c),cn=[],kn=!1;const u=isStsModel();Ve.ondataavailable=m=>cn.
push(m.data),Ve.onstop=async()=>{if(kn){cn=[],get("file-preview").classList.add("hidden"),c.getTracks().
forEach(_=>_.stop()),hn(),Mt(),u||(Lt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),fn=setTimeout(()=>Lt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const m=new Blob(cn,{type:"audio/web\
m"}),f=new File([m],"recording.webm",{type:"audio/webm"}),v=new FormData;v.append("file",f),get("fil\
e-preview").classList.remove("hidden");const k=u;get("file-name").innerText=k?"Processing voice...":
"Transcribing...";try{if(k){if(!currentThreadId){const z=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=z.id!==null&&z.id!==void 0?String(z.id):z.id,setTemporaryChatUiState(!!(z&&z.
is_temporary)),setCurrentChatHeaderTitle(z&&z.title),applyTemporaryChatRuntimeMeta(z||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+z.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),v.append("model",get(
"model-select").value),v.append("thread_id",currentThreadId),get("sts-voice")&&v.append("sts_voice",
get("sts-voice").value||""),get("sts-speed")&&v.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&v.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&v.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&v.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&v.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const _=await apiFetch(
"/sts",{method:"POST",body:v});if(!_.ok){const H=await _.json().catch(()=>({}));throw new Error(H.error||
"Speech-to-speech failed")}const C=_.body.getReader(),A=new TextDecoder;let B="",$=null,D=null;stsOpt(
"sts-auto-play")&&(D=new nn,Gt=D),setStsStatus(isTranscriptionModel()?"Transcribing...":"Processing \
audio...",!0);let ee=!0,P="",M="";for(;;){const{done:H,value:z}=await C.read();if(H)break;B+=A.decode(
z,{stream:!0});const ye=B.split(`
`);B=ye.pop();for(const V of ye){if(!V.trim())continue;const ie=JSON.parse(V);if(ie.error)throw new Error(
ie.error);ie.audio_delta&&D&&(ee&&(setStsStatus("Playing response...",!1),ee=!1),await D.addChunk(ie.
audio_delta)),ie.input_delta&&(P+=ie.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",P)),
ie.transcript_delta&&(M+=ie.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assistant",
M)),ie.final&&($=ie)}}window.VoiceStudio&&!P.trim()&&window.VoiceStudio.log("user","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),
$&&($.audio_url||$.transcription_only)&&(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus(
"Listening...",!0),get("mic-btn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{
const _=get("set-mic-transcribe-mode");if(!!(_&&_.value==="llm")&&!supportsAudioInputModel()){showToast(
"\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306FLLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}v.append("llm_model",get("model-select")&&get("model-select").value||"");const B=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:v})).json();if(B.transcript){const $=get("prompt-inp\
ut");$.value+=($.value?" ":"")+B.transcript,$.style.height="auto",$.style.height=$.scrollHeight+"px"}else
showToast(B.error||"Transcription failed","error",!0)}}catch(_){showToast("Audio processing error: "+
_.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),c.getTracks().forEach(_=>_.
stop()),hn(),Mt(),k||Lt("","hidden"),k&&setStsStatus("Tap to speak",!1)}},Ve.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Lt("\u9332\u97F3\u4E2D\u2026","recording"),Un(c)),zn(c),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Mt(),isStsModel()||Lt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const sn=a((c,u)=>{if(!c)return;const m=c.querySelector("span");m?m.textContent=u:c.textContent=
u},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const c=lib.
selected.size,u=get("lib-del-btn"),m=get("lib-download-btn"),f=get("lib-attach-btn"),v=get("lib-rena\
me-btn"),k=get("lib-usage-btn");if(u&&(u.disabled=c===0,sn(u,c?`\u524A\u9664 (${c})`:"\u524A\u9664")),
m&&(m.disabled=c===0,sn(m,c?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${c})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
f&&(f.disabled=c===0,sn(f,c?`\u6DFB\u4ED8 (${c})`:"\u6DFB\u4ED8")),v&&(v.disabled=c!==1,sn(v,"\u540D\u524D\u5909\u66F4")),
k&&(k.disabled=c!==1,sn(k,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const _=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",_&&c>0)}};function Kn(c){lib.
attachMode=!!c}a(Kn,"setLibAttachMode");const Xn=a((c=!1)=>{Kn(c),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(c=!1)=>{hideModal("lib-modal"),!c&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>Xn(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>Xn(!0)),
get("lib-search")){let c=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),c&&clearTimeout(c),c=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const c=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=c,get("lib-sort").onchange=()=>{const u=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,u),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const ci=a(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",get("gem-default-model").value="",get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-con\
tainer").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"open\
GemModal");window.closeGemModal=(c=!1)=>{hideModal("gem-modal"),!c&&location.pathname==="/gem"&&history.
back()},get("add-gem-btn").onclick=()=>ci(),get("save-gem-btn").onclick=async()=>{const c=get("gem-n\
ame").value,u=get("gem-desc").value,m=get("gem-inst").value,f=collectGemFixedPrompts();if(c&&m){const v=editingGemUuid?
"PUT":"POST",k=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
k,{method:v,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:c,description:u,instruction:m,
fixed_prompts:f,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=c,activeGem.instruction=
m,activeGem.fixed_prompts=f,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(c){if(c.target.closest(".edit-btn")){const m=c.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(m)}if(c.target.closest(".code-toggle")){
const u=c.target.closest(".code-toggle"),m=u.closest(".code-wrapper");if(!m)return;const f=m.classList.
toggle("collapsed");m.setAttribute("data-collapsed",f?"true":"false"),u.setAttribute("aria-expanded",
f?"false":"true"),u.innerHTML=f?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',u.title=f?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",u.setAttribute("aria-label",f?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(c.target.closest(".download-btn")){const u=c.target.closest(".d\
ownload-btn"),m=u.getAttribute("data-code"),f=(u.getAttribute("data-lang")||"txt").toLowerCase();if(m)
try{const v=decodeURIComponent(m),k=new Blob([v],{type:"text/plain"}),_=URL.createObjectURL(k),C=document.
createElement("a");C.href=_;let B={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[f]||f;(f.length>8||/[^a-z0-9]/.test(f))&&(B="txt");let $=`code.${B}`;f==="dockerfile"&&($="Docke\
rfile"),f==="makefile"&&($="Makefile"),C.download=$,document.body.appendChild(C),C.click(),document.
body.removeChild(C),URL.revokeObjectURL(_)}catch(v){console.error("Download failed",v)}}if(c.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(c.target.closest(".coding-target-btn")),
c.target.closest(".copy-btn")){const u=c.target.closest(".copy-btn"),m=u.getAttribute("data-code");m&&
window.copyCode(u,m)}if(c.target.closest(".html-preview-btn")){const m=c.target.closest(".html-previ\
ew-btn").getAttribute("data-code");m&&openHtmlCodePreview(m)}if(c.target.closest(".canvas-preview-bt\
n")){const u=c.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(u)}}),document.querySelectorAll(
".modal-overlay").forEach(c=>{c.addEventListener("click",u=>{u.target===c&&Dn(c.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){const e=get(
"file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-progress-bar"),
s=get("file-preview-thumbs"),o=get("upload-modal-status-text"),r=get("upload-modal-total-progress"),
l=get("upload-modal-total-progress-bar");if(!e||!t)return;if(s){const T=document.querySelectorAll("#\
upload-list .upload-row");s.innerHTML="",T.forEach((E,F)=>{const J=E.getAttribute("data-local-url"),
X=E.getAttribute("data-filename"),Te=E.querySelector("img.upload-preview")!==null;let O;if(Te){let q=J;
if(!q&&X){const Y=X.replace(/^\d+\//,"");q=buildAttachmentPreviewUrl(Y)}q&&(O=document.createElement(
"img"),O.src=q,O.className="thumb-item shadow-sm",O.dataset.viewerSrc=q,O.dataset.viewerFilename=X||
q.split("/").pop(),O.onclick=function(Y){Y.preventDefault(),openImageViewer(this.dataset.viewerSrc,"\
.thumb-item")},O.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}O||(O=d("FILE")),
O.style.animationDelay=`${F*32}ms`,s.appendChild(O)}),T.length>0?s.classList.remove("hidden"):s.classList.
add("hidden")}function d(T){const E=document.createElement("div");return E.className="thumb-item bg-\
gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold",E.innerText=
T,E}a(d,"createFileThumb");const p=collectImageUrlsForSend(),h=uploadProgressState.total,g=uploadProgressState.
completed,y=uploadProgressState.active;h===0&&(e.classList.add("hidden"),n&&n.classList.add("hidden"),
r&&r.classList.add("hidden"),s&&s.classList.add("hidden"));const b=get("send-btn"),w=get("mic-btn"),
x=get("mask-btn"),S=isStopMode;if(y>0?(b&&(b.disabled=!0),w&&(w.disabled=!0),x&&(x.disabled=!0)):S||
(b&&(b.disabled=!1),w&&(w.disabled=!1),x&&(x.disabled=!1)),y>0){const T=`Preparing... (${g}/${h})`;e.
classList.remove("hidden"),t.innerText=T,o&&(o.innerText=`(${g}/${h})`);let E=g*100,F=0;for(let Te in uploadProgressState.
perFilePct)E+=uploadProgressState.perFilePct[Te],F++;const J=h>0?E/(h*100)*100:0,X=`${Math.min(100,J)}\
%`;n&&i&&(n.classList.remove("hidden"),i.style.width=X),r&&l&&(r.classList.remove("hidden"),l.style.
width=X)}else o&&(o.innerText=""),r&&r.classList.add("hidden"),p.length>0?(e.classList.remove("hidde\
n"),t.innerText=`${p.length} files ready`,n&&n.classList.add("hidden")):(e.classList.add("hidden"),t.
innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}a(updateFilePreview,"update\
FilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get("mask-name");!e||!t||(currentMaskImage?
(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.split("/").pop()}`):(e.classList.
add("hidden"),t.innerText=""))}a(updateMaskPreview,"updateMaskPreview");const markerToolHints={draw:"\
\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],s=t[3];return`#${n}${n}${i}${i}${s}${s}`}
return"#facc15"}a(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const t=normalizeMarkerHexColor(
e);return{r:parseInt(t.slice(1,3),16),g:parseInt(t.slice(3,5),16),b:parseInt(t.slice(5,7),16)}}a(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,t=60){const n=Number(e),i=Number.isFinite(n)?n:t;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,i))}a(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const t=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(t)?String(t):String(t).replace(/\.0$/,"")}a(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${t})`}a(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),s=get("ma\
rker-color-picker");s&&s.value!==e&&(s.value=e);const o=get("marker-opacity");o&&o.value!==i&&(o.value=
i);const r=get("marker-opacity-number");r&&r.value!==i&&(r.value=i);const l=get("marker-opacity-valu\
e");l&&(l.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(p=>{const h=normalizeMarkerHexColor(p.getAttribute("data-marker-color"));p.classList.
toggle("active",h===e)})}a(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}a(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}a(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const s=get("marker-tool-hi\
nt");s&&(s.textContent=markerToolHints[e]||"");const o=get("marker-crop-reset");o&&o.classList.toggle(
"hidden",e!=="crop");const r=get("marker-canvas");r&&(r.style.pointerEvents=e==="crop"?"none":"auto");
const l=get("marker-crop-canvas");l&&(l.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.
cropRect||markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}
a(setMarkerMode,"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}
a(clearCropRect,"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)
return;const t=Math.max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,
y:0,w:t,h:n})}a(resetCropRectToFull,"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.
scale<=markerView.minScale+1e-4){markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marke\
r-stage"),t=get("marker-viewport");if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(
1,e.clientHeight||0),s=Math.max(1,t.offsetWidth||t.clientWidth||0),o=Math.max(1,t.offsetHeight||t.clientHeight||
0);if(n<=1||i<=1||s<=1||o<=1)return;const r=(n-s)/2,l=(i-o)/2,d=s*markerView.scale,p=o*markerView.scale,
h=Math.min(n*.45,Math.max(24,n*.12)),g=Math.min(i*.45,Math.max(24,i*.12)),y=h-r-d,b=n-h-r,w=g-l-p,x=i-
g-l,S=a((T,E,F)=>Number.isFinite(T)?E>F?(E+F)/2:Math.min(F,Math.max(E,T)):0,"clampOffset");markerView.
offsetX=S(markerView.offsetX,y,b),markerView.offsetY=S(markerView.offsetY,w,x)}a(clampMarkerViewOffset,
"clampMarkerViewOffset");function applyMarkerTransform(){const e=get("marker-viewport");e&&(clampMarkerViewOffset(),
e.style.transform=`translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale}\
)`)}a(applyMarkerTransform,"applyMarkerTransform");function resetMarkerTransform(){markerView.scale=
1,markerView.offsetX=0,markerView.offsetY=0,applyMarkerTransform()}a(resetMarkerTransform,"resetMark\
erTransform");function getRowMarkerKey(e){return e&&(e.dataset.uploadId||e.getAttribute("data-filena\
me"))||null}a(getRowMarkerKey,"getRowMarkerKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(
e);n&&(t?markerAppliedUploads.add(n):markerAppliedUploads.delete(n));const i=e?e.querySelector(".upl\
oad-marker-tag"):null;i&&i.classList.toggle("hidden",!t)}a(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){
return markerAppliedUploads.size>0}a(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){
const t=String(e||"").trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t===
"uploaded"?"upload":"unknown"}a(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let t=String(e).replace(/\u0000/g,"");return t=t.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),t=t.trim(),!t||(t=t.split("/").pop().split("\\").pop().trim(),t=t.replace(/\s{2,}/g,
" "),t=t.replace(/[<>:"/\\|?*]+/g,"_"),!t||t==="."||t==="..")?"":(t.length>180&&(t=t.slice(0,180).trim()),
t)}a(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const t=normalizeAttachmentPath(e);return t?t.split("/").pop()||t:""}a(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,t){const n=normalizeAttachmentPath(
e);if(!n)return;const i=normalizeAttachmentDisplayName(t)||defaultAttachmentDisplayName(n);i&&attachmentNameByPath.
set(n,i)}a(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const t=normalizeAttachmentPath(e);if(!t)return"";const n=normalizeAttachmentDisplayName(attachmentNameByPath.
get(t));return n||defaultAttachmentDisplayName(t)}a(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,t){if(!e)return;const n=normalizeAttachmentDisplayName(t)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=n;const i=e.querySelector(".truncate");
i&&(i.textContent=n);const s=e.getAttribute("data-filename");s&&setAttachmentNameForPath(s,n)}a(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}a(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}a(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
a(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const s=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,s),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}a(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const s=e.querySelector(".truncate"),o=normalizeAttachmentDisplayName(
s?s.textContent:"");return o||getAttachmentNameForPath(t)}a(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,t){const n=normalizeAttachmentPath(e);if(!n)return;const i=normalizeAttachmentSource(
t);i!=="unknown"&&attachmentSourceByPath.set(n,i)}a(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const t=normalizeAttachmentPath(e);return t?normalizeAttachmentSource(
attachmentSourceByPath.get(t)):"unknown"}a(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,t){if(!e)return;const n=normalizeAttachmentSource(t);e.dataset.fileSource=
n;const i=e.getAttribute("data-filename");i&&setAttachmentSourceForPath(i,n)}a(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const t=normalizeAttachmentSource(
e.dataset.fileSource);if(t!=="unknown")return t;const n=e.getAttribute("data-filename");return getAttachmentSourceForPath(
n)}a(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const t=normalizeAttachmentSource(e.dataset.originalSource);if(t!=="unknown")return t;
const n=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(n)}a(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,t,n){const i=document.createElement(
"canvas");i.width=t,i.height=n;const s=i.getContext("2d");s?(s.drawImage(e,0,0,t,n),markerState.baseImageData=
s.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}a(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=a((r,l,d=null,p=!1)=>{if(!r)return;const h=Math.max(0,r.x),g=Math.max(0,r.y),y=Math.
max(1,r.w),b=Math.max(1,r.h);d&&(t.fillStyle=d,t.fillRect(h,g,y,b)),t.save(),p&&t.setLineDash([6,4]),
t.strokeStyle=l,t.lineWidth=2,t.strokeRect(h+.5,g+.5,Math.max(1,y-1),Math.max(1,b-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,s=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!s)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const r=Math.max(0,i.x),l=Math.max(0,i.y),d=Math.max(1,i.w),p=Math.max(1,i.h);t.clearRect(r,l,d,p),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(r=>n(r,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}a(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
a(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=a((s,o,r)=>{const l=normalizeAttachmentPath(s);if(!l)return;const d=normalizeAttachmentSource(
o),p=normalizeAttachmentDisplayName(r)||getAttachmentNameForPath(l),h=t.get(l);if(h===void 0){const b=e.
length;t.set(l,b),e.push({path:l,source:d,name:p});return}const g=e[h];if(!g)return;const y=normalizeAttachmentSource(
g.source);(y==="unknown"&&d!=="unknown"||y==="library"&&d==="upload")&&(g.source=d),!normalizeAttachmentDisplayName(
g.name)&&p&&(g.name=p)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(s=>{const o=s.getAttribute("data-filename");n(o,getRowAttachmentSource(s),getRowAttachmentName(
s));const r=s.getAttribute("data-original-filename");s.dataset.attachOriginal==="1"&&n(r,getRowOriginalAttachmentSource(
s),getAttachmentNameForPath(r))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
s=>{n(s,getAttachmentSourceForPath(s),getAttachmentNameForPath(s))}),e}a(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}a(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const o=[];currentImageUrls.forEach(r=>{const l=normalizeAttachmentPath(r);if(!l)return;const d=isAudioPath(
l),p=isVideoPath(l);if(d&&!t.audio||p&&!t.video){d&&(n+=1),p&&(i+=1);return}o.push(l)}),o.length!==currentImageUrls.
length&&(currentImageUrls=o)}const s=get("upload-list");if(s&&(s.querySelectorAll("[data-filename]").
forEach(o=>{const r=o.getAttribute("data-filename");r&&!currentImageUrls.includes(r)&&(isAudioPath(r)||
isVideoPath(r))&&(setRowMarkerState(o,!1),o.remove())}),s.children.length===0&&(s.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const o=[];n&&o.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&o.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${o.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}a(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const t=e.getAttribute("data-local-url");if(t)return t;const n=e.getAttribute("data-f\
ilename");return n?buildFileUrl(n):""}a(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const t=normalizeAttachmentPath(e);return t?FILE_BASE_URL+t:""}a(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const t=normalizeAttachmentPath(e);return t?isImagePath(t)?FILE_THUMB_BASE_URL+t:FILE_BASE_URL+t:""}
a(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e){
const t=getRowImageSource(e);if(!t){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e;const n=e?e.querySelector(".truncate"):null;markerState.filename=
n?n.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=[],markerState.naturalWidth=
0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=[],markerState.mosaicPreviewRect=
null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode("draw");const i=get("m\
arker-attach-original");i&&(i.checked=e.dataset.attachOriginal==="1");const s=get("marker-image"),o=get(
"marker-canvas"),r=get("marker-crop-canvas");if(o){const l=o.getContext("2d");l&&l.clearRect(0,0,o.width,
o.height)}if(r){const l=r.getContext("2d");l&&l.clearRect(0,0,r.width,r.height)}resetMarkerTransform(),
showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState({modal:"marker"},"","\
/edit-image"),s&&(s.onload=()=>{if(!get("marker-stage")||!o)return;const d=Math.max(1,Math.floor(s.clientWidth)),
p=Math.max(1,Math.floor(s.clientHeight));o.width=d,o.height=p,o.style.width=`${d}px`,o.style.height=
`${p}px`,o.style.left="0px",o.style.top="0px",r&&(r.width=d,r.height=p,r.style.width=`${d}px`,r.style.
height=`${p}px`,r.style.left="0px",r.style.top="0px"),markerState.naturalWidth=s.naturalWidth||d,markerState.
naturalHeight=s.naturalHeight||p;const h=o.getContext("2d");h&&h.clearRect(0,0,o.width,o.height),prepareMarkerBaseCanvas(
s,d,p),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},s.src=t)}a(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
a(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(r=>{const l=r&&r.rowObj?r.rowObj.row:null,d=l?l.getAttribute("data-local-url"):null;d&&URL.revokeObjectURL(
d)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.clear();const e=get(
"file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");t&&(t.innerHTML="",t.
classList.add("hidden")),updateFilePreview(),updateMaskPreview();const n=get("upload-list");n&&(n.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>');
const i=get("file-input");i&&(i.value="");const s=get("photo-input");s&&(s.value="");const o=get("ma\
sk-input");o&&(o.value="")}a(resetUploadState,"resetUploadState");async function uploadMaskFile(e){if(!e)
return;const t=new FormData;t.append("file",e);try{const n=await fetch(CHAT_CONFIG.urls.upload,{method:"\
POST",body:t}),i=await n.json();n.ok&&i.filename?(currentMaskImage=i.filename,updateMaskPreview()):showToast(
i.error||"Mask upload failed","error",!0)}catch{showToast("Mask upload failed","error",!0)}}a(uploadMaskFile,
"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){const n=get("camera-status");n&&(n.textContent=
e||"",n.classList.toggle("text-red-300",!!t),n.classList.toggle("text-gray-400",!t))}a(setCameraCaptureStatus,
"setCameraCaptureStatus");function updateCameraCapturePendingUi(){const e=cameraCapturePendingFiles.
length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,t.textContent=e?`\u6DFB\u4ED8 (${e}\
)`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=e===0||cameraCaptureBusy);const i=get(
"camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.forEach((s,o)=>{const r=document.
createElement("div");r.className="relative rounded overflow-hidden border border-gray-700 bg-black a\
spect-square",r.innerHTML=`
                        <img src="${s}" alt="capture ${o+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${o+1}</div>
                    `,i.appendChild(r)}),i.classList.toggle("hidden",e===0))}a(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const t=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(t)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}a(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(i=>{try{i.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const t=get("camera-capture-btn");t&&(t.disabled=!0);const n=get(
"camera-switch-btn");n&&(n.disabled=!0)}a(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const t=get("camera-video");if(!t)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const n=get(
"camera-switch-btn");n&&(n.disabled=!0);const i=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let s=null;for(const o of i)
try{const r=await navigator.mediaDevices.getUserMedia(o);cameraCaptureStream=r,t.srcObject=r;try{await t.
play()}catch{}const l=r.getVideoTracks&&r.getVideoTracks()[0],d=l&&l.getSettings?l.getSettings():{},
p=String(d.facingMode||"").toLowerCase();p==="user"||p==="environment"?cameraCaptureFacingMode=p:cameraCaptureFacingMode=
e;const h=get("camera-capture-btn");return h&&(h.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),r}catch(r){s=r}throw s||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
a(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const t=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(t,!0),showToast(t,"error",!0);const n=get("camera-capture-btn");n&&(n.disabled=
!0);const i=get("camera-attach-btn");i&&(i.disabled=!0)}}a(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const t=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!t&&location.pathname==="/camera"&&history.back()}a(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const t=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=t;try{await startCameraCaptureStream(t)}catch(n){const i=n&&
n.message?n.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
i,!0),showToast(i,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}a(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,t=a(s=>String(s).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}a(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const s=get("camera-attach-btn");s&&(s.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const o=t.getContext("2d");if(!o)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");o.drawImage(e,0,0,t.width,t.height);
const r=await new Promise((d,p)=>{t.toBlob(h=>{h?d(h):p(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),l=new File([r],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(l),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(r)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(o){
const r=o&&o.message?o.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
r,!0),showToast(r,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}a(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),s=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),s&&(s.disabled=!0);const o=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${o.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{await handleFiles(o,{openModal:!1}),showToast(`${o.length}\u679A\u306E\
\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(r){const l=r&&r.message?r.message:
"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";showToast(l,"\
error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.classList.contains(
"hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}a(attachCameraCapturedFiles,
"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay=="function"&&window.
hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.pathname!=="/upload"&&
history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-info");if(e){const n=(get(
"model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v4.1-flash"||n==="deepse\
ek-v4-flash-vision-exp",s=n.includes("deepseek")&&!i;e.classList.toggle("hidden",!s)}_syncVisionModelDisplay()}
a(openUploadModal,"openUploadModal");function _syncVisionModelDisplay(){const e=get("vision-model-di\
splay");if(!e)return;const t=currentVisionModel;if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(
s=>{s.id===t&&(n=s.name)})),e.textContent=n}else e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}
a(_syncVisionModelDisplay,"_syncVisionModelDisplay");function _openVisionModelSelector(){window._visionPickerActive=
!0,openModelModal(),setTimeout(()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},
50)}a(_openVisionModelSelector,"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.
hideDropOverlay=="function"&&window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname===
"/upload"&&history.back()}a(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){
const e=get("upload-list");if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(
n=>{const i=n.getAttribute("data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(
n,{source:getAttachmentSourceForPath(n),displayName:getAttachmentNameForPath(n)})}),e.children.length===
0&&(e.innerHTML='<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
a(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}a(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const s=e.split("/").pop()||e,o=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||s,r=(s.split(".").pop()||"").toLowerCase(),l=["png","jpg","jpeg","webp",
"gif"].includes(r),d=buildFileUrl(e),p=l?buildAttachmentPreviewUrl(e):d,h=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,g=document.createElement("div");g.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",g.dataset.uploadId=h,g.setAttribute("data-filename",e),g.dataset.fileSource=
n,g.dataset.displayName=o,g.dataset.defaultDisplayName=o,g.dataset.sendNameCustomized="";const y=escapeHtml(
o),b=l&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",w=l?`<img src="${p}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${y}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';g.innerHTML=`
                <div class="flex items-center gap-3">
                    ${w}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${y}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${b}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;const x=g.querySelector(".upload-preview");x&&(x.onclick=()=>openFileViewer(d,getRowAttachmentName(
g)||o));const S=g.querySelector(".upload-send-name");S&&(S.onclick=()=>promptRowAttachmentName(g));const T=g.
querySelector(".upload-remove");T&&(T.onclick=()=>{uploadCancelTokens.add(h),browserFastLocalFiles.delete(
h),decrementUploadTotal(h);const F=g.getAttribute("data-filename");F&&(currentImageUrls=currentImageUrls.
filter(J=>J!==F)),setRowMarkerState(g,!1),g.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const E=g.querySelector(".upload-marker");return E&&(E.onclick=()=>openMarkerModalForRow(g)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,o),i.prepend(g),{row:g,bar:g.querySelector(".upload-progress > div"),
status:g.querySelector(".upload-status"),uploadId:h}}a(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const s=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=s,i.dataset.defaultDisplayName=s,i.dataset.sendNameCustomized="";const o=escapeHtml(
s),r=e&&e.type&&e.type.startsWith("image/");let l='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const d=r&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let p="";r?(p=URL.createObjectURL(e),l=`<img src="${p}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${o}">`):(p=URL.createObjectURL(e),l=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${l}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${o}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">\u5F85\u6A5F\u4E2D</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${d}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `,p&&i.setAttribute("data-local-url",p);const h=i.querySelector(".upload-preview");h&&(h.
onclick=()=>{const w=i.getAttribute("data-filename"),x=w?buildFileUrl(w):i.getAttribute("data-local-\
url"),S=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||w||"";openFileViewer(x,S)});const g=i.
querySelector(".upload-remove");g&&(g.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const w=i.getAttribute("data-local-url");w&&URL.revokeObjectURL(w);const x=i.
getAttribute("data-filename");x&&(currentImageUrls=currentImageUrls.filter(S=>S!==x)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const y=i.querySelector(".upload-marker");y&&(y.onclick=()=>openMarkerModalForRow(i));const b=i.querySelector(
".upload-send-name");return b&&(b.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}a(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(!e)
return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);try{const i=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),s=await i.json();if(!i.ok){const g=s&&s.error?s.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast(g,"error",!0),!1}const o=s.upload_id,r=s.chunk_size||10*1024*1024,l=Math.ceil(
e.size/r);for(let g=0;g<l;g++){const y=g*r,b=Math.min(e.size,y+r),w=e.slice(y,b);if(!await new Promise(
S=>{const T=new XMLHttpRequest;T.open("POST","/upload/chunk",!0),T.setRequestHeader("X-CSRF-Token",csrfToken),
T.upload.onprogress=F=>{if(F.lengthComputable&&t&&t.bar){const J=y+F.loaded,X=Math.min(100,Math.floor(
J/e.size*100));t.bar.style.width=`${X}%`,t.status&&(t.status.textContent=`${X}%`),t.uploadId&&updateGlobalUploadProgress(
t.uploadId,X)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},T.onload=()=>{T.status>=
200&&T.status<300?S(!0):S(!1)},T.onerror=()=>S(!1);const E=new FormData;E.append("upload_id",o),E.append(
"index",String(g)),E.append("total",String(l)),E.append("chunk",w,e.name),T.send(E)}))return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:o})}),p=await d.json();if(d.ok&&p&&p.filename){if(t&&t.row&&t.uploadId&&
uploadCancelTokens.has(t.uploadId))return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const b=t.
row.getAttribute("data-local-url");b&&URL.revokeObjectURL(b),t.row.removeAttribute("data-local-url");
const w=t.row.querySelector("img.upload-preview");if(w){const x=p.filename.replace(/^\d+\//,"");w.src=
buildAttachmentPreviewUrl(x)}}const g=normalizeAttachmentPath(p.filename);if(g&&currentImageUrls.push(
g),t&&t.row&&(t.row.setAttribute("data-filename",g||p.filename),setRowAttachmentSource(t.row,"upload"),
g)){const b=isRowAttachmentNameCustomized(t.row),w=defaultAttachmentDisplayName(g),x=b&&normalizeAttachmentDisplayName(
t.row.dataset.displayName)||w;t.row.dataset.defaultDisplayName=w,setRowAttachmentName(t.row,x)}return g&&
setAttachmentSourceForPath(g,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(p.filenames)&&p.filenames.length?p.filenames:[p.filename]).forEach(b=>addLibraryFileFromPath(
b)),!0}const h=p&&p.error?p.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),!1}catch{return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}a(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
i=!0);const s=a(()=>{i&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),i=!1)},
"finishUploadOp"),o=new XMLHttpRequest;o.open("POST",CHAT_CONFIG.urls.upload,!0),o.setRequestHeader(
"X-CSRF-Token",csrfToken),o.upload.onprogress=l=>{if(l.lengthComputable&&t&&t.bar){const d=Math.min(
100,Math.floor(l.loaded/l.total*100));t.bar.style.width=`${d}%`,t.status&&(t.status.textContent=`${d}\
%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,d)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},o.onload=()=>{let l={};try{l=JSON.parse(o.responseText||"{}")}catch{}if(o.status>=
200&&o.status<300&&l&&l.filename){if(t&&t.row&&t.uploadId&&uploadCancelTokens.has(t.uploadId)){t.row&&
t.row.parentNode&&t.row.remove(),s(),n(!1);return}if(t&&t.row){const h=t.row.getAttribute("data-loca\
l-url");h&&URL.revokeObjectURL(h),t.row.removeAttribute("data-local-url");const g=t.row.querySelector(
"img.upload-preview");if(g){const y=l.filename.replace(/^\d+\//,"");g.src=buildAttachmentPreviewUrl(
y)}}const d=normalizeAttachmentPath(l.filename);if(d&&currentImageUrls.push(d),t&&t.row&&(t.row.setAttribute(
"data-filename",d||l.filename),setRowAttachmentSource(t.row,"upload"),d)){const h=isRowAttachmentNameCustomized(
t.row),g=defaultAttachmentDisplayName(d),y=h&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
g;t.row.dataset.defaultDisplayName=g,setRowAttachmentName(t.row,y)}d&&setAttachmentSourceForPath(d,"\
upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(l.filenames)&&
l.filenames.length?l.filenames:[l.filename]).forEach(h=>addLibraryFileFromPath(h)),s(),n(!0)}else{const d=l&&
l.error?l.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(d,"error",!0),s(),n(!1)}},o.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),s(),n(!1)};const r=new FormData;r.append("file",e),o.send(r)})}a(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}a(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}a(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(p=>{n+=p.length});const i=new Float32Array(n);
let s=0;e.forEach(p=>{i.set(p,s),s+=p.length});const o=new ArrayBuffer(44+i.length*2),r=new DataView(
o),l=a((p,h)=>{for(let g=0;g<h.length;g++)r.setUint8(p+g,h.charCodeAt(g))},"writeString");l(0,"RIFF"),
r.setUint32(4,36+i.length*2,!0),l(8,"WAVE"),l(12,"fmt "),r.setUint32(16,16,!0),r.setUint16(20,1,!0),
r.setUint16(22,1,!0),r.setUint32(24,t,!0),r.setUint32(28,t*2,!0),r.setUint16(32,2,!0),r.setUint16(34,
16,!0),l(36,"data"),r.setUint32(40,i.length*2,!0);let d=44;for(let p=0;p<i.length;p++){const h=Math.
max(-1,Math.min(1,i[p]));r.setInt16(d,h<0?h*32768:h*32767,!0),d+=2}return new Blob([r],{type:"audio/\
wav"})}a(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}a(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),s=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=s),e.row.dataset.displayName=
s,i||(e.row.dataset.defaultDisplayName=s);const o=e.row.getAttribute("data-local-url");o&&URL.revokeObjectURL(
o);const r=URL.createObjectURL(t);e.row.setAttribute("data-local-url",r);const l=t.type&&t.type.startsWith(
"image/"),d=escapeHtml(s),p=l?`<img src="${r}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${d}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',h=e.row.querySelector(".upload-preview");h&&(h.outerHTML=p);const g=e.row.querySelector(
".upload-preview");g&&(g.onclick=()=>{const b=e.row.getAttribute("data-filename"),w=b?buildFileUrl(b):
e.row.getAttribute("data-local-url");openFileViewer(w,getRowAttachmentName(e.row)||s||b||"")});const y=e.
row.querySelector(".upload-marker");y&&y.classList.toggle("hidden",!l),l||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}a(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const t=e.getContext("2d");if(!t)return;const n=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[];markerState.history.push({imageData:t.
getImageData(0,0,e.width,e.height),mosaicRects:n}),markerState.history.length>40&&markerState.history.
shift()}a(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const t=e.getContext(
"2d");if(!t)return;const n=markerState.history[markerState.history.length-1];t.clearRect(0,0,e.width,
e.height),n&&n.imageData?(t.putImageData(n.imageData,0,0),markerState.mosaicRects=Array.isArray(n.mosaicRects)?
n.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[]):n?(t.putImageData(n,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}a(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const t=e.getContext("2d");t&&t.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}a(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let s=!1,o=0,
r=markerView.scale,l={x:0,y:0},d={x:0,y:0},p=[],h=16,g="",y=null,b=null,w=null,x=null,S=!1,T=null;const E=a(
L=>{const R=e.getBoundingClientRect(),G=(L.clientX-R.left)*(e.width/R.width),K=(L.clientY-R.top)*(e.
height/R.height);return{x:G,y:K}},"getPoint"),F=a((L,R)=>({x:(L.x+R.x)/2,y:(L.y+R.y)/2}),"getMid"),J=a(
(L,R)=>Math.hypot(L.x-R.x,L.y-R.y),"getDist");let X=!1;const Te=a(()=>{y||(y=document.createElement(
"canvas"),b=y.getContext("2d")),w||(w=document.createElement("canvas"),x=w.getContext("2d")),(y.width!==
e.width||y.height!==e.height)&&(y.width=e.width,y.height=e.height),(w.width!==e.width||w.height!==e.
height)&&(w.width=e.width,w.height=e.height)},"ensureDrawBuffers"),O=a(()=>{if(!t||!y||!w)return;const L=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(0,0,e.width,e.
height),t.drawImage(y,0,0),t.save(),t.globalAlpha=L,t.drawImage(w,0,0),t.restore()},"renderDrawPrevi\
ew"),q=a(()=>{x&&(x.strokeStyle=g,x.fillStyle=g,x.lineWidth=h,x.lineCap="round",x.lineJoin="round")},
"applyMarkerBrush"),Y=a(L=>{if(!L)return!1;if(p.length===0)return p.push(L),!0;const R=p[p.length-1],
G=L.x-R.x,K=L.y-R.y,Z=Math.hypot(G,K),we=Math.max(.35,h*.04);if(Z<we)return!1;const Q=Math.max(1,h*.25),
re=Math.max(1,Math.ceil(Z/Q));for(let U=1;U<=re;U++){const he=U/re;p.push({x:R.x+G*he,y:R.y+K*he})}return!0},
"appendStrokePoint"),pe=a(()=>{if(x&&(x.clearRect(0,0,w.width,w.height),p.length!==0)){if(q(),p.length===
1){const L=p[0];x.beginPath(),x.arc(L.x,L.y,h/2,0,Math.PI*2),x.fill();return}if(x.beginPath(),x.moveTo(
p[0].x,p[0].y),p.length===2)x.lineTo(p[1].x,p[1].y);else{for(let G=1;G<p.length-2;G++){const K=p[G],
Z=p[G+1],we=F(K,Z);x.quadraticCurveTo(K.x,K.y,we.x,we.y)}const L=p[p.length-2],R=p[p.length-1];x.quadraticCurveTo(
L.x,L.y,R.x,R.y)}x.stroke()}},"renderStrokeLayer"),oe=a((L,R)=>{if(!L||!R)return null;const G=Math.min(
L.x,R.x),K=Math.min(L.y,R.y),Z=Math.abs(L.x-R.x),we=Math.abs(L.y-R.y);return{x:G,y:K,w:Z,h:we}},"nor\
malizeMosaicRect"),ke=a(L=>{const R=n?Number(n.value||16):16,G=Math.max(6,Math.floor(R)),K=Math.floor(
G/2);return{x:L.x-K,y:L.y-K,w:G,h:G}},"buildMosaicRectFromPoint"),me=a(()=>{const L=document.createElement(
"canvas");L.width=e.width,L.height=e.height;const R=L.getContext("2d");if(!R)return null;markerState.
baseCanvas&&R.drawImage(markerState.baseCanvas,0,0),R.drawImage(e,0,0);try{return R.getImageData(0,0,
e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),_e=a(L=>{if(!t||!L)return!1;const R=me();
if(!R)return!1;const G=n?Number(n.value||16):16,K=Math.max(4,Math.floor(G/2)),Z=Math.max(0,Math.floor(
L.x)),we=Math.max(0,Math.floor(L.y)),Q=Math.min(e.width,Math.ceil(L.x+L.w)),re=Math.min(e.height,Math.
ceil(L.y+L.h));if(Q<=Z||re<=we)return!1;for(let U=we;U<re;U+=K)for(let he=Z;he<Q;he+=K){const tt=Math.
min(K,Q-he),Ce=Math.min(K,re-U),st=Math.min(e.width-1,Math.max(0,he+Math.floor(tt/2))),Je=(Math.min(
e.height-1,Math.max(0,U+Math.floor(Ce/2)))*e.width+st)*4,rt=R.data[Je],wt=R.data[Je+1],gt=R.data[Je+
2];t.fillStyle=`rgb(${rt},${wt},${gt})`,t.fillRect(he,U,tt,Ce)}return!0},"applyMosaicRect"),ne=a(L=>{
if(!t)return;if(i.set(L.pointerId,{x:L.clientX,y:L.clientY}),i.size>=2){const G=Array.from(i.values()),
K=G[0],Z=G[1];s=!0,X=!1,p=[],S=!1,T=null,markerState.mosaicPreviewRect=null,o=J(K,Z)||1,r=markerView.
scale,l={x:markerView.offsetX,y:markerView.offsetY},d=F(K,Z),renderCropOverlay(),e.setPointerCapture&&
e.setPointerCapture(L.pointerId),L.preventDefault();return}if(s||markerState.mode==="crop")return;X=
!0;const R=E(L);if(markerState.mode==="mosaic")S=!0,T=R,markerState.mosaicPreviewRect=ke(R),renderCropOverlay();else{
if(Te(),!b||!x)return;b.clearRect(0,0,y.width,y.height),b.drawImage(e,0,0),x.clearRect(0,0,w.width,w.
height),h=n?Number(n.value||16):16,g=normalizeMarkerHexColor(markerState.colorHex),p=[],Y(R),pe(),markerState.
hasStroke=!0,O()}e.setPointerCapture&&e.setPointerCapture(L.pointerId),L.preventDefault()},"start"),
se=a(L=>{if(i.has(L.pointerId)&&i.set(L.pointerId,{x:L.clientX,y:L.clientY}),s&&i.size>=2){const G=Array.
from(i.values()),K=G[0],Z=G[1],we=F(K,Z),Q=J(K,Z)||1,re=r*(Q/o);markerView.scale=Math.min(markerView.
maxScale,Math.max(markerView.minScale,re)),markerView.offsetX=l.x+(we.x-d.x),markerView.offsetY=l.y+
(we.y-d.y),applyMarkerTransform(),L.preventDefault();return}if(!X||!t)return;const R=E(L);if(markerState.
mode==="mosaic"){if(!S||!T)return;markerState.mosaicPreviewRect=oe(T,R)||ke(R),renderCropOverlay()}else
Y(R)&&(pe(),O());L.preventDefault()},"move"),W=a(L=>{const R=X;if(i.delete(L.pointerId),i.size<2&&(s=
!1),i.size===0){if(X=!1,R&&t&&markerState.mode==="draw"&&p.length>0&&(pe(),O()),R&&markerState.mode===
"mosaic"&&T){const G=E(L);let K=oe(T,G);(!K||K.w<2||K.h<2)&&(K=ke(T)),_e(K)&&(markerState.hasStroke=
!0,markerState.mosaicRects.push(K))}p=[],S=!1,T=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),
R&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(L.pointerId),L.preventDefault()},
"end");e.addEventListener("pointerdown",ne),e.addEventListener("pointermove",se),e.addEventListener(
"pointerup",W),e.addEventListener("pointercancel",W)}a(initMarkerCanvas,"initMarkerCanvas");function initCropCanvas(){
const e=get("marker-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,s=null,
o=null,r=null,l=!1,d=0,p=markerView.scale,h={x:0,y:0},g={x:0,y:0};const y=8,b=14,w=a((O,q,Y)=>Math.min(
Y,Math.max(q,O)),"clamp"),x=a(O=>{const q=e.getBoundingClientRect(),Y=(O.clientX-q.left)*(e.width/q.
width),pe=(O.clientY-q.top)*(e.height/q.height);return{x:Y,y:pe}},"getPoint"),S=a((O,q)=>({x:(O.x+q.
x)/2,y:(O.y+q.y)/2}),"getMid"),T=a((O,q)=>Math.hypot(O.x-q.x,O.y-q.y),"getDist"),E=a(()=>(markerState.
cropRect||resetCropRectToFull(),markerState.cropRect),"ensureCropRect"),F=a((O,q)=>{if(!q)return"mov\
e";const Y=q.x,pe=q.y,oe=q.x+q.w,ke=q.y+q.h,me=Math.abs(O.x-Y)<=b,_e=Math.abs(O.x-oe)<=b,ne=Math.abs(
O.y-pe)<=b,se=Math.abs(O.y-ke)<=b;if(me&&ne)return"nw";if(_e&&ne)return"ne";if(me&&se)return"sw";if(_e&&
se)return"se";if(ne)return"n";if(se)return"s";if(me)return"w";if(_e)return"e";if(O.x>Y+b&&O.x<oe-b&&
O.y>pe+b&&O.y<ke-b)return"move";const L=O.x<Y?"left":O.x>oe?"right":null,R=O.y<pe?"top":O.y>ke?"bott\
om":null;if(L&&R){if(L==="left"&&R==="top")return"nw";if(L==="right"&&R==="top")return"ne";if(L==="l\
eft"&&R==="bottom")return"sw";if(L==="right"&&R==="bottom")return"se"}return L?L==="left"?"w":"e":R?
R==="top"?"n":"s":"move"},"hitTest"),J=a(O=>{if(markerState.mode!=="crop")return;if(n.set(O.pointerId,
{x:O.clientX,y:O.clientY}),n.size>=2){const pe=Array.from(n.values()),oe=pe[0],ke=pe[1];l=!0,i=!1,d=
T(oe,ke)||1,p=markerView.scale,h={x:markerView.offsetX,y:markerView.offsetY},g=S(oe,ke),e.setPointerCapture&&
e.setPointerCapture(O.pointerId),O.preventDefault();return}if(l)return;i=!0;const q=x(O),Y=E();o=F(q,
Y),s=q,r=Y?{x:Y.x,y:Y.y,w:Y.w,h:Y.h}:null,renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
O.pointerId),O.preventDefault()},"start"),X=a(O=>{if(markerState.mode!=="crop")return;if(n.has(O.pointerId)&&
n.set(O.pointerId,{x:O.clientX,y:O.clientY}),l&&n.size>=2){const L=Array.from(n.values()),R=L[0],G=L[1],
K=S(R,G),Z=T(R,G)||1,we=p*(Z/d);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,
we)),markerView.offsetX=h.x+(K.x-g.x),markerView.offsetY=h.y+(K.y-g.y),applyMarkerTransform(),renderCropOverlay(),
O.preventDefault();return}if(!i||!s||!r)return;const q=x(O),Y=e.width,pe=e.height,oe={x:r.x,y:r.y,w:r.
w,h:r.h},ke=r.x+r.w,me=r.y+r.h,_e=a(()=>{const L=w(q.x,0,ke-y);oe.x=L,oe.w=ke-L},"applyW"),ne=a(()=>{
oe.w=w(q.x-r.x,y,Y-r.x)},"applyE"),se=a(()=>{const L=w(q.y,0,me-y);oe.y=L,oe.h=me-L},"applyN"),W=a(()=>{
oe.h=w(q.y-r.y,y,pe-r.y)},"applyS");switch(o){case"move":{const L=q.x-s.x,R=q.y-s.y;oe.x=w(r.x+L,0,Y-
r.w),oe.y=w(r.y+R,0,pe-r.h);break}case"w":_e();break;case"e":ne();break;case"n":se();break;case"s":W();
break;case"nw":se(),_e();break;case"ne":se(),ne();break;case"sw":W(),_e();break;case"se":W(),ne();break;default:
break}oe.x=w(oe.x,0,Y-oe.w),oe.y=w(oe.y,0,pe-oe.h),markerState.cropRect=oe,renderCropOverlay(),O.preventDefault()},
"move"),Te=a(O=>{n.delete(O.pointerId),n.size<2&&(l=!1),n.size===0&&(renderCropOverlay(),i=!1,s=null,
o=null,r=null),e.releasePointerCapture&&e.releasePointerCapture(O.pointerId),O.preventDefault()},"en\
d");e.addEventListener("pointerdown",J),e.addEventListener("pointermove",X),e.addEventListener("poin\
terup",Te),e.addEventListener("pointercancel",Te),e.addEventListener("pointerleave",Te)}a(initCropCanvas,
"initCropCanvas");async function saveMarkerToRow(){const e=markerState.row,t=get("marker-image"),n=get(
"marker-canvas");if(!e||!t||!n)return;const i=get("marker-attach-original");i&&(e.dataset.attachOriginal=
i.checked?"1":"");let s=document.createElement("canvas");const o=markerState.naturalWidth||t.naturalWidth||
n.width,r=markerState.naturalHeight||t.naturalHeight||n.height;s.width=o,s.height=r;const l=s.getContext(
"2d");if(!l)return;if(l.drawImage(t,0,0,o,r),l.drawImage(n,0,0,o,r),markerState.cropRect){const S=o/
n.width,T=r/n.height,E=Math.max(0,Math.floor(markerState.cropRect.x*S)),F=Math.max(0,Math.floor(markerState.
cropRect.y*T)),J=Math.min(o,Math.max(1,Math.floor(markerState.cropRect.w*S))),X=Math.min(r,Math.max(
1,Math.floor(markerState.cropRect.h*T))),Te=document.createElement("canvas");Te.width=J,Te.height=X;
const O=Te.getContext("2d");O&&(O.drawImage(s,E,F,J,X,0,0,J,X),s=Te)}const d=await new Promise(S=>s.
toBlob(S,"image/png",.92));if(!d){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const h=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),g=new File([
d],`${h}_marked.png`,{type:"image/png"}),y={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(
".upload-status"),bar:e.querySelector(".upload-progress > div")};y.status&&(y.status.textContent="\u7DE8\u96C6\
\u53CD\u6620\u4E2D..."),updateUploadRowFile(y,g);const b=e.getAttribute("data-filename"),w=getRowAttachmentSource(
e);b&&!e.dataset.originalFilename&&(e.dataset.originalFilename=b,e.dataset.originalSource=w,setAttachmentSourceForPath(
b,w)),await uploadFileWithProgress(g,y)?(b&&(currentImageUrls=currentImageUrls.filter(S=>S!==b)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}a(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,i.src=URL.createObjectURL(
e);let s=null,o=null,r=null,l=null,d=[],p=null;const h=a(()=>{p&&clearTimeout(p);try{URL.revokeObjectURL(
i.src)}catch{}try{i.remove()}catch{}if(s&&s.getTracks().forEach(y=>y.stop()),r)try{r.disconnect()}catch{}
if(l)try{l.disconnect()}catch{}if(o)try{o.close()}catch{}},"cleanup"),g=a(()=>{h(),n(null)},"fail");
i.onloadedmetadata=async()=>{try{s=i.captureStream();const y=s.getAudioTracks();if(!y||!y.length)return g();
o=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),l=o.createMediaStreamSource(
new MediaStream(y)),r=o.createScriptProcessor(4096,1,1),r.onaudioprocess=w=>{const x=w.inputBuffer.getChannelData(
0);d.push(new Float32Array(x))},l.connect(r),r.connect(o.destination);const b=isFinite(i.duration)?Math.
max(1,Math.ceil(i.duration*1e3)):0;b>0&&(p=setTimeout(()=>{const w=(e.name||"video").replace(/\.[^/.]+$/,
""),x=encodeWav(d,o.sampleRate),S=new File([x],`${w}.audio.wav`,{type:"audio/wav"});h(),n(S)},b+250)),
await i.play(),i.onended=()=>{const w=(e.name||"video").replace(/\.[^/.]+$/,""),x=encodeWav(d,o.sampleRate),
S=new File([x],`${w}.audio.wav`,{type:"audio/wav"});h(),n(S)}}catch{g()}},i.onerror=()=>g()}))}a(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,t={}){if(!e||!e.length)return;const n=Array.from(
e).filter(Boolean);if(!n.length)return;const i=collectImageUrlsForSend().length+browserFastLocalFiles.
size+Math.max(0,Number(uploadProgressState.active)||0);let s=n;if(i+n.length>ATTACHMENT_MAX_FILES){const g=Math.
max(0,ATTACHMENT_MAX_FILES-i);if(g<=0){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059`,"error",!0);return}s=n.slice(0,g),showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u5148\u982D${g}\u4EF6\u306E\u307F\u8FFD\u52A0\u3057\u307E\u3059\u3002`,"war\
ning",!0)}t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=
s.length,uploadProgressState.active+=s.length,updateFilePreview();const o=!!(get("upload-audio-only")&&
get("upload-audio-only").checked),r=getModelMediaSupport(get("model-select").value),l=a(async g=>{let y=null;
try{if(isAudioFile(g)&&!r.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(g)&&!r.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!g.type||!g.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const b=addUploadRow(g);updateFilePreview(),y=b.uploadId,uploadProgressState.perFilePct[y]=
0;let w=g;if(o&&isVideoFile(g)){const x=await extractAudioFromVideo(g,b);x?(w=x,updateUploadRowFile(
b,x),b&&b.status&&(b.status.textContent="\u97F3\u58F0\u306E\u307F")):(b&&b.status&&(b.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&g.type.startsWith("image/"))try{const x=getCompressionOutputType();
if(getCompressionFormatOnly())w=await convertImageFormatOnly(g,x);else{const T={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};x&&x!=="original"&&(T.fileType=x),await ensureImageCompression();
const E=await window.imageCompression(g,T),F=new File([E],imageFilenameForMime(g.name,E.type||(x!=="\
original"?x:g.type)),{type:E.type||g.type,lastModified:g.lastModified||Date.now()});F.size>g.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
g.size)} -> ${formatBytes(F.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),w=g):w=F}w!==g&&updateUploadRowFile(b,w)}catch{}if(browserFastModeEnabled){const x=Array.from(
browserFastLocalFiles.values()).reduce((S,T)=>S+Number(T.file&&T.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||x+w.size>BROWSER_FAST_MAX_BYTES?(b&&b.status&&(b.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),b&&b.row&&b.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(b.uploadId,{file:w,rowObj:b}),b.status&&(b.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),b.bar&&(b.bar.style.width="100\
%"),b.row&&(b.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(w,b)}finally{
y&&uploadProgressState.perFilePct.hasOwnProperty(y)&&(delete uploadProgressState.perFilePct[y],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let d=0;const p=Math.min(UPLOAD_CONCURRENCY,s.length),h=Array.from({length:p}).map(async()=>{
for(;;){const g=d++;if(g>=s.length)break;await l(s[g])}});await Promise.all(h)}a(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){const n=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).replace(/\n/g,
"<br>");const i=protectMathSegments(n),s=window.marked.parse(i.text),o=restoreMathSegments(s,i.blocks,
t);return window.DOMPurify.sanitize(o)}a(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");function getCanvasModeElements(){
const e=get("canvas-panel");return e?{panel:e,stage:get("conversation-stage"),title:get("canvas-pane\
l-title"),status:get("canvas-panel-status"),blockCount:get("canvas-block-count"),blockList:get("canv\
as-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get("canvas-preview-lang"),sourceSelect:get(
"canvas-source-select"),frame:get("canvas-preview-frame"),empty:get("canvas-preview-empty"),sourceScroll:get(
"canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get("canvas-panel-copy-btn"),clearBtn:get(
"canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:null}a(getCanvasModeElements,"getC\
anvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(t||"");return/<!doctype\s+html/i.
test(i)||/<html[\s>]/i.test(i)}a(isCanvasHtmlPreviewCandidate,"isCanvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){
const n=String(e&&e.lang?e.lang:"").trim(),i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),s=!!(e&&
e.open);return{...e,index:t,lang:n,code:i,open:s,key:hashString(`${n||"TEXT"}
${i||""}`)}}a(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],s=[],o=/^(\s*)(`{3,}|~{3,})(.*)$/;let r=null,l="",d=[];for(const g of n){
if(!r){const w=g.match(o);if(w){r=w[2],l=String(w[3]||"").trim(),d=[],i.push({lang:l,code:"",open:!0}),
s.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}s.push(
g);continue}const y=String(g||"").trim();if(y&&y.replace(/\s+/g,"")===r){const w=i[i.length-1];w&&(w.
code=d.join(`
`),w.open=!1),r=null,l="",d=[];continue}d.push(g);const b=i[i.length-1];b&&(b.code=d.join(`
`))}if(r&&i.length){const g=i[i.length-1];g&&(g.code=d.join(`
`),g.open=!0)}const p=i.map((g,y)=>normalizeCanvasBlock(g,y)),h=selectCanvasPreviewBlock(p,t);return{
renderText:s.join(`
`),blocks:p,primaryBlock:h?h.block:null,primaryIndex:h?h.index:-1,rawText:t}}a(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const o=i[n];return{block:o,index:n,previewType:isCanvasHtmlPreviewCandidate(
o.lang,o.code)?"html":"code"}}if(i.length>0){const o=i.length-1,r=i[o];return{block:r,index:o,previewType:isCanvasHtmlPreviewCandidate(
r.lang,r.code)?"html":"code"}}const s=String(t||"");return isCanvasHtmlPreviewCandidate("",s)?{block:normalizeCanvasBlock(
{lang:"html",code:s,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}a(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}a(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),s=!!t&&t===i;n.classList.toggle("can\
vas-active",s),n.setAttribute("aria-pressed",s?"true":"false"),n.setAttribute("data-canvas-active",s?
"1":"0"),n.innerHTML=s?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=s?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",s?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
a(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}a(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
s={preview:0,blocks:1,source:2},o=i[n];if(!o||!(t in s)||!(n in s))return;canvasPreviewState.viewAnimationToken+=
1;const r=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(d=>{d&&d.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),o.
offsetWidth;const l=s[n]<s[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";o.classList.
add(l),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{r===canvasPreviewState.viewAnimationToken&&
(o.classList.remove(l),canvasPreviewState.viewAnimationTimer=null)},340)}a(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var r,l;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",s=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const p=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",p),d.setAttribute("aria-pressed",p?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,s,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((l=(r=n.blockList).
focus)==null||l.call(r,{preventScroll:!0})))}a(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,s)=>{const o=String(i&&i.lang?i.lang:"text").trim()||"text",r=s===n,l=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",h=(String(i&&i.code?i.code:"").split(/\r?\n/).find(b=>b.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),g=`${r?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${o}`,y=`${g}\u3001${h}`;return`<butto\
n type="button" class="canvas-block-chip${r?" active":""}" data-canvas-block-index="${s}" title="${escapeHtml(
g)}" aria-label="${escapeHtml(y)}" aria-pressed="${r?"true":"false"}"><span class="canvas-block-chip\
-index">#${s+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
o)}</span><span class="canvas-block-chip-preview">${escapeHtml(h)}</span></span><span class="canvas-\
block-chip-state">${r?"\u8868\u793A\u4E2D":l}</span></button>`}).join("")}a(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((o,r)=>{const l=String(
o&&o.lang?o.lang:"text").trim()||"text";return`#${r+1} ${l}`}),s=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==s&&(e.sourceSelect.innerHTML=i.map((o,r)=>`<option value="${r}">${escapeHtml(
o)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=s),e.sourceSelect.value=String(
n)}a(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}a(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),s=String(e||""),o=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const r=new DOMParser().parseFromString(s,"text/html"),l=r.createElement("scr\
ipt");return l.setAttribute("data-canvas-scroll-bridge","true"),l.textContent=o,(r.body||r.documentElement).
appendChild(l),`<!DOCTYPE html>
`+r.documentElement.outerHTML}catch{return`${s}<script data-canvas-scroll-bridge>${o}<\/script>`}}a(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const t=e&&e.data?e.data:null;if(!t||t.type!=="canvas-preview-scroll")return;const n=getCanvasModeElements();
!n||!n.frame||e.source!==n.frame.contentWindow||t.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(t.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(t.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const t=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{t===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}a(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const t=getCanvasModeElements();if(t){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){t.panel.classList.add("hidden"),t.panel.classList.remove(
"canvas-panel-open","canvas-closing"),t.stage&&t.stage.classList.remove("canvas-enabled");return}t.panel.
classList.remove("canvas-panel-open"),t.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{t.panel.classList.add("hidden"),t.panel.classList.remove("canvas-closing"),t.
stage&&t.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}a(hideCanvasPreviewPanel,
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
en")),t.empty&&t.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}a(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const t=e||canvasPreviewState.lastCanvasData;
if(!t)return null;canvasPreviewState.lastCanvasData=t,canvasPreviewState.blocks=Array.isArray(t.blocks)?
t.blocks.slice():[],canvasPreviewState.rawText=String(t.rawText||""),canvasPreviewState.renderText=String(
t.renderText||"");const n=canvasPreviewState.blocks,i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!n.length){const r=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return r&&r.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=r.block.
key||"",r.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let s=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?s=i:canvasPreviewState.selectionMode="auto";const o=n[s]||null;return canvasPreviewState.
selectedIndex=o?s:-1,canvasPreviewState.selectedKey=o&&o.key?o.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),o}a(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,s=n&&Number.isInteger(
n.index)?n.index:-1,o=!!i,r=String(i&&i.lang?i.lang:"").trim(),l=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),d=o?isCanvasHtmlPreviewCandidate(r,l):!1,p=o?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",h=o?d?`HTML Canvas Preview${t.length>
1&&s>=0?` #${s+1}/${t.length}`:""}`:`Canvas Preview: ${r||"text"}${t.length>1&&s>=0?` #${s+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=h),e.status&&(e.status.textContent=
p),e.previewLang&&(e.previewLang.textContent=o?r||"text":"idle");const g=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,y=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=l),e.sourceScroll&&(e.sourceScroll.scrollTop=g,e.sourceScroll.
scrollLeft=y,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),o){canvasPreviewState.frameRenderToken+=1;const b=canvasPreviewState.frameRenderToken,
w=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),b);e.frame&&(e.frame.srcdoc=w,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{b!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:b,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}a(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const s=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
s&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}a(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(s=>s&&s.key===n);return i===-1?!1:applyCanvasSelection(i)}
a(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
s=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:s}}a(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const l=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[l],selectedIndex:0,selectedKey:l.key||t.codeKey||""}}const s=[];let o=-1;if(i.forEach(
(l,d)=>{const p=decodeCanvasPreviewButtonCode(l);if(!p)return;const h=normalizeCanvasBlock({lang:p.lang,
code:p.code,open:!1},d);s.push(h),o===-1&&p.codeKey===t.codeKey&&(o=s.length-1)}),!s.length)return null;
o===-1&&(o=0);const r=s[o]||s[0]||null;return{blocks:s,selectedIndex:o,selectedKey:r&&r.key?r.key:t.
codeKey||""}}a(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
if(!e)return!1;const t=collectCanvasBlocksFromButton(e);if(!t||!t.blocks||!t.blocks.length)return!1;
const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[],i=n.findIndex(o=>o&&o.
key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(i);const s=t.blocks[t.selectedIndex]||
t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.rawText=s&&s.code!==void 0&&
s.code!==null?String(s.code):"",canvasPreviewState.renderText=canvasPreviewState.rawText,canvasPreviewState.
selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.selectedKey=t.selectedKey||
s&&s.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),canvasPreviewState.lastCanvasData=
{renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:s,primaryIndex:canvasPreviewState.
selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.mobileView="preview",syncCanvasPanelViewUi(
"preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}a(previewCanvasCodeFromButton,"previewCanvasCod\
eFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&e.code!==null?
e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(n,t))return sanitizeHtmlForPreview(
t);const s=n?`Canvas Preview: ${n}`:"Canvas Preview",o=escapeHtml(t||"");return`<!doctype html><html\
 lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-sc\
ale=1"><title>${escapeHtml(s)}</title><style>
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
            </style></head><body><div class="frame"><div class="label">${escapeHtml(s)}</div><pre>${o||
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}a(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,t={}){const n=t.persist!==
!1;if(canvasModeEnabled=!!e,n)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const i=get("enable-canvas-mode");if(i&&i.checked!==canvasModeEnabled&&(i.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(t.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!t.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}a(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}a(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,t,n){let i=normalizeMarkdownNewlines(
e);const s=normalizeMarkdownNewlines(n);if(!s&&s!=="")return i;const o=t?[String(t),""]:[""];for(const r of[
"`","~"])for(let l=3;l<=10;l++){const d=r.repeat(l);for(const p of o){const h=`${d}${p}
`,g=`
${d}`,y=h+s+g;i.includes(y)&&(i=i.split(y).join(""))}}return i}a(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),s=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const o of s)for(const r of["`","~"])for(let l=3;l<=10;l++){const d=r.repeat(l);
[`${o}${d}
${i}
${d}`,`${o}
${d}
${i}
${d}`,`
${o}${d}
${i}
${d}`,`
${o}
${d}
${i}
${d}`].forEach(h=>{n.includes(h)&&(n=n.split(h).join(`
`))})}return n}a(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}a(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",s=String(t==null?"":t).replace(/\s+$/,"");return s?
s+`

`+i:i}a(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[];if(!t)return{text:"",executions:n};const i=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let s=t.replace(i,(o,r,l)=>{const d=String(l||"").trim();try{const p=JSON.parse(d);n.push({code:p&&p.
code!=null?String(p.code):"",output:p&&p.output!=null?String(p.output):""})}catch{n.push({code:d,output:""})}
return`
`});return n.forEach(o=>{o.code&&(s=stripExactFencedBlock(s,"python",o.code),s=stripExactFencedBlock(
s,"py",o.code)),s=stripVisiblePythonOutputBlock(s,o.output)}),s=s.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:s,executions:n}}a(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(o=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(o)?n.push(o.trim()):
i.push(o)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}a(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}a(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",s=e&&e.output!=null?String(e.output):"";let o="";try{window.
hljs&&typeof window.hljs.highlight=="function"?o=window.hljs.highlight(i,{language:"python"}).value:
o=escapeHtml(i)}catch{o=escapeHtml(i)}const r=escapeHtml(s),l=encodeURIComponent(i).replace(/'/g,"%2\
7"),d=encodeURIComponent(s).replace(/'/g,"%27"),p=hashString(`pyexec-detail
${i}
${s}
${t}`),h=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",g=`<button class="download-btn" data-\
code="${l}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,y=`<button class="coding-target-btn" data-code="${l}" data-code-key="${p}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${p}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(h)}</span><div class="code-actions">${y}${g}<button class="copy-btn" data-copy="code" \
data-code="${l}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${d}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${o}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${r}</code></pre></div></div></div>`}a(buildPythonExecDetailBoxHtml,"buildP\
ythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/python-\
execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"","/py\
thon-execution")}showModal("python-exec-modal")}a(showPythonExecDetailModal,"showPythonExecDetailMod\
al");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("pytho\
n-exec-modal-body"),s=get("python-exec-modal-title");if(!n||!i)return;const o=t&&Array.isArray(t.python_executions)?
t.python_executions:[];if(!o.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(s){const r=o.length>1?`\uFF08${o.length}\u4EF6\uFF09`:"";s.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${r}`}i.innerHTML=o.map((r,l)=>buildPythonExecDetailBoxHtml(r,l,o.length)).join(
""),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(!0,{persist:!1})),showPythonExecDetailModal(
e)}a(openPythonExecDetail,"openPythonExecDetail"),window.openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){
get("python-exec-modal")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&
history.back())}a(closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;
function buildAiMarkdownHtml(e){const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(
t.text,t.notes),i=canvasModeEnabled?parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,
rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(i),refreshCanvasPreviewPanel());const s=document.
createElement("div");return s.className="prose prose-invert text-sm break-words",s.innerHTML=sanitizeMarkdownHtml(
i.renderText),wrapRenderedSvgBoxes(s),lowBandwidthMode||(maybeNeedsHighlight(i.renderText,s)&&ensureHighlightLoaded().
catch(()=>{}),maybeNeedsMathJax(i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),s.outerHTML}a(buildAiMarkdownHtml,
"buildAiMarkdownHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),s=appendMcpExecutionNotes(i.text,i.notes),o=canvasModeEnabled?parseCanvasMarkdown(s):{renderText:s,
blocks:[],primaryBlock:null,rawText:s};if(canvasModeEnabled&&(updateCanvasPreviewState(o),refreshCanvasPreviewPanel()),
n.incrementalMath){const r=document.createElement("template");r.innerHTML=sanitizeMarkdownHtml(o.renderText,
{streamMathSegments:!0});const l=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(p=>{const h=p.getAttribute("data-stream-math-key");h&&l.set(h,p)});const d=[];r.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(p=>{const h=l.get(p.getAttribute(
"data-stream-math-key"));h?p.replaceWith(h):d.push(p)}),e.replaceChildren(r.content),wrapRenderedSvgBoxes(
e),queueHighlight(e,o.renderText),queueIncrementalMathTypeset(d);return}e.innerHTML=sanitizeMarkdownHtml(
o.renderText),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,o.renderText)}a(renderAiMarkdownInto,
"renderAiMarkdownInto");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}a(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");function renderMessage(e,t,n,i,s,o,r=null,l=!0,d=null,p=null,h=null,g=null,y=null,b=null,w=null,x=null,S=!0,T=null,E=null,F=null){
const J=t==="user",X=J?"bg-blue-600":"bg-gray-700",Te=J?"justify-end":"justify-start";messageStore[e]=
n;const O=!J&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},q=J?n:O.text;let Y=p;
if(Y==null){const Q=h!=null?Number(h):0,re=g!=null?Number(g):0;(h!=null||g!=null)&&(Y=Q+re)}messageMeta[e]=
{tokens_in:h,tokens_out:g,tokens_total:Y,tokens_content:b,tokens_thought:w,is_encrypted:y,role:t,model:o,
parent_id:T,quote_text:d,image_url:i,gem_name:E,batch_job:F,python_executions:J?[]:O.executions||[]};
let pe="";d&&(pe=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-g\
ray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(d)}\
</div>`);let oe="";if(s&&!J){let Q="";try{Q=JSON.parse(s).text||""}catch{Q=s}Q&&(oe=`<div class="tho\
ught-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain te\
xt-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(Q)}</d\
iv></div>`)}let ke="";if(i)try{const Q=JSON.parse(i);if(Q.length){const re=[];if(Q.forEach(U=>{let he=U,
tt="unknown";if(he&&typeof he=="object"&&(tt=normalizeAttachmentSource(he.source),he=he.filepath||he.
path||he.url||he.file||""),he=normalizeAttachmentPath(he)||he,!he)return;setAttachmentSourceForPath(
he,tt);const Ce=he.replace(/^\d+\//,""),st=buildFileUrl(Ce),ut=buildAttachmentPreviewUrl(Ce),Je=he.split(
"/").pop(),rt=Je.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(rt)?re.push(
buildChatImageHtml(ut,{viewerSrc:st,alt:Je,title:Je,filename:Je})):re.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${st}')" title="${Je}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Je}</span></div>`)}),re.length>
0){let U="grid-multi";re.length===1?U="grid-1":re.length===2?U="grid-2":re.length===3?U="grid-3":re.
length===4&&(U="grid-4"),ke=`<div class="image-grid ${U}">${re.join("")}</div>`}}}catch{}const me=J?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,_e=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${J?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${me}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
ne=[];!J&&o&&ne.push(escapeHtml(o)),E&&(J?ne.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml(E)}</span>`):ne.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml(E)}</span>`));const se=[];if(h!=null&&se.push(`In ${h}`),g!=null){let Q=`\
Out ${g}`;w!=null&&Number(w)>0&&(Q+=` (Thought ${w})`),se.push(Q)}if(se.length||p!=null){const Q=se.
length?se.join(" / "):`${p} tokens`;ne.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${Q}</button>`)}if(y!=null){const Q=y?"fa-l\
ock":"fa-lock-open",re=isAdminUser?y?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":y?"Encrypted":
"Plain",U=isAdminUser?y?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";ne.push(`<button class="${U}" title="${re}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${Q}"></i></button>`)}if(!J&&O.executions&&O.executions.length){
const Q=O.executions.length,re=Q>1?`Python \xD7${Q}`:"Python";ne.push(`<button type="button" class="\
python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\u884C\u7D50\u679C\
\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${re}</span></button>`)}const W=ne.length?`\
<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${ne.join("\
 \u2022 ")}</div>`:"";let L;const R=!J&&F?(()=>{const Q=String(F.state||"").toUpperCase(),re=F.status_text||
(Q==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":Q==="JOB_ST\
ATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\u3067\u3059");
return`<div class="batch-status-card mb-3 rounded-lg border ${Q==="JOB_STATE_FAILED"||Q==="JOB_STATE\
_CANCELLED"||Q==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":Q==="JOB_STATE_S\
UCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet-400/40 bg-violet\
-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="fas fa-layer-group\
 mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(re)}</div></div>`})():"";J?L=`<div c\
lass="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||"")}</div>`:(L=
R+(q&&String(q).trim()?buildAiMarkdownHtml(q):F?'<div class="content-area prose prose-invert text-sm\
 break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(q)),L.includes("content-area")||(L=L.replace("prose ","content-area prose ")));let G="";
if(r){const Q=r.siblings[r.current-2],re=r.siblings[r.current];G=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${Q}\
)" ${Q?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${r.current} / ${r.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${re}\
)" ${re?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const K=l?"fade-in":"",Z=document.createElement("div");Z.className=`flex ${Te} mb-\
4 ${K} relative message-group group`,Z.id=`msg-${e}`,Z.innerHTML=`<div class="message-bubble ${X} te\
xt-white p-4 rounded-2xl shadow-md relative">${_e}${pe}${oe}${L}${ke}${G}${W}</div>`;const we=x||get(
"chat-container");return we&&(we.appendChild(Z),S&&scrollToBottom(),J||(queueMessageDecorations(Z,q),
syncCodingTargetButtons(Z),syncCodingModeUi(codingModeEnabled,{persist:!1}))),Z}a(renderMessage,"ren\
derMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const t={
modal:"token-details"};e!==null&&(t.messageId=e),history.pushState(t,"","/token-details")}showModal(
"token-detail-modal")}a(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];
if(!t||!get("token-detail-modal"))return;const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",s=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",o=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",r=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",l=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detai\
l-in").innerText=s,get("token-detail-out").innerText=o,get("token-detail-content").innerText=r,get("\
token-detail-thought").innerText=l,get("token-detail-encrypted").innerText=d;const p=t.model?`${t.model}\
 (${t.role})`:`${t.role}`;get("token-detail-title").innerText=p,showTokenDetailModal(e)}a(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){get("token-detail-modal")&&(hideModal("token-deta\
il-modal"),!e&&location.pathname==="/token-details"&&history.back())}a(closeTokenDetail,"closeTokenD\
etail");function openEncryptionSettings(e){const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}
a(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get("encrypti\
on-status-modal"))return;const n=get("encryption-status-title"),i=get("encryption-status-body"),s=get(
"encryption-status-admin-actions"),o=get("encryption-status-admin-toggle"),r=!!e;r?(n&&(n.innerText=
"\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\
\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
s&&o&&(!!(isAdminUser&&currentThreadId)?(s.classList.remove("hidden"),o.dataset.enable=r?"0":"1",o.disabled=
!1,o.textContent=r?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
o.className=r?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):s.classList.add("hidden")),showEncryptionStatusModal()}a(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}a(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const t=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${t?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const i=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,t,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=i}}a(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}a(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}a(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}a(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=a((e,t,n,i)=>{
const s=(e||"").toLowerCase();return!s.includes("gemini")||s.includes("image")||s.includes("nano")||
s.includes("tts")||s.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=a(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),s=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(o=>{let r=!1;function l(){n&&n.removeEventListener("cli\
ck",p),i&&i.removeEventListener("click",h),s&&s.removeEventListener("click",h),e.removeEventListener(
"click",g,!0)}a(l,"cleanup");function d(y){if(r)return;r=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),l(),hideModal("gemini-local-python-modal"),o(y)}a(d,"finalize");
function p(){d(!0)}a(p,"onOk");function h(){d(!1)}a(h,"onCancel");function g(y){y.target===e&&(y.preventDefault(),
y.stopImmediatePropagation(),h())}a(g,"onOverlay"),n&&n.addEventListener("click",p),i&&i.addEventListener(
"click",h),s&&s.addEventListener("click",h),e.addEventListener("click",g,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,s=null){const o=t?"fade-in":"",r=i?
` id="${i}"`:"",l=buildPendingSkeletonHtml(s,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ${o}"><div${r} class="message-bubble ai-pending-bubble bg-gray-700 text-w\
hite p-4 rounded-2xl rounded-tl-none shadow-md relative">${l}</div></div>`,p=e||get("chat-container");
if(p){if(typeof p.insertAdjacentHTML=="function")p.insertAdjacentHTML("beforeend",d);else{const h=document.
createElement("div");h.innerHTML=d;const g=h.firstElementChild;g&&p.appendChild(g)}n&&scrollToBottom()}}
a(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){if(!e||e.getAttribute(
"data-stream-transition")==="1")return;const t=e.querySelector(".content-area");t&&(t.classList.remove(
"pending-shimmer","skeleton-pending"),t.removeAttribute("data-skeleton-kind")),e.setAttribute("data-\
stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add("ai-stream-transitio\
n"),t&&(t.classList.add("ai-stream-content-transition"),setTimeout(()=>{t&&t.classList.remove("ai-st\
ream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-transition")},320)}
a(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){return e==
null||e===""?null:String(e)}a(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}a(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const t=Array.from(e.querySelectorAll(".prose")).some(l=>String(l.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(l=>!!String(l.textContent||
"").trim()&&l.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const s=e.parentElement;
if(!s)return null;const o=s.cloneNode(!0);o.setAttribute("data-local-stopped-partial","1"),o.classList.
remove("fade-in");const r=o.querySelector(".message-bubble");if(r&&(r.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),r.removeAttribute("data-stream-transition"),r.removeAttribute("id"),!o.
querySelector('[data-stopped-partial-note="1"]'))){const l=document.createElement("div");l.setAttribute(
"data-stopped-partial-note","1"),l.className="text-[10px] text-amber-200/90 mt-2 text-right",l.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",r.appendChild(l)}return{html:o.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}a(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const s=get("chat-\
container");return s?(s.querySelectorAll('[data-local-stopped-partial="1"]').forEach(o=>o.remove()),
s.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}a(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}a(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}a(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}a(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var l,d;const n=Math.max(0,Number((l=t.retries)!=null?l:1)||0),i=Math.max(0,Number((d=t.retryDelayMs)!=
null?d:180)||0),s=!!t.notifyOnFailure,o=e!=null&&e!==""?String(e):null,r=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!r||o&&r!==o)return!1;for(let p=0;p<=n;p++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==r?!1:(await loadMessages(r,{preserveDraft:!0,silent:!0}),
!0)}catch{p<n&&i>0&&await new Promise(g=>setTimeout(g,i))}return s&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===r&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}a(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(t){console.warn("Vi\
bration failed:",t)}}a(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const t=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(n=>n.kind==="minimal"&&!minimalPromptMode?!1:n.label.
toLowerCase().includes(t)||n.description.toLowerCase().includes(t))}a(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,t){if(String(e||"").toLowerCase()!=="thinking")
return e;const i=String(t||"").trimStart().match(/^\/thinking(\s+.*)$/i);return i?`thinking${i[1]}`.
toLowerCase():e}a(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const t=String(e||"").trim().toLowerCase();if(!t||t==="toggle"||t==="\u5207\u66FF"||t==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(t))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(t))return!1}a(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(o=>o.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(o=>o.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const s=String(n.presetArgument||t||"").trim();if(i.selectId){if(!s)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} ${n.id==="effort"?"none / low / medium / high / xhigh / max":"default\
 / none"}`,"info"),!1;const o=get(i.selectId),r=s.toLowerCase(),l=o?Array.from(o.options).find(d=>d.
value.toLowerCase()===r||d.textContent.trim().toLowerCase()===r):null;return!o||!l?(showToast(`${n.label}\
: \u6307\u5B9A\u5024\u300C${s}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(o.value=l.value,o.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${i.label}: ${l.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&s){const o=s.toLowerCase(),
r={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},l=parseSlashToggleArgument(
s),d=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(r,o)){d&&!d.checked&&!d.disabled&&(d.
checked=!0,d.dispatchEvent(new Event("change",{bubbles:!0})));const p=get("thinking-level");return p&&
(p.value=r[o],p.dispatchEvent(new Event("change",{bubbles:!0}))),refreshMinimalOptionItems(),showToast(
`Thinking: ${o}`,"success"),!0}if(l===void 0)return showToast("\u4F7F\u3044\u65B9: /thinking on / off / min / low /\
 mid / high","info"),!1}if(i.checkboxId&&s){const o=parseSlashToggleArgument(s);if(o===void 0)return showToast(
`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const r=get(i.checkboxId);if(o!==null&&r&&r.checked===
o)return showToast(`${i.label}: ${o?"ON":"OFF"}`,"info"),!0}return handleMinimalOptionClick(i),!0}a(
executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){const t=String(
e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||"",i=n.
match(/^[a-z][\w-]*/i);return i?i[0]:n}a(extractSlashCommandToken,"extractSlashCommandToken");function hideSlashCommandSuggestions(){
const e=get("slash-command-suggestions");e&&e.classList.add("hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=
0}a(hideSlashCommandSuggestions,"hideSlashCommandSuggestions");function showPendingSlashCommandIndicator(e){
const t=get("slash-command-indicator"),n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.
find(o=>o.id===e);n.textContent=i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");
const s=get("prompt-input");s&&i&&(s.dataset.originalPlaceholder=s.placeholder,s.placeholder=i.argumentHint||
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}
a(showPendingSlashCommandIndicator,"showPendingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){
const e=get("slash-command-indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get(
"prompt-input");t&&t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.
dataset.originalPlaceholder);const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}
a(hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const s=visibleSlashCommands(e);if(s.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,s.length-1),n.innerHTML="",s.forEach((y,b)=>{const w=document.createElement(
"div");w.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${b===
slashSelectedIndex?"bg-gray-700":""}`,w.innerHTML=`
                    <i class="fas ${y.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${y.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${y.description}</div>
                    </div>
                `;let x=!1;w.addEventListener("pointerdown",S=>{typeof S.button=="number"&&S.button!==
0||(S.preventDefault(),x=!0,selectSlashCommand(y.id))}),w.addEventListener("click",S=>{S.preventDefault(),
x||selectSlashCommand(y.id)}),w.onmouseenter=()=>{slashSelectedIndex=b,showSlashCommandSuggestions(e)},
n.appendChild(w)});const o=i.getBoundingClientRect(),r=window.innerHeight,l=r-o.bottom,d=o.top,p=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,o.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",l<180&&d>l){const y=Math.min(p,d-h);t.style.top="auto",t.style.bottom=`${r-o.top+4}px`,n.style.
maxHeight=`${y}px`}else{const y=Math.min(p,l-h);t.style.top=`${o.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${y}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}a(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const r=String(n||"").trimStart();t.value=
r.substring(1+i.length).trimStart()}else{const r=n.lastIndexOf("/");r!==-1?t.value=n.substring(0,r).
trimEnd():t.value=""}hideSlashCommandSuggestions();const s=SLASH_COMMANDS.find(r=>r.id===e),o=t.value.
trim();if(s&&s.autocompleteArgument&&!o){t.value=`${s.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(s&&s.kind==="minimal"&&(!s.
requiresArgument||o)){t.value="",executeMinimalSlashCommand(e,o),t.dispatchEvent(new Event("input",{
bubbles:!0})),t.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),t.focus(),t.
dispatchEvent(new Event("input",{bubbles:!0}))}a(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
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
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}a(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,t){
const n=get(`tab-${e}`);let i=get(t);if(!n||!i)return null;for(;i.parentElement&&i.parentElement!==n;)
i=i.parentElement;return i.parentElement===n?i:get(t)}a(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const t=AI_SETTING_JUMP_TARGETS[e];if(!t){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(t.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const n=get(t.control);n&&(n.scrollIntoView({behavior:"smooth",block:"center"}),n.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const n=findSettingsJumpElement(t.tab,t.control);n?jumpToSetting(t.tab,n):switchTab(t.tab||"general")},
320)}a(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const t=e.querySelector(".msg-controls");t&&t.remove()}a(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,t,n="update"){const i=Object.
entries(e||{}),s=`settings-result-${Date.now()}`,o=n==="inspect",r=i.length?o?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
o?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
l=renderMessage(s,"assistant",r,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!l)
return;removeEphemeralMessageControls(l);const d=l.querySelector(".message-bubble");if(!d||!i.length)
return;const p=document.createElement("div");p.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([g,y])=>{const b=AI_SETTING_JUMP_TARGETS[g]||{label:g},w=document.createElement("button");w.
type="button",w.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const x=document.createElement("span");x.className="min-w-0 flex-1";const S=document.createElement(
"span");S.className="block text-xs font-bold text-blue-200",S.textContent=b.label;const T=document.createElement(
"span");T.className="block mt-0.5 text-[11px] text-gray-300 break-words",T.textContent=formatAiSettingValue(
y);const E=document.createElement("i");E.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",x.appendChild(S),x.appendChild(T),w.appendChild(x),w.appendChild(E),w.addEventListener(
"click",()=>openAiSettingJumpTarget(g)),p.appendChild(w)});const h=d.querySelector(".message-footer-\
meta");h?d.insertBefore(p,h):d.appendChild(p),scrollToBottom()}a(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const s=get(
"welcome-screen");s&&s.classList.add("hidden");const o=`settings-pending-${n}`,r=get("chat-container");
r&&(r.insertAdjacentHTML("beforeend",`<div id="${o}" class="flex justify-start mb-4 fade-in"><div cl\
ass="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-\
md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const d=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),p=get(o);if(p&&p.remove(),d&&d.status==="ok"&&d.mode==="inspect"&&d.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(d.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
d.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(d.current,t,"inspect");
return}if(d&&d.status==="ok"&&d.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
d.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(d.applied).length}\u9805\u76EE\uFF09`,"success");try{const y=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(b=>b.json());populateAiSafeFormFields(y),cacheUserSettings(y)}catch{}renderAiSettingsResultBubble(
d.applied,t);return}const h=d.message||d.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${h}`);
const g=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${h}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
g),showToast(h,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const d=get(o);d&&d.remove();const p=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(p),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
a(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}a(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const s=e.toLowerCase(),o=loadedGems.filter(b=>b.name.toLowerCase().includes(s)||b.description&&
b.description.toLowerCase().includes(s));if(o.length===0){hideGemSuggestions();return}gemSelectedIndex>=
o.length&&(gemSelectedIndex=0),n.innerHTML="",o.forEach((b,w)=>{const x=document.createElement("div");
x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===gemSelectedIndex?
"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(b.name)}</div>
                        ${b.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
b.description)}</div>`:""}
                    </div>
                `,x.onclick=()=>selectGemSuggestion(b),x.onmouseenter=()=>{gemSelectedIndex=w,showGemSuggestions(
e)},n.appendChild(x)});const r=i.getBoundingClientRect(),l=window.innerHeight,d=l-r.bottom,p=r.top,h=260,
g=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&p>d){const b=Math.min(h,p-g);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${b}px`}else{const b=Math.min(h,d-g);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${b}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}a(showGemSuggestions,
"showGemSuggestions");function selectGemSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideGemSuggestions(),
activateGem(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}a(selectGemSuggestion,"sel\
ectGemSuggestion");function browserFastModeIneligibility(e){const t=String(get("model-select")?get("\
model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!t.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(t))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeGem)return"Gems\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(currentQuote||editingMessageId)return"\u5F15\u7528\u30FB\u7DE8\u96C6\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(codingModeEnabled)return"Coding Mode\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(["enable-search","enable-url-context","enable-maps","enable-sys-prompt","enable-prompt-cache","en\
able-mcp"].some(r=>{const l=get(r);return!!(l&&l.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const i=get("thread-custom-instruction");if(i&&String(i.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const s=Array.from(browserFastLocalFiles.values());return s.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":s.reduce((r,l)=>r+Number(l.file&&l.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":s.some(r=>!r.
file||!String(r.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}a(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const s=String(i.result||""),o=s.indexOf(",");if(o<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
s.slice(o+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}a(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const s=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(o=>{o&&
s.push({thoughtSignature:String(o)})}),i.text&&s.push({text:String(i.text)});for(const o of Array.isArray(
i.images)?i.images:[])try{const r=await fetch(buildFileUrl(o.path),{credentials:"same-origin",cache:"\
no-store"});if(!r.ok)throw new Error(`HTTP ${r.status}`);const l=await r.blob();s.push({inlineData:{
mimeType:o.mime_type||l.type||"application/octet-stream",data:await fileToBase64Payload(l)}})}catch{
n++}s.length&&t.push({role:i.role,parts:s})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}a(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(){
const e=Array.from(browserFastLocalFiles.entries());for(const[t,n]of e){if(!n||!n.file||!n.rowObj)throw new Error(
"\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");
if(n.rowObj.status&&(n.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(n.file,n.rowObj))throw new Error(`${n.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);browserFastLocalFiles.delete(t)}}a(uploadBrowserFastLocalFiles,
"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){const t=get("enable-thinking");
if(!t||!t.checked)return null;const n=String(get("thinking-level")?get("thinking-level").value:"high").
toLowerCase();if(e.includes("2.5")){const s=Number(get("thinking-budget")?get("thinking-budget").value:
4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(s)?Math.max(0,Math.min(32768,Math.trunc(
s))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["MEDIUM","HIGH"].includes(i)&&(i="MEDIU\
M"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}
a(browserFastThinkingConfig,"browserFastThinkingConfig");function browserFastPythonBoxHtml(e){return`\
<div class="code-wrapper python-box collapsed" data-py-id="${e}" data-collapsed="true" data-code-key\
="${e}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Exec\
ution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" a\
ria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code"\
 data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class\
="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-alig\
n-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="pyth\
on-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div clas\
s="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext p\
ython-output"></code></pre></div></div></div>`}a(browserFastPythonBoxHtml,"browserFastPythonBoxHtml");
function updateBrowserFastPythonBox(e,t,n){if(e){if(t==="code"){const i=n==null?"":String(n),s=e.querySelector(
".python-code");s&&(s.textContent=i,s.removeAttribute("data-highlighted"),queueHighlight(e,i));const o=e.
querySelector('.copy-btn[data-copy="code"]');o&&o.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}else if(t==="output"){const i=n==null?"":String(n),s=e.querySelector(".python-output");
s&&(s.textContent=i);const o=e.querySelector('.copy-btn[data-copy="output"]');o&&o.setAttribute("dat\
a-code",encodeURIComponent(i).replace(/'/g,"%27"))}}}a(updateBrowserFastPythonBox,"updateBrowserFast\
PythonBox");async function sendBrowserFastMessage(e){const t=String(get("model-select").value||"").trim(),
n=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==t)throw new Error(
"\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const i=Array.from(browserFastLocalFiles.values()),s=[];for(const O of i)s.push({inlineData:{mimeType:O.
file.type,data:await fileToBase64Payload(O.file)}});s.push({text:e});const o={},r=browserFastThinkingConfig(
t.toLowerCase());r&&(o.thinkingConfig=r);const l={contents:[...await buildBrowserFastHistoryContents(
n.history),{role:"user",parts:s}],generationConfig:o};!!(get("enable-python")&&get("enable-python").
checked)&&(l.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden"),renderMessage(Date.now(),"user",
e,null,null,null,null,!0,null,null,null,null,null,null,null,null,!0);const p=`browser-fast-${Date.now()}`;
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${p}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const h=get(p);activeStreamingBubbleId=p,setSendBtnToStopMode(),resumeChatAutoScroll(),
abortController=new AbortController;let g="",y="";const b=[];let w=null,x=null,S=!1;const T={},E=[];
let F=null,J="";const X=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Te=!1;try{const O=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
t)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(l),signal:abortController.
signal}));if(!O.ok){const se=await O.json().catch(()=>({}));throw new Error(se&&se.error&&se.error.message?
se.error.message:`Gemini API HTTP ${O.status}`)}window.ConnectionMonitor&&(Te=!0,window.ConnectionMonitor.
operationStarted()),X&&X.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const q=O.body.getReader(),Y=new TextDecoder;let pe="";const oe=a(se=>{const W=se.split(
/\r?\n/).filter(G=>G.startsWith("data:")).map(G=>G.slice(5).trim()).join("");if(!W||W==="[DONE]")return;
const L=JSON.parse(W);if(L.error)throw new Error(L.error.message||"Gemini API error");if((Array.isArray(
L.candidates)?L.candidates:[]).forEach(G=>{(G&&G.content&&Array.isArray(G.content.parts)?G.content.parts:
[]).forEach(Z=>{if(Z&&typeof Z.thoughtSignature=="string"&&!b.includes(Z.thoughtSignature)&&b.push(Z.
thoughtSignature),Z&&Z.executableCode&&typeof Z.executableCode.code=="string"){const Q=Z.executableCode.
code;g+=`
\`\`\`python
${Q}
\`\`\`
`,F=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,J=Q,T[F]||(h.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(F)),T[F]=h.querySelector(`[data-py-id="${F}"]`)),updateBrowserFastPythonBox(
T[F],"code",Q);return}if(Z&&Z.codeExecutionResult&&typeof Z.codeExecutionResult.output=="string"){const Q=Z.
codeExecutionResult.output;g+=`
**Output:**
\`\`\`
${Q}
\`\`\`
`;const re=F||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;E.push({code:J||
"",output:Q}),T[re]||(h.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(re)),T[re]=h.querySelector(
`[data-py-id="${re}"]`)),updateBrowserFastPythonBox(T[re],"output",Q);return}const we=typeof Z.text==
"string"?Z.text:"";we&&(Z.thought===!0?y+=we:g+=we)})}),!S&&(g||y)){beginPendingToStreamTransition(h);
const G=h.querySelector(".content-area");G&&G.remove(),S=!0}y&&(x||(h.insertAdjacentHTML("afterbegin",
'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>'),
x=h.querySelector(".thought-content")),x.textContent=y),g&&(w||(w=document.createElement("div"),w.className=
"content-area prose prose-invert text-sm break-words",h.appendChild(w)),renderAiMarkdownInto(w,g,{incrementalMath:!0})),
scrollToBottom()},"consumeEvent");for(;;){const{done:se,value:W}=await q.read();if(se)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),X&&X.setPhase("receiving"),pe+=Y.decode(W,{stream:!0});const L=pe.
split(/\r?\n\r?\n/);pe=L.pop()||"",L.forEach(oe)}if(pe+=Y.decode(),pe.trim()&&oe(pe),!g.trim())throw new Error(
"Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
w&&renderAiMarkdownInto(w,g,{incrementalMath:!0}),x&&x.classList.add("collapsed"),E.length&&(g+=E.map(
se=>`
\`\`\`pyexec
${JSON.stringify(se)}
\`\`\`
`).join("")),i.length&&(X&&X.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),X&&X.setPhase("saving");const ke=collectImageUrlsForSend(),
me=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:g,thought_content:y,model:t,image_urls:ke,temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:n.parent_id||null,thought_signatures:b,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}),h),_e=await me.json().catch(()=>({}));if(!me.ok||!_e.thread_id)throw new Error(
_e.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");const ne=!currentThreadId;currentThreadId=
String(_e.thread_id),currentParentId=_e.assistant_message_id||null,currentLeafId=_e.assistant_message_id||
null,resetUploadState(),browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,skipHistory:!ne}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\
\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F","success",!1)}catch(O){if(O.name!=="AbortError"){showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9: ${O.message}`,"error",!0),get("prompt-input").value||(get("prompt-i\
nput").value=e);const q=O.message||"\u30A8\u30E9\u30FC";h&&h.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
q));try{let Y=g||"";E.length&&(Y+=E.map(_e=>`
\`\`\`pyexec
${JSON.stringify(_e)}
\`\`\`
`).join(""));const pe=buildChatErrorMarkdown(q,Y),oe=i.length?[]:collectImageUrlsForSend(),ke=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:pe,
thought_content:y||"",model:t,image_urls:oe,temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||
null,parent_id:n&&n.parent_id?n.parent_id:null,thought_signatures:b,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),h),me=await ke.
json().catch(()=>({}));if(ke.ok&&me.thread_id){const _e=!currentThreadId;currentThreadId=String(me.thread_id),
currentParentId=me.assistant_message_id||null,currentLeafId=me.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!_e}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(Y){sendClientDebugLog("error",`Browser fas\
t error persist failed: ${Y&&Y.message?Y.message:Y}`)}}}finally{Te&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),X&&X(),setSendBtnToSendMode(),activeStreamingBubbleId===p&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}a(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Vt;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const N=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(N);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const t=get("prompt-inp\
ut").value;if(pendingSlashCommand){const N=pendingSlashCommand,ae=t.trim(),Se=get("model-select")?get(
"model-select").value:null;if(N==="settings"){if(!ae){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!Se){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
ae,Se)}else executeMinimalSlashCommand(N,ae)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const n=t.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(n&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(N=>N.id===
n[1].toLowerCase())){executeMinimalSlashCommand(n[1].toLowerCase(),n[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const i=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(i&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(i)setBrowserFastModeEnabled(!1);else{const N=browserFastModeIneligibility(
t);if(!N){try{await sendBrowserFastMessage(t)}catch(ae){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${ae.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${N}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(ae){showToast(
ae.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}t.trim()&&(promptHistory.length===
0||promptHistory[0]!==t)&&(promptHistory.unshift(t),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const s=collectAttachmentItemsForSend(),o=s.map(N=>N.path),r=s.filter(N=>normalizeAttachmentSource(
N.source)==="upload").map(N=>N.path);if(o.length>ATTACHMENT_MAX_FILES){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u6DFB\u4ED8\u3092\u6E1B\u3089\u3057\u3066\u518D\u9001\u3057\u3066\u304F\u3060\u3055\u3044\u3002`,
"error",!0);return}const l=getModelMediaSupport(get("model-select").value),d=o.some(N=>isAudioPath(N)),
p=o.some(N=>isVideoPath(N)),h=(get("model-select").value||"").toLowerCase(),g=get("enable-python"),y=!!(g&&
g.checked);if(d&&!l.audio||p&&!l.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!t.trim()&&o.length===0)return;if(isMistralOcrModel(
h)){const N=/https?:\/\/\S+/i.test(t);if(o.filter(Se=>isAudioPath(Se)||isVideoPath(Se)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!o.length&&!N){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const b=t.trim();if(/^\/settings(?:\s|$)/i.
test(b)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(b)){const N=b.replace(/^\/settings\s*/i,"").trim();
if(!N){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Se=get("prompt-input");Se.value="/settings ";const Me=extractSlashCommandToken(Se.value);
lastSlashFilter=Me,showSlashCommandSuggestions(Me),Se.focus();return}const ae=get("model-select")?get(
"model-select").value:null;if(!ae){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
N,ae);return}if(isGeminiLocalPythonMode(h,d,p,y)&&!await confirmGeminiLocalPythonSwitch())return;let w=null,
x=[];if(codingModeEnabled){const N=collectCodingCandidates(t),ae=N.filter(He=>He.prompt_source),Se=N.
filter(He=>!He.prompt_source),Me=ae.reduce((He,Oe)=>He+String(Oe.code||"").length,0);if(Me>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let De=3e5-Me;const qe=[];for(let He=Se.length-1;He>=0;He--){const Oe=String(Se[He].
code||"").length;Oe>De||(qe.unshift(Se[He]),De-=Oe)}x=codingTargetSelection?qe.slice(-1):[...ae,...qe];
const nt=ae.length?ae[ae.length-1]:null;if(w=codingTargetSelection?x[0]:nt||x[x.length-1]||null,codingModeEffective=
!!(w&&String(w.code||"").trim()),codingModeEffective&&w.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const He=String(((Vt=get(
"model-select"))==null?void 0:Vt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(He)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const S=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${t.length}\
 attachments=${o.length} search=${get("enable-search").checked}`);const T=t,E=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&o.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const F=editingMessageId,J=currentParentId,X=F!=null;F&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Te=[],O=a(N=>{if(N==null)
return;let ae=document.getElementById(`msg-${N}`);for(;ae;)ae.classList&&ae.classList.contains("mess\
age-group")&&(Te.push({node:ae,prevDisplay:ae.style.display}),ae.style.display="none"),ae=ae.nextElementSibling},
"hideRenderedBranchFrom"),q=a(()=>{Te.forEach(({node:N,prevDisplay:ae})=>{N&&(N.style.display=ae||"")}),
Te.length=0},"restoreHiddenBranch");F&&O(F);const Y=Date.now(),pe=renderMessage(Y,"user",T,JSON.stringify(
o),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,J,activeGem?activeGem.name:null);
let oe=!1;const ke=/(https?:\/\/)?(x\.com|twitter\.com)\//i,me=ke.test(T||"")||ke.test(currentQuote||
""),_e="grok-4-fast-reasoning",ne=a(()=>{get("enable-search").checked=!0,get("model-select").value!==
_e&&selectModelById(_e)},"applyXLinkAuto");if(me&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)ne();else{const N=get("auto-search-banner"),ae=get("auto-search-on-btn"),Se=get(
"auto-search-off-btn"),Me=get("auto-search-remember");N&&ae&&Se&&(Me&&(Me.checked=!1),await new Promise(
De=>{N.classList.remove("hidden");const qe=a(nt=>{N.classList.add("hidden"),ae.onclick=null,Se.onclick=
null,De(nt)},"cleanup");ae.onclick=()=>qe("enable"),Se.onclick=()=>qe("disable")}).then(async De=>{De===
"enable"?(ne(),Me&&Me.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
oe=!0}))}const se=String(get("reasoning-effort").value||"").toLowerCase(),W=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&se==="none",L={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:T,model:get("model-select").value,image_urls:o,image_items:s,uploaded_image_urls:r,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:W?!1:get("enable-thin\
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
parent_id:J,parent_id_explicit:X,disable_auto_search:oe,image_vision_model:currentVisionModel||null,
coding_mode:S,coding_target:S?{id:w.candidate_id,code:w.prompt_source?null:w.code,language:w.language||
"text",key:w.key||null,message_id:w.message_id||null,source:w.prompt_source?"prompt":"history",explicit:w.
explicit===!0}:null,coding_candidates:S?x.map(N=>({id:N.candidate_id,source:N.prompt_source?"prompt":
"history",prompt_index:N.prompt_source?N.prompt_index:null,code:N.prompt_source?null:N.code,language:N.
language||"text",explicit:N.explicit===!0})):[],batch_mode:i};e&&(L.turnstile_token=e);const R=get("\
thread-custom-instruction");R&&(L.thread_custom_instruction=R.value||""),activeGem?(L.system_prompt=
activeGem.instruction,L.enable_system_prompt=!0,L.gem_uuid=activeGem.uuid):L.gem_uuid=null,setSendBtnToStopMode();
const G="ai-"+Date.now(),K=String(L.model||"").toLowerCase(),Z=!!L.enable_thinking||!!se&&se!=="none",
we=K.includes("gemini")||K.includes("o1")||K.includes("o3")||K.includes("gpt-5")||K.includes("reason\
ing")&&!K.includes("non-reasoning"),Q=Z&&we;let re=buildPendingSkeletonHtml(L.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${G}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${re}</div></div>`),resumeChatAutoScroll();const U=get(G);activeStreamingBubbleId=
G,canvasModeEnabled&&resetCanvasPreviewPanel();let he=null;const tt=a(N=>!Q||!U?null:((!he||!U.contains(
he))&&(he=U.querySelector(".thought-content")),he||(U.insertAdjacentHTML("afterbegin",'<div class="t\
hought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i cla\
ss="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" \
data-placeholder="1"></div></div>'),he=U.querySelector(".thought-content")),he&&(he.setAttribute("da\
ta-placeholder","1"),he.textContent=N||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
he),"ensureThoughtPlaceholder");Q&&tt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const Ce=currentThreadId,st=nowPerfMs(),ut=Date.now();let Je=!1,
rt=!1,wt=!1,gt=null,xt=null,lt=null,It=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Pt=a((N,ae)=>{if(!ae||N==="status"&&Je||N==="thought"&&rt||N==="content"&&wt)return;const Se=Math.
max(0,nowPerfMs()-st);N==="status"?gt=Se:N==="thought"?xt=Se:N==="content"&&(lt=Se),reportFirstTokenLatency(
{latency_seconds:Se/1e3,latency_ms:Se,thread_id:It||currentThreadId,job_id:currentJobId,model:L.model,
first_event_type:N,client_sent_at_ms:ut}),N==="status"?Je=!0:N==="thought"?rt=!0:N==="content"&&(wt=
!0)},"maybeReportFirstEventLatency"),at=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let pt=!1,kt=!1,_t=null,mt=null,zt=!1;try{L.thread_id&&activeGem&&(threadGemMap[L.thread_id]=
activeGem,pendingGemForNewThread=null);const N=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(L),signal:abortController.signal}),U);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${N.status}`),!N.ok){const Le=await N.json().catch(()=>({})),Be=new Error(Le.error||
`HTTP ${N.status}`);throw Be.serverCode=Le.code||null,Be.serverModel=Le.model||L.model,Be.acceptedJobId=
Le.job_id||null,Be.acceptedThreadId=Le.thread_id||null,Be}pt=!0,window.ConnectionMonitor&&(zt=!0,window.
ConnectionMonitor.operationStarted()),at&&at.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const ae=a(()=>{if(!U)return;const Le=U.querySelector(
".content-area");if(Le&&Le.getAttribute("data-api-accepted")!=="1"&&(Le.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(U,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
Le.outerHTML=buildPendingSkeletonHtml(L.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const Be=U.querySelector(".content-area");Be&&Be.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
U,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");ae();
const Se=N.body.getReader(),Me=new TextDecoder;let De="",qe="",nt="",He=!0,Oe=null,Re=null,Ge=null,Wt=!1;
const Ot={};let et=0,Nt=!1;for(;!Nt;){const{done:Le,value:Be}=await Se.read();if(Le)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),at&&at.setPhase("receiving"),De+=Me.decode(Be,{stream:!0});
let Ye=De.split(`
`);De=Ye.pop();let Jt=!1,on=!1;for(let ct of Ye)if(ct.trim())try{const le=JSON.parse(ct);if(le.type===
"thread_id"){ae();const ve=le.content!==null&&le.content!==void 0?String(le.content):le.content;ve&&
(It=ve,currentThreadId!==ve&&(currentThreadId=ve,history.pushState({},"","/c/"+ve)),activeGem&&(threadGemMap[ve]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(le.type==="job_\
id"){ae(),currentJobId=le.content,i&&showToast("Batch\u767B\u9332","info");continue}if(le.type==="se\
arch_status"){le.content==="searching"&&!Ge?(U.insertAdjacentHTML("afterbegin",'<div class="search-b\
ox visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>'),Ge=U.querySelector(
".search-box")):le.content==="done"&&Ge&&(Ge.classList.remove("animate-pulse"),Ge.innerHTML='<i clas\
s="fas fa-check-circle text-green-400"></i> Search complete',setTimeout(()=>{Ge&&Ge.remove(),Ge=null},
2e3));continue}if(le.type==="mcp"){handleMcpStreamEvent(U,le.content||{});continue}if(le.type==="mcp\
_decision_request"){openMcpDecisionModal(le.content||{});continue}if(le.type==="status"){ae();const ve=le.
content===null||le.content===void 0?"":String(le.content);if(Pt("status",!!ve),He&&U){const Ue=ve||"\
\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(U,Ue,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ze=U.querySelector(".content-area");ze&&(ze.outerHTML=buildPendingSkeletonHtml(L.model,Ue),updatePendingSkeletonStatus(
U,Ue,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
Q&&tt(ve||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(He){beginPendingToStreamTransition(
U);const ve=U.querySelector(".content-area");ve&&(ve.innerHTML=""),He=!1}if(le.type==="coding_diff")
appendCodingLiveDiff(U,le.content||{}),Pt("content",!0);else if(le.type==="thought"){if(Oe||(Oe=U.querySelector(
".thought-content")),nt+=le.content,Pt("thought",!!le.content),!Oe){const ve='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';Ge?Ge.insertAdjacentHTML(
"afterend",ve):U.insertAdjacentHTML("afterbegin",ve),Oe=U.querySelector(".thought-content")}if(Oe&&Oe.
getAttribute("data-placeholder")==="1"){if(Oe.textContent="",Oe.removeAttribute("data-placeholder"),
Oe){const ve=Oe.parentElement.querySelector(".thought-header");ve&&ve.classList.remove("thinking-shi\
mmer")}nt=le.content}Oe.classList.remove("collapsed"),on=!0}else if(le.type==="image_analysis"){const ve=le.
content===null||le.content===void 0?"":String(le.content);if(!U)continue;let Ue=U.querySelector(".im\
age-analysis-box");if(!Ue){const Qe='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';Ge?Ge.insertAdjacentHTML("afterend",Qe):U.insertAdjacentHTML("afterbegin",Qe),Ue=U.querySelector(
".image-analysis-box")}const ze=Ue.querySelector(".image-analysis-text");ze&&(ze.textContent=ve)}else if(le.
type==="python"){const ve=le.content||{},Ue=ve.id||`py_${Date.now()}`;if(!Ot[Ue]){const Qe=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Ue}" data-collapsed="true" data-code-key="${Ue}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;Ge?Ge.insertAdjacentHTML("afterend",Qe):U.insertAdjacentHTML(
"afterbegin",Qe),Ot[Ue]=U.querySelector(`[data-py-id="${Ue}"]`)}const ze=Ot[Ue];if(ze){if(ve.code!==
void 0){const Qe=ve.code==null?"":String(ve.code),dt=ze.querySelector(".python-code");dt&&(dt.textContent=
Qe,dt.removeAttribute("data-highlighted"),queueHighlight(ze,Qe));const St=ze.querySelector('.copy-bt\
n[data-copy="code"]');St&&St.setAttribute("data-code",encodeURIComponent(Qe).replace(/'/g,"%27"))}if(ve.
output!==void 0){const Qe=ve.output==null?"":String(ve.output),dt=ze.querySelector(".python-output");
dt&&(dt.textContent=Qe);const St=ze.querySelector('.copy-btn[data-copy="output"]');St&&St.setAttribute(
"data-code",encodeURIComponent(Qe).replace(/'/g,"%27"))}}}else if(le.type==="content"){const ve=le.content===
null||le.content===void 0?"":String(le.content);qe+=ve,/[`~]/.test(ve)&&activateDeferredCodingModeFromStream(
qe),Re||(Re=U.querySelector(".content-area")||document.createElement("div"),Re.className="prose pros\
e-invert text-sm break-words",U.contains(Re)||U.appendChild(Re)),Jt=!0,Pt("content",!!ve)}else if(le.
type==="error"){Wt=!0,Nt=!0,U.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(le.content)),showToast(
le.content||"Unknown error","error",!0);break}}catch{}if(on&&Oe&&(Oe.textContent=nt,userAutoScroll&&
(Oe.scrollTop=Oe.scrollHeight)),Jt&&Re){const ct=Date.now();if(ct-et>100){const le=snapshotCodeCollapse(
Re);renderAiMarkdownInto(Re,qe,{incrementalMath:!0}),applyCodeCollapse(Re,le,!0),et=ct}}scrollToBottom()}
if(at&&at(),Re){const Le=snapshotCodeCollapse(Re);renderAiMarkdownInto(Re,qe,{incrementalMath:!0}),applyCodeCollapse(
Re,Le,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),U)if(queueHighlight(U,qe),enableLatencyMetrics){
const Le=nowPerfMs()-st;reportFirstTokenLatency({is_total:!0,latency_seconds:Le/1e3,latency_ms:Le,thread_id:It||
currentThreadId,job_id:currentJobId,model:L.model,client_sent_at_ms:ut,client_done_at_ms:Date.now()});
let Be='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',Ye=null;gt!==null&&(Ye=gt),xt!==null&&(Ye===null||xt<Ye)&&(Ye=
xt),lt!==null&&(Ye===null||lt<Ye)&&(Ye=lt),Ye!==null&&(Be+=`<div>Initial: ${(Ye/1e3).toFixed(2)}s</d\
iv>`),lt!==null&&lt!==Ye&&(Be+=`<div>Content: ${(lt/1e3).toFixed(2)}s</div>`),Be+=`<div class="font-\
bold text-gray-300">Total: ${(Le/1e3).toFixed(2)}s</div>`,currentJobId&&(Be+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),Be+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,Be+="</div>",U.insertAdjacentHTML("beforeend",Be)}else U.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),U&&U.querySelectorAll(".thought-co\
ntent").forEach(Be=>Be.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0}),!Wt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(!0,{persist:!1})),userAutoScroll&&
scrollToBottom(),document.querySelectorAll(".message-group").length<=2||!currentThreadTitle||currentThreadTitle===
"New Chat"||currentThreadTitle==="No Title"?apiFetch("/api/generate_title",{method:"POST",headers:{"\
Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,model_id:get("model\
-select").value})}).then(Le=>Le.json()).then(Le=>{Le.title&&(document.title=Le.title+" - AI Chat",setCurrentChatHeaderTitle(
Le.title),loadThreads())}):loadThreads(!1)}catch(N){let ae=!1;const Se=N.name==="AbortError"&&isManualStopAbortForThread(
Ce);if(N.name==="AbortError"&&!Se&&(ae=await syncThreadAfterAbortedStream(Ce,{retries:2,retryDelayMs:180,
notifyOnFailure:!0})),sendClientDebugLog("error",`Prompt send error: ${N.message}`),!pt){pe&&pe.remove();
const Me=U&&U.closest(".fade-in");Me&&Me.remove(),delete messageStore[Y],delete messageMeta[Y]}if(N.
serverCode==="request_already_accepted"&&N.acceptedJobId&&N.acceptedThreadId)pt=!0,_t={job_id:N.acceptedJobId,
thread_id:String(N.acceptedThreadId),model:L.model},get("prompt-input").value="",get("prompt-input").
style.height="auto",resetUploadState(),clearQuote();else if(pt&&!Se)mt={job_id:normalizeJobIdForUi(currentJobId),
thread_id:currentThreadId!=null?String(currentThreadId):null,model:L.model},window.ConnectionMonitor.
setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(N.serverCode==="turnstile_required"){const Me=await getTurnstileToken();Me?(await verifyTurnstileOnServer(
Me,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(N.serverCode==="api_key_missing"){const Me=N.serverModel||L.model,De=await showApiKeyRequiredModalAsync(
Me);De==="set"?kt=!0:De==="switch"?showModal("model-modal"):showToast(N.message||`${getModelNameById(
Me)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(N.
name!=="AbortError"){const Me="Connection Error: "+N.message;showToast(Me,"error",!0)}F&&!ae&&q()}finally{
zt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),at&&at(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===G&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(_t){const N=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=_t.thread_id,(N!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(_t,currentThreadId)}
if(mt&&mt.thread_id)return reconnectPendingStreamUntilAvailable(mt,mt.thread_id);if(kt)return sendMessage()}
a(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?
String(e.model):"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const s=get(n);if(!s)
return;if(activeStreamingBubbleId=n,s.classList.add("ai-pending-bubble"),!s.querySelector(".content-\
area.skeleton-pending")){const q=s.querySelector(".content-area");q?q.outerHTML=buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):s.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const o=currentThreadId,
r=i.toLowerCase(),l=r.includes("gemini")||r.includes("o1")||r.includes("o3")||r.includes("gpt-5")||r.
includes("reasoning")&&!r.includes("non-reasoning");let d=null;const p=a(q=>!l||!s?null:((!d||!s.contains(
d))&&(d=s.querySelector(".thought-content")),d||(s.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),d=s.querySelector(".thought-content")),d&&(d.setAttribute("data-pla\
ceholder","1"),d.textContent=q||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),d),
"ensureThoughtPlaceholder");l&&p("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let h="",g="",y="",b=!0,w=null,x=null,S=null,T=!1;const E={};let F=0,J=!1;const X=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Te=!1,O=!1;try{const q=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!q.ok)throw new Error(`Resume failed (${q.status})`);window.ConnectionMonitor&&
(O=!0,window.ConnectionMonitor.operationStarted()),X&&X.setPhase("waiting");const Y=q.body.getReader(),
pe=new TextDecoder;for(;!J;){const{done:oe,value:ke}=await Y.read();if(oe)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),X&&X.setPhase("receiving"),h+=pe.decode(ke,{stream:!0});let me=h.
split(`
`);h=me.pop();let _e=!1,ne=!1;for(let se of me)if(se.trim())try{const W=JSON.parse(se);if(W.type==="\
job_id"){currentJobId=W.content||t;continue}if(W.type==="search_status"){W.content==="searching"&&!S?
(s.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),S=s.querySelector(".search-box")):W.content==="done"&&S&&(S.
classList.remove("animate-pulse"),S.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{S&&S.remove(),S=null},2e3));continue}if(W.type==="mcp"){handleMcpStreamEvent(
s,W.content||{});continue}if(W.type==="mcp_decision_request"){openMcpDecisionModal(W.content||{});continue}
if(W.type==="status"){const L=W.content===null||W.content===void 0?"":String(W.content);if(b&&s){const R=L||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(s,R,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const G=s.querySelector(".content-area");G&&(G.outerHTML=buildPendingSkeletonHtml(i,R),updatePendingSkeletonStatus(
s,R,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
l&&p(L||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(b){beginPendingToStreamTransition(
s);const L=s.querySelector(".content-area");L&&(L.innerHTML=""),b=!1}if(W.type==="coding_diff")appendCodingLiveDiff(
s,W.content||{});else if(W.type==="thought"){if(w||(w=s.querySelector(".thought-content")),y+=W.content,
!w){const L='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';S?S.insertAdjacentHTML("afterend",L):s.insertAdjacentHTML("afterbegin",L),w=s.querySelector(
".thought-content")}if(w&&w.getAttribute("data-placeholder")==="1"){if(w.textContent="",w.removeAttribute(
"data-placeholder"),w){const L=w.parentElement.querySelector(".thought-header");L&&L.classList.remove(
"thinking-shimmer")}y=W.content}w.classList.remove("collapsed"),ne=!0}else if(W.type==="image_analys\
is"){const L=W.content===null||W.content===void 0?"":String(W.content);if(!s)continue;let R=s.querySelector(
".image-analysis-box");if(!R){const K='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';S?S.insertAdjacentHTML("afterend",K):s.insertAdjacentHTML("afterbegin",K),R=s.querySelector(
".image-analysis-box")}const G=R.querySelector(".image-analysis-text");G&&(G.textContent=L)}else if(W.
type==="python"){const L=W.content||{},R=L.id||`py_${Date.now()}`;if(!E[R]){const K=`<div class="cod\
e-wrapper python-box collapsed" data-py-id="${R}" data-collapsed="true" data-code-key="${R}"><div cl\
ass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><di\
v class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B">\
<i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" t\
itle="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" dat\
a-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></b\
utton></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code<\
/div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-secti\
on"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"><\
/code></pre></div></div></div>`;S?S.insertAdjacentHTML("afterend",K):s.insertAdjacentHTML("afterbegi\
n",K),E[R]=s.querySelector(`[data-py-id="${R}"]`)}const G=E[R];if(G){if(L.code!==void 0){const K=L.code==
null?"":String(L.code),Z=G.querySelector(".python-code");Z&&(Z.textContent=K,Z.removeAttribute("data\
-highlighted"),queueHighlight(G,K));const we=G.querySelector('.copy-btn[data-copy="code"]');we&&we.setAttribute(
"data-code",encodeURIComponent(K).replace(/'/g,"%27"))}if(L.output!==void 0){const K=L.output==null?
"":String(L.output),Z=G.querySelector(".python-output");Z&&(Z.textContent=K);const we=G.querySelector(
'.copy-btn[data-copy="output"]');we&&we.setAttribute("data-code",encodeURIComponent(K).replace(/'/g,
"%27"))}}}else if(W.type==="content"){const L=W.content===null||W.content===void 0?"":String(W.content);
g+=L,/[`~]/.test(L)&&activateDeferredCodingModeFromStream(g),x||(x=s.querySelector(".content-area")||
document.createElement("div"),x.className="prose prose-invert text-sm break-words",s.contains(x)||s.
appendChild(x)),_e=!0}else if(W.type==="error"){T=!0,J=!0,s.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
W.content)),showToast(W.content||"Unknown error","error",!0);break}}catch{}if(ne&&w&&(w.textContent=
y,userAutoScroll&&(w.scrollTop=w.scrollHeight)),_e&&x){const se=Date.now();if(se-F>100){const W=snapshotCodeCollapse(
x);renderAiMarkdownInto(x,g,{incrementalMath:!0}),applyCodeCollapse(x,W,!0),F=se}}scrollToBottom()}if(X&&
X(),x){const oe=snapshotCodeCollapse(x);renderAiMarkdownInto(x,g,{incrementalMath:!0}),applyCodeCollapse(
x,oe,!0)}vibrateHelper([100,50,100]),s&&queueHighlight(s,g),s&&s.querySelectorAll(".thought-content").
forEach(ke=>ke.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(q){const Y=q.name==="AbortError"&&isManualStopAbortForThread(o);q.name==="Abor\
tError"&&!Y&&await syncThreadAfterAbortedStream(o,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),Y||
(Te=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{O&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),X&&X(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Te)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i},o)}a(resumePendingStream,"resumePendingStream");function updateThreadHighlighting(){
const e=get("thread-list");if(!e)return;e.querySelectorAll("[data-thread-id]").forEach(n=>{n.dataset.
threadId===String(currentThreadId)?n.classList.add("bg-gray-700/60","border-l-2","border-blue-500"):
n.classList.remove("bg-gray-700/60","border-l-2","border-blue-500")})}a(updateThreadHighlighting,"up\
dateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){snapshotSidebarHistory("\
loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory("loadThreads-s\
tart append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?t.value:"";
if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");return}const s=await(await apiFetch(
`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&page=${threadPage}`)).json(),o=get("th\
read-list");if(!o)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-r\
eplace-settings-open");return}const l=s&&Array.isArray(s.threads)?s.threads.length:-1,d=o.querySelectorAll(
"[data-thread-id]").length;if(l===0&&d>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(o.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const p=get("scroll-sentinel");p&&threadObserver.
observe(p)}}const r=get("scroll-sentinel");s&&Array.isArray(s.threads)?(s.threads.forEach(l=>{const d=String(
l.id),p=document.createElement("div"),h=l.is_bookmarked?"text-yellow-400":"text-gray-500",g=l.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",b=d===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";p.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${b}`,p.dataset.threadId=d,p.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${h} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${d}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(l.title||"No Title")}</spa\
n>${g}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${d}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${d}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,p.onclick=w=>{w.target.closest("button")||w.target.closest("[dat\
a-thread-actions]")||loadMessages(d)},r?o.insertBefore(p,r):o.appendChild(p)}),hasMoreThreads=!!s.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+s.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error("Faile\
d to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}a(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,s=60,o=88,r=52,l=.5,d=8;let p=0,h=!1,g=0,y=null;
const b=a(()=>get(i),"indicatorEl"),w=a(()=>{const T=b();return T?T.querySelector(".ptr-pull-label"):
null},"labelEl"),x=a(T=>{const E=b();if(!E)return;E.style.height=Math.min(T,o)+"px",E.classList.toggle(
"active",T>2),E.classList.toggle("pull-ready",T>=s);const F=w();F&&(F.textContent=T>=s?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),S=a(()=>{const T=b();T&&(T.style.height=
"0px",T.classList.remove("active","pull-ready","refreshing"),T.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",T=>{if(y){h=!1;return}if(n.scrollTop>0){h=!1;return}const E=T.
touches[0];E&&(p=E.clientY,g=0,h=!0)},{passive:!0}),n.addEventListener("touchmove",T=>{if(!h||y)return;
if(n.scrollTop>0){h=!1;return}const E=T.touches[0];if(!E)return;const F=E.clientY-p;if(F<=0){g>0&&(g=
0,x(0)),h=!1;return}const J=b();J&&!J.classList.contains("dragging")&&J.classList.add("dragging"),g=
Math.min(F*l,o),x(g),F>=d&&T.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!h||
(h=!1,y))return;const T=b();T&&T.classList.remove("dragging");const E=g>=s;if(g=0,!E){S();return}let F;
try{F=t()}catch{F=null}const J=b();if(J){J.classList.add("refreshing"),J.style.height=r+"px";const X=J.
querySelector(".ptr-pull-label");X&&(X.textContent="\u66F4\u65B0\u4E2D...")}F&&typeof F.then=="funct\
ion"?(y=F,F.catch(()=>{}).finally(()=>{y=null,S()})):(y=Promise.resolve(),setTimeout(()=>{y=null,S()},
400))}),n.addEventListener("touchcancel",()=>{h=!1,g=0,S()})}a(initPullToRefresh,"initPullToRefresh");
const initThreadPullToRefresh=a(()=>initPullToRefresh("thread-list",()=>loadThreads(!1)),"initThread\
PullToRefresh"),initGemPullToRefresh=a(()=>initPullToRefresh("gem-list",()=>loadGems()),"initGemPull\
ToRefresh"),initPullToRefreshAll=a(()=>{initThreadPullToRefresh(),initGemPullToRefresh()},"initPullT\
oRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;const mcpCardIdSelector=a(e=>"mcp_\
card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),mcpEscHtml=a(e=>String(e==null?
"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]),"mcpE\
scHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||"MCP")} / ${mcpEscHtml(e.tool_name||
e.internal_name||"")}`}a(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){if(!e)return null;
let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("div"),t.className=
"mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),e.appendChild(
t)),t}a(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){if(!e||!t||!t.
type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(e):null;if(n&&
!i)return;const s=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.querySelector(
'[data-mcp-card="'+s+'"]'))return;const o=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${s}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",o);return}if(t.type==="result"){let o=i.querySelector('[dat\
a-mcp-card="'+s+'"]');const r=t.summary||"";if(o)o.classList.remove("mcp-running"),o.classList.add("\
mcp-done"),o.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const l=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${s}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",l),o=i.querySelector('[data-mcp-card="'+s+'"]')}if(r){const l=document.
createElement("div");l.className="mcp-box-note",l.textContent=r.split(`
`)[0].slice(0,220),o&&o.appendChild(l)}return}if(t.type==="error"){let o=i.querySelector('[data-mcp-\
card="'+s+'"]');const r=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(o)o.classList.remove("mcp-running"),o.classList.add("mcp-error"),o.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const d=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${s}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",d),o=i.querySelector('[data-mcp-card="'+s+'"]')}const l=document.
createElement("div");l.className="mcp-box-note mcp-box-note-err",l.textContent=String(r).slice(0,300),
o&&o.appendChild(l);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const o=get("mcp-decision-modal");if(o&&!o.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}a(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),s=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),s){let l=e.
args_preview||"";try{const d=JSON.parse(l);l=JSON.stringify(d,null,2)}catch{}s.textContent=l}const o=get(
"mcp-decision-allow"),r=get("mcp-decision-deny");o&&(o.onclick=()=>submitMcpDecision("allow")),r&&(r.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}a(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}a(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const s=e[0],o=s.
thread_id;n.textContent=e.length===1?`${s.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),o&&await loadMessages(o)});const r=get(
"batch-notification-close");r&&(r.onclick=()=>t.classList.add("hidden"))}a(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({}));(t.active||[]).some(s=>currentThreadId&&String(s.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const i=t.
completed||[];i.length&&(showGeminiBatchCompletionBanner(i),i.some(s=>String(s.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}a(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);async function toggleBookmark(e,t){e&&e.stopPropagation(),await apiFetch(`/api/threads/${t}/boo\
kmark`,{method:"POST"}),loadThreads()}a(toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){
const n=++threadLoadSequence;window.closeHistoryModal&&window.closeHistoryModal();const i=!!t.preserveDraft,
s=!!t.silent;s||resumeChatAutoScroll({scroll:!1});const o=s?snapshotCodeCollapseByMessage(get("chat-\
container")):null;let r="",l="",d=[];if(i){const p=get("prompt-input");r=p?p.value:"",l=p?p.style.height:
"",d=currentImageUrls?currentImageUrls.slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();
currentThreadId=e!=null?String(e):e,t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),
syncActiveGemForThread(currentThreadId),get("welcome-screen").classList.add("hidden"),s||(get("chat-\
container").innerHTML=buildChatLoadingSkeletonHtml());try{const p=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",e),window.location.origin);p.searchParams.set("limit",String(getEffectiveThreadInitialMessageLimit()));
const h=await apiFetch(p.toString());if(!h.ok)throw new Error(`thread request failed (${h.status})`);
const g=await h.json();if(!g||!Array.isArray(g.messages))throw new Error("invalid thread response");
if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(g&&g.title),allMessages=g.messages,threadHasOlderMessages=
!!g.has_older_messages,oldestLoadedMessageId=g.oldest_loaded_id||(allMessages.length?allMessages[0].
id:null);const y=(allMessages||[]).filter(w=>w.role==="user"&&w.content).map(w=>w.content);if(promptHistory=
[...new Set(y.slice().reverse())],historyIndex=-1,tempPrompt="",currentThreadPending=g.pending_job||
null,setTemporaryChatUiState(!!(g&&g.is_temporary)),applyTemporaryChatRuntimeMeta(g||{}),ensureTemporaryChatHeartbeat(
!0),get("thread-custom-instruction")&&(get("thread-custom-instruction").value=g.custom_instruction||
""),g.last_model&&selectModelById(g.last_model),get("enable-prompt-cache")&&(get("enable-prompt-cach\
e").checked=!!g.enable_prompt_caching,updatePromptCacheUi()),g.last_gem_uuid&&loadedGems.length>0){const w=loadedGems.
find(x=>x.uuid===g.last_gem_uuid);w&&(threadGemMap[currentThreadId]=w,applyActiveGem(w))}const b=localStorage.
getItem(`fixed_branch_${currentThreadId}`);if(b&&allMessages.find(w=>String(w.id)===String(b))?currentLeafId=
b:allMessages.length>0?currentLeafId=allMessages[allMessages.length-1].id:currentLeafId=null,renderThreadTree(
{silent:s,keepScroll:s}),s&&o?applyCodeCollapseByMessage(get("chat-container"),o,!0):s||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!s&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),i){const w=get("prompt-input");w&&(w.value=r||"",
l?w.style.height=l:w.style.height="auto"),currentImageUrls=d,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(p){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",p),s||showChatLoadError(e),s||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}a(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const o=await(await apiFetch(i.toString())).json(),r=Array.isArray(o.messages)?o.messages:[];
if(r.length){const l=new Set(allMessages.map(p=>p.id)),d=r.filter(p=>!l.has(p.id));d.length&&(allMessages=
d.concat(allMessages))}if(threadHasOlderMessages=!!o.has_older_messages,oldestLoadedMessageId=o.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const l=e.
scrollHeight;e.scrollTop=Math.max(0,n+(l-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
a(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,s=get("chat-container");if(!s)return;let o=null;if(i&&(o=s.scrollTop),s.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const r={};allMessages.
forEach(b=>{r[b.id]=b,b.childrenIds=[]}),allMessages.forEach(b=>{b.parent_id&&r[b.parent_id]&&r[b.parent_id].
childrenIds.push(b.id)}),(!currentLeafId||!r[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const l=[];let d=r[currentLeafId];for(;d;)l.unshift(d),d=r[d.parent_id];const p=buildTokenTotals(
l),h=buildTokenTotals(allMessages),g=document.createDocumentFragment();if(threadHasOlderMessages){const b=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
w=loadingOlderMessages?"disabled":"",x=document.createElement("div");x.className="mb-3 text-center",
x.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${w}><i class="fas fa-clock-rotate-left mr-1"></i>${b}</button>`,g.appendChild(x)}
l.forEach(b=>{const w=b.parent_id?r[b.parent_id]:null,x=w?w.childrenIds:allMessages.filter(T=>!T.parent_id).
map(T=>T.id),S=x.length>1?{current:x.indexOf(b.id)+1,total:x.length,siblings:x}:null;renderMessage(b.
id,b.role,b.content,b.image_url,b.thought_data,b.model,S,n,b.quote_text,b.tokens,b.tokens_in,b.tokens_out,
b.is_encrypted,b.tokens_content,b.tokens_thought,g,!1,b.parent_id,b.gem_name,b.batch_job)});const y=currentThreadPending;
if(y&&!isPendingJobSuppressed(y.job_id)){const b=y.message_id,w=new Set(l.map(T=>T.id)),x=l.length?l[l.
length-1]:null;if(b&&w.has(b)&&currentLeafId===b||!b&&x&&x.role==="user"){const T=y.job_id?`pending-${y.
job_id}`:null;renderPendingMessage(g,n,!1,T,y.model||null)}}if(s.appendChild(g),updateTotalTokenBar(
p.tokens_total,p,h),currentParentId=currentLeafId,i&&o!==null?restoreThreadTreeScroll(s,o):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(s,s&&s.textContent||"");else if(queueHighlight(s),l.length){
const b=l[l.length-1]&&l[l.length-1].content;queueMathTypeset(s,b)}}a(renderThreadTree,"renderThread\
Tree");function restoreThreadTreeScroll(e,t){if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&
!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=
e.scrollTop,syncScrollToBottomButton()}a(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){
currentLeafId=e;const t={};allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(
i=>{i.parent_id&&t[i.parent_id]&&t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=
allMessages.length>0?allMessages[allMessages.length-1].id:null,renderThreadTree({animate:!0});return}
for(;t[n]&&t[n].childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree(
{animate:!0})}a(switchVersion,"switchVersion");async function loadGems(){try{const t=await(await apiFetch(
CHAT_CONFIG.urls.handleGems)).json();loadedGems=t;const n=get("gem-list");if(!n)return;n.innerHTML='\
<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-do\
wn ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-l\
abel"></span></div>',Array.isArray(t)&&t.forEach(i=>{const s=document.createElement("div");s.className=
"gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between it\
ems-center group",s.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa\
-gem text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><div class="flex it\
ems-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group\
-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${i.uuid}')"><i class="fas fa-p\
encil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:o\
pacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${i.uuid}')"><i class\
="fas fa-trash text-[10px]"></i></button></div>`,s.onclick=o=>{o.target.closest("button")||activateGem(
i)},n.appendChild(s)})}catch(e){console.error("Failed to load gems:",e)}}a(loadGems,"loadGems");async function openEditGemModal(e,t){
e.stopPropagation(),editingGemUuid=t;try{const i=await(await apiFetch(`/api/gems/${t}`)).json();get(
"gem-name").value=i.name,get("gem-desc").value=i.description||"",get("gem-inst").value=i.instruction,
get("gem-default-model").value=i.default_model||"",renderGemFixedPromptsForEdit(i.fixed_prompts),get(
"gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-b\
tn").innerText="Save Changes",showModal("gem-modal"),location.pathname!=="/gem"&&history.pushState({
modal:"gem"},"","/gem")}catch{showToast("Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}a(openEditGemModal,"openEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.
urls.handleGems,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,
instruction:t})}),loadGems()}a(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get(
"fixed-prompts-bar");if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),
get("active-gem-name").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),
t){t.innerHTML="";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}
n.length>0?(t.classList.remove("hidden"),n.forEach((i,s)=>{const o=document.createElement("button");
o.className="fixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover\
:bg-gray-600 text-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex item\
s-center",o.style.animationDelay=`${s*40}ms`,o.textContent=String(i.name||""),o.onclick=()=>{const r=get(
"prompt-input");r&&(r.value=i.content,r.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(
o)})):t.classList.add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML=
"",t.classList.add("hidden"));get("sys-prompt-option").style.opacity="1"}a(applyActiveGem,"applyActi\
veGem");function syncActiveGemForThread(e){const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(
t)}a(syncActiveGemForThread,"syncActiveGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({last_gem_uuid:t,thread_id:e})})}catch{}}a(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){
currentThreadId?(threadGemMap[currentThreadId]=e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\
\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,"success"),t||saveThreadGemUuid(currentThreadId,e?
e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(e),allMessages&&allMessages.length>0&&startNewChat(
{preserveGem:!0}))}a(activateGem,"activateGem");function clearActiveGem(){currentThreadId&&(delete threadGemMap[currentThreadId],
saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=null,applyActiveGem(null)}a(clearActiveGem,
"clearActiveGem");function addGemFixedPromptRow(e="",t=""){const n=get("gem-fixed-prompts-container");
if(!n)return;const i=document.createElement("div");i.className="flex gap-2 items-start gem-fixed-pro\
mpt-row ui-enter",i.innerHTML=`
                <input type="text" class="gem-fp-name bg-gray-900 border border-gray-600 rounded p-1\
.5 text-white text-[10px] w-24" placeholder="\u540D\u524D" value="${escapeHtml(e)}" autocomplete="of\
f" spellcheck="false">
                <textarea class="gem-fp-content flex-1 bg-gray-900 border border-gray-600 rounded p-\
1.5 text-white text-[10px] h-9 resize-none" placeholder="\u30D7\u30ED\u30F3\u30D7\u30C8\u5185\u5BB9" spellcheck="false">${escapeHtml(
t)}</textarea>
                <button type="button" class="text-gray-500 hover:text-red-400 p-1.5" onclick="this.p\
arentElement.remove()"><i class="fas fa-times"></i></button>
            `,n.appendChild(i)}a(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),t=[];return e.forEach(n=>{const i=n.querySelector(
".gem-fp-name").value.trim(),s=n.querySelector(".gem-fp-content").value.trim();i&&s&&t.push({name:i,
content:s})}),t.length>0?JSON.stringify(t):null}a(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const t=get("gem-fixed-prompts-container");if(t){t.innerHTML="";try{e&&JSON.parse(e).forEach(i=>addGemFixedPromptRow(
i.name,i.content))}catch{}}}a(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}a(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}a(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=[
"sidebar-chat-title","mobile-chat-title"],s=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],o=["sidebar-chat-ttl","mobile-chat-ttl"];i.forEach(r=>{const l=get(r);l&&(l.textContent=e)}),
s.forEach(r=>{const l=get(r);l&&l.classList.toggle("hidden",!n)}),o.forEach(r=>{const l=get(r);l&&(n&&
t?(l.textContent=t,l.classList.remove("hidden")):(l.textContent="",l.classList.add("hidden")))})}a(updateCurrentChatHeaderUi,
"updateCurrentChatHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="str\
ing"?e:null,updateCurrentChatHeaderUi()}a(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){
tempChatExpiresAtMs=null,updateCurrentChatHeaderUi()}a(resetTemporaryChatExpiresAt,"resetTemporaryCh\
atExpiresAt");function applyTemporaryChatRuntimeMeta(e){if(!e||typeof e!="object")return;Object.prototype.
hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(e.timeout_seconds);let t=null;
const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.floor(n*1e3);else{const i=Number(
e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=
t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=null),updateCurrentChatHeaderUi()}
a(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
a(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,t=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let n=Number(e);return Number.isFinite(n)||(n=Number(t)),Number.isFinite(n)||(n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
n=Math.trunc(n),n<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(n=TEMP_CHAT_TIMEOUT_MIN_SECONDS),n>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(n=TEMP_CHAT_TIMEOUT_MAX_SECONDS),n}a(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
n=get("temporary-chat-welcome-desc");n&&(n.textContent=t);const i=get("temporary-chat-container");i&&
(i.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
a(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const t=get(
"set-temp-chat-timeout-seconds");t&&(t.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}a(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,t))}a(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const t=get("enable-temporary-c\
hat");t&&t.checked!==temporaryChatEnabled&&(t.checked=temporaryChatEnabled);const n=get("welcome-def\
ault-content");n&&n.classList.toggle("hidden",temporaryChatEnabled);const i=get("welcome-temporary-c\
ontent");i&&i.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}a(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}a(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}a(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const t=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
n=await t.json().catch(()=>({}));t.ok&&n&&applyTemporaryChatRuntimeMeta(n),t.ok&&n&&n.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}a(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const t=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==t)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=t,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}a(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const t=!!e;if(setTemporaryChatUiState(
t),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const n=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:t})}),i=await n.json().catch(()=>({}));if(!n.ok)throw new Error(i&&i.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(i&&i.is_temporary)),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}a(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
if(threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),resetUploadState(),stopTemporaryChatHeartbeat(),
setTemporaryChatUiState(!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,allMessages=
[],promptHistory=[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=null,
loadingOlderMessages=!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(
0),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
e.skipHistory||history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").
classList.remove("hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("threa\
d-custom-instruction").value=""),get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!1,
updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(activeGem):applyActiveGem(null),loadThreads(),
window.innerWidth<768&&get("overlay").click()}a(startNewChat,"startNewChat");let threadModalLoadSeq=0;
window.openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.
handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=i.id!==null&&i.id!==void 0?String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.
is_temporary)),setCurrentChatHeaderTitle(i&&i.title),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,t=String(currentThreadId);modalThreadId=t,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[n,i]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${t}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==t)return;if(n.ok){const s=await n.
json(),o=get("thread-app-global-sys-prompt-preview");o&&(o.value=s.global_system_prompt_effective||"");
const r=get("thread-app-global-sys-prompt-preview-status");r&&(s.global_system_prompt_enabled===!1?r.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":s.global_system_prompt_uses_time_fallback?
r.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
r.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=s.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=s.system_prompt_enabled!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=s.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",s.auto_system_prompt_notices_config||{})}if(i.ok){const s=await i.json();if(e!==threadModalLoadSeq||
modalThreadId!==t)return;const o=get("thread-custom-instruction");o&&(o.value=s.custom_instruction||
"");const r=get("thread-include-global-instruction");r&&(r.checked=s.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),s=i?i.value:"",o=get("thread-include-global-instruction"),r=o?o.checked:
!0,l=get("thread-global-sys-prompt"),d=get("thread-global-sys-prompt-enabled");let p=null;try{p=l||d?
{system_prompt:l?l.value:"",system_prompt_enabled:d?d.checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:collectAutoSystemPromptConfigFromForm(
"thread")}:null}catch(h){sendClientDebugLog("error","Payload construction failed: "+h.message)}try{sendClientDebugLog(
"info","Starting PUT request for thread: "+e);const h=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:s,include_global_instruction:r})});
sendClientDebugLog("info","PUT request finished, status: "+h.status);let g=!0;if(p){sendClientDebugLog(
"info","Starting POST request for user settings");const y=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});g=y.ok,sendClientDebugLog(
"info","POST request finished, status: "+y.status)}h.ok&&g?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\
\u308C\u307E\u3057\u305F","success")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(h){sendClientDebugLog("error","Save failed with error: "+h.message),showToast("\u30A8\u30E9\u30FC\
: "+h.message,"error",!0)}finally{t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=
()=>{syncCompressionSettingsUi(),showModal("compression-modal"),location.pathname!=="/compression"&&
history.pushState({modal:"compression"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal(
"compression-modal"),!e&&location.pathname==="/compression"&&history.back()},get("save-compression-s\
ettings-btn").onclick=()=>{const e=get("compression-max-size").value,t=get("compression-max-dim").value,
n=get("compression-output-type").value,i=get("compression-format-only").checked;setCompressionSettings(
e,t,n,i);const s=a((r,l)=>{get(r)&&get(l)&&(get(l).value=get(r).value)},"syncBack");s("modal-gpt-ima\
ge-size","gpt-image-size"),s("modal-gpt-image-quality","gpt-image-quality"),s("modal-gpt-image-forma\
t","gpt-image-format"),s("modal-gpt-image-compression","gpt-image-compression"),s("modal-gemini-imag\
e-aspect","gemini-image-aspect"),s("modal-gemini-image-size","gemini-image-size"),s("modal-grok-imag\
e-aspect","grok-image-aspect"),s("modal-grok-image-resolution","grok-image-resolution"),s("modal-gro\
k-image-quality","grok-image-quality"),s("modal-ocr-table-format","ocr-table-format"),s("modal-ocr-p\
ages","ocr-pages");const o=a((r,l)=>{get(r)&&get(l)&&(get(l).checked=get(r).checked)},"syncBackChk");
o("modal-ocr-extract-header","ocr-extract-header"),o("modal-ocr-extract-footer","ocr-extract-footer"),
o("modal-ocr-include-blocks","ocr-include-blocks"),o("modal-ocr-include-images","ocr-include-images"),
window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","s\
uccess")};async function deleteGem(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.
urls.handleGemItem.replace("0",t),{method:"DELETE"}),loadGems())}a(deleteGem,"deleteGem");async function renameThread(e,t){
e.stopPropagation();const n=prompt("Title:");if(n){const i=await apiFetch(CHAT_CONFIG.urls.updateTitle.
replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:n})}),
s=await i.json().catch(()=>({}));i.ok&&currentThreadId===String(t)&&setCurrentChatHeaderTitle(s&&s.title||
n),loadThreads()}}a(renameThread,"renameThread");async function deleteThread(e,t){e.stopPropagation(),
confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),
currentThreadId===t?startNewChat():loadThreads())}a(deleteThread,"deleteThread");async function deleteMessage(e){
confirm("Delete this message and subsequent history?")&&(await apiFetch(CHAT_CONFIG.urls.deleteMessage.
replace("0",e),{method:"DELETE"}),loadMessages(currentThreadId))}a(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=a(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=a(
e=>{if(!e)return"";try{const t=new Date(e);return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(t)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=a(e=>{if(!e)
return"";let t=String(e).trim();if(!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=a(e=>{const t=pdfNormalizeAttachmentPath(e);return t?
`${window.location.origin}/files/${encodeURI(t)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=a(
e=>{const t=pdfNormalizeAttachmentPath(e);return t?`${window.location.origin}/${PDF_IMAGE_EXTS.has((t.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(t)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=a(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const s=n&&n.filename?n.
filename:i.split("/").pop(),o=n&&n.source?String(n.source):"attachment",r=!!(n&&n.is_image),l=n&&n.url?
n.url:buildPdfAttachmentUrl(i),d=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:s,source:o,isImage:r,url:l,previewUrl:d}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=a(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],s=n.some(d=>maybeNeedsMathJax(d.content)||maybeNeedsMathJax(d.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",o=t.title||"AI Chat",r=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],l=n.map(d=>{const p=d.
role==="user",h=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",g=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",y=p?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(d.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(d.content||"")}</div>`,b=buildPdfMessageAttachments(d),w=b.length?`<div \
class="attachments">${b.map(T=>T.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(T.previewUrl)}\
" alt="${pdfEscapeAttr(T.filename)}"><div class="file-caption">${pdfEscapeAttr(T.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(T.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
T.filename)}</span><span class="file-source">${pdfEscapeAttr(T.source)}</span></span></a></div>`).join(
"")}</div>`:"",x=[];d.model&&!p&&x.push(d.model),d.tokens!==null&&d.tokens!==void 0&&x.push(`tokens:${d.
tokens}`),d.tokens_in!==null&&d.tokens_in!==void 0&&x.push(`in:${d.tokens_in}`),d.tokens_out!==null&&
d.tokens_out!==void 0&&x.push(`out:${d.tokens_out}`),d.tokens_thought!==null&&d.tokens_thought!==void 0&&
x.push(`thought:${d.tokens_thought}`),d.is_encrypted&&x.push("encrypted"),d.parent_id!==null&&d.parent_id!==
void 0&&x.push(`parent:#${d.parent_id}`);const S=x.length?`<div class="message-meta">${pdfEscapeAttr(
x.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${p?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${p?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${p?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(d.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${h}
                            ${y}
                            ${g}
                            ${w}
                            ${S}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(o)} - PDF Export</title>
        ${s}
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
        <h1>${pdfEscapeAttr(o)}</h1>
        <p>\u30B9\u30EC\u30C3\u30C9 ID: ${pdfEscapeAttr(t.public_id||"")}\u3002\u8868\u793A\u4E2D\u306E\u5C65\u6B74\u3092\u305D\u306E\u307E\u307E\u5370\u5237\u3067\u304D\u308B\u3088\u3046\u306B\u3001\u753B\u9762\u30AD\u30E3\u30D7\u30C1\
\u30E3\u3067\u306F\u306A\u304F\u5168\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u518D\u69CB\u6210\u3057\u3066\u51FA\u529B\u3057\u3066\u3044\u307E\u3059\u3002</p>
        <div class="meta-grid">
        ${r.map(d=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(d.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(d.value)}</div></div>`).join("")}
        </div>
        </section>
        <main id="pdf-message-list" class="message-list">${l||'<div class="meta-card" style="margin-\
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
"error",!0);return}const s=await i.json().catch(()=>null);if(t.update(30),!s){activePdfPrintFrame=null,
t&&t.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u89E3\u6790\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const o=document.createElement("iframe");activePdfPrintFrame=o,o.setAttribute("ar\
ia-hidden","true"),o.style.position="fixed",o.style.right="0",o.style.bottom="0",o.style.width="1px",
o.style.height="1px",o.style.opacity="0",o.style.pointerEvents="none",o.style.border="0";let r=null;
const l=a(()=>{r&&(clearTimeout(r),r=null),t&&t.remove(),(activePdfPrintFrame===o||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{o.parentNode&&o.parentNode.removeChild(o)}catch{}},"cleanup");r=setTimeout(
()=>{activePdfPrintFrame===o&&(console.log("PDF print cleanup fallback triggered"),l())},6e4),o.onload=
async()=>{try{const h=o.contentDocument,g=o.contentWindow;if(!h||!g){l(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(s&&s.messages)?s.messages:[]).
some(T=>maybeNeedsMathJax(T.content)||maybeNeedsMathJax(T.thought_text))&&(g.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),h.fonts&&h.fonts.ready)try{await h.fonts.ready}catch{}t.update(60);const w=Array.from(h.
images||[]),x=Promise.all(w.map(T=>T.complete?Promise.resolve():new Promise(E=>{T.addEventListener("\
load",E,{once:!0}),T.addEventListener("error",E,{once:!0})})));if(await Promise.race([x,new Promise(
T=>setTimeout(T,5e3))]),t.update(80),h.getElementById("MathJax-script")){let T=0;for(;T<100&&(!g.MathJax||
typeof g.MathJax.typesetPromise!="function");)await new Promise(E=>setTimeout(E,50)),T++;if(g.MathJax&&
typeof g.MathJax.typesetPromise=="function")try{await g.MathJax.typesetPromise()}catch(E){console.error(
"PDF MathJax typeset failed",E)}}t.update(95),setTimeout(()=>{try{g.focus(),g.addEventListener("afte\
rprint",()=>{l()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),g.print()}catch{l(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{l(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(s),p=new Blob([d],{type:"text/html"});o.src=URL.createObjectURL(
p),document.body.appendChild(o)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}a(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}a(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}a(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),s=a((d,p)=>i.
compare(d.filename||"",p.filename||""),"nameAsc"),o=a((d,p)=>i.compare(p.filename||"",d.filename||""),
"nameDesc"),r=a((d,p)=>(Number(p.ts)||0)-(Number(d.ts)||0),"tsDesc"),l=a((d,p)=>(Number(d.ts)||0)-(Number(
p.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,p)=>s(d,p)||r(d,p)):t==="name_desc"?n.sort((d,p)=>o(
d,p)||r(d,p)):t==="oldest"?n.sort((d,p)=>l(d,p)||s(d,p)):n.sort((d,p)=>r(d,p)||s(d,p)),n}a(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}a(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}a(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}a(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}a(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const p=t.querySelector(".lib-empty-state");p&&p.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const p=get("lib-total-count");p&&(p.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
s=getLibSearchQuery(),o=i.filter(p=>lib.favoritesOnly&&!p.is_favorite?!1:!s||fileNameForSearch(p).includes(
s)),r=get("lib-total-count");if(r){const p=Number(lib.totalCount)||lib.files.length;lib.hasMore||s||
lib.favoritesOnly?r.innerText=`${lib.files.length} / ${p} files`:r.innerText=`${p} files`}if(!o.length){
if(n)return;const p=lib.favoritesOnly&&!s?"fa-star":"fa-search",h=lib.favoritesOnly&&!s?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",g=lib.favoritesOnly&&
!s?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${p}"></i></div>\
<p class="lib-empty-title">${h}</p><p class="lib-empty-sub">${g}</p></div>`;return}let l=0;(n?sortLibraryFiles(
e).filter(p=>lib.favoritesOnly&&!p.is_favorite?!1:!s||fileNameForSearch(p).includes(s)):o).forEach(p=>{
try{const h=renderLibraryItem(p,l++);t.appendChild(h)}catch{}})}a(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
s=(n?t.filter(d=>fileNameForSearch(d).includes(n)):t).filter(d=>d.type==="image"),o=lib.favoritesOnly?
s.filter(d=>d.is_favorite):s;if(!o.length)return;const r=o.map(d=>({url:d.url,filename:d.filename||d.
original_filename||d.url.split("/").pop(),element:null}));let l=r.findIndex(d=>d.url===e.url);l===-1&&
(l=0),openViewerWithItems(r,l)}a(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}a(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,s=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),o=e.type==="\
image"?`<img src="${escapeHtml(i)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(s)}"></i></div><span class="lib-file-badge">${escapeHtml(s?s.toUpperCase():"FI\
LE")}</span></div>`,r=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,l=e.is_favorite?" is-favorite":"",d=e.is_favorite?"fas fa-star":"far f\
a-star",p=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
h=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${l}" title="${p}\
" aria-label="${p}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
g=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${o}</div>${r}${h}${g}`,n.
onclick=()=>{lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-se\
lected")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(x=>{x.onclick=S=>{S.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const b=n.querySelector(".lib-del-btn");b&&(b.onclick=async x=>{x.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const w=n.querySelector(".lib-favorite-btn");return w&&
(w.onclick=async x=>{x.stopPropagation(),w.disabled=!0;try{const S=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
T=await S.json().catch(()=>({}));if(!S.ok||typeof T.is_favorite!="boolean")throw new Error(T.error||
"favorite update failed");e.is_favorite=T.is_favorite,renderLibraryGrid(),showToast(T.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),w.disabled=!1}}),n}a(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}a(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",s=FILE_BASE_URL+e,o=i==="image"?FILE_THUMB_BASE_URL+
e:null,r={filename:t,original_filename:t,filepath:e,url:s,thumbnail_url:o,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(r),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
a(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(o=>o.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const s=(i||"").trim();if(!s){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const o=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:s})}),r=await o.json().catch(
()=>({}));if(!o.ok){showToast(r.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=r.filename||s,setAttachmentNameForPath(e,t.filename));const l=get(
"upload-list");l&&l.querySelectorAll("[data-filename]").forEach(d=>{d.getAttribute("data-filename")===
e&&setRowAttachmentName(d,t?t.filename:r.filename||s)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}a(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,t){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),t&&t.parentNode&&t.remove(),lib.files&&(lib.files=
lib.files.filter(n=>n.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}a(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}a(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],t=(lib.files||[]).find(s=>s&&s.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const s=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);s.searchParams.set("filepath",e);const o=await apiFetch(
s.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),r=await o.json().catch(()=>({}));
if(!o.ok)throw new Error(r.error||`HTTP ${o.status}`);const l=Array.isArray(r.chats)?r.chats:[];if(!l.
length){i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(i.innerHTML="",l.forEach(d=>{const p=document.createElement("div");p.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const h=d.updated_at?new Date(d.
updated_at).toLocaleString():"";p.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(h)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const g=p.querySelector("button");g&&(g.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(p)}),r.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}a(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const s=CHAT_CONFIG.
urls.getFilesLib;let o=null,r=!1;try{const d=getLibSortOrder(),p=getLibSearchQuery(),h=e?lib.nextOffset:
0,g=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(h),sort:d,q:p,favorites_only:lib.
favoritesOnly?"1":"0"}),y=await apiFetch(s+"?"+g.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!y.ok)throw new Error("HTTP "+y.status);o=await y.json(),r=!0}catch(d){i=d}if(!r){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let l=Array.isArray(o)?o:
o&&Array.isArray(o.files)?o.files:[];o&&!Array.isArray(o)&&(lib.totalCount=Number(o.total)||0,lib.hasMore=
!!o.has_more,lib.nextOffset=(Number(o.offset)||0)+(Number(o.limit)||l.length));try{const d=FILE_BASE_URL,
p=FILE_THUMB_BASE_URL,h=new Set(l.map(y=>y&&y.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(y=>{if(l.length>=LIBRARY_PAGE_SIZE||!y||h.has(y))return;const b=getAttachmentNameForPath(
y)||y.split("/").pop()||y,w=(b.split(".").pop()||"").toLowerCase(),x=["png","jpg","jpeg","webp","gif"].
includes(w)?"image":"file",S=x==="image"?p+y:null;l.unshift({filename:b,original_filename:b,filepath:y,
url:d+y,thumbnail_url:S,type:x,ext:w,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),h.add(y)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=l.filter(h=>h&&h.filepath&&
h.url);let p=[];if(e){const h=new Set(lib.files.map(g=>g.filepath));p=d.filter(g=>!h.has(g.filepath)),
lib.files.push(...p)}else lib.files=d;lib.files.forEach(h=>{h&&h.filepath&&setAttachmentNameForPath(
h.filepath,h.filename||h.original_filename||"")}),lib.fileSet=new Set(lib.files.map(h=>h.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
p:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}a(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}a(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(s=>{const o=isAudioPath(s),
r=isVideoPath(s);if(o&&!e.audio||r&&!e.video){o&&(t+=1),r&&(n+=1);return}const l=normalizeAttachmentPath(
s);if(!l)return;const d=(lib.files||[]).find(p=>p&&p.filepath===s);d&&d.filename&&setAttachmentNameForPath(
l,d.filename),currentImageUrls.includes(l)||currentImageUrls.push(l),setAttachmentSourceForPath(l,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const s=[];t&&s.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&s.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${s.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}a(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(t=>{const n=(lib.
files||[]).find(i=>i&&i.filepath===t);if(n&&n.url){const i=document.createElement("a");i.href=n.url,
i.download=n.filename||n.original_filename||t.split("/").pop()||"file",document.body.appendChild(i),
i.click(),document.body.removeChild(i)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}a(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const t=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=t,showModal("legal-modal");const n=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!n.ok)return;const i=await n.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
i)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,t)=>{const n=decodeURIComponent(t),i=a(()=>{const s=e.getAttribute("data-copy")||"";e.innerHTML=s===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},s=>{console.error(s),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(i,2e3)})},window.copyMessage=(e,t)=>{const n=messageStore[e]||
"";copyToClipboard(n,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i \
class="fas fa-copy"></i>',2e3)},i=>{console.error(i),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(
()=>t.innerHTML='<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;
t.classList.contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}a(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}a(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let t=0,n=e;const i={};for((allMessages||[]).forEach(s=>i[s.id]=s);n&&i[n];){const s=i[n];t+=s.tokens||
Number(s.tokens_in||0)+Number(s.tokens_out||0),n=s.parent_id}return t}a(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(s=>i[s.id]=s);n&&i[n];){const s=i[n],o=s.model||"Unknown";t[o]||(t[o]={total:0,in:0,out:0,
thought:0});const r=s.tokens||Number(s.tokens_in||0)+Number(s.tokens_out||0);t[o].total+=r,t[o].in+=
Number(s.tokens_in||0),t[o].out+=Number(s.tokens_out||0),t[o].thought+=Number(s.tokens_thought||0),n=
s.parent_id}return t}a(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(s=>t[s.id]={...s,children:[]}),
allMessages.forEach(s=>{s.parent_id&&t[s.parent_id]?t[s.parent_id].children.push(t[s.id]):s.parent_id||
n.push(t[s.id])});function i(s){const o=document.createElement("div");o.className="flex flex-col ite\
ms-center mt-4";const r=document.createElement("div"),l=String(s.id)===String(currentLeafId),d=s.id===
threadFixedBranchId,p=branchLabelNames[s.id]||(s.role==="user"?"User":"AI"),h=getCumulativeTokensForNode(
s.id);if(r.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===s.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${l?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,r.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(p)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(s.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${h}</span>
                    </div>
                    ${d?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${l?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,r.onclick=g=>{g.stopPropagation(),selectedBranchNodeId=s.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},o.appendChild(r),s.children.length>0){const g=document.createElement("div");
g.className="w-px h-4 bg-gray-700",o.appendChild(g);const y=document.createElement("div");y.className=
"flex gap-4 items-start",s.children.forEach(b=>y.appendChild(i(b))),o.appendChild(y)}return o}a(i,"r\
enderNodeRecursive"),n.forEach(s=>e.appendChild(i(s)))}a(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function updateBranchDetailPane(){const e=get("branch-detail-panel"),t=get("bran\
ch-empty-panel");if(!selectedBranchNodeId||!allMessages){e.classList.add("hidden"),t.classList.remove(
"hidden");return}const n=allMessages.find(d=>d.id===selectedBranchNodeId);if(!n)return;e.classList.remove(
"hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get("br-date").innerText=n.created_at||
"-",get("br-model").innerText=n.model||"-";const i=n.tokens||Number(n.tokens_in||0)+Number(n.tokens_out||
0),s=getCumulativeTokensForNode(n.id);get("br-tokens").innerHTML=`<span title="Current message token\
s">${i}</span> <span class="text-gray-500">/</span> <span class="text-purple-400 font-bold" title="P\
ath total tokens">${s} total</span>`;const o=get("branch-model-breakdown"),r=getPerModelTokensForPath(
n.id);o.innerHTML="",Object.entries(r).sort((d,p)=>p[1].total-d[1].total).forEach(([d,p])=>{const h=document.
createElement("div");h.className="bg-gray-800/50 p-2 rounded border border-gray-700/50",h.innerHTML=
`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${d}</span>
                        <span class="text-blue-400 shrink-0">${p.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${p.in}</div>
                        <div title="Output tokens">Out: ${p.out}</div>
                        <div title="Thought/Reasoning tokens">${p.thought>0?`Th: ${p.thought}`:""}</\
div>
                    </div>
                `,o.appendChild(h)}),get("br-name-input").value=branchLabelNames[n.id]||"";const l=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(l.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
l.classList.replace("bg-amber-600","bg-gray-600")):(l.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
l.classList.replace("bg-gray-600","bg-amber-600"))}a(updateBranchDetailPane,"updateBranchDetailPane"),
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
(deleteMessage(selectedBranchNodeId),selectedBranchNodeId=null,setTimeout(()=>{renderBranchTreeVisualization(),
updateBranchDetailPane()},500))};let batchJobsCache=[],batchFilterMode="all",batchListTimer=null;function batchProviderLabel(e){
return{gemini:"Gemini",openai:"OpenAI",xai:"xAI"}[String(e||"").toLowerCase()]||e||"Batch"}a(batchProviderLabel,
"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\u9001\u4FE1\u5F85\u3061",
JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",JOB_STATE_RUNNING:"\
\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\u5B8C\u4E86",
JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\u505C\u6B62",
JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}a(batchStateLabelShort,
"batchStateLabelShort");function batchStateTone(e){const t=String(e||"").toUpperCase();return t==="J\
OB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":t==="JOB_STATE_FAILED"?
"border-red-500/40 bg-red-900/20 text-red-200":t==="JOB_STATE_CANCELLED"||t==="JOB_STATE_EXPIRED"?"b\
order-gray-500/40 bg-gray-700/30 text-gray-300":t==="JOB_STATE_CANCELLING"?"border-amber-500/40 bg-a\
mber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}a(batchStateTone,
"batchStateTone");function batchFormatTime(e){if(!e)return"";let t=String(e);!/[zZ]$/.test(t)&&!/[+-]\d\d:?\d\d$/.
test(t)&&(t+="Z");const n=new Date(t);return isNaN(n.getTime())?String(e):n.toLocaleString("ja-JP",{
month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}a(batchFormatTime,"batchFormatTime");
function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove("batch-list-enter"),
e.offsetWidth,e.classList.add("batch-list-enter"))}a(playBatchListAnimation,"playBatchListAnimation");
function renderBatchJobs(e={}){const t=get("batch-list");if(!t)return;const n=batchJobsCache.filter(
s=>batchFilterMode==="active"?!!s.is_active:batchFilterMode==="done"?!s.is_active:!0),i=get("batch-c\
ount");if(i&&(i.textContent=`${n.length}\u4EF6`),t.innerHTML="",!n.length){t.innerHTML='<div class="\
batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}n.forEach(s=>{const o=document.createElement("div");o.className=
"batch-job-card";const r=escapeHtml(s.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),l=escapeHtml(
batchProviderLabel(s.provider)),d=escapeHtml(s.model||""),p=escapeHtml(batchFormatTime(s.created_at)),
h=escapeHtml(s.status_text||""),g=batchStateTone(s.state),y=s.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",b=s.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",w=s.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';o.innerHTML=`
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${r}">${r}\
</div>
                            <div class="batch-job-meta mt-1 flex flex-wrap items-center gap-2 text-[\
10px]">
                                <span class="inline-flex items-center gap-1"><i class="fas fa-layer-\
group"></i>${l}</span>
                                <span class="truncate max-w-[16rem]">${d}</span>
                                <span><i class="fas fa-history mr-1"></i>${p}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${g}">${escapeHtml(batchStateLabelShort(
s.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${h}</div>
                    ${s.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
s.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${y}${b}${w}
                    </div>`;const x=o.querySelector("[data-batch-open]");x&&(x.onclick=()=>{window.closeBatchModal(),
loadMessages(s.thread_id)});const S=o.querySelector("[data-batch-cancel]");S&&(S.onclick=()=>cancelBatchJob(
s));const T=o.querySelector("[data-batch-delete]");T&&(T.onclick=()=>deleteBatchJob(s)),t.appendChild(
o)}),e.animate&&playBatchListAnimation()}a(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const t=await apiFetch("/api/batch/jobs");if(!t.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const n=await t.json().catch(()=>({}));batchJobsCache=Array.isArray(n.jobs)?n.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}a(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),n=await t.json().catch(()=>({}));if(!t.ok){showToast(n.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}a(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),n=await t.json().catch(()=>({}));
if(!t.ok){showToast(n.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}a(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(t=>{t.classList.toggle("is-active",t===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=a(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const s=i?get(i.inputId):null;get("api-key-modal-input").
value=s?s.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const o=get(
"api-key-modal-save-btn"),r=get("api-key-modal-fallback-btn"),l=get("api-key-modal-cancel-btn"),d=a(
()=>{o.onclick=null,r.onclick=null,l.onclick=null},"cleanup"),p=a(h=>{h.key==="Enter"&&(h.preventDefault(),
o.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",p),o.onclick=async()=>{
const h=get("api-key-modal-input").value.trim();if(!h){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const g=get(i.inputId);g&&(g.value=h);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:h})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=h)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",p),d(),t("set")},r.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",p),d(),t("switch")},l.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",p),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const h=get("api-key-modal-input");h&&h.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let s=!1;async function o(r,l){if(s||!isClientDebugLogEnabled()||l&&l[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;s=!0;const d=l.map(p=>{try{return p instanceof Error?p.stack||p.message:typeof p=="object"?JSON.
stringify(p):String(p)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(r,d)}catch{}finally{
s=!1}}a(o,"sendToServer"),console.log=function(...r){e.apply(console,r),o("log",r)},console.error=function(...r){
t.apply(console,r),o("error",r)},console.warn=function(...r){n.apply(console,r),o("warn",r)},console.
info=function(...r){i.apply(console,r),o("info",r)},window.addEventListener("error",function(r){o("e\
xception",[r.message,r.filename,r.lineno,r.colno,r.error])}),window.addEventListener("unhandledrejec\
tion",function(r){o("promise-rejection",[r.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();
