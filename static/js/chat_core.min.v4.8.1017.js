var gi=Object.defineProperty;var r=(e,n)=>gi(e,"name",{value:n,configurable:!0});const get=r(e=>document.getElementById(e),"get"),nativeConsoleLog=typeof console.log=="function"?console.
log.bind(console):function(){},nativeConsoleInfo=typeof console.info=="function"?console.info.bind(console):
nativeConsoleLog;let settingsModalLoaded=!1;const setSettingsSaveEnabled=r(e=>{const n=get("save-set\
tings-btn");n&&(n.disabled=!e,n.classList.toggle("opacity-60",!e),n.classList.toggle("cursor-not-all\
owed",!e),n.setAttribute("title",e?"":"\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u5B8C\u4E86\u5F8C\u306B\u4FDD\u5B58\u3067\u304D\u307E\u3059"))},
"setSettingsSaveEnabled");(function(){const n=r(c=>/(\/files\/thumb\/|\/files\/)/.test(String(c||"")),
"isFileUrl"),i=r(c=>fetch(c,{method:"GET",headers:{Range:"bytes=0-0"},cache:"no-store"}).then(d=>d.status).
catch(()=>-1),"fileUrlStatus");document.addEventListener("load",c=>{const d=c.target;if(!d||d.tagName!==
"IMG"||!d.classList.contains("chat-image"))return;const m=d.closest(".chat-image-frame");m&&(m.dataset.
chatImageState="loaded",m.removeAttribute("aria-busy"))},!0);const a=r((c,d)=>{const m=document.createElement(
"div");return m.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",d?m.innerHTML=
'<i class="fas fa-key" style="font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd\
34d;font-weight:700;line-height:1.3">\u6697\u53F7\u30AD\u30FC\u304C\u4E00\u81F4\u3057\u306A\u3044\u305F\u3081<br>\u95B2\u89A7\u3067\u304D\u307E\u305B\u3093</div>':
m.innerHTML='<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9\
px;color:#9ca3af;font-weight:700">\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093</div>',
c&&m.setAttribute("data-file-name",String(c)),m},"buildWarning"),o=r(c=>{const d=document.createElement(
"div");return d.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",d.innerHTML='\
<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;\
color:#bfdbfe;font-weight:700;line-height:1.3">\u4E00\u6642\u7684\u306B\u6DF7\u96D1\u3057\u3066\u3044\u307E\u3059<br>\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u304F\u3060\u3055\u3044</div>',
c&&d.setAttribute("data-file-name",String(c)),d},"buildBusyWarning"),l=r(c=>String(c||"").split("?")[0].
replace("/files/thumb/","/files/"),"fullFileUrl");document.addEventListener("error",c=>{const d=c.target;
if(!d||d.tagName!=="IMG")return;const m=d.currentSrc||d.src||"";if(!n(m)){const S=d.closest&&d.closest(
".chat-image-frame");if(S){S.dataset.chatImageState="error",S.removeAttribute("aria-busy");const L=S.
querySelector(".chat-image-loading"),M=L&&L.querySelector("span");M&&(M.textContent="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F");
const P=L&&L.querySelector("i");P&&(P.className="fas fa-image")}return}c.stopImmediatePropagation(),
c.preventDefault();const h=String(m).split("?")[0],y=d.getAttribute("data-viewer-filename")||h.split(
"/").pop(),v=r(S=>{const L=d.closest&&d.closest(".chat-image-frame");L&&(L.dataset.chatImageState="f\
ailed",L.removeAttribute("aria-busy"));const M=a(y,!!S);try{d.replaceWith(M)}catch{}},"showWarning"),
x=r(()=>{const S=d.closest&&d.closest(".chat-image-frame");S&&(S.dataset.chatImageState="busy",S.removeAttribute(
"aria-busy"));const L=o(y);try{d.replaceWith(L)}catch{}},"showBusyWarning"),w=r((S,L)=>{const M=d.cloneNode(
!1);M.setAttribute("data-file-retry",String(L));const P=S+(S.includes("?")?"&":"?")+"retry="+Date.now()+
"_"+L;M.setAttribute("src",P);try{d.replaceWith(M)}catch{}},"retryLoad"),_=r(S=>{if(S===429||S===503){
x();return}if(S===409){v(!0);return}if(S===404||S===410||S===403){v(!1);return}const L=parseInt(d.getAttribute&&
d.getAttribute("data-file-retry")||"0",10);if(L<2){w(m,L+1);return}if(m.includes("/files/thumb/")&&!d.
getAttribute("data-file-fallback")){d.setAttribute("data-file-fallback","1"),w(l(m),0);return}v(!1)},
"handleStatus");i(m).then(_).catch(()=>{const S=parseInt(d.getAttribute&&d.getAttribute("data-file-r\
etry")||"0",10);if(S<2){w(m,S+1);return}if(m.includes("/files/thumb/")){x();return}v(!1)})},!0)})();
function buildChatImageHtml(e,n={}){const i=String(e||""),a=String(n.alt||""),o=String(n.title||""),
l=String(n.viewerSrc||i),c=String(n.filename||"");if(i.startsWith("sandbox:"))return`<span class="te\
xt-xs text-gray-500" title="${escapeHtml(i)}">${escapeHtml(a)||"\uFF08\u753B\u50CF\u30C7\u30FC\u30BF\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09"}\
</span>`;const d=o?` title="${escapeHtml(o)}"`:"",m=c?` data-viewer-filename="${escapeHtml(c)}"`:"";
return`<span class="chat-image-frame" data-chat-image-state="loading" aria-busy="true"><span class="\
chat-image-loading" role="status" aria-label="\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D"><i class="fas fa-spinner fa-spin" aria-hidde\
n="true"></i><span>\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</span></span><img src="${escapeHtml(
i)}" data-viewer-src="${escapeHtml(l)}" alt="${escapeHtml(a)}"${d}${m} class="chat-image" loading="l\
azy" decoding="async" width="320" height="320"></span>`}r(buildChatImageHtml,"buildChatImageHtml");const isAdminSidebarDebugEnabled=r(
()=>{try{const e=window.CHAT_CONFIG||{};return!!(e.botConfig&&e.botConfig.isAdmin)}catch{return!1}},
"isAdminSidebarDebugEnabled"),ADMIN_SIDEBAR_DEBUG_PREFIX="[admin-sidebar]",adminSidebarDebugEntries=[],
snapshotSidebarHistory=r(e=>{if(!isAdminSidebarDebugEnabled())return null;const n=get("thread-list"),
i=get("sidebar"),a=get("settings-modal"),o=get("history-modal"),l=n?window.getComputedStyle(n):null,
c=i?window.getComputedStyle(i):null,d=n?Array.from(n.querySelectorAll("[data-thread-id]")):[],m=d[0]||
null,h=m?window.getComputedStyle(m):null;let y=null;try{y=typeof threadLoading=="boolean"?threadLoading:
null}catch{y=null}const v={t:Date.now(),reason:String(e||""),path:location.pathname,vw:window.innerWidth,
liteHtml:document.documentElement.classList.contains("performance-lite-mode"),blurHtml:document.documentElement.
classList.contains("performance-blur-disabled"),liquidBody:!!(document.body&&document.body.classList.
contains("liquid-glass-mode")),blurMode:adaptiveBlurPreferenceMode,liteEnabled:adaptiveBlurLiteEnabled,
sidebarClass:i?i.className:null,sidebarDisplay:c?c.display:null,sidebarOpacity:c?c.opacity:null,sidebarVisibility:c?
c.visibility:null,compact:!!(i&&i.classList.contains("compact")),sidebarOpen:!!(i&&i.classList.contains(
"open")),listExists:!!n,listParent:n&&n.parentElement?n.parentElement.id||n.parentElement.className:
null,listClass:n?n.className:null,listChildCount:n?n.children.length:0,listItemCount:d.length,listDisplay:l?
l.display:null,listOpacity:l?l.opacity:null,listVisibility:l?l.visibility:null,listHeight:l?l.height:
null,hideCompact:!!(n&&n.classList.contains("hide-compact")),searchLen:(()=>{const x=get("search-box");
return x?String(x.value||"").length:0})(),firstItemText:m&&m.textContent?m.textContent.trim().slice(
0,40):null,firstItemOpacity:h?h.opacity:null,firstItemDisplay:h?h.display:null,firstItemVisibility:h?
h.visibility:null,firstItemClass:m?m.className:null,settingsHidden:a?a.classList.contains("hidden"):
null,settingsOpen:a?a.classList.contains("modal-open"):null,settingsDisplay:a&&a.style.display||null,
historyHidden:o?o.classList.contains("hidden"):null,threadLoading:y};adminSidebarDebugEntries.push(v),
adminSidebarDebugEntries.length>80&&adminSidebarDebugEntries.shift();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,
e,v)}catch{}return v},"snapshotSidebarHistory"),installAdminSidebarDebugObserver=r(()=>{if(!isAdminSidebarDebugEnabled())
return;const e=get("thread-list");if(!(!e||e.dataset.adminSidebarDebugObserved==="1")){e.dataset.adminSidebarDebugObserved=
"1";try{new MutationObserver(i=>{const a=i.reduce((l,c)=>l+Array.from(c.removedNodes||[]).filter(d=>d&&
d.nodeType===1&&d.getAttribute&&d.getAttribute("data-thread-id")).length,0),o=i.reduce((l,c)=>l+Array.
from(c.addedNodes||[]).filter(d=>d&&d.nodeType===1&&d.getAttribute&&d.getAttribute("data-thread-id")).
length,0);snapshotSidebarHistory(`thread-list-mutated added=${o} removed=${a}`)}).observe(e,{childList:!0,
attributes:!0,attributeFilter:["class","style"]})}catch{}}},"installAdminSidebarDebugObserver");window.
__adminSidebarDebugDump=()=>{if(!isAdminSidebarDebugEnabled())return[];const e=adminSidebarDebugEntries.
slice();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"dump",e)}catch{}return e},window.copyAdminSidebarDebug=
async()=>{if(!isAdminSidebarDebugEnabled())return!1;const e=JSON.stringify(adminSidebarDebugEntries,
null,2);try{return navigator.clipboard&&navigator.clipboard.writeText&&await navigator.clipboard.writeText(
e),nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copied",adminSidebarDebugEntries.length,"entries"),!0}catch{
try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,"copy-failed",e)}catch{}return!1}};const ADAPTIVE_BLUR_COOKIE="\
adaptive_blur_disabled",ADAPTIVE_LITE_COOKIE="adaptive_lite_mode",ADAPTIVE_BLUR_MODE_COOKIE="adaptiv\
e_blur_mode",readCookieValue=r(e=>{try{const n=document.cookie.split(";").map(i=>i.trim()).find(i=>i.
startsWith(`${e}=`));return n?decodeURIComponent(n.slice(e.length+1)):""}catch{return""}},"readCooki\
eValue"),normalizeAdaptiveBlurMode=r(e=>["enabled","disabled","lite"].includes(e)?e:"auto","normaliz\
eAdaptiveBlurMode"),writeAdaptiveBlurCookie=r((e,n,i=31536e3)=>{try{const a=window.location.protocol===
"https:"?"; Secure":"";document.cookie=`${e}=${encodeURIComponent(n)}; Path=/; Max-Age=${i}; SameSit\
e=Lax${a}`}catch{}},"writeAdaptiveBlurCookie"),adaptiveBlurInteractionCooldownMs=3e3;let adaptiveBlurPreferenceMode=normalizeAdaptiveBlurMode(
readCookieValue(ADAPTIVE_BLUR_MODE_COOKIE)),adaptiveBlurMeasurementActive=!1,adaptiveBlurMeasurementLastAt=0,
adaptiveBlurFallbackEnabled=document.documentElement.classList.contains("performance-blur-disabled"),
adaptiveBlurLiteEnabled=document.documentElement.classList.contains("performance-lite-mode");const syncAdaptiveBlurSettingsUi=r(
()=>{const e=get("set-background-blur-mode"),n=get("background-blur-mode-status");e&&(e.value=adaptiveBlurPreferenceMode),
n&&(adaptiveBlurPreferenceMode==="lite"?n.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="enabled"?n.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u5E38\u306B\u6709\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurPreferenceMode==="disabled"?n.textContent="\u624B\u52D5\u8A2D\u5B9A\u306B\u3088\u308A\u3001\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurLiteEnabled?n.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u8CA0\u8377\u304C\u975E\u5E38\u306B\u9AD8\u3044\u305F\u3081\u3001\u73FE\u5728\u306F\u6700\u5C0F\u8CA0\u8377\u306E\u8EFD\u91CF\u8868\u793A\u3092\u9069\u7528\u3057\u3066\u3044\u307E\u3059\u3002":
adaptiveBlurFallbackEnabled?n.textContent="\u81EA\u52D5\u5224\u5B9A\u3067\u63CF\u753B\u8CA0\u8377\u3092\u691C\u51FA\u3057\u305F\u305F\u3081\u3001\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u3092\u7121\u52B9\u306B\u3057\u3066\u3044\u307E\u3059\u3002":
n.textContent="\u73FE\u5728\u306F\u80CC\u666F\u307C\u304B\u3057\u304C\u6709\u52B9\u3067\u3059\u3002\u64CD\u4F5C\u6642\u306E\u63CF\u753B\u304C\u91CD\u3044\u5834\u5408\u306F\u81EA\u52D5\u3067\u7121\u52B9\u5316\u3057\u307E\u3059\u3002")},
"syncAdaptiveBlurSettingsUi"),enableAdaptiveBlurFallback=r(()=>{adaptiveBlurPreferenceMode!=="auto"||
adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.classList.add(
"performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1"),syncAdaptiveBlurSettingsUi())},
"enableAdaptiveBlurFallback"),enableAdaptiveBlurLite=r(()=>{adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
(adaptiveBlurLiteEnabled=!0,adaptiveBlurFallbackEnabled||(adaptiveBlurFallbackEnabled=!0,document.documentElement.
classList.add("performance-blur-disabled"),writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,"1")),document.
documentElement.classList.add("performance-lite-mode"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"lite-auto-enabled"),syncAdaptiveBlurSettingsUi(),showToast("\u63CF\u753B\u8CA0\u8377\u304C\u9AD8\u3044\u305F\u3081\u3001\u8EFD\u91CF\u8868\u793A\uFF08\u6700\u5C0F\u8CA0\u8377\uFF09\u3092\u81EA\u52D5\u9069\u7528\u3057\u307E\u3057\u305F\u3002\u30BF\u30C3\u30D7\u3067\u8A2D\u5B9A\u3092\u958B\u304F",
"info",!1,openAdaptiveBlurSettingsFromToast),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"1"))},"en\
ableAdaptiveBlurLite"),openAdaptiveBlurSettingsFromToast=r(()=>{typeof window.openSettingsModal=="fu\
nction"&&window.openSettingsModal();const e=get("set-background-blur-mode"),n=get("tab-display")||get(
"tab-general");if(!(!e||!n)){for(const i of n.children)if(i.contains(e)){jumpToSetting(n.id==="tab-d\
isplay"?"display":"general",i);return}}},"openAdaptiveBlurSettingsFromToast"),applyAdaptiveBlurPreference=r(
e=>{const n=normalizeAdaptiveBlurMode(e);n!==adaptiveBlurPreferenceMode&&(adaptiveBlurPreferenceMode=
n,adaptiveBlurMeasurementActive=!1,adaptiveBlurLiteEnabled=!1,writeAdaptiveBlurCookie(ADAPTIVE_BLUR_COOKIE,
"",0),writeAdaptiveBlurCookie(ADAPTIVE_LITE_COOKIE,"",0),n==="auto"?writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,
"",0):writeAdaptiveBlurCookie(ADAPTIVE_BLUR_MODE_COOKIE,n),adaptiveBlurFallbackEnabled=n==="disabled"||
n==="lite",adaptiveBlurLiteEnabled=n==="lite",document.documentElement.classList.toggle("performance\
-blur-disabled",adaptiveBlurFallbackEnabled),document.documentElement.classList.toggle("performance-\
lite-mode",adaptiveBlurLiteEnabled),revealPersistentSidebarLists(),snapshotSidebarHistory("blur-pref\
erence-applied:"+n),syncAdaptiveBlurSettingsUi())},"applyAdaptiveBlurPreference"),isSettingsModalOpen=r(
()=>{const e=get("settings-modal");return e?e.classList.contains("modal-open")||e.classList.contains(
"modal-prep")?!0:e.classList.contains("hidden")?!1:e.style.display&&e.style.display!=="none":!1},"is\
SettingsModalOpen"),restoreThreadSearchValue=r((e,n)=>{const i=get("search-box");i&&i.value!==e&&(i.
value=e,clearTimeout(searchTimeout),snapshotSidebarHistory(n||"restored-search-box"))},"restoreThrea\
dSearchValue"),THREAD_SEARCH_INPUT_IDS=["search-box","history-search-box"],isUserInitiatedSearchInput=r(
e=>!!(e&&e.inputType),"isUserInitiatedSearchInput"),unlockThreadSearchInput=r(e=>{e&&e.hasAttribute(
"readonly")&&e.removeAttribute("readonly")},"unlockThreadSearchInput"),markThreadSearchUserEdited=r(
e=>{e&&(e.dataset.userEdited="1")},"markThreadSearchUserEdited"),discardAutofilledThreadSearch=r(e=>{
const n=get("search-box");if(!n||n.dataset.userEdited||!n.value)return;restoreThreadSearchValue("",e||
"cleared-autofill-search-box");const i=get("history-search-box");i&&!i.dataset.userEdited&&(i.value=
"")},"discardAutofilledThreadSearch"),hardenThreadSearchInputs=r(()=>{THREAD_SEARCH_INPUT_IDS.forEach(
e=>{const n=get(e);if(!n)return;const i=r(()=>unlockThreadSearchInput(n),"unlock");n.addEventListener(
"pointerdown",i),n.addEventListener("touchstart",i,{passive:!0}),n.addEventListener("keydown",i),n.addEventListener(
"focus",i)}),discardAutofilledThreadSearch("cleared-autofill-search-box-init"),[0,50,250,1e3].forEach(
e=>{setTimeout(()=>discardAutofilledThreadSearch("cleared-autofill-search-box-"+e+"ms"),e)})},"harde\
nThreadSearchInputs"),revealPersistentSidebarLists=r(()=>{document.querySelectorAll("#thread-list > \
[data-thread-id], #gem-list > .gem-item").forEach(e=>{e.classList.remove("model-list-animate","slide\
-in-animate","fade-in","opacity-0"),e.style.removeProperty("opacity"),e.style.removeProperty("transf\
orm"),e.style.removeProperty("animation"),e.style.removeProperty("animation-delay"),e.style.removeProperty(
"visibility")}),["thread-list","gem-list"].forEach(e=>{const n=get(e);n&&(n.style.removeProperty("op\
acity"),n.style.removeProperty("visibility"))}),snapshotSidebarHistory("reveal-sidebar-lists")},"rev\
ealPersistentSidebarLists"),adaptiveBlurIsBusy=r(()=>!!(activeStreamingBubbleId||document.querySelector(
".modal-overlay.modal-open, .modal-overlay.modal-prep, .modal-overlay.modal-close")),"adaptiveBlurIs\
Busy"),measureInteractionFrames=r((e=!1)=>{if(adaptiveBlurPreferenceMode!=="auto"||adaptiveBlurLiteEnabled||
adaptiveBlurMeasurementActive||document.visibilityState!=="visible")return;if(e)adaptiveBlurMeasurementLastAt=
Date.now();else{const o=Date.now();if(o-adaptiveBlurMeasurementLastAt<adaptiveBlurInteractionCooldownMs||
adaptiveBlurIsBusy())return;adaptiveBlurMeasurementLastAt=o}adaptiveBlurMeasurementActive=!0;const n=[];
let i=0;const a=r(o=>{if(document.visibilityState!=="visible"){adaptiveBlurMeasurementActive=!1;return}
if(i){const v=o-i;v<=200&&n.push(v)}if(i=o,n.length<30){requestAnimationFrame(a);return}adaptiveBlurMeasurementActive=
!1;const l=[...n].sort((v,x)=>v-x),c=Math.min(17.5,Math.max(7,l[Math.floor(l.length*.2)])),d=Math.max(
28,c*1.75),m=Math.max(44,c*2.7),h=n.filter(v=>v>=d).length,y=n.filter(v=>v>=m).length;(h>=5||h>=4&&y>=
2)&&(adaptiveBlurFallbackEnabled?enableAdaptiveBlurLite():enableAdaptiveBlurFallback())},"sampleFram\
e");requestAnimationFrame(a)},"measureInteractionFrames"),measureAdaptiveBlurAfterInteraction=r(()=>{
document.readyState!=="complete"||adaptiveBlurLiteEnabled||requestAnimationFrame(()=>{adaptiveBlurLiteEnabled||
measureInteractionFrames()})},"measureAdaptiveBlurAfterInteraction");document.addEventListener("clic\
k",e=>{const n=e.target instanceof Element?e.target:null;n&&n.closest('button, a, input, select, tex\
tarea, [role="button"], [tabindex]')&&measureAdaptiveBlurAfterInteraction()},!0);const externalScriptLoads=new Map,
loadExternalScript=r((e,n)=>{if(typeof n=="function"&&n())return Promise.resolve();if(externalScriptLoads.
has(e))return externalScriptLoads.get(e);const i=new Promise((a,o)=>{const l=document.createElement(
"script");l.src=e,l.async=!0,l.crossOrigin="anonymous",l.referrerPolicy="no-referrer",l.onload=()=>a(),
l.onerror=()=>o(new Error(`\u30E9\u30A4\u30D6\u30E9\u30EA\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F: ${e}`)),
document.head.appendChild(l)});return externalScriptLoads.set(e,i),i.catch(()=>externalScriptLoads.delete(
e)),i},"loadExternalScript"),ensurePdfLibraries=r(()=>Promise.all([loadExternalScript("/static/vendo\
r/html2canvas-pro-2.3.2.min.js",()=>typeof window.html2canvas=="function"),loadExternalScript("/stat\
ic/vendor/jspdf-2.5.1.umd.min.js",()=>!!(window.jspdf&&window.jspdf.jsPDF))]),"ensurePdfLibraries"),
ensureImageCompression=r(()=>loadExternalScript("https://cdn.jsdelivr.net/npm/browser-image-compress\
ion@2.0.2/dist/browser-image-compression.js",()=>typeof window.imageCompression=="function"),"ensure\
ImageCompression");let webauthnJsonLoad=null;const ensureWebAuthnJson=r(async()=>(window.webauthnJSON||
(webauthnJsonLoad||(webauthnJsonLoad=import("https://esm.sh/@github/webauthn-json@2.1.1").then(({create:e,
get:n})=>({create:e,get:n}))),window.webauthnJSON=await webauthnJsonLoad),window.webauthnJSON),"ensu\
reWebAuthnJson");window.DOMPurify&&window.DOMPurify.setConfig(window.CHAT_DOMPURIFY_CONFIG||{ADD_TAGS:[
"video","source"],ADD_ATTR:["controls","src","class","autoplay","loop","muted","poster","width","hei\
ght","start","type","reversed"],FORBID_TAGS:["iframe","object","embed"]});const THEME_DEFAULT="#0dd4\
bf",THEME_STORAGE_KEY="theme_color",INITIAL_THEME_COLOR=window.CHAT_CONFIG&&window.CHAT_CONFIG.initialThemeColor||
null,INITIAL_LIGHT_MODE_ENABLED=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.initialLightModeEnabled),INITIAL_LIQUID_GLASS_ENABLED=!!(window.
CHAT_CONFIG&&window.CHAT_CONFIG.initialLiquidGlassEnabled),RICH_PASTE_DEFAULT_PROMPT="\u3053\u306EPDF\u3092Markdown\
\u5F62\u5F0F\u306B\u5909\u63DB\u3057\u3001\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u306B\u66F8\u304D\u51FA\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
GEMINI_LOCAL_PY_DIALOG_KEY="gemini_local_py_dialog_enabled",COMPRESSION_SIZE_KEY="compression_max_si\
ze_mb",COMPRESSION_DIM_KEY="compression_max_dim",COMPRESSION_TYPE_KEY="compression_output_type",COMPRESSION_FORMAT_ONLY_KEY="\
compression_format_only",getCompressionMaxSizeMB=r(()=>parseFloat(localStorage.getItem(COMPRESSION_SIZE_KEY)||
"1.0"),"getCompressionMaxSizeMB"),getCompressionMaxDim=r(()=>parseInt(localStorage.getItem(COMPRESSION_DIM_KEY)||
"1920"),"getCompressionMaxDim"),getCompressionOutputType=r(()=>localStorage.getItem(COMPRESSION_TYPE_KEY)||
"original","getCompressionOutputType"),getCompressionFormatOnly=r(()=>localStorage.getItem(COMPRESSION_FORMAT_ONLY_KEY)===
"true","getCompressionFormatOnly"),IMAGE_EXTENSION_BY_MIME={"image/jpeg":".jpg","image/png":".png","\
image/webp":".webp"},imageFilenameForMime=r((e,n)=>{const i=IMAGE_EXTENSION_BY_MIME[String(n||"").toLowerCase()];
return i?`${String(e||"image").replace(/\.[^./\\]+$/,"")||"image"}${i}`:e||"image"},"imageFilenameFo\
rMime"),convertImageFormatOnly=r(async(e,n)=>{if(!e||!n||n==="original"||n===e.type)return e;await ensureImageCompression();
const i=await window.imageCompression.drawFileInCanvas(e,{fileType:n}),a=i&&i[0],o=i&&i[1];if(!o)throw new Error(
"Image conversion canvas is unavailable");let l;try{typeof o.convertToBlob=="function"?l=await o.convertToBlob(
{type:n,quality:1}):l=await new Promise((c,d)=>{o.toBlob(m=>m?c(m):d(new Error("Image conversion fai\
led")),n,1)})}finally{try{window.imageCompression.cleanupCanvasMemory(o)}catch{}try{a&&typeof a.close==
"function"&&a.close()}catch{}}return new File([l],imageFilenameForMime(e.name,n),{type:n,lastModified:e.
lastModified||Date.now()})},"convertImageFormatOnly"),setCompressionSettings=r((e,n,i,a)=>{localStorage.
setItem(COMPRESSION_SIZE_KEY,e),localStorage.setItem(COMPRESSION_DIM_KEY,n),localStorage.setItem(COMPRESSION_TYPE_KEY,
i),localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY,a)},"setCompressionSettings"),syncCompressionSettingsUi=r(
()=>{const e=get("compression-max-size"),n=get("compression-max-dim"),i=get("compression-output-type"),
a=get("compression-format-only");if(e&&(e.value=getCompressionMaxSizeMB()),n&&(n.value=getCompressionMaxDim()),
i&&(i.value=getCompressionOutputType()),a){a.checked=getCompressionFormatOnly();const y=a.checked;e&&
(e.disabled=y),n&&(n.disabled=y);const v=get("compression-size-wrap"),x=get("compression-dim-wrap");
v&&(v.style.opacity=y?"0.4":"1"),x&&(x.style.opacity=y?"0.4":"1")}const o=r((y,v)=>{get(y)&&get(v)&&
(get(v).value=get(y).value)},"sync");o("gpt-image-size","modal-gpt-image-size"),o("gpt-image-quality",
"modal-gpt-image-quality"),o("gpt-image-format","modal-gpt-image-format"),o("gpt-image-compression",
"modal-gpt-image-compression"),o("gemini-image-aspect","modal-gemini-image-aspect"),o("gemini-image-\
size","modal-gemini-image-size"),o("grok-image-aspect","modal-grok-image-aspect"),o("grok-image-reso\
lution","modal-grok-image-resolution"),o("grok-image-quality","modal-grok-image-quality"),o("ocr-tab\
le-format","modal-ocr-table-format"),o("ocr-pages","modal-ocr-pages");const l=r((y,v)=>{get(y)&&get(
v)&&(get(v).checked=get(y).checked)},"syncChk");l("ocr-extract-header","modal-ocr-extract-header"),l(
"ocr-extract-footer","modal-ocr-extract-footer"),l("ocr-include-blocks","modal-ocr-include-blocks"),
l("ocr-include-images","modal-ocr-include-images");const c=get("model-select").value,d=isGptImageModel(
c),m=isGeminiImageModel(c),h=isGrokImageModel(c);get("modal-gpt-image-options")&&get("modal-gpt-imag\
e-options").classList.toggle("hidden",!d),get("modal-gemini-image-options")&&get("modal-gemini-image\
-options").classList.toggle("hidden",!m),get("modal-grok-image-options")&&get("modal-grok-image-opti\
ons").classList.toggle("hidden",!h),get("modal-mistral-ocr-options")&&get("modal-mistral-ocr-options").
classList.toggle("hidden",!isMistralOcrModel(c))},"syncCompressionSettingsUi"),isGeminiLocalPyDialogEnabled=r(
()=>{const e=localStorage.getItem(GEMINI_LOCAL_PY_DIALOG_KEY);return e===null?!0:e==="1"||e==="true"},
"isGeminiLocalPyDialogEnabled"),setGeminiLocalPyDialogEnabled=r(e=>{localStorage.setItem(GEMINI_LOCAL_PY_DIALOG_KEY,
e?"1":"0")},"setGeminiLocalPyDialogEnabled"),syncGeminiLocalPyDialogSetting=r(()=>{const e=get("set-\
gemini-local-python-dialog");e&&(e.checked=isGeminiLocalPyDialogEnabled())},"syncGeminiLocalPyDialog\
Setting"),normalizeGeminiBackend=r(e=>{const n=String(e||"").trim().toLowerCase().replace("-","_");return n===
"vertex_ai"||n==="vertex"||n==="vertexai"?"vertex_ai":"gemini_api"},"normalizeGeminiBackend"),normalizeAdminApiKeyMode=r(
e=>{const n=String(e||"").trim().toLowerCase().replace("-","_");return n==="user_only"||n==="user"||
n==="settings"||n==="user_settings"?"user_only":"env_fallback"},"normalizeAdminApiKeyMode"),syncToggleButtons=r(
(e,n,i)=>{(e||[]).forEach(a=>{const o=a.getAttribute(i)===n;a.classList.toggle("border-cyan-400",o),
a.classList.toggle("bg-cyan-900/30",o),a.classList.toggle("text-white",o),a.classList.toggle("border\
-gray-600",!o),a.classList.toggle("bg-gray-800/70",!o)})},"syncToggleButtons"),syncAdminApiKeyModeUi=r(
()=>{const e=get("set-admin-api-key-mode"),n=get("admin-api-key-mode-note"),i=get("admin-api-key-mod\
e-status"),a=get("admin-api-key-mode-toggle");if(!e)return;const o=normalizeAdminApiKeyMode(e.value);
e.value=o,a&&!a.dataset.bound&&(a.dataset.bound="1",a.querySelectorAll("[data-admin-api-key-mode]").
forEach(l=>{l.addEventListener("click",()=>{e.value=normalizeAdminApiKeyMode(l.getAttribute("data-ad\
min-api-key-mode")),syncAdminApiKeyModeUi()})})),syncToggleButtons(a?a.querySelectorAll("[data-admin\
-api-key-mode]"):[],o,"data-admin-api-key-mode"),n&&(n.textContent=o==="user_only"?"\u901A\u5E38\u30E6\u30FC\u30B6\u30FC\u3068\u540C\u3058\u304F\u3001\u3053\u306E\u753B\u9762\u3067\
\u4FDD\u5B58\u3057\u305FAPI\u30AD\u30FC/Vertex\u8A2D\u5B9A\u306E\u307F\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002":
"\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u3068\u304D\u3060\u3051 .env \u3092\u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u5229\u7528\u3057\u307E\u3059\uFF08\u65E2\u5B9A\uFF09\u3002"),
i&&(i.textContent=o==="user_only"?"\u73FE\u5728: \u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u307F\uFF08\u63A8\u5968: \u8A2D\u5B9A\u5024\u3092\u660E\u793A\u7BA1\u7406\uFF09":
"\u73FE\u5728: .env \u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u6709\u52B9\uFF08\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306A\u3089 .env\uFF09")},
"syncAdminApiKeyModeUi"),ensureGeminiVertexCredentialsField=r(()=>{const e=get("gemini-vertex-settin\
gs");if(!e||get("set-gemini-vertex-credentials-json"))return;const n=document.createElement("div");n.
innerHTML=`
                <label class="text-xs text-gray-500 block">Vertex Service Account JSON (\u4EFB\u610F)</label>
                <textarea id="set-gemini-vertex-credentials-json" class="w-full h-28 bg-gray-800 bor\
der border-gray-600 rounded px-2 py-1 text-[11px] text-white font-mono" placeholder='{"type":"servic\
e_account", ...}'></textarea>
                <div class="text-[10px] text-gray-500 mt-1">\u672A\u5165\u529B\u6642\u306F\u30B5\u30FC\u30D0\u30FC\u5074ADC\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002\u5165\u529B\u3059\u308B\u3068\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u8A2D\u5B9A\u3060\u3051\u3067Ver\
tex\u8A8D\u8A3C\u3067\u304D\u307E\u3059\u3002</div>
            `,e.appendChild(n)},"ensureGeminiVertexCredentialsField"),syncGeminiBackendUi=r(()=>{const e=get(
"set-gemini-backend"),n=get("gemini-vertex-settings"),i=get("gemini-backend-note"),a=get("gemini-bac\
kend-status"),o=get("gemini-backend-toggle");if(!e)return;ensureGeminiVertexCredentialsField();const l=normalizeGeminiBackend(
e.value);e.value=l,o&&!o.dataset.bound&&(o.dataset.bound="1",o.querySelectorAll("[data-gemini-backen\
d]").forEach(c=>{c.addEventListener("click",()=>{e.value=normalizeGeminiBackend(c.getAttribute("data\
-gemini-backend")),syncGeminiBackendUi()})})),syncToggleButtons(o?o.querySelectorAll("[data-gemini-b\
ackend]"):[],l,"data-gemini-backend"),n&&n.classList.toggle("hidden",l!=="vertex_ai"),i&&(i.textContent=
l==="vertex_ai"?"Vertex AI \u3092\u5229\u7528\u3057\u307E\u3059\u3002Project ID / Location \u3092\u8A2D\u5B9A\u3057\u3001ADC \u307E\u305F\u306F Vertex Service Account JSON \u3092\u7528\u610F\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002":"Gemini API \u3092\u5229\u7528\u3057\u307E\u3059\u3002API Key \u3092\u8A2D\u5B9A\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
a&&(a.textContent=l==="vertex_ai"?"\u73FE\u5728: Vertex AI\uFF08Project ID / Location / \u8A8D\u8A3C\u60C5\u5831\u304C\u5FC5\u8981\uFF09":
"\u73FE\u5728: Gemini API\uFF08Gemini API Key \u3092\u4F7F\u7528\uFF09")},"syncGeminiBackendUi"),normalizeHex=r(
e=>{if(!e)return null;let n=String(e).trim();return!n||(n.startsWith("#")||(n=`#${n}`),n.length===4&&
(n=`#${n[1]}${n[1]}${n[2]}${n[2]}${n[3]}${n[3]}`),!/^#[0-9a-fA-F]{6}$/.test(n))?null:n.toLowerCase()},
"normalizeHex"),hexToRgb=r(e=>{const n=e.replace("#",""),i=parseInt(n.slice(0,2),16),a=parseInt(n.slice(
2,4),16),o=parseInt(n.slice(4,6),16);return[i,a,o]},"hexToRgb"),mix=r((e,n,i)=>Math.round(e+(n-e)*i),
"mix"),rgbToHex=r((e,n,i)=>`#${[e,n,i].map(a=>a.toString(16).padStart(2,"0")).join("")}`,"rgbToHex"),
deriveTheme=r(e=>{const[n,i,a]=hexToRgb(e),o=rgbToHex(mix(n,255,.45),mix(i,255,.45),mix(a,255,.45)),
l=rgbToHex(mix(n,255,.7),mix(i,255,.7),mix(a,255,.7)),c=rgbToHex(mix(n,0,.18),mix(i,0,.18),mix(a,0,.18)),
d=rgbToHex(mix(n,0,.32),mix(i,0,.32),mix(a,0,.32));return{base:e,light:o,lighter:l,dark:c,darker:d,rgb:`${n}\
, ${i}, ${a}`}},"deriveTheme"),applyThemeColor=r((e,n=!1)=>{const i=normalizeHex(e)||THEME_DEFAULT,a=deriveTheme(
i),o=document.documentElement;[["--theme-500",a.base],["--theme-600",a.dark],["--theme-700",a.darker],
["--theme-300",a.light],["--theme-200",a.lighter],["--theme-rgb",a.rgb]].forEach(([c,d])=>{o.style.getPropertyValue(
c).trim()!==String(d).trim()&&o.style.setProperty(c,d)}),n&&localStorage.setItem(THEME_STORAGE_KEY,i)},
"applyThemeColor"),applyLightMode=r(e=>{const n=!!e;let i=get("manual-theme-light-css");if(n&&!i){const a=window.
CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.manualLightTheme;if(!a)return;i=document.
createElement("link"),i.id="manual-theme-light-css",i.rel="stylesheet",i.href=a,document.head.appendChild(
i)}else!n&&i&&i.remove()},"applyLightMode"),syncThemeInputs=r(e=>{const n=normalizeHex(e)||THEME_DEFAULT,
i=get("set-theme-color"),a=get("set-theme-color-text");i&&(i.value=n),a&&(a.value=n),document.querySelectorAll(
"#theme-presets .theme-swatch").forEach(l=>{const c=normalizeHex(l.getAttribute("data-color"));l.classList.
toggle("active",c===n)})},"syncThemeInputs"),initThemeFromServer=r(()=>{INITIAL_LIGHT_MODE_ENABLED&&
applyLightMode(!0);const e=normalizeHex(INITIAL_THEME_COLOR);if(e){applyThemeColor(e,!1);return}const n=normalizeHex(
localStorage.getItem(THEME_STORAGE_KEY));applyThemeColor(n||THEME_DEFAULT,!1)},"initThemeFromServer"),
LIQUID_GLASS_SURFACE_SELECTOR=["#sidebar",".composer-dock","body > .flex-1 > header","#top-model-bar",
".modal-panel",".modal-glass-panel",".viewer-toolbar",".viewer-meta","#quote-bar","#slash-command-su\
ggestions","#gem-suggestions","#total-token-bar"].join(","),refreshLiquidGlassSurfaces=r(()=>{document.
querySelectorAll(LIQUID_GLASS_SURFACE_SELECTOR).forEach(e=>{e.classList.add("liquid-glass-surface"),
e.matches(".viewer-toolbar, .viewer-meta")&&e.classList.add("liquid-glass-clear");const n=e.matches(
'[data-liquid-glass-background="none"]')||!!e.closest(".liquid-glass-no-backdrop");e.classList.toggle(
"liquid-glass-no-background",n)})},"refreshLiquidGlassSurfaces"),applyLiquidGlassMode=r(e=>{document.
body&&(document.body.classList.toggle("liquid-glass-mode",!!e),e&&refreshLiquidGlassSurfaces())},"ap\
plyLiquidGlassMode");let pendingLiquidGlassPointer=null,liquidGlassPointerFrame=0,liquidGlassPointerPaintAt=0,
liquidGlassPointerSurface=null,liquidGlassPointerRect=null;const paintLiquidGlassPointer=r(e=>{if(!pendingLiquidGlassPointer||
!document.body||!document.body.classList.contains("liquid-glass-mode")){liquidGlassPointerFrame=0;return}
if(e-liquidGlassPointerPaintAt<30){liquidGlassPointerFrame=requestAnimationFrame(paintLiquidGlassPointer);
return}const n=pendingLiquidGlassPointer;pendingLiquidGlassPointer=null;const i=n.target&&n.target.closest?
n.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;if(!i){liquidGlassPointerFrame=0;return}(i!==liquidGlassPointerSurface||
!liquidGlassPointerRect)&&(liquidGlassPointerSurface=i,liquidGlassPointerRect=i.getBoundingClientRect());
const a=liquidGlassPointerRect;if(a.width&&a.height){const o=Math.max(0,Math.min(100,(n.clientX-a.left)/
a.width*100)),l=Math.max(0,Math.min(100,(n.clientY-a.top)/a.height*100));i.style.setProperty("--glas\
s-light-x",`${o.toFixed(1)}%`),i.style.setProperty("--glass-light-y",`${l.toFixed(1)}%`),liquidGlassPointerPaintAt=
e}liquidGlassPointerFrame=pendingLiquidGlassPointer?requestAnimationFrame(paintLiquidGlassPointer):0},
"paintLiquidGlassPointer");document.addEventListener("pointermove",e=>{!document.body||!document.body.
classList.contains("liquid-glass-mode")||(pendingLiquidGlassPointer={target:e.target,clientX:e.clientX,
clientY:e.clientY},liquidGlassPointerFrame||(liquidGlassPointerFrame=requestAnimationFrame(paintLiquidGlassPointer)))},
{passive:!0}),document.addEventListener("pointerout",e=>{const n=e.target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):
null;!n||e.relatedTarget&&n.contains(e.relatedTarget)||(pendingLiquidGlassPointer=null,n.style.removeProperty(
"--glass-light-x"),n.style.removeProperty("--glass-light-y"),n.classList.remove("liquid-glass-presse\
d"),n===liquidGlassPointerSurface&&(liquidGlassPointerSurface=null,liquidGlassPointerRect=null))},{passive:!0}),
document.addEventListener("pointerdown",e=>{if(!document.body||!document.body.classList.contains("li\
quid-glass-mode"))return;const n=e.target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;
n&&n.classList.add("liquid-glass-pressed")},{passive:!0});const releaseLiquidGlassPress=r(e=>{const n=e.
target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;n&&n.classList.remove("liquid-gl\
ass-pressed")},"releaseLiquidGlassPress");document.addEventListener("pointerup",releaseLiquidGlassPress,
{passive:!0}),document.addEventListener("pointercancel",releaseLiquidGlassPress,{passive:!0});let liquidGlassScrollTimer=0;
document.addEventListener("scroll",()=>{!document.body||!document.body.classList.contains("liquid-gl\
ass-mode")||(liquidGlassPointerRect=null,document.body.classList.add("liquid-glass-scrolling"),window.
clearTimeout(liquidGlassScrollTimer),liquidGlassScrollTimer=window.setTimeout(()=>{document.body&&document.
body.classList.remove("liquid-glass-scrolling")},140))},{passive:!0,capture:!0}),window.addEventListener(
"resize",()=>{liquidGlassPointerRect=null},{passive:!0});const MODAL_ANIM_MS=280,formatBytes=r(e=>{if(e==
null)return"0MB";const n=e/(1024*1024);return n<1024?`${n.toFixed(1)}MB`:`${(n/1024).toFixed(2)}GB`},
"formatBytes"),inspectSiteCacheStorage=r(async()=>{const e={cacheCount:0,entryCount:0,totalBytes:0,storageUsageBytes:null,
storageQuotaBytes:null};if("caches"in window)try{const n=await caches.keys();e.cacheCount=n.length;for(const i of n){
const a=await caches.open(i),o=await a.keys();e.entryCount+=o.length;for(const l of o)try{const c=await a.
match(l);if(!c)continue;const d=parseInt(c.headers.get("content-length")||"",10);if(Number.isFinite(
d)&&d>=0)e.totalBytes+=d;else{const m=await c.clone().blob();e.totalBytes+=m.size||0}}catch{}}}catch{}
if(navigator.storage&&navigator.storage.estimate)try{const n=await navigator.storage.estimate();e.storageUsageBytes=
Number(n.usage||0),e.storageQuotaBytes=Number(n.quota||0)}catch{}return e},"inspectSiteCacheStorage"),
loadSiteCacheUsage=r(async()=>{const e=get("site-cache-usage-text"),n=get("site-cache-usage-detail");
if(!(!e&&!n)){e&&(e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D..."),n&&(n.innerText="");try{const i=await inspectSiteCacheStorage(),
a=`\u30AD\u30E3\u30C3\u30B7\u30E5\u4F7F\u7528\u91CF: ${formatBytes(i.totalBytes)} (${i.cacheCount}\u30AD\u30E3\
\u30C3\u30B7\u30E5 / ${i.entryCount}\u4EF6)`;if(i.storageQuotaBytes){const o=Math.min(100,Math.round(
i.totalBytes/i.storageQuotaBytes*100));if(e&&(e.innerText=`${a} / \u4FDD\u5B58\u9818\u57DF\u4E0A\u9650 ${formatBytes(
i.storageQuotaBytes)} (${o}%)`),n){const l=i.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
i.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: \u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
n.innerText=`${l} / \u30D6\u30E9\u30A6\u30B6\u306E\u5B9F\u6E2C\u5024\u3067\u3059`}}else e&&(e.innerText=
a),n&&(n.innerText=i.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
i.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4E0A\u9650\u306F\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u3067\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093")}catch{
e&&(e.innerText="\u30AD\u30E3\u30C3\u30B7\u30E5\u5BB9\u91CF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),
n&&(n.innerText="")}}},"loadSiteCacheUsage");let versionUpdateCachePreferenceSavePromise=Promise.resolve();
const loadStorageUsage=r(async()=>{const e=get("storage-usage-text"),n=get("storage-usage-bar");if(!(!e||
!n)){e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D...";try{const i=await apiFetch("/api/storage",{cache:"\
no-store"});if(!i.ok)throw new Error("HTTP "+i.status);const a=await i.json(),o=Number(a.used_bytes||
0),l=Number(a.limit_bytes||0);if(a.is_unlimited||!l)e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(o)}\
 (\u7121\u5236\u9650)`,n.style.width="0%",n.style.opacity="0.5";else{const c=Math.min(100,Math.round(
o/l*100));e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(o)} / ${formatBytes(l)} (${c}%)`,n.style.width=
`${c}%`,n.style.opacity="1"}}catch{e.innerText="\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
n.style.width="0%",n.style.opacity="0.5"}}},"loadStorageUsage"),clearSiteCacheAndReload=r(async(e,n={})=>{
const{scanFirst:i=!0}=n||{},a=e?e.innerText:"";e&&(e.disabled=!0,e.innerText="\u524A\u9664\u4E2D...");
try{const o=i?await inspectSiteCacheStorage():null;await purgeCaches();const l=o?`\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5 ${formatBytes(
o.totalBytes)} \u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002`:"\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002";
showToast(`${l} \u518D\u8AAD\u307F\u8FBC\u307F\u3057\u307E\u3059\u3002`,"success"),window.setTimeout(
()=>location.reload(),900)}catch{showToast("\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u306E\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{e&&(e.disabled=!1,e.innerText=a||"\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664")}},
"clearSiteCacheAndReload"),syncVersionUpdateCachePreferenceUi=r(()=>{const e=get("version-update-cle\
ar-cache");e&&(e.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate))},"syn\
cVersionUpdateCachePreferenceUi"),saveVersionUpdateCachePreference=r(async e=>{window.CHAT_CONFIG&&(window.
CHAT_CONFIG.clearCacheOnVersionUpdate=!!e);try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({clear_cache_on_version_update:!!e})})}catch{}},
"saveVersionUpdateCachePreference");initThemeFromServer(),applyLiquidGlassMode(INITIAL_LIQUID_GLASS_ENABLED),
measureInteractionFrames(!0);const modalCloseTimers=new WeakMap,modalOpenFrames=new WeakMap,cancelModalTransitions=r(
e=>{const n=modalCloseTimers.get(e);n&&(clearTimeout(n),modalCloseTimers.delete(e));const i=modalOpenFrames.
get(e);i&&(cancelAnimationFrame(i.first),i.second&&cancelAnimationFrame(i.second),modalOpenFrames.delete(
e))},"cancelModalTransitions"),showModal=r(e=>{const n=get(e);if(!n||n.classList.contains("modal-ope\
n"))return;cancelModalTransitions(n),n.classList.remove("hidden"),n.style.display="flex",n.classList.
remove("modal-close"),n.classList.remove("modal-open"),n.classList.add("modal-prep");const i={first:0,
second:0};i.first=requestAnimationFrame(()=>{i.second=requestAnimationFrame(()=>{modalOpenFrames.delete(
n),n.classList.remove("modal-prep"),n.classList.add("modal-open")})}),modalOpenFrames.set(n,i)},"sho\
wModal");window.showModal=showModal;const hideModal=r((e,n={})=>{const i=get(e);if(!i)return;cancelModalTransitions(
i);const a=!!(n&&n.skipConfirm),o=!!(n&&n.skipReset);if(e==="camera-capture-modal"&&cameraCapturePendingFiles.
length>0&&!a&&!cameraCaptureBusy){attachCameraCapturedFiles();return}if(e==="rich-paste-modal"&&!a&&
hasRichPasteContent()&&!confirm("\u8CBC\u308A\u4ED8\u3051\u305F\u5185\u5BB9\u3092\u7834\u68C4\u3057\u3066\u9589\u3058\u307E\u3059\u304B\uFF1F"))
return;if(e==="marker-modal"&&(markerState.row=null),e==="camera-capture-modal"&&(o||resetCameraCapturePending(),
stopCameraCaptureStream()),!i.classList.contains("modal-open")){i.style.display="none",i.classList.remove(
"modal-close"),i.classList.remove("modal-prep"),i.classList.add("hidden");return}i.classList.remove(
"modal-open"),i.classList.add("modal-close");const l=setTimeout(()=>{i.style.display="none",i.classList.
remove("modal-close"),i.classList.remove("modal-prep"),i.classList.add("hidden"),modalCloseTimers.delete(
i)},MODAL_ANIM_MS);modalCloseTimers.set(i,l)},"hideModal");window.hideModal=hideModal;const RICH_PASTE_ALLOWED_TAGS=[
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
userSettingsSnapshotPromise=null,richPastePromptSaveTimer=null,richPastePromptPreferenceSyncing=!1;const getRichPasteEditor=r(
()=>get("rich-paste-storage"),"getRichPasteEditor"),getRichPasteCapture=r(()=>get("rich-paste-captur\
e"),"getRichPasteCapture"),getRichPastePrompt=r(()=>get("rich-paste-prompt"),"getRichPastePrompt"),getRichPasteUseDefaultCheckbox=r(
()=>get("rich-paste-use-default"),"getRichPasteUseDefaultCheckbox"),getRichPasteStatus=r(()=>get("ri\
ch-paste-status"),"getRichPasteStatus"),downloadBlob=r((e,n)=>{const i=URL.createObjectURL(e),a=document.
createElement("a");a.href=i,a.download=n,document.body.appendChild(a),a.click(),setTimeout(()=>{document.
body.removeChild(a),URL.revokeObjectURL(i)},100)},"downloadBlob"),getRichPasteEffectivePrompt=r((e=null)=>{
if(e&&e.rich_paste_prompt_use_custom_default){const n=String(e.rich_paste_prompt_default||"").trim();
if(n)return n}return RICH_PASTE_DEFAULT_PROMPT},"getRichPasteEffectivePrompt"),syncRichPastePromptPreferencesUi=r(
(e=null,n={})=>{const i=!!n.preservePrompt,a=getRichPastePrompt(),o=getRichPasteUseDefaultCheckbox();
o&&(o.checked=!!(e&&e.rich_paste_prompt_use_custom_default)),a&&!richPastePromptPreferenceSyncing&&!i&&
(a.value=getRichPasteEffectivePrompt(e))},"syncRichPastePromptPreferencesUi"),cacheUserSettings=r((e,n={})=>(userSettingsSnapshot=
e||null,syncRichPastePromptPreferencesUi(userSettingsSnapshot,n),userSettingsSnapshot),"cacheUserSet\
tings"),SETTINGS_LOAD_TIMEOUT_MS=15e3,fetchSettingsSnapshot=r(async()=>{const e=new AbortController,
n=setTimeout(()=>e.abort(),SETTINGS_LOAD_TIMEOUT_MS);try{const i=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,
{cache:"no-store",signal:e.signal});if(!i.ok)throw new Error("HTTP "+i.status);const a=await i.json();
if(!a||typeof a!="object")throw new Error("Invalid settings response");return cacheUserSettings(a)}finally{
clearTimeout(n)}},"fetchSettingsSnapshot"),ensureUserSettingsSnapshot=r(async()=>userSettingsSnapshot||
(userSettingsSnapshotPromise||(userSettingsSnapshotPromise=fetchSettingsSnapshot().catch(()=>null).finally(
()=>{userSettingsSnapshotPromise=null})),await userSettingsSnapshotPromise),"ensureUserSettingsSnaps\
hot"),saveRichPastePromptPreferences=r(async()=>{const e=getRichPastePrompt(),n=getRichPasteUseDefaultCheckbox();
if(!e||!n)return;const i={rich_paste_prompt_default:e.value||"",rich_paste_prompt_use_custom_default:!!n.
checked};try{await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(i)}),cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},i),{preservePrompt:!0})}catch{}},"saveRichPastePromptPreferences"),queueRichPastePromptPreferenceSave=r(
()=>{richPastePromptSaveTimer&&clearTimeout(richPastePromptSaveTimer),richPastePromptSaveTimer=setTimeout(
()=>{richPastePromptSaveTimer=null,saveRichPastePromptPreferences()},500)},"queueRichPastePromptPref\
erenceSave"),hasRichPasteContent=r(()=>{const e=getRichPasteEditor();return e?(e.textContent||"").trim()?
!0:!!e.querySelector("img,table,ul,ol,blockquote,h1,h2,h3,h4,h5,h6,pre,code"):!1},"hasRichPasteConte\
nt"),updateRichPasteStatus=r(()=>{const e=getRichPasteEditor(),n=getRichPasteStatus();if(!n||!e)return;
const i=(e.innerText||"").trim();if(!i){n.textContent="\u307E\u3060\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002";
return}const a=e.querySelectorAll("img").length,o=e.querySelectorAll("table").length,l=e.querySelectorAll(
"a").length,c=e.querySelectorAll("h1,h2,h3,h4,h5,h6").length;n.textContent=`${i.length} \u6587\u5B57 / \u753B\u50CF ${a}\
 / \u8868 ${o} / \u30EA\u30F3\u30AF ${l} / \u898B\u51FA\u3057 ${c}`},"updateRichPasteStatus"),focusRichPasteEditor=r(
()=>{const e=getRichPasteCapture();if(!e)return;e.focus(),e.value=e.value||"",window.getSelection&&window.
getSelection()&&e.select&&e.select()},"focusRichPasteEditor"),clearRichPasteEditor=r((e=!0)=>{const n=getRichPasteEditor();
n&&(n.innerHTML="");const i=getRichPasteCapture();if(i&&(i.value=""),!e){const a=getRichPastePrompt();
a&&(a.value=RICH_PASTE_DEFAULT_PROMPT)}updateRichPasteStatus()},"clearRichPasteEditor"),sanitizeRichPasteStyle=r(
e=>{if(!e)return"";const n=[];return String(e).split(";").forEach(i=>{const a=i.trim();if(!a)return;
const o=a.indexOf(":");if(o<=0)return;const l=a.slice(0,o).trim().toLowerCase(),c=a.slice(o+1).trim();
if(!RICH_PASTE_SAFE_STYLE_PROPS.has(l)||!c||c.length>1e3)return;const d=c.toLowerCase();d.includes("\
url(")||d.includes("expression(")||d.includes("javascript:")||d.includes("@import")||d.includes("beh\
avior:")||d.includes("-moz-binding")||d.includes("var(")||d.includes("env(")||n.push(`${l}: ${c}`)}),
n.join("; ")},"sanitizeRichPasteStyle");let richPasteColorCanvasContext=null;const parseRichPasteCssColor=r(
e=>{const n=String(e||"").trim();if(!n||n==="inherit"||n==="currentcolor"||n==="transparent"||window.
CSS&&typeof window.CSS.supports=="function"&&!window.CSS.supports("color",n))return null;try{if(!richPasteColorCanvasContext){
const o=document.createElement("canvas");o.width=1,o.height=1,richPasteColorCanvasContext=o.getContext(
"2d",{willReadFrequently:!0})}const i=richPasteColorCanvasContext;if(!i)return null;i.clearRect(0,0,
1,1),i.fillStyle="rgba(1, 2, 3, 0.004)",i.fillStyle=n,i.fillRect(0,0,1,1);const a=i.getImageData(0,0,
1,1).data;return!a||a[3]===0?null:{r:a[0],g:a[1],b:a[2],a:a[3]/255}}catch{return null}},"parseRichPa\
steCssColor"),richPasteColorLuminance=r(e=>{if(!e)return 0;const n=r(i=>{const a=Math.max(0,Math.min(
255,Number(i)||0))/255;return a<=.04045?a/12.92:Math.pow((a+.055)/1.055,2.4)},"channel");return .2126*
n(e.r)+.7152*n(e.g)+.0722*n(e.b)},"richPasteColorLuminance"),richPasteColorContrast=r((e,n)=>{const i=richPasteColorLuminance(
e),a=richPasteColorLuminance(n);return(Math.max(i,a)+.05)/(Math.min(i,a)+.05)},"richPasteColorContra\
st"),richPasteColorCss=r(e=>e?`rgb(${Math.round(e.r)}, ${Math.round(e.g)}, ${Math.round(e.b)})`:"","\
richPasteColorCss"),makeRichPasteTheme=r((e,n)=>{const i=richPasteColorLuminance(e)<.32;let a=n;return(!a||
richPasteColorContrast(e,a)<3)&&(a=i?{r:244,g:244,b:245,a:1}:{r:17,g:24,b:39,a:1}),{mode:i?"dark":"l\
ight",background:richPasteColorCss(e),foreground:richPasteColorCss(a),muted:i?"rgb(161, 161, 170)":"\
rgb(100, 116, 139)",border:i?"rgb(63, 63, 70)":"rgb(203, 213, 225)",surface:i?"rgb(33, 33, 33)":"rgb\
(248, 250, 252)",quote:i?"rgb(39, 39, 42)":"rgb(255, 249, 235)",link:i?"rgb(125, 211, 252)":"rgb(15,\
 118, 110)"}},"makeRichPasteTheme"),detectRichPasteTheme=r(e=>{const n={r:255,g:255,b:255,a:1},i={r:17,
g:24,b:39,a:1},a=document.createElement("template");if(a.innerHTML=String(e||""),!a.content.querySelector(
"*"))return makeRichPasteTheme(n,i);const o=document.createElement("div");o.setAttribute("aria-hidde\
n","true"),o.style.position="fixed",o.style.left="-100000px",o.style.top="0",o.style.width="794px",o.
style.visibility="hidden",o.style.pointerEvents="none",o.style.color="#111827",o.style.background="t\
ransparent",o.appendChild(a.content.cloneNode(!0)),document.body.appendChild(o);try{const l=[o,...Array.
from(o.querySelectorAll("*")).slice(0,5e3)],c=[],d=new Map;let m=0;const h=r(_=>Array.from(_.childNodes||
[]).reduce((S,L)=>L&&L.nodeType===Node.TEXT_NODE?S+String(L.textContent||"").replace(/\s+/g," ").trim().
length:S,0),"directTextLength");l.forEach(_=>{if(!_||_===o||!_.style)return;const S=window.getComputedStyle(
_),L=h(_);if(L>0){const P=parseRichPasteCssColor(S.color);if(P&&P.a>=.5){const H=richPasteColorCss(P),
Q=d.get(H)||{color:P,weight:0};Q.weight+=L,d.set(H,Q),m+=L}}if(!!(String(_.style.backgroundColor||"").
trim()||String(_.style.background||"").trim())){const P=parseRichPasteCssColor(S.backgroundColor);if(P&&
P.a>=.72){const H=String(_.textContent||"").replace(/\s+/g," ").trim().length;c.push({color:P,weight:Math.
max(1,H)})}}});const y=Array.from(d.values()).sort((_,S)=>S.weight-_.weight),v=y.length?y[0].color:null,
x=y.reduce((_,S)=>_+(richPasteColorLuminance(S.color)>=.6?S.weight:0),0);c.sort((_,S)=>S.weight-_.weight);
let w=c.length?c[0].color:null;return w||(w=m>0&&x/m>=.55?{r:11,g:11,b:12,a:1}:n),makeRichPasteTheme(
w,v||i)}catch{return makeRichPasteTheme(n,i)}finally{o.parentNode&&o.parentNode.removeChild(o)}},"de\
tectRichPasteTheme"),prepareRichPastePdfClone=r((e,n)=>{if(!e)return;const i=e.head||e.querySelector(
"head");i&&Array.from(i.querySelectorAll('link[rel="stylesheet"]')).forEach(a=>{try{a.remove()}catch{}}),
e.body&&(e.body.style.margin="0",e.body.style.background=n.background,e.body.style.color=n.foreground)},
"prepareRichPastePdfClone"),normalizeRichPasteTree=r(e=>{!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("*").forEach(n=>{if(!n||!n.getAttribute||!n.parentNode)return;const i=String(n.tagName||
"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(i)){n.remove();return}n.removeAttribute("class"),n.removeAttribute(
"id"),n.removeAttribute("role"),n.removeAttribute("aria-label"),i==="img"&&(n.setAttribute("loading",
"eager"),n.setAttribute("decoding","sync"),n.removeAttribute("srcset"),n.removeAttribute("sizes"));const a=n.
getAttribute("style");if(a){const o=sanitizeRichPasteStyle(a);o?n.setAttribute("style",o):n.removeAttribute(
"style")}})},"normalizeRichPasteTree"),extractRichPasteArticleHtml=r(e=>{const i=new DOMParser().parseFromString(
String(e||""),"text/html");if(!i.body)return"";const a=(i.body.textContent||"").replace(/\s+/g," ").
trim().length,o=i.body.querySelectorAll("*").length;if(a<1e3||o<120)return i.body.innerHTML;const c=[
...Array.from(i.body.querySelectorAll("article")),...Array.from(i.body.querySelectorAll("main")),...Array.
from(i.body.querySelectorAll('[role="main"],[role="article"]'))].filter(m=>(m.textContent||"").replace(
/\s+/g," ").trim().length>=a*.65);c.sort((m,h)=>{const y=+!!h.querySelector("h1")-+!!m.querySelector(
"h1");return y||m.querySelectorAll("*").length-h.querySelectorAll("*").length});const d=c[0]||null;return d?
d.outerHTML:i.body.innerHTML},"extractRichPasteArticleHtml"),sanitizeRichPasteHtml=r(e=>{if(!window.
DOMPurify||typeof window.DOMPurify.sanitize!="function"){const o=new DOMParser().parseFromString(String(
e||""),"text/html");return escapeHtml(o.body?o.body.textContent:"")}let n=extractRichPasteArticleHtml(
e),i=window.DOMPurify.sanitize(n||"",{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,
KEEP_CONTENT:!0});if((!i||i.trim()==="")&&e&&e.trim()!==""&&(i=window.DOMPurify.sanitize(e,{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,
ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,KEEP_CONTENT:!0})),!i)return"";const a=document.createElement("\
template");return a.innerHTML=i,normalizeRichPasteTree(a.content),a.innerHTML},"sanitizeRichPasteHtm\
l"),normalizeRichPastePrintHtml=r(e=>{const n=document.createElement("template");n.innerHTML=String(
e||"");const i=Array.from(n.content.querySelectorAll("*")),a=i.reduce((m,h)=>{const y=String(h.style&&
h.style.display||"").trim().toLowerCase();return m+(["flex","inline-flex","grid","inline-grid"].includes(
y)?1:0)},0),o=i.reduce((m,h)=>{if(!h||!h.style||!["article","div","main","section"].includes(String(
h.tagName||"").toLowerCase()))return m;const y=String(h.getAttribute("style")||""),v=Array.from(y.matchAll(
/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)).some(w=>Array.
from(w[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi)).some(_=>Math.abs(Number(_[1])||0)>=96)),x=Array.from(y.
matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)).some(w=>Math.abs(Number(w[1])||
0)>720);return m+(v||x?1:0)},0);if(i.length<=500&&a<=24&&o===0)return n.innerHTML;const l=new Set(["\
align-items","align-self","column-gap","flex","flex-basis","flex-direction","flex-grow","flex-shrink",
"flex-wrap","gap","grid","grid-auto-columns","grid-auto-flow","grid-auto-rows","grid-column","grid-c\
olumn-end","grid-column-start","grid-row","grid-row-end","grid-row-start","grid-template","grid-temp\
late-areas","grid-template-columns","grid-template-rows","justify-content","justify-items","justify-\
self","order","row-gap"]),c=new Set(["article","div","main","section"]),d=new Set(["padding","paddin\
g-left","padding-right","padding-inline","padding-inline-start","padding-inline-end"]);return i.forEach(
m=>{if(!m||!m.style)return;const h=String(m.tagName||"").toLowerCase(),y=[];String(m.getAttribute("s\
tyle")||"").split(";").forEach(v=>{if(!v||v.indexOf(":")<0)return;const x=v.indexOf(":"),w=v.slice(0,
x).trim().toLowerCase();let _=v.slice(x+1).trim();if(!(!w||!_||l.has(w))&&!["height","max-height","m\
in-height","overflow","overflow-x","overflow-y"].includes(w)&&!(["width","min-width"].includes(w)&&c.
has(h))){if(d.has(w)&&c.has(h)&&Array.from(_.matchAll(/(-?\d+(?:\.\d+)?)px/gi)).map(L=>Math.abs(Number(
L[1])||0)).some(L=>L>=96)&&(_="0px"),w==="display"){const S=_.toLowerCase();["flex","grid"].includes(
S)?_="block":["inline-flex","inline-grid"].includes(S)&&(_="inline-block")}y.push(`${w}: ${_}`)}}),y.
length?m.setAttribute("style",y.join("; ")):m.removeAttribute("style")}),n.innerHTML},"normalizeRich\
PastePrintHtml"),getRichPasteSelectionRange=r(e=>{const n=window.getSelection&&window.getSelection();
if(!n||!n.rangeCount)return null;const i=n.getRangeAt(0);if(e&&e.contains(i.commonAncestorContainer))
return i;const a=document.createRange();return a.selectNodeContents(e),a.collapse(!1),a},"getRichPas\
teSelectionRange"),insertNodeIntoRichPasteEditor=r(e=>{const n=getRichPasteEditor();!n||!e||(n.appendChild(
e),updateRichPasteStatus())},"insertNodeIntoRichPasteEditor"),insertHtmlIntoRichPasteEditor=r(e=>{const n=sanitizeRichPasteHtml(
e);if(!n||n.trim()==="")return!1;const i=document.createElement("template");i.innerHTML=n;const a=i.
content.cloneNode(!0);return insertNodeIntoRichPasteEditor(a),!0},"insertHtmlIntoRichPasteEditor"),insertTextIntoRichPasteEditor=r(
e=>{if(e==null)return;const n=document.createTextNode(String(e));insertNodeIntoRichPasteEditor(n)},"\
insertTextIntoRichPasteEditor"),blobToDataUrl=r(e=>new Promise((n,i)=>{const a=new FileReader;a.onload=
()=>n(String(a.result||"")),a.onerror=()=>i(a.error||new Error("clipboard_image_read_failed")),a.readAsDataURL(
e)}),"blobToDataUrl"),insertClipboardImageBlob=r(async(e,n="clipboard-image")=>{if(!e)return!1;const i=await blobToDataUrl(
e);return i?(insertHtmlIntoRichPasteEditor(`<p><img src="${escapeHtml(i)}" alt="${escapeHtml(n)}"></\
p>`),!0):!1},"insertClipboardImageBlob"),readClipboardRichContent=r(async()=>{if(!navigator.clipboard||
!navigator.clipboard.read)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30EA\u30C3\u30C1\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u8AAD\u307F\u53D6\u308A\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
const e=getRichPasteCapture();e&&(e.value="");const n=await navigator.clipboard.read();if(!n||!n.length)
return!1;let i=!1;for(const a of n){if(!a)continue;const o=Array.from(a.types||[]);let l=!1;if(o.includes(
"text/html")){const m=await(await a.getType("text/html")).text();m&&insertHtmlIntoRichPasteEditor(m)&&
(i=!0,l=!0)}if(!l&&o.includes("text/plain")){const m=await(await a.getType("text/plain")).text();m&&
(insertTextIntoRichPasteEditor(m),i=!0)}const c=o.find(d=>d&&d.startsWith("image/"));if(!l&&c){const d=await a.
getType(c);await insertClipboardImageBlob(d,"clipboard-image")&&(i=!0)}}return i},"readClipboardRich\
Content"),ingestRichPasteClipboardData=r(async e=>{if(!e)return!1;let n=!1;const i=e.getData&&e.getData(
"text/html"),a=e.getData&&e.getData("text/plain");let o=!1;i&&insertHtmlIntoRichPasteEditor(i)&&(n=!0,
o=!0),!o&&a&&(insertTextIntoRichPasteEditor(a),n=!0);const c=Array.from(e.items||[]).filter(d=>d&&d.
kind==="file").map(d=>d.getAsFile()).filter(d=>d&&d.type&&d.type.startsWith("image/"));if(!o&&c.length)
for(const d of c)try{await insertClipboardImageBlob(d,d.name||"clipboard-image")&&(n=!0)}catch{}return n},
"ingestRichPasteClipboardData"),buildRichPastePdfFilename=r(()=>{const e=new Date,n=r(i=>String(i).padStart(
2,"0"),"pad");return`clipboard_rich_${e.getFullYear()}${n(e.getMonth()+1)}${n(e.getDate())}_${n(e.getHours())}${n(
e.getMinutes())}${n(e.getSeconds())}.pdf`},"buildRichPastePdfFilename"),getRichPasteProgressElements=r(
()=>({container:get("rich-paste-progress-container"),bar:get("rich-paste-progress-bar"),text:get("ri\
ch-paste-progress-text")}),"getRichPasteProgressElements"),setRichPasteProgress=r((e,n=null)=>{const{
container:i,bar:a,text:o}=getRichPasteProgressElements(),l=Math.max(0,Math.min(100,Number(e)||0));if(i&&
(i.classList.remove("hidden"),i.style.setProperty("display","block","important")),a&&(a.style.width=
`${l}%`,a.style.transform="none"),o&&(o.textContent=`${Math.round(l)}%`),n&&i){const c=i.querySelector(
".text-amber-400");c&&(c.innerHTML=`<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(n)}`)}},"set\
RichPasteProgress"),hideRichPasteProgress=r(()=>{const{container:e,bar:n}=getRichPasteProgressElements();
n&&(n.style.transform="scaleX(0)"),e&&(e.classList.add("hidden"),e.style.display="none")},"hideRichP\
asteProgress"),inferRichPasteTitle=r(()=>{const e=getRichPasteEditor();if(!e)return"Clipboard Export";
const n=e.querySelector("h1, h2, h3, h4, h5, h6");if(n&&n.textContent&&n.textContent.trim())return n.
textContent.trim().slice(0,48);const i=(e.innerText||"").trim().replace(/\s+/g," ");return i?i.slice(
0,48):"Clipboard Export"},"inferRichPasteTitle"),waitForRichPasteMedia=r(async(e,n=2500)=>{if(!e)return;
const i=new Promise(o=>setTimeout(o,Math.max(0,n))),a=Promise.all(Array.from(e.querySelectorAll("img")||
[]).map(o=>!o||o.complete?Promise.resolve():new Promise(l=>{let c=!1;const d=r(()=>{c||(c=!0,l())},"\
finish");o.addEventListener("load",d,{once:!0}),o.addEventListener("error",d,{once:!0}),setTimeout(d,
Math.max(250,Math.min(n,2e3)))})));if(await Promise.race([a,i]),document.fonts&&document.fonts.ready)
try{await Promise.race([document.fonts.ready,i])}catch{}},"waitForRichPasteMedia"),normalizeRichPastePdfText=r(
e=>String(e||"").replace(/\u00a0/g," ").replace(/\r\n?/g,`
`).replace(/[ \t\f\v]+/g," ").replace(/\n[ \t]+/g,`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).trim(),"normalizeRichPastePdfText"),normalizeRichPastePdfCodeText=r(e=>String(e||"").replace(/\u00a0/g,
" ").replace(/\r\n?/g,`
`),"normalizeRichPastePdfCodeText"),collectRichPasteInlineSegments=r((e,n={})=>{if(!e)return[];const i=n.
allowLinks!==!1,a=[],o=r((l,c)=>{if(!l)return;if(l.nodeType===Node.TEXT_NODE){const h=l.textContent||
"";h&&a.push(Object.assign({},c,{text:h}));return}if(l.nodeType!==Node.ELEMENT_NODE)return;const d=String(
l.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(d))return;if(d==="br"){a.push({text:`
`});return}const m=Object.assign({},c);["b","strong"].includes(d)&&(m.bold=!0),["i","em"].includes(d)&&
(m.italic=!0),d==="a"&&i&&(m.link=String(l.getAttribute("href")||"").trim()),d==="code"&&(m.monospace=
!0),Array.from(l.childNodes||[]).forEach(h=>o(h,m))},"walk");return o(e,{bold:!!n.bold,italic:!!n.italic}),
a},"collectRichPasteInlineSegments"),collectRichPasteInlineText=r((e,n={})=>collectRichPasteInlineSegments(
e,n).map(a=>a.text).join(""),"collectRichPasteInlineText"),collectRichPasteTableRows=r(e=>{const n=[];
return Array.from(e.querySelectorAll("tr")||[]).forEach(i=>{i&&i.closest&&i.closest("table")===e&&n.
push(i)}),n},"collectRichPasteTableRows"),makeRichPasteTableMarkdown=r(e=>{const n=e&&e.querySelector?
e.querySelector("caption"):null,i=n?normalizeRichPastePdfText(collectRichPasteInlineText(n)):"",a=collectRichPasteTableRows(
e).map(m=>Array.from(m.children||[]).filter(y=>{const v=String(y.tagName||"").toLowerCase();return v===
"th"||v==="td"}).map(y=>normalizeRichPastePdfText(collectRichPasteInlineText(y))||" ")).filter(m=>m.
length);if(!a.length)return i||"[table]";const o=a.reduce((m,h)=>Math.max(m,h.length),0),l=a.map(m=>{
const h=m.slice(0,o);for(;h.length<o;)h.push(" ");return h}),c=`| ${Array(o).fill("---").join(" | ")}\
 |`,d=[];i&&(d.push(`Table: ${i}`),d.push("")),d.push(`| ${l[0].join(" | ")} |`),d.push(c);for(let m=1;m<
l.length;m+=1)d.push(`| ${l[m].join(" | ")} |`);return d.join(`
`)},"makeRichPasteTableMarkdown"),collectRichPasteListBlocks=r((e,n=!1,i=0)=>{const a=[],o=Array.from(
e.children||[]).filter(c=>String(c.tagName||"").toLowerCase()==="li");let l=1;return o.forEach(c=>{const d=c.
cloneNode(!0);Array.from(d.querySelectorAll("ul,ol")||[]).forEach(h=>{try{h.remove()}catch{}});const m=collectRichPasteInlineSegments(
d);m.length>0&&a.push({type:"list_item",ordered:n,depth:i,index:l,segments:m}),Array.from(c.children||
[]).forEach(h=>{const y=String(h.tagName||"").toLowerCase();(y==="ul"||y==="ol")&&a.push(...collectRichPasteListBlocks(
h,y==="ol",i+1))}),l+=1}),a},"collectRichPasteListBlocks"),collectRichPastePdfBlocks=r((e,n=0)=>{const i=[];
if(!e)return i;let a=[];const o=r(()=>{a.length!==0&&(i.push({type:"paragraph",segments:[...a]}),a=[])},
"flushBuffer");return Array.from(e.childNodes||[]).forEach(l=>{if(!l)return;if(l.nodeType===Node.TEXT_NODE){
const h=(l.textContent||"").replace(/\u00a0/g," ");h&&a.push({text:h});return}if(l.nodeType!==Node.ELEMENT_NODE)
return;const c=String(l.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(c))return;if(c==="br"){
a.push({text:`
`});return}if(/^h[1-6]$/.test(c)){o();const h=collectRichPasteInlineSegments(l);h.length>0&&i.push({
type:"heading",level:Number(c.slice(1))||1,segments:h});return}if(c==="p"){o();const h=collectRichPasteInlineSegments(
l);h.length>0&&i.push({type:"paragraph",segments:h});return}if(c==="blockquote"){o();const h=collectRichPasteInlineSegments(
l,{italic:!0});h.length>0&&i.push({type:"blockquote",segments:h});return}if(c==="pre"){o();const h=normalizeRichPastePdfCodeText(
l.innerText||l.textContent||"");h.trim()&&i.push({type:"code",text:h});return}if(c==="table"){o();const h=makeRichPasteTableMarkdown(
l);h&&i.push({type:"table",text:h});return}if(c==="ul"||c==="ol"){o(),i.push(...collectRichPasteListBlocks(
l,c==="ol",n));return}if(c==="hr"){o(),i.push({type:"hr"});return}if(c==="figure"){o();const h=l.querySelector(
"img");h&&i.push({type:"image",src:String(h.getAttribute("src")||"").trim(),alt:String(h.getAttribute(
"alt")||h.getAttribute("title")||"").trim(),title:String(h.getAttribute("title")||"").trim()});const y=l.
querySelector("figcaption");if(y){const v=collectRichPasteInlineSegments(y);v.length>0&&i.push({type:"\
paragraph",segments:v})}return}if(c==="img"){o(),i.push({type:"image",src:String(l.getAttribute("src")||
"").trim(),alt:String(l.getAttribute("alt")||l.getAttribute("title")||"").trim(),title:String(l.getAttribute(
"title")||"").trim()});return}if(c==="li"){o(),i.push(...collectRichPasteListBlocks(l,!1,n));return}
if(Array.from(l.children||[]).some(h=>{const y=String(h.tagName||"").toLowerCase();return/^h[1-6]$/.
test(y)||["p","div","section","article","main","blockquote","pre","table","ul","ol","hr","figure","i\
mg","li"].includes(y)})&&["div","section","article","main","figure"].includes(c)){o(),i.push(...collectRichPastePdfBlocks(
l,n+1));return}const m=collectRichPasteInlineSegments(l);m.length>0&&a.push(...m)}),o(),i},"collectR\
ichPastePdfBlocks"),detectImageMimeType=r(e=>{const n=String(e||"").match(/^data:(image\/[a-z0-9.+-]+);/i);
return n?n[1].toLowerCase():"image/png"},"detectImageMimeType"),loadRichPasteImageData=r(async(e,n=3e3)=>{
const i=String(e||"").trim();if(!i)return null;if(i.startsWith("data:image/"))return{dataUrl:i,mimeType:detectImageMimeType(
i)};let a=null;try{a=new URL(i,window.location.href)}catch{return null}if(!(a.origin===window.location.
origin))return null;const l=(async()=>{try{const c=await fetch(a.toString(),{credentials:"same-origi\
n",cache:"force-cache"});if(!c.ok)return null;const d=await c.blob(),m=await blobToDataUrl(d);return{
dataUrl:m,mimeType:d.type||detectImageMimeType(m)}}catch{return null}})();return await Promise.race(
[l,new Promise(c=>setTimeout(()=>c(null),Math.max(250,n)))])},"loadRichPasteImageData"),buildRichPastePreviewHtml=r(
(e="preview")=>{const n=getRichPasteEditor();if(!n)return"";const i=inferRichPasteTitle(),a=new Date().
toLocaleString("ja-JP"),o=sanitizeRichPasteHtml(n.innerHTML||""),l=detectRichPasteTheme(o),c=normalizeRichPastePrintHtml(
o),d=e==="pdf";return`<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(i)} - Preview</title>
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
	    body { margin: 0; background: ${d?"var(--rp-background)":"#eef2f7"}; color: var(--rp-foreground\
); font-family: "Noto Sans JP", system-ui, sans-serif; }
	    .page { max-width: ${d?"794px":"920px"}; margin: 0 auto; padding: ${d?"28px 30px 36px":"24px"};\
 }
	    .card { background: var(--rp-background); color: var(--rp-foreground); border: 1px solid var(--\
rp-border); border-radius: 18px; padding: 20px; box-shadow: ${d?"none":"0 18px 45px rgba(15,23,42,0.\
14)"}; }
	    .title { margin: 0; font-size: ${d?"22px":"24px"}; line-height: 1.35; color: var(--rp-foregroun\
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
    .toolbar { display:${d?"none":"flex"}; gap:10px; margin-top: 16px; flex-wrap: wrap; }
    .toolbar button { border: 1px solid var(--rp-border); background: var(--rp-surface); color: var(\
--rp-foreground); border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    ${d?".card { border-radius: 0; } .page { max-width: none; padding: 0; }":""}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1 class="title">${escapeHtml(i)}</h1>
      <div class="meta">Clipboard import | ${escapeHtml(a)} | \u672C\u6587\u78BA\u8A8D\u7528\u30D7\u30EC\u30D3\u30E5\u30FC</div>
      <div class="toolbar">
        <button onclick="window.close()">\u9589\u3058\u308B</button>
      </div>
      <div class="content">${c||"<p>\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</p>"}</di\
v>
    </div>
  </div>
</body>
</html>`},"buildRichPastePreviewHtml"),openSandboxedHtmlTab=r(e=>{const i=`<!doctype html><html><hea\
d><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:10\
0%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe i\
d="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referre\
r"></iframe><script>document.getElementById('preview').srcdoc=${JSON.stringify(String(e||"")).replace(
/</g,"\\u003c").replace(/\u2028/g,"\\u2028").replace(/\u2029/g,"\\u2029")};<\/script></body></html>`,
a=new Blob([i],{type:"text/html;charset=utf-8"}),o=URL.createObjectURL(a);return window.open(o,"_bla\
nk","noopener,noreferrer")?(setTimeout(()=>URL.revokeObjectURL(o),6e4),!0):(URL.revokeObjectURL(o),!1)},
"openSandboxedHtmlTab"),openRichPastePreviewTab=r(()=>{const e=buildRichPastePreviewHtml("preview");
if(!e){showToast("\u78BA\u8A8D\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093","warning",
!0);return}openSandboxedHtmlTab(e)||showToast("\u5225\u30BF\u30D6\u306E\u8868\u793A\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},"openRichPastePreviewTab"),renderRichPastePdfBlob=r(async()=>{const e=get("rich-paste-p\
rogress-container"),n=get("rich-paste-progress-bar"),i=get("rich-paste-progress-text"),a=r(x=>{const w=Math.
max(0,Math.min(100,Number(x)||0));n&&(n.style.width="100%",n.style.transformOrigin="left center",(!n.
style.transition||n.style.transition.indexOf("transform")===-1)&&(n.style.transition="transform 0.45\
s cubic-bezier(0.22, 1, 0.36, 1)"),n.style.transform=`scaleX(${w/100})`,n.style.willChange="transfor\
m"),i&&(i.innerText=`${Math.round(w)}%`)},"updateProgress");e&&(e.classList.remove("hidden"),e.style.
setProperty("display","block","important")),n&&(n.style.transition="none",n.style.width="100%",n.style.
transformOrigin="left center",n.style.transform="scaleX(0)",n.offsetHeight,n.style.transition="trans\
form 0.45s cubic-bezier(0.22, 1, 0.36, 1)"),a(0),await new Promise(x=>requestAnimationFrame(()=>setTimeout(
x,150)));const o=getRichPasteEditor();if(!o)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093");
const l=inferRichPasteTitle(),c=sanitizeRichPasteHtml(o.innerHTML||""),d=detectRichPasteTheme(c),m=normalizeRichPastePrintHtml(
c);await ensurePdfLibraries();const h=window.jspdf&&window.jspdf.jsPDF?window.jspdf.jsPDF:null;if(!h)
throw new Error("jsPDF \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
const y=window.html2canvas;if(typeof y!="function")throw new Error("html2canvas \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
a(5);const v=document.createElement("div");v.style.position="absolute",v.style.left="-10000px",v.style.
top="0",v.style.width="794px",v.style.background=d.background,v.style.color=d.foreground,v.style.boxSizing=
"border-box",v.style.fontFamily='"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif',v.
innerHTML=`
                <style>
                        :root {
                            color-scheme: ${d.mode};
                            --rp-background: ${d.background};
                            --rp-foreground: ${d.foreground};
                            --rp-muted: ${d.muted};
                            --rp-border: ${d.border};
                            --rp-surface: ${d.surface};
                            --rp-quote: ${d.quote};
                            --rp-link: ${d.link};
                        }
	                    .pdf-root-wrapper {
	                        background-color: var(--rp-background);
	                        color: var(--rp-foreground);
	                        padding: 40px;
	                        width: 794px;
	                        min-height: 1123px;
	                        box-sizing: border-box;
	                        color-scheme: ${d.mode};
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
                    <div class="pdf-title">${escapeHtml(l)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString("ja-JP")}</div>
                    <div class="pdf-content">${m}</div>
                </div>
            `,document.body.appendChild(v),await waitForRichPasteMedia(v,4e3),a(15);try{const x=new h(
{unit:"mm",format:"a4",orientation:"portrait",compress:!0}),w=x.internal.pageSize.getWidth(),_=x.internal.
pageSize.getHeight(),S=794,L=Math.floor(_/w*S),M=v.scrollHeight||v.offsetHeight;let P=0,H=!0;const Q=Math.
ceil(M/L);let ee=0;for(;P<M;){if(richPasteAbortController&&richPasteAbortController.signal.aborted)throw new DOMException(
"Aborted","AbortError");const B=Math.min(L,M-P),te=(await new Promise((be,Le)=>{const oe=setTimeout(
()=>Le(new Error("PDF chunk rendering timed out")),12e4);y(v,{scale:1,useCORS:!0,allowTaint:!1,backgroundColor:d.
background,logging:!1,imageTimeout:5e3,x:0,y:P,width:S,height:B,windowWidth:S,scrollX:0,scrollY:0,signal:richPasteAbortController?
richPasteAbortController.signal:void 0,onclone:r(le=>{prepareRichPastePdfClone(le,d);const Y=le.querySelector(
".pdf-root-wrapper");Y&&(Y.style.position="relative",Y.style.left="0",Y.style.top="0")},"onclone")}).
then(le=>{clearTimeout(oe),be(le)}).catch(le=>{clearTimeout(oe),Le(le)})})).toDataURL("image/jpeg",.95),
ge=x.getImageProperties(te),de=Math.min(_,ge.height*w/ge.width);H||x.addPage(),x.addImage(te,"JPEG",
0,0,w,de),H=!1,P+=B,ee++;const Ce=Math.min(100,15+Math.round(ee/Q*85));a(Ce),await new Promise(be=>setTimeout(
be,100))}return a(100),{blob:x.output("blob"),fileName:buildRichPastePdfFilename()}}finally{e&&(e.classList.
add("hidden"),e.style.display="none"),v&&v.parentNode&&document.body.removeChild(v)}},"renderRichPas\
tePdfBlob"),createRichPastePdfBlob=r(async()=>await renderRichPastePdfBlob(),"createRichPastePdfBlob"),
buildRichPasteServerPayload=r(()=>{const e=getRichPasteEditor();if(!e)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\
\u3093");const n=String(e.innerHTML||"").trim(),i=String(e.textContent||"").trim(),a=n||(i?`<p>${escapeHtml(
i).replace(/\n/g,"<br/>")}</p>`:"");return{title:inferRichPasteTitle(),html:a,created_at:new Date().
toLocaleString("ja-JP"),theme:detectRichPasteTheme(sanitizeRichPasteHtml(a))}},"buildRichPasteServer\
Payload"),attachRichPastePdfAndSend=r(async(e,n,i,a)=>{const o=new Set(collectAttachmentItemsForSend().
map(y=>y.path)),l=new File([e],n,{type:"application/pdf",lastModified:Date.now()}),c=get("prompt-inp\
ut");if(c&&(c.value=i),await handleFiles([l],{openModal:!1}),!collectAttachmentItemsForSend().map(y=>y.
path).some(y=>!o.has(y)))throw c&&(c.value=a),new Error("PDF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const h=sendMessage();clearRichPasteEditor(!0),window.closeRichPasteModal(),showToast("PDF\u3092\u6DFB\u4ED8\u3057\u3066\u9001\u4FE1\u3092\u958B\u59CB\
\u3057\u307E\u3057\u305F","success"),h&&typeof h.catch=="function"&&h.catch(()=>{})},"attachRichPast\
ePdfAndSend"),openRichPasteModal=r(async()=>{await ensureUserSettingsSnapshot(),showModal("rich-past\
e-modal"),location.pathname!=="/paste"&&history.pushState({modal:"paste"},"","/paste");const e=getRichPastePrompt();
e&&(richPastePromptPreferenceSyncing=!0,e.value=getRichPasteEffectivePrompt(userSettingsSnapshot),richPastePromptPreferenceSyncing=
!1),updateRichPasteStatus(),setTimeout(()=>focusRichPasteEditor(),80)},"openRichPasteModal");window.
closeRichPasteModal=(e=!1)=>{hideModal("rich-paste-modal"),!e&&location.pathname==="/paste"&&history.
back()};const sendRichPasteToModel=r(async(e={})=>{const n=!!(e&&e.serverSide);if(abortController||richPasteAbortController){
showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u307E\u305F\u306FPDF\u5909\u63DB\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const i=getRichPasteEditor(),a=getRichPastePrompt(),o=get(n?"rich-paste-send-se\
rver-btn":"rich-paste-send-btn"),l=get("rich-paste-cancel-btn");if(!i||!i.innerText||!i.innerText.trim()){
showToast("\u8CBC\u308A\u4ED8\u3051\u308B\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}richPasteAbortController=new AbortController,l&&(l.onclick=()=>{richPasteAbortController&&
(richPasteAbortController.abort(),showToast("PDF\u5909\u63DB\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info"))});const c=a&&a.value&&a.value.trim()?a.value.trim():RICH_PASTE_DEFAULT_PROMPT,d=get("prompt\
-input")?get("prompt-input").value:"";o&&(o.disabled=!0);try{const m=get("toast-stack");if(m&&m.querySelectorAll(
".toast").forEach(h=>{(h.innerText.includes("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059")||
h.innerText.includes("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059"))&&
h.remove()}),n?(showToast("\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...",
"info",!0),setRichPasteProgress(2,"\u30B5\u30FC\u30D0\u30FC\u5074\u3067PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...")):
showToast("PDF\u3092\u751F\u6210\u3057\u3066\u3044\u307E\u3059...","info",!0),n){if(!RICH_PASTE_PDF_SERVER_ROUTE)
throw new Error("\u30B5\u30FC\u30D0\u30FC\u5074PDF\u751F\u6210\u306EURL\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093");
const h=buildRichPasteServerPayload();setRichPasteProgress(10,"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u4FE1\u4E2D...");
const y=await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(h),signal:richPasteAbortController.signal});if(setRichPasteProgress(60,
"PDF\u3092\u53D7\u4FE1\u4E2D..."),!y.ok){let _="";try{const S=await y.json();_=S&&(S.message||S.error)?
String(S.message||S.error):""}catch{try{_=await y.text()}catch{_=""}}throw _==="missing_html"?new Error(
"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u308BHTML\u304C\u7A7A\u3067\u3059\u3002\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u5185\u5BB9\u306E\u53D6\u308A\u8FBC\u307F\u3092\u5148\u306B\u884C\u3063\u3066\u304F\u3060\u3055\u3044"):
new Error(_?`\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${_}`:
"\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}setRichPasteProgress(
75,"PDF\u3092\u6DFB\u4ED8\u4E2D...");const v=await y.blob(),x=y.headers.get("X-Rich-Paste-Filename")||
buildRichPastePdfFilename();!!(get("rich-paste-download-only")&&get("rich-paste-download-only").checked)?
(setRichPasteProgress(90,"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D..."),downloadBlob(v,x),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(v,x,c,d),setRichPasteProgress(100,"\u5B8C\u4E86"),
setTimeout(()=>hideRichPasteProgress(),400)}else{const h=await createRichPastePdfBlob();!!(get("rich\
-paste-download-only")&&get("rich-paste-download-only").checked)?(downloadBlob(h.blob,h.fileName),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(h.blob,h.fileName,c,d)}}catch(m){if(m.
name==="AbortError"){console.log("PDF generation aborted by user"),n&&(setRichPasteProgress(0,"\u30AD\u30E3\u30F3\u30BB\u30EB\
\u3055\u308C\u307E\u3057\u305F"),setTimeout(()=>hideRichPasteProgress(),800));return}get("prompt-inp\
ut")&&(get("prompt-input").value=d);const h=m&&m.message?m.message:"PDF\u5316\u3057\u3066\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
showToast(h,"error",!0),n&&(setRichPasteProgress(0,"\u5931\u6557\u3057\u307E\u3057\u305F"),setTimeout(
()=>hideRichPasteProgress(),1200))}finally{o&&(o.disabled=!1),richPasteAbortController=null}},"sendR\
ichPasteToModel");let csrfToken=document.querySelector('meta[name="csrf-token"]').content,csrfRefreshPromise=null;
const refreshCsrfToken=r(async()=>csrfRefreshPromise||(csrfRefreshPromise=(async()=>{const e=await fetch(
"/api/csrf_token",{method:"GET",credentials:"include",cache:"no-store",headers:{Accept:"application/\
json"}});if(!e.ok)return!1;const n=await e.json().catch(()=>({})),i=n&&typeof n.csrf_token=="string"?
n.csrf_token:"";if(!i)return!1;csrfToken=i;const a=document.querySelector('meta[name="csrf-token"]');
return a&&a.setAttribute("content",i),!0})().catch(()=>!1).finally(()=>{csrfRefreshPromise=null}),csrfRefreshPromise),
"refreshCsrfToken"),apiFetch=r(async(e,n={})=>{const i=(n.method||"GET").toUpperCase(),a=Object.assign(
{},n.headers||{}),o=!["GET","HEAD","OPTIONS"].includes(i);o&&(a["X-CSRF-Token"]=csrfToken);const l=n.
credentials||"include";let c=await fetch(e,Object.assign({},n,{headers:a,credentials:l}));if(o&&(c.status===
403||c.status===404)){let d=null;try{d=await c.clone().json()}catch{}const m=d&&d.error;if(m==="acco\
unt_locked")return!isAdminUser&&!document.getElementById("bot-lock-overlay")&&showBotLockOverlay(d.message||
"\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u3066\u3044\u307E\u3059\u3002",
d.remaining_seconds),c;if(m==="banned"||m==="turnstile_failed"||m==="rate_limit")return c;if(m==="tu\
rnstile_required"&&isBotDetectionActive())return botDetectionVerified=!1,await Promise.race([runBotDetectionGate(),
new Promise(v=>setTimeout(()=>v(!1),3e4))])&&(a["X-CSRF-Token"]=csrfToken,c=await fetch(e,Object.assign(
{},n,{headers:a,credentials:l}))),c;await refreshCsrfToken()&&(a["X-CSRF-Token"]=csrfToken,c=await fetch(
e,Object.assign({},n,{headers:a,credentials:l})))}return c},"apiFetch"),manualSpinnerRequestOptions=r(
e=>window.ProgressSpinner?window.ProgressSpinner.manualRequestOptions(e):e,"manualSpinnerRequestOpti\
ons");window.updateGoogleLinkUI=e=>{const n=get("google-link-text"),i=get("google-email-text"),a=get(
"google-action-area"),o=get("google-link-icon");!n||!a||(e.google_id?(n.innerText="\u9023\u643A\u6E08\u307F",
n.classList.replace("text-gray-200","text-green-400"),i.innerText=e.google_email||"\u9023\u643A\u4E2D\u306E Google \u30A2\u30AB\u30A6\u30F3\u30C8",
o.classList.replace("bg-gray-800","bg-green-900/30"),o.classList.add("text-green-400"),a.innerHTML='\
<button onclick="unlinkGoogleAccount()" class="px-4 py-2 bg-red-900/20 hover:bg-red-900/40 text-red-\
400 border border-red-800 rounded text-xs font-bold transition btn-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):
(n.innerText="\u672A\u9023\u643A",n.classList.replace("text-green-400","text-gray-200"),i.innerText=
"Google \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
o.classList.replace("bg-green-900/30","bg-gray-800"),o.classList.remove("text-green-400"),a.innerHTML=
'<a href="/login/google" class="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white roun\
ded text-xs font-bold transition btn-hover">Google \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkGoogleAccount=
async()=>{if(confirm(`Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Google \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkGoogleAccount,{method:"POST"});if(e.ok)showToast("\
Google \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(n=>n.json()).then(n=>updateGoogleLinkUI(n));else{const n=await e.json();showToast(n.error||"\u89E3\u9664\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}},window.updateMinashinLinkUI=e=>{const n=get("minashin-link-text"),i=get("minashin-emai\
l-text"),a=get("minashin-action-area"),o=get("minashin-link-icon");!n||!a||(e.minashin_sub?(n.innerText=
"\u9023\u643A\u6E08\u307F",n.classList.replace("text-gray-200","text-green-400"),i.innerText=e.minashin_email||
"\u9023\u643A\u4E2D\u306E Minashin \u30A2\u30AB\u30A6\u30F3\u30C8",o.classList.replace("bg-gray-800",
"bg-green-900/30"),a.innerHTML='<button onclick="unlinkMinashinAccount()" class="px-4 py-2 bg-red-90\
0/20 hover:bg-red-900/40 text-red-400 border border-red-800 rounded text-xs font-bold transition btn\
-hover">\u9023\u643A\u3092\u89E3\u9664</button>'):(n.innerText="\u672A\u9023\u643A",n.classList.replace(
"text-green-400","text-gray-200"),i.innerText="Minashin \u30A2\u30AB\u30A6\u30F3\u30C8\u3067\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002",
o.classList.replace("bg-green-900/30","bg-gray-800"),a.innerHTML='<a href="/login/minashin" class="i\
nline-block px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition \
btn-hover">Minashin \u3068\u9023\u643A\u3059\u308B</a>'))},window.unlinkMinashinAccount=async()=>{if(confirm(
`Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u5F8C\u306F Minashin \u30ED\u30B0\u30A4\u30F3\u304C\u5229\u7528\u3067\u304D\u306A\u304F\u306A\u308A\u307E\u3059\uFF08\u30D1\u30B9\u30EF\u30FC\u30C9\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u306A\u3044\u5834\u5408\u306F\u30ED\u30B0\u30A4\u30F3\u3067\u304D\u306A\u304F\u306A\u308B\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09\u3002`))
try{const e=await apiFetch(CHAT_CONFIG.urls.unlinkMinashinAccount,{method:"POST"});if(e.ok)showToast(
"Minashin \u9023\u643A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"),apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(n=>n.json()).then(n=>updateMinashinLinkUI(n));else{const n=await e.json();showToast(n.error||"\u89E3\
\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}catch{showToast("\u30CD\u30C3\u30C8\u30EF\u30FC\u30AF\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}};let lastClientDebugEnabled=null;const isClientDebugLogEnabled=r(()=>{const e=get("set-\
client-debug-log");return!!(e&&e.checked)},"isClientDebugLogEnabled"),sendClientDebugLog=r((e,n)=>{if(!isClientDebugLogEnabled())
return;const i={level:String(e||"info"),message:String(n||"")};apiFetch("/api/debug/client_log",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(i)}).catch(()=>{})},"sendClien\
tDebugLog"),syncClientDebugLogToggle=r((e,n)=>{const i=get("set-client-debug-log");i&&(i.checked=!!e);
const a=!!e;a&&lastClientDebugEnabled!==!0&&sendClientDebugLog("info",`Client debug logging enabled \
(${n}).`),lastClientDebugEnabled=a},"syncClientDebugLogToggle"),nowPerfMs=r(()=>window.performance&&
typeof window.performance.now=="function"?window.performance.now():Date.now(),"nowPerfMs"),reportFirstTokenLatency=r(
e=>{if(enableLatencyMetrics)try{if(!e||typeof e!="object")return;const n=Number(e.latency_seconds);if(!Number.
isFinite(n)||n<0||n>600)return;const i=Number(e.latency_ms),a={latency_seconds:Number(n.toFixed(6)),
latency_ms:Number.isFinite(i)?Math.max(0,Math.round(i)):Math.round(n*1e3),thread_id:e.thread_id?String(
e.thread_id):null,job_id:e.job_id?String(e.job_id):null,model:e.model?String(e.model):null,first_event_type:e.
first_event_type?String(e.first_event_type):"content",client_sent_at_ms:Number.isFinite(Number(e.client_sent_at_ms))?
Math.round(Number(e.client_sent_at_ms)):Date.now(),is_total:!!e.is_total,client_done_at_ms:Number.isFinite(
Number(e.client_done_at_ms))?Math.round(Number(e.client_done_at_ms)):null};apiFetch("/api/metrics/fi\
rst_token",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(a)}).catch(
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
function loadScriptOnce(e,n){const i=n?document.getElementById(n):null;return i?i.dataset.loaded==="\
1"?Promise.resolve(i):new Promise((a,o)=>{i.addEventListener("load",()=>a(i),{once:!0}),i.addEventListener(
"error",o,{once:!0})}):new Promise((a,o)=>{const l=document.createElement("script");n&&(l.id=n),l.src=
e,l.async=!0,l.onload=()=>{l.dataset.loaded="1",a(l)},l.onerror=o,document.head.appendChild(l)})}r(loadScriptOnce,
"loadScriptOnce");function loadStylesheetOnce(e,n){const i=n?document.getElementById(n):null;if(i)return Promise.
resolve(i);const a=Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find(o=>o.href===
e);return a?Promise.resolve(a):new Promise((o,l)=>{const c=document.createElement("link");n&&(c.id=n),
c.rel="stylesheet",c.href=e,c.onload=()=>o(c),c.onerror=l,document.head.appendChild(c)})}r(loadStylesheetOnce,
"loadStylesheetOnce");async function ensureMathJaxLoaded(){return window.MathJax&&typeof window.MathJax.
typesetPromise=="function"?window.MathJax:(mathJaxLoadPromise||(window.MathJax=window.MathJax||{tex:{
inlineMath:[["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{
ignoreHtmlClass:"tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},
startup:{typeset:!1}},mathJaxLoadPromise=loadScriptOnce(MATHJAX_SRC,"MathJax-script").catch(e=>{throw mathJaxLoadPromise=
null,e})),await mathJaxLoadPromise,window.MathJax||null)}r(ensureMathJaxLoaded,"ensureMathJaxLoaded");
async function ensureHighlightLoaded(){return window.hljs?window.hljs:(highlightLoadPromise||(highlightLoadPromise=
Promise.all([loadStylesheetOnce(HLJS_CSS_SRC,"hljs-theme-chat"),loadScriptOnce(HLJS_JS_SRC,"hljs-scr\
ipt")]).then(()=>window.hljs||null).catch(e=>{throw highlightLoadPromise=null,e})),await highlightLoadPromise)}
r(ensureHighlightLoaded,"ensureHighlightLoaded");function maybeNeedsMathJax(e){const n=String(e||"");
return n.includes("$$")||n.includes("\\(")||n.includes("\\[")||n.includes("\\begin{")?!0:/(?<!\$)\$(?!\$)(?=[\s\S]*?[A-Za-z\\^_{}])(?:[^$\n\\]|\\.)+?\$(?!\$)/.
test(n)}r(maybeNeedsMathJax,"maybeNeedsMathJax");function protectMathSegments(e){const n=String(e||""),
i=[],a=r(h=>{const y=`@@MATHJAX_BLOCK_${i.length}@@`;return i.push(h),y},"stash"),o=[],l=/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
let c=0,d;for(;(d=l.exec(n))!==null;){const h=d.index;h>c&&o.push({type:"text",value:n.slice(c,h)}),
o.push({type:"code",value:d[0]}),c=h+d[0].length}return c<n.length&&o.push({type:"text",value:n.slice(
c)}),o.length||o.push({type:"text",value:n}),{text:o.map(h=>{if(h.type==="code")return h.value;let y=h.
value;return y=y.replace(/\$\$([\s\S]+?)\$\$/g,a),y=y.replace(/\\\(([\s\S]+?)\\\)/g,a),y=y.replace(/\\\[([\s\S]+?)\\\]/g,
a),y=y.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g,a),y=y.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g,
a),y}).join(""),blocks:i}}r(protectMathSegments,"protectMathSegments");function getStreamMathSegmentKey(e,n){
const i=String(n||"");let a=2166136261;for(let o=0;o<i.length;o++)a^=i.charCodeAt(o),a=Math.imul(a,16777619);
return`${e}-${i.length}-${(a>>>0).toString(16)}`}r(getStreamMathSegmentKey,"getStreamMathSegmentKey");
function restoreMathSegments(e,n,i={}){return!n||!n.length?String(e||""):String(e||"").replace(/@@MATHJAX_BLOCK_(\d+)@@/g,
(a,o)=>{const l=n[Number(o)];if(l==null)return"";const c=String(l).replace(/&/g,"&amp;").replace(/</g,
"&lt;").replace(/>/g,"&gt;");return i.streamMathSegments?`<span class="stream-math-segment mathjax_p\
rocess" data-stream-math-key="${getStreamMathSegmentKey(Number(o),l)}">${c}</span>`:c})}r(restoreMathSegments,
"restoreMathSegments");function maybeNeedsHighlight(e,n=null){return String(e||"").includes("```")?!0:
!n||typeof n.querySelector!="function"?!1:!!n.querySelector("pre code")}r(maybeNeedsHighlight,"maybe\
NeedsHighlight");function queueMathTypeset(e,n="",i={}){lowBandwidthMode&&!i.force||!e||!maybeNeedsMathJax(
n)||ensureMathJaxLoaded().then(()=>{if(!(!window.MathJax||typeof window.MathJax.typesetPromise!="fun\
ction")){try{typeof window.MathJax.typesetClear=="function"&&window.MathJax.typesetClear([e])}catch{}
return window.MathJax.typesetPromise([e]).catch(()=>{})}}).catch(()=>{})}r(queueMathTypeset,"queueMa\
thTypeset");function queueIncrementalMathTypeset(e){const n=Array.from(e||[]).filter(i=>i&&i.isConnected&&
!i.getAttribute("data-stream-math-state"));!n.length||lowBandwidthMode||(n.forEach(i=>i.setAttribute(
"data-stream-math-state","queued")),incrementalMathTypesetChain=incrementalMathTypesetChain.catch(()=>{}).
then(async()=>{await ensureMathJaxLoaded();const i=n.filter(a=>a.isConnected&&a.getAttribute("data-s\
tream-math-state")==="queued");if(!(!i.length||!window.MathJax||typeof window.MathJax.typesetPromise!=
"function")){i.forEach(a=>a.setAttribute("data-stream-math-state","rendering"));try{await window.MathJax.
typesetPromise(i),i.forEach(a=>{a.isConnected&&a.setAttribute("data-stream-math-state","rendered")})}catch{
i.forEach(o=>o.removeAttribute("data-stream-math-state"))}}}).catch(()=>{n.forEach(i=>i.removeAttribute(
"data-stream-math-state"))}))}r(queueIncrementalMathTypeset,"queueIncrementalMathTypeset");function queueHighlight(e,n="",i={}){
lowBandwidthMode&&!i.force||!e||!maybeNeedsHighlight(n,e)||activeStreamingBubbleId&&e.closest(`#${activeStreamingBubbleId}`)||
ensureHighlightLoaded().then(()=>{window.hljs&&e.querySelectorAll("pre code").forEach(a=>{if(!(a.getAttribute(
"data-highlighted")==="true"&&!i.force))try{window.hljs.highlightElement(a)}catch{}})}).catch(()=>{})}
r(queueHighlight,"queueHighlight");function getNetworkConnectionInfo(){return navigator.connection||
navigator.mozConnection||navigator.webkitConnection||null}r(getNetworkConnectionInfo,"getNetworkConn\
ectionInfo");function detectLowBandwidthModeAuto(){const e=getNetworkConnectionInfo();if(!e)return{enabled:!1,
reason:""};const n=!!e.saveData,i=String(e.effectiveType||"").toLowerCase(),a=Number(e.downlink||0),
o=i==="slow-2g"||i==="2g"||i==="3g",l=Number.isFinite(a)&&a>0&&a<1.3,c=n||o||l,d=[];return n&&d.push(
"\u30C7\u30FC\u30BF\u7BC0\u7D04"),i&&d.push(`\u56DE\u7DDA:${i}`),l&&d.push(`\u4E0B\u308A:${a}Mbps`),
{enabled:c,reason:d.join(" / ")}}r(detectLowBandwidthModeAuto,"detectLowBandwidthModeAuto");function normalizeLowBandwidthModePreference(e){
const n=String(e||"").trim().toLowerCase();return n==="on"||n==="off"||n==="auto"?n:"auto"}r(normalizeLowBandwidthModePreference,
"normalizeLowBandwidthModePreference");function readLowBandwidthModePreference(){try{return normalizeLowBandwidthModePreference(
localStorage.getItem(LOW_BANDWIDTH_MODE_STORAGE_KEY)||"auto")}catch{return"auto"}}r(readLowBandwidthModePreference,
"readLowBandwidthModePreference");function persistLowBandwidthModePreference(e){const n=normalizeLowBandwidthModePreference(
e);lowBandwidthModePreference=n;try{n==="auto"?localStorage.removeItem(LOW_BANDWIDTH_MODE_STORAGE_KEY):
localStorage.setItem(LOW_BANDWIDTH_MODE_STORAGE_KEY,n)}catch{}}r(persistLowBandwidthModePreference,"\
persistLowBandwidthModePreference");function getEffectiveThreadInitialMessageLimit(){return lowBandwidthMode?
LOW_BANDWIDTH_INITIAL_MESSAGE_LIMIT:THREAD_INITIAL_MESSAGE_LIMIT}r(getEffectiveThreadInitialMessageLimit,
"getEffectiveThreadInitialMessageLimit");function getEffectiveThreadOlderPageSize(){return lowBandwidthMode?
LOW_BANDWIDTH_OLDER_PAGE_SIZE:THREAD_OLDER_PAGE_SIZE}r(getEffectiveThreadOlderPageSize,"getEffective\
ThreadOlderPageSize");function mergeBtnClasses(e,n=[],i=[]){e&&(i.forEach(a=>e.classList.remove(a)),
n.forEach(a=>e.classList.add(a)))}r(mergeBtnClasses,"mergeBtnClasses");function updateLowBandwidthModeUi(){
const e=get("low-bandwidth-toggle-btn"),n=get("low-bandwidth-status-pill"),i=lowBandwidthModePreference===
"auto"?"\u81EA\u52D5":lowBandwidthModePreference==="on"?"\u56FA\u5B9AON":"\u56FA\u5B9AOFF",a=lowBandwidthMode?
"ON":"OFF",o=lowBandwidthModeReason?` (${lowBandwidthModeReason})`:"";if(e&&(e.setAttribute("title",
`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9 ${a} / ${i}${o}`),e.setAttribute("aria-pressed",lowBandwidthMode?
"true":"false"),lowBandwidthMode?mergeBtnClasses(e,["text-amber-200","bg-amber-900/30","border","bor\
der-amber-600/40"],["text-gray-400"]):mergeBtnClasses(e,["text-gray-400"],["text-amber-200","bg-ambe\
r-900/30","border","border-amber-600/40"])),n)if(lowBandwidthMode){n.classList.remove("hidden");const l=lowBandwidthModePreference===
"auto"?" (\u81EA\u52D5)":" (\u624B\u52D5)";n.innerHTML=`<i class="fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9${l}${lowBandwidthModeReason?
`: ${escapeHtml(lowBandwidthModeReason)}`:""}`}else n.classList.add("hidden"),n.innerHTML='<i class=\
"fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9'}r(updateLowBandwidthModeUi,"updat\
eLowBandwidthModeUi");function refreshDecorationsForVisibleChat(){const e=get("chat-container");e&&(queueHighlight(
e,e.textContent||"",{force:!0}),queueMathTypeset(e,e.textContent||"",{force:!0}))}r(refreshDecorationsForVisibleChat,
"refreshDecorationsForVisibleChat");function applyLowBandwidthModeState(e,n={}){const i=lowBandwidthMode;
if(lowBandwidthMode=!!e,updateLowBandwidthModeUi(),i&&!lowBandwidthMode&&refreshDecorationsForVisibleChat(),
n.notify){const a=lowBandwidthModePreference==="auto"?"\u81EA\u52D5":"\u624B\u52D5",o=lowBandwidthModeReason?
` (${lowBandwidthModeReason})`:"";showToast(`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9\u3092${lowBandwidthMode?
"ON":"OFF"}\u306B\u3057\u307E\u3057\u305F [${a}]${o}`,"info",!1)}}r(applyLowBandwidthModeState,"appl\
yLowBandwidthModeState");function recomputeLowBandwidthMode(e={}){const n=detectLowBandwidthModeAuto();
lowBandwidthModeAuto=!!n.enabled,lowBandwidthModeReason=n.reason||"",applyLowBandwidthModeState(lowBandwidthModePreference===
"on"?!0:lowBandwidthModePreference==="off"?!1:lowBandwidthModeAuto,e)}r(recomputeLowBandwidthMode,"r\
ecomputeLowBandwidthMode");function cycleLowBandwidthModePreference(){const e=normalizeLowBandwidthModePreference(
lowBandwidthModePreference);persistLowBandwidthModePreference(e==="auto"?"on":e==="on"?"off":"auto"),
recomputeLowBandwidthMode({notify:!0})}r(cycleLowBandwidthModePreference,"cycleLowBandwidthModePrefe\
rence");function ensureDeferredDecorationObserver(){if(deferredDecorationObserver||typeof IntersectionObserver==
"undefined")return deferredDecorationObserver;const e=get("chat-container")||null;return deferredDecorationObserver=
new IntersectionObserver(n=>{n.forEach(i=>{!i.isIntersecting||!i.target||runDeferredDecorations(i.target)})},
{root:e,threshold:LOW_BANDWIDTH_DECORATION_VISIBILITY_THRESHOLD}),deferredDecorationObserver}r(ensureDeferredDecorationObserver,
"ensureDeferredDecorationObserver");function runDeferredDecorations(e){if(!e)return;if(deferredDecorationObserver)
try{deferredDecorationObserver.unobserve(e)}catch{}const n=deferredDecorationTextMap.get(e)||"";queueHighlight(
e,n,{force:!0}),queueMathTypeset(e,n,{force:!0})}r(runDeferredDecorations,"runDeferredDecorations");
function queueMessageDecorations(e,n=""){if(!e)return;if(!lowBandwidthMode){queueHighlight(e,n),queueMathTypeset(
e,n);return}if(!maybeNeedsHighlight(n,e)&&!maybeNeedsMathJax(n))return;deferredDecorationTextMap.set(
e,String(n||""));const i=get("chat-container");if(i&&e===i){window.setTimeout(()=>runDeferredDecorations(
e),250);return}if(!e.isConnected)return;const a=ensureDeferredDecorationObserver();if(a){a.observe(e);
return}window.setTimeout(()=>runDeferredDecorations(e),250)}r(queueMessageDecorations,"queueMessageD\
ecorations");function initLowBandwidthMode(){lowBandwidthModePreference=readLowBandwidthModePreference(),
recomputeLowBandwidthMode({notify:!1});const e=get("low-bandwidth-toggle-btn");e&&!e.__lowBandwidthBound&&
(e.__lowBandwidthBound=!0,e.addEventListener("click",i=>{i&&i.preventDefault(),cycleLowBandwidthModePreference()}));
const n=getNetworkConnectionInfo();n&&typeof n.addEventListener=="function"&&!lowBandwidthConnectionListenerAttached&&
(lowBandwidthConnectionListenerAttached=!0,n.addEventListener("change",()=>{if(lowBandwidthModePreference===
"auto")recomputeLowBandwidthMode({notify:!0});else{const i=detectLowBandwidthModeAuto();lowBandwidthModeAuto=
!!i.enabled,lowBandwidthModeReason=i.reason||"",updateLowBandwidthModeUi()}}))}r(initLowBandwidthMode,
"initLowBandwidthMode");function escapeHtml(e){return e==null?"":String(e).replace(/&/g,"&amp;").replace(
/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;")}r(escapeHtml,"escape\
Html");const BLOCKED_SCRIPT_HOSTS=["polyfill.io","cdn.polyfill.io"];function isBlockedScriptSrc(e){if(!e)
return!1;const n=String(e).trim();if(!n)return!1;let i=n;n.startsWith("//")?i="https:"+n:!/^https?:\/\//i.
test(n)&&!n.startsWith("data:")&&!n.startsWith("blob:")&&(i="https://"+n);try{const o=(new URL(i,"ht\
tps://example.com").hostname||"").toLowerCase();return BLOCKED_SCRIPT_HOSTS.some(l=>o===l||o.endsWith(
"."+l))}catch{return/polyfill\.io/i.test(n)}}r(isBlockedScriptSrc,"isBlockedScriptSrc");function isPasswordPromptingScript(e){
if(!e)return!1;const n=String(e),i=n.toLowerCase();return!!(/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.
test(n)||/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(n)||
/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.
test(i)||/prompt\s*\(/.test(n)&&/(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.
test(n))}r(isPasswordPromptingScript,"isPasswordPromptingScript");function detectBlockedScriptsInCode(e){
if(!e)return!1;const n=String(e),i=/<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;let a;for(;(a=i.exec(
n))!==null;)if(isBlockedScriptSrc(a[1]))return!0;const o=/<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
for(;(a=o.exec(n))!==null;)if(isPasswordPromptingScript(a[1]))return!0;return!!(/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.
test(n)||/src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(n))}r(detectBlockedScriptsInCode,"detectBlockedS\
criptsInCode");function sanitizeHtmlForPreview(e){if(!e)return"";const n=detectBlockedScriptsInCode(
e);let i=String(e);try{const o=new DOMParser().parseFromString(i,"text/html");let l=!1;o.querySelectorAll(
"script").forEach(d=>{const m=d.getAttribute("src")||"";let h=!1;if(m&&isBlockedScriptSrc(m)){const y=o.
createElement("div");y.setAttribute("data-blocked-script","true"),y.style.cssText="background:#fee2e\
2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6p\
x 0;font-family:system-ui;";const v=m.length>70?m.slice(0,67)+"...":m;y.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: "+
v+" \uFF08polyfill.io \u306A\u3069\u306E\u5371\u967A\u30C9\u30E1\u30A4\u30F3\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u7121\u52B9\u5316\u3055\u308C\u307E\u3059\uFF09",
d.parentNode&&d.parentNode.replaceChild(y,d),l=!0,h=!0}else if(!m){const y=d.textContent||"";if(isPasswordPromptingScript(
y)){const v=o.createElement("div");v.setAttribute("data-blocked-script","true"),v.style.cssText="bac\
kground:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:\
12px;margin:6px 0;font-family:system-ui;",v.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: \u30D1\u30B9\u30EF\u30FC\u30C9\u5165\u529B\u8981\u6C42\u306A\u3069\u306E\u7591\u308F\u3057\u3044\u30A4\u30F3\u30E9\u30A4\u30F3\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\
\u305F",d.parentNode&&d.parentNode.replaceChild(v,d),l=!0,h=!0}}}),o.querySelectorAll('a[href^="java\
script:" i], area[href^="javascript:" i]').forEach(d=>{d.setAttribute("href","#"),d.setAttribute("ti\
tle",(d.getAttribute("title")||"")+" [javascript: disabled in preview]")});const c=o.head||o.querySelector(
"head");if(c&&!c.querySelector("base")){const d=o.createElement("base");d.setAttribute("href",`${window.
location.origin}/`),c.insertBefore(d,c.firstChild)}if(n||l){const d=o.body||o.documentElement;if(d){
const m=o.createElement("div");m.style.cssText="position:sticky;top:0;left:0;right:0;z-index:2147483\
647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:syst\
em-ui;border-bottom:1px solid #b91c1c;",m.innerHTML="\u26A0 <strong>\u5B89\u5168\u30D7\u30EC\u30D3\u30E5\u30FC</strong>: polyfill.io \u306A\u3069\u306E\u5371\u967A\u306A\u30B9\
\u30AF\u30EA\u30D7\u30C8\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002\u5B9F\u884C\u306F\u81EA\u5DF1\u8CAC\u4EFB\u3067\u3002",
d.firstChild?d.insertBefore(m,d.firstChild):d.appendChild(m)}}i=`<!DOCTYPE html>
`+(o.documentElement?o.documentElement.outerHTML:i)}catch{i=i.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
"<!-- blocked polyfill.io script for safety -->")}return i}r(sanitizeHtmlForPreview,"sanitizeHtmlFor\
Preview");function wrapTextWave(e){return e?e.split("").map((n,i)=>`<span class="wave-char" style="a\
nimation-delay: ${i*.028}s">${escapeHtml(n)}</span>`).join(""):""}r(wrapTextWave,"wrapTextWave");function getPendingSkeletonKind(e){
let n=String(e||"").toLowerCase();if(!n)try{n=String(get("model-select")&&get("model-select").value||
"").toLowerCase()}catch{n=""}return n.includes("video")?"video":n.includes("tts")||n.includes("trans\
cribe")||n.includes("realtime")||n.includes("voice")||n.includes("native-audio")||n.includes("live")&&
n.includes("gemini")?"audio":n.includes("gpt-image")||n.includes("imagine-image")||n.includes("image")&&
!n.includes("vision")||n.includes("gemini")&&(n.includes("image")||n.includes("nano"))?"image":n.includes(
"ocr")||n.includes("mistral-ocr")?"text":n.includes("build")||n.includes("code-fast")||n.includes("c\
oding")?"code":"text"}r(getPendingSkeletonKind,"getPendingSkeletonKind");function buildPendingSkeletonBody(e){
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
/div>'}r(buildPendingSkeletonBody,"buildPendingSkeletonBody");function buildPendingSkeletonHtml(e,n){
const i=getPendingSkeletonKind(e),a=n==null||n===""?"\u56DE\u7B54\u3092\u751F\u6210\u4E2D...":String(
n);return`<div class="content-area pending-shimmer skeleton-pending" data-skeleton-kind="${escapeHtml(
i)}">${buildPendingSkeletonBody(i)}<div class="skeleton-status">${escapeHtml(a)}</div></div>`}r(buildPendingSkeletonHtml,
"buildPendingSkeletonHtml");function updatePendingSkeletonStatus(e,n,i){if(!e)return!1;const a=e.querySelector(
".content-area.skeleton-pending");if(!a)return!1;let o=a.querySelector(".skeleton-status");o||(o=document.
createElement("div"),o.className="skeleton-status",a.appendChild(o));const l=n==null?"":String(n),c=i==
null||i===""?"":String(i);return c?o.innerHTML=`${escapeHtml(l)}<span class="skeleton-status-sub">${escapeHtml(
c)}</span>`:o.textContent=l,!0}r(updatePendingSkeletonStatus,"updatePendingSkeletonStatus");function buildChatLoadingSkeletonHtml(){
return`<div class="chat-load-skeleton" role="status" aria-live="polite" aria-label="\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D">${[
{role:"user",widths:["62%","44%"]},{role:"ai",widths:["88%","76%","92%","58%"]},{role:"user",widths:[
"48%"]},{role:"ai",widths:["82%","70%","54%"]}].map((i,a)=>{const o=i.role==="user",l=o?"justify-end":
"justify-start",c=o?"message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4\
 rounded-2xl rounded-tr-none shadow-md relative":"message-bubble chat-load-skeleton-bubble chat-load\
-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative",d=i.widths.map(
(m,h)=>`<div class="skeleton-line" style="width:${m};animation-delay:${(a*.08+h*.06).toFixed(2)}s"><\
/div>`).join("");return`<div class="flex ${l} mb-4 chat-load-skeleton-row" style="animation-delay:${(a*
.07).toFixed(2)}s" aria-hidden="true"><div class="${c}"><div class="content-area pending-shimmer ske\
leton-pending chat-load-skeleton-body" data-skeleton-kind="text"><div class="skeleton-lines">${d}</d\
iv></div></div></div>`}).join("")}<div class="chat-load-skeleton-caption"><span class="chat-load-ske\
leton-caption-dot"></span>\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D...</div></div>`}
r(buildChatLoadingSkeletonHtml,"buildChatLoadingSkeletonHtml");function showChatLoadError(e){const n=get(
"chat-container");if(!n)return;n.innerHTML='<div class="min-h-[45vh] flex items-center justify-cente\
r px-4"><div class="max-w-md w-full rounded-2xl border border-red-500/40 bg-red-950/30 p-5 text-cent\
er" role="alert"><i class="fas fa-triangle-exclamation text-red-300 text-xl mb-3"></i><p class="text\
-sm font-semibold text-red-100">\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F</p><p class="mt-2 text-xs text-red-200/80">\u901A\u4FE1\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\
\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p><button type="button" data-chat-load-retry class="mt-4 rounded-lg border border-red\
-300/40 px-4 py-2 text-sm text-red-100 hover:bg-red-500/20"><i class="fas fa-rotate-right mr-1"></i>\
\u518D\u8A66\u884C</button></div></div>';const i=n.querySelector("[data-chat-load-retry]");i&&i.addEventListener(
"click",()=>loadMessages(e))}r(showChatLoadError,"showChatLoadError");function hashString(e){let n=0;
if(!e)return"0";for(let i=0;i<e.length;i++)n=(n<<5)-n+e.charCodeAt(i),n|=0;return Math.abs(n).toString(
36)}r(hashString,"hashString");function decodeCodeButtonValue(e){if(!e)return"";try{return decodeURIComponent(
e)}catch{return""}}r(decodeCodeButtonValue,"decodeCodeButtonValue");function getCodingTargetFromButton(e){
if(!e)return null;const n=decodeCodeButtonValue(e.getAttribute("data-code")||"");if(!n)return null;const i=e.
closest(".code-wrapper"),a=e.closest(".message-group");return{code:n,language:String(e.getAttribute(
"data-coding-lang")||"text").trim().slice(0,40)||"text",key:String(e.getAttribute("data-code-key")||
(i==null?void 0:i.getAttribute("data-code-key"))||hashString(n)),message_id:a!=null&&a.id?a.id.replace(
/^msg-/,""):null,thread_id:currentThreadId?String(currentThreadId):null}}r(getCodingTargetFromButton,
"getCodingTargetFromButton");function findLatestCodingTarget(){const e=get("chat-container");if(!e)return null;
const n=Array.from(e.querySelectorAll(".message-group .coding-target-btn"));for(let i=n.length-1;i>=
0;i--){const a=getCodingTargetFromButton(n[i]);if(a)return a}return null}r(findLatestCodingTarget,"f\
indLatestCodingTarget");function extractPromptCodingTargets(e){const n=String(e||"").replace(/\r\n?/g,
`
`).split(`
`),i=[];let a=null;for(const o of n){if(!a){const d=o.match(/^\s*(`{3,}|~{3,})(.*)$/);if(!d)continue;
const m=String(d[2]||"").trim();a={markerChar:d[1][0],markerLength:d[1].length,language:(m.split(/\s+/)[0]||
"text").replace(/^\{?\.?/,"").replace(/\}$/,"")||"text",buffer:[]};continue}const l=String(o||"").trim();
if(new RegExp(`^\\${a.markerChar}{${a.markerLength},}\\s*$`).test(l)){const d=a.buffer.join(`
`);d.trim()&&i.push({code:d,language:a.language,key:hashString(`prompt\\n${a.language}\\n${d}`),candidate_id:`\
prompt-${i.length+1}`,prompt_index:i.length,message_id:null,thread_id:currentThreadId?String(currentThreadId):
null,prompt_source:!0}),a=null;continue}a.buffer.push(o)}return i}r(extractPromptCodingTargets,"extr\
actPromptCodingTargets");function extractLatestPromptCodingTarget(e){const n=extractPromptCodingTargets(
e);return n.length?n[n.length-1]:null}r(extractLatestPromptCodingTarget,"extractLatestPromptCodingTa\
rget");function collectCodingCandidates(e){if(codingTargetSelection){const l=codingTargetSelection.thread_id;
if(!l||!currentThreadId||String(l)===String(currentThreadId))return[{...codingTargetSelection,candidate_id:"\
selected-1",source:"history",explicit:!0}];codingTargetSelection=null}const n=extractPromptCodingTargets(
e),i=new Set(n.map(l=>`${l.language}
${l.code}`)),a=get("chat-container"),o=[];return a&&Array.from(a.querySelectorAll(".message-group .c\
oding-target-btn")).forEach(l=>{const c=getCodingTargetFromButton(l);if(!c)return;const d=`${c.language}\

${c.code}`;i.has(d)||(i.add(d),o.push(c))}),o.slice(-20).forEach((l,c)=>{n.push({...l,candidate_id:`\
history-${c+1}`,source:"history",explicit:!1})}),n}r(collectCodingCandidates,"collectCodingCandidate\
s");function resolveCodingTarget(e=null){var o;const n=String(e===null?((o=get("prompt-input"))==null?
void 0:o.value)||"":e||"");if(codingTargetSelection){const l=codingTargetSelection.thread_id;if(!l||
!currentThreadId||String(l)===String(currentThreadId))return{...codingTargetSelection,explicit:!0};codingTargetSelection=
null}const i=extractLatestPromptCodingTarget(n);if(i)return{...i,explicit:!1};const a=findLatestCodingTarget();
return a?{...a,explicit:!1}:null}r(resolveCodingTarget,"resolveCodingTarget");function syncCodingTargetButtons(e=document){
if(!e||typeof e.querySelectorAll!="function")return;const n=codingTargetSelection?String(codingTargetSelection.
key||""):"";e.querySelectorAll(".coding-target-btn").forEach(i=>{const a=!!n&&String(i.getAttribute(
"data-code-key")||"")===n;i.classList.toggle("coding-target-active",a),i.setAttribute("aria-pressed",
a?"true":"false"),i.innerHTML=a?'<i class="fas fa-thumbtack"></i>':'<i class="fas fa-quote-right"></\
i>',i.title=a?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A",
i.setAttribute("aria-label",a?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A")})}
r(syncCodingTargetButtons,"syncCodingTargetButtons");function syncCodingModeUi(e=codingModeEnabled,n={}){
var m;if(codingModeEnabled=!!e,n.persist!==!1)try{localStorage.setItem(CODING_MODE_STORAGE_KEY,codingModeEnabled?
"true":"false")}catch{}const i=get("enable-coding-mode");i&&i.checked!==codingModeEnabled&&(i.checked=
codingModeEnabled);const a=get("coding-target-bar"),o=get("coding-target-text"),l=get("clear-coding-\
target-btn");a&&a.classList.toggle("visible",codingModeEnabled);const c=resolveCodingTarget(),d=codingTargetSelection?
[c].filter(Boolean):collectCodingCandidates(String(((m=get("prompt-input"))==null?void 0:m.value)||""));
if(codingModeEffective=codingModeEnabled&&d.length>0,o)if(codingTargetSelection&&c)o.textContent=`\u7DE8\u96C6\
\u5BFE\u8C61: ${c.language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`;else if(d.length>1){
const h=d.filter(v=>v.prompt_source).length,y=d.length-h;o.textContent=`\u30E2\u30C7\u30EB\u304C\u7DE8\u96C6\u5BFE\u8C61\u3092\u5224\u65AD: \u5165\u529B${h}\
\u4EF6 / \u5C65\u6B74${y}\u4EF6`}else c&&c.prompt_source?o.textContent=`\u5165\u529B\u4E2D: ${c.language||
"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:c?o.textContent=`\u81EA\u52D5\u9078\u629E: \u6700\u65B0\u306E ${c.
language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:o.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u751F\u6210\u5F8C\u306B\u81EA\u52D5\u6709\u52B9\u5316";
l&&l.classList.toggle("hidden",!codingTargetSelection),syncCodingTargetButtons()}r(syncCodingModeUi,
"syncCodingModeUi");function activateDeferredCodingModeFromStream(e){if(!codingModeEnabled||codingModeEffective||
extractPromptCodingTargets(e).length===0)return!1;codingModeEffective=!0;const n=get("coding-target-\
text");return n&&(n.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u691C\u51FA: \u6B21\u306E\u9001\u4FE1\u304B\u3089\u6709\u52B9"),
!0}r(activateDeferredCodingModeFromStream,"activateDeferredCodingModeFromStream");function selectCodingTargetFromButton(e){
const n=getCodingTargetFromButton(e);if(!n){showToast("\u3053\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u7DE8\u96C6\u5BFE\u8C61\u306B\u3067\u304D\u307E\u305B\u3093",
"error",!0);return}codingTargetSelection=n,syncCodingModeUi(codingModeEnabled,{persist:!1}),codingModeEnabled?
showToast("Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u3057\u307E\u3057\u305F","suc\
cess"):showToast("\u7DE8\u96C6\u5BFE\u8C61\u3092\u9078\u629E\u3057\u307E\u3057\u305F\u3002\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306ECoding\u3092\u30AA\u30F3\u306B\u3059\u308B\u3068\u4F7F\u7528\u3057\u307E\u3059",
"info")}r(selectCodingTargetFromButton,"selectCodingTargetFromButton");function renderCodingDiffLines(e){
return String(e||"").split(`
`).map(n=>{let i="coding-diff-context";return n.startsWith("+++")||n.startsWith("---")?i="coding-dif\
f-file":n.startsWith("@@")?i="coding-diff-hunk":n.startsWith("+")?i="coding-diff-added":n.startsWith(
"-")&&(i="coding-diff-removed"),`<span class="${i}">${escapeHtml(n||" ")}</span>`}).join(`
`)}r(renderCodingDiffLines,"renderCodingDiffLines");function appendCodingLiveDiff(e,n){if(!e||!n||!n.
diff)return;let i=e.querySelector(".coding-live-diff");i||(i=document.createElement("div"),i.className=
"coding-live-diff",i.innerHTML='<div class="coding-live-diff-header"><span><i class="fas fa-code-bra\
nch"></i> Live Code Changes</span><span class="coding-live-diff-count">0 edits</span></div><div clas\
s="coding-live-diff-list"></div>',e.appendChild(i));const a=Math.max(0,Number(n.edit_index||0));if(a&&
i.querySelector(`[data-coding-edit-index="${a}"]`))return;const o=i.querySelector(".coding-live-diff\
-list"),l=document.createElement("div");l.className="coding-live-diff-edit",a&&l.setAttribute("data-\
coding-edit-index",String(a));const c=Number(n.repair_attempt||0)>0?` \xB7 Auto repair ${Number(n.repair_attempt)}`:
"";l.innerHTML=`<div class="coding-live-diff-meta">Edit ${a} \xB7 ${escapeHtml(n.language||"text")}${c}\
</div><pre>${renderCodingDiffLines(n.diff)}</pre>`,o&&o.appendChild(l);const d=i.querySelector(".cod\
ing-live-diff-count"),m=i.querySelectorAll(".coding-live-diff-edit").length;d&&(d.textContent=`${m} \
edit${m===1?"":"s"}`),i.scrollIntoView({block:"nearest",behavior:"smooth"})}r(appendCodingLiveDiff,"\
appendCodingLiveDiff");function isHtmlPreviewCandidate(e,n){const i=String(e||"").trim().toLowerCase();
return i==="html"||i==="htm"||i==="xhtml"?!0:i?!1:/<!doctype\s+html/i.test(n||"")}r(isHtmlPreviewCandidate,
"isHtmlPreviewCandidate");function openHtmlCodePreview(e){if(!e)return;let n="";try{n=decodeURIComponent(
e)}catch{showToast("HTML\u30D7\u30EC\u30D3\u30E5\u30FC\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}detectBlockedScriptsInCode(n)&&showToast("\u26A0 \u5371\u967A\u306A\u5916\u90E8\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u691C\u77E5 (polyfill.io \u306A\u3069)\u3002\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\
\u306F\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u958B\u304D\u307E\u3059\u3002","warning",!0);const a=sanitizeHtmlForPreview(
n);openSandboxedHtmlTab(a)}r(openHtmlCodePreview,"openHtmlCodePreview");function snapshotCodeCollapse(e){
if(!e)return[];const n=[];return e.querySelectorAll(".code-wrapper").forEach((i,a)=>{const o=String(
a),l=i.classList.contains("collapsed")||i.getAttribute("data-collapsed")==="true";n.push({key:o,collapsed:l})}),
n}r(snapshotCodeCollapse,"snapshotCodeCollapse");function applyCodeCollapse(e,n=[],i=!1){if(!e)return;
const a=new Map;n.forEach(o=>a.set(o.key,o.collapsed)),e.querySelectorAll(".code-wrapper").forEach((o,l)=>{
const c=String(l),d=a.has(c)?a.get(c):i;o.setAttribute("data-collapsed",d?"true":"false"),o.classList.
toggle("collapsed",!!d);const m=o.querySelector(".code-toggle");m&&(m.setAttribute("aria-expanded",d?
"false":"true"),m.innerHTML=d?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></\
i>',m.title=d?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",m.setAttribute("aria-label",d?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080"))})}r(applyCodeCollapse,"applyCodeCollapse");function snapshotCodeCollapseByMessage(e){
if(!e)return new Map;const n=new Map;return e.querySelectorAll(".message-group").forEach(i=>{const a=i.
getAttribute("id")||"";i.querySelectorAll(".code-wrapper").forEach((o,l)=>{const c=o.getAttribute("d\
ata-code-key")||String(l),d=o.classList.contains("collapsed")||o.getAttribute("data-collapsed")==="t\
rue";n.set(`${a}:${c}`,d)})}),n}r(snapshotCodeCollapseByMessage,"snapshotCodeCollapseByMessage");function applyCodeCollapseByMessage(e,n,i=!1){
e&&e.querySelectorAll(".message-group").forEach(a=>{const o=a.getAttribute("id")||"";a.querySelectorAll(
".code-wrapper").forEach((l,c)=>{const d=l.getAttribute("data-code-key")||String(c),m=`${o}:${d}`,h=n&&
n.has(m)?n.get(m):i;l.setAttribute("data-collapsed",h?"true":"false"),l.classList.toggle("collapsed",
!!h);const y=l.querySelector(".code-toggle");y&&(y.setAttribute("aria-expanded",h?"false":"true"),y.
innerHTML=h?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></i>',y.title=h?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080",y.setAttribute("aria-label",h?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080"))})})}
r(applyCodeCollapseByMessage,"applyCodeCollapseByMessage");function buildTokenTotals(e){const n={tokens_total:0,
tokens_in:0,tokens_out:0,tokens_content:0,tokens_thought:0};let i=!1,a=!1,o=!1,l=!1,c=!1;return(e||[]).
forEach(d=>{if(!d)return;let m=null;d.tokens!==null&&d.tokens!==void 0?m=Number(d.tokens||0):(d.tokens_in!==
null&&d.tokens_in!==void 0||d.tokens_out!==null&&d.tokens_out!==void 0)&&(m=Number(d.tokens_in||0)+Number(
d.tokens_out||0)),m!==null&&(n.tokens_total+=m,i=!0),d.tokens_in!==null&&d.tokens_in!==void 0&&(n.tokens_in+=
Number(d.tokens_in||0),a=!0),d.tokens_out!==null&&d.tokens_out!==void 0&&(n.tokens_out+=Number(d.tokens_out||
0),o=!0),d.tokens_content!==null&&d.tokens_content!==void 0&&(n.tokens_content+=Number(d.tokens_content||
0),l=!0),d.tokens_thought!==null&&d.tokens_thought!==void 0&&(n.tokens_thought+=Number(d.tokens_thought||
0),c=!0)}),{tokens_total:i?n.tokens_total:0,tokens_in:a?n.tokens_in:null,tokens_out:o?n.tokens_out:null,
tokens_content:l?n.tokens_content:null,tokens_thought:c?n.tokens_thought:null}}r(buildTokenTotals,"b\
uildTokenTotals");function updateTotalTokenBar(e,n=null,i=null){const a=get("total-token-bar"),o=get(
"total-token-count"),l=get("total-token-count-all-branches");if(!a||!o)return;const c=Number(e||0),d=Number(
i&&i.tokens_total||0);c>0||d>0?(a.classList.remove("hidden"),o.innerText=`Total: ${c} tokens`,n?(o.classList.
add("cursor-pointer","underline","decoration-dotted"),messageMeta.__total__={tokens_total:c,tokens_in:n.
tokens_in,tokens_out:n.tokens_out,tokens_content:n.tokens_content,tokens_thought:n.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation"},o.onclick=()=>openTokenDetail("__total__")):(o.classList.remove("\
cursor-pointer","underline","decoration-dotted"),o.onclick=null,delete messageMeta.__total__),l&&(i&&
d>0?(l.classList.remove("hidden"),l.classList.add("cursor-pointer","underline","decoration-dotted"),
l.innerText=`All branches: ${d} tokens`,messageMeta.__total_all_branches__={tokens_total:d,tokens_in:i.
tokens_in,tokens_out:i.tokens_out,tokens_content:i.tokens_content,tokens_thought:i.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation (All branches)"},l.onclick=()=>openTokenDetail("__total_all_branche\
s__")):(l.classList.add("hidden"),l.classList.remove("cursor-pointer","underline","decoration-dotted"),
l.innerText="All branches: 0 tokens",l.onclick=null,delete messageMeta.__total_all_branches__))):(a.
classList.add("hidden"),o.innerText="Total: 0 tokens",o.classList.remove("cursor-pointer","underline",
"decoration-dotted"),o.onclick=null,delete messageMeta.__total__,l&&(l.classList.add("hidden"),l.classList.
remove("cursor-pointer","underline","decoration-dotted"),l.innerText="All branches: 0 tokens",l.onclick=
null),delete messageMeta.__total_all_branches__)}r(updateTotalTokenBar,"updateTotalTokenBar");const PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS=300;
let promptTokenEstimateTimer=null,promptTokenEstimateAbort=null,promptTokenEstimateSeq=0,promptTokenEstimateLastKey="",
promptTokenEstimateLastData=null;function setPromptTokenEstimateText(e,n="text-gray-400"){const i=get(
"prompt-token-estimate");if(i){if(!e){i.classList.add("hidden"),i.innerText="";return}i.className=`m\
t-1 px-1 text-[10px] ${n}`,i.classList.remove("hidden"),i.innerText=e}}r(setPromptTokenEstimateText,
"setPromptTokenEstimateText");function buildPromptTokenEstimatePayload(){return{model:get("model-sel\
ect")&&get("model-select").value?get("model-select").value:"",message:get("prompt-input")&&get("prom\
pt-input").value?get("prompt-input").value:"",quote_text:currentQuote||"",image_urls:collectImageUrlsForSend()}}
r(buildPromptTokenEstimatePayload,"buildPromptTokenEstimatePayload");function renderPromptTokenEstimate(e,n=null){
const i=n||buildPromptTokenEstimatePayload(),a=!!((i.message||"").trim()||(i.quote_text||"").trim()),
o=Array.isArray(i.image_urls)&&i.image_urls.length>0;if(!a&&!o){setPromptTokenEstimateText("");return}
if(e&&e.pending){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u4E2D...",
"text-gray-500");return}if(!e){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"text-red-300");return}if(!e.countable){setPromptTokenEstimateText("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u8868\u793A\u5BFE\u8C61\u5916\u3067\u3059",
"text-gray-500");return}const l=Number(e.tokens_total||0),c=Number(e.tokens_prompt||0),d=Number(e.tokens_files||
0),m=[];Number(e.files_non_text||0)>0&&m.push(`\u975E\u30C6\u30AD\u30B9\u30C8${e.files_non_text}\u4EF6\u306F0\u63DB\
\u7B97`),Number(e.files_missing||0)>0&&m.push(`\u672A\u691C\u51FA${e.files_missing}\u4EF6`),Number(e.
files_error||0)>0&&m.push(`\u5931\u6557${e.files_error}\u4EF6`);const h=m.length?` \u30FB ${m.join("\
 / ")}`:"";setPromptTokenEstimateText(`\u5165\u529B\u898B\u7A4D: ${l} tokens (\u672C\u6587 ${c} / \u30D5\u30A1\
\u30A4\u30EB ${d})${h}`,"text-cyan-300")}r(renderPromptTokenEstimate,"renderPromptTokenEstimate");function schedulePromptTokenEstimate(e=!1){
const n=buildPromptTokenEstimatePayload(),i=!!((n.message||"").trim()||(n.quote_text||"").trim()),a=Array.
isArray(n.image_urls)&&n.image_urls.length>0;if(!i&&!a){promptTokenEstimateLastKey="",promptTokenEstimateLastData=
null,promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=null),
promptTokenEstimateAbort&&(promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=null),renderPromptTokenEstimate(
null,n);return}const o=JSON.stringify([n.model||"",n.message||"",n.quote_text||"",n.image_urls||[]]);
if(o===promptTokenEstimateLastKey&&promptTokenEstimateLastData){renderPromptTokenEstimate(promptTokenEstimateLastData,
n);return}promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=
null);const l=r(async()=>{promptTokenEstimateAbort&&promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=
new AbortController;const c=++promptTokenEstimateSeq;renderPromptTokenEstimate({pending:!0},n);try{const d=await apiFetch(
CHAT_CONFIG.urls.estimatePromptTokensApi,{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(n),signal:promptTokenEstimateAbort.signal});if(!d.ok)throw new Error(`HTTP ${d.status}`);
const m=await d.json();if(c!==promptTokenEstimateSeq)return;promptTokenEstimateLastKey=o,promptTokenEstimateLastData=
m,renderPromptTokenEstimate(m,n)}catch(d){if(d&&d.name==="AbortError"||c!==promptTokenEstimateSeq)return;
promptTokenEstimateLastKey="",promptTokenEstimateLastData=null,renderPromptTokenEstimate(null,n)}},"\
run");e?l():promptTokenEstimateTimer=setTimeout(l,PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS)}r(schedulePromptTokenEstimate,
"schedulePromptTokenEstimate");function updatePromptPlaceholder(){const e=get("prompt-input");e&&(editingMessageId?
e.placeholder="\u7DE8\u96C6\u4E2D... (Enter\u9001\u4FE1\u306F\u8A2D\u5B9A\u306B\u5F93\u3044\u307E\u3059)":
enterToSend?e.placeholder="Enter \u3067\u9001\u4FE1 (Shift+Enter \u3067\u6539\u884C)":e.placeholder=
"Ctrl + Enter \u3067\u9001\u4FE1...")}r(updatePromptPlaceholder,"updatePromptPlaceholder");function readPromptBarModeFromForm(){
return get("set-minimal-prompt-mode")&&get("set-minimal-prompt-mode").checked?{compact_prompt_mode:!1,
minimal_prompt_mode:!0}:get("set-compact-prompt-mode")&&get("set-compact-prompt-mode").checked?{compact_prompt_mode:!0,
minimal_prompt_mode:!1}:{compact_prompt_mode:!1,minimal_prompt_mode:!1}}r(readPromptBarModeFromForm,
"readPromptBarModeFromForm");function writePromptBarModeToForm(e,n){const i=get("set-prompt-bar-mode\
-normal"),a=get("set-compact-prompt-mode"),o=get("set-minimal-prompt-mode");n&&o?o.checked=!0:e&&a?a.
checked=!0:i&&(i.checked=!0)}r(writePromptBarModeToForm,"writePromptBarModeToForm");function placeModelSelectorButton(){
const e=get("model-selector-btn"),n=get("top-model-bar"),i=get("prompt-primary-controls"),a=get("mod\
el-select");if(!(!e||!n||!i)){if(minimalPromptMode){e.parentElement!==n&&n.appendChild(e);return}if(a&&
a.parentElement===i){e.previousElementSibling!==a&&a.insertAdjacentElement("afterend",e);return}e.parentElement!==
i&&i.insertBefore(e,i.firstChild)}}r(placeModelSelectorButton,"placeModelSelectorButton");function applyMinimalPromptMode(){
const e=!!minimalPromptMode;document.body.classList.toggle("minimal-prompt-mode",e);const n=get("top\
-model-bar");n&&(n.classList.toggle("hidden",!e),n.classList.toggle("flex",e));const i=get("upload-b\
tn"),a=i?i.querySelector("i"):null;a&&(a.className=e?"fas fa-plus":"fas fa-paperclip"),i&&(i.title=e?
"\u30AA\u30D7\u30B7\u30E7\u30F3":"Upload"),e||(closeMinimalOptions(),hideThinkingSlider()),placeModelSelectorButton()}
r(applyMinimalPromptMode,"applyMinimalPromptMode");function applyPromptControlMode(){const e=get("pr\
ompt-details-controls"),n=get("prompt-controls-toggle-btn"),i=get("prompt-controls-toggle-text"),a=get(
"prompt-controls-toggle-icon"),o=get("prompt-controls-row");if(applyMinimalPromptMode(),!e||!n)return;
const l=compactPromptMode&&!minimalPromptMode,c=!l||promptControlsExpanded;o&&o.classList.toggle("co\
mpact-collapsed",l&&!c),l?c?(e.classList.remove("collapsed"),e.classList.add("expanded"),e.classList.
remove("hidden")):(e.classList.remove("expanded"),e.classList.add("collapsed")):(e.classList.remove(
"hidden"),e.classList.remove("collapsed"),e.classList.remove("expanded")),l?(n.classList.remove("hid\
den"),n.classList.add("inline-flex"),n.setAttribute("aria-expanded",c?"true":"false"),i&&(i.textContent=
c?"\u6298\u308A\u305F\u305F\u3080":"\u8A73\u7D30"),a&&(a.className=c?"fas fa-chevron-up text-[10px]":
"fas fa-chevron-down text-[10px]")):(n.classList.add("hidden"),n.classList.remove("inline-flex"),n.setAttribute(
"aria-expanded","true"),i&&(i.textContent="\u8A73\u7D30"),a&&(a.className="fas fa-chevron-down text-\
[10px]"))}r(applyPromptControlMode,"applyPromptControlMode");function setCompactPromptMode(e,n=!1){compactPromptMode=
!!e,compactPromptMode&&(minimalPromptMode=!1),compactPromptMode?n||(promptControlsExpanded=!1):promptControlsExpanded=
!0,applyPromptControlMode()}r(setCompactPromptMode,"setCompactPromptMode");function setMinimalPromptMode(e){
minimalPromptMode=!!e,minimalPromptMode&&(compactPromptMode=!1,promptControlsExpanded=!1),applyPromptControlMode()}
r(setMinimalPromptMode,"setMinimalPromptMode");function togglePromptControlDetails(){compactPromptMode&&
(promptControlsExpanded=!promptControlsExpanded,applyPromptControlMode())}r(togglePromptControlDetails,
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
containerId:"sys-prompt-option",gear:!0,gearAction:r(()=>{window.openThreadModal&&window.openThreadModal()},
"gearAction")},{key:"thinking",icon:"fa-brain",label:"Thinking",checkboxId:"enable-thinking",containerId:"\
thinking-options",special:"thinking"},{key:"effort",icon:"fa-sliders-h",label:"Effort",containerId:"\
reasoning-effort-container",selectId:"reasoning-effort"},{key:"safety",icon:"fa-shield-halved",label:"\
Safety",selectId:"safety-setting"},{key:"promptcache",icon:"fa-database",label:"PromptCache",checkboxId:"\
enable-prompt-cache",containerId:"prompt-cache-container"},{key:"compress",icon:"fa-compress-alt",label:"\
Compress",checkboxId:"enable-compression",containerId:"compression-option",gear:!0,gearAction:r(()=>{
window.openCompressionModal&&window.openCompressionModal()},"gearAction")},{key:"tempchat",icon:"fa-\
hourglass-half",label:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8",checkboxId:"enable-temporary-chat",containerId:"\
temporary-chat-container",gear:!0,gearAction:r(()=>openTemporaryChatSettings(),"gearAction")}];let minimalOptionsOpen=!1,
thinkingSliderOpen=!1,thinkingSliderTimer=null,thinkingSliderStartY=0,thinkingSliderStartX=0,thinkingSliderDragging=!1,
thinkingSliderAxis=null,popupSwipeStartY=0,popupSwipeStartX=0,popupSwipeDragging=!1,popupSwipeAtTop=!1,
popupSwipeAxis=null;const minimalPanelOrigins=new Map;function minimalOptionVisible(e){if(e.containerId){
const n=get(e.containerId);if(!n||n.classList.contains("hidden"))return!1}return!0}r(minimalOptionVisible,
"minimalOptionVisible");function minimalOptionDisabled(e){if(e.special==="thinking"){const n=get(e.containerId);
return!!(n&&n.classList.contains("pointer-events-none"))}if(e.checkboxId){const n=get(e.checkboxId);
if(n&&n.disabled)return!0}if(e.containerId){const n=get(e.containerId);if(n&&n.classList.contains("p\
ointer-events-none"))return!0}return!1}r(minimalOptionDisabled,"minimalOptionDisabled");function minimalOptionChecked(e){
if(!e.checkboxId)return!1;const n=get(e.checkboxId);return!!n&&n.checked}r(minimalOptionChecked,"min\
imalOptionChecked");function currentThinkingLevelLabel(){const e=get("thinking-level");if(!e)return THINKING_LEVELS[3].
label;const n=THINKING_LEVELS.find(i=>i.value===e.value);return n?n.label:e.selectedOptions[0]?e.selectedOptions[0].
textContent.trim():THINKING_LEVELS[3].label}r(currentThinkingLevelLabel,"currentThinkingLevelLabel");
function buildMinimalOptionItem(e){const n=document.createElement("div");n.className="minimal-option\
-item",n.dataset.key=e.key,e.action&&n.classList.add("action-"+e.action),minimalOptionChecked(e)?n.classList.
add("on"):n.classList.add("off"),minimalOptionDisabled(e)&&n.classList.add("disabled");const i=document.
createElement("i");i.className="fas "+e.icon+" minimal-option-icon",n.appendChild(i);const a=document.
createElement("span");if(a.className="minimal-option-label",a.textContent=e.label,n.appendChild(a),e.
special==="thinking"){const o=document.createElement("span");o.className="thinking-slide-value minim\
al-option-thinking-level",o.textContent=currentThinkingLevelLabel(),n.appendChild(o)}if(e.selectId){
const o=get(e.selectId);if(o){const l=o.cloneNode(!0);l.removeAttribute("id"),l.className="minimal-o\
ption-select",l.addEventListener("change",()=>{o.value=l.value,o.dispatchEvent(new Event("change",{bubbles:!0})),
refreshMinimalOptionItems()}),n.appendChild(l)}}if(e.gear){const o=document.createElement("button");
o.type="button",o.className="minimal-option-gear",o.title=e.label+"\u8A2D\u5B9A";const l=document.createElement(
"i");l.className="fas fa-cog",o.appendChild(l),o.addEventListener("click",c=>{c.stopPropagation(),closeMinimalOptions(),
typeof e.gearAction=="function"&&e.gearAction()}),n.appendChild(o)}return n.addEventListener("click",
()=>handleMinimalOptionClick(e)),n}r(buildMinimalOptionItem,"buildMinimalOptionItem");function renderMinimalOptionItems(){
const e=get("minimal-options-items");if(!e)return;const n=document.createDocumentFragment();MINIMAL_POPUP_ITEMS.
forEach(i=>{minimalOptionVisible(i)&&n.appendChild(buildMinimalOptionItem(i))}),e.innerHTML="",e.appendChild(
n)}r(renderMinimalOptionItems,"renderMinimalOptionItems");function refreshMinimalOptionItems(){const e=get(
"minimal-options-items");if(!e||!minimalOptionsOpen)return;const n=e.querySelectorAll(".minimal-opti\
on-item"),i={};n.forEach(a=>{i[a.dataset.key]=a}),MINIMAL_POPUP_ITEMS.forEach(a=>{const o=i[a.key];if(o){
if(!minimalOptionVisible(a)){o.classList.add("hidden");return}if(o.classList.remove("hidden"),o.classList.
toggle("on",minimalOptionChecked(a)),o.classList.toggle("off",!minimalOptionChecked(a)),o.classList.
toggle("disabled",minimalOptionDisabled(a)),a.special==="thinking"){const l=o.querySelector(".minima\
l-option-thinking-level");l&&(l.textContent=currentThinkingLevelLabel())}if(a.selectId){const l=get(
a.selectId),c=o.querySelector(".minimal-option-select");l&&c&&document.activeElement!==c&&c.value!==
l.value&&(c.value=l.value)}}})}r(refreshMinimalOptionItems,"refreshMinimalOptionItems");function handleMinimalOptionClick(e){
if(e.action==="upload"){closeMinimalOptions(),openUploadModal();return}if(e.action==="button"){const i=get(
e.buttonId);closeMinimalOptions(),i&&i.click();return}if(e.special==="thinking"){const i=get(e.checkboxId);
if(i&&!i.disabled){const a=!i.checked;i.checked=a,i.dispatchEvent(new Event("change",{bubbles:!0})),
a?(closeMinimalOptions(),showThinkingSlider()):hideThinkingSlider(),refreshMinimalOptionItems()}else
closeMinimalOptions(),showThinkingSlider();return}if(minimalOptionDisabled(e)||e.selectId)return;const n=get(
e.checkboxId);n&&(n.disabled||(n.checked=!n.checked,n.dispatchEvent(new Event("change",{bubbles:!0})),
refreshMinimalOptionItems(),e.key==="fast"?(closeMinimalOptions(),setTimeout(()=>refreshMinimalOptionItems(),
350)):e.key==="tempchat"&&setTimeout(()=>refreshMinimalOptionItems(),350)))}r(handleMinimalOptionClick,
"handleMinimalOptionClick");function moveModelPanelsIntoPopup(){const e=get("minimal-options-model-b\
ody");if(!e)return;let n=!1;MINIMAL_MODEL_PANEL_IDS.forEach(i=>{const a=get(i);if(a){if(a.parentElement===
e){a.classList.contains("hidden")||(n=!0);return}minimalPanelOrigins.has(a)||(minimalPanelOrigins.set(
a,{parent:a.parentElement,next:a.nextSibling}),e.appendChild(a),a.classList.contains("hidden")||(n=!0))}}),
refreshMinimalModelSection()}r(moveModelPanelsIntoPopup,"moveModelPanelsIntoPopup");function restoreModelPanelsFromPopup(){
get("minimal-options-model-body")&&(minimalPanelOrigins.forEach((n,i)=>{n.parent&&n.parent.contains(
i)&&(n.next&&n.next.parentNode===n.parent?n.parent.insertBefore(i,n.next):n.parent.appendChild(i))}),
minimalPanelOrigins.clear())}r(restoreModelPanelsFromPopup,"restoreModelPanelsFromPopup");function refreshMinimalModelSection(){
const e=get("minimal-options-model-body"),n=get("minimal-options-model-section");if(!e||!n)return;let i=!1;
Array.from(e.children).forEach(a=>{a.classList.contains("hidden")||(i=!0)}),n.classList.toggle("hidd\
en",!i)}r(refreshMinimalModelSection,"refreshMinimalModelSection");function openMinimalOptions(){if(minimalOptionsOpen||
!minimalPromptMode)return;hideThinkingSlider(),minimalOptionsOpen=!0,renderMinimalOptionItems(),moveModelPanelsIntoPopup();
const e=get("minimal-options-popup");if(!e)return;const n=get("minimal-options-panel");n&&(n.style.cssText=
""),e.classList.remove("minimal-options-closing","minimal-options-open"),e.classList.remove("hidden"),
e.setAttribute("aria-hidden","false"),e.offsetWidth,e.classList.add("minimal-options-open")}r(openMinimalOptions,
"openMinimalOptions");function closeMinimalOptions(){if(!minimalOptionsOpen)return;minimalOptionsOpen=
!1;const e=get("minimal-options-popup");e&&(e.classList.add("minimal-options-closing"),e.setAttribute(
"aria-hidden","true"),setTimeout(()=>{minimalOptionsOpen||(e.classList.remove("minimal-options-open",
"minimal-options-closing"),e.classList.add("hidden"))},560)),restoreModelPanelsFromPopup(),hideThinkingSlider()}
r(closeMinimalOptions,"closeMinimalOptions");function toggleMinimalOptions(){minimalOptionsOpen?closeMinimalOptions():
openMinimalOptions()}r(toggleMinimalOptions,"toggleMinimalOptions");function refreshMinimalOptionsIfOpen(){
minimalOptionsOpen&&(renderMinimalOptionItems(),refreshMinimalModelSection())}r(refreshMinimalOptionsIfOpen,
"refreshMinimalOptionsIfOpen");function allowedThinkingValues(){const e=get("thinking-level");return e?
Array.from(e.options).filter(i=>!i.disabled&&!i.classList.contains("hidden")).map(i=>i.value):THINKING_LEVELS.
map(i=>i.value)}r(allowedThinkingValues,"allowedThinkingValues");function thinkingIndexFromValue(e){
const n=THINKING_LEVELS.findIndex(i=>i.value===e);return n<0?3:n}r(thinkingIndexFromValue,"thinkingI\
ndexFromValue");function syncThinkingSliderUi(){const e=get("thinking-slider"),n=get("thinking-slide\
-value"),i=get("thinking-level"),a=thinkingIndexFromValue(i?i.value:"high");e&&(e.value=String(a)),n&&
(n.textContent=THINKING_LEVELS[a].label)}r(syncThinkingSliderUi,"syncThinkingSliderUi");function scheduleThinkingSliderHide(){
thinkingSliderTimer&&clearTimeout(thinkingSliderTimer),thinkingSliderTimer=setTimeout(()=>{thinkingSliderTimer=
null,hideThinkingSlider()},2500)}r(scheduleThinkingSliderHide,"scheduleThinkingSliderHide");function showThinkingSlider(){
if(thinkingSliderOpen){scheduleThinkingSliderHide();return}const e=get("thinking-slide-bar");if(!e)return;
const n=get("thinking-slide-inner");n&&(n.style.transform=""),thinkingSliderOpen=!0,e.classList.remove(
"hidden"),e.setAttribute("aria-hidden","false"),syncThinkingSliderUi(),e.offsetWidth,e.classList.add(
"thinking-slide-open"),scheduleThinkingSliderHide()}r(showThinkingSlider,"showThinkingSlider");function hideThinkingSlider(){
thinkingSliderTimer&&(clearTimeout(thinkingSliderTimer),thinkingSliderTimer=null);const e=get("think\
ing-slide-bar");e&&(thinkingSliderOpen=!1,e.classList.remove("thinking-slide-open"),e.setAttribute("\
aria-hidden","true"),setTimeout(()=>{thinkingSliderOpen||e.classList.add("hidden");const n=get("thin\
king-slide-inner");n&&(n.style.transform="")},360))}r(hideThinkingSlider,"hideThinkingSlider");function bindMinimalOptionsEvents(){
const e=get("minimal-options-backdrop"),n=get("minimal-options-close-btn"),i=get("minimal-options-po\
pup");i&&i.parentNode!==document.body&&document.body.appendChild(i),e&&e.addEventListener("click",()=>closeMinimalOptions()),
n&&n.addEventListener("click",()=>closeMinimalOptions()),document.addEventListener("keydown",d=>{if(d.
key==="Escape"){if(minimalOptionsOpen){closeMinimalOptions();return}thinkingSliderOpen&&hideThinkingSlider()}});
const a=get("thinking-slider");a&&a.addEventListener("input",()=>{const d=Number(a.value),m=allowedThinkingValues(),
h=get("thinking-level");if(m.length){const y=m.map(x=>thinkingIndexFromValue(x)),v=y.includes(d)?d:y.
reduce((x,w)=>Math.abs(w-d)<Math.abs(x-d)?w:x,y[0]);h&&(h.value=THINKING_LEVELS[v].value,h.dispatchEvent(
new Event("change",{bubbles:!0})))}syncThinkingSliderUi(),scheduleThinkingSliderHide()});const o=get(
"thinking-slide-close-btn");o&&o.addEventListener("click",d=>{d.stopPropagation(),hideThinkingSlider()});
const l=get("thinking-slide-bar");if(l){const d=get("thinking-slide-inner");l.addEventListener("touc\
hstart",m=>{thinkingSliderOpen&&(thinkingSliderDragging=!0,thinkingSliderStartY=m.touches[0].clientY,
thinkingSliderStartX=m.touches[0].clientX,thinkingSliderAxis=null,d&&d.classList.add("dragging"))},{
passive:!0}),l.addEventListener("touchmove",m=>{if(!thinkingSliderDragging)return;const h=m.touches[0].
clientX-thinkingSliderStartX,y=m.touches[0].clientY-thinkingSliderStartY;if(thinkingSliderAxis===null&&
(Math.abs(h)>8||Math.abs(y)>8)&&(thinkingSliderAxis=Math.abs(y)>Math.abs(h)?"v":"h"),thinkingSliderAxis===
"v")if(y>0){m.cancelable&&m.preventDefault();const v=Math.min((y-8)*.5,120);d&&(d.style.transform=v>
0?`translateY(${v}px)`:"")}else d&&(d.style.transform="")},{passive:!1}),l.addEventListener("touchen\
d",m=>{if(!thinkingSliderDragging)return;thinkingSliderDragging=!1;const h=m.changedTouches[0].clientY-
thinkingSliderStartY;d&&d.classList.remove("dragging"),thinkingSliderAxis==="v"&&h>100?(d&&(d.style.
transform=`translateY(${Math.max(h*.5,60)}px)`),hideThinkingSlider()):(d&&(d.style.transform=""),scheduleThinkingSliderHide())},
{passive:!0}),l.addEventListener("touchcancel",()=>{thinkingSliderDragging=!1,d&&(d.classList.remove(
"dragging"),d.style.transform=""),scheduleThinkingSliderHide()},{passive:!0})}const c=get("minimal-o\
ptions-panel");c&&(c.addEventListener("touchstart",d=>{if(!minimalOptionsOpen)return;popupSwipeDragging=
!0,popupSwipeStartY=d.touches[0].clientY,popupSwipeStartX=d.touches[0].clientX,popupSwipeAxis=null;let m=d.
target instanceof Element?d.target:null,h=!0;for(;m&&m!==c;){if(m.scrollTop>0){h=!1;break}m=m.parentElement}
popupSwipeAtTop=h,h&&c.classList.add("dragging")},{passive:!0}),c.addEventListener("touchmove",d=>{if(!popupSwipeDragging||
!popupSwipeAtTop||!minimalOptionsOpen)return;const m=d.touches[0].clientX-popupSwipeStartX,h=d.touches[0].
clientY-popupSwipeStartY;popupSwipeAxis===null&&(Math.abs(m)>8||Math.abs(h)>8)&&(popupSwipeAxis=Math.
abs(h)>Math.abs(m)?"v":"h"),popupSwipeAxis==="v"&&h>0&&(d.cancelable&&d.preventDefault(),c.style.transform=
`translateY(${Math.min(h*.6,140)}px)`)},{passive:!1}),c.addEventListener("touchend",d=>{if(!popupSwipeDragging)
return;popupSwipeDragging=!1;const m=d.changedTouches[0].clientY-popupSwipeStartY;c.classList.remove(
"dragging"),popupSwipeAtTop&&popupSwipeAxis!=="h"&&m>70?(c.style.transform=`translateY(${Math.max(m*
.6,100)}px)`,c.style.opacity="0",closeMinimalOptions()):c.style.transform=""},{passive:!0}),c.addEventListener(
"touchcancel",()=>{popupSwipeDragging=!1,c.classList.remove("dragging"),c.style.transform="",c.style.
opacity=""},{passive:!0}))}r(bindMinimalOptionsEvents,"bindMinimalOptionsEvents");function bindUploadButton(){
const e=get("upload-btn");e&&(e.onclick=()=>{minimalPromptMode?toggleMinimalOptions():openUploadModal()})}
r(bindUploadButton,"bindUploadButton");function applyChatDefaults(e){if(!e||(Object.prototype.hasOwnProperty.
call(e,"voice_studio_ui")&&(voiceStudioUiEnabled=e.voice_studio_ui!==!1),applyTemporaryChatTimeoutSeconds(
e.temp_chat_timeout_seconds),chatDefaultsLoaded))return;const i=!!e.use_last_chat_settings?{model:e.
last_model,enable_search:e.last_enable_search,enable_url_context:e.last_enable_url_context,enable_maps:e.
last_enable_maps,enable_python:e.last_enable_python,enable_file_creation:e.last_enable_file_creation,
enable_thinking:e.last_enable_thinking,thinking_level:e.last_thinking_level,thinking_budget:e.last_thinking_budget,
reasoning_effort:e.last_reasoning_effort,enable_system_prompt:e.last_enable_system_prompt,enable_mcp:e.
last_enable_mcp,safety_setting:e.last_safety_setting}:{model:e.default_model,enable_search:e.default_enable_search,
enable_url_context:e.default_enable_url_context,enable_maps:e.default_enable_maps,enable_python:e.default_enable_python,
enable_file_creation:e.default_enable_file_creation,enable_thinking:e.default_enable_thinking,thinking_level:e.
default_thinking_level,thinking_budget:e.default_thinking_budget,reasoning_effort:e.default_reasoning_effort,
enable_system_prompt:e.default_enable_system_prompt,enable_mcp:e.default_enable_mcp,safety_setting:e.
default_safety_setting},a=r((o,l)=>o==null||o===""?l:o,"s");i.model&&selectModelById(i.model),get("e\
nable-search")&&(get("enable-search").checked=!!a(i.enable_search,get("enable-search").checked)),get(
"enable-url-context")&&(get("enable-url-context").checked=!!a(i.enable_url_context,get("enable-url-c\
ontext").checked)),get("enable-maps")&&(get("enable-maps").checked=!!a(i.enable_maps,get("enable-map\
s").checked)),get("enable-python")&&(get("enable-python").checked=!!a(i.enable_python,get("enable-py\
thon").checked)),get("enable-file-creation")&&(get("enable-file-creation").checked=!!a(i.enable_file_creation,
get("enable-file-creation").checked)),get("enable-thinking")&&(get("enable-thinking").checked=!!a(i.
enable_thinking,get("enable-thinking").checked)),get("thinking-level")&&(get("thinking-level").value=
a(i.thinking_level,get("thinking-level").value||"high")),get("thinking-budget")&&(get("thinking-budg\
et").value=a(i.thinking_budget,get("thinking-budget").value||4096)),get("reasoning-effort")&&(get("r\
easoning-effort").value=a(i.reasoning_effort,get("reasoning-effort").value||"medium")),get("enable-s\
ys-prompt")&&(get("enable-sys-prompt").checked=!!a(i.enable_system_prompt,get("enable-sys-prompt").checked)),
get("enable-mcp")&&(get("enable-mcp").checked=!!a(i.enable_mcp,get("enable-mcp").checked)),get("safe\
ty-setting")&&(get("safety-setting").value=a(i.safety_setting,get("safety-setting").value||"default")),
chatDefaultsLoaded=!0,toggleOptions(),applyMcpPromptChipUi()}r(applyChatDefaults,"applyChatDefaults");
function setEditUi(e){const n=get("edit-bar");n&&(e?(n.classList.remove("hidden"),n.classList.add("f\
lex")):(n.classList.add("hidden"),n.classList.remove("flex")),updatePromptPlaceholder())}r(setEditUi,
"setEditUi");function cancelEdit(){editingMessageId=null,currentParentId=currentLeafId||null;const e=get(
"prompt-input");e&&(e.value="",e.style.height="auto"),currentImageUrls=[],get("file-preview").classList.
add("hidden"),get("file-input").value="",clearQuote(),setEditUi(!1)}r(cancelEdit,"cancelEdit");function beginEditMessage(e,n=!1){
const i=messageStore[e];if(i==null)return;const a=get("prompt-input");a.value=i||"",a.focus(),a.style.
height="auto",a.style.height=a.scrollHeight+"px";const o=allMessages.find(m=>m.id==e),l=messageMeta[e]||
{};o?currentParentId=o.parent_id===void 0?null:o.parent_id:l.parent_id!==void 0&&(currentParentId=l.
parent_id),editingMessageId=e,setEditUi(!0);const c=o?o.image_url:l.image_url;if(c)try{const m=JSON.
parse(c);Array.isArray(m)&&m.length?(currentImageUrls=m.map(h=>{let y="unknown",v=h;h&&typeof h=="ob\
ject"&&(y=normalizeAttachmentSource(h.source),v=h.filepath||h.path||h.url||h.file||"");const x=normalizeAttachmentPath(
v);return x&&setAttachmentSourceForPath(x,y),x}).filter(Boolean),get("file-preview").classList.remove(
"hidden"),get("file-name").innerText=`${currentImageUrls.length} files ready`):(currentImageUrls=[],
get("file-preview").classList.add("hidden"),get("file-input").value="")}catch{currentImageUrls=[],get(
"file-preview").classList.add("hidden"),get("file-input").value=""}else currentImageUrls=[],get("fil\
e-preview").classList.add("hidden"),get("file-input").value="";const d=o?o.quote_text:l.quote_text;d?
(currentQuote=d,get("quote-text-display").innerText=currentQuote,get("quote-bar").classList.add("vis\
ible")):clearQuote(),schedulePromptTokenEstimate(!0),n&&sendMessage()}r(beginEditMessage,"beginEditM\
essage");function playSendAnimation(){const e=get("send-btn");e&&(e.classList.remove("fly"),e.offsetWidth,
e.classList.add("fly"))}r(playSendAnimation,"playSendAnimation");function setSendBtnToStopMode(){const e=get(
"send-btn");if(!e)return;e.onclick=stopGeneration,isStopMode=!0,e.disabled=!1;const n=r(()=>{!e||!isStopMode||
(e.classList.add("stop-mode"),e.innerHTML='<span style="font-size:20px;line-height:1;color:#fff;">\u25A0<\
/span>',e.classList.add("btn-swap"),setTimeout(()=>e.classList.remove("btn-swap"),300))},"applyStopU\
i");if(e.classList.contains("fly")){const i=r(a=>{a.animationName==="sendBtnPop"&&(e.removeEventListener(
"animationend",i),n())},"onEnd");e.addEventListener("animationend",i),setTimeout(n,700)}else n()}r(setSendBtnToStopMode,
"setSendBtnToStopMode");function setSendBtnToSendMode(){const e=get("send-btn");e&&(e.classList.remove(
"stop-mode","fly","btn-swap"),e.innerHTML='<i class="fas fa-paper-plane"></i>',e.classList.add("btn-\
swap"),setTimeout(()=>e.classList.remove("btn-swap"),300),e.onclick=sendMessage,isStopMode=!1)}r(setSendBtnToSendMode,
"setSendBtnToSendMode");async function stopGeneration(){const e=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null,n=normalizeJobIdForUi(currentJobId),i=++manualStopSeq,a=captureStoppedPartialBubbleSnapshot(
getActiveStreamingBubbleElement());manualStopContext={seq:i,threadId:e,jobId:n,partialSnapshot:a},n&&
suppressPendingJob(n),abortController&&abortController.abort();try{if(n||e){const o={};n&&(o.job_id=
n),e&&(o.thread_id=e);const c=await(await apiFetch("/api/stop_chat",{method:"POST",headers:{"Content\
-Type":"application/json"},body:JSON.stringify(o)})).json().catch(()=>({})),d=normalizeJobIdForUi(c&&
c.job_id);d&&(suppressPendingJob(d),manualStopContext&&manualStopContext.seq===i&&(manualStopContext.
jobId=d))}manualStopContext&&manualStopContext.seq===i&&await syncThreadAfterAbortedStream(e,{retries:2,
retryDelayMs:180,notifyOnFailure:!0})&&manualStopContext.partialSnapshot&&appendStoppedPartialBubbleSnapshot(
manualStopContext.partialSnapshot,e)}finally{manualStopContext&&manualStopContext.seq===i&&(manualStopContext=
null),setSendBtnToSendMode(),updateFilePreview()}}r(stopGeneration,"stopGeneration");async function purgeCaches(){
if("caches"in window){const e=await caches.keys();await Promise.all(e.map(n=>caches.delete(n)))}if(navigator.
serviceWorker){const e=await navigator.serviceWorker.getRegistrations();await Promise.all(e.map(n=>n.
unregister()))}}r(purgeCaches,"purgeCaches");const SW_CACHE_MODE_STORAGE_KEY="ai_sw_cache_mode_v2";async function applyCacheMode(e,n={}){
if("serviceWorker"in navigator)if(e)try{await navigator.serviceWorker.register(`/sw.js?v=${encodeURIComponent(
appVersion)}`),localStorage.setItem(SW_CACHE_MODE_STORAGE_KEY,"enabled")}catch{}else{const i=localStorage.
getItem(SW_CACHE_MODE_STORAGE_KEY);(!!n.forceCleanup||i!=="disabled")&&await purgeCaches(),localStorage.
setItem(SW_CACHE_MODE_STORAGE_KEY,"disabled")}}r(applyCacheMode,"applyCacheMode");function checkAndNotifyVersion(e){
!e||!appVersion||e===appVersion||(localStorage.getItem("version_notified")||"")===e||(localStorage.setItem(
"app_version",e),syncVersionUpdateCachePreferenceUi(),showModal("version-update-modal"))}r(checkAndNotifyVersion,
"checkAndNotifyVersion");async function checkVersion(){try{const e=await fetch("/api/version",{cache:"\
no-store"});if(!e.ok)return;const i=(await e.json()).version||"",a=localStorage.getItem("app_version")||
"";i&&!a&&localStorage.setItem("app_version",i),i&&a&&i!==a&&(await purgeCaches(),checkAndNotifyVersion(
i))}catch{}}r(checkVersion,"checkVersion");async function fetchChatStreamWithUnavailableRetry(e,n,i){
let a=0;for(;;){if(n.signal&&n.signal.aborted)throw new DOMException("Aborted","AbortError");try{const o=await apiFetch(
e,n),l=window.ConnectionMonitor.retryModeForResponse(o);let c=!1;if(o.status===425&&(c=(await o.clone().
json().catch(()=>({}))).code==="submission_in_progress"),!l&&!c)return window.ConnectionMonitor.markReachable(),
o;a+=1,l&&window.ConnectionMonitor.setUnavailable(l),updatePendingSkeletonStatus(i,l==="maintenance"?
"\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u7D42\u4E86\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...":"\u30B5\u30FC\u30D0\
\u30FC\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${a}\
\u56DE\u76EE\uFF09`)}catch(o){if(n.signal&&n.signal.aborted||o.name==="AbortError")throw o;a+=1,window.
ConnectionMonitor.setUnavailable("offline"),updatePendingSkeletonStatus(i,"\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u63A5\u7D9A\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${a}\
\u56DE\u76EE\uFF09`)}await window.ConnectionMonitor.waitForRetry(n.signal)}}r(fetchChatStreamWithUnavailableRetry,
"fetchChatStreamWithUnavailableRetry");function createClientRequestId(){return window.crypto&&typeof window.
crypto.randomUUID=="function"?window.crypto.randomUUID():`req-${window.crypto&&typeof window.crypto.
getRandomValues=="function"?Array.from(window.crypto.getRandomValues(new Uint32Array(4))).map(n=>n.toString(
16)).join(""):`${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`}`.slice(0,64)}r(createClientRequestId,
"createClientRequestId");async function reconnectPendingStreamUntilAvailable(e,n){const i=n!=null?String(
n):"",a=normalizeJobIdForUi(e&&e.job_id),o=a||`thread:${i}`;if(!i||pendingStreamReconnectJobs.has(o))
return;pendingStreamReconnectJobs.add(o);const l=new AbortController;let c=!1;abortController=l,currentJobId=
a,setSendBtnToStopMode();try{for(;!l.signal.aborted;){if(String(currentThreadId||"")!==i||a&&isPendingJobSuppressed(
a))return;const d=getActiveStreamingBubbleElement();if(updatePendingSkeletonStatus(d,"\u30B5\u30FC\u30D0\u30FC\u3078\u306E\u518D\u63A5\u7D9A\u3092\u5F85\u3063\u3066\u3044\
\u307E\u3059...","\u56DE\u7B54\u51E6\u7406\u306F\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u7D99\u7D9A\u3057\u3066\u3044\u307E\u3059"),
await window.ConnectionMonitor.waitForRetry(l.signal),!await loadMessages(i,{preserveDraft:!0,silent:!0,
skipHistory:!0})){window.ConnectionMonitor.probeNow();continue}const h=currentThreadPending;h&&h.job_id&&
!isPendingJobSuppressed(h.job_id)?(abortController===l&&(abortController=null),c=!0,resumePendingStream(
h)):window.ConnectionMonitor.markReachable();return}}catch(d){d.name!=="AbortError"&&sendClientDebugLog(
"error",`Stream reconnect failed: ${d.message}`)}finally{pendingStreamReconnectJobs.delete(o),abortController===
l&&(abortController=null),c||(currentJobId=null,setSendBtnToSendMode(),updateFilePreview())}}r(reconnectPendingStreamUntilAvailable,
"reconnectPendingStreamUntilAvailable"),window.initTurnstileWidget=()=>{if(!botConfig||!botConfig.turnstileSiteKey||
!window.turnstile||turnstileWidgetId!==null)return;const e=document.getElementById("turnstile-contai\
ner");e&&(e.classList.remove("hidden"),turnstileWidgetId=window.turnstile.render(e,{sitekey:botConfig.
turnstileSiteKey,size:"compact",appearance:"interaction-only",callback:r(n=>{turnstileToken=n,turnstilePending=
!1,verifyTurnstileOnServer(n)},"callback"),"expired-callback":r(()=>{turnstileToken=null,turnstilePending=
!1},"expired-callback"),"error-callback":r(()=>{turnstileToken=null,turnstilePending=!1},"error-call\
back")}),isBotDetectionActive()&&runBotDetectionGate())};async function getTurnstileToken(e=1500){if(!botConfig||
!botConfig.turnstileSiteKey)return null;if(turnstileToken)return turnstileToken;if(!window.turnstile)
return null;if(botDetectionOverlayShown&&botDetectionDialogWidgetId!==null)return turnstilePending=!0,
await new Promise(i=>{const a=turnstileToken,o=setTimeout(()=>i(null),Math.max(500,Number(e)||1500)),
l=setInterval(()=>{turnstileToken&&turnstileToken!==a&&(clearTimeout(o),clearInterval(l),i(turnstileToken))},
50)});if(turnstileWidgetId===null)return null;const n=document.getElementById("turnstile-container");
return n&&n.classList.remove("hidden"),turnstilePending=!0,await new Promise(i=>{const a=turnstileToken,
o=setTimeout(()=>i(null),Math.max(500,Number(e)||1500));try{window.turnstile.execute(turnstileWidgetId)}catch{
clearTimeout(o),i(null);return}const l=setInterval(()=>{turnstileToken&&turnstileToken!==a&&(clearTimeout(
o),clearInterval(l),verifyTurnstileOnServer(turnstileToken),i(turnstileToken))},50)})}r(getTurnstileToken,
"getTurnstileToken");function resetTurnstileToken(){if(turnstileToken=null,turnstilePending=!1,window.
turnstile&&turnstileWidgetId!==null)try{window.turnstile.reset(turnstileWidgetId)}catch{}if(window.turnstile&&
botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}}r(resetTurnstileToken,
"resetTurnstileToken");function isBotDetectionActive(){return!!(botConfig&&botConfig.globalEnabled&&
botConfig.accountEnabled&&!isAdminUser&&botConfig.turnstileSiteKey)}r(isBotDetectionActive,"isBotDet\
ectionActive");function renderBotDetectionDialogWidget(){if(botDetectionDialogWidgetId!==null||!botConfig||
!botConfig.turnstileSiteKey)return;const e=document.getElementById("bot-detection-widget-box");if(e){
if(!window.turnstile){setTimeout(renderBotDetectionDialogWidget,250);return}try{botDetectionDialogWidgetId=
window.turnstile.render(e,{sitekey:botConfig.turnstileSiteKey,theme:"dark",size:"flexible",callback:r(
n=>{turnstileToken=n,turnstilePending=!1,verifyTurnstileOnServer(n,!0,!0)},"callback"),"expired-call\
back":r(()=>{if(turnstileToken=null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.
turnstile.reset(botDetectionDialogWidgetId)}catch{}},"expired-callback"),"error-callback":r(()=>{if(turnstileToken=
null,turnstilePending=!1,botDetectionDialogWidgetId!==null)try{window.turnstile.reset(botDetectionDialogWidgetId)}catch{}},
"error-callback")})}catch(n){console.error("bot-detection dialog widget error",n)}}}r(renderBotDetectionDialogWidget,
"renderBotDetectionDialogWidget");function showBotDetectionOverlay(e=""){let n=document.getElementById(
"bot-detection-overlay");if(n)n.style.display="flex";else{n=document.createElement("div"),n.id="bot-\
detection-overlay",n.style.cssText="position:fixed;inset:0;z-index:2147483000;background:rgba(3,7,18\
,0.92);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;";const a=document.
createElement("div");a.style.cssText="max-width:420px;width:100%;background:#0f172a;border:1px solid\
 #334155;border-radius:12px;padding:24px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,.5);dis\
play:flex;flex-direction:column;align-items:stretch;gap:12px;";const o=document.createElement("div");
o.id="bot-detection-overlay-title",o.style.cssText="font-weight:700;font-size:15px;color:#f1f5f9;",o.
textContent=e||"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u4E2D...";const l=document.createElement("div");
l.style.cssText="font-size:12px;color:#94a3b8;line-height:1.6;",l.textContent="\u81EA\u52D5\u30A2\u30AF\u30BB\u30B9\u9632\u6B62\u306E\u305F\u3081\u3001\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u3066\u304F\u3060\
\u3055\u3044\u3002";const c=document.createElement("div");c.id="bot-detection-widget-box",c.style.cssText=
"margin-top:8px;min-height:65px;display:flex;justify-content:center;",a.appendChild(o),a.appendChild(
l),a.appendChild(c),n.appendChild(a),document.body.appendChild(n)}const i=document.getElementById("b\
ot-detection-overlay-title");e&&i&&(i.textContent=e),botDetectionOverlayShown=!0,renderBotDetectionDialogWidget()}
r(showBotDetectionOverlay,"showBotDetectionOverlay");function hideBotDetectionOverlay(){if(botDetectionOverlayShown=
!1,botDetectionDialogWidgetId!==null){try{window.turnstile.remove(botDetectionDialogWidgetId)}catch{}
botDetectionDialogWidgetId=null}const e=document.getElementById("bot-detection-widget-box");e&&e.replaceChildren();
const n=document.getElementById("bot-detection-overlay");n&&n.remove()}r(hideBotDetectionOverlay,"hi\
deBotDetectionOverlay");let botLockOverlay=null,botLockTimer=null;function showBotLockOverlay(e="\u9001\u4FE1\u64CD\
\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",n=600){
hideBotDetectionOverlay();let i=document.getElementById("bot-lock-overlay");if(i){i.style.display="f\
lex";const a=document.getElementById("bot-lock-overlay-message");a&&e&&(a.textContent=e)}else{i=document.
createElement("div"),i.id="bot-lock-overlay",i.style.cssText="position:fixed;inset:0;z-index:2147483\
000;background:rgba(3,7,18,0.94);display:flex;flex-direction:column;align-items:center;justify-conte\
nt:center;padding:24px;";const a=document.createElement("div");a.style.cssText="max-width:440px;widt\
h:100%;background:#0f172a;border:1px solid #f59e0b;border-radius:12px;padding:24px;text-align:center\
;box-shadow:0 10px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;align-items:center;gap:12p\
x;";const o=document.createElement("div");o.style.cssText="font-size:26px;color:#fbbf24;",o.innerHTML=
'<i class="fas fa-lock"></i>';const l=document.createElement("div");l.id="bot-lock-overlay-title",l.
style.cssText="font-weight:700;font-size:16px;color:#fbbf24;",l.textContent="\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F";
const c=document.createElement("div");c.id="bot-lock-overlay-message",c.style.cssText="font-size:13p\
x;color:#f1f5f9;line-height:1.7;",c.textContent=e;const d=document.createElement("div");d.id="bot-lo\
ck-overlay-timer",d.style.cssText="font-size:12px;color:#94a3b8;margin-top:2px;";const m=document.createElement(
"div");m.style.cssText="font-size:11px;color:#94a3b8;line-height:1.6;",m.textContent="\u30ED\u30C3\u30AF\u89E3\u9664\u307E\u3067\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\
\u304F\u3060\u3055\u3044\u3002\u540C\u3058\u64CD\u4F5C\u3092\u7E70\u308A\u8FD4\u3059\u3068BAN\u3055\u308C\u308B\u5834\u5408\u304C\u3042\u308A\u307E\u3059\u3002",
a.appendChild(o),a.appendChild(l),a.appendChild(c),a.appendChild(d),a.appendChild(m),i.appendChild(a),
document.body.appendChild(i)}return botLockOverlay=i,updateBotLockTimer(n),i}r(showBotLockOverlay,"s\
howBotLockOverlay");function updateBotLockTimer(e){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const n=document.getElementById("bot-lock-overlay-timer");if(!n)return;const i=r(()=>{const a=Math.
max(0,Math.round(Number(e)||0)),o=Math.floor(a/60),l=String(a%60).padStart(2,"0");n.textContent=`\u30ED\u30C3\u30AF\
\u89E3\u9664\u307E\u3067: ${o}:${l}`},"render");i(),botLockTimer=setInterval(()=>{e-=1,i(),e<=0&&(botLockTimer&&
(clearInterval(botLockTimer),botLockTimer=null),location.reload())},1e3)}r(updateBotLockTimer,"updat\
eBotLockTimer");function hideBotLockOverlay(){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const e=document.getElementById("bot-lock-overlay");e&&e.remove(),botLockOverlay=null}r(hideBotLockOverlay,
"hideBotLockOverlay");async function applyBotLockFromServer(e){if(isAdminUser)return!0;let n=600;try{
const i=await apiFetch("/api/bot/lock",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({reason:e||""})});if(i.status===403){let o=null;try{o=await i.json()}catch{}if(o&&o.error===
"banned")return showToast("\u30ED\u30C3\u30AF\u304C\u7E70\u308A\u8FD4\u3055\u308C\u305F\u305F\u3081BAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800),!1}const a=await i.json().catch(()=>({}));
if(a&&(a.status==="skipped"||a.skipped))return!0;a&&typeof a.remaining_seconds=="number"&&(n=a.remaining_seconds)}catch{}
return showBotLockOverlay(e||"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",
n),!1}r(applyBotLockFromServer,"applyBotLockFromServer");const runBotDetectionGate=r(()=>botDetectionVerified||
!isBotDetectionActive()?Promise.resolve(!0):botDetectionGatePromise||(botDetectionGatePromise=(async()=>{
let e=0;for(;!botDetectionVerified;){if(!botDetectionOverlayShown){if(!window.__turnstileApiLoaded||
turnstileWidgetId===null){await new Promise(o=>setTimeout(o,1e3));continue}const i=await getTurnstileToken(
8e3);if(i&&await verifyTurnstileOnServer(i,!0,!1))break;e+=1;let a=!1;try{a=!!(botTelemetry&&botTelemetry.
looksSuspicious&&botTelemetry.looksSuspicious())}catch{}(e>=2||a)&&showBotDetectionOverlay();continue}
const n=await getTurnstileToken(25e3);if(n&&await verifyTurnstileOnServer(n,!0,!0))break;try{botTelemetry.
send(!0,{forceReport:!0})}catch{}await new Promise(i=>setTimeout(i,5e3))}return hideBotDetectionOverlay(),
!0})().finally(()=>{botDetectionGatePromise=null}),botDetectionGatePromise),"runBotDetectionGate");function registerSendButtonSpam(){
const e=performance.now();return sendButtonSpamTimestamps.push(e),sendButtonSpamTimestamps=sendButtonSpamTimestamps.
filter(n=>e-n<=3e3),sendButtonSpamTimestamps.length}r(registerSendButtonSpam,"registerSendButtonSpam");
function resetSendButtonSpam(){sendButtonSpamTimestamps=[]}r(resetSendButtonSpam,"resetSendButtonSpa\
m");async function runSendSpamVerification(){return isBotDetectionActive()?await applyBotLockFromServer(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002"):
!0}r(runSendSpamVerification,"runSendSpamVerification");let turnstileServerVerifiedAt=0,turnstileVerifyInFlight=null,
turnstileVerifyInFlightToken=null,turnstileLastSubmittedToken=null;async function verifyTurnstileOnServer(e,n=!1,i=null){
if(!e||!isBotDetectionActive()||botDetectionVerified)return!0;i===null&&(i=botDetectionOverlayShown);
const a=Date.now();if(!n&&a-turnstileServerVerifiedAt<60*1e3)return!0;if(turnstileVerifyInFlight&&turnstileVerifyInFlightToken===
e)return turnstileVerifyInFlight;if(turnstileLastSubmittedToken===e&&!n)return!!botDetectionVerified;
if(turnstileLastSubmittedToken===e)return turnstileVerifyInFlight&&turnstileVerifyInFlightToken===e?
turnstileVerifyInFlight:!!botDetectionVerified;turnstileLastSubmittedToken=e,turnstileVerifyInFlightToken=
e;const o=!!i;return turnstileVerifyInFlight=(async()=>{try{return(await apiFetch("/api/bot/turnstil\
e-verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({turnstile_token:e,
challenged:o})})).ok?(turnstileServerVerifiedAt=Date.now(),botDetectionVerified=!0,hideBotDetectionOverlay(),
!0):!1}catch{return!1}finally{turnstileVerifyInFlightToken===e&&(turnstileVerifyInFlight=null,turnstileVerifyInFlightToken=
null)}})(),turnstileVerifyInFlight}r(verifyTurnstileOnServer,"verifyTurnstileOnServer");function botTurnstileTokenForRequest(){
return isBotDetectionActive()?turnstileToken:null}r(botTurnstileTokenForRequest,"botTurnstileTokenFo\
rRequest");const botTelemetry=(()=>{const e={enabled:!1,windowStart:performance.now(),lastSend:0,clicks:0,
keys:0,moves:0,fastClicks:0,fastKeys:0,untrustedInput:!1,clickTimes:[],keyTimes:[],clickIntervals:[],
lastClickTs:0,lastKeyTs:0,lastMove:null,speedMax:0,speedSum:0,speedSamples:0,lastMoveSample:0},n=r(()=>{
e.enabled=!!(botConfig&&botConfig.globalEnabled&&botConfig.accountEnabled&&!isAdminUser)},"refreshEn\
abled"),i=r(()=>{e.windowStart=performance.now(),e.clicks=0,e.keys=0,e.moves=0,e.fastClicks=0,e.fastKeys=
0,e.untrustedInput=!1,e.clickTimes=[],e.keyTimes=[],e.clickIntervals=[],e.speedMax=0,e.speedSum=0,e.
speedSamples=0},"resetWindow"),a=r(x=>{const w=x&&x.target;return!w||typeof w.closest!="function"?!1:
!!w.closest("[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay")},
"isControlClick"),o=r(x=>{if(a(x))return;if(x&&x.isTrusted===!1){e.untrustedInput=!0,h(!0);return}const w=performance.
now();if(e.clicks+=1,e.lastClickTs){const _=w-e.lastClickTs;e.clickIntervals.push(_),e.clickIntervals.
length>10&&e.clickIntervals.shift(),_<120&&(e.fastClicks+=1)}e.lastClickTs=w,e.clickTimes.push(w),e.
clickTimes=e.clickTimes.filter(_=>w-_<=2e3),e.fastClicks>=4&&h(!0)},"recordClick"),l=r(x=>{if(x&&x.isTrusted===
!1){e.untrustedInput=!0,h(!0);return}const w=performance.now();e.keys+=1,e.lastKeyTs&&w-e.lastKeyTs<
50&&(e.fastKeys+=1),e.lastKeyTs=w,e.keyTimes.push(w),e.keyTimes=e.keyTimes.filter(_=>w-_<=2e3)},"rec\
ordKey"),c=r(x=>{const w=performance.now();if(!(w-e.lastMoveSample<80)){if(e.lastMoveSample=w,e.moves+=
1,e.lastMove){const _=x.clientX-e.lastMove.x,S=x.clientY-e.lastMove.y,L=w-e.lastMove.t;if(L>0){const M=Math.
sqrt(_*_+S*S)/(L/1e3);e.speedMax=Math.max(e.speedMax,M),e.speedSum+=M,e.speedSamples+=1}}e.lastMove=
{x:x.clientX,y:x.clientY,t:w}}},"recordMove"),d=r(()=>{const x=Math.max(1,performance.now()-e.windowStart),
w=e.clickTimes.length,_=e.keyTimes.length,S=e.speedSamples?e.speedSum/e.speedSamples:0;let L=0,M=1;if(e.
clickIntervals.length>=3){const P=e.clickIntervals.reduce((Q,ee)=>Q+ee,0)/e.clickIntervals.length,H=e.
clickIntervals.reduce((Q,ee)=>Q+Math.pow(ee-P,2),0)/e.clickIntervals.length;L=P,M=P>0?Math.sqrt(H)/P:
1}return{window_ms:Math.round(x),clicks:e.clicks,keys:e.keys,moves:e.moves,fast_clicks:e.fastClicks,
fast_keys:e.fastKeys,untrusted_input:!!e.untrustedInput,click_burst:w,key_burst:_,avg_click_ms:L,click_cv:M,
event_rate:(e.clicks+e.keys+e.moves)/(x/1e3),pointer_speed_max:e.speedMax,pointer_speed_avg:S}},"com\
puteStats"),m=r(x=>x.fast_clicks>=4||x.fast_keys>=8||x.click_burst>=8||x.key_burst>=14||x.event_rate>=
20||x.avg_click_ms>0&&x.avg_click_ms<160&&x.click_cv<.08,"isSuspicious"),h=r(async(x=!1,w={})=>{if(!e.
enabled)return;const _=performance.now();if(!x&&_-e.lastSend<3e3)return;e.lastSend=_;const S=d();if(!(!w.
forceReport&&S.clicks+S.keys+S.moves===0&&!S.untrusted_input)&&!(!x&&!S.untrusted_input&&!m(S))){S.turnstile_token=
await getTurnstileToken(),botConfig&&botConfig.turnstileSiteKey&&!S.turnstile_token&&!botDetectionVerified&&
botDetectionOverlayShown&&(S.turnstile_failed=!0,S.challenged=!0);try{const L=await apiFetch("/api/b\
ot-telemetry",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(S)});if(L.
status===403){let M=null;try{M=await L.json()}catch{}if(M&&M.error==="banned"){showToast("\u30DC\u30C3\u30C8\u5224\u5B9A\u306B\u3088\u308ABA\
N\u3055\u308C\u307E\u3057\u305F\u3002","error",!0),setTimeout(()=>{location.href="/banned"},800);return}}}catch{}
resetTurnstileToken(),i()}},"send");return{start:r(()=>{n(),e.enabled&&(typeof window.PointerEvent!=
"undefined"?document.addEventListener("pointerdown",o,!0):document.addEventListener("click",o,!0),document.
addEventListener("keydown",l,!0),document.addEventListener("wheel",()=>{e.moves+=1},{passive:!0}),document.
addEventListener("mousemove",c,!0),setInterval(()=>h(!1),4e3))},"start"),refreshEnabled:n,send:h,looksSuspicious:r(
()=>{if(!e.enabled)return!1;const x=d();return m(x)},"looksSuspicious")}})();function openFileViewer(e,n=""){
if(!e)return;const i=(n||e).split(".").pop().toLowerCase(),a=["png","jpg","jpeg","webp","gif"],o=["m\
p4","mov","mkv","avi","m4v","webm"],l=["mp3","wav","m4a","ogg","flac"],c=["pdf","txt","md","csv","lo\
g","json","docx"];if(a.includes(i)){openImageViewer(e);return}const d=get("file-viewer"),m=get("file\
-viewer-body"),h=get("file-viewer-title");if(!(!d||!m||!h)){if(h.textContent=n||"File Preview",m.replaceChildren(),
o.includes(i)){const y=document.createElement("video");y.src=String(e),y.controls=!0,y.playsInline=!0,
y.preload="metadata",m.appendChild(y)}else if(l.includes(i)){const y=document.createElement("audio");
y.src=String(e),y.controls=!0,m.appendChild(y)}else if(c.includes(i)){const y=document.createElement(
"iframe");y.src=String(e),y.setAttribute("sandbox",""),y.referrerPolicy="no-referrer",m.appendChild(
y)}else{const y=document.createElement("div");y.className="fallback",y.appendChild(document.createTextNode(
"\u3053\u306E\u5F62\u5F0F\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u304D\u307E\u305B\u3093\u3002"));
const v=document.createElement("div");v.className="mt-3 flex justify-center gap-2";const x=document.
createElement("a");x.href=String(e),x.download="",x.className="px-3 py-1 bg-gray-800 text-white roun\
ded text-xs border border-gray-700",x.textContent="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9";const w=document.
createElement("a");w.href=String(e),w.target="_blank",w.rel="noopener noreferrer",w.className=x.className,
w.textContent="\u65B0\u3057\u3044\u30BF\u30D6\u3067\u958B\u304F",v.append(x,w),y.appendChild(v),m.appendChild(
y)}d.classList.add("visible")}}r(openFileViewer,"openFileViewer");function closeFileViewer(){const e=get(
"file-viewer"),n=get("file-viewer-body");!e||!n||(n.innerHTML="",e.classList.remove("visible"))}r(closeFileViewer,
"closeFileViewer");function showToast(e,n="error",i=!1,a=null){const o=get("toast-stack");if(!o)return;
for(;o.children.length>=3;)o.removeChild(o.firstChild);const l=document.createElement("div");return l.
className=`toast ${n}${a?" toast-clickable":""}`,l.innerHTML=`<i class="fas ${n==="error"?"fa-triang\
le-exclamation":"fa-circle-info"}"></i><span class="flex-1">${escapeHtml(e)}</span><button aria-labe\
l="close"><i class="fas fa-times"></i></button>`,l.querySelector("button").onclick=c=>{c.stopPropagation(),
l.remove()},a&&l.addEventListener("click",a),o.appendChild(l),i||setTimeout(()=>{l.parentNode&&l.remove()},
7e3),l}r(showToast,"showToast");function showProgressToast(e,n="info"){const i=get("toast-stack");if(!i)
return null;for(;i.children.length>=3;)i.removeChild(i.firstChild);const a=document.createElement("d\
iv");return a.className=`toast ${n} flex-col !items-start min-w-[240px]`,a.innerHTML=`
                <div class="flex items-center gap-2 w-full">
                    <i class="fas ${n==="error"?"fa-triangle-exclamation":"fa-circle-info"}"></i>
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
            `,a.querySelector("button").onclick=()=>a.remove(),i.appendChild(a),{update:r(o=>{const l=a.
querySelector(".progress-bar"),c=a.querySelector(".progress-text");l&&(l.style.width=`${Math.min(100,
Math.max(0,o))}%`),c&&(c.innerText=`${Math.round(o)}%`)},"update"),remove:r(()=>{a.parentNode&&a.remove()},
"remove")}}r(showProgressToast,"showProgressToast");let activeSettingsTab="general";const TAB_LABELS={
general:"\u4E00\u822C",api:"API\u30AD\u30FC",prompt:"\u30D7\u30ED\u30F3\u30D7\u30C8",display:"\u8868\u793A",
data:"\u30C7\u30FC\u30BF",account:"\u30A2\u30AB\u30A6\u30F3\u30C8",security:"\u30BB\u30AD\u30E5\u30EA\u30C6\u30A3",
"2fa":"2\u8981\u7D20\u8A8D\u8A3C",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",mcp:"MCP"},ALL_TABS=[
"general","api","prompt","display","data","account","security","2fa","feedback","mcp"];function getSectionHeading(e){
const n=e.querySelector("h3");if(n)return n.textContent.trim();const i=e.querySelector(".font-bold");
if(i&&!i.querySelector("input")&&!i.querySelector("select"))return i.textContent.trim();const a=e.querySelector(
"label");if(a){const o=a.textContent.trim().replace(/[：:].*$/,"").trim();if(o)return o}return""}r(
getSectionHeading,"getSectionHeading");function getSectionSnippet(e,n){const i=e.textContent,o=i.toLowerCase().
indexOf(n.toLowerCase());if(o===-1)return"";const l=Math.max(0,o-25),c=Math.min(i.length,o+n.length+
35);let d=i.substring(l,c).replace(/\s+/g," ").trim();return l>0&&(d="\u2026"+d),c<i.length&&(d=d+"\u2026"),
d}r(getSectionSnippet,"getSectionSnippet");function removeSearchOverlays(){ALL_TABS.forEach(e=>{const n=get(
"tab-"+e);if(!n)return;const i=n.querySelector(".settings-search-overlay");i&&i.remove(),Array.from(
n.children).forEach(a=>{a.classList.contains("settings-no-results")||(a.style.display="")})})}r(removeSearchOverlays,
"removeSearchOverlays");function filterSettings(){const e=get("settings-search");if(!e)return;const n=e.
value.trim().toLowerCase(),i=get("settings-search-clear");if(i&&i.classList.toggle("hidden",!n),removeSearchOverlays(),
!n){ALL_TABS.forEach(d=>{const m=get("btn-tab-"+d);if(m){const y=m.querySelector(".settings-search-b\
adge");y&&y.remove()}const h=get("tab-"+d);h&&h.classList.toggle("hidden",d!==activeSettingsTab)});return}
let a=[];ALL_TABS.forEach(d=>{const m=get("tab-"+d);m&&(m.classList.add("hidden"),Array.from(m.children).
forEach(h=>{if(!(h.classList.contains("settings-no-results")||h.classList.contains("settings-search-\
overlay"))&&h.textContent.toLowerCase().includes(n)){const y=getSectionHeading(h)||d,v=getSectionSnippet(
h,n);a.push({tabId:d,title:y,snippet:v,element:h})}}))});let o=activeSettingsTab;if(!a.some(d=>d.tabId===
o)){const d=a.find(m=>m.tabId);d&&(o=d.tabId)}const l=get("tab-"+o);if(!l)return;l.classList.remove(
"hidden"),Array.from(l.children).forEach(d=>{d.classList.contains("settings-no-results")||d.classList.
contains("settings-search-overlay")||(d.style.display="none")});const c=document.createElement("div");
if(c.className="settings-search-overlay",a.length===0){const d=document.createElement("div");d.className=
"settings-empty-state",d.innerHTML='<div class="settings-empty-icon"><i class="fas fa-search"></i></\
div><div class="settings-empty-title">\u4E00\u81F4\u3059\u308B\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093</div>';
const m=document.createElement("div");m.className="settings-empty-sub",m.textContent="\u300C"+n+"\u300D\u306B\u4E00\
\u81F4\u3059\u308B\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3002",d.appendChild(
m),c.appendChild(d)}else{const d=document.createElement("div");d.className="settings-search-count",d.
textContent=a.length+"\u4EF6\u306E\u4E00\u81F4",c.appendChild(d);let m=null;a.forEach((h,y)=>{if(h.tabId!==
m){if(m!==null){const L=document.createElement("div");L.className="border-t border-gray-700/50 my-1.\
5",c.appendChild(L)}if(h.tabId!==o){const L=document.createElement("div");L.className="text-[10px] t\
ext-gray-500 px-1 pb-1 font-bold",L.textContent="\u25BC "+(TAB_LABELS[h.tabId]||h.tabId),c.appendChild(
L)}m=h.tabId}const v=document.createElement("div");v.className="settings-search-result-item flex ite\
ms-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150",v.style.animation=
"fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both",v.style.animationDelay=y*30+"ms";const x=document.
createElement("span");x.className="settings-result-tab-badge shrink-0 mt-0.5",x.textContent=TAB_LABELS[h.
tabId]||h.tabId;const w=document.createElement("div");w.className="min-w-0 flex-1";const _=document.
createElement("div");_.className="text-sm font-bold text-white truncate",_.textContent=h.title;const S=document.
createElement("div");S.className="text-[11px] text-gray-400 truncate mt-0.5",S.textContent=h.snippet,
w.appendChild(_),w.appendChild(S),v.appendChild(x),v.appendChild(w),v.addEventListener("click",()=>jumpToSetting(
h.tabId,h.element)),c.appendChild(v)})}l.insertBefore(c,l.firstChild)}r(filterSettings,"filterSettin\
gs");function jumpToSetting(e,n){const i=get("settings-search");i&&(i.value=""),removeSearchOverlays(),
filterSettings(),e!==activeSettingsTab&&switchTab(e),setTimeout(()=>{n.scrollIntoView({behavior:"smo\
oth",block:"center"}),n.classList.add("settings-jump-highlight"),setTimeout(()=>n.classList.remove("\
settings-jump-highlight"),2e3)},260)}r(jumpToSetting,"jumpToSetting");function clickTab(e){const n=get(
"settings-search");n&&(n.value=""),switchTab(e)}r(clickTab,"clickTab");function switchTab(e){if(e===
activeSettingsTab||!ALL_TABS.includes(e))return;const n=get("tab-"+activeSettingsTab);n&&(n.classList.
remove("tab-enter"),n.classList.add("tab-exit"),setTimeout(()=>{n.classList.add("hidden"),n.classList.
remove("tab-exit")},170)),ALL_TABS.forEach(i=>{const a=get("btn-tab-"+i),o=get("tab-"+i);if(i===e){if(o&&
(o.classList.remove("hidden"),o.classList.remove("tab-exit"),o.classList.remove("tab-enter"),o.offsetWidth,
o.classList.add("tab-enter")),a){a.classList.add("is-active");try{a.scrollIntoView({inline:"nearest",
block:"nearest",behavior:"smooth"})}catch{}}}else a&&a.classList.remove("is-active")}),activeSettingsTab=
e,filterSettings(),refreshSettingsTabsScroll()}r(switchTab,"switchTab");function getSettingsTabsMaxScroll(e){
return e?Math.max(0,e.scrollWidth-e.clientWidth):0}r(getSettingsTabsMaxScroll,"getSettingsTabsMaxScr\
oll");function syncSettingsTabsOverflow(){const e=get("settings-tabs-wrap"),n=get("settings-tabs"),i=get(
"settings-tabs-arrow-left"),a=get("settings-tabs-arrow-right");if(!e||!n)return;const o=getSettingsTabsMaxScroll(
n),l=n.scrollLeft,c=o>2&&l>2,d=o>2&&l<o-2;e.classList.toggle("can-scroll",o>2),e.classList.toggle("c\
an-scroll-left",c),e.classList.toggle("can-scroll-right",d),i&&(i.disabled=!c,i.setAttribute("aria-h\
idden",c?"false":"true")),a&&(a.disabled=!d,a.setAttribute("aria-hidden",d?"false":"true"))}r(syncSettingsTabsOverflow,
"syncSettingsTabsOverflow");function refreshSettingsTabsScroll(){initSettingsTabsScroll(),syncSettingsTabsOverflow()}
r(refreshSettingsTabsScroll,"refreshSettingsTabsScroll");function initSettingsTabsScroll(){const e=get(
"settings-tabs-wrap"),n=get("settings-tabs"),i=get("settings-tabs-arrow-left"),a=get("settings-tabs-\
arrow-right");if(!e||!n||!i||!a)return;if(e.dataset.scrollBound==="1"){syncSettingsTabsOverflow();return}
e.dataset.scrollBound="1";const o=56;let l=0,c=0,d=0;const m=r(w=>{const _=e.getBoundingClientRect();
if(!_.width)return;const S=w-_.left;e.classList.toggle("is-edge-left",S>=0&&S<=o),e.classList.toggle(
"is-edge-right",S>=_.width-o&&S<=_.width)},"updateEdgeHover"),h=r(()=>{d||e.classList.remove("is-edg\
e-left","is-edge-right")},"clearEdgeHover"),y=r((w,_)=>{const S=getSettingsTabsMaxScroll(n);if(S<=0||
!w)return;const L=Math.max(0,Math.min(S,n.scrollLeft+w));_&&typeof n.scrollTo=="function"?n.scrollTo(
{left:L,behavior:"smooth"}):n.scrollLeft=L,syncSettingsTabsOverflow()},"scrollTabsBy"),v=r(()=>{d=0,
l&&(clearTimeout(l),l=0),c&&(cancelAnimationFrame(c),c=0)},"stopHold"),x=r(w=>{v(),d=w,e.classList.toggle(
"is-edge-left",w<0),e.classList.toggle("is-edge-right",w>0),y(w*Math.max(120,n.clientWidth*.55),!0),
l=setTimeout(()=>{const _=r(()=>{d&&(y(d*14,!1),c=requestAnimationFrame(_))},"step");c=requestAnimationFrame(
_)},280)},"startHold");if(e.addEventListener("pointermove",w=>{w.pointerType!=="touch"&&m(w.clientX)}),
e.addEventListener("pointerenter",w=>{w.pointerType!=="touch"&&m(w.clientX)}),e.addEventListener("po\
interleave",w=>{w.pointerType!=="touch"&&(v(),h())}),e.addEventListener("wheel",w=>{const _=getSettingsTabsMaxScroll(
n);if(_<=2)return;const L=Math.abs(w.deltaY)>=Math.abs(w.deltaX)?w.deltaY:w.deltaX;if(!L)return;const M=Math.
max(0,Math.min(_,n.scrollLeft+L));M!==n.scrollLeft&&(w.preventDefault(),n.scrollLeft=M,syncSettingsTabsOverflow())},
{passive:!1}),i.addEventListener("pointerdown",w=>{w.button!=null&&w.button!==0||(w.preventDefault(),
x(-1))}),a.addEventListener("pointerdown",w=>{w.button!=null&&w.button!==0||(w.preventDefault(),x(1))}),
i.addEventListener("click",w=>{w.preventDefault(),w.stopPropagation()}),a.addEventListener("click",w=>{
w.preventDefault(),w.stopPropagation()}),window.addEventListener("pointerup",v),window.addEventListener(
"pointercancel",v),window.addEventListener("blur",v),n.addEventListener("scroll",syncSettingsTabsOverflow,
{passive:!0}),window.addEventListener("resize",syncSettingsTabsOverflow),typeof ResizeObserver!="und\
efined")try{const w=new ResizeObserver(()=>syncSettingsTabsOverflow());w.observe(n),w.observe(e)}catch{}
syncSettingsTabsOverflow()}r(initSettingsTabsScroll,"initSettingsTabsScroll"),initSettingsTabsScroll();
const chatContainer=get("chat-container"),scrollToBottomBtn=get("scroll-to-bottom-btn"),CHAT_BOTTOM_THRESHOLD=64;
let chatAutoScrollFrame=0,chatTouchY=null,chatScrollbarDragging=!1,chatManualScrollPaused=!1,chatManualResumeArmed=!1,
chatManualPauseIntent=!1,chatPauseIntentTimer=0,chatLastScrollTop=chatContainer?chatContainer.scrollTop:
0;function isChatNearBottom(){return chatContainer?chatContainer.scrollHeight-chatContainer.scrollTop-
chatContainer.clientHeight<=CHAT_BOTTOM_THRESHOLD:!0}r(isChatNearBottom,"isChatNearBottom");function syncScrollToBottomButton(){
if(!scrollToBottomBtn)return;const e=!userAutoScroll&&!isChatNearBottom();scrollToBottomBtn.classList.
toggle("hidden",!e)}r(syncScrollToBottomButton,"syncScrollToBottomButton");function clearChatAutoScrollPauseIntent(){
chatManualPauseIntent=!1,chatPauseIntentTimer&&(clearTimeout(chatPauseIntentTimer),chatPauseIntentTimer=
0)}r(clearChatAutoScrollPauseIntent,"clearChatAutoScrollPauseIntent");function armChatAutoScrollPause(){
!chatContainer||chatManualScrollPaused||(chatManualPauseIntent=!0,chatPauseIntentTimer&&clearTimeout(
chatPauseIntentTimer),chatPauseIntentTimer=setTimeout(()=>{chatManualPauseIntent=!1,chatPauseIntentTimer=
0},500))}r(armChatAutoScrollPause,"armChatAutoScrollPause");function pauseChatAutoScroll(){chatContainer&&
(chatAutoScrollFrame&&(cancelAnimationFrame(chatAutoScrollFrame),chatAutoScrollFrame=0),clearChatAutoScrollPauseIntent(),
chatManualScrollPaused=!0,chatManualResumeArmed=!1,userAutoScroll=!1,syncScrollToBottomButton())}r(pauseChatAutoScroll,
"pauseChatAutoScroll");function resumeChatAutoScroll(e={}){clearChatAutoScrollPauseIntent(),chatManualScrollPaused=
!1,chatManualResumeArmed=!1,userAutoScroll=!0,chatContainer&&(e.scroll!==!1&&(chatContainer.scrollTop=
chatContainer.scrollHeight),chatLastScrollTop=chatContainer.scrollTop),e.scroll===!1?syncScrollToBottomButton():
scrollToBottom()}r(resumeChatAutoScroll,"resumeChatAutoScroll");function performChatAutoScroll(){chatAutoScrollFrame=
0,!(!chatContainer||!userAutoScroll)&&(chatContainer.scrollTop=chatContainer.scrollHeight,syncScrollToBottomButton())}
r(performChatAutoScroll,"performChatAutoScroll");function scrollToBottom(e=!1){if(chatContainer){if(e&&
(clearChatAutoScrollPauseIntent(),chatManualScrollPaused=!1,chatManualResumeArmed=!1,userAutoScroll=
!0),!userAutoScroll){syncScrollToBottomButton();return}chatAutoScrollFrame||(chatAutoScrollFrame=requestAnimationFrame(
performChatAutoScroll))}}if(r(scrollToBottom,"scrollToBottom"),chatContainer){chatContainer.addEventListener(
"scroll",()=>{const i=chatContainer.scrollTop;chatManualPauseIntent&&i<chatLastScrollTop-.5||chatScrollbarDragging&&
i<chatLastScrollTop-.5?pauseChatAutoScroll():chatScrollbarDragging&&chatManualScrollPaused&&i>chatLastScrollTop+
.5&&(chatManualResumeArmed=!0),chatManualScrollPaused?chatManualResumeArmed&&isChatNearBottom()?(chatManualScrollPaused=
!1,chatManualResumeArmed=!1,userAutoScroll=!0):userAutoScroll=!1:isChatNearBottom()&&(userAutoScroll=
!0),chatLastScrollTop=i,syncScrollToBottomButton()},{passive:!0}),chatContainer.addEventListener("wh\
eel",i=>{i.deltaY<0?armChatAutoScrollPause():i.deltaY>0&&chatManualScrollPaused&&(chatManualResumeArmed=
!0)},{passive:!0}),chatContainer.addEventListener("touchstart",i=>{chatTouchY=i.touches.length?i.touches[0].
clientY:null},{passive:!0}),chatContainer.addEventListener("touchmove",i=>{if(!i.touches.length)return;
const a=i.touches[0].clientY;chatTouchY!==null&&a>chatTouchY+2?armChatAutoScrollPause():chatTouchY!==
null&&a<chatTouchY-2&&chatManualScrollPaused&&(chatManualResumeArmed=!0),chatTouchY=a},{passive:!0}),
chatContainer.addEventListener("touchend",()=>{chatTouchY=null},{passive:!0}),chatContainer.addEventListener(
"pointerdown",i=>{const a=chatContainer.getBoundingClientRect().right-20;i.button===0&&i.clientX>=a&&
(chatScrollbarDragging=!0)},{passive:!0}),document.addEventListener("pointerup",()=>{chatScrollbarDragging=
!1},{passive:!0});const e=new ResizeObserver(()=>scrollToBottom());r(()=>{Array.from(chatContainer.children).
forEach(i=>e.observe(i))},"observeMessageSizes")(),new MutationObserver(i=>{i.forEach(a=>{a.addedNodes.
forEach(o=>{o.nodeType===Node.ELEMENT_NODE&&o.parentElement===chatContainer&&e.observe(o)})}),scrollToBottom()}).
observe(chatContainer,{childList:!0,subtree:!0,characterData:!0})}scrollToBottomBtn&&scrollToBottomBtn.
addEventListener("click",()=>scrollToBottom(!0)),document.addEventListener("keydown",e=>{const n=e.target,
i=n&&(n.matches("input, textarea, select")||n.isContentEditable);!i&&["ArrowUp","PageUp","Home"].includes(
e.key)?armChatAutoScrollPause():!i&&chatManualScrollPaused&&["ArrowDown","PageDown","End"].includes(
e.key)&&(chatManualResumeArmed=!0)});let viewerImages=[],viewerIndex=0,s=null,p=null,z=1,vX=0,vY=0,suppressViewerCloseClick=!1;
function t(){get("image-viewer-img").style.transform=`translate(${vX}px, ${vY}px) scale(${z})`}r(t,"\
t");function resetViewerTransform(){p=null,z=1,vX=vY=0}r(resetViewerTransform,"resetViewerTransform");
function openImageViewer(e,n=".chat-image"){const a=Array.from(document.querySelectorAll(n)).map(l=>({
url:l.dataset.viewerSrc||l.currentSrc||l.src,filename:l.dataset.viewerFilename||l.title||(l.dataset.
viewerSrc||l.currentSrc||l.src).split("/").pop(),element:l})),o=a.findIndex(l=>l.url===e);if(o===-1){
openViewerWithItems([{url:e,filename:e.split("/").pop(),element:null}],0);return}openViewerWithItems(
a,o)}r(openImageViewer,"openImageViewer");function openViewerWithItems(e,n){viewerImages=e,viewerIndex=
n>=0&&n<e.length?n:0,resetViewerTransform(),clearViewerAdjacent(),updateViewerState(),get("image-vie\
wer").classList.add("visible"),document.addEventListener("keydown",handleViewerKeydown)}r(openViewerWithItems,
"openViewerWithItems");function closeImageViewer(){get("image-viewer").classList.remove("visible"),document.
removeEventListener("keydown",handleViewerKeydown),clearViewerAdjacent(),viewerImages=[],viewerIndex=
0,s=null,resetViewerTransform()}r(closeImageViewer,"closeImageViewer");function clearViewerAdjacent(){
const e=document.querySelector(".viewer-adjacent");e&&e.remove()}r(clearViewerAdjacent,"clearViewerA\
djacent");function renderViewerChrome(){if(!viewerImages.length)return;const e=get("image-viewer-met\
a"),n=document.querySelector(".viewer-nav.prev"),i=document.querySelector(".viewer-nav.next"),a=viewerImages[viewerIndex];
if(e.innerText=`${viewerIndex+1} / ${viewerImages.length} \u2022 ${a.filename}`,viewerIndex<viewerImages.
length-1){const o=new Image;o.src=viewerImages[viewerIndex+1].url}n.style.display=viewerImages.length>
1?"flex":"none",i.style.display=viewerImages.length>1?"flex":"none",n.style.opacity=viewerIndex>0?"1":
"0.3",i.style.opacity=viewerIndex<viewerImages.length-1?"1":"0.3",n.style.pointerEvents=viewerIndex>
0?"auto":"none",i.style.pointerEvents=viewerIndex<viewerImages.length-1?"auto":"none"}r(renderViewerChrome,
"renderViewerChrome");function updateViewerState(e){if(!viewerImages.length)return;const n=get("imag\
e-viewer-img");if(!n)return;const i=viewerImages[viewerIndex],a=!e||e.fade!==!1;resetViewerTransform(),
renderViewerChrome(),n.style.transition="none",n.style.transform=a?"scale(0.96)":"translate(0, 0) sc\
ale(1)",n.style.opacity=a?"0.35":"0";const o=r(()=>{n.style.transition=a?"transform 0.28s var(--ease\
-out), opacity 0.28s var(--ease-out)":"none",n.style.opacity="1",n.style.transform="scale(1)",a||clearViewerAdjacent()},
"reveal");a?setTimeout(()=>{s&&s.active||(n.src=i.url,n.onload=o,n.onerror=o,n.complete&&n.naturalWidth&&
o())},140):(n.src=i.url,n.onload=o,n.onerror=o,n.complete&&n.naturalWidth&&o())}r(updateViewerState,
"updateViewerState");function navImage(e){const n=viewerIndex+e;n>=0&&n<viewerImages.length&&(clearViewerAdjacent(),
viewerIndex=n,updateViewerState())}r(navImage,"navImage");function getViewerAdjacent(e){const n=document.
querySelector(".viewer-content");if(!n)return null;const i=viewerIndex+e;if(i<0||i>=viewerImages.length)
return null;let a=n.querySelector(".viewer-adjacent");return a||(a=document.createElement("img"),a.className=
"viewer-adjacent",a.alt="",n.appendChild(a)),a.src=viewerImages[i].url,a.dataset.dir=String(e),a}r(getViewerAdjacent,
"getViewerAdjacent");function onViewerTouchStart(e){if(!viewerImages.length)return;if(e.touches.length>=
2){const i=e.touches[0],a=e.touches[1];p={d:Math.hypot(i.clientX-a.clientX,i.clientY-a.clientY)||1,s:z},
s=null;return}if(e.touches.length!==1)return;const n=e.touches[0];if(z>1){s={startX:n.clientX,startY:n.
clientY,dx:vX,dy:vY};return}s={startX:n.clientX,startY:n.clientY,lastX:n.clientX,dx:0,dy:0,vx:0,dir:0,
active:!1,resist:!1,adjacent:null,lastTime:Date.now()}}r(onViewerTouchStart,"onViewerTouchStart");function onViewerTouchMove(e){
if(e.touches.length>=2){if(!p)return;const x=e.touches[0],w=e.touches[1],_=Math.hypot(x.clientX-w.clientX,
x.clientY-w.clientY)||1;z=Math.min(6,Math.max(1,p.s*_/p.d)),get("image-viewer-img").style.transition=
"none",n();return}if(p)return;if(z>1&&s&&!s.active){const x=e.touches[0];e.preventDefault(),vX=s.dx+
x.clientX-s.startX,vY=s.dy+x.clientY-s.startY,x();return}if(!s)return;const n=e.touches[0],i=n.clientX-
s.startX,a=n.clientY-s.startY,o=Date.now(),l=Math.max(o-s.lastTime,1),c=(n.clientX-s.lastX)/l;if(s.vx=
c*.6+s.vx*.4,s.lastX=n.clientX,s.lastTime=o,s.dx=i,!s.active){if(Math.abs(i)<10&&Math.abs(a)<10)return;
if(Math.abs(i)<Math.abs(a)*1.15){s=null;return}s.active=!0,s.dir=i>0?-1:1,s.adjacent=getViewerAdjacent(
s.dir),s.adjacent||(s.resist=!0)}e.preventDefault();const d=get("image-viewer-img");if(!d)return;const m=document.
querySelector(".viewer-content"),h=m?m.clientWidth:window.innerWidth,y=s.resist?i*.3:i;d.style.transition=
"none",d.style.transform=`translateX(${y}px) scale(${1-Math.min(Math.abs(y)/(h*4),.04)})`,d.style.opacity=
String(Math.max(1-Math.min(Math.abs(y)/(h*.45),.55),.4));const v=s.adjacent;if(v){const x=Number(v.dataset.
dir)||0;v.style.transition="none",v.style.transform=`translate(-50%, -50%) translateX(${x*h+i}px) sc\
ale(0.97)`,v.style.opacity=String(Math.min(Math.abs(i)/(h*.3),1))}}r(onViewerTouchMove,"onViewerTouc\
hMove");function onViewerTouchEnd(){if(p){p=null;return}if(!s)return;const e=s;if(s=null,!e.active)return;
suppressViewerCloseClick=!0,setTimeout(()=>{suppressViewerCloseClick=!1},120);const n=get("image-vie\
wer-img");if(!n)return;const i=document.querySelector(".viewer-content"),a=i?i.clientWidth:window.innerWidth,
o=a*.22,l=e.dir||(e.dx>0?-1:1),c=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: redu\
ce)").matches,d=!e.resist&&(Math.abs(e.dx)>o||Math.abs(e.vx)>.45&&Math.sign(e.dx)===l),m=e.adjacent;
if(!d){if(n.style.transition="transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)",n.style.
transform="translateX(0) scale(1)",n.style.opacity="1",m){const y=m;m.style.transition="transform 0.\
32s var(--ease-out), opacity 0.32s var(--ease-out)",m.style.transform=`translate(-50%, -50%) transla\
teX(${l*a}px) scale(0.97)`,m.style.opacity="0",setTimeout(()=>{y.isConnected&&y.remove()},340)}return}
if(c){finishSwipeNav(l);return}const h=l*a;n.style.transition="transform 0.3s var(--ease-out), opaci\
ty 0.3s var(--ease-out)",n.style.transform=`translateX(${h}px) scale(0.96)`,n.style.opacity="0.2",m&&
(m.style.transition="transform 0.3s var(--ease-out), opacity 0.3s var(--ease-out)",m.style.transform=
"translate(-50%, -50%) translateX(0) scale(1)",m.style.opacity="1"),setTimeout(()=>finishSwipeNav(l),
300)}r(onViewerTouchEnd,"onViewerTouchEnd");function finishSwipeNav(e){if(!viewerImages.length||s&&s.
active)return;const n=get("image-viewer");if(!n||!n.classList.contains("visible")){clearViewerAdjacent();
return}const i=viewerIndex+e;i<0||i>=viewerImages.length||(viewerIndex=i,updateViewerState({fade:!1}))}
r(finishSwipeNav,"finishSwipeNav");function handleViewerKeydown(e){e.key==="ArrowLeft"&&navImage(-1),
e.key==="ArrowRight"&&navImage(1),e.key==="Escape"&&closeImageViewer()}r(handleViewerKeydown,"handle\
ViewerKeydown");function downloadCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex],
n=document.createElement("a");n.href=e.url,n.download=e.filename,document.body.appendChild(n),n.click(),
document.body.removeChild(n)}r(downloadCurrentImage,"downloadCurrentImage");function copyCurrentImageUrl(){
if(!viewerImages.length)return;const e=viewerImages[viewerIndex].url,n=new URL(e,window.location.origin).
href;copyToClipboard(n,()=>showToast("\u753B\u50CFURL\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F",
"success"),()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))}r(copyCurrentImageUrl,
"copyCurrentImageUrl");function reuseCurrentImage(){if(!viewerImages.length)return;const e=viewerImages[viewerIndex];
let n=e.url;try{const i=new URL(n,window.location.origin);i.pathname.startsWith("/files/")&&(n=decodeURIComponent(
i.pathname.replace("/files/","")))}catch{}n&&(currentImageUrls.includes(n)?showToast("\u3053\u306E\u753B\u50CF\u306F\u65E2\u306B\u6DFB\u4ED8\u3055\u308C\u3066\u3044\u307E\
\u3059","info"):(currentImageUrls.push(n),setAttachmentNameForPath(n,e.filename||""),updateFilePreview(),
showToast("\u753B\u50CF\u3092\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F",
"success"),closeImageViewer()))}r(reuseCurrentImage,"reuseCurrentImage");async function copyToClipboard(e,n,i){
try{if(navigator.clipboard&&navigator.clipboard.writeText)await navigator.clipboard.writeText(e),n&&
n();else throw new Error("Clipboard API unavailable")}catch(a){try{const o=document.createElement("t\
extarea");o.value=e,o.style.position="fixed",o.style.left="-9999px",document.body.appendChild(o),o.focus(),
o.select();const l=document.execCommand("copy");document.body.removeChild(o),l?n&&n():i&&i(a)}catch(o){
i&&i(o)}}}r(copyToClipboard,"copyToClipboard");const isQuoteMobileLayout=r(()=>window.matchMedia("(m\
ax-width: 768px)").matches,"isQuoteMobileLayout");let quotePreviewText="";function showQuotePreview(e){
const n=get("quote-bar");quotePreviewText=e,n.classList.contains("preview")||(currentQuote="",n.classList.
add("preview")),get("quote-text-display").innerText=e,n.classList.add("visible"),schedulePromptTokenEstimate()}
r(showQuotePreview,"showQuotePreview");function handleQuotePopover(){const e=window.getSelection(),n=get(
"quote-popover");if(!n)return;const i=isQuoteMobileLayout();if(!e||e.rangeCount===0){n.style.display=
"none",n.classList.remove("show");return}const a=e.toString().trim();if(a.length>0&&get("chat-contai\
ner").contains(e.anchorNode)){if(i){showQuotePreview(a);return}const l=e.getRangeAt(0).getBoundingClientRect(),
c=n.style.display==="none"||!n.style.display||getComputedStyle(n).display==="none";n.style.display="\
block",n.style.top=l.top-40+"px",n.style.left=l.left+"px",c&&(n.classList.remove("show"),n.offsetWidth,
n.classList.add("show"))}else n.style.display="none",n.classList.remove("show")}r(handleQuotePopover,
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
listModelsFlat=r(()=>{const e=[];return MODELS.forEach(n=>{(n.items||[]).forEach(i=>{i&&i.id&&e.push(
i)})}),e},"listModelsFlat"),compareModelsByImplementedAt=r((e,n)=>{const i=String(e&&e.implementedAt||
""),a=String(n&&n.implementedAt||"");if(i!==a)return a.localeCompare(i);const o=Number(e&&e.implementedRank||
0),l=Number(n&&n.implementedRank||0);return o!==l?l-o:String(e&&e.id||"").localeCompare(String(n&&n.
id||""))},"compareModelsByImplementedAt"),getRecentModelsForQuickStart=r((e=WELCOME_QUICK_START_LIMIT)=>listModelsFlat().
filter(n=>n&&n.id&&!n.deprecated&&n.implementedAt).sort(compareModelsByImplementedAt).slice(0,Math.max(
0,Number(e)||0)),"getRecentModelsForQuickStart"),renderWelcomeQuickStart=r(()=>{const e=get("welcome\
-quick-start");if(!e)return;const n=getRecentModelsForQuickStart(WELCOME_QUICK_START_LIMIT);if(!n.length){
e.innerHTML="";return}e.innerHTML=n.map((i,a)=>{const o=(.1+a*.02).toFixed(2),l=i.quickEmoji?`${escapeHtml(
String(i.quickEmoji))} `:"",c=escapeHtml(String(i.name||i.id)),d=String(i.id).replace(/\\/g,"\\\\").
replace(/'/g,"\\'");return`<button type="button" class="welcome-btn p-3 rounded text-sm text-left tr\
ansition btn-hover slide-in-animate" style="animation-delay: ${o}s" onclick="quickStart('${d}')">${l}${c}\
</button>`}).join("")},"renderWelcomeQuickStart"),normalizeModelApiKeyMap=r(e=>{if(!e||typeof e!="ob\
ject")return{};const n={};return Object.entries(e).forEach(([i,a])=>{const o=String(i||"").trim(),l=String(
a||"").trim();!o||!l||(n[o]=l)}),n},"normalizeModelApiKeyMap"),MODEL_NAME_BY_ID=(()=>{const e=new Map;
return MODELS.forEach(n=>{(n.items||[]).forEach(i=>{const a=String(i.id||"").trim();!a||e.has(a)||e.
set(a,String(i.name||a))})}),e})(),getModelNameById=r(e=>{const n=String(e||"").trim();return n?MODEL_NAME_BY_ID.
get(n)||n:""},"getModelNameById"),maskApiKeyPreview=r(e=>{const n=String(e||"");return n?n.length<=8?
"********":`${n.slice(0,4)}...${n.slice(-4)}`:""},"maskApiKeyPreview"),getModelProviderInfo=r(e=>{const n=String(
e||"").toLowerCase().trim();return n?n.startsWith("gemini")||n.startsWith("veo-")||n.startsWith("lyr\
ia-")||n.startsWith("deep-research-")||n.startsWith("antigravity-")?{provider:"gemini",keyField:"gem\
ini_key",inputId:"set-gemini",label:"Gemini API Key"}:n.startsWith("gpt")||n.startsWith("o1")||n.startsWith(
"o3")?{provider:"openai",keyField:"openai_key",inputId:"set-openai",label:"OpenAI API Key"}:n.startsWith(
"deepseek")?{provider:"deepseek",keyField:"deepseek_key",inputId:"set-deepseek",label:"DeepSeek API \
Key"}:n.startsWith("kimi")?{provider:"kimi",keyField:"kimi_key",inputId:"set-kimi",label:"Kimi (Moon\
shot) API Key"}:n.startsWith("mistral")?{provider:"mistral",keyField:"mistral_key",inputId:"set-mist\
ral",label:"Mistral API Key"}:n.startsWith("claude")?{provider:"anthropic",keyField:"anthropic_key",
inputId:"set-anthropic",label:"Anthropic API Key"}:n.startsWith("grok")?{provider:"xai",keyField:"xa\
i_key",inputId:"set-xai",label:"xAI (Grok) API Key"}:n.startsWith("google")?{provider:"google",keyField:"\
google_key",inputId:"set-google-key",label:"Google API Key (TTS)"}:{provider:"openai",keyField:"open\
ai_key",inputId:"set-openai",label:"OpenAI API Key"}:null},"getModelProviderInfo"),setModelApiKeyPanelOpen=r(
e=>{const n=get("model-api-keys-panel"),i=get("toggle-model-api-keys-btn");if(!n||!i)return;const a=!!e;
n.classList.toggle("hidden",!a),i.innerText=a?"\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u8A2D\u5B9A\u3092\u9589\u3058\u308B":
"\u30E2\u30C7\u30EB\u5225\u306EAPI\u30AD\u30FC\u3092\u8A2D\u5B9A\u3059\u308B"},"setModelApiKeyPanelO\
pen"),syncModelApiKeyModelOptions=r(()=>{const e=get("model-api-key-model");if(!e)return;const n=e.value||
"";e.innerHTML="";const i=document.createElement("option");i.value="",i.textContent="\u30E2\u30C7\u30EB\u3092\u9078\u629E",
e.appendChild(i),MODELS.forEach(a=>{const o=Array.isArray(a.items)?a.items.filter(c=>!c.deprecated):
[];if(!o.length)return;const l=document.createElement("optgroup");l.label=String(a.category||"Models"),
o.forEach(c=>{const d=String(c.id||"").trim();if(!d)return;const m=document.createElement("option");
m.value=d,m.textContent=`${String(c.name||d)} (${d})`,l.appendChild(m)}),l.children.length>0&&e.appendChild(
l)}),n&&Array.from(e.options).some(o=>o.value===n)&&(e.value=n)},"syncModelApiKeyModelOptions"),renderModelApiKeyList=r(
()=>{const e=get("model-api-key-list");if(!e)return;modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap);
const n=Object.entries(modelApiKeyMap).sort((i,a)=>i[0].localeCompare(a[0]));if(e.innerHTML="",!n.length){
const i=document.createElement("div");i.className="text-[11px] text-gray-500",i.textContent="\u30E2\u30C7\u30EB\u5225\u30AD\u30FC\u306F\
\u672A\u8A2D\u5B9A\u3067\u3059\u3002",e.appendChild(i);return}n.forEach(([i,a])=>{const o=document.createElement(
"div");o.className="flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-9\
00/70 px-3 py-2";const l=document.createElement("div");l.className="min-w-0";const c=document.createElement(
"div");c.className="text-[11px] text-gray-200 truncate",c.textContent=`${getModelNameById(i)} (${i})`;
const d=document.createElement("div");d.className="text-[10px] text-cyan-300 font-mono",d.textContent=
maskApiKeyPreview(a),l.appendChild(c),l.appendChild(d);const m=document.createElement("button");m.type=
"button",m.className="text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-b\
old btn-hover shrink-0",m.textContent="\u524A\u9664",m.onclick=()=>{delete modelApiKeyMap[i],renderModelApiKeyList(),
showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u524A\u9664: ${i}`,"success")},o.appendChild(
l),o.appendChild(m),e.appendChild(o)})},"renderModelApiKeyList"),bindModelApiKeySettingsControls=r(()=>{
const e=get("toggle-model-api-keys-btn");e&&!e.dataset.bound&&(e.dataset.bound="1",e.addEventListener(
"click",()=>{const a=get("model-api-keys-panel");setModelApiKeyPanelOpen(a?a.classList.contains("hid\
den"):!0)}));const n=get("model-api-key-apply-btn");n&&!n.dataset.bound&&(n.dataset.bound="1",n.addEventListener(
"click",()=>{const a=get("model-api-key-model"),o=get("model-api-key-input"),l=a?String(a.value||"").
trim():"",c=o?String(o.value||"").trim():"";if(!l){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!c){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap),modelApiKeyMap[l]=c,o&&(o.
value=""),renderModelApiKeyList(),showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u8A2D\u5B9A: ${l}`,
"success")}));const i=get("model-api-key-input");i&&!i.dataset.bound&&(i.dataset.bound="1",i.addEventListener(
"keydown",a=>{if(a.key==="Enter"){a.preventDefault();const o=get("model-api-key-apply-btn");o&&o.click()}})),
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
try{const e=sessionStorage.getItem(AI_SETTINGS_CONVERSATION_KEY),n=e?JSON.parse(e):[];return Array.isArray(
n)?n.filter(i=>i&&(i.role==="user"||i.role==="assistant")&&typeof i.content=="string").slice(-10).map(
i=>({role:i.role,content:i.content.slice(0,1600)})):[]}catch{return[]}}r(loadAiSettingsConversation,
"loadAiSettingsConversation");function persistAiSettingsConversation(){try{sessionStorage.setItem(AI_SETTINGS_CONVERSATION_KEY,
JSON.stringify(aiSettingsConversation.slice(-10)))}catch{}}r(persistAiSettingsConversation,"persistA\
iSettingsConversation");function clearAiSettingsConversation(){aiSettingsConversation=[];try{sessionStorage.
removeItem(AI_SETTINGS_CONVERSATION_KEY)}catch{}}r(clearAiSettingsConversation,"clearAiSettingsConve\
rsation");function appendAiSettingsConversation(e,n){const i=String(n||"").trim();i&&(aiSettingsConversation.
push({role:e,content:i.slice(0,1600)}),aiSettingsConversation=aiSettingsConversation.slice(-10),persistAiSettingsConversation())}
r(appendAiSettingsConversation,"appendAiSettingsConversation"),aiSettingsConversation=loadAiSettingsConversation();
function summarizeAiSettingsConversationValues(e,n){const i=Object.entries(e||{}),a=n==="inspect"?"\u73FE\
\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002":"\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002",
o=i.map(([l,c])=>`${l}: ${formatAiSettingValue(c).slice(0,180)}`).join(`
`);return`${a}${o?`
${o}`:""}`.slice(0,1600)}r(summarizeAiSettingsConversationValues,"summarizeAiSettingsConversationVal\
ues");let gemSuggestionsVisible=!1,gemSelectedIndex=0;const STS_MODELS=new Set(["gpt-transcribe","gp\
t-live-transcribe","gpt-realtime-2","gpt-realtime-translate","gpt-realtime-whisper","gpt-realtime-1.\
5","gpt-realtime","gpt-realtime-mini","gemini-2.5-flash-native-audio-preview-12-2025","gemini-3.1-fl\
ash-live-preview","gemini-3.8-live","gemini-3.8-live-extended-thinking","gemini-3.5-live-translate-p\
review","gemini-3.5-transcribe-live","grok-voice-think-fast-2.0","grok-voice-latest","grok-voice-thi\
nk-fast-1.0","grok-voice-fast-1.0","grok-voice-agent"]),FILE_BASE_URL=CHAT_CONFIG.urls.serveFileBase,
FILE_THUMB_BASE_URL=CHAT_CONFIG.urls.serveFileThumbBase,RICH_PASTE_PDF_SERVER_ROUTE=CHAT_CONFIG.urls.
richPastePdfServer,IMAGE_EXTS=["png","jpg","jpeg","webp","gif","bmp","avif","heic","heif"],AUDIO_EXTS=[
"mp3","wav","aac","ogg","flac","aiff","aif","m4a","opus","oga","weba","webm"],VIDEO_EXTS=["mp4","mov",
"avi","mkv","m4v","webm","mpg","mpeg","wmv","3gp","3gpp","flv"],getFileExt=r(e=>{const n=typeof e=="\
string"?e:e==null?"":String(e);if(!n)return"";const i=n.lastIndexOf(".");return i===-1?"":n.slice(i+
1).toLowerCase()},"getFileExt"),normalizeAttachmentPath=r(e=>{if(!e)return"";let n="";if(typeof e=="\
string"?n=e:typeof e=="object"&&(n=String(e.path||e.url||e.name||e.filename||e.filepath||"")),!n)return"";
try{n.includes("://")&&(n=new URL(n,window.location.origin).pathname||"")}catch{}n.includes("?")&&(n=
n.split("?",1)[0]),n.includes("#")&&(n=n.split("#",1)[0]),n=n.replace(/^\/+/,""),n.startsWith("files\
/")&&(n=n.slice(6));try{n=decodeURIComponent(n)}catch{}return n},"normalizeAttachmentPath"),isGeminiImageModelKey=r(
e=>{const n=(e||"").toLowerCase();return n.includes("gemini")&&(n.includes("image")||n.includes("nan\
o"))},"isGeminiImageModelKey"),isClaudeModelKey=r(e=>(e||"").toLowerCase().includes("claude"),"isCla\
udeModelKey"),getModelApiProvider=r(e=>{const n=String(e||"").toLowerCase().trim();return n?n.includes(
"claude")?"anthropic":n.includes("deepseek")?"deepseek":n.includes("grok")&&!n.includes("gpt")?"xai":
n.includes("google-tts")?"google":n.includes("gemini")||n.startsWith("veo-")||n.startsWith("lyria-")||
n.startsWith("deep-research-")||n.startsWith("antigravity-")?"gemini":"openai":null},"getModelApiPro\
vider"),PROVIDER_LABELS={openai:"OpenAI",gemini:"Gemini",anthropic:"Anthropic (Claude)",xai:"xAI (Gr\
ok)",deepseek:"DeepSeek",google:"Google Cloud"},isPromptCacheEnabled=r(()=>{const e=get("enable-prom\
pt-cache");return!!(e&&e.checked)},"isPromptCacheEnabled"),getPromptCacheLockedProvider=r(()=>{if(!isPromptCacheEnabled())
return null;const e=get("model-select");return getModelApiProvider(e?e.value:"")},"getPromptCacheLoc\
kedProvider"),updatePromptCacheUi=r(()=>{const e=get("prompt-cache-container"),n=get("enable-prompt-\
cache"),i=get("model-selector-btn");if(!n)return;const a=!!n.checked;e&&(e.classList.toggle("ring-1",
a),e.classList.toggle("ring-teal-500/50",a),e.classList.toggle("rounded",a),e.classList.toggle("px-1",
a)),i&&(a?(i.title="PromptCache\u6709\u52B9: \u540C\u4E00API\u30D7\u30ED\u30D0\u30A4\u30C0\u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\u629E\u53EF\u80FD",
i.classList.add("border-teal-500/60")):(i.title="",i.classList.remove("border-teal-500/60")))},"upda\
tePromptCacheUi"),bindPromptCacheControls=r(()=>{const e=get("enable-prompt-cache");!e||e.dataset.bound===
"1"||(e.dataset.bound="1",e.addEventListener("change",()=>{if(updatePromptCacheUi(),e.checked){const n=getModelApiProvider(
get("model-select")?get("model-select").value:""),i=PROVIDER_LABELS[n]||n||"\u73FE\u5728\u306EAPI";showToast(
`PromptCache \u3092\u6709\u52B9\u5316\u3057\u307E\u3057\u305F\u3002\u4EE5\u964D\u306F ${i} \u4EE5\u5916\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002`,"info",!0)}}))},"bindPromptCacheControls"),getModelMediaSupport=r(
e=>{const n=(e||"").toLowerCase();return n.includes("gemini")?n.includes("image")||n.includes("nano")||
n.includes("tts")||n.includes("native-audio")||n.includes("live")?{audio:!1,video:!1}:n.includes("em\
bedding")||n.startsWith("veo-")||n.includes("omni-flash")||n.includes("omni-1.1-flash")||n.startsWith(
"lyria-")?{audio:!1,video:!1}:{audio:!0,video:!0}:{audio:!1,video:!1}},"getModelMediaSupport"),supportsAudioInputModel=r(
()=>getModelMediaSupport(get("model-select").value).audio,"supportsAudioInputModel"),supportsVideoInputModel=r(
()=>getModelMediaSupport(get("model-select").value).video,"supportsVideoInputModel"),isImagePath=r(e=>IMAGE_EXTS.
includes(getFileExt(e||"")),"isImagePath"),isAudioPath=r(e=>AUDIO_EXTS.includes(getFileExt(e||"")),"\
isAudioPath"),isVideoPath=r(e=>VIDEO_EXTS.includes(getFileExt(e||"")),"isVideoPath"),OPENAI_TTS_VOICES=[
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
48e3],isTtsModel=r(()=>get("model-select").value.includes("tts"),"isTtsModel"),isGptImageModel=r(()=>(get(
"model-select").value||"").includes("gpt-image"),"isGptImageModel"),isGeminiImageModel=r(()=>isGeminiImageModelKey(
get("model-select").value),"isGeminiImageModel"),isMistralOcrModel=r(e=>{const n=String(e!=null?e:get(
"model-select")&&get("model-select").value||"").toLowerCase();return n==="mistral-ocr-4-0"||n==="mis\
tral-ocr-latest"||n.startsWith("mistral-ocr")},"isMistralOcrModel"),isLlmModel=r(()=>{const e=(get("\
model-select").value||"").toLowerCase();return isMistralOcrModel(e)||e.includes("tts")||e.includes("\
transcribe")||e.includes("realtime")||e.includes("voice-agent")||e.includes("native-audio")||e.includes(
"live")||e.includes("image")||e.includes("video")||isGeminiVideoModelKey(e)||isGeminiMusicModelKey(e)||
isGeminiEmbeddingModelKey(e)||e.includes("gemini")&&(e.includes("image")||e.includes("nano"))?!1:e.includes(
"gpt")||e.includes("gemini")||e.includes("grok")||e.includes("deepseek")||e.startsWith("deep-researc\
h-")||e.startsWith("antigravity-")},"isLlmModel"),isGrokImageModel=r(()=>{const e=(get("model-select").
value||"").toLowerCase();return e.includes("grok")&&(e.includes("imagine")||e.includes("image"))&&!e.
includes("video")},"isGrokImageModel"),isGrokVideoModel=r(()=>{const e=(get("model-select").value||"").
toLowerCase();return e.includes("grok")&&e.includes("video")},"isGrokVideoModel"),isGeminiVideoModelKey=r(
e=>{const n=(e||"").toLowerCase();return n.startsWith("veo-")||n.includes("omni-flash")||n.includes(
"omni-1.1-flash")},"isGeminiVideoModelKey"),isGeminiVideoModel=r(()=>isGeminiVideoModelKey(get("mode\
l-select").value),"isGeminiVideoModel"),isGeminiMusicModelKey=r(e=>(e||"").toLowerCase().startsWith(
"lyria-"),"isGeminiMusicModelKey"),isGeminiMusicModel=r(()=>isGeminiMusicModelKey(get("model-select").
value),"isGeminiMusicModel"),isGeminiEmbeddingModelKey=r(e=>(e||"").toLowerCase().includes("gemini-e\
mbedding"),"isGeminiEmbeddingModelKey"),isGeminiEmbeddingModel=r(()=>isGeminiEmbeddingModelKey(get("\
model-select").value),"isGeminiEmbeddingModel"),isStsModel=r(()=>STS_MODELS.has(get("model-select").
value),"isStsModel"),isTranscriptionModel=r(()=>{const e=get("model-select")?get("model-select").value:
"";return e==="gpt-transcribe"||e==="gpt-live-transcribe"},"isTranscriptionModel"),isGeminiLiveModel=r(
()=>{const e=get("model-select").value;return e==="gemini-3.1-flash-live-preview"||e==="gemini-3.8-l\
ive"||e==="gemini-3.8-live-extended-thinking"||e==="gemini-3.5-live-translate-preview"||e==="gemini-\
3.5-transcribe-live"},"isGeminiLiveModel"),isGeminiLiveExtendedThinkingModel=r(()=>get("model-select").
value==="gemini-3.8-live-extended-thinking","isGeminiLiveExtendedThinkingModel"),isGeminiLiveTranslateModel=r(
()=>get("model-select").value==="gemini-3.5-live-translate-preview","isGeminiLiveTranslateModel"),isGeminiLiveTranscribeModel=r(
()=>get("model-select").value==="gemini-3.5-transcribe-live","isGeminiLiveTranscribeModel"),isGeminiRealtimeMusicModel=r(
()=>(get("model-select").value||"")==="lyria-realtime-exp","isGeminiRealtimeMusicModel"),isLyriaRealtimeModel=r(
()=>isGeminiRealtimeMusicModel(),"isLyriaRealtimeModel"),isRealtimeSessionModel=r(()=>!(!isStsModel()||
isGeminiLiveModel()||isTranscriptionModel()||get("model-select")&&get("model-select").value==="gpt-r\
ealtime-whisper"),"isRealtimeSessionModel"),getStsProvider=r(e=>{const n=(e||"").toLowerCase();return n.
includes("gpt-realtime")||n==="gpt-transcribe"||n==="gpt-live-transcribe"?"openai":n.includes("grok-\
voice")?"xai":n.includes("gemini")&&(n.includes("native-audio")||n.includes("live"))?"gemini":null},
"getStsProvider");function setStsStatus(e,n=!1){const i=get("sts-status"),a=get("sts-mic-btn");i&&e&&
(i.innerText=e),a&&(n?(a.classList.add("bg-red-600","animate-pulse"),a.classList.remove("bg-cyan-600")):
(a.classList.remove("bg-red-600","animate-pulse"),a.classList.add("bg-cyan-600")))}r(setStsStatus,"s\
etStsStatus");function updateStsUi(){const e=isStsModel(),n=e&&voiceStudioUiEnabled!==!1,i=get("inpu\
t-row"),a=get("sts-panel"),o=get("voice-studio-bar"),l=get("file-preview");e?(i&&i.classList.add("hi\
dden"),l&&l.classList.add("hidden"),n?(a&&(window.VoiceStudioOpen?a.classList.remove("hidden"):a.classList.
add("hidden")),o&&o.classList.remove("hidden")):(a&&a.classList.remove("hidden"),o&&o.classList.add(
"hidden"),window.VoiceStudio&&window.VoiceStudio.closeIfOpen()),setStsStatus("Tap to speak",!1)):(i&&
i.classList.remove("hidden"),a&&a.classList.add("hidden"),o&&o.classList.add("hidden"),window.VoiceStudio&&
window.VoiceStudio.closeIfOpen())}r(updateStsUi,"updateStsUi");function updateStsOptions(){if(!isStsModel())
return;const e=get("model-select").value||"",n=getStsProvider(e),i=get("sts-voice"),a=get("sts-speed\
-wrap"),o=get("sts-speed"),l=get("sts-speed-label"),c=get("sts-rate-wrap"),d=get("sts-rate-in"),m=get(
"sts-rate-out"),h=get("sts-thinking-wrap"),y=get("sts-note"),v=get("sts-voice-wrap"),x=get("sts-auto\
-play-wrap"),w=get("sts-mode-label"),_=isTranscriptionModel()||isGeminiLiveTranscribeModel(),S=get("\
sts-lang-wrap");if(_){w&&(w.textContent="Realtime Speech-to-Text"),v&&v.classList.add("hidden"),x&&x.
classList.add("hidden"),a&&a.classList.add("hidden"),c&&c.classList.add("hidden"),h&&h.classList.add(
"hidden"),S&&S.classList.add("hidden");const L=get("sts-transcribe-wrap"),M=get("sts-custom-vocab-wr\
ap");L&&L.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),M&&M.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),
y&&(y.textContent=isGeminiLiveTranscribeModel()?"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F4E\u9045\u5EF6\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM / \u6700\u592710\u5206\uFF09":
e==="gpt-live-transcribe"?"\u4F4E\u9045\u5EF6\u30E9\u30A4\u30D6\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
"\u9AD8\u7CBE\u5EA6\u306A\u30B3\u30DF\u30C3\u30C8\u5358\u4F4D\u306E\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09")}else if(n===
"openai")w&&(w.textContent="Speech-to-Speech Live"),v&&v.classList.remove("hidden"),x&&x.classList.remove(
"hidden"),setSelectOptions(i,OPENAI_STS_VOICES,i.value||"alloy"),a&&a.classList.remove("hidden"),o&&
(o.min=.25,o.max=1.5,o.step=.05,o.value||(o.value=1),Number(o.value)<.25&&(o.value=.25),Number(o.value)>
1.5&&(o.value=1.5)),c&&c.classList.add("hidden"),h&&h.classList.add("hidden"),S&&S.classList.add("hi\
dden"),y&&(y.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A");else if(n==="xai")w&&(w.textContent=
"Speech-to-Speech Live"),v&&v.classList.remove("hidden"),x&&x.classList.remove("hidden"),setSelectOptions(
i,GROK_STS_VOICES,i.value||"Ara"),a&&a.classList.add("hidden"),c&&c.classList.remove("hidden"),h&&h.
classList.add("hidden"),S&&S.classList.add("hidden"),setSelectOptions(d,GROK_PCM_RATES,Number(d.value||
24e3)),setSelectOptions(m,GROK_PCM_RATES,Number(m.value||24e3)),y&&(y.textContent="xAI\u306FPCM\u30B5\u30F3\u30D7\u30EB\u30EC\u30FC\u30C8\u5909\u66F4\u53EF");else if(n===
"gemini"){if(w&&(w.textContent="Speech-to-Speech Live"),v&&v.classList.remove("hidden"),x&&x.classList.
remove("hidden"),setSelectOptions(i,GEMINI_STS_VOICES,i.value||"Kore"),a&&a.classList.add("hidden"),
c&&c.classList.add("hidden"),h&&h.classList.remove("hidden"),S&&S.classList.add("hidden"),y&&(y.textContent=
"Gemini Live\u306F\u97F3\u58F0\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC"),e==="gemini-3.8-live")h&&
h.classList.add("hidden"),y&&(y.textContent="Gemini 3.8 Flash Live\u306F\u56FA\u5B9A\u30EC\u30A4\u30C6\u30F3\u30B7\u306ELive API\u30E2\u30C7\u30EB\uFF08Thinking leve\
l\u975E\u5BFE\u5FDC\uFF09");else if(e==="gemini-3.8-live-extended-thinking"){y&&(y.textContent="Gemi\
ni 3.8 Live Extended Thinking\u306Flow / medium / high\u306E\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u63A8\u8AD6\u306B\u5BFE\u5FDC");
const L=get("sts-thinking-level");Array.from(L&&L.options||[]).forEach(M=>{M.disabled=M.value==="min\
imal"}),L&&!["low","medium","high"].includes(L.value)&&(L.value="medium")}e==="gemini-3.5-live-trans\
late-preview"&&(w&&(w.textContent="Realtime Translation"),h&&h.classList.add("hidden"),v&&v.classList.
add("hidden"),S&&S.classList.remove("hidden"),y&&(y.textContent="70\u4EE5\u4E0A\u306E\u8A00\u8A9E\u306B\u5BFE\u5FDC\u3059\u308B\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\uFF08Think\u975E\u5BFE\u5FDC\u30FB\u97F3\u58F0\u9078\
\u629E\u4E0D\u53EF\uFF09"))}a&&l&&o&&!a.classList.contains("hidden")&&(l.textContent=`${Number(o.value||
1).toFixed(2)}x`)}r(updateStsOptions,"updateStsOptions");function stsOpt(e){const n=get(e);return e===
"sts-auto-play"||e==="sts-auto-restart"?n?!!n.checked:!0:n?!!n.checked:!1}r(stsOpt,"stsOpt");function getStsSilenceMs(){
const e=get("sts-silence-sec");let n=e?parseFloat(e.value):1.5;return(isNaN(n)||n<.5)&&(n=.5),n>10&&
(n=10),Math.round(n*1e3)}r(getStsSilenceMs,"getStsSilenceMs");function getTtsProvider(e){if(!e)return null;
const n=e.toLowerCase();return n.includes("google-tts")?"google":n.includes("gemini")&&n.includes("t\
ts")?"gemini":n.includes("grok-tts")||n.includes("xai-tts")?"xai":n.includes("tts")?"openai":null}r(
getTtsProvider,"getTtsProvider");function setSelectOptions(e,n,i){e&&(e.innerHTML="",n.forEach(a=>{const o=document.
createElement("option");o.value=a.value||a,o.textContent=a.label||a,(a.value||a)===i&&(o.selected=!0),
e.appendChild(o)}))}r(setSelectOptions,"setSelectOptions");function updateTtsUi(){const e=get("model\
-select").value||"",n=getTtsProvider(e),i=get("audio-gen-options");if(!i)return;if(!n){i.classList.add(
"hidden");return}i.classList.remove("hidden");const a=get("tts-voice"),o=get("tts-voice-custom-wrap"),
l=get("tts-voice-custom"),c=get("tts-language-wrap"),d=get("tts-language"),m=get("tts-speed-wrap"),h=get(
"tts-speed"),y=get("tts-speed-label"),v=get("tts-speed-note");n==="openai"?(setSelectOptions(a,OPENAI_TTS_VOICES,
a.value||"alloy"),o.classList.add("hidden"),c.classList.add("hidden"),h&&(h.min=.25,h.max=4,h.step=.05,
h.value||(h.value=1),Number(h.value)<.25&&(h.value=.25),Number(h.value)>4&&(h.value=4),h.disabled=!1),
v&&(v.textContent="")):n==="gemini"?(setSelectOptions(a,GEMINI_TTS_VOICES,a.value||"Kore"),o.classList.
add("hidden"),c.classList.add("hidden"),h&&(h.disabled=!0),v&&(v.textContent="(Gemini TTS\u306F\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC)")):
n==="google"?(setSelectOptions(a,[{value:"auto",label:"Auto (Studio/Neural2)"},{value:"custom",label:"\
Custom Voice Name"}],a.value||"auto"),a.value==="custom"?o.classList.remove("hidden"):(o.classList.add(
"hidden"),l&&(l.value="")),c.classList.remove("hidden"),d&&!d.value&&(d.value="ja-JP"),h&&(h.min=.25,
h.max=2,h.step=.05,h.value||(h.value=1),Number(h.value)<.25&&(h.value=.25),Number(h.value)>2&&(h.value=
2),h.disabled=!1),v&&(v.textContent="")):n==="xai"&&(setSelectOptions(a,GROK_TTS_VOICES,a.value||"Ev\
e"),o.classList.remove("hidden"),c.classList.remove("hidden"),d&&!d.value&&(d.value="ja"),h&&(h.min=
.7,h.max=1.5,h.step=.05,h.value||(h.value=1),Number(h.value)<.7&&(h.value=.7),Number(h.value)>1.5&&(h.
value=1.5),h.disabled=!1),v&&(v.textContent="xAI TTS supports speed 0.7\u20131.5 and speech tags")),
h&&y&&(y.textContent=`${Number(h.value||1).toFixed(2)}x`)}r(updateTtsUi,"updateTtsUi");let mcpServers=[],
mcpLoaded=!1,mcpLoadPromise=null,mcpOauthPopups=[];const MCP_URLS={servers:r(()=>"/api/mcp/servers",
"servers"),server:r(e=>`/api/mcp/servers/${encodeURIComponent(e)}`,"server"),test:r(e=>`/api/mcp/ser\
vers/${encodeURIComponent(e)}/test`,"test"),authStart:r(e=>`/api/mcp/servers/${encodeURIComponent(e)}\
/auth/start`,"authStart"),authDisconnect:r(e=>`/api/mcp/servers/${encodeURIComponent(e)}/auth/discon\
nect`,"authDisconnect"),tools:r(e=>`/api/mcp/servers/${encodeURIComponent(e)}/tools`,"tools"),oauthClient:r(
()=>"/api/mcp/oauth-client","oauthClient"),permission:r((e,n)=>`/api/mcp/servers/${encodeURIComponent(
e)}/tools/${encodeURIComponent(n)}/permission`,"permission")},mcpGoogleProviderKey="google_workspace",
mcpEsc=r(e=>String(e==null?"":e).replace(/[&<>"']/g,n=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quo\
t;","'":"&#39;"})[n]),"mcpEsc"),mcpStatusMsg=r((e,n,i)=>{const a=get(e);a&&(a.textContent=n||"",a.style.
color=i?"#f87171":"#9ca3af")},"mcpStatusMsg");function mcpAuthStatusLabel(e){return e.auth_type==="n\
one"?"\u8A8D\u8A3C\u4E0D\u8981":e.auth_status==="connected"?"\u63A5\u7D9A\u6E08\u307F":e.auth_status===
"expired"?"\u671F\u9650\u5207\u308C\uFF08\u518D\u8A8D\u8A3C\uFF09":e.auth_status==="needs_auth"?"\u8A8D\u8A3C\u304C\
\u5FC5\u8981":"\u672A\u8A8D\u8A3C"}r(mcpAuthStatusLabel,"mcpAuthStatusLabel");function mcpConnectionStateLabel(e){
return e.connection_state==="error"?"\u30A8\u30E9\u30FC":e.connection_state==="connected"?"\u63A5\u7D9AOK":
e.connection_state==="needs_auth"?"\u8A8D\u8A3C\u5F85\u3061":"\u672A\u63A5\u7D9A"}r(mcpConnectionStateLabel,
"mcpConnectionStateLabel");function mcpBadgeClass(e){return e==="ok"||e==="connected"?"bg-emerald-70\
0/60 text-emerald-100":e==="error"||e==="expired"?"bg-red-700/60 text-red-100":e==="auth"?"bg-amber-\
600/50 text-amber-100":"bg-gray-700 text-gray-300"}r(mcpBadgeClass,"mcpBadgeClass");function mcpStateBadge(e){
const n=mcpAuthStatusLabel(e),i=e.auth_status==="connected"?"ok":e.auth_status==="expired"?"expired":
e.auth_status==="needs_auth"?"auth":"neutral";return`<span class="text-[9px] font-bold px-2 py-0.5 r\
ounded-full ${mcpBadgeClass(i)}">${mcpEsc(n)}</span>`}r(mcpStateBadge,"mcpStateBadge");function mcpOauthProviderLabel(e){
return e==="google_workspace"?"Google Workspace":e||"OAuth"}r(mcpOauthProviderLabel,"mcpOauthProvide\
rLabel");async function loadMcpServers(e){if(!get("mcp-server-list")||mcpLoadPromise&&(await mcpLoadPromise,
!e))return;if(!e&&mcpLoaded){renderMcpServers();return}mcpStatusMsg("mcp-status-msg","\u8AAD\u307F\u8FBC\u307F\u4E2D...",
!1);let i;i=(async()=>{try{const a=await apiFetch(MCP_URLS.servers());if(!a.ok){const l=await a.json().
catch(()=>({}));mcpStatusMsg("mcp-status-msg",l.error||"MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const o=await a.json();mcpServers=o&&Array.isArray(o.servers)?o.servers:[],mcpLoaded=!0,renderMcpServers(),
applyMcpPromptChipUi()}catch(a){mcpStatusMsg("mcp-status-msg","MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(a&&a.message?a.message:a),!0)}finally{mcpLoadPromise===i&&(mcpLoadPromise=null)}})(),mcpLoadPromise=
i,await i}r(loadMcpServers,"loadMcpServers");function mcpHasEnabledServer(){return(mcpServers||[]).some(
e=>!!e.enabled)}r(mcpHasEnabledServer,"mcpHasEnabledServer");function isMcpEnabledForSend(){const e=get(
"mcp-container");if(!e||e.classList.contains("hidden"))return!1;const n=get("enable-mcp");return!!n&&
n.checked}r(isMcpEnabledForSend,"isMcpEnabledForSend");function mcpModelSupported(){try{const e=String(
get("model-select")&&get("model-select").value||"").toLowerCase();return e?!!(e.includes("claude")||
e.startsWith("kimi")||typeof isLlmModel=="function"&&isLlmModel()):!1}catch{return!1}}r(mcpModelSupported,
"mcpModelSupported");function applyMcpPromptChipUi(){const e=get("mcp-container");if(!e)return;const n=mcpModelSupported()&&
mcpHasEnabledServer();if(e.classList.toggle("hidden",!n),syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}}r(applyMcpPromptChipUi,"applyMcpPromptChipUi");
function syncMcpAutoSysRows(){["set","thread"].forEach(e=>{const n=get(`${e}-auto-sys-mcp-enabled`);
n&&(n.disabled=!0,n.checked=isMcpEnabledForSend())})}r(syncMcpAutoSysRows,"syncMcpAutoSysRows");function renderMcpServers(){
const e=get("mcp-server-list"),n=get("mcp-server-count");if(!e)return;const i=mcpServers.length;if(n&&
(n.textContent=`${i}\u4EF6`),!i){e.innerHTML='<div class="text-[11px] text-gray-600 py-2">\u307E\u3060\u30B5\u30FC\u30D0\u30FC\u304C\u3042\u308A\u307E\
\u305B\u3093\u3002\u4E0A\u306E\u30AB\u30B9\u30BF\u30E0\u8FFD\u52A0\u30D5\u30A9\u30FC\u30E0\u304B\u3089\u767B\u9332\u3059\u308B\u304B\u3001Google Workspace \u306E\u8A8D\u8A3C\u3092\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>',
mcpStatusMsg("mcp-status-msg","");return}const a=mcpServers.map((o,l)=>mcpServerCard(o,l)).join("");
e.innerHTML=a,mcpStatusMsg("mcp-status-msg","")}r(renderMcpServers,"renderMcpServers");function mcpServerCard(e,n){
const i=!!e.is_preset,a=e.auth_type==="oauth",o=e.auth_type==="bearer",l=a||o,c=a&&!e.oauth_client_registered,
d=Number(e.tool_count||0),m=d>0?`${d}\u30C4\u30FC\u30EB`:"\u30C4\u30FC\u30EB\u672A\u53D6\u5F97",h=mcpStateBadge(
e),y=i?'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">\u30D7\u30EA\u30BB\u30C3\u30C8<\
/span>':'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">\u30AB\
\u30B9\u30BF\u30E0</span>',v=mcpAuthBlock(e),x=a?mcpOauthClientBlock(e):"";return`
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(e.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${e.enabled?"text-cyan-300":"text-gray-600"}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(e.name)}</span>
                ${y} ${h}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${l?mcpAuthActionButton(e):""}
            ${o&&e.auth_status!=="connected",""}
            ${i?"":`<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-da\
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
            <span class="${d>0?"text-emerald-300":"text-gray-500"}">${m}</span>
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="too\
ls" data-id="${e.id}">\u30C4\u30FC\u30EB\u4E00\u89A7</button>
        </div>
        <div class="flex items-center gap-1 flex-wrap">
            <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="tes\
t" data-id="${e.id}"><i class="fas fa-plug"></i> \u63A5\u7D9A\u30C6\u30B9\u30C8</button>
            <span class="text-[9px] text-gray-600">${mcpEsc(mcpConnectionStateLabel(e))}</span>
        </div>
    </div>
    ${d>0?`<div class="hidden mt-2" data-mcp-toolbox="${e.id}"></div>`:`<div class="hidden mt-2" dat\
a-mcp-toolbox="${e.id}"><div class="text-[10px] text-gray-600">\u63A5\u7D9A\u30C6\u30B9\u30C8\u5F8C\u306B\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u8868\u793A\u3055\u308C\u307E\u3059\u3002</div></div>`}\

    ${x}
    ${v}
</div>`}r(mcpServerCard,"mcpServerCard");function mcpAuthActionButton(e){return e.auth_type==="beare\
r"?"":e.auth_status==="connected"||e.auth_status==="expired"?`<button type="button" data-progress-no\
-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="reconnect" data-id="${e.id}"><i class="f\
as fa-sync"></i> \u518D\u8A8D\u8A3C</button>
                        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mc\
p-danger-btn" data-act="disconnect" data-id="${e.id}"><i class="fas fa-unlink"></i> \u63A5\u7D9A\u89E3\u9664</button>`:
`<button type="button" data-progress-no-spinner="true" class="mcp-mini-btn mcp-auth-btn" data-act="a\
uth" data-id="${e.id}"><i class="fas fa-key"></i> \u8A8D\u8A3C\u3059\u308B</button>`}r(mcpAuthActionButton,
"mcpAuthActionButton");function mcpOauthClientBlock(e){const n=e.oauth_provider_key||e.slug||"",i=mcpOauthProviderLabel(
n);return e.oauth_client_registered?`
<div class="mt-2 rounded border border-gray-800 bg-black/20 p-2">
    <div class="text-[10px] text-gray-400 flex items-center justify-between">
        <span>OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\uFF08${mcpEsc(i)}\uFF09: ${mcpEsc(e.oauth_client_id_masked||
"\u767B\u9332\u6E08\u307F")}</span>
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="edit-oa\
uth" data-id="${e.id}">\u5909\u66F4</button>
    </div>
</div>`:`
<div class="mt-2 rounded border border-amber-700/50 bg-amber-950/20 p-2">
    <div class="text-[10px] text-amber-300 mb-1">${mcpEsc(i)} \u306E OAuth \u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\uFF08Client ID / Secret\uFF09\u304C\u5FC5\
\u8981\u3067\u3059\u3002</div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-1">
        <input type="text" data-oauth-pk="${mcpEsc(n)}" data-oauth-role="cid" placeholder="Client ID\
" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-700 rounded \
px-2 py-1 text-xs text-white">
        <input type="password" data-oauth-pk="${mcpEsc(n)}" data-oauth-role="secret" placeholder="Cl\
ient Secret" autocomplete="off" data-1p-ignore="true" class="w-full bg-gray-800 border border-gray-7\
00 rounded px-2 py-1 text-xs text-white">
    </div>
    <div class="flex justify-end mt-1">
        <button type="button" data-progress-no-spinner="true" class="mcp-mini-btn" data-act="save-oa\
uth" data-id="${e.id}" data-pk="${mcpEsc(n)}">\u4FDD\u5B58</button>
    </div>
</div>`}r(mcpOauthClientBlock,"mcpOauthClientBlock");function mcpAuthBlock(e){if(e.auth_type==="bear\
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
</div>`;if(e.auth_type==="oauth"){const n=e.oauth_provider_key||e.slug||"";return`
<div class="text-[10px] text-gray-600 mt-1">${!e.oauth_client_registered?"OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3059\u308B\u3068\u300C\u8A8D\u8A3C\u3059\u308B\u300D\u304C\
\u4F7F\u3048\u307E\u3059\u3002":""}</div>`}return""}r(mcpAuthBlock,"mcpAuthBlock");async function mcpToggleEnabled(e,n){
mcpStatusMsg("mcp-status-msg",n?"\u6709\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...":"\u7121\u52B9\u5316\u3057\u3066\u3044\u307E\u3059...",
!1);try{const i=await apiFetch(MCP_URLS.server(e),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({enabled:n})});if(!i.ok){const o=await i.json().catch(()=>({}));mcpStatusMsg(
"mcp-status-msg",o.error||"\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}const a=await i.
json();mcpStatusMsg("mcp-status-msg",n?"\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u30C1\u30E3\u30C3\u30C8\u306E\u30E2\u30C7\u30EB\u3078\u30C4\u30FC\u30EB\u304C\u516C\u958B\u3055\u308C\u307E\u3059\u3002":
"\u7121\u52B9\u306B\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(i){mcpStatusMsg("mcp\
-status-msg","\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(i&&i.message?i.message:i),!0)}}
r(mcpToggleEnabled,"mcpToggleEnabled");async function mcpOpenAuth(e){mcpStatusMsg("mcp-status-msg","\
\u8A8D\u53EFURL\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059...",!1);try{const n=await apiFetch(MCP_URLS.
authStart(e),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})});if(!n.
ok){const o=await n.json().catch(()=>({}));o.requires_oauth_client?mcpStatusMsg("mcp-status-msg",o.error||
"OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u5148\u306B\u767B\u9332\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
!0):mcpStatusMsg("mcp-status-msg",o.error||"\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const i=await n.json();if(!i.url){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u304C\u8FD4\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
!0);return}const a=window.open(i.url,"_blank","width=520,height=680");if(a){mcpOauthPopups.push(a),mcpStatusMsg(
"mcp-status-msg","Google\u306E\u753B\u9762\u3067\u8A31\u53EF\u3057\u3066\u304F\u3060\u3055\u3044\u3002\u5B8C\u4E86\u5F8C\u3053\u306E\u30BF\u30D6\u306B\u53CD\u6620\u3055\u308C\u307E\u3059\u3002",
!1);const o=window.setInterval(()=>{(!a||a.closed)&&(window.clearInterval(o),loadMcpServers(!0))},1200)}else
mcpStatusMsg("mcp-status-msg","\u30DD\u30C3\u30D7\u30A2\u30C3\u30D7\u304C\u30D6\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F\u3002",
!0)}catch(n){mcpStatusMsg("mcp-status-msg","\u8A8D\u53EFURL\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(n&&n.message?n.message:n),!0)}}r(mcpOpenAuth,"mcpOpenAuth");async function mcpDisconnect(e){if(window.
confirm("\u3053\u306E\u30B5\u30FC\u30D0\u30FC\u306E\u8A8D\u8A3C\u60C5\u5831\uFF08\u30C8\u30FC\u30AF\u30F3\uFF09\u3092\u524A\u9664\u3057\u3066\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const n=await apiFetch(MCP_URLS.authDisconnect(e),{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:"{}"});if(!n.ok){const i=await n.json().catch(()=>({}));mcpStatusMsg("mcp-status-m\
sg",i.error||"\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg(
"mcp-status-msg","\u63A5\u7D9A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(n){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(n&&n.message?n.message:n),!0)}}r(mcpDisconnect,"mcpDisconnect");async function mcpDeleteServer(e){if(window.
confirm("\u3053\u306E\u30AB\u30B9\u30BF\u30E0MCP\u30B5\u30FC\u30D0\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{const n=await apiFetch(MCP_URLS.server(e),{method:"DELETE"});if(!n.ok){const i=await n.json().catch(
()=>({}));mcpStatusMsg("mcp-status-msg",i.error||"\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","\u524A\u9664\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(
!0)}catch(n){mcpStatusMsg("mcp-status-msg","\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(n&&n.message?n.message:n),!0)}}r(mcpDeleteServer,"mcpDeleteServer");async function mcpTestServer(e,n){
mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",!1);try{const i=await apiFetch(
MCP_URLS.test(e),{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}),a=await i.json().
catch(()=>({}));if(!i.ok){mcpStatusMsg("mcp-status-msg",a.error||"\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}a.probe&&a.probe.message&&mcpStatusMsg("mcp-status-msg",a.probe.message,!a.probe.ok),loadMcpServers(
!0)}catch(i){mcpStatusMsg("mcp-status-msg","\u63A5\u7D9A\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(i&&i.message?i.message:i),!0)}}r(mcpTestServer,"mcpTestServer");async function mcpLoadTools(e){const n=document.
querySelector(`[data-mcp-toolbox="${e}"]`);if(n){n.classList.remove("hidden"),n.innerHTML='<div clas\
s="text-[10px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const i=await apiFetch(MCP_URLS.
tools(e)),a=await i.json().catch(()=>({}));if(!i.ok){n.innerHTML=`<div class="text-[10px] text-red-4\
00">${mcpEsc(a.error||"\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}</div>`;return}const o=a&&
Array.isArray(a.tools)?a.tools:[];if(!o.length){n.innerHTML='<div class="text-[10px] text-gray-600">\
\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u300C\u63A5\u7D9A\u30C6\u30B9\u30C8\u300D\u3067\u53D6\u5F97\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>';
return}const l=o.map((c,d)=>`
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(c.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(c.description||"")}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${c.read_only?"bg-emerald-800/40 text-eme\
rald-200":"bg-amber-800/40 text-amber-200"}">${c.read_only?"\u8AAD\u307F\u53D6\u308A":"\u5909\u66F4"}\
</span>
</div>`).join("");n.innerHTML=`<div class="rounded border border-gray-800 bg-black/20 p-2">${l}</div\
>`}catch{n.innerHTML='<div class="text-[10px] text-red-400">\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}}
r(mcpLoadTools,"mcpLoadTools");async function mcpSaveOauthClient(e,n,i,a){mcpStatusMsg("mcp-status-m\
sg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);const o={provider_key:e,client_id:n,client_secret:i};
try{const l=await apiFetch(MCP_URLS.oauthClient(),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(o)}),c=await l.json().catch(()=>({}));if(!l.ok){mcpStatusMsg("mcp-status\
-msg",c.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg("mcp\
-status-msg","OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch(l){mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(l&&l.message?l.message:l),!0)}}r(mcpSaveOauthClient,"mcpSaveOauthClient");async function mcpAddCustomServer(){
const e=get("mcp-custom-name"),n=get("mcp-custom-url"),i=get("mcp-custom-auth"),a=get("mcp-custom-de\
sc"),o=get("mcp-custom-bearer"),l=get("mcp-custom-status"),c=get("mcp-add-server-btn");if(!e||!n||!i)
return;const d=(e.value||"").trim(),m=(n.value||"").trim(),h=i.value||"none",y=a?(a.value||"").trim():
"",v=(o&&o.value||"").trim();if(!d||!m){l&&(l.textContent="\u8868\u793A\u540D\u3068URL\u306F\u5FC5\u9808\u3067\u3059",
l.style.color="#f87171");return}c&&(c.disabled=!0),l&&(l.textContent="\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",
l.style.color="#9ca3af");const x={name:d,url:m,auth_type:h,description:y};h==="bearer"&&v&&(x.bearer_token=
v);try{const w=await apiFetch(MCP_URLS.servers(),{method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(x)}),_=await w.json().catch(()=>({}));if(!w.ok){l&&(l.textContent=_.error||
"\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",l.style.color="#f87171");return}l&&(l.textContent=
_.probe&&_.probe.message||"\u8FFD\u52A0\u3057\u307E\u3057\u305F",l.style.color=_.probe&&_.probe.ok?"\
#34d399":"#fbbf24"),e.value="",n.value="",a&&(a.value=""),o&&(o.value=""),mcpLoaded=!1,loadMcpServers(
!0)}catch(w){l&&(l.textContent="\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(w&&w.message?
w.message:w),l.style.color="#f87171")}finally{c&&(c.disabled=!1)}}r(mcpAddCustomServer,"mcpAddCustom\
Server");function bindMcpSettingsUi(){const e=get("mcp-server-list");if(!e)return;const n=get("mcp-a\
dd-server-btn");n&&n.addEventListener("click",mcpAddCustomServer);const i=get("mcp-custom-auth"),a=get(
"mcp-custom-bearer-wrap");if(i&&a){const l=r(()=>{a.classList.toggle("hidden",i.value!=="bearer")},"\
syncBearer");i.addEventListener("change",l),l()}const o=get("mcp-save-google-client-btn");o&&o.addEventListener(
"click",async()=>{const l=get("mcp-google-client-id"),c=get("mcp-google-client-secret"),d=get("mcp-g\
oogle-client-state"),m=l?l.value:"",h=c?c.value:"";if(!m&&!h){d&&(d.textContent="Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
d.style.color="#f87171");return}await mcpSaveOauthClient(mcpGoogleProviderKey,m||"********",h||"****\
****",null)}),e.addEventListener("click",async l=>{const c=l.target.closest("[data-act]");if(!c)return;
const d=c.getAttribute("data-act"),m=c.getAttribute("data-id");if(d==="test"){l.preventDefault(),mcpTestServer(
m);return}if(d==="tools"){l.preventDefault(),mcpLoadTools(m);return}if(d==="auth"||d==="reconnect"){
l.preventDefault(),mcpOpenAuth(m);return}if(d==="disconnect"){l.preventDefault(),mcpDisconnect(m);return}
if(d==="delete"){l.preventDefault(),mcpDeleteServer(m);return}if(d==="edit-oauth"){if(l.preventDefault(),
c.closest("[data-mcp-server]")){const y=c.getAttribute("data-oauth-pk")||"",v=mcpServers.find(_=>String(
_.id)===String(m)),x=document.createElement("div");x.className="mt-2 rounded border border-amber-700\
/50 bg-amber-950/20 p-2",x.innerHTML=`
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
uth" data-id="${m}" data-pk="${mcpEsc(v&&(v.oauth_provider_key||v.slug)||"")}">\u4FDD\u5B58</button>
    </div>`;const w=c.closest("div");w.parentNode.insertBefore(x,w.nextSibling),c.remove()}return}if(d===
"save-oauth"){l.preventDefault();const h=c.getAttribute("data-pk")||"",y=c.closest("[data-mcp-server\
]")||document,v=y.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid'),x=y.querySelectorAll(
'[data-oauth-role="secret"], .mcp-oauth-edit-sec'),w=v.length?v[v.length-1].value:"",_=x.length?x[x.
length-1].value:"";if(!w&&!_){mcpStatusMsg("mcp-status-msg","Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpSaveOauthClient(h,w||"********",_||"********",m);return}if(d==="save-bearer"){l.preventDefault();
const h=document.querySelector(`[data-bearer-id="${m}"]`),y=h?h.value:"";if(!y||y.trim()===""){mcpStatusMsg(
"mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);try{const v=await apiFetch(
MCP_URLS.server(m),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({bearer_token:y})}),
x=await v.json().catch(()=>({}));if(!v.ok){mcpStatusMsg("mcp-status-msg",x.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch{mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}return}}),e.addEventListener("change",l=>{const c=l.target.closest(".mcp-enable-toggle");c&&mcpToggleEnabled(
c.getAttribute("data-id"),c.checked)})}r(bindMcpSettingsUi,"bindMcpSettingsUi");const initMcpUi=r(()=>{
try{bindMcpSettingsUi()}catch{}try{loadMcpServers()}catch{}},"initMcpUi");document.readyState==="loa\
ding"?document.addEventListener("DOMContentLoaded",initMcpUi,{once:!0}):initMcpUi();function bindMcpPromptToggle(){
const e=get("enable-mcp");e&&e.addEventListener("change",()=>{if(syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}})}if(r(bindMcpPromptToggle,"bindMcpPromptToggle"),
document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>{try{bindMcpPromptToggle()}catch{}});else
try{bindMcpPromptToggle()}catch{}function getModelTags(e,n){const i=[],a=(e.id||"").toLowerCase(),o=(e.
name||"").toLowerCase(),l=(e.desc||"").toLowerCase(),c=(n.category||"").toLowerCase();return(c.includes(
"gemini")||a.includes("gemini")||o.includes("gemini")||l.includes("gemini")||c.includes("banana")||o.
includes("banana"))&&i.push("gemini"),(c.includes("deepseek")||a.includes("deepseek")||o.includes("d\
eepseek")||l.includes("deepseek"))&&i.push("deepseek"),(c.includes("mistral")||a.includes("mistral")||
o.includes("mistral")||l.includes("mistral")||a.includes("ocr")||c.includes("ocr"))&&i.push("mistral"),
(c.includes("gpt")||c.includes("openai")||a.includes("gpt")||o.includes("gpt")||l.includes("openai"))&&
i.push("openai"),(c.includes("xai")||c.includes("grok")||a.includes("grok")||o.includes("grok")||l.includes(
"xai"))&&i.push("xai"),(c.includes("image")||a.includes("image")||o.includes("image")||l.includes("i\
mage"))&&i.push("image"),(c.includes("audio")||c.includes("music")||c.includes("transcription")||c.includes(
"speech")||a.includes("tts")||a.includes("transcri")||o.includes("tts")||o.includes("transcri")||o.includes(
"voice")||l.includes("tts")||a.includes("realtime")||a.includes("live")||a.includes("voice-agent")||
a.includes("native-audio")||o.includes("audio")||l.includes("audio")||l.includes("speech-to-text"))&&
i.push("audio"),(a.includes("reasoning")||o.includes("reasoning")||l.includes("reasoning"))&&i.push(
"reasoning"),(c.includes("deepseek")||a.includes("deepseek")||o.includes("deepseek"))&&!i.includes("\
reasoning")&&i.push("reasoning"),(a.includes("fast")||o.includes("fast")||l.includes("fast")||c.includes(
"fast"))&&i.push("fast"),(a.includes("deepseek-v4-flash")||c.includes("deepseek")&&o.includes("flash"))&&
!i.includes("fast")&&i.push("fast"),(c.includes("anthropic")||a.includes("claude")||o.includes("clau\
de")||l.includes("anthropic"))&&i.push("anthropic"),(c.includes("kimi")||a.includes("kimi")||o.includes(
"kimi")||l.includes("moonshot"))&&i.push("kimi"),(c.includes("video")||a.includes("video")||a.startsWith(
"veo-")||a.includes("omni-")||o.includes("video")||l.includes("video"))&&i.push("video"),(c.includes(
"music")||a.startsWith("lyria-")||o.includes("music")||l.includes("music")||l.includes("song"))&&i.push(
"music"),(c.includes("transcription")||a.includes("transcri")||o.includes("transcri")||l.includes("t\
ranscription")||l.includes("speech-to-text"))&&i.push("transcription"),(c.includes("ocr")||a.includes(
"ocr")||o.includes("ocr")||l.includes("ocr"))&&i.push("ocr"),(c.includes("agent")||a.includes("agent")||
o.includes("agent")||l.includes("agentic")||l.includes("computer use")||l.includes("deep research"))&&
i.push("agent"),e.agenticView&&i.push("agentic view"),i}r(getModelTags,"getModelTags");function updateModelTagUi(){
const e=get("model-tag-bar");if(!e)return;e.querySelectorAll(".model-tag-btn").forEach(i=>{const a=i.
innerText.trim().toLowerCase(),o=(a==="all"?"all":a)===activeModelTag;i.classList.toggle("is-active",
o)})}r(updateModelTagUi,"updateModelTagUi");function getModelCapabilitySearchTerms(e){const n=String(
e.id||"").toLowerCase(),i=[],a=n.includes("deepseek"),o=n.includes("tts"),l=n.startsWith("mistral-oc\
r"),c=o||l||n.includes("transcribe")||n.includes("realtime")||n.includes("voice-agent")||n.includes(
"native-audio")||n.includes("live")||n.includes("image")||n.includes("video")||n.startsWith("veo-")||
n.includes("omni-flash")||n.startsWith("lyria-")||n.includes("embedding"),d=!c&&(n.includes("gpt")||
n.includes("gemini")||n.includes("grok")||a||n.startsWith("deep-research-")||n.startsWith("antigravi\
ty-")),m=r((...y)=>y.forEach(v=>i.push(v,v.replace(/-/g," "))),"add");if((n.includes("gemini-3.1-fla\
sh-image")||n.includes("gemini-3-pro-image")||n.includes("gemini-2.5-flash-image"))&&m("image genera\
tion","image editing"),n==="gemini-3.1-flash-lite-image"||n==="gemini-3.1-flash-image"?m("thinking",
"\u601D\u8003","minimal","high","thinking level"):n.includes("gemini")&&!c&&(m("thinking","\u601D\u8003",
"thinking level"),n==="gemini-3.8-flash"||n==="gemini-3.7-flash"?m("low","medium","high"):n==="gemin\
i-3.6-flash"?m("medium","high"):n==="gemini-3.5-flash-lite"?m("minimal","medium","high"):n.includes(
"flash")?m("minimal","low","medium","high"):m("low","high")),a&&(m("thinking","\u601D\u8003","reason\
ing","\u63A8\u8AD6","reasoning effort","high"),n!=="deepseek-v4-pro"&&m("low"),n.includes("v4-flash")&&
m("none","max")),d&&(n.includes("gpt-5")||n.includes("o1")||n.includes("o3")||n.includes("grok-4.3")||
n.includes("grok-4.5")||n.includes("grok-4.6")||n.includes("grok-4.20-0309-reasoning")||n.includes("\
grok-build")||n.includes("multi-agent")||n.includes("gpt")&&!o)){m("reasoning","\u63A8\u8AD6","reaso\
ning effort","low","high");const y=n==="gpt-5.6"||n.startsWith("gpt-5.6-"),v=n.includes("grok-4.6"),
x=n.includes("grok-4.3")||n.includes("grok-4.5")||v||n.includes("grok-4.20-0309-reasoning")||n.includes(
"grok-build")||n.includes("multi-agent")||n.includes("gpt-5")||n.includes("o1")||n.includes("o3"),w=n.
includes("grok-4.3")||n.includes("grok-build")||n.includes("gpt-5")||a;x&&m("medium"),w&&m("none"),(y||
a)&&m("max"),(v||n.includes("multi-agent")||y)&&m("xhigh")}return n.includes("claude")&&m("thinking",
"\u601D\u8003","thinking budget","budget"),e.agenticView&&m("agentic view"),[...new Set(i)]}r(getModelCapabilitySearchTerms,
"getModelCapabilitySearchTerms");const modelListGroups=[];let modelListBanner=null,modelListEmpty=null,
modelListBuilt=!1,modelListAnimated=!1,modelListRenderFrame=0;function buildModelList(){const e=get(
"model-list-container");!e||modelListBuilt||(e.innerHTML="",modelListBanner=document.createElement("\
div"),modelListBanner.className="model-banner hidden",e.appendChild(modelListBanner),MODELS.forEach(
n=>{const i=n.items.filter(c=>!c.deprecated);if(!i.length)return;const a=document.createElement("sec\
tion");a.className="model-list-group",a.innerHTML=`
                    <div class="model-group-header">
                        <i class="${n.icon}"></i>
                        <div>
                            <h3 class="model-group-title">${n.category}</h3>
                            <p class="model-group-desc">${n.description}</p>
                        </div>
                    </div>
                    <div class="model-group-grid"></div>
                `;const o=a.querySelector(".model-group-grid"),l=i.map(c=>{const d=document.createElement(
"button"),m=String(c.apiId||c.id||"").trim(),h=c.agenticView?'<span class="inline-flex items-center \
gap-1 rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold tex\
t-teal-200 whitespace-nowrap" title="Agentic View\u5BFE\u5FDC\uFF1A\u753B\u50CF\u3092\u30AF\u30ED\u30C3\u30D7\u3057\u3066\u518D\u89B3\u5BDF\u3057\u306A\u304C\u3089\u63A8\u8AD6\u3092\u7D99\u7D9A\u3067\u304D\u307E\u3059"><i class="fas fa-eye"\
 aria-hidden="true"></i>Agentic View</span>':"",y=m?`<div class="text-[10px] text-cyan-300/90 mt-1.5\
 font-mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(m)}</\
div>`:"",v=c.price?`<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-\
1"><i class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${c.price}</span></div>`:"";
return d.type="button",d.className="model-card",d.dataset.selected="0",d.onclick=()=>selectModel(c.id,
c.name),d.innerHTML=`
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="model-name font-bold text-sm">${c.name}</span>
                                ${h}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle hidden shrink-0 mt-0.5\
"></i>
                        </div>
                        <span class="model-desc text-[10px]">${c.desc}</span>
                        ${y}
                        ${v}
                    `,o.appendChild(d),{model:c,button:d,searchText:`${c.name} ${c.id} ${m} ${c.agenticView?
"agentic view":""} ${n.category} ${getModelTags(c,n).join(" ")} ${getModelCapabilitySearchTerms(c).join(
" ")}`.toLowerCase(),provider:getModelApiProvider(c.id),tags:new Set(getModelTags(c,n))}});modelListGroups.
push({element:a,entries:l}),e.appendChild(a)}),modelListEmpty=document.createElement("div"),modelListEmpty.
className="model-list-empty hidden",e.appendChild(modelListEmpty),modelListBuilt=!0)}r(buildModelList,
"buildModelList");function updateModelButtonSelection(e,n){const i=n===e.model.id;if(e.button.dataset.
selected===(i?"1":"0"))return;e.button.dataset.selected=i?"1":"0",e.button.classList.toggle("is-sele\
cted",i);const a=e.button.querySelector(".model-selected-icon");a&&a.classList.toggle("hidden",!i)}r(
updateModelButtonSelection,"updateModelButtonSelection");function renderModelList(e="",n={}){const i=get(
"model-list-container");if(!i)return;buildModelList();const a=e.toLowerCase(),o=window._visionPickerActive?
null:getPromptCacheLockedProvider(),l=o?PROVIDER_LABELS[o]||o:"",c=get("model-select")?get("model-se\
lect").value:"";let d=0;modelListBanner.classList.toggle("hidden",!o),o&&(modelListBanner.innerHTML=
`<i class="fas fa-database mr-1.5"></i>PromptCache \u6709\u52B9\u4E2D: <strong>${l}</strong> \u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\
\u629E\u3067\u304D\u307E\u3059\uFF08\u4ED6API\u3078\u306E\u5207\u66FF\u306F\u4E0D\u53EF\uFF09`),modelListGroups.
forEach(m=>{let h=0;m.entries.forEach(y=>{const v=y.searchText.includes(a)&&(!o||y.provider===o)&&(activeModelTag===
"all"||y.tags.has(activeModelTag));y.button.classList.toggle("hidden",!v),updateModelButtonSelection(
y,c),v&&(h+=1)}),m.element.classList.toggle("hidden",h===0),d+=h}),modelListEmpty.classList.toggle("\
hidden",d!==0),d===0&&(modelListEmpty.textContent=o?`No ${l} models found.`:"No models found."),n.animate&&
!modelListAnimated&&(modelListAnimated=!0,i.classList.add("model-list-animate"))}r(renderModelList,"\
renderModelList");function scheduleModelListRender(e){modelListRenderFrame&&cancelAnimationFrame(modelListRenderFrame),
modelListRenderFrame=requestAnimationFrame(()=>{modelListRenderFrame=0,renderModelList(e)})}r(scheduleModelListRender,
"scheduleModelListRender");function animateModelCategoryChange(){const e=get("model-list-container");
e&&(e.classList.remove("model-category-enter"),e.offsetWidth,e.classList.add("model-category-enter"))}
r(animateModelCategoryChange,"animateModelCategoryChange");let modelListScrollFrame=0;function scrollSelectedModelIntoView(){
const e=get("model-list-container"),n=get("model-select")?get("model-select").value:"",i=modelListGroups.
flatMap(x=>x.entries).find(x=>x.model.id===n);if(!e||!i||i.button.classList.contains("hidden"))return;
const a=e.getBoundingClientRect(),o=i.button.getBoundingClientRect(),l=12;let c=e.scrollTop;if(o.top<
a.top+l?c+=o.top-a.top-l:o.bottom>a.bottom-l&&(c+=o.bottom-a.bottom+l),c=Math.max(0,Math.min(c,e.scrollHeight-
e.clientHeight)),Math.abs(c-e.scrollTop)<1)return;if(modelListScrollFrame&&cancelAnimationFrame(modelListScrollFrame),
window.matchMedia("(prefers-reduced-motion: reduce)").matches){e.scrollTop=c;return}const d=e.scrollTop,
m=c-d,h=performance.now(),y=160,v=r(x=>{const w=Math.min(1,(x-h)/y),_=1-Math.pow(1-w,3);e.scrollTop=
d+m*_,w<1?modelListScrollFrame=requestAnimationFrame(v):modelListScrollFrame=0},"step");modelListScrollFrame=
requestAnimationFrame(v)}r(scrollSelectedModelIntoView,"scrollSelectedModelIntoView");function openModelModal(){
location.pathname!=="/model"&&history.pushState({modal:"model"},"","/model");const e=get("model-sear\
ch");e&&(e.value=""),updateModelTagUi(),syncModelSearchClear(),renderModelList("",{animate:!0}),showModal(
"model-modal"),requestAnimationFrame(()=>requestAnimationFrame(scrollSelectedModelIntoView)),e&&window.
innerWidth>768&&requestAnimationFrame(()=>e.focus({preventScroll:!0}))}r(openModelModal,"openModelMo\
dal"),window.closeModelModal=(e=!1)=>{hideModal("model-modal"),!e&&location.pathname==="/model"&&history.
back()};function selectModel(e,n){if(window._visionPickerActive){currentVisionModel=e,window._visionPickerActive=
!1,window.closeModelModal(),_syncVisionModelDisplay();return}if(isPromptCacheEnabled()){const o=getModelApiProvider(
get("model-select")?get("model-select").value:""),l=getModelApiProvider(e);if(o&&l&&o!==l){const c=PROVIDER_LABELS[o]||
o,d=PROVIDER_LABELS[l]||l;showToast(`PromptCache \u6709\u52B9\u4E2D\u306F\u4ED6API\uFF08${d}\uFF09\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\
\u3067\u304D\u307E\u305B\u3093\u3002\u73FE\u5728: ${c}`,"warning",!0);return}}const i=get("model-sel\
ect");i.value=e,get("model-selector-text").innerText=n,window.closeModelModal();const a=new Event("c\
hange");i.dispatchEvent(a)}r(selectModel,"selectModel");function selectModelById(e){let n=e;for(const i of MODELS){
const a=i.items.find(o=>o.id===e);if(a){n=a.name;break}}selectModel(e,n)}r(selectModelById,"selectMo\
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
efault-2fa-method")&&(get("set-default-2fa-method").value=e.default_2fa_method||"totp")}catch{}}r(populateAiSafeFormFields,
"populateAiSafeFormFields");function syncModelSearchClear(){const e=get("model-search"),n=get("model\
-search-clear");n&&n.classList.toggle("hidden",!e||!e.value)}r(syncModelSearchClear,"syncModelSearch\
Clear"),get("model-search")&&get("model-search").addEventListener("input",e=>{scheduleModelListRender(
e.target.value),syncModelSearchClear()}),get("model-search-clear")&&get("model-search-clear").addEventListener(
"click",()=>{const e=get("model-search");e&&(e.value="",syncModelSearchClear(),scheduleModelListRender(
""),e.focus())}),get("model-tag-bar")&&(get("model-tag-bar").addEventListener("click",e=>{const n=e.
target.closest(".model-tag-btn");if(!n)return;const i=n.innerText.trim().toLowerCase(),a=MODEL_TAGS.
includes(i)?i:"all";if(a===activeModelTag)return;activeModelTag=a,updateModelTagUi();const o=get("mo\
del-search");renderModelList(o?o.value:""),animateModelCategoryChange()}),updateModelTagUi()),window.
quickStart=e=>{selectModelById(e),get("welcome-screen").classList.add("hidden")};const BROWSER_FAST_DISABLED_OPTIONS=[
["enable-search","search-container"],["enable-url-context","url-context-container"],["enable-maps","\
maps-grounding-container"],["enable-sys-prompt","sys-prompt-option"],["enable-prompt-cache","prompt-\
cache-container"],["enable-mcp","mcp-container"],["enable-file-creation","file-creation-container"]];
function applyBrowserFastModeRestrictions(){if(!browserFastModeEnabled)return;browserFastPreviousOptions||
(browserFastPreviousOptions={checks:Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([i])=>[i,!!(get(
i)&&get(i).checked)])),coding:!!codingModeEnabled}),BROWSER_FAST_DISABLED_OPTIONS.forEach(([i,a])=>{
const o=get(i),l=get(a);o&&(o.checked=!1,o.disabled=!0),l&&l.classList.add("opacity-50","pointer-eve\
nts-none")}),codingModeEnabled&&syncCodingModeUi(!1,{persist:!1});const e=get("enable-coding-mode"),
n=get("coding-mode-container");e&&(e.disabled=!0),n&&n.classList.add("opacity-50","pointer-events-no\
ne"),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}r(applyBrowserFastModeRestrictions,
"applyBrowserFastModeRestrictions");function restoreBrowserFastModeOptions(){const e=browserFastPreviousOptions;
if(!e)return;BROWSER_FAST_DISABLED_OPTIONS.forEach(([a,o])=>{const l=get(a),c=get(o);l&&(l.disabled=
!1,e&&e.checks&&Object.prototype.hasOwnProperty.call(e.checks,a)&&(l.checked=!!e.checks[a])),c&&c.classList.
remove("opacity-50","pointer-events-none")});const n=get("enable-coding-mode"),i=get("coding-mode-co\
ntainer");n&&(n.disabled=!1),i&&i.classList.remove("opacity-50","pointer-events-none"),e&&e.coding&&
syncCodingModeUi(!0,{persist:!1}),browserFastPreviousOptions=null,typeof updatePromptCacheUi=="funct\
ion"&&updatePromptCacheUi(),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}
r(restoreBrowserFastModeOptions,"restoreBrowserFastModeOptions");function isBatchModelKey(e){const n=String(
e||"").trim().toLowerCase();return n.startsWith("gpt-")?!/(image|audio|tts|transcribe|realtime|search)/.
test(n):n.startsWith("grok-")?!/(image|video|voice|audio|tts|realtime)/.test(n):n.startsWith("gemini\
-")?!/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.
test(n):!1}r(isBatchModelKey,"isBatchModelKey");function updateBatchUi(e){const n=get("batch-mode-co\
ntainer"),i=get("enable-batch-mode");if(!n||!i)return;const a=isBatchModelKey(e);n.classList.toggle(
"hidden",!a),i.disabled=!a||browserFastModeEnabled,a||(i.checked=!1),n.classList.toggle("ring-1",a&&
i.checked),n.classList.toggle("ring-violet-300",a&&i.checked)}r(updateBatchUi,"updateBatchUi");function setBrowserFastModeEnabled(e,n={}){
browserFastModeEnabled=!!e;const i=get("enable-browser-fast-mode");i&&(i.checked=browserFastModeEnabled);
const a=get("browser-fast-mode-container");a&&(a.classList.toggle("ring-1",browserFastModeEnabled),a.
classList.toggle("ring-amber-300",browserFastModeEnabled)),!browserFastModeEnabled&&n.clearKey!==!1&&
(browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null),browserFastModeEnabled?applyBrowserFastModeRestrictions():
n.restoreOptions!==!1&&restoreBrowserFastModeOptions(),updateBatchUi(get("model-select")?get("model-\
select").value:"")}r(setBrowserFastModeEnabled,"setBrowserFastModeEnabled");function openBrowserFastModeModal(e=!0){
const n=get("browser-fast-mode-warning"),i=get("browser-fast-mode-ignore-row");n&&n.classList.toggle(
"hidden",!e),i&&i.classList.toggle("hidden",!e);const a=get("browser-fast-mode-key-description"),o=String(
get("model-select")?get("model-select").value:"Gemini");a&&(a.textContent=`${o} \u306E\u30E2\u30C7\u30EB\u5225\u30AD\u30FC \u2192 \u5171\u901AGemini\u30AD\u30FC\
\u306E\u9806\u306B\u3001\u30B5\u30FC\u30D0\u30FC\u304B\u3089\u81EA\u52D5\u53D6\u5F97\u3057\u307E\u3059\u3002`),
showModal("browser-fast-mode-modal")}r(openBrowserFastModeModal,"openBrowserFastModeModal");function browserFastBootstrapMatches(e,n,i,a){
return!e||e.model!==n||String(e.thread_id||"")!==String(i||"")?!1:String(e.parent_id||"")===String(a||
"")}r(browserFastBootstrapMatches,"browserFastBootstrapMatches");async function fetchBrowserFastBootstrap(e=!1){
const n=String(get("model-select")?get("model-select").value:"").trim(),i=currentThreadId||null,a=i&&
currentParentId||null;if(!e&&browserFastBootstrapMatches(browserFastBootstrap,n,i,a)&&browserFastApiKey)
return browserFastBootstrap;const o=await apiFetch("/api/browser_fast_mode/bootstrap",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({model:n,thread_id:i,parent_id:a})}),
l=await o.json().catch(()=>({}));if(!o.ok||!l.api_key)throw new Error(l.error||"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u306EGemini API\u30AD\
\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");return browserFastApiKey=
String(l.api_key),browserFastApiKeyModel=n,browserFastBootstrap=l,l}r(fetchBrowserFastBootstrap,"fet\
chBrowserFastBootstrap");async function requestBrowserFastModeEnable(){const e=String(get("model-sel\
ect")?get("model-select").value:"").toLowerCase();if(!e.startsWith("gemini-")||/(image|native-audio|tts|live)/.
test(e)){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306FGemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059",
"warning",!0),setBrowserFastModeEnabled(!1);return}if(currentImageUrls.length||uploadProgressState.active>
0||browserFastLocalFiles.size){showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u308B\u524D\u306B\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u3092\u30AF\u30EA\u30A2\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0),setBrowserFastModeEnabled(!1);return}const n=(()=>{try{return localStorage.getItem(BROWSER_FAST_IGNORE_WARNING_STORAGE)===
"1"}catch{return!1}})();if(n){try{await fetchBrowserFastBootstrap(!0),setBrowserFastModeEnabled(!0,{
clearKey:!1}),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F",
"warning",!1)}catch(i){setBrowserFastModeEnabled(!1),showToast(i.message||"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u5316\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}return}openBrowserFastModeModal(!n)}r(requestBrowserFastModeEnable,"requestBrowserFastMo\
deEnable"),document.addEventListener("DOMContentLoaded",()=>{get("menu-btn")&&(get("menu-btn").onclick=
()=>{get("sidebar").classList.toggle("open"),get("overlay").classList.toggle("active")}),get("overla\
y")&&(get("overlay").onclick=()=>{get("sidebar").classList.remove("open"),get("overlay").classList.remove(
"active")})}),document.addEventListener("DOMContentLoaded",()=>{var ti,ni;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const u=get("enable-batch-mode");u&&u.checked&&(u.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const n=get("enable-batch-mode");n&&(n.onchange=()=>{if(n.checked){browserFastModeEnabled&&
setBrowserFastModeEnabled(!1);const u=get("enable-coding-mode");u&&u.checked&&(u.checked=!1,typeof syncCodingModeUi==
"function"&&syncCodingModeUi(!1),showToast("Batch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const i=get("model-\
select");i&&i.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const u=String(
i.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!u.startsWith("gemini-")||/(image|native-audio|tts|live|flash-cyber)/.test(u)?(setBrowserFastModeEnabled(
!1),i.dispatchEvent(new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const a=get("browser-fast-mode-enable-btn");
a&&(a.onclick=async()=>{const u=a.innerHTML;a.disabled=!0,a.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const f=get("browser-fast-mode-ignore-warning");if(f&&f.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(f){showToast(f.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{a.disabled=!1,a.innerHTML=u}});const o=get("browser-fast-mode-cancel-btn");o&&(o.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const l=document.getElementById(
"alpha-bar");setTimeout(()=>{if(l){const u=document.getElementById("version-display");if(u){const f=l.
getBoundingClientRect(),g=u.getBoundingClientRect(),b=g.left+g.width/2-(f.left+f.width/2),k=g.top+g.
height/2-(f.top+f.height/2);l.style.transform=`translate(${b}px, ${k}px) scale(0.1)`,l.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),l.remove()},800)}else l.style.opacity="0",setTimeout(()=>l.remove(),1e3)}},3e3);function c(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const f=get("gpt-image-format"),g=get("gpt-image-compression-wrap");f&&g&&(f.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}r(c,"updateGptImageUi");function d(){const u=get(
"gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.remove("hidden"):u.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{b&&(Array.from(b.options).
forEach(k=>{k.value!=="1K"&&(k.disabled=g)}),g&&b.value!=="1K"&&(b.value="1K"))})}r(d,"updateGeminiI\
mageUi");function m(){const u=get("grok-image-options");if(!u)return;const f=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),b=f==="grok-imagine-image-quality"||f==="grok-imagine-image-2\
.0",k=f==="grok-imagine-image-2.0";if(g){u.classList.remove("hidden");const C=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;C&&C.classList.toggle("hidden",!b);const A=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.toggle("hidden",!k)}else
u.classList.add("hidden");if(get("modal-grok-image-options")){const C=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;C&&C.classList.toggle("hidden",!b);const A=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;A&&A.classList.toggle(
"hidden",!k)}}r(m,"updateGrokImageUi");function h(){var b;const u=get("grok-video-options");if(!u)return;
const f=String(((b=get("model-select"))==null?void 0:b.value)||"").toLowerCase();isGrokVideoModel()?
u.classList.remove("hidden"):u.classList.add("hidden");const g=get("grok-video-resolution");if(g){const k=Array.
from(g.options).find(T=>T.value==="1080p");k&&(k.disabled=f!=="grok-imagine-video-1.5"),f!=="grok-im\
agine-video-1.5"&&g.value==="1080p"&&(g.value="720p")}}r(h,"updateGrokVideoUi");function y(){var k;const u=get(
"gemini-video-options");if(!u)return;const f=String(((k=get("model-select"))==null?void 0:k.value)||
"").toLowerCase();isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");const g=get(
"gemini-video-resolution");if(g){const T=Array.from(g.options).find(A=>A.value==="4K"),C=f==="veo-3.\
1-lite-generate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-flash";T&&(T.disabled=
C),C&&g.value==="4K"&&(g.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle(
"hidden",f==="gemini-omni-1.1-flash")}r(y,"updateGeminiVideoUi");function v(){const u=get("gemini-mu\
sic-options");if(!u)return;const f=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!f;u.classList.
toggle("hidden",!g);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!f)}r(v,
"updateGeminiMusicUi");function x(){var C;const u=get("xai-chat-options");if(!u)return;const f=String(
((C=get("model-select"))==null?void 0:C.value)||"").toLowerCase(),g=f.startsWith("grok-")&&!isGrokImageModel(
f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!g);const b=get("xai-logp\
robs"),k=get("xai-top-logprobs"),T=f.includes("grok-4.20");b&&(b.disabled=T,T&&(b.checked=!1)),k&&(k.
disabled=T,T&&(k.value=""))}r(x,"updateXaiChatUi");function w(){const u=isMistralOcrModel(),f=get("m\
istral-ocr-options");f&&f.classList.toggle("hidden",!u);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const k=get(b);k&&(k.classList.toggle("opacity-50",u),k.classList.toggle("po\
inter-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(w,"updateMistralOcrUi");function _(){const u=get("image-input-limit\
s");if(!u)return;const f=(get("model-select").value||"").toLowerCase();let g="",b=!1;f.includes("gpt\
-image")?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):f==="deepseek-v4.1-flash"||f==="deepseek-v4-flash-vision-exp"?(b=!0,g=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):f.includes("deepseek")||(isGeminiImageModelKey(f)?(b=!0,f.includes("gemini-3.1-flash-lite-\
image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):f.includes("gemini-3.1-flash-image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):f.includes("gemini-2.5")&&f.includes("image")?g=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):g=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(f)?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):f.includes("grok")?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):f.includes("grok")&&f.includes("video")&&(b=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(u.innerHTML=g,u.classList.remove("hidden")):(u.classList.add("hidden"),u.innerHTML="")}
r(_,"updateImageInputLimits");function S(){const u=get("model-select");if(!u)return;const f=u.value,
g=String(f||"").toLowerCase(),b=g.includes("deepseek"),k=get("thinking-options"),T=get("reasoning-ef\
fort-container"),C=get("enable-thinking"),A=get("thinking-level"),I=get("thinking-budget"),D=get("en\
able-search"),O=get("search-container"),G=get("url-context-container"),se=get("enable-maps"),R=get("\
maps-grounding-container"),$=get("enable-sys-prompt"),U=get("sys-prompt-option"),K=get("enable-pytho\
n"),ke=get("python-container"),X=get("prompt-cache-container"),re=get("enable-prompt-cache"),Te=f===
"gpt-5-search-api",Re=f.includes("tts"),je=isMistralOcrModel(f),Ze=g.includes("gemini-3.1-flash-lite\
-image"),it=g.includes("gemini-3.1-flash-image")&&!Ze,Pt=isClaudeModelKey(f),vt=g==="gemini-3.8-flas\
h-cyber",Ot=isLlmModel()&&!b&&!Re&&!g.includes("realtime")&&!g.includes("native-audio")&&!g.includes(
"live");X&&(Ot?(X.classList.remove("hidden","opacity-50","pointer-events-none"),re&&(re.disabled=!1)):
(re&&(re.checked=!1,re.disabled=!0),X.classList.add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),
k&&k.classList.add("hidden"),T&&T.classList.add("hidden");const kt=get("vision-model-info");if(kt&&kt.
classList.add("hidden"),T){const me=get("reasoning-effort");if(me){Array.from(me.options).forEach(Oe=>{
const _t=g==="gpt-5.6"||g.startsWith("gpt-5.6-"),Jt=g==="deepseek-v4.1-flash"||g==="deepseek-v4-flas\
h-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash-vision-exp",qe=g==="deepseek-v4-pro",dt=g.includes(
"grok-4.5"),cn=g.includes("grok-4.6");Oe.value==="max"?Oe.classList.toggle("hidden",!_t&&!Jt&&!qe):Oe.
value==="xhigh"?Oe.classList.toggle("hidden",!cn&&!g.includes("multi-agent")&&!_t):Oe.value==="mediu\
m"?Oe.classList.toggle("hidden",!(g.includes("grok-4.3")||dt||cn||g.includes("grok-4.20-0309-reasoni\
ng")||g.includes("grok-build")||g.includes("multi-agent")||g.includes("gpt-5")||g.includes("o1")||g.
includes("o3"))):Oe.value==="none"?Oe.classList.toggle("hidden",!g.includes("grok-4.3")&&!g.includes(
"grok-build")&&!g.includes("gpt-5")&&!Jt&&!qe):Oe.value==="low"&&Oe.classList.toggle("hidden",qe)});
const we=me.selectedOptions&&me.selectedOptions[0];we&&we.classList.contains("hidden")&&(me.value=b?
"high":"medium")}}G&&G.classList.add("hidden"),R&&R.classList.add("hidden"),C&&(C.disabled=!1),I&&(I.
disabled=!0,I.classList.add("opacity-50"));const fe=isGeminiImageModelKey(f);if(Re||je)O&&(get("enab\
le-search").checked=!1,O.classList.add("opacity-50","pointer-events-none")),G&&(get("enable-url-cont\
ext").checked=!1,G.classList.add("opacity-50","pointer-events-none")),R&&se&&(se.checked=!1,R.classList.
add("opacity-50","pointer-events-none")),ke&&(K.checked=!1,ke.classList.add("opacity-50","pointer-ev\
ents-none")),$&&U&&($.checked=!1,$.disabled=!0,U.classList.add("opacity-50"));else if(it||Ze)R&&se&&
(se.checked=!1,R.classList.add("hidden","opacity-50","pointer-events-none")),k.classList.remove("hid\
den"),Array.from(A.options).forEach(me=>{["low","medium"].includes(me.value)&&(me.disabled=!0),["min\
imal","high"].includes(me.value)&&(me.disabled=!1)}),["minimal","high"].includes(A.value)||(A.value=
Ze?"minimal":"high"),C&&(C.disabled=!1),Ze&&(D&&(D.checked=!1,D.disabled=!0),O&&O.classList.add("opa\
city-50","pointer-events-none"));else if(fe)R&&se&&(se.checked=!1,R.classList.add("hidden","opacity-\
50","pointer-events-none"));else if(Pt)k.classList.remove("hidden"),I&&(I.disabled=!1,I.classList.remove(
"opacity-50")),Array.from(A.options).forEach(me=>{me.disabled=!0}),ke&&(K.checked=!1,ke.classList.add(
"opacity-50","pointer-events-none"));else if(vt){k&&k.classList.remove("hidden"),C&&(C.checked=!0,C.
disabled=!0),Array.from(A.options).forEach(we=>{we.disabled=!["low","medium","high"].includes(we.value)}),
["low","medium","high"].includes(A.value)||(A.value="medium"),[O,G,R,ke].forEach(we=>{we&&we.classList.
add("opacity-50","pointer-events-none")}),[D,se,K].forEach(we=>{we&&(we.checked=!1,we.disabled=!0)});
const me=get("enable-url-context");me&&(me.checked=!1,me.disabled=!0),$&&U&&($.disabled=!1,U.classList.
remove("opacity-50"))}else if(f.includes("gemini")&&!fe){k.classList.remove("hidden"),G&&G.classList.
remove("hidden","opacity-50","pointer-events-none");const me=f.includes("gemini-3");R&&(me?R.classList.
remove("hidden","opacity-50","pointer-events-none"):(se&&(se.checked=!1),R.classList.add("hidden","o\
pacity-50","pointer-events-none")));const we=f.includes("flash");Array.from(A.options).forEach(Oe=>{
f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?Oe.disabled=!["low","medium","high"].includes(Oe.value):
f==="gemini-3.6-flash"?Oe.disabled=!["medium","high"].includes(Oe.value):f==="gemini-3.5-flash-lite"?
Oe.disabled=!["minimal","medium","high"].includes(Oe.value):["minimal","medium"].includes(Oe.value)?
Oe.disabled=!we:Oe.disabled=!1}),(f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium",
"high"].includes(A.value)||f==="gemini-3.6-flash"&&!["medium","high"].includes(A.value)?A.value="med\
ium":f==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(A.value)?A.value="minimal":!we&&
["minimal","medium"].includes(A.value)&&(A.value="high"),me?C&&(C.checked=!0,C.disabled=!0):C&&(C.disabled=
!1),I&&f.includes("gemini-2.5")&&(I.disabled=!1,I.classList.remove("opacity-50")),I&&!f.includes("ge\
mini-2.5")&&(I.disabled=!0,I.classList.add("opacity-50"))}if(isLlmModel()&&(g.includes("gpt-5")||g.includes(
"o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4.5")||g.includes("grok-4.6")||g.includes(
"grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("multi-agent")||g.includes("gpt")&&
!g.includes("tts")))T.classList.remove("hidden"),O&&O.classList.remove("opacity-50","pointer-events-\
none");else if(b){T.classList.remove("hidden");const me=get("vision-model-info");if(me&&me.classList.
toggle("hidden",g==="deepseek-v4.1-flash"||g==="deepseek-v4-flash-vision-exp"),D&&(D.checked=!1,D.disabled=
!0),O&&O.classList.add("opacity-50","pointer-events-none"),G){const we=get("enable-url-context");we&&
(we.checked=!1),G.classList.add("opacity-50","pointer-events-none")}R&&se&&(se.checked=!1,R.classList.
add("opacity-50","pointer-events-none"))}else je||(O&&O.classList.remove("opacity-50","pointer-event\
s-none"),R&&se&&(se.checked=!1,R.classList.add("hidden","opacity-50","pointer-events-none")));if(Re?
ke&&ke.classList.add("opacity-50","pointer-events-none"):(ke&&ke.classList.remove("opacity-50","poin\
ter-events-none"),(!fe||it)&&!f.includes("gpt-image")&&($.disabled=!1,U.classList.remove("opacity-50"))),
(fe&&!it||f.includes("gpt-image")||isGrokImageModel()||isGrokVideoModel()||je)&&$&&U&&($.checked=!1,
$.disabled=!0,U.classList.add("opacity-50")),ke&&(isLlmModel()?(ke.classList.remove("hidden"),K.disabled=
!1):(K.checked=!1,K.disabled=!0,ke.classList.add("hidden"))),Te?(D&&(D.checked=!0,D.disabled=!0),O&&
O.classList.add("opacity-50","pointer-events-none"),ke&&(K.checked=!1,K.disabled=!0,ke.classList.add(
"opacity-50","pointer-events-none"))):D&&!f.includes("tts")&&!je&&!b&&!Ze&&(D.disabled=!1),vt){[D,se,
K].forEach(we=>{we&&(we.checked=!1,we.disabled=!0)});const me=get("enable-url-context");me&&(me.checked=
!1,me.disabled=!0),[O,G,R,ke].forEach(we=>{we&&we.classList.add("opacity-50","pointer-events-none")})}
const Be=get("mask-btn");Be&&(isGptImageModel()?Be.classList.remove("hidden"):(Be.classList.add("hid\
den"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),c(),
d(),m(),h(),y(),v(),updateBatchUi(f),x(),w(),_(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}r(S,"toggleOptions"),get("model-select")&&(get("model-select").addEventListener(
"change",S),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
S(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const L=get("enable-canvas-mode");L&&(L.checked=canvasModeEnabled,L.addEventListener("change",()=>syncCanvasModeUi(
L.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const M=get("enable-codin\
g-mode");M&&(M.checked=codingModeEnabled,M.addEventListener("change",()=>syncCodingModeUi(M.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",u=>{const f=u.
target.closest("[data-canvas-block-index]");if(!f)return;const g=Number(f.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(g,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",u=>{if(u.target.value===
"")return;const f=Number(u.target.value);Number.isInteger(f)&&applyCanvasSelection(f,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",u=>{const f=u.target.closest(
"[data-canvas-panel-view]");if(!f)return;const g=f.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
g,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const u=getCanvasModeElements(),f=u&&u.code&&u.code.textContent||"";if(!f.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
f,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const P=get(
"prompt-controls-toggle-btn");P&&(P.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>c()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>d()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,g){const b=(f||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(u||"");const k=u||"",T=(b||"").toLowerCase();
let C="";try{const $=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&k.length>2e4?C=escapeHtml(
k):C=hljs.highlight(k,{language:$}).value}catch{C=escapeHtml(k)}const A=encodeURIComponent(k).replace(
/'/g,"%27"),I=detectBlockedScriptsInCode(k),D=hashString(`${b||"TEXT"}
${k||""}`);let O="";if(canvasModeEnabled){const $=String(canvasPreviewState.selectedKey||"")===D,U=$?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";O=`<button\
 class="canvas-preview-btn${$?" canvas-active":""}" data-code="${A}" data-code-key="${D}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${U}" aria-label="${U}" aria-pressed="${$?"true":"false"}"><\
i class="fas ${$?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
T,k)){const $=I?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
O=`<button class="html-preview-btn" data-code="${A}" ${I?'data-suspicious="1"':""} title="${$}" aria\
-label="${$}"><i class="fas ${I?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const G=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,se=T==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${D}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,R=(b||"TEXT")+(I?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\u305F\
">\u26A0</span>':"");return`<div class="code-wrapper collapsed" data-collapsed="true" data-code-key=\
"${D}"><div class="code-header"><span class="code-lang">${R}</span><div class="code-actions"><button\
 class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"\
></i></button>${se}${O}${G}<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i\
 class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language\
-${b}">${C}</code></pre></div></div>`},link(u,f,g){return`<a href="${u}" title="${f||""}" target="_b\
lank">${g}</a>`},image(u,f,g){return buildChatImageHtml(u,{alt:g,title:f})}},breaks:!0,gfm:!0}),threadObserver=
new IntersectionObserver(u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread\
-list"),threshold:.1}),threadObserver.observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),
(ti=get("version-update-dismiss"))==null||ti.addEventListener("click",()=>{const u=localStorage.getItem(
"app_version")||"";u&&localStorage.setItem("version_notified",u),hideModal("version-update-modal")});
const H=get("version-update-clear-cache");if(H&&(H.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.
clearCacheOnVersionUpdate),H.addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=
saveVersionUpdateCachePreference(H.checked)})),(ni=get("version-update-reload"))==null||ni.addEventListener(
"click",async()=>{var f;await versionUpdateCachePreferenceSavePromise.catch(()=>{}),!!((f=get("versi\
on-update-clear-cache"))!=null&&f.checked)?await clearSiteCacheAndReload(get("version-update-reload"),
{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.setVersionChangeHandler(
u=>{u&&u!==appVersion&&(localStorage.getItem("version_notified")||"")!==u&&(localStorage.setItem("ap\
p_version",u),purgeCaches().then(()=>checkAndNotifyVersion(u)))}),window.ConnectionMonitor.start(),window.
addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener("offline",
()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offline")}),window.
addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener("visibil\
itychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener("page\
hide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&botConfig.
lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(u){console.error(u)}try{runBotDetectionGate()}catch(u){console.error(
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const Q=r(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),ee=r(u=>{const f=Array.isArray(u)?u:[],g=get("passkey-list"),b=get("passkey-count");if(b&&
(b.innerText=String(f.length)),!!g){if(!f.length){g.innerHTML='<div class="text-[11px] text-gray-500\
">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}g.innerHTML="",f.forEach((k,T)=>{const C=k&&k.id?String(k.id):"",A=document.createElement("di\
v");A.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const I=document.createElement("div");I.className="min-w-0";const D=document.createElement("\
div");D.className="text-xs text-gray-200 truncate",D.innerText=k&&k.name?String(k.name):`Security Ke\
y ${T+1}`;const O=document.createElement("div");O.className="text-[10px] text-gray-500 mt-1",O.innerText=
k&&k.created_at?`\u767B\u9332\u65E5\u6642: ${Q(k.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
I.appendChild(D),I.appendChild(O),A.appendChild(I);const G=document.createElement("button");G.type="\
button",G.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",G.innerText="\u524A\u9664",G.disabled=!C,C&&(G.onclick=()=>window.removeWebAuthnCredential(
C)),A.appendChild(G),g.appendChild(A)})}},"renderPasskeyList"),Ae=r(u=>{const f=get("session-list");
if(f){if(!u||!u.length){f.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(g=>{const b=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",k=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",T=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
C=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${k}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(C)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(Q(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(Q(g.created_at))}</div></div>${T}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const b=g.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const k=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let T={};try{T=await k.json()}catch{}if(k.ok){if(T.logged_out){
location.href="/login";return}await B()}else showToast(T&&T.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),B=r(async()=>{const u=get("session-list");u&&(u.innerHTML='<div c\
lass="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api/\
sessions");let g={};try{g=await f.json()}catch{}if(!f.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(g.sessions||[]).filter(k=>!k.is_revoked);Ae(b)},"loadSessions"),V=r(()=>{const u=get(
"session-refresh-btn");u&&(u.onclick=()=>B());const f=get("session-revoke-others-btn");f&&(f.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await B():showToast("\u64CD\u4F5C\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(u=>{u&&(currentVisionModel=u.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
u);try{loadMcpServers()}catch{}u&&u.theme_color&&applyThemeColor(u.theme_color,!0),u&&Object.prototype.
hasOwnProperty.call(u,"minimal_prompt_mode")&&u.minimal_prompt_mode?setMinimalPromptMode(!0):u&&Object.
prototype.hasOwnProperty.call(u,"compact_prompt_mode")&&setCompactPromptMode(!!u.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(u.enable_client_debug_log===!0,"settings sync");
const f=get("enable-sys-prompt");f&&u&&u.system_prompt&&String(u.system_prompt).trim()&&(!f.disabled&&
!u.default_enable_system_prompt&&!u.use_last_chat_settings&&(f.checked=!0),S())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const te=get("vision-model-change-btn");te&&(te.onclick=()=>_openVisionModelSelector());const ge=get(
"compression-format-only");ge&&(ge.onchange=()=>{const u=ge.checked,f=get("compression-max-size"),g=get(
"compression-max-dim");f&&(f.disabled=u),g&&(g.disabled=u);const b=get("compression-size-wrap"),k=get(
"compression-dim-wrap");b&&(b.style.opacity=u?"0.4":"1"),k&&(k.style.opacity=u?"0.4":"1")});const de=r(
()=>{const u=get("enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=
!!temporaryChatEnabled,u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(
u.checked)||(setTemporaryChatUiState(f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");de(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const Ce=get("storage-u\
sage-refresh");Ce&&(Ce.onclick=()=>loadStorageUsage());let be=null;const Le=r(()=>{const u=new Uint8Array(
16);return window.crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),oe=r((u={})=>{const f=get("account-transfer-progress"),g=get("account-tra\
nsfer-progress-bar"),b=get("account-transfer-progress-percent"),k=get("account-transfer-progress-tex\
t"),T=get("account-transfer-progress-detail"),C=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&
f.classList.remove("hidden"),g&&(g.style.width=`${C}%`),b&&(b.textContent=`${Math.round(C)}%`),k&&(k.
textContent=u.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),T){
const I={queued:"\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};T.textContent=I[u.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),le=r(u=>{Se&&(Se.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!u)},"setAccountTransferControls"),Y=r((u={})=>{const f=get("account-export-ready"),g=get(
"account-export-ready-text"),b=get("account-export-expiry"),k=get("account-export-download-btn"),T=!!(u.
available&&u.download_url);if(f&&f.classList.toggle("hidden",!T),!T){k&&k.removeAttribute("href");return}
const C=Math.max(0,Number(u.size_bytes)||0),A=C>=1024*1024*1024?`${(C/(1024*1024*1024)).toFixed(2)} \
GB`:`${(C/(1024*1024)).toFixed(1)} MB`;if(g){const I=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${I}`}
if(b){const I=u.expires_at?new Date(u.expires_at):null;b.textContent=I&&!Number.isNaN(I.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${I.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
k&&(k.href=u.download_url)},"renderAccountExportAvailability"),E=r(async u=>{for(;be===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await f.json().catch(()=>({}));if(f.ok&&(g.state!=="pending"&&oe(g),["ready","completed","\
failed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(f=>setTimeout(f,
700))}return null},"pollAccountTransfer"),j=r((u,f,g=!0)=>{f&&(oe(f),Y(f),g&&f.state==="ready"?showToast(
f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):g&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),Z(u))},"handleFinishedAccountExport"),W=r(async()=>{try{const u=await apiFetch("/api/acc\
ount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(Y(f),f.state==="ready"){oe(f);return}if(["failed","cancelled","expired"].includes(
f.state)){oe(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||be&&be.id===
f.job_id||be)return;const g={id:f.job_id,type:"export",stopped:!1,restored:!0};be=g,le(!0),oe(f);const b=await E(
g);b&&j(g,b,!0)}catch{}},"refreshLatestAccountExport"),Z=r(u=>{be===u&&(be=null),u.stopped=!0,le(!1)},
"finishAccountTransfer"),ie=get("account-transfer-cancel-btn");ie&&(ie.onclick=async()=>{const u=be;
if(!(!u||u.stopped)){u.cancelRequested=!0,ie.disabled=!0,oe({progress:0,phase:"cancelling",message:"\
\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/tra\
nsfer/${u.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}u.controller&&u.controller.
abort(),oe({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
u.type==="export"&&Y({available:!1}),Z(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const Se=get("account-export-btn");Se&&(Se.onclick=async()=>{if(be)return;const u={id:Le(),
type:"export",stopped:!1};be=u,le(!0),Y({available:!1}),oe({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:u.id}),keepalive:!0})),g=await f.json().catch(()=>({}));if(f.status===409&&
g.error==="export_in_progress"&&g.job_id)u.id=g.job_id;else if(!f.ok)throw new Error(g.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
oe({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await E(u);!u.cancelRequested&&b&&j(u,b,!0)}catch(f){const g=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";oe({progress:0,phase:"failed",message:g}),
showToast(g,"error",!0),Z(u)}});const ne=get("account-export-download-btn");ne&&ne.addEventListener(
"click",async u=>{const f=ne.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await g.json().catch(
()=>({}));g.ok&&b.available&&b.download_url?(ne.href=b.download_url,window.location.assign(b.download_url)):
(Y(b),oe(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),W())}catch{window.location.assign(f)}}}),le(!1),W();const ue=get("import-files-grid"),
J=get("import-files-info"),ve=get("import-files-summary"),at=r(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let Ee=null;const lt=r(()=>{
if(!Ee)return;const u=Ee.files,f=Ee.selection;let g=0;u.forEach(T=>{f.has(T.archive_path)&&(g+=Number(
T.size_bytes)||0)});const b=Number(Ee.available_bytes)||0,k=g>b;ve&&(ve.textContent=`\u9078\u629E\u4E2D: ${at(
g)} / \u5229\u7528\u53EF\u80FD: ${at(b)}${k?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,ve.classList.
toggle("text-red-300",k)),J&&(J.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),ht=r(
()=>{if(!ue||!Ee)return;ue.innerHTML="";const u=Ee.files;if(!u.length){ue.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
lt();return}u.forEach(f=>{const g=document.createElement("label"),b=Ee.selection.has(f.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${at(f.size_bytes)}</div></div>`;const k=g.
querySelector(".import-file-check");k.addEventListener("change",()=>{k.checked?Ee.selection.add(f.archive_path):
Ee.selection.delete(f.archive_path),g.classList.toggle("border-blue-500",k.checked),g.classList.toggle(
"border-gray-600",!k.checked),lt()}),ue.appendChild(g)}),lt()},"renderImportFileItems"),Qe=r(u=>new Promise(
f=>{if(Ee={files:u.files||[],selection:new Set((u.files||[]).map(g=>g.archive_path)),available_bytes:u.
available_bytes,resolve:f},ht(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),ut=r(u=>{if(hideModal("import-files-modal"),Ee){const f=Ee.resolve;
Ee=null,f(u)}},"closeImportFileSelection"),St=get("import-files-close");St&&(St.onclick=()=>ut(null));
const wt=get("import-files-cancel");wt&&(wt.onclick=()=>ut(null));const Tt=get("import-files-confirm");
Tt&&(Tt.onclick=()=>{if(!Ee)return;const u=Array.from(Ee.selection);ut(u.length?u.join(","):"__none_\
_")});const pt=get("import-files-select-all");pt&&(pt.onclick=()=>{Ee&&(Ee.files.forEach(u=>Ee.selection.
add(u.archive_path)),ht())});const Rt=get("import-files-none");Rt&&(Rt.onclick=()=>{Ee&&(Ee.selection.
clear(),ht())});const Bt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},ct=r(u=>{
if(u===!0)return"ON";if(u===!1)return"OFF";if(u==null||u==="")return"\u672A\u8A2D\u5B9A";const f=String(
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let gt=null;const Ct=r(
u=>{if(gt){const f=gt;gt=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),Lt=r(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}gt=f;const b=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],k=get("settings-confirmation-list");k&&(b.length?
k.innerHTML=b.map(C=>{const A=Bt[C.field]||C.field,I=ct(C.current),D=ct(C.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
I)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(D)}</d\
iv>
                            </div>`}).join(""):k.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const T=get("settin\
gs-confirmation-count");T&&(T.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),bt=get("settings-confir\
mation-modal");bt&&bt.addEventListener("click",u=>{u.target===bt&&Ct(!1)});const Kt=get("settings-co\
nfirmation-close");Kt&&(Kt.onclick=()=>Ct(!1));const Xt=get("settings-confirmation-cancel");Xt&&(Xt.
onclick=()=>Ct(!1));const F=get("settings-confirmation-confirm");F&&(F.onclick=()=>Ct(!0));const ce=get(
"account-import-btn"),Me=get("account-import-inplace"),Ie=get("account-import-inplace-warning");if(Me&&
Ie){const u=r(()=>Ie.classList.toggle("hidden",!Me.checked),"syncInplaceWarn");Me.addEventListener("\
change",u),u()}ce&&(ce.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,g=get("account-import-categories"),b=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(R=>R.value):[],k=get("account-import-inplace"),T=!!(k&&k.checked),C=get("account-import\
-settings-bypass"),A=!!(C&&C.checked);let I=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const D=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(R=>(R.
closest("label")&&R.closest("label").textContent||R.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\u5B58\u30C7\
\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${D.join("\u3001")}${T?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const O={id:Le(),type:"import",stopped:!1,controller:new AbortController};
be=O,le(!0),oe({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const G=get("account-import-result");let se=Promise.resolve(null);try{const $=Math.max(1,Math.ceil(f.
size/10485760)),U=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:O.controller.
signal})),K=await U.json().catch(()=>({}));if(!U.ok)throw new Error(K.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
O.uploadId=K.upload_id;const ke=K.chunk_size||10485760;let X=0,re=0;const Te=r(async()=>{for(;;){const fe=re++;
if(fe>=$)return;const he=f.slice(fe*ke,Math.min(f.size,(fe+1)*ke)),Be=new FormData;Be.append("chunk",
he,f.name),Be.append("index",String(fe));const me=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
O.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Be,signal:O.controller.signal})),
we=await me.json().catch(()=>({}));if(!me.ok)throw new Error(we.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
X++,oe({progress:Math.min(35,Math.round(X/$*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${X}\
/${$}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let Re=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),Re=!0);try{await Promise.
all([Te(),Te(),Te()]);const fe=await apiFetch(`/api/account/import/upload/${encodeURIComponent(O.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:O.controller.signal})),he=await fe.json().
catch(()=>({}));if(!fe.ok)throw new Error(he.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
oe({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
Re&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let je="",Ze=!1,it=0;const Pt=r(
async()=>{let fe=!1;const he=r(()=>{fe||(fe=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Be=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),me=await Be.
json().catch(()=>null);if(!Be.ok||!me){he();return}cacheUserSettings(me);const we=get("settings-moda\
l");if(we&&we.classList.contains("modal-open"))try{Bn(me)}catch{}me.theme_color&&applyThemeColor(me.
theme_color,!0),Object.prototype.hasOwnProperty.call(me,"minimal_prompt_mode")&&me.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(me,"compact_prompt_mode")&&setCompactPromptMode(
!!me.compact_prompt_mode)}catch{}he()},"refreshSettingsFormAfterImport"),vt=r(fe=>{const he=fe&&fe.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";G&&(G.textContent=`\u5B8C\u4E86: ${he}`,
G.classList.remove("hidden","text-red-300"),G.classList.add("text-emerald-300")),oe({progress:100,phase:"\
completed",message:he}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&Pt()},"finishImportSuccess"),Ot=r(async()=>{
try{const he=await(await apiFetch(`/api/account/transfer/${O.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return he&&he.state?he:null}catch{return null}},"fetchImportSta\
tus"),kt=r(async()=>{const fe=await Ot();if(!fe)return{status:"unknown"};if(fe.state==="completed")return vt(
fe),{status:"done"};if(["failed","cancelled","expired"].includes(fe.state))throw new Error(fe.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(fe.state==="needs_sel\
ection"&&Array.isArray(fe.files)){const he=await Qe({files:fe.files,available_bytes:fe.available_bytes});
return he===null?(oe({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
O.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(O.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(je=he,{status:"reselect"})}if(fe.state===
"needs_settings_confirmation"&&Array.isArray(fe.settings_changes))return await Lt({settings_changes:fe.
settings_changes})?(I=!0,{status:"reselect"}):(oe({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\
\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),O.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
O.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(fe.state==="running"){const he=await Promise.race([se.catch(()=>null),new Promise(Be=>setTimeout(
()=>Be(null),6e4))]);if(he&&he.state==="completed")return vt(he),{status:"done"};throw he&&["failed",
"cancelled","expired"].includes(he.state)?new Error(he.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ze;){O.stopped=!0,await se.catch(()=>null),
O.stopped=!1,se=E(O);let fe;try{fe=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:O.uploadId,
categories:b.join(","),job_id:O.id,selected_files:je,restore_inplace:T,confirm_settings:I||A}),signal:O.
controller.signal}))}catch(qe){if(O.cancelRequested||qe&&qe.name==="AbortError")throw qe;const dt=await kt();
if(dt.status==="done"){Ze=!0;break}if(dt.status==="cancelled")return;if(dt.status==="reselect")continue;
if(it<2){it++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let he=null;try{he=await fe.json()}catch{he=null}if(he===null){const qe=await kt();if(qe.status==="d\
one"){Ze=!0;break}if(qe.status==="cancelled")return;if(qe.status==="reselect")continue;if(fe.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(it<2){it++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!fe.ok&&he.error==="storage_limit_files"&&he.files){const qe=await Qe(he);if(qe===null){oe({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
O.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(O.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}je=qe;continue}if(he&&he.status==="settings_confirmation"&&
Array.isArray(he.settings_changes)){if(!await Lt(he)){oe({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\
\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),O.uploadId&&
apiFetch(`/api/account/import/upload/${encodeURIComponent(O.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}I=!0;continue}if(!fe.ok)throw new Error(he.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F");const Be=he.imported||{},me=[`\u8A2D\u5B9A ${Be.settings||0}\u4EF6`,
`API\u8A8D\u8A3C ${Be.api_credentials||0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Be.chats||0}\u4EF6`,`Ge\
m ${Be.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Be.files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Be.
feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Be.diagnostics||0}\u4EF6`].join(" / "),we=he.duplicates||
{},Oe={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\
\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"},_t=[];for(const qe of Object.keys(Oe)){const dt=Number(
we[qe])||0;dt>0&&_t.push(`${Oe[qe]} ${dt}\u4EF6`)}const Jt=_t.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${_t.
join("\u3001")}\uFF09`:"";G&&(G.textContent=`\u5B8C\u4E86: ${me}${Jt}`,G.classList.remove("hidden","\
text-red-300"),G.classList.add("text-emerald-300")),oe({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\
\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&Pt(),Ze=!0}}catch(R){if(O.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(O.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),O.cancelRequested||R&&R.name==="AbortError")return;const $=R&&R.message?R.
message:"",U=$==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
$||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";oe({progress:0,phase:"\
failed",message:U}),G&&(G.textContent=U,G.classList.remove("hidden","text-emerald-300"),G.classList.
add("text-red-300")),showToast(U,"error",!0)}finally{O.stopped=!0,await se.catch(()=>null),Z(O)}});const Ue=get(
"account-dedupe-btn"),Ve=get("account-dedupe-result"),ot=r((u,f=!1)=>{Ve&&(Ve.textContent=u,Ve.classList.
remove("hidden"),Ve.classList.toggle("text-red-300",!!f),Ve.classList.toggle("text-emerald-300",!f))},
"showDedupeResult");Ue&&(Ue.onclick=async()=>{const u=r(async()=>{const f=await apiFetch("/api/accou\
nt/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})}),
g=await f.json().catch(()=>null);if(!f.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){ot("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],k={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const O of[
"chats","gems","files","feedback","diagnostics"]){const G=Number(g.duplicates&&g.duplicates[O])||0;G>
0&&b.push(`${k[O]} ${G}\u4EF6`)}const T=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${T}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const C=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await C.json().catch(()=>null);if(!C.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const I=[];for(const O of["chats","gems","files","feedback","diagnostics"]){const G=Number(A.removed&&
A.removed[O])||0;G>0&&I.push(`${k[O]} ${G}\u4EF6`)}const D=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";ot(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${I.join("\u3001")||"0\u4EF6"}${D}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!Ue.disabled){Ue.disabled=!0,ot("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await u()}catch(f){ot(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{Ue.disabled=!1}}});const ze=get("site-cache-usage-refresh");ze&&(ze.onclick=()=>loadSiteCacheUsage());
const Fe=get("clear-site-cache-btn");Fe&&(Fe.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Fe)});
const De=get("enc-scan-result"),We=r(async(u=null)=>{De&&(De.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const g=await apiFetch(
f,{cache:"no-store"}),b=await g.json();if(!g.ok){De&&(De.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const k=b.total||0,T=b.encrypted||0,C=b.unencrypted||0;let A=`Total: ${k} / Encrypted: ${T} /\
 Plain: ${C}`;if(b.samples&&b.samples.length){const I=b.samples.slice(0,8).map(D=>{const O=D.timestamp?
new Date(D.timestamp).toLocaleString():"";return`#${D.id} (${D.role||""}) ${O}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${I}</div>`}De&&(De.innerHTML=A)}catch{De&&(De.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Yt=get("enc-scan-all");Yt&&(Yt.onclick=()=>We(
null));const Ft=get("enc-scan-thread");Ft&&(Ft.onclick=()=>currentThreadId?We(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const st=get("admin-enc-\
list");let jt=null,$e=!1;const He=r(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThread\
EncryptedFromMessages"),tt=r(()=>{jt=He(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Qt=r(
async(u,f,{confirmPrompt:g=!0,reloadCurrent:b=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const k=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;$e=!0;try{const T=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
C=await T.json().catch(()=>({}));return T.ok?(showToast(`${k}\u3057\u307E\u3057\u305F\uFF08${C.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),jt=!!f,b&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),st&&await mt(),!0):
(showToast(C.error||`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{$e=!1}},"setAdminThreadEncry\
ption"),dn=r(u=>{if(!st)return;const f=u.threads||[];if(!f.length){st.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
st.innerHTML=f.map(g=>{const b=g.encrypted_count>0?"enc":"plain",k=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",T=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",C=g.
updated_at?new Date(g.updated_at).toLocaleString():"",A=escapeHtml(String(g.thread_id)),I=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${I?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${C} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${T} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${k}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),mt=r(async()=>{if(st){st.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){st.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(dn(f),currentThreadId&&Array.isArray(f.threads)){const g=f.threads.find(b=>String(
b.thread_id)===String(currentThreadId));g&&(jt=!!g.encrypted)}}catch{st.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>mt()),window.__loadAdminEncThreads=
mt,window.__refreshAdminThreadEncState=tt,window.__setAdminThreadEncryption=Qt;const pe=get("encrypt\
ion-status-admin-toggle");pe&&pe.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),st&&(st.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const A=f.getAttribute("data-id");if(!A)return;typeof Dt==
"function"?Dt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=u.target.closest(".admin-enc-toggle");if(!g||$e)return;const b=g.getAttribute("data-i\
d"),k=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const C=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await Qt(b,k,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=C,await mt()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const _e=r(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(g=>{const b=g.status||"new",k=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
T=g.created_at?new Date(g.created_at).toLocaleString():"",C=g.replied_at?new Date(g.replied_at).toLocaleString():
"",A=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${k}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(T)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(b)}</di\
v>
                            <div class="text-xs text-gray-200 mt-2 whitespace-pre-wrap">${escapeHtml(
g.message||"")}</div>
                            <div class="text-[10px] text-gray-500 mt-2">BAN\u7406\u7531: ${escapeHtml(
g.ban_reason||"N/A")}</div>
                            ${g.evidence?`<details class="mt-2"><summary class="text-[10px] text-cya\
n-300 cursor-pointer">\u4E0D\u5BE9\u306A\u5C65\u6B74\uFF08\u8A18\u9332\uFF09\u3092\u8868\u793A</summary><pre class="mt-1 text-[10px] text-gray-300 whitespace-pr\
e-wrap bg-gray-950/70 border border-gray-700 rounded p-2 max-h-60 overflow-auto">${escapeHtml(g.evidence)}\
</pre></details>`:""}
                            <div class="mt-3">
                                <label class="text-[10px] text-gray-400">\u7BA1\u7406\u8005\u8FD4\u4FE1</label>
                                <textarea class="ban-appeal-reply w-full mt-1 bg-gray-800 border bor\
der-gray-700 rounded px-2 py-1 text-[11px] text-gray-100" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(
A)}</textarea>
                                ${A?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
C)}</div>`:""}
                            </div>
                            <div class="mt-2 flex flex-wrap gap-2">
                                <button class="ban-appeal-mark text-[10px] px-2 py-1 bg-gray-700 hov\
er:bg-gray-600 rounded" data-id="${g.id}">\u65E2\u8AAD</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-blue-700 h\
over:bg-blue-600 rounded" data-id="${g.id}" data-status="in_review">\u5BFE\u5FDC\u4E2D</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-green-700 \
hover:bg-green-600 rounded" data-id="${g.id}" data-status="resolved">\u5B8C\u4E86</button>
                                <button class="ban-appeal-status text-[10px] px-2 py-1 bg-red-700 ho\
ver:bg-red-600 rounded" data-id="${g.id}" data-status="rejected">\u5374\u4E0B</button>
                                <button class="ban-appeal-reply-send text-[10px] px-2 py-1 bg-sky-70\
0 hover:bg-sky-600 rounded" data-id="${g.id}">\u8FD4\u4FE1\u9001\u4FE1</button>
                                <button class="ban-appeal-block text-[10px] px-2 py-1 bg-rose-700 ho\
ver:bg-rose-600 rounded" data-id="${g.id}">\u7533\u3057\u7ACB\u3066\u30D6\u30ED\u30C3\u30AF</button>
                            </div>
                        </div>
                    `}).join("")}},"renderBanAppeals"),Je=r(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const k=(await g.json()).unread_count||0;f.textContent=String(k),u&&k>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${k}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Ke=r(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const g=await f.json();_e(g.items||[]),
await Je(!1)}catch{}}},"loadBanAppeals"),nt=r(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await Ke()}catch{}},"markBanAppealsRead"),ft=r(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await Ke()}catch{}},"updateBanAppealStatus"),Mt=r(()=>{
const u=get("tab-general");if(!u||get("temp-chat-settings-card"))return;const f=document.createElement(
"div");f.id="temp-chat-settings-card",f.className="settings-card",f.innerHTML=`
                    <h3 class="settings-card-title">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">\u5207\u65AD\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\uFF08\u79D2\uFF09</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}\
" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rou\
nded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3067\u30DA\u30FC\u30B8\u306E\u8868\u793A/\u63A5\u7D9A\u304C\u9014\u5207\u308C\u305F\u72B6\u614B\u304C\u3053\u306E\u79D2\u6570\u3092\u8D85\u3048\u308B\u3068\u3001\u81EA\u52D5\u524A\
\u9664\u3055\u308C\u307E\u3059\u3002</div>
                    </div>
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),Nn=r(()=>{const u=get("set-st\
t-model");if(!u||get("set-llm-transcribe-prompt"))return;const f=u.closest(".space-y-2");if(!f)return;
const g=document.createElement("div");g.className="pt-2 border-t border-gray-700/60",g.innerHTML=`
                    <label class="text-xs text-gray-500 block">LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\uFF08LLM\u65B9\u5F0F\uFF09</label>
                    <textarea id="set-llm-transcribe-prompt" class="w-full h-24 bg-gray-800 border b\
order-gray-600 rounded px-2 py-2 text-xs text-white mt-1" placeholder=""></textarea>
                    <div class="flex items-center gap-2 mt-2">
                        <button type="button" id="reset-llm-transcribe-prompt" class="bg-gray-700 ho\
ver:bg-gray-600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover">\u65E2\u5B9A\u306B\u623B\u3059</button>
                        <div class="text-[10px] text-gray-500">LLM\u65B9\u5F0F\u306E\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u6642\u306E\u307F\u4F7F\u7528\u3002\u7A7A\u6B04\u3067\u4FDD\u5B58\u3059\u308B\u3068\u65E2\u5B9A\u6587\u9762\u3092\u4F7F\u3044\u307E\u3059\
\uFF08\u7121\u97F3\u6642\u306E\u5B89\u5168\u30AC\u30FC\u30C9\u306F\u5225\u9014\u81EA\u52D5\u4ED8\u4E0E\uFF09\u3002</div>
                    </div>
                `,f.appendChild(g);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const k=get(
"set-llm-transcribe-prompt");k&&(k.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),un=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
{key:"gemini_local_python",label:"Gemini \u97F3\u58F0/\u52D5\u753B/PDF/DOCX + Python\uFF08\u30ED\u30FC\u30AB\u30EB\u5B9F\u884C\uFF09"},
{key:"grok_search",label:"Search\u88DC\u52A9\uFF08Grok\uFF09"},{key:"openai_search",label:"Search\u88DC\u52A9\uFF08\
OpenAI/xAI Responses\uFF09"},{key:"marker",label:"Marker\u7DE8\u96C6\u6642"},{key:"attachment_names",
label:"\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u540D\uFF08LLM\u5165\u529B\u6642\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{\
attachment_names}} / {{attachment_count}}"},{key:"mathjax",label:"MathJax\uFF08LaTeX\u6570\u5F0F\uFF09"},
{key:"image_analysis",label:"\u753B\u50CF\u89E3\u6790\uFF08Vision Model\u6307\u793A\u6587\uFF09"},{key:"\
mcp",label:"MCP\uFF08\u5916\u90E8\u30C4\u30FC\u30EB\u63A5\u7D9A\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{mcp_tools}}\uFF08\u63A5\
\u7D9A\u4E2D\u306EMCP\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u5165\u308A\u307E\u3059\uFF09",mcpLocked:!0}];
window.buildAutoSystemPromptRows=(u,f=!1)=>{const g=f?"w-full h-14 bg-gray-950 border border-gray-70\
0 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 rounded p-2\
 text-xs text-gray-200";return un.map(b=>{const k=b.mcpLocked===!0,T=k?'<div class="text-[10px] text\
-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>':
"",C=k?`<input type="checkbox" id="${u}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${u}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${b.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${k?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${C}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${u}-auto-sys-${b.key}-text" class="${g}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${T}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{un.forEach(g=>{const b=f&&
typeof f=="object"?f[g.key]||{}:{},k=get(`${u}-auto-sys-${g.key}-enabled`),T=get(`${u}-auto-sys-${g.
key}-text`);k&&(g.mcpLocked===!0?k.disabled=!0:k.checked=b.enabled!==!1),T&&(T.value=b.text||"",T.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Rn=r((u,f=null)=>{if(f){const g=get(f);g&&(g.checked=!0)}un.forEach(g=>{const b=get(`${u}-auto\
-sys-${g.key}-enabled`),k=get(`${u}-auto-sys-${g.key}-text`);if(b&&(g.mcpLocked!==!0?b.checked=!0:b.
disabled=!0),k){const T=k.placeholder||"";k.value=T}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),si=r(u=>{const f={};return un.forEach(g=>{const b=get(`${u}\
-auto-sys-${g.key}-enabled`),k=get(`${u}-auto-sys-${g.key}-text`);f[g.key]={enabled:g.mcpLocked===!0?
!0:b?b.checked:!0,text:k?k.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.ensureAutoSystemPromptSettingsCard=
()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.closest(".space-y-4"):null;if(!f||get("auto-\
sys-prompt-settings"))return;const g=document.createElement("div");g.id="auto-sys-prompt-settings",g.
className="border-t border-gray-700 pt-3",g.innerHTML=`
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
                `,f.appendChild(g)},window.ensureThreadAutoSystemPromptCard=()=>{const u=get("thread\
-global-sys-prompt"),f=u?u.closest(".space-y-3"):null;if(!f||get("thread-auto-sys-prompt-settings"))
return;const g=document.createElement("div");g.id="thread-auto-sys-prompt-settings",g.className="bor\
der-t border-gray-700 pt-3",g.innerHTML=`
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
                `,f.appendChild(g)},Mt(),Nn(),de();const ai=r(()=>{const u=get("set-default-model");
if(!u)return;const f=u.value;u.innerHTML="",MODELS.forEach(b=>{const k=document.createElement("optgr\
oup");k.label=b.category,(b.items||[]).forEach(T=>{const C=document.createElement("option");C.value=
T.id,C.textContent=T.name,k.appendChild(C)}),u.appendChild(k)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";g&&Array.from(u.options).some(b=>b.value===g)&&(u.value=g)},"po\
pulateDefaultModelOptions"),oi=r(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(b=>{const k=(b.items||[]).filter(C=>{const A=(C.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")});if(k.length===0)return;const T=document.createElement("optgroup");T.label=b.category,k.forEach(
C=>{const A=document.createElement("option");A.value=C.id,A.textContent=C.name+" \u2605",T.appendChild(
A)}),u.appendChild(T)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";g&&Array.from(u.options).some(b=>b.value===g)&&(u.value=g)},"populateDefaul\
tVisionModelOptions"),Bn=r(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-global-sys-prompt-\
preview");f&&(f.value=u.global_system_prompt_effective||"");const g=get("app-global-sys-prompt-previ\
ew-status");g&&(u.global_system_prompt_enabled===!1?g.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
u.global_system_prompt_uses_time_fallback?g.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
g.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("sys-prompt-text")&&(get("sys-prompt-text").value=u.system_prompt||""),get("set-global-sys-promp\
t-enabled")&&(get("set-global-sys-prompt-enabled").checked=u.system_prompt_enabled!==!1),window.ensureAutoSystemPromptSettingsCard(),
get("set-apply-global-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=u.apply_global_system_prompt!==
!1),get("set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=u.apply_auto_system_prompt_notices!==
!1),window.applyAutoSystemPromptConfigToForm("set",u.auto_system_prompt_notices_config||{}),get("set\
-latency-metrics")&&(get("set-latency-metrics").checked=u.enable_latency_metrics===!0),get("set-clie\
nt-debug-log")&&syncClientDebugLogToggle(u.enable_client_debug_log===!0,"settings modal sync"),get("\
set-openai")&&(get("set-openai").value=u.openai_key||""),get("set-gemini")&&(get("set-gemini").value=
u.gemini_key||""),get("set-deepseek")&&(get("set-deepseek").value=u.deepseek_key||""),get("set-kimi")&&
(get("set-kimi").value=u.kimi_key||""),get("set-mistral")&&(get("set-mistral").value=u.mistral_key||
""),get("set-anthropic")&&(get("set-anthropic").value=u.anthropic_key||""),get("set-gemini-backend")&&
(get("set-gemini-backend").value=normalizeGeminiBackend(u.gemini_backend||"gemini_api")),get("set-ge\
mini-vertex-project")&&(get("set-gemini-vertex-project").value=u.gemini_vertex_project||""),get("set\
-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=u.gemini_vertex_location||"globa\
l"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-gemini\
-vertex-credentials-json").value=u.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),get("se\
t-admin-api-key-mode")&&(get("set-admin-api-key-mode").value=normalizeAdminApiKeyMode(u.admin_api_key_mode||
"env_fallback")),syncAdminApiKeyModeUi(),get("set-xai")&&(get("set-xai").value=u.xai_key||""),get("s\
et-google-key")&&(get("set-google-key").value=u.google_key||""),get("set-google-project")&&(get("set\
-google-project").value=u.google_project||""),modelApiKeyMap=normalizeModelApiKeyMap(u.model_api_keys||
{}),syncModelApiKeyModelOptions(),renderModelApiKeyList(),setModelApiKeyPanelOpen(!1),get("set-mic-t\
ranscribe-mode")&&(get("set-mic-transcribe-mode").value=u.mic_transcribe_mode||"stt_api"),get("set-s\
tt-model")&&(get("set-stt-model").value=u.stt_model||"gpt-4o-mini-transcribe"),get("set-llm-transcri\
be-prompt")&&(get("set-llm-transcribe-prompt").value=u.llm_transcribe_prompt||"",get("set-llm-transc\
ribe-prompt").placeholder=u.llm_transcribe_prompt_default||""),syncRichPastePromptPreferencesUi(u),updateGoogleLinkUI(
u),updateMinashinLinkUI(u),get("set-enter-to-send")&&(get("set-enter-to-send").checked=!!u.enter_to_send),
writePromptBarModeToForm(!!u.compact_prompt_mode,!!u.minimal_prompt_mode),get("set-use-sw-cache")&&(get(
"set-use-sw-cache").checked=!!u.use_sw_cache),get("set-clear-cache-on-version-update")&&(get("set-cl\
ear-cache-on-version-update").checked=!!u.clear_cache_on_version_update),get("set-liquid-glass")&&(get(
"set-liquid-glass").checked=!!u.liquid_glass_enabled),get("set-light-mode")&&(get("set-light-mode").
checked=!!u.light_mode_enabled),get("set-auto-search-links")&&(get("set-auto-search-links").checked=
u.auto_search_on_links!==!1),get("set-use-last-settings")&&(get("set-use-last-settings").checked=!!u.
use_last_chat_settings),get("set-default-model")&&(get("set-default-model").value=u.default_model||"\
gemini-3.6-flash"),get("set-default-vision-model")&&(get("set-default-vision-model").value=u.default_vision_model||
"gemini-3-flash-preview"),applyTemporaryChatTimeoutSeconds(u.temp_chat_timeout_seconds),get("set-def\
ault-search")&&(get("set-default-search").checked=!!u.default_enable_search),get("set-default-url-co\
ntext")&&(get("set-default-url-context").checked=!!u.default_enable_url_context),get("set-default-ma\
ps")&&(get("set-default-maps").checked=!!u.default_enable_maps),get("set-default-python")&&(get("set\
-default-python").checked=!!u.default_enable_python),get("set-default-file-creation")&&(get("set-def\
ault-file-creation").checked=!!u.default_enable_file_creation),get("set-default-thinking")&&(get("se\
t-default-thinking").checked=!!u.default_enable_thinking),get("set-default-sys-prompt")&&(get("set-d\
efault-sys-prompt").checked=!!u.default_enable_system_prompt),get("set-default-mcp")&&(get("set-defa\
ult-mcp").checked=u.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-default-th\
inking-level").value=u.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get("set\
-default-thinking-budget").value=u.default_thinking_budget||4096),get("set-default-reasoning-effort")&&
(get("set-default-reasoning-effort").value=u.default_reasoning_effort||"medium"),get("set-default-sa\
fety")&&(get("set-default-safety").value=u.default_safety_setting||"default"),get("set-e2ee").checked=
u.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=u.bot_detection_enabled!==!1),get(
"set-bot-detect-global")&&(get("set-bot-detect-global").checked=u.bot_detection_global_enabled!==!1);
const b=get("bot-status");b&&(u.is_bot_banned?(b.textContent=`BAN\u4E2D: ${u.bot_ban_reason||"Bot de\
tection"}`,b.classList.remove("hidden"),b.classList.add("text-red-400")):b.classList.add("hidden")),
u&&u.theme_color?(applyThemeColor(u.theme_color,!0),syncThemeInputs(u.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=u.username);const k=get("2fa-badge"),T=get("disable-2fa-btn");u.is_2fa_enabled?
(k.innerText="ENABLED",k.classList.replace("bg-gray-700","bg-green-600"),k.classList.replace("text-g\
ray-400","text-white"),T.classList.remove("hidden")):(k.innerText="DISABLED",k.classList.replace("bg\
-green-600","bg-gray-700"),k.classList.replace("text-white","text-gray-400"),T.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!u.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=u.default_2fa_method||"totp");const C=get(
"set-passkey-only-login"),A=get("passkey-only-note"),I=Array.isArray(u.passkey_credentials)?u.passkey_credentials:
[];if(ee(I),C){C.checked=!!u.passkey_only_login;const R=I.length>0||!!u.has_webauthn;C.disabled=!R,R||
(C.checked=!1),A&&(R?A.classList.add("hidden"):A.classList.remove("hidden"))}const D=get("mig-status\
-box"),O=get("mig-progress-text"),G=get("mig-progress-bar");if((u.migration_status||"idle")==="proce\
ssing"){D.classList.remove("hidden");const R=(u.migration_progress||"").split("/");if(R.length===2){
const $=parseInt(R[0]||"0",10),U=parseInt(R[1]||"0",10);O&&(O.innerText=`${$} / ${U}`),G&&U>0&&(G.style.
width=`${Math.min(100,Math.floor($/U*100))}%`)}}else D.classList.add("hidden"),G&&(G.style.width="0%"),
O&&(O.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettingsFormFromData");
window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),snapshotSidebarHistory(
"settings-open-before");const u=await ensureUserSettingsSnapshot();u&&Bn(u);const f=get("search-box"),
g=f?f.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=""),filterSettings(),
ai(),oi(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(()=>refreshSettingsTabsScroll()),
restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-open-after"),[50,200,400,800].forEach(k=>{setTimeout(()=>{restoreThreadSearchValue(g,"rest\
ored-search-box-"+k+"ms"),snapshotSidebarHistory("settings-open-later-"+k+"ms")},k)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),W(),Nn(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),Je(!0),Ke(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),Tn(),V(),B();try{loadMcpServers()}catch{}};const Dt=r((u=!1)=>{snapshotSidebarHistory("\
settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!u&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),ri=r(()=>{const u=get("set-th\
eme-color"),f=get("set-theme-color-text"),g=get("theme-reset-btn"),b=document.querySelectorAll("#the\
me-presets .theme-swatch"),k=r((T,C=!0)=>{const A=normalizeHex(T);A&&(applyThemeColor(A,C),syncThemeInputs(
A))},"applyFromValue");u&&u.addEventListener("input",()=>k(u.value,!0)),f&&(f.addEventListener("chan\
ge",()=>{const T=normalizeHex(f.value);if(!T){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}k(T,!0)}),f.addEventListener("keydown",T=>{T.key==="Enter"&&(T.preventDefault(),
f.blur())})),g&&(g.onclick=()=>k(THEME_DEFAULT,!0)),b.forEach(T=>{T.addEventListener("click",()=>k(T.
getAttribute("data-color"),!0))})},"bindThemeControls"),li=r(()=>{const u=get("reset-global-sys-prom\
pt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const f=get(
"reset-set-auto-sys-prompt-defaults");f&&(f.onclick=()=>{Rn("set","set-apply-auto-sys-prompt-notices"),
showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-thread-auto-sys-prompt-defaults");g&&(g.onclick=()=>{Rn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Dt();const Fn=get("settings-header-close");Fn&&(Fn.onclick=()=>Dt());
const Ht=get("settings-search");Ht&&(Ht.addEventListener("input",filterSettings),Ht.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const g=f.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const jn=get(
"settings-search-clear");jn&&jn.addEventListener("click",()=>{Ht&&(Ht.value="",filterSettings(),Ht.focus())}),
ri(),li(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const wn=get("set-gemini-local-python-dialog");wn&&(wn.onchange=()=>setGeminiLocalPyDialogEnabled(wn.
checked));const Dn=get("set-gemini-backend");Dn&&(Dn.onchange=()=>syncGeminiBackendUi());const Hn=get(
"set-admin-api-key-mode");Hn&&(Hn.onchange=()=>syncAdminApiKeyModeUi());const xn=get("set-temp-chat-\
timeout-seconds");xn&&(xn.onchange=()=>{applyTemporaryChatTimeoutSeconds(xn.value)});const qn=get("s\
lash-command-cancel-btn");qn&&(qn.onclick=()=>{hidePendingSlashCommandIndicator();const u=get("promp\
t-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),g=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:si(
"set"),theme_color:normalizeHex(get("set-theme-color-text")?get("set-theme-color-text").value:"")||THEME_DEFAULT,
light_mode_enabled:get("set-light-mode")?get("set-light-mode").checked:!1,mic_transcribe_mode:get("s\
et-mic-transcribe-mode")?get("set-mic-transcribe-mode").value:"stt_api",stt_model:get("set-stt-model")?
get("set-stt-model").value:null,llm_transcribe_prompt:get("set-llm-transcribe-prompt")?get("set-llm-\
transcribe-prompt").value:"",enter_to_send:get("set-enter-to-send")?get("set-enter-to-send").checked:
!1,compact_prompt_mode:g.compact_prompt_mode,minimal_prompt_mode:g.minimal_prompt_mode,use_sw_cache:get(
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
-method").value:"totp",new_username:u?u.value:null,new_password:f?f.value:null},k=get("set-e2ee")?get(
"set-e2ee").checked:!1,T=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
k!==T&&(b.enable_e2ee=k),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
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
lobal").checked);const C=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(b)});if(C.ok){let A="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const G=await C.json();G&&G.message&&(A=G.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),Dt();
const I=currentUsername,D=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;
const O=useSwCache;useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=
!!b.clear_cache_on_version_update),compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,
voiceStudioUiEnabled=b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,
applyThemeColor(b.theme_color,!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),
applyLiquidGlassMode(b.liquid_glass_enabled),applyAdaptiveBlurPreference(get("set-background-blur-mo\
de")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),minimalPromptMode?setMinimalPromptMode(
!0):setCompactPromptMode(compactPromptMode),updateStsUi(),O!==useSwCache&&applyCacheMode(useSwCache,
{forceCleanup:!useSwCache}),showToast(A,"success"),syncClientDebugLogToggle(b.enable_client_debug_log,
"settings saved"),b.new_username&&b.new_username!==I?setTimeout(()=>location.reload(),1e3):b.new_password&&
showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let A={};try{A=await C.json()}catch{}showToast(A.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const f=get("2fa-badge");f&&(f.innerText="D\
ISABLED",f.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const u=get("bot-unban-username"),f=u?u.value.trim():
"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"single"})}),b=await g.json(),k=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")k&&(k.textContent=`${f} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
k.classList.remove("hidden")),u&&(u.value="");else{const T=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(T,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const u=get("bot-unban-username"),f=u?u.value.trim():"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"linked"})}),b=await g.json(),k=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")k&&(k.textContent=`${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
k.classList.remove("hidden")),u&&(u.value="");else{const T=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(T,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const u=get(
"bot-speed-test-btn"),f=get("bot-speed-test-result");u&&(u.disabled=!0),u&&u.classList.add("opacity-\
60","cursor-not-allowed"),f&&(f.classList.remove("hidden"),f.textContent="\u5B9F\u884C\u4E2D...");try{
const g=r(X=>{f&&(f.textContent=X)},"setBox"),b=r(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),k=r((X,re)=>!X||!re||re<=0?0:X*8/(re/1e3)/1e3/1e3,"toMbps"),T=r(X=>Number.isFinite(
X)?`${X.toFixed(0)} ms`:"-","fmtMs"),C=r(X=>Number.isFinite(X)?`${X.toFixed(X>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=r(async(X,re)=>{const Te=await X.json().catch(()=>({}));return Te&&Te.error?Te.error:re},
"parseErr"),I=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let X=0;X<4;X++){const re=performance.now(),Te=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),Re=performance.now();if(!Te.ok)throw new Error(await A(
Te,"ping_failed"));await Te.json().catch(()=>({})),I.push(Re-re)}const D=I.reduce((X,re)=>X+re,0)/Math.
max(1,I.length),O=Math.min(...I),G=r(async X=>{const re=performance.now(),Te=await apiFetch(`/api/sp\
eedtest/download?bytes=${X}&_=${b()}`,{cache:"no-store"});if(!Te.ok)throw new Error(await A(Te,"down\
load_failed"));const Re=await Te.arrayBuffer(),je=performance.now();return{bytes:Re.byteLength||X,ms:je-
re,mbps:k(Re.byteLength||X,je-re)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${T(D)}
\u6E2C\u5B9A\u4E2D... download`);const se=[];for(const X of[2*1024*1024,8*1024*1024])se.push(await G(
X)),g(`\u6E2C\u5B9A\u4E2D... ping ${T(D)}
download ${C(Math.max(...se.map(re=>re.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const R=Math.max(...se.map(X=>X.mbps)),$=r(async X=>{const re=new Uint8Array(
X),Te=performance.now(),Re=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:re,cache:"no-store"}),je=performance.now();if(!Re.ok)
throw new Error(await A(Re,"upload_failed"));const Ze=await Re.json().catch(()=>({})),it=Number(Ze.bytes_received||
X)||X;return{bytes:it,ms:je-Te,mbps:k(it,je-Te),serverMs:Number(Ze.server_elapsed_ms||0)||0}},"runUp\
load"),U=[];for(const X of[1*1024*1024,4*1024*1024])U.push(await $(X));const K=Math.max(...U.map(X=>X.
mbps)),ke=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",`Ping\
 (avg/min): ${T(D)} / ${T(O)}`,`Download (best): ${C(R)}`,`Upload (best): ${C(K)}`,`Download runs: ${se.
map(X=>`${Math.round(X.bytes/1024/1024)}MB=${C(X.mbps)}`).join(", ")}`,`Upload runs: ${U.map(X=>`${Math.
round(X.bytes/1024/1024)}MB=${C(X.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\
\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];g(ke.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Ke()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>nt()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const g=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){g&&await nt([Number(g)]);return}if(f.classList.contains(
"ban-appeal-status")){const b=f.getAttribute("data-status");g&&b&&await ft({id:Number(g),status:b});
return}if(f.classList.contains("ban-appeal-reply-send")){const b=f.closest("[data-appeal-id]"),k=b?b.
querySelector(".ban-appeal-reply"):null,T=k?k.value:"";g&&await ft({id:Number(g),admin_reply:T});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await ft({id:Number(
g),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const u=get("photo-input");u&&u.click()}),get("upload-clear-b\
tn")&&(get("upload-clear-btn").onclick=()=>{resetUploadState()}),get("marker-modal-close")&&(get("ma\
rker-modal-close").onclick=()=>{closeMarkerModal(),markerState.row=null}),get("marker-tool-draw")&&(get(
"marker-tool-draw").onclick=()=>setMarkerMode("draw")),get("marker-tool-mosaic")&&(get("marker-tool-\
mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool-crop")&&(get("marker-tool-crop").onclick=
()=>setMarkerMode("crop"));const kn=get("marker-color-picker");kn&&(kn.oninput=u=>setMarkerColor(u.target.
value),kn.onchange=u=>setMarkerColor(u.target.value));const _n=get("marker-opacity");_n&&(_n.oninput=
u=>setMarkerOpacity(u.target.value),_n.onchange=u=>setMarkerOpacity(u.target.value));const pn=get("m\
arker-opacity-number");pn&&(pn.onchange=u=>setMarkerOpacity(u.target.value),pn.onblur=u=>setMarkerOpacity(
u.target.value),pn.onkeydown=u=>{u.key==="Enter"&&(setMarkerOpacity(u.target.value),u.target.blur())}),
document.querySelectorAll("#marker-toolbar .marker-color-chip[data-marker-color]").forEach(u=>{u.onclick=
()=>setMarkerColor(u.getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-vie\
w-reset").onclick=()=>resetMarkerTransform()),get("marker-crop-reset")&&(get("marker-crop-reset").onclick=
()=>clearCropRect()),get("marker-undo")&&(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("ma\
rker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),get("marker-save")&&(get("marker\
-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),initCropCanvas(),
window.addEventListener("resize",()=>{const u=get("marker-modal");!u||u.classList.contains("hidden")||
(applyMarkerTransform(),renderCropOverlay())});const ci=r(()=>{const u=get("upload-modal");return!!(u&&
!u.classList.contains("hidden"))},"isUploadModalOpen"),qt=get("drop-overlay");let Zt=0;const di=r(()=>{
ci()||qt&&(qt.classList.remove("hidden"),qt.classList.add("flex"))},"showDropOverlay"),en=r(()=>{Zt=
0,qt&&(qt.classList.add("hidden"),qt.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
en;const yt=get("upload-dropzone");yt&&(yt.addEventListener("dragover",u=>{u.preventDefault(),yt.classList.
add("dragover")}),yt.addEventListener("dragleave",()=>{yt.classList.remove("dragover")}),yt.addEventListener(
"drop",u=>{u.preventDefault(),u.stopPropagation(),yt.classList.remove("dragover"),en();const f=u.dataTransfer?
u.dataTransfer.files:null;f&&f.length&&handleFiles(f)})),window.addEventListener("dragenter",u=>{!u.
dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(Zt+=1,di())}),window.
addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes(
"Files")||u.preventDefault()}),window.addEventListener("dragleave",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||(Zt=Math.max(0,Zt-1),(Zt===0||!u.relatedTarget||u.clientY<=
0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=window.innerHeight)&&en())}),window.addEventListener(
"dragend",()=>{en()}),window.addEventListener("drop",u=>{en(),!(!u.dataTransfer||!u.dataTransfer.files||
u.dataTransfer.files.length===0)&&(u.preventDefault(),!(yt&&yt.contains(u.target))&&handleFiles(u.dataTransfer.
files))});const Gn=get("bot-admin-modal"),ui=r(u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML=
"",!u||!u.length){f.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((g,b)=>{const k=!!g.is_bot_banned,T=g.bot_detection_enabled!==!1,C=document.createElement(
"div");C.className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs m\
odel-list-animate",C.style.animationDelay=`${Math.min(b,12)*.02}s`,C.innerHTML=`
                        <div class="flex-1">
                            <div class="text-gray-200 font-bold">${escapeHtml(g.username)}</div>
                            <div class="text-[10px] text-gray-500">${k?"BAN\u4E2D":"\u6B63\u5E38"} ${g.
bot_ban_reason?" / "+escapeHtml(g.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${T?"1":"0"}">${T?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${k?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-bann\
ed="${k?"1":"0"}">${k?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${k?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(g.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-\
2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(g.username)}">\u524A\u9664</button>\

                    `,f.appendChild(C)})}},"renderBotUsers"),tn=r(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),b=await g.json();g.ok&&b&&b.users?ui(b.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{f&&(f.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Sn=r(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Gn))return;const f=get(
"settings-modal");f&&(f.classList.contains("modal-open")||f.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await tn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Sn,window.closeBotAdminModal=(u=!1)=>{(get(
"bot-admin-modal")||Gn)&&hideModal("bot-admin-modal"),!u&&location.pathname==="/admin-bots"&&history.
back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{Sn()}),get("bot-admin-close")&&(get(
"bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-admin-search-btn")&&(get("bot-admin-se\
arch-btn").onclick=async()=>{await tn(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}),
get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=async()=>{await tn("")}),get("bo\
t-admin-search")&&get("bot-admin-search").addEventListener("keydown",async u=>{u.key==="Enter"&&await tn(
get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get("bot-admin-list").onclick=async u=>{
const f=u.target.closest("button");if(!f)return;const g=f.getAttribute("data-user");if(!g)return;let b;
if(f.classList.contains("bot-toggle-detect")){const k=f.getAttribute("data-enabled")!=="1";b=await apiFetch(
"/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,
action:"toggle_detection",enabled:k})})}else if(f.classList.contains("bot-toggle-ban"))if(f.getAttribute(
"data-banned")==="1")b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"appl\
ication/json"},body:JSON.stringify({username:g,action:"unban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))return;b=await apiFetch("/api/bot/update",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,action:"ban",reason:"Adm\
in ban"})})}else if(f.classList.contains("bot-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))return;b=await apiFetch("/a\
pi/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,
action:"unban_linked"})})}else if(f.classList.contains("bot-delete-account")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"delete_account"})})}if(b){if(b.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(b.ok){if(f.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),g===currentUsername)){location.href="/";return}}else{
let k={};try{k=await b.json()}catch{}showToast(k.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}await tn(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}});const Gt={"\
/settings":{id:"settings-modal",open:r(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload\
-modal",open:r(()=>openUploadModal(),"open")},"/library":{id:"lib-modal",open:r(()=>{Zn(!1),showModal(
"lib-modal"),loadLibraryFiles()},"open")},"/history":{id:"history-modal",open:r(()=>window.showHistoryModal(),
"open")},"/branch":{id:"branch-modal",open:r(()=>window.showBranchModal(),"open")},"/batch":{id:"bat\
ch-modal",open:r(()=>window.showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:r(()=>openRichPasteModal(),
"open")},"/camera":{id:"camera-capture-modal",open:r(()=>openCameraCaptureModal(),"open")},"/edit-im\
age":{id:"marker-modal",open:r(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:r(()=>window.
openThreadModal(),"open")},"/model":{id:"model-modal",open:r(()=>openModelModal(),"open")},"/token-d\
etails":{id:"token-detail-modal",open:r(()=>showTokenDetailModal(),"open")},"/encryption-status":{id:"\
encryption-status-modal",open:r(()=>showEncryptionStatusModal(),"open")},"/python-execution":{id:"py\
thon-exec-modal",open:r(()=>showPythonExecDetailModal(),"open")},"/gem":{id:"gem-modal",open:r(()=>{
editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>C\
reate New Gem',showModal("gem-modal")},"open")},"/compression":{id:"compression-modal",open:r(()=>window.
openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:r(()=>Sn(),"open")}},Un=r((u,f=!1)=>{
switch(u){case"settings-modal":Dt(f);break;case"upload-modal":closeUploadModal(f);break;case"camera-\
capture-modal":closeCameraCaptureModal(f?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(f);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(f);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(f);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(f);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
f);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(f);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(f);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(f);break;case"token-detail-modal":closeTokenDetail(f);break;case"encryption-s\
tatus-modal":closeEncryptionModal(f);break;case"python-exec-modal":closePythonExecDetail(f);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(f);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(f);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
f);break;case"version-update-modal":const g=localStorage.getItem("app_version")||"";g&&localStorage.
setItem("version_notified",g),hideModal(u);break;default:hideModal(u);break}},"closeModalById");window.
addEventListener("popstate",u=>{let f=!1;Object.values(Gt).forEach(k=>{const T=get(k.id);T&&T.classList.
contains("modal-open")&&location.pathname!==Object.keys(Gt).find(C=>Gt[C].id===k.id)&&(Un(k.id,!0),f=
!0)});const g=location.pathname.match(/^\/c\/(.+)$/);if(g){const k=decodeURIComponent(g[1]);String(currentThreadId)!==
String(k)&&loadMessages(k,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat(
{skipHistory:!0});const b=Gt[location.pathname];if(b){const k=get(b.id);k&&!k.classList.contains("mo\
dal-open")&&b.open()}});const zn=location.pathname;Gt[zn]&&(history.replaceState({},"","/"),setTimeout(
()=>Gt[zn].open(),500)),get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const u=get(
"easy-login-mins"),f=u?parseInt(u.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${f}\
\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const b=await(await apiFetch("\
/api/easy_login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:f})})).
json();b&&b.temp_password?(get("easy-login-code").textContent=b.temp_password,get("easy-login-exp").
textContent=b.expires_at||"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\
\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}),get("easy-login-cancel")&&(get(
"easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(f&&f.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("fb-submit").onclick=async()=>{const u=get("fb-title").value.trim(),f=get("fb-mess\
age").value.trim();if(!f){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}await apiFetch("/api/feedback",{method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({title:u,message:f})}),get("fb-title").value="",get("fb-message").value=
"",Tn()};async function Tn(){const f=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-l\
ist");g.innerHTML="",(f.items||[]).filter(T=>!f.is_admin||T.user_id===void 0||T.user_id===null||!0).
forEach(T=>{if(f.is_admin)return;const C=document.createElement("div");C.className="p-2 rounded bord\
er border-gray-700 bg-gray-800/50",C.innerHTML=`<div class="text-[11px] text-gray-400">${T.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(T.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(T.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
T.status)}</div>${T.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(T.
admin_reply)}</div>`:""}`,g.appendChild(C)});const b=get("fb-admin-panel"),k=get("fb-admin-list");f.
is_admin?(b.classList.remove("hidden"),k.innerHTML="",(f.items||[]).forEach(T=>{const C=document.createElement(
"div");C.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",C.innerHTML=`
                            <div class="text-[11px] text-gray-400">#${T.id} / user:${T.user_id} / ${T.
created_at}</div>
                            <div class="font-bold text-sm">${escapeHtml(T.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(T.message)}</div>
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
ded px-2 py-1 text-xs text-white" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(T.admin_reply||
"")}</textarea>
                        `,C.querySelector(".fb-status").value=T.status||"new",C.querySelector(".fb-s\
ave").onclick=async()=>{const A=C.querySelector(".fb-status").value,I=C.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${T.id}/update`,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({status:A,admin_reply:I})}),Tn()},k.appendChild(C)})):b.classList.add("hidd\
en")}if(r(Tn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/2fa/totp/\
setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=f.secret,
get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get("totp-ver\
ify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),g=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),k=await b.json();if(!b.ok){showToast(k.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const C=await(await ensureWebAuthnJson()).create({publicKey:k}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},C,{name:g}))}),I=await A.json().catch(()=>({}));A.ok?(f&&(f.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(I.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(b){showToast(`WebAuthn Error: ${b}`,"error",!0)}finally{u&&(u.disabled=!1)}},window.
removeWebAuthnCredential=async u=>{if(!u||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:u})}),g=await f.json().catch(()=>({}));if(f.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(g.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let u;try{u=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(u.ok){location.href="/";
return}let f={};try{f=await u.json()}catch{}if(f&&f.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(f.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=u=>{if(u.isComposing)return;const f=get("prompt-input");
if(slashSuggestionsVisible){const g=get("slash-command-suggestions");if(u.key==="ArrowDown"){u.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(f.value),f.value));
return}if(u.key==="ArrowUp"){u.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(f.value),f.value));
return}if(u.key==="Enter"){u.preventDefault();const b=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(f.value),f.value));b[slashSelectedIndex]?selectSlashCommand(b[slashSelectedIndex].
id):b.length>0&&selectSlashCommand(b[0].id);return}if(u.key==="Escape"){u.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const g=f.value.trim();if(u.key==="ArrowDown"){u.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(g.substring(1));return}if(u.key==="ArrowUp"){
u.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(g.substring(1));
return}if(u.key==="Enter"){u.preventDefault();const b=g.substring(1).toLowerCase(),k=loadedGems.filter(
T=>T.name.toLowerCase().includes(b)||T.description&&T.description.toLowerCase().includes(b));k[gemSelectedIndex]?
selectGemSuggestion(k[gemSelectedIndex]):k.length>0&&selectGemSuggestion(k[0]);return}if(u.key==="Es\
cape"){u.preventDefault(),hideGemSuggestions();return}}if(u.key==="Escape"&&pendingSlashCommand){u.preventDefault(),
hidePendingSlashCommandIndicator();return}u.key==="ArrowUp"&&(f.selectionStart===0||u.ctrlKey)?promptHistory.
length>0&&(historyIndex===-1&&(tempPrompt=f.value),historyIndex<promptHistory.length-1&&(u.preventDefault(),
historyIndex++,f.value=promptHistory[historyIndex],f.dispatchEvent(new Event("input")))):u.key==="Ar\
rowDown"&&(f.selectionEnd===f.value.length||u.ctrlKey)&&historyIndex>-1&&(u.preventDefault(),historyIndex--,
historyIndex===-1?f.value=tempPrompt:f.value=promptHistory[historyIndex],f.dispatchEvent(new Event("\
input"))),enterToSend?u.key==="Enter"&&!u.shiftKey&&(u.preventDefault(),sendMessage()):(u.metaKey||u.
ctrlKey)&&u.key==="Enter"&&(u.preventDefault(),sendMessage())},get("prompt-input")&&(get("prompt-inp\
ut").addEventListener("input",function(){this.style.height="auto",this.style.height=this.scrollHeight+
"px",schedulePromptTokenEstimate(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const u=this.
value.trim();if(pendingSlashCommand)gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&
hideSlashCommandSuggestions(),lastSlashFilter=null;else if(u.startsWith("@")){const f=u.substring(1);
showGemSuggestions(f),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=null}else if(u.
startsWith("/")){const f=slashCommandSuggestionFilter(extractSlashCommandToken(u),this.value);(!slashSuggestionsVisible||
f!==lastSlashFilter)&&(lastSlashFilter=f,showSlashCommandSuggestions(f)),gemSuggestionsVisible&&hideGemSuggestions()}else
gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=
null}),get("prompt-input").addEventListener("blur",()=>{setTimeout(()=>{slashSuggestionsVisible&&hideSlashCommandSuggestions(),
gemSuggestionsVisible&&hideGemSuggestions()},150)})),get("cancel-edit-btn")&&(get("cancel-edit-btn").
onclick=cancelEdit),updatePromptPlaceholder(),aiSettingsConversation.length>0&&(pendingSlashCommand=
"settings",showPendingSlashCommandIndicator("settings")),get("search-box")&&(get("search-box").addEventListener(
"input",u=>{const f=get("search-box");if(f&&isUserInitiatedSearchInput(u))markThreadSearchUserEdited(
f);else if(f&&!f.dataset.userEdited){discardAutofilledThreadSearch("cleared-autofill-search-box-inpu\
t");return}if(isSettingsModalOpen()){snapshotSidebarHistory("ignore-search-input-settings-open");return}
clearTimeout(searchTimeout),searchTimeout=setTimeout(()=>{loadThreads(!1)},300)}),hardenThreadSearchInputs()),
get("mobile-new-chat-btn")&&(get("mobile-new-chat-btn").onclick=()=>startNewChat()),get("sts-mic-btn")&&
(get("sts-mic-btn").onclick=()=>{isStsModel()&&get("mic-btn").click()}),get("sts-cancel-btn")&&(get(
"sts-cancel-btn").onclick=()=>{isStsModel()&&Yn()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async u=>{const f=(u.clipboardData||window.clipboardData).items,g=[];for(let b=0;b<f.length;b++)
if(f[b].kind==="file"){const k=f[b].getAsFile();k&&g.push(k)}g.length>0&&(u.preventDefault(),await handleFiles(
g,{openModal:!1}))}),get("rich-paste-btn")&&(get("rich-paste-btn").onclick=()=>openRichPasteModal()),
get("rich-paste-modal-close")&&(get("rich-paste-modal-close").onclick=()=>closeRichPasteModal()),get(
"rich-paste-close-btn")&&(get("rich-paste-close-btn").onclick=()=>closeRichPasteModal()),get("rich-p\
aste-focus-btn")&&(get("rich-paste-focus-btn").onclick=()=>focusRichPasteEditor()),get("rich-paste-c\
lear-btn")&&(get("rich-paste-clear-btn").onclick=()=>clearRichPasteEditor(!0)),get("rich-paste-previ\
ew-btn")&&(get("rich-paste-preview-btn").onclick=()=>openRichPastePreviewTab()),get("rich-paste-send\
-btn")&&(get("rich-paste-send-btn").onclick=()=>sendRichPasteToModel()),get("rich-paste-send-server-\
btn")&&(get("rich-paste-send-server-btn").onclick=()=>sendRichPasteToModel({serverSide:!0})),get("ri\
ch-paste-import-btn")&&(get("rich-paste-import-btn").onclick=async()=>{try{await readClipboardRichContent()||
showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u30EA\u30C3\u30C1\u30C6\u30AD\u30B9\u30C8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002Ctrl+V \u3067\u8CBC\u308A\u4ED8\u3051\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(u){const f=u&&u.message?u.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(f,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const u=get("rich-paste-capture");
u.addEventListener("paste",async f=>{const g=f.clipboardData||window.clipboardData;if(g){f.preventDefault();
try{await ingestRichPasteClipboardData(g)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),u.addEventListener("input",()=>{u.value=""})}get("chat-container").addEventListener(
"click",u=>{const f=u.target.closest("img.chat-image"),g=f?f.dataset.viewerSrc||f.currentSrc||f.src:
"";f&&g&&(u.preventDefault(),openImageViewer(g))});const nn=document.querySelector(".viewer-content");
nn&&(nn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),nn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),nn.addEventListener("touchend",onViewerTouchEnd),nn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",u=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(u.target.id==="image-viewer"||u.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",u=>{u.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",u=>{u.key==="Escape"&&closeImageViewer()});
let Xe,Pe=null,mn=[],Cn=!1,fn=null,Ut=null,At=null,sn=null,Ln=0,hn=!1,an=null,zt=null,xt=null,on=null,
gn=null,Et=null,bn=null,yn=null;function Vn(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray(
bn)&&bn.length)return bn;u.innerHTML="";const f=[];for(let g=0;g<24;g++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",f.push(b),u.appendChild(b)}return bn=f,f}r(Vn,"ensureMicWaveformBars");function $t(u,f="hidden"){
const g=get("mic-recording-indicator"),b=get("mic-recording-text");if(g){if(yn&&(clearTimeout(yn),yn=
null),f==="hidden"){g.classList.add("hidden");return}b&&u&&(b.innerText=u),g.classList.remove("hidde\
n"),f==="recording"?g.style.color="rgb(252 165 165)":f==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}r($t,"setMicRecordingIndicator");function Wn(){Vn().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}r(Wn,"resetMicWaveformBars");function It(){if(gn&&(cancelAnimationFrame(
gn),gn=null),on){try{on.disconnect()}catch{}on=null}if(zt){try{zt.close()}catch{}zt=null}xt=null,Et=
null,Wn()}r(It,"stopMicWaveform");function Jn(u){It();const f=Vn();if(!f.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{zt=new g,xt=zt.createAnalyser(),xt.fftSize=
256,xt.smoothingTimeConstant=0,on=zt.createMediaStreamSource(u),on.connect(xt),Et=new Uint8Array(xt.
frequencyBinCount)}catch{It();return}const b=r(()=>{if(!xt||!Et)return;xt.getByteFrequencyData(Et);const k=Math.
max(1,Math.floor(Et.length/f.length));for(let T=0;T<f.length;T++){const A=(Et[Math.min(Et.length-1,T*
k)]||0)/255,I=Math.max(2,Math.round(2+A*10));f[T].style.height=`${I}px`,f[T].style.opacity=`${.35+A*
.65}`}gn=requestAnimationFrame(b)},"render");b()}r(Jn,"startMicWaveform");function vn(){if(fn&&(clearInterval(
fn),fn=null),sn){try{sn.disconnect()}catch{}sn=null}if(Ut){try{Ut.close()}catch{}Ut=null}At=null}r(vn,
"stopSilenceMonitor");function Kn(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;vn();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;Ut=new f,At=Ut.createAnalyser(),At.fftSize=2048,
sn=Ut.createMediaStreamSource(u),sn.connect(At);const g=new Uint8Array(At.fftSize),b=getStsSilenceMs(),
k=.02;Ln=0,hn=!1,fn=setInterval(()=>{if(!At)return;At.getByteTimeDomainData(g);let T=0;for(let A=0;A<
g.length;A++){const I=(g[A]-128)/128;T+=I*I}if(Math.sqrt(T/g.length)>k){hn||(hn=!0),Ln=Date.now();return}
hn&&Date.now()-Ln>b&&Xe&&Xe.state==="recording"&&Xe.stop()},200)}r(Kn,"startSilenceMonitor");const En=class En{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null}async start(f,g,b,k={}){this.model=b,this.ws=new WebSocket(`${g}?access_token=${f}`),
this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Send\
ing setup...");const A=!!(k&&k.transcriptionConfig),I={setup:{model:`models/${b}`,generationConfig:{
responseModalities:A?["TEXT"]:["AUDIO"]},inputAudioTranscription:A?k.transcriptionConfig||{}:{},outputAudioTranscription:{}}};
k.speechConfig&&(I.setup.generationConfig.speechConfig=k.speechConfig),k.thinkingConfig&&(I.setup.generationConfig.
thinkingConfig=k.thinkingConfig),k.translationConfig&&(I.setup.translationConfig=k.translationConfig),
console.log("Sending setup:",JSON.stringify(I)),this.ws.send(JSON.stringify(I))},this.ws.onmessage=A=>this.
_handleMessage(A),this.ws.onerror=A=>{console.error("Gemini Live WebSocket error:",A),this.onError&&
this.onError(A)},this.ws.onclose=A=>{console.log("Gemini Live WebSocket closed:",A.code,A.reason),this.
onClose&&this.onClose(A)},this.audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
this.stream=await navigator.mediaDevices.getUserMedia({audio:!0});const T=this.audioContext.createMediaStreamSource(
this.stream);this.processor=this.audioContext.createScriptProcessor(4096,1,1),this.userAudioChunks=[];
const C=new MediaRecorder(this.stream);C.ondataavailable=A=>{A.data.size>0&&this.userAudioChunks.push(
A.data)},C.start(500),this.backupRecorder=C,this.processor.onaudioprocess=A=>{if(!this.ws||this.ws.readyState!==
WebSocket.OPEN||!this.setupComplete)return;const I=A.inputBuffer.getChannelData(0),D=new Int16Array(
I.length);for(let O=0;O<I.length;O++)D[O]=Math.max(-1,Math.min(1,I[O]))*32767;this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(D.buffer))),mimeType:"\
audio/pcm;rate=16000"}}}))},T.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(f){
const g=JSON.parse(f.data);if(console.log("Gemini Live raw message received:",g),g.setupComplete&&(console.
log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){const b=g.serverContent;
b.modelTurn&&b.modelTurn.parts.forEach(k=>{if(k.text&&(k.thought?(console.log("Gemini thought delta:",
k.text),this.assistantThought+=k.text):(console.log("Gemini transcript delta (parts):",k.text),this.
assistantText+=k.text)),k.inlineData&&k.inlineData.data){const T=k.inlineData.data;console.log("Gemi\
ni audio chunk received, size:",T.length),this.rtPlayer&&this.rtPlayer.addChunk(T);const C=atob(T),A=new Uint8Array(
C.length);for(let I=0;I<C.length;I++)A[I]=C.charCodeAt(I);this.assistantAudioChunks.push(A)}}),b.outputTranscription&&
(console.log("Gemini output transcription delta:",b.outputTranscription.text),this.assistantText.includes(
b.outputTranscription.text)||(this.assistantText+=b.outputTranscription.text)),b.inputTranscription&&
(console.log("User input transcription delta:",b.inputTranscription.text),this.inputTranscript+=b.inputTranscription.
text,this.interimInputTranscript=""),b.interimInputTranscription&&(console.log("User interim transcr\
iption:",b.interimInputTranscription.text),this.interimInputTranscript=b.interimInputTranscription.text)}
this.onMessage&&this.onMessage(g)}stop(){this.ws&&this.ws.close(),this.processor&&this.processor.disconnect(),
this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().forEach(f=>f.stop()),
this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const f=new Blob(this.assistantAudioChunks),
g=await this._blobToBase64(f),b=new Blob(this.userAudioChunks),k=await this._blobToBase64(b);return{
user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.assistantThought,
audio_base64:g,user_audio_base64:k}}_blobToBase64(f){return new Promise(g=>{const b=new FileReader;b.
onloadend=()=>g(b.result.split(",")[1]),b.readAsDataURL(f)})}};r(En,"GeminiLiveClient");let Mn=En;const $n=class $n{constructor(f=24e3){
const g=window.AudioContext||window.webkitAudioContext;this.ctx=new g({sampleRate:f}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(f){if(!this.ctx)return;const g=atob(f),b=new Uint8Array(
g.length);for(let D=0;D<g.length;D++)b[D]=g.charCodeAt(D);const k=new Int16Array(b.buffer),T=new Float32Array(
k.length);for(let D=0;D<k.length;D++)T[D]=k[D]/32768;const C=this.ctx.createBuffer(1,T.length,this.ctx.
sampleRate);C.getChannelData(0).set(T),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.
ctx.createBufferSource();A.buffer=C,A.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const I=Math.max(this.ctx.currentTime,this.nextStartTime);
A.start(I),this.nextStartTime=I+C.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r($n,
"RealTimeAudioPlayer");let rn=$n;const In=class In{constructor(){this.active=!1,this.capturing=!1,this.
sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=
null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript=
"",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=
null,this.saved=!1,this.saving=!1,this.stopping=!1}isActive(){return this.active}async start(){if(this.
active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const f=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const k=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=k.id!==null&&k.id!==void 0?String(k.id):k.id,setTemporaryChatUiState(!!(k&&k.
is_temporary)),setCurrentChatHeaderTitle(k&&k.title),applyTemporaryChatRuntimeMeta(k||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+k.id),get("welcome-screen").classList.add("hidden")}catch(b){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+b.message,"\
error",!0);return}const g={model:f,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,target_lang:isGeminiLiveTranslateModel()&&get("sts-target-lang")?get(
"sts-target-lang").value:""};setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const b=await apiFetch("/a\
pi/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),
k=await b.json().catch(()=>({}));if(!b.ok)throw new Error(k.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=k.session_id,this.rateIn=k.rate_in||this.rateIn,this.rateOut=k.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null}catch(b){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
b.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(b){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
b.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const f="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),g=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(f,g).then(b=>{if(!b.ok)throw new Error("SSE stream failed ("+b.status+")");this.
reader=b.body.getReader(),this._readLoop()}).catch(b=>{b&&b.name==="AbortError"||(this.streamError=b&&
b.message?b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const f=new TextDecoder;let g="";try{for(;this.reader;){const{done:b,
value:k}=await this.reader.read();if(b)break;g+=f.decode(k,{stream:!0});let T;for(;(T=g.indexOf(`

`))>=0;){const C=g.slice(0,T);g=g.slice(T+2);for(const A of C.split(`
`)){if(!A.startsWith("data: "))continue;let I=null;try{I=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent(I)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
if(f)switch(f.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new rn(this.rateOut||24e3),Wt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(f.data));break;case"transcript":f.role==="user"?(f.cumulative?this.userTranscript=
f.delta:this.userTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
f.role==="assistant"?(this.assistantTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):f.role==="thought"&&(this.assistantThought+=f.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":f.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"error":this.streamError=f.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Wt=null}_startCapture(){const f=window.
AudioContext||window.webkitAudioContext;if(!f)throw new Error("AudioContext not supported");return this.
audioCtx=new f({sampleRate:this.rateIn||24e3}),navigator.mediaDevices.getUserMedia(Qn()).then(g=>{this.
stream=g;const b=this.audioCtx.createMediaStreamSource(g),k=this.rateIn||24e3,T=this.audioCtx.sampleRate,
C=4096;this.processor=this.audioCtx.createScriptProcessor(C,1,1),this.processor.onaudioprocess=A=>{if(!this.
active||!this.capturing)return;const I=A.inputBuffer.getChannelData(0),D=pi(I,T,k);!D||!D.byteLength||
this._sendAudio(D)},b.connect(this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(f){
if(!this.sessionId||!this.active)return;const g="/api/realtime/audio?session_id="+encodeURIComponent(
this.sessionId),b={method:"POST",credentials:"include",headers:{"X-CSRF-Token":csrfToken,"Content-Ty\
pe":"application/octet-stream"},body:f},k=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions==
"function"?window.ProgressSpinner.manualRequestOptions(b):b;fetch(g,k).catch(()=>{})}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}vn(),It()}async stop(){if(!this.active)return;this.active=
!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0);try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({session_id:this.sessionId})})}catch{}const f=Date.now(),g=this.responseDoneCount;
let b=this.lastAudioAt;for(;Date.now()-f<2e4&&!(this.responseDoneCount>g||(this.lastAudioAt>b&&(b=this.
lastAudioAt),!this.speechActive&&Date.now()-f>2e3&&Date.now()-b>2500));)await new Promise(k=>setTimeout(
k,250));await this._save()}async _save(){if(!this.saved){this.saved=!0,this.saving=!0;try{const f=await apiFetch(
"/api/realtime/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:this.sessionId,thread_id:currentThreadId})}),g=await f.json().catch(()=>({}));if(!f.ok)throw new Error(
g.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(f){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(f&&f.message?f.message:f),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this._stopCapture(),this._stopPlayback(),this.abortCtrl){try{this.abortCtrl.
abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const f=get("mic-btn");f&&(f.
classList.remove("bg-red-600","animate-pulse"),f.classList.add("bg-gray-700"))}};r(In,"RealtimeVoice\
Session");let An=In;function pi(u,f,g){let b=u;if(f!==g&&f>0&&g>0){const T=f/g,C=Math.floor(b.length/
T),A=new Float32Array(C);for(let I=0;I<C;I++)A[I]=b[Math.min(Math.floor(I*T),b.length-1)];b=A}const k=new Int16Array(
b.length);for(let T=0;T<b.length;T++){const C=Math.max(-1,Math.min(1,b[T]));k[T]=C<0?C*32768:C*32767}
return k.buffer}r(pi,"pcm16FromFloat32");const Vt=new An;(()=>{const g={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let b=null,k=null,T=!1,C=null,A=!1,I=0,D=0,O=null,
G="idle",se=!1,R=null;const $=r(N=>document.getElementById(N),"$"),U=r(N=>{const q=Object.assign({},
N||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?
window.ProgressSpinner.manualRequestOptions(q):(q.progressSpinner=!1,q)},"noSpinner");function K(N,q){
G=q;const ye=$("lyria-status-text"),ae=$("lyria-status-dot");ye&&(ye.textContent=N),ae&&(ae.className=
"w-2 h-2 rounded-full inline-block "+(g[q]||g.idle)),Te(),Re()}r(K,"setStatus");function ke(){const N=D?
Math.floor((Date.now()-D)/1e3):0,q=String(Math.floor(N/60)).padStart(2,"0"),ye=String(N%60).padStart(
2,"0");return`${q}:${ye}`}r(ke,"formatElapsed");function X(){D||(D=Date.now());const N=$("lyria-elap\
sed");N&&(N.textContent=ke()),O||(O=window.setInterval(()=>{const q=$("lyria-elapsed");q&&(q.textContent=
ke())},1e3))}r(X,"startElapsedTimer");function re(){O&&(window.clearInterval(O),O=null)}r(re,"stopEl\
apsedTimer");function Te(){const N=$("lyria-play-btn"),q=$("lyria-pause-btn"),ye=$("lyria-stop-btn"),
ae=$("lyria-reset-btn"),xe=!!b,Ne=G==="streaming"||G==="connecting";if(N){N.disabled=se||!xe;const Ye=N.
querySelector("i");Ye&&(Ye.className="fas fa-play")}q&&(q.disabled=se||!Ne),ye&&(ye.disabled=se||!xe||
!Ne),ae&&(ae.disabled=se||!xe||!Ne)}r(Te,"updateTransportButtons");function Re(){const N=$("lyria-sa\
ve-btn");if(!N)return;const q=!!b&&G!=="idle"&&G!=="connecting"&&G!=="error";N.classList.toggle("hid\
den",!q)}r(Re,"updateSaveButton");function je(N,q){const ye=$("lyria-prompt-rows");if(!ye)return;const ae=document.
createElement("div");ae.className="flex items-center gap-2",ae.innerHTML=`
                        <input type="text" value="${escapeHtml(N||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof q=="num\
ber"?q:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof q=="number"?q:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const xe=ae.querySelector('input[type="range"]'),Ne=ae.querySelector(".lyria-w\
eight-label");xe&&Ne&&xe.addEventListener("input",()=>{Ne.textContent=parseFloat(xe.value).toFixed(1)});
const Ye=ae.querySelector(".lyria-prompt-remove");Ye&&Ye.addEventListener("click",()=>{ye.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ae.remove()}),ae.classList.add("lyria-prompt-row-wrap"),ye.appendChild(
ae)}r(je,"addPromptRow");function Ze(){const N=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),q=[];return N.forEach(ye=>{const ae=ye.querySelector('input[type="text"]'),xe=ye.querySelector(
'input[type="range"]'),Ne=(ae?ae.value:"").trim();Ne&&q.push({text:Ne,weight:parseFloat(xe?xe.value:
1)||1})}),q}r(Ze,"collectPrompts");function it(){const N={},q=r(hi=>{const On=$(hi);return On&&On.value!==
""?parseFloat(On.value):void 0},"num"),ye=q("lyria-bpm");ye!==void 0&&(N.bpm=Math.round(ye));const ae=q(
"lyria-guidance");ae!==void 0&&(N.guidance=ae);const xe=q("lyria-density");xe!==void 0&&(N.density=xe);
const Ne=q("lyria-brightness");Ne!==void 0&&(N.brightness=Ne);const Ye=q("lyria-temperature");Ye!==void 0&&
(N.temperature=Ye);const Ge=$("lyria-scale");Ge&&Ge.value&&(N.scale=Ge.value);const et=$("lyria-mode");
et&&et.value&&(N.music_generation_mode=et.value);const rt=$("lyria-mute-bass"),Nt=$("lyria-mute-drum\
s"),ii=$("lyria-only-bass-drums");return rt&&(N.mute_bass=rt.checked),Nt&&(N.mute_drums=Nt.checked),
ii&&(N.only_bass_and_drums=ii.checked),N}r(it,"collectConfig");function Pt(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([q,ye])=>{
const ae=$(q),xe=$(ye);!ae||!xe||ae.addEventListener("input",()=>{const Ne=parseFloat(ae.value);xe.textContent=
q==="lyria-bpm"?String(Math.round(Ne)):Ne.toFixed(1)})})}r(Pt,"bindRangeLabels");function vt(){if(C){
try{C.close()}catch{}C=null}A=!1,I=0}r(vt,"resetPlayback");function Ot(){if(T=!1,k&&typeof k.abort==
"function")try{k.abort()}catch{}k=null}r(Ot,"closeStream");async function kt(){Ot(),k=new AbortController,
T=!0;try{const N=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(b)}`,U({method:"\
GET",signal:k.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!N.ok){const xe=await N.
json().catch(()=>({}));throw new Error(xe.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const q=N.body.getReader(),ye=new TextDecoder;let ae="";for(;T;){const{done:xe,value:Ne}=await q.read();
if(xe)break;ae+=ye.decode(Ne,{stream:!0});const Ye=ae.split(`

`);ae=Ye.pop();for(const Ge of Ye){const et=Ge.split(`
`).find(Nt=>Nt.startsWith("data: "));if(!et)continue;const rt=et.slice(6);try{const Nt=JSON.parse(rt);
fe(Nt)}catch{}}}}catch(N){if(N&&N.name==="AbortError")return;T&&(K("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{T&&b&&kt()},1200))}finally{T=!1}}r(kt,"openStream");function fe(N){
if(N&&N.snapshot){const q=N.status;if(q==="error"){K("\u30A8\u30E9\u30FC","error"),re();return}if(q===
"closed"||q==="stopped"){K("\u7D42\u4E86","closed"),re();return}K(q==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",q==="paused"?"paused":"connecting");return}if(N&&N.audio){K("\u518D\u751F\u4E2D...",
"streaming"),X(),he(N.audio);return}if(N&&N.error){K("\u30A8\u30E9\u30FC: "+N.error,"error"),re();return}
if(N&&N.final){K("\u7D42\u4E86","closed"),re(),Te();return}}r(fe,"handleStreamMessage");function he(N){
if(!N)return;if(!C){const Ge=window.AudioContext||window.webkitAudioContext;if(!Ge)return;C=new Ge({
sampleRate:48e3}),A=!1,I=0}let q;try{const Ge=atob(N);q=new Uint8Array(Ge.length);for(let et=0;et<Ge.
length;et++)q[et]=Ge.charCodeAt(et)}catch{return}const ye=new Int16Array(q.buffer),ae=Math.floor(ye.
length/2);if(ae<1)return;const xe=C.createBuffer(2,ae,48e3);for(let Ge=0;Ge<2;Ge++){const et=xe.getChannelData(
Ge);for(let rt=0;rt<ae;rt++)et[rt]=ye[rt*2+Ge]/32768}C.state==="suspended"&&C.resume();const Ne=C.createBufferSource();
Ne.buffer=xe,Ne.connect(C.destination),A||(I=C.currentTime+.08,A=!0);const Ye=Math.max(C.currentTime,
I);Ne.start(Ye),I=Ye+xe.duration}r(he,"playChunk");async function Be(N,q){const ye=await fetch("/api\
/gemini/music/command",U({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:N},q||{}))})),ae=await ye.json().catch(()=>({}));if(!ye.ok)throw new Error(
ae.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ae}
r(Be,"apiCommand");async function me(){if(se)return;const N=Ze();if(!N.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}se=!0,Te(),K("\u63A5\u7D9A\u4E2D...","connecting");try{
const q=await fetch("/api/gemini/music/start",U({method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({weighted_prompts:N,config:it()})})),ye=await q.json().catch(()=>({}));if(!q.
ok)throw new Error(ye.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=ye.session_id,R=it(),K("\u63A5\u7D9A\u4E2D...","connecting"),kt()}catch(q){K("\u30A8\u30E9\u30FC: "+
q.message,"error"),showToast("Lyria RealTime: "+q.message,"error",!0)}finally{se=!1,Te()}}r(me,"star\
tSession");async function we(N){if(b){se=!0,Te();try{await Be("control",{action:N}),N==="PLAY"?K("\u518D\u751F\
\u4E2D...","streaming"):N==="PAUSE"?K("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):N==="STOP"?K("\u505C\u6B62\u4E2D",
"stopped"):N==="RESET_CONTEXT"&&K("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(q){showToast("Lyria RealTime: "+q.message,"error",!0),K("\u30A8\u30E9\u30FC: "+q.
message,"error")}finally{se=!1,Te()}}}r(we,"control");async function Oe(){if(!b)return;const N=Ze();
if(!N.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}se=!0;try{await Be("prompts",{weighted_prompts:N}),K("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
G==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(q){showToast("Lyria RealTime: "+q.message,"error",!0)}finally{se=!1,Te()}}r(Oe,"app\
lyPrompts");async function _t(){if(!b)return;const N=it(),q=R||{},ye=N.bpm!==void 0&&N.bpm!==q.bpm,ae=N.
scale!==void 0&&N.scale!==q.scale,xe=ye||ae;se=!0;try{await Be("config",{config:N,reset_context:xe}),
R=N,K(xe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",G==="paused"?"paused":"streaming"),showToast(
xe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Ne){showToast("Lyria RealT\
ime: "+Ne.message,"error",!0)}finally{se=!1,Te()}}r(_t,"applyConfig");async function Jt(){if(b){se=!0,
K("\u4FDD\u5B58\u4E2D...","connecting"),Te();try{const N=await fetch("/api/gemini/music/save",U({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),q=await N.json().catch(()=>({}));if(!N.ok)throw new Error(q.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
K("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),re(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),q.thread_id&&(currentThreadId=String(q.thread_id),history.pushState({},"","/c/"+q.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(q.thread_id||currentThreadId),cn(!0)}catch(N){
K("\u30A8\u30E9\u30FC: "+N.message,"error"),showToast("Lyria RealTime: "+N.message,"error",!0)}finally{
se=!1,Te()}}}r(Jt,"saveSession");async function qe(){if(Ot(),b)try{await fetch("/api/gemini/music/ca\
ncel",U({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})}))}catch{}
b=null,re(),vt(),K("\u6E96\u5099\u5B8C\u4E86","idle")}r(qe,"cancelSession");function dt(){const N=$(
"lyria-prompt-rows");N&&(N.innerHTML=""),je("",1),R=null,D=0,["lyria-bpm","lyria-guidance","lyria-de\
nsity","lyria-brightness","lyria-temperature"].forEach(ae=>{const xe=$(ae);xe&&(xe.value=ae==="lyria\
-bpm"?"120":ae==="lyria-guidance"?"4":ae==="lyria-temperature"?"1.1":"0.5")});const q=$("lyria-scale");
q&&(q.value="");const ye=$("lyria-mode");ye&&(ye.value="QUALITY"),["lyria-mute-bass","lyria-mute-dru\
ms","lyria-only-bass-drums"].forEach(ae=>{const xe=$(ae);xe&&(xe.checked=!1)}),Pt()}r(dt,"resetContr\
ols");function cn(N){Ot(),b&&fetch("/api/gemini/music/cancel",U({method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),b=null,T=!1,re(),vt(),hideModal(
"lyria-studio-modal")}r(cn,"closeAndCleanup");function Pn(N){if(!isLyriaRealtimeModel()){showToast("\
Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const q=$("lyria-studio-modal");if(q&&q.classList.contains("modal-open")&&b){if(N&&
typeof N=="string"){const ae=$("lyria-prompt-rows");ae&&(ae.innerHTML=""),je(N,1)}return}if(b&&qe(),
dt(),N&&typeof N=="string"){const ae=$("lyria-prompt-rows");ae&&(ae.innerHTML=""),je(N,1)}b=null,T=!1,
re(),vt(),K("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(Pn,"open");function fi(){
const N=$("lyria-open-studio-btn");N&&N.addEventListener("click",()=>Pn(""));const q=$("lyria-studio\
-close");q&&q.addEventListener("click",()=>cn(!1));const ye=$("lyria-play-btn");ye&&ye.addEventListener(
"click",()=>{if(!b){me();return}we("PLAY")});const ae=$("lyria-pause-btn");ae&&ae.addEventListener("\
click",()=>we("PAUSE"));const xe=$("lyria-stop-btn");xe&&xe.addEventListener("click",()=>we("STOP"));
const Ne=$("lyria-reset-btn");Ne&&Ne.addEventListener("click",()=>we("RESET_CONTEXT"));const Ye=$("l\
yria-add-prompt-btn");Ye&&Ye.addEventListener("click",()=>je("",1));const Ge=$("lyria-apply-prompts-\
btn");Ge&&Ge.addEventListener("click",Oe);const et=$("lyria-apply-config-btn");et&&et.addEventListener(
"click",_t);const rt=$("lyria-save-btn");rt&&rt.addEventListener("click",Jt),Pt(),dt(),window.openLyriaStudio=
Pn}return r(fi,"init"),{init:fi,open:Pn}})().init(),(()=>{let u=null,f=null;const g=r(R=>document.getElementById(
R),"$");function b(){return isStsModel()&&voiceStudioUiEnabled!==!1}r(b,"isStudioMode");function k(){
const R=get("model-select")?get("model-select").value:"",$=g("voice-studio-title");$&&(R==="gpt-tran\
scribe"||R==="gpt-live-transcribe"?$.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":
R==="gemini-3.5-live-translate-preview"?$.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
$.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(k,"updateTitle");function T(){const R=g("voi\
ce-studio-transcript");R&&(R.innerHTML='<div class="text-[10px] text-gray-500">\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</\
div>')}r(T,"resetTranscript");function C(R,$){if(!$||!String($).trim())return;const U=g("voice-studi\
o-transcript");if(!U||!window.VoiceStudioOpen)return;const K=R==="user"?"\u3042\u306A\u305F":"AI",ke=R===
"user"?"text-cyan-300":"text-gray-100",X=U.querySelectorAll(".voice-studio-line");let re=null;for(let Re=X.
length-1;Re>=0;Re--)if(X[Re].dataset.role===R){re=X[Re];break}const Te=`<span class="${ke} font-bold\
">${escapeHtml(K)}:</span> <span class="text-gray-200">${escapeHtml($)}</span>`;if(re)re.innerHTML=Te;else{
const Re=U.querySelector(".text-gray-500");Re&&Re.remove();const je=document.createElement("div");je.
className="voice-studio-line",je.dataset.role=R,je.innerHTML=Te,U.appendChild(je)}U.scrollTop=U.scrollHeight}
r(C,"log");function A(){const R=g("sts-panel"),$=g("voice-studio-panel-host");R&&$&&R.parentNode!==$&&
(u=R.parentNode,$.appendChild(R));const U=g("file-preview"),K=g("voice-studio-file-host");U&&K&&U.parentNode!==
K&&(f=U.parentNode,K.appendChild(U),K.classList.remove("hidden"))}r(A,"movePanelIntoModal");function I(){
const R=g("sts-panel");R&&u&&R.parentNode!==u&&u.appendChild(R);const $=g("file-preview");$&&f&&$.parentNode!==
f&&f.appendChild($);const U=g("voice-studio-file-host");U&&U.classList.add("hidden"),u=null,f=null}r(
I,"movePanelBack");function D(){if(!b()){showToast("\u97F3\u58F0\u7CFB\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}A();const R=g("sts-panel");R&&R.classList.remove("hidden"),k(),T(),window.VoiceStudioOpen=
!0,showModal("voice-studio-modal")}r(D,"open");function O(){if(window.VoiceStudioOpen&&(Pe||Xe&&Xe.state===
"recording"||Vt.isActive())&&Yn(),window.VoiceStudioOpen=!1,I(),hideModal("voice-studio-modal"),isStsModel()&&
voiceStudioUiEnabled!==!1){const R=g("sts-panel");R&&R.classList.add("hidden")}}r(O,"close");function G(){
window.VoiceStudioOpen&&O()}r(G,"closeIfOpen");function se(){window.VoiceStudioOpen=!1;const R=g("vo\
ice-studio-open-btn");R&&R.addEventListener("click",()=>D());const $=g("voice-studio-close");$&&$.addEventListener(
"click",()=>O()),window.VoiceStudio={open:D,close:O,closeIfOpen:G,log:C,isStudioMode:b}}return r(se,
"init"),{init:se,open:D,close:O,closeIfOpen:G,log:C,isStudioMode:b}})().init();let Wt=null;function Xn(){
if(Wt&&(Wt.stop(),Wt=null),an){try{an.pause()}catch{}try{an.src=""}catch{}an=null}}r(Xn,"stopStsPlay\
back");async function vi(u){Xn();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=!0,f.playsInline=
!0,an=f,await f.play(),new Promise(g=>{f.onended=()=>g("ended"),f.onerror=()=>g("error")})}r(vi,"pla\
yStsAudio");function Yn(){if(Vt.isActive()){Vt._cancel();return}if(Pe){Pe.stop(),Pe=null,Xn(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),It();return}Xe&&Xe.state===
"recording"&&(Cn=!0,Xe.stop())}r(Yn,"cancelRecording");function Qn(){if(isStsModel())return{audio:!0};
const f=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return f.echoCancellation&&(g.echoCancellation=!1),f.
noiseSuppression&&(g.noiseSuppression=!1),f.autoGainControl&&(g.autoGainControl=!1),{audio:g}}r(Qn,"\
getMicCaptureConstraints"),get("mic-btn").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\
\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Pe){setStsStatus("Processing...",!0);const u=Pe;Pe=null,u.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const f=await u.
getFinalData();if(isGeminiLiveTranscribeModel()&&(f.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
f.assistant_text=(u.inputTranscript||"").trim(),f.assistant_thought="",!f.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){
const b=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=
String(b.id),history.pushState({},"","/c/"+b.id),get("welcome-screen").classList.add("hidden")}f.thread_id=
currentThreadId,f.model=get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)}),setStsStatus("Saved",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(f){console.error(
"Failed to save Gemini Live session:",f),setStsStatus("Error saving",!1)}return}if(Vt.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
Vt.stop();return}if(Xe&&Xe.state==="recording"){Xe.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||$t("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const b={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(b.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const R=get("sts-custom\
-vocab").value.split(/[,、\n]/).map($=>$.trim()).filter(Boolean);R.length&&(b.custom_vocabulary=R.slice(
0,1e3))}}else b.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(b.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",b.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(b.target_lang=get("sts-target-lang").value);const k=await apiFetch("/api/gemini\
/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});if(!k.
ok)throw new Error("Failed to get session token");const{token:T,url:C}=await k.json(),A=get("model-s\
elect").value,I=get("sts-voice")?get("sts-voice").value:"Kore",D=get("sts-thinking-level")?get("sts-\
thinking-level").value:"minimal",O=get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1;
if(Pe=new Mn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Pe.rtPlayer=new rn),isGeminiLiveTranscribeModel()){
const R=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",$={languageCodes:[]};
if((R==="SMART"||R==="VERBATIM")&&($.mode=R),get("sts-custom-vocab")){const U=get("sts-custom-vocab").
value.split(/[,、\n]/).map(K=>K.trim()).filter(Boolean);U.length&&($.customVocabulary=U.slice(0,1e3))}
await Pe.start(T,C,A,{transcriptionConfig:$})}else if(isGeminiLiveTranslateModel()){const R=get("sts\
-target-lang")?get("sts-target-lang").value:"ja";await Pe.start(T,C,A,{translationConfig:{targetLanguageCode:R,
echoTargetLanguage:!0}})}else{const R={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:I}}}};
isGeminiLiveExtendedThinkingModel()&&(R.thinkingConfig={thinkingLevel:D,includeThoughts:O}),await Pe.
start(T,C,A,R)}Xe=Pe.backupRecorder,Xe.onstop=()=>{Pe&&get("mic-btn").click()};let G=!0,se="live-sts\
-"+Date.now();Pe.onMessage=R=>{if(R.serverContent){if(isGeminiLiveTranscribeModel()){const $=Pe.interimInputTranscript,
U=Pe.inputTranscript,K=U+($&&!U.endsWith($)?(U?`
`:"")+$:""),ke=get("chat-messages");let X=document.getElementById(se);X||(X=document.createElement("\
div"),X.id=se,X.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg\
 border border-slate-700/50",X.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ke.appendChild(X),ke.scrollTop=ke.scrollHeight);const re=X.
querySelector(".message-content");re.innerText=K||"\u8074\u304D\u53D6\u308A\u4E2D...",ke.scrollTop=ke.
scrollHeight,window.VoiceStudio&&U&&window.VoiceStudio.log("user",U);return}if(R.serverContent.modelTurn){
G&&(setStsStatus("Gemini is speaking...",!1),G=!1);const $=get("chat-messages");let U=document.getElementById(
se);U||(U=document.createElement("div"),U.id=se,U.className="flex flex-col gap-2 mb-4 assistant-mess\
age bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",U.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,$.appendChild(U),$.scrollTop=$.scrollHeight);const K=U.
querySelector(".thought-container"),ke=U.querySelector(".message-content");Pe.assistantThought&&(K.classList.
remove("hidden"),K.innerText=Pe.assistantThought),ke.innerText=Pe.assistantText,$.scrollTop=$.scrollHeight,
window.VoiceStudio&&(Pe.inputTranscript&&window.VoiceStudio.log("user",Pe.inputTranscript),Pe.assistantText&&
window.VoiceStudio.log("assistant",Pe.assistantText))}}},setStsStatus("Listening...",!0),get("mic-bt\
n").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),Jn(Pe.
stream),Kn(Pe.stream);return}catch(g){showToast("Gemini Live connection failed: "+g.message,"error",
!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){await Vt.start();return}isStsModel()||
(Wn(),$t("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));const u=await navigator.mediaDevices.
getUserMedia(Qn());Xe=new MediaRecorder(u),mn=[],Cn=!1;const f=isStsModel();Xe.ondataavailable=g=>mn.
push(g.data),Xe.onstop=async()=>{if(Cn){mn=[],get("file-preview").classList.add("hidden"),u.getTracks().
forEach(C=>C.stop()),vn(),It(),f||($t("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),yn=setTimeout(()=>$t("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(mn,{type:"audio/web\
m"}),b=new File([g],"recording.webm",{type:"audio/webm"}),k=new FormData;k.append("file",b),get("fil\
e-preview").classList.remove("hidden");const T=f;get("file-name").innerText=T?"Processing voice...":
"Transcribing...";try{if(T){if(!currentThreadId){const K=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=K.id!==null&&K.id!==void 0?String(K.id):K.id,setTemporaryChatUiState(!!(K&&K.
is_temporary)),setCurrentChatHeaderTitle(K&&K.title),applyTemporaryChatRuntimeMeta(K||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+K.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),k.append("model",get(
"model-select").value),k.append("thread_id",currentThreadId),get("sts-voice")&&k.append("sts_voice",
get("sts-voice").value||""),get("sts-speed")&&k.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&k.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&k.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&k.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&k.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const C=await apiFetch(
"/sts",{method:"POST",body:k});if(!C.ok){const U=await C.json().catch(()=>({}));throw new Error(U.error||
"Speech-to-speech failed")}const A=C.body.getReader(),I=new TextDecoder;let D="",O=null,G=null;stsOpt(
"sts-auto-play")&&(G=new rn,Wt=G),setStsStatus(isTranscriptionModel()?"Transcribing...":"Processing \
audio...",!0);let se=!0,R="",$="";for(;;){const{done:U,value:K}=await A.read();if(U)break;D+=I.decode(
K,{stream:!0});const ke=D.split(`
`);D=ke.pop();for(const X of ke){if(!X.trim())continue;const re=JSON.parse(X);if(re.error)throw new Error(
re.error);re.audio_delta&&G&&(se&&(setStsStatus("Playing response...",!1),se=!1),await G.addChunk(re.
audio_delta)),re.input_delta&&(R+=re.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",R)),
re.transcript_delta&&($+=re.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assistant",
$)),re.final&&(O=re)}}window.VoiceStudio&&!R.trim()&&window.VoiceStudio.log("user","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),
O&&(O.audio_url||O.transcription_only)&&(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus(
"Listening...",!0),get("mic-btn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{
const C=get("set-mic-transcribe-mode");if(!!(C&&C.value==="llm")&&!supportsAudioInputModel()){showToast(
"\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306FLLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}k.append("llm_model",get("model-select")&&get("model-select").value||"");const D=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:k})).json();if(D.transcript){const O=get("prompt-inp\
ut");O.value+=(O.value?" ":"")+D.transcript,O.style.height="auto",O.style.height=O.scrollHeight+"px"}else
showToast(D.error||"Transcription failed","error",!0)}}catch(C){showToast("Audio processing error: "+
C.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(C=>C.
stop()),vn(),It(),T||$t("","hidden"),T&&setStsStatus("Tap to speak",!1)}},Xe.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
($t("\u9332\u97F3\u4E2D\u2026","recording"),Jn(u)),Kn(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{It(),isStsModel()||$t("","hidden"),alert("Microphone access denied or not avail\
able.")}};const ln=r((u,f)=>{if(!u)return;const g=u.querySelector("span");g?g.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),g=get("lib-download-btn"),b=get("lib-attach-btn"),k=get("lib-rena\
me-btn"),T=get("lib-usage-btn");if(f&&(f.disabled=u===0,ln(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
g&&(g.disabled=u===0,ln(g,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=u===0,ln(b,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),k&&(k.disabled=u!==1,ln(k,"\u540D\u524D\u5909\u66F4")),
T&&(T.disabled=u!==1,ln(T,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const C=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",C&&u>0)}};function Zn(u){lib.
attachMode=!!u}r(Zn,"setLibAttachMode");const ei=r((u=!1)=>{Zn(u),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(u=!1)=>{hideModal("lib-modal"),!u&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>ei(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>ei(!0)),
get("lib-search")){let u=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),u&&clearTimeout(u),u=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const u=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=u,get("lib-sort").onchange=()=>{const f=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,f),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const mi=r(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",get("gem-default-model").value="",get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-con\
tainer").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"open\
GemModal");window.closeGemModal=(u=!1)=>{hideModal("gem-modal"),!u&&location.pathname==="/gem"&&history.
back()},get("add-gem-btn").onclick=()=>mi(),get("save-gem-btn").onclick=async()=>{const u=get("gem-n\
ame").value,f=get("gem-desc").value,g=get("gem-inst").value,b=collectGemFixedPrompts();if(u&&g){const k=editingGemUuid?
"PUT":"POST",T=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
T,{method:k,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:u,description:f,instruction:g,
fixed_prompts:b,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=u,activeGem.instruction=
g,activeGem.fixed_prompts=b,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(u){if(u.target.closest(".edit-btn")){const g=u.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(g)}if(u.target.closest(".code-toggle")){
const f=u.target.closest(".code-toggle"),g=f.closest(".code-wrapper");if(!g)return;const b=g.classList.
toggle("collapsed");g.setAttribute("data-collapsed",b?"true":"false"),f.setAttribute("aria-expanded",
b?"false":"true"),f.innerHTML=b?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',f.title=b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",f.setAttribute("aria-label",b?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(u.target.closest(".download-btn")){const f=u.target.closest(".d\
ownload-btn"),g=f.getAttribute("data-code"),b=(f.getAttribute("data-lang")||"txt").toLowerCase();if(g)
try{const k=decodeURIComponent(g),T=new Blob([k],{type:"text/plain"}),C=URL.createObjectURL(T),A=document.
createElement("a");A.href=C;let D={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(D="txt");let O=`code.${D}`;b==="dockerfile"&&(O="Docke\
rfile"),b==="makefile"&&(O="Makefile"),A.download=O,document.body.appendChild(A),A.click(),document.
body.removeChild(A),URL.revokeObjectURL(C)}catch(k){console.error("Download failed",k)}}if(u.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(u.target.closest(".coding-target-btn")),
u.target.closest(".copy-btn")){const f=u.target.closest(".copy-btn"),g=f.getAttribute("data-code");g&&
window.copyCode(f,g)}if(u.target.closest(".html-preview-btn")){const g=u.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(u.target.closest(".canvas-preview-bt\
n")){const f=u.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(f)}}),document.querySelectorAll(
".modal-overlay").forEach(u=>{u.addEventListener("click",f=>{f.target===u&&Un(u.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){const e=get(
"file-preview"),n=get("file-name"),i=get("upload-total-progress"),a=get("upload-total-progress-bar"),
o=get("file-preview-thumbs"),l=get("upload-modal-status-text"),c=get("upload-modal-total-progress"),
d=get("upload-modal-total-progress-bar");if(!e||!n)return;if(o){const M=document.querySelectorAll("#\
upload-list .upload-row");o.innerHTML="",M.forEach((P,H)=>{const Q=P.getAttribute("data-local-url"),
ee=P.getAttribute("data-filename"),Ae=P.querySelector("img.upload-preview")!==null;let B;if(Ae){let V=Q;
if(!V&&ee){const te=ee.replace(/^\d+\//,"");V=buildAttachmentPreviewUrl(te)}V&&(B=document.createElement(
"img"),B.src=V,B.className="thumb-item shadow-sm",B.dataset.viewerSrc=V,B.dataset.viewerFilename=ee||
V.split("/").pop(),B.onclick=function(te){te.preventDefault(),openImageViewer(this.dataset.viewerSrc,
".thumb-item")},B.onerror=function(){this.parentElement.replaceChild(m("ERR"),this)})}B||(B=m("FILE")),
B.style.animationDelay=`${H*32}ms`,o.appendChild(B)}),M.length>0?o.classList.remove("hidden"):o.classList.
add("hidden")}function m(M){const P=document.createElement("div");return P.className="thumb-item bg-\
gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold",P.innerText=
M,P}r(m,"createFileThumb");const h=collectImageUrlsForSend(),y=uploadProgressState.total,v=uploadProgressState.
completed,x=uploadProgressState.active;y===0&&(e.classList.add("hidden"),i&&i.classList.add("hidden"),
c&&c.classList.add("hidden"),o&&o.classList.add("hidden"));const w=get("send-btn"),_=get("mic-btn"),
S=get("mask-btn"),L=isStopMode;if(x>0?(w&&(w.disabled=!0),_&&(_.disabled=!0),S&&(S.disabled=!0)):L||
(w&&(w.disabled=!1),_&&(_.disabled=!1),S&&(S.disabled=!1)),x>0){const M=`Preparing... (${v}/${y})`;e.
classList.remove("hidden"),n.innerText=M,l&&(l.innerText=`(${v}/${y})`);let P=v*100,H=0;for(let Ae in uploadProgressState.
perFilePct)P+=uploadProgressState.perFilePct[Ae],H++;const Q=y>0?P/(y*100)*100:0,ee=`${Math.min(100,
Q)}%`;i&&a&&(i.classList.remove("hidden"),a.style.width=ee),c&&d&&(c.classList.remove("hidden"),d.style.
width=ee)}else l&&(l.innerText=""),c&&c.classList.add("hidden"),h.length>0?(e.classList.remove("hidd\
en"),n.innerText=`${h.length} files ready`,i&&i.classList.add("hidden")):(e.classList.add("hidden"),
n.innerText="",i&&i.classList.add("hidden"));schedulePromptTokenEstimate()}r(updateFilePreview,"upda\
teFilePreview");function updateMaskPreview(){const e=get("mask-preview"),n=get("mask-name");!e||!n||
(currentMaskImage?(e.classList.remove("hidden"),n.innerText=`Mask: ${currentMaskImage.split("/").pop()}`):
(e.classList.add("hidden"),n.innerText=""))}r(updateMaskPreview,"updateMaskPreview");const markerToolHints={
draw:"\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const n=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
n))return n;if(/^#[0-9a-f]{3}$/.test(n)){const i=n[1],a=n[2],o=n[3];return`#${i}${i}${a}${a}${o}${o}`}
return"#facc15"}r(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const n=normalizeMarkerHexColor(
e);return{r:parseInt(n.slice(1,3),16),g:parseInt(n.slice(3,5),16),b:parseInt(n.slice(5,7),16)}}r(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,n=60){const i=Number(e),a=Number.isFinite(i)?i:n;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,a))}r(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const n=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(n)?String(n):String(n).replace(/\.0$/,"")}r(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
n=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${n})`}r(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const n=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=n;const i=n*100,a=formatMarkerOpacityPct(i),o=get("ma\
rker-color-picker");o&&o.value!==e&&(o.value=e);const l=get("marker-opacity");l&&l.value!==a&&(l.value=
a);const c=get("marker-opacity-number");c&&c.value!==a&&(c.value=a);const d=get("marker-opacity-valu\
e");d&&(d.textContent=`${a}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(h=>{const y=normalizeMarkerHexColor(h.getAttribute("data-marker-color"));h.classList.
toggle("active",y===e)})}r(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}r(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const n=clampMarkerOpacityPct(e,60);markerState.opacity=n/100,
syncMarkerColorControls()}r(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const n=get("marker-tool-draw"),i=get("mar\
ker-tool-mosaic"),a=get("marker-tool-crop");n&&n.classList.toggle("active",e==="draw"),i&&i.classList.
toggle("active",e==="mosaic"),a&&a.classList.toggle("active",e==="crop");const o=get("marker-tool-hi\
nt");o&&(o.textContent=markerToolHints[e]||"");const l=get("marker-crop-reset");l&&l.classList.toggle(
"hidden",e!=="crop");const c=get("marker-canvas");c&&(c.style.pointerEvents=e==="crop"?"none":"auto");
const d=get("marker-crop-canvas");d&&(d.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.
cropRect||markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}
r(setMarkerMode,"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}
r(clearCropRect,"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)
return;const n=Math.max(1,e.width||0),i=Math.max(1,e.height||0);n<=1||i<=1||(markerState.cropRect={x:0,
y:0,w:n,h:i})}r(resetCropRectToFull,"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.
scale<=markerView.minScale+1e-4){markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marke\
r-stage"),n=get("marker-viewport");if(!e||!n)return;const i=Math.max(1,e.clientWidth||0),a=Math.max(
1,e.clientHeight||0),o=Math.max(1,n.offsetWidth||n.clientWidth||0),l=Math.max(1,n.offsetHeight||n.clientHeight||
0);if(i<=1||a<=1||o<=1||l<=1)return;const c=(i-o)/2,d=(a-l)/2,m=o*markerView.scale,h=l*markerView.scale,
y=Math.min(i*.45,Math.max(24,i*.12)),v=Math.min(a*.45,Math.max(24,a*.12)),x=y-c-m,w=i-y-c,_=v-d-h,S=a-
v-d,L=r((M,P,H)=>Number.isFinite(M)?P>H?(P+H)/2:Math.min(H,Math.max(P,M)):0,"clampOffset");markerView.
offsetX=L(markerView.offsetX,x,w),markerView.offsetY=L(markerView.offsetY,_,S)}r(clampMarkerViewOffset,
"clampMarkerViewOffset");function applyMarkerTransform(){const e=get("marker-viewport");e&&(clampMarkerViewOffset(),
e.style.transform=`translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale}\
)`)}r(applyMarkerTransform,"applyMarkerTransform");function resetMarkerTransform(){markerView.scale=
1,markerView.offsetX=0,markerView.offsetY=0,applyMarkerTransform()}r(resetMarkerTransform,"resetMark\
erTransform");function getRowMarkerKey(e){return e&&(e.dataset.uploadId||e.getAttribute("data-filena\
me"))||null}r(getRowMarkerKey,"getRowMarkerKey");function setRowMarkerState(e,n){const i=getRowMarkerKey(
e);i&&(n?markerAppliedUploads.add(i):markerAppliedUploads.delete(i));const a=e?e.querySelector(".upl\
oad-marker-tag"):null;a&&a.classList.toggle("hidden",!n)}r(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){
return markerAppliedUploads.size>0}r(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){
const n=String(e||"").trim().toLowerCase();return n==="library"||n==="lib"?"library":n==="upload"||n===
"uploaded"?"upload":"unknown"}r(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let n=String(e).replace(/\u0000/g,"");return n=n.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),n=n.trim(),!n||(n=n.split("/").pop().split("\\").pop().trim(),n=n.replace(/\s{2,}/g,
" "),n=n.replace(/[<>:"/\\|?*]+/g,"_"),!n||n==="."||n==="..")?"":(n.length>180&&(n=n.slice(0,180).trim()),
n)}r(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const n=normalizeAttachmentPath(e);return n?n.split("/").pop()||n:""}r(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,n){const i=normalizeAttachmentPath(
e);if(!i)return;const a=normalizeAttachmentDisplayName(n)||defaultAttachmentDisplayName(i);a&&attachmentNameByPath.
set(i,a)}r(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const n=normalizeAttachmentPath(e);if(!n)return"";const i=normalizeAttachmentDisplayName(attachmentNameByPath.
get(n));return i||defaultAttachmentDisplayName(n)}r(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,n){if(!e)return;const i=normalizeAttachmentDisplayName(n)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=i;const a=e.querySelector(".truncate");
a&&(a.textContent=i);const o=e.getAttribute("data-filename");o&&setAttachmentNameForPath(o,i)}r(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}r(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,n){
e&&(e.dataset.sendNameCustomized=n?"1":"")}r(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const n=e.getAttribute("data-fi\
lename");if(n)return defaultAttachmentDisplayName(n)||"file";const i=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return i||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
r(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const n=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",i=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
n);if(i===null)return;const a=normalizeAttachmentDisplayName(i);if(!a){const o=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,o),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,a),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}r(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const n=e.getAttribute("data-filename"),i=getAttachmentNameForPath(n);if(i)return i;const a=normalizeAttachmentDisplayName(
e.dataset.displayName);if(a)return a;const o=e.querySelector(".truncate"),l=normalizeAttachmentDisplayName(
o?o.textContent:"");return l||getAttachmentNameForPath(n)}r(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,n){const i=normalizeAttachmentPath(e);if(!i)return;const a=normalizeAttachmentSource(
n);a!=="unknown"&&attachmentSourceByPath.set(i,a)}r(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const n=normalizeAttachmentPath(e);return n?normalizeAttachmentSource(
attachmentSourceByPath.get(n)):"unknown"}r(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,n){if(!e)return;const i=normalizeAttachmentSource(n);e.dataset.fileSource=
i;const a=e.getAttribute("data-filename");a&&setAttachmentSourceForPath(a,i)}r(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const n=normalizeAttachmentSource(
e.dataset.fileSource);if(n!=="unknown")return n;const i=e.getAttribute("data-filename");return getAttachmentSourceForPath(
i)}r(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const n=normalizeAttachmentSource(e.dataset.originalSource);if(n!=="unknown")return n;
const i=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(i)}r(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,n,i){const a=document.createElement(
"canvas");a.width=n,a.height=i;const o=a.getContext("2d");o?(o.drawImage(e,0,0,n,i),markerState.baseImageData=
o.getImageData(0,0,n,i),markerState.baseCanvas=a):(markerState.baseImageData=null,markerState.baseCanvas=
null)}r(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const n=e.getContext("2d");if(!n)return;n.clearRect(0,0,e.width,e.
height);const i=r((c,d,m=null,h=!1)=>{if(!c)return;const y=Math.max(0,c.x),v=Math.max(0,c.y),x=Math.
max(1,c.w),w=Math.max(1,c.h);m&&(n.fillStyle=m,n.fillRect(y,v,x,w)),n.save(),h&&n.setLineDash([6,4]),
n.strokeStyle=d,n.lineWidth=2,n.strokeRect(y+.5,v+.5,Math.max(1,x-1),Math.max(1,w-1)),n.restore()},"\
drawRect"),a=markerState.cropRect,o=a&&a.x===0&&a.y===0&&Math.abs(a.w-e.width)<1&&Math.abs(a.h-e.height)<
1;if(a&&(markerState.mode==="crop"||!o)){n.fillStyle="rgba(0,0,0,0.35)",n.fillRect(0,0,e.width,e.height);
const c=Math.max(0,a.x),d=Math.max(0,a.y),m=Math.max(1,a.w),h=Math.max(1,a.h);n.clearRect(c,d,m,h),markerState.
mode==="crop"?i(a,"rgba(250,204,21,0.9)"):i(a,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(c=>i(c,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&i(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}r(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
r(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
n=new Map,i=r((o,l,c)=>{const d=normalizeAttachmentPath(o);if(!d)return;const m=normalizeAttachmentSource(
l),h=normalizeAttachmentDisplayName(c)||getAttachmentNameForPath(d),y=n.get(d);if(y===void 0){const w=e.
length;n.set(d,w),e.push({path:d,source:m,name:h});return}const v=e[y];if(!v)return;const x=normalizeAttachmentSource(
v.source);(x==="unknown"&&m!=="unknown"||x==="library"&&m==="upload")&&(v.source=m),!normalizeAttachmentDisplayName(
v.name)&&h&&(v.name=h)},"pushItem"),a=get("upload-list");return a&&a.querySelectorAll("[data-filenam\
e]").forEach(o=>{const l=o.getAttribute("data-filename");i(l,getRowAttachmentSource(o),getRowAttachmentName(
o));const c=o.getAttribute("data-original-filename");o.dataset.attachOriginal==="1"&&i(c,getRowOriginalAttachmentSource(
o),getAttachmentNameForPath(c))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
o=>{i(o,getAttachmentSourceForPath(o),getAttachmentNameForPath(o))}),e}r(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}r(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const n=getModelMediaSupport(
get("model-select").value);let i=0,a=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const l=[];currentImageUrls.forEach(c=>{const d=normalizeAttachmentPath(c);if(!d)return;const m=isAudioPath(
d),h=isVideoPath(d);if(m&&!n.audio||h&&!n.video){m&&(i+=1),h&&(a+=1);return}l.push(d)}),l.length!==currentImageUrls.
length&&(currentImageUrls=l)}const o=get("upload-list");if(o&&(o.querySelectorAll("[data-filename]").
forEach(l=>{const c=l.getAttribute("data-filename");c&&!currentImageUrls.includes(c)&&(isAudioPath(c)||
isVideoPath(c))&&(setRowMarkerState(l,!1),l.remove())}),o.children.length===0&&(o.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(i||a)){const l=[];i&&l.push(`${i}\u4EF6\u306E\u97F3\u58F0`),a&&l.push(`${a}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${l.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}r(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const n=e.getAttribute("data-local-url");if(n)return n;const i=e.getAttribute("data-f\
ilename");return i?buildFileUrl(i):""}r(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const n=normalizeAttachmentPath(e);return n?FILE_BASE_URL+n:""}r(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const n=normalizeAttachmentPath(e);return n?isImagePath(n)?FILE_THUMB_BASE_URL+n:FILE_BASE_URL+n:""}
r(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e){
const n=getRowImageSource(e);if(!n){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e;const i=e?e.querySelector(".truncate"):null;markerState.filename=
i?i.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=[],markerState.naturalWidth=
0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=[],markerState.mosaicPreviewRect=
null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode("draw");const a=get("m\
arker-attach-original");a&&(a.checked=e.dataset.attachOriginal==="1");const o=get("marker-image"),l=get(
"marker-canvas"),c=get("marker-crop-canvas");if(l){const d=l.getContext("2d");d&&d.clearRect(0,0,l.width,
l.height)}if(c){const d=c.getContext("2d");d&&d.clearRect(0,0,c.width,c.height)}resetMarkerTransform(),
showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState({modal:"marker"},"","\
/edit-image"),o&&(o.onload=()=>{if(!get("marker-stage")||!l)return;const m=Math.max(1,Math.floor(o.clientWidth)),
h=Math.max(1,Math.floor(o.clientHeight));l.width=m,l.height=h,l.style.width=`${m}px`,l.style.height=
`${h}px`,l.style.left="0px",l.style.top="0px",c&&(c.width=m,c.height=h,c.style.width=`${m}px`,c.style.
height=`${h}px`,c.style.left="0px",c.style.top="0px"),markerState.naturalWidth=o.naturalWidth||m,markerState.
naturalHeight=o.naturalHeight||h;const y=l.getContext("2d");y&&y.clearRect(0,0,l.width,l.height),prepareMarkerBaseCanvas(
o,m,h),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},o.src=n)}r(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,n){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=n,updateFilePreview())}
r(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(c=>{const d=c&&c.rowObj?c.rowObj.row:null,m=d?d.getAttribute("data-local-url"):null;m&&URL.revokeObjectURL(
m)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.clear();const e=get(
"file-preview");e&&e.classList.add("hidden");const n=get("file-preview-thumbs");n&&(n.innerHTML="",n.
classList.add("hidden")),updateFilePreview(),updateMaskPreview();const i=get("upload-list");i&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>');
const a=get("file-input");a&&(a.value="");const o=get("photo-input");o&&(o.value="");const l=get("ma\
sk-input");l&&(l.value="")}r(resetUploadState,"resetUploadState");async function uploadMaskFile(e){if(!e)
return;const n=new FormData;n.append("file",e);try{const i=await fetch(CHAT_CONFIG.urls.upload,{method:"\
POST",body:n}),a=await i.json();i.ok&&a.filename?(currentMaskImage=a.filename,updateMaskPreview()):showToast(
a.error||"Mask upload failed","error",!0)}catch{showToast("Mask upload failed","error",!0)}}r(uploadMaskFile,
"uploadMaskFile");function setCameraCaptureStatus(e,n=!1){const i=get("camera-status");i&&(i.textContent=
e||"",i.classList.toggle("text-red-300",!!n),i.classList.toggle("text-gray-400",!n))}r(setCameraCaptureStatus,
"setCameraCaptureStatus");function updateCameraCapturePendingUi(){const e=cameraCapturePendingFiles.
length,n=get("camera-attach-btn");n&&(n.disabled=e===0||cameraCaptureBusy,n.textContent=e?`\u6DFB\u4ED8 (${e}\
)`:"\u6DFB\u4ED8 (0)");const i=get("camera-clear-btn");i&&(i.disabled=e===0||cameraCaptureBusy);const a=get(
"camera-capture-preview-list");a&&(a.innerHTML="",cameraCapturePendingPreviewUrls.forEach((o,l)=>{const c=document.
createElement("div");c.className="relative rounded overflow-hidden border border-gray-700 bg-black a\
spect-square",c.innerHTML=`
                        <img src="${o}" alt="capture ${l+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${l+1}</div>
                    `,a.appendChild(c)}),a.classList.toggle("hidden",e===0))}r(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const n=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(n)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}r(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(a=>{try{a.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const n=get("camera-capture-btn");n&&(n.disabled=!0);const i=get(
"camera-switch-btn");i&&(i.disabled=!0)}r(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const n=get("camera-video");if(!n)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const i=get(
"camera-switch-btn");i&&(i.disabled=!0);const a=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let o=null;for(const l of a)
try{const c=await navigator.mediaDevices.getUserMedia(l);cameraCaptureStream=c,n.srcObject=c;try{await n.
play()}catch{}const d=c.getVideoTracks&&c.getVideoTracks()[0],m=d&&d.getSettings?d.getSettings():{},
h=String(m.facingMode||"").toLowerCase();h==="user"||h==="environment"?cameraCaptureFacingMode=h:cameraCaptureFacingMode=
e;const y=get("camera-capture-btn");return y&&(y.disabled=!1),i&&(i.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),c}catch(c){o=c}throw o||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
r(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const n=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(n,!0),showToast(n,"error",!0);const i=get("camera-capture-btn");i&&(i.disabled=
!0);const a=get("camera-attach-btn");a&&(a.disabled=!0)}}r(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const n=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!n&&location.pathname==="/camera"&&history.back()}r(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const n=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=n;try{await startCameraCaptureStream(n)}catch(i){const a=i&&
i.message?i.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
a,!0),showToast(a,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}r(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,n=r(o=>String(o).padStart(2,"0"),"pad"),i=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const a=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${n(e.getMonth()+1)}${n(e.getDate())}_${n(e.getHours())}${n(e.getMinutes())}${n(
e.getSeconds())}_${i}_${a}.jpg`}r(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),n=get("camera-canvas"),i=get("camera-capture\
-modal");if(!e||!n||!i)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const a=get("camera-capture-btn");
a&&(a.disabled=!0);const o=get("camera-attach-btn");o&&(o.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{n.width=e.videoWidth,n.height=e.videoHeight;const l=n.getContext("2d");if(!l)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");l.drawImage(e,0,0,n.width,n.height);
const c=await new Promise((m,h)=>{n.toBlob(y=>{y?m(y):h(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),d=new File([c],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(d),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(c)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(l){
const c=l&&l.message?l.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
c,!0),showToast(c,"error",!0)}finally{cameraCaptureBusy=!1,a&&i&&!i.classList.contains("hidden")&&(a.
disabled=!1),updateCameraCapturePendingUi()}}r(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const n=get("camera-ca\
pture-btn"),i=get("camera-switch-btn"),a=get("camera-attach-btn"),o=get("camera-clear-btn");n&&(n.disabled=
!0),i&&(i.disabled=!0),a&&(a.disabled=!0),o&&(o.disabled=!0);const l=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${l.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{await handleFiles(l,{openModal:!1}),showToast(`${l.length}\u679A\u306E\
\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(c){const d=c&&c.message?c.message:
"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";showToast(d,"\
error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.classList.contains(
"hidden")&&(n&&(n.disabled=!1),i&&(i.disabled=!1),updateCameraCapturePendingUi())}}r(attachCameraCapturedFiles,
"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay=="function"&&window.
hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.pathname!=="/upload"&&
history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-info");if(e){const i=(get(
"model-select")?get("model-select").value:"").toLowerCase(),a=i==="deepseek-v4.1-flash"||i==="deepse\
ek-v4-flash-vision-exp",o=i.includes("deepseek")&&!a;e.classList.toggle("hidden",!o)}_syncVisionModelDisplay()}
r(openUploadModal,"openUploadModal");function _syncVisionModelDisplay(){const e=get("vision-model-di\
splay");if(!e)return;const n=currentVisionModel;if(n){let i=n;MODELS.forEach(a=>(a.items||[]).forEach(
o=>{o.id===n&&(i=o.name)})),e.textContent=i}else e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}
r(_syncVisionModelDisplay,"_syncVisionModelDisplay");function _openVisionModelSelector(){window._visionPickerActive=
!0,openModelModal(),setTimeout(()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},
50)}r(_openVisionModelSelector,"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.
hideDropOverlay=="function"&&window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname===
"/upload"&&history.back()}r(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){
const e=get("upload-list");if(!e)return;const n=new Set;e.querySelectorAll("[data-filename]").forEach(
i=>{const a=i.getAttribute("data-filename");a&&n.add(a)}),currentImageUrls.forEach(i=>{n.has(i)||addStoredUploadRow(
i,{source:getAttachmentSourceForPath(i),displayName:getAttachmentNameForPath(i)})}),e.children.length===
0&&(e.innerHTML='<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
r(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}r(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,n={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const i=normalizeAttachmentSource(n.source),a=get(
"upload-list");if(!a)return null;a.children.length===1&&a.children[0].classList.contains("text-gray-\
500")&&(a.innerHTML="");const o=e.split("/").pop()||e,l=normalizeAttachmentDisplayName(n.displayName)||
getAttachmentNameForPath(e)||o,c=(o.split(".").pop()||"").toLowerCase(),d=["png","jpg","jpeg","webp",
"gif"].includes(c),m=buildFileUrl(e),h=d?buildAttachmentPreviewUrl(e):m,y=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,v=document.createElement("div");v.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",v.dataset.uploadId=y,v.setAttribute("data-filename",e),v.dataset.fileSource=
i,v.dataset.displayName=l,v.dataset.defaultDisplayName=l,v.dataset.sendNameCustomized="";const x=escapeHtml(
l),w=d&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",_=d?`<img src="${h}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${x}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';v.innerHTML=`
                <div class="flex items-center gap-3">
                    ${_}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${x}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${w}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;const S=v.querySelector(".upload-preview");S&&(S.onclick=()=>openFileViewer(m,getRowAttachmentName(
v)||l));const L=v.querySelector(".upload-send-name");L&&(L.onclick=()=>promptRowAttachmentName(v));const M=v.
querySelector(".upload-remove");M&&(M.onclick=()=>{uploadCancelTokens.add(y),browserFastLocalFiles.delete(
y),decrementUploadTotal(y);const H=v.getAttribute("data-filename");H&&(currentImageUrls=currentImageUrls.
filter(Q=>Q!==H)),setRowMarkerState(v,!1),v.remove(),updateFilePreview(),a.children.length===0&&(a.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const P=v.querySelector(".upload-marker");return P&&(P.onclick=()=>openMarkerModalForRow(v)),setAttachmentSourceForPath(
e,i),setAttachmentNameForPath(e,l),a.prepend(v),{row:v,bar:v.querySelector(".upload-progress > div"),
status:v.querySelector(".upload-status"),uploadId:y}}r(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const n=get("upload-list");if(!n)return null;n.children.length===1&&n.children[0].classList.contains(
"text-gray-500")&&(n.innerHTML="");const i=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
a=document.createElement("div");a.className="upload-row ui-enter bg-gray-900/60 rounded p-2",a.dataset.
uploadId=i,a.dataset.fileSource="upload";const o=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";a.dataset.displayName=o,a.dataset.defaultDisplayName=o,a.dataset.sendNameCustomized="";const l=escapeHtml(
o),c=e&&e.type&&e.type.startsWith("image/");let d='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const m=c&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let h="";c?(h=URL.createObjectURL(e),d=`<img src="${h}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${l}">`):(h=URL.createObjectURL(e),d=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),a.innerHTML=`
                <div class="flex items-center gap-3">
                    ${d}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${l}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">\u5F85\u6A5F\u4E2D</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${m}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `,h&&a.setAttribute("data-local-url",h);const y=a.querySelector(".upload-preview");y&&(y.
onclick=()=>{const _=a.getAttribute("data-filename"),S=_?buildFileUrl(_):a.getAttribute("data-local-\
url"),L=normalizeAttachmentDisplayName(a.dataset.displayName)||e.name||_||"";openFileViewer(S,L)});const v=a.
querySelector(".upload-remove");v&&(v.onclick=()=>{uploadCancelTokens.add(i),browserFastLocalFiles.delete(
i),decrementUploadTotal(i);const _=a.getAttribute("data-local-url");_&&URL.revokeObjectURL(_);const S=a.
getAttribute("data-filename");S&&(currentImageUrls=currentImageUrls.filter(L=>L!==S)),setRowMarkerState(
a,!1),a.remove(),updateFilePreview(),n.children.length===0&&(n.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const x=a.querySelector(".upload-marker");x&&(x.onclick=()=>openMarkerModalForRow(a));const w=a.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(a)),n.prepend(a),{uploadId:i,row:a,
status:a.querySelector(".upload-status"),bar:a.querySelector(".upload-progress > div")}}r(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,n){if(!e)
return!1;let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),i=!0);try{const a=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),o=await a.json();if(!a.ok){const v=o&&o.error?o.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return n&&n.status&&(n.status.textContent=
"\u5931\u6557"),showToast(v,"error",!0),!1}const l=o.upload_id,c=o.chunk_size||10*1024*1024,d=Math.ceil(
e.size/c);for(let v=0;v<d;v++){const x=v*c,w=Math.min(e.size,x+c),_=e.slice(x,w);if(!await new Promise(
L=>{const M=new XMLHttpRequest;M.open("POST","/upload/chunk",!0),M.setRequestHeader("X-CSRF-Token",csrfToken),
M.upload.onprogress=H=>{if(H.lengthComputable&&n&&n.bar){const Q=x+H.loaded,ee=Math.min(100,Math.floor(
Q/e.size*100));n.bar.style.width=`${ee}%`,n.status&&(n.status.textContent=`${ee}%`),n.uploadId&&updateGlobalUploadProgress(
n.uploadId,ee)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},M.onload=()=>{M.
status>=200&&M.status<300?L(!0):L(!1)},M.onerror=()=>L(!1);const P=new FormData;P.append("upload_id",
l),P.append("index",String(v)),P.append("total",String(d)),P.append("chunk",_,e.name),M.send(P)}))return n&&
n.status&&(n.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}n&&n.status&&(n.status.textContent="\u51E6\u7406\u4E2D...");const m=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:l})}),h=await m.json();if(m.ok&&h&&h.filename){if(n&&n.row&&n.uploadId&&
uploadCancelTokens.has(n.uploadId))return n.row&&n.row.parentNode&&n.row.remove(),!1;if(n&&n.row){const w=n.
row.getAttribute("data-local-url");w&&URL.revokeObjectURL(w),n.row.removeAttribute("data-local-url");
const _=n.row.querySelector("img.upload-preview");if(_){const S=h.filename.replace(/^\d+\//,"");_.src=
buildAttachmentPreviewUrl(S)}}const v=normalizeAttachmentPath(h.filename);if(v&&currentImageUrls.push(
v),n&&n.row&&(n.row.setAttribute("data-filename",v||h.filename),setRowAttachmentSource(n.row,"upload"),
v)){const w=isRowAttachmentNameCustomized(n.row),_=defaultAttachmentDisplayName(v),S=w&&normalizeAttachmentDisplayName(
n.row.dataset.displayName)||_;n.row.dataset.defaultDisplayName=_,setRowAttachmentName(n.row,S)}return v&&
setAttachmentSourceForPath(v,"upload"),n&&n.status&&(n.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(h.filenames)&&h.filenames.length?h.filenames:[h.filename]).forEach(w=>addLibraryFileFromPath(
w)),!0}const y=h&&h.error?h.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return n&&n.status&&(n.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}catch{return n&&
n.status&&(n.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{i&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}r(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,n){return new Promise(i=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,n).then(i);return}let a=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
a=!0);const o=r(()=>{a&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),a=!1)},
"finishUploadOp"),l=new XMLHttpRequest;l.open("POST",CHAT_CONFIG.urls.upload,!0),l.setRequestHeader(
"X-CSRF-Token",csrfToken),l.upload.onprogress=d=>{if(d.lengthComputable&&n&&n.bar){const m=Math.min(
100,Math.floor(d.loaded/d.total*100));n.bar.style.width=`${m}%`,n.status&&(n.status.textContent=`${m}\
%`),n.uploadId&&updateGlobalUploadProgress(n.uploadId,m)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},l.onload=()=>{let d={};try{d=JSON.parse(l.responseText||"{}")}catch{}if(l.status>=
200&&l.status<300&&d&&d.filename){if(n&&n.row&&n.uploadId&&uploadCancelTokens.has(n.uploadId)){n.row&&
n.row.parentNode&&n.row.remove(),o(),i(!1);return}if(n&&n.row){const y=n.row.getAttribute("data-loca\
l-url");y&&URL.revokeObjectURL(y),n.row.removeAttribute("data-local-url");const v=n.row.querySelector(
"img.upload-preview");if(v){const x=d.filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(
x)}}const m=normalizeAttachmentPath(d.filename);if(m&&currentImageUrls.push(m),n&&n.row&&(n.row.setAttribute(
"data-filename",m||d.filename),setRowAttachmentSource(n.row,"upload"),m)){const y=isRowAttachmentNameCustomized(
n.row),v=defaultAttachmentDisplayName(m),x=y&&normalizeAttachmentDisplayName(n.row.dataset.displayName)||
v;n.row.dataset.defaultDisplayName=v,setRowAttachmentName(n.row,x)}m&&setAttachmentSourceForPath(m,"\
upload"),n&&n.status&&(n.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(d.filenames)&&
d.filenames.length?d.filenames:[d.filename]).forEach(y=>addLibraryFileFromPath(y)),o(),i(!0)}else{const m=d&&
d.error?d.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";n&&
n.status&&(n.status.textContent="\u5931\u6557"),showToast(m,"error",!0),o(),i(!1)}},l.onerror=()=>{n&&
n.status&&(n.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),o(),i(!1)};const c=new FormData;c.append("file",e),l.send(c)})}r(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}r(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}r(isAudioFile,"i\
sAudioFile");function encodeWav(e,n){let i=0;e.forEach(h=>{i+=h.length});const a=new Float32Array(i);
let o=0;e.forEach(h=>{a.set(h,o),o+=h.length});const l=new ArrayBuffer(44+a.length*2),c=new DataView(
l),d=r((h,y)=>{for(let v=0;v<y.length;v++)c.setUint8(h+v,y.charCodeAt(v))},"writeString");d(0,"RIFF"),
c.setUint32(4,36+a.length*2,!0),d(8,"WAVE"),d(12,"fmt "),c.setUint32(16,16,!0),c.setUint16(20,1,!0),
c.setUint16(22,1,!0),c.setUint32(24,n,!0),c.setUint32(28,n*2,!0),c.setUint16(32,2,!0),c.setUint16(34,
16,!0),d(36,"data"),c.setUint32(40,a.length*2,!0);let m=44;for(let h=0;h<a.length;h++){const y=Math.
max(-1,Math.min(1,a[h]));c.setInt16(m,y<0?y*32768:y*32767,!0),m+=2}return new Blob([c],{type:"audio/\
wav"})}r(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const n of e)
if(MediaRecorder.isTypeSupported(n))return n;return""}r(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,n){if(!e||!e.row||!n)return;const i=e.row.querySelector(".truncate"),
a=isRowAttachmentNameCustomized(e.row),o=a?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(n.name||"file")||"file";i&&(i.textContent=o),e.row.dataset.displayName=
o,a||(e.row.dataset.defaultDisplayName=o);const l=e.row.getAttribute("data-local-url");l&&URL.revokeObjectURL(
l);const c=URL.createObjectURL(n);e.row.setAttribute("data-local-url",c);const d=n.type&&n.type.startsWith(
"image/"),m=escapeHtml(o),h=d?`<img src="${c}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${m}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',y=e.row.querySelector(".upload-preview");y&&(y.outerHTML=h);const v=e.row.querySelector(
".upload-preview");v&&(v.onclick=()=>{const w=e.row.getAttribute("data-filename"),_=w?buildFileUrl(w):
e.row.getAttribute("data-local-url");openFileViewer(_,getRowAttachmentName(e.row)||o||w||"")});const x=e.
row.querySelector(".upload-marker");x&&x.classList.toggle("hidden",!d),d||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}r(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const n=e.getContext("2d");if(!n)return;const i=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(a=>({x:a.x,y:a.y,w:a.w,h:a.h})):[];markerState.history.push({imageData:n.
getImageData(0,0,e.width,e.height),mosaicRects:i}),markerState.history.length>40&&markerState.history.
shift()}r(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const n=e.getContext(
"2d");if(!n)return;const i=markerState.history[markerState.history.length-1];n.clearRect(0,0,e.width,
e.height),i&&i.imageData?(n.putImageData(i.imageData,0,0),markerState.mosaicRects=Array.isArray(i.mosaicRects)?
i.mosaicRects.map(a=>({x:a.x,y:a.y,w:a.w,h:a.h})):[]):i?(n.putImageData(i,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}r(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const n=e.getContext("2d");n&&n.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}r(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const n=e.getContext("2d"),i=get("marker-size"),a=new Map;let o=!1,l=0,
c=markerView.scale,d={x:0,y:0},m={x:0,y:0},h=[],y=16,v="",x=null,w=null,_=null,S=null,L=!1,M=null;const P=r(
E=>{const j=e.getBoundingClientRect(),W=(E.clientX-j.left)*(e.width/j.width),Z=(E.clientY-j.top)*(e.
height/j.height);return{x:W,y:Z}},"getPoint"),H=r((E,j)=>({x:(E.x+j.x)/2,y:(E.y+j.y)/2}),"getMid"),Q=r(
(E,j)=>Math.hypot(E.x-j.x,E.y-j.y),"getDist");let ee=!1;const Ae=r(()=>{x||(x=document.createElement(
"canvas"),w=x.getContext("2d")),_||(_=document.createElement("canvas"),S=_.getContext("2d")),(x.width!==
e.width||x.height!==e.height)&&(x.width=e.width,x.height=e.height),(_.width!==e.width||_.height!==e.
height)&&(_.width=e.width,_.height=e.height)},"ensureDrawBuffers"),B=r(()=>{if(!n||!x||!_)return;const E=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));n.clearRect(0,0,e.width,e.
height),n.drawImage(x,0,0),n.save(),n.globalAlpha=E,n.drawImage(_,0,0),n.restore()},"renderDrawPrevi\
ew"),V=r(()=>{S&&(S.strokeStyle=v,S.fillStyle=v,S.lineWidth=y,S.lineCap="round",S.lineJoin="round")},
"applyMarkerBrush"),te=r(E=>{if(!E)return!1;if(h.length===0)return h.push(E),!0;const j=h[h.length-1],
W=E.x-j.x,Z=E.y-j.y,ie=Math.hypot(W,Z),Se=Math.max(.35,y*.04);if(ie<Se)return!1;const ne=Math.max(1,
y*.25),ue=Math.max(1,Math.ceil(ie/ne));for(let J=1;J<=ue;J++){const ve=J/ue;h.push({x:j.x+W*ve,y:j.y+
Z*ve})}return!0},"appendStrokePoint"),ge=r(()=>{if(S&&(S.clearRect(0,0,_.width,_.height),h.length!==
0)){if(V(),h.length===1){const E=h[0];S.beginPath(),S.arc(E.x,E.y,y/2,0,Math.PI*2),S.fill();return}if(S.
beginPath(),S.moveTo(h[0].x,h[0].y),h.length===2)S.lineTo(h[1].x,h[1].y);else{for(let W=1;W<h.length-
2;W++){const Z=h[W],ie=h[W+1],Se=H(Z,ie);S.quadraticCurveTo(Z.x,Z.y,Se.x,Se.y)}const E=h[h.length-2],
j=h[h.length-1];S.quadraticCurveTo(E.x,E.y,j.x,j.y)}S.stroke()}},"renderStrokeLayer"),de=r((E,j)=>{if(!E||
!j)return null;const W=Math.min(E.x,j.x),Z=Math.min(E.y,j.y),ie=Math.abs(E.x-j.x),Se=Math.abs(E.y-j.
y);return{x:W,y:Z,w:ie,h:Se}},"normalizeMosaicRect"),Ce=r(E=>{const j=i?Number(i.value||16):16,W=Math.
max(6,Math.floor(j)),Z=Math.floor(W/2);return{x:E.x-Z,y:E.y-Z,w:W,h:W}},"buildMosaicRectFromPoint"),
be=r(()=>{const E=document.createElement("canvas");E.width=e.width,E.height=e.height;const j=E.getContext(
"2d");if(!j)return null;markerState.baseCanvas&&j.drawImage(markerState.baseCanvas,0,0),j.drawImage(
e,0,0);try{return j.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),
Le=r(E=>{if(!n||!E)return!1;const j=be();if(!j)return!1;const W=i?Number(i.value||16):16,Z=Math.max(
4,Math.floor(W/2)),ie=Math.max(0,Math.floor(E.x)),Se=Math.max(0,Math.floor(E.y)),ne=Math.min(e.width,
Math.ceil(E.x+E.w)),ue=Math.min(e.height,Math.ceil(E.y+E.h));if(ne<=ie||ue<=Se)return!1;for(let J=Se;J<
ue;J+=Z)for(let ve=ie;ve<ne;ve+=Z){const at=Math.min(Z,ne-ve),Ee=Math.min(Z,ue-J),lt=Math.min(e.width-
1,Math.max(0,ve+Math.floor(at/2))),Qe=(Math.min(e.height-1,Math.max(0,J+Math.floor(Ee/2)))*e.width+lt)*
4,ut=j.data[Qe],St=j.data[Qe+1],wt=j.data[Qe+2];n.fillStyle=`rgb(${ut},${St},${wt})`,n.fillRect(ve,J,
at,Ee)}return!0},"applyMosaicRect"),oe=r(E=>{if(!n)return;if(a.set(E.pointerId,{x:E.clientX,y:E.clientY}),
a.size>=2){const W=Array.from(a.values()),Z=W[0],ie=W[1];o=!0,ee=!1,h=[],L=!1,M=null,markerState.mosaicPreviewRect=
null,l=Q(Z,ie)||1,c=markerView.scale,d={x:markerView.offsetX,y:markerView.offsetY},m=H(Z,ie),renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(E.pointerId),E.preventDefault();return}if(o||markerState.mode===
"crop")return;ee=!0;const j=P(E);if(markerState.mode==="mosaic")L=!0,M=j,markerState.mosaicPreviewRect=
Ce(j),renderCropOverlay();else{if(Ae(),!w||!S)return;w.clearRect(0,0,x.width,x.height),w.drawImage(e,
0,0),S.clearRect(0,0,_.width,_.height),y=i?Number(i.value||16):16,v=normalizeMarkerHexColor(markerState.
colorHex),h=[],te(j),ge(),markerState.hasStroke=!0,B()}e.setPointerCapture&&e.setPointerCapture(E.pointerId),
E.preventDefault()},"start"),le=r(E=>{if(a.has(E.pointerId)&&a.set(E.pointerId,{x:E.clientX,y:E.clientY}),
o&&a.size>=2){const W=Array.from(a.values()),Z=W[0],ie=W[1],Se=H(Z,ie),ne=Q(Z,ie)||1,ue=c*(ne/l);markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,ue)),markerView.offsetX=d.x+(Se.x-m.
x),markerView.offsetY=d.y+(Se.y-m.y),applyMarkerTransform(),E.preventDefault();return}if(!ee||!n)return;
const j=P(E);if(markerState.mode==="mosaic"){if(!L||!M)return;markerState.mosaicPreviewRect=de(M,j)||
Ce(j),renderCropOverlay()}else te(j)&&(ge(),B());E.preventDefault()},"move"),Y=r(E=>{const j=ee;if(a.
delete(E.pointerId),a.size<2&&(o=!1),a.size===0){if(ee=!1,j&&n&&markerState.mode==="draw"&&h.length>
0&&(ge(),B()),j&&markerState.mode==="mosaic"&&M){const W=P(E);let Z=de(M,W);(!Z||Z.w<2||Z.h<2)&&(Z=Ce(
M)),Le(Z)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(Z))}h=[],L=!1,M=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),j&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(E.pointerId),
E.preventDefault()},"end");e.addEventListener("pointerdown",oe),e.addEventListener("pointermove",le),
e.addEventListener("pointerup",Y),e.addEventListener("pointercancel",Y)}r(initMarkerCanvas,"initMark\
erCanvas");function initCropCanvas(){const e=get("marker-crop-canvas");if(!e)return;const n=e.getContext(
"2d"),i=new Map;let a=!1,o=null,l=null,c=null,d=!1,m=0,h=markerView.scale,y={x:0,y:0},v={x:0,y:0};const x=8,
w=14,_=r((B,V,te)=>Math.min(te,Math.max(V,B)),"clamp"),S=r(B=>{const V=e.getBoundingClientRect(),te=(B.
clientX-V.left)*(e.width/V.width),ge=(B.clientY-V.top)*(e.height/V.height);return{x:te,y:ge}},"getPo\
int"),L=r((B,V)=>({x:(B.x+V.x)/2,y:(B.y+V.y)/2}),"getMid"),M=r((B,V)=>Math.hypot(B.x-V.x,B.y-V.y),"g\
etDist"),P=r(()=>(markerState.cropRect||resetCropRectToFull(),markerState.cropRect),"ensureCropRect"),
H=r((B,V)=>{if(!V)return"move";const te=V.x,ge=V.y,de=V.x+V.w,Ce=V.y+V.h,be=Math.abs(B.x-te)<=w,Le=Math.
abs(B.x-de)<=w,oe=Math.abs(B.y-ge)<=w,le=Math.abs(B.y-Ce)<=w;if(be&&oe)return"nw";if(Le&&oe)return"n\
e";if(be&&le)return"sw";if(Le&&le)return"se";if(oe)return"n";if(le)return"s";if(be)return"w";if(Le)return"\
e";if(B.x>te+w&&B.x<de-w&&B.y>ge+w&&B.y<Ce-w)return"move";const E=B.x<te?"left":B.x>de?"right":null,
j=B.y<ge?"top":B.y>Ce?"bottom":null;if(E&&j){if(E==="left"&&j==="top")return"nw";if(E==="right"&&j===
"top")return"ne";if(E==="left"&&j==="bottom")return"sw";if(E==="right"&&j==="bottom")return"se"}return E?
E==="left"?"w":"e":j?j==="top"?"n":"s":"move"},"hitTest"),Q=r(B=>{if(markerState.mode!=="crop")return;
if(i.set(B.pointerId,{x:B.clientX,y:B.clientY}),i.size>=2){const ge=Array.from(i.values()),de=ge[0],
Ce=ge[1];d=!0,a=!1,m=M(de,Ce)||1,h=markerView.scale,y={x:markerView.offsetX,y:markerView.offsetY},v=
L(de,Ce),e.setPointerCapture&&e.setPointerCapture(B.pointerId),B.preventDefault();return}if(d)return;
a=!0;const V=S(B),te=P();l=H(V,te),o=V,c=te?{x:te.x,y:te.y,w:te.w,h:te.h}:null,renderCropOverlay(),e.
setPointerCapture&&e.setPointerCapture(B.pointerId),B.preventDefault()},"start"),ee=r(B=>{if(markerState.
mode!=="crop")return;if(i.has(B.pointerId)&&i.set(B.pointerId,{x:B.clientX,y:B.clientY}),d&&i.size>=
2){const E=Array.from(i.values()),j=E[0],W=E[1],Z=L(j,W),ie=M(j,W)||1,Se=h*(ie/m);markerView.scale=Math.
min(markerView.maxScale,Math.max(markerView.minScale,Se)),markerView.offsetX=y.x+(Z.x-v.x),markerView.
offsetY=y.y+(Z.y-v.y),applyMarkerTransform(),renderCropOverlay(),B.preventDefault();return}if(!a||!o||
!c)return;const V=S(B),te=e.width,ge=e.height,de={x:c.x,y:c.y,w:c.w,h:c.h},Ce=c.x+c.w,be=c.y+c.h,Le=r(
()=>{const E=_(V.x,0,Ce-x);de.x=E,de.w=Ce-E},"applyW"),oe=r(()=>{de.w=_(V.x-c.x,x,te-c.x)},"applyE"),
le=r(()=>{const E=_(V.y,0,be-x);de.y=E,de.h=be-E},"applyN"),Y=r(()=>{de.h=_(V.y-c.y,x,ge-c.y)},"appl\
yS");switch(l){case"move":{const E=V.x-o.x,j=V.y-o.y;de.x=_(c.x+E,0,te-c.w),de.y=_(c.y+j,0,ge-c.h);break}case"\
w":Le();break;case"e":oe();break;case"n":le();break;case"s":Y();break;case"nw":le(),Le();break;case"\
ne":le(),oe();break;case"sw":Y(),Le();break;case"se":Y(),oe();break;default:break}de.x=_(de.x,0,te-de.
w),de.y=_(de.y,0,ge-de.h),markerState.cropRect=de,renderCropOverlay(),B.preventDefault()},"move"),Ae=r(
B=>{i.delete(B.pointerId),i.size<2&&(d=!1),i.size===0&&(renderCropOverlay(),a=!1,o=null,l=null,c=null),
e.releasePointerCapture&&e.releasePointerCapture(B.pointerId),B.preventDefault()},"end");e.addEventListener(
"pointerdown",Q),e.addEventListener("pointermove",ee),e.addEventListener("pointerup",Ae),e.addEventListener(
"pointercancel",Ae),e.addEventListener("pointerleave",Ae)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,n=get("marker-image"),i=get("marker-canvas");if(!e||!n||!i)return;const a=get(
"marker-attach-original");a&&(e.dataset.attachOriginal=a.checked?"1":"");let o=document.createElement(
"canvas");const l=markerState.naturalWidth||n.naturalWidth||i.width,c=markerState.naturalHeight||n.naturalHeight||
i.height;o.width=l,o.height=c;const d=o.getContext("2d");if(!d)return;if(d.drawImage(n,0,0,l,c),d.drawImage(
i,0,0,l,c),markerState.cropRect){const L=l/i.width,M=c/i.height,P=Math.max(0,Math.floor(markerState.
cropRect.x*L)),H=Math.max(0,Math.floor(markerState.cropRect.y*M)),Q=Math.min(l,Math.max(1,Math.floor(
markerState.cropRect.w*L))),ee=Math.min(c,Math.max(1,Math.floor(markerState.cropRect.h*M))),Ae=document.
createElement("canvas");Ae.width=Q,Ae.height=ee;const B=Ae.getContext("2d");B&&(B.drawImage(o,P,H,Q,
ee,0,0,Q,ee),o=Ae)}const m=await new Promise(L=>o.toBlob(L,"image/png",.92));if(!m){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\
\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}const y=(markerState.filename||
"marked.png").replace(/\.[^/.]+$/,""),v=new File([m],`${y}_marked.png`,{type:"image/png"}),x={row:e,
uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.querySelector(".upload-pr\
ogress > div")};x.status&&(x.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),updateUploadRowFile(
x,v);const w=e.getAttribute("data-filename"),_=getRowAttachmentSource(e);w&&!e.dataset.originalFilename&&
(e.dataset.originalFilename=w,e.dataset.originalSource=_,setAttachmentSourceForPath(w,_)),await uploadFileWithProgress(
v,x)?(w&&(currentImageUrls=currentImageUrls.filter(L=>L!==w)),setRowAttachmentSource(e,"upload"),setRowMarkerState(
e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}r(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,n){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(n&&n.status&&(n.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
i=>{const a=document.createElement("video");a.preload="auto",a.muted=!0,a.playsInline=!0,a.src=URL.createObjectURL(
e);let o=null,l=null,c=null,d=null,m=[],h=null;const y=r(()=>{h&&clearTimeout(h);try{URL.revokeObjectURL(
a.src)}catch{}try{a.remove()}catch{}if(o&&o.getTracks().forEach(x=>x.stop()),c)try{c.disconnect()}catch{}
if(d)try{d.disconnect()}catch{}if(l)try{l.close()}catch{}},"cleanup"),v=r(()=>{y(),i(null)},"fail");
a.onloadedmetadata=async()=>{try{o=a.captureStream();const x=o.getAudioTracks();if(!x||!x.length)return v();
l=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),d=l.createMediaStreamSource(
new MediaStream(x)),c=l.createScriptProcessor(4096,1,1),c.onaudioprocess=_=>{const S=_.inputBuffer.getChannelData(
0);m.push(new Float32Array(S))},d.connect(c),c.connect(l.destination);const w=isFinite(a.duration)?Math.
max(1,Math.ceil(a.duration*1e3)):0;w>0&&(h=setTimeout(()=>{const _=(e.name||"video").replace(/\.[^/.]+$/,
""),S=encodeWav(m,l.sampleRate),L=new File([S],`${_}.audio.wav`,{type:"audio/wav"});y(),i(L)},w+250)),
await a.play(),a.onended=()=>{const _=(e.name||"video").replace(/\.[^/.]+$/,""),S=encodeWav(m,l.sampleRate),
L=new File([S],`${_}.audio.wav`,{type:"audio/wav"});y(),i(L)}}catch{v()}},a.onerror=()=>v()}))}r(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,n={}){if(!e||!e.length)return;const i=Array.from(
e).filter(Boolean);if(!i.length)return;const a=collectImageUrlsForSend().length+browserFastLocalFiles.
size+Math.max(0,Number(uploadProgressState.active)||0);let o=i;if(a+i.length>ATTACHMENT_MAX_FILES){const v=Math.
max(0,ATTACHMENT_MAX_FILES-a);if(v<=0){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059`,"error",!0);return}o=i.slice(0,v),showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u5148\u982D${v}\u4EF6\u306E\u307F\u8FFD\u52A0\u3057\u307E\u3059\u3002`,"war\
ning",!0)}n.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=
o.length,uploadProgressState.active+=o.length,updateFilePreview();const l=!!(get("upload-audio-only")&&
get("upload-audio-only").checked),c=getModelMediaSupport(get("model-select").value),d=r(async v=>{let x=null;
try{if(isAudioFile(v)&&!c.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(v)&&!c.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!v.type||!v.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const w=addUploadRow(v);updateFilePreview(),x=w.uploadId,uploadProgressState.perFilePct[x]=
0;let _=v;if(l&&isVideoFile(v)){const S=await extractAudioFromVideo(v,w);S?(_=S,updateUploadRowFile(
w,S),w&&w.status&&(w.status.textContent="\u97F3\u58F0\u306E\u307F")):(w&&w.status&&(w.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&v.type.startsWith("image/"))try{const S=getCompressionOutputType();
if(getCompressionFormatOnly())_=await convertImageFormatOnly(v,S);else{const M={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};S&&S!=="original"&&(M.fileType=S),await ensureImageCompression();
const P=await window.imageCompression(v,M),H=new File([P],imageFilenameForMime(v.name,P.type||(S!=="\
original"?S:v.type)),{type:P.type||v.type,lastModified:v.lastModified||Date.now()});H.size>v.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
v.size)} -> ${formatBytes(H.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),_=v):_=H}_!==v&&updateUploadRowFile(w,_)}catch{}if(browserFastModeEnabled){const S=Array.from(
browserFastLocalFiles.values()).reduce((L,M)=>L+Number(M.file&&M.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||S+_.size>BROWSER_FAST_MAX_BYTES?(w&&w.status&&(w.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),w&&w.row&&w.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(w.uploadId,{file:_,rowObj:w}),w.status&&(w.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),w.bar&&(w.bar.style.width="100\
%"),w.row&&(w.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(_,w)}finally{
x&&uploadProgressState.perFilePct.hasOwnProperty(x)&&(delete uploadProgressState.perFilePct[x],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let m=0;const h=Math.min(UPLOAD_CONCURRENCY,o.length),y=Array.from({length:h}).map(async()=>{
for(;;){const v=m++;if(v>=o.length)break;await d(o[v])}});await Promise.all(y)}r(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const n=e.target.files&&e.target.files[0];n&&(await uploadMaskFile(n),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,n={}){const i=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(i).replace(/\n/g,
"<br>");const a=protectMathSegments(i),o=window.marked.parse(a.text),l=restoreMathSegments(o,a.blocks,
n);return window.DOMPurify.sanitize(l)}r(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");function getCanvasModeElements(){
const e=get("canvas-panel");return e?{panel:e,stage:get("conversation-stage"),title:get("canvas-pane\
l-title"),status:get("canvas-panel-status"),blockCount:get("canvas-block-count"),blockList:get("canv\
as-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get("canvas-preview-lang"),sourceSelect:get(
"canvas-source-select"),frame:get("canvas-preview-frame"),empty:get("canvas-preview-empty"),sourceScroll:get(
"canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get("canvas-panel-copy-btn"),clearBtn:get(
"canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:null}r(getCanvasModeElements,"getC\
anvasModeElements");function isCanvasHtmlPreviewCandidate(e,n){const i=String(e||"").trim().toLowerCase();
if(i==="html"||i==="htm"||i==="xhtml")return!0;if(i)return!1;const a=String(n||"");return/<!doctype\s+html/i.
test(a)||/<html[\s>]/i.test(a)}r(isCanvasHtmlPreviewCandidate,"isCanvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,n){
const i=String(e&&e.lang?e.lang:"").trim(),a=String(e&&e.code!==void 0&&e.code!==null?e.code:""),o=!!(e&&
e.open);return{...e,index:n,lang:i,code:a,open:o,key:hashString(`${i||"TEXT"}
${a||""}`)}}r(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const n=String(
e||""),i=n.split(/\r?\n/),a=[],o=[],l=/^(\s*)(`{3,}|~{3,})(.*)$/;let c=null,d="",m=[];for(const v of i){
if(!c){const _=v.match(l);if(_){c=_[2],d=String(_[3]||"").trim(),m=[],a.push({lang:d,code:"",open:!0}),
o.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}o.push(
v);continue}const x=String(v||"").trim();if(x&&x.replace(/\s+/g,"")===c){const _=a[a.length-1];_&&(_.
code=m.join(`
`),_.open=!1),c=null,d="",m=[];continue}m.push(v);const w=a[a.length-1];w&&(w.code=m.join(`
`))}if(c&&a.length){const v=a[a.length-1];v&&(v.code=m.join(`
`),v.open=!0)}const h=a.map((v,x)=>normalizeCanvasBlock(v,x)),y=selectCanvasPreviewBlock(h,n);return{
renderText:o.join(`
`),blocks:h,primaryBlock:y?y.block:null,primaryIndex:y?y.index:-1,rawText:n}}r(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,n="",i=-1){const a=Array.isArray(e)?e:[];if(Number.
isInteger(i)&&i>=0&&i<a.length){const l=a[i];return{block:l,index:i,previewType:isCanvasHtmlPreviewCandidate(
l.lang,l.code)?"html":"code"}}if(a.length>0){const l=a.length-1,c=a[l];return{block:c,index:l,previewType:isCanvasHtmlPreviewCandidate(
c.lang,c.code)?"html":"code"}}const o=String(n||"");return isCanvasHtmlPreviewCandidate("",o)?{block:normalizeCanvasBlock(
{lang:"html",code:o,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}r(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const a=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",a)?{block:normalizeCanvasBlock({lang:"html",code:a,open:!0,fallback:!0},0),index:-1}:null}const n=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,i=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,n);return!i||!i.block?null:i}r(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const n=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(i=>{const a=String(i.getAttribute("data-code-key")||""),o=!!n&&n===a;i.classList.toggle("can\
vas-active",o),i.setAttribute("aria-pressed",o?"true":"false"),i.setAttribute("data-canvas-active",o?
"1":"0"),i.innerHTML=o?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
i.title=o?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
i.setAttribute("aria-label",o?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
r(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}r(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,n,i){if(!e||!isCanvasMobileLayout()||n===i)return;const a={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
o={preview:0,blocks:1,source:2},l=a[i];if(!l||!(n in o)||!(i in o))return;canvasPreviewState.viewAnimationToken+=
1;const c=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(a).
forEach(m=>{m&&m.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),l.
offsetWidth;const d=o[i]<o[n]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";l.classList.
add(d),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{c===canvasPreviewState.viewAnimationToken&&
(l.classList.remove(d),canvasPreviewState.viewAnimationTimer=null)},340)}r(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,n={}){
var c,d;const i=getCanvasModeElements();if(!i||!i.panel)return;const a=["preview","blocks","source"].
includes(e)?e:"preview",o=["preview","blocks","source"].includes(n.fromView)?n.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=a,i.panel.dataset.canvasMobileView=a,(i.panelTabs?Array.from(
i.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(m=>{const h=m.getAttribute("da\
ta-canvas-panel-view")===a;m.classList.toggle("active",h),m.setAttribute("aria-pressed",h?"true":"fa\
lse")}),n.animate===!0&&animateCanvasMobileViewEntry(i,o,a),n.focus!==!1&&isCanvasMobileLayout()&&(a===
"preview"&&i.frame&&!i.frame.classList.contains("hidden")?i.frame.focus({preventScroll:!0}):a==="sou\
rce"&&i.sourceScroll?i.sourceScroll.focus({preventScroll:!0}):a==="blocks"&&i.blockList&&((d=(c=i.blockList).
focus)==null||d.call(c,{preventScroll:!0})))}r(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(n.length)),!n.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const i=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=n.map((a,o)=>{const l=String(a&&a.lang?a.lang:"text").trim()||"text",c=o===i,d=a&&
a.open?"\u751F\u6210\u4E2D":"\u8868\u793A",y=(String(a&&a.code?a.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),v=`${c?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${l}`,x=`${v}\u3001${y}`;return`<butto\
n type="button" class="canvas-block-chip${c?" active":""}" data-canvas-block-index="${o}" title="${escapeHtml(
v)}" aria-label="${escapeHtml(x)}" aria-pressed="${c?"true":"false"}"><span class="canvas-block-chip\
-index">#${o+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
l)}</span><span class="canvas-block-chip-preview">${escapeHtml(y)}</span></span><span class="canvas-\
block-chip-state">${c?"\u8868\u793A\u4E2D":d}</span></button>`}).join("")}r(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!n.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:n.length-1;e.sourceSelect.disabled=!1;const a=n.map((l,c)=>{const d=String(
l&&l.lang?l.lang:"text").trim()||"text";return`#${c+1} ${d}`}),o=JSON.stringify(a);e.sourceSelect.dataset.
canvasOptionsSignature!==o&&(e.sourceSelect.innerHTML=a.map((l,c)=>`<option value="${c}">${escapeHtml(
l)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=o),e.sourceSelect.value=String(
i)}r(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}r(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,n){
const i=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),a=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),o=String(e||""),l=`(function(){const token=${JSON.stringify(n)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${i},${a}\
);report()})})();`;try{const c=new DOMParser().parseFromString(o,"text/html"),d=c.createElement("scr\
ipt");return d.setAttribute("data-canvas-scroll-bridge","true"),d.textContent=l,(c.body||c.documentElement).
appendChild(d),`<!DOCTYPE html>
`+c.documentElement.outerHTML}catch{return`${o}<script data-canvas-scroll-bridge>${l}<\/script>`}}r(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const n=e&&e.data?e.data:null;if(!n||n.type!=="canvas-preview-scroll")return;const i=getCanvasModeElements();
!i||!i.frame||e.source!==i.frame.contentWindow||n.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(n.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(n.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const n=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{n===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}r(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const n=getCanvasModeElements();if(n){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){n.panel.classList.add("hidden"),n.panel.classList.remove(
"canvas-panel-open","canvas-closing"),n.stage&&n.stage.classList.remove("canvas-enabled");return}n.panel.
classList.remove("canvas-panel-open"),n.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{n.panel.classList.add("hidden"),n.panel.classList.remove("canvas-closing"),n.
stage&&n.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}r(hideCanvasPreviewPanel,
"hideCanvasPreviewPanel");function resetCanvasPreviewPanel(e="Canvas\u3067\u8868\u793A\u4E2D"){const n=getCanvasModeElements();
n&&(canvasPreviewState.blocks=[],canvasPreviewState.rawText="",canvasPreviewState.renderText="",canvasPreviewState.
selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.selectionMode="auto",canvasPreviewState.
mobileView="preview",canvasPreviewState.lastCanvasData=null,resetCanvasScrollState(),showCanvasPreviewPanel(),
syncCanvasPanelViewUi("preview",{focus:!1}),n.title&&(n.title.textContent=e),n.status&&(n.status.textContent=
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D"),n.previewLang&&(n.previewLang.
textContent="idle"),n.sourceSelect&&(n.sourceSelect.innerHTML='<option value="">-</option>',n.sourceSelect.
disabled=!0,n.sourceSelect.dataset.canvasOptionsSignature=""),n.code&&(n.code.textContent=""),n.blockCount&&
(n.blockCount.textContent="0"),n.blockList&&(n.blockList.innerHTML='<div class="px-2 py-3 text-xs te\
xt-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>'),n.sourceScroll&&
(n.sourceScroll.scrollTop=0),n.frame&&(n.frame.removeAttribute("srcdoc"),n.frame.classList.add("hidd\
en")),n.empty&&n.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}r(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const n=e||canvasPreviewState.lastCanvasData;
if(!n)return null;canvasPreviewState.lastCanvasData=n,canvasPreviewState.blocks=Array.isArray(n.blocks)?
n.blocks.slice():[],canvasPreviewState.rawText=String(n.rawText||""),canvasPreviewState.renderText=String(
n.renderText||"");const i=canvasPreviewState.blocks,a=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!i.length){const c=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return c&&c.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=c.block.
key||"",c.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",a!==-1&&resetCanvasScrollState(),null)}let o=i.length-1;canvasPreviewState.selectionMode===
"manual"&&a>=0&&a<i.length?o=a:canvasPreviewState.selectionMode="auto";const l=i[o]||null;return canvasPreviewState.
selectedIndex=l?o:-1,canvasPreviewState.selectedKey=l&&l.key?l.key:"",a!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),l}r(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const n=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],i=getCanvasSelectedBlock(),a=i&&i.block?i.block:null,o=i&&Number.isInteger(
i.index)?i.index:-1,l=!!a,c=String(a&&a.lang?a.lang:"").trim(),d=String(a&&a.code!==void 0&&a.code!==
null?a.code:""),m=l?isCanvasHtmlPreviewCandidate(c,d):!1,h=l?m?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
a&&a.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",y=l?m?`HTML Canvas Preview${n.length>
1&&o>=0?` #${o+1}/${n.length}`:""}`:`Canvas Preview: ${c||"text"}${n.length>1&&o>=0?` #${o+1}/${n.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=y),e.status&&(e.status.textContent=
h),e.previewLang&&(e.previewLang.textContent=l?c||"text":"idle");const v=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,x=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=d),e.sourceScroll&&(e.sourceScroll.scrollTop=v,e.sourceScroll.
scrollLeft=x,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(n.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),l){canvasPreviewState.frameRenderToken+=1;const w=canvasPreviewState.frameRenderToken,
_=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(a),w);e.frame&&(e.frame.srcdoc=_,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{w!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:w,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}r(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,n={}){const i=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!i.length)return!1;const a=Number(e);if(!Number.isInteger(a)||
a<0||a>=i.length)return!1;const o=canvasPreviewState.selectedIndex!==a;return canvasPreviewState.selectedIndex=
a,canvasPreviewState.selectedKey=i[a]&&i[a].key?i[a].key:"",canvasPreviewState.selectionMode="manual",
o&&resetCanvasScrollState(),syncCanvasPanelViewUi(n.view||"preview",{focus:!1,animate:n.animateView===
!0,fromView:n.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}r(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const n=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=String(
e||"");if(!i)return!1;const a=n.findIndex(o=>o&&o.key===i);return a===-1?!1:applyCanvasSelection(a)}
r(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const n=e.getAttribute("data-code")||"";if(!n)return null;let i="";try{i=decodeURIComponent(
n)}catch{i=n}const a=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
o=String(e.getAttribute("data-code-key")||hashString(`${a||"TEXT"}
${i||""}`));return{code:i,lang:a,codeKey:o}}r(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const n=decodeCanvasPreviewButtonCode(e);if(!n)return null;
const i=e&&typeof e.closest=="function"?e.closest(".message-group"):null,a=i?Array.from(i.querySelectorAll(
".canvas-preview-btn")):[];if(!a.length){const d=normalizeCanvasBlock({lang:n.lang,code:n.code,open:!1},
0);return{blocks:[d],selectedIndex:0,selectedKey:d.key||n.codeKey||""}}const o=[];let l=-1;if(a.forEach(
(d,m)=>{const h=decodeCanvasPreviewButtonCode(d);if(!h)return;const y=normalizeCanvasBlock({lang:h.lang,
code:h.code,open:!1},m);o.push(y),l===-1&&h.codeKey===n.codeKey&&(l=o.length-1)}),!o.length)return null;
l===-1&&(l=0);const c=o[l]||o[0]||null;return{blocks:o,selectedIndex:l,selectedKey:c&&c.key?c.key:n.
codeKey||""}}r(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
if(!e)return!1;const n=collectCanvasBlocksFromButton(e);if(!n||!n.blocks||!n.blocks.length)return!1;
const i=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[],a=i.findIndex(l=>l&&l.
key===n.selectedKey);if(a!==-1&&i.length>1)return applyCanvasSelection(a);const o=n.blocks[n.selectedIndex]||
n.blocks[0]||null;return canvasPreviewState.blocks=n.blocks,canvasPreviewState.rawText=o&&o.code!==void 0&&
o.code!==null?String(o.code):"",canvasPreviewState.renderText=canvasPreviewState.rawText,canvasPreviewState.
selectedIndex=Number.isInteger(n.selectedIndex)?n.selectedIndex:0,canvasPreviewState.selectedKey=n.selectedKey||
o&&o.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),canvasPreviewState.lastCanvasData=
{renderText:canvasPreviewState.renderText,blocks:n.blocks,primaryBlock:o,primaryIndex:canvasPreviewState.
selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.mobileView="preview",syncCanvasPanelViewUi(
"preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}r(previewCanvasCodeFromButton,"previewCanvasCod\
eFromButton");function buildCanvasPreviewDocument(e){const n=String(e&&e.code!==void 0&&e.code!==null?
e.code:""),i=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(i,n))return sanitizeHtmlForPreview(
n);const o=i?`Canvas Preview: ${i}`:"Canvas Preview",l=escapeHtml(n||"");return`<!doctype html><html\
 lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-sc\
ale=1"><title>${escapeHtml(o)}</title><style>
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
            </style></head><body><div class="frame"><div class="label">${escapeHtml(o)}</div><pre>${l||
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}r(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,n={}){const i=n.persist!==
!1;if(canvasModeEnabled=!!e,i)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const a=get("enable-canvas-mode");if(a&&a.checked!==canvasModeEnabled&&(a.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(n.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!n.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}r(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}r(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,n,i){let a=normalizeMarkdownNewlines(
e);const o=normalizeMarkdownNewlines(i);if(!o&&o!=="")return a;const l=n?[String(n),""]:[""];for(const c of[
"`","~"])for(let d=3;d<=10;d++){const m=c.repeat(d);for(const h of l){const y=`${m}${h}
`,v=`
${m}`,x=y+o+v;a.includes(x)&&(a=a.split(x).join(""))}}return a}r(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,n){let i=normalizeMarkdownNewlines(e);const a=normalizeMarkdownNewlines(
n==null?"":String(n)),o=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const l of o)for(const c of["`","~"])for(let d=3;d<=10;d++){const m=c.repeat(d);
[`${l}${m}
${a}
${m}`,`${l}
${m}
${a}
${m}`,`
${l}${m}
${a}
${m}`,`
${l}
${m}
${a}
${m}`].forEach(y=>{i.includes(y)&&(i=i.split(y).join(`
`))})}return i}r(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const n=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(n)}</div>`}r(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,n=""){
let i=String(e==null?"":e).trim()||"Unknown error";i=i.replace(/```/g,"'''"),i.length>5e4&&(i=i.slice(
0,5e4)+"\u2026");const a="```chat_error\n"+i+"\n```",o=String(n==null?"":n).replace(/\s+$/,"");return o?
o+`

`+a:a}r(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const n=normalizeMarkdownNewlines(e),i=[];if(!n)return{text:"",executions:i};const a=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let o=n.replace(a,(l,c,d)=>{const m=String(d||"").trim();try{const h=JSON.parse(m);i.push({code:h&&h.
code!=null?String(h.code):"",output:h&&h.output!=null?String(h.output):""})}catch{i.push({code:m,output:""})}
return`
`});return i.forEach(l=>{l.code&&(o=stripExactFencedBlock(o,"python",l.code),o=stripExactFencedBlock(
o,"py",l.code)),o=stripVisiblePythonOutputBlock(o,l.output)}),o=o.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:o,executions:i}}r(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const n=normalizeMarkdownNewlines(
e),i=[];if(!n)return{text:"",notes:i};const a=[];return n.split(`
`).forEach(l=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(l)?i.push(l.trim()):
a.push(l)}),{text:a.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:i}}r(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,n){const i=String(e||"").trim(),a=Array.
isArray(n)?n.filter(Boolean):[];return a.length?i?`${i}

${a.join(`
`)}`:a.join(`
`):i}r(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,n,i){
const a=e&&e.code!=null?String(e.code):"",o=e&&e.output!=null?String(e.output):"";let l="";try{window.
hljs&&typeof window.hljs.highlight=="function"?l=window.hljs.highlight(a,{language:"python"}).value:
l=escapeHtml(a)}catch{l=escapeHtml(a)}const c=escapeHtml(o),d=encodeURIComponent(a).replace(/'/g,"%2\
7"),m=encodeURIComponent(o).replace(/'/g,"%27"),h=hashString(`pyexec-detail
${a}
${o}
${n}`),y=i>1?`Python Execution ${n+1}/${i}`:"Python Execution",v=`<button class="download-btn" data-\
code="${d}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,x=`<button class="coding-target-btn" data-code="${d}" data-code-key="${h}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${h}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(y)}</span><div class="code-actions">${x}${v}<button class="copy-btn" data-copy="code" \
data-code="${d}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${m}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${l}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${c}</code></pre></div></div></div>`}r(buildPythonExecDetailBoxHtml,"buildP\
ythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/python-\
execution"){const n={modal:"python-execution"};e!==null&&(n.messageId=e),history.pushState(n,"","/py\
thon-execution")}showModal("python-exec-modal")}r(showPythonExecDetailModal,"showPythonExecDetailMod\
al");function openPythonExecDetail(e){const n=messageMeta[e],i=get("python-exec-modal"),a=get("pytho\
n-exec-modal-body"),o=get("python-exec-modal-title");if(!i||!a)return;const l=n&&Array.isArray(n.python_executions)?
n.python_executions:[];if(!l.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(o){const c=l.length>1?`\uFF08${l.length}\u4EF6\uFF09`:"";o.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${c}`}a.innerHTML=l.map((c,d)=>buildPythonExecDetailBoxHtml(c,d,l.length)).join(
""),codingModeEnabled&&(syncCodingTargetButtons(a),syncCodingModeUi(!0,{persist:!1})),showPythonExecDetailModal(
e)}r(openPythonExecDetail,"openPythonExecDetail"),window.openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){
get("python-exec-modal")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&
history.back())}r(closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;
function buildAiMarkdownHtml(e){const n=extractMcpExecutionNotesFromContent(e),i=appendMcpExecutionNotes(
n.text,n.notes),a=canvasModeEnabled?parseCanvasMarkdown(i):{renderText:i,blocks:[],primaryBlock:null,
rawText:i};canvasModeEnabled&&(updateCanvasPreviewState(a),refreshCanvasPreviewPanel());const o=document.
createElement("div");return o.className="prose prose-invert text-sm break-words",o.innerHTML=sanitizeMarkdownHtml(
a.renderText),wrapRenderedSvgBoxes(o),lowBandwidthMode||(maybeNeedsHighlight(a.renderText,o)&&ensureHighlightLoaded().
catch(()=>{}),maybeNeedsMathJax(a.renderText)&&ensureMathJaxLoaded().catch(()=>{})),o.outerHTML}r(buildAiMarkdownHtml,
"buildAiMarkdownHtml");function renderAiMarkdownInto(e,n,i={}){if(!e)return;const a=extractMcpExecutionNotesFromContent(
n),o=appendMcpExecutionNotes(a.text,a.notes),l=canvasModeEnabled?parseCanvasMarkdown(o):{renderText:o,
blocks:[],primaryBlock:null,rawText:o};if(canvasModeEnabled&&(updateCanvasPreviewState(l),refreshCanvasPreviewPanel()),
i.incrementalMath){const c=document.createElement("template");c.innerHTML=sanitizeMarkdownHtml(l.renderText,
{streamMathSegments:!0});const d=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(h=>{const y=h.getAttribute("data-stream-math-key");y&&d.set(y,h)});const m=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(h=>{const y=d.get(h.getAttribute(
"data-stream-math-key"));y?h.replaceWith(y):m.push(h)}),e.replaceChildren(c.content),wrapRenderedSvgBoxes(
e),queueHighlight(e,l.renderText),queueIncrementalMathTypeset(m);return}e.innerHTML=sanitizeMarkdownHtml(
l.renderText),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,l.renderText)}r(renderAiMarkdownInto,
"renderAiMarkdownInto");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(n=>{if(!n||!n.parentNode||n.closest(".svg-render-box")||n.closest(
"pre, code, .code-wrapper, .thought-container"))return;const i=document.createElement("span");i.className=
"svg-render-box",n.parentNode.insertBefore(i,n),i.appendChild(n)})}r(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");function renderMessage(e,n,i,a,o,l,c=null,d=!0,m=null,h=null,y=null,v=null,x=null,w=null,_=null,S=null,L=!0,M=null,P=null,H=null){
const Q=n==="user",ee=Q?"bg-blue-600":"bg-gray-700",Ae=Q?"justify-end":"justify-start";messageStore[e]=
i;const B=!Q&&i?extractPythonExecutionsFromContent(i):{text:i||"",executions:[]},V=Q?i:B.text;let te=h;
if(te==null){const ne=y!=null?Number(y):0,ue=v!=null?Number(v):0;(y!=null||v!=null)&&(te=ne+ue)}messageMeta[e]=
{tokens_in:y,tokens_out:v,tokens_total:te,tokens_content:w,tokens_thought:_,is_encrypted:x,role:n,model:l,
parent_id:M,quote_text:m,image_url:a,gem_name:P,batch_job:H,python_executions:Q?[]:B.executions||[]};
let ge="";m&&(ge=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-g\
ray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(m)}\
</div>`);let de="";if(o&&!Q){let ne="";try{ne=JSON.parse(o).text||""}catch{ne=o}ne&&(de=`<div class=\
"thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brai\
n text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(ne)}\
</div></div>`)}let Ce="";if(a)try{const ne=JSON.parse(a);if(ne.length){const ue=[];if(ne.forEach(J=>{
let ve=J,at="unknown";if(ve&&typeof ve=="object"&&(at=normalizeAttachmentSource(ve.source),ve=ve.filepath||
ve.path||ve.url||ve.file||""),ve=normalizeAttachmentPath(ve)||ve,!ve)return;setAttachmentSourceForPath(
ve,at);const Ee=ve.replace(/^\d+\//,""),lt=buildFileUrl(Ee),ht=buildAttachmentPreviewUrl(Ee),Qe=ve.split(
"/").pop(),ut=Qe.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(ut)?ue.push(
buildChatImageHtml(ht,{viewerSrc:lt,alt:Qe,title:Qe,filename:Qe})):ue.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${lt}')" title="${Qe}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Qe}</span></div>`)}),ue.length>
0){let J="grid-multi";ue.length===1?J="grid-1":ue.length===2?J="grid-2":ue.length===3?J="grid-3":ue.
length===4&&(J="grid-4"),Ce=`<div class="image-grid ${J}">${ue.join("")}</div>`}}}catch{}const be=Q?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,Le=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${Q?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${be}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
oe=[];!Q&&l&&oe.push(escapeHtml(l)),P&&(Q?oe.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml(P)}</span>`):oe.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml(P)}</span>`));const le=[];if(y!=null&&le.push(`In ${y}`),v!=null){let ne=`\
Out ${v}`;_!=null&&Number(_)>0&&(ne+=` (Thought ${_})`),le.push(ne)}if(le.length||h!=null){const ne=le.
length?le.join(" / "):`${h} tokens`;oe.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${ne}</button>`)}if(x!=null){const ne=x?"fa\
-lock":"fa-lock-open",ue=isAdminUser?x?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":x?"Encrypted":
"Plain",J=isAdminUser?x?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";oe.push(`<button class="${J}" title="${ue}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${ne}"></i></button>`)}if(!Q&&B.executions&&B.executions.length){
const ne=B.executions.length,ue=ne>1?`Python \xD7${ne}`:"Python";oe.push(`<button type="button" clas\
s="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\
\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${ue}</span></button>`)}const Y=oe.
length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${oe.
join(" \u2022 ")}</div>`:"";let E;const j=!Q&&H?(()=>{const ne=String(H.state||"").toUpperCase(),ue=H.
status_text||(ne==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ne==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ne==="JOB_STATE_FAILED"||
ne==="JOB_STATE_CANCELLED"||ne==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ne==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(ue)}</div></div>`})():
"";Q?E=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(i||
"")}</div>`:(E=j+(V&&String(V).trim()?buildAiMarkdownHtml(V):H?'<div class="content-area prose prose\
-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(V)),E.includes("content-area")||(E=E.replace("prose ","content-area prose ")));let W="";
if(c){const ne=c.siblings[c.current-2],ue=c.siblings[c.current];W=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ne}\
)" ${ne?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${c.current} / ${c.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ue}\
)" ${ue?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const Z=d?"fade-in":"",ie=document.createElement("div");ie.className=`flex ${Ae} m\
b-4 ${Z} relative message-group group`,ie.id=`msg-${e}`,ie.innerHTML=`<div class="message-bubble ${ee}\
 text-white p-4 rounded-2xl shadow-md relative">${Le}${ge}${de}${E}${Ce}${W}${Y}</div>`;const Se=S||
get("chat-container");return Se&&(Se.appendChild(ie),L&&scrollToBottom(),Q||(queueMessageDecorations(
ie,V),syncCodingTargetButtons(ie),syncCodingModeUi(codingModeEnabled,{persist:!1}))),ie}r(renderMessage,
"renderMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const n={
modal:"token-details"};e!==null&&(n.messageId=e),history.pushState(n,"","/token-details")}showModal(
"token-detail-modal")}r(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const n=messageMeta[e];
if(!n||!get("token-detail-modal"))return;const a=n.tokens_total!==null&&n.tokens_total!==void 0?n.tokens_total:
"-",o=n.tokens_in!==null&&n.tokens_in!==void 0?n.tokens_in:"-",l=n.tokens_out!==null&&n.tokens_out!==
void 0?n.tokens_out:"-",c=n.tokens_content!==null&&n.tokens_content!==void 0?n.tokens_content:"-",d=n.
tokens_thought!==null&&n.tokens_thought!==void 0?n.tokens_thought:"-",m=n.is_encrypted===null||n.is_encrypted===
void 0?"-":n.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=a,get("token-detai\
l-in").innerText=o,get("token-detail-out").innerText=l,get("token-detail-content").innerText=c,get("\
token-detail-thought").innerText=d,get("token-detail-encrypted").innerText=m;const h=n.model?`${n.model}\
 (${n.role})`:`${n.role}`;get("token-detail-title").innerText=h,showTokenDetailModal(e)}r(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){get("token-detail-modal")&&(hideModal("token-deta\
il-modal"),!e&&location.pathname==="/token-details"&&history.back())}r(closeTokenDetail,"closeTokenD\
etail");function openEncryptionSettings(e){const n=messageMeta[e];n&&openEncryptionModal(n.is_encrypted)}
r(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get("encrypti\
on-status-modal"))return;const i=get("encryption-status-title"),a=get("encryption-status-body"),o=get(
"encryption-status-admin-actions"),l=get("encryption-status-admin-toggle"),c=!!e;c?(i&&(i.innerText=
"\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),a&&(a.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\
\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(i&&(i.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),a&&(a.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
o&&l&&(!!(isAdminUser&&currentThreadId)?(o.classList.remove("hidden"),l.dataset.enable=c?"0":"1",l.disabled=
!1,l.textContent=c?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
l.className=c?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):o.classList.add("hidden")),showEncryptionStatusModal()}r(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}r(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const n=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${n?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const a=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,n,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=a}}r(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}r(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}r(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}r(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=r((e,n,i,a)=>{
const o=(e||"").toLowerCase();return!o.includes("gemini")||o.includes("image")||o.includes("nano")||
o.includes("tts")||o.includes("native-audio")?!1:!!a&&(n||i)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=r(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const n=get("gemini-local-python-dont-show"),i=get("gemini-local-python-continue"),a=get("g\
emini-local-python-cancel"),o=get("gemini-local-python-close");return n&&(n.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(l=>{let c=!1;function d(){i&&i.removeEventListener("cli\
ck",h),a&&a.removeEventListener("click",y),o&&o.removeEventListener("click",y),e.removeEventListener(
"click",v,!0)}r(d,"cleanup");function m(x){if(c)return;c=!0,n&&n.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),d(),hideModal("gemini-local-python-modal"),l(x)}r(m,"finalize");
function h(){m(!0)}r(h,"onOk");function y(){m(!1)}r(y,"onCancel");function v(x){x.target===e&&(x.preventDefault(),
x.stopImmediatePropagation(),y())}r(v,"onOverlay"),i&&i.addEventListener("click",h),a&&a.addEventListener(
"click",y),o&&o.addEventListener("click",y),e.addEventListener("click",v,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,n=!0,i=!0,a=null,o=null){const l=n?"fade-in":"",c=a?
` id="${a}"`:"",d=buildPendingSkeletonHtml(o,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),m=`<div clas\
s="flex justify-start mb-4 ${l}"><div${c} class="message-bubble ai-pending-bubble bg-gray-700 text-w\
hite p-4 rounded-2xl rounded-tl-none shadow-md relative">${d}</div></div>`,h=e||get("chat-container");
if(h){if(typeof h.insertAdjacentHTML=="function")h.insertAdjacentHTML("beforeend",m);else{const y=document.
createElement("div");y.innerHTML=m;const v=y.firstElementChild;v&&h.appendChild(v)}i&&scrollToBottom()}}
r(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){if(!e||e.getAttribute(
"data-stream-transition")==="1")return;const n=e.querySelector(".content-area");n&&(n.classList.remove(
"pending-shimmer","skeleton-pending"),n.removeAttribute("data-skeleton-kind")),e.setAttribute("data-\
stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add("ai-stream-transitio\
n"),n&&(n.classList.add("ai-stream-content-transition"),setTimeout(()=>{n&&n.classList.remove("ai-st\
ream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-transition")},320)}
r(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){return e==
null||e===""?null:String(e)}r(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}r(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const n=Array.from(e.querySelectorAll(".prose")).some(d=>String(d.textContent||"").trim()),i=!!e.querySelector(
".python-box"),a=Array.from(e.querySelectorAll(".thought-content")).some(d=>!!String(d.textContent||
"").trim()&&d.getAttribute("data-placeholder")!=="1");if(!n&&!i&&!a)return null;const o=e.parentElement;
if(!o)return null;const l=o.cloneNode(!0);l.setAttribute("data-local-stopped-partial","1"),l.classList.
remove("fade-in");const c=l.querySelector(".message-bubble");if(c&&(c.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),c.removeAttribute("data-stream-transition"),c.removeAttribute("id"),!l.
querySelector('[data-stopped-partial-note="1"]'))){const d=document.createElement("div");d.setAttribute(
"data-stopped-partial-note","1"),d.className="text-[10px] text-amber-200/90 mt-2 text-right",d.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",c.appendChild(d)}return{html:l.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}r(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,n=null){if(!e||
!e.html)return!1;const i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,a=n!=
null&&n!==""?String(n):e.threadId?String(e.threadId):null;if(a&&i&&a!==i)return!1;const o=get("chat-\
container");return o?(o.querySelectorAll('[data-local-stopped-partial="1"]').forEach(l=>l.remove()),
o.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}r(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const n=normalizeJobIdForUi(e);
n&&suppressedPendingJobIds.add(n)}r(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const n=normalizeJobIdForUi(e);return!!(n&&suppressedPendingJobIds.has(n))}r(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const n=manualStopContext.threadId?String(manualStopContext.threadId):null,i=e!=null&&e!==""?String(
e):null,a=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(n&&i&&n!==
i||n&&a&&n!==a)}r(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,n={}){
var d,m;const i=Math.max(0,Number((d=n.retries)!=null?d:1)||0),a=Math.max(0,Number((m=n.retryDelayMs)!=
null?m:180)||0),o=!!n.notifyOnFailure,l=e!=null&&e!==""?String(e):null,c=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!c||l&&c!==l)return!1;for(let h=0;h<=i;h++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==c?!1:(await loadMessages(c,{preserveDraft:!0,silent:!0}),
!0)}catch{h<i&&a>0&&await new Promise(v=>setTimeout(v,a))}return o&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===c&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}r(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(n){console.warn("Vi\
bration failed:",n)}}r(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const n=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(i=>i.kind==="minimal"&&!minimalPromptMode?!1:i.label.
toLowerCase().includes(n)||i.description.toLowerCase().includes(n))}r(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,n){if(String(e||"").toLowerCase()!=="thinking")
return e;const a=String(n||"").trimStart().match(/^\/thinking(\s+.*)$/i);return a?`thinking${a[1]}`.
toLowerCase():e}r(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const n=String(e||"").trim().toLowerCase();if(!n||n==="toggle"||n==="\u5207\u66FF"||n==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(n))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(n))return!1}r(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,n=""){const i=MINIMAL_SLASH_COMMANDS.find(l=>l.id===
e);if(!i||!minimalPromptMode)return!1;if(i.action==="options")return openMinimalOptions(),!0;const a=MINIMAL_POPUP_ITEMS.
find(l=>l.key===i.itemKey);if(!a||!minimalOptionVisible(a))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(a)&&a.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const o=String(i.presetArgument||n||"").trim();if(a.selectId){if(!o)return showToast(`\
\u4F7F\u3044\u65B9: ${i.label} ${i.id==="effort"?"none / low / medium / high / xhigh / max":"default\
 / none"}`,"info"),!1;const l=get(a.selectId),c=o.toLowerCase(),d=l?Array.from(l.options).find(m=>m.
value.toLowerCase()===c||m.textContent.trim().toLowerCase()===c):null;return!l||!d?(showToast(`${i.label}\
: \u6307\u5B9A\u5024\u300C${o}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(l.value=d.value,l.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${a.label}: ${d.textContent.trim()}`,"success"),!0)}if(a.special==="thinking"&&o){const l=o.toLowerCase(),
c={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},d=parseSlashToggleArgument(
o),m=get(a.checkboxId);if(Object.prototype.hasOwnProperty.call(c,l)){m&&!m.checked&&!m.disabled&&(m.
checked=!0,m.dispatchEvent(new Event("change",{bubbles:!0})));const h=get("thinking-level");return h&&
(h.value=c[l],h.dispatchEvent(new Event("change",{bubbles:!0}))),refreshMinimalOptionItems(),showToast(
`Thinking: ${l}`,"success"),!0}if(d===void 0)return showToast("\u4F7F\u3044\u65B9: /thinking on / off / min / low /\
 mid / high","info"),!1}if(a.checkboxId&&o){const l=parseSlashToggleArgument(o);if(l===void 0)return showToast(
`\u4F7F\u3044\u65B9: ${i.label} on / off`,"info"),!1;const c=get(a.checkboxId);if(l!==null&&c&&c.checked===
l)return showToast(`${a.label}: ${l?"ON":"OFF"}`,"info"),!0}return handleMinimalOptionClick(a),!0}r(
executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){const n=String(
e||"").trimStart();if(!n.startsWith("/"))return null;const i=n.substring(1).split(/\s+/)[0]||"",a=i.
match(/^[a-z][\w-]*/i);return a?a[0]:i}r(extractSlashCommandToken,"extractSlashCommandToken");function hideSlashCommandSuggestions(){
const e=get("slash-command-suggestions");e&&e.classList.add("hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=
0}r(hideSlashCommandSuggestions,"hideSlashCommandSuggestions");function showPendingSlashCommandIndicator(e){
const n=get("slash-command-indicator"),i=get("slash-command-name");if(!n||!i)return;const a=SLASH_COMMANDS.
find(l=>l.id===e);i.textContent=a?a.label:`/${e}`,n.classList.remove("hidden"),n.classList.add("flex");
const o=get("prompt-input");o&&a&&(o.dataset.originalPlaceholder=o.placeholder,o.placeholder=a.argumentHint||
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}
r(showPendingSlashCommandIndicator,"showPendingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){
const e=get("slash-command-indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const n=get(
"prompt-input");n&&n.dataset.originalPlaceholder&&(n.placeholder=n.dataset.originalPlaceholder,delete n.
dataset.originalPlaceholder);const i=pendingSlashCommand==="settings";pendingSlashCommand=null,i&&clearAiSettingsConversation()}
r(hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const n=get("slash-command-suggestions"),i=get("slash-command-list"),a=get("input-row");if(!n||!i||!a)
return;const o=visibleSlashCommands(e);if(o.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,o.length-1),i.innerHTML="",o.forEach((x,w)=>{const _=document.createElement(
"div");_.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===
slashSelectedIndex?"bg-gray-700":""}`,_.innerHTML=`
                    <i class="fas ${x.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${x.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${x.description}</div>
                    </div>
                `;let S=!1;_.addEventListener("pointerdown",L=>{typeof L.button=="number"&&L.button!==
0||(L.preventDefault(),S=!0,selectSlashCommand(x.id))}),_.addEventListener("click",L=>{L.preventDefault(),
S||selectSlashCommand(x.id)}),_.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
i.appendChild(_)});const l=a.getBoundingClientRect(),c=window.innerHeight,d=c-l.bottom,m=l.top,h=260,
y=8;if(n.style.position="fixed",n.style.left=`${Math.max(8,l.left)}px`,n.style.zIndex="80",n.style.maxHeight=
"none",d<180&&m>d){const x=Math.min(h,m-y);n.style.top="auto",n.style.bottom=`${c-l.top+4}px`,i.style.
maxHeight=`${x}px`}else{const x=Math.min(h,d-y);n.style.top=`${l.bottom+4}px`,n.style.bottom="auto",
i.style.maxHeight=`${x}px`}n.classList.remove("hidden"),slashSuggestionsVisible=!0}r(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const n=get("prompt-input");if(!n)return;
const i=n.value,a=extractSlashCommandToken(i);if(a!==null){const c=String(i||"").trimStart();n.value=
c.substring(1+a.length).trimStart()}else{const c=i.lastIndexOf("/");c!==-1?n.value=i.substring(0,c).
trimEnd():n.value=""}hideSlashCommandSuggestions();const o=SLASH_COMMANDS.find(c=>c.id===e),l=n.value.
trim();if(o&&o.autocompleteArgument&&!l){n.value=`${o.label} `,slashSelectedIndex=0,lastSlashFilter=
null,n.dispatchEvent(new Event("input",{bubbles:!0})),n.focus();return}if(o&&o.kind==="minimal"&&(!o.
requiresArgument||l)){n.value="",executeMinimalSlashCommand(e,l),n.dispatchEvent(new Event("input",{
bubbles:!0})),n.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),n.focus(),n.
dispatchEvent(new Event("input",{bubbles:!0}))}r(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
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
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}r(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,n){
const i=get(`tab-${e}`);let a=get(n);if(!i||!a)return null;for(;a.parentElement&&a.parentElement!==i;)
a=a.parentElement;return a.parentElement===i?a:get(n)}r(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const n=AI_SETTING_JUMP_TARGETS[e];if(!n){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(n.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const i=get(n.control);i&&(i.scrollIntoView({behavior:"smooth",block:"center"}),i.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const i=findSettingsJumpElement(n.tab,n.control);i?jumpToSetting(n.tab,i):switchTab(n.tab||"general")},
320)}r(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const n=e.querySelector(".msg-controls");n&&n.remove()}r(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,n,i="update"){const a=Object.
entries(e||{}),o=`settings-result-${Date.now()}`,l=i==="inspect",c=a.length?l?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
l?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
d=renderMessage(o,"assistant",c,null,null,n,null,!0,null,null,null,null,null,null,null,null,!0);if(!d)
return;removeEphemeralMessageControls(d);const m=d.querySelector(".message-bubble");if(!m||!a.length)
return;const h=document.createElement("div");h.className="mt-3 space-y-2 ai-settings-result-list",a.
forEach(([v,x])=>{const w=AI_SETTING_JUMP_TARGETS[v]||{label:v},_=document.createElement("button");_.
type="button",_.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const S=document.createElement("span");S.className="min-w-0 flex-1";const L=document.createElement(
"span");L.className="block text-xs font-bold text-blue-200",L.textContent=w.label;const M=document.createElement(
"span");M.className="block mt-0.5 text-[11px] text-gray-300 break-words",M.textContent=formatAiSettingValue(
x);const P=document.createElement("i");P.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",S.appendChild(L),S.appendChild(M),_.appendChild(S),_.appendChild(P),_.addEventListener(
"click",()=>openAiSettingJumpTarget(v)),h.appendChild(_)});const y=m.querySelector(".message-footer-\
meta");y?m.insertBefore(h,y):m.appendChild(h),scrollToBottom()}r(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,n){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const i=Date.now(),a=renderMessage(`settings-user-${i}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(a);const o=get(
"welcome-screen");o&&o.classList.add("hidden");const l=`settings-pending-${i}`,c=get("chat-container");
c&&(c.insertAdjacentHTML("beforeend",`<div id="${l}" class="flex justify-start mb-4 fade-in"><div cl\
ass="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-\
md relative">${buildPendingSkeletonHtml(n,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const m=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:n,conversation:aiSettingsConversation})})).
json().catch(()=>({})),h=get(l);if(h&&h.remove(),m&&m.status==="ok"&&m.mode==="inspect"&&m.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(m.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
m.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(m.current,n,"inspect");
return}if(m&&m.status==="ok"&&m.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
m.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(m.applied).length}\u9805\u76EE\uFF09`,"success");try{const x=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(x),cacheUserSettings(x)}catch{}renderAiSettingsResultBubble(
m.applied,n);return}const y=m.message||m.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${y}`);
const v=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${y}`,null,null,n,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
v),showToast(y,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const m=get(l);m&&m.remove();const h=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,n,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(h),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
r(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}r(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const n=get("gem-suggestions"),i=get("gem-su\
ggestions-list"),a=get("input-row");if(!n||!i||!a)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const o=e.toLowerCase(),l=loadedGems.filter(w=>w.name.toLowerCase().includes(o)||w.description&&
w.description.toLowerCase().includes(o));if(l.length===0){hideGemSuggestions();return}gemSelectedIndex>=
l.length&&(gemSelectedIndex=0),i.innerHTML="",l.forEach((w,_)=>{const S=document.createElement("div");
S.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${_===gemSelectedIndex?
"bg-gray-700":""}`,S.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(w.name)}</div>
                        ${w.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
w.description)}</div>`:""}
                    </div>
                `,S.onclick=()=>selectGemSuggestion(w),S.onmouseenter=()=>{gemSelectedIndex=_,showGemSuggestions(
e)},i.appendChild(S)});const c=a.getBoundingClientRect(),d=window.innerHeight,m=d-c.bottom,h=c.top,y=260,
v=8;if(n.style.position="fixed",n.style.left=`${Math.max(8,c.left)}px`,n.style.zIndex="80",n.style.maxHeight=
"none",m<180&&h>m){const w=Math.min(y,h-v);n.style.top="auto",n.style.bottom=`${d-c.top+4}px`,i.style.
maxHeight=`${w}px`}else{const w=Math.min(y,m-v);n.style.top=`${c.bottom+4}px`,n.style.bottom="auto",
i.style.maxHeight=`${w}px`}n.classList.remove("hidden"),gemSuggestionsVisible=!0}r(showGemSuggestions,
"showGemSuggestions");function selectGemSuggestion(e){const n=get("prompt-input");if(!n)return;const i=n.
value,a=i.lastIndexOf("@");a!==-1?n.value=i.substring(0,a).trimEnd():n.value="",hideGemSuggestions(),
activateGem(e),n.focus(),n.dispatchEvent(new Event("input",{bubbles:!0}))}r(selectGemSuggestion,"sel\
ectGemSuggestion");function browserFastModeIneligibility(e){const n=String(get("model-select")?get("\
model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!n.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(n))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeGem)return"Gems\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(currentQuote||editingMessageId)return"\u5F15\u7528\u30FB\u7DE8\u96C6\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(codingModeEnabled)return"Coding Mode\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(["enable-search","enable-url-context","enable-maps","enable-sys-prompt","enable-prompt-cache","en\
able-mcp"].some(c=>{const d=get(c);return!!(d&&d.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const a=get("thread-custom-instruction");if(a&&String(a.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const o=Array.from(browserFastLocalFiles.values());return o.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":o.reduce((c,d)=>c+Number(d.file&&d.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":o.some(c=>!c.
file||!String(c.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}r(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(n,i)=>{const a=new FileReader;a.onload=()=>{const o=String(a.result||""),l=o.indexOf(",");if(l<0)return i(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));n(
o.slice(l+1))},a.onerror=()=>i(a.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
a.readAsDataURL(e)})}r(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const n=[];let i=0;for(const a of Array.isArray(e)?e:[]){if(!a||!["user","model"].includes(a.role))continue;
const o=[];a.role==="model"&&Array.isArray(a.thought_signatures)&&a.thought_signatures.forEach(l=>{l&&
o.push({thoughtSignature:String(l)})}),a.text&&o.push({text:String(a.text)});for(const l of Array.isArray(
a.images)?a.images:[])try{const c=await fetch(buildFileUrl(l.path),{credentials:"same-origin",cache:"\
no-store"});if(!c.ok)throw new Error(`HTTP ${c.status}`);const d=await c.blob();o.push({inlineData:{
mimeType:l.mime_type||d.type||"application/octet-stream",data:await fileToBase64Payload(d)}})}catch{
i++}o.length&&n.push({role:a.role,parts:o})}return i&&showToast(`\u5C65\u6B74\u753B\u50CF${i}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),n}r(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(){
const e=Array.from(browserFastLocalFiles.entries());for(const[n,i]of e){if(!i||!i.file||!i.rowObj)throw new Error(
"\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");
if(i.rowObj.status&&(i.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(i.file,i.rowObj))throw new Error(`${i.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);browserFastLocalFiles.delete(n)}}r(uploadBrowserFastLocalFiles,
"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){const n=get("enable-thinking");
if(!n||!n.checked)return null;const i=String(get("thinking-level")?get("thinking-level").value:"high").
toLowerCase();if(e.includes("2.5")){const o=Number(get("thinking-budget")?get("thinking-budget").value:
4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(o)?Math.max(0,Math.min(32768,Math.trunc(
o))):4096}}let a=i.toUpperCase();return e.includes("3.6")&&!["MEDIUM","HIGH"].includes(a)&&(a="MEDIU\
M"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(a)&&(a="MINIMAL"),{includeThoughts:!0,thinkingLevel:a}}
r(browserFastThinkingConfig,"browserFastThinkingConfig");function browserFastPythonBoxHtml(e){return`\
<div class="code-wrapper python-box collapsed" data-py-id="${e}" data-collapsed="true" data-code-key\
="${e}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Exec\
ution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" a\
ria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code"\
 data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class\
="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-alig\
n-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="pyth\
on-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div clas\
s="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext p\
ython-output"></code></pre></div></div></div>`}r(browserFastPythonBoxHtml,"browserFastPythonBoxHtml");
function updateBrowserFastPythonBox(e,n,i){if(e){if(n==="code"){const a=i==null?"":String(i),o=e.querySelector(
".python-code");o&&(o.textContent=a,o.removeAttribute("data-highlighted"),queueHighlight(e,a));const l=e.
querySelector('.copy-btn[data-copy="code"]');l&&l.setAttribute("data-code",encodeURIComponent(a).replace(
/'/g,"%27"))}else if(n==="output"){const a=i==null?"":String(i),o=e.querySelector(".python-output");
o&&(o.textContent=a);const l=e.querySelector('.copy-btn[data-copy="output"]');l&&l.setAttribute("dat\
a-code",encodeURIComponent(a).replace(/'/g,"%27"))}}}r(updateBrowserFastPythonBox,"updateBrowserFast\
PythonBox");async function sendBrowserFastMessage(e){const n=String(get("model-select").value||"").trim(),
i=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error(
"\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const a=Array.from(browserFastLocalFiles.values()),o=[];for(const B of a)o.push({inlineData:{mimeType:B.
file.type,data:await fileToBase64Payload(B.file)}});o.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const d={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:o}],generationConfig:l};!!(get("enable-python")&&get("enable-python").
checked)&&(d.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden"),renderMessage(Date.now(),"user",
e,null,null,null,null,!0,null,null,null,null,null,null,null,null,!0);const h=`browser-fast-${Date.now()}`;
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${h}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${buildPendingSkeletonHtml(n,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const y=get(h);activeStreamingBubbleId=h,setSendBtnToStopMode(),resumeChatAutoScroll(),
abortController=new AbortController;let v="",x="";const w=[];let _=null,S=null,L=!1;const M={},P=[];
let H=null,Q="";const ee=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Ae=!1;try{const B=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(d),signal:abortController.
signal}));if(!B.ok){const le=await B.json().catch(()=>({}));throw new Error(le&&le.error&&le.error.message?
le.error.message:`Gemini API HTTP ${B.status}`)}window.ConnectionMonitor&&(Ae=!0,window.ConnectionMonitor.
operationStarted()),ee&&ee.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const V=B.body.getReader(),te=new TextDecoder;let ge="";const de=r(le=>{const Y=le.split(
/\r?\n/).filter(W=>W.startsWith("data:")).map(W=>W.slice(5).trim()).join("");if(!Y||Y==="[DONE]")return;
const E=JSON.parse(Y);if(E.error)throw new Error(E.error.message||"Gemini API error");if((Array.isArray(
E.candidates)?E.candidates:[]).forEach(W=>{(W&&W.content&&Array.isArray(W.content.parts)?W.content.parts:
[]).forEach(ie=>{if(ie&&typeof ie.thoughtSignature=="string"&&!w.includes(ie.thoughtSignature)&&w.push(
ie.thoughtSignature),ie&&ie.executableCode&&typeof ie.executableCode.code=="string"){const ne=ie.executableCode.
code;v+=`
\`\`\`python
${ne}
\`\`\`
`,H=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,Q=ne,M[H]||(y.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(H)),M[H]=y.querySelector(`[data-py-id="${H}"]`)),updateBrowserFastPythonBox(
M[H],"code",ne);return}if(ie&&ie.codeExecutionResult&&typeof ie.codeExecutionResult.output=="string"){
const ne=ie.codeExecutionResult.output;v+=`
**Output:**
\`\`\`
${ne}
\`\`\`
`;const ue=H||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;P.push({code:Q||
"",output:ne}),M[ue]||(y.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(ue)),M[ue]=y.querySelector(
`[data-py-id="${ue}"]`)),updateBrowserFastPythonBox(M[ue],"output",ne);return}const Se=typeof ie.text==
"string"?ie.text:"";Se&&(ie.thought===!0?x+=Se:v+=Se)})}),!L&&(v||x)){beginPendingToStreamTransition(
y);const W=y.querySelector(".content-area");W&&W.remove(),L=!0}x&&(S||(y.insertAdjacentHTML("afterbe\
gin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></\
div>'),S=y.querySelector(".thought-content")),S.textContent=x),v&&(_||(_=document.createElement("div"),
_.className="content-area prose prose-invert text-sm break-words",y.appendChild(_)),renderAiMarkdownInto(
_,v,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:le,value:Y}=await V.
read();if(le)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),ee&&ee.setPhase(
"receiving"),ge+=te.decode(Y,{stream:!0});const E=ge.split(/\r?\n\r?\n/);ge=E.pop()||"",E.forEach(de)}
if(ge+=te.decode(),ge.trim()&&de(ge),!v.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
_&&renderAiMarkdownInto(_,v,{incrementalMath:!0}),S&&S.classList.add("collapsed"),P.length&&(v+=P.map(
le=>`
\`\`\`pyexec
${JSON.stringify(le)}
\`\`\`
`).join("")),a.length&&(ee&&ee.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),ee&&ee.setPhase("saving");const Ce=collectImageUrlsForSend(),
be=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:v,thought_content:x,model:n,image_urls:Ce,temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:w,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}),y),Le=await be.json().catch(()=>({}));if(!be.ok||!Le.thread_id)throw new Error(
Le.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");const oe=!currentThreadId;currentThreadId=
String(Le.thread_id),currentParentId=Le.assistant_message_id||null,currentLeafId=Le.assistant_message_id||
null,resetUploadState(),browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,skipHistory:!oe}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\
\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F","success",!1)}catch(B){if(B.name!=="AbortError"){showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9: ${B.message}`,"error",!0),get("prompt-input").value||(get("prompt-i\
nput").value=e);const V=B.message||"\u30A8\u30E9\u30FC";y&&y.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
V));try{let te=v||"";P.length&&(te+=P.map(Le=>`
\`\`\`pyexec
${JSON.stringify(Le)}
\`\`\`
`).join(""));const ge=buildChatErrorMarkdown(V,te),de=a.length?[]:collectImageUrlsForSend(),Ce=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:ge,
thought_content:x||"",model:n,image_urls:de,temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||
null,parent_id:i&&i.parent_id?i.parent_id:null,thought_signatures:w,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),y),be=await Ce.
json().catch(()=>({}));if(Ce.ok&&be.thread_id){const Le=!currentThreadId;currentThreadId=String(be.thread_id),
currentParentId=be.assistant_message_id||null,currentLeafId=be.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!Le}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(te){sendClientDebugLog("error",`Browser fa\
st error persist failed: ${te&&te.message?te.message:te}`)}}}finally{Ae&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),ee&&ee(),setSendBtnToSendMode(),activeStreamingBubbleId===h&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}r(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Xt;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const F=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(F);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const n=get("prompt-inp\
ut").value;if(pendingSlashCommand){const F=pendingSlashCommand,ce=n.trim(),Me=get("model-select")?get(
"model-select").value:null;if(F==="settings"){if(!ce){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!Me){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
ce,Me)}else executeMinimalSlashCommand(F,ce)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const i=n.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(i&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(F=>F.id===
i[1].toLowerCase())){executeMinimalSlashCommand(i[1].toLowerCase(),i[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const a=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(a&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(a)setBrowserFastModeEnabled(!1);else{const F=browserFastModeIneligibility(
n);if(!F){try{await sendBrowserFastMessage(n)}catch(ce){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${ce.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${F}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(ce){showToast(
ce.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}n.trim()&&(promptHistory.length===
0||promptHistory[0]!==n)&&(promptHistory.unshift(n),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const o=collectAttachmentItemsForSend(),l=o.map(F=>F.path),c=o.filter(F=>normalizeAttachmentSource(
F.source)==="upload").map(F=>F.path);if(l.length>ATTACHMENT_MAX_FILES){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u6DFB\u4ED8\u3092\u6E1B\u3089\u3057\u3066\u518D\u9001\u3057\u3066\u304F\u3060\u3055\u3044\u3002`,
"error",!0);return}const d=getModelMediaSupport(get("model-select").value),m=l.some(F=>isAudioPath(F)),
h=l.some(F=>isVideoPath(F)),y=(get("model-select").value||"").toLowerCase(),v=get("enable-python"),x=!!(v&&
v.checked);if(m&&!d.audio||h&&!d.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!n.trim()&&l.length===0)return;if(isMistralOcrModel(
y)){const F=/https?:\/\/\S+/i.test(n);if(l.filter(Me=>isAudioPath(Me)||isVideoPath(Me)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!l.length&&!F){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const w=n.trim();if(/^\/settings(?:\s|$)/i.
test(w)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(w)){const F=w.replace(/^\/settings\s*/i,"").trim();
if(!F){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Me=get("prompt-input");Me.value="/settings ";const Ie=extractSlashCommandToken(Me.value);
lastSlashFilter=Ie,showSlashCommandSuggestions(Ie),Me.focus();return}const ce=get("model-select")?get(
"model-select").value:null;if(!ce){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
F,ce);return}if(isGeminiLocalPythonMode(y,m,h,x)&&!await confirmGeminiLocalPythonSwitch())return;let _=null,
S=[];if(codingModeEnabled){const F=collectCodingCandidates(n),ce=F.filter(ze=>ze.prompt_source),Me=F.
filter(ze=>!ze.prompt_source),Ie=ce.reduce((ze,Fe)=>ze+String(Fe.code||"").length,0);if(Ie>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let Ue=3e5-Ie;const Ve=[];for(let ze=Me.length-1;ze>=0;ze--){const Fe=String(Me[ze].
code||"").length;Fe>Ue||(Ve.unshift(Me[ze]),Ue-=Fe)}S=codingTargetSelection?Ve.slice(-1):[...ce,...Ve];
const ot=ce.length?ce[ce.length-1]:null;if(_=codingTargetSelection?S[0]:ot||S[S.length-1]||null,codingModeEffective=
!!(_&&String(_.code||"").trim()),codingModeEffective&&_.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const ze=String(((Xt=get(
"model-select"))==null?void 0:Xt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(ze)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const L=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${n.length}\
 attachments=${l.length} search=${get("enable-search").checked}`);const M=n,P=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&l.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const H=editingMessageId,Q=currentParentId,ee=H!=null;H&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Ae=[],B=r(F=>{if(F==null)
return;let ce=document.getElementById(`msg-${F}`);for(;ce;)ce.classList&&ce.classList.contains("mess\
age-group")&&(Ae.push({node:ce,prevDisplay:ce.style.display}),ce.style.display="none"),ce=ce.nextElementSibling},
"hideRenderedBranchFrom"),V=r(()=>{Ae.forEach(({node:F,prevDisplay:ce})=>{F&&(F.style.display=ce||"")}),
Ae.length=0},"restoreHiddenBranch");H&&B(H);const te=Date.now(),ge=renderMessage(te,"user",M,JSON.stringify(
l),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,Q,activeGem?activeGem.name:null);
let de=!1;const Ce=/(https?:\/\/)?(x\.com|twitter\.com)\//i,be=Ce.test(M||"")||Ce.test(currentQuote||
""),Le="grok-4-fast-reasoning",oe=r(()=>{get("enable-search").checked=!0,get("model-select").value!==
Le&&selectModelById(Le)},"applyXLinkAuto");if(be&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)oe();else{const F=get("auto-search-banner"),ce=get("auto-search-on-btn"),Me=get(
"auto-search-off-btn"),Ie=get("auto-search-remember");F&&ce&&Me&&(Ie&&(Ie.checked=!1),await new Promise(
Ue=>{F.classList.remove("hidden");const Ve=r(ot=>{F.classList.add("hidden"),ce.onclick=null,Me.onclick=
null,Ue(ot)},"cleanup");ce.onclick=()=>Ve("enable"),Me.onclick=()=>Ve("disable")}).then(async Ue=>{Ue===
"enable"?(oe(),Ie&&Ie.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
de=!0}))}const le=String(get("reasoning-effort").value||"").toLowerCase(),Y=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&le==="none",E={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:M,model:get("model-select").value,image_urls:l,image_items:o,uploaded_image_urls:c,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:Y?!1:get("enable-thin\
king").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:P,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
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
parent_id:Q,parent_id_explicit:ee,disable_auto_search:de,image_vision_model:currentVisionModel||null,
coding_mode:L,coding_target:L?{id:_.candidate_id,code:_.prompt_source?null:_.code,language:_.language||
"text",key:_.key||null,message_id:_.message_id||null,source:_.prompt_source?"prompt":"history",explicit:_.
explicit===!0}:null,coding_candidates:L?S.map(F=>({id:F.candidate_id,source:F.prompt_source?"prompt":
"history",prompt_index:F.prompt_source?F.prompt_index:null,code:F.prompt_source?null:F.code,language:F.
language||"text",explicit:F.explicit===!0})):[],batch_mode:a};e&&(E.turnstile_token=e);const j=get("\
thread-custom-instruction");j&&(E.thread_custom_instruction=j.value||""),activeGem?(E.system_prompt=
activeGem.instruction,E.enable_system_prompt=!0,E.gem_uuid=activeGem.uuid):E.gem_uuid=null,setSendBtnToStopMode();
const W="ai-"+Date.now(),Z=String(E.model||"").toLowerCase(),ie=!!E.enable_thinking||!!le&&le!=="non\
e",Se=Z.includes("gemini")||Z.includes("o1")||Z.includes("o3")||Z.includes("gpt-5")||Z.includes("rea\
soning")&&!Z.includes("non-reasoning"),ne=ie&&Se;let ue=buildPendingSkeletonHtml(E.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${W}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${ue}</div></div>`),resumeChatAutoScroll();const J=get(W);activeStreamingBubbleId=
W,canvasModeEnabled&&resetCanvasPreviewPanel();let ve=null;const at=r(F=>!ne||!J?null:((!ve||!J.contains(
ve))&&(ve=J.querySelector(".thought-content")),ve||(J.insertAdjacentHTML("afterbegin",'<div class="t\
hought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i cla\
ss="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" \
data-placeholder="1"></div></div>'),ve=J.querySelector(".thought-content")),ve&&(ve.setAttribute("da\
ta-placeholder","1"),ve.textContent=F||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
ve),"ensureThoughtPlaceholder");ne&&at("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const Ee=currentThreadId,lt=nowPerfMs(),ht=Date.now();let Qe=!1,
ut=!1,St=!1,wt=null,Tt=null,pt=null,Rt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Bt=r((F,ce)=>{if(!ce||F==="status"&&Qe||F==="thought"&&ut||F==="content"&&St)return;const Me=Math.
max(0,nowPerfMs()-lt);F==="status"?wt=Me:F==="thought"?Tt=Me:F==="content"&&(pt=Me),reportFirstTokenLatency(
{latency_seconds:Me/1e3,latency_ms:Me,thread_id:Rt||currentThreadId,job_id:currentJobId,model:E.model,
first_event_type:F,client_sent_at_ms:ht}),F==="status"?Qe=!0:F==="thought"?ut=!0:F==="content"&&(St=
!0)},"maybeReportFirstEventLatency"),ct=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let gt=!1,Ct=!1,Lt=null,bt=null,Kt=!1;try{E.thread_id&&activeGem&&(threadGemMap[E.thread_id]=
activeGem,pendingGemForNewThread=null);const F=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(E),signal:abortController.signal}),J);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${F.status}`),!F.ok){const $e=await F.json().catch(()=>({})),He=new Error($e.error||
`HTTP ${F.status}`);throw He.serverCode=$e.code||null,He.serverModel=$e.model||E.model,He.acceptedJobId=
$e.job_id||null,He.acceptedThreadId=$e.thread_id||null,He}gt=!0,window.ConnectionMonitor&&(Kt=!0,window.
ConnectionMonitor.operationStarted()),ct&&ct.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const ce=r(()=>{if(!J)return;const $e=J.querySelector(
".content-area");if($e&&$e.getAttribute("data-api-accepted")!=="1"&&($e.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(J,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
$e.outerHTML=buildPendingSkeletonHtml(E.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const He=J.querySelector(".content-area");He&&He.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
J,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");ce();
const Me=F.body.getReader(),Ie=new TextDecoder;let Ue="",Ve="",ot="",ze=!0,Fe=null,De=null,We=null,Yt=!1;
const Ft={};let st=0,jt=!1;for(;!jt;){const{done:$e,value:He}=await Me.read();if($e)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ct&&ct.setPhase("receiving"),Ue+=Ie.decode(He,{stream:!0});
let tt=Ue.split(`
`);Ue=tt.pop();let Qt=!1,dn=!1;for(let mt of tt)if(mt.trim())try{const pe=JSON.parse(mt);if(pe.type===
"thread_id"){ce();const _e=pe.content!==null&&pe.content!==void 0?String(pe.content):pe.content;_e&&
(Rt=_e,currentThreadId!==_e&&(currentThreadId=_e,history.pushState({},"","/c/"+_e)),activeGem&&(threadGemMap[_e]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(pe.type==="job_\
id"){ce(),currentJobId=pe.content,a&&showToast("Batch\u767B\u9332","info");continue}if(pe.type==="se\
arch_status"){pe.content==="searching"&&!We?(J.insertAdjacentHTML("afterbegin",'<div class="search-b\
ox visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>'),We=J.querySelector(
".search-box")):pe.content==="done"&&We&&(We.classList.remove("animate-pulse"),We.innerHTML='<i clas\
s="fas fa-check-circle text-green-400"></i> Search complete',setTimeout(()=>{We&&We.remove(),We=null},
2e3));continue}if(pe.type==="mcp"){handleMcpStreamEvent(J,pe.content||{});continue}if(pe.type==="mcp\
_decision_request"){openMcpDecisionModal(pe.content||{});continue}if(pe.type==="status"){ce();const _e=pe.
content===null||pe.content===void 0?"":String(pe.content);if(Bt("status",!!_e),ze&&J){const Je=_e||"\
\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(J,Je,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Ke=J.querySelector(".content-area");Ke&&(Ke.outerHTML=buildPendingSkeletonHtml(E.model,Je),updatePendingSkeletonStatus(
J,Je,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
ne&&at(_e||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(ze){beginPendingToStreamTransition(
J);const _e=J.querySelector(".content-area");_e&&(_e.innerHTML=""),ze=!1}if(pe.type==="coding_diff")
appendCodingLiveDiff(J,pe.content||{}),Bt("content",!0);else if(pe.type==="thought"){if(Fe||(Fe=J.querySelector(
".thought-content")),ot+=pe.content,Bt("thought",!!pe.content),!Fe){const _e='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';We?We.insertAdjacentHTML(
"afterend",_e):J.insertAdjacentHTML("afterbegin",_e),Fe=J.querySelector(".thought-content")}if(Fe&&Fe.
getAttribute("data-placeholder")==="1"){if(Fe.textContent="",Fe.removeAttribute("data-placeholder"),
Fe){const _e=Fe.parentElement.querySelector(".thought-header");_e&&_e.classList.remove("thinking-shi\
mmer")}ot=pe.content}Fe.classList.remove("collapsed"),dn=!0}else if(pe.type==="image_analysis"){const _e=pe.
content===null||pe.content===void 0?"":String(pe.content);if(!J)continue;let Je=J.querySelector(".im\
age-analysis-box");if(!Je){const nt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';We?We.insertAdjacentHTML("afterend",nt):J.insertAdjacentHTML("afterbegin",nt),Je=J.querySelector(
".image-analysis-box")}const Ke=Je.querySelector(".image-analysis-text");Ke&&(Ke.textContent=_e)}else if(pe.
type==="python"){const _e=pe.content||{},Je=_e.id||`py_${Date.now()}`;if(!Ft[Je]){const nt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Je}" data-collapsed="true" data-code-key="${Je}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;We?We.insertAdjacentHTML("afterend",nt):J.insertAdjacentHTML(
"afterbegin",nt),Ft[Je]=J.querySelector(`[data-py-id="${Je}"]`)}const Ke=Ft[Je];if(Ke){if(_e.code!==
void 0){const nt=_e.code==null?"":String(_e.code),ft=Ke.querySelector(".python-code");ft&&(ft.textContent=
nt,ft.removeAttribute("data-highlighted"),queueHighlight(Ke,nt));const Mt=Ke.querySelector('.copy-bt\
n[data-copy="code"]');Mt&&Mt.setAttribute("data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}if(_e.
output!==void 0){const nt=_e.output==null?"":String(_e.output),ft=Ke.querySelector(".python-output");
ft&&(ft.textContent=nt);const Mt=Ke.querySelector('.copy-btn[data-copy="output"]');Mt&&Mt.setAttribute(
"data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}}}else if(pe.type==="content"){const _e=pe.content===
null||pe.content===void 0?"":String(pe.content);Ve+=_e,/[`~]/.test(_e)&&activateDeferredCodingModeFromStream(
Ve),De||(De=J.querySelector(".content-area")||document.createElement("div"),De.className="prose pros\
e-invert text-sm break-words",J.contains(De)||J.appendChild(De)),Qt=!0,Bt("content",!!_e)}else if(pe.
type==="error"){Yt=!0,jt=!0,J.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(pe.content)),showToast(
pe.content||"Unknown error","error",!0);break}}catch{}if(dn&&Fe&&(Fe.textContent=ot,userAutoScroll&&
(Fe.scrollTop=Fe.scrollHeight)),Qt&&De){const mt=Date.now();if(mt-st>100){const pe=snapshotCodeCollapse(
De);renderAiMarkdownInto(De,Ve,{incrementalMath:!0}),applyCodeCollapse(De,pe,!0),st=mt}}scrollToBottom()}
if(ct&&ct(),De){const $e=snapshotCodeCollapse(De);renderAiMarkdownInto(De,Ve,{incrementalMath:!0}),applyCodeCollapse(
De,$e,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),J)if(queueHighlight(J,Ve),enableLatencyMetrics){
const $e=nowPerfMs()-lt;reportFirstTokenLatency({is_total:!0,latency_seconds:$e/1e3,latency_ms:$e,thread_id:Rt||
currentThreadId,job_id:currentJobId,model:E.model,client_sent_at_ms:ht,client_done_at_ms:Date.now()});
let He='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',tt=null;wt!==null&&(tt=wt),Tt!==null&&(tt===null||Tt<tt)&&(tt=
Tt),pt!==null&&(tt===null||pt<tt)&&(tt=pt),tt!==null&&(He+=`<div>Initial: ${(tt/1e3).toFixed(2)}s</d\
iv>`),pt!==null&&pt!==tt&&(He+=`<div>Content: ${(pt/1e3).toFixed(2)}s</div>`),He+=`<div class="font-\
bold text-gray-300">Total: ${($e/1e3).toFixed(2)}s</div>`,currentJobId&&(He+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),He+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,He+="</div>",J.insertAdjacentHTML("beforeend",He)}else J.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),J&&J.querySelectorAll(".thought-co\
ntent").forEach(He=>He.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0}),!Yt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(!0,{persist:!1})),userAutoScroll&&
scrollToBottom(),document.querySelectorAll(".message-group").length<=2||!currentThreadTitle||currentThreadTitle===
"New Chat"||currentThreadTitle==="No Title"?apiFetch("/api/generate_title",{method:"POST",headers:{"\
Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,model_id:get("model\
-select").value})}).then($e=>$e.json()).then($e=>{$e.title&&(document.title=$e.title+" - AI Chat",setCurrentChatHeaderTitle(
$e.title),loadThreads())}):loadThreads(!1)}catch(F){let ce=!1;const Me=F.name==="AbortError"&&isManualStopAbortForThread(
Ee);if(F.name==="AbortError"&&!Me&&(ce=await syncThreadAfterAbortedStream(Ee,{retries:2,retryDelayMs:180,
notifyOnFailure:!0})),sendClientDebugLog("error",`Prompt send error: ${F.message}`),!gt){ge&&ge.remove();
const Ie=J&&J.closest(".fade-in");Ie&&Ie.remove(),delete messageStore[te],delete messageMeta[te]}if(F.
serverCode==="request_already_accepted"&&F.acceptedJobId&&F.acceptedThreadId)gt=!0,Lt={job_id:F.acceptedJobId,
thread_id:String(F.acceptedThreadId),model:E.model},get("prompt-input").value="",get("prompt-input").
style.height="auto",resetUploadState(),clearQuote();else if(gt&&!Me)bt={job_id:normalizeJobIdForUi(currentJobId),
thread_id:currentThreadId!=null?String(currentThreadId):null,model:E.model},window.ConnectionMonitor.
setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(F.serverCode==="turnstile_required"){const Ie=await getTurnstileToken();Ie?(await verifyTurnstileOnServer(
Ie,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(F.serverCode==="api_key_missing"){const Ie=F.serverModel||E.model,Ue=await showApiKeyRequiredModalAsync(
Ie);Ue==="set"?Ct=!0:Ue==="switch"?showModal("model-modal"):showToast(F.message||`${getModelNameById(
Ie)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(F.
name!=="AbortError"){const Ie="Connection Error: "+F.message;showToast(Ie,"error",!0)}H&&!ce&&V()}finally{
Kt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),ct&&ct(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===W&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(Lt){const F=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=Lt.thread_id,(F!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(Lt,currentThreadId)}
if(bt&&bt.thread_id)return reconnectPendingStreamUntilAvailable(bt,bt.thread_id);if(Ct)return sendMessage()}
r(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const n=e.job_id,i=`pending-${n}`,a=e&&e.model?
String(e.model):"";get(i)||renderPendingMessage(get("chat-container"),!0,!0,i,a);const o=get(i);if(!o)
return;if(activeStreamingBubbleId=i,o.classList.add("ai-pending-bubble"),!o.querySelector(".content-\
area.skeleton-pending")){const V=o.querySelector(".content-area");V?V.outerHTML=buildPendingSkeletonHtml(
a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):o.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=n,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const l=currentThreadId,
c=a.toLowerCase(),d=c.includes("gemini")||c.includes("o1")||c.includes("o3")||c.includes("gpt-5")||c.
includes("reasoning")&&!c.includes("non-reasoning");let m=null;const h=r(V=>!d||!o?null:((!m||!o.contains(
m))&&(m=o.querySelector(".thought-content")),m||(o.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),m=o.querySelector(".thought-content")),m&&(m.setAttribute("data-pla\
ceholder","1"),m.textContent=V||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),m),
"ensureThoughtPlaceholder");d&&h("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",v="",x="",w=!0,_=null,S=null,L=null,M=!1;const P={};let H=0,Q=!1;const ee=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Ae=!1,B=!1;try{const V=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:n,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!V.ok)throw new Error(`Resume failed (${V.status})`);window.ConnectionMonitor&&
(B=!0,window.ConnectionMonitor.operationStarted()),ee&&ee.setPhase("waiting");const te=V.body.getReader(),
ge=new TextDecoder;for(;!Q;){const{done:de,value:Ce}=await te.read();if(de)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ee&&ee.setPhase("receiving"),y+=ge.decode(Ce,{stream:!0});
let be=y.split(`
`);y=be.pop();let Le=!1,oe=!1;for(let le of be)if(le.trim())try{const Y=JSON.parse(le);if(Y.type==="\
job_id"){currentJobId=Y.content||n;continue}if(Y.type==="search_status"){Y.content==="searching"&&!L?
(o.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),L=o.querySelector(".search-box")):Y.content==="done"&&L&&(L.
classList.remove("animate-pulse"),L.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{L&&L.remove(),L=null},2e3));continue}if(Y.type==="mcp"){handleMcpStreamEvent(
o,Y.content||{});continue}if(Y.type==="mcp_decision_request"){openMcpDecisionModal(Y.content||{});continue}
if(Y.type==="status"){const E=Y.content===null||Y.content===void 0?"":String(Y.content);if(w&&o){const j=E||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(o,j,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const W=o.querySelector(".content-area");W&&(W.outerHTML=buildPendingSkeletonHtml(a,j),updatePendingSkeletonStatus(
o,j,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
d&&h(E||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(w){beginPendingToStreamTransition(
o);const E=o.querySelector(".content-area");E&&(E.innerHTML=""),w=!1}if(Y.type==="coding_diff")appendCodingLiveDiff(
o,Y.content||{});else if(Y.type==="thought"){if(_||(_=o.querySelector(".thought-content")),x+=Y.content,
!_){const E='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';L?L.insertAdjacentHTML("afterend",E):o.insertAdjacentHTML("afterbegin",E),_=o.querySelector(
".thought-content")}if(_&&_.getAttribute("data-placeholder")==="1"){if(_.textContent="",_.removeAttribute(
"data-placeholder"),_){const E=_.parentElement.querySelector(".thought-header");E&&E.classList.remove(
"thinking-shimmer")}x=Y.content}_.classList.remove("collapsed"),oe=!0}else if(Y.type==="image_analys\
is"){const E=Y.content===null||Y.content===void 0?"":String(Y.content);if(!o)continue;let j=o.querySelector(
".image-analysis-box");if(!j){const Z='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';L?L.insertAdjacentHTML("afterend",Z):o.insertAdjacentHTML("afterbegin",Z),j=o.querySelector(
".image-analysis-box")}const W=j.querySelector(".image-analysis-text");W&&(W.textContent=E)}else if(Y.
type==="python"){const E=Y.content||{},j=E.id||`py_${Date.now()}`;if(!P[j]){const Z=`<div class="cod\
e-wrapper python-box collapsed" data-py-id="${j}" data-collapsed="true" data-code-key="${j}"><div cl\
ass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><di\
v class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B">\
<i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" t\
itle="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" dat\
a-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></b\
utton></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code<\
/div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-secti\
on"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"><\
/code></pre></div></div></div>`;L?L.insertAdjacentHTML("afterend",Z):o.insertAdjacentHTML("afterbegi\
n",Z),P[j]=o.querySelector(`[data-py-id="${j}"]`)}const W=P[j];if(W){if(E.code!==void 0){const Z=E.code==
null?"":String(E.code),ie=W.querySelector(".python-code");ie&&(ie.textContent=Z,ie.removeAttribute("\
data-highlighted"),queueHighlight(W,Z));const Se=W.querySelector('.copy-btn[data-copy="code"]');Se&&
Se.setAttribute("data-code",encodeURIComponent(Z).replace(/'/g,"%27"))}if(E.output!==void 0){const Z=E.
output==null?"":String(E.output),ie=W.querySelector(".python-output");ie&&(ie.textContent=Z);const Se=W.
querySelector('.copy-btn[data-copy="output"]');Se&&Se.setAttribute("data-code",encodeURIComponent(Z).
replace(/'/g,"%27"))}}}else if(Y.type==="content"){const E=Y.content===null||Y.content===void 0?"":String(
Y.content);v+=E,/[`~]/.test(E)&&activateDeferredCodingModeFromStream(v),S||(S=o.querySelector(".cont\
ent-area")||document.createElement("div"),S.className="prose prose-invert text-sm break-words",o.contains(
S)||o.appendChild(S)),Le=!0}else if(Y.type==="error"){M=!0,Q=!0,o.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
Y.content)),showToast(Y.content||"Unknown error","error",!0);break}}catch{}if(oe&&_&&(_.textContent=
x,userAutoScroll&&(_.scrollTop=_.scrollHeight)),Le&&S){const le=Date.now();if(le-H>100){const Y=snapshotCodeCollapse(
S);renderAiMarkdownInto(S,v,{incrementalMath:!0}),applyCodeCollapse(S,Y,!0),H=le}}scrollToBottom()}if(ee&&
ee(),S){const de=snapshotCodeCollapse(S);renderAiMarkdownInto(S,v,{incrementalMath:!0}),applyCodeCollapse(
S,de,!0)}vibrateHelper([100,50,100]),o&&queueHighlight(o,v),o&&o.querySelectorAll(".thought-content").
forEach(Ce=>Ce.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(V){const te=V.name==="AbortError"&&isManualStopAbortForThread(l);V.name==="Abo\
rtError"&&!te&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
te||(Ae=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{B&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),ee&&ee(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===i&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Ae)return reconnectPendingStreamUntilAvailable(
{job_id:n,model:a},l)}r(resumePendingStream,"resumePendingStream");function updateThreadHighlighting(){
const e=get("thread-list");if(!e)return;e.querySelectorAll("[data-thread-id]").forEach(i=>{i.dataset.
threadId===String(currentThreadId)?i.classList.add("bg-gray-700/60","border-l-2","border-blue-500"):
i.classList.remove("bg-gray-700/60","border-l-2","border-blue-500")})}r(updateThreadHighlighting,"up\
dateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){snapshotSidebarHistory("\
loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory("loadThreads-s\
tart append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const n=get("search-box"),i=n?n.value:"";
if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");return}const o=await(await apiFetch(
`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(i)}&page=${threadPage}`)).json(),l=get("th\
read-list");if(!l)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-r\
eplace-settings-open");return}const d=o&&Array.isArray(o.threads)?o.threads.length:-1,m=l.querySelectorAll(
"[data-thread-id]").length;if(d===0&&m>0&&String(i||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(l.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const h=get("scroll-sentinel");h&&threadObserver.
observe(h)}}const c=get("scroll-sentinel");o&&Array.isArray(o.threads)?(o.threads.forEach(d=>{const m=String(
d.id),h=document.createElement("div"),y=d.is_bookmarked?"text-yellow-400":"text-gray-500",v=d.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=m===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";h.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,h.dataset.threadId=m,h.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${y} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${m}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(d.title||"No Title")}</spa\
n>${v}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${m}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${m}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,h.onclick=_=>{_.target.closest("button")||_.target.closest("[dat\
a-thread-actions]")||loadMessages(m)},c?l.insertBefore(h,c):l.appendChild(h)}),hasMoreThreads=!!o.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+o.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(n){console.error("Faile\
d to load threads:",n),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}r(loadThreads,"loadThreads");function initPullToRefresh(e,n){
const i=get(e);if(!i)return;const a=`${e}-pull-indicator`,o=60,l=88,c=52,d=.5,m=8;let h=0,y=!1,v=0,x=null;
const w=r(()=>get(a),"indicatorEl"),_=r(()=>{const M=w();return M?M.querySelector(".ptr-pull-label"):
null},"labelEl"),S=r(M=>{const P=w();if(!P)return;P.style.height=Math.min(M,l)+"px",P.classList.toggle(
"active",M>2),P.classList.toggle("pull-ready",M>=o);const H=_();H&&(H.textContent=M>=o?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),L=r(()=>{const M=w();M&&(M.style.height=
"0px",M.classList.remove("active","pull-ready","refreshing"),M.classList.remove("dragging"))},"reset\
PullUI");i.addEventListener("touchstart",M=>{if(x){y=!1;return}if(i.scrollTop>0){y=!1;return}const P=M.
touches[0];P&&(h=P.clientY,v=0,y=!0)},{passive:!0}),i.addEventListener("touchmove",M=>{if(!y||x)return;
if(i.scrollTop>0){y=!1;return}const P=M.touches[0];if(!P)return;const H=P.clientY-h;if(H<=0){v>0&&(v=
0,S(0)),y=!1;return}const Q=w();Q&&!Q.classList.contains("dragging")&&Q.classList.add("dragging"),v=
Math.min(H*d,l),S(v),H>=m&&M.preventDefault()},{passive:!1}),i.addEventListener("touchend",()=>{if(!y||
(y=!1,x))return;const M=w();M&&M.classList.remove("dragging");const P=v>=o;if(v=0,!P){L();return}let H;
try{H=n()}catch{H=null}const Q=w();if(Q){Q.classList.add("refreshing"),Q.style.height=c+"px";const ee=Q.
querySelector(".ptr-pull-label");ee&&(ee.textContent="\u66F4\u65B0\u4E2D...")}H&&typeof H.then=="fun\
ction"?(x=H,H.catch(()=>{}).finally(()=>{x=null,L()})):(x=Promise.resolve(),setTimeout(()=>{x=null,L()},
400))}),i.addEventListener("touchcancel",()=>{y=!1,v=0,L()})}r(initPullToRefresh,"initPullToRefresh");
const initThreadPullToRefresh=r(()=>initPullToRefresh("thread-list",()=>loadThreads(!1)),"initThread\
PullToRefresh"),initGemPullToRefresh=r(()=>initPullToRefresh("gem-list",()=>loadGems()),"initGemPull\
ToRefresh"),initPullToRefreshAll=r(()=>{initThreadPullToRefresh(),initGemPullToRefresh()},"initPullT\
oRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;const mcpCardIdSelector=r(e=>"mcp_\
card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),mcpEscHtml=r(e=>String(e==null?
"":e).replace(/[&<>"']/g,n=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[n]),"mcpE\
scHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||"MCP")} / ${mcpEscHtml(e.tool_name||
e.internal_name||"")}`}r(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){if(!e)return null;
let n=e.querySelector(".mcp-execution-list");return n||(n=document.createElement("div"),n.className=
"mcp-execution-list mt-3",n.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),e.appendChild(
n)),n}r(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,n){if(!e||!n||!n.
type)return;const i=["start","result","error"].includes(n.type),a=i?getMcpExecutionList(e):null;if(i&&
!a)return;const o=mcpCardIdSelector(n.id||"mcp_"+Date.now());if(n.type==="start"){if(a.querySelector(
'[data-mcp-card="'+o+'"]'))return;const l=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${o}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(n)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;a.insertAdjacentHTML("beforeend",l);return}if(n.type==="result"){let l=a.querySelector('[dat\
a-mcp-card="'+o+'"]');const c=n.summary||"";if(l)l.classList.remove("mcp-running"),l.classList.add("\
mcp-done"),l.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(n)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const d=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${o}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(n)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;a.insertAdjacentHTML("beforeend",d),l=a.querySelector('[data-mcp-card="'+o+'"]')}if(c){const d=document.
createElement("div");d.className="mcp-box-note",d.textContent=c.split(`
`)[0].slice(0,220),l&&l.appendChild(d)}return}if(n.type==="error"){let l=a.querySelector('[data-mcp-\
card="'+o+'"]');const c=n.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(l)l.classList.remove("mcp-running"),l.classList.add("mcp-error"),l.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(n)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const m=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${o}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(n)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;a.insertAdjacentHTML("beforeend",m),l=a.querySelector('[data-mcp-card="'+o+'"]')}const d=document.
createElement("div");d.className="mcp-box-note mcp-box-note-err",d.textContent=String(c).slice(0,300),
l&&l.appendChild(d);return}if(n.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&n.id&&activeMcpDecision.id===n.id){const l=get("mcp-decision-modal");if(l&&!l.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}r(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const i=get(
"mcp-decision-server"),a=get("mcp-decision-tool"),o=get("mcp-decision-args");if(i&&(i.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),a&&(a.textContent=e.tool_name||""),o){let d=e.
args_preview||"";try{const m=JSON.parse(d);d=JSON.stringify(m,null,2)}catch{}o.textContent=d}const l=get(
"mcp-decision-allow"),c=get("mcp-decision-deny");l&&(l.onclick=()=>submitMcpDecision("allow")),c&&(c.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}r(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const n=get("mcp-decision-modal");try{n&&
hideModal("mcp-decision-modal")}catch{}const i=activeMcpDecision?activeMcpDecision.jobId:null,a=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!i)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
i)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:a})})}catch{}}r(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const n=get("batch-notification-banner"),i=get("batch-no\
tification-text"),a=get("batch-notification-open");if(!n||!i||!e||!e.length)return;const o=e[0],l=o.
thread_id;i.textContent=e.length===1?`${o.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,n.classList.
remove("hidden"),a&&(a.onclick=async()=>{n.classList.add("hidden"),l&&await loadMessages(l)});const c=get(
"batch-notification-close");c&&(c.onclick=()=>n.classList.add("hidden"))}r(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const n=await e.json().catch(()=>({}));(n.active||[]).some(o=>currentThreadId&&String(o.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const a=n.
completed||[];a.length&&(showGeminiBatchCompletionBanner(a),a.some(o=>String(o.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}r(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);let chatTransitionSequence=0,chatTransitionTimer=null;const CHAT_TRANSITION_DURATION_MS=460;function playChatTransition(e){
const n=get("chat-transition-veil");if(!n)return;const i=++chatTransitionSequence;if(chatTransitionTimer&&
(clearTimeout(chatTransitionTimer),chatTransitionTimer=null),window.matchMedia&&window.matchMedia("(\
prefers-reduced-motion: reduce)").matches){n.classList.remove("is-active"),n.removeAttribute("data-t\
ransition-kind");return}n.dataset.transitionKind=e||"history",n.classList.remove("is-active"),n.offsetWidth,
n.classList.add("is-active"),chatTransitionTimer=setTimeout(()=>{i===chatTransitionSequence&&(n.classList.
remove("is-active"),chatTransitionTimer=null)},CHAT_TRANSITION_DURATION_MS)}r(playChatTransition,"pl\
ayChatTransition");async function toggleBookmark(e,n){e&&e.stopPropagation(),await apiFetch(`/api/th\
reads/${n}/bookmark`,{method:"POST"}),loadThreads()}r(toggleBookmark,"toggleBookmark");async function loadMessages(e,n={}){
const i=++threadLoadSequence;window.closeHistoryModal&&window.closeHistoryModal();const a=!!n.preserveDraft,
o=!!n.silent;o||resumeChatAutoScroll({scroll:!1});const l=o?snapshotCodeCollapseByMessage(get("chat-\
container")):null;let c="",d="",m=[];if(a){const h=get("prompt-input");c=h?h.value:"",d=h?h.style.height:
"",m=currentImageUrls?currentImageUrls.slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();
o||playChatTransition("history"),currentThreadId=e!=null?String(e):e,n.skipHistory||history.pushState(
{},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),get("welcome-scree\
n").classList.add("hidden"),o||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());try{
const h=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);h.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const y=await apiFetch(h.toString());if(!y.
ok)throw new Error(`thread request failed (${y.status})`);const v=await y.json();if(!v||!Array.isArray(
v.messages))throw new Error("invalid thread response");if(i!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(
v&&v.title),allMessages=v.messages,threadHasOlderMessages=!!v.has_older_messages,oldestLoadedMessageId=
v.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const x=(allMessages||[]).filter(_=>_.
role==="user"&&_.content).map(_=>_.content);if(promptHistory=[...new Set(x.slice().reverse())],historyIndex=
-1,tempPrompt="",currentThreadPending=v.pending_job||null,setTemporaryChatUiState(!!(v&&v.is_temporary)),
applyTemporaryChatRuntimeMeta(v||{}),ensureTemporaryChatHeartbeat(!0),get("thread-custom-instruction")&&
(get("thread-custom-instruction").value=v.custom_instruction||""),v.last_model&&selectModelById(v.last_model),
get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!!v.enable_prompt_caching,updatePromptCacheUi()),
v.last_gem_uuid&&loadedGems.length>0){const _=loadedGems.find(S=>S.uuid===v.last_gem_uuid);_&&(threadGemMap[currentThreadId]=
_,applyActiveGem(_))}const w=localStorage.getItem(`fixed_branch_${currentThreadId}`);if(w&&allMessages.
find(_=>String(_.id)===String(w))?currentLeafId=w:allMessages.length>0?currentLeafId=allMessages[allMessages.
length-1].id:currentLeafId=null,renderThreadTree(o?{silent:o,keepScroll:o}:{silent:o,keepScroll:o,animate:!0}),
o&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):o||applyCodeCollapseByMessage(get("chat-\
container"),null,!0),currentThreadPending&&!o&&!isPendingJobSuppressed(currentThreadPending.job_id)&&
resumePendingStream(currentThreadPending),a){const _=get("prompt-input");_&&(_.value=c||"",d?_.style.
height=d:_.style.height="auto"),currentImageUrls=m,currentImageUrls&&currentImageUrls.length?(get("f\
ile-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length} file\
s ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(a||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(h){return i!==threadLoadSequence||
(console.error("Failed to load chat thread:",h),o||showChatLoadError(e),o||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}r(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),n=e?e.scrollHeight:0,i=e?e.scrollTop:0;try{const a=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);a.searchParams.set("before_id",String(oldestLoadedMessageId)),
a.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),a.searchParams.set("include_me\
ta","0");const l=await(await apiFetch(a.toString())).json(),c=Array.isArray(l.messages)?l.messages:[];
if(c.length){const d=new Set(allMessages.map(h=>h.id)),m=c.filter(h=>!d.has(h.id));m.length&&(allMessages=
m.concat(allMessages))}if(threadHasOlderMessages=!!l.has_older_messages,oldestLoadedMessageId=l.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const d=e.
scrollHeight;e.scrollTop=Math.max(0,i+(d-n))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const a=get("load-older-messages-btn");a&&threadHasOlderMessages&&
(a.disabled=!1,a.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
r(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const n=!!e.silent,i=!!e.animate&&
!n,a=!!e.keepScroll,o=get("chat-container");if(!o)return;let l=null;if(a&&(l=o.scrollTop),o.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const c={};allMessages.
forEach(w=>{c[w.id]=w,w.childrenIds=[]}),allMessages.forEach(w=>{w.parent_id&&c[w.parent_id]&&c[w.parent_id].
childrenIds.push(w.id)}),(!currentLeafId||!c[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const d=[];let m=c[currentLeafId];for(;m;)d.unshift(m),m=c[m.parent_id];const h=buildTokenTotals(
d),y=buildTokenTotals(allMessages),v=document.createDocumentFragment();if(threadHasOlderMessages){const w=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
_=loadingOlderMessages?"disabled":"",S=document.createElement("div");S.className="mb-3 text-center",
S.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${_}><i class="fas fa-clock-rotate-left mr-1"></i>${w}</button>`,v.appendChild(S)}
d.forEach(w=>{const _=w.parent_id?c[w.parent_id]:null,S=_?_.childrenIds:allMessages.filter(M=>!M.parent_id).
map(M=>M.id),L=S.length>1?{current:S.indexOf(w.id)+1,total:S.length,siblings:S}:null;renderMessage(w.
id,w.role,w.content,w.image_url,w.thought_data,w.model,L,i,w.quote_text,w.tokens,w.tokens_in,w.tokens_out,
w.is_encrypted,w.tokens_content,w.tokens_thought,v,!1,w.parent_id,w.gem_name,w.batch_job)});const x=currentThreadPending;
if(x&&!isPendingJobSuppressed(x.job_id)){const w=x.message_id,_=new Set(d.map(M=>M.id)),S=d.length?d[d.
length-1]:null;if(w&&_.has(w)&&currentLeafId===w||!w&&S&&S.role==="user"){const M=x.job_id?`pending-${x.
job_id}`:null;renderPendingMessage(v,i,!1,M,x.model||null)}}if(o.appendChild(v),updateTotalTokenBar(
h.tokens_total,h,y),currentParentId=currentLeafId,a&&l!==null?restoreThreadTreeScroll(o,l):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(o,o&&o.textContent||"");else if(queueHighlight(o),d.length){
const w=d[d.length-1]&&d[d.length-1].content;queueMathTypeset(o,w)}}r(renderThreadTree,"renderThread\
Tree");function restoreThreadTreeScroll(e,n){if(!e)return;const i=e.scrollHeight-e.clientHeight;userAutoScroll&&
!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(n,i)),chatLastScrollTop=
e.scrollTop,syncScrollToBottomButton()}r(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){
currentLeafId=e;const n={};allMessages.forEach(a=>{n[a.id]=a,a.childrenIds=[]}),allMessages.forEach(
a=>{a.parent_id&&n[a.parent_id]&&n[a.parent_id].childrenIds.push(a.id)});let i=e;if(!n[i]){currentLeafId=
allMessages.length>0?allMessages[allMessages.length-1].id:null,renderThreadTree({animate:!0});return}
for(;n[i]&&n[i].childrenIds.length>0;){const a=n[i].childrenIds;i=Math.max(...a)}currentLeafId=i,renderThreadTree(
{animate:!0})}r(switchVersion,"switchVersion");async function loadGems(){try{const n=await(await apiFetch(
CHAT_CONFIG.urls.handleGems)).json();loadedGems=n;const i=get("gem-list");if(!i)return;i.innerHTML='\
<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-do\
wn ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-l\
abel"></span></div>',Array.isArray(n)&&n.forEach(a=>{const o=document.createElement("div");o.className=
"gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between it\
ems-center group",o.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa\
-gem text-blue-500"></i><span class="truncate">${escapeHtml(a.name)}</span></div><div class="flex it\
ems-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group\
-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${a.uuid}')"><i class="fas fa-p\
encil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:o\
pacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${a.uuid}')"><i class\
="fas fa-trash text-[10px]"></i></button></div>`,o.onclick=l=>{l.target.closest("button")||activateGem(
a)},i.appendChild(o)})}catch(e){console.error("Failed to load gems:",e)}}r(loadGems,"loadGems");async function openEditGemModal(e,n){
e.stopPropagation(),editingGemUuid=n;try{const a=await(await apiFetch(`/api/gems/${n}`)).json();get(
"gem-name").value=a.name,get("gem-desc").value=a.description||"",get("gem-inst").value=a.instruction,
get("gem-default-model").value=a.default_model||"",renderGemFixedPromptsForEdit(a.fixed_prompts),get(
"gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-b\
tn").innerText="Save Changes",showModal("gem-modal"),location.pathname!=="/gem"&&history.pushState({
modal:"gem"},"","/gem")}catch{showToast("Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}r(openEditGemModal,"openEditGemModal");async function createGem(e,n){await apiFetch(CHAT_CONFIG.
urls.handleGems,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,
instruction:n})}),loadGems()}r(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const n=get(
"fixed-prompts-bar");if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),
get("active-gem-name").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),
n){n.innerHTML="";let i=[];try{activeGem.fixed_prompts&&(i=JSON.parse(activeGem.fixed_prompts))}catch{}
i.length>0?(n.classList.remove("hidden"),i.forEach((a,o)=>{const l=document.createElement("button");
l.className="fixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover\
:bg-gray-600 text-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex item\
s-center",l.style.animationDelay=`${o*40}ms`,l.textContent=String(a.name||""),l.onclick=()=>{const c=get(
"prompt-input");c&&(c.value=a.content,c.dispatchEvent(new Event("input")),sendMessage())},n.appendChild(
l)})):n.classList.add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),n&&(n.innerHTML=
"",n.classList.add("hidden"));get("sys-prompt-option").style.opacity="1"}r(applyActiveGem,"applyActi\
veGem");function syncActiveGemForThread(e){const n=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(
n)}r(syncActiveGemForThread,"syncActiveGemForThread");async function saveThreadGemUuid(e,n){try{await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({last_gem_uuid:n,thread_id:e})})}catch{}}r(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,n){
currentThreadId?(threadGemMap[currentThreadId]=e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\
\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,"success"),n||saveThreadGemUuid(currentThreadId,e?
e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(e),allMessages&&allMessages.length>0&&startNewChat(
{preserveGem:!0}))}r(activateGem,"activateGem");function clearActiveGem(){currentThreadId&&(delete threadGemMap[currentThreadId],
saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=null,applyActiveGem(null)}r(clearActiveGem,
"clearActiveGem");function addGemFixedPromptRow(e="",n=""){const i=get("gem-fixed-prompts-container");
if(!i)return;const a=document.createElement("div");a.className="flex gap-2 items-start gem-fixed-pro\
mpt-row ui-enter",a.innerHTML=`
                <input type="text" class="gem-fp-name bg-gray-900 border border-gray-600 rounded p-1\
.5 text-white text-[10px] w-24" placeholder="\u540D\u524D" value="${escapeHtml(e)}" autocomplete="of\
f" spellcheck="false">
                <textarea class="gem-fp-content flex-1 bg-gray-900 border border-gray-600 rounded p-\
1.5 text-white text-[10px] h-9 resize-none" placeholder="\u30D7\u30ED\u30F3\u30D7\u30C8\u5185\u5BB9" spellcheck="false">${escapeHtml(
n)}</textarea>
                <button type="button" class="text-gray-500 hover:text-red-400 p-1.5" onclick="this.p\
arentElement.remove()"><i class="fas fa-times"></i></button>
            `,i.appendChild(a)}r(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),n=[];return e.forEach(i=>{const a=i.querySelector(
".gem-fp-name").value.trim(),o=i.querySelector(".gem-fp-content").value.trim();a&&o&&n.push({name:a,
content:o})}),n.length>0?JSON.stringify(n):null}r(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const n=get("gem-fixed-prompts-container");if(n){n.innerHTML="";try{e&&JSON.parse(e).forEach(a=>addGemFixedPromptRow(
a.name,a.content))}catch{}}}r(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}r(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}r(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),n=getTemporaryChatTimeoutLabel(),i=!!temporaryChatEnabled,a=[
"sidebar-chat-title","mobile-chat-title"],o=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],l=["sidebar-chat-ttl","mobile-chat-ttl"];a.forEach(c=>{const d=get(c);d&&(d.textContent=e)}),
o.forEach(c=>{const d=get(c);d&&d.classList.toggle("hidden",!i)}),l.forEach(c=>{const d=get(c);d&&(i&&
n?(d.textContent=n,d.classList.remove("hidden")):(d.textContent="",d.classList.add("hidden")))})}r(updateCurrentChatHeaderUi,
"updateCurrentChatHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="str\
ing"?e:null,updateCurrentChatHeaderUi()}r(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){
tempChatExpiresAtMs=null,updateCurrentChatHeaderUi()}r(resetTemporaryChatExpiresAt,"resetTemporaryCh\
atExpiresAt");function applyTemporaryChatRuntimeMeta(e){if(!e||typeof e!="object")return;Object.prototype.
hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(e.timeout_seconds);let n=null;
const i=Number(e.temp_chat_expires_at);if(Number.isFinite(i)&&i>0)n=Math.floor(i*1e3);else{const a=Number(
e.temp_chat_remaining_seconds);Number.isFinite(a)&&a>=0&&(n=Date.now()+Math.floor(a*1e3))}n!==null?tempChatExpiresAtMs=
n:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=null),updateCurrentChatHeaderUi()}
r(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
r(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let i=Number(e);return Number.isFinite(i)||(i=Number(n)),Number.isFinite(i)||(i=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
i=Math.trunc(i),i<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(i=TEMP_CHAT_TIMEOUT_MIN_SECONDS),i>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(i=TEMP_CHAT_TIMEOUT_MAX_SECONDS),i}r(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),n=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
i=get("temporary-chat-welcome-desc");i&&(i.textContent=n);const a=get("temporary-chat-container");a&&
(a.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
r(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const n=get(
"set-temp-chat-timeout-seconds");n&&(n.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}r(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),n=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,n))}r(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const n=get("enable-temporary-c\
hat");n&&n.checked!==temporaryChatEnabled&&(n.checked=temporaryChatEnabled);const i=get("welcome-def\
ault-content");i&&i.classList.toggle("hidden",temporaryChatEnabled);const a=get("welcome-temporary-c\
ontent");a&&a.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}r(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}r(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}r(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const n=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
i=await n.json().catch(()=>({}));n.ok&&i&&applyTemporaryChatRuntimeMeta(i),n.ok&&i&&i.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}r(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const n=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==n)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=n,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}r(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const n=!!e;if(setTemporaryChatUiState(
n),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const i=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:n})}),a=await i.json().catch(()=>({}));if(!i.ok)throw new Error(a&&a.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(a&&a.is_temporary)),applyTemporaryChatRuntimeMeta(a||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}r(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
if(playChatTransition("new"),threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),
resetUploadState(),stopTemporaryChatHeartbeat(),setTemporaryChatUiState(!1),currentThreadTitle=null,
tempChatExpiresAtMs=null,currentThreadId=null,allMessages=[],promptHistory=[],historyIndex=-1,tempPrompt=
"",threadHasOlderMessages=!1,oldestLoadedMessageId=null,loadingOlderMessages=!1,currentLeafId=null,currentParentId=
null,currentThreadPending=null,updateTotalTokenBar(0),typeof window.__refreshAdminThreadEncState=="f\
unction")try{window.__refreshAdminThreadEncState()}catch{}e.skipHistory||history.pushState({},"","/"),
get("chat-container").innerHTML="",get("welcome-screen").classList.remove("hidden"),updateCurrentChatHeaderUi(),
get("thread-custom-instruction")&&(get("thread-custom-instruction").value=""),get("enable-prompt-cac\
he")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(
activeGem):applyActiveGem(null),loadThreads(),window.innerWidth<768&&get("overlay").click()}r(startNewChat,
"startNewChat");let threadModalLoadSeq=0;window.openThreadModal=async()=>{if(!currentThreadId)try{const a=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=a.id!==null&&a.id!==void 0?
String(a.id):a.id,setTemporaryChatUiState(!!(a&&a.is_temporary)),setCurrentChatHeaderTitle(a&&a.title),
applyTemporaryChatRuntimeMeta(a||{}),ensureTemporaryChatHeartbeat(!0),history.pushState({},"","/c/"+
a.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,n=String(currentThreadId);modalThreadId=n,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[i,a]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${n}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==n)return;if(i.ok){const o=await i.
json(),l=get("thread-app-global-sys-prompt-preview");l&&(l.value=o.global_system_prompt_effective||"");
const c=get("thread-app-global-sys-prompt-preview-status");c&&(o.global_system_prompt_enabled===!1?c.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":o.global_system_prompt_uses_time_fallback?
c.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
c.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=o.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=o.system_prompt_enabled!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=o.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",o.auto_system_prompt_notices_config||{})}if(a.ok){const o=await a.json();if(e!==threadModalLoadSeq||
modalThreadId!==n)return;const l=get("thread-custom-instruction");l&&(l.value=o.custom_instruction||
"");const c=get("thread-include-global-instruction");c&&(c.checked=o.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const n=get("save-thread-setti\
ngs-btn"),i=n?n.textContent:"";n&&(n.disabled=!0,n.textContent="\u4FDD\u5B58\u4E2D...");const a=get(
"thread-custom-instruction"),o=a?a.value:"",l=get("thread-include-global-instruction"),c=l?l.checked:
!0,d=get("thread-global-sys-prompt"),m=get("thread-global-sys-prompt-enabled");let h=null;try{h=d||m?
{system_prompt:d?d.value:"",system_prompt_enabled:m?m.checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:collectAutoSystemPromptConfigFromForm(
"thread")}:null}catch(y){sendClientDebugLog("error","Payload construction failed: "+y.message)}try{sendClientDebugLog(
"info","Starting PUT request for thread: "+e);const y=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:o,include_global_instruction:c})});
sendClientDebugLog("info","PUT request finished, status: "+y.status);let v=!0;if(h){sendClientDebugLog(
"info","Starting POST request for user settings");const x=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)});v=x.ok,sendClientDebugLog(
"info","POST request finished, status: "+x.status)}y.ok&&v?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\
\u308C\u307E\u3057\u305F","success")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(y){sendClientDebugLog("error","Save failed with error: "+y.message),showToast("\u30A8\u30E9\u30FC\
: "+y.message,"error",!0)}finally{n&&(n.disabled=!1,n.textContent=i||"\u4FDD\u5B58")}},window.openCompressionModal=
()=>{syncCompressionSettingsUi(),showModal("compression-modal"),location.pathname!=="/compression"&&
history.pushState({modal:"compression"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal(
"compression-modal"),!e&&location.pathname==="/compression"&&history.back()},get("save-compression-s\
ettings-btn").onclick=()=>{const e=get("compression-max-size").value,n=get("compression-max-dim").value,
i=get("compression-output-type").value,a=get("compression-format-only").checked;setCompressionSettings(
e,n,i,a);const o=r((c,d)=>{get(c)&&get(d)&&(get(d).value=get(c).value)},"syncBack");o("modal-gpt-ima\
ge-size","gpt-image-size"),o("modal-gpt-image-quality","gpt-image-quality"),o("modal-gpt-image-forma\
t","gpt-image-format"),o("modal-gpt-image-compression","gpt-image-compression"),o("modal-gemini-imag\
e-aspect","gemini-image-aspect"),o("modal-gemini-image-size","gemini-image-size"),o("modal-grok-imag\
e-aspect","grok-image-aspect"),o("modal-grok-image-resolution","grok-image-resolution"),o("modal-gro\
k-image-quality","grok-image-quality"),o("modal-ocr-table-format","ocr-table-format"),o("modal-ocr-p\
ages","ocr-pages");const l=r((c,d)=>{get(c)&&get(d)&&(get(d).checked=get(c).checked)},"syncBackChk");
l("modal-ocr-extract-header","ocr-extract-header"),l("modal-ocr-extract-footer","ocr-extract-footer"),
l("modal-ocr-include-blocks","ocr-include-blocks"),l("modal-ocr-include-images","ocr-include-images"),
window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","s\
uccess")};async function deleteGem(e,n){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.
urls.handleGemItem.replace("0",n),{method:"DELETE"}),loadGems())}r(deleteGem,"deleteGem");async function renameThread(e,n){
e.stopPropagation();const i=prompt("Title:");if(i){const a=await apiFetch(CHAT_CONFIG.urls.updateTitle.
replace("0",n),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:i})}),
o=await a.json().catch(()=>({}));a.ok&&currentThreadId===String(n)&&setCurrentChatHeaderTitle(o&&o.title||
i),loadThreads()}}r(renameThread,"renameThread");async function deleteThread(e,n){e.stopPropagation(),
confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace("0",n),{method:"DELETE"}),
currentThreadId===n?startNewChat():loadThreads())}r(deleteThread,"deleteThread");async function deleteMessage(e){
confirm("Delete this message and subsequent history?")&&(await apiFetch(CHAT_CONFIG.urls.deleteMessage.
replace("0",e),{method:"DELETE"}),loadMessages(currentThreadId))}r(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=r(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=r(
e=>{if(!e)return"";try{const n=new Date(e);return Number.isNaN(n.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(n)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=r(e=>{if(!e)
return"";let n=String(e).trim();if(!n)return"";try{n.includes("://")&&(n=new URL(n,window.location.origin).
pathname||"")}catch{}n.includes("?")&&(n=n.split("?",1)[0]),n.includes("#")&&(n=n.split("#",1)[0]),n=
n.replace(/^\/+/,""),n.startsWith("files/")&&(n=n.slice(6));try{n=decodeURIComponent(n)}catch{}return n},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=r(e=>{const n=pdfNormalizeAttachmentPath(e);return n?
`${window.location.origin}/files/${encodeURI(n)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=r(
e=>{const n=pdfNormalizeAttachmentPath(e);return n?`${window.location.origin}/${PDF_IMAGE_EXTS.has((n.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(n)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=r(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
i=>{const a=pdfNormalizeAttachmentPath(i&&i.path?i.path:i);if(!a)return null;const o=i&&i.filename?i.
filename:a.split("/").pop(),l=i&&i.source?String(i.source):"attachment",c=!!(i&&i.is_image),d=i&&i.url?
i.url:buildPdfAttachmentUrl(a),m=i&&i.preview_url?i.preview_url:buildPdfAttachmentPreviewUrl(a);return{
path:a,filename:o,source:l,isImage:c,url:d,previewUrl:m}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=r(e=>{const n=e&&e.thread?e.thread:{},i=Array.isArray(e&&e.messages)?e.messages:
[],o=i.some(m=>maybeNeedsMathJax(m.content)||maybeNeedsMathJax(m.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",l=n.title||"AI Chat",c=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(i.length)},{label:"Version",value:`AI Playground ${appVersion}`}],d=i.map(m=>{const h=m.
role==="user",y=m.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(m.quote_text)}\
</div>`:"",v=m.thought_text?`<div class="thought">${escapeHtml(m.thought_text)}</div>`:"",x=h?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(m.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(m.content||"")}</div>`,w=buildPdfMessageAttachments(m),_=w.length?`<div \
class="attachments">${w.map(M=>M.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(M.previewUrl)}\
" alt="${pdfEscapeAttr(M.filename)}"><div class="file-caption">${pdfEscapeAttr(M.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(M.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
M.filename)}</span><span class="file-source">${pdfEscapeAttr(M.source)}</span></span></a></div>`).join(
"")}</div>`:"",S=[];m.model&&!h&&S.push(m.model),m.tokens!==null&&m.tokens!==void 0&&S.push(`tokens:${m.
tokens}`),m.tokens_in!==null&&m.tokens_in!==void 0&&S.push(`in:${m.tokens_in}`),m.tokens_out!==null&&
m.tokens_out!==void 0&&S.push(`out:${m.tokens_out}`),m.tokens_thought!==null&&m.tokens_thought!==void 0&&
S.push(`thought:${m.tokens_thought}`),m.is_encrypted&&S.push("encrypted"),m.parent_id!==null&&m.parent_id!==
void 0&&S.push(`parent:#${m.parent_id}`);const L=S.length?`<div class="message-meta">${pdfEscapeAttr(
S.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${h?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${h?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${h?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(m.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${y}
                            ${x}
                            ${v}
                            ${_}
                            ${L}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(l)} - PDF Export</title>
        ${o}
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
        <h1>${pdfEscapeAttr(l)}</h1>
        <p>\u30B9\u30EC\u30C3\u30C9 ID: ${pdfEscapeAttr(n.public_id||"")}\u3002\u8868\u793A\u4E2D\u306E\u5C65\u6B74\u3092\u305D\u306E\u307E\u307E\u5370\u5237\u3067\u304D\u308B\u3088\u3046\u306B\u3001\u753B\u9762\u30AD\u30E3\u30D7\u30C1\
\u30E3\u3067\u306F\u306A\u304F\u5168\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u518D\u69CB\u6210\u3057\u3066\u51FA\u529B\u3057\u3066\u3044\u307E\u3059\u3002</p>
        <div class="meta-grid">
        ${c.map(m=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(m.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(m.value)}</div></div>`).join("")}
        </div>
        </section>
        <main id="pdf-message-list" class="message-list">${d||'<div class="meta-card" style="margin-\
top:20px;background:#fff;color:var(--muted);text-align:center;border:1px dashed rgba(148,163,184,0.5\
);padding:28px;border-radius:20px;">\u3053\u306E\u30B9\u30EC\u30C3\u30C9\u306B\u306F\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'}\
</main>
        </div>
        </body>
        </html>`},"buildPdfDocumentHtml");async function openThreadPdfPrintDialog(){if(!currentThreadId){
showToast("PDF\u5316\u3059\u308B\u30B9\u30EC\u30C3\u30C9\u3092\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}if(activePdfPrintFrame){showToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const e={isLock:!0};activePdfPrintFrame=e;const n=showProgressToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\
\u3059","info");n.update(5);try{const i=new URL(PDF_PRINT_ROUTE.replace("0",currentThreadId),window.
location.origin);currentLeafId!=null&&String(currentLeafId).trim()&&i.searchParams.set("leaf_id",String(
currentLeafId));const a=await apiFetch(i.toString(),{headers:{Accept:"application/json"}});if(n.update(
20),!a.ok){activePdfPrintFrame=null,n&&n.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const o=await a.json().catch(()=>null);if(n.update(30),!o){activePdfPrintFrame=null,
n&&n.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u89E3\u6790\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const l=document.createElement("iframe");activePdfPrintFrame=l,l.setAttribute("ar\
ia-hidden","true"),l.style.position="fixed",l.style.right="0",l.style.bottom="0",l.style.width="1px",
l.style.height="1px",l.style.opacity="0",l.style.pointerEvents="none",l.style.border="0";let c=null;
const d=r(()=>{c&&(clearTimeout(c),c=null),n&&n.remove(),(activePdfPrintFrame===l||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{l.parentNode&&l.parentNode.removeChild(l)}catch{}},"cleanup");c=setTimeout(
()=>{activePdfPrintFrame===l&&(console.log("PDF print cleanup fallback triggered"),d())},6e4),l.onload=
async()=>{try{const y=l.contentDocument,v=l.contentWindow;if(!y||!v){d(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(n.update(40),(Array.isArray(o&&o.messages)?o.messages:[]).
some(M=>maybeNeedsMathJax(M.content)||maybeNeedsMathJax(M.thought_text))&&(v.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
n.update(50),y.fonts&&y.fonts.ready)try{await y.fonts.ready}catch{}n.update(60);const _=Array.from(y.
images||[]),S=Promise.all(_.map(M=>M.complete?Promise.resolve():new Promise(P=>{M.addEventListener("\
load",P,{once:!0}),M.addEventListener("error",P,{once:!0})})));if(await Promise.race([S,new Promise(
M=>setTimeout(M,5e3))]),n.update(80),y.getElementById("MathJax-script")){let M=0;for(;M<100&&(!v.MathJax||
typeof v.MathJax.typesetPromise!="function");)await new Promise(P=>setTimeout(P,50)),M++;if(v.MathJax&&
typeof v.MathJax.typesetPromise=="function")try{await v.MathJax.typesetPromise()}catch(P){console.error(
"PDF MathJax typeset failed",P)}}n.update(95),setTimeout(()=>{try{v.focus(),v.addEventListener("afte\
rprint",()=>{d()},{once:!0}),n.update(100),setTimeout(()=>{n&&n.remove()},1e3),v.print()}catch{d(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{d(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const m=buildPdfDocumentHtml(o),h=new Blob([m],{type:"text/html"});l.src=URL.createObjectURL(
h),document.body.appendChild(l)}catch{n&&n.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}r(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}r(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const n=allMessages.
find(i=>i.id==e);if(!n||!n.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(n.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let n=e?e.value:"";return n||(n=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==n&&(e.
value=n),n||"newest"}r(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const n=getLibSortOrder(),
i=Array.isArray(e)?e.slice():[],a=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),o=r((m,h)=>a.
compare(m.filename||"",h.filename||""),"nameAsc"),l=r((m,h)=>a.compare(h.filename||"",m.filename||""),
"nameDesc"),c=r((m,h)=>(Number(h.ts)||0)-(Number(m.ts)||0),"tsDesc"),d=r((m,h)=>(Number(m.ts)||0)-(Number(
h.ts)||0),"tsAsc");return n==="name_asc"?i.sort((m,h)=>o(m,h)||c(m,h)):n==="name_desc"?i.sort((m,h)=>l(
m,h)||c(m,h)):n==="oldest"?i.sort((m,h)=>d(m,h)||o(m,h)):i.sort((m,h)=>c(m,h)||o(m,h)),i}r(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}r(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}r(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const n=!!lib.favoritesOnly;e.classList.toggle("\
is-active",n),e.setAttribute("aria-pressed",n?"true":"false");const i=e.querySelector("i");i&&(i.className=
n?"fas fa-star":"far fa-star")}r(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}r(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const n=get("lib-grid");if(!n)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const i=Array.
isArray(e);if(i){const h=n.querySelector(".lib-empty-state");h&&h.remove()}else n.innerHTML="";if(!lib.
files||!lib.files.length){if(i)return;n.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const h=get("lib-total-count");h&&(h.innerText="0 files");return}const a=sortLibraryFiles(lib.files),
o=getLibSearchQuery(),l=a.filter(h=>lib.favoritesOnly&&!h.is_favorite?!1:!o||fileNameForSearch(h).includes(
o)),c=get("lib-total-count");if(c){const h=Number(lib.totalCount)||lib.files.length;lib.hasMore||o||
lib.favoritesOnly?c.innerText=`${lib.files.length} / ${h} files`:c.innerText=`${h} files`}if(!l.length){
if(i)return;const h=lib.favoritesOnly&&!o?"fa-star":"fa-search",y=lib.favoritesOnly&&!o?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",v=lib.favoritesOnly&&
!o?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
n.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${h}"></i></div>\
<p class="lib-empty-title">${y}</p><p class="lib-empty-sub">${v}</p></div>`;return}let d=0;(i?sortLibraryFiles(
e).filter(h=>lib.favoritesOnly&&!h.is_favorite?!1:!o||fileNameForSearch(h).includes(o)):l).forEach(h=>{
try{const y=renderLibraryItem(h,d++);n.appendChild(y)}catch{}})}r(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const n=sortLibraryFiles(lib.files),i=getLibSearchQuery(),
o=(i?n.filter(m=>fileNameForSearch(m).includes(i)):n).filter(m=>m.type==="image"),l=lib.favoritesOnly?
o.filter(m=>m.is_favorite):o;if(!l.length)return;const c=l.map(m=>({url:m.url,filename:m.filename||m.
original_filename||m.url.split("/").pop(),element:null}));let d=c.findIndex(m=>m.url===e.url);d===-1&&
(d=0),openViewerWithItems(c,d)}r(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const n={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},i=String(e||"").toLowerCase();return i==="pdf"?n.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(i)?n.image:n.file}r(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,n=0){const i=document.createElement("div");i.className=
"library-thumb-card",n!=null&&(i.style.animationDelay=`${Math.min(n*.035,.45)}s`);const a=e.thumbnail_url||
e.thumb_url||e.url,o=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),l=e.type==="\
image"?`<img src="${escapeHtml(a)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(o)}"></i></div><span class="lib-file-badge">${escapeHtml(o?o.toUpperCase():"FI\
LE")}</span></div>`,c=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,d=e.is_favorite?" is-favorite":"",m=e.is_favorite?"fas fa-star":"far f\
a-star",h=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
y=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${d}" title="${h}\
" aria-label="${h}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${m}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
v=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;i.innerHTML=`<div class="lib-thumb-media-wrap">${l}</div>${c}${y}${v}`,i.
onclick=()=>{lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),i.classList.remove("is-se\
lected")):(lib.selected.add(e.filepath),i.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&i.classList.add("is-selected"),i.querySelectorAll(".lib-\
open-btn").forEach(S=>{S.onclick=L=>{L.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=i.querySelector(".lib-del-btn");w&&(w.onclick=async S=>{S.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,i)});const _=i.querySelector(".lib-favorite-btn");return _&&
(_.onclick=async S=>{S.stopPropagation(),_.disabled=!0;try{const L=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
M=await L.json().catch(()=>({}));if(!L.ok||typeof M.is_favorite!="boolean")throw new Error(M.error||
"favorite update failed");e.is_favorite=M.is_favorite,renderLibraryGrid(),showToast(M.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),_.disabled=!1}}),i}r(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let n=0;n<12;n++){const i=document.createElement("div");i.className="lib-sk\
eleton-card",i.style.animationDelay=`${Math.min(n*.04,.5)}s`,i.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(i)}}}r(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const n=e.split("/").pop()||e,i=(n.split(".").pop()||"").toLowerCase(),a=[
"png","jpg","jpeg","webp","gif"].includes(i)?"image":"file",o=FILE_BASE_URL+e,l=a==="image"?FILE_THUMB_BASE_URL+
e:null,c={filename:n,original_filename:n,filepath:e,url:o,thumbnail_url:l,type:a,ext:i,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,n),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(c),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
r(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],n=(lib.files||[]).find(l=>l.
filepath===e),i=n&&n.filename||e.split("/").pop()||e,a=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
i);if(a===null)return;const o=(a||"").trim();if(!o){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const l=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:o})}),c=await l.json().catch(
()=>({}));if(!l.ok){showToast(c.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}n&&(n.filename=c.filename||o,setAttachmentNameForPath(e,n.filename));const d=get(
"upload-list");d&&d.querySelectorAll("[data-filename]").forEach(m=>{m.getAttribute("data-filename")===
e&&setRowAttachmentName(m,n?n.filename:c.filename||o)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}r(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,n){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),n&&n.parentNode&&n.remove(),lib.files&&(lib.files=
lib.files.filter(i=>i.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}r(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}r(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],n=(lib.files||[]).find(o=>o&&o.filepath===e),i=get("lib-usage-title"),a=get("l\
ib-usage-list");if(a){i&&(i.textContent=n&&(n.filename||n.original_filename)||e.split("/").pop()||e),
a.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const o=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);o.searchParams.set("filepath",e);const l=await apiFetch(
o.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),c=await l.json().catch(()=>({}));
if(!l.ok)throw new Error(c.error||`HTTP ${l.status}`);const d=Array.isArray(c.chats)?c.chats:[];if(!d.
length){a.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(a.innerHTML="",d.forEach(m=>{const h=document.createElement("div");h.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const y=m.updated_at?new Date(m.
updated_at).toLocaleString():"";h.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(m.title||"")}">${escapeHtml(m.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(y)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const v=h.querySelector("button");v&&(v.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(m.id))}),a.appendChild(h)}),c.has_more){const m=document.
createElement("p");m.className="text-[11px] text-gray-500 text-center pt-2",m.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",a.appendChild(m)}}catch{a.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}r(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const n=get(
"lib-grid"),i=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(n);let a=null;const o=CHAT_CONFIG.
urls.getFilesLib;let l=null,c=!1;try{const m=getLibSortOrder(),h=getLibSearchQuery(),y=e?lib.nextOffset:
0,v=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(y),sort:m,q:h,favorites_only:lib.
favoritesOnly?"1":"0"}),x=await apiFetch(o+"?"+v.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!x.ok)throw new Error("HTTP "+x.status);l=await x.json(),c=!0}catch(m){a=m}if(!c){
console.error("Library load failed:",a),!e&&n?n.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,i&&(i.disabled=!1,i.hidden=!lib.hasMore);return}let d=Array.isArray(l)?l:
l&&Array.isArray(l.files)?l.files:[];l&&!Array.isArray(l)&&(lib.totalCount=Number(l.total)||0,lib.hasMore=
!!l.has_more,lib.nextOffset=(Number(l.offset)||0)+(Number(l.limit)||d.length));try{const m=FILE_BASE_URL,
h=FILE_THUMB_BASE_URL,y=new Set(d.map(x=>x&&x.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(x=>{if(d.length>=LIBRARY_PAGE_SIZE||!x||y.has(x))return;const w=getAttachmentNameForPath(
x)||x.split("/").pop()||x,_=(w.split(".").pop()||"").toLowerCase(),S=["png","jpg","jpeg","webp","gif"].
includes(_)?"image":"file",L=S==="image"?h+x:null;d.unshift({filename:w,original_filename:w,filepath:x,
url:m+x,thumbnail_url:L,type:S,ext:_,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),y.add(x)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const m=d.filter(y=>y&&y.filepath&&
y.url);let h=[];if(e){const y=new Set(lib.files.map(v=>v.filepath));h=m.filter(v=>!y.has(v.filepath)),
lib.files.push(...h)}else lib.files=m;lib.files.forEach(y=>{y&&y.filepath&&setAttachmentNameForPath(
y.filepath,y.filename||y.original_filename||"")}),lib.fileSet=new Set(lib.files.map(y=>y.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
h:null)}catch(m){a=a||m}a&&n&&(console.error("Library load failed:",a),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):n.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,i&&(i.disabled=!1,i.hidden=!lib.hasMore)}r(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}r(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.from(lib.selected).forEach(o=>{const l=isAudioPath(o),
c=isVideoPath(o);if(l&&!e.audio||c&&!e.video){l&&(n+=1),c&&(i+=1);return}const d=normalizeAttachmentPath(
o);if(!d)return;const m=(lib.files||[]).find(h=>h&&h.filepath===o);m&&m.filename&&setAttachmentNameForPath(
d,m.filename),currentImageUrls.includes(d)||currentImageUrls.push(d),setAttachmentSourceForPath(d,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),n||i){const o=[];n&&o.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&o.push(`${i}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${o.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}r(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(n=>{const i=(lib.
files||[]).find(a=>a&&a.filepath===n);if(i&&i.url){const a=document.createElement("a");a.href=i.url,
a.download=i.filename||i.original_filename||n.split("/").pop()||"file",document.body.appendChild(a),
a.click(),document.body.removeChild(a)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}r(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const n=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=n,showModal("legal-modal");const i=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!i.ok)return;const a=await i.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
a)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,n)=>{const i=decodeURIComponent(n),a=r(()=>{const o=e.getAttribute("data-copy")||"";e.innerHTML=o===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
i,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(a,2e3)},o=>{console.error(o),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(a,2e3)})},window.copyMessage=(e,n)=>{const i=messageStore[e]||
"";copyToClipboard(i,()=>{n.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>n.innerHTML='<i \
class="fas fa-copy"></i>',2e3)},a=>{console.error(a),n.innerHTML='<i class="fas fa-times"></i>',setTimeout(
()=>n.innerHTML='<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const n=e.nextElementSibling;
n.classList.contains("collapsed")?n.classList.remove("collapsed"):n.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}r(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}r(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let n=0,i=e;const a={};for((allMessages||[]).forEach(o=>a[o.id]=o);i&&a[i];){const o=a[i];n+=o.tokens||
Number(o.tokens_in||0)+Number(o.tokens_out||0),i=o.parent_id}return n}r(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const n={};let i=e;const a={};for((allMessages||
[]).forEach(o=>a[o.id]=o);i&&a[i];){const o=a[i],l=o.model||"Unknown";n[l]||(n[l]={total:0,in:0,out:0,
thought:0});const c=o.tokens||Number(o.tokens_in||0)+Number(o.tokens_out||0);n[l].total+=c,n[l].in+=
Number(o.tokens_in||0),n[l].out+=Number(o.tokens_out||0),n[l].thought+=Number(o.tokens_thought||0),i=
o.parent_id}return n}r(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const n={},i=[];allMessages.forEach(o=>n[o.id]={...o,children:[]}),
allMessages.forEach(o=>{o.parent_id&&n[o.parent_id]?n[o.parent_id].children.push(n[o.id]):o.parent_id||
i.push(n[o.id])});function a(o){const l=document.createElement("div");l.className="flex flex-col ite\
ms-center mt-4";const c=document.createElement("div"),d=String(o.id)===String(currentLeafId),m=o.id===
threadFixedBranchId,h=branchLabelNames[o.id]||(o.role==="user"?"User":"AI"),y=getCumulativeTokensForNode(
o.id);if(c.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===o.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${d?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,c.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(h)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(o.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${y}</span>
                    </div>
                    ${m?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${d?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,c.onclick=v=>{v.stopPropagation(),selectedBranchNodeId=o.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},l.appendChild(c),o.children.length>0){const v=document.createElement("div");
v.className="w-px h-4 bg-gray-700",l.appendChild(v);const x=document.createElement("div");x.className=
"flex gap-4 items-start",o.children.forEach(w=>x.appendChild(a(w))),l.appendChild(x)}return l}r(a,"r\
enderNodeRecursive"),i.forEach(o=>e.appendChild(a(o)))}r(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function updateBranchDetailPane(){const e=get("branch-detail-panel"),n=get("bran\
ch-empty-panel");if(!selectedBranchNodeId||!allMessages){e.classList.add("hidden"),n.classList.remove(
"hidden");return}const i=allMessages.find(m=>m.id===selectedBranchNodeId);if(!i)return;e.classList.remove(
"hidden"),n.classList.add("hidden"),get("br-id").innerText=i.id,get("br-date").innerText=i.created_at||
"-",get("br-model").innerText=i.model||"-";const a=i.tokens||Number(i.tokens_in||0)+Number(i.tokens_out||
0),o=getCumulativeTokensForNode(i.id);get("br-tokens").innerHTML=`<span title="Current message token\
s">${a}</span> <span class="text-gray-500">/</span> <span class="text-purple-400 font-bold" title="P\
ath total tokens">${o} total</span>`;const l=get("branch-model-breakdown"),c=getPerModelTokensForPath(
i.id);l.innerHTML="",Object.entries(c).sort((m,h)=>h[1].total-m[1].total).forEach(([m,h])=>{const y=document.
createElement("div");y.className="bg-gray-800/50 p-2 rounded border border-gray-700/50",y.innerHTML=
`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${m}</span>
                        <span class="text-blue-400 shrink-0">${h.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${h.in}</div>
                        <div title="Output tokens">Out: ${h.out}</div>
                        <div title="Thought/Reasoning tokens">${h.thought>0?`Th: ${h.thought}`:""}</\
div>
                    </div>
                `,l.appendChild(y)}),get("br-name-input").value=branchLabelNames[i.id]||"";const d=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(d.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
d.classList.replace("bg-amber-600","bg-gray-600")):(d.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
d.classList.replace("bg-gray-600","bg-amber-600"))}r(updateBranchDetailPane,"updateBranchDetailPane"),
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
return{gemini:"Gemini",openai:"OpenAI",xai:"xAI"}[String(e||"").toLowerCase()]||e||"Batch"}r(batchProviderLabel,
"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\u9001\u4FE1\u5F85\u3061",
JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",JOB_STATE_RUNNING:"\
\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\u5B8C\u4E86",
JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\u505C\u6B62",
JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}r(batchStateLabelShort,
"batchStateLabelShort");function batchStateTone(e){const n=String(e||"").toUpperCase();return n==="J\
OB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":n==="JOB_STATE_FAILED"?
"border-red-500/40 bg-red-900/20 text-red-200":n==="JOB_STATE_CANCELLED"||n==="JOB_STATE_EXPIRED"?"b\
order-gray-500/40 bg-gray-700/30 text-gray-300":n==="JOB_STATE_CANCELLING"?"border-amber-500/40 bg-a\
mber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}r(batchStateTone,
"batchStateTone");function batchFormatTime(e){if(!e)return"";let n=String(e);!/[zZ]$/.test(n)&&!/[+-]\d\d:?\d\d$/.
test(n)&&(n+="Z");const i=new Date(n);return isNaN(i.getTime())?String(e):i.toLocaleString("ja-JP",{
month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}r(batchFormatTime,"batchFormatTime");
function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove("batch-list-enter"),
e.offsetWidth,e.classList.add("batch-list-enter"))}r(playBatchListAnimation,"playBatchListAnimation");
function renderBatchJobs(e={}){const n=get("batch-list");if(!n)return;const i=batchJobsCache.filter(
o=>batchFilterMode==="active"?!!o.is_active:batchFilterMode==="done"?!o.is_active:!0),a=get("batch-c\
ount");if(a&&(a.textContent=`${i.length}\u4EF6`),n.innerHTML="",!i.length){n.innerHTML='<div class="\
batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}i.forEach(o=>{const l=document.createElement("div");l.className=
"batch-job-card";const c=escapeHtml(o.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),d=escapeHtml(
batchProviderLabel(o.provider)),m=escapeHtml(o.model||""),h=escapeHtml(batchFormatTime(o.created_at)),
y=escapeHtml(o.status_text||""),v=batchStateTone(o.state),x=o.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",w=o.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",_=o.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';l.innerHTML=`
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${c}">${c}\
</div>
                            <div class="batch-job-meta mt-1 flex flex-wrap items-center gap-2 text-[\
10px]">
                                <span class="inline-flex items-center gap-1"><i class="fas fa-layer-\
group"></i>${d}</span>
                                <span class="truncate max-w-[16rem]">${m}</span>
                                <span><i class="fas fa-history mr-1"></i>${h}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${v}">${escapeHtml(batchStateLabelShort(
o.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${y}</div>
                    ${o.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
o.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${x}${w}${_}
                    </div>`;const S=l.querySelector("[data-batch-open]");S&&(S.onclick=()=>{window.closeBatchModal(),
loadMessages(o.thread_id)});const L=l.querySelector("[data-batch-cancel]");L&&(L.onclick=()=>cancelBatchJob(
o));const M=l.querySelector("[data-batch-delete]");M&&(M.onclick=()=>deleteBatchJob(o)),n.appendChild(
l)}),e.animate&&playBatchListAnimation()}r(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const n=await apiFetch("/api/batch/jobs");if(!n.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const i=await n.json().catch(()=>({}));batchJobsCache=Array.isArray(i.jobs)?i.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}r(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const n=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),i=await n.json().catch(()=>({}));if(!n.ok){showToast(i.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}r(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const n=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),i=await n.json().catch(()=>({}));
if(!n.ok){showToast(i.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}r(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(n=>{n.classList.toggle("is-active",n===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=r(e=>new Promise(n=>{const i=getModelNameById(
e),a=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${i}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${a?
a.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=a?a.label:"API Key";const o=a?get(a.inputId):null;get("api-key-modal-input").
value=o?o.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const l=get(
"api-key-modal-save-btn"),c=get("api-key-modal-fallback-btn"),d=get("api-key-modal-cancel-btn"),m=r(
()=>{l.onclick=null,c.onclick=null,d.onclick=null},"cleanup"),h=r(y=>{y.key==="Enter"&&(y.preventDefault(),
l.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",h),l.onclick=async()=>{
const y=get("api-key-modal-input").value.trim();if(!y){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(a){const v=get(a.inputId);v&&(v.value=y);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[a.keyField]:y})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[a.keyField]=y)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",h),m(),n("set")},c.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",h),m(),n("switch")},d.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",h),m(),n("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const y=get("api-key-modal-input");y&&y.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,n=console.error,i=console.warn,a=console.info;
let o=!1;async function l(c,d){if(o||!isClientDebugLogEnabled()||d&&d[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;o=!0;const m=d.map(h=>{try{return h instanceof Error?h.stack||h.message:typeof h=="object"?JSON.
stringify(h):String(h)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(c,m)}catch{}finally{
o=!1}}r(l,"sendToServer"),console.log=function(...c){e.apply(console,c),l("log",c)},console.error=function(...c){
n.apply(console,c),l("error",c)},console.warn=function(...c){i.apply(console,c),l("warn",c)},console.
info=function(...c){a.apply(console,c),l("info",c)},window.addEventListener("error",function(c){l("e\
xception",[c.message,c.filename,c.lineno,c.colno,c.error])}),window.addEventListener("unhandledrejec\
tion",function(c){l("promise-rejection",[c.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();
