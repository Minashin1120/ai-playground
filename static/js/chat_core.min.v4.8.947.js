var pi=Object.defineProperty;var o=(e,t)=>pi(e,"name",{value:t,configurable:!0});const get=o(e=>document.getElementById(e),"get"),nativeConsoleLog=typeof console.log=="function"?console.
log.bind(console):function(){},nativeConsoleInfo=typeof console.info=="function"?console.info.bind(console):
nativeConsoleLog;let settingsModalLoaded=!1;const setSettingsSaveEnabled=o(e=>{const t=get("save-set\
tings-btn");t&&(t.disabled=!e,t.classList.toggle("opacity-60",!e),t.classList.toggle("cursor-not-all\
owed",!e),t.setAttribute("title",e?"":"\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u5B8C\u4E86\u5F8C\u306B\u4FDD\u5B58\u3067\u304D\u307E\u3059"))},
"setSettingsSaveEnabled");(function(){const t=o(r=>/(\/files\/thumb\/|\/files\/)/.test(String(r||"")),
"isFileUrl"),n=o(r=>fetch(r,{method:"GET",headers:{Range:"bytes=0-0"},cache:"no-store"}).then(l=>l.status).
catch(()=>-1),"fileUrlStatus"),i=o((r,l)=>{const d=document.createElement("div");return d.style.cssText=
"display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;height:100%\
;min-height:80px;text-align:center;padding:8px;gap:4px;",l?d.innerHTML='<i class="fas fa-key" style=\
"font-size:16px;color:#fbbf24"></i><div style="font-size:9px;color:#fcd34d;font-weight:700;line-heig\
ht:1.3">\u6697\u53F7\u30AD\u30FC\u304C\u4E00\u81F4\u3057\u306A\u3044\u305F\u3081<br>\u95B2\u89A7\u3067\u304D\u307E\u305B\u3093</div>':
d.innerHTML='<i class="fas fa-file" style="font-size:16px;color:#6b7280"></i><div style="font-size:9\
px;color:#9ca3af;font-weight:700">\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093</div>',
r&&d.setAttribute("data-file-name",String(r)),d},"buildWarning"),s=o(r=>{const l=document.createElement(
"div");return l.style.cssText="display:flex;flex-direction:column;align-items:center;justify-content\
:center;width:100%;height:100%;min-height:80px;text-align:center;padding:8px;gap:4px;",l.innerHTML='\
<i class="fas fa-hourglass-half" style="font-size:16px;color:#93c5fd"></i><div style="font-size:9px;\
color:#bfdbfe;font-weight:700;line-height:1.3">\u4E00\u6642\u7684\u306B\u6DF7\u96D1\u3057\u3066\u3044\u307E\u3059<br>\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u304F\u3060\u3055\u3044</div>',
r&&l.setAttribute("data-file-name",String(r)),l},"buildBusyWarning"),a=o(r=>String(r||"").split("?")[0].
replace("/files/thumb/","/files/"),"fullFileUrl");document.addEventListener("error",r=>{const l=r.target;
if(!l||l.tagName!=="IMG")return;const d=l.currentSrc||l.src||"";if(!t(d))return;r.stopImmediatePropagation(),
r.preventDefault();const p=String(d).split("?")[0],g=l.getAttribute("data-viewer-filename")||p.split(
"/").pop(),h=o(x=>{const L=i(g,!!x);try{l.replaceWith(L)}catch{}},"showWarning"),v=o(()=>{const x=s(
g);try{l.replaceWith(x)}catch{}},"showBusyWarning"),b=o((x,L)=>{const T=l.cloneNode(!1);T.setAttribute(
"data-file-retry",String(L));const $=x+(x.includes("?")?"&":"?")+"retry="+Date.now()+"_"+L;T.setAttribute(
"src",$);try{l.replaceWith(T)}catch{}},"retryLoad"),w=o(x=>{if(x===429||x===503){v();return}if(x===409){
h(!0);return}if(x===404||x===410||x===403){h(!1);return}const L=parseInt(l.getAttribute&&l.getAttribute(
"data-file-retry")||"0",10);if(L<2){b(d,L+1);return}if(d.includes("/files/thumb/")&&!l.getAttribute(
"data-file-fallback")){l.setAttribute("data-file-fallback","1"),b(a(d),0);return}h(!1)},"handleStatu\
s");n(d).then(w).catch(()=>{const x=parseInt(l.getAttribute&&l.getAttribute("data-file-retry")||"0",
10);if(x<2){b(d,x+1);return}if(d.includes("/files/thumb/")){v();return}h(!1)})},!0)})();const isAdminSidebarDebugEnabled=o(
()=>{try{const e=window.CHAT_CONFIG||{};return!!(e.botConfig&&e.botConfig.isAdmin)}catch{return!1}},
"isAdminSidebarDebugEnabled"),ADMIN_SIDEBAR_DEBUG_PREFIX="[admin-sidebar]",adminSidebarDebugEntries=[],
snapshotSidebarHistory=o(e=>{if(!isAdminSidebarDebugEnabled())return null;const t=get("thread-list"),
n=get("sidebar"),i=get("settings-modal"),s=get("history-modal"),a=t?window.getComputedStyle(t):null,
r=n?window.getComputedStyle(n):null,l=t?Array.from(t.querySelectorAll("[data-thread-id]")):[],d=l[0]||
null,p=d?window.getComputedStyle(d):null;let g=null;try{g=typeof threadLoading=="boolean"?threadLoading:
null}catch{g=null}const h={t:Date.now(),reason:String(e||""),path:location.pathname,vw:window.innerWidth,
liteHtml:document.documentElement.classList.contains("performance-lite-mode"),blurHtml:document.documentElement.
classList.contains("performance-blur-disabled"),liquidBody:!!(document.body&&document.body.classList.
contains("liquid-glass-mode")),blurMode:adaptiveBlurPreferenceMode,liteEnabled:adaptiveBlurLiteEnabled,
sidebarClass:n?n.className:null,sidebarDisplay:r?r.display:null,sidebarOpacity:r?r.opacity:null,sidebarVisibility:r?
r.visibility:null,compact:!!(n&&n.classList.contains("compact")),sidebarOpen:!!(n&&n.classList.contains(
"open")),listExists:!!t,listParent:t&&t.parentElement?t.parentElement.id||t.parentElement.className:
null,listClass:t?t.className:null,listChildCount:t?t.children.length:0,listItemCount:l.length,listDisplay:a?
a.display:null,listOpacity:a?a.opacity:null,listVisibility:a?a.visibility:null,listHeight:a?a.height:
null,hideCompact:!!(t&&t.classList.contains("hide-compact")),searchLen:(()=>{const v=get("search-box");
return v?String(v.value||"").length:0})(),firstItemText:d&&d.textContent?d.textContent.trim().slice(
0,40):null,firstItemOpacity:p?p.opacity:null,firstItemDisplay:p?p.display:null,firstItemVisibility:p?
p.visibility:null,firstItemClass:d?d.className:null,settingsHidden:i?i.classList.contains("hidden"):
null,settingsOpen:i?i.classList.contains("modal-open"):null,settingsDisplay:i&&i.style.display||null,
historyHidden:s?s.classList.contains("hidden"):null,threadLoading:g};adminSidebarDebugEntries.push(h),
adminSidebarDebugEntries.length>80&&adminSidebarDebugEntries.shift();try{nativeConsoleLog(ADMIN_SIDEBAR_DEBUG_PREFIX,
e,h)}catch{}return h},"snapshotSidebarHistory"),installAdminSidebarDebugObserver=o(()=>{if(!isAdminSidebarDebugEnabled())
return;const e=get("thread-list");if(!(!e||e.dataset.adminSidebarDebugObserved==="1")){e.dataset.adminSidebarDebugObserved=
"1";try{new MutationObserver(n=>{const i=n.reduce((a,r)=>a+Array.from(r.removedNodes||[]).filter(l=>l&&
l.nodeType===1&&l.getAttribute&&l.getAttribute("data-thread-id")).length,0),s=n.reduce((a,r)=>a+Array.
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
Date.now();else{const s=Date.now();if(s-adaptiveBlurMeasurementLastAt<adaptiveBlurInteractionCooldownMs||
adaptiveBlurIsBusy())return;adaptiveBlurMeasurementLastAt=s}adaptiveBlurMeasurementActive=!0;const t=[];
let n=0;const i=o(s=>{if(document.visibilityState!=="visible"){adaptiveBlurMeasurementActive=!1;return}
if(n){const h=s-n;h<=200&&t.push(h)}if(n=s,t.length<30){requestAnimationFrame(i);return}adaptiveBlurMeasurementActive=
!1;const a=[...t].sort((h,v)=>h-v),r=Math.min(17.5,Math.max(7,a[Math.floor(a.length*.2)])),l=Math.max(
28,r*1.75),d=Math.max(44,r*2.7),p=t.filter(h=>h>=l).length,g=t.filter(h=>h>=d).length;(p>=5||p>=4&&g>=
2)&&(adaptiveBlurFallbackEnabled?enableAdaptiveBlurLite():enableAdaptiveBlurFallback())},"sampleFram\
e");requestAnimationFrame(i)},"measureInteractionFrames"),measureAdaptiveBlurAfterInteraction=o(()=>{
document.readyState!=="complete"||adaptiveBlurLiteEnabled||requestAnimationFrame(()=>{adaptiveBlurLiteEnabled||
measureInteractionFrames()})},"measureAdaptiveBlurAfterInteraction");document.addEventListener("clic\
k",e=>{const t=e.target instanceof Element?e.target:null;t&&t.closest('button, a, input, select, tex\
tarea, [role="button"], [tabindex]')&&measureAdaptiveBlurAfterInteraction()},!0);const externalScriptLoads=new Map,
loadExternalScript=o((e,t)=>{if(typeof t=="function"&&t())return Promise.resolve();if(externalScriptLoads.
has(e))return externalScriptLoads.get(e);const n=new Promise((i,s)=>{const a=document.createElement(
"script");a.src=e,a.async=!0,a.crossOrigin="anonymous",a.referrerPolicy="no-referrer",a.onload=()=>i(),
a.onerror=()=>s(new Error(`\u30E9\u30A4\u30D6\u30E9\u30EA\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F: ${e}`)),
document.head.appendChild(a)});return externalScriptLoads.set(e,n),n.catch(()=>externalScriptLoads.delete(
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
null,INITIAL_LIQUID_GLASS_ENABLED=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.initialLiquidGlassEnabled),
RICH_PASTE_DEFAULT_PROMPT="\u3053\u306EPDF\u3092Markdown\u5F62\u5F0F\u306B\u5909\u63DB\u3057\u3001\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u306B\u66F8\u304D\u51FA\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
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
const n=await window.imageCompression.drawFileInCanvas(e,{fileType:t}),i=n&&n[0],s=n&&n[1];if(!s)throw new Error(
"Image conversion canvas is unavailable");let a;try{typeof s.convertToBlob=="function"?a=await s.convertToBlob(
{type:t,quality:1}):a=await new Promise((r,l)=>{s.toBlob(d=>d?r(d):l(new Error("Image conversion fai\
led")),t,1)})}finally{try{window.imageCompression.cleanupCanvasMemory(s)}catch{}try{i&&typeof i.close==
"function"&&i.close()}catch{}}return new File([a],imageFilenameForMime(e.name,t),{type:t,lastModified:e.
lastModified||Date.now()})},"convertImageFormatOnly"),setCompressionSettings=o((e,t,n,i)=>{localStorage.
setItem(COMPRESSION_SIZE_KEY,e),localStorage.setItem(COMPRESSION_DIM_KEY,t),localStorage.setItem(COMPRESSION_TYPE_KEY,
n),localStorage.setItem(COMPRESSION_FORMAT_ONLY_KEY,i)},"setCompressionSettings"),syncCompressionSettingsUi=o(
()=>{const e=get("compression-max-size"),t=get("compression-max-dim"),n=get("compression-output-type"),
i=get("compression-format-only");if(e&&(e.value=getCompressionMaxSizeMB()),t&&(t.value=getCompressionMaxDim()),
n&&(n.value=getCompressionOutputType()),i){i.checked=getCompressionFormatOnly();const g=i.checked;e&&
(e.disabled=g),t&&(t.disabled=g);const h=get("compression-size-wrap"),v=get("compression-dim-wrap");
h&&(h.style.opacity=g?"0.4":"1"),v&&(v.style.opacity=g?"0.4":"1")}const s=o((g,h)=>{get(g)&&get(h)&&
(get(h).value=get(g).value)},"sync");s("gpt-image-size","modal-gpt-image-size"),s("gpt-image-quality",
"modal-gpt-image-quality"),s("gpt-image-format","modal-gpt-image-format"),s("gpt-image-compression",
"modal-gpt-image-compression"),s("gemini-image-aspect","modal-gemini-image-aspect"),s("gemini-image-\
size","modal-gemini-image-size"),s("grok-image-aspect","modal-grok-image-aspect"),s("grok-image-reso\
lution","modal-grok-image-resolution"),s("grok-image-quality","modal-grok-image-quality"),s("ocr-tab\
le-format","modal-ocr-table-format"),s("ocr-pages","modal-ocr-pages");const a=o((g,h)=>{get(g)&&get(
h)&&(get(h).checked=get(g).checked)},"syncChk");a("ocr-extract-header","modal-ocr-extract-header"),a(
"ocr-extract-footer","modal-ocr-extract-footer"),a("ocr-include-blocks","modal-ocr-include-blocks"),
a("ocr-include-images","modal-ocr-include-images");const r=get("model-select").value,l=isGptImageModel(
r),d=isGeminiImageModel(r),p=isGrokImageModel(r);get("modal-gpt-image-options")&&get("modal-gpt-imag\
e-options").classList.toggle("hidden",!l),get("modal-gemini-image-options")&&get("modal-gemini-image\
-options").classList.toggle("hidden",!d),get("modal-grok-image-options")&&get("modal-grok-image-opti\
ons").classList.toggle("hidden",!p),get("modal-mistral-ocr-options")&&get("modal-mistral-ocr-options").
classList.toggle("hidden",!isMistralOcrModel(r))},"syncCompressionSettingsUi"),isGeminiLocalPyDialogEnabled=o(
()=>{const e=localStorage.getItem(GEMINI_LOCAL_PY_DIALOG_KEY);return e===null?!0:e==="1"||e==="true"},
"isGeminiLocalPyDialogEnabled"),setGeminiLocalPyDialogEnabled=o(e=>{localStorage.setItem(GEMINI_LOCAL_PY_DIALOG_KEY,
e?"1":"0")},"setGeminiLocalPyDialogEnabled"),syncGeminiLocalPyDialogSetting=o(()=>{const e=get("set-\
gemini-local-python-dialog");e&&(e.checked=isGeminiLocalPyDialogEnabled())},"syncGeminiLocalPyDialog\
Setting"),normalizeGeminiBackend=o(e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t===
"vertex_ai"||t==="vertex"||t==="vertexai"?"vertex_ai":"gemini_api"},"normalizeGeminiBackend"),normalizeAdminApiKeyMode=o(
e=>{const t=String(e||"").trim().toLowerCase().replace("-","_");return t==="user_only"||t==="user"||
t==="settings"||t==="user_settings"?"user_only":"env_fallback"},"normalizeAdminApiKeyMode"),syncToggleButtons=o(
(e,t,n)=>{(e||[]).forEach(i=>{const s=i.getAttribute(n)===t;i.classList.toggle("border-cyan-400",s),
i.classList.toggle("bg-cyan-900/30",s),i.classList.toggle("text-white",s),i.classList.toggle("border\
-gray-600",!s),i.classList.toggle("bg-gray-800/70",!s)})},"syncToggleButtons"),syncAdminApiKeyModeUi=o(
()=>{const e=get("set-admin-api-key-mode"),t=get("admin-api-key-mode-note"),n=get("admin-api-key-mod\
e-status"),i=get("admin-api-key-mode-toggle");if(!e)return;const s=normalizeAdminApiKeyMode(e.value);
e.value=s,i&&!i.dataset.bound&&(i.dataset.bound="1",i.querySelectorAll("[data-admin-api-key-mode]").
forEach(a=>{a.addEventListener("click",()=>{e.value=normalizeAdminApiKeyMode(a.getAttribute("data-ad\
min-api-key-mode")),syncAdminApiKeyModeUi()})})),syncToggleButtons(i?i.querySelectorAll("[data-admin\
-api-key-mode]"):[],s,"data-admin-api-key-mode"),t&&(t.textContent=s==="user_only"?"\u901A\u5E38\u30E6\u30FC\u30B6\u30FC\u3068\u540C\u3058\u304F\u3001\u3053\u306E\u753B\u9762\u3067\
\u4FDD\u5B58\u3057\u305FAPI\u30AD\u30FC/Vertex\u8A2D\u5B9A\u306E\u307F\u3092\u4F7F\u7528\u3057\u307E\u3059\u3002":
"\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u3068\u304D\u3060\u3051 .env \u3092\u30D5\u30A9\u30FC\u30EB\u30D0\u30C3\u30AF\u5229\u7528\u3057\u307E\u3059\uFF08\u65E2\u5B9A\uFF09\u3002"),
n&&(n.textContent=s==="user_only"?"\u73FE\u5728: \u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u307F\uFF08\u63A8\u5968: \u8A2D\u5B9A\u5024\u3092\u660E\u793A\u7BA1\u7406\uFF09":
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
kend-status"),s=get("gemini-backend-toggle");if(!e)return;ensureGeminiVertexCredentialsField();const a=normalizeGeminiBackend(
e.value);e.value=a,s&&!s.dataset.bound&&(s.dataset.bound="1",s.querySelectorAll("[data-gemini-backen\
d]").forEach(r=>{r.addEventListener("click",()=>{e.value=normalizeGeminiBackend(r.getAttribute("data\
-gemini-backend")),syncGeminiBackendUi()})})),syncToggleButtons(s?s.querySelectorAll("[data-gemini-b\
ackend]"):[],a,"data-gemini-backend"),t&&t.classList.toggle("hidden",a!=="vertex_ai"),n&&(n.textContent=
a==="vertex_ai"?"Vertex AI \u3092\u5229\u7528\u3057\u307E\u3059\u3002Project ID / Location \u3092\u8A2D\u5B9A\u3057\u3001ADC \u307E\u305F\u306F Vertex Service Account JSON \u3092\u7528\u610F\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002":"Gemini API \u3092\u5229\u7528\u3057\u307E\u3059\u3002API Key \u3092\u8A2D\u5B9A\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
i&&(i.textContent=a==="vertex_ai"?"\u73FE\u5728: Vertex AI\uFF08Project ID / Location / \u8A8D\u8A3C\u60C5\u5831\u304C\u5FC5\u8981\uFF09":
"\u73FE\u5728: Gemini API\uFF08Gemini API Key \u3092\u4F7F\u7528\uFF09")},"syncGeminiBackendUi"),normalizeHex=o(
e=>{if(!e)return null;let t=String(e).trim();return!t||(t.startsWith("#")||(t=`#${t}`),t.length===4&&
(t=`#${t[1]}${t[1]}${t[2]}${t[2]}${t[3]}${t[3]}`),!/^#[0-9a-fA-F]{6}$/.test(t))?null:t.toLowerCase()},
"normalizeHex"),hexToRgb=o(e=>{const t=e.replace("#",""),n=parseInt(t.slice(0,2),16),i=parseInt(t.slice(
2,4),16),s=parseInt(t.slice(4,6),16);return[n,i,s]},"hexToRgb"),mix=o((e,t,n)=>Math.round(e+(t-e)*n),
"mix"),rgbToHex=o((e,t,n)=>`#${[e,t,n].map(i=>i.toString(16).padStart(2,"0")).join("")}`,"rgbToHex"),
deriveTheme=o(e=>{const[t,n,i]=hexToRgb(e),s=rgbToHex(mix(t,255,.45),mix(n,255,.45),mix(i,255,.45)),
a=rgbToHex(mix(t,255,.7),mix(n,255,.7),mix(i,255,.7)),r=rgbToHex(mix(t,0,.18),mix(n,0,.18),mix(i,0,.18)),
l=rgbToHex(mix(t,0,.32),mix(n,0,.32),mix(i,0,.32));return{base:e,light:s,lighter:a,dark:r,darker:l,rgb:`${t}\
, ${n}, ${i}`}},"deriveTheme"),applyThemeColor=o((e,t=!1)=>{const n=normalizeHex(e)||THEME_DEFAULT,i=deriveTheme(
n),s=document.documentElement;[["--theme-500",i.base],["--theme-600",i.dark],["--theme-700",i.darker],
["--theme-300",i.light],["--theme-200",i.lighter],["--theme-rgb",i.rgb]].forEach(([r,l])=>{s.style.getPropertyValue(
r).trim()!==String(l).trim()&&s.style.setProperty(r,l)}),t&&localStorage.setItem(THEME_STORAGE_KEY,n)},
"applyThemeColor"),syncThemeInputs=o(e=>{const t=normalizeHex(e)||THEME_DEFAULT,n=get("set-theme-col\
or"),i=get("set-theme-color-text");n&&(n.value=t),i&&(i.value=t),document.querySelectorAll("#theme-p\
resets .theme-swatch").forEach(a=>{const r=normalizeHex(a.getAttribute("data-color"));a.classList.toggle(
"active",r===t)})},"syncThemeInputs"),initThemeFromServer=o(()=>{const e=normalizeHex(INITIAL_THEME_COLOR);
if(e){applyThemeColor(e,!1);return}const t=normalizeHex(localStorage.getItem(THEME_STORAGE_KEY));applyThemeColor(
t||THEME_DEFAULT,!1)},"initThemeFromServer"),LIQUID_GLASS_SURFACE_SELECTOR=["#sidebar",".composer-do\
ck","body > .flex-1 > header","#top-model-bar",".modal-panel",".modal-glass-panel",".viewer-toolbar",
".viewer-meta","#quote-bar","#slash-command-suggestions","#gem-suggestions","#total-token-bar"].join(
","),refreshLiquidGlassSurfaces=o(()=>{document.querySelectorAll(LIQUID_GLASS_SURFACE_SELECTOR).forEach(
e=>{e.classList.add("liquid-glass-surface"),e.matches(".viewer-toolbar, .viewer-meta")&&e.classList.
add("liquid-glass-clear");const t=e.matches('[data-liquid-glass-background="none"]')||!!e.closest(".\
liquid-glass-no-backdrop");e.classList.toggle("liquid-glass-no-background",t)})},"refreshLiquidGlass\
Surfaces"),applyLiquidGlassMode=o(e=>{document.body&&(document.body.classList.toggle("liquid-glass-m\
ode",!!e),e&&refreshLiquidGlassSurfaces())},"applyLiquidGlassMode");let pendingLiquidGlassPointer=null,
liquidGlassPointerFrame=0,liquidGlassPointerPaintAt=0,liquidGlassPointerSurface=null,liquidGlassPointerRect=null;
const paintLiquidGlassPointer=o(e=>{if(!pendingLiquidGlassPointer||!document.body||!document.body.classList.
contains("liquid-glass-mode")){liquidGlassPointerFrame=0;return}if(e-liquidGlassPointerPaintAt<30){liquidGlassPointerFrame=
requestAnimationFrame(paintLiquidGlassPointer);return}const t=pendingLiquidGlassPointer;pendingLiquidGlassPointer=
null;const n=t.target&&t.target.closest?t.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;if(!n){
liquidGlassPointerFrame=0;return}(n!==liquidGlassPointerSurface||!liquidGlassPointerRect)&&(liquidGlassPointerSurface=
n,liquidGlassPointerRect=n.getBoundingClientRect());const i=liquidGlassPointerRect;if(i.width&&i.height){
const s=Math.max(0,Math.min(100,(t.clientX-i.left)/i.width*100)),a=Math.max(0,Math.min(100,(t.clientY-
i.top)/i.height*100));n.style.setProperty("--glass-light-x",`${s.toFixed(1)}%`),n.style.setProperty(
"--glass-light-y",`${a.toFixed(1)}%`),liquidGlassPointerPaintAt=e}liquidGlassPointerFrame=pendingLiquidGlassPointer?
requestAnimationFrame(paintLiquidGlassPointer):0},"paintLiquidGlassPointer");document.addEventListener(
"pointermove",e=>{!document.body||!document.body.classList.contains("liquid-glass-mode")||(pendingLiquidGlassPointer=
{target:e.target,clientX:e.clientX,clientY:e.clientY},liquidGlassPointerFrame||(liquidGlassPointerFrame=
requestAnimationFrame(paintLiquidGlassPointer)))},{passive:!0}),document.addEventListener("pointerou\
t",e=>{const t=e.target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;!t||e.relatedTarget&&
t.contains(e.relatedTarget)||(pendingLiquidGlassPointer=null,t.style.removeProperty("--glass-light-x"),
t.style.removeProperty("--glass-light-y"),t.classList.remove("liquid-glass-pressed"),t===liquidGlassPointerSurface&&
(liquidGlassPointerSurface=null,liquidGlassPointerRect=null))},{passive:!0}),document.addEventListener(
"pointerdown",e=>{if(!document.body||!document.body.classList.contains("liquid-glass-mode"))return;const t=e.
target.closest?e.target.closest(LIQUID_GLASS_SURFACE_SELECTOR):null;t&&t.classList.add("liquid-glass\
-pressed")},{passive:!0});const releaseLiquidGlassPress=o(e=>{const t=e.target.closest?e.target.closest(
LIQUID_GLASS_SURFACE_SELECTOR):null;t&&t.classList.remove("liquid-glass-pressed")},"releaseLiquidGla\
ssPress");document.addEventListener("pointerup",releaseLiquidGlassPress,{passive:!0}),document.addEventListener(
"pointercancel",releaseLiquidGlassPress,{passive:!0});let liquidGlassScrollTimer=0;document.addEventListener(
"scroll",()=>{!document.body||!document.body.classList.contains("liquid-glass-mode")||(liquidGlassPointerRect=
null,document.body.classList.add("liquid-glass-scrolling"),window.clearTimeout(liquidGlassScrollTimer),
liquidGlassScrollTimer=window.setTimeout(()=>{document.body&&document.body.classList.remove("liquid-\
glass-scrolling")},140))},{passive:!0,capture:!0}),window.addEventListener("resize",()=>{liquidGlassPointerRect=
null},{passive:!0});const MODAL_ANIM_MS=280,formatBytes=o(e=>{if(e==null)return"0MB";const t=e/(1024*
1024);return t<1024?`${t.toFixed(1)}MB`:`${(t/1024).toFixed(2)}GB`},"formatBytes"),inspectSiteCacheStorage=o(
async()=>{const e={cacheCount:0,entryCount:0,totalBytes:0,storageUsageBytes:null,storageQuotaBytes:null};
if("caches"in window)try{const t=await caches.keys();e.cacheCount=t.length;for(const n of t){const i=await caches.
open(n),s=await i.keys();e.entryCount+=s.length;for(const a of s)try{const r=await i.match(a);if(!r)
continue;const l=parseInt(r.headers.get("content-length")||"",10);if(Number.isFinite(l)&&l>=0)e.totalBytes+=
l;else{const d=await r.clone().blob();e.totalBytes+=d.size||0}}catch{}}}catch{}if(navigator.storage&&
navigator.storage.estimate)try{const t=await navigator.storage.estimate();e.storageUsageBytes=Number(
t.usage||0),e.storageQuotaBytes=Number(t.quota||0)}catch{}return e},"inspectSiteCacheStorage"),loadSiteCacheUsage=o(
async()=>{const e=get("site-cache-usage-text"),t=get("site-cache-usage-detail");if(!(!e&&!t)){e&&(e.
innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D..."),t&&(t.innerText="");try{const n=await inspectSiteCacheStorage(),
i=`\u30AD\u30E3\u30C3\u30B7\u30E5\u4F7F\u7528\u91CF: ${formatBytes(n.totalBytes)} (${n.cacheCount}\u30AD\u30E3\
\u30C3\u30B7\u30E5 / ${n.entryCount}\u4EF6)`;if(n.storageQuotaBytes){const s=Math.min(100,Math.round(
n.totalBytes/n.storageQuotaBytes*100));if(e&&(e.innerText=`${i} / \u4FDD\u5B58\u9818\u57DF\u4E0A\u9650 ${formatBytes(
n.storageQuotaBytes)} (${s}%)`),t){const a=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: \u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
t.innerText=`${a} / \u30D6\u30E9\u30A6\u30B6\u306E\u5B9F\u6E2C\u5024\u3067\u3059`}}else e&&(e.innerText=
i),t&&(t.innerText=n.storageUsageBytes!==null?`\u4FDD\u5B58\u9818\u57DF\u4F7F\u7528\u91CF: ${formatBytes(
n.storageUsageBytes)}`:"\u4FDD\u5B58\u9818\u57DF\u4E0A\u9650\u306F\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u3067\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093")}catch{
e&&(e.innerText="\u30AD\u30E3\u30C3\u30B7\u30E5\u5BB9\u91CF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),
t&&(t.innerText="")}}},"loadSiteCacheUsage");let versionUpdateCachePreferenceSavePromise=Promise.resolve();
const loadStorageUsage=o(async()=>{const e=get("storage-usage-text"),t=get("storage-usage-bar");if(!(!e||
!t)){e.innerText="\u8AAD\u307F\u8FBC\u307F\u4E2D...";try{const n=await apiFetch("/api/storage",{cache:"\
no-store"});if(!n.ok)throw new Error("HTTP "+n.status);const i=await n.json(),s=Number(i.used_bytes||
0),a=Number(i.limit_bytes||0);if(i.is_unlimited||!a)e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(s)}\
 (\u7121\u5236\u9650)`,t.style.width="0%",t.style.opacity="0.5";else{const r=Math.min(100,Math.round(
s/a*100));e.innerText=`\u4F7F\u7528\u91CF: ${formatBytes(s)} / ${formatBytes(a)} (${r}%)`,t.style.width=
`${r}%`,t.style.opacity="1"}}catch{e.innerText="\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
t.style.width="0%",t.style.opacity="0.5"}}},"loadStorageUsage"),clearSiteCacheAndReload=o(async(e,t={})=>{
const{scanFirst:n=!0}=t||{},i=e?e.innerText:"";e&&(e.disabled=!0,e.innerText="\u524A\u9664\u4E2D...");
try{const s=n?await inspectSiteCacheStorage():null;await purgeCaches();const a=s?`\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5 ${formatBytes(
s.totalBytes)} \u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002`:"\u30ED\u30FC\u30AB\u30EB\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3057\u305F\u3002";
showToast(`${a} \u518D\u8AAD\u307F\u8FBC\u307F\u3057\u307E\u3059\u3002`,"success"),window.setTimeout(
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
n);const i=!!(t&&t.skipConfirm),s=!!(t&&t.skipReset);if(e==="camera-capture-modal"&&cameraCapturePendingFiles.
length>0&&!i&&!cameraCaptureBusy){attachCameraCapturedFiles();return}if(e==="rich-paste-modal"&&!i&&
hasRichPasteContent()&&!confirm("\u8CBC\u308A\u4ED8\u3051\u305F\u5185\u5BB9\u3092\u7834\u68C4\u3057\u3066\u9589\u3058\u307E\u3059\u304B\uFF1F"))
return;if(e==="marker-modal"&&(markerState.row=null),e==="camera-capture-modal"&&(s||resetCameraCapturePending(),
stopCameraCaptureStream()),!n.classList.contains("modal-open")){n.style.display="none",n.classList.remove(
"modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden");return}n.classList.remove(
"modal-open"),n.classList.add("modal-close");const a=setTimeout(()=>{n.style.display="none",n.classList.
remove("modal-close"),n.classList.remove("modal-prep"),n.classList.add("hidden"),modalCloseTimers.delete(
n)},MODAL_ANIM_MS);modalCloseTimers.set(n,a)},"hideModal");window.hideModal=hideModal;const RICH_PASTE_ALLOWED_TAGS=[
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
(e=null,t={})=>{const n=!!t.preservePrompt,i=getRichPastePrompt(),s=getRichPasteUseDefaultCheckbox();
s&&(s.checked=!!(e&&e.rich_paste_prompt_use_custom_default)),i&&!richPastePromptPreferenceSyncing&&!n&&
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
return}const i=e.querySelectorAll("img").length,s=e.querySelectorAll("table").length,a=e.querySelectorAll(
"a").length,r=e.querySelectorAll("h1,h2,h3,h4,h5,h6").length;t.textContent=`${n.length} \u6587\u5B57 / \u753B\u50CF ${i}\
 / \u8868 ${s} / \u30EA\u30F3\u30AF ${a} / \u898B\u51FA\u3057 ${r}`},"updateRichPasteStatus"),focusRichPasteEditor=o(
()=>{const e=getRichPasteCapture();if(!e)return;e.focus(),e.value=e.value||"",window.getSelection&&window.
getSelection()&&e.select&&e.select()},"focusRichPasteEditor"),clearRichPasteEditor=o((e=!0)=>{const t=getRichPasteEditor();
t&&(t.innerHTML="");const n=getRichPasteCapture();if(n&&(n.value=""),!e){const i=getRichPastePrompt();
i&&(i.value=RICH_PASTE_DEFAULT_PROMPT)}updateRichPasteStatus()},"clearRichPasteEditor"),sanitizeRichPasteStyle=o(
e=>{if(!e)return"";const t=[];return String(e).split(";").forEach(n=>{const i=n.trim();if(!i)return;
const s=i.indexOf(":");if(s<=0)return;const a=i.slice(0,s).trim().toLowerCase(),r=i.slice(s+1).trim();
if(!RICH_PASTE_SAFE_STYLE_PROPS.has(a)||!r||r.length>1e3)return;const l=r.toLowerCase();l.includes("\
url(")||l.includes("expression(")||l.includes("javascript:")||l.includes("@import")||l.includes("beh\
avior:")||l.includes("-moz-binding")||l.includes("var(")||l.includes("env(")||t.push(`${a}: ${r}`)}),
t.join("; ")},"sanitizeRichPasteStyle");let richPasteColorCanvasContext=null;const parseRichPasteCssColor=o(
e=>{const t=String(e||"").trim();if(!t||t==="inherit"||t==="currentcolor"||t==="transparent"||window.
CSS&&typeof window.CSS.supports=="function"&&!window.CSS.supports("color",t))return null;try{if(!richPasteColorCanvasContext){
const s=document.createElement("canvas");s.width=1,s.height=1,richPasteColorCanvasContext=s.getContext(
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
"*"))return makeRichPasteTheme(t,n);const s=document.createElement("div");s.setAttribute("aria-hidde\
n","true"),s.style.position="fixed",s.style.left="-100000px",s.style.top="0",s.style.width="794px",s.
style.visibility="hidden",s.style.pointerEvents="none",s.style.color="#111827",s.style.background="t\
ransparent",s.appendChild(i.content.cloneNode(!0)),document.body.appendChild(s);try{const a=[s,...Array.
from(s.querySelectorAll("*")).slice(0,5e3)],r=[],l=new Map;let d=0;const p=o(w=>Array.from(w.childNodes||
[]).reduce((x,L)=>L&&L.nodeType===Node.TEXT_NODE?x+String(L.textContent||"").replace(/\s+/g," ").trim().
length:x,0),"directTextLength");a.forEach(w=>{if(!w||w===s||!w.style)return;const x=window.getComputedStyle(
w),L=p(w);if(L>0){const $=parseRichPasteCssColor(x.color);if($&&$.a>=.5){const j=richPasteColorCss($),
J=l.get(j)||{color:$,weight:0};J.weight+=L,l.set(j,J),d+=L}}if(!!(String(w.style.backgroundColor||"").
trim()||String(w.style.background||"").trim())){const $=parseRichPasteCssColor(x.backgroundColor);if($&&
$.a>=.72){const j=String(w.textContent||"").replace(/\s+/g," ").trim().length;r.push({color:$,weight:Math.
max(1,j)})}}});const g=Array.from(l.values()).sort((w,x)=>x.weight-w.weight),h=g.length?g[0].color:null,
v=g.reduce((w,x)=>w+(richPasteColorLuminance(x.color)>=.6?x.weight:0),0);r.sort((w,x)=>x.weight-w.weight);
let b=r.length?r[0].color:null;return b||(b=d>0&&v/d>=.55?{r:11,g:11,b:12,a:1}:t),makeRichPasteTheme(
b,h||n)}catch{return makeRichPasteTheme(t,n)}finally{s.parentNode&&s.parentNode.removeChild(s)}},"de\
tectRichPasteTheme"),prepareRichPastePdfClone=o((e,t)=>{if(!e)return;const n=e.head||e.querySelector(
"head");n&&Array.from(n.querySelectorAll('link[rel="stylesheet"]')).forEach(i=>{try{i.remove()}catch{}}),
e.body&&(e.body.style.margin="0",e.body.style.background=t.background,e.body.style.color=t.foreground)},
"prepareRichPastePdfClone"),normalizeRichPasteTree=o(e=>{!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("*").forEach(t=>{if(!t||!t.getAttribute||!t.parentNode)return;const n=String(t.tagName||
"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(n)){t.remove();return}t.removeAttribute("class"),t.removeAttribute(
"id"),t.removeAttribute("role"),t.removeAttribute("aria-label"),n==="img"&&(t.setAttribute("loading",
"eager"),t.setAttribute("decoding","sync"),t.removeAttribute("srcset"),t.removeAttribute("sizes"));const i=t.
getAttribute("style");if(i){const s=sanitizeRichPasteStyle(i);s?t.setAttribute("style",s):t.removeAttribute(
"style")}})},"normalizeRichPasteTree"),extractRichPasteArticleHtml=o(e=>{const n=new DOMParser().parseFromString(
String(e||""),"text/html");if(!n.body)return"";const i=(n.body.textContent||"").replace(/\s+/g," ").
trim().length,s=n.body.querySelectorAll("*").length;if(i<1e3||s<120)return n.body.innerHTML;const r=[
...Array.from(n.body.querySelectorAll("article")),...Array.from(n.body.querySelectorAll("main")),...Array.
from(n.body.querySelectorAll('[role="main"],[role="article"]'))].filter(d=>(d.textContent||"").replace(
/\s+/g," ").trim().length>=i*.65);r.sort((d,p)=>{const g=+!!p.querySelector("h1")-+!!d.querySelector(
"h1");return g||d.querySelectorAll("*").length-p.querySelectorAll("*").length});const l=r[0]||null;return l?
l.outerHTML:n.body.innerHTML},"extractRichPasteArticleHtml"),sanitizeRichPasteHtml=o(e=>{if(!window.
DOMPurify||typeof window.DOMPurify.sanitize!="function"){const s=new DOMParser().parseFromString(String(
e||""),"text/html");return escapeHtml(s.body?s.body.textContent:"")}let t=extractRichPasteArticleHtml(
e),n=window.DOMPurify.sanitize(t||"",{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,
KEEP_CONTENT:!0});if((!n||n.trim()==="")&&e&&e.trim()!==""&&(n=window.DOMPurify.sanitize(e,{ALLOWED_TAGS:RICH_PASTE_ALLOWED_TAGS,
ALLOWED_ATTR:RICH_PASTE_ALLOWED_ATTR,KEEP_CONTENT:!0})),!n)return"";const i=document.createElement("\
template");return i.innerHTML=n,normalizeRichPasteTree(i.content),i.innerHTML},"sanitizeRichPasteHtm\
l"),normalizeRichPastePrintHtml=o(e=>{const t=document.createElement("template");t.innerHTML=String(
e||"");const n=Array.from(t.content.querySelectorAll("*")),i=n.reduce((d,p)=>{const g=String(p.style&&
p.style.display||"").trim().toLowerCase();return d+(["flex","inline-flex","grid","inline-grid"].includes(
g)?1:0)},0),s=n.reduce((d,p)=>{if(!p||!p.style||!["article","div","main","section"].includes(String(
p.tagName||"").toLowerCase()))return d;const g=String(p.getAttribute("style")||""),h=Array.from(g.matchAll(
/(?:^|;)\s*padding(?:-left|-right|-inline|-inline-start|-inline-end)?\s*:\s*([^;]+)/gi)).some(b=>Array.
from(b[1].matchAll(/(-?\d+(?:\.\d+)?)px/gi)).some(w=>Math.abs(Number(w[1])||0)>=96)),v=Array.from(g.
matchAll(/(?:^|;)\s*(?:width|min-width)\s*:\s*(-?\d+(?:\.\d+)?)px/gi)).some(b=>Math.abs(Number(b[1])||
0)>720);return d+(h||v?1:0)},0);if(n.length<=500&&i<=24&&s===0)return t.innerHTML;const a=new Set(["\
align-items","align-self","column-gap","flex","flex-basis","flex-direction","flex-grow","flex-shrink",
"flex-wrap","gap","grid","grid-auto-columns","grid-auto-flow","grid-auto-rows","grid-column","grid-c\
olumn-end","grid-column-start","grid-row","grid-row-end","grid-row-start","grid-template","grid-temp\
late-areas","grid-template-columns","grid-template-rows","justify-content","justify-items","justify-\
self","order","row-gap"]),r=new Set(["article","div","main","section"]),l=new Set(["padding","paddin\
g-left","padding-right","padding-inline","padding-inline-start","padding-inline-end"]);return n.forEach(
d=>{if(!d||!d.style)return;const p=String(d.tagName||"").toLowerCase(),g=[];String(d.getAttribute("s\
tyle")||"").split(";").forEach(h=>{if(!h||h.indexOf(":")<0)return;const v=h.indexOf(":"),b=h.slice(0,
v).trim().toLowerCase();let w=h.slice(v+1).trim();if(!(!b||!w||a.has(b))&&!["height","max-height","m\
in-height","overflow","overflow-x","overflow-y"].includes(b)&&!(["width","min-width"].includes(b)&&r.
has(p))){if(l.has(b)&&r.has(p)&&Array.from(w.matchAll(/(-?\d+(?:\.\d+)?)px/gi)).map(L=>Math.abs(Number(
L[1])||0)).some(L=>L>=96)&&(w="0px"),b==="display"){const x=w.toLowerCase();["flex","grid"].includes(
x)?w="block":["inline-flex","inline-grid"].includes(x)&&(w="inline-block")}g.push(`${b}: ${w}`)}}),g.
length?d.setAttribute("style",g.join("; ")):d.removeAttribute("style")}),t.innerHTML},"normalizeRich\
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
return!1;let n=!1;for(const i of t){if(!i)continue;const s=Array.from(i.types||[]);let a=!1;if(s.includes(
"text/html")){const d=await(await i.getType("text/html")).text();d&&insertHtmlIntoRichPasteEditor(d)&&
(n=!0,a=!0)}if(!a&&s.includes("text/plain")){const d=await(await i.getType("text/plain")).text();d&&
(insertTextIntoRichPasteEditor(d),n=!0)}const r=s.find(l=>l&&l.startsWith("image/"));if(!a&&r){const l=await i.
getType(r);await insertClipboardImageBlob(l,"clipboard-image")&&(n=!0)}}return n},"readClipboardRich\
Content"),ingestRichPasteClipboardData=o(async e=>{if(!e)return!1;let t=!1;const n=e.getData&&e.getData(
"text/html"),i=e.getData&&e.getData("text/plain");let s=!1;n&&insertHtmlIntoRichPasteEditor(n)&&(t=!0,
s=!0),!s&&i&&(insertTextIntoRichPasteEditor(i),t=!0);const r=Array.from(e.items||[]).filter(l=>l&&l.
kind==="file").map(l=>l.getAsFile()).filter(l=>l&&l.type&&l.type.startsWith("image/"));if(!s&&r.length)
for(const l of r)try{await insertClipboardImageBlob(l,l.name||"clipboard-image")&&(t=!0)}catch{}return t},
"ingestRichPasteClipboardData"),buildRichPastePdfFilename=o(()=>{const e=new Date,t=o(n=>String(n).padStart(
2,"0"),"pad");return`clipboard_rich_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(
e.getMinutes())}${t(e.getSeconds())}.pdf`},"buildRichPastePdfFilename"),getRichPasteProgressElements=o(
()=>({container:get("rich-paste-progress-container"),bar:get("rich-paste-progress-bar"),text:get("ri\
ch-paste-progress-text")}),"getRichPasteProgressElements"),setRichPasteProgress=o((e,t=null)=>{const{
container:n,bar:i,text:s}=getRichPasteProgressElements(),a=Math.max(0,Math.min(100,Number(e)||0));if(n&&
(n.classList.remove("hidden"),n.style.setProperty("display","block","important")),i&&(i.style.width=
`${a}%`,i.style.transform="none"),s&&(s.textContent=`${Math.round(a)}%`),t&&n){const r=n.querySelector(
".text-amber-400");r&&(r.innerHTML=`<i class="fas fa-spinner fa-spin"></i> ${escapeHtml(t)}`)}},"set\
RichPasteProgress"),hideRichPasteProgress=o(()=>{const{container:e,bar:t}=getRichPasteProgressElements();
t&&(t.style.transform="scaleX(0)"),e&&(e.classList.add("hidden"),e.style.display="none")},"hideRichP\
asteProgress"),inferRichPasteTitle=o(()=>{const e=getRichPasteEditor();if(!e)return"Clipboard Export";
const t=e.querySelector("h1, h2, h3, h4, h5, h6");if(t&&t.textContent&&t.textContent.trim())return t.
textContent.trim().slice(0,48);const n=(e.innerText||"").trim().replace(/\s+/g," ");return n?n.slice(
0,48):"Clipboard Export"},"inferRichPasteTitle"),waitForRichPasteMedia=o(async(e,t=2500)=>{if(!e)return;
const n=new Promise(s=>setTimeout(s,Math.max(0,t))),i=Promise.all(Array.from(e.querySelectorAll("img")||
[]).map(s=>!s||s.complete?Promise.resolve():new Promise(a=>{let r=!1;const l=o(()=>{r||(r=!0,a())},"\
finish");s.addEventListener("load",l,{once:!0}),s.addEventListener("error",l,{once:!0}),setTimeout(l,
Math.max(250,Math.min(t,2e3)))})));if(await Promise.race([i,n]),document.fonts&&document.fonts.ready)
try{await Promise.race([document.fonts.ready,n])}catch{}},"waitForRichPasteMedia"),normalizeRichPastePdfText=o(
e=>String(e||"").replace(/\u00a0/g," ").replace(/\r\n?/g,`
`).replace(/[ \t\f\v]+/g," ").replace(/\n[ \t]+/g,`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).trim(),"normalizeRichPastePdfText"),normalizeRichPastePdfCodeText=o(e=>String(e||"").replace(/\u00a0/g,
" ").replace(/\r\n?/g,`
`),"normalizeRichPastePdfCodeText"),collectRichPasteInlineSegments=o((e,t={})=>{if(!e)return[];const n=t.
allowLinks!==!1,i=[],s=o((a,r)=>{if(!a)return;if(a.nodeType===Node.TEXT_NODE){const p=a.textContent||
"";p&&i.push(Object.assign({},r,{text:p}));return}if(a.nodeType!==Node.ELEMENT_NODE)return;const l=String(
a.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(l))return;if(l==="br"){i.push({text:`
`});return}const d=Object.assign({},r);["b","strong"].includes(l)&&(d.bold=!0),["i","em"].includes(l)&&
(d.italic=!0),l==="a"&&n&&(d.link=String(a.getAttribute("href")||"").trim()),l==="code"&&(d.monospace=
!0),Array.from(a.childNodes||[]).forEach(p=>s(p,d))},"walk");return s(e,{bold:!!t.bold,italic:!!t.italic}),
i},"collectRichPasteInlineSegments"),collectRichPasteInlineText=o((e,t={})=>collectRichPasteInlineSegments(
e,t).map(i=>i.text).join(""),"collectRichPasteInlineText"),collectRichPasteTableRows=o(e=>{const t=[];
return Array.from(e.querySelectorAll("tr")||[]).forEach(n=>{n&&n.closest&&n.closest("table")===e&&t.
push(n)}),t},"collectRichPasteTableRows"),makeRichPasteTableMarkdown=o(e=>{const t=e&&e.querySelector?
e.querySelector("caption"):null,n=t?normalizeRichPastePdfText(collectRichPasteInlineText(t)):"",i=collectRichPasteTableRows(
e).map(d=>Array.from(d.children||[]).filter(g=>{const h=String(g.tagName||"").toLowerCase();return h===
"th"||h==="td"}).map(g=>normalizeRichPastePdfText(collectRichPasteInlineText(g))||" ")).filter(d=>d.
length);if(!i.length)return n||"[table]";const s=i.reduce((d,p)=>Math.max(d,p.length),0),a=i.map(d=>{
const p=d.slice(0,s);for(;p.length<s;)p.push(" ");return p}),r=`| ${Array(s).fill("---").join(" | ")}\
 |`,l=[];n&&(l.push(`Table: ${n}`),l.push("")),l.push(`| ${a[0].join(" | ")} |`),l.push(r);for(let d=1;d<
a.length;d+=1)l.push(`| ${a[d].join(" | ")} |`);return l.join(`
`)},"makeRichPasteTableMarkdown"),collectRichPasteListBlocks=o((e,t=!1,n=0)=>{const i=[],s=Array.from(
e.children||[]).filter(r=>String(r.tagName||"").toLowerCase()==="li");let a=1;return s.forEach(r=>{const l=r.
cloneNode(!0);Array.from(l.querySelectorAll("ul,ol")||[]).forEach(p=>{try{p.remove()}catch{}});const d=collectRichPasteInlineSegments(
l);d.length>0&&i.push({type:"list_item",ordered:t,depth:n,index:a,segments:d}),Array.from(r.children||
[]).forEach(p=>{const g=String(p.tagName||"").toLowerCase();(g==="ul"||g==="ol")&&i.push(...collectRichPasteListBlocks(
p,g==="ol",n+1))}),a+=1}),i},"collectRichPasteListBlocks"),collectRichPastePdfBlocks=o((e,t=0)=>{const n=[];
if(!e)return n;let i=[];const s=o(()=>{i.length!==0&&(n.push({type:"paragraph",segments:[...i]}),i=[])},
"flushBuffer");return Array.from(e.childNodes||[]).forEach(a=>{if(!a)return;if(a.nodeType===Node.TEXT_NODE){
const p=(a.textContent||"").replace(/\u00a0/g," ");p&&i.push({text:p});return}if(a.nodeType!==Node.ELEMENT_NODE)
return;const r=String(a.tagName||"").toLowerCase();if(RICH_PASTE_NOISE_TAGS.has(r))return;if(r==="br"){
i.push({text:`
`});return}if(/^h[1-6]$/.test(r)){s();const p=collectRichPasteInlineSegments(a);p.length>0&&n.push({
type:"heading",level:Number(r.slice(1))||1,segments:p});return}if(r==="p"){s();const p=collectRichPasteInlineSegments(
a);p.length>0&&n.push({type:"paragraph",segments:p});return}if(r==="blockquote"){s();const p=collectRichPasteInlineSegments(
a,{italic:!0});p.length>0&&n.push({type:"blockquote",segments:p});return}if(r==="pre"){s();const p=normalizeRichPastePdfCodeText(
a.innerText||a.textContent||"");p.trim()&&n.push({type:"code",text:p});return}if(r==="table"){s();const p=makeRichPasteTableMarkdown(
a);p&&n.push({type:"table",text:p});return}if(r==="ul"||r==="ol"){s(),n.push(...collectRichPasteListBlocks(
a,r==="ol",t));return}if(r==="hr"){s(),n.push({type:"hr"});return}if(r==="figure"){s();const p=a.querySelector(
"img");p&&n.push({type:"image",src:String(p.getAttribute("src")||"").trim(),alt:String(p.getAttribute(
"alt")||p.getAttribute("title")||"").trim(),title:String(p.getAttribute("title")||"").trim()});const g=a.
querySelector("figcaption");if(g){const h=collectRichPasteInlineSegments(g);h.length>0&&n.push({type:"\
paragraph",segments:h})}return}if(r==="img"){s(),n.push({type:"image",src:String(a.getAttribute("src")||
"").trim(),alt:String(a.getAttribute("alt")||a.getAttribute("title")||"").trim(),title:String(a.getAttribute(
"title")||"").trim()});return}if(r==="li"){s(),n.push(...collectRichPasteListBlocks(a,!1,t));return}
if(Array.from(a.children||[]).some(p=>{const g=String(p.tagName||"").toLowerCase();return/^h[1-6]$/.
test(g)||["p","div","section","article","main","blockquote","pre","table","ul","ol","hr","figure","i\
mg","li"].includes(g)})&&["div","section","article","main","figure"].includes(r)){s(),n.push(...collectRichPastePdfBlocks(
a,t+1));return}const d=collectRichPasteInlineSegments(a);d.length>0&&i.push(...d)}),s(),n},"collectR\
ichPastePdfBlocks"),detectImageMimeType=o(e=>{const t=String(e||"").match(/^data:(image\/[a-z0-9.+-]+);/i);
return t?t[1].toLowerCase():"image/png"},"detectImageMimeType"),loadRichPasteImageData=o(async(e,t=3e3)=>{
const n=String(e||"").trim();if(!n)return null;if(n.startsWith("data:image/"))return{dataUrl:n,mimeType:detectImageMimeType(
n)};let i=null;try{i=new URL(n,window.location.href)}catch{return null}if(!(i.origin===window.location.
origin))return null;const a=(async()=>{try{const r=await fetch(i.toString(),{credentials:"same-origi\
n",cache:"force-cache"});if(!r.ok)return null;const l=await r.blob(),d=await blobToDataUrl(l);return{
dataUrl:d,mimeType:l.type||detectImageMimeType(d)}}catch{return null}})();return await Promise.race(
[a,new Promise(r=>setTimeout(()=>r(null),Math.max(250,t)))])},"loadRichPasteImageData"),buildRichPastePreviewHtml=o(
(e="preview")=>{const t=getRichPasteEditor();if(!t)return"";const n=inferRichPasteTitle(),i=new Date().
toLocaleString("ja-JP"),s=sanitizeRichPasteHtml(t.innerHTML||""),a=detectRichPasteTheme(s),r=normalizeRichPastePrintHtml(
s),l=e==="pdf";return`<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(n)} - Preview</title>
  <style>
        :root {
          color-scheme: ${a.mode};
          --rp-background: ${a.background};
          --rp-foreground: ${a.foreground};
          --rp-muted: ${a.muted};
          --rp-border: ${a.border};
          --rp-surface: ${a.surface};
          --rp-quote: ${a.quote};
          --rp-link: ${a.link};
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
</html>`},"buildRichPastePreviewHtml"),openSandboxedHtmlTab=o(e=>{const n=`<!doctype html><html><hea\
d><meta charset="utf-8"><meta name="referrer" content="no-referrer"><style>html,body,iframe{width:10\
0%;height:100%;margin:0;border:0;background:#fff}body{overflow:hidden}</style></head><body><iframe i\
d="preview" sandbox="allow-scripts allow-forms allow-modals allow-popups" referrerpolicy="no-referre\
r"></iframe><script>document.getElementById('preview').srcdoc=${JSON.stringify(String(e||"")).replace(
/</g,"\\u003c").replace(/\u2028/g,"\\u2028").replace(/\u2029/g,"\\u2029")};<\/script></body></html>`,
i=new Blob([n],{type:"text/html;charset=utf-8"}),s=URL.createObjectURL(i);return window.open(s,"_bla\
nk","noopener,noreferrer")?(setTimeout(()=>URL.revokeObjectURL(s),6e4),!0):(URL.revokeObjectURL(s),!1)},
"openSandboxedHtmlTab"),openRichPastePreviewTab=o(()=>{const e=buildRichPastePreviewHtml("preview");
if(!e){showToast("\u78BA\u8A8D\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093","warning",
!0);return}openSandboxedHtmlTab(e)||showToast("\u5225\u30BF\u30D6\u306E\u8868\u793A\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},"openRichPastePreviewTab"),renderRichPastePdfBlob=o(async()=>{const e=get("rich-paste-p\
rogress-container"),t=get("rich-paste-progress-bar"),n=get("rich-paste-progress-text"),i=o(v=>{const b=Math.
max(0,Math.min(100,Number(v)||0));t&&(t.style.width="100%",t.style.transformOrigin="left center",(!t.
style.transition||t.style.transition.indexOf("transform")===-1)&&(t.style.transition="transform 0.45\
s cubic-bezier(0.22, 1, 0.36, 1)"),t.style.transform=`scaleX(${b/100})`,t.style.willChange="transfor\
m"),n&&(n.innerText=`${Math.round(b)}%`)},"updateProgress");e&&(e.classList.remove("hidden"),e.style.
setProperty("display","block","important")),t&&(t.style.transition="none",t.style.width="100%",t.style.
transformOrigin="left center",t.style.transform="scaleX(0)",t.offsetHeight,t.style.transition="trans\
form 0.45s cubic-bezier(0.22, 1, 0.36, 1)"),i(0),await new Promise(v=>requestAnimationFrame(()=>setTimeout(
v,150)));const s=getRichPasteEditor();if(!s)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093");
const a=inferRichPasteTitle(),r=sanitizeRichPasteHtml(s.innerHTML||""),l=detectRichPasteTheme(r),d=normalizeRichPastePrintHtml(
r);await ensurePdfLibraries();const p=window.jspdf&&window.jspdf.jsPDF?window.jspdf.jsPDF:null;if(!p)
throw new Error("jsPDF \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
const g=window.html2canvas;if(typeof g!="function")throw new Error("html2canvas \u30E9\u30A4\u30D6\u30E9\u30EA\u304C\u8AAD\u307F\u8FBC\u307E\u308C\u3066\u3044\u307E\u305B\u3093");
i(5);const h=document.createElement("div");h.style.position="absolute",h.style.left="-10000px",h.style.
top="0",h.style.width="794px",h.style.background=l.background,h.style.color=l.foreground,h.style.boxSizing=
"border-box",h.style.fontFamily='"Noto Sans JP", "Segoe UI", "Helvetica Neue", Arial, sans-serif',h.
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
                    <div class="pdf-title">${escapeHtml(a)}</div>
                    <div class="pdf-meta">Created at: ${new Date().toLocaleString("ja-JP")}</div>
                    <div class="pdf-content">${d}</div>
                </div>
            `,document.body.appendChild(h),await waitForRichPasteMedia(h,4e3),i(15);try{const v=new p(
{unit:"mm",format:"a4",orientation:"portrait",compress:!0}),b=v.internal.pageSize.getWidth(),w=v.internal.
pageSize.getHeight(),x=794,L=Math.floor(w/b*x),T=h.scrollHeight||h.offsetHeight;let $=0,j=!0;const J=Math.
ceil(T/L);let X=0;for(;$<T;){if(richPasteAbortController&&richPasteAbortController.signal.aborted)throw new DOMException(
"Aborted","AbortError");const P=Math.min(L,T-$),Y=(await new Promise((pe,ke)=>{const te=setTimeout(()=>ke(
new Error("PDF chunk rendering timed out")),12e4);g(h,{scale:1,useCORS:!0,allowTaint:!1,backgroundColor:l.
background,logging:!1,imageTimeout:5e3,x:0,y:$,width:x,height:P,windowWidth:x,scrollX:0,scrollY:0,signal:richPasteAbortController?
richPasteAbortController.signal:void 0,onclone:o(se=>{prepareRichPastePdfClone(se,l);const W=se.querySelector(
".pdf-root-wrapper");W&&(W.style.position="relative",W.style.left="0",W.style.top="0")},"onclone")}).
then(se=>{clearTimeout(te),pe(se)}).catch(se=>{clearTimeout(te),ke(se)})})).toDataURL("image/jpeg",.95),
de=v.getImageProperties(Y),oe=Math.min(w,de.height*b/de.width);j||v.addPage(),v.addImage(Y,"JPEG",0,
0,b,oe),j=!1,$+=P,X++;const xe=Math.min(100,15+Math.round(X/J*85));i(xe),await new Promise(pe=>setTimeout(
pe,100))}return i(100),{blob:v.output("blob"),fileName:buildRichPastePdfFilename()}}finally{e&&(e.classList.
add("hidden"),e.style.display="none"),h&&h.parentNode&&document.body.removeChild(h)}},"renderRichPas\
tePdfBlob"),createRichPastePdfBlob=o(async()=>await renderRichPastePdfBlob(),"createRichPastePdfBlob"),
buildRichPasteServerPayload=o(()=>{const e=getRichPasteEditor();if(!e)throw new Error("PDF\u5316\u3059\u308B\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\
\u3093");const t=String(e.innerHTML||"").trim(),n=String(e.textContent||"").trim(),i=t||(n?`<p>${escapeHtml(
n).replace(/\n/g,"<br/>")}</p>`:"");return{title:inferRichPasteTitle(),html:i,created_at:new Date().
toLocaleString("ja-JP"),theme:detectRichPasteTheme(sanitizeRichPasteHtml(i))}},"buildRichPasteServer\
Payload"),attachRichPastePdfAndSend=o(async(e,t,n,i)=>{const s=new Set(collectAttachmentItemsForSend().
map(g=>g.path)),a=new File([e],t,{type:"application/pdf",lastModified:Date.now()}),r=get("prompt-inp\
ut");if(r&&(r.value=n),await handleFiles([a],{openModal:!1}),!collectAttachmentItemsForSend().map(g=>g.
path).some(g=>!s.has(g)))throw r&&(r.value=i),new Error("PDF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const p=sendMessage();clearRichPasteEditor(!0),window.closeRichPasteModal(),showToast("PDF\u3092\u6DFB\u4ED8\u3057\u3066\u9001\u4FE1\u3092\u958B\u59CB\
\u3057\u307E\u3057\u305F","success"),p&&typeof p.catch=="function"&&p.catch(()=>{})},"attachRichPast\
ePdfAndSend"),openRichPasteModal=o(async()=>{await ensureUserSettingsSnapshot(),showModal("rich-past\
e-modal"),location.pathname!=="/paste"&&history.pushState({modal:"paste"},"","/paste");const e=getRichPastePrompt();
e&&(richPastePromptPreferenceSyncing=!0,e.value=getRichPasteEffectivePrompt(userSettingsSnapshot),richPastePromptPreferenceSyncing=
!1),updateRichPasteStatus(),setTimeout(()=>focusRichPasteEditor(),80)},"openRichPasteModal");window.
closeRichPasteModal=(e=!1)=>{hideModal("rich-paste-modal"),!e&&location.pathname==="/paste"&&history.
back()};const sendRichPasteToModel=o(async(e={})=>{const t=!!(e&&e.serverSide);if(abortController||richPasteAbortController){
showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u307E\u305F\u306FPDF\u5909\u63DB\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const n=getRichPasteEditor(),i=getRichPastePrompt(),s=get(t?"rich-paste-send-se\
rver-btn":"rich-paste-send-btn"),a=get("rich-paste-cancel-btn");if(!n||!n.innerText||!n.innerText.trim()){
showToast("\u8CBC\u308A\u4ED8\u3051\u308B\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}richPasteAbortController=new AbortController,a&&(a.onclick=()=>{richPasteAbortController&&
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
const g=await apiFetch(RICH_PASTE_PDF_SERVER_ROUTE,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(p),signal:richPasteAbortController.signal});if(setRichPasteProgress(60,
"PDF\u3092\u53D7\u4FE1\u4E2D..."),!g.ok){let w="";try{const x=await g.json();w=x&&(x.message||x.error)?
String(x.message||x.error):""}catch{try{w=await g.text()}catch{w=""}}throw w==="missing_html"?new Error(
"\u30B5\u30FC\u30D0\u30FC\u3078\u9001\u308BHTML\u304C\u7A7A\u3067\u3059\u3002\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u5185\u5BB9\u306E\u53D6\u308A\u8FBC\u307F\u3092\u5148\u306B\u884C\u3063\u3066\u304F\u3060\u3055\u3044"):
new Error(w?`\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${w}`:
"\u30B5\u30FC\u30D0\u30FCPDF\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}setRichPasteProgress(
75,"PDF\u3092\u6DFB\u4ED8\u4E2D...");const h=await g.blob(),v=g.headers.get("X-Rich-Paste-Filename")||
buildRichPastePdfFilename();!!(get("rich-paste-download-only")&&get("rich-paste-download-only").checked)?
(setRichPasteProgress(90,"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D..."),downloadBlob(h,v),showToast(
"PDF\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F","success"),hideModal("rich-p\
aste-modal",{skipConfirm:!0})):await attachRichPastePdfAndSend(h,v,r,l),setRichPasteProgress(100,"\u5B8C\u4E86"),
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
const refreshCsrfToken=o(async()=>csrfRefreshPromise||(csrfRefreshPromise=(async()=>{const e=await fetch(
"/api/csrf_token",{method:"GET",credentials:"include",cache:"no-store",headers:{Accept:"application/\
json"}});if(!e.ok)return!1;const t=await e.json().catch(()=>({})),n=t&&typeof t.csrf_token=="string"?
t.csrf_token:"";if(!n)return!1;csrfToken=n;const i=document.querySelector('meta[name="csrf-token"]');
return i&&i.setAttribute("content",n),!0})().catch(()=>!1).finally(()=>{csrfRefreshPromise=null}),csrfRefreshPromise),
"refreshCsrfToken"),apiFetch=o(async(e,t={})=>{const n=(t.method||"GET").toUpperCase(),i=Object.assign(
{},t.headers||{}),s=!["GET","HEAD","OPTIONS"].includes(n);s&&(i["X-CSRF-Token"]=csrfToken);const a=t.
credentials||"include";let r=await fetch(e,Object.assign({},t,{headers:i,credentials:a}));if(s&&(r.status===
403||r.status===404)){let l=null;try{l=await r.clone().json()}catch{}const d=l&&l.error;if(d==="acco\
unt_locked")return!isAdminUser&&!document.getElementById("bot-lock-overlay")&&showBotLockOverlay(l.message||
"\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u3066\u3044\u307E\u3059\u3002",
l.remaining_seconds),r;if(d==="banned"||d==="turnstile_failed"||d==="rate_limit")return r;if(d==="tu\
rnstile_required"&&isBotDetectionActive())return botDetectionVerified=!1,await Promise.race([runBotDetectionGate(),
new Promise(h=>setTimeout(()=>h(!1),3e4))])&&(i["X-CSRF-Token"]=csrfToken,r=await fetch(e,Object.assign(
{},t,{headers:i,credentials:a}))),r;await refreshCsrfToken()&&(i["X-CSRF-Token"]=csrfToken,r=await fetch(
e,Object.assign({},t,{headers:i,credentials:a})))}return r},"apiFetch"),manualSpinnerRequestOptions=o(
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
1"?Promise.resolve(n):new Promise((i,s)=>{n.addEventListener("load",()=>i(n),{once:!0}),n.addEventListener(
"error",s,{once:!0})}):new Promise((i,s)=>{const a=document.createElement("script");t&&(a.id=t),a.src=
e,a.async=!0,a.onload=()=>{a.dataset.loaded="1",i(a)},a.onerror=s,document.head.appendChild(a)})}o(loadScriptOnce,
"loadScriptOnce");function loadStylesheetOnce(e,t){const n=t?document.getElementById(t):null;if(n)return Promise.
resolve(n);const i=Array.from(document.querySelectorAll('link[rel="stylesheet"]')).find(s=>s.href===
e);return i?Promise.resolve(i):new Promise((s,a)=>{const r=document.createElement("link");t&&(r.id=t),
r.rel="stylesheet",r.href=e,r.onload=()=>s(r),r.onerror=a,document.head.appendChild(r)})}o(loadStylesheetOnce,
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
n=[],i=o(p=>{const g=`@@MATHJAX_BLOCK_${n.length}@@`;return n.push(p),g},"stash"),s=[],a=/(^|\n)([ \t]*)(`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:\n\2\3[ \t]*(?:\n|$)|$)/g;
let r=0,l;for(;(l=a.exec(t))!==null;){const p=l.index;p>r&&s.push({type:"text",value:t.slice(r,p)}),
s.push({type:"code",value:l[0]}),r=p+l[0].length}return r<t.length&&s.push({type:"text",value:t.slice(
r)}),s.length||s.push({type:"text",value:t}),{text:s.map(p=>{if(p.type==="code")return p.value;let g=p.
value;return g=g.replace(/\$\$([\s\S]+?)\$\$/g,i),g=g.replace(/\\\(([\s\S]+?)\\\)/g,i),g=g.replace(/\\\[([\s\S]+?)\\\]/g,
i),g=g.replace(/\\begin\{([a-zA-Z*]+)\}([\s\S]+?)\\end\{\1\}/g,i),g=g.replace(/(?<!\$)\$(?!\$)([^\s$](?:(?:[^$\n\\]|\\.)*?[^\s$])?)\$(?!\$)/g,
i),g}).join(""),blocks:n}}o(protectMathSegments,"protectMathSegments");function getStreamMathSegmentKey(e,t){
const n=String(t||"");let i=2166136261;for(let s=0;s<n.length;s++)i^=n.charCodeAt(s),i=Math.imul(i,16777619);
return`${e}-${n.length}-${(i>>>0).toString(16)}`}o(getStreamMathSegmentKey,"getStreamMathSegmentKey");
function restoreMathSegments(e,t,n={}){return!t||!t.length?String(e||""):String(e||"").replace(/@@MATHJAX_BLOCK_(\d+)@@/g,
(i,s)=>{const a=t[Number(s)];if(a==null)return"";const r=String(a).replace(/&/g,"&amp;").replace(/</g,
"&lt;").replace(/>/g,"&gt;");return n.streamMathSegments?`<span class="stream-math-segment mathjax_p\
rocess" data-stream-math-key="${getStreamMathSegmentKey(Number(s),a)}">${r}</span>`:r})}o(restoreMathSegments,
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
n.forEach(s=>s.removeAttribute("data-stream-math-state"))}}}).catch(()=>{t.forEach(n=>n.removeAttribute(
"data-stream-math-state"))}))}o(queueIncrementalMathTypeset,"queueIncrementalMathTypeset");function queueHighlight(e,t="",n={}){
lowBandwidthMode&&!n.force||!e||!maybeNeedsHighlight(t,e)||activeStreamingBubbleId&&e.closest(`#${activeStreamingBubbleId}`)||
ensureHighlightLoaded().then(()=>{window.hljs&&e.querySelectorAll("pre code").forEach(i=>{if(!(i.getAttribute(
"data-highlighted")==="true"&&!n.force))try{window.hljs.highlightElement(i)}catch{}})}).catch(()=>{})}
o(queueHighlight,"queueHighlight");function getNetworkConnectionInfo(){return navigator.connection||
navigator.mozConnection||navigator.webkitConnection||null}o(getNetworkConnectionInfo,"getNetworkConn\
ectionInfo");function detectLowBandwidthModeAuto(){const e=getNetworkConnectionInfo();if(!e)return{enabled:!1,
reason:""};const t=!!e.saveData,n=String(e.effectiveType||"").toLowerCase(),i=Number(e.downlink||0),
s=n==="slow-2g"||n==="2g"||n==="3g",a=Number.isFinite(i)&&i>0&&i<1.3,r=t||s||a,l=[];return t&&l.push(
"\u30C7\u30FC\u30BF\u7BC0\u7D04"),n&&l.push(`\u56DE\u7DDA:${n}`),a&&l.push(`\u4E0B\u308A:${i}Mbps`),
{enabled:r,reason:l.join(" / ")}}o(detectLowBandwidthModeAuto,"detectLowBandwidthModeAuto");function normalizeLowBandwidthModePreference(e){
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
"ON":"OFF",s=lowBandwidthModeReason?` (${lowBandwidthModeReason})`:"";if(e&&(e.setAttribute("title",
`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9 ${i} / ${n}${s}`),e.setAttribute("aria-pressed",lowBandwidthMode?
"true":"false"),lowBandwidthMode?mergeBtnClasses(e,["text-amber-200","bg-amber-900/30","border","bor\
der-amber-600/40"],["text-gray-400"]):mergeBtnClasses(e,["text-gray-400"],["text-amber-200","bg-ambe\
r-900/30","border","border-amber-600/40"])),t)if(lowBandwidthMode){t.classList.remove("hidden");const a=lowBandwidthModePreference===
"auto"?" (\u81EA\u52D5)":" (\u624B\u52D5)";t.innerHTML=`<i class="fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9${a}${lowBandwidthModeReason?
`: ${escapeHtml(lowBandwidthModeReason)}`:""}`}else t.classList.add("hidden"),t.innerHTML='<i class=\
"fas fa-wifi mr-1"></i>\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9'}o(updateLowBandwidthModeUi,"updat\
eLowBandwidthModeUi");function refreshDecorationsForVisibleChat(){const e=get("chat-container");e&&(queueHighlight(
e,e.textContent||"",{force:!0}),queueMathTypeset(e,e.textContent||"",{force:!0}))}o(refreshDecorationsForVisibleChat,
"refreshDecorationsForVisibleChat");function applyLowBandwidthModeState(e,t={}){const n=lowBandwidthMode;
if(lowBandwidthMode=!!e,updateLowBandwidthModeUi(),n&&!lowBandwidthMode&&refreshDecorationsForVisibleChat(),
t.notify){const i=lowBandwidthModePreference==="auto"?"\u81EA\u52D5":"\u624B\u52D5",s=lowBandwidthModeReason?
` (${lowBandwidthModeReason})`:"";showToast(`\u4F4E\u901F\u56DE\u7DDA\u30E2\u30FC\u30C9\u3092${lowBandwidthMode?
"ON":"OFF"}\u306B\u3057\u307E\u3057\u305F [${i}]${s}`,"info",!1)}}o(applyLowBandwidthModeState,"appl\
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
test(t)&&!t.startsWith("data:")&&!t.startsWith("blob:")&&(n="https://"+t);try{const s=(new URL(n,"ht\
tps://example.com").hostname||"").toLowerCase();return BLOCKED_SCRIPT_HOSTS.some(a=>s===a||s.endsWith(
"."+a))}catch{return/polyfill\.io/i.test(t)}}o(isBlockedScriptSrc,"isBlockedScriptSrc");function isPasswordPromptingScript(e){
if(!e)return!1;const t=String(e),n=t.toLowerCase();return!!(/prompt\s*\(\s*(['"`]).{0,40}(pass|pwd|password|secret|credential|認証|パスワード|login|pin|暗証)/i.
test(t)||/confirm\s*\(\s*(['"`]).{0,40}(pass|password|削除|重要|delete all|全削除)/i.test(t)||
/(type\s*=\s*['"]?password|name\s*=\s*['"]?password|password.*input|input.*password|getPassword|promptForPass)/i.
test(n)||/prompt\s*\(/.test(t)&&/(fetch\(|XMLHttpRequest|\.send\(|navigator\.sendBeacon|location\s*\.\s*(href|replace)|document\.cookie\s*=)/i.
test(t))}o(isPasswordPromptingScript,"isPasswordPromptingScript");function detectBlockedScriptsInCode(e){
if(!e)return!1;const t=String(e),n=/<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi;let i;for(;(i=n.exec(
t))!==null;)if(isBlockedScriptSrc(i[1]))return!0;const s=/<script\b(?![^>]*\bsrc\s*=)[^>]*>([\s\S]*?)<\/script>/gi;
for(;(i=s.exec(t))!==null;)if(isPasswordPromptingScript(i[1]))return!0;return!!(/["'`]https?:\/\/[^"'`\s]*polyfill\.io/i.
test(t)||/src\s*=\s*["'`][^"'`]*polyfill\.io/i.test(t))}o(detectBlockedScriptsInCode,"detectBlockedS\
criptsInCode");function sanitizeHtmlForPreview(e){if(!e)return"";const t=detectBlockedScriptsInCode(
e);let n=String(e);try{const s=new DOMParser().parseFromString(n,"text/html");let a=!1;s.querySelectorAll(
"script").forEach(l=>{const d=l.getAttribute("src")||"";let p=!1;if(d&&isBlockedScriptSrc(d)){const g=s.
createElement("div");g.setAttribute("data-blocked-script","true"),g.style.cssText="background:#fee2e\
2;border:1px solid #ef4444;color:#991b1b;padding:6px 10px;border-radius:6px;font-size:12px;margin:6p\
x 0;font-family:system-ui;";const h=d.length>70?d.slice(0,67)+"...":d;g.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: "+
h+" \uFF08polyfill.io \u306A\u3069\u306E\u5371\u967A\u30C9\u30E1\u30A4\u30F3\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u7121\u52B9\u5316\u3055\u308C\u307E\u3059\uFF09",
l.parentNode&&l.parentNode.replaceChild(g,l),a=!0,p=!0}else if(!d){const g=l.textContent||"";if(isPasswordPromptingScript(
g)){const h=s.createElement("div");h.setAttribute("data-blocked-script","true"),h.style.cssText="bac\
kground:#fef3c7;border:1px solid #f59e0b;color:#92400e;padding:6px 10px;border-radius:6px;font-size:\
12px;margin:6px 0;font-family:system-ui;",h.textContent="\u26A0 \u30D6\u30ED\u30C3\u30AF\u6E08\u307F: \u30D1\u30B9\u30EF\u30FC\u30C9\u5165\u529B\u8981\u6C42\u306A\u3069\u306E\u7591\u308F\u3057\u3044\u30A4\u30F3\u30E9\u30A4\u30F3\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\
\u305F",l.parentNode&&l.parentNode.replaceChild(h,l),a=!0,p=!0}}}),s.querySelectorAll('a[href^="java\
script:" i], area[href^="javascript:" i]').forEach(l=>{l.setAttribute("href","#"),l.setAttribute("ti\
tle",(l.getAttribute("title")||"")+" [javascript: disabled in preview]")});const r=s.head||s.querySelector(
"head");if(r&&!r.querySelector("base")){const l=s.createElement("base");l.setAttribute("href",`${window.
location.origin}/`),r.insertBefore(l,r.firstChild)}if(t||a){const l=s.body||s.documentElement;if(l){
const d=s.createElement("div");d.style.cssText="position:sticky;top:0;left:0;right:0;z-index:2147483\
647;background:#7f1d1d;color:#fff;padding:8px 12px;text-align:center;font-size:12px;font-family:syst\
em-ui;border-bottom:1px solid #b91c1c;",d.innerHTML="\u26A0 <strong>\u5B89\u5168\u30D7\u30EC\u30D3\u30E5\u30FC</strong>: polyfill.io \u306A\u3069\u306E\u5371\u967A\u306A\u30B9\
\u30AF\u30EA\u30D7\u30C8\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002\u5B9F\u884C\u306F\u81EA\u5DF1\u8CAC\u4EFB\u3067\u3002",
l.firstChild?l.insertBefore(d,l.firstChild):l.appendChild(d)}}n=`<!DOCTYPE html>
`+(s.documentElement?s.documentElement.outerHTML:n)}catch{n=n.replace(/<script\b([^>]*\bsrc\s*=\s*["']?[^"'\s>]*polyfill\.io[^"'\s>]*)["']?[^>]*>[\s\S]*?<\/script>/gi,
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
".content-area.skeleton-pending");if(!i)return!1;let s=i.querySelector(".skeleton-status");s||(s=document.
createElement("div"),s.className="skeleton-status",i.appendChild(s));const a=t==null?"":String(t),r=n==
null||n===""?"":String(n);return r?s.innerHTML=`${escapeHtml(a)}<span class="skeleton-status-sub">${escapeHtml(
r)}</span>`:s.textContent=a,!0}o(updatePendingSkeletonStatus,"updatePendingSkeletonStatus");function buildChatLoadingSkeletonHtml(){
return`<div class="chat-load-skeleton" role="status" aria-live="polite" aria-label="\u30C1\u30E3\u30C3\u30C8\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D">${[
{role:"user",widths:["62%","44%"]},{role:"ai",widths:["88%","76%","92%","58%"]},{role:"user",widths:[
"48%"]},{role:"ai",widths:["82%","70%","54%"]}].map((n,i)=>{const s=n.role==="user",a=s?"justify-end":
"justify-start",r=s?"message-bubble chat-load-skeleton-bubble chat-load-skeleton-user text-white p-4\
 rounded-2xl rounded-tr-none shadow-md relative":"message-bubble chat-load-skeleton-bubble chat-load\
-skeleton-ai bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative",l=n.widths.map(
(d,p)=>`<div class="skeleton-line" style="width:${d};animation-delay:${(i*.08+p*.06).toFixed(2)}s"><\
/div>`).join("");return`<div class="flex ${a} mb-4 chat-load-skeleton-row" style="animation-delay:${(i*
.07).toFixed(2)}s" aria-hidden="true"><div class="${r}"><div class="content-area pending-shimmer ske\
leton-pending chat-load-skeleton-body" data-skeleton-kind="text"><div class="skeleton-lines">${l}</d\
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
`),n=[];let i=null;for(const s of t){if(!i){const l=s.match(/^\s*(`{3,}|~{3,})(.*)$/);if(!l)continue;
const d=String(l[2]||"").trim();i={markerChar:l[1][0],markerLength:l[1].length,language:(d.split(/\s+/)[0]||
"text").replace(/^\{?\.?/,"").replace(/\}$/,"")||"text",buffer:[]};continue}const a=String(s||"").trim();
if(new RegExp(`^\\${i.markerChar}{${i.markerLength},}\\s*$`).test(a)){const l=i.buffer.join(`
`);l.trim()&&n.push({code:l,language:i.language,key:hashString(`prompt\\n${i.language}\\n${l}`),candidate_id:`\
prompt-${n.length+1}`,prompt_index:n.length,message_id:null,thread_id:currentThreadId?String(currentThreadId):
null,prompt_source:!0}),i=null;continue}i.buffer.push(s)}return n}o(extractPromptCodingTargets,"extr\
actPromptCodingTargets");function extractLatestPromptCodingTarget(e){const t=extractPromptCodingTargets(
e);return t.length?t[t.length-1]:null}o(extractLatestPromptCodingTarget,"extractLatestPromptCodingTa\
rget");function collectCodingCandidates(e){if(codingTargetSelection){const a=codingTargetSelection.thread_id;
if(!a||!currentThreadId||String(a)===String(currentThreadId))return[{...codingTargetSelection,candidate_id:"\
selected-1",source:"history",explicit:!0}];codingTargetSelection=null}const t=extractPromptCodingTargets(
e),n=new Set(t.map(a=>`${a.language}
${a.code}`)),i=get("chat-container"),s=[];return i&&Array.from(i.querySelectorAll(".message-group .c\
oding-target-btn")).forEach(a=>{const r=getCodingTargetFromButton(a);if(!r)return;const l=`${r.language}\

${r.code}`;n.has(l)||(n.add(l),s.push(r))}),s.slice(-20).forEach((a,r)=>{t.push({...a,candidate_id:`\
history-${r+1}`,source:"history",explicit:!1})}),t}o(collectCodingCandidates,"collectCodingCandidate\
s");function resolveCodingTarget(e=null){var s;const t=String(e===null?((s=get("prompt-input"))==null?
void 0:s.value)||"":e||"");if(codingTargetSelection){const a=codingTargetSelection.thread_id;if(!a||
!currentThreadId||String(a)===String(currentThreadId))return{...codingTargetSelection,explicit:!0};codingTargetSelection=
null}const n=extractLatestPromptCodingTarget(t);if(n)return{...n,explicit:!1};const i=findLatestCodingTarget();
return i?{...i,explicit:!1}:null}o(resolveCodingTarget,"resolveCodingTarget");function syncCodingTargetButtons(e=document){
if(!e||typeof e.querySelectorAll!="function")return;const t=codingTargetSelection?String(codingTargetSelection.
key||""):"";e.querySelectorAll(".coding-target-btn").forEach(n=>{const i=!!t&&String(n.getAttribute(
"data-code-key")||"")===t;n.classList.toggle("coding-target-active",i),n.setAttribute("aria-pressed",
i?"true":"false"),n.innerHTML=i?'<i class="fas fa-thumbtack"></i>':'<i class="fas fa-quote-right"></\
i>',n.title=i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A",
n.setAttribute("aria-label",i?"\u7DE8\u96C6\u5BFE\u8C61\u306B\u8A2D\u5B9A\u6E08\u307F":"\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A")})}
o(syncCodingTargetButtons,"syncCodingTargetButtons");function syncCodingModeUi(e=codingModeEnabled,t={}){
var d;if(codingModeEnabled=!!e,t.persist!==!1)try{localStorage.setItem(CODING_MODE_STORAGE_KEY,codingModeEnabled?
"true":"false")}catch{}const n=get("enable-coding-mode");n&&n.checked!==codingModeEnabled&&(n.checked=
codingModeEnabled);const i=get("coding-target-bar"),s=get("coding-target-text"),a=get("clear-coding-\
target-btn");i&&i.classList.toggle("visible",codingModeEnabled);const r=resolveCodingTarget(),l=codingTargetSelection?
[r].filter(Boolean):collectCodingCandidates(String(((d=get("prompt-input"))==null?void 0:d.value)||""));
if(codingModeEffective=codingModeEnabled&&l.length>0,s)if(codingTargetSelection&&r)s.textContent=`\u7DE8\u96C6\
\u5BFE\u8C61: ${r.language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`;else if(l.length>1){
const p=l.filter(h=>h.prompt_source).length,g=l.length-p;s.textContent=`\u30E2\u30C7\u30EB\u304C\u7DE8\u96C6\u5BFE\u8C61\u3092\u5224\u65AD: \u5165\u529B${p}\
\u4EF6 / \u5C65\u6B74${g}\u4EF6`}else r&&r.prompt_source?s.textContent=`\u5165\u529B\u4E2D: ${r.language||
"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:r?s.textContent=`\u81EA\u52D5\u9078\u629E: \u6700\u65B0\u306E ${r.
language||"text"} \u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF`:s.textContent="\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u751F\u6210\u5F8C\u306B\u81EA\u52D5\u6709\u52B9\u5316";
a&&a.classList.toggle("hidden",!codingTargetSelection),syncCodingTargetButtons()}o(syncCodingModeUi,
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
n.querySelector(`[data-coding-edit-index="${i}"]`))return;const s=n.querySelector(".coding-live-diff\
-list"),a=document.createElement("div");a.className="coding-live-diff-edit",i&&a.setAttribute("data-\
coding-edit-index",String(i));const r=Number(t.repair_attempt||0)>0?` \xB7 Auto repair ${Number(t.repair_attempt)}`:
"";a.innerHTML=`<div class="coding-live-diff-meta">Edit ${i} \xB7 ${escapeHtml(t.language||"text")}${r}\
</div><pre>${renderCodingDiffLines(t.diff)}</pre>`,s&&s.appendChild(a);const l=n.querySelector(".cod\
ing-live-diff-count"),d=n.querySelectorAll(".coding-live-diff-edit").length;l&&(l.textContent=`${d} \
edit${d===1?"":"s"}`),n.scrollIntoView({block:"nearest",behavior:"smooth"})}o(appendCodingLiveDiff,"\
appendCodingLiveDiff");function isHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
return n==="html"||n==="htm"||n==="xhtml"?!0:n?!1:/<!doctype\s+html/i.test(t||"")}o(isHtmlPreviewCandidate,
"isHtmlPreviewCandidate");function openHtmlCodePreview(e){if(!e)return;let t="";try{t=decodeURIComponent(
e)}catch{showToast("HTML\u30D7\u30EC\u30D3\u30E5\u30FC\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}detectBlockedScriptsInCode(t)&&showToast("\u26A0 \u5371\u967A\u306A\u5916\u90E8\u30B9\u30AF\u30EA\u30D7\u30C8\u3092\u691C\u77E5 (polyfill.io \u306A\u3069)\u3002\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\
\u306F\u30D6\u30ED\u30C3\u30AF\u3057\u3066\u958B\u304D\u307E\u3059\u3002","warning",!0);const i=sanitizeHtmlForPreview(
t);openSandboxedHtmlTab(i)}o(openHtmlCodePreview,"openHtmlCodePreview");function snapshotCodeCollapse(e){
if(!e)return[];const t=[];return e.querySelectorAll(".code-wrapper").forEach((n,i)=>{const s=String(
i),a=n.classList.contains("collapsed")||n.getAttribute("data-collapsed")==="true";t.push({key:s,collapsed:a})}),
t}o(snapshotCodeCollapse,"snapshotCodeCollapse");function applyCodeCollapse(e,t=[],n=!1){if(!e)return;
const i=new Map;t.forEach(s=>i.set(s.key,s.collapsed)),e.querySelectorAll(".code-wrapper").forEach((s,a)=>{
const r=String(a),l=i.has(r)?i.get(r):n;s.setAttribute("data-collapsed",l?"true":"false"),s.classList.
toggle("collapsed",!!l);const d=s.querySelector(".code-toggle");d&&(d.setAttribute("aria-expanded",l?
"false":"true"),d.innerHTML=l?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></\
i>',d.title=l?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",d.setAttribute("aria-label",l?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080"))})}o(applyCodeCollapse,"applyCodeCollapse");function snapshotCodeCollapseByMessage(e){
if(!e)return new Map;const t=new Map;return e.querySelectorAll(".message-group").forEach(n=>{const i=n.
getAttribute("id")||"";n.querySelectorAll(".code-wrapper").forEach((s,a)=>{const r=s.getAttribute("d\
ata-code-key")||String(a),l=s.classList.contains("collapsed")||s.getAttribute("data-collapsed")==="t\
rue";t.set(`${i}:${r}`,l)})}),t}o(snapshotCodeCollapseByMessage,"snapshotCodeCollapseByMessage");function applyCodeCollapseByMessage(e,t,n=!1){
e&&e.querySelectorAll(".message-group").forEach(i=>{const s=i.getAttribute("id")||"";i.querySelectorAll(
".code-wrapper").forEach((a,r)=>{const l=a.getAttribute("data-code-key")||String(r),d=`${s}:${l}`,p=t&&
t.has(d)?t.get(d):n;a.setAttribute("data-collapsed",p?"true":"false"),a.classList.toggle("collapsed",
!!p);const g=a.querySelector(".code-toggle");g&&(g.setAttribute("aria-expanded",p?"false":"true"),g.
innerHTML=p?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up"></i>',g.title=p?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080",g.setAttribute("aria-label",p?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080"))})})}
o(applyCodeCollapseByMessage,"applyCodeCollapseByMessage");function buildTokenTotals(e){const t={tokens_total:0,
tokens_in:0,tokens_out:0,tokens_content:0,tokens_thought:0};let n=!1,i=!1,s=!1,a=!1,r=!1;return(e||[]).
forEach(l=>{if(!l)return;let d=null;l.tokens!==null&&l.tokens!==void 0?d=Number(l.tokens||0):(l.tokens_in!==
null&&l.tokens_in!==void 0||l.tokens_out!==null&&l.tokens_out!==void 0)&&(d=Number(l.tokens_in||0)+Number(
l.tokens_out||0)),d!==null&&(t.tokens_total+=d,n=!0),l.tokens_in!==null&&l.tokens_in!==void 0&&(t.tokens_in+=
Number(l.tokens_in||0),i=!0),l.tokens_out!==null&&l.tokens_out!==void 0&&(t.tokens_out+=Number(l.tokens_out||
0),s=!0),l.tokens_content!==null&&l.tokens_content!==void 0&&(t.tokens_content+=Number(l.tokens_content||
0),a=!0),l.tokens_thought!==null&&l.tokens_thought!==void 0&&(t.tokens_thought+=Number(l.tokens_thought||
0),r=!0)}),{tokens_total:n?t.tokens_total:0,tokens_in:i?t.tokens_in:null,tokens_out:s?t.tokens_out:null,
tokens_content:a?t.tokens_content:null,tokens_thought:r?t.tokens_thought:null}}o(buildTokenTotals,"b\
uildTokenTotals");function updateTotalTokenBar(e,t=null,n=null){const i=get("total-token-bar"),s=get(
"total-token-count"),a=get("total-token-count-all-branches");if(!i||!s)return;const r=Number(e||0),l=Number(
n&&n.tokens_total||0);r>0||l>0?(i.classList.remove("hidden"),s.innerText=`Total: ${r} tokens`,t?(s.classList.
add("cursor-pointer","underline","decoration-dotted"),messageMeta.__total__={tokens_total:r,tokens_in:t.
tokens_in,tokens_out:t.tokens_out,tokens_content:t.tokens_content,tokens_thought:t.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation"},s.onclick=()=>openTokenDetail("__total__")):(s.classList.remove("\
cursor-pointer","underline","decoration-dotted"),s.onclick=null,delete messageMeta.__total__),a&&(n&&
l>0?(a.classList.remove("hidden"),a.classList.add("cursor-pointer","underline","decoration-dotted"),
a.innerText=`All branches: ${l} tokens`,messageMeta.__total_all_branches__={tokens_total:l,tokens_in:n.
tokens_in,tokens_out:n.tokens_out,tokens_content:n.tokens_content,tokens_thought:n.tokens_thought,is_encrypted:null,
role:"total",model:"Conversation (All branches)"},a.onclick=()=>openTokenDetail("__total_all_branche\
s__")):(a.classList.add("hidden"),a.classList.remove("cursor-pointer","underline","decoration-dotted"),
a.innerText="All branches: 0 tokens",a.onclick=null,delete messageMeta.__total_all_branches__))):(i.
classList.add("hidden"),s.innerText="Total: 0 tokens",s.classList.remove("cursor-pointer","underline",
"decoration-dotted"),s.onclick=null,delete messageMeta.__total__,a&&(a.classList.add("hidden"),a.classList.
remove("cursor-pointer","underline","decoration-dotted"),a.innerText="All branches: 0 tokens",a.onclick=
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
s=Array.isArray(n.image_urls)&&n.image_urls.length>0;if(!i&&!s){setPromptTokenEstimateText("");return}
if(e&&e.pending){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u4E2D...",
"text-gray-500");return}if(!e){setPromptTokenEstimateText("\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u3092\u8A08\u7B97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"text-red-300");return}if(!e.countable){setPromptTokenEstimateText("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u5165\u529B\u30C8\u30FC\u30AF\u30F3\u8868\u793A\u5BFE\u8C61\u5916\u3067\u3059",
"text-gray-500");return}const a=Number(e.tokens_total||0),r=Number(e.tokens_prompt||0),l=Number(e.tokens_files||
0),d=[];Number(e.files_non_text||0)>0&&d.push(`\u975E\u30C6\u30AD\u30B9\u30C8${e.files_non_text}\u4EF6\u306F0\u63DB\
\u7B97`),Number(e.files_missing||0)>0&&d.push(`\u672A\u691C\u51FA${e.files_missing}\u4EF6`),Number(e.
files_error||0)>0&&d.push(`\u5931\u6557${e.files_error}\u4EF6`);const p=d.length?` \u30FB ${d.join("\
 / ")}`:"";setPromptTokenEstimateText(`\u5165\u529B\u898B\u7A4D: ${a} tokens (\u672C\u6587 ${r} / \u30D5\u30A1\
\u30A4\u30EB ${l})${p}`,"text-cyan-300")}o(renderPromptTokenEstimate,"renderPromptTokenEstimate");function schedulePromptTokenEstimate(e=!1){
const t=buildPromptTokenEstimatePayload(),n=!!((t.message||"").trim()||(t.quote_text||"").trim()),i=Array.
isArray(t.image_urls)&&t.image_urls.length>0;if(!n&&!i){promptTokenEstimateLastKey="",promptTokenEstimateLastData=
null,promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=null),
promptTokenEstimateAbort&&(promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=null),renderPromptTokenEstimate(
null,t);return}const s=JSON.stringify([t.model||"",t.message||"",t.quote_text||"",t.image_urls||[]]);
if(s===promptTokenEstimateLastKey&&promptTokenEstimateLastData){renderPromptTokenEstimate(promptTokenEstimateLastData,
t);return}promptTokenEstimateTimer&&(clearTimeout(promptTokenEstimateTimer),promptTokenEstimateTimer=
null);const a=o(async()=>{promptTokenEstimateAbort&&promptTokenEstimateAbort.abort(),promptTokenEstimateAbort=
new AbortController;const r=++promptTokenEstimateSeq;renderPromptTokenEstimate({pending:!0},t);try{const l=await apiFetch(
CHAT_CONFIG.urls.estimatePromptTokensApi,{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(t),signal:promptTokenEstimateAbort.signal});if(!l.ok)throw new Error(`HTTP ${l.status}`);
const d=await l.json();if(r!==promptTokenEstimateSeq)return;promptTokenEstimateLastKey=s,promptTokenEstimateLastData=
d,renderPromptTokenEstimate(d,t)}catch(l){if(l&&l.name==="AbortError"||r!==promptTokenEstimateSeq)return;
promptTokenEstimateLastKey="",promptTokenEstimateLastData=null,renderPromptTokenEstimate(null,t)}},"\
run");e?a():promptTokenEstimateTimer=setTimeout(a,PROMPT_TOKEN_ESTIMATE_DEBOUNCE_MS)}o(schedulePromptTokenEstimate,
"schedulePromptTokenEstimate");function updatePromptPlaceholder(){const e=get("prompt-input");e&&(editingMessageId?
e.placeholder="\u7DE8\u96C6\u4E2D... (Enter\u9001\u4FE1\u306F\u8A2D\u5B9A\u306B\u5F93\u3044\u307E\u3059)":
enterToSend?e.placeholder="Enter \u3067\u9001\u4FE1 (Shift+Enter \u3067\u6539\u884C)":e.placeholder=
"Ctrl + Enter \u3067\u9001\u4FE1...")}o(updatePromptPlaceholder,"updatePromptPlaceholder");function readPromptBarModeFromForm(){
return get("set-minimal-prompt-mode")&&get("set-minimal-prompt-mode").checked?{compact_prompt_mode:!1,
minimal_prompt_mode:!0}:get("set-compact-prompt-mode")&&get("set-compact-prompt-mode").checked?{compact_prompt_mode:!0,
minimal_prompt_mode:!1}:{compact_prompt_mode:!1,minimal_prompt_mode:!1}}o(readPromptBarModeFromForm,
"readPromptBarModeFromForm");function writePromptBarModeToForm(e,t){const n=get("set-prompt-bar-mode\
-normal"),i=get("set-compact-prompt-mode"),s=get("set-minimal-prompt-mode");t&&s?s.checked=!0:e&&i?i.
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
"prompt-controls-toggle-icon"),s=get("prompt-controls-row");if(applyMinimalPromptMode(),!e||!t)return;
const a=compactPromptMode&&!minimalPromptMode,r=!a||promptControlsExpanded;s&&s.classList.toggle("co\
mpact-collapsed",a&&!r),a?r?(e.classList.remove("collapsed"),e.classList.add("expanded"),e.classList.
remove("hidden")):(e.classList.remove("expanded"),e.classList.add("collapsed")):(e.classList.remove(
"hidden"),e.classList.remove("collapsed"),e.classList.remove("expanded")),a?(t.classList.remove("hid\
den"),t.classList.add("inline-flex"),t.setAttribute("aria-expanded",r?"true":"false"),n&&(n.textContent=
r?"\u6298\u308A\u305F\u305F\u3080":"\u8A73\u7D30"),i&&(i.className=r?"fas fa-chevron-up text-[10px]":
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
imalOptionChecked");function buildMinimalOptionItem(e){const t=document.createElement("div");t.className=
"minimal-option-item",t.dataset.key=e.key,e.action&&t.classList.add("action-"+e.action),minimalOptionChecked(
e)?t.classList.add("on"):t.classList.add("off"),minimalOptionDisabled(e)&&t.classList.add("disabled");
const n=document.createElement("i");n.className="fas "+e.icon+" minimal-option-icon",t.appendChild(n);
const i=document.createElement("span");if(i.className="minimal-option-label",i.textContent=e.label,t.
appendChild(i),e.selectId){const s=get(e.selectId);if(s){const a=s.cloneNode(!0);a.removeAttribute("\
id"),a.className="minimal-option-select",a.addEventListener("change",()=>{s.value=a.value,s.dispatchEvent(
new Event("change",{bubbles:!0})),refreshMinimalOptionItems()}),t.appendChild(a)}}if(e.gear){const s=document.
createElement("button");s.type="button",s.className="minimal-option-gear",s.title=e.label+"\u8A2D\u5B9A";
const a=document.createElement("i");a.className="fas fa-cog",s.appendChild(a),s.addEventListener("cl\
ick",r=>{r.stopPropagation(),closeMinimalOptions(),typeof e.gearAction=="function"&&e.gearAction()}),
t.appendChild(s)}return t.addEventListener("click",()=>handleMinimalOptionClick(e)),t}o(buildMinimalOptionItem,
"buildMinimalOptionItem");function renderMinimalOptionItems(){const e=get("minimal-options-items");if(!e)
return;const t=document.createDocumentFragment();MINIMAL_POPUP_ITEMS.forEach(n=>{minimalOptionVisible(
n)&&t.appendChild(buildMinimalOptionItem(n))}),e.innerHTML="",e.appendChild(t)}o(renderMinimalOptionItems,
"renderMinimalOptionItems");function refreshMinimalOptionItems(){const e=get("minimal-options-items");
if(!e||!minimalOptionsOpen)return;const t=e.querySelectorAll(".minimal-option-item"),n={};t.forEach(
i=>{n[i.dataset.key]=i}),MINIMAL_POPUP_ITEMS.forEach(i=>{const s=n[i.key];if(s){if(!minimalOptionVisible(
i)){s.classList.add("hidden");return}if(s.classList.remove("hidden"),s.classList.toggle("on",minimalOptionChecked(
i)),s.classList.toggle("off",!minimalOptionChecked(i)),s.classList.toggle("disabled",minimalOptionDisabled(
i)),i.selectId){const a=get(i.selectId),r=s.querySelector(".minimal-option-select");a&&r&&document.activeElement!==
r&&r.value!==a.value&&(r.value=a.value)}}})}o(refreshMinimalOptionItems,"refreshMinimalOptionItems");
function handleMinimalOptionClick(e){if(e.action==="upload"){closeMinimalOptions(),openUploadModal();
return}if(e.action==="button"){const n=get(e.buttonId);closeMinimalOptions(),n&&n.click();return}if(e.
special==="thinking"){const n=get(e.checkboxId);if(n&&!n.disabled){const i=!n.checked;n.checked=i,n.
dispatchEvent(new Event("change",{bubbles:!0})),i?(closeMinimalOptions(),showThinkingSlider()):hideThinkingSlider(),
refreshMinimalOptionItems()}else closeMinimalOptions(),showThinkingSlider();return}if(minimalOptionDisabled(
e)||e.selectId)return;const t=get(e.checkboxId);t&&(t.disabled||(t.checked=!t.checked,t.dispatchEvent(
new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),e.key==="fast"?(closeMinimalOptions(),
setTimeout(()=>refreshMinimalOptionItems(),350)):e.key==="tempchat"&&setTimeout(()=>refreshMinimalOptionItems(),
350)))}o(handleMinimalOptionClick,"handleMinimalOptionClick");function moveModelPanelsIntoPopup(){const e=get(
"minimal-options-model-body");if(!e)return;let t=!1;MINIMAL_MODEL_PANEL_IDS.forEach(n=>{const i=get(
n);if(i){if(i.parentElement===e){i.classList.contains("hidden")||(t=!0);return}minimalPanelOrigins.has(
i)||(minimalPanelOrigins.set(i,{parent:i.parentElement,next:i.nextSibling}),e.appendChild(i),i.classList.
contains("hidden")||(t=!0))}}),refreshMinimalModelSection()}o(moveModelPanelsIntoPopup,"moveModelPan\
elsIntoPopup");function restoreModelPanelsFromPopup(){get("minimal-options-model-body")&&(minimalPanelOrigins.
forEach((t,n)=>{t.parent&&t.parent.contains(n)&&(t.next&&t.next.parentNode===t.parent?t.parent.insertBefore(
n,t.next):t.parent.appendChild(n))}),minimalPanelOrigins.clear())}o(restoreModelPanelsFromPopup,"res\
toreModelPanelsFromPopup");function refreshMinimalModelSection(){const e=get("minimal-options-model-\
body"),t=get("minimal-options-model-section");if(!e||!t)return;let n=!1;Array.from(e.children).forEach(
i=>{i.classList.contains("hidden")||(n=!0)}),t.classList.toggle("hidden",!n)}o(refreshMinimalModelSection,
"refreshMinimalModelSection");function openMinimalOptions(){if(minimalOptionsOpen||!minimalPromptMode)
return;hideThinkingSlider(),minimalOptionsOpen=!0,renderMinimalOptionItems(),moveModelPanelsIntoPopup();
const e=get("minimal-options-popup");if(!e)return;const t=get("minimal-options-panel");t&&(t.style.transform=
"",t.style.opacity=""),e.classList.remove("hidden"),e.setAttribute("aria-hidden","false"),e.offsetWidth,
e.classList.add("minimal-options-open")}o(openMinimalOptions,"openMinimalOptions");function closeMinimalOptions(){
if(!minimalOptionsOpen)return;minimalOptionsOpen=!1;const e=get("minimal-options-popup");e&&(e.classList.
remove("minimal-options-open"),e.setAttribute("aria-hidden","true"),setTimeout(()=>{minimalOptionsOpen||
e.classList.add("hidden")},400)),restoreModelPanelsFromPopup(),hideThinkingSlider()}o(closeMinimalOptions,
"closeMinimalOptions");function toggleMinimalOptions(){minimalOptionsOpen?closeMinimalOptions():openMinimalOptions()}
o(toggleMinimalOptions,"toggleMinimalOptions");function refreshMinimalOptionsIfOpen(){minimalOptionsOpen&&
(renderMinimalOptionItems(),refreshMinimalModelSection())}o(refreshMinimalOptionsIfOpen,"refreshMini\
malOptionsIfOpen");function allowedThinkingValues(){const e=get("thinking-level");return e?Array.from(
e.options).filter(n=>!n.disabled&&!n.classList.contains("hidden")).map(n=>n.value):THINKING_LEVELS.map(
n=>n.value)}o(allowedThinkingValues,"allowedThinkingValues");function thinkingIndexFromValue(e){const t=THINKING_LEVELS.
findIndex(n=>n.value===e);return t<0?3:t}o(thinkingIndexFromValue,"thinkingIndexFromValue");function syncThinkingSliderUi(){
const e=get("thinking-slider"),t=get("thinking-slide-value"),n=get("thinking-level"),i=thinkingIndexFromValue(
n?n.value:"high");e&&(e.value=String(i)),t&&(t.textContent=THINKING_LEVELS[i].label)}o(syncThinkingSliderUi,
"syncThinkingSliderUi");function scheduleThinkingSliderHide(){thinkingSliderTimer&&clearTimeout(thinkingSliderTimer),
thinkingSliderTimer=setTimeout(()=>{thinkingSliderTimer=null,hideThinkingSlider()},2500)}o(scheduleThinkingSliderHide,
"scheduleThinkingSliderHide");function showThinkingSlider(){if(thinkingSliderOpen){scheduleThinkingSliderHide();
return}const e=get("thinking-slide-bar");if(!e)return;const t=get("thinking-slide-inner");t&&(t.style.
transform=""),thinkingSliderOpen=!0,e.classList.remove("hidden"),e.setAttribute("aria-hidden","false"),
syncThinkingSliderUi(),e.offsetWidth,e.classList.add("thinking-slide-open"),scheduleThinkingSliderHide()}
o(showThinkingSlider,"showThinkingSlider");function hideThinkingSlider(){thinkingSliderTimer&&(clearTimeout(
thinkingSliderTimer),thinkingSliderTimer=null);const e=get("thinking-slide-bar");e&&(thinkingSliderOpen=
!1,e.classList.remove("thinking-slide-open"),e.setAttribute("aria-hidden","true"),setTimeout(()=>{thinkingSliderOpen||
e.classList.add("hidden");const t=get("thinking-slide-inner");t&&(t.style.transform="")},360))}o(hideThinkingSlider,
"hideThinkingSlider");function bindMinimalOptionsEvents(){const e=get("minimal-options-backdrop"),t=get(
"minimal-options-close-btn");e&&e.addEventListener("click",()=>closeMinimalOptions()),t&&t.addEventListener(
"click",()=>closeMinimalOptions()),document.addEventListener("keydown",r=>{if(r.key==="Escape"){if(minimalOptionsOpen){
closeMinimalOptions();return}thinkingSliderOpen&&hideThinkingSlider()}});const n=get("thinking-slide\
r");n&&n.addEventListener("input",()=>{const r=Number(n.value),l=allowedThinkingValues(),d=get("thin\
king-level");if(l.length){const p=l.map(h=>thinkingIndexFromValue(h)),g=p.includes(r)?r:p.reduce((h,v)=>Math.
abs(v-r)<Math.abs(h-r)?v:h,p[0]);d&&(d.value=THINKING_LEVELS[g].value,d.dispatchEvent(new Event("cha\
nge",{bubbles:!0})))}syncThinkingSliderUi(),scheduleThinkingSliderHide()});const i=get("thinking-sli\
de-close-btn");i&&i.addEventListener("click",r=>{r.stopPropagation(),hideThinkingSlider()});const s=get(
"thinking-slide-bar");if(s){const r=get("thinking-slide-inner");s.addEventListener("touchstart",l=>{
thinkingSliderOpen&&(thinkingSliderDragging=!0,thinkingSliderStartY=l.touches[0].clientY,thinkingSliderStartX=
l.touches[0].clientX,thinkingSliderAxis=null,r&&r.classList.add("dragging"))},{passive:!0}),s.addEventListener(
"touchmove",l=>{if(!thinkingSliderDragging)return;const d=l.touches[0].clientX-thinkingSliderStartX,
p=l.touches[0].clientY-thinkingSliderStartY;if(thinkingSliderAxis===null&&(Math.abs(d)>8||Math.abs(p)>
8)&&(thinkingSliderAxis=Math.abs(p)>Math.abs(d)?"v":"h"),thinkingSliderAxis==="v")if(p>0){l.cancelable&&
l.preventDefault();const g=Math.min((p-8)*.5,120);r&&(r.style.transform=g>0?`translateY(${g}px)`:"")}else
r&&(r.style.transform="")},{passive:!1}),s.addEventListener("touchend",l=>{if(!thinkingSliderDragging)
return;thinkingSliderDragging=!1;const d=l.changedTouches[0].clientY-thinkingSliderStartY;r&&r.classList.
remove("dragging"),thinkingSliderAxis==="v"&&d>100?(r&&(r.style.transform=`translateY(${Math.max(d*.5,
60)}px)`),hideThinkingSlider()):(r&&(r.style.transform=""),scheduleThinkingSliderHide())},{passive:!0}),
s.addEventListener("touchcancel",()=>{thinkingSliderDragging=!1,r&&(r.classList.remove("dragging"),r.
style.transform=""),scheduleThinkingSliderHide()},{passive:!0})}const a=get("minimal-options-panel");
a&&(a.addEventListener("touchstart",r=>{if(!minimalOptionsOpen)return;popupSwipeDragging=!0,popupSwipeStartY=
r.touches[0].clientY,popupSwipeStartX=r.touches[0].clientX,popupSwipeAxis=null;let l=r.target instanceof
Element?r.target:null,d=!0;for(;l&&l!==a;){if(l.scrollTop>0){d=!1;break}l=l.parentElement}popupSwipeAtTop=
d,d&&a.classList.add("dragging")},{passive:!0}),a.addEventListener("touchmove",r=>{if(!popupSwipeDragging||
!popupSwipeAtTop||!minimalOptionsOpen)return;const l=r.touches[0].clientX-popupSwipeStartX,d=r.touches[0].
clientY-popupSwipeStartY;popupSwipeAxis===null&&(Math.abs(l)>8||Math.abs(d)>8)&&(popupSwipeAxis=Math.
abs(d)>Math.abs(l)?"v":"h"),popupSwipeAxis==="v"&&d>0&&(r.cancelable&&r.preventDefault(),a.style.transform=
`translateY(${Math.min(d*.6,140)}px)`)},{passive:!1}),a.addEventListener("touchend",r=>{if(!popupSwipeDragging)
return;popupSwipeDragging=!1;const l=r.changedTouches[0].clientY-popupSwipeStartY;a.classList.remove(
"dragging"),popupSwipeAtTop&&popupSwipeAxis!=="h"&&l>70?(a.style.transform=`translateY(${Math.max(l*
.6,100)}px)`,a.style.opacity="0",closeMinimalOptions(),setTimeout(()=>{a.style.transform="",a.style.
opacity=""},340)):a.style.transform=""},{passive:!0}),a.addEventListener("touchcancel",()=>{popupSwipeDragging=
!1,a.classList.remove("dragging"),a.style.transform="",a.style.opacity=""},{passive:!0}))}o(bindMinimalOptionsEvents,
"bindMinimalOptionsEvents");function bindUploadButton(){const e=get("upload-btn");e&&(e.onclick=()=>{
minimalPromptMode?toggleMinimalOptions():openUploadModal()})}o(bindUploadButton,"bindUploadButton");
function applyChatDefaults(e){if(!e||(Object.prototype.hasOwnProperty.call(e,"voice_studio_ui")&&(voiceStudioUiEnabled=
e.voice_studio_ui!==!1),applyTemporaryChatTimeoutSeconds(e.temp_chat_timeout_seconds),chatDefaultsLoaded))
return;const n=!!e.use_last_chat_settings?{model:e.last_model,enable_search:e.last_enable_search,enable_url_context:e.
last_enable_url_context,enable_maps:e.last_enable_maps,enable_python:e.last_enable_python,enable_file_creation:e.
last_enable_file_creation,enable_thinking:e.last_enable_thinking,thinking_level:e.last_thinking_level,
thinking_budget:e.last_thinking_budget,reasoning_effort:e.last_reasoning_effort,enable_system_prompt:e.
last_enable_system_prompt,enable_mcp:e.last_enable_mcp,safety_setting:e.last_safety_setting}:{model:e.
default_model,enable_search:e.default_enable_search,enable_url_context:e.default_enable_url_context,
enable_maps:e.default_enable_maps,enable_python:e.default_enable_python,enable_file_creation:e.default_enable_file_creation,
enable_thinking:e.default_enable_thinking,thinking_level:e.default_thinking_level,thinking_budget:e.
default_thinking_budget,reasoning_effort:e.default_reasoning_effort,enable_system_prompt:e.default_enable_system_prompt,
enable_mcp:e.default_enable_mcp,safety_setting:e.default_safety_setting},i=o((s,a)=>s==null||s===""?
a:s,"s");n.model&&selectModelById(n.model),get("enable-search")&&(get("enable-search").checked=!!i(n.
enable_search,get("enable-search").checked)),get("enable-url-context")&&(get("enable-url-context").checked=
!!i(n.enable_url_context,get("enable-url-context").checked)),get("enable-maps")&&(get("enable-maps").
checked=!!i(n.enable_maps,get("enable-maps").checked)),get("enable-python")&&(get("enable-python").checked=
!!i(n.enable_python,get("enable-python").checked)),get("enable-file-creation")&&(get("enable-file-cr\
eation").checked=!!i(n.enable_file_creation,get("enable-file-creation").checked)),get("enable-thinki\
ng")&&(get("enable-thinking").checked=!!i(n.enable_thinking,get("enable-thinking").checked)),get("th\
inking-level")&&(get("thinking-level").value=i(n.thinking_level,get("thinking-level").value||"high")),
get("thinking-budget")&&(get("thinking-budget").value=i(n.thinking_budget,get("thinking-budget").value||
4096)),get("reasoning-effort")&&(get("reasoning-effort").value=i(n.reasoning_effort,get("reasoning-e\
ffort").value||"medium")),get("enable-sys-prompt")&&(get("enable-sys-prompt").checked=!!i(n.enable_system_prompt,
get("enable-sys-prompt").checked)),get("enable-mcp")&&(get("enable-mcp").checked=!!i(n.enable_mcp,get(
"enable-mcp").checked)),get("safety-setting")&&(get("safety-setting").value=i(n.safety_setting,get("\
safety-setting").value||"default")),chatDefaultsLoaded=!0,toggleOptions(),applyMcpPromptChipUi()}o(applyChatDefaults,
"applyChatDefaults");function setEditUi(e){const t=get("edit-bar");t&&(e?(t.classList.remove("hidden"),
t.classList.add("flex")):(t.classList.add("hidden"),t.classList.remove("flex")),updatePromptPlaceholder())}
o(setEditUi,"setEditUi");function cancelEdit(){editingMessageId=null,currentParentId=currentLeafId||
null;const e=get("prompt-input");e&&(e.value="",e.style.height="auto"),currentImageUrls=[],get("file\
-preview").classList.add("hidden"),get("file-input").value="",clearQuote(),setEditUi(!1)}o(cancelEdit,
"cancelEdit");function beginEditMessage(e,t=!1){const n=messageStore[e];if(n==null)return;const i=get(
"prompt-input");i.value=n||"",i.focus(),i.style.height="auto",i.style.height=i.scrollHeight+"px";const s=allMessages.
find(d=>d.id==e),a=messageMeta[e]||{};s?currentParentId=s.parent_id===void 0?null:s.parent_id:a.parent_id!==
void 0&&(currentParentId=a.parent_id),editingMessageId=e,setEditUi(!0);const r=s?s.image_url:a.image_url;
if(r)try{const d=JSON.parse(r);Array.isArray(d)&&d.length?(currentImageUrls=d.map(p=>{let g="unknown",
h=p;p&&typeof p=="object"&&(g=normalizeAttachmentSource(p.source),h=p.filepath||p.path||p.url||p.file||
"");const v=normalizeAttachmentPath(h);return v&&setAttachmentSourceForPath(v,g),v}).filter(Boolean),
get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):(currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value=
"")}catch{currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value=""}else
currentImageUrls=[],get("file-preview").classList.add("hidden"),get("file-input").value="";const l=s?
s.quote_text:a.quote_text;l?(currentQuote=l,get("quote-text-display").innerText=currentQuote,get("qu\
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
abortController.abort();try{if(t||e){const s={};t&&(s.job_id=t),e&&(s.thread_id=e);const r=await(await apiFetch(
"/api/stop_chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(s)})).
json().catch(()=>({})),l=normalizeJobIdForUi(r&&r.job_id);l&&(suppressPendingJob(l),manualStopContext&&
manualStopContext.seq===n&&(manualStopContext.jobId=l))}manualStopContext&&manualStopContext.seq===n&&
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
let i=0;for(;;){if(t.signal&&t.signal.aborted)throw new DOMException("Aborted","AbortError");try{const s=await apiFetch(
e,t),a=window.ConnectionMonitor.retryModeForResponse(s);let r=!1;if(s.status===425&&(r=(await s.clone().
json().catch(()=>({}))).code==="submission_in_progress"),!a&&!r)return window.ConnectionMonitor.markReachable(),
s;i+=1,a&&window.ConnectionMonitor.setUnavailable(a),updatePendingSkeletonStatus(n,a==="maintenance"?
"\u30E1\u30F3\u30C6\u30CA\u30F3\u30B9\u7D42\u4E86\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...":"\u30B5\u30FC\u30D0\
\u30FC\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}catch(s){if(t.signal&&t.signal.aborted||s.name==="AbortError")throw s;i+=1,window.
ConnectionMonitor.setUnavailable("offline"),updatePendingSkeletonStatus(n,"\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u63A5\u7D9A\u306E\u5FA9\u5E30\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
`\u9001\u4FE1\u5185\u5BB9\u3092\u4FDD\u6301\u3057\u3066\u81EA\u52D5\u518D\u8A66\u884C\u4E2D\uFF08${i}\
\u56DE\u76EE\uFF09`)}await window.ConnectionMonitor.waitForRetry(t.signal)}}o(fetchChatStreamWithUnavailableRetry,
"fetchChatStreamWithUnavailableRetry");function createClientRequestId(){return window.crypto&&typeof window.
crypto.randomUUID=="function"?window.crypto.randomUUID():`req-${window.crypto&&typeof window.crypto.
getRandomValues=="function"?Array.from(window.crypto.getRandomValues(new Uint32Array(4))).map(t=>t.toString(
16)).join(""):`${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`}`.slice(0,64)}o(createClientRequestId,
"createClientRequestId");async function reconnectPendingStreamUntilAvailable(e,t){const n=t!=null?String(
t):"",i=normalizeJobIdForUi(e&&e.job_id),s=i||`thread:${n}`;if(!n||pendingStreamReconnectJobs.has(s))
return;pendingStreamReconnectJobs.add(s);const a=new AbortController;let r=!1;abortController=a,currentJobId=
i,setSendBtnToStopMode();try{for(;!a.signal.aborted;){if(String(currentThreadId||"")!==n||i&&isPendingJobSuppressed(
i))return;const l=getActiveStreamingBubbleElement();if(updatePendingSkeletonStatus(l,"\u30B5\u30FC\u30D0\u30FC\u3078\u306E\u518D\u63A5\u7D9A\u3092\u5F85\u3063\u3066\u3044\
\u307E\u3059...","\u56DE\u7B54\u51E6\u7406\u306F\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u7D99\u7D9A\u3057\u3066\u3044\u307E\u3059"),
await window.ConnectionMonitor.waitForRetry(a.signal),!await loadMessages(n,{preserveDraft:!0,silent:!0,
skipHistory:!0})){window.ConnectionMonitor.probeNow();continue}const p=currentThreadPending;p&&p.job_id&&
!isPendingJobSuppressed(p.job_id)?(abortController===a&&(abortController=null),r=!0,resumePendingStream(
p)):window.ConnectionMonitor.markReachable();return}}catch(l){l.name!=="AbortError"&&sendClientDebugLog(
"error",`Stream reconnect failed: ${l.message}`)}finally{pendingStreamReconnectJobs.delete(s),abortController===
a&&(abortController=null),r||(currentJobId=null,setSendBtnToSendMode(),updateFilePreview())}}o(reconnectPendingStreamUntilAvailable,
"reconnectPendingStreamUntilAvailable"),window.initTurnstileWidget=()=>{if(!botConfig||!botConfig.turnstileSiteKey||
!window.turnstile||turnstileWidgetId!==null)return;const e=document.getElementById("turnstile-contai\
ner");e&&(e.classList.remove("hidden"),turnstileWidgetId=window.turnstile.render(e,{sitekey:botConfig.
turnstileSiteKey,size:"compact",appearance:"interaction-only",callback:o(t=>{turnstileToken=t,turnstilePending=
!1,verifyTurnstileOnServer(t)},"callback"),"expired-callback":o(()=>{turnstileToken=null,turnstilePending=
!1},"expired-callback"),"error-callback":o(()=>{turnstileToken=null,turnstilePending=!1},"error-call\
back")}),isBotDetectionActive()&&runBotDetectionGate())};async function getTurnstileToken(e=1500){if(!botConfig||
!botConfig.turnstileSiteKey)return null;if(turnstileToken)return turnstileToken;if(!window.turnstile)
return null;if(botDetectionOverlayShown&&botDetectionDialogWidgetId!==null)return turnstilePending=!0,
await new Promise(n=>{const i=turnstileToken,s=setTimeout(()=>n(null),Math.max(500,Number(e)||1500)),
a=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(s),clearInterval(a),n(turnstileToken))},
50)});if(turnstileWidgetId===null)return null;const t=document.getElementById("turnstile-container");
return t&&t.classList.remove("hidden"),turnstilePending=!0,await new Promise(n=>{const i=turnstileToken,
s=setTimeout(()=>n(null),Math.max(500,Number(e)||1500));try{window.turnstile.execute(turnstileWidgetId)}catch{
clearTimeout(s),n(null);return}const a=setInterval(()=>{turnstileToken&&turnstileToken!==i&&(clearTimeout(
s),clearInterval(a),verifyTurnstileOnServer(turnstileToken),n(turnstileToken))},50)})}o(getTurnstileToken,
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
play:flex;flex-direction:column;align-items:stretch;gap:12px;";const s=document.createElement("div");
s.id="bot-detection-overlay-title",s.style.cssText="font-weight:700;font-size:15px;color:#f1f5f9;",s.
textContent=e||"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u4E2D...";const a=document.createElement("div");
a.style.cssText="font-size:12px;color:#94a3b8;line-height:1.6;",a.textContent="\u81EA\u52D5\u30A2\u30AF\u30BB\u30B9\u9632\u6B62\u306E\u305F\u3081\u3001\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u3066\u304F\u3060\
\u3055\u3044\u3002";const r=document.createElement("div");r.id="bot-detection-widget-box",r.style.cssText=
"margin-top:8px;min-height:65px;display:flex;justify-content:center;",i.appendChild(s),i.appendChild(
a),i.appendChild(r),t.appendChild(i),document.body.appendChild(t)}const n=document.getElementById("b\
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
x;";const s=document.createElement("div");s.style.cssText="font-size:26px;color:#fbbf24;",s.innerHTML=
'<i class="fas fa-lock"></i>';const a=document.createElement("div");a.id="bot-lock-overlay-title",a.
style.cssText="font-weight:700;font-size:16px;color:#fbbf24;",a.textContent="\u30A2\u30AB\u30A6\u30F3\u30C8\u304C\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3055\u308C\u307E\u3057\u305F";
const r=document.createElement("div");r.id="bot-lock-overlay-message",r.style.cssText="font-size:13p\
x;color:#f1f5f9;line-height:1.7;",r.textContent=e;const l=document.createElement("div");l.id="bot-lo\
ck-overlay-timer",l.style.cssText="font-size:12px;color:#94a3b8;margin-top:2px;";const d=document.createElement(
"div");d.style.cssText="font-size:11px;color:#94a3b8;line-height:1.6;",d.textContent="\u30ED\u30C3\u30AF\u89E3\u9664\u307E\u3067\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\
\u304F\u3060\u3055\u3044\u3002\u540C\u3058\u64CD\u4F5C\u3092\u7E70\u308A\u8FD4\u3059\u3068BAN\u3055\u308C\u308B\u5834\u5408\u304C\u3042\u308A\u307E\u3059\u3002",
i.appendChild(s),i.appendChild(a),i.appendChild(r),i.appendChild(l),i.appendChild(d),n.appendChild(i),
document.body.appendChild(n)}return botLockOverlay=n,updateBotLockTimer(t),n}o(showBotLockOverlay,"s\
howBotLockOverlay");function updateBotLockTimer(e){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const t=document.getElementById("bot-lock-overlay-timer");if(!t)return;const n=o(()=>{const i=Math.
max(0,Math.round(Number(e)||0)),s=Math.floor(i/60),a=String(i%60).padStart(2,"0");t.textContent=`\u30ED\u30C3\u30AF\
\u89E3\u9664\u307E\u3067: ${s}:${a}`},"render");n(),botLockTimer=setInterval(()=>{e-=1,n(),e<=0&&(botLockTimer&&
(clearInterval(botLockTimer),botLockTimer=null),location.reload())},1e3)}o(updateBotLockTimer,"updat\
eBotLockTimer");function hideBotLockOverlay(){botLockTimer&&(clearInterval(botLockTimer),botLockTimer=
null);const e=document.getElementById("bot-lock-overlay");e&&e.remove(),botLockOverlay=null}o(hideBotLockOverlay,
"hideBotLockOverlay");async function applyBotLockFromServer(e){if(isAdminUser)return!0;let t=600;try{
const n=await apiFetch("/api/bot/lock",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({reason:e||""})});if(n.status===403){let s=null;try{s=await n.json()}catch{}if(s&&s.error===
"banned")return showToast("\u30ED\u30C3\u30AF\u304C\u7E70\u308A\u8FD4\u3055\u308C\u305F\u305F\u3081BAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800),!1}const i=await n.json().catch(()=>({}));
if(i&&(i.status==="skipped"||i.skipped))return!0;i&&typeof i.remaining_seconds=="number"&&(t=i.remaining_seconds)}catch{}
return showBotLockOverlay(e||"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u4E00\u6642\u7684\u306B\u30ED\u30C3\u30AF\u3057\u3066\u3044\u307E\u3059\u3002",
t),!1}o(applyBotLockFromServer,"applyBotLockFromServer");const runBotDetectionGate=o(()=>botDetectionVerified||
!isBotDetectionActive()?Promise.resolve(!0):botDetectionGatePromise||(botDetectionGatePromise=(async()=>{
let e=0;for(;!botDetectionVerified;){if(!botDetectionOverlayShown){if(!window.__turnstileApiLoaded||
turnstileWidgetId===null){await new Promise(s=>setTimeout(s,1e3));continue}const n=await getTurnstileToken(
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
e;const s=!!n;return turnstileVerifyInFlight=(async()=>{try{return(await apiFetch("/api/bot/turnstil\
e-verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({turnstile_token:e,
challenged:s})})).ok?(turnstileServerVerifiedAt=Date.now(),botDetectionVerified=!0,hideBotDetectionOverlay(),
!0):!1}catch{return!1}finally{turnstileVerifyInFlightToken===e&&(turnstileVerifyInFlight=null,turnstileVerifyInFlightToken=
null)}})(),turnstileVerifyInFlight}o(verifyTurnstileOnServer,"verifyTurnstileOnServer");function botTurnstileTokenForRequest(){
return isBotDetectionActive()?turnstileToken:null}o(botTurnstileTokenForRequest,"botTurnstileTokenFo\
rRequest");const botTelemetry=(()=>{const e={enabled:!1,windowStart:performance.now(),lastSend:0,clicks:0,
keys:0,moves:0,fastClicks:0,fastKeys:0,untrustedInput:!1,clickTimes:[],keyTimes:[],clickIntervals:[],
lastClickTs:0,lastKeyTs:0,lastMove:null,speedMax:0,speedSum:0,speedSamples:0,lastMoveSample:0},t=o(()=>{
e.enabled=!!(botConfig&&botConfig.globalEnabled&&botConfig.accountEnabled&&!isAdminUser)},"refreshEn\
abled"),n=o(()=>{e.windowStart=performance.now(),e.clicks=0,e.keys=0,e.moves=0,e.fastClicks=0,e.fastKeys=
0,e.untrustedInput=!1,e.clickTimes=[],e.keyTimes=[],e.clickIntervals=[],e.speedMax=0,e.speedSum=0,e.
speedSamples=0},"resetWindow"),i=o(v=>{const b=v&&v.target;return!b||typeof b.closest!="function"?!1:
!!b.closest("[data-bot-ignore-click], #new-chat-btn, #mobile-new-chat-btn, #bot-detection-overlay")},
"isControlClick"),s=o(v=>{if(i(v))return;if(v&&v.isTrusted===!1){e.untrustedInput=!0,p(!0);return}const b=performance.
now();if(e.clicks+=1,e.lastClickTs){const w=b-e.lastClickTs;e.clickIntervals.push(w),e.clickIntervals.
length>10&&e.clickIntervals.shift(),w<120&&(e.fastClicks+=1)}e.lastClickTs=b,e.clickTimes.push(b),e.
clickTimes=e.clickTimes.filter(w=>b-w<=2e3),e.fastClicks>=4&&p(!0)},"recordClick"),a=o(v=>{if(v&&v.isTrusted===
!1){e.untrustedInput=!0,p(!0);return}const b=performance.now();e.keys+=1,e.lastKeyTs&&b-e.lastKeyTs<
50&&(e.fastKeys+=1),e.lastKeyTs=b,e.keyTimes.push(b),e.keyTimes=e.keyTimes.filter(w=>b-w<=2e3)},"rec\
ordKey"),r=o(v=>{const b=performance.now();if(!(b-e.lastMoveSample<80)){if(e.lastMoveSample=b,e.moves+=
1,e.lastMove){const w=v.clientX-e.lastMove.x,x=v.clientY-e.lastMove.y,L=b-e.lastMove.t;if(L>0){const T=Math.
sqrt(w*w+x*x)/(L/1e3);e.speedMax=Math.max(e.speedMax,T),e.speedSum+=T,e.speedSamples+=1}}e.lastMove=
{x:v.clientX,y:v.clientY,t:b}}},"recordMove"),l=o(()=>{const v=Math.max(1,performance.now()-e.windowStart),
b=e.clickTimes.length,w=e.keyTimes.length,x=e.speedSamples?e.speedSum/e.speedSamples:0;let L=0,T=1;if(e.
clickIntervals.length>=3){const $=e.clickIntervals.reduce((J,X)=>J+X,0)/e.clickIntervals.length,j=e.
clickIntervals.reduce((J,X)=>J+Math.pow(X-$,2),0)/e.clickIntervals.length;L=$,T=$>0?Math.sqrt(j)/$:1}
return{window_ms:Math.round(v),clicks:e.clicks,keys:e.keys,moves:e.moves,fast_clicks:e.fastClicks,fast_keys:e.
fastKeys,untrusted_input:!!e.untrustedInput,click_burst:b,key_burst:w,avg_click_ms:L,click_cv:T,event_rate:(e.
clicks+e.keys+e.moves)/(v/1e3),pointer_speed_max:e.speedMax,pointer_speed_avg:x}},"computeStats"),d=o(
v=>v.fast_clicks>=4||v.fast_keys>=8||v.click_burst>=8||v.key_burst>=14||v.event_rate>=20||v.avg_click_ms>
0&&v.avg_click_ms<160&&v.click_cv<.08,"isSuspicious"),p=o(async(v=!1,b={})=>{if(!e.enabled)return;const w=performance.
now();if(!v&&w-e.lastSend<3e3)return;e.lastSend=w;const x=l();if(!(!b.forceReport&&x.clicks+x.keys+x.
moves===0&&!x.untrusted_input)&&!(!v&&!x.untrusted_input&&!d(x))){x.turnstile_token=await getTurnstileToken(),
botConfig&&botConfig.turnstileSiteKey&&!x.turnstile_token&&!botDetectionVerified&&botDetectionOverlayShown&&
(x.turnstile_failed=!0,x.challenged=!0);try{const L=await apiFetch("/api/bot-telemetry",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(x)});if(L.status===403){let T=null;
try{T=await L.json()}catch{}if(T&&T.error==="banned"){showToast("\u30DC\u30C3\u30C8\u5224\u5B9A\u306B\u3088\u308ABAN\u3055\u308C\u307E\u3057\u305F\u3002",
"error",!0),setTimeout(()=>{location.href="/banned"},800);return}}}catch{}resetTurnstileToken(),n()}},
"send");return{start:o(()=>{t(),e.enabled&&(typeof window.PointerEvent!="undefined"?document.addEventListener(
"pointerdown",s,!0):document.addEventListener("click",s,!0),document.addEventListener("keydown",a,!0),
document.addEventListener("wheel",()=>{e.moves+=1},{passive:!0}),document.addEventListener("mousemov\
e",r,!0),setInterval(()=>p(!1),4e3))},"start"),refreshEnabled:t,send:p,looksSuspicious:o(()=>{if(!e.
enabled)return!1;const v=l();return d(v)},"looksSuspicious")}})();function openFileViewer(e,t=""){if(!e)
return;const n=(t||e).split(".").pop().toLowerCase(),i=["png","jpg","jpeg","webp","gif"],s=["mp4","m\
ov","mkv","avi","m4v","webm"],a=["mp3","wav","m4a","ogg","flac"],r=["pdf","txt","md","csv","log","js\
on","docx"];if(i.includes(n)){openImageViewer(e);return}const l=get("file-viewer"),d=get("file-viewe\
r-body"),p=get("file-viewer-title");if(!(!l||!d||!p)){if(p.textContent=t||"File Preview",d.replaceChildren(),
s.includes(n)){const g=document.createElement("video");g.src=String(e),g.controls=!0,g.playsInline=!0,
g.preload="metadata",d.appendChild(g)}else if(a.includes(n)){const g=document.createElement("audio");
g.src=String(e),g.controls=!0,d.appendChild(g)}else if(r.includes(n)){const g=document.createElement(
"iframe");g.src=String(e),g.setAttribute("sandbox",""),g.referrerPolicy="no-referrer",d.appendChild(
g)}else{const g=document.createElement("div");g.className="fallback",g.appendChild(document.createTextNode(
"\u3053\u306E\u5F62\u5F0F\u306F\u30D7\u30EC\u30D3\u30E5\u30FC\u3067\u304D\u307E\u305B\u3093\u3002"));
const h=document.createElement("div");h.className="mt-3 flex justify-center gap-2";const v=document.
createElement("a");v.href=String(e),v.download="",v.className="px-3 py-1 bg-gray-800 text-white roun\
ded text-xs border border-gray-700",v.textContent="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9";const b=document.
createElement("a");b.href=String(e),b.target="_blank",b.rel="noopener noreferrer",b.className=v.className,
b.textContent="\u65B0\u3057\u3044\u30BF\u30D6\u3067\u958B\u304F",h.append(v,b),g.appendChild(h),d.appendChild(
g)}l.classList.add("visible")}}o(openFileViewer,"openFileViewer");function closeFileViewer(){const e=get(
"file-viewer"),t=get("file-viewer-body");!e||!t||(t.innerHTML="",e.classList.remove("visible"))}o(closeFileViewer,
"closeFileViewer");function showToast(e,t="error",n=!1,i=null){const s=get("toast-stack");if(!s)return;
for(;s.children.length>=3;)s.removeChild(s.firstChild);const a=document.createElement("div");return a.
className=`toast ${t}${i?" toast-clickable":""}`,a.innerHTML=`<i class="fas ${t==="error"?"fa-triang\
le-exclamation":"fa-circle-info"}"></i><span class="flex-1">${escapeHtml(e)}</span><button aria-labe\
l="close"><i class="fas fa-times"></i></button>`,a.querySelector("button").onclick=r=>{r.stopPropagation(),
a.remove()},i&&a.addEventListener("click",i),s.appendChild(a),n||setTimeout(()=>{a.parentNode&&a.remove()},
7e3),a}o(showToast,"showToast");function showProgressToast(e,t="info"){const n=get("toast-stack");if(!n)
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
            `,i.querySelector("button").onclick=()=>i.remove(),n.appendChild(i),{update:o(s=>{const a=i.
querySelector(".progress-bar"),r=i.querySelector(".progress-text");a&&(a.style.width=`${Math.min(100,
Math.max(0,s))}%`),r&&(r.innerText=`${Math.round(s)}%`)},"update"),remove:o(()=>{i.parentNode&&i.remove()},
"remove")}}o(showProgressToast,"showProgressToast");let activeSettingsTab="general";const TAB_LABELS={
general:"\u4E00\u822C",api:"API\u30AD\u30FC",prompt:"\u30D7\u30ED\u30F3\u30D7\u30C8",display:"\u8868\u793A",
data:"\u30C7\u30FC\u30BF",account:"\u30A2\u30AB\u30A6\u30F3\u30C8",security:"\u30BB\u30AD\u30E5\u30EA\u30C6\u30A3",
"2fa":"2\u8981\u7D20\u8A8D\u8A3C",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",mcp:"MCP"},ALL_TABS=[
"general","api","prompt","display","data","account","security","2fa","feedback","mcp"];function getSectionHeading(e){
const t=e.querySelector("h3");if(t)return t.textContent.trim();const n=e.querySelector(".font-bold");
if(n&&!n.querySelector("input")&&!n.querySelector("select"))return n.textContent.trim();const i=e.querySelector(
"label");if(i){const s=i.textContent.trim().replace(/[：:].*$/,"").trim();if(s)return s}return""}o(
getSectionHeading,"getSectionHeading");function getSectionSnippet(e,t){const n=e.textContent,s=n.toLowerCase().
indexOf(t.toLowerCase());if(s===-1)return"";const a=Math.max(0,s-25),r=Math.min(n.length,s+t.length+
35);let l=n.substring(a,r).replace(/\s+/g," ").trim();return a>0&&(l="\u2026"+l),r<n.length&&(l=l+"\u2026"),
l}o(getSectionSnippet,"getSectionSnippet");function removeSearchOverlays(){ALL_TABS.forEach(e=>{const t=get(
"tab-"+e);if(!t)return;const n=t.querySelector(".settings-search-overlay");n&&n.remove(),Array.from(
t.children).forEach(i=>{i.classList.contains("settings-no-results")||(i.style.display="")})})}o(removeSearchOverlays,
"removeSearchOverlays");function filterSettings(){const e=get("settings-search");if(!e)return;const t=e.
value.trim().toLowerCase(),n=get("settings-search-clear");if(n&&n.classList.toggle("hidden",!t),removeSearchOverlays(),
!t){ALL_TABS.forEach(l=>{const d=get("btn-tab-"+l);if(d){const g=d.querySelector(".settings-search-b\
adge");g&&g.remove()}const p=get("tab-"+l);p&&p.classList.toggle("hidden",l!==activeSettingsTab)});return}
let i=[];ALL_TABS.forEach(l=>{const d=get("tab-"+l);d&&(d.classList.add("hidden"),Array.from(d.children).
forEach(p=>{if(!(p.classList.contains("settings-no-results")||p.classList.contains("settings-search-\
overlay"))&&p.textContent.toLowerCase().includes(t)){const g=getSectionHeading(p)||l,h=getSectionSnippet(
p,t);i.push({tabId:l,title:g,snippet:h,element:p})}}))});let s=activeSettingsTab;if(!i.some(l=>l.tabId===
s)){const l=i.find(d=>d.tabId);l&&(s=l.tabId)}const a=get("tab-"+s);if(!a)return;a.classList.remove(
"hidden"),Array.from(a.children).forEach(l=>{l.classList.contains("settings-no-results")||l.classList.
contains("settings-search-overlay")||(l.style.display="none")});const r=document.createElement("div");
if(r.className="settings-search-overlay",i.length===0){const l=document.createElement("div");l.className=
"settings-empty-state",l.innerHTML='<div class="settings-empty-icon"><i class="fas fa-search"></i></\
div><div class="settings-empty-title">\u4E00\u81F4\u3059\u308B\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093</div>';
const d=document.createElement("div");d.className="settings-empty-sub",d.textContent="\u300C"+t+"\u300D\u306B\u4E00\
\u81F4\u3059\u308B\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3002",l.appendChild(
d),r.appendChild(l)}else{const l=document.createElement("div");l.className="settings-search-count",l.
textContent=i.length+"\u4EF6\u306E\u4E00\u81F4",r.appendChild(l);let d=null;i.forEach((p,g)=>{if(p.tabId!==
d){if(d!==null){const L=document.createElement("div");L.className="border-t border-gray-700/50 my-1.\
5",r.appendChild(L)}if(p.tabId!==s){const L=document.createElement("div");L.className="text-[10px] t\
ext-gray-500 px-1 pb-1 font-bold",L.textContent="\u25BC "+(TAB_LABELS[p.tabId]||p.tabId),r.appendChild(
L)}d=p.tabId}const h=document.createElement("div");h.className="settings-search-result-item flex ite\
ms-start gap-2.5 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-150",h.style.animation=
"fadeIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both",h.style.animationDelay=g*30+"ms";const v=document.
createElement("span");v.className="settings-result-tab-badge shrink-0 mt-0.5",v.textContent=TAB_LABELS[p.
tabId]||p.tabId;const b=document.createElement("div");b.className="min-w-0 flex-1";const w=document.
createElement("div");w.className="text-sm font-bold text-white truncate",w.textContent=p.title;const x=document.
createElement("div");x.className="text-[11px] text-gray-400 truncate mt-0.5",x.textContent=p.snippet,
b.appendChild(w),b.appendChild(x),h.appendChild(v),h.appendChild(b),h.addEventListener("click",()=>jumpToSetting(
p.tabId,p.element)),r.appendChild(h)})}a.insertBefore(r,a.firstChild)}o(filterSettings,"filterSettin\
gs");function jumpToSetting(e,t){const n=get("settings-search");n&&(n.value=""),removeSearchOverlays(),
filterSettings(),e!==activeSettingsTab&&switchTab(e),setTimeout(()=>{t.scrollIntoView({behavior:"smo\
oth",block:"center"}),t.classList.add("settings-jump-highlight"),setTimeout(()=>t.classList.remove("\
settings-jump-highlight"),2e3)},260)}o(jumpToSetting,"jumpToSetting");function clickTab(e){const t=get(
"settings-search");t&&(t.value=""),switchTab(e)}o(clickTab,"clickTab");function switchTab(e){if(e===
activeSettingsTab||!ALL_TABS.includes(e))return;const t=get("tab-"+activeSettingsTab);t&&(t.classList.
remove("tab-enter"),t.classList.add("tab-exit"),setTimeout(()=>{t.classList.add("hidden"),t.classList.
remove("tab-exit")},170)),ALL_TABS.forEach(n=>{const i=get("btn-tab-"+n),s=get("tab-"+n);if(n===e){if(s&&
(s.classList.remove("hidden"),s.classList.remove("tab-exit"),s.classList.remove("tab-enter"),s.offsetWidth,
s.classList.add("tab-enter")),i){i.classList.add("is-active");try{i.scrollIntoView({inline:"nearest",
block:"nearest",behavior:"smooth"})}catch{}}}else i&&i.classList.remove("is-active")}),activeSettingsTab=
e,filterSettings(),refreshSettingsTabsScroll()}o(switchTab,"switchTab");function getSettingsTabsMaxScroll(e){
return e?Math.max(0,e.scrollWidth-e.clientWidth):0}o(getSettingsTabsMaxScroll,"getSettingsTabsMaxScr\
oll");function syncSettingsTabsOverflow(){const e=get("settings-tabs-wrap"),t=get("settings-tabs"),n=get(
"settings-tabs-arrow-left"),i=get("settings-tabs-arrow-right");if(!e||!t)return;const s=getSettingsTabsMaxScroll(
t),a=t.scrollLeft,r=s>2&&a>2,l=s>2&&a<s-2;e.classList.toggle("can-scroll",s>2),e.classList.toggle("c\
an-scroll-left",r),e.classList.toggle("can-scroll-right",l),n&&(n.disabled=!r,n.setAttribute("aria-h\
idden",r?"false":"true")),i&&(i.disabled=!l,i.setAttribute("aria-hidden",l?"false":"true"))}o(syncSettingsTabsOverflow,
"syncSettingsTabsOverflow");function refreshSettingsTabsScroll(){initSettingsTabsScroll(),syncSettingsTabsOverflow()}
o(refreshSettingsTabsScroll,"refreshSettingsTabsScroll");function initSettingsTabsScroll(){const e=get(
"settings-tabs-wrap"),t=get("settings-tabs"),n=get("settings-tabs-arrow-left"),i=get("settings-tabs-\
arrow-right");if(!e||!t||!n||!i)return;if(e.dataset.scrollBound==="1"){syncSettingsTabsOverflow();return}
e.dataset.scrollBound="1";const s=56;let a=0,r=0,l=0;const d=o(b=>{const w=e.getBoundingClientRect();
if(!w.width)return;const x=b-w.left;e.classList.toggle("is-edge-left",x>=0&&x<=s),e.classList.toggle(
"is-edge-right",x>=w.width-s&&x<=w.width)},"updateEdgeHover"),p=o(()=>{l||e.classList.remove("is-edg\
e-left","is-edge-right")},"clearEdgeHover"),g=o((b,w)=>{const x=getSettingsTabsMaxScroll(t);if(x<=0||
!b)return;const L=Math.max(0,Math.min(x,t.scrollLeft+b));w&&typeof t.scrollTo=="function"?t.scrollTo(
{left:L,behavior:"smooth"}):t.scrollLeft=L,syncSettingsTabsOverflow()},"scrollTabsBy"),h=o(()=>{l=0,
a&&(clearTimeout(a),a=0),r&&(cancelAnimationFrame(r),r=0)},"stopHold"),v=o(b=>{h(),l=b,e.classList.toggle(
"is-edge-left",b<0),e.classList.toggle("is-edge-right",b>0),g(b*Math.max(120,t.clientWidth*.55),!0),
a=setTimeout(()=>{const w=o(()=>{l&&(g(l*14,!1),r=requestAnimationFrame(w))},"step");r=requestAnimationFrame(
w)},280)},"startHold");if(e.addEventListener("pointermove",b=>{b.pointerType!=="touch"&&d(b.clientX)}),
e.addEventListener("pointerenter",b=>{b.pointerType!=="touch"&&d(b.clientX)}),e.addEventListener("po\
interleave",b=>{b.pointerType!=="touch"&&(h(),p())}),e.addEventListener("wheel",b=>{const w=getSettingsTabsMaxScroll(
t);if(w<=2)return;const L=Math.abs(b.deltaY)>=Math.abs(b.deltaX)?b.deltaY:b.deltaX;if(!L)return;const T=Math.
max(0,Math.min(w,t.scrollLeft+L));T!==t.scrollLeft&&(b.preventDefault(),t.scrollLeft=T,syncSettingsTabsOverflow())},
{passive:!1}),n.addEventListener("pointerdown",b=>{b.button!=null&&b.button!==0||(b.preventDefault(),
v(-1))}),i.addEventListener("pointerdown",b=>{b.button!=null&&b.button!==0||(b.preventDefault(),v(1))}),
n.addEventListener("click",b=>{b.preventDefault(),b.stopPropagation()}),i.addEventListener("click",b=>{
b.preventDefault(),b.stopPropagation()}),window.addEventListener("pointerup",h),window.addEventListener(
"pointercancel",h),window.addEventListener("blur",h),t.addEventListener("scroll",syncSettingsTabsOverflow,
{passive:!0}),window.addEventListener("resize",syncSettingsTabsOverflow),typeof ResizeObserver!="und\
efined")try{const b=new ResizeObserver(()=>syncSettingsTabsOverflow());b.observe(t),b.observe(e)}catch{}
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
forEach(s=>{s.nodeType===Node.ELEMENT_NODE&&s.parentElement===chatContainer&&e.observe(s)})}),scrollToBottom()}).
observe(chatContainer,{childList:!0,subtree:!0,characterData:!0})}scrollToBottomBtn&&scrollToBottomBtn.
addEventListener("click",()=>scrollToBottom(!0)),document.addEventListener("keydown",e=>{const t=e.target,
n=t&&(t.matches("input, textarea, select")||t.isContentEditable);!n&&["ArrowUp","PageUp","Home"].includes(
e.key)?armChatAutoScrollPause():!n&&chatManualScrollPaused&&["ArrowDown","PageDown","End"].includes(
e.key)&&(chatManualResumeArmed=!0)});let viewerImages=[],viewerIndex=0,viewerSwipe=null,suppressViewerCloseClick=!1;
function openImageViewer(e,t=".chat-image"){const i=Array.from(document.querySelectorAll(t)).map(a=>({
url:a.dataset.viewerSrc||a.currentSrc||a.src,filename:a.dataset.viewerFilename||a.title||(a.dataset.
viewerSrc||a.currentSrc||a.src).split("/").pop(),element:a})),s=i.findIndex(a=>a.url===e);if(s===-1){
openViewerWithItems([{url:e,filename:e.split("/").pop(),element:null}],0);return}openViewerWithItems(
i,s)}o(openImageViewer,"openImageViewer");function openViewerWithItems(e,t){viewerImages=e,viewerIndex=
t>=0&&t<e.length?t:0,clearViewerAdjacent(),updateViewerState(),get("image-viewer").classList.add("vi\
sible"),document.addEventListener("keydown",handleViewerKeydown)}o(openViewerWithItems,"openViewerWi\
thItems");function closeImageViewer(){get("image-viewer").classList.remove("visible"),document.removeEventListener(
"keydown",handleViewerKeydown),clearViewerAdjacent(),viewerImages=[],viewerIndex=0,viewerSwipe=null}
o(closeImageViewer,"closeImageViewer");function clearViewerAdjacent(){const e=document.querySelector(
".viewer-adjacent");e&&e.remove()}o(clearViewerAdjacent,"clearViewerAdjacent");function renderViewerChrome(){
if(!viewerImages.length)return;const e=get("image-viewer-meta"),t=document.querySelector(".viewer-na\
v.prev"),n=document.querySelector(".viewer-nav.next"),i=viewerImages[viewerIndex];if(e.innerText=`${viewerIndex+
1} / ${viewerImages.length} \u2022 ${i.filename}`,viewerIndex<viewerImages.length-1){const s=new Image;
s.src=viewerImages[viewerIndex+1].url}t.style.display=viewerImages.length>1?"flex":"none",n.style.display=
viewerImages.length>1?"flex":"none",t.style.opacity=viewerIndex>0?"1":"0.3",n.style.opacity=viewerIndex<
viewerImages.length-1?"1":"0.3",t.style.pointerEvents=viewerIndex>0?"auto":"none",n.style.pointerEvents=
viewerIndex<viewerImages.length-1?"auto":"none"}o(renderViewerChrome,"renderViewerChrome");function updateViewerState(e){
if(!viewerImages.length)return;const t=get("image-viewer-img");if(!t)return;const n=viewerImages[viewerIndex],
i=!e||e.fade!==!1;renderViewerChrome(),t.style.transition="none",t.style.transform=i?"scale(0.96)":"\
translateX(0) scale(1)",t.style.opacity=i?"0.35":"0";const s=o(()=>{t.style.transition=i?"transform \
0.28s var(--ease-out), opacity 0.28s var(--ease-out)":"none",t.style.opacity="1",t.style.transform="\
scale(1)",i||clearViewerAdjacent()},"reveal");i?setTimeout(()=>{viewerSwipe&&viewerSwipe.active||(t.
src=n.url,t.onload=s,t.onerror=s,t.complete&&t.naturalWidth&&s())},140):(t.src=n.url,t.onload=s,t.onerror=
s,t.complete&&t.naturalWidth&&s())}o(updateViewerState,"updateViewerState");function navImage(e){const t=viewerIndex+
e;t>=0&&t<viewerImages.length&&(clearViewerAdjacent(),viewerIndex=t,updateViewerState())}o(navImage,
"navImage");function getViewerAdjacent(e){const t=document.querySelector(".viewer-content");if(!t)return null;
const n=viewerIndex+e;if(n<0||n>=viewerImages.length)return null;let i=t.querySelector(".viewer-adja\
cent");return i||(i=document.createElement("img"),i.className="viewer-adjacent",i.alt="",t.appendChild(
i)),i.src=viewerImages[n].url,i.dataset.dir=String(e),i}o(getViewerAdjacent,"getViewerAdjacent");function onViewerTouchStart(e){
if(!viewerImages.length||e.touches.length!==1)return;const t=e.touches[0];viewerSwipe={startX:t.clientX,
startY:t.clientY,lastX:t.clientX,lastY:t.clientY,dx:0,dy:0,vx:0,dir:0,active:!1,resist:!1,adjacent:null,
lastTime:Date.now()}}o(onViewerTouchStart,"onViewerTouchStart");function onViewerTouchMove(e){if(!viewerSwipe)
return;const t=e.touches[0],n=t.clientX-viewerSwipe.startX,i=t.clientY-viewerSwipe.startY,s=Date.now(),
a=Math.max(s-viewerSwipe.lastTime,1),r=(t.clientX-viewerSwipe.lastX)/a;if(viewerSwipe.vx=r*.6+viewerSwipe.
vx*.4,viewerSwipe.lastX=t.clientX,viewerSwipe.lastY=t.clientY,viewerSwipe.lastTime=s,viewerSwipe.dx=
n,viewerSwipe.dy=i,!viewerSwipe.active){if(Math.abs(n)<10&&Math.abs(i)<10)return;if(Math.abs(n)<Math.
abs(i)*1.15){viewerSwipe=null;return}viewerSwipe.active=!0,viewerSwipe.dir=n>0?-1:1,viewerSwipe.adjacent=
getViewerAdjacent(viewerSwipe.dir),viewerSwipe.adjacent||(viewerSwipe.resist=!0)}e.preventDefault();
const l=get("image-viewer-img");if(!l)return;const d=document.querySelector(".viewer-content"),p=d?d.
clientWidth:window.innerWidth,g=viewerSwipe.resist?n*.3:n;l.style.transition="none",l.style.transform=
`translateX(${g}px) scale(${1-Math.min(Math.abs(g)/(p*4),.04)})`,l.style.opacity=String(Math.max(1-Math.
min(Math.abs(g)/(p*.45),.55),.4));const h=viewerSwipe.adjacent;if(h){const v=Number(h.dataset.dir)||
0;h.style.transition="none",h.style.transform=`translate(-50%, -50%) translateX(${v*p+n}px) scale(0.\
97)`,h.style.opacity=String(Math.min(Math.abs(n)/(p*.3),1))}}o(onViewerTouchMove,"onViewerTouchMove");
function onViewerTouchEnd(){if(!viewerSwipe)return;const e=viewerSwipe;if(viewerSwipe=null,!e.active)
return;suppressViewerCloseClick=!0,setTimeout(()=>{suppressViewerCloseClick=!1},120);const t=get("im\
age-viewer-img");if(!t)return;const n=document.querySelector(".viewer-content"),i=n?n.clientWidth:window.
innerWidth,s=i*.22,a=e.dir||(e.dx>0?-1:1),r=window.matchMedia&&window.matchMedia("(prefers-reduced-m\
otion: reduce)").matches,l=!e.resist&&(Math.abs(e.dx)>s||Math.abs(e.vx)>.45&&Math.sign(e.dx)===a),d=e.
adjacent;if(!l){if(t.style.transition="transform 0.32s var(--ease-out), opacity 0.32s var(--ease-out\
)",t.style.transform="translateX(0) scale(1)",t.style.opacity="1",d){const g=d;d.style.transition="t\
ransform 0.32s var(--ease-out), opacity 0.32s var(--ease-out)",d.style.transform=`translate(-50%, -5\
0%) translateX(${a*i}px) scale(0.97)`,d.style.opacity="0",setTimeout(()=>{g.isConnected&&g.remove()},
340)}return}if(r){finishSwipeNav(a);return}const p=a*i;t.style.transition="transform 0.3s var(--ease\
-out), opacity 0.3s var(--ease-out)",t.style.transform=`translateX(${p}px) scale(0.96)`,t.style.opacity=
"0.2",d&&(d.style.transition="transform 0.3s var(--ease-out), opacity 0.3s var(--ease-out)",d.style.
transform="translate(-50%, -50%) translateX(0) scale(1)",d.style.opacity="1"),setTimeout(()=>finishSwipeNav(
a),300)}o(onViewerTouchEnd,"onViewerTouchEnd");function finishSwipeNav(e){if(!viewerImages.length||viewerSwipe&&
viewerSwipe.active)return;const t=get("image-viewer");if(!t||!t.classList.contains("visible")){clearViewerAdjacent();
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
t();else throw new Error("Clipboard API unavailable")}catch(i){try{const s=document.createElement("t\
extarea");s.value=e,s.style.position="fixed",s.style.left="-9999px",document.body.appendChild(s),s.focus(),
s.select();const a=document.execCommand("copy");document.body.removeChild(s),a?t&&t():n&&n(i)}catch(s){
n&&n(s)}}}o(copyToClipboard,"copyToClipboard");const isQuoteMobileLayout=o(()=>window.matchMedia("(m\
ax-width: 768px)").matches,"isQuoteMobileLayout");let quotePreviewText="";function showQuotePreview(e){
const t=get("quote-bar");quotePreviewText=e,t.classList.contains("preview")||(currentQuote="",t.classList.
add("preview")),get("quote-text-display").innerText=e,t.classList.add("visible"),schedulePromptTokenEstimate()}
o(showQuotePreview,"showQuotePreview");function handleQuotePopover(){const e=window.getSelection(),t=get(
"quote-popover");if(!t)return;const n=isQuoteMobileLayout();if(!e||e.rangeCount===0){t.style.display=
"none",t.classList.remove("show");return}const i=e.toString().trim();if(i.length>0&&get("chat-contai\
ner").contains(e.anchorNode)){if(n){showQuotePreview(i);return}const a=e.getRangeAt(0).getBoundingClientRect(),
r=t.style.display==="none"||!t.style.display||getComputedStyle(t).display==="none";t.style.display="\
block",t.style.top=a.top-40+"px",t.style.left=a.left+"px",r&&(t.classList.remove("show"),t.offsetWidth,
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
ni-3.7-flash",implementedAt:"2026-08-14",implementedRank:8e3,quickEmoji:"\u26A1",name:"Gemini 3.7 Fl\
ash",desc:"Most capable Flash model for complex coding, agentic workflows, and multimodal tasks.",price:"\
In $0.75/1M, Out $3.75/1M (introductory)",agenticView:!0},{id:"gemini-3.6-flash",implementedAt:"2026\
-07-30",implementedRank:6411,quickEmoji:"\u26A1",name:"Gemini 3.6 Flash",desc:"Latest Flash model fo\
r agentic, coding, and multimodal tasks.",price:"In $1.50/1M, Out $7.50/1M",agenticView:!0},{id:"gem\
ini-3.5-flash",implementedAt:"2026-06-13",implementedRank:5900,quickEmoji:"\u2728",name:"Gemini 3.5 \
Flash",desc:"Most intelligent Gemini 3.5 model built for speed.",price:"In $1.50/1M, Out $9.00/1M",agenticView:!0},
{id:"gemini-3.5-flash-lite",implementedAt:"2026-07-30",implementedRank:6410,quickEmoji:"\u{1F680}",name:"\
Gemini 3.5 Flash-Lite",desc:"Fastest, lowest-cost Gemini 3.5 model for high-throughput execution.",price:"\
In $0.30/1M, Out $2.50/1M",agenticView:!0}]},{category:"Gemini 3.1 / Previous",icon:"fas fa-star tex\
t-yellow-400",description:"Previous Gemini 3.x generation models",items:[{id:"gemini-3.1-flash-lite",
implementedAt:"2026-07-30",implementedRank:6440,quickEmoji:"\u{1F4A8}",name:"Gemini 3.1 Flash-Lite",
desc:"Stable, cost-efficient model for high-volume lightweight tasks.",price:"In $0.25/1M, Out $1.50\
/1M",agenticView:!0},{id:"gemini-3.1-pro-preview",implementedAt:"2026-02-20",implementedRank:2430,name:"\
Gemini 3.1 Pro",desc:"Next-gen native multimodal model.",price:"In $2.00/1M, Out $12.00/1M (\u2264200k)"},
{id:"gemini-3.1-flash-lite-preview",implementedAt:"2026-03-04",implementedRank:3e3,name:"Gemini 3.1 \
Flash-Lite Preview",desc:"Retired preview model retained for chat history compatibility.",price:"In \
$0.25/1M, Out $1.50/1M",deprecated:!0},{id:"gemini-3-flash-preview",implementedAt:"2026-06-13",implementedRank:5930,
name:"Gemini 3.0 Flash",desc:"Fastest and most cost-efficient.",price:"In $0.50/1M, Out $3.00/1M"},{
id:"gemini-3-pro-preview",implementedAt:"2026-01-15",implementedRank:100,name:"Gemini 3.0 Pro",desc:"\
Shut down by Google (March 2026). Retained for chat history compatibility.",price:"In $2.00/1M, Out \
$12.00/1M (\u2264200k)",deprecated:!0}]},{category:"Gemini 2.5",icon:"fas fa-history text-gray-400",
description:"Gemini 2.5 generation models",items:[{id:"gemini-2.5-pro",implementedAt:"2026-08-25",implementedRank:8524,
quickEmoji:"\u{1F9E0}",name:"Gemini 2.5 Pro",desc:"Most advanced Gemini 2.5 model for complex reason\
ing, coding, and long-context analysis.",price:"In $1.25/1M (\u2264200k), Out $10.00/1M (\u2264200k)"},
{id:"gemini-2.5-flash-lite",implementedAt:"2026-02-07",implementedRank:1530,name:"Gemini 2.5 Flash-L\
ite",desc:"Fastest and most cost-efficient Gemini 2.5 model.",price:"In $0.10/1M, Out $0.40/1M"},{id:"\
gemini-2.5-flash",implementedAt:"2026-02-07",implementedRank:1531,name:"Gemini 2.5 Flash",desc:"Bala\
nced performance.",price:"In $0.30/1M, Out $2.50/1M"}]},{category:"Gemini Image (Banana)",icon:"fas \
fa-image text-pink-400",description:"Gemini image generation models",items:[{id:"gemini-2.5-flash-im\
age",implementedAt:"2026-01-20",implementedRank:120,quickEmoji:"\u{1F34C}",name:"Nano Banana",desc:"\
Fast image generation.",price:"In $0.30/1M, Out $0.039/image"},{id:"gemini-3.1-flash-image",implementedAt:"\
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
agship models",items:[{id:"gpt-5.6-sol",implementedAt:"2026-07-31",implementedRank:6550,quickEmoji:"\
\u2600\uFE0F",name:"GPT-5.6 Sol",desc:"Frontier reasoning model for complex professional work with 1\
.05M context.",price:"In $5.00/1M, Cached $0.50/1M, Out $30.00/1M (over 272K: In $10.00, Out $45.00)"},
{id:"gpt-5.6-terra",implementedAt:"2026-07-31",implementedRank:6560,quickEmoji:"\u{1F30D}",name:"GPT\
-5.6 Terra",desc:"Balanced intelligence and cost for everyday work with 1.05M context.",price:"In $2\
.00/1M, Cached $0.20/1M, Out $12.00/1M (over 272K: In $4.00, Out $18.00)"},{id:"gpt-5.6-luna",implementedAt:"\
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
l and efficient.",price:"In $0.25/1M, Out $2.00/1M"}]},{category:"DeepSeek V4",icon:"fas fa-bolt tex\
t-cyan-400",description:"DeepSeek's OpenAI-compatible text models",items:[{id:"deepseek-v4-flash-vis\
ion-exp",implementedAt:"2026-08-23",implementedRank:8260,quickEmoji:"\u{1F441}\uFE0F",name:"DeepSeek\
 V4 Flash Vision Exp",desc:"Experimental V4 Flash with native image input (JPEG/PNG/GIF/WebP), 1M co\
ntext, up to 384K output, thinking, tools, and JSON output.",price:"In $0.007/1M (hit), $0.22/1M (mi\
ss), Out $0.66/1M (off-peak)"},{id:"deepseek-v4-flash-0731",implementedAt:"2026-07-31",implementedRank:6610,
quickEmoji:"\u26A1",apiId:"deepseek-v4-flash",name:"DeepSeek V4 Flash",desc:"Official V4 Flash relea\
se with 1M context, up to 384K output, thinking, tools, and JSON output.",price:"In $0.0028/1M (hit)\
, $0.14/1M (miss), Out $0.28/1M"},{id:"deepseek-v4-flash",implementedAt:"2026-04-26",implementedRank:4510,
name:"DeepSeek V4 Flash Preview",desc:"Retired preview key retained for chat history compatibility.",
price:"Legacy preview",deprecated:!0},{id:"deepseek-v4-pro",implementedAt:"2026-04-26",implementedRank:4511,
name:"DeepSeek V4 Pro",desc:"Higher-capacity DeepSeek V4 model with 1M context and up to 384K output\
.",price:"In $0.003625/1M (hit), $0.435/1M (miss), Out $0.87/1M"}]},{category:"Kimi K3",icon:"fas fa\
-brain text-violet-400",description:"Moonshot AI's flagship 2.8T-parameter model with 1M context and\
 always-on thinking",items:[{id:"kimi-k3",implementedAt:"2026-07-30",implementedRank:6340,quickEmoji:"\
\u{1F9E0}",name:"Kimi K3",desc:"Always-reasoning flagship model with 1M context, vision, tool callin\
g.",price:"In $3.00/1M (miss), $0.30/1M (hit), Out $15.00/1M"}]},{category:"Mistral Document OCR",icon:"\
fas fa-file text-orange-300",description:"Document OCR (PDF / image / DOCX / PPTX). Not a chat compl\
etion model.",items:[{id:"mistral-ocr-4-0",implementedAt:"2026-08-15",implementedRank:8130,quickEmoji:"\
\u{1F4C4}",name:"Mistral OCR 4",desc:"Document AI OCR with markdown, tables, headers/footers, and pa\
ragraph bounding boxes. Chat history is not sent.",price:"$4 / 1,000 pages ($5 / 1,000 annotated pag\
es)"}]},{category:"Anthropic Claude",icon:"fas fa-brain text-orange-400",description:"Anthropic's la\
test deep reasoning models",items:[{id:"claude-opus-4-6",implementedAt:"2026-05-01",implementedRank:480,
name:"Claude Opus 4.6",desc:"Most capable model for deep reasoning and complex tasks.",price:"In $5.\
00/1M, Out $25.00/1M"},{id:"claude-sonnet-4-6",implementedAt:"2026-05-01",implementedRank:481,name:"\
Claude Sonnet 4.6",desc:"Excellent balance of speed and intelligence with adaptive thinking.",price:"\
In $3.00/1M, Out $15.00/1M"}]},{category:"Audio (TTS)",icon:"fas fa-microphone text-red-400",description:"\
Text-to-Speech models",items:[{id:"gemini-3.1-flash-tts-preview",implementedAt:"2026-04-17",implementedRank:4250,
name:"Gemini 3.1 Flash TTS",desc:"Google TTS (Preview).",price:"Text In $1.00/1M, Audio Out $20.00/1\
M"},{id:"gpt-4o-mini-tts",implementedAt:"2026-03-01",implementedRank:250,name:"GPT-4o Mini TTS",desc:"\
OpenAI TTS.",price:"Text In $0.60/1M, Audio Out $12.00/1M"},{id:"gemini-2.5-flash-preview-tts",implementedAt:"\
2026-02-10",implementedRank:160,name:"Gemini 2.5 Flash TTS",desc:"Google TTS (Preview).",price:"Text\
 In $0.50/1M, Audio Out $10.00/1M"},{id:"gemini-2.5-pro-preview-tts",implementedAt:"2026-02-10",implementedRank:161,
name:"Gemini 2.5 Pro TTS",desc:"Google TTS Pro (Preview).",price:"Text In $1.00/1M, Audio Out $20.00\
/1M"},{id:"google-tts-studio",implementedAt:"2026-01-20",implementedRank:110,name:"Google TTS (Studi\
o)",desc:"High fidelity studio voices.",price:"$160 / 1M chars"},{id:"google-tts-neural",implementedAt:"\
2026-01-20",implementedRank:111,name:"Google TTS (Neural2)",desc:"Standard neural voices.",price:"$1\
6 / 1M chars"},{id:"grok-tts",implementedAt:"2026-05-27",implementedRank:5560,quickEmoji:"\u{1F50A}",
name:"Grok TTS",desc:"xAI Text-to-Speech with expressive voices.",price:"$15.00 / 1M chars"}]},{category:"\
OpenAI Transcription",icon:"fas fa-closed-captioning text-emerald-400",description:"Speech-to-text m\
odels (audio in / text out)",items:[{id:"gpt-transcribe",implementedAt:"2026-07-29",implementedRank:6330,
name:"GPT Transcribe",desc:"High-accuracy file and committed-turn transcription.",price:"$0.0045 / m\
inute"},{id:"gpt-live-transcribe",implementedAt:"2026-07-29",implementedRank:6331,name:"GPT Live Tra\
nscribe",desc:"Low-latency realtime transcription.",price:"$0.017 / minute"}]},{category:"Realtime A\
udio (STS)",icon:"fas fa-headset text-cyan-400",description:"Realtime voice models (audio in / audio\
 out)",items:[{id:"gpt-realtime-2",implementedAt:"2026-05-11",implementedRank:5080,name:"OpenAI Real\
time 2",desc:"Most capable speech-to-speech reasoning model.",price:"Audio In $32/1M, Audio Out $64/\
1M"},{id:"gpt-realtime-translate",implementedAt:"2026-05-11",implementedRank:5081,name:"OpenAI Realt\
ime Translate",desc:"Streaming speech-to-speech translation.",price:"$0.034 / minute"},{id:"gpt-real\
time-whisper",implementedAt:"2026-05-11",implementedRank:5082,name:"OpenAI Realtime Whisper",desc:"S\
treaming speech-to-text (transcription).",price:"$0.017 / minute"},{id:"gpt-realtime-1.5",implementedAt:"\
2026-02-24",implementedRank:2530,name:"OpenAI Realtime 1.5",desc:"Latest OpenAI speech-to-speech fla\
gship model.",price:"Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime",implementedAt:"2026-02-2\
4",implementedRank:2531,name:"OpenAI Realtime",desc:"OpenAI realtime speech-to-speech model.",price:"\
Audio In $32/1M, Audio Out $64/1M"},{id:"gpt-realtime-mini",implementedAt:"2026-02-24",implementedRank:2532,
name:"OpenAI Realtime Mini",desc:"Lower-latency, smaller realtime model.",price:"Audio In $10/1M, Au\
dio Out $20/1M"},{id:"gemini-2.5-flash-native-audio-preview-12-2025",implementedAt:"2026-01-15",implementedRank:90,
name:"Gemini 2.5 Flash Native Audio (Live)",desc:"Google Live native audio model.",price:"Audio In $\
3.00/1M, Audio Out $12.00/1M"},{id:"gemini-3.1-flash-live-preview",implementedAt:"2026-03-29",implementedRank:3870,
name:"Gemini 3.1 Flash Live",desc:"Google Live native audio model.",price:"Audio In $3.00/1M (~$0.00\
5/min), Out $12.00/1M"},{id:"gemini-3.5-live-translate-preview",implementedAt:"2026-08-25",implementedRank:8523,
quickEmoji:"\u{1F310}",name:"Gemini 3.5 Live Translate",desc:"Low-latency real-time speech-to-speech\
 translation supporting 70+ languages.",price:"Audio In $3.50/1M, Audio Out $21.00/1M"},{id:"grok-vo\
ice-think-fast-2.0",implementedAt:"2026-08-25",implementedRank:8502,quickEmoji:"\u{1F3A4}",name:"Gro\
k Voice Think Fast 2.0",desc:"Current xAI speech-to-speech model.",price:"$0.08 / min ($4.80 / hr) a\
udio + $0.004 / text input"},{id:"grok-voice-latest",implementedAt:"2026-05-27",implementedRank:5550,
name:"Grok Voice Latest",desc:"Alias for the current flagship voice model.",price:"$0.08 / min ($4.8\
0 / hr) audio + $0.004 / text input"},{id:"grok-voice-think-fast-1.0",implementedAt:"2026-05-11",implementedRank:5140,
name:"Grok Voice Think Fast 1.0",desc:"Deprecated xAI realtime voice model retained for history comp\
atibility.",price:"$0.05 / min ($3.00 / hr)",deprecated:!0},{id:"grok-voice-fast-1.0",implementedAt:"\
2026-05-01",implementedRank:500,name:"Grok Voice Fast 1.0",desc:"Legacy xAI realtime voice model ret\
ained for history compatibility.",price:"$0.05 / min ($3.00 / hr)",deprecated:!0},{id:"grok-voice-ag\
ent",implementedAt:"2026-04-01",implementedRank:380,name:"Grok Voice Agent",desc:"xAI realtime voice\
 agent API.",price:"$0.05 / min (Realtime)",deprecated:!0}]},{category:"Gemini Agent / Specialized",
icon:"fas fa-robot text-indigo-400",description:"Gemini agent and specialized models",items:[{id:"ge\
mini-robotics-er-2-preview",implementedAt:"2026-08-25",implementedRank:8515,name:"Gemini Robotics ER\
 2",desc:"Embodied reasoning model for robots with advanced video understanding.",price:"In $2.00/1M\
, Out $8.00/1M"},{id:"deep-research-preview-04-2026",implementedAt:"2026-08-25",implementedRank:8514,
quickEmoji:"\u{1F50E}",name:"Gemini Deep Research",desc:"Agentic multi-step research producing compr\
ehensive cited reports.",price:"Standard Gemini rates + tool usage fees"},{id:"deep-research-max-pre\
view-04-2026",implementedAt:"2026-08-25",implementedRank:8513,name:"Gemini Deep Research Max",desc:"\
Maximum-comprehension research agent over hundreds of sources.",price:"Standard Gemini rates + tool \
usage fees"},{id:"antigravity-preview-05-2026",implementedAt:"2026-08-25",implementedRank:8512,name:"\
Antigravity Agent",desc:"Managed agent that plans, runs code, manages files, and browses the web in \
a sandbox.",price:"Standard Gemini rates (sandbox compute free during preview)"},{id:"gemini-2.5-com\
puter-use-preview-10-2025",implementedAt:"2026-08-25",implementedRank:8511,name:"Gemini 2.5 Computer\
 Use",desc:"Browser / desktop control agent model for UI automation.",price:"In $1.25/1M (\u2264200k), Ou\
t $10.00/1M (\u2264200k)"},{id:"gemini-embedding-2",implementedAt:"2026-08-25",implementedRank:8510,
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
""),i=String(t&&t.implementedAt||"");if(n!==i)return i.localeCompare(n);const s=Number(e&&e.implementedRank||
0),a=Number(t&&t.implementedRank||0);return s!==a?a-s:String(e&&e.id||"").localeCompare(String(t&&t.
id||""))},"compareModelsByImplementedAt"),getRecentModelsForQuickStart=o((e=WELCOME_QUICK_START_LIMIT)=>listModelsFlat().
filter(t=>t&&t.id&&!t.deprecated&&t.implementedAt).sort(compareModelsByImplementedAt).slice(0,Math.max(
0,Number(e)||0)),"getRecentModelsForQuickStart"),renderWelcomeQuickStart=o(()=>{const e=get("welcome\
-quick-start");if(!e)return;const t=getRecentModelsForQuickStart(WELCOME_QUICK_START_LIMIT);if(!t.length){
e.innerHTML="";return}e.innerHTML=t.map((n,i)=>{const s=(.1+i*.02).toFixed(2),a=n.quickEmoji?`${escapeHtml(
String(n.quickEmoji))} `:"",r=escapeHtml(String(n.name||n.id)),l=String(n.id).replace(/\\/g,"\\\\").
replace(/'/g,"\\'");return`<button type="button" class="welcome-btn p-3 rounded text-sm text-left tr\
ansition btn-hover slide-in-animate" style="animation-delay: ${s}s" onclick="quickStart('${l}')">${a}${r}\
</button>`}).join("")},"renderWelcomeQuickStart"),normalizeModelApiKeyMap=o(e=>{if(!e||typeof e!="ob\
ject")return{};const t={};return Object.entries(e).forEach(([n,i])=>{const s=String(n||"").trim(),a=String(
i||"").trim();!s||!a||(t[s]=a)}),t},"normalizeModelApiKeyMap"),MODEL_NAME_BY_ID=(()=>{const e=new Map;
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
e.appendChild(n),MODELS.forEach(i=>{const s=Array.isArray(i.items)?i.items.filter(r=>!r.deprecated):
[];if(!s.length)return;const a=document.createElement("optgroup");a.label=String(i.category||"Models"),
s.forEach(r=>{const l=String(r.id||"").trim();if(!l)return;const d=document.createElement("option");
d.value=l,d.textContent=`${String(r.name||l)} (${l})`,a.appendChild(d)}),a.children.length>0&&e.appendChild(
a)}),t&&Array.from(e.options).some(s=>s.value===t)&&(e.value=t)},"syncModelApiKeyModelOptions"),renderModelApiKeyList=o(
()=>{const e=get("model-api-key-list");if(!e)return;modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap);
const t=Object.entries(modelApiKeyMap).sort((n,i)=>n[0].localeCompare(i[0]));if(e.innerHTML="",!t.length){
const n=document.createElement("div");n.className="text-[11px] text-gray-500",n.textContent="\u30E2\u30C7\u30EB\u5225\u30AD\u30FC\u306F\
\u672A\u8A2D\u5B9A\u3067\u3059\u3002",e.appendChild(n);return}t.forEach(([n,i])=>{const s=document.createElement(
"div");s.className="flex items-center justify-between gap-3 rounded border border-gray-700 bg-gray-9\
00/70 px-3 py-2";const a=document.createElement("div");a.className="min-w-0";const r=document.createElement(
"div");r.className="text-[11px] text-gray-200 truncate",r.textContent=`${getModelNameById(n)} (${n})`;
const l=document.createElement("div");l.className="text-[10px] text-cyan-300 font-mono",l.textContent=
maskApiKeyPreview(i),a.appendChild(r),a.appendChild(l);const d=document.createElement("button");d.type=
"button",d.className="text-[10px] bg-red-700/80 hover:bg-red-600 text-white px-2 py-1 rounded font-b\
old btn-hover shrink-0",d.textContent="\u524A\u9664",d.onclick=()=>{delete modelApiKeyMap[n],renderModelApiKeyList(),
showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u524A\u9664: ${n}`,"success")},s.appendChild(
a),s.appendChild(d),e.appendChild(s)})},"renderModelApiKeyList"),bindModelApiKeySettingsControls=o(()=>{
const e=get("toggle-model-api-keys-btn");e&&!e.dataset.bound&&(e.dataset.bound="1",e.addEventListener(
"click",()=>{const i=get("model-api-keys-panel");setModelApiKeyPanelOpen(i?i.classList.contains("hid\
den"):!0)}));const t=get("model-api-key-apply-btn");t&&!t.dataset.bound&&(t.dataset.bound="1",t.addEventListener(
"click",()=>{const i=get("model-api-key-model"),s=get("model-api-key-input"),a=i?String(i.value||"").
trim():"",r=s?String(s.value||"").trim():"";if(!a){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!r){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}modelApiKeyMap=normalizeModelApiKeyMap(modelApiKeyMap),modelApiKeyMap[a]=r,s&&(s.
value=""),renderModelApiKeyList(),showToast(`\u30E2\u30C7\u30EB\u5225API\u30AD\u30FC\u3092\u8A2D\u5B9A: ${a}`,
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
s=n.map(([a,r])=>`${a}: ${formatAiSettingValue(r).slice(0,180)}`).join(`
`);return`${i}${s?`
${s}`:""}`.slice(0,1600)}o(summarizeAiSettingsConversationValues,"summarizeAiSettingsConversationVal\
ues");let gemSuggestionsVisible=!1,gemSelectedIndex=0;const STS_MODELS=new Set(["gpt-transcribe","gp\
t-live-transcribe","gpt-realtime-2","gpt-realtime-translate","gpt-realtime-whisper","gpt-realtime-1.\
5","gpt-realtime","gpt-realtime-mini","gemini-2.5-flash-native-audio-preview-12-2025","gemini-3.1-fl\
ash-live-preview","gemini-3.5-live-translate-preview","gemini-3.5-transcribe-live","grok-voice-think\
-fast-2.0","grok-voice-latest","grok-voice-think-fast-1.0","grok-voice-fast-1.0","grok-voice-agent"]),
FILE_BASE_URL=CHAT_CONFIG.urls.serveFileBase,FILE_THUMB_BASE_URL=CHAT_CONFIG.urls.serveFileThumbBase,
RICH_PASTE_PDF_SERVER_ROUTE=CHAT_CONFIG.urls.richPastePdfServer,IMAGE_EXTS=["png","jpg","jpeg","webp",
"gif","bmp","avif","heic","heif"],AUDIO_EXTS=["mp3","wav","aac","ogg","flac","aiff","aif","m4a","opu\
s","oga","weba","webm"],VIDEO_EXTS=["mp4","mov","avi","mkv","m4v","webm","mpg","mpeg","wmv","3gp","3\
gpp","flv"],getFileExt=o(e=>{const t=typeof e=="string"?e:e==null?"":String(e);if(!t)return"";const n=t.
lastIndexOf(".");return n===-1?"":t.slice(n+1).toLowerCase()},"getFileExt"),normalizeAttachmentPath=o(
e=>{if(!e)return"";let t="";if(typeof e=="string"?t=e:typeof e=="object"&&(t=String(e.path||e.url||e.
name||e.filename||e.filepath||"")),!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"normalizeAttachmentPath"),isGeminiImageModelKey=o(e=>{const t=(e||"").toLowerCase();return t.includes(
"gemini")&&(t.includes("image")||t.includes("nano"))},"isGeminiImageModelKey"),isClaudeModelKey=o(e=>(e||
"").toLowerCase().includes("claude"),"isClaudeModelKey"),getModelApiProvider=o(e=>{const t=String(e||
"").toLowerCase().trim();return t?t.includes("claude")?"anthropic":t.includes("deepseek")?"deepseek":
t.includes("grok")&&!t.includes("gpt")?"xai":t.includes("google-tts")?"google":t.includes("gemini")||
t.startsWith("veo-")||t.startsWith("lyria-")||t.startsWith("deep-research-")||t.startsWith("antigrav\
ity-")?"gemini":"openai":null},"getModelApiProvider"),PROVIDER_LABELS={openai:"OpenAI",gemini:"Gemin\
i",anthropic:"Anthropic (Claude)",xai:"xAI (Grok)",deepseek:"DeepSeek",google:"Google Cloud"},isPromptCacheEnabled=o(
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
()=>{const e=get("model-select").value;return e==="gemini-3.1-flash-live-preview"||e==="gemini-3.5-l\
ive-translate-preview"||e==="gemini-3.5-transcribe-live"},"isGeminiLiveModel"),isGeminiLiveTranslateModel=o(
()=>get("model-select").value==="gemini-3.5-live-translate-preview","isGeminiLiveTranslateModel"),isGeminiLiveTranscribeModel=o(
()=>get("model-select").value==="gemini-3.5-transcribe-live","isGeminiLiveTranscribeModel"),isGeminiRealtimeMusicModel=o(
()=>(get("model-select").value||"")==="lyria-realtime-exp","isGeminiRealtimeMusicModel"),isLyriaRealtimeModel=o(
()=>isGeminiRealtimeMusicModel(),"isLyriaRealtimeModel"),isRealtimeSessionModel=o(()=>!(!isStsModel()||
isGeminiLiveModel()||isTranscriptionModel()||get("model-select")&&get("model-select").value==="gpt-r\
ealtime-whisper"),"isRealtimeSessionModel"),getStsProvider=o(e=>{const t=(e||"").toLowerCase();return t.
includes("gpt-realtime")||t==="gpt-transcribe"||t==="gpt-live-transcribe"?"openai":t.includes("grok-\
voice")?"xai":t.includes("gemini")&&(t.includes("native-audio")||t.includes("live"))?"gemini":null},
"getStsProvider");function setStsStatus(e,t=!1){const n=get("sts-status"),i=get("sts-mic-btn");n&&e&&
(n.innerText=e),i&&(t?(i.classList.add("bg-red-600","animate-pulse"),i.classList.remove("bg-cyan-600")):
(i.classList.remove("bg-red-600","animate-pulse"),i.classList.add("bg-cyan-600")))}o(setStsStatus,"s\
etStsStatus");function updateStsUi(){const e=isStsModel(),t=e&&voiceStudioUiEnabled!==!1,n=get("inpu\
t-row"),i=get("sts-panel"),s=get("voice-studio-bar"),a=get("file-preview");e?(n&&n.classList.add("hi\
dden"),a&&a.classList.add("hidden"),t?(i&&(window.VoiceStudioOpen?i.classList.remove("hidden"):i.classList.
add("hidden")),s&&s.classList.remove("hidden")):(i&&i.classList.remove("hidden"),s&&s.classList.add(
"hidden"),window.VoiceStudio&&window.VoiceStudio.closeIfOpen()),setStsStatus("Tap to speak",!1)):(n&&
n.classList.remove("hidden"),i&&i.classList.add("hidden"),s&&s.classList.add("hidden"),window.VoiceStudio&&
window.VoiceStudio.closeIfOpen())}o(updateStsUi,"updateStsUi");function updateStsOptions(){if(!isStsModel())
return;const e=get("model-select").value||"",t=getStsProvider(e),n=get("sts-voice"),i=get("sts-speed\
-wrap"),s=get("sts-speed"),a=get("sts-speed-label"),r=get("sts-rate-wrap"),l=get("sts-rate-in"),d=get(
"sts-rate-out"),p=get("sts-thinking-wrap"),g=get("sts-note"),h=get("sts-voice-wrap"),v=get("sts-auto\
-play-wrap"),b=get("sts-mode-label"),w=isTranscriptionModel()||isGeminiLiveTranscribeModel(),x=get("\
sts-lang-wrap");if(w){b&&(b.textContent="Realtime Speech-to-Text"),h&&h.classList.add("hidden"),v&&v.
classList.add("hidden"),i&&i.classList.add("hidden"),r&&r.classList.add("hidden"),p&&p.classList.add(
"hidden"),x&&x.classList.add("hidden");const L=get("sts-transcribe-wrap"),T=get("sts-custom-vocab-wr\
ap");L&&L.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),T&&T.classList.toggle("hidden",!isGeminiLiveTranscribeModel()),
g&&(g.textContent=isGeminiLiveTranscribeModel()?"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F4E\u9045\u5EF6\u6587\u5B57\u8D77\u3053\u3057\uFF0816kHz PCM / \u6700\u592710\u5206\uFF09":
e==="gpt-live-transcribe"?"\u4F4E\u9045\u5EF6\u30E9\u30A4\u30D6\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09":
"\u9AD8\u7CBE\u5EA6\u306A\u30B3\u30DF\u30C3\u30C8\u5358\u4F4D\u306E\u6587\u5B57\u8D77\u3053\u3057\uFF0824kHz PCM\uFF09")}else
t==="openai"?(b&&(b.textContent="Speech-to-Speech Live"),h&&h.classList.remove("hidden"),v&&v.classList.
remove("hidden"),setSelectOptions(n,OPENAI_STS_VOICES,n.value||"alloy"),i&&i.classList.remove("hidde\
n"),s&&(s.min=.25,s.max=1.5,s.step=.05,s.value||(s.value=1),Number(s.value)<.25&&(s.value=.25),Number(
s.value)>1.5&&(s.value=1.5)),r&&r.classList.add("hidden"),p&&p.classList.add("hidden"),x&&x.classList.
add("hidden"),g&&(g.textContent="OpenAI Realtime\u306F24kHz PCM\u56FA\u5B9A")):t==="xai"?(b&&(b.textContent=
"Speech-to-Speech Live"),h&&h.classList.remove("hidden"),v&&v.classList.remove("hidden"),setSelectOptions(
n,GROK_STS_VOICES,n.value||"Ara"),i&&i.classList.add("hidden"),r&&r.classList.remove("hidden"),p&&p.
classList.add("hidden"),x&&x.classList.add("hidden"),setSelectOptions(l,GROK_PCM_RATES,Number(l.value||
24e3)),setSelectOptions(d,GROK_PCM_RATES,Number(d.value||24e3)),g&&(g.textContent="xAI\u306FPCM\u30B5\u30F3\u30D7\u30EB\u30EC\u30FC\u30C8\u5909\u66F4\u53EF")):
t==="gemini"&&(b&&(b.textContent="Speech-to-Speech Live"),h&&h.classList.remove("hidden"),v&&v.classList.
remove("hidden"),setSelectOptions(n,GEMINI_STS_VOICES,n.value||"Kore"),i&&i.classList.add("hidden"),
r&&r.classList.add("hidden"),p&&p.classList.remove("hidden"),x&&x.classList.add("hidden"),g&&(g.textContent=
"Gemini Live\u306F\u97F3\u58F0\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC"),e==="gemini-3.5-live-tran\
slate-preview"&&(b&&(b.textContent="Realtime Translation"),p&&p.classList.add("hidden"),h&&h.classList.
add("hidden"),x&&x.classList.remove("hidden"),g&&(g.textContent="70\u4EE5\u4E0A\u306E\u8A00\u8A9E\u306B\u5BFE\u5FDC\u3059\u308B\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\uFF08Think\u975E\u5BFE\u5FDC\u30FB\u97F3\u58F0\u9078\
\u629E\u4E0D\u53EF\uFF09")));i&&a&&s&&!i.classList.contains("hidden")&&(a.textContent=`${Number(s.value||
1).toFixed(2)}x`)}o(updateStsOptions,"updateStsOptions");function stsOpt(e){const t=get(e);return e===
"sts-auto-play"||e==="sts-auto-restart"?t?!!t.checked:!0:t?!!t.checked:!1}o(stsOpt,"stsOpt");function getStsSilenceMs(){
const e=get("sts-silence-sec");let t=e?parseFloat(e.value):1.5;return(isNaN(t)||t<.5)&&(t=.5),t>10&&
(t=10),Math.round(t*1e3)}o(getStsSilenceMs,"getStsSilenceMs");function getTtsProvider(e){if(!e)return null;
const t=e.toLowerCase();return t.includes("google-tts")?"google":t.includes("gemini")&&t.includes("t\
ts")?"gemini":t.includes("grok-tts")||t.includes("xai-tts")?"xai":t.includes("tts")?"openai":null}o(
getTtsProvider,"getTtsProvider");function setSelectOptions(e,t,n){e&&(e.innerHTML="",t.forEach(i=>{const s=document.
createElement("option");s.value=i.value||i,s.textContent=i.label||i,(i.value||i)===n&&(s.selected=!0),
e.appendChild(s)}))}o(setSelectOptions,"setSelectOptions");function updateTtsUi(){const e=get("model\
-select").value||"",t=getTtsProvider(e),n=get("audio-gen-options");if(!n)return;if(!t){n.classList.add(
"hidden");return}n.classList.remove("hidden");const i=get("tts-voice"),s=get("tts-voice-custom-wrap"),
a=get("tts-voice-custom"),r=get("tts-language-wrap"),l=get("tts-language"),d=get("tts-speed-wrap"),p=get(
"tts-speed"),g=get("tts-speed-label"),h=get("tts-speed-note");t==="openai"?(setSelectOptions(i,OPENAI_TTS_VOICES,
i.value||"alloy"),s.classList.add("hidden"),r.classList.add("hidden"),p&&(p.min=.25,p.max=4,p.step=.05,
p.value||(p.value=1),Number(p.value)<.25&&(p.value=.25),Number(p.value)>4&&(p.value=4),p.disabled=!1),
h&&(h.textContent="")):t==="gemini"?(setSelectOptions(i,GEMINI_TTS_VOICES,i.value||"Kore"),s.classList.
add("hidden"),r.classList.add("hidden"),p&&(p.disabled=!0),h&&(h.textContent="(Gemini TTS\u306F\u901F\u5EA6\u5909\u66F4\u975E\u5BFE\u5FDC)")):
t==="google"?(setSelectOptions(i,[{value:"auto",label:"Auto (Studio/Neural2)"},{value:"custom",label:"\
Custom Voice Name"}],i.value||"auto"),i.value==="custom"?s.classList.remove("hidden"):(s.classList.add(
"hidden"),a&&(a.value="")),r.classList.remove("hidden"),l&&!l.value&&(l.value="ja-JP"),p&&(p.min=.25,
p.max=2,p.step=.05,p.value||(p.value=1),Number(p.value)<.25&&(p.value=.25),Number(p.value)>2&&(p.value=
2),p.disabled=!1),h&&(h.textContent="")):t==="xai"&&(setSelectOptions(i,GROK_TTS_VOICES,i.value||"Ev\
e"),s.classList.remove("hidden"),r.classList.remove("hidden"),l&&!l.value&&(l.value="ja"),p&&(p.min=
.7,p.max=1.5,p.step=.05,p.value||(p.value=1),Number(p.value)<.7&&(p.value=.7),Number(p.value)>1.5&&(p.
value=1.5),p.disabled=!1),h&&(h.textContent="xAI TTS supports speed 0.7\u20131.5 and speech tags")),
p&&g&&(g.textContent=`${Number(p.value||1).toFixed(2)}x`)}o(updateTtsUi,"updateTtsUi");let mcpServers=[],
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
!1);let n;n=(async()=>{try{const i=await apiFetch(MCP_URLS.servers());if(!i.ok){const a=await i.json().
catch(()=>({}));mcpStatusMsg("mcp-status-msg",a.error||"MCP\u30B5\u30FC\u30D0\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}const s=await i.json();mcpServers=s&&Array.isArray(s.servers)?s.servers:[],mcpLoaded=!0,renderMcpServers(),
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
mcpStatusMsg("mcp-status-msg","");return}const i=mcpServers.map((s,a)=>mcpServerCard(s,a)).join("");
e.innerHTML=i,mcpStatusMsg("mcp-status-msg","")}o(renderMcpServers,"renderMcpServers");function mcpServerCard(e,t){
const n=!!e.is_preset,i=e.auth_type==="oauth",s=e.auth_type==="bearer",a=i||s,r=i&&!e.oauth_client_registered,
l=Number(e.tool_count||0),d=l>0?`${l}\u30C4\u30FC\u30EB`:"\u30C4\u30FC\u30EB\u672A\u53D6\u5F97",p=mcpStateBadge(
e),g=n?'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-700/50 text-blue-100">\u30D7\u30EA\u30BB\u30C3\u30C8<\
/span>':'<span class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-700/50 text-purple-100">\u30AB\
\u30B9\u30BF\u30E0</span>',h=mcpAuthBlock(e),v=i?mcpOauthClientBlock(e):"";return`
<div class="rounded border border-gray-700 bg-gray-950/50 p-3" data-mcp-server="${mcpEsc(e.slug)}">
    <div class="flex items-center justify-between gap-2 flex-wrap">
        <div class="flex items-center gap-2 min-w-0">
            <i class="fas fa-plug ${e.enabled?"text-cyan-300":"text-gray-600"}"></i>
            <div class="min-w-0">
                <span class="text-xs font-bold text-white">${mcpEsc(e.name)}</span>
                ${g} ${p}
            </div>
        </div>
        <div class="flex items-center gap-1 shrink-0">
            ${a?mcpAuthActionButton(e):""}
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

    ${v}
    ${h}
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
/json"},body:JSON.stringify({enabled:t})});if(!n.ok){const s=await n.json().catch(()=>({}));mcpStatusMsg(
"mcp-status-msg",s.error||"\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}const i=await n.
json();mcpStatusMsg("mcp-status-msg",t?"\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u30C1\u30E3\u30C3\u30C8\u306E\u30E2\u30C7\u30EB\u3078\u30C4\u30FC\u30EB\u304C\u516C\u958B\u3055\u308C\u307E\u3059\u3002":
"\u7121\u52B9\u306B\u3057\u307E\u3057\u305F\u3002",!1),loadMcpServers(!0)}catch(n){mcpStatusMsg("mcp\
-status-msg","\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(n&&n.message?n.message:n),!0)}}
o(mcpToggleEnabled,"mcpToggleEnabled");async function mcpOpenAuth(e){mcpStatusMsg("mcp-status-msg","\
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
00">${mcpEsc(i.error||"\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}</div>`;return}const s=i&&
Array.isArray(i.tools)?i.tools:[];if(!s.length){t.innerHTML='<div class="text-[10px] text-gray-600">\
\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u300C\u63A5\u7D9A\u30C6\u30B9\u30C8\u300D\u3067\u53D6\u5F97\u3057\u3066\u304F\u3060\u3055\u3044\u3002</div>';
return}const a=s.map((r,l)=>`
<div class="flex items-start justify-between gap-2 py-1 border-b border-gray-800 last:border-0">
    <div class="min-w-0">
        <div class="text-[11px] text-cyan-200 font-mono">${mcpEsc(r.name)}</div>
        <div class="text-[10px] text-gray-500 line-clamp-2">${mcpEsc(r.description||"")}</div>
    </div>
    <span class="text-[9px] shrink-0 px-1.5 py-0.5 rounded ${r.read_only?"bg-emerald-800/40 text-eme\
rald-200":"bg-amber-800/40 text-amber-200"}">${r.read_only?"\u8AAD\u307F\u53D6\u308A":"\u5909\u66F4"}\
</span>
</div>`).join("");t.innerHTML=`<div class="rounded border border-gray-800 bg-black/20 p-2">${a}</div\
>`}catch{t.innerHTML='<div class="text-[10px] text-red-400">\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}}
o(mcpLoadTools,"mcpLoadTools");async function mcpSaveOauthClient(e,t,n,i){mcpStatusMsg("mcp-status-m\
sg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);const s={provider_key:e,client_id:t,client_secret:n};
try{const a=await apiFetch(MCP_URLS.oauthClient(),{method:"PUT",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(s)}),r=await a.json().catch(()=>({}));if(!a.ok){mcpStatusMsg("mcp-status\
-msg",r.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",!0);return}mcpStatusMsg("mcp\
-status-msg","OAuth\u30AF\u30E9\u30A4\u30A2\u30F3\u30C8\u60C5\u5831\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch(a){mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(a&&a.message?a.message:a),!0)}}o(mcpSaveOauthClient,"mcpSaveOauthClient");async function mcpAddCustomServer(){
const e=get("mcp-custom-name"),t=get("mcp-custom-url"),n=get("mcp-custom-auth"),i=get("mcp-custom-de\
sc"),s=get("mcp-custom-bearer"),a=get("mcp-custom-status"),r=get("mcp-add-server-btn");if(!e||!t||!n)
return;const l=(e.value||"").trim(),d=(t.value||"").trim(),p=n.value||"none",g=i?(i.value||"").trim():
"",h=(s&&s.value||"").trim();if(!l||!d){a&&(a.textContent="\u8868\u793A\u540D\u3068URL\u306F\u5FC5\u9808\u3067\u3059",
a.style.color="#f87171");return}r&&(r.disabled=!0),a&&(a.textContent="\u63A5\u7D9A\u30C6\u30B9\u30C8\u4E2D...",
a.style.color="#9ca3af");const v={name:l,url:d,auth_type:p,description:g};p==="bearer"&&h&&(v.bearer_token=
h);try{const b=await apiFetch(MCP_URLS.servers(),{method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify(v)}),w=await b.json().catch(()=>({}));if(!b.ok){a&&(a.textContent=w.error||
"\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",a.style.color="#f87171");return}a&&(a.textContent=
w.probe&&w.probe.message||"\u8FFD\u52A0\u3057\u307E\u3057\u305F",a.style.color=w.probe&&w.probe.ok?"\
#34d399":"#fbbf24"),e.value="",t.value="",i&&(i.value=""),s&&(s.value=""),mcpLoaded=!1,loadMcpServers(
!0)}catch(b){a&&(a.textContent="\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+(b&&b.message?
b.message:b),a.style.color="#f87171")}finally{r&&(r.disabled=!1)}}o(mcpAddCustomServer,"mcpAddCustom\
Server");function bindMcpSettingsUi(){const e=get("mcp-server-list");if(!e)return;const t=get("mcp-a\
dd-server-btn");t&&t.addEventListener("click",mcpAddCustomServer);const n=get("mcp-custom-auth"),i=get(
"mcp-custom-bearer-wrap");if(n&&i){const a=o(()=>{i.classList.toggle("hidden",n.value!=="bearer")},"\
syncBearer");n.addEventListener("change",a),a()}const s=get("mcp-save-google-client-btn");s&&s.addEventListener(
"click",async()=>{const a=get("mcp-google-client-id"),r=get("mcp-google-client-secret"),l=get("mcp-g\
oogle-client-state"),d=a?a.value:"",p=r?r.value:"";if(!d&&!p){l&&(l.textContent="Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
l.style.color="#f87171");return}await mcpSaveOauthClient(mcpGoogleProviderKey,d||"********",p||"****\
****",null)}),e.addEventListener("click",async a=>{const r=a.target.closest("[data-act]");if(!r)return;
const l=r.getAttribute("data-act"),d=r.getAttribute("data-id");if(l==="test"){a.preventDefault(),mcpTestServer(
d);return}if(l==="tools"){a.preventDefault(),mcpLoadTools(d);return}if(l==="auth"||l==="reconnect"){
a.preventDefault(),mcpOpenAuth(d);return}if(l==="disconnect"){a.preventDefault(),mcpDisconnect(d);return}
if(l==="delete"){a.preventDefault(),mcpDeleteServer(d);return}if(l==="edit-oauth"){if(a.preventDefault(),
r.closest("[data-mcp-server]")){const g=r.getAttribute("data-oauth-pk")||"",h=mcpServers.find(w=>String(
w.id)===String(d)),v=document.createElement("div");v.className="mt-2 rounded border border-amber-700\
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
uth" data-id="${d}" data-pk="${mcpEsc(h&&(h.oauth_provider_key||h.slug)||"")}">\u4FDD\u5B58</button>
    </div>`;const b=r.closest("div");b.parentNode.insertBefore(v,b.nextSibling),r.remove()}return}if(l===
"save-oauth"){a.preventDefault();const p=r.getAttribute("data-pk")||"",g=r.closest("[data-mcp-server\
]")||document,h=g.querySelectorAll('[data-oauth-role="cid"], .mcp-oauth-edit-cid'),v=g.querySelectorAll(
'[data-oauth-role="secret"], .mcp-oauth-edit-sec'),b=h.length?h[h.length-1].value:"",w=v.length?v[v.
length-1].value:"";if(!b&&!w){mcpStatusMsg("mcp-status-msg","Client ID \u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpSaveOauthClient(p,b||"********",w||"********",d);return}if(l==="save-bearer"){a.preventDefault();
const p=document.querySelector(`[data-bearer-id="${d}"]`),g=p?p.value:"";if(!g||g.trim()===""){mcpStatusMsg(
"mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
!0);return}mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059...",!1);try{const h=await apiFetch(
MCP_URLS.server(d),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({bearer_token:g})}),
v=await h.json().catch(()=>({}));if(!h.ok){mcpStatusMsg("mcp-status-msg",v.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0);return}mcpStatusMsg("mcp-status-msg","Bearer\u30C8\u30FC\u30AF\u30F3\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F\u3002",
!1),loadMcpServers(!0)}catch{mcpStatusMsg("mcp-status-msg","\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}return}}),e.addEventListener("change",a=>{const r=a.target.closest(".mcp-enable-toggle");r&&mcpToggleEnabled(
r.getAttribute("data-id"),r.checked)})}o(bindMcpSettingsUi,"bindMcpSettingsUi");const initMcpUi=o(()=>{
try{bindMcpSettingsUi()}catch{}try{loadMcpServers()}catch{}},"initMcpUi");document.readyState==="loa\
ding"?document.addEventListener("DOMContentLoaded",initMcpUi,{once:!0}):initMcpUi();function bindMcpPromptToggle(){
const e=get("enable-mcp");e&&e.addEventListener("change",()=>{if(syncMcpAutoSysRows(),typeof refreshMinimalOptionsIfOpen==
"function")try{refreshMinimalOptionsIfOpen()}catch{}})}if(o(bindMcpPromptToggle,"bindMcpPromptToggle"),
document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>{try{bindMcpPromptToggle()}catch{}});else
try{bindMcpPromptToggle()}catch{}function getModelTags(e,t){const n=[],i=(e.id||"").toLowerCase(),s=(e.
name||"").toLowerCase(),a=(e.desc||"").toLowerCase(),r=(t.category||"").toLowerCase();return(r.includes(
"gemini")||i.includes("gemini")||s.includes("gemini")||a.includes("gemini")||r.includes("banana")||s.
includes("banana"))&&n.push("gemini"),(r.includes("deepseek")||i.includes("deepseek")||s.includes("d\
eepseek")||a.includes("deepseek"))&&n.push("deepseek"),(r.includes("mistral")||i.includes("mistral")||
s.includes("mistral")||a.includes("mistral")||i.includes("ocr")||r.includes("ocr"))&&n.push("mistral"),
(r.includes("gpt")||r.includes("openai")||i.includes("gpt")||s.includes("gpt")||a.includes("openai"))&&
n.push("openai"),(r.includes("xai")||r.includes("grok")||i.includes("grok")||s.includes("grok")||a.includes(
"xai"))&&n.push("xai"),(r.includes("image")||i.includes("image")||s.includes("image")||a.includes("i\
mage"))&&n.push("image"),(r.includes("audio")||r.includes("music")||r.includes("transcription")||r.includes(
"speech")||i.includes("tts")||i.includes("transcri")||s.includes("tts")||s.includes("transcri")||s.includes(
"voice")||a.includes("tts")||i.includes("realtime")||i.includes("live")||i.includes("voice-agent")||
i.includes("native-audio")||s.includes("audio")||a.includes("audio")||a.includes("speech-to-text"))&&
n.push("audio"),(i.includes("reasoning")||s.includes("reasoning")||a.includes("reasoning"))&&n.push(
"reasoning"),(r.includes("deepseek")||i.includes("deepseek")||s.includes("deepseek"))&&!n.includes("\
reasoning")&&n.push("reasoning"),(i.includes("fast")||s.includes("fast")||a.includes("fast")||r.includes(
"fast"))&&n.push("fast"),(i.includes("deepseek-v4-flash")||r.includes("deepseek")&&s.includes("flash"))&&
!n.includes("fast")&&n.push("fast"),(r.includes("anthropic")||i.includes("claude")||s.includes("clau\
de")||a.includes("anthropic"))&&n.push("anthropic"),(r.includes("kimi")||i.includes("kimi")||s.includes(
"kimi")||a.includes("moonshot"))&&n.push("kimi"),(r.includes("video")||i.includes("video")||i.startsWith(
"veo-")||i.includes("omni-")||s.includes("video")||a.includes("video"))&&n.push("video"),(r.includes(
"music")||i.startsWith("lyria-")||s.includes("music")||a.includes("music")||a.includes("song"))&&n.push(
"music"),(r.includes("transcription")||i.includes("transcri")||s.includes("transcri")||a.includes("t\
ranscription")||a.includes("speech-to-text"))&&n.push("transcription"),(r.includes("ocr")||i.includes(
"ocr")||s.includes("ocr")||a.includes("ocr"))&&n.push("ocr"),(r.includes("agent")||i.includes("agent")||
s.includes("agent")||a.includes("agentic")||a.includes("computer use")||a.includes("deep research"))&&
n.push("agent"),e.agenticView&&n.push("agentic view"),n}o(getModelTags,"getModelTags");function updateModelTagUi(){
const e=get("model-tag-bar");if(!e)return;e.querySelectorAll(".model-tag-btn").forEach(n=>{const i=n.
innerText.trim().toLowerCase(),s=(i==="all"?"all":i)===activeModelTag;n.className=`model-tag-btn px-\
2 py-1 text-[10px] rounded border transition ${s?"bg-blue-600/20 border-blue-500 text-blue-300":"bg-\
gray-800 border-gray-700 text-gray-300 hover:border-gray-500"}`})}o(updateModelTagUi,"updateModelTag\
Ui");function getModelCapabilitySearchTerms(e){const t=String(e.id||"").toLowerCase(),n=[],i=t.includes(
"deepseek"),s=t.includes("tts"),a=t.startsWith("mistral-ocr"),r=s||a||t.includes("transcribe")||t.includes(
"realtime")||t.includes("voice-agent")||t.includes("native-audio")||t.includes("live")||t.includes("\
image")||t.includes("video")||t.startsWith("veo-")||t.includes("omni-flash")||t.startsWith("lyria-")||
t.includes("embedding"),l=!r&&(t.includes("gpt")||t.includes("gemini")||t.includes("grok")||i||t.startsWith(
"deep-research-")||t.startsWith("antigravity-")),d=o((...g)=>g.forEach(h=>n.push(h,h.replace(/-/g," "))),
"add");if((t.includes("gemini-3.1-flash-image")||t.includes("gemini-3-pro-image")||t.includes("gemin\
i-2.5-flash-image"))&&d("image generation","image editing"),t==="gemini-3.1-flash-lite-image"||t==="\
gemini-3.1-flash-image"?d("thinking","\u601D\u8003","minimal","high","thinking level"):t.includes("g\
emini")&&!r&&(d("thinking","\u601D\u8003","thinking level"),t==="gemini-3.8-flash"||t==="gemini-3.7-\
flash"?d("low","medium","high"):t==="gemini-3.6-flash"?d("medium","high"):t==="gemini-3.5-flash-lite"?
d("minimal","medium","high"):t.includes("flash")?d("minimal","low","medium","high"):d("low","high")),
i&&(d("thinking","\u601D\u8003","reasoning","\u63A8\u8AD6","reasoning effort","high"),t!=="deepseek-\
v4-pro"&&d("low"),t.includes("v4-flash")&&d("none","max")),l&&(t.includes("gpt-5")||t.includes("o1")||
t.includes("o3")||t.includes("grok-4.3")||t.includes("grok-4.5")||t.includes("grok-4.6")||t.includes(
"grok-4.20-0309-reasoning")||t.includes("grok-build")||t.includes("multi-agent")||t.includes("gpt")&&
!s)){d("reasoning","\u63A8\u8AD6","reasoning effort","low","high");const g=t==="gpt-5.6"||t.startsWith(
"gpt-5.6-"),h=t.includes("grok-4.6"),v=t.includes("grok-4.3")||t.includes("grok-4.5")||h||t.includes(
"grok-4.20-0309-reasoning")||t.includes("grok-build")||t.includes("multi-agent")||t.includes("gpt-5")||
t.includes("o1")||t.includes("o3"),b=t.includes("grok-4.3")||t.includes("grok-build")||t.includes("g\
pt-5")||i;v&&d("medium"),b&&d("none"),(g||i)&&d("max"),(h||t.includes("multi-agent")||g)&&d("xhigh")}
return t.includes("claude")&&d("thinking","\u601D\u8003","thinking budget","budget"),e.agenticView&&
d("agentic view"),[...new Set(n)]}o(getModelCapabilitySearchTerms,"getModelCapabilitySearchTerms");const modelListGroups=[];
let modelListBanner=null,modelListEmpty=null,modelListBuilt=!1,modelListAnimated=!1,modelListRenderFrame=0;
function buildModelList(){const e=get("model-list-container");!e||modelListBuilt||(e.innerHTML="",modelListBanner=
document.createElement("div"),modelListBanner.className="hidden mb-4 px-3 py-2 rounded-lg border bor\
der-teal-500/40 bg-teal-900/20 text-[11px] text-teal-200",e.appendChild(modelListBanner),MODELS.forEach(
t=>{const n=t.items.filter(r=>!r.deprecated);if(!n.length)return;const i=document.createElement("sec\
tion");i.className="model-list-group",i.innerHTML=`
                    <div class="flex items-center gap-2 mb-3 px-2">
                        <i class="${t.icon}"></i>
                        <div>
                            <h3 class="font-bold text-gray-200 text-sm">${t.category}</h3>
                            <p class="text-[10px] text-gray-500">${t.description}</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2 mb-6"></div>
                `;const s=i.querySelector(".grid"),a=n.map(r=>{const l=document.createElement("butto\
n"),d=String(r.apiId||r.id||"").trim(),p=r.agenticView?'<span class="inline-flex items-center gap-1 \
rounded-full border border-teal-500/40 bg-teal-900/20 px-2 py-0.5 text-[9px] font-semibold text-teal\
-200 whitespace-nowrap" title="Agentic View\u5BFE\u5FDC\uFF1A\u753B\u50CF\u3092\u30AF\u30ED\u30C3\u30D7\u3057\u3066\u518D\u89B3\u5BDF\u3057\u306A\u304C\u3089\u63A8\u8AD6\u3092\u7D99\u7D9A\u3067\u304D\u307E\u3059"><i class="fas fa-eye" aria-\
hidden="true"></i>Agentic View</span>':"",g=d?`<div class="text-[10px] text-cyan-300/90 mt-1.5 font-\
mono break-all"><span class="font-sans text-gray-500 mr-1">API model:</span>${escapeHtml(d)}</div>`:
"",h=r.price?`<div class="text-[10px] text-amber-400/90 mt-1.5 font-mono flex items-start gap-1"><i \
class="fas fa-tag text-[9px] mt-0.5 opacity-70 shrink-0"></i><span>${r.price}</span></div>`:"";return l.
type="button",l.className="flex flex-col text-left p-3 rounded-lg border transition bg-gray-800 bord\
er-gray-700 hover:border-gray-500 hover:bg-gray-750",l.dataset.selected="0",l.onclick=()=>selectModel(
r.id,r.name),l.innerHTML=`
                        <div class="flex justify-between items-start gap-2 w-full mb-1">
                            <div class="flex flex-wrap items-center gap-2 min-w-0">
                                <span class="font-bold text-sm text-gray-200">${r.name}</span>
                                ${p}
                            </div>
                            <i class="model-selected-icon fas fa-check-circle text-blue-400 hidden s\
hrink-0 mt-0.5"></i>
                        </div>
                        <span class="text-[10px] text-gray-400">${r.desc}</span>
                        ${g}
                        ${h}
                    `,s.appendChild(l),{model:r,button:l,searchText:`${r.name} ${r.id} ${d} ${r.agenticView?
"agentic view":""} ${t.category} ${getModelTags(r,t).join(" ")} ${getModelCapabilitySearchTerms(r).join(
" ")}`.toLowerCase(),provider:getModelApiProvider(r.id),tags:new Set(getModelTags(r,t))}});modelListGroups.
push({element:i,entries:a}),e.appendChild(i)}),modelListEmpty=document.createElement("div"),modelListEmpty.
className="hidden text-center text-gray-500 py-8",e.appendChild(modelListEmpty),modelListBuilt=!0)}o(
buildModelList,"buildModelList");function updateModelButtonSelection(e,t){const n=t===e.model.id;if(e.
button.dataset.selected===(n?"1":"0"))return;e.button.dataset.selected=n?"1":"0",e.button.classList.
toggle("bg-blue-600/20",n),e.button.classList.toggle("border-blue-500",n),e.button.classList.toggle(
"ring-1",n),e.button.classList.toggle("ring-blue-500",n),e.button.classList.toggle("bg-gray-800",!n),
e.button.classList.toggle("border-gray-700",!n),e.button.classList.toggle("hover:border-gray-500",!n),
e.button.classList.toggle("hover:bg-gray-750",!n);const i=e.button.querySelector(".model-selected-ic\
on");i&&i.classList.toggle("hidden",!n)}o(updateModelButtonSelection,"updateModelButtonSelection");function renderModelList(e="",t={}){
const n=get("model-list-container");if(!n)return;buildModelList();const i=e.toLowerCase(),s=window._visionPickerActive?
null:getPromptCacheLockedProvider(),a=s?PROVIDER_LABELS[s]||s:"",r=get("model-select")?get("model-se\
lect").value:"";let l=0;modelListBanner.classList.toggle("hidden",!s),s&&(modelListBanner.innerHTML=
`<i class="fas fa-database mr-1.5"></i>PromptCache \u6709\u52B9\u4E2D: <strong>${a}</strong> \u306E\u30E2\u30C7\u30EB\u306E\u307F\u9078\
\u629E\u3067\u304D\u307E\u3059\uFF08\u4ED6API\u3078\u306E\u5207\u66FF\u306F\u4E0D\u53EF\uFF09`),modelListGroups.
forEach(d=>{let p=0;d.entries.forEach(g=>{const h=g.searchText.includes(i)&&(!s||g.provider===s)&&(activeModelTag===
"all"||g.tags.has(activeModelTag));g.button.classList.toggle("hidden",!h),updateModelButtonSelection(
g,r),h&&(p+=1)}),d.element.classList.toggle("hidden",p===0),l+=p}),modelListEmpty.classList.toggle("\
hidden",l!==0),l===0&&(modelListEmpty.textContent=s?`No ${a} models found.`:"No models found."),t.animate&&
!modelListAnimated&&(modelListAnimated=!0,n.classList.add("model-list-animate"))}o(renderModelList,"\
renderModelList");function scheduleModelListRender(e){modelListRenderFrame&&cancelAnimationFrame(modelListRenderFrame),
modelListRenderFrame=requestAnimationFrame(()=>{modelListRenderFrame=0,renderModelList(e)})}o(scheduleModelListRender,
"scheduleModelListRender");function openModelModal(){location.pathname!=="/model"&&history.pushState(
{modal:"model"},"","/model");const e=get("model-search");e&&(e.value=""),updateModelTagUi(),renderModelList(
"",{animate:!0}),showModal("model-modal"),e&&window.innerWidth>768&&requestAnimationFrame(()=>e.focus(
{preventScroll:!0}))}o(openModelModal,"openModelModal"),window.closeModelModal=(e=!1)=>{hideModal("m\
odel-modal"),!e&&location.pathname==="/model"&&history.back()};function selectModel(e,t){if(window._visionPickerActive){
currentVisionModel=e,window._visionPickerActive=!1,window.closeModelModal(),_syncVisionModelDisplay();
return}if(isPromptCacheEnabled()){const s=getModelApiProvider(get("model-select")?get("model-select").
value:""),a=getModelApiProvider(e);if(s&&a&&s!==a){const r=PROVIDER_LABELS[s]||s,l=PROVIDER_LABELS[a]||
a;showToast(`PromptCache \u6709\u52B9\u4E2D\u306F\u4ED6API\uFF08${l}\uFF09\u306E\u30E2\u30C7\u30EB\u306B\u5909\u66F4\u3067\u304D\u307E\u305B\u3093\u3002\u73FE\u5728: ${r}`,
"warning",!0);return}}const n=get("model-select");n.value=e,get("model-selector-text").innerText=t,window.
closeModelModal();const i=new Event("change");n.dispatchEvent(i)}o(selectModel,"selectModel");function selectModelById(e){
let t=e;for(const n of MODELS){const i=n.items.find(s=>s.id===e);if(i){t=i.name;break}}selectModel(e,
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
"populateAiSafeFormFields"),get("model-search")&&get("model-search").addEventListener("input",e=>scheduleModelListRender(
e.target.value)),get("model-tag-bar")&&(get("model-tag-bar").addEventListener("click",e=>{const t=e.
target.closest(".model-tag-btn");if(!t)return;const n=t.innerText.trim().toLowerCase();activeModelTag=
MODEL_TAGS.includes(n)?n:"all",updateModelTagUi(),renderModelList(get("model-search").value)}),updateModelTagUi()),
window.quickStart=e=>{selectModelById(e),get("welcome-screen").classList.add("hidden")};const BROWSER_FAST_DISABLED_OPTIONS=[
["enable-search","search-container"],["enable-url-context","url-context-container"],["enable-maps","\
maps-grounding-container"],["enable-sys-prompt","sys-prompt-option"],["enable-prompt-cache","prompt-\
cache-container"],["enable-mcp","mcp-container"],["enable-file-creation","file-creation-container"]];
function applyBrowserFastModeRestrictions(){if(!browserFastModeEnabled)return;browserFastPreviousOptions||
(browserFastPreviousOptions={checks:Object.fromEntries(BROWSER_FAST_DISABLED_OPTIONS.map(([n])=>[n,!!(get(
n)&&get(n).checked)])),coding:!!codingModeEnabled}),BROWSER_FAST_DISABLED_OPTIONS.forEach(([n,i])=>{
const s=get(n),a=get(i);s&&(s.checked=!1,s.disabled=!0),a&&a.classList.add("opacity-50","pointer-eve\
nts-none")}),codingModeEnabled&&syncCodingModeUi(!1,{persist:!1});const e=get("enable-coding-mode"),
t=get("coding-mode-container");e&&(e.disabled=!0),t&&t.classList.add("opacity-50","pointer-events-no\
ne"),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}o(applyBrowserFastModeRestrictions,
"applyBrowserFastModeRestrictions");function restoreBrowserFastModeOptions(){const e=browserFastPreviousOptions;
if(!e)return;BROWSER_FAST_DISABLED_OPTIONS.forEach(([i,s])=>{const a=get(i),r=get(s);a&&(a.disabled=
!1,e&&e.checks&&Object.prototype.hasOwnProperty.call(e.checks,i)&&(a.checked=!!e.checks[i])),r&&r.classList.
remove("opacity-50","pointer-events-none")});const t=get("enable-coding-mode"),n=get("coding-mode-co\
ntainer");t&&(t.disabled=!1),n&&n.classList.remove("opacity-50","pointer-events-none"),e&&e.coding&&
syncCodingModeUi(!0,{persist:!1}),browserFastPreviousOptions=null,typeof updatePromptCacheUi=="funct\
ion"&&updatePromptCacheUi(),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows(),refreshMinimalOptionsIfOpen()}
o(restoreBrowserFastModeOptions,"restoreBrowserFastModeOptions");function isBatchModelKey(e){const t=String(
e||"").trim().toLowerCase();return t.startsWith("gpt-")?!/(image|audio|tts|transcribe|realtime|search)/.
test(t):t.startsWith("gemini-")?!/(embedding|video|veo|music|lyria|native-audio|tts|live|transcribe|agent|deep-research|robotics|computer-use)/.
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
"hidden",!e),n&&n.classList.toggle("hidden",!e);const i=get("browser-fast-mode-key-description"),s=String(
get("model-select")?get("model-select").value:"Gemini");i&&(i.textContent=`${s} \u306E\u30E2\u30C7\u30EB\u5225\u30AD\u30FC \u2192 \u5171\u901AGemini\u30AD\u30FC\
\u306E\u9806\u306B\u3001\u30B5\u30FC\u30D0\u30FC\u304B\u3089\u81EA\u52D5\u53D6\u5F97\u3057\u307E\u3059\u3002`),
showModal("browser-fast-mode-modal")}o(openBrowserFastModeModal,"openBrowserFastModeModal");function browserFastBootstrapMatches(e,t,n,i){
return!e||e.model!==t||String(e.thread_id||"")!==String(n||"")?!1:String(e.parent_id||"")===String(i||
"")}o(browserFastBootstrapMatches,"browserFastBootstrapMatches");async function fetchBrowserFastBootstrap(e=!1){
const t=String(get("model-select")?get("model-select").value:"").trim(),n=currentThreadId||null,i=n&&
currentParentId||null;if(!e&&browserFastBootstrapMatches(browserFastBootstrap,t,n,i)&&browserFastApiKey)
return browserFastBootstrap;const s=await apiFetch("/api/browser_fast_mode/bootstrap",{method:"POST",
headers:{"Content-Type":"application/json"},body:JSON.stringify({model:t,thread_id:n,parent_id:i})}),
a=await s.json().catch(()=>({}));if(!s.ok||!a.api_key)throw new Error(a.error||"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u306EGemini API\u30AD\
\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");return browserFastApiKey=
String(a.api_key),browserFastApiKeyModel=t,browserFastBootstrap=a,a}o(fetchBrowserFastBootstrap,"fet\
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
"active")})}),document.addEventListener("DOMContentLoaded",()=>{var Xn,Yn;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const c=get("enable-batch-mode");c&&c.checked&&(c.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-batch-mode");t&&(t.onchange=()=>{if(t.checked){browserFastModeEnabled&&
setBrowserFastModeEnabled(!1);const c=get("enable-coding-mode");c&&c.checked&&(c.checked=!1,typeof syncCodingModeUi==
"function"&&syncCodingModeUi(!1),showToast("Batch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const n=get("model-\
select");n&&n.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const c=String(
n.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!c.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(c)?(setBrowserFastModeEnabled(!1),n.dispatchEvent(
new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const i=get("browser-fast-mode-enable-btn");
i&&(i.onclick=async()=>{const c=i.innerHTML;i.disabled=!0,i.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const u=get("browser-fast-mode-ignore-warning");if(u&&u.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(u){showToast(u.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{i.disabled=!1,i.innerHTML=c}});const s=get("browser-fast-mode-cancel-btn");s&&(s.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const a=document.getElementById(
"alpha-bar");setTimeout(()=>{if(a){const c=document.getElementById("version-display");if(c){const u=a.
getBoundingClientRect(),m=c.getBoundingClientRect(),f=m.left+m.width/2-(u.left+u.width/2),y=m.top+m.
height/2-(u.top+u.height/2);a.style.transform=`translate(${f}px, ${y}px) scale(0.1)`,a.style.opacity=
"0",setTimeout(()=>{c.classList.add("pulse-target"),setTimeout(()=>c.classList.remove("pulse-target"),
2e3),a.remove()},800)}else a.style.opacity="0",setTimeout(()=>a.remove(),1e3)}},3e3);function r(){const c=get(
"gpt-image-options");if(!c)return;isGptImageModel()?c.classList.remove("hidden"):c.classList.add("hi\
dden");const u=get("gpt-image-format"),m=get("gpt-image-compression-wrap");u&&m&&(u.value==="png"?m.
classList.add("hidden"):m.classList.remove("hidden"))}o(r,"updateGptImageUi");function l(){const c=get(
"gemini-image-options");if(!c)return;isGeminiImageModel()?c.classList.remove("hidden"):c.classList.add(
"hidden");const m=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(f=>{f&&(Array.from(f.options).
forEach(y=>{y.value!=="1K"&&(y.disabled=m)}),m&&f.value!=="1K"&&(f.value="1K"))})}o(l,"updateGeminiI\
mageUi");function d(){const c=get("grok-image-options");if(!c)return;const u=(get("model-select").value||
"").toLowerCase(),m=isGrokImageModel(),f=u==="grok-imagine-image-quality"||u==="grok-imagine-image-2\
.0",y=u==="grok-imagine-image-2.0";if(m){c.classList.remove("hidden");const _=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;_&&_.classList.toggle("hidden",!f);const S=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;S&&S.classList.toggle("hidden",!y)}else
c.classList.add("hidden");if(get("modal-grok-image-options")){const _=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;_&&_.classList.toggle("hidden",!f);const S=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;S&&S.classList.toggle(
"hidden",!y)}}o(d,"updateGrokImageUi");function p(){var f;const c=get("grok-video-options");if(!c)return;
const u=String(((f=get("model-select"))==null?void 0:f.value)||"").toLowerCase();isGrokVideoModel()?
c.classList.remove("hidden"):c.classList.add("hidden");const m=get("grok-video-resolution");if(m){const y=Array.
from(m.options).find(k=>k.value==="1080p");y&&(y.disabled=u!=="grok-imagine-video-1.5"),u!=="grok-im\
agine-video-1.5"&&m.value==="1080p"&&(m.value="720p")}}o(p,"updateGrokVideoUi");function g(){var y;const c=get(
"gemini-video-options");if(!c)return;const u=String(((y=get("model-select"))==null?void 0:y.value)||
"").toLowerCase();isGeminiVideoModel()?c.classList.remove("hidden"):c.classList.add("hidden");const m=get(
"gemini-video-resolution");if(m){const k=Array.from(m.options).find(S=>S.value==="4K"),_=u==="veo-3.\
1-lite-generate-preview"||u==="veo-3.1-fast-generate-preview"||u==="gemini-omni-flash";k&&(k.disabled=
_),_&&m.value==="4K"&&(m.value="1080p")}const f=get("gemini-video-duration-wrap");f&&f.classList.toggle(
"hidden",u==="gemini-omni-1.1-flash")}o(g,"updateGeminiVideoUi");function h(){const c=get("gemini-mu\
sic-options");if(!c)return;const u=isGeminiRealtimeMusicModel(),m=isGeminiMusicModel()&&!u;c.classList.
toggle("hidden",!m);const f=get("lyria-realtime-studio-bar");f&&f.classList.toggle("hidden",!u)}o(h,
"updateGeminiMusicUi");function v(){var _;const c=get("xai-chat-options");if(!c)return;const u=String(
((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase(),m=u.startsWith("grok-")&&!isGrokImageModel(
u)&&!isGrokVideoModel(u)&&!u.includes("voice");c.classList.toggle("hidden",!m);const f=get("xai-logp\
robs"),y=get("xai-top-logprobs"),k=u.includes("grok-4.20");f&&(f.disabled=k,k&&(f.checked=!1)),y&&(y.
disabled=k,k&&(y.value=""))}o(v,"updateXaiChatUi");function b(){const c=isMistralOcrModel(),u=get("m\
istral-ocr-options");u&&u.classList.toggle("hidden",!c);const m=get("modal-mistral-ocr-options");m&&
m.classList.toggle("hidden",!c),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(f=>{const y=get(f);y&&(y.classList.toggle("opacity-50",c),y.classList.toggle("po\
inter-events-none",c))}),c&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(b,"updateMistralOcrUi");function w(){const c=get("image-input-limit\
s");if(!c)return;const u=(get("model-select").value||"").toLowerCase();let m="",f=!1;u.includes("gpt\
-image")?(f=!0,m=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):u==="deepseek-v4-flash-vision-exp"?(f=!0,m=['<div class="font-bold text-gray-300 mb-1">Dee\
pSeek V4 Flash Vision Exp \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP / \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \
\u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>","<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</di\
v>"].join("")):u.includes("deepseek")||(isGeminiImageModelKey(u)?(f=!0,u.includes("gemini-3.1-flash-\
lite-image")?m=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
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
o(w,"updateImageInputLimits");function x(){const c=get("model-select");if(!c)return;const u=c.value,
m=String(u||"").toLowerCase(),f=m.includes("deepseek"),y=get("thinking-options"),k=get("reasoning-ef\
fort-container"),_=get("enable-thinking"),S=get("thinking-level"),M=get("thinking-budget"),B=get("en\
able-search"),I=get("search-container"),D=get("url-context-container"),ne=get("enable-maps"),O=get("\
maps-grounding-container"),A=get("enable-sys-prompt"),H=get("sys-prompt-option"),V=get("enable-pytho\
n"),ve=get("python-container"),z=get("prompt-cache-container"),ie=get("enable-prompt-cache"),we=u===
"gpt-5-search-api",$e=u.includes("tts"),Oe=isMistralOcrModel(u),Ve=m.includes("gemini-3.1-flash-lite\
-image"),Qe=m.includes("gemini-3.1-flash-image")&&!Ve,Et=isClaudeModelKey(u),bt=isLlmModel()&&!f&&!$e&&
!m.includes("realtime")&&!m.includes("native-audio")&&!m.includes("live");z&&(bt?(z.classList.remove(
"hidden","opacity-50","pointer-events-none"),ie&&(ie.disabled=!1)):(ie&&(ie.checked=!1,ie.disabled=!0),
z.classList.add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),y&&y.classList.add("hidd\
en"),k&&k.classList.add("hidden");const yt=get("vision-model-info");if(yt&&yt.classList.add("hidden"),
k){const ue=get("reasoning-effort");if(ue){Array.from(ue.options).forEach(Se=>{const vt=m==="gpt-5.6"||
m.startsWith("gpt-5.6-"),wt=m==="deepseek-v4-flash-0731"||m==="deepseek-v4-flash"||m==="deepseek-v4-\
flash-vision-exp",$t=m==="deepseek-v4-pro",De=m.includes("grok-4.5"),it=m.includes("grok-4.6");Se.value===
"max"?Se.classList.toggle("hidden",!vt&&!wt&&!$t):Se.value==="xhigh"?Se.classList.toggle("hidden",!it&&
!m.includes("multi-agent")&&!vt):Se.value==="medium"?Se.classList.toggle("hidden",!(m.includes("grok\
-4.3")||De||it||m.includes("grok-4.20-0309-reasoning")||m.includes("grok-build")||m.includes("multi-\
agent")||m.includes("gpt-5")||m.includes("o1")||m.includes("o3"))):Se.value==="none"?Se.classList.toggle(
"hidden",!m.includes("grok-4.3")&&!m.includes("grok-build")&&!m.includes("gpt-5")&&!wt&&!$t):Se.value===
"low"&&Se.classList.toggle("hidden",$t)});const Ie=ue.selectedOptions&&ue.selectedOptions[0];Ie&&Ie.
classList.contains("hidden")&&(ue.value=f?"high":"medium")}}D&&D.classList.add("hidden"),O&&O.classList.
add("hidden"),_&&(_.disabled=!1),M&&(M.disabled=!0,M.classList.add("opacity-50"));const dt=isGeminiImageModelKey(
u);if($e||Oe)I&&(get("enable-search").checked=!1,I.classList.add("opacity-50","pointer-events-none")),
D&&(get("enable-url-context").checked=!1,D.classList.add("opacity-50","pointer-events-none")),O&&ne&&
(ne.checked=!1,O.classList.add("opacity-50","pointer-events-none")),ve&&(V.checked=!1,ve.classList.add(
"opacity-50","pointer-events-none")),A&&H&&(A.checked=!1,A.disabled=!0,H.classList.add("opacity-50"));else if(Qe||
Ve)O&&ne&&(ne.checked=!1,O.classList.add("hidden","opacity-50","pointer-events-none")),y.classList.remove(
"hidden"),Array.from(S.options).forEach(ue=>{["low","medium"].includes(ue.value)&&(ue.disabled=!0),[
"minimal","high"].includes(ue.value)&&(ue.disabled=!1)}),["minimal","high"].includes(S.value)||(S.value=
Ve?"minimal":"high"),_&&(_.disabled=!1),Ve&&(B&&(B.checked=!1,B.disabled=!0),I&&I.classList.add("opa\
city-50","pointer-events-none"));else if(dt)O&&ne&&(ne.checked=!1,O.classList.add("hidden","opacity-\
50","pointer-events-none"));else if(Et)y.classList.remove("hidden"),M&&(M.disabled=!1,M.classList.remove(
"opacity-50")),Array.from(S.options).forEach(ue=>{ue.disabled=!0}),ve&&(V.checked=!1,ve.classList.add(
"opacity-50","pointer-events-none"));else if(u.includes("gemini")&&!dt){y.classList.remove("hidden"),
D&&D.classList.remove("hidden","opacity-50","pointer-events-none");const ue=u.includes("gemini-3");O&&
(ue?O.classList.remove("hidden","opacity-50","pointer-events-none"):(ne&&(ne.checked=!1),O.classList.
add("hidden","opacity-50","pointer-events-none")));const Ie=u.includes("flash");Array.from(S.options).
forEach(Se=>{u==="gemini-3.8-flash"||u==="gemini-3.7-flash"?Se.disabled=!["low","medium","high"].includes(
Se.value):u==="gemini-3.6-flash"?Se.disabled=!["medium","high"].includes(Se.value):u==="gemini-3.5-f\
lash-lite"?Se.disabled=!["minimal","medium","high"].includes(Se.value):["minimal","medium"].includes(
Se.value)?Se.disabled=!Ie:Se.disabled=!1}),(u==="gemini-3.8-flash"||u==="gemini-3.7-flash")&&!["low",
"medium","high"].includes(S.value)||u==="gemini-3.6-flash"&&!["medium","high"].includes(S.value)?S.value=
"medium":u==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(S.value)?S.value="minima\
l":!Ie&&["minimal","medium"].includes(S.value)&&(S.value="high"),ue?_&&(_.checked=!0,_.disabled=!0):
_&&(_.disabled=!1),M&&u.includes("gemini-2.5")&&(M.disabled=!1,M.classList.remove("opacity-50")),M&&
!u.includes("gemini-2.5")&&(M.disabled=!0,M.classList.add("opacity-50"))}if(isLlmModel()&&(m.includes(
"gpt-5")||m.includes("o1")||m.includes("o3")||m.includes("grok-4.3")||m.includes("grok-4.5")||m.includes(
"grok-4.6")||m.includes("grok-4.20-0309-reasoning")||m.includes("grok-build")||m.includes("multi-age\
nt")||m.includes("gpt")&&!m.includes("tts")))k.classList.remove("hidden"),I&&I.classList.remove("opa\
city-50","pointer-events-none");else if(f){k.classList.remove("hidden");const ue=get("vision-model-i\
nfo");if(ue&&ue.classList.toggle("hidden",m==="deepseek-v4-flash-vision-exp"),B&&(B.checked=!1,B.disabled=
!0),I&&I.classList.add("opacity-50","pointer-events-none"),D){const Ie=get("enable-url-context");Ie&&
(Ie.checked=!1),D.classList.add("opacity-50","pointer-events-none")}O&&ne&&(ne.checked=!1,O.classList.
add("opacity-50","pointer-events-none"))}else Oe||(I&&I.classList.remove("opacity-50","pointer-event\
s-none"),O&&ne&&(ne.checked=!1,O.classList.add("hidden","opacity-50","pointer-events-none")));$e?ve&&
ve.classList.add("opacity-50","pointer-events-none"):(ve&&ve.classList.remove("opacity-50","pointer-\
events-none"),(!dt||Qe)&&!u.includes("gpt-image")&&(A.disabled=!1,H.classList.remove("opacity-50"))),
(dt&&!Qe||u.includes("gpt-image")||isGrokImageModel()||isGrokVideoModel()||Oe)&&A&&H&&(A.checked=!1,
A.disabled=!0,H.classList.add("opacity-50")),ve&&(isLlmModel()?(ve.classList.remove("hidden"),V.disabled=
!1):(V.checked=!1,V.disabled=!0,ve.classList.add("hidden"))),we?(B&&(B.checked=!0,B.disabled=!0),I&&
I.classList.add("opacity-50","pointer-events-none"),ve&&(V.checked=!1,V.disabled=!0,ve.classList.add(
"opacity-50","pointer-events-none"))):B&&!u.includes("tts")&&!Oe&&!f&&!Ve&&(B.disabled=!1);const re=get(
"mask-btn");re&&(isGptImageModel()?re.classList.remove("hidden"):(re.classList.add("hidden"),currentMaskImage=
null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),r(),l(),d(),p(),g(),h(),updateBatchUi(
u),v(),b(),w(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}
o(x,"toggleOptions"),get("model-select")&&(get("model-select").addEventListener("change",x),get("mod\
el-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
x(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const L=get("enable-canvas-mode");L&&(L.checked=canvasModeEnabled,L.addEventListener("change",()=>syncCanvasModeUi(
L.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const T=get("enable-codin\
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const $=get(
"prompt-controls-toggle-btn");$&&($.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>r()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>l()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(c,u,m){const f=(u||"").match(/\S*/)[0];if(f==="py\
exec")return"";if(f==="chat_error")return buildChatErrorBubbleHtml(c||"");const y=c||"",k=(f||"").toLowerCase();
let _="";try{const A=hljs.getLanguage(f)?f:"plaintext";activeStreamingBubbleId&&y.length>2e4?_=escapeHtml(
y):_=hljs.highlight(y,{language:A}).value}catch{_=escapeHtml(y)}const S=encodeURIComponent(y).replace(
/'/g,"%27"),M=detectBlockedScriptsInCode(y),B=hashString(`${f||"TEXT"}
${y||""}`);let I="";if(canvasModeEnabled){const A=String(canvasPreviewState.selectedKey||"")===B,H=A?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";I=`<button\
 class="canvas-preview-btn${A?" canvas-active":""}" data-code="${S}" data-code-key="${B}" data-canva\
s-lang="${escapeHtml(f||"txt")}" title="${H}" aria-label="${H}" aria-pressed="${A?"true":"false"}"><\
i class="fas ${A?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
k,y)){const A=M?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
I=`<button class="html-preview-btn" data-code="${S}" ${M?'data-suspicious="1"':""} title="${A}" aria\
-label="${A}"><i class="fas ${M?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const D=`\
<button class="download-btn" data-code="${S}" data-lang="${f||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ne=k==="diff"?"":`<button class="codin\
g-target-btn" data-code="${S}" data-code-key="${B}" data-coding-lang="${escapeHtml(f||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,O=(f||"TEXT")+(M?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\u305F\
">\u26A0</span>':"");return`<div class="code-wrapper collapsed" data-collapsed="true" data-code-key=\
"${B}"><div class="code-header"><span class="code-lang">${O}</span><div class="code-actions"><button\
 class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"\
></i></button>${ne}${I}${D}<button class="copy-btn" data-code="${S}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i\
 class="fas fa-copy"></i></button></div></div><div class="code-body"><pre><code class="hljs language\
-${f}">${_}</code></pre></div></div>`},link(c,u,m){return`<a href="${c}" title="${u||""}" target="_b\
lank">${m}</a>`},image(c,u,m){const f=escapeHtml(m||""),y=u?` title="${escapeHtml(u)}"`:"";if(String(
c||"").startsWith("sandbox:"))return`<span class="text-xs text-gray-500" title="${escapeHtml(c)}">${f||
"\uFF08\u753B\u50CF\u30C7\u30FC\u30BF\u306F\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09"}\
</span>`;const k=escapeHtml(c||"");return`<img src="${c}" data-viewer-src="${k}" alt="${f}"${y} clas\
s="chat-image" loading="lazy" width="320" height="320">`}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
c=>{c[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(Xn=get("version-update-dismis\
s"))==null||Xn.addEventListener("click",()=>{const c=localStorage.getItem("app_version")||"";c&&localStorage.
setItem("version_notified",c),hideModal("version-update-modal")});const j=get("version-update-clear-\
cache");if(j&&(j.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),j.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(j.checked)})),
(Yn=get("version-update-reload"))==null||Yn.addEventListener("click",async()=>{var u;await versionUpdateCachePreferenceSavePromise.
catch(()=>{}),!!((u=get("version-update-clear-cache"))!=null&&u.checked)?await clearSiteCacheAndReload(
get("version-update-reload"),{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.
setVersionChangeHandler(c=>{c&&c!==appVersion&&(localStorage.getItem("version_notified")||"")!==c&&(localStorage.
setItem("app_version",c),purgeCaches().then(()=>checkAndNotifyVersion(c)))}),window.ConnectionMonitor.
start(),window.addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener(
"offline",()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offli\
ne")}),window.addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener(
"visibilitychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener(
"pagehide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&
botConfig.lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(c){console.error(c)}try{runBotDetectionGate()}catch(c){console.error(
c)}}else{const c=get("turnstile-container");c&&c.classList.add("hidden")}const J=o(c=>{if(!c)return"\
\u4E0D\u660E";const u=new Date(c);return Number.isNaN(u.getTime())?c:u.toLocaleString()},"formatSess\
ionTime"),X=o(c=>{const u=Array.isArray(c)?c:[],m=get("passkey-list"),f=get("passkey-count");if(f&&(f.
innerText=String(u.length)),!!m){if(!u.length){m.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML="",u.forEach((y,k)=>{const _=y&&y.id?String(y.id):"",S=document.createElement("div");S.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const M=document.
createElement("div");M.className="min-w-0";const B=document.createElement("div");B.className="text-x\
s text-gray-200 truncate",B.innerText=y&&y.name?String(y.name):`Security Key ${k+1}`;const I=document.
createElement("div");I.className="text-[10px] text-gray-500 mt-1",I.innerText=y&&y.created_at?`\u767B\u9332\u65E5\u6642:\
 ${J(y.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",M.appendChild(B),M.appendChild(I),S.appendChild(
M);const D=document.createElement("button");D.type="button",D.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",D.innerText="\u524A\u9664",D.
disabled=!_,_&&(D.onclick=()=>window.removeWebAuthnCredential(_)),S.appendChild(D),m.appendChild(S)})}},
"renderPasskeyList"),Te=o(c=>{const u=get("session-list");if(u){if(!c||!c.length){u.innerHTML='<div \
class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}u.innerHTML=c.map(m=>{const f=m.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",y=m.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",k=!m.is_current&&!m.is_revoked?`<button \
data-session-id="${escapeHtml(m.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
_=(m.user_agent||"Unknown").slice(0,120),S=m.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${f}${y}<div class="text-xs text-gray-200">${escapeHtml(
S)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(_)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(J(m.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(J(m.created_at))}</div></div>${k}</div>`}).join(""),u.querySelectorAll(
".session-revoke-btn").forEach(m=>{m.onclick=async()=>{const f=m.getAttribute("data-session-id");if(!f||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const y=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:f})});let k={};try{k=await y.json()}catch{}if(y.ok){if(k.logged_out){
location.href="/login";return}await P()}else showToast(k&&k.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),P=o(async()=>{const c=get("session-list");c&&(c.innerHTML='<div c\
lass="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const u=await apiFetch("/api/\
sessions");let m={};try{m=await u.json()}catch{}if(!u.ok){if(m&&m.error==="session_revoked"){location.
href="/login";return}c&&(c.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const f=(m.sessions||[]).filter(y=>!y.is_revoked);Te(f)},"loadSessions"),q=o(()=>{const c=get(
"session-refresh-btn");c&&(c.onclick=()=>P());const u=get("session-revoke-others-btn");u&&(u.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await P():showToast("\u64CD\u4F5C\u306B\u5931\u6557\
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
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),typeof refreshGeminiBatchStatus=="funct\
ion"&&(refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,3e4)),get("send-btn").onclick=
()=>{isStopMode?stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),
bindMinimalOptionsEvents();const Y=get("vision-model-change-btn");Y&&(Y.onclick=()=>_openVisionModelSelector());
const de=get("compression-format-only");de&&(de.onchange=()=>{const c=de.checked,u=get("compression-\
max-size"),m=get("compression-max-dim");u&&(u.disabled=c),m&&(m.disabled=c);const f=get("compression\
-size-wrap"),y=get("compression-dim-wrap");f&&(f.style.opacity=c?"0.4":"1"),y&&(y.style.opacity=c?"0\
.4":"1")});const oe=o(()=>{const c=get("enable-temporary-chat");!c||c.dataset.bound==="1"||(c.dataset.
bound="1",c.checked=!!temporaryChatEnabled,c.onchange=async()=>{const u=temporaryChatEnabled;await applyTemporaryChatSetting(
c.checked)||(setTemporaryChatUiState(u),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");oe(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const xe=get("storage-u\
sage-refresh");xe&&(xe.onclick=()=>loadStorageUsage());let pe=null;const ke=o(()=>{const c=new Uint8Array(
16);return window.crypto.getRandomValues(c),Array.from(c,u=>u.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),te=o((c={})=>{const u=get("account-transfer-progress"),m=get("account-tra\
nsfer-progress-bar"),f=get("account-transfer-progress-percent"),y=get("account-transfer-progress-tex\
t"),k=get("account-transfer-progress-detail"),_=Math.max(0,Math.min(100,Number(c.progress)||0));if(u&&
u.classList.remove("hidden"),m&&(m.style.width=`${_}%`),f&&(f.textContent=`${Math.round(_)}%`),y&&(y.
textContent=c.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),k){
const M={queued:"\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};k.textContent=M[c.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const S=get("account-transfer-cancel-btn");S&&S.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(c.phase))},"renderAccountTransferProgress"),se=o(c=>{ye&&(ye.disabled=
!!c);const u=get("account-import-btn");u&&(u.disabled=!!c);const m=get("account-transfer-cancel-btn");
m&&(m.disabled=!c)},"setAccountTransferControls"),W=o((c={})=>{const u=get("account-export-ready"),m=get(
"account-export-ready-text"),f=get("account-export-expiry"),y=get("account-export-download-btn"),k=!!(c.
available&&c.download_url);if(u&&u.classList.toggle("hidden",!k),!k){y&&y.removeAttribute("href");return}
const _=Math.max(0,Number(c.size_bytes)||0),S=_>=1024*1024*1024?`${(_/(1024*1024*1024)).toFixed(2)} \
GB`:`${(_/(1024*1024)).toFixed(1)} MB`;if(m){const M=Number(c.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
c.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";m.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${S}${M}`}
if(f){const M=c.expires_at?new Date(c.expires_at):null;f.textContent=M&&!Number.isNaN(M.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${M.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
y&&(y.href=c.download_url)},"renderAccountExportAvailability"),C=o(async c=>{for(;pe===c&&!c.stopped;){
try{const u=await apiFetch(`/api/account/transfer/${c.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),m=await u.json().catch(()=>({}));if(u.ok&&(m.state!=="pending"&&te(m),["ready","completed","\
failed","cancelled","expired"].includes(m.state)))return m}catch{}await new Promise(u=>setTimeout(u,
700))}return null},"pollAccountTransfer"),R=o((c,u,m=!0)=>{u&&(te(u),W(u),m&&u.state==="ready"?showToast(
u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(u.unreadable_count)>0?"warning":"success",Number(u.unreadable_count)>0):m&&u.state==="failed"&&
showToast(u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),K(c))},"handleFinishedAccountExport"),G=o(async()=>{try{const c=await apiFetch("/api/acc\
ount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),u=await c.json().catch(()=>({}));
if(!c.ok)return;if(W(u),u.state==="ready"){te(u);return}if(["failed","cancelled","expired"].includes(
u.state)){te(u);return}if(!["queued","running","cancelling"].includes(u.state)||!u.job_id||pe&&pe.id===
u.job_id||pe)return;const m={id:u.job_id,type:"export",stopped:!1,restored:!0};pe=m,se(!0),te(u);const f=await C(
m);f&&R(m,f,!0)}catch{}},"refreshLatestAccountExport"),K=o(c=>{pe===c&&(pe=null),c.stopped=!0,se(!1)},
"finishAccountTransfer"),Z=get("account-transfer-cancel-btn");Z&&(Z.onclick=async()=>{const c=pe;if(!(!c||
c.stopped)){c.cancelRequested=!0,Z.disabled=!0,te({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${c.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
c.controller&&c.controller.abort(),te({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
c.type==="export"&&W({available:!1}),K(c),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const ye=get("account-export-btn");ye&&(ye.onclick=async()=>{if(pe)return;const c={id:ke(),
type:"export",stopped:!1};pe=c,se(!0),W({available:!1}),te({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const u=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:c.id}),keepalive:!0})),m=await u.json().catch(()=>({}));if(u.status===409&&
m.error==="export_in_progress"&&m.job_id)c.id=m.job_id;else if(!u.ok)throw new Error(m.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
m.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
te({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const f=await C(c);!c.cancelRequested&&f&&R(c,f,!0)}catch(u){const m=u&&u.message?u.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";te({progress:0,phase:"failed",message:m}),
showToast(m,"error",!0),K(c)}});const Q=get("account-export-download-btn");Q&&Q.addEventListener("cl\
ick",async c=>{const u=Q.getAttribute("href");if(!(!u||u==="#")){c.preventDefault();try{const m=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await m.json().catch(
()=>({}));m.ok&&f.available&&f.download_url?(Q.href=f.download_url,window.location.assign(f.download_url)):
(W(f),te(f),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),G())}catch{window.location.assign(u)}}}),se(!1),G();const le=get("import-files-grid"),
U=get("import-files-info"),fe=get("import-files-summary"),et=o(c=>{const u=Math.max(0,Number(c)||0);
return u>=1024*1024*1024?`${(u/(1024*1024*1024)).toFixed(2)} GB`:u>=1024*1024?`${(u/(1024*1024)).toFixed(
1)} MB`:u>=1024?`${Math.round(u/1024)} KB`:`${u} B`},"importFormatBytes");let Ce=null;const tt=o(()=>{
if(!Ce)return;const c=Ce.files,u=Ce.selection;let m=0;c.forEach(k=>{u.has(k.archive_path)&&(m+=Number(
k.size_bytes)||0)});const f=Number(Ce.available_bytes)||0,y=m>f;fe&&(fe.textContent=`\u9078\u629E\u4E2D: ${et(
m)} / \u5229\u7528\u53EF\u80FD: ${et(f)}${y?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,fe.classList.
toggle("text-red-300",y)),U&&(U.textContent=`${c.length} files`)},"updateImportFileSelectionUi"),ut=o(
()=>{if(!le||!Ce)return;le.innerHTML="";const c=Ce.files;if(!c.length){le.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
tt();return}c.forEach(u=>{const m=document.createElement("label"),f=Ce.selection.has(u.archive_path);
m.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${f?"border-blue-500":"border-gray-600"}`,m.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${f?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(u.display_name)}">${escapeHtml(
u.display_name)}</div><div class="text-[10px] text-gray-500">${et(u.size_bytes)}</div></div>`;const y=m.
querySelector(".import-file-check");y.addEventListener("change",()=>{y.checked?Ce.selection.add(u.archive_path):
Ce.selection.delete(u.archive_path),m.classList.toggle("border-blue-500",y.checked),m.classList.toggle(
"border-gray-600",!y.checked),tt()}),le.appendChild(m)}),tt()},"renderImportFileItems"),Ke=o(c=>new Promise(
u=>{if(Ce={files:c.files||[],selection:new Set((c.files||[]).map(m=>m.archive_path)),available_bytes:c.
available_bytes,resolve:u},ut(),!get("import-files-modal")){u(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),ot=o(c=>{if(hideModal("import-files-modal"),Ce){const u=Ce.resolve;
Ce=null,u(c)}},"closeImportFileSelection"),xt=get("import-files-close");xt&&(xt.onclick=()=>ot(null));
const gt=get("import-files-cancel");gt&&(gt.onclick=()=>ot(null));const kt=get("import-files-confirm");
kt&&(kt.onclick=()=>{if(!Ce)return;const c=Array.from(Ce.selection);ot(c.length?c.join(","):"__none_\
_")});const rt=get("import-files-select-all");rt&&(rt.onclick=()=>{Ce&&(Ce.files.forEach(c=>Ce.selection.
add(c.archive_path)),ut())});const Pt=get("import-files-none");Pt&&(Pt.onclick=()=>{Ce&&(Ce.selection.
clear(),ut())});const Ot={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u4F7F\u7528",apply_global_system_prompt:"\
\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528",apply_auto_system_prompt_notices:"\
\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528",auto_system_prompt_notices_config:"\
\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u306E\u7A2E\u985E\u5225\u8A2D\u5B9A",
gemini_backend:"Gemini \u30D0\u30C3\u30AF\u30A8\u30F3\u30C9",gemini_vertex_location:"Vertex AI \u30ED\u30B1\u30FC\u30B7\u30E7\
\u30F3",mic_transcribe_mode:"\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u65B9\u5F0F",stt_model:"\
STT\u30E2\u30C7\u30EB",llm_transcribe_prompt:"LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8",
enter_to_send:"Enter\u30AD\u30FC\u3067\u9001\u4FE1",use_sw_cache:"Service Worker\u30AD\u30E3\u30C3\u30B7\u30E5",
clear_cache_on_version_update:"\u30D0\u30FC\u30B8\u30E7\u30F3\u66F4\u65B0\u6642\u30AD\u30E3\u30C3\u30B7\u30E5\u524A\u9664",
theme_color:"\u30C6\u30FC\u30DE\u30AB\u30E9\u30FC",liquid_glass_enabled:"Liquid Glass",auto_search_on_links:"\
\u30EA\u30F3\u30AF\u3067\u81EA\u52D5\u691C\u7D22",compact_prompt_mode:"\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u8868\u793A\uFF08\u30B3\u30F3\u30D1\u30AF\u30C8\uFF09",
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},at=o(c=>{
if(c===!0)return"ON";if(c===!1)return"OFF";if(c==null||c==="")return"\u672A\u8A2D\u5B9A";const u=String(
c);return u.length>60?u.slice(0,60)+"\u2026":u},"formatAccountSettingValue");let pt=null;const _t=o(
c=>{if(pt){const u=pt;pt=null,hideModal("settings-confirmation-modal"),u(c)}},"resolveSettingsImport\
Confirmation"),St=o(c=>new Promise(u=>{if(!get("settings-confirmation-modal")){u(!0);return}pt=u;const f=Array.
isArray(c&&c.settings_changes)?c.settings_changes:[],y=get("settings-confirmation-list");y&&(f.length?
y.innerHTML=f.map(_=>{const S=Ot[_.field]||_.field,M=at(_.current),B=at(_.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(S)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
M)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(B)}</d\
iv>
                            </div>`}).join(""):y.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const k=get("settin\
gs-confirmation-count");k&&(k.textContent=`${f.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),mt=get("settings-confir\
mation-modal");mt&&mt.addEventListener("click",c=>{c.target===mt&&_t(!1)});const zt=get("settings-co\
nfirmation-close");zt&&(zt.onclick=()=>_t(!1));const Wt=get("settings-confirmation-cancel");Wt&&(Wt.
onclick=()=>_t(!1));const N=get("settings-confirmation-confirm");N&&(N.onclick=()=>_t(!0));const ae=get(
"account-import-btn"),_e=get("account-import-inplace"),Me=get("account-import-inplace-warning");if(_e&&
Me){const c=o(()=>Me.classList.toggle("hidden",!_e.checked),"syncInplaceWarn");_e.addEventListener("\
change",c),c()}ae&&(ae.onclick=async()=>{const c=get("account-import-file"),u=c&&c.files?c.files[0]:
null,m=get("account-import-categories"),f=m?Array.from(m.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(O=>O.value):[],y=get("account-import-inplace"),k=!!(y&&y.checked),_=get("account-import\
-settings-bypass"),S=!!(_&&_.checked);let M=!1;if(!u){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!f.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const B=m?Array.from(m.querySelectorAll('input[type="checkbox"]:checked')).map(O=>(O.
closest("label")&&O.closest("label").textContent||O.value).trim()):f;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\u5B58\u30C7\
\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${B.join("\u3001")}${k?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const I={id:ke(),type:"import",stopped:!1,controller:new AbortController};
pe=I,se(!0),te({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const D=get("account-import-result");let ne=Promise.resolve(null);try{const A=Math.max(1,Math.ceil(u.
size/10485760)),H=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:u.size}),signal:I.controller.
signal})),V=await H.json().catch(()=>({}));if(!H.ok)throw new Error(V.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
I.uploadId=V.upload_id;const ve=V.chunk_size||10485760;let z=0,ie=0;const we=o(async()=>{for(;;){const ge=ie++;
if(ge>=A)return;const re=u.slice(ge*ve,Math.min(u.size,(ge+1)*ve)),ue=new FormData;ue.append("chunk",
re,u.name),ue.append("index",String(ge));const Ie=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
I.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:ue,signal:I.controller.signal})),
Se=await Ie.json().catch(()=>({}));if(!Ie.ok)throw new Error(Se.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
z++,te({progress:Math.min(35,Math.round(z/A*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${z}\
/${A}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let $e=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),$e=!0);try{await Promise.
all([we(),we(),we()]);const ge=await apiFetch(`/api/account/import/upload/${encodeURIComponent(I.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:I.controller.signal})),re=await ge.json().
catch(()=>({}));if(!ge.ok)throw new Error(re.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
te({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
$e&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let Oe="",Ve=!1,Qe=0;const Et=o(
async()=>{let ge=!1;const re=o(()=>{ge||(ge=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const ue=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Ie=await ue.
json().catch(()=>null);if(!ue.ok||!Ie){re();return}cacheUserSettings(Ie);const Se=get("settings-moda\
l");if(Se&&Se.classList.contains("modal-open"))try{In(Ie)}catch{}Ie.theme_color&&applyThemeColor(Ie.
theme_color,!0),Object.prototype.hasOwnProperty.call(Ie,"minimal_prompt_mode")&&Ie.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Ie,"compact_prompt_mode")&&setCompactPromptMode(
!!Ie.compact_prompt_mode)}catch{}re()},"refreshSettingsFormAfterImport"),bt=o(ge=>{const re=ge&&ge.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";D&&(D.textContent=`\u5B8C\u4E86: ${re}`,
D.classList.remove("hidden","text-red-300"),D.classList.add("text-emerald-300")),te({progress:100,phase:"\
completed",message:re}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),f.includes("chats")&&loadThreads(),f.includes("gems")&&loadGems(),f.includes("files")&&loadStorageUsage(),
(f.includes("settings")||f.includes("api_credentials"))&&Et()},"finishImportSuccess"),yt=o(async()=>{
try{const re=await(await apiFetch(`/api/account/transfer/${I.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return re&&re.state?re:null}catch{return null}},"fetchImportSta\
tus"),dt=o(async()=>{const ge=await yt();if(!ge)return{status:"unknown"};if(ge.state==="completed")return bt(
ge),{status:"done"};if(["failed","cancelled","expired"].includes(ge.state))throw new Error(ge.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(ge.state==="needs_sel\
ection"&&Array.isArray(ge.files)){const re=await Ke({files:ge.files,available_bytes:ge.available_bytes});
return re===null?(te({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
I.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(I.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(Oe=re,{status:"reselect"})}if(ge.state===
"needs_settings_confirmation"&&Array.isArray(ge.settings_changes))return await St({settings_changes:ge.
settings_changes})?(M=!0,{status:"reselect"}):(te({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\
\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),I.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
I.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(ge.state==="running"){const re=await Promise.race([ne.catch(()=>null),new Promise(ue=>setTimeout(
()=>ue(null),6e4))]);if(re&&re.state==="completed")return bt(re),{status:"done"};throw re&&["failed",
"cancelled","expired"].includes(re.state)?new Error(re.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ve;){I.stopped=!0,await ne.catch(()=>null),
I.stopped=!1,ne=C(I);let ge;try{ge=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:I.uploadId,
categories:f.join(","),job_id:I.id,selected_files:Oe,restore_inplace:k,confirm_settings:M||S}),signal:I.
controller.signal}))}catch(De){if(I.cancelRequested||De&&De.name==="AbortError")throw De;const it=await dt();
if(it.status==="done"){Ve=!0;break}if(it.status==="cancelled")return;if(it.status==="reselect")continue;
if(Qe<2){Qe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let re=null;try{re=await ge.json()}catch{re=null}if(re===null){const De=await dt();if(De.status==="d\
one"){Ve=!0;break}if(De.status==="cancelled")return;if(De.status==="reselect")continue;if(ge.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Qe<2){Qe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!ge.ok&&re.error==="storage_limit_files"&&re.files){const De=await Ke(re);if(De===null){te({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
I.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(I.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}Oe=De;continue}if(re&&re.status==="settings_confirmation"&&
Array.isArray(re.settings_changes)){if(!await St(re)){te({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\
\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),I.uploadId&&
apiFetch(`/api/account/import/upload/${encodeURIComponent(I.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}M=!0;continue}if(!ge.ok)throw new Error(re.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F");const ue=re.imported||{},Ie=[`\u8A2D\u5B9A ${ue.settings||0}\u4EF6`,
`API\u8A8D\u8A3C ${ue.api_credentials||0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${ue.chats||0}\u4EF6`,`Ge\
m ${ue.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${ue.files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${ue.
feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${ue.diagnostics||0}\u4EF6`].join(" / "),Se=re.duplicates||
{},vt={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\
\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"},wt=[];for(const De of Object.keys(vt)){const it=Number(
Se[De])||0;it>0&&wt.push(`${vt[De]} ${it}\u4EF6`)}const $t=wt.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${wt.
join("\u3001")}\uFF09`:"";D&&(D.textContent=`\u5B8C\u4E86: ${Ie}${$t}`,D.classList.remove("hidden","\
text-red-300"),D.classList.add("text-emerald-300")),te({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\
\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),f.includes("chats")&&loadThreads(),f.includes("gems")&&loadGems(),f.includes("files")&&loadStorageUsage(),
(f.includes("settings")||f.includes("api_credentials"))&&Et(),Ve=!0}}catch(O){if(I.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(I.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),I.cancelRequested||O&&O.name==="AbortError")return;const A=O&&O.message?O.
message:"",H=A==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
A||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";te({progress:0,phase:"\
failed",message:H}),D&&(D.textContent=H,D.classList.remove("hidden","text-emerald-300"),D.classList.
add("text-red-300")),showToast(H,"error",!0)}finally{I.stopped=!0,await ne.catch(()=>null),K(I)}});const je=get(
"account-dedupe-btn"),He=get("account-dedupe-result"),nt=o((c,u=!1)=>{He&&(He.textContent=c,He.classList.
remove("hidden"),He.classList.toggle("text-red-300",!!u),He.classList.toggle("text-emerald-300",!u))},
"showDedupeResult");je&&(je.onclick=async()=>{const c=o(async()=>{const u=await apiFetch("/api/accou\
nt/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})}),
m=await u.json().catch(()=>null);if(!u.ok||!m)throw new Error(m&&m.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!m.has_duplicates){nt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const f=[],y={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const I of[
"chats","gems","files","feedback","diagnostics"]){const D=Number(m.duplicates&&m.duplicates[I])||0;D>
0&&f.push(`${y[I]} ${D}\u4EF6`)}const k=Number(m.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${m.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${m.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${f.join("\u3001")}${k}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const _=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),S=await _.json().catch(()=>null);if(!_.ok||!S)throw new Error(
S&&S.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const M=[];for(const I of["chats","gems","files","feedback","diagnostics"]){const D=Number(S.removed&&
S.removed[I])||0;D>0&&M.push(`${y[I]} ${D}\u4EF6`)}const B=Number(S.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${S.kept_referenced_files}\u4EF6\uFF09`:"";nt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${M.join("\u3001")||"0\u4EF6"}${B}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!je.disabled){je.disabled=!0,nt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await c()}catch(u){nt(u&&u.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{je.disabled=!1}}});const Fe=get("site-cache-usage-refresh");Fe&&(Fe.onclick=()=>loadSiteCacheUsage());
const Pe=get("clear-site-cache-btn");Pe&&(Pe.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Pe)});
const Ne=get("enc-scan-result"),qe=o(async(c=null)=>{Ne&&(Ne.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let u="/api/encryption_scan";c&&(u+=`?thread_id=${encodeURIComponent(c)}`);try{const m=await apiFetch(
u,{cache:"no-store"}),f=await m.json();if(!m.ok){Ne&&(Ne.textContent=f.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const y=f.total||0,k=f.encrypted||0,_=f.unencrypted||0;let S=`Total: ${y} / Encrypted: ${k} /\
 Plain: ${_}`;if(f.samples&&f.samples.length){const M=f.samples.slice(0,8).map(B=>{const I=B.timestamp?
new Date(B.timestamp).toLocaleString():"";return`#${B.id} (${B.role||""}) ${I}`}).join(" / ");S+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${M}</div>`}Ne&&(Ne.innerHTML=S)}catch{Ne&&(Ne.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Vt=get("enc-scan-all");Vt&&(Vt.onclick=()=>qe(
null));const Nt=get("enc-scan-thread");Nt&&(Nt.onclick=()=>currentThreadId?qe(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ze=get("admin-enc-\
list");let Rt=null,Le=!1;const Re=o(c=>!c||!c.length?null:c.some(u=>!!u.is_encrypted),"computeThread\
EncryptedFromMessages"),Xe=o(()=>{Rt=Re(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Jt=o(
async(c,u,{confirmPrompt:m=!0,reloadCurrent:f=!0}={})=>{if(!c)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const y=u?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(m&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${y}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Le=!0;try{const k=await apiFetch(`/api/admin/threads/${encodeURIComponent(
c)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:u})}),
_=await k.json().catch(()=>({}));return k.ok?(showToast(`${y}\u3057\u307E\u3057\u305F\uFF08${_.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Rt=!!u,f&&currentThreadId&&String(currentThreadId)===String(
c)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ze&&await lt(),!0):
(showToast(_.error||`${y}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${y}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Le=!1}},"setAdminThreadEncry\
ption"),an=o(c=>{if(!Ze)return;const u=c.threads||[];if(!u.length){Ze.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ze.innerHTML=u.map(m=>{const f=m.encrypted_count>0?"enc":"plain",y=f==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",k=f==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",_=m.
updated_at?new Date(m.updated_at).toLocaleString():"",S=escapeHtml(String(m.thread_id)),M=currentThreadId&&
String(currentThreadId)===String(m.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(m.title||
"")}">${escapeHtml(m.title||"(\u7121\u984C)")}${M?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${_} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${m.
message_count} / \u6697\u53F7\u5316: ${m.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${S}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${k} text-white px-2 py-1 roun\
ded shrink-0" data-id="${S}" data-enable="${f==="enc"?"0":"1"}" data-progress-expected-slow="true">${y}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),lt=o(async()=>{if(Ze){Ze.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const c=await apiFetch("/api/admin/threads",{cache:"no-store"}),u=await c.json().catch(()=>({}));
if(!c.ok){Ze.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(u.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(an(u),currentThreadId&&Array.isArray(u.threads)){const m=u.threads.find(f=>String(
f.thread_id)===String(currentThreadId));m&&(Rt=!!m.encrypted)}}catch{Ze.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>lt()),window.__loadAdminEncThreads=
lt,window.__refreshAdminThreadEncState=Xe,window.__setAdminThreadEncryption=Jt;const ce=get("encrypt\
ion-status-admin-toggle");ce&&ce.addEventListener("click",c=>{c.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ze&&(Ze.onclick=async c=>{const u=c.target.closest("\
.admin-enc-open");if(u){c.preventDefault();const S=u.getAttribute("data-id");if(!S)return;typeof Bt==
"function"?Bt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(S)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const m=c.target.closest(".admin-enc-toggle");if(!m||Le)return;const f=m.getAttribute("data-i\
d"),y=m.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${y?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;m.disabled=!0;
const _=m.textContent;m.textContent="\u51E6\u7406\u4E2D...";try{await Jt(f,y,{confirmPrompt:!1,reloadCurrent:!0})}finally{
m.disabled=!1,m.textContent=_,await lt()}}),get("file-input").onchange=c=>{const u=Array.from(c.target.
files||[]);c.target.value="",u.length&&handleFiles(u)},get("photo-input")&&(get("photo-input").onchange=
c=>{const u=Array.from(c.target.files||[]);c.target.value="",u.length&&handleFiles(u)});const be=o(c=>{
const u=get("ban-appeal-list");if(u){if(!c||!c.length){u.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
u.innerHTML=c.map(m=>{const f=m.status||"new",y=m.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
k=m.created_at?new Date(m.created_at).toLocaleString():"",_=m.replied_at?new Date(m.replied_at).toLocaleString():
"",S=m.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${m.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(m.username||
"")}${y}</div>
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
S)}</textarea>
                                ${S?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
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
                    `}).join("")}},"renderBanAppeals"),Ge=o(async(c=!1)=>{if(!isAdminUser)return;const u=get(
"ban-appeal-count");if(u)try{const m=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!m.
ok)return;const y=(await m.json()).unread_count||0;u.textContent=String(y),c&&y>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${y}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Ue=o(async()=>{if(!isAdminUser)return;const c=get("ban-appeal-list");if(c){c.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const u=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!u.ok)return;const m=await u.json();be(m.items||[]),
await Ge(!1)}catch{}}},"loadBanAppeals"),Ye=o(async(c=null)=>{if(!isAdminUser)return;const u=c?{ids:c}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(u)})).ok&&await Ue()}catch{}},"markBanAppealsRead"),ct=o(async c=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(c)})).ok&&await Ue()}catch{}},"updateBanAppealStatus"),Tt=o(()=>{
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
                `,c.appendChild(u)},"ensureTemporaryChatSettingsCard"),En=o(()=>{const c=get("set-st\
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
                `,u.appendChild(m);const f=get("reset-llm-transcribe-prompt");f&&(f.onclick=()=>{const y=get(
"set-llm-transcribe-prompt");y&&(y.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),on=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
 text-xs text-gray-200";return on.map(f=>{const y=f.mcpLocked===!0,k=y?'<div class="text-[10px] text\
-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>':
"",_=y?`<input type="checkbox" id="${c}-auto-sys-${f.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${c}-auto-sys-${f.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${f.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${y?'ti\
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
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(c,u={})=>{on.forEach(m=>{const f=u&&
typeof u=="object"?u[m.key]||{}:{},y=get(`${c}-auto-sys-${m.key}-enabled`),k=get(`${c}-auto-sys-${m.
key}-text`);y&&(m.mcpLocked===!0?y.disabled=!0:y.checked=f.enabled!==!1),k&&(k.value=f.text||"",k.placeholder=
f.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const $n=o((c,u=null)=>{if(u){const m=get(u);m&&(m.checked=!0)}on.forEach(m=>{const f=get(`${c}-auto\
-sys-${m.key}-enabled`),y=get(`${c}-auto-sys-${m.key}-text`);if(f&&(m.mcpLocked!==!0?f.checked=!0:f.
disabled=!0),y){const k=y.placeholder||"";y.value=k}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),ei=o(c=>{const u={};return on.forEach(m=>{const f=get(`${c}\
-auto-sys-${m.key}-enabled`),y=get(`${c}-auto-sys-${m.key}-text`);u[m.key]={enabled:m.mcpLocked===!0?
!0:f?f.checked:!0,text:y?y.value:""}}),u},"collectAutoSystemPromptConfigFromForm");window.ensureAutoSystemPromptSettingsCard=
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
                `,u.appendChild(m)},Tt(),En(),oe();const ti=o(()=>{const c=get("set-default-model");
if(!c)return;const u=c.value;c.innerHTML="",MODELS.forEach(f=>{const y=document.createElement("optgr\
oup");y.label=f.category,(f.items||[]).forEach(k=>{const _=document.createElement("option");_.value=
k.id,_.textContent=k.name,y.appendChild(_)}),c.appendChild(y)});const m=userSettingsSnapshot&&userSettingsSnapshot.
default_model||u||"gemini-3.6-flash";m&&Array.from(c.options).some(f=>f.value===m)&&(c.value=m)},"po\
pulateDefaultModelOptions"),ni=o(()=>{const c=get("set-default-vision-model");if(!c)return;const u=c.
value;c.innerHTML="",MODELS.forEach(f=>{const y=(f.items||[]).filter(_=>{const S=(_.id||"").toLowerCase();
return S.startsWith("gemini-")||S.startsWith("gpt-4o")||S.startsWith("claude-")||S.startsWith("grok-\
3")});if(y.length===0)return;const k=document.createElement("optgroup");k.label=f.category,y.forEach(
_=>{const S=document.createElement("option");S.value=_.id,S.textContent=_.name+" \u2605",k.appendChild(
S)}),c.appendChild(k)});const m=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||u||
"gemini-3-flash-preview";m&&Array.from(c.options).some(f=>f.value===m)&&(c.value=m)},"populateDefaul\
tVisionModelOptions"),In=o(c=>{if(!c)return;cacheUserSettings(c);const u=get("app-global-sys-prompt-\
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
"set-liquid-glass").checked=!!c.liquid_glass_enabled),get("set-auto-search-links")&&(get("set-auto-s\
earch-links").checked=c.auto_search_on_links!==!1),get("set-use-last-settings")&&(get("set-use-last-\
settings").checked=!!c.use_last_chat_settings),get("set-default-model")&&(get("set-default-model").value=
c.default_model||"gemini-3.6-flash"),get("set-default-vision-model")&&(get("set-default-vision-model").
value=c.default_vision_model||"gemini-3-flash-preview"),applyTemporaryChatTimeoutSeconds(c.temp_chat_timeout_seconds),
get("set-default-search")&&(get("set-default-search").checked=!!c.default_enable_search),get("set-de\
fault-url-context")&&(get("set-default-url-context").checked=!!c.default_enable_url_context),get("se\
t-default-maps")&&(get("set-default-maps").checked=!!c.default_enable_maps),get("set-default-python")&&
(get("set-default-python").checked=!!c.default_enable_python),get("set-default-file-creation")&&(get(
"set-default-file-creation").checked=!!c.default_enable_file_creation),get("set-default-thinking")&&
(get("set-default-thinking").checked=!!c.default_enable_thinking),get("set-default-sys-prompt")&&(get(
"set-default-sys-prompt").checked=!!c.default_enable_system_prompt),get("set-default-mcp")&&(get("se\
t-default-mcp").checked=c.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-defa\
ult-thinking-level").value=c.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get(
"set-default-thinking-budget").value=c.default_thinking_budget||4096),get("set-default-reasoning-eff\
ort")&&(get("set-default-reasoning-effort").value=c.default_reasoning_effort||"medium"),get("set-def\
ault-safety")&&(get("set-default-safety").value=c.default_safety_setting||"default"),get("set-e2ee").
checked=c.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=c.bot_detection_enabled!==
!1),get("set-bot-detect-global")&&(get("set-bot-detect-global").checked=c.bot_detection_global_enabled!==
!1);const f=get("bot-status");f&&(c.is_bot_banned?(f.textContent=`BAN\u4E2D: ${c.bot_ban_reason||"Bo\
t detection"}`,f.classList.remove("hidden"),f.classList.add("text-red-400")):f.classList.add("hidden")),
c&&c.theme_color?(applyThemeColor(c.theme_color,!0),syncThemeInputs(c.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=c.username);const y=get("2fa-badge"),k=get("disable-2fa-btn");c.is_2fa_enabled?
(y.innerText="ENABLED",y.classList.replace("bg-gray-700","bg-green-600"),y.classList.replace("text-g\
ray-400","text-white"),k.classList.remove("hidden")):(y.innerText="DISABLED",y.classList.replace("bg\
-green-600","bg-gray-700"),y.classList.replace("text-white","text-gray-400"),k.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!c.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=c.default_2fa_method||"totp");const _=get(
"set-passkey-only-login"),S=get("passkey-only-note"),M=Array.isArray(c.passkey_credentials)?c.passkey_credentials:
[];if(X(M),_){_.checked=!!c.passkey_only_login;const O=M.length>0||!!c.has_webauthn;_.disabled=!O,O||
(_.checked=!1),S&&(O?S.classList.add("hidden"):S.classList.remove("hidden"))}const B=get("mig-status\
-box"),I=get("mig-progress-text"),D=get("mig-progress-bar");if((c.migration_status||"idle")==="proce\
ssing"){B.classList.remove("hidden");const O=(c.migration_progress||"").split("/");if(O.length===2){
const A=parseInt(O[0]||"0",10),H=parseInt(O[1]||"0",10);I&&(I.innerText=`${A} / ${H}`),D&&H>0&&(D.style.
width=`${Math.min(100,Math.floor(A/H*100))}%`)}}else B.classList.add("hidden"),D&&(D.style.width="0%"),
I&&(I.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettingsFormFromData");
window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),snapshotSidebarHistory(
"settings-open-before");const c=await ensureUserSettingsSnapshot();c&&In(c);const u=get("search-box"),
m=u?u.value:"";clearTimeout(searchTimeout);const f=get("settings-search");if(f&&(f.value=""),filterSettings(),
ti(),ni(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(()=>refreshSettingsTabsScroll()),
restoreThreadSearchValue(m,"restored-search-box-open"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-open-after"),[50,200,400,800].forEach(y=>{setTimeout(()=>{restoreThreadSearchValue(m,"rest\
ored-search-box-"+y+"ms"),snapshotSidebarHistory("settings-open-later-"+y+"ms")},y)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),G(),En(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),Ge(!0),Ue(),c||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),wn(),q(),P();try{loadMcpServers()}catch{}};const Bt=o((c=!1)=>{snapshotSidebarHistory("\
settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!c&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),ii=o(()=>{const c=get("set-th\
eme-color"),u=get("set-theme-color-text"),m=get("theme-reset-btn"),f=document.querySelectorAll("#the\
me-presets .theme-swatch"),y=o((k,_=!0)=>{const S=normalizeHex(k);S&&(applyThemeColor(S,_),syncThemeInputs(
S))},"applyFromValue");c&&c.addEventListener("input",()=>y(c.value,!0)),u&&(u.addEventListener("chan\
ge",()=>{const k=normalizeHex(u.value);if(!k){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}y(k,!0)}),u.addEventListener("keydown",k=>{k.key==="Enter"&&(k.preventDefault(),
u.blur())})),m&&(m.onclick=()=>y(THEME_DEFAULT,!0)),f.forEach(k=>{k.addEventListener("click",()=>y(k.
getAttribute("data-color"),!0))})},"bindThemeControls"),si=o(()=>{const c=get("reset-global-sys-prom\
pt");c&&(c.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const u=get(
"reset-set-auto-sys-prompt-defaults");u&&(u.onclick=()=>{$n("set","set-apply-auto-sys-prompt-notices"),
showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const m=get("reset-thread-auto-sys-prompt-defaults");m&&(m.onclick=()=>{$n("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Bt();const Pn=get("settings-header-close");Pn&&(Pn.onclick=()=>Bt());
const jt=get("settings-search");jt&&(jt.addEventListener("input",filterSettings),jt.addEventListener(
"keydown",c=>{if(c.key==="Enter"){const u=get("tab-"+activeSettingsTab);if(!u)return;const m=u.querySelector(
":scope > .settings-match");m&&m.scrollIntoView({behavior:"smooth",block:"start"})}}));const On=get(
"settings-search-clear");On&&On.addEventListener("click",()=>{jt&&(jt.value="",filterSettings(),jt.focus())}),
ii(),si(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const gn=get("set-gemini-local-python-dialog");gn&&(gn.onchange=()=>setGeminiLocalPyDialogEnabled(gn.
checked));const Nn=get("set-gemini-backend");Nn&&(Nn.onchange=()=>syncGeminiBackendUi());const Rn=get(
"set-admin-api-key-mode");Rn&&(Rn.onchange=()=>syncAdminApiKeyModeUi());const hn=get("set-temp-chat-\
timeout-seconds");hn&&(hn.onchange=()=>{applyTemporaryChatTimeoutSeconds(hn.value)});const Bn=get("s\
lash-command-cancel-btn");Bn&&(Bn.onclick=()=>{hidePendingSlashCommandIndicator();const c=get("promp\
t-input");c&&c.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const c=get("set-username"),u=get("set-password"),m=readPromptBarModeFromForm(),f={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:ei(
"set"),theme_color:normalizeHex(get("set-theme-color-text")?get("set-theme-color-text").value:"")||THEME_DEFAULT,
mic_transcribe_mode:get("set-mic-transcribe-mode")?get("set-mic-transcribe-mode").value:"stt_api",stt_model:get(
"set-stt-model")?get("set-stt-model").value:null,llm_transcribe_prompt:get("set-llm-transcribe-promp\
t")?get("set-llm-transcribe-prompt").value:"",enter_to_send:get("set-enter-to-send")?get("set-enter-\
to-send").checked:!1,compact_prompt_mode:m.compact_prompt_mode,minimal_prompt_mode:m.minimal_prompt_mode,
use_sw_cache:get("set-use-sw-cache")?get("set-use-sw-cache").checked:!1,clear_cache_on_version_update:get(
"set-clear-cache-on-version-update")?get("set-clear-cache-on-version-update").checked:!1,liquid_glass_enabled:get(
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
-method").value:"totp",new_username:c?c.value:null,new_password:u?u.value:null},y=get("set-e2ee")?get(
"set-e2ee").checked:!1,k=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
y!==k&&(f.enable_e2ee=y),get("set-openai")&&(f.openai_key=get("set-openai").value),get("set-gemini")&&
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
tent-Type":"application/json"},body:JSON.stringify(f)});if(_.ok){let S="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const D=await _.json();D&&D.message&&(S=D.message)}catch{}Bt();const M=currentUsername,B=CHAT_CONFIG.
enableE2EE;enterToSend=f.enter_to_send,autoSearchOnLinks=f.auto_search_on_links;const I=useSwCache;useSwCache=
f.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!f.clear_cache_on_version_update),
compactPromptMode=f.compact_prompt_mode,minimalPromptMode=f.minimal_prompt_mode,voiceStudioUiEnabled=
f.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=f.temp_chat_timeout_seconds,applyThemeColor(f.theme_color,
!0),syncThemeInputs(f.theme_color),applyLiquidGlassMode(f.liquid_glass_enabled),applyAdaptiveBlurPreference(
get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),minimalPromptMode?
setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),I!==useSwCache&&applyCacheMode(
useSwCache,{forceCleanup:!useSwCache}),showToast(S,"success"),syncClientDebugLogToggle(f.enable_client_debug_log,
"settings saved"),f.new_username&&f.new_username!==M?setTimeout(()=>location.reload(),1e3):f.new_password&&
showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let S={};try{S=await _.json()}catch{}showToast(S.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
son"},body:JSON.stringify({username:u,mode:"single"})}),f=await m.json(),y=get("bot-unban-result");if(m.
ok&&f&&f.status==="ok")y&&(y.textContent=`${u} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
y.classList.remove("hidden")),c&&(c.value="");else{const k=f&&f.error?f.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(k,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const c=get("bot-unban-username"),u=c?c.value.trim():"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const m=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"linked"})}),f=await m.json(),y=get("bot-unban-result");if(m.
ok&&f&&f.status==="ok")y&&(y.textContent=`${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
y.classList.remove("hidden")),c&&(c.value="");else{const k=f&&f.error?f.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(k,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const c=get(
"bot-speed-test-btn"),u=get("bot-speed-test-result");c&&(c.disabled=!0),c&&c.classList.add("opacity-\
60","cursor-not-allowed"),u&&(u.classList.remove("hidden"),u.textContent="\u5B9F\u884C\u4E2D...");try{
const m=o(z=>{u&&(u.textContent=z)},"setBox"),f=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),y=o((z,ie)=>!z||!ie||ie<=0?0:z*8/(ie/1e3)/1e3/1e3,"toMbps"),k=o(z=>Number.isFinite(
z)?`${z.toFixed(0)} ms`:"-","fmtMs"),_=o(z=>Number.isFinite(z)?`${z.toFixed(z>=100?0:1)} Mbps`:"-","\
fmtMbps"),S=o(async(z,ie)=>{const we=await z.json().catch(()=>({}));return we&&we.error?we.error:ie},
"parseErr"),M=[];m("\u6E2C\u5B9A\u4E2D... ping");for(let z=0;z<4;z++){const ie=performance.now(),we=await apiFetch(
`/api/speedtest/ping?_=${f()}`,{cache:"no-store"}),$e=performance.now();if(!we.ok)throw new Error(await S(
we,"ping_failed"));await we.json().catch(()=>({})),M.push($e-ie)}const B=M.reduce((z,ie)=>z+ie,0)/Math.
max(1,M.length),I=Math.min(...M),D=o(async z=>{const ie=performance.now(),we=await apiFetch(`/api/sp\
eedtest/download?bytes=${z}&_=${f()}`,{cache:"no-store"});if(!we.ok)throw new Error(await S(we,"down\
load_failed"));const $e=await we.arrayBuffer(),Oe=performance.now();return{bytes:$e.byteLength||z,ms:Oe-
ie,mbps:y($e.byteLength||z,Oe-ie)}},"runDownload");m(`\u6E2C\u5B9A\u4E2D... ping ${k(B)}
\u6E2C\u5B9A\u4E2D... download`);const ne=[];for(const z of[2*1024*1024,8*1024*1024])ne.push(await D(
z)),m(`\u6E2C\u5B9A\u4E2D... ping ${k(B)}
download ${_(Math.max(...ne.map(ie=>ie.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const O=Math.max(...ne.map(z=>z.mbps)),A=o(async z=>{const ie=new Uint8Array(
z),we=performance.now(),$e=await apiFetch(`/api/speedtest/upload?_=${f()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ie,cache:"no-store"}),Oe=performance.now();if(!$e.ok)
throw new Error(await S($e,"upload_failed"));const Ve=await $e.json().catch(()=>({})),Qe=Number(Ve.bytes_received||
z)||z;return{bytes:Qe,ms:Oe-we,mbps:y(Qe,Oe-we),serverMs:Number(Ve.server_elapsed_ms||0)||0}},"runUp\
load"),H=[];for(const z of[1*1024*1024,4*1024*1024])H.push(await A(z));const V=Math.max(...H.map(z=>z.
mbps)),ve=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",`Ping\
 (avg/min): ${k(B)} / ${k(I)}`,`Download (best): ${_(O)}`,`Upload (best): ${_(V)}`,`Download runs: ${ne.
map(z=>`${Math.round(z.bytes/1024/1024)}MB=${_(z.mbps)}`).join(", ")}`,`Upload runs: ${H.map(z=>`${Math.
round(z.bytes/1024/1024)}MB=${_(z.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\
\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];m(ve.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(m){u&&(u.textContent=`\u30A8\u30E9\u30FC: ${m&&m.message?m.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{c&&(c.disabled=!1,c.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Ue()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Ye()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async c=>{const u=c.target.closest("button");if(!u)return;const m=u.getAttribute("data-id");
if(u.classList.contains("ban-appeal-mark")){m&&await Ye([Number(m)]);return}if(u.classList.contains(
"ban-appeal-status")){const f=u.getAttribute("data-status");m&&f&&await ct({id:Number(m),status:f});
return}if(u.classList.contains("ban-appeal-reply-send")){const f=u.closest("[data-appeal-id]"),y=f?f.
querySelector(".ban-appeal-reply"):null,k=y?y.value:"";m&&await ct({id:Number(m),admin_reply:k});return}
if(u.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";m&&await ct({id:Number(
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
()=>setMarkerMode("crop"));const bn=get("marker-color-picker");bn&&(bn.oninput=c=>setMarkerColor(c.target.
value),bn.onchange=c=>setMarkerColor(c.target.value));const yn=get("marker-opacity");yn&&(yn.oninput=
c=>setMarkerOpacity(c.target.value),yn.onchange=c=>setMarkerOpacity(c.target.value));const rn=get("m\
arker-opacity-number");rn&&(rn.onchange=c=>setMarkerOpacity(c.target.value),rn.onblur=c=>setMarkerOpacity(
c.target.value),rn.onkeydown=c=>{c.key==="Enter"&&(setMarkerOpacity(c.target.value),c.target.blur())}),
document.querySelectorAll("#marker-toolbar .marker-color-chip[data-marker-color]").forEach(c=>{c.onclick=
()=>setMarkerColor(c.getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-vie\
w-reset").onclick=()=>resetMarkerTransform()),get("marker-crop-reset")&&(get("marker-crop-reset").onclick=
()=>clearCropRect()),get("marker-undo")&&(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("ma\
rker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),get("marker-save")&&(get("marker\
-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),initCropCanvas(),
window.addEventListener("resize",()=>{const c=get("marker-modal");!c||c.classList.contains("hidden")||
(applyMarkerTransform(),renderCropOverlay())});const ai=o(()=>{const c=get("upload-modal");return!!(c&&
!c.classList.contains("hidden"))},"isUploadModalOpen"),Ft=get("drop-overlay");let Kt=0;const oi=o(()=>{
ai()||Ft&&(Ft.classList.remove("hidden"),Ft.classList.add("flex"))},"showDropOverlay"),Xt=o(()=>{Kt=
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
files))});const jn=get("bot-admin-modal"),ri=o(c=>{const u=get("bot-admin-list");if(u){if(u.innerHTML=
"",!c||!c.length){u.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}c.forEach((m,f)=>{const y=!!m.is_bot_banned,k=m.bot_detection_enabled!==!1,_=document.createElement(
"div");_.className="flex items-center gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs m\
odel-list-animate",_.style.animationDelay=`${Math.min(f,12)*.02}s`,_.innerHTML=`
                        <div class="flex-1">
                            <div class="text-gray-200 font-bold">${escapeHtml(m.username)}</div>
                            <div class="text-[10px] text-gray-500">${y?"BAN\u4E2D":"\u6B63\u5E38"} ${m.
bot_ban_reason?" / "+escapeHtml(m.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(m.username)}" data-enabled="${k?"1":"0"}">${k?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${y?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(m.username)}" data-bann\
ed="${y?"1":"0"}">${y?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${y?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(m.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-\
2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(m.username)}">\u524A\u9664</button>\

                    `,u.appendChild(_)})}},"renderBotUsers"),Yt=o(async(c="")=>{const u=get("bot-adm\
in-list");u&&(u.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const m=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
c)}`),f=await m.json();m.ok&&f&&f.users?ri(f.users):(u&&(u.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{u&&(u.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),vn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||jn))return;const u=get(
"settings-modal");u&&(u.classList.contains("modal-open")||u.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await Yt(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=vn,window.closeBotAdminModal=(c=!1)=>{(get(
"bot-admin-modal")||jn)&&hideModal("bot-admin-modal"),!c&&location.pathname==="/admin-bots"&&history.
back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{vn()}),get("bot-admin-close")&&(get(
"bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-admin-search-btn")&&(get("bot-admin-se\
arch-btn").onclick=async()=>{await Yt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}),
get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=async()=>{await Yt("")}),get("bo\
t-admin-search")&&get("bot-admin-search").addEventListener("keydown",async c=>{c.key==="Enter"&&await Yt(
get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get("bot-admin-list").onclick=async c=>{
const u=c.target.closest("button");if(!u)return;const m=u.getAttribute("data-user");if(!m)return;let f;
if(u.classList.contains("bot-toggle-detect")){const y=u.getAttribute("data-enabled")!=="1";f=await apiFetch(
"/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:m,
action:"toggle_detection",enabled:y})})}else if(u.classList.contains("bot-toggle-ban"))if(u.getAttribute(
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
let y={};try{y=await f.json()}catch{}showToast(y.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}await Yt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}});const Dt={"\
/settings":{id:"settings-modal",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload\
-modal",open:o(()=>openUploadModal(),"open")},"/library":{id:"lib-modal",open:o(()=>{Jn(!1),showModal(
"lib-modal"),loadLibraryFiles()},"open")},"/history":{id:"history-modal",open:o(()=>window.showHistoryModal(),
"open")},"/branch":{id:"branch-modal",open:o(()=>window.showBranchModal(),"open")},"/paste":{id:"ric\
h-paste-modal",open:o(()=>openRichPasteModal(),"open")},"/camera":{id:"camera-capture-modal",open:o(
()=>openCameraCaptureModal(),"open")},"/edit-image":{id:"marker-modal",open:o(()=>{},"open")},"/chat\
-settings":{id:"thread-modal",open:o(()=>window.openThreadModal(),"open")},"/model":{id:"model-modal",
open:o(()=>openModelModal(),"open")},"/token-details":{id:"token-detail-modal",open:o(()=>showTokenDetailModal(),
"open")},"/encryption-status":{id:"encryption-status-modal",open:o(()=>showEncryptionStatusModal(),"\
open")},"/python-execution":{id:"python-exec-modal",open:o(()=>showPythonExecDetailModal(),"open")},
"/gem":{id:"gem-modal",open:o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="f\
as fa-gem text-blue-500 mr-2"></i>Create New Gem',showModal("gem-modal")},"open")},"/compression":{id:"\
compression-modal",open:o(()=>window.openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-mo\
dal",open:o(()=>vn(),"open")}},Fn=o((c,u=!1)=>{switch(c){case"settings-modal":Bt(u);break;case"uploa\
d-modal":closeUploadModal(u);break;case"camera-capture-modal":closeCameraCaptureModal(u?{skipHistory:!0}:
{});break;case"history-modal":window.closeHistoryModal&&window.closeHistoryModal(u);break;case"lib-m\
odal":window.closeLibModal&&window.closeLibModal(u);break;case"branch-modal":window.closeBranchModal&&
window.closeBranchModal(u);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
u);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(u);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(u);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(u);break;case"token-detail-modal":closeTokenDetail(u);break;case"encryption-s\
tatus-modal":closeEncryptionModal(u);break;case"python-exec-modal":closePythonExecDetail(u);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(u);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(u);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
u);break;case"version-update-modal":const m=localStorage.getItem("app_version")||"";m&&localStorage.
setItem("version_notified",m),hideModal(c);break;default:hideModal(c);break}},"closeModalById");window.
addEventListener("popstate",c=>{let u=!1;Object.values(Dt).forEach(y=>{const k=get(y.id);k&&k.classList.
contains("modal-open")&&location.pathname!==Object.keys(Dt).find(_=>Dt[_].id===y.id)&&(Fn(y.id,!0),u=
!0)});const m=location.pathname.match(/^\/c\/(.+)$/);if(m){const y=decodeURIComponent(m[1]);String(currentThreadId)!==
String(y)&&loadMessages(y,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat(
{skipHistory:!0});const f=Dt[location.pathname];if(f){const y=get(f.id);y&&!y.classList.contains("mo\
dal-open")&&f.open()}});const Dn=location.pathname;Dt[Dn]&&(history.replaceState({},"","/"),setTimeout(
()=>Dt[Dn].open(),500)),get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const c=get(
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
"",wn()};async function wn(){const u=await(await apiFetch("/api/feedback?all=1")).json(),m=get("fb-l\
ist");m.innerHTML="",(u.items||[]).filter(k=>!u.is_admin||k.user_id===void 0||k.user_id===null||!0).
forEach(k=>{if(u.is_admin)return;const _=document.createElement("div");_.className="p-2 rounded bord\
er border-gray-700 bg-gray-800/50",_.innerHTML=`<div class="text-[11px] text-gray-400">${k.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(k.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(k.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
k.status)}</div>${k.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(k.
admin_reply)}</div>`:""}`,m.appendChild(_)});const f=get("fb-admin-panel"),y=get("fb-admin-list");u.
is_admin?(f.classList.remove("hidden"),y.innerHTML="",(u.items||[]).forEach(k=>{const _=document.createElement(
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
ave").onclick=async()=>{const S=_.querySelector(".fb-status").value,M=_.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${k.id}/update`,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({status:S,admin_reply:M})}),wn()},y.appendChild(_)})):f.classList.add("hidd\
en")}if(o(wn,"loadFeedback"),window.setupTOTP=async()=>{const u=await(await apiFetch("/api/2fa/totp/\
setup",{method:"POST"})).json();get("totp-qr").src=u.qr_image,get("totp-secret-disp").innerText=u.secret,
get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const c=get("totp-ver\
ify-code").value;if(!c)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({code:c})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const c=get("register-webauthn-btn"),u=get("webauthn-nam\
e"),m=u?String(u.value||"").trim():"";try{c&&(c.disabled=!0);const f=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),y=await f.json();if(!f.ok){showToast(y.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const _=await(await ensureWebAuthnJson()).create({publicKey:y}),S=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},_,{name:m}))}),M=await S.json().catch(()=>({}));S.ok?(u&&(u.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(M.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
return}if(c.key==="Enter"){c.preventDefault();const f=m.substring(1).toLowerCase(),y=loadedGems.filter(
k=>k.name.toLowerCase().includes(f)||k.description&&k.description.toLowerCase().includes(f));y[gemSelectedIndex]?
selectGemSuggestion(y[gemSelectedIndex]):y.length>0&&selectGemSuggestion(y[0]);return}if(c.key==="Es\
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
if(u[f].kind==="file"){const y=u[f].getAsFile();y&&m.push(y)}m.length>0&&(c.preventDefault(),await handleFiles(
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
let ze,Ae=null,ln=[],xn=!1,cn=null,Ht=null,Ct=null,Zt=null,kn=0,dn=!1,en=null,qt=null,ht=null,tn=null,
un=null,Lt=null,pn=null,mn=null;function Hn(){const c=get("mic-waveform");if(!c)return[];if(Array.isArray(
pn)&&pn.length)return pn;c.innerHTML="";const u=[];for(let m=0;m<24;m++){const f=document.createElement(
"span");f.className="block rounded-full",f.style.background="rgba(252, 165, 165, 0.92)",f.style.width=
"2px",f.style.transition="height 75ms linear, opacity 75ms linear",f.style.height="2px",f.style.opacity=
"0.4",u.push(f),c.appendChild(f)}return pn=u,u}o(Hn,"ensureMicWaveformBars");function Mt(c,u="hidden"){
const m=get("mic-recording-indicator"),f=get("mic-recording-text");if(m){if(mn&&(clearTimeout(mn),mn=
null),u==="hidden"){m.classList.add("hidden");return}f&&c&&(f.innerText=c),m.classList.remove("hidde\
n"),u==="recording"?m.style.color="rgb(252 165 165)":u==="processing"?m.style.color="rgb(253 224 71)":
m.style.color="rgb(209 213 219)"}}o(Mt,"setMicRecordingIndicator");function qn(){Hn().forEach(u=>{u.
style.height="2px",u.style.opacity="0.35"})}o(qn,"resetMicWaveformBars");function At(){if(un&&(cancelAnimationFrame(
un),un=null),tn){try{tn.disconnect()}catch{}tn=null}if(qt){try{qt.close()}catch{}qt=null}ht=null,Lt=
null,qn()}o(At,"stopMicWaveform");function Gn(c){At();const u=Hn();if(!u.length)return;const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;try{qt=new m,ht=qt.createAnalyser(),ht.fftSize=
256,ht.smoothingTimeConstant=0,tn=qt.createMediaStreamSource(c),tn.connect(ht),Lt=new Uint8Array(ht.
frequencyBinCount)}catch{At();return}const f=o(()=>{if(!ht||!Lt)return;ht.getByteFrequencyData(Lt);const y=Math.
max(1,Math.floor(Lt.length/u.length));for(let k=0;k<u.length;k++){const S=(Lt[Math.min(Lt.length-1,k*
y)]||0)/255,M=Math.max(2,Math.round(2+S*10));u[k].style.height=`${M}px`,u[k].style.opacity=`${.35+S*
.65}`}un=requestAnimationFrame(f)},"render");f()}o(Gn,"startMicWaveform");function fn(){if(cn&&(clearInterval(
cn),cn=null),Zt){try{Zt.disconnect()}catch{}Zt=null}if(Ht){try{Ht.close()}catch{}Ht=null}Ct=null}o(fn,
"stopSilenceMonitor");function Un(c){if(!isStsModel()||!stsOpt("sts-auto-send"))return;fn();const u=window.
AudioContext||window.webkitAudioContext;if(!u)return;Ht=new u,Ct=Ht.createAnalyser(),Ct.fftSize=2048,
Zt=Ht.createMediaStreamSource(c),Zt.connect(Ct);const m=new Uint8Array(Ct.fftSize),f=getStsSilenceMs(),
y=.02;kn=0,dn=!1,cn=setInterval(()=>{if(!Ct)return;Ct.getByteTimeDomainData(m);let k=0;for(let S=0;S<
m.length;S++){const M=(m[S]-128)/128;k+=M*M}if(Math.sqrt(k/m.length)>y){dn||(dn=!0),kn=Date.now();return}
dn&&Date.now()-kn>f&&ze&&ze.state==="recording"&&ze.stop()},200)}o(Un,"startSilenceMonitor");const Tn=class Tn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null}async start(u,m,f,y={}){this.model=f,this.ws=new WebSocket(`${m}?access_token=${u}`),
this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Send\
ing setup...");const S=!!(y&&y.transcriptionConfig),M={setup:{model:`models/${f}`,generationConfig:{
responseModalities:S?["TEXT"]:["AUDIO"]},inputAudioTranscription:S?y.transcriptionConfig||{}:{},outputAudioTranscription:{}}};
y.speechConfig&&(M.setup.generationConfig.speechConfig=y.speechConfig),y.thinkingConfig&&(M.setup.generationConfig.
thinkingConfig=y.thinkingConfig),y.translationConfig&&(M.setup.translationConfig=y.translationConfig),
console.log("Sending setup:",JSON.stringify(M)),this.ws.send(JSON.stringify(M))},this.ws.onmessage=S=>this.
_handleMessage(S),this.ws.onerror=S=>{console.error("Gemini Live WebSocket error:",S),this.onError&&
this.onError(S)},this.ws.onclose=S=>{console.log("Gemini Live WebSocket closed:",S.code,S.reason),this.
onClose&&this.onClose(S)},this.audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
this.stream=await navigator.mediaDevices.getUserMedia({audio:!0});const k=this.audioContext.createMediaStreamSource(
this.stream);this.processor=this.audioContext.createScriptProcessor(4096,1,1),this.userAudioChunks=[];
const _=new MediaRecorder(this.stream);_.ondataavailable=S=>{S.data.size>0&&this.userAudioChunks.push(
S.data)},_.start(500),this.backupRecorder=_,this.processor.onaudioprocess=S=>{if(!this.ws||this.ws.readyState!==
WebSocket.OPEN||!this.setupComplete)return;const M=S.inputBuffer.getChannelData(0),B=new Int16Array(
M.length);for(let I=0;I<M.length;I++)B[I]=Math.max(-1,Math.min(1,M[I]))*32767;this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(B.buffer))),mimeType:"\
audio/pcm;rate=16000"}}}))},k.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(u){
const m=JSON.parse(u.data);if(console.log("Gemini Live raw message received:",m),m.setupComplete&&(console.
log("Gemini Live setup complete confirmed"),this.setupComplete=!0),m.serverContent){const f=m.serverContent;
f.modelTurn&&f.modelTurn.parts.forEach(y=>{if(y.text&&(y.thought?(console.log("Gemini thought delta:",
y.text),this.assistantThought+=y.text):(console.log("Gemini transcript delta (parts):",y.text),this.
assistantText+=y.text)),y.inlineData&&y.inlineData.data){const k=y.inlineData.data;console.log("Gemi\
ni audio chunk received, size:",k.length),this.rtPlayer&&this.rtPlayer.addChunk(k);const _=atob(k),S=new Uint8Array(
_.length);for(let M=0;M<_.length;M++)S[M]=_.charCodeAt(M);this.assistantAudioChunks.push(S)}}),f.outputTranscription&&
(console.log("Gemini output transcription delta:",f.outputTranscription.text),this.assistantText.includes(
f.outputTranscription.text)||(this.assistantText+=f.outputTranscription.text)),f.inputTranscription&&
(console.log("User input transcription delta:",f.inputTranscription.text),this.inputTranscript+=f.inputTranscription.
text,this.interimInputTranscript=""),f.interimInputTranscription&&(console.log("User interim transcr\
iption:",f.interimInputTranscription.text),this.interimInputTranscript=f.interimInputTranscription.text)}
this.onMessage&&this.onMessage(m)}stop(){this.ws&&this.ws.close(),this.processor&&this.processor.disconnect(),
this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().forEach(u=>u.stop()),
this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const u=new Blob(this.assistantAudioChunks),
m=await this._blobToBase64(u),f=new Blob(this.userAudioChunks),y=await this._blobToBase64(f);return{
user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.assistantThought,
audio_base64:m,user_audio_base64:y}}_blobToBase64(u){return new Promise(m=>{const f=new FileReader;f.
onloadend=()=>m(f.result.split(",")[1]),f.readAsDataURL(u)})}};o(Tn,"GeminiLiveClient");let _n=Tn;const Cn=class Cn{constructor(u=24e3){
const m=window.AudioContext||window.webkitAudioContext;this.ctx=new m({sampleRate:u}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(u){if(!this.ctx)return;const m=atob(u),f=new Uint8Array(
m.length);for(let B=0;B<m.length;B++)f[B]=m.charCodeAt(B);const y=new Int16Array(f.buffer),k=new Float32Array(
y.length);for(let B=0;B<y.length;B++)k[B]=y[B]/32768;const _=this.ctx.createBuffer(1,k.length,this.ctx.
sampleRate);_.getChannelData(0).set(k),this.ctx.state==="suspended"&&await this.ctx.resume();const S=this.
ctx.createBufferSource();S.buffer=_,S.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const M=Math.max(this.ctx.currentTime,this.nextStartTime);
S.start(M),this.nextStartTime=M+_.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Cn,
"RealTimeAudioPlayer");let nn=Cn;const Ln=class Ln{constructor(){this.active=!1,this.capturing=!1,this.
sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=
null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript=
"",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=
null,this.saved=!1,this.saving=!1,this.stopping=!1}isActive(){return this.active}async start(){if(this.
active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const u=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const y=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=y.id!==null&&y.id!==void 0?String(y.id):y.id,setTemporaryChatUiState(!!(y&&y.
is_temporary)),setCurrentChatHeaderTitle(y&&y.title),applyTemporaryChatRuntimeMeta(y||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+y.id),get("welcome-screen").classList.add("hidden")}catch(f){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+f.message,"\
error",!0);return}const m={model:u,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,target_lang:isGeminiLiveTranslateModel()&&get("sts-target-lang")?get(
"sts-target-lang").value:""};setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const f=await apiFetch("/a\
pi/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)}),
y=await f.json().catch(()=>({}));if(!f.ok)throw new Error(y.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=y.session_id,this.rateIn=y.rate_in||this.rateIn,this.rateOut=y.rate_out||this.rateOut,
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
value:y}=await this.reader.read();if(f)break;m+=u.decode(y,{stream:!0});let k;for(;(k=m.indexOf(`

`))>=0;){const _=m.slice(0,k);m=m.slice(k+2);for(const S of _.split(`
`)){if(!S.startsWith("data: "))continue;let M=null;try{M=JSON.parse(S.slice(6))}catch{continue}this.
_handleEvent(M)}}}}catch(f){if(f&&f.name==="AbortError")return;this.active&&(this.streamError=f&&f.message?
f.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(u){
if(u)switch(u.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new nn(this.rateOut||24e3),Ut=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Ut=null}_startCapture(){const u=window.
AudioContext||window.webkitAudioContext;if(!u)throw new Error("AudioContext not supported");return this.
audioCtx=new u({sampleRate:this.rateIn||24e3}),navigator.mediaDevices.getUserMedia(Vn()).then(m=>{this.
stream=m;const f=this.audioCtx.createMediaStreamSource(m),y=this.rateIn||24e3,k=this.audioCtx.sampleRate,
_=4096;this.processor=this.audioCtx.createScriptProcessor(_,1,1),this.processor.onaudioprocess=S=>{if(!this.
active||!this.capturing)return;const M=S.inputBuffer.getChannelData(0),B=li(M,k,y);!B||!B.byteLength||
this._sendAudio(B)},f.connect(this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(u){
if(!this.sessionId||!this.active)return;const m="/api/realtime/audio?session_id="+encodeURIComponent(
this.sessionId),f={method:"POST",credentials:"include",headers:{"X-CSRF-Token":csrfToken,"Content-Ty\
pe":"application/octet-stream"},body:u},y=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions==
"function"?window.ProgressSpinner.manualRequestOptions(f):f;fetch(m,y).catch(()=>{})}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(u=>u.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}fn(),At()}async stop(){if(!this.active)return;this.active=
!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0);try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify({session_id:this.sessionId})})}catch{}const u=Date.now(),m=this.responseDoneCount;
let f=this.lastAudioAt;for(;Date.now()-u<2e4&&!(this.responseDoneCount>m||(this.lastAudioAt>f&&(f=this.
lastAudioAt),!this.speechActive&&Date.now()-u>2e3&&Date.now()-f>2500));)await new Promise(y=>setTimeout(
y,250));await this._save()}async _save(){if(!this.saved){this.saved=!0,this.saving=!0;try{const u=await apiFetch(
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
classList.remove("bg-red-600","animate-pulse"),u.classList.add("bg-gray-700"))}};o(Ln,"RealtimeVoice\
Session");let Sn=Ln;function li(c,u,m){let f=c;if(u!==m&&u>0&&m>0){const k=u/m,_=Math.floor(f.length/
k),S=new Float32Array(_);for(let M=0;M<_;M++)S[M]=f[Math.min(Math.floor(M*k),f.length-1)];f=S}const y=new Int16Array(
f.length);for(let k=0;k<f.length;k++){const _=Math.max(-1,Math.min(1,f[k]));y[k]=_<0?_*32768:_*32767}
return y.buffer}o(li,"pcm16FromFloat32");const Gt=new Sn;(()=>{const m={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let f=null,y=null,k=!1,_=null,S=!1,M=0,B=0,I=null,
D="idle",ne=!1,O=null;const A=o(E=>document.getElementById(E),"$"),H=o(E=>{const F=Object.assign({},
E||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?
window.ProgressSpinner.manualRequestOptions(F):(F.progressSpinner=!1,F)},"noSpinner");function V(E,F){
D=F;const me=A("lyria-status-text"),ee=A("lyria-status-dot");me&&(me.textContent=E),ee&&(ee.className=
"w-2 h-2 rounded-full inline-block "+(m[F]||m.idle)),we(),$e()}o(V,"setStatus");function ve(){const E=B?
Math.floor((Date.now()-B)/1e3):0,F=String(Math.floor(E/60)).padStart(2,"0"),me=String(E%60).padStart(
2,"0");return`${F}:${me}`}o(ve,"formatElapsed");function z(){B||(B=Date.now());const E=A("lyria-elap\
sed");E&&(E.textContent=ve()),I||(I=window.setInterval(()=>{const F=A("lyria-elapsed");F&&(F.textContent=
ve())},1e3))}o(z,"startElapsedTimer");function ie(){I&&(window.clearInterval(I),I=null)}o(ie,"stopEl\
apsedTimer");function we(){const E=A("lyria-play-btn"),F=A("lyria-pause-btn"),me=A("lyria-stop-btn"),
ee=A("lyria-reset-btn"),he=!!f,Ee=D==="streaming"||D==="connecting";if(E){E.disabled=ne||!he;const We=E.
querySelector("i");We&&(We.className="fas fa-play")}F&&(F.disabled=ne||!Ee),me&&(me.disabled=ne||!he||
!Ee),ee&&(ee.disabled=ne||!he||!Ee)}o(we,"updateTransportButtons");function $e(){const E=A("lyria-sa\
ve-btn");if(!E)return;const F=!!f&&D!=="idle"&&D!=="connecting"&&D!=="error";E.classList.toggle("hid\
den",!F)}o($e,"updateSaveButton");function Oe(E,F){const me=A("lyria-prompt-rows");if(!me)return;const ee=document.
createElement("div");ee.className="flex items-center gap-2",ee.innerHTML=`
                        <input type="text" value="${escapeHtml(E||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof F=="num\
ber"?F:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof F=="number"?F:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const he=ee.querySelector('input[type="range"]'),Ee=ee.querySelector(".lyria-w\
eight-label");he&&Ee&&he.addEventListener("input",()=>{Ee.textContent=parseFloat(he.value).toFixed(1)});
const We=ee.querySelector(".lyria-prompt-remove");We&&We.addEventListener("click",()=>{me.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ee.remove()}),ee.classList.add("lyria-prompt-row-wrap"),me.appendChild(
ee)}o(Oe,"addPromptRow");function Ve(){const E=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),F=[];return E.forEach(me=>{const ee=me.querySelector('input[type="text"]'),he=me.querySelector(
'input[type="range"]'),Ee=(ee?ee.value:"").trim();Ee&&F.push({text:Ee,weight:parseFloat(he?he.value:
1)||1})}),F}o(Ve,"collectPrompts");function Qe(){const E={},F=o(ui=>{const An=A(ui);return An&&An.value!==
""?parseFloat(An.value):void 0},"num"),me=F("lyria-bpm");me!==void 0&&(E.bpm=Math.round(me));const ee=F(
"lyria-guidance");ee!==void 0&&(E.guidance=ee);const he=F("lyria-density");he!==void 0&&(E.density=he);
const Ee=F("lyria-brightness");Ee!==void 0&&(E.brightness=Ee);const We=F("lyria-temperature");We!==void 0&&
(E.temperature=We);const Be=A("lyria-scale");Be&&Be.value&&(E.scale=Be.value);const Je=A("lyria-mode");
Je&&Je.value&&(E.music_generation_mode=Je.value);const st=A("lyria-mute-bass"),It=A("lyria-mute-drum\
s"),Zn=A("lyria-only-bass-drums");return st&&(E.mute_bass=st.checked),It&&(E.mute_drums=It.checked),
Zn&&(E.only_bass_and_drums=Zn.checked),E}o(Qe,"collectConfig");function Et(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([F,me])=>{
const ee=A(F),he=A(me);!ee||!he||ee.addEventListener("input",()=>{const Ee=parseFloat(ee.value);he.textContent=
F==="lyria-bpm"?String(Math.round(Ee)):Ee.toFixed(1)})})}o(Et,"bindRangeLabels");function bt(){if(_){
try{_.close()}catch{}_=null}S=!1,M=0}o(bt,"resetPlayback");function yt(){if(k=!1,y&&typeof y.abort==
"function")try{y.abort()}catch{}y=null}o(yt,"closeStream");async function dt(){yt(),y=new AbortController,
k=!0;try{const E=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(f)}`,H({method:"\
GET",signal:y.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!E.ok){const he=await E.
json().catch(()=>({}));throw new Error(he.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const F=E.body.getReader(),me=new TextDecoder;let ee="";for(;k;){const{done:he,value:Ee}=await F.read();
if(he)break;ee+=me.decode(Ee,{stream:!0});const We=ee.split(`

`);ee=We.pop();for(const Be of We){const Je=Be.split(`
`).find(It=>It.startsWith("data: "));if(!Je)continue;const st=Je.slice(6);try{const It=JSON.parse(st);
ge(It)}catch{}}}}catch(E){if(E&&E.name==="AbortError")return;k&&(V("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{k&&f&&dt()},1200))}finally{k=!1}}o(dt,"openStream");function ge(E){
if(E&&E.snapshot){const F=E.status;if(F==="error"){V("\u30A8\u30E9\u30FC","error"),ie();return}if(F===
"closed"||F==="stopped"){V("\u7D42\u4E86","closed"),ie();return}V(F==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",F==="paused"?"paused":"connecting");return}if(E&&E.audio){V("\u518D\u751F\u4E2D...",
"streaming"),z(),re(E.audio);return}if(E&&E.error){V("\u30A8\u30E9\u30FC: "+E.error,"error"),ie();return}
if(E&&E.final){V("\u7D42\u4E86","closed"),ie(),we();return}}o(ge,"handleStreamMessage");function re(E){
if(!E)return;if(!_){const Be=window.AudioContext||window.webkitAudioContext;if(!Be)return;_=new Be({
sampleRate:48e3}),S=!1,M=0}let F;try{const Be=atob(E);F=new Uint8Array(Be.length);for(let Je=0;Je<Be.
length;Je++)F[Je]=Be.charCodeAt(Je)}catch{return}const me=new Int16Array(F.buffer),ee=Math.floor(me.
length/2);if(ee<1)return;const he=_.createBuffer(2,ee,48e3);for(let Be=0;Be<2;Be++){const Je=he.getChannelData(
Be);for(let st=0;st<ee;st++)Je[st]=me[st*2+Be]/32768}_.state==="suspended"&&_.resume();const Ee=_.createBufferSource();
Ee.buffer=he,Ee.connect(_.destination),S||(M=_.currentTime+.08,S=!0);const We=Math.max(_.currentTime,
M);Ee.start(We),M=We+he.duration}o(re,"playChunk");async function ue(E,F){const me=await fetch("/api\
/gemini/music/command",H({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:f,type:E},F||{}))})),ee=await me.json().catch(()=>({}));if(!me.ok)throw new Error(
ee.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ee}
o(ue,"apiCommand");async function Ie(){if(ne)return;const E=Ve();if(!E.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ne=!0,we(),V("\u63A5\u7D9A\u4E2D...","connecting");try{
const F=await fetch("/api/gemini/music/start",H({method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({weighted_prompts:E,config:Qe()})})),me=await F.json().catch(()=>({}));if(!F.
ok)throw new Error(me.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
f=me.session_id,O=Qe(),V("\u63A5\u7D9A\u4E2D...","connecting"),dt()}catch(F){V("\u30A8\u30E9\u30FC: "+
F.message,"error"),showToast("Lyria RealTime: "+F.message,"error",!0)}finally{ne=!1,we()}}o(Ie,"star\
tSession");async function Se(E){if(f){ne=!0,we();try{await ue("control",{action:E}),E==="PLAY"?V("\u518D\u751F\
\u4E2D...","streaming"):E==="PAUSE"?V("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):E==="STOP"?V("\u505C\u6B62\u4E2D",
"stopped"):E==="RESET_CONTEXT"&&V("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(F){showToast("Lyria RealTime: "+F.message,"error",!0),V("\u30A8\u30E9\u30FC: "+F.
message,"error")}finally{ne=!1,we()}}}o(Se,"control");async function vt(){if(!f)return;const E=Ve();
if(!E.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ne=!0;try{await ue("prompts",{weighted_prompts:E}),V("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
D==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(F){showToast("Lyria RealTime: "+F.message,"error",!0)}finally{ne=!1,we()}}o(vt,"app\
lyPrompts");async function wt(){if(!f)return;const E=Qe(),F=O||{},me=E.bpm!==void 0&&E.bpm!==F.bpm,ee=E.
scale!==void 0&&E.scale!==F.scale,he=me||ee;ne=!0;try{await ue("config",{config:E,reset_context:he}),
O=E,V(he?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",D==="paused"?"paused":"streaming"),showToast(
he?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Ee){showToast("Lyria RealT\
ime: "+Ee.message,"error",!0)}finally{ne=!1,we()}}o(wt,"applyConfig");async function $t(){if(f){ne=!0,
V("\u4FDD\u5B58\u4E2D...","connecting"),we();try{const E=await fetch("/api/gemini/music/save",H({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:f,thread_id:currentThreadId||
null})})),F=await E.json().catch(()=>({}));if(!E.ok)throw new Error(F.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
V("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ie(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),F.thread_id&&(currentThreadId=String(F.thread_id),history.pushState({},"","/c/"+F.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(F.thread_id||currentThreadId),Qn(!0)}catch(E){
V("\u30A8\u30E9\u30FC: "+E.message,"error"),showToast("Lyria RealTime: "+E.message,"error",!0)}finally{
ne=!1,we()}}}o($t,"saveSession");async function De(){if(yt(),f)try{await fetch("/api/gemini/music/ca\
ncel",H({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:f})}))}catch{}
f=null,ie(),bt(),V("\u6E96\u5099\u5B8C\u4E86","idle")}o(De,"cancelSession");function it(){const E=A(
"lyria-prompt-rows");E&&(E.innerHTML=""),Oe("",1),O=null,B=0,["lyria-bpm","lyria-guidance","lyria-de\
nsity","lyria-brightness","lyria-temperature"].forEach(ee=>{const he=A(ee);he&&(he.value=ee==="lyria\
-bpm"?"120":ee==="lyria-guidance"?"4":ee==="lyria-temperature"?"1.1":"0.5")});const F=A("lyria-scale");
F&&(F.value="");const me=A("lyria-mode");me&&(me.value="QUALITY"),["lyria-mute-bass","lyria-mute-dru\
ms","lyria-only-bass-drums"].forEach(ee=>{const he=A(ee);he&&(he.checked=!1)}),Et()}o(it,"resetContr\
ols");function Qn(E){yt(),f&&fetch("/api/gemini/music/cancel",H({method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify({session_id:f})})).catch(()=>{}),f=null,k=!1,ie(),bt(),hideModal(
"lyria-studio-modal")}o(Qn,"closeAndCleanup");function Mn(E){if(!isLyriaRealtimeModel()){showToast("\
Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const F=A("lyria-studio-modal");if(F&&F.classList.contains("modal-open")&&f){if(E&&
typeof E=="string"){const ee=A("lyria-prompt-rows");ee&&(ee.innerHTML=""),Oe(E,1)}return}if(f&&De(),
it(),E&&typeof E=="string"){const ee=A("lyria-prompt-rows");ee&&(ee.innerHTML=""),Oe(E,1)}f=null,k=!1,
ie(),bt(),V("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Mn,"open");function di(){
const E=A("lyria-open-studio-btn");E&&E.addEventListener("click",()=>Mn(""));const F=A("lyria-studio\
-close");F&&F.addEventListener("click",()=>Qn(!1));const me=A("lyria-play-btn");me&&me.addEventListener(
"click",()=>{if(!f){Ie();return}Se("PLAY")});const ee=A("lyria-pause-btn");ee&&ee.addEventListener("\
click",()=>Se("PAUSE"));const he=A("lyria-stop-btn");he&&he.addEventListener("click",()=>Se("STOP"));
const Ee=A("lyria-reset-btn");Ee&&Ee.addEventListener("click",()=>Se("RESET_CONTEXT"));const We=A("l\
yria-add-prompt-btn");We&&We.addEventListener("click",()=>Oe("",1));const Be=A("lyria-apply-prompts-\
btn");Be&&Be.addEventListener("click",vt);const Je=A("lyria-apply-config-btn");Je&&Je.addEventListener(
"click",wt);const st=A("lyria-save-btn");st&&st.addEventListener("click",$t),Et(),it(),window.openLyriaStudio=
Mn}return o(di,"init"),{init:di,open:Mn}})().init(),(()=>{let c=null,u=null;const m=o(O=>document.getElementById(
O),"$");function f(){return isStsModel()&&voiceStudioUiEnabled!==!1}o(f,"isStudioMode");function y(){
const O=get("model-select")?get("model-select").value:"",A=m("voice-studio-title");A&&(O==="gpt-tran\
scribe"||O==="gpt-live-transcribe"?A.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":
O==="gemini-3.5-live-translate-preview"?A.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
A.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(y,"updateTitle");function k(){const O=m("voi\
ce-studio-transcript");O&&(O.innerHTML='<div class="text-[10px] text-gray-500">\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</\
div>')}o(k,"resetTranscript");function _(O,A){if(!A||!String(A).trim())return;const H=m("voice-studi\
o-transcript");if(!H||!window.VoiceStudioOpen)return;const V=O==="user"?"\u3042\u306A\u305F":"AI",ve=O===
"user"?"text-cyan-300":"text-gray-100",z=H.querySelectorAll(".voice-studio-line");let ie=null;for(let $e=z.
length-1;$e>=0;$e--)if(z[$e].dataset.role===O){ie=z[$e];break}const we=`<span class="${ve} font-bold\
">${escapeHtml(V)}:</span> <span class="text-gray-200">${escapeHtml(A)}</span>`;if(ie)ie.innerHTML=we;else{
const $e=H.querySelector(".text-gray-500");$e&&$e.remove();const Oe=document.createElement("div");Oe.
className="voice-studio-line",Oe.dataset.role=O,Oe.innerHTML=we,H.appendChild(Oe)}H.scrollTop=H.scrollHeight}
o(_,"log");function S(){const O=m("sts-panel"),A=m("voice-studio-panel-host");O&&A&&O.parentNode!==A&&
(c=O.parentNode,A.appendChild(O));const H=m("file-preview"),V=m("voice-studio-file-host");H&&V&&H.parentNode!==
V&&(u=H.parentNode,V.appendChild(H),V.classList.remove("hidden"))}o(S,"movePanelIntoModal");function M(){
const O=m("sts-panel");O&&c&&O.parentNode!==c&&c.appendChild(O);const A=m("file-preview");A&&u&&A.parentNode!==
u&&u.appendChild(A);const H=m("voice-studio-file-host");H&&H.classList.add("hidden"),c=null,u=null}o(
M,"movePanelBack");function B(){if(!f()){showToast("\u97F3\u58F0\u7CFB\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}S();const O=m("sts-panel");O&&O.classList.remove("hidden"),y(),k(),window.VoiceStudioOpen=
!0,showModal("voice-studio-modal")}o(B,"open");function I(){if(window.VoiceStudioOpen&&(Ae||ze&&ze.state===
"recording"||Gt.isActive())&&Wn(),window.VoiceStudioOpen=!1,M(),hideModal("voice-studio-modal"),isStsModel()&&
voiceStudioUiEnabled!==!1){const O=m("sts-panel");O&&O.classList.add("hidden")}}o(I,"close");function D(){
window.VoiceStudioOpen&&I()}o(D,"closeIfOpen");function ne(){window.VoiceStudioOpen=!1;const O=m("vo\
ice-studio-open-btn");O&&O.addEventListener("click",()=>B());const A=m("voice-studio-close");A&&A.addEventListener(
"click",()=>I()),window.VoiceStudio={open:B,close:I,closeIfOpen:D,log:_,isStudioMode:f}}return o(ne,
"init"),{init:ne,open:B,close:I,closeIfOpen:D,log:_,isStudioMode:f}})().init();let Ut=null;function zn(){
if(Ut&&(Ut.stop(),Ut=null),en){try{en.pause()}catch{}try{en.src=""}catch{}en=null}}o(zn,"stopStsPlay\
back");async function gi(c){zn();const u=new Audio;return u.src=c,u.preload="auto",u.autoplay=!0,u.playsInline=
!0,en=u,await u.play(),new Promise(m=>{u.onended=()=>m("ended"),u.onerror=()=>m("error")})}o(gi,"pla\
yStsAudio");function Wn(){if(Gt.isActive()){Gt._cancel();return}if(Ae){Ae.stop(),Ae=null,zn(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),At();return}ze&&ze.state===
"recording"&&(xn=!0,ze.stop())}o(Wn,"cancelRecording");function Vn(){if(isStsModel())return{audio:!0};
const u=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},m={channelCount:1};return u.echoCancellation&&(m.echoCancellation=!1),u.
noiseSuppression&&(m.noiseSuppression=!1),u.autoGainControl&&(m.autoGainControl=!1),{audio:m}}o(Vn,"\
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
"Failed to save Gemini Live session:",u),setStsStatus("Error saving",!1)}return}if(Gt.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
Gt.stop();return}if(ze&&ze.state==="recording"){ze.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||Mt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const m=new Audio;
m.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",m.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const f={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(f.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const O=get("sts-custom\
-vocab").value.split(/[,、\n]/).map(A=>A.trim()).filter(Boolean);O.length&&(f.custom_vocabulary=O.slice(
0,1e3))}}else f.voice=get("sts-voice")?get("sts-voice").value:"Kore",f.thinking_level=get("sts-think\
ing-level")?get("sts-thinking-level").value:"minimal",f.include_thoughts=get("sts-include-thoughts")?
get("sts-include-thoughts").checked:!1,isGeminiLiveTranslateModel()&&get("sts-target-lang")&&(f.target_lang=
get("sts-target-lang").value);const y=await apiFetch("/api/gemini/session",{method:"POST",headers:{"\
Content-Type":"application/json"},body:JSON.stringify(f)});if(!y.ok)throw new Error("Failed to get s\
ession token");const{token:k,url:_}=await y.json(),S=get("model-select").value,M=get("sts-voice")?get(
"sts-voice").value:"Kore",B=get("sts-thinking-level")?get("sts-thinking-level").value:"minimal",I=get(
"sts-include-thoughts")?get("sts-include-thoughts").checked:!1;if(Ae=new _n,stsOpt("sts-auto-play")&&
!isGeminiLiveTranscribeModel()&&(Ae.rtPlayer=new nn),isGeminiLiveTranscribeModel()){const O=get("sts\
-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",A={languageCodes:[]};if((O==="SMART"||
O==="VERBATIM")&&(A.mode=O),get("sts-custom-vocab")){const H=get("sts-custom-vocab").value.split(/[,、\n]/).
map(V=>V.trim()).filter(Boolean);H.length&&(A.customVocabulary=H.slice(0,1e3))}await Ae.start(k,_,S,
{transcriptionConfig:A})}else if(isGeminiLiveTranslateModel()){const O=get("sts-target-lang")?get("s\
ts-target-lang").value:"ja";await Ae.start(k,_,S,{translationConfig:{targetLanguageCode:O,echoTargetLanguage:!0}})}else
await Ae.start(k,_,S,{speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:M}}},thinkingConfig:{
thinkingLevel:B,includeThoughts:I}});ze=Ae.backupRecorder,ze.onstop=()=>{Ae&&get("mic-btn").click()};
let D=!0,ne="live-sts-"+Date.now();Ae.onMessage=O=>{if(O.serverContent){if(isGeminiLiveTranscribeModel()){
const A=Ae.interimInputTranscript,H=Ae.inputTranscript,V=H+(A&&!H.endsWith(A)?(H?`
`:"")+A:""),ve=get("chat-messages");let z=document.getElementById(ne);z||(z=document.createElement("\
div"),z.id=ne,z.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-lg\
 border border-slate-700/50",z.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ve.appendChild(z),ve.scrollTop=ve.scrollHeight);const ie=z.
querySelector(".message-content");ie.innerText=V||"\u8074\u304D\u53D6\u308A\u4E2D...",ve.scrollTop=ve.
scrollHeight,window.VoiceStudio&&H&&window.VoiceStudio.log("user",H);return}if(O.serverContent.modelTurn){
D&&(setStsStatus("Gemini is speaking...",!1),D=!1);const A=get("chat-messages");let H=document.getElementById(
ne);H||(H=document.createElement("div"),H.id=ne,H.className="flex flex-col gap-2 mb-4 assistant-mess\
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
                                            `,A.appendChild(H),A.scrollTop=A.scrollHeight);const V=H.
querySelector(".thought-container"),ve=H.querySelector(".message-content");Ae.assistantThought&&(V.classList.
remove("hidden"),V.innerText=Ae.assistantThought),ve.innerText=Ae.assistantText,A.scrollTop=A.scrollHeight,
window.VoiceStudio&&(Ae.inputTranscript&&window.VoiceStudio.log("user",Ae.inputTranscript),Ae.assistantText&&
window.VoiceStudio.log("assistant",Ae.assistantText))}}},setStsStatus("Listening...",!0),get("mic-bt\
n").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),Gn(Ae.
stream),Un(Ae.stream);return}catch(m){showToast("Gemini Live connection failed: "+m.message,"error",
!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){await Gt.start();return}isStsModel()||
(qn(),Mt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));const c=await navigator.mediaDevices.
getUserMedia(Vn());ze=new MediaRecorder(c),ln=[],xn=!1;const u=isStsModel();ze.ondataavailable=m=>ln.
push(m.data),ze.onstop=async()=>{if(xn){ln=[],get("file-preview").classList.add("hidden"),c.getTracks().
forEach(_=>_.stop()),fn(),At(),u||(Mt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),mn=setTimeout(()=>Mt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const m=new Blob(ln,{type:"audio/web\
m"}),f=new File([m],"recording.webm",{type:"audio/webm"}),y=new FormData;y.append("file",f),get("fil\
e-preview").classList.remove("hidden");const k=u;get("file-name").innerText=k?"Processing voice...":
"Transcribing...";try{if(k){if(!currentThreadId){const V=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=V.id!==null&&V.id!==void 0?String(V.id):V.id,setTemporaryChatUiState(!!(V&&V.
is_temporary)),setCurrentChatHeaderTitle(V&&V.title),applyTemporaryChatRuntimeMeta(V||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+V.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),y.append("model",get(
"model-select").value),y.append("thread_id",currentThreadId),get("sts-voice")&&y.append("sts_voice",
get("sts-voice").value||""),get("sts-speed")&&y.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&y.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&y.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&y.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&y.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const _=await apiFetch(
"/sts",{method:"POST",body:y});if(!_.ok){const H=await _.json().catch(()=>({}));throw new Error(H.error||
"Speech-to-speech failed")}const S=_.body.getReader(),M=new TextDecoder;let B="",I=null,D=null;stsOpt(
"sts-auto-play")&&(D=new nn,Ut=D),setStsStatus(isTranscriptionModel()?"Transcribing...":"Processing \
audio...",!0);let ne=!0,O="",A="";for(;;){const{done:H,value:V}=await S.read();if(H)break;B+=M.decode(
V,{stream:!0});const ve=B.split(`
`);B=ve.pop();for(const z of ve){if(!z.trim())continue;const ie=JSON.parse(z);if(ie.error)throw new Error(
ie.error);ie.audio_delta&&D&&(ne&&(setStsStatus("Playing response...",!1),ne=!1),await D.addChunk(ie.
audio_delta)),ie.input_delta&&(O+=ie.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",O)),
ie.transcript_delta&&(A+=ie.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assistant",
A)),ie.final&&(I=ie)}}window.VoiceStudio&&!O.trim()&&window.VoiceStudio.log("user","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),
I&&(I.audio_url||I.transcription_only)&&(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus(
"Listening...",!0),get("mic-btn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{
const _=get("set-mic-transcribe-mode");if(!!(_&&_.value==="llm")&&!supportsAudioInputModel()){showToast(
"\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306FLLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}y.append("llm_model",get("model-select")&&get("model-select").value||"");const B=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:y})).json();if(B.transcript){const I=get("prompt-inp\
ut");I.value+=(I.value?" ":"")+B.transcript,I.style.height="auto",I.style.height=I.scrollHeight+"px"}else
showToast(B.error||"Transcription failed","error",!0)}}catch(_){showToast("Audio processing error: "+
_.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),c.getTracks().forEach(_=>_.
stop()),fn(),At(),k||Mt("","hidden"),k&&setStsStatus("Tap to speak",!1)}},ze.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Mt("\u9332\u97F3\u4E2D\u2026","recording"),Gn(c)),Un(c),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{At(),isStsModel()||Mt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const sn=o((c,u)=>{if(!c)return;const m=c.querySelector("span");m?m.textContent=u:c.textContent=
u},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const c=lib.
selected.size,u=get("lib-del-btn"),m=get("lib-download-btn"),f=get("lib-attach-btn"),y=get("lib-rena\
me-btn"),k=get("lib-usage-btn");if(u&&(u.disabled=c===0,sn(u,c?`\u524A\u9664 (${c})`:"\u524A\u9664")),
m&&(m.disabled=c===0,sn(m,c?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${c})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
f&&(f.disabled=c===0,sn(f,c?`\u6DFB\u4ED8 (${c})`:"\u6DFB\u4ED8")),y&&(y.disabled=c!==1,sn(y,"\u540D\u524D\u5909\u66F4")),
k&&(k.disabled=c!==1,sn(k,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const _=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",_&&c>0)}};function Jn(c){lib.
attachMode=!!c}o(Jn,"setLibAttachMode");const Kn=o((c=!1)=>{Jn(c),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(c=!1)=>{hideModal("lib-modal"),!c&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>Kn(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>Kn(!0)),
get("lib-search")){let c=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),c&&clearTimeout(c),c=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const c=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=c,get("lib-sort").onchange=()=>{const u=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,u),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const ci=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",get("gem-default-model").value="",get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-con\
tainer").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"open\
GemModal");window.closeGemModal=(c=!1)=>{hideModal("gem-modal"),!c&&location.pathname==="/gem"&&history.
back()},get("add-gem-btn").onclick=()=>ci(),get("save-gem-btn").onclick=async()=>{const c=get("gem-n\
ame").value,u=get("gem-desc").value,m=get("gem-inst").value,f=collectGemFixedPrompts();if(c&&m){const y=editingGemUuid?
"PUT":"POST",k=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
k,{method:y,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:c,description:u,instruction:m,
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
try{const y=decodeURIComponent(m),k=new Blob([y],{type:"text/plain"}),_=URL.createObjectURL(k),S=document.
createElement("a");S.href=_;let B={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[f]||f;(f.length>8||/[^a-z0-9]/.test(f))&&(B="txt");let I=`code.${B}`;f==="dockerfile"&&(I="Docke\
rfile"),f==="makefile"&&(I="Makefile"),S.download=I,document.body.appendChild(S),S.click(),document.
body.removeChild(S),URL.revokeObjectURL(_)}catch(y){console.error("Download failed",y)}}if(c.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(c.target.closest(".coding-target-btn")),
c.target.closest(".copy-btn")){const u=c.target.closest(".copy-btn"),m=u.getAttribute("data-code");m&&
window.copyCode(u,m)}if(c.target.closest(".html-preview-btn")){const m=c.target.closest(".html-previ\
ew-btn").getAttribute("data-code");m&&openHtmlCodePreview(m)}if(c.target.closest(".canvas-preview-bt\
n")){const u=c.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(u)}}),document.querySelectorAll(
".modal-overlay").forEach(c=>{c.addEventListener("click",u=>{u.target===c&&Fn(c.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){const e=get(
"file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-progress-bar"),
s=get("file-preview-thumbs"),a=get("upload-modal-status-text"),r=get("upload-modal-total-progress"),
l=get("upload-modal-total-progress-bar");if(!e||!t)return;if(s){const T=document.querySelectorAll("#\
upload-list .upload-row");s.innerHTML="",T.forEach(($,j)=>{const J=$.getAttribute("data-local-url"),
X=$.getAttribute("data-filename"),Te=$.querySelector("img.upload-preview")!==null;let P;if(Te){let q=J;
if(!q&&X){const Y=X.replace(/^\d+\//,"");q=buildAttachmentPreviewUrl(Y)}q&&(P=document.createElement(
"img"),P.src=q,P.className="thumb-item shadow-sm",P.dataset.viewerSrc=q,P.dataset.viewerFilename=X||
q.split("/").pop(),P.onclick=function(Y){Y.preventDefault(),openImageViewer(this.dataset.viewerSrc,"\
.thumb-item")},P.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}P||(P=d("FILE")),
P.style.animationDelay=`${j*32}ms`,s.appendChild(P)}),T.length>0?s.classList.remove("hidden"):s.classList.
add("hidden")}function d(T){const $=document.createElement("div");return $.className="thumb-item bg-\
gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bold",$.innerText=
T,$}o(d,"createFileThumb");const p=collectImageUrlsForSend(),g=uploadProgressState.total,h=uploadProgressState.
completed,v=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.classList.add("hidden"),
r&&r.classList.add("hidden"),s&&s.classList.add("hidden"));const b=get("send-btn"),w=get("mic-btn"),
x=get("mask-btn"),L=isStopMode;if(v>0?(b&&(b.disabled=!0),w&&(w.disabled=!0),x&&(x.disabled=!0)):L||
(b&&(b.disabled=!1),w&&(w.disabled=!1),x&&(x.disabled=!1)),v>0){const T=`Preparing... (${h}/${g})`;e.
classList.remove("hidden"),t.innerText=T,a&&(a.innerText=`(${h}/${g})`);let $=h*100,j=0;for(let Te in uploadProgressState.
perFilePct)$+=uploadProgressState.perFilePct[Te],j++;const J=g>0?$/(g*100)*100:0,X=`${Math.min(100,J)}\
%`;n&&i&&(n.classList.remove("hidden"),i.style.width=X),r&&l&&(r.classList.remove("hidden"),l.style.
width=X)}else a&&(a.innerText=""),r&&r.classList.add("hidden"),p.length>0?(e.classList.remove("hidde\
n"),t.innerText=`${p.length} files ready`,n&&n.classList.add("hidden")):(e.classList.add("hidden"),t.
innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}o(updateFilePreview,"update\
FilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get("mask-name");!e||!t||(currentMaskImage?
(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.split("/").pop()}`):(e.classList.
add("hidden"),t.innerText=""))}o(updateMaskPreview,"updateMaskPreview");const markerToolHints={draw:"\
\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],s=t[3];return`#${n}${n}${i}${i}${s}${s}`}
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
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),s=get("ma\
rker-color-picker");s&&s.value!==e&&(s.value=e);const a=get("marker-opacity");a&&a.value!==i&&(a.value=
i);const r=get("marker-opacity-number");r&&r.value!==i&&(r.value=i);const l=get("marker-opacity-valu\
e");l&&(l.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(p=>{const g=normalizeMarkerHexColor(p.getAttribute("data-marker-color"));p.classList.
toggle("active",g===e)})}o(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}o(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}o(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const s=get("marker-tool-hi\
nt");s&&(s.textContent=markerToolHints[e]||"");const a=get("marker-crop-reset");a&&a.classList.toggle(
"hidden",e!=="crop");const r=get("marker-canvas");r&&(r.style.pointerEvents=e==="crop"?"none":"auto");
const l=get("marker-crop-canvas");l&&(l.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.
cropRect||markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}
o(setMarkerMode,"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}
o(clearCropRect,"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)
return;const t=Math.max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,
y:0,w:t,h:n})}o(resetCropRectToFull,"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.
scale<=markerView.minScale+1e-4){markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marke\
r-stage"),t=get("marker-viewport");if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(
1,e.clientHeight||0),s=Math.max(1,t.offsetWidth||t.clientWidth||0),a=Math.max(1,t.offsetHeight||t.clientHeight||
0);if(n<=1||i<=1||s<=1||a<=1)return;const r=(n-s)/2,l=(i-a)/2,d=s*markerView.scale,p=a*markerView.scale,
g=Math.min(n*.45,Math.max(24,n*.12)),h=Math.min(i*.45,Math.max(24,i*.12)),v=g-r-d,b=n-g-r,w=h-l-p,x=i-
h-l,L=o((T,$,j)=>Number.isFinite(T)?$>j?($+j)/2:Math.min(j,Math.max($,T)):0,"clampOffset");markerView.
offsetX=L(markerView.offsetX,v,b),markerView.offsetY=L(markerView.offsetY,w,x)}o(clampMarkerViewOffset,
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
i&&(i.textContent=n);const s=e.getAttribute("data-filename");s&&setAttachmentNameForPath(s,n)}o(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}o(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}o(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
o(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const s=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,s),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}o(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const s=e.querySelector(".truncate"),a=normalizeAttachmentDisplayName(
s?s.textContent:"");return a||getAttachmentNameForPath(t)}o(getRowAttachmentName,"getRowAttachmentNa\
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
"canvas");i.width=t,i.height=n;const s=i.getContext("2d");s?(s.drawImage(e,0,0,t,n),markerState.baseImageData=
s.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}o(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=o((r,l,d=null,p=!1)=>{if(!r)return;const g=Math.max(0,r.x),h=Math.max(0,r.y),v=Math.
max(1,r.w),b=Math.max(1,r.h);d&&(t.fillStyle=d,t.fillRect(g,h,v,b)),t.save(),p&&t.setLineDash([6,4]),
t.strokeStyle=l,t.lineWidth=2,t.strokeRect(g+.5,h+.5,Math.max(1,v-1),Math.max(1,b-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,s=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!s)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const r=Math.max(0,i.x),l=Math.max(0,i.y),d=Math.max(1,i.w),p=Math.max(1,i.h);t.clearRect(r,l,d,p),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(r=>n(r,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((s,a,r)=>{const l=normalizeAttachmentPath(s);if(!l)return;const d=normalizeAttachmentSource(
a),p=normalizeAttachmentDisplayName(r)||getAttachmentNameForPath(l),g=t.get(l);if(g===void 0){const b=e.
length;t.set(l,b),e.push({path:l,source:d,name:p});return}const h=e[g];if(!h)return;const v=normalizeAttachmentSource(
h.source);(v==="unknown"&&d!=="unknown"||v==="library"&&d==="upload")&&(h.source=d),!normalizeAttachmentDisplayName(
h.name)&&p&&(h.name=p)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(s=>{const a=s.getAttribute("data-filename");n(a,getRowAttachmentSource(s),getRowAttachmentName(
s));const r=s.getAttribute("data-original-filename");s.dataset.attachOriginal==="1"&&n(r,getRowOriginalAttachmentSource(
s),getAttachmentNameForPath(r))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
s=>{n(s,getAttachmentSourceForPath(s),getAttachmentNameForPath(s))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const a=[];currentImageUrls.forEach(r=>{const l=normalizeAttachmentPath(r);if(!l)return;const d=isAudioPath(
l),p=isVideoPath(l);if(d&&!t.audio||p&&!t.video){d&&(n+=1),p&&(i+=1);return}a.push(l)}),a.length!==currentImageUrls.
length&&(currentImageUrls=a)}const s=get("upload-list");if(s&&(s.querySelectorAll("[data-filename]").
forEach(a=>{const r=a.getAttribute("data-filename");r&&!currentImageUrls.includes(r)&&(isAudioPath(r)||
isVideoPath(r))&&(setRowMarkerState(a,!1),a.remove())}),s.children.length===0&&(s.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const a=[];n&&a.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&a.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${a.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
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
arker-attach-original");i&&(i.checked=e.dataset.attachOriginal==="1");const s=get("marker-image"),a=get(
"marker-canvas"),r=get("marker-crop-canvas");if(a){const l=a.getContext("2d");l&&l.clearRect(0,0,a.width,
a.height)}if(r){const l=r.getContext("2d");l&&l.clearRect(0,0,r.width,r.height)}resetMarkerTransform(),
showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState({modal:"marker"},"","\
/edit-image"),s&&(s.onload=()=>{if(!get("marker-stage")||!a)return;const d=Math.max(1,Math.floor(s.clientWidth)),
p=Math.max(1,Math.floor(s.clientHeight));a.width=d,a.height=p,a.style.width=`${d}px`,a.style.height=
`${p}px`,a.style.left="0px",a.style.top="0px",r&&(r.width=d,r.height=p,r.style.width=`${d}px`,r.style.
height=`${p}px`,r.style.left="0px",r.style.top="0px"),markerState.naturalWidth=s.naturalWidth||d,markerState.
naturalHeight=s.naturalHeight||p;const g=a.getContext("2d");g&&g.clearRect(0,0,a.width,a.height),prepareMarkerBaseCanvas(
s,d,p),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},s.src=t)}o(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
o(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(r=>{const l=r&&r.rowObj?r.rowObj.row:null,d=l?l.getAttribute("data-local-url"):null;d&&URL.revokeObjectURL(
d)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.clear();const e=get(
"file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");t&&(t.innerHTML="",t.
classList.add("hidden")),updateFilePreview(),updateMaskPreview();const n=get("upload-list");n&&(n.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>');
const i=get("file-input");i&&(i.value="");const s=get("photo-input");s&&(s.value="");const a=get("ma\
sk-input");a&&(a.value="")}o(resetUploadState,"resetUploadState");async function uploadMaskFile(e){if(!e)
return;const t=new FormData;t.append("file",e);try{const n=await fetch(CHAT_CONFIG.urls.upload,{method:"\
POST",body:t}),i=await n.json();n.ok&&i.filename?(currentMaskImage=i.filename,updateMaskPreview()):showToast(
i.error||"Mask upload failed","error",!0)}catch{showToast("Mask upload failed","error",!0)}}o(uploadMaskFile,
"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){const n=get("camera-status");n&&(n.textContent=
e||"",n.classList.toggle("text-red-300",!!t),n.classList.toggle("text-gray-400",!t))}o(setCameraCaptureStatus,
"setCameraCaptureStatus");function updateCameraCapturePendingUi(){const e=cameraCapturePendingFiles.
length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,t.textContent=e?`\u6DFB\u4ED8 (${e}\
)`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=e===0||cameraCaptureBusy);const i=get(
"camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.forEach((s,a)=>{const r=document.
createElement("div");r.className="relative rounded overflow-hidden border border-gray-700 bg-black a\
spect-square",r.innerHTML=`
                        <img src="${s}" alt="capture ${a+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${a+1}</div>
                    `,i.appendChild(r)}),i.classList.toggle("hidden",e===0))}o(updateCameraCapturePendingUi,
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
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let s=null;for(const a of i)
try{const r=await navigator.mediaDevices.getUserMedia(a);cameraCaptureStream=r,t.srcObject=r;try{await t.
play()}catch{}const l=r.getVideoTracks&&r.getVideoTracks()[0],d=l&&l.getSettings?l.getSettings():{},
p=String(d.facingMode||"").toLowerCase();p==="user"||p==="environment"?cameraCaptureFacingMode=p:cameraCaptureFacingMode=
e;const g=get("camera-capture-btn");return g&&(g.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),r}catch(r){s=r}throw s||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
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
const e=new Date,t=o(s=>String(s).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}o(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const s=get("camera-attach-btn");s&&(s.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const a=t.getContext("2d");if(!a)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");a.drawImage(e,0,0,t.width,t.height);
const r=await new Promise((d,p)=>{t.toBlob(g=>{g?d(g):p(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),l=new File([r],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(l),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(r)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(a){
const r=a&&a.message?a.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
r,!0),showToast(r,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}o(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),s=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),s&&(s.disabled=!0);const a=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${a.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{await handleFiles(a,{openModal:!1}),showToast(`${a.length}\u679A\u306E\
\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(r){const l=r&&r.message?r.message:
"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";showToast(l,"\
error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.classList.contains(
"hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}o(attachCameraCapturedFiles,
"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay=="function"&&window.
hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.pathname!=="/upload"&&
history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-info");if(e){const n=(get(
"model-select")?get("model-select").value:"").toLowerCase(),i=n.includes("deepseek")&&n!=="deepseek-\
v4-flash-vision-exp";e.classList.toggle("hidden",!i)}_syncVisionModelDisplay()}o(openUploadModal,"op\
enUploadModal");function _syncVisionModelDisplay(){const e=get("vision-model-display");if(!e)return;
const t=currentVisionModel;if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(s=>{s.id===t&&(n=s.
name)})),e.textContent=n}else e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}o(_syncVisionModelDisplay,
"_syncVisionModelDisplay");function _openVisionModelSelector(){window._visionPickerActive=!0,openModelModal(),
setTimeout(()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},50)}o(_openVisionModelSelector,
"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.hideDropOverlay=="function"&&
window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname==="/upload"&&history.back()}
o(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){const e=get("upload-list");
if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(n=>{const i=n.getAttribute(
"data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(n,{source:getAttachmentSourceForPath(
n),displayName:getAttachmentNameForPath(n)})}),e.children.length===0&&(e.innerHTML='<div class="text\
-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
o(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}o(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const s=e.split("/").pop()||e,a=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||s,r=(s.split(".").pop()||"").toLowerCase(),l=["png","jpg","jpeg","webp",
"gif"].includes(r),d=buildFileUrl(e),p=l?buildAttachmentPreviewUrl(e):d,g=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,h=document.createElement("div");h.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",h.dataset.uploadId=g,h.setAttribute("data-filename",e),h.dataset.fileSource=
n,h.dataset.displayName=a,h.dataset.defaultDisplayName=a,h.dataset.sendNameCustomized="";const v=escapeHtml(
a),b=l&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",w=l?`<img src="${p}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';h.innerHTML=`
                <div class="flex items-center gap-3">
                    ${w}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${v}</div>
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
            `;const x=h.querySelector(".upload-preview");x&&(x.onclick=()=>openFileViewer(d,getRowAttachmentName(
h)||a));const L=h.querySelector(".upload-send-name");L&&(L.onclick=()=>promptRowAttachmentName(h));const T=h.
querySelector(".upload-remove");T&&(T.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const j=h.getAttribute("data-filename");j&&(currentImageUrls=currentImageUrls.
filter(J=>J!==j)),setRowMarkerState(h,!1),h.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const $=h.querySelector(".upload-marker");return $&&($.onclick=()=>openMarkerModalForRow(h)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,a),i.prepend(h),{row:h,bar:h.querySelector(".upload-progress > div"),
status:h.querySelector(".upload-status"),uploadId:g}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const s=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=s,i.dataset.defaultDisplayName=s,i.dataset.sendNameCustomized="";const a=escapeHtml(
s),r=e&&e.type&&e.type.startsWith("image/");let l='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const d=r&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let p="";r?(p=URL.createObjectURL(e),l=`<img src="${p}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${a}">`):(p=URL.createObjectURL(e),l=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${l}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${a}</div>
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
            `,p&&i.setAttribute("data-local-url",p);const g=i.querySelector(".upload-preview");g&&(g.
onclick=()=>{const w=i.getAttribute("data-filename"),x=w?buildFileUrl(w):i.getAttribute("data-local-\
url"),L=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||w||"";openFileViewer(x,L)});const h=i.
querySelector(".upload-remove");h&&(h.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const w=i.getAttribute("data-local-url");w&&URL.revokeObjectURL(w);const x=i.
getAttribute("data-filename");x&&(currentImageUrls=currentImageUrls.filter(L=>L!==x)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const v=i.querySelector(".upload-marker");v&&(v.onclick=()=>openMarkerModalForRow(i));const b=i.querySelector(
".upload-send-name");return b&&(b.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(!e)
return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);try{const i=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),s=await i.json();if(!i.ok){const h=s&&s.error?s.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast(h,"error",!0),!1}const a=s.upload_id,r=s.chunk_size||10*1024*1024,l=Math.ceil(
e.size/r);for(let h=0;h<l;h++){const v=h*r,b=Math.min(e.size,v+r),w=e.slice(v,b);if(!await new Promise(
L=>{const T=new XMLHttpRequest;T.open("POST","/upload/chunk",!0),T.setRequestHeader("X-CSRF-Token",csrfToken),
T.upload.onprogress=j=>{if(j.lengthComputable&&t&&t.bar){const J=v+j.loaded,X=Math.min(100,Math.floor(
J/e.size*100));t.bar.style.width=`${X}%`,t.status&&(t.status.textContent=`${X}%`),t.uploadId&&updateGlobalUploadProgress(
t.uploadId,X)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},T.onload=()=>{T.status>=
200&&T.status<300?L(!0):L(!1)},T.onerror=()=>L(!1);const $=new FormData;$.append("upload_id",a),$.append(
"index",String(h)),$.append("total",String(l)),$.append("chunk",w,e.name),T.send($)}))return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:a})}),p=await d.json();if(d.ok&&p&&p.filename){if(t&&t.row&&t.uploadId&&
uploadCancelTokens.has(t.uploadId))return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const b=t.
row.getAttribute("data-local-url");b&&URL.revokeObjectURL(b),t.row.removeAttribute("data-local-url");
const w=t.row.querySelector("img.upload-preview");if(w){const x=p.filename.replace(/^\d+\//,"");w.src=
buildAttachmentPreviewUrl(x)}}const h=normalizeAttachmentPath(p.filename);if(h&&currentImageUrls.push(
h),t&&t.row&&(t.row.setAttribute("data-filename",h||p.filename),setRowAttachmentSource(t.row,"upload"),
h)){const b=isRowAttachmentNameCustomized(t.row),w=defaultAttachmentDisplayName(h),x=b&&normalizeAttachmentDisplayName(
t.row.dataset.displayName)||w;t.row.dataset.defaultDisplayName=w,setRowAttachmentName(t.row,x)}return h&&
setAttachmentSourceForPath(h,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(p.filenames)&&p.filenames.length?p.filenames:[p.filename]).forEach(b=>addLibraryFileFromPath(
b)),!0}const g=p&&p.error?p.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(g,"error",!0),!1}catch{return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
i=!0);const s=o(()=>{i&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),i=!1)},
"finishUploadOp"),a=new XMLHttpRequest;a.open("POST",CHAT_CONFIG.urls.upload,!0),a.setRequestHeader(
"X-CSRF-Token",csrfToken),a.upload.onprogress=l=>{if(l.lengthComputable&&t&&t.bar){const d=Math.min(
100,Math.floor(l.loaded/l.total*100));t.bar.style.width=`${d}%`,t.status&&(t.status.textContent=`${d}\
%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,d)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},a.onload=()=>{let l={};try{l=JSON.parse(a.responseText||"{}")}catch{}if(a.status>=
200&&a.status<300&&l&&l.filename){if(t&&t.row&&t.uploadId&&uploadCancelTokens.has(t.uploadId)){t.row&&
t.row.parentNode&&t.row.remove(),s(),n(!1);return}if(t&&t.row){const g=t.row.getAttribute("data-loca\
l-url");g&&URL.revokeObjectURL(g),t.row.removeAttribute("data-local-url");const h=t.row.querySelector(
"img.upload-preview");if(h){const v=l.filename.replace(/^\d+\//,"");h.src=buildAttachmentPreviewUrl(
v)}}const d=normalizeAttachmentPath(l.filename);if(d&&currentImageUrls.push(d),t&&t.row&&(t.row.setAttribute(
"data-filename",d||l.filename),setRowAttachmentSource(t.row,"upload"),d)){const g=isRowAttachmentNameCustomized(
t.row),h=defaultAttachmentDisplayName(d),v=g&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
h;t.row.dataset.defaultDisplayName=h,setRowAttachmentName(t.row,v)}d&&setAttachmentSourceForPath(d,"\
upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(l.filenames)&&
l.filenames.length?l.filenames:[l.filename]).forEach(g=>addLibraryFileFromPath(g)),s(),n(!0)}else{const d=l&&
l.error?l.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(d,"error",!0),s(),n(!1)}},a.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),s(),n(!1)};const r=new FormData;r.append("file",e),a.send(r)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(p=>{n+=p.length});const i=new Float32Array(n);
let s=0;e.forEach(p=>{i.set(p,s),s+=p.length});const a=new ArrayBuffer(44+i.length*2),r=new DataView(
a),l=o((p,g)=>{for(let h=0;h<g.length;h++)r.setUint8(p+h,g.charCodeAt(h))},"writeString");l(0,"RIFF"),
r.setUint32(4,36+i.length*2,!0),l(8,"WAVE"),l(12,"fmt "),r.setUint32(16,16,!0),r.setUint16(20,1,!0),
r.setUint16(22,1,!0),r.setUint32(24,t,!0),r.setUint32(28,t*2,!0),r.setUint16(32,2,!0),r.setUint16(34,
16,!0),l(36,"data"),r.setUint32(40,i.length*2,!0);let d=44;for(let p=0;p<i.length;p++){const g=Math.
max(-1,Math.min(1,i[p]));r.setInt16(d,g<0?g*32768:g*32767,!0),d+=2}return new Blob([r],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),s=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=s),e.row.dataset.displayName=
s,i||(e.row.dataset.defaultDisplayName=s);const a=e.row.getAttribute("data-local-url");a&&URL.revokeObjectURL(
a);const r=URL.createObjectURL(t);e.row.setAttribute("data-local-url",r);const l=t.type&&t.type.startsWith(
"image/"),d=escapeHtml(s),p=l?`<img src="${r}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${d}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',g=e.row.querySelector(".upload-preview");g&&(g.outerHTML=p);const h=e.row.querySelector(
".upload-preview");h&&(h.onclick=()=>{const b=e.row.getAttribute("data-filename"),w=b?buildFileUrl(b):
e.row.getAttribute("data-local-url");openFileViewer(w,getRowAttachmentName(e.row)||s||b||"")});const v=e.
row.querySelector(".upload-marker");v&&v.classList.toggle("hidden",!l),l||(setRowMarkerState(e.row,!1),
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
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let s=!1,a=0,
r=markerView.scale,l={x:0,y:0},d={x:0,y:0},p=[],g=16,h="",v=null,b=null,w=null,x=null,L=!1,T=null;const $=o(
C=>{const R=e.getBoundingClientRect(),G=(C.clientX-R.left)*(e.width/R.width),K=(C.clientY-R.top)*(e.
height/R.height);return{x:G,y:K}},"getPoint"),j=o((C,R)=>({x:(C.x+R.x)/2,y:(C.y+R.y)/2}),"getMid"),J=o(
(C,R)=>Math.hypot(C.x-R.x,C.y-R.y),"getDist");let X=!1;const Te=o(()=>{v||(v=document.createElement(
"canvas"),b=v.getContext("2d")),w||(w=document.createElement("canvas"),x=w.getContext("2d")),(v.width!==
e.width||v.height!==e.height)&&(v.width=e.width,v.height=e.height),(w.width!==e.width||w.height!==e.
height)&&(w.width=e.width,w.height=e.height)},"ensureDrawBuffers"),P=o(()=>{if(!t||!v||!w)return;const C=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(0,0,e.width,e.
height),t.drawImage(v,0,0),t.save(),t.globalAlpha=C,t.drawImage(w,0,0),t.restore()},"renderDrawPrevi\
ew"),q=o(()=>{x&&(x.strokeStyle=h,x.fillStyle=h,x.lineWidth=g,x.lineCap="round",x.lineJoin="round")},
"applyMarkerBrush"),Y=o(C=>{if(!C)return!1;if(p.length===0)return p.push(C),!0;const R=p[p.length-1],
G=C.x-R.x,K=C.y-R.y,Z=Math.hypot(G,K),ye=Math.max(.35,g*.04);if(Z<ye)return!1;const Q=Math.max(1,g*.25),
le=Math.max(1,Math.ceil(Z/Q));for(let U=1;U<=le;U++){const fe=U/le;p.push({x:R.x+G*fe,y:R.y+K*fe})}return!0},
"appendStrokePoint"),de=o(()=>{if(x&&(x.clearRect(0,0,w.width,w.height),p.length!==0)){if(q(),p.length===
1){const C=p[0];x.beginPath(),x.arc(C.x,C.y,g/2,0,Math.PI*2),x.fill();return}if(x.beginPath(),x.moveTo(
p[0].x,p[0].y),p.length===2)x.lineTo(p[1].x,p[1].y);else{for(let G=1;G<p.length-2;G++){const K=p[G],
Z=p[G+1],ye=j(K,Z);x.quadraticCurveTo(K.x,K.y,ye.x,ye.y)}const C=p[p.length-2],R=p[p.length-1];x.quadraticCurveTo(
C.x,C.y,R.x,R.y)}x.stroke()}},"renderStrokeLayer"),oe=o((C,R)=>{if(!C||!R)return null;const G=Math.min(
C.x,R.x),K=Math.min(C.y,R.y),Z=Math.abs(C.x-R.x),ye=Math.abs(C.y-R.y);return{x:G,y:K,w:Z,h:ye}},"nor\
malizeMosaicRect"),xe=o(C=>{const R=n?Number(n.value||16):16,G=Math.max(6,Math.floor(R)),K=Math.floor(
G/2);return{x:C.x-K,y:C.y-K,w:G,h:G}},"buildMosaicRectFromPoint"),pe=o(()=>{const C=document.createElement(
"canvas");C.width=e.width,C.height=e.height;const R=C.getContext("2d");if(!R)return null;markerState.
baseCanvas&&R.drawImage(markerState.baseCanvas,0,0),R.drawImage(e,0,0);try{return R.getImageData(0,0,
e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),ke=o(C=>{if(!t||!C)return!1;const R=pe();
if(!R)return!1;const G=n?Number(n.value||16):16,K=Math.max(4,Math.floor(G/2)),Z=Math.max(0,Math.floor(
C.x)),ye=Math.max(0,Math.floor(C.y)),Q=Math.min(e.width,Math.ceil(C.x+C.w)),le=Math.min(e.height,Math.
ceil(C.y+C.h));if(Q<=Z||le<=ye)return!1;for(let U=ye;U<le;U+=K)for(let fe=Z;fe<Q;fe+=K){const et=Math.
min(K,Q-fe),Ce=Math.min(K,le-U),tt=Math.min(e.width-1,Math.max(0,fe+Math.floor(et/2))),Ke=(Math.min(
e.height-1,Math.max(0,U+Math.floor(Ce/2)))*e.width+tt)*4,ot=R.data[Ke],xt=R.data[Ke+1],gt=R.data[Ke+
2];t.fillStyle=`rgb(${ot},${xt},${gt})`,t.fillRect(fe,U,et,Ce)}return!0},"applyMosaicRect"),te=o(C=>{
if(!t)return;if(i.set(C.pointerId,{x:C.clientX,y:C.clientY}),i.size>=2){const G=Array.from(i.values()),
K=G[0],Z=G[1];s=!0,X=!1,p=[],L=!1,T=null,markerState.mosaicPreviewRect=null,a=J(K,Z)||1,r=markerView.
scale,l={x:markerView.offsetX,y:markerView.offsetY},d=j(K,Z),renderCropOverlay(),e.setPointerCapture&&
e.setPointerCapture(C.pointerId),C.preventDefault();return}if(s||markerState.mode==="crop")return;X=
!0;const R=$(C);if(markerState.mode==="mosaic")L=!0,T=R,markerState.mosaicPreviewRect=xe(R),renderCropOverlay();else{
if(Te(),!b||!x)return;b.clearRect(0,0,v.width,v.height),b.drawImage(e,0,0),x.clearRect(0,0,w.width,w.
height),g=n?Number(n.value||16):16,h=normalizeMarkerHexColor(markerState.colorHex),p=[],Y(R),de(),markerState.
hasStroke=!0,P()}e.setPointerCapture&&e.setPointerCapture(C.pointerId),C.preventDefault()},"start"),
se=o(C=>{if(i.has(C.pointerId)&&i.set(C.pointerId,{x:C.clientX,y:C.clientY}),s&&i.size>=2){const G=Array.
from(i.values()),K=G[0],Z=G[1],ye=j(K,Z),Q=J(K,Z)||1,le=r*(Q/a);markerView.scale=Math.min(markerView.
maxScale,Math.max(markerView.minScale,le)),markerView.offsetX=l.x+(ye.x-d.x),markerView.offsetY=l.y+
(ye.y-d.y),applyMarkerTransform(),C.preventDefault();return}if(!X||!t)return;const R=$(C);if(markerState.
mode==="mosaic"){if(!L||!T)return;markerState.mosaicPreviewRect=oe(T,R)||xe(R),renderCropOverlay()}else
Y(R)&&(de(),P());C.preventDefault()},"move"),W=o(C=>{const R=X;if(i.delete(C.pointerId),i.size<2&&(s=
!1),i.size===0){if(X=!1,R&&t&&markerState.mode==="draw"&&p.length>0&&(de(),P()),R&&markerState.mode===
"mosaic"&&T){const G=$(C);let K=oe(T,G);(!K||K.w<2||K.h<2)&&(K=xe(T)),ke(K)&&(markerState.hasStroke=
!0,markerState.mosaicRects.push(K))}p=[],L=!1,T=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),
R&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(C.pointerId),C.preventDefault()},
"end");e.addEventListener("pointerdown",te),e.addEventListener("pointermove",se),e.addEventListener(
"pointerup",W),e.addEventListener("pointercancel",W)}o(initMarkerCanvas,"initMarkerCanvas");function initCropCanvas(){
const e=get("marker-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,s=null,
a=null,r=null,l=!1,d=0,p=markerView.scale,g={x:0,y:0},h={x:0,y:0};const v=8,b=14,w=o((P,q,Y)=>Math.min(
Y,Math.max(q,P)),"clamp"),x=o(P=>{const q=e.getBoundingClientRect(),Y=(P.clientX-q.left)*(e.width/q.
width),de=(P.clientY-q.top)*(e.height/q.height);return{x:Y,y:de}},"getPoint"),L=o((P,q)=>({x:(P.x+q.
x)/2,y:(P.y+q.y)/2}),"getMid"),T=o((P,q)=>Math.hypot(P.x-q.x,P.y-q.y),"getDist"),$=o(()=>(markerState.
cropRect||resetCropRectToFull(),markerState.cropRect),"ensureCropRect"),j=o((P,q)=>{if(!q)return"mov\
e";const Y=q.x,de=q.y,oe=q.x+q.w,xe=q.y+q.h,pe=Math.abs(P.x-Y)<=b,ke=Math.abs(P.x-oe)<=b,te=Math.abs(
P.y-de)<=b,se=Math.abs(P.y-xe)<=b;if(pe&&te)return"nw";if(ke&&te)return"ne";if(pe&&se)return"sw";if(ke&&
se)return"se";if(te)return"n";if(se)return"s";if(pe)return"w";if(ke)return"e";if(P.x>Y+b&&P.x<oe-b&&
P.y>de+b&&P.y<xe-b)return"move";const C=P.x<Y?"left":P.x>oe?"right":null,R=P.y<de?"top":P.y>xe?"bott\
om":null;if(C&&R){if(C==="left"&&R==="top")return"nw";if(C==="right"&&R==="top")return"ne";if(C==="l\
eft"&&R==="bottom")return"sw";if(C==="right"&&R==="bottom")return"se"}return C?C==="left"?"w":"e":R?
R==="top"?"n":"s":"move"},"hitTest"),J=o(P=>{if(markerState.mode!=="crop")return;if(n.set(P.pointerId,
{x:P.clientX,y:P.clientY}),n.size>=2){const de=Array.from(n.values()),oe=de[0],xe=de[1];l=!0,i=!1,d=
T(oe,xe)||1,p=markerView.scale,g={x:markerView.offsetX,y:markerView.offsetY},h=L(oe,xe),e.setPointerCapture&&
e.setPointerCapture(P.pointerId),P.preventDefault();return}if(l)return;i=!0;const q=x(P),Y=$();a=j(q,
Y),s=q,r=Y?{x:Y.x,y:Y.y,w:Y.w,h:Y.h}:null,renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
P.pointerId),P.preventDefault()},"start"),X=o(P=>{if(markerState.mode!=="crop")return;if(n.has(P.pointerId)&&
n.set(P.pointerId,{x:P.clientX,y:P.clientY}),l&&n.size>=2){const C=Array.from(n.values()),R=C[0],G=C[1],
K=L(R,G),Z=T(R,G)||1,ye=p*(Z/d);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,
ye)),markerView.offsetX=g.x+(K.x-h.x),markerView.offsetY=g.y+(K.y-h.y),applyMarkerTransform(),renderCropOverlay(),
P.preventDefault();return}if(!i||!s||!r)return;const q=x(P),Y=e.width,de=e.height,oe={x:r.x,y:r.y,w:r.
w,h:r.h},xe=r.x+r.w,pe=r.y+r.h,ke=o(()=>{const C=w(q.x,0,xe-v);oe.x=C,oe.w=xe-C},"applyW"),te=o(()=>{
oe.w=w(q.x-r.x,v,Y-r.x)},"applyE"),se=o(()=>{const C=w(q.y,0,pe-v);oe.y=C,oe.h=pe-C},"applyN"),W=o(()=>{
oe.h=w(q.y-r.y,v,de-r.y)},"applyS");switch(a){case"move":{const C=q.x-s.x,R=q.y-s.y;oe.x=w(r.x+C,0,Y-
r.w),oe.y=w(r.y+R,0,de-r.h);break}case"w":ke();break;case"e":te();break;case"n":se();break;case"s":W();
break;case"nw":se(),ke();break;case"ne":se(),te();break;case"sw":W(),ke();break;case"se":W(),te();break;default:
break}oe.x=w(oe.x,0,Y-oe.w),oe.y=w(oe.y,0,de-oe.h),markerState.cropRect=oe,renderCropOverlay(),P.preventDefault()},
"move"),Te=o(P=>{n.delete(P.pointerId),n.size<2&&(l=!1),n.size===0&&(renderCropOverlay(),i=!1,s=null,
a=null,r=null),e.releasePointerCapture&&e.releasePointerCapture(P.pointerId),P.preventDefault()},"en\
d");e.addEventListener("pointerdown",J),e.addEventListener("pointermove",X),e.addEventListener("poin\
terup",Te),e.addEventListener("pointercancel",Te),e.addEventListener("pointerleave",Te)}o(initCropCanvas,
"initCropCanvas");async function saveMarkerToRow(){const e=markerState.row,t=get("marker-image"),n=get(
"marker-canvas");if(!e||!t||!n)return;const i=get("marker-attach-original");i&&(e.dataset.attachOriginal=
i.checked?"1":"");let s=document.createElement("canvas");const a=markerState.naturalWidth||t.naturalWidth||
n.width,r=markerState.naturalHeight||t.naturalHeight||n.height;s.width=a,s.height=r;const l=s.getContext(
"2d");if(!l)return;if(l.drawImage(t,0,0,a,r),l.drawImage(n,0,0,a,r),markerState.cropRect){const L=a/
n.width,T=r/n.height,$=Math.max(0,Math.floor(markerState.cropRect.x*L)),j=Math.max(0,Math.floor(markerState.
cropRect.y*T)),J=Math.min(a,Math.max(1,Math.floor(markerState.cropRect.w*L))),X=Math.min(r,Math.max(
1,Math.floor(markerState.cropRect.h*T))),Te=document.createElement("canvas");Te.width=J,Te.height=X;
const P=Te.getContext("2d");P&&(P.drawImage(s,$,j,J,X,0,0,J,X),s=Te)}const d=await new Promise(L=>s.
toBlob(L,"image/png",.92));if(!d){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const g=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),h=new File([
d],`${g}_marked.png`,{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(
".upload-status"),bar:e.querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\
\u53CD\u6620\u4E2D..."),updateUploadRowFile(v,h);const b=e.getAttribute("data-filename"),w=getRowAttachmentSource(
e);b&&!e.dataset.originalFilename&&(e.dataset.originalFilename=b,e.dataset.originalSource=w,setAttachmentSourceForPath(
b,w)),await uploadFileWithProgress(h,v)?(b&&(currentImageUrls=currentImageUrls.filter(L=>L!==b)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}o(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,i.src=URL.createObjectURL(
e);let s=null,a=null,r=null,l=null,d=[],p=null;const g=o(()=>{p&&clearTimeout(p);try{URL.revokeObjectURL(
i.src)}catch{}try{i.remove()}catch{}if(s&&s.getTracks().forEach(v=>v.stop()),r)try{r.disconnect()}catch{}
if(l)try{l.disconnect()}catch{}if(a)try{a.close()}catch{}},"cleanup"),h=o(()=>{g(),n(null)},"fail");
i.onloadedmetadata=async()=>{try{s=i.captureStream();const v=s.getAudioTracks();if(!v||!v.length)return h();
a=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),l=a.createMediaStreamSource(
new MediaStream(v)),r=a.createScriptProcessor(4096,1,1),r.onaudioprocess=w=>{const x=w.inputBuffer.getChannelData(
0);d.push(new Float32Array(x))},l.connect(r),r.connect(a.destination);const b=isFinite(i.duration)?Math.
max(1,Math.ceil(i.duration*1e3)):0;b>0&&(p=setTimeout(()=>{const w=(e.name||"video").replace(/\.[^/.]+$/,
""),x=encodeWav(d,a.sampleRate),L=new File([x],`${w}.audio.wav`,{type:"audio/wav"});g(),n(L)},b+250)),
await i.play(),i.onended=()=>{const w=(e.name||"video").replace(/\.[^/.]+$/,""),x=encodeWav(d,a.sampleRate),
L=new File([x],`${w}.audio.wav`,{type:"audio/wav"});g(),n(L)}}catch{h()}},i.onerror=()=>h()}))}o(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,t={}){if(!e||!e.length)return;const n=Array.from(
e).filter(Boolean);if(!n.length)return;const i=collectImageUrlsForSend().length+browserFastLocalFiles.
size+Math.max(0,Number(uploadProgressState.active)||0);let s=n;if(i+n.length>ATTACHMENT_MAX_FILES){const h=Math.
max(0,ATTACHMENT_MAX_FILES-i);if(h<=0){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059`,"error",!0);return}s=n.slice(0,h),showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u5148\u982D${h}\u4EF6\u306E\u307F\u8FFD\u52A0\u3057\u307E\u3059\u3002`,"war\
ning",!0)}t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=
s.length,uploadProgressState.active+=s.length,updateFilePreview();const a=!!(get("upload-audio-only")&&
get("upload-audio-only").checked),r=getModelMediaSupport(get("model-select").value),l=o(async h=>{let v=null;
try{if(isAudioFile(h)&&!r.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(h)&&!r.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!h.type||!h.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const b=addUploadRow(h);updateFilePreview(),v=b.uploadId,uploadProgressState.perFilePct[v]=
0;let w=h;if(a&&isVideoFile(h)){const x=await extractAudioFromVideo(h,b);x?(w=x,updateUploadRowFile(
b,x),b&&b.status&&(b.status.textContent="\u97F3\u58F0\u306E\u307F")):(b&&b.status&&(b.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&h.type.startsWith("image/"))try{const x=getCompressionOutputType();
if(getCompressionFormatOnly())w=await convertImageFormatOnly(h,x);else{const T={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};x&&x!=="original"&&(T.fileType=x),await ensureImageCompression();
const $=await window.imageCompression(h,T),j=new File([$],imageFilenameForMime(h.name,$.type||(x!=="\
original"?x:h.type)),{type:$.type||h.type,lastModified:h.lastModified||Date.now()});j.size>h.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
h.size)} -> ${formatBytes(j.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),w=h):w=j}w!==h&&updateUploadRowFile(b,w)}catch{}if(browserFastModeEnabled){const x=Array.from(
browserFastLocalFiles.values()).reduce((L,T)=>L+Number(T.file&&T.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||x+w.size>BROWSER_FAST_MAX_BYTES?(b&&b.status&&(b.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),b&&b.row&&b.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(b.uploadId,{file:w,rowObj:b}),b.status&&(b.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),b.bar&&(b.bar.style.width="100\
%"),b.row&&(b.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(w,b)}finally{
v&&uploadProgressState.perFilePct.hasOwnProperty(v)&&(delete uploadProgressState.perFilePct[v],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let d=0;const p=Math.min(UPLOAD_CONCURRENCY,s.length),g=Array.from({length:p}).map(async()=>{
for(;;){const h=d++;if(h>=s.length)break;await l(s[h])}});await Promise.all(g)}o(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){const n=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).replace(/\n/g,
"<br>");const i=protectMathSegments(n),s=window.marked.parse(i.text),a=restoreMathSegments(s,i.blocks,
t);return window.DOMPurify.sanitize(a)}o(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");function getCanvasModeElements(){
const e=get("canvas-panel");return e?{panel:e,stage:get("conversation-stage"),title:get("canvas-pane\
l-title"),status:get("canvas-panel-status"),blockCount:get("canvas-block-count"),blockList:get("canv\
as-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get("canvas-preview-lang"),sourceSelect:get(
"canvas-source-select"),frame:get("canvas-preview-frame"),empty:get("canvas-preview-empty"),sourceScroll:get(
"canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get("canvas-panel-copy-btn"),clearBtn:get(
"canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:null}o(getCanvasModeElements,"getC\
anvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(t||"");return/<!doctype\s+html/i.
test(i)||/<html[\s>]/i.test(i)}o(isCanvasHtmlPreviewCandidate,"isCanvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){
const n=String(e&&e.lang?e.lang:"").trim(),i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),s=!!(e&&
e.open);return{...e,index:t,lang:n,code:i,open:s,key:hashString(`${n||"TEXT"}
${i||""}`)}}o(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],s=[],a=/^(\s*)(`{3,}|~{3,})(.*)$/;let r=null,l="",d=[];for(const h of n){
if(!r){const w=h.match(a);if(w){r=w[2],l=String(w[3]||"").trim(),d=[],i.push({lang:l,code:"",open:!0}),
s.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}s.push(
h);continue}const v=String(h||"").trim();if(v&&v.replace(/\s+/g,"")===r){const w=i[i.length-1];w&&(w.
code=d.join(`
`),w.open=!1),r=null,l="",d=[];continue}d.push(h);const b=i[i.length-1];b&&(b.code=d.join(`
`))}if(r&&i.length){const h=i[i.length-1];h&&(h.code=d.join(`
`),h.open=!0)}const p=i.map((h,v)=>normalizeCanvasBlock(h,v)),g=selectCanvasPreviewBlock(p,t);return{
renderText:s.join(`
`),blocks:p,primaryBlock:g?g.block:null,primaryIndex:g?g.index:-1,rawText:t}}o(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const a=i[n];return{block:a,index:n,previewType:isCanvasHtmlPreviewCandidate(
a.lang,a.code)?"html":"code"}}if(i.length>0){const a=i.length-1,r=i[a];return{block:r,index:a,previewType:isCanvasHtmlPreviewCandidate(
r.lang,r.code)?"html":"code"}}const s=String(t||"");return isCanvasHtmlPreviewCandidate("",s)?{block:normalizeCanvasBlock(
{lang:"html",code:s,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}o(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}o(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),s=!!t&&t===i;n.classList.toggle("can\
vas-active",s),n.setAttribute("aria-pressed",s?"true":"false"),n.setAttribute("data-canvas-active",s?
"1":"0"),n.innerHTML=s?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=s?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",s?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
o(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}o(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
s={preview:0,blocks:1,source:2},a=i[n];if(!a||!(t in s)||!(n in s))return;canvasPreviewState.viewAnimationToken+=
1;const r=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(d=>{d&&d.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),a.
offsetWidth;const l=s[n]<s[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";a.classList.
add(l),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{r===canvasPreviewState.viewAnimationToken&&
(a.classList.remove(l),canvasPreviewState.viewAnimationTimer=null)},340)}o(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var r,l;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",s=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const p=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",p),d.setAttribute("aria-pressed",p?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,s,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((l=(r=n.blockList).
focus)==null||l.call(r,{preventScroll:!0})))}o(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,s)=>{const a=String(i&&i.lang?i.lang:"text").trim()||"text",r=s===n,l=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",g=(String(i&&i.code?i.code:"").split(/\r?\n/).find(b=>b.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),h=`${r?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${a}`,v=`${h}\u3001${g}`;return`<butto\
n type="button" class="canvas-block-chip${r?" active":""}" data-canvas-block-index="${s}" title="${escapeHtml(
h)}" aria-label="${escapeHtml(v)}" aria-pressed="${r?"true":"false"}"><span class="canvas-block-chip\
-index">#${s+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
a)}</span><span class="canvas-block-chip-preview">${escapeHtml(g)}</span></span><span class="canvas-\
block-chip-state">${r?"\u8868\u793A\u4E2D":l}</span></button>`}).join("")}o(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((a,r)=>{const l=String(
a&&a.lang?a.lang:"text").trim()||"text";return`#${r+1} ${l}`}),s=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==s&&(e.sourceSelect.innerHTML=i.map((a,r)=>`<option value="${r}">${escapeHtml(
a)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=s),e.sourceSelect.value=String(
n)}o(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}o(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),s=String(e||""),a=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const r=new DOMParser().parseFromString(s,"text/html"),l=r.createElement("scr\
ipt");return l.setAttribute("data-canvas-scroll-bridge","true"),l.textContent=a,(r.body||r.documentElement).
appendChild(l),`<!DOCTYPE html>
`+r.documentElement.outerHTML}catch{return`${s}<script data-canvas-scroll-bridge>${a}<\/script>`}}o(
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
canvasPreviewState.selectedIndex:-1;if(!n.length){const r=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return r&&r.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=r.block.
key||"",r.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let s=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?s=i:canvasPreviewState.selectionMode="auto";const a=n[s]||null;return canvasPreviewState.
selectedIndex=a?s:-1,canvasPreviewState.selectedKey=a&&a.key?a.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),a}o(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,s=n&&Number.isInteger(
n.index)?n.index:-1,a=!!i,r=String(i&&i.lang?i.lang:"").trim(),l=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),d=a?isCanvasHtmlPreviewCandidate(r,l):!1,p=a?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",g=a?d?`HTML Canvas Preview${t.length>
1&&s>=0?` #${s+1}/${t.length}`:""}`:`Canvas Preview: ${r||"text"}${t.length>1&&s>=0?` #${s+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=g),e.status&&(e.status.textContent=
p),e.previewLang&&(e.previewLang.textContent=a?r||"text":"idle");const h=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,v=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=l),e.sourceScroll&&(e.sourceScroll.scrollTop=h,e.sourceScroll.
scrollLeft=v,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),a){canvasPreviewState.frameRenderToken+=1;const b=canvasPreviewState.frameRenderToken,
w=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),b);e.frame&&(e.frame.srcdoc=w,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{b!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:b,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}o(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const s=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
s&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}o(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(s=>s&&s.key===n);return i===-1?!1:applyCanvasSelection(i)}
o(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
s=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:s}}o(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const l=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[l],selectedIndex:0,selectedKey:l.key||t.codeKey||""}}const s=[];let a=-1;if(i.forEach(
(l,d)=>{const p=decodeCanvasPreviewButtonCode(l);if(!p)return;const g=normalizeCanvasBlock({lang:p.lang,
code:p.code,open:!1},d);s.push(g),a===-1&&p.codeKey===t.codeKey&&(a=s.length-1)}),!s.length)return null;
a===-1&&(a=0);const r=s[a]||s[0]||null;return{blocks:s,selectedIndex:a,selectedKey:r&&r.key?r.key:t.
codeKey||""}}o(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
if(!e)return!1;const t=collectCanvasBlocksFromButton(e);if(!t||!t.blocks||!t.blocks.length)return!1;
const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[],i=n.findIndex(a=>a&&a.
key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(i);const s=t.blocks[t.selectedIndex]||
t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.rawText=s&&s.code!==void 0&&
s.code!==null?String(s.code):"",canvasPreviewState.renderText=canvasPreviewState.rawText,canvasPreviewState.
selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.selectedKey=t.selectedKey||
s&&s.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),canvasPreviewState.lastCanvasData=
{renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:s,primaryIndex:canvasPreviewState.
selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.mobileView="preview",syncCanvasPanelViewUi(
"preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}o(previewCanvasCodeFromButton,"previewCanvasCod\
eFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&e.code!==null?
e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(n,t))return sanitizeHtmlForPreview(
t);const s=n?`Canvas Preview: ${n}`:"Canvas Preview",a=escapeHtml(t||"");return`<!doctype html><html\
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
            </style></head><body><div class="frame"><div class="label">${escapeHtml(s)}</div><pre>${a||
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
e);const s=normalizeMarkdownNewlines(n);if(!s&&s!=="")return i;const a=t?[String(t),""]:[""];for(const r of[
"`","~"])for(let l=3;l<=10;l++){const d=r.repeat(l);for(const p of a){const g=`${d}${p}
`,h=`
${d}`,v=g+s+h;i.includes(v)&&(i=i.split(v).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),s=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const a of s)for(const r of["`","~"])for(let l=3;l<=10;l++){const d=r.repeat(l);
[`${a}${d}
${i}
${d}`,`${a}
${d}
${i}
${d}`,`
${a}${d}
${i}
${d}`,`
${a}
${d}
${i}
${d}`].forEach(g=>{n.includes(g)&&(n=n.split(g).join(`
`))})}return n}o(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}o(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",s=String(t==null?"":t).replace(/\s+$/,"");return s?
s+`

`+i:i}o(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[];if(!t)return{text:"",executions:n};const i=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let s=t.replace(i,(a,r,l)=>{const d=String(l||"").trim();try{const p=JSON.parse(d);n.push({code:p&&p.
code!=null?String(p.code):"",output:p&&p.output!=null?String(p.output):""})}catch{n.push({code:d,output:""})}
return`
`});return n.forEach(a=>{a.code&&(s=stripExactFencedBlock(s,"python",a.code),s=stripExactFencedBlock(
s,"py",a.code)),s=stripVisiblePythonOutputBlock(s,a.output)}),s=s.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:s,executions:n}}o(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(a=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(a)?n.push(a.trim()):
i.push(a)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}o(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}o(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",s=e&&e.output!=null?String(e.output):"";let a="";try{window.
hljs&&typeof window.hljs.highlight=="function"?a=window.hljs.highlight(i,{language:"python"}).value:
a=escapeHtml(i)}catch{a=escapeHtml(i)}const r=escapeHtml(s),l=encodeURIComponent(i).replace(/'/g,"%2\
7"),d=encodeURIComponent(s).replace(/'/g,"%27"),p=hashString(`pyexec-detail
${i}
${s}
${t}`),g=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",h=`<button class="download-btn" data-\
code="${l}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,v=`<button class="coding-target-btn" data-code="${l}" data-code-key="${p}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${p}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(g)}</span><div class="code-actions">${v}${h}<button class="copy-btn" data-copy="code" \
data-code="${l}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${d}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${a}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${r}</code></pre></div></div></div>`}o(buildPythonExecDetailBoxHtml,"buildP\
ythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/python-\
execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"","/py\
thon-execution")}showModal("python-exec-modal")}o(showPythonExecDetailModal,"showPythonExecDetailMod\
al");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("pytho\
n-exec-modal-body"),s=get("python-exec-modal-title");if(!n||!i)return;const a=t&&Array.isArray(t.python_executions)?
t.python_executions:[];if(!a.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(s){const r=a.length>1?`\uFF08${a.length}\u4EF6\uFF09`:"";s.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${r}`}i.innerHTML=a.map((r,l)=>buildPythonExecDetailBoxHtml(r,l,a.length)).join(
""),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(!0,{persist:!1})),showPythonExecDetailModal(
e)}o(openPythonExecDetail,"openPythonExecDetail"),window.openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){
get("python-exec-modal")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&
history.back())}o(closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;
function buildAiMarkdownHtml(e){const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(
t.text,t.notes),i=canvasModeEnabled?parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,
rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(i),refreshCanvasPreviewPanel());const s=document.
createElement("div");return s.className="prose prose-invert text-sm break-words",s.innerHTML=sanitizeMarkdownHtml(
i.renderText),wrapRenderedSvgBoxes(s),lowBandwidthMode||(maybeNeedsHighlight(i.renderText,s)&&ensureHighlightLoaded().
catch(()=>{}),maybeNeedsMathJax(i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),s.outerHTML}o(buildAiMarkdownHtml,
"buildAiMarkdownHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),s=appendMcpExecutionNotes(i.text,i.notes),a=canvasModeEnabled?parseCanvasMarkdown(s):{renderText:s,
blocks:[],primaryBlock:null,rawText:s};if(canvasModeEnabled&&(updateCanvasPreviewState(a),refreshCanvasPreviewPanel()),
n.incrementalMath){const r=document.createElement("template");r.innerHTML=sanitizeMarkdownHtml(a.renderText,
{streamMathSegments:!0});const l=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(p=>{const g=p.getAttribute("data-stream-math-key");g&&l.set(g,p)});const d=[];r.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(p=>{const g=l.get(p.getAttribute(
"data-stream-math-key"));g?p.replaceWith(g):d.push(p)}),e.replaceChildren(r.content),wrapRenderedSvgBoxes(
e),queueHighlight(e,a.renderText),queueIncrementalMathTypeset(d);return}e.innerHTML=sanitizeMarkdownHtml(
a.renderText),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,a.renderText)}o(renderAiMarkdownInto,
"renderAiMarkdownInto");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}o(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");function renderMessage(e,t,n,i,s,a,r=null,l=!0,d=null,p=null,g=null,h=null,v=null,b=null,w=null,x=null,L=!0,T=null,$=null,j=null){
const J=t==="user",X=J?"bg-blue-600":"bg-gray-700",Te=J?"justify-end":"justify-start";messageStore[e]=
n;const P=!J&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},q=J?n:P.text;let Y=p;
if(Y==null){const Q=g!=null?Number(g):0,le=h!=null?Number(h):0;(g!=null||h!=null)&&(Y=Q+le)}messageMeta[e]=
{tokens_in:g,tokens_out:h,tokens_total:Y,tokens_content:b,tokens_thought:w,is_encrypted:v,role:t,model:a,
parent_id:T,quote_text:d,image_url:i,gem_name:$,batch_job:j,python_executions:J?[]:P.executions||[]};
let de="";d&&(de=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-g\
ray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(d)}\
</div>`);let oe="";if(s&&!J){let Q="";try{Q=JSON.parse(s).text||""}catch{Q=s}Q&&(oe=`<div class="tho\
ught-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain te\
xt-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(Q)}</d\
iv></div>`)}let xe="";if(i)try{const Q=JSON.parse(i);if(Q.length){const le=[];if(Q.forEach(U=>{let fe=U,
et="unknown";if(fe&&typeof fe=="object"&&(et=normalizeAttachmentSource(fe.source),fe=fe.filepath||fe.
path||fe.url||fe.file||""),fe=normalizeAttachmentPath(fe)||fe,!fe)return;setAttachmentSourceForPath(
fe,et);const Ce=fe.replace(/^\d+\//,""),tt=buildFileUrl(Ce),ut=buildAttachmentPreviewUrl(Ce),Ke=fe.split(
"/").pop(),ot=Ke.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(ot)?le.push(
`<img src="${ut}" data-viewer-src="${tt}" data-viewer-filename="${escapeHtml(Ke)}" class="chat-image\
" loading="lazy" onclick="openImageViewer('${tt}')" title="${Ke}">`):le.push(`<div class="file-thumb\
 bg-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer\
 hover:bg-gray-700" onclick="window.open('${tt}')" title="${Ke}"><i class="fas fa-file text-2xl text\
-gray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Ke}</span></div>`)}),le.length>
0){let U="grid-multi";le.length===1?U="grid-1":le.length===2?U="grid-2":le.length===3?U="grid-3":le.
length===4&&(U="grid-4"),xe=`<div class="image-grid ${U}">${le.join("")}</div>`}}}catch{}const pe=J?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,ke=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${J?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${pe}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
te=[];!J&&a&&te.push(escapeHtml(a)),$&&(J?te.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml($)}</span>`):te.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml($)}</span>`));const se=[];if(g!=null&&se.push(`In ${g}`),h!=null){let Q=`\
Out ${h}`;w!=null&&Number(w)>0&&(Q+=` (Thought ${w})`),se.push(Q)}if(se.length||p!=null){const Q=se.
length?se.join(" / "):`${p} tokens`;te.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${Q}</button>`)}if(v!=null){const Q=v?"fa-l\
ock":"fa-lock-open",le=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":v?"Encrypted":
"Plain",U=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";te.push(`<button class="${U}" title="${le}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${Q}"></i></button>`)}if(!J&&P.executions&&P.executions.length){
const Q=P.executions.length,le=Q>1?`Python \xD7${Q}`:"Python";te.push(`<button type="button" class="\
python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\u884C\u7D50\u679C\
\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${le}</span></button>`)}const W=te.length?`\
<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${te.join("\
 \u2022 ")}</div>`:"";let C;const R=!J&&j?(()=>{const Q=String(j.state||"").toUpperCase(),le=j.status_text||
(Q==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":Q==="JOB_ST\
ATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\u3067\u3059");
return`<div class="batch-status-card mb-3 rounded-lg border ${Q==="JOB_STATE_FAILED"||Q==="JOB_STATE\
_CANCELLED"||Q==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":Q==="JOB_STATE_S\
UCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet-400/40 bg-violet\
-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="fas fa-layer-group\
 mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(le)}</div></div>`})():"";J?C=`<div c\
lass="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||"")}</div>`:(C=
R+(q&&String(q).trim()?buildAiMarkdownHtml(q):j?'<div class="content-area prose prose-invert text-sm\
 break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(q)),C.includes("content-area")||(C=C.replace("prose ","content-area prose ")));let G="";
if(r){const Q=r.siblings[r.current-2],le=r.siblings[r.current];G=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${Q}\
)" ${Q?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${r.current} / ${r.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${le}\
)" ${le?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const K=l?"fade-in":"",Z=document.createElement("div");Z.className=`flex ${Te} mb-\
4 ${K} relative message-group group`,Z.id=`msg-${e}`,Z.innerHTML=`<div class="message-bubble ${X} te\
xt-white p-4 rounded-2xl shadow-md relative">${ke}${de}${oe}${C}${xe}${G}${W}</div>`;const ye=x||get(
"chat-container");return ye&&(ye.appendChild(Z),L&&scrollToBottom(),J||(queueMessageDecorations(Z,q),
syncCodingTargetButtons(Z),syncCodingModeUi(codingModeEnabled,{persist:!1}))),Z}o(renderMessage,"ren\
derMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const t={
modal:"token-details"};e!==null&&(t.messageId=e),history.pushState(t,"","/token-details")}showModal(
"token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];
if(!t||!get("token-detail-modal"))return;const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",s=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",a=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",r=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",l=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detai\
l-in").innerText=s,get("token-detail-out").innerText=a,get("token-detail-content").innerText=r,get("\
token-detail-thought").innerText=l,get("token-detail-encrypted").innerText=d;const p=t.model?`${t.model}\
 (${t.role})`:`${t.role}`;get("token-detail-title").innerText=p,showTokenDetailModal(e)}o(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){get("token-detail-modal")&&(hideModal("token-deta\
il-modal"),!e&&location.pathname==="/token-details"&&history.back())}o(closeTokenDetail,"closeTokenD\
etail");function openEncryptionSettings(e){const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}
o(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get("encrypti\
on-status-modal"))return;const n=get("encryption-status-title"),i=get("encryption-status-body"),s=get(
"encryption-status-admin-actions"),a=get("encryption-status-admin-toggle"),r=!!e;r?(n&&(n.innerText=
"\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\
\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
s&&a&&(!!(isAdminUser&&currentThreadId)?(s.classList.remove("hidden"),a.dataset.enable=r?"0":"1",a.disabled=
!1,a.textContent=r?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
a.className=r?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):s.classList.add("hidden")),showEncryptionStatusModal()}o(openEncryptionModal,"openEncryptio\
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
const s=(e||"").toLowerCase();return!s.includes("gemini")||s.includes("image")||s.includes("nano")||
s.includes("tts")||s.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=o(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),s=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(a=>{let r=!1;function l(){n&&n.removeEventListener("cli\
ck",p),i&&i.removeEventListener("click",g),s&&s.removeEventListener("click",g),e.removeEventListener(
"click",h,!0)}o(l,"cleanup");function d(v){if(r)return;r=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),l(),hideModal("gemini-local-python-modal"),a(v)}o(d,"finalize");
function p(){d(!0)}o(p,"onOk");function g(){d(!1)}o(g,"onCancel");function h(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),g())}o(h,"onOverlay"),n&&n.addEventListener("click",p),i&&i.addEventListener(
"click",g),s&&s.addEventListener("click",g),e.addEventListener("click",h,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,s=null){const a=t?"fade-in":"",r=i?
` id="${i}"`:"",l=buildPendingSkeletonHtml(s,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ${a}"><div${r} class="message-bubble ai-pending-bubble bg-gray-700 text-w\
hite p-4 rounded-2xl rounded-tl-none shadow-md relative">${l}</div></div>`,p=e||get("chat-container");
if(p){if(typeof p.insertAdjacentHTML=="function")p.insertAdjacentHTML("beforeend",d);else{const g=document.
createElement("div");g.innerHTML=d;const h=g.firstElementChild;h&&p.appendChild(h)}n&&scrollToBottom()}}
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
const t=Array.from(e.querySelectorAll(".prose")).some(l=>String(l.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(l=>!!String(l.textContent||
"").trim()&&l.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const s=e.parentElement;
if(!s)return null;const a=s.cloneNode(!0);a.setAttribute("data-local-stopped-partial","1"),a.classList.
remove("fade-in");const r=a.querySelector(".message-bubble");if(r&&(r.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),r.removeAttribute("data-stream-transition"),r.removeAttribute("id"),!a.
querySelector('[data-stopped-partial-note="1"]'))){const l=document.createElement("div");l.setAttribute(
"data-stopped-partial-note","1"),l.className="text-[10px] text-amber-200/90 mt-2 text-right",l.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",r.appendChild(l)}return{html:a.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}o(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const s=get("chat-\
container");return s?(s.querySelectorAll('[data-local-stopped-partial="1"]').forEach(a=>a.remove()),
s.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}o(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}o(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}o(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}o(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var l,d;const n=Math.max(0,Number((l=t.retries)!=null?l:1)||0),i=Math.max(0,Number((d=t.retryDelayMs)!=
null?d:180)||0),s=!!t.notifyOnFailure,a=e!=null&&e!==""?String(e):null,r=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!r||a&&r!==a)return!1;for(let p=0;p<=n;p++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==r?!1:(await loadMessages(r,{preserveDraft:!0,silent:!0}),
!0)}catch{p<n&&i>0&&await new Promise(h=>setTimeout(h,i))}return s&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===r&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
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
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(a=>a.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(a=>a.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const s=String(n.presetArgument||t||"").trim();if(i.selectId){if(!s)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} ${n.id==="effort"?"none / low / medium / high / xhigh / max":"default\
 / none"}`,"info"),!1;const a=get(i.selectId),r=s.toLowerCase(),l=a?Array.from(a.options).find(d=>d.
value.toLowerCase()===r||d.textContent.trim().toLowerCase()===r):null;return!a||!l?(showToast(`${n.label}\
: \u6307\u5B9A\u5024\u300C${s}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(a.value=l.value,a.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${i.label}: ${l.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&s){const a=s.toLowerCase(),
r={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},l=parseSlashToggleArgument(
s),d=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(r,a)){d&&!d.checked&&!d.disabled&&(d.
checked=!0,d.dispatchEvent(new Event("change",{bubbles:!0})));const p=get("thinking-level");return p&&
(p.value=r[a],p.dispatchEvent(new Event("change",{bubbles:!0}))),refreshMinimalOptionItems(),showToast(
`Thinking: ${a}`,"success"),!0}if(l===void 0)return showToast("\u4F7F\u3044\u65B9: /thinking on / off / min / low /\
 mid / high","info"),!1}if(i.checkboxId&&s){const a=parseSlashToggleArgument(s);if(a===void 0)return showToast(
`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const r=get(i.checkboxId);if(a!==null&&r&&r.checked===
a)return showToast(`${i.label}: ${a?"ON":"OFF"}`,"info"),!0}return handleMinimalOptionClick(i),!0}o(
executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){const t=String(
e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||"",i=n.
match(/^[a-z][\w-]*/i);return i?i[0]:n}o(extractSlashCommandToken,"extractSlashCommandToken");function hideSlashCommandSuggestions(){
const e=get("slash-command-suggestions");e&&e.classList.add("hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=
0}o(hideSlashCommandSuggestions,"hideSlashCommandSuggestions");function showPendingSlashCommandIndicator(e){
const t=get("slash-command-indicator"),n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.
find(a=>a.id===e);n.textContent=i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");
const s=get("prompt-input");s&&i&&(s.dataset.originalPlaceholder=s.placeholder,s.placeholder=i.argumentHint||
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}
o(showPendingSlashCommandIndicator,"showPendingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){
const e=get("slash-command-indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get(
"prompt-input");t&&t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.
dataset.originalPlaceholder);const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}
o(hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const s=visibleSlashCommands(e);if(s.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,s.length-1),n.innerHTML="",s.forEach((v,b)=>{const w=document.createElement(
"div");w.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${b===
slashSelectedIndex?"bg-gray-700":""}`,w.innerHTML=`
                    <i class="fas ${v.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${v.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${v.description}</div>
                    </div>
                `;let x=!1;w.addEventListener("pointerdown",L=>{typeof L.button=="number"&&L.button!==
0||(L.preventDefault(),x=!0,selectSlashCommand(v.id))}),w.addEventListener("click",L=>{L.preventDefault(),
x||selectSlashCommand(v.id)}),w.onmouseenter=()=>{slashSelectedIndex=b,showSlashCommandSuggestions(e)},
n.appendChild(w)});const a=i.getBoundingClientRect(),r=window.innerHeight,l=r-a.bottom,d=a.top,p=260,
g=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,a.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",l<180&&d>l){const v=Math.min(p,d-g);t.style.top="auto",t.style.bottom=`${r-a.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(p,l-g);t.style.top=`${a.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${v}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}o(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const r=String(n||"").trimStart();t.value=
r.substring(1+i.length).trimStart()}else{const r=n.lastIndexOf("/");r!==-1?t.value=n.substring(0,r).
trimEnd():t.value=""}hideSlashCommandSuggestions();const s=SLASH_COMMANDS.find(r=>r.id===e),a=t.value.
trim();if(s&&s.autocompleteArgument&&!a){t.value=`${s.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(s&&s.kind==="minimal"&&(!s.
requiresArgument||a)){t.value="",executeMinimalSlashCommand(e,a),t.dispatchEvent(new Event("input",{
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
entries(e||{}),s=`settings-result-${Date.now()}`,a=n==="inspect",r=i.length?a?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
a?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
l=renderMessage(s,"assistant",r,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!l)
return;removeEphemeralMessageControls(l);const d=l.querySelector(".message-bubble");if(!d||!i.length)
return;const p=document.createElement("div");p.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([h,v])=>{const b=AI_SETTING_JUMP_TARGETS[h]||{label:h},w=document.createElement("button");w.
type="button",w.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const x=document.createElement("span");x.className="min-w-0 flex-1";const L=document.createElement(
"span");L.className="block text-xs font-bold text-blue-200",L.textContent=b.label;const T=document.createElement(
"span");T.className="block mt-0.5 text-[11px] text-gray-300 break-words",T.textContent=formatAiSettingValue(
v);const $=document.createElement("i");$.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",x.appendChild(L),x.appendChild(T),w.appendChild(x),w.appendChild($),w.addEventListener(
"click",()=>openAiSettingJumpTarget(h)),p.appendChild(w)});const g=d.querySelector(".message-footer-\
meta");g?d.insertBefore(p,g):d.appendChild(p),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const s=get(
"welcome-screen");s&&s.classList.add("hidden");const a=`settings-pending-${n}`,r=get("chat-container");
r&&(r.insertAdjacentHTML("beforeend",`<div id="${a}" class="flex justify-start mb-4 fade-in"><div cl\
ass="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-\
md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const d=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),p=get(a);if(p&&p.remove(),d&&d.status==="ok"&&d.mode==="inspect"&&d.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(d.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
d.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(d.current,t,"inspect");
return}if(d&&d.status==="ok"&&d.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
d.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(d.applied).length}\u9805\u76EE\uFF09`,"success");try{const v=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(b=>b.json());populateAiSafeFormFields(v),cacheUserSettings(v)}catch{}renderAiSettingsResultBubble(
d.applied,t);return}const g=d.message||d.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${g}`);
const h=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${g}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
h),showToast(g,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const d=get(a);d&&d.remove();const p=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(p),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
o(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}o(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const s=e.toLowerCase(),a=loadedGems.filter(b=>b.name.toLowerCase().includes(s)||b.description&&
b.description.toLowerCase().includes(s));if(a.length===0){hideGemSuggestions();return}gemSelectedIndex>=
a.length&&(gemSelectedIndex=0),n.innerHTML="",a.forEach((b,w)=>{const x=document.createElement("div");
x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===gemSelectedIndex?
"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(b.name)}</div>
                        ${b.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
b.description)}</div>`:""}
                    </div>
                `,x.onclick=()=>selectGemSuggestion(b),x.onmouseenter=()=>{gemSelectedIndex=w,showGemSuggestions(
e)},n.appendChild(x)});const r=i.getBoundingClientRect(),l=window.innerHeight,d=l-r.bottom,p=r.top,g=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&p>d){const b=Math.min(g,p-h);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${b}px`}else{const b=Math.min(g,d-h);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${b}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}o(showGemSuggestions,
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
able-mcp"].some(r=>{const l=get(r);return!!(l&&l.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const i=get("thread-custom-instruction");if(i&&String(i.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const s=Array.from(browserFastLocalFiles.values());return s.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":s.reduce((r,l)=>r+Number(l.file&&l.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":s.some(r=>!r.
file||!String(r.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}o(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const s=String(i.result||""),a=s.indexOf(",");if(a<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
s.slice(a+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}o(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const s=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(a=>{a&&
s.push({thoughtSignature:String(a)})}),i.text&&s.push({text:String(i.text)});for(const a of Array.isArray(
i.images)?i.images:[])try{const r=await fetch(buildFileUrl(a.path),{credentials:"same-origin",cache:"\
no-store"});if(!r.ok)throw new Error(`HTTP ${r.status}`);const l=await r.blob();s.push({inlineData:{
mimeType:a.mime_type||l.type||"application/octet-stream",data:await fileToBase64Payload(l)}})}catch{
n++}s.length&&t.push({role:i.role,parts:s})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}o(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(){
const e=Array.from(browserFastLocalFiles.entries());for(const[t,n]of e){if(!n||!n.file||!n.rowObj)throw new Error(
"\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");
if(n.rowObj.status&&(n.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(n.file,n.rowObj))throw new Error(`${n.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);browserFastLocalFiles.delete(t)}}o(uploadBrowserFastLocalFiles,
"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){const t=get("enable-thinking");
if(!t||!t.checked)return null;const n=String(get("thinking-level")?get("thinking-level").value:"high").
toLowerCase();if(e.includes("2.5")){const s=Number(get("thinking-budget")?get("thinking-budget").value:
4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(s)?Math.max(0,Math.min(32768,Math.trunc(
s))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["MEDIUM","HIGH"].includes(i)&&(i="MEDIU\
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
function updateBrowserFastPythonBox(e,t,n){if(e){if(t==="code"){const i=n==null?"":String(n),s=e.querySelector(
".python-code");s&&(s.textContent=i,s.removeAttribute("data-highlighted"),queueHighlight(e,i));const a=e.
querySelector('.copy-btn[data-copy="code"]');a&&a.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}else if(t==="output"){const i=n==null?"":String(n),s=e.querySelector(".python-output");
s&&(s.textContent=i);const a=e.querySelector('.copy-btn[data-copy="output"]');a&&a.setAttribute("dat\
a-code",encodeURIComponent(i).replace(/'/g,"%27"))}}}o(updateBrowserFastPythonBox,"updateBrowserFast\
PythonBox");async function sendBrowserFastMessage(e){const t=String(get("model-select").value||"").trim(),
n=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==t)throw new Error(
"\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const i=Array.from(browserFastLocalFiles.values()),s=[];for(const P of i)s.push({inlineData:{mimeType:P.
file.type,data:await fileToBase64Payload(P.file)}});s.push({text:e});const a={},r=browserFastThinkingConfig(
t.toLowerCase());r&&(a.thinkingConfig=r);const l={contents:[...await buildBrowserFastHistoryContents(
n.history),{role:"user",parts:s}],generationConfig:a};!!(get("enable-python")&&get("enable-python").
checked)&&(l.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden"),renderMessage(Date.now(),"user",
e,null,null,null,null,!0,null,null,null,null,null,null,null,null,!0);const p=`browser-fast-${Date.now()}`;
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${p}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const g=get(p);activeStreamingBubbleId=p,setSendBtnToStopMode(),resumeChatAutoScroll(),
abortController=new AbortController;let h="",v="";const b=[];let w=null,x=null,L=!1;const T={},$=[];
let j=null,J="";const X=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Te=!1;try{const P=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
t)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(l),signal:abortController.
signal}));if(!P.ok){const se=await P.json().catch(()=>({}));throw new Error(se&&se.error&&se.error.message?
se.error.message:`Gemini API HTTP ${P.status}`)}window.ConnectionMonitor&&(Te=!0,window.ConnectionMonitor.
operationStarted()),X&&X.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const q=P.body.getReader(),Y=new TextDecoder;let de="";const oe=o(se=>{const W=se.split(
/\r?\n/).filter(G=>G.startsWith("data:")).map(G=>G.slice(5).trim()).join("");if(!W||W==="[DONE]")return;
const C=JSON.parse(W);if(C.error)throw new Error(C.error.message||"Gemini API error");if((Array.isArray(
C.candidates)?C.candidates:[]).forEach(G=>{(G&&G.content&&Array.isArray(G.content.parts)?G.content.parts:
[]).forEach(Z=>{if(Z&&typeof Z.thoughtSignature=="string"&&!b.includes(Z.thoughtSignature)&&b.push(Z.
thoughtSignature),Z&&Z.executableCode&&typeof Z.executableCode.code=="string"){const Q=Z.executableCode.
code;h+=`
\`\`\`python
${Q}
\`\`\`
`,j=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,J=Q,T[j]||(g.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(j)),T[j]=g.querySelector(`[data-py-id="${j}"]`)),updateBrowserFastPythonBox(
T[j],"code",Q);return}if(Z&&Z.codeExecutionResult&&typeof Z.codeExecutionResult.output=="string"){const Q=Z.
codeExecutionResult.output;h+=`
**Output:**
\`\`\`
${Q}
\`\`\`
`;const le=j||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;$.push({code:J||
"",output:Q}),T[le]||(g.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(le)),T[le]=g.querySelector(
`[data-py-id="${le}"]`)),updateBrowserFastPythonBox(T[le],"output",Q);return}const ye=typeof Z.text==
"string"?Z.text:"";ye&&(Z.thought===!0?v+=ye:h+=ye)})}),!L&&(h||v)){beginPendingToStreamTransition(g);
const G=g.querySelector(".content-area");G&&G.remove(),L=!0}v&&(x||(g.insertAdjacentHTML("afterbegin",
'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>'),
x=g.querySelector(".thought-content")),x.textContent=v),h&&(w||(w=document.createElement("div"),w.className=
"content-area prose prose-invert text-sm break-words",g.appendChild(w)),renderAiMarkdownInto(w,h,{incrementalMath:!0})),
scrollToBottom()},"consumeEvent");for(;;){const{done:se,value:W}=await q.read();if(se)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),X&&X.setPhase("receiving"),de+=Y.decode(W,{stream:!0});const C=de.
split(/\r?\n\r?\n/);de=C.pop()||"",C.forEach(oe)}if(de+=Y.decode(),de.trim()&&oe(de),!h.trim())throw new Error(
"Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
w&&renderAiMarkdownInto(w,h,{incrementalMath:!0}),x&&x.classList.add("collapsed"),$.length&&(h+=$.map(
se=>`
\`\`\`pyexec
${JSON.stringify(se)}
\`\`\`
`).join("")),i.length&&(X&&X.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),X&&X.setPhase("saving");const xe=collectImageUrlsForSend(),
pe=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:h,thought_content:v,model:t,image_urls:xe,temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:n.parent_id||null,thought_signatures:b,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}),g),ke=await pe.json().catch(()=>({}));if(!pe.ok||!ke.thread_id)throw new Error(
ke.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");const te=!currentThreadId;currentThreadId=
String(ke.thread_id),currentParentId=ke.assistant_message_id||null,currentLeafId=ke.assistant_message_id||
null,resetUploadState(),browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,skipHistory:!te}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\
\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F","success",!1)}catch(P){if(P.name!=="AbortError"){showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9: ${P.message}`,"error",!0),get("prompt-input").value||(get("prompt-i\
nput").value=e);const q=P.message||"\u30A8\u30E9\u30FC";g&&g.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
q));try{let Y=h||"";$.length&&(Y+=$.map(ke=>`
\`\`\`pyexec
${JSON.stringify(ke)}
\`\`\`
`).join(""));const de=buildChatErrorMarkdown(q,Y),oe=i.length?[]:collectImageUrlsForSend(),xe=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:de,
thought_content:v||"",model:t,image_urls:oe,temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||
null,parent_id:n&&n.parent_id?n.parent_id:null,thought_signatures:b,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),g),pe=await xe.
json().catch(()=>({}));if(xe.ok&&pe.thread_id){const ke=!currentThreadId;currentThreadId=String(pe.thread_id),
currentParentId=pe.assistant_message_id||null,currentLeafId=pe.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!ke}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(Y){sendClientDebugLog("error",`Browser fas\
t error persist failed: ${Y&&Y.message?Y.message:Y}`)}}}finally{Te&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),X&&X(),setSendBtnToSendMode(),activeStreamingBubbleId===p&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Wt;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const N=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(N);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const t=get("prompt-inp\
ut").value;if(pendingSlashCommand){const N=pendingSlashCommand,ae=t.trim(),_e=get("model-select")?get(
"model-select").value:null;if(N==="settings"){if(!ae){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!_e){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
ae,_e)}else executeMinimalSlashCommand(N,ae)?(get("prompt-input").value="",get("prompt-input").style.
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
-1,tempPrompt="";const s=collectAttachmentItemsForSend(),a=s.map(N=>N.path),r=s.filter(N=>normalizeAttachmentSource(
N.source)==="upload").map(N=>N.path);if(a.length>ATTACHMENT_MAX_FILES){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u6DFB\u4ED8\u3092\u6E1B\u3089\u3057\u3066\u518D\u9001\u3057\u3066\u304F\u3060\u3055\u3044\u3002`,
"error",!0);return}const l=getModelMediaSupport(get("model-select").value),d=a.some(N=>isAudioPath(N)),
p=a.some(N=>isVideoPath(N)),g=(get("model-select").value||"").toLowerCase(),h=get("enable-python"),v=!!(h&&
h.checked);if(d&&!l.audio||p&&!l.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!t.trim()&&a.length===0)return;if(isMistralOcrModel(
g)){const N=/https?:\/\/\S+/i.test(t);if(a.filter(_e=>isAudioPath(_e)||isVideoPath(_e)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!a.length&&!N){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const b=t.trim();if(/^\/settings(?:\s|$)/i.
test(b)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(b)){const N=b.replace(/^\/settings\s*/i,"").trim();
if(!N){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const _e=get("prompt-input");_e.value="/settings ";const Me=extractSlashCommandToken(_e.value);
lastSlashFilter=Me,showSlashCommandSuggestions(Me),_e.focus();return}const ae=get("model-select")?get(
"model-select").value:null;if(!ae){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
N,ae);return}if(isGeminiLocalPythonMode(g,d,p,v)&&!await confirmGeminiLocalPythonSwitch())return;let w=null,
x=[];if(codingModeEnabled){const N=collectCodingCandidates(t),ae=N.filter(Fe=>Fe.prompt_source),_e=N.
filter(Fe=>!Fe.prompt_source),Me=ae.reduce((Fe,Pe)=>Fe+String(Pe.code||"").length,0);if(Me>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let je=3e5-Me;const He=[];for(let Fe=_e.length-1;Fe>=0;Fe--){const Pe=String(_e[Fe].
code||"").length;Pe>je||(He.unshift(_e[Fe]),je-=Pe)}x=codingTargetSelection?He.slice(-1):[...ae,...He];
const nt=ae.length?ae[ae.length-1]:null;if(w=codingTargetSelection?x[0]:nt||x[x.length-1]||null,codingModeEffective=
!!(w&&String(w.code||"").trim()),codingModeEffective&&w.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const Fe=String(((Wt=get(
"model-select"))==null?void 0:Wt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(Fe)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const L=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${t.length}\
 attachments=${a.length} search=${get("enable-search").checked}`);const T=t,$=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&a.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=editingMessageId,J=currentParentId,X=j!=null;j&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Te=[],P=o(N=>{if(N==null)
return;let ae=document.getElementById(`msg-${N}`);for(;ae;)ae.classList&&ae.classList.contains("mess\
age-group")&&(Te.push({node:ae,prevDisplay:ae.style.display}),ae.style.display="none"),ae=ae.nextElementSibling},
"hideRenderedBranchFrom"),q=o(()=>{Te.forEach(({node:N,prevDisplay:ae})=>{N&&(N.style.display=ae||"")}),
Te.length=0},"restoreHiddenBranch");j&&P(j);const Y=Date.now(),de=renderMessage(Y,"user",T,JSON.stringify(
a),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,J,activeGem?activeGem.name:null);
let oe=!1;const xe=/(https?:\/\/)?(x\.com|twitter\.com)\//i,pe=xe.test(T||"")||xe.test(currentQuote||
""),ke="grok-4-fast-reasoning",te=o(()=>{get("enable-search").checked=!0,get("model-select").value!==
ke&&selectModelById(ke)},"applyXLinkAuto");if(pe&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)te();else{const N=get("auto-search-banner"),ae=get("auto-search-on-btn"),_e=get(
"auto-search-off-btn"),Me=get("auto-search-remember");N&&ae&&_e&&(Me&&(Me.checked=!1),await new Promise(
je=>{N.classList.remove("hidden");const He=o(nt=>{N.classList.add("hidden"),ae.onclick=null,_e.onclick=
null,je(nt)},"cleanup");ae.onclick=()=>He("enable"),_e.onclick=()=>He("disable")}).then(async je=>{je===
"enable"?(te(),Me&&Me.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
oe=!0}))}const se=String(get("reasoning-effort").value||"").toLowerCase(),W=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&se==="none",C={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:T,model:get("model-select").value,image_urls:a,image_items:s,uploaded_image_urls:r,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:W?!1:get("enable-thin\
king").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:$,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
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
coding_mode:L,coding_target:L?{id:w.candidate_id,code:w.prompt_source?null:w.code,language:w.language||
"text",key:w.key||null,message_id:w.message_id||null,source:w.prompt_source?"prompt":"history",explicit:w.
explicit===!0}:null,coding_candidates:L?x.map(N=>({id:N.candidate_id,source:N.prompt_source?"prompt":
"history",prompt_index:N.prompt_source?N.prompt_index:null,code:N.prompt_source?null:N.code,language:N.
language||"text",explicit:N.explicit===!0})):[],batch_mode:i};e&&(C.turnstile_token=e);const R=get("\
thread-custom-instruction");R&&(C.thread_custom_instruction=R.value||""),activeGem?(C.system_prompt=
activeGem.instruction,C.enable_system_prompt=!0,C.gem_uuid=activeGem.uuid):C.gem_uuid=null,setSendBtnToStopMode();
const G="ai-"+Date.now(),K=String(C.model||"").toLowerCase(),Z=!!C.enable_thinking||!!se&&se!=="none",
ye=K.includes("gemini")||K.includes("o1")||K.includes("o3")||K.includes("gpt-5")||K.includes("reason\
ing")&&!K.includes("non-reasoning"),Q=Z&&ye;let le=buildPendingSkeletonHtml(C.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 fade-in"><\
div id="${G}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded\
-tl-none shadow-md relative">${le}</div></div>`),resumeChatAutoScroll();const U=get(G);activeStreamingBubbleId=
G,canvasModeEnabled&&resetCanvasPreviewPanel();let fe=null;const et=o(N=>!Q||!U?null:((!fe||!U.contains(
fe))&&(fe=U.querySelector(".thought-content")),fe||(U.insertAdjacentHTML("afterbegin",'<div class="t\
hought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i cla\
ss="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" \
data-placeholder="1"></div></div>'),fe=U.querySelector(".thought-content")),fe&&(fe.setAttribute("da\
ta-placeholder","1"),fe.textContent=N||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
fe),"ensureThoughtPlaceholder");Q&&et("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const Ce=currentThreadId,tt=nowPerfMs(),ut=Date.now();let Ke=!1,
ot=!1,xt=!1,gt=null,kt=null,rt=null,Pt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Ot=o((N,ae)=>{if(!ae||N==="status"&&Ke||N==="thought"&&ot||N==="content"&&xt)return;const _e=Math.
max(0,nowPerfMs()-tt);N==="status"?gt=_e:N==="thought"?kt=_e:N==="content"&&(rt=_e),reportFirstTokenLatency(
{latency_seconds:_e/1e3,latency_ms:_e,thread_id:Pt||currentThreadId,job_id:currentJobId,model:C.model,
first_event_type:N,client_sent_at_ms:ut}),N==="status"?Ke=!0:N==="thought"?ot=!0:N==="content"&&(xt=
!0)},"maybeReportFirstEventLatency"),at=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let pt=!1,_t=!1,St=null,mt=null,zt=!1;try{C.thread_id&&activeGem&&(threadGemMap[C.thread_id]=
activeGem,pendingGemForNewThread=null);const N=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(C),signal:abortController.signal}),U);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${N.status}`),!N.ok){const Le=await N.json().catch(()=>({})),Re=new Error(Le.error||
`HTTP ${N.status}`);throw Re.serverCode=Le.code||null,Re.serverModel=Le.model||C.model,Re.acceptedJobId=
Le.job_id||null,Re.acceptedThreadId=Le.thread_id||null,Re}pt=!0,window.ConnectionMonitor&&(zt=!0,window.
ConnectionMonitor.operationStarted()),at&&at.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const ae=o(()=>{if(!U)return;const Le=U.querySelector(
".content-area");if(Le&&Le.getAttribute("data-api-accepted")!=="1"&&(Le.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(U,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
Le.outerHTML=buildPendingSkeletonHtml(C.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const Re=U.querySelector(".content-area");Re&&Re.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
U,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");ae();
const _e=N.body.getReader(),Me=new TextDecoder;let je="",He="",nt="",Fe=!0,Pe=null,Ne=null,qe=null,Vt=!1;
const Nt={};let Ze=0,Rt=!1;for(;!Rt;){const{done:Le,value:Re}=await _e.read();if(Le)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),at&&at.setPhase("receiving"),je+=Me.decode(Re,{stream:!0});
let Xe=je.split(`
`);je=Xe.pop();let Jt=!1,an=!1;for(let lt of Xe)if(lt.trim())try{const ce=JSON.parse(lt);if(ce.type===
"thread_id"){ae();const be=ce.content!==null&&ce.content!==void 0?String(ce.content):ce.content;be&&
(Pt=be,currentThreadId!==be&&(currentThreadId=be,history.pushState({},"","/c/"+be)),activeGem&&(threadGemMap[be]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(ce.type==="job_\
id"){ae(),currentJobId=ce.content;continue}if(ce.type==="search_status"){ce.content==="searching"&&!qe?
(U.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),qe=U.querySelector(".search-box")):ce.content==="done"&&qe&&
(qe.classList.remove("animate-pulse"),qe.innerHTML='<i class="fas fa-check-circle text-green-400"></\
i> Search complete',setTimeout(()=>{qe&&qe.remove(),qe=null},2e3));continue}if(ce.type==="mcp"){handleMcpStreamEvent(
U,ce.content||{});continue}if(ce.type==="mcp_decision_request"){openMcpDecisionModal(ce.content||{});
continue}if(ce.type==="status"){ae();const be=ce.content===null||ce.content===void 0?"":String(ce.content);
if(Ot("status",!!be),Fe&&U){const Ge=be||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(
U,Ge,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Ue=U.querySelector(".content-area");Ue&&(Ue.outerHTML=buildPendingSkeletonHtml(C.model,Ge),updatePendingSkeletonStatus(
U,Ge,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
Q&&et(be||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Fe){beginPendingToStreamTransition(
U);const be=U.querySelector(".content-area");be&&(be.innerHTML=""),Fe=!1}if(ce.type==="coding_diff")
appendCodingLiveDiff(U,ce.content||{}),Ot("content",!0);else if(ce.type==="thought"){if(Pe||(Pe=U.querySelector(
".thought-content")),nt+=ce.content,Ot("thought",!!ce.content),!Pe){const be='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';qe?qe.insertAdjacentHTML(
"afterend",be):U.insertAdjacentHTML("afterbegin",be),Pe=U.querySelector(".thought-content")}if(Pe&&Pe.
getAttribute("data-placeholder")==="1"){if(Pe.textContent="",Pe.removeAttribute("data-placeholder"),
Pe){const be=Pe.parentElement.querySelector(".thought-header");be&&be.classList.remove("thinking-shi\
mmer")}nt=ce.content}Pe.classList.remove("collapsed"),an=!0}else if(ce.type==="image_analysis"){const be=ce.
content===null||ce.content===void 0?"":String(ce.content);if(!U)continue;let Ge=U.querySelector(".im\
age-analysis-box");if(!Ge){const Ye='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';qe?qe.insertAdjacentHTML("afterend",Ye):U.insertAdjacentHTML("afterbegin",Ye),Ge=U.querySelector(
".image-analysis-box")}const Ue=Ge.querySelector(".image-analysis-text");Ue&&(Ue.textContent=be)}else if(ce.
type==="python"){const be=ce.content||{},Ge=be.id||`py_${Date.now()}`;if(!Nt[Ge]){const Ye=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Ge}" data-collapsed="true" data-code-key="${Ge}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;qe?qe.insertAdjacentHTML("afterend",Ye):U.insertAdjacentHTML(
"afterbegin",Ye),Nt[Ge]=U.querySelector(`[data-py-id="${Ge}"]`)}const Ue=Nt[Ge];if(Ue){if(be.code!==
void 0){const Ye=be.code==null?"":String(be.code),ct=Ue.querySelector(".python-code");ct&&(ct.textContent=
Ye,ct.removeAttribute("data-highlighted"),queueHighlight(Ue,Ye));const Tt=Ue.querySelector('.copy-bt\
n[data-copy="code"]');Tt&&Tt.setAttribute("data-code",encodeURIComponent(Ye).replace(/'/g,"%27"))}if(be.
output!==void 0){const Ye=be.output==null?"":String(be.output),ct=Ue.querySelector(".python-output");
ct&&(ct.textContent=Ye);const Tt=Ue.querySelector('.copy-btn[data-copy="output"]');Tt&&Tt.setAttribute(
"data-code",encodeURIComponent(Ye).replace(/'/g,"%27"))}}}else if(ce.type==="content"){const be=ce.content===
null||ce.content===void 0?"":String(ce.content);He+=be,/[`~]/.test(be)&&activateDeferredCodingModeFromStream(
He),Ne||(Ne=U.querySelector(".content-area")||document.createElement("div"),Ne.className="prose pros\
e-invert text-sm break-words",U.contains(Ne)||U.appendChild(Ne)),Jt=!0,Ot("content",!!be)}else if(ce.
type==="error"){Vt=!0,Rt=!0,U.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(ce.content)),showToast(
ce.content||"Unknown error","error",!0);break}}catch{}if(an&&Pe&&(Pe.textContent=nt,userAutoScroll&&
(Pe.scrollTop=Pe.scrollHeight)),Jt&&Ne){const lt=Date.now();if(lt-Ze>100){const ce=snapshotCodeCollapse(
Ne);renderAiMarkdownInto(Ne,He,{incrementalMath:!0}),applyCodeCollapse(Ne,ce,!0),Ze=lt}}scrollToBottom()}
if(at&&at(),Ne){const Le=snapshotCodeCollapse(Ne);renderAiMarkdownInto(Ne,He,{incrementalMath:!0}),applyCodeCollapse(
Ne,Le,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),U)if(queueHighlight(U,He),enableLatencyMetrics){
const Le=nowPerfMs()-tt;reportFirstTokenLatency({is_total:!0,latency_seconds:Le/1e3,latency_ms:Le,thread_id:Pt||
currentThreadId,job_id:currentJobId,model:C.model,client_sent_at_ms:ut,client_done_at_ms:Date.now()});
let Re='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',Xe=null;gt!==null&&(Xe=gt),kt!==null&&(Xe===null||kt<Xe)&&(Xe=
kt),rt!==null&&(Xe===null||rt<Xe)&&(Xe=rt),Xe!==null&&(Re+=`<div>Initial: ${(Xe/1e3).toFixed(2)}s</d\
iv>`),rt!==null&&rt!==Xe&&(Re+=`<div>Content: ${(rt/1e3).toFixed(2)}s</div>`),Re+=`<div class="font-\
bold text-gray-300">Total: ${(Le/1e3).toFixed(2)}s</div>`,currentJobId&&(Re+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),Re+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,Re+="</div>",U.insertAdjacentHTML("beforeend",Re)}else U.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),U&&U.querySelectorAll(".thought-co\
ntent").forEach(Re=>Re.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0}),!Vt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(!0,{persist:!1})),userAutoScroll&&
scrollToBottom(),document.querySelectorAll(".message-group").length<=2||!currentThreadTitle||currentThreadTitle===
"New Chat"||currentThreadTitle==="No Title"?apiFetch("/api/generate_title",{method:"POST",headers:{"\
Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,model_id:get("model\
-select").value})}).then(Le=>Le.json()).then(Le=>{Le.title&&(document.title=Le.title+" - AI Chat",setCurrentChatHeaderTitle(
Le.title),loadThreads())}):loadThreads(!1)}catch(N){let ae=!1;const _e=N.name==="AbortError"&&isManualStopAbortForThread(
Ce);if(N.name==="AbortError"&&!_e&&(ae=await syncThreadAfterAbortedStream(Ce,{retries:2,retryDelayMs:180,
notifyOnFailure:!0})),sendClientDebugLog("error",`Prompt send error: ${N.message}`),!pt){de&&de.remove();
const Me=U&&U.closest(".fade-in");Me&&Me.remove(),delete messageStore[Y],delete messageMeta[Y]}if(N.
serverCode==="request_already_accepted"&&N.acceptedJobId&&N.acceptedThreadId)pt=!0,St={job_id:N.acceptedJobId,
thread_id:String(N.acceptedThreadId),model:C.model},get("prompt-input").value="",get("prompt-input").
style.height="auto",resetUploadState(),clearQuote();else if(pt&&!_e)mt={job_id:normalizeJobIdForUi(currentJobId),
thread_id:currentThreadId!=null?String(currentThreadId):null,model:C.model},window.ConnectionMonitor.
setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(N.serverCode==="turnstile_required"){const Me=await getTurnstileToken();Me?(await verifyTurnstileOnServer(
Me,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(N.serverCode==="api_key_missing"){const Me=N.serverModel||C.model,je=await showApiKeyRequiredModalAsync(
Me);je==="set"?_t=!0:je==="switch"?showModal("model-modal"):showToast(N.message||`${getModelNameById(
Me)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(N.
name!=="AbortError"){const Me="Connection Error: "+N.message;showToast(Me,"error",!0)}j&&!ae&&q()}finally{
zt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),at&&at(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===G&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(St){const N=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=St.thread_id,(N!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(St,currentThreadId)}
if(mt&&mt.thread_id)return reconnectPendingStreamUntilAvailable(mt,mt.thread_id);if(_t)return sendMessage()}
o(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?
String(e.model):"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const s=get(n);if(!s)
return;if(activeStreamingBubbleId=n,s.classList.add("ai-pending-bubble"),!s.querySelector(".content-\
area.skeleton-pending")){const q=s.querySelector(".content-area");q?q.outerHTML=buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):s.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const a=currentThreadId,
r=i.toLowerCase(),l=r.includes("gemini")||r.includes("o1")||r.includes("o3")||r.includes("gpt-5")||r.
includes("reasoning")&&!r.includes("non-reasoning");let d=null;const p=o(q=>!l||!s?null:((!d||!s.contains(
d))&&(d=s.querySelector(".thought-content")),d||(s.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),d=s.querySelector(".thought-content")),d&&(d.setAttribute("data-pla\
ceholder","1"),d.textContent=q||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),d),
"ensureThoughtPlaceholder");l&&p("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let g="",h="",v="",b=!0,w=null,x=null,L=null,T=!1;const $={};let j=0,J=!1;const X=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Te=!1,P=!1;try{const q=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!q.ok)throw new Error(`Resume failed (${q.status})`);window.ConnectionMonitor&&
(P=!0,window.ConnectionMonitor.operationStarted()),X&&X.setPhase("waiting");const Y=q.body.getReader(),
de=new TextDecoder;for(;!J;){const{done:oe,value:xe}=await Y.read();if(oe)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),X&&X.setPhase("receiving"),g+=de.decode(xe,{stream:!0});let pe=g.
split(`
`);g=pe.pop();let ke=!1,te=!1;for(let se of pe)if(se.trim())try{const W=JSON.parse(se);if(W.type==="\
job_id"){currentJobId=W.content||t;continue}if(W.type==="search_status"){W.content==="searching"&&!L?
(s.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),L=s.querySelector(".search-box")):W.content==="done"&&L&&(L.
classList.remove("animate-pulse"),L.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{L&&L.remove(),L=null},2e3));continue}if(W.type==="mcp"){handleMcpStreamEvent(
s,W.content||{});continue}if(W.type==="mcp_decision_request"){openMcpDecisionModal(W.content||{});continue}
if(W.type==="status"){const C=W.content===null||W.content===void 0?"":String(W.content);if(b&&s){const R=C||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(s,R,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const G=s.querySelector(".content-area");G&&(G.outerHTML=buildPendingSkeletonHtml(i,R),updatePendingSkeletonStatus(
s,R,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
l&&p(C||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(b){beginPendingToStreamTransition(
s);const C=s.querySelector(".content-area");C&&(C.innerHTML=""),b=!1}if(W.type==="coding_diff")appendCodingLiveDiff(
s,W.content||{});else if(W.type==="thought"){if(w||(w=s.querySelector(".thought-content")),v+=W.content,
!w){const C='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';L?L.insertAdjacentHTML("afterend",C):s.insertAdjacentHTML("afterbegin",C),w=s.querySelector(
".thought-content")}if(w&&w.getAttribute("data-placeholder")==="1"){if(w.textContent="",w.removeAttribute(
"data-placeholder"),w){const C=w.parentElement.querySelector(".thought-header");C&&C.classList.remove(
"thinking-shimmer")}v=W.content}w.classList.remove("collapsed"),te=!0}else if(W.type==="image_analys\
is"){const C=W.content===null||W.content===void 0?"":String(W.content);if(!s)continue;let R=s.querySelector(
".image-analysis-box");if(!R){const K='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';L?L.insertAdjacentHTML("afterend",K):s.insertAdjacentHTML("afterbegin",K),R=s.querySelector(
".image-analysis-box")}const G=R.querySelector(".image-analysis-text");G&&(G.textContent=C)}else if(W.
type==="python"){const C=W.content||{},R=C.id||`py_${Date.now()}`;if(!$[R]){const K=`<div class="cod\
e-wrapper python-box collapsed" data-py-id="${R}" data-collapsed="true" data-code-key="${R}"><div cl\
ass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><di\
v class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B">\
<i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" t\
itle="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" dat\
a-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></b\
utton></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code<\
/div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-secti\
on"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"><\
/code></pre></div></div></div>`;L?L.insertAdjacentHTML("afterend",K):s.insertAdjacentHTML("afterbegi\
n",K),$[R]=s.querySelector(`[data-py-id="${R}"]`)}const G=$[R];if(G){if(C.code!==void 0){const K=C.code==
null?"":String(C.code),Z=G.querySelector(".python-code");Z&&(Z.textContent=K,Z.removeAttribute("data\
-highlighted"),queueHighlight(G,K));const ye=G.querySelector('.copy-btn[data-copy="code"]');ye&&ye.setAttribute(
"data-code",encodeURIComponent(K).replace(/'/g,"%27"))}if(C.output!==void 0){const K=C.output==null?
"":String(C.output),Z=G.querySelector(".python-output");Z&&(Z.textContent=K);const ye=G.querySelector(
'.copy-btn[data-copy="output"]');ye&&ye.setAttribute("data-code",encodeURIComponent(K).replace(/'/g,
"%27"))}}}else if(W.type==="content"){const C=W.content===null||W.content===void 0?"":String(W.content);
h+=C,/[`~]/.test(C)&&activateDeferredCodingModeFromStream(h),x||(x=s.querySelector(".content-area")||
document.createElement("div"),x.className="prose prose-invert text-sm break-words",s.contains(x)||s.
appendChild(x)),ke=!0}else if(W.type==="error"){T=!0,J=!0,s.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
W.content)),showToast(W.content||"Unknown error","error",!0);break}}catch{}if(te&&w&&(w.textContent=
v,userAutoScroll&&(w.scrollTop=w.scrollHeight)),ke&&x){const se=Date.now();if(se-j>100){const W=snapshotCodeCollapse(
x);renderAiMarkdownInto(x,h,{incrementalMath:!0}),applyCodeCollapse(x,W,!0),j=se}}scrollToBottom()}if(X&&
X(),x){const oe=snapshotCodeCollapse(x);renderAiMarkdownInto(x,h,{incrementalMath:!0}),applyCodeCollapse(
x,oe,!0)}vibrateHelper([100,50,100]),s&&queueHighlight(s,h),s&&s.querySelectorAll(".thought-content").
forEach(xe=>xe.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(q){const Y=q.name==="AbortError"&&isManualStopAbortForThread(a);q.name==="Abor\
tError"&&!Y&&await syncThreadAfterAbortedStream(a,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),Y||
(Te=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{P&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),X&&X(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Te)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i},a)}o(resumePendingStream,"resumePendingStream");function updateThreadHighlighting(){
const e=get("thread-list");if(!e)return;e.querySelectorAll("[data-thread-id]").forEach(n=>{n.dataset.
threadId===String(currentThreadId)?n.classList.add("bg-gray-700/60","border-l-2","border-blue-500"):
n.classList.remove("bg-gray-700/60","border-l-2","border-blue-500")})}o(updateThreadHighlighting,"up\
dateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){snapshotSidebarHistory("\
loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory("loadThreads-s\
tart append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?t.value:"";
if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");return}const s=await(await apiFetch(
`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&page=${threadPage}`)).json(),a=get("th\
read-list");if(!a)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-r\
eplace-settings-open");return}const l=s&&Array.isArray(s.threads)?s.threads.length:-1,d=a.querySelectorAll(
"[data-thread-id]").length;if(l===0&&d>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(a.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const p=get("scroll-sentinel");p&&threadObserver.
observe(p)}}const r=get("scroll-sentinel");s&&Array.isArray(s.threads)?(s.threads.forEach(l=>{const d=String(
l.id),p=document.createElement("div"),g=l.is_bookmarked?"text-yellow-400":"text-gray-500",h=l.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",b=d===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";p.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${b}`,p.dataset.threadId=d,p.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${g} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${d}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(l.title||"No Title")}</spa\
n>${h}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${d}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${d}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,p.onclick=w=>{w.target.closest("button")||w.target.closest("[dat\
a-thread-actions]")||loadMessages(d)},r?a.insertBefore(p,r):a.appendChild(p)}),hasMoreThreads=!!s.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+s.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error("Faile\
d to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,s=60,a=88,r=52,l=.5,d=8;let p=0,g=!1,h=0,v=null;
const b=o(()=>get(i),"indicatorEl"),w=o(()=>{const T=b();return T?T.querySelector(".ptr-pull-label"):
null},"labelEl"),x=o(T=>{const $=b();if(!$)return;$.style.height=Math.min(T,a)+"px",$.classList.toggle(
"active",T>2),$.classList.toggle("pull-ready",T>=s);const j=w();j&&(j.textContent=T>=s?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),L=o(()=>{const T=b();T&&(T.style.height=
"0px",T.classList.remove("active","pull-ready","refreshing"),T.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",T=>{if(v){g=!1;return}if(n.scrollTop>0){g=!1;return}const $=T.
touches[0];$&&(p=$.clientY,h=0,g=!0)},{passive:!0}),n.addEventListener("touchmove",T=>{if(!g||v)return;
if(n.scrollTop>0){g=!1;return}const $=T.touches[0];if(!$)return;const j=$.clientY-p;if(j<=0){h>0&&(h=
0,x(0)),g=!1;return}const J=b();J&&!J.classList.contains("dragging")&&J.classList.add("dragging"),h=
Math.min(j*l,a),x(h),j>=d&&T.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!g||
(g=!1,v))return;const T=b();T&&T.classList.remove("dragging");const $=h>=s;if(h=0,!$){L();return}let j;
try{j=t()}catch{j=null}const J=b();if(J){J.classList.add("refreshing"),J.style.height=r+"px";const X=J.
querySelector(".ptr-pull-label");X&&(X.textContent="\u66F4\u65B0\u4E2D...")}j&&typeof j.then=="funct\
ion"?(v=j,j.catch(()=>{}).finally(()=>{v=null,L()})):(v=Promise.resolve(),setTimeout(()=>{v=null,L()},
400))}),n.addEventListener("touchcancel",()=>{g=!1,h=0,L()})}o(initPullToRefresh,"initPullToRefresh");
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
!i)return;const s=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.querySelector(
'[data-mcp-card="'+s+'"]'))return;const a=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${s}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",a);return}if(t.type==="result"){let a=i.querySelector('[dat\
a-mcp-card="'+s+'"]');const r=t.summary||"";if(a)a.classList.remove("mcp-running"),a.classList.add("\
mcp-done"),a.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const l=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${s}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",l),a=i.querySelector('[data-mcp-card="'+s+'"]')}if(r){const l=document.
createElement("div");l.className="mcp-box-note",l.textContent=r.split(`
`)[0].slice(0,220),a&&a.appendChild(l)}return}if(t.type==="error"){let a=i.querySelector('[data-mcp-\
card="'+s+'"]');const r=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(a)a.classList.remove("mcp-running"),a.classList.add("mcp-error"),a.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const d=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${s}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",d),a=i.querySelector('[data-mcp-card="'+s+'"]')}const l=document.
createElement("div");l.className="mcp-box-note mcp-box-note-err",l.textContent=String(r).slice(0,300),
a&&a.appendChild(l);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const a=get("mcp-decision-modal");if(a&&!a.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}o(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),s=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),s){let l=e.
args_preview||"";try{const d=JSON.parse(l);l=JSON.stringify(d,null,2)}catch{}s.textContent=l}const a=get(
"mcp-decision-allow"),r=get("mcp-decision-deny");a&&(a.onclick=()=>submitMcpDecision("allow")),r&&(r.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}o(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}o(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const s=e[0],a=s.
thread_id;n.textContent=e.length===1?`${s.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),a&&await loadMessages(a)});const r=get(
"batch-notification-close");r&&(r.onclick=()=>t.classList.add("hidden"))}o(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({})),n=new Set((t.active||[]).map(i=>String(i.thread_id)));currentThreadId&&
n.has(String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),Array.
isArray(t.completed)&&t.completed.length&&(showGeminiBatchCompletionBanner(t.completed),loadThreads(
!1))}catch{}finally{geminiBatchStatusPollBusy=!1}}}o(refreshGeminiBatchStatus,"refreshGeminiBatchSta\
tus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,15e3);async function toggleBookmark(e,t){
e&&e.stopPropagation(),await apiFetch(`/api/threads/${t}/bookmark`,{method:"POST"}),loadThreads()}o(
toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){const n=++threadLoadSequence;window.
closeHistoryModal&&window.closeHistoryModal();const i=!!t.preserveDraft,s=!!t.silent;s||resumeChatAutoScroll(
{scroll:!1});const a=s?snapshotCodeCollapseByMessage(get("chat-container")):null;let r="",l="",d=[];
if(i){const p=get("prompt-input");r=p?p.value:"",l=p?p.style.height:"",d=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();currentThreadId=e!=null?String(e):e,
t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),
get("welcome-screen").classList.add("hidden"),s||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const p=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);p.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const g=await apiFetch(p.toString());if(!g.
ok)throw new Error(`thread request failed (${g.status})`);const h=await g.json();if(!h||!Array.isArray(
h.messages))throw new Error("invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(
h&&h.title),allMessages=h.messages,threadHasOlderMessages=!!h.has_older_messages,oldestLoadedMessageId=
h.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const v=(allMessages||[]).filter(w=>w.
role==="user"&&w.content).map(w=>w.content);if(promptHistory=[...new Set(v.slice().reverse())],historyIndex=
-1,tempPrompt="",currentThreadPending=h.pending_job||null,setTemporaryChatUiState(!!(h&&h.is_temporary)),
applyTemporaryChatRuntimeMeta(h||{}),ensureTemporaryChatHeartbeat(!0),get("thread-custom-instruction")&&
(get("thread-custom-instruction").value=h.custom_instruction||""),h.last_model&&selectModelById(h.last_model),
get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!!h.enable_prompt_caching,updatePromptCacheUi()),
h.last_gem_uuid&&loadedGems.length>0){const w=loadedGems.find(x=>x.uuid===h.last_gem_uuid);w&&(threadGemMap[currentThreadId]=
w,applyActiveGem(w))}const b=localStorage.getItem(`fixed_branch_${currentThreadId}`);if(b&&allMessages.
find(w=>String(w.id)===String(b))?currentLeafId=b:allMessages.length>0?currentLeafId=allMessages[allMessages.
length-1].id:currentLeafId=null,renderThreadTree({silent:s,keepScroll:s}),s&&a?applyCodeCollapseByMessage(
get("chat-container"),a,!0):s||applyCodeCollapseByMessage(get("chat-container"),null,!0),currentThreadPending&&
!s&&!isPendingJobSuppressed(currentThreadPending.job_id)&&resumePendingStream(currentThreadPending),
i){const w=get("prompt-input");w&&(w.value=r||"",l?w.style.height=l:w.style.height="auto"),currentImageUrls=
d,currentImageUrls&&currentImageUrls.length?(get("file-preview").classList.remove("hidden"),get("fil\
e-name").innerText=`${currentImageUrls.length} files ready`):get("file-preview").classList.add("hidd\
en"),schedulePromptTokenEstimate(!0)}if(i||schedulePromptTokenEstimate(!0),window.innerWidth<768&&get(
"overlay").click(),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
return!0}catch(p){return n!==threadLoadSequence||(console.error("Failed to load chat thread:",p),s||
showChatLoadError(e),s||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)),!1}}o(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const a=await(await apiFetch(i.toString())).json(),r=Array.isArray(a.messages)?a.messages:[];
if(r.length){const l=new Set(allMessages.map(p=>p.id)),d=r.filter(p=>!l.has(p.id));d.length&&(allMessages=
d.concat(allMessages))}if(threadHasOlderMessages=!!a.has_older_messages,oldestLoadedMessageId=a.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const l=e.
scrollHeight;e.scrollTop=Math.max(0,n+(l-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,s=get("chat-container");if(!s)return;let a=null;if(i&&(a=s.scrollTop),s.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const r={};allMessages.
forEach(b=>{r[b.id]=b,b.childrenIds=[]}),allMessages.forEach(b=>{b.parent_id&&r[b.parent_id]&&r[b.parent_id].
childrenIds.push(b.id)}),(!currentLeafId||!r[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const l=[];let d=r[currentLeafId];for(;d;)l.unshift(d),d=r[d.parent_id];const p=buildTokenTotals(
l),g=buildTokenTotals(allMessages),h=document.createDocumentFragment();if(threadHasOlderMessages){const b=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
w=loadingOlderMessages?"disabled":"",x=document.createElement("div");x.className="mb-3 text-center",
x.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${w}><i class="fas fa-clock-rotate-left mr-1"></i>${b}</button>`,h.appendChild(x)}
l.forEach(b=>{const w=b.parent_id?r[b.parent_id]:null,x=w?w.childrenIds:allMessages.filter(T=>!T.parent_id).
map(T=>T.id),L=x.length>1?{current:x.indexOf(b.id)+1,total:x.length,siblings:x}:null;renderMessage(b.
id,b.role,b.content,b.image_url,b.thought_data,b.model,L,n,b.quote_text,b.tokens,b.tokens_in,b.tokens_out,
b.is_encrypted,b.tokens_content,b.tokens_thought,h,!1,b.parent_id,b.gem_name,b.batch_job)});const v=currentThreadPending;
if(v&&!isPendingJobSuppressed(v.job_id)){const b=v.message_id,w=new Set(l.map(T=>T.id)),x=l.length?l[l.
length-1]:null;if(b&&w.has(b)&&currentLeafId===b||!b&&x&&x.role==="user"){const T=v.job_id?`pending-${v.
job_id}`:null;renderPendingMessage(h,n,!1,T,v.model||null)}}if(s.appendChild(h),updateTotalTokenBar(
p.tokens_total,p,g),currentParentId=currentLeafId,i&&a!==null?restoreThreadTreeScroll(s,a):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(s,s&&s.textContent||"");else if(queueHighlight(s),l.length){
const b=l[l.length-1]&&l[l.length-1].content;queueMathTypeset(s,b)}}o(renderThreadTree,"renderThread\
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
abel"></span></div>',Array.isArray(t)&&t.forEach(i=>{const s=document.createElement("div");s.className=
"gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between it\
ems-center group",s.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa\
-gem text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><div class="flex it\
ems-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group\
-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${i.uuid}')"><i class="fas fa-p\
encil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:o\
pacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${i.uuid}')"><i class\
="fas fa-trash text-[10px]"></i></button></div>`,s.onclick=a=>{a.target.closest("button")||activateGem(
i)},n.appendChild(s)})}catch(e){console.error("Failed to load gems:",e)}}o(loadGems,"loadGems");async function openEditGemModal(e,t){
e.stopPropagation(),editingGemUuid=t;try{const i=await(await apiFetch(`/api/gems/${t}`)).json();get(
"gem-name").value=i.name,get("gem-desc").value=i.description||"",get("gem-inst").value=i.instruction,
get("gem-default-model").value=i.default_model||"",renderGemFixedPromptsForEdit(i.fixed_prompts),get(
"gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-b\
tn").innerText="Save Changes",showModal("gem-modal"),location.pathname!=="/gem"&&history.pushState({
modal:"gem"},"","/gem")}catch{showToast("Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}o(openEditGemModal,"openEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.
urls.handleGems,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,
instruction:t})}),loadGems()}o(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get(
"fixed-prompts-bar");if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),
get("active-gem-name").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),
t){t.innerHTML="";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}
n.length>0?(t.classList.remove("hidden"),n.forEach((i,s)=>{const a=document.createElement("button");
a.className="fixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover\
:bg-gray-600 text-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex item\
s-center",a.style.animationDelay=`${s*40}ms`,a.textContent=String(i.name||""),a.onclick=()=>{const r=get(
"prompt-input");r&&(r.value=i.content,r.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(
a)})):t.classList.add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML=
"",t.classList.add("hidden"));get("sys-prompt-option").style.opacity="1"}o(applyActiveGem,"applyActi\
veGem");function syncActiveGemForThread(e){const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(
t)}o(syncActiveGemForThread,"syncActiveGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({last_gem_uuid:t,thread_id:e})})}catch{}}o(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){
currentThreadId?(threadGemMap[currentThreadId]=e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\
\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,"success"),t||saveThreadGemUuid(currentThreadId,e?
e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(e),allMessages&&allMessages.length>0&&startNewChat(
{preserveGem:!0}))}o(activateGem,"activateGem");function clearActiveGem(){currentThreadId&&(delete threadGemMap[currentThreadId],
saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=null,applyActiveGem(null)}o(clearActiveGem,
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
            `,n.appendChild(i)}o(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),t=[];return e.forEach(n=>{const i=n.querySelector(
".gem-fp-name").value.trim(),s=n.querySelector(".gem-fp-content").value.trim();i&&s&&t.push({name:i,
content:s})}),t.length>0?JSON.stringify(t):null}o(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const t=get("gem-fixed-prompts-container");if(t){t.innerHTML="";try{e&&JSON.parse(e).forEach(i=>addGemFixedPromptRow(
i.name,i.content))}catch{}}}o(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}o(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}o(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=[
"sidebar-chat-title","mobile-chat-title"],s=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],a=["sidebar-chat-ttl","mobile-chat-ttl"];i.forEach(r=>{const l=get(r);l&&(l.textContent=e)}),
s.forEach(r=>{const l=get(r);l&&l.classList.toggle("hidden",!n)}),a.forEach(r=>{const l=get(r);l&&(n&&
t?(l.textContent=t,l.classList.remove("hidden")):(l.textContent="",l.classList.add("hidden")))})}o(updateCurrentChatHeaderUi,
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
if(threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),resetUploadState(),stopTemporaryChatHeartbeat(),
setTemporaryChatUiState(!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,allMessages=
[],promptHistory=[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=null,
loadingOlderMessages=!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(
0),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
e.skipHistory||history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").
classList.remove("hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("threa\
d-custom-instruction").value=""),get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!1,
updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(activeGem):applyActiveGem(null),loadThreads(),
window.innerWidth<768&&get("overlay").click()}o(startNewChat,"startNewChat");let threadModalLoadSeq=0;
window.openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.
handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=i.id!==null&&i.id!==void 0?String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.
is_temporary)),setCurrentChatHeaderTitle(i&&i.title),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,t=String(currentThreadId);modalThreadId=t,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[n,i]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${t}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==t)return;if(n.ok){const s=await n.
json(),a=get("thread-app-global-sys-prompt-preview");a&&(a.value=s.global_system_prompt_effective||"");
const r=get("thread-app-global-sys-prompt-preview-status");r&&(s.global_system_prompt_enabled===!1?r.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":s.global_system_prompt_uses_time_fallback?
r.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
r.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=s.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=s.system_prompt_enabled!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=s.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",s.auto_system_prompt_notices_config||{})}if(i.ok){const s=await i.json();if(e!==threadModalLoadSeq||
modalThreadId!==t)return;const a=get("thread-custom-instruction");a&&(a.value=s.custom_instruction||
"");const r=get("thread-include-global-instruction");r&&(r.checked=s.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),s=i?i.value:"",a=get("thread-include-global-instruction"),r=a?a.checked:
!0,l=get("thread-global-sys-prompt"),d=get("thread-global-sys-prompt-enabled");let p=null;try{p=l||d?
{system_prompt:l?l.value:"",system_prompt_enabled:d?d.checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:collectAutoSystemPromptConfigFromForm(
"thread")}:null}catch(g){sendClientDebugLog("error","Payload construction failed: "+g.message)}try{sendClientDebugLog(
"info","Starting PUT request for thread: "+e);const g=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:s,include_global_instruction:r})});
sendClientDebugLog("info","PUT request finished, status: "+g.status);let h=!0;if(p){sendClientDebugLog(
"info","Starting POST request for user settings");const v=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(p)});h=v.ok,sendClientDebugLog(
"info","POST request finished, status: "+v.status)}g.ok&&h?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\
\u308C\u307E\u3057\u305F","success")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(g){sendClientDebugLog("error","Save failed with error: "+g.message),showToast("\u30A8\u30E9\u30FC\
: "+g.message,"error",!0)}finally{t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=
()=>{syncCompressionSettingsUi(),showModal("compression-modal"),location.pathname!=="/compression"&&
history.pushState({modal:"compression"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal(
"compression-modal"),!e&&location.pathname==="/compression"&&history.back()},get("save-compression-s\
ettings-btn").onclick=()=>{const e=get("compression-max-size").value,t=get("compression-max-dim").value,
n=get("compression-output-type").value,i=get("compression-format-only").checked;setCompressionSettings(
e,t,n,i);const s=o((r,l)=>{get(r)&&get(l)&&(get(l).value=get(r).value)},"syncBack");s("modal-gpt-ima\
ge-size","gpt-image-size"),s("modal-gpt-image-quality","gpt-image-quality"),s("modal-gpt-image-forma\
t","gpt-image-format"),s("modal-gpt-image-compression","gpt-image-compression"),s("modal-gemini-imag\
e-aspect","gemini-image-aspect"),s("modal-gemini-image-size","gemini-image-size"),s("modal-grok-imag\
e-aspect","grok-image-aspect"),s("modal-grok-image-resolution","grok-image-resolution"),s("modal-gro\
k-image-quality","grok-image-quality"),s("modal-ocr-table-format","ocr-table-format"),s("modal-ocr-p\
ages","ocr-pages");const a=o((r,l)=>{get(r)&&get(l)&&(get(l).checked=get(r).checked)},"syncBackChk");
a("modal-ocr-extract-header","ocr-extract-header"),a("modal-ocr-extract-footer","ocr-extract-footer"),
a("modal-ocr-include-blocks","ocr-include-blocks"),a("modal-ocr-include-images","ocr-include-images"),
window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","s\
uccess")};async function deleteGem(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.
urls.handleGemItem.replace("0",t),{method:"DELETE"}),loadGems())}o(deleteGem,"deleteGem");async function renameThread(e,t){
e.stopPropagation();const n=prompt("Title:");if(n){const i=await apiFetch(CHAT_CONFIG.urls.updateTitle.
replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:n})}),
s=await i.json().catch(()=>({}));i.ok&&currentThreadId===String(t)&&setCurrentChatHeaderTitle(s&&s.title||
n),loadThreads()}}o(renameThread,"renameThread");async function deleteThread(e,t){e.stopPropagation(),
confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),
currentThreadId===t?startNewChat():loadThreads())}o(deleteThread,"deleteThread");async function deleteMessage(e){
confirm("Delete this message and subsequent history?")&&(await apiFetch(CHAT_CONFIG.urls.deleteMessage.
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
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const s=n&&n.filename?n.
filename:i.split("/").pop(),a=n&&n.source?String(n.source):"attachment",r=!!(n&&n.is_image),l=n&&n.url?
n.url:buildPdfAttachmentUrl(i),d=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:s,source:a,isImage:r,url:l,previewUrl:d}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],s=n.some(d=>maybeNeedsMathJax(d.content)||maybeNeedsMathJax(d.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",a=t.title||"AI Chat",r=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],l=n.map(d=>{const p=d.
role==="user",g=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",h=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",v=p?`<div \
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
void 0&&x.push(`parent:#${d.parent_id}`);const L=x.length?`<div class="message-meta">${pdfEscapeAttr(
x.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${p?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${p?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${p?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(d.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${g}
                            ${v}
                            ${h}
                            ${w}
                            ${L}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(a)} - PDF Export</title>
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
        <h1>${pdfEscapeAttr(a)}</h1>
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
"error",!0);return}const a=document.createElement("iframe");activePdfPrintFrame=a,a.setAttribute("ar\
ia-hidden","true"),a.style.position="fixed",a.style.right="0",a.style.bottom="0",a.style.width="1px",
a.style.height="1px",a.style.opacity="0",a.style.pointerEvents="none",a.style.border="0";let r=null;
const l=o(()=>{r&&(clearTimeout(r),r=null),t&&t.remove(),(activePdfPrintFrame===a||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{a.parentNode&&a.parentNode.removeChild(a)}catch{}},"cleanup");r=setTimeout(
()=>{activePdfPrintFrame===a&&(console.log("PDF print cleanup fallback triggered"),l())},6e4),a.onload=
async()=>{try{const g=a.contentDocument,h=a.contentWindow;if(!g||!h){l(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(s&&s.messages)?s.messages:[]).
some(T=>maybeNeedsMathJax(T.content)||maybeNeedsMathJax(T.thought_text))&&(h.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),g.fonts&&g.fonts.ready)try{await g.fonts.ready}catch{}t.update(60);const w=Array.from(g.
images||[]),x=Promise.all(w.map(T=>T.complete?Promise.resolve():new Promise($=>{T.addEventListener("\
load",$,{once:!0}),T.addEventListener("error",$,{once:!0})})));if(await Promise.race([x,new Promise(
T=>setTimeout(T,5e3))]),t.update(80),g.getElementById("MathJax-script")){let T=0;for(;T<100&&(!h.MathJax||
typeof h.MathJax.typesetPromise!="function");)await new Promise($=>setTimeout($,50)),T++;if(h.MathJax&&
typeof h.MathJax.typesetPromise=="function")try{await h.MathJax.typesetPromise()}catch($){console.error(
"PDF MathJax typeset failed",$)}}t.update(95),setTimeout(()=>{try{h.focus(),h.addEventListener("afte\
rprint",()=>{l()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),h.print()}catch{l(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{l(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(s),p=new Blob([d],{type:"text/html"});a.src=URL.createObjectURL(
p),document.body.appendChild(a)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),s=o((d,p)=>i.
compare(d.filename||"",p.filename||""),"nameAsc"),a=o((d,p)=>i.compare(p.filename||"",d.filename||""),
"nameDesc"),r=o((d,p)=>(Number(p.ts)||0)-(Number(d.ts)||0),"tsDesc"),l=o((d,p)=>(Number(d.ts)||0)-(Number(
p.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,p)=>s(d,p)||r(d,p)):t==="name_desc"?n.sort((d,p)=>a(
d,p)||r(d,p)):t==="oldest"?n.sort((d,p)=>l(d,p)||s(d,p)):n.sort((d,p)=>r(d,p)||s(d,p)),n}o(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}o(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}o(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}o(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}o(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const p=t.querySelector(".lib-empty-state");p&&p.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const p=get("lib-total-count");p&&(p.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
s=getLibSearchQuery(),a=i.filter(p=>lib.favoritesOnly&&!p.is_favorite?!1:!s||fileNameForSearch(p).includes(
s)),r=get("lib-total-count");if(r){const p=Number(lib.totalCount)||lib.files.length;lib.hasMore||s||
lib.favoritesOnly?r.innerText=`${lib.files.length} / ${p} files`:r.innerText=`${p} files`}if(!a.length){
if(n)return;const p=lib.favoritesOnly&&!s?"fa-star":"fa-search",g=lib.favoritesOnly&&!s?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",h=lib.favoritesOnly&&
!s?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${p}"></i></div>\
<p class="lib-empty-title">${g}</p><p class="lib-empty-sub">${h}</p></div>`;return}let l=0;(n?sortLibraryFiles(
e).filter(p=>lib.favoritesOnly&&!p.is_favorite?!1:!s||fileNameForSearch(p).includes(s)):a).forEach(p=>{
try{const g=renderLibraryItem(p,l++);t.appendChild(g)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
s=(n?t.filter(d=>fileNameForSearch(d).includes(n)):t).filter(d=>d.type==="image"),a=lib.favoritesOnly?
s.filter(d=>d.is_favorite):s;if(!a.length)return;const r=a.map(d=>({url:d.url,filename:d.filename||d.
original_filename||d.url.split("/").pop(),element:null}));let l=r.findIndex(d=>d.url===e.url);l===-1&&
(l=0),openViewerWithItems(r,l)}o(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}o(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,s=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),a=e.type==="\
image"?`<img src="${escapeHtml(i)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(s)}"></i></div><span class="lib-file-badge">${escapeHtml(s?s.toUpperCase():"FI\
LE")}</span></div>`,r=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,l=e.is_favorite?" is-favorite":"",d=e.is_favorite?"fas fa-star":"far f\
a-star",p=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
g=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${l}" title="${p}\
" aria-label="${p}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
h=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${a}</div>${r}${g}${h}`,n.
onclick=()=>{lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-se\
lected")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(x=>{x.onclick=L=>{L.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const b=n.querySelector(".lib-del-btn");b&&(b.onclick=async x=>{x.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const w=n.querySelector(".lib-favorite-btn");return w&&
(w.onclick=async x=>{x.stopPropagation(),w.disabled=!0;try{const L=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
T=await L.json().catch(()=>({}));if(!L.ok||typeof T.is_favorite!="boolean")throw new Error(T.error||
"favorite update failed");e.is_favorite=T.is_favorite,renderLibraryGrid(),showToast(T.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),w.disabled=!1}}),n}o(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}o(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",s=FILE_BASE_URL+e,a=i==="image"?FILE_THUMB_BASE_URL+
e:null,r={filename:t,original_filename:t,filepath:e,url:s,thumbnail_url:a,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(r),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
o(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(a=>a.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const s=(i||"").trim();if(!s){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const a=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:s})}),r=await a.json().catch(
()=>({}));if(!a.ok){showToast(r.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=r.filename||s,setAttachmentNameForPath(e,t.filename));const l=get(
"upload-list");l&&l.querySelectorAll("[data-filename]").forEach(d=>{d.getAttribute("data-filename")===
e&&setRowAttachmentName(d,t?t.filename:r.filename||s)}),renderLibraryGrid(),window.updateLibSelectionUi(),
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
from(lib.selected)[0],t=(lib.files||[]).find(s=>s&&s.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const s=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);s.searchParams.set("filepath",e);const a=await apiFetch(
s.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),r=await a.json().catch(()=>({}));
if(!a.ok)throw new Error(r.error||`HTTP ${a.status}`);const l=Array.isArray(r.chats)?r.chats:[];if(!l.
length){i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(i.innerHTML="",l.forEach(d=>{const p=document.createElement("div");p.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const g=d.updated_at?new Date(d.
updated_at).toLocaleString():"";p.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(g)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const h=p.querySelector("button");h&&(h.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(p)}),r.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const s=CHAT_CONFIG.
urls.getFilesLib;let a=null,r=!1;try{const d=getLibSortOrder(),p=getLibSearchQuery(),g=e?lib.nextOffset:
0,h=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(g),sort:d,q:p,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(s+"?"+h.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);a=await v.json(),r=!0}catch(d){i=d}if(!r){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let l=Array.isArray(a)?a:
a&&Array.isArray(a.files)?a.files:[];a&&!Array.isArray(a)&&(lib.totalCount=Number(a.total)||0,lib.hasMore=
!!a.has_more,lib.nextOffset=(Number(a.offset)||0)+(Number(a.limit)||l.length));try{const d=FILE_BASE_URL,
p=FILE_THUMB_BASE_URL,g=new Set(l.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(l.length>=LIBRARY_PAGE_SIZE||!v||g.has(v))return;const b=getAttachmentNameForPath(
v)||v.split("/").pop()||v,w=(b.split(".").pop()||"").toLowerCase(),x=["png","jpg","jpeg","webp","gif"].
includes(w)?"image":"file",L=x==="image"?p+v:null;l.unshift({filename:b,original_filename:b,filepath:v,
url:d+v,thumbnail_url:L,type:x,ext:w,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),g.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=l.filter(g=>g&&g.filepath&&
g.url);let p=[];if(e){const g=new Set(lib.files.map(h=>h.filepath));p=d.filter(h=>!g.has(h.filepath)),
lib.files.push(...p)}else lib.files=d;lib.files.forEach(g=>{g&&g.filepath&&setAttachmentNameForPath(
g.filepath,g.filename||g.original_filename||"")}),lib.fileSet=new Set(lib.files.map(g=>g.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
p:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}o(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}o(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(s=>{const a=isAudioPath(s),
r=isVideoPath(s);if(a&&!e.audio||r&&!e.video){a&&(t+=1),r&&(n+=1);return}const l=normalizeAttachmentPath(
s);if(!l)return;const d=(lib.files||[]).find(p=>p&&p.filepath===s);d&&d.filename&&setAttachmentNameForPath(
l,d.filename),currentImageUrls.includes(l)||currentImageUrls.push(l),setAttachmentSourceForPath(l,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const s=[];t&&s.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&s.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${s.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
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
(e,t)=>{const n=decodeURIComponent(t),i=o(()=>{const s=e.getAttribute("data-copy")||"";e.innerHTML=s===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},s=>{console.error(s),e.innerHTML=
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
let t=0,n=e;const i={};for((allMessages||[]).forEach(s=>i[s.id]=s);n&&i[n];){const s=i[n];t+=s.tokens||
Number(s.tokens_in||0)+Number(s.tokens_out||0),n=s.parent_id}return t}o(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(s=>i[s.id]=s);n&&i[n];){const s=i[n],a=s.model||"Unknown";t[a]||(t[a]={total:0,in:0,out:0,
thought:0});const r=s.tokens||Number(s.tokens_in||0)+Number(s.tokens_out||0);t[a].total+=r,t[a].in+=
Number(s.tokens_in||0),t[a].out+=Number(s.tokens_out||0),t[a].thought+=Number(s.tokens_thought||0),n=
s.parent_id}return t}o(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(s=>t[s.id]={...s,children:[]}),
allMessages.forEach(s=>{s.parent_id&&t[s.parent_id]?t[s.parent_id].children.push(t[s.id]):s.parent_id||
n.push(t[s.id])});function i(s){const a=document.createElement("div");a.className="flex flex-col ite\
ms-center mt-4";const r=document.createElement("div"),l=String(s.id)===String(currentLeafId),d=s.id===
threadFixedBranchId,p=branchLabelNames[s.id]||(s.role==="user"?"User":"AI"),g=getCumulativeTokensForNode(
s.id);if(r.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===s.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${l?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,r.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(p)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(s.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${g}</span>
                    </div>
                    ${d?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${l?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,r.onclick=h=>{h.stopPropagation(),selectedBranchNodeId=s.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},a.appendChild(r),s.children.length>0){const h=document.createElement("div");
h.className="w-px h-4 bg-gray-700",a.appendChild(h);const v=document.createElement("div");v.className=
"flex gap-4 items-start",s.children.forEach(b=>v.appendChild(i(b))),a.appendChild(v)}return a}o(i,"r\
enderNodeRecursive"),n.forEach(s=>e.appendChild(i(s)))}o(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function updateBranchDetailPane(){const e=get("branch-detail-panel"),t=get("bran\
ch-empty-panel");if(!selectedBranchNodeId||!allMessages){e.classList.add("hidden"),t.classList.remove(
"hidden");return}const n=allMessages.find(d=>d.id===selectedBranchNodeId);if(!n)return;e.classList.remove(
"hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get("br-date").innerText=n.created_at||
"-",get("br-model").innerText=n.model||"-";const i=n.tokens||Number(n.tokens_in||0)+Number(n.tokens_out||
0),s=getCumulativeTokensForNode(n.id);get("br-tokens").innerHTML=`<span title="Current message token\
s">${i}</span> <span class="text-gray-500">/</span> <span class="text-purple-400 font-bold" title="P\
ath total tokens">${s} total</span>`;const a=get("branch-model-breakdown"),r=getPerModelTokensForPath(
n.id);a.innerHTML="",Object.entries(r).sort((d,p)=>p[1].total-d[1].total).forEach(([d,p])=>{const g=document.
createElement("div");g.className="bg-gray-800/50 p-2 rounded border border-gray-700/50",g.innerHTML=
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
                `,a.appendChild(g)}),get("br-name-input").value=branchLabelNames[n.id]||"";const l=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(l.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
l.classList.replace("bg-amber-600","bg-gray-600")):(l.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
l.classList.replace("bg-gray-600","bg-amber-600"))}o(updateBranchDetailPane,"updateBranchDetailPane"),
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
updateBranchDetailPane()},500))};const showApiKeyRequiredModalAsync=o(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const s=i?get(i.inputId):null;get("api-key-modal-input").
value=s?s.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const a=get(
"api-key-modal-save-btn"),r=get("api-key-modal-fallback-btn"),l=get("api-key-modal-cancel-btn"),d=o(
()=>{a.onclick=null,r.onclick=null,l.onclick=null},"cleanup"),p=o(g=>{g.key==="Enter"&&(g.preventDefault(),
a.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",p),a.onclick=async()=>{
const g=get("api-key-modal-input").value.trim();if(!g){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const h=get(i.inputId);h&&(h.value=g);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:g})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=g)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",p),d(),t("set")},r.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",p),d(),t("switch")},l.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",p),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const g=get("api-key-modal-input");g&&g.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let s=!1;async function a(r,l){if(s||!isClientDebugLogEnabled()||l&&l[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;s=!0;const d=l.map(p=>{try{return p instanceof Error?p.stack||p.message:typeof p=="object"?JSON.
stringify(p):String(p)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(r,d)}catch{}finally{
s=!1}}o(a,"sendToServer"),console.log=function(...r){e.apply(console,r),a("log",r)},console.error=function(...r){
t.apply(console,r),a("error",r)},console.warn=function(...r){n.apply(console,r),a("warn",r)},console.
info=function(...r){i.apply(console,r),a("info",r)},window.addEventListener("error",function(r){a("e\
xception",[r.message,r.filename,r.lineno,r.colno,r.error])}),window.addEventListener("unhandledrejec\
tion",function(r){a("promise-rejection",[r.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();
