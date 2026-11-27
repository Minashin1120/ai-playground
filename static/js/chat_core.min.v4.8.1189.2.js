document.addEventListener("DOMContentLoaded",()=>{var bi,yi;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const u=get("enable-batch-mode");u&&u.checked&&(u.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-gemini-free-tier");t&&t.addEventListener("change",
()=>{if(browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,t.checked){const u=get(
"enable-gemini-flex");u&&u.checked&&(u.checked=!1)}typeof updateGeminiFlexUi=="function"&&updateGeminiFlexUi(
get("model-select")?get("model-select").value:""),typeof schedulePromptTokenEstimate=="function"&&schedulePromptTokenEstimate(
!0)});const n=get("enable-gemini-flex");n&&n.addEventListener("change",()=>{const u=get("gemini-flex\
-container");if(u&&(u.classList.toggle("ring-1",n.checked),u.classList.toggle("ring-lime-300",n.checked)),
n.checked){const f=get("enable-gemini-free-tier");f&&f.checked&&(f.checked=!1,showToast("Flex\u306F\u6709\u6599\u67A0\u306E\u307F\u306E\
\u305F\u3081\u3001\u7121\u6599\u67A0\u3092\u89E3\u9664\u3057\u307E\u3057\u305F","info",!1));const h=get(
"enable-batch-mode");h&&h.checked&&(h.checked=!1),typeof updateBatchUi=="function"&&updateBatchUi(get(
"model-select")?get("model-select").value:"")}});const i=get("enable-batch-mode");i&&(i.onchange=()=>{
if(i.checked){browserFastModeEnabled&&setBrowserFastModeEnabled(!1);const u=get("enable-coding-mode");
u&&u.checked&&(u.checked=!1,typeof syncCodingModeUi=="function"&&syncCodingModeUi(!1),showToast("Bat\
ch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const a=get("model-\
select");a&&a.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const u=String(
a.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!u.startsWith("gemini-")||/(image|native-audio|tts|live|flash-cyber)/.test(u)?(setBrowserFastModeEnabled(
!1),a.dispatchEvent(new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const o=get("browser-fast-mode-enable-btn");
o&&(o.onclick=async()=>{const u=o.innerHTML;o.disabled=!0,o.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const f=get("browser-fast-mode-ignore-warning");if(f&&f.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(f){showToast(f.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{o.disabled=!1,o.innerHTML=u}});const l=get("browser-fast-mode-cancel-btn");l&&(l.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const c=document.getElementById(
"alpha-bar");setTimeout(()=>{if(c){const u=document.getElementById("version-display");if(u){const f=c.
getBoundingClientRect(),h=u.getBoundingClientRect(),y=h.left+h.width/2-(f.left+f.width/2),_=h.top+h.
height/2-(f.top+f.height/2);c.style.transform=`translate(${y}px, ${_}px) scale(0.1)`,c.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),c.remove()},800)}else c.style.opacity="0",setTimeout(()=>c.remove(),1e3)}},3e3);function d(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const h=(get("model-select").value||"").toLowerCase().includes("gpt-image-2.6"),y=get("gpt-im\
age-format");get("gpt-image-complexity-wrap")&&get("gpt-image-complexity-wrap").classList.toggle("hi\
dden",!h),y&&(Array.from(y.options).forEach(M=>{M.hasAttribute("data-gpt-image-26")&&(M.hidden=!h)}),
!h&&y.value==="gif"&&(y.value="jpeg"));const _=h&&y&&y.value==="gif",S=get("gpt-image-compression-wr\
ap");y&&S&&(y.value==="png"||y.value==="gif"?S.classList.add("hidden"):S.classList.remove("hidden")),
get("gpt-image-gif-wrap")&&get("gpt-image-gif-wrap").classList.toggle("hidden",!_)}r(d,"updateGptIma\
geUi");function m(){const u=get("gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.
remove("hidden"):u.classList.add("hidden");const h=(get("model-select").value||"").toLowerCase().includes(
"gemini-3.1-flash-lite-image");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(y=>{
y&&(Array.from(y.options).forEach(_=>{_.value!=="1K"&&(_.disabled=h)}),h&&y.value!=="1K"&&(y.value="\
1K"))})}r(m,"updateGeminiImageUi");function g(){const u=get("grok-image-options");if(!u)return;const f=(get(
"model-select").value||"").toLowerCase(),h=isGrokImageModel(),y=f==="grok-imagine-image-quality"||f===
"grok-imagine-image-2.0",_=f==="grok-imagine-image-2.0";if(h){u.classList.remove("hidden");const M=get(
"grok-image-resolution")?get("grok-image-resolution").parentElement:null;M&&M.classList.toggle("hidd\
en",!y);const P=get("grok-image-quality")?get("grok-image-quality").parentElement:null;P&&P.classList.
toggle("hidden",!_)}else u.classList.add("hidden");if(get("modal-grok-image-options")){const M=get("\
modal-grok-image-resolution")?get("modal-grok-image-resolution").parentElement:null;M&&M.classList.toggle(
"hidden",!y);const P=get("modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;
P&&P.classList.toggle("hidden",!_)}}r(g,"updateGrokImageUi");function b(){var q,W;const u=get("ideog\
ram-image-options"),f=get("modal-ideogram-image-options"),h=get("model-select")&&get("model-select").
value||"",y=isIdeogramModel(h),_=ideogramModelTraits(h),S={resolution:_.resolution,quality:_.quality,
speed:_.speed,style:_.style,negative:_.negative};if(u&&u.classList.toggle("hidden",!y),f&&f.classList.
toggle("hidden",!y),updateIdeogramPromptInputUi(),!y)return;const M=((q=get("ideogram-image-operatio\
n"))==null?void 0:q.value)==="remove_background",P=((W=get("ideogram-image-operation"))==null?void 0:
W.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(ie=>{const re=ie==="operation"||P&&ie==="se\
ed"||!(M||P)&&S[ie]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(pe=>{const Z=get(pe+ie);
!Z||!Z.parentElement||Z.parentElement.classList.toggle("hidden",!re)})}),["ideogram-image-style","mo\
dal-ideogram-image-style"].forEach(ie=>{const re=get(ie);if(!re)return;const pe=_.styleOptions.join(
"|");if(re.dataset.options!==pe){const Z=re.value;re.innerHTML=_.styleOptions.map(Se=>`<option value\
="${Se}">${Se==="auto"?"Auto":Se}</option>`).join(""),re.dataset.options=pe,re.value=_.styleOptions.
includes(Z)?Z:"auto"}});const O=get("ideogram-image-note");O&&(O.textContent=M?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\
\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
P?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
_.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
r(b,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(u=>{
var f;(f=get(u))==null||f.addEventListener("change",()=>{const h=get(u.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");h&&(h.value=get(u).value),b(),E()})});function v(){
var y;const u=get("grok-video-options");if(!u)return;const f=String(((y=get("model-select"))==null?void 0:
y.value)||"").toLowerCase();isGrokVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");
const h=get("grok-video-resolution");if(h){const _=Array.from(h.options).find(M=>M.value==="1080p"),
S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(f);_&&(_.disabled=!S),!S&&h.value===
"1080p"&&(h.value="720p")}}r(v,"updateGrokVideoUi");function w(){var _;const u=get("gemini-video-opt\
ions");if(!u)return;const f=String(((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase();
isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");const h=get("gemini-vide\
o-resolution");if(h){const S=Array.from(h.options).find(P=>P.value==="4K"),M=f==="veo-3.1-lite-gener\
ate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-flash";S&&(S.disabled=M),M&&h.value===
"4K"&&(h.value="1080p")}const y=get("gemini-video-duration-wrap");y&&y.classList.toggle("hidden",f===
"gemini-omni-1.1-flash")}r(w,"updateGeminiVideoUi");function x(){const u=get("gemini-music-options");
if(!u)return;const f=isGeminiRealtimeMusicModel(),h=isGeminiMusicModel()&&!f;u.classList.toggle("hid\
den",!h);const y=get("lyria-realtime-studio-bar");y&&y.classList.toggle("hidden",!f)}r(x,"updateGemi\
niMusicUi");function k(){var M;const u=get("xai-chat-options");if(!u)return;const f=String(((M=get("\
model-select"))==null?void 0:M.value)||"").toLowerCase(),h=f.startsWith("grok-")&&!isGrokImageModel(
f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!h);const y=get("xai-logp\
robs"),_=get("xai-top-logprobs"),S=f.includes("grok-4.20");y&&(y.disabled=S,S&&(y.checked=!1)),_&&(_.
disabled=S,S&&(_.value=""))}r(k,"updateXaiChatUi");function C(){const u=isMistralOcrModel(),f=get("m\
istral-ocr-options");f&&f.classList.toggle("hidden",!u);const h=get("modal-mistral-ocr-options");h&&
h.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(y=>{const _=get(y);_&&(_.classList.toggle("opacity-50",u),_.classList.toggle("po\
inter-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(C,"updateMistralOcrUi");function T(){const u=get("image-input-limit\
s");if(!u)return;const f=(get("model-select").value||"").toLowerCase();let h="",y=!1;f.includes("gpt\
-image")?(y=!0,h=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):f==="deepseek-v4.1-flash"||f==="deepseek-v4-flash-vision-exp"?(y=!0,h=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):f.includes("deepseek")||(isGeminiImageModelKey(f)?(y=!0,f.includes("gemini-3.1-flash-lite-\
image")?h=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):f==="gemini-nano-banana-2.1"?h=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2.1 \
\u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u30FB2K\u30FB4K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u52D5\u753B\u3092\u53C2\u8003\u306B\u3057\u305F\u753B\u50CF\u751F\u6210\u306B\u3082\u5BFE\u5FDC\u3057\u307E\u3059</div>"].
join(""):f.includes("gemini-3.1-flash-image")?h=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):f.includes("gemini-2.5")&&f.includes("image")?h=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):h=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(f)?(y=!0,h=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):f.includes("grok")?(y=!0,h=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):isIdeogramModel(f)?(y=!0,h=ideogramModelTraits(f).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u3082\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.5 \u306E\u307F\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u306F\u5168\u30E2\u30C7\u30EB\u3067\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join("")):f.includes("grok")&&f.includes("video")&&(y=!0,h=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),y?(u.innerHTML=h,u.classList.remove("hidden")):(u.classList.add("hidden"),u.innerHTML="")}
r(T,"updateImageInputLimits");function E(){const u=get("enable-sys-prompt"),f=!!(u&&u.disabled),h=!!(u&&
u.checked);U(),u&&(u.disabled?!f&&h?u.dataset.restoreChecked="1":f||delete u.dataset.restoreChecked:
(f&&u.dataset.restoreChecked==="1"&&(u.checked=!0),delete u.dataset.restoreChecked))}r(E,"toggleOpti\
ons"),window.toggleOptions=E;function N(u){const f=get("thinking-options"),h=get("thinking-level"),y=get(
"reasoning-effort"),_=get("enable-thinking"),S=get("thinking-budget-label"),M=get("thinking-budget"),
P=u==="effort";if(f&&(f.dataset.thinkingMode=P?"effort":"thinking"),h&&h.classList.toggle("hidden",P),
y&&y.classList.toggle("hidden",!P),S&&S.classList.toggle("hidden",P),M&&M.classList.toggle("hidden",
P),!!_)if(P){_.dataset.thinkingChecked===void 0&&(_.dataset.thinkingChecked=_.checked?"1":"0");const O=y?
Array.from(y.options).find(W=>W.value==="none"):null,q=!!(O&&!O.disabled&&!O.classList.contains("hid\
den"));_.checked=q?!!(y&&String(y.value||"").toLowerCase()!=="none"):!0,_.disabled=!q}else _.dataset.
thinkingChecked!==void 0&&(_.checked=_.dataset.thinkingChecked==="1",delete _.dataset.thinkingChecked)}
r(N,"setComposerThinkingMode");function U(){const u=get("model-select");if(!u)return;const f=u.value,
h=String(f||"").toLowerCase(),y=h.includes("deepseek"),_=h.startsWith("glm-"),S=get("thinking-option\
s"),M=get("reasoning-effort-container"),P=get("enable-thinking"),O=get("thinking-level"),q=get("thin\
king-budget"),W=get("enable-search"),ie=get("search-container"),re=get("url-context-container"),pe=get(
"enable-maps"),Z=get("maps-grounding-container"),Se=get("enable-sys-prompt"),oe=get("sys-prompt-opti\
on"),Oe=get("enable-python"),le=get("python-container"),Ce=get("prompt-cache-container"),$e=get("ena\
ble-prompt-cache"),fe=f==="gpt-5-search-api",ke=f.includes("tts"),Ne=isMistralOcrModel(f),Re=h.includes(
"gemini-3.1-flash-lite-image"),ft=h==="gemini-nano-banana-2.1",et=h.includes("gemini-3.1-flash-image")&&
!Re||ft,St=isClaudeModelKey(f),Ft=h==="gemini-3.8-flash-cyber",Ae=isLlmModel()&&!y&&!ke&&!h.includes(
"realtime")&&!h.includes("native-audio")&&!h.includes("live");Ce&&(Ae?(Ce.classList.remove("hidden",
"opacity-50","pointer-events-none"),$e&&($e.disabled=!1)):($e&&($e.checked=!1,$e.disabled=!0),Ce.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),M&&M.classList.
add("hidden"),N("thinking");const Te=get("vision-model-info");if(Te&&Te.classList.add("hidden"),M){const Le=get(
"reasoning-effort");if(Le){Array.from(Le.options).forEach(De=>{const Je=h==="gpt-5.6"||h.startsWith(
"gpt-5.6-"),Qt=h.startsWith("gpt-6"),Sn=h==="gpt-6-sol"||h==="gpt-6-luna"||h==="gpt-6.1-terra",On=h===
"deepseek-v4.1-flash"||h==="deepseek-v4-flash-0731"||h==="deepseek-v4-flash"||h==="deepseek-v4-flash\
-vision-exp",Cn=h==="deepseek-v4-pro",K=h.includes("grok-4.5"),se=h.includes("grok-4.6");De.value===
"max"?De.classList.toggle("hidden",!Je&&!Qt&&!On&&!Cn):De.value==="xhigh"?De.classList.toggle("hidde\
n",!se&&!h.includes("multi-agent")&&!Je&&!Qt):De.value==="medium"?De.classList.toggle("hidden",!(h.includes(
"grok-4.3")||K||se||h.includes("grok-4.20-0309-reasoning")||h.includes("grok-build")||h.includes("mu\
lti-agent")||h.includes("gpt-5")||Qt||h.includes("o1")||h.includes("o3"))):De.value==="none"?De.classList.
toggle("hidden",!h.includes("grok-4.3")&&!h.includes("grok-build")&&!h.includes("gpt-5")&&!Sn&&!On&&
!Cn):De.value==="low"&&De.classList.toggle("hidden",Cn)});const qe=Le.selectedOptions&&Le.selectedOptions[0];
qe&&qe.classList.contains("hidden")&&(Le.value=y?"high":"medium")}}re&&re.classList.add("hidden"),Z&&
Z.classList.add("hidden"),P&&(P.disabled=!1),q&&(q.disabled=!0,q.classList.add("opacity-50"));const We=isGeminiImageModelKey(
f);if(ke||Ne)ie&&(get("enable-search").checked=!1,ie.classList.add("opacity-50","pointer-events-none")),
re&&(get("enable-url-context").checked=!1,re.classList.add("opacity-50","pointer-events-none")),Z&&pe&&
(pe.checked=!1,Z.classList.add("opacity-50","pointer-events-none")),le&&(Oe.checked=!1,le.classList.
add("opacity-50","pointer-events-none")),Se&&oe&&(Se.checked=!1,Se.disabled=!0,oe.classList.add("opa\
city-50"));else if(et||Re)Z&&pe&&(pe.checked=!1,Z.classList.add("hidden","opacity-50","pointer-event\
s-none")),S.classList.remove("hidden"),ft?(Array.from(O.options).forEach(Le=>{Le.disabled=Le.value===
"low"}),["minimal","medium","high"].includes(O.value)||(O.value="medium")):(Array.from(O.options).forEach(
Le=>{["low","medium"].includes(Le.value)&&(Le.disabled=!0),["minimal","high"].includes(Le.value)&&(Le.
disabled=!1)}),["minimal","high"].includes(O.value)||(O.value=Re?"minimal":"high")),P&&(P.disabled=!1),
Re&&(W&&(W.checked=!1,W.disabled=!0),ie&&ie.classList.add("opacity-50","pointer-events-none"));else if(We)
Z&&pe&&(pe.checked=!1,Z.classList.add("hidden","opacity-50","pointer-events-none"));else if(St)S.classList.
remove("hidden"),q&&(q.disabled=!1,q.classList.remove("opacity-50")),Array.from(O.options).forEach(Le=>{
Le.disabled=!0}),le&&(Oe.checked=!1,le.classList.add("opacity-50","pointer-events-none"));else if(Ft){
S&&S.classList.remove("hidden"),P&&(P.checked=!0,P.disabled=!0),Array.from(O.options).forEach(qe=>{qe.
disabled=!["low","medium","high"].includes(qe.value)}),["low","medium","high"].includes(O.value)||(O.
value="medium"),[ie,re,Z,le].forEach(qe=>{qe&&qe.classList.add("opacity-50","pointer-events-none")}),
[W,pe,Oe].forEach(qe=>{qe&&(qe.checked=!1,qe.disabled=!0)});const Le=get("enable-url-context");Le&&(Le.
checked=!1,Le.disabled=!0),Se&&oe&&(Se.disabled=!1,oe.classList.remove("opacity-50"))}else if(f.includes(
"gemini")&&!We){S.classList.remove("hidden"),re&&re.classList.remove("hidden","opacity-50","pointer-\
events-none");const Le=f.includes("gemini-3")||f.includes("gemini-4");Z&&(Le?Z.classList.remove("hid\
den","opacity-50","pointer-events-none"):(pe&&(pe.checked=!1),Z.classList.add("hidden","opacity-50",
"pointer-events-none")));const qe=f.includes("flash"),De=f==="gemini-4-hydrogen";Array.from(O.options).
forEach(Je=>{De?Je.disabled=!["low","medium","high","xhigh"].includes(Je.value):f==="gemini-4-carbon"||
f==="gemini-4-argon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?Je.disabled=!["low","medium","h\
igh"].includes(Je.value):f==="gemini-3.6-flash"?Je.disabled=!["medium","high"].includes(Je.value):f===
"gemini-3.5-flash-lite"?Je.disabled=!["minimal","medium","high"].includes(Je.value):Je.value==="xhig\
h"?Je.disabled=!0:["minimal","medium"].includes(Je.value)?Je.disabled=!qe:Je.disabled=!1}),De&&!["lo\
w","medium","high","xhigh"].includes(O.value)?O.value="high":(f==="gemini-4-carbon"||f==="gemini-4-a\
rgon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium","high"].includes(O.value)||
f==="gemini-3.6-flash"&&!["medium","high"].includes(O.value)?O.value="medium":f==="gemini-3.5-flash-\
lite"&&!["minimal","medium","high"].includes(O.value)?O.value="minimal":!qe&&["minimal","medium"].includes(
O.value)&&(O.value="high"),Le?P&&(P.checked=!0,P.disabled=!0):P&&(P.disabled=!1),q&&f.includes("gemi\
ni-2.5")&&(q.disabled=!1,q.classList.remove("opacity-50")),q&&!f.includes("gemini-2.5")&&(q.disabled=
!0,q.classList.add("opacity-50"))}if(isLlmModel()&&(h.includes("gpt-5")||h.includes("o1")||h.includes(
"o3")||h.includes("grok-4.3")||h.includes("grok-4.5")||h.includes("grok-4.6")||h.includes("grok-4.20\
-0309-reasoning")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt")&&!h.includes(
"tts")))S&&S.classList.remove("hidden"),N("effort"),ie&&ie.classList.remove("opacity-50","pointer-ev\
ents-none");else if(y){S&&S.classList.remove("hidden"),N("effort");const Le=get("vision-model-info");
if(Le&&Le.classList.toggle("hidden",h==="deepseek-v4.1-flash"||h==="deepseek-v4-flash-vision-exp"),W&&
(W.disabled=!1),ie&&ie.classList.remove("opacity-50","pointer-events-none"),re){const qe=get("enable\
-url-context");qe&&(qe.checked=!1),re.classList.add("opacity-50","pointer-events-none")}Z&&pe&&(pe.checked=
!1,Z.classList.add("opacity-50","pointer-events-none"))}else Ne||(ie&&ie.classList.remove("opacity-5\
0","pointer-events-none"),Z&&pe&&(pe.checked=!1,Z.classList.add("hidden","opacity-50","pointer-event\
s-none")));if(ke?le&&le.classList.add("opacity-50","pointer-events-none"):(le&&le.classList.remove("\
opacity-50","pointer-events-none"),(!We||et)&&!f.includes("gpt-image")&&(Se.disabled=!1,oe.classList.
remove("opacity-50"))),(We&&!et||f.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(f)||isGrokVideoModel()||
Ne)&&Se&&oe&&(Se.checked=!1,Se.disabled=!0,oe.classList.add("opacity-50")),le&&(isLlmModel()?(le.classList.
remove("hidden"),Oe.disabled=!1):(Oe.checked=!1,Oe.disabled=!0,le.classList.add("hidden"))),fe?(W&&(W.
checked=!0,W.disabled=!0),ie&&ie.classList.add("opacity-50","pointer-events-none"),le&&(Oe.checked=!1,
Oe.disabled=!0,le.classList.add("opacity-50","pointer-events-none"))):W&&!f.includes("tts")&&!Ne&&!Re&&
(W.disabled=!1),Ft){[W,pe,Oe].forEach(qe=>{qe&&(qe.checked=!1,qe.disabled=!0)});const Le=get("enable\
-url-context");Le&&(Le.checked=!1,Le.disabled=!0),[ie,re,Z,le].forEach(qe=>{qe&&qe.classList.add("op\
acity-50","pointer-events-none")})}_&&(Te&&Te.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-fl\
ashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(h)),[W,Oe].forEach(Le=>{Le&&
(Le.checked=!1,Le.disabled=!0)}),[ie,le].forEach(Le=>{Le&&Le.classList.add("opacity-50","pointer-eve\
nts-none")}));const gt=get("mask-btn");gt&&(isGptImageModel()||isIdeogramObjectRemoval()?gt.classList.
remove("hidden"):(gt.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),
updateStsUi(),updateStsOptions(),d(),m(),g(),b(),v(),w(),x(),updateBatchUi(f);const Dt=get("gemini-f\
ree-tier-container");if(Dt){const Le=typeof isGeminiApiModel=="function"&&isGeminiApiModel(f)&&(typeof currentGeminiBackend!=
"function"||currentGeminiBackend()!=="vertex_ai");Dt.classList.toggle("hidden",!Le)}k(),C(),T(),purgeUnsupportedAttachments(
!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}r(U,"toggleOptionsForModel"),get("model-sel\
ect")&&(get("model-select").addEventListener("change",E),get("model-select").addEventListener("chang\
e",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),E(),minimalPromptMode?setMinimalPromptMode(
!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();const L=get("enable-canvas-\
mode");L&&(L.checked=canvasModeEnabled,L.addEventListener("change",()=>syncCanvasModeUi(L.checked))),
syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const I=get("enable-coding-mode");I&&(I.
checked=codingModeEnabled,I.addEventListener("change",()=>syncCodingModeUi(I.checked))),get("clear-c\
oding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",u=>{const f=u.
target.closest("[data-canvas-block-index]");if(!f)return;const h=Number(f.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(h,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",u=>{if(u.target.value===
"")return;const f=Number(u.target.value);Number.isInteger(f)&&applyCanvasSelection(f,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",u=>{const f=u.target.closest(
"[data-canvas-panel-view]");if(!f)return;const h=f.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
h,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const u=getCanvasModeElements(),f=u&&u.code&&u.code.textContent||"";if(!f.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
f,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const A=get(
"prompt-controls-toggle-btn");A&&(A.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>d()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>m()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,h){const y=(f||"").match(/\S*/)[0];if(y==="py\
exec")return"";if(y==="chat_error")return buildChatErrorBubbleHtml(u||"");const _=u||"",S=(y||"").toLowerCase();
if(S==="mathsvg"){const oe=getRenderableSvgCode("mathsvg",_);return oe&&typeof window.buildMathSvgRenderHtml==
"function"?window.buildMathSvgRenderHtml(oe):""}let M="";try{const oe=hljs.getLanguage(y)?y:"plainte\
xt";activeStreamingBubbleId&&_.length>2e4?M=escapeHtml(_):M=hljs.highlight(_,{language:oe}).value}catch{
M=escapeHtml(_)}const P=encodeURIComponent(_).replace(/'/g,"%27"),O=detectBlockedScriptsInCode(_),q=hashString(
`${y||"TEXT"}
${_||""}`);let W="";if(canvasModeEnabled){const oe=String(canvasPreviewState.selectedKey||"")===q,Oe=oe?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";W=`<button\
 class="canvas-preview-btn${oe?" canvas-active":""}" data-code="${P}" data-code-key="${q}" data-canv\
as-lang="${escapeHtml(y||"txt")}" title="${Oe}" aria-label="${Oe}" aria-pressed="${oe?"true":"false"}\
"><i class="fas ${oe?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,_)){const oe=O?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
W=`<button class="html-preview-btn" data-code="${P}" ${O?'data-suspicious="1"':""} title="${oe}" ari\
a-label="${oe}"><i class="fas ${O?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const ie=`\
<button class="download-btn" data-code="${P}" data-lang="${y||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,re=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${P}" data-code-key="${q}" data-coding-lang="${escapeHtml(y||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,pe=(y||"TEXT")+(O?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),Z=getRenderableSvgCode(S,_);return`${Z?buildSvgCodeRenderHtml(Z,q):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${q}"><div class="code-header"\
><span class="code-lang">${pe}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${re}${W}${ie}\
<button class="copy-btn" data-code="${P}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${y}">${M}</code></pre></d\
iv></div>`},link(u,f,h){return`<a href="${u}" title="${f||""}" target="_blank">${h}</a>`},image(u,f,h){
return buildChatImageHtml(u,{alt:h,title:f})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(bi=get("version-update-dismis\
s"))==null||bi.addEventListener("click",()=>{const u=localStorage.getItem("app_version")||"";u&&localStorage.
setItem("version_notified",u),hideModal("version-update-modal")});const j=get("version-update-clear-\
cache");if(j&&(j.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),j.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(j.checked)})),
(yi=get("version-update-reload"))==null||yi.addEventListener("click",async()=>{var f;await versionUpdateCachePreferenceSavePromise.
catch(()=>{}),!!((f=get("version-update-clear-cache"))!=null&&f.checked)?await clearSiteCacheAndReload(
get("version-update-reload"),{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.
setVersionChangeHandler(u=>{u&&u!==appVersion&&(localStorage.getItem("version_notified")||"")!==u&&(localStorage.
setItem("app_version",u),purgeCaches().then(()=>checkAndNotifyVersion(u)))}),window.ConnectionMonitor.
start(),window.addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener(
"offline",()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offli\
ne")}),window.addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener(
"visibilitychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener(
"pagehide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&
botConfig.lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(u){console.error(u)}try{runBotDetectionGate()}catch(u){console.error(
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const B=r(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),D=r(u=>{const f=Array.isArray(u)?u:[],h=get("passkey-list"),y=get("passkey-count");if(y&&(y.
innerText=String(f.length)),!!h){if(!f.length){h.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
h.innerHTML="",f.forEach((_,S)=>{const M=_&&_.id?String(_.id):"",P=document.createElement("div");P.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const O=document.
createElement("div");O.className="min-w-0";const q=document.createElement("div");q.className="text-x\
s text-gray-200 truncate",q.innerText=_&&_.name?String(_.name):`Security Key ${S+1}`;const W=document.
createElement("div");W.className="text-[10px] text-gray-500 mt-1",W.innerText=_&&_.created_at?`\u767B\u9332\u65E5\u6642:\
 ${B(_.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",O.appendChild(q),O.appendChild(W),P.appendChild(
O);const ie=document.createElement("button");ie.type="button",ie.className="bg-red-700 hover:bg-red-\
600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",ie.innerText="\u524A\u9664",
ie.disabled=!M,M&&(ie.onclick=()=>window.removeWebAuthnCredential(M)),P.appendChild(ie),h.appendChild(
P)})}},"renderPasskeyList"),V=r(u=>{const f=get("session-list");if(f){if(!u||!u.length){f.innerHTML=
'<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(h=>{const y=h.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",_=h.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!h.is_current&&!h.is_revoked?`<button \
data-session-id="${escapeHtml(h.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
M=(h.user_agent||"Unknown").slice(0,120),P=h.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${y}${_}<div class="text-xs text-gray-200">${escapeHtml(
P)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(M)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(B(h.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(B(h.created_at))}</div></div>${S}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(h=>{h.onclick=async()=>{const y=h.getAttribute("data-session-id");if(!y||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const _=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:y})});let S={};try{S=await _.json()}catch{}if(_.ok){if(S.logged_out){
location.href="/login";return}await ee()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),ee=r(async()=>{const u=get("session-list");u&&(u.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api\
/sessions");let h={};try{h=await f.json()}catch{}if(!f.ok){if(h&&h.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const y=(h.sessions||[]).filter(_=>!_.is_revoked);V(y)},"loadSessions"),$=r(()=>{const u=get(
"session-refresh-btn");u&&(u.onclick=()=>ee());const f=get("session-revoke-others-btn");f&&(f.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await ee():showToast("\u64CD\u4F5C\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0)});const h=get("session-revoke-all-btn");h&&(h.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(u=>{u&&(currentVisionModel=u.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
u);try{loadMcpServers()}catch{}u&&u.theme_color&&applyThemeColor(u.theme_color,!0),u&&Object.prototype.
hasOwnProperty.call(u,"minimal_prompt_mode")&&u.minimal_prompt_mode?setMinimalPromptMode(!0):u&&Object.
prototype.hasOwnProperty.call(u,"compact_prompt_mode")&&setCompactPromptMode(!!u.compact_prompt_mode),
get("set-server-diagnostic-log-retention-hours")&&(get("set-server-diagnostic-log-retention-hours").
value=u.server_diagnostic_log_retention_hours||1),get("set-server-diagnostic-log")&&(get("set-server\
-diagnostic-log").checked=u.enable_server_diagnostic_log===!0),get("set-client-debug-log")&&syncClientDebugLogToggle(
u.enable_client_debug_log===!0,"settings sync");const f=get("enable-sys-prompt");f&&u&&u.system_prompt&&
String(u.system_prompt).trim()&&(!f.disabled&&!u.default_enable_system_prompt&&!u.use_last_chat_settings&&
(f.checked=!0),E())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadSkills(),typeof window.loadProjects=="function"&&window.loadProjects(),
get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},get("new-chat-btn").onclick=
()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();const R=get(
"vision-model-change-btn");R&&(R.onclick=()=>_openVisionModelSelector());const J=get("compression-fo\
rmat-only");J&&(J.onchange=()=>{const u=J.checked,f=get("compression-max-size"),h=get("compression-m\
ax-dim");f&&(f.disabled=u),h&&(h.disabled=u);const y=get("compression-size-wrap"),_=get("compression\
-dim-wrap");y&&(y.style.opacity=u?"0.4":"1"),_&&(_.style.opacity=u?"0.4":"1")});const F=r(()=>{const u=get(
"enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=!!temporaryChatEnabled,
u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(u.checked)||(setTemporaryChatUiState(
f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggle");F(),document.addEventListener("vi\
sibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(!0)}),window.
addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener("beforeunlo\
ad",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const H=get("storage-usage-refresh");
H&&(H.onclick=()=>loadStorageUsage());let te=null;const G=r(()=>{const u=new Uint8Array(16);return window.
crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},"createAccountTr\
ansferId"),X=r((u={})=>{const f=get("account-transfer-progress"),h=get("account-transfer-progress-ba\
r"),y=get("account-transfer-progress-percent"),_=get("account-transfer-progress-text"),S=get("accoun\
t-transfer-progress-detail"),M=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&f.classList.remove(
"hidden"),h&&(h.style.width=`${M}%`),y&&(y.textContent=`${Math.round(M)}%`),_&&(_.textContent=u.message||
"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const O={queued:"\u9806\u756A\u5F85\u3061",
preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\u30D5\u30A1\u30A4\u30EB\u3092\u66F8\u304D\u51FA\u3057\u4E2D",
finalizing:"\u6700\u7D42\u51E6\u7406\u4E2D",ready:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u6E96\u5099\u5B8C\u4E86",
downloading:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D",uploading:"ZIP\u3092\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D",
validating:"ZIP\u3092\u691C\u8A3C\u4E2D",validating_files:"\u30D5\u30A1\u30A4\u30EB\u60C5\u5831\u3092\u691C\u8A3C\u4E2D",
reading_files:"\u30D5\u30A1\u30A4\u30EB\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D",importing_settings:"\u8A2D\u5B9A\u3092\
\u53CD\u6620\u4E2D",importing_credentials:"\u8A8D\u8A3C\u60C5\u5831\u3092\u53CD\u6620\u4E2D",importing_skills:"\
Skill\u3092\u8FFD\u52A0\u4E2D",saving_files:"\u30D5\u30A1\u30A4\u30EB\u3092\u4FDD\u5B58\u4E2D",importing_chats:"\
\u30C1\u30E3\u30C3\u30C8\u5C65\u6B74\u3092\u8FFD\u52A0\u4E2D",importing_feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u8FFD\u52A0\u4E2D",
importing_diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF\u3092\u8FFD\u52A0\u4E2D",cancelling:"\u30AD\u30E3\u30F3\u30BB\u30EB\u51E6\u7406\u4E2D",
cancelled:"\u30AD\u30E3\u30F3\u30BB\u30EB\u6E08\u307F",expired:"\u4FDD\u5B58\u671F\u9650\u5207\u308C",
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=O[u.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const P=get("account-transfer-cancel-btn");P&&P.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),ce=r(u=>{ge&&(ge.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const h=get("account-transfer-cancel-btn");
h&&(h.disabled=!u)},"setAccountTransferControls"),de=r((u={})=>{const f=get("account-export-ready"),
h=get("account-export-ready-text"),y=get("account-export-expiry"),_=get("account-export-download-btn"),
S=!!(u.available&&u.download_url);if(f&&f.classList.toggle("hidden",!S),!S){_&&_.removeAttribute("hr\
ef");return}const M=Math.max(0,Number(u.size_bytes)||0),P=M>=1024*1024*1024?`${(M/(1024*1024*1024)).
toFixed(2)} GB`:`${(M/(1024*1024)).toFixed(1)} MB`;if(h){const O=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
h.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${P}${O}`}
if(y){const O=u.expires_at?new Date(u.expires_at):null;y.textContent=O&&!Number.isNaN(O.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${O.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
_&&(_.href=u.download_url)},"renderAccountExportAvailability"),Q=r(async u=>{for(;te===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),h=await f.json().catch(()=>({}));if(f.ok&&(h.state!=="pending"&&X(h),["ready","completed","f\
ailed","cancelled","expired"].includes(h.state)))return h}catch{}await new Promise(f=>setTimeout(f,700))}
return null},"pollAccountTransfer"),me=r((u,f,h=!0)=>{f&&(X(f),de(f),h&&f.state==="ready"?showToast(
f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):h&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),ve(u))},"handleFinishedAccountExport"),ne=r(async()=>{try{const u=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(de(f),f.state==="ready"){X(f);return}if(["failed","cancelled","expired"].includes(
f.state)){X(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||te&&te.id===
f.job_id||te)return;const h={id:f.job_id,type:"export",stopped:!1,restored:!0};te=h,ce(!0),X(f);const y=await Q(
h);y&&me(h,y,!0)}catch{}},"refreshLatestAccountExport"),ve=r(u=>{te===u&&(te=null),u.stopped=!0,ce(!1)},
"finishAccountTransfer"),ue=get("account-transfer-cancel-btn");ue&&(ue.onclick=async()=>{const u=te;
if(!(!u||u.stopped)){u.cancelRequested=!0,ue.disabled=!0,X({progress:0,phase:"cancelling",message:"\u30AD\
\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/transfer/${u.
id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}u.controller&&u.controller.abort(),
X({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),u.
type==="export"&&de({available:!1}),ve(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const ge=get("account-export-btn");ge&&(ge.onclick=async()=>{if(te)return;const u={id:G(),
type:"export",stopped:!1};te=u,ce(!0),de({available:!1}),X({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:u.id}),keepalive:!0})),h=await f.json().catch(()=>({}));if(f.status===409&&
h.error==="export_in_progress"&&h.job_id)u.id=h.job_id;else if(!f.ok)throw new Error(h.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
h.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
X({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const y=await Q(u);!u.cancelRequested&&y&&me(u,y,!0)}catch(f){const h=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";X({progress:0,phase:"failed",message:h}),
showToast(h,"error",!0),ve(u)}});const je=get("account-export-download-btn");je&&je.addEventListener(
"click",async u=>{const f=je.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();try{const h=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),y=await h.json().catch(
()=>({}));h.ok&&y.available&&y.download_url?(je.href=y.download_url,window.location.assign(y.download_url)):
(de(y),X(y),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),ne())}catch{window.location.assign(f)}}}),ce(!1),ne();const Me=get("import-files-grid"),
lt=get("import-files-info"),he=get("import-files-summary"),we=r(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let be=null;const ae=r(()=>{
if(!be)return;const u=be.files,f=be.selection;let h=0;u.forEach(S=>{f.has(S.archive_path)&&(h+=Number(
S.size_bytes)||0)});const y=Number(be.available_bytes)||0,_=h>y;he&&(he.textContent=`\u9078\u629E\u4E2D: ${we(
h)} / \u5229\u7528\u53EF\u80FD: ${we(y)}${_?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,he.classList.
toggle("text-red-300",_)),lt&&(lt.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),it=r(
()=>{if(!Me||!be)return;Me.innerHTML="";const u=be.files;if(!u.length){Me.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
ae();return}u.forEach(f=>{const h=document.createElement("label"),y=be.selection.has(f.archive_path);
h.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${y?"border-blue-500":"border-gray-600"}`,h.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${y?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${we(f.size_bytes)}</div></div>`;const _=h.
querySelector(".import-file-check");_.addEventListener("change",()=>{_.checked?be.selection.add(f.archive_path):
be.selection.delete(f.archive_path),h.classList.toggle("border-blue-500",_.checked),h.classList.toggle(
"border-gray-600",!_.checked),ae()}),Me.appendChild(h)}),ae()},"renderImportFileItems"),st=r(u=>new Promise(
f=>{if(be={files:u.files||[],selection:new Set((u.files||[]).map(h=>h.archive_path)),available_bytes:u.
available_bytes,resolve:f},it(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),kt=r(u=>{if(hideModal("import-files-modal"),be){const f=be.resolve;
be=null,f(u)}},"closeImportFileSelection"),Lt=get("import-files-close");Lt&&(Lt.onclick=()=>kt(null));
const He=get("import-files-cancel");He&&(He.onclick=()=>kt(null));const at=get("import-files-confirm");
at&&(at.onclick=()=>{if(!be)return;const u=Array.from(be.selection);kt(u.length?u.join(","):"__none_\
_")});const Xe=get("import-files-select-all");Xe&&(Xe.onclick=()=>{be&&(be.files.forEach(u=>be.selection.
add(u.archive_path)),it())});const Mt=get("import-files-none");Mt&&(Mt.onclick=()=>{be&&(be.selection.
clear(),it())});const Ht={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
\u97F3\u58F0\u30B9\u30BF\u30B8\u30AAUI",math_steps_enabled:"\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u89E3\u8AAC",
math_steps_config:"\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u306E\u8A2D\u5B9A",default_math_steps:"\u65E2\u5B9A: \u6570\
\u5B66\u30B9\u30C6\u30C3\u30D7",temp_chat_timeout_seconds:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u306E\u6709\u52B9\u6642\u9593\uFF08\u79D2\uFF09",
default_model:"\u65E2\u5B9A\u306E\u30E2\u30C7\u30EB",default_enable_search:"\u65E2\u5B9A: Search",default_gemini_free_tier:"\
\u65E2\u5B9A: Gemini\u7121\u6599\u67A0",default_gemini_flex:"\u65E2\u5B9A: Gemini Flex",default_enable_url_context:"\
\u65E2\u5B9A: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",default_enable_maps:"\u65E2\u5B9A: Maps",default_enable_python:"\
\u65E2\u5B9A: Python",default_enable_file_creation:"\u65E2\u5B9A: File",default_enable_thinking:"\u65E2\u5B9A:\
 Thinking",default_thinking_level:"\u65E2\u5B9A: Thinking\u30EC\u30D9\u30EB",default_thinking_budget:"\
\u65E2\u5B9A: Thinking budget",default_reasoning_effort:"\u65E2\u5B9A: Reasoning effort",default_enable_system_prompt:"\
\u65E2\u5B9A: \u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",default_enable_mcp:"\u65E2\u5B9A: MCP",
default_safety_setting:"\u65E2\u5B9A: \u5B89\u5168\u8A2D\u5B9A",default_vision_model:"Vision Model",
rich_paste_prompt_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30D7\u30ED\u30F3\u30D7\u30C8",
rich_paste_prompt_use_custom_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30AB\u30B9\u30BF\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u65E2\u5B9A",
last_model:"\u76F4\u524D\u306E\u30E2\u30C7\u30EB",last_enable_search:"\u76F4\u524D: Search",last_gemini_free_tier:"\
\u76F4\u524D: Gemini\u7121\u6599\u67A0",last_gemini_flex:"\u76F4\u524D: Gemini Flex",last_enable_url_context:"\
\u76F4\u524D: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",last_enable_maps:"\u76F4\u524D: Maps",last_enable_python:"\
\u76F4\u524D: Python",last_enable_file_creation:"\u76F4\u524D: File",last_enable_thinking:"\u76F4\u524D: Think\
ing",last_thinking_level:"\u76F4\u524D: Thinking\u30EC\u30D9\u30EB",last_thinking_budget:"\u76F4\u524D: Thinki\
ng budget",last_reasoning_effort:"\u76F4\u524D: Reasoning effort",last_enable_system_prompt:"\u76F4\u524D: \u30B7\u30B9\u30C6\
\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",last_enable_mcp:"\u76F4\u524D: MCP",last_safety_setting:"\u76F4\u524D: \u5B89\
\u5168\u8A2D\u5B9A",last_math_steps:"\u76F4\u524D: \u6570\u5B66\u30B9\u30C6\u30C3\u30D7",enable_latency_metrics:"\
\u30EC\u30B9\u30DD\u30F3\u30B9\u901F\u5EA6\u306E\u8A08\u6E2C",enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},
At=r(u=>{if(u===!0)return"ON";if(u===!1)return"OFF";if(u==null||u==="")return"\u672A\u8A2D\u5B9A";const f=String(
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let Ge=null;const ht=r(
u=>{if(Ge){const f=Ge;Ge=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),ze=r(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}Ge=f;const y=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],_=get("settings-confirmation-list");_&&(y.length?
_.innerHTML=y.map(M=>{const P=Ht[M.field]||M.field,O=At(M.current),q=At(M.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(P)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
O)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(q)}</d\
iv>
                            </div>`}).join(""):_.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${y.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),_t=get("settings-confir\
mation-modal");_t&&_t.addEventListener("click",u=>{u.target===_t&&ht(!1)});const tt=get("settings-co\
nfirmation-close");tt&&(tt.onclick=()=>ht(!1));const bt=get("settings-confirmation-cancel");bt&&(bt.
onclick=()=>ht(!1));const rn=get("settings-confirmation-confirm");rn&&(rn.onclick=()=>ht(!0));const qt=get(
"account-import-btn"),Et=get("account-import-inplace"),ln=get("account-import-inplace-warning");if(Et&&
ln){const u=r(()=>ln.classList.toggle("hidden",!Et.checked),"syncInplaceWarn");Et.addEventListener("\
change",u),u()}qt&&(qt.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,h=get("account-import-categories"),y=h?Array.from(h.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(pe=>pe.value):[],_=get("account-import-inplace"),S=!!(_&&_.checked),M=get("account-impo\
rt-settings-bypass"),P=!!(M&&M.checked);let O=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!y.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const q=h?Array.from(h.querySelectorAll('input[type="checkbox"]:checked')).map(pe=>(pe.
closest("label")&&pe.closest("label").textContent||pe.value).trim()):y;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${q.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const W={id:G(),type:"import",stopped:!1,controller:new AbortController};
te=W,ce(!0),X({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const ie=get("account-import-result");let re=Promise.resolve(null);try{const Z=Math.max(1,Math.ceil(
f.size/10485760)),Se=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({
method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:W.
controller.signal})),oe=await Se.json().catch(()=>({}));if(!Se.ok)throw new Error(oe.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");W.uploadId=oe.upload_id;const Oe=oe.chunk_size||10485760;
let le=0,Ce=0;const $e=r(async()=>{for(;;){const Ae=Ce++;if(Ae>=Z)return;const Te=f.slice(Ae*Oe,Math.
min(f.size,(Ae+1)*Oe)),We=new FormData;We.append("chunk",Te,f.name),We.append("index",String(Ae));const ct=await apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"\
POST",body:We,signal:W.controller.signal})),gt=await ct.json().catch(()=>({}));if(!ct.ok)throw new Error(
gt.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");le++,X({
progress:Math.min(35,Math.round(le/Z*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${le}\
/${Z}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let fe=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),fe=!0);try{await Promise.
all([$e(),$e(),$e()]);const Ae=await apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:W.controller.signal})),Te=await Ae.json().
catch(()=>({}));if(!Ae.ok)throw new Error(Te.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
X({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
fe&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ke="",Ne=!1,Re=0;const ft=r(
async()=>{let Ae=!1;const Te=r(()=>{Ae||(Ae=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const We=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),ct=await We.
json().catch(()=>null);if(!We.ok||!ct){Te();return}cacheUserSettings(ct);const gt=get("settings-moda\
l");if(gt&&gt.classList.contains("modal-open"))try{mt(ct)}catch{}ct.theme_color&&applyThemeColor(ct.
theme_color,!0),Object.prototype.hasOwnProperty.call(ct,"minimal_prompt_mode")&&ct.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(ct,"compact_prompt_mode")&&setCompactPromptMode(
!!ct.compact_prompt_mode)}catch{}Te()},"refreshSettingsFormAfterImport"),et=r(Ae=>{const Te=Ae&&Ae.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";ie&&(ie.textContent=`\u5B8C\u4E86: ${Te}`,
ie.classList.remove("hidden","text-red-300"),ie.classList.add("text-emerald-300")),X({progress:100,phase:"\
completed",message:Te}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),y.includes("chats")&&loadThreads(),y.includes("skills")&&loadSkills(),y.includes("project\
s")&&typeof window.loadProjects=="function"&&window.loadProjects(),y.includes("files")&&loadStorageUsage(),
(y.includes("settings")||y.includes("api_credentials"))&&ft()},"finishImportSuccess"),St=r(async()=>{
try{const Te=await(await apiFetch(`/api/account/transfer/${W.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return Te&&Te.state?Te:null}catch{return null}},"fetchImportSta\
tus"),Ft=r(async()=>{const Ae=await St();if(!Ae)return{status:"unknown"};if(Ae.state==="completed")return et(
Ae),{status:"done"};if(["failed","cancelled","expired"].includes(Ae.state))throw new Error(Ae.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Ae.state==="needs_sel\
ection"&&Array.isArray(Ae.files)){const Te=await st({files:Ae.files,available_bytes:Ae.available_bytes});
return Te===null?(X({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ke=Te,{status:"reselect"})}if(Ae.state===
"needs_settings_confirmation"&&Array.isArray(Ae.settings_changes))return await ze({settings_changes:Ae.
settings_changes})?(O=!0,{status:"reselect"}):(X({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
W.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Ae.state==="running"){const Te=await Promise.race([re.catch(()=>null),new Promise(We=>setTimeout(
()=>We(null),6e4))]);if(Te&&Te.state==="completed")return et(Te),{status:"done"};throw Te&&["failed",
"cancelled","expired"].includes(Te.state)?new Error(Te.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ne;){W.stopped=!0,await re.catch(()=>null),
W.stopped=!1,re=Q(W);let Ae;try{Ae=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:W.uploadId,
categories:y.join(","),job_id:W.id,selected_files:ke,restore_inplace:S,confirm_settings:O||P}),signal:W.
controller.signal}))}catch(De){if(W.cancelRequested||De&&De.name==="AbortError")throw De;const Je=await Ft();
if(Je.status==="done"){Ne=!0;break}if(Je.status==="cancelled")return;if(Je.status==="reselect")continue;
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Te=null;try{Te=await Ae.json()}catch{Te=null}if(Te===null){const De=await Ft();if(De.status==="d\
one"){Ne=!0;break}if(De.status==="cancelled")return;if(De.status==="reselect")continue;if(Ae.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Ae.ok&&Te.error==="storage_limit_files"&&Te.files){const De=await st(Te);if(De===null){X({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ke=De;continue}if(Te&&Te.status==="settings_confirmation"&&
Array.isArray(Te.settings_changes)){if(!await ze(Te)){X({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),W.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}O=!0;continue}if(!Ae.ok)throw new Error(Te.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const We=Te.imported||{},ct=[`\u8A2D\u5B9A ${We.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${We.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${We.chats||0}\u4EF6`,`Skill ${We.skills||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${We.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${We.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${We.
diagnostics||0}\u4EF6`].join(" / "),gt=Te.duplicates||{},Dt={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"\
Skill",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Le=[];for(const De of Object.keys(Dt)){const Je=Number(gt[De])||0;Je>
0&&Le.push(`${Dt[De]} ${Je}\u4EF6`)}const qe=Le.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Le.
join("\u3001")}\uFF09`:"";ie&&(ie.textContent=`\u5B8C\u4E86: ${ct}${qe}`,ie.classList.remove("hidden",
"text-red-300"),ie.classList.add("text-emerald-300")),X({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\
\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),y.includes("chats")&&loadThreads(),y.includes("skills")&&loadSkills(),y.includes("project\
s")&&typeof window.loadProjects=="function"&&window.loadProjects(),y.includes("files")&&loadStorageUsage(),
(y.includes("settings")||y.includes("api_credentials"))&&ft(),Ne=!0}}catch(pe){if(W.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),W.cancelRequested||pe&&pe.name==="AbortError")return;const Z=pe&&pe.message?
pe.message:"",Se=Z==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
Z||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";X({progress:0,phase:"f\
ailed",message:Se}),ie&&(ie.textContent=Se,ie.classList.remove("hidden","text-emerald-300"),ie.classList.
add("text-red-300")),showToast(Se,"error",!0)}finally{W.stopped=!0,await re.catch(()=>null),ve(W)}});
const It=get("account-dedupe-btn"),Ot=get("account-dedupe-result"),Gt=r((u,f=!1)=>{Ot&&(Ot.textContent=
u,Ot.classList.remove("hidden"),Ot.classList.toggle("text-red-300",!!f),Ot.classList.toggle("text-em\
erald-300",!f))},"showDedupeResult");It&&(It.onclick=async()=>{const u=r(async()=>{const f=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),h=await f.json().catch(()=>null);if(!f.ok||!h)throw new Error(h&&h.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!h.has_duplicates){Gt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const y=[],_={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",
feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};
for(const W of["chats","skills","files","feedback","diagnostics"]){const ie=Number(h.duplicates&&h.duplicates[W])||
0;ie>0&&y.push(`${_[W]} ${ie}\u4EF6`)}const S=Number(h.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${h.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${h.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${y.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const M=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),P=await M.json().catch(()=>null);if(!M.ok||!P)throw new Error(
P&&P.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const O=[];for(const W of["chats","skills","files","feedback","diagnostics"]){const ie=Number(P.removed&&
P.removed[W])||0;ie>0&&O.push(`${_[W]} ${ie}\u4EF6`)}const q=Number(P.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\
\u305F\u3081\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${P.kept_referenced_files}\u4EF6\uFF09`:"";Gt(
`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\u3057\u305F: ${O.join("\u3001")||"0\u4EF6"}${q}`),
loadThreads(),loadSkills(),loadStorageUsage()},"run");if(!It.disabled){It.disabled=!0,Gt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\
\u3044\u307E\u3059...");try{await u()}catch(f){Gt(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{It.disabled=!1}}});const cn=get("site-cache-usage-refresh");cn&&(cn.onclick=()=>loadSiteCacheUsage());
const en=get("clear-site-cache-btn");en&&(en.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(en)});
const Y=get("enc-scan-result"),xe=r(async(u=null)=>{Y&&(Y.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const h=await apiFetch(
f,{cache:"no-store"}),y=await h.json();if(!h.ok){Y&&(Y.textContent=y.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const _=y.total||0,S=y.encrypted||0,M=y.unencrypted||0;let P=`Total: ${_} / Encrypted: ${S} /\
 Plain: ${M}`;if(y.samples&&y.samples.length){const O=y.samples.slice(0,8).map(q=>{const W=q.timestamp?
new Date(q.timestamp).toLocaleString():"";return`#${q.id} (${q.role||""}) ${W}`}).join(" / ");P+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${O}</div>`}Y&&(Y.innerHTML=P)}catch{Y&&(Y.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Be=get("enc-scan-all");Be&&(Be.onclick=()=>xe(
null));const Ue=get("enc-scan-thread");Ue&&(Ue.onclick=()=>currentThreadId?xe(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ye=get("admin-enc-\
list");let yt=null,vt=!1;const ot=r(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThread\
EncryptedFromMessages"),Tt=r(()=>{yt=ot(allMessages)},"refreshCurrentThreadEncStateFromMessages"),rt=r(
async(u,f,{confirmPrompt:h=!0,reloadCurrent:y=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const _=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(h&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;vt=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
M=await S.json().catch(()=>({}));return S.ok?(showToast(`${_}\u3057\u307E\u3057\u305F\uFF08${M.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),yt=!!f,y&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ye&&await wt(),!0):
(showToast(M.error||`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{vt=!1}},"setAdminThreadEncry\
ption"),Ze=r(u=>{if(!Ye)return;const f=u.threads||[];if(!f.length){Ye.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ye.innerHTML=f.map(h=>{const y=h.encrypted_count>0?"enc":"plain",_=y==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=y==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",M=h.
updated_at?new Date(h.updated_at).toLocaleString():"",P=escapeHtml(String(h.thread_id)),O=currentThreadId&&
String(currentThreadId)===String(h.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(h.title||
"")}">${escapeHtml(h.title||"(\u7121\u984C)")}${O?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${M} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${h.
message_count} / \u6697\u53F7\u5316: ${h.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${P}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${P}" data-enable="${y==="enc"?"0":"1"}" data-progress-expected-slow="true">${_}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),wt=r(async()=>{if(Ye){Ye.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){Ye.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(Ze(f),currentThreadId&&Array.isArray(f.threads)){const h=f.threads.find(y=>String(
y.thread_id)===String(currentThreadId));h&&(yt=!!h.encrypted)}}catch{Ye.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>wt()),window.__loadAdminEncThreads=
wt,window.__refreshAdminThreadEncState=Tt,window.__setAdminThreadEncryption=rt;const dn=get("encrypt\
ion-status-admin-toggle");dn&&dn.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ye&&(Ye.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const P=f.getAttribute("data-id");if(!P)return;typeof nt==
"function"?nt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(P)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const h=u.target.closest(".admin-enc-toggle");if(!h||vt)return;const y=h.getAttribute("data-i\
d"),_=h.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;h.disabled=!0;
const M=h.textContent;h.textContent="\u51E6\u7406\u4E2D...";try{await rt(y,_,{confirmPrompt:!1,reloadCurrent:!0})}finally{
h.disabled=!1,h.textContent=M,await wt()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const un=r(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(h=>{const y=h.status||"new",_=h.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=h.created_at?new Date(h.created_at).toLocaleString():"",M=h.replied_at?new Date(h.replied_at).toLocaleString():
"",P=h.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${h.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(h.username||
"")}${_}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(S)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(y)}</di\
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
P)}</textarea>
                                ${P?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
M)}</div>`:""}
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
                    `}).join("")}},"renderBanAppeals"),pn=r(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const h=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!h.
ok)return;const _=(await h.json()).unread_count||0;f.textContent=String(_),u&&_>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${_}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Ut=r(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const h=await f.json();un(h.items||[]),
await pn(!1)}catch{}}},"loadBanAppeals"),Fe=r(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await Ut()}catch{}},"markBanAppealsRead"),Ve=r(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await Ut()}catch{}},"updateBanAppealStatus"),zt=r(()=>{
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
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),$t=r(()=>{const u=get("set-st\
t-model");if(!u||get("set-llm-transcribe-prompt"))return;const f=u.closest(".space-y-2");if(!f)return;
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
                `,f.appendChild(h);const y=get("reset-llm-transcribe-prompt");y&&(y.onclick=()=>{const _=get(
"set-llm-transcribe-prompt");_&&(_.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),Vt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
{key:"deepseek_python",label:"DeepSeek Python\uFF08\u30C4\u30FC\u30EB\u5B9F\u884C\uFF09"},{key:"gemi\
ni_code_execution",label:"Gemini \u30B3\u30FC\u30C9\u5B9F\u884C\uFF08\u30B5\u30F3\u30C9\u30DC\u30C3\u30AF\u30B9\u306E\u5B9F\u884C\u6642\u9593\u5236\u9650\uFF09"},
{key:"python_file_creation",label:"Python + \u30D5\u30A1\u30A4\u30EB\u4F5C\u6210\uFF08Gemini \u30B3\u30FC\u30C9\u5B9F\u884C\u6642\uFF09"},
{key:"gemini_local_python",label:"Gemini \u97F3\u58F0/\u52D5\u753B/PDF/DOCX + Python\uFF08\u30ED\u30FC\u30AB\u30EB\u5B9F\u884C\uFF09"},
{key:"grok_search",label:"Search\u88DC\u52A9\uFF08Grok\uFF09"},{key:"openai_search",label:"Search\u88DC\u52A9\uFF08\
OpenAI/xAI Responses\uFF09"},{key:"marker",label:"Marker\u7DE8\u96C6\u6642"},{key:"attachment_names",
label:"\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u540D\uFF08LLM\u5165\u529B\u6642\uFF09",hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{\
attachment_names}} / {{attachment_count}}"},{key:"quote_source",label:"\u5F15\u7528\u5143\uFF08\u5F15\u7528\u3057\u3066\u9001\u4FE1\u6642\uFF09",
hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{quote_source}}\uFF08\u5F15\u7528\u5143\u306E\u767A\u8A00\u8005\u3068\u4F1A\u8A71\u5185\u306E\u756A\u53F7\u304C\u5165\u308A\u307E\u3059\uFF09"},
{key:"mathjax",label:"MathJax\uFF08LaTeX\u6570\u5F0F\uFF09"},{key:"image_analysis",label:"\u753B\u50CF\u89E3\u6790\uFF08Visio\
n Model\u6307\u793A\u6587\uFF09"},{key:"mcp",label:"MCP\uFF08\u5916\u90E8\u30C4\u30FC\u30EB\u63A5\u7D9A\uFF09",
hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{mcp_tools}}\uFF08\u63A5\u7D9A\u4E2D\u306EMCP\u30C4\u30FC\u30EB\u4E00\u89A7\u304C\u5165\u308A\u307E\u3059\uFF09",
barLocked:!0,barLabel:"MCP"},{key:"skills",label:"Skills\uFF08\u4FDD\u5B58\u3057\u305F\u6307\u793A\u306E\u5229\u7528\uFF09",
hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{skills}} / {{skill_count}}",barLocked:!0,barLabel:"Ski\
lls"},{key:"project_files",label:"\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u5171\u6709\u30D5\u30A1\u30A4\u30EB\uFF08\u53C2\u7167\u6848\u5185\uFF09",
hint:"\u5229\u7528\u53EF\u80FD\u5909\u6570: {{project_name}} / {{project_files}} / {{project_file_count}}"}];
window.buildAutoSystemPromptRows=(u,f=!1)=>{const h=f?"w-full h-14 bg-gray-950 border border-gray-70\
0 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 rounded p-2\
 text-xs text-gray-200";return Vt.map(y=>{const _=y.barLocked===!0,S=y.barLabel||"MCP",M=_?`<div cla\
ss="text-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\
\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>`:
"",P=_?`<input type="checkbox" id="${u}-auto-sys-${y.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${u}-auto-sys-${y.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${y.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${_?`ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"`:
""}>
                                ${P}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${u}-auto-sys-${y.key}-text" class="${h}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${y.hint?`<div class="text-[10px] text-gray-500 mt-1">${y.hint}</div>`:""}
                        ${M}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{Vt.forEach(h=>{const y=f&&
typeof f=="object"?f[h.key]||{}:{},_=get(`${u}-auto-sys-${h.key}-enabled`),S=get(`${u}-auto-sys-${h.
key}-text`);_&&(h.barLocked===!0?_.disabled=!0:_.checked=y.enabled!==!1),S&&(S.value=y.text||"",S.placeholder=
y.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const jt=r((u,f=null)=>{if(f){const h=get(f);h&&(h.checked=!0)}Vt.forEach(h=>{const y=get(`${u}-auto\
-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);if(y&&(h.barLocked!==!0?y.checked=!0:y.
disabled=!0),_){const S=_.placeholder||"";_.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),_e=r(u=>{const f={};return Vt.forEach(h=>{const y=get(`${u}\
-auto-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);f[h.key]={enabled:h.barLocked===!0?
!0:y?y.checked:!0,text:_?_.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
_e,window.ensureAutoSystemPromptSettingsCard=()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.
closest(".space-y-4"):null;if(!f||get("auto-sys-prompt-settings"))return;const h=document.createElement(
"div");h.id="auto-sys-prompt-settings",h.className="border-t border-gray-700 pt-3",h.innerHTML=`
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
                `,f.appendChild(h)},window.ensureThreadAutoSystemPromptCard=()=>{const u=get("thread\
-global-sys-prompt"),f=u?u.closest(".space-y-3"):null;if(!f||get("thread-auto-sys-prompt-settings"))
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
                `,f.appendChild(h)},zt(),$t(),F();const Ie=r(()=>{const u=get("set-default-model");if(!u)
return;const f=u.value;u.innerHTML="",MODELS.forEach(y=>{const _=document.createElement("optgroup");
_.label=y.category,(y.items||[]).forEach(S=>{const M=document.createElement("option");M.value=S.id,M.
textContent=S.name,_.appendChild(M)}),u.appendChild(_)});const h=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";h&&Array.from(u.options).some(y=>y.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),ut=r(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(y=>{const _=(y.items||[]).filter(M=>{const P=(M.id||"").toLowerCase();
return P.startsWith("gemini-")||P.startsWith("gpt-4o")||P.startsWith("claude-")||P.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
P)});if(_.length===0)return;const S=document.createElement("optgroup");S.label=y.category,_.forEach(
M=>{const P=document.createElement("option");P.value=M.id,P.textContent=M.name+" \u2605",S.appendChild(
P)}),u.appendChild(S)});const h=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";h&&Array.from(u.options).some(y=>y.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),mt=r(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-glo\
bal-sys-prompt-preview");f&&(f.value=u.global_system_prompt_effective||"");const h=get("app-global-s\
ys-prompt-preview-status");h&&(u.global_system_prompt_enabled===!1?h.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
u.global_system_prompt_uses_time_fallback?h.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
h.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("sys-prompt-text")&&(get("sys-prompt-text").value=u.system_prompt||""),get("set-global-sys-promp\
t-enabled")&&(get("set-global-sys-prompt-enabled").checked=u.system_prompt_enabled!==!1),window.ensureAutoSystemPromptSettingsCard(),
get("set-apply-global-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=u.apply_global_system_prompt!==
!1),get("set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=u.apply_auto_system_prompt_notices!==
!1),window.applyAutoSystemPromptConfigToForm("set",u.auto_system_prompt_notices_config||{}),get("set\
-latency-metrics")&&(get("set-latency-metrics").checked=u.enable_latency_metrics===!0),get("set-clie\
nt-debug-log")&&syncClientDebugLogToggle(u.enable_client_debug_log===!0,"settings modal sync"),get("\
set-openai")&&(get("set-openai").value=u.openai_key||""),get("set-gemini")&&(get("set-gemini").value=
u.gemini_key||""),get("set-gemini-free")&&(get("set-gemini-free").value=u.gemini_free_key||""),get("\
set-deepseek")&&(get("set-deepseek").value=u.deepseek_key||""),get("set-zai")&&(get("set-zai").value=
u.zai_key||""),get("set-kimi")&&(get("set-kimi").value=u.kimi_key||""),get("set-mistral")&&(get("set\
-mistral").value=u.mistral_key||""),get("set-ideogram")&&(get("set-ideogram").value=u.ideogram_key||
""),get("set-anthropic")&&(get("set-anthropic").value=u.anthropic_key||""),get("set-gemini-backend")&&
(get("set-gemini-backend").value=normalizeGeminiBackend(u.gemini_backend||"gemini_api")),get("set-ge\
mini-vertex-project")&&(get("set-gemini-vertex-project").value=u.gemini_vertex_project||""),get("set\
-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=u.gemini_vertex_location||"globa\
l"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-gemini\
-vertex-credentials-json").value=u.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),typeof window.
toggleOptions=="function"&&window.toggleOptions(),get("set-admin-api-key-mode")&&(get("set-admin-api\
-key-mode").value=normalizeAdminApiKeyMode(u.admin_api_key_mode||"env_fallback")),syncAdminApiKeyModeUi(),
get("set-xai")&&(get("set-xai").value=u.xai_key||""),get("set-google-key")&&(get("set-google-key").value=
u.google_key||""),get("set-google-project")&&(get("set-google-project").value=u.google_project||""),
modelApiKeyMap=normalizeModelApiKeyMap(u.model_api_keys||{}),syncModelApiKeyModelOptions(),renderModelApiKeyList(),
setModelApiKeyPanelOpen(!1),get("set-mic-transcribe-mode")&&(get("set-mic-transcribe-mode").value=u.
mic_transcribe_mode||"stt_api"),get("set-stt-model")&&(get("set-stt-model").value=u.stt_model||"gpt-\
4o-mini-transcribe"),get("set-llm-transcribe-prompt")&&(get("set-llm-transcribe-prompt").value=u.llm_transcribe_prompt||
"",get("set-llm-transcribe-prompt").placeholder=u.llm_transcribe_prompt_default||""),syncRichPastePromptPreferencesUi(
u),updateGoogleLinkUI(u),updateMinashinLinkUI(u),get("set-enter-to-send")&&(get("set-enter-to-send").
checked=!!u.enter_to_send),typeof window.applyMathStepsSettings=="function"&&window.applyMathStepsSettings(
u),get("set-math-steps-enabled")&&(get("set-math-steps-enabled").checked=u.math_steps_enabled===!0),
get("set-default-math-steps")&&(get("set-default-math-steps").checked=u.default_math_steps===!0),writePromptBarModeToForm(
!!u.compact_prompt_mode,!!u.minimal_prompt_mode),get("set-use-sw-cache")&&(get("set-use-sw-cache").checked=
!!u.use_sw_cache),get("set-clear-cache-on-version-update")&&(get("set-clear-cache-on-version-update").
checked=!!u.clear_cache_on_version_update),get("set-liquid-glass")&&(get("set-liquid-glass").checked=
!!u.liquid_glass_enabled),get("set-light-mode")&&(get("set-light-mode").checked=!!u.light_mode_enabled),
get("set-auto-search-links")&&(get("set-auto-search-links").checked=u.auto_search_on_links!==!1),get(
"set-use-last-settings")&&(get("set-use-last-settings").checked=!!u.use_last_chat_settings),get("set\
-default-model")&&(get("set-default-model").value=u.default_model||"gemini-3.6-flash"),get("set-defa\
ult-vision-model")&&(get("set-default-vision-model").value=u.default_vision_model||"gemini-3-flash-p\
review"),syncSettingsModelPickerLabels(),applyTemporaryChatTimeoutSeconds(u.temp_chat_timeout_seconds),
get("set-default-search")&&(get("set-default-search").checked=!!u.default_enable_search),get("set-de\
fault-gemini-free-tier")&&(get("set-default-gemini-free-tier").checked=!!u.default_gemini_free_tier),
get("set-default-gemini-flex")&&(get("set-default-gemini-flex").checked=!!u.default_gemini_flex),get(
"set-default-url-context")&&(get("set-default-url-context").checked=!!u.default_enable_url_context),
get("set-default-maps")&&(get("set-default-maps").checked=!!u.default_enable_maps),get("set-default-\
python")&&(get("set-default-python").checked=!!u.default_enable_python),get("set-default-file-creati\
on")&&(get("set-default-file-creation").checked=!!u.default_enable_file_creation),get("set-default-t\
hinking")&&(get("set-default-thinking").checked=!!u.default_enable_thinking),get("set-default-sys-pr\
ompt")&&(get("set-default-sys-prompt").checked=!!u.default_enable_system_prompt),get("set-default-mc\
p")&&(get("set-default-mcp").checked=u.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get(
"set-default-thinking-level").value=u.default_thinking_level||"high"),get("set-default-thinking-budg\
et")&&(get("set-default-thinking-budget").value=u.default_thinking_budget||4096),get("set-default-re\
asoning-effort")&&(get("set-default-reasoning-effort").value=u.default_reasoning_effort||"medium"),get(
"set-default-safety")&&(get("set-default-safety").value=u.default_safety_setting||"default"),get("se\
t-e2ee").checked=u.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=u.bot_detection_enabled!==
!1),get("set-bot-detect-global")&&(get("set-bot-detect-global").checked=u.bot_detection_global_enabled!==
!1);const y=get("bot-status");y&&(u.is_bot_banned?(y.textContent=`BAN\u4E2D: ${u.bot_ban_reason||"Bo\
t detection"}`,y.classList.remove("hidden"),y.classList.add("text-red-400")):y.classList.add("hidden")),
u&&u.theme_color?(applyThemeColor(u.theme_color,!0),syncThemeInputs(u.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=u.username);const _=get("2fa-badge"),S=get("disable-2fa-btn");u.is_2fa_enabled?
(_.innerText="ENABLED",_.classList.replace("bg-gray-700","bg-green-600"),_.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(_.innerText="DISABLED",_.classList.replace("bg\
-green-600","bg-gray-700"),_.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!u.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=u.default_2fa_method||"totp");const M=get(
"set-passkey-only-login"),P=get("passkey-only-note"),O=Array.isArray(u.passkey_credentials)?u.passkey_credentials:
[];if(D(O),M){M.checked=!!u.passkey_only_login;const pe=O.length>0||!!u.has_webauthn;M.disabled=!pe,
pe||(M.checked=!1),P&&(pe?P.classList.add("hidden"):P.classList.remove("hidden"))}const q=get("mig-s\
tatus-box"),W=get("mig-progress-text"),ie=get("mig-progress-bar");if((u.migration_status||"idle")===
"processing"){q.classList.remove("hidden");const pe=(u.migration_progress||"").split("/");if(pe.length===
2){const Z=parseInt(pe[0]||"0",10),Se=parseInt(pe[1]||"0",10);W&&(W.innerText=`${Z} / ${Se}`),ie&&Se>
0&&(ie.style.width=`${Math.min(100,Math.floor(Z/Se*100))}%`)}}else q.classList.add("hidden"),ie&&(ie.
style.width="0%"),W&&(W.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSe\
ttingsFormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),snapshotSidebarHistory("settings-open-before");const u=await ensureUserSettingsSnapshot();u&&mt(
u);const f=get("search-box"),h=f?f.value:"";clearTimeout(searchTimeout);const y=get("settings-search");
if(y&&(y.value=""),filterSettings(),Ie(),ut(),showModal("settings-modal"),refreshSettingsTabsScroll(),
requestAnimationFrame(()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(h,"restored-search-b\
ox-open"),revealPersistentSidebarLists(),snapshotSidebarHistory("settings-open-after"),[50,200,400,800].
forEach(_=>{setTimeout(()=>{restoreThreadSearchValue(h,"restored-search-box-"+_+"ms"),snapshotSidebarHistory(
"settings-open-later-"+_+"ms")},_)}),syncAdaptiveBlurSettingsUi(),loadStorageUsage(),loadSiteCacheUsage(),
ne(),$t(),typeof window.__loadAdminEncThreads=="function")try{window.__loadAdminEncThreads()}catch{}
location.pathname!=="/settings"&&history.pushState({modal:"settings",from:location.pathname},"","/se\
ttings"),pn(!0),Ut(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\
\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044","error",!0)),hn(),$(),
ee();try{loadMcpServers()}catch{}};const nt=r((u=!1)=>{settingsModelPickerTarget&&window.closeModelModal(
!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close\
-later-300ms"),300),!u&&location.pathname==="/settings"&&history.back()},"closeSettingsModal"),Nt=r(
()=>{const u=get("set-theme-color"),f=get("set-theme-color-text"),h=get("theme-reset-btn"),y=document.
querySelectorAll("#theme-presets .theme-swatch"),_=r((S,M=!0)=>{const P=normalizeHex(S);P&&(applyThemeColor(
P,M),syncThemeInputs(P))},"applyFromValue");u&&u.addEventListener("input",()=>_(u.value,!0)),f&&(f.addEventListener(
"change",()=>{const S=normalizeHex(f.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}_(S,!0)}),f.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
f.blur())})),h&&(h.onclick=()=>_(THEME_DEFAULT,!0)),y.forEach(S=>{S.addEventListener("click",()=>_(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),Wt=r(()=>{const u=get("reset-global-sys-prom\
pt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const f=get(
"reset-thread-sys-prompt");f&&(f.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-set-auto-sys-prompt-defaults");h&&(h.onclick=()=>{jt("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const y=get("reset-thread-auto-sys-prompt-defaults");y&&(y.onclick=()=>{jt("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>nt();const Jn=get("settings-header-close");Jn&&(Jn.onclick=()=>nt());
const tn=get("settings-search");tn&&(tn.addEventListener("input",filterSettings),tn.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const h=f.querySelector(
":scope > .settings-match");h&&h.scrollIntoView({behavior:"smooth",block:"start"})}}));const Kn=get(
"settings-search-clear");Kn&&Kn.addEventListener("click",()=>{tn&&(tn.value="",filterSettings(),tn.focus())}),
Nt(),Wt(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const jn=get("set-gemini-local-python-dialog");jn&&(jn.onchange=()=>setGeminiLocalPyDialogEnabled(jn.
checked));const Xn=get("set-gemini-backend");Xn&&(Xn.onchange=()=>{syncGeminiBackendUi(),typeof window.
toggleOptions=="function"&&window.toggleOptions()});const Yn=get("set-admin-api-key-mode");Yn&&(Yn.onchange=
()=>syncAdminApiKeyModeUi());const Nn=get("set-temp-chat-timeout-seconds");Nn&&(Nn.onchange=()=>{applyTemporaryChatTimeoutSeconds(
Nn.value)});const Qn=get("slash-command-cancel-btn");Qn&&(Qn.onclick=()=>{hidePendingSlashCommandIndicator();
const u=get("prompt-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-s\
ettings-btn").onclick=async()=>{var P;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),h=readPromptBarModeFromForm(),y={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:_e(
"set"),theme_color:normalizeHex(get("set-theme-color-text")?get("set-theme-color-text").value:"")||THEME_DEFAULT,
light_mode_enabled:get("set-light-mode")?get("set-light-mode").checked:!1,mic_transcribe_mode:get("s\
et-mic-transcribe-mode")?get("set-mic-transcribe-mode").value:"stt_api",stt_model:get("set-stt-model")?
get("set-stt-model").value:null,llm_transcribe_prompt:get("set-llm-transcribe-prompt")?get("set-llm-\
transcribe-prompt").value:"",enter_to_send:get("set-enter-to-send")?get("set-enter-to-send").checked:
!1,math_steps_enabled:get("set-math-steps-enabled")?get("set-math-steps-enabled").checked:!1,default_math_steps:get(
"set-default-math-steps")?get("set-default-math-steps").checked:!1,compact_prompt_mode:h.compact_prompt_mode,
minimal_prompt_mode:h.minimal_prompt_mode,use_sw_cache:get("set-use-sw-cache")?get("set-use-sw-cache").
checked:!1,clear_cache_on_version_update:get("set-clear-cache-on-version-update")?get("set-clear-cac\
he-on-version-update").checked:!1,liquid_glass_enabled:get("set-liquid-glass")?get("set-liquid-glass").
checked:!1,auto_search_on_links:get("set-auto-search-links")?get("set-auto-search-links").checked:!0,
use_last_chat_settings:get("set-use-last-settings")?get("set-use-last-settings").checked:!1,voice_studio_ui:get(
"set-voice-studio-ui")?get("set-voice-studio-ui").checked:!0,default_model:get("set-default-model")?
get("set-default-model").value:null,default_vision_model:get("set-default-vision-model")?get("set-de\
fault-vision-model").value:null,temp_chat_timeout_seconds:normalizeTemporaryChatTimeoutSeconds(get("\
set-temp-chat-timeout-seconds")?get("set-temp-chat-timeout-seconds").value:temporaryChatTimeoutSeconds),
default_enable_search:get("set-default-search")?get("set-default-search").checked:!1,default_gemini_free_tier:get(
"set-default-gemini-free-tier")?get("set-default-gemini-free-tier").checked:!1,default_gemini_flex:get(
"set-default-gemini-flex")?get("set-default-gemini-flex").checked:!1,default_enable_url_context:get(
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
"set-latency-metrics")?get("set-latency-metrics").checked:!1,server_diagnostic_log_retention_hours:Number(
((P=get("set-server-diagnostic-log-retention-hours"))==null?void 0:P.value)||1),enable_server_diagnostic_log:get(
"set-server-diagnostic-log")?get("set-server-diagnostic-log").checked:!1,enable_client_debug_log:get(
"set-client-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-o\
nly-login")?get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-goog\
le")?get("set-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set\
-default-2fa-method").value:"totp",new_username:u?u.value:null,new_password:f?f.value:null},_=get("s\
et-e2ee")?get("set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
_!==S&&(y.enable_e2ee=_),get("set-openai")&&(y.openai_key=get("set-openai").value),get("set-gemini")&&
(y.gemini_key=get("set-gemini").value),get("set-gemini-free")&&(y.gemini_free_key=get("set-gemini-fr\
ee").value),get("set-deepseek")&&(y.deepseek_key=get("set-deepseek").value),get("set-zai")&&(y.zai_key=
get("set-zai").value),get("set-kimi")&&(y.kimi_key=get("set-kimi").value),get("set-mistral")&&(y.mistral_key=
get("set-mistral").value),get("set-ideogram")&&(y.ideogram_key=get("set-ideogram").value),get("set-a\
nthropic")&&(y.anthropic_key=get("set-anthropic").value),y.model_api_keys=normalizeModelApiKeyMap(modelApiKeyMap),
get("set-gemini-backend")&&(y.gemini_backend=normalizeGeminiBackend(get("set-gemini-backend").value)),
get("set-gemini-vertex-project")&&(y.gemini_vertex_project=get("set-gemini-vertex-project").value),get(
"set-gemini-vertex-location")&&(y.gemini_vertex_location=get("set-gemini-vertex-location").value),get(
"set-gemini-vertex-credentials-json")&&(y.gemini_vertex_credentials_json=get("set-gemini-vertex-cred\
entials-json").value),get("set-xai")&&(y.xai_key=get("set-xai").value),get("set-google-key")&&(y.google_key=
get("set-google-key").value),get("set-google-project")&&(y.google_project=get("set-google-project").
value),get("set-admin-api-key-mode")&&(y.admin_api_key_mode=normalizeAdminApiKeyMode(get("set-admin-\
api-key-mode").value)),get("set-bot-detect")&&(y.bot_detection_enabled=get("set-bot-detect").checked),
get("set-bot-detect-global")&&(y.bot_detection_global_enabled=get("set-bot-detect-global").checked);
const M=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"appli\
cation/json"},body:JSON.stringify(y)});if(M.ok){let O="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const re=await M.json();re&&re.message&&(O=re.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!y.light_mode_enabled,liquid_glass_enabled:!!y.liquid_glass_enabled,
math_steps_enabled:!!y.math_steps_enabled,default_math_steps:!!y.default_math_steps})),typeof window.
applyMathStepsSettings=="function"&&(window.applyMathStepsSettings({math_steps_enabled:!!y.math_steps_enabled}),
y.math_steps_enabled||setMathStepsEnabled(!1,{persist:!1})),window.applySavedUserSystemPromptSettings(
{system_prompt:y.system_prompt,system_prompt_enabled:y.system_prompt_enabled,apply_global_system_prompt:y.
apply_global_system_prompt,apply_auto_system_prompt_notices:y.apply_auto_system_prompt_notices,auto_system_prompt_notices_config:y.
auto_system_prompt_notices_config}),nt();const q=currentUsername,W=CHAT_CONFIG.enableE2EE;enterToSend=
y.enter_to_send,autoSearchOnLinks=y.auto_search_on_links;const ie=useSwCache;useSwCache=y.use_sw_cache,
window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!y.clear_cache_on_version_update),
compactPromptMode=y.compact_prompt_mode,minimalPromptMode=y.minimal_prompt_mode,voiceStudioUiEnabled=
y.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=y.temp_chat_timeout_seconds,applyThemeColor(y.theme_color,
!0),syncThemeInputs(y.theme_color),applyLightMode(y.light_mode_enabled),applyLiquidGlassMode(y.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),ie!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(O,"success"),syncClientDebugLogToggle(
y.enable_client_debug_log,"settings saved"),y.new_username&&y.new_username!==q?setTimeout(()=>location.
reload(),1e3):y.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let O={};try{O=await M.json()}catch{}showToast(O.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const f=get("2fa-badge");f&&(f.innerText="D\
ISABLED",f.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const u=get("bot-unban-username"),f=u?u.value.trim():
"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"single"})}),y=await h.json(),_=get("bot-unban-result");if(h.
ok&&y&&y.status==="ok")_&&(_.textContent=`${f} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=y&&y.error?y.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const u=get("bot-unban-username"),f=u?u.value.trim():"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"linked"})}),y=await h.json(),_=get("bot-unban-result");if(h.
ok&&y&&y.status==="ok")_&&(_.textContent=`${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=y&&y.error?y.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const u=get(
"bot-speed-test-btn"),f=get("bot-speed-test-result");u&&(u.disabled=!0),u&&u.classList.add("opacity-\
60","cursor-not-allowed"),f&&(f.classList.remove("hidden"),f.textContent="\u5B9F\u884C\u4E2D...");try{
const h=r(le=>{f&&(f.textContent=le)},"setBox"),y=r(()=>`${Date.now()}_${Math.random().toString(36).
slice(2)}`,"cacheBust"),_=r((le,Ce)=>!le||!Ce||Ce<=0?0:le*8/(Ce/1e3)/1e3/1e3,"toMbps"),S=r(le=>Number.
isFinite(le)?`${le.toFixed(0)} ms`:"-","fmtMs"),M=r(le=>Number.isFinite(le)?`${le.toFixed(le>=100?0:
1)} Mbps`:"-","fmtMbps"),P=r(async(le,Ce)=>{const $e=await le.json().catch(()=>({}));return $e&&$e.error?
$e.error:Ce},"parseErr"),O=[];h("\u6E2C\u5B9A\u4E2D... ping");for(let le=0;le<4;le++){const Ce=performance.
now(),$e=await apiFetch(`/api/speedtest/ping?_=${y()}`,{cache:"no-store"}),fe=performance.now();if(!$e.
ok)throw new Error(await P($e,"ping_failed"));await $e.json().catch(()=>({})),O.push(fe-Ce)}const q=O.
reduce((le,Ce)=>le+Ce,0)/Math.max(1,O.length),W=Math.min(...O),ie=r(async le=>{const Ce=performance.
now(),$e=await apiFetch(`/api/speedtest/download?bytes=${le}&_=${y()}`,{cache:"no-store"});if(!$e.ok)
throw new Error(await P($e,"download_failed"));const fe=await $e.arrayBuffer(),ke=performance.now();
return{bytes:fe.byteLength||le,ms:ke-Ce,mbps:_(fe.byteLength||le,ke-Ce)}},"runDownload");h(`\u6E2C\u5B9A\u4E2D... p\
ing ${S(q)}
\u6E2C\u5B9A\u4E2D... download`);const re=[];for(const le of[2*1024*1024,8*1024*1024])re.push(await ie(
le)),h(`\u6E2C\u5B9A\u4E2D... ping ${S(q)}
download ${M(Math.max(...re.map(Ce=>Ce.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const pe=Math.max(...re.map(le=>le.mbps)),Z=r(async le=>{const Ce=new Uint8Array(
le),$e=performance.now(),fe=await apiFetch(`/api/speedtest/upload?_=${y()}`,{method:"POST",headers:{
"Content-Type":"application/octet-stream"},body:Ce,cache:"no-store"}),ke=performance.now();if(!fe.ok)
throw new Error(await P(fe,"upload_failed"));const Ne=await fe.json().catch(()=>({})),Re=Number(Ne.bytes_received||
le)||le;return{bytes:Re,ms:ke-$e,mbps:_(Re,ke-$e),serverMs:Number(Ne.server_elapsed_ms||0)||0}},"run\
Upload"),Se=[];for(const le of[1*1024*1024,4*1024*1024])Se.push(await Z(le));const oe=Math.max(...Se.
map(le=>le.mbps)),Oe=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(q)} / ${S(W)}`,`Download (best): ${M(pe)}`,`Upload (best): ${M(oe)}`,`Download \
runs: ${re.map(le=>`${Math.round(le.bytes/1024/1024)}MB=${M(le.mbps)}`).join(", ")}`,`Upload runs: ${Se.
map(le=>`${Math.round(le.bytes/1024/1024)}MB=${M(le.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\
\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
h(Oe.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(h){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${h&&h.message?h.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Ut()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Fe()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){h&&await Fe([Number(h)]);return}if(f.classList.contains(
"ban-appeal-status")){const y=f.getAttribute("data-status");h&&y&&await Ve({id:Number(h),status:y});
return}if(f.classList.contains("ban-appeal-reply-send")){const y=f.closest("[data-appeal-id]"),_=y?y.
querySelector(".ban-appeal-reply"):null,S=_?_.value:"";h&&await Ve({id:Number(h),admin_reply:S});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const y=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";h&&await Ve({id:Number(
h),block_user:!0,block_reason:y});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const u=get(get("camera-split-enabled").checked?"image-split-\
input":"photo-input");u&&u.click()}),get("upload-clear-btn")&&(get("upload-clear-btn").onclick=()=>{
resetUploadState()}),get("marker-modal-close")&&(get("marker-modal-close").onclick=()=>{closeMarkerModal(),
markerState.row=null}),get("marker-tool-draw")&&(get("marker-tool-draw").onclick=()=>setMarkerMode("\
draw")),get("marker-tool-auto")&&(get("marker-tool-auto").onclick=()=>setMarkerMode("auto")),get("ma\
rker-tool-mosaic")&&(get("marker-tool-mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool\
-crop")&&(get("marker-tool-crop").onclick=()=>setMarkerMode("crop")),get("marker-auto-tolerance")&&(get(
"marker-auto-tolerance").oninput=u=>{const f=get("marker-auto-tolerance-value");f&&(f.textContent=String(
u.target.value))});const Rn=get("marker-color-picker");Rn&&(Rn.oninput=u=>setMarkerColor(u.target.value),
Rn.onchange=u=>setMarkerColor(u.target.value));const Bn=get("marker-opacity");Bn&&(Bn.oninput=u=>setMarkerOpacity(
u.target.value),Bn.onchange=u=>setMarkerOpacity(u.target.value));const Tn=get("marker-opacity-number");
Tn&&(Tn.onchange=u=>setMarkerOpacity(u.target.value),Tn.onblur=u=>setMarkerOpacity(u.target.value),Tn.
onkeydown=u=>{u.key==="Enter"&&(setMarkerOpacity(u.target.value),u.target.blur())}),document.querySelectorAll(
"#marker-toolbar .marker-color-chip[data-marker-color]").forEach(u=>{u.onclick=()=>setMarkerColor(u.
getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),
get("marker-crop-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&
(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=
()=>clearMarkerCanvas()),get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),
initMarkerCanvas(),initCropCanvas(),window.addEventListener("resize",()=>{const u=get("marker-modal");
!u||u.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const wi=r(()=>{const u=get(
"upload-modal");return!!(u&&!u.classList.contains("hidden"))},"isUploadModalOpen"),nn=get("drop-over\
lay");let mn=0;const xi=r(()=>{wi()||nn&&(nn.classList.remove("hidden"),nn.classList.add("flex"))},"\
showDropOverlay"),fn=r(()=>{mn=0,nn&&(nn.classList.add("hidden"),nn.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=fn;const Pt=get("upload-dropzone");Pt&&(Pt.addEventListener("dr\
agover",u=>{u.preventDefault(),Pt.classList.add("dragover")}),Pt.addEventListener("dragleave",()=>{Pt.
classList.remove("dragover")}),Pt.addEventListener("drop",u=>{u.preventDefault(),u.stopPropagation(),
Pt.classList.remove("dragover"),fn();const f=u.dataTransfer?u.dataTransfer.files:null;f&&f.length&&handleFiles(
f)})),window.addEventListener("dragenter",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.
types.includes("Files")||(mn+=1,xi())}),window.addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||u.preventDefault()}),window.addEventListener("dragle\
ave",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(mn=Math.max(
0,mn-1),(mn===0||!u.relatedTarget||u.clientY<=0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=
window.innerHeight)&&fn())}),window.addEventListener("dragend",()=>{fn()}),window.addEventListener("\
drop",u=>{fn(),!(!u.dataTransfer||!u.dataTransfer.files||u.dataTransfer.files.length===0)&&(u.preventDefault(),
!(Pt&&Pt.contains(u.target))&&handleFiles(u.dataTransfer.files))});const Zn=get("bot-admin-modal"),ki=r(
u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML="",!u||!u.length){f.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((h,y)=>{const _=!!h.is_bot_banned,S=h.bot_detection_enabled!==!1,M=Number(h.lock_remaining_seconds)||
0,P=document.createElement("div");P.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",P.style.animationDelay=`${Math.min(y,12)*.02}\
s`,P.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(h.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${_?"BAN\u4E2D":"\u6B63\u5E38"}${M?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(M/60)}\u5206\uFF09`:""} ${h.bot_ban_reason?
" / "+escapeHtml(h.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(h.username)}" data-enabled="${S?"1":"0"}">${S?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${_?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(h.username)}" data-bann\
ed="${_?"1":"0"}">${_?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${_?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(h.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        ${h.is_admin?"":`<button class="bot-delete-account bg-red-800 hover:bg-red-7\
00 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(h.username)}\
">\u524A\u9664</button>`}
                    `,f.appendChild(P)})}},"renderBotUsers"),gn=r(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const h=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),y=await h.json();h.ok&&y&&y.users?ki(y.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{f&&(f.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Fn=r(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Zn))return;const f=get(
"settings-modal");f&&(f.classList.contains("modal-open")||f.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await gn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Fn;const Dn=r(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const u=get("settings-modal");u&&(u.classList.contains(
"modal-open")||u.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=Dn,window.closeBotLogModal=(u=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!u&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{Dn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(u=!1)=>{(get("bot-admin-modal")||Zn)&&hideModal("bot-admin-modal"),!u&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Fn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await gn(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await gn("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async u=>{u.key==="Enter"&&await gn(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute(
"data-user");if(!h)return;let y;if(f.classList.contains("bot-toggle-detect")){const _=f.getAttribute(
"data-enabled")!=="1";y=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:h,action:"toggle_detection",enabled:_})})}else if(f.classList.
contains("bot-toggle-ban"))if(f.getAttribute("data-banned")==="1")y=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;y=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"ban",reason:"Admin ban"})})}else if(f.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;y=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"unban_linked"})})}else if(f.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;y=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"delete_account"})})}if(y){if(y.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(y.ok){if(f.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),h===currentUsername)){location.href="/";return}}else{
let _={};try{_=await y.json()}catch{}showToast(_.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
_.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await gn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Rt={"/settings":{id:"settings-mod\
al",open:r(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:r(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:r(()=>{gi(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
pen")},"/history":{id:"history-modal",open:r(()=>window.showHistoryModal(),"open")},"/branch":{id:"b\
ranch-modal",open:r(()=>window.showBranchModal(),"open")},"/batch":{id:"batch-modal",open:r(()=>window.
showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:r(()=>openRichPasteModal(),"open")},"\
/camera":{id:"camera-capture-modal",open:r(()=>openCameraCaptureModal(),"open")},"/edit-image":{id:"\
marker-modal",open:r(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:r(()=>window.openThreadModal(),
"open")},"/model":{id:"model-modal",open:r(()=>openModelModal(),"open")},"/token-details":{id:"token\
-detail-modal",open:r(()=>{history.state&&history.state.promptEstimate?restorePromptTokenDetail():history.
state&&history.state.messageId!==void 0?openTokenDetail(history.state.messageId):showTokenDetailModal()},
"open")},"/encryption-status":{id:"encryption-status-modal",open:r(()=>showEncryptionStatusModal(),"\
open")},"/python-execution":{id:"python-exec-modal",open:r(()=>showPythonExecDetailModal(),"open")},
"/skill":{id:"skill-modal",open:r(()=>{editingSkillUuid=null,get("skill-modal-title").textContent="C\
reate New Skill",showModal("skill-modal")},"open")},"/project":{id:"project-modal",open:r(()=>{typeof window.
openProjectModal=="function"&&window.openProjectModal(null)},"open")},"/compression":{id:"compressio\
n-modal",open:r(()=>window.openCompressionModal(),"open")},"/math-steps":{id:"math-steps-config-moda\
l",open:r(()=>window.openMathStepsModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:r(()=>Fn(),
"open")},"/admin-bot-logs":{id:"bot-log-modal",open:r(()=>Dn(),"open")}};Rt["/gem"]=Rt["/skill"];const ei=r(
(u,f=!1)=>{switch(u){case"settings-modal":nt(f);break;case"upload-modal":closeUploadModal(f);break;case"\
image-split-modal":closeImageSplit();break;case"image-split-compression-modal":hideModal(u);break;case"\
camera-capture-modal":closeCameraCaptureModal(f?{skipHistory:!0}:{});break;case"history-modal":window.
closeHistoryModal&&window.closeHistoryModal(f);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(
f);break;case"branch-modal":window.closeBranchModal&&window.closeBranchModal(f);break;case"batch-mod\
al":window.closeBatchModal&&window.closeBatchModal(f);break;case"rich-paste-modal":window.closeRichPasteModal&&
window.closeRichPasteModal(f);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(
f);break;case"thread-modal":window.closeThreadModal&&window.closeThreadModal(f);break;case"model-mod\
al":window.closeModelModal&&window.closeModelModal(f);break;case"token-detail-modal":closeTokenDetail(
f);break;case"encryption-status-modal":closeEncryptionModal(f);break;case"python-exec-modal":closePythonExecDetail(
f);break;case"skill-modal":window.closeSkillModal&&window.closeSkillModal(f);break;case"project-moda\
l":window.closeProjectModal?window.closeProjectModal(f):hideModal(u);break;case"move-thread-project-\
modal":window.closeMoveThreadProjectModal?window.closeMoveThreadProjectModal():hideModal(u);break;case"\
compression-modal":window.closeCompressionModal&&window.closeCompressionModal(f);break;case"math-ste\
ps-config-modal":window.closeMathStepsModal?window.closeMathStepsModal(f):hideModal(u);break;case"bo\
t-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(f);break;case"bot-log-modal":window.
closeBotLogModal&&window.closeBotLogModal(f);break;case"mcp-decision-modal":typeof submitMcpDecision==
"function"?submitMcpDecision("deny"):hideModal(u);break;case"api-key-required-modal":{const y=get("a\
pi-key-modal-cancel-btn");y&&typeof y.onclick=="function"?y.click():hideModal(u);break}case"lyria-st\
udio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(u);break;case"voice-studio-m\
odal":window.VoiceStudio?window.VoiceStudio.close():hideModal(u);break;case"version-update-modal":const h=localStorage.
getItem("app_version")||"";h&&localStorage.setItem("version_notified",h),hideModal(u);break;default:
hideModal(u);break}},"closeModalById");window.addEventListener("popstate",u=>{var M,P;const f=(M=u.state)==
null?void 0:M.settingsModelPicker;if(location.pathname==="/settings"&&settingsModelPickerTargets.includes(
f)){const O=r(()=>openSettingsModelPicker(f,!0),"restorePicker");(P=get("settings-modal"))!=null&&P.
classList.contains("modal-open")?O():window.openSettingsModal().then(()=>{var q;location.pathname===
"/settings"&&((q=history.state)==null?void 0:q.settingsModelPicker)===f&&O()});return}settingsModelPickerTarget&&
window.closeModelModal(!0);let h=!1;Object.values(Rt).forEach(O=>{const q=get(O.id);q&&q.classList.contains(
"modal-open")&&location.pathname!==Object.keys(Rt).find(W=>Rt[W].id===O.id)&&(ei(O.id,!0),h=!0)});const y=location.
pathname.match(/^\/c\/(.+)$/),_=location.pathname.match(/^\/p\/(.+)$/);if(y){const O=decodeURIComponent(
y[1]);String(currentThreadId)!==String(O)&&loadMessages(O,{skipHistory:!0})}else if(_){const O=decodeURIComponent(
_[1]);typeof window.openProjectView=="function"&&window.openProjectView(O,{skipHistory:!0})}else location.
pathname==="/"&&(typeof window.closeProjectView=="function"&&window.closeProjectView({skipHistory:!0}),
currentThreadId&&startNewChat({skipHistory:!0}));const S=Rt[location.pathname];if(S){const O=get(S.id);
O&&!O.classList.contains("modal-open")&&S.open()}}),document.addEventListener("keydown",u=>{u.key===
"Escape"&&settingsModelPickerTarget&&(u.preventDefault(),window.closeModelModal())});const ti=location.
pathname;Rt[ti]&&(history.replaceState({},"","/"),setTimeout(()=>Rt[ti].open(),500)),get("easy-login\
-generate")&&(get("easy-login-generate").onclick=async()=>{const u=get("easy-login-mins"),f=u?parseInt(
u.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${f}\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))
return;const y=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({minutes:f})})).json();y&&y.temp_password?(get("easy-login-code").textContent=
y.temp_password,get("easy-login-exp").textContent=y.expires_at||"",get("easy-login-result").classList.
remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(f&&f.cancelled){const h=get("easy-login-r\
esult");h&&h.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const _i=4,Si=5*1024*1024;function ni(u,f,h){let y=[];const _=r(()=>{h&&(h.innerHTML="",
y.forEach((S,M)=>{const P=document.createElement("div");P.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const O=document.createElement("img");O.src=S.url,O.alt=S.file.name,
O.className="w-full h-full object-cover";const q=document.createElement("button");q.type="button",q.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",q.textContent=
"\xD7",q.onclick=()=>{URL.revokeObjectURL(S.url),y.splice(M,1),_()},P.appendChild(O),P.appendChild(q),
h.appendChild(P)}))},"render");return u&&f&&(u.onclick=()=>f.click(),f.onchange=S=>{for(const M of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(M.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(M.
size>Si){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(y.
length>=_i){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}y.push({file:M,url:URL.createObjectURL(M)})}S.target.value="",_()}),{files:r(()=>y.
map(S=>S.file),"files"),clear:r(()=>{y.forEach(S=>URL.revokeObjectURL(S.url)),y=[],_()},"clear")}}r(
ni,"createFbImagePicker");async function ii(u,f){try{const h=new FormData;return f.forEach(y=>h.append(
"images",y,y.name)),(await apiFetch(u,{method:"POST",body:h})).ok}catch{return!1}}r(ii,"uploadFbImag\
es");const si=ni(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));function Ci(u){const f=String(
u||"");if(!f)return;const h=`[data-feedback-id="${f.replace(/["\\]/g,"\\$&")}"]`,y=r(()=>document.querySelector(
h),"findRow"),_=r(()=>{const q=y();return q?(typeof jumpToSetting=="function"?jumpToSetting("feedbac\
k",q):(q.scrollIntoView({behavior:"smooth",block:"center"}),q.classList.add("settings-jump-highlight"),
setTimeout(()=>q.classList.remove("settings-jump-highlight"),2e3)),!0):!1},"reveal");if(_())return;const S=get(
"settings-modal");!(S&&(S.classList.contains("modal-open")||S.classList.contains("modal-prep")))&&typeof window.
openSettingsModal=="function"&&window.openSettingsModal();let P=0;const O=setInterval(()=>{P++,(_()||
P>=40)&&clearInterval(O)},150)}r(Ci,"openFeedbackById"),get("fb-submit").onclick=async()=>{const u=get(
"fb-title").value.trim(),f=get("fb-message").value.trim();if(!f){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const h=r(q=>{try{window.ActivityLog&&window.ActivityLog.log("change",{tab:"feedb\
ack",state:q})}catch{}},"logFeedback"),y={title:u,message:f,client:"web",version:window.CHAT_CONFIG&&
window.CHAT_CONFIG.appVersion||""},_=get("fb-attach-logs"),S=_&&_.checked&&window.ActivityLog?window.
ActivityLog.feedbackPayload():null;S&&(y.client_logs=S);const M=get("fb-attach-chat"),P=!!(M&&M.checked);
if(P&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}P&&(y.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const O=typeof window.showTopProgress=="function"?
window.showTopProgress("\u9001\u4FE1\u5185\u5BB9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"):null;
O==null||O.set(8,"\u9001\u4FE1\u5185\u5BB9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"),h("star\
t");try{O==null||O.set(22,"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059"),
O==null||O.ramp(58,"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059");
const q=await apiFetch("/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(y)}),W=q.ok?await q.clone().json().catch(()=>({})):{};O==null||O.set(q.ok?62:92,q.ok?"\u30D5\u30A3\u30FC\u30C9\
\u30D0\u30C3\u30AF\u3092\u53D7\u3051\u4ED8\u3051\u307E\u3057\u305F":"\u9001\u4FE1\u7D50\u679C\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059");
const ie=si.files(),re={count:ie.length,saved:!0};q.ok&&re.count&&(O==null||O.set(68,`\u753B\u50CF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059\uFF08${re.
count}\u679A\uFF09`),O==null||O.ramp(90,`\u753B\u50CF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059\uFF08${re.
count}\u679A\uFF09`),re.saved=await ii(`/api/feedback/${W.public_id}/images`,ie)),O==null||O.set(92,
"\u9001\u4FE1\u7D50\u679C\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059");const pe=q.ok&&W.public_id?
W:null;if(!(window.ActivityLog?await window.ActivityLog.reportFeedback(q,S,P,re,pe?()=>Ci(pe.public_id):
null):q.ok)){h("error"),O==null||O.fail("\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return}
h("done"),O==null||O.done("\u9001\u4FE1\u3057\u307E\u3057\u305F"),get("fb-title").value="",get("fb-m\
essage").value="",si.clear(),M&&(M.checked=!1),_&&(_.checked=!0),hn()}catch{h("error"),O==null||O.fail(
"\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u306E\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};async function ai(u){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${u}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),hn()}r(ai,"deleteFeedback");const Ti="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Hn=[];function oi(u,f){const h=u.replies||[];return h.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${h.map(y=>{const _=y.role==="admin"?f&&y.author?`\u7BA1\u7406\u8005 (${y.
author})`:"\u7BA1\u7406\u8005":f?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=f?[y.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${y.
log_file}`,y.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${y.chat_file}`,y.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${y.image_count}\u679A\uFF09: ${y.image_dir}`]:[y.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${y.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(_)}\
 / ${escapeHtml(y.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${y.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(y.message)}</div>${S.filter(Boolean).map(M=>`<div cla\
ss="${f?"text-amber-300":"text-gray-400"}">${escapeHtml(M)}</div>`).join("")}</div>`}).join("")}</di\
v>`:""}r(oi,"feedbackRepliesHtml");function ri(){return`<div class="fb-reply-form mt-2 space-y-2">
                    <textarea class="fb-reply-message w-full bg-gray-900 border border-gray-700 roun\
ded px-2 py-1 text-xs text-white" rows="2" placeholder="\u8FD4\u4FE1\u5185\u5BB9"></textarea>
                    <div>
                        <input type="file" class="fb-reply-images-input hidden" accept="image/png,im\
age/jpeg,image/webp,image/gif" multiple>
                        <button type="button" class="fb-reply-images-add bg-gray-700 hover:bg-gray-6\
00 text-white rounded px-3 py-1 text-xs"><i class="fas fa-image mr-1"></i>\u753B\u50CF\u3092\u6DFB\u4ED8</button>
                        <span class="text-[10px] text-gray-500 ml-2">PNG\u30FBJPEG\u30FBWebP\u30FBGIF\u30014\u679A\u307E\u3067\u30011\u679A5MB\u307E\u3067<\
/span>
                        <div class="fb-reply-images-list flex flex-wrap gap-2 mt-2"></div>
                    </div>
                    <label class="flex items-start gap-2 cursor-pointer">
                        <input type="checkbox" class="fb-reply-logs accent-amber-500 mt-0.5 shrink-0\
">
                        <span>
                            <span class="block text-xs font-bold text-white">${Ti}</span>
                            <span class="fb-reply-logs-note block text-[10px] text-gray-500 mt-1"></\
span>
                        </span>
                    </label>
                    <label class="flex items-start gap-2 cursor-pointer">
                        <input type="checkbox" class="fb-reply-chat accent-amber-500 mt-0.5 shrink-0\
">
                        <span class="text-xs text-white">\u73FE\u5728\u958B\u3044\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B<span class="block text\
-[10px] text-gray-500">\u958B\u3044\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u3001\u56DE\u7B54\u3001\u601D\u8003\u904E\u7A0B\u3001\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u3001\u30C1\u30E3\u30C3\u30C8\u306E\u8A2D\u5B9A\u3068\u3001\u305D\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u95A2\u3059\u308B\u30A8\u30E9\u30FC\u30ED\u30B0\u306A\u3069\u306E\u8A18\u9332\u3092\u5FA9\u53F7\u3057\u3066\u3001\u4E0D\u5177\u5408\u306E\u8ABF\u67FB\u7528\u306B\u9001\u4FE1\u3057\
\u307E\u3059\u3002\u9001\u4FE1\u3057\u305F\u30B3\u30D4\u30FC\u306F\u3001\u30C1\u30E3\u30C3\u30C8\u3092\u524A\u9664\u3057\u3066\u3082\u6B8B\u308A\u307E\u3059\u3002</span></span>
                    </label>
                    <button type="button" class="fb-reply-send bg-blue-600 hover:bg-blue-500 text-wh\
ite px-3 py-1 rounded text-xs">\u8FD4\u4FE1</button>
                </div>`}r(ri,"feedbackReplyFormHtml");function li(u,f){const h=f.querySelector(".fb-\
reply-form");if(!h)return;const y=h.querySelector(".fb-reply-logs"),_=h.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());y.disabled=!S,_.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const M=ni(h.querySelector(".fb-reply-images-add"),h.querySelector(".fb-reply-images-input"),h.querySelector(
".fb-reply-images-list"));Hn.push(M);const P=h.querySelector(".fb-reply-send");P.onclick=async()=>{const O=h.
querySelector(".fb-reply-message").value.trim();if(!O){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const q={message:O,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},W=y.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;W&&(q.client_logs=W);const ie=h.
querySelector(".fb-reply-chat").checked;if(ie&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}ie&&(q.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const re=M.files();P.disabled=!0;try{const pe=await apiFetch(
`/api/feedback/${u.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(q)}),Z=await pe.json().catch(()=>({}));if(!pe.ok){showToast(Z.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const Se=!re.length||await ii(`/api/feedback/${u.public_id}/replies/${Z.reply_id}\
/images`,re),oe=[];if(W&&Z.logs_saved===!1&&oe.push("\u30ED\u30B0"),Se||oe.push("\u753B\u50CF"),ie&&
Z.chat_copy_saved!==!0&&oe.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),oe.length)showToast(
`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${oe.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,
"error",!0);else{const Oe=["\u8FD4\u4FE1"];ie&&Oe.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),
W&&Oe.push(`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof Z.logs_count=="number"?Z.
logs_count:W.entries.length}\u4EF6\uFF09`),re.length&&Oe.push(`\u753B\u50CF\uFF08${re.length}\u679A\uFF09`),
showToast(Oe.join(Oe.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","succe\
ss")}hn()}finally{P.disabled=!1}}}r(li,"bindFeedbackReply");function ci(){const u=get("fb-admin-stat\
us-filter").value;let f=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(h=>{
const y=u==="all"||(u==="unresolved"?["new","in_review","replied"].includes(h.dataset.feedbackStatus):
h.dataset.feedbackStatus===u);h.hidden=!y,y&&f++}),get("fb-admin-empty").classList.toggle("hidden",f!==
0)}r(ci,"filterAdminFeedback"),get("fb-admin-status-filter").onchange=ci;async function hn(){Hn.forEach(
S=>S.clear()),Hn=[];const f=await(await apiFetch("/api/feedback?all=1")).json(),h=get("fb-list");h.innerHTML=
"",(f.items||[]).filter(S=>!f.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(f.is_admin)
return;const M=document.createElement("div");M.className="p-2 rounded border border-gray-700 bg-gray\
-800/50",S.public_id&&(M.dataset.feedbackId=String(S.public_id)),M.innerHTML=`<div class="text-[11px\
] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(S.public_id||"")}</span> / ${S.
created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div><div class="\
text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class="text-[11px] text-gray-400 mt-\
1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non-released\uFF09":S.status)}\
</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.image_count}\
\u679A</div>`:""}${oi(S,!1)}${ri()}<div class="mt-2"><button type="button" class="fb-delete bg-red-7\
00 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,M.querySelector(
".fb-delete").onclick=()=>ai(S.public_id),li(S,M),h.appendChild(M)});const y=get("fb-admin-panel"),_=get(
"fb-admin-list");f.is_admin?(y.classList.remove("hidden"),_.innerHTML="",(f.items||[]).forEach(S=>{const M=document.
createElement("div");M.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",M.dataset.
feedbackStatus=S.status||"new",S.public_id&&(M.dataset.feedbackId=String(S.public_id)),M.innerHTML=`\

                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(S.public_id||"")}</span> / ${S.agent_name?`agent:${escapeHtml(S.agent_name)}`:`us\
er:${S.user_id}`} / ${S.created_at}${S.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(S.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div>
                            ${[S.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${S.log_file}`,S.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${S.chat_file}`,S.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${S.
image_count}\u679A\uFF09: ${S.image_dir}`].filter(Boolean).map(P=>`<div class="text-[11px] text-ambe\
r-300">${escapeHtml(P)}</div>`).join("")}
                            <div class="flex items-center gap-2">
                                <select class="fb-status bg-gray-900 border border-gray-700 rounded \
px-2 py-1 text-xs text-white">
                                    <option value="new">new</option>
                                    <option value="in_review">in_review</option>
                                    <option value="replied">replied</option>
                                    <option value="rejected">rejected</option>
                                    <option value="resolved">resolved</option>
                                    <option value="resolved_unreleased">resolved\uFF08non-released\uFF09</opti\
on>
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3\
 py-1 rounded text-xs">\u72B6\u614B\u3092\u4FDD\u5B58</button>
                                <button type="button" class="fb-delete bg-red-700 hover:bg-red-600 t\
ext-white px-3 py-1 rounded text-xs">\u524A\u9664</button>
                            </div>
                            ${oi(S,!0)}
                            ${ri()}
                        `,M.querySelector(".fb-status").value=S.status||"new",M.querySelector(".fb-s\
ave").onclick=async()=>{const P=M.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:P})}),
hn()},M.querySelector(".fb-delete").onclick=()=>ai(S.public_id),li(S,M),_.appendChild(M)}),ci()):y.classList.
add("hidden")}if(r(hn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=
f.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get(
"totp-verify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),h=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const y=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),_=await y.json();if(!y.ok){showToast(_.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const M=await(await ensureWebAuthnJson()).create({publicKey:_}),P=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},M,{name:h}))}),O=await P.json().catch(()=>({}));P.ok?(f&&(f.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(O.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(y){showToast(`WebAuthn Error: ${y}`,"error",!0)}finally{u&&(u.disabled=!1)}},window.
removeWebAuthnCredential=async u=>{if(!u||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:u})}),h=await f.json().catch(()=>({}));if(f.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(h.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let u;try{u=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(u.ok){location.href="/";
return}let f={};try{f=await u.json()}catch{}if(f&&f.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(f.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=u=>{if(u.isComposing)return;const f=get("prompt-input");
if(slashSuggestionsVisible){const h=get("slash-command-suggestions");if(u.key==="ArrowDown"){u.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(f.value),f.value));
return}if(u.key==="ArrowUp"){u.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(f.value),f.value));
return}if(u.key==="Enter"){u.preventDefault();const y=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(f.value),f.value));y[slashSelectedIndex]?selectSlashCommand(y[slashSelectedIndex].
id):y.length>0&&selectSlashCommand(y[0].id);return}if(u.key==="Escape"){u.preventDefault(),hideSlashCommandSuggestions();
return}}if(skillSuggestionsVisible){const h=f.value.trim();if(u.key==="ArrowDown"){u.preventDefault(),
skillSelectedIndex=skillSelectedIndex+1,showSkillSuggestions(h.substring(1));return}if(u.key==="Arro\
wUp"){u.preventDefault(),skillSelectedIndex=Math.max(skillSelectedIndex-1,0),showSkillSuggestions(h.
substring(1));return}if(u.key==="Enter"){u.preventDefault();const y=h.substring(1).toLowerCase(),_=loadedSkills.
filter(S=>S.name.toLowerCase().includes(y)||S.description&&S.description.toLowerCase().includes(y));
_[skillSelectedIndex]?selectSkillSuggestion(_[skillSelectedIndex]):_.length>0&&selectSkillSuggestion(
_[0]);return}if(u.key==="Escape"){u.preventDefault(),hideSkillSuggestions();return}}if(u.key==="Esca\
pe"&&pendingSlashCommand){u.preventDefault(),hidePendingSlashCommandIndicator();return}u.key==="Arro\
wUp"&&(f.selectionStart===0||u.ctrlKey)?promptHistory.length>0&&(historyIndex===-1&&(tempPrompt=f.value),
historyIndex<promptHistory.length-1&&(u.preventDefault(),historyIndex++,f.value=promptHistory[historyIndex],
f.dispatchEvent(new Event("input")))):u.key==="ArrowDown"&&(f.selectionEnd===f.value.length||u.ctrlKey)&&
historyIndex>-1&&(u.preventDefault(),historyIndex--,historyIndex===-1?f.value=tempPrompt:f.value=promptHistory[historyIndex],
f.dispatchEvent(new Event("input"))),enterToSend?u.key==="Enter"&&!u.shiftKey&&(u.preventDefault(),sendMessage()):
(u.metaKey||u.ctrlKey)&&u.key==="Enter"&&(u.preventDefault(),sendMessage())},get("prompt-input")&&(get(
"prompt-input").addEventListener("input",function(){this.style.height="auto",this.style.height=this.
scrollHeight+"px",schedulePromptTokenEstimate(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});
const u=this.value.trim();if(pendingSlashCommand)skillSuggestionsVisible&&hideSkillSuggestions(),slashSuggestionsVisible&&
hideSlashCommandSuggestions(),lastSlashFilter=null;else if(u.startsWith("@")){const f=u.substring(1);
showSkillSuggestions(f),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=null}else if(u.
startsWith("/")){const f=slashCommandSuggestionFilter(extractSlashCommandToken(u),this.value);(!slashSuggestionsVisible||
f!==lastSlashFilter)&&(lastSlashFilter=f,showSlashCommandSuggestions(f)),skillSuggestionsVisible&&hideSkillSuggestions()}else
skillSuggestionsVisible&&hideSkillSuggestions(),slashSuggestionsVisible&&hideSlashCommandSuggestions(),
lastSlashFilter=null}),get("prompt-input").addEventListener("blur",()=>{setTimeout(()=>{slashSuggestionsVisible&&
hideSlashCommandSuggestions(),skillSuggestionsVisible&&hideSkillSuggestions()},150)})),get("cancel-e\
dit-btn")&&(get("cancel-edit-btn").onclick=cancelEdit),updatePromptPlaceholder(),aiSettingsConversation.
length>0&&(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),get("search-\
box")&&(get("search-box").addEventListener("input",u=>{const f=get("search-box");if(f&&isUserInitiatedSearchInput(
u))markThreadSearchUserEdited(f);else if(f&&!f.dataset.userEdited){discardAutofilledThreadSearch("cl\
eared-autofill-search-box-input");return}if(isSettingsModalOpen()){snapshotSidebarHistory("ignore-se\
arch-input-settings-open");return}clearTimeout(searchTimeout),searchTimeout=setTimeout(()=>{loadThreads(
!1)},300)}),hardenThreadSearchInputs()),get("mobile-new-chat-btn")&&(get("mobile-new-chat-btn").onclick=
()=>startNewChat()),get("sts-mic-btn")&&(get("sts-mic-btn").onclick=()=>{isStsModel()&&get("mic-btn").
click()}),get("sts-cancel-btn")&&(get("sts-cancel-btn").onclick=()=>{isStsModel()&&Ii()}),get("promp\
t-input")&&get("prompt-input").addEventListener("paste",async u=>{const f=(u.clipboardData||window.clipboardData).
items,h=[];for(let y=0;y<f.length;y++)if(f[y].kind==="file"){const _=f[y].getAsFile();_&&h.push(_)}h.
length>0&&(u.preventDefault(),await handleFiles(h,{openModal:!1}))}),get("rich-paste-btn")&&(get("ri\
ch-paste-btn").onclick=()=>openRichPasteModal()),get("rich-paste-modal-close")&&(get("rich-paste-mod\
al-close").onclick=()=>closeRichPasteModal()),get("rich-paste-close-btn")&&(get("rich-paste-close-bt\
n").onclick=()=>closeRichPasteModal()),get("rich-paste-focus-btn")&&(get("rich-paste-focus-btn").onclick=
()=>focusRichPasteEditor()),get("rich-paste-clear-btn")&&(get("rich-paste-clear-btn").onclick=()=>clearRichPasteEditor(
!0)),get("rich-paste-preview-btn")&&(get("rich-paste-preview-btn").onclick=()=>openRichPastePreviewTab()),
get("rich-paste-send-btn")&&(get("rich-paste-send-btn").onclick=()=>sendRichPasteToModel()),get("ric\
h-paste-send-server-btn")&&(get("rich-paste-send-server-btn").onclick=()=>sendRichPasteToModel({serverSide:!0})),
get("rich-paste-import-btn")&&(get("rich-paste-import-btn").onclick=async()=>{try{await readClipboardRichContent()||
showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u30EA\u30C3\u30C1\u30C6\u30AD\u30B9\u30C8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002Ctrl+V \u3067\u8CBC\u308A\u4ED8\u3051\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(u){const f=u&&u.message?u.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(f,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const u=get("rich-paste-capture");
u.addEventListener("paste",async f=>{const h=f.clipboardData||window.clipboardData;if(h){f.preventDefault();
try{await ingestRichPasteClipboardData(h)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),u.addEventListener("input",()=>{u.value=""})}get("chat-container").addEventListener(
"click",u=>{const f=u.target.closest("img.chat-image"),h=f?f.dataset.viewerSrc||f.currentSrc||f.src:
"";f&&h&&(u.preventDefault(),openImageViewer(h))});const bn=document.querySelector(".viewer-content");
bn&&(bn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),bn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),bn.addEventListener("touchend",onViewerTouchEnd),bn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",u=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(u.target.id==="image-viewer"||u.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",u=>{u.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",u=>{u.key==="Escape"&&closeImageViewer()});
let xt,Ln=[],qn=!1,Mn=null,sn=null,Jt=null,yn=null,Gn=0,An=!1,vn=null,an=null,Bt=null,wn=null,En=null,
Kt=null,$n=null,Pn=null;function di(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray($n)&&
$n.length)return $n;u.innerHTML="";const f=[];for(let h=0;h<24;h++){const y=document.createElement("\
span");y.className="block rounded-full",y.style.background="rgba(252, 165, 165, 0.92)",y.style.width=
"2px",y.style.transition="height 75ms linear, opacity 75ms linear",y.style.height="2px",y.style.opacity=
"0.4",f.push(y),u.appendChild(y)}return $n=f,f}r(di,"ensureMicWaveformBars");function Xt(u,f="hidden"){
const h=get("mic-recording-indicator"),y=get("mic-recording-text");if(h){if(Pn&&(clearTimeout(Pn),Pn=
null),f==="hidden"){h.classList.add("hidden");return}y&&u&&(y.innerText=u),h.classList.remove("hidde\
n"),f==="recording"?h.style.color="rgb(252 165 165)":f==="processing"?h.style.color="rgb(253 224 71)":
h.style.color="rgb(209 213 219)"}}r(Xt,"setMicRecordingIndicator");function ui(){di().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}r(ui,"resetMicWaveformBars");function on(){if(En&&(cancelAnimationFrame(
En),En=null),wn){try{wn.disconnect()}catch{}wn=null}if(an){try{an.close()}catch{}an=null}Bt=null,Kt=
null,ui()}r(on,"stopMicWaveform");function Li(u){on();const f=di();if(!f.length)return;const h=window.
AudioContext||window.webkitAudioContext;if(!h)return;try{an=new h,Bt=an.createAnalyser(),Bt.fftSize=
256,Bt.smoothingTimeConstant=0,wn=an.createMediaStreamSource(u),wn.connect(Bt),Kt=new Uint8Array(Bt.
frequencyBinCount)}catch{on();return}const y=r(()=>{if(!Bt||!Kt)return;Bt.getByteFrequencyData(Kt);const _=Math.
max(1,Math.floor(Kt.length/f.length));for(let S=0;S<f.length;S++){const P=(Kt[Math.min(Kt.length-1,S*
_)]||0)/255,O=Math.max(2,Math.round(2+P*10));f[S].style.height=`${O}px`,f[S].style.opacity=`${.35+P*
.65}`}En=requestAnimationFrame(y)},"render");y()}r(Li,"startMicWaveform");function In(){if(Mn&&(clearInterval(
Mn),Mn=null),yn){try{yn.disconnect()}catch{}yn=null}if(sn){try{sn.close()}catch{}sn=null}Jt=null}r(In,
"stopSilenceMonitor");function Mi(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;In();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;sn=new f,Jt=sn.createAnalyser(),Jt.fftSize=2048,
yn=sn.createMediaStreamSource(u),yn.connect(Jt);const h=new Uint8Array(Jt.fftSize),y=getStsSilenceMs(),
_=.02;Gn=0,An=!1,Mn=setInterval(()=>{if(!Jt)return;Jt.getByteTimeDomainData(h);let S=0;for(let P=0;P<
h.length;P++){const O=(h[P]-128)/128;S+=O*O}if(Math.sqrt(S/h.length)>_){An||(An=!0),Gn=Date.now();return}
An&&Date.now()-Gn>y&&xt&&xt.state==="recording"&&xt.stop()},200)}r(Mi,"startSilenceMonitor");const zn=class zn{constructor(f=24e3){
const h=window.AudioContext||window.webkitAudioContext;this.ctx=new h({sampleRate:f}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(f){if(!this.ctx)return;const h=atob(f),y=new Uint8Array(
h.length);for(let q=0;q<h.length;q++)y[q]=h.charCodeAt(q);const _=new Int16Array(y.buffer),S=new Float32Array(
_.length);for(let q=0;q<_.length;q++)S[q]=_[q]/32768;const M=this.ctx.createBuffer(1,S.length,this.ctx.
sampleRate);M.getChannelData(0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const P=this.
ctx.createBufferSource();P.buffer=M,P.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const O=Math.max(this.ctx.currentTime,this.nextStartTime);
P.start(O),this.nextStartTime=O+M.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r(zn,
"RealTimeAudioPlayer");let xn=zn;const Vn=class Vn{constructor(){this.active=!1,this.capturing=!1,this.
sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=
null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript=
"",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=
null,this.saved=!1,this.saving=!1,this.stopping=!1,this.audioQueue=[],this.audioFlush=null,this.direct=
!1,this.ws=null,this.directReady=null,this.directUrl="",this.directToken="",this.directExpiresAt=0,this.
resumeHandle=null,this.directResumes=0,this.directSetup=null,this.directModel="",this.directTranscribe=
!1,this.directTranslate=!1,this.directThinking=!1,this.transcriptionMode="VERBATIM",this.assistantAudioChunks=
[],this.assistantAudioBytes=0,this.userRecorder=null,this.userAudioChunks=[],this.userRecorderDone=null,
this.userTurnBreak=!1,this.assistantTurnBreak=!1,this.goAwayNotified=!1}isActive(){return this.active}async start(){
if(this.active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const f=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const S=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=S.id!==null&&S.id!==void 0?String(S.id):S.id,setTemporaryChatUiState(!!(S&&S.
is_temporary)),setCurrentChatHeaderTitle(S&&S.title),applyTemporaryChatRuntimeMeta(S||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+S.id),get("welcome-screen").classList.add("hidden")}catch(_){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+_.message,"\
error",!0);return}const h={model:f,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||f==="gpt-realtime-translate")&&get("sts-targ\
et-lang")?get("sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(h.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),isGeminiLiveTranscribeModel()&&(h.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",h.custom_vocabulary=get("sts-\
custom-vocab")?get("sts-custom-vocab").value.split(/[,、\n]/).map(_=>_.trim()).filter(Boolean).slice(
0,1e3):[]),this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.assistantThought="",
this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=null,this.audioQueue=
[],this.audioFlush=null,this.assistantAudioChunks=[],this.assistantAudioBytes=0,this.userAudioChunks=
[],this.userTurnBreak=!1,this.assistantTurnBreak=!1,setStsStatus("\u63A5\u7D9A\u4E2D...",!0);let y=!1;
if(isGeminiLiveModel()&&(y=await this._tryStartDirect(h)),!y){try{const _=await apiFetch("/api/realt\
ime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)}),S=await _.
json().catch(()=>({}));if(!_.ok)throw new Error(S.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=S.session_id,this.rateIn=S.rate_in||this.rateIn,this.rateOut=S.rate_out||this.rateOut}catch(_){
setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
_.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream()}this.active=!0,this.
capturing=!0;try{await this._startCapture()}catch(_){setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",
!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+_.message,"erro\
r",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.
add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",!0)}_buildDirectSetup(f){
const h=f.model,y=this.directTranscribe,_=this.directTranslate,S={responseModalities:y?["TEXT"]:["AU\
DIO"]},M={model:"models/"+h,generationConfig:S,sessionResumption:{}};return y?(M.inputAudioTranscription=
{mode:f.transcription_mode||"VERBATIM",languageCodes:[]},f.custom_vocabulary&&f.custom_vocabulary.length&&
(M.inputAudioTranscription.customVocabulary=f.custom_vocabulary)):(M.inputAudioTranscription={},M.outputAudioTranscription=
{},_?S.translationConfig={targetLanguageCode:normalizeGeminiTargetLang(f.target_lang),echoTargetLanguage:!1}:
f.voice&&(S.speechConfig={voiceConfig:{prebuiltVoiceConfig:{voiceName:f.voice}}}),this.directThinking&&
f.thinking_level&&(S.thinkingConfig={thinkingLevel:f.thinking_level,includeThoughts:!!f.include_thoughts})),
{setup:M}}async _tryStartDirect(f){let h=null;this.directModel=f.model,this.directTranscribe=isGeminiLiveTranscribeModel(),
this.directTranslate=isGeminiLiveTranslateModel(),this.directThinking=isGeminiLiveThinkingModel();const y={
model:f.model};this.directTranscribe?(y.transcription_mode=f.transcription_mode||"VERBATIM",y.custom_vocabulary=
f.custom_vocabulary||[]):(y.voice=f.voice,y.thinking_level=f.thinking_level,y.include_thoughts=f.include_thoughts,
y.target_lang=f.target_lang);try{const _=await apiFetch("/api/gemini/session",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify(y)}),S=await _.json().catch(()=>({}));if(!_.ok||
!S.direct||!S.token)return S.reason!=="vertex_ai"&&console.warn("Gemini Live: browser-direct unavail\
able, using the server session",S.reason||S.error||_.status),!1;h=S}catch(_){return console.warn("Ge\
mini Live: token request failed, using the server session",_),!1}this.rateIn=h.rate_in||16e3,this.rateOut=
h.rate_out||24e3,this.directUrl=h.url,this.directToken=h.token,this.directExpiresAt=Date.parse(h.expires_at||
"")||0,this.directSetup=this._buildDirectSetup(f),this.transcriptionMode=(f.transcription_mode||"VER\
BATIM").toUpperCase(),this.resumeHandle=null,this.directResumes=0,this.goAwayNotified=!1;try{return await this.
_connectDirect(),!0}catch(_){if(console.warn("Gemini Live: direct connection failed, using the serve\
r session",_),this.direct=!1,this.directSetup=null,this.ws){try{this.ws.close()}catch{}this.ws=null}
return this.rateIn=24e3,this.rateOut=24e3,!1}}_connectDirect(){return new Promise((f,h)=>{let y=!1;const _=r(
P=>{y||(y=!0,clearTimeout(S),P?h(P):f())},"finish"),S=setTimeout(()=>_(new Error("\u63A5\u7D9A\u304C\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\u3057\u307E\u3057\u305F")),
12e3);let M;try{M=new WebSocket(this.directUrl+"?access_token="+encodeURIComponent(this.directToken))}catch(P){
_(P);return}this.ws=M,this.direct=!0,this.directReady=_,M.binaryType="arraybuffer",M.onopen=()=>{if(this.
ws!==M)return;const P=JSON.parse(JSON.stringify(this.directSetup));this.resumeHandle&&(P.setup.sessionResumption=
{handle:this.resumeHandle});try{M.send(JSON.stringify(P))}catch(O){_(O)}},M.onmessage=P=>{this.ws===
M&&this._handleDirectMessage(P)},M.onerror=()=>_(new Error("\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
M.onclose=P=>{this.ws===M&&this._directClosed(P)}})}_handleDirectMessage(f){let h=null;try{h=JSON.parse(
typeof f.data=="string"?f.data:new TextDecoder().decode(f.data))}catch{return}if(h.setupComplete!==void 0){
const M=this.directReady;this.directReady=null,M&&M()}const y=h.serverContent;if(y){if(y.modelTurn&&
(y.modelTurn.parts||[]).forEach(M=>{const P=M.inlineData||{};P.data&&this._playDirectAudio(P.data),M.
text&&(M.thought?this.assistantThought+=M.text:this.directTranscribe||this._appendDirectAssistant(M.
text))}),y.outputTranscription&&y.outputTranscription.text&&this._appendDirectAssistant(y.outputTranscription.
text),y.inputTranscription&&y.inputTranscription.text){const M=String(y.inputTranscription.text);this.
directTranscribe&&this.transcriptionMode==="SMART"?this.userTranscript=M:this.userTranscript+=(this.
userTurnBreak&&this.userTranscript?`
`:"")+M,this.userTurnBreak=!1,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)}
y.turnComplete&&(this.userTurnBreak=!0,this.assistantTurnBreak=!0,this.responseDoneCount+=1)}const _=h.
interactionStatus||h.serverContent&&h.serverContent.interactionStatus;if(_){const M=String(_).toUpperCase();
setStsStatus(M==="IN_PROGRESS"?"\u8003\u3048\u4E2D...":"\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0),M==="IDLE"&&(this.responseDoneCount+=1)}const S=h.sessionResumptionUpdate;S&&S.resumable&&S.newHandle&&
(this.resumeHandle=S.newHandle),h.goAway&&!this.goAwayNotified&&(this.goAwayNotified=!0,showToast("G\
emini Live\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u4E0A\u9650\u304C\u8FD1\u3065\u3044\u3066\u3044\u307E\u3059\u3002\u307E\u3082\u306A\u304F\u81EA\u52D5\u3067\u7D42\u4E86\u3057\u307E\u3059\u3002",
"warning",!0)),h.error&&(this.streamError=h.error&&h.error.message||"Gemini Live \u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1))}_appendDirectAssistant(f){f&&(this.assistantTurnBreak&&this.assistantTranscript&&
!this.assistantTranscript.endsWith(`
`)&&(f=`
`+f),this.assistantTurnBreak=!1,this.assistantTranscript+=f,window.VoiceStudio&&window.VoiceStudio.log(
"assistant",this.assistantTranscript))}_playDirectAudio(f){this.lastAudioAt=Date.now();const h=atob(
f),y=new Uint8Array(h.length);for(let _=0;_<h.length;_++)y[_]=h.charCodeAt(_);this.assistantAudioBytes+
y.length<=20*1024*1024&&(this.assistantAudioChunks.push(y),this.assistantAudioBytes+=y.length),stsOpt(
"sts-auto-play")&&(this.rtPlayer||(this.rtPlayer=new xn(this.rateOut||24e3),Yt=this.rtPlayer),setStsStatus(
"\u518D\u751F\u4E2D...",!0),this.rtPlayer.addChunk(f))}_directClosed(f){const h=this.directReady;if(this.
directReady=null,h&&h(new Error("\u63A5\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F")),this.ws=null,
!(!this.direct||!this.active||this.saved||this.stopping)){if(this.resumeHandle&&(!this.directExpiresAt||
Date.now()<this.directExpiresAt)&&this.directResumes<3){this.directResumes+=1,setStsStatus("\u518D\u63A5\u7D9A\u4E2D...",
!0),this._connectDirect().catch(()=>{this.streamError="Gemini Live \u306E\u518D\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
this.stop()});return}f&&f.code&&f.code!==1e3&&(this.streamError="Gemini Live \u306E\u63A5\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+
(f.reason?": "+f.reason:" (code "+f.code+")")),this.stop()}}_openStream(){const f="/api/realtime/str\
eam?session_id="+encodeURIComponent(this.sessionId),h=window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions({credentials:"include",
signal:this.abortCtrl.signal}):{credentials:"include",signal:this.abortCtrl.signal};fetch(f,h).then(
y=>{if(!y.ok)throw new Error("SSE stream failed ("+y.status+")");this.reader=y.body.getReader(),this.
_readLoop()}).catch(y=>{y&&y.name==="AbortError"||(this.streamError=y&&y.message?y.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",
this.active&&(setStsStatus("\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\
\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F","error",!0)))})}async _readLoop(){const f=new TextDecoder;
let h="";try{for(;this.reader;){const{done:y,value:_}=await this.reader.read();if(y)break;h+=f.decode(
_,{stream:!0});let S;for(;(S=h.indexOf(`

`))>=0;){const M=h.slice(0,S);h=h.slice(S+2);for(const P of M.split(`
`)){if(!P.startsWith("data: "))continue;let O=null;try{O=JSON.parse(P.slice(6))}catch{continue}this.
_handleEvent(O)}}}}catch(y){if(y&&y.name==="AbortError")return;this.active&&(this.streamError=y&&y.message?
y.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
if(f)switch(f.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new xn(this.rateOut||24e3),Yt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(f.data));break;case"transcript":f.role==="user"?(f.cumulative?this.userTranscript=
f.delta:this.userTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
f.role==="assistant"?(this.assistantTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):f.role==="thought"&&(this.assistantThought+=f.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":f.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"interaction_status":this.active&&setStsStatus(f.status==="IN_PROGRESS"?"\u8003\u3048\u4E2D...":
"\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",!0);break;case"notice":f.message&&showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\
\u58F0: "+f.message,"warning",!0);break;case"error":this.streamError=f.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Yt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(fi()).then(f=>{if(this.stream=f,this.direct&&window.MediaRecorder)try{this.
userAudioChunks=[];try{this.userRecorder=new MediaRecorder(f,{audioBitsPerSecond:32e3})}catch{this.userRecorder=
new MediaRecorder(f)}this.userRecorder.ondataavailable=P=>{P.data&&P.data.size&&this.userAudioChunks.
push(P.data)},this.userRecorder.start(500)}catch{this.userRecorder=null}const h=this.rateIn||24e3,y=Ai(
f,h);this.audioCtx=y.ctx;const _=y.source,S=this.audioCtx.sampleRate,M=4096;this.processor=this.audioCtx.
createScriptProcessor(M,1,1),this.processor.onaudioprocess=P=>{if(!this.active||!this.capturing)return;
const O=P.inputBuffer.getChannelData(0),q=Ei(O,S,h);!q||!q.byteLength||this._sendAudio(q)},_.connect(
this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(f){if(this.active){if(this.
direct){if(!this.ws||this.ws.readyState!==WebSocket.OPEN)return;try{this.ws.send(JSON.stringify({realtimeInput:{
audio:{data:pi(new Uint8Array(f)),mimeType:"audio/pcm;rate="+(this.rateIn||16e3)}}}))}catch{}return}
this.sessionId&&(this.audioQueue.push(new Uint8Array(f)),this.audioFlush||(this.audioFlush=this._flushAudio()))}}async _flushAudio(){
try{for(;this.audioQueue.length&&this.sessionId;){const f=this.audioQueue.splice(0,this.audioQueue.length),
h=f.reduce((O,q)=>O+q.byteLength,0),y=new Uint8Array(h);let _=0;f.forEach(O=>{y.set(O,_),_+=O.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),M={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:y.buffer},
P=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(M):M;try{await fetch(S,P)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
userRecorder){const f=this.userRecorder;this.userRecorder=null,this.userRecorderDone=new Promise(h=>{
let y=!1;const _=r(()=>{y||(y=!0,h())},"finish");f.addEventListener("stop",_,{once:!0});try{f.stop()}catch{
_()}setTimeout(_,1500)})}if(this.stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.
stream=null}if(this.audioCtx){try{this.audioCtx.close()}catch{}this.audioCtx=null}In(),on()}async stop(){
if(!this.active)return;if(this.active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\
\u3059...",!0),this.audioFlush)try{await this.audioFlush}catch{}if(this.direct){if(this.ws&&this.ws.
readyState===WebSocket.OPEN)try{this.ws.send(JSON.stringify({realtimeInput:{audioStreamEnd:!0}}))}catch{}}else
try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({session_id:this.sessionId})})}catch{}const f=Date.now(),h=this.responseDoneCount;
let y=this.lastAudioAt;for(;Date.now()-f<2e4&&!(this.responseDoneCount>h||(this.lastAudioAt>y&&(y=this.
lastAudioAt),!this.speechActive&&Date.now()-f>2e3&&Date.now()-y>2500));)await new Promise(_=>setTimeout(
_,250));await this._save()}async _save(){if(!this.saved){this.saved=!0,this.saving=!0;try{if(this.direct)
await this._postDirectSave();else{const f=await apiFetch("/api/realtime/save",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId,thread_id:currentThreadId})}),
h=await f.json().catch(()=>({}));if(!f.ok)throw new Error(h.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
h.thread_id&&!currentThreadId&&(currentThreadId=String(h.thread_id))}if(this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(f){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(f&&f.message?f.message:f),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}async _postDirectSave(){
let f="";if(this.assistantAudioChunks.length){const S=this.assistantAudioChunks.reduce((O,q)=>O+q.byteLength,
0),M=new Uint8Array(S);let P=0;this.assistantAudioChunks.forEach(O=>{M.set(O,P),P+=O.byteLength}),f=
pi(M)}if(this.userRecorderDone)try{await this.userRecorderDone}catch{}const h=this.userAudioChunks.length?
await $i(new Blob(this.userAudioChunks)):"",y=await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,model:this.directModel||
get("model-select").value,user_text:this.userTranscript,assistant_text:this.assistantTranscript,assistant_thought:this.
assistantThought,audio_base64:f,user_audio_base64:h})}),_=await y.json().catch(()=>({}));if(!y.ok)throw new Error(
_.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");_.thread_id&&!currentThreadId&&(currentThreadId=
String(_.thread_id))}_cancel(){if(this.direct){if(this.active=!1,this.ws){try{this.ws.close(1e3)}catch{}
this.ws=null}}else this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-\
Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{});this._cleanup(),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=
!1,this.capturing=!1,this.stopping=!1,this.audioQueue=[],this.ws){try{this.ws.close(1e3)}catch{}this.
ws=null}if(this.direct=!1,this.directReady=null,this.directSetup=null,this.resumeHandle=null,this.assistantAudioChunks=
[],this.assistantAudioBytes=0,this.userAudioChunks=[],this.userRecorderDone=null,this._stopCapture(),
this._stopPlayback(),this.abortCtrl){try{this.abortCtrl.abort()}catch{}this.abortCtrl=null}this.reader=
null,this.sessionId=null;const f=get("mic-btn");f&&(f.classList.remove("bg-red-600","animate-pulse"),
f.classList.add("bg-gray-700"))}};r(Vn,"RealtimeVoiceSession");let Un=Vn;function Ai(u,f){const h=window.
AudioContext||window.webkitAudioContext;if(!h)throw new Error("AudioContext not supported");let y=null;
try{return y=new h({sampleRate:f}),{ctx:y,source:y.createMediaStreamSource(u)}}catch{if(y)try{y.close()}catch{}
return y=new h,{ctx:y,source:y.createMediaStreamSource(u)}}}r(Ai,"openMicAudioSource");function Ei(u,f,h){
let y=u;if(f!==h&&f>0&&h>0){const S=f/h,M=Math.floor(y.length/S),P=new Float32Array(M);for(let O=0;O<
M;O++)P[O]=y[Math.min(Math.floor(O*S),y.length-1)];y=P}const _=new Int16Array(y.length);for(let S=0;S<
y.length;S++){const M=Math.max(-1,Math.min(1,y[S]));_[S]=M<0?M*32768:M*32767}return _.buffer}r(Ei,"p\
cm16FromFloat32");function pi(u){let f="";for(let y=0;y<u.length;y+=32768)f+=String.fromCharCode.apply(
null,u.subarray(y,y+32768));return btoa(f)}r(pi,"bytesToBase64");function $i(u){return new Promise((f,h)=>{
const y=new FileReader;y.onload=()=>f(String(y.result||"").split(",")[1]||""),y.onerror=()=>h(y.error||
new Error("\u97F3\u58F0\u306E\u8AAD\u307F\u53D6\u308A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),y.
readAsDataURL(u)})}r($i,"blobToBase64");const kn=new Un,mi=(()=>{const h={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let y=null,_=null,S=!1,M=null,P=!1,O=0,q=0,W=null,
ie="idle",re=!1,pe=null;const Z=r(K=>document.getElementById(K),"$"),Se=r(K=>{const se=Object.assign(
{},K||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="functi\
on"?window.ProgressSpinner.manualRequestOptions(se):(se.progressSpinner=!1,se)},"noSpinner");function oe(K,se){
ie=se;const Ee=Z("lyria-status-text"),ye=Z("lyria-status-dot");Ee&&(Ee.textContent=K),ye&&(ye.className=
"w-2 h-2 rounded-full inline-block "+(h[se]||h.idle)),$e(),fe()}r(oe,"setStatus");function Oe(){const K=q?
Math.floor((Date.now()-q)/1e3):0,se=String(Math.floor(K/60)).padStart(2,"0"),Ee=String(K%60).padStart(
2,"0");return`${se}:${Ee}`}r(Oe,"formatElapsed");function le(){q||(q=Date.now());const K=Z("lyria-el\
apsed");K&&(K.textContent=Oe()),W||(W=window.setInterval(()=>{const se=Z("lyria-elapsed");se&&(se.textContent=
Oe())},1e3))}r(le,"startElapsedTimer");function Ce(){W&&(window.clearInterval(W),W=null)}r(Ce,"stopE\
lapsedTimer");function $e(){const K=Z("lyria-play-btn"),se=Z("lyria-pause-btn"),Ee=Z("lyria-stop-btn"),
ye=Z("lyria-reset-btn"),Pe=!!y,Ke=ie==="streaming"||ie==="connecting";if(K){K.disabled=re;const dt=K.
querySelector("i");dt&&(dt.className="fas fa-play")}se&&(se.disabled=re||!Ke),Ee&&(Ee.disabled=re||!Pe||
!Ke),ye&&(ye.disabled=re||!Pe||!Ke)}r($e,"updateTransportButtons");function fe(){const K=Z("lyria-sa\
ve-btn");if(!K)return;const se=!!y&&ie!=="idle"&&ie!=="connecting"&&ie!=="error";K.classList.toggle(
"hidden",!se)}r(fe,"updateSaveButton");function ke(K,se){const Ee=Z("lyria-prompt-rows");if(!Ee)return;
const ye=document.createElement("div");ye.className="flex items-center gap-2",ye.innerHTML=`
                        <input type="text" value="${escapeHtml(K||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof se=="nu\
mber"?se:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof se=="number"?se:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Pe=ye.querySelector('input[type="range"]'),Ke=ye.querySelector(".lyria-w\
eight-label");Pe&&Ke&&Pe.addEventListener("input",()=>{Ke.textContent=parseFloat(Pe.value).toFixed(1)});
const dt=ye.querySelector(".lyria-prompt-remove");dt&&dt.addEventListener("click",()=>{Ee.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ye.remove()}),ye.classList.add("lyria-prompt-row-wrap"),Ee.appendChild(
ye)}r(ke,"addPromptRow");function Ne(){const K=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),se=[];return K.forEach(Ee=>{const ye=Ee.querySelector('input[type="text"]'),Pe=Ee.
querySelector('input[type="range"]'),Ke=(ye?ye.value:"").trim();Ke&&se.push({text:Ke,weight:parseFloat(
Pe?Pe.value:1)||1})}),se}r(Ne,"collectPrompts");function Re(){const K={},se=r(ji=>{const Wn=Z(ji);return Wn&&
Wn.value!==""?parseFloat(Wn.value):void 0},"num"),Ee=se("lyria-bpm");Ee!==void 0&&(K.bpm=Math.round(
Ee));const ye=se("lyria-guidance");ye!==void 0&&(K.guidance=ye);const Pe=se("lyria-density");Pe!==void 0&&
(K.density=Pe);const Ke=se("lyria-brightness");Ke!==void 0&&(K.brightness=Ke);const dt=se("lyria-tem\
perature");dt!==void 0&&(K.temperature=dt);const Qe=Z("lyria-scale");Qe&&Qe.value&&(K.scale=Qe.value);
const pt=Z("lyria-mode");pt&&pt.value&&(K.music_generation_mode=pt.value);const Ct=Z("lyria-mute-bas\
s"),Zt=Z("lyria-mute-drums"),vi=Z("lyria-only-bass-drums");return Ct&&(K.mute_bass=Ct.checked),Zt&&(K.
mute_drums=Zt.checked),vi&&(K.only_bass_and_drums=vi.checked),K}r(Re,"collectConfig");function ft(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([se,Ee])=>{const ye=Z(se),Pe=Z(Ee);!ye||!Pe||ye.addEventListener("input",()=>{const Ke=parseFloat(
ye.value);Pe.textContent=se==="lyria-bpm"?String(Math.round(Ke)):Ke.toFixed(1)})})}r(ft,"bindRangeLa\
bels");function et(){if(M){try{M.close()}catch{}M=null}P=!1,O=0}r(et,"resetPlayback");function St(){
if(S=!1,_&&typeof _.abort=="function")try{_.abort()}catch{}_=null}r(St,"closeStream");async function Ft(){
St(),_=new AbortController,S=!0;try{const K=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(
y)}`,Se({method:"GET",signal:_.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!K.
ok){const Pe=await K.json().catch(()=>({}));throw new Error(Pe.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const se=K.body.getReader(),Ee=new TextDecoder;let ye="";for(;S;){const{done:Pe,value:Ke}=await se.read();
if(Pe)break;ye+=Ee.decode(Ke,{stream:!0});const dt=ye.split(`

`);ye=dt.pop();for(const Qe of dt){const pt=Qe.split(`
`).find(Zt=>Zt.startsWith("data: "));if(!pt)continue;const Ct=pt.slice(6);try{const Zt=JSON.parse(Ct);
Ae(Zt)}catch{}}}}catch(K){if(K&&K.name==="AbortError")return;S&&(oe("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&y&&Ft()},1200))}finally{S=!1}}r(Ft,"openStream");function Ae(K){
if(K&&K.snapshot){const se=K.status;if(se==="error"){oe("\u30A8\u30E9\u30FC","error"),Ce();return}if(se===
"closed"||se==="stopped"){oe("\u7D42\u4E86","closed"),Ce();return}oe(se==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",se==="paused"?"paused":"connecting");return}if(K&&K.audio){oe("\u518D\u751F\u4E2D...",
"streaming"),le(),Te(K.audio);return}if(K&&K.error){oe("\u30A8\u30E9\u30FC: "+K.error,"error"),Ce();
return}if(K&&K.final){oe("\u7D42\u4E86","closed"),Ce(),$e();return}}r(Ae,"handleStreamMessage");function Te(K){
if(!K)return;if(!M){const Qe=window.AudioContext||window.webkitAudioContext;if(!Qe)return;M=new Qe({
sampleRate:48e3}),P=!1,O=0}let se;try{const Qe=atob(K);se=new Uint8Array(Qe.length);for(let pt=0;pt<
Qe.length;pt++)se[pt]=Qe.charCodeAt(pt)}catch{return}const Ee=new Int16Array(se.buffer),ye=Math.floor(
Ee.length/2);if(ye<1)return;const Pe=M.createBuffer(2,ye,48e3);for(let Qe=0;Qe<2;Qe++){const pt=Pe.getChannelData(
Qe);for(let Ct=0;Ct<ye;Ct++)pt[Ct]=Ee[Ct*2+Qe]/32768}M.state==="suspended"&&M.resume();const Ke=M.createBufferSource();
Ke.buffer=Pe,Ke.connect(M.destination),P||(O=M.currentTime+.08,P=!0);const dt=Math.max(M.currentTime,
O);Ke.start(dt),O=dt+Pe.duration}r(Te,"playChunk");async function We(K,se){const Ee=await fetch("/ap\
i/gemini/music/command",Se({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:y,type:K},se||{}))})),ye=await Ee.json().catch(()=>({}));if(!Ee.ok)throw new Error(
ye.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ye}
r(We,"apiCommand");async function ct(){if(re)return;const K=Ne();if(!K.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}re=!0,$e(),oe("\u63A5\u7D9A\u4E2D...","connecting");try{
const se=await fetch("/api/gemini/music/start",Se({method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({weighted_prompts:K,config:Re()})})),Ee=await se.json().catch(()=>({}));
if(!se.ok)throw new Error(Ee.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
y=Ee.session_id,pe=Re(),oe("\u63A5\u7D9A\u4E2D...","connecting"),Ft()}catch(se){oe("\u30A8\u30E9\u30FC: "+
se.message,"error"),showToast("Lyria RealTime: "+se.message,"error",!0)}finally{re=!1,$e()}}r(ct,"st\
artSession");async function gt(K){if(y){re=!0,$e();try{await We("control",{action:K}),K==="PLAY"?oe(
"\u518D\u751F\u4E2D...","streaming"):K==="PAUSE"?oe("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):K==="\
STOP"?oe("\u505C\u6B62\u4E2D","stopped"):K==="RESET_CONTEXT"&&oe("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(se){showToast("Lyria RealTime: "+se.message,"error",!0),oe("\u30A8\u30E9\u30FC: "+
se.message,"error")}finally{re=!1,$e()}}}r(gt,"control");async function Dt(){if(!y)return;const K=Ne();
if(!K.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}re=!0;try{await We("prompts",{weighted_prompts:K}),oe("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
ie==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(se){showToast("Lyria RealTime: "+se.message,"error",!0)}finally{re=!1,$e()}}r(Dt,"a\
pplyPrompts");async function Le(){if(!y)return;const K=Re(),se=pe||{},Ee=K.bpm!==void 0&&K.bpm!==se.
bpm,ye=K.scale!==void 0&&K.scale!==se.scale,Pe=Ee||ye;re=!0;try{await We("config",{config:K,reset_context:Pe}),
pe=K,oe(Pe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",ie==="paused"?"paused":"streaming"),showToast(
Pe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Ke){showToast("Lyria RealT\
ime: "+Ke.message,"error",!0)}finally{re=!1,$e()}}r(Le,"applyConfig");async function qe(){if(y){re=!0,
oe("\u4FDD\u5B58\u4E2D...","connecting"),$e();try{const K=await fetch("/api/gemini/music/save",Se({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:y,thread_id:currentThreadId||
null})})),se=await K.json().catch(()=>({}));if(!K.ok)throw new Error(se.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
oe("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),Ce(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),se.thread_id&&(currentThreadId=String(se.thread_id),history.pushState({},"","/c/"+se.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(se.thread_id||currentThreadId),Qt(
!0)}catch(K){oe("\u30A8\u30E9\u30FC: "+K.message,"error"),showToast("Lyria RealTime: "+K.message,"er\
ror",!0)}finally{re=!1,$e()}}}r(qe,"saveSession");async function De(){if(St(),y)try{await fetch("/ap\
i/gemini/music/cancel",Se({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:y})}))}catch{}y=null,Ce(),et(),oe("\u6E96\u5099\u5B8C\u4E86","idle")}r(De,"cancelSession");
function Je(){const K=Z("lyria-prompt-rows");K&&(K.innerHTML=""),ke("",1),pe=null,q=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ye=>{const Pe=Z(ye);
Pe&&(Pe.value=ye==="lyria-bpm"?"120":ye==="lyria-guidance"?"4":ye==="lyria-temperature"?"1.1":"0.5")});
const se=Z("lyria-scale");se&&(se.value="");const Ee=Z("lyria-mode");Ee&&(Ee.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ye=>{const Pe=Z(ye);Pe&&(Pe.checked=
!1)}),ft()}r(Je,"resetControls");function Qt(K){St(),y&&fetch("/api/gemini/music/cancel",Se({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:y})})).catch(()=>{}),
y=null,S=!1,Ce(),et(),hideModal("lyria-studio-modal")}r(Qt,"closeAndCleanup");function Sn(K){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const se=Z("lyria-studio-modal");if(se&&se.classList.contains("modal-open")&&y){
if(K&&typeof K=="string"){const ye=Z("lyria-prompt-rows");ye&&(ye.innerHTML=""),ke(K,1)}return}if(y&&
De(),Je(),K&&typeof K=="string"){const ye=Z("lyria-prompt-rows");ye&&(ye.innerHTML=""),ke(K,1)}y=null,
S=!1,Ce(),et(),oe("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(Sn,"open");function On(){
const K=Z("lyria-open-studio-btn");K&&K.addEventListener("click",()=>Sn(""));const se=Z("lyria-studi\
o-close");se&&se.addEventListener("click",()=>Qt(!1));const Ee=Z("lyria-play-btn");Ee&&Ee.addEventListener(
"click",()=>{if(!y){ct();return}gt("PLAY")});const ye=Z("lyria-pause-btn");ye&&ye.addEventListener("\
click",()=>gt("PAUSE"));const Pe=Z("lyria-stop-btn");Pe&&Pe.addEventListener("click",()=>gt("STOP"));
const Ke=Z("lyria-reset-btn");Ke&&Ke.addEventListener("click",()=>gt("RESET_CONTEXT"));const dt=Z("l\
yria-add-prompt-btn");dt&&dt.addEventListener("click",()=>ke("",1));const Qe=Z("lyria-apply-prompts-\
btn");Qe&&Qe.addEventListener("click",Dt);const pt=Z("lyria-apply-config-btn");pt&&pt.addEventListener(
"click",Le);const Ct=Z("lyria-save-btn");Ct&&Ct.addEventListener("click",qe),ft(),Je(),window.openLyriaStudio=
Sn}r(On,"init");function Cn(){re||Qt(!1)}return r(Cn,"requestClose"),{init:On,open:Sn,requestClose:Cn}})();
mi.init(),window.closeLyriaStudio=()=>mi.requestClose(),(()=>{let u=null,f=null,h=null;const y="voic\
eDockSettingsOpen",_="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=r(fe=>document.getElementById(fe),"$");function M(){return isStsModel()&&voiceStudioUiEnabled!==!1}
r(M,"isStudioMode");function P(){const fe=get("model-select")?get("model-select").value:"",ke=S("voi\
ce-studio-title");ke&&(fe==="gpt-transcribe"||fe==="gpt-live-transcribe"||fe==="grok-voice-transcrib\
e-2.0-file"?ke.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":fe===
"gemini-3.5-live-translate-preview"?ke.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ke.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(P,"updateTitle");function O(){const fe=S("v\
oice-studio-transcript");fe&&(fe.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${_}</div>`);const ke=S("sts-live-transcript");ke&&(ke.innerHTML="",ke.classList.add("hidden"))}
r(O,"resetTranscript");function q(fe,ke,Ne){const Re=fe.querySelectorAll(".voice-studio-line");let ft=null;
for(let et=Re.length-1;et>=0;et--)if(Re[et].dataset.role===ke){ft=Re[et];break}if(ft)ft.innerHTML=Ne;else{
const et=fe.querySelector(".voice-studio-placeholder");et&&et.remove();const St=document.createElement(
"div");St.className="voice-studio-line",St.dataset.role=ke,St.innerHTML=Ne,fe.appendChild(St)}fe.classList.
remove("hidden"),fe.scrollTop=fe.scrollHeight}r(q,"writeLine");function W(fe,ke){if(!ke||!String(ke).
trim()||!M())return;const ft=`<span class="${fe==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(fe==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
ke)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(et=>q(et,
fe,ft))}r(W,"log");function ie(fe,ke=!0){const Ne=S("sts-panel"),Re=S("sts-settings-toggle");if(Ne&&
Ne.classList.toggle("settings-open",!!fe),Re&&Re.setAttribute("aria-expanded",fe?"true":"false"),ke)
try{localStorage.setItem(y,fe?"1":"0")}catch{}}r(ie,"setSettingsOpen");function re(){try{return localStorage.
getItem(y)==="1"}catch{return!1}}r(re,"readSettingsOpen");let pe=null;function Z(){const fe=get("mod\
el-select")?get("model-select").value:"";fe!==pe&&(pe=fe,O()),ie(re(),!1),P()}r(Z,"syncDock");function Se(){
const fe=S("sts-panel"),ke=S("voice-studio-panel-host");fe&&ke&&fe.parentNode!==ke&&(u=fe.parentNode,
f=fe.nextSibling,ke.appendChild(fe));const Ne=S("file-preview"),Re=S("voice-studio-file-host");Ne&&Re&&
Ne.parentNode!==Re&&(h=Ne.parentNode,Re.appendChild(Ne),Re.classList.remove("hidden"))}r(Se,"movePan\
elIntoModal");function oe(){const fe=S("sts-panel");fe&&u&&fe.parentNode!==u&&(f&&f.parentNode===u?u.
insertBefore(fe,f):u.appendChild(fe));const ke=S("file-preview");ke&&h&&ke.parentNode!==h&&h.appendChild(
ke);const Ne=S("voice-studio-file-host");Ne&&Ne.classList.add("hidden"),u=null,f=null,h=null}r(oe,"m\
ovePanelBack");function Oe(){if(!M())return;Se();const fe=S("sts-panel");fe&&fe.classList.remove("hi\
dden"),P(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}r(Oe,"open");function le(){window.
VoiceStudioOpen=!1,oe(),hideModal("voice-studio-modal")}r(le,"close");function Ce(){window.VoiceStudioOpen&&
le()}r(Ce,"closeIfOpen");function $e(){window.VoiceStudioOpen=!1;const fe=S("voice-studio-open-btn");
fe&&fe.addEventListener("click",()=>Oe());const ke=S("voice-studio-close");ke&&ke.addEventListener("\
click",()=>le());const Ne=S("sts-settings-toggle");Ne&&Ne.addEventListener("click",()=>{const Re=S("\
sts-panel");ie(!(Re&&Re.classList.contains("settings-open")))}),window.VoiceStudio={open:Oe,close:le,
closeIfOpen:Ce,log:W,isStudioMode:M,syncDock:Z},Z()}return r($e,"init"),{init:$e}})().init();let Yt=null;
function Pi(){if(Yt&&(Yt.stop(),Yt=null),vn){try{vn.pause()}catch{}try{vn.src=""}catch{}vn=null}}r(Pi,
"stopStsPlayback");async function Bi(u){Pi();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=
!0,f.playsInline=!0,vn=f,await f.play(),new Promise(h=>{f.onended=()=>h("ended"),f.onerror=()=>h("er\
ror")})}r(Bi,"playStsAudio");function Ii(){if(kn.isActive()){kn._cancel();return}xt&&xt.state==="rec\
ording"&&(qn=!0,xt.stop())}r(Ii,"cancelRecording");function fi(){if(isStsModel())return{audio:!0};const f=navigator.
mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.getSupportedConstraints():
{},h={channelCount:1};return f.echoCancellation&&(h.echoCancellation=!1),f.noiseSuppression&&(h.noiseSuppression=
!1),f.autoGainControl&&(h.autoGainControl=!1),{audio:h}}r(fi,"getMicCaptureConstraints"),get("mic-bt\
n").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(kn.isActive()){get("mic-btn").classList.remove("bg-red-600","animate-pulse"),
get("mic-btn").classList.add("bg-gray-700"),kn.stop();return}if(xt&&xt.state==="recording"){xt.stop(),
get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-\
700"),isStsModel()||Xt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026","processing"),isStsModel()&&setStsStatus(
"Processing...",!0);return}try{if(isStsModel())try{const h=new Audio;h.src="data:audio/wav;base64,Uk\
lGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",h.play().catch(()=>{})}catch{}if(isRealtimeSessionModel()){
await kn.start();return}isStsModel()||(ui(),Xt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const u=await navigator.mediaDevices.getUserMedia(fi());xt=new MediaRecorder(u),Ln=[],qn=!1;const f=isStsModel();
xt.ondataavailable=h=>Ln.push(h.data),xt.onstop=async()=>{if(qn){Ln=[],get("file-preview").classList.
add("hidden"),u.getTracks().forEach(M=>M.stop()),In(),on(),f||(Xt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Pn=setTimeout(()=>Xt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const h=new Blob(Ln,{type:"audio/web\
m"}),y=new File([h],"recording.webm",{type:"audio/webm"}),_=new FormData;_.append("file",y),get("fil\
e-preview").classList.remove("hidden");const S=f;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const oe=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=oe.id!==null&&oe.id!==void 0?String(oe.id):oe.id,setTemporaryChatUiState(!!(oe&&
oe.is_temporary)),setCurrentChatHeaderTitle(oe&&oe.title),applyTemporaryChatRuntimeMeta(oe||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+oe.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeSkill&&(threadSkillMap[currentThreadId]=activeSkill,pendingSkillForNewThread=null),_.append("m\
odel",get("model-select").value),_.append("thread_id",currentThreadId),get("sts-voice")&&_.append("s\
ts_voice",get("sts-voice").value||""),get("sts-reasoning-effort")&&_.append("sts_reasoning_effort",get(
"sts-reasoning-effort").value||""),get("sts-speed")&&_.append("sts_speed",get("sts-speed").value||""),
get("sts-rate-in")&&_.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&_.append(
"sts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&_.append("sts_thinking_leve\
l",get("sts-thinking-level").value||""),get("sts-include-thoughts")&&_.append("sts_include_thoughts",
get("sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const M=await apiFetch(
"/sts",{method:"POST",body:_});if(!M.ok){const Se=await M.json().catch(()=>({}));throw new Error(Se.
error||"Speech-to-speech failed")}const P=M.body.getReader(),O=new TextDecoder;let q="",W=null,ie=null;
stsOpt("sts-auto-play")&&(ie=new xn,Yt=ie),setStsStatus(isTranscriptionModel()?"Transcribing...":"Pr\
ocessing audio...",!0);let re=!0,pe="",Z="";for(;;){const{done:Se,value:oe}=await P.read();if(Se)break;
q+=O.decode(oe,{stream:!0});const Oe=q.split(`
`);q=Oe.pop();for(const le of Oe){if(!le.trim())continue;const Ce=JSON.parse(le);if(Ce.error)throw new Error(
Ce.error);Ce.audio_delta&&ie&&(re&&(setStsStatus("Playing response...",!1),re=!1),await ie.addChunk(
Ce.audio_delta)),Ce.input_delta&&(pe+=Ce.input_delta,window.VoiceStudio&&window.VoiceStudio.log("use\
r",pe)),Ce.transcript_delta&&(Z+=Ce.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("ass\
istant",Z)),(Ce.final||Ce.audio_url)&&(W=Ce)}}window.VoiceStudio&&!pe.trim()&&window.VoiceStudio.log(
"user","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),W&&(W.audio_url||W.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const M=get(
"set-mic-transcribe-mode");if(!!(M&&M.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}_.append("llm_model",get("model-select")&&get("model-select").value||"");const q=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:_})).json();if(q.transcript){const W=get("prompt-inp\
ut");W.value+=(W.value?" ":"")+q.transcript,W.style.height="auto",W.style.height=W.scrollHeight+"px"}else
showToast(q.error||"Transcription failed","error",!0)}}catch(M){showToast("Audio processing error: "+
M.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(M=>M.
stop()),In(),on(),S||Xt("","hidden"),S&&setStsStatus("Tap to speak",!1)}},xt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Xt("\u9332\u97F3\u4E2D\u2026","recording"),Li(u)),Mi(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{on(),isStsModel()||Xt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const _n=r((u,f)=>{if(!u)return;const h=u.querySelector("span");h?h.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),h=get("lib-download-btn"),y=get("lib-attach-btn"),_=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(f&&(f.disabled=u===0,_n(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
h&&(h.disabled=u===0,_n(h,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
y&&(y.disabled=u===0,_n(y,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),_&&(_.disabled=u!==1,_n(_,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=u!==1,_n(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const M=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",M&&u>0)}};function gi(u){lib.
attachMode=!!u}r(gi,"setLibAttachMode");const hi=r((u=!1)=>{gi(u),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(u=!1)=>{window.__libProjectAttachCallback=null,hideModal("lib-modal"),!u&&location.
pathname==="/library"&&history.back()},get("lib-btn").onclick=()=>hi(!1),get("lib-del-btn").onclick=
deleteSelectedFiles,get("lib-download-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),
get("lib-attach-btn")&&(get("lib-attach-btn").onclick=()=>{if(typeof window.__libProjectAttachCallback==
"function"){const u=window.__libProjectAttachCallback;window.__libProjectAttachCallback=null,u(Array.
from(lib.selected||[]));return}attachSelectedLibraryFiles()}),get("lib-rename-btn")&&(get("lib-renam\
e-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").onclick=
()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>hi(!0)),get("\
lib-search")){let u=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||"").
trim(),u&&clearTimeout(u),u=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const u=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=u,get("lib-sort").onchange=()=>{const f=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,f),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0));const Oi=r(()=>{editingSkillUuid=null,get("skill-modal-title").textContent=
"Create New Skill",get("save-skill-btn").innerText="Create Skill",showModal("skill-modal"),get("skil\
l-name").value="",get("skill-desc").value="",get("skill-inst").value="",location.pathname!=="/skill"&&
history.pushState({modal:"skill"},"","/skill")},"openSkillModal");window.closeSkillModal=(u=!1)=>{hideModal(
"skill-modal"),!u&&location.pathname==="/skill"&&history.back()},get("add-skill-btn").onclick=()=>Oi(),
get("save-skill-btn").onclick=async()=>{const u=get("skill-name").value,f=get("skill-desc").value,h=get(
"skill-inst").value;if(u&&h){const y=editingSkillUuid?"PUT":"POST",_=editingSkillUuid?`/api/skills/${editingSkillUuid}`:
CHAT_CONFIG.urls.handleSkills;await apiFetch(_,{method:y,headers:{"Content-Type":"application/json"},
body:JSON.stringify({name:u,description:f,instruction:h})}),window.closeSkillModal(),loadSkills(),editingSkillUuid&&
activeSkill&&activeSkill.uuid===editingSkillUuid&&(activeSkill.name=u,activeSkill.instruction=h,applyActiveSkill(
activeSkill))}else alert("Name and Instruction are required.")},document.addEventListener("click",function(u){
if(u.target.closest(".edit-btn")){const h=u.target.closest(".edit-btn").getAttribute("data-id");beginEditMessage(
h)}if(u.target.closest(".code-toggle")){const f=u.target.closest(".code-toggle"),h=f.closest(".code-\
wrapper");if(!h)return;const y=h.classList.toggle("collapsed");h.setAttribute("data-collapsed",y?"tr\
ue":"false"),f.setAttribute("aria-expanded",y?"false":"true"),f.innerHTML=y?'<i class="fas fa-chevro\
n-down"></i>':'<i class="fas fa-chevron-up"></i>',f.title=y?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",
f.setAttribute("aria-label",y?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080")}if(u.target.closest("\
.download-btn")){const f=u.target.closest(".download-btn"),h=f.getAttribute("data-code"),y=(f.getAttribute(
"data-lang")||"txt").toLowerCase();if(h)try{const _=decodeURIComponent(h),S=new Blob([_],{type:"text\
/plain"}),M=URL.createObjectURL(S),P=document.createElement("a");P.href=M;let q={python:"py",javascript:"\
js",typescript:"ts",markdown:"md",html:"html",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"\
sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"\
go",rust:"rs",ruby:"rb",php:"php",perl:"pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"\
toml",ini:"ini",plaintext:"txt",text:"txt"}[y]||y;(y.length>8||/[^a-z0-9]/.test(y))&&(q="txt");let W=`\
code.${q}`;y==="dockerfile"&&(W="Dockerfile"),y==="makefile"&&(W="Makefile"),P.download=W,document.body.
appendChild(P),P.click(),document.body.removeChild(P),URL.revokeObjectURL(M)}catch(_){console.error(
"Download failed",_)}}if(u.target.closest(".coding-target-btn")&&selectCodingTargetFromButton(u.target.
closest(".coding-target-btn")),u.target.closest(".copy-btn")){const f=u.target.closest(".copy-btn"),
h=f.getAttribute("data-code");h&&window.copyCode(f,h)}if(u.target.closest(".html-preview-btn")){const h=u.
target.closest(".html-preview-btn").getAttribute("data-code");h&&openHtmlCodePreview(h)}if(u.target.
closest(".canvas-preview-btn")){const f=u.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(
f)}}),document.querySelectorAll(".modal-overlay").forEach(u=>{u.addEventListener("click",f=>{f.target===
u&&ei(u.id)})}),currentThreadId?loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){
const e=get("file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-pr\
ogress-bar"),a=get("file-preview-thumbs"),o=get("upload-modal-status-text"),l=get("upload-modal-tota\
l-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;if(a){const T=document.querySelectorAll(
"#upload-list .upload-row");a.innerHTML="",T.forEach((E,N)=>{const U=E.getAttribute("data-local-url"),
L=E.getAttribute("data-filename"),I=E.querySelector("img.upload-preview")!==null;let A;if(I){let j=U;
if(!j&&L){const B=L.replace(/^\d+\//,"");j=buildAttachmentPreviewUrl(B)}j&&(A=document.createElement(
"img"),A.src=j,A.className="thumb-item shadow-sm",A.dataset.viewerSrc=j,A.dataset.viewerFilename=L||
j.split("/").pop(),A.uploadRow=E,A.onclick=function(B){B.preventDefault(),openImageViewer(this.dataset.
viewerSrc,".thumb-item")},A.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}A||(A=
d("FILE")),A.style.animationDelay=`${N*32}ms`,a.appendChild(A)}),T.length>0?a.classList.remove("hidd\
en"):a.classList.add("hidden")}function d(T){const E=document.createElement("div");return E.className=
"thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bol\
d",E.innerText=T,E}r(d,"createFileThumb");const m=collectImageUrlsForSend(),g=uploadProgressState.total,
b=uploadProgressState.completed,v=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.
classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const w=get("send\
-btn"),x=get("mic-btn"),k=get("mask-btn"),C=isStopMode;if(v>0?(w&&(w.disabled=!0),x&&(x.disabled=!0),
k&&(k.disabled=!0)):C||(w&&(w.disabled=!1),x&&(x.disabled=!1),k&&(k.disabled=!1)),v>0){const T=`Prep\
aring... (${b}/${g})`;e.classList.remove("hidden"),t.innerText=T,o&&(o.innerText=`(${b}/${g})`);let E=b*
100,N=0;for(let I in uploadProgressState.perFilePct)E+=uploadProgressState.perFilePct[I],N++;const U=g>
0?E/(g*100)*100:0,L=`${Math.min(100,U)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=L),l&&c&&
(l.classList.remove("hidden"),c.style.width=L)}else o&&(o.innerText=""),l&&l.classList.add("hidden"),
m.length>0?(e.classList.remove("hidden"),t.innerText=`${m.length} files ready`,n&&n.classList.add("h\
idden")):(e.classList.add("hidden"),t.innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}
r(updateFilePreview,"updateFilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get(
"mask-name");!e||!t||(currentMaskImage?(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.
split("/").pop()}`):(e.classList.add("hidden"),t.innerText=""))}r(updateMaskPreview,"updateMaskPrevi\
ew");const markerToolHints={draw:"\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],a=t[3];return`#${n}${n}${i}${i}${a}${a}`}
return"#facc15"}r(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const t=normalizeMarkerHexColor(
e);return{r:parseInt(t.slice(1,3),16),g:parseInt(t.slice(3,5),16),b:parseInt(t.slice(5,7),16)}}r(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,t=60){const n=Number(e),i=Number.isFinite(n)?n:t;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,i))}r(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const t=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(t)?String(t):String(t).replace(/\.0$/,"")}r(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${t})`}r(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),a=get("ma\
rker-color-picker");a&&a.value!==e&&(a.value=e);const o=get("marker-opacity");o&&o.value!==i&&(o.value=
i);const l=get("marker-opacity-number");l&&l.value!==i&&(l.value=i);const c=get("marker-opacity-valu\
e");c&&(c.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(m=>{const g=normalizeMarkerHexColor(m.getAttribute("data-marker-color"));m.classList.
toggle("active",g===e)})}r(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}r(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}r(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const a=get("marker-tool-hi\
nt");a&&(a.textContent=markerState.maskMode?e==="auto"?"\u753B\u50CF\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u8FD1\u4F3C\u8272\u306E\u3064\u306A\u304C\u3063\u305F\u7BC4\u56F2\u3092\u5857\u308A\u307E\u3059":
"\u767D\u304F\u5857\u3063\u305F\u90E8\u5206\u304C\u524A\u9664\u5BFE\u8C61\u306B\u306A\u308A\u307E\u3059 / \u4E8C\u672C\u6307\u3067\u62E1\u5927":
markerToolHints[e]||"");const o=get("marker-crop-reset");o&&o.classList.toggle("hidden",e!=="crop"),
document.querySelectorAll(".marker-edit-only").forEach(T=>T.classList.toggle("hidden",markerState.maskMode));
const l=get("marker-tool-mosaic"),c=get("marker-tool-crop");l&&l.classList.toggle("hidden",markerState.
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const d=get("marker-tool-draw"),m=get(
"marker-tool-auto");d&&(d.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
m&&(m.classList.toggle("hidden",!markerState.maskMode),m.classList.toggle("active",e==="auto")),d&&d.
classList.toggle("active",e==="draw");const g=get("marker-brush-control");g&&g.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const b=get("marker-auto-tolerance-control");b&&b.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const v=get("marker-modal-title"),w=get("marker-modal-de\
scription"),x=get("marker-save");v&&(v.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u4F5C\u6210":
"\u753B\u50CF\u7DE8\u96C6"),w&&(w.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\
\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":
"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
x&&(x.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const k=get("marker-canvas");k&&(k.style.pointerEvents=e==="crop"?"none":"auto");const C=get("marker\
-crop-canvas");C&&(C.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.cropRect||
markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}r(setMarkerMode,
"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}r(clearCropRect,
"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)return;const t=Math.
max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,y:0,w:t,h:n})}r(resetCropRectToFull,
"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.scale=Math.min(markerView.maxScale,
Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.scale<=markerView.minScale+1e-4){
markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marker-stage"),t=get("marker-viewport");
if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(1,e.clientHeight||0),a=Math.max(1,t.
offsetWidth||t.clientWidth||0),o=Math.max(1,t.offsetHeight||t.clientHeight||0);if(n<=1||i<=1||a<=1||
o<=1)return;const l=(n-a)/2,c=(i-o)/2,d=a*markerView.scale,m=o*markerView.scale,g=Math.min(n*.45,Math.
max(24,n*.12)),b=Math.min(i*.45,Math.max(24,i*.12)),v=g-l-d,w=n-g-l,x=b-c-m,k=i-b-c,C=r((T,E,N)=>Number.
isFinite(T)?E>N?(E+N)/2:Math.min(N,Math.max(E,T)):0,"clampOffset");markerView.offsetX=C(markerView.offsetX,
v,w),markerView.offsetY=C(markerView.offsetY,x,k)}r(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
const e=get("marker-viewport");e&&(clampMarkerViewOffset(),e.style.transform=`translate(${markerView.
offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale})`)}r(applyMarkerTransform,"applyMarke\
rTransform");function resetMarkerTransform(){markerView.scale=1,markerView.offsetX=0,markerView.offsetY=
0,applyMarkerTransform()}r(resetMarkerTransform,"resetMarkerTransform");function getRowMarkerKey(e){
return e&&(e.dataset.uploadId||e.getAttribute("data-filename"))||null}r(getRowMarkerKey,"getRowMarke\
rKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(e);n&&(t?markerAppliedUploads.add(n):
markerAppliedUploads.delete(n));const i=e?e.querySelector(".upload-marker-tag"):null;i&&i.classList.
toggle("hidden",!t)}r(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){return markerAppliedUploads.
size>0}r(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){const t=String(e||"").
trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t==="uploaded"?"upload":
"unknown"}r(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let t=String(e).replace(/\u0000/g,"");return t=t.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),t=t.trim(),!t||(t=t.split("/").pop().split("\\").pop().trim(),t=t.replace(/\s{2,}/g,
" "),t=t.replace(/[<>:"/\\|?*]+/g,"_"),!t||t==="."||t==="..")?"":(t.length>180&&(t=t.slice(0,180).trim()),
t)}r(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const t=normalizeAttachmentPath(e);return t?t.split("/").pop()||t:""}r(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,t){const n=normalizeAttachmentPath(
e);if(!n)return;const i=normalizeAttachmentDisplayName(t)||defaultAttachmentDisplayName(n);i&&attachmentNameByPath.
set(n,i)}r(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const t=normalizeAttachmentPath(e);if(!t)return"";const n=normalizeAttachmentDisplayName(attachmentNameByPath.
get(t));return n||defaultAttachmentDisplayName(t)}r(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,t){if(!e)return;const n=normalizeAttachmentDisplayName(t)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=n;const i=e.querySelector(".truncate");
i&&(i.textContent=n);const a=e.getAttribute("data-filename");a&&setAttachmentNameForPath(a,n)}r(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}r(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}r(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
r(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const a=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,a),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}r(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const a=e.querySelector(".truncate"),o=normalizeAttachmentDisplayName(
a?a.textContent:"");return o||getAttachmentNameForPath(t)}r(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,t){const n=normalizeAttachmentPath(e);if(!n)return;const i=normalizeAttachmentSource(
t);i!=="unknown"&&attachmentSourceByPath.set(n,i)}r(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const t=normalizeAttachmentPath(e);return t?normalizeAttachmentSource(
attachmentSourceByPath.get(t)):"unknown"}r(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,t){if(!e)return;const n=normalizeAttachmentSource(t);e.dataset.fileSource=
n;const i=e.getAttribute("data-filename");i&&setAttachmentSourceForPath(i,n)}r(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const t=normalizeAttachmentSource(
e.dataset.fileSource);if(t!=="unknown")return t;const n=e.getAttribute("data-filename");return getAttachmentSourceForPath(
n)}r(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const t=normalizeAttachmentSource(e.dataset.originalSource);if(t!=="unknown")return t;
const n=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(n)}r(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,t,n){const i=document.createElement(
"canvas");i.width=t,i.height=n;const a=i.getContext("2d");a?(a.drawImage(e,0,0,t,n),markerState.baseImageData=
a.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}r(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=r((l,c,d=null,m=!1)=>{if(!l)return;const g=Math.max(0,l.x),b=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);d&&(t.fillStyle=d,t.fillRect(g,b,v,w)),t.save(),m&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(g+.5,b+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),d=Math.max(1,i.w),m=Math.max(1,i.h);t.clearRect(l,c,d,m),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}r(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
r(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=r((a,o,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const d=normalizeAttachmentSource(
o),m=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),g=t.get(c);if(g===void 0){const w=e.
length;t.set(c,w),e.push({path:c,source:d,name:m});return}const b=e[g];if(!b)return;const v=normalizeAttachmentSource(
b.source);(v==="unknown"&&d!=="unknown"||v==="library"&&d==="upload")&&(b.source=d),!normalizeAttachmentDisplayName(
b.name)&&m&&(b.name=m)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const o=a.getAttribute("data-filename");n(o,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}r(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}r(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const o=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const d=isAudioPath(
c),m=isVideoPath(c);if(d&&!t.audio||m&&!t.video){d&&(n+=1),m&&(i+=1);return}o.push(c)}),o.length!==currentImageUrls.
length&&(currentImageUrls=o)}const a=get("upload-list");if(a&&(a.querySelectorAll("[data-filename]").
forEach(o=>{const l=o.getAttribute("data-filename");l&&!currentImageUrls.includes(l)&&(isAudioPath(l)||
isVideoPath(l))&&(setRowMarkerState(o,!1),o.remove())}),a.children.length===0&&(a.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const o=[];n&&o.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&o.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${o.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}r(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const t=e.getAttribute("data-local-url");if(t)return t;const n=e.getAttribute("data-f\
ilename");return n?buildFileUrl(n):""}r(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const t=normalizeAttachmentPath(e);return t?FILE_BASE_URL+t:""}r(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const t=normalizeAttachmentPath(e);return t?isImagePath(t)?FILE_THUMB_BASE_URL+t:FILE_BASE_URL+t:""}
r(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e,t=!1){
const n=getRowImageSource(e);if(!n){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e,markerState.maskMode=t;const i=e?e.querySelector(".truncate"):null;
markerState.filename=i?i.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=
[],markerState.naturalWidth=0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=
[],markerState.mosaicPreviewRect=null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode(
"draw");const a=get("marker-attach-original");a&&(a.checked=e.dataset.attachOriginal==="1");const o=get(
"marker-image"),l=get("marker-canvas"),c=get("marker-crop-canvas");if(l){const d=l.getContext("2d");
d&&d.clearRect(0,0,l.width,l.height)}if(c){const d=c.getContext("2d");d&&d.clearRect(0,0,c.width,c.height)}
resetMarkerTransform(),showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState(
{modal:"marker"},"","/edit-image"),o&&(o.onload=()=>{if(!get("marker-stage")||!l)return;const m=Math.
max(1,Math.floor(o.clientWidth)),g=Math.max(1,Math.floor(o.clientHeight));l.width=m,l.height=g,l.style.
width=`${m}px`,l.style.height=`${g}px`,l.style.left="0px",l.style.top="0px",c&&(c.width=m,c.height=g,
c.style.width=`${m}px`,c.style.height=`${g}px`,c.style.left="0px",c.style.top="0px"),markerState.naturalWidth=
o.naturalWidth||m,markerState.naturalHeight=o.naturalHeight||g;const b=l.getContext("2d");b&&b.clearRect(
0,0,l.width,l.height),prepareMarkerBaseCanvas(o,m,g),saveMarkerHistory(),markerState.mode==="crop"&&
!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),resetMarkerTransform()},o.src=n)}r(
openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={total:0,completed:0,active:0,
perFilePct:{}};const uploadCancelTokens=new Set;let uploadGeneration=0;function isUploadCancelled(e){
return!!(e&&(e.generation!==void 0&&e.generation!==uploadGeneration||e.uploadId&&uploadCancelTokens.
has(e.uploadId)||e.row&&!e.row.isConnected))}r(isUploadCancelled,"isUploadCancelled");function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
r(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){uploadGeneration++,
browserFastLocalFiles.forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,d=c?c.getAttribute("data-loc\
al-url"):null;d&&URL.revokeObjectURL(d)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=
null,uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.
clear();const e=get("file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");
t&&(t.innerHTML="",t.classList.add("hidden"));const n=get("upload-list");n&&(n.innerHTML='<div class\
="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'),
updateFilePreview(),updateMaskPreview();const i=get("file-input");i&&(i.value="");const a=get("photo\
-input");a&&(a.value="");const o=get("mask-input");o&&(o.value="")}r(resetUploadState,"resetUploadSt\
ate");async function uploadMaskFile(e){if(!e)return!1;const t=new FormData;t.append("file",e);try{const n=await fetch(
CHAT_CONFIG.urls.upload,{method:"POST",body:t}),i=await n.json();if(n.ok&&i.filename)return currentMaskImage=
i.filename,updateMaskPreview(),!0;showToast(i.error||"Mask upload failed","error",!0)}catch{showToast(
"Mask upload failed","error",!0)}return!1}r(uploadMaskFile,"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){
const n=get("camera-status");n&&(n.textContent=e||"",n.classList.toggle("text-red-300",!!t),n.classList.
toggle("text-gray-400",!t))}r(setCameraCaptureStatus,"setCameraCaptureStatus");function updateCameraCapturePendingUi(){
const e=cameraCapturePendingFiles.length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,
t.textContent=e?`\u6DFB\u4ED8 (${e})`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=
e===0||cameraCaptureBusy);const i=get("camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.
forEach((a,o)=>{const l=document.createElement("div");l.className="relative rounded overflow-hidden \
border border-gray-700 bg-black aspect-square",l.innerHTML=`
                        <img src="${a}" alt="capture ${o+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${o+1}</div>
                    `,i.appendChild(l)}),i.classList.toggle("hidden",e===0))}r(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const t=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(t)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}r(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(i=>{try{i.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const t=get("camera-capture-btn");t&&(t.disabled=!0);const n=get(
"camera-switch-btn");n&&(n.disabled=!0)}r(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const t=get("camera-video");if(!t)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const n=get(
"camera-switch-btn");n&&(n.disabled=!0);const i=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let a=null;for(const o of i)
try{const l=await navigator.mediaDevices.getUserMedia(o);cameraCaptureStream=l,t.srcObject=l;try{await t.
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],d=c&&c.getSettings?c.getSettings():{},
m=String(d.facingMode||"").toLowerCase();m==="user"||m==="environment"?cameraCaptureFacingMode=m:cameraCaptureFacingMode=
e;const g=get("camera-capture-btn");return g&&(g.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),l}catch(l){a=l}throw a||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
r(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const t=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(t,!0),showToast(t,"error",!0);const n=get("camera-capture-btn");n&&(n.disabled=
!0);const i=get("camera-attach-btn");i&&(i.disabled=!0)}}r(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const t=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!t&&location.pathname==="/camera"&&history.back()}r(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const t=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=t;try{await startCameraCaptureStream(t)}catch(n){const i=n&&
n.message?n.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
i,!0),showToast(i,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}r(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,t=r(a=>String(a).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}r(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const a=get("camera-attach-btn");a&&(a.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const o=t.getContext("2d");if(!o)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");o.drawImage(e,0,0,t.width,t.height);
const l=await new Promise((d,m)=>{t.toBlob(g=>{g?d(g):m(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),c=new File([l],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(c),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(l)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(o){
const l=o&&o.message?o.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
l,!0),showToast(l,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}r(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),a=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),a&&(a.disabled=!0);const o=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${o.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{if(get("camera-split-enabled").checked)await openImageSplit(
o);else{const c=(await handleFiles(o,{openModal:!1,cameraCapture:!0})).filter(Boolean).length;c>0&&showToast(
`${c}\u679A\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success"),c<o.length&&(showToast(
"\u6DFB\u4ED8\u3067\u304D\u306A\u304B\u3063\u305F\u753B\u50CF\u306F\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u753B\u9762\u304B\u3089\u518D\u8A66\u884C\u3067\u304D\u307E\u3059\u3002",
"warning",!0),openUploadModal())}}catch(l){const c=l&&l.message?l.message:"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(c,"error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.
classList.contains("hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}
r(attachCameraCapturedFiles,"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay==
"function"&&window.hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.
pathname!=="/upload"&&history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-inf\
o");if(e){const n=(get("model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v\
4.1-flash"||n==="deepseek-v4-flash-vision-exp",a=["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4\
.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(n),o=n.includes("deepseek")&&!i||n.startsWith("glm\
-")&&!a;e.classList.toggle("hidden",!o)}_syncVisionModelDisplay()}r(openUploadModal,"openUploadModal");
function _syncVisionModelDisplay(){const e=get("vision-model-display");if(!e)return;const t=currentVisionModel;
if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(a=>{a.id===t&&(n=a.name)})),e.textContent=n}else
e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}r(_syncVisionModelDisplay,"_syncVisionModelDisp\
lay");function _openVisionModelSelector(){window._visionPickerActive=!0,openModelModal(),setTimeout(
()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},50)}r(_openVisionModelSelector,
"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.hideDropOverlay=="function"&&
window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname==="/upload"&&history.back()}
r(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){const e=get("upload-list");
if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(n=>{const i=n.getAttribute(
"data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(n,{source:getAttachmentSourceForPath(
n),displayName:getAttachmentNameForPath(n)})}),e.children.length===0&&(e.innerHTML='<div class="text\
-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
r(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}r(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const a=e.split("/").pop()||e,o=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||a,l=(a.split(".").pop()||"").toLowerCase(),c=["png","jpg","jpeg","webp",
"gif"].includes(l),d=buildFileUrl(e),m=c?buildAttachmentPreviewUrl(e):d,g=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,b=document.createElement("div");b.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",b.dataset.uploadId=g,b.setAttribute("data-filename",e),b.dataset.fileSource=
n,b.dataset.displayName=o,b.dataset.defaultDisplayName=o,b.dataset.sendNameCustomized="";const v=escapeHtml(
o),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",x=c?`<img src="${m}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';b.innerHTML=`
                <div class="flex items-center gap-3">
                    ${x}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${v}</div>
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
            `;const k=b.querySelector(".upload-preview");k&&(k.onclick=()=>openFileViewer(d,getRowAttachmentName(
b)||o));const C=b.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(b));const T=b.
querySelector(".upload-remove");T&&(T.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const N=b.getAttribute("data-filename");N&&(currentImageUrls=currentImageUrls.
filter(U=>U!==N)),setRowMarkerState(b,!1),b.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const E=b.querySelector(".upload-marker");return E&&(E.onclick=()=>openMarkerModalForRow(b)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,o),i.prepend(b),{row:b,bar:b.querySelector(".upload-progress > div"),
status:b.querySelector(".upload-status"),uploadId:g}}r(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const o=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const d=l&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let m="";l?(m=URL.createObjectURL(e),c=`<img src="${m}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${o}">`):(m=URL.createObjectURL(e),c=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${c}
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
            `,m&&i.setAttribute("data-local-url",m);const g=i.querySelector(".upload-preview");g&&(g.
onclick=()=>{const x=i.getAttribute("data-filename"),k=x?buildFileUrl(x):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||x||"";openFileViewer(k,C)});const b=i.
querySelector(".upload-remove");b&&(b.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const x=i.getAttribute("data-local-url");x&&URL.revokeObjectURL(x);const k=i.
getAttribute("data-filename");k&&(currentImageUrls=currentImageUrls.filter(C=>C!==k)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const v=i.querySelector(".upload-marker");v&&(v.onclick=()=>openMarkerModalForRow(i));const w=i.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}r(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(isUploadCancelled(
t)||!e)return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);
try{const i=await apiFetch("/upload/init",{method:"POST",headers:{"Content-Type":"application/json",
"X-CSRF-Token":csrfToken},body:JSON.stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.
ok){const b=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(b,"error",!0),!1}const o=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let b=0;b<c;b++){const v=b*l,w=Math.min(e.size,
v+l),x=e.slice(v,w);if(!await new Promise(C=>{const T=new XMLHttpRequest;T.open("POST","/upload/chun\
k",!0),T.setRequestHeader("X-CSRF-Token",csrfToken),T.upload.onprogress=N=>{if(N.lengthComputable&&t&&
t.bar){const U=v+N.loaded,L=Math.min(100,Math.floor(U/e.size*100));t.bar.style.width=`${L}%`,t.status&&
(t.status.textContent=`${L}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,L)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},T.onload=()=>{T.status>=200&&T.status<300?C(!0):C(!1)},T.
onerror=()=>C(!1);const E=new FormData;E.append("upload_id",o),E.append("index",String(b)),E.append(
"total",String(c)),E.append("chunk",x,e.name),T.send(E)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:o})}),m=await d.json();if(d.ok&&m&&m.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const x=t.row.querySelector(
"img.upload-preview");if(x){const k=m.filename.replace(/^\d+\//,"");x.src=buildAttachmentPreviewUrl(
k)}}const b=normalizeAttachmentPath(m.filename);if(b&&currentImageUrls.push(b),t&&t.row&&(t.row.setAttribute(
"data-filename",b||m.filename),setRowAttachmentSource(t.row,"upload"),b)){const w=isRowAttachmentNameCustomized(
t.row),x=defaultAttachmentDisplayName(b),k=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
x;t.row.dataset.defaultDisplayName=x,setRowAttachmentName(t.row,k)}return b&&setAttachmentSourceForPath(
b,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(m.
filenames)&&m.filenames.length?m.filenames:[m.filename]).forEach(w=>addLibraryFileFromPath(w)),!0}const g=m&&
m.error?m.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(g,"error",!0),!1}catch{return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}r(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t,n={}){return new Promise(i=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(i);return}if(isUploadCancelled(t)){i(!1);return}let a=!1;window.ConnectionMonitor&&
(window.ConnectionMonitor.operationStarted(),a=!0);const o=r(()=>{a&&window.ConnectionMonitor&&(window.
ConnectionMonitor.operationEnded(),a=!1)},"finishUploadOp"),l=new XMLHttpRequest;l.open("POST",CHAT_CONFIG.
urls.upload,!0),l.setRequestHeader("X-CSRF-Token",csrfToken),l.upload.onprogress=d=>{if(d.lengthComputable&&
t&&t.bar){const m=Math.min(100,Math.floor(d.loaded/d.total*100));t.bar.style.width=`${m}%`,t.status&&
(t.status.textContent=`${m}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,m)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},l.onload=()=>{let d={};try{d=JSON.parse(l.responseText||"\
{}")}catch{}if(l.status>=200&&l.status<300&&d&&d.filename){if(isUploadCancelled(t)){t.row&&t.row.parentNode&&
t.row.remove(),o(),i(!1);return}if(t&&t.row){const b=t.row.getAttribute("data-local-url");b&&URL.revokeObjectURL(
b),t.row.removeAttribute("data-local-url");const v=t.row.querySelector("img.upload-preview");if(v){const w=d.
filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(w)}}const m=normalizeAttachmentPath(d.
filename);if(m&&currentImageUrls.push(m),t&&t.row&&(t.row.setAttribute("data-filename",m||d.filename),
setRowAttachmentSource(t.row,"upload"),m)){const b=isRowAttachmentNameCustomized(t.row),v=defaultAttachmentDisplayName(
m),w=b&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||v;t.row.dataset.defaultDisplayName=
v,setRowAttachmentName(t.row,w)}m&&setAttachmentSourceForPath(m,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(d.filenames)&&d.filenames.length?d.filenames:[d.filename]).
forEach(b=>addLibraryFileFromPath(b)),o(),i(!0)}else{const m=[502,503,504].includes(l.status)||l.status>=
520&&l.status<=527;if(n.cameraCapture&&m&&!d.error&&!isUploadCancelled(t)){t&&t.status&&(t.status.textContent=
"\u5206\u5272\u3057\u3066\u518D\u9001\u4E2D..."),o(),uploadFileChunked(e,t).then(i);return}const g=d&&
d.error?d.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(g,"error",!0),o(),i(!1)}},l.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),o(),i(!1)};const c=new FormData;c.append("file",e),l.send(c)})}r(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}r(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}r(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(m=>{n+=m.length});const i=new Float32Array(n);
let a=0;e.forEach(m=>{i.set(m,a),a+=m.length});const o=new ArrayBuffer(44+i.length*2),l=new DataView(
o),c=r((m,g)=>{for(let b=0;b<g.length;b++)l.setUint8(m+b,g.charCodeAt(b))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let d=44;for(let m=0;m<i.length;m++){const g=Math.
max(-1,Math.min(1,i[m]));l.setInt16(d,g<0?g*32768:g*32767,!0),d+=2}return new Blob([l],{type:"audio/\
wav"})}r(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}r(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const o=e.row.getAttribute("data-local-url");o&&URL.revokeObjectURL(
o);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),d=escapeHtml(a),m=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${d}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',g=e.row.querySelector(".upload-preview");g&&(g.outerHTML=m);const b=e.row.querySelector(
".upload-preview");b&&(b.onclick=()=>{const w=e.row.getAttribute("data-filename"),x=w?buildFileUrl(w):
e.row.getAttribute("data-local-url");openFileViewer(x,getRowAttachmentName(e.row)||a||w||"")});const v=e.
row.querySelector(".upload-marker");v&&v.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}r(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const t=e.getContext("2d");if(!t)return;const n=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[];markerState.history.push({imageData:t.
getImageData(0,0,e.width,e.height),mosaicRects:n}),markerState.history.length>40&&markerState.history.
shift()}r(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const t=e.getContext(
"2d");if(!t)return;const n=markerState.history[markerState.history.length-1];t.clearRect(0,0,e.width,
e.height),n&&n.imageData?(t.putImageData(n.imageData,0,0),markerState.mosaicRects=Array.isArray(n.mosaicRects)?
n.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[]):n?(t.putImageData(n,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}r(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const t=e.getContext("2d");t&&t.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}r(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let a=!1,o=0,
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},m=[],g=16,b="",v=null,w=null,x=null,k=null,C=null,T=!1,E=null;
const N=r(G=>{const X=e.getBoundingClientRect(),ce=(G.clientX-X.left)*(e.width/X.width),de=(G.clientY-
X.top)*(e.height/X.height);return{x:ce,y:de}},"getPoint"),U=r((G,X)=>({x:(G.x+X.x)/2,y:(G.y+X.y)/2}),
"getMid"),L=r((G,X)=>Math.hypot(G.x-X.x,G.y-X.y),"getDist");let I=!1;const A=r(()=>{w||(w=document.createElement(
"canvas"),x=w.getContext("2d")),k||(k=document.createElement("canvas"),C=k.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),j=r(()=>{if(!t||!w||!k)return;const G=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=G,t.drawImage(k,0,0),t.restore()},"r\
enderDrawPreview"),B=r(()=>{C&&(C.strokeStyle=b,C.fillStyle=b,C.lineWidth=g,C.lineCap="round",C.lineJoin=
"round")},"applyMarkerBrush"),D=r(G=>{if(!G)return!1;if(m.length===0)return m.push(G),!0;const X=m[m.
length-1],ce=G.x-X.x,de=G.y-X.y,Q=Math.hypot(ce,de),me=Math.max(.35,g*.04);if(Q<me)return!1;const ne=Math.
max(1,g*.25),ve=Math.max(1,Math.ceil(Q/ne));for(let ue=1;ue<=ve;ue++){const ge=ue/ve;m.push({x:X.x+ce*
ge,y:X.y+de*ge})}return!0},"appendStrokePoint"),V=r(()=>{if(C&&(C.clearRect(0,0,k.width,k.height),m.
length!==0)){if(B(),m.length===1){const G=m[0];C.beginPath(),C.arc(G.x,G.y,g/2,0,Math.PI*2),C.fill();
return}if(C.beginPath(),C.moveTo(m[0].x,m[0].y),m.length===2)C.lineTo(m[1].x,m[1].y);else{for(let ce=1;ce<
m.length-2;ce++){const de=m[ce],Q=m[ce+1],me=U(de,Q);C.quadraticCurveTo(de.x,de.y,me.x,me.y)}const G=m[m.
length-2],X=m[m.length-1];C.quadraticCurveTo(G.x,G.y,X.x,X.y)}C.stroke()}},"renderStrokeLayer"),ee=r(
(G,X)=>{if(!G||!X)return null;const ce=Math.min(G.x,X.x),de=Math.min(G.y,X.y),Q=Math.abs(G.x-X.x),me=Math.
abs(G.y-X.y);return{x:ce,y:de,w:Q,h:me}},"normalizeMosaicRect"),$=r(G=>{const X=n?Number(n.value||16):
16,ce=Math.max(6,Math.floor(X)),de=Math.floor(ce/2);return{x:G.x-de,y:G.y-de,w:ce,h:ce}},"buildMosai\
cRectFromPoint"),R=r(()=>{const G=document.createElement("canvas");G.width=e.width,G.height=e.height;
const X=G.getContext("2d");if(!X)return null;markerState.baseCanvas&&X.drawImage(markerState.baseCanvas,
0,0),X.drawImage(e,0,0);try{return X.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),J=r(G=>{if(!t||!G)return!1;const X=R();if(!X)return!1;const ce=n?Number(n.value||
16):16,de=Math.max(4,Math.floor(ce/2)),Q=Math.max(0,Math.floor(G.x)),me=Math.max(0,Math.floor(G.y)),
ne=Math.min(e.width,Math.ceil(G.x+G.w)),ve=Math.min(e.height,Math.ceil(G.y+G.h));if(ne<=Q||ve<=me)return!1;
for(let ue=me;ue<ve;ue+=de)for(let ge=Q;ge<ne;ge+=de){const je=Math.min(de,ne-ge),Me=Math.min(de,ve-
ue),lt=Math.min(e.width-1,Math.max(0,ge+Math.floor(je/2))),we=(Math.min(e.height-1,Math.max(0,ue+Math.
floor(Me/2)))*e.width+lt)*4,be=X.data[we],ae=X.data[we+1],it=X.data[we+2];t.fillStyle=`rgb(${be},${ae}\
,${it})`,t.fillRect(ge,ue,je,Me)}return!0},"applyMosaicRect"),F=r(G=>{if(!t)return;if(i.set(G.pointerId,
{x:G.clientX,y:G.clientY}),i.size>=2){const ce=Array.from(i.values()),de=ce[0],Q=ce[1];a=!0,I=!1,m=[],
T=!1,E=null,v=null,markerState.mosaicPreviewRect=null,o=L(de,Q)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=U(de,Q),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
G.pointerId),G.preventDefault();return}if(a||markerState.mode==="crop")return;I=!0;const X=N(G);if(markerState.
mode==="auto"&&markerState.maskMode)v=X;else if(markerState.mode==="mosaic")T=!0,E=X,markerState.mosaicPreviewRect=
$(X),renderCropOverlay();else{if(A(),!x||!C)return;x.clearRect(0,0,w.width,w.height),x.drawImage(e,0,
0),C.clearRect(0,0,k.width,k.height),g=n?Number(n.value||16):16,b=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),m=[],D(X),V(),markerState.hasStroke=!0,j()}e.setPointerCapture&&e.setPointerCapture(
G.pointerId),G.preventDefault()},"start"),H=r(G=>{if(i.has(G.pointerId)&&i.set(G.pointerId,{x:G.clientX,
y:G.clientY}),a&&i.size>=2){const ce=Array.from(i.values()),de=ce[0],Q=ce[1],me=U(de,Q),ne=L(de,Q)||
1,ve=l*(ne/o);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,ve)),markerView.
offsetX=c.x+(me.x-d.x),markerView.offsetY=c.y+(me.y-d.y),applyMarkerTransform(),G.preventDefault();return}
if(!I||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){G.preventDefault();return}const X=N(
G);if(markerState.mode==="mosaic"){if(!T||!E)return;markerState.mosaicPreviewRect=ee(E,X)||$(X),renderCropOverlay()}else
D(X)&&(V(),j());G.preventDefault()},"move"),te=r(G=>{const X=I;if(i.delete(G.pointerId),i.size<2&&(a=
!1),i.size===0){I=!1;const ce=X&&G.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.maskMode&&
v?fillMaskSimilarRegion(v):!1;if(v=null,X&&t&&markerState.mode==="draw"&&m.length>0&&(V(),j()),X&&markerState.
mode==="mosaic"&&E){const de=N(G);let Q=ee(E,de);(!Q||Q.w<2||Q.h<2)&&(Q=$(E)),J(Q)&&(markerState.hasStroke=
!0,markerState.mosaicRects.push(Q))}m=[],T=!1,E=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),
X&&(markerState.mode!=="auto"||ce)&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(
G.pointerId),G.preventDefault()},"end");e.addEventListener("pointerdown",F),e.addEventListener("poin\
termove",H),e.addEventListener("pointerup",te),e.addEventListener("pointercancel",te)}r(initMarkerCanvas,
"initMarkerCanvas");function fillMaskSimilarRegion(e){var I;const t=get("marker-canvas"),n=markerState.
baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,o=Math.floor(e.x),l=Math.floor(e.y);if(o<0||l<0||o>=i||l>=
a)return!1;const c=n.data,d=l*i+o,m=d*4;if(c[m+3]<16)return!1;const g=c[m],b=c[m+1],v=c[m+2],w=Number(
((I=get("marker-auto-tolerance"))==null?void 0:I.value)||32),x=new Uint8Array(i*a),k=[d],C=[],T=25e4,
E=r(A=>{const j=A*4;return c[j+3]>=16&&Math.max(Math.abs(c[j]-g),Math.abs(c[j+1]-b),Math.abs(c[j+2]-
v))<=w},"matches");for(x[d]=1;k.length;){const A=k.pop();if(x[A]===2||!E(A))continue;const j=Math.floor(
A/i);let B=A%i,D=B;for(;B>0&&x[j*i+B-1]!==2&&E(j*i+B-1);)B--;for(;D+1<i&&x[j*i+D+1]!==2&&E(j*i+D+1);)
D++;for(let V=B;V<=D;V++)x[j*i+V]=2;if(C.push(j,B,D+1),C.length/3>T)return showToast("\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\
\u3066\u304F\u3060\u3055\u3044","error",!0),!1;for(const V of[j-1,j+1]){if(V<0||V>=a)continue;let ee=B;
for(;ee<=D;){const $=V*i+ee;if(x[$]===0&&E($))for(x[$]=1,k.push($);ee+1<=D&&x[V*i+ee+1]===0&&E(V*i+ee+
1);)ee++,x[V*i+ee]=1;ee++}}}if(!C.length)return!1;const N=t.getContext("2d");if(!N)return!1;const U=N.
getImageData(0,0,i,a),L=U.data;for(let A=0;A<C.length;A+=3){const j=C[A]*i+C[A+1],B=C[A]*i+C[A+2];for(let D=j;D<
B;D++){const V=D*4;L[V]=L[V+1]=L[V+2]=L[V+3]=255}}return N.putImageData(U,0,0),markerState.hasStroke=
!0,!0}r(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,o=null,l=null,c=!1,
d=0,m=markerView.scale,g={x:0,y:0},b={x:0,y:0};const v=8,w=14,x=r((A,j,B)=>Math.min(B,Math.max(j,A)),
"clamp"),k=r(A=>{const j=e.getBoundingClientRect(),B=(A.clientX-j.left)*(e.width/j.width),D=(A.clientY-
j.top)*(e.height/j.height);return{x:B,y:D}},"getPoint"),C=r((A,j)=>({x:(A.x+j.x)/2,y:(A.y+j.y)/2}),"\
getMid"),T=r((A,j)=>Math.hypot(A.x-j.x,A.y-j.y),"getDist"),E=r(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),N=r((A,j)=>{if(!j)return"move";const B=j.x,D=j.y,V=j.x+j.w,ee=j.
y+j.h,$=Math.abs(A.x-B)<=w,R=Math.abs(A.x-V)<=w,J=Math.abs(A.y-D)<=w,F=Math.abs(A.y-ee)<=w;if($&&J)return"\
nw";if(R&&J)return"ne";if($&&F)return"sw";if(R&&F)return"se";if(J)return"n";if(F)return"s";if($)return"\
w";if(R)return"e";if(A.x>B+w&&A.x<V-w&&A.y>D+w&&A.y<ee-w)return"move";const te=A.x<B?"left":A.x>V?"r\
ight":null,G=A.y<D?"top":A.y>ee?"bottom":null;if(te&&G){if(te==="left"&&G==="top")return"nw";if(te===
"right"&&G==="top")return"ne";if(te==="left"&&G==="bottom")return"sw";if(te==="right"&&G==="bottom")
return"se"}return te?te==="left"?"w":"e":G?G==="top"?"n":"s":"move"},"hitTest"),U=r(A=>{if(markerState.
mode!=="crop")return;if(n.set(A.pointerId,{x:A.clientX,y:A.clientY}),n.size>=2){const D=Array.from(n.
values()),V=D[0],ee=D[1];c=!0,i=!1,d=T(V,ee)||1,m=markerView.scale,g={x:markerView.offsetX,y:markerView.
offsetY},b=C(V,ee),e.setPointerCapture&&e.setPointerCapture(A.pointerId),A.preventDefault();return}if(c)
return;i=!0;const j=k(A),B=E();o=N(j,B),a=j,l=B?{x:B.x,y:B.y,w:B.w,h:B.h}:null,renderCropOverlay(),e.
setPointerCapture&&e.setPointerCapture(A.pointerId),A.preventDefault()},"start"),L=r(A=>{if(markerState.
mode!=="crop")return;if(n.has(A.pointerId)&&n.set(A.pointerId,{x:A.clientX,y:A.clientY}),c&&n.size>=
2){const te=Array.from(n.values()),G=te[0],X=te[1],ce=C(G,X),de=T(G,X)||1,Q=m*(de/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Q)),markerView.offsetX=g.x+(ce.x-b.x),markerView.
offsetY=g.y+(ce.y-b.y),applyMarkerTransform(),renderCropOverlay(),A.preventDefault();return}if(!i||!a||
!l)return;const j=k(A),B=e.width,D=e.height,V={x:l.x,y:l.y,w:l.w,h:l.h},ee=l.x+l.w,$=l.y+l.h,R=r(()=>{
const te=x(j.x,0,ee-v);V.x=te,V.w=ee-te},"applyW"),J=r(()=>{V.w=x(j.x-l.x,v,B-l.x)},"applyE"),F=r(()=>{
const te=x(j.y,0,$-v);V.y=te,V.h=$-te},"applyN"),H=r(()=>{V.h=x(j.y-l.y,v,D-l.y)},"applyS");switch(o){case"\
move":{const te=j.x-a.x,G=j.y-a.y;V.x=x(l.x+te,0,B-l.w),V.y=x(l.y+G,0,D-l.h);break}case"w":R();break;case"\
e":J();break;case"n":F();break;case"s":H();break;case"nw":F(),R();break;case"ne":F(),J();break;case"\
sw":H(),R();break;case"se":H(),J();break;default:break}V.x=x(V.x,0,B-V.w),V.y=x(V.y,0,D-V.h),markerState.
cropRect=V,renderCropOverlay(),A.preventDefault()},"move"),I=r(A=>{n.delete(A.pointerId),n.size<2&&(c=
!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,o=null,l=null),e.releasePointerCapture&&e.releasePointerCapture(
A.pointerId),A.preventDefault()},"end");e.addEventListener("pointerdown",U),e.addEventListener("poin\
termove",L),e.addEventListener("pointerup",I),e.addEventListener("pointercancel",I),e.addEventListener(
"pointerleave",I)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){const e=markerState.
row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get("marker-attach-ori\
ginal");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement("canvas");const o=markerState.
naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||n.height;if(markerState.
maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const C=document.createElement("canvas");C.width=o,C.height=l;const T=C.getContext(
"2d");if(!T)return;T.fillStyle="#000000",T.fillRect(0,0,o,l),T.drawImage(n,0,0,o,l);const E=await new Promise(
L=>C.toBlob(L,"image/png"));if(!E){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const N=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([E],`${N}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=o,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,o,l),c.drawImage(n,0,0,o,l),markerState.cropRect){const C=o/n.width,T=l/n.height,E=Math.max(0,Math.
floor(markerState.cropRect.x*C)),N=Math.max(0,Math.floor(markerState.cropRect.y*T)),U=Math.min(o,Math.
max(1,Math.floor(markerState.cropRect.w*C))),L=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*T))),I=document.createElement("canvas");I.width=U,I.height=L;const A=I.getContext("2d");A&&(A.drawImage(
a,E,N,U,L,0,0,U,L),a=I)}const d=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const g=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),b=new File([d],`${g}_marked.png`,
{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(v,b);const w=e.getAttribute("data-filename"),x=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=x,setAttachmentSourceForPath(
w,x)),await uploadFileWithProgress(b,v)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}r(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,o=null,l=null,c=null,d=[],m=null;const g=r(()=>{m&&clearTimeout(
m);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(o)try{o.close()}catch{}},"cleanup"),b=r(
()=>{g(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();
if(!v||!v.length)return b();o=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=o.createMediaStreamSource(new MediaStream(v)),l=o.createScriptProcessor(4096,1,1),l.onaudioprocess=
x=>{const k=x.inputBuffer.getChannelData(0);d.push(new Float32Array(k))},c.connect(l),l.connect(o.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(m=setTimeout(()=>{const x=(e.
name||"video").replace(/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{
type:"audio/wav"});g(),n(C)},w+250)),await i.play(),i.onended=()=>{const x=(e.name||"video").replace(
/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{type:"audio/wav"});g(),
n(C)}}catch{b()}},i.onerror=()=>b()}))}r(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
if(!e||!e.length)return;const n=Array.from(e).filter(Boolean);if(!n.length)return;const i=n,a=uploadGeneration;
t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=i.length,uploadProgressState.
active+=i.length,updateFilePreview();const o=!!(get("upload-audio-only")&&get("upload-audio-only").checked),
l=getModelMediaSupport(get("model-select").value),c=r(async v=>{let w=null;if(a!==uploadGeneration)return!1;
try{if(isAudioFile(v)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(v)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!v.type||!v.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const x=addUploadRow(v);x.generation=a,updateFilePreview(),w=x.uploadId,uploadProgressState.
perFilePct[w]=0;let k=v;if(o&&isVideoFile(v)){const T=await extractAudioFromVideo(v,x);T?(k=T,updateUploadRowFile(
x,T),x&&x.status&&(x.status.textContent="\u97F3\u58F0\u306E\u307F")):(x&&x.status&&(x.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{k=await applyImageCompression(v,
readComposerCompression()),k!==v&&updateUploadRowFile(x,k)}catch{}if(isUploadCancelled(x))return!1;if(browserFastModeEnabled){
const T=Array.from(browserFastLocalFiles.values()).reduce((E,N)=>E+Number(N.file&&N.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||T+k.size>BROWSER_FAST_MAX_BYTES?(x&&x.status&&
(x.status.textContent="\u4E0A\u9650\u8D85\u904E"),x&&x.row&&x.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(x.uploadId,{file:k,rowObj:x}),
x.status&&(x.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
x.bar&&(x.bar.style.width="100%"),x.row&&(x.row.dataset.browserFastLocal="1"),!0)}const C=await uploadFileWithProgress(
k,x,t);if(!C&&t.cameraCapture&&!isUploadCancelled(x)){const T=document.createElement("button");T.className=
"text-xs border border-gray-700 rounded px-2 py-1",T.textContent="\u518D\u8A66\u884C",T.onclick=()=>{
if(isUploadCancelled(x))return;T.disabled=!0;const E=x.row.getAttribute("data-local-url");E&&URL.revokeObjectURL(
E),x.row.remove(),handleFiles([k],{...t,openModal:!0,skipCompression:!0})},x.row.appendChild(T)}return C}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),d=new Array(i.length).fill(!1);let m=0;const g=Math.
min(UPLOAD_CONCURRENCY,i.length),b=Array.from({length:g}).map(async()=>{for(;;){const v=m++;if(v>=i.
length)break;d[v]=await c(i[v])}});return await Promise.all(b),d}r(handleFiles,"handleFiles"),get("c\
lear-file-btn").onclick=e=>{e.stopPropagation(),resetUploadState()},get("clear-mask-btn")&&(get("cle\
ar-mask-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-in\
put")&&(get("mask-btn").onclick=()=>{if(isIdeogramObjectRemoval()){const e=Array.from(document.querySelectorAll(
"#upload-list .upload-row")).find(t=>getRowImageSource(t));if(!e){showToast("\u5148\u306B\u30DE\u30B9\u30AF\u5BFE\u8C61\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}openMarkerModalForRow(e,!0);return}get("mask-input").click()},get("mask-input").addEventListener(
"change",async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value=
"")}));const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){
const n=String(e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||
typeof window.DOMPurify.sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=
!0,console.error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).
replace(/\n/g,"<br>");const i=protectMathSegments(n);if(t.inline===!0&&typeof window.marked.parseInline==
"function")return window.DOMPurify.sanitize(restoreMathSegments(window.marked.parseInline(i.text),i.
blocks,t));markdownOpenFenceBody=/<\/svg/i.test(n)?findOpenMarkdownFenceBody(n):null;let o="";try{o=
window.marked.parse(i.text)}finally{markdownOpenFenceBody=null}const l=restoreMathSegments(o,i.blocks,
t);return window.DOMPurify.sanitize(l)}r(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");let markdownOpenFenceBody=null;
function findOpenMarkdownFenceBody(e){const t=String(e||"").split(/\r?\n/);let n=null,i=[];for(const a of t){
if(!n){const l=a.match(/^\s*(`{3,}|~{3,})(.*)$/);l&&!(l[1][0]==="`"&&l[2].includes("`"))&&(n=l[1],i=
[]);continue}const o=a.trim();if(o.length>=n.length&&o[0]===n[0]&&/^(`+|~+)$/.test(o)){n=null;continue}
i.push(a)}return n?i.join(`
`):null}r(findOpenMarkdownFenceBody,"findOpenMarkdownFenceBody");const SVG_CODE_RENDER_MAX_CHARS=3e5;
function getRenderableSvgCode(e,t){const n=String(e||"").trim().toLowerCase();if(n!=="svg"&&n!=="xml"&&
n!=="image/svg+xml"&&n!=="mathsvg")return"";const i=String(t||"").trim();if(!i||i.length>SVG_CODE_RENDER_MAX_CHARS)
return"";const a=i.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i,"");return!/^<svg[\s>]/i.
test(a)||!/<\/svg\s*>$/i.test(a)||markdownOpenFenceBody!==null&&markdownOpenFenceBody.replace(/\s+/g,
"")===i.replace(/\s+/g,"")?"":i}r(getRenderableSvgCode,"getRenderableSvgCode");
