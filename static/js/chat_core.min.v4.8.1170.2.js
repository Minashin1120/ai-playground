document.addEventListener("DOMContentLoaded",()=>{var Si,Ci;initThemeFromServer(),applyLiquidGlassMode(
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
getBoundingClientRect(),h=u.getBoundingClientRect(),b=h.left+h.width/2-(f.left+f.width/2),_=h.top+h.
height/2-(f.top+f.height/2);c.style.transform=`translate(${b}px, ${_}px) scale(0.1)`,c.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),c.remove()},800)}else c.style.opacity="0",setTimeout(()=>c.remove(),1e3)}},3e3);function d(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const h=(get("model-select").value||"").toLowerCase().includes("gpt-image-2.6"),b=get("gpt-im\
age-format");get("gpt-image-complexity-wrap")&&get("gpt-image-complexity-wrap").classList.toggle("hi\
dden",!h),b&&(Array.from(b.options).forEach(T=>{T.hasAttribute("data-gpt-image-26")&&(T.hidden=!h)}),
!h&&b.value==="gif"&&(b.value="jpeg"));const _=h&&b&&b.value==="gif",S=get("gpt-image-compression-wr\
ap");b&&S&&(b.value==="png"||b.value==="gif"?S.classList.add("hidden"):S.classList.remove("hidden")),
get("gpt-image-gif-wrap")&&get("gpt-image-gif-wrap").classList.toggle("hidden",!_)}r(d,"updateGptIma\
geUi");function m(){const u=get("gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.
remove("hidden"):u.classList.add("hidden");const h=(get("model-select").value||"").toLowerCase().includes(
"gemini-3.1-flash-lite-image");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{
b&&(Array.from(b.options).forEach(_=>{_.value!=="1K"&&(_.disabled=h)}),h&&b.value!=="1K"&&(b.value="\
1K"))})}r(m,"updateGeminiImageUi");function g(){const u=get("grok-image-options");if(!u)return;const f=(get(
"model-select").value||"").toLowerCase(),h=isGrokImageModel(),b=f==="grok-imagine-image-quality"||f===
"grok-imagine-image-2.0",_=f==="grok-imagine-image-2.0";if(h){u.classList.remove("hidden");const T=get(
"grok-image-resolution")?get("grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidd\
en",!b);const A=get("grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.
toggle("hidden",!_)}else u.classList.add("hidden");if(get("modal-grok-image-options")){const T=get("\
modal-grok-image-resolution")?get("modal-grok-image-resolution").parentElement:null;T&&T.classList.toggle(
"hidden",!b);const A=get("modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;
A&&A.classList.toggle("hidden",!_)}}r(g,"updateGrokImageUi");function y(){var R,F;const u=get("ideog\
ram-image-options"),f=get("modal-ideogram-image-options"),h=get("model-select")&&get("model-select").
value||"",b=isIdeogramModel(h),_=ideogramModelTraits(h),S={resolution:_.resolution,quality:_.quality,
speed:_.speed,style:_.style,negative:_.negative};if(u&&u.classList.toggle("hidden",!b),f&&f.classList.
toggle("hidden",!b),updateIdeogramPromptInputUi(),!b)return;const T=((R=get("ideogram-image-operatio\
n"))==null?void 0:R.value)==="remove_background",A=((F=get("ideogram-image-operation"))==null?void 0:
F.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(Q=>{const oe=Q==="operation"||A&&Q==="seed"||
!(T||A)&&S[Q]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(ce=>{const K=get(ce+Q);!K||!K.
parentElement||K.parentElement.classList.toggle("hidden",!oe)})}),["ideogram-image-style","modal-ide\
ogram-image-style"].forEach(Q=>{const oe=get(Q);if(!oe)return;const ce=_.styleOptions.join("|");if(oe.
dataset.options!==ce){const K=oe.value;oe.innerHTML=_.styleOptions.map(ve=>`<option value="${ve}">${ve===
"auto"?"Auto":ve}</option>`).join(""),oe.dataset.options=ce,oe.value=_.styleOptions.includes(K)?K:"a\
uto"}});const I=get("ideogram-image-note");I&&(I.textContent=T?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWE\
BP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
A?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
_.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
r(y,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(u=>{
var f;(f=get(u))==null||f.addEventListener("change",()=>{const h=get(u.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");h&&(h.value=get(u).value),y(),M()})});function v(){
var b;const u=get("grok-video-options");if(!u)return;const f=String(((b=get("model-select"))==null?void 0:
b.value)||"").toLowerCase();isGrokVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");
const h=get("grok-video-resolution");if(h){const _=Array.from(h.options).find(T=>T.value==="1080p"),
S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(f);_&&(_.disabled=!S),!S&&h.value===
"1080p"&&(h.value="720p")}}r(v,"updateGrokVideoUi");function w(){var _;const u=get("gemini-video-opt\
ions");if(!u)return;const f=String(((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase();
isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");const h=get("gemini-vide\
o-resolution");if(h){const S=Array.from(h.options).find(A=>A.value==="4K"),T=f==="veo-3.1-lite-gener\
ate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-flash";S&&(S.disabled=T),T&&h.value===
"4K"&&(h.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle("hidden",f===
"gemini-omni-1.1-flash")}r(w,"updateGeminiVideoUi");function x(){const u=get("gemini-music-options");
if(!u)return;const f=isGeminiRealtimeMusicModel(),h=isGeminiMusicModel()&&!f;u.classList.toggle("hid\
den",!h);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!f)}r(x,"updateGemi\
niMusicUi");function k(){var T;const u=get("xai-chat-options");if(!u)return;const f=String(((T=get("\
model-select"))==null?void 0:T.value)||"").toLowerCase(),h=f.startsWith("grok-")&&!isGrokImageModel(
f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!h);const b=get("xai-logp\
robs"),_=get("xai-top-logprobs"),S=f.includes("grok-4.20");b&&(b.disabled=S,S&&(b.checked=!1)),_&&(_.
disabled=S,S&&(_.value=""))}r(k,"updateXaiChatUi");function C(){const u=isMistralOcrModel(),f=get("m\
istral-ocr-options");f&&f.classList.toggle("hidden",!u);const h=get("modal-mistral-ocr-options");h&&
h.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const _=get(b);_&&(_.classList.toggle("opacity-50",u),_.classList.toggle("po\
inter-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(C,"updateMistralOcrUi");function L(){const u=get("image-input-limit\
s");if(!u)return;const f=(get("model-select").value||"").toLowerCase();let h="",b=!1;f.includes("gpt\
-image")?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):f==="deepseek-v4.1-flash"||f==="deepseek-v4-flash-vision-exp"?(b=!0,h=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):f.includes("deepseek")||(isGeminiImageModelKey(f)?(b=!0,f.includes("gemini-3.1-flash-lite-\
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
join("")):isMistralOcrModel(f)?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):f.includes("grok")?(b=!0,h=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):isIdeogramModel(f)?(b=!0,h=ideogramModelTraits(f).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u3082\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.5 \u306E\u307F\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u306F\u5168\u30E2\u30C7\u30EB\u3067\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join("")):f.includes("grok")&&f.includes("video")&&(b=!0,h=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(u.innerHTML=h,u.classList.remove("hidden")):(u.classList.add("hidden"),u.innerHTML="")}
r(L,"updateImageInputLimits");function M(){const u=get("enable-sys-prompt"),f=!!(u&&u.disabled),h=!!(u&&
u.checked);G(),u&&(u.disabled?!f&&h?u.dataset.restoreChecked="1":f||delete u.dataset.restoreChecked:
(f&&u.dataset.restoreChecked==="1"&&(u.checked=!0),delete u.dataset.restoreChecked))}r(M,"toggleOpti\
ons"),window.toggleOptions=M;function E(u){const f=get("thinking-options"),h=get("thinking-level"),b=get(
"reasoning-effort"),_=get("enable-thinking"),S=get("thinking-budget-label"),T=get("thinking-budget"),
A=u==="effort";if(f&&(f.dataset.thinkingMode=A?"effort":"thinking"),h&&h.classList.toggle("hidden",A),
b&&b.classList.toggle("hidden",!A),S&&S.classList.toggle("hidden",A),T&&T.classList.toggle("hidden",
A),!!_)if(A){_.dataset.thinkingChecked===void 0&&(_.dataset.thinkingChecked=_.checked?"1":"0");const I=b?
Array.from(b.options).find(F=>F.value==="none"):null,R=!!(I&&!I.disabled&&!I.classList.contains("hid\
den"));_.checked=R?!!(b&&String(b.value||"").toLowerCase()!=="none"):!0,_.disabled=!R}else _.dataset.
thinkingChecked!==void 0&&(_.checked=_.dataset.thinkingChecked==="1",delete _.dataset.thinkingChecked)}
r(E,"setComposerThinkingMode");function G(){const u=get("model-select");if(!u)return;const f=u.value,
h=String(f||"").toLowerCase(),b=h.includes("deepseek"),_=h.startsWith("glm-"),S=get("thinking-option\
s"),T=get("reasoning-effort-container"),A=get("enable-thinking"),I=get("thinking-level"),R=get("thin\
king-budget"),F=get("enable-search"),Q=get("search-container"),oe=get("url-context-container"),ce=get(
"enable-maps"),K=get("maps-grounding-container"),ve=get("enable-sys-prompt"),W=get("sys-prompt-optio\
n"),ge=get("enable-python"),X=get("python-container"),we=get("prompt-cache-container"),Le=get("enabl\
e-prompt-cache"),ae=f==="gpt-5-search-api",ke=f.includes("tts"),Ne=isMistralOcrModel(f),Re=h.includes(
"gemini-3.1-flash-lite-image"),mt=h==="gemini-nano-banana-2.1",Ze=h.includes("gemini-3.1-flash-image")&&
!Re||mt,vt=isClaudeModelKey(f),Bt=h==="gemini-3.8-flash-cyber",Ee=isLlmModel()&&!b&&!ke&&!h.includes(
"realtime")&&!h.includes("native-audio")&&!h.includes("live");we&&(Ee?(we.classList.remove("hidden",
"opacity-50","pointer-events-none"),Le&&(Le.disabled=!1)):(Le&&(Le.checked=!1,Le.disabled=!0),we.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),T&&T.classList.
add("hidden"),E("thinking");const Me=get("vision-model-info");if(Me&&Me.classList.add("hidden"),T){const Te=get(
"reasoning-effort");if(Te){Array.from(Te.options).forEach(De=>{const Ve=h==="gpt-5.6"||h.startsWith(
"gpt-5.6-"),Kt=h.startsWith("gpt-6"),Sn=h==="gpt-6-sol"||h==="gpt-6-luna"||h==="gpt-6.1-terra",Rn=h===
"deepseek-v4.1-flash"||h==="deepseek-v4-flash-0731"||h==="deepseek-v4-flash"||h==="deepseek-v4-flash\
-vision-exp",Cn=h==="deepseek-v4-pro",q=h.includes("grok-4.5"),te=h.includes("grok-4.6");De.value===
"max"?De.classList.toggle("hidden",!Ve&&!Kt&&!Rn&&!Cn):De.value==="xhigh"?De.classList.toggle("hidde\
n",!te&&!h.includes("multi-agent")&&!Ve&&!Kt):De.value==="medium"?De.classList.toggle("hidden",!(h.includes(
"grok-4.3")||q||te||h.includes("grok-4.20-0309-reasoning")||h.includes("grok-build")||h.includes("mu\
lti-agent")||h.includes("gpt-5")||Kt||h.includes("o1")||h.includes("o3"))):De.value==="none"?De.classList.
toggle("hidden",!h.includes("grok-4.3")&&!h.includes("grok-build")&&!h.includes("gpt-5")&&!Sn&&!Rn&&
!Cn):De.value==="low"&&De.classList.toggle("hidden",Cn)});const qe=Te.selectedOptions&&Te.selectedOptions[0];
qe&&qe.classList.contains("hidden")&&(Te.value=b?"high":"medium")}}oe&&oe.classList.add("hidden"),K&&
K.classList.add("hidden"),A&&(A.disabled=!1),R&&(R.disabled=!0,R.classList.add("opacity-50"));const We=isGeminiImageModelKey(
f);if(ke||Ne)Q&&(get("enable-search").checked=!1,Q.classList.add("opacity-50","pointer-events-none")),
oe&&(get("enable-url-context").checked=!1,oe.classList.add("opacity-50","pointer-events-none")),K&&ce&&
(ce.checked=!1,K.classList.add("opacity-50","pointer-events-none")),X&&(ge.checked=!1,X.classList.add(
"opacity-50","pointer-events-none")),ve&&W&&(ve.checked=!1,ve.disabled=!0,W.classList.add("opacity-5\
0"));else if(Ze||Re)K&&ce&&(ce.checked=!1,K.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),mt?(Array.from(I.options).forEach(Te=>{Te.disabled=Te.value==="low"}),[
"minimal","medium","high"].includes(I.value)||(I.value="medium")):(Array.from(I.options).forEach(Te=>{
["low","medium"].includes(Te.value)&&(Te.disabled=!0),["minimal","high"].includes(Te.value)&&(Te.disabled=
!1)}),["minimal","high"].includes(I.value)||(I.value=Re?"minimal":"high")),A&&(A.disabled=!1),Re&&(F&&
(F.checked=!1,F.disabled=!0),Q&&Q.classList.add("opacity-50","pointer-events-none"));else if(We)K&&ce&&
(ce.checked=!1,K.classList.add("hidden","opacity-50","pointer-events-none"));else if(vt)S.classList.
remove("hidden"),R&&(R.disabled=!1,R.classList.remove("opacity-50")),Array.from(I.options).forEach(Te=>{
Te.disabled=!0}),X&&(ge.checked=!1,X.classList.add("opacity-50","pointer-events-none"));else if(Bt){
S&&S.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from(I.options).forEach(qe=>{qe.
disabled=!["low","medium","high"].includes(qe.value)}),["low","medium","high"].includes(I.value)||(I.
value="medium"),[Q,oe,K,X].forEach(qe=>{qe&&qe.classList.add("opacity-50","pointer-events-none")}),[
F,ce,ge].forEach(qe=>{qe&&(qe.checked=!1,qe.disabled=!0)});const Te=get("enable-url-context");Te&&(Te.
checked=!1,Te.disabled=!0),ve&&W&&(ve.disabled=!1,W.classList.remove("opacity-50"))}else if(f.includes(
"gemini")&&!We){S.classList.remove("hidden"),oe&&oe.classList.remove("hidden","opacity-50","pointer-\
events-none");const Te=f.includes("gemini-3")||f.includes("gemini-4");K&&(Te?K.classList.remove("hid\
den","opacity-50","pointer-events-none"):(ce&&(ce.checked=!1),K.classList.add("hidden","opacity-50",
"pointer-events-none")));const qe=f.includes("flash"),De=f==="gemini-4-hydrogen";Array.from(I.options).
forEach(Ve=>{De?Ve.disabled=!["low","medium","high","xhigh"].includes(Ve.value):f==="gemini-4-carbon"||
f==="gemini-4-argon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?Ve.disabled=!["low","medium","h\
igh"].includes(Ve.value):f==="gemini-3.6-flash"?Ve.disabled=!["medium","high"].includes(Ve.value):f===
"gemini-3.5-flash-lite"?Ve.disabled=!["minimal","medium","high"].includes(Ve.value):Ve.value==="xhig\
h"?Ve.disabled=!0:["minimal","medium"].includes(Ve.value)?Ve.disabled=!qe:Ve.disabled=!1}),De&&!["lo\
w","medium","high","xhigh"].includes(I.value)?I.value="high":(f==="gemini-4-carbon"||f==="gemini-4-a\
rgon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium","high"].includes(I.value)||
f==="gemini-3.6-flash"&&!["medium","high"].includes(I.value)?I.value="medium":f==="gemini-3.5-flash-\
lite"&&!["minimal","medium","high"].includes(I.value)?I.value="minimal":!qe&&["minimal","medium"].includes(
I.value)&&(I.value="high"),Te?A&&(A.checked=!0,A.disabled=!0):A&&(A.disabled=!1),R&&f.includes("gemi\
ni-2.5")&&(R.disabled=!1,R.classList.remove("opacity-50")),R&&!f.includes("gemini-2.5")&&(R.disabled=
!0,R.classList.add("opacity-50"))}if(isLlmModel()&&(h.includes("gpt-5")||h.includes("o1")||h.includes(
"o3")||h.includes("grok-4.3")||h.includes("grok-4.5")||h.includes("grok-4.6")||h.includes("grok-4.20\
-0309-reasoning")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt")&&!h.includes(
"tts")))S&&S.classList.remove("hidden"),E("effort"),Q&&Q.classList.remove("opacity-50","pointer-even\
ts-none");else if(b){S&&S.classList.remove("hidden"),E("effort");const Te=get("vision-model-info");if(Te&&
Te.classList.toggle("hidden",h==="deepseek-v4.1-flash"||h==="deepseek-v4-flash-vision-exp"),F&&(F.disabled=
!1),Q&&Q.classList.remove("opacity-50","pointer-events-none"),oe){const qe=get("enable-url-context");
qe&&(qe.checked=!1),oe.classList.add("opacity-50","pointer-events-none")}K&&ce&&(ce.checked=!1,K.classList.
add("opacity-50","pointer-events-none"))}else Ne||(Q&&Q.classList.remove("opacity-50","pointer-event\
s-none"),K&&ce&&(ce.checked=!1,K.classList.add("hidden","opacity-50","pointer-events-none")));if(ke?
X&&X.classList.add("opacity-50","pointer-events-none"):(X&&X.classList.remove("opacity-50","pointer-\
events-none"),(!We||Ze)&&!f.includes("gpt-image")&&(ve.disabled=!1,W.classList.remove("opacity-50"))),
(We&&!Ze||f.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(f)||isGrokVideoModel()||Ne)&&
ve&&W&&(ve.checked=!1,ve.disabled=!0,W.classList.add("opacity-50")),X&&(isLlmModel()?(X.classList.remove(
"hidden"),ge.disabled=!1):(ge.checked=!1,ge.disabled=!0,X.classList.add("hidden"))),ae?(F&&(F.checked=
!0,F.disabled=!0),Q&&Q.classList.add("opacity-50","pointer-events-none"),X&&(ge.checked=!1,ge.disabled=
!0,X.classList.add("opacity-50","pointer-events-none"))):F&&!f.includes("tts")&&!Ne&&!Re&&(F.disabled=
!1),Bt){[F,ce,ge].forEach(qe=>{qe&&(qe.checked=!1,qe.disabled=!0)});const Te=get("enable-url-context");
Te&&(Te.checked=!1,Te.disabled=!0),[Q,oe,K,X].forEach(qe=>{qe&&qe.classList.add("opacity-50","pointe\
r-events-none")})}_&&(Me&&Me.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v",
"glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(h)),[F,ge].forEach(Te=>{Te&&(Te.checked=!1,Te.
disabled=!0)}),[Q,X].forEach(Te=>{Te&&Te.classList.add("opacity-50","pointer-events-none")}));const ft=get(
"mask-btn");ft&&(isGptImageModel()||isIdeogramObjectRemoval()?ft.classList.remove("hidden"):(ft.classList.
add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),
d(),m(),g(),y(),v(),w(),x(),updateBatchUi(f);const jt=get("gemini-free-tier-container");if(jt){const Te=typeof isGeminiApiModel==
"function"&&isGeminiApiModel(f)&&(typeof currentGeminiBackend!="function"||currentGeminiBackend()!==
"vertex_ai");jt.classList.toggle("hidden",!Te)}k(),C(),L(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}r(G,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener(
"change",M),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
M(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const O=get("enable-canvas-mode");O&&(O.checked=canvasModeEnabled,O.addEventListener("change",()=>syncCanvasModeUi(
O.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const Y=get("enable-codin\
g-mode");Y&&(Y.checked=codingModeEnabled,Y.addEventListener("change",()=>syncCodingModeUi(Y.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const P=get(
"prompt-controls-toggle-btn");P&&(P.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>d()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>m()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,h){const b=(f||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(u||"");const _=u||"",S=(b||"").toLowerCase();
if(S==="mathsvg"){const W=getRenderableSvgCode("mathsvg",_);return W&&typeof window.buildMathSvgRenderHtml==
"function"?window.buildMathSvgRenderHtml(W):""}let T="";try{const W=hljs.getLanguage(b)?b:"plaintext";
activeStreamingBubbleId&&_.length>2e4?T=escapeHtml(_):T=hljs.highlight(_,{language:W}).value}catch{T=
escapeHtml(_)}const A=encodeURIComponent(_).replace(/'/g,"%27"),I=detectBlockedScriptsInCode(_),R=hashString(
`${b||"TEXT"}
${_||""}`);let F="";if(canvasModeEnabled){const W=String(canvasPreviewState.selectedKey||"")===R,ge=W?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";F=`<button\
 class="canvas-preview-btn${W?" canvas-active":""}" data-code="${A}" data-code-key="${R}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${ge}" aria-label="${ge}" aria-pressed="${W?"true":"false"}"\
><i class="fas ${W?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,_)){const W=I?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
F=`<button class="html-preview-btn" data-code="${A}" ${I?'data-suspicious="1"':""} title="${W}" aria\
-label="${W}"><i class="fas ${I?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const Q=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,oe=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${R}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,ce=(b||"TEXT")+(I?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),K=getRenderableSvgCode(S,_);return`${K?buildSvgCodeRenderHtml(K,R):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${R}"><div class="code-header"\
><span class="code-lang">${ce}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${oe}${F}${Q}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${T}</code></pre></d\
iv></div>`},link(u,f,h){return`<a href="${u}" title="${f||""}" target="_blank">${h}</a>`},image(u,f,h){
return buildChatImageHtml(u,{alt:h,title:f})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(Si=get("version-update-dismis\
s"))==null||Si.addEventListener("click",()=>{const u=localStorage.getItem("app_version")||"";u&&localStorage.
setItem("version_notified",u),hideModal("version-update-modal")});const H=get("version-update-clear-\
cache");if(H&&(H.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),H.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(H.checked)})),
(Ci=get("version-update-reload"))==null||Ci.addEventListener("click",async()=>{var f;await versionUpdateCachePreferenceSavePromise.
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
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const re=r(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),me=r(u=>{const f=Array.isArray(u)?u:[],h=get("passkey-list"),b=get("passkey-count");if(b&&
(b.innerText=String(f.length)),!!h){if(!f.length){h.innerHTML='<div class="text-[11px] text-gray-500\
">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}h.innerHTML="",f.forEach((_,S)=>{const T=_&&_.id?String(_.id):"",A=document.createElement("di\
v");A.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const I=document.createElement("div");I.className="min-w-0";const R=document.createElement("\
div");R.className="text-xs text-gray-200 truncate",R.innerText=_&&_.name?String(_.name):`Security Ke\
y ${S+1}`;const F=document.createElement("div");F.className="text-[10px] text-gray-500 mt-1",F.innerText=
_&&_.created_at?`\u767B\u9332\u65E5\u6642: ${re(_.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
I.appendChild(R),I.appendChild(F),A.appendChild(I);const Q=document.createElement("button");Q.type="\
button",Q.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",Q.innerText="\u524A\u9664",Q.disabled=!T,T&&(Q.onclick=()=>window.removeWebAuthnCredential(
T)),A.appendChild(Q),h.appendChild(A)})}},"renderPasskeyList"),ie=r(u=>{const f=get("session-list");
if(f){if(!u||!u.length){f.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(h=>{const b=h.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",_=h.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!h.is_current&&!h.is_revoked?`<button \
data-session-id="${escapeHtml(h.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
T=(h.user_agent||"Unknown").slice(0,120),A=h.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${_}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(T)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(re(h.last_seen_at))}\
 / \u4F5C\u6210: ${escapeHtml(re(h.created_at))}</div></div>${S}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(h=>{h.onclick=async()=>{const b=h.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const _=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let S={};try{S=await _.json()}catch{}if(_.ok){if(S.logged_out){
location.href="/login";return}await he()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),he=r(async()=>{const u=get("session-list");u&&(u.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api\
/sessions");let h={};try{h=await f.json()}catch{}if(!f.ok){if(h&&h.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(h.sessions||[]).filter(_=>!_.is_revoked);ie(b)},"loadSessions"),$=r(()=>{const u=get(
"session-refresh-btn");u&&(u.onclick=()=>he());const f=get("session-revoke-others-btn");f&&(f.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await he():showToast("\u64CD\u4F5C\u306B\u5931\
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
(f.checked=!0),M())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadSkills(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const N=get("vision-model-change-btn");N&&(N.onclick=()=>_openVisionModelSelector());const V=get("co\
mpression-format-only");V&&(V.onchange=()=>{const u=V.checked,f=get("compression-max-size"),h=get("c\
ompression-max-dim");f&&(f.disabled=u),h&&(h.disabled=u);const b=get("compression-size-wrap"),_=get(
"compression-dim-wrap");b&&(b.style.opacity=u?"0.4":"1"),_&&(_.style.opacity=u?"0.4":"1")});const j=r(
()=>{const u=get("enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=
!!temporaryChatEnabled,u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(
u.checked)||(setTemporaryChatUiState(f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");j(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const U=get("storage-us\
age-refresh");U&&(U.onclick=()=>loadStorageUsage());let se=null;const B=r(()=>{const u=new Uint8Array(
16);return window.crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),D=r((u={})=>{const f=get("account-transfer-progress"),h=get("account-tran\
sfer-progress-bar"),b=get("account-transfer-progress-percent"),_=get("account-transfer-progress-text"),
S=get("account-transfer-progress-detail"),T=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&f.
classList.remove("hidden"),h&&(h.style.width=`${T}%`),b&&(b.textContent=`${Math.round(T)}%`),_&&(_.textContent=
u.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const I={queued:"\
\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\u30D5\u30A1\
\u30A4\u30EB\u3092\u66F8\u304D\u51FA\u3057\u4E2D",finalizing:"\u6700\u7D42\u51E6\u7406\u4E2D",ready:"\
\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u6E96\u5099\u5B8C\u4E86",downloading:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D",
uploading:"ZIP\u3092\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D",validating:"ZIP\u3092\u691C\u8A3C\u4E2D",
validating_files:"\u30D5\u30A1\u30A4\u30EB\u60C5\u5831\u3092\u691C\u8A3C\u4E2D",reading_files:"\u30D5\u30A1\u30A4\u30EB\u3092\
\u8AAD\u307F\u8FBC\u307F\u4E2D",importing_settings:"\u8A2D\u5B9A\u3092\u53CD\u6620\u4E2D",importing_credentials:"\
\u8A8D\u8A3C\u60C5\u5831\u3092\u53CD\u6620\u4E2D",importing_skills:"Skill\u3092\u8FFD\u52A0\u4E2D",saving_files:"\
\u30D5\u30A1\u30A4\u30EB\u3092\u4FDD\u5B58\u4E2D",importing_chats:"\u30C1\u30E3\u30C3\u30C8\u5C65\u6B74\u3092\u8FFD\u52A0\u4E2D",
importing_feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u8FFD\u52A0\u4E2D",importing_diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF\u3092\u8FFD\u52A0\u4E2D",cancelling:"\u30AD\u30E3\u30F3\u30BB\u30EB\u51E6\u7406\u4E2D",
cancelled:"\u30AD\u30E3\u30F3\u30BB\u30EB\u6E08\u307F",expired:"\u4FDD\u5B58\u671F\u9650\u5207\u308C",
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=I[u.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),pe=r(u=>{Ce&&(Ce.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const h=get("account-transfer-cancel-btn");
h&&(h.disabled=!u)},"setAccountTransferControls"),le=r((u={})=>{const f=get("account-export-ready"),
h=get("account-export-ready-text"),b=get("account-export-expiry"),_=get("account-export-download-btn"),
S=!!(u.available&&u.download_url);if(f&&f.classList.toggle("hidden",!S),!S){_&&_.removeAttribute("hr\
ef");return}const T=Math.max(0,Number(u.size_bytes)||0),A=T>=1024*1024*1024?`${(T/(1024*1024*1024)).
toFixed(2)} GB`:`${(T/(1024*1024)).toFixed(1)} MB`;if(h){const I=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
h.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${I}`}
if(b){const I=u.expires_at?new Date(u.expires_at):null;b.textContent=I&&!Number.isNaN(I.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${I.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
_&&(_.href=u.download_url)},"renderAccountExportAvailability"),Z=r(async u=>{for(;se===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),h=await f.json().catch(()=>({}));if(f.ok&&(h.state!=="pending"&&D(h),["ready","completed","f\
ailed","cancelled","expired"].includes(h.state)))return h}catch{}await new Promise(f=>setTimeout(f,700))}
return null},"pollAccountTransfer"),de=r((u,f,h=!0)=>{f&&(D(f),le(f),h&&f.state==="ready"?showToast(
f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):h&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),xe(u))},"handleFinishedAccountExport"),ee=r(async()=>{try{const u=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(le(f),f.state==="ready"){D(f);return}if(["failed","cancelled","expired"].includes(
f.state)){D(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||se&&se.id===
f.job_id||se)return;const h={id:f.job_id,type:"export",stopped:!1,restored:!0};se=h,pe(!0),D(f);const b=await Z(
h);b&&de(h,b,!0)}catch{}},"refreshLatestAccountExport"),xe=r(u=>{se===u&&(se=null),u.stopped=!0,pe(!1)},
"finishAccountTransfer"),ue=get("account-transfer-cancel-btn");ue&&(ue.onclick=async()=>{const u=se;
if(!(!u||u.stopped)){u.cancelRequested=!0,ue.disabled=!0,D({progress:0,phase:"cancelling",message:"\u30AD\
\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/transfer/${u.
id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}u.controller&&u.controller.abort(),
D({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),u.
type==="export"&&le({available:!1}),xe(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const Ce=get("account-export-btn");Ce&&(Ce.onclick=async()=>{if(se)return;const u={id:B(),
type:"export",stopped:!1};se=u,pe(!0),le({available:!1}),D({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:u.id}),keepalive:!0})),h=await f.json().catch(()=>({}));if(f.status===409&&
h.error==="export_in_progress"&&h.job_id)u.id=h.job_id;else if(!f.ok)throw new Error(h.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
h.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
D({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await Z(u);!u.cancelRequested&&b&&de(u,b,!0)}catch(f){const h=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";D({progress:0,phase:"failed",message:h}),
showToast(h,"error",!0),xe(u)}});const Be=get("account-export-download-btn");Be&&Be.addEventListener(
"click",async u=>{const f=Be.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();try{const h=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await h.json().catch(
()=>({}));h.ok&&b.available&&b.download_url?(Be.href=b.download_url,window.location.assign(b.download_url)):
(le(b),D(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),ee())}catch{window.location.assign(f)}}}),pe(!1),ee();const ze=get("import-files-grid"),
Ke=get("import-files-info"),be=get("import-files-summary"),fe=r(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let _e=null;const ne=r(()=>{
if(!_e)return;const u=_e.files,f=_e.selection;let h=0;u.forEach(S=>{f.has(S.archive_path)&&(h+=Number(
S.size_bytes)||0)});const b=Number(_e.available_bytes)||0,_=h>b;be&&(be.textContent=`\u9078\u629E\u4E2D: ${fe(
h)} / \u5229\u7528\u53EF\u80FD: ${fe(b)}${_?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,be.classList.
toggle("text-red-300",_)),Ke&&(Ke.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),it=r(
()=>{if(!ze||!_e)return;ze.innerHTML="";const u=_e.files;if(!u.length){ze.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
ne();return}u.forEach(f=>{const h=document.createElement("label"),b=_e.selection.has(f.archive_path);
h.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,h.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${fe(f.size_bytes)}</div></div>`;const _=h.
querySelector(".import-file-check");_.addEventListener("change",()=>{_.checked?_e.selection.add(f.archive_path):
_e.selection.delete(f.archive_path),h.classList.toggle("border-blue-500",_.checked),h.classList.toggle(
"border-gray-600",!_.checked),ne()}),ze.appendChild(h)}),ne()},"renderImportFileItems"),Oe=r(u=>new Promise(
f=>{if(_e={files:u.files||[],selection:new Set((u.files||[]).map(h=>h.archive_path)),available_bytes:u.
available_bytes,resolve:f},it(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),gt=r(u=>{if(hideModal("import-files-modal"),_e){const f=_e.resolve;
_e=null,f(u)}},"closeImportFileSelection"),Xe=get("import-files-close");Xe&&(Xe.onclick=()=>gt(null));
const ot=get("import-files-cancel");ot&&(ot.onclick=()=>gt(null));const ct=get("import-files-confirm");
ct&&(ct.onclick=()=>{if(!_e)return;const u=Array.from(_e.selection);gt(u.length?u.join(","):"__none_\
_")});const xt=get("import-files-select-all");xt&&(xt.onclick=()=>{_e&&(_e.files.forEach(u=>_e.selection.
add(u.archive_path)),it())});const on=get("import-files-none");on&&(on.onclick=()=>{_e&&(_e.selection.
clear(),it())});const Ln={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
Yt=r(u=>{if(u===!0)return"ON";if(u===!1)return"OFF";if(u==null||u==="")return"\u672A\u8A2D\u5B9A";const f=String(
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let St=null;const Ct=r(
u=>{if(St){const f=St;St=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),Qt=r(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}St=f;const b=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],_=get("settings-confirmation-list");_&&(b.length?
_.innerHTML=b.map(T=>{const A=Ln[T.field]||T.field,I=Yt(T.current),R=Yt(T.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
I)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(R)}</d\
iv>
                            </div>`}).join(""):_.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Tt=get("settings-confir\
mation-modal");Tt&&Tt.addEventListener("click",u=>{u.target===Tt&&Ct(!1)});const kt=get("settings-co\
nfirmation-close");kt&&(kt.onclick=()=>Ct(!1));const At=get("settings-confirmation-cancel");At&&(At.
onclick=()=>Ct(!1));const rn=get("settings-confirmation-confirm");rn&&(rn.onclick=()=>Ct(!0));const Ft=get(
"account-import-btn"),Lt=get("account-import-inplace"),ln=get("account-import-inplace-warning");if(Lt&&
ln){const u=r(()=>ln.classList.toggle("hidden",!Lt.checked),"syncInplaceWarn");Lt.addEventListener("\
change",u),u()}Ft&&(Ft.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,h=get("account-import-categories"),b=h?Array.from(h.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(ce=>ce.value):[],_=get("account-import-inplace"),S=!!(_&&_.checked),T=get("account-impo\
rt-settings-bypass"),A=!!(T&&T.checked);let I=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const R=h?Array.from(h.querySelectorAll('input[type="checkbox"]:checked')).map(ce=>(ce.
closest("label")&&ce.closest("label").textContent||ce.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${R.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const F={id:B(),type:"import",stopped:!1,controller:new AbortController};
se=F,pe(!0),D({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const Q=get("account-import-result");let oe=Promise.resolve(null);try{const K=Math.max(1,Math.ceil(f.
size/10485760)),ve=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:F.controller.
signal})),W=await ve.json().catch(()=>({}));if(!ve.ok)throw new Error(W.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
F.uploadId=W.upload_id;const ge=W.chunk_size||10485760;let X=0,we=0;const Le=r(async()=>{for(;;){const Ee=we++;
if(Ee>=K)return;const Me=f.slice(Ee*ge,Math.min(f.size,(Ee+1)*ge)),We=new FormData;We.append("chunk",
Me,f.name),We.append("index",String(Ee));const st=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:We,signal:F.controller.signal})),
ft=await st.json().catch(()=>({}));if(!st.ok)throw new Error(ft.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
X++,D({progress:Math.min(35,Math.round(X/K*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${X}\
/${K}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ae=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ae=!0);try{await Promise.
all([Le(),Le(),Le()]);const Ee=await apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:F.controller.signal})),Me=await Ee.json().
catch(()=>({}));if(!Ee.ok)throw new Error(Me.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
D({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ae&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ke="",Ne=!1,Re=0;const mt=r(
async()=>{let Ee=!1;const Me=r(()=>{Ee||(Ee=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const We=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),st=await We.
json().catch(()=>null);if(!We.ok||!st){Me();return}cacheUserSettings(st);const ft=get("settings-moda\
l");if(ft&&ft.classList.contains("modal-open"))try{ut(st)}catch{}st.theme_color&&applyThemeColor(st.
theme_color,!0),Object.prototype.hasOwnProperty.call(st,"minimal_prompt_mode")&&st.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(st,"compact_prompt_mode")&&setCompactPromptMode(
!!st.compact_prompt_mode)}catch{}Me()},"refreshSettingsFormAfterImport"),Ze=r(Ee=>{const Me=Ee&&Ee.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";Q&&(Q.textContent=`\u5B8C\u4E86: ${Me}`,
Q.classList.remove("hidden","text-red-300"),Q.classList.add("text-emerald-300")),D({progress:100,phase:"\
completed",message:Me}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&mt()},"finishImportSucce\
ss"),vt=r(async()=>{try{const Me=await(await apiFetch(`/api/account/transfer/${F.id}`,manualSpinnerRequestOptions(
{cache:"no-store"}))).json().catch(()=>null);return Me&&Me.state?Me:null}catch{return null}},"fetchI\
mportStatus"),Bt=r(async()=>{const Ee=await vt();if(!Ee)return{status:"unknown"};if(Ee.state==="comp\
leted")return Ze(Ee),{status:"done"};if(["failed","cancelled","expired"].includes(Ee.state))throw new Error(
Ee.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Ee.state===
"needs_selection"&&Array.isArray(Ee.files)){const Me=await Oe({files:Ee.files,available_bytes:Ee.available_bytes});
return Me===null?(D({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ke=Me,{status:"reselect"})}if(Ee.state===
"needs_settings_confirmation"&&Array.isArray(Ee.settings_changes))return await Qt({settings_changes:Ee.
settings_changes})?(I=!0,{status:"reselect"}):(D({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Ee.state==="running"){const Me=await Promise.race([oe.catch(()=>null),new Promise(We=>setTimeout(
()=>We(null),6e4))]);if(Me&&Me.state==="completed")return Ze(Me),{status:"done"};throw Me&&["failed",
"cancelled","expired"].includes(Me.state)?new Error(Me.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ne;){F.stopped=!0,await oe.catch(()=>null),
F.stopped=!1,oe=Z(F);let Ee;try{Ee=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:F.uploadId,
categories:b.join(","),job_id:F.id,selected_files:ke,restore_inplace:S,confirm_settings:I||A}),signal:F.
controller.signal}))}catch(De){if(F.cancelRequested||De&&De.name==="AbortError")throw De;const Ve=await Bt();
if(Ve.status==="done"){Ne=!0;break}if(Ve.status==="cancelled")return;if(Ve.status==="reselect")continue;
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Me=null;try{Me=await Ee.json()}catch{Me=null}if(Me===null){const De=await Bt();if(De.status==="d\
one"){Ne=!0;break}if(De.status==="cancelled")return;if(De.status==="reselect")continue;if(Ee.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Ee.ok&&Me.error==="storage_limit_files"&&Me.files){const De=await Oe(Me);if(De===null){D({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ke=De;continue}if(Me&&Me.status==="settings_confirmation"&&
Array.isArray(Me.settings_changes)){if(!await Qt(Me)){D({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}I=!0;continue}if(!Ee.ok)throw new Error(Me.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const We=Me.imported||{},st=[`\u8A2D\u5B9A ${We.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${We.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${We.chats||0}\u4EF6`,`Skill ${We.skills||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${We.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${We.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${We.
diagnostics||0}\u4EF6`].join(" / "),ft=Me.duplicates||{},jt={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"\
Skill",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Te=[];for(const De of Object.keys(jt)){const Ve=Number(ft[De])||0;Ve>
0&&Te.push(`${jt[De]} ${Ve}\u4EF6`)}const qe=Te.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Te.
join("\u3001")}\uFF09`:"";Q&&(Q.textContent=`\u5B8C\u4E86: ${st}${qe}`,Q.classList.remove("hidden","\
text-red-300"),Q.classList.add("text-emerald-300")),D({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&mt(),Ne=!0}}catch(ce){if(F.
uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),F.cancelRequested||ce&&ce.name==="AbortError")return;const K=ce&&
ce.message?ce.message:"",ve=K==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
K||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";D({progress:0,phase:"f\
ailed",message:ve}),Q&&(Q.textContent=ve,Q.classList.remove("hidden","text-emerald-300"),Q.classList.
add("text-red-300")),showToast(ve,"error",!0)}finally{F.stopped=!0,await oe.catch(()=>null),xe(F)}});
const Et=get("account-dedupe-btn"),$t=get("account-dedupe-result"),Dt=r((u,f=!1)=>{$t&&($t.textContent=
u,$t.classList.remove("hidden"),$t.classList.toggle("text-red-300",!!f),$t.classList.toggle("text-em\
erald-300",!f))},"showDedupeResult");Et&&(Et.onclick=async()=>{const u=r(async()=>{const f=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),h=await f.json().catch(()=>null);if(!f.ok||!h)throw new Error(h&&h.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!h.has_duplicates){Dt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],_={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",
feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};
for(const F of["chats","skills","files","feedback","diagnostics"]){const Q=Number(h.duplicates&&h.duplicates[F])||
0;Q>0&&b.push(`${_[F]} ${Q}\u4EF6`)}const S=Number(h.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${h.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${h.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const T=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await T.json().catch(()=>null);if(!T.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const I=[];for(const F of["chats","skills","files","feedback","diagnostics"]){const Q=Number(A.removed&&
A.removed[F])||0;Q>0&&I.push(`${_[F]} ${Q}\u4EF6`)}const R=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";Dt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${I.join("\u3001")||"0\u4EF6"}${R}`),loadThreads(),loadSkills(),loadStorageUsage()},"r\
un");if(!Et.disabled){Et.disabled=!0,Dt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await u()}catch(f){Dt(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{Et.disabled=!1}}});const cn=get("site-cache-usage-refresh");cn&&(cn.onclick=()=>loadSiteCacheUsage());
const Zt=get("clear-site-cache-btn");Zt&&(Zt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Zt)});
const J=get("enc-scan-result"),Se=r(async(u=null)=>{J&&(J.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const h=await apiFetch(
f,{cache:"no-store"}),b=await h.json();if(!h.ok){J&&(J.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const _=b.total||0,S=b.encrypted||0,T=b.unencrypted||0;let A=`Total: ${_} / Encrypted: ${S} /\
 Plain: ${T}`;if(b.samples&&b.samples.length){const I=b.samples.slice(0,8).map(R=>{const F=R.timestamp?
new Date(R.timestamp).toLocaleString():"";return`#${R.id} (${R.role||""}) ${F}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${I}</div>`}J&&(J.innerHTML=A)}catch{J&&(J.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),je=get("enc-scan-all");je&&(je.onclick=()=>Se(
null));const Ge=get("enc-scan-thread");Ge&&(Ge.onclick=()=>currentThreadId?Se(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ye=get("admin-enc-\
list");let ht=null,bt=!1;const tt=r(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThread\
EncryptedFromMessages"),_t=r(()=>{ht=tt(allMessages)},"refreshCurrentThreadEncStateFromMessages"),nt=r(
async(u,f,{confirmPrompt:h=!0,reloadCurrent:b=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const _=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(h&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;bt=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
T=await S.json().catch(()=>({}));return S.ok?(showToast(`${_}\u3057\u307E\u3057\u305F\uFF08${T.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),ht=!!f,b&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ye&&await yt(),!0):
(showToast(T.error||`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{bt=!1}},"setAdminThreadEncry\
ption"),dt=r(u=>{if(!Ye)return;const f=u.threads||[];if(!f.length){Ye.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ye.innerHTML=f.map(h=>{const b=h.encrypted_count>0?"enc":"plain",_=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",T=h.
updated_at?new Date(h.updated_at).toLocaleString():"",A=escapeHtml(String(h.thread_id)),I=currentThreadId&&
String(currentThreadId)===String(h.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(h.title||
"")}">${escapeHtml(h.title||"(\u7121\u984C)")}${I?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${T} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${h.
message_count} / \u6697\u53F7\u5316: ${h.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${_}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),yt=r(async()=>{if(Ye){Ye.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){Ye.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(dt(f),currentThreadId&&Array.isArray(f.threads)){const h=f.threads.find(b=>String(
b.thread_id)===String(currentThreadId));h&&(ht=!!h.encrypted)}}catch{Ye.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>yt()),window.__loadAdminEncThreads=
yt,window.__refreshAdminThreadEncState=_t,window.__setAdminThreadEncryption=nt;const dn=get("encrypt\
ion-status-admin-toggle");dn&&dn.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ye&&(Ye.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const A=f.getAttribute("data-id");if(!A)return;typeof et==
"function"?et():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const h=u.target.closest(".admin-enc-toggle");if(!h||bt)return;const b=h.getAttribute("data-i\
d"),_=h.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;h.disabled=!0;
const T=h.textContent;h.textContent="\u51E6\u7406\u4E2D...";try{await nt(b,_,{confirmPrompt:!1,reloadCurrent:!0})}finally{
h.disabled=!1,h.textContent=T,await yt()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const un=r(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(h=>{const b=h.status||"new",_=h.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=h.created_at?new Date(h.created_at).toLocaleString():"",T=h.replied_at?new Date(h.replied_at).toLocaleString():
"",A=h.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${h.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(h.username||
"")}${_}</div>
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
A)}</textarea>
                                ${A?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
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
                    `}).join("")}},"renderBanAppeals"),pn=r(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const h=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!h.
ok)return;const _=(await h.json()).unread_count||0;f.textContent=String(_),u&&_>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${_}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Ht=r(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const h=await f.json();un(h.items||[]),
await pn(!1)}catch{}}},"loadBanAppeals"),Fe=r(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await Ht()}catch{}},"markBanAppealsRead"),Ue=r(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await Ht()}catch{}},"updateBanAppealStatus"),qt=r(()=>{
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
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),It=r(()=>{const u=get("set-st\
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
                `,f.appendChild(h);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const _=get(
"set-llm-transcribe-prompt");_&&(_.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),Gt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
lls"}];window.buildAutoSystemPromptRows=(u,f=!1)=>{const h=f?"w-full h-14 bg-gray-950 border border-\
gray-700 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 roun\
ded p-2 text-xs text-gray-200";return Gt.map(b=>{const _=b.barLocked===!0,S=b.barLabel||"MCP",T=_?`<\
div class="text-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\
\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>`:
"",A=_?`<input type="checkbox" id="${u}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${u}-auto-sys-${b.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${b.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${_?`ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"`:
""}>
                                ${A}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${u}-auto-sys-${b.key}-text" class="${h}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${T}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{Gt.forEach(h=>{const b=f&&
typeof f=="object"?f[h.key]||{}:{},_=get(`${u}-auto-sys-${h.key}-enabled`),S=get(`${u}-auto-sys-${h.
key}-text`);_&&(h.barLocked===!0?_.disabled=!0:_.checked=b.enabled!==!1),S&&(S.value=b.text||"",S.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Pt=r((u,f=null)=>{if(f){const h=get(f);h&&(h.checked=!0)}Gt.forEach(h=>{const b=get(`${u}-auto\
-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);if(b&&(h.barLocked!==!0?b.checked=!0:b.
disabled=!0),_){const S=_.placeholder||"";_.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Ae=r(u=>{const f={};return Gt.forEach(h=>{const b=get(`${u}\
-auto-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);f[h.key]={enabled:h.barLocked===!0?
!0:b?b.checked:!0,text:_?_.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
Ae,window.ensureAutoSystemPromptSettingsCard=()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.
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
                `,f.appendChild(h)},qt(),It(),j();const Pe=r(()=>{const u=get("set-default-model");if(!u)
return;const f=u.value;u.innerHTML="",MODELS.forEach(b=>{const _=document.createElement("optgroup");
_.label=b.category,(b.items||[]).forEach(S=>{const T=document.createElement("option");T.value=S.id,T.
textContent=S.name,_.appendChild(T)}),u.appendChild(_)});const h=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";h&&Array.from(u.options).some(b=>b.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),rt=r(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(b=>{const _=(b.items||[]).filter(T=>{const A=(T.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(_.length===0)return;const S=document.createElement("optgroup");S.label=b.category,_.forEach(
T=>{const A=document.createElement("option");A.value=T.id,A.textContent=T.name+" \u2605",S.appendChild(
A)}),u.appendChild(S)});const h=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";h&&Array.from(u.options).some(b=>b.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),ut=r(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-glo\
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
!1);const b=get("bot-status");b&&(u.is_bot_banned?(b.textContent=`BAN\u4E2D: ${u.bot_ban_reason||"Bo\
t detection"}`,b.classList.remove("hidden"),b.classList.add("text-red-400")):b.classList.add("hidden")),
u&&u.theme_color?(applyThemeColor(u.theme_color,!0),syncThemeInputs(u.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=u.username);const _=get("2fa-badge"),S=get("disable-2fa-btn");u.is_2fa_enabled?
(_.innerText="ENABLED",_.classList.replace("bg-gray-700","bg-green-600"),_.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(_.innerText="DISABLED",_.classList.replace("bg\
-green-600","bg-gray-700"),_.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!u.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=u.default_2fa_method||"totp");const T=get(
"set-passkey-only-login"),A=get("passkey-only-note"),I=Array.isArray(u.passkey_credentials)?u.passkey_credentials:
[];if(me(I),T){T.checked=!!u.passkey_only_login;const ce=I.length>0||!!u.has_webauthn;T.disabled=!ce,
ce||(T.checked=!1),A&&(ce?A.classList.add("hidden"):A.classList.remove("hidden"))}const R=get("mig-s\
tatus-box"),F=get("mig-progress-text"),Q=get("mig-progress-bar");if((u.migration_status||"idle")==="\
processing"){R.classList.remove("hidden");const ce=(u.migration_progress||"").split("/");if(ce.length===
2){const K=parseInt(ce[0]||"0",10),ve=parseInt(ce[1]||"0",10);F&&(F.innerText=`${K} / ${ve}`),Q&&ve>
0&&(Q.style.width=`${Math.min(100,Math.floor(K/ve*100))}%`)}}else R.classList.add("hidden"),Q&&(Q.style.
width="0%"),F&&(F.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const u=await ensureUserSettingsSnapshot();u&&ut(u);const f=get(
"search-box"),h=f?f.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),Pe(),rt(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(h,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(_=>{setTimeout(()=>{restoreThreadSearchValue(
h,"restored-search-box-"+_+"ms"),snapshotSidebarHistory("settings-open-later-"+_+"ms")},_)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),ee(),It(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),pn(!0),Ht(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),hn(),$(),he();try{loadMcpServers()}catch{}};const et=r((u=!1)=>{settingsModelPickerTarget&&
window.closeModelModal(!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),
revealPersistentSidebarLists(),snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory(
"settings-close-later-300ms"),300),!u&&location.pathname==="/settings"&&history.back()},"closeSettin\
gsModal"),Ot=r(()=>{const u=get("set-theme-color"),f=get("set-theme-color-text"),h=get("theme-reset-\
btn"),b=document.querySelectorAll("#theme-presets .theme-swatch"),_=r((S,T=!0)=>{const A=normalizeHex(
S);A&&(applyThemeColor(A,T),syncThemeInputs(A))},"applyFromValue");u&&u.addEventListener("input",()=>_(
u.value,!0)),f&&(f.addEventListener("change",()=>{const S=normalizeHex(f.value);if(!S){syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||THEME_DEFAULT);return}_(S,!0)}),f.addEventListener("keydown",
S=>{S.key==="Enter"&&(S.preventDefault(),f.blur())})),h&&(h.onclick=()=>_(THEME_DEFAULT,!0)),b.forEach(
S=>{S.addEventListener("click",()=>_(S.getAttribute("data-color"),!0))})},"bindThemeControls"),Ut=r(
()=>{const u=get("reset-global-sys-prompt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-pro\
mpt-text").value=""),get("set-global-sys-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=
!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const f=get("reset-thread-sys-prompt");f&&(f.onclick=()=>{get("thread-global-sys-prompt")&&
(get("thread-global-sys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-glo\
bal-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-set-auto-sys-prompt-defaults");h&&(h.onclick=()=>{Pt("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const b=get("reset-thread-auto-sys-prompt-defaults");b&&(b.onclick=()=>{Pt("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>et();const Qn=get("settings-header-close");Qn&&(Qn.onclick=()=>et());
const en=get("settings-search");en&&(en.addEventListener("input",filterSettings),en.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const h=f.querySelector(
":scope > .settings-match");h&&h.scrollIntoView({behavior:"smooth",block:"start"})}}));const Zn=get(
"settings-search-clear");Zn&&Zn.addEventListener("click",()=>{en&&(en.value="",filterSettings(),en.focus())}),
Ot(),Ut(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Bn=get("set-gemini-local-python-dialog");Bn&&(Bn.onchange=()=>setGeminiLocalPyDialogEnabled(Bn.
checked));const ei=get("set-gemini-backend");ei&&(ei.onchange=()=>{syncGeminiBackendUi(),typeof window.
toggleOptions=="function"&&window.toggleOptions()});const ti=get("set-admin-api-key-mode");ti&&(ti.onchange=
()=>syncAdminApiKeyModeUi());const jn=get("set-temp-chat-timeout-seconds");jn&&(jn.onchange=()=>{applyTemporaryChatTimeoutSeconds(
jn.value)});const ni=get("slash-command-cancel-btn");ni&&(ni.onclick=()=>{hidePendingSlashCommandIndicator();
const u=get("prompt-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-s\
ettings-btn").onclick=async()=>{var A;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),h=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:Ae(
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
((A=get("set-server-diagnostic-log-retention-hours"))==null?void 0:A.value)||1),enable_server_diagnostic_log:get(
"set-server-diagnostic-log")?get("set-server-diagnostic-log").checked:!1,enable_client_debug_log:get(
"set-client-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-o\
nly-login")?get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-goog\
le")?get("set-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set\
-default-2fa-method").value:"totp",new_username:u?u.value:null,new_password:f?f.value:null},_=get("s\
et-e2ee")?get("set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
_!==S&&(b.enable_e2ee=_),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
(b.gemini_key=get("set-gemini").value),get("set-gemini-free")&&(b.gemini_free_key=get("set-gemini-fr\
ee").value),get("set-deepseek")&&(b.deepseek_key=get("set-deepseek").value),get("set-zai")&&(b.zai_key=
get("set-zai").value),get("set-kimi")&&(b.kimi_key=get("set-kimi").value),get("set-mistral")&&(b.mistral_key=
get("set-mistral").value),get("set-ideogram")&&(b.ideogram_key=get("set-ideogram").value),get("set-a\
nthropic")&&(b.anthropic_key=get("set-anthropic").value),b.model_api_keys=normalizeModelApiKeyMap(modelApiKeyMap),
get("set-gemini-backend")&&(b.gemini_backend=normalizeGeminiBackend(get("set-gemini-backend").value)),
get("set-gemini-vertex-project")&&(b.gemini_vertex_project=get("set-gemini-vertex-project").value),get(
"set-gemini-vertex-location")&&(b.gemini_vertex_location=get("set-gemini-vertex-location").value),get(
"set-gemini-vertex-credentials-json")&&(b.gemini_vertex_credentials_json=get("set-gemini-vertex-cred\
entials-json").value),get("set-xai")&&(b.xai_key=get("set-xai").value),get("set-google-key")&&(b.google_key=
get("set-google-key").value),get("set-google-project")&&(b.google_project=get("set-google-project").
value),get("set-admin-api-key-mode")&&(b.admin_api_key_mode=normalizeAdminApiKeyMode(get("set-admin-\
api-key-mode").value)),get("set-bot-detect")&&(b.bot_detection_enabled=get("set-bot-detect").checked),
get("set-bot-detect-global")&&(b.bot_detection_global_enabled=get("set-bot-detect-global").checked);
const T=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"appli\
cation/json"},body:JSON.stringify(b)});if(T.ok){let I="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const oe=await T.json();oe&&oe.message&&(I=oe.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled,
math_steps_enabled:!!b.math_steps_enabled,default_math_steps:!!b.default_math_steps})),typeof window.
applyMathStepsSettings=="function"&&(window.applyMathStepsSettings({math_steps_enabled:!!b.math_steps_enabled}),
b.math_steps_enabled||setMathStepsEnabled(!1,{persist:!1})),window.applySavedUserSystemPromptSettings(
{system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,apply_global_system_prompt:b.
apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,auto_system_prompt_notices_config:b.
auto_system_prompt_notices_config}),et();const R=currentUsername,F=CHAT_CONFIG.enableE2EE;enterToSend=
b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const Q=useSwCache;useSwCache=b.use_sw_cache,
window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),Q!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(I,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==R?setTimeout(()=>location.
reload(),1e3):b.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let I={};try{I=await T.json()}catch{}showToast(I.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
son"},body:JSON.stringify({username:f,mode:"single"})}),b=await h.json(),_=get("bot-unban-result");if(h.
ok&&b&&b.status==="ok")_&&(_.textContent=`${f} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const u=get("bot-unban-username"),f=u?u.value.trim():"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"linked"})}),b=await h.json(),_=get("bot-unban-result");if(h.
ok&&b&&b.status==="ok")_&&(_.textContent=`${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const u=get(
"bot-speed-test-btn"),f=get("bot-speed-test-result");u&&(u.disabled=!0),u&&u.classList.add("opacity-\
60","cursor-not-allowed"),f&&(f.classList.remove("hidden"),f.textContent="\u5B9F\u884C\u4E2D...");try{
const h=r(X=>{f&&(f.textContent=X)},"setBox"),b=r(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),_=r((X,we)=>!X||!we||we<=0?0:X*8/(we/1e3)/1e3/1e3,"toMbps"),S=r(X=>Number.isFinite(
X)?`${X.toFixed(0)} ms`:"-","fmtMs"),T=r(X=>Number.isFinite(X)?`${X.toFixed(X>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=r(async(X,we)=>{const Le=await X.json().catch(()=>({}));return Le&&Le.error?Le.error:we},
"parseErr"),I=[];h("\u6E2C\u5B9A\u4E2D... ping");for(let X=0;X<4;X++){const we=performance.now(),Le=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),ae=performance.now();if(!Le.ok)throw new Error(await A(
Le,"ping_failed"));await Le.json().catch(()=>({})),I.push(ae-we)}const R=I.reduce((X,we)=>X+we,0)/Math.
max(1,I.length),F=Math.min(...I),Q=r(async X=>{const we=performance.now(),Le=await apiFetch(`/api/sp\
eedtest/download?bytes=${X}&_=${b()}`,{cache:"no-store"});if(!Le.ok)throw new Error(await A(Le,"down\
load_failed"));const ae=await Le.arrayBuffer(),ke=performance.now();return{bytes:ae.byteLength||X,ms:ke-
we,mbps:_(ae.byteLength||X,ke-we)}},"runDownload");h(`\u6E2C\u5B9A\u4E2D... ping ${S(R)}
\u6E2C\u5B9A\u4E2D... download`);const oe=[];for(const X of[2*1024*1024,8*1024*1024])oe.push(await Q(
X)),h(`\u6E2C\u5B9A\u4E2D... ping ${S(R)}
download ${T(Math.max(...oe.map(we=>we.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const ce=Math.max(...oe.map(X=>X.mbps)),K=r(async X=>{const we=new Uint8Array(
X),Le=performance.now(),ae=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:we,cache:"no-store"}),ke=performance.now();if(!ae.ok)
throw new Error(await A(ae,"upload_failed"));const Ne=await ae.json().catch(()=>({})),Re=Number(Ne.bytes_received||
X)||X;return{bytes:Re,ms:ke-Le,mbps:_(Re,ke-Le),serverMs:Number(Ne.server_elapsed_ms||0)||0}},"runUp\
load"),ve=[];for(const X of[1*1024*1024,4*1024*1024])ve.push(await K(X));const W=Math.max(...ve.map(
X=>X.mbps)),ge=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(R)} / ${S(F)}`,`Download (best): ${T(ce)}`,`Upload (best): ${T(W)}`,`Download r\
uns: ${oe.map(X=>`${Math.round(X.bytes/1024/1024)}MB=${T(X.mbps)}`).join(", ")}`,`Upload runs: ${ve.
map(X=>`${Math.round(X.bytes/1024/1024)}MB=${T(X.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
h(ge.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(h){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${h&&h.message?h.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Ht()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Fe()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){h&&await Fe([Number(h)]);return}if(f.classList.contains(
"ban-appeal-status")){const b=f.getAttribute("data-status");h&&b&&await Ue({id:Number(h),status:b});
return}if(f.classList.contains("ban-appeal-reply-send")){const b=f.closest("[data-appeal-id]"),_=b?b.
querySelector(".ban-appeal-reply"):null,S=_?_.value:"";h&&await Ue({id:Number(h),admin_reply:S});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";h&&await Ue({id:Number(
h),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
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
u.target.value))});const Fn=get("marker-color-picker");Fn&&(Fn.oninput=u=>setMarkerColor(u.target.value),
Fn.onchange=u=>setMarkerColor(u.target.value));const Dn=get("marker-opacity");Dn&&(Dn.oninput=u=>setMarkerOpacity(
u.target.value),Dn.onchange=u=>setMarkerOpacity(u.target.value));const Mn=get("marker-opacity-number");
Mn&&(Mn.onchange=u=>setMarkerOpacity(u.target.value),Mn.onblur=u=>setMarkerOpacity(u.target.value),Mn.
onkeydown=u=>{u.key==="Enter"&&(setMarkerOpacity(u.target.value),u.target.blur())}),document.querySelectorAll(
"#marker-toolbar .marker-color-chip[data-marker-color]").forEach(u=>{u.onclick=()=>setMarkerColor(u.
getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),
get("marker-crop-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&
(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=
()=>clearMarkerCanvas()),get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),
initMarkerCanvas(),initCropCanvas(),window.addEventListener("resize",()=>{const u=get("marker-modal");
!u||u.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const Mi=r(()=>{const u=get(
"upload-modal");return!!(u&&!u.classList.contains("hidden"))},"isUploadModalOpen"),tn=get("drop-over\
lay");let mn=0;const Ti=r(()=>{Mi()||tn&&(tn.classList.remove("hidden"),tn.classList.add("flex"))},"\
showDropOverlay"),fn=r(()=>{mn=0,tn&&(tn.classList.add("hidden"),tn.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=fn;const Mt=get("upload-dropzone");Mt&&(Mt.addEventListener("dr\
agover",u=>{u.preventDefault(),Mt.classList.add("dragover")}),Mt.addEventListener("dragleave",()=>{Mt.
classList.remove("dragover")}),Mt.addEventListener("drop",u=>{u.preventDefault(),u.stopPropagation(),
Mt.classList.remove("dragover"),fn();const f=u.dataTransfer?u.dataTransfer.files:null;f&&f.length&&handleFiles(
f)})),window.addEventListener("dragenter",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.
types.includes("Files")||(mn+=1,Ti())}),window.addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||u.preventDefault()}),window.addEventListener("dragle\
ave",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(mn=Math.max(
0,mn-1),(mn===0||!u.relatedTarget||u.clientY<=0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=
window.innerHeight)&&fn())}),window.addEventListener("dragend",()=>{fn()}),window.addEventListener("\
drop",u=>{fn(),!(!u.dataTransfer||!u.dataTransfer.files||u.dataTransfer.files.length===0)&&(u.preventDefault(),
!(Mt&&Mt.contains(u.target))&&handleFiles(u.dataTransfer.files))});const ii=get("bot-admin-modal"),Ai=r(
u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML="",!u||!u.length){f.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((h,b)=>{const _=!!h.is_bot_banned,S=h.bot_detection_enabled!==!1,T=Number(h.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(b,12)*.02}\
s`,A.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(h.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${_?"BAN\u4E2D":"\u6B63\u5E38"}${T?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(T/60)}\u5206\uFF09`:""} ${h.bot_ban_reason?
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
                    `,f.appendChild(A)})}},"renderBotUsers"),gn=r(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const h=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),b=await h.json();h.ok&&b&&b.users?Ai(b.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{f&&(f.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Hn=r(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||ii))return;const f=get(
"settings-modal");f&&(f.classList.contains("modal-open")||f.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await gn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Hn;const qn=r(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const u=get("settings-modal");u&&(u.classList.contains(
"modal-open")||u.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=qn,window.closeBotLogModal=(u=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!u&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{qn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(u=!1)=>{(get("bot-admin-modal")||ii)&&hideModal("bot-admin-modal"),!u&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Hn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await gn(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await gn("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async u=>{u.key==="Enter"&&await gn(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute(
"data-user");if(!h)return;let b;if(f.classList.contains("bot-toggle-detect")){const _=f.getAttribute(
"data-enabled")!=="1";b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:h,action:"toggle_detection",enabled:_})})}else if(f.classList.
contains("bot-toggle-ban"))if(f.getAttribute("data-banned")==="1")b=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"ban",reason:"Admin ban"})})}else if(f.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"unban_linked"})})}else if(f.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"delete_account"})})}if(b){if(b.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(b.ok){if(f.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),h===currentUsername)){location.href="/";return}}else{
let _={};try{_=await b.json()}catch{}showToast(_.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
_.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await gn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Nt={"/settings":{id:"settings-mod\
al",open:r(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:r(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:r(()=>{ki(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
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
reate New Skill",showModal("skill-modal")},"open")},"/compression":{id:"compression-modal",open:r(()=>window.
openCompressionModal(),"open")},"/math-steps":{id:"math-steps-config-modal",open:r(()=>window.openMathStepsModal(),
"open")},"/admin-bots":{id:"bot-admin-modal",open:r(()=>Hn(),"open")},"/admin-bot-logs":{id:"bot-log\
-modal",open:r(()=>qn(),"open")}};Nt["/gem"]=Nt["/skill"];const si=r((u,f=!1)=>{switch(u){case"setti\
ngs-modal":et(f);break;case"upload-modal":closeUploadModal(f);break;case"image-split-modal":closeImageSplit();
break;case"image-split-compression-modal":hideModal(u);break;case"camera-capture-modal":closeCameraCaptureModal(
f?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&window.closeHistoryModal(
f);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(f);break;case"branch-modal":window.
closeBranchModal&&window.closeBranchModal(f);break;case"batch-modal":window.closeBatchModal&&window.
closeBatchModal(f);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
f);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(f);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(f);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(f);break;case"token-detail-modal":closeTokenDetail(f);break;case"encryption-s\
tatus-modal":closeEncryptionModal(f);break;case"python-exec-modal":closePythonExecDetail(f);break;case"\
skill-modal":window.closeSkillModal&&window.closeSkillModal(f);break;case"compression-modal":window.
closeCompressionModal&&window.closeCompressionModal(f);break;case"math-steps-config-modal":window.closeMathStepsModal?
window.closeMathStepsModal(f):hideModal(u);break;case"bot-admin-modal":window.closeBotAdminModal&&window.
closeBotAdminModal(f);break;case"bot-log-modal":window.closeBotLogModal&&window.closeBotLogModal(f);
break;case"mcp-decision-modal":typeof submitMcpDecision=="function"?submitMcpDecision("deny"):hideModal(
u);break;case"api-key-required-modal":{const b=get("api-key-modal-cancel-btn");b&&typeof b.onclick==
"function"?b.click():hideModal(u);break}case"lyria-studio-modal":window.closeLyriaStudio?window.closeLyriaStudio():
hideModal(u);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(
u);break;case"version-update-modal":const h=localStorage.getItem("app_version")||"";h&&localStorage.
setItem("version_notified",h),hideModal(u);break;default:hideModal(u);break}},"closeModalById");window.
addEventListener("popstate",u=>{var S,T;const f=(S=u.state)==null?void 0:S.settingsModelPicker;if(location.
pathname==="/settings"&&settingsModelPickerTargets.includes(f)){const A=r(()=>openSettingsModelPicker(
f,!0),"restorePicker");(T=get("settings-modal"))!=null&&T.classList.contains("modal-open")?A():window.
openSettingsModal().then(()=>{var I;location.pathname==="/settings"&&((I=history.state)==null?void 0:
I.settingsModelPicker)===f&&A()});return}settingsModelPickerTarget&&window.closeModelModal(!0);let h=!1;
Object.values(Nt).forEach(A=>{const I=get(A.id);I&&I.classList.contains("modal-open")&&location.pathname!==
Object.keys(Nt).find(R=>Nt[R].id===A.id)&&(si(A.id,!0),h=!0)});const b=location.pathname.match(/^\/c\/(.+)$/);
if(b){const A=decodeURIComponent(b[1]);String(currentThreadId)!==String(A)&&loadMessages(A,{skipHistory:!0})}else
location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});const _=Nt[location.pathname];
if(_){const A=get(_.id);A&&!A.classList.contains("modal-open")&&_.open()}}),document.addEventListener(
"keydown",u=>{u.key==="Escape"&&settingsModelPickerTarget&&(u.preventDefault(),window.closeModelModal())});
const ai=location.pathname;Nt[ai]&&(history.replaceState({},"","/"),setTimeout(()=>Nt[ai].open(),500)),
get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const u=get("easy-login-mi\
ns"),f=u?parseInt(u.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${f}\u5206\u9593\u6709\u52B9\
\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const b=await(await apiFetch("/api/easy_login",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:f})})).json();b&&b.temp_password?
(get("easy-login-code").textContent=b.temp_password,get("easy-login-exp").textContent=b.expires_at||
"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(f&&f.cancelled){const h=get("easy-login-r\
esult");h&&h.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const Ei=4,$i=5*1024*1024;function oi(u,f,h){let b=[];const _=r(()=>{h&&(h.innerHTML="",
b.forEach((S,T)=>{const A=document.createElement("div");A.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const I=document.createElement("img");I.src=S.url,I.alt=S.file.name,
I.className="w-full h-full object-cover";const R=document.createElement("button");R.type="button",R.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",R.textContent=
"\xD7",R.onclick=()=>{URL.revokeObjectURL(S.url),b.splice(T,1),_()},A.appendChild(I),A.appendChild(R),
h.appendChild(A)}))},"render");return u&&f&&(u.onclick=()=>f.click(),f.onchange=S=>{for(const T of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(T.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(T.
size>$i){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(b.
length>=Ei){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}b.push({file:T,url:URL.createObjectURL(T)})}S.target.value="",_()}),{files:r(()=>b.
map(S=>S.file),"files"),clear:r(()=>{b.forEach(S=>URL.revokeObjectURL(S.url)),b=[],_()},"clear")}}r(
oi,"createFbImagePicker");async function ri(u,f){try{const h=new FormData;return f.forEach(b=>h.append(
"images",b,b.name)),(await apiFetch(u,{method:"POST",body:h})).ok}catch{return!1}}r(ri,"uploadFbImag\
es");const li=oi(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const u=get("fb-title").value.trim(),f=get("fb-message").value.trim();if(!f){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const h={title:u,message:f,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=get("fb-attach-logs"),_=b&&b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;_&&(h.client_logs=_);const S=get("fb-attach-chat"),T=!!(S&&S.checked);if(T&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}T&&(h.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const A=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)}),I=A.ok?await A.clone().json().
catch(()=>({})):{},R=li.files(),F={count:R.length,saved:!0};A.ok&&F.count&&(F.saved=await ri(`/api/f\
eedback/${I.public_id}/images`,R)),(window.ActivityLog?await window.ActivityLog.reportFeedback(A,_,T,
F):A.ok)&&(get("fb-title").value="",get("fb-message").value="",li.clear(),S&&(S.checked=!1),b&&(b.checked=
!0),hn())};async function ci(u){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${u}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),hn()}r(ci,"deleteFeedback");const Ii="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Gn=[];function di(u,f){const h=u.replies||[];return h.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${h.map(b=>{const _=b.role==="admin"?f&&b.author?`\u7BA1\u7406\u8005 (${b.
author})`:"\u7BA1\u7406\u8005":f?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=f?[b.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${b.
log_file}`,b.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${b.chat_file}`,b.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${b.image_count}\u679A\uFF09: ${b.image_dir}`]:[b.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${b.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(_)}\
 / ${escapeHtml(b.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${b.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(b.message)}</div>${S.filter(Boolean).map(T=>`<div cla\
ss="${f?"text-amber-300":"text-gray-400"}">${escapeHtml(T)}</div>`).join("")}</div>`}).join("")}</di\
v>`:""}r(di,"feedbackRepliesHtml");function ui(){return`<div class="fb-reply-form mt-2 space-y-2">
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
                            <span class="block text-xs font-bold text-white">${Ii}</span>
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
                </div>`}r(ui,"feedbackReplyFormHtml");function pi(u,f){const h=f.querySelector(".fb-\
reply-form");if(!h)return;const b=h.querySelector(".fb-reply-logs"),_=h.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());b.disabled=!S,_.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const T=oi(h.querySelector(".fb-reply-images-add"),h.querySelector(".fb-reply-images-input"),h.querySelector(
".fb-reply-images-list"));Gn.push(T);const A=h.querySelector(".fb-reply-send");A.onclick=async()=>{const I=h.
querySelector(".fb-reply-message").value.trim();if(!I){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const R={message:I,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},F=b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;F&&(R.client_logs=F);const Q=h.
querySelector(".fb-reply-chat").checked;if(Q&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}Q&&(R.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const oe=T.files();A.disabled=!0;try{const ce=await apiFetch(
`/api/feedback/${u.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(R)}),K=await ce.json().catch(()=>({}));if(!ce.ok){showToast(K.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const ve=!oe.length||await ri(`/api/feedback/${u.public_id}/replies/${K.reply_id}\
/images`,oe),W=[];if(F&&K.logs_saved===!1&&W.push("\u30ED\u30B0"),ve||W.push("\u753B\u50CF"),Q&&K.chat_copy_saved!==
!0&&W.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),W.length)showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${W.
join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,"error",!0);else{
const ge=["\u8FD4\u4FE1"];Q&&ge.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),F&&ge.push(
`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof K.logs_count=="number"?K.logs_count:
F.entries.length}\u4EF6\uFF09`),oe.length&&ge.push(`\u753B\u50CF\uFF08${oe.length}\u679A\uFF09`),showToast(
ge.join(ge.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","success")}hn()}finally{
A.disabled=!1}}}r(pi,"bindFeedbackReply");function mi(){const u=get("fb-admin-status-filter").value;
let f=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(h=>{const b=u==="all"||
(u==="unresolved"?["new","in_review","replied"].includes(h.dataset.feedbackStatus):h.dataset.feedbackStatus===
u);h.hidden=!b,b&&f++}),get("fb-admin-empty").classList.toggle("hidden",f!==0)}r(mi,"filterAdminFeed\
back"),get("fb-admin-status-filter").onchange=mi;async function hn(){Gn.forEach(S=>S.clear()),Gn=[];
const f=await(await apiFetch("/api/feedback?all=1")).json(),h=get("fb-list");h.innerHTML="",(f.items||
[]).filter(S=>!f.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(f.is_admin)return;
const T=document.createElement("div");T.className="p-2 rounded border border-gray-700 bg-gray-800/50",
T.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
S.public_id||"")}</span> / ${S.created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.
image_count}\u679A</div>`:""}${di(S,!1)}${ui()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
T.querySelector(".fb-delete").onclick=()=>ci(S.public_id),pi(S,T),h.appendChild(T)});const b=get("fb\
-admin-panel"),_=get("fb-admin-list");f.is_admin?(b.classList.remove("hidden"),_.innerHTML="",(f.items||
[]).forEach(S=>{const T=document.createElement("div");T.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",T.dataset.feedbackStatus=S.status||"new",T.innerHTML=`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(S.public_id||"")}</span> / ${S.agent_name?`agent:${escapeHtml(S.agent_name)}`:`us\
er:${S.user_id}`} / ${S.created_at}${S.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(S.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div>
                            ${[S.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${S.log_file}`,S.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${S.chat_file}`,S.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${S.
image_count}\u679A\uFF09: ${S.image_dir}`].filter(Boolean).map(A=>`<div class="text-[11px] text-ambe\
r-300">${escapeHtml(A)}</div>`).join("")}
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
                            ${di(S,!0)}
                            ${ui()}
                        `,T.querySelector(".fb-status").value=S.status||"new",T.querySelector(".fb-s\
ave").onclick=async()=>{const A=T.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:A})}),
hn()},T.querySelector(".fb-delete").onclick=()=>ci(S.public_id),pi(S,T),_.appendChild(T)}),mi()):b.classList.
add("hidden")}if(r(hn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=
f.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get(
"totp-verify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),h=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),_=await b.json();if(!b.ok){showToast(_.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const T=await(await ensureWebAuthnJson()).create({publicKey:_}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},T,{name:h}))}),I=await A.json().catch(()=>({}));A.ok?(f&&(f.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(I.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(b){showToast(`WebAuthn Error: ${b}`,"error",!0)}finally{u&&(u.disabled=!1)}},window.
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
return}if(u.key==="Enter"){u.preventDefault();const b=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(f.value),f.value));b[slashSelectedIndex]?selectSlashCommand(b[slashSelectedIndex].
id):b.length>0&&selectSlashCommand(b[0].id);return}if(u.key==="Escape"){u.preventDefault(),hideSlashCommandSuggestions();
return}}if(skillSuggestionsVisible){const h=f.value.trim();if(u.key==="ArrowDown"){u.preventDefault(),
skillSelectedIndex=skillSelectedIndex+1,showSkillSuggestions(h.substring(1));return}if(u.key==="Arro\
wUp"){u.preventDefault(),skillSelectedIndex=Math.max(skillSelectedIndex-1,0),showSkillSuggestions(h.
substring(1));return}if(u.key==="Enter"){u.preventDefault();const b=h.substring(1).toLowerCase(),_=loadedSkills.
filter(S=>S.name.toLowerCase().includes(b)||S.description&&S.description.toLowerCase().includes(b));
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
click()}),get("sts-cancel-btn")&&(get("sts-cancel-btn").onclick=()=>{isStsModel()&&Pi()}),get("promp\
t-input")&&get("prompt-input").addEventListener("paste",async u=>{const f=(u.clipboardData||window.clipboardData).
items,h=[];for(let b=0;b<f.length;b++)if(f[b].kind==="file"){const _=f[b].getAsFile();_&&h.push(_)}h.
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
let pt,He=null,Tn=[],Un=!1,An=null,nn=null,zt=null,yn=null,zn=0,En=!1,vn=null,sn=null,Rt=null,wn=null,
$n=null,Wt=null,In=null,Pn=null;function fi(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray(
In)&&In.length)return In;u.innerHTML="";const f=[];for(let h=0;h<24;h++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",f.push(b),u.appendChild(b)}return In=f,f}r(fi,"ensureMicWaveformBars");function Vt(u,f="hidden"){
const h=get("mic-recording-indicator"),b=get("mic-recording-text");if(h){if(Pn&&(clearTimeout(Pn),Pn=
null),f==="hidden"){h.classList.add("hidden");return}b&&u&&(b.innerText=u),h.classList.remove("hidde\
n"),f==="recording"?h.style.color="rgb(252 165 165)":f==="processing"?h.style.color="rgb(253 224 71)":
h.style.color="rgb(209 213 219)"}}r(Vt,"setMicRecordingIndicator");function gi(){fi().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}r(gi,"resetMicWaveformBars");function Jt(){if($n&&(cancelAnimationFrame(
$n),$n=null),wn){try{wn.disconnect()}catch{}wn=null}if(sn){try{sn.close()}catch{}sn=null}Rt=null,Wt=
null,gi()}r(Jt,"stopMicWaveform");function hi(u){Jt();const f=fi();if(!f.length)return;const h=window.
AudioContext||window.webkitAudioContext;if(!h)return;try{sn=new h,Rt=sn.createAnalyser(),Rt.fftSize=
256,Rt.smoothingTimeConstant=0,wn=sn.createMediaStreamSource(u),wn.connect(Rt),Wt=new Uint8Array(Rt.
frequencyBinCount)}catch{Jt();return}const b=r(()=>{if(!Rt||!Wt)return;Rt.getByteFrequencyData(Wt);const _=Math.
max(1,Math.floor(Wt.length/f.length));for(let S=0;S<f.length;S++){const A=(Wt[Math.min(Wt.length-1,S*
_)]||0)/255,I=Math.max(2,Math.round(2+A*10));f[S].style.height=`${I}px`,f[S].style.opacity=`${.35+A*
.65}`}$n=requestAnimationFrame(b)},"render");b()}r(hi,"startMicWaveform");function On(){if(An&&(clearInterval(
An),An=null),yn){try{yn.disconnect()}catch{}yn=null}if(nn){try{nn.close()}catch{}nn=null}zt=null}r(On,
"stopSilenceMonitor");function bi(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;On();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;nn=new f,zt=nn.createAnalyser(),zt.fftSize=2048,
yn=nn.createMediaStreamSource(u),yn.connect(zt);const h=new Uint8Array(zt.fftSize),b=getStsSilenceMs(),
_=.02;zn=0,En=!1,An=setInterval(()=>{if(!zt)return;zt.getByteTimeDomainData(h);let S=0;for(let A=0;A<
h.length;A++){const I=(h[A]-128)/128;S+=I*I}if(Math.sqrt(S/h.length)>_){En||(En=!0),zn=Date.now();return}
En&&Date.now()-zn>b&&pt&&pt.state==="recording"&&pt.stop()},200)}r(bi,"startSilenceMonitor");const Jn=class Jn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(f,h,b,_={},S=null){if(this.
model=b,this.ws=new WebSocket(`${h}?access_token=${encodeURIComponent(f)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const I=!!(_&&
_.transcriptionConfig),R={setup:{model:`models/${b}`,generationConfig:{responseModalities:I?["TEXT"]:
["AUDIO"]},inputAudioTranscription:I?_.transcriptionConfig||{}:{}}};I||(R.setup.outputAudioTranscription=
{}),_.speechConfig&&(R.setup.generationConfig.speechConfig=_.speechConfig),_.thinkingConfig&&(R.setup.
generationConfig.thinkingConfig=_.thinkingConfig),_.translationConfig&&(R.setup.generationConfig.translationConfig=
_.translationConfig),console.log("Sending setup:",JSON.stringify(R)),this.ws.send(JSON.stringify(R))},
this.ws.onmessage=I=>this._handleMessage(I),this.ws.onerror=I=>{console.error("Gemini Live WebSocket\
 error:",I),this.onError&&this.onError(I)},this.ws.onclose=I=>{console.log("Gemini Live WebSocket cl\
osed:",I.code,I.reason),this.closedEvent=I,this.onClose&&this.onClose(I)},S&&S.stream&&S.ctx&&S.source){
this.stream=S.stream,this.audioContext=S.ctx;var T=S.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(Nn());const I=yi(this.stream,16e3);this.audioContext=I.ctx;var T=I.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=I=>{I.data.
size>0&&this.userAudioChunks.push(I.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
I=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const R=vi(I.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!R||!R.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(R))),mimeType:"audio/\
pcm;rate=16000"}}}))},T.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(f){
let h=null;try{h=JSON.parse(typeof f.data=="string"?f.data:new TextDecoder().decode(f.data))}catch{return}
if(h.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),h.serverContent){
const b=h.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(_=>{if(_.text&&(_.thought?(console.log(
"Gemini thought delta:",_.text),this.assistantThought+=_.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(_.text)),_.inlineData&&_.inlineData.data){const S=_.inlineData.data;
console.log("Gemini audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);
const T=atob(S),A=new Uint8Array(T.length);for(let I=0;I<T.length;I++)A[I]=T.charCodeAt(I);this.assistantAudioChunks.
push(A)}}),b.outputTranscription&&b.outputTranscription.text&&this._appendAssistantText(b.outputTranscription.
text),b.inputTranscription&&b.inputTranscription.text&&(this.userTurnBreak&&this.inputTranscript&&!this.
inputTranscript.endsWith(`
`)&&(this.inputTranscript+=`
`),this.userTurnBreak=!1,this.inputTranscript+=b.inputTranscription.text,this.interimInputTranscript=
""),b.interimInputTranscription&&(this.interimInputTranscript=b.interimInputTranscription.text||""),
b.turnComplete&&(this.assistantTurnBreak=!0,this.model!=="gemini-3.5-transcribe-live"&&(this.userTurnBreak=
!0))}h.error&&this.onError&&this.onError(h.error),this.onMessage&&this.onMessage(h)}_appendAssistantText(f){
f&&(this.assistantTurnBreak&&this.assistantText&&!this.assistantText.endsWith(`
`)&&(this.assistantText+=`
`),this.assistantTurnBreak=!1,this.assistantText+=f)}stop(){const f=this.backupRecorder;if(this.backupRecorder=
null,this._stopped=new Promise(b=>{let _=!1;const S=r(()=>{_||(_=!0,b())},"finish");if(!f||f.state===
"inactive"){S();return}f.addEventListener("stop",S,{once:!0});try{f.stop()}catch{S()}setTimeout(S,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const h=r(
()=>{if(this.stream&&this.stream.getTracks().forEach(b=>{try{b.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(h)}async getFinalData(){this._stopped&&
await this._stopped;const f=new Blob(this.assistantAudioChunks),h=await this._blobToBase64(f),b=new Blob(
this.userAudioChunks),_=await this._blobToBase64(b);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:h,user_audio_base64:_}}_blobToBase64(f){
return new Promise(h=>{const b=new FileReader;b.onloadend=()=>h(b.result.split(",")[1]),b.readAsDataURL(
f)})}};r(Jn,"GeminiLiveClient");let Wn=Jn;const Kn=class Kn{constructor(f=24e3){const h=window.AudioContext||
window.webkitAudioContext;this.ctx=new h({sampleRate:f}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(f){if(!this.ctx)return;const h=atob(f),b=new Uint8Array(h.length);for(let R=0;R<
h.length;R++)b[R]=h.charCodeAt(R);const _=new Int16Array(b.buffer),S=new Float32Array(_.length);for(let R=0;R<
_.length;R++)S[R]=_[R]/32768;const T=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);T.getChannelData(
0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();
A.buffer=T,A.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const I=Math.max(this.ctx.currentTime,this.nextStartTime);A.start(I),this.
nextStartTime=I+T.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r(Kn,"RealTimeAudioPl\
ayer");let xn=Kn;const Xn=class Xn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=null,this.rtPlayer=
null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript="",this.assistantThought=
"",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.saved=
!1,this.saving=!1,this.stopping=!1,this.audioQueue=[],this.audioFlush=null}isActive(){return this.active}async start(){
if(this.active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const f=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const _=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=_.id!==null&&_.id!==void 0?String(_.id):_.id,setTemporaryChatUiState(!!(_&&_.
is_temporary)),setCurrentChatHeaderTitle(_&&_.title),applyTemporaryChatRuntimeMeta(_||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+_.id),get("welcome-screen").classList.add("hidden")}catch(b){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+b.message,"\
error",!0);return}const h={model:f,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||f==="gpt-realtime-translate")&&get("sts-targ\
et-lang")?get("sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(h.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const b=await apiFetch(
"/api/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
h)}),_=await b.json().catch(()=>({}));if(!b.ok)throw new Error(_.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=_.session_id,this.rateIn=_.rate_in||this.rateIn,this.rateOut=_.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null,this.audioQueue=[],this.audioFlush=null}catch(b){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",
!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
b.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(b){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
b.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const f="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),h=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(f,h).then(b=>{if(!b.ok)throw new Error("SSE stream failed ("+b.status+")");this.
reader=b.body.getReader(),this._readLoop()}).catch(b=>{b&&b.name==="AbortError"||(this.streamError=b&&
b.message?b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const f=new TextDecoder;let h="";try{for(;this.reader;){const{done:b,
value:_}=await this.reader.read();if(b)break;h+=f.decode(_,{stream:!0});let S;for(;(S=h.indexOf(`

`))>=0;){const T=h.slice(0,S);h=h.slice(S+2);for(const A of T.split(`
`)){if(!A.startsWith("data: "))continue;let I=null;try{I=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent(I)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
if(f)switch(f.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new xn(this.rateOut||24e3),an=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(f.data));break;case"transcript":f.role==="user"?(f.cumulative?this.userTranscript=
f.delta:this.userTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
f.role==="assistant"?(this.assistantTranscript+=f.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):f.role==="thought"&&(this.assistantThought+=f.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":f.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"notice":f.message&&showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0: "+f.message,
"warning",!0);break;case"error":this.streamError=f.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}an=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(Nn()).then(f=>{this.stream=f;const h=this.rateIn||24e3,b=yi(f,h);this.audioCtx=
b.ctx;const _=b.source,S=this.audioCtx.sampleRate,T=4096;this.processor=this.audioCtx.createScriptProcessor(
T,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const I=A.inputBuffer.
getChannelData(0),R=vi(I,S,h);!R||!R.byteLength||this._sendAudio(R)},_.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(f){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(f)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const f=this.audioQueue.splice(0,this.audioQueue.length),
h=f.reduce((I,R)=>I+R.byteLength,0),b=new Uint8Array(h);let _=0;f.forEach(I=>{b.set(I,_),_+=I.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),T={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(T):T;try{await fetch(S,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}On(),Jt()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const f=Date.now(),h=this.responseDoneCount;let b=this.lastAudioAt;for(;Date.now()-f<2e4&&!(this.responseDoneCount>
h||(this.lastAudioAt>b&&(b=this.lastAudioAt),!this.speechActive&&Date.now()-f>2e3&&Date.now()-b>2500));)
await new Promise(_=>setTimeout(_,250));await this._save()}async _save(){if(!this.saved){this.saved=
!0,this.saving=!0;try{const f=await apiFetch("/api/realtime/save",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify({session_id:this.sessionId,thread_id:currentThreadId})}),
h=await f.json().catch(()=>({}));if(!f.ok)throw new Error(h.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(h.thread_id&&!currentThreadId&&(currentThreadId=String(h.thread_id)),this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(f){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(f&&f.message?f.message:f),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this.audioQueue=[],this._stopCapture(),this._stopPlayback(),this.abortCtrl){
try{this.abortCtrl.abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const f=get(
"mic-btn");f&&(f.classList.remove("bg-red-600","animate-pulse"),f.classList.add("bg-gray-700"))}};r(
Xn,"RealtimeVoiceSession");let Vn=Xn;function yi(u,f){const h=window.AudioContext||window.webkitAudioContext;
if(!h)throw new Error("AudioContext not supported");let b=null;try{return b=new h({sampleRate:f}),{ctx:b,
source:b.createMediaStreamSource(u)}}catch{if(b)try{b.close()}catch{}return b=new h,{ctx:b,source:b.
createMediaStreamSource(u)}}}r(yi,"openMicAudioSource");function vi(u,f,h){let b=u;if(f!==h&&f>0&&h>
0){const S=f/h,T=Math.floor(b.length/S),A=new Float32Array(T);for(let I=0;I<T;I++)A[I]=b[Math.min(Math.
floor(I*S),b.length-1)];b=A}const _=new Int16Array(b.length);for(let S=0;S<b.length;S++){const T=Math.
max(-1,Math.min(1,b[S]));_[S]=T<0?T*32768:T*32767}return _.buffer}r(vi,"pcm16FromFloat32");const kn=new Vn,
wi=(()=>{const h={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,_=null,S=!1,T=null,A=!1,I=0,R=0,F=null,Q="idle",oe=!1,ce=null;const K=r(q=>document.getElementById(
q),"$"),ve=r(q=>{const te=Object.assign({},q||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(te):(te.progressSpinner=
!1,te)},"noSpinner");function W(q,te){Q=te;const $e=K("lyria-status-text"),ye=K("lyria-status-dot");
$e&&($e.textContent=q),ye&&(ye.className="w-2 h-2 rounded-full inline-block "+(h[te]||h.idle)),Le(),
ae()}r(W,"setStatus");function ge(){const q=R?Math.floor((Date.now()-R)/1e3):0,te=String(Math.floor(
q/60)).padStart(2,"0"),$e=String(q%60).padStart(2,"0");return`${te}:${$e}`}r(ge,"formatElapsed");function X(){
R||(R=Date.now());const q=K("lyria-elapsed");q&&(q.textContent=ge()),F||(F=window.setInterval(()=>{const te=K(
"lyria-elapsed");te&&(te.textContent=ge())},1e3))}r(X,"startElapsedTimer");function we(){F&&(window.
clearInterval(F),F=null)}r(we,"stopElapsedTimer");function Le(){const q=K("lyria-play-btn"),te=K("ly\
ria-pause-btn"),$e=K("lyria-stop-btn"),ye=K("lyria-reset-btn"),Ie=!!b,Je=Q==="streaming"||Q==="conne\
cting";if(q){q.disabled=oe;const at=q.querySelector("i");at&&(at.className="fas fa-play")}te&&(te.disabled=
oe||!Je),$e&&($e.disabled=oe||!Ie||!Je),ye&&(ye.disabled=oe||!Ie||!Je)}r(Le,"updateTransportButtons");
function ae(){const q=K("lyria-save-btn");if(!q)return;const te=!!b&&Q!=="idle"&&Q!=="connecting"&&Q!==
"error";q.classList.toggle("hidden",!te)}r(ae,"updateSaveButton");function ke(q,te){const $e=K("lyri\
a-prompt-rows");if(!$e)return;const ye=document.createElement("div");ye.className="flex items-center\
 gap-2",ye.innerHTML=`
                        <input type="text" value="${escapeHtml(q||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof te=="nu\
mber"?te:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof te=="number"?te:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Ie=ye.querySelector('input[type="range"]'),Je=ye.querySelector(".lyria-w\
eight-label");Ie&&Je&&Ie.addEventListener("input",()=>{Je.textContent=parseFloat(Ie.value).toFixed(1)});
const at=ye.querySelector(".lyria-prompt-remove");at&&at.addEventListener("click",()=>{$e.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ye.remove()}),ye.classList.add("lyria-prompt-row-wrap"),$e.appendChild(
ye)}r(ke,"addPromptRow");function Ne(){const q=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),te=[];return q.forEach($e=>{const ye=$e.querySelector('input[type="text"]'),Ie=$e.
querySelector('input[type="range"]'),Je=(ye?ye.value:"").trim();Je&&te.push({text:Je,weight:parseFloat(
Ie?Ie.value:1)||1})}),te}r(Ne,"collectPrompts");function Re(){const q={},te=r(Bi=>{const Yn=K(Bi);return Yn&&
Yn.value!==""?parseFloat(Yn.value):void 0},"num"),$e=te("lyria-bpm");$e!==void 0&&(q.bpm=Math.round(
$e));const ye=te("lyria-guidance");ye!==void 0&&(q.guidance=ye);const Ie=te("lyria-density");Ie!==void 0&&
(q.density=Ie);const Je=te("lyria-brightness");Je!==void 0&&(q.brightness=Je);const at=te("lyria-tem\
perature");at!==void 0&&(q.temperature=at);const Qe=K("lyria-scale");Qe&&Qe.value&&(q.scale=Qe.value);
const lt=K("lyria-mode");lt&&lt.value&&(q.music_generation_mode=lt.value);const wt=K("lyria-mute-bas\
s"),Xt=K("lyria-mute-drums"),Li=K("lyria-only-bass-drums");return wt&&(q.mute_bass=wt.checked),Xt&&(q.
mute_drums=Xt.checked),Li&&(q.only_bass_and_drums=Li.checked),q}r(Re,"collectConfig");function mt(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([te,$e])=>{const ye=K(te),Ie=K($e);!ye||!Ie||ye.addEventListener("input",()=>{const Je=parseFloat(
ye.value);Ie.textContent=te==="lyria-bpm"?String(Math.round(Je)):Je.toFixed(1)})})}r(mt,"bindRangeLa\
bels");function Ze(){if(T){try{T.close()}catch{}T=null}A=!1,I=0}r(Ze,"resetPlayback");function vt(){
if(S=!1,_&&typeof _.abort=="function")try{_.abort()}catch{}_=null}r(vt,"closeStream");async function Bt(){
vt(),_=new AbortController,S=!0;try{const q=await fetch(`/api/skillini/music/stream?session_id=${encodeURIComponent(
b)}`,ve({method:"GET",signal:_.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!q.
ok){const Ie=await q.json().catch(()=>({}));throw new Error(Ie.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const te=q.body.getReader(),$e=new TextDecoder;let ye="";for(;S;){const{done:Ie,value:Je}=await te.read();
if(Ie)break;ye+=$e.decode(Je,{stream:!0});const at=ye.split(`

`);ye=at.pop();for(const Qe of at){const lt=Qe.split(`
`).find(Xt=>Xt.startsWith("data: "));if(!lt)continue;const wt=lt.slice(6);try{const Xt=JSON.parse(wt);
Ee(Xt)}catch{}}}}catch(q){if(q&&q.name==="AbortError")return;S&&(W("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&b&&Bt()},1200))}finally{S=!1}}r(Bt,"openStream");function Ee(q){
if(q&&q.snapshot){const te=q.status;if(te==="error"){W("\u30A8\u30E9\u30FC","error"),we();return}if(te===
"closed"||te==="stopped"){W("\u7D42\u4E86","closed"),we();return}W(te==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",te==="paused"?"paused":"connecting");return}if(q&&q.audio){W("\u518D\u751F\u4E2D...",
"streaming"),X(),Me(q.audio);return}if(q&&q.error){W("\u30A8\u30E9\u30FC: "+q.error,"error"),we();return}
if(q&&q.final){W("\u7D42\u4E86","closed"),we(),Le();return}}r(Ee,"handleStreamMessage");function Me(q){
if(!q)return;if(!T){const Qe=window.AudioContext||window.webkitAudioContext;if(!Qe)return;T=new Qe({
sampleRate:48e3}),A=!1,I=0}let te;try{const Qe=atob(q);te=new Uint8Array(Qe.length);for(let lt=0;lt<
Qe.length;lt++)te[lt]=Qe.charCodeAt(lt)}catch{return}const $e=new Int16Array(te.buffer),ye=Math.floor(
$e.length/2);if(ye<1)return;const Ie=T.createBuffer(2,ye,48e3);for(let Qe=0;Qe<2;Qe++){const lt=Ie.getChannelData(
Qe);for(let wt=0;wt<ye;wt++)lt[wt]=$e[wt*2+Qe]/32768}T.state==="suspended"&&T.resume();const Je=T.createBufferSource();
Je.buffer=Ie,Je.connect(T.destination),A||(I=T.currentTime+.08,A=!0);const at=Math.max(T.currentTime,
I);Je.start(at),I=at+Ie.duration}r(Me,"playChunk");async function We(q,te){const $e=await fetch("/ap\
i/skillini/music/command",ve({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:q},te||{}))})),ye=await $e.json().catch(()=>({}));if(!$e.ok)throw new Error(
ye.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ye}
r(We,"apiCommand");async function st(){if(oe)return;const q=Ne();if(!q.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}oe=!0,Le(),W("\u63A5\u7D9A\u4E2D...","connecting");try{
const te=await fetch("/api/skillini/music/start",ve({method:"POST",headers:{"Content-Type":"applicat\
ion/json"},body:JSON.stringify({weighted_prompts:q,config:Re()})})),$e=await te.json().catch(()=>({}));
if(!te.ok)throw new Error($e.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=$e.session_id,ce=Re(),W("\u63A5\u7D9A\u4E2D...","connecting"),Bt()}catch(te){W("\u30A8\u30E9\u30FC: "+
te.message,"error"),showToast("Lyria RealTime: "+te.message,"error",!0)}finally{oe=!1,Le()}}r(st,"st\
artSession");async function ft(q){if(b){oe=!0,Le();try{await We("control",{action:q}),q==="PLAY"?W("\
\u518D\u751F\u4E2D...","streaming"):q==="PAUSE"?W("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):q==="ST\
OP"?W("\u505C\u6B62\u4E2D","stopped"):q==="RESET_CONTEXT"&&W("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0),W("\u30A8\u30E9\u30FC: "+
te.message,"error")}finally{oe=!1,Le()}}}r(ft,"control");async function jt(){if(!b)return;const q=Ne();
if(!q.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}oe=!0;try{await We("prompts",{weighted_prompts:q}),W("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
Q==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0)}finally{oe=!1,Le()}}r(jt,"a\
pplyPrompts");async function Te(){if(!b)return;const q=Re(),te=ce||{},$e=q.bpm!==void 0&&q.bpm!==te.
bpm,ye=q.scale!==void 0&&q.scale!==te.scale,Ie=$e||ye;oe=!0;try{await We("config",{config:q,reset_context:Ie}),
ce=q,W(Ie?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",Q==="paused"?"paused":"streaming"),showToast(
Ie?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Je){showToast("Lyria RealT\
ime: "+Je.message,"error",!0)}finally{oe=!1,Le()}}r(Te,"applyConfig");async function qe(){if(b){oe=!0,
W("\u4FDD\u5B58\u4E2D...","connecting"),Le();try{const q=await fetch("/api/skillini/music/save",ve({
method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),te=await q.json().catch(()=>({}));if(!q.ok)throw new Error(te.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),we(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),te.thread_id&&(currentThreadId=String(te.thread_id),history.pushState({},"","/c/"+te.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(te.thread_id||currentThreadId),Kt(
!0)}catch(q){W("\u30A8\u30E9\u30FC: "+q.message,"error"),showToast("Lyria RealTime: "+q.message,"err\
or",!0)}finally{oe=!1,Le()}}}r(qe,"saveSession");async function De(){if(vt(),b)try{await fetch("/api\
/skillini/music/cancel",ve({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:b})}))}catch{}b=null,we(),Ze(),W("\u6E96\u5099\u5B8C\u4E86","idle")}r(De,"cancelSession");
function Ve(){const q=K("lyria-prompt-rows");q&&(q.innerHTML=""),ke("",1),ce=null,R=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ye=>{const Ie=K(ye);
Ie&&(Ie.value=ye==="lyria-bpm"?"120":ye==="lyria-guidance"?"4":ye==="lyria-temperature"?"1.1":"0.5")});
const te=K("lyria-scale");te&&(te.value="");const $e=K("lyria-mode");$e&&($e.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ye=>{const Ie=K(ye);Ie&&(Ie.checked=
!1)}),mt()}r(Ve,"resetControls");function Kt(q){vt(),b&&fetch("/api/skillini/music/cancel",ve({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),
b=null,S=!1,we(),Ze(),hideModal("lyria-studio-modal")}r(Kt,"closeAndCleanup");function Sn(q){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const te=K("lyria-studio-modal");if(te&&te.classList.contains("modal-open")&&b){
if(q&&typeof q=="string"){const ye=K("lyria-prompt-rows");ye&&(ye.innerHTML=""),ke(q,1)}return}if(b&&
De(),Ve(),q&&typeof q=="string"){const ye=K("lyria-prompt-rows");ye&&(ye.innerHTML=""),ke(q,1)}b=null,
S=!1,we(),Ze(),W("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(Sn,"open");function Rn(){
const q=K("lyria-open-studio-btn");q&&q.addEventListener("click",()=>Sn(""));const te=K("lyria-studi\
o-close");te&&te.addEventListener("click",()=>Kt(!1));const $e=K("lyria-play-btn");$e&&$e.addEventListener(
"click",()=>{if(!b){st();return}ft("PLAY")});const ye=K("lyria-pause-btn");ye&&ye.addEventListener("\
click",()=>ft("PAUSE"));const Ie=K("lyria-stop-btn");Ie&&Ie.addEventListener("click",()=>ft("STOP"));
const Je=K("lyria-reset-btn");Je&&Je.addEventListener("click",()=>ft("RESET_CONTEXT"));const at=K("l\
yria-add-prompt-btn");at&&at.addEventListener("click",()=>ke("",1));const Qe=K("lyria-apply-prompts-\
btn");Qe&&Qe.addEventListener("click",jt);const lt=K("lyria-apply-config-btn");lt&&lt.addEventListener(
"click",Te);const wt=K("lyria-save-btn");wt&&wt.addEventListener("click",qe),mt(),Ve(),window.openLyriaStudio=
Sn}r(Rn,"init");function Cn(){oe||Kt(!1)}return r(Cn,"requestClose"),{init:Rn,open:Sn,requestClose:Cn}})();
wi.init(),window.closeLyriaStudio=()=>wi.requestClose(),(()=>{let u=null,f=null,h=null;const b="voic\
eDockSettingsOpen",_="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=r(ae=>document.getElementById(ae),"$");function T(){return isStsModel()&&voiceStudioUiEnabled!==!1}
r(T,"isStudioMode");function A(){const ae=get("model-select")?get("model-select").value:"",ke=S("voi\
ce-studio-title");ke&&(ae==="gpt-transcribe"||ae==="gpt-live-transcribe"||ae==="grok-voice-transcrib\
e-2.0-file"?ke.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ae===
"gemini-3.5-live-translate-preview"?ke.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ke.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(A,"updateTitle");function I(){const ae=S("v\
oice-studio-transcript");ae&&(ae.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${_}</div>`);const ke=S("sts-live-transcript");ke&&(ke.innerHTML="",ke.classList.add("hidden"))}
r(I,"resetTranscript");function R(ae,ke,Ne){const Re=ae.querySelectorAll(".voice-studio-line");let mt=null;
for(let Ze=Re.length-1;Ze>=0;Ze--)if(Re[Ze].dataset.role===ke){mt=Re[Ze];break}if(mt)mt.innerHTML=Ne;else{
const Ze=ae.querySelector(".voice-studio-placeholder");Ze&&Ze.remove();const vt=document.createElement(
"div");vt.className="voice-studio-line",vt.dataset.role=ke,vt.innerHTML=Ne,ae.appendChild(vt)}ae.classList.
remove("hidden"),ae.scrollTop=ae.scrollHeight}r(R,"writeLine");function F(ae,ke){if(!ke||!String(ke).
trim()||!T())return;const mt=`<span class="${ae==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ae==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
ke)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(Ze=>R(Ze,
ae,mt))}r(F,"log");function Q(ae,ke=!0){const Ne=S("sts-panel"),Re=S("sts-settings-toggle");if(Ne&&Ne.
classList.toggle("settings-open",!!ae),Re&&Re.setAttribute("aria-expanded",ae?"true":"false"),ke)try{
localStorage.setItem(b,ae?"1":"0")}catch{}}r(Q,"setSettingsOpen");function oe(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}r(oe,"readSettingsOpen");let ce=null;function K(){const ae=get("mod\
el-select")?get("model-select").value:"";ae!==ce&&(ce=ae,I()),Q(oe(),!1),A()}r(K,"syncDock");function ve(){
const ae=S("sts-panel"),ke=S("voice-studio-panel-host");ae&&ke&&ae.parentNode!==ke&&(u=ae.parentNode,
f=ae.nextSibling,ke.appendChild(ae));const Ne=S("file-preview"),Re=S("voice-studio-file-host");Ne&&Re&&
Ne.parentNode!==Re&&(h=Ne.parentNode,Re.appendChild(Ne),Re.classList.remove("hidden"))}r(ve,"movePan\
elIntoModal");function W(){const ae=S("sts-panel");ae&&u&&ae.parentNode!==u&&(f&&f.parentNode===u?u.
insertBefore(ae,f):u.appendChild(ae));const ke=S("file-preview");ke&&h&&ke.parentNode!==h&&h.appendChild(
ke);const Ne=S("voice-studio-file-host");Ne&&Ne.classList.add("hidden"),u=null,f=null,h=null}r(W,"mo\
vePanelBack");function ge(){if(!T())return;ve();const ae=S("sts-panel");ae&&ae.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}r(ge,"open");function X(){window.
VoiceStudioOpen=!1,W(),hideModal("voice-studio-modal")}r(X,"close");function we(){window.VoiceStudioOpen&&
X()}r(we,"closeIfOpen");function Le(){window.VoiceStudioOpen=!1;const ae=S("voice-studio-open-btn");
ae&&ae.addEventListener("click",()=>ge());const ke=S("voice-studio-close");ke&&ke.addEventListener("\
click",()=>X());const Ne=S("sts-settings-toggle");Ne&&Ne.addEventListener("click",()=>{const Re=S("s\
ts-panel");Q(!(Re&&Re.classList.contains("settings-open")))}),window.VoiceStudio={open:ge,close:X,closeIfOpen:we,
log:F,isStudioMode:T,syncDock:K},K()}return r(Le,"init"),{init:Le}})().init();let an=null;function xi(){
if(an&&(an.stop(),an=null),vn){try{vn.pause()}catch{}try{vn.src=""}catch{}vn=null}}r(xi,"stopStsPlay\
back");async function Di(u){xi();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=!0,f.playsInline=
!0,vn=f,await f.play(),new Promise(h=>{f.onended=()=>h("ended"),f.onerror=()=>h("error")})}r(Di,"pla\
yStsAudio");function Pi(){if(kn.isActive()){kn._cancel();return}if(He){He.stop(),He=null,xi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Jt();return}pt&&pt.state===
"recording"&&(Un=!0,pt.stop())}r(Pi,"cancelRecording");function Nn(){if(isStsModel())return{audio:!0};
const f=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},h={channelCount:1};return f.echoCancellation&&(h.echoCancellation=!1),f.
noiseSuppression&&(h.noiseSuppression=!1),f.autoGainControl&&(h.autoGainControl=!1),{audio:h}}r(Nn,"\
getMicCaptureConstraints");function Oi(){const u=window.AudioContext||window.webkitAudioContext;if(!u||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let f=null;try{f=new u({sampleRate:16e3})}catch{if(f){try{f.close()}catch{}f=null}f=new u}const h=f.
resume(),b=navigator.mediaDevices.getUserMedia(Nn());return{ctx:f,resumePromise:h,streamPromise:b}}r(
Oi,"beginGeminiLiveMic");function Ni(u){if(u){if(u.streamPromise)u.streamPromise.then(f=>{try{f.getTracks().
forEach(h=>h.stop())}catch{}}).catch(()=>{});else if(u.stream)try{u.stream.getTracks().forEach(f=>f.
stop())}catch{}if(u.ctx)try{u.ctx.close()}catch{}}}r(Ni,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(He){setStsStatus("Processing...",!0);const u=He;He=null,u.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const f=await u.
getFinalData();if(isGeminiLiveTranscribeModel()&&(f.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
f.assistant_text=(u.inputTranscript||"").trim(),f.assistant_thought="",!f.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const h=(f.user_text||
"").trim()||(f.assistant_text||"").trim()||(f.assistant_thought||"").trim(),b=!!(f.audio_base64||f.user_audio_base64);
if(!h&&!b){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const S=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(S.id),history.pushState(
{},"","/c/"+S.id),get("welcome-screen").classList.add("hidden")}f.thread_id=currentThreadId,f.model=
get("model-select").value,await apiFetch("/api/skillini/save_sts",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify(f)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(f){console.error("Failed to save G\
emini Live session:",f),setStsStatus("Error saving",!1)}return}if(kn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),kn.stop();return}if(pt&&
pt.state==="recording"){pt.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Vt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const h=new Audio;
h.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",h.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let h=null;try{h=Oi();
const _={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(_.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const W=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(ge=>ge.trim()).filter(Boolean);W.length&&(_.custom_vocabulary=
W.slice(0,1e3))}}else _.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(_.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",_.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(_.target_lang=get("sts-target-lang").value);const S=await h.streamPromise;try{await h.
resumePromise}catch{}h={stream:S,ctx:h.ctx,source:h.ctx.createMediaStreamSource(S)};const T=await apiFetch(
"/api/skillini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
_)});if(!T.ok)throw new Error("Failed to get session token");const{token:A,url:I}=await T.json(),R=get(
"model-select").value,F=get("sts-voice")?get("sts-voice").value:"Kore",Q=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",oe=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(He=new Wn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(He.rtPlayer=new xn),isGeminiLiveTranscribeModel()){
const W=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",ge={languageCodes:[]};
if((W==="SMART"||W==="VERBATIM")&&(ge.mode=W),get("sts-custom-vocab")){const X=get("sts-custom-vocab").
value.split(/[,、\n]/).map(we=>we.trim()).filter(Boolean);X.length&&(ge.customVocabulary=X.slice(0,
1e3))}await He.start(A,I,R,{transcriptionConfig:ge},h)}else if(isGeminiLiveTranslateModel()){const W=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await He.start(A,I,R,{translationConfig:{targetLanguageCode:W,
echoTargetLanguage:!0}},h)}else{const W={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:F}}}};
isGeminiLiveExtendedThinkingModel()&&(W.thinkingConfig={thinkingLevel:Q,includeThoughts:oe}),await He.
start(A,I,R,W,h)}h=null,pt=He.backupRecorder,pt.onstop=()=>{He&&get("mic-btn").click()};let ce=!0,K="\
live-sts-"+Date.now();He.onMessage=W=>{if(W.serverContent){if(isGeminiLiveTranscribeModel()){const ge=He.
interimInputTranscript,X=He.inputTranscript,we=X+(ge&&!X.endsWith(ge)?(X?`
`:"")+ge:""),Le=get("chat-messages");let ae=document.getElementById(K);ae||(ae=document.createElement(
"div"),ae.id=K,ae.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ae.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,Le.appendChild(ae),Le.scrollTop=Le.scrollHeight);const ke=ae.
querySelector(".message-content");ke.innerText=we||"\u8074\u304D\u53D6\u308A\u4E2D...",Le.scrollTop=
Le.scrollHeight,window.VoiceStudio&&X&&window.VoiceStudio.log("user",X);return}if(W.serverContent.modelTurn){
ce&&(setStsStatus("Gemini is speaking...",!1),ce=!1);const ge=get("chat-messages");let X=document.getElementById(
K);X||(X=document.createElement("div"),X.id=K,X.className="flex flex-col gap-2 mb-4 assistant-messag\
e bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",X.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ge.appendChild(X),ge.scrollTop=ge.scrollHeight);const we=X.
querySelector(".thought-container"),Le=X.querySelector(".message-content");He.assistantThought&&(we.
classList.remove("hidden"),we.innerText=He.assistantThought),Le.innerText=He.assistantText,ge.scrollTop=
ge.scrollHeight,window.VoiceStudio&&(He.inputTranscript&&window.VoiceStudio.log("user",He.inputTranscript),
He.assistantText&&window.VoiceStudio.log("assistant",He.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),hi(He.stream),bi(He.stream);const ve=He;ve.onError=W=>{const ge=W&&W.message?W.message:
typeof W=="string"?W:"";ge&&showToast("Gemini Live: "+ge,"error",!0)},ve.onClose=W=>{if(He===ve){if(W&&
W.code&&W.code!==1e3){const ge=W.reason?": "+W.reason:" (code "+W.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+ge,"error",!0)}get("mic-btn").click()}},ve.closedEvent&&
ve.onClose(ve.closedEvent);return}catch(b){if(Ni(h),He){try{He.stop()}catch{}He=null}showToast("Gemi\
ni Live connection failed: "+b.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await kn.start();return}isStsModel()||(gi(),Vt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const u=await navigator.mediaDevices.getUserMedia(Nn());pt=new MediaRecorder(u),Tn=[],Un=!1;const f=isStsModel();
pt.ondataavailable=h=>Tn.push(h.data),pt.onstop=async()=>{if(Un){Tn=[],get("file-preview").classList.
add("hidden"),u.getTracks().forEach(T=>T.stop()),On(),Jt(),f||(Vt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Pn=setTimeout(()=>Vt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const h=new Blob(Tn,{type:"audio/web\
m"}),b=new File([h],"recording.webm",{type:"audio/webm"}),_=new FormData;_.append("file",b),get("fil\
e-preview").classList.remove("hidden");const S=f;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const W=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=W.id!==null&&W.id!==void 0?String(W.id):W.id,setTemporaryChatUiState(!!(W&&W.
is_temporary)),setCurrentChatHeaderTitle(W&&W.title),applyTemporaryChatRuntimeMeta(W||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+W.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeSkill&&(threadSkillMap[currentThreadId]=activeSkill,pendingSkillForNewThread=null),_.append("m\
odel",get("model-select").value),_.append("thread_id",currentThreadId),get("sts-voice")&&_.append("s\
ts_voice",get("sts-voice").value||""),get("sts-reasoning-effort")&&_.append("sts_reasoning_effort",get(
"sts-reasoning-effort").value||""),get("sts-speed")&&_.append("sts_speed",get("sts-speed").value||""),
get("sts-rate-in")&&_.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&_.append(
"sts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&_.append("sts_thinking_leve\
l",get("sts-thinking-level").value||""),get("sts-include-thoughts")&&_.append("sts_include_thoughts",
get("sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const T=await apiFetch(
"/sts",{method:"POST",body:_});if(!T.ok){const ve=await T.json().catch(()=>({}));throw new Error(ve.
error||"Speech-to-speech failed")}const A=T.body.getReader(),I=new TextDecoder;let R="",F=null,Q=null;
stsOpt("sts-auto-play")&&(Q=new xn,an=Q),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let oe=!0,ce="",K="";for(;;){const{done:ve,value:W}=await A.read();if(ve)break;
R+=I.decode(W,{stream:!0});const ge=R.split(`
`);R=ge.pop();for(const X of ge){if(!X.trim())continue;const we=JSON.parse(X);if(we.error)throw new Error(
we.error);we.audio_delta&&Q&&(oe&&(setStsStatus("Playing response...",!1),oe=!1),await Q.addChunk(we.
audio_delta)),we.input_delta&&(ce+=we.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
ce)),we.transcript_delta&&(K+=we.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",K)),(we.final||we.audio_url)&&(F=we)}}window.VoiceStudio&&!ce.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),F&&(F.audio_url||F.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const T=get(
"set-mic-transcribe-mode");if(!!(T&&T.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}_.append("llm_model",get("model-select")&&get("model-select").value||"");const R=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:_})).json();if(R.transcript){const F=get("prompt-inp\
ut");F.value+=(F.value?" ":"")+R.transcript,F.style.height="auto",F.style.height=F.scrollHeight+"px"}else
showToast(R.error||"Transcription failed","error",!0)}}catch(T){showToast("Audio processing error: "+
T.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(T=>T.
stop()),On(),Jt(),S||Vt("","hidden"),S&&setStsStatus("Tap to speak",!1)}},pt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Vt("\u9332\u97F3\u4E2D\u2026","recording"),hi(u)),bi(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Jt(),isStsModel()||Vt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const _n=r((u,f)=>{if(!u)return;const h=u.querySelector("span");h?h.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),h=get("lib-download-btn"),b=get("lib-attach-btn"),_=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(f&&(f.disabled=u===0,_n(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
h&&(h.disabled=u===0,_n(h,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=u===0,_n(b,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),_&&(_.disabled=u!==1,_n(_,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=u!==1,_n(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const T=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",T&&u>0)}};function ki(u){lib.
attachMode=!!u}r(ki,"setLibAttachMode");const _i=r((u=!1)=>{ki(u),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(u=!1)=>{hideModal("lib-modal"),!u&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>_i(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>_i(!0)),
get("lib-search")){let u=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),u&&clearTimeout(u),u=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const u=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=u,get("lib-sort").onchange=()=>{const f=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,f),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0));const Ri=r(()=>{editingSkillUuid=null,get("skill-modal-title").textContent=
"Create New Skill",get("save-skill-btn").innerText="Create Skill",showModal("skill-modal"),get("skil\
l-name").value="",get("skill-desc").value="",get("skill-inst").value="",location.pathname!=="/skill"&&
history.pushState({modal:"skill"},"","/skill")},"openSkillModal");window.closeSkillModal=(u=!1)=>{hideModal(
"skill-modal"),!u&&location.pathname==="/skill"&&history.back()},get("add-skill-btn").onclick=()=>Ri(),
get("save-skill-btn").onclick=async()=>{const u=get("skill-name").value,f=get("skill-desc").value,h=get(
"skill-inst").value;if(u&&h){const b=editingSkillUuid?"PUT":"POST",_=editingSkillUuid?`/api/skills/${editingSkillUuid}`:
CHAT_CONFIG.urls.handleSkills;await apiFetch(_,{method:b,headers:{"Content-Type":"application/json"},
body:JSON.stringify({name:u,description:f,instruction:h})}),window.closeSkillModal(),loadSkills(),editingSkillUuid&&
activeSkill&&activeSkill.uuid===editingSkillUuid&&(activeSkill.name=u,activeSkill.instruction=h,applyActiveSkill(
activeSkill))}else alert("Name and Instruction are required.")},document.addEventListener("click",function(u){
if(u.target.closest(".edit-btn")){const h=u.target.closest(".edit-btn").getAttribute("data-id");beginEditMessage(
h)}if(u.target.closest(".code-toggle")){const f=u.target.closest(".code-toggle"),h=f.closest(".code-\
wrapper");if(!h)return;const b=h.classList.toggle("collapsed");h.setAttribute("data-collapsed",b?"tr\
ue":"false"),f.setAttribute("aria-expanded",b?"false":"true"),f.innerHTML=b?'<i class="fas fa-chevro\
n-down"></i>':'<i class="fas fa-chevron-up"></i>',f.title=b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",
f.setAttribute("aria-label",b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080")}if(u.target.closest("\
.download-btn")){const f=u.target.closest(".download-btn"),h=f.getAttribute("data-code"),b=(f.getAttribute(
"data-lang")||"txt").toLowerCase();if(h)try{const _=decodeURIComponent(h),S=new Blob([_],{type:"text\
/plain"}),T=URL.createObjectURL(S),A=document.createElement("a");A.href=T;let R={python:"py",javascript:"\
js",typescript:"ts",markdown:"md",html:"html",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"\
sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"\
go",rust:"rs",ruby:"rb",php:"php",perl:"pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"\
toml",ini:"ini",plaintext:"txt",text:"txt"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(R="txt");let F=`\
code.${R}`;b==="dockerfile"&&(F="Dockerfile"),b==="makefile"&&(F="Makefile"),A.download=F,document.body.
appendChild(A),A.click(),document.body.removeChild(A),URL.revokeObjectURL(T)}catch(_){console.error(
"Download failed",_)}}if(u.target.closest(".coding-target-btn")&&selectCodingTargetFromButton(u.target.
closest(".coding-target-btn")),u.target.closest(".copy-btn")){const f=u.target.closest(".copy-btn"),
h=f.getAttribute("data-code");h&&window.copyCode(f,h)}if(u.target.closest(".html-preview-btn")){const h=u.
target.closest(".html-preview-btn").getAttribute("data-code");h&&openHtmlCodePreview(h)}if(u.target.
closest(".canvas-preview-btn")){const f=u.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(
f)}}),document.querySelectorAll(".modal-overlay").forEach(u=>{u.addEventListener("click",f=>{f.target===
u&&si(u.id)})}),currentThreadId?loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){
const e=get("file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-pr\
ogress-bar"),a=get("file-preview-thumbs"),o=get("upload-modal-status-text"),l=get("upload-modal-tota\
l-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;if(a){const L=document.querySelectorAll(
"#upload-list .upload-row");a.innerHTML="",L.forEach((M,E)=>{const G=M.getAttribute("data-local-url"),
O=M.getAttribute("data-filename"),Y=M.querySelector("img.upload-preview")!==null;let P;if(Y){let H=G;
if(!H&&O){const re=O.replace(/^\d+\//,"");H=buildAttachmentPreviewUrl(re)}H&&(P=document.createElement(
"img"),P.src=H,P.className="thumb-item shadow-sm",P.dataset.viewerSrc=H,P.dataset.viewerFilename=O||
H.split("/").pop(),P.uploadRow=M,P.onclick=function(re){re.preventDefault(),openImageViewer(this.dataset.
viewerSrc,".thumb-item")},P.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}P||(P=
d("FILE")),P.style.animationDelay=`${E*32}ms`,a.appendChild(P)}),L.length>0?a.classList.remove("hidd\
en"):a.classList.add("hidden")}function d(L){const M=document.createElement("div");return M.className=
"thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bol\
d",M.innerText=L,M}r(d,"createFileThumb");const m=collectImageUrlsForSend(),g=uploadProgressState.total,
y=uploadProgressState.completed,v=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.
classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const w=get("send\
-btn"),x=get("mic-btn"),k=get("mask-btn"),C=isStopMode;if(v>0?(w&&(w.disabled=!0),x&&(x.disabled=!0),
k&&(k.disabled=!0)):C||(w&&(w.disabled=!1),x&&(x.disabled=!1),k&&(k.disabled=!1)),v>0){const L=`Prep\
aring... (${y}/${g})`;e.classList.remove("hidden"),t.innerText=L,o&&(o.innerText=`(${y}/${g})`);let M=y*
100,E=0;for(let Y in uploadProgressState.perFilePct)M+=uploadProgressState.perFilePct[Y],E++;const G=g>
0?M/(g*100)*100:0,O=`${Math.min(100,G)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=O),l&&c&&
(l.classList.remove("hidden"),c.style.width=O)}else o&&(o.innerText=""),l&&l.classList.add("hidden"),
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
document.querySelectorAll(".marker-edit-only").forEach(L=>L.classList.toggle("hidden",markerState.maskMode));
const l=get("marker-tool-mosaic"),c=get("marker-tool-crop");l&&l.classList.toggle("hidden",markerState.
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const d=get("marker-tool-draw"),m=get(
"marker-tool-auto");d&&(d.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
m&&(m.classList.toggle("hidden",!markerState.maskMode),m.classList.toggle("active",e==="auto")),d&&d.
classList.toggle("active",e==="draw");const g=get("marker-brush-control");g&&g.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
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
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=g-l-d,w=n-g-l,x=y-c-m,k=i-y-c,C=r((L,M,E)=>Number.
isFinite(L)?M>E?(M+E)/2:Math.min(E,Math.max(M,L)):0,"clampOffset");markerView.offsetX=C(markerView.offsetX,
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
height);const n=r((l,c,d=null,m=!1)=>{if(!l)return;const g=Math.max(0,l.x),y=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);d&&(t.fillStyle=d,t.fillRect(g,y,v,w)),t.save(),m&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(g+.5,y+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
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
length;t.set(c,w),e.push({path:c,source:d,name:m});return}const y=e[g];if(!y)return;const v=normalizeAttachmentSource(
y.source);(v==="unknown"&&d!=="unknown"||v==="library"&&d==="upload")&&(y.source=d),!normalizeAttachmentDisplayName(
y.name)&&m&&(y.name=m)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
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
o.naturalWidth||m,markerState.naturalHeight=o.naturalHeight||g;const y=l.getContext("2d");y&&y.clearRect(
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
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=g,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=o,y.dataset.defaultDisplayName=o,y.dataset.sendNameCustomized="";const v=escapeHtml(
o),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",x=c?`<img src="${m}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
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
            `;const k=y.querySelector(".upload-preview");k&&(k.onclick=()=>openFileViewer(d,getRowAttachmentName(
y)||o));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const L=y.
querySelector(".upload-remove");L&&(L.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const E=y.getAttribute("data-filename");E&&(currentImageUrls=currentImageUrls.
filter(G=>G!==E)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const M=y.querySelector(".upload-marker");return M&&(M.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,o),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:g}}r(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
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
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||x||"";openFileViewer(k,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
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
ok){const y=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}const o=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let y=0;y<c;y++){const v=y*l,w=Math.min(e.size,
v+l),x=e.slice(v,w);if(!await new Promise(C=>{const L=new XMLHttpRequest;L.open("POST","/upload/chun\
k",!0),L.setRequestHeader("X-CSRF-Token",csrfToken),L.upload.onprogress=E=>{if(E.lengthComputable&&t&&
t.bar){const G=v+E.loaded,O=Math.min(100,Math.floor(G/e.size*100));t.bar.style.width=`${O}%`,t.status&&
(t.status.textContent=`${O}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,O)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},L.onload=()=>{L.status>=200&&L.status<300?C(!0):C(!1)},L.
onerror=()=>C(!1);const M=new FormData;M.append("upload_id",o),M.append("index",String(y)),M.append(
"total",String(c)),M.append("chunk",x,e.name),L.send(M)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:o})}),m=await d.json();if(d.ok&&m&&m.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const x=t.row.querySelector(
"img.upload-preview");if(x){const k=m.filename.replace(/^\d+\//,"");x.src=buildAttachmentPreviewUrl(
k)}}const y=normalizeAttachmentPath(m.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||m.filename),setRowAttachmentSource(t.row,"upload"),y)){const w=isRowAttachmentNameCustomized(
t.row),x=defaultAttachmentDisplayName(y),k=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
x;t.row.dataset.defaultDisplayName=x,setRowAttachmentName(t.row,k)}return y&&setAttachmentSourceForPath(
y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(m.
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
t.row.remove(),o(),i(!1);return}if(t&&t.row){const y=t.row.getAttribute("data-local-url");y&&URL.revokeObjectURL(
y),t.row.removeAttribute("data-local-url");const v=t.row.querySelector("img.upload-preview");if(v){const w=d.
filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(w)}}const m=normalizeAttachmentPath(d.
filename);if(m&&currentImageUrls.push(m),t&&t.row&&(t.row.setAttribute("data-filename",m||d.filename),
setRowAttachmentSource(t.row,"upload"),m)){const y=isRowAttachmentNameCustomized(t.row),v=defaultAttachmentDisplayName(
m),w=y&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||v;t.row.dataset.defaultDisplayName=
v,setRowAttachmentName(t.row,w)}m&&setAttachmentSourceForPath(m,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(d.filenames)&&d.filenames.length?d.filenames:[d.filename]).
forEach(y=>addLibraryFileFromPath(y)),o(),i(!0)}else{const m=[502,503,504].includes(l.status)||l.status>=
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
o),c=r((m,g)=>{for(let y=0;y<g.length;y++)l.setUint8(m+y,g.charCodeAt(y))},"writeString");c(0,"RIFF"),
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
er">FILE</div>',g=e.row.querySelector(".upload-preview");g&&(g.outerHTML=m);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const w=e.row.getAttribute("data-filename"),x=w?buildFileUrl(w):
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
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},m=[],g=16,y="",v=null,w=null,x=null,k=null,C=null,L=!1,M=null;
const E=r(B=>{const D=e.getBoundingClientRect(),pe=(B.clientX-D.left)*(e.width/D.width),le=(B.clientY-
D.top)*(e.height/D.height);return{x:pe,y:le}},"getPoint"),G=r((B,D)=>({x:(B.x+D.x)/2,y:(B.y+D.y)/2}),
"getMid"),O=r((B,D)=>Math.hypot(B.x-D.x,B.y-D.y),"getDist");let Y=!1;const P=r(()=>{w||(w=document.createElement(
"canvas"),x=w.getContext("2d")),k||(k=document.createElement("canvas"),C=k.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),H=r(()=>{if(!t||!w||!k)return;const B=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=B,t.drawImage(k,0,0),t.restore()},"r\
enderDrawPreview"),re=r(()=>{C&&(C.strokeStyle=y,C.fillStyle=y,C.lineWidth=g,C.lineCap="round",C.lineJoin=
"round")},"applyMarkerBrush"),me=r(B=>{if(!B)return!1;if(m.length===0)return m.push(B),!0;const D=m[m.
length-1],pe=B.x-D.x,le=B.y-D.y,Z=Math.hypot(pe,le),de=Math.max(.35,g*.04);if(Z<de)return!1;const ee=Math.
max(1,g*.25),xe=Math.max(1,Math.ceil(Z/ee));for(let ue=1;ue<=xe;ue++){const Ce=ue/xe;m.push({x:D.x+pe*
Ce,y:D.y+le*Ce})}return!0},"appendStrokePoint"),ie=r(()=>{if(C&&(C.clearRect(0,0,k.width,k.height),m.
length!==0)){if(re(),m.length===1){const B=m[0];C.beginPath(),C.arc(B.x,B.y,g/2,0,Math.PI*2),C.fill();
return}if(C.beginPath(),C.moveTo(m[0].x,m[0].y),m.length===2)C.lineTo(m[1].x,m[1].y);else{for(let pe=1;pe<
m.length-2;pe++){const le=m[pe],Z=m[pe+1],de=G(le,Z);C.quadraticCurveTo(le.x,le.y,de.x,de.y)}const B=m[m.
length-2],D=m[m.length-1];C.quadraticCurveTo(B.x,B.y,D.x,D.y)}C.stroke()}},"renderStrokeLayer"),he=r(
(B,D)=>{if(!B||!D)return null;const pe=Math.min(B.x,D.x),le=Math.min(B.y,D.y),Z=Math.abs(B.x-D.x),de=Math.
abs(B.y-D.y);return{x:pe,y:le,w:Z,h:de}},"normalizeMosaicRect"),$=r(B=>{const D=n?Number(n.value||16):
16,pe=Math.max(6,Math.floor(D)),le=Math.floor(pe/2);return{x:B.x-le,y:B.y-le,w:pe,h:pe}},"buildMosai\
cRectFromPoint"),N=r(()=>{const B=document.createElement("canvas");B.width=e.width,B.height=e.height;
const D=B.getContext("2d");if(!D)return null;markerState.baseCanvas&&D.drawImage(markerState.baseCanvas,
0,0),D.drawImage(e,0,0);try{return D.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),V=r(B=>{if(!t||!B)return!1;const D=N();if(!D)return!1;const pe=n?Number(n.value||
16):16,le=Math.max(4,Math.floor(pe/2)),Z=Math.max(0,Math.floor(B.x)),de=Math.max(0,Math.floor(B.y)),
ee=Math.min(e.width,Math.ceil(B.x+B.w)),xe=Math.min(e.height,Math.ceil(B.y+B.h));if(ee<=Z||xe<=de)return!1;
for(let ue=de;ue<xe;ue+=le)for(let Ce=Z;Ce<ee;Ce+=le){const Be=Math.min(le,ee-Ce),ze=Math.min(le,xe-
ue),Ke=Math.min(e.width-1,Math.max(0,Ce+Math.floor(Be/2))),fe=(Math.min(e.height-1,Math.max(0,ue+Math.
floor(ze/2)))*e.width+Ke)*4,_e=D.data[fe],ne=D.data[fe+1],it=D.data[fe+2];t.fillStyle=`rgb(${_e},${ne}\
,${it})`,t.fillRect(Ce,ue,Be,ze)}return!0},"applyMosaicRect"),j=r(B=>{if(!t)return;if(i.set(B.pointerId,
{x:B.clientX,y:B.clientY}),i.size>=2){const pe=Array.from(i.values()),le=pe[0],Z=pe[1];a=!0,Y=!1,m=[],
L=!1,M=null,v=null,markerState.mosaicPreviewRect=null,o=O(le,Z)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=G(le,Z),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault();return}if(a||markerState.mode==="crop")return;Y=!0;const D=E(B);if(markerState.
mode==="auto"&&markerState.maskMode)v=D;else if(markerState.mode==="mosaic")L=!0,M=D,markerState.mosaicPreviewRect=
$(D),renderCropOverlay();else{if(P(),!x||!C)return;x.clearRect(0,0,w.width,w.height),x.drawImage(e,0,
0),C.clearRect(0,0,k.width,k.height),g=n?Number(n.value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),m=[],me(D),ie(),markerState.hasStroke=!0,H()}e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault()},"start"),U=r(B=>{if(i.has(B.pointerId)&&i.set(B.pointerId,{x:B.clientX,
y:B.clientY}),a&&i.size>=2){const pe=Array.from(i.values()),le=pe[0],Z=pe[1],de=G(le,Z),ee=O(le,Z)||
1,xe=l*(ee/o);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,xe)),markerView.
offsetX=c.x+(de.x-d.x),markerView.offsetY=c.y+(de.y-d.y),applyMarkerTransform(),B.preventDefault();return}
if(!Y||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){B.preventDefault();return}const D=E(
B);if(markerState.mode==="mosaic"){if(!L||!M)return;markerState.mosaicPreviewRect=he(M,D)||$(D),renderCropOverlay()}else
me(D)&&(ie(),H());B.preventDefault()},"move"),se=r(B=>{const D=Y;if(i.delete(B.pointerId),i.size<2&&
(a=!1),i.size===0){Y=!1;const pe=D&&B.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.
maskMode&&v?fillMaskSimilarRegion(v):!1;if(v=null,D&&t&&markerState.mode==="draw"&&m.length>0&&(ie(),
H()),D&&markerState.mode==="mosaic"&&M){const le=E(B);let Z=he(M,le);(!Z||Z.w<2||Z.h<2)&&(Z=$(M)),V(
Z)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(Z))}m=[],L=!1,M=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),D&&(markerState.mode!=="auto"||pe)&&saveMarkerHistory()}e.releasePointerCapture&&
e.releasePointerCapture(B.pointerId),B.preventDefault()},"end");e.addEventListener("pointerdown",j),
e.addEventListener("pointermove",U),e.addEventListener("pointerup",se),e.addEventListener("pointerca\
ncel",se)}r(initMarkerCanvas,"initMarkerCanvas");function fillMaskSimilarRegion(e){var Y;const t=get(
"marker-canvas"),n=markerState.baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,o=Math.floor(e.x),l=Math.floor(e.y);if(o<0||l<0||o>=i||l>=
a)return!1;const c=n.data,d=l*i+o,m=d*4;if(c[m+3]<16)return!1;const g=c[m],y=c[m+1],v=c[m+2],w=Number(
((Y=get("marker-auto-tolerance"))==null?void 0:Y.value)||32),x=new Uint8Array(i*a),k=[d],C=[],L=25e4,
M=r(P=>{const H=P*4;return c[H+3]>=16&&Math.max(Math.abs(c[H]-g),Math.abs(c[H+1]-y),Math.abs(c[H+2]-
v))<=w},"matches");for(x[d]=1;k.length;){const P=k.pop();if(x[P]===2||!M(P))continue;const H=Math.floor(
P/i);let re=P%i,me=re;for(;re>0&&x[H*i+re-1]!==2&&M(H*i+re-1);)re--;for(;me+1<i&&x[H*i+me+1]!==2&&M(
H*i+me+1);)me++;for(let ie=re;ie<=me;ie++)x[H*i+ie]=2;if(C.push(H,re,me+1),C.length/3>L)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const ie of[H-1,H+1]){if(ie<0||ie>=a)continue;let he=re;for(;he<=me;){const $=ie*
i+he;if(x[$]===0&&M($))for(x[$]=1,k.push($);he+1<=me&&x[ie*i+he+1]===0&&M(ie*i+he+1);)he++,x[ie*i+he]=
1;he++}}}if(!C.length)return!1;const E=t.getContext("2d");if(!E)return!1;const G=E.getImageData(0,0,
i,a),O=G.data;for(let P=0;P<C.length;P+=3){const H=C[P]*i+C[P+1],re=C[P]*i+C[P+2];for(let me=H;me<re;me++){
const ie=me*4;O[ie]=O[ie+1]=O[ie+2]=O[ie+3]=255}}return E.putImageData(G,0,0),markerState.hasStroke=
!0,!0}r(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,o=null,l=null,c=!1,
d=0,m=markerView.scale,g={x:0,y:0},y={x:0,y:0};const v=8,w=14,x=r((P,H,re)=>Math.min(re,Math.max(H,P)),
"clamp"),k=r(P=>{const H=e.getBoundingClientRect(),re=(P.clientX-H.left)*(e.width/H.width),me=(P.clientY-
H.top)*(e.height/H.height);return{x:re,y:me}},"getPoint"),C=r((P,H)=>({x:(P.x+H.x)/2,y:(P.y+H.y)/2}),
"getMid"),L=r((P,H)=>Math.hypot(P.x-H.x,P.y-H.y),"getDist"),M=r(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),E=r((P,H)=>{if(!H)return"move";const re=H.x,me=H.y,ie=H.x+H.
w,he=H.y+H.h,$=Math.abs(P.x-re)<=w,N=Math.abs(P.x-ie)<=w,V=Math.abs(P.y-me)<=w,j=Math.abs(P.y-he)<=w;
if($&&V)return"nw";if(N&&V)return"ne";if($&&j)return"sw";if(N&&j)return"se";if(V)return"n";if(j)return"\
s";if($)return"w";if(N)return"e";if(P.x>re+w&&P.x<ie-w&&P.y>me+w&&P.y<he-w)return"move";const se=P.x<
re?"left":P.x>ie?"right":null,B=P.y<me?"top":P.y>he?"bottom":null;if(se&&B){if(se==="left"&&B==="top")
return"nw";if(se==="right"&&B==="top")return"ne";if(se==="left"&&B==="bottom")return"sw";if(se==="ri\
ght"&&B==="bottom")return"se"}return se?se==="left"?"w":"e":B?B==="top"?"n":"s":"move"},"hitTest"),G=r(
P=>{if(markerState.mode!=="crop")return;if(n.set(P.pointerId,{x:P.clientX,y:P.clientY}),n.size>=2){const me=Array.
from(n.values()),ie=me[0],he=me[1];c=!0,i=!1,d=L(ie,he)||1,m=markerView.scale,g={x:markerView.offsetX,
y:markerView.offsetY},y=C(ie,he),e.setPointerCapture&&e.setPointerCapture(P.pointerId),P.preventDefault();
return}if(c)return;i=!0;const H=k(P),re=M();o=E(H,re),a=H,l=re?{x:re.x,y:re.y,w:re.w,h:re.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(P.pointerId),P.preventDefault()},"start"),O=r(P=>{if(markerState.
mode!=="crop")return;if(n.has(P.pointerId)&&n.set(P.pointerId,{x:P.clientX,y:P.clientY}),c&&n.size>=
2){const se=Array.from(n.values()),B=se[0],D=se[1],pe=C(B,D),le=L(B,D)||1,Z=m*(le/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Z)),markerView.offsetX=g.x+(pe.x-y.x),markerView.
offsetY=g.y+(pe.y-y.y),applyMarkerTransform(),renderCropOverlay(),P.preventDefault();return}if(!i||!a||
!l)return;const H=k(P),re=e.width,me=e.height,ie={x:l.x,y:l.y,w:l.w,h:l.h},he=l.x+l.w,$=l.y+l.h,N=r(
()=>{const se=x(H.x,0,he-v);ie.x=se,ie.w=he-se},"applyW"),V=r(()=>{ie.w=x(H.x-l.x,v,re-l.x)},"applyE"),
j=r(()=>{const se=x(H.y,0,$-v);ie.y=se,ie.h=$-se},"applyN"),U=r(()=>{ie.h=x(H.y-l.y,v,me-l.y)},"appl\
yS");switch(o){case"move":{const se=H.x-a.x,B=H.y-a.y;ie.x=x(l.x+se,0,re-l.w),ie.y=x(l.y+B,0,me-l.h);
break}case"w":N();break;case"e":V();break;case"n":j();break;case"s":U();break;case"nw":j(),N();break;case"\
ne":j(),V();break;case"sw":U(),N();break;case"se":U(),V();break;default:break}ie.x=x(ie.x,0,re-ie.w),
ie.y=x(ie.y,0,me-ie.h),markerState.cropRect=ie,renderCropOverlay(),P.preventDefault()},"move"),Y=r(P=>{
n.delete(P.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,o=null,l=null),e.
releasePointerCapture&&e.releasePointerCapture(P.pointerId),P.preventDefault()},"end");e.addEventListener(
"pointerdown",G),e.addEventListener("pointermove",O),e.addEventListener("pointerup",Y),e.addEventListener(
"pointercancel",Y),e.addEventListener("pointerleave",Y)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const o=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const C=document.createElement("canvas");C.width=o,C.height=l;const L=C.getContext(
"2d");if(!L)return;L.fillStyle="#000000",L.fillRect(0,0,o,l),L.drawImage(n,0,0,o,l);const M=await new Promise(
O=>C.toBlob(O,"image/png"));if(!M){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const E=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([M],`${E}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=o,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,o,l),c.drawImage(n,0,0,o,l),markerState.cropRect){const C=o/n.width,L=l/n.height,M=Math.max(0,Math.
floor(markerState.cropRect.x*C)),E=Math.max(0,Math.floor(markerState.cropRect.y*L)),G=Math.min(o,Math.
max(1,Math.floor(markerState.cropRect.w*C))),O=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*L))),Y=document.createElement("canvas");Y.width=G,Y.height=O;const P=Y.getContext("2d");P&&(P.drawImage(
a,M,E,G,O,0,0,G,O),a=Y)}const d=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const g=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([d],`${g}_marked.png`,
{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),x=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=x,setAttachmentSourceForPath(
w,x)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}r(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,o=null,l=null,c=null,d=[],m=null;const g=r(()=>{m&&clearTimeout(
m);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(o)try{o.close()}catch{}},"cleanup"),y=r(
()=>{g(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();
if(!v||!v.length)return y();o=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=o.createMediaStreamSource(new MediaStream(v)),l=o.createScriptProcessor(4096,1,1),l.onaudioprocess=
x=>{const k=x.inputBuffer.getChannelData(0);d.push(new Float32Array(k))},c.connect(l),l.connect(o.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(m=setTimeout(()=>{const x=(e.
name||"video").replace(/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{
type:"audio/wav"});g(),n(C)},w+250)),await i.play(),i.onended=()=>{const x=(e.name||"video").replace(
/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{type:"audio/wav"});g(),
n(C)}}catch{y()}},i.onerror=()=>y()}))}r(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
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
perFilePct[w]=0;let k=v;if(o&&isVideoFile(v)){const L=await extractAudioFromVideo(v,x);L?(k=L,updateUploadRowFile(
x,L),x&&x.status&&(x.status.textContent="\u97F3\u58F0\u306E\u307F")):(x&&x.status&&(x.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{k=await applyImageCompression(v,
readComposerCompression()),k!==v&&updateUploadRowFile(x,k)}catch{}if(isUploadCancelled(x))return!1;if(browserFastModeEnabled){
const L=Array.from(browserFastLocalFiles.values()).reduce((M,E)=>M+Number(E.file&&E.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||L+k.size>BROWSER_FAST_MAX_BYTES?(x&&x.status&&
(x.status.textContent="\u4E0A\u9650\u8D85\u904E"),x&&x.row&&x.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(x.uploadId,{file:k,rowObj:x}),
x.status&&(x.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
x.bar&&(x.bar.style.width="100%"),x.row&&(x.row.dataset.browserFastLocal="1"),!0)}const C=await uploadFileWithProgress(
k,x,t);if(!C&&t.cameraCapture&&!isUploadCancelled(x)){const L=document.createElement("button");L.className=
"text-xs border border-gray-700 rounded px-2 py-1",L.textContent="\u518D\u8A66\u884C",L.onclick=()=>{
if(isUploadCancelled(x))return;L.disabled=!0;const M=x.row.getAttribute("data-local-url");M&&URL.revokeObjectURL(
M),x.row.remove(),handleFiles([k],{...t,openModal:!0,skipCompression:!0})},x.row.appendChild(L)}return C}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),d=new Array(i.length).fill(!1);let m=0;const g=Math.
min(UPLOAD_CONCURRENCY,i.length),y=Array.from({length:g}).map(async()=>{for(;;){const v=m++;if(v>=i.
length)break;d[v]=await c(i[v])}});return await Promise.all(y),d}r(handleFiles,"handleFiles"),get("c\
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
