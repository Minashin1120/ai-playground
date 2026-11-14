document.addEventListener("DOMContentLoaded",()=>{var _i,Si;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const u=get("enable-batch-mode");u&&u.checked&&(u.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-gemini-free-tier");t&&t.addEventListener("change",
()=>{browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,typeof schedulePromptTokenEstimate==
"function"&&schedulePromptTokenEstimate(!0)});const n=get("enable-batch-mode");n&&(n.onchange=()=>{if(n.
checked){browserFastModeEnabled&&setBrowserFastModeEnabled(!1);const u=get("enable-coding-mode");u&&
u.checked&&(u.checked=!1,typeof syncCodingModeUi=="function"&&syncCodingModeUi(!1),showToast("Batch \
API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
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
getBoundingClientRect(),h=u.getBoundingClientRect(),b=h.left+h.width/2-(f.left+f.width/2),_=h.top+h.
height/2-(f.top+f.height/2);l.style.transform=`translate(${b}px, ${_}px) scale(0.1)`,l.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),l.remove()},800)}else l.style.opacity="0",setTimeout(()=>l.remove(),1e3)}},3e3);function c(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const h=(get("model-select").value||"").toLowerCase().includes("gpt-image-2.6"),b=get("gpt-im\
age-format");get("gpt-image-complexity-wrap")&&get("gpt-image-complexity-wrap").classList.toggle("hi\
dden",!h),b&&(Array.from(b.options).forEach(M=>{M.hasAttribute("data-gpt-image-26")&&(M.hidden=!h)}),
!h&&b.value==="gif"&&(b.value="jpeg"));const _=h&&b&&b.value==="gif",S=get("gpt-image-compression-wr\
ap");b&&S&&(b.value==="png"||b.value==="gif"?S.classList.add("hidden"):S.classList.remove("hidden")),
get("gpt-image-gif-wrap")&&get("gpt-image-gif-wrap").classList.toggle("hidden",!_)}r(c,"updateGptIma\
geUi");function d(){const u=get("gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.
remove("hidden"):u.classList.add("hidden");const h=(get("model-select").value||"").toLowerCase().includes(
"gemini-3.1-flash-lite-image");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{
b&&(Array.from(b.options).forEach(_=>{_.value!=="1K"&&(_.disabled=h)}),h&&b.value!=="1K"&&(b.value="\
1K"))})}r(d,"updateGeminiImageUi");function m(){const u=get("grok-image-options");if(!u)return;const f=(get(
"model-select").value||"").toLowerCase(),h=isGrokImageModel(),b=f==="grok-imagine-image-quality"||f===
"grok-imagine-image-2.0",_=f==="grok-imagine-image-2.0";if(h){u.classList.remove("hidden");const M=get(
"grok-image-resolution")?get("grok-image-resolution").parentElement:null;M&&M.classList.toggle("hidd\
en",!b);const A=get("grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.
toggle("hidden",!_)}else u.classList.add("hidden");if(get("modal-grok-image-options")){const M=get("\
modal-grok-image-resolution")?get("modal-grok-image-resolution").parentElement:null;M&&M.classList.toggle(
"hidden",!b);const A=get("modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;
A&&A.classList.toggle("hidden",!_)}}r(m,"updateGrokImageUi");function g(){var B,D;const u=get("ideog\
ram-image-options"),f=get("modal-ideogram-image-options"),h=get("model-select")&&get("model-select").
value||"",b=isIdeogramModel(h),_=ideogramModelTraits(h),S={resolution:_.resolution,quality:_.quality,
speed:_.speed,style:_.style,negative:_.negative};if(u&&u.classList.toggle("hidden",!b),f&&f.classList.
toggle("hidden",!b),updateIdeogramPromptInputUi(),!b)return;const M=((B=get("ideogram-image-operatio\
n"))==null?void 0:B.value)==="remove_background",A=((D=get("ideogram-image-operation"))==null?void 0:
D.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(Q=>{const oe=Q==="operation"||A&&Q==="seed"||
!(M||A)&&S[Q]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(de=>{const W=get(de+Q);!W||!W.
parentElement||W.parentElement.classList.toggle("hidden",!oe)})}),["ideogram-image-style","modal-ide\
ogram-image-style"].forEach(Q=>{const oe=get(Q);if(!oe)return;const de=_.styleOptions.join("|");if(oe.
dataset.options!==de){const W=oe.value;oe.innerHTML=_.styleOptions.map(ve=>`<option value="${ve}">${ve===
"auto"?"Auto":ve}</option>`).join(""),oe.dataset.options=de,oe.value=_.styleOptions.includes(W)?W:"a\
uto"}});const I=get("ideogram-image-note");I&&(I.textContent=M?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWE\
BP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
A?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
_.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
r(g,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(u=>{
var f;(f=get(u))==null||f.addEventListener("change",()=>{const h=get(u.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");h&&(h.value=get(u).value),g(),L()})});function y(){
var b;const u=get("grok-video-options");if(!u)return;const f=String(((b=get("model-select"))==null?void 0:
b.value)||"").toLowerCase();isGrokVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");
const h=get("grok-video-resolution");if(h){const _=Array.from(h.options).find(M=>M.value==="1080p"),
S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(f);_&&(_.disabled=!S),!S&&h.value===
"1080p"&&(h.value="720p")}}r(y,"updateGrokVideoUi");function v(){var _;const u=get("gemini-video-opt\
ions");if(!u)return;const f=String(((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase();
isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");const h=get("gemini-vide\
o-resolution");if(h){const S=Array.from(h.options).find(A=>A.value==="4K"),M=f==="veo-3.1-lite-gener\
ate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-flash";S&&(S.disabled=M),M&&h.value===
"4K"&&(h.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle("hidden",f===
"gemini-omni-1.1-flash")}r(v,"updateGeminiVideoUi");function w(){const u=get("gemini-music-options");
if(!u)return;const f=isGeminiRealtimeMusicModel(),h=isGeminiMusicModel()&&!f;u.classList.toggle("hid\
den",!h);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!f)}r(w,"updateGemi\
niMusicUi");function k(){var M;const u=get("xai-chat-options");if(!u)return;const f=String(((M=get("\
model-select"))==null?void 0:M.value)||"").toLowerCase(),h=f.startsWith("grok-")&&!isGrokImageModel(
f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!h);const b=get("xai-logp\
robs"),_=get("xai-top-logprobs"),S=f.includes("grok-4.20");b&&(b.disabled=S,S&&(b.checked=!1)),_&&(_.
disabled=S,S&&(_.value=""))}r(k,"updateXaiChatUi");function x(){const u=isMistralOcrModel(),f=get("m\
istral-ocr-options");f&&f.classList.toggle("hidden",!u);const h=get("modal-mistral-ocr-options");h&&
h.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const _=get(b);_&&(_.classList.toggle("opacity-50",u),_.classList.toggle("po\
inter-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(x,"updateMistralOcrUi");function C(){const u=get("image-input-limit\
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
r(C,"updateImageInputLimits");function L(){const u=get("enable-sys-prompt"),f=!!(u&&u.disabled),h=!!(u&&
u.checked);E(),u&&(u.disabled?!f&&h?u.dataset.restoreChecked="1":f||delete u.dataset.restoreChecked:
(f&&u.dataset.restoreChecked==="1"&&(u.checked=!0),delete u.dataset.restoreChecked))}r(L,"toggleOpti\
ons"),window.toggleOptions=L;function T(u){const f=get("thinking-options"),h=get("thinking-level"),b=get(
"reasoning-effort"),_=get("enable-thinking"),S=get("thinking-budget-label"),M=get("thinking-budget"),
A=u==="effort";if(f&&(f.dataset.thinkingMode=A?"effort":"thinking"),h&&h.classList.toggle("hidden",A),
b&&b.classList.toggle("hidden",!A),S&&S.classList.toggle("hidden",A),M&&M.classList.toggle("hidden",
A),!!_)if(A){_.dataset.thinkingChecked===void 0&&(_.dataset.thinkingChecked=_.checked?"1":"0");const I=b?
Array.from(b.options).find(D=>D.value==="none"):null,B=!!(I&&!I.disabled&&!I.classList.contains("hid\
den"));_.checked=B?!!(b&&String(b.value||"").toLowerCase()!=="none"):!0,_.disabled=!B}else _.dataset.
thinkingChecked!==void 0&&(_.checked=_.dataset.thinkingChecked==="1",delete _.dataset.thinkingChecked)}
r(T,"setComposerThinkingMode");function E(){const u=get("model-select");if(!u)return;const f=u.value,
h=String(f||"").toLowerCase(),b=h.includes("deepseek"),_=h.startsWith("glm-"),S=get("thinking-option\
s"),M=get("reasoning-effort-container"),A=get("enable-thinking"),I=get("thinking-level"),B=get("thin\
king-budget"),D=get("enable-search"),Q=get("search-container"),oe=get("url-context-container"),de=get(
"enable-maps"),W=get("maps-grounding-container"),ve=get("enable-sys-prompt"),G=get("sys-prompt-optio\
n"),he=get("enable-python"),X=get("python-container"),we=get("prompt-cache-container"),Se=get("enabl\
e-prompt-cache"),ae=f==="gpt-5-search-api",xe=f.includes("tts"),Oe=isMistralOcrModel(f),Ne=h.includes(
"gemini-3.1-flash-lite-image"),pt=h==="gemini-nano-banana-2.1",Qe=h.includes("gemini-3.1-flash-image")&&
!Ne||pt,bt=isClaudeModelKey(f),Dt=h==="gemini-3.8-flash-cyber",Te=isLlmModel()&&!b&&!xe&&!h.includes(
"realtime")&&!h.includes("native-audio")&&!h.includes("live");we&&(Te?(we.classList.remove("hidden",
"opacity-50","pointer-events-none"),Se&&(Se.disabled=!1)):(Se&&(Se.checked=!1,Se.disabled=!0),we.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),M&&M.classList.
add("hidden"),T("thinking");const Ce=get("vision-model-info");if(Ce&&Ce.classList.add("hidden"),M){const Le=get(
"reasoning-effort");if(Le){Array.from(Le.options).forEach(Fe=>{const ze=h==="gpt-5.6"||h.startsWith(
"gpt-5.6-"),Yt=h.startsWith("gpt-6"),_n=h==="gpt-6-sol"||h==="gpt-6-luna"||h==="gpt-6.1-terra",Rn=h===
"deepseek-v4.1-flash"||h==="deepseek-v4-flash-0731"||h==="deepseek-v4-flash"||h==="deepseek-v4-flash\
-vision-exp",Sn=h==="deepseek-v4-pro",q=h.includes("grok-4.5"),te=h.includes("grok-4.6");Fe.value===
"max"?Fe.classList.toggle("hidden",!ze&&!Yt&&!Rn&&!Sn):Fe.value==="xhigh"?Fe.classList.toggle("hidde\
n",!te&&!h.includes("multi-agent")&&!ze&&!Yt):Fe.value==="medium"?Fe.classList.toggle("hidden",!(h.includes(
"grok-4.3")||q||te||h.includes("grok-4.20-0309-reasoning")||h.includes("grok-build")||h.includes("mu\
lti-agent")||h.includes("gpt-5")||Yt||h.includes("o1")||h.includes("o3"))):Fe.value==="none"?Fe.classList.
toggle("hidden",!h.includes("grok-4.3")&&!h.includes("grok-build")&&!h.includes("gpt-5")&&!_n&&!Rn&&
!Sn):Fe.value==="low"&&Fe.classList.toggle("hidden",Sn)});const Ge=Le.selectedOptions&&Le.selectedOptions[0];
Ge&&Ge.classList.contains("hidden")&&(Le.value=b?"high":"medium")}}oe&&oe.classList.add("hidden"),W&&
W.classList.add("hidden"),A&&(A.disabled=!1),B&&(B.disabled=!0,B.classList.add("opacity-50"));const Ue=isGeminiImageModelKey(
f);if(xe||Oe)Q&&(get("enable-search").checked=!1,Q.classList.add("opacity-50","pointer-events-none")),
oe&&(get("enable-url-context").checked=!1,oe.classList.add("opacity-50","pointer-events-none")),W&&de&&
(de.checked=!1,W.classList.add("opacity-50","pointer-events-none")),X&&(he.checked=!1,X.classList.add(
"opacity-50","pointer-events-none")),ve&&G&&(ve.checked=!1,ve.disabled=!0,G.classList.add("opacity-5\
0"));else if(Qe||Ne)W&&de&&(de.checked=!1,W.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),pt?(Array.from(I.options).forEach(Le=>{Le.disabled=Le.value==="low"}),[
"minimal","medium","high"].includes(I.value)||(I.value="medium")):(Array.from(I.options).forEach(Le=>{
["low","medium"].includes(Le.value)&&(Le.disabled=!0),["minimal","high"].includes(Le.value)&&(Le.disabled=
!1)}),["minimal","high"].includes(I.value)||(I.value=Ne?"minimal":"high")),A&&(A.disabled=!1),Ne&&(D&&
(D.checked=!1,D.disabled=!0),Q&&Q.classList.add("opacity-50","pointer-events-none"));else if(Ue)W&&de&&
(de.checked=!1,W.classList.add("hidden","opacity-50","pointer-events-none"));else if(bt)S.classList.
remove("hidden"),B&&(B.disabled=!1,B.classList.remove("opacity-50")),Array.from(I.options).forEach(Le=>{
Le.disabled=!0}),X&&(he.checked=!1,X.classList.add("opacity-50","pointer-events-none"));else if(Dt){
S&&S.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from(I.options).forEach(Ge=>{Ge.
disabled=!["low","medium","high"].includes(Ge.value)}),["low","medium","high"].includes(I.value)||(I.
value="medium"),[Q,oe,W,X].forEach(Ge=>{Ge&&Ge.classList.add("opacity-50","pointer-events-none")}),[
D,de,he].forEach(Ge=>{Ge&&(Ge.checked=!1,Ge.disabled=!0)});const Le=get("enable-url-context");Le&&(Le.
checked=!1,Le.disabled=!0),ve&&G&&(ve.disabled=!1,G.classList.remove("opacity-50"))}else if(f.includes(
"gemini")&&!Ue){S.classList.remove("hidden"),oe&&oe.classList.remove("hidden","opacity-50","pointer-\
events-none");const Le=f.includes("gemini-3")||f.includes("gemini-4");W&&(Le?W.classList.remove("hid\
den","opacity-50","pointer-events-none"):(de&&(de.checked=!1),W.classList.add("hidden","opacity-50",
"pointer-events-none")));const Ge=f.includes("flash"),Fe=f==="gemini-4-hydrogen";Array.from(I.options).
forEach(ze=>{Fe?ze.disabled=!["low","medium","high","xhigh"].includes(ze.value):f==="gemini-4-carbon"||
f==="gemini-4-argon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?ze.disabled=!["low","medium","h\
igh"].includes(ze.value):f==="gemini-3.6-flash"?ze.disabled=!["medium","high"].includes(ze.value):f===
"gemini-3.5-flash-lite"?ze.disabled=!["minimal","medium","high"].includes(ze.value):ze.value==="xhig\
h"?ze.disabled=!0:["minimal","medium"].includes(ze.value)?ze.disabled=!Ge:ze.disabled=!1}),Fe&&!["lo\
w","medium","high","xhigh"].includes(I.value)?I.value="high":(f==="gemini-4-carbon"||f==="gemini-4-a\
rgon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium","high"].includes(I.value)||
f==="gemini-3.6-flash"&&!["medium","high"].includes(I.value)?I.value="medium":f==="gemini-3.5-flash-\
lite"&&!["minimal","medium","high"].includes(I.value)?I.value="minimal":!Ge&&["minimal","medium"].includes(
I.value)&&(I.value="high"),Le?A&&(A.checked=!0,A.disabled=!0):A&&(A.disabled=!1),B&&f.includes("gemi\
ni-2.5")&&(B.disabled=!1,B.classList.remove("opacity-50")),B&&!f.includes("gemini-2.5")&&(B.disabled=
!0,B.classList.add("opacity-50"))}if(isLlmModel()&&(h.includes("gpt-5")||h.includes("o1")||h.includes(
"o3")||h.includes("grok-4.3")||h.includes("grok-4.5")||h.includes("grok-4.6")||h.includes("grok-4.20\
-0309-reasoning")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt")&&!h.includes(
"tts")))S&&S.classList.remove("hidden"),T("effort"),Q&&Q.classList.remove("opacity-50","pointer-even\
ts-none");else if(b){S&&S.classList.remove("hidden"),T("effort");const Le=get("vision-model-info");if(Le&&
Le.classList.toggle("hidden",h==="deepseek-v4.1-flash"||h==="deepseek-v4-flash-vision-exp"),D&&(D.disabled=
!1),Q&&Q.classList.remove("opacity-50","pointer-events-none"),oe){const Ge=get("enable-url-context");
Ge&&(Ge.checked=!1),oe.classList.add("opacity-50","pointer-events-none")}W&&de&&(de.checked=!1,W.classList.
add("opacity-50","pointer-events-none"))}else Oe||(Q&&Q.classList.remove("opacity-50","pointer-event\
s-none"),W&&de&&(de.checked=!1,W.classList.add("hidden","opacity-50","pointer-events-none")));if(xe?
X&&X.classList.add("opacity-50","pointer-events-none"):(X&&X.classList.remove("opacity-50","pointer-\
events-none"),(!Ue||Qe)&&!f.includes("gpt-image")&&(ve.disabled=!1,G.classList.remove("opacity-50"))),
(Ue&&!Qe||f.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(f)||isGrokVideoModel()||Oe)&&
ve&&G&&(ve.checked=!1,ve.disabled=!0,G.classList.add("opacity-50")),X&&(isLlmModel()?(X.classList.remove(
"hidden"),he.disabled=!1):(he.checked=!1,he.disabled=!0,X.classList.add("hidden"))),ae?(D&&(D.checked=
!0,D.disabled=!0),Q&&Q.classList.add("opacity-50","pointer-events-none"),X&&(he.checked=!1,he.disabled=
!0,X.classList.add("opacity-50","pointer-events-none"))):D&&!f.includes("tts")&&!Oe&&!Ne&&(D.disabled=
!1),Dt){[D,de,he].forEach(Ge=>{Ge&&(Ge.checked=!1,Ge.disabled=!0)});const Le=get("enable-url-context");
Le&&(Le.checked=!1,Le.disabled=!0),[Q,oe,W,X].forEach(Ge=>{Ge&&Ge.classList.add("opacity-50","pointe\
r-events-none")})}_&&(Ce&&Ce.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v",
"glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(h)),[D,he].forEach(Le=>{Le&&(Le.checked=!1,Le.
disabled=!0)}),[Q,X].forEach(Le=>{Le&&Le.classList.add("opacity-50","pointer-events-none")}));const mt=get(
"mask-btn");mt&&(isGptImageModel()||isIdeogramObjectRemoval()?mt.classList.remove("hidden"):(mt.classList.
add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),
c(),d(),m(),g(),y(),v(),w(),updateBatchUi(f);const Ht=get("gemini-free-tier-container");if(Ht){const Le=typeof isGeminiApiModel==
"function"&&isGeminiApiModel(f)&&(typeof currentGeminiBackend!="function"||currentGeminiBackend()!==
"vertex_ai");Ht.classList.toggle("hidden",!Le)}k(),x(),C(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}r(E,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener(
"change",L),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
L(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const H=get("enable-canvas-mode");H&&(H.checked=canvasModeEnabled,H.addEventListener("change",()=>syncCanvasModeUi(
H.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const N=get("enable-codin\
g-mode");N&&(N.checked=codingModeEnabled,N.addEventListener("change",()=>syncCodingModeUi(N.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const Y=get(
"prompt-controls-toggle-btn");Y&&(Y.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>c()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>d()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,h){const b=(f||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(u||"");const _=u||"",S=(b||"").toLowerCase();
if(S==="mathsvg"){const G=getRenderableSvgCode("mathsvg",_);return G&&typeof window.buildMathSvgRenderHtml==
"function"?window.buildMathSvgRenderHtml(G):""}let M="";try{const G=hljs.getLanguage(b)?b:"plaintext";
activeStreamingBubbleId&&_.length>2e4?M=escapeHtml(_):M=hljs.highlight(_,{language:G}).value}catch{M=
escapeHtml(_)}const A=encodeURIComponent(_).replace(/'/g,"%27"),I=detectBlockedScriptsInCode(_),B=hashString(
`${b||"TEXT"}
${_||""}`);let D="";if(canvasModeEnabled){const G=String(canvasPreviewState.selectedKey||"")===B,he=G?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";D=`<button\
 class="canvas-preview-btn${G?" canvas-active":""}" data-code="${A}" data-code-key="${B}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${he}" aria-label="${he}" aria-pressed="${G?"true":"false"}"\
><i class="fas ${G?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,_)){const G=I?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
D=`<button class="html-preview-btn" data-code="${A}" ${I?'data-suspicious="1"':""} title="${G}" aria\
-label="${G}"><i class="fas ${I?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const Q=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,oe=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${B}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,de=(b||"TEXT")+(I?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),W=getRenderableSvgCode(S,_);return`${W?buildSvgCodeRenderHtml(W,B):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${B}"><div class="code-header"\
><span class="code-lang">${de}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${oe}${D}${Q}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${M}</code></pre></d\
iv></div>`},link(u,f,h){return`<a href="${u}" title="${f||""}" target="_blank">${h}</a>`},image(u,f,h){
return buildChatImageHtml(u,{alt:h,title:f})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(_i=get("version-update-dismis\
s"))==null||_i.addEventListener("click",()=>{const u=localStorage.getItem("app_version")||"";u&&localStorage.
setItem("version_notified",u),hideModal("version-update-modal")});const O=get("version-update-clear-\
cache");if(O&&(O.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),O.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(O.checked)})),
(Si=get("version-update-reload"))==null||Si.addEventListener("click",async()=>{var f;await versionUpdateCachePreferenceSavePromise.
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
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const V=r(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),ue=r(u=>{const f=Array.isArray(u)?u:[],h=get("passkey-list"),b=get("passkey-count");if(b&&
(b.innerText=String(f.length)),!!h){if(!f.length){h.innerHTML='<div class="text-[11px] text-gray-500\
">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}h.innerHTML="",f.forEach((_,S)=>{const M=_&&_.id?String(_.id):"",A=document.createElement("di\
v");A.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const I=document.createElement("div");I.className="min-w-0";const B=document.createElement("\
div");B.className="text-xs text-gray-200 truncate",B.innerText=_&&_.name?String(_.name):`Security Ke\
y ${S+1}`;const D=document.createElement("div");D.className="text-[10px] text-gray-500 mt-1",D.innerText=
_&&_.created_at?`\u767B\u9332\u65E5\u6642: ${V(_.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
I.appendChild(B),I.appendChild(D),A.appendChild(I);const Q=document.createElement("button");Q.type="\
button",Q.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",Q.innerText="\u524A\u9664",Q.disabled=!M,M&&(Q.onclick=()=>window.removeWebAuthnCredential(
M)),A.appendChild(Q),h.appendChild(A)})}},"renderPasskeyList"),ge=r(u=>{const f=get("session-list");
if(f){if(!u||!u.length){f.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(h=>{const b=h.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",_=h.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!h.is_current&&!h.is_revoked?`<button \
data-session-id="${escapeHtml(h.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
M=(h.user_agent||"Unknown").slice(0,120),A=h.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${_}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(M)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(V(h.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(V(h.created_at))}</div></div>${S}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(h=>{h.onclick=async()=>{const b=h.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const _=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let S={};try{S=await _.json()}catch{}if(_.ok){if(S.logged_out){
location.href="/login";return}await ne()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),ne=r(async()=>{const u=get("session-list");u&&(u.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api\
/sessions");let h={};try{h=await f.json()}catch{}if(!f.ok){if(h&&h.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(h.sessions||[]).filter(_=>!_.is_revoked);ge(b)},"loadSessions"),ke=r(()=>{const u=get(
"session-refresh-btn");u&&(u.onclick=()=>ne());const f=get("session-revoke-others-btn");f&&(f.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await ne():showToast("\u64CD\u4F5C\u306B\u5931\
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
(f.checked=!0),L())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadSkills(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const $=get("vision-model-change-btn");$&&($.onclick=()=>_openVisionModelSelector());const R=get("co\
mpression-format-only");R&&(R.onchange=()=>{const u=R.checked,f=get("compression-max-size"),h=get("c\
ompression-max-dim");f&&(f.disabled=u),h&&(h.disabled=u);const b=get("compression-size-wrap"),_=get(
"compression-dim-wrap");b&&(b.style.opacity=u?"0.4":"1"),_&&(_.style.opacity=u?"0.4":"1")});const U=r(
()=>{const u=get("enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=
!!temporaryChatEnabled,u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(
u.checked)||(setTemporaryChatUiState(f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");U(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const j=get("storage-us\
age-refresh");j&&(j.onclick=()=>loadStorageUsage());let F=null;const pe=r(()=>{const u=new Uint8Array(
16);return window.crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),P=r((u={})=>{const f=get("account-transfer-progress"),h=get("account-tran\
sfer-progress-bar"),b=get("account-transfer-progress-percent"),_=get("account-transfer-progress-text"),
S=get("account-transfer-progress-detail"),M=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&f.
classList.remove("hidden"),h&&(h.style.width=`${M}%`),b&&(b.textContent=`${Math.round(M)}%`),_&&(_.textContent=
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
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),J=r(u=>{me&&(me.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const h=get("account-transfer-cancel-btn");
h&&(h.disabled=!u)},"setAccountTransferControls"),fe=r((u={})=>{const f=get("account-export-ready"),
h=get("account-export-ready-text"),b=get("account-export-expiry"),_=get("account-export-download-btn"),
S=!!(u.available&&u.download_url);if(f&&f.classList.toggle("hidden",!S),!S){_&&_.removeAttribute("hr\
ef");return}const M=Math.max(0,Number(u.size_bytes)||0),A=M>=1024*1024*1024?`${(M/(1024*1024*1024)).
toFixed(2)} GB`:`${(M/(1024*1024)).toFixed(1)} MB`;if(h){const I=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
h.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${I}`}
if(b){const I=u.expires_at?new Date(u.expires_at):null;b.textContent=I&&!Number.isNaN(I.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${I.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
_&&(_.href=u.download_url)},"renderAccountExportAvailability"),re=r(async u=>{for(;F===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),h=await f.json().catch(()=>({}));if(f.ok&&(h.state!=="pending"&&P(h),["ready","completed","f\
ailed","cancelled","expired"].includes(h.state)))return h}catch{}await new Promise(f=>setTimeout(f,700))}
return null},"pollAccountTransfer"),Z=r((u,f,h=!0)=>{f&&(P(f),fe(f),h&&f.state==="ready"?showToast(f.
message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):h&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),ee(u))},"handleFinishedAccountExport"),le=r(async()=>{try{const u=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(fe(f),f.state==="ready"){P(f);return}if(["failed","cancelled","expired"].includes(
f.state)){P(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||F&&F.id===
f.job_id||F)return;const h={id:f.job_id,type:"export",stopped:!1,restored:!0};F=h,J(!0),P(f);const b=await re(
h);b&&Z(h,b,!0)}catch{}},"refreshLatestAccountExport"),ee=r(u=>{F===u&&(F=null),u.stopped=!0,J(!1)},
"finishAccountTransfer"),ce=get("account-transfer-cancel-btn");ce&&(ce.onclick=async()=>{const u=F;if(!(!u||
u.stopped)){u.cancelRequested=!0,ce.disabled=!0,P({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${u.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
u.controller&&u.controller.abort(),P({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
u.type==="export"&&fe({available:!1}),ee(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const me=get("account-export-btn");me&&(me.onclick=async()=>{if(F)return;const u={id:pe(),
type:"export",stopped:!1};F=u,J(!0),fe({available:!1}),P({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/account/export",
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{job_id:u.id}),keepalive:!0})),h=await f.json().catch(()=>({}));if(f.status===409&&h.error==="export\
_in_progress"&&h.job_id)u.id=h.job_id;else if(!f.ok)throw new Error(h.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\
\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":h.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
P({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await re(u);!u.cancelRequested&&b&&Z(u,b,!0)}catch(f){const h=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";P({progress:0,phase:"failed",message:h}),
showToast(h,"error",!0),ee(u)}});const De=get("account-export-download-btn");De&&De.addEventListener(
"click",async u=>{const f=De.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();try{const h=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await h.json().catch(
()=>({}));h.ok&&b.available&&b.download_url?(De.href=b.download_url,window.location.assign(b.download_url)):
(fe(b),P(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),le())}catch{window.location.assign(f)}}}),J(!1),le();const Re=get("import-files-grid"),
Ke=get("import-files-info"),Ve=get("import-files-summary"),be=r(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let se=null;const He=r(()=>{
if(!se)return;const u=se.files,f=se.selection;let h=0;u.forEach(S=>{f.has(S.archive_path)&&(h+=Number(
S.size_bytes)||0)});const b=Number(se.available_bytes)||0,_=h>b;Ve&&(Ve.textContent=`\u9078\u629E\u4E2D: ${be(
h)} / \u5229\u7528\u53EF\u80FD: ${be(b)}${_?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,Ve.classList.
toggle("text-red-300",_)),Ke&&(Ke.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),ie=r(
()=>{if(!Re||!se)return;Re.innerHTML="";const u=se.files;if(!u.length){Re.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
He();return}u.forEach(f=>{const h=document.createElement("label"),b=se.selection.has(f.archive_path);
h.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,h.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${be(f.size_bytes)}</div></div>`;const _=h.
querySelector(".import-file-check");_.addEventListener("change",()=>{_.checked?se.selection.add(f.archive_path):
se.selection.delete(f.archive_path),h.classList.toggle("border-blue-500",_.checked),h.classList.toggle(
"border-gray-600",!_.checked),He()}),Re.appendChild(h)}),He()},"renderImportFileItems"),ot=r(u=>new Promise(
f=>{if(se={files:u.files||[],selection:new Set((u.files||[]).map(h=>h.archive_path)),available_bytes:u.
available_bytes,resolve:f},ie(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),Ie=r(u=>{if(hideModal("import-files-modal"),se){const f=se.resolve;
se=null,f(u)}},"closeImportFileSelection"),ht=get("import-files-close");ht&&(ht.onclick=()=>Ie(null));
const Xe=get("import-files-cancel");Xe&&(Xe.onclick=()=>Ie(null));const rt=get("import-files-confirm");
rt&&(rt.onclick=()=>{if(!se)return;const u=Array.from(se.selection);Ie(u.length?u.join(","):"__none_\
_")});const ct=get("import-files-select-all");ct&&(ct.onclick=()=>{se&&(se.files.forEach(u=>se.selection.
add(u.archive_path)),ie())});const vt=get("import-files-none");vt&&(vt.onclick=()=>{se&&(se.selection.
clear(),ie())});const Cn={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
\u65E2\u5B9A: Gemini\u7121\u6599\u67A0",default_enable_url_context:"\u65E2\u5B9A: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",
default_enable_maps:"\u65E2\u5B9A: Maps",default_enable_python:"\u65E2\u5B9A: Python",default_enable_file_creation:"\
\u65E2\u5B9A: File",default_enable_thinking:"\u65E2\u5B9A: Thinking",default_thinking_level:"\u65E2\u5B9A: Thi\
nking\u30EC\u30D9\u30EB",default_thinking_budget:"\u65E2\u5B9A: Thinking budget",default_reasoning_effort:"\
\u65E2\u5B9A: Reasoning effort",default_enable_system_prompt:"\u65E2\u5B9A: \u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
default_enable_mcp:"\u65E2\u5B9A: MCP",default_safety_setting:"\u65E2\u5B9A: \u5B89\u5168\u8A2D\u5B9A",
default_vision_model:"Vision Model",rich_paste_prompt_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30D7\u30ED\u30F3\u30D7\u30C8",
rich_paste_prompt_use_custom_default:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30AB\u30B9\u30BF\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u65E2\u5B9A",
last_model:"\u76F4\u524D\u306E\u30E2\u30C7\u30EB",last_enable_search:"\u76F4\u524D: Search",last_gemini_free_tier:"\
\u76F4\u524D: Gemini\u7121\u6599\u67A0",last_enable_url_context:"\u76F4\u524D: URL\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8",
last_enable_maps:"\u76F4\u524D: Maps",last_enable_python:"\u76F4\u524D: Python",last_enable_file_creation:"\
\u76F4\u524D: File",last_enable_thinking:"\u76F4\u524D: Thinking",last_thinking_level:"\u76F4\u524D: Thinking\u30EC\
\u30D9\u30EB",last_thinking_budget:"\u76F4\u524D: Thinking budget",last_reasoning_effort:"\u76F4\u524D: Reason\
ing effort",last_enable_system_prompt:"\u76F4\u524D: \u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
last_enable_mcp:"\u76F4\u524D: MCP",last_safety_setting:"\u76F4\u524D: \u5B89\u5168\u8A2D\u5B9A",last_math_steps:"\
\u76F4\u524D: \u6570\u5B66\u30B9\u30C6\u30C3\u30D7",enable_latency_metrics:"\u30EC\u30B9\u30DD\u30F3\u30B9\u901F\u5EA6\u306E\u8A08\u6E2C",
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},rn=r(u=>{
if(u===!0)return"ON";if(u===!1)return"OFF";if(u==null||u==="")return"\u672A\u8A2D\u5B9A";const f=String(
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let Tt=null;const Lt=r(
u=>{if(Tt){const f=Tt;Tt=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),qt=r(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}Tt=f;const b=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],_=get("settings-confirmation-list");_&&(b.length?
_.innerHTML=b.map(M=>{const A=Cn[M.field]||M.field,I=rn(M.current),B=rn(M.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
I)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(B)}</d\
iv>
                            </div>`}).join(""):_.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Gt=get("settings-confir\
mation-modal");Gt&&Gt.addEventListener("click",u=>{u.target===Gt&&Lt(!1)});const Ut=get("settings-co\
nfirmation-close");Ut&&(Ut.onclick=()=>Lt(!1));const wt=get("settings-confirmation-cancel");wt&&(wt.
onclick=()=>Lt(!1));const At=get("settings-confirmation-confirm");At&&(At.onclick=()=>Lt(!0));const ln=get(
"account-import-btn"),Et=get("account-import-inplace"),$t=get("account-import-inplace-warning");if(Et&&
$t){const u=r(()=>$t.classList.toggle("hidden",!Et.checked),"syncInplaceWarn");Et.addEventListener("\
change",u),u()}ln&&(ln.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,h=get("account-import-categories"),b=h?Array.from(h.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(de=>de.value):[],_=get("account-import-inplace"),S=!!(_&&_.checked),M=get("account-impo\
rt-settings-bypass"),A=!!(M&&M.checked);let I=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const B=h?Array.from(h.querySelectorAll('input[type="checkbox"]:checked')).map(de=>(de.
closest("label")&&de.closest("label").textContent||de.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${B.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const D={id:pe(),type:"import",stopped:!1,controller:new AbortController};
F=D,J(!0),P({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const Q=get("account-import-result");let oe=Promise.resolve(null);try{const W=Math.max(1,Math.ceil(f.
size/10485760)),ve=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:D.controller.
signal})),G=await ve.json().catch(()=>({}));if(!ve.ok)throw new Error(G.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
D.uploadId=G.upload_id;const he=G.chunk_size||10485760;let X=0,we=0;const Se=r(async()=>{for(;;){const Te=we++;
if(Te>=W)return;const Ce=f.slice(Te*he,Math.min(f.size,(Te+1)*he)),Ue=new FormData;Ue.append("chunk",
Ce,f.name),Ue.append("index",String(Te));const st=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
D.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Ue,signal:D.controller.signal})),
mt=await st.json().catch(()=>({}));if(!st.ok)throw new Error(mt.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
X++,P({progress:Math.min(35,Math.round(X/W*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${X}\
/${W}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ae=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ae=!0);try{await Promise.
all([Se(),Se(),Se()]);const Te=await apiFetch(`/api/account/import/upload/${encodeURIComponent(D.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:D.controller.signal})),Ce=await Te.json().
catch(()=>({}));if(!Te.ok)throw new Error(Ce.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
P({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ae&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let xe="",Oe=!1,Ne=0;const pt=r(
async()=>{let Te=!1;const Ce=r(()=>{Te||(Te=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Ue=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),st=await Ue.
json().catch(()=>null);if(!Ue.ok||!st){Ce();return}cacheUserSettings(st);const mt=get("settings-moda\
l");if(mt&&mt.classList.contains("modal-open"))try{it(st)}catch{}st.theme_color&&applyThemeColor(st.
theme_color,!0),Object.prototype.hasOwnProperty.call(st,"minimal_prompt_mode")&&st.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(st,"compact_prompt_mode")&&setCompactPromptMode(
!!st.compact_prompt_mode)}catch{}Ce()},"refreshSettingsFormAfterImport"),Qe=r(Te=>{const Ce=Te&&Te.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";Q&&(Q.textContent=`\u5B8C\u4E86: ${Ce}`,
Q.classList.remove("hidden","text-red-300"),Q.classList.add("text-emerald-300")),P({progress:100,phase:"\
completed",message:Ce}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&pt()},"finishImportSucce\
ss"),bt=r(async()=>{try{const Ce=await(await apiFetch(`/api/account/transfer/${D.id}`,manualSpinnerRequestOptions(
{cache:"no-store"}))).json().catch(()=>null);return Ce&&Ce.state?Ce:null}catch{return null}},"fetchI\
mportStatus"),Dt=r(async()=>{const Te=await bt();if(!Te)return{status:"unknown"};if(Te.state==="comp\
leted")return Qe(Te),{status:"done"};if(["failed","cancelled","expired"].includes(Te.state))throw new Error(
Te.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Te.state===
"needs_selection"&&Array.isArray(Te.files)){const Ce=await ot({files:Te.files,available_bytes:Te.available_bytes});
return Ce===null?(P({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
D.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(D.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(xe=Ce,{status:"reselect"})}if(Te.state===
"needs_settings_confirmation"&&Array.isArray(Te.settings_changes))return await qt({settings_changes:Te.
settings_changes})?(I=!0,{status:"reselect"}):(P({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),D.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
D.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Te.state==="running"){const Ce=await Promise.race([oe.catch(()=>null),new Promise(Ue=>setTimeout(
()=>Ue(null),6e4))]);if(Ce&&Ce.state==="completed")return Qe(Ce),{status:"done"};throw Ce&&["failed",
"cancelled","expired"].includes(Ce.state)?new Error(Ce.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Oe;){D.stopped=!0,await oe.catch(()=>null),
D.stopped=!1,oe=re(D);let Te;try{Te=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:D.uploadId,
categories:b.join(","),job_id:D.id,selected_files:xe,restore_inplace:S,confirm_settings:I||A}),signal:D.
controller.signal}))}catch(Fe){if(D.cancelRequested||Fe&&Fe.name==="AbortError")throw Fe;const ze=await Dt();
if(ze.status==="done"){Oe=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;
if(Ne<2){Ne++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Ce=null;try{Ce=await Te.json()}catch{Ce=null}if(Ce===null){const Fe=await Dt();if(Fe.status==="d\
one"){Oe=!0;break}if(Fe.status==="cancelled")return;if(Fe.status==="reselect")continue;if(Te.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Ne<2){Ne++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Te.ok&&Ce.error==="storage_limit_files"&&Ce.files){const Fe=await ot(Ce);if(Fe===null){P({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
D.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(D.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}xe=Fe;continue}if(Ce&&Ce.status==="settings_confirmation"&&
Array.isArray(Ce.settings_changes)){if(!await qt(Ce)){P({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),D.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(D.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}I=!0;continue}if(!Te.ok)throw new Error(Ce.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Ue=Ce.imported||{},st=[`\u8A2D\u5B9A ${Ue.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Ue.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Ue.chats||0}\u4EF6`,`Skill ${Ue.skills||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Ue.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Ue.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Ue.
diagnostics||0}\u4EF6`].join(" / "),mt=Ce.duplicates||{},Ht={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"\
Skill",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Le=[];for(const Fe of Object.keys(Ht)){const ze=Number(mt[Fe])||0;ze>
0&&Le.push(`${Ht[Fe]} ${ze}\u4EF6`)}const Ge=Le.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Le.
join("\u3001")}\uFF09`:"";Q&&(Q.textContent=`\u5B8C\u4E86: ${st}${Ge}`,Q.classList.remove("hidden","\
text-red-300"),Q.classList.add("text-emerald-300")),P({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&pt(),Oe=!0}}catch(de){if(D.
uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(D.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),D.cancelRequested||de&&de.name==="AbortError")return;const W=de&&
de.message?de.message:"",ve=W==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
W||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";P({progress:0,phase:"f\
ailed",message:ve}),Q&&(Q.textContent=ve,Q.classList.remove("hidden","text-emerald-300"),Q.classList.
add("text-red-300")),showToast(ve,"error",!0)}finally{D.stopped=!0,await oe.catch(()=>null),ee(D)}});
const It=get("account-dedupe-btn"),Pt=get("account-dedupe-result"),zt=r((u,f=!1)=>{Pt&&(Pt.textContent=
u,Pt.classList.remove("hidden"),Pt.classList.toggle("text-red-300",!!f),Pt.classList.toggle("text-em\
erald-300",!f))},"showDedupeResult");It&&(It.onclick=async()=>{const u=r(async()=>{const f=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),h=await f.json().catch(()=>null);if(!f.ok||!h)throw new Error(h&&h.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!h.has_duplicates){zt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],_={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",
feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};
for(const D of["chats","skills","files","feedback","diagnostics"]){const Q=Number(h.duplicates&&h.duplicates[D])||
0;Q>0&&b.push(`${_[D]} ${Q}\u4EF6`)}const S=Number(h.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${h.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${h.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const M=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await M.json().catch(()=>null);if(!M.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const I=[];for(const D of["chats","skills","files","feedback","diagnostics"]){const Q=Number(A.removed&&
A.removed[D])||0;Q>0&&I.push(`${_[D]} ${Q}\u4EF6`)}const B=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";zt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${I.join("\u3001")||"0\u4EF6"}${B}`),loadThreads(),loadSkills(),loadStorageUsage()},"r\
un");if(!It.disabled){It.disabled=!0,zt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await u()}catch(f){zt(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{It.disabled=!1}}});const cn=get("site-cache-usage-refresh");cn&&(cn.onclick=()=>loadSiteCacheUsage());
const Zt=get("clear-site-cache-btn");Zt&&(Zt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Zt)});
const kt=get("enc-scan-result"),K=r(async(u=null)=>{kt&&(kt.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const h=await apiFetch(
f,{cache:"no-store"}),b=await h.json();if(!h.ok){kt&&(kt.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const _=b.total||0,S=b.encrypted||0,M=b.unencrypted||0;let A=`Total: ${_} / Encrypted: ${S} /\
 Plain: ${M}`;if(b.samples&&b.samples.length){const I=b.samples.slice(0,8).map(B=>{const D=B.timestamp?
new Date(B.timestamp).toLocaleString():"";return`#${B.id} (${B.role||""}) ${D}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${I}</div>`}kt&&(kt.innerHTML=A)}catch{kt&&(kt.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),_e=get("enc-scan-all");_e&&(_e.onclick=()=>K(
null));const Be=get("enc-scan-thread");Be&&(Be.onclick=()=>currentThreadId?K(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ee=get("admin-enc-\
list");let dt=null,ft=!1;const xt=r(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThread\
EncryptedFromMessages"),et=r(()=>{dt=xt(allMessages)},"refreshCurrentThreadEncStateFromMessages"),_t=r(
async(u,f,{confirmPrompt:h=!0,reloadCurrent:b=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const _=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(h&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;ft=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
M=await S.json().catch(()=>({}));return S.ok?(showToast(`${_}\u3057\u307E\u3057\u305F\uFF08${M.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),dt=!!f,b&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ee&&await tt(),!0):
(showToast(M.error||`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{ft=!1}},"setAdminThreadEncry\
ption"),nt=r(u=>{if(!Ee)return;const f=u.threads||[];if(!f.length){Ee.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ee.innerHTML=f.map(h=>{const b=h.encrypted_count>0?"enc":"plain",_=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",M=h.
updated_at?new Date(h.updated_at).toLocaleString():"",A=escapeHtml(String(h.thread_id)),I=currentThreadId&&
String(currentThreadId)===String(h.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(h.title||
"")}">${escapeHtml(h.title||"(\u7121\u984C)")}${I?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${M} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${h.
message_count} / \u6697\u53F7\u5316: ${h.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${_}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),tt=r(async()=>{if(Ee){Ee.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){Ee.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(nt(f),currentThreadId&&Array.isArray(f.threads)){const h=f.threads.find(b=>String(
b.thread_id)===String(currentThreadId));h&&(dt=!!h.encrypted)}}catch{Ee.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>tt()),window.__loadAdminEncThreads=
tt,window.__refreshAdminThreadEncState=et,window.__setAdminThreadEncryption=_t;const St=get("encrypt\
ion-status-admin-toggle");St&&St.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ee&&(Ee.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const A=f.getAttribute("data-id");if(!A)return;typeof Ze==
"function"?Ze():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const h=u.target.closest(".admin-enc-toggle");if(!h||ft)return;const b=h.getAttribute("data-i\
d"),_=h.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;h.disabled=!0;
const M=h.textContent;h.textContent="\u51E6\u7406\u4E2D...";try{await _t(b,_,{confirmPrompt:!1,reloadCurrent:!0})}finally{
h.disabled=!1,h.textContent=M,await tt()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const Ln=r(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(h=>{const b=h.status||"new",_=h.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=h.created_at?new Date(h.created_at).toLocaleString():"",M=h.replied_at?new Date(h.replied_at).toLocaleString():
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
                    `}).join("")}},"renderBanAppeals"),en=r(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const h=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!h.
ok)return;const _=(await h.json()).unread_count||0;f.textContent=String(_),u&&_>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${_}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Wt=r(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const h=await f.json();Ln(h.items||[]),
await en(!1)}catch{}}},"loadBanAppeals"),dn=r(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await Wt()}catch{}},"markBanAppealsRead"),je=r(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await Wt()}catch{}},"updateBanAppealStatus"),Je=r(()=>{
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
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),Ot=r(()=>{const u=get("set-st\
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
"success")})},"ensureLlmTranscribePromptSettingsUi"),Ct=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
ded p-2 text-xs text-gray-200";return Ct.map(b=>{const _=b.barLocked===!0,S=b.barLabel||"MCP",M=_?`<\
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
                        ${M}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{Ct.forEach(h=>{const b=f&&
typeof f=="object"?f[h.key]||{}:{},_=get(`${u}-auto-sys-${h.key}-enabled`),S=get(`${u}-auto-sys-${h.
key}-text`);_&&(h.barLocked===!0?_.disabled=!0:_.checked=b.enabled!==!1),S&&(S.value=b.text||"",S.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const un=r((u,f=null)=>{if(f){const h=get(f);h&&(h.checked=!0)}Ct.forEach(h=>{const b=get(`${u}-auto\
-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);if(b&&(h.barLocked!==!0?b.checked=!0:b.
disabled=!0),_){const S=_.placeholder||"";_.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Nt=r(u=>{const f={};return Ct.forEach(h=>{const b=get(`${u}\
-auto-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);f[h.key]={enabled:h.barLocked===!0?
!0:b?b.checked:!0,text:_?_.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
Nt,window.ensureAutoSystemPromptSettingsCard=()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.
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
                `,f.appendChild(h)},Je(),Ot(),U();const Me=r(()=>{const u=get("set-default-model");if(!u)
return;const f=u.value;u.innerHTML="",MODELS.forEach(b=>{const _=document.createElement("optgroup");
_.label=b.category,(b.items||[]).forEach(S=>{const M=document.createElement("option");M.value=S.id,M.
textContent=S.name,_.appendChild(M)}),u.appendChild(_)});const h=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";h&&Array.from(u.options).some(b=>b.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),Pe=r(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(b=>{const _=(b.items||[]).filter(M=>{const A=(M.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(_.length===0)return;const S=document.createElement("optgroup");S.label=b.category,_.forEach(
M=>{const A=document.createElement("option");A.value=M.id,A.textContent=M.name+" \u2605",S.appendChild(
A)}),u.appendChild(S)});const h=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";h&&Array.from(u.options).some(b=>b.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),it=r(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-glo\
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
get("set-default-url-context")&&(get("set-default-url-context").checked=!!u.default_enable_url_context),
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
et-default-2fa-method")&&(get("set-default-2fa-method").value=u.default_2fa_method||"totp");const M=get(
"set-passkey-only-login"),A=get("passkey-only-note"),I=Array.isArray(u.passkey_credentials)?u.passkey_credentials:
[];if(ue(I),M){M.checked=!!u.passkey_only_login;const de=I.length>0||!!u.has_webauthn;M.disabled=!de,
de||(M.checked=!1),A&&(de?A.classList.add("hidden"):A.classList.remove("hidden"))}const B=get("mig-s\
tatus-box"),D=get("mig-progress-text"),Q=get("mig-progress-bar");if((u.migration_status||"idle")==="\
processing"){B.classList.remove("hidden");const de=(u.migration_progress||"").split("/");if(de.length===
2){const W=parseInt(de[0]||"0",10),ve=parseInt(de[1]||"0",10);D&&(D.innerText=`${W} / ${ve}`),Q&&ve>
0&&(Q.style.width=`${Math.min(100,Math.floor(W/ve*100))}%`)}}else B.classList.add("hidden"),Q&&(Q.style.
width="0%"),D&&(D.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const u=await ensureUserSettingsSnapshot();u&&it(u);const f=get(
"search-box"),h=f?f.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),Me(),Pe(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(h,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(_=>{setTimeout(()=>{restoreThreadSearchValue(
h,"restored-search-box-"+_+"ms"),snapshotSidebarHistory("settings-open-later-"+_+"ms")},_)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),le(),Ot(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),en(!0),Wt(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),gn(),ke(),ne();try{loadMcpServers()}catch{}};const Ze=r((u=!1)=>{settingsModelPickerTarget&&
window.closeModelModal(!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),
revealPersistentSidebarLists(),snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory(
"settings-close-later-300ms"),300),!u&&location.pathname==="/settings"&&history.back()},"closeSettin\
gsModal"),gt=r(()=>{const u=get("set-theme-color"),f=get("set-theme-color-text"),h=get("theme-reset-\
btn"),b=document.querySelectorAll("#theme-presets .theme-swatch"),_=r((S,M=!0)=>{const A=normalizeHex(
S);A&&(applyThemeColor(A,M),syncThemeInputs(A))},"applyFromValue");u&&u.addEventListener("input",()=>_(
u.value,!0)),f&&(f.addEventListener("change",()=>{const S=normalizeHex(f.value);if(!S){syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||THEME_DEFAULT);return}_(S,!0)}),f.addEventListener("keydown",
S=>{S.key==="Enter"&&(S.preventDefault(),f.blur())})),h&&(h.onclick=()=>_(THEME_DEFAULT,!0)),b.forEach(
S=>{S.addEventListener("click",()=>_(S.getAttribute("data-color"),!0))})},"bindThemeControls"),Rt=r(
()=>{const u=get("reset-global-sys-prompt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-pro\
mpt-text").value=""),get("set-global-sys-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=
!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const f=get("reset-thread-sys-prompt");f&&(f.onclick=()=>{get("thread-global-sys-prompt")&&
(get("thread-global-sys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-glo\
bal-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-set-auto-sys-prompt-defaults");h&&(h.onclick=()=>{un("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const b=get("reset-thread-auto-sys-prompt-defaults");b&&(b.onclick=()=>{un("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Ze();const Bt=get("settings-header-close");Bt&&(Bt.onclick=()=>Ze());
const tn=get("settings-search");tn&&(tn.addEventListener("input",filterSettings),tn.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const h=f.querySelector(
":scope > .settings-match");h&&h.scrollIntoView({behavior:"smooth",block:"start"})}}));const Qn=get(
"settings-search-clear");Qn&&Qn.addEventListener("click",()=>{tn&&(tn.value="",filterSettings(),tn.focus())}),
gt(),Rt(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Bn=get("set-gemini-local-python-dialog");Bn&&(Bn.onchange=()=>setGeminiLocalPyDialogEnabled(Bn.
checked));const Zn=get("set-gemini-backend");Zn&&(Zn.onchange=()=>{syncGeminiBackendUi(),typeof window.
toggleOptions=="function"&&window.toggleOptions()});const ei=get("set-admin-api-key-mode");ei&&(ei.onchange=
()=>syncAdminApiKeyModeUi());const jn=get("set-temp-chat-timeout-seconds");jn&&(jn.onchange=()=>{applyTemporaryChatTimeoutSeconds(
jn.value)});const ti=get("slash-command-cancel-btn");ti&&(ti.onclick=()=>{hidePendingSlashCommandIndicator();
const u=get("prompt-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-s\
ettings-btn").onclick=async()=>{var A;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),h=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:Nt(
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
"set-default-gemini-free-tier")?get("set-default-gemini-free-tier").checked:!1,default_enable_url_context:get(
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
const M=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"appli\
cation/json"},body:JSON.stringify(b)});if(M.ok){let I="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const oe=await M.json();oe&&oe.message&&(I=oe.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled,
math_steps_enabled:!!b.math_steps_enabled,default_math_steps:!!b.default_math_steps})),typeof window.
applyMathStepsSettings=="function"&&(window.applyMathStepsSettings({math_steps_enabled:!!b.math_steps_enabled}),
b.math_steps_enabled||setMathStepsEnabled(!1,{persist:!1})),window.applySavedUserSystemPromptSettings(
{system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,apply_global_system_prompt:b.
apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,auto_system_prompt_notices_config:b.
auto_system_prompt_notices_config}),Ze();const B=currentUsername,D=CHAT_CONFIG.enableE2EE;enterToSend=
b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const Q=useSwCache;useSwCache=b.use_sw_cache,
window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),Q!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(I,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==B?setTimeout(()=>location.
reload(),1e3):b.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let I={};try{I=await M.json()}catch{}showToast(I.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
X)?`${X.toFixed(0)} ms`:"-","fmtMs"),M=r(X=>Number.isFinite(X)?`${X.toFixed(X>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=r(async(X,we)=>{const Se=await X.json().catch(()=>({}));return Se&&Se.error?Se.error:we},
"parseErr"),I=[];h("\u6E2C\u5B9A\u4E2D... ping");for(let X=0;X<4;X++){const we=performance.now(),Se=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),ae=performance.now();if(!Se.ok)throw new Error(await A(
Se,"ping_failed"));await Se.json().catch(()=>({})),I.push(ae-we)}const B=I.reduce((X,we)=>X+we,0)/Math.
max(1,I.length),D=Math.min(...I),Q=r(async X=>{const we=performance.now(),Se=await apiFetch(`/api/sp\
eedtest/download?bytes=${X}&_=${b()}`,{cache:"no-store"});if(!Se.ok)throw new Error(await A(Se,"down\
load_failed"));const ae=await Se.arrayBuffer(),xe=performance.now();return{bytes:ae.byteLength||X,ms:xe-
we,mbps:_(ae.byteLength||X,xe-we)}},"runDownload");h(`\u6E2C\u5B9A\u4E2D... ping ${S(B)}
\u6E2C\u5B9A\u4E2D... download`);const oe=[];for(const X of[2*1024*1024,8*1024*1024])oe.push(await Q(
X)),h(`\u6E2C\u5B9A\u4E2D... ping ${S(B)}
download ${M(Math.max(...oe.map(we=>we.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const de=Math.max(...oe.map(X=>X.mbps)),W=r(async X=>{const we=new Uint8Array(
X),Se=performance.now(),ae=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:we,cache:"no-store"}),xe=performance.now();if(!ae.ok)
throw new Error(await A(ae,"upload_failed"));const Oe=await ae.json().catch(()=>({})),Ne=Number(Oe.bytes_received||
X)||X;return{bytes:Ne,ms:xe-Se,mbps:_(Ne,xe-Se),serverMs:Number(Oe.server_elapsed_ms||0)||0}},"runUp\
load"),ve=[];for(const X of[1*1024*1024,4*1024*1024])ve.push(await W(X));const G=Math.max(...ve.map(
X=>X.mbps)),he=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(B)} / ${S(D)}`,`Download (best): ${M(de)}`,`Upload (best): ${M(G)}`,`Download r\
uns: ${oe.map(X=>`${Math.round(X.bytes/1024/1024)}MB=${M(X.mbps)}`).join(", ")}`,`Upload runs: ${ve.
map(X=>`${Math.round(X.bytes/1024/1024)}MB=${M(X.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
h(he.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(h){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${h&&h.message?h.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Wt()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>dn()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){h&&await dn([Number(h)]);return}if(f.classList.contains(
"ban-appeal-status")){const b=f.getAttribute("data-status");h&&b&&await je({id:Number(h),status:b});
return}if(f.classList.contains("ban-appeal-reply-send")){const b=f.closest("[data-appeal-id]"),_=b?b.
querySelector(".ban-appeal-reply"):null,S=_?_.value:"";h&&await je({id:Number(h),admin_reply:S});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";h&&await je({id:Number(
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
!u||u.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const Li=r(()=>{const u=get(
"upload-modal");return!!(u&&!u.classList.contains("hidden"))},"isUploadModalOpen"),nn=get("drop-over\
lay");let pn=0;const Mi=r(()=>{Li()||nn&&(nn.classList.remove("hidden"),nn.classList.add("flex"))},"\
showDropOverlay"),mn=r(()=>{pn=0,nn&&(nn.classList.add("hidden"),nn.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=mn;const Mt=get("upload-dropzone");Mt&&(Mt.addEventListener("dr\
agover",u=>{u.preventDefault(),Mt.classList.add("dragover")}),Mt.addEventListener("dragleave",()=>{Mt.
classList.remove("dragover")}),Mt.addEventListener("drop",u=>{u.preventDefault(),u.stopPropagation(),
Mt.classList.remove("dragover"),mn();const f=u.dataTransfer?u.dataTransfer.files:null;f&&f.length&&handleFiles(
f)})),window.addEventListener("dragenter",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.
types.includes("Files")||(pn+=1,Mi())}),window.addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||u.preventDefault()}),window.addEventListener("dragle\
ave",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(pn=Math.max(
0,pn-1),(pn===0||!u.relatedTarget||u.clientY<=0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=
window.innerHeight)&&mn())}),window.addEventListener("dragend",()=>{mn()}),window.addEventListener("\
drop",u=>{mn(),!(!u.dataTransfer||!u.dataTransfer.files||u.dataTransfer.files.length===0)&&(u.preventDefault(),
!(Mt&&Mt.contains(u.target))&&handleFiles(u.dataTransfer.files))});const ni=get("bot-admin-modal"),Ti=r(
u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML="",!u||!u.length){f.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((h,b)=>{const _=!!h.is_bot_banned,S=h.bot_detection_enabled!==!1,M=Number(h.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(b,12)*.02}\
s`,A.innerHTML=`
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
                    `,f.appendChild(A)})}},"renderBotUsers"),fn=r(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const h=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),b=await h.json();h.ok&&b&&b.users?Ti(b.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{f&&(f.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Hn=r(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||ni))return;const f=get(
"settings-modal");f&&(f.classList.contains("modal-open")||f.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await fn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Hn;const qn=r(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const u=get("settings-modal");u&&(u.classList.contains(
"modal-open")||u.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=qn,window.closeBotLogModal=(u=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!u&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{qn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(u=!1)=>{(get("bot-admin-modal")||ni)&&hideModal("bot-admin-modal"),!u&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Hn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await fn(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await fn("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async u=>{u.key==="Enter"&&await fn(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
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
_.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await fn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const jt={"/settings":{id:"settings-mod\
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
-modal",open:r(()=>qn(),"open")}};jt["/gem"]=jt["/skill"];const ii=r((u,f=!1)=>{switch(u){case"setti\
ngs-modal":Ze(f);break;case"upload-modal":closeUploadModal(f);break;case"image-split-modal":closeImageSplit();
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
addEventListener("popstate",u=>{var S,M;const f=(S=u.state)==null?void 0:S.settingsModelPicker;if(location.
pathname==="/settings"&&settingsModelPickerTargets.includes(f)){const A=r(()=>openSettingsModelPicker(
f,!0),"restorePicker");(M=get("settings-modal"))!=null&&M.classList.contains("modal-open")?A():window.
openSettingsModal().then(()=>{var I;location.pathname==="/settings"&&((I=history.state)==null?void 0:
I.settingsModelPicker)===f&&A()});return}settingsModelPickerTarget&&window.closeModelModal(!0);let h=!1;
Object.values(jt).forEach(A=>{const I=get(A.id);I&&I.classList.contains("modal-open")&&location.pathname!==
Object.keys(jt).find(B=>jt[B].id===A.id)&&(ii(A.id,!0),h=!0)});const b=location.pathname.match(/^\/c\/(.+)$/);
if(b){const A=decodeURIComponent(b[1]);String(currentThreadId)!==String(A)&&loadMessages(A,{skipHistory:!0})}else
location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});const _=jt[location.pathname];
if(_){const A=get(_.id);A&&!A.classList.contains("modal-open")&&_.open()}}),document.addEventListener(
"keydown",u=>{u.key==="Escape"&&settingsModelPickerTarget&&(u.preventDefault(),window.closeModelModal())});
const si=location.pathname;jt[si]&&(history.replaceState({},"","/"),setTimeout(()=>jt[si].open(),500)),
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
"error",!0)});const Ai=4,Ei=5*1024*1024;function ai(u,f,h){let b=[];const _=r(()=>{h&&(h.innerHTML="",
b.forEach((S,M)=>{const A=document.createElement("div");A.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const I=document.createElement("img");I.src=S.url,I.alt=S.file.name,
I.className="w-full h-full object-cover";const B=document.createElement("button");B.type="button",B.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",B.textContent=
"\xD7",B.onclick=()=>{URL.revokeObjectURL(S.url),b.splice(M,1),_()},A.appendChild(I),A.appendChild(B),
h.appendChild(A)}))},"render");return u&&f&&(u.onclick=()=>f.click(),f.onchange=S=>{for(const M of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(M.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(M.
size>Ei){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(b.
length>=Ai){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}b.push({file:M,url:URL.createObjectURL(M)})}S.target.value="",_()}),{files:r(()=>b.
map(S=>S.file),"files"),clear:r(()=>{b.forEach(S=>URL.revokeObjectURL(S.url)),b=[],_()},"clear")}}r(
ai,"createFbImagePicker");async function oi(u,f){try{const h=new FormData;return f.forEach(b=>h.append(
"images",b,b.name)),(await apiFetch(u,{method:"POST",body:h})).ok}catch{return!1}}r(oi,"uploadFbImag\
es");const ri=ai(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const u=get("fb-title").value.trim(),f=get("fb-message").value.trim();if(!f){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const h={title:u,message:f,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=get("fb-attach-logs"),_=b&&b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;_&&(h.client_logs=_);const S=get("fb-attach-chat"),M=!!(S&&S.checked);if(M&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}M&&(h.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const A=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)}),I=A.ok?await A.clone().json().
catch(()=>({})):{},B=ri.files(),D={count:B.length,saved:!0};A.ok&&D.count&&(D.saved=await oi(`/api/f\
eedback/${I.public_id}/images`,B)),(window.ActivityLog?await window.ActivityLog.reportFeedback(A,_,M,
D):A.ok)&&(get("fb-title").value="",get("fb-message").value="",ri.clear(),S&&(S.checked=!1),b&&(b.checked=
!0),gn())};async function li(u){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${u}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),gn()}r(li,"deleteFeedback");const $i="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Gn=[];function ci(u,f){const h=u.replies||[];return h.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${h.map(b=>{const _=b.role==="admin"?f&&b.author?`\u7BA1\u7406\u8005 (${b.
author})`:"\u7BA1\u7406\u8005":f?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=f?[b.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${b.
log_file}`,b.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${b.chat_file}`,b.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${b.image_count}\u679A\uFF09: ${b.image_dir}`]:[b.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${b.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(_)}\
 / ${escapeHtml(b.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${b.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(b.message)}</div>${S.filter(Boolean).map(M=>`<div cla\
ss="${f?"text-amber-300":"text-gray-400"}">${escapeHtml(M)}</div>`).join("")}</div>`}).join("")}</di\
v>`:""}r(ci,"feedbackRepliesHtml");function di(){return`<div class="fb-reply-form mt-2 space-y-2">
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
                            <span class="block text-xs font-bold text-white">${$i}</span>
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
                </div>`}r(di,"feedbackReplyFormHtml");function ui(u,f){const h=f.querySelector(".fb-\
reply-form");if(!h)return;const b=h.querySelector(".fb-reply-logs"),_=h.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());b.disabled=!S,_.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const M=ai(h.querySelector(".fb-reply-images-add"),h.querySelector(".fb-reply-images-input"),h.querySelector(
".fb-reply-images-list"));Gn.push(M);const A=h.querySelector(".fb-reply-send");A.onclick=async()=>{const I=h.
querySelector(".fb-reply-message").value.trim();if(!I){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const B={message:I,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},D=b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;D&&(B.client_logs=D);const Q=h.
querySelector(".fb-reply-chat").checked;if(Q&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}Q&&(B.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const oe=M.files();A.disabled=!0;try{const de=await apiFetch(
`/api/feedback/${u.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(B)}),W=await de.json().catch(()=>({}));if(!de.ok){showToast(W.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const ve=!oe.length||await oi(`/api/feedback/${u.public_id}/replies/${W.reply_id}\
/images`,oe),G=[];if(D&&W.logs_saved===!1&&G.push("\u30ED\u30B0"),ve||G.push("\u753B\u50CF"),Q&&W.chat_copy_saved!==
!0&&G.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),G.length)showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${G.
join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,"error",!0);else{
const he=["\u8FD4\u4FE1"];Q&&he.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),D&&he.push(
`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof W.logs_count=="number"?W.logs_count:
D.entries.length}\u4EF6\uFF09`),oe.length&&he.push(`\u753B\u50CF\uFF08${oe.length}\u679A\uFF09`),showToast(
he.join(he.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","success")}gn()}finally{
A.disabled=!1}}}r(ui,"bindFeedbackReply");function pi(){const u=get("fb-admin-status-filter").value;
let f=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(h=>{const b=u==="all"||
(u==="unresolved"?["new","in_review","replied"].includes(h.dataset.feedbackStatus):h.dataset.feedbackStatus===
u);h.hidden=!b,b&&f++}),get("fb-admin-empty").classList.toggle("hidden",f!==0)}r(pi,"filterAdminFeed\
back"),get("fb-admin-status-filter").onchange=pi;async function gn(){Gn.forEach(S=>S.clear()),Gn=[];
const f=await(await apiFetch("/api/feedback?all=1")).json(),h=get("fb-list");h.innerHTML="",(f.items||
[]).filter(S=>!f.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(f.is_admin)return;
const M=document.createElement("div");M.className="p-2 rounded border border-gray-700 bg-gray-800/50",
M.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
S.public_id||"")}</span> / ${S.created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.
image_count}\u679A</div>`:""}${ci(S,!1)}${di()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
M.querySelector(".fb-delete").onclick=()=>li(S.public_id),ui(S,M),h.appendChild(M)});const b=get("fb\
-admin-panel"),_=get("fb-admin-list");f.is_admin?(b.classList.remove("hidden"),_.innerHTML="",(f.items||
[]).forEach(S=>{const M=document.createElement("div");M.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",M.dataset.feedbackStatus=S.status||"new",M.innerHTML=`
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
                            ${ci(S,!0)}
                            ${di()}
                        `,M.querySelector(".fb-status").value=S.status||"new",M.querySelector(".fb-s\
ave").onclick=async()=>{const A=M.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:A})}),
gn()},M.querySelector(".fb-delete").onclick=()=>li(S.public_id),ui(S,M),_.appendChild(M)}),pi()):b.classList.
add("hidden")}if(r(gn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=
f.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get(
"totp-verify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),h=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),_=await b.json();if(!b.ok){showToast(_.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const M=await(await ensureWebAuthnJson()).create({publicKey:_}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},M,{name:h}))}),I=await A.json().catch(()=>({}));A.ok?(f&&(f.value=""),showToast(
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
click()}),get("sts-cancel-btn")&&(get("sts-cancel-btn").onclick=()=>{isStsModel()&&Ii()}),get("promp\
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
"";f&&h&&(u.preventDefault(),openImageViewer(h))});const hn=document.querySelector(".viewer-content");
hn&&(hn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),hn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),hn.addEventListener("touchend",onViewerTouchEnd),hn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",u=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(u.target.id==="image-viewer"||u.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",u=>{u.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",u=>{u.key==="Escape"&&closeImageViewer()});
let ut,qe=null,Tn=[],Un=!1,An=null,sn=null,Vt=null,bn=null,zn=0,En=!1,yn=null,an=null,Ft=null,vn=null,
$n=null,Jt=null,In=null,Pn=null;function mi(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray(
In)&&In.length)return In;u.innerHTML="";const f=[];for(let h=0;h<24;h++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",f.push(b),u.appendChild(b)}return In=f,f}r(mi,"ensureMicWaveformBars");function Kt(u,f="hidden"){
const h=get("mic-recording-indicator"),b=get("mic-recording-text");if(h){if(Pn&&(clearTimeout(Pn),Pn=
null),f==="hidden"){h.classList.add("hidden");return}b&&u&&(b.innerText=u),h.classList.remove("hidde\
n"),f==="recording"?h.style.color="rgb(252 165 165)":f==="processing"?h.style.color="rgb(253 224 71)":
h.style.color="rgb(209 213 219)"}}r(Kt,"setMicRecordingIndicator");function fi(){mi().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}r(fi,"resetMicWaveformBars");function Xt(){if($n&&(cancelAnimationFrame(
$n),$n=null),vn){try{vn.disconnect()}catch{}vn=null}if(an){try{an.close()}catch{}an=null}Ft=null,Jt=
null,fi()}r(Xt,"stopMicWaveform");function gi(u){Xt();const f=mi();if(!f.length)return;const h=window.
AudioContext||window.webkitAudioContext;if(!h)return;try{an=new h,Ft=an.createAnalyser(),Ft.fftSize=
256,Ft.smoothingTimeConstant=0,vn=an.createMediaStreamSource(u),vn.connect(Ft),Jt=new Uint8Array(Ft.
frequencyBinCount)}catch{Xt();return}const b=r(()=>{if(!Ft||!Jt)return;Ft.getByteFrequencyData(Jt);const _=Math.
max(1,Math.floor(Jt.length/f.length));for(let S=0;S<f.length;S++){const A=(Jt[Math.min(Jt.length-1,S*
_)]||0)/255,I=Math.max(2,Math.round(2+A*10));f[S].style.height=`${I}px`,f[S].style.opacity=`${.35+A*
.65}`}$n=requestAnimationFrame(b)},"render");b()}r(gi,"startMicWaveform");function On(){if(An&&(clearInterval(
An),An=null),bn){try{bn.disconnect()}catch{}bn=null}if(sn){try{sn.close()}catch{}sn=null}Vt=null}r(On,
"stopSilenceMonitor");function hi(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;On();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;sn=new f,Vt=sn.createAnalyser(),Vt.fftSize=2048,
bn=sn.createMediaStreamSource(u),bn.connect(Vt);const h=new Uint8Array(Vt.fftSize),b=getStsSilenceMs(),
_=.02;zn=0,En=!1,An=setInterval(()=>{if(!Vt)return;Vt.getByteTimeDomainData(h);let S=0;for(let A=0;A<
h.length;A++){const I=(h[A]-128)/128;S+=I*I}if(Math.sqrt(S/h.length)>_){En||(En=!0),zn=Date.now();return}
En&&Date.now()-zn>b&&ut&&ut.state==="recording"&&ut.stop()},200)}r(hi,"startSilenceMonitor");const Jn=class Jn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(f,h,b,_={},S=null){if(this.
model=b,this.ws=new WebSocket(`${h}?access_token=${encodeURIComponent(f)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const I=!!(_&&
_.transcriptionConfig),B={setup:{model:`models/${b}`,generationConfig:{responseModalities:I?["TEXT"]:
["AUDIO"]},inputAudioTranscription:I?_.transcriptionConfig||{}:{}}};I||(B.setup.outputAudioTranscription=
{}),_.speechConfig&&(B.setup.generationConfig.speechConfig=_.speechConfig),_.thinkingConfig&&(B.setup.
generationConfig.thinkingConfig=_.thinkingConfig),_.translationConfig&&(B.setup.generationConfig.translationConfig=
_.translationConfig),console.log("Sending setup:",JSON.stringify(B)),this.ws.send(JSON.stringify(B))},
this.ws.onmessage=I=>this._handleMessage(I),this.ws.onerror=I=>{console.error("Gemini Live WebSocket\
 error:",I),this.onError&&this.onError(I)},this.ws.onclose=I=>{console.log("Gemini Live WebSocket cl\
osed:",I.code,I.reason),this.closedEvent=I,this.onClose&&this.onClose(I)},S&&S.stream&&S.ctx&&S.source){
this.stream=S.stream,this.audioContext=S.ctx;var M=S.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(Nn());const I=bi(this.stream,16e3);this.audioContext=I.ctx;var M=I.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=I=>{I.data.
size>0&&this.userAudioChunks.push(I.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
I=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const B=yi(I.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!B||!B.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(B))),mimeType:"audio/\
pcm;rate=16000"}}}))},M.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(f){
let h=null;try{h=JSON.parse(typeof f.data=="string"?f.data:new TextDecoder().decode(f.data))}catch{return}
if(h.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),h.serverContent){
const b=h.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(_=>{if(_.text&&(_.thought?(console.log(
"Gemini thought delta:",_.text),this.assistantThought+=_.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(_.text)),_.inlineData&&_.inlineData.data){const S=_.inlineData.data;
console.log("Gemini audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);
const M=atob(S),A=new Uint8Array(M.length);for(let I=0;I<M.length;I++)A[I]=M.charCodeAt(I);this.assistantAudioChunks.
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
started=!1}async addChunk(f){if(!this.ctx)return;const h=atob(f),b=new Uint8Array(h.length);for(let B=0;B<
h.length;B++)b[B]=h.charCodeAt(B);const _=new Int16Array(b.buffer),S=new Float32Array(_.length);for(let B=0;B<
_.length;B++)S[B]=_[B]/32768;const M=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);M.getChannelData(
0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();
A.buffer=M,A.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const I=Math.max(this.ctx.currentTime,this.nextStartTime);A.start(I),this.
nextStartTime=I+M.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r(Kn,"RealTimeAudioPl\
ayer");let wn=Kn;const Xn=class Xn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
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

`))>=0;){const M=h.slice(0,S);h=h.slice(S+2);for(const A of M.split(`
`)){if(!A.startsWith("data: "))continue;let I=null;try{I=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent(I)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
if(f)switch(f.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new wn(this.rateOut||24e3),on=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}on=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(Nn()).then(f=>{this.stream=f;const h=this.rateIn||24e3,b=bi(f,h);this.audioCtx=
b.ctx;const _=b.source,S=this.audioCtx.sampleRate,M=4096;this.processor=this.audioCtx.createScriptProcessor(
M,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const I=A.inputBuffer.
getChannelData(0),B=yi(I,S,h);!B||!B.byteLength||this._sendAudio(B)},_.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(f){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(f)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const f=this.audioQueue.splice(0,this.audioQueue.length),
h=f.reduce((I,B)=>I+B.byteLength,0),b=new Uint8Array(h);let _=0;f.forEach(I=>{b.set(I,_),_+=I.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),M={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(M):M;try{await fetch(S,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}On(),Xt()}async stop(){if(!this.active)return;if(this.
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
Xn,"RealtimeVoiceSession");let Vn=Xn;function bi(u,f){const h=window.AudioContext||window.webkitAudioContext;
if(!h)throw new Error("AudioContext not supported");let b=null;try{return b=new h({sampleRate:f}),{ctx:b,
source:b.createMediaStreamSource(u)}}catch{if(b)try{b.close()}catch{}return b=new h,{ctx:b,source:b.
createMediaStreamSource(u)}}}r(bi,"openMicAudioSource");function yi(u,f,h){let b=u;if(f!==h&&f>0&&h>
0){const S=f/h,M=Math.floor(b.length/S),A=new Float32Array(M);for(let I=0;I<M;I++)A[I]=b[Math.min(Math.
floor(I*S),b.length-1)];b=A}const _=new Int16Array(b.length);for(let S=0;S<b.length;S++){const M=Math.
max(-1,Math.min(1,b[S]));_[S]=M<0?M*32768:M*32767}return _.buffer}r(yi,"pcm16FromFloat32");const kn=new Vn,
vi=(()=>{const h={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,_=null,S=!1,M=null,A=!1,I=0,B=0,D=null,Q="idle",oe=!1,de=null;const W=r(q=>document.getElementById(
q),"$"),ve=r(q=>{const te=Object.assign({},q||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(te):(te.progressSpinner=
!1,te)},"noSpinner");function G(q,te){Q=te;const Ae=W("lyria-status-text"),ye=W("lyria-status-dot");
Ae&&(Ae.textContent=q),ye&&(ye.className="w-2 h-2 rounded-full inline-block "+(h[te]||h.idle)),Se(),
ae()}r(G,"setStatus");function he(){const q=B?Math.floor((Date.now()-B)/1e3):0,te=String(Math.floor(
q/60)).padStart(2,"0"),Ae=String(q%60).padStart(2,"0");return`${te}:${Ae}`}r(he,"formatElapsed");function X(){
B||(B=Date.now());const q=W("lyria-elapsed");q&&(q.textContent=he()),D||(D=window.setInterval(()=>{const te=W(
"lyria-elapsed");te&&(te.textContent=he())},1e3))}r(X,"startElapsedTimer");function we(){D&&(window.
clearInterval(D),D=null)}r(we,"stopElapsedTimer");function Se(){const q=W("lyria-play-btn"),te=W("ly\
ria-pause-btn"),Ae=W("lyria-stop-btn"),ye=W("lyria-reset-btn"),$e=!!b,We=Q==="streaming"||Q==="conne\
cting";if(q){q.disabled=oe;const at=q.querySelector("i");at&&(at.className="fas fa-play")}te&&(te.disabled=
oe||!We),Ae&&(Ae.disabled=oe||!$e||!We),ye&&(ye.disabled=oe||!$e||!We)}r(Se,"updateTransportButtons");
function ae(){const q=W("lyria-save-btn");if(!q)return;const te=!!b&&Q!=="idle"&&Q!=="connecting"&&Q!==
"error";q.classList.toggle("hidden",!te)}r(ae,"updateSaveButton");function xe(q,te){const Ae=W("lyri\
a-prompt-rows");if(!Ae)return;const ye=document.createElement("div");ye.className="flex items-center\
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
                    `;const $e=ye.querySelector('input[type="range"]'),We=ye.querySelector(".lyria-w\
eight-label");$e&&We&&$e.addEventListener("input",()=>{We.textContent=parseFloat($e.value).toFixed(1)});
const at=ye.querySelector(".lyria-prompt-remove");at&&at.addEventListener("click",()=>{Ae.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ye.remove()}),ye.classList.add("lyria-prompt-row-wrap"),Ae.appendChild(
ye)}r(xe,"addPromptRow");function Oe(){const q=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),te=[];return q.forEach(Ae=>{const ye=Ae.querySelector('input[type="text"]'),$e=Ae.
querySelector('input[type="range"]'),We=(ye?ye.value:"").trim();We&&te.push({text:We,weight:parseFloat(
$e?$e.value:1)||1})}),te}r(Oe,"collectPrompts");function Ne(){const q={},te=r(Ri=>{const Yn=W(Ri);return Yn&&
Yn.value!==""?parseFloat(Yn.value):void 0},"num"),Ae=te("lyria-bpm");Ae!==void 0&&(q.bpm=Math.round(
Ae));const ye=te("lyria-guidance");ye!==void 0&&(q.guidance=ye);const $e=te("lyria-density");$e!==void 0&&
(q.density=$e);const We=te("lyria-brightness");We!==void 0&&(q.brightness=We);const at=te("lyria-tem\
perature");at!==void 0&&(q.temperature=at);const Ye=W("lyria-scale");Ye&&Ye.value&&(q.scale=Ye.value);
const lt=W("lyria-mode");lt&&lt.value&&(q.music_generation_mode=lt.value);const yt=W("lyria-mute-bas\
s"),Qt=W("lyria-mute-drums"),Ci=W("lyria-only-bass-drums");return yt&&(q.mute_bass=yt.checked),Qt&&(q.
mute_drums=Qt.checked),Ci&&(q.only_bass_and_drums=Ci.checked),q}r(Ne,"collectConfig");function pt(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([te,Ae])=>{const ye=W(te),$e=W(Ae);!ye||!$e||ye.addEventListener("input",()=>{const We=parseFloat(
ye.value);$e.textContent=te==="lyria-bpm"?String(Math.round(We)):We.toFixed(1)})})}r(pt,"bindRangeLa\
bels");function Qe(){if(M){try{M.close()}catch{}M=null}A=!1,I=0}r(Qe,"resetPlayback");function bt(){
if(S=!1,_&&typeof _.abort=="function")try{_.abort()}catch{}_=null}r(bt,"closeStream");async function Dt(){
bt(),_=new AbortController,S=!0;try{const q=await fetch(`/api/skillini/music/stream?session_id=${encodeURIComponent(
b)}`,ve({method:"GET",signal:_.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!q.
ok){const $e=await q.json().catch(()=>({}));throw new Error($e.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const te=q.body.getReader(),Ae=new TextDecoder;let ye="";for(;S;){const{done:$e,value:We}=await te.read();
if($e)break;ye+=Ae.decode(We,{stream:!0});const at=ye.split(`

`);ye=at.pop();for(const Ye of at){const lt=Ye.split(`
`).find(Qt=>Qt.startsWith("data: "));if(!lt)continue;const yt=lt.slice(6);try{const Qt=JSON.parse(yt);
Te(Qt)}catch{}}}}catch(q){if(q&&q.name==="AbortError")return;S&&(G("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&b&&Dt()},1200))}finally{S=!1}}r(Dt,"openStream");function Te(q){
if(q&&q.snapshot){const te=q.status;if(te==="error"){G("\u30A8\u30E9\u30FC","error"),we();return}if(te===
"closed"||te==="stopped"){G("\u7D42\u4E86","closed"),we();return}G(te==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",te==="paused"?"paused":"connecting");return}if(q&&q.audio){G("\u518D\u751F\u4E2D...",
"streaming"),X(),Ce(q.audio);return}if(q&&q.error){G("\u30A8\u30E9\u30FC: "+q.error,"error"),we();return}
if(q&&q.final){G("\u7D42\u4E86","closed"),we(),Se();return}}r(Te,"handleStreamMessage");function Ce(q){
if(!q)return;if(!M){const Ye=window.AudioContext||window.webkitAudioContext;if(!Ye)return;M=new Ye({
sampleRate:48e3}),A=!1,I=0}let te;try{const Ye=atob(q);te=new Uint8Array(Ye.length);for(let lt=0;lt<
Ye.length;lt++)te[lt]=Ye.charCodeAt(lt)}catch{return}const Ae=new Int16Array(te.buffer),ye=Math.floor(
Ae.length/2);if(ye<1)return;const $e=M.createBuffer(2,ye,48e3);for(let Ye=0;Ye<2;Ye++){const lt=$e.getChannelData(
Ye);for(let yt=0;yt<ye;yt++)lt[yt]=Ae[yt*2+Ye]/32768}M.state==="suspended"&&M.resume();const We=M.createBufferSource();
We.buffer=$e,We.connect(M.destination),A||(I=M.currentTime+.08,A=!0);const at=Math.max(M.currentTime,
I);We.start(at),I=at+$e.duration}r(Ce,"playChunk");async function Ue(q,te){const Ae=await fetch("/ap\
i/skillini/music/command",ve({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:q},te||{}))})),ye=await Ae.json().catch(()=>({}));if(!Ae.ok)throw new Error(
ye.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ye}
r(Ue,"apiCommand");async function st(){if(oe)return;const q=Oe();if(!q.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}oe=!0,Se(),G("\u63A5\u7D9A\u4E2D...","connecting");try{
const te=await fetch("/api/skillini/music/start",ve({method:"POST",headers:{"Content-Type":"applicat\
ion/json"},body:JSON.stringify({weighted_prompts:q,config:Ne()})})),Ae=await te.json().catch(()=>({}));
if(!te.ok)throw new Error(Ae.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=Ae.session_id,de=Ne(),G("\u63A5\u7D9A\u4E2D...","connecting"),Dt()}catch(te){G("\u30A8\u30E9\u30FC: "+
te.message,"error"),showToast("Lyria RealTime: "+te.message,"error",!0)}finally{oe=!1,Se()}}r(st,"st\
artSession");async function mt(q){if(b){oe=!0,Se();try{await Ue("control",{action:q}),q==="PLAY"?G("\
\u518D\u751F\u4E2D...","streaming"):q==="PAUSE"?G("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):q==="ST\
OP"?G("\u505C\u6B62\u4E2D","stopped"):q==="RESET_CONTEXT"&&G("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0),G("\u30A8\u30E9\u30FC: "+
te.message,"error")}finally{oe=!1,Se()}}}r(mt,"control");async function Ht(){if(!b)return;const q=Oe();
if(!q.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}oe=!0;try{await Ue("prompts",{weighted_prompts:q}),G("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
Q==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0)}finally{oe=!1,Se()}}r(Ht,"a\
pplyPrompts");async function Le(){if(!b)return;const q=Ne(),te=de||{},Ae=q.bpm!==void 0&&q.bpm!==te.
bpm,ye=q.scale!==void 0&&q.scale!==te.scale,$e=Ae||ye;oe=!0;try{await Ue("config",{config:q,reset_context:$e}),
de=q,G($e?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",Q==="paused"?"paused":"streaming"),showToast(
$e?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(We){showToast("Lyria RealT\
ime: "+We.message,"error",!0)}finally{oe=!1,Se()}}r(Le,"applyConfig");async function Ge(){if(b){oe=!0,
G("\u4FDD\u5B58\u4E2D...","connecting"),Se();try{const q=await fetch("/api/skillini/music/save",ve({
method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),te=await q.json().catch(()=>({}));if(!q.ok)throw new Error(te.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
G("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),we(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),te.thread_id&&(currentThreadId=String(te.thread_id),history.pushState({},"","/c/"+te.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(te.thread_id||currentThreadId),Yt(
!0)}catch(q){G("\u30A8\u30E9\u30FC: "+q.message,"error"),showToast("Lyria RealTime: "+q.message,"err\
or",!0)}finally{oe=!1,Se()}}}r(Ge,"saveSession");async function Fe(){if(bt(),b)try{await fetch("/api\
/skillini/music/cancel",ve({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:b})}))}catch{}b=null,we(),Qe(),G("\u6E96\u5099\u5B8C\u4E86","idle")}r(Fe,"cancelSession");
function ze(){const q=W("lyria-prompt-rows");q&&(q.innerHTML=""),xe("",1),de=null,B=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ye=>{const $e=W(ye);
$e&&($e.value=ye==="lyria-bpm"?"120":ye==="lyria-guidance"?"4":ye==="lyria-temperature"?"1.1":"0.5")});
const te=W("lyria-scale");te&&(te.value="");const Ae=W("lyria-mode");Ae&&(Ae.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ye=>{const $e=W(ye);$e&&($e.checked=
!1)}),pt()}r(ze,"resetControls");function Yt(q){bt(),b&&fetch("/api/skillini/music/cancel",ve({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),
b=null,S=!1,we(),Qe(),hideModal("lyria-studio-modal")}r(Yt,"closeAndCleanup");function _n(q){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const te=W("lyria-studio-modal");if(te&&te.classList.contains("modal-open")&&b){
if(q&&typeof q=="string"){const ye=W("lyria-prompt-rows");ye&&(ye.innerHTML=""),xe(q,1)}return}if(b&&
Fe(),ze(),q&&typeof q=="string"){const ye=W("lyria-prompt-rows");ye&&(ye.innerHTML=""),xe(q,1)}b=null,
S=!1,we(),Qe(),G("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(_n,"open");function Rn(){
const q=W("lyria-open-studio-btn");q&&q.addEventListener("click",()=>_n(""));const te=W("lyria-studi\
o-close");te&&te.addEventListener("click",()=>Yt(!1));const Ae=W("lyria-play-btn");Ae&&Ae.addEventListener(
"click",()=>{if(!b){st();return}mt("PLAY")});const ye=W("lyria-pause-btn");ye&&ye.addEventListener("\
click",()=>mt("PAUSE"));const $e=W("lyria-stop-btn");$e&&$e.addEventListener("click",()=>mt("STOP"));
const We=W("lyria-reset-btn");We&&We.addEventListener("click",()=>mt("RESET_CONTEXT"));const at=W("l\
yria-add-prompt-btn");at&&at.addEventListener("click",()=>xe("",1));const Ye=W("lyria-apply-prompts-\
btn");Ye&&Ye.addEventListener("click",Ht);const lt=W("lyria-apply-config-btn");lt&&lt.addEventListener(
"click",Le);const yt=W("lyria-save-btn");yt&&yt.addEventListener("click",Ge),pt(),ze(),window.openLyriaStudio=
_n}r(Rn,"init");function Sn(){oe||Yt(!1)}return r(Sn,"requestClose"),{init:Rn,open:_n,requestClose:Sn}})();
vi.init(),window.closeLyriaStudio=()=>vi.requestClose(),(()=>{let u=null,f=null,h=null;const b="voic\
eDockSettingsOpen",_="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=r(ae=>document.getElementById(ae),"$");function M(){return isStsModel()&&voiceStudioUiEnabled!==!1}
r(M,"isStudioMode");function A(){const ae=get("model-select")?get("model-select").value:"",xe=S("voi\
ce-studio-title");xe&&(ae==="gpt-transcribe"||ae==="gpt-live-transcribe"||ae==="grok-voice-transcrib\
e-2.0-file"?xe.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ae===
"gemini-3.5-live-translate-preview"?xe.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
xe.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(A,"updateTitle");function I(){const ae=S("v\
oice-studio-transcript");ae&&(ae.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${_}</div>`);const xe=S("sts-live-transcript");xe&&(xe.innerHTML="",xe.classList.add("hidden"))}
r(I,"resetTranscript");function B(ae,xe,Oe){const Ne=ae.querySelectorAll(".voice-studio-line");let pt=null;
for(let Qe=Ne.length-1;Qe>=0;Qe--)if(Ne[Qe].dataset.role===xe){pt=Ne[Qe];break}if(pt)pt.innerHTML=Oe;else{
const Qe=ae.querySelector(".voice-studio-placeholder");Qe&&Qe.remove();const bt=document.createElement(
"div");bt.className="voice-studio-line",bt.dataset.role=xe,bt.innerHTML=Oe,ae.appendChild(bt)}ae.classList.
remove("hidden"),ae.scrollTop=ae.scrollHeight}r(B,"writeLine");function D(ae,xe){if(!xe||!String(xe).
trim()||!M())return;const pt=`<span class="${ae==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ae==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
xe)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(Qe=>B(Qe,
ae,pt))}r(D,"log");function Q(ae,xe=!0){const Oe=S("sts-panel"),Ne=S("sts-settings-toggle");if(Oe&&Oe.
classList.toggle("settings-open",!!ae),Ne&&Ne.setAttribute("aria-expanded",ae?"true":"false"),xe)try{
localStorage.setItem(b,ae?"1":"0")}catch{}}r(Q,"setSettingsOpen");function oe(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}r(oe,"readSettingsOpen");let de=null;function W(){const ae=get("mod\
el-select")?get("model-select").value:"";ae!==de&&(de=ae,I()),Q(oe(),!1),A()}r(W,"syncDock");function ve(){
const ae=S("sts-panel"),xe=S("voice-studio-panel-host");ae&&xe&&ae.parentNode!==xe&&(u=ae.parentNode,
f=ae.nextSibling,xe.appendChild(ae));const Oe=S("file-preview"),Ne=S("voice-studio-file-host");Oe&&Ne&&
Oe.parentNode!==Ne&&(h=Oe.parentNode,Ne.appendChild(Oe),Ne.classList.remove("hidden"))}r(ve,"movePan\
elIntoModal");function G(){const ae=S("sts-panel");ae&&u&&ae.parentNode!==u&&(f&&f.parentNode===u?u.
insertBefore(ae,f):u.appendChild(ae));const xe=S("file-preview");xe&&h&&xe.parentNode!==h&&h.appendChild(
xe);const Oe=S("voice-studio-file-host");Oe&&Oe.classList.add("hidden"),u=null,f=null,h=null}r(G,"mo\
vePanelBack");function he(){if(!M())return;ve();const ae=S("sts-panel");ae&&ae.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}r(he,"open");function X(){window.
VoiceStudioOpen=!1,G(),hideModal("voice-studio-modal")}r(X,"close");function we(){window.VoiceStudioOpen&&
X()}r(we,"closeIfOpen");function Se(){window.VoiceStudioOpen=!1;const ae=S("voice-studio-open-btn");
ae&&ae.addEventListener("click",()=>he());const xe=S("voice-studio-close");xe&&xe.addEventListener("\
click",()=>X());const Oe=S("sts-settings-toggle");Oe&&Oe.addEventListener("click",()=>{const Ne=S("s\
ts-panel");Q(!(Ne&&Ne.classList.contains("settings-open")))}),window.VoiceStudio={open:he,close:X,closeIfOpen:we,
log:D,isStudioMode:M,syncDock:W},W()}return r(Se,"init"),{init:Se}})().init();let on=null;function wi(){
if(on&&(on.stop(),on=null),yn){try{yn.pause()}catch{}try{yn.src=""}catch{}yn=null}}r(wi,"stopStsPlay\
back");async function Fi(u){wi();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=!0,f.playsInline=
!0,yn=f,await f.play(),new Promise(h=>{f.onended=()=>h("ended"),f.onerror=()=>h("error")})}r(Fi,"pla\
yStsAudio");function Ii(){if(kn.isActive()){kn._cancel();return}if(qe){qe.stop(),qe=null,wi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Xt();return}ut&&ut.state===
"recording"&&(Un=!0,ut.stop())}r(Ii,"cancelRecording");function Nn(){if(isStsModel())return{audio:!0};
const f=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},h={channelCount:1};return f.echoCancellation&&(h.echoCancellation=!1),f.
noiseSuppression&&(h.noiseSuppression=!1),f.autoGainControl&&(h.autoGainControl=!1),{audio:h}}r(Nn,"\
getMicCaptureConstraints");function Pi(){const u=window.AudioContext||window.webkitAudioContext;if(!u||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let f=null;try{f=new u({sampleRate:16e3})}catch{if(f){try{f.close()}catch{}f=null}f=new u}const h=f.
resume(),b=navigator.mediaDevices.getUserMedia(Nn());return{ctx:f,resumePromise:h,streamPromise:b}}r(
Pi,"beginGeminiLiveMic");function Oi(u){if(u){if(u.streamPromise)u.streamPromise.then(f=>{try{f.getTracks().
forEach(h=>h.stop())}catch{}}).catch(()=>{});else if(u.stream)try{u.stream.getTracks().forEach(f=>f.
stop())}catch{}if(u.ctx)try{u.ctx.close()}catch{}}}r(Oi,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(qe){setStsStatus("Processing...",!0);const u=qe;qe=null,u.stop(),get("mic-bt\
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
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),kn.stop();return}if(ut&&
ut.state==="recording"){ut.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Kt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const h=new Audio;
h.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",h.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let h=null;try{h=Pi();
const _={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(_.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const G=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(he=>he.trim()).filter(Boolean);G.length&&(_.custom_vocabulary=
G.slice(0,1e3))}}else _.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(_.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",_.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(_.target_lang=get("sts-target-lang").value);const S=await h.streamPromise;try{await h.
resumePromise}catch{}h={stream:S,ctx:h.ctx,source:h.ctx.createMediaStreamSource(S)};const M=await apiFetch(
"/api/skillini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
_)});if(!M.ok)throw new Error("Failed to get session token");const{token:A,url:I}=await M.json(),B=get(
"model-select").value,D=get("sts-voice")?get("sts-voice").value:"Kore",Q=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",oe=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(qe=new Wn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(qe.rtPlayer=new wn),isGeminiLiveTranscribeModel()){
const G=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",he={languageCodes:[]};
if((G==="SMART"||G==="VERBATIM")&&(he.mode=G),get("sts-custom-vocab")){const X=get("sts-custom-vocab").
value.split(/[,、\n]/).map(we=>we.trim()).filter(Boolean);X.length&&(he.customVocabulary=X.slice(0,
1e3))}await qe.start(A,I,B,{transcriptionConfig:he},h)}else if(isGeminiLiveTranslateModel()){const G=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await qe.start(A,I,B,{translationConfig:{targetLanguageCode:G,
echoTargetLanguage:!0}},h)}else{const G={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:D}}}};
isGeminiLiveExtendedThinkingModel()&&(G.thinkingConfig={thinkingLevel:Q,includeThoughts:oe}),await qe.
start(A,I,B,G,h)}h=null,ut=qe.backupRecorder,ut.onstop=()=>{qe&&get("mic-btn").click()};let de=!0,W="\
live-sts-"+Date.now();qe.onMessage=G=>{if(G.serverContent){if(isGeminiLiveTranscribeModel()){const he=qe.
interimInputTranscript,X=qe.inputTranscript,we=X+(he&&!X.endsWith(he)?(X?`
`:"")+he:""),Se=get("chat-messages");let ae=document.getElementById(W);ae||(ae=document.createElement(
"div"),ae.id=W,ae.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ae.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,Se.appendChild(ae),Se.scrollTop=Se.scrollHeight);const xe=ae.
querySelector(".message-content");xe.innerText=we||"\u8074\u304D\u53D6\u308A\u4E2D...",Se.scrollTop=
Se.scrollHeight,window.VoiceStudio&&X&&window.VoiceStudio.log("user",X);return}if(G.serverContent.modelTurn){
de&&(setStsStatus("Gemini is speaking...",!1),de=!1);const he=get("chat-messages");let X=document.getElementById(
W);X||(X=document.createElement("div"),X.id=W,X.className="flex flex-col gap-2 mb-4 assistant-messag\
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
                                            `,he.appendChild(X),he.scrollTop=he.scrollHeight);const we=X.
querySelector(".thought-container"),Se=X.querySelector(".message-content");qe.assistantThought&&(we.
classList.remove("hidden"),we.innerText=qe.assistantThought),Se.innerText=qe.assistantText,he.scrollTop=
he.scrollHeight,window.VoiceStudio&&(qe.inputTranscript&&window.VoiceStudio.log("user",qe.inputTranscript),
qe.assistantText&&window.VoiceStudio.log("assistant",qe.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),gi(qe.stream),hi(qe.stream);const ve=qe;ve.onError=G=>{const he=G&&G.message?G.message:
typeof G=="string"?G:"";he&&showToast("Gemini Live: "+he,"error",!0)},ve.onClose=G=>{if(qe===ve){if(G&&
G.code&&G.code!==1e3){const he=G.reason?": "+G.reason:" (code "+G.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+he,"error",!0)}get("mic-btn").click()}},ve.closedEvent&&
ve.onClose(ve.closedEvent);return}catch(b){if(Oi(h),qe){try{qe.stop()}catch{}qe=null}showToast("Gemi\
ni Live connection failed: "+b.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await kn.start();return}isStsModel()||(fi(),Kt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const u=await navigator.mediaDevices.getUserMedia(Nn());ut=new MediaRecorder(u),Tn=[],Un=!1;const f=isStsModel();
ut.ondataavailable=h=>Tn.push(h.data),ut.onstop=async()=>{if(Un){Tn=[],get("file-preview").classList.
add("hidden"),u.getTracks().forEach(M=>M.stop()),On(),Xt(),f||(Kt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Pn=setTimeout(()=>Kt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const h=new Blob(Tn,{type:"audio/web\
m"}),b=new File([h],"recording.webm",{type:"audio/webm"}),_=new FormData;_.append("file",b),get("fil\
e-preview").classList.remove("hidden");const S=f;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const G=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=G.id!==null&&G.id!==void 0?String(G.id):G.id,setTemporaryChatUiState(!!(G&&G.
is_temporary)),setCurrentChatHeaderTitle(G&&G.title),applyTemporaryChatRuntimeMeta(G||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+G.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeSkill&&(threadSkillMap[currentThreadId]=activeSkill,pendingSkillForNewThread=null),_.append("m\
odel",get("model-select").value),_.append("thread_id",currentThreadId),get("sts-voice")&&_.append("s\
ts_voice",get("sts-voice").value||""),get("sts-reasoning-effort")&&_.append("sts_reasoning_effort",get(
"sts-reasoning-effort").value||""),get("sts-speed")&&_.append("sts_speed",get("sts-speed").value||""),
get("sts-rate-in")&&_.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&_.append(
"sts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&_.append("sts_thinking_leve\
l",get("sts-thinking-level").value||""),get("sts-include-thoughts")&&_.append("sts_include_thoughts",
get("sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const M=await apiFetch(
"/sts",{method:"POST",body:_});if(!M.ok){const ve=await M.json().catch(()=>({}));throw new Error(ve.
error||"Speech-to-speech failed")}const A=M.body.getReader(),I=new TextDecoder;let B="",D=null,Q=null;
stsOpt("sts-auto-play")&&(Q=new wn,on=Q),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let oe=!0,de="",W="";for(;;){const{done:ve,value:G}=await A.read();if(ve)break;
B+=I.decode(G,{stream:!0});const he=B.split(`
`);B=he.pop();for(const X of he){if(!X.trim())continue;const we=JSON.parse(X);if(we.error)throw new Error(
we.error);we.audio_delta&&Q&&(oe&&(setStsStatus("Playing response...",!1),oe=!1),await Q.addChunk(we.
audio_delta)),we.input_delta&&(de+=we.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
de)),we.transcript_delta&&(W+=we.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",W)),(we.final||we.audio_url)&&(D=we)}}window.VoiceStudio&&!de.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),D&&(D.audio_url||D.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const M=get(
"set-mic-transcribe-mode");if(!!(M&&M.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}_.append("llm_model",get("model-select")&&get("model-select").value||"");const B=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:_})).json();if(B.transcript){const D=get("prompt-inp\
ut");D.value+=(D.value?" ":"")+B.transcript,D.style.height="auto",D.style.height=D.scrollHeight+"px"}else
showToast(B.error||"Transcription failed","error",!0)}}catch(M){showToast("Audio processing error: "+
M.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(M=>M.
stop()),On(),Xt(),S||Kt("","hidden"),S&&setStsStatus("Tap to speak",!1)}},ut.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Kt("\u9332\u97F3\u4E2D\u2026","recording"),gi(u)),hi(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Xt(),isStsModel()||Kt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const xn=r((u,f)=>{if(!u)return;const h=u.querySelector("span");h?h.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),h=get("lib-download-btn"),b=get("lib-attach-btn"),_=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(f&&(f.disabled=u===0,xn(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
h&&(h.disabled=u===0,xn(h,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=u===0,xn(b,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),_&&(_.disabled=u!==1,xn(_,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=u!==1,xn(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const M=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",M&&u>0)}};function ki(u){lib.
attachMode=!!u}r(ki,"setLibAttachMode");const xi=r((u=!1)=>{ki(u),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(u=!1)=>{hideModal("lib-modal"),!u&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>xi(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>xi(!0)),
get("lib-search")){let u=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),u&&clearTimeout(u),u=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const u=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=u,get("lib-sort").onchange=()=>{const f=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,f),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0));const Ni=r(()=>{editingSkillUuid=null,get("skill-modal-title").textContent=
"Create New Skill",get("save-skill-btn").innerText="Create Skill",showModal("skill-modal"),get("skil\
l-name").value="",get("skill-desc").value="",get("skill-inst").value="",location.pathname!=="/skill"&&
history.pushState({modal:"skill"},"","/skill")},"openSkillModal");window.closeSkillModal=(u=!1)=>{hideModal(
"skill-modal"),!u&&location.pathname==="/skill"&&history.back()},get("add-skill-btn").onclick=()=>Ni(),
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
/plain"}),M=URL.createObjectURL(S),A=document.createElement("a");A.href=M;let B={python:"py",javascript:"\
js",typescript:"ts",markdown:"md",html:"html",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"\
sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"\
go",rust:"rs",ruby:"rb",php:"php",perl:"pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"\
toml",ini:"ini",plaintext:"txt",text:"txt"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(B="txt");let D=`\
code.${B}`;b==="dockerfile"&&(D="Dockerfile"),b==="makefile"&&(D="Makefile"),A.download=D,document.body.
appendChild(A),A.click(),document.body.removeChild(A),URL.revokeObjectURL(M)}catch(_){console.error(
"Download failed",_)}}if(u.target.closest(".coding-target-btn")&&selectCodingTargetFromButton(u.target.
closest(".coding-target-btn")),u.target.closest(".copy-btn")){const f=u.target.closest(".copy-btn"),
h=f.getAttribute("data-code");h&&window.copyCode(f,h)}if(u.target.closest(".html-preview-btn")){const h=u.
target.closest(".html-preview-btn").getAttribute("data-code");h&&openHtmlCodePreview(h)}if(u.target.
closest(".canvas-preview-btn")){const f=u.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(
f)}}),document.querySelectorAll(".modal-overlay").forEach(u=>{u.addEventListener("click",f=>{f.target===
u&&ii(u.id)})}),currentThreadId?loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});function updateFilePreview(){
const e=get("file-preview"),t=get("file-name"),n=get("upload-total-progress"),i=get("upload-total-pr\
ogress-bar"),a=get("file-preview-thumbs"),o=get("upload-modal-status-text"),l=get("upload-modal-tota\
l-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;if(a){const L=document.querySelectorAll(
"#upload-list .upload-row");a.innerHTML="",L.forEach((T,E)=>{const H=T.getAttribute("data-local-url"),
N=T.getAttribute("data-filename"),Y=T.querySelector("img.upload-preview")!==null;let O;if(Y){let V=H;
if(!V&&N){const ue=N.replace(/^\d+\//,"");V=buildAttachmentPreviewUrl(ue)}V&&(O=document.createElement(
"img"),O.src=V,O.className="thumb-item shadow-sm",O.dataset.viewerSrc=V,O.dataset.viewerFilename=N||
V.split("/").pop(),O.uploadRow=T,O.onclick=function(ue){ue.preventDefault(),openImageViewer(this.dataset.
viewerSrc,".thumb-item")},O.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}O||(O=
d("FILE")),O.style.animationDelay=`${E*32}ms`,a.appendChild(O)}),L.length>0?a.classList.remove("hidd\
en"):a.classList.add("hidden")}function d(L){const T=document.createElement("div");return T.className=
"thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bol\
d",T.innerText=L,T}r(d,"createFileThumb");const m=collectImageUrlsForSend(),g=uploadProgressState.total,
y=uploadProgressState.completed,v=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.
classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const w=get("send\
-btn"),k=get("mic-btn"),x=get("mask-btn"),C=isStopMode;if(v>0?(w&&(w.disabled=!0),k&&(k.disabled=!0),
x&&(x.disabled=!0)):C||(w&&(w.disabled=!1),k&&(k.disabled=!1),x&&(x.disabled=!1)),v>0){const L=`Prep\
aring... (${y}/${g})`;e.classList.remove("hidden"),t.innerText=L,o&&(o.innerText=`(${y}/${g})`);let T=y*
100,E=0;for(let Y in uploadProgressState.perFilePct)T+=uploadProgressState.perFilePct[Y],E++;const H=g>
0?T/(g*100)*100:0,N=`${Math.min(100,H)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=N),l&&c&&
(l.classList.remove("hidden"),c.style.width=N)}else o&&(o.innerText=""),l&&l.classList.add("hidden"),
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
scription"),k=get("marker-save");v&&(v.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u4F5C\u6210":
"\u753B\u50CF\u7DE8\u96C6"),w&&(w.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\
\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":
"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
k&&(k.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const x=get("marker-canvas");x&&(x.style.pointerEvents=e==="crop"?"none":"auto");const C=get("marker\
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
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=g-l-d,w=n-g-l,k=y-c-m,x=i-y-c,C=r((L,T,E)=>Number.
isFinite(L)?T>E?(T+E)/2:Math.min(E,Math.max(T,L)):0,"clampOffset");markerView.offsetX=C(markerView.offsetX,
v,w),markerView.offsetY=C(markerView.offsetY,k,x)}r(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
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
\u753B\u50CF\u7DE8\u96C6</button>':"",k=c?`<img src="${m}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${k}
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
            `;const x=y.querySelector(".upload-preview");x&&(x.onclick=()=>openFileViewer(d,getRowAttachmentName(
y)||o));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const L=y.
querySelector(".upload-remove");L&&(L.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const E=y.getAttribute("data-filename");E&&(currentImageUrls=currentImageUrls.
filter(H=>H!==E)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const T=y.querySelector(".upload-marker");return T&&(T.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
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
onclick=()=>{const k=i.getAttribute("data-filename"),x=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(x,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const x=i.
getAttribute("data-filename");x&&(currentImageUrls=currentImageUrls.filter(C=>C!==x)),setRowMarkerState(
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
v+l),k=e.slice(v,w);if(!await new Promise(C=>{const L=new XMLHttpRequest;L.open("POST","/upload/chun\
k",!0),L.setRequestHeader("X-CSRF-Token",csrfToken),L.upload.onprogress=E=>{if(E.lengthComputable&&t&&
t.bar){const H=v+E.loaded,N=Math.min(100,Math.floor(H/e.size*100));t.bar.style.width=`${N}%`,t.status&&
(t.status.textContent=`${N}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,N)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},L.onload=()=>{L.status>=200&&L.status<300?C(!0):C(!1)},L.
onerror=()=>C(!1);const T=new FormData;T.append("upload_id",o),T.append("index",String(y)),T.append(
"total",String(c)),T.append("chunk",k,e.name),L.send(T)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:o})}),m=await d.json();if(d.ok&&m&&m.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const k=t.row.querySelector(
"img.upload-preview");if(k){const x=m.filename.replace(/^\d+\//,"");k.src=buildAttachmentPreviewUrl(
x)}}const y=normalizeAttachmentPath(m.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||m.filename),setRowAttachmentSource(t.row,"upload"),y)){const w=isRowAttachmentNameCustomized(
t.row),k=defaultAttachmentDisplayName(y),x=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,x)}return y&&setAttachmentSourceForPath(
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
".upload-preview");y&&(y.onclick=()=>{const w=e.row.getAttribute("data-filename"),k=w?buildFileUrl(w):
e.row.getAttribute("data-local-url");openFileViewer(k,getRowAttachmentName(e.row)||a||w||"")});const v=e.
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
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},m=[],g=16,y="",v=null,w=null,k=null,x=null,C=null,L=!1,T=null;
const E=r(P=>{const J=e.getBoundingClientRect(),fe=(P.clientX-J.left)*(e.width/J.width),re=(P.clientY-
J.top)*(e.height/J.height);return{x:fe,y:re}},"getPoint"),H=r((P,J)=>({x:(P.x+J.x)/2,y:(P.y+J.y)/2}),
"getMid"),N=r((P,J)=>Math.hypot(P.x-J.x,P.y-J.y),"getDist");let Y=!1;const O=r(()=>{w||(w=document.createElement(
"canvas"),k=w.getContext("2d")),x||(x=document.createElement("canvas"),C=x.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(x.width!==e.width||x.height!==e.
height)&&(x.width=e.width,x.height=e.height)},"ensureDrawBuffers"),V=r(()=>{if(!t||!w||!x)return;const P=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=P,t.drawImage(x,0,0),t.restore()},"r\
enderDrawPreview"),ue=r(()=>{C&&(C.strokeStyle=y,C.fillStyle=y,C.lineWidth=g,C.lineCap="round",C.lineJoin=
"round")},"applyMarkerBrush"),ge=r(P=>{if(!P)return!1;if(m.length===0)return m.push(P),!0;const J=m[m.
length-1],fe=P.x-J.x,re=P.y-J.y,Z=Math.hypot(fe,re),le=Math.max(.35,g*.04);if(Z<le)return!1;const ee=Math.
max(1,g*.25),ce=Math.max(1,Math.ceil(Z/ee));for(let me=1;me<=ce;me++){const De=me/ce;m.push({x:J.x+fe*
De,y:J.y+re*De})}return!0},"appendStrokePoint"),ne=r(()=>{if(C&&(C.clearRect(0,0,x.width,x.height),m.
length!==0)){if(ue(),m.length===1){const P=m[0];C.beginPath(),C.arc(P.x,P.y,g/2,0,Math.PI*2),C.fill();
return}if(C.beginPath(),C.moveTo(m[0].x,m[0].y),m.length===2)C.lineTo(m[1].x,m[1].y);else{for(let fe=1;fe<
m.length-2;fe++){const re=m[fe],Z=m[fe+1],le=H(re,Z);C.quadraticCurveTo(re.x,re.y,le.x,le.y)}const P=m[m.
length-2],J=m[m.length-1];C.quadraticCurveTo(P.x,P.y,J.x,J.y)}C.stroke()}},"renderStrokeLayer"),ke=r(
(P,J)=>{if(!P||!J)return null;const fe=Math.min(P.x,J.x),re=Math.min(P.y,J.y),Z=Math.abs(P.x-J.x),le=Math.
abs(P.y-J.y);return{x:fe,y:re,w:Z,h:le}},"normalizeMosaicRect"),$=r(P=>{const J=n?Number(n.value||16):
16,fe=Math.max(6,Math.floor(J)),re=Math.floor(fe/2);return{x:P.x-re,y:P.y-re,w:fe,h:fe}},"buildMosai\
cRectFromPoint"),R=r(()=>{const P=document.createElement("canvas");P.width=e.width,P.height=e.height;
const J=P.getContext("2d");if(!J)return null;markerState.baseCanvas&&J.drawImage(markerState.baseCanvas,
0,0),J.drawImage(e,0,0);try{return J.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),U=r(P=>{if(!t||!P)return!1;const J=R();if(!J)return!1;const fe=n?Number(n.value||
16):16,re=Math.max(4,Math.floor(fe/2)),Z=Math.max(0,Math.floor(P.x)),le=Math.max(0,Math.floor(P.y)),
ee=Math.min(e.width,Math.ceil(P.x+P.w)),ce=Math.min(e.height,Math.ceil(P.y+P.h));if(ee<=Z||ce<=le)return!1;
for(let me=le;me<ce;me+=re)for(let De=Z;De<ee;De+=re){const Re=Math.min(re,ee-De),Ke=Math.min(re,ce-
me),Ve=Math.min(e.width-1,Math.max(0,De+Math.floor(Re/2))),se=(Math.min(e.height-1,Math.max(0,me+Math.
floor(Ke/2)))*e.width+Ve)*4,He=J.data[se],ie=J.data[se+1],ot=J.data[se+2];t.fillStyle=`rgb(${He},${ie}\
,${ot})`,t.fillRect(De,me,Re,Ke)}return!0},"applyMosaicRect"),j=r(P=>{if(!t)return;if(i.set(P.pointerId,
{x:P.clientX,y:P.clientY}),i.size>=2){const fe=Array.from(i.values()),re=fe[0],Z=fe[1];a=!0,Y=!1,m=[],
L=!1,T=null,v=null,markerState.mosaicPreviewRect=null,o=N(re,Z)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=H(re,Z),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
P.pointerId),P.preventDefault();return}if(a||markerState.mode==="crop")return;Y=!0;const J=E(P);if(markerState.
mode==="auto"&&markerState.maskMode)v=J;else if(markerState.mode==="mosaic")L=!0,T=J,markerState.mosaicPreviewRect=
$(J),renderCropOverlay();else{if(O(),!k||!C)return;k.clearRect(0,0,w.width,w.height),k.drawImage(e,0,
0),C.clearRect(0,0,x.width,x.height),g=n?Number(n.value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),m=[],ge(J),ne(),markerState.hasStroke=!0,V()}e.setPointerCapture&&e.setPointerCapture(
P.pointerId),P.preventDefault()},"start"),F=r(P=>{if(i.has(P.pointerId)&&i.set(P.pointerId,{x:P.clientX,
y:P.clientY}),a&&i.size>=2){const fe=Array.from(i.values()),re=fe[0],Z=fe[1],le=H(re,Z),ee=N(re,Z)||
1,ce=l*(ee/o);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,ce)),markerView.
offsetX=c.x+(le.x-d.x),markerView.offsetY=c.y+(le.y-d.y),applyMarkerTransform(),P.preventDefault();return}
if(!Y||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){P.preventDefault();return}const J=E(
P);if(markerState.mode==="mosaic"){if(!L||!T)return;markerState.mosaicPreviewRect=ke(T,J)||$(J),renderCropOverlay()}else
ge(J)&&(ne(),V());P.preventDefault()},"move"),pe=r(P=>{const J=Y;if(i.delete(P.pointerId),i.size<2&&
(a=!1),i.size===0){Y=!1;const fe=J&&P.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.
maskMode&&v?fillMaskSimilarRegion(v):!1;if(v=null,J&&t&&markerState.mode==="draw"&&m.length>0&&(ne(),
V()),J&&markerState.mode==="mosaic"&&T){const re=E(P);let Z=ke(T,re);(!Z||Z.w<2||Z.h<2)&&(Z=$(T)),U(
Z)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(Z))}m=[],L=!1,T=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),J&&(markerState.mode!=="auto"||fe)&&saveMarkerHistory()}e.releasePointerCapture&&
e.releasePointerCapture(P.pointerId),P.preventDefault()},"end");e.addEventListener("pointerdown",j),
e.addEventListener("pointermove",F),e.addEventListener("pointerup",pe),e.addEventListener("pointerca\
ncel",pe)}r(initMarkerCanvas,"initMarkerCanvas");function fillMaskSimilarRegion(e){var Y;const t=get(
"marker-canvas"),n=markerState.baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,o=Math.floor(e.x),l=Math.floor(e.y);if(o<0||l<0||o>=i||l>=
a)return!1;const c=n.data,d=l*i+o,m=d*4;if(c[m+3]<16)return!1;const g=c[m],y=c[m+1],v=c[m+2],w=Number(
((Y=get("marker-auto-tolerance"))==null?void 0:Y.value)||32),k=new Uint8Array(i*a),x=[d],C=[],L=25e4,
T=r(O=>{const V=O*4;return c[V+3]>=16&&Math.max(Math.abs(c[V]-g),Math.abs(c[V+1]-y),Math.abs(c[V+2]-
v))<=w},"matches");for(k[d]=1;x.length;){const O=x.pop();if(k[O]===2||!T(O))continue;const V=Math.floor(
O/i);let ue=O%i,ge=ue;for(;ue>0&&k[V*i+ue-1]!==2&&T(V*i+ue-1);)ue--;for(;ge+1<i&&k[V*i+ge+1]!==2&&T(
V*i+ge+1);)ge++;for(let ne=ue;ne<=ge;ne++)k[V*i+ne]=2;if(C.push(V,ue,ge+1),C.length/3>L)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const ne of[V-1,V+1]){if(ne<0||ne>=a)continue;let ke=ue;for(;ke<=ge;){const $=ne*
i+ke;if(k[$]===0&&T($))for(k[$]=1,x.push($);ke+1<=ge&&k[ne*i+ke+1]===0&&T(ne*i+ke+1);)ke++,k[ne*i+ke]=
1;ke++}}}if(!C.length)return!1;const E=t.getContext("2d");if(!E)return!1;const H=E.getImageData(0,0,
i,a),N=H.data;for(let O=0;O<C.length;O+=3){const V=C[O]*i+C[O+1],ue=C[O]*i+C[O+2];for(let ge=V;ge<ue;ge++){
const ne=ge*4;N[ne]=N[ne+1]=N[ne+2]=N[ne+3]=255}}return E.putImageData(H,0,0),markerState.hasStroke=
!0,!0}r(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,o=null,l=null,c=!1,
d=0,m=markerView.scale,g={x:0,y:0},y={x:0,y:0};const v=8,w=14,k=r((O,V,ue)=>Math.min(ue,Math.max(V,O)),
"clamp"),x=r(O=>{const V=e.getBoundingClientRect(),ue=(O.clientX-V.left)*(e.width/V.width),ge=(O.clientY-
V.top)*(e.height/V.height);return{x:ue,y:ge}},"getPoint"),C=r((O,V)=>({x:(O.x+V.x)/2,y:(O.y+V.y)/2}),
"getMid"),L=r((O,V)=>Math.hypot(O.x-V.x,O.y-V.y),"getDist"),T=r(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),E=r((O,V)=>{if(!V)return"move";const ue=V.x,ge=V.y,ne=V.x+V.
w,ke=V.y+V.h,$=Math.abs(O.x-ue)<=w,R=Math.abs(O.x-ne)<=w,U=Math.abs(O.y-ge)<=w,j=Math.abs(O.y-ke)<=w;
if($&&U)return"nw";if(R&&U)return"ne";if($&&j)return"sw";if(R&&j)return"se";if(U)return"n";if(j)return"\
s";if($)return"w";if(R)return"e";if(O.x>ue+w&&O.x<ne-w&&O.y>ge+w&&O.y<ke-w)return"move";const pe=O.x<
ue?"left":O.x>ne?"right":null,P=O.y<ge?"top":O.y>ke?"bottom":null;if(pe&&P){if(pe==="left"&&P==="top")
return"nw";if(pe==="right"&&P==="top")return"ne";if(pe==="left"&&P==="bottom")return"sw";if(pe==="ri\
ght"&&P==="bottom")return"se"}return pe?pe==="left"?"w":"e":P?P==="top"?"n":"s":"move"},"hitTest"),H=r(
O=>{if(markerState.mode!=="crop")return;if(n.set(O.pointerId,{x:O.clientX,y:O.clientY}),n.size>=2){const ge=Array.
from(n.values()),ne=ge[0],ke=ge[1];c=!0,i=!1,d=L(ne,ke)||1,m=markerView.scale,g={x:markerView.offsetX,
y:markerView.offsetY},y=C(ne,ke),e.setPointerCapture&&e.setPointerCapture(O.pointerId),O.preventDefault();
return}if(c)return;i=!0;const V=x(O),ue=T();o=E(V,ue),a=V,l=ue?{x:ue.x,y:ue.y,w:ue.w,h:ue.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(O.pointerId),O.preventDefault()},"start"),N=r(O=>{if(markerState.
mode!=="crop")return;if(n.has(O.pointerId)&&n.set(O.pointerId,{x:O.clientX,y:O.clientY}),c&&n.size>=
2){const pe=Array.from(n.values()),P=pe[0],J=pe[1],fe=C(P,J),re=L(P,J)||1,Z=m*(re/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Z)),markerView.offsetX=g.x+(fe.x-y.x),markerView.
offsetY=g.y+(fe.y-y.y),applyMarkerTransform(),renderCropOverlay(),O.preventDefault();return}if(!i||!a||
!l)return;const V=x(O),ue=e.width,ge=e.height,ne={x:l.x,y:l.y,w:l.w,h:l.h},ke=l.x+l.w,$=l.y+l.h,R=r(
()=>{const pe=k(V.x,0,ke-v);ne.x=pe,ne.w=ke-pe},"applyW"),U=r(()=>{ne.w=k(V.x-l.x,v,ue-l.x)},"applyE"),
j=r(()=>{const pe=k(V.y,0,$-v);ne.y=pe,ne.h=$-pe},"applyN"),F=r(()=>{ne.h=k(V.y-l.y,v,ge-l.y)},"appl\
yS");switch(o){case"move":{const pe=V.x-a.x,P=V.y-a.y;ne.x=k(l.x+pe,0,ue-l.w),ne.y=k(l.y+P,0,ge-l.h);
break}case"w":R();break;case"e":U();break;case"n":j();break;case"s":F();break;case"nw":j(),R();break;case"\
ne":j(),U();break;case"sw":F(),R();break;case"se":F(),U();break;default:break}ne.x=k(ne.x,0,ue-ne.w),
ne.y=k(ne.y,0,ge-ne.h),markerState.cropRect=ne,renderCropOverlay(),O.preventDefault()},"move"),Y=r(O=>{
n.delete(O.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,o=null,l=null),e.
releasePointerCapture&&e.releasePointerCapture(O.pointerId),O.preventDefault()},"end");e.addEventListener(
"pointerdown",H),e.addEventListener("pointermove",N),e.addEventListener("pointerup",Y),e.addEventListener(
"pointercancel",Y),e.addEventListener("pointerleave",Y)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const o=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const C=document.createElement("canvas");C.width=o,C.height=l;const L=C.getContext(
"2d");if(!L)return;L.fillStyle="#000000",L.fillRect(0,0,o,l),L.drawImage(n,0,0,o,l);const T=await new Promise(
N=>C.toBlob(N,"image/png"));if(!T){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const E=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([T],`${E}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=o,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,o,l),c.drawImage(n,0,0,o,l),markerState.cropRect){const C=o/n.width,L=l/n.height,T=Math.max(0,Math.
floor(markerState.cropRect.x*C)),E=Math.max(0,Math.floor(markerState.cropRect.y*L)),H=Math.min(o,Math.
max(1,Math.floor(markerState.cropRect.w*C))),N=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*L))),Y=document.createElement("canvas");Y.width=H,Y.height=N;const O=Y.getContext("2d");O&&(O.drawImage(
a,T,E,H,N,0,0,H,N),a=Y)}const d=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const g=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([d],`${g}_marked.png`,
{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),k=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=k,setAttachmentSourceForPath(
w,k)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
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
k=>{const x=k.inputBuffer.getChannelData(0);d.push(new Float32Array(x))},c.connect(l),l.connect(o.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(m=setTimeout(()=>{const k=(e.
name||"video").replace(/\.[^/.]+$/,""),x=encodeWav(d,o.sampleRate),C=new File([x],`${k}.audio.wav`,{
type:"audio/wav"});g(),n(C)},w+250)),await i.play(),i.onended=()=>{const k=(e.name||"video").replace(
/\.[^/.]+$/,""),x=encodeWav(d,o.sampleRate),C=new File([x],`${k}.audio.wav`,{type:"audio/wav"});g(),
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
active--,!1;const k=addUploadRow(v);k.generation=a,updateFilePreview(),w=k.uploadId,uploadProgressState.
perFilePct[w]=0;let x=v;if(o&&isVideoFile(v)){const L=await extractAudioFromVideo(v,k);L?(x=L,updateUploadRowFile(
k,L),k&&k.status&&(k.status.textContent="\u97F3\u58F0\u306E\u307F")):(k&&k.status&&(k.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{x=await applyImageCompression(v,
readComposerCompression()),x!==v&&updateUploadRowFile(k,x)}catch{}if(isUploadCancelled(k))return!1;if(browserFastModeEnabled){
const L=Array.from(browserFastLocalFiles.values()).reduce((T,E)=>T+Number(E.file&&E.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||L+x.size>BROWSER_FAST_MAX_BYTES?(k&&k.status&&
(k.status.textContent="\u4E0A\u9650\u8D85\u904E"),k&&k.row&&k.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(k.uploadId,{file:x,rowObj:k}),
k.status&&(k.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
k.bar&&(k.bar.style.width="100%"),k.row&&(k.row.dataset.browserFastLocal="1"),!0)}const C=await uploadFileWithProgress(
x,k,t);if(!C&&t.cameraCapture&&!isUploadCancelled(k)){const L=document.createElement("button");L.className=
"text-xs border border-gray-700 rounded px-2 py-1",L.textContent="\u518D\u8A66\u884C",L.onclick=()=>{
if(isUploadCancelled(k))return;L.disabled=!0;const T=k.row.getAttribute("data-local-url");T&&URL.revokeObjectURL(
T),k.row.remove(),handleFiles([x],{...t,openModal:!0,skipCompression:!0})},k.row.appendChild(L)}return C}finally{
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
