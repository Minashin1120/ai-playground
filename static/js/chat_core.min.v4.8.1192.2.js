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
getBoundingClientRect(),h=u.getBoundingClientRect(),v=h.left+h.width/2-(f.left+f.width/2),_=h.top+h.
height/2-(f.top+f.height/2);c.style.transform=`translate(${v}px, ${_}px) scale(0.1)`,c.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),c.remove()},800)}else c.style.opacity="0",setTimeout(()=>c.remove(),1e3)}},3e3);function d(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const h=(get("model-select").value||"").toLowerCase().includes("gpt-image-2.6"),v=get("gpt-im\
age-format");get("gpt-image-complexity-wrap")&&get("gpt-image-complexity-wrap").classList.toggle("hi\
dden",!h),v&&(Array.from(v.options).forEach(E=>{E.hasAttribute("data-gpt-image-26")&&(E.hidden=!h)}),
!h&&v.value==="gif"&&(v.value="jpeg"));const _=h&&v&&v.value==="gif",S=get("gpt-image-compression-wr\
ap");v&&S&&(v.value==="png"||v.value==="gif"?S.classList.add("hidden"):S.classList.remove("hidden")),
get("gpt-image-gif-wrap")&&get("gpt-image-gif-wrap").classList.toggle("hidden",!_)}r(d,"updateGptIma\
geUi");function m(){const u=get("gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.
remove("hidden"):u.classList.add("hidden");const h=(get("model-select").value||"").toLowerCase().includes(
"gemini-3.1-flash-lite-image");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(v=>{
v&&(Array.from(v.options).forEach(_=>{_.value!=="1K"&&(_.disabled=h)}),h&&v.value!=="1K"&&(v.value="\
1K"))})}r(m,"updateGeminiImageUi");function g(){const u=get("grok-image-options");if(!u)return;const f=(get(
"model-select").value||"").toLowerCase(),h=isGrokImageModel(),v=f==="grok-imagine-image-quality"||f===
"grok-imagine-image-2.0",_=f==="grok-imagine-image-2.0";if(h){u.classList.remove("hidden");const E=get(
"grok-image-resolution")?get("grok-image-resolution").parentElement:null;E&&E.classList.toggle("hidd\
en",!v);const I=get("grok-image-quality")?get("grok-image-quality").parentElement:null;I&&I.classList.
toggle("hidden",!_)}else u.classList.add("hidden");if(get("modal-grok-image-options")){const E=get("\
modal-grok-image-resolution")?get("modal-grok-image-resolution").parentElement:null;E&&E.classList.toggle(
"hidden",!v);const I=get("modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;
I&&I.classList.toggle("hidden",!_)}}r(g,"updateGrokImageUi");function y(){var se,de,me;const u=get("\
ideogram-image-options"),f=get("modal-ideogram-image-options"),h=get("model-select")&&get("model-sel\
ect").value||"",v=isIdeogramModel(h),_=ideogramModelTraits(h),S={resolution:_.resolution,quality:_.quality,
speed:_.speed,style:_.style,negative:_.negative};if(u&&u.classList.toggle("hidden",!v),f&&f.classList.
toggle("hidden",!v),updateIdeogramPromptInputUi(),!v)return;const E=((se=get("ideogram-image-operati\
on"))==null?void 0:se.value)==="remove_background",I=((de=get("ideogram-image-operation"))==null?void 0:
de.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(Z=>{const he=Z==="operation"||I&&Z==="seed"||
!(E||I)&&S[Z]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(ae=>{const Ee=get(ae+Z);!Ee||
!Ee.parentElement||Ee.parentElement.classList.toggle("hidden",!he)})}),["ideogram-image-style","moda\
l-ideogram-image-style"].forEach(Z=>{const he=get(Z);if(!he)return;const ae=_.styleOptions.join("|");
if(he.dataset.options!==ae){const Ee=he.value;he.innerHTML=_.styleOptions.map(le=>`<option value="${le}\
">${le==="auto"?"Auto":le}</option>`).join(""),he.dataset.options=ae,he.value=_.styleOptions.includes(
Ee)?Ee:"auto"}});const j=String(((me=get("ideogram-image-quality"))==null?void 0:me.value)||"").toLowerCase(),
U=!!_.resolution4k&&j!=="very_low";["ideogram-image-resolution","modal-ideogram-image-resolution"].forEach(
Z=>{const he=get(Z);if(!he)return;const ae=Array.from(he.options||[]).find(Ee=>Ee.value==="4k");ae&&
(ae.disabled=!U),!U&&he.value==="4k"&&(he.value="2k")});const W=get("ideogram-image-note");W&&(W.textContent=
E?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
I?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
h.toLowerCase()==="ideogram-4.6"?"\u203B Ideogram 4.6: 4K\u51FA\u529B\u5BFE\u5FDC\uFF08Very low\u306F2K\u307E\u3067\uFF09\u3002\u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precis\
e Edit\uFF08\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA\uFF09":_.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\
\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.6 / 4.5 \u306E\u307F\uFF09")}
r(y,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation","ideogram-\
image-quality","modal-ideogram-image-quality"].forEach(u=>{var f;(f=get(u))==null||f.addEventListener(
"change",()=>{const h=u.startsWith("modal-")?u.slice(6):`modal-${u}`,v=get(h);v&&(v.value=get(u).value),
y(),u.endsWith("operation")&&A()})});function b(){var v;const u=get("grok-video-options");if(!u)return;
const f=String(((v=get("model-select"))==null?void 0:v.value)||"").toLowerCase();isGrokVideoModel()?
u.classList.remove("hidden"):u.classList.add("hidden");const h=get("grok-video-resolution");if(h){const _=Array.
from(h.options).find(E=>E.value==="1080p"),S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].
includes(f);_&&(_.disabled=!S),!S&&h.value==="1080p"&&(h.value="720p")}}r(b,"updateGrokVideoUi");function w(){
var _;const u=get("gemini-video-options");if(!u)return;const f=String(((_=get("model-select"))==null?
void 0:_.value)||"").toLowerCase();isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add(
"hidden");const h=get("gemini-video-resolution");if(h){const S=Array.from(h.options).find(I=>I.value===
"4K"),E=f==="veo-3.1-lite-generate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-f\
lash";S&&(S.disabled=E),E&&h.value==="4K"&&(h.value="1080p")}const v=get("gemini-video-duration-wrap");
v&&v.classList.toggle("hidden",f==="gemini-omni-1.1-flash")}r(w,"updateGeminiVideoUi");function x(){
const u=get("gemini-music-options");if(!u)return;const f=isGeminiRealtimeMusicModel(),h=isGeminiMusicModel()&&
!f;u.classList.toggle("hidden",!h);const v=get("lyria-realtime-studio-bar");v&&v.classList.toggle("h\
idden",!f)}r(x,"updateGeminiMusicUi");function k(){var E;const u=get("xai-chat-options");if(!u)return;
const f=String(((E=get("model-select"))==null?void 0:E.value)||"").toLowerCase(),h=f.startsWith("gro\
k-")&&!isGrokImageModel(f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!h);
const v=get("xai-logprobs"),_=get("xai-top-logprobs"),S=f.includes("grok-4.20");v&&(v.disabled=S,S&&
(v.checked=!1)),_&&(_.disabled=S,S&&(_.value=""))}r(k,"updateXaiChatUi");function C(){const u=isMistralOcrModel(),
f=get("mistral-ocr-options");f&&f.classList.toggle("hidden",!u);const h=get("modal-mistral-ocr-optio\
ns");h&&h.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fa\
st-mode-container"].forEach(v=>{const _=get(v);_&&(_.classList.toggle("opacity-50",u),_.classList.toggle(
"pointer-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(C,"updateMistralOcrUi");function T(){const u=get("image-input-limit\
s");if(!u)return;const f=(get("model-select").value||"").toLowerCase();let h="",v=!1;if(f.includes("\
gpt-image"))v=!0,h=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("");else if(f==="deepseek-v4.1-flash"||f==="deepseek-v4-flash-vision-exp")v=!0,h=['<div class="\
font-bold text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\
\u30FBWebP / \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("");else if(!f.includes("deepseek"))if(isGeminiImageModelKey(f))v=!0,f.includes("gemini-3.1-fla\
sh-lite-image")?h=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
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
join("");else if(isMistralOcrModel(f))v=!0,h=['<div class="font-bold text-gray-300 mb-1">Mistral OCR\
 4 \u5165\u529B</div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("");else if(f.includes("grok"))v=!0,h=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</\
div>',"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("");else if(isIdeogramModel(f)){v=!0;const _=f.toLowerCase()==="ideogram-4.6"?"Ideogram 4.6 \u5165\u529B\u5236\
\u9650":"Ideogram 4.5 \u5165\u529B\u5236\u9650";h=ideogramModelTraits(f).edit?[`<div class="font-bol\
d text-gray-300 mb-1">${_}</div>`,"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP</div>",
"<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u3082\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.6 / 4.5 \u306E\u307F\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u306F\u5168\u30E2\u30C7\u30EB\u3067\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join("")}else f.includes("grok")&&f.includes("video")&&(v=!0,h=['<div class="font-bold text-gray-300\
 mb-1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 48\
0p</div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""));v?(u.innerHTML=h,u.classList.remove("hidden")):(u.classList.add("hidden"),u.innerHTML="")}
r(T,"updateImageInputLimits");function A(){const u=get("enable-sys-prompt"),f=!!(u&&u.disabled),h=!!(u&&
u.checked);G(),u&&(u.disabled?!f&&h?u.dataset.restoreChecked="1":f||delete u.dataset.restoreChecked:
(f&&u.dataset.restoreChecked==="1"&&(u.checked=!0),delete u.dataset.restoreChecked))}r(A,"toggleOpti\
ons"),window.toggleOptions=A;function N(u){const f=get("thinking-options"),h=get("thinking-level"),v=get(
"reasoning-effort"),_=get("enable-thinking"),S=get("thinking-budget-label"),E=get("thinking-budget"),
I=u==="effort";if(f&&(f.dataset.thinkingMode=I?"effort":"thinking"),h&&h.classList.toggle("hidden",I),
v&&v.classList.toggle("hidden",!I),S&&S.classList.toggle("hidden",I),E&&E.classList.toggle("hidden",
I),!!_)if(I){_.dataset.thinkingChecked===void 0&&(_.dataset.thinkingChecked=_.checked?"1":"0");const j=v?
Array.from(v.options).find(W=>W.value==="none"):null,U=!!(j&&!j.disabled&&!j.classList.contains("hid\
den"));_.checked=U?!!(v&&String(v.value||"").toLowerCase()!=="none"):!0,_.disabled=!U}else _.dataset.
thinkingChecked!==void 0&&(_.checked=_.dataset.thinkingChecked==="1",delete _.dataset.thinkingChecked)}
r(N,"setComposerThinkingMode");function G(){const u=get("model-select");if(!u)return;const f=u.value,
h=String(f||"").toLowerCase(),v=h.includes("deepseek"),_=h.startsWith("glm-"),S=get("thinking-option\
s"),E=get("reasoning-effort-container"),I=get("enable-thinking"),j=get("thinking-level"),U=get("thin\
king-budget"),W=get("enable-search"),se=get("search-container"),de=get("url-context-container"),me=get(
"enable-maps"),Z=get("maps-grounding-container"),he=get("enable-sys-prompt"),ae=get("sys-prompt-opti\
on"),Ee=get("enable-python"),le=get("python-container"),Te=get("prompt-cache-container"),je=get("ena\
ble-prompt-cache"),pe=f==="gpt-5-search-api",_e=f.includes("tts"),Fe=isMistralOcrModel(f),De=h.includes(
"gemini-3.1-flash-lite-image"),_t=h==="gemini-nano-banana-2.1",ct=h.includes("gemini-3.1-flash-image")&&
!De||_t,Pt=isClaudeModelKey(f),Xt=h==="gemini-3.8-flash-cyber",Pe=isLlmModel()&&!v&&!_e&&!h.includes(
"realtime")&&!h.includes("native-audio")&&!h.includes("live");Te&&(Pe?(Te.classList.remove("hidden",
"opacity-50","pointer-events-none"),je&&(je.disabled=!1)):(je&&(je.checked=!1,je.disabled=!0),Te.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),E&&E.classList.
add("hidden"),N("thinking");const Me=get("vision-model-info");if(Me&&Me.classList.add("hidden"),E){const Ae=get(
"reasoning-effort");if(Ae){Array.from(Ae.options).forEach(ze=>{const Qe=h==="gpt-5.6"||h.startsWith(
"gpt-5.6-"),on=h.startsWith("gpt-6"),Sn=h==="gpt-6-sol"||h==="gpt-6-luna"||h==="gpt-6.1-terra",On=h===
"deepseek-v4.1-flash"||h==="deepseek-v4-flash-0731"||h==="deepseek-v4-flash"||h==="deepseek-v4-flash\
-vision-exp",Cn=h==="deepseek-v4-pro",K=h.includes("grok-4.5"),oe=h.includes("grok-4.6");ze.value===
"max"?ze.classList.toggle("hidden",!Qe&&!on&&!On&&!Cn):ze.value==="xhigh"?ze.classList.toggle("hidde\
n",!oe&&!h.includes("multi-agent")&&!Qe&&!on):ze.value==="medium"?ze.classList.toggle("hidden",!(h.includes(
"grok-4.3")||K||oe||h.includes("grok-4.20-0309-reasoning")||h.includes("grok-build")||h.includes("mu\
lti-agent")||h.includes("gpt-5")||on||h.includes("o1")||h.includes("o3"))):ze.value==="none"?ze.classList.
toggle("hidden",!h.includes("grok-4.3")&&!h.includes("grok-build")&&!h.includes("gpt-5")&&!Sn&&!On&&
!Cn):ze.value==="low"&&ze.classList.toggle("hidden",Cn)});const Je=Ae.selectedOptions&&Ae.selectedOptions[0];
Je&&Je.classList.contains("hidden")&&(Ae.value=v?"high":"medium")}}de&&de.classList.add("hidden"),Z&&
Z.classList.add("hidden"),I&&(I.disabled=!1),U&&(U.disabled=!0,U.classList.add("opacity-50"));const Ye=isGeminiImageModelKey(
f);if(_e||Fe)se&&(get("enable-search").checked=!1,se.classList.add("opacity-50","pointer-events-none")),
de&&(get("enable-url-context").checked=!1,de.classList.add("opacity-50","pointer-events-none")),Z&&me&&
(me.checked=!1,Z.classList.add("opacity-50","pointer-events-none")),le&&(Ee.checked=!1,le.classList.
add("opacity-50","pointer-events-none")),he&&ae&&(he.checked=!1,he.disabled=!0,ae.classList.add("opa\
city-50"));else if(ct||De)Z&&me&&(me.checked=!1,Z.classList.add("hidden","opacity-50","pointer-event\
s-none")),S.classList.remove("hidden"),_t?(Array.from(j.options).forEach(Ae=>{Ae.disabled=Ae.value===
"low"}),["minimal","medium","high"].includes(j.value)||(j.value="medium")):(Array.from(j.options).forEach(
Ae=>{["low","medium"].includes(Ae.value)&&(Ae.disabled=!0),["minimal","high"].includes(Ae.value)&&(Ae.
disabled=!1)}),["minimal","high"].includes(j.value)||(j.value=De?"minimal":"high")),I&&(I.disabled=!1),
De&&(W&&(W.checked=!1,W.disabled=!0),se&&se.classList.add("opacity-50","pointer-events-none"));else if(Ye)
Z&&me&&(me.checked=!1,Z.classList.add("hidden","opacity-50","pointer-events-none"));else if(Pt)S.classList.
remove("hidden"),U&&(U.disabled=!1,U.classList.remove("opacity-50")),Array.from(j.options).forEach(Ae=>{
Ae.disabled=!0}),le&&(Ee.checked=!1,le.classList.add("opacity-50","pointer-events-none"));else if(Xt){
S&&S.classList.remove("hidden"),I&&(I.checked=!0,I.disabled=!0),Array.from(j.options).forEach(Je=>{Je.
disabled=!["low","medium","high"].includes(Je.value)}),["low","medium","high"].includes(j.value)||(j.
value="medium"),[se,de,Z,le].forEach(Je=>{Je&&Je.classList.add("opacity-50","pointer-events-none")}),
[W,me,Ee].forEach(Je=>{Je&&(Je.checked=!1,Je.disabled=!0)});const Ae=get("enable-url-context");Ae&&(Ae.
checked=!1,Ae.disabled=!0),he&&ae&&(he.disabled=!1,ae.classList.remove("opacity-50"))}else if(f.includes(
"gemini")&&!Ye){S.classList.remove("hidden"),de&&de.classList.remove("hidden","opacity-50","pointer-\
events-none");const Ae=f.includes("gemini-3")||f.includes("gemini-4");Z&&(Ae?Z.classList.remove("hid\
den","opacity-50","pointer-events-none"):(me&&(me.checked=!1),Z.classList.add("hidden","opacity-50",
"pointer-events-none")));const Je=f.includes("flash"),ze=f==="gemini-4-hydrogen";Array.from(j.options).
forEach(Qe=>{ze?Qe.disabled=!["low","medium","high","xhigh"].includes(Qe.value):f==="gemini-4-carbon"||
f==="gemini-4-argon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?Qe.disabled=!["low","medium","h\
igh"].includes(Qe.value):f==="gemini-3.6-flash"?Qe.disabled=!["medium","high"].includes(Qe.value):f===
"gemini-3.5-flash-lite"?Qe.disabled=!["minimal","medium","high"].includes(Qe.value):Qe.value==="xhig\
h"?Qe.disabled=!0:["minimal","medium"].includes(Qe.value)?Qe.disabled=!Je:Qe.disabled=!1}),ze&&!["lo\
w","medium","high","xhigh"].includes(j.value)?j.value="high":(f==="gemini-4-carbon"||f==="gemini-4-a\
rgon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium","high"].includes(j.value)||
f==="gemini-3.6-flash"&&!["medium","high"].includes(j.value)?j.value="medium":f==="gemini-3.5-flash-\
lite"&&!["minimal","medium","high"].includes(j.value)?j.value="minimal":!Je&&["minimal","medium"].includes(
j.value)&&(j.value="high"),Ae?I&&(I.checked=!0,I.disabled=!0):I&&(I.disabled=!1),U&&f.includes("gemi\
ni-2.5")&&(U.disabled=!1,U.classList.remove("opacity-50")),U&&!f.includes("gemini-2.5")&&(U.disabled=
!0,U.classList.add("opacity-50"))}if(isLlmModel()&&(h.includes("gpt-5")||h.includes("o1")||h.includes(
"o3")||h.includes("grok-4.3")||h.includes("grok-4.5")||h.includes("grok-4.6")||h.includes("grok-4.20\
-0309-reasoning")||h.includes("grok-build")||h.includes("multi-agent")||h.includes("gpt")&&!h.includes(
"tts")))S&&S.classList.remove("hidden"),N("effort"),se&&se.classList.remove("opacity-50","pointer-ev\
ents-none");else if(v){S&&S.classList.remove("hidden"),N("effort");const Ae=get("vision-model-info");
if(Ae&&Ae.classList.toggle("hidden",h==="deepseek-v4.1-flash"||h==="deepseek-v4-flash-vision-exp"),W&&
(W.disabled=!1),se&&se.classList.remove("opacity-50","pointer-events-none"),de){const Je=get("enable\
-url-context");Je&&(Je.checked=!1),de.classList.add("opacity-50","pointer-events-none")}Z&&me&&(me.checked=
!1,Z.classList.add("opacity-50","pointer-events-none"))}else Fe||(se&&se.classList.remove("opacity-5\
0","pointer-events-none"),Z&&me&&(me.checked=!1,Z.classList.add("hidden","opacity-50","pointer-event\
s-none")));if(_e?le&&le.classList.add("opacity-50","pointer-events-none"):(le&&le.classList.remove("\
opacity-50","pointer-events-none"),(!Ye||ct)&&!f.includes("gpt-image")&&(he.disabled=!1,ae.classList.
remove("opacity-50"))),(Ye&&!ct||f.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(f)||isGrokVideoModel()||
Fe)&&he&&ae&&(he.checked=!1,he.disabled=!0,ae.classList.add("opacity-50")),le&&(isLlmModel()?(le.classList.
remove("hidden"),Ee.disabled=!1):(Ee.checked=!1,Ee.disabled=!0,le.classList.add("hidden"))),pe?(W&&(W.
checked=!0,W.disabled=!0),se&&se.classList.add("opacity-50","pointer-events-none"),le&&(Ee.checked=!1,
Ee.disabled=!0,le.classList.add("opacity-50","pointer-events-none"))):W&&!f.includes("tts")&&!Fe&&!De&&
(W.disabled=!1),Xt){[W,me,Ee].forEach(Je=>{Je&&(Je.checked=!1,Je.disabled=!0)});const Ae=get("enable\
-url-context");Ae&&(Ae.checked=!1,Ae.disabled=!0),[se,de,Z,le].forEach(Je=>{Je&&Je.classList.add("op\
acity-50","pointer-events-none")})}_&&(Me&&Me.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-fl\
ashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(h)),[W,Ee].forEach(Ae=>{Ae&&
(Ae.checked=!1,Ae.disabled=!0)}),[se,le].forEach(Ae=>{Ae&&Ae.classList.add("opacity-50","pointer-eve\
nts-none")}));const St=get("mask-btn");St&&(isGptImageModel()||isIdeogramObjectRemoval()?St.classList.
remove("hidden"):(St.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),
updateStsUi(),updateStsOptions(),d(),m(),g(),y(),b(),w(),x(),updateBatchUi(f);const Yt=get("gemini-f\
ree-tier-container");if(Yt){const Ae=typeof isGeminiApiModel=="function"&&isGeminiApiModel(f)&&(typeof currentGeminiBackend!=
"function"||currentGeminiBackend()!=="vertex_ai");Yt.classList.toggle("hidden",!Ae)}k(),C(),T(),purgeUnsupportedAttachments(
!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}r(G,"toggleOptionsForModel"),get("model-sel\
ect")&&(get("model-select").addEventListener("change",A),get("model-select").addEventListener("chang\
e",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),A(),minimalPromptMode?setMinimalPromptMode(
!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();const L=get("enable-canvas-\
mode");L&&(L.checked=canvasModeEnabled,L.addEventListener("change",()=>syncCanvasModeUi(L.checked))),
syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const $=get("enable-coding-mode");$&&($.
checked=codingModeEnabled,$.addEventListener("change",()=>syncCodingModeUi($.checked))),get("clear-c\
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const M=get(
"prompt-controls-toggle-btn");M&&(M.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>d()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>m()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,h){const v=(f||"").match(/\S*/)[0];if(v==="py\
exec")return"";if(v==="chat_error")return buildChatErrorBubbleHtml(u||"");const _=u||"",S=(v||"").toLowerCase();
if(S==="mathsvg"){const ae=getRenderableSvgCode("mathsvg",_);return ae&&typeof window.buildMathSvgRenderHtml==
"function"?window.buildMathSvgRenderHtml(ae):""}let E="";try{const ae=hljs.getLanguage(v)?v:"plainte\
xt";activeStreamingBubbleId&&_.length>2e4?E=escapeHtml(_):E=hljs.highlight(_,{language:ae}).value}catch{
E=escapeHtml(_)}const I=encodeURIComponent(_).replace(/'/g,"%27"),j=detectBlockedScriptsInCode(_),U=hashString(
`${v||"TEXT"}
${_||""}`);let W="";if(canvasModeEnabled){const ae=String(canvasPreviewState.selectedKey||"")===U,Ee=ae?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";W=`<button\
 class="canvas-preview-btn${ae?" canvas-active":""}" data-code="${I}" data-code-key="${U}" data-canv\
as-lang="${escapeHtml(v||"txt")}" title="${Ee}" aria-label="${Ee}" aria-pressed="${ae?"true":"false"}\
"><i class="fas ${ae?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,_)){const ae=j?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
W=`<button class="html-preview-btn" data-code="${I}" ${j?'data-suspicious="1"':""} title="${ae}" ari\
a-label="${ae}"><i class="fas ${j?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const se=`\
<button class="download-btn" data-code="${I}" data-lang="${v||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,de=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${I}" data-code-key="${U}" data-coding-lang="${escapeHtml(v||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,me=(v||"TEXT")+(j?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),Z=getRenderableSvgCode(S,_);return`${Z?buildSvgCodeRenderHtml(Z,U):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${U}"><div class="code-header"\
><span class="code-lang">${me}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${de}${W}${se}\
<button class="copy-btn" data-code="${I}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${v}">${E}</code></pre></d\
iv></div>`},link(u,f,h){return`<a href="${u}" title="${f||""}" target="_blank">${h}</a>`},image(u,f,h){
return buildChatImageHtml(u,{alt:h,title:f})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(bi=get("version-update-dismis\
s"))==null||bi.addEventListener("click",()=>{const u=localStorage.getItem("app_version")||"";u&&localStorage.
setItem("version_notified",u),hideModal("version-update-modal")});const P=get("version-update-clear-\
cache");if(P&&(P.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),P.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(P.checked)})),
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
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const R=r(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),D=r(u=>{const f=Array.isArray(u)?u:[],h=get("passkey-list"),v=get("passkey-count");if(v&&(v.
innerText=String(f.length)),!!h){if(!f.length){h.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
h.innerHTML="",f.forEach((_,S)=>{const E=_&&_.id?String(_.id):"",I=document.createElement("div");I.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const j=document.
createElement("div");j.className="min-w-0";const U=document.createElement("div");U.className="text-x\
s text-gray-200 truncate",U.innerText=_&&_.name?String(_.name):`Security Key ${S+1}`;const W=document.
createElement("div");W.className="text-[10px] text-gray-500 mt-1",W.innerText=_&&_.created_at?`\u767B\u9332\u65E5\u6642:\
 ${R(_.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",j.appendChild(U),j.appendChild(W),I.appendChild(
j);const se=document.createElement("button");se.type="button",se.className="bg-red-700 hover:bg-red-\
600 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",se.innerText="\u524A\u9664",
se.disabled=!E,E&&(se.onclick=()=>window.removeWebAuthnCredential(E)),I.appendChild(se),h.appendChild(
I)})}},"renderPasskeyList"),H=r(u=>{const f=get("session-list");if(f){if(!u||!u.length){f.innerHTML=
'<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(h=>{const v=h.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",_=h.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!h.is_current&&!h.is_revoked?`<button \
data-session-id="${escapeHtml(h.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
E=(h.user_agent||"Unknown").slice(0,120),I=h.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${v}${_}<div class="text-xs text-gray-200">${escapeHtml(
I)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(E)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(R(h.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(R(h.created_at))}</div></div>${S}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(h=>{h.onclick=async()=>{const v=h.getAttribute("data-session-id");if(!v||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const _=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:v})});let S={};try{S=await _.json()}catch{}if(_.ok){if(S.logged_out){
location.href="/login";return}await ee()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),ee=r(async()=>{const u=get("session-list");u&&(u.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api\
/sessions");let h={};try{h=await f.json()}catch{}if(!f.ok){if(h&&h.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const v=(h.sessions||[]).filter(_=>!_.is_revoked);H(v)},"loadSessions"),O=r(()=>{const u=get(
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
(f.checked=!0),A())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadSkills(),typeof window.loadProjects=="function"&&window.loadProjects(),
get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},get("new-chat-btn").onclick=
()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();const B=get(
"vision-model-change-btn");B&&(B.onclick=()=>_openVisionModelSelector());const J=get("compression-fo\
rmat-only");J&&(J.onchange=()=>{const u=J.checked,f=get("compression-max-size"),h=get("compression-m\
ax-dim");f&&(f.disabled=u),h&&(h.disabled=u);const v=get("compression-size-wrap"),_=get("compression\
-dim-wrap");v&&(v.style.opacity=u?"0.4":"1"),_&&(_.style.opacity=u?"0.4":"1")});const F=r(()=>{const u=get(
"enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=!!temporaryChatEnabled,
u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(u.checked)||(setTemporaryChatUiState(
f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggle");F(),document.addEventListener("vi\
sibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(!0)}),window.
addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener("beforeunlo\
ad",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const q=get("storage-usage-refresh");
q&&(q.onclick=()=>loadStorageUsage());let te=null;const V=r(()=>{const u=new Uint8Array(16);return window.
crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},"createAccountTr\
ansferId"),X=r((u={})=>{const f=get("account-transfer-progress"),h=get("account-transfer-progress-ba\
r"),v=get("account-transfer-progress-percent"),_=get("account-transfer-progress-text"),S=get("accoun\
t-transfer-progress-detail"),E=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&f.classList.remove(
"hidden"),h&&(h.style.width=`${E}%`),v&&(v.textContent=`${Math.round(E)}%`),_&&(_.textContent=u.message||
"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const j={queued:"\u9806\u756A\u5F85\u3061",
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=j[u.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const I=get("account-transfer-cancel-btn");I&&I.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),re=r(u=>{we&&(we.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const h=get("account-transfer-cancel-btn");
h&&(h.disabled=!u)},"setAccountTransferControls"),ce=r((u={})=>{const f=get("account-export-ready"),
h=get("account-export-ready-text"),v=get("account-export-expiry"),_=get("account-export-download-btn"),
S=!!(u.available&&u.download_url);if(f&&f.classList.toggle("hidden",!S),!S){_&&_.removeAttribute("hr\
ef");return}const E=Math.max(0,Number(u.size_bytes)||0),I=E>=1024*1024*1024?`${(E/(1024*1024*1024)).
toFixed(2)} GB`:`${(E/(1024*1024)).toFixed(1)} MB`;if(h){const j=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
h.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${I}${j}`}
if(v){const j=u.expires_at?new Date(u.expires_at):null;v.textContent=j&&!Number.isNaN(j.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${j.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
_&&(_.href=u.download_url)},"renderAccountExportAvailability"),Q=r(async u=>{for(;te===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),h=await f.json().catch(()=>({}));if(f.ok&&(h.state!=="pending"&&X(h),["ready","completed","f\
ailed","cancelled","expired"].includes(h.state)))return h}catch{}await new Promise(f=>setTimeout(f,700))}
return null},"pollAccountTransfer"),ue=r((u,f,h=!0)=>{f&&(X(f),ce(f),h&&f.state==="ready"?showToast(
f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):h&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),xe(u))},"handleFinishedAccountExport"),ne=r(async()=>{try{const u=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(ce(f),f.state==="ready"){X(f);return}if(["failed","cancelled","expired"].includes(
f.state)){X(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||te&&te.id===
f.job_id||te)return;const h={id:f.job_id,type:"export",stopped:!1,restored:!0};te=h,re(!0),X(f);const v=await Q(
h);v&&ue(h,v,!0)}catch{}},"refreshLatestAccountExport"),xe=r(u=>{te===u&&(te=null),u.stopped=!0,re(!1)},
"finishAccountTransfer"),ve=get("account-transfer-cancel-btn");ve&&(ve.onclick=async()=>{const u=te;
if(!(!u||u.stopped)){u.cancelRequested=!0,ve.disabled=!0,X({progress:0,phase:"cancelling",message:"\u30AD\
\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/transfer/${u.
id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}u.controller&&u.controller.abort(),
X({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),u.
type==="export"&&ce({available:!1}),xe(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const we=get("account-export-btn");we&&(we.onclick=async()=>{if(te)return;const u={id:V(),
type:"export",stopped:!1};te=u,re(!0),ce({available:!1}),X({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/acc\
ount/export",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({job_id:u.id}),keepalive:!0})),h=await f.json().catch(()=>({}));if(f.status===409&&
h.error==="export_in_progress"&&h.job_id)u.id=h.job_id;else if(!f.ok)throw new Error(h.error==="rate\
_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
h.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
X({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const v=await Q(u);!u.cancelRequested&&v&&ue(u,v,!0)}catch(f){const h=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";X({progress:0,phase:"failed",message:h}),
showToast(h,"error",!0),xe(u)}});const Oe=get("account-export-download-btn");Oe&&Oe.addEventListener(
"click",async u=>{const f=Oe.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();try{const h=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),v=await h.json().catch(
()=>({}));h.ok&&v.available&&v.download_url?(Oe.href=v.download_url,window.location.assign(v.download_url)):
(ce(v),X(v),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),ne())}catch{window.location.assign(f)}}}),re(!1),ne();const Ce=get("import-files-grid"),
lt=get("import-files-info"),fe=get("import-files-summary"),ye=r(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let ge=null;const ie=r(()=>{
if(!ge)return;const u=ge.files,f=ge.selection;let h=0;u.forEach(S=>{f.has(S.archive_path)&&(h+=Number(
S.size_bytes)||0)});const v=Number(ge.available_bytes)||0,_=h>v;fe&&(fe.textContent=`\u9078\u629E\u4E2D: ${ye(
h)} / \u5229\u7528\u53EF\u80FD: ${ye(v)}${_?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,fe.classList.
toggle("text-red-300",_)),lt&&(lt.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),it=r(
()=>{if(!Ce||!ge)return;Ce.innerHTML="";const u=ge.files;if(!u.length){Ce.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
ie();return}u.forEach(f=>{const h=document.createElement("label"),v=ge.selection.has(f.archive_path);
h.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${v?"border-blue-500":"border-gray-600"}`,h.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${v?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${ye(f.size_bytes)}</div></div>`;const _=h.
querySelector(".import-file-check");_.addEventListener("change",()=>{_.checked?ge.selection.add(f.archive_path):
ge.selection.delete(f.archive_path),h.classList.toggle("border-blue-500",_.checked),h.classList.toggle(
"border-gray-600",!_.checked),ie()}),Ce.appendChild(h)}),ie()},"renderImportFileItems"),st=r(u=>new Promise(
f=>{if(ge={files:u.files||[],selection:new Set((u.files||[]).map(h=>h.archive_path)),available_bytes:u.
available_bytes,resolve:f},it(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),vt=r(u=>{if(hideModal("import-files-modal"),ge){const f=ge.resolve;
ge=null,f(u)}},"closeImportFileSelection"),Mt=get("import-files-close");Mt&&(Mt.onclick=()=>vt(null));
const He=get("import-files-cancel");He&&(He.onclick=()=>vt(null));const at=get("import-files-confirm");
at&&(at.onclick=()=>{if(!ge)return;const u=Array.from(ge.selection);vt(u.length?u.join(","):"__none_\
_")});const Ke=get("import-files-select-all");Ke&&(Ke.onclick=()=>{ge&&(ge.files.forEach(u=>ge.selection.
add(u.archive_path)),it())});const Lt=get("import-files-none");Lt&&(Lt.onclick=()=>{ge&&(ge.selection.
clear(),it())});const Rt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let qe=null;const ut=r(
u=>{if(qe){const f=qe;qe=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),Ve=r(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}qe=f;const v=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],_=get("settings-confirmation-list");_&&(v.length?
_.innerHTML=v.map(E=>{const I=Rt[E.field]||E.field,j=At(E.current),U=At(E.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(I)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
j)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(U)}</d\
iv>
                            </div>`}).join(""):_.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${v.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),wt=get("settings-confir\
mation-modal");wt&&wt.addEventListener("click",u=>{u.target===wt&&ut(!1)});const tt=get("settings-co\
nfirmation-close");tt&&(tt.onclick=()=>ut(!1));const pt=get("settings-confirmation-cancel");pt&&(pt.
onclick=()=>ut(!1));const Gt=get("settings-confirmation-confirm");Gt&&(Gt.onclick=()=>ut(!0));const Bt=get(
"account-import-btn"),Et=get("account-import-inplace"),Ut=get("account-import-inplace-warning");if(Et&&
Ut){const u=r(()=>Ut.classList.toggle("hidden",!Et.checked),"syncInplaceWarn");Et.addEventListener("\
change",u),u()}Bt&&(Bt.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,h=get("account-import-categories"),v=h?Array.from(h.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(me=>me.value):[],_=get("account-import-inplace"),S=!!(_&&_.checked),E=get("account-impo\
rt-settings-bypass"),I=!!(E&&E.checked);let j=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!v.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const U=h?Array.from(h.querySelectorAll('input[type="checkbox"]:checked')).map(me=>(me.
closest("label")&&me.closest("label").textContent||me.value).trim()):v;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${U.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const W={id:V(),type:"import",stopped:!1,controller:new AbortController};
te=W,re(!0),X({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const se=get("account-import-result");let de=Promise.resolve(null);try{const Z=Math.max(1,Math.ceil(
f.size/10485760)),he=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({
method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:W.
controller.signal})),ae=await he.json().catch(()=>({}));if(!he.ok)throw new Error(ae.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\
\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");W.uploadId=ae.upload_id;const Ee=ae.chunk_size||10485760;
let le=0,Te=0;const je=r(async()=>{for(;;){const Pe=Te++;if(Pe>=Z)return;const Me=f.slice(Pe*Ee,Math.
min(f.size,(Pe+1)*Ee)),Ye=new FormData;Ye.append("chunk",Me,f.name),Ye.append("index",String(Pe));const gt=await apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"\
POST",body:Ye,signal:W.controller.signal})),St=await gt.json().catch(()=>({}));if(!gt.ok)throw new Error(
St.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");le++,X({
progress:Math.min(35,Math.round(le/Z*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${le}\
/${Z}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let pe=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),pe=!0);try{await Promise.
all([je(),je(),je()]);const Pe=await apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:W.controller.signal})),Me=await Pe.json().
catch(()=>({}));if(!Pe.ok)throw new Error(Me.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
X({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
pe&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let _e="",Fe=!1,De=0;const _t=r(
async()=>{let Pe=!1;const Me=r(()=>{Pe||(Pe=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Ye=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),gt=await Ye.
json().catch(()=>null);if(!Ye.ok||!gt){Me();return}cacheUserSettings(gt);const St=get("settings-moda\
l");if(St&&St.classList.contains("modal-open"))try{kt(gt)}catch{}gt.theme_color&&applyThemeColor(gt.
theme_color,!0),Object.prototype.hasOwnProperty.call(gt,"minimal_prompt_mode")&&gt.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(gt,"compact_prompt_mode")&&setCompactPromptMode(
!!gt.compact_prompt_mode)}catch{}Me()},"refreshSettingsFormAfterImport"),ct=r(Pe=>{const Me=Pe&&Pe.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";se&&(se.textContent=`\u5B8C\u4E86: ${Me}`,
se.classList.remove("hidden","text-red-300"),se.classList.add("text-emerald-300")),X({progress:100,phase:"\
completed",message:Me}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),v.includes("chats")&&loadThreads(),v.includes("skills")&&loadSkills(),v.includes("project\
s")&&typeof window.loadProjects=="function"&&window.loadProjects(),v.includes("files")&&loadStorageUsage(),
(v.includes("settings")||v.includes("api_credentials"))&&_t()},"finishImportSuccess"),Pt=r(async()=>{
try{const Me=await(await apiFetch(`/api/account/transfer/${W.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return Me&&Me.state?Me:null}catch{return null}},"fetchImportSta\
tus"),Xt=r(async()=>{const Pe=await Pt();if(!Pe)return{status:"unknown"};if(Pe.state==="completed")return ct(
Pe),{status:"done"};if(["failed","cancelled","expired"].includes(Pe.state))throw new Error(Pe.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Pe.state==="needs_sel\
ection"&&Array.isArray(Pe.files)){const Me=await st({files:Pe.files,available_bytes:Pe.available_bytes});
return Me===null?(X({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(_e=Me,{status:"reselect"})}if(Pe.state===
"needs_settings_confirmation"&&Array.isArray(Pe.settings_changes))return await Ve({settings_changes:Pe.
settings_changes})?(j=!0,{status:"reselect"}):(X({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
W.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Pe.state==="running"){const Me=await Promise.race([de.catch(()=>null),new Promise(Ye=>setTimeout(
()=>Ye(null),6e4))]);if(Me&&Me.state==="completed")return ct(Me),{status:"done"};throw Me&&["failed",
"cancelled","expired"].includes(Me.state)?new Error(Me.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Fe;){W.stopped=!0,await de.catch(()=>null),
W.stopped=!1,de=Q(W);let Pe;try{Pe=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:W.uploadId,
categories:v.join(","),job_id:W.id,selected_files:_e,restore_inplace:S,confirm_settings:j||I}),signal:W.
controller.signal}))}catch(ze){if(W.cancelRequested||ze&&ze.name==="AbortError")throw ze;const Qe=await Xt();
if(Qe.status==="done"){Fe=!0;break}if(Qe.status==="cancelled")return;if(Qe.status==="reselect")continue;
if(De<2){De++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Me=null;try{Me=await Pe.json()}catch{Me=null}if(Me===null){const ze=await Xt();if(ze.status==="d\
one"){Fe=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;if(Pe.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(De<2){De++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Pe.ok&&Me.error==="storage_limit_files"&&Me.files){const ze=await st(Me);if(ze===null){X({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
W.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}_e=ze;continue}if(Me&&Me.status==="settings_confirmation"&&
Array.isArray(Me.settings_changes)){if(!await Ve(Me)){X({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),W.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}j=!0;continue}if(!Pe.ok)throw new Error(Me.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Ye=Me.imported||{},gt=[`\u8A2D\u5B9A ${Ye.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Ye.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Ye.chats||0}\u4EF6`,`Skill ${Ye.skills||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Ye.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Ye.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Ye.
diagnostics||0}\u4EF6`].join(" / "),St=Me.duplicates||{},Yt={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"\
Skill",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Ae=[];for(const ze of Object.keys(Yt)){const Qe=Number(St[ze])||0;Qe>
0&&Ae.push(`${Yt[ze]} ${Qe}\u4EF6`)}const Je=Ae.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Ae.
join("\u3001")}\uFF09`:"";se&&(se.textContent=`\u5B8C\u4E86: ${gt}${Je}`,se.classList.remove("hidden",
"text-red-300"),se.classList.add("text-emerald-300")),X({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\
\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),v.includes("chats")&&loadThreads(),v.includes("skills")&&loadSkills(),v.includes("project\
s")&&typeof window.loadProjects=="function"&&window.loadProjects(),v.includes("files")&&loadStorageUsage(),
(v.includes("settings")||v.includes("api_credentials"))&&_t(),Fe=!0}}catch(me){if(W.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(W.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),W.cancelRequested||me&&me.name==="AbortError")return;const Z=me&&me.message?
me.message:"",he=Z==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
Z||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";X({progress:0,phase:"f\
ailed",message:he}),se&&(se.textContent=he,se.classList.remove("hidden","text-emerald-300"),se.classList.
add("text-red-300")),showToast(he,"error",!0)}finally{W.stopped=!0,await de.catch(()=>null),xe(W)}});
const Ot=get("account-dedupe-btn"),jt=get("account-dedupe-result"),Ft=r((u,f=!1)=>{jt&&(jt.textContent=
u,jt.classList.remove("hidden"),jt.classList.toggle("text-red-300",!!f),jt.classList.toggle("text-em\
erald-300",!f))},"showDedupeResult");Ot&&(Ot.onclick=async()=>{const u=r(async()=>{const f=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),h=await f.json().catch(()=>null);if(!f.ok||!h)throw new Error(h&&h.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!h.has_duplicates){Ft("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const v=[],_={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",
feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};
for(const W of["chats","skills","files","feedback","diagnostics"]){const se=Number(h.duplicates&&h.duplicates[W])||
0;se>0&&v.push(`${_[W]} ${se}\u4EF6`)}const S=Number(h.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${h.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${h.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${v.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const E=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),I=await E.json().catch(()=>null);if(!E.ok||!I)throw new Error(
I&&I.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const j=[];for(const W of["chats","skills","files","feedback","diagnostics"]){const se=Number(I.removed&&
I.removed[W])||0;se>0&&j.push(`${_[W]} ${se}\u4EF6`)}const U=Number(I.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\
\u305F\u3081\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${I.kept_referenced_files}\u4EF6\uFF09`:"";Ft(
`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\u3057\u305F: ${j.join("\u3001")||"0\u4EF6"}${U}`),
loadThreads(),loadSkills(),loadStorageUsage()},"run");if(!Ot.disabled){Ot.disabled=!0,Ft("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\
\u3044\u307E\u3059...");try{await u()}catch(f){Ft(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{Ot.disabled=!1}}});const zt=get("site-cache-usage-refresh");zt&&(zt.onclick=()=>loadSiteCacheUsage());
const Dt=get("clear-site-cache-btn");Dt&&(Dt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Dt)});
const Y=get("enc-scan-result"),ke=r(async(u=null)=>{Y&&(Y.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const h=await apiFetch(
f,{cache:"no-store"}),v=await h.json();if(!h.ok){Y&&(Y.textContent=v.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const _=v.total||0,S=v.encrypted||0,E=v.unencrypted||0;let I=`Total: ${_} / Encrypted: ${S} /\
 Plain: ${E}`;if(v.samples&&v.samples.length){const j=v.samples.slice(0,8).map(U=>{const W=U.timestamp?
new Date(U.timestamp).toLocaleString():"";return`#${U.id} (${U.role||""}) ${W}`}).join(" / ");I+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${j}</div>`}Y&&(Y.innerHTML=I)}catch{Y&&(Y.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Re=get("enc-scan-all");Re&&(Re.onclick=()=>ke(
null));const Ge=get("enc-scan-thread");Ge&&(Ge.onclick=()=>currentThreadId?ke(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const et=get("admin-enc-\
list");let mt=null,ft=!1;const ot=r(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThread\
EncryptedFromMessages"),Le=r(()=>{mt=ot(allMessages)},"refreshCurrentThreadEncStateFromMessages"),$e=r(
async(u,f,{confirmPrompt:h=!0,reloadCurrent:v=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const _=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(h&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;ft=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
E=await S.json().catch(()=>({}));return S.ok?(showToast(`${_}\u3057\u307E\u3057\u305F\uFF08${E.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),mt=!!f,v&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),et&&await nt(),!0):
(showToast(E.error||`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{ft=!1}},"setAdminThreadEncry\
ption"),We=r(u=>{if(!et)return;const f=u.threads||[];if(!f.length){et.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
et.innerHTML=f.map(h=>{const v=h.encrypted_count>0?"enc":"plain",_=v==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=v==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",E=h.
updated_at?new Date(h.updated_at).toLocaleString():"",I=escapeHtml(String(h.thread_id)),j=currentThreadId&&
String(currentThreadId)===String(h.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(h.title||
"")}">${escapeHtml(h.title||"(\u7121\u984C)")}${j?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${E} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${h.
message_count} / \u6697\u53F7\u5316: ${h.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${I}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${I}" data-enable="${v==="enc"?"0":"1"}" data-progress-expected-slow="true">${_}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),nt=r(async()=>{if(et){et.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){et.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(We(f),currentThreadId&&Array.isArray(f.threads)){const h=f.threads.find(v=>String(
v.thread_id)===String(currentThreadId));h&&(mt=!!h.encrypted)}}catch{et.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>nt()),window.__loadAdminEncThreads=
nt,window.__refreshAdminThreadEncState=Le,window.__setAdminThreadEncryption=$e;const Ct=get("encrypt\
ion-status-admin-toggle");Ct&&Ct.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),et&&(et.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const I=f.getAttribute("data-id");if(!I)return;typeof dt==
"function"?dt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(I)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const h=u.target.closest(".admin-enc-toggle");if(!h||ft)return;const v=h.getAttribute("data-i\
d"),_=h.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;h.disabled=!0;
const E=h.textContent;h.textContent="\u51E6\u7406\u4E2D...";try{await $e(v,_,{confirmPrompt:!1,reloadCurrent:!0})}finally{
h.disabled=!1,h.textContent=E,await nt()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const $t=r(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(h=>{const v=h.status||"new",_=h.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=h.created_at?new Date(h.created_at).toLocaleString():"",E=h.replied_at?new Date(h.replied_at).toLocaleString():
"",I=h.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${h.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(h.username||
"")}${_}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(S)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(v)}</di\
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
I)}</textarea>
                                ${I?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
E)}</div>`:""}
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
                    `}).join("")}},"renderBanAppeals"),Nt=r(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const h=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!h.
ok)return;const _=(await h.json()).unread_count||0;f.textContent=String(_),u&&_>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${_}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),xt=r(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const h=await f.json();$t(h.items||[]),
await Nt(!1)}catch{}}},"loadBanAppeals"),Ue=r(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await xt()}catch{}},"markBanAppealsRead"),Xe=r(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await xt()}catch{}},"updateBanAppealStatus"),Qt=r(()=>{
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
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),Ht=r(()=>{const u=get("set-st\
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
                `,f.appendChild(h);const v=get("reset-llm-transcribe-prompt");v&&(v.onclick=()=>{const _=get(
"set-llm-transcribe-prompt");_&&(_.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),Zt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
 text-xs text-gray-200";return Zt.map(v=>{const _=v.barLocked===!0,S=v.barLabel||"MCP",E=_?`<div cla\
ss="text-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\
\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002</div>`:
"",I=_?`<input type="checkbox" id="${u}-auto-sys-${v.key}-enabled" class="accent-yellow-500 w-3 h-3"\
 disabled>`:`<input type="checkbox" id="${u}-auto-sys-${v.key}-enabled" class="accent-yellow-500 w-3\
 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${v.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${_?`ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306E${S}\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"`:
""}>
                                ${I}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${u}-auto-sys-${v.key}-text" class="${h}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${v.hint?`<div class="text-[10px] text-gray-500 mt-1">${v.hint}</div>`:""}
                        ${E}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{Zt.forEach(h=>{const v=f&&
typeof f=="object"?f[h.key]||{}:{},_=get(`${u}-auto-sys-${h.key}-enabled`),S=get(`${u}-auto-sys-${h.
key}-text`);_&&(h.barLocked===!0?_.disabled=!0:_.checked=v.enabled!==!1),S&&(S.value=v.text||"",S.placeholder=
v.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Vt=r((u,f=null)=>{if(f){const h=get(f);h&&(h.checked=!0)}Zt.forEach(h=>{const v=get(`${u}-auto\
-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);if(v&&(h.barLocked!==!0?v.checked=!0:v.
disabled=!0),_){const S=_.placeholder||"";_.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Se=r(u=>{const f={};return Zt.forEach(h=>{const v=get(`${u}\
-auto-sys-${h.key}-enabled`),_=get(`${u}-auto-sys-${h.key}-text`);f[h.key]={enabled:h.barLocked===!0?
!0:v?v.checked:!0,text:_?_.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
Se,window.ensureAutoSystemPromptSettingsCard=()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.
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
                `,f.appendChild(h)},Qt(),Ht(),F();const Be=r(()=>{const u=get("set-default-model");if(!u)
return;const f=u.value;u.innerHTML="",MODELS.forEach(v=>{const _=document.createElement("optgroup");
_.label=v.category,(v.items||[]).forEach(S=>{const E=document.createElement("option");E.value=S.id,E.
textContent=S.name,_.appendChild(E)}),u.appendChild(_)});const h=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";h&&Array.from(u.options).some(v=>v.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),bt=r(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(v=>{const _=(v.items||[]).filter(E=>{const I=(E.id||"").toLowerCase();
return I.startsWith("gemini-")||I.startsWith("gpt-4o")||I.startsWith("claude-")||I.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
I)});if(_.length===0)return;const S=document.createElement("optgroup");S.label=v.category,_.forEach(
E=>{const I=document.createElement("option");I.value=E.id,I.textContent=E.name+" \u2605",S.appendChild(
I)}),u.appendChild(S)});const h=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";h&&Array.from(u.options).some(v=>v.value===h)&&(u.value=h),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),kt=r(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-glo\
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
!1);const v=get("bot-status");v&&(u.is_bot_banned?(v.textContent=`BAN\u4E2D: ${u.bot_ban_reason||"Bo\
t detection"}`,v.classList.remove("hidden"),v.classList.add("text-red-400")):v.classList.add("hidden")),
u&&u.theme_color?(applyThemeColor(u.theme_color,!0),syncThemeInputs(u.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=u.username);const _=get("2fa-badge"),S=get("disable-2fa-btn");u.is_2fa_enabled?
(_.innerText="ENABLED",_.classList.replace("bg-gray-700","bg-green-600"),_.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(_.innerText="DISABLED",_.classList.replace("bg\
-green-600","bg-gray-700"),_.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!u.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=u.default_2fa_method||"totp");const E=get(
"set-passkey-only-login"),I=get("passkey-only-note"),j=Array.isArray(u.passkey_credentials)?u.passkey_credentials:
[];if(D(j),E){E.checked=!!u.passkey_only_login;const me=j.length>0||!!u.has_webauthn;E.disabled=!me,
me||(E.checked=!1),I&&(me?I.classList.add("hidden"):I.classList.remove("hidden"))}const U=get("mig-s\
tatus-box"),W=get("mig-progress-text"),se=get("mig-progress-bar");if((u.migration_status||"idle")===
"processing"){U.classList.remove("hidden");const me=(u.migration_progress||"").split("/");if(me.length===
2){const Z=parseInt(me[0]||"0",10),he=parseInt(me[1]||"0",10);W&&(W.innerText=`${Z} / ${he}`),se&&he>
0&&(se.style.width=`${Math.min(100,Math.floor(Z/he*100))}%`)}}else U.classList.add("hidden"),se&&(se.
style.width="0%"),W&&(W.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSe\
ttingsFormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),snapshotSidebarHistory("settings-open-before");const u=await ensureUserSettingsSnapshot();u&&kt(
u);const f=get("search-box"),h=f?f.value:"";clearTimeout(searchTimeout);const v=get("settings-search");
if(v&&(v.value=""),filterSettings(),Be(),bt(),showModal("settings-modal"),refreshSettingsTabsScroll(),
requestAnimationFrame(()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(h,"restored-search-b\
ox-open"),revealPersistentSidebarLists(),snapshotSidebarHistory("settings-open-after"),[50,200,400,800].
forEach(_=>{setTimeout(()=>{restoreThreadSearchValue(h,"restored-search-box-"+_+"ms"),snapshotSidebarHistory(
"settings-open-later-"+_+"ms")},_)}),syncAdaptiveBlurSettingsUi(),loadStorageUsage(),loadSiteCacheUsage(),
ne(),Ht(),typeof window.__loadAdminEncThreads=="function")try{window.__loadAdminEncThreads()}catch{}
location.pathname!=="/settings"&&history.pushState({modal:"settings",from:location.pathname},"","/se\
ttings"),Nt(!0),xt(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\
\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044","error",!0)),hn(),O(),
ee();try{loadMcpServers()}catch{}};const dt=r((u=!1)=>{settingsModelPickerTarget&&window.closeModelModal(
!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close\
-later-300ms"),300),!u&&location.pathname==="/settings"&&history.back()},"closeSettingsModal"),Wt=r(
()=>{const u=get("set-theme-color"),f=get("set-theme-color-text"),h=get("theme-reset-btn"),v=document.
querySelectorAll("#theme-presets .theme-swatch"),_=r((S,E=!0)=>{const I=normalizeHex(S);I&&(applyThemeColor(
I,E),syncThemeInputs(I))},"applyFromValue");u&&u.addEventListener("input",()=>_(u.value,!0)),f&&(f.addEventListener(
"change",()=>{const S=normalizeHex(f.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}_(S,!0)}),f.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
f.blur())})),h&&(h.onclick=()=>_(THEME_DEFAULT,!0)),v.forEach(S=>{S.addEventListener("click",()=>_(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),en=r(()=>{const u=get("reset-global-sys-prom\
pt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const f=get(
"reset-thread-sys-prompt");f&&(f.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-set-auto-sys-prompt-defaults");h&&(h.onclick=()=>{Vt("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const v=get("reset-thread-auto-sys-prompt-defaults");v&&(v.onclick=()=>{Vt("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>dt();const Jn=get("settings-header-close");Jn&&(Jn.onclick=()=>dt());
const ln=get("settings-search");ln&&(ln.addEventListener("input",filterSettings),ln.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const h=f.querySelector(
":scope > .settings-match");h&&h.scrollIntoView({behavior:"smooth",block:"start"})}}));const Kn=get(
"settings-search-clear");Kn&&Kn.addEventListener("click",()=>{ln&&(ln.value="",filterSettings(),ln.focus())}),
Wt(),en(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const jn=get("set-gemini-local-python-dialog");jn&&(jn.onchange=()=>setGeminiLocalPyDialogEnabled(jn.
checked));const Xn=get("set-gemini-backend");Xn&&(Xn.onchange=()=>{syncGeminiBackendUi(),typeof window.
toggleOptions=="function"&&window.toggleOptions()});const Yn=get("set-admin-api-key-mode");Yn&&(Yn.onchange=
()=>syncAdminApiKeyModeUi());const Nn=get("set-temp-chat-timeout-seconds");Nn&&(Nn.onchange=()=>{applyTemporaryChatTimeoutSeconds(
Nn.value)});const Qn=get("slash-command-cancel-btn");Qn&&(Qn.onclick=()=>{hidePendingSlashCommandIndicator();
const u=get("prompt-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-s\
ettings-btn").onclick=async()=>{var I;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),h=readPromptBarModeFromForm(),v={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:Se(
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
((I=get("set-server-diagnostic-log-retention-hours"))==null?void 0:I.value)||1),enable_server_diagnostic_log:get(
"set-server-diagnostic-log")?get("set-server-diagnostic-log").checked:!1,enable_client_debug_log:get(
"set-client-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-o\
nly-login")?get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-goog\
le")?get("set-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set\
-default-2fa-method").value:"totp",new_username:u?u.value:null,new_password:f?f.value:null},_=get("s\
et-e2ee")?get("set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
_!==S&&(v.enable_e2ee=_),get("set-openai")&&(v.openai_key=get("set-openai").value),get("set-gemini")&&
(v.gemini_key=get("set-gemini").value),get("set-gemini-free")&&(v.gemini_free_key=get("set-gemini-fr\
ee").value),get("set-deepseek")&&(v.deepseek_key=get("set-deepseek").value),get("set-zai")&&(v.zai_key=
get("set-zai").value),get("set-kimi")&&(v.kimi_key=get("set-kimi").value),get("set-mistral")&&(v.mistral_key=
get("set-mistral").value),get("set-ideogram")&&(v.ideogram_key=get("set-ideogram").value),get("set-a\
nthropic")&&(v.anthropic_key=get("set-anthropic").value),v.model_api_keys=normalizeModelApiKeyMap(modelApiKeyMap),
get("set-gemini-backend")&&(v.gemini_backend=normalizeGeminiBackend(get("set-gemini-backend").value)),
get("set-gemini-vertex-project")&&(v.gemini_vertex_project=get("set-gemini-vertex-project").value),get(
"set-gemini-vertex-location")&&(v.gemini_vertex_location=get("set-gemini-vertex-location").value),get(
"set-gemini-vertex-credentials-json")&&(v.gemini_vertex_credentials_json=get("set-gemini-vertex-cred\
entials-json").value),get("set-xai")&&(v.xai_key=get("set-xai").value),get("set-google-key")&&(v.google_key=
get("set-google-key").value),get("set-google-project")&&(v.google_project=get("set-google-project").
value),get("set-admin-api-key-mode")&&(v.admin_api_key_mode=normalizeAdminApiKeyMode(get("set-admin-\
api-key-mode").value)),get("set-bot-detect")&&(v.bot_detection_enabled=get("set-bot-detect").checked),
get("set-bot-detect-global")&&(v.bot_detection_global_enabled=get("set-bot-detect-global").checked);
const E=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"appli\
cation/json"},body:JSON.stringify(v)});if(E.ok){let j="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const de=await E.json();de&&de.message&&(j=de.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!v.light_mode_enabled,liquid_glass_enabled:!!v.liquid_glass_enabled,
math_steps_enabled:!!v.math_steps_enabled,default_math_steps:!!v.default_math_steps})),typeof window.
applyMathStepsSettings=="function"&&(window.applyMathStepsSettings({math_steps_enabled:!!v.math_steps_enabled}),
v.math_steps_enabled||setMathStepsEnabled(!1,{persist:!1})),window.applySavedUserSystemPromptSettings(
{system_prompt:v.system_prompt,system_prompt_enabled:v.system_prompt_enabled,apply_global_system_prompt:v.
apply_global_system_prompt,apply_auto_system_prompt_notices:v.apply_auto_system_prompt_notices,auto_system_prompt_notices_config:v.
auto_system_prompt_notices_config}),dt();const U=currentUsername,W=CHAT_CONFIG.enableE2EE;enterToSend=
v.enter_to_send,autoSearchOnLinks=v.auto_search_on_links;const se=useSwCache;useSwCache=v.use_sw_cache,
window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!v.clear_cache_on_version_update),
compactPromptMode=v.compact_prompt_mode,minimalPromptMode=v.minimal_prompt_mode,voiceStudioUiEnabled=
v.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=v.temp_chat_timeout_seconds,applyThemeColor(v.theme_color,
!0),syncThemeInputs(v.theme_color),applyLightMode(v.light_mode_enabled),applyLiquidGlassMode(v.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),se!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(j,"success"),syncClientDebugLogToggle(
v.enable_client_debug_log,"settings saved"),v.new_username&&v.new_username!==U?setTimeout(()=>location.
reload(),1e3):v.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let j={};try{j=await E.json()}catch{}showToast(j.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
son"},body:JSON.stringify({username:f,mode:"single"})}),v=await h.json(),_=get("bot-unban-result");if(h.
ok&&v&&v.status==="ok")_&&(_.textContent=`${f} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=v&&v.error?v.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const u=get("bot-unban-username"),f=u?u.value.trim():"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"linked"})}),v=await h.json(),_=get("bot-unban-result");if(h.
ok&&v&&v.status==="ok")_&&(_.textContent=`${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=v&&v.error?v.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const u=get(
"bot-speed-test-btn"),f=get("bot-speed-test-result");u&&(u.disabled=!0),u&&u.classList.add("opacity-\
60","cursor-not-allowed"),f&&(f.classList.remove("hidden"),f.textContent="\u5B9F\u884C\u4E2D...");try{
const h=r(le=>{f&&(f.textContent=le)},"setBox"),v=r(()=>`${Date.now()}_${Math.random().toString(36).
slice(2)}`,"cacheBust"),_=r((le,Te)=>!le||!Te||Te<=0?0:le*8/(Te/1e3)/1e3/1e3,"toMbps"),S=r(le=>Number.
isFinite(le)?`${le.toFixed(0)} ms`:"-","fmtMs"),E=r(le=>Number.isFinite(le)?`${le.toFixed(le>=100?0:
1)} Mbps`:"-","fmtMbps"),I=r(async(le,Te)=>{const je=await le.json().catch(()=>({}));return je&&je.error?
je.error:Te},"parseErr"),j=[];h("\u6E2C\u5B9A\u4E2D... ping");for(let le=0;le<4;le++){const Te=performance.
now(),je=await apiFetch(`/api/speedtest/ping?_=${v()}`,{cache:"no-store"}),pe=performance.now();if(!je.
ok)throw new Error(await I(je,"ping_failed"));await je.json().catch(()=>({})),j.push(pe-Te)}const U=j.
reduce((le,Te)=>le+Te,0)/Math.max(1,j.length),W=Math.min(...j),se=r(async le=>{const Te=performance.
now(),je=await apiFetch(`/api/speedtest/download?bytes=${le}&_=${v()}`,{cache:"no-store"});if(!je.ok)
throw new Error(await I(je,"download_failed"));const pe=await je.arrayBuffer(),_e=performance.now();
return{bytes:pe.byteLength||le,ms:_e-Te,mbps:_(pe.byteLength||le,_e-Te)}},"runDownload");h(`\u6E2C\u5B9A\u4E2D... p\
ing ${S(U)}
\u6E2C\u5B9A\u4E2D... download`);const de=[];for(const le of[2*1024*1024,8*1024*1024])de.push(await se(
le)),h(`\u6E2C\u5B9A\u4E2D... ping ${S(U)}
download ${E(Math.max(...de.map(Te=>Te.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const me=Math.max(...de.map(le=>le.mbps)),Z=r(async le=>{const Te=new Uint8Array(
le),je=performance.now(),pe=await apiFetch(`/api/speedtest/upload?_=${v()}`,{method:"POST",headers:{
"Content-Type":"application/octet-stream"},body:Te,cache:"no-store"}),_e=performance.now();if(!pe.ok)
throw new Error(await I(pe,"upload_failed"));const Fe=await pe.json().catch(()=>({})),De=Number(Fe.bytes_received||
le)||le;return{bytes:De,ms:_e-je,mbps:_(De,_e-je),serverMs:Number(Fe.server_elapsed_ms||0)||0}},"run\
Upload"),he=[];for(const le of[1*1024*1024,4*1024*1024])he.push(await Z(le));const ae=Math.max(...he.
map(le=>le.mbps)),Ee=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(U)} / ${S(W)}`,`Download (best): ${E(me)}`,`Upload (best): ${E(ae)}`,`Download \
runs: ${de.map(le=>`${Math.round(le.bytes/1024/1024)}MB=${E(le.mbps)}`).join(", ")}`,`Upload runs: ${he.
map(le=>`${Math.round(le.bytes/1024/1024)}MB=${E(le.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\
\u4F53\u306E\u901F\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
h(Ee.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(h){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${h&&h.message?h.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>xt()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Ue()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const h=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){h&&await Ue([Number(h)]);return}if(f.classList.contains(
"ban-appeal-status")){const v=f.getAttribute("data-status");h&&v&&await Xe({id:Number(h),status:v});
return}if(f.classList.contains("ban-appeal-reply-send")){const v=f.closest("[data-appeal-id]"),_=v?v.
querySelector(".ban-appeal-reply"):null,S=_?_.value:"";h&&await Xe({id:Number(h),admin_reply:S});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const v=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";h&&await Xe({id:Number(
h),block_user:!0,block_reason:v});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
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
"upload-modal");return!!(u&&!u.classList.contains("hidden"))},"isUploadModalOpen"),cn=get("drop-over\
lay");let mn=0;const xi=r(()=>{wi()||cn&&(cn.classList.remove("hidden"),cn.classList.add("flex"))},"\
showDropOverlay"),fn=r(()=>{mn=0,cn&&(cn.classList.add("hidden"),cn.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=fn;const qt=get("upload-dropzone");qt&&(qt.addEventListener("dr\
agover",u=>{u.preventDefault(),qt.classList.add("dragover")}),qt.addEventListener("dragleave",()=>{qt.
classList.remove("dragover")}),qt.addEventListener("drop",u=>{u.preventDefault(),u.stopPropagation(),
qt.classList.remove("dragover"),fn();const f=u.dataTransfer?u.dataTransfer.files:null;f&&f.length&&handleFiles(
f)})),window.addEventListener("dragenter",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.
types.includes("Files")||(mn+=1,xi())}),window.addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||u.preventDefault()}),window.addEventListener("dragle\
ave",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(mn=Math.max(
0,mn-1),(mn===0||!u.relatedTarget||u.clientY<=0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=
window.innerHeight)&&fn())}),window.addEventListener("dragend",()=>{fn()}),window.addEventListener("\
drop",u=>{fn(),!(!u.dataTransfer||!u.dataTransfer.files||u.dataTransfer.files.length===0)&&(u.preventDefault(),
!(qt&&qt.contains(u.target))&&handleFiles(u.dataTransfer.files))});const Zn=get("bot-admin-modal"),ki=r(
u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML="",!u||!u.length){f.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((h,v)=>{const _=!!h.is_bot_banned,S=h.bot_detection_enabled!==!1,E=Number(h.lock_remaining_seconds)||
0,I=document.createElement("div");I.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",I.style.animationDelay=`${Math.min(v,12)*.02}\
s`,I.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(h.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${_?"BAN\u4E2D":"\u6B63\u5E38"}${E?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(E/60)}\u5206\uFF09`:""} ${h.bot_ban_reason?
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
                    `,f.appendChild(I)})}},"renderBotUsers"),gn=r(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const h=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),v=await h.json();h.ok&&v&&v.users?ki(v.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
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
"data-user");if(!h)return;let v;if(f.classList.contains("bot-toggle-detect")){const _=f.getAttribute(
"data-enabled")!=="1";v=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:h,action:"toggle_detection",enabled:_})})}else if(f.classList.
contains("bot-toggle-ban"))if(f.getAttribute("data-banned")==="1")v=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:h,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;v=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"ban",reason:"Admin ban"})})}else if(f.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;v=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"unban_linked"})})}else if(f.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${h} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;v=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:h,action:"delete_account"})})}if(v){if(v.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(v.ok){if(f.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${h}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),h===currentUsername)){location.href="/";return}}else{
let _={};try{_=await v.json()}catch{}showToast(_.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
_.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await gn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Jt={"/settings":{id:"settings-mod\
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
"open")},"/admin-bot-logs":{id:"bot-log-modal",open:r(()=>Dn(),"open")}};Jt["/gem"]=Jt["/skill"];const ei=r(
(u,f=!1)=>{switch(u){case"settings-modal":dt(f);break;case"upload-modal":closeUploadModal(f);break;case"\
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
"function"?submitMcpDecision("deny"):hideModal(u);break;case"api-key-required-modal":{const v=get("a\
pi-key-modal-cancel-btn");v&&typeof v.onclick=="function"?v.click():hideModal(u);break}case"lyria-st\
udio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(u);break;case"voice-studio-m\
odal":window.VoiceStudio?window.VoiceStudio.close():hideModal(u);break;case"version-update-modal":const h=localStorage.
getItem("app_version")||"";h&&localStorage.setItem("version_notified",h),hideModal(u);break;default:
hideModal(u);break}},"closeModalById");window.addEventListener("popstate",u=>{var E,I;const f=(E=u.state)==
null?void 0:E.settingsModelPicker;if(location.pathname==="/settings"&&settingsModelPickerTargets.includes(
f)){const j=r(()=>openSettingsModelPicker(f,!0),"restorePicker");(I=get("settings-modal"))!=null&&I.
classList.contains("modal-open")?j():window.openSettingsModal().then(()=>{var U;location.pathname===
"/settings"&&((U=history.state)==null?void 0:U.settingsModelPicker)===f&&j()});return}settingsModelPickerTarget&&
window.closeModelModal(!0);let h=!1;Object.values(Jt).forEach(j=>{const U=get(j.id);U&&U.classList.contains(
"modal-open")&&location.pathname!==Object.keys(Jt).find(W=>Jt[W].id===j.id)&&(ei(j.id,!0),h=!0)});const v=location.
pathname.match(/^\/c\/(.+)$/),_=location.pathname.match(/^\/p\/(.+)$/);if(v){const j=decodeURIComponent(
v[1]);String(currentThreadId)!==String(j)&&loadMessages(j,{skipHistory:!0})}else if(_){const j=decodeURIComponent(
_[1]);typeof window.openProjectView=="function"&&window.openProjectView(j,{skipHistory:!0})}else location.
pathname==="/"&&(typeof window.closeProjectView=="function"&&window.closeProjectView({skipHistory:!0}),
currentThreadId&&startNewChat({skipHistory:!0}));const S=Jt[location.pathname];if(S){const j=get(S.id);
j&&!j.classList.contains("modal-open")&&S.open()}}),document.addEventListener("keydown",u=>{u.key===
"Escape"&&settingsModelPickerTarget&&(u.preventDefault(),window.closeModelModal())});const ti=location.
pathname;Jt[ti]&&(history.replaceState({},"","/"),setTimeout(()=>Jt[ti].open(),500)),get("easy-login\
-generate")&&(get("easy-login-generate").onclick=async()=>{const u=get("easy-login-mins"),f=u?parseInt(
u.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${f}\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))
return;const v=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({minutes:f})})).json();v&&v.temp_password?(get("easy-login-code").textContent=
v.temp_password,get("easy-login-exp").textContent=v.expires_at||"",get("easy-login-result").classList.
remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(f&&f.cancelled){const h=get("easy-login-r\
esult");h&&h.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const _i=4,Si=5*1024*1024;function ni(u,f,h){let v=[];const _=r(()=>{h&&(h.innerHTML="",
v.forEach((S,E)=>{const I=document.createElement("div");I.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const j=document.createElement("img");j.src=S.url,j.alt=S.file.name,
j.className="w-full h-full object-cover";const U=document.createElement("button");U.type="button",U.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",U.textContent=
"\xD7",U.onclick=()=>{URL.revokeObjectURL(S.url),v.splice(E,1),_()},I.appendChild(j),I.appendChild(U),
h.appendChild(I)}))},"render");return u&&f&&(u.onclick=()=>f.click(),f.onchange=S=>{for(const E of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(E.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(E.
size>Si){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(v.
length>=_i){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}v.push({file:E,url:URL.createObjectURL(E)})}S.target.value="",_()}),{files:r(()=>v.
map(S=>S.file),"files"),clear:r(()=>{v.forEach(S=>URL.revokeObjectURL(S.url)),v=[],_()},"clear")}}r(
ni,"createFbImagePicker");async function ii(u,f){try{const h=new FormData;return f.forEach(v=>h.append(
"images",v,v.name)),(await apiFetch(u,{method:"POST",body:h})).ok}catch{return!1}}r(ii,"uploadFbImag\
es");const si=ni(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));function Ci(u){const f=String(
u||"");if(!f)return;const h=`[data-feedback-id="${f.replace(/["\\]/g,"\\$&")}"]`,v=r(()=>document.querySelector(
h),"findRow"),_=r(()=>{const U=v();return U?(typeof jumpToSetting=="function"?jumpToSetting("feedbac\
k",U):(U.scrollIntoView({behavior:"smooth",block:"center"}),U.classList.add("settings-jump-highlight"),
setTimeout(()=>U.classList.remove("settings-jump-highlight"),2e3)),!0):!1},"reveal");if(_())return;const S=get(
"settings-modal");!(S&&(S.classList.contains("modal-open")||S.classList.contains("modal-prep")))&&typeof window.
openSettingsModal=="function"&&window.openSettingsModal();let I=0;const j=setInterval(()=>{I++,(_()||
I>=40)&&clearInterval(j)},150)}r(Ci,"openFeedbackById"),get("fb-submit").onclick=async()=>{const u=get(
"fb-title").value.trim(),f=get("fb-message").value.trim();if(!f){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const h=r(U=>{try{window.ActivityLog&&window.ActivityLog.log("change",{tab:"feedb\
ack",state:U})}catch{}},"logFeedback"),v={title:u,message:f,client:"web",version:window.CHAT_CONFIG&&
window.CHAT_CONFIG.appVersion||""},_=get("fb-attach-logs"),S=_&&_.checked&&window.ActivityLog?window.
ActivityLog.feedbackPayload():null;S&&(v.client_logs=S);const E=get("fb-attach-chat"),I=!!(E&&E.checked);
if(I&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}I&&(v.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const j=typeof window.showTopProgress=="function"?
window.showTopProgress("\u9001\u4FE1\u5185\u5BB9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"):null;
j==null||j.set(8,"\u9001\u4FE1\u5185\u5BB9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"),h("star\
t");try{j==null||j.set(22,"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059"),
j==null||j.ramp(58,"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059");
const U=await apiFetch("/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(v)}),W=U.ok?await U.clone().json().catch(()=>({})):{};j==null||j.set(U.ok?62:92,U.ok?"\u30D5\u30A3\u30FC\u30C9\
\u30D0\u30C3\u30AF\u3092\u53D7\u3051\u4ED8\u3051\u307E\u3057\u305F":"\u9001\u4FE1\u7D50\u679C\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059");
const se=si.files(),de={count:se.length,saved:!0};U.ok&&de.count&&(j==null||j.set(68,`\u753B\u50CF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059\uFF08${de.
count}\u679A\uFF09`),j==null||j.ramp(90,`\u753B\u50CF\u3092\u9001\u4FE1\u3057\u3066\u3044\u307E\u3059\uFF08${de.
count}\u679A\uFF09`),de.saved=await ii(`/api/feedback/${W.public_id}/images`,se)),j==null||j.set(92,
"\u9001\u4FE1\u7D50\u679C\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059");const me=U.ok&&W.public_id?
W:null;if(!(window.ActivityLog?await window.ActivityLog.reportFeedback(U,S,I,de,me?()=>Ci(me.public_id):
null):U.ok)){h("error"),j==null||j.fail("\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return}
h("done"),j==null||j.done("\u9001\u4FE1\u3057\u307E\u3057\u305F"),get("fb-title").value="",get("fb-m\
essage").value="",si.clear(),E&&(E.checked=!1),_&&(_.checked=!0),hn()}catch{h("error"),j==null||j.fail(
"\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F"),showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u306E\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};async function ai(u){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${u}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),hn()}r(ai,"deleteFeedback");const Ti="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Hn=[];function oi(u,f){const h=u.replies||[];return h.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${h.map(v=>{const _=v.role==="admin"?f&&v.author?`\u7BA1\u7406\u8005 (${v.
author})`:"\u7BA1\u7406\u8005":f?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=f?[v.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${v.
log_file}`,v.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${v.chat_file}`,v.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${v.image_count}\u679A\uFF09: ${v.image_dir}`]:[v.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${v.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(_)}\
 / ${escapeHtml(v.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${v.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(v.message)}</div>${S.filter(Boolean).map(E=>`<div cla\
ss="${f?"text-amber-300":"text-gray-400"}">${escapeHtml(E)}</div>`).join("")}</div>`}).join("")}</di\
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
reply-form");if(!h)return;const v=h.querySelector(".fb-reply-logs"),_=h.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());v.disabled=!S,_.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const E=ni(h.querySelector(".fb-reply-images-add"),h.querySelector(".fb-reply-images-input"),h.querySelector(
".fb-reply-images-list"));Hn.push(E);const I=h.querySelector(".fb-reply-send");I.onclick=async()=>{const j=h.
querySelector(".fb-reply-message").value.trim();if(!j){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const U={message:j,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},W=v.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;W&&(U.client_logs=W);const se=h.
querySelector(".fb-reply-chat").checked;if(se&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}se&&(U.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const de=E.files();I.disabled=!0;try{const me=await apiFetch(
`/api/feedback/${u.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(U)}),Z=await me.json().catch(()=>({}));if(!me.ok){showToast(Z.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const he=!de.length||await ii(`/api/feedback/${u.public_id}/replies/${Z.reply_id}\
/images`,de),ae=[];if(W&&Z.logs_saved===!1&&ae.push("\u30ED\u30B0"),he||ae.push("\u753B\u50CF"),se&&
Z.chat_copy_saved!==!0&&ae.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),ae.length)showToast(
`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${ae.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,
"error",!0);else{const Ee=["\u8FD4\u4FE1"];se&&Ee.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),
W&&Ee.push(`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof Z.logs_count=="number"?Z.
logs_count:W.entries.length}\u4EF6\uFF09`),de.length&&Ee.push(`\u753B\u50CF\uFF08${de.length}\u679A\uFF09`),
showToast(Ee.join(Ee.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","succe\
ss")}hn()}finally{I.disabled=!1}}}r(li,"bindFeedbackReply");function ci(){const u=get("fb-admin-stat\
us-filter").value;let f=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(h=>{
const v=u==="all"||(u==="unresolved"?["new","in_review","replied"].includes(h.dataset.feedbackStatus):
h.dataset.feedbackStatus===u);h.hidden=!v,v&&f++}),get("fb-admin-empty").classList.toggle("hidden",f!==
0)}r(ci,"filterAdminFeedback"),get("fb-admin-status-filter").onchange=ci;async function hn(){Hn.forEach(
S=>S.clear()),Hn=[];const f=await(await apiFetch("/api/feedback?all=1")).json(),h=get("fb-list");h.innerHTML=
"",(f.items||[]).filter(S=>!f.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(f.is_admin)
return;const E=document.createElement("div");E.className="p-2 rounded border border-gray-700 bg-gray\
-800/50",S.public_id&&(E.dataset.feedbackId=String(S.public_id)),E.innerHTML=`<div class="text-[11px\
] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(S.public_id||"")}</span> / ${S.
created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div><div class="\
text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class="text-[11px] text-gray-400 mt-\
1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non-released\uFF09":S.status)}\
</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.image_count}\
\u679A</div>`:""}${oi(S,!1)}${ri()}<div class="mt-2"><button type="button" class="fb-delete bg-red-7\
00 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,E.querySelector(
".fb-delete").onclick=()=>ai(S.public_id),li(S,E),h.appendChild(E)});const v=get("fb-admin-panel"),_=get(
"fb-admin-list");f.is_admin?(v.classList.remove("hidden"),_.innerHTML="",(f.items||[]).forEach(S=>{const E=document.
createElement("div");E.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",E.dataset.
feedbackStatus=S.status||"new",S.public_id&&(E.dataset.feedbackId=String(S.public_id)),E.innerHTML=`\

                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(S.public_id||"")}</span> / ${S.agent_name?`agent:${escapeHtml(S.agent_name)}`:`us\
er:${S.user_id}`} / ${S.created_at}${S.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(S.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div>
                            ${[S.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${S.log_file}`,S.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${S.chat_file}`,S.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${S.
image_count}\u679A\uFF09: ${S.image_dir}`].filter(Boolean).map(I=>`<div class="text-[11px] text-ambe\
r-300">${escapeHtml(I)}</div>`).join("")}
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
                        `,E.querySelector(".fb-status").value=S.status||"new",E.querySelector(".fb-s\
ave").onclick=async()=>{const I=E.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:I})}),
hn()},E.querySelector(".fb-delete").onclick=()=>ai(S.public_id),li(S,E),_.appendChild(E)}),ci()):v.classList.
add("hidden")}if(r(hn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=
f.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get(
"totp-verify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),h=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const v=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),_=await v.json();if(!v.ok){showToast(_.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const E=await(await ensureWebAuthnJson()).create({publicKey:_}),I=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},E,{name:h}))}),j=await I.json().catch(()=>({}));I.ok?(f&&(f.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(j.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(v){showToast(`WebAuthn Error: ${v}`,"error",!0)}finally{u&&(u.disabled=!1)}},window.
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
return}if(u.key==="Enter"){u.preventDefault();const v=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(f.value),f.value));v[slashSelectedIndex]?selectSlashCommand(v[slashSelectedIndex].
id):v.length>0&&selectSlashCommand(v[0].id);return}if(u.key==="Escape"){u.preventDefault(),hideSlashCommandSuggestions();
return}}if(skillSuggestionsVisible){const h=f.value.trim();if(u.key==="ArrowDown"){u.preventDefault(),
skillSelectedIndex=skillSelectedIndex+1,showSkillSuggestions(h.substring(1));return}if(u.key==="Arro\
wUp"){u.preventDefault(),skillSelectedIndex=Math.max(skillSelectedIndex-1,0),showSkillSuggestions(h.
substring(1));return}if(u.key==="Enter"){u.preventDefault();const v=h.substring(1).toLowerCase(),_=loadedSkills.
filter(S=>S.name.toLowerCase().includes(v)||S.description&&S.description.toLowerCase().includes(v));
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
items,h=[];for(let v=0;v<f.length;v++)if(f[v].kind==="file"){const _=f[v].getAsFile();_&&h.push(_)}h.
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
let Tt,Mn=[],qn=!1,Ln=null,dn=null,tn=null,yn=null,Gn=0,An=!1,vn=null,un=null,Kt=null,wn=null,En=null,
nn=null,$n=null,Pn=null;function di(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray($n)&&
$n.length)return $n;u.innerHTML="";const f=[];for(let h=0;h<24;h++){const v=document.createElement("\
span");v.className="block rounded-full",v.style.background="rgba(252, 165, 165, 0.92)",v.style.width=
"2px",v.style.transition="height 75ms linear, opacity 75ms linear",v.style.height="2px",v.style.opacity=
"0.4",f.push(v),u.appendChild(v)}return $n=f,f}r(di,"ensureMicWaveformBars");function sn(u,f="hidden"){
const h=get("mic-recording-indicator"),v=get("mic-recording-text");if(h){if(Pn&&(clearTimeout(Pn),Pn=
null),f==="hidden"){h.classList.add("hidden");return}v&&u&&(v.innerText=u),h.classList.remove("hidde\
n"),f==="recording"?h.style.color="rgb(252 165 165)":f==="processing"?h.style.color="rgb(253 224 71)":
h.style.color="rgb(209 213 219)"}}r(sn,"setMicRecordingIndicator");function ui(){di().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}r(ui,"resetMicWaveformBars");function pn(){if(En&&(cancelAnimationFrame(
En),En=null),wn){try{wn.disconnect()}catch{}wn=null}if(un){try{un.close()}catch{}un=null}Kt=null,nn=
null,ui()}r(pn,"stopMicWaveform");function Mi(u){pn();const f=di();if(!f.length)return;const h=window.
AudioContext||window.webkitAudioContext;if(!h)return;try{un=new h,Kt=un.createAnalyser(),Kt.fftSize=
256,Kt.smoothingTimeConstant=0,wn=un.createMediaStreamSource(u),wn.connect(Kt),nn=new Uint8Array(Kt.
frequencyBinCount)}catch{pn();return}const v=r(()=>{if(!Kt||!nn)return;Kt.getByteFrequencyData(nn);const _=Math.
max(1,Math.floor(nn.length/f.length));for(let S=0;S<f.length;S++){const I=(nn[Math.min(nn.length-1,S*
_)]||0)/255,j=Math.max(2,Math.round(2+I*10));f[S].style.height=`${j}px`,f[S].style.opacity=`${.35+I*
.65}`}En=requestAnimationFrame(v)},"render");v()}r(Mi,"startMicWaveform");function In(){if(Ln&&(clearInterval(
Ln),Ln=null),yn){try{yn.disconnect()}catch{}yn=null}if(dn){try{dn.close()}catch{}dn=null}tn=null}r(In,
"stopSilenceMonitor");function Li(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;In();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;dn=new f,tn=dn.createAnalyser(),tn.fftSize=2048,
yn=dn.createMediaStreamSource(u),yn.connect(tn);const h=new Uint8Array(tn.fftSize),v=getStsSilenceMs(),
_=.02;Gn=0,An=!1,Ln=setInterval(()=>{if(!tn)return;tn.getByteTimeDomainData(h);let S=0;for(let I=0;I<
h.length;I++){const j=(h[I]-128)/128;S+=j*j}if(Math.sqrt(S/h.length)>_){An||(An=!0),Gn=Date.now();return}
An&&Date.now()-Gn>v&&Tt&&Tt.state==="recording"&&Tt.stop()},200)}r(Li,"startSilenceMonitor");const zn=class zn{constructor(f=24e3){
const h=window.AudioContext||window.webkitAudioContext;this.ctx=new h({sampleRate:f}),this.nextStartTime=
0,this.bufferDelay=.1,this.started=!1}async addChunk(f){if(!this.ctx)return;const h=atob(f),v=new Uint8Array(
h.length);for(let U=0;U<h.length;U++)v[U]=h.charCodeAt(U);const _=new Int16Array(v.buffer),S=new Float32Array(
_.length);for(let U=0;U<_.length;U++)S[U]=_[U]/32768;const E=this.ctx.createBuffer(1,S.length,this.ctx.
sampleRate);E.getChannelData(0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const I=this.
ctx.createBufferSource();I.buffer=E,I.connect(this.ctx.destination),this.started||(this.nextStartTime=
this.ctx.currentTime+this.bufferDelay,this.started=!0);const j=Math.max(this.ctx.currentTime,this.nextStartTime);
I.start(j),this.nextStartTime=j+E.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r(zn,
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
[],this.userTurnBreak=!1,this.assistantTurnBreak=!1,setStsStatus("\u63A5\u7D9A\u4E2D...",!0);let v=!1;
if(isGeminiLiveModel()&&(v=await this._tryStartDirect(h)),!v){try{const _=await apiFetch("/api/realt\
ime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)}),S=await _.
json().catch(()=>({}));if(!_.ok)throw new Error(S.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=S.session_id,this.rateIn=S.rate_in||this.rateIn,this.rateOut=S.rate_out||this.rateOut}catch(_){
setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
_.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream()}this.active=!0,this.
capturing=!0;try{await this._startCapture()}catch(_){setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",
!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+_.message,"erro\
r",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.
add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",!0)}_buildDirectSetup(f){
const h=f.model,v=this.directTranscribe,_=this.directTranslate,S={responseModalities:v?["TEXT"]:["AU\
DIO"]},E={model:"models/"+h,generationConfig:S,sessionResumption:{}};return v?(E.inputAudioTranscription=
{mode:f.transcription_mode||"VERBATIM",languageCodes:[]},f.custom_vocabulary&&f.custom_vocabulary.length&&
(E.inputAudioTranscription.customVocabulary=f.custom_vocabulary)):(E.inputAudioTranscription={},E.outputAudioTranscription=
{},_?S.translationConfig={targetLanguageCode:normalizeGeminiTargetLang(f.target_lang),echoTargetLanguage:!1}:
f.voice&&(S.speechConfig={voiceConfig:{prebuiltVoiceConfig:{voiceName:f.voice}}}),this.directThinking&&
f.thinking_level&&(S.thinkingConfig={thinkingLevel:f.thinking_level,includeThoughts:!!f.include_thoughts})),
{setup:E}}async _tryStartDirect(f){let h=null;this.directModel=f.model,this.directTranscribe=isGeminiLiveTranscribeModel(),
this.directTranslate=isGeminiLiveTranslateModel(),this.directThinking=isGeminiLiveThinkingModel();const v={
model:f.model};this.directTranscribe?(v.transcription_mode=f.transcription_mode||"VERBATIM",v.custom_vocabulary=
f.custom_vocabulary||[]):(v.voice=f.voice,v.thinking_level=f.thinking_level,v.include_thoughts=f.include_thoughts,
v.target_lang=f.target_lang);try{const _=await apiFetch("/api/gemini/session",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify(v)}),S=await _.json().catch(()=>({}));if(!_.ok||
!S.direct||!S.token)return S.reason!=="vertex_ai"&&console.warn("Gemini Live: browser-direct unavail\
able, using the server session",S.reason||S.error||_.status),!1;h=S}catch(_){return console.warn("Ge\
mini Live: token request failed, using the server session",_),!1}this.rateIn=h.rate_in||16e3,this.rateOut=
h.rate_out||24e3,this.directUrl=h.url,this.directToken=h.token,this.directExpiresAt=Date.parse(h.expires_at||
"")||0,this.directSetup=this._buildDirectSetup(f),this.transcriptionMode=(f.transcription_mode||"VER\
BATIM").toUpperCase(),this.resumeHandle=null,this.directResumes=0,this.goAwayNotified=!1,this.usageMetadata=
null,this.directStartedAt=Date.now();try{return await this._connectDirect(),!0}catch(_){if(console.warn(
"Gemini Live: direct connection failed, using the server session",_),this.direct=!1,this.directSetup=
null,this.ws){try{this.ws.close()}catch{}this.ws=null}return this.rateIn=24e3,this.rateOut=24e3,!1}}_connectDirect(){
return new Promise((f,h)=>{let v=!1;const _=r(I=>{v||(v=!0,clearTimeout(S),I?h(I):f())},"finish"),S=setTimeout(
()=>_(new Error("\u63A5\u7D9A\u304C\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\u3057\u307E\u3057\u305F")),12e3);
let E;try{E=new WebSocket(this.directUrl+"?access_token="+encodeURIComponent(this.directToken))}catch(I){
_(I);return}this.ws=E,this.direct=!0,this.directReady=_,E.binaryType="arraybuffer",E.onopen=()=>{if(this.
ws!==E)return;const I=JSON.parse(JSON.stringify(this.directSetup));this.resumeHandle&&(I.setup.sessionResumption=
{handle:this.resumeHandle});try{E.send(JSON.stringify(I))}catch(j){_(j)}},E.onmessage=I=>{this.ws===
E&&this._handleDirectMessage(I)},E.onerror=()=>_(new Error("\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
E.onclose=I=>{this.ws===E&&this._directClosed(I)}})}_handleDirectMessage(f){let h=null;try{h=JSON.parse(
typeof f.data=="string"?f.data:new TextDecoder().decode(f.data))}catch{return}if(h.setupComplete!==void 0){
const I=this.directReady;this.directReady=null,I&&I()}const v=h.usageMetadata||h.usage_metadata;v&&typeof v==
"object"&&(this.usageMetadata=v);const _=h.serverContent;if(_){if(_.modelTurn&&(_.modelTurn.parts||[]).
forEach(I=>{const j=I.inlineData||{};j.data&&this._playDirectAudio(j.data),I.text&&(I.thought?this.assistantThought+=
I.text:this.directTranscribe||this._appendDirectAssistant(I.text))}),_.outputTranscription&&_.outputTranscription.
text&&this._appendDirectAssistant(_.outputTranscription.text),_.inputTranscription&&_.inputTranscription.
text){const I=String(_.inputTranscription.text);this.directTranscribe&&this.transcriptionMode==="SMA\
RT"?this.userTranscript=I:this.userTranscript+=(this.userTurnBreak&&this.userTranscript?`
`:"")+I,this.userTurnBreak=!1,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)}
_.turnComplete&&(this.userTurnBreak=!0,this.assistantTurnBreak=!0,this.responseDoneCount+=1)}const S=h.
interactionStatus||h.serverContent&&h.serverContent.interactionStatus;if(S){const I=String(S).toUpperCase();
setStsStatus(I==="IN_PROGRESS"?"\u8003\u3048\u4E2D...":"\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0),I==="IDLE"&&(this.responseDoneCount+=1)}const E=h.sessionResumptionUpdate;E&&E.resumable&&E.newHandle&&
(this.resumeHandle=E.newHandle),h.goAway&&!this.goAwayNotified&&(this.goAwayNotified=!0,showToast("G\
emini Live\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u4E0A\u9650\u304C\u8FD1\u3065\u3044\u3066\u3044\u307E\u3059\u3002\u307E\u3082\u306A\u304F\u81EA\u52D5\u3067\u7D42\u4E86\u3057\u307E\u3059\u3002",
"warning",!0)),h.error&&(this.streamError=h.error&&h.error.message||"Gemini Live \u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1))}_appendDirectAssistant(f){f&&(this.assistantTurnBreak&&this.assistantTranscript&&
!this.assistantTranscript.endsWith(`
`)&&(f=`
`+f),this.assistantTurnBreak=!1,this.assistantTranscript+=f,window.VoiceStudio&&window.VoiceStudio.log(
"assistant",this.assistantTranscript))}_playDirectAudio(f){this.lastAudioAt=Date.now();const h=atob(
f),v=new Uint8Array(h.length);for(let _=0;_<h.length;_++)v[_]=h.charCodeAt(_);this.assistantAudioBytes+
v.length<=20*1024*1024&&(this.assistantAudioChunks.push(v),this.assistantAudioBytes+=v.length),stsOpt(
"sts-auto-play")&&(this.rtPlayer||(this.rtPlayer=new xn(this.rateOut||24e3),an=this.rtPlayer),setStsStatus(
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
v=>{if(!v.ok)throw new Error("SSE stream failed ("+v.status+")");this.reader=v.body.getReader(),this.
_readLoop()}).catch(v=>{v&&v.name==="AbortError"||(this.streamError=v&&v.message?v.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",
this.active&&(setStsStatus("\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\
\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F","error",!0)))})}async _readLoop(){const f=new TextDecoder;
let h="";try{for(;this.reader;){const{done:v,value:_}=await this.reader.read();if(v)break;h+=f.decode(
_,{stream:!0});let S;for(;(S=h.indexOf(`

`))>=0;){const E=h.slice(0,S);h=h.slice(S+2);for(const I of E.split(`
`)){if(!I.startsWith("data: "))continue;let j=null;try{j=JSON.parse(I.slice(6))}catch{continue}this.
_handleEvent(j)}}}}catch(v){if(v&&v.name==="AbortError")return;this.active&&(this.streamError=v&&v.message?
v.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
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
!0);break;case"interaction_status":this.active&&setStsStatus(f.status==="IN_PROGRESS"?"\u8003\u3048\u4E2D...":
"\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",!0);break;case"notice":f.message&&showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\
\u58F0: "+f.message,"warning",!0);break;case"error":this.streamError=f.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}an=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(fi()).then(f=>{if(this.stream=f,this.direct&&window.MediaRecorder)try{this.
userAudioChunks=[];try{this.userRecorder=new MediaRecorder(f,{audioBitsPerSecond:32e3})}catch{this.userRecorder=
new MediaRecorder(f)}this.userRecorder.ondataavailable=I=>{I.data&&I.data.size&&this.userAudioChunks.
push(I.data)},this.userRecorder.start(500)}catch{this.userRecorder=null}const h=this.rateIn||24e3,v=Ai(
f,h);this.audioCtx=v.ctx;const _=v.source,S=this.audioCtx.sampleRate,E=4096;this.processor=this.audioCtx.
createScriptProcessor(E,1,1),this.processor.onaudioprocess=I=>{if(!this.active||!this.capturing)return;
const j=I.inputBuffer.getChannelData(0),U=Ei(j,S,h);!U||!U.byteLength||this._sendAudio(U)},_.connect(
this.processor),this.processor.connect(this.audioCtx.destination)})}_sendAudio(f){if(this.active){if(this.
direct){if(!this.ws||this.ws.readyState!==WebSocket.OPEN)return;try{this.ws.send(JSON.stringify({realtimeInput:{
audio:{data:pi(new Uint8Array(f)),mimeType:"audio/pcm;rate="+(this.rateIn||16e3)}}}))}catch{}return}
this.sessionId&&(this.audioQueue.push(new Uint8Array(f)),this.audioFlush||(this.audioFlush=this._flushAudio()))}}async _flushAudio(){
try{for(;this.audioQueue.length&&this.sessionId;){const f=this.audioQueue.splice(0,this.audioQueue.length),
h=f.reduce((j,U)=>j+U.byteLength,0),v=new Uint8Array(h);let _=0;f.forEach(j=>{v.set(j,_),_+=j.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),E={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:v.buffer},
I=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(E):E;try{await fetch(S,I)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
userRecorder){const f=this.userRecorder;this.userRecorder=null,this.userRecorderDone=new Promise(h=>{
let v=!1;const _=r(()=>{v||(v=!0,h())},"finish");f.addEventListener("stop",_,{once:!0});try{f.stop()}catch{
_()}setTimeout(_,1500)})}if(this.stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.
stream=null}if(this.audioCtx){try{this.audioCtx.close()}catch{}this.audioCtx=null}In(),pn()}async stop(){
if(!this.active)return;if(this.active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\
\u3059...",!0),this.audioFlush)try{await this.audioFlush}catch{}if(this.direct){if(this.ws&&this.ws.
readyState===WebSocket.OPEN)try{this.ws.send(JSON.stringify({realtimeInput:{audioStreamEnd:!0}}))}catch{}}else
try{await apiFetch("/api/realtime/commit",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({session_id:this.sessionId})})}catch{}const f=Date.now(),h=this.responseDoneCount;
let v=this.lastAudioAt;for(;Date.now()-f<2e4&&!(this.responseDoneCount>h||(this.lastAudioAt>v&&(v=this.
lastAudioAt),!this.speechActive&&Date.now()-f>2e3&&Date.now()-v>2500));)await new Promise(_=>setTimeout(
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
let f="";if(this.assistantAudioChunks.length){const S=this.assistantAudioChunks.reduce((j,U)=>j+U.byteLength,
0),E=new Uint8Array(S);let I=0;this.assistantAudioChunks.forEach(j=>{E.set(j,I),I+=j.byteLength}),f=
pi(E)}if(this.userRecorderDone)try{await this.userRecorderDone}catch{}const h=this.userAudioChunks.length?
await $i(new Blob(this.userAudioChunks)):"",v=await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,model:this.directModel||
get("model-select").value,user_text:this.userTranscript,assistant_text:this.assistantTranscript,assistant_thought:this.
assistantThought,audio_base64:f,user_audio_base64:h,usage_metadata:this.usageMetadata||null,duration_ms:this.
directStartedAt?Math.max(0,Date.now()-this.directStartedAt):null})}),_=await v.json().catch(()=>({}));
if(!v.ok)throw new Error(_.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");_.thread_id&&
!currentThreadId&&(currentThreadId=String(_.thread_id))}_cancel(){if(this.direct){if(this.active=!1,
this.ws){try{this.ws.close(1e3)}catch{}this.ws=null}}else this.sessionId&&apiFetch("/api/realtime/ca\
ncel",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.
sessionId})}).catch(()=>{});this._cleanup(),setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.capturing=!1,this.stopping=!1,this.audioQueue=
[],this.ws){try{this.ws.close(1e3)}catch{}this.ws=null}if(this.direct=!1,this.directReady=null,this.
directSetup=null,this.resumeHandle=null,this.assistantAudioChunks=[],this.assistantAudioBytes=0,this.
userAudioChunks=[],this.userRecorderDone=null,this._stopCapture(),this._stopPlayback(),this.abortCtrl){
try{this.abortCtrl.abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const f=get(
"mic-btn");f&&(f.classList.remove("bg-red-600","animate-pulse"),f.classList.add("bg-gray-700"))}};r(
Vn,"RealtimeVoiceSession");let Un=Vn;function Ai(u,f){const h=window.AudioContext||window.webkitAudioContext;
if(!h)throw new Error("AudioContext not supported");let v=null;try{return v=new h({sampleRate:f}),{ctx:v,
source:v.createMediaStreamSource(u)}}catch{if(v)try{v.close()}catch{}return v=new h,{ctx:v,source:v.
createMediaStreamSource(u)}}}r(Ai,"openMicAudioSource");function Ei(u,f,h){let v=u;if(f!==h&&f>0&&h>
0){const S=f/h,E=Math.floor(v.length/S),I=new Float32Array(E);for(let j=0;j<E;j++)I[j]=v[Math.min(Math.
floor(j*S),v.length-1)];v=I}const _=new Int16Array(v.length);for(let S=0;S<v.length;S++){const E=Math.
max(-1,Math.min(1,v[S]));_[S]=E<0?E*32768:E*32767}return _.buffer}r(Ei,"pcm16FromFloat32");function pi(u){
let f="";for(let v=0;v<u.length;v+=32768)f+=String.fromCharCode.apply(null,u.subarray(v,v+32768));return btoa(
f)}r(pi,"bytesToBase64");function $i(u){return new Promise((f,h)=>{const v=new FileReader;v.onload=()=>f(
String(v.result||"").split(",")[1]||""),v.onerror=()=>h(v.error||new Error("\u97F3\u58F0\u306E\u8AAD\u307F\u53D6\u308A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
v.readAsDataURL(u)})}r($i,"blobToBase64");const kn=new Un,mi=(()=>{const h={idle:"bg-gray-600",connecting:"\
bg-amber-500 animate-pulse",streaming:"bg-emerald-600 animate-pulse",paused:"bg-amber-500",stopped:"\
bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};let v=null,_=null,S=!1,E=null,I=!1,j=0,U=0,W=null,
se="idle",de=!1,me=null;const Z=r(K=>document.getElementById(K),"$"),he=r(K=>{const oe=Object.assign(
{},K||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="functi\
on"?window.ProgressSpinner.manualRequestOptions(oe):(oe.progressSpinner=!1,oe)},"noSpinner");function ae(K,oe){
se=oe;const Ie=Z("lyria-status-text"),be=Z("lyria-status-dot");Ie&&(Ie.textContent=K),be&&(be.className=
"w-2 h-2 rounded-full inline-block "+(h[oe]||h.idle)),je(),pe()}r(ae,"setStatus");function Ee(){const K=U?
Math.floor((Date.now()-U)/1e3):0,oe=String(Math.floor(K/60)).padStart(2,"0"),Ie=String(K%60).padStart(
2,"0");return`${oe}:${Ie}`}r(Ee,"formatElapsed");function le(){U||(U=Date.now());const K=Z("lyria-el\
apsed");K&&(K.textContent=Ee()),W||(W=window.setInterval(()=>{const oe=Z("lyria-elapsed");oe&&(oe.textContent=
Ee())},1e3))}r(le,"startElapsedTimer");function Te(){W&&(window.clearInterval(W),W=null)}r(Te,"stopE\
lapsedTimer");function je(){const K=Z("lyria-play-btn"),oe=Z("lyria-pause-btn"),Ie=Z("lyria-stop-btn"),
be=Z("lyria-reset-btn"),Ne=!!v,Ze=se==="streaming"||se==="connecting";if(K){K.disabled=de;const ht=K.
querySelector("i");ht&&(ht.className="fas fa-play")}oe&&(oe.disabled=de||!Ze),Ie&&(Ie.disabled=de||!Ne||
!Ze),be&&(be.disabled=de||!Ne||!Ze)}r(je,"updateTransportButtons");function pe(){const K=Z("lyria-sa\
ve-btn");if(!K)return;const oe=!!v&&se!=="idle"&&se!=="connecting"&&se!=="error";K.classList.toggle(
"hidden",!oe)}r(pe,"updateSaveButton");function _e(K,oe){const Ie=Z("lyria-prompt-rows");if(!Ie)return;
const be=document.createElement("div");be.className="flex items-center gap-2",be.innerHTML=`
                        <input type="text" value="${escapeHtml(K||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof oe=="nu\
mber"?oe:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof oe=="number"?oe:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Ne=be.querySelector('input[type="range"]'),Ze=be.querySelector(".lyria-w\
eight-label");Ne&&Ze&&Ne.addEventListener("input",()=>{Ze.textContent=parseFloat(Ne.value).toFixed(1)});
const ht=be.querySelector(".lyria-prompt-remove");ht&&ht.addEventListener("click",()=>{Ie.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||be.remove()}),be.classList.add("lyria-prompt-row-wrap"),Ie.appendChild(
be)}r(_e,"addPromptRow");function Fe(){const K=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),oe=[];return K.forEach(Ie=>{const be=Ie.querySelector('input[type="text"]'),Ne=Ie.
querySelector('input[type="range"]'),Ze=(be?be.value:"").trim();Ze&&oe.push({text:Ze,weight:parseFloat(
Ne?Ne.value:1)||1})}),oe}r(Fe,"collectPrompts");function De(){const K={},oe=r(ji=>{const Wn=Z(ji);return Wn&&
Wn.value!==""?parseFloat(Wn.value):void 0},"num"),Ie=oe("lyria-bpm");Ie!==void 0&&(K.bpm=Math.round(
Ie));const be=oe("lyria-guidance");be!==void 0&&(K.guidance=be);const Ne=oe("lyria-density");Ne!==void 0&&
(K.density=Ne);const Ze=oe("lyria-brightness");Ze!==void 0&&(K.brightness=Ze);const ht=oe("lyria-tem\
perature");ht!==void 0&&(K.temperature=ht);const rt=Z("lyria-scale");rt&&rt.value&&(K.scale=rt.value);
const yt=Z("lyria-mode");yt&&yt.value&&(K.music_generation_mode=yt.value);const It=Z("lyria-mute-bas\
s"),rn=Z("lyria-mute-drums"),vi=Z("lyria-only-bass-drums");return It&&(K.mute_bass=It.checked),rn&&(K.
mute_drums=rn.checked),vi&&(K.only_bass_and_drums=vi.checked),K}r(De,"collectConfig");function _t(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([oe,Ie])=>{const be=Z(oe),Ne=Z(Ie);!be||!Ne||be.addEventListener("input",()=>{const Ze=parseFloat(
be.value);Ne.textContent=oe==="lyria-bpm"?String(Math.round(Ze)):Ze.toFixed(1)})})}r(_t,"bindRangeLa\
bels");function ct(){if(E){try{E.close()}catch{}E=null}I=!1,j=0}r(ct,"resetPlayback");function Pt(){
if(S=!1,_&&typeof _.abort=="function")try{_.abort()}catch{}_=null}r(Pt,"closeStream");async function Xt(){
Pt(),_=new AbortController,S=!0;try{const K=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(
v)}`,he({method:"GET",signal:_.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!K.
ok){const Ne=await K.json().catch(()=>({}));throw new Error(Ne.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const oe=K.body.getReader(),Ie=new TextDecoder;let be="";for(;S;){const{done:Ne,value:Ze}=await oe.read();
if(Ne)break;be+=Ie.decode(Ze,{stream:!0});const ht=be.split(`

`);be=ht.pop();for(const rt of ht){const yt=rt.split(`
`).find(rn=>rn.startsWith("data: "));if(!yt)continue;const It=yt.slice(6);try{const rn=JSON.parse(It);
Pe(rn)}catch{}}}}catch(K){if(K&&K.name==="AbortError")return;S&&(ae("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&v&&Xt()},1200))}finally{S=!1}}r(Xt,"openStream");function Pe(K){
if(K&&K.snapshot){const oe=K.status;if(oe==="error"){ae("\u30A8\u30E9\u30FC","error"),Te();return}if(oe===
"closed"||oe==="stopped"){ae("\u7D42\u4E86","closed"),Te();return}ae(oe==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",oe==="paused"?"paused":"connecting");return}if(K&&K.audio){ae("\u518D\u751F\u4E2D...",
"streaming"),le(),Me(K.audio);return}if(K&&K.error){ae("\u30A8\u30E9\u30FC: "+K.error,"error"),Te();
return}if(K&&K.final){ae("\u7D42\u4E86","closed"),Te(),je();return}}r(Pe,"handleStreamMessage");function Me(K){
if(!K)return;if(!E){const rt=window.AudioContext||window.webkitAudioContext;if(!rt)return;E=new rt({
sampleRate:48e3}),I=!1,j=0}let oe;try{const rt=atob(K);oe=new Uint8Array(rt.length);for(let yt=0;yt<
rt.length;yt++)oe[yt]=rt.charCodeAt(yt)}catch{return}const Ie=new Int16Array(oe.buffer),be=Math.floor(
Ie.length/2);if(be<1)return;const Ne=E.createBuffer(2,be,48e3);for(let rt=0;rt<2;rt++){const yt=Ne.getChannelData(
rt);for(let It=0;It<be;It++)yt[It]=Ie[It*2+rt]/32768}E.state==="suspended"&&E.resume();const Ze=E.createBufferSource();
Ze.buffer=Ne,Ze.connect(E.destination),I||(j=E.currentTime+.08,I=!0);const ht=Math.max(E.currentTime,
j);Ze.start(ht),j=ht+Ne.duration}r(Me,"playChunk");async function Ye(K,oe){const Ie=await fetch("/ap\
i/gemini/music/command",he({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:v,type:K},oe||{}))})),be=await Ie.json().catch(()=>({}));if(!Ie.ok)throw new Error(
be.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return be}
r(Ye,"apiCommand");async function gt(){if(de)return;const K=Fe();if(!K.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}de=!0,je(),ae("\u63A5\u7D9A\u4E2D...","connecting");try{
const oe=await fetch("/api/gemini/music/start",he({method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({weighted_prompts:K,config:De()})})),Ie=await oe.json().catch(()=>({}));
if(!oe.ok)throw new Error(Ie.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
v=Ie.session_id,me=De(),ae("\u63A5\u7D9A\u4E2D...","connecting"),Xt()}catch(oe){ae("\u30A8\u30E9\u30FC: "+
oe.message,"error"),showToast("Lyria RealTime: "+oe.message,"error",!0)}finally{de=!1,je()}}r(gt,"st\
artSession");async function St(K){if(v){de=!0,je();try{await Ye("control",{action:K}),K==="PLAY"?ae(
"\u518D\u751F\u4E2D...","streaming"):K==="PAUSE"?ae("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):K==="\
STOP"?ae("\u505C\u6B62\u4E2D","stopped"):K==="RESET_CONTEXT"&&ae("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(oe){showToast("Lyria RealTime: "+oe.message,"error",!0),ae("\u30A8\u30E9\u30FC: "+
oe.message,"error")}finally{de=!1,je()}}}r(St,"control");async function Yt(){if(!v)return;const K=Fe();
if(!K.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}de=!0;try{await Ye("prompts",{weighted_prompts:K}),ae("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
se==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(oe){showToast("Lyria RealTime: "+oe.message,"error",!0)}finally{de=!1,je()}}r(Yt,"a\
pplyPrompts");async function Ae(){if(!v)return;const K=De(),oe=me||{},Ie=K.bpm!==void 0&&K.bpm!==oe.
bpm,be=K.scale!==void 0&&K.scale!==oe.scale,Ne=Ie||be;de=!0;try{await Ye("config",{config:K,reset_context:Ne}),
me=K,ae(Ne?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",se==="paused"?"paused":"streaming"),showToast(
Ne?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Ze){showToast("Lyria RealT\
ime: "+Ze.message,"error",!0)}finally{de=!1,je()}}r(Ae,"applyConfig");async function Je(){if(v){de=!0,
ae("\u4FDD\u5B58\u4E2D...","connecting"),je();try{const K=await fetch("/api/gemini/music/save",he({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:v,thread_id:currentThreadId||
null})})),oe=await K.json().catch(()=>({}));if(!K.ok)throw new Error(oe.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
ae("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),Te(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),oe.thread_id&&(currentThreadId=String(oe.thread_id),history.pushState({},"","/c/"+oe.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(oe.thread_id||currentThreadId),on(
!0)}catch(K){ae("\u30A8\u30E9\u30FC: "+K.message,"error"),showToast("Lyria RealTime: "+K.message,"er\
ror",!0)}finally{de=!1,je()}}}r(Je,"saveSession");async function ze(){if(Pt(),v)try{await fetch("/ap\
i/gemini/music/cancel",he({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:v})}))}catch{}v=null,Te(),ct(),ae("\u6E96\u5099\u5B8C\u4E86","idle")}r(ze,"cancelSession");
function Qe(){const K=Z("lyria-prompt-rows");K&&(K.innerHTML=""),_e("",1),me=null,U=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(be=>{const Ne=Z(be);
Ne&&(Ne.value=be==="lyria-bpm"?"120":be==="lyria-guidance"?"4":be==="lyria-temperature"?"1.1":"0.5")});
const oe=Z("lyria-scale");oe&&(oe.value="");const Ie=Z("lyria-mode");Ie&&(Ie.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(be=>{const Ne=Z(be);Ne&&(Ne.checked=
!1)}),_t()}r(Qe,"resetControls");function on(K){Pt(),v&&fetch("/api/gemini/music/cancel",he({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:v})})).catch(()=>{}),
v=null,S=!1,Te(),ct(),hideModal("lyria-studio-modal")}r(on,"closeAndCleanup");function Sn(K){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const oe=Z("lyria-studio-modal");if(oe&&oe.classList.contains("modal-open")&&v){
if(K&&typeof K=="string"){const be=Z("lyria-prompt-rows");be&&(be.innerHTML=""),_e(K,1)}return}if(v&&
ze(),Qe(),K&&typeof K=="string"){const be=Z("lyria-prompt-rows");be&&(be.innerHTML=""),_e(K,1)}v=null,
S=!1,Te(),ct(),ae("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(Sn,"open");function On(){
const K=Z("lyria-open-studio-btn");K&&K.addEventListener("click",()=>Sn(""));const oe=Z("lyria-studi\
o-close");oe&&oe.addEventListener("click",()=>on(!1));const Ie=Z("lyria-play-btn");Ie&&Ie.addEventListener(
"click",()=>{if(!v){gt();return}St("PLAY")});const be=Z("lyria-pause-btn");be&&be.addEventListener("\
click",()=>St("PAUSE"));const Ne=Z("lyria-stop-btn");Ne&&Ne.addEventListener("click",()=>St("STOP"));
const Ze=Z("lyria-reset-btn");Ze&&Ze.addEventListener("click",()=>St("RESET_CONTEXT"));const ht=Z("l\
yria-add-prompt-btn");ht&&ht.addEventListener("click",()=>_e("",1));const rt=Z("lyria-apply-prompts-\
btn");rt&&rt.addEventListener("click",Yt);const yt=Z("lyria-apply-config-btn");yt&&yt.addEventListener(
"click",Ae);const It=Z("lyria-save-btn");It&&It.addEventListener("click",Je),_t(),Qe(),window.openLyriaStudio=
Sn}r(On,"init");function Cn(){de||on(!1)}return r(Cn,"requestClose"),{init:On,open:Sn,requestClose:Cn}})();
mi.init(),window.closeLyriaStudio=()=>mi.requestClose(),(()=>{let u=null,f=null,h=null;const v="voic\
eDockSettingsOpen",_="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=r(pe=>document.getElementById(pe),"$");function E(){return isStsModel()&&voiceStudioUiEnabled!==!1}
r(E,"isStudioMode");function I(){const pe=get("model-select")?get("model-select").value:"",_e=S("voi\
ce-studio-title");_e&&(pe==="gpt-transcribe"||pe==="gpt-live-transcribe"||pe==="grok-voice-transcrib\
e-2.0-file"?_e.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":pe===
"gemini-3.5-live-translate-preview"?_e.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
_e.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(I,"updateTitle");function j(){const pe=S("v\
oice-studio-transcript");pe&&(pe.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${_}</div>`);const _e=S("sts-live-transcript");_e&&(_e.innerHTML="",_e.classList.add("hidden"))}
r(j,"resetTranscript");function U(pe,_e,Fe){const De=pe.querySelectorAll(".voice-studio-line");let _t=null;
for(let ct=De.length-1;ct>=0;ct--)if(De[ct].dataset.role===_e){_t=De[ct];break}if(_t)_t.innerHTML=Fe;else{
const ct=pe.querySelector(".voice-studio-placeholder");ct&&ct.remove();const Pt=document.createElement(
"div");Pt.className="voice-studio-line",Pt.dataset.role=_e,Pt.innerHTML=Fe,pe.appendChild(Pt)}pe.classList.
remove("hidden"),pe.scrollTop=pe.scrollHeight}r(U,"writeLine");function W(pe,_e){if(!_e||!String(_e).
trim()||!E())return;const _t=`<span class="${pe==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(pe==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
_e)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(ct=>U(ct,
pe,_t))}r(W,"log");function se(pe,_e=!0){const Fe=S("sts-panel"),De=S("sts-settings-toggle");if(Fe&&
Fe.classList.toggle("settings-open",!!pe),De&&De.setAttribute("aria-expanded",pe?"true":"false"),_e)
try{localStorage.setItem(v,pe?"1":"0")}catch{}}r(se,"setSettingsOpen");function de(){try{return localStorage.
getItem(v)==="1"}catch{return!1}}r(de,"readSettingsOpen");let me=null;function Z(){const pe=get("mod\
el-select")?get("model-select").value:"";pe!==me&&(me=pe,j()),se(de(),!1),I()}r(Z,"syncDock");function he(){
const pe=S("sts-panel"),_e=S("voice-studio-panel-host");pe&&_e&&pe.parentNode!==_e&&(u=pe.parentNode,
f=pe.nextSibling,_e.appendChild(pe));const Fe=S("file-preview"),De=S("voice-studio-file-host");Fe&&De&&
Fe.parentNode!==De&&(h=Fe.parentNode,De.appendChild(Fe),De.classList.remove("hidden"))}r(he,"movePan\
elIntoModal");function ae(){const pe=S("sts-panel");pe&&u&&pe.parentNode!==u&&(f&&f.parentNode===u?u.
insertBefore(pe,f):u.appendChild(pe));const _e=S("file-preview");_e&&h&&_e.parentNode!==h&&h.appendChild(
_e);const Fe=S("voice-studio-file-host");Fe&&Fe.classList.add("hidden"),u=null,f=null,h=null}r(ae,"m\
ovePanelBack");function Ee(){if(!E())return;he();const pe=S("sts-panel");pe&&pe.classList.remove("hi\
dden"),I(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}r(Ee,"open");function le(){window.
VoiceStudioOpen=!1,ae(),hideModal("voice-studio-modal")}r(le,"close");function Te(){window.VoiceStudioOpen&&
le()}r(Te,"closeIfOpen");function je(){window.VoiceStudioOpen=!1;const pe=S("voice-studio-open-btn");
pe&&pe.addEventListener("click",()=>Ee());const _e=S("voice-studio-close");_e&&_e.addEventListener("\
click",()=>le());const Fe=S("sts-settings-toggle");Fe&&Fe.addEventListener("click",()=>{const De=S("\
sts-panel");se(!(De&&De.classList.contains("settings-open")))}),window.VoiceStudio={open:Ee,close:le,
closeIfOpen:Te,log:W,isStudioMode:E,syncDock:Z},Z()}return r(je,"init"),{init:je}})().init();let an=null;
function Pi(){if(an&&(an.stop(),an=null),vn){try{vn.pause()}catch{}try{vn.src=""}catch{}vn=null}}r(Pi,
"stopStsPlayback");async function Bi(u){Pi();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=
!0,f.playsInline=!0,vn=f,await f.play(),new Promise(h=>{f.onended=()=>h("ended"),f.onerror=()=>h("er\
ror")})}r(Bi,"playStsAudio");function Ii(){if(kn.isActive()){kn._cancel();return}Tt&&Tt.state==="rec\
ording"&&(qn=!0,Tt.stop())}r(Ii,"cancelRecording");function fi(){if(isStsModel())return{audio:!0};const f=navigator.
mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.getSupportedConstraints():
{},h={channelCount:1};return f.echoCancellation&&(h.echoCancellation=!1),f.noiseSuppression&&(h.noiseSuppression=
!1),f.autoGainControl&&(h.autoGainControl=!1),{audio:h}}r(fi,"getMicCaptureConstraints"),get("mic-bt\
n").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(kn.isActive()){get("mic-btn").classList.remove("bg-red-600","animate-pulse"),
get("mic-btn").classList.add("bg-gray-700"),kn.stop();return}if(Tt&&Tt.state==="recording"){Tt.stop(),
get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-\
700"),isStsModel()||sn("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026","processing"),isStsModel()&&setStsStatus(
"Processing...",!0);return}try{if(isStsModel())try{const h=new Audio;h.src="data:audio/wav;base64,Uk\
lGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",h.play().catch(()=>{})}catch{}if(isRealtimeSessionModel()){
await kn.start();return}isStsModel()||(ui(),sn("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const u=await navigator.mediaDevices.getUserMedia(fi());Tt=new MediaRecorder(u),Mn=[],qn=!1;const f=isStsModel();
Tt.ondataavailable=h=>Mn.push(h.data),Tt.onstop=async()=>{if(qn){Mn=[],get("file-preview").classList.
add("hidden"),u.getTracks().forEach(E=>E.stop()),In(),pn(),f||(sn("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Pn=setTimeout(()=>sn("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const h=new Blob(Mn,{type:"audio/web\
m"}),v=new File([h],"recording.webm",{type:"audio/webm"}),_=new FormData;_.append("file",v),get("fil\
e-preview").classList.remove("hidden");const S=f;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const ae=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=ae.id!==null&&ae.id!==void 0?String(ae.id):ae.id,setTemporaryChatUiState(!!(ae&&
ae.is_temporary)),setCurrentChatHeaderTitle(ae&&ae.title),applyTemporaryChatRuntimeMeta(ae||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+ae.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeSkill&&(threadSkillMap[currentThreadId]=activeSkill,pendingSkillForNewThread=null),_.append("m\
odel",get("model-select").value),_.append("thread_id",currentThreadId),get("sts-voice")&&_.append("s\
ts_voice",get("sts-voice").value||""),get("sts-reasoning-effort")&&_.append("sts_reasoning_effort",get(
"sts-reasoning-effort").value||""),get("sts-speed")&&_.append("sts_speed",get("sts-speed").value||""),
get("sts-rate-in")&&_.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&_.append(
"sts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&_.append("sts_thinking_leve\
l",get("sts-thinking-level").value||""),get("sts-include-thoughts")&&_.append("sts_include_thoughts",
get("sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const E=await apiFetch(
"/sts",{method:"POST",body:_});if(!E.ok){const he=await E.json().catch(()=>({}));throw new Error(he.
error||"Speech-to-speech failed")}const I=E.body.getReader(),j=new TextDecoder;let U="",W=null,se=null;
stsOpt("sts-auto-play")&&(se=new xn,an=se),setStsStatus(isTranscriptionModel()?"Transcribing...":"Pr\
ocessing audio...",!0);let de=!0,me="",Z="";for(;;){const{done:he,value:ae}=await I.read();if(he)break;
U+=j.decode(ae,{stream:!0});const Ee=U.split(`
`);U=Ee.pop();for(const le of Ee){if(!le.trim())continue;const Te=JSON.parse(le);if(Te.error)throw new Error(
Te.error);Te.audio_delta&&se&&(de&&(setStsStatus("Playing response...",!1),de=!1),await se.addChunk(
Te.audio_delta)),Te.input_delta&&(me+=Te.input_delta,window.VoiceStudio&&window.VoiceStudio.log("use\
r",me)),Te.transcript_delta&&(Z+=Te.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("ass\
istant",Z)),(Te.final||Te.audio_url)&&(W=Te)}}window.VoiceStudio&&!me.trim()&&window.VoiceStudio.log(
"user","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),W&&(W.audio_url||W.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const E=get(
"set-mic-transcribe-mode");if(!!(E&&E.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}_.append("llm_model",get("model-select")&&get("model-select").value||"");const U=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:_})).json();if(U.transcript){const W=get("prompt-inp\
ut");W.value+=(W.value?" ":"")+U.transcript,W.style.height="auto",W.style.height=W.scrollHeight+"px"}else
showToast(U.error||"Transcription failed","error",!0)}}catch(E){showToast("Audio processing error: "+
E.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(E=>E.
stop()),In(),pn(),S||sn("","hidden"),S&&setStsStatus("Tap to speak",!1)}},Tt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(sn("\u9332\u97F3\u4E2D\u2026","recording"),Mi(u)),Li(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{pn(),isStsModel()||sn("","hidden"),alert("Microphone access denied or not avail\
able.")}};const _n=r((u,f)=>{if(!u)return;const h=u.querySelector("span");h?h.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),h=get("lib-download-btn"),v=get("lib-attach-btn"),_=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(f&&(f.disabled=u===0,_n(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
h&&(h.disabled=u===0,_n(h,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
v&&(v.disabled=u===0,_n(v,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),_&&(_.disabled=u!==1,_n(_,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=u!==1,_n(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const E=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",E&&u>0)}};function gi(u){lib.
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
"skill-inst").value;if(u&&h){const v=editingSkillUuid?"PUT":"POST",_=editingSkillUuid?`/api/skills/${editingSkillUuid}`:
CHAT_CONFIG.urls.handleSkills;await apiFetch(_,{method:v,headers:{"Content-Type":"application/json"},
body:JSON.stringify({name:u,description:f,instruction:h})}),window.closeSkillModal(),loadSkills(),editingSkillUuid&&
activeSkill&&activeSkill.uuid===editingSkillUuid&&(activeSkill.name=u,activeSkill.instruction=h,applyActiveSkill(
activeSkill))}else alert("Name and Instruction are required.")},document.addEventListener("click",function(u){
if(u.target.closest(".edit-btn")){const h=u.target.closest(".edit-btn").getAttribute("data-id");beginEditMessage(
h)}if(u.target.closest(".code-toggle")){const f=u.target.closest(".code-toggle"),h=f.closest(".code-\
wrapper");if(!h)return;const v=h.classList.toggle("collapsed");h.setAttribute("data-collapsed",v?"tr\
ue":"false"),f.setAttribute("aria-expanded",v?"false":"true"),f.innerHTML=v?'<i class="fas fa-chevro\
n-down"></i>':'<i class="fas fa-chevron-up"></i>',f.title=v?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",
f.setAttribute("aria-label",v?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080")}if(u.target.closest("\
.download-btn")){const f=u.target.closest(".download-btn"),h=f.getAttribute("data-code"),v=(f.getAttribute(
"data-lang")||"txt").toLowerCase();if(h)try{const _=decodeURIComponent(h),S=new Blob([_],{type:"text\
/plain"}),E=URL.createObjectURL(S),I=document.createElement("a");I.href=E;let U={python:"py",javascript:"\
js",typescript:"ts",markdown:"md",html:"html",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"\
sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"\
go",rust:"rs",ruby:"rb",php:"php",perl:"pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"\
toml",ini:"ini",plaintext:"txt",text:"txt"}[v]||v;(v.length>8||/[^a-z0-9]/.test(v))&&(U="txt");let W=`\
code.${U}`;v==="dockerfile"&&(W="Dockerfile"),v==="makefile"&&(W="Makefile"),I.download=W,document.body.
appendChild(I),I.click(),document.body.removeChild(I),URL.revokeObjectURL(E)}catch(_){console.error(
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
"#upload-list .upload-row");a.innerHTML="",T.forEach((A,N)=>{const G=A.getAttribute("data-local-url"),
L=A.getAttribute("data-filename"),$=A.querySelector("img.upload-preview")!==null;let M;if($){let P=G;
if(!P&&L){const R=L.replace(/^\d+\//,"");P=buildAttachmentPreviewUrl(R)}P&&(M=document.createElement(
"img"),M.src=P,M.className="thumb-item shadow-sm",M.dataset.viewerSrc=P,M.dataset.viewerFilename=L||
P.split("/").pop(),M.uploadRow=A,M.onclick=function(R){R.preventDefault(),openImageViewer(this.dataset.
viewerSrc,".thumb-item")},M.onerror=function(){this.parentElement.replaceChild(d("ERR"),this)})}M||(M=
d("FILE")),M.style.animationDelay=`${N*32}ms`,a.appendChild(M)}),T.length>0?a.classList.remove("hidd\
en"):a.classList.add("hidden")}function d(T){const A=document.createElement("div");return A.className=
"thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-sm font-bol\
d",A.innerText=T,A}r(d,"createFileThumb");const m=collectImageUrlsForSend(),g=uploadProgressState.total,
y=uploadProgressState.completed,b=uploadProgressState.active;g===0&&(e.classList.add("hidden"),n&&n.
classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const w=get("send\
-btn"),x=get("mic-btn"),k=get("mask-btn"),C=isStopMode;if(b>0?(w&&(w.disabled=!0),x&&(x.disabled=!0),
k&&(k.disabled=!0)):C||(w&&(w.disabled=!1),x&&(x.disabled=!1),k&&(k.disabled=!1)),b>0){const T=`Prep\
aring... (${y}/${g})`;e.classList.remove("hidden"),t.innerText=T,o&&(o.innerText=`(${y}/${g})`);let A=y*
100,N=0;for(let $ in uploadProgressState.perFilePct)A+=uploadProgressState.perFilePct[$],N++;const G=g>
0?A/(g*100)*100:0,L=`${Math.min(100,G)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=L),l&&c&&
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
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const b=get("marker-modal-title"),w=get("marker-modal-de\
scription"),x=get("marker-save");b&&(b.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u4F5C\u6210":
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
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),b=g-l-d,w=n-g-l,x=y-c-m,k=i-y-c,C=r((T,A,N)=>Number.
isFinite(T)?A>N?(A+N)/2:Math.min(N,Math.max(A,T)):0,"clampOffset");markerView.offsetX=C(markerView.offsetX,
b,w),markerView.offsetY=C(markerView.offsetY,x,k)}r(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
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
height);const n=r((l,c,d=null,m=!1)=>{if(!l)return;const g=Math.max(0,l.x),y=Math.max(0,l.y),b=Math.
max(1,l.w),w=Math.max(1,l.h);d&&(t.fillStyle=d,t.fillRect(g,y,b,w)),t.save(),m&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(g+.5,y+.5,Math.max(1,b-1),Math.max(1,w-1)),t.restore()},"\
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
length;t.set(c,w),e.push({path:c,source:d,name:m});return}const y=e[g];if(!y)return;const b=normalizeAttachmentSource(
y.source);(b==="unknown"&&d!=="unknown"||b==="library"&&d==="upload")&&(y.source=d),!normalizeAttachmentDisplayName(
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
n,y.dataset.displayName=o,y.dataset.defaultDisplayName=o,y.dataset.sendNameCustomized="";const b=escapeHtml(
o),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",x=c?`<img src="${m}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${b}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${x}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${b}</div>
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
y)||o));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const T=y.
querySelector(".upload-remove");T&&(T.onclick=()=>{uploadCancelTokens.add(g),browserFastLocalFiles.delete(
g),decrementUploadTotal(g);const N=y.getAttribute("data-filename");N&&(currentImageUrls=currentImageUrls.
filter(G=>G!==N)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const A=y.querySelector(".upload-marker");return A&&(A.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
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
const b=i.querySelector(".upload-marker");b&&(b.onclick=()=>openMarkerModalForRow(i));const w=i.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}r(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(isUploadCancelled(
t)||!e)return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);
try{const i=await apiFetch("/upload/init",{method:"POST",headers:{"Content-Type":"application/json",
"X-CSRF-Token":csrfToken},body:JSON.stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.
ok){const y=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}const o=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let y=0;y<c;y++){const b=y*l,w=Math.min(e.size,
b+l),x=e.slice(b,w);if(!await new Promise(C=>{const T=new XMLHttpRequest;T.open("POST","/upload/chun\
k",!0),T.setRequestHeader("X-CSRF-Token",csrfToken),T.upload.onprogress=N=>{if(N.lengthComputable&&t&&
t.bar){const G=b+N.loaded,L=Math.min(100,Math.floor(G/e.size*100));t.bar.style.width=`${L}%`,t.status&&
(t.status.textContent=`${L}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,L)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},T.onload=()=>{T.status>=200&&T.status<300?C(!0):C(!1)},T.
onerror=()=>C(!1);const A=new FormData;A.append("upload_id",o),A.append("index",String(y)),A.append(
"total",String(c)),A.append("chunk",x,e.name),T.send(A)}))return t&&t.status&&(t.status.textContent=
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
y),t.row.removeAttribute("data-local-url");const b=t.row.querySelector("img.upload-preview");if(b){const w=d.
filename.replace(/^\d+\//,"");b.src=buildAttachmentPreviewUrl(w)}}const m=normalizeAttachmentPath(d.
filename);if(m&&currentImageUrls.push(m),t&&t.row&&(t.row.setAttribute("data-filename",m||d.filename),
setRowAttachmentSource(t.row,"upload"),m)){const y=isRowAttachmentNameCustomized(t.row),b=defaultAttachmentDisplayName(
m),w=y&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||b;t.row.dataset.defaultDisplayName=
b,setRowAttachmentName(t.row,w)}m&&setAttachmentSourceForPath(m,"upload"),t&&t.status&&(t.status.textContent=
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
e.row.getAttribute("data-local-url");openFileViewer(x,getRowAttachmentName(e.row)||a||w||"")});const b=e.
row.querySelector(".upload-marker");b&&b.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
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
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},m=[],g=16,y="",b=null,w=null,x=null,k=null,C=null,T=!1,A=null;
const N=r(V=>{const X=e.getBoundingClientRect(),re=(V.clientX-X.left)*(e.width/X.width),ce=(V.clientY-
X.top)*(e.height/X.height);return{x:re,y:ce}},"getPoint"),G=r((V,X)=>({x:(V.x+X.x)/2,y:(V.y+X.y)/2}),
"getMid"),L=r((V,X)=>Math.hypot(V.x-X.x,V.y-X.y),"getDist");let $=!1;const M=r(()=>{w||(w=document.createElement(
"canvas"),x=w.getContext("2d")),k||(k=document.createElement("canvas"),C=k.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),P=r(()=>{if(!t||!w||!k)return;const V=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=V,t.drawImage(k,0,0),t.restore()},"r\
enderDrawPreview"),R=r(()=>{C&&(C.strokeStyle=y,C.fillStyle=y,C.lineWidth=g,C.lineCap="round",C.lineJoin=
"round")},"applyMarkerBrush"),D=r(V=>{if(!V)return!1;if(m.length===0)return m.push(V),!0;const X=m[m.
length-1],re=V.x-X.x,ce=V.y-X.y,Q=Math.hypot(re,ce),ue=Math.max(.35,g*.04);if(Q<ue)return!1;const ne=Math.
max(1,g*.25),xe=Math.max(1,Math.ceil(Q/ne));for(let ve=1;ve<=xe;ve++){const we=ve/xe;m.push({x:X.x+re*
we,y:X.y+ce*we})}return!0},"appendStrokePoint"),H=r(()=>{if(C&&(C.clearRect(0,0,k.width,k.height),m.
length!==0)){if(R(),m.length===1){const V=m[0];C.beginPath(),C.arc(V.x,V.y,g/2,0,Math.PI*2),C.fill();
return}if(C.beginPath(),C.moveTo(m[0].x,m[0].y),m.length===2)C.lineTo(m[1].x,m[1].y);else{for(let re=1;re<
m.length-2;re++){const ce=m[re],Q=m[re+1],ue=G(ce,Q);C.quadraticCurveTo(ce.x,ce.y,ue.x,ue.y)}const V=m[m.
length-2],X=m[m.length-1];C.quadraticCurveTo(V.x,V.y,X.x,X.y)}C.stroke()}},"renderStrokeLayer"),ee=r(
(V,X)=>{if(!V||!X)return null;const re=Math.min(V.x,X.x),ce=Math.min(V.y,X.y),Q=Math.abs(V.x-X.x),ue=Math.
abs(V.y-X.y);return{x:re,y:ce,w:Q,h:ue}},"normalizeMosaicRect"),O=r(V=>{const X=n?Number(n.value||16):
16,re=Math.max(6,Math.floor(X)),ce=Math.floor(re/2);return{x:V.x-ce,y:V.y-ce,w:re,h:re}},"buildMosai\
cRectFromPoint"),B=r(()=>{const V=document.createElement("canvas");V.width=e.width,V.height=e.height;
const X=V.getContext("2d");if(!X)return null;markerState.baseCanvas&&X.drawImage(markerState.baseCanvas,
0,0),X.drawImage(e,0,0);try{return X.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),J=r(V=>{if(!t||!V)return!1;const X=B();if(!X)return!1;const re=n?Number(n.value||
16):16,ce=Math.max(4,Math.floor(re/2)),Q=Math.max(0,Math.floor(V.x)),ue=Math.max(0,Math.floor(V.y)),
ne=Math.min(e.width,Math.ceil(V.x+V.w)),xe=Math.min(e.height,Math.ceil(V.y+V.h));if(ne<=Q||xe<=ue)return!1;
for(let ve=ue;ve<xe;ve+=ce)for(let we=Q;we<ne;we+=ce){const Oe=Math.min(ce,ne-we),Ce=Math.min(ce,xe-
ve),lt=Math.min(e.width-1,Math.max(0,we+Math.floor(Oe/2))),ye=(Math.min(e.height-1,Math.max(0,ve+Math.
floor(Ce/2)))*e.width+lt)*4,ge=X.data[ye],ie=X.data[ye+1],it=X.data[ye+2];t.fillStyle=`rgb(${ge},${ie}\
,${it})`,t.fillRect(we,ve,Oe,Ce)}return!0},"applyMosaicRect"),F=r(V=>{if(!t)return;if(i.set(V.pointerId,
{x:V.clientX,y:V.clientY}),i.size>=2){const re=Array.from(i.values()),ce=re[0],Q=re[1];a=!0,$=!1,m=[],
T=!1,A=null,b=null,markerState.mosaicPreviewRect=null,o=L(ce,Q)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=G(ce,Q),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
V.pointerId),V.preventDefault();return}if(a||markerState.mode==="crop")return;$=!0;const X=N(V);if(markerState.
mode==="auto"&&markerState.maskMode)b=X;else if(markerState.mode==="mosaic")T=!0,A=X,markerState.mosaicPreviewRect=
O(X),renderCropOverlay();else{if(M(),!x||!C)return;x.clearRect(0,0,w.width,w.height),x.drawImage(e,0,
0),C.clearRect(0,0,k.width,k.height),g=n?Number(n.value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),m=[],D(X),H(),markerState.hasStroke=!0,P()}e.setPointerCapture&&e.setPointerCapture(
V.pointerId),V.preventDefault()},"start"),q=r(V=>{if(i.has(V.pointerId)&&i.set(V.pointerId,{x:V.clientX,
y:V.clientY}),a&&i.size>=2){const re=Array.from(i.values()),ce=re[0],Q=re[1],ue=G(ce,Q),ne=L(ce,Q)||
1,xe=l*(ne/o);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,xe)),markerView.
offsetX=c.x+(ue.x-d.x),markerView.offsetY=c.y+(ue.y-d.y),applyMarkerTransform(),V.preventDefault();return}
if(!$||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){V.preventDefault();return}const X=N(
V);if(markerState.mode==="mosaic"){if(!T||!A)return;markerState.mosaicPreviewRect=ee(A,X)||O(X),renderCropOverlay()}else
D(X)&&(H(),P());V.preventDefault()},"move"),te=r(V=>{const X=$;if(i.delete(V.pointerId),i.size<2&&(a=
!1),i.size===0){$=!1;const re=X&&V.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.maskMode&&
b?fillMaskSimilarRegion(b):!1;if(b=null,X&&t&&markerState.mode==="draw"&&m.length>0&&(H(),P()),X&&markerState.
mode==="mosaic"&&A){const ce=N(V);let Q=ee(A,ce);(!Q||Q.w<2||Q.h<2)&&(Q=O(A)),J(Q)&&(markerState.hasStroke=
!0,markerState.mosaicRects.push(Q))}m=[],T=!1,A=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),
X&&(markerState.mode!=="auto"||re)&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(
V.pointerId),V.preventDefault()},"end");e.addEventListener("pointerdown",F),e.addEventListener("poin\
termove",q),e.addEventListener("pointerup",te),e.addEventListener("pointercancel",te)}r(initMarkerCanvas,
"initMarkerCanvas");function fillMaskSimilarRegion(e){var $;const t=get("marker-canvas"),n=markerState.
baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,o=Math.floor(e.x),l=Math.floor(e.y);if(o<0||l<0||o>=i||l>=
a)return!1;const c=n.data,d=l*i+o,m=d*4;if(c[m+3]<16)return!1;const g=c[m],y=c[m+1],b=c[m+2],w=Number(
(($=get("marker-auto-tolerance"))==null?void 0:$.value)||32),x=new Uint8Array(i*a),k=[d],C=[],T=25e4,
A=r(M=>{const P=M*4;return c[P+3]>=16&&Math.max(Math.abs(c[P]-g),Math.abs(c[P+1]-y),Math.abs(c[P+2]-
b))<=w},"matches");for(x[d]=1;k.length;){const M=k.pop();if(x[M]===2||!A(M))continue;const P=Math.floor(
M/i);let R=M%i,D=R;for(;R>0&&x[P*i+R-1]!==2&&A(P*i+R-1);)R--;for(;D+1<i&&x[P*i+D+1]!==2&&A(P*i+D+1);)
D++;for(let H=R;H<=D;H++)x[P*i+H]=2;if(C.push(P,R,D+1),C.length/3>T)return showToast("\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\
\u3066\u304F\u3060\u3055\u3044","error",!0),!1;for(const H of[P-1,P+1]){if(H<0||H>=a)continue;let ee=R;
for(;ee<=D;){const O=H*i+ee;if(x[O]===0&&A(O))for(x[O]=1,k.push(O);ee+1<=D&&x[H*i+ee+1]===0&&A(H*i+ee+
1);)ee++,x[H*i+ee]=1;ee++}}}if(!C.length)return!1;const N=t.getContext("2d");if(!N)return!1;const G=N.
getImageData(0,0,i,a),L=G.data;for(let M=0;M<C.length;M+=3){const P=C[M]*i+C[M+1],R=C[M]*i+C[M+2];for(let D=P;D<
R;D++){const H=D*4;L[H]=L[H+1]=L[H+2]=L[H+3]=255}}return N.putImageData(G,0,0),markerState.hasStroke=
!0,!0}r(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,o=null,l=null,c=!1,
d=0,m=markerView.scale,g={x:0,y:0},y={x:0,y:0};const b=8,w=14,x=r((M,P,R)=>Math.min(R,Math.max(P,M)),
"clamp"),k=r(M=>{const P=e.getBoundingClientRect(),R=(M.clientX-P.left)*(e.width/P.width),D=(M.clientY-
P.top)*(e.height/P.height);return{x:R,y:D}},"getPoint"),C=r((M,P)=>({x:(M.x+P.x)/2,y:(M.y+P.y)/2}),"\
getMid"),T=r((M,P)=>Math.hypot(M.x-P.x,M.y-P.y),"getDist"),A=r(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),N=r((M,P)=>{if(!P)return"move";const R=P.x,D=P.y,H=P.x+P.w,ee=P.
y+P.h,O=Math.abs(M.x-R)<=w,B=Math.abs(M.x-H)<=w,J=Math.abs(M.y-D)<=w,F=Math.abs(M.y-ee)<=w;if(O&&J)return"\
nw";if(B&&J)return"ne";if(O&&F)return"sw";if(B&&F)return"se";if(J)return"n";if(F)return"s";if(O)return"\
w";if(B)return"e";if(M.x>R+w&&M.x<H-w&&M.y>D+w&&M.y<ee-w)return"move";const te=M.x<R?"left":M.x>H?"r\
ight":null,V=M.y<D?"top":M.y>ee?"bottom":null;if(te&&V){if(te==="left"&&V==="top")return"nw";if(te===
"right"&&V==="top")return"ne";if(te==="left"&&V==="bottom")return"sw";if(te==="right"&&V==="bottom")
return"se"}return te?te==="left"?"w":"e":V?V==="top"?"n":"s":"move"},"hitTest"),G=r(M=>{if(markerState.
mode!=="crop")return;if(n.set(M.pointerId,{x:M.clientX,y:M.clientY}),n.size>=2){const D=Array.from(n.
values()),H=D[0],ee=D[1];c=!0,i=!1,d=T(H,ee)||1,m=markerView.scale,g={x:markerView.offsetX,y:markerView.
offsetY},y=C(H,ee),e.setPointerCapture&&e.setPointerCapture(M.pointerId),M.preventDefault();return}if(c)
return;i=!0;const P=k(M),R=A();o=N(P,R),a=P,l=R?{x:R.x,y:R.y,w:R.w,h:R.h}:null,renderCropOverlay(),e.
setPointerCapture&&e.setPointerCapture(M.pointerId),M.preventDefault()},"start"),L=r(M=>{if(markerState.
mode!=="crop")return;if(n.has(M.pointerId)&&n.set(M.pointerId,{x:M.clientX,y:M.clientY}),c&&n.size>=
2){const te=Array.from(n.values()),V=te[0],X=te[1],re=C(V,X),ce=T(V,X)||1,Q=m*(ce/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Q)),markerView.offsetX=g.x+(re.x-y.x),markerView.
offsetY=g.y+(re.y-y.y),applyMarkerTransform(),renderCropOverlay(),M.preventDefault();return}if(!i||!a||
!l)return;const P=k(M),R=e.width,D=e.height,H={x:l.x,y:l.y,w:l.w,h:l.h},ee=l.x+l.w,O=l.y+l.h,B=r(()=>{
const te=x(P.x,0,ee-b);H.x=te,H.w=ee-te},"applyW"),J=r(()=>{H.w=x(P.x-l.x,b,R-l.x)},"applyE"),F=r(()=>{
const te=x(P.y,0,O-b);H.y=te,H.h=O-te},"applyN"),q=r(()=>{H.h=x(P.y-l.y,b,D-l.y)},"applyS");switch(o){case"\
move":{const te=P.x-a.x,V=P.y-a.y;H.x=x(l.x+te,0,R-l.w),H.y=x(l.y+V,0,D-l.h);break}case"w":B();break;case"\
e":J();break;case"n":F();break;case"s":q();break;case"nw":F(),B();break;case"ne":F(),J();break;case"\
sw":q(),B();break;case"se":q(),J();break;default:break}H.x=x(H.x,0,R-H.w),H.y=x(H.y,0,D-H.h),markerState.
cropRect=H,renderCropOverlay(),M.preventDefault()},"move"),$=r(M=>{n.delete(M.pointerId),n.size<2&&(c=
!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,o=null,l=null),e.releasePointerCapture&&e.releasePointerCapture(
M.pointerId),M.preventDefault()},"end");e.addEventListener("pointerdown",G),e.addEventListener("poin\
termove",L),e.addEventListener("pointerup",$),e.addEventListener("pointercancel",$),e.addEventListener(
"pointerleave",$)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){const e=markerState.
row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get("marker-attach-ori\
ginal");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement("canvas");const o=markerState.
naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||n.height;if(markerState.
maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const C=document.createElement("canvas");C.width=o,C.height=l;const T=C.getContext(
"2d");if(!T)return;T.fillStyle="#000000",T.fillRect(0,0,o,l),T.drawImage(n,0,0,o,l);const A=await new Promise(
L=>C.toBlob(L,"image/png"));if(!A){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const N=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([A],`${N}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=o,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,o,l),c.drawImage(n,0,0,o,l),markerState.cropRect){const C=o/n.width,T=l/n.height,A=Math.max(0,Math.
floor(markerState.cropRect.x*C)),N=Math.max(0,Math.floor(markerState.cropRect.y*T)),G=Math.min(o,Math.
max(1,Math.floor(markerState.cropRect.w*C))),L=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*T))),$=document.createElement("canvas");$.width=G,$.height=L;const M=$.getContext("2d");M&&(M.drawImage(
a,A,N,G,L,0,0,G,L),a=$)}const d=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const g=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([d],`${g}_marked.png`,
{type:"image/png"}),b={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};b.status&&(b.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(b,y);const w=e.getAttribute("data-filename"),x=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=x,setAttachmentSourceForPath(
w,x)),await uploadFileWithProgress(y,b)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}r(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,o=null,l=null,c=null,d=[],m=null;const g=r(()=>{m&&clearTimeout(
m);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(b=>b.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(o)try{o.close()}catch{}},"cleanup"),y=r(
()=>{g(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const b=a.getAudioTracks();
if(!b||!b.length)return y();o=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=o.createMediaStreamSource(new MediaStream(b)),l=o.createScriptProcessor(4096,1,1),l.onaudioprocess=
x=>{const k=x.inputBuffer.getChannelData(0);d.push(new Float32Array(k))},c.connect(l),l.connect(o.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(m=setTimeout(()=>{const x=(e.
name||"video").replace(/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{
type:"audio/wav"});g(),n(C)},w+250)),await i.play(),i.onended=()=>{const x=(e.name||"video").replace(
/\.[^/.]+$/,""),k=encodeWav(d,o.sampleRate),C=new File([k],`${x}.audio.wav`,{type:"audio/wav"});g(),
n(C)}}catch{y()}},i.onerror=()=>y()}))}r(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
if(!e||!e.length)return;const n=Array.from(e).filter(Boolean);if(!n.length)return;const i=n,a=uploadGeneration;
t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=i.length,uploadProgressState.
active+=i.length,updateFilePreview();const o=!!(get("upload-audio-only")&&get("upload-audio-only").checked),
l=getModelMediaSupport(get("model-select").value),c=r(async b=>{let w=null;if(a!==uploadGeneration)return!1;
try{if(isAudioFile(b)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(b)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!b.type||!b.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const x=addUploadRow(b);x.generation=a,updateFilePreview(),w=x.uploadId,uploadProgressState.
perFilePct[w]=0;let k=b;if(o&&isVideoFile(b)){const T=await extractAudioFromVideo(b,x);T?(k=T,updateUploadRowFile(
x,T),x&&x.status&&(x.status.textContent="\u97F3\u58F0\u306E\u307F")):(x&&x.status&&(x.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&b.type.startsWith("image/"))try{k=await applyImageCompression(b,
readComposerCompression()),k!==b&&updateUploadRowFile(x,k)}catch{}if(isUploadCancelled(x))return!1;if(browserFastModeEnabled){
const T=Array.from(browserFastLocalFiles.values()).reduce((A,N)=>A+Number(N.file&&N.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||T+k.size>BROWSER_FAST_MAX_BYTES?(x&&x.status&&
(x.status.textContent="\u4E0A\u9650\u8D85\u904E"),x&&x.row&&x.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(x.uploadId,{file:k,rowObj:x}),
x.status&&(x.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
x.bar&&(x.bar.style.width="100%"),x.row&&(x.row.dataset.browserFastLocal="1"),!0)}const C=await uploadFileWithProgress(
k,x,t);if(!C&&t.cameraCapture&&!isUploadCancelled(x)){const T=document.createElement("button");T.className=
"text-xs border border-gray-700 rounded px-2 py-1",T.textContent="\u518D\u8A66\u884C",T.onclick=()=>{
if(isUploadCancelled(x))return;T.disabled=!0;const A=x.row.getAttribute("data-local-url");A&&URL.revokeObjectURL(
A),x.row.remove(),handleFiles([k],{...t,openModal:!0,skipCompression:!0})},x.row.appendChild(T)}return C}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),d=new Array(i.length).fill(!1);let m=0;const g=Math.
min(UPLOAD_CONCURRENCY,i.length),y=Array.from({length:g}).map(async()=>{for(;;){const b=m++;if(b>=i.
length)break;d[b]=await c(i[b])}});return await Promise.all(y),d}r(handleFiles,"handleFiles"),get("c\
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
