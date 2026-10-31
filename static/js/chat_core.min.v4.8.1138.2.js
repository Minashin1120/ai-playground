document.addEventListener("DOMContentLoaded",()=>{var ki,_i;initThemeFromServer(),applyLiquidGlassMode(
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
getBoundingClientRect(),g=d.getBoundingClientRect(),b=g.left+g.width/2-(m.left+m.width/2),k=g.top+g.
height/2-(m.top+m.height/2);r.style.transform=`translate(${b}px, ${k}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const m=get("gpt-image-format"),g=get("gpt-image-compression-wrap");m&&g&&(m.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}o(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{b&&(Array.from(b.options).
forEach(k=>{k.value!=="1K"&&(k.disabled=g)}),g&&b.value!=="1K"&&(b.value="1K"))})}o(c,"updateGeminiI\
mageUi");function u(){const d=get("grok-image-options");if(!d)return;const m=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),b=m==="grok-imagine-image-quality"||m==="grok-imagine-image-2\
.0",k=m==="grok-imagine-image-2.0";if(g){d.classList.remove("hidden");const T=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const M=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;M&&M.classList.toggle("hidden",!k)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const T=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const M=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;M&&M.classList.toggle(
"hidden",!k)}}o(u,"updateGrokImageUi");function f(){var O,B;const d=get("ideogram-image-options"),m=get(
"modal-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",b=isIdeogramModel(
g),k=ideogramModelTraits(g),_={resolution:k.resolution,quality:k.quality,speed:k.speed,style:k.style,
negative:k.negative};if(d&&d.classList.toggle("hidden",!b),m&&m.classList.toggle("hidden",!b),updateIdeogramPromptInputUi(),
!b)return;const T=((O=get("ideogram-image-operation"))==null?void 0:O.value)==="remove_background",M=((B=
get("ideogram-image-operation"))==null?void 0:B.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(
Q=>{const de=Q==="operation"||M&&Q==="seed"||!(T||M)&&_[Q]!==!1;["ideogram-image-","modal-ideogram-i\
mage-"].forEach(ce=>{const W=get(ce+Q);!W||!W.parentElement||W.parentElement.classList.toggle("hidde\
n",!de)})}),["ideogram-image-style","modal-ideogram-image-style"].forEach(Q=>{const de=get(Q);if(!de)
return;const ce=k.styleOptions.join("|");if(de.dataset.options!==ce){const W=de.value;de.innerHTML=k.
styleOptions.map(pe=>`<option value="${pe}">${pe==="auto"?"Auto":pe}</option>`).join(""),de.dataset.
options=ce,de.value=k.styleOptions.includes(W)?W:"auto"}});const $=get("ideogram-image-note");$&&($.
textContent=T?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
M?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
k.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(d=>{
var m;(m=get(d))==null||m.addEventListener("change",()=>{const g=get(d.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");g&&(g.value=get(d).value),f(),L()})});function h(){
var b;const d=get("grok-video-options");if(!d)return;const m=String(((b=get("model-select"))==null?void 0:
b.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");
const g=get("grok-video-resolution");if(g){const k=Array.from(g.options).find(T=>T.value==="1080p"),
_=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(m);k&&(k.disabled=!_),!_&&g.value===
"1080p"&&(g.value="720p")}}o(h,"updateGrokVideoUi");function y(){var k;const d=get("gemini-video-opt\
ions");if(!d)return;const m=String(((k=get("model-select"))==null?void 0:k.value)||"").toLowerCase();
isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get("gemini-vide\
o-resolution");if(g){const _=Array.from(g.options).find(M=>M.value==="4K"),T=m==="veo-3.1-lite-gener\
ate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";_&&(_.disabled=T),T&&g.value===
"4K"&&(g.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle("hidden",m===
"gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function w(){const d=get("gemini-music-options");
if(!d)return;const m=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!m;d.classList.toggle("hid\
den",!g);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!m)}o(w,"updateGemi\
niMusicUi");function v(){var T;const d=get("xai-chat-options");if(!d)return;const m=String(((T=get("\
model-select"))==null?void 0:T.value)||"").toLowerCase(),g=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!g);const b=get("xai-logp\
robs"),k=get("xai-top-logprobs"),_=m.includes("grok-4.20");b&&(b.disabled=_,_&&(b.checked=!1)),k&&(k.
disabled=_,_&&(k.value=""))}o(v,"updateXaiChatUi");function x(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const k=get(b);k&&(k.classList.toggle("opacity-50",d),k.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(x,"updateMistralOcrUi");function S(){const d=get("image-input-limit\
s");if(!d)return;const m=(get("model-select").value||"").toLowerCase();let g="",b=!1;m.includes("gpt\
-image")?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):m==="deepseek-v4.1-flash"||m==="deepseek-v4-flash-vision-exp"?(b=!0,g=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):m.includes("deepseek")||(isGeminiImageModelKey(m)?(b=!0,m.includes("gemini-3.1-flash-lite-\
image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):m==="gemini-nano-banana-2.1"?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2.1 \
\u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u30FB2K\u30FB4K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u52D5\u753B\u3092\u53C2\u8003\u306B\u3057\u305F\u753B\u50CF\u751F\u6210\u306B\u3082\u5BFE\u5FDC\u3057\u307E\u3059</div>"].
join(""):m.includes("gemini-3.1-flash-image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):m.includes("gemini-2.5")&&m.includes("image")?g=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):g=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(m)?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):m.includes("grok")?(b=!0,g=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):isIdeogramModel(m)?(b=!0,g=ideogramModelTraits(m).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u3082\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.5 \u306E\u307F\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u306F\u5168\u30E2\u30C7\u30EB\u3067\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join("")):m.includes("grok")&&m.includes("video")&&(b=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(d.innerHTML=g,d.classList.remove("hidden")):(d.classList.add("hidden"),d.innerHTML="")}
o(S,"updateImageInputLimits");function L(){const d=get("enable-sys-prompt"),m=!!(d&&d.disabled),g=!!(d&&
d.checked);A(),d&&(d.disabled?!m&&g?d.dataset.restoreChecked="1":m||delete d.dataset.restoreChecked:
(m&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(L,"toggleOpti\
ons"),window.toggleOptions=L;function C(d){const m=get("thinking-options"),g=get("thinking-level"),b=get(
"reasoning-effort"),k=get("enable-thinking"),_=get("thinking-budget-label"),T=get("thinking-budget"),
M=d==="effort";if(m&&(m.dataset.thinkingMode=M?"effort":"thinking"),g&&g.classList.toggle("hidden",M),
b&&b.classList.toggle("hidden",!M),_&&_.classList.toggle("hidden",M),T&&T.classList.toggle("hidden",
M),!!k)if(M){k.dataset.thinkingChecked===void 0&&(k.dataset.thinkingChecked=k.checked?"1":"0");const $=b?
Array.from(b.options).find(B=>B.value==="none"):null,O=!!($&&!$.disabled&&!$.classList.contains("hid\
den"));k.checked=O?!!(b&&String(b.value||"").toLowerCase()!=="none"):!0,k.disabled=!O}else k.dataset.
thinkingChecked!==void 0&&(k.checked=k.dataset.thinkingChecked==="1",delete k.dataset.thinkingChecked)}
o(C,"setComposerThinkingMode");function A(){const d=get("model-select");if(!d)return;const m=d.value,
g=String(m||"").toLowerCase(),b=g.includes("deepseek"),k=g.startsWith("glm-"),_=get("thinking-option\
s"),T=get("reasoning-effort-container"),M=get("enable-thinking"),$=get("thinking-level"),O=get("thin\
king-budget"),B=get("enable-search"),Q=get("search-container"),de=get("url-context-container"),ce=get(
"enable-maps"),W=get("maps-grounding-container"),pe=get("enable-sys-prompt"),q=get("sys-prompt-optio\
n"),ye=get("enable-python"),V=get("python-container"),he=get("prompt-cache-container"),ke=get("enabl\
e-prompt-cache"),ae=m==="gpt-5-search-api",ve=m.includes("tts"),Pe=isMistralOcrModel(m),Oe=g.includes(
"gemini-3.1-flash-lite-image"),lt=g==="gemini-nano-banana-2.1",Xe=g.includes("gemini-3.1-flash-image")&&
!Oe||lt,ht=isClaudeModelKey(m),Pt=g==="gemini-3.8-flash-cyber",Ae=isLlmModel()&&!b&&!ve&&!g.includes(
"realtime")&&!g.includes("native-audio")&&!g.includes("live");he&&(Ae?(he.classList.remove("hidden",
"opacity-50","pointer-events-none"),ke&&(ke.disabled=!1)):(ke&&(ke.checked=!1,ke.disabled=!0),he.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),_&&_.classList.add("hidden"),T&&T.classList.
add("hidden"),C("thinking");const _e=get("vision-model-info");if(_e&&_e.classList.add("hidden"),T){const Se=get(
"reasoning-effort");if(Se){Array.from(Se.options).forEach(Ue=>{const Ye=g==="gpt-5.6"||g.startsWith(
"gpt-5.6-"),mt=g.startsWith("gpt-6"),_n=g==="gpt-6-sol"||g==="gpt-6-luna"||g==="gpt-6.1-terra",en=g===
"deepseek-v4.1-flash"||g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash\
-vision-exp",Sn=g==="deepseek-v4-pro",Kn=g.includes("grok-4.5"),F=g.includes("grok-4.6");Ue.value===
"max"?Ue.classList.toggle("hidden",!Ye&&!mt&&!en&&!Sn):Ue.value==="xhigh"?Ue.classList.toggle("hidde\
n",!F&&!g.includes("multi-agent")&&!Ye&&!mt):Ue.value==="medium"?Ue.classList.toggle("hidden",!(g.includes(
"grok-4.3")||Kn||F||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("mu\
lti-agent")||g.includes("gpt-5")||mt||g.includes("o1")||g.includes("o3"))):Ue.value==="none"?Ue.classList.
toggle("hidden",!g.includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!_n&&!en&&
!Sn):Ue.value==="low"&&Ue.classList.toggle("hidden",Sn)});const Ne=Se.selectedOptions&&Se.selectedOptions[0];
Ne&&Ne.classList.contains("hidden")&&(Se.value=b?"high":"medium")}}de&&de.classList.add("hidden"),W&&
W.classList.add("hidden"),M&&(M.disabled=!1),O&&(O.disabled=!0,O.classList.add("opacity-50"));const De=isGeminiImageModelKey(
m);if(ve||Pe)Q&&(get("enable-search").checked=!1,Q.classList.add("opacity-50","pointer-events-none")),
de&&(get("enable-url-context").checked=!1,de.classList.add("opacity-50","pointer-events-none")),W&&ce&&
(ce.checked=!1,W.classList.add("opacity-50","pointer-events-none")),V&&(ye.checked=!1,V.classList.add(
"opacity-50","pointer-events-none")),pe&&q&&(pe.checked=!1,pe.disabled=!0,q.classList.add("opacity-5\
0"));else if(Xe||Oe)W&&ce&&(ce.checked=!1,W.classList.add("hidden","opacity-50","pointer-events-none")),
_.classList.remove("hidden"),lt?(Array.from($.options).forEach(Se=>{Se.disabled=Se.value==="low"}),[
"minimal","medium","high"].includes($.value)||($.value="medium")):(Array.from($.options).forEach(Se=>{
["low","medium"].includes(Se.value)&&(Se.disabled=!0),["minimal","high"].includes(Se.value)&&(Se.disabled=
!1)}),["minimal","high"].includes($.value)||($.value=Oe?"minimal":"high")),M&&(M.disabled=!1),Oe&&(B&&
(B.checked=!1,B.disabled=!0),Q&&Q.classList.add("opacity-50","pointer-events-none"));else if(De)W&&ce&&
(ce.checked=!1,W.classList.add("hidden","opacity-50","pointer-events-none"));else if(ht)_.classList.
remove("hidden"),O&&(O.disabled=!1,O.classList.remove("opacity-50")),Array.from($.options).forEach(Se=>{
Se.disabled=!0}),V&&(ye.checked=!1,V.classList.add("opacity-50","pointer-events-none"));else if(Pt){
_&&_.classList.remove("hidden"),M&&(M.checked=!0,M.disabled=!0),Array.from($.options).forEach(Ne=>{Ne.
disabled=!["low","medium","high"].includes(Ne.value)}),["low","medium","high"].includes($.value)||($.
value="medium"),[Q,de,W,V].forEach(Ne=>{Ne&&Ne.classList.add("opacity-50","pointer-events-none")}),[
B,ce,ye].forEach(Ne=>{Ne&&(Ne.checked=!1,Ne.disabled=!0)});const Se=get("enable-url-context");Se&&(Se.
checked=!1,Se.disabled=!0),pe&&q&&(pe.disabled=!1,q.classList.remove("opacity-50"))}else if(m.includes(
"gemini")&&!De){_.classList.remove("hidden"),de&&de.classList.remove("hidden","opacity-50","pointer-\
events-none");const Se=m.includes("gemini-3")||m.includes("gemini-4");W&&(Se?W.classList.remove("hid\
den","opacity-50","pointer-events-none"):(ce&&(ce.checked=!1),W.classList.add("hidden","opacity-50",
"pointer-events-none")));const Ne=m.includes("flash");Array.from($.options).forEach(Ue=>{m==="gemini\
-4-argon"||m==="gemini-3.8-flash"||m==="gemini-3.7-flash"?Ue.disabled=!["low","medium","high"].includes(
Ue.value):m==="gemini-3.6-flash"?Ue.disabled=!["medium","high"].includes(Ue.value):m==="gemini-3.5-f\
lash-lite"?Ue.disabled=!["minimal","medium","high"].includes(Ue.value):["minimal","medium"].includes(
Ue.value)?Ue.disabled=!Ne:Ue.disabled=!1}),(m==="gemini-4-argon"||m==="gemini-3.8-flash"||m==="gemin\
i-3.7-flash")&&!["low","medium","high"].includes($.value)||m==="gemini-3.6-flash"&&!["medium","high"].
includes($.value)?$.value="medium":m==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(
$.value)?$.value="minimal":!Ne&&["minimal","medium"].includes($.value)&&($.value="high"),Se?M&&(M.checked=
!0,M.disabled=!0):M&&(M.disabled=!1),O&&m.includes("gemini-2.5")&&(O.disabled=!1,O.classList.remove(
"opacity-50")),O&&!m.includes("gemini-2.5")&&(O.disabled=!0,O.classList.add("opacity-50"))}if(isLlmModel()&&
(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4\
.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes(
"multi-agent")||g.includes("gpt")&&!g.includes("tts")))_&&_.classList.remove("hidden"),C("effort"),Q&&
Q.classList.remove("opacity-50","pointer-events-none");else if(b){_&&_.classList.remove("hidden"),C(
"effort");const Se=get("vision-model-info");if(Se&&Se.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),B&&(B.checked=!1,B.disabled=!0),Q&&Q.classList.add("opaci\
ty-50","pointer-events-none"),de){const Ne=get("enable-url-context");Ne&&(Ne.checked=!1),de.classList.
add("opacity-50","pointer-events-none")}W&&ce&&(ce.checked=!1,W.classList.add("opacity-50","pointer-\
events-none"))}else Pe||(Q&&Q.classList.remove("opacity-50","pointer-events-none"),W&&ce&&(ce.checked=
!1,W.classList.add("hidden","opacity-50","pointer-events-none")));if(ve?V&&V.classList.add("opacity-\
50","pointer-events-none"):(V&&V.classList.remove("opacity-50","pointer-events-none"),(!De||Xe)&&!m.
includes("gpt-image")&&(pe.disabled=!1,q.classList.remove("opacity-50"))),(De&&!Xe||m.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(m)||isGrokVideoModel()||Pe)&&pe&&q&&(pe.checked=!1,pe.disabled=
!0,q.classList.add("opacity-50")),V&&(isLlmModel()?(V.classList.remove("hidden"),ye.disabled=!1):(ye.
checked=!1,ye.disabled=!0,V.classList.add("hidden"))),ae?(B&&(B.checked=!0,B.disabled=!0),Q&&Q.classList.
add("opacity-50","pointer-events-none"),V&&(ye.checked=!1,ye.disabled=!0,V.classList.add("opacity-50",
"pointer-events-none"))):B&&!m.includes("tts")&&!Pe&&!b&&!Oe&&(B.disabled=!1),Pt){[B,ce,ye].forEach(
Ne=>{Ne&&(Ne.checked=!1,Ne.disabled=!0)});const Se=get("enable-url-context");Se&&(Se.checked=!1,Se.disabled=
!0),[Q,de,W,V].forEach(Ne=>{Ne&&Ne.classList.add("opacity-50","pointer-events-none")})}k&&(_e&&_e.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[B,ye].forEach(Se=>{Se&&(Se.checked=!1,Se.disabled=!0)}),[Q,V].forEach(Se=>{Se&&
Se.classList.add("opacity-50","pointer-events-none")}));const ct=get("mask-btn");ct&&(isGptImageModel()||
isIdeogramObjectRemoval()?ct.classList.remove("hidden"):(ct.classList.add("hidden"),currentMaskImage=
null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),u(),f(),h(),y(),w(),
updateBatchUi(m),v(),x(),S(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}
o(A,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",L),
get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
L(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const I=get("enable-canvas-mode");I&&(I.checked=canvasModeEnabled,I.addEventListener("change",()=>syncCanvasModeUi(
I.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const J=get("enable-codin\
g-mode");J&&(J.checked=codingModeEnabled,J.addEventListener("change",()=>syncCodingModeUi(J.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",d=>{const m=d.
target.closest("[data-canvas-block-index]");if(!m)return;const g=Number(m.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(g,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",d=>{if(d.target.value===
"")return;const m=Number(d.target.value);Number.isInteger(m)&&applyCanvasSelection(m,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",d=>{const m=d.target.closest(
"[data-canvas-panel-view]");if(!m)return;const g=m.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
g,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const d=getCanvasModeElements(),m=d&&d.code&&d.code.textContent||"";if(!m.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
m,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const H=get(
"prompt-controls-toggle-btn");H&&(H.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,m,g){const b=(m||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(d||"");const k=d||"",_=(b||"").toLowerCase();
let T="";try{const q=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&k.length>2e4?T=escapeHtml(
k):T=hljs.highlight(k,{language:q}).value}catch{T=escapeHtml(k)}const M=encodeURIComponent(k).replace(
/'/g,"%27"),$=detectBlockedScriptsInCode(k),O=hashString(`${b||"TEXT"}
${k||""}`);let B="";if(canvasModeEnabled){const q=String(canvasPreviewState.selectedKey||"")===O,ye=q?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";B=`<button\
 class="canvas-preview-btn${q?" canvas-active":""}" data-code="${M}" data-code-key="${O}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${ye}" aria-label="${ye}" aria-pressed="${q?"true":"false"}"\
><i class="fas ${q?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
_,k)){const q=$?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
B=`<button class="html-preview-btn" data-code="${M}" ${$?'data-suspicious="1"':""} title="${q}" aria\
-label="${q}"><i class="fas ${$?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const Q=`\
<button class="download-btn" data-code="${M}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,de=_==="diff"?"":`<button class="codin\
g-target-btn" data-code="${M}" data-code-key="${O}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,ce=(b||"TEXT")+($?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),W=getRenderableSvgCode(_,k);return`${W?buildSvgCodeRenderHtml(W,O):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${O}"><div class="code-header"\
><span class="code-lang">${ce}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${de}${B}${Q}\
<button class="copy-btn" data-code="${M}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${T}</code></pre></d\
iv></div>`},link(d,m,g){return`<a href="${d}" title="${m||""}" target="_blank">${g}</a>`},image(d,m,g){
return buildChatImageHtml(d,{alt:g,title:m})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(ki=get("version-update-dismis\
s"))==null||ki.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const ne=get("version-update-clear\
-cache");if(ne&&(ne.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),ne.
addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(
ne.checked)})),(_i=get("version-update-reload"))==null||_i.addEventListener("click",async()=>{var m;
await versionUpdateCachePreferenceSavePromise.catch(()=>{}),!!((m=get("version-update-clear-cache"))!=
null&&m.checked)?await clearSiteCacheAndReload(get("version-update-reload"),{scanFirst:!0}):location.
reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.setVersionChangeHandler(d=>{d&&d!==appVersion&&
(localStorage.getItem("version_notified")||"")!==d&&(localStorage.setItem("app_version",d),purgeCaches().
then(()=>checkAndNotifyVersion(d)))}),window.ConnectionMonitor.start(),window.addEventListener("onli\
ne",()=>window.ConnectionMonitor.probeNow()),window.addEventListener("offline",()=>{window.ConnectionMonitor.
cancelProbe(),window.ConnectionMonitor.setUnavailable("offline")}),window.addEventListener("focus",()=>window.
ConnectionMonitor.probeNow()),document.addEventListener("visibilitychange",()=>{document.hidden||window.
ConnectionMonitor.probeNow()}),window.addEventListener("pagehide",()=>window.ConnectionMonitor.stop())),
applyCacheMode(useSwCache),botConfig&&botConfig.lock&&botConfig.lock.active&&!isAdminUser&&showBotLockOverlay(
botConfig.lock.message,botConfig.lock.remaining_seconds),window.__turnstileApiLoaded&&window.initTurnstileWidget&&
window.initTurnstileWidget(),botConfig&&botConfig.globalEnabled&&botConfig.accountEnabled&&!isAdminUser){
botConfig.turnstileVerified&&(botDetectionVerified=!0);try{botTelemetry.start()}catch(d){console.error(
d)}try{runBotDetectionGate()}catch(d){console.error(d)}}else{const d=get("turnstile-container");d&&d.
classList.add("hidden")}const j=o(d=>{if(!d)return"\u4E0D\u660E";const m=new Date(d);return Number.isNaN(
m.getTime())?d:m.toLocaleString()},"formatSessionTime"),K=o(d=>{const m=Array.isArray(d)?d:[],g=get(
"passkey-list"),b=get("passkey-count");if(b&&(b.innerText=String(m.length)),!!g){if(!m.length){g.innerHTML=
'<div class="text-[11px] text-gray-500">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}g.innerHTML="",m.forEach((k,_)=>{const T=k&&k.id?String(k.id):"",M=document.createElement("di\
v");M.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const $=document.createElement("div");$.className="min-w-0";const O=document.createElement("\
div");O.className="text-xs text-gray-200 truncate",O.innerText=k&&k.name?String(k.name):`Security Ke\
y ${_+1}`;const B=document.createElement("div");B.className="text-[10px] text-gray-500 mt-1",B.innerText=
k&&k.created_at?`\u767B\u9332\u65E5\u6642: ${j(k.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
$.appendChild(O),$.appendChild(B),M.appendChild($);const Q=document.createElement("button");Q.type="\
button",Q.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",Q.innerText="\u524A\u9664",Q.disabled=!T,T&&(Q.onclick=()=>window.removeWebAuthnCredential(
T)),M.appendChild(Q),g.appendChild(M)})}},"renderPasskeyList"),ue=o(d=>{const m=get("session-list");
if(m){if(!d||!d.length){m.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(g=>{const b=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",k=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",_=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
T=(g.user_agent||"Unknown").slice(0,120),M=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${k}<div class="text-xs text-gray-200">${escapeHtml(
M)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(T)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(j(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(j(g.created_at))}</div></div>${_}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const b=g.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const k=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let _={};try{_=await k.json()}catch{}if(k.ok){if(_.logged_out){
location.href="/login";return}await me()}else showToast(_&&_.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),me=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api\
/sessions");let g={};try{g=await m.json()}catch{}if(!m.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(g.sessions||[]).filter(k=>!k.is_revoked);ue(b)},"loadSessions"),oe=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>me());const m=get("session-revoke-others-btn");m&&(m.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await me():showToast("\u64CD\u4F5C\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(d=>{d&&(currentVisionModel=d.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
d);try{loadMcpServers()}catch{}d&&d.theme_color&&applyThemeColor(d.theme_color,!0),d&&Object.prototype.
hasOwnProperty.call(d,"minimal_prompt_mode")&&d.minimal_prompt_mode?setMinimalPromptMode(!0):d&&Object.
prototype.hasOwnProperty.call(d,"compact_prompt_mode")&&setCompactPromptMode(!!d.compact_prompt_mode),
get("set-server-diagnostic-log-retention-hours")&&(get("set-server-diagnostic-log-retention-hours").
value=d.server_diagnostic_log_retention_hours||1),get("set-server-diagnostic-log")&&(get("set-server\
-diagnostic-log").checked=d.enable_server_diagnostic_log===!0),get("set-client-debug-log")&&syncClientDebugLogToggle(
d.enable_client_debug_log===!0,"settings sync");const m=get("enable-sys-prompt");m&&d&&d.system_prompt&&
String(d.system_prompt).trim()&&(!m.disabled&&!d.default_enable_system_prompt&&!d.use_last_chat_settings&&
(m.checked=!0),L())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const Le=get("vision-model-change-btn");Le&&(Le.onclick=()=>_openVisionModelSelector());const E=get(
"compression-format-only");E&&(E.onchange=()=>{const d=E.checked,m=get("compression-max-size"),g=get(
"compression-max-dim");m&&(m.disabled=d),g&&(g.disabled=d);const b=get("compression-size-wrap"),k=get(
"compression-dim-wrap");b&&(b.style.opacity=d?"0.4":"1"),k&&(k.style.opacity=d?"0.4":"1")});const R=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");R(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const X=get("storage-us\
age-refresh");X&&(X.onclick=()=>loadStorageUsage());let N=null;const D=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),G=o((d={})=>{const m=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),b=get("account-transfer-progress-percent"),k=get("account-transfer-progress-text"),
_=get("account-transfer-progress-detail"),T=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&m.
classList.remove("hidden"),g&&(g.style.width=`${T}%`),b&&(b.textContent=`${Math.round(T)}%`),k&&(k.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),_){const $={queued:"\
\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\u30D5\u30A1\
\u30A4\u30EB\u3092\u66F8\u304D\u51FA\u3057\u4E2D",finalizing:"\u6700\u7D42\u51E6\u7406\u4E2D",ready:"\
\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u6E96\u5099\u5B8C\u4E86",downloading:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D",
uploading:"ZIP\u3092\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D",validating:"ZIP\u3092\u691C\u8A3C\u4E2D",
validating_files:"\u30D5\u30A1\u30A4\u30EB\u60C5\u5831\u3092\u691C\u8A3C\u4E2D",reading_files:"\u30D5\u30A1\u30A4\u30EB\u3092\
\u8AAD\u307F\u8FBC\u307F\u4E2D",importing_settings:"\u8A2D\u5B9A\u3092\u53CD\u6620\u4E2D",importing_credentials:"\
\u8A8D\u8A3C\u60C5\u5831\u3092\u53CD\u6620\u4E2D",importing_gems:"Gem\u3092\u8FFD\u52A0\u4E2D",saving_files:"\
\u30D5\u30A1\u30A4\u30EB\u3092\u4FDD\u5B58\u4E2D",importing_chats:"\u30C1\u30E3\u30C3\u30C8\u5C65\u6B74\u3092\u8FFD\u52A0\u4E2D",
importing_feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u8FFD\u52A0\u4E2D",importing_diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF\u3092\u8FFD\u52A0\u4E2D",cancelling:"\u30AD\u30E3\u30F3\u30BB\u30EB\u51E6\u7406\u4E2D",
cancelled:"\u30AD\u30E3\u30F3\u30BB\u30EB\u6E08\u307F",expired:"\u4FDD\u5B58\u671F\u9650\u5207\u308C",
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};_.textContent=$[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const M=get("account-transfer-cancel-btn");M&&M.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),P=o(d=>{ie&&(ie.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),Y=o((d={})=>{const m=get("account-export-ready"),g=get(
"account-export-ready-text"),b=get("account-export-expiry"),k=get("account-export-download-btn"),_=!!(d.
available&&d.download_url);if(m&&m.classList.toggle("hidden",!_),!_){k&&k.removeAttribute("href");return}
const T=Math.max(0,Number(d.size_bytes)||0),M=T>=1024*1024*1024?`${(T/(1024*1024*1024)).toFixed(2)} \
GB`:`${(T/(1024*1024)).toFixed(1)} MB`;if(g){const $=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${M}${$}`}
if(b){const $=d.expires_at?new Date(d.expires_at):null;b.textContent=$&&!Number.isNaN($.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${$.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
k&&(k.href=d.download_url)},"renderAccountExportAvailability"),be=o(async d=>{for(;N===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await m.json().catch(()=>({}));if(m.ok&&(g.state!=="pending"&&G(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(m=>setTimeout(m,700))}
return null},"pollAccountTransfer"),fe=o((d,m,g=!0)=>{m&&(G(m),Y(m),g&&m.state==="ready"?showToast(m.
message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):g&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),re(d))},"handleFinishedAccountExport"),Z=o(async()=>{try{const d=await apiFetch("/api/ac\
count/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(Y(m),m.state==="ready"){G(m);return}if(["failed","cancelled","expired"].includes(
m.state)){G(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||N&&N.id===
m.job_id||N)return;const g={id:m.job_id,type:"export",stopped:!1,restored:!0};N=g,P(!0),G(m);const b=await be(
g);b&&fe(g,b,!0)}catch{}},"refreshLatestAccountExport"),re=o(d=>{N===d&&(N=null),d.stopped=!0,P(!1)},
"finishAccountTransfer"),ee=get("account-transfer-cancel-btn");ee&&(ee.onclick=async()=>{const d=N;if(!(!d||
d.stopped)){d.cancelRequested=!0,ee.disabled=!0,G({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),G({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&Y({available:!1}),re(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const ie=get("account-export-btn");ie&&(ie.onclick=async()=>{if(N)return;const d={id:D(),
type:"export",stopped:!1};N=d,P(!0),Y({available:!1}),G({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u53D7\
\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const m=await apiFetch("/api/account/export",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:d.id}),keepalive:!0})),
g=await m.json().catch(()=>({}));if(m.status===409&&g.error==="export_in_progress"&&g.job_id)d.id=g.
job_id;else if(!m.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
G({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await be(d);!d.cancelRequested&&b&&fe(d,b,!0)}catch(m){const g=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";G({progress:0,phase:"faile\
d",message:g}),showToast(g,"error",!0),re(d)}});const se=get("account-export-download-btn");se&&se.addEventListener(
"click",async d=>{const m=se.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await g.json().catch(
()=>({}));g.ok&&b.available&&b.download_url?(se.href=b.download_url,window.location.assign(b.download_url)):
(Y(b),G(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),Z())}catch{window.location.assign(m)}}}),P(!1),Z();const Me=get("import-files-grid"),xe=get(
"import-files-info"),Ge=get("import-files-summary"),Ze=o(d=>{const m=Math.max(0,Number(d)||0);return m>=
1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(1)} M\
B`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let je=null;const qe=o(()=>{if(!je)
return;const d=je.files,m=je.selection;let g=0;d.forEach(_=>{m.has(_.archive_path)&&(g+=Number(_.size_bytes)||
0)});const b=Number(je.available_bytes)||0,k=g>b;Ge&&(Ge.textContent=`\u9078\u629E\u4E2D: ${Ze(g)} /\
 \u5229\u7528\u53EF\u80FD: ${Ze(b)}${k?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,Ge.classList.toggle(
"text-red-300",k)),xe&&(xe.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),at=o(()=>{
if(!Me||!je)return;Me.innerHTML="";const d=je.files;if(!d.length){Me.innerHTML='<div class="text-xs \
text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
qe();return}d.forEach(m=>{const g=document.createElement("label"),b=je.selection.has(m.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${Ze(m.size_bytes)}</div></div>`;const k=g.
querySelector(".import-file-check");k.addEventListener("change",()=>{k.checked?je.selection.add(m.archive_path):
je.selection.delete(m.archive_path),g.classList.toggle("border-blue-500",k.checked),g.classList.toggle(
"border-gray-600",!k.checked),qe()}),Me.appendChild(g)}),qe()},"renderImportFileItems"),le=o(d=>new Promise(
m=>{if(je={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:m},at(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),pt=o(d=>{if(hideModal("import-files-modal"),je){const m=je.resolve;
je=null,m(d)}},"closeImportFileSelection"),Je=get("import-files-close");Je&&(Je.onclick=()=>pt(null));
const yt=get("import-files-cancel");yt&&(yt.onclick=()=>pt(null));const tn=get("import-files-confirm");
tn&&(tn.onclick=()=>{if(!je)return;const d=Array.from(je.selection);pt(d.length?d.join(","):"__none_\
_")});const nn=get("import-files-select-all");nn&&(nn.onclick=()=>{je&&(je.files.forEach(d=>je.selection.
add(d.archive_path)),at())});const sn=get("import-files-none");sn&&(sn.onclick=()=>{je&&(je.selection.
clear(),at())});const Ln={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},an=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const m=String(
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let Ot=null;const Mt=o(
d=>{if(Ot){const m=Ot;Ot=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Nt=o(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}Ot=m;const b=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],k=get("settings-confirmation-list");k&&(b.length?
k.innerHTML=b.map(T=>{const M=Ln[T.field]||T.field,$=an(T.current),O=an(T.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(M)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
$)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(O)}</d\
iv>
                            </div>`}).join(""):k.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const _=get("settin\
gs-confirmation-count");_&&(_.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),vt=get("settings-confir\
mation-modal");vt&&vt.addEventListener("click",d=>{d.target===vt&&Mt(!1)});const zt=get("settings-co\
nfirmation-close");zt&&(zt.onclick=()=>Mt(!1));const Rt=get("settings-confirmation-cancel");Rt&&(Rt.
onclick=()=>Mt(!1));const wt=get("settings-confirmation-confirm");wt&&(wt.onclick=()=>Mt(!0));const At=get(
"account-import-btn"),Wt=get("account-import-inplace"),Bt=get("account-import-inplace-warning");if(Wt&&
Bt){const d=o(()=>Bt.classList.toggle("hidden",!Wt.checked),"syncInplaceWarn");Wt.addEventListener("\
change",d),d()}At&&(At.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:
null,g=get("account-import-categories"),b=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(ce=>ce.value):[],k=get("account-import-inplace"),_=!!(k&&k.checked),T=get("account-impo\
rt-settings-bypass"),M=!!(T&&T.checked);let $=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(ce=>(ce.
closest("label")&&ce.closest("label").textContent||ce.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${O.join("\u3001")}${_?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const B={id:D(),type:"import",stopped:!1,controller:new AbortController};
N=B,P(!0),G({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const Q=get("account-import-result");let de=Promise.resolve(null);try{const W=Math.max(1,Math.ceil(m.
size/10485760)),pe=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:B.controller.
signal})),q=await pe.json().catch(()=>({}));if(!pe.ok)throw new Error(q.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
B.uploadId=q.upload_id;const ye=q.chunk_size||10485760;let V=0,he=0;const ke=o(async()=>{for(;;){const Ae=he++;
if(Ae>=W)return;const _e=m.slice(Ae*ye,Math.min(m.size,(Ae+1)*ye)),De=new FormData;De.append("chunk",
_e,m.name),De.append("index",String(Ae));const tt=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:De,signal:B.controller.signal})),
ct=await tt.json().catch(()=>({}));if(!tt.ok)throw new Error(ct.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
V++,G({progress:Math.min(35,Math.round(V/W*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${V}\
/${W}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ae=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ae=!0);try{await Promise.
all([ke(),ke(),ke()]);const Ae=await apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:B.controller.signal})),_e=await Ae.json().
catch(()=>({}));if(!Ae.ok)throw new Error(_e.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
G({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ae&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ve="",Pe=!1,Oe=0;const lt=o(
async()=>{let Ae=!1;const _e=o(()=>{Ae||(Ae=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const De=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),tt=await De.
json().catch(()=>null);if(!De.ok||!tt){_e();return}cacheUserSettings(tt);const ct=get("settings-moda\
l");if(ct&&ct.classList.contains("modal-open"))try{$e(tt)}catch{}tt.theme_color&&applyThemeColor(tt.
theme_color,!0),Object.prototype.hasOwnProperty.call(tt,"minimal_prompt_mode")&&tt.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(tt,"compact_prompt_mode")&&setCompactPromptMode(
!!tt.compact_prompt_mode)}catch{}_e()},"refreshSettingsFormAfterImport"),Xe=o(Ae=>{const _e=Ae&&Ae.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";Q&&(Q.textContent=`\u5B8C\u4E86: ${_e}`,
Q.classList.remove("hidden","text-red-300"),Q.classList.add("text-emerald-300")),G({progress:100,phase:"\
completed",message:_e}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&lt()},"finishImportSuccess"),ht=o(async()=>{
try{const _e=await(await apiFetch(`/api/account/transfer/${B.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return _e&&_e.state?_e:null}catch{return null}},"fetchImportSta\
tus"),Pt=o(async()=>{const Ae=await ht();if(!Ae)return{status:"unknown"};if(Ae.state==="completed")return Xe(
Ae),{status:"done"};if(["failed","cancelled","expired"].includes(Ae.state))throw new Error(Ae.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Ae.state==="needs_sel\
ection"&&Array.isArray(Ae.files)){const _e=await le({files:Ae.files,available_bytes:Ae.available_bytes});
return _e===null?(G({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ve=_e,{status:"reselect"})}if(Ae.state===
"needs_settings_confirmation"&&Array.isArray(Ae.settings_changes))return await Nt({settings_changes:Ae.
settings_changes})?($=!0,{status:"reselect"}):(G({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Ae.state==="running"){const _e=await Promise.race([de.catch(()=>null),new Promise(De=>setTimeout(
()=>De(null),6e4))]);if(_e&&_e.state==="completed")return Xe(_e),{status:"done"};throw _e&&["failed",
"cancelled","expired"].includes(_e.state)?new Error(_e.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Pe;){B.stopped=!0,await de.catch(()=>null),
B.stopped=!1,de=be(B);let Ae;try{Ae=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:B.uploadId,
categories:b.join(","),job_id:B.id,selected_files:ve,restore_inplace:_,confirm_settings:$||M}),signal:B.
controller.signal}))}catch(Ye){if(B.cancelRequested||Ye&&Ye.name==="AbortError")throw Ye;const mt=await Pt();
if(mt.status==="done"){Pe=!0;break}if(mt.status==="cancelled")return;if(mt.status==="reselect")continue;
if(Oe<2){Oe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let _e=null;try{_e=await Ae.json()}catch{_e=null}if(_e===null){const Ye=await Pt();if(Ye.status==="d\
one"){Pe=!0;break}if(Ye.status==="cancelled")return;if(Ye.status==="reselect")continue;if(Ae.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Oe<2){Oe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Ae.ok&&_e.error==="storage_limit_files"&&_e.files){const Ye=await le(_e);if(Ye===null){G({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ve=Ye;continue}if(_e&&_e.status==="settings_confirmation"&&
Array.isArray(_e.settings_changes)){if(!await Nt(_e)){G({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}$=!0;continue}if(!Ae.ok)throw new Error(_e.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const De=_e.imported||{},tt=[`\u8A2D\u5B9A ${De.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${De.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${De.chats||0}\u4EF6`,`Gem ${De.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${De.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${De.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${De.
diagnostics||0}\u4EF6`].join(" / "),ct=_e.duplicates||{},Se={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Ne=[];for(const Ye of Object.keys(Se)){const mt=Number(ct[Ye])||0;mt>
0&&Ne.push(`${Se[Ye]} ${mt}\u4EF6`)}const Ue=Ne.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Ne.
join("\u3001")}\uFF09`:"";Q&&(Q.textContent=`\u5B8C\u4E86: ${tt}${Ue}`,Q.classList.remove("hidden","\
text-red-300"),Q.classList.add("text-emerald-300")),G({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&lt(),Pe=!0}}catch(ce){if(B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),B.cancelRequested||ce&&ce.name==="AbortError")return;const W=ce&&ce.message?
ce.message:"",pe=W==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
W||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";G({progress:0,phase:"f\
ailed",message:pe}),Q&&(Q.textContent=pe,Q.classList.remove("hidden","text-emerald-300"),Q.classList.
add("text-red-300")),showToast(pe,"error",!0)}finally{B.stopped=!0,await de.catch(()=>null),re(B)}});
const xt=get("account-dedupe-btn"),Et=get("account-dedupe-result"),Ft=o((d,m=!1)=>{Et&&(Et.textContent=
d,Et.classList.remove("hidden"),Et.classList.toggle("text-red-300",!!m),Et.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");xt&&(xt.onclick=async()=>{const d=o(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await m.json().catch(()=>null);if(!m.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){Ft("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],k={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const B of[
"chats","gems","files","feedback","diagnostics"]){const Q=Number(g.duplicates&&g.duplicates[B])||0;Q>
0&&b.push(`${k[B]} ${Q}\u4EF6`)}const _=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${_}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const T=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),M=await T.json().catch(()=>null);if(!T.ok||!M)throw new Error(
M&&M.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const $=[];for(const B of["chats","gems","files","feedback","diagnostics"]){const Q=Number(M.removed&&
M.removed[B])||0;Q>0&&$.push(`${k[B]} ${Q}\u4EF6`)}const O=Number(M.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${M.kept_referenced_files}\u4EF6\uFF09`:"";Ft(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${$.join("\u3001")||"0\u4EF6"}${O}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!xt.disabled){xt.disabled=!0,Ft("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){Ft(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{xt.disabled=!1}}});const on=get("site-cache-usage-refresh");on&&(on.onclick=()=>loadSiteCacheUsage());
const Vt=get("clear-site-cache-btn");Vt&&(Vt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Vt)});
const kt=get("enc-scan-result"),rn=o(async(d=null)=>{kt&&(kt.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
m,{cache:"no-store"}),b=await g.json();if(!g.ok){kt&&(kt.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const k=b.total||0,_=b.encrypted||0,T=b.unencrypted||0;let M=`Total: ${k} / Encrypted: ${_} /\
 Plain: ${T}`;if(b.samples&&b.samples.length){const $=b.samples.slice(0,8).map(O=>{const B=O.timestamp?
new Date(O.timestamp).toLocaleString():"";return`#${O.id} (${O.role||""}) ${B}`}).join(" / ");M+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${$}</div>`}kt&&(kt.innerHTML=M)}catch{kt&&(kt.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),U=get("enc-scan-all");U&&(U.onclick=()=>rn(null));
const we=get("enc-scan-thread");we&&(we.onclick=()=>currentThreadId?rn(currentThreadId):showToast("\u30B9\
\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Te=get("admin-enc-list");
let Be=null,ot=!1;const ft=o(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThreadEncrypt\
edFromMessages"),_t=o(()=>{Be=ft(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Qe=o(async(d,m,{
confirmPrompt:g=!0,reloadCurrent:b=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const k=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;ot=!0;try{const _=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
T=await _.json().catch(()=>({}));return _.ok?(showToast(`${k}\u3057\u307E\u3057\u305F\uFF08${T.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Be=!!m,b&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Te&&await Ke(),!0):
(showToast(T.error||`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{ot=!1}},"setAdminThreadEncry\
ption"),St=o(d=>{if(!Te)return;const m=d.threads||[];if(!m.length){Te.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Te.innerHTML=m.map(g=>{const b=g.encrypted_count>0?"enc":"plain",k=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",_=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",T=g.
updated_at?new Date(g.updated_at).toLocaleString():"",M=escapeHtml(String(g.thread_id)),$=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${$?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${T} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${M}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${_} text-white px-2 py-1 roun\
ded shrink-0" data-id="${M}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${k}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Ke=o(async()=>{if(Te){Te.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){Te.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(St(m),currentThreadId&&Array.isArray(m.threads)){const g=m.threads.find(b=>String(
b.thread_id)===String(currentThreadId));g&&(Be=!!g.encrypted)}}catch{Te.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Ke()),window.__loadAdminEncThreads=
Ke,window.__refreshAdminThreadEncState=_t,window.__setAdminThreadEncryption=Qe;const it=get("encrypt\
ion-status-admin-toggle");it&&it.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Te&&(Te.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const M=m.getAttribute("data-id");if(!M)return;typeof We==
"function"?We():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(M)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||ot)return;const b=g.getAttribute("data-i\
d"),k=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const T=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await Qe(b,k,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=T,await Ke()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const Lt=o(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(g=>{const b=g.status||"new",k=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
_=g.created_at?new Date(g.created_at).toLocaleString():"",T=g.replied_at?new Date(g.replied_at).toLocaleString():
"",M=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${k}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(_)}</div>
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
M)}</textarea>
                                ${M?`<div class="text-[10px] text-gray-500 mt-1">\u8FD4\u4FE1\u65E5\u6642: ${escapeHtml(
T)}</div>`:""}
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
                    `}).join("")}},"renderBanAppeals"),ln=o(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const k=(await g.json()).unread_count||0;m.textContent=String(k),d&&k>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${k}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),$t=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const g=await m.json();Lt(g.items||[]),
await ln(!1)}catch{}}},"loadBanAppeals"),cn=o(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await $t()}catch{}},"markBanAppealsRead"),Jt=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await $t()}catch{}},"updateBanAppealStatus"),Fe=o(()=>{
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
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),ze=o(()=>{const d=get("set-st\
t-model");if(!d||get("set-llm-transcribe-prompt"))return;const m=d.closest(".space-y-2");if(!m)return;
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
                `,m.appendChild(g);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const k=get(
"set-llm-transcribe-prompt");k&&(k.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),et=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
mcpLocked:!0}];window.buildAutoSystemPromptRows=(d,m=!1)=>{const g=m?"w-full h-14 bg-gray-950 border\
 border-gray-700 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-\
700 rounded p-2 text-xs text-gray-200";return et.map(b=>{const k=b.mcpLocked===!0,_=k?'<div class="t\
ext-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002\
</div>':"",T=k?`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow-500 \
w-3 h-3" disabled>`:`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow\
-500 w-3 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${b.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${k?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${T}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${d}-auto-sys-${b.key}-text" class="${g}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${_}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{et.forEach(g=>{const b=m&&
typeof m=="object"?m[g.key]||{}:{},k=get(`${d}-auto-sys-${g.key}-enabled`),_=get(`${d}-auto-sys-${g.
key}-text`);k&&(g.mcpLocked===!0?k.disabled=!0:k.checked=b.enabled!==!1),_&&(_.value=b.text||"",_.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const dn=o((d,m=null)=>{if(m){const g=get(m);g&&(g.checked=!0)}et.forEach(g=>{const b=get(`${d}-auto\
-sys-${g.key}-enabled`),k=get(`${d}-auto-sys-${g.key}-text`);if(b&&(g.mcpLocked!==!0?b.checked=!0:b.
disabled=!0),k){const _=k.placeholder||"";k.value=_}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),un=o(d=>{const m={};return et.forEach(g=>{const b=get(`${d}\
-auto-sys-${g.key}-enabled`),k=get(`${d}-auto-sys-${g.key}-text`);m[g.key]={enabled:g.mcpLocked===!0?
!0:b?b.checked:!0,text:k?k.value:""}}),m},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
un,window.ensureAutoSystemPromptSettingsCard=()=>{const d=get("set-global-sys-prompt-enabled"),m=d?d.
closest(".space-y-4"):null;if(!m||get("auto-sys-prompt-settings"))return;const g=document.createElement(
"div");g.id="auto-sys-prompt-settings",g.className="border-t border-gray-700 pt-3",g.innerHTML=`
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
                `,m.appendChild(g)},window.ensureThreadAutoSystemPromptCard=()=>{const d=get("thread\
-global-sys-prompt"),m=d?d.closest(".space-y-3"):null;if(!m||get("thread-auto-sys-prompt-settings"))
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
                `,m.appendChild(g)},Fe(),ze(),R();const jt=o(()=>{const d=get("set-default-model");if(!d)
return;const m=d.value;d.innerHTML="",MODELS.forEach(b=>{const k=document.createElement("optgroup");
k.label=b.category,(b.items||[]).forEach(_=>{const T=document.createElement("option");T.value=_.id,T.
textContent=_.name,k.appendChild(T)}),d.appendChild(k)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),Ce=o(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(b=>{const k=(b.items||[]).filter(T=>{const M=(T.id||"").toLowerCase();
return M.startsWith("gemini-")||M.startsWith("gpt-4o")||M.startsWith("claude-")||M.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
M)});if(k.length===0)return;const _=document.createElement("optgroup");_.label=b.category,k.forEach(
T=>{const M=document.createElement("option");M.value=T.id,M.textContent=T.name+" \u2605",_.appendChild(
M)}),d.appendChild(_)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
"gemini-3-flash-preview";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),$e=o(d=>{if(!d)return;cacheUserSettings(d);const m=get("app-glo\
bal-sys-prompt-preview");m&&(m.value=d.global_system_prompt_effective||"");const g=get("app-global-s\
ys-prompt-preview-status");g&&(d.global_system_prompt_enabled===!1?g.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
d.global_system_prompt_uses_time_fallback?g.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
g.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("sys-prompt-text")&&(get("sys-prompt-text").value=d.system_prompt||""),get("set-global-sys-promp\
t-enabled")&&(get("set-global-sys-prompt-enabled").checked=d.system_prompt_enabled!==!1),window.ensureAutoSystemPromptSettingsCard(),
get("set-apply-global-sys-prompt")&&(get("set-apply-global-sys-prompt").checked=d.apply_global_system_prompt!==
!1),get("set-apply-auto-sys-prompt-notices")&&(get("set-apply-auto-sys-prompt-notices").checked=d.apply_auto_system_prompt_notices!==
!1),window.applyAutoSystemPromptConfigToForm("set",d.auto_system_prompt_notices_config||{}),get("set\
-latency-metrics")&&(get("set-latency-metrics").checked=d.enable_latency_metrics===!0),get("set-clie\
nt-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings modal sync"),get("\
set-openai")&&(get("set-openai").value=d.openai_key||""),get("set-gemini")&&(get("set-gemini").value=
d.gemini_key||""),get("set-deepseek")&&(get("set-deepseek").value=d.deepseek_key||""),get("set-zai")&&
(get("set-zai").value=d.zai_key||""),get("set-kimi")&&(get("set-kimi").value=d.kimi_key||""),get("se\
t-mistral")&&(get("set-mistral").value=d.mistral_key||""),get("set-ideogram")&&(get("set-ideogram").
value=d.ideogram_key||""),get("set-anthropic")&&(get("set-anthropic").value=d.anthropic_key||""),get(
"set-gemini-backend")&&(get("set-gemini-backend").value=normalizeGeminiBackend(d.gemini_backend||"ge\
mini_api")),get("set-gemini-vertex-project")&&(get("set-gemini-vertex-project").value=d.gemini_vertex_project||
""),get("set-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=d.gemini_vertex_location||
"global"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-\
gemini-vertex-credentials-json").value=d.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),get(
"set-admin-api-key-mode")&&(get("set-admin-api-key-mode").value=normalizeAdminApiKeyMode(d.admin_api_key_mode||
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
"gemini-3-flash-preview"),syncSettingsModelPickerLabels(),applyTemporaryChatTimeoutSeconds(d.temp_chat_timeout_seconds),
get("set-default-search")&&(get("set-default-search").checked=!!d.default_enable_search),get("set-de\
fault-url-context")&&(get("set-default-url-context").checked=!!d.default_enable_url_context),get("se\
t-default-maps")&&(get("set-default-maps").checked=!!d.default_enable_maps),get("set-default-python")&&
(get("set-default-python").checked=!!d.default_enable_python),get("set-default-file-creation")&&(get(
"set-default-file-creation").checked=!!d.default_enable_file_creation),get("set-default-thinking")&&
(get("set-default-thinking").checked=!!d.default_enable_thinking),get("set-default-sys-prompt")&&(get(
"set-default-sys-prompt").checked=!!d.default_enable_system_prompt),get("set-default-mcp")&&(get("se\
t-default-mcp").checked=d.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-defa\
ult-thinking-level").value=d.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get(
"set-default-thinking-budget").value=d.default_thinking_budget||4096),get("set-default-reasoning-eff\
ort")&&(get("set-default-reasoning-effort").value=d.default_reasoning_effort||"medium"),get("set-def\
ault-safety")&&(get("set-default-safety").value=d.default_safety_setting||"default"),get("set-e2ee").
checked=d.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=d.bot_detection_enabled!==
!1),get("set-bot-detect-global")&&(get("set-bot-detect-global").checked=d.bot_detection_global_enabled!==
!1);const b=get("bot-status");b&&(d.is_bot_banned?(b.textContent=`BAN\u4E2D: ${d.bot_ban_reason||"Bo\
t detection"}`,b.classList.remove("hidden"),b.classList.add("text-red-400")):b.classList.add("hidden")),
d&&d.theme_color?(applyThemeColor(d.theme_color,!0),syncThemeInputs(d.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=d.username);const k=get("2fa-badge"),_=get("disable-2fa-btn");d.is_2fa_enabled?
(k.innerText="ENABLED",k.classList.replace("bg-gray-700","bg-green-600"),k.classList.replace("text-g\
ray-400","text-white"),_.classList.remove("hidden")):(k.innerText="DISABLED",k.classList.replace("bg\
-green-600","bg-gray-700"),k.classList.replace("text-white","text-gray-400"),_.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const T=get(
"set-passkey-only-login"),M=get("passkey-only-note"),$=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(K($),T){T.checked=!!d.passkey_only_login;const ce=$.length>0||!!d.has_webauthn;T.disabled=!ce,
ce||(T.checked=!1),M&&(ce?M.classList.add("hidden"):M.classList.remove("hidden"))}const O=get("mig-s\
tatus-box"),B=get("mig-progress-text"),Q=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){O.classList.remove("hidden");const ce=(d.migration_progress||"").split("/");if(ce.length===
2){const W=parseInt(ce[0]||"0",10),pe=parseInt(ce[1]||"0",10);B&&(B.innerText=`${W} / ${pe}`),Q&&pe>
0&&(Q.style.width=`${Math.min(100,Math.floor(W/pe*100))}%`)}}else O.classList.add("hidden"),Q&&(Q.style.
width="0%"),B&&(B.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&$e(d);const m=get(
"search-box"),g=m?m.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),jt(),Ce(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(k=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+k+"ms"),snapshotSidebarHistory("settings-open-later-"+k+"ms")},k)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),Z(),ze(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),ln(!0),$t(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),gn(),oe(),me();try{loadMcpServers()}catch{}};const We=o((d=!1)=>{settingsModelPickerTarget&&
window.closeModelModal(!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),
revealPersistentSidebarLists(),snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory(
"settings-close-later-300ms"),300),!d&&location.pathname==="/settings"&&history.back()},"closeSettin\
gsModal"),dt=o(()=>{const d=get("set-theme-color"),m=get("set-theme-color-text"),g=get("theme-reset-\
btn"),b=document.querySelectorAll("#theme-presets .theme-swatch"),k=o((_,T=!0)=>{const M=normalizeHex(
_);M&&(applyThemeColor(M,T),syncThemeInputs(M))},"applyFromValue");d&&d.addEventListener("input",()=>k(
d.value,!0)),m&&(m.addEventListener("change",()=>{const _=normalizeHex(m.value);if(!_){syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||THEME_DEFAULT);return}k(_,!0)}),m.addEventListener("keydown",
_=>{_.key==="Enter"&&(_.preventDefault(),m.blur())})),g&&(g.onclick=()=>k(THEME_DEFAULT,!0)),b.forEach(
_=>{_.addEventListener("click",()=>k(_.getAttribute("data-color"),!0))})},"bindThemeControls"),ut=o(
()=>{const d=get("reset-global-sys-prompt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-pro\
mpt-text").value=""),get("set-global-sys-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=
!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const m=get("reset-thread-sys-prompt");m&&(m.onclick=()=>{get("thread-global-sys-prompt")&&
(get("thread-global-sys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-glo\
bal-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{dn("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const b=get("reset-thread-auto-sys-prompt-defaults");b&&(b.onclick=()=>{dn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>We();const Ct=get("settings-header-close");Ct&&(Ct.onclick=()=>We());
const gt=get("settings-search");gt&&(gt.addEventListener("input",filterSettings),gt.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const g=m.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Yn=get(
"settings-search-clear");Yn&&Yn.addEventListener("click",()=>{gt&&(gt.value="",filterSettings(),gt.focus())}),
dt(),ut(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Nn=get("set-gemini-local-python-dialog");Nn&&(Nn.onchange=()=>setGeminiLocalPyDialogEnabled(Nn.
checked));const Qn=get("set-gemini-backend");Qn&&(Qn.onchange=()=>syncGeminiBackendUi());const Zn=get(
"set-admin-api-key-mode");Zn&&(Zn.onchange=()=>syncAdminApiKeyModeUi());const Rn=get("set-temp-chat-\
timeout-seconds");Rn&&(Rn.onchange=()=>{applyTemporaryChatTimeoutSeconds(Rn.value)});const ei=get("s\
lash-command-cancel-btn");ei&&(ei.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{var M;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const d=get("set-username"),m=get("set-password"),g=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:un(
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
"set-latency-metrics")?get("set-latency-metrics").checked:!1,server_diagnostic_log_retention_hours:Number(
((M=get("set-server-diagnostic-log-retention-hours"))==null?void 0:M.value)||1),enable_server_diagnostic_log:get(
"set-server-diagnostic-log")?get("set-server-diagnostic-log").checked:!1,enable_client_debug_log:get(
"set-client-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-o\
nly-login")?get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-goog\
le")?get("set-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set\
-default-2fa-method").value:"totp",new_username:d?d.value:null,new_password:m?m.value:null},k=get("s\
et-e2ee")?get("set-e2ee").checked:!1,_=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
k!==_&&(b.enable_e2ee=k),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
(b.gemini_key=get("set-gemini").value),get("set-deepseek")&&(b.deepseek_key=get("set-deepseek").value),
get("set-zai")&&(b.zai_key=get("set-zai").value),get("set-kimi")&&(b.kimi_key=get("set-kimi").value),
get("set-mistral")&&(b.mistral_key=get("set-mistral").value),get("set-ideogram")&&(b.ideogram_key=get(
"set-ideogram").value),get("set-anthropic")&&(b.anthropic_key=get("set-anthropic").value),b.model_api_keys=
normalizeModelApiKeyMap(modelApiKeyMap),get("set-gemini-backend")&&(b.gemini_backend=normalizeGeminiBackend(
get("set-gemini-backend").value)),get("set-gemini-vertex-project")&&(b.gemini_vertex_project=get("se\
t-gemini-vertex-project").value),get("set-gemini-vertex-location")&&(b.gemini_vertex_location=get("s\
et-gemini-vertex-location").value),get("set-gemini-vertex-credentials-json")&&(b.gemini_vertex_credentials_json=
get("set-gemini-vertex-credentials-json").value),get("set-xai")&&(b.xai_key=get("set-xai").value),get(
"set-google-key")&&(b.google_key=get("set-google-key").value),get("set-google-project")&&(b.google_project=
get("set-google-project").value),get("set-admin-api-key-mode")&&(b.admin_api_key_mode=normalizeAdminApiKeyMode(
get("set-admin-api-key-mode").value)),get("set-bot-detect")&&(b.bot_detection_enabled=get("set-bot-d\
etect").checked),get("set-bot-detect-global")&&(b.bot_detection_global_enabled=get("set-bot-detect-g\
lobal").checked);const T=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(b)});if(T.ok){let $="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const de=await T.json();de&&de.message&&($=de.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),
window.applySavedUserSystemPromptSettings({system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,
apply_global_system_prompt:b.apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:b.auto_system_prompt_notices_config}),We();const O=currentUsername,
B=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const Q=useSwCache;
useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),Q!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast($,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==O?setTimeout(()=>location.
reload(),1e3):b.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let $={};try{$=await T.json()}catch{}showToast($.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const m=get("2fa-badge");m&&(m.innerText="D\
ISABLED",m.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const d=get("bot-unban-username"),m=d?d.value.trim():
"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"single"})}),b=await g.json(),k=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")k&&(k.textContent=`${m} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
k.classList.remove("hidden")),d&&(d.value="");else{const _=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),b=await g.json(),k=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")k&&(k.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
k.classList.remove("hidden")),d&&(d.value="");else{const _=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(V=>{m&&(m.textContent=V)},"setBox"),b=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),k=o((V,he)=>!V||!he||he<=0?0:V*8/(he/1e3)/1e3/1e3,"toMbps"),_=o(V=>Number.isFinite(
V)?`${V.toFixed(0)} ms`:"-","fmtMs"),T=o(V=>Number.isFinite(V)?`${V.toFixed(V>=100?0:1)} Mbps`:"-","\
fmtMbps"),M=o(async(V,he)=>{const ke=await V.json().catch(()=>({}));return ke&&ke.error?ke.error:he},
"parseErr"),$=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let V=0;V<4;V++){const he=performance.now(),ke=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),ae=performance.now();if(!ke.ok)throw new Error(await M(
ke,"ping_failed"));await ke.json().catch(()=>({})),$.push(ae-he)}const O=$.reduce((V,he)=>V+he,0)/Math.
max(1,$.length),B=Math.min(...$),Q=o(async V=>{const he=performance.now(),ke=await apiFetch(`/api/sp\
eedtest/download?bytes=${V}&_=${b()}`,{cache:"no-store"});if(!ke.ok)throw new Error(await M(ke,"down\
load_failed"));const ae=await ke.arrayBuffer(),ve=performance.now();return{bytes:ae.byteLength||V,ms:ve-
he,mbps:k(ae.byteLength||V,ve-he)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${_(O)}
\u6E2C\u5B9A\u4E2D... download`);const de=[];for(const V of[2*1024*1024,8*1024*1024])de.push(await Q(
V)),g(`\u6E2C\u5B9A\u4E2D... ping ${_(O)}
download ${T(Math.max(...de.map(he=>he.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const ce=Math.max(...de.map(V=>V.mbps)),W=o(async V=>{const he=new Uint8Array(
V),ke=performance.now(),ae=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:he,cache:"no-store"}),ve=performance.now();if(!ae.ok)
throw new Error(await M(ae,"upload_failed"));const Pe=await ae.json().catch(()=>({})),Oe=Number(Pe.bytes_received||
V)||V;return{bytes:Oe,ms:ve-ke,mbps:k(Oe,ve-ke),serverMs:Number(Pe.server_elapsed_ms||0)||0}},"runUp\
load"),pe=[];for(const V of[1*1024*1024,4*1024*1024])pe.push(await W(V));const q=Math.max(...pe.map(
V=>V.mbps)),ye=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${_(O)} / ${_(B)}`,`Download (best): ${T(ce)}`,`Upload (best): ${T(q)}`,`Download r\
uns: ${de.map(V=>`${Math.round(V.bytes/1024/1024)}MB=${T(V.mbps)}`).join(", ")}`,`Upload runs: ${pe.
map(V=>`${Math.round(V.bytes/1024/1024)}MB=${T(V.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(ye.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>$t()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>cn()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){g&&await cn([Number(g)]);return}if(m.classList.contains(
"ban-appeal-status")){const b=m.getAttribute("data-status");g&&b&&await Jt({id:Number(g),status:b});
return}if(m.classList.contains("ban-appeal-reply-send")){const b=m.closest("[data-appeal-id]"),k=b?b.
querySelector(".ban-appeal-reply"):null,_=k?k.value:"";g&&await Jt({id:Number(g),admin_reply:_});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await Jt({id:Number(
g),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const d=get(get("camera-split-enabled").checked?"image-split-\
input":"photo-input");d&&d.click()}),get("upload-clear-btn")&&(get("upload-clear-btn").onclick=()=>{
resetUploadState()}),get("marker-modal-close")&&(get("marker-modal-close").onclick=()=>{closeMarkerModal(),
markerState.row=null}),get("marker-tool-draw")&&(get("marker-tool-draw").onclick=()=>setMarkerMode("\
draw")),get("marker-tool-auto")&&(get("marker-tool-auto").onclick=()=>setMarkerMode("auto")),get("ma\
rker-tool-mosaic")&&(get("marker-tool-mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool\
-crop")&&(get("marker-tool-crop").onclick=()=>setMarkerMode("crop")),get("marker-auto-tolerance")&&(get(
"marker-auto-tolerance").oninput=d=>{const m=get("marker-auto-tolerance-value");m&&(m.textContent=String(
d.target.value))});const Bn=get("marker-color-picker");Bn&&(Bn.oninput=d=>setMarkerColor(d.target.value),
Bn.onchange=d=>setMarkerColor(d.target.value));const Fn=get("marker-opacity");Fn&&(Fn.oninput=d=>setMarkerOpacity(
d.target.value),Fn.onchange=d=>setMarkerOpacity(d.target.value));const Cn=get("marker-opacity-number");
Cn&&(Cn.onchange=d=>setMarkerOpacity(d.target.value),Cn.onblur=d=>setMarkerOpacity(d.target.value),Cn.
onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(d.target.value),d.target.blur())}),document.querySelectorAll(
"#marker-toolbar .marker-color-chip[data-marker-color]").forEach(d=>{d.onclick=()=>setMarkerColor(d.
getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),
get("marker-crop-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&
(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=
()=>clearMarkerCanvas()),get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),
initMarkerCanvas(),initCropCanvas(),window.addEventListener("resize",()=>{const d=get("marker-modal");
!d||d.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const Li=o(()=>{const d=get(
"upload-modal");return!!(d&&!d.classList.contains("hidden"))},"isUploadModalOpen"),Kt=get("drop-over\
lay");let mn=0;const Ci=o(()=>{Li()||Kt&&(Kt.classList.remove("hidden"),Kt.classList.add("flex"))},"\
showDropOverlay"),pn=o(()=>{mn=0,Kt&&(Kt.classList.add("hidden"),Kt.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=pn;const Tt=get("upload-dropzone");Tt&&(Tt.addEventListener("dr\
agover",d=>{d.preventDefault(),Tt.classList.add("dragover")}),Tt.addEventListener("dragleave",()=>{Tt.
classList.remove("dragover")}),Tt.addEventListener("drop",d=>{d.preventDefault(),d.stopPropagation(),
Tt.classList.remove("dragover"),pn();const m=d.dataTransfer?d.dataTransfer.files:null;m&&m.length&&handleFiles(
m)})),window.addEventListener("dragenter",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.
types.includes("Files")||(mn+=1,Ci())}),window.addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||d.preventDefault()}),window.addEventListener("dragle\
ave",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(mn=Math.max(
0,mn-1),(mn===0||!d.relatedTarget||d.clientY<=0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=
window.innerHeight)&&pn())}),window.addEventListener("dragend",()=>{pn()}),window.addEventListener("\
drop",d=>{pn(),!(!d.dataTransfer||!d.dataTransfer.files||d.dataTransfer.files.length===0)&&(d.preventDefault(),
!(Tt&&Tt.contains(d.target))&&handleFiles(d.dataTransfer.files))});const ti=get("bot-admin-modal"),Ti=o(
d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML="",!d||!d.length){m.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,b)=>{const k=!!g.is_bot_banned,_=g.bot_detection_enabled!==!1,T=Number(g.lock_remaining_seconds)||
0,M=document.createElement("div");M.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",M.style.animationDelay=`${Math.min(b,12)*.02}\
s`,M.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(g.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${k?"BAN\u4E2D":"\u6B63\u5E38"}${T?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(T/60)}\u5206\uFF09`:""} ${g.bot_ban_reason?
" / "+escapeHtml(g.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${_?"1":"0"}">${_?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${k?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-bann\
ed="${k?"1":"0"}">${k?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${k?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(g.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        ${g.is_admin?"":`<button class="bot-delete-account bg-red-800 hover:bg-red-7\
00 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(g.username)}\
">\u524A\u9664</button>`}
                    `,m.appendChild(M)})}},"renderBotUsers"),fn=o(async(d="")=>{const m=get("bot-adm\
in-list");m&&(m.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),b=await g.json();g.ok&&b&&b.users?Ti(b.users):(m&&(m.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{m&&(m.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),jn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||ti))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await fn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=jn;const Dn=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=Dn,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{Dn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||ti)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
jn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await fn(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await fn("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async d=>{d.key==="Enter"&&await fn(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute(
"data-user");if(!g)return;let b;if(m.classList.contains("bot-toggle-detect")){const k=m.getAttribute(
"data-enabled")!=="1";b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:g,action:"toggle_detection",enabled:k})})}else if(m.classList.
contains("bot-toggle-ban"))if(m.getAttribute("data-banned")==="1")b=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"ban",reason:"Admin ban"})})}else if(m.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"unban_linked"})})}else if(m.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"delete_account"})})}if(b){if(b.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(b.ok){if(m.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),g===currentUsername)){location.href="/";return}}else{
let k={};try{k=await b.json()}catch{}showToast(k.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
k.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await fn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Xt={"/settings":{id:"settings-mod\
al",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:o(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:o(()=>{wi(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
pen")},"/history":{id:"history-modal",open:o(()=>window.showHistoryModal(),"open")},"/branch":{id:"b\
ranch-modal",open:o(()=>window.showBranchModal(),"open")},"/batch":{id:"batch-modal",open:o(()=>window.
showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:o(()=>openRichPasteModal(),"open")},"\
/camera":{id:"camera-capture-modal",open:o(()=>openCameraCaptureModal(),"open")},"/edit-image":{id:"\
marker-modal",open:o(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:o(()=>window.openThreadModal(),
"open")},"/model":{id:"model-modal",open:o(()=>openModelModal(),"open")},"/token-details":{id:"token\
-detail-modal",open:o(()=>{history.state&&history.state.promptEstimate?restorePromptTokenDetail():history.
state&&history.state.messageId!==void 0?openTokenDetail(history.state.messageId):showTokenDetailModal()},
"open")},"/encryption-status":{id:"encryption-status-modal",open:o(()=>showEncryptionStatusModal(),"\
open")},"/python-execution":{id:"python-exec-modal",open:o(()=>showPythonExecDetailModal(),"open")},
"/gem":{id:"gem-modal",open:o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="f\
as fa-gem text-blue-500 mr-2"></i>Create New Gem',showModal("gem-modal")},"open")},"/compression":{id:"\
compression-modal",open:o(()=>window.openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-mo\
dal",open:o(()=>jn(),"open")},"/admin-bot-logs":{id:"bot-log-modal",open:o(()=>Dn(),"open")}},ni=o((d,m=!1)=>{
switch(d){case"settings-modal":We(m);break;case"upload-modal":closeUploadModal(m);break;case"image-s\
plit-modal":closeImageSplit();break;case"image-split-compression-modal":hideModal(d);break;case"came\
ra-capture-modal":closeCameraCaptureModal(m?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(m);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(m);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(m);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(m);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
m);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(m);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(m);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(m);break;case"token-detail-modal":closeTokenDetail(m);break;case"encryption-s\
tatus-modal":closeEncryptionModal(m);break;case"python-exec-modal":closePythonExecDetail(m);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(m);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(m);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
m);break;case"bot-log-modal":window.closeBotLogModal&&window.closeBotLogModal(m);break;case"mcp-deci\
sion-modal":typeof submitMcpDecision=="function"?submitMcpDecision("deny"):hideModal(d);break;case"a\
pi-key-required-modal":{const b=get("api-key-modal-cancel-btn");b&&typeof b.onclick=="function"?b.click():
hideModal(d);break}case"lyria-studio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(
d);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(d);break;case"\
version-update-modal":const g=localStorage.getItem("app_version")||"";g&&localStorage.setItem("versi\
on_notified",g),hideModal(d);break;default:hideModal(d);break}},"closeModalById");window.addEventListener(
"popstate",d=>{var _,T;const m=(_=d.state)==null?void 0:_.settingsModelPicker;if(location.pathname===
"/settings"&&settingsModelPickerTargets.includes(m)){const M=o(()=>openSettingsModelPicker(m,!0),"re\
storePicker");(T=get("settings-modal"))!=null&&T.classList.contains("modal-open")?M():window.openSettingsModal().
then(()=>{var $;location.pathname==="/settings"&&(($=history.state)==null?void 0:$.settingsModelPicker)===
m&&M()});return}settingsModelPickerTarget&&window.closeModelModal(!0);let g=!1;Object.values(Xt).forEach(
M=>{const $=get(M.id);$&&$.classList.contains("modal-open")&&location.pathname!==Object.keys(Xt).find(
O=>Xt[O].id===M.id)&&(ni(M.id,!0),g=!0)});const b=location.pathname.match(/^\/c\/(.+)$/);if(b){const M=decodeURIComponent(
b[1]);String(currentThreadId)!==String(M)&&loadMessages(M,{skipHistory:!0})}else location.pathname===
"/"&&currentThreadId&&startNewChat({skipHistory:!0});const k=Xt[location.pathname];if(k){const M=get(
k.id);M&&!M.classList.contains("modal-open")&&k.open()}}),document.addEventListener("keydown",d=>{d.
key==="Escape"&&settingsModelPickerTarget&&(d.preventDefault(),window.closeModelModal())});const ii=location.
pathname;Xt[ii]&&(history.replaceState({},"","/"),setTimeout(()=>Xt[ii].open(),500)),get("easy-login\
-generate")&&(get("easy-login-generate").onclick=async()=>{const d=get("easy-login-mins"),m=d?parseInt(
d.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${m}\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))
return;const b=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({minutes:m})})).json();b&&b.temp_password?(get("easy-login-code").textContent=
b.temp_password,get("easy-login-exp").textContent=b.expires_at||"",get("easy-login-result").classList.
remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const m=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(m&&m.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const Mi=4,Ai=5*1024*1024;function si(d,m,g){let b=[];const k=o(()=>{g&&(g.innerHTML="",
b.forEach((_,T)=>{const M=document.createElement("div");M.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const $=document.createElement("img");$.src=_.url,$.alt=_.file.name,
$.className="w-full h-full object-cover";const O=document.createElement("button");O.type="button",O.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",O.textContent=
"\xD7",O.onclick=()=>{URL.revokeObjectURL(_.url),b.splice(T,1),k()},M.appendChild($),M.appendChild(O),
g.appendChild(M)}))},"render");return d&&m&&(d.onclick=()=>m.click(),m.onchange=_=>{for(const T of Array.
from(_.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(T.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(T.
size>Ai){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(b.
length>=Mi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}b.push({file:T,url:URL.createObjectURL(T)})}_.target.value="",k()}),{files:o(()=>b.
map(_=>_.file),"files"),clear:o(()=>{b.forEach(_=>URL.revokeObjectURL(_.url)),b=[],k()},"clear")}}o(
si,"createFbImagePicker");async function ai(d,m){try{const g=new FormData;return m.forEach(b=>g.append(
"images",b,b.name)),(await apiFetch(d,{method:"POST",body:g})).ok}catch{return!1}}o(ai,"uploadFbImag\
es");const oi=si(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-message").value.trim();if(!m){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:m,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=get("fb-attach-logs"),k=b&&b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;k&&(g.client_logs=k);const _=get("fb-attach-chat"),T=!!(_&&_.checked);if(T&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}T&&(g.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const M=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),$=M.ok?await M.clone().json().
catch(()=>({})):{},O=oi.files(),B={count:O.length,saved:!0};M.ok&&B.count&&(B.saved=await ai(`/api/f\
eedback/${$.public_id}/images`,O)),(window.ActivityLog?await window.ActivityLog.reportFeedback(M,k,T,
B):M.ok)&&(get("fb-title").value="",get("fb-message").value="",oi.clear(),_&&(_.checked=!1),b&&(b.checked=
!0),gn())};async function ri(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),gn()}o(ri,"deleteFeedback");const Ei="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Hn=[];function li(d,m){const g=d.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(b=>{const k=b.role==="admin"?m&&b.author?`\u7BA1\u7406\u8005 (${b.
author})`:"\u7BA1\u7406\u8005":m?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",_=m?[b.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${b.
log_file}`,b.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${b.image_count}\u679A\uFF09: ${b.image_dir}`]:
[b.image_count&&`\u6DFB\u4ED8\u753B\u50CF: ${b.image_count}\u679A`];return`<div class="text-[11px]">\
<div class="text-gray-400">${escapeHtml(k)} / ${escapeHtml(b.created_at||"")}</div><div class="text-\
sm whitespace-pre-wrap ${b.role==="admin"?"text-green-300":"text-gray-200"}">${escapeHtml(b.message)}\
</div>${_.filter(Boolean).map(T=>`<div class="${m?"text-amber-300":"text-gray-400"}">${escapeHtml(T)}\
</div>`).join("")}</div>`}).join("")}</div>`:""}o(li,"feedbackRepliesHtml");function ci(){return`<di\
v class="fb-reply-form mt-2 space-y-2">
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
                            <span class="block text-xs font-bold text-white">${Ei}</span>
                            <span class="fb-reply-logs-note block text-[10px] text-gray-500 mt-1"></\
span>
                        </span>
                    </label>
                    <button type="button" class="fb-reply-send bg-blue-600 hover:bg-blue-500 text-wh\
ite px-3 py-1 rounded text-xs">\u8FD4\u4FE1</button>
                </div>`}o(ci,"feedbackReplyFormHtml");function di(d,m){const g=m.querySelector(".fb-\
reply-form");if(!g)return;const b=g.querySelector(".fb-reply-logs"),k=g.querySelector(".fb-reply-log\
s-note"),_=!!(window.ActivityLog&&window.ActivityLog.canAttach());b.disabled=!_,k.textContent=_?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const T=si(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));Hn.push(T);const M=g.querySelector(".fb-reply-send");M.onclick=async()=>{const $=g.
querySelector(".fb-reply-message").value.trim();if(!$){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O={message:$,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},B=b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;B&&(O.client_logs=B);const Q=T.
files();M.disabled=!0;try{const de=await apiFetch(`/api/feedback/${d.public_id}/replies`,{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(O)}),ce=await de.json().catch(()=>({}));
if(!de.ok){showToast(ce.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const W=!Q.length||await ai(`/api/feedback/${d.public_id}/replies/${ce.reply_id}/\
images`,Q),pe=[];if(B&&ce.logs_saved===!1&&pe.push("\u30ED\u30B0"),W||pe.push("\u753B\u50CF"),pe.length)
showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${pe.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\uFF09`,"error",!0);else{const q=["\u8FD4\u4FE1"];B&&q.push(`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof ce.
logs_count=="number"?ce.logs_count:B.entries.length}\u4EF6\uFF09`),Q.length&&q.push(`\u753B\u50CF\uFF08${Q.
length}\u679A\uFF09`),showToast(q.join(q.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F",
"success")}gn()}finally{M.disabled=!1}}}o(di,"bindFeedbackReply");function ui(){const d=get("fb-admi\
n-status-filter").value;let m=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(
g=>{const b=d==="all"||(d==="unresolved"?["new","in_review","replied"].includes(g.dataset.feedbackStatus):
g.dataset.feedbackStatus===d);g.hidden=!b,b&&m++}),get("fb-admin-empty").classList.toggle("hidden",m!==
0)}o(ui,"filterAdminFeedback"),get("fb-admin-status-filter").onchange=ui;async function gn(){Hn.forEach(
_=>_.clear()),Hn=[];const m=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML=
"",(m.items||[]).filter(_=>!m.is_admin||_.user_id===void 0||_.user_id===null||!0).forEach(_=>{if(m.is_admin)
return;const T=document.createElement("div");T.className="p-2 rounded border border-gray-700 bg-gray\
-800/50",T.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
_.public_id||"")}</span> / ${_.created_at}</div><div class="font-bold text-sm">${escapeHtml(_.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(_.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(_.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":_.status)}</div>${_.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${_.
image_count}\u679A</div>`:""}${li(_,!1)}${ci()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
T.querySelector(".fb-delete").onclick=()=>ri(_.public_id),di(_,T),g.appendChild(T)});const b=get("fb\
-admin-panel"),k=get("fb-admin-list");m.is_admin?(b.classList.remove("hidden"),k.innerHTML="",(m.items||
[]).forEach(_=>{const T=document.createElement("div");T.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",T.dataset.feedbackStatus=_.status||"new",T.innerHTML=`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(_.public_id||"")}</span> / ${_.agent_name?`agent:${escapeHtml(_.agent_name)}`:`us\
er:${_.user_id}`} / ${_.created_at}${_.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(_.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(_.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(_.message)}</div>
                            ${[_.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${_.log_file}`,_.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${_.chat_file}`,_.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${_.
image_count}\u679A\uFF09: ${_.image_dir}`].filter(Boolean).map(M=>`<div class="text-[11px] text-ambe\
r-300">${escapeHtml(M)}</div>`).join("")}
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
                            ${li(_,!0)}
                            ${ci()}
                        `,T.querySelector(".fb-status").value=_.status||"new",T.querySelector(".fb-s\
ave").onclick=async()=>{const M=T.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${_.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:M})}),
gn()},T.querySelector(".fb-delete").onclick=()=>ri(_.public_id),di(_,T),k.appendChild(T)}),ui()):b.classList.
add("hidden")}if(o(gn,"loadFeedback"),window.setupTOTP=async()=>{const m=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=m.qr_image,get("totp-secret-disp").innerText=
m.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const d=get(
"totp-verify-code").value;if(!d)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),m=get("webauthn-nam\
e"),g=m?String(m.value||"").trim():"";try{d&&(d.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),k=await b.json();if(!b.ok){showToast(k.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const T=await(await ensureWebAuthnJson()).create({publicKey:k}),M=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},T,{name:g}))}),$=await M.json().catch(()=>({}));M.ok?(m&&(m.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast($.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(b){showToast(`WebAuthn Error: ${b}`,"error",!0)}finally{d&&(d.disabled=!1)}},window.
removeWebAuthnCredential=async d=>{if(!d||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const m=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:d})}),g=await m.json().catch(()=>({}));if(m.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(g.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let d;try{d=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(d.ok){location.href="/";
return}let m={};try{m=await d.json()}catch{}if(m&&m.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(m.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=d=>{if(d.isComposing)return;const m=get("prompt-input");
if(slashSuggestionsVisible){const g=get("slash-command-suggestions");if(d.key==="ArrowDown"){d.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(m.value),m.value));
return}if(d.key==="ArrowUp"){d.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(m.value),m.value));
return}if(d.key==="Enter"){d.preventDefault();const b=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(m.value),m.value));b[slashSelectedIndex]?selectSlashCommand(b[slashSelectedIndex].
id):b.length>0&&selectSlashCommand(b[0].id);return}if(d.key==="Escape"){d.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const g=m.value.trim();if(d.key==="ArrowDown"){d.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(g.substring(1));return}if(d.key==="ArrowUp"){
d.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(g.substring(1));
return}if(d.key==="Enter"){d.preventDefault();const b=g.substring(1).toLowerCase(),k=loadedGems.filter(
_=>_.name.toLowerCase().includes(b)||_.description&&_.description.toLowerCase().includes(b));k[gemSelectedIndex]?
selectGemSuggestion(k[gemSelectedIndex]):k.length>0&&selectGemSuggestion(k[0]);return}if(d.key==="Es\
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
"sts-cancel-btn").onclick=()=>{isStsModel()&&$i()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async d=>{const m=(d.clipboardData||window.clipboardData).items,g=[];for(let b=0;b<m.length;b++)
if(m[b].kind==="file"){const k=m[b].getAsFile();k&&g.push(k)}g.length>0&&(d.preventDefault(),await handleFiles(
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
"warning",!0)}catch(d){const m=d&&d.message?d.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(m,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const d=get("rich-paste-capture");
d.addEventListener("paste",async m=>{const g=m.clipboardData||window.clipboardData;if(g){m.preventDefault();
try{await ingestRichPasteClipboardData(g)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),d.addEventListener("input",()=>{d.value=""})}get("chat-container").addEventListener(
"click",d=>{const m=d.target.closest("img.chat-image"),g=m?m.dataset.viewerSrc||m.currentSrc||m.src:
"";m&&g&&(d.preventDefault(),openImageViewer(g))});const hn=document.querySelector(".viewer-content");
hn&&(hn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),hn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),hn.addEventListener("touchend",onViewerTouchEnd),hn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",d=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(d.target.id==="image-viewer"||d.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",d=>{d.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",d=>{d.key==="Escape"&&closeImageViewer()});
let rt,Re=null,Tn=[],qn=!1,Mn=null,Yt=null,Dt=null,bn=null,Gn=0,An=!1,yn=null,Qt=null,It=null,vn=null,
En=null,Ht=null,$n=null,In=null;function mi(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
$n)&&$n.length)return $n;d.innerHTML="";const m=[];for(let g=0;g<24;g++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",m.push(b),d.appendChild(b)}return $n=m,m}o(mi,"ensureMicWaveformBars");function qt(d,m="hidden"){
const g=get("mic-recording-indicator"),b=get("mic-recording-text");if(g){if(In&&(clearTimeout(In),In=
null),m==="hidden"){g.classList.add("hidden");return}b&&d&&(b.innerText=d),g.classList.remove("hidde\
n"),m==="recording"?g.style.color="rgb(252 165 165)":m==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(qt,"setMicRecordingIndicator");function pi(){mi().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}o(pi,"resetMicWaveformBars");function Gt(){if(En&&(cancelAnimationFrame(
En),En=null),vn){try{vn.disconnect()}catch{}vn=null}if(Qt){try{Qt.close()}catch{}Qt=null}It=null,Ht=
null,pi()}o(Gt,"stopMicWaveform");function fi(d){Gt();const m=mi();if(!m.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{Qt=new g,It=Qt.createAnalyser(),It.fftSize=
256,It.smoothingTimeConstant=0,vn=Qt.createMediaStreamSource(d),vn.connect(It),Ht=new Uint8Array(It.
frequencyBinCount)}catch{Gt();return}const b=o(()=>{if(!It||!Ht)return;It.getByteFrequencyData(Ht);const k=Math.
max(1,Math.floor(Ht.length/m.length));for(let _=0;_<m.length;_++){const M=(Ht[Math.min(Ht.length-1,_*
k)]||0)/255,$=Math.max(2,Math.round(2+M*10));m[_].style.height=`${$}px`,m[_].style.opacity=`${.35+M*
.65}`}En=requestAnimationFrame(b)},"render");b()}o(fi,"startMicWaveform");function Pn(){if(Mn&&(clearInterval(
Mn),Mn=null),bn){try{bn.disconnect()}catch{}bn=null}if(Yt){try{Yt.close()}catch{}Yt=null}Dt=null}o(Pn,
"stopSilenceMonitor");function gi(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;Pn();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;Yt=new m,Dt=Yt.createAnalyser(),Dt.fftSize=2048,
bn=Yt.createMediaStreamSource(d),bn.connect(Dt);const g=new Uint8Array(Dt.fftSize),b=getStsSilenceMs(),
k=.02;Gn=0,An=!1,Mn=setInterval(()=>{if(!Dt)return;Dt.getByteTimeDomainData(g);let _=0;for(let M=0;M<
g.length;M++){const $=(g[M]-128)/128;_+=$*$}if(Math.sqrt(_/g.length)>k){An||(An=!0),Gn=Date.now();return}
An&&Date.now()-Gn>b&&rt&&rt.state==="recording"&&rt.stop()},200)}o(gi,"startSilenceMonitor");const Wn=class Wn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(m,g,b,k={},_=null){if(this.
model=b,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(m)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const $=!!(k&&
k.transcriptionConfig),O={setup:{model:`models/${b}`,generationConfig:{responseModalities:$?["TEXT"]:
["AUDIO"]},inputAudioTranscription:$?k.transcriptionConfig||{}:{}}};$||(O.setup.outputAudioTranscription=
{}),k.speechConfig&&(O.setup.generationConfig.speechConfig=k.speechConfig),k.thinkingConfig&&(O.setup.
generationConfig.thinkingConfig=k.thinkingConfig),k.translationConfig&&(O.setup.generationConfig.translationConfig=
k.translationConfig),console.log("Sending setup:",JSON.stringify(O)),this.ws.send(JSON.stringify(O))},
this.ws.onmessage=$=>this._handleMessage($),this.ws.onerror=$=>{console.error("Gemini Live WebSocket\
 error:",$),this.onError&&this.onError($)},this.ws.onclose=$=>{console.log("Gemini Live WebSocket cl\
osed:",$.code,$.reason),this.closedEvent=$,this.onClose&&this.onClose($)},_&&_.stream&&_.ctx&&_.source){
this.stream=_.stream,this.audioContext=_.ctx;var T=_.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(On());const $=hi(this.stream,16e3);this.audioContext=$.ctx;var T=$.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const M=new MediaRecorder(this.stream);M.ondataavailable=$=>{$.data.
size>0&&this.userAudioChunks.push($.data)},M.start(500),this.backupRecorder=M,this.processor.onaudioprocess=
$=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const O=bi($.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!O||!O.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(O))),mimeType:"audio/\
pcm;rate=16000"}}}))},T.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
let g=null;try{g=JSON.parse(typeof m.data=="string"?m.data:new TextDecoder().decode(m.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const b=g.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(k=>{if(k.text&&(k.thought?(console.log(
"Gemini thought delta:",k.text),this.assistantThought+=k.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(k.text)),k.inlineData&&k.inlineData.data){const _=k.inlineData.data;
console.log("Gemini audio chunk received, size:",_.length),this.rtPlayer&&this.rtPlayer.addChunk(_);
const T=atob(_),M=new Uint8Array(T.length);for(let $=0;$<T.length;$++)M[$]=T.charCodeAt($);this.assistantAudioChunks.
push(M)}}),b.outputTranscription&&b.outputTranscription.text&&this._appendAssistantText(b.outputTranscription.
text),b.inputTranscription&&b.inputTranscription.text&&(this.userTurnBreak&&this.inputTranscript&&!this.
inputTranscript.endsWith(`
`)&&(this.inputTranscript+=`
`),this.userTurnBreak=!1,this.inputTranscript+=b.inputTranscription.text,this.interimInputTranscript=
""),b.interimInputTranscription&&(this.interimInputTranscript=b.interimInputTranscription.text||""),
b.turnComplete&&(this.assistantTurnBreak=!0,this.model!=="gemini-3.5-transcribe-live"&&(this.userTurnBreak=
!0))}g.error&&this.onError&&this.onError(g.error),this.onMessage&&this.onMessage(g)}_appendAssistantText(m){
m&&(this.assistantTurnBreak&&this.assistantText&&!this.assistantText.endsWith(`
`)&&(this.assistantText+=`
`),this.assistantTurnBreak=!1,this.assistantText+=m)}stop(){const m=this.backupRecorder;if(this.backupRecorder=
null,this._stopped=new Promise(b=>{let k=!1;const _=o(()=>{k||(k=!0,b())},"finish");if(!m||m.state===
"inactive"){_();return}m.addEventListener("stop",_,{once:!0});try{m.stop()}catch{_()}setTimeout(_,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=o(
()=>{if(this.stream&&this.stream.getTracks().forEach(b=>{try{b.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const m=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(m),b=new Blob(
this.userAudioChunks),k=await this._blobToBase64(b);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:k}}_blobToBase64(m){
return new Promise(g=>{const b=new FileReader;b.onloadend=()=>g(b.result.split(",")[1]),b.readAsDataURL(
m)})}};o(Wn,"GeminiLiveClient");let Un=Wn;const Vn=class Vn{constructor(m=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:m}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(m){if(!this.ctx)return;const g=atob(m),b=new Uint8Array(g.length);for(let O=0;O<
g.length;O++)b[O]=g.charCodeAt(O);const k=new Int16Array(b.buffer),_=new Float32Array(k.length);for(let O=0;O<
k.length;O++)_[O]=k[O]/32768;const T=this.ctx.createBuffer(1,_.length,this.ctx.sampleRate);T.getChannelData(
0).set(_),this.ctx.state==="suspended"&&await this.ctx.resume();const M=this.ctx.createBufferSource();
M.buffer=T,M.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const $=Math.max(this.ctx.currentTime,this.nextStartTime);M.start($),this.
nextStartTime=$+T.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Vn,"RealTimeAudioPl\
ayer");let wn=Vn;const Jn=class Jn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=null,this.rtPlayer=
null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript="",this.assistantThought=
"",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.saved=
!1,this.saving=!1,this.stopping=!1,this.audioQueue=[],this.audioFlush=null}isActive(){return this.active}async start(){
if(this.active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const m=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const k=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=k.id!==null&&k.id!==void 0?String(k.id):k.id,setTemporaryChatUiState(!!(k&&k.
is_temporary)),setCurrentChatHeaderTitle(k&&k.title),applyTemporaryChatRuntimeMeta(k||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+k.id),get("welcome-screen").classList.add("hidden")}catch(b){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+b.message,"\
error",!0);return}const g={model:m,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||m==="gpt-realtime-translate")&&get("sts-targ\
et-lang")?get("sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(g.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const b=await apiFetch(
"/api/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
g)}),k=await b.json().catch(()=>({}));if(!b.ok)throw new Error(k.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=k.session_id,this.rateIn=k.rate_in||this.rateIn,this.rateOut=k.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null,this.audioQueue=[],this.audioFlush=null}catch(b){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",
!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
b.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(b){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
b.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const m="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),g=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(m,g).then(b=>{if(!b.ok)throw new Error("SSE stream failed ("+b.status+")");this.
reader=b.body.getReader(),this._readLoop()}).catch(b=>{b&&b.name==="AbortError"||(this.streamError=b&&
b.message?b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const m=new TextDecoder;let g="";try{for(;this.reader;){const{done:b,
value:k}=await this.reader.read();if(b)break;g+=m.decode(k,{stream:!0});let _;for(;(_=g.indexOf(`

`))>=0;){const T=g.slice(0,_);g=g.slice(_+2);for(const M of T.split(`
`)){if(!M.startsWith("data: "))continue;let $=null;try{$=JSON.parse(M.slice(6))}catch{continue}this.
_handleEvent($)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new wn(this.rateOut||24e3),Zt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(m.data));break;case"transcript":m.role==="user"?(m.cumulative?this.userTranscript=
m.delta:this.userTranscript+=m.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
m.role==="assistant"?(this.assistantTranscript+=m.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):m.role==="thought"&&(this.assistantThought+=m.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":m.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"notice":m.message&&showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0: "+m.message,
"warning",!0);break;case"error":this.streamError=m.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Zt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(On()).then(m=>{this.stream=m;const g=this.rateIn||24e3,b=hi(m,g);this.audioCtx=
b.ctx;const k=b.source,_=this.audioCtx.sampleRate,T=4096;this.processor=this.audioCtx.createScriptProcessor(
T,1,1),this.processor.onaudioprocess=M=>{if(!this.active||!this.capturing)return;const $=M.inputBuffer.
getChannelData(0),O=bi($,_,g);!O||!O.byteLength||this._sendAudio(O)},k.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(m){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(m)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const m=this.audioQueue.splice(0,this.audioQueue.length),
g=m.reduce(($,O)=>$+O.byteLength,0),b=new Uint8Array(g);let k=0;m.forEach($=>{b.set($,k),k+=$.byteLength});
const _="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),T={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
M=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(T):T;try{await fetch(_,M)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}Pn(),Gt()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const m=Date.now(),g=this.responseDoneCount;let b=this.lastAudioAt;for(;Date.now()-m<2e4&&!(this.responseDoneCount>
g||(this.lastAudioAt>b&&(b=this.lastAudioAt),!this.speechActive&&Date.now()-m>2e3&&Date.now()-b>2500));)
await new Promise(k=>setTimeout(k,250));await this._save()}async _save(){if(!this.saved){this.saved=
!0,this.saving=!0;try{const m=await apiFetch("/api/realtime/save",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify({session_id:this.sessionId,thread_id:currentThreadId})}),
g=await m.json().catch(()=>({}));if(!m.ok)throw new Error(g.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(g.thread_id&&!currentThreadId&&(currentThreadId=String(g.thread_id)),this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(m){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(m&&m.message?m.message:m),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this.audioQueue=[],this._stopCapture(),this._stopPlayback(),this.abortCtrl){
try{this.abortCtrl.abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const m=get(
"mic-btn");m&&(m.classList.remove("bg-red-600","animate-pulse"),m.classList.add("bg-gray-700"))}};o(
Jn,"RealtimeVoiceSession");let zn=Jn;function hi(d,m){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let b=null;try{return b=new g({sampleRate:m}),{ctx:b,
source:b.createMediaStreamSource(d)}}catch{if(b)try{b.close()}catch{}return b=new g,{ctx:b,source:b.
createMediaStreamSource(d)}}}o(hi,"openMicAudioSource");function bi(d,m,g){let b=d;if(m!==g&&m>0&&g>
0){const _=m/g,T=Math.floor(b.length/_),M=new Float32Array(T);for(let $=0;$<T;$++)M[$]=b[Math.min(Math.
floor($*_),b.length-1)];b=M}const k=new Int16Array(b.length);for(let _=0;_<b.length;_++){const T=Math.
max(-1,Math.min(1,b[_]));k[_]=T<0?T*32768:T*32767}return k.buffer}o(bi,"pcm16FromFloat32");const xn=new zn,
yi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,k=null,_=!1,T=null,M=!1,$=0,O=0,B=null,Q="idle",de=!1,ce=null;const W=o(F=>document.getElementById(
F),"$"),pe=o(F=>{const te=Object.assign({},F||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(te):(te.progressSpinner=
!1,te)},"noSpinner");function q(F,te){Q=te;const Ee=W("lyria-status-text"),ge=W("lyria-status-dot");
Ee&&(Ee.textContent=F),ge&&(ge.className="w-2 h-2 rounded-full inline-block "+(g[te]||g.idle)),ke(),
ae()}o(q,"setStatus");function ye(){const F=O?Math.floor((Date.now()-O)/1e3):0,te=String(Math.floor(
F/60)).padStart(2,"0"),Ee=String(F%60).padStart(2,"0");return`${te}:${Ee}`}o(ye,"formatElapsed");function V(){
O||(O=Date.now());const F=W("lyria-elapsed");F&&(F.textContent=ye()),B||(B=window.setInterval(()=>{const te=W(
"lyria-elapsed");te&&(te.textContent=ye())},1e3))}o(V,"startElapsedTimer");function he(){B&&(window.
clearInterval(B),B=null)}o(he,"stopElapsedTimer");function ke(){const F=W("lyria-play-btn"),te=W("ly\
ria-pause-btn"),Ee=W("lyria-stop-btn"),ge=W("lyria-reset-btn"),Ie=!!b,He=Q==="streaming"||Q==="conne\
cting";if(F){F.disabled=de;const nt=F.querySelector("i");nt&&(nt.className="fas fa-play")}te&&(te.disabled=
de||!He),Ee&&(Ee.disabled=de||!Ie||!He),ge&&(ge.disabled=de||!Ie||!He)}o(ke,"updateTransportButtons");
function ae(){const F=W("lyria-save-btn");if(!F)return;const te=!!b&&Q!=="idle"&&Q!=="connecting"&&Q!==
"error";F.classList.toggle("hidden",!te)}o(ae,"updateSaveButton");function ve(F,te){const Ee=W("lyri\
a-prompt-rows");if(!Ee)return;const ge=document.createElement("div");ge.className="flex items-center\
 gap-2",ge.innerHTML=`
                        <input type="text" value="${escapeHtml(F||"")}" placeholder="\u4F8B: minimal tech\
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
                    `;const Ie=ge.querySelector('input[type="range"]'),He=ge.querySelector(".lyria-w\
eight-label");Ie&&He&&Ie.addEventListener("input",()=>{He.textContent=parseFloat(Ie.value).toFixed(1)});
const nt=ge.querySelector(".lyria-prompt-remove");nt&&nt.addEventListener("click",()=>{Ee.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ge.remove()}),ge.classList.add("lyria-prompt-row-wrap"),Ee.appendChild(
ge)}o(ve,"addPromptRow");function Pe(){const F=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),te=[];return F.forEach(Ee=>{const ge=Ee.querySelector('input[type="text"]'),Ie=Ee.
querySelector('input[type="range"]'),He=(ge?ge.value:"").trim();He&&te.push({text:He,weight:parseFloat(
Ie?Ie.value:1)||1})}),te}o(Pe,"collectPrompts");function Oe(){const F={},te=o(Ni=>{const Xn=W(Ni);return Xn&&
Xn.value!==""?parseFloat(Xn.value):void 0},"num"),Ee=te("lyria-bpm");Ee!==void 0&&(F.bpm=Math.round(
Ee));const ge=te("lyria-guidance");ge!==void 0&&(F.guidance=ge);const Ie=te("lyria-density");Ie!==void 0&&
(F.density=Ie);const He=te("lyria-brightness");He!==void 0&&(F.brightness=He);const nt=te("lyria-tem\
perature");nt!==void 0&&(F.temperature=nt);const Ve=W("lyria-scale");Ve&&Ve.value&&(F.scale=Ve.value);
const st=W("lyria-mode");st&&st.value&&(F.music_generation_mode=st.value);const bt=W("lyria-mute-bas\
s"),Ut=W("lyria-mute-drums"),Si=W("lyria-only-bass-drums");return bt&&(F.mute_bass=bt.checked),Ut&&(F.
mute_drums=Ut.checked),Si&&(F.only_bass_and_drums=Si.checked),F}o(Oe,"collectConfig");function lt(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([te,Ee])=>{const ge=W(te),Ie=W(Ee);!ge||!Ie||ge.addEventListener("input",()=>{const He=parseFloat(
ge.value);Ie.textContent=te==="lyria-bpm"?String(Math.round(He)):He.toFixed(1)})})}o(lt,"bindRangeLa\
bels");function Xe(){if(T){try{T.close()}catch{}T=null}M=!1,$=0}o(Xe,"resetPlayback");function ht(){
if(_=!1,k&&typeof k.abort=="function")try{k.abort()}catch{}k=null}o(ht,"closeStream");async function Pt(){
ht(),k=new AbortController,_=!0;try{const F=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(
b)}`,pe({method:"GET",signal:k.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!F.
ok){const Ie=await F.json().catch(()=>({}));throw new Error(Ie.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const te=F.body.getReader(),Ee=new TextDecoder;let ge="";for(;_;){const{done:Ie,value:He}=await te.read();
if(Ie)break;ge+=Ee.decode(He,{stream:!0});const nt=ge.split(`

`);ge=nt.pop();for(const Ve of nt){const st=Ve.split(`
`).find(Ut=>Ut.startsWith("data: "));if(!st)continue;const bt=st.slice(6);try{const Ut=JSON.parse(bt);
Ae(Ut)}catch{}}}}catch(F){if(F&&F.name==="AbortError")return;_&&(q("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{_&&b&&Pt()},1200))}finally{_=!1}}o(Pt,"openStream");function Ae(F){
if(F&&F.snapshot){const te=F.status;if(te==="error"){q("\u30A8\u30E9\u30FC","error"),he();return}if(te===
"closed"||te==="stopped"){q("\u7D42\u4E86","closed"),he();return}q(te==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",te==="paused"?"paused":"connecting");return}if(F&&F.audio){q("\u518D\u751F\u4E2D...",
"streaming"),V(),_e(F.audio);return}if(F&&F.error){q("\u30A8\u30E9\u30FC: "+F.error,"error"),he();return}
if(F&&F.final){q("\u7D42\u4E86","closed"),he(),ke();return}}o(Ae,"handleStreamMessage");function _e(F){
if(!F)return;if(!T){const Ve=window.AudioContext||window.webkitAudioContext;if(!Ve)return;T=new Ve({
sampleRate:48e3}),M=!1,$=0}let te;try{const Ve=atob(F);te=new Uint8Array(Ve.length);for(let st=0;st<
Ve.length;st++)te[st]=Ve.charCodeAt(st)}catch{return}const Ee=new Int16Array(te.buffer),ge=Math.floor(
Ee.length/2);if(ge<1)return;const Ie=T.createBuffer(2,ge,48e3);for(let Ve=0;Ve<2;Ve++){const st=Ie.getChannelData(
Ve);for(let bt=0;bt<ge;bt++)st[bt]=Ee[bt*2+Ve]/32768}T.state==="suspended"&&T.resume();const He=T.createBufferSource();
He.buffer=Ie,He.connect(T.destination),M||($=T.currentTime+.08,M=!0);const nt=Math.max(T.currentTime,
$);He.start(nt),$=nt+Ie.duration}o(_e,"playChunk");async function De(F,te){const Ee=await fetch("/ap\
i/gemini/music/command",pe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:F},te||{}))})),ge=await Ee.json().catch(()=>({}));if(!Ee.ok)throw new Error(
ge.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ge}
o(De,"apiCommand");async function tt(){if(de)return;const F=Pe();if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}de=!0,ke(),q("\u63A5\u7D9A\u4E2D...","connecting");try{
const te=await fetch("/api/gemini/music/start",pe({method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({weighted_prompts:F,config:Oe()})})),Ee=await te.json().catch(()=>({}));
if(!te.ok)throw new Error(Ee.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=Ee.session_id,ce=Oe(),q("\u63A5\u7D9A\u4E2D...","connecting"),Pt()}catch(te){q("\u30A8\u30E9\u30FC: "+
te.message,"error"),showToast("Lyria RealTime: "+te.message,"error",!0)}finally{de=!1,ke()}}o(tt,"st\
artSession");async function ct(F){if(b){de=!0,ke();try{await De("control",{action:F}),F==="PLAY"?q("\
\u518D\u751F\u4E2D...","streaming"):F==="PAUSE"?q("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):F==="ST\
OP"?q("\u505C\u6B62\u4E2D","stopped"):F==="RESET_CONTEXT"&&q("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0),q("\u30A8\u30E9\u30FC: "+
te.message,"error")}finally{de=!1,ke()}}}o(ct,"control");async function Se(){if(!b)return;const F=Pe();
if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}de=!0;try{await De("prompts",{weighted_prompts:F}),q("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
Q==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(te){showToast("Lyria RealTime: "+te.message,"error",!0)}finally{de=!1,ke()}}o(Se,"a\
pplyPrompts");async function Ne(){if(!b)return;const F=Oe(),te=ce||{},Ee=F.bpm!==void 0&&F.bpm!==te.
bpm,ge=F.scale!==void 0&&F.scale!==te.scale,Ie=Ee||ge;de=!0;try{await De("config",{config:F,reset_context:Ie}),
ce=F,q(Ie?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",Q==="paused"?"paused":"streaming"),showToast(
Ie?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(He){showToast("Lyria RealT\
ime: "+He.message,"error",!0)}finally{de=!1,ke()}}o(Ne,"applyConfig");async function Ue(){if(b){de=!0,
q("\u4FDD\u5B58\u4E2D...","connecting"),ke();try{const F=await fetch("/api/gemini/music/save",pe({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),te=await F.json().catch(()=>({}));if(!F.ok)throw new Error(te.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
q("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),he(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),te.thread_id&&(currentThreadId=String(te.thread_id),history.pushState({},"","/c/"+te.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(te.thread_id||currentThreadId),_n(
!0)}catch(F){q("\u30A8\u30E9\u30FC: "+F.message,"error"),showToast("Lyria RealTime: "+F.message,"err\
or",!0)}finally{de=!1,ke()}}}o(Ue,"saveSession");async function Ye(){if(ht(),b)try{await fetch("/api\
/gemini/music/cancel",pe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:b})}))}catch{}b=null,he(),Xe(),q("\u6E96\u5099\u5B8C\u4E86","idle")}o(Ye,"cancelSession");
function mt(){const F=W("lyria-prompt-rows");F&&(F.innerHTML=""),ve("",1),ce=null,O=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ge=>{const Ie=W(ge);
Ie&&(Ie.value=ge==="lyria-bpm"?"120":ge==="lyria-guidance"?"4":ge==="lyria-temperature"?"1.1":"0.5")});
const te=W("lyria-scale");te&&(te.value="");const Ee=W("lyria-mode");Ee&&(Ee.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ge=>{const Ie=W(ge);Ie&&(Ie.checked=
!1)}),lt()}o(mt,"resetControls");function _n(F){ht(),b&&fetch("/api/gemini/music/cancel",pe({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),
b=null,_=!1,he(),Xe(),hideModal("lyria-studio-modal")}o(_n,"closeAndCleanup");function en(F){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const te=W("lyria-studio-modal");if(te&&te.classList.contains("modal-open")&&b){
if(F&&typeof F=="string"){const ge=W("lyria-prompt-rows");ge&&(ge.innerHTML=""),ve(F,1)}return}if(b&&
Ye(),mt(),F&&typeof F=="string"){const ge=W("lyria-prompt-rows");ge&&(ge.innerHTML=""),ve(F,1)}b=null,
_=!1,he(),Xe(),q("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(en,"open");function Sn(){
const F=W("lyria-open-studio-btn");F&&F.addEventListener("click",()=>en(""));const te=W("lyria-studi\
o-close");te&&te.addEventListener("click",()=>_n(!1));const Ee=W("lyria-play-btn");Ee&&Ee.addEventListener(
"click",()=>{if(!b){tt();return}ct("PLAY")});const ge=W("lyria-pause-btn");ge&&ge.addEventListener("\
click",()=>ct("PAUSE"));const Ie=W("lyria-stop-btn");Ie&&Ie.addEventListener("click",()=>ct("STOP"));
const He=W("lyria-reset-btn");He&&He.addEventListener("click",()=>ct("RESET_CONTEXT"));const nt=W("l\
yria-add-prompt-btn");nt&&nt.addEventListener("click",()=>ve("",1));const Ve=W("lyria-apply-prompts-\
btn");Ve&&Ve.addEventListener("click",Se);const st=W("lyria-apply-config-btn");st&&st.addEventListener(
"click",Ne);const bt=W("lyria-save-btn");bt&&bt.addEventListener("click",Ue),lt(),mt(),window.openLyriaStudio=
en}o(Sn,"init");function Kn(){de||_n(!1)}return o(Kn,"requestClose"),{init:Sn,open:en,requestClose:Kn}})();
yi.init(),window.closeLyriaStudio=()=>yi.requestClose(),(()=>{let d=null,m=null,g=null;const b="voic\
eDockSettingsOpen",k="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
_=o(ae=>document.getElementById(ae),"$");function T(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(T,"isStudioMode");function M(){const ae=get("model-select")?get("model-select").value:"",ve=_("voi\
ce-studio-title");ve&&(ae==="gpt-transcribe"||ae==="gpt-live-transcribe"||ae==="grok-voice-transcrib\
e-2.0-file"?ve.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ae===
"gemini-3.5-live-translate-preview"?ve.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ve.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(M,"updateTitle");function $(){const ae=_("v\
oice-studio-transcript");ae&&(ae.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${k}</div>`);const ve=_("sts-live-transcript");ve&&(ve.innerHTML="",ve.classList.add("hidden"))}
o($,"resetTranscript");function O(ae,ve,Pe){const Oe=ae.querySelectorAll(".voice-studio-line");let lt=null;
for(let Xe=Oe.length-1;Xe>=0;Xe--)if(Oe[Xe].dataset.role===ve){lt=Oe[Xe];break}if(lt)lt.innerHTML=Pe;else{
const Xe=ae.querySelector(".voice-studio-placeholder");Xe&&Xe.remove();const ht=document.createElement(
"div");ht.className="voice-studio-line",ht.dataset.role=ve,ht.innerHTML=Pe,ae.appendChild(ht)}ae.classList.
remove("hidden"),ae.scrollTop=ae.scrollHeight}o(O,"writeLine");function B(ae,ve){if(!ve||!String(ve).
trim()||!T())return;const lt=`<span class="${ae==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ae==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
ve)}</span>`;[_("voice-studio-transcript"),_("sts-live-transcript")].filter(Boolean).forEach(Xe=>O(Xe,
ae,lt))}o(B,"log");function Q(ae,ve=!0){const Pe=_("sts-panel"),Oe=_("sts-settings-toggle");if(Pe&&Pe.
classList.toggle("settings-open",!!ae),Oe&&Oe.setAttribute("aria-expanded",ae?"true":"false"),ve)try{
localStorage.setItem(b,ae?"1":"0")}catch{}}o(Q,"setSettingsOpen");function de(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}o(de,"readSettingsOpen");let ce=null;function W(){const ae=get("mod\
el-select")?get("model-select").value:"";ae!==ce&&(ce=ae,$()),Q(de(),!1),M()}o(W,"syncDock");function pe(){
const ae=_("sts-panel"),ve=_("voice-studio-panel-host");ae&&ve&&ae.parentNode!==ve&&(d=ae.parentNode,
m=ae.nextSibling,ve.appendChild(ae));const Pe=_("file-preview"),Oe=_("voice-studio-file-host");Pe&&Oe&&
Pe.parentNode!==Oe&&(g=Pe.parentNode,Oe.appendChild(Pe),Oe.classList.remove("hidden"))}o(pe,"movePan\
elIntoModal");function q(){const ae=_("sts-panel");ae&&d&&ae.parentNode!==d&&(m&&m.parentNode===d?d.
insertBefore(ae,m):d.appendChild(ae));const ve=_("file-preview");ve&&g&&ve.parentNode!==g&&g.appendChild(
ve);const Pe=_("voice-studio-file-host");Pe&&Pe.classList.add("hidden"),d=null,m=null,g=null}o(q,"mo\
vePanelBack");function ye(){if(!T())return;pe();const ae=_("sts-panel");ae&&ae.classList.remove("hid\
den"),M(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(ye,"open");function V(){window.
VoiceStudioOpen=!1,q(),hideModal("voice-studio-modal")}o(V,"close");function he(){window.VoiceStudioOpen&&
V()}o(he,"closeIfOpen");function ke(){window.VoiceStudioOpen=!1;const ae=_("voice-studio-open-btn");
ae&&ae.addEventListener("click",()=>ye());const ve=_("voice-studio-close");ve&&ve.addEventListener("\
click",()=>V());const Pe=_("sts-settings-toggle");Pe&&Pe.addEventListener("click",()=>{const Oe=_("s\
ts-panel");Q(!(Oe&&Oe.classList.contains("settings-open")))}),window.VoiceStudio={open:ye,close:V,closeIfOpen:he,
log:B,isStudioMode:T,syncDock:W},W()}return o(ke,"init"),{init:ke}})().init();let Zt=null;function vi(){
if(Zt&&(Zt.stop(),Zt=null),yn){try{yn.pause()}catch{}try{yn.src=""}catch{}yn=null}}o(vi,"stopStsPlay\
back");async function Fi(d){vi();const m=new Audio;return m.src=d,m.preload="auto",m.autoplay=!0,m.playsInline=
!0,yn=m,await m.play(),new Promise(g=>{m.onended=()=>g("ended"),m.onerror=()=>g("error")})}o(Fi,"pla\
yStsAudio");function $i(){if(xn.isActive()){xn._cancel();return}if(Re){Re.stop(),Re=null,vi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Gt();return}rt&&rt.state===
"recording"&&(qn=!0,rt.stop())}o($i,"cancelRecording");function On(){if(isStsModel())return{audio:!0};
const m=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return m.echoCancellation&&(g.echoCancellation=!1),m.
noiseSuppression&&(g.noiseSuppression=!1),m.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(On,"\
getMicCaptureConstraints");function Ii(){const d=window.AudioContext||window.webkitAudioContext;if(!d||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let m=null;try{m=new d({sampleRate:16e3})}catch{if(m){try{m.close()}catch{}m=null}m=new d}const g=m.
resume(),b=navigator.mediaDevices.getUserMedia(On());return{ctx:m,resumePromise:g,streamPromise:b}}o(
Ii,"beginGeminiLiveMic");function Pi(d){if(d){if(d.streamPromise)d.streamPromise.then(m=>{try{m.getTracks().
forEach(g=>g.stop())}catch{}}).catch(()=>{});else if(d.stream)try{d.stream.getTracks().forEach(m=>m.
stop())}catch{}if(d.ctx)try{d.ctx.close()}catch{}}}o(Pi,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Re){setStsStatus("Processing...",!0);const d=Re;Re=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const m=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(m.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
m.assistant_text=(d.inputTranscript||"").trim(),m.assistant_thought="",!m.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const g=(m.user_text||
"").trim()||(m.assistant_text||"").trim()||(m.assistant_thought||"").trim(),b=!!(m.audio_base64||m.user_audio_base64);
if(!g&&!b){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const _=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(_.id),history.pushState(
{},"","/c/"+_.id),get("welcome-screen").classList.add("hidden")}m.thread_id=currentThreadId,m.model=
get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error("Failed to save G\
emini Live session:",m),setStsStatus("Error saving",!1)}return}if(xn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),xn.stop();return}if(rt&&
rt.state==="recording"){rt.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||qt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ii();
const k={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(k.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const q=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(ye=>ye.trim()).filter(Boolean);q.length&&(k.custom_vocabulary=
q.slice(0,1e3))}}else k.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(k.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",k.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(k.target_lang=get("sts-target-lang").value);const _=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:_,ctx:g.ctx,source:g.ctx.createMediaStreamSource(_)};const T=await apiFetch(
"/api/gemini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
k)});if(!T.ok)throw new Error("Failed to get session token");const{token:M,url:$}=await T.json(),O=get(
"model-select").value,B=get("sts-voice")?get("sts-voice").value:"Kore",Q=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",de=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(Re=new Un,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Re.rtPlayer=new wn),isGeminiLiveTranscribeModel()){
const q=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",ye={languageCodes:[]};
if((q==="SMART"||q==="VERBATIM")&&(ye.mode=q),get("sts-custom-vocab")){const V=get("sts-custom-vocab").
value.split(/[,、\n]/).map(he=>he.trim()).filter(Boolean);V.length&&(ye.customVocabulary=V.slice(0,
1e3))}await Re.start(M,$,O,{transcriptionConfig:ye},g)}else if(isGeminiLiveTranslateModel()){const q=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Re.start(M,$,O,{translationConfig:{targetLanguageCode:q,
echoTargetLanguage:!0}},g)}else{const q={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:B}}}};
isGeminiLiveExtendedThinkingModel()&&(q.thinkingConfig={thinkingLevel:Q,includeThoughts:de}),await Re.
start(M,$,O,q,g)}g=null,rt=Re.backupRecorder,rt.onstop=()=>{Re&&get("mic-btn").click()};let ce=!0,W="\
live-sts-"+Date.now();Re.onMessage=q=>{if(q.serverContent){if(isGeminiLiveTranscribeModel()){const ye=Re.
interimInputTranscript,V=Re.inputTranscript,he=V+(ye&&!V.endsWith(ye)?(V?`
`:"")+ye:""),ke=get("chat-messages");let ae=document.getElementById(W);ae||(ae=document.createElement(
"div"),ae.id=W,ae.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ae.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ke.appendChild(ae),ke.scrollTop=ke.scrollHeight);const ve=ae.
querySelector(".message-content");ve.innerText=he||"\u8074\u304D\u53D6\u308A\u4E2D...",ke.scrollTop=
ke.scrollHeight,window.VoiceStudio&&V&&window.VoiceStudio.log("user",V);return}if(q.serverContent.modelTurn){
ce&&(setStsStatus("Gemini is speaking...",!1),ce=!1);const ye=get("chat-messages");let V=document.getElementById(
W);V||(V=document.createElement("div"),V.id=W,V.className="flex flex-col gap-2 mb-4 assistant-messag\
e bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",V.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ye.appendChild(V),ye.scrollTop=ye.scrollHeight);const he=V.
querySelector(".thought-container"),ke=V.querySelector(".message-content");Re.assistantThought&&(he.
classList.remove("hidden"),he.innerText=Re.assistantThought),ke.innerText=Re.assistantText,ye.scrollTop=
ye.scrollHeight,window.VoiceStudio&&(Re.inputTranscript&&window.VoiceStudio.log("user",Re.inputTranscript),
Re.assistantText&&window.VoiceStudio.log("assistant",Re.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),fi(Re.stream),gi(Re.stream);const pe=Re;pe.onError=q=>{const ye=q&&q.message?q.message:
typeof q=="string"?q:"";ye&&showToast("Gemini Live: "+ye,"error",!0)},pe.onClose=q=>{if(Re===pe){if(q&&
q.code&&q.code!==1e3){const ye=q.reason?": "+q.reason:" (code "+q.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+ye,"error",!0)}get("mic-btn").click()}},pe.closedEvent&&
pe.onClose(pe.closedEvent);return}catch(b){if(Pi(g),Re){try{Re.stop()}catch{}Re=null}showToast("Gemi\
ni Live connection failed: "+b.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await xn.start();return}isStsModel()||(pi(),qt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(On());rt=new MediaRecorder(d),Tn=[],qn=!1;const m=isStsModel();
rt.ondataavailable=g=>Tn.push(g.data),rt.onstop=async()=>{if(qn){Tn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(T=>T.stop()),Pn(),Gt(),m||(qt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),In=setTimeout(()=>qt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(Tn,{type:"audio/web\
m"}),b=new File([g],"recording.webm",{type:"audio/webm"}),k=new FormData;k.append("file",b),get("fil\
e-preview").classList.remove("hidden");const _=m;get("file-name").innerText=_?"Processing voice...":
"Transcribing...";try{if(_){if(!currentThreadId){const q=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=q.id!==null&&q.id!==void 0?String(q.id):q.id,setTemporaryChatUiState(!!(q&&q.
is_temporary)),setCurrentChatHeaderTitle(q&&q.title),applyTemporaryChatRuntimeMeta(q||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+q.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),k.append("model",get(
"model-select").value),k.append("thread_id",currentThreadId),get("sts-voice")&&k.append("sts_voice",
get("sts-voice").value||""),get("sts-reasoning-effort")&&k.append("sts_reasoning_effort",get("sts-re\
asoning-effort").value||""),get("sts-speed")&&k.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&k.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&k.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&k.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&k.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const T=await apiFetch(
"/sts",{method:"POST",body:k});if(!T.ok){const pe=await T.json().catch(()=>({}));throw new Error(pe.
error||"Speech-to-speech failed")}const M=T.body.getReader(),$=new TextDecoder;let O="",B=null,Q=null;
stsOpt("sts-auto-play")&&(Q=new wn,Zt=Q),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let de=!0,ce="",W="";for(;;){const{done:pe,value:q}=await M.read();if(pe)break;
O+=$.decode(q,{stream:!0});const ye=O.split(`
`);O=ye.pop();for(const V of ye){if(!V.trim())continue;const he=JSON.parse(V);if(he.error)throw new Error(
he.error);he.audio_delta&&Q&&(de&&(setStsStatus("Playing response...",!1),de=!1),await Q.addChunk(he.
audio_delta)),he.input_delta&&(ce+=he.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
ce)),he.transcript_delta&&(W+=he.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",W)),(he.final||he.audio_url)&&(B=he)}}window.VoiceStudio&&!ce.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),B&&(B.audio_url||B.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const T=get(
"set-mic-transcribe-mode");if(!!(T&&T.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}k.append("llm_model",get("model-select")&&get("model-select").value||"");const O=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:k})).json();if(O.transcript){const B=get("prompt-inp\
ut");B.value+=(B.value?" ":"")+O.transcript,B.style.height="auto",B.style.height=B.scrollHeight+"px"}else
showToast(O.error||"Transcription failed","error",!0)}}catch(T){showToast("Audio processing error: "+
T.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(T=>T.
stop()),Pn(),Gt(),_||qt("","hidden"),_&&setStsStatus("Tap to speak",!1)}},rt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(qt("\u9332\u97F3\u4E2D\u2026","recording"),fi(d)),gi(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Gt(),isStsModel()||qt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const kn=o((d,m)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),g=get("lib-download-btn"),b=get("lib-attach-btn"),k=get("lib-rena\
me-btn"),_=get("lib-usage-btn");if(m&&(m.disabled=d===0,kn(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,kn(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=d===0,kn(b,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),k&&(k.disabled=d!==1,kn(k,"\u540D\u524D\u5909\u66F4")),
_&&(_.disabled=d!==1,kn(_,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const T=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",T&&d>0)}};function wi(d){lib.
attachMode=!!d}o(wi,"setLibAttachMode");const xi=o((d=!1)=>{wi(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>xi(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>xi(!0)),
get("lib-search")){let d=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),d&&clearTimeout(d),d=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const d=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=d,get("lib-sort").onchange=()=>{const m=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,m),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const Oi=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>Oi(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
m=get("gem-desc").value,g=get("gem-inst").value,b=collectGemFixedPrompts();if(d&&g){const k=editingGemUuid?
"PUT":"POST",_=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
_,{method:k,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:g,
fixed_prompts:b,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=d,activeGem.instruction=
g,activeGem.fixed_prompts=b,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(d){if(d.target.closest(".edit-btn")){const g=d.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(g)}if(d.target.closest(".code-toggle")){
const m=d.target.closest(".code-toggle"),g=m.closest(".code-wrapper");if(!g)return;const b=g.classList.
toggle("collapsed");g.setAttribute("data-collapsed",b?"true":"false"),m.setAttribute("aria-expanded",
b?"false":"true"),m.innerHTML=b?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',m.title=b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",m.setAttribute("aria-label",b?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(d.target.closest(".download-btn")){const m=d.target.closest(".d\
ownload-btn"),g=m.getAttribute("data-code"),b=(m.getAttribute("data-lang")||"txt").toLowerCase();if(g)
try{const k=decodeURIComponent(g),_=new Blob([k],{type:"text/plain"}),T=URL.createObjectURL(_),M=document.
createElement("a");M.href=T;let O={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(O="txt");let B=`code.${O}`;b==="dockerfile"&&(B="Docke\
rfile"),b==="makefile"&&(B="Makefile"),M.download=B,document.body.appendChild(M),M.click(),document.
body.removeChild(M),URL.revokeObjectURL(T)}catch(k){console.error("Download failed",k)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),g=m.getAttribute("data-code");g&&
window.copyCode(m,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&ni(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
