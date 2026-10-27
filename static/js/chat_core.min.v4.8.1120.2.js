document.addEventListener("DOMContentLoaded",()=>{var yi,vi;initThemeFromServer(),applyLiquidGlassMode(
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
getBoundingClientRect(),g=d.getBoundingClientRect(),h=g.left+g.width/2-(m.left+m.width/2),x=g.top+g.
height/2-(m.top+m.height/2);r.style.transform=`translate(${h}px, ${x}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const m=get("gpt-image-format"),g=get("gpt-image-compression-wrap");m&&g&&(m.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}o(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(h=>{h&&(Array.from(h.options).
forEach(x=>{x.value!=="1K"&&(x.disabled=g)}),g&&h.value!=="1K"&&(h.value="1K"))})}o(c,"updateGeminiI\
mageUi");function u(){const d=get("grok-image-options");if(!d)return;const m=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),h=m==="grok-imagine-image-quality"||m==="grok-imagine-image-2\
.0",x=m==="grok-imagine-image-2.0";if(g){d.classList.remove("hidden");const L=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!h);const M=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;M&&M.classList.toggle("hidden",!x)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const L=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!h);const M=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;M&&M.classList.toggle(
"hidden",!x)}}o(u,"updateGrokImageUi");function f(){var E;const d=get("ideogram-image-options"),m=get(
"modal-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",h=isIdeogramModel(
g),x=ideogramModelTraits(g),_={resolution:x.resolution,quality:x.quality,speed:x.speed,style:x.style,
negative:x.negative};if(d&&d.classList.toggle("hidden",!h),m&&m.classList.toggle("hidden",!h),!h)return;
const L=((E=get("ideogram-image-operation"))==null?void 0:E.value)==="remove_background";IDEOGRAM_IMAGE_FIELDS.
forEach(O=>{const N=O==="operation"||!L&&_[O]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(
X=>{const oe=get(X+O);!oe||!oe.parentElement||oe.parentElement.classList.toggle("hidden",!N)})}),["i\
deogram-image-style","modal-ideogram-image-style"].forEach(O=>{const N=get(O);if(!N)return;const X=x.
styleOptions.join("|");if(N.dataset.options!==X){const oe=N.value;N.innerHTML=x.styleOptions.map(ie=>`\
<option value="${ie}">${ie==="auto"?"Auto":ie}</option>`).join(""),N.dataset.options=X,N.value=x.styleOptions.
includes(oe)?oe:"auto"}});const M=get("ideogram-image-note");M&&(M.textContent=L?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25\
MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3002\u900F\u904EPNG\u3067\u51FA\u529B\uFF08\u6307\u793A\u6587\u30FB\u751F\u6210\u8A2D\u5B9A\u306F\u4F7F\u3044\u307E\u305B\u3093\uFF09":
x.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(d=>{
var m;(m=get(d))==null||m.addEventListener("change",()=>{const g=get(d.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");g&&(g.value=get(d).value),f()})});function b(){
var h;const d=get("grok-video-options");if(!d)return;const m=String(((h=get("model-select"))==null?void 0:
h.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");
const g=get("grok-video-resolution");if(g){const x=Array.from(g.options).find(L=>L.value==="1080p"),
_=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(m);x&&(x.disabled=!_),!_&&g.value===
"1080p"&&(g.value="720p")}}o(b,"updateGrokVideoUi");function y(){var x;const d=get("gemini-video-opt\
ions");if(!d)return;const m=String(((x=get("model-select"))==null?void 0:x.value)||"").toLowerCase();
isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get("gemini-vide\
o-resolution");if(g){const _=Array.from(g.options).find(M=>M.value==="4K"),L=m==="veo-3.1-lite-gener\
ate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";_&&(_.disabled=L),L&&g.value===
"4K"&&(g.value="1080p")}const h=get("gemini-video-duration-wrap");h&&h.classList.toggle("hidden",m===
"gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function w(){const d=get("gemini-music-options");
if(!d)return;const m=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!m;d.classList.toggle("hid\
den",!g);const h=get("lyria-realtime-studio-bar");h&&h.classList.toggle("hidden",!m)}o(w,"updateGemi\
niMusicUi");function v(){var L;const d=get("xai-chat-options");if(!d)return;const m=String(((L=get("\
model-select"))==null?void 0:L.value)||"").toLowerCase(),g=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!g);const h=get("xai-logp\
robs"),x=get("xai-top-logprobs"),_=m.includes("grok-4.20");h&&(h.disabled=_,_&&(h.checked=!1)),x&&(x.
disabled=_,_&&(x.value=""))}o(v,"updateXaiChatUi");function k(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(h=>{const x=get(h);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function S(){const d=get("image-input-limit\
s");if(!d)return;const m=(get("model-select").value||"").toLowerCase();let g="",h=!1;m.includes("gpt\
-image")?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):m==="deepseek-v4.1-flash"||m==="deepseek-v4-flash-vision-exp"?(h=!0,g=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):m.includes("deepseek")||(isGeminiImageModelKey(m)?(h=!0,m.includes("gemini-3.1-flash-lite-\
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
join("")):isMistralOcrModel(m)?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):m.includes("grok")?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):isIdeogramModel(m)?(h=!0,g=ideogramModelTraits(m).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059 / \u30DE\u30B9\u30AF\u306B\u306F\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002\u80CC\u666F\u3092\u524A\u9664\u3067\u306F\u753B\u50CF1\u679A\u3092\u900F\u904EPNG\u306B\u3057\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.5 \u306E\u307F\u3002\u80CC\u666F\u3092\u524A\u9664\u3067\u306F\u5168\u30E2\u30C7\u30EB\u3067\u753B\u50CF1\u679A\uFF0825MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\uFF09\u306B\u5BFE\u5FDC\u3057\u307E\u3059</div>"].
join("")):m.includes("grok")&&m.includes("video")&&(h=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),h?(d.innerHTML=g,d.classList.remove("hidden")):(d.classList.add("hidden"),d.innerHTML="")}
o(S,"updateImageInputLimits");function C(){const d=get("enable-sys-prompt"),m=!!(d&&d.disabled),g=!!(d&&
d.checked);I(),d&&(d.disabled?!m&&g?d.dataset.restoreChecked="1":m||delete d.dataset.restoreChecked:
(m&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(C,"toggleOpti\
ons"),window.toggleOptions=C;function T(d){const m=get("thinking-options"),g=get("thinking-level"),h=get(
"reasoning-effort"),x=get("enable-thinking"),_=get("thinking-budget-label"),L=get("thinking-budget"),
M=d==="effort";if(m&&(m.dataset.thinkingMode=M?"effort":"thinking"),g&&g.classList.toggle("hidden",M),
h&&h.classList.toggle("hidden",!M),_&&_.classList.toggle("hidden",M),L&&L.classList.toggle("hidden",
M),!!x)if(M){x.dataset.thinkingChecked===void 0&&(x.dataset.thinkingChecked=x.checked?"1":"0");const E=h?
Array.from(h.options).find(N=>N.value==="none"):null,O=!!(E&&!E.disabled&&!E.classList.contains("hid\
den"));x.checked=O?!!(h&&String(h.value||"").toLowerCase()!=="none"):!0,x.disabled=!O}else x.dataset.
thinkingChecked!==void 0&&(x.checked=x.dataset.thinkingChecked==="1",delete x.dataset.thinkingChecked)}
o(T,"setComposerThinkingMode");function I(){const d=get("model-select");if(!d)return;const m=d.value,
g=String(m||"").toLowerCase(),h=g.includes("deepseek"),x=g.startsWith("glm-"),_=get("thinking-option\
s"),L=get("reasoning-effort-container"),M=get("enable-thinking"),E=get("thinking-level"),O=get("thin\
king-budget"),N=get("enable-search"),X=get("search-container"),oe=get("url-context-container"),ie=get(
"enable-maps"),Y=get("maps-grounding-container"),fe=get("enable-sys-prompt"),D=get("sys-prompt-optio\
n"),ge=get("enable-python"),W=get("python-container"),me=get("prompt-cache-container"),ye=get("enabl\
e-prompt-cache"),ee=m==="gpt-5-search-api",he=m.includes("tts"),Ee=isMistralOcrModel(m),$e=g.includes(
"gemini-3.1-flash-lite-image"),at=g==="gemini-nano-banana-2.1",Ue=g.includes("gemini-3.1-flash-image")&&
!$e||at,ut=isClaudeModelKey(m),Tt=g==="gemini-3.8-flash-cyber",Le=isLlmModel()&&!h&&!he&&!g.includes(
"realtime")&&!g.includes("native-audio")&&!g.includes("live");me&&(Le?(me.classList.remove("hidden",
"opacity-50","pointer-events-none"),ye&&(ye.disabled=!1)):(ye&&(ye.checked=!1,ye.disabled=!0),me.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),_&&_.classList.add("hidden"),L&&L.classList.
add("hidden"),T("thinking");const ve=get("vision-model-info");if(ve&&ve.classList.add("hidden"),L){const Se=get(
"reasoning-effort");if(Se){Array.from(Se.options).forEach(De=>{const ze=g==="gpt-5.6"||g.startsWith(
"gpt-5.6-"),ct=g.startsWith("gpt-6"),fn=g==="gpt-6-sol"||g==="gpt-6-luna"||g==="gpt-6.1-terra",Xt=g===
"deepseek-v4.1-flash"||g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash\
-vision-exp",gn=g==="deepseek-v4-pro",Hn=g.includes("grok-4.5"),F=g.includes("grok-4.6");De.value===
"max"?De.classList.toggle("hidden",!ze&&!ct&&!Xt&&!gn):De.value==="xhigh"?De.classList.toggle("hidde\
n",!F&&!g.includes("multi-agent")&&!ze&&!ct):De.value==="medium"?De.classList.toggle("hidden",!(g.includes(
"grok-4.3")||Hn||F||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("mu\
lti-agent")||g.includes("gpt-5")||ct||g.includes("o1")||g.includes("o3"))):De.value==="none"?De.classList.
toggle("hidden",!g.includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!fn&&!Xt&&
!gn):De.value==="low"&&De.classList.toggle("hidden",gn)});const Pe=Se.selectedOptions&&Se.selectedOptions[0];
Pe&&Pe.classList.contains("hidden")&&(Se.value=h?"high":"medium")}}oe&&oe.classList.add("hidden"),Y&&
Y.classList.add("hidden"),M&&(M.disabled=!1),O&&(O.disabled=!0,O.classList.add("opacity-50"));const Fe=isGeminiImageModelKey(
m);if(he||Ee)X&&(get("enable-search").checked=!1,X.classList.add("opacity-50","pointer-events-none")),
oe&&(get("enable-url-context").checked=!1,oe.classList.add("opacity-50","pointer-events-none")),Y&&ie&&
(ie.checked=!1,Y.classList.add("opacity-50","pointer-events-none")),W&&(ge.checked=!1,W.classList.add(
"opacity-50","pointer-events-none")),fe&&D&&(fe.checked=!1,fe.disabled=!0,D.classList.add("opacity-5\
0"));else if(Ue||$e)Y&&ie&&(ie.checked=!1,Y.classList.add("hidden","opacity-50","pointer-events-none")),
_.classList.remove("hidden"),at?(Array.from(E.options).forEach(Se=>{Se.disabled=Se.value==="low"}),[
"minimal","medium","high"].includes(E.value)||(E.value="medium")):(Array.from(E.options).forEach(Se=>{
["low","medium"].includes(Se.value)&&(Se.disabled=!0),["minimal","high"].includes(Se.value)&&(Se.disabled=
!1)}),["minimal","high"].includes(E.value)||(E.value=$e?"minimal":"high")),M&&(M.disabled=!1),$e&&(N&&
(N.checked=!1,N.disabled=!0),X&&X.classList.add("opacity-50","pointer-events-none"));else if(Fe)Y&&ie&&
(ie.checked=!1,Y.classList.add("hidden","opacity-50","pointer-events-none"));else if(ut)_.classList.
remove("hidden"),O&&(O.disabled=!1,O.classList.remove("opacity-50")),Array.from(E.options).forEach(Se=>{
Se.disabled=!0}),W&&(ge.checked=!1,W.classList.add("opacity-50","pointer-events-none"));else if(Tt){
_&&_.classList.remove("hidden"),M&&(M.checked=!0,M.disabled=!0),Array.from(E.options).forEach(Pe=>{Pe.
disabled=!["low","medium","high"].includes(Pe.value)}),["low","medium","high"].includes(E.value)||(E.
value="medium"),[X,oe,Y,W].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")}),[
N,ie,ge].forEach(Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const Se=get("enable-url-context");Se&&(Se.
checked=!1,Se.disabled=!0),fe&&D&&(fe.disabled=!1,D.classList.remove("opacity-50"))}else if(m.includes(
"gemini")&&!Fe){_.classList.remove("hidden"),oe&&oe.classList.remove("hidden","opacity-50","pointer-\
events-none");const Se=m.includes("gemini-3")||m.includes("gemini-4");Y&&(Se?Y.classList.remove("hid\
den","opacity-50","pointer-events-none"):(ie&&(ie.checked=!1),Y.classList.add("hidden","opacity-50",
"pointer-events-none")));const Pe=m.includes("flash");Array.from(E.options).forEach(De=>{m==="gemini\
-4-argon"||m==="gemini-3.8-flash"||m==="gemini-3.7-flash"?De.disabled=!["low","medium","high"].includes(
De.value):m==="gemini-3.6-flash"?De.disabled=!["medium","high"].includes(De.value):m==="gemini-3.5-f\
lash-lite"?De.disabled=!["minimal","medium","high"].includes(De.value):["minimal","medium"].includes(
De.value)?De.disabled=!Pe:De.disabled=!1}),(m==="gemini-4-argon"||m==="gemini-3.8-flash"||m==="gemin\
i-3.7-flash")&&!["low","medium","high"].includes(E.value)||m==="gemini-3.6-flash"&&!["medium","high"].
includes(E.value)?E.value="medium":m==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(
E.value)?E.value="minimal":!Pe&&["minimal","medium"].includes(E.value)&&(E.value="high"),Se?M&&(M.checked=
!0,M.disabled=!0):M&&(M.disabled=!1),O&&m.includes("gemini-2.5")&&(O.disabled=!1,O.classList.remove(
"opacity-50")),O&&!m.includes("gemini-2.5")&&(O.disabled=!0,O.classList.add("opacity-50"))}if(isLlmModel()&&
(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4\
.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes(
"multi-agent")||g.includes("gpt")&&!g.includes("tts")))_&&_.classList.remove("hidden"),T("effort"),X&&
X.classList.remove("opacity-50","pointer-events-none");else if(h){_&&_.classList.remove("hidden"),T(
"effort");const Se=get("vision-model-info");if(Se&&Se.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),N&&(N.checked=!1,N.disabled=!0),X&&X.classList.add("opaci\
ty-50","pointer-events-none"),oe){const Pe=get("enable-url-context");Pe&&(Pe.checked=!1),oe.classList.
add("opacity-50","pointer-events-none")}Y&&ie&&(ie.checked=!1,Y.classList.add("opacity-50","pointer-\
events-none"))}else Ee||(X&&X.classList.remove("opacity-50","pointer-events-none"),Y&&ie&&(ie.checked=
!1,Y.classList.add("hidden","opacity-50","pointer-events-none")));if(he?W&&W.classList.add("opacity-\
50","pointer-events-none"):(W&&W.classList.remove("opacity-50","pointer-events-none"),(!Fe||Ue)&&!m.
includes("gpt-image")&&(fe.disabled=!1,D.classList.remove("opacity-50"))),(Fe&&!Ue||m.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(m)||isGrokVideoModel()||Ee)&&fe&&D&&(fe.checked=!1,fe.disabled=
!0,D.classList.add("opacity-50")),W&&(isLlmModel()?(W.classList.remove("hidden"),ge.disabled=!1):(ge.
checked=!1,ge.disabled=!0,W.classList.add("hidden"))),ee?(N&&(N.checked=!0,N.disabled=!0),X&&X.classList.
add("opacity-50","pointer-events-none"),W&&(ge.checked=!1,ge.disabled=!0,W.classList.add("opacity-50",
"pointer-events-none"))):N&&!m.includes("tts")&&!Ee&&!h&&!$e&&(N.disabled=!1),Tt){[N,ie,ge].forEach(
Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const Se=get("enable-url-context");Se&&(Se.checked=!1,Se.disabled=
!0),[X,oe,Y,W].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")})}x&&(ve&&ve.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[N,ge].forEach(Se=>{Se&&(Se.checked=!1,Se.disabled=!0)}),[X,W].forEach(Se=>{Se&&
Se.classList.add("opacity-50","pointer-events-none")}));const ot=get("mask-btn");ot&&(isGptImageModel()?
ot.classList.remove("hidden"):(ot.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),
updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),u(),f(),b(),y(),w(),updateBatchUi(m),v(),k(),
S(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}o(I,"toggle\
OptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",C),get("model-\
select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),C(),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const B=get("enable-canvas-mode");B&&(B.checked=canvasModeEnabled,B.addEventListener("change",()=>syncCanvasModeUi(
B.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const Z=get("enable-codin\
g-mode");Z&&(Z.checked=codingModeEnabled,Z.addEventListener("change",()=>syncCodingModeUi(Z.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const J=get(
"prompt-controls-toggle-btn");J&&(J.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,m,g){const h=(m||"").match(/\S*/)[0];if(h==="py\
exec")return"";if(h==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",_=(h||"").toLowerCase();
let L="";try{const D=hljs.getLanguage(h)?h:"plaintext";activeStreamingBubbleId&&x.length>2e4?L=escapeHtml(
x):L=hljs.highlight(x,{language:D}).value}catch{L=escapeHtml(x)}const M=encodeURIComponent(x).replace(
/'/g,"%27"),E=detectBlockedScriptsInCode(x),O=hashString(`${h||"TEXT"}
${x||""}`);let N="";if(canvasModeEnabled){const D=String(canvasPreviewState.selectedKey||"")===O,ge=D?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";N=`<button\
 class="canvas-preview-btn${D?" canvas-active":""}" data-code="${M}" data-code-key="${O}" data-canva\
s-lang="${escapeHtml(h||"txt")}" title="${ge}" aria-label="${ge}" aria-pressed="${D?"true":"false"}"\
><i class="fas ${D?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
_,x)){const D=E?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
N=`<button class="html-preview-btn" data-code="${M}" ${E?'data-suspicious="1"':""} title="${D}" aria\
-label="${D}"><i class="fas ${E?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const X=`\
<button class="download-btn" data-code="${M}" data-lang="${h||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,oe=_==="diff"?"":`<button class="codin\
g-target-btn" data-code="${M}" data-code-key="${O}" data-coding-lang="${escapeHtml(h||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,ie=(h||"TEXT")+(E?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),Y=getRenderableSvgCode(_,x);return`${Y?buildSvgCodeRenderHtml(Y,O):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${O}"><div class="code-header"\
><span class="code-lang">${ie}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${oe}${N}${X}\
<button class="copy-btn" data-code="${M}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${h}">${L}</code></pre></d\
iv></div>`},link(d,m,g){return`<a href="${d}" title="${m||""}" target="_blank">${g}</a>`},image(d,m,g){
return buildChatImageHtml(d,{alt:g,title:m})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(yi=get("version-update-dismis\
s"))==null||yi.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const ke=get("version-update-clear\
-cache");if(ke&&(ke.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),ke.
addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(
ke.checked)})),(vi=get("version-update-reload"))==null||vi.addEventListener("click",async()=>{var m;
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
classList.add("hidden")}const U=o(d=>{if(!d)return"\u4E0D\u660E";const m=new Date(d);return Number.isNaN(
m.getTime())?d:m.toLocaleString()},"formatSessionTime"),re=o(d=>{const m=Array.isArray(d)?d:[],g=get(
"passkey-list"),h=get("passkey-count");if(h&&(h.innerText=String(m.length)),!!g){if(!m.length){g.innerHTML=
'<div class="text-[11px] text-gray-500">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}g.innerHTML="",m.forEach((x,_)=>{const L=x&&x.id?String(x.id):"",M=document.createElement("di\
v");M.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const E=document.createElement("div");E.className="min-w-0";const O=document.createElement("\
div");O.className="text-xs text-gray-200 truncate",O.innerText=x&&x.name?String(x.name):`Security Ke\
y ${_+1}`;const N=document.createElement("div");N.className="text-[10px] text-gray-500 mt-1",N.innerText=
x&&x.created_at?`\u767B\u9332\u65E5\u6642: ${U(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
E.appendChild(O),E.appendChild(N),M.appendChild(E);const X=document.createElement("button");X.type="\
button",X.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",X.innerText="\u524A\u9664",X.disabled=!L,L&&(X.onclick=()=>window.removeWebAuthnCredential(
L)),M.appendChild(X),g.appendChild(M)})}},"renderPasskeyList"),le=o(d=>{const m=get("session-list");
if(m){if(!d||!d.length){m.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(g=>{const h=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",_=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
L=(g.user_agent||"Unknown").slice(0,120),M=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${h}${x}<div class="text-xs text-gray-200">${escapeHtml(
M)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(L)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(U(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(U(g.created_at))}</div></div>${_}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const h=g.getAttribute("data-session-id");if(!h||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:h})});let _={};try{_=await x.json()}catch{}if(x.ok){if(_.logged_out){
location.href="/login";return}await _e()}else showToast(_&&_.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),_e=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api\
/sessions");let g={};try{g=await m.json()}catch{}if(!m.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const h=(g.sessions||[]).filter(x=>!x.is_revoked);le(h)},"loadSessions"),pe=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>_e());const m=get("session-revoke-others-btn");m&&(m.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await _e():showToast("\u64CD\u4F5C\u306B\u5931\
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
(m.checked=!0),C())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const we=get("vision-model-change-btn");we&&(we.onclick=()=>_openVisionModelSelector());const $=get(
"compression-format-only");$&&($.onchange=()=>{const d=$.checked,m=get("compression-max-size"),g=get(
"compression-max-dim");m&&(m.disabled=d),g&&(g.disabled=d);const h=get("compression-size-wrap"),x=get(
"compression-dim-wrap");h&&(h.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const R=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");R(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const q=get("storage-us\
age-refresh");q&&(q.onclick=()=>loadStorageUsage());let P=null;const H=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),A=o((d={})=>{const m=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),h=get("account-transfer-progress-percent"),x=get("account-transfer-progress-text"),
_=get("account-transfer-progress-detail"),L=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&m.
classList.remove("hidden"),g&&(g.style.width=`${L}%`),h&&(h.textContent=`${Math.round(L)}%`),x&&(x.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),_){const E={queued:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};_.textContent=E[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const M=get("account-transfer-cancel-btn");M&&M.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),j=o(d=>{V&&(V.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),ce=o((d={})=>{const m=get("account-export-ready"),
g=get("account-export-ready-text"),h=get("account-export-expiry"),x=get("account-export-download-btn"),
_=!!(d.available&&d.download_url);if(m&&m.classList.toggle("hidden",!_),!_){x&&x.removeAttribute("hr\
ef");return}const L=Math.max(0,Number(d.size_bytes)||0),M=L>=1024*1024*1024?`${(L/(1024*1024*1024)).
toFixed(2)} GB`:`${(L/(1024*1024)).toFixed(1)} MB`;if(g){const E=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
g.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${M}${E}`}
if(h){const E=d.expires_at?new Date(d.expires_at):null;h.textContent=E&&!Number.isNaN(E.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${E.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),se=o(async d=>{for(;P===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await m.json().catch(()=>({}));if(m.ok&&(g.state!=="pending"&&A(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(m=>setTimeout(m,700))}
return null},"pollAccountTransfer"),te=o((d,m,g=!0)=>{m&&(A(m),ce(m),g&&m.state==="ready"?showToast(
m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):g&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),ne(d))},"handleFinishedAccountExport"),ae=o(async()=>{try{const d=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(ce(m),m.state==="ready"){A(m);return}if(["failed","cancelled","expired"].includes(
m.state)){A(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||P&&P.id===
m.job_id||P)return;const g={id:m.job_id,type:"export",stopped:!1,restored:!0};P=g,j(!0),A(m);const h=await se(
g);h&&te(g,h,!0)}catch{}},"refreshLatestAccountExport"),ne=o(d=>{P===d&&(P=null),d.stopped=!0,j(!1)},
"finishAccountTransfer"),K=get("account-transfer-cancel-btn");K&&(K.onclick=async()=>{const d=P;if(!(!d||
d.stopped)){d.cancelRequested=!0,K.disabled=!0,A({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),A({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&ce({available:!1}),ne(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const V=get("account-export-btn");V&&(V.onclick=async()=>{if(P)return;const d={id:H(),type:"\
export",stopped:!1};P=d,j(!0),ce({available:!1}),A({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u53D7\u3051\u4ED8\u3051\u3066\u3044\
\u307E\u3059"});try{const m=await apiFetch("/api/account/export",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:d.id}),keepalive:!0})),
g=await m.json().catch(()=>({}));if(m.status===409&&g.error==="export_in_progress"&&g.job_id)d.id=g.
job_id;else if(!m.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
A({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const h=await se(d);!d.cancelRequested&&h&&te(d,h,!0)}catch(m){const g=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";A({progress:0,phase:"faile\
d",message:g}),showToast(g,"error",!0),ne(d)}});const de=get("account-export-download-btn");de&&de.addEventListener(
"click",async d=>{const m=de.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),h=await g.json().catch(
()=>({}));g.ok&&h.available&&h.download_url?(de.href=h.download_url,window.location.assign(h.download_url)):
(ce(h),A(h),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),ae())}catch{window.location.assign(m)}}}),j(!1),ae();const Be=get("import-files-grid"),
pt=get("import-files-info"),ft=get("import-files-summary"),wt=o(d=>{const m=Math.max(0,Number(d)||0);
return m>=1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(
1)} MB`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let Ae=null;const ht=o(()=>{
if(!Ae)return;const d=Ae.files,m=Ae.selection;let g=0;d.forEach(_=>{m.has(_.archive_path)&&(g+=Number(
_.size_bytes)||0)});const h=Number(Ae.available_bytes)||0,x=g>h;ft&&(ft.textContent=`\u9078\u629E\u4E2D: ${wt(
g)} / \u5229\u7528\u53EF\u80FD: ${wt(h)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,ft.classList.
toggle("text-red-300",x)),pt&&(pt.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),_t=o(
()=>{if(!Be||!Ae)return;Be.innerHTML="";const d=Ae.files;if(!d.length){Be.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
ht();return}d.forEach(m=>{const g=document.createElement("label"),h=Ae.selection.has(m.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${h?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${h?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${wt(m.size_bytes)}</div></div>`;const x=g.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?Ae.selection.add(m.archive_path):
Ae.selection.delete(m.archive_path),g.classList.toggle("border-blue-500",x.checked),g.classList.toggle(
"border-gray-600",!x.checked),ht()}),Be.appendChild(g)}),ht()},"renderImportFileItems"),St=o(d=>new Promise(
m=>{if(Ae={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:m},_t(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),Lt=o(d=>{if(hideModal("import-files-modal"),Ae){const m=Ae.resolve;
Ae=null,m(d)}},"closeImportFileSelection"),bt=get("import-files-close");bt&&(bt.onclick=()=>Lt(null));
const Dt=get("import-files-cancel");Dt&&(Dt.onclick=()=>Lt(null));const Mt=get("import-files-confirm");
Mt&&(Mt.onclick=()=>{if(!Ae)return;const d=Array.from(Ae.selection);Lt(d.length?d.join(","):"__none_\
_")});const gt=get("import-files-select-all");gt&&(gt.onclick=()=>{Ae&&(Ae.files.forEach(d=>Ae.selection.
add(d.archive_path)),_t())});const At=get("import-files-none");At&&(At.onclick=()=>{Ae&&(Ae.selection.
clear(),_t())});const hn={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},Et=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const m=String(
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let yt=null;const $t=o(
d=>{if(yt){const m=yt;yt=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Yt=o(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}yt=m;const h=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(h.length?
x.innerHTML=h.map(L=>{const M=hn[L.field]||L.field,E=Et(L.current),O=Et(L.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(M)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
E)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(O)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const _=get("settin\
gs-confirmation-count");_&&(_.textContent=`${h.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Ht=get("settings-confir\
mation-modal");Ht&&Ht.addEventListener("click",d=>{d.target===Ht&&$t(!1)});const Qt=get("settings-co\
nfirmation-close");Qt&&(Qt.onclick=()=>$t(!1));const Zt=get("settings-confirmation-cancel");Zt&&(Zt.
onclick=()=>$t(!1));const en=get("settings-confirmation-confirm");en&&(en.onclick=()=>$t(!0));const G=get(
"account-import-btn"),be=get("account-import-inplace"),Ie=get("account-import-inplace-warning");if(be&&
Ie){const d=o(()=>Ie.classList.toggle("hidden",!be.checked),"syncInplaceWarn");be.addEventListener("\
change",d),d()}G&&(G.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:null,
g=get("account-import-categories"),h=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).
map(ie=>ie.value):[],x=get("account-import-inplace"),_=!!(x&&x.checked),L=get("account-import-settin\
gs-bypass"),M=!!(L&&L.checked);let E=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!h.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(ie=>(ie.
closest("label")&&ie.closest("label").textContent||ie.value).trim()):h;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${O.join("\u3001")}${_?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const N={id:H(),type:"import",stopped:!1,controller:new AbortController};
P=N,j(!0),A({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const X=get("account-import-result");let oe=Promise.resolve(null);try{const Y=Math.max(1,Math.ceil(m.
size/10485760)),fe=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:N.controller.
signal})),D=await fe.json().catch(()=>({}));if(!fe.ok)throw new Error(D.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
N.uploadId=D.upload_id;const ge=D.chunk_size||10485760;let W=0,me=0;const ye=o(async()=>{for(;;){const Le=me++;
if(Le>=Y)return;const ve=m.slice(Le*ge,Math.min(m.size,(Le+1)*ge)),Fe=new FormData;Fe.append("chunk",
ve,m.name),Fe.append("index",String(Le));const Ye=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
N.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Fe,signal:N.controller.signal})),
ot=await Ye.json().catch(()=>({}));if(!Ye.ok)throw new Error(ot.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W++,A({progress:Math.min(35,Math.round(W/Y*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${W}\
/${Y}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ee=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ee=!0);try{await Promise.
all([ye(),ye(),ye()]);const Le=await apiFetch(`/api/account/import/upload/${encodeURIComponent(N.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:N.controller.signal})),ve=await Le.json().
catch(()=>({}));if(!Le.ok)throw new Error(ve.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
A({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ee&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let he="",Ee=!1,$e=0;const at=o(
async()=>{let Le=!1;const ve=o(()=>{Le||(Le=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Fe=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Ye=await Fe.
json().catch(()=>null);if(!Fe.ok||!Ye){ve();return}cacheUserSettings(Ye);const ot=get("settings-moda\
l");if(ot&&ot.classList.contains("modal-open"))try{zn(Ye)}catch{}Ye.theme_color&&applyThemeColor(Ye.
theme_color,!0),Object.prototype.hasOwnProperty.call(Ye,"minimal_prompt_mode")&&Ye.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Ye,"compact_prompt_mode")&&setCompactPromptMode(
!!Ye.compact_prompt_mode)}catch{}ve()},"refreshSettingsFormAfterImport"),Ue=o(Le=>{const ve=Le&&Le.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";X&&(X.textContent=`\u5B8C\u4E86: ${ve}`,
X.classList.remove("hidden","text-red-300"),X.classList.add("text-emerald-300")),A({progress:100,phase:"\
completed",message:ve}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&at()},"finishImportSuccess"),ut=o(async()=>{
try{const ve=await(await apiFetch(`/api/account/transfer/${N.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return ve&&ve.state?ve:null}catch{return null}},"fetchImportSta\
tus"),Tt=o(async()=>{const Le=await ut();if(!Le)return{status:"unknown"};if(Le.state==="completed")return Ue(
Le),{status:"done"};if(["failed","cancelled","expired"].includes(Le.state))throw new Error(Le.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Le.state==="needs_sel\
ection"&&Array.isArray(Le.files)){const ve=await St({files:Le.files,available_bytes:Le.available_bytes});
return ve===null?(A({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
N.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(N.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(he=ve,{status:"reselect"})}if(Le.state===
"needs_settings_confirmation"&&Array.isArray(Le.settings_changes))return await Yt({settings_changes:Le.
settings_changes})?(E=!0,{status:"reselect"}):(A({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),N.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
N.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Le.state==="running"){const ve=await Promise.race([oe.catch(()=>null),new Promise(Fe=>setTimeout(
()=>Fe(null),6e4))]);if(ve&&ve.state==="completed")return Ue(ve),{status:"done"};throw ve&&["failed",
"cancelled","expired"].includes(ve.state)?new Error(ve.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ee;){N.stopped=!0,await oe.catch(()=>null),
N.stopped=!1,oe=se(N);let Le;try{Le=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:N.uploadId,
categories:h.join(","),job_id:N.id,selected_files:he,restore_inplace:_,confirm_settings:E||M}),signal:N.
controller.signal}))}catch(ze){if(N.cancelRequested||ze&&ze.name==="AbortError")throw ze;const ct=await Tt();
if(ct.status==="done"){Ee=!0;break}if(ct.status==="cancelled")return;if(ct.status==="reselect")continue;
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let ve=null;try{ve=await Le.json()}catch{ve=null}if(ve===null){const ze=await Tt();if(ze.status==="d\
one"){Ee=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;if(Le.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Le.ok&&ve.error==="storage_limit_files"&&ve.files){const ze=await St(ve);if(ze===null){A({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
N.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(N.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}he=ze;continue}if(ve&&ve.status==="settings_confirmation"&&
Array.isArray(ve.settings_changes)){if(!await Yt(ve)){A({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),N.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(N.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}E=!0;continue}if(!Le.ok)throw new Error(ve.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Fe=ve.imported||{},Ye=[`\u8A2D\u5B9A ${Fe.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Fe.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Fe.chats||0}\u4EF6`,`Gem ${Fe.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Fe.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Fe.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Fe.
diagnostics||0}\u4EF6`].join(" / "),ot=ve.duplicates||{},Se={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Pe=[];for(const ze of Object.keys(Se)){const ct=Number(ot[ze])||0;ct>
0&&Pe.push(`${Se[ze]} ${ct}\u4EF6`)}const De=Pe.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Pe.
join("\u3001")}\uFF09`:"";X&&(X.textContent=`\u5B8C\u4E86: ${Ye}${De}`,X.classList.remove("hidden","\
text-red-300"),X.classList.add("text-emerald-300")),A({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&at(),Ee=!0}}catch(ie){if(N.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(N.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),N.cancelRequested||ie&&ie.name==="AbortError")return;const Y=ie&&ie.message?
ie.message:"",fe=Y==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
Y||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";A({progress:0,phase:"f\
ailed",message:fe}),X&&(X.textContent=fe,X.classList.remove("hidden","text-emerald-300"),X.classList.
add("text-red-300")),showToast(fe,"error",!0)}finally{N.stopped=!0,await oe.catch(()=>null),ne(N)}});
const Ne=get("account-dedupe-btn"),Ve=get("account-dedupe-result"),nt=o((d,m=!1)=>{Ve&&(Ve.textContent=
d,Ve.classList.remove("hidden"),Ve.classList.toggle("text-red-300",!!m),Ve.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");Ne&&(Ne.onclick=async()=>{const d=o(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await m.json().catch(()=>null);if(!m.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){nt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const h=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const N of[
"chats","gems","files","feedback","diagnostics"]){const X=Number(g.duplicates&&g.duplicates[N])||0;X>
0&&h.push(`${x[N]} ${X}\u4EF6`)}const _=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${h.join("\u3001")}${_}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const L=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),M=await L.json().catch(()=>null);if(!L.ok||!M)throw new Error(
M&&M.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const E=[];for(const N of["chats","gems","files","feedback","diagnostics"]){const X=Number(M.removed&&
M.removed[N])||0;X>0&&E.push(`${x[N]} ${X}\u4EF6`)}const O=Number(M.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${M.kept_referenced_files}\u4EF6\uFF09`:"";nt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${E.join("\u3001")||"0\u4EF6"}${O}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!Ne.disabled){Ne.disabled=!0,nt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){nt(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{Ne.disabled=!1}}});const dt=get("site-cache-usage-refresh");dt&&(dt.onclick=()=>loadSiteCacheUsage());
const Ge=get("clear-site-cache-btn");Ge&&(Ge.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Ge)});
const Je=get("enc-scan-result"),We=o(async(d=null)=>{Je&&(Je.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
m,{cache:"no-store"}),h=await g.json();if(!g.ok){Je&&(Je.textContent=h.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=h.total||0,_=h.encrypted||0,L=h.unencrypted||0;let M=`Total: ${x} / Encrypted: ${_} /\
 Plain: ${L}`;if(h.samples&&h.samples.length){const E=h.samples.slice(0,8).map(O=>{const N=O.timestamp?
new Date(O.timestamp).toLocaleString():"";return`#${O.id} (${O.role||""}) ${N}`}).join(" / ");M+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${E}</div>`}Je&&(Je.innerHTML=M)}catch{Je&&(Je.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Ze=get("enc-scan-all");Ze&&(Ze.onclick=()=>We(
null));const Ke=get("enc-scan-thread");Ke&&(Ke.onclick=()=>currentThreadId?We(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const lt=get("admin-enc-\
list");let It=null,qt=!1;const bn=o(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThread\
EncryptedFromMessages"),Re=o(()=>{It=bn(allMessages)},"refreshCurrentThreadEncStateFromMessages"),He=o(
async(d,m,{confirmPrompt:g=!0,reloadCurrent:h=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;qt=!0;try{const _=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
L=await _.json().catch(()=>({}));return _.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${L.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),It=!!m,h&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),lt&&await Pt(),!0):
(showToast(L.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{qt=!1}},"setAdminThreadEncry\
ption"),it=o(d=>{if(!lt)return;const m=d.threads||[];if(!m.length){lt.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
lt.innerHTML=m.map(g=>{const h=g.encrypted_count>0?"enc":"plain",x=h==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",_=h==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",L=g.
updated_at?new Date(g.updated_at).toLocaleString():"",M=escapeHtml(String(g.thread_id)),E=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${E?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${L} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${M}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${_} text-white px-2 py-1 roun\
ded shrink-0" data-id="${M}" data-enable="${h==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Pt=o(async()=>{if(lt){lt.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){lt.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(it(m),currentThreadId&&Array.isArray(m.threads)){const g=m.threads.find(h=>String(
h.thread_id)===String(currentThreadId));g&&(It=!!g.encrypted)}}catch{lt.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Pt()),window.__loadAdminEncThreads=
Pt,window.__refreshAdminThreadEncState=Re,window.__setAdminThreadEncryption=He;const tn=get("encrypt\
ion-status-admin-toggle");tn&&tn.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),lt&&(lt.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const M=m.getAttribute("data-id");if(!M)return;typeof Gt==
"function"?Gt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(M)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||qt)return;const h=g.getAttribute("data-i\
d"),x=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const L=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await He(h,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=L,await Pt()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const Ot=o(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(g=>{const h=g.status||"new",x=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
_=g.created_at?new Date(g.created_at).toLocaleString():"",L=g.replied_at?new Date(g.replied_at).toLocaleString():
"",M=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${x}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(_)}</div>
                            </div>
                            <div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(h)}</di\
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
L)}</div>`:""}
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
                    `}).join("")}},"renderBanAppeals"),xe=o(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const x=(await g.json()).unread_count||0;m.textContent=String(x),d&&x>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${x}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Te=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const g=await m.json();Ot(g.items||[]),
await xe(!1)}catch{}}},"loadBanAppeals"),Xe=o(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await Te()}catch{}},"markBanAppealsRead"),et=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await Te()}catch{}},"updateBanAppealStatus"),rt=o(()=>{
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
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),xt=o(()=>{const d=get("set-st\
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
                `,m.appendChild(g);const h=get("reset-llm-transcribe-prompt");h&&(h.onclick=()=>{const x=get(
"set-llm-transcribe-prompt");x&&(x.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),vt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
700 rounded p-2 text-xs text-gray-200";return vt.map(h=>{const x=h.mcpLocked===!0,_=x?'<div class="t\
ext-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002\
</div>':"",L=x?`<input type="checkbox" id="${d}-auto-sys-${h.key}-enabled" class="accent-yellow-500 \
w-3 h-3" disabled>`:`<input type="checkbox" id="${d}-auto-sys-${h.key}-enabled" class="accent-yellow\
-500 w-3 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${h.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${x?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${L}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${d}-auto-sys-${h.key}-text" class="${g}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${h.hint?`<div class="text-[10px] text-gray-500 mt-1">${h.hint}</div>`:""}
                        ${_}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{vt.forEach(g=>{const h=m&&
typeof m=="object"?m[g.key]||{}:{},x=get(`${d}-auto-sys-${g.key}-enabled`),_=get(`${d}-auto-sys-${g.
key}-text`);x&&(g.mcpLocked===!0?x.disabled=!0:x.checked=h.enabled!==!1),_&&(_.value=h.text||"",_.placeholder=
h.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Gn=o((d,m=null)=>{if(m){const g=get(m);g&&(g.checked=!0)}vt.forEach(g=>{const h=get(`${d}-auto\
-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);if(h&&(g.mcpLocked!==!0?h.checked=!0:h.
disabled=!0),x){const _=x.placeholder||"";x.value=_}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Un=o(d=>{const m={};return vt.forEach(g=>{const h=get(`${d}\
-auto-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);m[g.key]={enabled:g.mcpLocked===!0?
!0:h?h.checked:!0,text:x?x.value:""}}),m},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
Un,window.ensureAutoSystemPromptSettingsCard=()=>{const d=get("set-global-sys-prompt-enabled"),m=d?d.
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
                `,m.appendChild(g)},rt(),xt(),R();const xi=o(()=>{const d=get("set-default-model");if(!d)
return;const m=d.value;d.innerHTML="",MODELS.forEach(h=>{const x=document.createElement("optgroup");
x.label=h.category,(h.items||[]).forEach(_=>{const L=document.createElement("option");L.value=_.id,L.
textContent=_.name,x.appendChild(L)}),d.appendChild(x)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"po\
pulateDefaultModelOptions"),ki=o(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(h=>{const x=(h.items||[]).filter(L=>{const M=(L.id||"").toLowerCase();
return M.startsWith("gemini-")||M.startsWith("gpt-4o")||M.startsWith("claude-")||M.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
M)});if(x.length===0)return;const _=document.createElement("optgroup");_.label=h.category,x.forEach(
L=>{const M=document.createElement("option");M.value=L.id,M.textContent=L.name+" \u2605",_.appendChild(
M)}),d.appendChild(_)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
"gemini-3-flash-preview";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"populateDefaul\
tVisionModelOptions"),zn=o(d=>{if(!d)return;cacheUserSettings(d);const m=get("app-global-sys-prompt-\
preview");m&&(m.value=d.global_system_prompt_effective||"");const g=get("app-global-sys-prompt-previ\
ew-status");g&&(d.global_system_prompt_enabled===!1?g.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
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
const h=get("bot-status");h&&(d.is_bot_banned?(h.textContent=`BAN\u4E2D: ${d.bot_ban_reason||"Bot de\
tection"}`,h.classList.remove("hidden"),h.classList.add("text-red-400")):h.classList.add("hidden")),
d&&d.theme_color?(applyThemeColor(d.theme_color,!0),syncThemeInputs(d.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=d.username);const x=get("2fa-badge"),_=get("disable-2fa-btn");d.is_2fa_enabled?
(x.innerText="ENABLED",x.classList.replace("bg-gray-700","bg-green-600"),x.classList.replace("text-g\
ray-400","text-white"),_.classList.remove("hidden")):(x.innerText="DISABLED",x.classList.replace("bg\
-green-600","bg-gray-700"),x.classList.replace("text-white","text-gray-400"),_.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const L=get(
"set-passkey-only-login"),M=get("passkey-only-note"),E=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(re(E),L){L.checked=!!d.passkey_only_login;const ie=E.length>0||!!d.has_webauthn;L.disabled=!ie,
ie||(L.checked=!1),M&&(ie?M.classList.add("hidden"):M.classList.remove("hidden"))}const O=get("mig-s\
tatus-box"),N=get("mig-progress-text"),X=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){O.classList.remove("hidden");const ie=(d.migration_progress||"").split("/");if(ie.length===
2){const Y=parseInt(ie[0]||"0",10),fe=parseInt(ie[1]||"0",10);N&&(N.innerText=`${Y} / ${fe}`),X&&fe>
0&&(X.style.width=`${Math.min(100,Math.floor(Y/fe*100))}%`)}}else O.classList.add("hidden"),X&&(X.style.
width="0%"),N&&(N.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&zn(d);const m=get(
"search-box"),g=m?m.value:"";clearTimeout(searchTimeout);const h=get("settings-search");if(h&&(h.value=
""),filterSettings(),xi(),ki(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),ae(),xt(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),xe(!0),Te(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),on(),pe(),_e();try{loadMcpServers()}catch{}};const Gt=o((d=!1)=>{snapshotSidebarHistory(
"settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),_i=o(()=>{const d=get("set-th\
eme-color"),m=get("set-theme-color-text"),g=get("theme-reset-btn"),h=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((_,L=!0)=>{const M=normalizeHex(_);M&&(applyThemeColor(M,L),syncThemeInputs(
M))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),m&&(m.addEventListener("chan\
ge",()=>{const _=normalizeHex(m.value);if(!_){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(_,!0)}),m.addEventListener("keydown",_=>{_.key==="Enter"&&(_.preventDefault(),
m.blur())})),g&&(g.onclick=()=>x(THEME_DEFAULT,!0)),h.forEach(_=>{_.addEventListener("click",()=>x(_.
getAttribute("data-color"),!0))})},"bindThemeControls"),Si=o(()=>{const d=get("reset-global-sys-prom\
pt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const m=get(
"reset-thread-sys-prompt");m&&(m.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{Gn("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-thread-auto-sys-prompt-defaults");h&&(h.onclick=()=>{Gn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Gt();const Wn=get("settings-header-close");Wn&&(Wn.onclick=()=>Gt());
const Ut=get("settings-search");Ut&&(Ut.addEventListener("input",filterSettings),Ut.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const g=m.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Vn=get(
"settings-search-clear");Vn&&Vn.addEventListener("click",()=>{Ut&&(Ut.value="",filterSettings(),Ut.focus())}),
_i(),Si(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Tn=get("set-gemini-local-python-dialog");Tn&&(Tn.onchange=()=>setGeminiLocalPyDialogEnabled(Tn.
checked));const Jn=get("set-gemini-backend");Jn&&(Jn.onchange=()=>syncGeminiBackendUi());const Kn=get(
"set-admin-api-key-mode");Kn&&(Kn.onchange=()=>syncAdminApiKeyModeUi());const Mn=get("set-temp-chat-\
timeout-seconds");Mn&&(Mn.onchange=()=>{applyTemporaryChatTimeoutSeconds(Mn.value)});const Xn=get("s\
lash-command-cancel-btn");Xn&&(Xn.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{var M;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const d=get("set-username"),m=get("set-password"),g=readPromptBarModeFromForm(),h={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:Un(
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
-default-2fa-method").value:"totp",new_username:d?d.value:null,new_password:m?m.value:null},x=get("s\
et-e2ee")?get("set-e2ee").checked:!1,_=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
x!==_&&(h.enable_e2ee=x),get("set-openai")&&(h.openai_key=get("set-openai").value),get("set-gemini")&&
(h.gemini_key=get("set-gemini").value),get("set-deepseek")&&(h.deepseek_key=get("set-deepseek").value),
get("set-zai")&&(h.zai_key=get("set-zai").value),get("set-kimi")&&(h.kimi_key=get("set-kimi").value),
get("set-mistral")&&(h.mistral_key=get("set-mistral").value),get("set-ideogram")&&(h.ideogram_key=get(
"set-ideogram").value),get("set-anthropic")&&(h.anthropic_key=get("set-anthropic").value),h.model_api_keys=
normalizeModelApiKeyMap(modelApiKeyMap),get("set-gemini-backend")&&(h.gemini_backend=normalizeGeminiBackend(
get("set-gemini-backend").value)),get("set-gemini-vertex-project")&&(h.gemini_vertex_project=get("se\
t-gemini-vertex-project").value),get("set-gemini-vertex-location")&&(h.gemini_vertex_location=get("s\
et-gemini-vertex-location").value),get("set-gemini-vertex-credentials-json")&&(h.gemini_vertex_credentials_json=
get("set-gemini-vertex-credentials-json").value),get("set-xai")&&(h.xai_key=get("set-xai").value),get(
"set-google-key")&&(h.google_key=get("set-google-key").value),get("set-google-project")&&(h.google_project=
get("set-google-project").value),get("set-admin-api-key-mode")&&(h.admin_api_key_mode=normalizeAdminApiKeyMode(
get("set-admin-api-key-mode").value)),get("set-bot-detect")&&(h.bot_detection_enabled=get("set-bot-d\
etect").checked),get("set-bot-detect-global")&&(h.bot_detection_global_enabled=get("set-bot-detect-g\
lobal").checked);const L=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(h)});if(L.ok){let E="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const oe=await L.json();oe&&oe.message&&(E=oe.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!h.light_mode_enabled,liquid_glass_enabled:!!h.liquid_glass_enabled})),
window.applySavedUserSystemPromptSettings({system_prompt:h.system_prompt,system_prompt_enabled:h.system_prompt_enabled,
apply_global_system_prompt:h.apply_global_system_prompt,apply_auto_system_prompt_notices:h.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:h.auto_system_prompt_notices_config}),Gt();const O=currentUsername,
N=CHAT_CONFIG.enableE2EE;enterToSend=h.enter_to_send,autoSearchOnLinks=h.auto_search_on_links;const X=useSwCache;
useSwCache=h.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!h.clear_cache_on_version_update),
compactPromptMode=h.compact_prompt_mode,minimalPromptMode=h.minimal_prompt_mode,voiceStudioUiEnabled=
h.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=h.temp_chat_timeout_seconds,applyThemeColor(h.theme_color,
!0),syncThemeInputs(h.theme_color),applyLightMode(h.light_mode_enabled),applyLiquidGlassMode(h.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),X!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(E,"success"),syncClientDebugLogToggle(
h.enable_client_debug_log,"settings saved"),h.new_username&&h.new_username!==O?setTimeout(()=>location.
reload(),1e3):h.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let E={};try{E=await L.json()}catch{}showToast(E.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
son"},body:JSON.stringify({username:m,mode:"single"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${m} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const _=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const _=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(W=>{m&&(m.textContent=W)},"setBox"),h=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((W,me)=>!W||!me||me<=0?0:W*8/(me/1e3)/1e3/1e3,"toMbps"),_=o(W=>Number.isFinite(
W)?`${W.toFixed(0)} ms`:"-","fmtMs"),L=o(W=>Number.isFinite(W)?`${W.toFixed(W>=100?0:1)} Mbps`:"-","\
fmtMbps"),M=o(async(W,me)=>{const ye=await W.json().catch(()=>({}));return ye&&ye.error?ye.error:me},
"parseErr"),E=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let W=0;W<4;W++){const me=performance.now(),ye=await apiFetch(
`/api/speedtest/ping?_=${h()}`,{cache:"no-store"}),ee=performance.now();if(!ye.ok)throw new Error(await M(
ye,"ping_failed"));await ye.json().catch(()=>({})),E.push(ee-me)}const O=E.reduce((W,me)=>W+me,0)/Math.
max(1,E.length),N=Math.min(...E),X=o(async W=>{const me=performance.now(),ye=await apiFetch(`/api/sp\
eedtest/download?bytes=${W}&_=${h()}`,{cache:"no-store"});if(!ye.ok)throw new Error(await M(ye,"down\
load_failed"));const ee=await ye.arrayBuffer(),he=performance.now();return{bytes:ee.byteLength||W,ms:he-
me,mbps:x(ee.byteLength||W,he-me)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${_(O)}
\u6E2C\u5B9A\u4E2D... download`);const oe=[];for(const W of[2*1024*1024,8*1024*1024])oe.push(await X(
W)),g(`\u6E2C\u5B9A\u4E2D... ping ${_(O)}
download ${L(Math.max(...oe.map(me=>me.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const ie=Math.max(...oe.map(W=>W.mbps)),Y=o(async W=>{const me=new Uint8Array(
W),ye=performance.now(),ee=await apiFetch(`/api/speedtest/upload?_=${h()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:me,cache:"no-store"}),he=performance.now();if(!ee.ok)
throw new Error(await M(ee,"upload_failed"));const Ee=await ee.json().catch(()=>({})),$e=Number(Ee.bytes_received||
W)||W;return{bytes:$e,ms:he-ye,mbps:x($e,he-ye),serverMs:Number(Ee.server_elapsed_ms||0)||0}},"runUp\
load"),fe=[];for(const W of[1*1024*1024,4*1024*1024])fe.push(await Y(W));const D=Math.max(...fe.map(
W=>W.mbps)),ge=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${_(O)} / ${_(N)}`,`Download (best): ${L(ie)}`,`Upload (best): ${L(D)}`,`Download r\
uns: ${oe.map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,`Upload runs: ${fe.
map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(ge.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Te()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Xe()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){g&&await Xe([Number(g)]);return}if(m.classList.contains(
"ban-appeal-status")){const h=m.getAttribute("data-status");g&&h&&await et({id:Number(g),status:h});
return}if(m.classList.contains("ban-appeal-reply-send")){const h=m.closest("[data-appeal-id]"),x=h?h.
querySelector(".ban-appeal-reply"):null,_=x?x.value:"";g&&await et({id:Number(g),admin_reply:_});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const h=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await et({id:Number(
g),block_user:!0,block_reason:h});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
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
draw")),get("marker-tool-mosaic")&&(get("marker-tool-mosaic").onclick=()=>setMarkerMode("mosaic")),get(
"marker-tool-crop")&&(get("marker-tool-crop").onclick=()=>setMarkerMode("crop"));const An=get("marke\
r-color-picker");An&&(An.oninput=d=>setMarkerColor(d.target.value),An.onchange=d=>setMarkerColor(d.target.
value));const En=get("marker-opacity");En&&(En.oninput=d=>setMarkerOpacity(d.target.value),En.onchange=
d=>setMarkerOpacity(d.target.value));const yn=get("marker-opacity-number");yn&&(yn.onchange=d=>setMarkerOpacity(
d.target.value),yn.onblur=d=>setMarkerOpacity(d.target.value),yn.onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(
d.target.value),d.target.blur())}),document.querySelectorAll("#marker-toolbar .marker-color-chip[dat\
a-marker-color]").forEach(d=>{d.onclick=()=>setMarkerColor(d.getAttribute("data-marker-color"))}),get(
"marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),get("marker-crop\
-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&(get("marker-un\
do").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),
get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),
initCropCanvas(),window.addEventListener("resize",()=>{const d=get("marker-modal");!d||d.classList.contains(
"hidden")||(applyMarkerTransform(),renderCropOverlay())});const Li=o(()=>{const d=get("upload-modal");
return!!(d&&!d.classList.contains("hidden"))},"isUploadModalOpen"),zt=get("drop-overlay");let nn=0;const Ci=o(
()=>{Li()||zt&&(zt.classList.remove("hidden"),zt.classList.add("flex"))},"showDropOverlay"),sn=o(()=>{
nn=0,zt&&(zt.classList.add("hidden"),zt.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
sn;const kt=get("upload-dropzone");kt&&(kt.addEventListener("dragover",d=>{d.preventDefault(),kt.classList.
add("dragover")}),kt.addEventListener("dragleave",()=>{kt.classList.remove("dragover")}),kt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),kt.classList.remove("dragover"),sn();const m=d.dataTransfer?
d.dataTransfer.files:null;m&&m.length&&handleFiles(m)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(nn+=1,Ci())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(nn=Math.max(0,nn-1),(nn===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&sn())}),window.addEventListener(
"dragend",()=>{sn()}),window.addEventListener("drop",d=>{sn(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(kt&&kt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const Yn=get("bot-admin-modal"),Ti=o(d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML=
"",!d||!d.length){m.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,h)=>{const x=!!g.is_bot_banned,_=g.bot_detection_enabled!==!1,L=Number(g.lock_remaining_seconds)||
0,M=document.createElement("div");M.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",M.style.animationDelay=`${Math.min(h,12)*.02}\
s`,M.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(g.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${x?"BAN\u4E2D":"\u6B63\u5E38"}${L?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(L/60)}\u5206\uFF09`:""} ${g.bot_ban_reason?
" / "+escapeHtml(g.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${_?"1":"0"}">${_?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${x?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-bann\
ed="${x?"1":"0"}">${x?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${x?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(g.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        ${g.is_admin?"":`<button class="bot-delete-account bg-red-800 hover:bg-red-7\
00 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(g.username)}\
">\u524A\u9664</button>`}
                    `,m.appendChild(M)})}},"renderBotUsers"),an=o(async(d="")=>{const m=get("bot-adm\
in-list");m&&(m.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),h=await g.json();g.ok&&h&&h.users?Ti(h.users):(m&&(m.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{m&&(m.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),$n=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Yn))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await an(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=$n;const In=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=In,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{In()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||Yn)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
$n()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await an(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await an("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async d=>{d.key==="Enter"&&await an(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute(
"data-user");if(!g)return;let h;if(m.classList.contains("bot-toggle-detect")){const x=m.getAttribute(
"data-enabled")!=="1";h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:g,action:"toggle_detection",enabled:x})})}else if(m.classList.
contains("bot-toggle-ban"))if(m.getAttribute("data-banned")==="1")h=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"ban",reason:"Admin ban"})})}else if(m.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"unban_linked"})})}else if(m.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"delete_account"})})}if(h){if(h.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(h.ok){if(m.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),g===currentUsername)){location.href="/";return}}else{
let x={};try{x=await h.json()}catch{}showToast(x.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
x.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await an(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Wt={"/settings":{id:"settings-mod\
al",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:o(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:o(()=>{hi(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
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
dal",open:o(()=>$n(),"open")},"/admin-bot-logs":{id:"bot-log-modal",open:o(()=>In(),"open")}},Qn=o((d,m=!1)=>{
switch(d){case"settings-modal":Gt(m);break;case"upload-modal":closeUploadModal(m);break;case"image-s\
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
pi-key-required-modal":{const h=get("api-key-modal-cancel-btn");h&&typeof h.onclick=="function"?h.click():
hideModal(d);break}case"lyria-studio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(
d);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(d);break;case"\
version-update-modal":const g=localStorage.getItem("app_version")||"";g&&localStorage.setItem("versi\
on_notified",g),hideModal(d);break;default:hideModal(d);break}},"closeModalById");window.addEventListener(
"popstate",d=>{let m=!1;Object.values(Wt).forEach(x=>{const _=get(x.id);_&&_.classList.contains("mod\
al-open")&&location.pathname!==Object.keys(Wt).find(L=>Wt[L].id===x.id)&&(Qn(x.id,!0),m=!0)});const g=location.
pathname.match(/^\/c\/(.+)$/);if(g){const x=decodeURIComponent(g[1]);String(currentThreadId)!==String(
x)&&loadMessages(x,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});
const h=Wt[location.pathname];if(h){const x=get(h.id);x&&!x.classList.contains("modal-open")&&h.open()}});
const Zn=location.pathname;Wt[Zn]&&(history.replaceState({},"","/"),setTimeout(()=>Wt[Zn].open(),500)),
get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const d=get("easy-login-mi\
ns"),m=d?parseInt(d.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${m}\u5206\u9593\u6709\u52B9\
\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const h=await(await apiFetch("/api/easy_login",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:m})})).json();h&&h.temp_password?
(get("easy-login-code").textContent=h.temp_password,get("easy-login-exp").textContent=h.expires_at||
"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const m=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(m&&m.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const Mi=4,Ai=5*1024*1024;function ei(d,m,g){let h=[];const x=o(()=>{g&&(g.innerHTML="",
h.forEach((_,L)=>{const M=document.createElement("div");M.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const E=document.createElement("img");E.src=_.url,E.alt=_.file.name,
E.className="w-full h-full object-cover";const O=document.createElement("button");O.type="button",O.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",O.textContent=
"\xD7",O.onclick=()=>{URL.revokeObjectURL(_.url),h.splice(L,1),x()},M.appendChild(E),M.appendChild(O),
g.appendChild(M)}))},"render");return d&&m&&(d.onclick=()=>m.click(),m.onchange=_=>{for(const L of Array.
from(_.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(L.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(L.
size>Ai){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(h.
length>=Mi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}h.push({file:L,url:URL.createObjectURL(L)})}_.target.value="",x()}),{files:o(()=>h.
map(_=>_.file),"files"),clear:o(()=>{h.forEach(_=>URL.revokeObjectURL(_.url)),h=[],x()},"clear")}}o(
ei,"createFbImagePicker");async function ti(d,m){try{const g=new FormData;return m.forEach(h=>g.append(
"images",h,h.name)),(await apiFetch(d,{method:"POST",body:g})).ok}catch{return!1}}o(ti,"uploadFbImag\
es");const ni=ei(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-message").value.trim();if(!m){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:m,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},h=get("fb-attach-logs"),x=h&&h.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;x&&(g.client_logs=x);const _=get("fb-attach-chat"),L=!!(_&&_.checked);if(L&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}L&&(g.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const M=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),E=M.ok?await M.clone().json().
catch(()=>({})):{},O=ni.files(),N={count:O.length,saved:!0};M.ok&&N.count&&(N.saved=await ti(`/api/f\
eedback/${E.public_id}/images`,O)),(window.ActivityLog?await window.ActivityLog.reportFeedback(M,x,L,
N):M.ok)&&(get("fb-title").value="",get("fb-message").value="",ni.clear(),_&&(_.checked=!1),h&&(h.checked=
!0),on())};async function ii(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),on()}o(ii,"deleteFeedback");const Ei="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Pn=[];function si(d,m){const g=d.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(h=>{const x=h.role==="admin"?m&&h.author?`\u7BA1\u7406\u8005 (${h.
author})`:"\u7BA1\u7406\u8005":m?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",_=m?[h.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${h.
log_file}`,h.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${h.image_count}\u679A\uFF09: ${h.image_dir}`]:
[h.image_count&&`\u6DFB\u4ED8\u753B\u50CF: ${h.image_count}\u679A`];return`<div class="text-[11px]">\
<div class="text-gray-400">${escapeHtml(x)} / ${escapeHtml(h.created_at||"")}</div><div class="text-\
sm whitespace-pre-wrap ${h.role==="admin"?"text-green-300":"text-gray-200"}">${escapeHtml(h.message)}\
</div>${_.filter(Boolean).map(L=>`<div class="${m?"text-amber-300":"text-gray-400"}">${escapeHtml(L)}\
</div>`).join("")}</div>`}).join("")}</div>`:""}o(si,"feedbackRepliesHtml");function ai(){return`<di\
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
                </div>`}o(ai,"feedbackReplyFormHtml");function oi(d,m){const g=m.querySelector(".fb-\
reply-form");if(!g)return;const h=g.querySelector(".fb-reply-logs"),x=g.querySelector(".fb-reply-log\
s-note"),_=!!(window.ActivityLog&&window.ActivityLog.canAttach());h.disabled=!_,x.textContent=_?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const L=ei(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));Pn.push(L);const M=g.querySelector(".fb-reply-send");M.onclick=async()=>{const E=g.
querySelector(".fb-reply-message").value.trim();if(!E){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O={message:E,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},N=h.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;N&&(O.client_logs=N);const X=L.
files();M.disabled=!0;try{const oe=await apiFetch(`/api/feedback/${d.public_id}/replies`,{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(O)}),ie=await oe.json().catch(()=>({}));
if(!oe.ok){showToast(ie.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const Y=!X.length||await ti(`/api/feedback/${d.public_id}/replies/${ie.reply_id}/\
images`,X),fe=[];if(N&&ie.logs_saved===!1&&fe.push("\u30ED\u30B0"),Y||fe.push("\u753B\u50CF"),fe.length)
showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${fe.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\uFF09`,"error",!0);else{const D=["\u8FD4\u4FE1"];N&&D.push(`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof ie.
logs_count=="number"?ie.logs_count:N.entries.length}\u4EF6\uFF09`),X.length&&D.push(`\u753B\u50CF\uFF08${X.
length}\u679A\uFF09`),showToast(D.join(D.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F",
"success")}on()}finally{M.disabled=!1}}}o(oi,"bindFeedbackReply");function ri(){const d=get("fb-admi\
n-status-filter").value;let m=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(
g=>{const h=d==="all"||(d==="unresolved"?["new","in_review","replied"].includes(g.dataset.feedbackStatus):
g.dataset.feedbackStatus===d);g.hidden=!h,h&&m++}),get("fb-admin-empty").classList.toggle("hidden",m!==
0)}o(ri,"filterAdminFeedback"),get("fb-admin-status-filter").onchange=ri;async function on(){Pn.forEach(
_=>_.clear()),Pn=[];const m=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML=
"",(m.items||[]).filter(_=>!m.is_admin||_.user_id===void 0||_.user_id===null||!0).forEach(_=>{if(m.is_admin)
return;const L=document.createElement("div");L.className="p-2 rounded border border-gray-700 bg-gray\
-800/50",L.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
_.public_id||"")}</span> / ${_.created_at}</div><div class="font-bold text-sm">${escapeHtml(_.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(_.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(_.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":_.status)}</div>${_.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${_.
image_count}\u679A</div>`:""}${si(_,!1)}${ai()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
L.querySelector(".fb-delete").onclick=()=>ii(_.public_id),oi(_,L),g.appendChild(L)});const h=get("fb\
-admin-panel"),x=get("fb-admin-list");m.is_admin?(h.classList.remove("hidden"),x.innerHTML="",(m.items||
[]).forEach(_=>{const L=document.createElement("div");L.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",L.dataset.feedbackStatus=_.status||"new",L.innerHTML=`
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
                            ${si(_,!0)}
                            ${ai()}
                        `,L.querySelector(".fb-status").value=_.status||"new",L.querySelector(".fb-s\
ave").onclick=async()=>{const M=L.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${_.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:M})}),
on()},L.querySelector(".fb-delete").onclick=()=>ii(_.public_id),oi(_,L),x.appendChild(L)}),ri()):h.classList.
add("hidden")}if(o(on,"loadFeedback"),window.setupTOTP=async()=>{const m=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=m.qr_image,get("totp-secret-disp").innerText=
m.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const d=get(
"totp-verify-code").value;if(!d)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),m=get("webauthn-nam\
e"),g=m?String(m.value||"").trim():"";try{d&&(d.disabled=!0);const h=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),x=await h.json();if(!h.ok){showToast(x.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const L=await(await ensureWebAuthnJson()).create({publicKey:x}),M=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},L,{name:g}))}),E=await M.json().catch(()=>({}));M.ok?(m&&(m.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(E.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(h){showToast(`WebAuthn Error: ${h}`,"error",!0)}finally{d&&(d.disabled=!1)}},window.
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
return}if(d.key==="Enter"){d.preventDefault();const h=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(m.value),m.value));h[slashSelectedIndex]?selectSlashCommand(h[slashSelectedIndex].
id):h.length>0&&selectSlashCommand(h[0].id);return}if(d.key==="Escape"){d.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const g=m.value.trim();if(d.key==="ArrowDown"){d.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(g.substring(1));return}if(d.key==="ArrowUp"){
d.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(g.substring(1));
return}if(d.key==="Enter"){d.preventDefault();const h=g.substring(1).toLowerCase(),x=loadedGems.filter(
_=>_.name.toLowerCase().includes(h)||_.description&&_.description.toLowerCase().includes(h));x[gemSelectedIndex]?
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
"sts-cancel-btn").onclick=()=>{isStsModel()&&$i()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async d=>{const m=(d.clipboardData||window.clipboardData).items,g=[];for(let h=0;h<m.length;h++)
if(m[h].kind==="file"){const x=m[h].getAsFile();x&&g.push(x)}g.length>0&&(d.preventDefault(),await handleFiles(
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
"";m&&g&&(d.preventDefault(),openImageViewer(g))});const rn=document.querySelector(".viewer-content");
rn&&(rn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),rn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),rn.addEventListener("touchend",onViewerTouchEnd),rn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",d=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(d.target.id==="image-viewer"||d.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",d=>{d.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",d=>{d.key==="Escape"&&closeImageViewer()});
let st,Oe=null,vn=[],On=!1,wn=null,Vt=null,Nt=null,ln=null,Nn=0,xn=!1,cn=null,Jt=null,Ct=null,dn=null,
kn=null,Rt=null,_n=null,Sn=null;function li(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
_n)&&_n.length)return _n;d.innerHTML="";const m=[];for(let g=0;g<24;g++){const h=document.createElement(
"span");h.className="block rounded-full",h.style.background="rgba(252, 165, 165, 0.92)",h.style.width=
"2px",h.style.transition="height 75ms linear, opacity 75ms linear",h.style.height="2px",h.style.opacity=
"0.4",m.push(h),d.appendChild(h)}return _n=m,m}o(li,"ensureMicWaveformBars");function Bt(d,m="hidden"){
const g=get("mic-recording-indicator"),h=get("mic-recording-text");if(g){if(Sn&&(clearTimeout(Sn),Sn=
null),m==="hidden"){g.classList.add("hidden");return}h&&d&&(h.innerText=d),g.classList.remove("hidde\
n"),m==="recording"?g.style.color="rgb(252 165 165)":m==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(Bt,"setMicRecordingIndicator");function ci(){li().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}o(ci,"resetMicWaveformBars");function Ft(){if(kn&&(cancelAnimationFrame(
kn),kn=null),dn){try{dn.disconnect()}catch{}dn=null}if(Jt){try{Jt.close()}catch{}Jt=null}Ct=null,Rt=
null,ci()}o(Ft,"stopMicWaveform");function di(d){Ft();const m=li();if(!m.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{Jt=new g,Ct=Jt.createAnalyser(),Ct.fftSize=
256,Ct.smoothingTimeConstant=0,dn=Jt.createMediaStreamSource(d),dn.connect(Ct),Rt=new Uint8Array(Ct.
frequencyBinCount)}catch{Ft();return}const h=o(()=>{if(!Ct||!Rt)return;Ct.getByteFrequencyData(Rt);const x=Math.
max(1,Math.floor(Rt.length/m.length));for(let _=0;_<m.length;_++){const M=(Rt[Math.min(Rt.length-1,_*
x)]||0)/255,E=Math.max(2,Math.round(2+M*10));m[_].style.height=`${E}px`,m[_].style.opacity=`${.35+M*
.65}`}kn=requestAnimationFrame(h)},"render");h()}o(di,"startMicWaveform");function Ln(){if(wn&&(clearInterval(
wn),wn=null),ln){try{ln.disconnect()}catch{}ln=null}if(Vt){try{Vt.close()}catch{}Vt=null}Nt=null}o(Ln,
"stopSilenceMonitor");function ui(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;Ln();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;Vt=new m,Nt=Vt.createAnalyser(),Nt.fftSize=2048,
ln=Vt.createMediaStreamSource(d),ln.connect(Nt);const g=new Uint8Array(Nt.fftSize),h=getStsSilenceMs(),
x=.02;Nn=0,xn=!1,wn=setInterval(()=>{if(!Nt)return;Nt.getByteTimeDomainData(g);let _=0;for(let M=0;M<
g.length;M++){const E=(g[M]-128)/128;_+=E*E}if(Math.sqrt(_/g.length)>x){xn||(xn=!0),Nn=Date.now();return}
xn&&Date.now()-Nn>h&&st&&st.state==="recording"&&st.stop()},200)}o(ui,"startSilenceMonitor");const Fn=class Fn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(m,g,h,x={},_=null){if(this.
model=h,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(m)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const E=!!(x&&
x.transcriptionConfig),O={setup:{model:`models/${h}`,generationConfig:{responseModalities:E?["TEXT"]:
["AUDIO"]},inputAudioTranscription:E?x.transcriptionConfig||{}:{}}};E||(O.setup.outputAudioTranscription=
{}),x.speechConfig&&(O.setup.generationConfig.speechConfig=x.speechConfig),x.thinkingConfig&&(O.setup.
generationConfig.thinkingConfig=x.thinkingConfig),x.translationConfig&&(O.setup.generationConfig.translationConfig=
x.translationConfig),console.log("Sending setup:",JSON.stringify(O)),this.ws.send(JSON.stringify(O))},
this.ws.onmessage=E=>this._handleMessage(E),this.ws.onerror=E=>{console.error("Gemini Live WebSocket\
 error:",E),this.onError&&this.onError(E)},this.ws.onclose=E=>{console.log("Gemini Live WebSocket cl\
osed:",E.code,E.reason),this.closedEvent=E,this.onClose&&this.onClose(E)},_&&_.stream&&_.ctx&&_.source){
this.stream=_.stream,this.audioContext=_.ctx;var L=_.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(Cn());const E=mi(this.stream,16e3);this.audioContext=E.ctx;var L=E.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const M=new MediaRecorder(this.stream);M.ondataavailable=E=>{E.data.
size>0&&this.userAudioChunks.push(E.data)},M.start(500),this.backupRecorder=M,this.processor.onaudioprocess=
E=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const O=pi(E.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!O||!O.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(O))),mimeType:"audio/\
pcm;rate=16000"}}}))},L.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
let g=null;try{g=JSON.parse(typeof m.data=="string"?m.data:new TextDecoder().decode(m.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const h=g.serverContent;h.modelTurn&&h.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log(
"Gemini thought delta:",x.text),this.assistantThought+=x.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(x.text)),x.inlineData&&x.inlineData.data){const _=x.inlineData.data;
console.log("Gemini audio chunk received, size:",_.length),this.rtPlayer&&this.rtPlayer.addChunk(_);
const L=atob(_),M=new Uint8Array(L.length);for(let E=0;E<L.length;E++)M[E]=L.charCodeAt(E);this.assistantAudioChunks.
push(M)}}),h.outputTranscription&&h.outputTranscription.text&&this._appendAssistantText(h.outputTranscription.
text),h.inputTranscription&&h.inputTranscription.text&&(this.userTurnBreak&&this.inputTranscript&&!this.
inputTranscript.endsWith(`
`)&&(this.inputTranscript+=`
`),this.userTurnBreak=!1,this.inputTranscript+=h.inputTranscription.text,this.interimInputTranscript=
""),h.interimInputTranscription&&(this.interimInputTranscript=h.interimInputTranscription.text||""),
h.turnComplete&&(this.assistantTurnBreak=!0,this.model!=="gemini-3.5-transcribe-live"&&(this.userTurnBreak=
!0))}g.error&&this.onError&&this.onError(g.error),this.onMessage&&this.onMessage(g)}_appendAssistantText(m){
m&&(this.assistantTurnBreak&&this.assistantText&&!this.assistantText.endsWith(`
`)&&(this.assistantText+=`
`),this.assistantTurnBreak=!1,this.assistantText+=m)}stop(){const m=this.backupRecorder;if(this.backupRecorder=
null,this._stopped=new Promise(h=>{let x=!1;const _=o(()=>{x||(x=!0,h())},"finish");if(!m||m.state===
"inactive"){_();return}m.addEventListener("stop",_,{once:!0});try{m.stop()}catch{_()}setTimeout(_,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=o(
()=>{if(this.stream&&this.stream.getTracks().forEach(h=>{try{h.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const m=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(m),h=new Blob(
this.userAudioChunks),x=await this._blobToBase64(h);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:x}}_blobToBase64(m){
return new Promise(g=>{const h=new FileReader;h.onloadend=()=>g(h.result.split(",")[1]),h.readAsDataURL(
m)})}};o(Fn,"GeminiLiveClient");let Rn=Fn;const jn=class jn{constructor(m=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:m}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(m){if(!this.ctx)return;const g=atob(m),h=new Uint8Array(g.length);for(let O=0;O<
g.length;O++)h[O]=g.charCodeAt(O);const x=new Int16Array(h.buffer),_=new Float32Array(x.length);for(let O=0;O<
x.length;O++)_[O]=x[O]/32768;const L=this.ctx.createBuffer(1,_.length,this.ctx.sampleRate);L.getChannelData(
0).set(_),this.ctx.state==="suspended"&&await this.ctx.resume();const M=this.ctx.createBufferSource();
M.buffer=L,M.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const E=Math.max(this.ctx.currentTime,this.nextStartTime);M.start(E),this.
nextStartTime=E+L.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(jn,"RealTimeAudioPl\
ayer");let un=jn;const Dn=class Dn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=null,this.rtPlayer=
null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript="",this.assistantThought=
"",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.saved=
!1,this.saving=!1,this.stopping=!1,this.audioQueue=[],this.audioFlush=null}isActive(){return this.active}async start(){
if(this.active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const m=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const x=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=x.id!==null&&x.id!==void 0?String(x.id):x.id,setTemporaryChatUiState(!!(x&&x.
is_temporary)),setCurrentChatHeaderTitle(x&&x.title),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+x.id),get("welcome-screen").classList.add("hidden")}catch(h){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+h.message,"\
error",!0);return}const g={model:m,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||m==="gpt-realtime-translate")&&get("sts-targ\
et-lang")?get("sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(g.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const h=await apiFetch(
"/api/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
g)}),x=await h.json().catch(()=>({}));if(!h.ok)throw new Error(x.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=x.session_id,this.rateIn=x.rate_in||this.rateIn,this.rateOut=x.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null,this.audioQueue=[],this.audioFlush=null}catch(h){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",
!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
h.message,"error",!0);return}this.abortCtrl=new AbortController,this._openStream();try{await this._startCapture()}catch(h){
setStsStatus("\u30DE\u30A4\u30AF\u30A8\u30E9\u30FC",!1),showToast("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093: "+
h.message,"error",!0),this._cancel();return}get("mic-btn").classList.remove("bg-gray-700"),get("mic-\
btn").classList.add("bg-red-600","animate-pulse"),setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0)}_openStream(){const m="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),g=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(m,g).then(h=>{if(!h.ok)throw new Error("SSE stream failed ("+h.status+")");this.
reader=h.body.getReader(),this._readLoop()}).catch(h=>{h&&h.name==="AbortError"||(this.streamError=h&&
h.message?h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const m=new TextDecoder;let g="";try{for(;this.reader;){const{done:h,
value:x}=await this.reader.read();if(h)break;g+=m.decode(x,{stream:!0});let _;for(;(_=g.indexOf(`

`))>=0;){const L=g.slice(0,_);g=g.slice(_+2);for(const M of L.split(`
`)){if(!M.startsWith("data: "))continue;let E=null;try{E=JSON.parse(M.slice(6))}catch{continue}this.
_handleEvent(E)}}}}catch(h){if(h&&h.name==="AbortError")return;this.active&&(this.streamError=h&&h.message?
h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new un(this.rateOut||24e3),Kt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Kt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(Cn()).then(m=>{this.stream=m;const g=this.rateIn||24e3,h=mi(m,g);this.audioCtx=
h.ctx;const x=h.source,_=this.audioCtx.sampleRate,L=4096;this.processor=this.audioCtx.createScriptProcessor(
L,1,1),this.processor.onaudioprocess=M=>{if(!this.active||!this.capturing)return;const E=M.inputBuffer.
getChannelData(0),O=pi(E,_,g);!O||!O.byteLength||this._sendAudio(O)},x.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(m){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(m)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const m=this.audioQueue.splice(0,this.audioQueue.length),
g=m.reduce((E,O)=>E+O.byteLength,0),h=new Uint8Array(g);let x=0;m.forEach(E=>{h.set(E,x),x+=E.byteLength});
const _="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),L={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:h.buffer},
M=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(L):L;try{await fetch(_,M)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}Ln(),Ft()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const m=Date.now(),g=this.responseDoneCount;let h=this.lastAudioAt;for(;Date.now()-m<2e4&&!(this.responseDoneCount>
g||(this.lastAudioAt>h&&(h=this.lastAudioAt),!this.speechActive&&Date.now()-m>2e3&&Date.now()-h>2500));)
await new Promise(x=>setTimeout(x,250));await this._save()}async _save(){if(!this.saved){this.saved=
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
Dn,"RealtimeVoiceSession");let Bn=Dn;function mi(d,m){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let h=null;try{return h=new g({sampleRate:m}),{ctx:h,
source:h.createMediaStreamSource(d)}}catch{if(h)try{h.close()}catch{}return h=new g,{ctx:h,source:h.
createMediaStreamSource(d)}}}o(mi,"openMicAudioSource");function pi(d,m,g){let h=d;if(m!==g&&m>0&&g>
0){const _=m/g,L=Math.floor(h.length/_),M=new Float32Array(L);for(let E=0;E<L;E++)M[E]=h[Math.min(Math.
floor(E*_),h.length-1)];h=M}const x=new Int16Array(h.length);for(let _=0;_<h.length;_++){const L=Math.
max(-1,Math.min(1,h[_]));x[_]=L<0?L*32768:L*32767}return x.buffer}o(pi,"pcm16FromFloat32");const mn=new Bn,
fi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let h=null,x=null,_=!1,L=null,M=!1,E=0,O=0,N=null,X="idle",oe=!1,ie=null;const Y=o(F=>document.getElementById(
F),"$"),fe=o(F=>{const Q=Object.assign({},F||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(Q):(Q.progressSpinner=!1,
Q)},"noSpinner");function D(F,Q){X=Q;const Ce=Y("lyria-status-text"),ue=Y("lyria-status-dot");Ce&&(Ce.
textContent=F),ue&&(ue.className="w-2 h-2 rounded-full inline-block "+(g[Q]||g.idle)),ye(),ee()}o(D,
"setStatus");function ge(){const F=O?Math.floor((Date.now()-O)/1e3):0,Q=String(Math.floor(F/60)).padStart(
2,"0"),Ce=String(F%60).padStart(2,"0");return`${Q}:${Ce}`}o(ge,"formatElapsed");function W(){O||(O=Date.
now());const F=Y("lyria-elapsed");F&&(F.textContent=ge()),N||(N=window.setInterval(()=>{const Q=Y("l\
yria-elapsed");Q&&(Q.textContent=ge())},1e3))}o(W,"startElapsedTimer");function me(){N&&(window.clearInterval(
N),N=null)}o(me,"stopElapsedTimer");function ye(){const F=Y("lyria-play-btn"),Q=Y("lyria-pause-btn"),
Ce=Y("lyria-stop-btn"),ue=Y("lyria-reset-btn"),Me=!!h,je=X==="streaming"||X==="connecting";if(F){F.disabled=
oe;const Qe=F.querySelector("i");Qe&&(Qe.className="fas fa-play")}Q&&(Q.disabled=oe||!je),Ce&&(Ce.disabled=
oe||!Me||!je),ue&&(ue.disabled=oe||!Me||!je)}o(ye,"updateTransportButtons");function ee(){const F=Y(
"lyria-save-btn");if(!F)return;const Q=!!h&&X!=="idle"&&X!=="connecting"&&X!=="error";F.classList.toggle(
"hidden",!Q)}o(ee,"updateSaveButton");function he(F,Q){const Ce=Y("lyria-prompt-rows");if(!Ce)return;
const ue=document.createElement("div");ue.className="flex items-center gap-2",ue.innerHTML=`
                        <input type="text" value="${escapeHtml(F||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof Q=="num\
ber"?Q:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof Q=="number"?Q:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Me=ue.querySelector('input[type="range"]'),je=ue.querySelector(".lyria-w\
eight-label");Me&&je&&Me.addEventListener("input",()=>{je.textContent=parseFloat(Me.value).toFixed(1)});
const Qe=ue.querySelector(".lyria-prompt-remove");Qe&&Qe.addEventListener("click",()=>{Ce.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ue.remove()}),ue.classList.add("lyria-prompt-row-wrap"),Ce.appendChild(
ue)}o(he,"addPromptRow");function Ee(){const F=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),Q=[];return F.forEach(Ce=>{const ue=Ce.querySelector('input[type="text"]'),Me=Ce.querySelector(
'input[type="range"]'),je=(ue?ue.value:"").trim();je&&Q.push({text:je,weight:parseFloat(Me?Me.value:
1)||1})}),Q}o(Ee,"collectPrompts");function $e(){const F={},Q=o(Ni=>{const qn=Y(Ni);return qn&&qn.value!==
""?parseFloat(qn.value):void 0},"num"),Ce=Q("lyria-bpm");Ce!==void 0&&(F.bpm=Math.round(Ce));const ue=Q(
"lyria-guidance");ue!==void 0&&(F.guidance=ue);const Me=Q("lyria-density");Me!==void 0&&(F.density=Me);
const je=Q("lyria-brightness");je!==void 0&&(F.brightness=je);const Qe=Q("lyria-temperature");Qe!==void 0&&
(F.temperature=Qe);const qe=Y("lyria-scale");qe&&qe.value&&(F.scale=qe.value);const tt=Y("lyria-mode");
tt&&tt.value&&(F.music_generation_mode=tt.value);const mt=Y("lyria-mute-bass"),jt=Y("lyria-mute-drum\
s"),wi=Y("lyria-only-bass-drums");return mt&&(F.mute_bass=mt.checked),jt&&(F.mute_drums=jt.checked),
wi&&(F.only_bass_and_drums=wi.checked),F}o($e,"collectConfig");function at(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([Q,Ce])=>{
const ue=Y(Q),Me=Y(Ce);!ue||!Me||ue.addEventListener("input",()=>{const je=parseFloat(ue.value);Me.textContent=
Q==="lyria-bpm"?String(Math.round(je)):je.toFixed(1)})})}o(at,"bindRangeLabels");function Ue(){if(L){
try{L.close()}catch{}L=null}M=!1,E=0}o(Ue,"resetPlayback");function ut(){if(_=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(ut,"closeStream");async function Tt(){ut(),x=new AbortController,
_=!0;try{const F=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(h)}`,fe({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!F.ok){const Me=await F.
json().catch(()=>({}));throw new Error(Me.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const Q=F.body.getReader(),Ce=new TextDecoder;let ue="";for(;_;){const{done:Me,value:je}=await Q.read();
if(Me)break;ue+=Ce.decode(je,{stream:!0});const Qe=ue.split(`

`);ue=Qe.pop();for(const qe of Qe){const tt=qe.split(`
`).find(jt=>jt.startsWith("data: "));if(!tt)continue;const mt=tt.slice(6);try{const jt=JSON.parse(mt);
Le(jt)}catch{}}}}catch(F){if(F&&F.name==="AbortError")return;_&&(D("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{_&&h&&Tt()},1200))}finally{_=!1}}o(Tt,"openStream");function Le(F){
if(F&&F.snapshot){const Q=F.status;if(Q==="error"){D("\u30A8\u30E9\u30FC","error"),me();return}if(Q===
"closed"||Q==="stopped"){D("\u7D42\u4E86","closed"),me();return}D(Q==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",Q==="paused"?"paused":"connecting");return}if(F&&F.audio){D("\u518D\u751F\u4E2D...",
"streaming"),W(),ve(F.audio);return}if(F&&F.error){D("\u30A8\u30E9\u30FC: "+F.error,"error"),me();return}
if(F&&F.final){D("\u7D42\u4E86","closed"),me(),ye();return}}o(Le,"handleStreamMessage");function ve(F){
if(!F)return;if(!L){const qe=window.AudioContext||window.webkitAudioContext;if(!qe)return;L=new qe({
sampleRate:48e3}),M=!1,E=0}let Q;try{const qe=atob(F);Q=new Uint8Array(qe.length);for(let tt=0;tt<qe.
length;tt++)Q[tt]=qe.charCodeAt(tt)}catch{return}const Ce=new Int16Array(Q.buffer),ue=Math.floor(Ce.
length/2);if(ue<1)return;const Me=L.createBuffer(2,ue,48e3);for(let qe=0;qe<2;qe++){const tt=Me.getChannelData(
qe);for(let mt=0;mt<ue;mt++)tt[mt]=Ce[mt*2+qe]/32768}L.state==="suspended"&&L.resume();const je=L.createBufferSource();
je.buffer=Me,je.connect(L.destination),M||(E=L.currentTime+.08,M=!0);const Qe=Math.max(L.currentTime,
E);je.start(Qe),E=Qe+Me.duration}o(ve,"playChunk");async function Fe(F,Q){const Ce=await fetch("/api\
/gemini/music/command",fe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:h,type:F},Q||{}))})),ue=await Ce.json().catch(()=>({}));if(!Ce.ok)throw new Error(
ue.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ue}
o(Fe,"apiCommand");async function Ye(){if(oe)return;const F=Ee();if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}oe=!0,ye(),D("\u63A5\u7D9A\u4E2D...","connecting");try{
const Q=await fetch("/api/gemini/music/start",fe({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:F,config:$e()})})),Ce=await Q.json().catch(()=>({}));if(!Q.
ok)throw new Error(Ce.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
h=Ce.session_id,ie=$e(),D("\u63A5\u7D9A\u4E2D...","connecting"),Tt()}catch(Q){D("\u30A8\u30E9\u30FC: "+
Q.message,"error"),showToast("Lyria RealTime: "+Q.message,"error",!0)}finally{oe=!1,ye()}}o(Ye,"star\
tSession");async function ot(F){if(h){oe=!0,ye();try{await Fe("control",{action:F}),F==="PLAY"?D("\u518D\u751F\
\u4E2D...","streaming"):F==="PAUSE"?D("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):F==="STOP"?D("\u505C\u6B62\u4E2D",
"stopped"):F==="RESET_CONTEXT"&&D("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(Q){showToast("Lyria RealTime: "+Q.message,"error",!0),D("\u30A8\u30E9\u30FC: "+Q.
message,"error")}finally{oe=!1,ye()}}}o(ot,"control");async function Se(){if(!h)return;const F=Ee();
if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}oe=!0;try{await Fe("prompts",{weighted_prompts:F}),D("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
X==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(Q){showToast("Lyria RealTime: "+Q.message,"error",!0)}finally{oe=!1,ye()}}o(Se,"app\
lyPrompts");async function Pe(){if(!h)return;const F=$e(),Q=ie||{},Ce=F.bpm!==void 0&&F.bpm!==Q.bpm,
ue=F.scale!==void 0&&F.scale!==Q.scale,Me=Ce||ue;oe=!0;try{await Fe("config",{config:F,reset_context:Me}),
ie=F,D(Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",X==="paused"?"paused":"streaming"),showToast(
Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(je){showToast("Lyria RealT\
ime: "+je.message,"error",!0)}finally{oe=!1,ye()}}o(Pe,"applyConfig");async function De(){if(h){oe=!0,
D("\u4FDD\u5B58\u4E2D...","connecting"),ye();try{const F=await fetch("/api/gemini/music/save",fe({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h,thread_id:currentThreadId||
null})})),Q=await F.json().catch(()=>({}));if(!F.ok)throw new Error(Q.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
D("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),me(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),Q.thread_id&&(currentThreadId=String(Q.thread_id),history.pushState({},"","/c/"+Q.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(Q.thread_id||currentThreadId),fn(!0)}catch(F){
D("\u30A8\u30E9\u30FC: "+F.message,"error"),showToast("Lyria RealTime: "+F.message,"error",!0)}finally{
oe=!1,ye()}}}o(De,"saveSession");async function ze(){if(ut(),h)try{await fetch("/api/gemini/music/ca\
ncel",fe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h})}))}catch{}
h=null,me(),Ue(),D("\u6E96\u5099\u5B8C\u4E86","idle")}o(ze,"cancelSession");function ct(){const F=Y(
"lyria-prompt-rows");F&&(F.innerHTML=""),he("",1),ie=null,O=0,["lyria-bpm","lyria-guidance","lyria-d\
ensity","lyria-brightness","lyria-temperature"].forEach(ue=>{const Me=Y(ue);Me&&(Me.value=ue==="lyri\
a-bpm"?"120":ue==="lyria-guidance"?"4":ue==="lyria-temperature"?"1.1":"0.5")});const Q=Y("lyria-scal\
e");Q&&(Q.value="");const Ce=Y("lyria-mode");Ce&&(Ce.value="QUALITY"),["lyria-mute-bass","lyria-mute\
-drums","lyria-only-bass-drums"].forEach(ue=>{const Me=Y(ue);Me&&(Me.checked=!1)}),at()}o(ct,"resetC\
ontrols");function fn(F){ut(),h&&fetch("/api/gemini/music/cancel",fe({method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({session_id:h})})).catch(()=>{}),h=null,_=!1,me(),Ue(),
hideModal("lyria-studio-modal")}o(fn,"closeAndCleanup");function Xt(F){if(!isLyriaRealtimeModel()){showToast(
"Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const Q=Y("lyria-studio-modal");if(Q&&Q.classList.contains("modal-open")&&h){if(F&&
typeof F=="string"){const ue=Y("lyria-prompt-rows");ue&&(ue.innerHTML=""),he(F,1)}return}if(h&&ze(),
ct(),F&&typeof F=="string"){const ue=Y("lyria-prompt-rows");ue&&(ue.innerHTML=""),he(F,1)}h=null,_=!1,
me(),Ue(),D("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Xt,"open");function gn(){
const F=Y("lyria-open-studio-btn");F&&F.addEventListener("click",()=>Xt(""));const Q=Y("lyria-studio\
-close");Q&&Q.addEventListener("click",()=>fn(!1));const Ce=Y("lyria-play-btn");Ce&&Ce.addEventListener(
"click",()=>{if(!h){Ye();return}ot("PLAY")});const ue=Y("lyria-pause-btn");ue&&ue.addEventListener("\
click",()=>ot("PAUSE"));const Me=Y("lyria-stop-btn");Me&&Me.addEventListener("click",()=>ot("STOP"));
const je=Y("lyria-reset-btn");je&&je.addEventListener("click",()=>ot("RESET_CONTEXT"));const Qe=Y("l\
yria-add-prompt-btn");Qe&&Qe.addEventListener("click",()=>he("",1));const qe=Y("lyria-apply-prompts-\
btn");qe&&qe.addEventListener("click",Se);const tt=Y("lyria-apply-config-btn");tt&&tt.addEventListener(
"click",Pe);const mt=Y("lyria-save-btn");mt&&mt.addEventListener("click",De),at(),ct(),window.openLyriaStudio=
Xt}o(gn,"init");function Hn(){oe||fn(!1)}return o(Hn,"requestClose"),{init:gn,open:Xt,requestClose:Hn}})();
fi.init(),window.closeLyriaStudio=()=>fi.requestClose(),(()=>{let d=null,m=null,g=null;const h="voic\
eDockSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
_=o(ee=>document.getElementById(ee),"$");function L(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(L,"isStudioMode");function M(){const ee=get("model-select")?get("model-select").value:"",he=_("voi\
ce-studio-title");he&&(ee==="gpt-transcribe"||ee==="gpt-live-transcribe"||ee==="grok-voice-transcrib\
e-2.0-file"?he.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ee===
"gemini-3.5-live-translate-preview"?he.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
he.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(M,"updateTitle");function E(){const ee=_("v\
oice-studio-transcript");ee&&(ee.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${x}</div>`);const he=_("sts-live-transcript");he&&(he.innerHTML="",he.classList.add("hidden"))}
o(E,"resetTranscript");function O(ee,he,Ee){const $e=ee.querySelectorAll(".voice-studio-line");let at=null;
for(let Ue=$e.length-1;Ue>=0;Ue--)if($e[Ue].dataset.role===he){at=$e[Ue];break}if(at)at.innerHTML=Ee;else{
const Ue=ee.querySelector(".voice-studio-placeholder");Ue&&Ue.remove();const ut=document.createElement(
"div");ut.className="voice-studio-line",ut.dataset.role=he,ut.innerHTML=Ee,ee.appendChild(ut)}ee.classList.
remove("hidden"),ee.scrollTop=ee.scrollHeight}o(O,"writeLine");function N(ee,he){if(!he||!String(he).
trim()||!L())return;const at=`<span class="${ee==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ee==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
he)}</span>`;[_("voice-studio-transcript"),_("sts-live-transcript")].filter(Boolean).forEach(Ue=>O(Ue,
ee,at))}o(N,"log");function X(ee,he=!0){const Ee=_("sts-panel"),$e=_("sts-settings-toggle");if(Ee&&Ee.
classList.toggle("settings-open",!!ee),$e&&$e.setAttribute("aria-expanded",ee?"true":"false"),he)try{
localStorage.setItem(h,ee?"1":"0")}catch{}}o(X,"setSettingsOpen");function oe(){try{return localStorage.
getItem(h)==="1"}catch{return!1}}o(oe,"readSettingsOpen");let ie=null;function Y(){const ee=get("mod\
el-select")?get("model-select").value:"";ee!==ie&&(ie=ee,E()),X(oe(),!1),M()}o(Y,"syncDock");function fe(){
const ee=_("sts-panel"),he=_("voice-studio-panel-host");ee&&he&&ee.parentNode!==he&&(d=ee.parentNode,
m=ee.nextSibling,he.appendChild(ee));const Ee=_("file-preview"),$e=_("voice-studio-file-host");Ee&&$e&&
Ee.parentNode!==$e&&(g=Ee.parentNode,$e.appendChild(Ee),$e.classList.remove("hidden"))}o(fe,"movePan\
elIntoModal");function D(){const ee=_("sts-panel");ee&&d&&ee.parentNode!==d&&(m&&m.parentNode===d?d.
insertBefore(ee,m):d.appendChild(ee));const he=_("file-preview");he&&g&&he.parentNode!==g&&g.appendChild(
he);const Ee=_("voice-studio-file-host");Ee&&Ee.classList.add("hidden"),d=null,m=null,g=null}o(D,"mo\
vePanelBack");function ge(){if(!L())return;fe();const ee=_("sts-panel");ee&&ee.classList.remove("hid\
den"),M(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(ge,"open");function W(){window.
VoiceStudioOpen=!1,D(),hideModal("voice-studio-modal")}o(W,"close");function me(){window.VoiceStudioOpen&&
W()}o(me,"closeIfOpen");function ye(){window.VoiceStudioOpen=!1;const ee=_("voice-studio-open-btn");
ee&&ee.addEventListener("click",()=>ge());const he=_("voice-studio-close");he&&he.addEventListener("\
click",()=>W());const Ee=_("sts-settings-toggle");Ee&&Ee.addEventListener("click",()=>{const $e=_("s\
ts-panel");X(!($e&&$e.classList.contains("settings-open")))}),window.VoiceStudio={open:ge,close:W,closeIfOpen:me,
log:N,isStudioMode:L,syncDock:Y},Y()}return o(ye,"init"),{init:ye}})().init();let Kt=null;function gi(){
if(Kt&&(Kt.stop(),Kt=null),cn){try{cn.pause()}catch{}try{cn.src=""}catch{}cn=null}}o(gi,"stopStsPlay\
back");async function Fi(d){gi();const m=new Audio;return m.src=d,m.preload="auto",m.autoplay=!0,m.playsInline=
!0,cn=m,await m.play(),new Promise(g=>{m.onended=()=>g("ended"),m.onerror=()=>g("error")})}o(Fi,"pla\
yStsAudio");function $i(){if(mn.isActive()){mn._cancel();return}if(Oe){Oe.stop(),Oe=null,gi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Ft();return}st&&st.state===
"recording"&&(On=!0,st.stop())}o($i,"cancelRecording");function Cn(){if(isStsModel())return{audio:!0};
const m=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return m.echoCancellation&&(g.echoCancellation=!1),m.
noiseSuppression&&(g.noiseSuppression=!1),m.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(Cn,"\
getMicCaptureConstraints");function Ii(){const d=window.AudioContext||window.webkitAudioContext;if(!d||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let m=null;try{m=new d({sampleRate:16e3})}catch{if(m){try{m.close()}catch{}m=null}m=new d}const g=m.
resume(),h=navigator.mediaDevices.getUserMedia(Cn());return{ctx:m,resumePromise:g,streamPromise:h}}o(
Ii,"beginGeminiLiveMic");function Pi(d){if(d){if(d.streamPromise)d.streamPromise.then(m=>{try{m.getTracks().
forEach(g=>g.stop())}catch{}}).catch(()=>{});else if(d.stream)try{d.stream.getTracks().forEach(m=>m.
stop())}catch{}if(d.ctx)try{d.ctx.close()}catch{}}}o(Pi,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Oe){setStsStatus("Processing...",!0);const d=Oe;Oe=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const m=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(m.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
m.assistant_text=(d.inputTranscript||"").trim(),m.assistant_thought="",!m.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const g=(m.user_text||
"").trim()||(m.assistant_text||"").trim()||(m.assistant_thought||"").trim(),h=!!(m.audio_base64||m.user_audio_base64);
if(!g&&!h){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const _=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(_.id),history.pushState(
{},"","/c/"+_.id),get("welcome-screen").classList.add("hidden")}m.thread_id=currentThreadId,m.model=
get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error("Failed to save G\
emini Live session:",m),setStsStatus("Error saving",!1)}return}if(mn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),mn.stop();return}if(st&&
st.state==="recording"){st.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Bt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ii();
const x={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(x.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const D=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(ge=>ge.trim()).filter(Boolean);D.length&&(x.custom_vocabulary=
D.slice(0,1e3))}}else x.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(x.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",x.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(x.target_lang=get("sts-target-lang").value);const _=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:_,ctx:g.ctx,source:g.ctx.createMediaStreamSource(_)};const L=await apiFetch(
"/api/gemini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
x)});if(!L.ok)throw new Error("Failed to get session token");const{token:M,url:E}=await L.json(),O=get(
"model-select").value,N=get("sts-voice")?get("sts-voice").value:"Kore",X=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",oe=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(Oe=new Rn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Oe.rtPlayer=new un),isGeminiLiveTranscribeModel()){
const D=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",ge={languageCodes:[]};
if((D==="SMART"||D==="VERBATIM")&&(ge.mode=D),get("sts-custom-vocab")){const W=get("sts-custom-vocab").
value.split(/[,、\n]/).map(me=>me.trim()).filter(Boolean);W.length&&(ge.customVocabulary=W.slice(0,
1e3))}await Oe.start(M,E,O,{transcriptionConfig:ge},g)}else if(isGeminiLiveTranslateModel()){const D=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Oe.start(M,E,O,{translationConfig:{targetLanguageCode:D,
echoTargetLanguage:!0}},g)}else{const D={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:N}}}};
isGeminiLiveExtendedThinkingModel()&&(D.thinkingConfig={thinkingLevel:X,includeThoughts:oe}),await Oe.
start(M,E,O,D,g)}g=null,st=Oe.backupRecorder,st.onstop=()=>{Oe&&get("mic-btn").click()};let ie=!0,Y="\
live-sts-"+Date.now();Oe.onMessage=D=>{if(D.serverContent){if(isGeminiLiveTranscribeModel()){const ge=Oe.
interimInputTranscript,W=Oe.inputTranscript,me=W+(ge&&!W.endsWith(ge)?(W?`
`:"")+ge:""),ye=get("chat-messages");let ee=document.getElementById(Y);ee||(ee=document.createElement(
"div"),ee.id=Y,ee.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ee.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ye.appendChild(ee),ye.scrollTop=ye.scrollHeight);const he=ee.
querySelector(".message-content");he.innerText=me||"\u8074\u304D\u53D6\u308A\u4E2D...",ye.scrollTop=
ye.scrollHeight,window.VoiceStudio&&W&&window.VoiceStudio.log("user",W);return}if(D.serverContent.modelTurn){
ie&&(setStsStatus("Gemini is speaking...",!1),ie=!1);const ge=get("chat-messages");let W=document.getElementById(
Y);W||(W=document.createElement("div"),W.id=Y,W.className="flex flex-col gap-2 mb-4 assistant-messag\
e bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",W.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ge.appendChild(W),ge.scrollTop=ge.scrollHeight);const me=W.
querySelector(".thought-container"),ye=W.querySelector(".message-content");Oe.assistantThought&&(me.
classList.remove("hidden"),me.innerText=Oe.assistantThought),ye.innerText=Oe.assistantText,ge.scrollTop=
ge.scrollHeight,window.VoiceStudio&&(Oe.inputTranscript&&window.VoiceStudio.log("user",Oe.inputTranscript),
Oe.assistantText&&window.VoiceStudio.log("assistant",Oe.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),di(Oe.stream),ui(Oe.stream);const fe=Oe;fe.onError=D=>{const ge=D&&D.message?D.message:
typeof D=="string"?D:"";ge&&showToast("Gemini Live: "+ge,"error",!0)},fe.onClose=D=>{if(Oe===fe){if(D&&
D.code&&D.code!==1e3){const ge=D.reason?": "+D.reason:" (code "+D.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+ge,"error",!0)}get("mic-btn").click()}},fe.closedEvent&&
fe.onClose(fe.closedEvent);return}catch(h){if(Pi(g),Oe){try{Oe.stop()}catch{}Oe=null}showToast("Gemi\
ni Live connection failed: "+h.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await mn.start();return}isStsModel()||(ci(),Bt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(Cn());st=new MediaRecorder(d),vn=[],On=!1;const m=isStsModel();
st.ondataavailable=g=>vn.push(g.data),st.onstop=async()=>{if(On){vn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(L=>L.stop()),Ln(),Ft(),m||(Bt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Sn=setTimeout(()=>Bt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(vn,{type:"audio/web\
m"}),h=new File([g],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",h),get("fil\
e-preview").classList.remove("hidden");const _=m;get("file-name").innerText=_?"Processing voice...":
"Transcribing...";try{if(_){if(!currentThreadId){const D=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=D.id!==null&&D.id!==void 0?String(D.id):D.id,setTemporaryChatUiState(!!(D&&D.
is_temporary)),setCurrentChatHeaderTitle(D&&D.title),applyTemporaryChatRuntimeMeta(D||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+D.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),x.append("model",get(
"model-select").value),x.append("thread_id",currentThreadId),get("sts-voice")&&x.append("sts_voice",
get("sts-voice").value||""),get("sts-reasoning-effort")&&x.append("sts_reasoning_effort",get("sts-re\
asoning-effort").value||""),get("sts-speed")&&x.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&x.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&x.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&x.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&x.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const L=await apiFetch(
"/sts",{method:"POST",body:x});if(!L.ok){const fe=await L.json().catch(()=>({}));throw new Error(fe.
error||"Speech-to-speech failed")}const M=L.body.getReader(),E=new TextDecoder;let O="",N=null,X=null;
stsOpt("sts-auto-play")&&(X=new un,Kt=X),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let oe=!0,ie="",Y="";for(;;){const{done:fe,value:D}=await M.read();if(fe)break;
O+=E.decode(D,{stream:!0});const ge=O.split(`
`);O=ge.pop();for(const W of ge){if(!W.trim())continue;const me=JSON.parse(W);if(me.error)throw new Error(
me.error);me.audio_delta&&X&&(oe&&(setStsStatus("Playing response...",!1),oe=!1),await X.addChunk(me.
audio_delta)),me.input_delta&&(ie+=me.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
ie)),me.transcript_delta&&(Y+=me.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",Y)),(me.final||me.audio_url)&&(N=me)}}window.VoiceStudio&&!ie.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),N&&(N.audio_url||N.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const L=get(
"set-mic-transcribe-mode");if(!!(L&&L.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const O=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(O.transcript){const N=get("prompt-inp\
ut");N.value+=(N.value?" ":"")+O.transcript,N.style.height="auto",N.style.height=N.scrollHeight+"px"}else
showToast(O.error||"Transcription failed","error",!0)}}catch(L){showToast("Audio processing error: "+
L.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(L=>L.
stop()),Ln(),Ft(),_||Bt("","hidden"),_&&setStsStatus("Tap to speak",!1)}},st.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Bt("\u9332\u97F3\u4E2D\u2026","recording"),di(d)),ui(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Ft(),isStsModel()||Bt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const pn=o((d,m)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),g=get("lib-download-btn"),h=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),_=get("lib-usage-btn");if(m&&(m.disabled=d===0,pn(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,pn(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
h&&(h.disabled=d===0,pn(h,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,pn(x,"\u540D\u524D\u5909\u66F4")),
_&&(_.disabled=d!==1,pn(_,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const L=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",L&&d>0)}};function hi(d){lib.
attachMode=!!d}o(hi,"setLibAttachMode");const bi=o((d=!1)=>{hi(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>bi(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>bi(!0)),
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
m=get("gem-desc").value,g=get("gem-inst").value,h=collectGemFixedPrompts();if(d&&g){const x=editingGemUuid?
"PUT":"POST",_=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
_,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:g,
fixed_prompts:h,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=d,activeGem.instruction=
g,activeGem.fixed_prompts=h,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(d){if(d.target.closest(".edit-btn")){const g=d.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(g)}if(d.target.closest(".code-toggle")){
const m=d.target.closest(".code-toggle"),g=m.closest(".code-wrapper");if(!g)return;const h=g.classList.
toggle("collapsed");g.setAttribute("data-collapsed",h?"true":"false"),m.setAttribute("aria-expanded",
h?"false":"true"),m.innerHTML=h?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',m.title=h?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",m.setAttribute("aria-label",h?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(d.target.closest(".download-btn")){const m=d.target.closest(".d\
ownload-btn"),g=m.getAttribute("data-code"),h=(m.getAttribute("data-lang")||"txt").toLowerCase();if(g)
try{const x=decodeURIComponent(g),_=new Blob([x],{type:"text/plain"}),L=URL.createObjectURL(_),M=document.
createElement("a");M.href=L;let O={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[h]||h;(h.length>8||/[^a-z0-9]/.test(h))&&(O="txt");let N=`code.${O}`;h==="dockerfile"&&(N="Docke\
rfile"),h==="makefile"&&(N="Makefile"),M.download=N,document.body.appendChild(M),M.click(),document.
body.removeChild(M),URL.revokeObjectURL(L)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),g=m.getAttribute("data-code");g&&
window.copyCode(m,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&Qn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
