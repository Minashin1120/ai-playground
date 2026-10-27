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
"hidden",!x)}}o(u,"updateGrokImageUi");function f(){var A;const d=get("ideogram-image-options"),m=get(
"modal-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",h=isIdeogramModel(
g),x=ideogramModelTraits(g),S={resolution:x.resolution,quality:x.quality,speed:x.speed,style:x.style,
negative:x.negative};if(d&&d.classList.toggle("hidden",!h),m&&m.classList.toggle("hidden",!h),updateIdeogramPromptInputUi(),
!h)return;const L=((A=get("ideogram-image-operation"))==null?void 0:A.value)==="remove_background";IDEOGRAM_IMAGE_FIELDS.
forEach(N=>{const R=N==="operation"||!L&&S[N]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(
X=>{const ae=get(X+N);!ae||!ae.parentElement||ae.parentElement.classList.toggle("hidden",!R)})}),["i\
deogram-image-style","modal-ideogram-image-style"].forEach(N=>{const R=get(N);if(!R)return;const X=x.
styleOptions.join("|");if(R.dataset.options!==X){const ae=R.value;R.innerHTML=x.styleOptions.map(ie=>`\
<option value="${ie}">${ie==="auto"?"Auto":ie}</option>`).join(""),R.dataset.options=X,R.value=x.styleOptions.
includes(ae)?ae:"auto"}});const M=get("ideogram-image-note");M&&(M.textContent=L?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25\
MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
x.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(d=>{
var m;(m=get(d))==null||m.addEventListener("change",()=>{const g=get(d.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");g&&(g.value=get(d).value),f()})});function b(){
var h;const d=get("grok-video-options");if(!d)return;const m=String(((h=get("model-select"))==null?void 0:
h.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");
const g=get("grok-video-resolution");if(g){const x=Array.from(g.options).find(L=>L.value==="1080p"),
S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(m);x&&(x.disabled=!S),!S&&g.value===
"1080p"&&(g.value="720p")}}o(b,"updateGrokVideoUi");function y(){var x;const d=get("gemini-video-opt\
ions");if(!d)return;const m=String(((x=get("model-select"))==null?void 0:x.value)||"").toLowerCase();
isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get("gemini-vide\
o-resolution");if(g){const S=Array.from(g.options).find(M=>M.value==="4K"),L=m==="veo-3.1-lite-gener\
ate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";S&&(S.disabled=L),L&&g.value===
"4K"&&(g.value="1080p")}const h=get("gemini-video-duration-wrap");h&&h.classList.toggle("hidden",m===
"gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function w(){const d=get("gemini-music-options");
if(!d)return;const m=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!m;d.classList.toggle("hid\
den",!g);const h=get("lyria-realtime-studio-bar");h&&h.classList.toggle("hidden",!m)}o(w,"updateGemi\
niMusicUi");function v(){var L;const d=get("xai-chat-options");if(!d)return;const m=String(((L=get("\
model-select"))==null?void 0:L.value)||"").toLowerCase(),g=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!g);const h=get("xai-logp\
robs"),x=get("xai-top-logprobs"),S=m.includes("grok-4.20");h&&(h.disabled=S,S&&(h.checked=!1)),x&&(x.
disabled=S,S&&(x.value=""))}o(v,"updateXaiChatUi");function k(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(h=>{const x=get(h);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function _(){const d=get("image-input-limit\
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
o(_,"updateImageInputLimits");function C(){const d=get("enable-sys-prompt"),m=!!(d&&d.disabled),g=!!(d&&
d.checked);E(),d&&(d.disabled?!m&&g?d.dataset.restoreChecked="1":m||delete d.dataset.restoreChecked:
(m&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(C,"toggleOpti\
ons"),window.toggleOptions=C;function T(d){const m=get("thinking-options"),g=get("thinking-level"),h=get(
"reasoning-effort"),x=get("enable-thinking"),S=get("thinking-budget-label"),L=get("thinking-budget"),
M=d==="effort";if(m&&(m.dataset.thinkingMode=M?"effort":"thinking"),g&&g.classList.toggle("hidden",M),
h&&h.classList.toggle("hidden",!M),S&&S.classList.toggle("hidden",M),L&&L.classList.toggle("hidden",
M),!!x)if(M){x.dataset.thinkingChecked===void 0&&(x.dataset.thinkingChecked=x.checked?"1":"0");const A=h?
Array.from(h.options).find(R=>R.value==="none"):null,N=!!(A&&!A.disabled&&!A.classList.contains("hid\
den"));x.checked=N?!!(h&&String(h.value||"").toLowerCase()!=="none"):!0,x.disabled=!N}else x.dataset.
thinkingChecked!==void 0&&(x.checked=x.dataset.thinkingChecked==="1",delete x.dataset.thinkingChecked)}
o(T,"setComposerThinkingMode");function E(){const d=get("model-select");if(!d)return;const m=d.value,
g=String(m||"").toLowerCase(),h=g.includes("deepseek"),x=g.startsWith("glm-"),S=get("thinking-option\
s"),L=get("reasoning-effort-container"),M=get("enable-thinking"),A=get("thinking-level"),N=get("thin\
king-budget"),R=get("enable-search"),X=get("search-container"),ae=get("url-context-container"),ie=get(
"enable-maps"),Y=get("maps-grounding-container"),pe=get("enable-sys-prompt"),H=get("sys-prompt-optio\
n"),fe=get("enable-python"),V=get("python-container"),ue=get("prompt-cache-container"),ve=get("enabl\
e-prompt-cache"),te=m==="gpt-5-search-api",ge=m.includes("tts"),Ie=isMistralOcrModel(m),Pe=g.includes(
"gemini-3.1-flash-lite-image"),it=g==="gemini-nano-banana-2.1",ze=g.includes("gemini-3.1-flash-image")&&
!Pe||it,ct=isClaudeModelKey(m),Ct=g==="gemini-3.8-flash-cyber",Le=isLlmModel()&&!h&&!ge&&!g.includes(
"realtime")&&!g.includes("native-audio")&&!g.includes("live");ue&&(Le?(ue.classList.remove("hidden",
"opacity-50","pointer-events-none"),ve&&(ve.disabled=!1)):(ve&&(ve.checked=!1,ve.disabled=!0),ue.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),L&&L.classList.
add("hidden"),T("thinking");const we=get("vision-model-info");if(we&&we.classList.add("hidden"),L){const _e=get(
"reasoning-effort");if(_e){Array.from(_e.options).forEach(qe=>{const We=g==="gpt-5.6"||g.startsWith(
"gpt-5.6-"),lt=g.startsWith("gpt-6"),fn=g==="gpt-6-sol"||g==="gpt-6-luna"||g==="gpt-6.1-terra",Jt=g===
"deepseek-v4.1-flash"||g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash\
-vision-exp",gn=g==="deepseek-v4-pro",qn=g.includes("grok-4.5"),F=g.includes("grok-4.6");qe.value===
"max"?qe.classList.toggle("hidden",!We&&!lt&&!Jt&&!gn):qe.value==="xhigh"?qe.classList.toggle("hidde\
n",!F&&!g.includes("multi-agent")&&!We&&!lt):qe.value==="medium"?qe.classList.toggle("hidden",!(g.includes(
"grok-4.3")||qn||F||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("mu\
lti-agent")||g.includes("gpt-5")||lt||g.includes("o1")||g.includes("o3"))):qe.value==="none"?qe.classList.
toggle("hidden",!g.includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!fn&&!Jt&&
!gn):qe.value==="low"&&qe.classList.toggle("hidden",gn)});const Oe=_e.selectedOptions&&_e.selectedOptions[0];
Oe&&Oe.classList.contains("hidden")&&(_e.value=h?"high":"medium")}}ae&&ae.classList.add("hidden"),Y&&
Y.classList.add("hidden"),M&&(M.disabled=!1),N&&(N.disabled=!0,N.classList.add("opacity-50"));const Fe=isGeminiImageModelKey(
m);if(ge||Ie)X&&(get("enable-search").checked=!1,X.classList.add("opacity-50","pointer-events-none")),
ae&&(get("enable-url-context").checked=!1,ae.classList.add("opacity-50","pointer-events-none")),Y&&ie&&
(ie.checked=!1,Y.classList.add("opacity-50","pointer-events-none")),V&&(fe.checked=!1,V.classList.add(
"opacity-50","pointer-events-none")),pe&&H&&(pe.checked=!1,pe.disabled=!0,H.classList.add("opacity-5\
0"));else if(ze||Pe)Y&&ie&&(ie.checked=!1,Y.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),it?(Array.from(A.options).forEach(_e=>{_e.disabled=_e.value==="low"}),[
"minimal","medium","high"].includes(A.value)||(A.value="medium")):(Array.from(A.options).forEach(_e=>{
["low","medium"].includes(_e.value)&&(_e.disabled=!0),["minimal","high"].includes(_e.value)&&(_e.disabled=
!1)}),["minimal","high"].includes(A.value)||(A.value=Pe?"minimal":"high")),M&&(M.disabled=!1),Pe&&(R&&
(R.checked=!1,R.disabled=!0),X&&X.classList.add("opacity-50","pointer-events-none"));else if(Fe)Y&&ie&&
(ie.checked=!1,Y.classList.add("hidden","opacity-50","pointer-events-none"));else if(ct)S.classList.
remove("hidden"),N&&(N.disabled=!1,N.classList.remove("opacity-50")),Array.from(A.options).forEach(_e=>{
_e.disabled=!0}),V&&(fe.checked=!1,V.classList.add("opacity-50","pointer-events-none"));else if(Ct){
S&&S.classList.remove("hidden"),M&&(M.checked=!0,M.disabled=!0),Array.from(A.options).forEach(Oe=>{Oe.
disabled=!["low","medium","high"].includes(Oe.value)}),["low","medium","high"].includes(A.value)||(A.
value="medium"),[X,ae,Y,V].forEach(Oe=>{Oe&&Oe.classList.add("opacity-50","pointer-events-none")}),[
R,ie,fe].forEach(Oe=>{Oe&&(Oe.checked=!1,Oe.disabled=!0)});const _e=get("enable-url-context");_e&&(_e.
checked=!1,_e.disabled=!0),pe&&H&&(pe.disabled=!1,H.classList.remove("opacity-50"))}else if(m.includes(
"gemini")&&!Fe){S.classList.remove("hidden"),ae&&ae.classList.remove("hidden","opacity-50","pointer-\
events-none");const _e=m.includes("gemini-3")||m.includes("gemini-4");Y&&(_e?Y.classList.remove("hid\
den","opacity-50","pointer-events-none"):(ie&&(ie.checked=!1),Y.classList.add("hidden","opacity-50",
"pointer-events-none")));const Oe=m.includes("flash");Array.from(A.options).forEach(qe=>{m==="gemini\
-4-argon"||m==="gemini-3.8-flash"||m==="gemini-3.7-flash"?qe.disabled=!["low","medium","high"].includes(
qe.value):m==="gemini-3.6-flash"?qe.disabled=!["medium","high"].includes(qe.value):m==="gemini-3.5-f\
lash-lite"?qe.disabled=!["minimal","medium","high"].includes(qe.value):["minimal","medium"].includes(
qe.value)?qe.disabled=!Oe:qe.disabled=!1}),(m==="gemini-4-argon"||m==="gemini-3.8-flash"||m==="gemin\
i-3.7-flash")&&!["low","medium","high"].includes(A.value)||m==="gemini-3.6-flash"&&!["medium","high"].
includes(A.value)?A.value="medium":m==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(
A.value)?A.value="minimal":!Oe&&["minimal","medium"].includes(A.value)&&(A.value="high"),_e?M&&(M.checked=
!0,M.disabled=!0):M&&(M.disabled=!1),N&&m.includes("gemini-2.5")&&(N.disabled=!1,N.classList.remove(
"opacity-50")),N&&!m.includes("gemini-2.5")&&(N.disabled=!0,N.classList.add("opacity-50"))}if(isLlmModel()&&
(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4\
.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes(
"multi-agent")||g.includes("gpt")&&!g.includes("tts")))S&&S.classList.remove("hidden"),T("effort"),X&&
X.classList.remove("opacity-50","pointer-events-none");else if(h){S&&S.classList.remove("hidden"),T(
"effort");const _e=get("vision-model-info");if(_e&&_e.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),R&&(R.checked=!1,R.disabled=!0),X&&X.classList.add("opaci\
ty-50","pointer-events-none"),ae){const Oe=get("enable-url-context");Oe&&(Oe.checked=!1),ae.classList.
add("opacity-50","pointer-events-none")}Y&&ie&&(ie.checked=!1,Y.classList.add("opacity-50","pointer-\
events-none"))}else Ie||(X&&X.classList.remove("opacity-50","pointer-events-none"),Y&&ie&&(ie.checked=
!1,Y.classList.add("hidden","opacity-50","pointer-events-none")));if(ge?V&&V.classList.add("opacity-\
50","pointer-events-none"):(V&&V.classList.remove("opacity-50","pointer-events-none"),(!Fe||ze)&&!m.
includes("gpt-image")&&(pe.disabled=!1,H.classList.remove("opacity-50"))),(Fe&&!ze||m.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(m)||isGrokVideoModel()||Ie)&&pe&&H&&(pe.checked=!1,pe.disabled=
!0,H.classList.add("opacity-50")),V&&(isLlmModel()?(V.classList.remove("hidden"),fe.disabled=!1):(fe.
checked=!1,fe.disabled=!0,V.classList.add("hidden"))),te?(R&&(R.checked=!0,R.disabled=!0),X&&X.classList.
add("opacity-50","pointer-events-none"),V&&(fe.checked=!1,fe.disabled=!0,V.classList.add("opacity-50",
"pointer-events-none"))):R&&!m.includes("tts")&&!Ie&&!h&&!Pe&&(R.disabled=!1),Ct){[R,ie,fe].forEach(
Oe=>{Oe&&(Oe.checked=!1,Oe.disabled=!0)});const _e=get("enable-url-context");_e&&(_e.checked=!1,_e.disabled=
!0),[X,ae,Y,V].forEach(Oe=>{Oe&&Oe.classList.add("opacity-50","pointer-events-none")})}x&&(we&&we.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[R,fe].forEach(_e=>{_e&&(_e.checked=!1,_e.disabled=!0)}),[X,V].forEach(_e=>{_e&&
_e.classList.add("opacity-50","pointer-events-none")}));const st=get("mask-btn");st&&(isGptImageModel()?
st.classList.remove("hidden"):(st.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),
updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),u(),f(),b(),y(),w(),updateBatchUi(m),v(),k(),
_(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}o(E,"toggle\
OptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",C),get("model-\
select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),C(),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const j=get("enable-canvas-mode");j&&(j.checked=canvasModeEnabled,j.addEventListener("change",()=>syncCanvasModeUi(
j.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const Q=get("enable-codin\
g-mode");Q&&(Q.checked=codingModeEnabled,Q.addEventListener("change",()=>syncCodingModeUi(Q.checked))),
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
exec")return"";if(h==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",S=(h||"").toLowerCase();
let L="";try{const H=hljs.getLanguage(h)?h:"plaintext";activeStreamingBubbleId&&x.length>2e4?L=escapeHtml(
x):L=hljs.highlight(x,{language:H}).value}catch{L=escapeHtml(x)}const M=encodeURIComponent(x).replace(
/'/g,"%27"),A=detectBlockedScriptsInCode(x),N=hashString(`${h||"TEXT"}
${x||""}`);let R="";if(canvasModeEnabled){const H=String(canvasPreviewState.selectedKey||"")===N,fe=H?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";R=`<button\
 class="canvas-preview-btn${H?" canvas-active":""}" data-code="${M}" data-code-key="${N}" data-canva\
s-lang="${escapeHtml(h||"txt")}" title="${fe}" aria-label="${fe}" aria-pressed="${H?"true":"false"}"\
><i class="fas ${H?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,x)){const H=A?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
R=`<button class="html-preview-btn" data-code="${M}" ${A?'data-suspicious="1"':""} title="${H}" aria\
-label="${H}"><i class="fas ${A?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const X=`\
<button class="download-btn" data-code="${M}" data-lang="${h||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ae=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${M}" data-code-key="${N}" data-coding-lang="${escapeHtml(h||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,ie=(h||"TEXT")+(A?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),Y=getRenderableSvgCode(S,x);return`${Y?buildSvgCodeRenderHtml(Y,N):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${N}"><div class="code-header"\
><span class="code-lang">${ie}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${ae}${R}${X}\
<button class="copy-btn" data-code="${M}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${h}">${L}</code></pre></d\
iv></div>`},link(d,m,g){return`<a href="${d}" title="${m||""}" target="_blank">${g}</a>`},image(d,m,g){
return buildChatImageHtml(d,{alt:g,title:m})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(yi=get("version-update-dismis\
s"))==null||yi.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const Se=get("version-update-clear\
-cache");if(Se&&(Se.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),Se.
addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(
Se.checked)})),(vi=get("version-update-reload"))==null||vi.addEventListener("click",async()=>{var m;
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
classList.add("hidden")}const G=o(d=>{if(!d)return"\u4E0D\u660E";const m=new Date(d);return Number.isNaN(
m.getTime())?d:m.toLocaleString()},"formatSessionTime"),re=o(d=>{const m=Array.isArray(d)?d:[],g=get(
"passkey-list"),h=get("passkey-count");if(h&&(h.innerText=String(m.length)),!!g){if(!m.length){g.innerHTML=
'<div class="text-[11px] text-gray-500">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}g.innerHTML="",m.forEach((x,S)=>{const L=x&&x.id?String(x.id):"",M=document.createElement("di\
v");M.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const A=document.createElement("div");A.className="min-w-0";const N=document.createElement("\
div");N.className="text-xs text-gray-200 truncate",N.innerText=x&&x.name?String(x.name):`Security Ke\
y ${S+1}`;const R=document.createElement("div");R.className="text-[10px] text-gray-500 mt-1",R.innerText=
x&&x.created_at?`\u767B\u9332\u65E5\u6642: ${G(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
A.appendChild(N),A.appendChild(R),M.appendChild(A);const X=document.createElement("button");X.type="\
button",X.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",X.innerText="\u524A\u9664",X.disabled=!L,L&&(X.onclick=()=>window.removeWebAuthnCredential(
L)),M.appendChild(X),g.appendChild(M)})}},"renderPasskeyList"),le=o(d=>{const m=get("session-list");
if(m){if(!d||!d.length){m.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(g=>{const h=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
L=(g.user_agent||"Unknown").slice(0,120),M=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${h}${x}<div class="text-xs text-gray-200">${escapeHtml(
M)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(L)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(G(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(G(g.created_at))}</div></div>${S}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const h=g.getAttribute("data-session-id");if(!h||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:h})});let S={};try{S=await x.json()}catch{}if(x.ok){if(S.logged_out){
location.href="/login";return}await xe()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),xe=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api\
/sessions");let g={};try{g=await m.json()}catch{}if(!m.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const h=(g.sessions||[]).filter(x=>!x.is_revoked);le(h)},"loadSessions"),me=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>xe());const m=get("session-revoke-others-btn");m&&(m.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await xe():showToast("\u64CD\u4F5C\u306B\u5931\
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
const ke=get("vision-model-change-btn");ke&&(ke.onclick=()=>_openVisionModelSelector());const $=get(
"compression-format-only");$&&($.onchange=()=>{const d=$.checked,m=get("compression-max-size"),g=get(
"compression-max-dim");m&&(m.disabled=d),g&&(g.disabled=d);const h=get("compression-size-wrap"),x=get(
"compression-dim-wrap");h&&(h.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const B=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");B(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const q=get("storage-us\
age-refresh");q&&(q.onclick=()=>loadStorageUsage());let P=null;const D=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),I=o((d={})=>{const m=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),h=get("account-transfer-progress-percent"),x=get("account-transfer-progress-text"),
S=get("account-transfer-progress-detail"),L=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&m.
classList.remove("hidden"),g&&(g.style.width=`${L}%`),h&&(h.textContent=`${Math.round(L)}%`),x&&(x.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const A={queued:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=A[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const M=get("account-transfer-cancel-btn");M&&M.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),O=o(d=>{be&&(be.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),ce=o((d={})=>{const m=get("account-export-ready"),
g=get("account-export-ready-text"),h=get("account-export-expiry"),x=get("account-export-download-btn"),
S=!!(d.available&&d.download_url);if(m&&m.classList.toggle("hidden",!S),!S){x&&x.removeAttribute("hr\
ef");return}const L=Math.max(0,Number(d.size_bytes)||0),M=L>=1024*1024*1024?`${(L/(1024*1024*1024)).
toFixed(2)} GB`:`${(L/(1024*1024)).toFixed(1)} MB`;if(g){const A=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
g.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${M}${A}`}
if(h){const A=d.expires_at?new Date(d.expires_at):null;h.textContent=A&&!Number.isNaN(A.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${A.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),oe=o(async d=>{for(;P===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await m.json().catch(()=>({}));if(m.ok&&(g.state!=="pending"&&I(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(m=>setTimeout(m,700))}
return null},"pollAccountTransfer"),ee=o((d,m,g=!0)=>{m&&(I(m),ce(m),g&&m.state==="ready"?showToast(
m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):g&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),ne(d))},"handleFinishedAccountExport"),se=o(async()=>{try{const d=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(ce(m),m.state==="ready"){I(m);return}if(["failed","cancelled","expired"].includes(
m.state)){I(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||P&&P.id===
m.job_id||P)return;const g={id:m.job_id,type:"export",stopped:!1,restored:!0};P=g,O(!0),I(m);const h=await oe(
g);h&&ee(g,h,!0)}catch{}},"refreshLatestAccountExport"),ne=o(d=>{P===d&&(P=null),d.stopped=!0,O(!1)},
"finishAccountTransfer"),K=get("account-transfer-cancel-btn");K&&(K.onclick=async()=>{const d=P;if(!(!d||
d.stopped)){d.cancelRequested=!0,K.disabled=!0,I({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),I({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&ce({available:!1}),ne(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const be=get("account-export-btn");be&&(be.onclick=async()=>{if(P)return;const d={id:D(),
type:"export",stopped:!1};P=d,O(!0),ce({available:!1}),I({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const m=await apiFetch("/api/account/export",
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{job_id:d.id}),keepalive:!0})),g=await m.json().catch(()=>({}));if(m.status===409&&g.error==="export\
_in_progress"&&g.job_id)d.id=g.job_id;else if(!m.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\
\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
I({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const h=await oe(d);!d.cancelRequested&&h&&ee(d,h,!0)}catch(m){const g=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";I({progress:0,phase:"faile\
d",message:g}),showToast(g,"error",!0),ne(d)}});const W=get("account-export-download-btn");W&&W.addEventListener(
"click",async d=>{const m=W.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),h=await g.json().catch(
()=>({}));g.ok&&h.available&&h.download_url?(W.href=h.download_url,window.location.assign(h.download_url)):
(ce(h),I(h),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),se())}catch{window.location.assign(m)}}}),O(!1),se();const Ee=get("import-files-grid"),
ut=get("import-files-info"),mt=get("import-files-summary"),vt=o(d=>{const m=Math.max(0,Number(d)||0);
return m>=1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(
1)} MB`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let $e=null;const ht=o(()=>{
if(!$e)return;const d=$e.files,m=$e.selection;let g=0;d.forEach(S=>{m.has(S.archive_path)&&(g+=Number(
S.size_bytes)||0)});const h=Number($e.available_bytes)||0,x=g>h;mt&&(mt.textContent=`\u9078\u629E\u4E2D: ${vt(
g)} / \u5229\u7528\u53EF\u80FD: ${vt(h)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,mt.classList.
toggle("text-red-300",x)),ut&&(ut.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),kt=o(
()=>{if(!Ee||!$e)return;Ee.innerHTML="";const d=$e.files;if(!d.length){Ee.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
ht();return}d.forEach(m=>{const g=document.createElement("label"),h=$e.selection.has(m.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${h?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${h?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${vt(m.size_bytes)}</div></div>`;const x=g.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?$e.selection.add(m.archive_path):
$e.selection.delete(m.archive_path),g.classList.toggle("border-blue-500",x.checked),g.classList.toggle(
"border-gray-600",!x.checked),ht()}),Ee.appendChild(g)}),ht()},"renderImportFileItems"),Tt=o(d=>new Promise(
m=>{if($e={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:m},kt(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),Mt=o(d=>{if(hideModal("import-files-modal"),$e){const m=$e.resolve;
$e=null,m(d)}},"closeImportFileSelection"),At=get("import-files-close");At&&(At.onclick=()=>Mt(null));
const bt=get("import-files-cancel");bt&&(bt.onclick=()=>Mt(null));const Bt=get("import-files-confirm");
Bt&&(Bt.onclick=()=>{if(!$e)return;const d=Array.from($e.selection);Mt(d.length?d.join(","):"__none_\
_")});const Et=get("import-files-select-all");Et&&(Et.onclick=()=>{$e&&($e.files.forEach(d=>$e.selection.
add(d.archive_path)),kt())});const pt=get("import-files-none");pt&&(pt.onclick=()=>{$e&&($e.selection.
clear(),kt())});const Ft={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},Kt=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const m=String(
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let wt=null;const yt=o(
d=>{if(wt){const m=wt;wt=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Xt=o(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}wt=m;const h=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(h.length?
x.innerHTML=h.map(L=>{const M=Ft[L.field]||L.field,A=Kt(L.current),N=Kt(L.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(M)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
A)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(N)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${h.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),jt=get("settings-confir\
mation-modal");jt&&jt.addEventListener("click",d=>{d.target===jt&&yt(!1)});const Yt=get("settings-co\
nfirmation-close");Yt&&(Yt.onclick=()=>yt(!1));const Qt=get("settings-confirmation-cancel");Qt&&(Qt.
onclick=()=>yt(!1));const Zt=get("settings-confirmation-confirm");Zt&&(Zt.onclick=()=>yt(!0));const en=get(
"account-import-btn"),U=get("account-import-inplace"),he=get("account-import-inplace-warning");if(U&&
he){const d=o(()=>he.classList.toggle("hidden",!U.checked),"syncInplaceWarn");U.addEventListener("ch\
ange",d),d()}en&&(en.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:null,
g=get("account-import-categories"),h=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).
map(ie=>ie.value):[],x=get("account-import-inplace"),S=!!(x&&x.checked),L=get("account-import-settin\
gs-bypass"),M=!!(L&&L.checked);let A=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!h.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const N=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(ie=>(ie.
closest("label")&&ie.closest("label").textContent||ie.value).trim()):h;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${N.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const R={id:D(),type:"import",stopped:!1,controller:new AbortController};
P=R,O(!0),I({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const X=get("account-import-result");let ae=Promise.resolve(null);try{const Y=Math.max(1,Math.ceil(m.
size/10485760)),pe=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:R.controller.
signal})),H=await pe.json().catch(()=>({}));if(!pe.ok)throw new Error(H.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
R.uploadId=H.upload_id;const fe=H.chunk_size||10485760;let V=0,ue=0;const ve=o(async()=>{for(;;){const Le=ue++;
if(Le>=Y)return;const we=m.slice(Le*fe,Math.min(m.size,(Le+1)*fe)),Fe=new FormData;Fe.append("chunk",
we,m.name),Fe.append("index",String(Le));const Xe=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
R.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Fe,signal:R.controller.signal})),
st=await Xe.json().catch(()=>({}));if(!Xe.ok)throw new Error(st.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
V++,I({progress:Math.min(35,Math.round(V/Y*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${V}\
/${Y}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let te=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),te=!0);try{await Promise.
all([ve(),ve(),ve()]);const Le=await apiFetch(`/api/account/import/upload/${encodeURIComponent(R.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:R.controller.signal})),we=await Le.json().
catch(()=>({}));if(!Le.ok)throw new Error(we.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
I({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
te&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ge="",Ie=!1,Pe=0;const it=o(
async()=>{let Le=!1;const we=o(()=>{Le||(Le=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Fe=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Xe=await Fe.
json().catch(()=>null);if(!Fe.ok||!Xe){we();return}cacheUserSettings(Xe);const st=get("settings-moda\
l");if(st&&st.classList.contains("modal-open"))try{zn(Xe)}catch{}Xe.theme_color&&applyThemeColor(Xe.
theme_color,!0),Object.prototype.hasOwnProperty.call(Xe,"minimal_prompt_mode")&&Xe.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Xe,"compact_prompt_mode")&&setCompactPromptMode(
!!Xe.compact_prompt_mode)}catch{}we()},"refreshSettingsFormAfterImport"),ze=o(Le=>{const we=Le&&Le.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";X&&(X.textContent=`\u5B8C\u4E86: ${we}`,
X.classList.remove("hidden","text-red-300"),X.classList.add("text-emerald-300")),I({progress:100,phase:"\
completed",message:we}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&it()},"finishImportSuccess"),ct=o(async()=>{
try{const we=await(await apiFetch(`/api/account/transfer/${R.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return we&&we.state?we:null}catch{return null}},"fetchImportSta\
tus"),Ct=o(async()=>{const Le=await ct();if(!Le)return{status:"unknown"};if(Le.state==="completed")return ze(
Le),{status:"done"};if(["failed","cancelled","expired"].includes(Le.state))throw new Error(Le.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Le.state==="needs_sel\
ection"&&Array.isArray(Le.files)){const we=await Tt({files:Le.files,available_bytes:Le.available_bytes});
return we===null?(I({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
R.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(R.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ge=we,{status:"reselect"})}if(Le.state===
"needs_settings_confirmation"&&Array.isArray(Le.settings_changes))return await Xt({settings_changes:Le.
settings_changes})?(A=!0,{status:"reselect"}):(I({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),R.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
R.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Le.state==="running"){const we=await Promise.race([ae.catch(()=>null),new Promise(Fe=>setTimeout(
()=>Fe(null),6e4))]);if(we&&we.state==="completed")return ze(we),{status:"done"};throw we&&["failed",
"cancelled","expired"].includes(we.state)?new Error(we.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ie;){R.stopped=!0,await ae.catch(()=>null),
R.stopped=!1,ae=oe(R);let Le;try{Le=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:R.uploadId,
categories:h.join(","),job_id:R.id,selected_files:ge,restore_inplace:S,confirm_settings:A||M}),signal:R.
controller.signal}))}catch(We){if(R.cancelRequested||We&&We.name==="AbortError")throw We;const lt=await Ct();
if(lt.status==="done"){Ie=!0;break}if(lt.status==="cancelled")return;if(lt.status==="reselect")continue;
if(Pe<2){Pe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let we=null;try{we=await Le.json()}catch{we=null}if(we===null){const We=await Ct();if(We.status==="d\
one"){Ie=!0;break}if(We.status==="cancelled")return;if(We.status==="reselect")continue;if(Le.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Pe<2){Pe++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Le.ok&&we.error==="storage_limit_files"&&we.files){const We=await Tt(we);if(We===null){I({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
R.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(R.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ge=We;continue}if(we&&we.status==="settings_confirmation"&&
Array.isArray(we.settings_changes)){if(!await Xt(we)){I({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),R.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(R.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}A=!0;continue}if(!Le.ok)throw new Error(we.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Fe=we.imported||{},Xe=[`\u8A2D\u5B9A ${Fe.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Fe.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Fe.chats||0}\u4EF6`,`Gem ${Fe.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Fe.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Fe.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Fe.
diagnostics||0}\u4EF6`].join(" / "),st=we.duplicates||{},_e={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Oe=[];for(const We of Object.keys(_e)){const lt=Number(st[We])||0;lt>
0&&Oe.push(`${_e[We]} ${lt}\u4EF6`)}const qe=Oe.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Oe.
join("\u3001")}\uFF09`:"";X&&(X.textContent=`\u5B8C\u4E86: ${Xe}${qe}`,X.classList.remove("hidden","\
text-red-300"),X.classList.add("text-emerald-300")),I({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&it(),Ie=!0}}catch(ie){if(R.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(R.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),R.cancelRequested||ie&&ie.name==="AbortError")return;const Y=ie&&ie.message?
ie.message:"",pe=Y==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
Y||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";I({progress:0,phase:"f\
ailed",message:pe}),X&&(X.textContent=pe,X.classList.remove("hidden","text-emerald-300"),X.classList.
add("text-red-300")),showToast(pe,"error",!0)}finally{R.stopped=!0,await ae.catch(()=>null),ne(R)}});
const Ae=get("account-dedupe-btn"),Be=get("account-dedupe-result"),Qe=o((d,m=!1)=>{Be&&(Be.textContent=
d,Be.classList.remove("hidden"),Be.classList.toggle("text-red-300",!!m),Be.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");Ae&&(Ae.onclick=async()=>{const d=o(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await m.json().catch(()=>null);if(!m.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){Qe("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const h=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const R of[
"chats","gems","files","feedback","diagnostics"]){const X=Number(g.duplicates&&g.duplicates[R])||0;X>
0&&h.push(`${x[R]} ${X}\u4EF6`)}const S=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${h.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const L=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),M=await L.json().catch(()=>null);if(!L.ok||!M)throw new Error(
M&&M.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const A=[];for(const R of["chats","gems","files","feedback","diagnostics"]){const X=Number(M.removed&&
M.removed[R])||0;X>0&&A.push(`${x[R]} ${X}\u4EF6`)}const N=Number(M.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${M.kept_referenced_files}\u4EF6\uFF09`:"";Qe(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${A.join("\u3001")||"0\u4EF6"}${N}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!Ae.disabled){Ae.disabled=!0,Qe("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){Qe(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{Ae.disabled=!1}}});const ot=get("site-cache-usage-refresh");ot&&(ot.onclick=()=>loadSiteCacheUsage());
const rt=get("clear-site-cache-btn");rt&&(rt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(rt)});
const De=get("enc-scan-result"),ft=o(async(d=null)=>{De&&(De.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
m,{cache:"no-store"}),h=await g.json();if(!g.ok){De&&(De.textContent=h.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=h.total||0,S=h.encrypted||0,L=h.unencrypted||0;let M=`Total: ${x} / Encrypted: ${S} /\
 Plain: ${L}`;if(h.samples&&h.samples.length){const A=h.samples.slice(0,8).map(N=>{const R=N.timestamp?
new Date(N.timestamp).toLocaleString():"";return`#${N.id} (${N.role||""}) ${R}`}).join(" / ");M+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${A}</div>`}De&&(De.innerHTML=M)}catch{De&&(De.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Ve=get("enc-scan-all");Ve&&(Ve.onclick=()=>ft(
null));const Ze=get("enc-scan-thread");Ze&&(Ze.onclick=()=>currentThreadId?ft(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const He=get("admin-enc-\
list");let Dt=null,$t=!1;const hn=o(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThread\
EncryptedFromMessages"),bn=o(()=>{Dt=hn(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Ne=o(
async(d,m,{confirmPrompt:g=!0,reloadCurrent:h=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;$t=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
L=await S.json().catch(()=>({}));return S.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${L.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Dt=!!m,h&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),He&&await Je(),!0):
(showToast(L.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{$t=!1}},"setAdminThreadEncry\
ption"),Ge=o(d=>{if(!He)return;const m=d.threads||[];if(!m.length){He.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
He.innerHTML=m.map(g=>{const h=g.encrypted_count>0?"enc":"plain",x=h==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=h==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",L=g.
updated_at?new Date(g.updated_at).toLocaleString():"",M=escapeHtml(String(g.thread_id)),A=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${A?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${L} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${M}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${M}" data-enable="${h==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Je=o(async()=>{if(He){He.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){He.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(Ge(m),currentThreadId&&Array.isArray(m.threads)){const g=m.threads.find(h=>String(
h.thread_id)===String(currentThreadId));g&&(Dt=!!g.encrypted)}}catch{He.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Je()),window.__loadAdminEncThreads=
Je,window.__refreshAdminThreadEncState=bn,window.__setAdminThreadEncryption=Ne;const tn=get("encrypt\
ion-status-admin-toggle");tn&&tn.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),He&&(He.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const M=m.getAttribute("data-id");if(!M)return;typeof Ht==
"function"?Ht():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(M)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||$t)return;const h=g.getAttribute("data-i\
d"),x=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const L=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await Ne(h,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=L,await Je()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const yn=o(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(g=>{const h=g.status||"new",x=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=g.created_at?new Date(g.created_at).toLocaleString():"",L=g.replied_at?new Date(g.replied_at).toLocaleString():
"",M=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${x}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(S)}</div>
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
                    `}).join("")}},"renderBanAppeals"),_t=o(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const x=(await g.json()).unread_count||0;m.textContent=String(x),d&&x>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${x}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),ye=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const g=await m.json();yn(g.items||[]),
await _t(!1)}catch{}}},"loadBanAppeals"),Te=o(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await ye()}catch{}},"markBanAppealsRead"),Ke=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await ye()}catch{}},"updateBanAppealStatus"),at=o(()=>{
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
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),tt=o(()=>{const d=get("set-st\
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
"success")})},"ensureLlmTranscribePromptSettingsUi"),gt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
700 rounded p-2 text-xs text-gray-200";return gt.map(h=>{const x=h.mcpLocked===!0,S=x?'<div class="t\
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
                        ${S}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{gt.forEach(g=>{const h=m&&
typeof m=="object"?m[g.key]||{}:{},x=get(`${d}-auto-sys-${g.key}-enabled`),S=get(`${d}-auto-sys-${g.
key}-text`);x&&(g.mcpLocked===!0?x.disabled=!0:x.checked=h.enabled!==!1),S&&(S.value=h.text||"",S.placeholder=
h.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const St=o((d,m=null)=>{if(m){const g=get(m);g&&(g.checked=!0)}gt.forEach(g=>{const h=get(`${d}-auto\
-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);if(h&&(g.mcpLocked!==!0?h.checked=!0:h.
disabled=!0),x){const S=x.placeholder||"";x.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Un=o(d=>{const m={};return gt.forEach(g=>{const h=get(`${d}\
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
                `,m.appendChild(g)},at(),tt(),B();const xi=o(()=>{const d=get("set-default-model");if(!d)
return;const m=d.value;d.innerHTML="",MODELS.forEach(h=>{const x=document.createElement("optgroup");
x.label=h.category,(h.items||[]).forEach(S=>{const L=document.createElement("option");L.value=S.id,L.
textContent=S.name,x.appendChild(L)}),d.appendChild(x)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"po\
pulateDefaultModelOptions"),ki=o(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(h=>{const x=(h.items||[]).filter(L=>{const M=(L.id||"").toLowerCase();
return M.startsWith("gemini-")||M.startsWith("gpt-4o")||M.startsWith("claude-")||M.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
M)});if(x.length===0)return;const S=document.createElement("optgroup");S.label=h.category,x.forEach(
L=>{const M=document.createElement("option");M.value=L.id,M.textContent=L.name+" \u2605",S.appendChild(
M)}),d.appendChild(S)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
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
ame")&&(get("set-username").value=d.username);const x=get("2fa-badge"),S=get("disable-2fa-btn");d.is_2fa_enabled?
(x.innerText="ENABLED",x.classList.replace("bg-gray-700","bg-green-600"),x.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(x.innerText="DISABLED",x.classList.replace("bg\
-green-600","bg-gray-700"),x.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const L=get(
"set-passkey-only-login"),M=get("passkey-only-note"),A=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(re(A),L){L.checked=!!d.passkey_only_login;const ie=A.length>0||!!d.has_webauthn;L.disabled=!ie,
ie||(L.checked=!1),M&&(ie?M.classList.add("hidden"):M.classList.remove("hidden"))}const N=get("mig-s\
tatus-box"),R=get("mig-progress-text"),X=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){N.classList.remove("hidden");const ie=(d.migration_progress||"").split("/");if(ie.length===
2){const Y=parseInt(ie[0]||"0",10),pe=parseInt(ie[1]||"0",10);R&&(R.innerText=`${Y} / ${pe}`),X&&pe>
0&&(X.style.width=`${Math.min(100,Math.floor(Y/pe*100))}%`)}}else N.classList.add("hidden"),X&&(X.style.
width="0%"),R&&(R.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&zn(d);const m=get(
"search-box"),g=m?m.value:"";clearTimeout(searchTimeout);const h=get("settings-search");if(h&&(h.value=
""),filterSettings(),xi(),ki(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),se(),tt(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),_t(!0),ye(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),on(),me(),xe();try{loadMcpServers()}catch{}};const Ht=o((d=!1)=>{snapshotSidebarHistory(
"settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),_i=o(()=>{const d=get("set-th\
eme-color"),m=get("set-theme-color-text"),g=get("theme-reset-btn"),h=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((S,L=!0)=>{const M=normalizeHex(S);M&&(applyThemeColor(M,L),syncThemeInputs(
M))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),m&&(m.addEventListener("chan\
ge",()=>{const S=normalizeHex(m.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(S,!0)}),m.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
m.blur())})),g&&(g.onclick=()=>x(THEME_DEFAULT,!0)),h.forEach(S=>{S.addEventListener("click",()=>x(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),Si=o(()=>{const d=get("reset-global-sys-prom\
pt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const m=get(
"reset-thread-sys-prompt");m&&(m.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{St("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-thread-auto-sys-prompt-defaults");h&&(h.onclick=()=>{St("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Ht();const Wn=get("settings-header-close");Wn&&(Wn.onclick=()=>Ht());
const qt=get("settings-search");qt&&(qt.addEventListener("input",filterSettings),qt.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const g=m.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Vn=get(
"settings-search-clear");Vn&&Vn.addEventListener("click",()=>{qt&&(qt.value="",filterSettings(),qt.focus())}),
_i(),Si(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Mn=get("set-gemini-local-python-dialog");Mn&&(Mn.onchange=()=>setGeminiLocalPyDialogEnabled(Mn.
checked));const Jn=get("set-gemini-backend");Jn&&(Jn.onchange=()=>syncGeminiBackendUi());const Kn=get(
"set-admin-api-key-mode");Kn&&(Kn.onchange=()=>syncAdminApiKeyModeUi());const An=get("set-temp-chat-\
timeout-seconds");An&&(An.onchange=()=>{applyTemporaryChatTimeoutSeconds(An.value)});const Xn=get("s\
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
et-e2ee")?get("set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
x!==S&&(h.enable_e2ee=x),get("set-openai")&&(h.openai_key=get("set-openai").value),get("set-gemini")&&
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
tent-Type":"application/json"},body:JSON.stringify(h)});if(L.ok){let A="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const ae=await L.json();ae&&ae.message&&(A=ae.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!h.light_mode_enabled,liquid_glass_enabled:!!h.liquid_glass_enabled})),
window.applySavedUserSystemPromptSettings({system_prompt:h.system_prompt,system_prompt_enabled:h.system_prompt_enabled,
apply_global_system_prompt:h.apply_global_system_prompt,apply_auto_system_prompt_notices:h.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:h.auto_system_prompt_notices_config}),Ht();const N=currentUsername,
R=CHAT_CONFIG.enableE2EE;enterToSend=h.enter_to_send,autoSearchOnLinks=h.auto_search_on_links;const X=useSwCache;
useSwCache=h.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!h.clear_cache_on_version_update),
compactPromptMode=h.compact_prompt_mode,minimalPromptMode=h.minimal_prompt_mode,voiceStudioUiEnabled=
h.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=h.temp_chat_timeout_seconds,applyThemeColor(h.theme_color,
!0),syncThemeInputs(h.theme_color),applyLightMode(h.light_mode_enabled),applyLiquidGlassMode(h.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),X!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(A,"success"),syncClientDebugLogToggle(
h.enable_client_debug_log,"settings saved"),h.new_username&&h.new_username!==N?setTimeout(()=>location.
reload(),1e3):h.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let A={};try{A=await L.json()}catch{}showToast(A.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
x.classList.remove("hidden")),d&&(d.value="");else{const S=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(V=>{m&&(m.textContent=V)},"setBox"),h=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((V,ue)=>!V||!ue||ue<=0?0:V*8/(ue/1e3)/1e3/1e3,"toMbps"),S=o(V=>Number.isFinite(
V)?`${V.toFixed(0)} ms`:"-","fmtMs"),L=o(V=>Number.isFinite(V)?`${V.toFixed(V>=100?0:1)} Mbps`:"-","\
fmtMbps"),M=o(async(V,ue)=>{const ve=await V.json().catch(()=>({}));return ve&&ve.error?ve.error:ue},
"parseErr"),A=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let V=0;V<4;V++){const ue=performance.now(),ve=await apiFetch(
`/api/speedtest/ping?_=${h()}`,{cache:"no-store"}),te=performance.now();if(!ve.ok)throw new Error(await M(
ve,"ping_failed"));await ve.json().catch(()=>({})),A.push(te-ue)}const N=A.reduce((V,ue)=>V+ue,0)/Math.
max(1,A.length),R=Math.min(...A),X=o(async V=>{const ue=performance.now(),ve=await apiFetch(`/api/sp\
eedtest/download?bytes=${V}&_=${h()}`,{cache:"no-store"});if(!ve.ok)throw new Error(await M(ve,"down\
load_failed"));const te=await ve.arrayBuffer(),ge=performance.now();return{bytes:te.byteLength||V,ms:ge-
ue,mbps:x(te.byteLength||V,ge-ue)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${S(N)}
\u6E2C\u5B9A\u4E2D... download`);const ae=[];for(const V of[2*1024*1024,8*1024*1024])ae.push(await X(
V)),g(`\u6E2C\u5B9A\u4E2D... ping ${S(N)}
download ${L(Math.max(...ae.map(ue=>ue.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const ie=Math.max(...ae.map(V=>V.mbps)),Y=o(async V=>{const ue=new Uint8Array(
V),ve=performance.now(),te=await apiFetch(`/api/speedtest/upload?_=${h()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ue,cache:"no-store"}),ge=performance.now();if(!te.ok)
throw new Error(await M(te,"upload_failed"));const Ie=await te.json().catch(()=>({})),Pe=Number(Ie.bytes_received||
V)||V;return{bytes:Pe,ms:ge-ve,mbps:x(Pe,ge-ve),serverMs:Number(Ie.server_elapsed_ms||0)||0}},"runUp\
load"),pe=[];for(const V of[1*1024*1024,4*1024*1024])pe.push(await Y(V));const H=Math.max(...pe.map(
V=>V.mbps)),fe=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(N)} / ${S(R)}`,`Download (best): ${L(ie)}`,`Upload (best): ${L(H)}`,`Download r\
uns: ${ae.map(V=>`${Math.round(V.bytes/1024/1024)}MB=${L(V.mbps)}`).join(", ")}`,`Upload runs: ${pe.
map(V=>`${Math.round(V.bytes/1024/1024)}MB=${L(V.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(fe.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>ye()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>Te()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){g&&await Te([Number(g)]);return}if(m.classList.contains(
"ban-appeal-status")){const h=m.getAttribute("data-status");g&&h&&await Ke({id:Number(g),status:h});
return}if(m.classList.contains("ban-appeal-reply-send")){const h=m.closest("[data-appeal-id]"),x=h?h.
querySelector(".ban-appeal-reply"):null,S=x?x.value:"";g&&await Ke({id:Number(g),admin_reply:S});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const h=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await Ke({id:Number(
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
"marker-tool-crop")&&(get("marker-tool-crop").onclick=()=>setMarkerMode("crop"));const En=get("marke\
r-color-picker");En&&(En.oninput=d=>setMarkerColor(d.target.value),En.onchange=d=>setMarkerColor(d.target.
value));const $n=get("marker-opacity");$n&&($n.oninput=d=>setMarkerOpacity(d.target.value),$n.onchange=
d=>setMarkerOpacity(d.target.value));const vn=get("marker-opacity-number");vn&&(vn.onchange=d=>setMarkerOpacity(
d.target.value),vn.onblur=d=>setMarkerOpacity(d.target.value),vn.onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(
d.target.value),d.target.blur())}),document.querySelectorAll("#marker-toolbar .marker-color-chip[dat\
a-marker-color]").forEach(d=>{d.onclick=()=>setMarkerColor(d.getAttribute("data-marker-color"))}),get(
"marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),get("marker-crop\
-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&(get("marker-un\
do").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),
get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),
initCropCanvas(),window.addEventListener("resize",()=>{const d=get("marker-modal");!d||d.classList.contains(
"hidden")||(applyMarkerTransform(),renderCropOverlay())});const Li=o(()=>{const d=get("upload-modal");
return!!(d&&!d.classList.contains("hidden"))},"isUploadModalOpen"),Gt=get("drop-overlay");let nn=0;const Ci=o(
()=>{Li()||Gt&&(Gt.classList.remove("hidden"),Gt.classList.add("flex"))},"showDropOverlay"),sn=o(()=>{
nn=0,Gt&&(Gt.classList.add("hidden"),Gt.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
sn;const xt=get("upload-dropzone");xt&&(xt.addEventListener("dragover",d=>{d.preventDefault(),xt.classList.
add("dragover")}),xt.addEventListener("dragleave",()=>{xt.classList.remove("dragover")}),xt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),xt.classList.remove("dragover"),sn();const m=d.dataTransfer?
d.dataTransfer.files:null;m&&m.length&&handleFiles(m)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(nn+=1,Ci())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(nn=Math.max(0,nn-1),(nn===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&sn())}),window.addEventListener(
"dragend",()=>{sn()}),window.addEventListener("drop",d=>{sn(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(xt&&xt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const Yn=get("bot-admin-modal"),Ti=o(d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML=
"",!d||!d.length){m.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,h)=>{const x=!!g.is_bot_banned,S=g.bot_detection_enabled!==!1,L=Number(g.lock_remaining_seconds)||
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
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${S?"1":"0"}">${S?"\u691C\u51FAON":
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
"error",!0)}},"loadBotUsers"),In=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Yn))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await an(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=In;const Pn=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=Pn,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{Pn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||Yn)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
In()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
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
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Ut={"/settings":{id:"settings-mod\
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
dal",open:o(()=>In(),"open")},"/admin-bot-logs":{id:"bot-log-modal",open:o(()=>Pn(),"open")}},Qn=o((d,m=!1)=>{
switch(d){case"settings-modal":Ht(m);break;case"upload-modal":closeUploadModal(m);break;case"image-s\
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
"popstate",d=>{let m=!1;Object.values(Ut).forEach(x=>{const S=get(x.id);S&&S.classList.contains("mod\
al-open")&&location.pathname!==Object.keys(Ut).find(L=>Ut[L].id===x.id)&&(Qn(x.id,!0),m=!0)});const g=location.
pathname.match(/^\/c\/(.+)$/);if(g){const x=decodeURIComponent(g[1]);String(currentThreadId)!==String(
x)&&loadMessages(x,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});
const h=Ut[location.pathname];if(h){const x=get(h.id);x&&!x.classList.contains("modal-open")&&h.open()}});
const Zn=location.pathname;Ut[Zn]&&(history.replaceState({},"","/"),setTimeout(()=>Ut[Zn].open(),500)),
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
h.forEach((S,L)=>{const M=document.createElement("div");M.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const A=document.createElement("img");A.src=S.url,A.alt=S.file.name,
A.className="w-full h-full object-cover";const N=document.createElement("button");N.type="button",N.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",N.textContent=
"\xD7",N.onclick=()=>{URL.revokeObjectURL(S.url),h.splice(L,1),x()},M.appendChild(A),M.appendChild(N),
g.appendChild(M)}))},"render");return d&&m&&(d.onclick=()=>m.click(),m.onchange=S=>{for(const L of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(L.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(L.
size>Ai){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(h.
length>=Mi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}h.push({file:L,url:URL.createObjectURL(L)})}S.target.value="",x()}),{files:o(()=>h.
map(S=>S.file),"files"),clear:o(()=>{h.forEach(S=>URL.revokeObjectURL(S.url)),h=[],x()},"clear")}}o(
ei,"createFbImagePicker");async function ti(d,m){try{const g=new FormData;return m.forEach(h=>g.append(
"images",h,h.name)),(await apiFetch(d,{method:"POST",body:g})).ok}catch{return!1}}o(ti,"uploadFbImag\
es");const ni=ei(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-message").value.trim();if(!m){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:m,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},h=get("fb-attach-logs"),x=h&&h.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;x&&(g.client_logs=x);const S=get("fb-attach-chat"),L=!!(S&&S.checked);if(L&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}L&&(g.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const M=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),A=M.ok?await M.clone().json().
catch(()=>({})):{},N=ni.files(),R={count:N.length,saved:!0};M.ok&&R.count&&(R.saved=await ti(`/api/f\
eedback/${A.public_id}/images`,N)),(window.ActivityLog?await window.ActivityLog.reportFeedback(M,x,L,
R):M.ok)&&(get("fb-title").value="",get("fb-message").value="",ni.clear(),S&&(S.checked=!1),h&&(h.checked=
!0),on())};async function ii(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),on()}o(ii,"deleteFeedback");const Ei="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let On=[];function si(d,m){const g=d.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(h=>{const x=h.role==="admin"?m&&h.author?`\u7BA1\u7406\u8005 (${h.
author})`:"\u7BA1\u7406\u8005":m?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=m?[h.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${h.
log_file}`,h.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${h.image_count}\u679A\uFF09: ${h.image_dir}`]:
[h.image_count&&`\u6DFB\u4ED8\u753B\u50CF: ${h.image_count}\u679A`];return`<div class="text-[11px]">\
<div class="text-gray-400">${escapeHtml(x)} / ${escapeHtml(h.created_at||"")}</div><div class="text-\
sm whitespace-pre-wrap ${h.role==="admin"?"text-green-300":"text-gray-200"}">${escapeHtml(h.message)}\
</div>${S.filter(Boolean).map(L=>`<div class="${m?"text-amber-300":"text-gray-400"}">${escapeHtml(L)}\
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
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());h.disabled=!S,x.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const L=ei(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));On.push(L);const M=g.querySelector(".fb-reply-send");M.onclick=async()=>{const A=g.
querySelector(".fb-reply-message").value.trim();if(!A){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const N={message:A,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},R=h.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;R&&(N.client_logs=R);const X=L.
files();M.disabled=!0;try{const ae=await apiFetch(`/api/feedback/${d.public_id}/replies`,{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(N)}),ie=await ae.json().catch(()=>({}));
if(!ae.ok){showToast(ie.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const Y=!X.length||await ti(`/api/feedback/${d.public_id}/replies/${ie.reply_id}/\
images`,X),pe=[];if(R&&ie.logs_saved===!1&&pe.push("\u30ED\u30B0"),Y||pe.push("\u753B\u50CF"),pe.length)
showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${pe.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\uFF09`,"error",!0);else{const H=["\u8FD4\u4FE1"];R&&H.push(`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof ie.
logs_count=="number"?ie.logs_count:R.entries.length}\u4EF6\uFF09`),X.length&&H.push(`\u753B\u50CF\uFF08${X.
length}\u679A\uFF09`),showToast(H.join(H.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F",
"success")}on()}finally{M.disabled=!1}}}o(oi,"bindFeedbackReply");function ri(){const d=get("fb-admi\
n-status-filter").value;let m=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(
g=>{const h=d==="all"||(d==="unresolved"?["new","in_review","replied"].includes(g.dataset.feedbackStatus):
g.dataset.feedbackStatus===d);g.hidden=!h,h&&m++}),get("fb-admin-empty").classList.toggle("hidden",m!==
0)}o(ri,"filterAdminFeedback"),get("fb-admin-status-filter").onchange=ri;async function on(){On.forEach(
S=>S.clear()),On=[];const m=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML=
"",(m.items||[]).filter(S=>!m.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(m.is_admin)
return;const L=document.createElement("div");L.className="p-2 rounded border border-gray-700 bg-gray\
-800/50",L.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
S.public_id||"")}</span> / ${S.created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.
image_count}\u679A</div>`:""}${si(S,!1)}${ai()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
L.querySelector(".fb-delete").onclick=()=>ii(S.public_id),oi(S,L),g.appendChild(L)});const h=get("fb\
-admin-panel"),x=get("fb-admin-list");m.is_admin?(h.classList.remove("hidden"),x.innerHTML="",(m.items||
[]).forEach(S=>{const L=document.createElement("div");L.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",L.dataset.feedbackStatus=S.status||"new",L.innerHTML=`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(S.public_id||"")}</span> / ${S.agent_name?`agent:${escapeHtml(S.agent_name)}`:`us\
er:${S.user_id}`} / ${S.created_at}${S.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(S.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div>
                            ${[S.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${S.log_file}`,S.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${S.chat_file}`,S.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${S.
image_count}\u679A\uFF09: ${S.image_dir}`].filter(Boolean).map(M=>`<div class="text-[11px] text-ambe\
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
                            ${si(S,!0)}
                            ${ai()}
                        `,L.querySelector(".fb-status").value=S.status||"new",L.querySelector(".fb-s\
ave").onclick=async()=>{const M=L.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:M})}),
on()},L.querySelector(".fb-delete").onclick=()=>ii(S.public_id),oi(S,L),x.appendChild(L)}),ri()):h.classList.
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
stringify(Object.assign({},L,{name:g}))}),A=await M.json().catch(()=>({}));M.ok?(m&&(m.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(A.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
S=>S.name.toLowerCase().includes(h)||S.description&&S.description.toLowerCase().includes(h));x[gemSelectedIndex]?
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
let nt,Re=null,wn=[],Nn=!1,xn=null,zt=null,It=null,ln=null,Rn=0,kn=!1,cn=null,Wt=null,Lt=null,dn=null,
_n=null,Pt=null,Sn=null,Ln=null;function li(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
Sn)&&Sn.length)return Sn;d.innerHTML="";const m=[];for(let g=0;g<24;g++){const h=document.createElement(
"span");h.className="block rounded-full",h.style.background="rgba(252, 165, 165, 0.92)",h.style.width=
"2px",h.style.transition="height 75ms linear, opacity 75ms linear",h.style.height="2px",h.style.opacity=
"0.4",m.push(h),d.appendChild(h)}return Sn=m,m}o(li,"ensureMicWaveformBars");function Ot(d,m="hidden"){
const g=get("mic-recording-indicator"),h=get("mic-recording-text");if(g){if(Ln&&(clearTimeout(Ln),Ln=
null),m==="hidden"){g.classList.add("hidden");return}h&&d&&(h.innerText=d),g.classList.remove("hidde\
n"),m==="recording"?g.style.color="rgb(252 165 165)":m==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(Ot,"setMicRecordingIndicator");function ci(){li().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}o(ci,"resetMicWaveformBars");function Nt(){if(_n&&(cancelAnimationFrame(
_n),_n=null),dn){try{dn.disconnect()}catch{}dn=null}if(Wt){try{Wt.close()}catch{}Wt=null}Lt=null,Pt=
null,ci()}o(Nt,"stopMicWaveform");function di(d){Nt();const m=li();if(!m.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{Wt=new g,Lt=Wt.createAnalyser(),Lt.fftSize=
256,Lt.smoothingTimeConstant=0,dn=Wt.createMediaStreamSource(d),dn.connect(Lt),Pt=new Uint8Array(Lt.
frequencyBinCount)}catch{Nt();return}const h=o(()=>{if(!Lt||!Pt)return;Lt.getByteFrequencyData(Pt);const x=Math.
max(1,Math.floor(Pt.length/m.length));for(let S=0;S<m.length;S++){const M=(Pt[Math.min(Pt.length-1,S*
x)]||0)/255,A=Math.max(2,Math.round(2+M*10));m[S].style.height=`${A}px`,m[S].style.opacity=`${.35+M*
.65}`}_n=requestAnimationFrame(h)},"render");h()}o(di,"startMicWaveform");function Cn(){if(xn&&(clearInterval(
xn),xn=null),ln){try{ln.disconnect()}catch{}ln=null}if(zt){try{zt.close()}catch{}zt=null}It=null}o(Cn,
"stopSilenceMonitor");function ui(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;Cn();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;zt=new m,It=zt.createAnalyser(),It.fftSize=2048,
ln=zt.createMediaStreamSource(d),ln.connect(It);const g=new Uint8Array(It.fftSize),h=getStsSilenceMs(),
x=.02;Rn=0,kn=!1,xn=setInterval(()=>{if(!It)return;It.getByteTimeDomainData(g);let S=0;for(let M=0;M<
g.length;M++){const A=(g[M]-128)/128;S+=A*A}if(Math.sqrt(S/g.length)>x){kn||(kn=!0),Rn=Date.now();return}
kn&&Date.now()-Rn>h&&nt&&nt.state==="recording"&&nt.stop()},200)}o(ui,"startSilenceMonitor");const jn=class jn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(m,g,h,x={},S=null){if(this.
model=h,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(m)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const A=!!(x&&
x.transcriptionConfig),N={setup:{model:`models/${h}`,generationConfig:{responseModalities:A?["TEXT"]:
["AUDIO"]},inputAudioTranscription:A?x.transcriptionConfig||{}:{}}};A||(N.setup.outputAudioTranscription=
{}),x.speechConfig&&(N.setup.generationConfig.speechConfig=x.speechConfig),x.thinkingConfig&&(N.setup.
generationConfig.thinkingConfig=x.thinkingConfig),x.translationConfig&&(N.setup.generationConfig.translationConfig=
x.translationConfig),console.log("Sending setup:",JSON.stringify(N)),this.ws.send(JSON.stringify(N))},
this.ws.onmessage=A=>this._handleMessage(A),this.ws.onerror=A=>{console.error("Gemini Live WebSocket\
 error:",A),this.onError&&this.onError(A)},this.ws.onclose=A=>{console.log("Gemini Live WebSocket cl\
osed:",A.code,A.reason),this.closedEvent=A,this.onClose&&this.onClose(A)},S&&S.stream&&S.ctx&&S.source){
this.stream=S.stream,this.audioContext=S.ctx;var L=S.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(Tn());const A=mi(this.stream,16e3);this.audioContext=A.ctx;var L=A.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const M=new MediaRecorder(this.stream);M.ondataavailable=A=>{A.data.
size>0&&this.userAudioChunks.push(A.data)},M.start(500),this.backupRecorder=M,this.processor.onaudioprocess=
A=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const N=pi(A.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!N||!N.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(N))),mimeType:"audio/\
pcm;rate=16000"}}}))},L.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
let g=null;try{g=JSON.parse(typeof m.data=="string"?m.data:new TextDecoder().decode(m.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const h=g.serverContent;h.modelTurn&&h.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log(
"Gemini thought delta:",x.text),this.assistantThought+=x.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(x.text)),x.inlineData&&x.inlineData.data){const S=x.inlineData.data;
console.log("Gemini audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);
const L=atob(S),M=new Uint8Array(L.length);for(let A=0;A<L.length;A++)M[A]=L.charCodeAt(A);this.assistantAudioChunks.
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
null,this._stopped=new Promise(h=>{let x=!1;const S=o(()=>{x||(x=!0,h())},"finish");if(!m||m.state===
"inactive"){S();return}m.addEventListener("stop",S,{once:!0});try{m.stop()}catch{S()}setTimeout(S,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=o(
()=>{if(this.stream&&this.stream.getTracks().forEach(h=>{try{h.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const m=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(m),h=new Blob(
this.userAudioChunks),x=await this._blobToBase64(h);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:x}}_blobToBase64(m){
return new Promise(g=>{const h=new FileReader;h.onloadend=()=>g(h.result.split(",")[1]),h.readAsDataURL(
m)})}};o(jn,"GeminiLiveClient");let Bn=jn;const Dn=class Dn{constructor(m=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:m}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(m){if(!this.ctx)return;const g=atob(m),h=new Uint8Array(g.length);for(let N=0;N<
g.length;N++)h[N]=g.charCodeAt(N);const x=new Int16Array(h.buffer),S=new Float32Array(x.length);for(let N=0;N<
x.length;N++)S[N]=x[N]/32768;const L=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);L.getChannelData(
0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const M=this.ctx.createBufferSource();
M.buffer=L,M.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const A=Math.max(this.ctx.currentTime,this.nextStartTime);M.start(A),this.
nextStartTime=A+L.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Dn,"RealTimeAudioPl\
ayer");let un=Dn;const Hn=class Hn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
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
value:x}=await this.reader.read();if(h)break;g+=m.decode(x,{stream:!0});let S;for(;(S=g.indexOf(`

`))>=0;){const L=g.slice(0,S);g=g.slice(S+2);for(const M of L.split(`
`)){if(!M.startsWith("data: "))continue;let A=null;try{A=JSON.parse(M.slice(6))}catch{continue}this.
_handleEvent(A)}}}}catch(h){if(h&&h.name==="AbortError")return;this.active&&(this.streamError=h&&h.message?
h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new un(this.rateOut||24e3),Vt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Vt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(Tn()).then(m=>{this.stream=m;const g=this.rateIn||24e3,h=mi(m,g);this.audioCtx=
h.ctx;const x=h.source,S=this.audioCtx.sampleRate,L=4096;this.processor=this.audioCtx.createScriptProcessor(
L,1,1),this.processor.onaudioprocess=M=>{if(!this.active||!this.capturing)return;const A=M.inputBuffer.
getChannelData(0),N=pi(A,S,g);!N||!N.byteLength||this._sendAudio(N)},x.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(m){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(m)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const m=this.audioQueue.splice(0,this.audioQueue.length),
g=m.reduce((A,N)=>A+N.byteLength,0),h=new Uint8Array(g);let x=0;m.forEach(A=>{h.set(A,x),x+=A.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),L={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:h.buffer},
M=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(L):L;try{await fetch(S,M)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}Cn(),Nt()}async stop(){if(!this.active)return;if(this.
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
Hn,"RealtimeVoiceSession");let Fn=Hn;function mi(d,m){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let h=null;try{return h=new g({sampleRate:m}),{ctx:h,
source:h.createMediaStreamSource(d)}}catch{if(h)try{h.close()}catch{}return h=new g,{ctx:h,source:h.
createMediaStreamSource(d)}}}o(mi,"openMicAudioSource");function pi(d,m,g){let h=d;if(m!==g&&m>0&&g>
0){const S=m/g,L=Math.floor(h.length/S),M=new Float32Array(L);for(let A=0;A<L;A++)M[A]=h[Math.min(Math.
floor(A*S),h.length-1)];h=M}const x=new Int16Array(h.length);for(let S=0;S<h.length;S++){const L=Math.
max(-1,Math.min(1,h[S]));x[S]=L<0?L*32768:L*32767}return x.buffer}o(pi,"pcm16FromFloat32");const mn=new Fn,
fi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let h=null,x=null,S=!1,L=null,M=!1,A=0,N=0,R=null,X="idle",ae=!1,ie=null;const Y=o(F=>document.getElementById(
F),"$"),pe=o(F=>{const Z=Object.assign({},F||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(Z):(Z.progressSpinner=!1,
Z)},"noSpinner");function H(F,Z){X=Z;const Ce=Y("lyria-status-text"),de=Y("lyria-status-dot");Ce&&(Ce.
textContent=F),de&&(de.className="w-2 h-2 rounded-full inline-block "+(g[Z]||g.idle)),ve(),te()}o(H,
"setStatus");function fe(){const F=N?Math.floor((Date.now()-N)/1e3):0,Z=String(Math.floor(F/60)).padStart(
2,"0"),Ce=String(F%60).padStart(2,"0");return`${Z}:${Ce}`}o(fe,"formatElapsed");function V(){N||(N=Date.
now());const F=Y("lyria-elapsed");F&&(F.textContent=fe()),R||(R=window.setInterval(()=>{const Z=Y("l\
yria-elapsed");Z&&(Z.textContent=fe())},1e3))}o(V,"startElapsedTimer");function ue(){R&&(window.clearInterval(
R),R=null)}o(ue,"stopElapsedTimer");function ve(){const F=Y("lyria-play-btn"),Z=Y("lyria-pause-btn"),
Ce=Y("lyria-stop-btn"),de=Y("lyria-reset-btn"),Me=!!h,je=X==="streaming"||X==="connecting";if(F){F.disabled=
ae;const Ye=F.querySelector("i");Ye&&(Ye.className="fas fa-play")}Z&&(Z.disabled=ae||!je),Ce&&(Ce.disabled=
ae||!Me||!je),de&&(de.disabled=ae||!Me||!je)}o(ve,"updateTransportButtons");function te(){const F=Y(
"lyria-save-btn");if(!F)return;const Z=!!h&&X!=="idle"&&X!=="connecting"&&X!=="error";F.classList.toggle(
"hidden",!Z)}o(te,"updateSaveButton");function ge(F,Z){const Ce=Y("lyria-prompt-rows");if(!Ce)return;
const de=document.createElement("div");de.className="flex items-center gap-2",de.innerHTML=`
                        <input type="text" value="${escapeHtml(F||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof Z=="num\
ber"?Z:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof Z=="number"?Z:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Me=de.querySelector('input[type="range"]'),je=de.querySelector(".lyria-w\
eight-label");Me&&je&&Me.addEventListener("input",()=>{je.textContent=parseFloat(Me.value).toFixed(1)});
const Ye=de.querySelector(".lyria-prompt-remove");Ye&&Ye.addEventListener("click",()=>{Ce.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||de.remove()}),de.classList.add("lyria-prompt-row-wrap"),Ce.appendChild(
de)}o(ge,"addPromptRow");function Ie(){const F=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),Z=[];return F.forEach(Ce=>{const de=Ce.querySelector('input[type="text"]'),Me=Ce.querySelector(
'input[type="range"]'),je=(de?de.value:"").trim();je&&Z.push({text:je,weight:parseFloat(Me?Me.value:
1)||1})}),Z}o(Ie,"collectPrompts");function Pe(){const F={},Z=o(Ni=>{const Gn=Y(Ni);return Gn&&Gn.value!==
""?parseFloat(Gn.value):void 0},"num"),Ce=Z("lyria-bpm");Ce!==void 0&&(F.bpm=Math.round(Ce));const de=Z(
"lyria-guidance");de!==void 0&&(F.guidance=de);const Me=Z("lyria-density");Me!==void 0&&(F.density=Me);
const je=Z("lyria-brightness");je!==void 0&&(F.brightness=je);const Ye=Z("lyria-temperature");Ye!==void 0&&
(F.temperature=Ye);const Ue=Y("lyria-scale");Ue&&Ue.value&&(F.scale=Ue.value);const et=Y("lyria-mode");
et&&et.value&&(F.music_generation_mode=et.value);const dt=Y("lyria-mute-bass"),Rt=Y("lyria-mute-drum\
s"),wi=Y("lyria-only-bass-drums");return dt&&(F.mute_bass=dt.checked),Rt&&(F.mute_drums=Rt.checked),
wi&&(F.only_bass_and_drums=wi.checked),F}o(Pe,"collectConfig");function it(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([Z,Ce])=>{
const de=Y(Z),Me=Y(Ce);!de||!Me||de.addEventListener("input",()=>{const je=parseFloat(de.value);Me.textContent=
Z==="lyria-bpm"?String(Math.round(je)):je.toFixed(1)})})}o(it,"bindRangeLabels");function ze(){if(L){
try{L.close()}catch{}L=null}M=!1,A=0}o(ze,"resetPlayback");function ct(){if(S=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(ct,"closeStream");async function Ct(){ct(),x=new AbortController,
S=!0;try{const F=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(h)}`,pe({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!F.ok){const Me=await F.
json().catch(()=>({}));throw new Error(Me.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const Z=F.body.getReader(),Ce=new TextDecoder;let de="";for(;S;){const{done:Me,value:je}=await Z.read();
if(Me)break;de+=Ce.decode(je,{stream:!0});const Ye=de.split(`

`);de=Ye.pop();for(const Ue of Ye){const et=Ue.split(`
`).find(Rt=>Rt.startsWith("data: "));if(!et)continue;const dt=et.slice(6);try{const Rt=JSON.parse(dt);
Le(Rt)}catch{}}}}catch(F){if(F&&F.name==="AbortError")return;S&&(H("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&h&&Ct()},1200))}finally{S=!1}}o(Ct,"openStream");function Le(F){
if(F&&F.snapshot){const Z=F.status;if(Z==="error"){H("\u30A8\u30E9\u30FC","error"),ue();return}if(Z===
"closed"||Z==="stopped"){H("\u7D42\u4E86","closed"),ue();return}H(Z==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",Z==="paused"?"paused":"connecting");return}if(F&&F.audio){H("\u518D\u751F\u4E2D...",
"streaming"),V(),we(F.audio);return}if(F&&F.error){H("\u30A8\u30E9\u30FC: "+F.error,"error"),ue();return}
if(F&&F.final){H("\u7D42\u4E86","closed"),ue(),ve();return}}o(Le,"handleStreamMessage");function we(F){
if(!F)return;if(!L){const Ue=window.AudioContext||window.webkitAudioContext;if(!Ue)return;L=new Ue({
sampleRate:48e3}),M=!1,A=0}let Z;try{const Ue=atob(F);Z=new Uint8Array(Ue.length);for(let et=0;et<Ue.
length;et++)Z[et]=Ue.charCodeAt(et)}catch{return}const Ce=new Int16Array(Z.buffer),de=Math.floor(Ce.
length/2);if(de<1)return;const Me=L.createBuffer(2,de,48e3);for(let Ue=0;Ue<2;Ue++){const et=Me.getChannelData(
Ue);for(let dt=0;dt<de;dt++)et[dt]=Ce[dt*2+Ue]/32768}L.state==="suspended"&&L.resume();const je=L.createBufferSource();
je.buffer=Me,je.connect(L.destination),M||(A=L.currentTime+.08,M=!0);const Ye=Math.max(L.currentTime,
A);je.start(Ye),A=Ye+Me.duration}o(we,"playChunk");async function Fe(F,Z){const Ce=await fetch("/api\
/gemini/music/command",pe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:h,type:F},Z||{}))})),de=await Ce.json().catch(()=>({}));if(!Ce.ok)throw new Error(
de.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return de}
o(Fe,"apiCommand");async function Xe(){if(ae)return;const F=Ie();if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ae=!0,ve(),H("\u63A5\u7D9A\u4E2D...","connecting");try{
const Z=await fetch("/api/gemini/music/start",pe({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:F,config:Pe()})})),Ce=await Z.json().catch(()=>({}));if(!Z.
ok)throw new Error(Ce.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
h=Ce.session_id,ie=Pe(),H("\u63A5\u7D9A\u4E2D...","connecting"),Ct()}catch(Z){H("\u30A8\u30E9\u30FC: "+
Z.message,"error"),showToast("Lyria RealTime: "+Z.message,"error",!0)}finally{ae=!1,ve()}}o(Xe,"star\
tSession");async function st(F){if(h){ae=!0,ve();try{await Fe("control",{action:F}),F==="PLAY"?H("\u518D\u751F\
\u4E2D...","streaming"):F==="PAUSE"?H("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):F==="STOP"?H("\u505C\u6B62\u4E2D",
"stopped"):F==="RESET_CONTEXT"&&H("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(Z){showToast("Lyria RealTime: "+Z.message,"error",!0),H("\u30A8\u30E9\u30FC: "+Z.
message,"error")}finally{ae=!1,ve()}}}o(st,"control");async function _e(){if(!h)return;const F=Ie();
if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ae=!0;try{await Fe("prompts",{weighted_prompts:F}),H("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
X==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(Z){showToast("Lyria RealTime: "+Z.message,"error",!0)}finally{ae=!1,ve()}}o(_e,"app\
lyPrompts");async function Oe(){if(!h)return;const F=Pe(),Z=ie||{},Ce=F.bpm!==void 0&&F.bpm!==Z.bpm,
de=F.scale!==void 0&&F.scale!==Z.scale,Me=Ce||de;ae=!0;try{await Fe("config",{config:F,reset_context:Me}),
ie=F,H(Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",X==="paused"?"paused":"streaming"),showToast(
Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(je){showToast("Lyria RealT\
ime: "+je.message,"error",!0)}finally{ae=!1,ve()}}o(Oe,"applyConfig");async function qe(){if(h){ae=!0,
H("\u4FDD\u5B58\u4E2D...","connecting"),ve();try{const F=await fetch("/api/gemini/music/save",pe({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h,thread_id:currentThreadId||
null})})),Z=await F.json().catch(()=>({}));if(!F.ok)throw new Error(Z.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
H("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ue(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),Z.thread_id&&(currentThreadId=String(Z.thread_id),history.pushState({},"","/c/"+Z.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(Z.thread_id||currentThreadId),fn(!0)}catch(F){
H("\u30A8\u30E9\u30FC: "+F.message,"error"),showToast("Lyria RealTime: "+F.message,"error",!0)}finally{
ae=!1,ve()}}}o(qe,"saveSession");async function We(){if(ct(),h)try{await fetch("/api/gemini/music/ca\
ncel",pe({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h})}))}catch{}
h=null,ue(),ze(),H("\u6E96\u5099\u5B8C\u4E86","idle")}o(We,"cancelSession");function lt(){const F=Y(
"lyria-prompt-rows");F&&(F.innerHTML=""),ge("",1),ie=null,N=0,["lyria-bpm","lyria-guidance","lyria-d\
ensity","lyria-brightness","lyria-temperature"].forEach(de=>{const Me=Y(de);Me&&(Me.value=de==="lyri\
a-bpm"?"120":de==="lyria-guidance"?"4":de==="lyria-temperature"?"1.1":"0.5")});const Z=Y("lyria-scal\
e");Z&&(Z.value="");const Ce=Y("lyria-mode");Ce&&(Ce.value="QUALITY"),["lyria-mute-bass","lyria-mute\
-drums","lyria-only-bass-drums"].forEach(de=>{const Me=Y(de);Me&&(Me.checked=!1)}),it()}o(lt,"resetC\
ontrols");function fn(F){ct(),h&&fetch("/api/gemini/music/cancel",pe({method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({session_id:h})})).catch(()=>{}),h=null,S=!1,ue(),ze(),
hideModal("lyria-studio-modal")}o(fn,"closeAndCleanup");function Jt(F){if(!isLyriaRealtimeModel()){showToast(
"Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const Z=Y("lyria-studio-modal");if(Z&&Z.classList.contains("modal-open")&&h){if(F&&
typeof F=="string"){const de=Y("lyria-prompt-rows");de&&(de.innerHTML=""),ge(F,1)}return}if(h&&We(),
lt(),F&&typeof F=="string"){const de=Y("lyria-prompt-rows");de&&(de.innerHTML=""),ge(F,1)}h=null,S=!1,
ue(),ze(),H("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Jt,"open");function gn(){
const F=Y("lyria-open-studio-btn");F&&F.addEventListener("click",()=>Jt(""));const Z=Y("lyria-studio\
-close");Z&&Z.addEventListener("click",()=>fn(!1));const Ce=Y("lyria-play-btn");Ce&&Ce.addEventListener(
"click",()=>{if(!h){Xe();return}st("PLAY")});const de=Y("lyria-pause-btn");de&&de.addEventListener("\
click",()=>st("PAUSE"));const Me=Y("lyria-stop-btn");Me&&Me.addEventListener("click",()=>st("STOP"));
const je=Y("lyria-reset-btn");je&&je.addEventListener("click",()=>st("RESET_CONTEXT"));const Ye=Y("l\
yria-add-prompt-btn");Ye&&Ye.addEventListener("click",()=>ge("",1));const Ue=Y("lyria-apply-prompts-\
btn");Ue&&Ue.addEventListener("click",_e);const et=Y("lyria-apply-config-btn");et&&et.addEventListener(
"click",Oe);const dt=Y("lyria-save-btn");dt&&dt.addEventListener("click",qe),it(),lt(),window.openLyriaStudio=
Jt}o(gn,"init");function qn(){ae||fn(!1)}return o(qn,"requestClose"),{init:gn,open:Jt,requestClose:qn}})();
fi.init(),window.closeLyriaStudio=()=>fi.requestClose(),(()=>{let d=null,m=null,g=null;const h="voic\
eDockSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=o(te=>document.getElementById(te),"$");function L(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(L,"isStudioMode");function M(){const te=get("model-select")?get("model-select").value:"",ge=S("voi\
ce-studio-title");ge&&(te==="gpt-transcribe"||te==="gpt-live-transcribe"||te==="grok-voice-transcrib\
e-2.0-file"?ge.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":te===
"gemini-3.5-live-translate-preview"?ge.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ge.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(M,"updateTitle");function A(){const te=S("v\
oice-studio-transcript");te&&(te.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${x}</div>`);const ge=S("sts-live-transcript");ge&&(ge.innerHTML="",ge.classList.add("hidden"))}
o(A,"resetTranscript");function N(te,ge,Ie){const Pe=te.querySelectorAll(".voice-studio-line");let it=null;
for(let ze=Pe.length-1;ze>=0;ze--)if(Pe[ze].dataset.role===ge){it=Pe[ze];break}if(it)it.innerHTML=Ie;else{
const ze=te.querySelector(".voice-studio-placeholder");ze&&ze.remove();const ct=document.createElement(
"div");ct.className="voice-studio-line",ct.dataset.role=ge,ct.innerHTML=Ie,te.appendChild(ct)}te.classList.
remove("hidden"),te.scrollTop=te.scrollHeight}o(N,"writeLine");function R(te,ge){if(!ge||!String(ge).
trim()||!L())return;const it=`<span class="${te==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(te==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
ge)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(ze=>N(ze,
te,it))}o(R,"log");function X(te,ge=!0){const Ie=S("sts-panel"),Pe=S("sts-settings-toggle");if(Ie&&Ie.
classList.toggle("settings-open",!!te),Pe&&Pe.setAttribute("aria-expanded",te?"true":"false"),ge)try{
localStorage.setItem(h,te?"1":"0")}catch{}}o(X,"setSettingsOpen");function ae(){try{return localStorage.
getItem(h)==="1"}catch{return!1}}o(ae,"readSettingsOpen");let ie=null;function Y(){const te=get("mod\
el-select")?get("model-select").value:"";te!==ie&&(ie=te,A()),X(ae(),!1),M()}o(Y,"syncDock");function pe(){
const te=S("sts-panel"),ge=S("voice-studio-panel-host");te&&ge&&te.parentNode!==ge&&(d=te.parentNode,
m=te.nextSibling,ge.appendChild(te));const Ie=S("file-preview"),Pe=S("voice-studio-file-host");Ie&&Pe&&
Ie.parentNode!==Pe&&(g=Ie.parentNode,Pe.appendChild(Ie),Pe.classList.remove("hidden"))}o(pe,"movePan\
elIntoModal");function H(){const te=S("sts-panel");te&&d&&te.parentNode!==d&&(m&&m.parentNode===d?d.
insertBefore(te,m):d.appendChild(te));const ge=S("file-preview");ge&&g&&ge.parentNode!==g&&g.appendChild(
ge);const Ie=S("voice-studio-file-host");Ie&&Ie.classList.add("hidden"),d=null,m=null,g=null}o(H,"mo\
vePanelBack");function fe(){if(!L())return;pe();const te=S("sts-panel");te&&te.classList.remove("hid\
den"),M(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(fe,"open");function V(){window.
VoiceStudioOpen=!1,H(),hideModal("voice-studio-modal")}o(V,"close");function ue(){window.VoiceStudioOpen&&
V()}o(ue,"closeIfOpen");function ve(){window.VoiceStudioOpen=!1;const te=S("voice-studio-open-btn");
te&&te.addEventListener("click",()=>fe());const ge=S("voice-studio-close");ge&&ge.addEventListener("\
click",()=>V());const Ie=S("sts-settings-toggle");Ie&&Ie.addEventListener("click",()=>{const Pe=S("s\
ts-panel");X(!(Pe&&Pe.classList.contains("settings-open")))}),window.VoiceStudio={open:fe,close:V,closeIfOpen:ue,
log:R,isStudioMode:L,syncDock:Y},Y()}return o(ve,"init"),{init:ve}})().init();let Vt=null;function gi(){
if(Vt&&(Vt.stop(),Vt=null),cn){try{cn.pause()}catch{}try{cn.src=""}catch{}cn=null}}o(gi,"stopStsPlay\
back");async function Fi(d){gi();const m=new Audio;return m.src=d,m.preload="auto",m.autoplay=!0,m.playsInline=
!0,cn=m,await m.play(),new Promise(g=>{m.onended=()=>g("ended"),m.onerror=()=>g("error")})}o(Fi,"pla\
yStsAudio");function $i(){if(mn.isActive()){mn._cancel();return}if(Re){Re.stop(),Re=null,gi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Nt();return}nt&&nt.state===
"recording"&&(Nn=!0,nt.stop())}o($i,"cancelRecording");function Tn(){if(isStsModel())return{audio:!0};
const m=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return m.echoCancellation&&(g.echoCancellation=!1),m.
noiseSuppression&&(g.noiseSuppression=!1),m.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(Tn,"\
getMicCaptureConstraints");function Ii(){const d=window.AudioContext||window.webkitAudioContext;if(!d||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let m=null;try{m=new d({sampleRate:16e3})}catch{if(m){try{m.close()}catch{}m=null}m=new d}const g=m.
resume(),h=navigator.mediaDevices.getUserMedia(Tn());return{ctx:m,resumePromise:g,streamPromise:h}}o(
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
"").trim()||(m.assistant_text||"").trim()||(m.assistant_thought||"").trim(),h=!!(m.audio_base64||m.user_audio_base64);
if(!g&&!h){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const S=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(S.id),history.pushState(
{},"","/c/"+S.id),get("welcome-screen").classList.add("hidden")}m.thread_id=currentThreadId,m.model=
get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error("Failed to save G\
emini Live session:",m),setStsStatus("Error saving",!1)}return}if(mn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),mn.stop();return}if(nt&&
nt.state==="recording"){nt.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Ot("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ii();
const x={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(x.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const H=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(fe=>fe.trim()).filter(Boolean);H.length&&(x.custom_vocabulary=
H.slice(0,1e3))}}else x.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(x.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",x.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(x.target_lang=get("sts-target-lang").value);const S=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:S,ctx:g.ctx,source:g.ctx.createMediaStreamSource(S)};const L=await apiFetch(
"/api/gemini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
x)});if(!L.ok)throw new Error("Failed to get session token");const{token:M,url:A}=await L.json(),N=get(
"model-select").value,R=get("sts-voice")?get("sts-voice").value:"Kore",X=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",ae=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(Re=new Bn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Re.rtPlayer=new un),isGeminiLiveTranscribeModel()){
const H=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",fe={languageCodes:[]};
if((H==="SMART"||H==="VERBATIM")&&(fe.mode=H),get("sts-custom-vocab")){const V=get("sts-custom-vocab").
value.split(/[,、\n]/).map(ue=>ue.trim()).filter(Boolean);V.length&&(fe.customVocabulary=V.slice(0,
1e3))}await Re.start(M,A,N,{transcriptionConfig:fe},g)}else if(isGeminiLiveTranslateModel()){const H=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Re.start(M,A,N,{translationConfig:{targetLanguageCode:H,
echoTargetLanguage:!0}},g)}else{const H={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:R}}}};
isGeminiLiveExtendedThinkingModel()&&(H.thinkingConfig={thinkingLevel:X,includeThoughts:ae}),await Re.
start(M,A,N,H,g)}g=null,nt=Re.backupRecorder,nt.onstop=()=>{Re&&get("mic-btn").click()};let ie=!0,Y="\
live-sts-"+Date.now();Re.onMessage=H=>{if(H.serverContent){if(isGeminiLiveTranscribeModel()){const fe=Re.
interimInputTranscript,V=Re.inputTranscript,ue=V+(fe&&!V.endsWith(fe)?(V?`
`:"")+fe:""),ve=get("chat-messages");let te=document.getElementById(Y);te||(te=document.createElement(
"div"),te.id=Y,te.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",te.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ve.appendChild(te),ve.scrollTop=ve.scrollHeight);const ge=te.
querySelector(".message-content");ge.innerText=ue||"\u8074\u304D\u53D6\u308A\u4E2D...",ve.scrollTop=
ve.scrollHeight,window.VoiceStudio&&V&&window.VoiceStudio.log("user",V);return}if(H.serverContent.modelTurn){
ie&&(setStsStatus("Gemini is speaking...",!1),ie=!1);const fe=get("chat-messages");let V=document.getElementById(
Y);V||(V=document.createElement("div"),V.id=Y,V.className="flex flex-col gap-2 mb-4 assistant-messag\
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
                                            `,fe.appendChild(V),fe.scrollTop=fe.scrollHeight);const ue=V.
querySelector(".thought-container"),ve=V.querySelector(".message-content");Re.assistantThought&&(ue.
classList.remove("hidden"),ue.innerText=Re.assistantThought),ve.innerText=Re.assistantText,fe.scrollTop=
fe.scrollHeight,window.VoiceStudio&&(Re.inputTranscript&&window.VoiceStudio.log("user",Re.inputTranscript),
Re.assistantText&&window.VoiceStudio.log("assistant",Re.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),di(Re.stream),ui(Re.stream);const pe=Re;pe.onError=H=>{const fe=H&&H.message?H.message:
typeof H=="string"?H:"";fe&&showToast("Gemini Live: "+fe,"error",!0)},pe.onClose=H=>{if(Re===pe){if(H&&
H.code&&H.code!==1e3){const fe=H.reason?": "+H.reason:" (code "+H.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+fe,"error",!0)}get("mic-btn").click()}},pe.closedEvent&&
pe.onClose(pe.closedEvent);return}catch(h){if(Pi(g),Re){try{Re.stop()}catch{}Re=null}showToast("Gemi\
ni Live connection failed: "+h.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await mn.start();return}isStsModel()||(ci(),Ot("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(Tn());nt=new MediaRecorder(d),wn=[],Nn=!1;const m=isStsModel();
nt.ondataavailable=g=>wn.push(g.data),nt.onstop=async()=>{if(Nn){wn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(L=>L.stop()),Cn(),Nt(),m||(Ot("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),Ln=setTimeout(()=>Ot("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(wn,{type:"audio/web\
m"}),h=new File([g],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",h),get("fil\
e-preview").classList.remove("hidden");const S=m;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const H=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=H.id!==null&&H.id!==void 0?String(H.id):H.id,setTemporaryChatUiState(!!(H&&H.
is_temporary)),setCurrentChatHeaderTitle(H&&H.title),applyTemporaryChatRuntimeMeta(H||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+H.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),x.append("model",get(
"model-select").value),x.append("thread_id",currentThreadId),get("sts-voice")&&x.append("sts_voice",
get("sts-voice").value||""),get("sts-reasoning-effort")&&x.append("sts_reasoning_effort",get("sts-re\
asoning-effort").value||""),get("sts-speed")&&x.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&x.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&x.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&x.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&x.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const L=await apiFetch(
"/sts",{method:"POST",body:x});if(!L.ok){const pe=await L.json().catch(()=>({}));throw new Error(pe.
error||"Speech-to-speech failed")}const M=L.body.getReader(),A=new TextDecoder;let N="",R=null,X=null;
stsOpt("sts-auto-play")&&(X=new un,Vt=X),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let ae=!0,ie="",Y="";for(;;){const{done:pe,value:H}=await M.read();if(pe)break;
N+=A.decode(H,{stream:!0});const fe=N.split(`
`);N=fe.pop();for(const V of fe){if(!V.trim())continue;const ue=JSON.parse(V);if(ue.error)throw new Error(
ue.error);ue.audio_delta&&X&&(ae&&(setStsStatus("Playing response...",!1),ae=!1),await X.addChunk(ue.
audio_delta)),ue.input_delta&&(ie+=ue.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
ie)),ue.transcript_delta&&(Y+=ue.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",Y)),(ue.final||ue.audio_url)&&(R=ue)}}window.VoiceStudio&&!ie.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),R&&(R.audio_url||R.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const L=get(
"set-mic-transcribe-mode");if(!!(L&&L.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const N=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(N.transcript){const R=get("prompt-inp\
ut");R.value+=(R.value?" ":"")+N.transcript,R.style.height="auto",R.style.height=R.scrollHeight+"px"}else
showToast(N.error||"Transcription failed","error",!0)}}catch(L){showToast("Audio processing error: "+
L.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(L=>L.
stop()),Cn(),Nt(),S||Ot("","hidden"),S&&setStsStatus("Tap to speak",!1)}},nt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Ot("\u9332\u97F3\u4E2D\u2026","recording"),di(d)),ui(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Nt(),isStsModel()||Ot("","hidden"),alert("Microphone access denied or not avail\
able.")}};const pn=o((d,m)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),g=get("lib-download-btn"),h=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(m&&(m.disabled=d===0,pn(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,pn(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
h&&(h.disabled=d===0,pn(h,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,pn(x,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=d!==1,pn(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const L=window.matchMedia(
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
"PUT":"POST",S=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
S,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:g,
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
try{const x=decodeURIComponent(g),S=new Blob([x],{type:"text/plain"}),L=URL.createObjectURL(S),M=document.
createElement("a");M.href=L;let N={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[h]||h;(h.length>8||/[^a-z0-9]/.test(h))&&(N="txt");let R=`code.${N}`;h==="dockerfile"&&(R="Docke\
rfile"),h==="makefile"&&(R="Makefile"),M.download=R,document.body.appendChild(M),M.click(),document.
body.removeChild(M),URL.revokeObjectURL(L)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),g=m.getAttribute("data-code");g&&
window.copyCode(m,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&Qn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
