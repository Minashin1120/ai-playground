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
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const o=document.getElementById(
"alpha-bar");setTimeout(()=>{if(o){const d=document.getElementById("version-display");if(d){const m=o.
getBoundingClientRect(),g=d.getBoundingClientRect(),b=g.left+g.width/2-(m.left+m.width/2),k=g.top+g.
height/2-(m.top+m.height/2);o.style.transform=`translate(${b}px, ${k}px) scale(0.1)`,o.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),o.remove()},800)}else o.style.opacity="0",setTimeout(()=>o.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const m=get("gpt-image-format"),g=get("gpt-image-compression-wrap");m&&g&&(m.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}r(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{b&&(Array.from(b.options).
forEach(k=>{k.value!=="1K"&&(k.disabled=g)}),g&&b.value!=="1K"&&(b.value="1K"))})}r(c,"updateGeminiI\
mageUi");function u(){const d=get("grok-image-options");if(!d)return;const m=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),b=m==="grok-imagine-image-quality"||m==="grok-imagine-image-2\
.0",k=m==="grok-imagine-image-2.0";if(g){d.classList.remove("hidden");const T=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const A=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.toggle("hidden",!k)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const T=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidden",!b);const A=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;A&&A.classList.toggle(
"hidden",!k)}}r(u,"updateGrokImageUi");function f(){var N,F;const d=get("ideogram-image-options"),m=get(
"modal-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",b=isIdeogramModel(
g),k=ideogramModelTraits(g),L={resolution:k.resolution,quality:k.quality,speed:k.speed,style:k.style,
negative:k.negative};if(d&&d.classList.toggle("hidden",!b),m&&m.classList.toggle("hidden",!b),updateIdeogramPromptInputUi(),
!b)return;const T=((N=get("ideogram-image-operation"))==null?void 0:N.value)==="remove_background",A=((F=
get("ideogram-image-operation"))==null?void 0:F.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(
Y=>{const oe=Y==="operation"||A&&Y==="seed"||!(T||A)&&L[Y]!==!1;["ideogram-image-","modal-ideogram-i\
mage-"].forEach(le=>{const U=get(le+Y);!U||!U.parentElement||U.parentElement.classList.toggle("hidde\
n",!oe)})}),["ideogram-image-style","modal-ideogram-image-style"].forEach(Y=>{const oe=get(Y);if(!oe)
return;const le=k.styleOptions.join("|");if(oe.dataset.options!==le){const U=oe.value;oe.innerHTML=k.
styleOptions.map(he=>`<option value="${he}">${he==="auto"?"Auto":he}</option>`).join(""),oe.dataset.
options=le,oe.value=k.styleOptions.includes(U)?U:"auto"}});const $=get("ideogram-image-note");$&&($.
textContent=T?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWEBP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
A?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
k.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
r(f,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(d=>{
var m;(m=get(d))==null||m.addEventListener("change",()=>{const g=get(d.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");g&&(g.value=get(d).value),f(),S()})});function h(){
var b;const d=get("grok-video-options");if(!d)return;const m=String(((b=get("model-select"))==null?void 0:
b.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");
const g=get("grok-video-resolution");if(g){const k=Array.from(g.options).find(T=>T.value==="1080p"),
L=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(m);k&&(k.disabled=!L),!L&&g.value===
"1080p"&&(g.value="720p")}}r(h,"updateGrokVideoUi");function y(){var k;const d=get("gemini-video-opt\
ions");if(!d)return;const m=String(((k=get("model-select"))==null?void 0:k.value)||"").toLowerCase();
isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get("gemini-vide\
o-resolution");if(g){const L=Array.from(g.options).find(A=>A.value==="4K"),T=m==="veo-3.1-lite-gener\
ate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";L&&(L.disabled=T),T&&g.value===
"4K"&&(g.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle("hidden",m===
"gemini-omni-1.1-flash")}r(y,"updateGeminiVideoUi");function v(){const d=get("gemini-music-options");
if(!d)return;const m=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!m;d.classList.toggle("hid\
den",!g);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!m)}r(v,"updateGemi\
niMusicUi");function w(){var T;const d=get("xai-chat-options");if(!d)return;const m=String(((T=get("\
model-select"))==null?void 0:T.value)||"").toLowerCase(),g=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!g);const b=get("xai-logp\
robs"),k=get("xai-top-logprobs"),L=m.includes("grok-4.20");b&&(b.disabled=L,L&&(b.checked=!1)),k&&(k.
disabled=L,L&&(k.value=""))}r(w,"updateXaiChatUi");function x(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const k=get(b);k&&(k.classList.toggle("opacity-50",d),k.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}r(x,"updateMistralOcrUi");function _(){const d=get("image-input-limit\
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
r(_,"updateImageInputLimits");function S(){const d=get("enable-sys-prompt"),m=!!(d&&d.disabled),g=!!(d&&
d.checked);M(),d&&(d.disabled?!m&&g?d.dataset.restoreChecked="1":m||delete d.dataset.restoreChecked:
(m&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}r(S,"toggleOpti\
ons"),window.toggleOptions=S;function C(d){const m=get("thinking-options"),g=get("thinking-level"),b=get(
"reasoning-effort"),k=get("enable-thinking"),L=get("thinking-budget-label"),T=get("thinking-budget"),
A=d==="effort";if(m&&(m.dataset.thinkingMode=A?"effort":"thinking"),g&&g.classList.toggle("hidden",A),
b&&b.classList.toggle("hidden",!A),L&&L.classList.toggle("hidden",A),T&&T.classList.toggle("hidden",
A),!!k)if(A){k.dataset.thinkingChecked===void 0&&(k.dataset.thinkingChecked=k.checked?"1":"0");const $=b?
Array.from(b.options).find(F=>F.value==="none"):null,N=!!($&&!$.disabled&&!$.classList.contains("hid\
den"));k.checked=N?!!(b&&String(b.value||"").toLowerCase()!=="none"):!0,k.disabled=!N}else k.dataset.
thinkingChecked!==void 0&&(k.checked=k.dataset.thinkingChecked==="1",delete k.dataset.thinkingChecked)}
r(C,"setComposerThinkingMode");function M(){const d=get("model-select");if(!d)return;const m=d.value,
g=String(m||"").toLowerCase(),b=g.includes("deepseek"),k=g.startsWith("glm-"),L=get("thinking-option\
s"),T=get("reasoning-effort-container"),A=get("enable-thinking"),$=get("thinking-level"),N=get("thin\
king-budget"),F=get("enable-search"),Y=get("search-container"),oe=get("url-context-container"),le=get(
"enable-maps"),U=get("maps-grounding-container"),he=get("enable-sys-prompt"),q=get("sys-prompt-optio\
n"),fe=get("enable-python"),J=get("python-container"),be=get("prompt-cache-container"),_e=get("enabl\
e-prompt-cache"),ae=m==="gpt-5-search-api",ve=m.includes("tts"),Ne=isMistralOcrModel(m),Re=g.includes(
"gemini-3.1-flash-lite-image"),rt=g==="gemini-nano-banana-2.1",Ke=g.includes("gemini-3.1-flash-image")&&
!Re||rt,ht=isClaudeModelKey(m),Nt=g==="gemini-3.8-flash-cyber",Ee=isLlmModel()&&!b&&!ve&&!g.includes(
"realtime")&&!g.includes("native-audio")&&!g.includes("live");be&&(Ee?(be.classList.remove("hidden",
"opacity-50","pointer-events-none"),_e&&(_e.disabled=!1)):(_e&&(_e.checked=!1,_e.disabled=!0),be.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),L&&L.classList.add("hidden"),T&&T.classList.
add("hidden"),C("thinking");const Se=get("vision-model-info");if(Se&&Se.classList.add("hidden"),T){const Le=get(
"reasoning-effort");if(Le){Array.from(Le.options).forEach(Ue=>{const Xe=g==="gpt-5.6"||g.startsWith(
"gpt-5.6-"),mt=g.startsWith("gpt-6"),_n=g==="gpt-6-sol"||g==="gpt-6-luna"||g==="gpt-6.1-terra",nn=g===
"deepseek-v4.1-flash"||g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash\
-vision-exp",Sn=g==="deepseek-v4-pro",Kn=g.includes("grok-4.5"),D=g.includes("grok-4.6");Ue.value===
"max"?Ue.classList.toggle("hidden",!Xe&&!mt&&!nn&&!Sn):Ue.value==="xhigh"?Ue.classList.toggle("hidde\
n",!D&&!g.includes("multi-agent")&&!Xe&&!mt):Ue.value==="medium"?Ue.classList.toggle("hidden",!(g.includes(
"grok-4.3")||Kn||D||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("mu\
lti-agent")||g.includes("gpt-5")||mt||g.includes("o1")||g.includes("o3"))):Ue.value==="none"?Ue.classList.
toggle("hidden",!g.includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!_n&&!nn&&
!Sn):Ue.value==="low"&&Ue.classList.toggle("hidden",Sn)});const Be=Le.selectedOptions&&Le.selectedOptions[0];
Be&&Be.classList.contains("hidden")&&(Le.value=b?"high":"medium")}}oe&&oe.classList.add("hidden"),U&&
U.classList.add("hidden"),A&&(A.disabled=!1),N&&(N.disabled=!0,N.classList.add("opacity-50"));const qe=isGeminiImageModelKey(
m);if(ve||Ne)Y&&(get("enable-search").checked=!1,Y.classList.add("opacity-50","pointer-events-none")),
oe&&(get("enable-url-context").checked=!1,oe.classList.add("opacity-50","pointer-events-none")),U&&le&&
(le.checked=!1,U.classList.add("opacity-50","pointer-events-none")),J&&(fe.checked=!1,J.classList.add(
"opacity-50","pointer-events-none")),he&&q&&(he.checked=!1,he.disabled=!0,q.classList.add("opacity-5\
0"));else if(Ke||Re)U&&le&&(le.checked=!1,U.classList.add("hidden","opacity-50","pointer-events-none")),
L.classList.remove("hidden"),rt?(Array.from($.options).forEach(Le=>{Le.disabled=Le.value==="low"}),[
"minimal","medium","high"].includes($.value)||($.value="medium")):(Array.from($.options).forEach(Le=>{
["low","medium"].includes(Le.value)&&(Le.disabled=!0),["minimal","high"].includes(Le.value)&&(Le.disabled=
!1)}),["minimal","high"].includes($.value)||($.value=Re?"minimal":"high")),A&&(A.disabled=!1),Re&&(F&&
(F.checked=!1,F.disabled=!0),Y&&Y.classList.add("opacity-50","pointer-events-none"));else if(qe)U&&le&&
(le.checked=!1,U.classList.add("hidden","opacity-50","pointer-events-none"));else if(ht)L.classList.
remove("hidden"),N&&(N.disabled=!1,N.classList.remove("opacity-50")),Array.from($.options).forEach(Le=>{
Le.disabled=!0}),J&&(fe.checked=!1,J.classList.add("opacity-50","pointer-events-none"));else if(Nt){
L&&L.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from($.options).forEach(Be=>{Be.
disabled=!["low","medium","high"].includes(Be.value)}),["low","medium","high"].includes($.value)||($.
value="medium"),[Y,oe,U,J].forEach(Be=>{Be&&Be.classList.add("opacity-50","pointer-events-none")}),[
F,le,fe].forEach(Be=>{Be&&(Be.checked=!1,Be.disabled=!0)});const Le=get("enable-url-context");Le&&(Le.
checked=!1,Le.disabled=!0),he&&q&&(he.disabled=!1,q.classList.remove("opacity-50"))}else if(m.includes(
"gemini")&&!qe){L.classList.remove("hidden"),oe&&oe.classList.remove("hidden","opacity-50","pointer-\
events-none");const Le=m.includes("gemini-3")||m.includes("gemini-4");U&&(Le?U.classList.remove("hid\
den","opacity-50","pointer-events-none"):(le&&(le.checked=!1),U.classList.add("hidden","opacity-50",
"pointer-events-none")));const Be=m.includes("flash");Array.from($.options).forEach(Ue=>{m==="gemini\
-4-argon"||m==="gemini-3.8-flash"||m==="gemini-3.7-flash"?Ue.disabled=!["low","medium","high"].includes(
Ue.value):m==="gemini-3.6-flash"?Ue.disabled=!["medium","high"].includes(Ue.value):m==="gemini-3.5-f\
lash-lite"?Ue.disabled=!["minimal","medium","high"].includes(Ue.value):["minimal","medium"].includes(
Ue.value)?Ue.disabled=!Be:Ue.disabled=!1}),(m==="gemini-4-argon"||m==="gemini-3.8-flash"||m==="gemin\
i-3.7-flash")&&!["low","medium","high"].includes($.value)||m==="gemini-3.6-flash"&&!["medium","high"].
includes($.value)?$.value="medium":m==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(
$.value)?$.value="minimal":!Be&&["minimal","medium"].includes($.value)&&($.value="high"),Le?A&&(A.checked=
!0,A.disabled=!0):A&&(A.disabled=!1),N&&m.includes("gemini-2.5")&&(N.disabled=!1,N.classList.remove(
"opacity-50")),N&&!m.includes("gemini-2.5")&&(N.disabled=!0,N.classList.add("opacity-50"))}if(isLlmModel()&&
(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4\
.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes(
"multi-agent")||g.includes("gpt")&&!g.includes("tts")))L&&L.classList.remove("hidden"),C("effort"),Y&&
Y.classList.remove("opacity-50","pointer-events-none");else if(b){L&&L.classList.remove("hidden"),C(
"effort");const Le=get("vision-model-info");if(Le&&Le.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),F&&(F.checked=!1,F.disabled=!0),Y&&Y.classList.add("opaci\
ty-50","pointer-events-none"),oe){const Be=get("enable-url-context");Be&&(Be.checked=!1),oe.classList.
add("opacity-50","pointer-events-none")}U&&le&&(le.checked=!1,U.classList.add("opacity-50","pointer-\
events-none"))}else Ne||(Y&&Y.classList.remove("opacity-50","pointer-events-none"),U&&le&&(le.checked=
!1,U.classList.add("hidden","opacity-50","pointer-events-none")));if(ve?J&&J.classList.add("opacity-\
50","pointer-events-none"):(J&&J.classList.remove("opacity-50","pointer-events-none"),(!qe||Ke)&&!m.
includes("gpt-image")&&(he.disabled=!1,q.classList.remove("opacity-50"))),(qe&&!Ke||m.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(m)||isGrokVideoModel()||Ne)&&he&&q&&(he.checked=!1,he.disabled=
!0,q.classList.add("opacity-50")),J&&(isLlmModel()?(J.classList.remove("hidden"),fe.disabled=!1):(fe.
checked=!1,fe.disabled=!0,J.classList.add("hidden"))),ae?(F&&(F.checked=!0,F.disabled=!0),Y&&Y.classList.
add("opacity-50","pointer-events-none"),J&&(fe.checked=!1,fe.disabled=!0,J.classList.add("opacity-50",
"pointer-events-none"))):F&&!m.includes("tts")&&!Ne&&!b&&!Re&&(F.disabled=!1),Nt){[F,le,fe].forEach(
Be=>{Be&&(Be.checked=!1,Be.disabled=!0)});const Le=get("enable-url-context");Le&&(Le.checked=!1,Le.disabled=
!0),[Y,oe,U,J].forEach(Be=>{Be&&Be.classList.add("opacity-50","pointer-events-none")})}k&&(Se&&Se.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[F,fe].forEach(Le=>{Le&&(Le.checked=!1,Le.disabled=!0)}),[Y,J].forEach(Le=>{Le&&
Le.classList.add("opacity-50","pointer-events-none")}));const lt=get("mask-btn");lt&&(isGptImageModel()||
isIdeogramObjectRemoval()?lt.classList.remove("hidden"):(lt.classList.add("hidden"),currentMaskImage=
null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),u(),f(),h(),y(),v(),
updateBatchUi(m),w(),x(),_(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}
r(M,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",S),
get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
S(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const E=get("enable-canvas-mode");E&&(E.checked=canvasModeEnabled,E.addEventListener("change",()=>syncCanvasModeUi(
E.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const H=get("enable-codin\
g-mode");H&&(H.checked=codingModeEnabled,H.addEventListener("change",()=>syncCodingModeUi(H.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const B=get(
"prompt-controls-toggle-btn");B&&(B.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,m,g){const b=(m||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(d||"");const k=d||"",L=(b||"").toLowerCase();
let T="";try{const q=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&k.length>2e4?T=escapeHtml(
k):T=hljs.highlight(k,{language:q}).value}catch{T=escapeHtml(k)}const A=encodeURIComponent(k).replace(
/'/g,"%27"),$=detectBlockedScriptsInCode(k),N=hashString(`${b||"TEXT"}
${k||""}`);let F="";if(canvasModeEnabled){const q=String(canvasPreviewState.selectedKey||"")===N,fe=q?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";F=`<button\
 class="canvas-preview-btn${q?" canvas-active":""}" data-code="${A}" data-code-key="${N}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${fe}" aria-label="${fe}" aria-pressed="${q?"true":"false"}"\
><i class="fas ${q?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
L,k)){const q=$?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
F=`<button class="html-preview-btn" data-code="${A}" ${$?'data-suspicious="1"':""} title="${q}" aria\
-label="${q}"><i class="fas ${$?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const Y=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,oe=L==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${N}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,le=(b||"TEXT")+($?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),U=getRenderableSvgCode(L,k);return`${U?buildSvgCodeRenderHtml(U,N):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${N}"><div class="code-header"\
><span class="code-lang">${le}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${oe}${F}${Y}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${T}</code></pre></d\
iv></div>`},link(d,m,g){return`<a href="${d}" title="${m||""}" target="_blank">${g}</a>`},image(d,m,g){
return buildChatImageHtml(d,{alt:g,title:m})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(ki=get("version-update-dismis\
s"))==null||ki.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const te=get("version-update-clear\
-cache");if(te&&(te.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),te.
addEventListener("change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(
te.checked)})),(_i=get("version-update-reload"))==null||_i.addEventListener("click",async()=>{var m;
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
classList.add("hidden")}const j=r(d=>{if(!d)return"\u4E0D\u660E";const m=new Date(d);return Number.isNaN(
m.getTime())?d:m.toLocaleString()},"formatSessionTime"),V=r(d=>{const m=Array.isArray(d)?d:[],g=get(
"passkey-list"),b=get("passkey-count");if(b&&(b.innerText=String(m.length)),!!g){if(!m.length){g.innerHTML=
'<div class="text-[11px] text-gray-500">\u767B\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}g.innerHTML="",m.forEach((k,L)=>{const T=k&&k.id?String(k.id):"",A=document.createElement("di\
v");A.className="bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between\
 gap-2";const $=document.createElement("div");$.className="min-w-0";const N=document.createElement("\
div");N.className="text-xs text-gray-200 truncate",N.innerText=k&&k.name?String(k.name):`Security Ke\
y ${L+1}`;const F=document.createElement("div");F.className="text-[10px] text-gray-500 mt-1",F.innerText=
k&&k.created_at?`\u767B\u9332\u65E5\u6642: ${j(k.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",
$.appendChild(N),$.appendChild(F),A.appendChild($);const Y=document.createElement("button");Y.type="\
button",Y.className="bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px] font-bold \
btn-hover shrink-0",Y.innerText="\u524A\u9664",Y.disabled=!T,T&&(Y.onclick=()=>window.removeWebAuthnCredential(
T)),A.appendChild(Y),g.appendChild(A)})}},"renderPasskeyList"),ue=r(d=>{const m=get("session-list");
if(m){if(!d||!d.length){m.innerHTML='<div class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(g=>{const b=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",k=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",L=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
T=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${k}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(T)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(j(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(j(g.created_at))}</div></div>${L}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const b=g.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const k=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let L={};try{L=await k.json()}catch{}if(k.ok){if(L.logged_out){
location.href="/login";return}await me()}else showToast(L&&L.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),me=r(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api\
/sessions");let g={};try{g=await m.json()}catch{}if(!m.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(g.sessions||[]).filter(k=>!k.is_revoked);ue(b)},"loadSessions"),re=r(()=>{const d=get(
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
(m.checked=!0),S())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const ke=get("vision-model-change-btn");ke&&(ke.onclick=()=>_openVisionModelSelector());const I=get(
"compression-format-only");I&&(I.onchange=()=>{const d=I.checked,m=get("compression-max-size"),g=get(
"compression-max-dim");m&&(m.disabled=d),g&&(g.disabled=d);const b=get("compression-size-wrap"),k=get(
"compression-dim-wrap");b&&(b.style.opacity=d?"0.4":"1"),k&&(k.style.opacity=d?"0.4":"1")});const O=r(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");O(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const K=get("storage-us\
age-refresh");K&&(K.onclick=()=>loadStorageUsage());let R=null;const G=r(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),Q=r((d={})=>{const m=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),b=get("account-transfer-progress-percent"),k=get("account-transfer-progress-text"),
L=get("account-transfer-progress-detail"),T=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&m.
classList.remove("hidden"),g&&(g.style.width=`${T}%`),b&&(b.textContent=`${Math.round(T)}%`),k&&(k.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),L){const $={queued:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};L.textContent=$[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),P=r(d=>{Te&&(Te.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),ee=r((d={})=>{const m=get("account-export-ready"),
g=get("account-export-ready-text"),b=get("account-export-expiry"),k=get("account-export-download-btn"),
L=!!(d.available&&d.download_url);if(m&&m.classList.toggle("hidden",!L),!L){k&&k.removeAttribute("hr\
ef");return}const T=Math.max(0,Number(d.size_bytes)||0),A=T>=1024*1024*1024?`${(T/(1024*1024*1024)).
toFixed(2)} GB`:`${(T/(1024*1024)).toFixed(1)} MB`;if(g){const $=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD\
 ${Number(d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";
g.textContent=`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${$}`}
if(b){const $=d.expires_at?new Date(d.expires_at):null;b.textContent=$&&!Number.isNaN($.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${$.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
k&&(k.href=d.download_url)},"renderAccountExportAvailability"),de=r(async d=>{for(;R===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await m.json().catch(()=>({}));if(m.ok&&(g.state!=="pending"&&Q(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(m=>setTimeout(m,700))}
return null},"pollAccountTransfer"),ye=r((d,m,g=!0)=>{m&&(Q(m),ee(m),g&&m.state==="ready"?showToast(
m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):g&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),pe(d))},"handleFinishedAccountExport"),Z=r(async()=>{try{const d=await apiFetch("/api/ac\
count/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(ee(m),m.state==="ready"){Q(m);return}if(["failed","cancelled","expired"].includes(
m.state)){Q(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||R&&R.id===
m.job_id||R)return;const g={id:m.job_id,type:"export",stopped:!1,restored:!0};R=g,P(!0),Q(m);const b=await de(
g);b&&ye(g,b,!0)}catch{}},"refreshLatestAccountExport"),pe=r(d=>{R===d&&(R=null),d.stopped=!0,P(!1)},
"finishAccountTransfer"),X=get("account-transfer-cancel-btn");X&&(X.onclick=async()=>{const d=R;if(!(!d||
d.stopped)){d.cancelRequested=!0,X.disabled=!0,Q({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),Q({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&ee({available:!1}),pe(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const Te=get("account-export-btn");Te&&(Te.onclick=async()=>{if(R)return;const d={id:G(),
type:"export",stopped:!1};R=d,P(!0),ee({available:!1}),Q({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const m=await apiFetch("/api/account/export",
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{job_id:d.id}),keepalive:!0})),g=await m.json().catch(()=>({}));if(m.status===409&&g.error==="export\
_in_progress"&&g.job_id)d.id=g.job_id;else if(!m.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\
\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
Q({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await de(d);!d.cancelRequested&&b&&ye(d,b,!0)}catch(m){const g=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";Q({progress:0,phase:"faile\
d",message:g}),showToast(g,"error",!0),pe(d)}});const ne=get("account-export-download-btn");ne&&ne.addEventListener(
"click",async d=>{const m=ne.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await g.json().catch(
()=>({}));g.ok&&b.available&&b.download_url?(ne.href=b.download_url,window.location.assign(b.download_url)):
(ee(b),Q(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),Z())}catch{window.location.assign(m)}}}),P(!1),Z();const ce=get("import-files-grid"),Me=get(
"import-files-info"),we=get("import-files-summary"),Ye=r(d=>{const m=Math.max(0,Number(d)||0);return m>=
1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(1)} M\
B`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let Ie=null;const st=r(()=>{if(!Ie)
return;const d=Ie.files,m=Ie.selection;let g=0;d.forEach(L=>{m.has(L.archive_path)&&(g+=Number(L.size_bytes)||
0)});const b=Number(Ie.available_bytes)||0,k=g>b;we&&(we.textContent=`\u9078\u629E\u4E2D: ${Ye(g)} /\
 \u5229\u7528\u53EF\u80FD: ${Ye(b)}${k?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,we.classList.toggle(
"text-red-300",k)),Me&&(Me.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),pt=r(()=>{
if(!ce||!Ie)return;ce.innerHTML="";const d=Ie.files;if(!d.length){ce.innerHTML='<div class="text-xs \
text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
st();return}d.forEach(m=>{const g=document.createElement("label"),b=Ie.selection.has(m.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${Ye(m.size_bytes)}</div></div>`;const k=g.
querySelector(".import-file-check");k.addEventListener("change",()=>{k.checked?Ie.selection.add(m.archive_path):
Ie.selection.delete(m.archive_path),g.classList.toggle("border-blue-500",k.checked),g.classList.toggle(
"border-gray-600",!k.checked),st()}),ce.appendChild(g)}),st()},"renderImportFileItems"),se=r(d=>new Promise(
m=>{if(Ie={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:m},pt(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),yt=r(d=>{if(hideModal("import-files-modal"),Ie){const m=Ie.resolve;
Ie=null,m(d)}},"closeImportFileSelection"),He=get("import-files-close");He&&(He.onclick=()=>yt(null));
const Tt=get("import-files-cancel");Tt&&(Tt.onclick=()=>yt(null));const ut=get("import-files-confirm");
ut&&(ut.onclick=()=>{if(!Ie)return;const d=Array.from(Ie.selection);yt(d.length?d.join(","):"__none_\
_")});const vt=get("import-files-select-all");vt&&(vt.onclick=()=>{Ie&&(Ie.files.forEach(d=>Ie.selection.
add(d.archive_path)),pt())});const sn=get("import-files-none");sn&&(sn.onclick=()=>{Ie&&(Ie.selection.
clear(),pt())});const Ln={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},an=r(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const m=String(
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let Rt=null;const Et=r(
d=>{if(Rt){const m=Rt;Rt=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Bt=r(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}Rt=m;const b=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],k=get("settings-confirmation-list");k&&(b.length?
k.innerHTML=b.map(T=>{const A=Ln[T.field]||T.field,$=an(T.current),N=an(T.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
$)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(N)}</d\
iv>
                            </div>`}).join(""):k.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const L=get("settin\
gs-confirmation-count");L&&(L.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),wt=get("settings-confir\
mation-modal");wt&&wt.addEventListener("click",d=>{d.target===wt&&Et(!1)});const Vt=get("settings-co\
nfirmation-close");Vt&&(Vt.onclick=()=>Et(!1));const jt=get("settings-confirmation-cancel");jt&&(jt.
onclick=()=>Et(!1));const xt=get("settings-confirmation-confirm");xt&&(xt.onclick=()=>Et(!0));const $t=get(
"account-import-btn"),Jt=get("account-import-inplace"),Ft=get("account-import-inplace-warning");if(Jt&&
Ft){const d=r(()=>Ft.classList.toggle("hidden",!Jt.checked),"syncInplaceWarn");Jt.addEventListener("\
change",d),d()}$t&&($t.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:
null,g=get("account-import-categories"),b=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(le=>le.value):[],k=get("account-import-inplace"),L=!!(k&&k.checked),T=get("account-impo\
rt-settings-bypass"),A=!!(T&&T.checked);let $=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const N=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(le=>(le.
closest("label")&&le.closest("label").textContent||le.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${N.join("\u3001")}${L?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const F={id:G(),type:"import",stopped:!1,controller:new AbortController};
R=F,P(!0),Q({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const Y=get("account-import-result");let oe=Promise.resolve(null);try{const U=Math.max(1,Math.ceil(m.
size/10485760)),he=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:F.controller.
signal})),q=await he.json().catch(()=>({}));if(!he.ok)throw new Error(q.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
F.uploadId=q.upload_id;const fe=q.chunk_size||10485760;let J=0,be=0;const _e=r(async()=>{for(;;){const Ee=be++;
if(Ee>=U)return;const Se=m.slice(Ee*fe,Math.min(m.size,(Ee+1)*fe)),qe=new FormData;qe.append("chunk",
Se,m.name),qe.append("index",String(Ee));const et=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:qe,signal:F.controller.signal})),
lt=await et.json().catch(()=>({}));if(!et.ok)throw new Error(lt.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
J++,Q({progress:Math.min(35,Math.round(J/U*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${J}\
/${U}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ae=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ae=!0);try{await Promise.
all([_e(),_e(),_e()]);const Ee=await apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:F.controller.signal})),Se=await Ee.json().
catch(()=>({}));if(!Ee.ok)throw new Error(Se.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
Q({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ae&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let ve="",Ne=!1,Re=0;const rt=r(
async()=>{let Ee=!1;const Se=r(()=>{Ee||(Ee=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const qe=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),et=await qe.
json().catch(()=>null);if(!qe.ok||!et){Se();return}cacheUserSettings(et);const lt=get("settings-moda\
l");if(lt&&lt.classList.contains("modal-open"))try{Pe(et)}catch{}et.theme_color&&applyThemeColor(et.
theme_color,!0),Object.prototype.hasOwnProperty.call(et,"minimal_prompt_mode")&&et.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(et,"compact_prompt_mode")&&setCompactPromptMode(
!!et.compact_prompt_mode)}catch{}Se()},"refreshSettingsFormAfterImport"),Ke=r(Ee=>{const Se=Ee&&Ee.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";Y&&(Y.textContent=`\u5B8C\u4E86: ${Se}`,
Y.classList.remove("hidden","text-red-300"),Y.classList.add("text-emerald-300")),Q({progress:100,phase:"\
completed",message:Se}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&rt()},"finishImportSuccess"),ht=r(async()=>{
try{const Se=await(await apiFetch(`/api/account/transfer/${F.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return Se&&Se.state?Se:null}catch{return null}},"fetchImportSta\
tus"),Nt=r(async()=>{const Ee=await ht();if(!Ee)return{status:"unknown"};if(Ee.state==="completed")return Ke(
Ee),{status:"done"};if(["failed","cancelled","expired"].includes(Ee.state))throw new Error(Ee.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Ee.state==="needs_sel\
ection"&&Array.isArray(Ee.files)){const Se=await se({files:Ee.files,available_bytes:Ee.available_bytes});
return Se===null?(Q({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(ve=Se,{status:"reselect"})}if(Ee.state===
"needs_settings_confirmation"&&Array.isArray(Ee.settings_changes))return await Bt({settings_changes:Ee.
settings_changes})?($=!0,{status:"reselect"}):(Q({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Ee.state==="running"){const Se=await Promise.race([oe.catch(()=>null),new Promise(qe=>setTimeout(
()=>qe(null),6e4))]);if(Se&&Se.state==="completed")return Ke(Se),{status:"done"};throw Se&&["failed",
"cancelled","expired"].includes(Se.state)?new Error(Se.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ne;){F.stopped=!0,await oe.catch(()=>null),
F.stopped=!1,oe=de(F);let Ee;try{Ee=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:F.uploadId,
categories:b.join(","),job_id:F.id,selected_files:ve,restore_inplace:L,confirm_settings:$||A}),signal:F.
controller.signal}))}catch(Xe){if(F.cancelRequested||Xe&&Xe.name==="AbortError")throw Xe;const mt=await Nt();
if(mt.status==="done"){Ne=!0;break}if(mt.status==="cancelled")return;if(mt.status==="reselect")continue;
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Se=null;try{Se=await Ee.json()}catch{Se=null}if(Se===null){const Xe=await Nt();if(Xe.status==="d\
one"){Ne=!0;break}if(Xe.status==="cancelled")return;if(Xe.status==="reselect")continue;if(Ee.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(Re<2){Re++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Ee.ok&&Se.error==="storage_limit_files"&&Se.files){const Xe=await se(Se);if(Xe===null){Q({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}ve=Xe;continue}if(Se&&Se.status==="settings_confirmation"&&
Array.isArray(Se.settings_changes)){if(!await Bt(Se)){Q({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}$=!0;continue}if(!Ee.ok)throw new Error(Se.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const qe=Se.imported||{},et=[`\u8A2D\u5B9A ${qe.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${qe.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${qe.chats||0}\u4EF6`,`Gem ${qe.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${qe.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${qe.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${qe.
diagnostics||0}\u4EF6`].join(" / "),lt=Se.duplicates||{},Le={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Be=[];for(const Xe of Object.keys(Le)){const mt=Number(lt[Xe])||0;mt>
0&&Be.push(`${Le[Xe]} ${mt}\u4EF6`)}const Ue=Be.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Be.
join("\u3001")}\uFF09`:"";Y&&(Y.textContent=`\u5B8C\u4E86: ${et}${Ue}`,Y.classList.remove("hidden","\
text-red-300"),Y.classList.add("text-emerald-300")),Q({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&rt(),Ne=!0}}catch(le){if(F.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),F.cancelRequested||le&&le.name==="AbortError")return;const U=le&&le.message?
le.message:"",he=U==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
U||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";Q({progress:0,phase:"f\
ailed",message:he}),Y&&(Y.textContent=he,Y.classList.remove("hidden","text-emerald-300"),Y.classList.
add("text-red-300")),showToast(he,"error",!0)}finally{F.stopped=!0,await oe.catch(()=>null),pe(F)}});
const kt=get("account-dedupe-btn"),It=get("account-dedupe-result"),Dt=r((d,m=!1)=>{It&&(It.textContent=
d,It.classList.remove("hidden"),It.classList.toggle("text-red-300",!!m),It.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");kt&&(kt.onclick=async()=>{const d=r(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await m.json().catch(()=>null);if(!m.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){Dt("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],k={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const F of[
"chats","gems","files","feedback","diagnostics"]){const Y=Number(g.duplicates&&g.duplicates[F])||0;Y>
0&&b.push(`${k[F]} ${Y}\u4EF6`)}const L=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${L}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const T=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await T.json().catch(()=>null);if(!T.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const $=[];for(const F of["chats","gems","files","feedback","diagnostics"]){const Y=Number(A.removed&&
A.removed[F])||0;Y>0&&$.push(`${k[F]} ${Y}\u4EF6`)}const N=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";Dt(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${$.join("\u3001")||"0\u4EF6"}${N}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!kt.disabled){kt.disabled=!0,Dt("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){Dt(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{kt.disabled=!1}}});const on=get("site-cache-usage-refresh");on&&(on.onclick=()=>loadSiteCacheUsage());
const Kt=get("clear-site-cache-btn");Kt&&(Kt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Kt)});
const _t=get("enc-scan-result"),rn=r(async(d=null)=>{_t&&(_t.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
m,{cache:"no-store"}),b=await g.json();if(!g.ok){_t&&(_t.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const k=b.total||0,L=b.encrypted||0,T=b.unencrypted||0;let A=`Total: ${k} / Encrypted: ${L} /\
 Plain: ${T}`;if(b.samples&&b.samples.length){const $=b.samples.slice(0,8).map(N=>{const F=N.timestamp?
new Date(N.timestamp).toLocaleString():"";return`#${N.id} (${N.role||""}) ${F}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${$}</div>`}_t&&(_t.innerHTML=A)}catch{_t&&(_t.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),W=get("enc-scan-all");W&&(W.onclick=()=>rn(null));
const xe=get("enc-scan-thread");xe&&(xe.onclick=()=>currentThreadId?rn(currentThreadId):showToast("\u30B9\
\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ae=get("admin-enc-list");
let Fe=null,at=!1;const ft=r(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThreadEncrypt\
edFromMessages"),St=r(()=>{Fe=ft(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Qe=r(async(d,m,{
confirmPrompt:g=!0,reloadCurrent:b=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const k=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;at=!0;try{const L=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
T=await L.json().catch(()=>({}));return L.ok?(showToast(`${k}\u3057\u307E\u3057\u305F\uFF08${T.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Fe=!!m,b&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ae&&await Je(),!0):
(showToast(T.error||`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${k}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{at=!1}},"setAdminThreadEncry\
ption"),Lt=r(d=>{if(!Ae)return;const m=d.threads||[];if(!m.length){Ae.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ae.innerHTML=m.map(g=>{const b=g.encrypted_count>0?"enc":"plain",k=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",L=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",T=g.
updated_at?new Date(g.updated_at).toLocaleString():"",A=escapeHtml(String(g.thread_id)),$=currentThreadId&&
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
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${L} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${k}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Je=r(async()=>{if(Ae){Ae.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){Ae.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(Lt(m),currentThreadId&&Array.isArray(m.threads)){const g=m.threads.find(b=>String(
b.thread_id)===String(currentThreadId));g&&(Fe=!!g.encrypted)}}catch{Ae.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Je()),window.__loadAdminEncThreads=
Je,window.__refreshAdminThreadEncState=St,window.__setAdminThreadEncryption=Qe;const nt=get("encrypt\
ion-status-admin-toggle");nt&&nt.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ae&&(Ae.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const A=m.getAttribute("data-id");if(!A)return;typeof We==
"function"?We():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||at)return;const b=g.getAttribute("data-i\
d"),k=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${k?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const T=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await Qe(b,k,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=T,await Je()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const Ct=r(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(g=>{const b=g.status||"new",k=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
L=g.created_at?new Date(g.created_at).toLocaleString():"",T=g.replied_at?new Date(g.replied_at).toLocaleString():
"",A=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${k}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(L)}</div>
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
                    `}).join("")}},"renderBanAppeals"),ln=r(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const k=(await g.json()).unread_count||0;m.textContent=String(k),d&&k>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${k}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Pt=r(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const g=await m.json();Ct(g.items||[]),
await ln(!1)}catch{}}},"loadBanAppeals"),cn=r(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await Pt()}catch{}},"markBanAppealsRead"),Xt=r(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await Pt()}catch{}},"updateBanAppealStatus"),De=r(()=>{
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
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),ze=r(()=>{const d=get("set-st\
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
"success")})},"ensureLlmTranscribePromptSettingsUi"),Ze=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
700 rounded p-2 text-xs text-gray-200";return Ze.map(b=>{const k=b.mcpLocked===!0,L=k?'<div class="t\
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
                        ${L}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{Ze.forEach(g=>{const b=m&&
typeof m=="object"?m[g.key]||{}:{},k=get(`${d}-auto-sys-${g.key}-enabled`),L=get(`${d}-auto-sys-${g.
key}-text`);k&&(g.mcpLocked===!0?k.disabled=!0:k.checked=b.enabled!==!1),L&&(L.value=b.text||"",L.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const dn=r((d,m=null)=>{if(m){const g=get(m);g&&(g.checked=!0)}Ze.forEach(g=>{const b=get(`${d}-auto\
-sys-${g.key}-enabled`),k=get(`${d}-auto-sys-${g.key}-text`);if(b&&(g.mcpLocked!==!0?b.checked=!0:b.
disabled=!0),k){const L=k.placeholder||"";k.value=L}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),un=r(d=>{const m={};return Ze.forEach(g=>{const b=get(`${d}\
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
                `,m.appendChild(g)},De(),ze(),O();const Ht=r(()=>{const d=get("set-default-model");if(!d)
return;const m=d.value;d.innerHTML="",MODELS.forEach(b=>{const k=document.createElement("optgroup");
k.label=b.category,(b.items||[]).forEach(L=>{const T=document.createElement("option");T.value=L.id,T.
textContent=L.name,k.appendChild(T)}),d.appendChild(k)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),Ce=r(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(b=>{const k=(b.items||[]).filter(T=>{const A=(T.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(k.length===0)return;const L=document.createElement("optgroup");L.label=b.category,k.forEach(
T=>{const A=document.createElement("option");A.value=T.id,A.textContent=T.name+" \u2605",L.appendChild(
A)}),d.appendChild(L)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
"gemini-3-flash-preview";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),Pe=r(d=>{if(!d)return;cacheUserSettings(d);const m=get("app-glo\
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
ame")&&(get("set-username").value=d.username);const k=get("2fa-badge"),L=get("disable-2fa-btn");d.is_2fa_enabled?
(k.innerText="ENABLED",k.classList.replace("bg-gray-700","bg-green-600"),k.classList.replace("text-g\
ray-400","text-white"),L.classList.remove("hidden")):(k.innerText="DISABLED",k.classList.replace("bg\
-green-600","bg-gray-700"),k.classList.replace("text-white","text-gray-400"),L.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const T=get(
"set-passkey-only-login"),A=get("passkey-only-note"),$=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(V($),T){T.checked=!!d.passkey_only_login;const le=$.length>0||!!d.has_webauthn;T.disabled=!le,
le||(T.checked=!1),A&&(le?A.classList.add("hidden"):A.classList.remove("hidden"))}const N=get("mig-s\
tatus-box"),F=get("mig-progress-text"),Y=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){N.classList.remove("hidden");const le=(d.migration_progress||"").split("/");if(le.length===
2){const U=parseInt(le[0]||"0",10),he=parseInt(le[1]||"0",10);F&&(F.innerText=`${U} / ${he}`),Y&&he>
0&&(Y.style.width=`${Math.min(100,Math.floor(U/he*100))}%`)}}else N.classList.add("hidden"),Y&&(Y.style.
width="0%"),F&&(F.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&Pe(d);const m=get(
"search-box"),g=m?m.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),Ht(),Ce(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(k=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+k+"ms"),snapshotSidebarHistory("settings-open-later-"+k+"ms")},k)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),Z(),ze(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),ln(!0),Pt(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),gn(),re(),me();try{loadMcpServers()}catch{}};const We=r((d=!1)=>{settingsModelPickerTarget&&
window.closeModelModal(!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),
revealPersistentSidebarLists(),snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory(
"settings-close-later-300ms"),300),!d&&location.pathname==="/settings"&&history.back()},"closeSettin\
gsModal"),ct=r(()=>{const d=get("set-theme-color"),m=get("set-theme-color-text"),g=get("theme-reset-\
btn"),b=document.querySelectorAll("#theme-presets .theme-swatch"),k=r((L,T=!0)=>{const A=normalizeHex(
L);A&&(applyThemeColor(A,T),syncThemeInputs(A))},"applyFromValue");d&&d.addEventListener("input",()=>k(
d.value,!0)),m&&(m.addEventListener("change",()=>{const L=normalizeHex(m.value);if(!L){syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||THEME_DEFAULT);return}k(L,!0)}),m.addEventListener("keydown",
L=>{L.key==="Enter"&&(L.preventDefault(),m.blur())})),g&&(g.onclick=()=>k(THEME_DEFAULT,!0)),b.forEach(
L=>{L.addEventListener("click",()=>k(L.getAttribute("data-color"),!0))})},"bindThemeControls"),dt=r(
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
"close-settings-btn").onclick=()=>We();const Mt=get("settings-header-close");Mt&&(Mt.onclick=()=>We());
const gt=get("settings-search");gt&&(gt.addEventListener("input",filterSettings),gt.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const g=m.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Yn=get(
"settings-search-clear");Yn&&Yn.addEventListener("click",()=>{gt&&(gt.value="",filterSettings(),gt.focus())}),
ct(),dt(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Nn=get("set-gemini-local-python-dialog");Nn&&(Nn.onchange=()=>setGeminiLocalPyDialogEnabled(Nn.
checked));const Qn=get("set-gemini-backend");Qn&&(Qn.onchange=()=>syncGeminiBackendUi());const Zn=get(
"set-admin-api-key-mode");Zn&&(Zn.onchange=()=>syncAdminApiKeyModeUi());const Rn=get("set-temp-chat-\
timeout-seconds");Rn&&(Rn.onchange=()=>{applyTemporaryChatTimeoutSeconds(Rn.value)});const ei=get("s\
lash-command-cancel-btn");ei&&(ei.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{var A;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
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
((A=get("set-server-diagnostic-log-retention-hours"))==null?void 0:A.value)||1),enable_server_diagnostic_log:get(
"set-server-diagnostic-log")?get("set-server-diagnostic-log").checked:!1,enable_client_debug_log:get(
"set-client-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-o\
nly-login")?get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-goog\
le")?get("set-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set\
-default-2fa-method").value:"totp",new_username:d?d.value:null,new_password:m?m.value:null},k=get("s\
et-e2ee")?get("set-e2ee").checked:!1,L=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
k!==L&&(b.enable_e2ee=k),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
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
try{const oe=await T.json();oe&&oe.message&&($=oe.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),
window.applySavedUserSystemPromptSettings({system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,
apply_global_system_prompt:b.apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:b.auto_system_prompt_notices_config}),We();const N=currentUsername,
F=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const Y=useSwCache;
useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),Y!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast($,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==N?setTimeout(()=>location.
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
k.classList.remove("hidden")),d&&(d.value="");else{const L=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(L,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),b=await g.json(),k=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")k&&(k.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
k.classList.remove("hidden")),d&&(d.value="");else{const L=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(L,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const g=r(J=>{m&&(m.textContent=J)},"setBox"),b=r(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),k=r((J,be)=>!J||!be||be<=0?0:J*8/(be/1e3)/1e3/1e3,"toMbps"),L=r(J=>Number.isFinite(
J)?`${J.toFixed(0)} ms`:"-","fmtMs"),T=r(J=>Number.isFinite(J)?`${J.toFixed(J>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=r(async(J,be)=>{const _e=await J.json().catch(()=>({}));return _e&&_e.error?_e.error:be},
"parseErr"),$=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let J=0;J<4;J++){const be=performance.now(),_e=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),ae=performance.now();if(!_e.ok)throw new Error(await A(
_e,"ping_failed"));await _e.json().catch(()=>({})),$.push(ae-be)}const N=$.reduce((J,be)=>J+be,0)/Math.
max(1,$.length),F=Math.min(...$),Y=r(async J=>{const be=performance.now(),_e=await apiFetch(`/api/sp\
eedtest/download?bytes=${J}&_=${b()}`,{cache:"no-store"});if(!_e.ok)throw new Error(await A(_e,"down\
load_failed"));const ae=await _e.arrayBuffer(),ve=performance.now();return{bytes:ae.byteLength||J,ms:ve-
be,mbps:k(ae.byteLength||J,ve-be)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${L(N)}
\u6E2C\u5B9A\u4E2D... download`);const oe=[];for(const J of[2*1024*1024,8*1024*1024])oe.push(await Y(
J)),g(`\u6E2C\u5B9A\u4E2D... ping ${L(N)}
download ${T(Math.max(...oe.map(be=>be.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const le=Math.max(...oe.map(J=>J.mbps)),U=r(async J=>{const be=new Uint8Array(
J),_e=performance.now(),ae=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:be,cache:"no-store"}),ve=performance.now();if(!ae.ok)
throw new Error(await A(ae,"upload_failed"));const Ne=await ae.json().catch(()=>({})),Re=Number(Ne.bytes_received||
J)||J;return{bytes:Re,ms:ve-_e,mbps:k(Re,ve-_e),serverMs:Number(Ne.server_elapsed_ms||0)||0}},"runUp\
load"),he=[];for(const J of[1*1024*1024,4*1024*1024])he.push(await U(J));const q=Math.max(...he.map(
J=>J.mbps)),fe=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${L(N)} / ${L(F)}`,`Download (best): ${T(le)}`,`Upload (best): ${T(q)}`,`Download r\
uns: ${oe.map(J=>`${Math.round(J.bytes/1024/1024)}MB=${T(J.mbps)}`).join(", ")}`,`Upload runs: ${he.
map(J=>`${Math.round(J.bytes/1024/1024)}MB=${T(J.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(fe.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Pt()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>cn()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){g&&await cn([Number(g)]);return}if(m.classList.contains(
"ban-appeal-status")){const b=m.getAttribute("data-status");g&&b&&await Xt({id:Number(g),status:b});
return}if(m.classList.contains("ban-appeal-reply-send")){const b=m.closest("[data-appeal-id]"),k=b?b.
querySelector(".ban-appeal-reply"):null,L=k?k.value:"";g&&await Xt({id:Number(g),admin_reply:L});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await Xt({id:Number(
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
Bn.onchange=d=>setMarkerColor(d.target.value));const jn=get("marker-opacity");jn&&(jn.oninput=d=>setMarkerOpacity(
d.target.value),jn.onchange=d=>setMarkerOpacity(d.target.value));const Cn=get("marker-opacity-number");
Cn&&(Cn.onchange=d=>setMarkerOpacity(d.target.value),Cn.onblur=d=>setMarkerOpacity(d.target.value),Cn.
onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(d.target.value),d.target.blur())}),document.querySelectorAll(
"#marker-toolbar .marker-color-chip[data-marker-color]").forEach(d=>{d.onclick=()=>setMarkerColor(d.
getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),
get("marker-crop-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&
(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=
()=>clearMarkerCanvas()),get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),
initMarkerCanvas(),initCropCanvas(),window.addEventListener("resize",()=>{const d=get("marker-modal");
!d||d.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const Li=r(()=>{const d=get(
"upload-modal");return!!(d&&!d.classList.contains("hidden"))},"isUploadModalOpen"),Yt=get("drop-over\
lay");let mn=0;const Ci=r(()=>{Li()||Yt&&(Yt.classList.remove("hidden"),Yt.classList.add("flex"))},"\
showDropOverlay"),pn=r(()=>{mn=0,Yt&&(Yt.classList.add("hidden"),Yt.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=pn;const At=get("upload-dropzone");At&&(At.addEventListener("dr\
agover",d=>{d.preventDefault(),At.classList.add("dragover")}),At.addEventListener("dragleave",()=>{At.
classList.remove("dragover")}),At.addEventListener("drop",d=>{d.preventDefault(),d.stopPropagation(),
At.classList.remove("dragover"),pn();const m=d.dataTransfer?d.dataTransfer.files:null;m&&m.length&&handleFiles(
m)})),window.addEventListener("dragenter",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.
types.includes("Files")||(mn+=1,Ci())}),window.addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||d.preventDefault()}),window.addEventListener("dragle\
ave",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(mn=Math.max(
0,mn-1),(mn===0||!d.relatedTarget||d.clientY<=0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=
window.innerHeight)&&pn())}),window.addEventListener("dragend",()=>{pn()}),window.addEventListener("\
drop",d=>{pn(),!(!d.dataTransfer||!d.dataTransfer.files||d.dataTransfer.files.length===0)&&(d.preventDefault(),
!(At&&At.contains(d.target))&&handleFiles(d.dataTransfer.files))});const ti=get("bot-admin-modal"),Ti=r(
d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML="",!d||!d.length){m.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,b)=>{const k=!!g.is_bot_banned,L=g.bot_detection_enabled!==!1,T=Number(g.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(b,12)*.02}\
s`,A.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(g.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${k?"BAN\u4E2D":"\u6B63\u5E38"}${T?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(T/60)}\u5206\uFF09`:""} ${g.bot_ban_reason?
" / "+escapeHtml(g.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${L?"1":"0"}">${L?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${k?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-bann\
ed="${k?"1":"0"}">${k?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${k?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(g.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        ${g.is_admin?"":`<button class="bot-delete-account bg-red-800 hover:bg-red-7\
00 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(g.username)}\
">\u524A\u9664</button>`}
                    `,m.appendChild(A)})}},"renderBotUsers"),fn=r(async(d="")=>{const m=get("bot-adm\
in-list");m&&(m.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),b=await g.json();g.ok&&b&&b.users?Ti(b.users):(m&&(m.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{m&&(m.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Fn=r(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||ti))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await fn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Fn;const Dn=r(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=Dn,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{Dn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||ti)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Fn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
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
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Qt={"/settings":{id:"settings-mod\
al",open:r(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:r(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:r(()=>{wi(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
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
"/gem":{id:"gem-modal",open:r(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="f\
as fa-gem text-blue-500 mr-2"></i>Create New Gem',showModal("gem-modal")},"open")},"/compression":{id:"\
compression-modal",open:r(()=>window.openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-mo\
dal",open:r(()=>Fn(),"open")},"/admin-bot-logs":{id:"bot-log-modal",open:r(()=>Dn(),"open")}},ni=r((d,m=!1)=>{
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
"popstate",d=>{var L,T;const m=(L=d.state)==null?void 0:L.settingsModelPicker;if(location.pathname===
"/settings"&&settingsModelPickerTargets.includes(m)){const A=r(()=>openSettingsModelPicker(m,!0),"re\
storePicker");(T=get("settings-modal"))!=null&&T.classList.contains("modal-open")?A():window.openSettingsModal().
then(()=>{var $;location.pathname==="/settings"&&(($=history.state)==null?void 0:$.settingsModelPicker)===
m&&A()});return}settingsModelPickerTarget&&window.closeModelModal(!0);let g=!1;Object.values(Qt).forEach(
A=>{const $=get(A.id);$&&$.classList.contains("modal-open")&&location.pathname!==Object.keys(Qt).find(
N=>Qt[N].id===A.id)&&(ni(A.id,!0),g=!0)});const b=location.pathname.match(/^\/c\/(.+)$/);if(b){const A=decodeURIComponent(
b[1]);String(currentThreadId)!==String(A)&&loadMessages(A,{skipHistory:!0})}else location.pathname===
"/"&&currentThreadId&&startNewChat({skipHistory:!0});const k=Qt[location.pathname];if(k){const A=get(
k.id);A&&!A.classList.contains("modal-open")&&k.open()}}),document.addEventListener("keydown",d=>{d.
key==="Escape"&&settingsModelPickerTarget&&(d.preventDefault(),window.closeModelModal())});const ii=location.
pathname;Qt[ii]&&(history.replaceState({},"","/"),setTimeout(()=>Qt[ii].open(),500)),get("easy-login\
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
"error",!0)});const Mi=4,Ai=5*1024*1024;function si(d,m,g){let b=[];const k=r(()=>{g&&(g.innerHTML="",
b.forEach((L,T)=>{const A=document.createElement("div");A.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const $=document.createElement("img");$.src=L.url,$.alt=L.file.name,
$.className="w-full h-full object-cover";const N=document.createElement("button");N.type="button",N.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",N.textContent=
"\xD7",N.onclick=()=>{URL.revokeObjectURL(L.url),b.splice(T,1),k()},A.appendChild($),A.appendChild(N),
g.appendChild(A)}))},"render");return d&&m&&(d.onclick=()=>m.click(),m.onchange=L=>{for(const T of Array.
from(L.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(T.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(T.
size>Ai){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(b.
length>=Mi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}b.push({file:T,url:URL.createObjectURL(T)})}L.target.value="",k()}),{files:r(()=>b.
map(L=>L.file),"files"),clear:r(()=>{b.forEach(L=>URL.revokeObjectURL(L.url)),b=[],k()},"clear")}}r(
si,"createFbImagePicker");async function ai(d,m){try{const g=new FormData;return m.forEach(b=>g.append(
"images",b,b.name)),(await apiFetch(d,{method:"POST",body:g})).ok}catch{return!1}}r(ai,"uploadFbImag\
es");const oi=si(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-message").value.trim();if(!m){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:m,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=get("fb-attach-logs"),k=b&&b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;k&&(g.client_logs=k);const L=get("fb-attach-chat"),T=!!(L&&L.checked);if(T&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}T&&(g.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const A=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),$=A.ok?await A.clone().json().
catch(()=>({})):{},N=oi.files(),F={count:N.length,saved:!0};A.ok&&F.count&&(F.saved=await ai(`/api/f\
eedback/${$.public_id}/images`,N)),(window.ActivityLog?await window.ActivityLog.reportFeedback(A,k,T,
F):A.ok)&&(get("fb-title").value="",get("fb-message").value="",oi.clear(),L&&(L.checked=!1),b&&(b.checked=
!0),gn())};async function ri(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),gn()}r(ri,"deleteFeedback");const Ei="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Hn=[];function li(d,m){const g=d.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(b=>{const k=b.role==="admin"?m&&b.author?`\u7BA1\u7406\u8005 (${b.
author})`:"\u7BA1\u7406\u8005":m?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",L=m?[b.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${b.
log_file}`,b.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${b.chat_file}`,b.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${b.image_count}\u679A\uFF09: ${b.image_dir}`]:[b.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${b.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(k)}\
 / ${escapeHtml(b.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${b.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(b.message)}</div>${L.filter(Boolean).map(T=>`<div cla\
ss="${m?"text-amber-300":"text-gray-400"}">${escapeHtml(T)}</div>`).join("")}</div>`}).join("")}</di\
v>`:""}r(li,"feedbackRepliesHtml");function ci(){return`<div class="fb-reply-form mt-2 space-y-2">
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
                    <label class="flex items-start gap-2 cursor-pointer">
                        <input type="checkbox" class="fb-reply-chat accent-amber-500 mt-0.5 shrink-0\
">
                        <span class="text-xs text-white">\u73FE\u5728\u958B\u3044\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B<span class="block text\
-[10px] text-gray-500">\u958B\u3044\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u3001\u56DE\u7B54\u3001\u601D\u8003\u904E\u7A0B\u3001\u6DFB\u4ED8\u30D5\u30A1\u30A4\u30EB\u3001\u30C1\u30E3\u30C3\u30C8\u306E\u8A2D\u5B9A\u3068\u3001\u305D\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u95A2\u3059\u308B\u30A8\u30E9\u30FC\u30ED\u30B0\u306A\u3069\u306E\u8A18\u9332\u3092\u5FA9\u53F7\u3057\u3066\u3001\u4E0D\u5177\u5408\u306E\u8ABF\u67FB\u7528\u306B\u9001\u4FE1\u3057\
\u307E\u3059\u3002\u9001\u4FE1\u3057\u305F\u30B3\u30D4\u30FC\u306F\u3001\u30C1\u30E3\u30C3\u30C8\u3092\u524A\u9664\u3057\u3066\u3082\u6B8B\u308A\u307E\u3059\u3002</span></span>
                    </label>
                    <button type="button" class="fb-reply-send bg-blue-600 hover:bg-blue-500 text-wh\
ite px-3 py-1 rounded text-xs">\u8FD4\u4FE1</button>
                </div>`}r(ci,"feedbackReplyFormHtml");function di(d,m){const g=m.querySelector(".fb-\
reply-form");if(!g)return;const b=g.querySelector(".fb-reply-logs"),k=g.querySelector(".fb-reply-log\
s-note"),L=!!(window.ActivityLog&&window.ActivityLog.canAttach());b.disabled=!L,k.textContent=L?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const T=si(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));Hn.push(T);const A=g.querySelector(".fb-reply-send");A.onclick=async()=>{const $=g.
querySelector(".fb-reply-message").value.trim();if(!$){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const N={message:$,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},F=b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;F&&(N.client_logs=F);const Y=g.
querySelector(".fb-reply-chat").checked;if(Y&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}Y&&(N.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const oe=T.files();A.disabled=!0;try{const le=await apiFetch(
`/api/feedback/${d.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(N)}),U=await le.json().catch(()=>({}));if(!le.ok){showToast(U.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const he=!oe.length||await ai(`/api/feedback/${d.public_id}/replies/${U.reply_id}\
/images`,oe),q=[];if(F&&U.logs_saved===!1&&q.push("\u30ED\u30B0"),he||q.push("\u753B\u50CF"),Y&&U.chat_copy_saved!==
!0&&q.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),q.length)showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${q.
join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,"error",!0);else{
const fe=["\u8FD4\u4FE1"];Y&&fe.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),F&&fe.push(
`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof U.logs_count=="number"?U.logs_count:
F.entries.length}\u4EF6\uFF09`),oe.length&&fe.push(`\u753B\u50CF\uFF08${oe.length}\u679A\uFF09`),showToast(
fe.join(fe.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","success")}gn()}finally{
A.disabled=!1}}}r(di,"bindFeedbackReply");function ui(){const d=get("fb-admin-status-filter").value;
let m=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(g=>{const b=d==="all"||
(d==="unresolved"?["new","in_review","replied"].includes(g.dataset.feedbackStatus):g.dataset.feedbackStatus===
d);g.hidden=!b,b&&m++}),get("fb-admin-empty").classList.toggle("hidden",m!==0)}r(ui,"filterAdminFeed\
back"),get("fb-admin-status-filter").onchange=ui;async function gn(){Hn.forEach(L=>L.clear()),Hn=[];
const m=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML="",(m.items||
[]).filter(L=>!m.is_admin||L.user_id===void 0||L.user_id===null||!0).forEach(L=>{if(m.is_admin)return;
const T=document.createElement("div");T.className="p-2 rounded border border-gray-700 bg-gray-800/50",
T.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
L.public_id||"")}</span> / ${L.created_at}</div><div class="font-bold text-sm">${escapeHtml(L.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(L.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(L.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":L.status)}</div>${L.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${L.
image_count}\u679A</div>`:""}${li(L,!1)}${ci()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
T.querySelector(".fb-delete").onclick=()=>ri(L.public_id),di(L,T),g.appendChild(T)});const b=get("fb\
-admin-panel"),k=get("fb-admin-list");m.is_admin?(b.classList.remove("hidden"),k.innerHTML="",(m.items||
[]).forEach(L=>{const T=document.createElement("div");T.className="p-2 rounded border border-gray-70\
0 bg-gray-800/50 space-y-2",T.dataset.feedbackStatus=L.status||"new",T.innerHTML=`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(L.public_id||"")}</span> / ${L.agent_name?`agent:${escapeHtml(L.agent_name)}`:`us\
er:${L.user_id}`} / ${L.created_at}${L.handled_by?` / \u5BFE\u5FDC: ${escapeHtml(L.handled_by)}`:""}\
</div>
                            <div class="font-bold text-sm">${escapeHtml(L.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(L.message)}</div>
                            ${[L.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${L.log_file}`,L.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${L.chat_file}`,L.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${L.
image_count}\u679A\uFF09: ${L.image_dir}`].filter(Boolean).map(A=>`<div class="text-[11px] text-ambe\
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
                            ${li(L,!0)}
                            ${ci()}
                        `,T.querySelector(".fb-status").value=L.status||"new",T.querySelector(".fb-s\
ave").onclick=async()=>{const A=T.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${L.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:A})}),
gn()},T.querySelector(".fb-delete").onclick=()=>ri(L.public_id),di(L,T),k.appendChild(T)}),ui()):b.classList.
add("hidden")}if(r(gn,"loadFeedback"),window.setupTOTP=async()=>{const m=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=m.qr_image,get("totp-secret-disp").innerText=
m.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const d=get(
"totp-verify-code").value;if(!d)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),m=get("webauthn-nam\
e"),g=m?String(m.value||"").trim():"";try{d&&(d.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),k=await b.json();if(!b.ok){showToast(k.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const T=await(await ensureWebAuthnJson()).create({publicKey:k}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},T,{name:g}))}),$=await A.json().catch(()=>({}));A.ok?(m&&(m.value=""),showToast(
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
L=>L.name.toLowerCase().includes(b)||L.description&&L.description.toLowerCase().includes(b));k[gemSelectedIndex]?
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
let ot,je=null,Tn=[],qn=!1,Mn=null,Zt=null,qt=null,bn=null,Gn=0,An=!1,yn=null,en=null,Ot=null,vn=null,
En=null,Gt=null,$n=null,In=null;function mi(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
$n)&&$n.length)return $n;d.innerHTML="";const m=[];for(let g=0;g<24;g++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",m.push(b),d.appendChild(b)}return $n=m,m}r(mi,"ensureMicWaveformBars");function Ut(d,m="hidden"){
const g=get("mic-recording-indicator"),b=get("mic-recording-text");if(g){if(In&&(clearTimeout(In),In=
null),m==="hidden"){g.classList.add("hidden");return}b&&d&&(b.innerText=d),g.classList.remove("hidde\
n"),m==="recording"?g.style.color="rgb(252 165 165)":m==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}r(Ut,"setMicRecordingIndicator");function pi(){mi().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}r(pi,"resetMicWaveformBars");function zt(){if(En&&(cancelAnimationFrame(
En),En=null),vn){try{vn.disconnect()}catch{}vn=null}if(en){try{en.close()}catch{}en=null}Ot=null,Gt=
null,pi()}r(zt,"stopMicWaveform");function fi(d){zt();const m=mi();if(!m.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{en=new g,Ot=en.createAnalyser(),Ot.fftSize=
256,Ot.smoothingTimeConstant=0,vn=en.createMediaStreamSource(d),vn.connect(Ot),Gt=new Uint8Array(Ot.
frequencyBinCount)}catch{zt();return}const b=r(()=>{if(!Ot||!Gt)return;Ot.getByteFrequencyData(Gt);const k=Math.
max(1,Math.floor(Gt.length/m.length));for(let L=0;L<m.length;L++){const A=(Gt[Math.min(Gt.length-1,L*
k)]||0)/255,$=Math.max(2,Math.round(2+A*10));m[L].style.height=`${$}px`,m[L].style.opacity=`${.35+A*
.65}`}En=requestAnimationFrame(b)},"render");b()}r(fi,"startMicWaveform");function Pn(){if(Mn&&(clearInterval(
Mn),Mn=null),bn){try{bn.disconnect()}catch{}bn=null}if(Zt){try{Zt.close()}catch{}Zt=null}qt=null}r(Pn,
"stopSilenceMonitor");function gi(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;Pn();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;Zt=new m,qt=Zt.createAnalyser(),qt.fftSize=2048,
bn=Zt.createMediaStreamSource(d),bn.connect(qt);const g=new Uint8Array(qt.fftSize),b=getStsSilenceMs(),
k=.02;Gn=0,An=!1,Mn=setInterval(()=>{if(!qt)return;qt.getByteTimeDomainData(g);let L=0;for(let A=0;A<
g.length;A++){const $=(g[A]-128)/128;L+=$*$}if(Math.sqrt(L/g.length)>k){An||(An=!0),Gn=Date.now();return}
An&&Date.now()-Gn>b&&ot&&ot.state==="recording"&&ot.stop()},200)}r(gi,"startSilenceMonitor");const Wn=class Wn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(m,g,b,k={},L=null){if(this.
model=b,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(m)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const $=!!(k&&
k.transcriptionConfig),N={setup:{model:`models/${b}`,generationConfig:{responseModalities:$?["TEXT"]:
["AUDIO"]},inputAudioTranscription:$?k.transcriptionConfig||{}:{}}};$||(N.setup.outputAudioTranscription=
{}),k.speechConfig&&(N.setup.generationConfig.speechConfig=k.speechConfig),k.thinkingConfig&&(N.setup.
generationConfig.thinkingConfig=k.thinkingConfig),k.translationConfig&&(N.setup.generationConfig.translationConfig=
k.translationConfig),console.log("Sending setup:",JSON.stringify(N)),this.ws.send(JSON.stringify(N))},
this.ws.onmessage=$=>this._handleMessage($),this.ws.onerror=$=>{console.error("Gemini Live WebSocket\
 error:",$),this.onError&&this.onError($)},this.ws.onclose=$=>{console.log("Gemini Live WebSocket cl\
osed:",$.code,$.reason),this.closedEvent=$,this.onClose&&this.onClose($)},L&&L.stream&&L.ctx&&L.source){
this.stream=L.stream,this.audioContext=L.ctx;var T=L.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(On());const $=hi(this.stream,16e3);this.audioContext=$.ctx;var T=$.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=$=>{$.data.
size>0&&this.userAudioChunks.push($.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
$=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const N=bi($.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!N||!N.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(N))),mimeType:"audio/\
pcm;rate=16000"}}}))},T.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
let g=null;try{g=JSON.parse(typeof m.data=="string"?m.data:new TextDecoder().decode(m.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const b=g.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(k=>{if(k.text&&(k.thought?(console.log(
"Gemini thought delta:",k.text),this.assistantThought+=k.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(k.text)),k.inlineData&&k.inlineData.data){const L=k.inlineData.data;
console.log("Gemini audio chunk received, size:",L.length),this.rtPlayer&&this.rtPlayer.addChunk(L);
const T=atob(L),A=new Uint8Array(T.length);for(let $=0;$<T.length;$++)A[$]=T.charCodeAt($);this.assistantAudioChunks.
push(A)}}),b.outputTranscription&&b.outputTranscription.text&&this._appendAssistantText(b.outputTranscription.
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
null,this._stopped=new Promise(b=>{let k=!1;const L=r(()=>{k||(k=!0,b())},"finish");if(!m||m.state===
"inactive"){L();return}m.addEventListener("stop",L,{once:!0});try{m.stop()}catch{L()}setTimeout(L,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=r(
()=>{if(this.stream&&this.stream.getTracks().forEach(b=>{try{b.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const m=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(m),b=new Blob(
this.userAudioChunks),k=await this._blobToBase64(b);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:k}}_blobToBase64(m){
return new Promise(g=>{const b=new FileReader;b.onloadend=()=>g(b.result.split(",")[1]),b.readAsDataURL(
m)})}};r(Wn,"GeminiLiveClient");let Un=Wn;const Vn=class Vn{constructor(m=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:m}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(m){if(!this.ctx)return;const g=atob(m),b=new Uint8Array(g.length);for(let N=0;N<
g.length;N++)b[N]=g.charCodeAt(N);const k=new Int16Array(b.buffer),L=new Float32Array(k.length);for(let N=0;N<
k.length;N++)L[N]=k[N]/32768;const T=this.ctx.createBuffer(1,L.length,this.ctx.sampleRate);T.getChannelData(
0).set(L),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();
A.buffer=T,A.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const $=Math.max(this.ctx.currentTime,this.nextStartTime);A.start($),this.
nextStartTime=$+T.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};r(Vn,"RealTimeAudioPl\
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
value:k}=await this.reader.read();if(b)break;g+=m.decode(k,{stream:!0});let L;for(;(L=g.indexOf(`

`))>=0;){const T=g.slice(0,L);g=g.slice(L+2);for(const A of T.split(`
`)){if(!A.startsWith("data: "))continue;let $=null;try{$=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent($)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new wn(this.rateOut||24e3),tn=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}tn=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(On()).then(m=>{this.stream=m;const g=this.rateIn||24e3,b=hi(m,g);this.audioCtx=
b.ctx;const k=b.source,L=this.audioCtx.sampleRate,T=4096;this.processor=this.audioCtx.createScriptProcessor(
T,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const $=A.inputBuffer.
getChannelData(0),N=bi($,L,g);!N||!N.byteLength||this._sendAudio(N)},k.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(m){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(m)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const m=this.audioQueue.splice(0,this.audioQueue.length),
g=m.reduce(($,N)=>$+N.byteLength,0),b=new Uint8Array(g);let k=0;m.forEach($=>{b.set($,k),k+=$.byteLength});
const L="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),T={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(T):T;try{await fetch(L,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}Pn(),zt()}async stop(){if(!this.active)return;if(this.
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
"mic-btn");m&&(m.classList.remove("bg-red-600","animate-pulse"),m.classList.add("bg-gray-700"))}};r(
Jn,"RealtimeVoiceSession");let zn=Jn;function hi(d,m){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let b=null;try{return b=new g({sampleRate:m}),{ctx:b,
source:b.createMediaStreamSource(d)}}catch{if(b)try{b.close()}catch{}return b=new g,{ctx:b,source:b.
createMediaStreamSource(d)}}}r(hi,"openMicAudioSource");function bi(d,m,g){let b=d;if(m!==g&&m>0&&g>
0){const L=m/g,T=Math.floor(b.length/L),A=new Float32Array(T);for(let $=0;$<T;$++)A[$]=b[Math.min(Math.
floor($*L),b.length-1)];b=A}const k=new Int16Array(b.length);for(let L=0;L<b.length;L++){const T=Math.
max(-1,Math.min(1,b[L]));k[L]=T<0?T*32768:T*32767}return k.buffer}r(bi,"pcm16FromFloat32");const xn=new zn,
yi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,k=null,L=!1,T=null,A=!1,$=0,N=0,F=null,Y="idle",oe=!1,le=null;const U=r(D=>document.getElementById(
D),"$"),he=r(D=>{const ie=Object.assign({},D||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(ie):(ie.progressSpinner=
!1,ie)},"noSpinner");function q(D,ie){Y=ie;const $e=U("lyria-status-text"),ge=U("lyria-status-dot");
$e&&($e.textContent=D),ge&&(ge.className="w-2 h-2 rounded-full inline-block "+(g[ie]||g.idle)),_e(),
ae()}r(q,"setStatus");function fe(){const D=N?Math.floor((Date.now()-N)/1e3):0,ie=String(Math.floor(
D/60)).padStart(2,"0"),$e=String(D%60).padStart(2,"0");return`${ie}:${$e}`}r(fe,"formatElapsed");function J(){
N||(N=Date.now());const D=U("lyria-elapsed");D&&(D.textContent=fe()),F||(F=window.setInterval(()=>{const ie=U(
"lyria-elapsed");ie&&(ie.textContent=fe())},1e3))}r(J,"startElapsedTimer");function be(){F&&(window.
clearInterval(F),F=null)}r(be,"stopElapsedTimer");function _e(){const D=U("lyria-play-btn"),ie=U("ly\
ria-pause-btn"),$e=U("lyria-stop-btn"),ge=U("lyria-reset-btn"),Oe=!!b,Ge=Y==="streaming"||Y==="conne\
cting";if(D){D.disabled=oe;const tt=D.querySelector("i");tt&&(tt.className="fas fa-play")}ie&&(ie.disabled=
oe||!Ge),$e&&($e.disabled=oe||!Oe||!Ge),ge&&(ge.disabled=oe||!Oe||!Ge)}r(_e,"updateTransportButtons");
function ae(){const D=U("lyria-save-btn");if(!D)return;const ie=!!b&&Y!=="idle"&&Y!=="connecting"&&Y!==
"error";D.classList.toggle("hidden",!ie)}r(ae,"updateSaveButton");function ve(D,ie){const $e=U("lyri\
a-prompt-rows");if(!$e)return;const ge=document.createElement("div");ge.className="flex items-center\
 gap-2",ge.innerHTML=`
                        <input type="text" value="${escapeHtml(D||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof ie=="nu\
mber"?ie:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof ie=="number"?ie:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Oe=ge.querySelector('input[type="range"]'),Ge=ge.querySelector(".lyria-w\
eight-label");Oe&&Ge&&Oe.addEventListener("input",()=>{Ge.textContent=parseFloat(Oe.value).toFixed(1)});
const tt=ge.querySelector(".lyria-prompt-remove");tt&&tt.addEventListener("click",()=>{$e.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ge.remove()}),ge.classList.add("lyria-prompt-row-wrap"),$e.appendChild(
ge)}r(ve,"addPromptRow");function Ne(){const D=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),ie=[];return D.forEach($e=>{const ge=$e.querySelector('input[type="text"]'),Oe=$e.
querySelector('input[type="range"]'),Ge=(ge?ge.value:"").trim();Ge&&ie.push({text:Ge,weight:parseFloat(
Oe?Oe.value:1)||1})}),ie}r(Ne,"collectPrompts");function Re(){const D={},ie=r(Ni=>{const Xn=U(Ni);return Xn&&
Xn.value!==""?parseFloat(Xn.value):void 0},"num"),$e=ie("lyria-bpm");$e!==void 0&&(D.bpm=Math.round(
$e));const ge=ie("lyria-guidance");ge!==void 0&&(D.guidance=ge);const Oe=ie("lyria-density");Oe!==void 0&&
(D.density=Oe);const Ge=ie("lyria-brightness");Ge!==void 0&&(D.brightness=Ge);const tt=ie("lyria-tem\
perature");tt!==void 0&&(D.temperature=tt);const Ve=U("lyria-scale");Ve&&Ve.value&&(D.scale=Ve.value);
const it=U("lyria-mode");it&&it.value&&(D.music_generation_mode=it.value);const bt=U("lyria-mute-bas\
s"),Wt=U("lyria-mute-drums"),Si=U("lyria-only-bass-drums");return bt&&(D.mute_bass=bt.checked),Wt&&(D.
mute_drums=Wt.checked),Si&&(D.only_bass_and_drums=Si.checked),D}r(Re,"collectConfig");function rt(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([ie,$e])=>{const ge=U(ie),Oe=U($e);!ge||!Oe||ge.addEventListener("input",()=>{const Ge=parseFloat(
ge.value);Oe.textContent=ie==="lyria-bpm"?String(Math.round(Ge)):Ge.toFixed(1)})})}r(rt,"bindRangeLa\
bels");function Ke(){if(T){try{T.close()}catch{}T=null}A=!1,$=0}r(Ke,"resetPlayback");function ht(){
if(L=!1,k&&typeof k.abort=="function")try{k.abort()}catch{}k=null}r(ht,"closeStream");async function Nt(){
ht(),k=new AbortController,L=!0;try{const D=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(
b)}`,he({method:"GET",signal:k.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!D.
ok){const Oe=await D.json().catch(()=>({}));throw new Error(Oe.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const ie=D.body.getReader(),$e=new TextDecoder;let ge="";for(;L;){const{done:Oe,value:Ge}=await ie.read();
if(Oe)break;ge+=$e.decode(Ge,{stream:!0});const tt=ge.split(`

`);ge=tt.pop();for(const Ve of tt){const it=Ve.split(`
`).find(Wt=>Wt.startsWith("data: "));if(!it)continue;const bt=it.slice(6);try{const Wt=JSON.parse(bt);
Ee(Wt)}catch{}}}}catch(D){if(D&&D.name==="AbortError")return;L&&(q("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{L&&b&&Nt()},1200))}finally{L=!1}}r(Nt,"openStream");function Ee(D){
if(D&&D.snapshot){const ie=D.status;if(ie==="error"){q("\u30A8\u30E9\u30FC","error"),be();return}if(ie===
"closed"||ie==="stopped"){q("\u7D42\u4E86","closed"),be();return}q(ie==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",ie==="paused"?"paused":"connecting");return}if(D&&D.audio){q("\u518D\u751F\u4E2D...",
"streaming"),J(),Se(D.audio);return}if(D&&D.error){q("\u30A8\u30E9\u30FC: "+D.error,"error"),be();return}
if(D&&D.final){q("\u7D42\u4E86","closed"),be(),_e();return}}r(Ee,"handleStreamMessage");function Se(D){
if(!D)return;if(!T){const Ve=window.AudioContext||window.webkitAudioContext;if(!Ve)return;T=new Ve({
sampleRate:48e3}),A=!1,$=0}let ie;try{const Ve=atob(D);ie=new Uint8Array(Ve.length);for(let it=0;it<
Ve.length;it++)ie[it]=Ve.charCodeAt(it)}catch{return}const $e=new Int16Array(ie.buffer),ge=Math.floor(
$e.length/2);if(ge<1)return;const Oe=T.createBuffer(2,ge,48e3);for(let Ve=0;Ve<2;Ve++){const it=Oe.getChannelData(
Ve);for(let bt=0;bt<ge;bt++)it[bt]=$e[bt*2+Ve]/32768}T.state==="suspended"&&T.resume();const Ge=T.createBufferSource();
Ge.buffer=Oe,Ge.connect(T.destination),A||($=T.currentTime+.08,A=!0);const tt=Math.max(T.currentTime,
$);Ge.start(tt),$=tt+Oe.duration}r(Se,"playChunk");async function qe(D,ie){const $e=await fetch("/ap\
i/gemini/music/command",he({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:D},ie||{}))})),ge=await $e.json().catch(()=>({}));if(!$e.ok)throw new Error(
ge.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ge}
r(qe,"apiCommand");async function et(){if(oe)return;const D=Ne();if(!D.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}oe=!0,_e(),q("\u63A5\u7D9A\u4E2D...","connecting");try{
const ie=await fetch("/api/gemini/music/start",he({method:"POST",headers:{"Content-Type":"applicatio\
n/json"},body:JSON.stringify({weighted_prompts:D,config:Re()})})),$e=await ie.json().catch(()=>({}));
if(!ie.ok)throw new Error($e.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=$e.session_id,le=Re(),q("\u63A5\u7D9A\u4E2D...","connecting"),Nt()}catch(ie){q("\u30A8\u30E9\u30FC: "+
ie.message,"error"),showToast("Lyria RealTime: "+ie.message,"error",!0)}finally{oe=!1,_e()}}r(et,"st\
artSession");async function lt(D){if(b){oe=!0,_e();try{await qe("control",{action:D}),D==="PLAY"?q("\
\u518D\u751F\u4E2D...","streaming"):D==="PAUSE"?q("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):D==="ST\
OP"?q("\u505C\u6B62\u4E2D","stopped"):D==="RESET_CONTEXT"&&q("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(ie){showToast("Lyria RealTime: "+ie.message,"error",!0),q("\u30A8\u30E9\u30FC: "+
ie.message,"error")}finally{oe=!1,_e()}}}r(lt,"control");async function Le(){if(!b)return;const D=Ne();
if(!D.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}oe=!0;try{await qe("prompts",{weighted_prompts:D}),q("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
Y==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(ie){showToast("Lyria RealTime: "+ie.message,"error",!0)}finally{oe=!1,_e()}}r(Le,"a\
pplyPrompts");async function Be(){if(!b)return;const D=Re(),ie=le||{},$e=D.bpm!==void 0&&D.bpm!==ie.
bpm,ge=D.scale!==void 0&&D.scale!==ie.scale,Oe=$e||ge;oe=!0;try{await qe("config",{config:D,reset_context:Oe}),
le=D,q(Oe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",Y==="paused"?"paused":"streaming"),showToast(
Oe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Ge){showToast("Lyria RealT\
ime: "+Ge.message,"error",!0)}finally{oe=!1,_e()}}r(Be,"applyConfig");async function Ue(){if(b){oe=!0,
q("\u4FDD\u5B58\u4E2D...","connecting"),_e();try{const D=await fetch("/api/gemini/music/save",he({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),ie=await D.json().catch(()=>({}));if(!D.ok)throw new Error(ie.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
q("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),be(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),ie.thread_id&&(currentThreadId=String(ie.thread_id),history.pushState({},"","/c/"+ie.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(ie.thread_id||currentThreadId),_n(
!0)}catch(D){q("\u30A8\u30E9\u30FC: "+D.message,"error"),showToast("Lyria RealTime: "+D.message,"err\
or",!0)}finally{oe=!1,_e()}}}r(Ue,"saveSession");async function Xe(){if(ht(),b)try{await fetch("/api\
/gemini/music/cancel",he({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:b})}))}catch{}b=null,be(),Ke(),q("\u6E96\u5099\u5B8C\u4E86","idle")}r(Xe,"cancelSession");
function mt(){const D=U("lyria-prompt-rows");D&&(D.innerHTML=""),ve("",1),le=null,N=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ge=>{const Oe=U(ge);
Oe&&(Oe.value=ge==="lyria-bpm"?"120":ge==="lyria-guidance"?"4":ge==="lyria-temperature"?"1.1":"0.5")});
const ie=U("lyria-scale");ie&&(ie.value="");const $e=U("lyria-mode");$e&&($e.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ge=>{const Oe=U(ge);Oe&&(Oe.checked=
!1)}),rt()}r(mt,"resetControls");function _n(D){ht(),b&&fetch("/api/gemini/music/cancel",he({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),
b=null,L=!1,be(),Ke(),hideModal("lyria-studio-modal")}r(_n,"closeAndCleanup");function nn(D){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const ie=U("lyria-studio-modal");if(ie&&ie.classList.contains("modal-open")&&b){
if(D&&typeof D=="string"){const ge=U("lyria-prompt-rows");ge&&(ge.innerHTML=""),ve(D,1)}return}if(b&&
Xe(),mt(),D&&typeof D=="string"){const ge=U("lyria-prompt-rows");ge&&(ge.innerHTML=""),ve(D,1)}b=null,
L=!1,be(),Ke(),q("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}r(nn,"open");function Sn(){
const D=U("lyria-open-studio-btn");D&&D.addEventListener("click",()=>nn(""));const ie=U("lyria-studi\
o-close");ie&&ie.addEventListener("click",()=>_n(!1));const $e=U("lyria-play-btn");$e&&$e.addEventListener(
"click",()=>{if(!b){et();return}lt("PLAY")});const ge=U("lyria-pause-btn");ge&&ge.addEventListener("\
click",()=>lt("PAUSE"));const Oe=U("lyria-stop-btn");Oe&&Oe.addEventListener("click",()=>lt("STOP"));
const Ge=U("lyria-reset-btn");Ge&&Ge.addEventListener("click",()=>lt("RESET_CONTEXT"));const tt=U("l\
yria-add-prompt-btn");tt&&tt.addEventListener("click",()=>ve("",1));const Ve=U("lyria-apply-prompts-\
btn");Ve&&Ve.addEventListener("click",Le);const it=U("lyria-apply-config-btn");it&&it.addEventListener(
"click",Be);const bt=U("lyria-save-btn");bt&&bt.addEventListener("click",Ue),rt(),mt(),window.openLyriaStudio=
nn}r(Sn,"init");function Kn(){oe||_n(!1)}return r(Kn,"requestClose"),{init:Sn,open:nn,requestClose:Kn}})();
yi.init(),window.closeLyriaStudio=()=>yi.requestClose(),(()=>{let d=null,m=null,g=null;const b="voic\
eDockSettingsOpen",k="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
L=r(ae=>document.getElementById(ae),"$");function T(){return isStsModel()&&voiceStudioUiEnabled!==!1}
r(T,"isStudioMode");function A(){const ae=get("model-select")?get("model-select").value:"",ve=L("voi\
ce-studio-title");ve&&(ae==="gpt-transcribe"||ae==="gpt-live-transcribe"||ae==="grok-voice-transcrib\
e-2.0-file"?ve.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ae===
"gemini-3.5-live-translate-preview"?ve.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
ve.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}r(A,"updateTitle");function $(){const ae=L("v\
oice-studio-transcript");ae&&(ae.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${k}</div>`);const ve=L("sts-live-transcript");ve&&(ve.innerHTML="",ve.classList.add("hidden"))}
r($,"resetTranscript");function N(ae,ve,Ne){const Re=ae.querySelectorAll(".voice-studio-line");let rt=null;
for(let Ke=Re.length-1;Ke>=0;Ke--)if(Re[Ke].dataset.role===ve){rt=Re[Ke];break}if(rt)rt.innerHTML=Ne;else{
const Ke=ae.querySelector(".voice-studio-placeholder");Ke&&Ke.remove();const ht=document.createElement(
"div");ht.className="voice-studio-line",ht.dataset.role=ve,ht.innerHTML=Ne,ae.appendChild(ht)}ae.classList.
remove("hidden"),ae.scrollTop=ae.scrollHeight}r(N,"writeLine");function F(ae,ve){if(!ve||!String(ve).
trim()||!T())return;const rt=`<span class="${ae==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ae==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
ve)}</span>`;[L("voice-studio-transcript"),L("sts-live-transcript")].filter(Boolean).forEach(Ke=>N(Ke,
ae,rt))}r(F,"log");function Y(ae,ve=!0){const Ne=L("sts-panel"),Re=L("sts-settings-toggle");if(Ne&&Ne.
classList.toggle("settings-open",!!ae),Re&&Re.setAttribute("aria-expanded",ae?"true":"false"),ve)try{
localStorage.setItem(b,ae?"1":"0")}catch{}}r(Y,"setSettingsOpen");function oe(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}r(oe,"readSettingsOpen");let le=null;function U(){const ae=get("mod\
el-select")?get("model-select").value:"";ae!==le&&(le=ae,$()),Y(oe(),!1),A()}r(U,"syncDock");function he(){
const ae=L("sts-panel"),ve=L("voice-studio-panel-host");ae&&ve&&ae.parentNode!==ve&&(d=ae.parentNode,
m=ae.nextSibling,ve.appendChild(ae));const Ne=L("file-preview"),Re=L("voice-studio-file-host");Ne&&Re&&
Ne.parentNode!==Re&&(g=Ne.parentNode,Re.appendChild(Ne),Re.classList.remove("hidden"))}r(he,"movePan\
elIntoModal");function q(){const ae=L("sts-panel");ae&&d&&ae.parentNode!==d&&(m&&m.parentNode===d?d.
insertBefore(ae,m):d.appendChild(ae));const ve=L("file-preview");ve&&g&&ve.parentNode!==g&&g.appendChild(
ve);const Ne=L("voice-studio-file-host");Ne&&Ne.classList.add("hidden"),d=null,m=null,g=null}r(q,"mo\
vePanelBack");function fe(){if(!T())return;he();const ae=L("sts-panel");ae&&ae.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}r(fe,"open");function J(){window.
VoiceStudioOpen=!1,q(),hideModal("voice-studio-modal")}r(J,"close");function be(){window.VoiceStudioOpen&&
J()}r(be,"closeIfOpen");function _e(){window.VoiceStudioOpen=!1;const ae=L("voice-studio-open-btn");
ae&&ae.addEventListener("click",()=>fe());const ve=L("voice-studio-close");ve&&ve.addEventListener("\
click",()=>J());const Ne=L("sts-settings-toggle");Ne&&Ne.addEventListener("click",()=>{const Re=L("s\
ts-panel");Y(!(Re&&Re.classList.contains("settings-open")))}),window.VoiceStudio={open:fe,close:J,closeIfOpen:be,
log:F,isStudioMode:T,syncDock:U},U()}return r(_e,"init"),{init:_e}})().init();let tn=null;function vi(){
if(tn&&(tn.stop(),tn=null),yn){try{yn.pause()}catch{}try{yn.src=""}catch{}yn=null}}r(vi,"stopStsPlay\
back");async function ji(d){vi();const m=new Audio;return m.src=d,m.preload="auto",m.autoplay=!0,m.playsInline=
!0,yn=m,await m.play(),new Promise(g=>{m.onended=()=>g("ended"),m.onerror=()=>g("error")})}r(ji,"pla\
yStsAudio");function $i(){if(xn.isActive()){xn._cancel();return}if(je){je.stop(),je=null,vi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),zt();return}ot&&ot.state===
"recording"&&(qn=!0,ot.stop())}r($i,"cancelRecording");function On(){if(isStsModel())return{audio:!0};
const m=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return m.echoCancellation&&(g.echoCancellation=!1),m.
noiseSuppression&&(g.noiseSuppression=!1),m.autoGainControl&&(g.autoGainControl=!1),{audio:g}}r(On,"\
getMicCaptureConstraints");function Ii(){const d=window.AudioContext||window.webkitAudioContext;if(!d||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let m=null;try{m=new d({sampleRate:16e3})}catch{if(m){try{m.close()}catch{}m=null}m=new d}const g=m.
resume(),b=navigator.mediaDevices.getUserMedia(On());return{ctx:m,resumePromise:g,streamPromise:b}}r(
Ii,"beginGeminiLiveMic");function Pi(d){if(d){if(d.streamPromise)d.streamPromise.then(m=>{try{m.getTracks().
forEach(g=>g.stop())}catch{}}).catch(()=>{});else if(d.stream)try{d.stream.getTracks().forEach(m=>m.
stop())}catch{}if(d.ctx)try{d.ctx.close()}catch{}}}r(Pi,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(je){setStsStatus("Processing...",!0);const d=je;je=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const m=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(m.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
m.assistant_text=(d.inputTranscript||"").trim(),m.assistant_thought="",!m.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const g=(m.user_text||
"").trim()||(m.assistant_text||"").trim()||(m.assistant_thought||"").trim(),b=!!(m.audio_base64||m.user_audio_base64);
if(!g&&!b){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const L=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(L.id),history.pushState(
{},"","/c/"+L.id),get("welcome-screen").classList.add("hidden")}m.thread_id=currentThreadId,m.model=
get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error("Failed to save G\
emini Live session:",m),setStsStatus("Error saving",!1)}return}if(xn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),xn.stop();return}if(ot&&
ot.state==="recording"){ot.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Ut("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ii();
const k={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(k.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const q=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(fe=>fe.trim()).filter(Boolean);q.length&&(k.custom_vocabulary=
q.slice(0,1e3))}}else k.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(k.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",k.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(k.target_lang=get("sts-target-lang").value);const L=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:L,ctx:g.ctx,source:g.ctx.createMediaStreamSource(L)};const T=await apiFetch(
"/api/gemini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
k)});if(!T.ok)throw new Error("Failed to get session token");const{token:A,url:$}=await T.json(),N=get(
"model-select").value,F=get("sts-voice")?get("sts-voice").value:"Kore",Y=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",oe=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(je=new Un,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(je.rtPlayer=new wn),isGeminiLiveTranscribeModel()){
const q=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",fe={languageCodes:[]};
if((q==="SMART"||q==="VERBATIM")&&(fe.mode=q),get("sts-custom-vocab")){const J=get("sts-custom-vocab").
value.split(/[,、\n]/).map(be=>be.trim()).filter(Boolean);J.length&&(fe.customVocabulary=J.slice(0,
1e3))}await je.start(A,$,N,{transcriptionConfig:fe},g)}else if(isGeminiLiveTranslateModel()){const q=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await je.start(A,$,N,{translationConfig:{targetLanguageCode:q,
echoTargetLanguage:!0}},g)}else{const q={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:F}}}};
isGeminiLiveExtendedThinkingModel()&&(q.thinkingConfig={thinkingLevel:Y,includeThoughts:oe}),await je.
start(A,$,N,q,g)}g=null,ot=je.backupRecorder,ot.onstop=()=>{je&&get("mic-btn").click()};let le=!0,U="\
live-sts-"+Date.now();je.onMessage=q=>{if(q.serverContent){if(isGeminiLiveTranscribeModel()){const fe=je.
interimInputTranscript,J=je.inputTranscript,be=J+(fe&&!J.endsWith(fe)?(J?`
`:"")+fe:""),_e=get("chat-messages");let ae=document.getElementById(U);ae||(ae=document.createElement(
"div"),ae.id=U,ae.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ae.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,_e.appendChild(ae),_e.scrollTop=_e.scrollHeight);const ve=ae.
querySelector(".message-content");ve.innerText=be||"\u8074\u304D\u53D6\u308A\u4E2D...",_e.scrollTop=
_e.scrollHeight,window.VoiceStudio&&J&&window.VoiceStudio.log("user",J);return}if(q.serverContent.modelTurn){
le&&(setStsStatus("Gemini is speaking...",!1),le=!1);const fe=get("chat-messages");let J=document.getElementById(
U);J||(J=document.createElement("div"),J.id=U,J.className="flex flex-col gap-2 mb-4 assistant-messag\
e bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",J.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,fe.appendChild(J),fe.scrollTop=fe.scrollHeight);const be=J.
querySelector(".thought-container"),_e=J.querySelector(".message-content");je.assistantThought&&(be.
classList.remove("hidden"),be.innerText=je.assistantThought),_e.innerText=je.assistantText,fe.scrollTop=
fe.scrollHeight,window.VoiceStudio&&(je.inputTranscript&&window.VoiceStudio.log("user",je.inputTranscript),
je.assistantText&&window.VoiceStudio.log("assistant",je.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),fi(je.stream),gi(je.stream);const he=je;he.onError=q=>{const fe=q&&q.message?q.message:
typeof q=="string"?q:"";fe&&showToast("Gemini Live: "+fe,"error",!0)},he.onClose=q=>{if(je===he){if(q&&
q.code&&q.code!==1e3){const fe=q.reason?": "+q.reason:" (code "+q.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+fe,"error",!0)}get("mic-btn").click()}},he.closedEvent&&
he.onClose(he.closedEvent);return}catch(b){if(Pi(g),je){try{je.stop()}catch{}je=null}showToast("Gemi\
ni Live connection failed: "+b.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await xn.start();return}isStsModel()||(pi(),Ut("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(On());ot=new MediaRecorder(d),Tn=[],qn=!1;const m=isStsModel();
ot.ondataavailable=g=>Tn.push(g.data),ot.onstop=async()=>{if(qn){Tn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(T=>T.stop()),Pn(),zt(),m||(Ut("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),In=setTimeout(()=>Ut("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(Tn,{type:"audio/web\
m"}),b=new File([g],"recording.webm",{type:"audio/webm"}),k=new FormData;k.append("file",b),get("fil\
e-preview").classList.remove("hidden");const L=m;get("file-name").innerText=L?"Processing voice...":
"Transcribing...";try{if(L){if(!currentThreadId){const q=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
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
"/sts",{method:"POST",body:k});if(!T.ok){const he=await T.json().catch(()=>({}));throw new Error(he.
error||"Speech-to-speech failed")}const A=T.body.getReader(),$=new TextDecoder;let N="",F=null,Y=null;
stsOpt("sts-auto-play")&&(Y=new wn,tn=Y),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let oe=!0,le="",U="";for(;;){const{done:he,value:q}=await A.read();if(he)break;
N+=$.decode(q,{stream:!0});const fe=N.split(`
`);N=fe.pop();for(const J of fe){if(!J.trim())continue;const be=JSON.parse(J);if(be.error)throw new Error(
be.error);be.audio_delta&&Y&&(oe&&(setStsStatus("Playing response...",!1),oe=!1),await Y.addChunk(be.
audio_delta)),be.input_delta&&(le+=be.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
le)),be.transcript_delta&&(U+=be.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",U)),(be.final||be.audio_url)&&(F=be)}}window.VoiceStudio&&!le.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),F&&(F.audio_url||F.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const T=get(
"set-mic-transcribe-mode");if(!!(T&&T.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}k.append("llm_model",get("model-select")&&get("model-select").value||"");const N=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:k})).json();if(N.transcript){const F=get("prompt-inp\
ut");F.value+=(F.value?" ":"")+N.transcript,F.style.height="auto",F.style.height=F.scrollHeight+"px"}else
showToast(N.error||"Transcription failed","error",!0)}}catch(T){showToast("Audio processing error: "+
T.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(T=>T.
stop()),Pn(),zt(),L||Ut("","hidden"),L&&setStsStatus("Tap to speak",!1)}},ot.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Ut("\u9332\u97F3\u4E2D\u2026","recording"),fi(d)),gi(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{zt(),isStsModel()||Ut("","hidden"),alert("Microphone access denied or not avail\
able.")}};const kn=r((d,m)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),g=get("lib-download-btn"),b=get("lib-attach-btn"),k=get("lib-rena\
me-btn"),L=get("lib-usage-btn");if(m&&(m.disabled=d===0,kn(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,kn(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=d===0,kn(b,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),k&&(k.disabled=d!==1,kn(k,"\u540D\u524D\u5909\u66F4")),
L&&(L.disabled=d!==1,kn(L,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const T=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",T&&d>0)}};function wi(d){lib.
attachMode=!!d}r(wi,"setLibAttachMode");const xi=r((d=!1)=>{wi(d),showModal("lib-modal"),loadLibraryFiles(),
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
onclick=()=>addGemFixedPromptRow());const Oi=r(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>Oi(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
m=get("gem-desc").value,g=get("gem-inst").value,b=collectGemFixedPrompts();if(d&&g){const k=editingGemUuid?
"PUT":"POST",L=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
L,{method:k,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:g,
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
try{const k=decodeURIComponent(g),L=new Blob([k],{type:"text/plain"}),T=URL.createObjectURL(L),A=document.
createElement("a");A.href=T;let N={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(N="txt");let F=`code.${N}`;b==="dockerfile"&&(F="Docke\
rfile"),b==="makefile"&&(F="Makefile"),A.download=F,document.body.appendChild(A),A.click(),document.
body.removeChild(A),URL.revokeObjectURL(T)}catch(k){console.error("Download failed",k)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),g=m.getAttribute("data-code");g&&
window.copyCode(m,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&ni(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
