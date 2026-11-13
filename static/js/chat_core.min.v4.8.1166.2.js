document.addEventListener("DOMContentLoaded",()=>{var xi,_i;initThemeFromServer(),applyLiquidGlassMode(
INITIAL_LIQUID_GLASS_ENABLED),updateCurrentChatHeaderUi();try{sessionStorage.removeItem("browser_fas\
t_mode_gemini_key")}catch{}const e=get("enable-browser-fast-mode");e&&(e.checked=!1,e.onchange=()=>{
if(e.checked){const u=get("enable-batch-mode");u&&u.checked&&(u.checked=!1),requestBrowserFastModeEnable()}else
setBrowserFastModeEnabled(!1)});const t=get("enable-batch-mode");t&&(t.onchange=()=>{if(t.checked){browserFastModeEnabled&&
setBrowserFastModeEnabled(!1);const u=get("enable-coding-mode");u&&u.checked&&(u.checked=!1,typeof syncCodingModeUi==
"function"&&syncCodingModeUi(!1),showToast("Batch API\u3067\u306FCoding Mode\u3092\u5229\u7528\u3067\u304D\u306A\u3044\u305F\u3081\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0))}updateBatchUi(get("model-select")?get("model-select").value:"")});const n=get("model-\
select");n&&n.addEventListener("change",()=>{setTimeout(()=>{if(!browserFastModeEnabled)return;const u=String(
n.value||"").toLowerCase();browserFastApiKey="",browserFastApiKeyModel="",browserFastBootstrap=null,
!u.startsWith("gemini-")||/(image|native-audio|tts|live|flash-cyber)/.test(u)?(setBrowserFastModeEnabled(
!1),n.dispatchEvent(new Event("change")),showToast("\u5BFE\u8C61\u5916\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u305F\u305F\u3081\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"warning",!0)):applyBrowserFastModeRestrictions()},0)});const i=get("browser-fast-mode-enable-btn");
i&&(i.onclick=async()=>{const u=i.innerHTML;i.disabled=!0,i.innerHTML='<i class="fas fa-spinner fa-s\
pin mr-1"></i>\u4FDD\u5B58\u6E08\u307F\u30AD\u30FC\u3092\u53D6\u5F97\u4E2D...';try{await fetchBrowserFastBootstrap(
!0);const f=get("browser-fast-mode-ignore-warning");if(f&&f.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(f){showToast(f.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{i.disabled=!1,i.innerHTML=u}});const a=get("browser-fast-mode-cancel-btn");a&&(a.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const r=document.getElementById(
"alpha-bar");setTimeout(()=>{if(r){const u=document.getElementById("version-display");if(u){const f=r.
getBoundingClientRect(),g=u.getBoundingClientRect(),b=g.left+g.width/2-(f.left+f.width/2),_=g.top+g.
height/2-(f.top+f.height/2);r.style.transform=`translate(${b}px, ${_}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{u.classList.add("pulse-target"),setTimeout(()=>u.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const u=get(
"gpt-image-options");if(!u)return;isGptImageModel()?u.classList.remove("hidden"):u.classList.add("hi\
dden");const g=(get("model-select").value||"").toLowerCase().includes("gpt-image-2.6"),b=get("gpt-im\
age-format");get("gpt-image-complexity-wrap")&&get("gpt-image-complexity-wrap").classList.toggle("hi\
dden",!g),b&&(Array.from(b.options).forEach(T=>{T.hasAttribute("data-gpt-image-26")&&(T.hidden=!g)}),
!g&&b.value==="gif"&&(b.value="jpeg"));const _=g&&b&&b.value==="gif",S=get("gpt-image-compression-wr\
ap");b&&S&&(b.value==="png"||b.value==="gif"?S.classList.add("hidden"):S.classList.remove("hidden")),
get("gpt-image-gif-wrap")&&get("gpt-image-gif-wrap").classList.toggle("hidden",!_)}o(l,"updateGptIma\
geUi");function c(){const u=get("gemini-image-options");if(!u)return;isGeminiImageModel()?u.classList.
remove("hidden"):u.classList.add("hidden");const g=(get("model-select").value||"").toLowerCase().includes(
"gemini-3.1-flash-lite-image");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{
b&&(Array.from(b.options).forEach(_=>{_.value!=="1K"&&(_.disabled=g)}),g&&b.value!=="1K"&&(b.value="\
1K"))})}o(c,"updateGeminiImageUi");function d(){const u=get("grok-image-options");if(!u)return;const f=(get(
"model-select").value||"").toLowerCase(),g=isGrokImageModel(),b=f==="grok-imagine-image-quality"||f===
"grok-imagine-image-2.0",_=f==="grok-imagine-image-2.0";if(g){u.classList.remove("hidden");const T=get(
"grok-image-resolution")?get("grok-image-resolution").parentElement:null;T&&T.classList.toggle("hidd\
en",!b);const A=get("grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.
toggle("hidden",!_)}else u.classList.add("hidden");if(get("modal-grok-image-options")){const T=get("\
modal-grok-image-resolution")?get("modal-grok-image-resolution").parentElement:null;T&&T.classList.toggle(
"hidden",!b);const A=get("modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;
A&&A.classList.toggle("hidden",!_)}}o(d,"updateGrokImageUi");function m(){var O,F;const u=get("ideog\
ram-image-options"),f=get("modal-ideogram-image-options"),g=get("model-select")&&get("model-select").
value||"",b=isIdeogramModel(g),_=ideogramModelTraits(g),S={resolution:_.resolution,quality:_.quality,
speed:_.speed,style:_.style,negative:_.negative};if(u&&u.classList.toggle("hidden",!b),f&&f.classList.
toggle("hidden",!b),updateIdeogramPromptInputUi(),!b)return;const T=((O=get("ideogram-image-operatio\
n"))==null?void 0:O.value)==="remove_background",A=((F=get("ideogram-image-operation"))==null?void 0:
F.value)==="remove_object";IDEOGRAM_IMAGE_FIELDS.forEach(Z=>{const ae=Z==="operation"||A&&Z==="seed"||
!(T||A)&&S[Z]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(le=>{const V=get(le+Z);!V||!V.
parentElement||V.parentElement.classList.toggle("hidden",!ae)})}),["ideogram-image-style","modal-ide\
ogram-image-style"].forEach(Z=>{const ae=get(Z);if(!ae)return;const le=_.styleOptions.join("|");if(ae.
dataset.options!==le){const V=ae.value;ae.innerHTML=_.styleOptions.map(we=>`<option value="${we}">${we===
"auto"?"Auto":we}</option>`).join(""),ae.dataset.options=le,ae.value=_.styleOptions.includes(V)?V:"a\
uto"}});const I=get("ideogram-image-note");I&&(I.textContent=T?"\u203B \u80CC\u666F\u3092\u524A\u9664: \u753B\u50CF1\u679A / 25MB\u307E\u3067 / PNG\u30FBJPEG\u30FBWE\
BP\u3002\u6DFB\u4ED8\u304C\u306A\u3044\u5834\u5408\u306F\u8868\u793A\u4E2D\u306E\u30D6\u30E9\u30F3\u30C1\u306E\u6700\u5F8C\u306E\u753B\u50CF\u3002\u9001\u4FE1\u30DC\u30BF\u30F3\u3060\u3051\u3067\u5B9F\u884C\u3057\u3001\u900F\u904EPNG\u3067\u51FA\u529B":
A?"\u203B Object Remover: \u753B\u50CF1\u679A + \u540C\u3058\u30B5\u30A4\u30BA\u306E\u767D\u9ED2\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059\u3002\u30DE\u30B9\u30AF\u306E\u767D\u3044\u90E8\u5206\u3092\u524A\u9664\u3057\u307E\u3059":
_.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F Precise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":
"\u203B Ideogram: \u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(m,"updateIdeogramImageUi"),["ideogram-image-operation","modal-ideogram-image-operation"].forEach(u=>{
var f;(f=get(u))==null||f.addEventListener("change",()=>{const g=get(u.startsWith("modal-")?"ideogra\
m-image-operation":"modal-ideogram-image-operation");g&&(g.value=get(u).value),m(),C()})});function h(){
var b;const u=get("grok-video-options");if(!u)return;const f=String(((b=get("model-select"))==null?void 0:
b.value)||"").toLowerCase();isGrokVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");
const g=get("grok-video-resolution");if(g){const _=Array.from(g.options).find(T=>T.value==="1080p"),
S=["grok-imagine-video-1.5","grok-imagine-video-1.5-lite"].includes(f);_&&(_.disabled=!S),!S&&g.value===
"1080p"&&(g.value="720p")}}o(h,"updateGrokVideoUi");function y(){var _;const u=get("gemini-video-opt\
ions");if(!u)return;const f=String(((_=get("model-select"))==null?void 0:_.value)||"").toLowerCase();
isGeminiVideoModel()?u.classList.remove("hidden"):u.classList.add("hidden");const g=get("gemini-vide\
o-resolution");if(g){const S=Array.from(g.options).find(A=>A.value==="4K"),T=f==="veo-3.1-lite-gener\
ate-preview"||f==="veo-3.1-fast-generate-preview"||f==="gemini-omni-flash";S&&(S.disabled=T),T&&g.value===
"4K"&&(g.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle("hidden",f===
"gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function v(){const u=get("gemini-music-options");
if(!u)return;const f=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!f;u.classList.toggle("hid\
den",!g);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!f)}o(v,"updateGemi\
niMusicUi");function w(){var T;const u=get("xai-chat-options");if(!u)return;const f=String(((T=get("\
model-select"))==null?void 0:T.value)||"").toLowerCase(),g=f.startsWith("grok-")&&!isGrokImageModel(
f)&&!isGrokVideoModel(f)&&!f.includes("voice");u.classList.toggle("hidden",!g);const b=get("xai-logp\
robs"),_=get("xai-top-logprobs"),S=f.includes("grok-4.20");b&&(b.disabled=S,S&&(b.checked=!1)),_&&(_.
disabled=S,S&&(_.value=""))}o(w,"updateXaiChatUi");function k(){const u=isMistralOcrModel(),f=get("m\
istral-ocr-options");f&&f.classList.toggle("hidden",!u);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!u),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const _=get(b);_&&(_.classList.toggle("opacity-50",u),_.classList.toggle("po\
inter-events-none",u))}),u&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function x(){const u=get("image-input-limit\
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
join(""):f==="gemini-nano-banana-2.1"?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2.1 \
\u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u30FB2K\u30FB4K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u52D5\u753B\u3092\u53C2\u8003\u306B\u3057\u305F\u753B\u50CF\u751F\u6210\u306B\u3082\u5BFE\u5FDC\u3057\u307E\u3059</div>"].
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
join("")):isIdeogramModel(f)?(b=!0,g=ideogramModelTraits(f).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u3082\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u751F\u6210\u30FB\u7DE8\u96C6\u3067\u306F\u753B\u50CF\u5165\u529B\u306F\
 Ideogram 4.5 \u306E\u307F\u3002\u80CC\u666F\u524A\u9664\u3068 Object Remover \u306F\u5168\u30E2\u30C7\u30EB\u3067\u5229\u7528\u3067\u304D\u307E\u3059</div>"].
join("")):f.includes("grok")&&f.includes("video")&&(b=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(u.innerHTML=g,u.classList.remove("hidden")):(u.classList.add("hidden"),u.innerHTML="")}
o(x,"updateImageInputLimits");function C(){const u=get("enable-sys-prompt"),f=!!(u&&u.disabled),g=!!(u&&
u.checked);M(),u&&(u.disabled?!f&&g?u.dataset.restoreChecked="1":f||delete u.dataset.restoreChecked:
(f&&u.dataset.restoreChecked==="1"&&(u.checked=!0),delete u.dataset.restoreChecked))}o(C,"toggleOpti\
ons"),window.toggleOptions=C;function L(u){const f=get("thinking-options"),g=get("thinking-level"),b=get(
"reasoning-effort"),_=get("enable-thinking"),S=get("thinking-budget-label"),T=get("thinking-budget"),
A=u==="effort";if(f&&(f.dataset.thinkingMode=A?"effort":"thinking"),g&&g.classList.toggle("hidden",A),
b&&b.classList.toggle("hidden",!A),S&&S.classList.toggle("hidden",A),T&&T.classList.toggle("hidden",
A),!!_)if(A){_.dataset.thinkingChecked===void 0&&(_.dataset.thinkingChecked=_.checked?"1":"0");const I=b?
Array.from(b.options).find(F=>F.value==="none"):null,O=!!(I&&!I.disabled&&!I.classList.contains("hid\
den"));_.checked=O?!!(b&&String(b.value||"").toLowerCase()!=="none"):!0,_.disabled=!O}else _.dataset.
thinkingChecked!==void 0&&(_.checked=_.dataset.thinkingChecked==="1",delete _.dataset.thinkingChecked)}
o(L,"setComposerThinkingMode");function M(){const u=get("model-select");if(!u)return;const f=u.value,
g=String(f||"").toLowerCase(),b=g.includes("deepseek"),_=g.startsWith("glm-"),S=get("thinking-option\
s"),T=get("reasoning-effort-container"),A=get("enable-thinking"),I=get("thinking-level"),O=get("thin\
king-budget"),F=get("enable-search"),Z=get("search-container"),ae=get("url-context-container"),le=get(
"enable-maps"),V=get("maps-grounding-container"),we=get("enable-sys-prompt"),W=get("sys-prompt-optio\
n"),be=get("enable-python"),Y=get("python-container"),ke=get("prompt-cache-container"),Ce=get("enabl\
e-prompt-cache"),se=f==="gpt-5-search-api",xe=f.includes("tts"),Be=isMistralOcrModel(f),je=g.includes(
"gemini-3.1-flash-lite-image"),lt=g==="gemini-nano-banana-2.1",Qe=g.includes("gemini-3.1-flash-image")&&
!je||lt,yt=isClaudeModelKey(f),Ft=g==="gemini-3.8-flash-cyber",Ee=isLlmModel()&&!b&&!xe&&!g.includes(
"realtime")&&!g.includes("native-audio")&&!g.includes("live");ke&&(Ee?(ke.classList.remove("hidden",
"opacity-50","pointer-events-none"),Ce&&(Ce.disabled=!1)):(Ce&&(Ce.checked=!1,Ce.disabled=!0),ke.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),T&&T.classList.
add("hidden"),L("thinking");const Le=get("vision-model-info");if(Le&&Le.classList.add("hidden"),T){const Te=get(
"reasoning-effort");if(Te){Array.from(Te.options).forEach(nt=>{const Re=g==="gpt-5.6"||g.startsWith(
"gpt-5.6-"),ft=g.startsWith("gpt-6"),Sn=g==="gpt-6-sol"||g==="gpt-6-luna"||g==="gpt-6.1-terra",on=g===
"deepseek-v4.1-flash"||g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash\
-vision-exp",Cn=g==="deepseek-v4-pro",Kn=g.includes("grok-4.5"),H=g.includes("grok-4.6");nt.value===
"max"?nt.classList.toggle("hidden",!Re&&!ft&&!on&&!Cn):nt.value==="xhigh"?nt.classList.toggle("hidde\
n",!H&&!g.includes("multi-agent")&&!Re&&!ft):nt.value==="medium"?nt.classList.toggle("hidden",!(g.includes(
"grok-4.3")||Kn||H||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("mu\
lti-agent")||g.includes("gpt-5")||ft||g.includes("o1")||g.includes("o3"))):nt.value==="none"?nt.classList.
toggle("hidden",!g.includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!Sn&&!on&&
!Cn):nt.value==="low"&&nt.classList.toggle("hidden",Cn)});const Fe=Te.selectedOptions&&Te.selectedOptions[0];
Fe&&Fe.classList.contains("hidden")&&(Te.value=b?"high":"medium")}}ae&&ae.classList.add("hidden"),V&&
V.classList.add("hidden"),A&&(A.disabled=!1),O&&(O.disabled=!0,O.classList.add("opacity-50"));const Ve=isGeminiImageModelKey(
f);if(xe||Be)Z&&(get("enable-search").checked=!1,Z.classList.add("opacity-50","pointer-events-none")),
ae&&(get("enable-url-context").checked=!1,ae.classList.add("opacity-50","pointer-events-none")),V&&le&&
(le.checked=!1,V.classList.add("opacity-50","pointer-events-none")),Y&&(be.checked=!1,Y.classList.add(
"opacity-50","pointer-events-none")),we&&W&&(we.checked=!1,we.disabled=!0,W.classList.add("opacity-5\
0"));else if(Qe||je)V&&le&&(le.checked=!1,V.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),lt?(Array.from(I.options).forEach(Te=>{Te.disabled=Te.value==="low"}),[
"minimal","medium","high"].includes(I.value)||(I.value="medium")):(Array.from(I.options).forEach(Te=>{
["low","medium"].includes(Te.value)&&(Te.disabled=!0),["minimal","high"].includes(Te.value)&&(Te.disabled=
!1)}),["minimal","high"].includes(I.value)||(I.value=je?"minimal":"high")),A&&(A.disabled=!1),je&&(F&&
(F.checked=!1,F.disabled=!0),Z&&Z.classList.add("opacity-50","pointer-events-none"));else if(Ve)V&&le&&
(le.checked=!1,V.classList.add("hidden","opacity-50","pointer-events-none"));else if(yt)S.classList.
remove("hidden"),O&&(O.disabled=!1,O.classList.remove("opacity-50")),Array.from(I.options).forEach(Te=>{
Te.disabled=!0}),Y&&(be.checked=!1,Y.classList.add("opacity-50","pointer-events-none"));else if(Ft){
S&&S.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from(I.options).forEach(Fe=>{Fe.
disabled=!["low","medium","high"].includes(Fe.value)}),["low","medium","high"].includes(I.value)||(I.
value="medium"),[Z,ae,V,Y].forEach(Fe=>{Fe&&Fe.classList.add("opacity-50","pointer-events-none")}),[
F,le,be].forEach(Fe=>{Fe&&(Fe.checked=!1,Fe.disabled=!0)});const Te=get("enable-url-context");Te&&(Te.
checked=!1,Te.disabled=!0),we&&W&&(we.disabled=!1,W.classList.remove("opacity-50"))}else if(f.includes(
"gemini")&&!Ve){S.classList.remove("hidden"),ae&&ae.classList.remove("hidden","opacity-50","pointer-\
events-none");const Te=f.includes("gemini-3")||f.includes("gemini-4");V&&(Te?V.classList.remove("hid\
den","opacity-50","pointer-events-none"):(le&&(le.checked=!1),V.classList.add("hidden","opacity-50",
"pointer-events-none")));const Fe=f.includes("flash"),nt=f==="gemini-4-hydrogen";Array.from(I.options).
forEach(Re=>{nt?Re.disabled=!["low","medium","high","xhigh"].includes(Re.value):f==="gemini-4-carbon"||
f==="gemini-4-argon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash"?Re.disabled=!["low","medium","h\
igh"].includes(Re.value):f==="gemini-3.6-flash"?Re.disabled=!["medium","high"].includes(Re.value):f===
"gemini-3.5-flash-lite"?Re.disabled=!["minimal","medium","high"].includes(Re.value):Re.value==="xhig\
h"?Re.disabled=!0:["minimal","medium"].includes(Re.value)?Re.disabled=!Fe:Re.disabled=!1}),nt&&!["lo\
w","medium","high","xhigh"].includes(I.value)?I.value="high":(f==="gemini-4-carbon"||f==="gemini-4-a\
rgon"||f==="gemini-3.8-flash"||f==="gemini-3.7-flash")&&!["low","medium","high"].includes(I.value)||
f==="gemini-3.6-flash"&&!["medium","high"].includes(I.value)?I.value="medium":f==="gemini-3.5-flash-\
lite"&&!["minimal","medium","high"].includes(I.value)?I.value="minimal":!Fe&&["minimal","medium"].includes(
I.value)&&(I.value="high"),Te?A&&(A.checked=!0,A.disabled=!0):A&&(A.disabled=!1),O&&f.includes("gemi\
ni-2.5")&&(O.disabled=!1,O.classList.remove("opacity-50")),O&&!f.includes("gemini-2.5")&&(O.disabled=
!0,O.classList.add("opacity-50"))}if(isLlmModel()&&(g.includes("gpt-5")||g.includes("o1")||g.includes(
"o3")||g.includes("grok-4.3")||g.includes("grok-4.5")||g.includes("grok-4.6")||g.includes("grok-4.20\
-0309-reasoning")||g.includes("grok-build")||g.includes("multi-agent")||g.includes("gpt")&&!g.includes(
"tts")))S&&S.classList.remove("hidden"),L("effort"),Z&&Z.classList.remove("opacity-50","pointer-even\
ts-none");else if(b){S&&S.classList.remove("hidden"),L("effort");const Te=get("vision-model-info");if(Te&&
Te.classList.toggle("hidden",g==="deepseek-v4.1-flash"||g==="deepseek-v4-flash-vision-exp"),F&&(F.disabled=
!1),Z&&Z.classList.remove("opacity-50","pointer-events-none"),ae){const Fe=get("enable-url-context");
Fe&&(Fe.checked=!1),ae.classList.add("opacity-50","pointer-events-none")}V&&le&&(le.checked=!1,V.classList.
add("opacity-50","pointer-events-none"))}else Be||(Z&&Z.classList.remove("opacity-50","pointer-event\
s-none"),V&&le&&(le.checked=!1,V.classList.add("hidden","opacity-50","pointer-events-none")));if(xe?
Y&&Y.classList.add("opacity-50","pointer-events-none"):(Y&&Y.classList.remove("opacity-50","pointer-\
events-none"),(!Ve||Qe)&&!f.includes("gpt-image")&&(we.disabled=!1,W.classList.remove("opacity-50"))),
(Ve&&!Qe||f.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(f)||isGrokVideoModel()||Be)&&
we&&W&&(we.checked=!1,we.disabled=!0,W.classList.add("opacity-50")),Y&&(isLlmModel()?(Y.classList.remove(
"hidden"),be.disabled=!1):(be.checked=!1,be.disabled=!0,Y.classList.add("hidden"))),se?(F&&(F.checked=
!0,F.disabled=!0),Z&&Z.classList.add("opacity-50","pointer-events-none"),Y&&(be.checked=!1,be.disabled=
!0,Y.classList.add("opacity-50","pointer-events-none"))):F&&!f.includes("tts")&&!Be&&!je&&(F.disabled=
!1),Ft){[F,le,be].forEach(Fe=>{Fe&&(Fe.checked=!1,Fe.disabled=!0)});const Te=get("enable-url-context");
Te&&(Te.checked=!1,Te.disabled=!0),[Z,ae,V,Y].forEach(Fe=>{Fe&&Fe.classList.add("opacity-50","pointe\
r-events-none")})}_&&(Le&&Le.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v",
"glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(g)),[F,be].forEach(Te=>{Te&&(Te.checked=!1,Te.
disabled=!0)}),[Z,Y].forEach(Te=>{Te&&Te.classList.add("opacity-50","pointer-events-none")}));const ct=get(
"mask-btn");ct&&(isGptImageModel()||isIdeogramObjectRemoval()?ct.classList.remove("hidden"):(ct.classList.
add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),
l(),c(),d(),m(),h(),y(),v(),updateBatchUi(f),w(),k(),x(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}o(M,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener(
"change",C),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
C(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const E=get("enable-canvas-mode");E&&(E.checked=canvasModeEnabled,E.addEventListener("change",()=>syncCanvasModeUi(
E.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const D=get("enable-codin\
g-mode");D&&(D.checked=codingModeEnabled,D.addEventListener("change",()=>syncCodingModeUi(D.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const N=get(
"prompt-controls-toggle-btn");N&&(N.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(u,f,g){const b=(f||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(u||"");const _=u||"",S=(b||"").toLowerCase();
let T="";try{const W=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&_.length>2e4?T=escapeHtml(
_):T=hljs.highlight(_,{language:W}).value}catch{T=escapeHtml(_)}const A=encodeURIComponent(_).replace(
/'/g,"%27"),I=detectBlockedScriptsInCode(_),O=hashString(`${b||"TEXT"}
${_||""}`);let F="";if(canvasModeEnabled){const W=String(canvasPreviewState.selectedKey||"")===O,be=W?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";F=`<button\
 class="canvas-preview-btn${W?" canvas-active":""}" data-code="${A}" data-code-key="${O}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${be}" aria-label="${be}" aria-pressed="${W?"true":"false"}"\
><i class="fas ${W?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,_)){const W=I?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
F=`<button class="html-preview-btn" data-code="${A}" ${I?'data-suspicious="1"':""} title="${W}" aria\
-label="${W}"><i class="fas ${I?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const Z=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ae=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${O}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,le=(b||"TEXT")+(I?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),V=getRenderableSvgCode(S,_);return`${V?buildSvgCodeRenderHtml(V,O):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${O}"><div class="code-header"\
><span class="code-lang">${le}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${ae}${F}${Z}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${T}</code></pre></d\
iv></div>`},link(u,f,g){return`<a href="${u}" title="${f||""}" target="_blank">${g}</a>`},image(u,f,g){
return buildChatImageHtml(u,{alt:g,title:f})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
u=>{u[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(xi=get("version-update-dismis\
s"))==null||xi.addEventListener("click",()=>{const u=localStorage.getItem("app_version")||"";u&&localStorage.
setItem("version_notified",u),hideModal("version-update-modal")});const G=get("version-update-clear-\
cache");if(G&&(G.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),G.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(G.checked)})),
(_i=get("version-update-reload"))==null||_i.addEventListener("click",async()=>{var f;await versionUpdateCachePreferenceSavePromise.
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
u)}}else{const u=get("turnstile-container");u&&u.classList.add("hidden")}const j=o(u=>{if(!u)return"\
\u4E0D\u660E";const f=new Date(u);return Number.isNaN(f.getTime())?u:f.toLocaleString()},"formatSess\
ionTime"),J=o(u=>{const f=Array.isArray(u)?u:[],g=get("passkey-list"),b=get("passkey-count");if(b&&(b.
innerText=String(f.length)),!!g){if(!f.length){g.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
g.innerHTML="",f.forEach((_,S)=>{const T=_&&_.id?String(_.id):"",A=document.createElement("div");A.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const I=document.
createElement("div");I.className="min-w-0";const O=document.createElement("div");O.className="text-x\
s text-gray-200 truncate",O.innerText=_&&_.name?String(_.name):`Security Key ${S+1}`;const F=document.
createElement("div");F.className="text-[10px] text-gray-500 mt-1",F.innerText=_&&_.created_at?`\u767B\u9332\u65E5\u6642:\
 ${j(_.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",I.appendChild(O),I.appendChild(F),A.appendChild(
I);const Z=document.createElement("button");Z.type="button",Z.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",Z.innerText="\u524A\u9664",Z.
disabled=!T,T&&(Z.onclick=()=>window.removeWebAuthnCredential(T)),A.appendChild(Z),g.appendChild(A)})}},
"renderPasskeyList"),oe=o(u=>{const f=get("session-list");if(f){if(!u||!u.length){f.innerHTML='<div \
class="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}f.innerHTML=u.map(g=>{const b=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",_=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
T=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${_}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(T)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(j(g.last_seen_at))} \
/ \u4F5C\u6210: ${escapeHtml(j(g.created_at))}</div></div>${S}</div>`}).join(""),f.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const b=g.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const _=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let S={};try{S=await _.json()}catch{}if(_.ok){if(S.logged_out){
location.href="/login";return}await me()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),me=o(async()=>{const u=get("session-list");u&&(u.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const f=await apiFetch("/api\
/sessions");let g={};try{g=await f.json()}catch{}if(!f.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}u&&(u.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(g.sessions||[]).filter(_=>!_.is_revoked);oe(b)},"loadSessions"),ie=o(()=>{const u=get(
"session-refresh-btn");u&&(u.onclick=()=>me());const f=get("session-revoke-others-btn");f&&(f.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await me():showToast("\u64CD\u4F5C\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
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
(f.checked=!0),C())}).catch(()=>{}),installAdminSidebarDebugObserver(),isAdminSidebarDebugEnabled())
try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the browser DevTools Console (F12). \
After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}snapshotSidebarHistory(
"page-init"),loadThreads(),loadSkills(),get("send-btn").onclick=()=>{isStopMode?stopGeneration():sendMessage()},
get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindThinkingEffortSync(),bindMinimalOptionsEvents();
const ye=get("vision-model-change-btn");ye&&(ye.onclick=()=>_openVisionModelSelector());const $=get(
"compression-format-only");$&&($.onchange=()=>{const u=$.checked,f=get("compression-max-size"),g=get(
"compression-max-dim");f&&(f.disabled=u),g&&(g.disabled=u);const b=get("compression-size-wrap"),_=get(
"compression-dim-wrap");b&&(b.style.opacity=u?"0.4":"1"),_&&(_.style.opacity=u?"0.4":"1")});const R=o(
()=>{const u=get("enable-temporary-chat");!u||u.dataset.bound==="1"||(u.dataset.bound="1",u.checked=
!!temporaryChatEnabled,u.onchange=async()=>{const f=temporaryChatEnabled;await applyTemporaryChatSetting(
u.checked)||(setTemporaryChatUiState(f),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");R(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const U=get("storage-us\
age-refresh");U&&(U.onclick=()=>loadStorageUsage());let P=null;const q=o(()=>{const u=new Uint8Array(
16);return window.crypto.getRandomValues(u),Array.from(u,f=>f.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),ee=o((u={})=>{const f=get("account-transfer-progress"),g=get("account-tra\
nsfer-progress-bar"),b=get("account-transfer-progress-percent"),_=get("account-transfer-progress-tex\
t"),S=get("account-transfer-progress-detail"),T=Math.max(0,Math.min(100,Number(u.progress)||0));if(f&&
f.classList.remove("hidden"),g&&(g.style.width=`${T}%`),b&&(b.textContent=`${Math.round(T)}%`),_&&(_.
textContent=u.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){
const I={queued:"\u9806\u756A\u5F85\u3061",preparing:"\u30C7\u30FC\u30BF\u3092\u6E96\u5099\u4E2D",exporting_files:"\
\u30D5\u30A1\u30A4\u30EB\u3092\u66F8\u304D\u51FA\u3057\u4E2D",finalizing:"\u6700\u7D42\u51E6\u7406\u4E2D",
ready:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u6E96\u5099\u5B8C\u4E86",downloading:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u4E2D",
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
ed","cancelled","expired"].includes(u.phase))},"renderAccountTransferProgress"),B=o(u=>{fe&&(fe.disabled=
!!u);const f=get("account-import-btn");f&&(f.disabled=!!u);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!u)},"setAccountTransferControls"),K=o((u={})=>{const f=get("account-export-ready"),g=get(
"account-export-ready-text"),b=get("account-export-expiry"),_=get("account-export-download-btn"),S=!!(u.
available&&u.download_url);if(f&&f.classList.toggle("hidden",!S),!S){_&&_.removeAttribute("href");return}
const T=Math.max(0,Number(u.size_bytes)||0),A=T>=1024*1024*1024?`${(T/(1024*1024*1024)).toFixed(2)} \
GB`:`${(T/(1024*1024)).toFixed(1)} MB`;if(g){const I=Number(u.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
u.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${I}`}
if(b){const I=u.expires_at?new Date(u.expires_at):null;b.textContent=I&&!Number.isNaN(I.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${I.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
_&&(_.href=u.download_url)},"renderAccountExportAvailability"),ce=o(async u=>{for(;P===u&&!u.stopped;){
try{const f=await apiFetch(`/api/account/transfer/${u.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await f.json().catch(()=>({}));if(f.ok&&(g.state!=="pending"&&ee(g),["ready","completed","\
failed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(f=>setTimeout(f,
700))}return null},"pollAccountTransfer"),de=o((u,f,g=!0)=>{f&&(ee(f),K(f),g&&f.state==="ready"?showToast(
f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(f.unreadable_count)>0?"warning":"success",Number(f.unreadable_count)>0):g&&f.state==="failed"&&
showToast(f.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),re(u))},"handleFinishedAccountExport"),te=o(async()=>{try{const u=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),f=await u.json().catch(()=>({}));
if(!u.ok)return;if(K(f),f.state==="ready"){ee(f);return}if(["failed","cancelled","expired"].includes(
f.state)){ee(f);return}if(!["queued","running","cancelling"].includes(f.state)||!f.job_id||P&&P.id===
f.job_id||P)return;const g={id:f.job_id,type:"export",stopped:!1,restored:!0};P=g,B(!0),ee(f);const b=await ce(
g);b&&de(g,b,!0)}catch{}},"refreshLatestAccountExport"),re=o(u=>{P===u&&(P=null),u.stopped=!0,B(!1)},
"finishAccountTransfer"),Q=get("account-transfer-cancel-btn");Q&&(Q.onclick=async()=>{const u=P;if(!(!u||
u.stopped)){u.cancelRequested=!0,Q.disabled=!0,ee({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${u.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
u.controller&&u.controller.abort(),ee({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
u.type==="export"&&K({available:!1}),re(u),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const fe=get("account-export-btn");fe&&(fe.onclick=async()=>{if(P)return;const u={id:q(),
type:"export",stopped:!1};P=u,B(!0),K({available:!1}),ee({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const f=await apiFetch("/api/account/export",
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{job_id:u.id}),keepalive:!0})),g=await f.json().catch(()=>({}));if(f.status===409&&g.error==="export\
_in_progress"&&g.job_id)u.id=g.job_id;else if(!f.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\
\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
ee({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await ce(u);!u.cancelRequested&&b&&de(u,b,!0)}catch(f){const g=f&&f.message?f.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\
\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";ee({progress:0,phase:"fail\
ed",message:g}),showToast(g,"error",!0),re(u)}});const ue=get("account-export-download-btn");ue&&ue.
addEventListener("click",async u=>{const f=ue.getAttribute("href");if(!(!f||f==="#")){u.preventDefault();
try{const g=await apiFetch("/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),
b=await g.json().catch(()=>({}));g.ok&&b.available&&b.download_url?(ue.href=b.download_url,window.location.
assign(b.download_url)):(K(b),ee(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),te())}catch{window.location.assign(f)}}}),B(!1),te();const Ge=get("import-files-grid"),
Oe=get("import-files-info"),et=get("import-files-summary"),ge=o(u=>{const f=Math.max(0,Number(u)||0);
return f>=1024*1024*1024?`${(f/(1024*1024*1024)).toFixed(2)} GB`:f>=1024*1024?`${(f/(1024*1024)).toFixed(
1)} MB`:f>=1024?`${Math.round(f/1024)} KB`:`${f} B`},"importFormatBytes");let pe=null;const Se=o(()=>{
if(!pe)return;const u=pe.files,f=pe.selection;let g=0;u.forEach(S=>{f.has(S.archive_path)&&(g+=Number(
S.size_bytes)||0)});const b=Number(pe.available_bytes)||0,_=g>b;et&&(et.textContent=`\u9078\u629E\u4E2D: ${ge(
g)} / \u5229\u7528\u53EF\u80FD: ${ge(b)}${_?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,et.classList.
toggle("text-red-300",_)),Oe&&(Oe.textContent=`${u.length} files`)},"updateImportFileSelectionUi"),De=o(
()=>{if(!Ge||!pe)return;Ge.innerHTML="";const u=pe.files;if(!u.length){Ge.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
Se();return}u.forEach(f=>{const g=document.createElement("label"),b=pe.selection.has(f.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(f.display_name)}">${escapeHtml(
f.display_name)}</div><div class="text-[10px] text-gray-500">${ge(f.size_bytes)}</div></div>`;const _=g.
querySelector(".import-file-check");_.addEventListener("change",()=>{_.checked?pe.selection.add(f.archive_path):
pe.selection.delete(f.archive_path),g.classList.toggle("border-blue-500",_.checked),g.classList.toggle(
"border-gray-600",!_.checked),Se()}),Ge.appendChild(g)}),Se()},"renderImportFileItems"),he=o(u=>new Promise(
f=>{if(pe={files:u.files||[],selection:new Set((u.files||[]).map(g=>g.archive_path)),available_bytes:u.
available_bytes,resolve:f},De(),!get("import-files-modal")){f(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),dt=o(u=>{if(hideModal("import-files-modal"),pe){const f=pe.resolve;
pe=null,f(u)}},"closeImportFileSelection"),Ne=get("import-files-close");Ne&&(Ne.onclick=()=>dt(null));
const gt=get("import-files-cancel");gt&&(gt.onclick=()=>dt(null));const ze=get("import-files-confirm");
ze&&(ze.onclick=()=>{if(!pe)return;const u=Array.from(pe.selection);dt(u.length?u.join(","):"__none_\
_")});const wt=get("import-files-select-all");wt&&(wt.onclick=()=>{pe&&(pe.files.forEach(u=>pe.selection.
add(u.archive_path)),De())});const pt=get("import-files-none");pt&&(pt.onclick=()=>{pe&&(pe.selection.
clear(),De())});const Ct={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},rn=o(u=>{
if(u===!0)return"ON";if(u===!1)return"OFF";if(u==null||u==="")return"\u672A\u8A2D\u5B9A";const f=String(
u);return f.length>60?f.slice(0,60)+"\u2026":f},"formatAccountSettingValue");let Dt=null;const $t=o(
u=>{if(Dt){const f=Dt;Dt=null,hideModal("settings-confirmation-modal"),f(u)}},"resolveSettingsImport\
Confirmation"),Ht=o(u=>new Promise(f=>{if(!get("settings-confirmation-modal")){f(!0);return}Dt=f;const b=Array.
isArray(u&&u.settings_changes)?u.settings_changes:[],_=get("settings-confirmation-list");_&&(b.length?
_.innerHTML=b.map(T=>{const A=Ct[T.field]||T.field,I=rn(T.current),O=rn(T.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
I)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(O)}</d\
iv>
                            </div>`}).join(""):_.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),It=get("settings-confir\
mation-modal");It&&It.addEventListener("click",u=>{u.target===It&&$t(!1)});const Yt=get("settings-co\
nfirmation-close");Yt&&(Yt.onclick=()=>$t(!1));const qt=get("settings-confirmation-cancel");qt&&(qt.
onclick=()=>$t(!1));const kt=get("settings-confirmation-confirm");kt&&(kt.onclick=()=>$t(!0));const Pt=get(
"account-import-btn"),Qt=get("account-import-inplace"),Gt=get("account-import-inplace-warning");if(Qt&&
Gt){const u=o(()=>Gt.classList.toggle("hidden",!Qt.checked),"syncInplaceWarn");Qt.addEventListener("\
change",u),u()}Pt&&(Pt.onclick=async()=>{const u=get("account-import-file"),f=u&&u.files?u.files[0]:
null,g=get("account-import-categories"),b=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(le=>le.value):[],_=get("account-import-inplace"),S=!!(_&&_.checked),T=get("account-impo\
rt-settings-bypass"),A=!!(T&&T.checked);let I=!1;if(!f){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(le=>(le.
closest("label")&&le.closest("label").textContent||le.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${O.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const F={id:q(),type:"import",stopped:!1,controller:new AbortController};
P=F,B(!0),ee({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const Z=get("account-import-result");let ae=Promise.resolve(null);try{const V=Math.max(1,Math.ceil(f.
size/10485760)),we=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:f.size}),signal:F.controller.
signal})),W=await we.json().catch(()=>({}));if(!we.ok)throw new Error(W.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
F.uploadId=W.upload_id;const be=W.chunk_size||10485760;let Y=0,ke=0;const Ce=o(async()=>{for(;;){const Ee=ke++;
if(Ee>=V)return;const Le=f.slice(Ee*be,Math.min(f.size,(Ee+1)*be)),Ve=new FormData;Ve.append("chunk",
Le,f.name),Ve.append("index",String(Ee));const tt=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Ve,signal:F.controller.signal})),
ct=await tt.json().catch(()=>({}));if(!tt.ok)throw new Error(ct.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
Y++,ee({progress:Math.min(35,Math.round(Y/V*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${Y}\
/${V}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let se=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),se=!0);try{await Promise.
all([Ce(),Ce(),Ce()]);const Ee=await apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:F.controller.signal})),Le=await Ee.json().
catch(()=>({}));if(!Ee.ok)throw new Error(Le.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
ee({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
se&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let xe="",Be=!1,je=0;const lt=o(
async()=>{let Ee=!1;const Le=o(()=>{Ee||(Ee=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Ve=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),tt=await Ve.
json().catch(()=>null);if(!Ve.ok||!tt){Le();return}cacheUserSettings(tt);const ct=get("settings-moda\
l");if(ct&&ct.classList.contains("modal-open"))try{Ie(tt)}catch{}tt.theme_color&&applyThemeColor(tt.
theme_color,!0),Object.prototype.hasOwnProperty.call(tt,"minimal_prompt_mode")&&tt.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(tt,"compact_prompt_mode")&&setCompactPromptMode(
!!tt.compact_prompt_mode)}catch{}Le()},"refreshSettingsFormAfterImport"),Qe=o(Ee=>{const Le=Ee&&Ee.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";Z&&(Z.textContent=`\u5B8C\u4E86: ${Le}`,
Z.classList.remove("hidden","text-red-300"),Z.classList.add("text-emerald-300")),ee({progress:100,phase:"\
completed",message:Le}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&lt()},"finishImportSucce\
ss"),yt=o(async()=>{try{const Le=await(await apiFetch(`/api/account/transfer/${F.id}`,manualSpinnerRequestOptions(
{cache:"no-store"}))).json().catch(()=>null);return Le&&Le.state?Le:null}catch{return null}},"fetchI\
mportStatus"),Ft=o(async()=>{const Ee=await yt();if(!Ee)return{status:"unknown"};if(Ee.state==="comp\
leted")return Qe(Ee),{status:"done"};if(["failed","cancelled","expired"].includes(Ee.state))throw new Error(
Ee.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(Ee.state===
"needs_selection"&&Array.isArray(Ee.files)){const Le=await he({files:Ee.files,available_bytes:Ee.available_bytes});
return Le===null?(ee({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(xe=Le,{status:"reselect"})}if(Ee.state===
"needs_settings_confirmation"&&Array.isArray(Ee.settings_changes))return await Ht({settings_changes:Ee.
settings_changes})?(I=!0,{status:"reselect"}):(ee({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\
\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
F.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(Ee.state==="running"){const Le=await Promise.race([ae.catch(()=>null),new Promise(Ve=>setTimeout(
()=>Ve(null),6e4))]);if(Le&&Le.state==="completed")return Qe(Le),{status:"done"};throw Le&&["failed",
"cancelled","expired"].includes(Le.state)?new Error(Le.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Be;){F.stopped=!0,await ae.catch(()=>null),
F.stopped=!1,ae=ce(F);let Ee;try{Ee=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:F.uploadId,
categories:b.join(","),job_id:F.id,selected_files:xe,restore_inplace:S,confirm_settings:I||A}),signal:F.
controller.signal}))}catch(Re){if(F.cancelRequested||Re&&Re.name==="AbortError")throw Re;const ft=await Ft();
if(ft.status==="done"){Be=!0;break}if(ft.status==="cancelled")return;if(ft.status==="reselect")continue;
if(je<2){je++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let Le=null;try{Le=await Ee.json()}catch{Le=null}if(Le===null){const Re=await Ft();if(Re.status==="d\
one"){Be=!0;break}if(Re.status==="cancelled")return;if(Re.status==="reselect")continue;if(Ee.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if(je<2){je++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!Ee.ok&&Le.error==="storage_limit_files"&&Le.files){const Re=await he(Le);if(Re===null){ee({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
F.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}xe=Re;continue}if(Le&&Le.status==="settings_confirmation"&&
Array.isArray(Le.settings_changes)){if(!await Ht(Le)){ee({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\
\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),F.uploadId&&
apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}I=!0;continue}if(!Ee.ok)throw new Error(Le.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\
\u5931\u6557\u3057\u307E\u3057\u305F");const Ve=Le.imported||{},tt=[`\u8A2D\u5B9A ${Ve.settings||0}\u4EF6`,
`API\u8A8D\u8A3C ${Ve.api_credentials||0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Ve.chats||0}\u4EF6`,`Sk\
ill ${Ve.skills||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Ve.files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Ve.
feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Ve.diagnostics||0}\u4EF6`].join(" / "),ct=Le.duplicates||
{},Te={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\
\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"},Fe=[];for(const Re of Object.
keys(Te)){const ft=Number(ct[Re])||0;ft>0&&Fe.push(`${Te[Re]} ${ft}\u4EF6`)}const nt=Fe.length?`\uFF08\u91CD\u8907\u3092\
\u30B9\u30AD\u30C3\u30D7: ${Fe.join("\u3001")}\uFF09`:"";Z&&(Z.textContent=`\u5B8C\u4E86: ${tt}${nt}`,
Z.classList.remove("hidden","text-red-300"),Z.classList.add("text-emerald-300")),ee({progress:100,phase:"\
completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast(
"\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("skills")&&loadSkills(),b.includes("files")&&
loadStorageUsage(),(b.includes("settings")||b.includes("api_credentials"))&&lt(),Be=!0}}catch(le){if(F.
uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(F.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),F.cancelRequested||le&&le.name==="AbortError")return;const V=le&&
le.message?le.message:"",we=V==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
V||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";ee({progress:0,phase:"\
failed",message:we}),Z&&(Z.textContent=we,Z.classList.remove("hidden","text-emerald-300"),Z.classList.
add("text-red-300")),showToast(we,"error",!0)}finally{F.stopped=!0,await ae.catch(()=>null),re(F)}});
const xt=get("account-dedupe-btn"),Ot=get("account-dedupe-result"),Ut=o((u,f=!1)=>{Ot&&(Ot.textContent=
u,Ot.classList.remove("hidden"),Ot.classList.toggle("text-red-300",!!f),Ot.classList.toggle("text-em\
erald-300",!f))},"showDedupeResult");xt&&(xt.onclick=async()=>{const u=o(async()=>{const f=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await f.json().catch(()=>null);if(!f.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){Ut("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],_={chats:"\u30C1\u30E3\u30C3\u30C8",skills:"Skill",files:"\u30D5\u30A1\u30A4\u30EB",
feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};
for(const F of["chats","skills","files","feedback","diagnostics"]){const Z=Number(g.duplicates&&g.duplicates[F])||
0;Z>0&&b.push(`${_[F]} ${Z}\u4EF6`)}const S=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const T=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await T.json().catch(()=>null);if(!T.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const I=[];for(const F of["chats","skills","files","feedback","diagnostics"]){const Z=Number(A.removed&&
A.removed[F])||0;Z>0&&I.push(`${_[F]} ${Z}\u4EF6`)}const O=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";Ut(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${I.join("\u3001")||"0\u4EF6"}${O}`),loadThreads(),loadSkills(),loadStorageUsage()},"r\
un");if(!xt.disabled){xt.disabled=!0,Ut("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await u()}catch(f){Ut(f&&f.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{xt.disabled=!1}}});const ln=get("site-cache-usage-refresh");ln&&(ln.onclick=()=>loadSiteCacheUsage());
const Zt=get("clear-site-cache-btn");Zt&&(Zt.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(Zt)});
const _t=get("enc-scan-result"),cn=o(async(u=null)=>{_t&&(_t.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let f="/api/encryption_scan";u&&(f+=`?thread_id=${encodeURIComponent(u)}`);try{const g=await apiFetch(
f,{cache:"no-store"}),b=await g.json();if(!g.ok){_t&&(_t.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const _=b.total||0,S=b.encrypted||0,T=b.unencrypted||0;let A=`Total: ${_} / Encrypted: ${S} /\
 Plain: ${T}`;if(b.samples&&b.samples.length){const I=b.samples.slice(0,8).map(O=>{const F=O.timestamp?
new Date(O.timestamp).toLocaleString():"";return`#${O.id} (${O.role||""}) ${F}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${I}</div>`}_t&&(_t.innerHTML=A)}catch{_t&&(_t.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),X=get("enc-scan-all");X&&(X.onclick=()=>cn(null));
const _e=get("enc-scan-thread");_e&&(_e.onclick=()=>currentThreadId?cn(currentThreadId):showToast("\u30B9\
\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ae=get("admin-enc-list");
let Ue=null,ot=!1;const ht=o(u=>!u||!u.length?null:u.some(f=>!!f.is_encrypted),"computeThreadEncrypt\
edFromMessages"),St=o(()=>{Ue=ht(allMessages)},"refreshCurrentThreadEncStateFromMessages"),Ze=o(async(u,f,{
confirmPrompt:g=!0,reloadCurrent:b=!0}={})=>{if(!u)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const _=f?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;ot=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
u)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:f})}),
T=await S.json().catch(()=>({}));return S.ok?(showToast(`${_}\u3057\u307E\u3057\u305F\uFF08${T.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),Ue=!!f,b&&currentThreadId&&String(currentThreadId)===String(
u)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ae&&await Ye(),!0):
(showToast(T.error||`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${_}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{ot=!1}},"setAdminThreadEncry\
ption"),Lt=o(u=>{if(!Ae)return;const f=u.threads||[];if(!f.length){Ae.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ae.innerHTML=f.map(g=>{const b=g.encrypted_count>0?"enc":"plain",_=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",T=g.
updated_at?new Date(g.updated_at).toLocaleString():"",A=escapeHtml(String(g.thread_id)),I=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${I?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${T} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${_}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Ye=o(async()=>{if(Ae){Ae.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const u=await apiFetch("/api/admin/threads",{cache:"no-store"}),f=await u.json().catch(()=>({}));
if(!u.ok){Ae.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(f.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(Lt(f),currentThreadId&&Array.isArray(f.threads)){const g=f.threads.find(b=>String(
b.thread_id)===String(currentThreadId));g&&(Ue=!!g.encrypted)}}catch{Ae.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Ye()),window.__loadAdminEncThreads=
Ye,window.__refreshAdminThreadEncState=St,window.__setAdminThreadEncryption=Ze;const st=get("encrypt\
ion-status-admin-toggle");st&&st.addEventListener("click",u=>{u.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ae&&(Ae.onclick=async u=>{const f=u.target.closest("\
.admin-enc-open");if(f){u.preventDefault();const A=f.getAttribute("data-id");if(!A)return;typeof Ke==
"function"?Ke():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=u.target.closest(".admin-enc-toggle");if(!g||ot)return;const b=g.getAttribute("data-i\
d"),_=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${_?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const T=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await Ze(b,_,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=T,await Ye()}}),get("file-input").onchange=u=>{const f=Array.from(u.target.
files||[]);u.target.value="",f.length&&handleFiles(f)},get("photo-input")&&(get("photo-input").onchange=
u=>{const f=Array.from(u.target.files||[]);u.target.value="",f.length&&handleFiles(f)});const Tt=o(u=>{
const f=get("ban-appeal-list");if(f){if(!u||!u.length){f.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
f.innerHTML=u.map(g=>{const b=g.status||"new",_=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=g.created_at?new Date(g.created_at).toLocaleString():"",T=g.replied_at?new Date(g.replied_at).toLocaleString():
"",A=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${_}</div>
                                <div class="text-[10px] text-gray-500">${escapeHtml(S)}</div>
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
                    `}).join("")}},"renderBanAppeals"),dn=o(async(u=!1)=>{if(!isAdminUser)return;const f=get(
"ban-appeal-count");if(f)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const _=(await g.json()).unread_count||0;f.textContent=String(_),u&&_>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${_}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),Nt=o(async()=>{if(!isAdminUser)return;const u=get("ban-appeal-list");if(u){u.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const f=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!f.ok)return;const g=await f.json();Tt(g.items||[]),
await dn(!1)}catch{}}},"loadBanAppeals"),un=o(async(u=null)=>{if(!isAdminUser)return;const f=u?{ids:u}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(f)})).ok&&await Nt()}catch{}},"markBanAppealsRead"),en=o(async u=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(u)})).ok&&await Nt()}catch{}},"updateBanAppealStatus"),He=o(()=>{
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
                `,u.appendChild(f)},"ensureTemporaryChatSettingsCard"),We=o(()=>{const u=get("set-st\
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
                `,f.appendChild(g);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const _=get(
"set-llm-transcribe-prompt");_&&(_.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),Mt=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
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
lls"}];window.buildAutoSystemPromptRows=(u,f=!1)=>{const g=f?"w-full h-14 bg-gray-950 border border-\
gray-700 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-700 roun\
ded p-2 text-xs text-gray-200";return Mt.map(b=>{const _=b.barLocked===!0,S=b.barLabel||"MCP",T=_?`<\
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
                        <textarea id="${u}-auto-sys-${b.key}-text" class="${g}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${T}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(u,f={})=>{Mt.forEach(g=>{const b=f&&
typeof f=="object"?f[g.key]||{}:{},_=get(`${u}-auto-sys-${g.key}-enabled`),S=get(`${u}-auto-sys-${g.
key}-text`);_&&(g.barLocked===!0?_.disabled=!0:_.checked=b.enabled!==!1),S&&(S.value=b.text||"",S.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Rt=o((u,f=null)=>{if(f){const g=get(f);g&&(g.checked=!0)}Mt.forEach(g=>{const b=get(`${u}-auto\
-sys-${g.key}-enabled`),_=get(`${u}-auto-sys-${g.key}-text`);if(b&&(g.barLocked!==!0?b.checked=!0:b.
disabled=!0),_){const S=_.placeholder||"";_.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),mn=o(u=>{const f={};return Mt.forEach(g=>{const b=get(`${u}\
-auto-sys-${g.key}-enabled`),_=get(`${u}-auto-sys-${g.key}-text`);f[g.key]={enabled:g.barLocked===!0?
!0:b?b.checked:!0,text:_?_.value:""}}),f},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
mn,window.ensureAutoSystemPromptSettingsCard=()=>{const u=get("set-global-sys-prompt-enabled"),f=u?u.
closest(".space-y-4"):null;if(!f||get("auto-sys-prompt-settings"))return;const g=document.createElement(
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
                `,f.appendChild(g)},He(),We(),R();const zt=o(()=>{const u=get("set-default-model");if(!u)
return;const f=u.value;u.innerHTML="",MODELS.forEach(b=>{const _=document.createElement("optgroup");
_.label=b.category,(b.items||[]).forEach(S=>{const T=document.createElement("option");T.value=S.id,T.
textContent=S.name,_.appendChild(T)}),u.appendChild(_)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||f||"gemini-3.6-flash";g&&Array.from(u.options).some(b=>b.value===g)&&(u.value=g),syncSettingsModelPickerLabels()},
"populateDefaultModelOptions"),Me=o(()=>{const u=get("set-default-vision-model");if(!u)return;const f=u.
value;u.innerHTML="",MODELS.forEach(b=>{const _=(b.items||[]).filter(T=>{const A=(T.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(_.length===0)return;const S=document.createElement("optgroup");S.label=b.category,_.forEach(
T=>{const A=document.createElement("option");A.value=T.id,A.textContent=T.name+" \u2605",S.appendChild(
A)}),u.appendChild(S)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||f||
"gemini-3-flash-preview";g&&Array.from(u.options).some(b=>b.value===g)&&(u.value=g),syncSettingsModelPickerLabels()},
"populateDefaultVisionModelOptions"),Ie=o(u=>{if(!u)return;cacheUserSettings(u);const f=get("app-glo\
bal-sys-prompt-preview");f&&(f.value=u.global_system_prompt_effective||"");const g=get("app-global-s\
ys-prompt-preview-status");g&&(u.global_system_prompt_enabled===!1?g.textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
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
u.gemini_key||""),get("set-deepseek")&&(get("set-deepseek").value=u.deepseek_key||""),get("set-zai")&&
(get("set-zai").value=u.zai_key||""),get("set-kimi")&&(get("set-kimi").value=u.kimi_key||""),get("se\
t-mistral")&&(get("set-mistral").value=u.mistral_key||""),get("set-ideogram")&&(get("set-ideogram").
value=u.ideogram_key||""),get("set-anthropic")&&(get("set-anthropic").value=u.anthropic_key||""),get(
"set-gemini-backend")&&(get("set-gemini-backend").value=normalizeGeminiBackend(u.gemini_backend||"ge\
mini_api")),get("set-gemini-vertex-project")&&(get("set-gemini-vertex-project").value=u.gemini_vertex_project||
""),get("set-gemini-vertex-location")&&(get("set-gemini-vertex-location").value=u.gemini_vertex_location||
"global"),ensureGeminiVertexCredentialsField(),get("set-gemini-vertex-credentials-json")&&(get("set-\
gemini-vertex-credentials-json").value=u.gemini_vertex_credentials_json||""),syncGeminiBackendUi(),get(
"set-admin-api-key-mode")&&(get("set-admin-api-key-mode").value=normalizeAdminApiKeyMode(u.admin_api_key_mode||
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
"gemini-3-flash-preview"),syncSettingsModelPickerLabels(),applyTemporaryChatTimeoutSeconds(u.temp_chat_timeout_seconds),
get("set-default-search")&&(get("set-default-search").checked=!!u.default_enable_search),get("set-de\
fault-url-context")&&(get("set-default-url-context").checked=!!u.default_enable_url_context),get("se\
t-default-maps")&&(get("set-default-maps").checked=!!u.default_enable_maps),get("set-default-python")&&
(get("set-default-python").checked=!!u.default_enable_python),get("set-default-file-creation")&&(get(
"set-default-file-creation").checked=!!u.default_enable_file_creation),get("set-default-thinking")&&
(get("set-default-thinking").checked=!!u.default_enable_thinking),get("set-default-sys-prompt")&&(get(
"set-default-sys-prompt").checked=!!u.default_enable_system_prompt),get("set-default-mcp")&&(get("se\
t-default-mcp").checked=u.default_enable_mcp!==!1),get("set-default-thinking-level")&&(get("set-defa\
ult-thinking-level").value=u.default_thinking_level||"high"),get("set-default-thinking-budget")&&(get(
"set-default-thinking-budget").value=u.default_thinking_budget||4096),get("set-default-reasoning-eff\
ort")&&(get("set-default-reasoning-effort").value=u.default_reasoning_effort||"medium"),get("set-def\
ault-safety")&&(get("set-default-safety").value=u.default_safety_setting||"default"),get("set-e2ee").
checked=u.enable_e2ee,get("set-bot-detect")&&(get("set-bot-detect").checked=u.bot_detection_enabled!==
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
[];if(J(I),T){T.checked=!!u.passkey_only_login;const le=I.length>0||!!u.has_webauthn;T.disabled=!le,
le||(T.checked=!1),A&&(le?A.classList.add("hidden"):A.classList.remove("hidden"))}const O=get("mig-s\
tatus-box"),F=get("mig-progress-text"),Z=get("mig-progress-bar");if((u.migration_status||"idle")==="\
processing"){O.classList.remove("hidden");const le=(u.migration_progress||"").split("/");if(le.length===
2){const V=parseInt(le[0]||"0",10),we=parseInt(le[1]||"0",10);F&&(F.innerText=`${V} / ${we}`),Z&&we>
0&&(Z.style.width=`${Math.min(100,Math.floor(V/we*100))}%`)}}else O.classList.add("hidden"),Z&&(Z.style.
width="0%"),F&&(F.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const u=await ensureUserSettingsSnapshot();u&&Ie(u);const f=get(
"search-box"),g=f?f.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),zt(),Me(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(_=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+_+"ms"),snapshotSidebarHistory("settings-open-later-"+_+"ms")},_)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),te(),We(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),dn(!0),Nt(),u||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),hn(),ie(),me();try{loadMcpServers()}catch{}};const Ke=o((u=!1)=>{settingsModelPickerTarget&&
window.closeModelModal(!0),snapshotSidebarHistory("settings-close-before"),hideModal("settings-modal"),
revealPersistentSidebarLists(),snapshotSidebarHistory("settings-close-after"),setTimeout(()=>snapshotSidebarHistory(
"settings-close-later-300ms"),300),!u&&location.pathname==="/settings"&&history.back()},"closeSettin\
gsModal"),ut=o(()=>{const u=get("set-theme-color"),f=get("set-theme-color-text"),g=get("theme-reset-\
btn"),b=document.querySelectorAll("#theme-presets .theme-swatch"),_=o((S,T=!0)=>{const A=normalizeHex(
S);A&&(applyThemeColor(A,T),syncThemeInputs(A))},"applyFromValue");u&&u.addEventListener("input",()=>_(
u.value,!0)),f&&(f.addEventListener("change",()=>{const S=normalizeHex(f.value);if(!S){syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||THEME_DEFAULT);return}_(S,!0)}),f.addEventListener("keydown",
S=>{S.key==="Enter"&&(S.preventDefault(),f.blur())})),g&&(g.onclick=()=>_(THEME_DEFAULT,!0)),b.forEach(
S=>{S.addEventListener("click",()=>_(S.getAttribute("data-color"),!0))})},"bindThemeControls"),mt=o(
()=>{const u=get("reset-global-sys-prompt");u&&(u.onclick=()=>{get("sys-prompt-text")&&(get("sys-pro\
mpt-text").value=""),get("set-global-sys-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=
!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const f=get("reset-thread-sys-prompt");f&&(f.onclick=()=>{get("thread-global-sys-prompt")&&
(get("thread-global-sys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-glo\
bal-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{Rt("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const b=get("reset-thread-auto-sys-prompt-defaults");b&&(b.onclick=()=>{Rt("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Ke();const At=get("settings-header-close");At&&(At.onclick=()=>Ke());
const bt=get("settings-search");bt&&(bt.addEventListener("input",filterSettings),bt.addEventListener(
"keydown",u=>{if(u.key==="Enter"){const f=get("tab-"+activeSettingsTab);if(!f)return;const g=f.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Yn=get(
"settings-search-clear");Yn&&Yn.addEventListener("click",()=>{bt&&(bt.value="",filterSettings(),bt.focus())}),
ut(),mt(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Nn=get("set-gemini-local-python-dialog");Nn&&(Nn.onchange=()=>setGeminiLocalPyDialogEnabled(Nn.
checked));const Qn=get("set-gemini-backend");Qn&&(Qn.onchange=()=>syncGeminiBackendUi());const Zn=get(
"set-admin-api-key-mode");Zn&&(Zn.onchange=()=>syncAdminApiKeyModeUi());const Rn=get("set-temp-chat-\
timeout-seconds");Rn&&(Rn.onchange=()=>{applyTemporaryChatTimeoutSeconds(Rn.value)});const ei=get("s\
lash-command-cancel-btn");ei&&(ei.onclick=()=>{hidePendingSlashCommandIndicator();const u=get("promp\
t-input");u&&u.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{var A;if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const u=get("set-username"),f=get("set-password"),g=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:mn(
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
-default-2fa-method").value:"totp",new_username:u?u.value:null,new_password:f?f.value:null},_=get("s\
et-e2ee")?get("set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
_!==S&&(b.enable_e2ee=_),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
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
tent-Type":"application/json"},body:JSON.stringify(b)});if(T.ok){let I="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const ae=await T.json();ae&&ae.message&&(I=ae.message)}catch{}cacheUserSettings(Object.assign({},
userSettingsSnapshot||{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),
window.applySavedUserSystemPromptSettings({system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,
apply_global_system_prompt:b.apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:b.auto_system_prompt_notices_config}),Ke();const O=currentUsername,
F=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const Z=useSwCache;
useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),Z!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(I,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==O?setTimeout(()=>location.
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
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"single"})}),b=await g.json(),_=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")_&&(_.textContent=`${f} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const u=get("bot-unban-username"),f=u?u.value.trim():"";if(!f){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:f,mode:"linked"})}),b=await g.json(),_=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")_&&(_.textContent=`${f} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
_.classList.remove("hidden")),u&&(u.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const u=get(
"bot-speed-test-btn"),f=get("bot-speed-test-result");u&&(u.disabled=!0),u&&u.classList.add("opacity-\
60","cursor-not-allowed"),f&&(f.classList.remove("hidden"),f.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(Y=>{f&&(f.textContent=Y)},"setBox"),b=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),_=o((Y,ke)=>!Y||!ke||ke<=0?0:Y*8/(ke/1e3)/1e3/1e3,"toMbps"),S=o(Y=>Number.isFinite(
Y)?`${Y.toFixed(0)} ms`:"-","fmtMs"),T=o(Y=>Number.isFinite(Y)?`${Y.toFixed(Y>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=o(async(Y,ke)=>{const Ce=await Y.json().catch(()=>({}));return Ce&&Ce.error?Ce.error:ke},
"parseErr"),I=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let Y=0;Y<4;Y++){const ke=performance.now(),Ce=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),se=performance.now();if(!Ce.ok)throw new Error(await A(
Ce,"ping_failed"));await Ce.json().catch(()=>({})),I.push(se-ke)}const O=I.reduce((Y,ke)=>Y+ke,0)/Math.
max(1,I.length),F=Math.min(...I),Z=o(async Y=>{const ke=performance.now(),Ce=await apiFetch(`/api/sp\
eedtest/download?bytes=${Y}&_=${b()}`,{cache:"no-store"});if(!Ce.ok)throw new Error(await A(Ce,"down\
load_failed"));const se=await Ce.arrayBuffer(),xe=performance.now();return{bytes:se.byteLength||Y,ms:xe-
ke,mbps:_(se.byteLength||Y,xe-ke)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${S(O)}
\u6E2C\u5B9A\u4E2D... download`);const ae=[];for(const Y of[2*1024*1024,8*1024*1024])ae.push(await Z(
Y)),g(`\u6E2C\u5B9A\u4E2D... ping ${S(O)}
download ${T(Math.max(...ae.map(ke=>ke.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const le=Math.max(...ae.map(Y=>Y.mbps)),V=o(async Y=>{const ke=new Uint8Array(
Y),Ce=performance.now(),se=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ke,cache:"no-store"}),xe=performance.now();if(!se.ok)
throw new Error(await A(se,"upload_failed"));const Be=await se.json().catch(()=>({})),je=Number(Be.bytes_received||
Y)||Y;return{bytes:je,ms:xe-Ce,mbps:_(je,xe-Ce),serverMs:Number(Be.server_elapsed_ms||0)||0}},"runUp\
load"),we=[];for(const Y of[1*1024*1024,4*1024*1024])we.push(await V(Y));const W=Math.max(...we.map(
Y=>Y.mbps)),be=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(O)} / ${S(F)}`,`Download (best): ${T(le)}`,`Upload (best): ${T(W)}`,`Download r\
uns: ${ae.map(Y=>`${Math.round(Y.bytes/1024/1024)}MB=${T(Y.mbps)}`).join(", ")}`,`Upload runs: ${we.
map(Y=>`${Math.round(Y.bytes/1024/1024)}MB=${T(Y.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(be.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){f&&(f.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{u&&(u.disabled=!1,u.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>Nt()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>un()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async u=>{const f=u.target.closest("button");if(!f)return;const g=f.getAttribute("data-id");
if(f.classList.contains("ban-appeal-mark")){g&&await un([Number(g)]);return}if(f.classList.contains(
"ban-appeal-status")){const b=f.getAttribute("data-status");g&&b&&await en({id:Number(g),status:b});
return}if(f.classList.contains("ban-appeal-reply-send")){const b=f.closest("[data-appeal-id]"),_=b?b.
querySelector(".ban-appeal-reply"):null,S=_?_.value:"";g&&await en({id:Number(g),admin_reply:S});return}
if(f.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await en({id:Number(
g),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
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
u.target.value))});const Bn=get("marker-color-picker");Bn&&(Bn.oninput=u=>setMarkerColor(u.target.value),
Bn.onchange=u=>setMarkerColor(u.target.value));const jn=get("marker-opacity");jn&&(jn.oninput=u=>setMarkerOpacity(
u.target.value),jn.onchange=u=>setMarkerOpacity(u.target.value));const Ln=get("marker-opacity-number");
Ln&&(Ln.onchange=u=>setMarkerOpacity(u.target.value),Ln.onblur=u=>setMarkerOpacity(u.target.value),Ln.
onkeydown=u=>{u.key==="Enter"&&(setMarkerOpacity(u.target.value),u.target.blur())}),document.querySelectorAll(
"#marker-toolbar .marker-color-chip[data-marker-color]").forEach(u=>{u.onclick=()=>setMarkerColor(u.
getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-view-reset").onclick=()=>resetMarkerTransform()),
get("marker-crop-reset")&&(get("marker-crop-reset").onclick=()=>clearCropRect()),get("marker-undo")&&
(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("marker-clear")&&(get("marker-clear").onclick=
()=>clearMarkerCanvas()),get("marker-save")&&(get("marker-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),
initMarkerCanvas(),initCropCanvas(),window.addEventListener("resize",()=>{const u=get("marker-modal");
!u||u.classList.contains("hidden")||(applyMarkerTransform(),renderCropOverlay())});const Ci=o(()=>{const u=get(
"upload-modal");return!!(u&&!u.classList.contains("hidden"))},"isUploadModalOpen"),tn=get("drop-over\
lay");let pn=0;const Li=o(()=>{Ci()||tn&&(tn.classList.remove("hidden"),tn.classList.add("flex"))},"\
showDropOverlay"),fn=o(()=>{pn=0,tn&&(tn.classList.add("hidden"),tn.classList.remove("flex"))},"hide\
DropOverlay");window.hideDropOverlay=fn;const Et=get("upload-dropzone");Et&&(Et.addEventListener("dr\
agover",u=>{u.preventDefault(),Et.classList.add("dragover")}),Et.addEventListener("dragleave",()=>{Et.
classList.remove("dragover")}),Et.addEventListener("drop",u=>{u.preventDefault(),u.stopPropagation(),
Et.classList.remove("dragover"),fn();const f=u.dataTransfer?u.dataTransfer.files:null;f&&f.length&&handleFiles(
f)})),window.addEventListener("dragenter",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.
types.includes("Files")||(pn+=1,Li())}),window.addEventListener("dragover",u=>{!u.dataTransfer||!u.dataTransfer.
types||!u.dataTransfer.types.includes("Files")||u.preventDefault()}),window.addEventListener("dragle\
ave",u=>{!u.dataTransfer||!u.dataTransfer.types||!u.dataTransfer.types.includes("Files")||(pn=Math.max(
0,pn-1),(pn===0||!u.relatedTarget||u.clientY<=0||u.clientX<=0||u.clientX>=window.innerWidth||u.clientY>=
window.innerHeight)&&fn())}),window.addEventListener("dragend",()=>{fn()}),window.addEventListener("\
drop",u=>{fn(),!(!u.dataTransfer||!u.dataTransfer.files||u.dataTransfer.files.length===0)&&(u.preventDefault(),
!(Et&&Et.contains(u.target))&&handleFiles(u.dataTransfer.files))});const ti=get("bot-admin-modal"),Ti=o(
u=>{const f=get("bot-admin-list");if(f){if(f.innerHTML="",!u||!u.length){f.innerHTML='<div class="te\
xt-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}u.forEach((g,b)=>{const _=!!g.is_bot_banned,S=g.bot_detection_enabled!==!1,T=Number(g.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(b,12)*.02}\
s`,A.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(g.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${_?"BAN\u4E2D":"\u6B63\u5E38"}${T?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil(T/60)}\u5206\uFF09`:""} ${g.bot_ban_reason?
" / "+escapeHtml(g.bot_ban_reason):""}</div>
                        </div>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-enabled="${S?"1":"0"}">${S?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${_?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(g.username)}" data-bann\
ed="${_?"1":"0"}">${_?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${_?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(g.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        ${g.is_admin?"":`<button class="bot-delete-account bg-red-800 hover:bg-red-7\
00 text-white px-2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(g.username)}\
">\u524A\u9664</button>`}
                    `,f.appendChild(A)})}},"renderBotUsers"),gn=o(async(u="")=>{const f=get("bot-adm\
in-list");f&&(f.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
u)}`),b=await g.json();g.ok&&b&&b.users?Ti(b.users):(f&&(f.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{f&&(f.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Fn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||ti))return;const f=get(
"settings-modal");f&&(f.classList.contains("modal-open")||f.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await gn(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Fn;const Dn=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const u=get("settings-modal");u&&(u.classList.contains(
"modal-open")||u.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=Dn,window.closeBotLogModal=(u=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!u&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{Dn()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(u=!1)=>{(get("bot-admin-modal")||ti)&&hideModal("bot-admin-modal"),!u&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Fn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await gn(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await gn("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async u=>{u.key==="Enter"&&await gn(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async u=>{const f=u.target.closest("button");if(!f)return;const g=f.getAttribute(
"data-user");if(!g)return;let b;if(f.classList.contains("bot-toggle-detect")){const _=f.getAttribute(
"data-enabled")!=="1";b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:g,action:"toggle_detection",enabled:_})})}else if(f.classList.
contains("bot-toggle-ban"))if(f.getAttribute("data-banned")==="1")b=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"ban",reason:"Admin ban"})})}else if(f.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"unban_linked"})})}else if(f.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"delete_account"})})}if(b){if(b.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(b.ok){if(f.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),g===currentUsername)){location.href="/";return}}else{
let _={};try{_=await b.json()}catch{}showToast(_.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
_.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await gn(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Bt={"/settings":{id:"settings-mod\
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
"/skill":{id:"skill-modal",open:o(()=>{editingSkillUuid=null,get("skill-modal-title").textContent="C\
reate New Skill",showModal("skill-modal")},"open")},"/compression":{id:"compression-modal",open:o(()=>window.
openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:o(()=>Fn(),"open")},"/admin\
-bot-logs":{id:"bot-log-modal",open:o(()=>Dn(),"open")}};Bt["/gem"]=Bt["/skill"];const ni=o((u,f=!1)=>{
switch(u){case"settings-modal":Ke(f);break;case"upload-modal":closeUploadModal(f);break;case"image-s\
plit-modal":closeImageSplit();break;case"image-split-compression-modal":hideModal(u);break;case"came\
ra-capture-modal":closeCameraCaptureModal(f?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(f);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(f);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(f);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(f);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
f);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(f);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(f);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(f);break;case"token-detail-modal":closeTokenDetail(f);break;case"encryption-s\
tatus-modal":closeEncryptionModal(f);break;case"python-exec-modal":closePythonExecDetail(f);break;case"\
skill-modal":window.closeSkillModal&&window.closeSkillModal(f);break;case"compression-modal":window.
closeCompressionModal&&window.closeCompressionModal(f);break;case"bot-admin-modal":window.closeBotAdminModal&&
window.closeBotAdminModal(f);break;case"bot-log-modal":window.closeBotLogModal&&window.closeBotLogModal(
f);break;case"mcp-decision-modal":typeof submitMcpDecision=="function"?submitMcpDecision("deny"):hideModal(
u);break;case"api-key-required-modal":{const b=get("api-key-modal-cancel-btn");b&&typeof b.onclick==
"function"?b.click():hideModal(u);break}case"lyria-studio-modal":window.closeLyriaStudio?window.closeLyriaStudio():
hideModal(u);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(
u);break;case"version-update-modal":const g=localStorage.getItem("app_version")||"";g&&localStorage.
setItem("version_notified",g),hideModal(u);break;default:hideModal(u);break}},"closeModalById");window.
addEventListener("popstate",u=>{var S,T;const f=(S=u.state)==null?void 0:S.settingsModelPicker;if(location.
pathname==="/settings"&&settingsModelPickerTargets.includes(f)){const A=o(()=>openSettingsModelPicker(
f,!0),"restorePicker");(T=get("settings-modal"))!=null&&T.classList.contains("modal-open")?A():window.
openSettingsModal().then(()=>{var I;location.pathname==="/settings"&&((I=history.state)==null?void 0:
I.settingsModelPicker)===f&&A()});return}settingsModelPickerTarget&&window.closeModelModal(!0);let g=!1;
Object.values(Bt).forEach(A=>{const I=get(A.id);I&&I.classList.contains("modal-open")&&location.pathname!==
Object.keys(Bt).find(O=>Bt[O].id===A.id)&&(ni(A.id,!0),g=!0)});const b=location.pathname.match(/^\/c\/(.+)$/);
if(b){const A=decodeURIComponent(b[1]);String(currentThreadId)!==String(A)&&loadMessages(A,{skipHistory:!0})}else
location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});const _=Bt[location.pathname];
if(_){const A=get(_.id);A&&!A.classList.contains("modal-open")&&_.open()}}),document.addEventListener(
"keydown",u=>{u.key==="Escape"&&settingsModelPickerTarget&&(u.preventDefault(),window.closeModelModal())});
const ii=location.pathname;Bt[ii]&&(history.replaceState({},"","/"),setTimeout(()=>Bt[ii].open(),500)),
get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const u=get("easy-login-mi\
ns"),f=u?parseInt(u.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${f}\u5206\u9593\u6709\u52B9\
\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const b=await(await apiFetch("/api/easy_login",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:f})})).json();b&&b.temp_password?
(get("easy-login-code").textContent=b.temp_password,get("easy-login-exp").textContent=b.expires_at||
"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const f=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(f&&f.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const Mi=4,Ai=5*1024*1024;function si(u,f,g){let b=[];const _=o(()=>{g&&(g.innerHTML="",
b.forEach((S,T)=>{const A=document.createElement("div");A.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const I=document.createElement("img");I.src=S.url,I.alt=S.file.name,
I.className="w-full h-full object-cover";const O=document.createElement("button");O.type="button",O.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",O.textContent=
"\xD7",O.onclick=()=>{URL.revokeObjectURL(S.url),b.splice(T,1),_()},A.appendChild(I),A.appendChild(O),
g.appendChild(A)}))},"render");return u&&f&&(u.onclick=()=>f.click(),f.onchange=S=>{for(const T of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(T.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(T.
size>Ai){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(b.
length>=Mi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}b.push({file:T,url:URL.createObjectURL(T)})}S.target.value="",_()}),{files:o(()=>b.
map(S=>S.file),"files"),clear:o(()=>{b.forEach(S=>URL.revokeObjectURL(S.url)),b=[],_()},"clear")}}o(
si,"createFbImagePicker");async function ai(u,f){try{const g=new FormData;return f.forEach(b=>g.append(
"images",b,b.name)),(await apiFetch(u,{method:"POST",body:g})).ok}catch{return!1}}o(ai,"uploadFbImag\
es");const oi=si(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const u=get("fb-title").value.trim(),f=get("fb-message").value.trim();if(!f){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:u,message:f,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=get("fb-attach-logs"),_=b&&b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():
null;_&&(g.client_logs=_);const S=get("fb-attach-chat"),T=!!(S&&S.checked);if(T&&!currentThreadId){showToast(
"\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}T&&(g.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const A=await apiFetch("/api/feedback",{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),I=A.ok?await A.clone().json().
catch(()=>({})):{},O=oi.files(),F={count:O.length,saved:!0};A.ok&&F.count&&(F.saved=await ai(`/api/f\
eedback/${I.public_id}/images`,O)),(window.ActivityLog?await window.ActivityLog.reportFeedback(A,_,T,
F):A.ok)&&(get("fb-title").value="",get("fb-message").value="",oi.clear(),S&&(S.checked=!1),b&&(b.checked=
!0),hn())};async function ri(u){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${u}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),hn()}o(ri,"deleteFeedback");const Ei="\u4FDD\u5B58\u671F\u9593\u5185\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let Hn=[];function li(u,f){const g=u.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(b=>{const _=b.role==="admin"?f&&b.author?`\u7BA1\u7406\u8005 (${b.
author})`:"\u7BA1\u7406\u8005":f?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=f?[b.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${b.
log_file}`,b.chat_file&&`\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${b.chat_file}`,b.image_dir&&
`\u6DFB\u4ED8\u753B\u50CF\uFF08${b.image_count}\u679A\uFF09: ${b.image_dir}`]:[b.image_count&&`\u6DFB\u4ED8\u753B\u50CF:\
 ${b.image_count}\u679A`];return`<div class="text-[11px]"><div class="text-gray-400">${escapeHtml(_)}\
 / ${escapeHtml(b.created_at||"")}</div><div class="text-sm whitespace-pre-wrap ${b.role==="admin"?"\
text-green-300":"text-gray-200"}">${escapeHtml(b.message)}</div>${S.filter(Boolean).map(T=>`<div cla\
ss="${f?"text-amber-300":"text-gray-400"}">${escapeHtml(T)}</div>`).join("")}</div>`}).join("")}</di\
v>`:""}o(li,"feedbackRepliesHtml");function ci(){return`<div class="fb-reply-form mt-2 space-y-2">
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
                </div>`}o(ci,"feedbackReplyFormHtml");function di(u,f){const g=f.querySelector(".fb-\
reply-form");if(!g)return;const b=g.querySelector(".fb-reply-logs"),_=g.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.canAttach());b.disabled=!S,_.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const T=si(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));Hn.push(T);const A=g.querySelector(".fb-reply-send");A.onclick=async()=>{const I=g.
querySelector(".fb-reply-message").value.trim();if(!I){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const O={message:I,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},F=b.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;F&&(O.client_logs=F);const Z=g.
querySelector(".fb-reply-chat").checked;if(Z&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}Z&&(O.chat_copy=window.ActivityLog?window.ActivityLog.chatCopyPayload(currentThreadId):
{client:"web",thread_id:String(currentThreadId)});const ae=T.files();A.disabled=!0;try{const le=await apiFetch(
`/api/feedback/${u.public_id}/replies`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(O)}),V=await le.json().catch(()=>({}));if(!le.ok){showToast(V.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const we=!ae.length||await ai(`/api/feedback/${u.public_id}/replies/${V.reply_id}\
/images`,ae),W=[];if(F&&V.logs_saved===!1&&W.push("\u30ED\u30B0"),we||W.push("\u753B\u50CF"),Z&&V.chat_copy_saved!==
!0&&W.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),W.length)showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${W.
join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\uFF09`,"error",!0);else{
const be=["\u8FD4\u4FE1"];Z&&be.push("\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC"),F&&be.push(
`\u4FDD\u5B58\u671F\u9593\u5185\u306E\u30ED\u30B0\uFF08${typeof V.logs_count=="number"?V.logs_count:
F.entries.length}\u4EF6\uFF09`),ae.length&&be.push(`\u753B\u50CF\uFF08${ae.length}\u679A\uFF09`),showToast(
be.join(be.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","success")}hn()}finally{
A.disabled=!1}}}o(di,"bindFeedbackReply");function ui(){const u=get("fb-admin-status-filter").value;
let f=0;get("fb-admin-list").querySelectorAll("[data-feedback-status]").forEach(g=>{const b=u==="all"||
(u==="unresolved"?["new","in_review","replied"].includes(g.dataset.feedbackStatus):g.dataset.feedbackStatus===
u);g.hidden=!b,b&&f++}),get("fb-admin-empty").classList.toggle("hidden",f!==0)}o(ui,"filterAdminFeed\
back"),get("fb-admin-status-filter").onchange=ui;async function hn(){Hn.forEach(S=>S.clear()),Hn=[];
const f=await(await apiFetch("/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML="",(f.items||
[]).filter(S=>!f.is_admin||S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(f.is_admin)return;
const T=document.createElement("div");T.className="p-2 rounded border border-gray-700 bg-gray-800/50",
T.innerHTML=`<div class="text-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(
S.public_id||"")}</span> / ${S.created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||
"No Title")}</div><div class="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class=\
"text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(S.status==="resolved_unreleased"?"resolved\uFF08non\
-released\uFF09":S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.
image_count}\u679A</div>`:""}${li(S,!1)}${ci()}<div class="mt-2"><button type="button" class="fb-del\
ete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,
T.querySelector(".fb-delete").onclick=()=>ri(S.public_id),di(S,T),g.appendChild(T)});const b=get("fb\
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
                            ${li(S,!0)}
                            ${ci()}
                        `,T.querySelector(".fb-status").value=S.status||"new",T.querySelector(".fb-s\
ave").onclick=async()=>{const A=T.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:A})}),
hn()},T.querySelector(".fb-delete").onclick=()=>ri(S.public_id),di(S,T),_.appendChild(T)}),ui()):b.classList.
add("hidden")}if(o(hn,"loadFeedback"),window.setupTOTP=async()=>{const f=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=f.qr_image,get("totp-secret-disp").innerText=
f.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const u=get(
"totp-verify-code").value;if(!u)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:u})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const u=get("register-webauthn-btn"),f=get("webauthn-nam\
e"),g=f?String(f.value||"").trim():"";try{u&&(u.disabled=!0);const b=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),_=await b.json();if(!b.ok){showToast(_.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const T=await(await ensureWebAuthnJson()).create({publicKey:_}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},T,{name:g}))}),I=await A.json().catch(()=>({}));A.ok?(f&&(f.value=""),showToast(
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
return}}if(skillSuggestionsVisible){const g=f.value.trim();if(u.key==="ArrowDown"){u.preventDefault(),
skillSelectedIndex=skillSelectedIndex+1,showSkillSuggestions(g.substring(1));return}if(u.key==="Arro\
wUp"){u.preventDefault(),skillSelectedIndex=Math.max(skillSelectedIndex-1,0),showSkillSuggestions(g.
substring(1));return}if(u.key==="Enter"){u.preventDefault();const b=g.substring(1).toLowerCase(),_=loadedSkills.
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
click()}),get("sts-cancel-btn")&&(get("sts-cancel-btn").onclick=()=>{isStsModel()&&$i()}),get("promp\
t-input")&&get("prompt-input").addEventListener("paste",async u=>{const f=(u.clipboardData||window.clipboardData).
items,g=[];for(let b=0;b<f.length;b++)if(f[b].kind==="file"){const _=f[b].getAsFile();_&&g.push(_)}g.
length>0&&(u.preventDefault(),await handleFiles(g,{openModal:!1}))}),get("rich-paste-btn")&&(get("ri\
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
u.addEventListener("paste",async f=>{const g=f.clipboardData||window.clipboardData;if(g){f.preventDefault();
try{await ingestRichPasteClipboardData(g)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),u.addEventListener("input",()=>{u.value=""})}get("chat-container").addEventListener(
"click",u=>{const f=u.target.closest("img.chat-image"),g=f?f.dataset.viewerSrc||f.currentSrc||f.src:
"";f&&g&&(u.preventDefault(),openImageViewer(g))});const bn=document.querySelector(".viewer-content");
bn&&(bn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),bn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),bn.addEventListener("touchend",onViewerTouchEnd),bn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",u=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(u.target.id==="image-viewer"||u.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",u=>{u.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",u=>{u.key==="Escape"&&closeImageViewer()});
let rt,qe=null,Tn=[],qn=!1,Mn=null,nn=null,Wt=null,yn=null,Gn=0,An=!1,vn=null,sn=null,jt=null,wn=null,
En=null,Vt=null,$n=null,In=null;function mi(){const u=get("mic-waveform");if(!u)return[];if(Array.isArray(
$n)&&$n.length)return $n;u.innerHTML="";const f=[];for(let g=0;g<24;g++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",f.push(b),u.appendChild(b)}return $n=f,f}o(mi,"ensureMicWaveformBars");function Jt(u,f="hidden"){
const g=get("mic-recording-indicator"),b=get("mic-recording-text");if(g){if(In&&(clearTimeout(In),In=
null),f==="hidden"){g.classList.add("hidden");return}b&&u&&(b.innerText=u),g.classList.remove("hidde\
n"),f==="recording"?g.style.color="rgb(252 165 165)":f==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(Jt,"setMicRecordingIndicator");function pi(){mi().forEach(f=>{f.
style.height="2px",f.style.opacity="0.35"})}o(pi,"resetMicWaveformBars");function Kt(){if(En&&(cancelAnimationFrame(
En),En=null),wn){try{wn.disconnect()}catch{}wn=null}if(sn){try{sn.close()}catch{}sn=null}jt=null,Vt=
null,pi()}o(Kt,"stopMicWaveform");function fi(u){Kt();const f=mi();if(!f.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{sn=new g,jt=sn.createAnalyser(),jt.fftSize=
256,jt.smoothingTimeConstant=0,wn=sn.createMediaStreamSource(u),wn.connect(jt),Vt=new Uint8Array(jt.
frequencyBinCount)}catch{Kt();return}const b=o(()=>{if(!jt||!Vt)return;jt.getByteFrequencyData(Vt);const _=Math.
max(1,Math.floor(Vt.length/f.length));for(let S=0;S<f.length;S++){const A=(Vt[Math.min(Vt.length-1,S*
_)]||0)/255,I=Math.max(2,Math.round(2+A*10));f[S].style.height=`${I}px`,f[S].style.opacity=`${.35+A*
.65}`}En=requestAnimationFrame(b)},"render");b()}o(fi,"startMicWaveform");function Pn(){if(Mn&&(clearInterval(
Mn),Mn=null),yn){try{yn.disconnect()}catch{}yn=null}if(nn){try{nn.close()}catch{}nn=null}Wt=null}o(Pn,
"stopSilenceMonitor");function gi(u){if(!isStsModel()||!stsOpt("sts-auto-send"))return;Pn();const f=window.
AudioContext||window.webkitAudioContext;if(!f)return;nn=new f,Wt=nn.createAnalyser(),Wt.fftSize=2048,
yn=nn.createMediaStreamSource(u),yn.connect(Wt);const g=new Uint8Array(Wt.fftSize),b=getStsSilenceMs(),
_=.02;Gn=0,An=!1,Mn=setInterval(()=>{if(!Wt)return;Wt.getByteTimeDomainData(g);let S=0;for(let A=0;A<
g.length;A++){const I=(g[A]-128)/128;S+=I*I}if(Math.sqrt(S/g.length)>_){An||(An=!0),Gn=Date.now();return}
An&&Date.now()-Gn>b&&rt&&rt.state==="recording"&&rt.stop()},200)}o(gi,"startSilenceMonitor");const Wn=class Wn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(f,g,b,_={},S=null){if(this.
model=b,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(f)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const I=!!(_&&
_.transcriptionConfig),O={setup:{model:`models/${b}`,generationConfig:{responseModalities:I?["TEXT"]:
["AUDIO"]},inputAudioTranscription:I?_.transcriptionConfig||{}:{}}};I||(O.setup.outputAudioTranscription=
{}),_.speechConfig&&(O.setup.generationConfig.speechConfig=_.speechConfig),_.thinkingConfig&&(O.setup.
generationConfig.thinkingConfig=_.thinkingConfig),_.translationConfig&&(O.setup.generationConfig.translationConfig=
_.translationConfig),console.log("Sending setup:",JSON.stringify(O)),this.ws.send(JSON.stringify(O))},
this.ws.onmessage=I=>this._handleMessage(I),this.ws.onerror=I=>{console.error("Gemini Live WebSocket\
 error:",I),this.onError&&this.onError(I)},this.ws.onclose=I=>{console.log("Gemini Live WebSocket cl\
osed:",I.code,I.reason),this.closedEvent=I,this.onClose&&this.onClose(I)},S&&S.stream&&S.ctx&&S.source){
this.stream=S.stream,this.audioContext=S.ctx;var T=S.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(On());const I=hi(this.stream,16e3);this.audioContext=I.ctx;var T=I.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=I=>{I.data.
size>0&&this.userAudioChunks.push(I.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
I=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const O=bi(I.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!O||!O.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(O))),mimeType:"audio/\
pcm;rate=16000"}}}))},T.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(f){
let g=null;try{g=JSON.parse(typeof f.data=="string"?f.data:new TextDecoder().decode(f.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const b=g.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(_=>{if(_.text&&(_.thought?(console.log(
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
!0))}g.error&&this.onError&&this.onError(g.error),this.onMessage&&this.onMessage(g)}_appendAssistantText(f){
f&&(this.assistantTurnBreak&&this.assistantText&&!this.assistantText.endsWith(`
`)&&(this.assistantText+=`
`),this.assistantTurnBreak=!1,this.assistantText+=f)}stop(){const f=this.backupRecorder;if(this.backupRecorder=
null,this._stopped=new Promise(b=>{let _=!1;const S=o(()=>{_||(_=!0,b())},"finish");if(!f||f.state===
"inactive"){S();return}f.addEventListener("stop",S,{once:!0});try{f.stop()}catch{S()}setTimeout(S,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=o(
()=>{if(this.stream&&this.stream.getTracks().forEach(b=>{try{b.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const f=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(f),b=new Blob(
this.userAudioChunks),_=await this._blobToBase64(b);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:_}}_blobToBase64(f){
return new Promise(g=>{const b=new FileReader;b.onloadend=()=>g(b.result.split(",")[1]),b.readAsDataURL(
f)})}};o(Wn,"GeminiLiveClient");let Un=Wn;const Vn=class Vn{constructor(f=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:f}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(f){if(!this.ctx)return;const g=atob(f),b=new Uint8Array(g.length);for(let O=0;O<
g.length;O++)b[O]=g.charCodeAt(O);const _=new Int16Array(b.buffer),S=new Float32Array(_.length);for(let O=0;O<
_.length;O++)S[O]=_[O]/32768;const T=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);T.getChannelData(
0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();
A.buffer=T,A.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const I=Math.max(this.ctx.currentTime,this.nextStartTime);A.start(I),this.
nextStartTime=I+T.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Vn,"RealTimeAudioPl\
ayer");let kn=Vn;const Jn=class Jn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
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
error",!0);return}const g={model:f,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||f==="gpt-realtime-translate")&&get("sts-targ\
et-lang")?get("sts-target-lang").value:""};isXaiLiveTranscribeModel()&&get("sts-custom-vocab")&&(g.custom_vocabulary=
get("sts-custom-vocab").value.split(/[,、\n]/)),setStsStatus("\u63A5\u7D9A\u4E2D...",!0);try{const b=await apiFetch(
"/api/realtime/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
g)}),_=await b.json().catch(()=>({}));if(!b.ok)throw new Error(_.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=_.session_id,this.rateIn=_.rate_in||this.rateIn,this.rateOut=_.rate_out||this.rateOut,
this.active=!0,this.capturing=!0,this.saved=!1,this.userTranscript="",this.assistantTranscript="",this.
assistantThought="",this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.rtPlayer=
null,this.audioQueue=[],this.audioFlush=null}catch(b){setStsStatus("\u63A5\u7D9A\u30A8\u30E9\u30FC",
!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F: "+
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
value:_}=await this.reader.read();if(b)break;g+=f.decode(_,{stream:!0});let S;for(;(S=g.indexOf(`

`))>=0;){const T=g.slice(0,S);g=g.slice(S+2);for(const A of T.split(`
`)){if(!A.startsWith("data: "))continue;let I=null;try{I=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent(I)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(f){
if(f)switch(f.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new kn(this.rateOut||24e3),an=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
mediaDevices.getUserMedia(On()).then(f=>{this.stream=f;const g=this.rateIn||24e3,b=hi(f,g);this.audioCtx=
b.ctx;const _=b.source,S=this.audioCtx.sampleRate,T=4096;this.processor=this.audioCtx.createScriptProcessor(
T,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const I=A.inputBuffer.
getChannelData(0),O=bi(I,S,g);!O||!O.byteLength||this._sendAudio(O)},_.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(f){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(f)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const f=this.audioQueue.splice(0,this.audioQueue.length),
g=f.reduce((I,O)=>I+O.byteLength,0),b=new Uint8Array(g);let _=0;f.forEach(I=>{b.set(I,_),_+=I.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),T={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(T):T;try{await fetch(S,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(f=>f.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}Pn(),Kt()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const f=Date.now(),g=this.responseDoneCount;let b=this.lastAudioAt;for(;Date.now()-f<2e4&&!(this.responseDoneCount>
g||(this.lastAudioAt>b&&(b=this.lastAudioAt),!this.speechActive&&Date.now()-f>2e3&&Date.now()-b>2500));)
await new Promise(_=>setTimeout(_,250));await this._save()}async _save(){if(!this.saved){this.saved=
!0,this.saving=!0;try{const f=await apiFetch("/api/realtime/save",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify({session_id:this.sessionId,thread_id:currentThreadId})}),
g=await f.json().catch(()=>({}));if(!f.ok)throw new Error(g.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(g.thread_id&&!currentThreadId&&(currentThreadId=String(g.thread_id)),this.streamError)setStsStatus(
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
"mic-btn");f&&(f.classList.remove("bg-red-600","animate-pulse"),f.classList.add("bg-gray-700"))}};o(
Jn,"RealtimeVoiceSession");let zn=Jn;function hi(u,f){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let b=null;try{return b=new g({sampleRate:f}),{ctx:b,
source:b.createMediaStreamSource(u)}}catch{if(b)try{b.close()}catch{}return b=new g,{ctx:b,source:b.
createMediaStreamSource(u)}}}o(hi,"openMicAudioSource");function bi(u,f,g){let b=u;if(f!==g&&f>0&&g>
0){const S=f/g,T=Math.floor(b.length/S),A=new Float32Array(T);for(let I=0;I<T;I++)A[I]=b[Math.min(Math.
floor(I*S),b.length-1)];b=A}const _=new Int16Array(b.length);for(let S=0;S<b.length;S++){const T=Math.
max(-1,Math.min(1,b[S]));_[S]=T<0?T*32768:T*32767}return _.buffer}o(bi,"pcm16FromFloat32");const xn=new zn,
yi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,_=null,S=!1,T=null,A=!1,I=0,O=0,F=null,Z="idle",ae=!1,le=null;const V=o(H=>document.getElementById(
H),"$"),we=o(H=>{const ne=Object.assign({},H||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(ne):(ne.progressSpinner=
!1,ne)},"noSpinner");function W(H,ne){Z=ne;const $e=V("lyria-status-text"),ve=V("lyria-status-dot");
$e&&($e.textContent=H),ve&&(ve.className="w-2 h-2 rounded-full inline-block "+(g[ne]||g.idle)),Ce(),
se()}o(W,"setStatus");function be(){const H=O?Math.floor((Date.now()-O)/1e3):0,ne=String(Math.floor(
H/60)).padStart(2,"0"),$e=String(H%60).padStart(2,"0");return`${ne}:${$e}`}o(be,"formatElapsed");function Y(){
O||(O=Date.now());const H=V("lyria-elapsed");H&&(H.textContent=be()),F||(F=window.setInterval(()=>{const ne=V(
"lyria-elapsed");ne&&(ne.textContent=be())},1e3))}o(Y,"startElapsedTimer");function ke(){F&&(window.
clearInterval(F),F=null)}o(ke,"stopElapsedTimer");function Ce(){const H=V("lyria-play-btn"),ne=V("ly\
ria-pause-btn"),$e=V("lyria-stop-btn"),ve=V("lyria-reset-btn"),Pe=!!b,Je=Z==="streaming"||Z==="conne\
cting";if(H){H.disabled=ae;const it=H.querySelector("i");it&&(it.className="fas fa-play")}ne&&(ne.disabled=
ae||!Je),$e&&($e.disabled=ae||!Pe||!Je),ve&&(ve.disabled=ae||!Pe||!Je)}o(Ce,"updateTransportButtons");
function se(){const H=V("lyria-save-btn");if(!H)return;const ne=!!b&&Z!=="idle"&&Z!=="connecting"&&Z!==
"error";H.classList.toggle("hidden",!ne)}o(se,"updateSaveButton");function xe(H,ne){const $e=V("lyri\
a-prompt-rows");if(!$e)return;const ve=document.createElement("div");ve.className="flex items-center\
 gap-2",ve.innerHTML=`
                        <input type="text" value="${escapeHtml(H||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof ne=="nu\
mber"?ne:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof ne=="number"?ne:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Pe=ve.querySelector('input[type="range"]'),Je=ve.querySelector(".lyria-w\
eight-label");Pe&&Je&&Pe.addEventListener("input",()=>{Je.textContent=parseFloat(Pe.value).toFixed(1)});
const it=ve.querySelector(".lyria-prompt-remove");it&&it.addEventListener("click",()=>{$e.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||ve.remove()}),ve.classList.add("lyria-prompt-row-wrap"),$e.appendChild(
ve)}o(xe,"addPromptRow");function Be(){const H=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),ne=[];return H.forEach($e=>{const ve=$e.querySelector('input[type="text"]'),Pe=$e.
querySelector('input[type="range"]'),Je=(ve?ve.value:"").trim();Je&&ne.push({text:Je,weight:parseFloat(
Pe?Pe.value:1)||1})}),ne}o(Be,"collectPrompts");function je(){const H={},ne=o(Ni=>{const Xn=V(Ni);return Xn&&
Xn.value!==""?parseFloat(Xn.value):void 0},"num"),$e=ne("lyria-bpm");$e!==void 0&&(H.bpm=Math.round(
$e));const ve=ne("lyria-guidance");ve!==void 0&&(H.guidance=ve);const Pe=ne("lyria-density");Pe!==void 0&&
(H.density=Pe);const Je=ne("lyria-brightness");Je!==void 0&&(H.brightness=Je);const it=ne("lyria-tem\
perature");it!==void 0&&(H.temperature=it);const Xe=V("lyria-scale");Xe&&Xe.value&&(H.scale=Xe.value);
const at=V("lyria-mode");at&&at.value&&(H.music_generation_mode=at.value);const vt=V("lyria-mute-bas\
s"),Xt=V("lyria-mute-drums"),Si=V("lyria-only-bass-drums");return vt&&(H.mute_bass=vt.checked),Xt&&(H.
mute_drums=Xt.checked),Si&&(H.only_bass_and_drums=Si.checked),H}o(je,"collectConfig");function lt(){
[["lyria-bpm","lyria-bpm-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-d\
ensity-label"],["lyria-brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature\
-label"]].forEach(([ne,$e])=>{const ve=V(ne),Pe=V($e);!ve||!Pe||ve.addEventListener("input",()=>{const Je=parseFloat(
ve.value);Pe.textContent=ne==="lyria-bpm"?String(Math.round(Je)):Je.toFixed(1)})})}o(lt,"bindRangeLa\
bels");function Qe(){if(T){try{T.close()}catch{}T=null}A=!1,I=0}o(Qe,"resetPlayback");function yt(){
if(S=!1,_&&typeof _.abort=="function")try{_.abort()}catch{}_=null}o(yt,"closeStream");async function Ft(){
yt(),_=new AbortController,S=!0;try{const H=await fetch(`/api/skillini/music/stream?session_id=${encodeURIComponent(
b)}`,we({method:"GET",signal:_.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!H.
ok){const Pe=await H.json().catch(()=>({}));throw new Error(Pe.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const ne=H.body.getReader(),$e=new TextDecoder;let ve="";for(;S;){const{done:Pe,value:Je}=await ne.read();
if(Pe)break;ve+=$e.decode(Je,{stream:!0});const it=ve.split(`

`);ve=it.pop();for(const Xe of it){const at=Xe.split(`
`).find(Xt=>Xt.startsWith("data: "));if(!at)continue;const vt=at.slice(6);try{const Xt=JSON.parse(vt);
Ee(Xt)}catch{}}}}catch(H){if(H&&H.name==="AbortError")return;S&&(W("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&b&&Ft()},1200))}finally{S=!1}}o(Ft,"openStream");function Ee(H){
if(H&&H.snapshot){const ne=H.status;if(ne==="error"){W("\u30A8\u30E9\u30FC","error"),ke();return}if(ne===
"closed"||ne==="stopped"){W("\u7D42\u4E86","closed"),ke();return}W(ne==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",ne==="paused"?"paused":"connecting");return}if(H&&H.audio){W("\u518D\u751F\u4E2D...",
"streaming"),Y(),Le(H.audio);return}if(H&&H.error){W("\u30A8\u30E9\u30FC: "+H.error,"error"),ke();return}
if(H&&H.final){W("\u7D42\u4E86","closed"),ke(),Ce();return}}o(Ee,"handleStreamMessage");function Le(H){
if(!H)return;if(!T){const Xe=window.AudioContext||window.webkitAudioContext;if(!Xe)return;T=new Xe({
sampleRate:48e3}),A=!1,I=0}let ne;try{const Xe=atob(H);ne=new Uint8Array(Xe.length);for(let at=0;at<
Xe.length;at++)ne[at]=Xe.charCodeAt(at)}catch{return}const $e=new Int16Array(ne.buffer),ve=Math.floor(
$e.length/2);if(ve<1)return;const Pe=T.createBuffer(2,ve,48e3);for(let Xe=0;Xe<2;Xe++){const at=Pe.getChannelData(
Xe);for(let vt=0;vt<ve;vt++)at[vt]=$e[vt*2+Xe]/32768}T.state==="suspended"&&T.resume();const Je=T.createBufferSource();
Je.buffer=Pe,Je.connect(T.destination),A||(I=T.currentTime+.08,A=!0);const it=Math.max(T.currentTime,
I);Je.start(it),I=it+Pe.duration}o(Le,"playChunk");async function Ve(H,ne){const $e=await fetch("/ap\
i/skillini/music/command",we({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:H},ne||{}))})),ve=await $e.json().catch(()=>({}));if(!$e.ok)throw new Error(
ve.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return ve}
o(Ve,"apiCommand");async function tt(){if(ae)return;const H=Be();if(!H.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ae=!0,Ce(),W("\u63A5\u7D9A\u4E2D...","connecting");try{
const ne=await fetch("/api/skillini/music/start",we({method:"POST",headers:{"Content-Type":"applicat\
ion/json"},body:JSON.stringify({weighted_prompts:H,config:je()})})),$e=await ne.json().catch(()=>({}));
if(!ne.ok)throw new Error($e.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=$e.session_id,le=je(),W("\u63A5\u7D9A\u4E2D...","connecting"),Ft()}catch(ne){W("\u30A8\u30E9\u30FC: "+
ne.message,"error"),showToast("Lyria RealTime: "+ne.message,"error",!0)}finally{ae=!1,Ce()}}o(tt,"st\
artSession");async function ct(H){if(b){ae=!0,Ce();try{await Ve("control",{action:H}),H==="PLAY"?W("\
\u518D\u751F\u4E2D...","streaming"):H==="PAUSE"?W("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):H==="ST\
OP"?W("\u505C\u6B62\u4E2D","stopped"):H==="RESET_CONTEXT"&&W("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(ne){showToast("Lyria RealTime: "+ne.message,"error",!0),W("\u30A8\u30E9\u30FC: "+
ne.message,"error")}finally{ae=!1,Ce()}}}o(ct,"control");async function Te(){if(!b)return;const H=Be();
if(!H.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ae=!0;try{await Ve("prompts",{weighted_prompts:H}),W("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
Z==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(ne){showToast("Lyria RealTime: "+ne.message,"error",!0)}finally{ae=!1,Ce()}}o(Te,"a\
pplyPrompts");async function Fe(){if(!b)return;const H=je(),ne=le||{},$e=H.bpm!==void 0&&H.bpm!==ne.
bpm,ve=H.scale!==void 0&&H.scale!==ne.scale,Pe=$e||ve;ae=!0;try{await Ve("config",{config:H,reset_context:Pe}),
le=H,W(Pe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",Z==="paused"?"paused":"streaming"),showToast(
Pe?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Je){showToast("Lyria RealT\
ime: "+Je.message,"error",!0)}finally{ae=!1,Ce()}}o(Fe,"applyConfig");async function nt(){if(b){ae=!0,
W("\u4FDD\u5B58\u4E2D...","connecting"),Ce();try{const H=await fetch("/api/skillini/music/save",we({
method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),ne=await H.json().catch(()=>({}));if(!H.ok)throw new Error(ne.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ke(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),ne.thread_id&&(currentThreadId=String(ne.thread_id),history.pushState({},"","/c/"+ne.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(ne.thread_id||currentThreadId),Sn(
!0)}catch(H){W("\u30A8\u30E9\u30FC: "+H.message,"error"),showToast("Lyria RealTime: "+H.message,"err\
or",!0)}finally{ae=!1,Ce()}}}o(nt,"saveSession");async function Re(){if(yt(),b)try{await fetch("/api\
/skillini/music/cancel",we({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{session_id:b})}))}catch{}b=null,ke(),Qe(),W("\u6E96\u5099\u5B8C\u4E86","idle")}o(Re,"cancelSession");
function ft(){const H=V("lyria-prompt-rows");H&&(H.innerHTML=""),xe("",1),le=null,O=0,["lyria-bpm","\
lyria-guidance","lyria-density","lyria-brightness","lyria-temperature"].forEach(ve=>{const Pe=V(ve);
Pe&&(Pe.value=ve==="lyria-bpm"?"120":ve==="lyria-guidance"?"4":ve==="lyria-temperature"?"1.1":"0.5")});
const ne=V("lyria-scale");ne&&(ne.value="");const $e=V("lyria-mode");$e&&($e.value="QUALITY"),["lyri\
a-mute-bass","lyria-mute-drums","lyria-only-bass-drums"].forEach(ve=>{const Pe=V(ve);Pe&&(Pe.checked=
!1)}),lt()}o(ft,"resetControls");function Sn(H){yt(),b&&fetch("/api/skillini/music/cancel",we({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),
b=null,S=!1,ke(),Qe(),hideModal("lyria-studio-modal")}o(Sn,"closeAndCleanup");function on(H){if(!isLyriaRealtimeModel()){
showToast("Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const ne=V("lyria-studio-modal");if(ne&&ne.classList.contains("modal-open")&&b){
if(H&&typeof H=="string"){const ve=V("lyria-prompt-rows");ve&&(ve.innerHTML=""),xe(H,1)}return}if(b&&
Re(),ft(),H&&typeof H=="string"){const ve=V("lyria-prompt-rows");ve&&(ve.innerHTML=""),xe(H,1)}b=null,
S=!1,ke(),Qe(),W("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(on,"open");function Cn(){
const H=V("lyria-open-studio-btn");H&&H.addEventListener("click",()=>on(""));const ne=V("lyria-studi\
o-close");ne&&ne.addEventListener("click",()=>Sn(!1));const $e=V("lyria-play-btn");$e&&$e.addEventListener(
"click",()=>{if(!b){tt();return}ct("PLAY")});const ve=V("lyria-pause-btn");ve&&ve.addEventListener("\
click",()=>ct("PAUSE"));const Pe=V("lyria-stop-btn");Pe&&Pe.addEventListener("click",()=>ct("STOP"));
const Je=V("lyria-reset-btn");Je&&Je.addEventListener("click",()=>ct("RESET_CONTEXT"));const it=V("l\
yria-add-prompt-btn");it&&it.addEventListener("click",()=>xe("",1));const Xe=V("lyria-apply-prompts-\
btn");Xe&&Xe.addEventListener("click",Te);const at=V("lyria-apply-config-btn");at&&at.addEventListener(
"click",Fe);const vt=V("lyria-save-btn");vt&&vt.addEventListener("click",nt),lt(),ft(),window.openLyriaStudio=
on}o(Cn,"init");function Kn(){ae||Sn(!1)}return o(Kn,"requestClose"),{init:Cn,open:on,requestClose:Kn}})();
yi.init(),window.closeLyriaStudio=()=>yi.requestClose(),(()=>{let u=null,f=null,g=null;const b="voic\
eDockSettingsOpen",_="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=o(se=>document.getElementById(se),"$");function T(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(T,"isStudioMode");function A(){const se=get("model-select")?get("model-select").value:"",xe=S("voi\
ce-studio-title");xe&&(se==="gpt-transcribe"||se==="gpt-live-transcribe"||se==="grok-voice-transcrib\
e-2.0-file"?xe.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":se===
"gemini-3.5-live-translate-preview"?xe.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
xe.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(A,"updateTitle");function I(){const se=S("v\
oice-studio-transcript");se&&(se.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${_}</div>`);const xe=S("sts-live-transcript");xe&&(xe.innerHTML="",xe.classList.add("hidden"))}
o(I,"resetTranscript");function O(se,xe,Be){const je=se.querySelectorAll(".voice-studio-line");let lt=null;
for(let Qe=je.length-1;Qe>=0;Qe--)if(je[Qe].dataset.role===xe){lt=je[Qe];break}if(lt)lt.innerHTML=Be;else{
const Qe=se.querySelector(".voice-studio-placeholder");Qe&&Qe.remove();const yt=document.createElement(
"div");yt.className="voice-studio-line",yt.dataset.role=xe,yt.innerHTML=Be,se.appendChild(yt)}se.classList.
remove("hidden"),se.scrollTop=se.scrollHeight}o(O,"writeLine");function F(se,xe){if(!xe||!String(xe).
trim()||!T())return;const lt=`<span class="${se==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(se==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
xe)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(Qe=>O(Qe,
se,lt))}o(F,"log");function Z(se,xe=!0){const Be=S("sts-panel"),je=S("sts-settings-toggle");if(Be&&Be.
classList.toggle("settings-open",!!se),je&&je.setAttribute("aria-expanded",se?"true":"false"),xe)try{
localStorage.setItem(b,se?"1":"0")}catch{}}o(Z,"setSettingsOpen");function ae(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}o(ae,"readSettingsOpen");let le=null;function V(){const se=get("mod\
el-select")?get("model-select").value:"";se!==le&&(le=se,I()),Z(ae(),!1),A()}o(V,"syncDock");function we(){
const se=S("sts-panel"),xe=S("voice-studio-panel-host");se&&xe&&se.parentNode!==xe&&(u=se.parentNode,
f=se.nextSibling,xe.appendChild(se));const Be=S("file-preview"),je=S("voice-studio-file-host");Be&&je&&
Be.parentNode!==je&&(g=Be.parentNode,je.appendChild(Be),je.classList.remove("hidden"))}o(we,"movePan\
elIntoModal");function W(){const se=S("sts-panel");se&&u&&se.parentNode!==u&&(f&&f.parentNode===u?u.
insertBefore(se,f):u.appendChild(se));const xe=S("file-preview");xe&&g&&xe.parentNode!==g&&g.appendChild(
xe);const Be=S("voice-studio-file-host");Be&&Be.classList.add("hidden"),u=null,f=null,g=null}o(W,"mo\
vePanelBack");function be(){if(!T())return;we();const se=S("sts-panel");se&&se.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(be,"open");function Y(){window.
VoiceStudioOpen=!1,W(),hideModal("voice-studio-modal")}o(Y,"close");function ke(){window.VoiceStudioOpen&&
Y()}o(ke,"closeIfOpen");function Ce(){window.VoiceStudioOpen=!1;const se=S("voice-studio-open-btn");
se&&se.addEventListener("click",()=>be());const xe=S("voice-studio-close");xe&&xe.addEventListener("\
click",()=>Y());const Be=S("sts-settings-toggle");Be&&Be.addEventListener("click",()=>{const je=S("s\
ts-panel");Z(!(je&&je.classList.contains("settings-open")))}),window.VoiceStudio={open:be,close:Y,closeIfOpen:ke,
log:F,isStudioMode:T,syncDock:V},V()}return o(Ce,"init"),{init:Ce}})().init();let an=null;function vi(){
if(an&&(an.stop(),an=null),vn){try{vn.pause()}catch{}try{vn.src=""}catch{}vn=null}}o(vi,"stopStsPlay\
back");async function ji(u){vi();const f=new Audio;return f.src=u,f.preload="auto",f.autoplay=!0,f.playsInline=
!0,vn=f,await f.play(),new Promise(g=>{f.onended=()=>g("ended"),f.onerror=()=>g("error")})}o(ji,"pla\
yStsAudio");function $i(){if(xn.isActive()){xn._cancel();return}if(qe){qe.stop(),qe=null,vi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Kt();return}rt&&rt.state===
"recording"&&(qn=!0,rt.stop())}o($i,"cancelRecording");function On(){if(isStsModel())return{audio:!0};
const f=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return f.echoCancellation&&(g.echoCancellation=!1),f.
noiseSuppression&&(g.noiseSuppression=!1),f.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(On,"\
getMicCaptureConstraints");function Ii(){const u=window.AudioContext||window.webkitAudioContext;if(!u||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let f=null;try{f=new u({sampleRate:16e3})}catch{if(f){try{f.close()}catch{}f=null}f=new u}const g=f.
resume(),b=navigator.mediaDevices.getUserMedia(On());return{ctx:f,resumePromise:g,streamPromise:b}}o(
Ii,"beginGeminiLiveMic");function Pi(u){if(u){if(u.streamPromise)u.streamPromise.then(f=>{try{f.getTracks().
forEach(g=>g.stop())}catch{}}).catch(()=>{});else if(u.stream)try{u.stream.getTracks().forEach(f=>f.
stop())}catch{}if(u.ctx)try{u.ctx.close()}catch{}}}o(Pi,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(qe){setStsStatus("Processing...",!0);const u=qe;qe=null,u.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const f=await u.
getFinalData();if(isGeminiLiveTranscribeModel()&&(f.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
f.assistant_text=(u.inputTranscript||"").trim(),f.assistant_thought="",!f.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const g=(f.user_text||
"").trim()||(f.assistant_text||"").trim()||(f.assistant_thought||"").trim(),b=!!(f.audio_base64||f.user_audio_base64);
if(!g&&!b){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const S=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(S.id),history.pushState(
{},"","/c/"+S.id),get("welcome-screen").classList.add("hidden")}f.thread_id=currentThreadId,f.model=
get("model-select").value,await apiFetch("/api/skillini/save_sts",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify(f)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(f){console.error("Failed to save G\
emini Live session:",f),setStsStatus("Error saving",!1)}return}if(xn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),xn.stop();return}if(rt&&
rt.state==="recording"){rt.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||Jt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ii();
const _={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(_.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const W=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(be=>be.trim()).filter(Boolean);W.length&&(_.custom_vocabulary=
W.slice(0,1e3))}}else _.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(_.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",_.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(_.target_lang=get("sts-target-lang").value);const S=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:S,ctx:g.ctx,source:g.ctx.createMediaStreamSource(S)};const T=await apiFetch(
"/api/skillini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
_)});if(!T.ok)throw new Error("Failed to get session token");const{token:A,url:I}=await T.json(),O=get(
"model-select").value,F=get("sts-voice")?get("sts-voice").value:"Kore",Z=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",ae=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(qe=new Un,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(qe.rtPlayer=new kn),isGeminiLiveTranscribeModel()){
const W=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",be={languageCodes:[]};
if((W==="SMART"||W==="VERBATIM")&&(be.mode=W),get("sts-custom-vocab")){const Y=get("sts-custom-vocab").
value.split(/[,、\n]/).map(ke=>ke.trim()).filter(Boolean);Y.length&&(be.customVocabulary=Y.slice(0,
1e3))}await qe.start(A,I,O,{transcriptionConfig:be},g)}else if(isGeminiLiveTranslateModel()){const W=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await qe.start(A,I,O,{translationConfig:{targetLanguageCode:W,
echoTargetLanguage:!0}},g)}else{const W={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:F}}}};
isGeminiLiveExtendedThinkingModel()&&(W.thinkingConfig={thinkingLevel:Z,includeThoughts:ae}),await qe.
start(A,I,O,W,g)}g=null,rt=qe.backupRecorder,rt.onstop=()=>{qe&&get("mic-btn").click()};let le=!0,V="\
live-sts-"+Date.now();qe.onMessage=W=>{if(W.serverContent){if(isGeminiLiveTranscribeModel()){const be=qe.
interimInputTranscript,Y=qe.inputTranscript,ke=Y+(be&&!Y.endsWith(be)?(Y?`
`:"")+be:""),Ce=get("chat-messages");let se=document.getElementById(V);se||(se=document.createElement(
"div"),se.id=V,se.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",se.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,Ce.appendChild(se),Ce.scrollTop=Ce.scrollHeight);const xe=se.
querySelector(".message-content");xe.innerText=ke||"\u8074\u304D\u53D6\u308A\u4E2D...",Ce.scrollTop=
Ce.scrollHeight,window.VoiceStudio&&Y&&window.VoiceStudio.log("user",Y);return}if(W.serverContent.modelTurn){
le&&(setStsStatus("Gemini is speaking...",!1),le=!1);const be=get("chat-messages");let Y=document.getElementById(
V);Y||(Y=document.createElement("div"),Y.id=V,Y.className="flex flex-col gap-2 mb-4 assistant-messag\
e bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",Y.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,be.appendChild(Y),be.scrollTop=be.scrollHeight);const ke=Y.
querySelector(".thought-container"),Ce=Y.querySelector(".message-content");qe.assistantThought&&(ke.
classList.remove("hidden"),ke.innerText=qe.assistantThought),Ce.innerText=qe.assistantText,be.scrollTop=
be.scrollHeight,window.VoiceStudio&&(qe.inputTranscript&&window.VoiceStudio.log("user",qe.inputTranscript),
qe.assistantText&&window.VoiceStudio.log("assistant",qe.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),fi(qe.stream),gi(qe.stream);const we=qe;we.onError=W=>{const be=W&&W.message?W.message:
typeof W=="string"?W:"";be&&showToast("Gemini Live: "+be,"error",!0)},we.onClose=W=>{if(qe===we){if(W&&
W.code&&W.code!==1e3){const be=W.reason?": "+W.reason:" (code "+W.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+be,"error",!0)}get("mic-btn").click()}},we.closedEvent&&
we.onClose(we.closedEvent);return}catch(b){if(Pi(g),qe){try{qe.stop()}catch{}qe=null}showToast("Gemi\
ni Live connection failed: "+b.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await xn.start();return}isStsModel()||(pi(),Jt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const u=await navigator.mediaDevices.getUserMedia(On());rt=new MediaRecorder(u),Tn=[],qn=!1;const f=isStsModel();
rt.ondataavailable=g=>Tn.push(g.data),rt.onstop=async()=>{if(qn){Tn=[],get("file-preview").classList.
add("hidden"),u.getTracks().forEach(T=>T.stop()),Pn(),Kt(),f||(Jt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),In=setTimeout(()=>Jt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(Tn,{type:"audio/web\
m"}),b=new File([g],"recording.webm",{type:"audio/webm"}),_=new FormData;_.append("file",b),get("fil\
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
"/sts",{method:"POST",body:_});if(!T.ok){const we=await T.json().catch(()=>({}));throw new Error(we.
error||"Speech-to-speech failed")}const A=T.body.getReader(),I=new TextDecoder;let O="",F=null,Z=null;
stsOpt("sts-auto-play")&&(Z=new kn,an=Z),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let ae=!0,le="",V="";for(;;){const{done:we,value:W}=await A.read();if(we)break;
O+=I.decode(W,{stream:!0});const be=O.split(`
`);O=be.pop();for(const Y of be){if(!Y.trim())continue;const ke=JSON.parse(Y);if(ke.error)throw new Error(
ke.error);ke.audio_delta&&Z&&(ae&&(setStsStatus("Playing response...",!1),ae=!1),await Z.addChunk(ke.
audio_delta)),ke.input_delta&&(le+=ke.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
le)),ke.transcript_delta&&(V+=ke.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",V)),(ke.final||ke.audio_url)&&(F=ke)}}window.VoiceStudio&&!le.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),F&&(F.audio_url||F.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const T=get(
"set-mic-transcribe-mode");if(!!(T&&T.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}_.append("llm_model",get("model-select")&&get("model-select").value||"");const O=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:_})).json();if(O.transcript){const F=get("prompt-inp\
ut");F.value+=(F.value?" ":"")+O.transcript,F.style.height="auto",F.style.height=F.scrollHeight+"px"}else
showToast(O.error||"Transcription failed","error",!0)}}catch(T){showToast("Audio processing error: "+
T.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),u.getTracks().forEach(T=>T.
stop()),Pn(),Kt(),S||Jt("","hidden"),S&&setStsStatus("Tap to speak",!1)}},rt.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Jt("\u9332\u97F3\u4E2D\u2026","recording"),fi(u)),gi(u),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Kt(),isStsModel()||Jt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const _n=o((u,f)=>{if(!u)return;const g=u.querySelector("span");g?g.textContent=f:u.textContent=
f},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const u=lib.
selected.size,f=get("lib-del-btn"),g=get("lib-download-btn"),b=get("lib-attach-btn"),_=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(f&&(f.disabled=u===0,_n(f,u?`\u524A\u9664 (${u})`:"\u524A\u9664")),
g&&(g.disabled=u===0,_n(g,u?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${u})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=u===0,_n(b,u?`\u6DFB\u4ED8 (${u})`:"\u6DFB\u4ED8")),_&&(_.disabled=u!==1,_n(_,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=u!==1,_n(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const T=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",T&&u>0)}};function wi(u){lib.
attachMode=!!u}o(wi,"setLibAttachMode");const ki=o((u=!1)=>{wi(u),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(u=!1)=>{hideModal("lib-modal"),!u&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>ki(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>ki(!0)),
get("lib-search")){let u=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),u&&clearTimeout(u),u=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const u=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=u,get("lib-sort").onchange=()=>{const f=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,f),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0));const Oi=o(()=>{editingSkillUuid=null,get("skill-modal-title").textContent=
"Create New Skill",get("save-skill-btn").innerText="Create Skill",showModal("skill-modal"),get("skil\
l-name").value="",get("skill-desc").value="",get("skill-inst").value="",location.pathname!=="/skill"&&
history.pushState({modal:"skill"},"","/skill")},"openSkillModal");window.closeSkillModal=(u=!1)=>{hideModal(
"skill-modal"),!u&&location.pathname==="/skill"&&history.back()},get("add-skill-btn").onclick=()=>Oi(),
get("save-skill-btn").onclick=async()=>{const u=get("skill-name").value,f=get("skill-desc").value,g=get(
"skill-inst").value;if(u&&g){const b=editingSkillUuid?"PUT":"POST",_=editingSkillUuid?`/api/skills/${editingSkillUuid}`:
CHAT_CONFIG.urls.handleSkills;await apiFetch(_,{method:b,headers:{"Content-Type":"application/json"},
body:JSON.stringify({name:u,description:f,instruction:g})}),window.closeSkillModal(),loadSkills(),editingSkillUuid&&
activeSkill&&activeSkill.uuid===editingSkillUuid&&(activeSkill.name=u,activeSkill.instruction=g,applyActiveSkill(
activeSkill))}else alert("Name and Instruction are required.")},document.addEventListener("click",function(u){
if(u.target.closest(".edit-btn")){const g=u.target.closest(".edit-btn").getAttribute("data-id");beginEditMessage(
g)}if(u.target.closest(".code-toggle")){const f=u.target.closest(".code-toggle"),g=f.closest(".code-\
wrapper");if(!g)return;const b=g.classList.toggle("collapsed");g.setAttribute("data-collapsed",b?"tr\
ue":"false"),f.setAttribute("aria-expanded",b?"false":"true"),f.innerHTML=b?'<i class="fas fa-chevro\
n-down"></i>':'<i class="fas fa-chevron-up"></i>',f.title=b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",
f.setAttribute("aria-label",b?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080")}if(u.target.closest("\
.download-btn")){const f=u.target.closest(".download-btn"),g=f.getAttribute("data-code"),b=(f.getAttribute(
"data-lang")||"txt").toLowerCase();if(g)try{const _=decodeURIComponent(g),S=new Blob([_],{type:"text\
/plain"}),T=URL.createObjectURL(S),A=document.createElement("a");A.href=T;let O={python:"py",javascript:"\
js",typescript:"ts",markdown:"md",html:"html",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"\
sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"\
go",rust:"rs",ruby:"rb",php:"php",perl:"pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"\
toml",ini:"ini",plaintext:"txt",text:"txt"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(O="txt");let F=`\
code.${O}`;b==="dockerfile"&&(F="Dockerfile"),b==="makefile"&&(F="Makefile"),A.download=F,document.body.
appendChild(A),A.click(),document.body.removeChild(A),URL.revokeObjectURL(T)}catch(_){console.error(
"Download failed",_)}}if(u.target.closest(".coding-target-btn")&&selectCodingTargetFromButton(u.target.
closest(".coding-target-btn")),u.target.closest(".copy-btn")){const f=u.target.closest(".copy-btn"),
g=f.getAttribute("data-code");g&&window.copyCode(f,g)}if(u.target.closest(".html-preview-btn")){const g=u.
target.closest(".html-preview-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(u.target.
closest(".canvas-preview-btn")){const f=u.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(
f)}}),document.querySelectorAll(".modal-overlay").forEach(u=>{u.addEventListener("click",f=>{f.target===
u&&ni(u.id)})}),currentThreadId?loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
