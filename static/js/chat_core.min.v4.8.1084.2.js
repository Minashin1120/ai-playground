document.addEventListener("DOMContentLoaded",()=>{var di,ui;initThemeFromServer(),applyLiquidGlassMode(
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
getBoundingClientRect(),g=d.getBoundingClientRect(),b=g.left+g.width/2-(m.left+m.width/2),x=g.top+g.
height/2-(m.top+m.height/2);r.style.transform=`translate(${b}px, ${x}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const m=get("gpt-image-format"),g=get("gpt-image-compression-wrap");m&&g&&(m.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}o(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(b=>{b&&(Array.from(b.options).
forEach(x=>{x.value!=="1K"&&(x.disabled=g)}),g&&b.value!=="1K"&&(b.value="1K"))})}o(c,"updateGeminiI\
mageUi");function u(){const d=get("grok-image-options");if(!d)return;const m=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),b=m==="grok-imagine-image-quality"||m==="grok-imagine-image-2\
.0",x=m==="grok-imagine-image-2.0";if(g){d.classList.remove("hidden");const L=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!b);const A=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.toggle("hidden",!x)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const L=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!b);const A=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;A&&A.classList.toggle(
"hidden",!x)}}o(u,"updateGrokImageUi");function f(){const d=get("ideogram-image-options"),m=get("mod\
al-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",b=isIdeogramModel(g),
x=ideogramModelTraits(g),S={resolution:x.resolution,quality:x.quality,speed:x.speed,style:x.style,negative:x.
negative};if(d&&d.classList.toggle("hidden",!b),m&&m.classList.toggle("hidden",!b),!b)return;IDEOGRAM_IMAGE_FIELDS.
forEach(A=>{const $=S[A]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(H=>{const B=get(H+
A);!B||!B.parentElement||B.parentElement.classList.toggle("hidden",!$)})}),["ideogram-image-style","\
modal-ideogram-image-style"].forEach(A=>{const $=get(A);if(!$)return;const H=x.styleOptions.join("|");
if($.dataset.options!==H){const B=$.value;$.innerHTML=x.styleOptions.map(W=>`<option value="${W}">${W===
"auto"?"Auto":W}</option>`).join(""),$.dataset.options=H,$.value=x.styleOptions.includes(B)?B:"auto"}});
const L=get("ideogram-image-note");L&&(L.textContent=x.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F P\
recise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":"\u203B Ideogram: \u30C6\u30AD\
\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi");function h(){var b;const d=get("grok-video-options");if(!d)return;const m=String(
((b=get("model-select"))==null?void 0:b.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove(
"hidden"):d.classList.add("hidden");const g=get("grok-video-resolution");if(g){const x=Array.from(g.
options).find(S=>S.value==="1080p");x&&(x.disabled=m!=="grok-imagine-video-1.5"),m!=="grok-imagine-v\
ideo-1.5"&&g.value==="1080p"&&(g.value="720p")}}o(h,"updateGrokVideoUi");function y(){var x;const d=get(
"gemini-video-options");if(!d)return;const m=String(((x=get("model-select"))==null?void 0:x.value)||
"").toLowerCase();isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get(
"gemini-video-resolution");if(g){const S=Array.from(g.options).find(A=>A.value==="4K"),L=m==="veo-3.\
1-lite-generate-preview"||m==="veo-3.1-fast-generate-preview"||m==="gemini-omni-flash";S&&(S.disabled=
L),L&&g.value==="4K"&&(g.value="1080p")}const b=get("gemini-video-duration-wrap");b&&b.classList.toggle(
"hidden",m==="gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function v(){const d=get("gemini-mu\
sic-options");if(!d)return;const m=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!m;d.classList.
toggle("hidden",!g);const b=get("lyria-realtime-studio-bar");b&&b.classList.toggle("hidden",!m)}o(v,
"updateGeminiMusicUi");function w(){var L;const d=get("xai-chat-options");if(!d)return;const m=String(
((L=get("model-select"))==null?void 0:L.value)||"").toLowerCase(),g=m.startsWith("grok-")&&!isGrokImageModel(
m)&&!isGrokVideoModel(m)&&!m.includes("voice");d.classList.toggle("hidden",!g);const b=get("xai-logp\
robs"),x=get("xai-top-logprobs"),S=m.includes("grok-4.20");b&&(b.disabled=S,S&&(b.checked=!1)),x&&(x.
disabled=S,S&&(x.value=""))}o(w,"updateXaiChatUi");function k(){const d=isMistralOcrModel(),m=get("m\
istral-ocr-options");m&&m.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(b=>{const x=get(b);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function _(){const d=get("image-input-limit\
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
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059 / \u30DE\u30B9\u30AF\u306B\u306F\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\u3067\
\u3059\u3002\u753B\u50CF\u5165\u529B\u306F Ideogram 4.5 \u3060\u3051\u304C\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u3059</div>"].
join("")):m.includes("grok")&&m.includes("video")&&(b=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),b?(d.innerHTML=g,d.classList.remove("hidden")):(d.classList.add("hidden"),d.innerHTML="")}
o(_,"updateImageInputLimits");function T(){const d=get("enable-sys-prompt"),m=!!(d&&d.disabled),g=!!(d&&
d.checked);M(),d&&(d.disabled?!m&&g?d.dataset.restoreChecked="1":m||delete d.dataset.restoreChecked:
(m&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(T,"toggleOpti\
ons"),window.toggleOptions=T;function M(){const d=get("model-select");if(!d)return;const m=d.value,g=String(
m||"").toLowerCase(),b=g.includes("deepseek"),x=g.startsWith("glm-"),S=get("thinking-options"),L=get(
"reasoning-effort-container"),A=get("enable-thinking"),$=get("thinking-level"),H=get("thinking-budge\
t"),B=get("enable-search"),W=get("search-container"),ue=get("url-context-container"),re=get("enable-\
maps"),F=get("maps-grounding-container"),ae=get("enable-sys-prompt"),V=get("sys-prompt-option"),Ae=get(
"enable-python"),K=get("python-container"),ce=get("prompt-cache-container"),xe=get("enable-prompt-ca\
che"),se=m==="gpt-5-search-api",fe=m.includes("tts"),Ee=isMistralOcrModel(m),$e=g.includes("gemini-3\
.1-flash-lite-image"),st=g==="gemini-nano-banana-2.1",Ue=g.includes("gemini-3.1-flash-image")&&!$e||
st,ut=isClaudeModelKey(m),Ct=g==="gemini-3.8-flash-cyber",ke=isLlmModel()&&!b&&!fe&&!g.includes("rea\
ltime")&&!g.includes("native-audio")&&!g.includes("live");ce&&(ke?(ce.classList.remove("hidden","opa\
city-50","pointer-events-none"),xe&&(xe.disabled=!1)):(xe&&(xe.checked=!1,xe.disabled=!0),ce.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),L&&L.classList.
add("hidden");const ge=get("vision-model-info");if(ge&&ge.classList.add("hidden"),L){const be=get("r\
easoning-effort");if(be){Array.from(be.options).forEach(He=>{const ze=g==="gpt-5.6"||g.startsWith("g\
pt-5.6-"),ct=g.startsWith("gpt-6"),cn=g==="gpt-6-sol"||g==="gpt-6-luna",Vt=g==="deepseek-v4.1-flash"||
g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash-vision-exp",dn=g==="de\
epseek-v4-pro",jn=g.includes("grok-4.5"),R=g.includes("grok-4.6");He.value==="max"?He.classList.toggle(
"hidden",!ze&&!ct&&!Vt&&!dn):He.value==="xhigh"?He.classList.toggle("hidden",!R&&!g.includes("multi-\
agent")&&!ze&&!ct):He.value==="medium"?He.classList.toggle("hidden",!(g.includes("grok-4.3")||jn||R||
g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("multi-agent")||g.includes(
"gpt-5")||ct||g.includes("o1")||g.includes("o3"))):He.value==="none"?He.classList.toggle("hidden",!g.
includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!cn&&!Vt&&!dn):He.value==="lo\
w"&&He.classList.toggle("hidden",dn)});const Pe=be.selectedOptions&&be.selectedOptions[0];Pe&&Pe.classList.
contains("hidden")&&(be.value=b?"high":"medium")}}ue&&ue.classList.add("hidden"),F&&F.classList.add(
"hidden"),A&&(A.disabled=!1),H&&(H.disabled=!0,H.classList.add("opacity-50"));const Be=isGeminiImageModelKey(
m);if(fe||Ee)W&&(get("enable-search").checked=!1,W.classList.add("opacity-50","pointer-events-none")),
ue&&(get("enable-url-context").checked=!1,ue.classList.add("opacity-50","pointer-events-none")),F&&re&&
(re.checked=!1,F.classList.add("opacity-50","pointer-events-none")),K&&(Ae.checked=!1,K.classList.add(
"opacity-50","pointer-events-none")),ae&&V&&(ae.checked=!1,ae.disabled=!0,V.classList.add("opacity-5\
0"));else if(Ue||$e)F&&re&&(re.checked=!1,F.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),st?(Array.from($.options).forEach(be=>{be.disabled=be.value==="low"}),[
"minimal","medium","high"].includes($.value)||($.value="medium")):(Array.from($.options).forEach(be=>{
["low","medium"].includes(be.value)&&(be.disabled=!0),["minimal","high"].includes(be.value)&&(be.disabled=
!1)}),["minimal","high"].includes($.value)||($.value=$e?"minimal":"high")),A&&(A.disabled=!1),$e&&(B&&
(B.checked=!1,B.disabled=!0),W&&W.classList.add("opacity-50","pointer-events-none"));else if(Be)F&&re&&
(re.checked=!1,F.classList.add("hidden","opacity-50","pointer-events-none"));else if(ut)S.classList.
remove("hidden"),H&&(H.disabled=!1,H.classList.remove("opacity-50")),Array.from($.options).forEach(be=>{
be.disabled=!0}),K&&(Ae.checked=!1,K.classList.add("opacity-50","pointer-events-none"));else if(Ct){
S&&S.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from($.options).forEach(Pe=>{Pe.
disabled=!["low","medium","high"].includes(Pe.value)}),["low","medium","high"].includes($.value)||($.
value="medium"),[W,ue,F,K].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")}),[
B,re,Ae].forEach(Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const be=get("enable-url-context");be&&(be.
checked=!1,be.disabled=!0),ae&&V&&(ae.disabled=!1,V.classList.remove("opacity-50"))}else if(m.includes(
"gemini")&&!Be){S.classList.remove("hidden"),ue&&ue.classList.remove("hidden","opacity-50","pointer-\
events-none");const be=m.includes("gemini-3");F&&(be?F.classList.remove("hidden","opacity-50","point\
er-events-none"):(re&&(re.checked=!1),F.classList.add("hidden","opacity-50","pointer-events-none")));
const Pe=m.includes("flash");Array.from($.options).forEach(He=>{m==="gemini-3.8-flash"||m==="gemini-\
3.7-flash"?He.disabled=!["low","medium","high"].includes(He.value):m==="gemini-3.6-flash"?He.disabled=
!["medium","high"].includes(He.value):m==="gemini-3.5-flash-lite"?He.disabled=!["minimal","medium","\
high"].includes(He.value):["minimal","medium"].includes(He.value)?He.disabled=!Pe:He.disabled=!1}),(m===
"gemini-3.8-flash"||m==="gemini-3.7-flash")&&!["low","medium","high"].includes($.value)||m==="gemini\
-3.6-flash"&&!["medium","high"].includes($.value)?$.value="medium":m==="gemini-3.5-flash-lite"&&!["m\
inimal","medium","high"].includes($.value)?$.value="minimal":!Pe&&["minimal","medium"].includes($.value)&&
($.value="high"),be?A&&(A.checked=!0,A.disabled=!0):A&&(A.disabled=!1),H&&m.includes("gemini-2.5")&&
(H.disabled=!1,H.classList.remove("opacity-50")),H&&!m.includes("gemini-2.5")&&(H.disabled=!0,H.classList.
add("opacity-50"))}if(isLlmModel()&&(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes(
"grok-4.3")||g.includes("grok-4.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||
g.includes("grok-build")||g.includes("multi-agent")||g.includes("gpt")&&!g.includes("tts")))L.classList.
remove("hidden"),W&&W.classList.remove("opacity-50","pointer-events-none");else if(b){L.classList.remove(
"hidden");const be=get("vision-model-info");if(be&&be.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),B&&(B.checked=!1,B.disabled=!0),W&&W.classList.add("opaci\
ty-50","pointer-events-none"),ue){const Pe=get("enable-url-context");Pe&&(Pe.checked=!1),ue.classList.
add("opacity-50","pointer-events-none")}F&&re&&(re.checked=!1,F.classList.add("opacity-50","pointer-\
events-none"))}else Ee||(W&&W.classList.remove("opacity-50","pointer-events-none"),F&&re&&(re.checked=
!1,F.classList.add("hidden","opacity-50","pointer-events-none")));if(fe?K&&K.classList.add("opacity-\
50","pointer-events-none"):(K&&K.classList.remove("opacity-50","pointer-events-none"),(!Be||Ue)&&!m.
includes("gpt-image")&&(ae.disabled=!1,V.classList.remove("opacity-50"))),(Be&&!Ue||m.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(m)||isGrokVideoModel()||Ee)&&ae&&V&&(ae.checked=!1,ae.disabled=
!0,V.classList.add("opacity-50")),K&&(isLlmModel()?(K.classList.remove("hidden"),Ae.disabled=!1):(Ae.
checked=!1,Ae.disabled=!0,K.classList.add("hidden"))),se?(B&&(B.checked=!0,B.disabled=!0),W&&W.classList.
add("opacity-50","pointer-events-none"),K&&(Ae.checked=!1,Ae.disabled=!0,K.classList.add("opacity-50",
"pointer-events-none"))):B&&!m.includes("tts")&&!Ee&&!b&&!$e&&(B.disabled=!1),Ct){[B,re,Ae].forEach(
Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const be=get("enable-url-context");be&&(be.checked=!1,be.disabled=
!0),[W,ue,F,K].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")})}x&&(ge&&ge.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[B,Ae].forEach(be=>{be&&(be.checked=!1,be.disabled=!0)}),[W,K].forEach(be=>{be&&
be.classList.add("opacity-50","pointer-events-none")}));const at=get("mask-btn");at&&(isGptImageModel()?
at.classList.remove("hidden"):(at.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),
updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),u(),f(),h(),y(),v(),updateBatchUi(m),w(),k(),
_(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}o(M,"toggle\
OptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",T),get("model-\
select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),T(),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const P=get("enable-canvas-mode");P&&(P.checked=canvasModeEnabled,P.addEventListener("change",()=>syncCanvasModeUi(
P.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const G=get("enable-codin\
g-mode");G&&(G.checked=codingModeEnabled,G.addEventListener("change",()=>syncCodingModeUi(G.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const ee=get(
"prompt-controls-toggle-btn");ee&&(ee.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,m,g){const b=(m||"").match(/\S*/)[0];if(b==="py\
exec")return"";if(b==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",S=(b||"").toLowerCase();
let L="";try{const V=hljs.getLanguage(b)?b:"plaintext";activeStreamingBubbleId&&x.length>2e4?L=escapeHtml(
x):L=hljs.highlight(x,{language:V}).value}catch{L=escapeHtml(x)}const A=encodeURIComponent(x).replace(
/'/g,"%27"),$=detectBlockedScriptsInCode(x),H=hashString(`${b||"TEXT"}
${x||""}`);let B="";if(canvasModeEnabled){const V=String(canvasPreviewState.selectedKey||"")===H,Ae=V?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";B=`<button\
 class="canvas-preview-btn${V?" canvas-active":""}" data-code="${A}" data-code-key="${H}" data-canva\
s-lang="${escapeHtml(b||"txt")}" title="${Ae}" aria-label="${Ae}" aria-pressed="${V?"true":"false"}"\
><i class="fas ${V?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,x)){const V=$?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
B=`<button class="html-preview-btn" data-code="${A}" ${$?'data-suspicious="1"':""} title="${V}" aria\
-label="${V}"><i class="fas ${$?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const W=`\
<button class="download-btn" data-code="${A}" data-lang="${b||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ue=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${H}" data-coding-lang="${escapeHtml(b||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,re=(b||"TEXT")+($?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),F=getRenderableSvgCode(S,x);return`${F?buildSvgCodeRenderHtml(F,H):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${H}"><div class="code-header"\
><span class="code-lang">${re}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${ue}${B}${W}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${b}">${L}</code></pre></d\
iv></div>`},link(d,m,g){return`<a href="${d}" title="${m||""}" target="_blank">${g}</a>`},image(d,m,g){
return buildChatImageHtml(d,{alt:g,title:m})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(di=get("version-update-dismis\
s"))==null||di.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const X=get("version-update-clear-\
cache");if(X&&(X.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),X.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(X.checked)})),
(ui=get("version-update-reload"))==null||ui.addEventListener("click",async()=>{var m;await versionUpdateCachePreferenceSavePromise.
catch(()=>{}),!!((m=get("version-update-clear-cache"))!=null&&m.checked)?await clearSiteCacheAndReload(
get("version-update-reload"),{scanFirst:!0}):location.reload()}),window.ConnectionMonitor&&(window.ConnectionMonitor.
setVersionChangeHandler(d=>{d&&d!==appVersion&&(localStorage.getItem("version_notified")||"")!==d&&(localStorage.
setItem("app_version",d),purgeCaches().then(()=>checkAndNotifyVersion(d)))}),window.ConnectionMonitor.
start(),window.addEventListener("online",()=>window.ConnectionMonitor.probeNow()),window.addEventListener(
"offline",()=>{window.ConnectionMonitor.cancelProbe(),window.ConnectionMonitor.setUnavailable("offli\
ne")}),window.addEventListener("focus",()=>window.ConnectionMonitor.probeNow()),document.addEventListener(
"visibilitychange",()=>{document.hidden||window.ConnectionMonitor.probeNow()}),window.addEventListener(
"pagehide",()=>window.ConnectionMonitor.stop())),applyCacheMode(useSwCache),botConfig&&botConfig.lock&&
botConfig.lock.active&&!isAdminUser&&showBotLockOverlay(botConfig.lock.message,botConfig.lock.remaining_seconds),
window.__turnstileApiLoaded&&window.initTurnstileWidget&&window.initTurnstileWidget(),botConfig&&botConfig.
globalEnabled&&botConfig.accountEnabled&&!isAdminUser){botConfig.turnstileVerified&&(botDetectionVerified=
!0);try{botTelemetry.start()}catch(d){console.error(d)}try{runBotDetectionGate()}catch(d){console.error(
d)}}else{const d=get("turnstile-container");d&&d.classList.add("hidden")}const ye=o(d=>{if(!d)return"\
\u4E0D\u660E";const m=new Date(d);return Number.isNaN(m.getTime())?d:m.toLocaleString()},"formatSess\
ionTime"),q=o(d=>{const m=Array.isArray(d)?d:[],g=get("passkey-list"),b=get("passkey-count");if(b&&(b.
innerText=String(m.length)),!!g){if(!m.length){g.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
g.innerHTML="",m.forEach((x,S)=>{const L=x&&x.id?String(x.id):"",A=document.createElement("div");A.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const $=document.
createElement("div");$.className="min-w-0";const H=document.createElement("div");H.className="text-x\
s text-gray-200 truncate",H.innerText=x&&x.name?String(x.name):`Security Key ${S+1}`;const B=document.
createElement("div");B.className="text-[10px] text-gray-500 mt-1",B.innerText=x&&x.created_at?`\u767B\u9332\u65E5\u6642:\
 ${ye(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",$.appendChild(H),$.appendChild(B),A.appendChild(
$);const W=document.createElement("button");W.type="button",W.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",W.innerText="\u524A\u9664",W.
disabled=!L,L&&(W.onclick=()=>window.removeWebAuthnCredential(L)),A.appendChild(W),g.appendChild(A)})}},
"renderPasskeyList"),Y=o(d=>{const m=get("session-list");if(m){if(!d||!d.length){m.innerHTML='<div c\
lass="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}m.innerHTML=d.map(g=>{const b=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
L=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${b}${x}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(L)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(ye(g.last_seen_at))}\
 / \u4F5C\u6210: ${escapeHtml(ye(g.created_at))}</div></div>${S}</div>`}).join(""),m.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const b=g.getAttribute("data-session-id");if(!b||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:b})});let S={};try{S=await x.json()}catch{}if(x.ok){if(S.logged_out){
location.href="/login";return}await ne()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),ne=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const m=await apiFetch("/api\
/sessions");let g={};try{g=await m.json()}catch{}if(!m.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const b=(g.sessions||[]).filter(x=>!x.is_revoked);Y(b)},"loadSessions"),ve=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>ne());const m=get("session-revoke-others-btn");m&&(m.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await ne():showToast("\u64CD\u4F5C\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(d=>{d&&(currentVisionModel=d.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
d);try{loadMcpServers()}catch{}d&&d.theme_color&&applyThemeColor(d.theme_color,!0),d&&Object.prototype.
hasOwnProperty.call(d,"minimal_prompt_mode")&&d.minimal_prompt_mode?setMinimalPromptMode(!0):d&&Object.
prototype.hasOwnProperty.call(d,"compact_prompt_mode")&&setCompactPromptMode(!!d.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings sync");
const m=get("enable-sys-prompt");m&&d&&d.system_prompt&&String(d.system_prompt).trim()&&(!m.disabled&&
!d.default_enable_system_prompt&&!d.use_last_chat_settings&&(m.checked=!0),T())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const pe=get("vision-model-change-btn");pe&&(pe.onclick=()=>_openVisionModelSelector());const Me=get(
"compression-format-only");Me&&(Me.onchange=()=>{const d=Me.checked,m=get("compression-max-size"),g=get(
"compression-max-dim");m&&(m.disabled=d),g&&(g.disabled=d);const b=get("compression-size-wrap"),x=get(
"compression-dim-wrap");b&&(b.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const I=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const m=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(m),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");I(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const N=get("storage-us\
age-refresh");N&&(N.onclick=()=>loadStorageUsage());let D=null;const O=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,m=>m.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),E=o((d={})=>{const m=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),b=get("account-transfer-progress-percent"),x=get("account-transfer-progress-text"),
S=get("account-transfer-progress-detail"),L=Math.max(0,Math.min(100,Number(d.progress)||0));if(m&&m.
classList.remove("hidden"),g&&(g.style.width=`${L}%`),b&&(b.textContent=`${Math.round(L)}%`),x&&(x.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const $={queued:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=$[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),C=o(d=>{te&&(te.disabled=
!!d);const m=get("account-import-btn");m&&(m.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),j=o((d={})=>{const m=get("account-export-ready"),g=get(
"account-export-ready-text"),b=get("account-export-expiry"),x=get("account-export-download-btn"),S=!!(d.
available&&d.download_url);if(m&&m.classList.toggle("hidden",!S),!S){x&&x.removeAttribute("href");return}
const L=Math.max(0,Number(d.size_bytes)||0),A=L>=1024*1024*1024?`${(L/(1024*1024*1024)).toFixed(2)} \
GB`:`${(L/(1024*1024)).toFixed(1)} MB`;if(g){const $=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${$}`}
if(b){const $=d.expires_at?new Date(d.expires_at):null;b.textContent=$&&!Number.isNaN($.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${$.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),Q=o(async d=>{for(;D===d&&!d.stopped;){
try{const m=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await m.json().catch(()=>({}));if(m.ok&&(g.state!=="pending"&&E(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(m=>setTimeout(m,700))}
return null},"pollAccountTransfer"),ie=o((d,m,g=!0)=>{m&&(E(m),j(m),g&&m.state==="ready"?showToast(m.
message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(m.unreadable_count)>0?"warning":"success",Number(m.unreadable_count)>0):g&&m.state==="failed"&&
showToast(m.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),Le(d))},"handleFinishedAccountExport"),le=o(async()=>{try{const d=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),m=await d.json().catch(()=>({}));
if(!d.ok)return;if(j(m),m.state==="ready"){E(m);return}if(["failed","cancelled","expired"].includes(
m.state)){E(m);return}if(!["queued","running","cancelling"].includes(m.state)||!m.job_id||D&&D.id===
m.job_id||D)return;const g={id:m.job_id,type:"export",stopped:!1,restored:!0};D=g,C(!0),E(m);const b=await Q(
g);b&&ie(g,b,!0)}catch{}},"refreshLatestAccountExport"),Le=o(d=>{D===d&&(D=null),d.stopped=!0,C(!1)},
"finishAccountTransfer"),oe=get("account-transfer-cancel-btn");oe&&(oe.onclick=async()=>{const d=D;if(!(!d||
d.stopped)){d.cancelRequested=!0,oe.disabled=!0,E({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),E({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&j({available:!1}),Le(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const te=get("account-export-btn");te&&(te.onclick=async()=>{if(D)return;const d={id:O(),
type:"export",stopped:!1};D=d,C(!0),j({available:!1}),E({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u53D7\
\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const m=await apiFetch("/api/account/export",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:d.id}),keepalive:!0})),
g=await m.json().catch(()=>({}));if(m.status===409&&g.error==="export_in_progress"&&g.job_id)d.id=g.
job_id;else if(!m.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
E({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const b=await Q(d);!d.cancelRequested&&b&&ie(d,b,!0)}catch(m){const g=m&&m.message?m.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";E({progress:0,phase:"failed",message:g}),
showToast(g,"error",!0),Le(d)}});const Z=get("account-export-download-btn");Z&&Z.addEventListener("c\
lick",async d=>{const m=Z.getAttribute("href");if(!(!m||m==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),b=await g.json().catch(
()=>({}));g.ok&&b.available&&b.download_url?(Z.href=b.download_url,window.location.assign(b.download_url)):
(j(b),E(b),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),le())}catch{window.location.assign(m)}}}),C(!1),le();const we=get("import-files-grid"),
rt=get("import-files-info"),dt=get("import-files-summary"),mt=o(d=>{const m=Math.max(0,Number(d)||0);
return m>=1024*1024*1024?`${(m/(1024*1024*1024)).toFixed(2)} GB`:m>=1024*1024?`${(m/(1024*1024)).toFixed(
1)} MB`:m>=1024?`${Math.round(m/1024)} KB`:`${m} B`},"importFormatBytes");let Ne=null;const Ve=o(()=>{
if(!Ne)return;const d=Ne.files,m=Ne.selection;let g=0;d.forEach(S=>{m.has(S.archive_path)&&(g+=Number(
S.size_bytes)||0)});const b=Number(Ne.available_bytes)||0,x=g>b;dt&&(dt.textContent=`\u9078\u629E\u4E2D: ${mt(
g)} / \u5229\u7528\u53EF\u80FD: ${mt(b)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,dt.classList.
toggle("text-red-300",x)),rt&&(rt.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),bt=o(
()=>{if(!we||!Ne)return;we.innerHTML="";const d=Ne.files;if(!d.length){we.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
Ve();return}d.forEach(m=>{const g=document.createElement("label"),b=Ne.selection.has(m.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${b?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${b?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(m.display_name)}">${escapeHtml(
m.display_name)}</div><div class="text-[10px] text-gray-500">${mt(m.size_bytes)}</div></div>`;const x=g.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?Ne.selection.add(m.archive_path):
Ne.selection.delete(m.archive_path),g.classList.toggle("border-blue-500",x.checked),g.classList.toggle(
"border-gray-600",!x.checked),Ve()}),we.appendChild(g)}),Ve()},"renderImportFileItems"),Mt=o(d=>new Promise(
m=>{if(Ne={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:m},bt(),!get("import-files-modal")){m(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),wt=o(d=>{if(hideModal("import-files-modal"),Ne){const m=Ne.resolve;
Ne=null,m(d)}},"closeImportFileSelection"),At=get("import-files-close");At&&(At.onclick=()=>wt(null));
const yt=get("import-files-cancel");yt&&(yt.onclick=()=>wt(null));const Rt=get("import-files-confirm");
Rt&&(Rt.onclick=()=>{if(!Ne)return;const d=Array.from(Ne.selection);wt(d.length?d.join(","):"__none_\
_")});const Et=get("import-files-select-all");Et&&(Et.onclick=()=>{Ne&&(Ne.files.forEach(d=>Ne.selection.
add(d.archive_path)),bt())});const ft=get("import-files-none");ft&&(ft.onclick=()=>{Ne&&(Ne.selection.
clear(),bt())});const Bt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
d);return m.length>60?m.slice(0,60)+"\u2026":m},"formatAccountSettingValue");let xt=null;const vt=o(
d=>{if(xt){const m=xt;xt=null,hideModal("settings-confirmation-modal"),m(d)}},"resolveSettingsImport\
Confirmation"),Jt=o(d=>new Promise(m=>{if(!get("settings-confirmation-modal")){m(!0);return}xt=m;const b=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(b.length?
x.innerHTML=b.map(L=>{const A=Bt[L.field]||L.field,$=Kt(L.current),H=Kt(L.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
$)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(H)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${b.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Ft=get("settings-confir\
mation-modal");Ft&&Ft.addEventListener("click",d=>{d.target===Ft&&vt(!1)});const U=get("settings-con\
firmation-close");U&&(U.onclick=()=>vt(!1));const me=get("settings-confirmation-cancel");me&&(me.onclick=
()=>vt(!1));const Ie=get("settings-confirmation-confirm");Ie&&(Ie.onclick=()=>vt(!0));const Re=get("\
account-import-btn"),Ke=get("account-import-inplace"),tt=get("account-import-inplace-warning");if(Ke&&
tt){const d=o(()=>tt.classList.toggle("hidden",!Ke.checked),"syncInplaceWarn");Ke.addEventListener("\
change",d),d()}Re&&(Re.onclick=async()=>{const d=get("account-import-file"),m=d&&d.files?d.files[0]:
null,g=get("account-import-categories"),b=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(re=>re.value):[],x=get("account-import-inplace"),S=!!(x&&x.checked),L=get("account-impo\
rt-settings-bypass"),A=!!(L&&L.checked);let $=!1;if(!m){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!b.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const H=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(re=>(re.
closest("label")&&re.closest("label").textContent||re.value).trim()):b;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${H.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const B={id:O(),type:"import",stopped:!1,controller:new AbortController};
D=B,C(!0),E({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const W=get("account-import-result");let ue=Promise.resolve(null);try{const F=Math.max(1,Math.ceil(m.
size/10485760)),ae=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:m.size}),signal:B.controller.
signal})),V=await ae.json().catch(()=>({}));if(!ae.ok)throw new Error(V.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
B.uploadId=V.upload_id;const Ae=V.chunk_size||10485760;let K=0,ce=0;const xe=o(async()=>{for(;;){const ke=ce++;
if(ke>=F)return;const ge=m.slice(ke*Ae,Math.min(m.size,(ke+1)*Ae)),Be=new FormData;Be.append("chunk",
ge,m.name),Be.append("index",String(ke));const Ye=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Be,signal:B.controller.signal})),
at=await Ye.json().catch(()=>({}));if(!Ye.ok)throw new Error(at.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
K++,E({progress:Math.min(35,Math.round(K/F*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${K}\
/${F}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let se=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),se=!0);try{await Promise.
all([xe(),xe(),xe()]);const ke=await apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:B.controller.signal})),ge=await ke.json().
catch(()=>({}));if(!ke.ok)throw new Error(ge.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
E({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
se&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let fe="",Ee=!1,$e=0;const st=o(
async()=>{let ke=!1;const ge=o(()=>{ke||(ke=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Be=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Ye=await Be.
json().catch(()=>null);if(!Be.ok||!Ye){ge();return}cacheUserSettings(Ye);const at=get("settings-moda\
l");if(at&&at.classList.contains("modal-open"))try{Un(Ye)}catch{}Ye.theme_color&&applyThemeColor(Ye.
theme_color,!0),Object.prototype.hasOwnProperty.call(Ye,"minimal_prompt_mode")&&Ye.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Ye,"compact_prompt_mode")&&setCompactPromptMode(
!!Ye.compact_prompt_mode)}catch{}ge()},"refreshSettingsFormAfterImport"),Ue=o(ke=>{const ge=ke&&ke.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";W&&(W.textContent=`\u5B8C\u4E86: ${ge}`,
W.classList.remove("hidden","text-red-300"),W.classList.add("text-emerald-300")),E({progress:100,phase:"\
completed",message:ge}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&st()},"finishImportSuccess"),ut=o(async()=>{
try{const ge=await(await apiFetch(`/api/account/transfer/${B.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return ge&&ge.state?ge:null}catch{return null}},"fetchImportSta\
tus"),Ct=o(async()=>{const ke=await ut();if(!ke)return{status:"unknown"};if(ke.state==="completed")return Ue(
ke),{status:"done"};if(["failed","cancelled","expired"].includes(ke.state))throw new Error(ke.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(ke.state==="needs_sel\
ection"&&Array.isArray(ke.files)){const ge=await Mt({files:ke.files,available_bytes:ke.available_bytes});
return ge===null?(E({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(fe=ge,{status:"reselect"})}if(ke.state===
"needs_settings_confirmation"&&Array.isArray(ke.settings_changes))return await Jt({settings_changes:ke.
settings_changes})?($=!0,{status:"reselect"}):(E({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(ke.state==="running"){const ge=await Promise.race([ue.catch(()=>null),new Promise(Be=>setTimeout(
()=>Be(null),6e4))]);if(ge&&ge.state==="completed")return Ue(ge),{status:"done"};throw ge&&["failed",
"cancelled","expired"].includes(ge.state)?new Error(ge.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ee;){B.stopped=!0,await ue.catch(()=>null),
B.stopped=!1,ue=Q(B);let ke;try{ke=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:B.uploadId,
categories:b.join(","),job_id:B.id,selected_files:fe,restore_inplace:S,confirm_settings:$||A}),signal:B.
controller.signal}))}catch(ze){if(B.cancelRequested||ze&&ze.name==="AbortError")throw ze;const ct=await Ct();
if(ct.status==="done"){Ee=!0;break}if(ct.status==="cancelled")return;if(ct.status==="reselect")continue;
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let ge=null;try{ge=await ke.json()}catch{ge=null}if(ge===null){const ze=await Ct();if(ze.status==="d\
one"){Ee=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;if(ke.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!ke.ok&&ge.error==="storage_limit_files"&&ge.files){const ze=await Mt(ge);if(ze===null){E({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}fe=ze;continue}if(ge&&ge.status==="settings_confirmation"&&
Array.isArray(ge.settings_changes)){if(!await Jt(ge)){E({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}$=!0;continue}if(!ke.ok)throw new Error(ge.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Be=ge.imported||{},Ye=[`\u8A2D\u5B9A ${Be.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Be.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Be.chats||0}\u4EF6`,`Gem ${Be.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Be.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Be.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Be.
diagnostics||0}\u4EF6`].join(" / "),at=ge.duplicates||{},be={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Pe=[];for(const ze of Object.keys(be)){const ct=Number(at[ze])||0;ct>
0&&Pe.push(`${be[ze]} ${ct}\u4EF6`)}const He=Pe.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Pe.
join("\u3001")}\uFF09`:"";W&&(W.textContent=`\u5B8C\u4E86: ${Ye}${He}`,W.classList.remove("hidden","\
text-red-300"),W.classList.add("text-emerald-300")),E({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),b.includes("chats")&&loadThreads(),b.includes("gems")&&loadGems(),b.includes("files")&&loadStorageUsage(),
(b.includes("settings")||b.includes("api_credentials"))&&st(),Ee=!0}}catch(re){if(B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),B.cancelRequested||re&&re.name==="AbortError")return;const F=re&&re.message?
re.message:"",ae=F==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
F||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";E({progress:0,phase:"f\
ailed",message:ae}),W&&(W.textContent=ae,W.classList.remove("hidden","text-emerald-300"),W.classList.
add("text-red-300")),showToast(ae,"error",!0)}finally{B.stopped=!0,await ue.catch(()=>null),Le(B)}});
const lt=get("account-dedupe-btn"),qe=get("account-dedupe-result"),je=o((d,m=!1)=>{qe&&(qe.textContent=
d,qe.classList.remove("hidden"),qe.classList.toggle("text-red-300",!!m),qe.classList.toggle("text-em\
erald-300",!m))},"showDedupeResult");lt&&(lt.onclick=async()=>{const d=o(async()=>{const m=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await m.json().catch(()=>null);if(!m.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){je("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const b=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const B of[
"chats","gems","files","feedback","diagnostics"]){const W=Number(g.duplicates&&g.duplicates[B])||0;W>
0&&b.push(`${x[B]} ${W}\u4EF6`)}const S=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${b.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const L=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await L.json().catch(()=>null);if(!L.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const $=[];for(const B of["chats","gems","files","feedback","diagnostics"]){const W=Number(A.removed&&
A.removed[B])||0;W>0&&$.push(`${x[B]} ${W}\u4EF6`)}const H=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";je(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${$.join("\u3001")||"0\u4EF6"}${H}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!lt.disabled){lt.disabled=!0,je("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(m){je(m&&m.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{lt.disabled=!1}}});const Ze=get("site-cache-usage-refresh");Ze&&(Ze.onclick=()=>loadSiteCacheUsage());
const We=get("clear-site-cache-btn");We&&(We.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(We)});
const gt=get("enc-scan-result"),jt=o(async(d=null)=>{gt&&(gt.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let m="/api/encryption_scan";d&&(m+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
m,{cache:"no-store"}),b=await g.json();if(!g.ok){gt&&(gt.textContent=b.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=b.total||0,S=b.encrypted||0,L=b.unencrypted||0;let A=`Total: ${x} / Encrypted: ${S} /\
 Plain: ${L}`;if(b.samples&&b.samples.length){const $=b.samples.slice(0,8).map(H=>{const B=H.timestamp?
new Date(H.timestamp).toLocaleString():"";return`#${H.id} (${H.role||""}) ${B}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${$}</div>`}gt&&(gt.innerHTML=A)}catch{gt&&(gt.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Xt=get("enc-scan-all");Xt&&(Xt.onclick=()=>jt(
null));const Yt=get("enc-scan-thread");Yt&&(Yt.onclick=()=>currentThreadId?jt(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Te=get("admin-enc-\
list");let De=null,Je=!1;const un=o(d=>!d||!d.length?null:d.some(m=>!!m.is_encrypted),"computeThread\
EncryptedFromMessages"),pn=o(()=>{De=un(allMessages)},"refreshCurrentThreadEncStateFromMessages"),_t=o(
async(d,m,{confirmPrompt:g=!0,reloadCurrent:b=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=m?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Je=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:m})}),
L=await S.json().catch(()=>({}));return S.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${L.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),De=!!m,b&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Te&&await Se(),!0):
(showToast(L.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Je=!1}},"setAdminThreadEncry\
ption"),he=o(d=>{if(!Te)return;const m=d.threads||[];if(!m.length){Te.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Te.innerHTML=m.map(g=>{const b=g.encrypted_count>0?"enc":"plain",x=b==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=b==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",L=g.
updated_at?new Date(g.updated_at).toLocaleString():"",A=escapeHtml(String(g.thread_id)),$=currentThreadId&&
String(currentThreadId)===String(g.thread_id);return`<div class="flex items-center gap-2 bg-gray-800\
/60 border border-gray-700 rounded p-2">
                        <div class="flex-1 min-w-0">
                            <div class="font-bold text-gray-200 truncate" title="${escapeHtml(g.title||
"")}">${escapeHtml(g.title||"(\u7121\u984C)")}${$?' <span class="text-[10px] text-cyan-300 font-norm\
al">\uFF08\u8868\u793A\u4E2D\uFF09</span>':""}</div>
                            <div class="text-[10px] text-gray-500">${L} / \u30E1\u30C3\u30BB\u30FC\u30B8: ${g.
message_count} / \u6697\u53F7\u5316: ${g.encrypted_count}</div>
                        </div>
                        <button type="button" class="admin-enc-open bg-gray-700 hover:bg-gray-600 te\
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${b==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Se=o(async()=>{if(Te){Te.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),m=await d.json().catch(()=>({}));
if(!d.ok){Te.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(m.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(he(m),currentThreadId&&Array.isArray(m.threads)){const g=m.threads.find(b=>String(
b.thread_id)===String(currentThreadId));g&&(De=!!g.encrypted)}}catch{Te.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Se()),window.__loadAdminEncThreads=
Se,window.__refreshAdminThreadEncState=pn,window.__setAdminThreadEncryption=_t;const Xe=get("encrypt\
ion-status-admin-toggle");Xe&&Xe.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Te&&(Te.onclick=async d=>{const m=d.target.closest("\
.admin-enc-open");if(m){d.preventDefault();const A=m.getAttribute("data-id");if(!A)return;typeof Dt==
"function"?Dt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||Je)return;const b=g.getAttribute("data-i\
d"),x=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const L=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await _t(b,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=L,await Se()}}),get("file-input").onchange=d=>{const m=Array.from(d.target.
files||[]);d.target.value="",m.length&&handleFiles(m)},get("photo-input")&&(get("photo-input").onchange=
d=>{const m=Array.from(d.target.files||[]);d.target.value="",m.length&&handleFiles(m)});const ot=o(d=>{
const m=get("ban-appeal-list");if(m){if(!d||!d.length){m.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
m.innerHTML=d.map(g=>{const b=g.status||"new",x=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
S=g.created_at?new Date(g.created_at).toLocaleString():"",L=g.replied_at?new Date(g.replied_at).toLocaleString():
"",A=g.admin_reply||"";return`
                        <div class="border border-gray-700/70 rounded p-2 bg-gray-900/60" data-appea\
l-id="${g.id}">
                            <div class="flex items-center justify-between">
                                <div class="text-xs text-blue-200 font-bold">${escapeHtml(g.username||
"")}${x}</div>
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
                    `}).join("")}},"renderBanAppeals"),nt=o(async(d=!1)=>{if(!isAdminUser)return;const m=get(
"ban-appeal-count");if(m)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const x=(await g.json()).unread_count||0;m.textContent=String(x),d&&x>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${x}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),ht=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const m=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!m.ok)return;const g=await m.json();ot(g.items||[]),
await nt(!1)}catch{}}},"loadBanAppeals"),St=o(async(d=null)=>{if(!isAdminUser)return;const m=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(m)})).ok&&await ht()}catch{}},"markBanAppealsRead"),_n=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await ht()}catch{}},"updateBanAppealStatus"),mi=o(()=>{
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
                `,d.appendChild(m)},"ensureTemporaryChatSettingsCard"),Hn=o(()=>{const d=get("set-st\
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
                `,m.appendChild(g);const b=get("reset-llm-transcribe-prompt");b&&(b.onclick=()=>{const x=get(
"set-llm-transcribe-prompt");x&&(x.value=""),showToast("LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"ensureLlmTranscribePromptSettingsUi"),mn=[{key:"python",label:"Python \u5B9F\u884C\u6848\u5185"},
{key:"gemini_code_execution",label:"Gemini \u30B3\u30FC\u30C9\u5B9F\u884C\uFF08\u30B5\u30F3\u30C9\u30DC\u30C3\u30AF\u30B9\u306E\u5B9F\u884C\u6642\u9593\u5236\u9650\uFF09"},
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
700 rounded p-2 text-xs text-gray-200";return mn.map(b=>{const x=b.mcpLocked===!0,S=x?'<div class="t\
ext-[10px] text-cyan-300/70 mt-1">\u3053\u306E\u9805\u76EE\u306E\u30AA\u30F3\u30FB\u30AA\u30D5\u306F\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059\uFF08\u30AA\u30D5\u6642\u306F\u6848\u5185\u6587\u306E\u6CE8\u5165\u3068\u30C4\u30FC\u30EB\u4ED8\u4E0E\u81EA\u4F53\u304C\u7121\u52B9\uFF09\u3002\u6587\u9762\u306F\u7DE8\u96C6\u3067\u304D\u307E\u3059\u3002\
</div>':"",L=x?`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow-500 \
w-3 h-3" disabled>`:`<input type="checkbox" id="${d}-auto-sys-${b.key}-enabled" class="accent-yellow\
-500 w-3 h-3">`;return`
                    <div class="rounded border border-gray-700 p-2 bg-gray-950/40">
                        <div class="flex items-center justify-between mb-1">
                            <div class="text-[11px] text-gray-300">${b.label}</div>
                            <label class="flex items-center gap-1 text-[10px] text-gray-500" ${x?'ti\
tle="\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u306EMCP\u30B9\u30A4\u30C3\u30C1\u306B\u9023\u52D5\u3057\u307E\u3059"':
""}>
                                ${L}
                                <span>\u9069\u7528</span>
                            </label>
                        </div>
                        <textarea id="${d}-auto-sys-${b.key}-text" class="${g}" placeholder="\u81EA\u52D5\u6CE8\u5165\u6587\u8A00"\
></textarea>
                        ${b.hint?`<div class="text-[10px] text-gray-500 mt-1">${b.hint}</div>`:""}
                        ${S}
                    </div>
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,m={})=>{mn.forEach(g=>{const b=m&&
typeof m=="object"?m[g.key]||{}:{},x=get(`${d}-auto-sys-${g.key}-enabled`),S=get(`${d}-auto-sys-${g.
key}-text`);x&&(g.mcpLocked===!0?x.disabled=!0:x.checked=b.enabled!==!1),S&&(S.value=b.text||"",S.placeholder=
b.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const qn=o((d,m=null)=>{if(m){const g=get(m);g&&(g.checked=!0)}mn.forEach(g=>{const b=get(`${d}-auto\
-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);if(b&&(g.mcpLocked!==!0?b.checked=!0:b.
disabled=!0),x){const S=x.placeholder||"";x.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),Gn=o(d=>{const m={};return mn.forEach(g=>{const b=get(`${d}\
-auto-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);m[g.key]={enabled:g.mcpLocked===!0?
!0:b?b.checked:!0,text:x?x.value:""}}),m},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
Gn,window.ensureAutoSystemPromptSettingsCard=()=>{const d=get("set-global-sys-prompt-enabled"),m=d?d.
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
                `,m.appendChild(g)},mi(),Hn(),I();const fi=o(()=>{const d=get("set-default-model");if(!d)
return;const m=d.value;d.innerHTML="",MODELS.forEach(b=>{const x=document.createElement("optgroup");
x.label=b.category,(b.items||[]).forEach(S=>{const L=document.createElement("option");L.value=S.id,L.
textContent=S.name,x.appendChild(L)}),d.appendChild(x)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||m||"gemini-3.6-flash";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g)},"po\
pulateDefaultModelOptions"),gi=o(()=>{const d=get("set-default-vision-model");if(!d)return;const m=d.
value;d.innerHTML="",MODELS.forEach(b=>{const x=(b.items||[]).filter(L=>{const A=(L.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(x.length===0)return;const S=document.createElement("optgroup");S.label=b.category,x.forEach(
L=>{const A=document.createElement("option");A.value=L.id,A.textContent=L.name+" \u2605",S.appendChild(
A)}),d.appendChild(S)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||m||
"gemini-3-flash-preview";g&&Array.from(d.options).some(b=>b.value===g)&&(d.value=g)},"populateDefaul\
tVisionModelOptions"),Un=o(d=>{if(!d)return;cacheUserSettings(d);const m=get("app-global-sys-prompt-\
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
const b=get("bot-status");b&&(d.is_bot_banned?(b.textContent=`BAN\u4E2D: ${d.bot_ban_reason||"Bot de\
tection"}`,b.classList.remove("hidden"),b.classList.add("text-red-400")):b.classList.add("hidden")),
d&&d.theme_color?(applyThemeColor(d.theme_color,!0),syncThemeInputs(d.theme_color)):syncThemeInputs(
localStorage.getItem(THEME_STORAGE_KEY)||INITIAL_THEME_COLOR||THEME_DEFAULT),snapshotSidebarHistory(
"settings-theme-synced"),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi(),get("set-usern\
ame")&&(get("set-username").value=d.username);const x=get("2fa-badge"),S=get("disable-2fa-btn");d.is_2fa_enabled?
(x.innerText="ENABLED",x.classList.replace("bg-gray-700","bg-green-600"),x.classList.replace("text-g\
ray-400","text-white"),S.classList.remove("hidden")):(x.innerText="DISABLED",x.classList.replace("bg\
-green-600","bg-gray-700"),x.classList.replace("text-white","text-gray-400"),S.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const L=get(
"set-passkey-only-login"),A=get("passkey-only-note"),$=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(q($),L){L.checked=!!d.passkey_only_login;const re=$.length>0||!!d.has_webauthn;L.disabled=!re,
re||(L.checked=!1),A&&(re?A.classList.add("hidden"):A.classList.remove("hidden"))}const H=get("mig-s\
tatus-box"),B=get("mig-progress-text"),W=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){H.classList.remove("hidden");const re=(d.migration_progress||"").split("/");if(re.length===
2){const F=parseInt(re[0]||"0",10),ae=parseInt(re[1]||"0",10);B&&(B.innerText=`${F} / ${ae}`),W&&ae>
0&&(W.style.width=`${Math.min(100,Math.floor(F/ae*100))}%`)}}else H.classList.add("hidden"),W&&(W.style.
width="0%"),B&&(B.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&Un(d);const m=get(
"search-box"),g=m?m.value:"";clearTimeout(searchTimeout);const b=get("settings-search");if(b&&(b.value=
""),filterSettings(),fi(),gi(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),le(),Hn(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),nt(!0),ht(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),gn(),ve(),ne();try{loadMcpServers()}catch{}};const Dt=o((d=!1)=>{snapshotSidebarHistory(
"settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),hi=o(()=>{const d=get("set-th\
eme-color"),m=get("set-theme-color-text"),g=get("theme-reset-btn"),b=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((S,L=!0)=>{const A=normalizeHex(S);A&&(applyThemeColor(A,L),syncThemeInputs(
A))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),m&&(m.addEventListener("chan\
ge",()=>{const S=normalizeHex(m.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(S,!0)}),m.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
m.blur())})),g&&(g.onclick=()=>x(THEME_DEFAULT,!0)),b.forEach(S=>{S.addEventListener("click",()=>x(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),bi=o(()=>{const d=get("reset-global-sys-prom\
pt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const m=get(
"reset-thread-sys-prompt");m&&(m.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{qn("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const b=get("reset-thread-auto-sys-prompt-defaults");b&&(b.onclick=()=>{qn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>Dt();const zn=get("settings-header-close");zn&&(zn.onclick=()=>Dt());
const Ht=get("settings-search");Ht&&(Ht.addEventListener("input",filterSettings),Ht.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const m=get("tab-"+activeSettingsTab);if(!m)return;const g=m.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const Wn=get(
"settings-search-clear");Wn&&Wn.addEventListener("click",()=>{Ht&&(Ht.value="",filterSettings(),Ht.focus())}),
hi(),bi(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Sn=get("set-gemini-local-python-dialog");Sn&&(Sn.onchange=()=>setGeminiLocalPyDialogEnabled(Sn.
checked));const Vn=get("set-gemini-backend");Vn&&(Vn.onchange=()=>syncGeminiBackendUi());const Kn=get(
"set-admin-api-key-mode");Kn&&(Kn.onchange=()=>syncAdminApiKeyModeUi());const Ln=get("set-temp-chat-\
timeout-seconds");Ln&&(Ln.onchange=()=>{applyTemporaryChatTimeoutSeconds(Ln.value)});const Jn=get("s\
lash-command-cancel-btn");Jn&&(Jn.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const d=get("set-username"),m=get("set-password"),g=readPromptBarModeFromForm(),b={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:Gn(
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
"set-latency-metrics")?get("set-latency-metrics").checked:!1,enable_client_debug_log:get("set-client\
-debug-log")?get("set-client-debug-log").checked:!1,passkey_only_login:get("set-passkey-only-login")?
get("set-passkey-only-login").checked:!1,skip_2fa_on_google_login:get("set-skip-2fa-google")?get("se\
t-skip-2fa-google").checked:!1,default_2fa_method:get("set-default-2fa-method")?get("set-default-2fa\
-method").value:"totp",new_username:d?d.value:null,new_password:m?m.value:null},x=get("set-e2ee")?get(
"set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
"enable_e2ee")?!!userSettingsSnapshot.enable_e2ee:!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.enableE2EE);
x!==S&&(b.enable_e2ee=x),get("set-openai")&&(b.openai_key=get("set-openai").value),get("set-gemini")&&
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
lobal").checked);const L=await apiFetch(CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Con\
tent-Type":"application/json"},body:JSON.stringify(b)});if(L.ok){let A="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const W=await L.json();W&&W.message&&(A=W.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!b.light_mode_enabled,liquid_glass_enabled:!!b.liquid_glass_enabled})),window.
applySavedUserSystemPromptSettings({system_prompt:b.system_prompt,system_prompt_enabled:b.system_prompt_enabled,
apply_global_system_prompt:b.apply_global_system_prompt,apply_auto_system_prompt_notices:b.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:b.auto_system_prompt_notices_config}),Dt();const $=currentUsername,
H=CHAT_CONFIG.enableE2EE;enterToSend=b.enter_to_send,autoSearchOnLinks=b.auto_search_on_links;const B=useSwCache;
useSwCache=b.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!b.clear_cache_on_version_update),
compactPromptMode=b.compact_prompt_mode,minimalPromptMode=b.minimal_prompt_mode,voiceStudioUiEnabled=
b.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=b.temp_chat_timeout_seconds,applyThemeColor(b.theme_color,
!0),syncThemeInputs(b.theme_color),applyLightMode(b.light_mode_enabled),applyLiquidGlassMode(b.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),B!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(A,"success"),syncClientDebugLogToggle(
b.enable_client_debug_log,"settings saved"),b.new_username&&b.new_username!==$?setTimeout(()=>location.
reload(),1e3):b.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
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
son"},body:JSON.stringify({username:m,mode:"single"})}),b=await g.json(),x=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")x&&(x.textContent=`${m} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),m=d?d.value.trim():"";if(!m){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:m,mode:"linked"})}),b=await g.json(),x=get("bot-unban-result");if(g.
ok&&b&&b.status==="ok")x&&(x.textContent=`${m} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=b&&b.error?b.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),m=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),m&&(m.classList.remove("hidden"),m.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(K=>{m&&(m.textContent=K)},"setBox"),b=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((K,ce)=>!K||!ce||ce<=0?0:K*8/(ce/1e3)/1e3/1e3,"toMbps"),S=o(K=>Number.isFinite(
K)?`${K.toFixed(0)} ms`:"-","fmtMs"),L=o(K=>Number.isFinite(K)?`${K.toFixed(K>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=o(async(K,ce)=>{const xe=await K.json().catch(()=>({}));return xe&&xe.error?xe.error:ce},
"parseErr"),$=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let K=0;K<4;K++){const ce=performance.now(),xe=await apiFetch(
`/api/speedtest/ping?_=${b()}`,{cache:"no-store"}),se=performance.now();if(!xe.ok)throw new Error(await A(
xe,"ping_failed"));await xe.json().catch(()=>({})),$.push(se-ce)}const H=$.reduce((K,ce)=>K+ce,0)/Math.
max(1,$.length),B=Math.min(...$),W=o(async K=>{const ce=performance.now(),xe=await apiFetch(`/api/sp\
eedtest/download?bytes=${K}&_=${b()}`,{cache:"no-store"});if(!xe.ok)throw new Error(await A(xe,"down\
load_failed"));const se=await xe.arrayBuffer(),fe=performance.now();return{bytes:se.byteLength||K,ms:fe-
ce,mbps:x(se.byteLength||K,fe-ce)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${S(H)}
\u6E2C\u5B9A\u4E2D... download`);const ue=[];for(const K of[2*1024*1024,8*1024*1024])ue.push(await W(
K)),g(`\u6E2C\u5B9A\u4E2D... ping ${S(H)}
download ${L(Math.max(...ue.map(ce=>ce.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const re=Math.max(...ue.map(K=>K.mbps)),F=o(async K=>{const ce=new Uint8Array(
K),xe=performance.now(),se=await apiFetch(`/api/speedtest/upload?_=${b()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ce,cache:"no-store"}),fe=performance.now();if(!se.ok)
throw new Error(await A(se,"upload_failed"));const Ee=await se.json().catch(()=>({})),$e=Number(Ee.bytes_received||
K)||K;return{bytes:$e,ms:fe-xe,mbps:x($e,fe-xe),serverMs:Number(Ee.server_elapsed_ms||0)||0}},"runUp\
load"),ae=[];for(const K of[1*1024*1024,4*1024*1024])ae.push(await F(K));const V=Math.max(...ae.map(
K=>K.mbps)),Ae=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(H)} / ${S(B)}`,`Download (best): ${L(re)}`,`Upload (best): ${L(V)}`,`Download r\
uns: ${ue.map(K=>`${Math.round(K.bytes/1024/1024)}MB=${L(K.mbps)}`).join(", ")}`,`Upload runs: ${ae.
map(K=>`${Math.round(K.bytes/1024/1024)}MB=${L(K.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(Ae.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){m&&(m.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>ht()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>St()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute("data-id");
if(m.classList.contains("ban-appeal-mark")){g&&await St([Number(g)]);return}if(m.classList.contains(
"ban-appeal-status")){const b=m.getAttribute("data-status");g&&b&&await _n({id:Number(g),status:b});
return}if(m.classList.contains("ban-appeal-reply-send")){const b=m.closest("[data-appeal-id]"),x=b?b.
querySelector(".ban-appeal-reply"):null,S=x?x.value:"";g&&await _n({id:Number(g),admin_reply:S});return}
if(m.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const b=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await _n({id:Number(
g),block_user:!0,block_reason:b});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
()=>closeUploadModal()),get("upload-select-btn")&&(get("upload-select-btn").onclick=()=>get("file-in\
put").click()),get("upload-camera-btn")&&(get("upload-camera-btn").onclick=()=>openCameraCaptureModal()),
get("upload-photo-btn")&&(get("upload-photo-btn").onclick=()=>get("photo-input").click()),get("camer\
a-modal-close")&&(get("camera-modal-close").onclick=()=>closeCameraCaptureModal()),get("camera-captu\
re-btn")&&(get("camera-capture-btn").onclick=()=>captureCameraShot()),get("camera-attach-btn")&&(get(
"camera-attach-btn").onclick=()=>attachCameraCapturedFiles()),get("camera-switch-btn")&&(get("camera\
-switch-btn").onclick=()=>toggleCameraCaptureFacing()),get("camera-clear-btn")&&(get("camera-clear-b\
tn").onclick=()=>resetCameraCapturePending()),get("camera-fallback-btn")&&(get("camera-fallback-btn").
onclick=()=>{closeCameraCaptureModal();const d=get("photo-input");d&&d.click()}),get("upload-clear-b\
tn")&&(get("upload-clear-btn").onclick=()=>{resetUploadState()}),get("marker-modal-close")&&(get("ma\
rker-modal-close").onclick=()=>{closeMarkerModal(),markerState.row=null}),get("marker-tool-draw")&&(get(
"marker-tool-draw").onclick=()=>setMarkerMode("draw")),get("marker-tool-mosaic")&&(get("marker-tool-\
mosaic").onclick=()=>setMarkerMode("mosaic")),get("marker-tool-crop")&&(get("marker-tool-crop").onclick=
()=>setMarkerMode("crop"));const Tn=get("marker-color-picker");Tn&&(Tn.oninput=d=>setMarkerColor(d.target.
value),Tn.onchange=d=>setMarkerColor(d.target.value));const Cn=get("marker-opacity");Cn&&(Cn.oninput=
d=>setMarkerOpacity(d.target.value),Cn.onchange=d=>setMarkerOpacity(d.target.value));const fn=get("m\
arker-opacity-number");fn&&(fn.onchange=d=>setMarkerOpacity(d.target.value),fn.onblur=d=>setMarkerOpacity(
d.target.value),fn.onkeydown=d=>{d.key==="Enter"&&(setMarkerOpacity(d.target.value),d.target.blur())}),
document.querySelectorAll("#marker-toolbar .marker-color-chip[data-marker-color]").forEach(d=>{d.onclick=
()=>setMarkerColor(d.getAttribute("data-marker-color"))}),get("marker-view-reset")&&(get("marker-vie\
w-reset").onclick=()=>resetMarkerTransform()),get("marker-crop-reset")&&(get("marker-crop-reset").onclick=
()=>clearCropRect()),get("marker-undo")&&(get("marker-undo").onclick=()=>undoMarkerCanvas()),get("ma\
rker-clear")&&(get("marker-clear").onclick=()=>clearMarkerCanvas()),get("marker-save")&&(get("marker\
-save").onclick=()=>saveMarkerToRow()),syncMarkerColorControls(),initMarkerCanvas(),initCropCanvas(),
window.addEventListener("resize",()=>{const d=get("marker-modal");!d||d.classList.contains("hidden")||
(applyMarkerTransform(),renderCropOverlay())});const yi=o(()=>{const d=get("upload-modal");return!!(d&&
!d.classList.contains("hidden"))},"isUploadModalOpen"),qt=get("drop-overlay");let Qt=0;const vi=o(()=>{
yi()||qt&&(qt.classList.remove("hidden"),qt.classList.add("flex"))},"showDropOverlay"),Zt=o(()=>{Qt=
0,qt&&(qt.classList.add("hidden"),qt.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
Zt;const kt=get("upload-dropzone");kt&&(kt.addEventListener("dragover",d=>{d.preventDefault(),kt.classList.
add("dragover")}),kt.addEventListener("dragleave",()=>{kt.classList.remove("dragover")}),kt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),kt.classList.remove("dragover"),Zt();const m=d.dataTransfer?
d.dataTransfer.files:null;m&&m.length&&handleFiles(m)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(Qt+=1,vi())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(Qt=Math.max(0,Qt-1),(Qt===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&Zt())}),window.addEventListener(
"dragend",()=>{Zt()}),window.addEventListener("drop",d=>{Zt(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(kt&&kt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const Xn=get("bot-admin-modal"),wi=o(d=>{const m=get("bot-admin-list");if(m){if(m.innerHTML=
"",!d||!d.length){m.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,b)=>{const x=!!g.is_bot_banned,S=g.bot_detection_enabled!==!1,L=Number(g.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(b,12)*.02}\
s`,A.innerHTML=`
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
                    `,m.appendChild(A)})}},"renderBotUsers"),en=o(async(d="")=>{const m=get("bot-adm\
in-list");m&&(m.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),b=await g.json();g.ok&&b&&b.users?wi(b.users):(m&&(m.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{m&&(m.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Mn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Xn))return;const m=get(
"settings-modal");m&&(m.classList.contains("modal-open")||m.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await en(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Mn;const An=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=An,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{An()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||Xn)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Mn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await en(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await en("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async d=>{d.key==="Enter"&&await en(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async d=>{const m=d.target.closest("button");if(!m)return;const g=m.getAttribute(
"data-user");if(!g)return;let b;if(m.classList.contains("bot-toggle-detect")){const x=m.getAttribute(
"data-enabled")!=="1";b=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:g,action:"toggle_detection",enabled:x})})}else if(m.classList.
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
let x={};try{x=await b.json()}catch{}showToast(x.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
x.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await en(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const Gt={"/settings":{id:"settings-mod\
al",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:o(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:o(()=>{li(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
pen")},"/history":{id:"history-modal",open:o(()=>window.showHistoryModal(),"open")},"/branch":{id:"b\
ranch-modal",open:o(()=>window.showBranchModal(),"open")},"/batch":{id:"batch-modal",open:o(()=>window.
showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:o(()=>openRichPasteModal(),"open")},"\
/camera":{id:"camera-capture-modal",open:o(()=>openCameraCaptureModal(),"open")},"/edit-image":{id:"\
marker-modal",open:o(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:o(()=>window.openThreadModal(),
"open")},"/model":{id:"model-modal",open:o(()=>openModelModal(),"open")},"/token-details":{id:"token\
-detail-modal",open:o(()=>showTokenDetailModal(),"open")},"/encryption-status":{id:"encryption-statu\
s-modal",open:o(()=>showEncryptionStatusModal(),"open")},"/python-execution":{id:"python-exec-modal",
open:o(()=>showPythonExecDetailModal(),"open")},"/gem":{id:"gem-modal",open:o(()=>{editingGemUuid=null,
get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',showModal(
"gem-modal")},"open")},"/compression":{id:"compression-modal",open:o(()=>window.openCompressionModal(),
"open")},"/admin-bots":{id:"bot-admin-modal",open:o(()=>Mn(),"open")},"/admin-bot-logs":{id:"bot-log\
-modal",open:o(()=>An(),"open")}},Yn=o((d,m=!1)=>{switch(d){case"settings-modal":Dt(m);break;case"up\
load-modal":closeUploadModal(m);break;case"camera-capture-modal":closeCameraCaptureModal(m?{skipHistory:!0}:
{});break;case"history-modal":window.closeHistoryModal&&window.closeHistoryModal(m);break;case"lib-m\
odal":window.closeLibModal&&window.closeLibModal(m);break;case"branch-modal":window.closeBranchModal&&
window.closeBranchModal(m);break;case"batch-modal":window.closeBatchModal&&window.closeBatchModal(m);
break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(m);break;case"ma\
rker-modal":window.closeMarkerModal&&window.closeMarkerModal(m);break;case"thread-modal":window.closeThreadModal&&
window.closeThreadModal(m);break;case"model-modal":window.closeModelModal&&window.closeModelModal(m);
break;case"token-detail-modal":closeTokenDetail(m);break;case"encryption-status-modal":closeEncryptionModal(
m);break;case"python-exec-modal":closePythonExecDetail(m);break;case"gem-modal":window.closeGemModal&&
window.closeGemModal(m);break;case"compression-modal":window.closeCompressionModal&&window.closeCompressionModal(
m);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(m);break;case"bo\
t-log-modal":window.closeBotLogModal&&window.closeBotLogModal(m);break;case"mcp-decision-modal":typeof submitMcpDecision==
"function"?submitMcpDecision("deny"):hideModal(d);break;case"api-key-required-modal":{const b=get("a\
pi-key-modal-cancel-btn");b&&typeof b.onclick=="function"?b.click():hideModal(d);break}case"lyria-st\
udio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(d);break;case"voice-studio-m\
odal":window.VoiceStudio?window.VoiceStudio.close():hideModal(d);break;case"version-update-modal":const g=localStorage.
getItem("app_version")||"";g&&localStorage.setItem("version_notified",g),hideModal(d);break;default:
hideModal(d);break}},"closeModalById");window.addEventListener("popstate",d=>{let m=!1;Object.values(
Gt).forEach(x=>{const S=get(x.id);S&&S.classList.contains("modal-open")&&location.pathname!==Object.
keys(Gt).find(L=>Gt[L].id===x.id)&&(Yn(x.id,!0),m=!0)});const g=location.pathname.match(/^\/c\/(.+)$/);
if(g){const x=decodeURIComponent(g[1]);String(currentThreadId)!==String(x)&&loadMessages(x,{skipHistory:!0})}else
location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});const b=Gt[location.pathname];
if(b){const x=get(b.id);x&&!x.classList.contains("modal-open")&&b.open()}});const Qn=location.pathname;
Gt[Qn]&&(history.replaceState({},"","/"),setTimeout(()=>Gt[Qn].open(),500)),get("easy-login-generate")&&
(get("easy-login-generate").onclick=async()=>{const d=get("easy-login-mins"),m=d?parseInt(d.value||"\
5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${m}\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))
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
"error",!0)});const xi=4,ki=5*1024*1024;let Lt=[];const En=o(()=>{const d=get("fb-images-list");d&&(d.
innerHTML="",Lt.forEach((m,g)=>{const b=document.createElement("div");b.className="relative w-16 h-1\
6 rounded border border-gray-600 overflow-hidden";const x=document.createElement("img");x.src=m.url,
x.alt=m.file.name,x.className="w-full h-full object-cover";const S=document.createElement("button");
S.type="button",S.className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none \
px-1 py-0.5",S.textContent="\xD7",S.onclick=()=>{URL.revokeObjectURL(m.url),Lt.splice(g,1),En()},b.appendChild(
x),b.appendChild(S),d.appendChild(b)}))},"renderFbImages"),_i=o(()=>{Lt.forEach(d=>URL.revokeObjectURL(
d.url)),Lt=[],En()},"clearFbImages");get("fb-images-add")&&get("fb-images-input")&&(get("fb-images-a\
dd").onclick=()=>get("fb-images-input").click(),get("fb-images-input").onchange=d=>{for(const m of Array.
from(d.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(m.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(m.
size>ki){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(Lt.
length>=xi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}Lt.push({file:m,url:URL.createObjectURL(m)})}d.target.value="",En()}),get("fb-subm\
it").onclick=async()=>{const d=get("fb-title").value.trim(),m=get("fb-message").value.trim();if(!m){
showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:m,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},b=window.ActivityLog?window.ActivityLog.feedbackPayload():null;b&&(g.client_logs=b);
const x=get("fb-attach-chat"),S=!!(x&&x.checked);if(S&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\
\u3066\u3044\u307E\u305B\u3093","error",!0);return}S&&(g.chat_copy=window.ActivityLog?window.ActivityLog.
chatCopyPayload(currentThreadId):{client:"web",thread_id:String(currentThreadId)});const L=await apiFetch(
"/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),
A=L.ok?await L.clone().json().catch(()=>({})):{},$={count:Lt.length,saved:!0};if(L.ok&&$.count)try{const H=new FormData;
Lt.forEach(W=>H.append("images",W.file,W.file.name));const B=await apiFetch(`/api/feedback/${A.public_id}\
/images`,{method:"POST",body:H});$.saved=B.ok}catch{$.saved=!1}(window.ActivityLog?await window.ActivityLog.
reportFeedback(L,b,S,$):L.ok)&&(get("fb-title").value="",get("fb-message").value="",_i(),x&&(x.checked=
!1),gn())};async function Zn(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),gn()}o(Zn,"deleteFeedback");async function gn(){const m=await(await apiFetch("/api/feedba\
ck?all=1")).json(),g=get("fb-list");g.innerHTML="",(m.items||[]).filter(S=>!m.is_admin||S.user_id===
void 0||S.user_id===null||!0).forEach(S=>{if(m.is_admin)return;const L=document.createElement("div");
L.className="p-2 rounded border border-gray-700 bg-gray-800/50",L.innerHTML=`<div class="text-[11px]\
 text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(S.public_id||"")}</span> / ${S.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(S.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.
image_count}\u679A</div>`:""}${S.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(
S.admin_reply)}</div>`:""}<div class="mt-2"><button type="button" class="fb-delete bg-red-700 hover:\
bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,L.querySelector(".\
fb-delete").onclick=()=>Zn(S.public_id),g.appendChild(L)});const b=get("fb-admin-panel"),x=get("fb-a\
dmin-list");m.is_admin?(b.classList.remove("hidden"),x.innerHTML="",(m.items||[]).forEach(S=>{const L=document.
createElement("div");L.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",L.innerHTML=
`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(S.public_id||"")}</span> / user:${S.user_id} / ${S.created_at}</div>
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
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3\
 py-1 rounded text-xs">\u4FDD\u5B58</button>
                                <button type="button" class="fb-delete bg-red-700 hover:bg-red-600 t\
ext-white px-3 py-1 rounded text-xs">\u524A\u9664</button>
                            </div>
                            <textarea class="fb-reply w-full bg-gray-900 border border-gray-700 roun\
ded px-2 py-1 text-xs text-white" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(S.admin_reply||
"")}</textarea>
                        `,L.querySelector(".fb-status").value=S.status||"new",L.querySelector(".fb-s\
ave").onclick=async()=>{const A=L.querySelector(".fb-status").value,$=L.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${S.public_id}/update`,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({status:A,admin_reply:$})}),gn()},L.querySelector(".fb-delete").onclick=
()=>Zn(S.public_id),x.appendChild(L)})):b.classList.add("hidden")}if(o(gn,"loadFeedback"),window.setupTOTP=
async()=>{const m=await(await apiFetch("/api/2fa/totp/setup",{method:"POST"})).json();get("totp-qr").
src=m.qr_image,get("totp-secret-disp").innerText=m.secret,get("totp-setup-area").classList.remove("h\
idden")},window.enableTOTP=async()=>{const d=get("totp-verify-code").value;if(!d)return;(await apiFetch(
"/api/2fa/totp/enable",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F","success"),get(
"totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):showToast(
"\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","error",!0)},
window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),m=get("webauthn-name"),g=m?String(
m.value||"").trim():"";try{d&&(d.disabled=!0);const b=await apiFetch("/api/2fa/webauthn/register/opt\
ions",{method:"POST"}),x=await b.json();if(!b.ok){showToast(x.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const L=await(await ensureWebAuthnJson()).create({publicKey:x}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},L,{name:g}))}),$=await A.json().catch(()=>({}));A.ok?(m&&(m.value=""),showToast(
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
return}if(d.key==="Enter"){d.preventDefault();const b=g.substring(1).toLowerCase(),x=loadedGems.filter(
S=>S.name.toLowerCase().includes(b)||S.description&&S.description.toLowerCase().includes(b));x[gemSelectedIndex]?
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
"sts-cancel-btn").onclick=()=>{isStsModel()&&Si()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async d=>{const m=(d.clipboardData||window.clipboardData).items,g=[];for(let b=0;b<m.length;b++)
if(m[b].kind==="file"){const x=m[b].getAsFile();x&&g.push(x)}g.length>0&&(d.preventDefault(),await handleFiles(
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
"";m&&g&&(d.preventDefault(),openImageViewer(g))});const tn=document.querySelector(".viewer-content");
tn&&(tn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),tn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),tn.addEventListener("touchend",onViewerTouchEnd),tn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",d=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(d.target.id==="image-viewer"||d.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",d=>{d.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",d=>{d.key==="Escape"&&closeImageViewer()});
let it,Oe=null,hn=[],$n=!1,bn=null,Ut=null,$t=null,nn=null,In=0,yn=!1,sn=null,zt=null,Tt=null,an=null,
vn=null,It=null,wn=null,xn=null;function ei(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
wn)&&wn.length)return wn;d.innerHTML="";const m=[];for(let g=0;g<24;g++){const b=document.createElement(
"span");b.className="block rounded-full",b.style.background="rgba(252, 165, 165, 0.92)",b.style.width=
"2px",b.style.transition="height 75ms linear, opacity 75ms linear",b.style.height="2px",b.style.opacity=
"0.4",m.push(b),d.appendChild(b)}return wn=m,m}o(ei,"ensureMicWaveformBars");function Pt(d,m="hidden"){
const g=get("mic-recording-indicator"),b=get("mic-recording-text");if(g){if(xn&&(clearTimeout(xn),xn=
null),m==="hidden"){g.classList.add("hidden");return}b&&d&&(b.innerText=d),g.classList.remove("hidde\
n"),m==="recording"?g.style.color="rgb(252 165 165)":m==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(Pt,"setMicRecordingIndicator");function ti(){ei().forEach(m=>{m.
style.height="2px",m.style.opacity="0.35"})}o(ti,"resetMicWaveformBars");function Ot(){if(vn&&(cancelAnimationFrame(
vn),vn=null),an){try{an.disconnect()}catch{}an=null}if(zt){try{zt.close()}catch{}zt=null}Tt=null,It=
null,ti()}o(Ot,"stopMicWaveform");function ni(d){Ot();const m=ei();if(!m.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{zt=new g,Tt=zt.createAnalyser(),Tt.fftSize=
256,Tt.smoothingTimeConstant=0,an=zt.createMediaStreamSource(d),an.connect(Tt),It=new Uint8Array(Tt.
frequencyBinCount)}catch{Ot();return}const b=o(()=>{if(!Tt||!It)return;Tt.getByteFrequencyData(It);const x=Math.
max(1,Math.floor(It.length/m.length));for(let S=0;S<m.length;S++){const A=(It[Math.min(It.length-1,S*
x)]||0)/255,$=Math.max(2,Math.round(2+A*10));m[S].style.height=`${$}px`,m[S].style.opacity=`${.35+A*
.65}`}vn=requestAnimationFrame(b)},"render");b()}o(ni,"startMicWaveform");function kn(){if(bn&&(clearInterval(
bn),bn=null),nn){try{nn.disconnect()}catch{}nn=null}if(Ut){try{Ut.close()}catch{}Ut=null}$t=null}o(kn,
"stopSilenceMonitor");function ii(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;kn();const m=window.
AudioContext||window.webkitAudioContext;if(!m)return;Ut=new m,$t=Ut.createAnalyser(),$t.fftSize=2048,
nn=Ut.createMediaStreamSource(d),nn.connect($t);const g=new Uint8Array($t.fftSize),b=getStsSilenceMs(),
x=.02;In=0,yn=!1,bn=setInterval(()=>{if(!$t)return;$t.getByteTimeDomainData(g);let S=0;for(let A=0;A<
g.length;A++){const $=(g[A]-128)/128;S+=$*$}if(Math.sqrt(S/g.length)>x){yn||(yn=!0),In=Date.now();return}
yn&&Date.now()-In>b&&it&&it.state==="recording"&&it.stop()},200)}o(ii,"startSilenceMonitor");const Rn=class Rn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(m,g,b,x={}){this.model=
b,this.ws=new WebSocket(`${g}?access_token=${m}`),this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{
console.log("Gemini Live WebSocket opened. Sending setup...");const $=!!(x&&x.transcriptionConfig),H={
setup:{model:`models/${b}`,generationConfig:{responseModalities:$?["TEXT"]:["AUDIO"]},inputAudioTranscription:$?
x.transcriptionConfig||{}:{}}};$||(H.setup.outputAudioTranscription={}),x.speechConfig&&(H.setup.generationConfig.
speechConfig=x.speechConfig),x.thinkingConfig&&(H.setup.generationConfig.thinkingConfig=x.thinkingConfig),
x.translationConfig&&(H.setup.generationConfig.translationConfig=x.translationConfig),console.log("S\
ending setup:",JSON.stringify(H)),this.ws.send(JSON.stringify(H))},this.ws.onmessage=$=>this._handleMessage(
$),this.ws.onerror=$=>{console.error("Gemini Live WebSocket error:",$),this.onError&&this.onError($)},
this.ws.onclose=$=>{console.log("Gemini Live WebSocket closed:",$.code,$.reason),this.closedEvent=$,
this.onClose&&this.onClose($)},this.stream=await navigator.mediaDevices.getUserMedia(Nn());const S=si(
this.stream,16e3);this.audioContext=S.ctx;const L=S.source;this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=$=>{$.data.
size>0&&this.userAudioChunks.push($.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
$=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const H=ai($.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!H||!H.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(H))),mimeType:"audio/\
pcm;rate=16000"}}}))},L.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(m){
let g=null;try{g=JSON.parse(typeof m.data=="string"?m.data:new TextDecoder().decode(m.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const b=g.serverContent;b.modelTurn&&b.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log(
"Gemini thought delta:",x.text),this.assistantThought+=x.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(x.text)),x.inlineData&&x.inlineData.data){const S=x.inlineData.data;
console.log("Gemini audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);
const L=atob(S),A=new Uint8Array(L.length);for(let $=0;$<L.length;$++)A[$]=L.charCodeAt($);this.assistantAudioChunks.
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
`),this.assistantTurnBreak=!1,this.assistantText+=m)}stop(){this.ws&&this.ws.close(),this.processor&&
this.processor.disconnect(),this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().
forEach(m=>m.stop()),this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const m=new Blob(
this.assistantAudioChunks),g=await this._blobToBase64(m),b=new Blob(this.userAudioChunks),x=await this.
_blobToBase64(b);return{user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.
assistantThought,audio_base64:g,user_audio_base64:x}}_blobToBase64(m){return new Promise(g=>{const b=new FileReader;
b.onloadend=()=>g(b.result.split(",")[1]),b.readAsDataURL(m)})}};o(Rn,"GeminiLiveClient");let Pn=Rn;
const Bn=class Bn{constructor(m=24e3){const g=window.AudioContext||window.webkitAudioContext;this.ctx=
new g({sampleRate:m}),this.nextStartTime=0,this.bufferDelay=.1,this.started=!1}async addChunk(m){if(!this.
ctx)return;const g=atob(m),b=new Uint8Array(g.length);for(let H=0;H<g.length;H++)b[H]=g.charCodeAt(H);
const x=new Int16Array(b.buffer),S=new Float32Array(x.length);for(let H=0;H<x.length;H++)S[H]=x[H]/32768;
const L=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);L.getChannelData(0).set(S),this.ctx.state===
"suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();A.buffer=L,A.connect(this.
ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.bufferDelay,this.started=
!0);const $=Math.max(this.ctx.currentTime,this.nextStartTime);A.start($),this.nextStartTime=$+L.duration}stop(){
this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Bn,"RealTimeAudioPlayer");let on=Bn;const Fn=class Fn{constructor(){
this.active=!1,this.capturing=!1,this.sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=
null,this.processor=null,this.stream=null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.
userTranscript="",this.assistantTranscript="",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=
0,this.lastAudioAt=0,this.streamError=null,this.saved=!1,this.saving=!1,this.stopping=!1,this.audioQueue=
[],this.audioFlush=null}isActive(){return this.active}async start(){if(this.active)return;if(this.saving||
this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const m=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const x=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=x.id!==null&&x.id!==void 0?String(x.id):x.id,setTemporaryChatUiState(!!(x&&x.
is_temporary)),setCurrentChatHeaderTitle(x&&x.title),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+x.id),get("welcome-screen").classList.add("hidden")}catch(b){showToast(
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
g)}),x=await b.json().catch(()=>({}));if(!b.ok)throw new Error(x.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
this.sessionId=x.session_id,this.rateIn=x.rate_in||this.rateIn,this.rateOut=x.rate_out||this.rateOut,
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
value:x}=await this.reader.read();if(b)break;g+=m.decode(x,{stream:!0});let S;for(;(S=g.indexOf(`

`))>=0;){const L=g.slice(0,S);g=g.slice(S+2);for(const A of L.split(`
`)){if(!A.startsWith("data: "))continue;let $=null;try{$=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent($)}}}}catch(b){if(b&&b.name==="AbortError")return;this.active&&(this.streamError=b&&b.message?
b.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(m){
if(m)switch(m.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new on(this.rateOut||24e3),Wt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Wt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(Nn()).then(m=>{this.stream=m;const g=this.rateIn||24e3,b=si(m,g);this.audioCtx=
b.ctx;const x=b.source,S=this.audioCtx.sampleRate,L=4096;this.processor=this.audioCtx.createScriptProcessor(
L,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const $=A.inputBuffer.
getChannelData(0),H=ai($,S,g);!H||!H.byteLength||this._sendAudio(H)},x.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(m){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(m)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const m=this.audioQueue.splice(0,this.audioQueue.length),
g=m.reduce(($,H)=>$+H.byteLength,0),b=new Uint8Array(g);let x=0;m.forEach($=>{b.set($,x),x+=$.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),L={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:b.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(L):L;try{await fetch(S,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(m=>m.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}kn(),Ot()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const m=Date.now(),g=this.responseDoneCount;let b=this.lastAudioAt;for(;Date.now()-m<2e4&&!(this.responseDoneCount>
g||(this.lastAudioAt>b&&(b=this.lastAudioAt),!this.speechActive&&Date.now()-m>2e3&&Date.now()-b>2500));)
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
Fn,"RealtimeVoiceSession");let On=Fn;function si(d,m){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let b=null;try{return b=new g({sampleRate:m}),{ctx:b,
source:b.createMediaStreamSource(d)}}catch{if(b)try{b.close()}catch{}return b=new g,{ctx:b,source:b.
createMediaStreamSource(d)}}}o(si,"openMicAudioSource");function ai(d,m,g){let b=d;if(m!==g&&m>0&&g>
0){const S=m/g,L=Math.floor(b.length/S),A=new Float32Array(L);for(let $=0;$<L;$++)A[$]=b[Math.min(Math.
floor($*S),b.length-1)];b=A}const x=new Int16Array(b.length);for(let S=0;S<b.length;S++){const L=Math.
max(-1,Math.min(1,b[S]));x[S]=L<0?L*32768:L*32767}return x.buffer}o(ai,"pcm16FromFloat32");const rn=new On,
oi=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let b=null,x=null,S=!1,L=null,A=!1,$=0,H=0,B=null,W="idle",ue=!1,re=null;const F=o(R=>document.getElementById(
R),"$"),ae=o(R=>{const J=Object.assign({},R||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(J):(J.progressSpinner=!1,
J)},"noSpinner");function V(R,J){W=J;const _e=F("lyria-status-text"),de=F("lyria-status-dot");_e&&(_e.
textContent=R),de&&(de.className="w-2 h-2 rounded-full inline-block "+(g[J]||g.idle)),xe(),se()}o(V,
"setStatus");function Ae(){const R=H?Math.floor((Date.now()-H)/1e3):0,J=String(Math.floor(R/60)).padStart(
2,"0"),_e=String(R%60).padStart(2,"0");return`${J}:${_e}`}o(Ae,"formatElapsed");function K(){H||(H=Date.
now());const R=F("lyria-elapsed");R&&(R.textContent=Ae()),B||(B=window.setInterval(()=>{const J=F("l\
yria-elapsed");J&&(J.textContent=Ae())},1e3))}o(K,"startElapsedTimer");function ce(){B&&(window.clearInterval(
B),B=null)}o(ce,"stopElapsedTimer");function xe(){const R=F("lyria-play-btn"),J=F("lyria-pause-btn"),
_e=F("lyria-stop-btn"),de=F("lyria-reset-btn"),Ce=!!b,Fe=W==="streaming"||W==="connecting";if(R){R.disabled=
ue;const Qe=R.querySelector("i");Qe&&(Qe.className="fas fa-play")}J&&(J.disabled=ue||!Fe),_e&&(_e.disabled=
ue||!Ce||!Fe),de&&(de.disabled=ue||!Ce||!Fe)}o(xe,"updateTransportButtons");function se(){const R=F(
"lyria-save-btn");if(!R)return;const J=!!b&&W!=="idle"&&W!=="connecting"&&W!=="error";R.classList.toggle(
"hidden",!J)}o(se,"updateSaveButton");function fe(R,J){const _e=F("lyria-prompt-rows");if(!_e)return;
const de=document.createElement("div");de.className="flex items-center gap-2",de.innerHTML=`
                        <input type="text" value="${escapeHtml(R||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof J=="num\
ber"?J:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof J=="number"?J:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Ce=de.querySelector('input[type="range"]'),Fe=de.querySelector(".lyria-w\
eight-label");Ce&&Fe&&Ce.addEventListener("input",()=>{Fe.textContent=parseFloat(Ce.value).toFixed(1)});
const Qe=de.querySelector(".lyria-prompt-remove");Qe&&Qe.addEventListener("click",()=>{_e.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||de.remove()}),de.classList.add("lyria-prompt-row-wrap"),_e.appendChild(
de)}o(fe,"addPromptRow");function Ee(){const R=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),J=[];return R.forEach(_e=>{const de=_e.querySelector('input[type="text"]'),Ce=_e.querySelector(
'input[type="range"]'),Fe=(de?de.value:"").trim();Fe&&J.push({text:Fe,weight:parseFloat(Ce?Ce.value:
1)||1})}),J}o(Ee,"collectPrompts");function $e(){const R={},J=o(Ti=>{const Dn=F(Ti);return Dn&&Dn.value!==
""?parseFloat(Dn.value):void 0},"num"),_e=J("lyria-bpm");_e!==void 0&&(R.bpm=Math.round(_e));const de=J(
"lyria-guidance");de!==void 0&&(R.guidance=de);const Ce=J("lyria-density");Ce!==void 0&&(R.density=Ce);
const Fe=J("lyria-brightness");Fe!==void 0&&(R.brightness=Fe);const Qe=J("lyria-temperature");Qe!==void 0&&
(R.temperature=Qe);const Ge=F("lyria-scale");Ge&&Ge.value&&(R.scale=Ge.value);const et=F("lyria-mode");
et&&et.value&&(R.music_generation_mode=et.value);const pt=F("lyria-mute-bass"),Nt=F("lyria-mute-drum\
s"),pi=F("lyria-only-bass-drums");return pt&&(R.mute_bass=pt.checked),Nt&&(R.mute_drums=Nt.checked),
pi&&(R.only_bass_and_drums=pi.checked),R}o($e,"collectConfig");function st(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([J,_e])=>{
const de=F(J),Ce=F(_e);!de||!Ce||de.addEventListener("input",()=>{const Fe=parseFloat(de.value);Ce.textContent=
J==="lyria-bpm"?String(Math.round(Fe)):Fe.toFixed(1)})})}o(st,"bindRangeLabels");function Ue(){if(L){
try{L.close()}catch{}L=null}A=!1,$=0}o(Ue,"resetPlayback");function ut(){if(S=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(ut,"closeStream");async function Ct(){ut(),x=new AbortController,
S=!0;try{const R=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(b)}`,ae({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!R.ok){const Ce=await R.
json().catch(()=>({}));throw new Error(Ce.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const J=R.body.getReader(),_e=new TextDecoder;let de="";for(;S;){const{done:Ce,value:Fe}=await J.read();
if(Ce)break;de+=_e.decode(Fe,{stream:!0});const Qe=de.split(`

`);de=Qe.pop();for(const Ge of Qe){const et=Ge.split(`
`).find(Nt=>Nt.startsWith("data: "));if(!et)continue;const pt=et.slice(6);try{const Nt=JSON.parse(pt);
ke(Nt)}catch{}}}}catch(R){if(R&&R.name==="AbortError")return;S&&(V("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&b&&Ct()},1200))}finally{S=!1}}o(Ct,"openStream");function ke(R){
if(R&&R.snapshot){const J=R.status;if(J==="error"){V("\u30A8\u30E9\u30FC","error"),ce();return}if(J===
"closed"||J==="stopped"){V("\u7D42\u4E86","closed"),ce();return}V(J==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",J==="paused"?"paused":"connecting");return}if(R&&R.audio){V("\u518D\u751F\u4E2D...",
"streaming"),K(),ge(R.audio);return}if(R&&R.error){V("\u30A8\u30E9\u30FC: "+R.error,"error"),ce();return}
if(R&&R.final){V("\u7D42\u4E86","closed"),ce(),xe();return}}o(ke,"handleStreamMessage");function ge(R){
if(!R)return;if(!L){const Ge=window.AudioContext||window.webkitAudioContext;if(!Ge)return;L=new Ge({
sampleRate:48e3}),A=!1,$=0}let J;try{const Ge=atob(R);J=new Uint8Array(Ge.length);for(let et=0;et<Ge.
length;et++)J[et]=Ge.charCodeAt(et)}catch{return}const _e=new Int16Array(J.buffer),de=Math.floor(_e.
length/2);if(de<1)return;const Ce=L.createBuffer(2,de,48e3);for(let Ge=0;Ge<2;Ge++){const et=Ce.getChannelData(
Ge);for(let pt=0;pt<de;pt++)et[pt]=_e[pt*2+Ge]/32768}L.state==="suspended"&&L.resume();const Fe=L.createBufferSource();
Fe.buffer=Ce,Fe.connect(L.destination),A||($=L.currentTime+.08,A=!0);const Qe=Math.max(L.currentTime,
$);Fe.start(Qe),$=Qe+Ce.duration}o(ge,"playChunk");async function Be(R,J){const _e=await fetch("/api\
/gemini/music/command",ae({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:b,type:R},J||{}))})),de=await _e.json().catch(()=>({}));if(!_e.ok)throw new Error(
de.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return de}
o(Be,"apiCommand");async function Ye(){if(ue)return;const R=Ee();if(!R.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ue=!0,xe(),V("\u63A5\u7D9A\u4E2D...","connecting");try{
const J=await fetch("/api/gemini/music/start",ae({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:R,config:$e()})})),_e=await J.json().catch(()=>({}));if(!J.
ok)throw new Error(_e.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
b=_e.session_id,re=$e(),V("\u63A5\u7D9A\u4E2D...","connecting"),Ct()}catch(J){V("\u30A8\u30E9\u30FC: "+
J.message,"error"),showToast("Lyria RealTime: "+J.message,"error",!0)}finally{ue=!1,xe()}}o(Ye,"star\
tSession");async function at(R){if(b){ue=!0,xe();try{await Be("control",{action:R}),R==="PLAY"?V("\u518D\u751F\
\u4E2D...","streaming"):R==="PAUSE"?V("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):R==="STOP"?V("\u505C\u6B62\u4E2D",
"stopped"):R==="RESET_CONTEXT"&&V("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(J){showToast("Lyria RealTime: "+J.message,"error",!0),V("\u30A8\u30E9\u30FC: "+J.
message,"error")}finally{ue=!1,xe()}}}o(at,"control");async function be(){if(!b)return;const R=Ee();
if(!R.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ue=!0;try{await Be("prompts",{weighted_prompts:R}),V("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
W==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(J){showToast("Lyria RealTime: "+J.message,"error",!0)}finally{ue=!1,xe()}}o(be,"app\
lyPrompts");async function Pe(){if(!b)return;const R=$e(),J=re||{},_e=R.bpm!==void 0&&R.bpm!==J.bpm,
de=R.scale!==void 0&&R.scale!==J.scale,Ce=_e||de;ue=!0;try{await Be("config",{config:R,reset_context:Ce}),
re=R,V(Ce?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",W==="paused"?"paused":"streaming"),showToast(
Ce?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Fe){showToast("Lyria RealT\
ime: "+Fe.message,"error",!0)}finally{ue=!1,xe()}}o(Pe,"applyConfig");async function He(){if(b){ue=!0,
V("\u4FDD\u5B58\u4E2D...","connecting"),xe();try{const R=await fetch("/api/gemini/music/save",ae({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b,thread_id:currentThreadId||
null})})),J=await R.json().catch(()=>({}));if(!R.ok)throw new Error(J.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
V("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ce(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),J.thread_id&&(currentThreadId=String(J.thread_id),history.pushState({},"","/c/"+J.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(J.thread_id||currentThreadId),cn(!0)}catch(R){
V("\u30A8\u30E9\u30FC: "+R.message,"error"),showToast("Lyria RealTime: "+R.message,"error",!0)}finally{
ue=!1,xe()}}}o(He,"saveSession");async function ze(){if(ut(),b)try{await fetch("/api/gemini/music/ca\
ncel",ae({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:b})}))}catch{}
b=null,ce(),Ue(),V("\u6E96\u5099\u5B8C\u4E86","idle")}o(ze,"cancelSession");function ct(){const R=F(
"lyria-prompt-rows");R&&(R.innerHTML=""),fe("",1),re=null,H=0,["lyria-bpm","lyria-guidance","lyria-d\
ensity","lyria-brightness","lyria-temperature"].forEach(de=>{const Ce=F(de);Ce&&(Ce.value=de==="lyri\
a-bpm"?"120":de==="lyria-guidance"?"4":de==="lyria-temperature"?"1.1":"0.5")});const J=F("lyria-scal\
e");J&&(J.value="");const _e=F("lyria-mode");_e&&(_e.value="QUALITY"),["lyria-mute-bass","lyria-mute\
-drums","lyria-only-bass-drums"].forEach(de=>{const Ce=F(de);Ce&&(Ce.checked=!1)}),st()}o(ct,"resetC\
ontrols");function cn(R){ut(),b&&fetch("/api/gemini/music/cancel",ae({method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({session_id:b})})).catch(()=>{}),b=null,S=!1,ce(),Ue(),
hideModal("lyria-studio-modal")}o(cn,"closeAndCleanup");function Vt(R){if(!isLyriaRealtimeModel()){showToast(
"Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const J=F("lyria-studio-modal");if(J&&J.classList.contains("modal-open")&&b){if(R&&
typeof R=="string"){const de=F("lyria-prompt-rows");de&&(de.innerHTML=""),fe(R,1)}return}if(b&&ze(),
ct(),R&&typeof R=="string"){const de=F("lyria-prompt-rows");de&&(de.innerHTML=""),fe(R,1)}b=null,S=!1,
ce(),Ue(),V("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Vt,"open");function dn(){
const R=F("lyria-open-studio-btn");R&&R.addEventListener("click",()=>Vt(""));const J=F("lyria-studio\
-close");J&&J.addEventListener("click",()=>cn(!1));const _e=F("lyria-play-btn");_e&&_e.addEventListener(
"click",()=>{if(!b){Ye();return}at("PLAY")});const de=F("lyria-pause-btn");de&&de.addEventListener("\
click",()=>at("PAUSE"));const Ce=F("lyria-stop-btn");Ce&&Ce.addEventListener("click",()=>at("STOP"));
const Fe=F("lyria-reset-btn");Fe&&Fe.addEventListener("click",()=>at("RESET_CONTEXT"));const Qe=F("l\
yria-add-prompt-btn");Qe&&Qe.addEventListener("click",()=>fe("",1));const Ge=F("lyria-apply-prompts-\
btn");Ge&&Ge.addEventListener("click",be);const et=F("lyria-apply-config-btn");et&&et.addEventListener(
"click",Pe);const pt=F("lyria-save-btn");pt&&pt.addEventListener("click",He),st(),ct(),window.openLyriaStudio=
Vt}o(dn,"init");function jn(){ue||cn(!1)}return o(jn,"requestClose"),{init:dn,open:Vt,requestClose:jn}})();
oi.init(),window.closeLyriaStudio=()=>oi.requestClose(),(()=>{let d=null,m=null,g=null;const b="voic\
eDockSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=o(se=>document.getElementById(se),"$");function L(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(L,"isStudioMode");function A(){const se=get("model-select")?get("model-select").value:"",fe=S("voi\
ce-studio-title");fe&&(se==="gpt-transcribe"||se==="gpt-live-transcribe"||se==="grok-voice-transcrib\
e-2.0-file"?fe.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":se===
"gemini-3.5-live-translate-preview"?fe.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
fe.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(A,"updateTitle");function $(){const se=S("v\
oice-studio-transcript");se&&(se.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${x}</div>`);const fe=S("sts-live-transcript");fe&&(fe.innerHTML="",fe.classList.add("hidden"))}
o($,"resetTranscript");function H(se,fe,Ee){const $e=se.querySelectorAll(".voice-studio-line");let st=null;
for(let Ue=$e.length-1;Ue>=0;Ue--)if($e[Ue].dataset.role===fe){st=$e[Ue];break}if(st)st.innerHTML=Ee;else{
const Ue=se.querySelector(".voice-studio-placeholder");Ue&&Ue.remove();const ut=document.createElement(
"div");ut.className="voice-studio-line",ut.dataset.role=fe,ut.innerHTML=Ee,se.appendChild(ut)}se.classList.
remove("hidden"),se.scrollTop=se.scrollHeight}o(H,"writeLine");function B(se,fe){if(!fe||!String(fe).
trim()||!L())return;const st=`<span class="${se==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(se==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
fe)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(Ue=>H(Ue,
se,st))}o(B,"log");function W(se,fe=!0){const Ee=S("sts-panel"),$e=S("sts-settings-toggle");if(Ee&&Ee.
classList.toggle("settings-open",!!se),$e&&$e.setAttribute("aria-expanded",se?"true":"false"),fe)try{
localStorage.setItem(b,se?"1":"0")}catch{}}o(W,"setSettingsOpen");function ue(){try{return localStorage.
getItem(b)==="1"}catch{return!1}}o(ue,"readSettingsOpen");let re=null;function F(){const se=get("mod\
el-select")?get("model-select").value:"";se!==re&&(re=se,$()),W(ue(),!1),A()}o(F,"syncDock");function ae(){
const se=S("sts-panel"),fe=S("voice-studio-panel-host");se&&fe&&se.parentNode!==fe&&(d=se.parentNode,
m=se.nextSibling,fe.appendChild(se));const Ee=S("file-preview"),$e=S("voice-studio-file-host");Ee&&$e&&
Ee.parentNode!==$e&&(g=Ee.parentNode,$e.appendChild(Ee),$e.classList.remove("hidden"))}o(ae,"movePan\
elIntoModal");function V(){const se=S("sts-panel");se&&d&&se.parentNode!==d&&(m&&m.parentNode===d?d.
insertBefore(se,m):d.appendChild(se));const fe=S("file-preview");fe&&g&&fe.parentNode!==g&&g.appendChild(
fe);const Ee=S("voice-studio-file-host");Ee&&Ee.classList.add("hidden"),d=null,m=null,g=null}o(V,"mo\
vePanelBack");function Ae(){if(!L())return;ae();const se=S("sts-panel");se&&se.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(Ae,"open");function K(){window.
VoiceStudioOpen=!1,V(),hideModal("voice-studio-modal")}o(K,"close");function ce(){window.VoiceStudioOpen&&
K()}o(ce,"closeIfOpen");function xe(){window.VoiceStudioOpen=!1;const se=S("voice-studio-open-btn");
se&&se.addEventListener("click",()=>Ae());const fe=S("voice-studio-close");fe&&fe.addEventListener("\
click",()=>K());const Ee=S("sts-settings-toggle");Ee&&Ee.addEventListener("click",()=>{const $e=S("s\
ts-panel");W(!($e&&$e.classList.contains("settings-open")))}),window.VoiceStudio={open:Ae,close:K,closeIfOpen:ce,
log:B,isStudioMode:L,syncDock:F},F()}return o(xe,"init"),{init:xe}})().init();let Wt=null;function ri(){
if(Wt&&(Wt.stop(),Wt=null),sn){try{sn.pause()}catch{}try{sn.src=""}catch{}sn=null}}o(ri,"stopStsPlay\
back");async function Ai(d){ri();const m=new Audio;return m.src=d,m.preload="auto",m.autoplay=!0,m.playsInline=
!0,sn=m,await m.play(),new Promise(g=>{m.onended=()=>g("ended"),m.onerror=()=>g("error")})}o(Ai,"pla\
yStsAudio");function Si(){if(rn.isActive()){rn._cancel();return}if(Oe){Oe.stop(),Oe=null,ri(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Ot();return}it&&it.state===
"recording"&&($n=!0,it.stop())}o(Si,"cancelRecording");function Nn(){if(isStsModel())return{audio:!0};
const m=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return m.echoCancellation&&(g.echoCancellation=!1),m.
noiseSuppression&&(g.noiseSuppression=!1),m.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(Nn,"\
getMicCaptureConstraints"),get("mic-btn").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\
\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Oe){setStsStatus("Processing...",!0);const d=Oe;Oe=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const m=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(m.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
m.assistant_text=(d.inputTranscript||"").trim(),m.assistant_thought="",!m.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){
const b=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=
String(b.id),history.pushState({},"","/c/"+b.id),get("welcome-screen").classList.add("hidden")}m.thread_id=
currentThreadId,m.model=get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)}),setStsStatus("Saved",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(m){console.error(
"Failed to save Gemini Live session:",m),setStsStatus("Error saving",!1)}return}if(rn.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
rn.stop();return}if(it&&it.state==="recording"){it.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||Pt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const b={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(b.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const F=get("sts-custom\
-vocab").value.split(/[,、\n]/).map(ae=>ae.trim()).filter(Boolean);F.length&&(b.custom_vocabulary=F.
slice(0,1e3))}}else b.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(b.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",b.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(b.target_lang=get("sts-target-lang").value);const x=await apiFetch("/api/gemini\
/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});if(!x.
ok)throw new Error("Failed to get session token");const{token:S,url:L}=await x.json(),A=get("model-s\
elect").value,$=get("sts-voice")?get("sts-voice").value:"Kore",H=get("sts-thinking-level")?get("sts-\
thinking-level").value:"minimal",B=get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1;
if(Oe=new Pn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Oe.rtPlayer=new on),isGeminiLiveTranscribeModel()){
const F=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",ae={languageCodes:[]};
if((F==="SMART"||F==="VERBATIM")&&(ae.mode=F),get("sts-custom-vocab")){const V=get("sts-custom-vocab").
value.split(/[,、\n]/).map(Ae=>Ae.trim()).filter(Boolean);V.length&&(ae.customVocabulary=V.slice(0,
1e3))}await Oe.start(S,L,A,{transcriptionConfig:ae})}else if(isGeminiLiveTranslateModel()){const F=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Oe.start(S,L,A,{translationConfig:{targetLanguageCode:F,
echoTargetLanguage:!0}})}else{const F={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:$}}}};
isGeminiLiveExtendedThinkingModel()&&(F.thinkingConfig={thinkingLevel:H,includeThoughts:B}),await Oe.
start(S,L,A,F)}it=Oe.backupRecorder,it.onstop=()=>{Oe&&get("mic-btn").click()};let W=!0,ue="live-sts\
-"+Date.now();Oe.onMessage=F=>{if(F.serverContent){if(isGeminiLiveTranscribeModel()){const ae=Oe.interimInputTranscript,
V=Oe.inputTranscript,Ae=V+(ae&&!V.endsWith(ae)?(V?`
`:"")+ae:""),K=get("chat-messages");let ce=document.getElementById(ue);ce||(ce=document.createElement(
"div"),ce.id=ue,ce.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded\
-lg border border-slate-700/50",ce.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,K.appendChild(ce),K.scrollTop=K.scrollHeight);const xe=ce.
querySelector(".message-content");xe.innerText=Ae||"\u8074\u304D\u53D6\u308A\u4E2D...",K.scrollTop=K.
scrollHeight,window.VoiceStudio&&V&&window.VoiceStudio.log("user",V);return}if(F.serverContent.modelTurn){
W&&(setStsStatus("Gemini is speaking...",!1),W=!1);const ae=get("chat-messages");let V=document.getElementById(
ue);V||(V=document.createElement("div"),V.id=ue,V.className="flex flex-col gap-2 mb-4 assistant-mess\
age bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",V.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ae.appendChild(V),ae.scrollTop=ae.scrollHeight);const Ae=V.
querySelector(".thought-container"),K=V.querySelector(".message-content");Oe.assistantThought&&(Ae.classList.
remove("hidden"),Ae.innerText=Oe.assistantThought),K.innerText=Oe.assistantText,ae.scrollTop=ae.scrollHeight,
window.VoiceStudio&&(Oe.inputTranscript&&window.VoiceStudio.log("user",Oe.inputTranscript),Oe.assistantText&&
window.VoiceStudio.log("assistant",Oe.assistantText))}}},setStsStatus("Listening...",!0),get("mic-bt\
n").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),ni(Oe.
stream),ii(Oe.stream);const re=Oe;re.onError=F=>{const ae=F&&F.message?F.message:typeof F=="string"?
F:"";ae&&showToast("Gemini Live: "+ae,"error",!0)},re.onClose=F=>{if(Oe===re){if(F&&F.code&&F.code!==
1e3){const ae=F.reason?": "+F.reason:" (code "+F.code+")";showToast("Gemini Live \u306E\u63A5\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+
ae,"error",!0)}get("mic-btn").click()}},re.closedEvent&&re.onClose(re.closedEvent);return}catch(g){showToast(
"Gemini Live connection failed: "+g.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await rn.start();return}isStsModel()||(ti(),Pt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(Nn());it=new MediaRecorder(d),hn=[],$n=!1;const m=isStsModel();
it.ondataavailable=g=>hn.push(g.data),it.onstop=async()=>{if($n){hn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(L=>L.stop()),kn(),Ot(),m||(Pt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),xn=setTimeout(()=>Pt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(hn,{type:"audio/web\
m"}),b=new File([g],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",b),get("fil\
e-preview").classList.remove("hidden");const S=m;get("file-name").innerText=S?"Processing voice...":
"Transcribing...";try{if(S){if(!currentThreadId){const V=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=V.id!==null&&V.id!==void 0?String(V.id):V.id,setTemporaryChatUiState(!!(V&&V.
is_temporary)),setCurrentChatHeaderTitle(V&&V.title),applyTemporaryChatRuntimeMeta(V||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+V.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),x.append("model",get(
"model-select").value),x.append("thread_id",currentThreadId),get("sts-voice")&&x.append("sts_voice",
get("sts-voice").value||""),get("sts-reasoning-effort")&&x.append("sts_reasoning_effort",get("sts-re\
asoning-effort").value||""),get("sts-speed")&&x.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&x.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&x.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&x.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&x.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const L=await apiFetch(
"/sts",{method:"POST",body:x});if(!L.ok){const ae=await L.json().catch(()=>({}));throw new Error(ae.
error||"Speech-to-speech failed")}const A=L.body.getReader(),$=new TextDecoder;let H="",B=null,W=null;
stsOpt("sts-auto-play")&&(W=new on,Wt=W),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let ue=!0,re="",F="";for(;;){const{done:ae,value:V}=await A.read();if(ae)break;
H+=$.decode(V,{stream:!0});const Ae=H.split(`
`);H=Ae.pop();for(const K of Ae){if(!K.trim())continue;const ce=JSON.parse(K);if(ce.error)throw new Error(
ce.error);ce.audio_delta&&W&&(ue&&(setStsStatus("Playing response...",!1),ue=!1),await W.addChunk(ce.
audio_delta)),ce.input_delta&&(re+=ce.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
re)),ce.transcript_delta&&(F+=ce.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",F)),(ce.final||ce.audio_url)&&(B=ce)}}window.VoiceStudio&&!re.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),B&&(B.audio_url||B.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const L=get(
"set-mic-transcribe-mode");if(!!(L&&L.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const H=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(H.transcript){const B=get("prompt-inp\
ut");B.value+=(B.value?" ":"")+H.transcript,B.style.height="auto",B.style.height=B.scrollHeight+"px"}else
showToast(H.error||"Transcription failed","error",!0)}}catch(L){showToast("Audio processing error: "+
L.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(L=>L.
stop()),kn(),Ot(),S||Pt("","hidden"),S&&setStsStatus("Tap to speak",!1)}},it.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Pt("\u9332\u97F3\u4E2D\u2026","recording"),ni(d)),ii(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Ot(),isStsModel()||Pt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const ln=o((d,m)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=m:d.textContent=
m},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,m=get("lib-del-btn"),g=get("lib-download-btn"),b=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(m&&(m.disabled=d===0,ln(m,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,ln(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
b&&(b.disabled=d===0,ln(b,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,ln(x,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=d!==1,ln(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const L=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",L&&d>0)}};function li(d){lib.
attachMode=!!d}o(li,"setLibAttachMode");const ci=o((d=!1)=>{li(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>ci(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>ci(!0)),
get("lib-search")){let d=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),d&&clearTimeout(d),d=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const d=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=d,get("lib-sort").onchange=()=>{const m=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,m),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const Li=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>Li(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
m=get("gem-desc").value,g=get("gem-inst").value,b=collectGemFixedPrompts();if(d&&g){const x=editingGemUuid?
"PUT":"POST",S=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
S,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:m,instruction:g,
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
try{const x=decodeURIComponent(g),S=new Blob([x],{type:"text/plain"}),L=URL.createObjectURL(S),A=document.
createElement("a");A.href=L;let H={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[b]||b;(b.length>8||/[^a-z0-9]/.test(b))&&(H="txt");let B=`code.${H}`;b==="dockerfile"&&(B="Docke\
rfile"),b==="makefile"&&(B="Makefile"),A.download=B,document.body.appendChild(A),A.click(),document.
body.removeChild(A),URL.revokeObjectURL(L)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const m=d.target.closest(".copy-btn"),g=m.getAttribute("data-code");g&&
window.copyCode(m,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const m=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(m)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",m=>{m.target===d&&Yn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
