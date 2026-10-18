document.addEventListener("DOMContentLoaded",()=>{var gi,hi;initThemeFromServer(),applyLiquidGlassMode(
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
!0);const u=get("browser-fast-mode-ignore-warning");if(u&&u.checked)try{localStorage.setItem(BROWSER_FAST_IGNORE_WARNING_STORAGE,
"1")}catch{}hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!0,{clearKey:!1}),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u3092\u6709\u52B9\u306B\u3057\u307E\u3057\u305F\u3002\u751F\u6210\u4E2D\u306F\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u306A\u3044\u3067\u304F\u3060\u3055\u3044\u3002",
"warning",!0)}catch(u){showToast(u.message||"\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{i.disabled=!1,i.innerHTML=d}});const a=get("browser-fast-mode-cancel-btn");a&&(a.
onclick=()=>{hideModal("browser-fast-mode-modal"),setBrowserFastModeEnabled(!1)});const r=document.getElementById(
"alpha-bar");setTimeout(()=>{if(r){const d=document.getElementById("version-display");if(d){const u=r.
getBoundingClientRect(),g=d.getBoundingClientRect(),h=g.left+g.width/2-(u.left+u.width/2),x=g.top+g.
height/2-(u.top+u.height/2);r.style.transform=`translate(${h}px, ${x}px) scale(0.1)`,r.style.opacity=
"0",setTimeout(()=>{d.classList.add("pulse-target"),setTimeout(()=>d.classList.remove("pulse-target"),
2e3),r.remove()},800)}else r.style.opacity="0",setTimeout(()=>r.remove(),1e3)}},3e3);function l(){const d=get(
"gpt-image-options");if(!d)return;isGptImageModel()?d.classList.remove("hidden"):d.classList.add("hi\
dden");const u=get("gpt-image-format"),g=get("gpt-image-compression-wrap");u&&g&&(u.value==="png"?g.
classList.add("hidden"):g.classList.remove("hidden"))}o(l,"updateGptImageUi");function c(){const d=get(
"gemini-image-options");if(!d)return;isGeminiImageModel()?d.classList.remove("hidden"):d.classList.add(
"hidden");const g=(get("model-select").value||"").toLowerCase().includes("gemini-3.1-flash-lite-imag\
e");[get("gemini-image-size"),get("modal-gemini-image-size")].forEach(h=>{h&&(Array.from(h.options).
forEach(x=>{x.value!=="1K"&&(x.disabled=g)}),g&&h.value!=="1K"&&(h.value="1K"))})}o(c,"updateGeminiI\
mageUi");function m(){const d=get("grok-image-options");if(!d)return;const u=(get("model-select").value||
"").toLowerCase(),g=isGrokImageModel(),h=u==="grok-imagine-image-quality"||u==="grok-imagine-image-2\
.0",x=u==="grok-imagine-image-2.0";if(g){d.classList.remove("hidden");const L=get("grok-image-resolu\
tion")?get("grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!h);const A=get(
"grok-image-quality")?get("grok-image-quality").parentElement:null;A&&A.classList.toggle("hidden",!x)}else
d.classList.add("hidden");if(get("modal-grok-image-options")){const L=get("modal-grok-image-resoluti\
on")?get("modal-grok-image-resolution").parentElement:null;L&&L.classList.toggle("hidden",!h);const A=get(
"modal-grok-image-quality")?get("modal-grok-image-quality").parentElement:null;A&&A.classList.toggle(
"hidden",!x)}}o(m,"updateGrokImageUi");function f(){const d=get("ideogram-image-options"),u=get("mod\
al-ideogram-image-options"),g=get("model-select")&&get("model-select").value||"",h=isIdeogramModel(g),
x=ideogramModelTraits(g),S={resolution:x.resolution,quality:x.quality,speed:x.speed,style:x.style,negative:x.
negative};if(d&&d.classList.toggle("hidden",!h),u&&u.classList.toggle("hidden",!h),!h)return;IDEOGRAM_IMAGE_FIELDS.
forEach(A=>{const E=S[A]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(R=>{const B=get(R+
A);!B||!B.parentElement||B.parentElement.classList.toggle("hidden",!E)})}),["ideogram-image-style","\
modal-ideogram-image-style"].forEach(A=>{const E=get(A);if(!E)return;const R=x.styleOptions.join("|");
if(E.dataset.options!==R){const B=E.value;E.innerHTML=x.styleOptions.map(V=>`<option value="${V}">${V===
"auto"?"Auto":V}</option>`).join(""),E.dataset.options=R,E.value=x.styleOptions.includes(B)?B:"auto"}});
const L=get("ideogram-image-note");L&&(L.textContent=x.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F P\
recise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":"\u203B Ideogram: \u30C6\u30AD\
\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi");function b(){var h;const d=get("grok-video-options");if(!d)return;const u=String(
((h=get("model-select"))==null?void 0:h.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove(
"hidden"):d.classList.add("hidden");const g=get("grok-video-resolution");if(g){const x=Array.from(g.
options).find(S=>S.value==="1080p");x&&(x.disabled=u!=="grok-imagine-video-1.5"),u!=="grok-imagine-v\
ideo-1.5"&&g.value==="1080p"&&(g.value="720p")}}o(b,"updateGrokVideoUi");function y(){var x;const d=get(
"gemini-video-options");if(!d)return;const u=String(((x=get("model-select"))==null?void 0:x.value)||
"").toLowerCase();isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get(
"gemini-video-resolution");if(g){const S=Array.from(g.options).find(A=>A.value==="4K"),L=u==="veo-3.\
1-lite-generate-preview"||u==="veo-3.1-fast-generate-preview"||u==="gemini-omni-flash";S&&(S.disabled=
L),L&&g.value==="4K"&&(g.value="1080p")}const h=get("gemini-video-duration-wrap");h&&h.classList.toggle(
"hidden",u==="gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function v(){const d=get("gemini-mu\
sic-options");if(!d)return;const u=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!u;d.classList.
toggle("hidden",!g);const h=get("lyria-realtime-studio-bar");h&&h.classList.toggle("hidden",!u)}o(v,
"updateGeminiMusicUi");function w(){var L;const d=get("xai-chat-options");if(!d)return;const u=String(
((L=get("model-select"))==null?void 0:L.value)||"").toLowerCase(),g=u.startsWith("grok-")&&!isGrokImageModel(
u)&&!isGrokVideoModel(u)&&!u.includes("voice");d.classList.toggle("hidden",!g);const h=get("xai-logp\
robs"),x=get("xai-top-logprobs"),S=u.includes("grok-4.20");h&&(h.disabled=S,S&&(h.checked=!1)),x&&(x.
disabled=S,S&&(x.value=""))}o(w,"updateXaiChatUi");function k(){const d=isMistralOcrModel(),u=get("m\
istral-ocr-options");u&&u.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(h=>{const x=get(h);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function _(){const d=get("image-input-limit\
s");if(!d)return;const u=(get("model-select").value||"").toLowerCase();let g="",h=!1;u.includes("gpt\
-image")?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">GPT-Image \u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 16 \u679A / \u753B\u50CF1\u679A\u3042\u305F\u308A 50MB \u672A\u6E80 / PNG\u30FBJPG\u30FBWEBP</div>",
"<div>\u30DE\u30B9\u30AF\u4F7F\u7528\u6642: PNG\u306E\u307F\u30014MB\u672A\u6E80\u3001\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA</div>"].
join("")):u==="deepseek-v4.1-flash"||u==="deepseek-v4-flash-vision-exp"?(h=!0,g=['<div class="font-b\
old text-gray-300 mb-1">DeepSeek V4.1 Flash \u5165\u529B\u5236\u9650</div>',"<div>JPEG\u30FBPNG\u30FBGIF\u30FBWebP \
/ \u753B\u50CF1\u679A\u3042\u305F\u308A\u6700\u592732MB / \u30EA\u30AF\u30A8\u30B9\u30C8\u5408\u8A0848MB</div>",
"<div>\u753B\u50CF\u306F\u7D04800\xD7800\u76F8\u5F53\u3078\u81EA\u52D5\u30EA\u30B5\u30A4\u30BA\uFF081\u679A\u3042\u305F\u308A\u6700\u5927384\u30C8\u30FC\u30AF\u30F3\uFF09</div>"].
join("")):u.includes("deepseek")||(isGeminiImageModelKey(u)?(h=!0,u.includes("gemini-3.1-flash-lite-\
image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2 Lite \u5165\u529B\u76EE\u5B89</div>',
"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u8907\u6570\u53C2\u7167\u3084\u9023\u7D9A\u7DE8\u96C6\u3088\u308A\u3001\u4F4E\u9045\u5EF6\u30FB\u5927\u91CF\u751F\u6210\u5411\u3051\u3067\u3059</div>"].
join(""):u==="gemini-nano-banana-2.1"?g=['<div class="font-bold text-gray-300 mb-1">Nano Banana 2.1 \
\u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u751F\u6210\u30FB\u7DE8\u96C6 / 1K\u30FB2K\u30FB4K\u51FA\u529B / \u6700\u592714\u679A\u306E\u53C2\u7167\u753B\u50CF\u306B\u5BFE\u5FDC</div>",
"<div>\u52D5\u753B\u3092\u53C2\u8003\u306B\u3057\u305F\u753B\u50CF\u751F\u6210\u306B\u3082\u5BFE\u5FDC\u3057\u307E\u3059</div>"].
join(""):u.includes("gemini-3.1-flash-image")?g=['<div class="font-bold text-gray-300 mb-1">Nano Ban\
ana 2 \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u7A0B\u5EA6\u3092\u63A8\u5968\uFF08Gemini 3.1 Flash Image\uFF09</div>"].
join(""):u.includes("gemini-2.5")&&u.includes("image")?g=['<div class="font-bold text-gray-300 mb-1"\
>Nano Banana \u5165\u529B\u76EE\u5B89</div>',"<div>\u753B\u50CF\u5165\u529B\u306F\u6700\u59273\u679A\u307E\u3067\u304C\u63A8\u5968</div>"].
join(""):g=['<div class="font-bold text-gray-300 mb-1">Nano Banana Pro \u5165\u529B\u76EE\u5B89</div>',
"<div>\u9AD8\u7CBE\u5EA6\u306F\u6700\u59275\u679A / \u5408\u8A0814\u679A\u307E\u3067\u5BFE\u5FDC</div>"].
join("")):isMistralOcrModel(u)?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">Mistral OCR 4 \u5165\u529B<\
/div>',"<div>PDF / PNG / JPEG / TIFF / BMP / GIF / WEBP / DOCX / PPTX\u3001\u307E\u305F\u306F\u516C\u958BURL</div>",
"<div>\u6700\u5927 512MB / \u4F1A\u8A71\u5C65\u6B74\u306F\u9001\u4FE1\u3057\u307E\u305B\u3093 / \u30C1\u30E3\u30C3\u30C8\u88DC\u5B8C\u30FBSearch\u30FBPython\u30FBCanvas \u975E\u5BFE\u5FDC</div>"].
join("")):u.includes("grok")?(h=!0,g=['<div class="font-bold text-gray-300 mb-1">Grok \u753B\u50CF\u5165\u529B\u5236\u9650</div>',
"<div>\u6700\u5927 20MiB / PNG\u30FBJPG \u306E\u307F / \u679A\u6570\u5236\u9650\u306A\u3057</div>"].
join("")):isIdeogramModel(u)?(h=!0,g=ideogramModelTraits(u).edit?['<div class="font-bold text-gray-3\
00 mb-1">Ideogram 4.5 \u5165\u529B\u5236\u9650</div>',"<div>\u7DE8\u96C6\u3059\u308B\u753B\u50CF1\u679A + \u53C2\u8003\u753B\u50CF\u306F\u6700\u59274\u679A / 1\u679A\u3042\u305F\u308A25MB\u307E\u3067 / PNG\
\u30FBJPEG\u30FBWEBP</div>","<div>\u753B\u50CF\u304C\u306A\u3044\u5834\u5408\u306F\u76F4\u524D\u306E\u753B\u50CF\u3092\u7DE8\u96C6\u3057\u307E\u3059 / \u30DE\u30B9\u30AF\u306B\u306F\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093</div>"].
join(""):['<div class="font-bold text-gray-300 mb-1">Ideogram \u5165\u529B</div>',"<div>\u30C6\u30AD\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\u3067\
\u3059\u3002\u753B\u50CF\u5165\u529B\u306F Ideogram 4.5 \u3060\u3051\u304C\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u3059</div>"].
join("")):u.includes("grok")&&u.includes("video")&&(h=!0,g=['<div class="font-bold text-gray-300 mb-\
1">Grok \u52D5\u753B\u751F\u6210\u5236\u9650</div>',"<div>Duration: 1-15s / Resolution: 720p, 480p</\
div>","<div>\u753B\u50CF\u304B\u3089\u306E\u52D5\u753B\u751F\u6210\u306B\u5BFE\u5FDC (PNG\u30FBJPG)</div>"].
join(""))),h?(d.innerHTML=g,d.classList.remove("hidden")):(d.classList.add("hidden"),d.innerHTML="")}
o(_,"updateImageInputLimits");function T(){const d=get("enable-sys-prompt"),u=!!(d&&d.disabled),g=!!(d&&
d.checked);M(),d&&(d.disabled?!u&&g?d.dataset.restoreChecked="1":u||delete d.dataset.restoreChecked:
(u&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(T,"toggleOpti\
ons"),window.toggleOptions=T;function M(){const d=get("model-select");if(!d)return;const u=d.value,g=String(
u||"").toLowerCase(),h=g.includes("deepseek"),x=g.startsWith("glm-"),S=get("thinking-options"),L=get(
"reasoning-effort-container"),A=get("enable-thinking"),E=get("thinking-level"),R=get("thinking-budge\
t"),B=get("enable-search"),V=get("search-container"),ue=get("url-context-container"),oe=get("enable-\
maps"),J=get("maps-grounding-container"),de=get("enable-sys-prompt"),H=get("sys-prompt-option"),me=get(
"enable-python"),W=get("python-container"),ce=get("prompt-cache-container"),he=get("enable-prompt-ca\
che"),ee=u==="gpt-5-search-api",fe=u.includes("tts"),Ee=isMistralOcrModel(u),$e=g.includes("gemini-3\
.1-flash-lite-image"),st=g==="gemini-nano-banana-2.1",Ue=g.includes("gemini-3.1-flash-image")&&!$e||
st,ut=isClaudeModelKey(u),Tt=g==="gemini-3.8-flash-cyber",_e=isLlmModel()&&!h&&!fe&&!g.includes("rea\
ltime")&&!g.includes("native-audio")&&!g.includes("live");ce&&(_e?(ce.classList.remove("hidden","opa\
city-50","pointer-events-none"),he&&(he.disabled=!1)):(he&&(he.checked=!1,he.disabled=!0),ce.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),S&&S.classList.add("hidden"),L&&L.classList.
add("hidden");const be=get("vision-model-info");if(be&&be.classList.add("hidden"),L){const ve=get("r\
easoning-effort");if(ve){Array.from(ve.options).forEach(He=>{const ze=g==="gpt-5.6"||g.startsWith("g\
pt-5.6-"),ct=g.startsWith("gpt-6"),cn=g==="gpt-6-sol"||g==="gpt-6-luna",Wt=g==="deepseek-v4.1-flash"||
g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash-vision-exp",dn=g==="de\
epseek-v4-pro",Fn=g.includes("grok-4.5"),F=g.includes("grok-4.6");He.value==="max"?He.classList.toggle(
"hidden",!ze&&!ct&&!Wt&&!dn):He.value==="xhigh"?He.classList.toggle("hidden",!F&&!g.includes("multi-\
agent")&&!ze&&!ct):He.value==="medium"?He.classList.toggle("hidden",!(g.includes("grok-4.3")||Fn||F||
g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("multi-agent")||g.includes(
"gpt-5")||ct||g.includes("o1")||g.includes("o3"))):He.value==="none"?He.classList.toggle("hidden",!g.
includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!cn&&!Wt&&!dn):He.value==="lo\
w"&&He.classList.toggle("hidden",dn)});const Pe=ve.selectedOptions&&ve.selectedOptions[0];Pe&&Pe.classList.
contains("hidden")&&(ve.value=h?"high":"medium")}}ue&&ue.classList.add("hidden"),J&&J.classList.add(
"hidden"),A&&(A.disabled=!1),R&&(R.disabled=!0,R.classList.add("opacity-50"));const Be=isGeminiImageModelKey(
u);if(fe||Ee)V&&(get("enable-search").checked=!1,V.classList.add("opacity-50","pointer-events-none")),
ue&&(get("enable-url-context").checked=!1,ue.classList.add("opacity-50","pointer-events-none")),J&&oe&&
(oe.checked=!1,J.classList.add("opacity-50","pointer-events-none")),W&&(me.checked=!1,W.classList.add(
"opacity-50","pointer-events-none")),de&&H&&(de.checked=!1,de.disabled=!0,H.classList.add("opacity-5\
0"));else if(Ue||$e)J&&oe&&(oe.checked=!1,J.classList.add("hidden","opacity-50","pointer-events-none")),
S.classList.remove("hidden"),st?(Array.from(E.options).forEach(ve=>{ve.disabled=ve.value==="low"}),[
"minimal","medium","high"].includes(E.value)||(E.value="medium")):(Array.from(E.options).forEach(ve=>{
["low","medium"].includes(ve.value)&&(ve.disabled=!0),["minimal","high"].includes(ve.value)&&(ve.disabled=
!1)}),["minimal","high"].includes(E.value)||(E.value=$e?"minimal":"high")),A&&(A.disabled=!1),$e&&(B&&
(B.checked=!1,B.disabled=!0),V&&V.classList.add("opacity-50","pointer-events-none"));else if(Be)J&&oe&&
(oe.checked=!1,J.classList.add("hidden","opacity-50","pointer-events-none"));else if(ut)S.classList.
remove("hidden"),R&&(R.disabled=!1,R.classList.remove("opacity-50")),Array.from(E.options).forEach(ve=>{
ve.disabled=!0}),W&&(me.checked=!1,W.classList.add("opacity-50","pointer-events-none"));else if(Tt){
S&&S.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from(E.options).forEach(Pe=>{Pe.
disabled=!["low","medium","high"].includes(Pe.value)}),["low","medium","high"].includes(E.value)||(E.
value="medium"),[V,ue,J,W].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")}),[
B,oe,me].forEach(Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const ve=get("enable-url-context");ve&&(ve.
checked=!1,ve.disabled=!0),de&&H&&(de.disabled=!1,H.classList.remove("opacity-50"))}else if(u.includes(
"gemini")&&!Be){S.classList.remove("hidden"),ue&&ue.classList.remove("hidden","opacity-50","pointer-\
events-none");const ve=u.includes("gemini-3")||u.includes("gemini-4");J&&(ve?J.classList.remove("hid\
den","opacity-50","pointer-events-none"):(oe&&(oe.checked=!1),J.classList.add("hidden","opacity-50",
"pointer-events-none")));const Pe=u.includes("flash");Array.from(E.options).forEach(He=>{u==="gemini\
-4-argon"||u==="gemini-3.8-flash"||u==="gemini-3.7-flash"?He.disabled=!["low","medium","high"].includes(
He.value):u==="gemini-3.6-flash"?He.disabled=!["medium","high"].includes(He.value):u==="gemini-3.5-f\
lash-lite"?He.disabled=!["minimal","medium","high"].includes(He.value):["minimal","medium"].includes(
He.value)?He.disabled=!Pe:He.disabled=!1}),(u==="gemini-4-argon"||u==="gemini-3.8-flash"||u==="gemin\
i-3.7-flash")&&!["low","medium","high"].includes(E.value)||u==="gemini-3.6-flash"&&!["medium","high"].
includes(E.value)?E.value="medium":u==="gemini-3.5-flash-lite"&&!["minimal","medium","high"].includes(
E.value)?E.value="minimal":!Pe&&["minimal","medium"].includes(E.value)&&(E.value="high"),ve?A&&(A.checked=
!0,A.disabled=!0):A&&(A.disabled=!1),R&&u.includes("gemini-2.5")&&(R.disabled=!1,R.classList.remove(
"opacity-50")),R&&!u.includes("gemini-2.5")&&(R.disabled=!0,R.classList.add("opacity-50"))}if(isLlmModel()&&
(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes("grok-4.3")||g.includes("grok-4\
.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes(
"multi-agent")||g.includes("gpt")&&!g.includes("tts")))L.classList.remove("hidden"),V&&V.classList.remove(
"opacity-50","pointer-events-none");else if(h){L.classList.remove("hidden");const ve=get("vision-mod\
el-info");if(ve&&ve.classList.toggle("hidden",g==="deepseek-v4.1-flash"||g==="deepseek-v4-flash-visi\
on-exp"),B&&(B.checked=!1,B.disabled=!0),V&&V.classList.add("opacity-50","pointer-events-none"),ue){
const Pe=get("enable-url-context");Pe&&(Pe.checked=!1),ue.classList.add("opacity-50","pointer-events\
-none")}J&&oe&&(oe.checked=!1,J.classList.add("opacity-50","pointer-events-none"))}else Ee||(V&&V.classList.
remove("opacity-50","pointer-events-none"),J&&oe&&(oe.checked=!1,J.classList.add("hidden","opacity-5\
0","pointer-events-none")));if(fe?W&&W.classList.add("opacity-50","pointer-events-none"):(W&&W.classList.
remove("opacity-50","pointer-events-none"),(!Be||Ue)&&!u.includes("gpt-image")&&(de.disabled=!1,H.classList.
remove("opacity-50"))),(Be&&!Ue||u.includes("gpt-image")||isGrokImageModel()||isIdeogramModel(u)||isGrokVideoModel()||
Ee)&&de&&H&&(de.checked=!1,de.disabled=!0,H.classList.add("opacity-50")),W&&(isLlmModel()?(W.classList.
remove("hidden"),me.disabled=!1):(me.checked=!1,me.disabled=!0,W.classList.add("hidden"))),ee?(B&&(B.
checked=!0,B.disabled=!0),V&&V.classList.add("opacity-50","pointer-events-none"),W&&(me.checked=!1,me.
disabled=!0,W.classList.add("opacity-50","pointer-events-none"))):B&&!u.includes("tts")&&!Ee&&!h&&!$e&&
(B.disabled=!1),Tt){[B,oe,me].forEach(Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const ve=get("enable\
-url-context");ve&&(ve.checked=!1,ve.disabled=!0),[V,ue,J,W].forEach(Pe=>{Pe&&Pe.classList.add("opac\
ity-50","pointer-events-none")})}x&&(be&&be.classList.toggle("hidden",["glm-5.3-flash","glm-5.3-flas\
hx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(g)),[B,me].forEach(ve=>{ve&&(ve.
checked=!1,ve.disabled=!0)}),[V,W].forEach(ve=>{ve&&ve.classList.add("opacity-50","pointer-events-no\
ne")}));const at=get("mask-btn");at&&(isGptImageModel()?at.classList.remove("hidden"):(at.classList.
add("hidden"),currentMaskImage=null,updateMaskPreview())),updateTtsUi(),updateStsUi(),updateStsOptions(),
l(),c(),m(),f(),b(),y(),v(),updateBatchUi(u),w(),k(),_(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),
applyMcpPromptChipUi()}o(M,"toggleOptionsForModel"),get("model-select")&&(get("model-select").addEventListener(
"change",T),get("model-select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),
T(),minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const P=get("enable-canvas-mode");P&&(P.checked=canvasModeEnabled,P.addEventListener("change",()=>syncCanvasModeUi(
P.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const G=get("enable-codin\
g-mode");G&&(G.checked=codingModeEnabled,G.addEventListener("change",()=>syncCodingModeUi(G.checked))),
get("clear-coding-target-btn")&&get("clear-coding-target-btn").addEventListener("click",()=>{codingTargetSelection=
null,syncCodingModeUi(codingModeEnabled,{persist:!1}),showToast("\u6700\u65B0\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u81EA\u52D5\u9078\u629E\u3057\u307E\u3059",
"info",!1)}),syncCodingModeUi(codingModeEnabled,{persist:!1}),get("canvas-panel-close-btn")&&get("ca\
nvas-panel-close-btn").addEventListener("click",()=>syncCanvasModeUi(!1)),get("canvas-panel-clear-bt\
n")&&get("canvas-panel-clear-btn").addEventListener("click",()=>{canvasModeEnabled&&(resetCanvasPreviewPanel(),
showToast("Canvas\u30D7\u30EC\u30D3\u30E5\u30FC\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F","in\
fo",!1))}),get("canvas-block-list")&&get("canvas-block-list").addEventListener("click",d=>{const u=d.
target.closest("[data-canvas-block-index]");if(!u)return;const g=Number(u.getAttribute("data-canvas-\
block-index"));applyCanvasSelection(g,{view:"preview",animateView:!0,transitionFrom:"blocks"})}),get(
"canvas-source-select")&&get("canvas-source-select").addEventListener("change",d=>{if(d.target.value===
"")return;const u=Number(d.target.value);Number.isInteger(u)&&applyCanvasSelection(u,{view:"source"})}),
get("canvas-panel-tabs")&&get("canvas-panel-tabs").addEventListener("click",d=>{const u=d.target.closest(
"[data-canvas-panel-view]");if(!u)return;const g=u.getAttribute("data-canvas-panel-view");syncCanvasPanelViewUi(
g,{focus:!1})}),get("canvas-panel-copy-btn")&&get("canvas-panel-copy-btn").addEventListener("click",
()=>{const d=getCanvasModeElements(),u=d&&d.code&&d.code.textContent||"";if(!u.trim()){showToast("\u30B3\u30D4\
\u30FC\u3059\u308B\u30B3\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","info",!1);return}copyToClipboard(
u,()=>showToast("Canvas\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC\u3057\u307E\u3057\u305F","success"),
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const te=get(
"prompt-controls-toggle-btn");te&&(te.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,u,g){const h=(u||"").match(/\S*/)[0];if(h==="py\
exec")return"";if(h==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",S=(h||"").toLowerCase();
let L="";try{const H=hljs.getLanguage(h)?h:"plaintext";activeStreamingBubbleId&&x.length>2e4?L=escapeHtml(
x):L=hljs.highlight(x,{language:H}).value}catch{L=escapeHtml(x)}const A=encodeURIComponent(x).replace(
/'/g,"%27"),E=detectBlockedScriptsInCode(x),R=hashString(`${h||"TEXT"}
${x||""}`);let B="";if(canvasModeEnabled){const H=String(canvasPreviewState.selectedKey||"")===R,me=H?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";B=`<button\
 class="canvas-preview-btn${H?" canvas-active":""}" data-code="${A}" data-code-key="${R}" data-canva\
s-lang="${escapeHtml(h||"txt")}" title="${me}" aria-label="${me}" aria-pressed="${H?"true":"false"}"\
><i class="fas ${H?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
S,x)){const H=E?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
B=`<button class="html-preview-btn" data-code="${A}" ${E?'data-suspicious="1"':""} title="${H}" aria\
-label="${H}"><i class="fas ${E?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const V=`\
<button class="download-btn" data-code="${A}" data-lang="${h||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,ue=S==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${R}" data-coding-lang="${escapeHtml(h||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,oe=(h||"TEXT")+(E?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),J=getRenderableSvgCode(S,x);return`${J?buildSvgCodeRenderHtml(J,R):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${R}"><div class="code-header"\
><span class="code-lang">${oe}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${ue}${B}${V}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${h}">${L}</code></pre></d\
iv></div>`},link(d,u,g){return`<a href="${d}" title="${u||""}" target="_blank">${g}</a>`},image(d,u,g){
return buildChatImageHtml(d,{alt:g,title:u})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(gi=get("version-update-dismis\
s"))==null||gi.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const X=get("version-update-clear-\
cache");if(X&&(X.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),X.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(X.checked)})),
(hi=get("version-update-reload"))==null||hi.addEventListener("click",async()=>{var u;await versionUpdateCachePreferenceSavePromise.
catch(()=>{}),!!((u=get("version-update-clear-cache"))!=null&&u.checked)?await clearSiteCacheAndReload(
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
d)}}else{const d=get("turnstile-container");d&&d.classList.add("hidden")}const we=o(d=>{if(!d)return"\
\u4E0D\u660E";const u=new Date(d);return Number.isNaN(u.getTime())?d:u.toLocaleString()},"formatSess\
ionTime"),q=o(d=>{const u=Array.isArray(d)?d:[],g=get("passkey-list"),h=get("passkey-count");if(h&&(h.
innerText=String(u.length)),!!g){if(!u.length){g.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
g.innerHTML="",u.forEach((x,S)=>{const L=x&&x.id?String(x.id):"",A=document.createElement("div");A.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const E=document.
createElement("div");E.className="min-w-0";const R=document.createElement("div");R.className="text-x\
s text-gray-200 truncate",R.innerText=x&&x.name?String(x.name):`Security Key ${S+1}`;const B=document.
createElement("div");B.className="text-[10px] text-gray-500 mt-1",B.innerText=x&&x.created_at?`\u767B\u9332\u65E5\u6642:\
 ${we(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",E.appendChild(R),E.appendChild(B),A.appendChild(
E);const V=document.createElement("button");V.type="button",V.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",V.innerText="\u524A\u9664",V.
disabled=!L,L&&(V.onclick=()=>window.removeWebAuthnCredential(L)),A.appendChild(V),g.appendChild(A)})}},
"renderPasskeyList"),Y=o(d=>{const u=get("session-list");if(u){if(!d||!d.length){u.innerHTML='<div c\
lass="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}u.innerHTML=d.map(g=>{const h=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",S=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
L=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${h}${x}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(L)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(we(g.last_seen_at))}\
 / \u4F5C\u6210: ${escapeHtml(we(g.created_at))}</div></div>${S}</div>`}).join(""),u.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const h=g.getAttribute("data-session-id");if(!h||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:h})});let S={};try{S=await x.json()}catch{}if(x.ok){if(S.logged_out){
location.href="/login";return}await ie()}else showToast(S&&S.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),ie=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div \
class="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const u=await apiFetch("/api\
/sessions");let g={};try{g=await u.json()}catch{}if(!u.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const h=(g.sessions||[]).filter(x=>!x.is_revoked);Y(h)},"loadSessions"),xe=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>ie());const u=get("session-revoke-others-btn");u&&(u.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await ie():showToast("\u64CD\u4F5C\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(d=>{d&&(currentVisionModel=d.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
d);try{loadMcpServers()}catch{}d&&d.theme_color&&applyThemeColor(d.theme_color,!0),d&&Object.prototype.
hasOwnProperty.call(d,"minimal_prompt_mode")&&d.minimal_prompt_mode?setMinimalPromptMode(!0):d&&Object.
prototype.hasOwnProperty.call(d,"compact_prompt_mode")&&setCompactPromptMode(!!d.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings sync");
const u=get("enable-sys-prompt");u&&d&&d.system_prompt&&String(d.system_prompt).trim()&&(!u.disabled&&
!d.default_enable_system_prompt&&!d.use_last_chat_settings&&(u.checked=!0),T())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const pe=get("vision-model-change-btn");pe&&(pe.onclick=()=>_openVisionModelSelector());const Ae=get(
"compression-format-only");Ae&&(Ae.onchange=()=>{const d=Ae.checked,u=get("compression-max-size"),g=get(
"compression-max-dim");u&&(u.disabled=d),g&&(g.disabled=d);const h=get("compression-size-wrap"),x=get(
"compression-dim-wrap");h&&(h.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const I=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const u=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(u),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");I(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const N=get("storage-us\
age-refresh");N&&(N.onclick=()=>loadStorageUsage());let D=null;const O=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,u=>u.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),$=o((d={})=>{const u=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),h=get("account-transfer-progress-percent"),x=get("account-transfer-progress-text"),
S=get("account-transfer-progress-detail"),L=Math.max(0,Math.min(100,Number(d.progress)||0));if(u&&u.
classList.remove("hidden"),g&&(g.style.width=`${L}%`),h&&(h.textContent=`${Math.round(L)}%`),x&&(x.textContent=
d.message||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059"),S){const E={queued:"\
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
completed:"\u5B8C\u4E86",failed:"\u5931\u6557"};S.textContent=E[d.phase]||"\u51E6\u7406\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059\u3002"}
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),C=o(d=>{ne&&(ne.disabled=
!!d);const u=get("account-import-btn");u&&(u.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),j=o((d={})=>{const u=get("account-export-ready"),g=get(
"account-export-ready-text"),h=get("account-export-expiry"),x=get("account-export-download-btn"),S=!!(d.
available&&d.download_url);if(u&&u.classList.toggle("hidden",!S),!S){x&&x.removeAttribute("href");return}
const L=Math.max(0,Number(d.size_bytes)||0),A=L>=1024*1024*1024?`${(L/(1024*1024*1024)).toFixed(2)} \
GB`:`${(L/(1024*1024)).toFixed(1)} MB`;if(g){const E=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${E}`}
if(h){const E=d.expires_at?new Date(d.expires_at):null;h.textContent=E&&!Number.isNaN(E.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${E.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),Q=o(async d=>{for(;D===d&&!d.stopped;){
try{const u=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await u.json().catch(()=>({}));if(u.ok&&(g.state!=="pending"&&$(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(u=>setTimeout(u,700))}
return null},"pollAccountTransfer"),se=o((d,u,g=!0)=>{u&&($(u),j(u),g&&u.state==="ready"?showToast(u.
message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(u.unreadable_count)>0?"warning":"success",Number(u.unreadable_count)>0):g&&u.state==="failed"&&
showToast(u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),Te(d))},"handleFinishedAccountExport"),re=o(async()=>{try{const d=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),u=await d.json().catch(()=>({}));
if(!d.ok)return;if(j(u),u.state==="ready"){$(u);return}if(["failed","cancelled","expired"].includes(
u.state)){$(u);return}if(!["queued","running","cancelling"].includes(u.state)||!u.job_id||D&&D.id===
u.job_id||D)return;const g={id:u.job_id,type:"export",stopped:!1,restored:!0};D=g,C(!0),$(u);const h=await Q(
g);h&&se(g,h,!0)}catch{}},"refreshLatestAccountExport"),Te=o(d=>{D===d&&(D=null),d.stopped=!0,C(!1)},
"finishAccountTransfer"),ae=get("account-transfer-cancel-btn");ae&&(ae.onclick=async()=>{const d=D;if(!(!d||
d.stopped)){d.cancelRequested=!0,ae.disabled=!0,$({progress:0,phase:"cancelling",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});
try{await apiFetch(`/api/account/transfer/${d.id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}
d.controller&&d.controller.abort(),$({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
d.type==="export"&&j({available:!1}),Te(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const ne=get("account-export-btn");ne&&(ne.onclick=async()=>{if(D)return;const d={id:O(),
type:"export",stopped:!1};D=d,C(!0),j({available:!1}),$({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u53D7\
\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const u=await apiFetch("/api/account/export",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:d.id}),keepalive:!0})),
g=await u.json().catch(()=>({}));if(u.status===409&&g.error==="export_in_progress"&&g.job_id)d.id=g.
job_id;else if(!u.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":
g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
$({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const h=await Q(d);!d.cancelRequested&&h&&se(d,h,!0)}catch(u){const g=u&&u.message?u.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";$({progress:0,phase:"failed",message:g}),
showToast(g,"error",!0),Te(d)}});const Z=get("account-export-download-btn");Z&&Z.addEventListener("c\
lick",async d=>{const u=Z.getAttribute("href");if(!(!u||u==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),h=await g.json().catch(
()=>({}));g.ok&&h.available&&h.download_url?(Z.href=h.download_url,window.location.assign(h.download_url)):
(j(h),$(h),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),re())}catch{window.location.assign(u)}}}),C(!1),re();const ke=get("import-files-grid"),
rt=get("import-files-info"),dt=get("import-files-summary"),mt=o(d=>{const u=Math.max(0,Number(d)||0);
return u>=1024*1024*1024?`${(u/(1024*1024*1024)).toFixed(2)} GB`:u>=1024*1024?`${(u/(1024*1024)).toFixed(
1)} MB`:u>=1024?`${Math.round(u/1024)} KB`:`${u} B`},"importFormatBytes");let Ne=null;const Ve=o(()=>{
if(!Ne)return;const d=Ne.files,u=Ne.selection;let g=0;d.forEach(S=>{u.has(S.archive_path)&&(g+=Number(
S.size_bytes)||0)});const h=Number(Ne.available_bytes)||0,x=g>h;dt&&(dt.textContent=`\u9078\u629E\u4E2D: ${mt(
g)} / \u5229\u7528\u53EF\u80FD: ${mt(h)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,dt.classList.
toggle("text-red-300",x)),rt&&(rt.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),bt=o(
()=>{if(!ke||!Ne)return;ke.innerHTML="";const d=Ne.files;if(!d.length){ke.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
Ve();return}d.forEach(u=>{const g=document.createElement("label"),h=Ne.selection.has(u.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${h?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${h?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(u.display_name)}">${escapeHtml(
u.display_name)}</div><div class="text-[10px] text-gray-500">${mt(u.size_bytes)}</div></div>`;const x=g.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?Ne.selection.add(u.archive_path):
Ne.selection.delete(u.archive_path),g.classList.toggle("border-blue-500",x.checked),g.classList.toggle(
"border-gray-600",!x.checked),Ve()}),ke.appendChild(g)}),Ve()},"renderImportFileItems"),Ct=o(d=>new Promise(
u=>{if(Ne={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:u},bt(),!get("import-files-modal")){u(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),wt=o(d=>{if(hideModal("import-files-modal"),Ne){const u=Ne.resolve;
Ne=null,u(d)}},"closeImportFileSelection"),Mt=get("import-files-close");Mt&&(Mt.onclick=()=>wt(null));
const yt=get("import-files-cancel");yt&&(yt.onclick=()=>wt(null));const Nt=get("import-files-confirm");
Nt&&(Nt.onclick=()=>{if(!Ne)return;const d=Array.from(Ne.selection);wt(d.length?d.join(","):"__none_\
_")});const At=get("import-files-select-all");At&&(At.onclick=()=>{Ne&&(Ne.files.forEach(d=>Ne.selection.
add(d.archive_path)),bt())});const ft=get("import-files-none");ft&&(ft.onclick=()=>{Ne&&(Ne.selection.
clear(),bt())});const Rt={system_prompt:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",system_prompt_enabled:"\
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},Vt=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const u=String(
d);return u.length>60?u.slice(0,60)+"\u2026":u},"formatAccountSettingValue");let xt=null;const vt=o(
d=>{if(xt){const u=xt;xt=null,hideModal("settings-confirmation-modal"),u(d)}},"resolveSettingsImport\
Confirmation"),Jt=o(d=>new Promise(u=>{if(!get("settings-confirmation-modal")){u(!0);return}xt=u;const h=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(h.length?
x.innerHTML=h.map(L=>{const A=Rt[L.field]||L.field,E=Vt(L.current),R=Vt(L.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
E)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(R)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const S=get("settin\
gs-confirmation-count");S&&(S.textContent=`${h.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Bt=get("settings-confir\
mation-modal");Bt&&Bt.addEventListener("click",d=>{d.target===Bt&&vt(!1)});const U=get("settings-con\
firmation-close");U&&(U.onclick=()=>vt(!1));const ge=get("settings-confirmation-cancel");ge&&(ge.onclick=
()=>vt(!1));const Ie=get("settings-confirmation-confirm");Ie&&(Ie.onclick=()=>vt(!0));const Re=get("\
account-import-btn"),Je=get("account-import-inplace"),tt=get("account-import-inplace-warning");if(Je&&
tt){const d=o(()=>tt.classList.toggle("hidden",!Je.checked),"syncInplaceWarn");Je.addEventListener("\
change",d),d()}Re&&(Re.onclick=async()=>{const d=get("account-import-file"),u=d&&d.files?d.files[0]:
null,g=get("account-import-categories"),h=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(oe=>oe.value):[],x=get("account-import-inplace"),S=!!(x&&x.checked),L=get("account-impo\
rt-settings-bypass"),A=!!(L&&L.checked);let E=!1;if(!u){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!h.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const R=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(oe=>(oe.
closest("label")&&oe.closest("label").textContent||oe.value).trim()):h;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${R.join("\u3001")}${S?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const B={id:O(),type:"import",stopped:!1,controller:new AbortController};
D=B,C(!0),$({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const V=get("account-import-result");let ue=Promise.resolve(null);try{const J=Math.max(1,Math.ceil(u.
size/10485760)),de=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:u.size}),signal:B.controller.
signal})),H=await de.json().catch(()=>({}));if(!de.ok)throw new Error(H.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
B.uploadId=H.upload_id;const me=H.chunk_size||10485760;let W=0,ce=0;const he=o(async()=>{for(;;){const _e=ce++;
if(_e>=J)return;const be=u.slice(_e*me,Math.min(u.size,(_e+1)*me)),Be=new FormData;Be.append("chunk",
be,u.name),Be.append("index",String(_e));const Ye=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Be,signal:B.controller.signal})),
at=await Ye.json().catch(()=>({}));if(!Ye.ok)throw new Error(at.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W++,$({progress:Math.min(35,Math.round(W/J*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${W}\
/${J}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ee=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ee=!0);try{await Promise.
all([he(),he(),he()]);const _e=await apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:B.controller.signal})),be=await _e.json().
catch(()=>({}));if(!_e.ok)throw new Error(be.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
$({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ee&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let fe="",Ee=!1,$e=0;const st=o(
async()=>{let _e=!1;const be=o(()=>{_e||(_e=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Be=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Ye=await Be.
json().catch(()=>null);if(!Be.ok||!Ye){be();return}cacheUserSettings(Ye);const at=get("settings-moda\
l");if(at&&at.classList.contains("modal-open"))try{Gn(Ye)}catch{}Ye.theme_color&&applyThemeColor(Ye.
theme_color,!0),Object.prototype.hasOwnProperty.call(Ye,"minimal_prompt_mode")&&Ye.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Ye,"compact_prompt_mode")&&setCompactPromptMode(
!!Ye.compact_prompt_mode)}catch{}be()},"refreshSettingsFormAfterImport"),Ue=o(_e=>{const be=_e&&_e.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";V&&(V.textContent=`\u5B8C\u4E86: ${be}`,
V.classList.remove("hidden","text-red-300"),V.classList.add("text-emerald-300")),$({progress:100,phase:"\
completed",message:be}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&st()},"finishImportSuccess"),ut=o(async()=>{
try{const be=await(await apiFetch(`/api/account/transfer/${B.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return be&&be.state?be:null}catch{return null}},"fetchImportSta\
tus"),Tt=o(async()=>{const _e=await ut();if(!_e)return{status:"unknown"};if(_e.state==="completed")return Ue(
_e),{status:"done"};if(["failed","cancelled","expired"].includes(_e.state))throw new Error(_e.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(_e.state==="needs_sel\
ection"&&Array.isArray(_e.files)){const be=await Ct({files:_e.files,available_bytes:_e.available_bytes});
return be===null?($({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(fe=be,{status:"reselect"})}if(_e.state===
"needs_settings_confirmation"&&Array.isArray(_e.settings_changes))return await Jt({settings_changes:_e.
settings_changes})?(E=!0,{status:"reselect"}):($({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(_e.state==="running"){const be=await Promise.race([ue.catch(()=>null),new Promise(Be=>setTimeout(
()=>Be(null),6e4))]);if(be&&be.state==="completed")return Ue(be),{status:"done"};throw be&&["failed",
"cancelled","expired"].includes(be.state)?new Error(be.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ee;){B.stopped=!0,await ue.catch(()=>null),
B.stopped=!1,ue=Q(B);let _e;try{_e=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:B.uploadId,
categories:h.join(","),job_id:B.id,selected_files:fe,restore_inplace:S,confirm_settings:E||A}),signal:B.
controller.signal}))}catch(ze){if(B.cancelRequested||ze&&ze.name==="AbortError")throw ze;const ct=await Tt();
if(ct.status==="done"){Ee=!0;break}if(ct.status==="cancelled")return;if(ct.status==="reselect")continue;
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let be=null;try{be=await _e.json()}catch{be=null}if(be===null){const ze=await Tt();if(ze.status==="d\
one"){Ee=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;if(_e.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!_e.ok&&be.error==="storage_limit_files"&&be.files){const ze=await Ct(be);if(ze===null){$({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}fe=ze;continue}if(be&&be.status==="settings_confirmation"&&
Array.isArray(be.settings_changes)){if(!await Jt(be)){$({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}E=!0;continue}if(!_e.ok)throw new Error(be.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Be=be.imported||{},Ye=[`\u8A2D\u5B9A ${Be.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Be.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Be.chats||0}\u4EF6`,`Gem ${Be.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Be.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Be.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Be.
diagnostics||0}\u4EF6`].join(" / "),at=be.duplicates||{},ve={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Pe=[];for(const ze of Object.keys(ve)){const ct=Number(at[ze])||0;ct>
0&&Pe.push(`${ve[ze]} ${ct}\u4EF6`)}const He=Pe.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Pe.
join("\u3001")}\uFF09`:"";V&&(V.textContent=`\u5B8C\u4E86: ${Ye}${He}`,V.classList.remove("hidden","\
text-red-300"),V.classList.add("text-emerald-300")),$({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&st(),Ee=!0}}catch(oe){if(B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),B.cancelRequested||oe&&oe.name==="AbortError")return;const J=oe&&oe.message?
oe.message:"",de=J==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
J||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";$({progress:0,phase:"f\
ailed",message:de}),V&&(V.textContent=de,V.classList.remove("hidden","text-emerald-300"),V.classList.
add("text-red-300")),showToast(de,"error",!0)}finally{B.stopped=!0,await ue.catch(()=>null),Te(B)}});
const lt=get("account-dedupe-btn"),qe=get("account-dedupe-result"),je=o((d,u=!1)=>{qe&&(qe.textContent=
d,qe.classList.remove("hidden"),qe.classList.toggle("text-red-300",!!u),qe.classList.toggle("text-em\
erald-300",!u))},"showDedupeResult");lt&&(lt.onclick=async()=>{const d=o(async()=>{const u=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await u.json().catch(()=>null);if(!u.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){je("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const h=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const B of[
"chats","gems","files","feedback","diagnostics"]){const V=Number(g.duplicates&&g.duplicates[B])||0;V>
0&&h.push(`${x[B]} ${V}\u4EF6`)}const S=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${h.join("\u3001")}${S}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const L=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await L.json().catch(()=>null);if(!L.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const E=[];for(const B of["chats","gems","files","feedback","diagnostics"]){const V=Number(A.removed&&
A.removed[B])||0;V>0&&E.push(`${x[B]} ${V}\u4EF6`)}const R=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";je(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${E.join("\u3001")||"0\u4EF6"}${R}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!lt.disabled){lt.disabled=!0,je("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(u){je(u&&u.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{lt.disabled=!1}}});const Ze=get("site-cache-usage-refresh");Ze&&(Ze.onclick=()=>loadSiteCacheUsage());
const We=get("clear-site-cache-btn");We&&(We.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(We)});
const gt=get("enc-scan-result"),Ft=o(async(d=null)=>{gt&&(gt.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let u="/api/encryption_scan";d&&(u+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
u,{cache:"no-store"}),h=await g.json();if(!g.ok){gt&&(gt.textContent=h.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=h.total||0,S=h.encrypted||0,L=h.unencrypted||0;let A=`Total: ${x} / Encrypted: ${S} /\
 Plain: ${L}`;if(h.samples&&h.samples.length){const E=h.samples.slice(0,8).map(R=>{const B=R.timestamp?
new Date(R.timestamp).toLocaleString():"";return`#${R.id} (${R.role||""}) ${B}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${E}</div>`}gt&&(gt.innerHTML=A)}catch{gt&&(gt.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Kt=get("enc-scan-all");Kt&&(Kt.onclick=()=>Ft(
null));const Xt=get("enc-scan-thread");Xt&&(Xt.onclick=()=>currentThreadId?Ft(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Ce=get("admin-enc-\
list");let De=null,Ke=!1;const un=o(d=>!d||!d.length?null:d.some(u=>!!u.is_encrypted),"computeThread\
EncryptedFromMessages"),pn=o(()=>{De=un(allMessages)},"refreshCurrentThreadEncStateFromMessages"),_t=o(
async(d,u,{confirmPrompt:g=!0,reloadCurrent:h=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=u?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Ke=!0;try{const S=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:u})}),
L=await S.json().catch(()=>({}));return S.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${L.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),De=!!u,h&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Ce&&await Le(),!0):
(showToast(L.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Ke=!1}},"setAdminThreadEncry\
ption"),ye=o(d=>{if(!Ce)return;const u=d.threads||[];if(!u.length){Ce.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Ce.innerHTML=u.map(g=>{const h=g.encrypted_count>0?"enc":"plain",x=h==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",S=h==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",L=g.
updated_at?new Date(g.updated_at).toLocaleString():"",A=escapeHtml(String(g.thread_id)),E=currentThreadId&&
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
xt-white px-2 py-1 rounded shrink-0" data-id="${A}" title="\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F"><i class="fas fa-external-link\
-alt mr-1"></i>\u958B\u304F</button>
                        <button type="button" class="admin-enc-toggle ${S} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${h==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),Le=o(async()=>{if(Ce){Ce.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),u=await d.json().catch(()=>({}));
if(!d.ok){Ce.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(u.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(ye(u),currentThreadId&&Array.isArray(u.threads)){const g=u.threads.find(h=>String(
h.thread_id)===String(currentThreadId));g&&(De=!!g.encrypted)}}catch{Ce.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>Le()),window.__loadAdminEncThreads=
Le,window.__refreshAdminThreadEncState=pn,window.__setAdminThreadEncryption=_t;const Xe=get("encrypt\
ion-status-admin-toggle");Xe&&Xe.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Ce&&(Ce.onclick=async d=>{const u=d.target.closest("\
.admin-enc-open");if(u){d.preventDefault();const A=u.getAttribute("data-id");if(!A)return;typeof jt==
"function"?jt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||Ke)return;const h=g.getAttribute("data-i\
d"),x=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const L=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await _t(h,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=L,await Le()}}),get("file-input").onchange=d=>{const u=Array.from(d.target.
files||[]);d.target.value="",u.length&&handleFiles(u)},get("photo-input")&&(get("photo-input").onchange=
d=>{const u=Array.from(d.target.files||[]);d.target.value="",u.length&&handleFiles(u)});const ot=o(d=>{
const u=get("ban-appeal-list");if(u){if(!d||!d.length){u.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
u.innerHTML=d.map(g=>{const h=g.status||"new",x=g.admin_read_at?'<span class="text-[10px] text-gray-\
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
                    `}).join("")}},"renderBanAppeals"),nt=o(async(d=!1)=>{if(!isAdminUser)return;const u=get(
"ban-appeal-count");if(u)try{const g=await apiFetch("/api/ban/appeals/summary",{cache:"no-store"});if(!g.
ok)return;const x=(await g.json()).unread_count||0;u.textContent=String(x),d&&x>0&&showToast(`BAN\u7570\u8B70\u7533\
\u3057\u7ACB\u3066\u304C${x}\u4EF6\u3042\u308A\u307E\u3059\u3002`,"success")}catch{}},"refreshBanApp\
ealSummary"),ht=o(async()=>{if(!isAdminUser)return;const d=get("ban-appeal-list");if(d){d.innerHTML=
'<div class="text-[11px] text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';try{const u=await apiFetch(
"/api/ban/appeals?limit=80",{cache:"no-store"});if(!u.ok)return;const g=await u.json();ot(g.items||[]),
await nt(!1)}catch{}}},"loadBanAppeals"),St=o(async(d=null)=>{if(!isAdminUser)return;const u=d?{ids:d}:
{all:!0};try{(await apiFetch("/api/ban/appeals/mark_read",{method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify(u)})).ok&&await ht()}catch{}},"markBanAppealsRead"),_n=o(async d=>{
if(isAdminUser)try{(await apiFetch("/api/ban/appeals/update",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify(d)})).ok&&await ht()}catch{}},"updateBanAppealStatus"),yi=o(()=>{
const d=get("tab-general");if(!d||get("temp-chat-settings-card"))return;const u=document.createElement(
"div");u.id="temp-chat-settings-card",u.className="settings-card",u.innerHTML=`
                    <h3 class="settings-card-title">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8</h3>
                    <div class="space-y-3 text-xs text-gray-300">
                        <label class="text-xs text-gray-500 block">\u5207\u65AD\u30BF\u30A4\u30E0\u30A2\u30A6\u30C8\uFF08\u79D2\uFF09</label>
                        <input id="set-temp-chat-timeout-seconds" type="number" min="${TEMP_CHAT_TIMEOUT_MIN_SECONDS}\
" max="${TEMP_CHAT_TIMEOUT_MAX_SECONDS}" step="1" class="w-28 bg-gray-800 border border-gray-600 rou\
nded px-2 py-1 text-xs text-white">
                        <div class="text-[10px] text-gray-500">\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3067\u30DA\u30FC\u30B8\u306E\u8868\u793A/\u63A5\u7D9A\u304C\u9014\u5207\u308C\u305F\u72B6\u614B\u304C\u3053\u306E\u79D2\u6570\u3092\u8D85\u3048\u308B\u3068\u3001\u81EA\u52D5\u524A\
\u9664\u3055\u308C\u307E\u3059\u3002</div>
                    </div>
                `,d.appendChild(u)},"ensureTemporaryChatSettingsCard"),Dn=o(()=>{const d=get("set-st\
t-model");if(!d||get("set-llm-transcribe-prompt"))return;const u=d.closest(".space-y-2");if(!u)return;
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
                `,u.appendChild(g);const h=get("reset-llm-transcribe-prompt");h&&(h.onclick=()=>{const x=get(
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
mcpLocked:!0}];window.buildAutoSystemPromptRows=(d,u=!1)=>{const g=u?"w-full h-14 bg-gray-950 border\
 border-gray-700 rounded p-2 text-[11px] text-gray-200":"w-full h-20 bg-gray-950 border border-gray-\
700 rounded p-2 text-xs text-gray-200";return mn.map(h=>{const x=h.mcpLocked===!0,S=x?'<div class="t\
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
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,u={})=>{mn.forEach(g=>{const h=u&&
typeof u=="object"?u[g.key]||{}:{},x=get(`${d}-auto-sys-${g.key}-enabled`),S=get(`${d}-auto-sys-${g.
key}-text`);x&&(g.mcpLocked===!0?x.disabled=!0:x.checked=h.enabled!==!1),S&&(S.value=h.text||"",S.placeholder=
h.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Hn=o((d,u=null)=>{if(u){const g=get(u);g&&(g.checked=!0)}mn.forEach(g=>{const h=get(`${d}-auto\
-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);if(h&&(g.mcpLocked!==!0?h.checked=!0:h.
disabled=!0),x){const S=x.placeholder||"";x.value=S}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
"resetAutoSystemPromptConfigToCodeDefaults"),qn=o(d=>{const u={};return mn.forEach(g=>{const h=get(`${d}\
-auto-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);u[g.key]={enabled:g.mcpLocked===!0?
!0:h?h.checked:!0,text:x?x.value:""}}),u},"collectAutoSystemPromptConfigFromForm");window.collectAutoSystemPromptConfigFromForm=
qn,window.ensureAutoSystemPromptSettingsCard=()=>{const d=get("set-global-sys-prompt-enabled"),u=d?d.
closest(".space-y-4"):null;if(!u||get("auto-sys-prompt-settings"))return;const g=document.createElement(
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
                `,u.appendChild(g)},window.ensureThreadAutoSystemPromptCard=()=>{const d=get("thread\
-global-sys-prompt"),u=d?d.closest(".space-y-3"):null;if(!u||get("thread-auto-sys-prompt-settings"))
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
                `,u.appendChild(g)},yi(),Dn(),I();const vi=o(()=>{const d=get("set-default-model");if(!d)
return;const u=d.value;d.innerHTML="",MODELS.forEach(h=>{const x=document.createElement("optgroup");
x.label=h.category,(h.items||[]).forEach(S=>{const L=document.createElement("option");L.value=S.id,L.
textContent=S.name,x.appendChild(L)}),d.appendChild(x)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||u||"gemini-3.6-flash";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"po\
pulateDefaultModelOptions"),wi=o(()=>{const d=get("set-default-vision-model");if(!d)return;const u=d.
value;d.innerHTML="",MODELS.forEach(h=>{const x=(h.items||[]).filter(L=>{const A=(L.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(x.length===0)return;const S=document.createElement("optgroup");S.label=h.category,x.forEach(
L=>{const A=document.createElement("option");A.value=L.id,A.textContent=L.name+" \u2605",S.appendChild(
A)}),d.appendChild(S)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||u||
"gemini-3-flash-preview";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"populateDefaul\
tVisionModelOptions"),Gn=o(d=>{if(!d)return;cacheUserSettings(d);const u=get("app-global-sys-prompt-\
preview");u&&(u.value=d.global_system_prompt_effective||"");const g=get("app-global-sys-prompt-previ\
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
"set-passkey-only-login"),A=get("passkey-only-note"),E=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(q(E),L){L.checked=!!d.passkey_only_login;const oe=E.length>0||!!d.has_webauthn;L.disabled=!oe,
oe||(L.checked=!1),A&&(oe?A.classList.add("hidden"):A.classList.remove("hidden"))}const R=get("mig-s\
tatus-box"),B=get("mig-progress-text"),V=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){R.classList.remove("hidden");const oe=(d.migration_progress||"").split("/");if(oe.length===
2){const J=parseInt(oe[0]||"0",10),de=parseInt(oe[1]||"0",10);B&&(B.innerText=`${J} / ${de}`),V&&de>
0&&(V.style.width=`${Math.min(100,Math.floor(J/de*100))}%`)}}else R.classList.add("hidden"),V&&(V.style.
width="0%"),B&&(B.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&Gn(d);const u=get(
"search-box"),g=u?u.value:"";clearTimeout(searchTimeout);const h=get("settings-search");if(h&&(h.value=
""),filterSettings(),vi(),wi(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),re(),Dn(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),nt(!0),ht(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),en(),xe(),ie();try{loadMcpServers()}catch{}};const jt=o((d=!1)=>{snapshotSidebarHistory(
"settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),xi=o(()=>{const d=get("set-th\
eme-color"),u=get("set-theme-color-text"),g=get("theme-reset-btn"),h=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((S,L=!0)=>{const A=normalizeHex(S);A&&(applyThemeColor(A,L),syncThemeInputs(
A))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),u&&(u.addEventListener("chan\
ge",()=>{const S=normalizeHex(u.value);if(!S){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(S,!0)}),u.addEventListener("keydown",S=>{S.key==="Enter"&&(S.preventDefault(),
u.blur())})),g&&(g.onclick=()=>x(THEME_DEFAULT,!0)),h.forEach(S=>{S.addEventListener("click",()=>x(S.
getAttribute("data-color"),!0))})},"bindThemeControls"),ki=o(()=>{const d=get("reset-global-sys-prom\
pt");d&&(d.onclick=()=>{get("sys-prompt-text")&&(get("sys-prompt-text").value=""),get("set-global-sy\
s-prompt-enabled")&&(get("set-global-sys-prompt-enabled").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\
\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09","success")});const u=get(
"reset-thread-sys-prompt");u&&(u.onclick=()=>{get("thread-global-sys-prompt")&&(get("thread-global-s\
ys-prompt").value=""),get("thread-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enable\
d").checked=!1),showToast("\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const g=get("reset-set-auto-sys-prompt-defaults");g&&(g.onclick=()=>{Hn("set","set-appl\
y-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")});const h=get("reset-thread-auto-sys-prompt-defaults");h&&(h.onclick=()=>{Hn("thread","th\
read-apply-auto-sys-prompt-notices"),showToast("\u81EA\u52D5\u6CE8\u5165\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u65E2\u5B9A\u5024\u306B\u623B\u3057\u307E\u3057\u305F\uFF08\u4FDD\u5B58\u3057\u3066\u304F\u3060\u3055\u3044\uFF09",
"success")})},"bindSystemPromptControls");get("settings-btn").onclick=()=>{openSettingsModal()},get(
"close-settings-btn").onclick=()=>jt();const Un=get("settings-header-close");Un&&(Un.onclick=()=>jt());
const Dt=get("settings-search");Dt&&(Dt.addEventListener("input",filterSettings),Dt.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const u=get("tab-"+activeSettingsTab);if(!u)return;const g=u.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const zn=get(
"settings-search-clear");zn&&zn.addEventListener("click",()=>{Dt&&(Dt.value="",filterSettings(),Dt.focus())}),
xi(),ki(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Sn=get("set-gemini-local-python-dialog");Sn&&(Sn.onchange=()=>setGeminiLocalPyDialogEnabled(Sn.
checked));const Wn=get("set-gemini-backend");Wn&&(Wn.onchange=()=>syncGeminiBackendUi());const Vn=get(
"set-admin-api-key-mode");Vn&&(Vn.onchange=()=>syncAdminApiKeyModeUi());const Ln=get("set-temp-chat-\
timeout-seconds");Ln&&(Ln.onchange=()=>{applyTemporaryChatTimeoutSeconds(Ln.value)});const Jn=get("s\
lash-command-cancel-btn");Jn&&(Jn.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
t-input");d&&d.focus()}),syncGeminiBackendUi(),syncAdminApiKeyModeUi(),get("save-settings-btn").onclick=
async()=>{if(!settingsModalLoaded){showToast("\u8A2D\u5B9A\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u3059\u308B\u307E\u3067\u304A\u5F85\u3061\u304F\u3060\u3055\u3044",
"error",!0);return}const d=get("set-username"),u=get("set-password"),g=readPromptBarModeFromForm(),h={
system_prompt:get("sys-prompt-text")?get("sys-prompt-text").value:"",system_prompt_enabled:get("set-\
global-sys-prompt-enabled")?get("set-global-sys-prompt-enabled").checked:!0,apply_global_system_prompt:get(
"set-apply-global-sys-prompt")?get("set-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"set-apply-auto-sys-prompt-notices")?get("set-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:qn(
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
-method").value:"totp",new_username:d?d.value:null,new_password:u?u.value:null},x=get("set-e2ee")?get(
"set-e2ee").checked:!1,S=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
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
try{const V=await L.json();V&&V.message&&(A=V.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!h.light_mode_enabled,liquid_glass_enabled:!!h.liquid_glass_enabled})),window.
applySavedUserSystemPromptSettings({system_prompt:h.system_prompt,system_prompt_enabled:h.system_prompt_enabled,
apply_global_system_prompt:h.apply_global_system_prompt,apply_auto_system_prompt_notices:h.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:h.auto_system_prompt_notices_config}),jt();const E=currentUsername,
R=CHAT_CONFIG.enableE2EE;enterToSend=h.enter_to_send,autoSearchOnLinks=h.auto_search_on_links;const B=useSwCache;
useSwCache=h.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!h.clear_cache_on_version_update),
compactPromptMode=h.compact_prompt_mode,minimalPromptMode=h.minimal_prompt_mode,voiceStudioUiEnabled=
h.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=h.temp_chat_timeout_seconds,applyThemeColor(h.theme_color,
!0),syncThemeInputs(h.theme_color),applyLightMode(h.light_mode_enabled),applyLiquidGlassMode(h.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),B!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(A,"success"),syncClientDebugLogToggle(
h.enable_client_debug_log,"settings saved"),h.new_username&&h.new_username!==E?setTimeout(()=>location.
reload(),1e3):h.new_password&&showToast("\u30D1\u30B9\u30EF\u30FC\u30C9\u3092\u5909\u66F4\u3057\u307E\u3057\u305F\u3002\u6B21\u56DE\u30ED\u30B0\u30A4\u30F3\u6642\u304B\u3089\u6709\u52B9\u3067\u3059\u3002",
"info")}else{let A={};try{A=await L.json()}catch{}showToast(A.error||"\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},get("disable-2fa-btn").onclick=async()=>{if(confirm("Disable 2FA?"))if((await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({disable_2fa:!0})})).ok){showToast("2FA\u3092\u7121\u52B9\u5316\u3057\u307E\u3057\u305F","\
success"),get("disable-2fa-btn").classList.add("hidden");const u=get("2fa-badge");u&&(u.innerText="D\
ISABLED",u.className="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-400")}else showToast(
"2FA\u306E\u7121\u52B9\u5316\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)},get("bot-unban-\
btn")&&(get("bot-unban-btn").onclick=async()=>{const d=get("bot-unban-username"),u=d?d.value.trim():
"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"single"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${u} \u306EBAN\u3092\u5358\u72EC\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),u=d?d.value.trim():"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"linked"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const S=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(S,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),u=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),u&&(u.classList.remove("hidden"),u.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(W=>{u&&(u.textContent=W)},"setBox"),h=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((W,ce)=>!W||!ce||ce<=0?0:W*8/(ce/1e3)/1e3/1e3,"toMbps"),S=o(W=>Number.isFinite(
W)?`${W.toFixed(0)} ms`:"-","fmtMs"),L=o(W=>Number.isFinite(W)?`${W.toFixed(W>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=o(async(W,ce)=>{const he=await W.json().catch(()=>({}));return he&&he.error?he.error:ce},
"parseErr"),E=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let W=0;W<4;W++){const ce=performance.now(),he=await apiFetch(
`/api/speedtest/ping?_=${h()}`,{cache:"no-store"}),ee=performance.now();if(!he.ok)throw new Error(await A(
he,"ping_failed"));await he.json().catch(()=>({})),E.push(ee-ce)}const R=E.reduce((W,ce)=>W+ce,0)/Math.
max(1,E.length),B=Math.min(...E),V=o(async W=>{const ce=performance.now(),he=await apiFetch(`/api/sp\
eedtest/download?bytes=${W}&_=${h()}`,{cache:"no-store"});if(!he.ok)throw new Error(await A(he,"down\
load_failed"));const ee=await he.arrayBuffer(),fe=performance.now();return{bytes:ee.byteLength||W,ms:fe-
ce,mbps:x(ee.byteLength||W,fe-ce)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${S(R)}
\u6E2C\u5B9A\u4E2D... download`);const ue=[];for(const W of[2*1024*1024,8*1024*1024])ue.push(await V(
W)),g(`\u6E2C\u5B9A\u4E2D... ping ${S(R)}
download ${L(Math.max(...ue.map(ce=>ce.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const oe=Math.max(...ue.map(W=>W.mbps)),J=o(async W=>{const ce=new Uint8Array(
W),he=performance.now(),ee=await apiFetch(`/api/speedtest/upload?_=${h()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:ce,cache:"no-store"}),fe=performance.now();if(!ee.ok)
throw new Error(await A(ee,"upload_failed"));const Ee=await ee.json().catch(()=>({})),$e=Number(Ee.bytes_received||
W)||W;return{bytes:$e,ms:fe-he,mbps:x($e,fe-he),serverMs:Number(Ee.server_elapsed_ms||0)||0}},"runUp\
load"),de=[];for(const W of[1*1024*1024,4*1024*1024])de.push(await J(W));const H=Math.max(...de.map(
W=>W.mbps)),me=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${S(R)} / ${S(B)}`,`Download (best): ${L(oe)}`,`Upload (best): ${L(H)}`,`Download r\
uns: ${ue.map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,`Upload runs: ${de.
map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(me.join(`
`)),showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u3092\u5B9F\u884C\u3057\u307E\u3057\u305F",
"success")}catch(g){u&&(u.textContent=`\u30A8\u30E9\u30FC: ${g&&g.message?g.message:"\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`),
showToast("\u56DE\u7DDA\u901F\u5EA6\u30C6\u30B9\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}finally{d&&(d.disabled=!1,d.classList.remove("opacity-60","cursor-not-allowed"))}}),get("ba\
n-appeal-refresh")&&(get("ban-appeal-refresh").onclick=()=>ht()),get("ban-appeal-mark-read")&&(get("\
ban-appeal-mark-read").onclick=()=>St()),get("ban-appeal-list")&&get("ban-appeal-list").addEventListener(
"click",async d=>{const u=d.target.closest("button");if(!u)return;const g=u.getAttribute("data-id");
if(u.classList.contains("ban-appeal-mark")){g&&await St([Number(g)]);return}if(u.classList.contains(
"ban-appeal-status")){const h=u.getAttribute("data-status");g&&h&&await _n({id:Number(g),status:h});
return}if(u.classList.contains("ban-appeal-reply-send")){const h=u.closest("[data-appeal-id]"),x=h?h.
querySelector(".ban-appeal-reply"):null,S=x?x.value:"";g&&await _n({id:Number(g),admin_reply:S});return}
if(u.classList.contains("ban-appeal-block")){if(!confirm("\u3053\u306E\u30E6\u30FC\u30B6\u30FC\u306E\u7570\u8B70\u7533\u3057\u7ACB\u3066\u3092\u30D6\u30ED\u30C3\u30AF\u3057\u307E\u3059\u304B\uFF1F"))
return;const h=prompt("\u30D6\u30ED\u30C3\u30AF\u7406\u7531 (\u4EFB\u610F)")||"";g&&await _n({id:Number(
g),block_user:!0,block_reason:h});return}}),get("upload-modal-close")&&(get("upload-modal-close").onclick=
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
(applyMarkerTransform(),renderCropOverlay())});const _i=o(()=>{const d=get("upload-modal");return!!(d&&
!d.classList.contains("hidden"))},"isUploadModalOpen"),Ht=get("drop-overlay");let Yt=0;const Si=o(()=>{
_i()||Ht&&(Ht.classList.remove("hidden"),Ht.classList.add("flex"))},"showDropOverlay"),Qt=o(()=>{Yt=
0,Ht&&(Ht.classList.add("hidden"),Ht.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
Qt;const kt=get("upload-dropzone");kt&&(kt.addEventListener("dragover",d=>{d.preventDefault(),kt.classList.
add("dragover")}),kt.addEventListener("dragleave",()=>{kt.classList.remove("dragover")}),kt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),kt.classList.remove("dragover"),Qt();const u=d.dataTransfer?
d.dataTransfer.files:null;u&&u.length&&handleFiles(u)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(Yt+=1,Si())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(Yt=Math.max(0,Yt-1),(Yt===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&Qt())}),window.addEventListener(
"dragend",()=>{Qt()}),window.addEventListener("drop",d=>{Qt(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(kt&&kt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const Kn=get("bot-admin-modal"),Li=o(d=>{const u=get("bot-admin-list");if(u){if(u.innerHTML=
"",!d||!d.length){u.innerHTML='<div class="text-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}d.forEach((g,h)=>{const x=!!g.is_bot_banned,S=g.bot_detection_enabled!==!1,L=Number(g.lock_remaining_seconds)||
0,A=document.createElement("div");A.className="flex flex-wrap items-center gap-2 bg-gray-900 border \
border-gray-700 rounded p-2 text-xs model-list-animate",A.style.animationDelay=`${Math.min(h,12)*.02}\
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
                    `,u.appendChild(A)})}},"renderBotUsers"),Zt=o(async(d="")=>{const u=get("bot-adm\
in-list");u&&(u.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin \
mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q=${encodeURIComponent(
d)}`),h=await g.json();g.ok&&h&&h.users?Li(h.users):(u&&(u.innerHTML='<div class="text-xs text-red-4\
00">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{u&&(u.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Mn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Kn))return;const u=get(
"settings-modal");u&&(u.classList.contains("modal-open")||u.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),showModal("bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState(
{modal:"admin-bots"},"","/admin-bots"),await Zt(get("bot-admin-search")?get("bot-admin-search").value.
trim():"")},"openBotAdminModal");window.openBotAdminModal=Mn;const An=o(async()=>{if(!isAdminUser||!get(
"bot-log-modal")||!window.BotAdminLog)return;const d=get("settings-modal");d&&(d.classList.contains(
"modal-open")||d.classList.contains("modal-prep"))&&hideModal("settings-modal"),showModal("bot-log-m\
odal"),location.pathname!=="/admin-bot-logs"&&history.pushState({modal:"admin-bot-logs"},"","/admin-\
bot-logs"),window.BotAdminLog.bind(),await window.BotAdminLog.openList()},"openBotLogModal");window.
openBotLogModal=An,window.closeBotLogModal=(d=!1)=>{get("bot-log-modal")&&hideModal("bot-log-modal"),
!d&&location.pathname==="/admin-bot-logs"&&history.back()},get("bot-log-open")&&(get("bot-log-open").
onclick=()=>{An()}),get("bot-log-close")&&(get("bot-log-close").onclick=()=>window.closeBotLogModal()),
window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||Kn)&&hideModal("bot-admin-modal"),!d&&location.
pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get("bot-admin-open").onclick=()=>{
Mn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),get("bot-ad\
min-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await Zt(get("bot-admin-search")?get(
"bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-refresh-btn").onclick=
async()=>{await Zt("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener("keydown",
async d=>{d.key==="Enter"&&await Zt(get("bot-admin-search").value.trim())}),get("bot-admin-list")&&(get(
"bot-admin-list").onclick=async d=>{const u=d.target.closest("button");if(!u)return;const g=u.getAttribute(
"data-user");if(!g)return;let h;if(u.classList.contains("bot-toggle-detect")){const x=u.getAttribute(
"data-enabled")!=="1";h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({username:g,action:"toggle_detection",enabled:x})})}else if(u.classList.
contains("bot-toggle-ban"))if(u.getAttribute("data-banned")==="1")h=await apiFetch("/api/bot/update",
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:g,action:"u\
nban"})});else{if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u3092BAN\u3057\u307E\u3059\u304B\uFF1F`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"ban",reason:"Admin ban"})})}else if(u.classList.contains("bo\
t-toggle-unban-linked")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"unban_linked"})})}else if(u.classList.contains("bot-delete-a\
ccount")){if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${g} \u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u5B8C\u5168\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u95A2\u9023\u30C7\u30FC\u30BF\u3082\u5373\u6642\u524A\u9664\u3055\u308C\u3001\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;h=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({username:g,action:"delete_account"})})}if(h){if(h.status===404)showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u306F\u65E2\u306B\u898B\u3064\u304B\u308A\u307E\u305B\u3093\uFF08\u524A\u9664\u3055\u308C\u305F\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059\uFF09`,
"error",!0);else if(h.ok){if(u.classList.contains("bot-delete-account")&&(showToast(`\u30E6\u30FC\u30B6\u30FC ${g}\
 \u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),g===currentUsername)){location.href="/";return}}else{
let x={};try{x=await h.json()}catch{}showToast(x.error==="admin_account"?"\u7BA1\u7406\u8005\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3067\u304D\u307E\u305B\u3093":
x.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F","error",!0)}await Zt(get("bo\
t-admin-search")?get("bot-admin-search").value.trim():"")}});const qt={"/settings":{id:"settings-mod\
al",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload-modal",open:o(()=>openUploadModal(),
"open")},"/library":{id:"lib-modal",open:o(()=>{mi(!1),showModal("lib-modal"),loadLibraryFiles()},"o\
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
-modal",open:o(()=>An(),"open")}},Xn=o((d,u=!1)=>{switch(d){case"settings-modal":jt(u);break;case"up\
load-modal":closeUploadModal(u);break;case"camera-capture-modal":closeCameraCaptureModal(u?{skipHistory:!0}:
{});break;case"history-modal":window.closeHistoryModal&&window.closeHistoryModal(u);break;case"lib-m\
odal":window.closeLibModal&&window.closeLibModal(u);break;case"branch-modal":window.closeBranchModal&&
window.closeBranchModal(u);break;case"batch-modal":window.closeBatchModal&&window.closeBatchModal(u);
break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(u);break;case"ma\
rker-modal":window.closeMarkerModal&&window.closeMarkerModal(u);break;case"thread-modal":window.closeThreadModal&&
window.closeThreadModal(u);break;case"model-modal":window.closeModelModal&&window.closeModelModal(u);
break;case"token-detail-modal":closeTokenDetail(u);break;case"encryption-status-modal":closeEncryptionModal(
u);break;case"python-exec-modal":closePythonExecDetail(u);break;case"gem-modal":window.closeGemModal&&
window.closeGemModal(u);break;case"compression-modal":window.closeCompressionModal&&window.closeCompressionModal(
u);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(u);break;case"bo\
t-log-modal":window.closeBotLogModal&&window.closeBotLogModal(u);break;case"mcp-decision-modal":typeof submitMcpDecision==
"function"?submitMcpDecision("deny"):hideModal(d);break;case"api-key-required-modal":{const h=get("a\
pi-key-modal-cancel-btn");h&&typeof h.onclick=="function"?h.click():hideModal(d);break}case"lyria-st\
udio-modal":window.closeLyriaStudio?window.closeLyriaStudio():hideModal(d);break;case"voice-studio-m\
odal":window.VoiceStudio?window.VoiceStudio.close():hideModal(d);break;case"version-update-modal":const g=localStorage.
getItem("app_version")||"";g&&localStorage.setItem("version_notified",g),hideModal(d);break;default:
hideModal(d);break}},"closeModalById");window.addEventListener("popstate",d=>{let u=!1;Object.values(
qt).forEach(x=>{const S=get(x.id);S&&S.classList.contains("modal-open")&&location.pathname!==Object.
keys(qt).find(L=>qt[L].id===x.id)&&(Xn(x.id,!0),u=!0)});const g=location.pathname.match(/^\/c\/(.+)$/);
if(g){const x=decodeURIComponent(g[1]);String(currentThreadId)!==String(x)&&loadMessages(x,{skipHistory:!0})}else
location.pathname==="/"&&currentThreadId&&startNewChat({skipHistory:!0});const h=qt[location.pathname];
if(h){const x=get(h.id);x&&!x.classList.contains("modal-open")&&h.open()}});const Yn=location.pathname;
qt[Yn]&&(history.replaceState({},"","/"),setTimeout(()=>qt[Yn].open(),500)),get("easy-login-generate")&&
(get("easy-login-generate").onclick=async()=>{const d=get("easy-login-mins"),u=d?parseInt(d.value||"\
5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${u}\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))
return;const h=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({minutes:u})})).json();h&&h.temp_password?(get("easy-login-code").textContent=
h.temp_password,get("easy-login-exp").textContent=h.expires_at||"",get("easy-login-result").classList.
remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}),get("easy-login-cancel")&&(get("easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\
\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const u=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(u&&u.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const Ti=4,Ci=5*1024*1024;function Qn(d,u,g){let h=[];const x=o(()=>{g&&(g.innerHTML="",
h.forEach((S,L)=>{const A=document.createElement("div");A.className="relative w-16 h-16 rounded bord\
er border-gray-600 overflow-hidden";const E=document.createElement("img");E.src=S.url,E.alt=S.file.name,
E.className="w-full h-full object-cover";const R=document.createElement("button");R.type="button",R.
className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none px-1 py-0.5",R.textContent=
"\xD7",R.onclick=()=>{URL.revokeObjectURL(S.url),h.splice(L,1),x()},A.appendChild(E),A.appendChild(R),
g.appendChild(A)}))},"render");return d&&u&&(d.onclick=()=>u.click(),u.onchange=S=>{for(const L of Array.
from(S.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(L.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(L.
size>Ci){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(h.
length>=Ti){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}h.push({file:L,url:URL.createObjectURL(L)})}S.target.value="",x()}),{files:o(()=>h.
map(S=>S.file),"files"),clear:o(()=>{h.forEach(S=>URL.revokeObjectURL(S.url)),h=[],x()},"clear")}}o(
Qn,"createFbImagePicker");async function Zn(d,u){try{const g=new FormData;return u.forEach(h=>g.append(
"images",h,h.name)),(await apiFetch(d,{method:"POST",body:g})).ok}catch{return!1}}o(Zn,"uploadFbImag\
es");const ei=Qn(get("fb-images-add"),get("fb-images-input"),get("fb-images-list"));get("fb-submit").
onclick=async()=>{const d=get("fb-title").value.trim(),u=get("fb-message").value.trim();if(!u){showToast(
"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:u,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},h=window.ActivityLog?window.ActivityLog.feedbackPayload():null;h&&(g.client_logs=h);
const x=get("fb-attach-chat"),S=!!(x&&x.checked);if(S&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\
\u3066\u3044\u307E\u305B\u3093","error",!0);return}S&&(g.chat_copy=window.ActivityLog?window.ActivityLog.
chatCopyPayload(currentThreadId):{client:"web",thread_id:String(currentThreadId)});const L=await apiFetch(
"/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),
A=L.ok?await L.clone().json().catch(()=>({})):{},E=ei.files(),R={count:E.length,saved:!0};L.ok&&R.count&&
(R.saved=await Zn(`/api/feedback/${A.public_id}/images`,E)),(window.ActivityLog?await window.ActivityLog.
reportFeedback(L,h,S,R):L.ok)&&(get("fb-title").value="",get("fb-message").value="",ei.clear(),x&&(x.
checked=!1),en())};async function ti(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),en()}o(ti,"deleteFeedback");const Mi="\u76F4\u8FD11\u6642\u9593\u306E\u64CD\u4F5C\u30ED\u30B0\u3092\u4E00\u7DD2\u306B\u9001\u4FE1\u3059\u308B";
let En=[];function ni(d,u){const g=d.replies||[];return g.length?`<div class="fb-replies mt-2 space-\
y-2 border-l-2 border-gray-700 pl-2">${g.map(h=>{const x=h.role==="admin"?u&&h.author?`\u7BA1\u7406\u8005 (${h.
author})`:"\u7BA1\u7406\u8005":u?"\u9001\u4FE1\u8005":"\u3042\u306A\u305F",S=u?[h.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${h.
log_file}`,h.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${h.image_count}\u679A\uFF09: ${h.image_dir}`]:
[h.image_count&&`\u6DFB\u4ED8\u753B\u50CF: ${h.image_count}\u679A`];return`<div class="text-[11px]">\
<div class="text-gray-400">${escapeHtml(x)} / ${escapeHtml(h.created_at||"")}</div><div class="text-\
sm whitespace-pre-wrap ${h.role==="admin"?"text-green-300":"text-gray-200"}">${escapeHtml(h.message)}\
</div>${S.filter(Boolean).map(L=>`<div class="${u?"text-amber-300":"text-gray-400"}">${escapeHtml(L)}\
</div>`).join("")}</div>`}).join("")}</div>`:""}o(ni,"feedbackRepliesHtml");function ii(){return`<di\
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
                            <span class="block text-xs font-bold text-white">${Mi}</span>
                            <span class="fb-reply-logs-note block text-[10px] text-gray-500 mt-1"></\
span>
                        </span>
                    </label>
                    <button type="button" class="fb-reply-send bg-blue-600 hover:bg-blue-500 text-wh\
ite px-3 py-1 rounded text-xs">\u8FD4\u4FE1</button>
                </div>`}o(ii,"feedbackReplyFormHtml");function si(d,u){const g=u.querySelector(".fb-\
reply-form");if(!g)return;const h=g.querySelector(".fb-reply-logs"),x=g.querySelector(".fb-reply-log\
s-note"),S=!!(window.ActivityLog&&window.ActivityLog.isEnabled());h.disabled=!S,x.textContent=S?"\u30C1\u30A7\u30C3\
\u30AF\u3057\u305F\u3068\u304D\u3060\u3051\u3001\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u3067\u8A18\u9332\u3057\u305F\u76F4\u8FD11\u6642\u9593\u306E\u30ED\u30B0\u3092\u8FD4\u4FE1\u3068\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u307E\u3059\u3002":
"\u300C\u30ED\u30B0\u306E\u53CE\u96C6\u3092\u5F37\u5316\u300D\u304C\u30AA\u30D5\u306E\u305F\u3081\u3001\u30ED\u30B0\u306F\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3002";
const L=Qn(g.querySelector(".fb-reply-images-add"),g.querySelector(".fb-reply-images-input"),g.querySelector(
".fb-reply-images-list"));En.push(L);const A=g.querySelector(".fb-reply-send");A.onclick=async()=>{const E=g.
querySelector(".fb-reply-message").value.trim();if(!E){showToast("\u8FD4\u4FE1\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const R={message:E,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.appVersion||
""},B=h.checked&&window.ActivityLog?window.ActivityLog.feedbackPayload():null;B&&(R.client_logs=B);const V=L.
files();A.disabled=!0;try{const ue=await apiFetch(`/api/feedback/${d.public_id}/replies`,{method:"PO\
ST",headers:{"Content-Type":"application/json"},body:JSON.stringify(R)}),oe=await ue.json().catch(()=>({}));
if(!ue.ok){showToast(oe.error||"\u8FD4\u4FE1\u3092\u9001\u4FE1\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}const J=!V.length||await Zn(`/api/feedback/${d.public_id}/replies/${oe.reply_id}/\
images`,V),de=[];if(B&&oe.logs_saved===!1&&de.push("\u30ED\u30B0"),J||de.push("\u753B\u50CF"),de.length)
showToast(`\u8FD4\u4FE1\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F\uFF08${de.join("\u3068")}\u306F\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\uFF09`,"error",!0);else{const H=["\u8FD4\u4FE1"];B&&H.push(`\u76F4\u8FD11\u6642\u9593\u306E\u30ED\u30B0\uFF08${B.
entries.length}\u4EF6\uFF09`),V.length&&H.push(`\u753B\u50CF\uFF08${V.length}\u679A\uFF09`),showToast(
H.join(H.length>2?"\u3001":"\u3068")+"\u3092\u9001\u4FE1\u3057\u307E\u3057\u305F","success")}en()}finally{
A.disabled=!1}}}o(si,"bindFeedbackReply");async function en(){En.forEach(S=>S.clear()),En=[];const u=await(await apiFetch(
"/api/feedback?all=1")).json(),g=get("fb-list");g.innerHTML="",(u.items||[]).filter(S=>!u.is_admin||
S.user_id===void 0||S.user_id===null||!0).forEach(S=>{if(u.is_admin)return;const L=document.createElement(
"div");L.className="p-2 rounded border border-gray-700 bg-gray-800/50",L.innerHTML=`<div class="text\
-[11px] text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(S.public_id||"")}</span> \
/ ${S.created_at}</div><div class="font-bold text-sm">${escapeHtml(S.title||"No Title")}</div><div c\
lass="text-sm whitespace-pre-wrap">${escapeHtml(S.message)}</div><div class="text-[11px] text-gray-4\
00 mt-1">Status: ${escapeHtml(S.status)}</div>${S.image_count?`<div class="text-[11px] text-gray-400\
 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${S.image_count}\u679A</div>`:""}${ni(S,!1)}${ii()}<div class="mt-2\
"><button type="button" class="fb-delete bg-red-700 hover:bg-red-600 text-white px-2 py-1 rounded te\
xt-[10px]">\u524A\u9664</button></div>`,L.querySelector(".fb-delete").onclick=()=>ti(S.public_id),si(
S,L),g.appendChild(L)});const h=get("fb-admin-panel"),x=get("fb-admin-list");u.is_admin?(h.classList.
remove("hidden"),x.innerHTML="",(u.items||[]).forEach(S=>{const L=document.createElement("div");L.className=
"p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",L.innerHTML=`
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
                                </select>
                                <button class="fb-save bg-blue-600 hover:bg-blue-500 text-white px-3\
 py-1 rounded text-xs">\u72B6\u614B\u3092\u4FDD\u5B58</button>
                                <button type="button" class="fb-delete bg-red-700 hover:bg-red-600 t\
ext-white px-3 py-1 rounded text-xs">\u524A\u9664</button>
                            </div>
                            ${ni(S,!0)}
                            ${ii()}
                        `,L.querySelector(".fb-status").value=S.status||"new",L.querySelector(".fb-s\
ave").onclick=async()=>{const A=L.querySelector(".fb-status").value;await apiFetch(`/api/feedback/${S.
public_id}/update`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:A})}),
en()},L.querySelector(".fb-delete").onclick=()=>ti(S.public_id),si(S,L),x.appendChild(L)})):h.classList.
add("hidden")}if(o(en,"loadFeedback"),window.setupTOTP=async()=>{const u=await(await apiFetch("/api/\
2fa/totp/setup",{method:"POST"})).json();get("totp-qr").src=u.qr_image,get("totp-secret-disp").innerText=
u.secret,get("totp-setup-area").classList.remove("hidden")},window.enableTOTP=async()=>{const d=get(
"totp-verify-code").value;if(!d)return;(await apiFetch("/api/2fa/totp/enable",{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F",
"success"),get("totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):
showToast("\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","er\
ror",!0)},window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),u=get("webauthn-nam\
e"),g=u?String(u.value||"").trim():"";try{d&&(d.disabled=!0);const h=await apiFetch("/api/2fa/webaut\
hn/register/options",{method:"POST"}),x=await h.json();if(!h.ok){showToast(x.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\
\u305F","error",!0);return}const L=await(await ensureWebAuthnJson()).create({publicKey:x}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},L,{name:g}))}),E=await A.json().catch(()=>({}));A.ok?(u&&(u.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast(E.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch(h){showToast(`WebAuthn Error: ${h}`,"error",!0)}finally{d&&(d.disabled=!1)}},window.
removeWebAuthnCredential=async d=>{if(!d||!confirm("\u3053\u306E\u30D1\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
return;const u=await apiFetch("/api/2fa/webauthn/remove",{method:"POST",headers:{"Content-Type":"app\
lication/json"},body:JSON.stringify({id:d})}),g=await u.json().catch(()=>({}));if(u.ok){showToast("\u30D1\
\u30B9\u30AD\u30FC\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),openSettingsModal();return}
showToast(g.error||"\u30D1\u30B9\u30AD\u30FC\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)},get("delete-account-btn")&&(get("delete-account-btn").onclick=async()=>{if(!confirm(`\u672C\u5F53\
\u306B\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))return;let d;try{d=
await apiFetch(CHAT_CONFIG.urls.deleteAccount,{method:"POST"})}catch{showToast("\u901A\u4FE1\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0);return}if(d.ok){location.href="/";
return}let u={};try{u=await d.json()}catch{}if(u&&u.error==="turnstile_required"){showToast("\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\
\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}showToast(u.error||"\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0)}),get("prompt-input").onkeydown=d=>{if(d.isComposing)return;const u=get("prompt-input");
if(slashSuggestionsVisible){const g=get("slash-command-suggestions");if(d.key==="ArrowDown"){d.preventDefault(),
slashSelectedIndex=Math.min(slashSelectedIndex+1,visibleSlashCommands(lastSlashFilter||"").length-1),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(u.value),u.value));
return}if(d.key==="ArrowUp"){d.preventDefault(),slashSelectedIndex=Math.max(slashSelectedIndex-1,0),
showSlashCommandSuggestions(slashCommandSuggestionFilter(extractSlashCommandToken(u.value),u.value));
return}if(d.key==="Enter"){d.preventDefault();const h=visibleSlashCommands(slashCommandSuggestionFilter(
extractSlashCommandToken(u.value),u.value));h[slashSelectedIndex]?selectSlashCommand(h[slashSelectedIndex].
id):h.length>0&&selectSlashCommand(h[0].id);return}if(d.key==="Escape"){d.preventDefault(),hideSlashCommandSuggestions();
return}}if(gemSuggestionsVisible){const g=u.value.trim();if(d.key==="ArrowDown"){d.preventDefault(),
gemSelectedIndex=gemSelectedIndex+1,showGemSuggestions(g.substring(1));return}if(d.key==="ArrowUp"){
d.preventDefault(),gemSelectedIndex=Math.max(gemSelectedIndex-1,0),showGemSuggestions(g.substring(1));
return}if(d.key==="Enter"){d.preventDefault();const h=g.substring(1).toLowerCase(),x=loadedGems.filter(
S=>S.name.toLowerCase().includes(h)||S.description&&S.description.toLowerCase().includes(h));x[gemSelectedIndex]?
selectGemSuggestion(x[gemSelectedIndex]):x.length>0&&selectGemSuggestion(x[0]);return}if(d.key==="Es\
cape"){d.preventDefault(),hideGemSuggestions();return}}if(d.key==="Escape"&&pendingSlashCommand){d.preventDefault(),
hidePendingSlashCommandIndicator();return}d.key==="ArrowUp"&&(u.selectionStart===0||d.ctrlKey)?promptHistory.
length>0&&(historyIndex===-1&&(tempPrompt=u.value),historyIndex<promptHistory.length-1&&(d.preventDefault(),
historyIndex++,u.value=promptHistory[historyIndex],u.dispatchEvent(new Event("input")))):d.key==="Ar\
rowDown"&&(u.selectionEnd===u.value.length||d.ctrlKey)&&historyIndex>-1&&(d.preventDefault(),historyIndex--,
historyIndex===-1?u.value=tempPrompt:u.value=promptHistory[historyIndex],u.dispatchEvent(new Event("\
input"))),enterToSend?d.key==="Enter"&&!d.shiftKey&&(d.preventDefault(),sendMessage()):(d.metaKey||d.
ctrlKey)&&d.key==="Enter"&&(d.preventDefault(),sendMessage())},get("prompt-input")&&(get("prompt-inp\
ut").addEventListener("input",function(){this.style.height="auto",this.style.height=this.scrollHeight+
"px",schedulePromptTokenEstimate(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const d=this.
value.trim();if(pendingSlashCommand)gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&
hideSlashCommandSuggestions(),lastSlashFilter=null;else if(d.startsWith("@")){const u=d.substring(1);
showGemSuggestions(u),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=null}else if(d.
startsWith("/")){const u=slashCommandSuggestionFilter(extractSlashCommandToken(d),this.value);(!slashSuggestionsVisible||
u!==lastSlashFilter)&&(lastSlashFilter=u,showSlashCommandSuggestions(u)),gemSuggestionsVisible&&hideGemSuggestions()}else
gemSuggestionsVisible&&hideGemSuggestions(),slashSuggestionsVisible&&hideSlashCommandSuggestions(),lastSlashFilter=
null}),get("prompt-input").addEventListener("blur",()=>{setTimeout(()=>{slashSuggestionsVisible&&hideSlashCommandSuggestions(),
gemSuggestionsVisible&&hideGemSuggestions()},150)})),get("cancel-edit-btn")&&(get("cancel-edit-btn").
onclick=cancelEdit),updatePromptPlaceholder(),aiSettingsConversation.length>0&&(pendingSlashCommand=
"settings",showPendingSlashCommandIndicator("settings")),get("search-box")&&(get("search-box").addEventListener(
"input",d=>{const u=get("search-box");if(u&&isUserInitiatedSearchInput(d))markThreadSearchUserEdited(
u);else if(u&&!u.dataset.userEdited){discardAutofilledThreadSearch("cleared-autofill-search-box-inpu\
t");return}if(isSettingsModalOpen()){snapshotSidebarHistory("ignore-search-input-settings-open");return}
clearTimeout(searchTimeout),searchTimeout=setTimeout(()=>{loadThreads(!1)},300)}),hardenThreadSearchInputs()),
get("mobile-new-chat-btn")&&(get("mobile-new-chat-btn").onclick=()=>startNewChat()),get("sts-mic-btn")&&
(get("sts-mic-btn").onclick=()=>{isStsModel()&&get("mic-btn").click()}),get("sts-cancel-btn")&&(get(
"sts-cancel-btn").onclick=()=>{isStsModel()&&Ai()}),get("prompt-input")&&get("prompt-input").addEventListener(
"paste",async d=>{const u=(d.clipboardData||window.clipboardData).items,g=[];for(let h=0;h<u.length;h++)
if(u[h].kind==="file"){const x=u[h].getAsFile();x&&g.push(x)}g.length>0&&(d.preventDefault(),await handleFiles(
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
"warning",!0)}catch(d){const u=d&&d.message?d.message:"\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(u,"error",!0)}}),get("rich-paste-prompt")&&get("rich-paste-prompt").addEventListener("inpu\
t",()=>{richPastePromptPreferenceSyncing||queueRichPastePromptPreferenceSave()}),get("rich-paste-use\
-default")&&get("rich-paste-use-default").addEventListener("change",()=>{richPastePromptPreferenceSyncing||
queueRichPastePromptPreferenceSave()}),get("rich-paste-capture")){const d=get("rich-paste-capture");
d.addEventListener("paste",async u=>{const g=u.clipboardData||window.clipboardData;if(g){u.preventDefault();
try{await ingestRichPasteClipboardData(g)||showToast("\u30AF\u30EA\u30C3\u30D7\u30DC\u30FC\u30C9\u306B\u8CBC\u308A\u4ED8\u3051\u53EF\u80FD\u306A\u5185\u5BB9\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F",
"warning",!0),updateRichPasteStatus()}catch{showToast("\u8CBC\u308A\u4ED8\u3051\u306E\u53D6\u308A\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}}),d.addEventListener("input",()=>{d.value=""})}get("chat-container").addEventListener(
"click",d=>{const u=d.target.closest("img.chat-image"),g=u?u.dataset.viewerSrc||u.currentSrc||u.src:
"";u&&g&&(d.preventDefault(),openImageViewer(g))});const tn=document.querySelector(".viewer-content");
tn&&(tn.addEventListener("touchstart",onViewerTouchStart,{passive:!1}),tn.addEventListener("touchmov\
e",onViewerTouchMove,{passive:!1}),tn.addEventListener("touchend",onViewerTouchEnd),tn.addEventListener(
"touchcancel",onViewerTouchEnd)),get("image-viewer").addEventListener("click",d=>{if(suppressViewerCloseClick){
suppressViewerCloseClick=!1;return}(d.target.id==="image-viewer"||d.target.classList.contains("viewe\
r-content"))&&closeImageViewer()}),get("file-viewer").addEventListener("click",d=>{d.target.id==="fi\
le-viewer"&&closeFileViewer()}),document.addEventListener("keydown",d=>{d.key==="Escape"&&closeImageViewer()});
let it,Oe=null,gn=[],$n=!1,hn=null,Gt=null,Et=null,nn=null,In=0,bn=!1,sn=null,Ut=null,Lt=null,an=null,
yn=null,$t=null,vn=null,wn=null;function ai(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
vn)&&vn.length)return vn;d.innerHTML="";const u=[];for(let g=0;g<24;g++){const h=document.createElement(
"span");h.className="block rounded-full",h.style.background="rgba(252, 165, 165, 0.92)",h.style.width=
"2px",h.style.transition="height 75ms linear, opacity 75ms linear",h.style.height="2px",h.style.opacity=
"0.4",u.push(h),d.appendChild(h)}return vn=u,u}o(ai,"ensureMicWaveformBars");function It(d,u="hidden"){
const g=get("mic-recording-indicator"),h=get("mic-recording-text");if(g){if(wn&&(clearTimeout(wn),wn=
null),u==="hidden"){g.classList.add("hidden");return}h&&d&&(h.innerText=d),g.classList.remove("hidde\
n"),u==="recording"?g.style.color="rgb(252 165 165)":u==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(It,"setMicRecordingIndicator");function oi(){ai().forEach(u=>{u.
style.height="2px",u.style.opacity="0.35"})}o(oi,"resetMicWaveformBars");function Pt(){if(yn&&(cancelAnimationFrame(
yn),yn=null),an){try{an.disconnect()}catch{}an=null}if(Ut){try{Ut.close()}catch{}Ut=null}Lt=null,$t=
null,oi()}o(Pt,"stopMicWaveform");function ri(d){Pt();const u=ai();if(!u.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{Ut=new g,Lt=Ut.createAnalyser(),Lt.fftSize=
256,Lt.smoothingTimeConstant=0,an=Ut.createMediaStreamSource(d),an.connect(Lt),$t=new Uint8Array(Lt.
frequencyBinCount)}catch{Pt();return}const h=o(()=>{if(!Lt||!$t)return;Lt.getByteFrequencyData($t);const x=Math.
max(1,Math.floor($t.length/u.length));for(let S=0;S<u.length;S++){const A=($t[Math.min($t.length-1,S*
x)]||0)/255,E=Math.max(2,Math.round(2+A*10));u[S].style.height=`${E}px`,u[S].style.opacity=`${.35+A*
.65}`}yn=requestAnimationFrame(h)},"render");h()}o(ri,"startMicWaveform");function xn(){if(hn&&(clearInterval(
hn),hn=null),nn){try{nn.disconnect()}catch{}nn=null}if(Gt){try{Gt.close()}catch{}Gt=null}Et=null}o(xn,
"stopSilenceMonitor");function li(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;xn();const u=window.
AudioContext||window.webkitAudioContext;if(!u)return;Gt=new u,Et=Gt.createAnalyser(),Et.fftSize=2048,
nn=Gt.createMediaStreamSource(d),nn.connect(Et);const g=new Uint8Array(Et.fftSize),h=getStsSilenceMs(),
x=.02;In=0,bn=!1,hn=setInterval(()=>{if(!Et)return;Et.getByteTimeDomainData(g);let S=0;for(let A=0;A<
g.length;A++){const E=(g[A]-128)/128;S+=E*E}if(Math.sqrt(S/g.length)>x){bn||(bn=!0),In=Date.now();return}
bn&&Date.now()-In>h&&it&&it.state==="recording"&&it.stop()},200)}o(li,"startSilenceMonitor");const Nn=class Nn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(u,g,h,x={},S=null){if(this.
model=h,this.ws=new WebSocket(`${g}?access_token=${encodeURIComponent(u)}`),this.ws.binaryType="arra\
ybuffer",this.ws.onopen=()=>{console.log("Gemini Live WebSocket opened. Sending setup...");const E=!!(x&&
x.transcriptionConfig),R={setup:{model:`models/${h}`,generationConfig:{responseModalities:E?["TEXT"]:
["AUDIO"]},inputAudioTranscription:E?x.transcriptionConfig||{}:{}}};E||(R.setup.outputAudioTranscription=
{}),x.speechConfig&&(R.setup.generationConfig.speechConfig=x.speechConfig),x.thinkingConfig&&(R.setup.
generationConfig.thinkingConfig=x.thinkingConfig),x.translationConfig&&(R.setup.generationConfig.translationConfig=
x.translationConfig),console.log("Sending setup:",JSON.stringify(R)),this.ws.send(JSON.stringify(R))},
this.ws.onmessage=E=>this._handleMessage(E),this.ws.onerror=E=>{console.error("Gemini Live WebSocket\
 error:",E),this.onError&&this.onError(E)},this.ws.onclose=E=>{console.log("Gemini Live WebSocket cl\
osed:",E.code,E.reason),this.closedEvent=E,this.onClose&&this.onClose(E)},S&&S.stream&&S.ctx&&S.source){
this.stream=S.stream,this.audioContext=S.ctx;var L=S.source}else{this.stream=await navigator.mediaDevices.
getUserMedia(kn());const E=ci(this.stream,16e3);this.audioContext=E.ctx;var L=E.source}if(this.audioContext.
state==="suspended")try{await this.audioContext.resume()}catch{}this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=E=>{E.data.
size>0&&this.userAudioChunks.push(E.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
E=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const R=di(E.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!R||!R.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(R))),mimeType:"audio/\
pcm;rate=16000"}}}))},L.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(u){
let g=null;try{g=JSON.parse(typeof u.data=="string"?u.data:new TextDecoder().decode(u.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const h=g.serverContent;h.modelTurn&&h.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log(
"Gemini thought delta:",x.text),this.assistantThought+=x.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(x.text)),x.inlineData&&x.inlineData.data){const S=x.inlineData.data;
console.log("Gemini audio chunk received, size:",S.length),this.rtPlayer&&this.rtPlayer.addChunk(S);
const L=atob(S),A=new Uint8Array(L.length);for(let E=0;E<L.length;E++)A[E]=L.charCodeAt(E);this.assistantAudioChunks.
push(A)}}),h.outputTranscription&&h.outputTranscription.text&&this._appendAssistantText(h.outputTranscription.
text),h.inputTranscription&&h.inputTranscription.text&&(this.userTurnBreak&&this.inputTranscript&&!this.
inputTranscript.endsWith(`
`)&&(this.inputTranscript+=`
`),this.userTurnBreak=!1,this.inputTranscript+=h.inputTranscription.text,this.interimInputTranscript=
""),h.interimInputTranscription&&(this.interimInputTranscript=h.interimInputTranscription.text||""),
h.turnComplete&&(this.assistantTurnBreak=!0,this.model!=="gemini-3.5-transcribe-live"&&(this.userTurnBreak=
!0))}g.error&&this.onError&&this.onError(g.error),this.onMessage&&this.onMessage(g)}_appendAssistantText(u){
u&&(this.assistantTurnBreak&&this.assistantText&&!this.assistantText.endsWith(`
`)&&(this.assistantText+=`
`),this.assistantTurnBreak=!1,this.assistantText+=u)}stop(){const u=this.backupRecorder;if(this.backupRecorder=
null,this._stopped=new Promise(h=>{let x=!1;const S=o(()=>{x||(x=!0,h())},"finish");if(!u||u.state===
"inactive"){S();return}u.addEventListener("stop",S,{once:!0});try{u.stop()}catch{S()}setTimeout(S,1500)}),
this.ws)try{this.ws.close()}catch{}if(this.processor)try{this.processor.disconnect()}catch{}const g=o(
()=>{if(this.stream&&this.stream.getTracks().forEach(h=>{try{h.stop()}catch{}}),this.audioContext)try{
this.audioContext.close()}catch{}},"release");this._stopped.then(g)}async getFinalData(){this._stopped&&
await this._stopped;const u=new Blob(this.assistantAudioChunks),g=await this._blobToBase64(u),h=new Blob(
this.userAudioChunks),x=await this._blobToBase64(h);return{user_text:this.inputTranscript,assistant_text:this.
assistantText,assistant_thought:this.assistantThought,audio_base64:g,user_audio_base64:x}}_blobToBase64(u){
return new Promise(g=>{const h=new FileReader;h.onloadend=()=>g(h.result.split(",")[1]),h.readAsDataURL(
u)})}};o(Nn,"GeminiLiveClient");let Pn=Nn;const Rn=class Rn{constructor(u=24e3){const g=window.AudioContext||
window.webkitAudioContext;this.ctx=new g({sampleRate:u}),this.nextStartTime=0,this.bufferDelay=.1,this.
started=!1}async addChunk(u){if(!this.ctx)return;const g=atob(u),h=new Uint8Array(g.length);for(let R=0;R<
g.length;R++)h[R]=g.charCodeAt(R);const x=new Int16Array(h.buffer),S=new Float32Array(x.length);for(let R=0;R<
x.length;R++)S[R]=x[R]/32768;const L=this.ctx.createBuffer(1,S.length,this.ctx.sampleRate);L.getChannelData(
0).set(S),this.ctx.state==="suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();
A.buffer=L,A.connect(this.ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.
bufferDelay,this.started=!0);const E=Math.max(this.ctx.currentTime,this.nextStartTime);A.start(E),this.
nextStartTime=E+L.duration}stop(){this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Rn,"RealTimeAudioPl\
ayer");let on=Rn;const Bn=class Bn{constructor(){this.active=!1,this.capturing=!1,this.sessionId=null,
this.abortCtrl=null,this.reader=null,this.audioCtx=null,this.processor=null,this.stream=null,this.rtPlayer=
null,this.rateIn=24e3,this.rateOut=24e3,this.userTranscript="",this.assistantTranscript="",this.assistantThought=
"",this.speechActive=!1,this.responseDoneCount=0,this.lastAudioAt=0,this.streamError=null,this.saved=
!1,this.saving=!1,this.stopping=!1,this.audioQueue=[],this.audioFlush=null}isActive(){return this.active}async start(){
if(this.active)return;if(this.saving||this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const u=get("model-select")?get("model-select").value:"";if(!isRealtimeSessionModel()){
showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"warning",!0);return}if(!currentThreadId)try{const x=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=x.id!==null&&x.id!==void 0?String(x.id):x.id,setTemporaryChatUiState(!!(x&&x.
is_temporary)),setCurrentChatHeaderTitle(x&&x.title),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+x.id),get("welcome-screen").classList.add("hidden")}catch(h){showToast(
"\u30B9\u30EC\u30C3\u30C9\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+h.message,"\
error",!0);return}const g={model:u,thread_id:currentThreadId,voice:get("sts-voice")?get("sts-voice").
value:"",speed:get("sts-speed")?get("sts-speed").value:"",rate_in:get("sts-rate-in")?get("sts-rate-i\
n").value:"",rate_out:get("sts-rate-out")?get("sts-rate-out").value:"",thinking_level:get("sts-think\
ing-level")?get("sts-thinking-level").value:"",include_thoughts:get("sts-include-thoughts")?get("sts\
-include-thoughts").checked:!1,reasoning_effort:get("sts-reasoning-effort")?get("sts-reasoning-effor\
t").value:"",target_lang:(isGeminiLiveTranslateModel()||u==="gpt-realtime-translate")&&get("sts-targ\
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
!0)}_openStream(){const u="/api/realtime/stream?session_id="+encodeURIComponent(this.sessionId),g=window.
ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions({credentials:"include",signal:this.abortCtrl.signal}):{credentials:"include",signal:this.
abortCtrl.signal};fetch(u,g).then(h=>{if(!h.ok)throw new Error("SSE stream failed ("+h.status+")");this.
reader=h.body.getReader(),this._readLoop()}).catch(h=>{h&&h.name==="AbortError"||(this.streamError=h&&
h.message?h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",this.active&&(setStsStatus("\u30B9\
\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u63A5\u7D9A\u304C\u5207\u65AD\u3055\u308C\u307E\u3057\u305F",
"error",!0)))})}async _readLoop(){const u=new TextDecoder;let g="";try{for(;this.reader;){const{done:h,
value:x}=await this.reader.read();if(h)break;g+=u.decode(x,{stream:!0});let S;for(;(S=g.indexOf(`

`))>=0;){const L=g.slice(0,S);g=g.slice(S+2);for(const A of L.split(`
`)){if(!A.startsWith("data: "))continue;let E=null;try{E=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent(E)}}}}catch(h){if(h&&h.name==="AbortError")return;this.active&&(this.streamError=h&&h.message?
h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(u){
if(u)switch(u.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new on(this.rateOut||24e3),zt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
this.rtPlayer.addChunk(u.data));break;case"transcript":u.role==="user"?(u.cumulative?this.userTranscript=
u.delta:this.userTranscript+=u.delta,window.VoiceStudio&&window.VoiceStudio.log("user",this.userTranscript)):
u.role==="assistant"?(this.assistantTranscript+=u.delta,window.VoiceStudio&&window.VoiceStudio.log("\
assistant",this.assistantTranscript)):u.role==="thought"&&(this.assistantThought+=u.delta);break;case"\
speech_started":this.speechActive=!0,this._stopPlayback(),setStsStatus("\u805E\u304D\u53D6\u308A\u4E2D...",
!0);break;case"speech_stopped":this.speechActive=!1,setStsStatus("\u5FDC\u7B54\u5F85\u3061...",!0);break;case"\
interrupted":this._stopPlayback();break;case"response_done":case"turn_complete":this.responseDoneCount+=
1;break;case"status":u.status==="ready"&&this.active&&setStsStatus("\u8A71\u3057\u3066\u304F\u3060\u3055\u3044...",
!0);break;case"notice":u.message&&showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0: "+u.message,
"warning",!0);break;case"error":this.streamError=u.message||"\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u30A8\u30E9\u30FC",
setStsStatus("\u30A8\u30E9\u30FC",!1);break;case"final":this.active&&!this.saved&&this._save();break}}_stopPlayback(){
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}zt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(kn()).then(u=>{this.stream=u;const g=this.rateIn||24e3,h=ci(u,g);this.audioCtx=
h.ctx;const x=h.source,S=this.audioCtx.sampleRate,L=4096;this.processor=this.audioCtx.createScriptProcessor(
L,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const E=A.inputBuffer.
getChannelData(0),R=di(E,S,g);!R||!R.byteLength||this._sendAudio(R)},x.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(u){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(u)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const u=this.audioQueue.splice(0,this.audioQueue.length),
g=u.reduce((E,R)=>E+R.byteLength,0),h=new Uint8Array(g);let x=0;u.forEach(E=>{h.set(E,x),x+=E.byteLength});
const S="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),L={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:h.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(L):L;try{await fetch(S,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(u=>u.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}xn(),Pt()}async stop(){if(!this.active)return;if(this.
active=!1,this.stopping=!0,this._stopCapture(),setStsStatus("\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
!0),this.audioFlush)try{await this.audioFlush}catch{}try{await apiFetch("/api/realtime/commit",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:this.sessionId})})}catch{}
const u=Date.now(),g=this.responseDoneCount;let h=this.lastAudioAt;for(;Date.now()-u<2e4&&!(this.responseDoneCount>
g||(this.lastAudioAt>h&&(h=this.lastAudioAt),!this.speechActive&&Date.now()-u>2e3&&Date.now()-h>2500));)
await new Promise(x=>setTimeout(x,250));await this._save()}async _save(){if(!this.saved){this.saved=
!0,this.saving=!0;try{const u=await apiFetch("/api/realtime/save",{method:"POST",headers:{"Content-T\
ype":"application/json"},body:JSON.stringify({session_id:this.sessionId,thread_id:currentThreadId})}),
g=await u.json().catch(()=>({}));if(!u.ok)throw new Error(g.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(g.thread_id&&!currentThreadId&&(currentThreadId=String(g.thread_id)),this.streamError)setStsStatus(
"\u30A8\u30E9\u30FC",!1),showToast("\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u4F1A\u8A71\u3067\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F: "+
this.streamError,"error",!0);else{setStsStatus("\u4FDD\u5B58\u3057\u307E\u3057\u305F",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1200);try{await loadMessages(currentThreadId)}catch{}}}catch(u){
setStsStatus("\u4FDD\u5B58\u30A8\u30E9\u30FC",!1),showToast("\u97F3\u58F0\u4F1A\u8A71\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F: "+
(u&&u.message?u.message:u),"error",!0)}finally{this.saving=!1,this.stopping=!1,this._cleanup()}}}_cancel(){
this.sessionId&&apiFetch("/api/realtime/cancel",{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify({session_id:this.sessionId})}).catch(()=>{}),this._cleanup(),setStsStatus(
"Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800)}_cleanup(){if(this.active=!1,this.
capturing=!1,this.stopping=!1,this.audioQueue=[],this._stopCapture(),this._stopPlayback(),this.abortCtrl){
try{this.abortCtrl.abort()}catch{}this.abortCtrl=null}this.reader=null,this.sessionId=null;const u=get(
"mic-btn");u&&(u.classList.remove("bg-red-600","animate-pulse"),u.classList.add("bg-gray-700"))}};o(
Bn,"RealtimeVoiceSession");let On=Bn;function ci(d,u){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let h=null;try{return h=new g({sampleRate:u}),{ctx:h,
source:h.createMediaStreamSource(d)}}catch{if(h)try{h.close()}catch{}return h=new g,{ctx:h,source:h.
createMediaStreamSource(d)}}}o(ci,"openMicAudioSource");function di(d,u,g){let h=d;if(u!==g&&u>0&&g>
0){const S=u/g,L=Math.floor(h.length/S),A=new Float32Array(L);for(let E=0;E<L;E++)A[E]=h[Math.min(Math.
floor(E*S),h.length-1)];h=A}const x=new Int16Array(h.length);for(let S=0;S<h.length;S++){const L=Math.
max(-1,Math.min(1,h[S]));x[S]=L<0?L*32768:L*32767}return x.buffer}o(di,"pcm16FromFloat32");const rn=new On,
ui=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let h=null,x=null,S=!1,L=null,A=!1,E=0,R=0,B=null,V="idle",ue=!1,oe=null;const J=o(F=>document.getElementById(
F),"$"),de=o(F=>{const K=Object.assign({},F||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(K):(K.progressSpinner=!1,
K)},"noSpinner");function H(F,K){V=K;const Se=J("lyria-status-text"),le=J("lyria-status-dot");Se&&(Se.
textContent=F),le&&(le.className="w-2 h-2 rounded-full inline-block "+(g[K]||g.idle)),he(),ee()}o(H,
"setStatus");function me(){const F=R?Math.floor((Date.now()-R)/1e3):0,K=String(Math.floor(F/60)).padStart(
2,"0"),Se=String(F%60).padStart(2,"0");return`${K}:${Se}`}o(me,"formatElapsed");function W(){R||(R=Date.
now());const F=J("lyria-elapsed");F&&(F.textContent=me()),B||(B=window.setInterval(()=>{const K=J("l\
yria-elapsed");K&&(K.textContent=me())},1e3))}o(W,"startElapsedTimer");function ce(){B&&(window.clearInterval(
B),B=null)}o(ce,"stopElapsedTimer");function he(){const F=J("lyria-play-btn"),K=J("lyria-pause-btn"),
Se=J("lyria-stop-btn"),le=J("lyria-reset-btn"),Me=!!h,Fe=V==="streaming"||V==="connecting";if(F){F.disabled=
ue;const Qe=F.querySelector("i");Qe&&(Qe.className="fas fa-play")}K&&(K.disabled=ue||!Fe),Se&&(Se.disabled=
ue||!Me||!Fe),le&&(le.disabled=ue||!Me||!Fe)}o(he,"updateTransportButtons");function ee(){const F=J(
"lyria-save-btn");if(!F)return;const K=!!h&&V!=="idle"&&V!=="connecting"&&V!=="error";F.classList.toggle(
"hidden",!K)}o(ee,"updateSaveButton");function fe(F,K){const Se=J("lyria-prompt-rows");if(!Se)return;
const le=document.createElement("div");le.className="flex items-center gap-2",le.innerHTML=`
                        <input type="text" value="${escapeHtml(F||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof K=="num\
ber"?K:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof K=="number"?K:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Me=le.querySelector('input[type="range"]'),Fe=le.querySelector(".lyria-w\
eight-label");Me&&Fe&&Me.addEventListener("input",()=>{Fe.textContent=parseFloat(Me.value).toFixed(1)});
const Qe=le.querySelector(".lyria-prompt-remove");Qe&&Qe.addEventListener("click",()=>{Se.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||le.remove()}),le.classList.add("lyria-prompt-row-wrap"),Se.appendChild(
le)}o(fe,"addPromptRow");function Ee(){const F=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),K=[];return F.forEach(Se=>{const le=Se.querySelector('input[type="text"]'),Me=Se.querySelector(
'input[type="range"]'),Fe=(le?le.value:"").trim();Fe&&K.push({text:Fe,weight:parseFloat(Me?Me.value:
1)||1})}),K}o(Ee,"collectPrompts");function $e(){const F={},K=o(Pi=>{const jn=J(Pi);return jn&&jn.value!==
""?parseFloat(jn.value):void 0},"num"),Se=K("lyria-bpm");Se!==void 0&&(F.bpm=Math.round(Se));const le=K(
"lyria-guidance");le!==void 0&&(F.guidance=le);const Me=K("lyria-density");Me!==void 0&&(F.density=Me);
const Fe=K("lyria-brightness");Fe!==void 0&&(F.brightness=Fe);const Qe=K("lyria-temperature");Qe!==void 0&&
(F.temperature=Qe);const Ge=J("lyria-scale");Ge&&Ge.value&&(F.scale=Ge.value);const et=J("lyria-mode");
et&&et.value&&(F.music_generation_mode=et.value);const pt=J("lyria-mute-bass"),Ot=J("lyria-mute-drum\
s"),bi=J("lyria-only-bass-drums");return pt&&(F.mute_bass=pt.checked),Ot&&(F.mute_drums=Ot.checked),
bi&&(F.only_bass_and_drums=bi.checked),F}o($e,"collectConfig");function st(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([K,Se])=>{
const le=J(K),Me=J(Se);!le||!Me||le.addEventListener("input",()=>{const Fe=parseFloat(le.value);Me.textContent=
K==="lyria-bpm"?String(Math.round(Fe)):Fe.toFixed(1)})})}o(st,"bindRangeLabels");function Ue(){if(L){
try{L.close()}catch{}L=null}A=!1,E=0}o(Ue,"resetPlayback");function ut(){if(S=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(ut,"closeStream");async function Tt(){ut(),x=new AbortController,
S=!0;try{const F=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(h)}`,de({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!F.ok){const Me=await F.
json().catch(()=>({}));throw new Error(Me.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const K=F.body.getReader(),Se=new TextDecoder;let le="";for(;S;){const{done:Me,value:Fe}=await K.read();
if(Me)break;le+=Se.decode(Fe,{stream:!0});const Qe=le.split(`

`);le=Qe.pop();for(const Ge of Qe){const et=Ge.split(`
`).find(Ot=>Ot.startsWith("data: "));if(!et)continue;const pt=et.slice(6);try{const Ot=JSON.parse(pt);
_e(Ot)}catch{}}}}catch(F){if(F&&F.name==="AbortError")return;S&&(H("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{S&&h&&Tt()},1200))}finally{S=!1}}o(Tt,"openStream");function _e(F){
if(F&&F.snapshot){const K=F.status;if(K==="error"){H("\u30A8\u30E9\u30FC","error"),ce();return}if(K===
"closed"||K==="stopped"){H("\u7D42\u4E86","closed"),ce();return}H(K==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",K==="paused"?"paused":"connecting");return}if(F&&F.audio){H("\u518D\u751F\u4E2D...",
"streaming"),W(),be(F.audio);return}if(F&&F.error){H("\u30A8\u30E9\u30FC: "+F.error,"error"),ce();return}
if(F&&F.final){H("\u7D42\u4E86","closed"),ce(),he();return}}o(_e,"handleStreamMessage");function be(F){
if(!F)return;if(!L){const Ge=window.AudioContext||window.webkitAudioContext;if(!Ge)return;L=new Ge({
sampleRate:48e3}),A=!1,E=0}let K;try{const Ge=atob(F);K=new Uint8Array(Ge.length);for(let et=0;et<Ge.
length;et++)K[et]=Ge.charCodeAt(et)}catch{return}const Se=new Int16Array(K.buffer),le=Math.floor(Se.
length/2);if(le<1)return;const Me=L.createBuffer(2,le,48e3);for(let Ge=0;Ge<2;Ge++){const et=Me.getChannelData(
Ge);for(let pt=0;pt<le;pt++)et[pt]=Se[pt*2+Ge]/32768}L.state==="suspended"&&L.resume();const Fe=L.createBufferSource();
Fe.buffer=Me,Fe.connect(L.destination),A||(E=L.currentTime+.08,A=!0);const Qe=Math.max(L.currentTime,
E);Fe.start(Qe),E=Qe+Me.duration}o(be,"playChunk");async function Be(F,K){const Se=await fetch("/api\
/gemini/music/command",de({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:h,type:F},K||{}))})),le=await Se.json().catch(()=>({}));if(!Se.ok)throw new Error(
le.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return le}
o(Be,"apiCommand");async function Ye(){if(ue)return;const F=Ee();if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}ue=!0,he(),H("\u63A5\u7D9A\u4E2D...","connecting");try{
const K=await fetch("/api/gemini/music/start",de({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:F,config:$e()})})),Se=await K.json().catch(()=>({}));if(!K.
ok)throw new Error(Se.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
h=Se.session_id,oe=$e(),H("\u63A5\u7D9A\u4E2D...","connecting"),Tt()}catch(K){H("\u30A8\u30E9\u30FC: "+
K.message,"error"),showToast("Lyria RealTime: "+K.message,"error",!0)}finally{ue=!1,he()}}o(Ye,"star\
tSession");async function at(F){if(h){ue=!0,he();try{await Be("control",{action:F}),F==="PLAY"?H("\u518D\u751F\
\u4E2D...","streaming"):F==="PAUSE"?H("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):F==="STOP"?H("\u505C\u6B62\u4E2D",
"stopped"):F==="RESET_CONTEXT"&&H("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(K){showToast("Lyria RealTime: "+K.message,"error",!0),H("\u30A8\u30E9\u30FC: "+K.
message,"error")}finally{ue=!1,he()}}}o(at,"control");async function ve(){if(!h)return;const F=Ee();
if(!F.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}ue=!0;try{await Be("prompts",{weighted_prompts:F}),H("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
V==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(K){showToast("Lyria RealTime: "+K.message,"error",!0)}finally{ue=!1,he()}}o(ve,"app\
lyPrompts");async function Pe(){if(!h)return;const F=$e(),K=oe||{},Se=F.bpm!==void 0&&F.bpm!==K.bpm,
le=F.scale!==void 0&&F.scale!==K.scale,Me=Se||le;ue=!0;try{await Be("config",{config:F,reset_context:Me}),
oe=F,H(Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",V==="paused"?"paused":"streaming"),showToast(
Me?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Fe){showToast("Lyria RealT\
ime: "+Fe.message,"error",!0)}finally{ue=!1,he()}}o(Pe,"applyConfig");async function He(){if(h){ue=!0,
H("\u4FDD\u5B58\u4E2D...","connecting"),he();try{const F=await fetch("/api/gemini/music/save",de({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h,thread_id:currentThreadId||
null})})),K=await F.json().catch(()=>({}));if(!F.ok)throw new Error(K.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
H("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),ce(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),K.thread_id&&(currentThreadId=String(K.thread_id),history.pushState({},"","/c/"+K.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(K.thread_id||currentThreadId),cn(!0)}catch(F){
H("\u30A8\u30E9\u30FC: "+F.message,"error"),showToast("Lyria RealTime: "+F.message,"error",!0)}finally{
ue=!1,he()}}}o(He,"saveSession");async function ze(){if(ut(),h)try{await fetch("/api/gemini/music/ca\
ncel",de({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h})}))}catch{}
h=null,ce(),Ue(),H("\u6E96\u5099\u5B8C\u4E86","idle")}o(ze,"cancelSession");function ct(){const F=J(
"lyria-prompt-rows");F&&(F.innerHTML=""),fe("",1),oe=null,R=0,["lyria-bpm","lyria-guidance","lyria-d\
ensity","lyria-brightness","lyria-temperature"].forEach(le=>{const Me=J(le);Me&&(Me.value=le==="lyri\
a-bpm"?"120":le==="lyria-guidance"?"4":le==="lyria-temperature"?"1.1":"0.5")});const K=J("lyria-scal\
e");K&&(K.value="");const Se=J("lyria-mode");Se&&(Se.value="QUALITY"),["lyria-mute-bass","lyria-mute\
-drums","lyria-only-bass-drums"].forEach(le=>{const Me=J(le);Me&&(Me.checked=!1)}),st()}o(ct,"resetC\
ontrols");function cn(F){ut(),h&&fetch("/api/gemini/music/cancel",de({method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({session_id:h})})).catch(()=>{}),h=null,S=!1,ce(),Ue(),
hideModal("lyria-studio-modal")}o(cn,"closeAndCleanup");function Wt(F){if(!isLyriaRealtimeModel()){showToast(
"Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const K=J("lyria-studio-modal");if(K&&K.classList.contains("modal-open")&&h){if(F&&
typeof F=="string"){const le=J("lyria-prompt-rows");le&&(le.innerHTML=""),fe(F,1)}return}if(h&&ze(),
ct(),F&&typeof F=="string"){const le=J("lyria-prompt-rows");le&&(le.innerHTML=""),fe(F,1)}h=null,S=!1,
ce(),Ue(),H("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Wt,"open");function dn(){
const F=J("lyria-open-studio-btn");F&&F.addEventListener("click",()=>Wt(""));const K=J("lyria-studio\
-close");K&&K.addEventListener("click",()=>cn(!1));const Se=J("lyria-play-btn");Se&&Se.addEventListener(
"click",()=>{if(!h){Ye();return}at("PLAY")});const le=J("lyria-pause-btn");le&&le.addEventListener("\
click",()=>at("PAUSE"));const Me=J("lyria-stop-btn");Me&&Me.addEventListener("click",()=>at("STOP"));
const Fe=J("lyria-reset-btn");Fe&&Fe.addEventListener("click",()=>at("RESET_CONTEXT"));const Qe=J("l\
yria-add-prompt-btn");Qe&&Qe.addEventListener("click",()=>fe("",1));const Ge=J("lyria-apply-prompts-\
btn");Ge&&Ge.addEventListener("click",ve);const et=J("lyria-apply-config-btn");et&&et.addEventListener(
"click",Pe);const pt=J("lyria-save-btn");pt&&pt.addEventListener("click",He),st(),ct(),window.openLyriaStudio=
Wt}o(dn,"init");function Fn(){ue||cn(!1)}return o(Fn,"requestClose"),{init:dn,open:Wt,requestClose:Fn}})();
ui.init(),window.closeLyriaStudio=()=>ui.requestClose(),(()=>{let d=null,u=null,g=null;const h="voic\
eDockSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
S=o(ee=>document.getElementById(ee),"$");function L(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(L,"isStudioMode");function A(){const ee=get("model-select")?get("model-select").value:"",fe=S("voi\
ce-studio-title");fe&&(ee==="gpt-transcribe"||ee==="gpt-live-transcribe"||ee==="grok-voice-transcrib\
e-2.0-file"?fe.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ee===
"gemini-3.5-live-translate-preview"?fe.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
fe.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(A,"updateTitle");function E(){const ee=S("v\
oice-studio-transcript");ee&&(ee.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${x}</div>`);const fe=S("sts-live-transcript");fe&&(fe.innerHTML="",fe.classList.add("hidden"))}
o(E,"resetTranscript");function R(ee,fe,Ee){const $e=ee.querySelectorAll(".voice-studio-line");let st=null;
for(let Ue=$e.length-1;Ue>=0;Ue--)if($e[Ue].dataset.role===fe){st=$e[Ue];break}if(st)st.innerHTML=Ee;else{
const Ue=ee.querySelector(".voice-studio-placeholder");Ue&&Ue.remove();const ut=document.createElement(
"div");ut.className="voice-studio-line",ut.dataset.role=fe,ut.innerHTML=Ee,ee.appendChild(ut)}ee.classList.
remove("hidden"),ee.scrollTop=ee.scrollHeight}o(R,"writeLine");function B(ee,fe){if(!fe||!String(fe).
trim()||!L())return;const st=`<span class="${ee==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ee==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
fe)}</span>`;[S("voice-studio-transcript"),S("sts-live-transcript")].filter(Boolean).forEach(Ue=>R(Ue,
ee,st))}o(B,"log");function V(ee,fe=!0){const Ee=S("sts-panel"),$e=S("sts-settings-toggle");if(Ee&&Ee.
classList.toggle("settings-open",!!ee),$e&&$e.setAttribute("aria-expanded",ee?"true":"false"),fe)try{
localStorage.setItem(h,ee?"1":"0")}catch{}}o(V,"setSettingsOpen");function ue(){try{return localStorage.
getItem(h)==="1"}catch{return!1}}o(ue,"readSettingsOpen");let oe=null;function J(){const ee=get("mod\
el-select")?get("model-select").value:"";ee!==oe&&(oe=ee,E()),V(ue(),!1),A()}o(J,"syncDock");function de(){
const ee=S("sts-panel"),fe=S("voice-studio-panel-host");ee&&fe&&ee.parentNode!==fe&&(d=ee.parentNode,
u=ee.nextSibling,fe.appendChild(ee));const Ee=S("file-preview"),$e=S("voice-studio-file-host");Ee&&$e&&
Ee.parentNode!==$e&&(g=Ee.parentNode,$e.appendChild(Ee),$e.classList.remove("hidden"))}o(de,"movePan\
elIntoModal");function H(){const ee=S("sts-panel");ee&&d&&ee.parentNode!==d&&(u&&u.parentNode===d?d.
insertBefore(ee,u):d.appendChild(ee));const fe=S("file-preview");fe&&g&&fe.parentNode!==g&&g.appendChild(
fe);const Ee=S("voice-studio-file-host");Ee&&Ee.classList.add("hidden"),d=null,u=null,g=null}o(H,"mo\
vePanelBack");function me(){if(!L())return;de();const ee=S("sts-panel");ee&&ee.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(me,"open");function W(){window.
VoiceStudioOpen=!1,H(),hideModal("voice-studio-modal")}o(W,"close");function ce(){window.VoiceStudioOpen&&
W()}o(ce,"closeIfOpen");function he(){window.VoiceStudioOpen=!1;const ee=S("voice-studio-open-btn");
ee&&ee.addEventListener("click",()=>me());const fe=S("voice-studio-close");fe&&fe.addEventListener("\
click",()=>W());const Ee=S("sts-settings-toggle");Ee&&Ee.addEventListener("click",()=>{const $e=S("s\
ts-panel");V(!($e&&$e.classList.contains("settings-open")))}),window.VoiceStudio={open:me,close:W,closeIfOpen:ce,
log:B,isStudioMode:L,syncDock:J},J()}return o(he,"init"),{init:he}})().init();let zt=null;function pi(){
if(zt&&(zt.stop(),zt=null),sn){try{sn.pause()}catch{}try{sn.src=""}catch{}sn=null}}o(pi,"stopStsPlay\
back");async function Ri(d){pi();const u=new Audio;return u.src=d,u.preload="auto",u.autoplay=!0,u.playsInline=
!0,sn=u,await u.play(),new Promise(g=>{u.onended=()=>g("ended"),u.onerror=()=>g("error")})}o(Ri,"pla\
yStsAudio");function Ai(){if(rn.isActive()){rn._cancel();return}if(Oe){Oe.stop(),Oe=null,pi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Pt();return}it&&it.state===
"recording"&&($n=!0,it.stop())}o(Ai,"cancelRecording");function kn(){if(isStsModel())return{audio:!0};
const u=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return u.echoCancellation&&(g.echoCancellation=!1),u.
noiseSuppression&&(g.noiseSuppression=!1),u.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(kn,"\
getMicCaptureConstraints");function Ei(){const d=window.AudioContext||window.webkitAudioContext;if(!d||
!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u30DE\u30A4\u30AF\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093");
let u=null;try{u=new d({sampleRate:16e3})}catch{if(u){try{u.close()}catch{}u=null}u=new d}const g=u.
resume(),h=navigator.mediaDevices.getUserMedia(kn());return{ctx:u,resumePromise:g,streamPromise:h}}o(
Ei,"beginGeminiLiveMic");function $i(d){if(d){if(d.streamPromise)d.streamPromise.then(u=>{try{u.getTracks().
forEach(g=>g.stop())}catch{}}).catch(()=>{});else if(d.stream)try{d.stream.getTracks().forEach(u=>u.
stop())}catch{}if(d.ctx)try{d.ctx.close()}catch{}}}o($i,"releaseGeminiLiveMic"),get("mic-btn").onclick=
async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Oe){setStsStatus("Processing...",!0);const d=Oe;Oe=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const u=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(u.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
u.assistant_text=(d.inputTranscript||"").trim(),u.assistant_thought="",!u.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}const g=(u.user_text||
"").trim()||(u.assistant_text||"").trim()||(u.assistant_thought||"").trim(),h=!!(u.audio_base64||u.user_audio_base64);
if(!g&&!h){setStsStatus("Error",!1),showToast("Gemini Live \u306E\u97F3\u58F0\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){const S=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=String(S.id),history.pushState(
{},"","/c/"+S.id),get("welcome-screen").classList.add("hidden")}u.thread_id=currentThreadId,u.model=
get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POST",headers:{"Content-Typ\
e":"application/json"},body:JSON.stringify(u)}),setStsStatus("Saved",!1),setTimeout(()=>setStsStatus(
"Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(u){console.error("Failed to save G\
emini Live session:",u),setStsStatus("Error saving",!1)}return}if(rn.isActive()){get("mic-btn").classList.
remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),rn.stop();return}if(it&&
it.state==="recording"){it.stop(),get("mic-btn").classList.remove("bg-red-600","animate-pulse"),get(
"mic-btn").classList.add("bg-gray-700"),isStsModel()||It("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);let g=null;try{g=Ei();
const x={model:get("model-select").value};if(isGeminiLiveTranscribeModel()){if(x.transcription_mode=
get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const H=get(
"sts-custom-vocab").value.split(/[,、\n]/).map(me=>me.trim()).filter(Boolean);H.length&&(x.custom_vocabulary=
H.slice(0,1e3))}}else x.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(x.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",x.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(x.target_lang=get("sts-target-lang").value);const S=await g.streamPromise;try{await g.
resumePromise}catch{}g={stream:S,ctx:g.ctx,source:g.ctx.createMediaStreamSource(S)};const L=await apiFetch(
"/api/gemini/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
x)});if(!L.ok)throw new Error("Failed to get session token");const{token:A,url:E}=await L.json(),R=get(
"model-select").value,B=get("sts-voice")?get("sts-voice").value:"Kore",V=get("sts-thinking-level")?get(
"sts-thinking-level").value:"minimal",ue=get("sts-include-thoughts")?get("sts-include-thoughts").checked:
!1;if(Oe=new Pn,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Oe.rtPlayer=new on),isGeminiLiveTranscribeModel()){
const H=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",me={languageCodes:[]};
if((H==="SMART"||H==="VERBATIM")&&(me.mode=H),get("sts-custom-vocab")){const W=get("sts-custom-vocab").
value.split(/[,、\n]/).map(ce=>ce.trim()).filter(Boolean);W.length&&(me.customVocabulary=W.slice(0,
1e3))}await Oe.start(A,E,R,{transcriptionConfig:me},g)}else if(isGeminiLiveTranslateModel()){const H=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Oe.start(A,E,R,{translationConfig:{targetLanguageCode:H,
echoTargetLanguage:!0}},g)}else{const H={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:B}}}};
isGeminiLiveExtendedThinkingModel()&&(H.thinkingConfig={thinkingLevel:V,includeThoughts:ue}),await Oe.
start(A,E,R,H,g)}g=null,it=Oe.backupRecorder,it.onstop=()=>{Oe&&get("mic-btn").click()};let oe=!0,J="\
live-sts-"+Date.now();Oe.onMessage=H=>{if(H.serverContent){if(isGeminiLiveTranscribeModel()){const me=Oe.
interimInputTranscript,W=Oe.inputTranscript,ce=W+(me&&!W.endsWith(me)?(W?`
`:"")+me:""),he=get("chat-messages");let ee=document.getElementById(J);ee||(ee=document.createElement(
"div"),ee.id=J,ee.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded-\
lg border border-slate-700/50",ee.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,he.appendChild(ee),he.scrollTop=he.scrollHeight);const fe=ee.
querySelector(".message-content");fe.innerText=ce||"\u8074\u304D\u53D6\u308A\u4E2D...",he.scrollTop=
he.scrollHeight,window.VoiceStudio&&W&&window.VoiceStudio.log("user",W);return}if(H.serverContent.modelTurn){
oe&&(setStsStatus("Gemini is speaking...",!1),oe=!1);const me=get("chat-messages");let W=document.getElementById(
J);W||(W=document.createElement("div"),W.id=J,W.className="flex flex-col gap-2 mb-4 assistant-messag\
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
                                            `,me.appendChild(W),me.scrollTop=me.scrollHeight);const ce=W.
querySelector(".thought-container"),he=W.querySelector(".message-content");Oe.assistantThought&&(ce.
classList.remove("hidden"),ce.innerText=Oe.assistantThought),he.innerText=Oe.assistantText,me.scrollTop=
me.scrollHeight,window.VoiceStudio&&(Oe.inputTranscript&&window.VoiceStudio.log("user",Oe.inputTranscript),
Oe.assistantText&&window.VoiceStudio.log("assistant",Oe.assistantText))}}},setStsStatus("Listening..\
.",!0),get("mic-btn").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","ani\
mate-pulse"),ri(Oe.stream),li(Oe.stream);const de=Oe;de.onError=H=>{const me=H&&H.message?H.message:
typeof H=="string"?H:"";me&&showToast("Gemini Live: "+me,"error",!0)},de.onClose=H=>{if(Oe===de){if(H&&
H.code&&H.code!==1e3){const me=H.reason?": "+H.reason:" (code "+H.code+")";showToast("Gemini Live \u306E\u63A5\
\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+me,"error",!0)}get("mic-btn").click()}},de.closedEvent&&
de.onClose(de.closedEvent);return}catch(h){if($i(g),Oe){try{Oe.stop()}catch{}Oe=null}showToast("Gemi\
ni Live connection failed: "+h.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await rn.start();return}isStsModel()||(oi(),It("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(kn());it=new MediaRecorder(d),gn=[],$n=!1;const u=isStsModel();
it.ondataavailable=g=>gn.push(g.data),it.onstop=async()=>{if($n){gn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(L=>L.stop()),xn(),Pt(),u||(It("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),wn=setTimeout(()=>It("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(gn,{type:"audio/web\
m"}),h=new File([g],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",h),get("fil\
e-preview").classList.remove("hidden");const S=u;get("file-name").innerText=S?"Processing voice...":
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
"/sts",{method:"POST",body:x});if(!L.ok){const de=await L.json().catch(()=>({}));throw new Error(de.
error||"Speech-to-speech failed")}const A=L.body.getReader(),E=new TextDecoder;let R="",B=null,V=null;
stsOpt("sts-auto-play")&&(V=new on,zt=V),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let ue=!0,oe="",J="";for(;;){const{done:de,value:H}=await A.read();if(de)break;
R+=E.decode(H,{stream:!0});const me=R.split(`
`);R=me.pop();for(const W of me){if(!W.trim())continue;const ce=JSON.parse(W);if(ce.error)throw new Error(
ce.error);ce.audio_delta&&V&&(ue&&(setStsStatus("Playing response...",!1),ue=!1),await V.addChunk(ce.
audio_delta)),ce.input_delta&&(oe+=ce.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
oe)),ce.transcript_delta&&(J+=ce.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",J)),(ce.final||ce.audio_url)&&(B=ce)}}window.VoiceStudio&&!oe.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),B&&(B.audio_url||B.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const L=get(
"set-mic-transcribe-mode");if(!!(L&&L.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const R=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(R.transcript){const B=get("prompt-inp\
ut");B.value+=(B.value?" ":"")+R.transcript,B.style.height="auto",B.style.height=B.scrollHeight+"px"}else
showToast(R.error||"Transcription failed","error",!0)}}catch(L){showToast("Audio processing error: "+
L.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(L=>L.
stop()),xn(),Pt(),S||It("","hidden"),S&&setStsStatus("Tap to speak",!1)}},it.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(It("\u9332\u97F3\u4E2D\u2026","recording"),ri(d)),li(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Pt(),isStsModel()||It("","hidden"),alert("Microphone access denied or not avail\
able.")}};const ln=o((d,u)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=u:d.textContent=
u},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,u=get("lib-del-btn"),g=get("lib-download-btn"),h=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),S=get("lib-usage-btn");if(u&&(u.disabled=d===0,ln(u,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,ln(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
h&&(h.disabled=d===0,ln(h,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,ln(x,"\u540D\u524D\u5909\u66F4")),
S&&(S.disabled=d!==1,ln(S,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const L=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",L&&d>0)}};function mi(d){lib.
attachMode=!!d}o(mi,"setLibAttachMode");const fi=o((d=!1)=>{mi(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>fi(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>fi(!0)),
get("lib-search")){let d=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),d&&clearTimeout(d),d=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const d=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=d,get("lib-sort").onchange=()=>{const u=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,u),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const Ii=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>Ii(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
u=get("gem-desc").value,g=get("gem-inst").value,h=collectGemFixedPrompts();if(d&&g){const x=editingGemUuid?
"PUT":"POST",S=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
S,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:u,instruction:g,
fixed_prompts:h,default_model:get("gem-default-model").value||null})}),window.closeGemModal(),loadGems(),
editingGemUuid&&activeGem&&activeGem.uuid===editingGemUuid&&(activeGem.name=d,activeGem.instruction=
g,activeGem.fixed_prompts=h,applyActiveGem(activeGem))}else alert("Name and Instruction are required\
.")},document.addEventListener("click",function(d){if(d.target.closest(".edit-btn")){const g=d.target.
closest(".edit-btn").getAttribute("data-id");beginEditMessage(g)}if(d.target.closest(".code-toggle")){
const u=d.target.closest(".code-toggle"),g=u.closest(".code-wrapper");if(!g)return;const h=g.classList.
toggle("collapsed");g.setAttribute("data-collapsed",h?"true":"false"),u.setAttribute("aria-expanded",
h?"false":"true"),u.innerHTML=h?'<i class="fas fa-chevron-down"></i>':'<i class="fas fa-chevron-up">\
</i>',u.title=h?"\u5C55\u958B":"\u6298\u308A\u305F\u305F\u3080",u.setAttribute("aria-label",h?"\u5C55\u958B":
"\u6298\u308A\u305F\u305F\u3080")}if(d.target.closest(".download-btn")){const u=d.target.closest(".d\
ownload-btn"),g=u.getAttribute("data-code"),h=(u.getAttribute("data-lang")||"txt").toLowerCase();if(g)
try{const x=decodeURIComponent(g),S=new Blob([x],{type:"text/plain"}),L=URL.createObjectURL(S),A=document.
createElement("a");A.href=L;let R={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[h]||h;(h.length>8||/[^a-z0-9]/.test(h))&&(R="txt");let B=`code.${R}`;h==="dockerfile"&&(B="Docke\
rfile"),h==="makefile"&&(B="Makefile"),A.download=B,document.body.appendChild(A),A.click(),document.
body.removeChild(A),URL.revokeObjectURL(L)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const u=d.target.closest(".copy-btn"),g=u.getAttribute("data-code");g&&
window.copyCode(u,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const u=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(u)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",u=>{u.target===d&&Xn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
