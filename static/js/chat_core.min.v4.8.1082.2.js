document.addEventListener("DOMContentLoaded",()=>{var ci,di;initThemeFromServer(),applyLiquidGlassMode(
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
x=ideogramModelTraits(g),_={resolution:x.resolution,quality:x.quality,speed:x.speed,style:x.style,negative:x.
negative};if(d&&d.classList.toggle("hidden",!h),u&&u.classList.toggle("hidden",!h),!h)return;IDEOGRAM_IMAGE_FIELDS.
forEach(A=>{const $=_[A]!==!1;["ideogram-image-","modal-ideogram-image-"].forEach(N=>{const B=get(N+
A);!B||!B.parentElement||B.parentElement.classList.toggle("hidden",!$)})}),["ideogram-image-style","\
modal-ideogram-image-style"].forEach(A=>{const $=get(A);if(!$)return;const N=x.styleOptions.join("|");
if($.dataset.options!==N){const B=$.value;$.innerHTML=x.styleOptions.map(q=>`<option value="${q}">${q===
"auto"?"Auto":q}</option>`).join(""),$.dataset.options=N,$.value=x.styleOptions.includes(B)?B:"auto"}});
const L=get("ideogram-image-note");L&&(L.textContent=x.edit?"\u203B Ideogram 4.5: \u753B\u50CF\u3092\u6DFB\u4ED8\uFF08\u307E\u305F\u306F\u76F4\u524D\u306E\u753B\u50CF\u304C\u3042\u308B\u5834\u5408\uFF09\u306F P\
recise Edit\u3002\u51FA\u529B\u306F\u5143\u753B\u50CF\u3068\u540C\u30B5\u30A4\u30BA":"\u203B Ideogram: \u30C6\u30AD\
\u30B9\u30C8\u304B\u3089\u306E\u751F\u6210\u306E\u307F\uFF08\u753B\u50CF\u7DE8\u96C6\u306F Ideogram 4.5 \u306E\u307F\uFF09")}
o(f,"updateIdeogramImageUi");function b(){var h;const d=get("grok-video-options");if(!d)return;const u=String(
((h=get("model-select"))==null?void 0:h.value)||"").toLowerCase();isGrokVideoModel()?d.classList.remove(
"hidden"):d.classList.add("hidden");const g=get("grok-video-resolution");if(g){const x=Array.from(g.
options).find(_=>_.value==="1080p");x&&(x.disabled=u!=="grok-imagine-video-1.5"),u!=="grok-imagine-v\
ideo-1.5"&&g.value==="1080p"&&(g.value="720p")}}o(b,"updateGrokVideoUi");function y(){var x;const d=get(
"gemini-video-options");if(!d)return;const u=String(((x=get("model-select"))==null?void 0:x.value)||
"").toLowerCase();isGeminiVideoModel()?d.classList.remove("hidden"):d.classList.add("hidden");const g=get(
"gemini-video-resolution");if(g){const _=Array.from(g.options).find(A=>A.value==="4K"),L=u==="veo-3.\
1-lite-generate-preview"||u==="veo-3.1-fast-generate-preview"||u==="gemini-omni-flash";_&&(_.disabled=
L),L&&g.value==="4K"&&(g.value="1080p")}const h=get("gemini-video-duration-wrap");h&&h.classList.toggle(
"hidden",u==="gemini-omni-1.1-flash")}o(y,"updateGeminiVideoUi");function v(){const d=get("gemini-mu\
sic-options");if(!d)return;const u=isGeminiRealtimeMusicModel(),g=isGeminiMusicModel()&&!u;d.classList.
toggle("hidden",!g);const h=get("lyria-realtime-studio-bar");h&&h.classList.toggle("hidden",!u)}o(v,
"updateGeminiMusicUi");function w(){var L;const d=get("xai-chat-options");if(!d)return;const u=String(
((L=get("model-select"))==null?void 0:L.value)||"").toLowerCase(),g=u.startsWith("grok-")&&!isGrokImageModel(
u)&&!isGrokVideoModel(u)&&!u.includes("voice");d.classList.toggle("hidden",!g);const h=get("xai-logp\
robs"),x=get("xai-top-logprobs"),_=u.includes("grok-4.20");h&&(h.disabled=_,_&&(h.checked=!1)),x&&(x.
disabled=_,_&&(x.value=""))}o(w,"updateXaiChatUi");function k(){const d=isMistralOcrModel(),u=get("m\
istral-ocr-options");u&&u.classList.toggle("hidden",!d);const g=get("modal-mistral-ocr-options");g&&
g.classList.toggle("hidden",!d),["canvas-mode-container","coding-mode-container","browser-fast-mode-\
container"].forEach(h=>{const x=get(h);x&&(x.classList.toggle("opacity-50",d),x.classList.toggle("po\
inter-events-none",d))}),d&&(canvasModeEnabled&&syncCanvasModeUi(!1,{persist:!1}),codingModeEnabled&&
syncCodingModeUi(!1,{persist:!1}),typeof browserFastModeEnabled!="undefined"&&browserFastModeEnabled&&
setBrowserFastModeEnabled(!1))}o(k,"updateMistralOcrUi");function S(){const d=get("image-input-limit\
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
o(S,"updateImageInputLimits");function C(){const d=get("enable-sys-prompt"),u=!!(d&&d.disabled),g=!!(d&&
d.checked);M(),d&&(d.disabled?!u&&g?d.dataset.restoreChecked="1":u||delete d.dataset.restoreChecked:
(u&&d.dataset.restoreChecked==="1"&&(d.checked=!0),delete d.dataset.restoreChecked))}o(C,"toggleOpti\
ons"),window.toggleOptions=C;function M(){const d=get("model-select");if(!d)return;const u=d.value,g=String(
u||"").toLowerCase(),h=g.includes("deepseek"),x=g.startsWith("glm-"),_=get("thinking-options"),L=get(
"reasoning-effort-container"),A=get("enable-thinking"),$=get("thinking-level"),N=get("thinking-budge\
t"),B=get("enable-search"),q=get("search-container"),de=get("url-context-container"),ie=get("enable-\
maps"),F=get("maps-grounding-container"),ne=get("enable-sys-prompt"),U=get("sys-prompt-option"),Me=get(
"enable-python"),W=get("python-container"),re=get("prompt-cache-container"),ve=get("enable-prompt-ca\
che"),ee=u==="gpt-5-search-api",pe=u.includes("tts"),Ee=isMistralOcrModel(u),$e=g.includes("gemini-3\
.1-flash-lite-image"),at=g==="gemini-nano-banana-2.1",Ue=g.includes("gemini-3.1-flash-image")&&!$e||
at,dt=isClaudeModelKey(u),Ct=g==="gemini-3.8-flash-cyber",we=isLlmModel()&&!h&&!pe&&!g.includes("rea\
ltime")&&!g.includes("native-audio")&&!g.includes("live");re&&(we?(re.classList.remove("hidden","opa\
city-50","pointer-events-none"),ve&&(ve.disabled=!1)):(ve&&(ve.checked=!1,ve.disabled=!0),re.classList.
add("opacity-50","pointer-events-none"))),updatePromptCacheUi(),_&&_.classList.add("hidden"),L&&L.classList.
add("hidden");const me=get("vision-model-info");if(me&&me.classList.add("hidden"),L){const be=get("r\
easoning-effort");if(be){Array.from(be.options).forEach(He=>{const ze=g==="gpt-5.6"||g.startsWith("g\
pt-5.6-"),lt=g.startsWith("gpt-6"),cn=g==="gpt-6-sol"||g==="gpt-6-luna",Kt=g==="deepseek-v4.1-flash"||
g==="deepseek-v4-flash-0731"||g==="deepseek-v4-flash"||g==="deepseek-v4-flash-vision-exp",dn=g==="de\
epseek-v4-pro",Fn=g.includes("grok-4.5"),R=g.includes("grok-4.6");He.value==="max"?He.classList.toggle(
"hidden",!ze&&!lt&&!Kt&&!dn):He.value==="xhigh"?He.classList.toggle("hidden",!R&&!g.includes("multi-\
agent")&&!ze&&!lt):He.value==="medium"?He.classList.toggle("hidden",!(g.includes("grok-4.3")||Fn||R||
g.includes("grok-4.20-0309-reasoning")||g.includes("grok-build")||g.includes("multi-agent")||g.includes(
"gpt-5")||lt||g.includes("o1")||g.includes("o3"))):He.value==="none"?He.classList.toggle("hidden",!g.
includes("grok-4.3")&&!g.includes("grok-build")&&!g.includes("gpt-5")&&!cn&&!Kt&&!dn):He.value==="lo\
w"&&He.classList.toggle("hidden",dn)});const Pe=be.selectedOptions&&be.selectedOptions[0];Pe&&Pe.classList.
contains("hidden")&&(be.value=h?"high":"medium")}}de&&de.classList.add("hidden"),F&&F.classList.add(
"hidden"),A&&(A.disabled=!1),N&&(N.disabled=!0,N.classList.add("opacity-50"));const Be=isGeminiImageModelKey(
u);if(pe||Ee)q&&(get("enable-search").checked=!1,q.classList.add("opacity-50","pointer-events-none")),
de&&(get("enable-url-context").checked=!1,de.classList.add("opacity-50","pointer-events-none")),F&&ie&&
(ie.checked=!1,F.classList.add("opacity-50","pointer-events-none")),W&&(Me.checked=!1,W.classList.add(
"opacity-50","pointer-events-none")),ne&&U&&(ne.checked=!1,ne.disabled=!0,U.classList.add("opacity-5\
0"));else if(Ue||$e)F&&ie&&(ie.checked=!1,F.classList.add("hidden","opacity-50","pointer-events-none")),
_.classList.remove("hidden"),at?(Array.from($.options).forEach(be=>{be.disabled=be.value==="low"}),[
"minimal","medium","high"].includes($.value)||($.value="medium")):(Array.from($.options).forEach(be=>{
["low","medium"].includes(be.value)&&(be.disabled=!0),["minimal","high"].includes(be.value)&&(be.disabled=
!1)}),["minimal","high"].includes($.value)||($.value=$e?"minimal":"high")),A&&(A.disabled=!1),$e&&(B&&
(B.checked=!1,B.disabled=!0),q&&q.classList.add("opacity-50","pointer-events-none"));else if(Be)F&&ie&&
(ie.checked=!1,F.classList.add("hidden","opacity-50","pointer-events-none"));else if(dt)_.classList.
remove("hidden"),N&&(N.disabled=!1,N.classList.remove("opacity-50")),Array.from($.options).forEach(be=>{
be.disabled=!0}),W&&(Me.checked=!1,W.classList.add("opacity-50","pointer-events-none"));else if(Ct){
_&&_.classList.remove("hidden"),A&&(A.checked=!0,A.disabled=!0),Array.from($.options).forEach(Pe=>{Pe.
disabled=!["low","medium","high"].includes(Pe.value)}),["low","medium","high"].includes($.value)||($.
value="medium"),[q,de,F,W].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")}),[
B,ie,Me].forEach(Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const be=get("enable-url-context");be&&(be.
checked=!1,be.disabled=!0),ne&&U&&(ne.disabled=!1,U.classList.remove("opacity-50"))}else if(u.includes(
"gemini")&&!Be){_.classList.remove("hidden"),de&&de.classList.remove("hidden","opacity-50","pointer-\
events-none");const be=u.includes("gemini-3");F&&(be?F.classList.remove("hidden","opacity-50","point\
er-events-none"):(ie&&(ie.checked=!1),F.classList.add("hidden","opacity-50","pointer-events-none")));
const Pe=u.includes("flash");Array.from($.options).forEach(He=>{u==="gemini-3.8-flash"||u==="gemini-\
3.7-flash"?He.disabled=!["low","medium","high"].includes(He.value):u==="gemini-3.6-flash"?He.disabled=
!["medium","high"].includes(He.value):u==="gemini-3.5-flash-lite"?He.disabled=!["minimal","medium","\
high"].includes(He.value):["minimal","medium"].includes(He.value)?He.disabled=!Pe:He.disabled=!1}),(u===
"gemini-3.8-flash"||u==="gemini-3.7-flash")&&!["low","medium","high"].includes($.value)||u==="gemini\
-3.6-flash"&&!["medium","high"].includes($.value)?$.value="medium":u==="gemini-3.5-flash-lite"&&!["m\
inimal","medium","high"].includes($.value)?$.value="minimal":!Pe&&["minimal","medium"].includes($.value)&&
($.value="high"),be?A&&(A.checked=!0,A.disabled=!0):A&&(A.disabled=!1),N&&u.includes("gemini-2.5")&&
(N.disabled=!1,N.classList.remove("opacity-50")),N&&!u.includes("gemini-2.5")&&(N.disabled=!0,N.classList.
add("opacity-50"))}if(isLlmModel()&&(g.includes("gpt-5")||g.includes("o1")||g.includes("o3")||g.includes(
"grok-4.3")||g.includes("grok-4.5")||g.includes("grok-4.6")||g.includes("grok-4.20-0309-reasoning")||
g.includes("grok-build")||g.includes("multi-agent")||g.includes("gpt")&&!g.includes("tts")))L.classList.
remove("hidden"),q&&q.classList.remove("opacity-50","pointer-events-none");else if(h){L.classList.remove(
"hidden");const be=get("vision-model-info");if(be&&be.classList.toggle("hidden",g==="deepseek-v4.1-f\
lash"||g==="deepseek-v4-flash-vision-exp"),B&&(B.checked=!1,B.disabled=!0),q&&q.classList.add("opaci\
ty-50","pointer-events-none"),de){const Pe=get("enable-url-context");Pe&&(Pe.checked=!1),de.classList.
add("opacity-50","pointer-events-none")}F&&ie&&(ie.checked=!1,F.classList.add("opacity-50","pointer-\
events-none"))}else Ee||(q&&q.classList.remove("opacity-50","pointer-events-none"),F&&ie&&(ie.checked=
!1,F.classList.add("hidden","opacity-50","pointer-events-none")));if(pe?W&&W.classList.add("opacity-\
50","pointer-events-none"):(W&&W.classList.remove("opacity-50","pointer-events-none"),(!Be||Ue)&&!u.
includes("gpt-image")&&(ne.disabled=!1,U.classList.remove("opacity-50"))),(Be&&!Ue||u.includes("gpt-\
image")||isGrokImageModel()||isIdeogramModel(u)||isGrokVideoModel()||Ee)&&ne&&U&&(ne.checked=!1,ne.disabled=
!0,U.classList.add("opacity-50")),W&&(isLlmModel()?(W.classList.remove("hidden"),Me.disabled=!1):(Me.
checked=!1,Me.disabled=!0,W.classList.add("hidden"))),ee?(B&&(B.checked=!0,B.disabled=!0),q&&q.classList.
add("opacity-50","pointer-events-none"),W&&(Me.checked=!1,Me.disabled=!0,W.classList.add("opacity-50",
"pointer-events-none"))):B&&!u.includes("tts")&&!Ee&&!h&&!$e&&(B.disabled=!1),Ct){[B,ie,Me].forEach(
Pe=>{Pe&&(Pe.checked=!1,Pe.disabled=!0)});const be=get("enable-url-context");be&&(be.checked=!1,be.disabled=
!0),[q,de,F,W].forEach(Pe=>{Pe&&Pe.classList.add("opacity-50","pointer-events-none")})}x&&(me&&me.classList.
toggle("hidden",["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm\
-4.5v"].includes(g)),[B,Me].forEach(be=>{be&&(be.checked=!1,be.disabled=!0)}),[q,W].forEach(be=>{be&&
be.classList.add("opacity-50","pointer-events-none")}));const st=get("mask-btn");st&&(isGptImageModel()?
st.classList.remove("hidden"):(st.classList.add("hidden"),currentMaskImage=null,updateMaskPreview())),
updateTtsUi(),updateStsUi(),updateStsOptions(),l(),c(),m(),f(),b(),y(),v(),updateBatchUi(u),w(),k(),
S(),purgeUnsupportedAttachments(!0),refreshMinimalOptionsIfOpen(),applyMcpPromptChipUi()}o(M,"toggle\
OptionsForModel"),get("model-select")&&(get("model-select").addEventListener("change",C),get("model-\
select").addEventListener("change",()=>schedulePromptTokenEstimate(!0))),bindPromptCacheControls(),C(),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode,!0),renderWelcomeQuickStart();
const P=get("enable-canvas-mode");P&&(P.checked=canvasModeEnabled,P.addEventListener("change",()=>syncCanvasModeUi(
P.checked))),syncCanvasModeUi(canvasModeEnabled,{persist:!1,skipReset:!1});const j=get("enable-codin\
g-mode");j&&(j.checked=codingModeEnabled,j.addEventListener("change",()=>syncCodingModeUi(j.checked))),
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
()=>showToast("\u30B3\u30D4\u30FC\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0))});const Z=get(
"prompt-controls-toggle-btn");Z&&(Z.onclick=()=>togglePromptControlDetails()),get("tts-voice")&&get(
"tts-voice").addEventListener("change",updateTtsUi),get("gpt-image-format")&&get("gpt-image-format").
addEventListener("change",()=>l()),get("gemini-image-size")&&get("gemini-image-size").addEventListener(
"change",()=>c()),get("tts-speed")&&get("tts-speed-label")&&get("tts-speed").addEventListener("input",
()=>{get("tts-speed-label").textContent=`${Number(get("tts-speed").value||1).toFixed(2)}x`}),get("st\
s-speed")&&get("sts-speed-label")&&get("sts-speed").addEventListener("input",()=>{get("sts-speed-lab\
el").textContent=`${Number(get("sts-speed").value||1).toFixed(2)}x`}),window.marked&&typeof window.marked.
use=="function"&&window.marked.use({renderer:{code(d,u,g){const h=(u||"").match(/\S*/)[0];if(h==="py\
exec")return"";if(h==="chat_error")return buildChatErrorBubbleHtml(d||"");const x=d||"",_=(h||"").toLowerCase();
let L="";try{const U=hljs.getLanguage(h)?h:"plaintext";activeStreamingBubbleId&&x.length>2e4?L=escapeHtml(
x):L=hljs.highlight(x,{language:U}).value}catch{L=escapeHtml(x)}const A=encodeURIComponent(x).replace(
/'/g,"%27"),$=detectBlockedScriptsInCode(x),N=hashString(`${h||"TEXT"}
${x||""}`);let B="";if(canvasModeEnabled){const U=String(canvasPreviewState.selectedKey||"")===N,Me=U?
"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B";B=`<button\
 class="canvas-preview-btn${U?" canvas-active":""}" data-code="${A}" data-code-key="${N}" data-canva\
s-lang="${escapeHtml(h||"txt")}" title="${Me}" aria-label="${Me}" aria-pressed="${U?"true":"false"}"\
><i class="fas ${U?"fa-layer-group":"fa-window-restore"}"></i></button>`}else if(isHtmlPreviewCandidate(
_,x)){const U=$?"\u30BB\u30FC\u30D5\u30D7\u30EC\u30D3\u30E5\u30FC":"\u30D7\u30EC\u30D3\u30E5\u30FC";
B=`<button class="html-preview-btn" data-code="${A}" ${$?'data-suspicious="1"':""} title="${U}" aria\
-label="${U}"><i class="fas ${$?"fa-shield-halved":"fa-up-right-from-square"}"></i></button>`}const q=`\
<button class="download-btn" data-code="${A}" data-lang="${h||"txt"}" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30C0\u30A6\u30F3\
\u30ED\u30FC\u30C9"><i class="fas fa-download"></i></button>`,de=_==="diff"?"":`<button class="codin\
g-target-btn" data-code="${A}" data-code-key="${N}" data-coding-lang="${escapeHtml(h||"text")}" aria\
-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas fa-quote-right"></i>\
</button>`,ie=(h||"TEXT")+($?' <span class="suspicious-badge" title="polyfill.io \u306A\u3069\u306E\u5371\u967A\u30B9\u30AF\u30EA\u30D7\u30C8URL\u3092\u691C\u51FA\u3057\u307E\u3057\
\u305F">\u26A0</span>':""),F=getRenderableSvgCode(_,x);return`${F?buildSvgCodeRenderHtml(F,N):""}<di\
v class="code-wrapper collapsed" data-collapsed="true" data-code-key="${N}"><div class="code-header"\
><span class="code-lang">${ie}</span><div class="code-actions"><button class="code-toggle" aria-expa\
nded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button>${de}${B}${q}\
<button class="copy-btn" data-code="${A}" title="\u30B3\u30D4\u30FC" aria-label="\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></\
button></div></div><div class="code-body"><pre><code class="hljs language-${h}">${L}</code></pre></d\
iv></div>`},link(d,u,g){return`<a href="${d}" title="${u||""}" target="_blank">${g}</a>`},image(d,u,g){
return buildChatImageHtml(d,{alt:g,title:u})}},breaks:!0,gfm:!0}),threadObserver=new IntersectionObserver(
d=>{d[0].isIntersecting&&hasMoreThreads&&loadThreads(!0)},{root:get("thread-list"),threshold:.1}),threadObserver.
observe(get("scroll-sentinel")),initLowBandwidthMode(),checkVersion(),(ci=get("version-update-dismis\
s"))==null||ci.addEventListener("click",()=>{const d=localStorage.getItem("app_version")||"";d&&localStorage.
setItem("version_notified",d),hideModal("version-update-modal")});const J=get("version-update-clear-\
cache");if(J&&(J.checked=!!(window.CHAT_CONFIG&&window.CHAT_CONFIG.clearCacheOnVersionUpdate),J.addEventListener(
"change",()=>{versionUpdateCachePreferenceSavePromise=saveVersionUpdateCachePreference(J.checked)})),
(di=get("version-update-reload"))==null||di.addEventListener("click",async()=>{var u;await versionUpdateCachePreferenceSavePromise.
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
d)}}else{const d=get("turnstile-container");d&&d.classList.add("hidden")}const Ce=o(d=>{if(!d)return"\
\u4E0D\u660E";const u=new Date(d);return Number.isNaN(u.getTime())?d:u.toLocaleString()},"formatSess\
ionTime"),T=o(d=>{const u=Array.isArray(d)?d:[],g=get("passkey-list"),h=get("passkey-count");if(h&&(h.
innerText=String(u.length)),!!g){if(!u.length){g.innerHTML='<div class="text-[11px] text-gray-500">\u767B\
\u9332\u6E08\u307F\u306E\u30D1\u30B9\u30AD\u30FC\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
g.innerHTML="",u.forEach((x,_)=>{const L=x&&x.id?String(x.id):"",A=document.createElement("div");A.className=
"bg-gray-800/60 border border-gray-700 rounded p-2 flex items-center justify-between gap-2";const $=document.
createElement("div");$.className="min-w-0";const N=document.createElement("div");N.className="text-x\
s text-gray-200 truncate",N.innerText=x&&x.name?String(x.name):`Security Key ${_+1}`;const B=document.
createElement("div");B.className="text-[10px] text-gray-500 mt-1",B.innerText=x&&x.created_at?`\u767B\u9332\u65E5\u6642:\
 ${Ce(x.created_at)}`:"\u767B\u9332\u65E5\u6642: \u4E0D\u660E",$.appendChild(N),$.appendChild(B),A.appendChild(
$);const q=document.createElement("button");q.type="button",q.className="bg-red-700 hover:bg-red-600\
 text-white px-2 py-1 rounded text-[10px] font-bold btn-hover shrink-0",q.innerText="\u524A\u9664",q.
disabled=!L,L&&(q.onclick=()=>window.removeWebAuthnCredential(L)),A.appendChild(q),g.appendChild(A)})}},
"renderPasskeyList"),I=o(d=>{const u=get("session-list");if(u){if(!d||!d.length){u.innerHTML='<div c\
lass="text-xs text-gray-500">\u30A2\u30AF\u30C6\u30A3\u30D6\u306A\u30BB\u30C3\u30B7\u30E7\u30F3\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}u.innerHTML=d.map(g=>{const h=g.is_current?'<span class="text-[10px] bg-blue-600 text-white p\
x-1.5 py-0.5 rounded">\u73FE\u5728</span>':"",x=g.is_revoked?'<span class="text-[10px] bg-gray-700 t\
ext-gray-300 px-1.5 py-0.5 rounded">\u5931\u52B9</span>':"",_=!g.is_current&&!g.is_revoked?`<button \
data-session-id="${escapeHtml(g.id)}" class="session-revoke-btn bg-gray-700 hover:bg-gray-600 text-w\
hite px-3 py-1 rounded text-[11px] font-bold btn-hover">\u30ED\u30B0\u30A2\u30A6\u30C8</button>`:"",
L=(g.user_agent||"Unknown").slice(0,120),A=g.ip_address||"Unknown";return`<div class="ui-enter-item \
bg-gray-800/60 border border-gray-700 rounded p-3 flex items-center justify-between gap-3"><div clas\
s="min-w-0"><div class="flex items-center gap-2 mb-1">${h}${x}<div class="text-xs text-gray-200">${escapeHtml(
A)}</div></div><div class="text-[11px] text-gray-400 truncate">${escapeHtml(L)}</div><div class="tex\
t-[10px] text-gray-500 mt-1">\u6700\u7D42\u30A2\u30AF\u30BB\u30B9: ${escapeHtml(Ce(g.last_seen_at))}\
 / \u4F5C\u6210: ${escapeHtml(Ce(g.created_at))}</div></div>${_}</div>`}).join(""),u.querySelectorAll(
".session-revoke-btn").forEach(g=>{g.onclick=async()=>{const h=g.getAttribute("data-session-id");if(!h||
!confirm("\u3053\u306E\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;const x=await apiFetch("/api/sessions/revoke",{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({id:h})});let _={};try{_=await x.json()}catch{}if(x.ok){if(_.logged_out){
location.href="/login";return}await O()}else showToast(_&&_.error||"\u30ED\u30B0\u30A2\u30A6\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}})}},"renderSessions"),O=o(async()=>{const d=get("session-list");d&&(d.innerHTML='<div c\
lass="text-xs text-gray-500">\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');const u=await apiFetch("/api/\
sessions");let g={};try{g=await u.json()}catch{}if(!u.ok){if(g&&g.error==="session_revoked"){location.
href="/login";return}d&&(d.innerHTML='<div class="text-xs text-red-400">\u30BB\u30C3\u30B7\u30E7\u30F3\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>');
return}const h=(g.sessions||[]).filter(x=>!x.is_revoked);I(h)},"loadSessions"),K=o(()=>{const d=get(
"session-refresh-btn");d&&(d.onclick=()=>O());const u=get("session-revoke-others-btn");u&&(u.onclick=
async()=>{if(!confirm("\u73FE\u5728\u306E\u7AEF\u672B\u4EE5\u5916\u3092\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_others",{method:"POST"})).ok?await O():showToast("\u64CD\u4F5C\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F","error",!0)});const g=get("session-revoke-all-btn");g&&(g.onclick=async()=>{
if(!confirm("\u5168\u30BB\u30C3\u30B7\u30E7\u30F3\u3092\u5F37\u5236\u30ED\u30B0\u30A2\u30A6\u30C8\u3057\u307E\u3059\u3002\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F"))
return;(await apiFetch("/api/sessions/revoke_all",{method:"POST"})).ok?location.href="/login":showToast(
"\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)})},"bindSessionButtons");if(ensureUserSettingsSnapshot().
then(d=>{d&&(currentVisionModel=d.default_vision_model||"gemini-3-flash-preview"),applyChatDefaults(
d);try{loadMcpServers()}catch{}d&&d.theme_color&&applyThemeColor(d.theme_color,!0),d&&Object.prototype.
hasOwnProperty.call(d,"minimal_prompt_mode")&&d.minimal_prompt_mode?setMinimalPromptMode(!0):d&&Object.
prototype.hasOwnProperty.call(d,"compact_prompt_mode")&&setCompactPromptMode(!!d.compact_prompt_mode),
get("set-client-debug-log")&&syncClientDebugLogToggle(d.enable_client_debug_log===!0,"settings sync");
const u=get("enable-sys-prompt");u&&d&&d.system_prompt&&String(d.system_prompt).trim()&&(!u.disabled&&
!d.default_enable_system_prompt&&!d.use_last_chat_settings&&(u.checked=!0),C())}).catch(()=>{}),installAdminSidebarDebugObserver(),
isAdminSidebarDebugEnabled())try{nativeConsoleInfo(ADMIN_SIDEBAR_DEBUG_PREFIX,"enabled. Open the bro\
wser DevTools Console (F12). After reproducing, run copyAdminSidebarDebug() and paste the result.")}catch{}
snapshotSidebarHistory("page-init"),loadThreads(),loadGems(),get("send-btn").onclick=()=>{isStopMode?
stopGeneration():sendMessage()},get("new-chat-btn").onclick=()=>startNewChat(),bindUploadButton(),bindMinimalOptionsEvents();
const X=get("vision-model-change-btn");X&&(X.onclick=()=>_openVisionModelSelector());const fe=get("c\
ompression-format-only");fe&&(fe.onchange=()=>{const d=fe.checked,u=get("compression-max-size"),g=get(
"compression-max-dim");u&&(u.disabled=d),g&&(g.disabled=d);const h=get("compression-size-wrap"),x=get(
"compression-dim-wrap");h&&(h.style.opacity=d?"0.4":"1"),x&&(x.style.opacity=d?"0.4":"1")});const _e=o(
()=>{const d=get("enable-temporary-chat");!d||d.dataset.bound==="1"||(d.dataset.bound="1",d.checked=
!!temporaryChatEnabled,d.onchange=async()=>{const u=temporaryChatEnabled;await applyTemporaryChatSetting(
d.checked)||(setTemporaryChatUiState(u),ensureTemporaryChatHeartbeat(!1))})},"bindTemporaryChatToggl\
e");_e(),document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&ensureTemporaryChatHeartbeat(
!0)}),window.addEventListener("focus",()=>{ensureTemporaryChatHeartbeat(!0)}),window.addEventListener(
"beforeunload",()=>{stopTemporaryChatHeartbeat(),stopCameraCaptureStream()});const Se=get("storage-u\
sage-refresh");Se&&(Se.onclick=()=>loadStorageUsage());let ae=null;const ce=o(()=>{const d=new Uint8Array(
16);return window.crypto.getRandomValues(d),Array.from(d,u=>u.toString(16).padStart(2,"0")).join("")},
"createAccountTransferId"),G=o((d={})=>{const u=get("account-transfer-progress"),g=get("account-tran\
sfer-progress-bar"),h=get("account-transfer-progress-percent"),x=get("account-transfer-progress-text"),
_=get("account-transfer-progress-detail"),L=Math.max(0,Math.min(100,Number(d.progress)||0));if(u&&u.
classList.remove("hidden"),g&&(g.style.width=`${L}%`),h&&(h.textContent=`${Math.round(L)}%`),x&&(x.textContent=
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
const A=get("account-transfer-cancel-btn");A&&A.classList.toggle("hidden",["ready","completed","fail\
ed","cancelled","expired"].includes(d.phase))},"renderAccountTransferProgress"),E=o(d=>{ge&&(ge.disabled=
!!d);const u=get("account-import-btn");u&&(u.disabled=!!d);const g=get("account-transfer-cancel-btn");
g&&(g.disabled=!d)},"setAccountTransferControls"),H=o((d={})=>{const u=get("account-export-ready"),g=get(
"account-export-ready-text"),h=get("account-export-expiry"),x=get("account-export-download-btn"),_=!!(d.
available&&d.download_url);if(u&&u.classList.toggle("hidden",!_),!_){x&&x.removeAttribute("href");return}
const L=Math.max(0,Number(d.size_bytes)||0),A=L>=1024*1024*1024?`${(L/(1024*1024*1024)).toFixed(2)} \
GB`:`${(L/(1024*1024)).toFixed(1)} MB`;if(g){const $=Number(d.unreadable_count)>0?`\uFF08\u8AAD\u53D6\u4E0D\u80FD ${Number(
d.unreadable_count)}\u4EF6\u3092\u5FA9\u65E7\u7528\u3068\u3057\u3066\u53CE\u9332\uFF09`:"";g.textContent=
`\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u3059\uFF1A${A}${$}`}
if(h){const $=d.expires_at?new Date(d.expires_at):null;h.textContent=$&&!Number.isNaN($.getTime())?`\
\u4FDD\u5B58\u671F\u9650\uFF1A${$.toLocaleString()}\uFF08\u671F\u9650\u5F8C\u306B\u81EA\u52D5\u524A\u9664\uFF09`:
"\u5B8C\u6210\u304B\u30891\u6642\u9593\u5F8C\u306B\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u3059\u3002"}
x&&(x.href=d.download_url)},"renderAccountExportAvailability"),Q=o(async d=>{for(;ae===d&&!d.stopped;){
try{const u=await apiFetch(`/api/account/transfer/${d.id}`,manualSpinnerRequestOptions({cache:"no-st\
ore"})),g=await u.json().catch(()=>({}));if(u.ok&&(g.state!=="pending"&&G(g),["ready","completed","f\
ailed","cancelled","expired"].includes(g.state)))return g}catch{}await new Promise(u=>setTimeout(u,700))}
return null},"pollAccountTransfer"),te=o((d,u,g=!0)=>{u&&(G(u),H(u),g&&u.state==="ready"?showToast(u.
message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u306E\u6E96\u5099\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F",
Number(u.unreadable_count)>0?"warning":"success",Number(u.unreadable_count)>0):g&&u.state==="failed"&&
showToast(u.message||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),Ae(d))},"handleFinishedAccountExport"),se=o(async()=>{try{const d=await apiFetch("/api/a\
ccount/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),u=await d.json().catch(()=>({}));
if(!d.ok)return;if(H(u),u.state==="ready"){G(u);return}if(["failed","cancelled","expired"].includes(
u.state)){G(u);return}if(!["queued","running","cancelling"].includes(u.state)||!u.job_id||ae&&ae.id===
u.job_id||ae)return;const g={id:u.job_id,type:"export",stopped:!1,restored:!0};ae=g,E(!0),G(u);const h=await Q(
g);h&&te(g,h,!0)}catch{}},"refreshLatestAccountExport"),Ae=o(d=>{ae===d&&(ae=null),d.stopped=!0,E(!1)},
"finishAccountTransfer"),oe=get("account-transfer-cancel-btn");oe&&(oe.onclick=async()=>{const d=ae;
if(!(!d||d.stopped)){d.cancelRequested=!0,oe.disabled=!0,G({progress:0,phase:"cancelling",message:"\u30AD\
\u30E3\u30F3\u30BB\u30EB\u3057\u3066\u3044\u307E\u3059"});try{await apiFetch(`/api/account/transfer/${d.
id}/cancel`,manualSpinnerRequestOptions({method:"POST"}))}catch{}d.controller&&d.controller.abort(),
G({progress:0,phase:"cancelled",message:"\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),d.
type==="export"&&H({available:!1}),Ae(d),showToast("\u51E6\u7406\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"info")}});const ge=get("account-export-btn");ge&&(ge.onclick=async()=>{if(ae)return;const d={id:ce(),
type:"export",stopped:!1};ae=d,E(!0),H({available:!1}),G({progress:0,phase:"queued",message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\
\u53D7\u3051\u4ED8\u3051\u3066\u3044\u307E\u3059"});try{const u=await apiFetch("/api/account/export",
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{job_id:d.id}),keepalive:!0})),g=await u.json().catch(()=>({}));if(u.status===409&&g.error==="export\
_in_progress"&&g.job_id)d.id=g.job_id;else if(!u.ok)throw new Error(g.error==="rate_limit"?"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u56DE\u6570\
\u306E\u4E0A\u9650\u306B\u9054\u3057\u307E\u3057\u305F":g.error||"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
G({progress:0,phase:"queued",message:"\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u3067\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\u3057\u3066\u3044\u307E\u3059"});
const h=await Q(d);!d.cancelRequested&&h&&te(d,h,!0)}catch(u){const g=u&&u.message?u.message:"\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8\
\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";G({progress:0,phase:"failed",message:g}),
showToast(g,"error",!0),Ae(d)}});const Y=get("account-export-download-btn");Y&&Y.addEventListener("c\
lick",async d=>{const u=Y.getAttribute("href");if(!(!u||u==="#")){d.preventDefault();try{const g=await apiFetch(
"/api/account/export/latest",manualSpinnerRequestOptions({cache:"no-store"})),h=await g.json().catch(
()=>({}));g.ok&&h.available&&h.download_url?(Y.href=h.download_url,window.location.assign(h.download_url)):
(H(h),G(h),showToast("\u30A8\u30AF\u30B9\u30DD\u30FC\u30C8ZIP\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3067\u304D\u307E\u305B\u3093\u3002\u6700\u65B0\u306E\u72B6\u614B\u3092\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0),se())}catch{window.location.assign(u)}}}),E(!1),se();const ye=get("import-files-grid"),
pt=get("import-files-info"),ct=get("import-files-summary"),mt=o(d=>{const u=Math.max(0,Number(d)||0);
return u>=1024*1024*1024?`${(u/(1024*1024*1024)).toFixed(2)} GB`:u>=1024*1024?`${(u/(1024*1024)).toFixed(
1)} MB`:u>=1024?`${Math.round(u/1024)} KB`:`${u} B`},"importFormatBytes");let Ne=null;const Ve=o(()=>{
if(!Ne)return;const d=Ne.files,u=Ne.selection;let g=0;d.forEach(_=>{u.has(_.archive_path)&&(g+=Number(
_.size_bytes)||0)});const h=Number(Ne.available_bytes)||0,x=g>h;ct&&(ct.textContent=`\u9078\u629E\u4E2D: ${mt(
g)} / \u5229\u7528\u53EF\u80FD: ${mt(h)}${x?" \uFF08\u5BB9\u91CF\u8D85\u904E\uFF09":""}`,ct.classList.
toggle("text-red-300",x)),pt&&(pt.textContent=`${d.length} files`)},"updateImportFileSelectionUi"),bt=o(
()=>{if(!ye||!Ne)return;ye.innerHTML="";const d=Ne.files;if(!d.length){ye.innerHTML='<div class="tex\
t-xs text-gray-500">\u30A4\u30F3\u30DD\u30FC\u30C8\u53EF\u80FD\u306A\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>',
Ve();return}d.forEach(u=>{const g=document.createElement("label"),h=Ne.selection.has(u.archive_path);
g.className=`relative bg-gray-800 border rounded flex items-center gap-2 p-2 cursor-pointer transiti\
on hover:border-blue-500 ${h?"border-blue-500":"border-gray-600"}`,g.innerHTML=`<input type="checkbo\
x" class="import-file-check accent-blue-500 w-4 h-4 shrink-0"${h?" checked":""}><div class="min-w-0 \
flex-1"><div class="text-xs text-gray-200 truncate" title="${escapeHtml(u.display_name)}">${escapeHtml(
u.display_name)}</div><div class="text-[10px] text-gray-500">${mt(u.size_bytes)}</div></div>`;const x=g.
querySelector(".import-file-check");x.addEventListener("change",()=>{x.checked?Ne.selection.add(u.archive_path):
Ne.selection.delete(u.archive_path),g.classList.toggle("border-blue-500",x.checked),g.classList.toggle(
"border-gray-600",!x.checked),Ve()}),ye.appendChild(g)}),Ve()},"renderImportFileItems"),Mt=o(d=>new Promise(
u=>{if(Ne={files:d.files||[],selection:new Set((d.files||[]).map(g=>g.archive_path)),available_bytes:d.
available_bytes,resolve:u},bt(),!get("import-files-modal")){u(null);return}showModal("import-files-m\
odal")}),"showImportFileSelection"),wt=o(d=>{if(hideModal("import-files-modal"),Ne){const u=Ne.resolve;
Ne=null,u(d)}},"closeImportFileSelection"),At=get("import-files-close");At&&(At.onclick=()=>wt(null));
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
enable_client_debug_log:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1"},Jt=o(d=>{
if(d===!0)return"ON";if(d===!1)return"OFF";if(d==null||d==="")return"\u672A\u8A2D\u5B9A";const u=String(
d);return u.length>60?u.slice(0,60)+"\u2026":u},"formatAccountSettingValue");let xt=null;const vt=o(
d=>{if(xt){const u=xt;xt=null,hideModal("settings-confirmation-modal"),u(d)}},"resolveSettingsImport\
Confirmation"),Xt=o(d=>new Promise(u=>{if(!get("settings-confirmation-modal")){u(!0);return}xt=u;const h=Array.
isArray(d&&d.settings_changes)?d.settings_changes:[],x=get("settings-confirmation-list");x&&(h.length?
x.innerHTML=h.map(L=>{const A=Bt[L.field]||L.field,$=Jt(L.current),N=Jt(L.incoming);return`<div clas\
s="rounded border border-gray-700 bg-gray-800/60 p-2">
                                <div class="text-xs font-bold text-gray-100">${escapeHtml(A)}</div>
                                <div class="text-[11px] text-gray-400 mt-1">\u73FE\u5728: ${escapeHtml(
$)}</div>
                                <div class="text-[11px] text-emerald-300">\u2192 ${escapeHtml(N)}</d\
iv>
                            </div>`}).join(""):x.innerHTML='<div class="text-xs text-gray-400">\u5909\u66F4\u3055\u308C\u308B\
\u8A2D\u5B9A\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>');const _=get("settin\
gs-confirmation-count");_&&(_.textContent=`${h.length}\u4EF6\u306E\u8A2D\u5B9A\u304C\u5909\u66F4\u3055\u308C\u307E\u3059`),
showModal("settings-confirmation-modal")}),"showSettingsImportConfirmation"),Ft=get("settings-confir\
mation-modal");Ft&&Ft.addEventListener("click",d=>{d.target===Ft&&vt(!1)});const D=get("settings-con\
firmation-close");D&&(D.onclick=()=>vt(!1));const ue=get("settings-confirmation-cancel");ue&&(ue.onclick=
()=>vt(!1));const Ie=get("settings-confirmation-confirm");Ie&&(Ie.onclick=()=>vt(!0));const Re=get("\
account-import-btn"),Ke=get("account-import-inplace"),tt=get("account-import-inplace-warning");if(Ke&&
tt){const d=o(()=>tt.classList.toggle("hidden",!Ke.checked),"syncInplaceWarn");Ke.addEventListener("\
change",d),d()}Re&&(Re.onclick=async()=>{const d=get("account-import-file"),u=d&&d.files?d.files[0]:
null,g=get("account-import-categories"),h=g?Array.from(g.querySelectorAll('input[type="checkbox"]:ch\
ecked')).map(ie=>ie.value):[],x=get("account-import-inplace"),_=!!(x&&x.checked),L=get("account-impo\
rt-settings-bypass"),A=!!(L&&L.checked);let $=!1;if(!u){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308BZIP\u30D5\u30A1\u30A4\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!h.length){showToast("\u30A4\u30F3\u30DD\u30FC\u30C8\u3059\u308B\u30C7\u30FC\u30BF\u30921\u3064\u4EE5\u4E0A\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const N=g?Array.from(g.querySelectorAll('input[type="checkbox"]:checked')).map(ie=>(ie.
closest("label")&&ie.closest("label").textContent||ie.value).trim()):h;if(!confirm(`\u6B21\u306E\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3059\u3002\u65E2\
\u5B58\u30C7\u30FC\u30BF\u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u3059\u3067\u306B\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u304C\u3042\u308B\u5834\u5408\u306F\u30B9\u30AD\u30C3\u30D7\u3055\u308C\u307E\u3059\u3002

${N.join("\u3001")}${_?`
\u203B\u300C\u5143\u306E\u5834\u6240\u3078\u5FA9\u5143\u300D: \u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u540C\u540D\u30D5\u30A1\u30A4\u30EB\u3092\u4E0A\u66F8\u304D\u3057\u307E\u3059`:
""}

\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))return;const B={id:ce(),type:"import",stopped:!1,controller:new AbortController};
ae=B,E(!0),G({progress:0,phase:"uploading",message:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059"});
const q=get("account-import-result");let de=Promise.resolve(null);try{const F=Math.max(1,Math.ceil(u.
size/10485760)),ne=await apiFetch("/api/account/import/upload/start",manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({size:u.size}),signal:B.controller.
signal})),U=await ne.json().catch(()=>({}));if(!ne.ok)throw new Error(U.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3067\u304D\u307E\u305B\u3093");
B.uploadId=U.upload_id;const Me=U.chunk_size||10485760;let W=0,re=0;const ve=o(async()=>{for(;;){const we=re++;
if(we>=F)return;const me=u.slice(we*Me,Math.min(u.size,(we+1)*Me)),Be=new FormData;Be.append("chunk",
me,u.name),Be.append("index",String(we));const Ye=await apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}/chunk`,manualSpinnerRequestOptions({method:"POST",body:Be,signal:B.controller.signal})),
st=await Ye.json().catch(()=>({}));if(!Ye.ok)throw new Error(st.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
W++,G({progress:Math.min(35,Math.round(W/F*35)),phase:"uploading",message:`ZIP\u3092\u4E26\u5217\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u3066\u3044\u307E\u3059\uFF08${W}\
/${F}\uFF09`}),window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()}},"uploadWorker");
let ee=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),ee=!0);try{await Promise.
all([ve(),ve(),ve()]);const we=await apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}\
/complete`,manualSpinnerRequestOptions({method:"POST",signal:B.controller.signal})),me=await we.json().
catch(()=>({}));if(!we.ok)throw new Error(me.error||"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093");
G({progress:35,phase:"validating",message:"ZIP\u3092\u691C\u8A3C\u3057\u3066\u3044\u307E\u3059"})}finally{
ee&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}let pe="",Ee=!1,$e=0;const at=o(
async()=>{let we=!1;const me=o(()=>{we||(we=!0,setTimeout(()=>{location.reload()},1100))},"scheduleR\
eload");try{const Be=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery,{cache:"no-store"}),Ye=await Be.
json().catch(()=>null);if(!Be.ok||!Ye){me();return}cacheUserSettings(Ye);const st=get("settings-moda\
l");if(st&&st.classList.contains("modal-open"))try{Gn(Ye)}catch{}Ye.theme_color&&applyThemeColor(Ye.
theme_color,!0),Object.prototype.hasOwnProperty.call(Ye,"minimal_prompt_mode")&&Ye.minimal_prompt_mode?
setMinimalPromptMode(!0):Object.prototype.hasOwnProperty.call(Ye,"compact_prompt_mode")&&setCompactPromptMode(
!!Ye.compact_prompt_mode)}catch{}me()},"refreshSettingsFormAfterImport"),Ue=o(we=>{const me=we&&we.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F";q&&(q.textContent=`\u5B8C\u4E86: ${me}`,
q.classList.remove("hidden","text-red-300"),q.classList.add("text-emerald-300")),G({progress:100,phase:"\
completed",message:me}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&at()},"finishImportSuccess"),dt=o(async()=>{
try{const me=await(await apiFetch(`/api/account/transfer/${B.id}`,manualSpinnerRequestOptions({cache:"\
no-store"}))).json().catch(()=>null);return me&&me.state?me:null}catch{return null}},"fetchImportSta\
tus"),Ct=o(async()=>{const we=await dt();if(!we)return{status:"unknown"};if(we.state==="completed")return Ue(
we),{status:"done"};if(["failed","cancelled","expired"].includes(we.state))throw new Error(we.message||
"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");if(we.state==="needs_sel\
ection"&&Array.isArray(we.files)){const me=await Mt({files:we.files,available_bytes:we.available_bytes});
return me===null?(G({progress:0,phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null),{status:"cancelled"}):(pe=me,{status:"reselect"})}if(we.state===
"needs_settings_confirmation"&&Array.isArray(we.settings_changes))return await Xt({settings_changes:we.
settings_changes})?($=!0,{status:"reselect"}):(G({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\
\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(
B.uploadId)}`,manualSpinnerRequestOptions({method:"DELETE"})).catch(()=>null),{status:"cancelled"});
if(we.state==="running"){const me=await Promise.race([de.catch(()=>null),new Promise(Be=>setTimeout(
()=>Be(null),6e4))]);if(me&&me.state==="completed")return Ue(me),{status:"done"};throw me&&["failed",
"cancelled","expired"].includes(me.state)?new Error(me.message||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F"):
new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u51E6\u7406\u304C\u30B5\u30FC\u30D0\u30FC\u5074\u3067\u7D99\u7D9A\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u3057\u3066\u304B\u3089\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044")}
return{status:"unknown"}},"settleUnreadableImport");for(;!Ee;){B.stopped=!0,await de.catch(()=>null),
B.stopped=!1,de=Q(B);let we;try{we=await apiFetch("/api/account/import",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({upload_id:B.uploadId,
categories:h.join(","),job_id:B.id,selected_files:pe,restore_inplace:_,confirm_settings:$||A}),signal:B.
controller.signal}))}catch(ze){if(B.cancelRequested||ze&&ze.name==="AbortError")throw ze;const lt=await Ct();
if(lt.status==="done"){Ee=!0;break}if(lt.status==="cancelled")return;if(lt.status==="reselect")continue;
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
let me=null;try{me=await we.json()}catch{me=null}if(me===null){const ze=await Ct();if(ze.status==="d\
one"){Ee=!0;break}if(ze.status==="cancelled")return;if(ze.status==="reselect")continue;if(we.ok)throw new Error(
"\u30A4\u30F3\u30DD\u30FC\u30C8\u7D50\u679C\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u30DA\u30FC\u30B8\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3057\u3066\u78BA\u8A8D\u3057\u3066\u304F\u3060\u3055\u3044");
if($e<2){$e++;continue}throw new Error("\u30A4\u30F3\u30DD\u30FC\u30C8\u5FDC\u7B54\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u901A\u4FE1\u74B0\u5883\u3092\u3054\u78BA\u8A8D\u306E\u3046\u3048\u3001\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044")}
if(!we.ok&&me.error==="storage_limit_files"&&me.files){const ze=await Mt(me);if(ze===null){G({progress:0,
phase:"cancelled",message:"\u30D5\u30A1\u30A4\u30EB\u9078\u629E\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),
B.uploadId&&apiFetch(`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions(
{method:"DELETE"})).catch(()=>null);return}pe=ze;continue}if(me&&me.status==="settings_confirmation"&&
Array.isArray(me.settings_changes)){if(!await Xt(me)){G({progress:0,phase:"cancelled",message:"\u8A2D\u5B9A\u306E\u30A4\u30F3\
\u30DD\u30FC\u30C8\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F"}),B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null);return}$=!0;continue}if(!we.ok)throw new Error(me.error||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Be=me.imported||{},Ye=[`\u8A2D\u5B9A ${Be.settings||0}\u4EF6`,`API\u8A8D\u8A3C ${Be.api_credentials||
0}\u4EF6`,`\u30C1\u30E3\u30C3\u30C8 ${Be.chats||0}\u4EF6`,`Gem ${Be.gems||0}\u4EF6`,`\u30D5\u30A1\u30A4\u30EB ${Be.
files||0}\u4EF6`,`\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF ${Be.feedback||0}\u4EF6`,`\u8A3A\u65AD\u30C7\u30FC\u30BF ${Be.
diagnostics||0}\u4EF6`].join(" / "),st=me.duplicates||{},be={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"\
Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\
\u8A3A\u65AD\u30C7\u30FC\u30BF"},Pe=[];for(const ze of Object.keys(be)){const lt=Number(st[ze])||0;lt>
0&&Pe.push(`${be[ze]} ${lt}\u4EF6`)}const He=Pe.length?`\uFF08\u91CD\u8907\u3092\u30B9\u30AD\u30C3\u30D7: ${Pe.
join("\u3001")}\uFF09`:"";q&&(q.textContent=`\u5B8C\u4E86: ${Ye}${He}`,q.classList.remove("hidden","\
text-red-300"),q.classList.add("text-emerald-300")),G({progress:100,phase:"completed",message:"\u30A4\u30F3\u30DD\u30FC\u30C8\
\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F"}),showToast("\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u30C7\u30FC\u30BF\u3092\u30A4\u30F3\u30DD\u30FC\u30C8\u3057\u307E\u3057\u305F",
"success"),h.includes("chats")&&loadThreads(),h.includes("gems")&&loadGems(),h.includes("files")&&loadStorageUsage(),
(h.includes("settings")||h.includes("api_credentials"))&&at(),Ee=!0}}catch(ie){if(B.uploadId&&apiFetch(
`/api/account/import/upload/${encodeURIComponent(B.uploadId)}`,manualSpinnerRequestOptions({method:"\
DELETE"})).catch(()=>null),B.cancelRequested||ie&&ie.name==="AbortError")return;const F=ie&&ie.message?
ie.message:"",ne=F==="storage_limit_exceeded"?"\u30B9\u30C8\u30EC\u30FC\u30B8\u4E0A\u9650\u3092\u8D85\u3048\u308B\u305F\u3081\u30A4\u30F3\u30DD\u30FC\u30C8\u3067\u304D\u307E\u305B\u3093":
F||"\u30A4\u30F3\u30DD\u30FC\u30C8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";G({progress:0,phase:"f\
ailed",message:ne}),q&&(q.textContent=ne,q.classList.remove("hidden","text-emerald-300"),q.classList.
add("text-red-300")),showToast(ne,"error",!0)}finally{B.stopped=!0,await de.catch(()=>null),Ae(B)}});
const rt=get("account-dedupe-btn"),qe=get("account-dedupe-result"),je=o((d,u=!1)=>{qe&&(qe.textContent=
d,qe.classList.remove("hidden"),qe.classList.toggle("text-red-300",!!u),qe.classList.toggle("text-em\
erald-300",!u))},"showDedupeResult");rt&&(rt.onclick=async()=>{const d=o(async()=>{const u=await apiFetch(
"/api/account/dedupe/preview",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{})}),g=await u.json().catch(()=>null);if(!u.ok||!g)throw new Error(g&&g.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
if(!g.has_duplicates){je("\u91CD\u8907\u30C7\u30FC\u30BF\u306F\u898B\u3064\u304B\u308A\u307E\u305B\u3093\u3067\u3057\u305F");
return}const h=[],x={chats:"\u30C1\u30E3\u30C3\u30C8",gems:"Gem",files:"\u30D5\u30A1\u30A4\u30EB",feedback:"\
\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF",diagnostics:"\u8A3A\u65AD\u30C7\u30FC\u30BF"};for(const B of[
"chats","gems","files","feedback","diagnostics"]){const q=Number(g.duplicates&&g.duplicates[B])||0;q>
0&&h.push(`${x[B]} ${q}\u4EF6`)}const _=Number(g.kept_referenced_files)>0?`
\u203B\u30C1\u30E3\u30C3\u30C8\u304B\u3089\u53C2\u7167\u3055\u308C\u3066\u3044\u308B\u305F\u3081\u3001\u30D5\u30A1\u30A4\u30EB ${g.
kept_referenced_files}\u4EF6\u306F\u524A\u9664\u305B\u305A\u6B8B\u3057\u307E\u3059\u3002`:"";if(!confirm(
`\u91CD\u8907\u30C7\u30FC\u30BF\u304C ${g.total}\u4EF6 \u898B\u3064\u304B\u308A\u307E\u3057\u305F\u3002

${h.join("\u3001")}${_}

\u540C\u3058\u5185\u5BB9\u306E\u30C7\u30FC\u30BF\u306F\u6700\u3082\u53E4\u30441\u4EF6\u3092\u6B8B\u3057\u3066\u524A\u9664\u3057\u307E\u3059\u3002\u7D9A\u884C\u3057\u307E\u3059\u304B\uFF1F`))
return;const L=await apiFetch("/api/account/dedupe/execute",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({})}),A=await L.json().catch(()=>null);if(!L.ok||!A)throw new Error(
A&&A.error||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const $=[];for(const B of["chats","gems","files","feedback","diagnostics"]){const q=Number(A.removed&&
A.removed[B])||0;q>0&&$.push(`${x[B]} ${q}\u4EF6`)}const N=Number(A.kept_referenced_files)>0?`\uFF08\u53C2\u7167\u306E\u305F\u3081\
\u6B8B\u3057\u305F\u30D5\u30A1\u30A4\u30EB ${A.kept_referenced_files}\u4EF6\uFF09`:"";je(`\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u4FEE\u5FA9\u3057\u307E\
\u3057\u305F: ${$.join("\u3001")||"0\u4EF6"}${N}`),loadThreads(),loadGems(),loadStorageUsage()},"run");
if(!rt.disabled){rt.disabled=!0,je("\u91CD\u8907\u30C7\u30FC\u30BF\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...");
try{await d()}catch(u){je(u&&u.message||"\u91CD\u8907\u30C7\u30FC\u30BF\u306E\u4FEE\u5FA9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
!0)}finally{rt.disabled=!1}}});const Ze=get("site-cache-usage-refresh");Ze&&(Ze.onclick=()=>loadSiteCacheUsage());
const We=get("clear-site-cache-btn");We&&(We.onclick=async()=>{confirm(`\u30B5\u30A4\u30C8\u30AD\u30E3\u30C3\u30B7\u30E5\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
Cookie \u306F\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002`)&&await clearSiteCacheAndReload(We)});
const gt=get("enc-scan-result"),jt=o(async(d=null)=>{gt&&(gt.textContent="\u30B9\u30AD\u30E3\u30F3\u4E2D...");
let u="/api/encryption_scan";d&&(u+=`?thread_id=${encodeURIComponent(d)}`);try{const g=await apiFetch(
u,{cache:"no-store"}),h=await g.json();if(!g.ok){gt&&(gt.textContent=h.error||"\u5931\u6557\u3057\u307E\u3057\u305F");
return}const x=h.total||0,_=h.encrypted||0,L=h.unencrypted||0;let A=`Total: ${x} / Encrypted: ${_} /\
 Plain: ${L}`;if(h.samples&&h.samples.length){const $=h.samples.slice(0,8).map(N=>{const B=N.timestamp?
new Date(N.timestamp).toLocaleString():"";return`#${N.id} (${N.role||""}) ${B}`}).join(" / ");A+=`<d\
iv class="text-[10px] text-gray-400 mt-1">\u4F8B: ${$}</div>`}gt&&(gt.innerHTML=A)}catch{gt&&(gt.textContent=
"\u5931\u6557\u3057\u307E\u3057\u305F")}},"runEncScan"),Yt=get("enc-scan-all");Yt&&(Yt.onclick=()=>jt(
null));const Qt=get("enc-scan-thread");Qt&&(Qt.onclick=()=>currentThreadId?jt(currentThreadId):showToast(
"\u30B9\u30EC\u30C3\u30C9\u304C\u3042\u308A\u307E\u305B\u3093","error",!0));const Te=get("admin-enc-\
list");let De=null,Je=!1;const un=o(d=>!d||!d.length?null:d.some(u=>!!u.is_encrypted),"computeThread\
EncryptedFromMessages"),pn=o(()=>{De=un(allMessages)},"refreshCurrentThreadEncStateFromMessages"),_t=o(
async(d,u,{confirmPrompt:g=!0,reloadCurrent:h=!0}={})=>{if(!d)return showToast("\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093",
"error",!0),!1;const x=u?"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316";if(g&&!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x}\
\u3057\u307E\u3059\u304B\uFF1F`))return!1;Je=!0;try{const _=await apiFetch(`/api/admin/threads/${encodeURIComponent(
d)}/encryption`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({enable:u})}),
L=await _.json().catch(()=>({}));return _.ok?(showToast(`${x}\u3057\u307E\u3057\u305F\uFF08${L.changed||
0}\u4EF6\u3092\u5909\u63DB\uFF09`,"success"),De=!!u,h&&currentThreadId&&String(currentThreadId)===String(
d)&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!0}),Te&&await ke(),!0):
(showToast(L.error||`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1)}catch{return showToast(
`${x}\u306B\u5931\u6557\u3057\u307E\u3057\u305F`,"error",!0),!1}finally{Je=!1}},"setAdminThreadEncry\
ption"),he=o(d=>{if(!Te)return;const u=d.threads||[];if(!u.length){Te.innerHTML='<div class="text-[1\
1px] text-gray-400">\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
Te.innerHTML=u.map(g=>{const h=g.encrypted_count>0?"enc":"plain",x=h==="enc"?"\u5FA9\u53F7\u5316":"\u518D\
\u6697\u53F7\u5316",_=h==="enc"?"bg-amber-600 hover:bg-amber-500":"bg-cyan-700 hover:bg-cyan-600",L=g.
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
                        <button type="button" class="admin-enc-toggle ${_} text-white px-2 py-1 roun\
ded shrink-0" data-id="${A}" data-enable="${h==="enc"?"0":"1"}" data-progress-expected-slow="true">${x}\
</button>
                    </div>`}).join("")},"renderAdminEncThreads"),ke=o(async()=>{if(Te){Te.innerHTML=
'<div class="text-[11px] text-gray-400"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';
try{const d=await apiFetch("/api/admin/threads",{cache:"no-store"}),u=await d.json().catch(()=>({}));
if(!d.ok){Te.innerHTML=`<div class="text-[11px] text-red-400">${escapeHtml(u.error||"\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}\
</div>`;return}if(he(u),currentThreadId&&Array.isArray(u.threads)){const g=u.threads.find(h=>String(
h.thread_id)===String(currentThreadId));g&&(De=!!g.encrypted)}}catch{Te.innerHTML='<div class="text-\
[11px] text-red-400">\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</div>'}}},"l\
oadAdminEncThreads");get("admin-enc-load")&&(get("admin-enc-load").onclick=()=>ke()),window.__loadAdminEncThreads=
ke,window.__refreshAdminThreadEncState=pn,window.__setAdminThreadEncryption=_t;const Xe=get("encrypt\
ion-status-admin-toggle");Xe&&Xe.addEventListener("click",d=>{d.preventDefault(),typeof toggleThreadEncryptionFromModal==
"function"&&toggleThreadEncryptionFromModal()}),Te&&(Te.onclick=async d=>{const u=d.target.closest("\
.admin-enc-open");if(u){d.preventDefault();const A=u.getAttribute("data-id");if(!A)return;typeof Dt==
"function"?Dt():typeof hideModal=="function"&&hideModal("settings-modal");try{await loadMessages(A)}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}
return}const g=d.target.closest(".admin-enc-toggle");if(!g||Je)return;const h=g.getAttribute("data-i\
d"),x=g.getAttribute("data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${x?
"\u518D\u6697\u53F7\u5316":"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;g.disabled=!0;
const L=g.textContent;g.textContent="\u51E6\u7406\u4E2D...";try{await _t(h,x,{confirmPrompt:!1,reloadCurrent:!0})}finally{
g.disabled=!1,g.textContent=L,await ke()}}),get("file-input").onchange=d=>{const u=Array.from(d.target.
files||[]);d.target.value="",u.length&&handleFiles(u)},get("photo-input")&&(get("photo-input").onchange=
d=>{const u=Array.from(d.target.files||[]);d.target.value="",u.length&&handleFiles(u)});const ot=o(d=>{
const u=get("ban-appeal-list");if(u){if(!d||!d.length){u.innerHTML='<div class="text-[11px] text-gra\
y-500">\u73FE\u5728\u3001\u7533\u3057\u7ACB\u3066\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';return}
u.innerHTML=d.map(g=>{const h=g.status||"new",x=g.admin_read_at?'<span class="text-[10px] text-gray-\
500 ml-2">\u65E2\u8AAD</span>':'<span class="text-[10px] text-yellow-300 ml-2">\u672A\u8AAD</span>',
_=g.created_at?new Date(g.created_at).toLocaleString():"",L=g.replied_at?new Date(g.replied_at).toLocaleString():
"",A=g.admin_reply||"";return`
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
application/json"},body:JSON.stringify(d)})).ok&&await ht()}catch{}},"updateBanAppealStatus"),pi=o(()=>{
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
700 rounded p-2 text-xs text-gray-200";return mn.map(h=>{const x=h.mcpLocked===!0,_=x?'<div class="t\
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
                `}).join("")},window.applyAutoSystemPromptConfigToForm=(d,u={})=>{mn.forEach(g=>{const h=u&&
typeof u=="object"?u[g.key]||{}:{},x=get(`${d}-auto-sys-${g.key}-enabled`),_=get(`${d}-auto-sys-${g.
key}-text`);x&&(g.mcpLocked===!0?x.disabled=!0:x.checked=h.enabled!==!1),_&&(_.value=h.text||"",_.placeholder=
h.default_text||"\u81EA\u52D5\u6CE8\u5165\u6587\u8A00")}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()};
const Hn=o((d,u=null)=>{if(u){const g=get(u);g&&(g.checked=!0)}mn.forEach(g=>{const h=get(`${d}-auto\
-sys-${g.key}-enabled`),x=get(`${d}-auto-sys-${g.key}-text`);if(h&&(g.mcpLocked!==!0?h.checked=!0:h.
disabled=!0),x){const _=x.placeholder||"";x.value=_}}),typeof syncMcpAutoSysRows=="function"&&syncMcpAutoSysRows()},
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
                `,u.appendChild(g)},pi(),Dn(),_e();const mi=o(()=>{const d=get("set-default-model");
if(!d)return;const u=d.value;d.innerHTML="",MODELS.forEach(h=>{const x=document.createElement("optgr\
oup");x.label=h.category,(h.items||[]).forEach(_=>{const L=document.createElement("option");L.value=
_.id,L.textContent=_.name,x.appendChild(L)}),d.appendChild(x)});const g=userSettingsSnapshot&&userSettingsSnapshot.
default_model||u||"gemini-3.6-flash";g&&Array.from(d.options).some(h=>h.value===g)&&(d.value=g)},"po\
pulateDefaultModelOptions"),fi=o(()=>{const d=get("set-default-vision-model");if(!d)return;const u=d.
value;d.innerHTML="",MODELS.forEach(h=>{const x=(h.items||[]).filter(L=>{const A=(L.id||"").toLowerCase();
return A.startsWith("gemini-")||A.startsWith("gpt-4o")||A.startsWith("claude-")||A.startsWith("grok-\
3")||["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(
A)});if(x.length===0)return;const _=document.createElement("optgroup");_.label=h.category,x.forEach(
L=>{const A=document.createElement("option");A.value=L.id,A.textContent=L.name+" \u2605",_.appendChild(
A)}),d.appendChild(_)});const g=userSettingsSnapshot&&userSettingsSnapshot.default_vision_model||u||
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
ame")&&(get("set-username").value=d.username);const x=get("2fa-badge"),_=get("disable-2fa-btn");d.is_2fa_enabled?
(x.innerText="ENABLED",x.classList.replace("bg-gray-700","bg-green-600"),x.classList.replace("text-g\
ray-400","text-white"),_.classList.remove("hidden")):(x.innerText="DISABLED",x.classList.replace("bg\
-green-600","bg-gray-700"),x.classList.replace("text-white","text-gray-400"),_.classList.add("hidden")),
get("set-skip-2fa-google")&&(get("set-skip-2fa-google").checked=!!d.skip_2fa_on_google_login),get("s\
et-default-2fa-method")&&(get("set-default-2fa-method").value=d.default_2fa_method||"totp");const L=get(
"set-passkey-only-login"),A=get("passkey-only-note"),$=Array.isArray(d.passkey_credentials)?d.passkey_credentials:
[];if(T($),L){L.checked=!!d.passkey_only_login;const ie=$.length>0||!!d.has_webauthn;L.disabled=!ie,
ie||(L.checked=!1),A&&(ie?A.classList.add("hidden"):A.classList.remove("hidden"))}const N=get("mig-s\
tatus-box"),B=get("mig-progress-text"),q=get("mig-progress-bar");if((d.migration_status||"idle")==="\
processing"){N.classList.remove("hidden");const ie=(d.migration_progress||"").split("/");if(ie.length===
2){const F=parseInt(ie[0]||"0",10),ne=parseInt(ie[1]||"0",10);B&&(B.innerText=`${F} / ${ne}`),q&&ne>
0&&(q.style.width=`${Math.min(100,Math.floor(F/ne*100))}%`)}}else N.classList.add("hidden"),q&&(q.style.
width="0%"),B&&(B.innerText="");settingsModalLoaded=!0,setSettingsSaveEnabled(!0)},"populateSettings\
FormFromData");window.openSettingsModal=async()=>{settingsModalLoaded=!1,setSettingsSaveEnabled(!1),
snapshotSidebarHistory("settings-open-before");const d=await ensureUserSettingsSnapshot();d&&Gn(d);const u=get(
"search-box"),g=u?u.value:"";clearTimeout(searchTimeout);const h=get("settings-search");if(h&&(h.value=
""),filterSettings(),mi(),fi(),showModal("settings-modal"),refreshSettingsTabsScroll(),requestAnimationFrame(
()=>refreshSettingsTabsScroll()),restoreThreadSearchValue(g,"restored-search-box-open"),revealPersistentSidebarLists(),
snapshotSidebarHistory("settings-open-after"),[50,200,400,800].forEach(x=>{setTimeout(()=>{restoreThreadSearchValue(
g,"restored-search-box-"+x+"ms"),snapshotSidebarHistory("settings-open-later-"+x+"ms")},x)}),syncAdaptiveBlurSettingsUi(),
loadStorageUsage(),loadSiteCacheUsage(),se(),Dn(),typeof window.__loadAdminEncThreads=="function")try{
window.__loadAdminEncThreads()}catch{}location.pathname!=="/settings"&&history.pushState({modal:"set\
tings",from:location.pathname},"","/settings"),nt(!0),ht(),d||(settingsModalLoaded=!1,setSettingsSaveEnabled(
!1),showToast("\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u9589\u3058\u3066\u518D\u5EA6\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"error",!0)),gn(),K(),O();try{loadMcpServers()}catch{}};const Dt=o((d=!1)=>{snapshotSidebarHistory("\
settings-close-before"),hideModal("settings-modal"),revealPersistentSidebarLists(),snapshotSidebarHistory(
"settings-close-after"),setTimeout(()=>snapshotSidebarHistory("settings-close-later-300ms"),300),!d&&
location.pathname==="/settings"&&history.back()},"closeSettingsModal"),gi=o(()=>{const d=get("set-th\
eme-color"),u=get("set-theme-color-text"),g=get("theme-reset-btn"),h=document.querySelectorAll("#the\
me-presets .theme-swatch"),x=o((_,L=!0)=>{const A=normalizeHex(_);A&&(applyThemeColor(A,L),syncThemeInputs(
A))},"applyFromValue");d&&d.addEventListener("input",()=>x(d.value,!0)),u&&(u.addEventListener("chan\
ge",()=>{const _=normalizeHex(u.value);if(!_){syncThemeInputs(localStorage.getItem(THEME_STORAGE_KEY)||
THEME_DEFAULT);return}x(_,!0)}),u.addEventListener("keydown",_=>{_.key==="Enter"&&(_.preventDefault(),
u.blur())})),g&&(g.onclick=()=>x(THEME_DEFAULT,!0)),h.forEach(_=>{_.addEventListener("click",()=>x(_.
getAttribute("data-color"),!0))})},"bindThemeControls"),hi=o(()=>{const d=get("reset-global-sys-prom\
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
"close-settings-btn").onclick=()=>Dt();const Un=get("settings-header-close");Un&&(Un.onclick=()=>Dt());
const Ht=get("settings-search");Ht&&(Ht.addEventListener("input",filterSettings),Ht.addEventListener(
"keydown",d=>{if(d.key==="Enter"){const u=get("tab-"+activeSettingsTab);if(!u)return;const g=u.querySelector(
":scope > .settings-match");g&&g.scrollIntoView({behavior:"smooth",block:"start"})}}));const zn=get(
"settings-search-clear");zn&&zn.addEventListener("click",()=>{Ht&&(Ht.value="",filterSettings(),Ht.focus())}),
gi(),hi(),bindModelApiKeySettingsControls(),syncGeminiLocalPyDialogSetting(),syncCompressionSettingsUi();
const Sn=get("set-gemini-local-python-dialog");Sn&&(Sn.onchange=()=>setGeminiLocalPyDialogEnabled(Sn.
checked));const Wn=get("set-gemini-backend");Wn&&(Wn.onchange=()=>syncGeminiBackendUi());const Vn=get(
"set-admin-api-key-mode");Vn&&(Vn.onchange=()=>syncAdminApiKeyModeUi());const Tn=get("set-temp-chat-\
timeout-seconds");Tn&&(Tn.onchange=()=>{applyTemporaryChatTimeoutSeconds(Tn.value)});const Kn=get("s\
lash-command-cancel-btn");Kn&&(Kn.onclick=()=>{hidePendingSlashCommandIndicator();const d=get("promp\
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
"set-e2ee").checked:!1,_=userSettingsSnapshot&&Object.prototype.hasOwnProperty.call(userSettingsSnapshot,
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
tent-Type":"application/json"},body:JSON.stringify(h)});if(L.ok){let A="\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F";
try{const q=await L.json();q&&q.message&&(A=q.message)}catch{}cacheUserSettings(Object.assign({},userSettingsSnapshot||
{},{light_mode_enabled:!!h.light_mode_enabled,liquid_glass_enabled:!!h.liquid_glass_enabled})),window.
applySavedUserSystemPromptSettings({system_prompt:h.system_prompt,system_prompt_enabled:h.system_prompt_enabled,
apply_global_system_prompt:h.apply_global_system_prompt,apply_auto_system_prompt_notices:h.apply_auto_system_prompt_notices,
auto_system_prompt_notices_config:h.auto_system_prompt_notices_config}),Dt();const $=currentUsername,
N=CHAT_CONFIG.enableE2EE;enterToSend=h.enter_to_send,autoSearchOnLinks=h.auto_search_on_links;const B=useSwCache;
useSwCache=h.use_sw_cache,window.CHAT_CONFIG&&(window.CHAT_CONFIG.clearCacheOnVersionUpdate=!!h.clear_cache_on_version_update),
compactPromptMode=h.compact_prompt_mode,minimalPromptMode=h.minimal_prompt_mode,voiceStudioUiEnabled=
h.voice_studio_ui!==!1,temporaryChatTimeoutSeconds=h.temp_chat_timeout_seconds,applyThemeColor(h.theme_color,
!0),syncThemeInputs(h.theme_color),applyLightMode(h.light_mode_enabled),applyLiquidGlassMode(h.liquid_glass_enabled),
applyAdaptiveBlurPreference(get("set-background-blur-mode")?get("set-background-blur-mode").value:adaptiveBlurPreferenceMode),
minimalPromptMode?setMinimalPromptMode(!0):setCompactPromptMode(compactPromptMode),updateStsUi(),B!==
useSwCache&&applyCacheMode(useSwCache,{forceCleanup:!useSwCache}),showToast(A,"success"),syncClientDebugLogToggle(
h.enable_client_debug_log,"settings saved"),h.new_username&&h.new_username!==$?setTimeout(()=>location.
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
x.classList.remove("hidden")),d&&(d.value="");else{const _=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-unban-linked-btn")&&(get("bot-unban-linked-btn").onclick=async()=>{
const d=get("bot-unban-username"),u=d?d.value.trim():"";if(!u){showToast("\u30E6\u30FC\u30B6\u30FC\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}if(!confirm(`\u30E6\u30FC\u30B6\u30FC ${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F`))
return;const g=await apiFetch("/api/bot/unban",{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({username:u,mode:"linked"})}),h=await g.json(),x=get("bot-unban-result");if(g.
ok&&h&&h.status==="ok")x&&(x.textContent=`${u} \u306E\u9023\u9396BAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F`,
x.classList.remove("hidden")),d&&(d.value="");else{const _=h&&h.error?h.error:"\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(_,"error",!0)}}),get("bot-speed-test-btn")&&(get("bot-speed-test-btn").onclick=async()=>{const d=get(
"bot-speed-test-btn"),u=get("bot-speed-test-result");d&&(d.disabled=!0),d&&d.classList.add("opacity-\
60","cursor-not-allowed"),u&&(u.classList.remove("hidden"),u.textContent="\u5B9F\u884C\u4E2D...");try{
const g=o(W=>{u&&(u.textContent=W)},"setBox"),h=o(()=>`${Date.now()}_${Math.random().toString(36).slice(
2)}`,"cacheBust"),x=o((W,re)=>!W||!re||re<=0?0:W*8/(re/1e3)/1e3/1e3,"toMbps"),_=o(W=>Number.isFinite(
W)?`${W.toFixed(0)} ms`:"-","fmtMs"),L=o(W=>Number.isFinite(W)?`${W.toFixed(W>=100?0:1)} Mbps`:"-","\
fmtMbps"),A=o(async(W,re)=>{const ve=await W.json().catch(()=>({}));return ve&&ve.error?ve.error:re},
"parseErr"),$=[];g("\u6E2C\u5B9A\u4E2D... ping");for(let W=0;W<4;W++){const re=performance.now(),ve=await apiFetch(
`/api/speedtest/ping?_=${h()}`,{cache:"no-store"}),ee=performance.now();if(!ve.ok)throw new Error(await A(
ve,"ping_failed"));await ve.json().catch(()=>({})),$.push(ee-re)}const N=$.reduce((W,re)=>W+re,0)/Math.
max(1,$.length),B=Math.min(...$),q=o(async W=>{const re=performance.now(),ve=await apiFetch(`/api/sp\
eedtest/download?bytes=${W}&_=${h()}`,{cache:"no-store"});if(!ve.ok)throw new Error(await A(ve,"down\
load_failed"));const ee=await ve.arrayBuffer(),pe=performance.now();return{bytes:ee.byteLength||W,ms:pe-
re,mbps:x(ee.byteLength||W,pe-re)}},"runDownload");g(`\u6E2C\u5B9A\u4E2D... ping ${_(N)}
\u6E2C\u5B9A\u4E2D... download`);const de=[];for(const W of[2*1024*1024,8*1024*1024])de.push(await q(
W)),g(`\u6E2C\u5B9A\u4E2D... ping ${_(N)}
download ${L(Math.max(...de.map(re=>re.mbps)))}
\u6E2C\u5B9A\u4E2D... upload`);const ie=Math.max(...de.map(W=>W.mbps)),F=o(async W=>{const re=new Uint8Array(
W),ve=performance.now(),ee=await apiFetch(`/api/speedtest/upload?_=${h()}`,{method:"POST",headers:{"\
Content-Type":"application/octet-stream"},body:re,cache:"no-store"}),pe=performance.now();if(!ee.ok)
throw new Error(await A(ee,"upload_failed"));const Ee=await ee.json().catch(()=>({})),$e=Number(Ee.bytes_received||
W)||W;return{bytes:$e,ms:pe-ve,mbps:x($e,pe-ve),serverMs:Number(Ee.server_elapsed_ms||0)||0}},"runUp\
load"),ne=[];for(const W of[1*1024*1024,4*1024*1024])ne.push(await F(W));const U=Math.max(...ne.map(
W=>W.mbps)),Me=["\u7D50\u679C (\u30D6\u30E9\u30A6\u30B6\u21D4\u3053\u306E\u30B5\u30FC\u30D0\u30FC)",
`Ping (avg/min): ${_(N)} / ${_(B)}`,`Download (best): ${L(ie)}`,`Upload (best): ${L(U)}`,`Download r\
uns: ${de.map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,`Upload runs: ${ne.
map(W=>`${Math.round(W.bytes/1024/1024)}MB=${L(W.mbps)}`).join(", ")}`,"\u6CE8\u8A18: fast.com \u306E\u3088\u3046\u306A\u30A4\u30F3\u30BF\u30FC\u30CD\u30C3\u30C8\u5168\u4F53\u306E\u901F\
\u5EA6\u3067\u306F\u306A\u304F\u3001\u3053\u306E\u30A2\u30D7\u30EA\u30B5\u30FC\u30D0\u30FC\u307E\u3067\u306E\u56DE\u7DDA\u901F\u5EA6\u306E\u76EE\u5B89\u3067\u3059\u3002"];
g(Me.join(`
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
querySelector(".ban-appeal-reply"):null,_=x?x.value:"";g&&await _n({id:Number(g),admin_reply:_});return}
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
()=>setMarkerMode("crop"));const Ln=get("marker-color-picker");Ln&&(Ln.oninput=d=>setMarkerColor(d.target.
value),Ln.onchange=d=>setMarkerColor(d.target.value));const Cn=get("marker-opacity");Cn&&(Cn.oninput=
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
(applyMarkerTransform(),renderCropOverlay())});const bi=o(()=>{const d=get("upload-modal");return!!(d&&
!d.classList.contains("hidden"))},"isUploadModalOpen"),qt=get("drop-overlay");let Zt=0;const yi=o(()=>{
bi()||qt&&(qt.classList.remove("hidden"),qt.classList.add("flex"))},"showDropOverlay"),en=o(()=>{Zt=
0,qt&&(qt.classList.add("hidden"),qt.classList.remove("flex"))},"hideDropOverlay");window.hideDropOverlay=
en;const kt=get("upload-dropzone");kt&&(kt.addEventListener("dragover",d=>{d.preventDefault(),kt.classList.
add("dragover")}),kt.addEventListener("dragleave",()=>{kt.classList.remove("dragover")}),kt.addEventListener(
"drop",d=>{d.preventDefault(),d.stopPropagation(),kt.classList.remove("dragover"),en();const u=d.dataTransfer?
d.dataTransfer.files:null;u&&u.length&&handleFiles(u)})),window.addEventListener("dragenter",d=>{!d.
dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes("Files")||(Zt+=1,yi())}),window.
addEventListener("dragover",d=>{!d.dataTransfer||!d.dataTransfer.types||!d.dataTransfer.types.includes(
"Files")||d.preventDefault()}),window.addEventListener("dragleave",d=>{!d.dataTransfer||!d.dataTransfer.
types||!d.dataTransfer.types.includes("Files")||(Zt=Math.max(0,Zt-1),(Zt===0||!d.relatedTarget||d.clientY<=
0||d.clientX<=0||d.clientX>=window.innerWidth||d.clientY>=window.innerHeight)&&en())}),window.addEventListener(
"dragend",()=>{en()}),window.addEventListener("drop",d=>{en(),!(!d.dataTransfer||!d.dataTransfer.files||
d.dataTransfer.files.length===0)&&(d.preventDefault(),!(kt&&kt.contains(d.target))&&handleFiles(d.dataTransfer.
files))});const Jn=get("bot-admin-modal"),vi=o(d=>{if(!d)return"";const u=new Date(d);return isNaN(u.
getTime())?"":u.toLocaleString("ja-JP")},"formatBotLogTime"),wi=o((d,u=[])=>{const g=get("bot-admin-\
list");if(!g)return;if(g.innerHTML="",(!d||!d.length)&&(!u||!u.length)){g.innerHTML='<div class="tex\
t-xs text-gray-400">\u8A72\u5F53\u30E6\u30FC\u30B6\u30FC\u304C\u3044\u307E\u305B\u3093\u3002</div>';
return}const h=o(x=>{const _=Number(x.evidence_count)||0;if(!_)return"\u8A18\u9332\u306A\u3057";const L=vi(
x.last_event_at);return`\u8A18\u9332 ${_}\u4EF6${L?"\u30FB\u6700\u7D42 "+escapeHtml(L):""}`},"logSum\
mary");if((d||[]).forEach((x,_)=>{const L=!!x.is_bot_banned,A=x.bot_detection_enabled!==!1,$=Number(
x.lock_remaining_seconds)||0,N=document.createElement("div");N.className="flex flex-wrap items-cente\
r gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs model-list-animate",N.style.animationDelay=
`${Math.min(_,12)*.02}s`,N.innerHTML=`
                        <div class="flex-1 min-w-0">
                            <div class="text-gray-200 font-bold bot-log-wrap">${escapeHtml(x.username)}\
</div>
                            <div class="text-[10px] text-gray-500">${L?"BAN\u4E2D":"\u6B63\u5E38"}${$?
`\u30FB\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.ceil($/60)}\u5206\uFF09`:""} ${x.bot_ban_reason?
" / "+escapeHtml(x.bot_ban_reason):""}</div>
                            <div class="text-[10px] text-gray-500">${h(x)}</div>
                        </div>
                        <button class="bot-open-log bg-gray-700 hover:bg-gray-600 text-white px-2 py\
-1 rounded" data-user-id="${escapeHtml(String(x.user_id||""))}" data-username="${escapeHtml(x.username)}\
">\u30ED\u30B0</button>
                        <button class="bot-toggle-detect bg-gray-700 hover:bg-gray-600 text-white px\
-2 py-1 rounded" data-user="${escapeHtml(x.username)}" data-enabled="${A?"1":"0"}">${A?"\u691C\u51FAON":
"\u691C\u51FAOFF"}</button>
                        <button class="bot-toggle-ban ${L?"bg-green-600 hover:bg-green-500":"bg-red-\
600 hover:bg-red-500"} text-white px-2 py-1 rounded" data-user="${escapeHtml(x.username)}" data-bann\
ed="${L?"1":"0"}">${L?"\u5358\u72EC\u89E3\u9664":"BAN"}</button>                        ${L?`<button\
 class="bot-toggle-unban-linked bg-rose-600 hover:bg-rose-500 text-white px-2 py-1 rounded" data-use\
r="${escapeHtml(x.username)}">\u9023\u9396\u89E3\u9664</button>`:""}
                        <button class="bot-delete-account bg-red-800 hover:bg-red-700 text-white px-\
2 py-1 rounded" data-progress-expected-slow="true" data-user="${escapeHtml(x.username)}">\u524A\u9664</button>\

                    `,g.appendChild(N)}),u&&u.length){const x=document.createElement("div");x.className=
"text-xs font-bold text-gray-300 pt-3",x.textContent="\u524A\u9664\u6E08\u307F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332",
g.appendChild(x),u.forEach(_=>{const L=document.createElement("div");L.className="flex items-center \
gap-2 bg-gray-900 border border-gray-700 rounded p-2 text-xs",L.innerHTML=`
                            <div class="flex-1 min-w-0">
                                <div class="text-gray-400 font-bold bot-log-wrap">${escapeHtml(_.username||
"ID "+_.user_id)}</div>
                                <div class="text-[10px] text-gray-500">\u524A\u9664\u6E08\u307F\u30FB${h(
_)}</div>
                            </div>
                            <button class="bot-open-log bg-gray-700 hover:bg-gray-600 text-white px-\
2 py-1 rounded" data-user-id="${escapeHtml(String(_.user_id||""))}" data-username="${escapeHtml(_.username||
"")}">\u30ED\u30B0</button>
                        `,g.appendChild(L)})}},"renderBotUsers"),Gt=o(async(d="")=>{const u=get("bot\
-admin-list");u&&(u.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-s\
pin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');try{const g=await apiFetch(`/api/bot/users?q\
=${encodeURIComponent(d)}`),h=await g.json();g.ok&&h&&h.users?wi(h.users,h.deleted_users||[]):(u&&(u.
innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0))}catch{u&&(u.innerHTML='<div class="text-xs text-red-400">\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002</div>'),
showToast("\u30E6\u30FC\u30B6\u30FC\u4E00\u89A7\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},"loadBotUsers"),Mn=o(async()=>{if(!isAdminUser||!(get("bot-admin-modal")||Jn))return;const u=get(
"settings-modal");u&&(u.classList.contains("modal-open")||u.classList.contains("modal-prep"))&&hideModal(
"settings-modal"),window.BotAdminLog&&(window.BotAdminLog.bind(),window.BotAdminLog.showList()),showModal(
"bot-admin-modal"),location.pathname!=="/admin-bots"&&history.pushState({modal:"admin-bots"},"","/ad\
min-bots"),await Gt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")},"openBotAdminM\
odal");window.openBotAdminModal=Mn,window.reloadBotAdminUsers=()=>Gt(get("bot-admin-search")?get("bo\
t-admin-search").value.trim():""),window.closeBotAdminModal=(d=!1)=>{(get("bot-admin-modal")||Jn)&&hideModal(
"bot-admin-modal"),!d&&location.pathname==="/admin-bots"&&history.back()},get("bot-admin-open")&&(get(
"bot-admin-open").onclick=()=>{Mn()}),get("bot-admin-close")&&(get("bot-admin-close").onclick=()=>closeBotAdminModal()),
get("bot-admin-search-btn")&&(get("bot-admin-search-btn").onclick=async()=>{await Gt(get("bot-admin-\
search")?get("bot-admin-search").value.trim():"")}),get("bot-admin-refresh-btn")&&(get("bot-admin-re\
fresh-btn").onclick=async()=>{await Gt("")}),get("bot-admin-search")&&get("bot-admin-search").addEventListener(
"keydown",async d=>{d.key==="Enter"&&await Gt(get("bot-admin-search").value.trim())}),get("bot-admin\
-list")&&(get("bot-admin-list").onclick=async d=>{const u=d.target.closest("button");if(!u)return;if(u.
classList.contains("bot-open-log")){const x=Number(u.getAttribute("data-user-id"));x&&window.BotAdminLog&&
await window.BotAdminLog.open(x,u.getAttribute("data-username")||"");return}const g=u.getAttribute("\
data-user");if(!g)return;let h;if(u.classList.contains("bot-toggle-detect")){const x=u.getAttribute(
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
let x={};try{x=await h.json()}catch{}showToast(x.error||"\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0)}await Gt(get("bot-admin-search")?get("bot-admin-search").value.trim():"")}});const Ut={"\
/settings":{id:"settings-modal",open:o(()=>window.openSettingsModal(),"open")},"/upload":{id:"upload\
-modal",open:o(()=>openUploadModal(),"open")},"/library":{id:"lib-modal",open:o(()=>{ri(!1),showModal(
"lib-modal"),loadLibraryFiles()},"open")},"/history":{id:"history-modal",open:o(()=>window.showHistoryModal(),
"open")},"/branch":{id:"branch-modal",open:o(()=>window.showBranchModal(),"open")},"/batch":{id:"bat\
ch-modal",open:o(()=>window.showBatchModal(),"open")},"/paste":{id:"rich-paste-modal",open:o(()=>openRichPasteModal(),
"open")},"/camera":{id:"camera-capture-modal",open:o(()=>openCameraCaptureModal(),"open")},"/edit-im\
age":{id:"marker-modal",open:o(()=>{},"open")},"/chat-settings":{id:"thread-modal",open:o(()=>window.
openThreadModal(),"open")},"/model":{id:"model-modal",open:o(()=>openModelModal(),"open")},"/token-d\
etails":{id:"token-detail-modal",open:o(()=>showTokenDetailModal(),"open")},"/encryption-status":{id:"\
encryption-status-modal",open:o(()=>showEncryptionStatusModal(),"open")},"/python-execution":{id:"py\
thon-exec-modal",open:o(()=>showPythonExecDetailModal(),"open")},"/gem":{id:"gem-modal",open:o(()=>{
editingGemUuid=null,get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>C\
reate New Gem',showModal("gem-modal")},"open")},"/compression":{id:"compression-modal",open:o(()=>window.
openCompressionModal(),"open")},"/admin-bots":{id:"bot-admin-modal",open:o(()=>Mn(),"open")}},Xn=o((d,u=!1)=>{
switch(d){case"settings-modal":Dt(u);break;case"upload-modal":closeUploadModal(u);break;case"camera-\
capture-modal":closeCameraCaptureModal(u?{skipHistory:!0}:{});break;case"history-modal":window.closeHistoryModal&&
window.closeHistoryModal(u);break;case"lib-modal":window.closeLibModal&&window.closeLibModal(u);break;case"\
branch-modal":window.closeBranchModal&&window.closeBranchModal(u);break;case"batch-modal":window.closeBatchModal&&
window.closeBatchModal(u);break;case"rich-paste-modal":window.closeRichPasteModal&&window.closeRichPasteModal(
u);break;case"marker-modal":window.closeMarkerModal&&window.closeMarkerModal(u);break;case"thread-mo\
dal":window.closeThreadModal&&window.closeThreadModal(u);break;case"model-modal":window.closeModelModal&&
window.closeModelModal(u);break;case"token-detail-modal":closeTokenDetail(u);break;case"encryption-s\
tatus-modal":closeEncryptionModal(u);break;case"python-exec-modal":closePythonExecDetail(u);break;case"\
gem-modal":window.closeGemModal&&window.closeGemModal(u);break;case"compression-modal":window.closeCompressionModal&&
window.closeCompressionModal(u);break;case"bot-admin-modal":window.closeBotAdminModal&&window.closeBotAdminModal(
u);break;case"mcp-decision-modal":typeof submitMcpDecision=="function"?submitMcpDecision("deny"):hideModal(
d);break;case"api-key-required-modal":{const h=get("api-key-modal-cancel-btn");h&&typeof h.onclick==
"function"?h.click():hideModal(d);break}case"lyria-studio-modal":window.closeLyriaStudio?window.closeLyriaStudio():
hideModal(d);break;case"voice-studio-modal":window.VoiceStudio?window.VoiceStudio.close():hideModal(
d);break;case"version-update-modal":const g=localStorage.getItem("app_version")||"";g&&localStorage.
setItem("version_notified",g),hideModal(d);break;default:hideModal(d);break}},"closeModalById");window.
addEventListener("popstate",d=>{let u=!1;Object.values(Ut).forEach(x=>{const _=get(x.id);_&&_.classList.
contains("modal-open")&&location.pathname!==Object.keys(Ut).find(L=>Ut[L].id===x.id)&&(Xn(x.id,!0),u=
!0)});const g=location.pathname.match(/^\/c\/(.+)$/);if(g){const x=decodeURIComponent(g[1]);String(currentThreadId)!==
String(x)&&loadMessages(x,{skipHistory:!0})}else location.pathname==="/"&&currentThreadId&&startNewChat(
{skipHistory:!0});const h=Ut[location.pathname];if(h){const x=get(h.id);x&&!x.classList.contains("mo\
dal-open")&&h.open()}});const Yn=location.pathname;Ut[Yn]&&(history.replaceState({},"","/"),setTimeout(
()=>Ut[Yn].open(),500)),get("easy-login-generate")&&(get("easy-login-generate").onclick=async()=>{const d=get(
"easy-login-mins"),u=d?parseInt(d.value||"5",10):5;if(!confirm(`\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092${u}\
\u5206\u9593\u6709\u52B9\u306B\u3057\u307E\u3059\u304B\uFF1F`))return;const h=await(await apiFetch("\
/api/easy_login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({minutes:u})})).
json();h&&h.temp_password?(get("easy-login-code").textContent=h.temp_password,get("easy-login-exp").
textContent=h.expires_at||"",get("easy-login-result").classList.remove("hidden")):showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u306E\
\u767A\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}),get("easy-login-cancel")&&(get(
"easy-login-cancel").onclick=async()=>{if(!confirm("\u73FE\u5728\u306E\u4E00\u6642\u30D1\u30B9\u30EF\u30FC\u30C9\u767A\u884C\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3059\u304B\uFF1F"))
return;const u=await(await apiFetch("/api/easy_login",{method:"POST",headers:{"Content-Type":"applic\
ation/json"},body:JSON.stringify({cancel:!0})})).json();if(u&&u.cancelled){const g=get("easy-login-r\
esult");g&&g.classList.add("hidden"),showToast("\u7C21\u6613\u30ED\u30B0\u30A4\u30F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"success")}else showToast("\u30AD\u30E3\u30F3\u30BB\u30EB\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)});const xi=4,ki=5*1024*1024;let Tt=[];const An=o(()=>{const d=get("fb-images-list");d&&(d.
innerHTML="",Tt.forEach((u,g)=>{const h=document.createElement("div");h.className="relative w-16 h-1\
6 rounded border border-gray-600 overflow-hidden";const x=document.createElement("img");x.src=u.url,
x.alt=u.file.name,x.className="w-full h-full object-cover";const _=document.createElement("button");
_.type="button",_.className="absolute top-0 right-0 bg-black/70 text-white text-[10px] leading-none \
px-1 py-0.5",_.textContent="\xD7",_.onclick=()=>{URL.revokeObjectURL(u.url),Tt.splice(g,1),An()},h.appendChild(
x),h.appendChild(_),d.appendChild(h)}))},"renderFbImages"),_i=o(()=>{Tt.forEach(d=>URL.revokeObjectURL(
d.url)),Tt=[],An()},"clearFbImages");get("fb-images-add")&&get("fb-images-input")&&(get("fb-images-a\
dd").onclick=()=>get("fb-images-input").click(),get("fb-images-input").onchange=d=>{for(const u of Array.
from(d.target.files||[])){if(!/^image\/(png|jpeg|webp|gif)$/.test(u.type)){showToast("PNG\u30FBJPEG\u30FBWebP\u30FB\
GIF \u306E\u753B\u50CF\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044","error",!0);continue}if(u.
size>ki){showToast("\u753B\u50CF\u306F1\u679A5MB\u307E\u3067\u3067\u3059","error",!0);continue}if(Tt.
length>=xi){showToast("\u6DFB\u4ED8\u3067\u304D\u308B\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059",
"error",!0);break}Tt.push({file:u,url:URL.createObjectURL(u)})}d.target.value="",An()}),get("fb-subm\
it").onclick=async()=>{const d=get("fb-title").value.trim(),u=get("fb-message").value.trim();if(!u){
showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u5185\u5BB9\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const g={title:d,message:u,client:"web",version:window.CHAT_CONFIG&&window.CHAT_CONFIG.
appVersion||""},h=window.ActivityLog?window.ActivityLog.feedbackPayload():null;h&&(g.client_logs=h);
const x=get("fb-attach-chat"),_=!!(x&&x.checked);if(_&&!currentThreadId){showToast("\u30B3\u30D4\u30FC\u3092\u9001\u4FE1\u3059\u308B\u30C1\u30E3\u30C3\u30C8\u304C\u958B\u304B\u308C\
\u3066\u3044\u307E\u305B\u3093","error",!0);return}_&&(g.chat_copy=window.ActivityLog?window.ActivityLog.
chatCopyPayload(currentThreadId):{client:"web",thread_id:String(currentThreadId)});const L=await apiFetch(
"/api/feedback",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)}),
A=L.ok?await L.clone().json().catch(()=>({})):{},$={count:Tt.length,saved:!0};if(L.ok&&$.count)try{const N=new FormData;
Tt.forEach(q=>N.append("images",q.file,q.file.name));const B=await apiFetch(`/api/feedback/${A.public_id}\
/images`,{method:"POST",body:N});$.saved=B.ok}catch{$.saved=!1}(window.ActivityLog?await window.ActivityLog.
reportFeedback(L,h,_,$):L.ok)&&(get("fb-title").value="",get("fb-message").value="",_i(),x&&(x.checked=
!1),gn())};async function Qn(d){if(!confirm(`\u3053\u306E\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u4E00\u7DD2\u306B\u9001\u4FE1\u3057\u305F\u64CD\u4F5C\u30ED\u30B0\u3068\u30C1\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC\u3082\u524A\u9664\u3055\u308C\u307E\u3059\u3002`))
return;if(!(await apiFetch(`/api/feedback/${d}`,{method:"DELETE"})).ok){showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F","error",!0);return}showToast("\u30D5\u30A3\u30FC\u30C9\u30D0\u30C3\u30AF\u3092\u524A\u9664\u3057\u307E\u3057\u305F",
"success"),gn()}o(Qn,"deleteFeedback");async function gn(){const u=await(await apiFetch("/api/feedba\
ck?all=1")).json(),g=get("fb-list");g.innerHTML="",(u.items||[]).filter(_=>!u.is_admin||_.user_id===
void 0||_.user_id===null||!0).forEach(_=>{if(u.is_admin)return;const L=document.createElement("div");
L.className="p-2 rounded border border-gray-700 bg-gray-800/50",L.innerHTML=`<div class="text-[11px]\
 text-gray-400">ID: <span class="select-all font-mono">${escapeHtml(_.public_id||"")}</span> / ${_.created_at}\
</div><div class="font-bold text-sm">${escapeHtml(_.title||"No Title")}</div><div class="text-sm whi\
tespace-pre-wrap">${escapeHtml(_.message)}</div><div class="text-[11px] text-gray-400 mt-1">Status: ${escapeHtml(
_.status)}</div>${_.image_count?`<div class="text-[11px] text-gray-400 mt-1">\u6DFB\u4ED8\u753B\u50CF: ${_.
image_count}\u679A</div>`:""}${_.admin_reply?`<div class="text-[11px] text-green-300 mt-1">Reply: ${escapeHtml(
_.admin_reply)}</div>`:""}<div class="mt-2"><button type="button" class="fb-delete bg-red-700 hover:\
bg-red-600 text-white px-2 py-1 rounded text-[10px]">\u524A\u9664</button></div>`,L.querySelector(".\
fb-delete").onclick=()=>Qn(_.public_id),g.appendChild(L)});const h=get("fb-admin-panel"),x=get("fb-a\
dmin-list");u.is_admin?(h.classList.remove("hidden"),x.innerHTML="",(u.items||[]).forEach(_=>{const L=document.
createElement("div");L.className="p-2 rounded border border-gray-700 bg-gray-800/50 space-y-2",L.innerHTML=
`
                            <div class="text-[11px] text-gray-400">ID: <span class="select-all font-\
mono">${escapeHtml(_.public_id||"")}</span> / user:${_.user_id} / ${_.created_at}</div>
                            <div class="font-bold text-sm">${escapeHtml(_.title||"No Title")}</div>
                            <div class="text-sm whitespace-pre-wrap">${escapeHtml(_.message)}</div>
                            ${[_.log_file&&`\u64CD\u4F5C\u30ED\u30B0: ${_.log_file}`,_.chat_file&&`\u30C1\
\u30E3\u30C3\u30C8\u306E\u30B3\u30D4\u30FC: ${_.chat_file}`,_.image_dir&&`\u6DFB\u4ED8\u753B\u50CF\uFF08${_.
image_count}\u679A\uFF09: ${_.image_dir}`].filter(Boolean).map(A=>`<div class="text-[11px] text-ambe\
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
ded px-2 py-1 text-xs text-white" rows="3" placeholder="\u8FD4\u4FE1\u5185\u5BB9">${escapeHtml(_.admin_reply||
"")}</textarea>
                        `,L.querySelector(".fb-status").value=_.status||"new",L.querySelector(".fb-s\
ave").onclick=async()=>{const A=L.querySelector(".fb-status").value,$=L.querySelector(".fb-reply").value;
await apiFetch(`/api/feedback/${_.public_id}/update`,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({status:A,admin_reply:$})}),gn()},L.querySelector(".fb-delete").onclick=
()=>Qn(_.public_id),x.appendChild(L)})):h.classList.add("hidden")}if(o(gn,"loadFeedback"),window.setupTOTP=
async()=>{const u=await(await apiFetch("/api/2fa/totp/setup",{method:"POST"})).json();get("totp-qr").
src=u.qr_image,get("totp-secret-disp").innerText=u.secret,get("totp-setup-area").classList.remove("h\
idden")},window.enableTOTP=async()=>{const d=get("totp-verify-code").value;if(!d)return;(await apiFetch(
"/api/2fa/totp/enable",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{code:d})})).ok?(showToast("TOTP\u304C\u6709\u52B9\u306B\u306A\u308A\u307E\u3057\u305F","success"),get(
"totp-setup-area").classList.add("hidden"),get("totp-verify-code").value="",openSettingsModal()):showToast(
"\u8A8D\u8A3C\u30B3\u30FC\u30C9\u304C\u6B63\u3057\u304F\u3042\u308A\u307E\u305B\u3093","error",!0)},
window.registerWebAuthn=async()=>{const d=get("register-webauthn-btn"),u=get("webauthn-name"),g=u?String(
u.value||"").trim():"";try{d&&(d.disabled=!0);const h=await apiFetch("/api/2fa/webauthn/register/opt\
ions",{method:"POST"}),x=await h.json();if(!h.ok){showToast(x.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const L=await(await ensureWebAuthnJson()).create({publicKey:x}),A=await apiFetch(
"/api/2fa/webauthn/register/verify",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(Object.assign({},L,{name:g}))}),$=await A.json().catch(()=>({}));A.ok?(u&&(u.value=""),showToast(
"\u30D1\u30B9\u30AD\u30FC\u3092\u767B\u9332\u3057\u307E\u3057\u305F","success"),openSettingsModal()):
showToast($.error||"\u30D1\u30B9\u30AD\u30FC\u767B\u9332\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
_=>_.name.toLowerCase().includes(h)||_.description&&_.description.toLowerCase().includes(h));x[gemSelectedIndex]?
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
"sts-cancel-btn").onclick=()=>{isStsModel()&&Si()}),get("prompt-input")&&get("prompt-input").addEventListener(
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
let it,Oe=null,hn=[],En=!1,bn=null,zt=null,$t=null,nn=null,$n=0,yn=!1,an=null,Wt=null,Lt=null,sn=null,
vn=null,It=null,wn=null,xn=null;function Zn(){const d=get("mic-waveform");if(!d)return[];if(Array.isArray(
wn)&&wn.length)return wn;d.innerHTML="";const u=[];for(let g=0;g<24;g++){const h=document.createElement(
"span");h.className="block rounded-full",h.style.background="rgba(252, 165, 165, 0.92)",h.style.width=
"2px",h.style.transition="height 75ms linear, opacity 75ms linear",h.style.height="2px",h.style.opacity=
"0.4",u.push(h),d.appendChild(h)}return wn=u,u}o(Zn,"ensureMicWaveformBars");function Pt(d,u="hidden"){
const g=get("mic-recording-indicator"),h=get("mic-recording-text");if(g){if(xn&&(clearTimeout(xn),xn=
null),u==="hidden"){g.classList.add("hidden");return}h&&d&&(h.innerText=d),g.classList.remove("hidde\
n"),u==="recording"?g.style.color="rgb(252 165 165)":u==="processing"?g.style.color="rgb(253 224 71)":
g.style.color="rgb(209 213 219)"}}o(Pt,"setMicRecordingIndicator");function ei(){Zn().forEach(u=>{u.
style.height="2px",u.style.opacity="0.35"})}o(ei,"resetMicWaveformBars");function Ot(){if(vn&&(cancelAnimationFrame(
vn),vn=null),sn){try{sn.disconnect()}catch{}sn=null}if(Wt){try{Wt.close()}catch{}Wt=null}Lt=null,It=
null,ei()}o(Ot,"stopMicWaveform");function ti(d){Ot();const u=Zn();if(!u.length)return;const g=window.
AudioContext||window.webkitAudioContext;if(!g)return;try{Wt=new g,Lt=Wt.createAnalyser(),Lt.fftSize=
256,Lt.smoothingTimeConstant=0,sn=Wt.createMediaStreamSource(d),sn.connect(Lt),It=new Uint8Array(Lt.
frequencyBinCount)}catch{Ot();return}const h=o(()=>{if(!Lt||!It)return;Lt.getByteFrequencyData(It);const x=Math.
max(1,Math.floor(It.length/u.length));for(let _=0;_<u.length;_++){const A=(It[Math.min(It.length-1,_*
x)]||0)/255,$=Math.max(2,Math.round(2+A*10));u[_].style.height=`${$}px`,u[_].style.opacity=`${.35+A*
.65}`}vn=requestAnimationFrame(h)},"render");h()}o(ti,"startMicWaveform");function kn(){if(bn&&(clearInterval(
bn),bn=null),nn){try{nn.disconnect()}catch{}nn=null}if(zt){try{zt.close()}catch{}zt=null}$t=null}o(kn,
"stopSilenceMonitor");function ni(d){if(!isStsModel()||!stsOpt("sts-auto-send"))return;kn();const u=window.
AudioContext||window.webkitAudioContext;if(!u)return;zt=new u,$t=zt.createAnalyser(),$t.fftSize=2048,
nn=zt.createMediaStreamSource(d),nn.connect($t);const g=new Uint8Array($t.fftSize),h=getStsSilenceMs(),
x=.02;$n=0,yn=!1,bn=setInterval(()=>{if(!$t)return;$t.getByteTimeDomainData(g);let _=0;for(let A=0;A<
g.length;A++){const $=(g[A]-128)/128;_+=$*$}if(Math.sqrt(_/g.length)>x){yn||(yn=!0),$n=Date.now();return}
yn&&Date.now()-$n>h&&it&&it.state==="recording"&&it.stop()},200)}o(ni,"startSilenceMonitor");const Nn=class Nn{constructor(){
this.ws=null,this.audioContext=null,this.processor=null,this.stream=null,this.rtPlayer=null,this.assistantText=
"",this.assistantThought="",this.inputTranscript="",this.interimInputTranscript="",this.assistantAudioChunks=
[],this.userAudioChunks=[],this.onMessage=null,this.onClose=null,this.onError=null,this.setupComplete=
!1,this.model=null,this.assistantTurnBreak=!1,this.userTurnBreak=!1}async start(u,g,h,x={}){this.model=
h,this.ws=new WebSocket(`${g}?access_token=${u}`),this.ws.binaryType="arraybuffer",this.ws.onopen=()=>{
console.log("Gemini Live WebSocket opened. Sending setup...");const $=!!(x&&x.transcriptionConfig),N={
setup:{model:`models/${h}`,generationConfig:{responseModalities:$?["TEXT"]:["AUDIO"]},inputAudioTranscription:$?
x.transcriptionConfig||{}:{}}};$||(N.setup.outputAudioTranscription={}),x.speechConfig&&(N.setup.generationConfig.
speechConfig=x.speechConfig),x.thinkingConfig&&(N.setup.generationConfig.thinkingConfig=x.thinkingConfig),
x.translationConfig&&(N.setup.generationConfig.translationConfig=x.translationConfig),console.log("S\
ending setup:",JSON.stringify(N)),this.ws.send(JSON.stringify(N))},this.ws.onmessage=$=>this._handleMessage(
$),this.ws.onerror=$=>{console.error("Gemini Live WebSocket error:",$),this.onError&&this.onError($)},
this.ws.onclose=$=>{console.log("Gemini Live WebSocket closed:",$.code,$.reason),this.closedEvent=$,
this.onClose&&this.onClose($)},this.stream=await navigator.mediaDevices.getUserMedia(On());const _=ii(
this.stream,16e3);this.audioContext=_.ctx;const L=_.source;this.processor=this.audioContext.createScriptProcessor(
4096,1,1),this.userAudioChunks=[];const A=new MediaRecorder(this.stream);A.ondataavailable=$=>{$.data.
size>0&&this.userAudioChunks.push($.data)},A.start(500),this.backupRecorder=A,this.processor.onaudioprocess=
$=>{if(!this.ws||this.ws.readyState!==WebSocket.OPEN||!this.setupComplete)return;const N=ai($.inputBuffer.
getChannelData(0),this.audioContext.sampleRate,16e3);!N||!N.byteLength||this.ws.send(JSON.stringify(
{realtimeInput:{audio:{data:btoa(String.fromCharCode.apply(null,new Uint8Array(N))),mimeType:"audio/\
pcm;rate=16000"}}}))},L.connect(this.processor),this.processor.connect(this.audioContext.destination)}_handleMessage(u){
let g=null;try{g=JSON.parse(typeof u.data=="string"?u.data:new TextDecoder().decode(u.data))}catch{return}
if(g.setupComplete&&(console.log("Gemini Live setup complete confirmed"),this.setupComplete=!0),g.serverContent){
const h=g.serverContent;h.modelTurn&&h.modelTurn.parts.forEach(x=>{if(x.text&&(x.thought?(console.log(
"Gemini thought delta:",x.text),this.assistantThought+=x.text):this.model!=="gemini-3.5-transcribe-l\
ive"&&this._appendAssistantText(x.text)),x.inlineData&&x.inlineData.data){const _=x.inlineData.data;
console.log("Gemini audio chunk received, size:",_.length),this.rtPlayer&&this.rtPlayer.addChunk(_);
const L=atob(_),A=new Uint8Array(L.length);for(let $=0;$<L.length;$++)A[$]=L.charCodeAt($);this.assistantAudioChunks.
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
`),this.assistantTurnBreak=!1,this.assistantText+=u)}stop(){this.ws&&this.ws.close(),this.processor&&
this.processor.disconnect(),this.audioContext&&this.audioContext.close(),this.stream&&this.stream.getTracks().
forEach(u=>u.stop()),this.backupRecorder&&this.backupRecorder.stop()}async getFinalData(){const u=new Blob(
this.assistantAudioChunks),g=await this._blobToBase64(u),h=new Blob(this.userAudioChunks),x=await this.
_blobToBase64(h);return{user_text:this.inputTranscript,assistant_text:this.assistantText,assistant_thought:this.
assistantThought,audio_base64:g,user_audio_base64:x}}_blobToBase64(u){return new Promise(g=>{const h=new FileReader;
h.onloadend=()=>g(h.result.split(",")[1]),h.readAsDataURL(u)})}};o(Nn,"GeminiLiveClient");let In=Nn;
const Rn=class Rn{constructor(u=24e3){const g=window.AudioContext||window.webkitAudioContext;this.ctx=
new g({sampleRate:u}),this.nextStartTime=0,this.bufferDelay=.1,this.started=!1}async addChunk(u){if(!this.
ctx)return;const g=atob(u),h=new Uint8Array(g.length);for(let N=0;N<g.length;N++)h[N]=g.charCodeAt(N);
const x=new Int16Array(h.buffer),_=new Float32Array(x.length);for(let N=0;N<x.length;N++)_[N]=x[N]/32768;
const L=this.ctx.createBuffer(1,_.length,this.ctx.sampleRate);L.getChannelData(0).set(_),this.ctx.state===
"suspended"&&await this.ctx.resume();const A=this.ctx.createBufferSource();A.buffer=L,A.connect(this.
ctx.destination),this.started||(this.nextStartTime=this.ctx.currentTime+this.bufferDelay,this.started=
!0);const $=Math.max(this.ctx.currentTime,this.nextStartTime);A.start($),this.nextStartTime=$+L.duration}stop(){
this.ctx&&(this.ctx.close(),this.ctx=null)}};o(Rn,"RealTimeAudioPlayer");let on=Rn;const Bn=class Bn{constructor(){
this.active=!1,this.capturing=!1,this.sessionId=null,this.abortCtrl=null,this.reader=null,this.audioCtx=
null,this.processor=null,this.stream=null,this.rtPlayer=null,this.rateIn=24e3,this.rateOut=24e3,this.
userTranscript="",this.assistantTranscript="",this.assistantThought="",this.speechActive=!1,this.responseDoneCount=
0,this.lastAudioAt=0,this.streamError=null,this.saved=!1,this.saving=!1,this.stopping=!1,this.audioQueue=
[],this.audioFlush=null}isActive(){return this.active}async start(){if(this.active)return;if(this.saving||
this.stopping){showToast("\u524D\u306E\u4F1A\u8A71\u3092\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
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
value:x}=await this.reader.read();if(h)break;g+=u.decode(x,{stream:!0});let _;for(;(_=g.indexOf(`

`))>=0;){const L=g.slice(0,_);g=g.slice(_+2);for(const A of L.split(`
`)){if(!A.startsWith("data: "))continue;let $=null;try{$=JSON.parse(A.slice(6))}catch{continue}this.
_handleEvent($)}}}}catch(h){if(h&&h.name==="AbortError")return;this.active&&(this.streamError=h&&h.message?
h.message:"\u30B9\u30C8\u30EA\u30FC\u30E0\u30A8\u30E9\u30FC")}finally{this.reader=null}}_handleEvent(u){
if(u)switch(u.type){case"audio":this.lastAudioAt=Date.now(),stsOpt("sts-auto-play")&&(this.rtPlayer||
(this.rtPlayer=new on(this.rateOut||24e3),Vt=this.rtPlayer),setStsStatus("\u518D\u751F\u4E2D...",!0),
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
if(this.rtPlayer){try{this.rtPlayer.stop()}catch{}this.rtPlayer=null}Vt=null}_startCapture(){return navigator.
mediaDevices.getUserMedia(On()).then(u=>{this.stream=u;const g=this.rateIn||24e3,h=ii(u,g);this.audioCtx=
h.ctx;const x=h.source,_=this.audioCtx.sampleRate,L=4096;this.processor=this.audioCtx.createScriptProcessor(
L,1,1),this.processor.onaudioprocess=A=>{if(!this.active||!this.capturing)return;const $=A.inputBuffer.
getChannelData(0),N=ai($,_,g);!N||!N.byteLength||this._sendAudio(N)},x.connect(this.processor),this.
processor.connect(this.audioCtx.destination)})}_sendAudio(u){!this.sessionId||!this.active||(this.audioQueue.
push(new Uint8Array(u)),this.audioFlush||(this.audioFlush=this._flushAudio()))}async _flushAudio(){try{
for(;this.audioQueue.length&&this.sessionId;){const u=this.audioQueue.splice(0,this.audioQueue.length),
g=u.reduce(($,N)=>$+N.byteLength,0),h=new Uint8Array(g);let x=0;u.forEach($=>{h.set($,x),x+=$.byteLength});
const _="/api/realtime/audio?session_id="+encodeURIComponent(this.sessionId),L={method:"POST",credentials:"\
include",headers:{"X-CSRF-Token":csrfToken,"Content-Type":"application/octet-stream"},body:h.buffer},
A=window.ProgressSpinner&&typeof window.ProgressSpinner.manualRequestOptions=="function"?window.ProgressSpinner.
manualRequestOptions(L):L;try{await fetch(_,A)}catch{}}}finally{this.audioFlush=null}}_stopCapture(){
if(this.capturing=!1,this.processor){try{this.processor.disconnect()}catch{}this.processor=null}if(this.
stream){try{this.stream.getTracks().forEach(u=>u.stop())}catch{}this.stream=null}if(this.audioCtx){try{
this.audioCtx.close()}catch{}this.audioCtx=null}kn(),Ot()}async stop(){if(!this.active)return;if(this.
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
Bn,"RealtimeVoiceSession");let Pn=Bn;function ii(d,u){const g=window.AudioContext||window.webkitAudioContext;
if(!g)throw new Error("AudioContext not supported");let h=null;try{return h=new g({sampleRate:u}),{ctx:h,
source:h.createMediaStreamSource(d)}}catch{if(h)try{h.close()}catch{}return h=new g,{ctx:h,source:h.
createMediaStreamSource(d)}}}o(ii,"openMicAudioSource");function ai(d,u,g){let h=d;if(u!==g&&u>0&&g>
0){const _=u/g,L=Math.floor(h.length/_),A=new Float32Array(L);for(let $=0;$<L;$++)A[$]=h[Math.min(Math.
floor($*_),h.length-1)];h=A}const x=new Int16Array(h.length);for(let _=0;_<h.length;_++){const L=Math.
max(-1,Math.min(1,h[_]));x[_]=L<0?L*32768:L*32767}return x.buffer}o(ai,"pcm16FromFloat32");const rn=new Pn,
si=(()=>{const g={idle:"bg-gray-600",connecting:"bg-amber-500 animate-pulse",streaming:"bg-emerald-6\
00 animate-pulse",paused:"bg-amber-500",stopped:"bg-gray-600",error:"bg-red-600",closed:"bg-gray-600"};
let h=null,x=null,_=!1,L=null,A=!1,$=0,N=0,B=null,q="idle",de=!1,ie=null;const F=o(R=>document.getElementById(
R),"$"),ne=o(R=>{const V=Object.assign({},R||{});return window.ProgressSpinner&&typeof window.ProgressSpinner.
manualRequestOptions=="function"?window.ProgressSpinner.manualRequestOptions(V):(V.progressSpinner=!1,
V)},"noSpinner");function U(R,V){q=V;const xe=F("lyria-status-text"),le=F("lyria-status-dot");xe&&(xe.
textContent=R),le&&(le.className="w-2 h-2 rounded-full inline-block "+(g[V]||g.idle)),ve(),ee()}o(U,
"setStatus");function Me(){const R=N?Math.floor((Date.now()-N)/1e3):0,V=String(Math.floor(R/60)).padStart(
2,"0"),xe=String(R%60).padStart(2,"0");return`${V}:${xe}`}o(Me,"formatElapsed");function W(){N||(N=Date.
now());const R=F("lyria-elapsed");R&&(R.textContent=Me()),B||(B=window.setInterval(()=>{const V=F("l\
yria-elapsed");V&&(V.textContent=Me())},1e3))}o(W,"startElapsedTimer");function re(){B&&(window.clearInterval(
B),B=null)}o(re,"stopElapsedTimer");function ve(){const R=F("lyria-play-btn"),V=F("lyria-pause-btn"),
xe=F("lyria-stop-btn"),le=F("lyria-reset-btn"),Le=!!h,Fe=q==="streaming"||q==="connecting";if(R){R.disabled=
de;const Qe=R.querySelector("i");Qe&&(Qe.className="fas fa-play")}V&&(V.disabled=de||!Fe),xe&&(xe.disabled=
de||!Le||!Fe),le&&(le.disabled=de||!Le||!Fe)}o(ve,"updateTransportButtons");function ee(){const R=F(
"lyria-save-btn");if(!R)return;const V=!!h&&q!=="idle"&&q!=="connecting"&&q!=="error";R.classList.toggle(
"hidden",!V)}o(ee,"updateSaveButton");function pe(R,V){const xe=F("lyria-prompt-rows");if(!xe)return;
const le=document.createElement("div");le.className="flex items-center gap-2",le.innerHTML=`
                        <input type="text" value="${escapeHtml(R||"")}" placeholder="\u4F8B: minimal tech\
no / warm acoustic guitar" class="flex-1 bg-gray-700 border border-gray-600 rounded px-2 py-1.5 text\
-[11px] text-white outline-none min-w-0" maxlength="4000">
                        <label class="flex items-center gap-1 text-[10px] text-gray-400 shrink-0">
                            <span>w</span>
                            <input type="range" min="0.1" max="5" step="0.1" value="${typeof V=="num\
ber"?V:1}" class="accent-purple-400 w-16">
                            <span class="lyria-weight-label font-mono text-purple-300 w-8 text-right\
">${(typeof V=="number"?V:1).toFixed(1)}</span>
                        </label>
                        <button type="button" data-progress-no-spinner="true" class="lyria-prompt-re\
move shrink-0 w-6 h-6 rounded-full bg-gray-800 hover:bg-red-600 text-gray-400 hover:text-white text-\
[10px] flex items-center justify-center transition btn-hover"><i class="fas fa-times"></i></button>
                    `;const Le=le.querySelector('input[type="range"]'),Fe=le.querySelector(".lyria-w\
eight-label");Le&&Fe&&Le.addEventListener("input",()=>{Fe.textContent=parseFloat(Le.value).toFixed(1)});
const Qe=le.querySelector(".lyria-prompt-remove");Qe&&Qe.addEventListener("click",()=>{xe.querySelectorAll(
".lyria-prompt-row-wrap").length<=1||le.remove()}),le.classList.add("lyria-prompt-row-wrap"),xe.appendChild(
le)}o(pe,"addPromptRow");function Ee(){const R=document.querySelectorAll("#lyria-prompt-rows .lyria-\
prompt-row-wrap"),V=[];return R.forEach(xe=>{const le=xe.querySelector('input[type="text"]'),Le=xe.querySelector(
'input[type="range"]'),Fe=(le?le.value:"").trim();Fe&&V.push({text:Fe,weight:parseFloat(Le?Le.value:
1)||1})}),V}o(Ee,"collectPrompts");function $e(){const R={},V=o(Li=>{const jn=F(Li);return jn&&jn.value!==
""?parseFloat(jn.value):void 0},"num"),xe=V("lyria-bpm");xe!==void 0&&(R.bpm=Math.round(xe));const le=V(
"lyria-guidance");le!==void 0&&(R.guidance=le);const Le=V("lyria-density");Le!==void 0&&(R.density=Le);
const Fe=V("lyria-brightness");Fe!==void 0&&(R.brightness=Fe);const Qe=V("lyria-temperature");Qe!==void 0&&
(R.temperature=Qe);const Ge=F("lyria-scale");Ge&&Ge.value&&(R.scale=Ge.value);const et=F("lyria-mode");
et&&et.value&&(R.music_generation_mode=et.value);const ut=F("lyria-mute-bass"),Nt=F("lyria-mute-drum\
s"),ui=F("lyria-only-bass-drums");return ut&&(R.mute_bass=ut.checked),Nt&&(R.mute_drums=Nt.checked),
ui&&(R.only_bass_and_drums=ui.checked),R}o($e,"collectConfig");function at(){[["lyria-bpm","lyria-bp\
m-label"],["lyria-guidance","lyria-guidance-label"],["lyria-density","lyria-density-label"],["lyria-\
brightness","lyria-brightness-label"],["lyria-temperature","lyria-temperature-label"]].forEach(([V,xe])=>{
const le=F(V),Le=F(xe);!le||!Le||le.addEventListener("input",()=>{const Fe=parseFloat(le.value);Le.textContent=
V==="lyria-bpm"?String(Math.round(Fe)):Fe.toFixed(1)})})}o(at,"bindRangeLabels");function Ue(){if(L){
try{L.close()}catch{}L=null}A=!1,$=0}o(Ue,"resetPlayback");function dt(){if(_=!1,x&&typeof x.abort==
"function")try{x.abort()}catch{}x=null}o(dt,"closeStream");async function Ct(){dt(),x=new AbortController,
_=!0;try{const R=await fetch(`/api/gemini/music/stream?session_id=${encodeURIComponent(h)}`,ne({method:"\
GET",signal:x.signal,headers:{Accept:"text/event-stream"},cache:"no-store"}));if(!R.ok){const Le=await R.
json().catch(()=>({}));throw new Error(Le.error||"\u30B9\u30C8\u30EA\u30FC\u30E0\u63A5\u7D9A\u306B\u5931\u6557\u3057\u307E\u3057\u305F")}
const V=R.body.getReader(),xe=new TextDecoder;let le="";for(;_;){const{done:Le,value:Fe}=await V.read();
if(Le)break;le+=xe.decode(Fe,{stream:!0});const Qe=le.split(`

`);le=Qe.pop();for(const Ge of Qe){const et=Ge.split(`
`).find(Nt=>Nt.startsWith("data: "));if(!et)continue;const ut=et.slice(6);try{const Nt=JSON.parse(ut);
we(Nt)}catch{}}}}catch(R){if(R&&R.name==="AbortError")return;_&&(U("\u30B9\u30C8\u30EA\u30FC\u30E0\u5207\u65AD\u3002\u518D\u63A5\u7D9A\u3057\u307E\u3059\u2026",
"connecting"),window.setTimeout(()=>{_&&h&&Ct()},1200))}finally{_=!1}}o(Ct,"openStream");function we(R){
if(R&&R.snapshot){const V=R.status;if(V==="error"){U("\u30A8\u30E9\u30FC","error"),re();return}if(V===
"closed"||V==="stopped"){U("\u7D42\u4E86","closed"),re();return}U(V==="paused"?"\u4E00\u6642\u505C\u6B62\u4E2D":
"\u63A5\u7D9A\u4E2D...",V==="paused"?"paused":"connecting");return}if(R&&R.audio){U("\u518D\u751F\u4E2D...",
"streaming"),W(),me(R.audio);return}if(R&&R.error){U("\u30A8\u30E9\u30FC: "+R.error,"error"),re();return}
if(R&&R.final){U("\u7D42\u4E86","closed"),re(),ve();return}}o(we,"handleStreamMessage");function me(R){
if(!R)return;if(!L){const Ge=window.AudioContext||window.webkitAudioContext;if(!Ge)return;L=new Ge({
sampleRate:48e3}),A=!1,$=0}let V;try{const Ge=atob(R);V=new Uint8Array(Ge.length);for(let et=0;et<Ge.
length;et++)V[et]=Ge.charCodeAt(et)}catch{return}const xe=new Int16Array(V.buffer),le=Math.floor(xe.
length/2);if(le<1)return;const Le=L.createBuffer(2,le,48e3);for(let Ge=0;Ge<2;Ge++){const et=Le.getChannelData(
Ge);for(let ut=0;ut<le;ut++)et[ut]=xe[ut*2+Ge]/32768}L.state==="suspended"&&L.resume();const Fe=L.createBufferSource();
Fe.buffer=Le,Fe.connect(L.destination),A||($=L.currentTime+.08,A=!0);const Qe=Math.max(L.currentTime,
$);Fe.start(Qe),$=Qe+Le.duration}o(me,"playChunk");async function Be(R,V){const xe=await fetch("/api\
/gemini/music/command",ne({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
Object.assign({session_id:h,type:R},V||{}))})),le=await xe.json().catch(()=>({}));if(!xe.ok)throw new Error(
le.error||"\u30B3\u30DE\u30F3\u30C9\u9001\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return le}
o(Be,"apiCommand");async function Ye(){if(de)return;const R=Ee();if(!R.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\
\u304F\u3060\u3055\u3044","warning",!0);return}de=!0,ve(),U("\u63A5\u7D9A\u4E2D...","connecting");try{
const V=await fetch("/api/gemini/music/start",ne({method:"POST",headers:{"Content-Type":"application\
/json"},body:JSON.stringify({weighted_prompts:R,config:$e()})})),xe=await V.json().catch(()=>({}));if(!V.
ok)throw new Error(xe.error||"\u30BB\u30C3\u30B7\u30E7\u30F3\u958B\u59CB\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
h=xe.session_id,ie=$e(),U("\u63A5\u7D9A\u4E2D...","connecting"),Ct()}catch(V){U("\u30A8\u30E9\u30FC: "+
V.message,"error"),showToast("Lyria RealTime: "+V.message,"error",!0)}finally{de=!1,ve()}}o(Ye,"star\
tSession");async function st(R){if(h){de=!0,ve();try{await Be("control",{action:R}),R==="PLAY"?U("\u518D\u751F\
\u4E2D...","streaming"):R==="PAUSE"?U("\u4E00\u6642\u505C\u6B62\u4E2D","paused"):R==="STOP"?U("\u505C\u6B62\u4E2D",
"stopped"):R==="RESET_CONTEXT"&&U("\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8...",
"connecting")}catch(V){showToast("Lyria RealTime: "+V.message,"error",!0),U("\u30A8\u30E9\u30FC: "+V.
message,"error")}finally{de=!1,ve()}}}o(st,"control");async function be(){if(!h)return;const R=Ee();
if(!R.length){showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}de=!0;try{await Be("prompts",{weighted_prompts:R}),U("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
q==="paused"?"paused":"streaming"),showToast("\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u9069\u7528\u3057\u307E\u3057\u305F",
"success")}catch(V){showToast("Lyria RealTime: "+V.message,"error",!0)}finally{de=!1,ve()}}o(be,"app\
lyPrompts");async function Pe(){if(!h)return;const R=$e(),V=ie||{},xe=R.bpm!==void 0&&R.bpm!==V.bpm,
le=R.scale!==void 0&&R.scale!==V.scale,Le=xe||le;de=!0;try{await Be("config",{config:R,reset_context:Le}),
ie=R,U(Le?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F",q==="paused"?"paused":"streaming"),showToast(
Le?"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F\uFF08\u30B3\u30F3\u30C6\u30AD\u30B9\u30C8\u3092\u30EA\u30BB\u30C3\u30C8\uFF09":
"\u8A2D\u5B9A\u3092\u9069\u7528\u3057\u307E\u3057\u305F","success")}catch(Fe){showToast("Lyria RealT\
ime: "+Fe.message,"error",!0)}finally{de=!1,ve()}}o(Pe,"applyConfig");async function He(){if(h){de=!0,
U("\u4FDD\u5B58\u4E2D...","connecting"),ve();try{const R=await fetch("/api/gemini/music/save",ne({method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h,thread_id:currentThreadId||
null})})),V=await R.json().catch(()=>({}));if(!R.ok)throw new Error(V.error||"\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
U("\u4FDD\u5B58\u3057\u307E\u3057\u305F","closed"),re(),showToast("\u30C1\u30E3\u30C3\u30C8\u306B\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success"),V.thread_id&&(currentThreadId=String(V.thread_id),history.pushState({},"","/c/"+V.thread_id),
get("welcome-screen").classList.add("hidden")),await loadMessages(V.thread_id||currentThreadId),cn(!0)}catch(R){
U("\u30A8\u30E9\u30FC: "+R.message,"error"),showToast("Lyria RealTime: "+R.message,"error",!0)}finally{
de=!1,ve()}}}o(He,"saveSession");async function ze(){if(dt(),h)try{await fetch("/api/gemini/music/ca\
ncel",ne({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({session_id:h})}))}catch{}
h=null,re(),Ue(),U("\u6E96\u5099\u5B8C\u4E86","idle")}o(ze,"cancelSession");function lt(){const R=F(
"lyria-prompt-rows");R&&(R.innerHTML=""),pe("",1),ie=null,N=0,["lyria-bpm","lyria-guidance","lyria-d\
ensity","lyria-brightness","lyria-temperature"].forEach(le=>{const Le=F(le);Le&&(Le.value=le==="lyri\
a-bpm"?"120":le==="lyria-guidance"?"4":le==="lyria-temperature"?"1.1":"0.5")});const V=F("lyria-scal\
e");V&&(V.value="");const xe=F("lyria-mode");xe&&(xe.value="QUALITY"),["lyria-mute-bass","lyria-mute\
-drums","lyria-only-bass-drums"].forEach(le=>{const Le=F(le);Le&&(Le.checked=!1)}),at()}o(lt,"resetC\
ontrols");function cn(R){dt(),h&&fetch("/api/gemini/music/cancel",ne({method:"POST",headers:{"Conten\
t-Type":"application/json"},body:JSON.stringify({session_id:h})})).catch(()=>{}),h=null,_=!1,re(),Ue(),
hideModal("lyria-studio-modal")}o(cn,"closeAndCleanup");function Kt(R){if(!isLyriaRealtimeModel()){showToast(
"Lyria RealTime \u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304B\u3089\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const V=F("lyria-studio-modal");if(V&&V.classList.contains("modal-open")&&h){if(R&&
typeof R=="string"){const le=F("lyria-prompt-rows");le&&(le.innerHTML=""),pe(R,1)}return}if(h&&ze(),
lt(),R&&typeof R=="string"){const le=F("lyria-prompt-rows");le&&(le.innerHTML=""),pe(R,1)}h=null,_=!1,
re(),Ue(),U("\u6E96\u5099\u5B8C\u4E86","idle"),showModal("lyria-studio-modal")}o(Kt,"open");function dn(){
const R=F("lyria-open-studio-btn");R&&R.addEventListener("click",()=>Kt(""));const V=F("lyria-studio\
-close");V&&V.addEventListener("click",()=>cn(!1));const xe=F("lyria-play-btn");xe&&xe.addEventListener(
"click",()=>{if(!h){Ye();return}st("PLAY")});const le=F("lyria-pause-btn");le&&le.addEventListener("\
click",()=>st("PAUSE"));const Le=F("lyria-stop-btn");Le&&Le.addEventListener("click",()=>st("STOP"));
const Fe=F("lyria-reset-btn");Fe&&Fe.addEventListener("click",()=>st("RESET_CONTEXT"));const Qe=F("l\
yria-add-prompt-btn");Qe&&Qe.addEventListener("click",()=>pe("",1));const Ge=F("lyria-apply-prompts-\
btn");Ge&&Ge.addEventListener("click",be);const et=F("lyria-apply-config-btn");et&&et.addEventListener(
"click",Pe);const ut=F("lyria-save-btn");ut&&ut.addEventListener("click",He),at(),lt(),window.openLyriaStudio=
Kt}o(dn,"init");function Fn(){de||cn(!1)}return o(Fn,"requestClose"),{init:dn,open:Kt,requestClose:Fn}})();
si.init(),window.closeLyriaStudio=()=>si.requestClose(),(()=>{let d=null,u=null,g=null;const h="voic\
eDockSettingsOpen",x="\u4F1A\u8A71\u306E\u6587\u5B57\u8D77\u3053\u3057\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002",
_=o(ee=>document.getElementById(ee),"$");function L(){return isStsModel()&&voiceStudioUiEnabled!==!1}
o(L,"isStudioMode");function A(){const ee=get("model-select")?get("model-select").value:"",pe=_("voi\
ce-studio-title");pe&&(ee==="gpt-transcribe"||ee==="gpt-live-transcribe"||ee==="grok-voice-transcrib\
e-2.0-file"?pe.textContent="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\u30B9\u30BF\u30B8\u30AA":ee===
"gemini-3.5-live-translate-preview"?pe.textContent="\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u97F3\u58F0\u7FFB\u8A33\u30B9\u30BF\u30B8\u30AA":
pe.textContent="\u97F3\u58F0\u30B9\u30BF\u30B8\u30AA")}o(A,"updateTitle");function $(){const ee=_("v\
oice-studio-transcript");ee&&(ee.innerHTML=`<div class="voice-studio-placeholder text-[10px] text-gr\
ay-500">${x}</div>`);const pe=_("sts-live-transcript");pe&&(pe.innerHTML="",pe.classList.add("hidden"))}
o($,"resetTranscript");function N(ee,pe,Ee){const $e=ee.querySelectorAll(".voice-studio-line");let at=null;
for(let Ue=$e.length-1;Ue>=0;Ue--)if($e[Ue].dataset.role===pe){at=$e[Ue];break}if(at)at.innerHTML=Ee;else{
const Ue=ee.querySelector(".voice-studio-placeholder");Ue&&Ue.remove();const dt=document.createElement(
"div");dt.className="voice-studio-line",dt.dataset.role=pe,dt.innerHTML=Ee,ee.appendChild(dt)}ee.classList.
remove("hidden"),ee.scrollTop=ee.scrollHeight}o(N,"writeLine");function B(ee,pe){if(!pe||!String(pe).
trim()||!L())return;const at=`<span class="${ee==="user"?"text-cyan-300":"text-gray-100"} font-bold"\
>${escapeHtml(ee==="user"?"\u3042\u306A\u305F":"AI")}:</span> <span class="text-gray-200">${escapeHtml(
pe)}</span>`;[_("voice-studio-transcript"),_("sts-live-transcript")].filter(Boolean).forEach(Ue=>N(Ue,
ee,at))}o(B,"log");function q(ee,pe=!0){const Ee=_("sts-panel"),$e=_("sts-settings-toggle");if(Ee&&Ee.
classList.toggle("settings-open",!!ee),$e&&$e.setAttribute("aria-expanded",ee?"true":"false"),pe)try{
localStorage.setItem(h,ee?"1":"0")}catch{}}o(q,"setSettingsOpen");function de(){try{return localStorage.
getItem(h)==="1"}catch{return!1}}o(de,"readSettingsOpen");let ie=null;function F(){const ee=get("mod\
el-select")?get("model-select").value:"";ee!==ie&&(ie=ee,$()),q(de(),!1),A()}o(F,"syncDock");function ne(){
const ee=_("sts-panel"),pe=_("voice-studio-panel-host");ee&&pe&&ee.parentNode!==pe&&(d=ee.parentNode,
u=ee.nextSibling,pe.appendChild(ee));const Ee=_("file-preview"),$e=_("voice-studio-file-host");Ee&&$e&&
Ee.parentNode!==$e&&(g=Ee.parentNode,$e.appendChild(Ee),$e.classList.remove("hidden"))}o(ne,"movePan\
elIntoModal");function U(){const ee=_("sts-panel");ee&&d&&ee.parentNode!==d&&(u&&u.parentNode===d?d.
insertBefore(ee,u):d.appendChild(ee));const pe=_("file-preview");pe&&g&&pe.parentNode!==g&&g.appendChild(
pe);const Ee=_("voice-studio-file-host");Ee&&Ee.classList.add("hidden"),d=null,u=null,g=null}o(U,"mo\
vePanelBack");function Me(){if(!L())return;ne();const ee=_("sts-panel");ee&&ee.classList.remove("hid\
den"),A(),window.VoiceStudioOpen=!0,showModal("voice-studio-modal")}o(Me,"open");function W(){window.
VoiceStudioOpen=!1,U(),hideModal("voice-studio-modal")}o(W,"close");function re(){window.VoiceStudioOpen&&
W()}o(re,"closeIfOpen");function ve(){window.VoiceStudioOpen=!1;const ee=_("voice-studio-open-btn");
ee&&ee.addEventListener("click",()=>Me());const pe=_("voice-studio-close");pe&&pe.addEventListener("\
click",()=>W());const Ee=_("sts-settings-toggle");Ee&&Ee.addEventListener("click",()=>{const $e=_("s\
ts-panel");q(!($e&&$e.classList.contains("settings-open")))}),window.VoiceStudio={open:Me,close:W,closeIfOpen:re,
log:B,isStudioMode:L,syncDock:F},F()}return o(ve,"init"),{init:ve}})().init();let Vt=null;function oi(){
if(Vt&&(Vt.stop(),Vt=null),an){try{an.pause()}catch{}try{an.src=""}catch{}an=null}}o(oi,"stopStsPlay\
back");async function Ai(d){oi();const u=new Audio;return u.src=d,u.preload="auto",u.autoplay=!0,u.playsInline=
!0,an=u,await u.play(),new Promise(g=>{u.onended=()=>g("ended"),u.onerror=()=>g("error")})}o(Ai,"pla\
yStsAudio");function Si(){if(rn.isActive()){rn._cancel();return}if(Oe){Oe.stop(),Oe=null,oi(),get("m\
ic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
setStsStatus("Canceled",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),800),Ot();return}it&&it.state===
"recording"&&(En=!0,it.stop())}o(Si,"cancelRecording");function On(){if(isStsModel())return{audio:!0};
const u=navigator.mediaDevices&&navigator.mediaDevices.getSupportedConstraints?navigator.mediaDevices.
getSupportedConstraints():{},g={channelCount:1};return u.echoCancellation&&(g.echoCancellation=!1),u.
noiseSuppression&&(g.noiseSuppression=!1),u.autoGainControl&&(g.autoGainControl=!1),{audio:g}}o(On,"\
getMicCaptureConstraints"),get("mic-btn").onclick=async()=>{if(abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\
\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(Oe){setStsStatus("Processing...",!0);const d=Oe;Oe=null,d.stop(),get("mic-bt\
n").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700");try{const u=await d.
getFinalData();if(isGeminiLiveTranscribeModel()&&(u.user_text="\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057",
u.assistant_text=(d.inputTranscript||"").trim(),u.assistant_thought="",!u.assistant_text)){setStsStatus(
"No transcript",!1),setTimeout(()=>setStsStatus("Tap to speak",!1),1e3);return}if(!currentThreadId){
const h=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=
String(h.id),history.pushState({},"","/c/"+h.id),get("welcome-screen").classList.add("hidden")}u.thread_id=
currentThreadId,u.model=get("model-select").value,await apiFetch("/api/gemini/save_sts",{method:"POS\
T",headers:{"Content-Type":"application/json"},body:JSON.stringify(u)}),setStsStatus("Saved",!1),setTimeout(
()=>setStsStatus("Tap to speak",!1),1e3),await loadMessages(currentThreadId)}catch(u){console.error(
"Failed to save Gemini Live session:",u),setStsStatus("Error saving",!1)}return}if(rn.isActive()){get(
"mic-btn").classList.remove("bg-red-600","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),
rn.stop();return}if(it&&it.state==="recording"){it.stop(),get("mic-btn").classList.remove("bg-red-60\
0","animate-pulse"),get("mic-btn").classList.add("bg-gray-700"),isStsModel()||Pt("\u9332\u97F3\u3092\u51E6\u7406\u4E2D\u2026",
"processing"),isStsModel()&&setStsStatus("Processing...",!0);return}try{if(isStsModel())try{const g=new Audio;
g.src="data:audio/wav;base64,UklGRiQAAABXQVZFRm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",g.play().
catch(()=>{})}catch{}if(isGeminiLiveModel()){setStsStatus("Connecting...",!0);try{const h={model:get(
"model-select").value};if(isGeminiLiveTranscribeModel()){if(h.transcription_mode=get("sts-transcribe\
-mode")?get("sts-transcribe-mode").value:"VERBATIM",get("sts-custom-vocab")){const F=get("sts-custom\
-vocab").value.split(/[,、\n]/).map(ne=>ne.trim()).filter(Boolean);F.length&&(h.custom_vocabulary=F.
slice(0,1e3))}}else h.voice=get("sts-voice")?get("sts-voice").value:"Kore",isGeminiLiveExtendedThinkingModel()&&
(h.thinking_level=get("sts-thinking-level")?get("sts-thinking-level").value:"medium",h.include_thoughts=
get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1),isGeminiLiveTranslateModel()&&get(
"sts-target-lang")&&(h.target_lang=get("sts-target-lang").value);const x=await apiFetch("/api/gemini\
/session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(h)});if(!x.
ok)throw new Error("Failed to get session token");const{token:_,url:L}=await x.json(),A=get("model-s\
elect").value,$=get("sts-voice")?get("sts-voice").value:"Kore",N=get("sts-thinking-level")?get("sts-\
thinking-level").value:"minimal",B=get("sts-include-thoughts")?get("sts-include-thoughts").checked:!1;
if(Oe=new In,stsOpt("sts-auto-play")&&!isGeminiLiveTranscribeModel()&&(Oe.rtPlayer=new on),isGeminiLiveTranscribeModel()){
const F=get("sts-transcribe-mode")?get("sts-transcribe-mode").value:"VERBATIM",ne={languageCodes:[]};
if((F==="SMART"||F==="VERBATIM")&&(ne.mode=F),get("sts-custom-vocab")){const U=get("sts-custom-vocab").
value.split(/[,、\n]/).map(Me=>Me.trim()).filter(Boolean);U.length&&(ne.customVocabulary=U.slice(0,
1e3))}await Oe.start(_,L,A,{transcriptionConfig:ne})}else if(isGeminiLiveTranslateModel()){const F=get(
"sts-target-lang")?get("sts-target-lang").value:"ja";await Oe.start(_,L,A,{translationConfig:{targetLanguageCode:F,
echoTargetLanguage:!0}})}else{const F={speechConfig:{voiceConfig:{prebuiltVoiceConfig:{voiceName:$}}}};
isGeminiLiveExtendedThinkingModel()&&(F.thinkingConfig={thinkingLevel:N,includeThoughts:B}),await Oe.
start(_,L,A,F)}it=Oe.backupRecorder,it.onstop=()=>{Oe&&get("mic-btn").click()};let q=!0,de="live-sts\
-"+Date.now();Oe.onMessage=F=>{if(F.serverContent){if(isGeminiLiveTranscribeModel()){const ne=Oe.interimInputTranscript,
U=Oe.inputTranscript,Me=U+(ne&&!U.endsWith(ne)?(U?`
`:"")+ne:""),W=get("chat-messages");let re=document.getElementById(de);re||(re=document.createElement(
"div"),re.id=de,re.className="flex flex-col gap-2 mb-4 assistant-message bg-slate-800/40 p-3 rounded\
-lg border border-slate-700/50",re.innerHTML=`
                                                <div class="text-[10px] text-teal-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-microphone"></i> Gemini 3.5 Tra\
nscribe Live
                                                </div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,W.appendChild(re),W.scrollTop=W.scrollHeight);const ve=re.
querySelector(".message-content");ve.innerText=Me||"\u8074\u304D\u53D6\u308A\u4E2D...",W.scrollTop=W.
scrollHeight,window.VoiceStudio&&U&&window.VoiceStudio.log("user",U);return}if(F.serverContent.modelTurn){
q&&(setStsStatus("Gemini is speaking...",!1),q=!1);const ne=get("chat-messages");let U=document.getElementById(
de);U||(U=document.createElement("div"),U.id=de,U.className="flex flex-col gap-2 mb-4 assistant-mess\
age bg-slate-800/40 p-3 rounded-lg border border-slate-700/50",U.innerHTML=`
                                                <div class="text-[10px] text-cyan-400 font-bold uppe\
rcase tracking-wider flex items-center gap-2">
                                                    <i class="fas fa-robot"></i> Gemini Live (Stream\
ing)
                                                </div>
                                                <div class="thought-container hidden italic text-sla\
te-400 text-xs border-l-2 border-slate-600 pl-2 my-1"></div>
                                                <div class="message-content text-sm text-slate-100 l\
eading-relaxed"></div>
                                            `,ne.appendChild(U),ne.scrollTop=ne.scrollHeight);const Me=U.
querySelector(".thought-container"),W=U.querySelector(".message-content");Oe.assistantThought&&(Me.classList.
remove("hidden"),Me.innerText=Oe.assistantThought),W.innerText=Oe.assistantText,ne.scrollTop=ne.scrollHeight,
window.VoiceStudio&&(Oe.inputTranscript&&window.VoiceStudio.log("user",Oe.inputTranscript),Oe.assistantText&&
window.VoiceStudio.log("assistant",Oe.assistantText))}}},setStsStatus("Listening...",!0),get("mic-bt\
n").classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),ti(Oe.
stream),ni(Oe.stream);const ie=Oe;ie.onError=F=>{const ne=F&&F.message?F.message:typeof F=="string"?
F:"";ne&&showToast("Gemini Live: "+ne,"error",!0)},ie.onClose=F=>{if(Oe===ie){if(F&&F.code&&F.code!==
1e3){const ne=F.reason?": "+F.reason:" (code "+F.code+")";showToast("Gemini Live \u306E\u63A5\u7D9A\u304C\u7D42\u4E86\u3057\u307E\u3057\u305F"+
ne,"error",!0)}get("mic-btn").click()}},ie.closedEvent&&ie.onClose(ie.closedEvent);return}catch(g){showToast(
"Gemini Live connection failed: "+g.message,"error",!0),setStsStatus("Error",!1);return}}if(isRealtimeSessionModel()){
await rn.start();return}isStsModel()||(ei(),Pt("\u9332\u97F3\u6E96\u5099\u4E2D\u2026","processing"));
const d=await navigator.mediaDevices.getUserMedia(On());it=new MediaRecorder(d),hn=[],En=!1;const u=isStsModel();
it.ondataavailable=g=>hn.push(g.data),it.onstop=async()=>{if(En){hn=[],get("file-preview").classList.
add("hidden"),d.getTracks().forEach(L=>L.stop()),kn(),Ot(),u||(Pt("\u9332\u97F3\u3092\u30AD\u30E3\u30F3\u30BB\u30EB\u3057\u307E\u3057\u305F",
"idle"),xn=setTimeout(()=>Pt("","hidden"),900)),isStsModel()&&setStsStatus("Canceled",!1),setTimeout(
()=>{isStsModel()&&setStsStatus("Tap to speak",!1)},800);return}const g=new Blob(hn,{type:"audio/web\
m"}),h=new File([g],"recording.webm",{type:"audio/webm"}),x=new FormData;x.append("file",h),get("fil\
e-preview").classList.remove("hidden");const _=u;get("file-name").innerText=_?"Processing voice...":
"Transcribing...";try{if(_){if(!currentThreadId){const U=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=U.id!==null&&U.id!==void 0?String(U.id):U.id,setTemporaryChatUiState(!!(U&&U.
is_temporary)),setCurrentChatHeaderTitle(U&&U.title),applyTemporaryChatRuntimeMeta(U||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+U.id),get("welcome-screen").classList.add("hidden")}currentThreadId&&
activeGem&&(threadGemMap[currentThreadId]=activeGem,pendingGemForNewThread=null),x.append("model",get(
"model-select").value),x.append("thread_id",currentThreadId),get("sts-voice")&&x.append("sts_voice",
get("sts-voice").value||""),get("sts-reasoning-effort")&&x.append("sts_reasoning_effort",get("sts-re\
asoning-effort").value||""),get("sts-speed")&&x.append("sts_speed",get("sts-speed").value||""),get("\
sts-rate-in")&&x.append("sts_rate_in",get("sts-rate-in").value||""),get("sts-rate-out")&&x.append("s\
ts_rate_out",get("sts-rate-out").value||""),get("sts-thinking-level")&&x.append("sts_thinking_level",
get("sts-thinking-level").value||""),get("sts-include-thoughts")&&x.append("sts_include_thoughts",get(
"sts-include-thoughts").checked?"true":""),setStsStatus("Sending audio...",!0);const L=await apiFetch(
"/sts",{method:"POST",body:x});if(!L.ok){const ne=await L.json().catch(()=>({}));throw new Error(ne.
error||"Speech-to-speech failed")}const A=L.body.getReader(),$=new TextDecoder;let N="",B=null,q=null;
stsOpt("sts-auto-play")&&(q=new on,Vt=q),setStsStatus(isTranscriptionModel()?"Transcribing...":"Proc\
essing audio...",!0);let de=!0,ie="",F="";for(;;){const{done:ne,value:U}=await A.read();if(ne)break;
N+=$.decode(U,{stream:!0});const Me=N.split(`
`);N=Me.pop();for(const W of Me){if(!W.trim())continue;const re=JSON.parse(W);if(re.error)throw new Error(
re.error);re.audio_delta&&q&&(de&&(setStsStatus("Playing response...",!1),de=!1),await q.addChunk(re.
audio_delta)),re.input_delta&&(ie+=re.input_delta,window.VoiceStudio&&window.VoiceStudio.log("user",
ie)),re.transcript_delta&&(F+=re.transcript_delta,window.VoiceStudio&&window.VoiceStudio.log("assist\
ant",F)),(re.final||re.audio_url)&&(B=re)}}window.VoiceStudio&&!ie.trim()&&window.VoiceStudio.log("u\
ser","\uFF08\u97F3\u58F0\u30E1\u30C3\u30BB\u30FC\u30B8\uFF09"),B&&(B.audio_url||B.transcription_only)&&
(stsOpt("sts-auto-restart")&&isStsModel()?setTimeout(()=>{setStsStatus("Listening...",!0),get("mic-b\
tn").click()},500):setStsStatus("Tap to speak",!1),await loadMessages(currentThreadId))}else{const L=get(
"set-mic-transcribe-mode");if(!!(L&&L.value==="llm")&&!supportsAudioInputModel()){showToast("\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u306F\
LLM\u97F3\u58F0\u6587\u5B57\u8D77\u3053\u3057\uFF08\u97F3\u58F0\u5165\u529B\uFF09\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}x.append("llm_model",get("model-select")&&get("model-select").value||"");const N=await(await apiFetch(
CHAT_CONFIG.urls.transcribe,{method:"POST",body:x})).json();if(N.transcript){const B=get("prompt-inp\
ut");B.value+=(B.value?" ":"")+N.transcript,B.style.height="auto",B.style.height=B.scrollHeight+"px"}else
showToast(N.error||"Transcription failed","error",!0)}}catch(L){showToast("Audio processing error: "+
L.message,"error",!0)}finally{get("file-preview").classList.add("hidden"),d.getTracks().forEach(L=>L.
stop()),kn(),Ot(),_||Pt("","hidden"),_&&setStsStatus("Tap to speak",!1)}},it.start(),get("mic-btn").
classList.remove("bg-gray-700"),get("mic-btn").classList.add("bg-red-600","animate-pulse"),isStsModel()||
(Pt("\u9332\u97F3\u4E2D\u2026","recording"),ti(d)),ni(d),isStsModel()&&setStsStatus("Recording... Ta\
p to stop",!0)}catch{Ot(),isStsModel()||Pt("","hidden"),alert("Microphone access denied or not avail\
able.")}};const ln=o((d,u)=>{if(!d)return;const g=d.querySelector("span");g?g.textContent=u:d.textContent=
u},"setLibBtnLabel");window.updateLibSelectionUi=function(){lib.selected||(lib.selected=new Set);const d=lib.
selected.size,u=get("lib-del-btn"),g=get("lib-download-btn"),h=get("lib-attach-btn"),x=get("lib-rena\
me-btn"),_=get("lib-usage-btn");if(u&&(u.disabled=d===0,ln(u,d?`\u524A\u9664 (${d})`:"\u524A\u9664")),
g&&(g.disabled=d===0,ln(g,d?`\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9 (${d})`:"\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9")),
h&&(h.disabled=d===0,ln(h,d?`\u6DFB\u4ED8 (${d})`:"\u6DFB\u4ED8")),x&&(x.disabled=d!==1,ln(x,"\u540D\u524D\u5909\u66F4")),
_&&(_.disabled=d!==1,ln(_,"\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8")),lib.modal){const L=window.matchMedia(
"(max-width: 768px)").matches;lib.modal.classList.toggle("lib-selecting",L&&d>0)}};function ri(d){lib.
attachMode=!!d}o(ri,"setLibAttachMode");const li=o((d=!1)=>{ri(d),showModal("lib-modal"),loadLibraryFiles(),
location.pathname!=="/library"&&history.pushState({modal:"library"},"","/library")},"openLibModal");
if(window.closeLibModal=(d=!1)=>{hideModal("lib-modal"),!d&&location.pathname==="/library"&&history.
back()},get("lib-btn").onclick=()=>li(!1),get("lib-del-btn").onclick=deleteSelectedFiles,get("lib-do\
wnload-btn")&&(get("lib-download-btn").onclick=()=>downloadSelectedLibraryFiles()),get("lib-attach-b\
tn")&&(get("lib-attach-btn").onclick=()=>attachSelectedLibraryFiles()),get("lib-rename-btn")&&(get("\
lib-rename-btn").onclick=()=>renameSelectedLibraryFile()),get("lib-usage-btn")&&(get("lib-usage-btn").
onclick=()=>showSelectedFileUsage()),get("upload-lib-btn")&&(get("upload-lib-btn").onclick=()=>li(!0)),
get("lib-search")){let d=null;get("lib-search").oninput=()=>{lib.searchQuery=(get("lib-search").value||
"").trim(),d&&clearTimeout(d),d=setTimeout(()=>loadLibraryFiles(),250)}}if(get("lib-sort")){const d=localStorage.
getItem(LIB_SORT_KEY)||"newest";get("lib-sort").value=d,get("lib-sort").onchange=()=>{const u=get("l\
ib-sort").value||"newest";localStorage.setItem(LIB_SORT_KEY,u),loadLibraryFiles()}}get("lib-favorite\
-filter-btn")&&(lib.favoritesOnly=localStorage.getItem(LIB_FAVORITES_ONLY_KEY)==="true",get("lib-fav\
orite-filter-btn").onclick=()=>{lib.favoritesOnly=!lib.favoritesOnly,localStorage.setItem(LIB_FAVORITES_ONLY_KEY,
String(lib.favoritesOnly)),loadLibraryFiles()}),get("lib-load-more-btn")&&(get("lib-load-more-btn").
onclick=()=>loadLibraryFiles(!0)),get("add-gem-fixed-prompt-row")&&(get("add-gem-fixed-prompt-row").
onclick=()=>addGemFixedPromptRow());const Ti=o(()=>{editingGemUuid=null,get("gem-modal-title").innerHTML=
'<i class="fas fa-gem text-blue-500 mr-2"></i>Create New Gem',get("save-gem-btn").innerText="Create \
Gem",showModal("gem-modal"),get("gem-name").value="",get("gem-desc").value="",get("gem-inst").value=
"",setGemDefaultModelSelect(""),get("gem-fixed-prompts-container")&&(get("gem-fixed-prompts-containe\
r").innerHTML=""),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")},"openGemMo\
dal");window.closeGemModal=(d=!1)=>{hideModal("gem-modal"),!d&&location.pathname==="/gem"&&history.back()},
get("add-gem-btn").onclick=()=>Ti(),get("save-gem-btn").onclick=async()=>{const d=get("gem-name").value,
u=get("gem-desc").value,g=get("gem-inst").value,h=collectGemFixedPrompts();if(d&&g){const x=editingGemUuid?
"PUT":"POST",_=editingGemUuid?`/api/gems/${editingGemUuid}`:CHAT_CONFIG.urls.handleGems;await apiFetch(
_,{method:x,headers:{"Content-Type":"application/json"},body:JSON.stringify({name:d,description:u,instruction:g,
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
try{const x=decodeURIComponent(g),_=new Blob([x],{type:"text/plain"}),L=URL.createObjectURL(_),A=document.
createElement("a");A.href=L;let N={python:"py",javascript:"js",typescript:"ts",markdown:"md",html:"h\
tml",css:"css",json:"json",xml:"xml",sql:"sql",bash:"sh",sh:"sh",shell:"sh",zsh:"sh",c:"c",cpp:"cpp",
csharp:"cs",cs:"cs",java:"java",kotlin:"kt",swift:"swift",go:"go",rust:"rs",ruby:"rb",php:"php",perl:"\
pl",lua:"lua",r:"r",matlab:"m",yaml:"yaml",yml:"yaml",toml:"toml",ini:"ini",plaintext:"txt",text:"tx\
t"}[h]||h;(h.length>8||/[^a-z0-9]/.test(h))&&(N="txt");let B=`code.${N}`;h==="dockerfile"&&(B="Docke\
rfile"),h==="makefile"&&(B="Makefile"),A.download=B,document.body.appendChild(A),A.click(),document.
body.removeChild(A),URL.revokeObjectURL(L)}catch(x){console.error("Download failed",x)}}if(d.target.
closest(".coding-target-btn")&&selectCodingTargetFromButton(d.target.closest(".coding-target-btn")),
d.target.closest(".copy-btn")){const u=d.target.closest(".copy-btn"),g=u.getAttribute("data-code");g&&
window.copyCode(u,g)}if(d.target.closest(".html-preview-btn")){const g=d.target.closest(".html-previ\
ew-btn").getAttribute("data-code");g&&openHtmlCodePreview(g)}if(d.target.closest(".canvas-preview-bt\
n")){const u=d.target.closest(".canvas-preview-btn");previewCanvasCodeFromButton(u)}}),document.querySelectorAll(
".modal-overlay").forEach(d=>{d.addEventListener("click",u=>{u.target===d&&Xn(d.id)})}),currentThreadId?
loadMessages(currentThreadId):schedulePromptTokenEstimate(!0)});
