function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),r=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const L=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",L.forEach(($,F)=>{
const Y=$.getAttribute("data-local-url"),X=$.getAttribute("data-filename"),Ae=$.querySelector("img.u\
pload-preview")!==null;let R;if(Ae){let V=Y;if(!V&&X){const ee=X.replace(/^\d+\//,"");V=buildAttachmentPreviewUrl(
ee)}V&&(R=document.createElement("img"),R.src=V,R.className="thumb-item shadow-sm",R.dataset.viewerSrc=
V,R.dataset.viewerFilename=X||V.split("/").pop(),R.onclick=function(ee){ee.preventDefault(),openImageViewer(
this.dataset.viewerSrc,".thumb-item")},R.onerror=function(){this.parentElement.replaceChild(u("ERR"),
this)})}R||(R=u("FILE")),R.style.animationDelay=`${F*32}ms`,a.appendChild(R)}),L.length>0?a.classList.
remove("hidden"):a.classList.add("hidden")}function u(L){const $=document.createElement("div");return $.
className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-s\
m font-bold",$.innerText=L,$}o(u,"createFileThumb");const f=collectImageUrlsForSend(),b=uploadProgressState.
total,y=uploadProgressState.completed,w=uploadProgressState.active;b===0&&(e.classList.add("hidden"),
n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const v=get(
"send-btn"),k=get("mic-btn"),_=get("mask-btn"),C=isStopMode;if(w>0?(v&&(v.disabled=!0),k&&(k.disabled=
!0),_&&(_.disabled=!0)):C||(v&&(v.disabled=!1),k&&(k.disabled=!1),_&&(_.disabled=!1)),w>0){const L=`\
Preparing... (${y}/${b})`;e.classList.remove("hidden"),t.innerText=L,r&&(r.innerText=`(${y}/${b})`);
let $=y*100,F=0;for(let Ae in uploadProgressState.perFilePct)$+=uploadProgressState.perFilePct[Ae],F++;
const Y=b>0?$/(b*100)*100:0,X=`${Math.min(100,Y)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=
X),l&&c&&(l.classList.remove("hidden"),c.style.width=X)}else r&&(r.innerText=""),l&&l.classList.add(
"hidden"),f.length>0?(e.classList.remove("hidden"),t.innerText=`${f.length} files ready`,n&&n.classList.
add("hidden")):(e.classList.add("hidden"),t.innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}
o(updateFilePreview,"updateFilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get(
"mask-name");!e||!t||(currentMaskImage?(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.
split("/").pop()}`):(e.classList.add("hidden"),t.innerText=""))}o(updateMaskPreview,"updateMaskPrevi\
ew");const markerToolHints={draw:"\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],a=t[3];return`#${n}${n}${i}${i}${a}${a}`}
return"#facc15"}o(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const t=normalizeMarkerHexColor(
e);return{r:parseInt(t.slice(1,3),16),g:parseInt(t.slice(3,5),16),b:parseInt(t.slice(5,7),16)}}o(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,t=60){const n=Number(e),i=Number.isFinite(n)?n:t;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,i))}o(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const t=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(t)?String(t):String(t).replace(/\.0$/,"")}o(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${t})`}o(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),a=get("ma\
rker-color-picker");a&&a.value!==e&&(a.value=e);const r=get("marker-opacity");r&&r.value!==i&&(r.value=
i);const l=get("marker-opacity-number");l&&l.value!==i&&(l.value=i);const c=get("marker-opacity-valu\
e");c&&(c.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(f=>{const b=normalizeMarkerHexColor(f.getAttribute("data-marker-color"));f.classList.
toggle("active",b===e)})}o(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}o(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}o(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const a=get("marker-tool-hi\
nt");a&&(a.textContent=markerToolHints[e]||"");const r=get("marker-crop-reset");r&&r.classList.toggle(
"hidden",e!=="crop");const l=get("marker-canvas");l&&(l.style.pointerEvents=e==="crop"?"none":"auto");
const c=get("marker-crop-canvas");c&&(c.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.
cropRect||markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}
o(setMarkerMode,"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}
o(clearCropRect,"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)
return;const t=Math.max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,
y:0,w:t,h:n})}o(resetCropRectToFull,"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.
scale<=markerView.minScale+1e-4){markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marke\
r-stage"),t=get("marker-viewport");if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(
1,e.clientHeight||0),a=Math.max(1,t.offsetWidth||t.clientWidth||0),r=Math.max(1,t.offsetHeight||t.clientHeight||
0);if(n<=1||i<=1||a<=1||r<=1)return;const l=(n-a)/2,c=(i-r)/2,u=a*markerView.scale,f=r*markerView.scale,
b=Math.min(n*.45,Math.max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),w=b-l-u,v=n-b-l,k=y-c-f,_=i-
y-c,C=o((L,$,F)=>Number.isFinite(L)?$>F?($+F)/2:Math.min(F,Math.max($,L)):0,"clampOffset");markerView.
offsetX=C(markerView.offsetX,w,v),markerView.offsetY=C(markerView.offsetY,k,_)}o(clampMarkerViewOffset,
"clampMarkerViewOffset");function applyMarkerTransform(){const e=get("marker-viewport");e&&(clampMarkerViewOffset(),
e.style.transform=`translate(${markerView.offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale}\
)`)}o(applyMarkerTransform,"applyMarkerTransform");function resetMarkerTransform(){markerView.scale=
1,markerView.offsetX=0,markerView.offsetY=0,applyMarkerTransform()}o(resetMarkerTransform,"resetMark\
erTransform");function getRowMarkerKey(e){return e&&(e.dataset.uploadId||e.getAttribute("data-filena\
me"))||null}o(getRowMarkerKey,"getRowMarkerKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(
e);n&&(t?markerAppliedUploads.add(n):markerAppliedUploads.delete(n));const i=e?e.querySelector(".upl\
oad-marker-tag"):null;i&&i.classList.toggle("hidden",!t)}o(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){
return markerAppliedUploads.size>0}o(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){
const t=String(e||"").trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t===
"uploaded"?"upload":"unknown"}o(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let t=String(e).replace(/\u0000/g,"");return t=t.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),t=t.trim(),!t||(t=t.split("/").pop().split("\\").pop().trim(),t=t.replace(/\s{2,}/g,
" "),t=t.replace(/[<>:"/\\|?*]+/g,"_"),!t||t==="."||t==="..")?"":(t.length>180&&(t=t.slice(0,180).trim()),
t)}o(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const t=normalizeAttachmentPath(e);return t?t.split("/").pop()||t:""}o(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,t){const n=normalizeAttachmentPath(
e);if(!n)return;const i=normalizeAttachmentDisplayName(t)||defaultAttachmentDisplayName(n);i&&attachmentNameByPath.
set(n,i)}o(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const t=normalizeAttachmentPath(e);if(!t)return"";const n=normalizeAttachmentDisplayName(attachmentNameByPath.
get(t));return n||defaultAttachmentDisplayName(t)}o(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,t){if(!e)return;const n=normalizeAttachmentDisplayName(t)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=n;const i=e.querySelector(".truncate");
i&&(i.textContent=n);const a=e.getAttribute("data-filename");a&&setAttachmentNameForPath(a,n)}o(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}o(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}o(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
o(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const a=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,a),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}o(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const a=e.querySelector(".truncate"),r=normalizeAttachmentDisplayName(
a?a.textContent:"");return r||getAttachmentNameForPath(t)}o(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,t){const n=normalizeAttachmentPath(e);if(!n)return;const i=normalizeAttachmentSource(
t);i!=="unknown"&&attachmentSourceByPath.set(n,i)}o(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const t=normalizeAttachmentPath(e);return t?normalizeAttachmentSource(
attachmentSourceByPath.get(t)):"unknown"}o(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,t){if(!e)return;const n=normalizeAttachmentSource(t);e.dataset.fileSource=
n;const i=e.getAttribute("data-filename");i&&setAttachmentSourceForPath(i,n)}o(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const t=normalizeAttachmentSource(
e.dataset.fileSource);if(t!=="unknown")return t;const n=e.getAttribute("data-filename");return getAttachmentSourceForPath(
n)}o(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const t=normalizeAttachmentSource(e.dataset.originalSource);if(t!=="unknown")return t;
const n=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(n)}o(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,t,n){const i=document.createElement(
"canvas");i.width=t,i.height=n;const a=i.getContext("2d");a?(a.drawImage(e,0,0,t,n),markerState.baseImageData=
a.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}o(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=o((l,c,u=null,f=!1)=>{if(!l)return;const b=Math.max(0,l.x),y=Math.max(0,l.y),w=Math.
max(1,l.w),v=Math.max(1,l.h);u&&(t.fillStyle=u,t.fillRect(b,y,w,v)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(b+.5,y+.5,Math.max(1,w-1),Math.max(1,v-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),u=Math.max(1,i.w),f=Math.max(1,i.h);t.clearRect(l,c,u,f),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((a,r,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const u=normalizeAttachmentSource(
r),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),b=t.get(c);if(b===void 0){const v=e.
length;t.set(c,v),e.push({path:c,source:u,name:f});return}const y=e[b];if(!y)return;const w=normalizeAttachmentSource(
y.source);(w==="unknown"&&u!=="unknown"||w==="library"&&u==="upload")&&(y.source=u),!normalizeAttachmentDisplayName(
y.name)&&f&&(y.name=f)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const r=a.getAttribute("data-filename");n(r,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const r=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const u=isAudioPath(
c),f=isVideoPath(c);if(u&&!t.audio||f&&!t.video){u&&(n+=1),f&&(i+=1);return}r.push(c)}),r.length!==currentImageUrls.
length&&(currentImageUrls=r)}const a=get("upload-list");if(a&&(a.querySelectorAll("[data-filename]").
forEach(r=>{const l=r.getAttribute("data-filename");l&&!currentImageUrls.includes(l)&&(isAudioPath(l)||
isVideoPath(l))&&(setRowMarkerState(r,!1),r.remove())}),a.children.length===0&&(a.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const r=[];n&&r.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&r.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${r.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}o(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const t=e.getAttribute("data-local-url");if(t)return t;const n=e.getAttribute("data-f\
ilename");return n?buildFileUrl(n):""}o(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const t=normalizeAttachmentPath(e);return t?FILE_BASE_URL+t:""}o(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const t=normalizeAttachmentPath(e);return t?isImagePath(t)?FILE_THUMB_BASE_URL+t:FILE_BASE_URL+t:""}
o(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e){
const t=getRowImageSource(e);if(!t){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e;const n=e?e.querySelector(".truncate"):null;markerState.filename=
n?n.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=[],markerState.naturalWidth=
0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=[],markerState.mosaicPreviewRect=
null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode("draw");const i=get("m\
arker-attach-original");i&&(i.checked=e.dataset.attachOriginal==="1");const a=get("marker-image"),r=get(
"marker-canvas"),l=get("marker-crop-canvas");if(r){const c=r.getContext("2d");c&&c.clearRect(0,0,r.width,
r.height)}if(l){const c=l.getContext("2d");c&&c.clearRect(0,0,l.width,l.height)}resetMarkerTransform(),
showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState({modal:"marker"},"","\
/edit-image"),a&&(a.onload=()=>{if(!get("marker-stage")||!r)return;const u=Math.max(1,Math.floor(a.clientWidth)),
f=Math.max(1,Math.floor(a.clientHeight));r.width=u,r.height=f,r.style.width=`${u}px`,r.style.height=
`${f}px`,r.style.left="0px",r.style.top="0px",l&&(l.width=u,l.height=f,l.style.width=`${u}px`,l.style.
height=`${f}px`,l.style.left="0px",l.style.top="0px"),markerState.naturalWidth=a.naturalWidth||u,markerState.
naturalHeight=a.naturalHeight||f;const b=r.getContext("2d");b&&b.clearRect(0,0,r.width,r.height),prepareMarkerBaseCanvas(
a,u,f),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},a.src=t)}o(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
o(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,u=c?c.getAttribute("data-local-url"):null;u&&URL.revokeObjectURL(
u)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.clear();const e=get(
"file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");t&&(t.innerHTML="",t.
classList.add("hidden")),updateFilePreview(),updateMaskPreview();const n=get("upload-list");n&&(n.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>');
const i=get("file-input");i&&(i.value="");const a=get("photo-input");a&&(a.value="");const r=get("ma\
sk-input");r&&(r.value="")}o(resetUploadState,"resetUploadState");async function uploadMaskFile(e){if(!e)
return;const t=new FormData;t.append("file",e);try{const n=await fetch(CHAT_CONFIG.urls.upload,{method:"\
POST",body:t}),i=await n.json();n.ok&&i.filename?(currentMaskImage=i.filename,updateMaskPreview()):showToast(
i.error||"Mask upload failed","error",!0)}catch{showToast("Mask upload failed","error",!0)}}o(uploadMaskFile,
"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){const n=get("camera-status");n&&(n.textContent=
e||"",n.classList.toggle("text-red-300",!!t),n.classList.toggle("text-gray-400",!t))}o(setCameraCaptureStatus,
"setCameraCaptureStatus");function updateCameraCapturePendingUi(){const e=cameraCapturePendingFiles.
length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,t.textContent=e?`\u6DFB\u4ED8 (${e}\
)`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=e===0||cameraCaptureBusy);const i=get(
"camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.forEach((a,r)=>{const l=document.
createElement("div");l.className="relative rounded overflow-hidden border border-gray-700 bg-black a\
spect-square",l.innerHTML=`
                        <img src="${a}" alt="capture ${r+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${r+1}</div>
                    `,i.appendChild(l)}),i.classList.toggle("hidden",e===0))}o(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const t=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(t)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}o(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(i=>{try{i.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const t=get("camera-capture-btn");t&&(t.disabled=!0);const n=get(
"camera-switch-btn");n&&(n.disabled=!0)}o(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const t=get("camera-video");if(!t)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const n=get(
"camera-switch-btn");n&&(n.disabled=!0);const i=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let a=null;for(const r of i)
try{const l=await navigator.mediaDevices.getUserMedia(r);cameraCaptureStream=l,t.srcObject=l;try{await t.
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],u=c&&c.getSettings?c.getSettings():{},
f=String(u.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
e;const b=get("camera-capture-btn");return b&&(b.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),l}catch(l){a=l}throw a||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
o(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const t=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(t,!0),showToast(t,"error",!0);const n=get("camera-capture-btn");n&&(n.disabled=
!0);const i=get("camera-attach-btn");i&&(i.disabled=!0)}}o(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const t=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!t&&location.pathname==="/camera"&&history.back()}o(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const t=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=t;try{await startCameraCaptureStream(t)}catch(n){const i=n&&
n.message?n.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
i,!0),showToast(i,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}o(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,t=o(a=>String(a).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}o(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const a=get("camera-attach-btn");a&&(a.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const r=t.getContext("2d");if(!r)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");r.drawImage(e,0,0,t.width,t.height);
const l=await new Promise((u,f)=>{t.toBlob(b=>{b?u(b):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),c=new File([l],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(c),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(l)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(r){
const l=r&&r.message?r.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
l,!0),showToast(l,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}o(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),a=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),a&&(a.disabled=!0);const r=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${r.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{await handleFiles(r,{openModal:!1}),showToast(`${r.length}\u679A\u306E\
\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(l){const c=l&&l.message?l.message:
"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";showToast(c,"\
error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.classList.contains(
"hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}o(attachCameraCapturedFiles,
"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay=="function"&&window.
hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.pathname!=="/upload"&&
history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-info");if(e){const n=(get(
"model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v4.1-flash"||n==="deepse\
ek-v4-flash-vision-exp",a=["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4.6v-flashx","glm-4.6v-f\
lash","glm-4.5v"].includes(n),r=n.includes("deepseek")&&!i||n.startsWith("glm-")&&!a;e.classList.toggle(
"hidden",!r)}_syncVisionModelDisplay()}o(openUploadModal,"openUploadModal");function _syncVisionModelDisplay(){
const e=get("vision-model-display");if(!e)return;const t=currentVisionModel;if(t){let n=t;MODELS.forEach(
i=>(i.items||[]).forEach(a=>{a.id===t&&(n=a.name)})),e.textContent=n}else e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}
o(_syncVisionModelDisplay,"_syncVisionModelDisplay");function _openVisionModelSelector(){window._visionPickerActive=
!0,openModelModal(),setTimeout(()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},
50)}o(_openVisionModelSelector,"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.
hideDropOverlay=="function"&&window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname===
"/upload"&&history.back()}o(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){
const e=get("upload-list");if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(
n=>{const i=n.getAttribute("data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(
n,{source:getAttachmentSourceForPath(n),displayName:getAttachmentNameForPath(n)})}),e.children.length===
0&&(e.innerHTML='<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
o(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}o(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const a=e.split("/").pop()||e,r=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||a,l=(a.split(".").pop()||"").toLowerCase(),c=["png","jpg","jpeg","webp",
"gif"].includes(l),u=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):u,b=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=b,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=r,y.dataset.defaultDisplayName=r,y.dataset.sendNameCustomized="";const w=escapeHtml(
r),v=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",k=c?`<img src="${f}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${w}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${k}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${w}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">ready</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${v}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:100%"></div>
                </div>
            `;const _=y.querySelector(".upload-preview");_&&(_.onclick=()=>openFileViewer(u,getRowAttachmentName(
y)||r));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const L=y.
querySelector(".upload-remove");L&&(L.onclick=()=>{uploadCancelTokens.add(b),browserFastLocalFiles.delete(
b),decrementUploadTotal(b);const F=y.getAttribute("data-filename");F&&(currentImageUrls=currentImageUrls.
filter(Y=>Y!==F)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const $=y.querySelector(".upload-marker");return $&&($.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,r),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:b}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const r=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const u=l&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let f="";l?(f=URL.createObjectURL(e),c=`<img src="${f}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${r}">`):(f=URL.createObjectURL(e),c=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${c}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${r}</div>
                        <div class="flex items-center gap-2">
                            <div class="upload-status text-[10px] text-gray-400">\u5F85\u6A5F\u4E2D</div>
                            <span class="upload-marker-tag hidden">\u7DE8\u96C6\u6E08\u307F</span>
                        </div>
                    </div>
                    <div class="flex items-center gap-1">
                        ${u}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `,f&&i.setAttribute("data-local-url",f);const b=i.querySelector(".upload-preview");b&&(b.
onclick=()=>{const k=i.getAttribute("data-filename"),_=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(_,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const _=i.
getAttribute("data-filename");_&&(currentImageUrls=currentImageUrls.filter(C=>C!==_)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const w=i.querySelector(".upload-marker");w&&(w.onclick=()=>openMarkerModalForRow(i));const v=i.querySelector(
".upload-send-name");return v&&(v.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(!e)
return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);try{const i=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.ok){const y=a&&a.error?a.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast(y,"error",!0),!1}const r=a.upload_id,l=a.chunk_size||10*1024*1024,c=Math.ceil(
e.size/l);for(let y=0;y<c;y++){const w=y*l,v=Math.min(e.size,w+l),k=e.slice(w,v);if(!await new Promise(
C=>{const L=new XMLHttpRequest;L.open("POST","/upload/chunk",!0),L.setRequestHeader("X-CSRF-Token",csrfToken),
L.upload.onprogress=F=>{if(F.lengthComputable&&t&&t.bar){const Y=w+F.loaded,X=Math.min(100,Math.floor(
Y/e.size*100));t.bar.style.width=`${X}%`,t.status&&(t.status.textContent=`${X}%`),t.uploadId&&updateGlobalUploadProgress(
t.uploadId,X)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},L.onload=()=>{L.status>=
200&&L.status<300?C(!0):C(!1)},L.onerror=()=>C(!1);const $=new FormData;$.append("upload_id",r),$.append(
"index",String(y)),$.append("total",String(c)),$.append("chunk",k,e.name),L.send($)}))return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const u=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),f=await u.json();if(u.ok&&f&&f.filename){if(t&&t.row&&t.uploadId&&
uploadCancelTokens.has(t.uploadId))return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const v=t.
row.getAttribute("data-local-url");v&&URL.revokeObjectURL(v),t.row.removeAttribute("data-local-url");
const k=t.row.querySelector("img.upload-preview");if(k){const _=f.filename.replace(/^\d+\//,"");k.src=
buildAttachmentPreviewUrl(_)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(
y),t&&t.row&&(t.row.setAttribute("data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),
y)){const v=isRowAttachmentNameCustomized(t.row),k=defaultAttachmentDisplayName(y),_=v&&normalizeAttachmentDisplayName(
t.row.dataset.displayName)||k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,_)}return y&&
setAttachmentSourceForPath(y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(f.filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(v=>addLibraryFileFromPath(
v)),!0}const b=f&&f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(b,"error",!0),!1}catch{return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
i=!0);const a=o(()=>{i&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),i=!1)},
"finishUploadOp"),r=new XMLHttpRequest;r.open("POST",CHAT_CONFIG.urls.upload,!0),r.setRequestHeader(
"X-CSRF-Token",csrfToken),r.upload.onprogress=c=>{if(c.lengthComputable&&t&&t.bar){const u=Math.min(
100,Math.floor(c.loaded/c.total*100));t.bar.style.width=`${u}%`,t.status&&(t.status.textContent=`${u}\
%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,u)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},r.onload=()=>{let c={};try{c=JSON.parse(r.responseText||"{}")}catch{}if(r.status>=
200&&r.status<300&&c&&c.filename){if(t&&t.row&&t.uploadId&&uploadCancelTokens.has(t.uploadId)){t.row&&
t.row.parentNode&&t.row.remove(),a(),n(!1);return}if(t&&t.row){const b=t.row.getAttribute("data-loca\
l-url");b&&URL.revokeObjectURL(b),t.row.removeAttribute("data-local-url");const y=t.row.querySelector(
"img.upload-preview");if(y){const w=c.filename.replace(/^\d+\//,"");y.src=buildAttachmentPreviewUrl(
w)}}const u=normalizeAttachmentPath(c.filename);if(u&&currentImageUrls.push(u),t&&t.row&&(t.row.setAttribute(
"data-filename",u||c.filename),setRowAttachmentSource(t.row,"upload"),u)){const b=isRowAttachmentNameCustomized(
t.row),y=defaultAttachmentDisplayName(u),w=b&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
y;t.row.dataset.defaultDisplayName=y,setRowAttachmentName(t.row,w)}u&&setAttachmentSourceForPath(u,"\
upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(c.filenames)&&
c.filenames.length?c.filenames:[c.filename]).forEach(b=>addLibraryFileFromPath(b)),a(),n(!0)}else{const u=c&&
c.error?c.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(u,"error",!0),a(),n(!1)}},r.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),a(),n(!1)};const l=new FormData;l.append("file",e),r.send(l)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(f=>{n+=f.length});const i=new Float32Array(n);
let a=0;e.forEach(f=>{i.set(f,a),a+=f.length});const r=new ArrayBuffer(44+i.length*2),l=new DataView(
r),c=o((f,b)=>{for(let y=0;y<b.length;y++)l.setUint8(f+y,b.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let u=44;for(let f=0;f<i.length;f++){const b=Math.
max(-1,Math.min(1,i[f]));l.setInt16(u,b<0?b*32768:b*32767,!0),u+=2}return new Blob([l],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const r=e.row.getAttribute("data-local-url");r&&URL.revokeObjectURL(
r);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),u=escapeHtml(a),f=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${u}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',b=e.row.querySelector(".upload-preview");b&&(b.outerHTML=f);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const v=e.row.getAttribute("data-filename"),k=v?buildFileUrl(v):
e.row.getAttribute("data-local-url");openFileViewer(k,getRowAttachmentName(e.row)||a||v||"")});const w=e.
row.querySelector(".upload-marker");w&&w.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}o(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const t=e.getContext("2d");if(!t)return;const n=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[];markerState.history.push({imageData:t.
getImageData(0,0,e.width,e.height),mosaicRects:n}),markerState.history.length>40&&markerState.history.
shift()}o(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const t=e.getContext(
"2d");if(!t)return;const n=markerState.history[markerState.history.length-1];t.clearRect(0,0,e.width,
e.height),n&&n.imageData?(t.putImageData(n.imageData,0,0),markerState.mosaicRects=Array.isArray(n.mosaicRects)?
n.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[]):n?(t.putImageData(n,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}o(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const t=e.getContext("2d");t&&t.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}o(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let a=!1,r=0,
l=markerView.scale,c={x:0,y:0},u={x:0,y:0},f=[],b=16,y="",w=null,v=null,k=null,_=null,C=!1,L=null;const $=o(
A=>{const j=e.getBoundingClientRect(),K=(A.clientX-j.left)*(e.width/j.width),Q=(A.clientY-j.top)*(e.
height/j.height);return{x:K,y:Q}},"getPoint"),F=o((A,j)=>({x:(A.x+j.x)/2,y:(A.y+j.y)/2}),"getMid"),Y=o(
(A,j)=>Math.hypot(A.x-j.x,A.y-j.y),"getDist");let X=!1;const Ae=o(()=>{w||(w=document.createElement(
"canvas"),v=w.getContext("2d")),k||(k=document.createElement("canvas"),_=k.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),R=o(()=>{if(!t||!w||!k)return;const A=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(0,0,e.width,e.
height),t.drawImage(w,0,0),t.save(),t.globalAlpha=A,t.drawImage(k,0,0),t.restore()},"renderDrawPrevi\
ew"),V=o(()=>{_&&(_.strokeStyle=y,_.fillStyle=y,_.lineWidth=b,_.lineCap="round",_.lineJoin="round")},
"applyMarkerBrush"),ee=o(A=>{if(!A)return!1;if(f.length===0)return f.push(A),!0;const j=f[f.length-1],
K=A.x-j.x,Q=A.y-j.y,ne=Math.hypot(K,Q),Te=Math.max(.35,b*.04);if(ne<Te)return!1;const ie=Math.max(1,
b*.25),pe=Math.max(1,Math.ceil(ne/ie));for(let W=1;W<=pe;W++){const he=W/pe;f.push({x:j.x+K*he,y:j.y+
Q*he})}return!0},"appendStrokePoint"),we=o(()=>{if(_&&(_.clearRect(0,0,k.width,k.height),f.length!==
0)){if(V(),f.length===1){const A=f[0];_.beginPath(),_.arc(A.x,A.y,b/2,0,Math.PI*2),_.fill();return}if(_.
beginPath(),_.moveTo(f[0].x,f[0].y),f.length===2)_.lineTo(f[1].x,f[1].y);else{for(let K=1;K<f.length-
2;K++){const Q=f[K],ne=f[K+1],Te=F(Q,ne);_.quadraticCurveTo(Q.x,Q.y,Te.x,Te.y)}const A=f[f.length-2],
j=f[f.length-1];_.quadraticCurveTo(A.x,A.y,j.x,j.y)}_.stroke()}},"renderStrokeLayer"),ce=o((A,j)=>{if(!A||
!j)return null;const K=Math.min(A.x,j.x),Q=Math.min(A.y,j.y),ne=Math.abs(A.x-j.x),Te=Math.abs(A.y-j.
y);return{x:K,y:Q,w:ne,h:Te}},"normalizeMosaicRect"),Se=o(A=>{const j=n?Number(n.value||16):16,K=Math.
max(6,Math.floor(j)),Q=Math.floor(K/2);return{x:A.x-Q,y:A.y-Q,w:K,h:K}},"buildMosaicRectFromPoint"),
Pe=o(()=>{const A=document.createElement("canvas");A.width=e.width,A.height=e.height;const j=A.getContext(
"2d");if(!j)return null;markerState.baseCanvas&&j.drawImage(markerState.baseCanvas,0,0),j.drawImage(
e,0,0);try{return j.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),
Ce=o(A=>{if(!t||!A)return!1;const j=Pe();if(!j)return!1;const K=n?Number(n.value||16):16,Q=Math.max(
4,Math.floor(K/2)),ne=Math.max(0,Math.floor(A.x)),Te=Math.max(0,Math.floor(A.y)),ie=Math.min(e.width,
Math.ceil(A.x+A.w)),pe=Math.min(e.height,Math.ceil(A.y+A.h));if(ie<=ne||pe<=Te)return!1;for(let W=Te;W<
pe;W+=Q)for(let he=ne;he<ie;he+=Q){const pt=Math.min(Q,ie-he),ct=Math.min(Q,pe-W),mt=Math.min(e.width-
1,Math.max(0,he+Math.floor(pt/2))),Ve=(Math.min(e.height-1,Math.max(0,W+Math.floor(ct/2)))*e.width+mt)*
4,bt=j.data[Ve],Lt=j.data[Ve+1],wt=j.data[Ve+2];t.fillStyle=`rgb(${bt},${Lt},${wt})`,t.fillRect(he,W,
pt,ct)}return!0},"applyMosaicRect"),ge=o(A=>{if(!t)return;if(i.set(A.pointerId,{x:A.clientX,y:A.clientY}),
i.size>=2){const K=Array.from(i.values()),Q=K[0],ne=K[1];a=!0,X=!1,f=[],C=!1,L=null,markerState.mosaicPreviewRect=
null,r=Y(Q,ne)||1,l=markerView.scale,c={x:markerView.offsetX,y:markerView.offsetY},u=F(Q,ne),renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(A.pointerId),A.preventDefault();return}if(a||markerState.mode===
"crop")return;X=!0;const j=$(A);if(markerState.mode==="mosaic")C=!0,L=j,markerState.mosaicPreviewRect=
Se(j),renderCropOverlay();else{if(Ae(),!v||!_)return;v.clearRect(0,0,w.width,w.height),v.drawImage(e,
0,0),_.clearRect(0,0,k.width,k.height),b=n?Number(n.value||16):16,y=normalizeMarkerHexColor(markerState.
colorHex),f=[],ee(j),we(),markerState.hasStroke=!0,R()}e.setPointerCapture&&e.setPointerCapture(A.pointerId),
A.preventDefault()},"start"),de=o(A=>{if(i.has(A.pointerId)&&i.set(A.pointerId,{x:A.clientX,y:A.clientY}),
a&&i.size>=2){const K=Array.from(i.values()),Q=K[0],ne=K[1],Te=F(Q,ne),ie=Y(Q,ne)||1,pe=l*(ie/r);markerView.
scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,pe)),markerView.offsetX=c.x+(Te.x-u.
x),markerView.offsetY=c.y+(Te.y-u.y),applyMarkerTransform(),A.preventDefault();return}if(!X||!t)return;
const j=$(A);if(markerState.mode==="mosaic"){if(!C||!L)return;markerState.mosaicPreviewRect=ce(L,j)||
Se(j),renderCropOverlay()}else ee(j)&&(we(),R());A.preventDefault()},"move"),H=o(A=>{const j=X;if(i.
delete(A.pointerId),i.size<2&&(a=!1),i.size===0){if(X=!1,j&&t&&markerState.mode==="draw"&&f.length>0&&
(we(),R()),j&&markerState.mode==="mosaic"&&L){const K=$(A);let Q=ce(L,K);(!Q||Q.w<2||Q.h<2)&&(Q=Se(L)),
Ce(Q)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(Q))}f=[],C=!1,L=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),j&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(A.pointerId),
A.preventDefault()},"end");e.addEventListener("pointerdown",ge),e.addEventListener("pointermove",de),
e.addEventListener("pointerup",H),e.addEventListener("pointercancel",H)}o(initMarkerCanvas,"initMark\
erCanvas");function initCropCanvas(){const e=get("marker-crop-canvas");if(!e)return;const t=e.getContext(
"2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,u=0,f=markerView.scale,b={x:0,y:0},y={x:0,y:0};const w=8,
v=14,k=o((R,V,ee)=>Math.min(ee,Math.max(V,R)),"clamp"),_=o(R=>{const V=e.getBoundingClientRect(),ee=(R.
clientX-V.left)*(e.width/V.width),we=(R.clientY-V.top)*(e.height/V.height);return{x:ee,y:we}},"getPo\
int"),C=o((R,V)=>({x:(R.x+V.x)/2,y:(R.y+V.y)/2}),"getMid"),L=o((R,V)=>Math.hypot(R.x-V.x,R.y-V.y),"g\
etDist"),$=o(()=>(markerState.cropRect||resetCropRectToFull(),markerState.cropRect),"ensureCropRect"),
F=o((R,V)=>{if(!V)return"move";const ee=V.x,we=V.y,ce=V.x+V.w,Se=V.y+V.h,Pe=Math.abs(R.x-ee)<=v,Ce=Math.
abs(R.x-ce)<=v,ge=Math.abs(R.y-we)<=v,de=Math.abs(R.y-Se)<=v;if(Pe&&ge)return"nw";if(Ce&&ge)return"n\
e";if(Pe&&de)return"sw";if(Ce&&de)return"se";if(ge)return"n";if(de)return"s";if(Pe)return"w";if(Ce)return"\
e";if(R.x>ee+v&&R.x<ce-v&&R.y>we+v&&R.y<Se-v)return"move";const A=R.x<ee?"left":R.x>ce?"right":null,
j=R.y<we?"top":R.y>Se?"bottom":null;if(A&&j){if(A==="left"&&j==="top")return"nw";if(A==="right"&&j===
"top")return"ne";if(A==="left"&&j==="bottom")return"sw";if(A==="right"&&j==="bottom")return"se"}return A?
A==="left"?"w":"e":j?j==="top"?"n":"s":"move"},"hitTest"),Y=o(R=>{if(markerState.mode!=="crop")return;
if(n.set(R.pointerId,{x:R.clientX,y:R.clientY}),n.size>=2){const we=Array.from(n.values()),ce=we[0],
Se=we[1];c=!0,i=!1,u=L(ce,Se)||1,f=markerView.scale,b={x:markerView.offsetX,y:markerView.offsetY},y=
C(ce,Se),e.setPointerCapture&&e.setPointerCapture(R.pointerId),R.preventDefault();return}if(c)return;
i=!0;const V=_(R),ee=$();r=F(V,ee),a=V,l=ee?{x:ee.x,y:ee.y,w:ee.w,h:ee.h}:null,renderCropOverlay(),e.
setPointerCapture&&e.setPointerCapture(R.pointerId),R.preventDefault()},"start"),X=o(R=>{if(markerState.
mode!=="crop")return;if(n.has(R.pointerId)&&n.set(R.pointerId,{x:R.clientX,y:R.clientY}),c&&n.size>=
2){const A=Array.from(n.values()),j=A[0],K=A[1],Q=C(j,K),ne=L(j,K)||1,Te=f*(ne/u);markerView.scale=Math.
min(markerView.maxScale,Math.max(markerView.minScale,Te)),markerView.offsetX=b.x+(Q.x-y.x),markerView.
offsetY=b.y+(Q.y-y.y),applyMarkerTransform(),renderCropOverlay(),R.preventDefault();return}if(!i||!a||
!l)return;const V=_(R),ee=e.width,we=e.height,ce={x:l.x,y:l.y,w:l.w,h:l.h},Se=l.x+l.w,Pe=l.y+l.h,Ce=o(
()=>{const A=k(V.x,0,Se-w);ce.x=A,ce.w=Se-A},"applyW"),ge=o(()=>{ce.w=k(V.x-l.x,w,ee-l.x)},"applyE"),
de=o(()=>{const A=k(V.y,0,Pe-w);ce.y=A,ce.h=Pe-A},"applyN"),H=o(()=>{ce.h=k(V.y-l.y,w,we-l.y)},"appl\
yS");switch(r){case"move":{const A=V.x-a.x,j=V.y-a.y;ce.x=k(l.x+A,0,ee-l.w),ce.y=k(l.y+j,0,we-l.h);break}case"\
w":Ce();break;case"e":ge();break;case"n":de();break;case"s":H();break;case"nw":de(),Ce();break;case"\
ne":de(),ge();break;case"sw":H(),Ce();break;case"se":H(),ge();break;default:break}ce.x=k(ce.x,0,ee-ce.
w),ce.y=k(ce.y,0,we-ce.h),markerState.cropRect=ce,renderCropOverlay(),R.preventDefault()},"move"),Ae=o(
R=>{n.delete(R.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),
e.releasePointerCapture&&e.releasePointerCapture(R.pointerId),R.preventDefault()},"end");e.addEventListener(
"pointerdown",Y),e.addEventListener("pointermove",X),e.addEventListener("pointerup",Ae),e.addEventListener(
"pointercancel",Ae),e.addEventListener("pointerleave",Ae)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const r=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;a.width=r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,0,r,l),c.drawImage(
n,0,0,r,l),markerState.cropRect){const C=r/n.width,L=l/n.height,$=Math.max(0,Math.floor(markerState.
cropRect.x*C)),F=Math.max(0,Math.floor(markerState.cropRect.y*L)),Y=Math.min(r,Math.max(1,Math.floor(
markerState.cropRect.w*C))),X=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.h*L))),Ae=document.
createElement("canvas");Ae.width=Y,Ae.height=X;const R=Ae.getContext("2d");R&&(R.drawImage(a,$,F,Y,X,
0,0,Y,X),a=Ae)}const u=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!u){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F","error",!0);return}const b=(markerState.filename||"marked.png").replace(
/\.[^/.]+$/,""),y=new File([u],`${b}_marked.png`,{type:"image/png"}),w={row:e,uploadId:e.dataset.uploadId,
status:e.querySelector(".upload-status"),bar:e.querySelector(".upload-progress > div")};w.status&&(w.
status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),updateUploadRowFile(w,y);const v=e.getAttribute(
"data-filename"),k=getRowAttachmentSource(e);v&&!e.dataset.originalFilename&&(e.dataset.originalFilename=
v,e.dataset.originalSource=k,setAttachmentSourceForPath(v,k)),await uploadFileWithProgress(y,w)?(v&&
(currentImageUrls=currentImageUrls.filter(C=>C!==v)),setRowAttachmentSource(e,"upload"),setRowMarkerState(
e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}o(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,i.src=URL.createObjectURL(
e);let a=null,r=null,l=null,c=null,u=[],f=null;const b=o(()=>{f&&clearTimeout(f);try{URL.revokeObjectURL(
i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(w=>w.stop()),l)try{l.disconnect()}catch{}
if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(()=>{b(),n(null)},"fail");
i.onloadedmetadata=async()=>{try{a=i.captureStream();const w=a.getAudioTracks();if(!w||!w.length)return y();
r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),c=r.createMediaStreamSource(
new MediaStream(w)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=k=>{const _=k.inputBuffer.getChannelData(
0);u.push(new Float32Array(_))},c.connect(l),l.connect(r.destination);const v=isFinite(i.duration)?Math.
max(1,Math.ceil(i.duration*1e3)):0;v>0&&(f=setTimeout(()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,
""),_=encodeWav(u,r.sampleRate),C=new File([_],`${k}.audio.wav`,{type:"audio/wav"});b(),n(C)},v+250)),
await i.play(),i.onended=()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,""),_=encodeWav(u,r.sampleRate),
C=new File([_],`${k}.audio.wav`,{type:"audio/wav"});b(),n(C)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,t={}){if(!e||!e.length)return;const n=Array.from(
e).filter(Boolean);if(!n.length)return;const i=collectImageUrlsForSend().length+browserFastLocalFiles.
size+Math.max(0,Number(uploadProgressState.active)||0);let a=n;if(i+n.length>ATTACHMENT_MAX_FILES){const y=Math.
max(0,ATTACHMENT_MAX_FILES-i);if(y<=0){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059`,"error",!0);return}a=n.slice(0,y),showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u5148\u982D${y}\u4EF6\u306E\u307F\u8FFD\u52A0\u3057\u307E\u3059\u3002`,"war\
ning",!0)}t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=
a.length,uploadProgressState.active+=a.length,updateFilePreview();const r=!!(get("upload-audio-only")&&
get("upload-audio-only").checked),l=getModelMediaSupport(get("model-select").value),c=o(async y=>{let w=null;
try{if(isAudioFile(y)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(y)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!y.type||!y.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const v=addUploadRow(y);updateFilePreview(),w=v.uploadId,uploadProgressState.perFilePct[w]=
0;let k=y;if(r&&isVideoFile(y)){const _=await extractAudioFromVideo(y,v);_?(k=_,updateUploadRowFile(
v,_),v&&v.status&&(v.status.textContent="\u97F3\u58F0\u306E\u307F")):(v&&v.status&&(v.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&y.type.startsWith("image/"))try{const _=getCompressionOutputType();
if(getCompressionFormatOnly())k=await convertImageFormatOnly(y,_);else{const L={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};_&&_!=="original"&&(L.fileType=_),await ensureImageCompression();
const $=await window.imageCompression(y,L),F=new File([$],imageFilenameForMime(y.name,$.type||(_!=="\
original"?_:y.type)),{type:$.type||y.type,lastModified:y.lastModified||Date.now()});F.size>y.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
y.size)} -> ${formatBytes(F.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),k=y):k=F}k!==y&&updateUploadRowFile(v,k)}catch{}if(browserFastModeEnabled){const _=Array.from(
browserFastLocalFiles.values()).reduce((C,L)=>C+Number(L.file&&L.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||_+k.size>BROWSER_FAST_MAX_BYTES?(v&&v.status&&(v.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),v&&v.row&&v.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(v.uploadId,{file:k,rowObj:v}),v.status&&(v.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),v.bar&&(v.bar.style.width="100\
%"),v.row&&(v.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(k,v)}finally{
w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.perFilePct[w],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let u=0;const f=Math.min(UPLOAD_CONCURRENCY,a.length),b=Array.from({length:f}).map(async()=>{
for(;;){const y=u++;if(y>=a.length)break;await c(a[y])}});await Promise.all(b)}o(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){const n=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).replace(/\n/g,
"<br>");const i=protectMathSegments(n),a=window.marked.parse(i.text),r=restoreMathSegments(a,i.blocks,
t);return window.DOMPurify.sanitize(r)}o(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");function getCanvasModeElements(){
const e=get("canvas-panel");return e?{panel:e,stage:get("conversation-stage"),title:get("canvas-pane\
l-title"),status:get("canvas-panel-status"),blockCount:get("canvas-block-count"),blockList:get("canv\
as-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get("canvas-preview-lang"),sourceSelect:get(
"canvas-source-select"),frame:get("canvas-preview-frame"),empty:get("canvas-preview-empty"),sourceScroll:get(
"canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get("canvas-panel-copy-btn"),clearBtn:get(
"canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:null}o(getCanvasModeElements,"getC\
anvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(e||"").trim().toLowerCase();
if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(t||"");return/<!doctype\s+html/i.
test(i)||/<html[\s>]/i.test(i)}o(isCanvasHtmlPreviewCandidate,"isCanvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){
const n=String(e&&e.lang?e.lang:"").trim(),i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),a=!!(e&&
e.open);return{...e,index:t,lang:n,code:i,open:a,key:hashString(`${n||"TEXT"}
${i||""}`)}}o(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],a=[],r=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",u=[];for(const y of n){
if(!l){const k=y.match(r);if(k){l=k[2],c=String(k[3]||"").trim(),u=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const w=String(y||"").trim();if(w&&w.replace(/\s+/g,"")===l){const k=i[i.length-1];k&&(k.
code=u.join(`
`),k.open=!1),l=null,c="",u=[];continue}u.push(y);const v=i[i.length-1];v&&(v.code=u.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=u.join(`
`),y.open=!0)}const f=i.map((y,w)=>normalizeCanvasBlock(y,w)),b=selectCanvasPreviewBlock(f,t);return{
renderText:a.join(`
`),blocks:f,primaryBlock:b?b.block:null,primaryIndex:b?b.index:-1,rawText:t}}o(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const r=i[n];return{block:r,index:n,previewType:isCanvasHtmlPreviewCandidate(
r.lang,r.code)?"html":"code"}}if(i.length>0){const r=i.length-1,l=i[r];return{block:l,index:r,previewType:isCanvasHtmlPreviewCandidate(
l.lang,l.code)?"html":"code"}}const a=String(t||"");return isCanvasHtmlPreviewCandidate("",a)?{block:normalizeCanvasBlock(
{lang:"html",code:a,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}o(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}o(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),a=!!t&&t===i;n.classList.toggle("can\
vas-active",a),n.setAttribute("aria-pressed",a?"true":"false"),n.setAttribute("data-canvas-active",a?
"1":"0"),n.innerHTML=a?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
o(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}o(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
a={preview:0,blocks:1,source:2},r=i[n];if(!r||!(t in a)||!(n in a))return;canvasPreviewState.viewAnimationToken+=
1;const l=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(u=>{u&&u.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),r.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";r.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(r.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}o(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(u=>{const f=u.getAttribute("da\
ta-canvas-panel-view")===i;u.classList.toggle("active",f),u.setAttribute("aria-pressed",f?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}o(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const r=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",b=(String(i&&i.code?i.code:"").split(/\r?\n/).find(v=>v.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${r}`,w=`${y}\u3001${b}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(w)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
r)}</span><span class="canvas-block-chip-preview">${escapeHtml(b)}</span></span><span class="canvas-\
block-chip-state">${l?"\u8868\u793A\u4E2D":c}</span></button>`}).join("")}o(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((r,l)=>{const c=String(
r&&r.lang?r.lang:"text").trim()||"text";return`#${l+1} ${c}`}),a=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==a&&(e.sourceSelect.innerHTML=i.map((r,l)=>`<option value="${l}">${escapeHtml(
r)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=a),e.sourceSelect.value=String(
n)}o(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}o(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),a=String(e||""),r=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const l=new DOMParser().parseFromString(a,"text/html"),c=l.createElement("scr\
ipt");return c.setAttribute("data-canvas-scroll-bridge","true"),c.textContent=r,(l.body||l.documentElement).
appendChild(c),`<!DOCTYPE html>
`+l.documentElement.outerHTML}catch{return`${a}<script data-canvas-scroll-bridge>${r}<\/script>`}}o(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const t=e&&e.data?e.data:null;if(!t||t.type!=="canvas-preview-scroll")return;const n=getCanvasModeElements();
!n||!n.frame||e.source!==n.frame.contentWindow||t.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(t.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(t.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const t=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{t===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}o(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const t=getCanvasModeElements();if(t){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){t.panel.classList.add("hidden"),t.panel.classList.remove(
"canvas-panel-open","canvas-closing"),t.stage&&t.stage.classList.remove("canvas-enabled");return}t.panel.
classList.remove("canvas-panel-open"),t.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{t.panel.classList.add("hidden"),t.panel.classList.remove("canvas-closing"),t.
stage&&t.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}o(hideCanvasPreviewPanel,
"hideCanvasPreviewPanel");function resetCanvasPreviewPanel(e="Canvas\u3067\u8868\u793A\u4E2D"){const t=getCanvasModeElements();
t&&(canvasPreviewState.blocks=[],canvasPreviewState.rawText="",canvasPreviewState.renderText="",canvasPreviewState.
selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.selectionMode="auto",canvasPreviewState.
mobileView="preview",canvasPreviewState.lastCanvasData=null,resetCanvasScrollState(),showCanvasPreviewPanel(),
syncCanvasPanelViewUi("preview",{focus:!1}),t.title&&(t.title.textContent=e),t.status&&(t.status.textContent=
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D"),t.previewLang&&(t.previewLang.
textContent="idle"),t.sourceSelect&&(t.sourceSelect.innerHTML='<option value="">-</option>',t.sourceSelect.
disabled=!0,t.sourceSelect.dataset.canvasOptionsSignature=""),t.code&&(t.code.textContent=""),t.blockCount&&
(t.blockCount.textContent="0"),t.blockList&&(t.blockList.innerHTML='<div class="px-2 py-3 text-xs te\
xt-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>'),t.sourceScroll&&
(t.sourceScroll.scrollTop=0),t.frame&&(t.frame.removeAttribute("srcdoc"),t.frame.classList.add("hidd\
en")),t.empty&&t.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}o(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const t=e||canvasPreviewState.lastCanvasData;
if(!t)return null;canvasPreviewState.lastCanvasData=t,canvasPreviewState.blocks=Array.isArray(t.blocks)?
t.blocks.slice():[],canvasPreviewState.rawText=String(t.rawText||""),canvasPreviewState.renderText=String(
t.renderText||"");const n=canvasPreviewState.blocks,i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!n.length){const l=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return l&&l.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=l.block.
key||"",l.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let a=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?a=i:canvasPreviewState.selectionMode="auto";const r=n[a]||null;return canvasPreviewState.
selectedIndex=r?a:-1,canvasPreviewState.selectedKey=r&&r.key?r.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),r}o(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,a=n&&Number.isInteger(
n.index)?n.index:-1,r=!!i,l=String(i&&i.lang?i.lang:"").trim(),c=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),u=r?isCanvasHtmlPreviewCandidate(l,c):!1,f=r?u?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",b=r?u?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=b),e.status&&(e.status.textContent=
f),e.previewLang&&(e.previewLang.textContent=r?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,w=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=w,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),r){canvasPreviewState.frameRenderToken+=1;const v=canvasPreviewState.frameRenderToken,
k=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),v);e.frame&&(e.frame.srcdoc=k,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{v!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:v,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}o(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const a=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
a&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}o(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(a=>a&&a.key===n);return i===-1?!1:applyCanvasSelection(i)}
o(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
a=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:a}}o(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const c=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[c],selectedIndex:0,selectedKey:c.key||t.codeKey||""}}const a=[];let r=-1;if(i.forEach(
(c,u)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const b=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},u);a.push(b),r===-1&&f.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
r===-1&&(r=0);const l=a[r]||a[0]||null;return{blocks:a,selectedIndex:r,selectedKey:l&&l.key?l.key:t.
codeKey||""}}o(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
if(!e)return!1;const t=collectCanvasBlocksFromButton(e);if(!t||!t.blocks||!t.blocks.length)return!1;
const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[],i=n.findIndex(r=>r&&r.
key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(i);const a=t.blocks[t.selectedIndex]||
t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.rawText=a&&a.code!==void 0&&
a.code!==null?String(a.code):"",canvasPreviewState.renderText=canvasPreviewState.rawText,canvasPreviewState.
selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.selectedKey=t.selectedKey||
a&&a.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),canvasPreviewState.lastCanvasData=
{renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:a,primaryIndex:canvasPreviewState.
selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.mobileView="preview",syncCanvasPanelViewUi(
"preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}o(previewCanvasCodeFromButton,"previewCanvasCod\
eFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&e.code!==null?
e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(n,t))return sanitizeHtmlForPreview(
t);const a=n?`Canvas Preview: ${n}`:"Canvas Preview",r=escapeHtml(t||"");return`<!doctype html><html\
 lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-sc\
ale=1"><title>${escapeHtml(a)}</title><style>
                :root { color-scheme: dark; }
                html, body { margin: 0; min-height: 100%; background: #0b1220; color: #e5e7eb; font-\
family: "Noto Sans JP", system-ui, -apple-system, "Segoe UI", sans-serif; }
                body { box-sizing: border-box; padding: 16px; }
                .frame {
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(2, 6, 23, 0.94)\
);
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    border-radius: 14px;
                    padding: 14px;
                    box-shadow: 0 20px 48px rgba(0, 0, 0, 0.34);
                }
                .label {
                    font-size: 11px;
                    text-transform: uppercase;
                    letter-spacing: 0.14em;
                    color: #67e8f9;
                    margin-bottom: 10px;
                }
                pre {
                    margin: 0;
                    white-space: pre-wrap;
                    word-break: break-word;
                    overflow-wrap: anywhere;
                    font-family: "JetBrains Mono", "Noto Sans Mono", ui-monospace, SFMono-Regular, M\
enlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    color: #e2e8f0;
                }
                .muted { color: #94a3b8; }
            </style></head><body><div class="frame"><div class="label">${escapeHtml(a)}</div><pre>${r||
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}o(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,t={}){const n=t.persist!==
!1;if(canvasModeEnabled=!!e,n)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const i=get("enable-canvas-mode");if(i&&i.checked!==canvasModeEnabled&&(i.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(t.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!t.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}o(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}o(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,t,n){let i=normalizeMarkdownNewlines(
e);const a=normalizeMarkdownNewlines(n);if(!a&&a!=="")return i;const r=t?[String(t),""]:[""];for(const l of[
"`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);for(const f of r){const b=`${u}${f}
`,y=`
${u}`,w=b+a+y;i.includes(w)&&(i=i.split(w).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const r of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);
[`${r}${u}
${i}
${u}`,`${r}
${u}
${i}
${u}`,`
${r}${u}
${i}
${u}`,`
${r}
${u}
${i}
${u}`].forEach(b=>{n.includes(b)&&(n=n.split(b).join(`
`))})}return n}o(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}o(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}o(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[];if(!t)return{text:"",executions:n};const i=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let a=t.replace(i,(r,l,c)=>{const u=String(c||"").trim();try{const f=JSON.parse(u);n.push({code:f&&f.
code!=null?String(f.code):"",output:f&&f.output!=null?String(f.output):""})}catch{n.push({code:u,output:""})}
return`
`});return n.forEach(r=>{r.code&&(a=stripExactFencedBlock(a,"python",r.code),a=stripExactFencedBlock(
a,"py",r.code)),a=stripVisiblePythonOutputBlock(a,r.output)}),a=a.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:a,executions:n}}o(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(r=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(r)?n.push(r.trim()):
i.push(r)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}o(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}o(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",a=e&&e.output!=null?String(e.output):"";let r="";try{window.
hljs&&typeof window.hljs.highlight=="function"?r=window.hljs.highlight(i,{language:"python"}).value:
r=escapeHtml(i)}catch{r=escapeHtml(i)}const l=escapeHtml(a),c=encodeURIComponent(i).replace(/'/g,"%2\
7"),u=encodeURIComponent(a).replace(/'/g,"%27"),f=hashString(`pyexec-detail
${i}
${a}
${t}`),b=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",y=`<button class="download-btn" data-\
code="${c}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,w=`<button class="coding-target-btn" data-code="${c}" data-code-key="${f}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${f}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(b)}</span><div class="code-actions">${w}${y}<button class="copy-btn" data-copy="code" \
data-code="${c}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${u}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${r}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${l}</code></pre></div></div></div>`}o(buildPythonExecDetailBoxHtml,"buildP\
ythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/python-\
execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"","/py\
thon-execution")}showModal("python-exec-modal")}o(showPythonExecDetailModal,"showPythonExecDetailMod\
al");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("pytho\
n-exec-modal-body"),a=get("python-exec-modal-title");if(!n||!i)return;const r=t&&Array.isArray(t.python_executions)?
t.python_executions:[];if(!r.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(a){const l=r.length>1?`\uFF08${r.length}\u4EF6\uFF09`:"";a.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${l}`}i.innerHTML=r.map((l,c)=>buildPythonExecDetailBoxHtml(l,c,r.length)).join(
""),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(!0,{persist:!1})),showPythonExecDetailModal(
e)}o(openPythonExecDetail,"openPythonExecDetail"),window.openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){
get("python-exec-modal")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&
history.back())}o(closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;
function buildAiMarkdownHtml(e){const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(
t.text,t.notes),i=canvasModeEnabled?parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,
rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(i),refreshCanvasPreviewPanel());const a=document.
createElement("div");return a.className="prose prose-invert text-sm break-words",a.innerHTML=sanitizeMarkdownHtml(
i.renderText),wrapRenderedSvgBoxes(a),lowBandwidthMode||(maybeNeedsHighlight(i.renderText,a)&&ensureHighlightLoaded().
catch(()=>{}),maybeNeedsMathJax(i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),a.outerHTML}o(buildAiMarkdownHtml,
"buildAiMarkdownHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes),r=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,
blocks:[],primaryBlock:null,rawText:a};if(canvasModeEnabled&&(updateCanvasPreviewState(r),refreshCanvasPreviewPanel()),
n.incrementalMath){const l=document.createElement("template");l.innerHTML=sanitizeMarkdownHtml(r.renderText,
{streamMathSegments:!0});const c=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(f=>{const b=f.getAttribute("data-stream-math-key");b&&c.set(b,f)});const u=[];l.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(f=>{const b=c.get(f.getAttribute(
"data-stream-math-key"));b?f.replaceWith(b):u.push(f)}),e.replaceChildren(l.content),wrapRenderedSvgBoxes(
e),queueHighlight(e,r.renderText),queueIncrementalMathTypeset(u);return}e.innerHTML=sanitizeMarkdownHtml(
r.renderText),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,r.renderText)}o(renderAiMarkdownInto,
"renderAiMarkdownInto");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}o(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");function renderMessage(e,t,n,i,a,r,l=null,c=!0,u=null,f=null,b=null,y=null,w=null,v=null,k=null,_=null,C=!0,L=null,$=null,F=null){
const Y=t==="user",X=Y?"bg-blue-600":"bg-gray-700",Ae=Y?"justify-end":"justify-start";messageStore[e]=
n;const R=!Y&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},V=Y?n:R.text;let ee=f;
if(ee==null){const ie=b!=null?Number(b):0,pe=y!=null?Number(y):0;(b!=null||y!=null)&&(ee=ie+pe)}messageMeta[e]=
{tokens_in:b,tokens_out:y,tokens_total:ee,tokens_content:v,tokens_thought:k,is_encrypted:w,role:t,model:r,
parent_id:L,quote_text:u,image_url:i,gem_name:$,batch_job:F,python_executions:Y?[]:R.executions||[]};
let we="";u&&(we=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-g\
ray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(u)}\
</div>`);let ce="";if(a&&!Y){let ie="";try{ie=JSON.parse(a).text||""}catch{ie=a}ie&&(ce=`<div class=\
"thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brai\
n text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(ie)}\
</div></div>`)}let Se="";if(i)try{const ie=JSON.parse(i);if(ie.length){const pe=[];if(ie.forEach(W=>{
let he=W,pt="unknown";if(he&&typeof he=="object"&&(pt=normalizeAttachmentSource(he.source),he=he.filepath||
he.path||he.url||he.file||""),he=normalizeAttachmentPath(he)||he,!he)return;setAttachmentSourceForPath(
he,pt);const ct=he.replace(/^\d+\//,""),mt=buildFileUrl(ct),Ne=buildAttachmentPreviewUrl(ct),Ve=he.split(
"/").pop(),bt=Ve.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(bt)?pe.push(
buildChatImageHtml(Ne,{viewerSrc:mt,alt:Ve,title:Ve,filename:Ve})):pe.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${mt}')" title="${Ve}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Ve}</span></div>`)}),pe.length>
0){let W="grid-multi";pe.length===1?W="grid-1":pe.length===2?W="grid-2":pe.length===3?W="grid-3":pe.
length===4&&(W="grid-4"),Se=`<div class="image-grid ${W}">${pe.join("")}</div>`}}}catch{}const Pe=Y?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,Ce=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${Y?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${Pe}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
ge=[];!Y&&r&&ge.push(escapeHtml(r)),$&&(Y?ge.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml($)}</span>`):ge.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml($)}</span>`));const de=[];if(b!=null&&de.push(`In ${b}`),y!=null){let ie=`\
Out ${y}`;k!=null&&Number(k)>0&&(ie+=` (Thought ${k})`),de.push(ie)}if(de.length||f!=null){const ie=de.
length?de.join(" / "):`${f} tokens`;ge.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${ie}</button>`)}if(w!=null){const ie=w?"fa\
-lock":"fa-lock-open",pe=isAdminUser?w?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":w?"Encrypted":
"Plain",W=isAdminUser?w?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";ge.push(`<button class="${W}" title="${pe}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${ie}"></i></button>`)}if(!Y&&R.executions&&R.executions.length){
const ie=R.executions.length,pe=ie>1?`Python \xD7${ie}`:"Python";ge.push(`<button type="button" clas\
s="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\
\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${pe}</span></button>`)}const H=ge.
length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${ge.
join(" \u2022 ")}</div>`:"";let A;const j=!Y&&F?(()=>{const ie=String(F.state||"").toUpperCase(),pe=F.
status_text||(ie==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ie==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ie==="JOB_STATE_FAILED"||
ie==="JOB_STATE_CANCELLED"||ie==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ie==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(pe)}</div></div>`})():
"";Y?A=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(A=j+(V&&String(V).trim()?buildAiMarkdownHtml(V):F?'<div class="content-area prose prose\
-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(V)),A.includes("content-area")||(A=A.replace("prose ","content-area prose ")));let K="";
if(l){const ie=l.siblings[l.current-2],pe=l.siblings[l.current];K=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ie}\
)" ${ie?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${pe}\
)" ${pe?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const Q=c?"fade-in":"",ne=document.createElement("div");ne.className=`flex ${Ae} m\
b-4 ${Q} relative message-group group`,ne.id=`msg-${e}`,ne.innerHTML=`<div class="message-bubble ${X}\
 text-white p-4 rounded-2xl shadow-md relative">${Ce}${we}${ce}${A}${Se}${K}${H}</div>`;const Te=_||
get("chat-container");return Te&&(Te.appendChild(ne),C&&scrollToBottom(),Y||(queueMessageDecorations(
ne,V),syncCodingTargetButtons(ne),syncCodingModeUi(codingModeEnabled,{persist:!1}))),ne}o(renderMessage,
"renderMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const t={
modal:"token-details"};e!==null&&(t.messageId=e),history.pushState(t,"","/token-details")}showModal(
"token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];
if(!t||!get("token-detail-modal"))return;const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",r=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",u=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detai\
l-in").innerText=a,get("token-detail-out").innerText=r,get("token-detail-content").innerText=l,get("\
token-detail-thought").innerText=c,get("token-detail-encrypted").innerText=u;const f=t.model?`${t.model}\
 (${t.role})`:`${t.role}`;get("token-detail-title").innerText=f,showTokenDetailModal(e)}o(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){get("token-detail-modal")&&(hideModal("token-deta\
il-modal"),!e&&location.pathname==="/token-details"&&history.back())}o(closeTokenDetail,"closeTokenD\
etail");function openEncryptionSettings(e){const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}
o(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get("encrypti\
on-status-modal"))return;const n=get("encryption-status-title"),i=get("encryption-status-body"),a=get(
"encryption-status-admin-actions"),r=get("encryption-status-admin-toggle"),l=!!e;l?(n&&(n.innerText=
"\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\
\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
a&&r&&(!!(isAdminUser&&currentThreadId)?(a.classList.remove("hidden"),r.dataset.enable=l?"0":"1",r.disabled=
!1,r.textContent=l?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
r.className=l?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):a.classList.add("hidden")),showEncryptionStatusModal()}o(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}o(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const t=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${t?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const i=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,t,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=i}}o(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}o(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}o(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}o(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=o((e,t,n,i)=>{
const a=(e||"").toLowerCase();return!a.includes("gemini")||a.includes("image")||a.includes("nano")||
a.includes("tts")||a.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=o(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),a=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(r=>{let l=!1;function c(){n&&n.removeEventListener("cli\
ck",f),i&&i.removeEventListener("click",b),a&&a.removeEventListener("click",b),e.removeEventListener(
"click",y,!0)}o(c,"cleanup");function u(w){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(w)}o(u,"finalize");
function f(){u(!0)}o(f,"onOk");function b(){u(!1)}o(b,"onCancel");function y(w){w.target===e&&(w.preventDefault(),
w.stopImmediatePropagation(),b())}o(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",b),a&&a.addEventListener("click",b),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),u=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${r}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("\
chat-container");if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",u);else{
const b=document.createElement("div");b.innerHTML=u;const y=b.firstElementChild;y&&f.appendChild(y)}
n&&scrollToBottom()}}o(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){
if(!e||e.getAttribute("data-stream-transition")==="1")return;const t=e.querySelector(".content-area");
t&&(t.classList.remove("pending-shimmer","skeleton-pending"),t.removeAttribute("data-skeleton-kind")),
e.setAttribute("data-stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add(
"ai-stream-transition"),t&&(t.classList.add("ai-stream-content-transition"),setTimeout(()=>{t&&t.classList.
remove("ai-stream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-trans\
ition")},320)}o(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){
return e==null||e===""?null:String(e)}o(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}o(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const t=Array.from(e.querySelectorAll(".prose")).some(c=>String(c.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(c=>!!String(c.textContent||
"").trim()&&c.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const a=e.parentElement;
if(!a)return null;const r=a.cloneNode(!0);r.setAttribute("data-local-stopped-partial","1"),r.classList.
remove("fade-in");const l=r.querySelector(".message-bubble");if(l&&(l.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),l.removeAttribute("data-stream-transition"),l.removeAttribute("id"),!r.
querySelector('[data-stopped-partial-note="1"]'))){const c=document.createElement("div");c.setAttribute(
"data-stopped-partial-note","1"),c.className="text-[10px] text-amber-200/90 mt-2 text-right",c.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",l.appendChild(c)}return{html:r.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}o(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const a=get("chat-\
container");return a?(a.querySelectorAll('[data-local-stopped-partial="1"]').forEach(r=>r.remove()),
a.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}o(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}o(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}o(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}o(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var c,u;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((u=t.retryDelayMs)!=
null?u:180)||0),a=!!t.notifyOnFailure,r=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!l||r&&l!==r)return!1;for(let f=0;f<=n;f++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==l?!1:(await loadMessages(l,{preserveDraft:!0,silent:!0}),
!0)}catch{f<n&&i>0&&await new Promise(y=>setTimeout(y,i))}return a&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===l&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}o(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(t){console.warn("Vi\
bration failed:",t)}}o(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const t=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(n=>n.kind==="minimal"&&!minimalPromptMode?!1:n.label.
toLowerCase().includes(t)||n.description.toLowerCase().includes(t))}o(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,t){if(String(e||"").toLowerCase()!=="thinking")
return e;const i=String(t||"").trimStart().match(/^\/thinking(\s+.*)$/i);return i?`thinking${i[1]}`.
toLowerCase():e}o(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const t=String(e||"").trim().toLowerCase();if(!t||t==="toggle"||t==="\u5207\u66FF"||t==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(t))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(t))return!1}o(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(r=>r.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(r=>r.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const a=String(n.presetArgument||t||"").trim();if(i.selectId){if(!a)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} ${n.id==="effort"?"none / low / medium / high / xhigh / max":"default\
 / none"}`,"info"),!1;const r=get(i.selectId),l=a.toLowerCase(),c=r?Array.from(r.options).find(u=>u.
value.toLowerCase()===l||u.textContent.trim().toLowerCase()===l):null;return!r||!c?(showToast(`${n.label}\
: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(r.value=c.value,r.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const r=a.toLowerCase(),
l={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},c=parseSlashToggleArgument(
a),u=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(l,r)){u&&!u.checked&&!u.disabled&&(u.
checked=!0,u.dispatchEvent(new Event("change",{bubbles:!0})));const f=get("thinking-level");return f&&
(f.value=l[r],f.dispatchEvent(new Event("change",{bubbles:!0}))),refreshMinimalOptionItems(),showToast(
`Thinking: ${r}`,"success"),!0}if(c===void 0)return showToast("\u4F7F\u3044\u65B9: /thinking on / off / min / low /\
 mid / high","info"),!1}if(i.checkboxId&&a){const r=parseSlashToggleArgument(a);if(r===void 0)return showToast(
`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const l=get(i.checkboxId);if(r!==null&&l&&l.checked===
r)return showToast(`${i.label}: ${r?"ON":"OFF"}`,"info"),!0}return handleMinimalOptionClick(i),!0}o(
executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){const t=String(
e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||"",i=n.
match(/^[a-z][\w-]*/i);return i?i[0]:n}o(extractSlashCommandToken,"extractSlashCommandToken");function hideSlashCommandSuggestions(){
const e=get("slash-command-suggestions");e&&e.classList.add("hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=
0}o(hideSlashCommandSuggestions,"hideSlashCommandSuggestions");function showPendingSlashCommandIndicator(e){
const t=get("slash-command-indicator"),n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.
find(r=>r.id===e);n.textContent=i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");
const a=get("prompt-input");a&&i&&(a.dataset.originalPlaceholder=a.placeholder,a.placeholder=i.argumentHint||
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}
o(showPendingSlashCommandIndicator,"showPendingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){
const e=get("slash-command-indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get(
"prompt-input");t&&t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.
dataset.originalPlaceholder);const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}
o(hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const a=visibleSlashCommands(e);if(a.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((w,v)=>{const k=document.createElement(
"div");k.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${v===
slashSelectedIndex?"bg-gray-700":""}`,k.innerHTML=`
                    <i class="fas ${w.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${w.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${w.description}</div>
                    </div>
                `;let _=!1;k.addEventListener("pointerdown",C=>{typeof C.button=="number"&&C.button!==
0||(C.preventDefault(),_=!0,selectSlashCommand(w.id))}),k.addEventListener("click",C=>{C.preventDefault(),
_||selectSlashCommand(w.id)}),k.onmouseenter=()=>{slashSelectedIndex=v,showSlashCommandSuggestions(e)},
n.appendChild(k)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,u=r.top,f=260,
b=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&u>c){const w=Math.min(f,u-b);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(f,c-b);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}o(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const l=String(n||"").trimStart();t.value=
l.substring(1+i.length).trimStart()}else{const l=n.lastIndexOf("/");l!==-1?t.value=n.substring(0,l).
trimEnd():t.value=""}hideSlashCommandSuggestions();const a=SLASH_COMMANDS.find(l=>l.id===e),r=t.value.
trim();if(a&&a.autocompleteArgument&&!r){t.value=`${a.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(a&&a.kind==="minimal"&&(!a.
requiresArgument||r)){t.value="",executeMinimalSlashCommand(e,r),t.dispatchEvent(new Event("input",{
bubbles:!0})),t.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),t.focus(),t.
dispatchEvent(new Event("input",{bubbles:!0}))}o(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
default_model:{label:"\u65E2\u5B9A\u306E\u30E2\u30C7\u30EB",tab:"general",control:"set-default-model"},
default_vision_model:{label:"Vision Model",tab:"general",control:"set-default-vision-model"},use_last_chat_settings:{
label:"\u524D\u56DE\u306E\u8A2D\u5B9A\u3092\u7D99\u7D9A",tab:"general",control:"set-use-last-setting\
s"},default_enable_search:{label:"\u65E2\u5B9A\u306ESearch",tab:"general",control:"set-default-searc\
h"},default_enable_url_context:{label:"\u65E2\u5B9A\u306EURLs",tab:"general",control:"set-default-ur\
l-context"},default_enable_maps:{label:"\u65E2\u5B9A\u306EMaps",tab:"general",control:"set-default-m\
aps"},default_enable_python:{label:"\u65E2\u5B9A\u306EPython",tab:"general",control:"set-default-pyt\
hon"},default_enable_file_creation:{label:"\u65E2\u5B9A\u306EFile",tab:"general",control:"set-defaul\
t-file-creation"},default_enable_thinking:{label:"\u65E2\u5B9A\u306EThinking",tab:"general",control:"\
set-default-thinking"},default_thinking_level:{label:"Thinking Level",tab:"general",control:"set-def\
ault-thinking-level"},default_thinking_budget:{label:"Thinking Budget",tab:"general",control:"set-de\
fault-thinking-budget"},default_reasoning_effort:{label:"Reasoning Effort",tab:"general",control:"se\
t-default-reasoning-effort"},default_enable_system_prompt:{label:"\u65E2\u5B9A\u306ESysPrompt",tab:"\
general",control:"set-default-sys-prompt"},default_enable_mcp:{label:"\u65E2\u5B9A\u306EMCP",tab:"ge\
neral",control:"set-default-mcp"},default_safety_setting:{label:"\u65E2\u5B9A\u306ESafety",tab:"gene\
ral",control:"set-default-safety"},auto_search_on_links:{label:"X\u30EA\u30F3\u30AF\u306E\u81EA\u52D5\u691C\u7D22",
tab:"general",control:"set-auto-search-links"},mic_transcribe_mode:{label:"\u30DE\u30A4\u30AF\u6587\u5B57\u8D77\u3053\u3057\u65B9\u5F0F",
tab:"general",control:"set-mic-transcribe-mode"},stt_model:{label:"STT\u30E2\u30C7\u30EB",tab:"gener\
al",control:"set-stt-model"},llm_transcribe_prompt:{label:"LLM\u6587\u5B57\u8D77\u3053\u3057\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"general",control:"set-llm-transcribe-prompt"},enter_to_send:{label:"Enter\u3067\u9001\u4FE1",tab:"\
general",control:"set-enter-to-send"},compact_prompt_mode:{label:"\u30D7\u30ED\u30F3\u30D7\u30C8\u30D0\u30FC\u8868\u793A",
tab:"general",control:"set-compact-prompt-mode"},minimal_prompt_mode:{label:"\u30DF\u30CB\u30DE\u30EB\u8868\u793A",
tab:"general",control:"set-minimal-prompt-mode"},temp_chat_timeout_seconds:{label:"\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u4FDD\u6301\u6642\u9593",
tab:"general",control:"set-temp-chat-timeout-seconds"},system_prompt:{label:"\u30E6\u30FC\u30B6\u30FC\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"prompt",control:"sys-prompt-text"},system_prompt_enabled:{label:"\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8",
tab:"prompt",control:"set-global-sys-prompt-enabled"},apply_global_system_prompt:{label:"\u30E6\u30FC\u30B6\u30FC\u30D7\u30ED\u30F3\u30D7\u30C8\u306E\u9069\
\u7528",tab:"prompt",control:"set-apply-global-sys-prompt"},apply_auto_system_prompt_notices:{label:"\
\u81EA\u52D5\u6CE8\u5165\u30D7\u30ED\u30F3\u30D7\u30C8",tab:"prompt",control:"set-apply-auto-sys-pro\
mpt-notices"},auto_system_prompt_notices_config:{label:"\u81EA\u52D5\u6CE8\u5165\u30D7\u30ED\u30F3\u30D7\u30C8\u8A2D\u5B9A",
tab:"prompt",control:"auto-sys-prompt-settings"},theme_color:{label:"\u30C6\u30FC\u30DE\u30AB\u30E9\u30FC",
tab:"display",control:"set-theme-color"},liquid_glass_enabled:{label:"Liquid Glass",tab:"display",control:"\
set-liquid-glass"},use_sw_cache:{label:"\u9AD8\u901F\u30AD\u30E3\u30C3\u30B7\u30E5",tab:"data",control:"\
set-use-sw-cache"},enable_latency_metrics:{label:"\u30EC\u30B9\u30DD\u30F3\u30B9\u901F\u5EA6\u306E\u8A08\u6E2C",
tab:"data",control:"set-latency-metrics"},enable_client_debug_log:{label:"\u30C7\u30D0\u30C3\u30B0\u30ED\u30B0\u306E\u62E1\u5F35\u9001\u4FE1",
tab:"data",control:"set-client-debug-log"},bot_detection_enabled:{label:"Bot Detection",tab:"securit\
y",control:"set-bot-detect"},skip_2fa_on_google_login:{label:"Google\u30ED\u30B0\u30A4\u30F3\u6642\u306E2FA",
tab:"2fa",control:"set-skip-2fa-google"},default_2fa_method:{label:"\u65E2\u5B9A\u306E2FA\u65B9\u5F0F",
tab:"2fa",control:"set-default-2fa-method"},rich_paste_prompt_default:{label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\u30D7\u30ED\u30F3\u30D7\u30C8",
modal:"rich-paste",control:"rich-paste-prompt"},rich_paste_prompt_use_custom_default:{label:"\u30EA\u30C3\u30C1\u8CBC\u308A\u4ED8\u3051\
\u306E\u65E2\u5B9A\u5024",modal:"rich-paste",control:"rich-paste-use-default"}};function formatAiSettingValue(e){
if(e===!0)return"ON";if(e===!1)return"OFF";if(e==="(\u66F4\u65B0)")return"\u66F4\u65B0\u6E08\u307F";
if(e==null||e==="")return"\u672A\u8A2D\u5B9A";if(typeof e=="object")try{return JSON.stringify(e)}catch{
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}o(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,t){
const n=get(`tab-${e}`);let i=get(t);if(!n||!i)return null;for(;i.parentElement&&i.parentElement!==n;)
i=i.parentElement;return i.parentElement===n?i:get(t)}o(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const t=AI_SETTING_JUMP_TARGETS[e];if(!t){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(t.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const n=get(t.control);n&&(n.scrollIntoView({behavior:"smooth",block:"center"}),n.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const n=findSettingsJumpElement(t.tab,t.control);n?jumpToSetting(t.tab,n):switchTab(t.tab||"general")},
320)}o(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const t=e.querySelector(".msg-controls");t&&t.remove()}o(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,t,n="update"){const i=Object.
entries(e||{}),a=`settings-result-${Date.now()}`,r=n==="inspect",l=i.length?r?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
r?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
c=renderMessage(a,"assistant",l,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!c)
return;removeEphemeralMessageControls(c);const u=c.querySelector(".message-bubble");if(!u||!i.length)
return;const f=document.createElement("div");f.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,w])=>{const v=AI_SETTING_JUMP_TARGETS[y]||{label:y},k=document.createElement("button");k.
type="button",k.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const _=document.createElement("span");_.className="min-w-0 flex-1";const C=document.createElement(
"span");C.className="block text-xs font-bold text-blue-200",C.textContent=v.label;const L=document.createElement(
"span");L.className="block mt-0.5 text-[11px] text-gray-300 break-words",L.textContent=formatAiSettingValue(
w);const $=document.createElement("i");$.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",_.appendChild(C),_.appendChild(L),k.appendChild(_),k.appendChild($),k.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(k)});const b=u.querySelector(".message-footer-\
meta");b?u.insertBefore(f,b):u.appendChild(f),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const r=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${r}" class="flex justify-start mb-4 ai-pending-row f\
ade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-\
tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const u=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),f=get(r);if(f&&f.remove(),u&&u.status==="ok"&&u.mode==="inspect"&&u.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(u.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
u.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(u.current,t,"inspect");
return}if(u&&u.status==="ok"&&u.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
u.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(u.applied).length}\u9805\u76EE\uFF09`,"success");try{const w=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(v=>v.json());populateAiSafeFormFields(w),cacheUserSettings(w)}catch{}renderAiSettingsResultBubble(
u.applied,t);return}const b=u.message||u.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${b}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${b}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(b,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const u=get(r);u&&u.remove();const f=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(f),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
o(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}o(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const a=e.toLowerCase(),r=loadedGems.filter(v=>v.name.toLowerCase().includes(a)||v.description&&
v.description.toLowerCase().includes(a));if(r.length===0){hideGemSuggestions();return}gemSelectedIndex>=
r.length&&(gemSelectedIndex=0),n.innerHTML="",r.forEach((v,k)=>{const _=document.createElement("div");
_.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${k===gemSelectedIndex?
"bg-gray-700":""}`,_.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(v.name)}</div>
                        ${v.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
v.description)}</div>`:""}
                    </div>
                `,_.onclick=()=>selectGemSuggestion(v),_.onmouseenter=()=>{gemSelectedIndex=k,showGemSuggestions(
e)},n.appendChild(_)});const l=i.getBoundingClientRect(),c=window.innerHeight,u=c-l.bottom,f=l.top,b=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",u<180&&f>u){const v=Math.min(b,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(b,u-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${v}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}o(showGemSuggestions,
"showGemSuggestions");function selectGemSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideGemSuggestions(),
activateGem(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}o(selectGemSuggestion,"sel\
ectGemSuggestion");function browserFastModeIneligibility(e){const t=String(get("model-select")?get("\
model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!t.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(t))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeGem)return"Gems\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(currentQuote||editingMessageId)return"\u5F15\u7528\u30FB\u7DE8\u96C6\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(codingModeEnabled)return"Coding Mode\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(["enable-search","enable-url-context","enable-maps","enable-sys-prompt","enable-prompt-cache","en\
able-mcp"].some(l=>{const c=get(l);return!!(c&&c.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const i=get("thread-custom-instruction");if(i&&String(i.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const a=Array.from(browserFastLocalFiles.values());return a.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":a.reduce((l,c)=>l+Number(c.file&&c.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":a.some(l=>!l.
file||!String(l.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}o(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const a=String(i.result||""),r=a.indexOf(",");if(r<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
a.slice(r+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}o(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const a=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(r=>{r&&
a.push({thoughtSignature:String(r)})}),i.text&&a.push({text:String(i.text)});for(const r of Array.isArray(
i.images)?i.images:[])try{const l=await fetch(buildFileUrl(r.path),{credentials:"same-origin",cache:"\
no-store"});if(!l.ok)throw new Error(`HTTP ${l.status}`);const c=await l.blob();a.push({inlineData:{
mimeType:r.mime_type||c.type||"application/octet-stream",data:await fileToBase64Payload(c)}})}catch{
n++}a.length&&t.push({role:i.role,parts:a})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}o(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(){
const e=Array.from(browserFastLocalFiles.entries());for(const[t,n]of e){if(!n||!n.file||!n.rowObj)throw new Error(
"\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");
if(n.rowObj.status&&(n.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(n.file,n.rowObj))throw new Error(`${n.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);browserFastLocalFiles.delete(t)}}o(uploadBrowserFastLocalFiles,
"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){const t=get("enable-thinking");
if(!t||!t.checked)return null;const n=String(get("thinking-level")?get("thinking-level").value:"high").
toLowerCase();if(e.includes("2.5")){const a=Number(get("thinking-budget")?get("thinking-budget").value:
4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(a)?Math.max(0,Math.min(32768,Math.trunc(
a))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["MEDIUM","HIGH"].includes(i)&&(i="MEDIU\
M"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}
o(browserFastThinkingConfig,"browserFastThinkingConfig");function browserFastPythonBoxHtml(e){return`\
<div class="code-wrapper python-box collapsed" data-py-id="${e}" data-collapsed="true" data-code-key\
="${e}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Exec\
ution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" a\
ria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code"\
 data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class\
="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-alig\
n-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="pyth\
on-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div clas\
s="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext p\
ython-output"></code></pre></div></div></div>`}o(browserFastPythonBoxHtml,"browserFastPythonBoxHtml");
function updateBrowserFastPythonBox(e,t,n){if(e){if(t==="code"){const i=n==null?"":String(n),a=e.querySelector(
".python-code");a&&(a.textContent=i,a.removeAttribute("data-highlighted"),queueHighlight(e,i));const r=e.
querySelector('.copy-btn[data-copy="code"]');r&&r.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}else if(t==="output"){const i=n==null?"":String(n),a=e.querySelector(".python-output");
a&&(a.textContent=i);const r=e.querySelector('.copy-btn[data-copy="output"]');r&&r.setAttribute("dat\
a-code",encodeURIComponent(i).replace(/'/g,"%27"))}}}o(updateBrowserFastPythonBox,"updateBrowserFast\
PythonBox");async function sendBrowserFastMessage(e){const t=String(get("model-select").value||"").trim(),
n=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==t)throw new Error(
"\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const i=Array.from(browserFastLocalFiles.values()),a=[];for(const R of i)a.push({inlineData:{mimeType:R.
file.type,data:await fileToBase64Payload(R.file)}});a.push({text:e});const r={},l=browserFastThinkingConfig(
t.toLowerCase());l&&(r.thinkingConfig=l);const c={contents:[...await buildBrowserFastHistoryContents(
n.history),{role:"user",parts:a}],generationConfig:r};!!(get("enable-python")&&get("enable-python").
checked)&&(c.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden"),renderMessage(Date.now(),"user",
e,null,null,null,null,!0,null,null,null,null,null,null,null,null,!0);const f=`browser-fast-${Date.now()}`;
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending\
-row fade-in"><div id="${f}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 roun\
ded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const b=get(f);activeStreamingBubbleId=f,setSendBtnToStopMode(),resumeChatAutoScroll(),
abortController=new AbortController;let y="",w="";const v=[];let k=null,_=null,C=!1;const L={},$=[];
let F=null,Y="";const X=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Ae=!1;try{const R=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
t)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(c),signal:abortController.
signal}));if(!R.ok){const de=await R.json().catch(()=>({}));throw new Error(de&&de.error&&de.error.message?
de.error.message:`Gemini API HTTP ${R.status}`)}window.ConnectionMonitor&&(Ae=!0,window.ConnectionMonitor.
operationStarted()),X&&X.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const V=R.body.getReader(),ee=new TextDecoder;let we="";const ce=o(de=>{const H=de.split(
/\r?\n/).filter(K=>K.startsWith("data:")).map(K=>K.slice(5).trim()).join("");if(!H||H==="[DONE]")return;
const A=JSON.parse(H);if(A.error)throw new Error(A.error.message||"Gemini API error");if((Array.isArray(
A.candidates)?A.candidates:[]).forEach(K=>{(K&&K.content&&Array.isArray(K.content.parts)?K.content.parts:
[]).forEach(ne=>{if(ne&&typeof ne.thoughtSignature=="string"&&!v.includes(ne.thoughtSignature)&&v.push(
ne.thoughtSignature),ne&&ne.executableCode&&typeof ne.executableCode.code=="string"){const ie=ne.executableCode.
code;y+=`
\`\`\`python
${ie}
\`\`\`
`,F=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,Y=ie,L[F]||(b.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(F)),L[F]=b.querySelector(`[data-py-id="${F}"]`)),updateBrowserFastPythonBox(
L[F],"code",ie);return}if(ne&&ne.codeExecutionResult&&typeof ne.codeExecutionResult.output=="string"){
const ie=ne.codeExecutionResult.output;y+=`
**Output:**
\`\`\`
${ie}
\`\`\`
`;const pe=F||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;$.push({code:Y||
"",output:ie}),L[pe]||(b.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(pe)),L[pe]=b.querySelector(
`[data-py-id="${pe}"]`)),updateBrowserFastPythonBox(L[pe],"output",ie);return}const Te=typeof ne.text==
"string"?ne.text:"";Te&&(ne.thought===!0?w+=Te:y+=Te)})}),!C&&(y||w)){beginPendingToStreamTransition(
b);const K=b.querySelector(".content-area");K&&K.remove(),C=!0}w&&(_||(b.insertAdjacentHTML("afterbe\
gin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></\
div>'),_=b.querySelector(".thought-content")),_.textContent=w),y&&(k||(k=document.createElement("div"),
k.className="content-area prose prose-invert text-sm break-words",b.appendChild(k)),renderAiMarkdownInto(
k,y,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:de,value:H}=await V.
read();if(de)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),X&&X.setPhase(
"receiving"),we+=ee.decode(H,{stream:!0});const A=we.split(/\r?\n\r?\n/);we=A.pop()||"",A.forEach(ce)}
if(we+=ee.decode(),we.trim()&&ce(we),!y.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
k&&renderAiMarkdownInto(k,y,{incrementalMath:!0}),_&&_.classList.add("collapsed"),$.length&&(y+=$.map(
de=>`
\`\`\`pyexec
${JSON.stringify(de)}
\`\`\`
`).join("")),i.length&&(X&&X.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),X&&X.setPhase("saving");const Se=collectImageUrlsForSend(),
Pe=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:y,thought_content:w,model:t,image_urls:Se,temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:n.parent_id||null,thought_signatures:v,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}),b),Ce=await Pe.json().catch(()=>({}));if(!Pe.ok||!Ce.thread_id)throw new Error(
Ce.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");const ge=!currentThreadId;currentThreadId=
String(Ce.thread_id),currentParentId=Ce.assistant_message_id||null,currentLeafId=Ce.assistant_message_id||
null,resetUploadState(),browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,skipHistory:!ge}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\
\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F","success",!1)}catch(R){if(R.name!=="AbortError"){showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9: ${R.message}`,"error",!0),get("prompt-input").value||(get("prompt-i\
nput").value=e);const V=R.message||"\u30A8\u30E9\u30FC";b&&b.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
V));try{let ee=y||"";$.length&&(ee+=$.map(Ce=>`
\`\`\`pyexec
${JSON.stringify(Ce)}
\`\`\`
`).join(""));const we=buildChatErrorMarkdown(V,ee),ce=i.length?[]:collectImageUrlsForSend(),Se=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:we,
thought_content:w||"",model:t,image_urls:ce,temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||
null,parent_id:n&&n.parent_id?n.parent_id:null,thought_signatures:v,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),b),Pe=await Se.
json().catch(()=>({}));if(Se.ok&&Pe.thread_id){const Ce=!currentThreadId;currentThreadId=String(Pe.thread_id),
currentParentId=Pe.assistant_message_id||null,currentLeafId=Pe.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!Ce}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(ee){sendClientDebugLog("error",`Browser fa\
st error persist failed: ${ee&&ee.message?ee.message:ee}`)}}}finally{Ae&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),X&&X(),setSendBtnToSendMode(),activeStreamingBubbleId===f&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Bt;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const B=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(B);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const t=get("prompt-inp\
ut").value;if(pendingSlashCommand){const B=pendingSlashCommand,re=t.trim(),Ee=get("model-select")?get(
"model-select").value:null;if(B==="settings"){if(!re){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!Ee){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
re,Ee)}else executeMinimalSlashCommand(B,re)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const n=t.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(n&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(B=>B.id===
n[1].toLowerCase())){executeMinimalSlashCommand(n[1].toLowerCase(),n[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const i=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(i&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(i)setBrowserFastModeEnabled(!1);else{const B=browserFastModeIneligibility(
t);if(!B){try{await sendBrowserFastMessage(t)}catch(re){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${re.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${B}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(re){showToast(
re.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}t.trim()&&(promptHistory.length===
0||promptHistory[0]!==t)&&(promptHistory.unshift(t),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const a=collectAttachmentItemsForSend(),r=a.map(B=>B.path),l=a.filter(B=>normalizeAttachmentSource(
B.source)==="upload").map(B=>B.path);if(r.length>ATTACHMENT_MAX_FILES){showToast(`\u6DFB\u4ED8\u306F\u6700\u5927${ATTACHMENT_MAX_FILES}\
\u4EF6\u3067\u3059\u3002\u6DFB\u4ED8\u3092\u6E1B\u3089\u3057\u3066\u518D\u9001\u3057\u3066\u304F\u3060\u3055\u3044\u3002`,
"error",!0);return}const c=getModelMediaSupport(get("model-select").value),u=r.some(B=>isAudioPath(B)),
f=r.some(B=>isVideoPath(B)),b=(get("model-select").value||"").toLowerCase(),y=get("enable-python"),w=!!(y&&
y.checked);if(u&&!c.audio||f&&!c.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!t.trim()&&r.length===0)return;if(isMistralOcrModel(
b)){const B=/https?:\/\/\S+/i.test(t);if(r.filter(Ee=>isAudioPath(Ee)||isVideoPath(Ee)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!r.length&&!B){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const v=t.trim();if(/^\/settings(?:\s|$)/i.
test(v)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(v)){const B=v.replace(/^\/settings\s*/i,"").trim();
if(!B){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ee=get("prompt-input");Ee.value="/settings ";const Re=extractSlashCommandToken(Ee.value);
lastSlashFilter=Re,showSlashCommandSuggestions(Re),Ee.focus();return}const re=get("model-select")?get(
"model-select").value:null;if(!re){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
B,re);return}if(isGeminiLocalPythonMode(b,u,f,w)&&!await confirmGeminiLocalPythonSwitch())return;let k=null,
_=[];if(codingModeEnabled){const B=collectCodingCandidates(t),re=B.filter(qe=>qe.prompt_source),Ee=B.
filter(qe=>!qe.prompt_source),Re=re.reduce((qe,je)=>qe+String(je.code||"").length,0);if(Re>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let Ke=3e5-Re;const tt=[];for(let qe=Ee.length-1;qe>=0;qe--){const je=String(Ee[qe].
code||"").length;je>Ke||(tt.unshift(Ee[qe]),Ke-=je)}_=codingTargetSelection?tt.slice(-1):[...re,...tt];
const rt=re.length?re[re.length-1]:null;if(k=codingTargetSelection?_[0]:rt||_[_.length-1]||null,codingModeEffective=
!!(k&&String(k.code||"").trim()),codingModeEffective&&k.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const qe=String(((Bt=get(
"model-select"))==null?void 0:Bt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(qe)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const C=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${t.length}\
 attachments=${r.length} search=${get("enable-search").checked}`);const L=t,$=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&r.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const F=editingMessageId,Y=currentParentId,X=F!=null;F&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Ae=[],R=o(B=>{if(B==null)
return;let re=document.getElementById(`msg-${B}`);for(;re;)re.classList&&re.classList.contains("mess\
age-group")&&(Ae.push({node:re,prevDisplay:re.style.display}),re.style.display="none"),re=re.nextElementSibling},
"hideRenderedBranchFrom"),V=o(()=>{Ae.forEach(({node:B,prevDisplay:re})=>{B&&(B.style.display=re||"")}),
Ae.length=0},"restoreHiddenBranch");F&&R(F);const ee=Date.now(),we=renderMessage(ee,"user",L,JSON.stringify(
r),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,Y,activeGem?activeGem.name:null);
let ce=!1;const Se=/(https?:\/\/)?(x\.com|twitter\.com)\//i,Pe=Se.test(L||"")||Se.test(currentQuote||
""),Ce="grok-4-fast-reasoning",ge=o(()=>{get("enable-search").checked=!0,get("model-select").value!==
Ce&&selectModelById(Ce)},"applyXLinkAuto");if(Pe&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)ge();else{const B=get("auto-search-banner"),re=get("auto-search-on-btn"),Ee=get(
"auto-search-off-btn"),Re=get("auto-search-remember");B&&re&&Ee&&(Re&&(Re.checked=!1),await new Promise(
Ke=>{B.classList.remove("hidden");const tt=o(rt=>{B.classList.add("hidden"),re.onclick=null,Ee.onclick=
null,Ke(rt)},"cleanup");re.onclick=()=>tt("enable"),Ee.onclick=()=>tt("disable")}).then(async Ke=>{Ke===
"enable"?(ge(),Re&&Re.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
ce=!0}))}const de=String(get("reasoning-effort").value||"").toLowerCase(),H=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&de==="none",A={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:L,model:get("model-select").value,image_urls:r,image_items:a,uploaded_image_urls:l,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:H?!1:get("enable-thin\
king").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:$,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&get("tts-voice-custom")?
get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-language")?get("tts-language").
value:null,tts_style:isTtsModel()&&get("tts-style")&&get("tts-style-wrap")&&!get("tts-style-wrap").classList.
contains("hidden")?get("tts-style").value:null,tts_speed:isTtsModel()&&get("tts-speed")?get("tts-spe\
ed").value:null,image_size:isGptImageModel()&&get("gpt-image-size")?get("gpt-image-size").value:null,
image_quality:isGptImageModel()&&get("gpt-image-quality")?get("gpt-image-quality").value:null,image_format:isGptImageModel()&&
get("gpt-image-format")?get("gpt-image-format").value:null,image_compression:isGptImageModel()&&get(
"gpt-image-compression")?get("gpt-image-compression").value:null,image_mask:isGptImageModel()?currentMaskImage:
null,gemini_image_aspect:isGeminiImageModel()&&get("gemini-image-aspect")?get("gemini-image-aspect").
value:null,gemini_image_size:isGeminiImageModel()&&get("gemini-image-size")?get("gemini-image-size").
value:null,grok_image_aspect:isGrokImageModel()&&get("grok-image-aspect")?get("grok-image-aspect").value:
null,grok_image_resolution:isGrokImageModel()&&get("grok-image-resolution")?get("grok-image-resoluti\
on").value:null,grok_image_quality:isGrokImageModel()&&get("grok-image-quality")?get("grok-image-qua\
lity").value:null,grok_image_format:isGrokImageModel()&&get("grok-image-format")?get("grok-image-for\
mat").value:null,grok_image_count:isGrokImageModel()&&get("grok-image-count")?get("grok-image-count").
value:null,ideogram_aspect:isIdeogramModel()&&get("ideogram-image-aspect")?get("ideogram-image-aspec\
t").value:null,ideogram_resolution:isIdeogramModel()&&get("ideogram-image-resolution")?get("ideogram\
-image-resolution").value:null,ideogram_quality:isIdeogramModel()&&get("ideogram-image-quality")?get(
"ideogram-image-quality").value:null,ideogram_speed:isIdeogramModel()&&get("ideogram-image-speed")?get(
"ideogram-image-speed").value:null,ideogram_magic_prompt:isIdeogramModel()&&get("ideogram-image-magi\
c")?get("ideogram-image-magic").value:null,ideogram_style_type:isIdeogramModel()&&get("ideogram-imag\
e-style")?get("ideogram-image-style").value:null,ideogram_negative_prompt:isIdeogramModel()&&get("id\
eogram-image-negative")?get("ideogram-image-negative").value:null,ideogram_count:isIdeogramModel()&&
get("ideogram-image-count")?get("ideogram-image-count").value:null,ideogram_seed:isIdeogramModel()&&
get("ideogram-image-seed")?get("ideogram-image-seed").value:null,xai_temperature:get("xai-temperatur\
e")?get("xai-temperature").value:null,xai_top_p:get("xai-top-p")?get("xai-top-p").value:null,xai_max_completion_tokens:get(
"xai-max-completion-tokens")?get("xai-max-completion-tokens").value:null,xai_seed:get("xai-seed")?get(
"xai-seed").value:null,xai_presence_penalty:get("xai-presence-penalty")?get("xai-presence-penalty").
value:null,xai_frequency_penalty:get("xai-frequency-penalty")?get("xai-frequency-penalty").value:null,
xai_stop:get("xai-stop")?get("xai-stop").value:null,xai_response_format:get("xai-response-format")?get(
"xai-response-format").value:null,xai_tool_choice:get("xai-tool-choice")?get("xai-tool-choice").value:
null,xai_parallel_tool_calls:get("xai-parallel-tool-calls")?get("xai-parallel-tool-calls").checked:!0,
xai_logprobs:get("xai-logprobs")?get("xai-logprobs").checked:!1,xai_top_logprobs:get("xai-top-logpro\
bs")?get("xai-top-logprobs").value:null,grok_video_duration:isGrokVideoModel()&&get("grok-video-dura\
tion")?get("grok-video-duration").value:null,grok_video_aspect:isGrokVideoModel()&&get("grok-video-a\
spect")?get("grok-video-aspect").value:null,grok_video_resolution:isGrokVideoModel()&&get("grok-vide\
o-resolution")?get("grok-video-resolution").value:null,gemini_video_duration:isGeminiVideoModel()&&get(
"gemini-video-duration")?get("gemini-video-duration").value:null,gemini_video_aspect:isGeminiVideoModel()&&
get("gemini-video-aspect")?get("gemini-video-aspect").value:null,gemini_video_resolution:isGeminiVideoModel()&&
get("gemini-video-resolution")?get("gemini-video-resolution").value:null,music_instrumental:isGeminiMusicModel()&&
get("music-instrumental")?get("music-instrumental").checked:!1,ocr_table_format:isMistralOcrModel()&&
get("ocr-table-format")?get("ocr-table-format").value:null,ocr_extract_header:isMistralOcrModel()&&get(
"ocr-extract-header")?get("ocr-extract-header").checked:!1,ocr_extract_footer:isMistralOcrModel()&&get(
"ocr-extract-footer")?get("ocr-extract-footer").checked:!1,ocr_include_blocks:isMistralOcrModel()&&get(
"ocr-include-blocks")?get("ocr-include-blocks").checked:!1,ocr_include_image_base64:isMistralOcrModel()&&
get("ocr-include-images")?get("ocr-include-images").checked:!0,ocr_pages:isMistralOcrModel()&&get("o\
cr-pages")?get("ocr-pages").value:null,transcription_language_codes:[],transcription_custom_vocabulary:[],
transcription_mode:"verbatim",transcription_diarization:!1,transcription_word_timestamps:!1,quote_text:currentQuote,
parent_id:Y,parent_id_explicit:X,disable_auto_search:ce,image_vision_model:currentVisionModel||null,
coding_mode:C,coding_target:C?{id:k.candidate_id,code:k.prompt_source?null:k.code,language:k.language||
"text",key:k.key||null,message_id:k.message_id||null,source:k.prompt_source?"prompt":"history",explicit:k.
explicit===!0}:null,coding_candidates:C?_.map(B=>({id:B.candidate_id,source:B.prompt_source?"prompt":
"history",prompt_index:B.prompt_source?B.prompt_index:null,code:B.prompt_source?null:B.code,language:B.
language||"text",explicit:B.explicit===!0})):[],batch_mode:i};e&&(A.turnstile_token=e);const j=get("\
thread-custom-instruction");j&&(A.thread_custom_instruction=j.value||""),activeGem?(A.system_prompt=
activeGem.instruction,A.enable_system_prompt=!0,A.gem_uuid=activeGem.uuid):A.gem_uuid=null,setSendBtnToStopMode();
const K="ai-"+Date.now(),Q=String(A.model||"").toLowerCase(),ne=!!A.enable_thinking||!!de&&de!=="non\
e",Te=Q.includes("gemini")||Q.includes("o1")||Q.includes("o3")||Q.includes("gpt-5")||Q.includes("rea\
soning")&&!Q.includes("non-reasoning"),ie=ne&&Te;let pe=buildPendingSkeletonHtml(A.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending\
-row fade-in"><div id="${K}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 roun\
ded-2xl rounded-tl-none shadow-md relative">${pe}</div></div>`),resumeChatAutoScroll();const W=get(K);
activeStreamingBubbleId=K,canvasModeEnabled&&resetCanvasPreviewPanel();let he=null;const pt=o(B=>!ie||
!W?null:((!he||!W.contains(he))&&(he=W.querySelector(".thought-content")),he||(W.insertAdjacentHTML(
"afterbegin",'<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="t\
oggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="\
thought-content collapsed" data-placeholder="1"></div></div>'),he=W.querySelector(".thought-content")),
he&&(he.setAttribute("data-placeholder","1"),he.textContent=B||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
he),"ensureThoughtPlaceholder");ie&&pt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const ct=currentThreadId,mt=nowPerfMs(),Ne=Date.now();let Ve=!1,
bt=!1,Lt=!1,wt=null,Mt=null,yt=null,Nt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const At=o((B,re)=>{if(!re||B==="status"&&Ve||B==="thought"&&bt||B==="content"&&Lt)return;const Ee=Math.
max(0,nowPerfMs()-mt);B==="status"?wt=Ee:B==="thought"?Mt=Ee:B==="content"&&(yt=Ee),reportFirstTokenLatency(
{latency_seconds:Ee/1e3,latency_ms:Ee,thread_id:Nt||currentThreadId,job_id:currentJobId,model:A.model,
first_event_type:B,client_sent_at_ms:Ne}),B==="status"?Ve=!0:B==="thought"?bt=!0:B==="content"&&(Lt=
!0)},"maybeReportFirstEventLatency"),ft=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let Rt=!1,Vt=!1,xt=null,vt=null,Kt=!1;try{A.thread_id&&activeGem&&(threadGemMap[A.thread_id]=
activeGem,pendingGemForNewThread=null);const B=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(A),signal:abortController.signal}),W);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${B.status}`),!B.ok){const ke=await B.json().catch(()=>({})),De=new Error(ke.error||
`HTTP ${B.status}`);throw De.serverCode=ke.code||null,De.serverModel=ke.model||A.model,De.acceptedJobId=
ke.job_id||null,De.acceptedThreadId=ke.thread_id||null,De}Rt=!0,window.ConnectionMonitor&&(Kt=!0,window.
ConnectionMonitor.operationStarted()),ft&&ft.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const re=o(()=>{if(!W)return;const ke=W.querySelector(
".content-area");if(ke&&ke.getAttribute("data-api-accepted")!=="1"&&(ke.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(W,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
ke.outerHTML=buildPendingSkeletonHtml(A.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const De=W.querySelector(".content-area");De&&De.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
W,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");re();
const Ee=B.body.getReader(),Re=new TextDecoder;let Ke="",tt="",rt="",qe=!0,je=null,Ze=null,We=null,gt=!1;
const Ft={};let Jt=0,Xt=!1;for(;!Xt;){const{done:ke,value:De}=await Ee.read();if(ke)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ft&&ft.setPhase("receiving"),Ke+=Re.decode(De,{stream:!0});
let Je=Ke.split(`
`);Ke=Je.pop();let dn=!1,un=!1;for(let _t of Je)if(_t.trim())try{const me=JSON.parse(_t);if(me.type===
"thread_id"){re();const xe=me.content!==null&&me.content!==void 0?String(me.content):me.content;xe&&
(Nt=xe,currentThreadId!==xe&&(currentThreadId=xe,history.pushState({},"","/c/"+xe)),activeGem&&(threadGemMap[xe]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(me.type==="job_\
id"){re(),currentJobId=me.content,i&&showToast("Batch\u767B\u9332","info");continue}if(me.type==="se\
arch_status"){me.content==="searching"&&!We?(W.insertAdjacentHTML("afterbegin",'<div class="search-b\
ox visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>'),We=W.querySelector(
".search-box")):me.content==="done"&&We&&(We.classList.remove("animate-pulse"),We.innerHTML='<i clas\
s="fas fa-check-circle text-green-400"></i> Search complete',setTimeout(()=>{We&&We.remove(),We=null},
2e3));continue}if(me.type==="mcp"){handleMcpStreamEvent(W,me.content||{});continue}if(me.type==="mcp\
_decision_request"){openMcpDecisionModal(me.content||{});continue}if(me.type==="status"){re();const xe=me.
content===null||me.content===void 0?"":String(me.content);if(At("status",!!xe),qe&&W){const Xe=xe||"\
\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(W,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ot=W.querySelector(".content-area");ot&&(ot.outerHTML=buildPendingSkeletonHtml(A.model,Xe),updatePendingSkeletonStatus(
W,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
ie&&pt(xe||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(qe){beginPendingToStreamTransition(
W);const xe=W.querySelector(".content-area");xe&&(xe.innerHTML=""),qe=!1}if(me.type==="coding_diff")
appendCodingLiveDiff(W,me.content||{}),At("content",!0);else if(me.type==="thought"){if(je||(je=W.querySelector(
".thought-content")),rt+=me.content,At("thought",!!me.content),!je){const xe='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';We?We.insertAdjacentHTML(
"afterend",xe):W.insertAdjacentHTML("afterbegin",xe),je=W.querySelector(".thought-content")}if(je&&je.
getAttribute("data-placeholder")==="1"){if(je.textContent="",je.removeAttribute("data-placeholder"),
je){const xe=je.parentElement.querySelector(".thought-header");xe&&xe.classList.remove("thinking-shi\
mmer")}rt=me.content}je.classList.remove("collapsed"),un=!0}else if(me.type==="image_analysis"){const xe=me.
content===null||me.content===void 0?"":String(me.content);if(!W)continue;let Xe=W.querySelector(".im\
age-analysis-box");if(!Xe){const nt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';We?We.insertAdjacentHTML("afterend",nt):W.insertAdjacentHTML("afterbegin",nt),Xe=W.querySelector(
".image-analysis-box")}const ot=Xe.querySelector(".image-analysis-text");ot&&(ot.textContent=xe)}else if(me.
type==="python"){const xe=me.content||{},Xe=xe.id||`py_${Date.now()}`;if(!Ft[Xe]){const nt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Xe}" data-collapsed="true" data-code-key="${Xe}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;We?We.insertAdjacentHTML("afterend",nt):W.insertAdjacentHTML(
"afterbegin",nt),Ft[Xe]=W.querySelector(`[data-py-id="${Xe}"]`)}const ot=Ft[Xe];if(ot){if(xe.code!==
void 0){const nt=xe.code==null?"":String(xe.code),ht=ot.querySelector(".python-code");ht&&(ht.textContent=
nt,ht.removeAttribute("data-highlighted"),queueHighlight(ot,nt));const St=ot.querySelector('.copy-bt\
n[data-copy="code"]');St&&St.setAttribute("data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}if(xe.
output!==void 0){const nt=xe.output==null?"":String(xe.output),ht=ot.querySelector(".python-output");
ht&&(ht.textContent=nt);const St=ot.querySelector('.copy-btn[data-copy="output"]');St&&St.setAttribute(
"data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}}}else if(me.type==="content"){const xe=me.content===
null||me.content===void 0?"":String(me.content);tt+=xe,/[`~]/.test(xe)&&activateDeferredCodingModeFromStream(
tt),Ze||(Ze=W.querySelector(".content-area")||document.createElement("div"),Ze.className="prose pros\
e-invert text-sm break-words",W.contains(Ze)||W.appendChild(Ze)),dn=!0,At("content",!!xe)}else if(me.
type==="error"){gt=!0,Xt=!0,W.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(me.content)),showToast(
me.content||"Unknown error","error",!0);break}}catch{}if(un&&je&&(je.textContent=rt,userAutoScroll&&
(je.scrollTop=je.scrollHeight)),dn&&Ze){const _t=Date.now();if(_t-Jt>100){const me=snapshotCodeCollapse(
Ze);renderAiMarkdownInto(Ze,tt,{incrementalMath:!0}),applyCodeCollapse(Ze,me,!0),Jt=_t}}scrollToBottom()}
if(ft&&ft(),Ze){const ke=snapshotCodeCollapse(Ze);renderAiMarkdownInto(Ze,tt,{incrementalMath:!0}),applyCodeCollapse(
Ze,ke,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),W)if(queueHighlight(W,tt),enableLatencyMetrics){
const ke=nowPerfMs()-mt;reportFirstTokenLatency({is_total:!0,latency_seconds:ke/1e3,latency_ms:ke,thread_id:Nt||
currentThreadId,job_id:currentJobId,model:A.model,client_sent_at_ms:Ne,client_done_at_ms:Date.now()});
let De='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',Je=null;wt!==null&&(Je=wt),Mt!==null&&(Je===null||Mt<Je)&&(Je=
Mt),yt!==null&&(Je===null||yt<Je)&&(Je=yt),Je!==null&&(De+=`<div>Initial: ${(Je/1e3).toFixed(2)}s</d\
iv>`),yt!==null&&yt!==Je&&(De+=`<div>Content: ${(yt/1e3).toFixed(2)}s</div>`),De+=`<div class="font-\
bold text-gray-300">Total: ${(ke/1e3).toFixed(2)}s</div>`,currentJobId&&(De+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),De+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,De+="</div>",W.insertAdjacentHTML("beforeend",De)}else W.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),W&&W.querySelectorAll(".thought-co\
ntent").forEach(De=>De.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!F}),!gt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(ke=>ke.json()).then(ke=>{ke.title&&
(document.title=ke.title+" - AI Chat",setCurrentChatHeaderTitle(ke.title),loadThreads())}):loadThreads(
!1)}catch(B){let re=!1;const Ee=B.name==="AbortError"&&isManualStopAbortForThread(ct);if(B.name==="A\
bortError"&&!Ee&&(re=await syncThreadAfterAbortedStream(ct,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${B.message}`),!Rt){we&&we.remove();const Re=W&&W.closest(
".fade-in");Re&&Re.remove(),delete messageStore[ee],delete messageMeta[ee]}if(B.serverCode==="reques\
t_already_accepted"&&B.acceptedJobId&&B.acceptedThreadId)Rt=!0,xt={job_id:B.acceptedJobId,thread_id:String(
B.acceptedThreadId),model:A.model},get("prompt-input").value="",get("prompt-input").style.height="au\
to",resetUploadState(),clearQuote();else if(Rt&&!Ee)vt={job_id:normalizeJobIdForUi(currentJobId),thread_id:currentThreadId!=
null?String(currentThreadId):null,model:A.model},window.ConnectionMonitor.setUnavailable("offline"),
showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(B.serverCode==="turnstile_required"){const Re=await getTurnstileToken();Re?(await verifyTurnstileOnServer(
Re,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(B.serverCode==="api_key_missing"){const Re=B.serverModel||A.model,Ke=await showApiKeyRequiredModalAsync(
Re);Ke==="set"?Vt=!0:Ke==="switch"?showModal("model-modal"):showToast(B.message||`${getModelNameById(
Re)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(B.
name!=="AbortError"){const Re="Connection Error: "+B.message;showToast(Re,"error",!0)}F&&!re&&V()}finally{
Kt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),ft&&ft(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===K&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(xt){const B=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=xt.thread_id,(B!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(xt,currentThreadId)}
if(vt&&vt.thread_id)return reconnectPendingStreamUntilAvailable(vt,vt.thread_id);if(Vt)return sendMessage()}
o(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?
String(e.model):"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)
return;if(activeStreamingBubbleId=n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-\
area.skeleton-pending")){const V=a.querySelector(".content-area");V?V.outerHTML=buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const r=currentThreadId,
l=i.toLowerCase(),c=l.includes("gemini")||l.includes("o1")||l.includes("o3")||l.includes("gpt-5")||l.
includes("reasoning")&&!l.includes("non-reasoning");let u=null;const f=o(V=>!c||!a?null:((!u||!a.contains(
u))&&(u=a.querySelector(".thought-content")),u||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),u=a.querySelector(".thought-content")),u&&(u.setAttribute("data-pla\
ceholder","1"),u.textContent=V||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),u),
"ensureThoughtPlaceholder");c&&f("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let b="",y="",w="",v=!0,k=null,_=null,C=null,L=!1;const $={};let F=0,Y=!1;const X=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Ae=!1,R=!1;try{const V=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!V.ok)throw new Error(`Resume failed (${V.status})`);window.ConnectionMonitor&&
(R=!0,window.ConnectionMonitor.operationStarted()),X&&X.setPhase("waiting");const ee=V.body.getReader(),
we=new TextDecoder;for(;!Y;){const{done:ce,value:Se}=await ee.read();if(ce)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),X&&X.setPhase("receiving"),b+=we.decode(Se,{stream:!0});let Pe=b.
split(`
`);b=Pe.pop();let Ce=!1,ge=!1;for(let de of Pe)if(de.trim())try{const H=JSON.parse(de);if(H.type==="\
job_id"){currentJobId=H.content||t;continue}if(H.type==="search_status"){H.content==="searching"&&!C?
(a.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),C=a.querySelector(".search-box")):H.content==="done"&&C&&(C.
classList.remove("animate-pulse"),C.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{C&&C.remove(),C=null},2e3));continue}if(H.type==="mcp"){handleMcpStreamEvent(
a,H.content||{});continue}if(H.type==="mcp_decision_request"){openMcpDecisionModal(H.content||{});continue}
if(H.type==="status"){const A=H.content===null||H.content===void 0?"":String(H.content);if(v&&a){const j=A||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(a,j,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const K=a.querySelector(".content-area");K&&(K.outerHTML=buildPendingSkeletonHtml(i,j),updatePendingSkeletonStatus(
a,j,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
c&&f(A||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(v){beginPendingToStreamTransition(
a);const A=a.querySelector(".content-area");A&&(A.innerHTML=""),v=!1}if(H.type==="coding_diff")appendCodingLiveDiff(
a,H.content||{});else if(H.type==="thought"){if(k||(k=a.querySelector(".thought-content")),w+=H.content,
!k){const A='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';C?C.insertAdjacentHTML("afterend",A):a.insertAdjacentHTML("afterbegin",A),k=a.querySelector(
".thought-content")}if(k&&k.getAttribute("data-placeholder")==="1"){if(k.textContent="",k.removeAttribute(
"data-placeholder"),k){const A=k.parentElement.querySelector(".thought-header");A&&A.classList.remove(
"thinking-shimmer")}w=H.content}k.classList.remove("collapsed"),ge=!0}else if(H.type==="image_analys\
is"){const A=H.content===null||H.content===void 0?"":String(H.content);if(!a)continue;let j=a.querySelector(
".image-analysis-box");if(!j){const Q='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';C?C.insertAdjacentHTML("afterend",Q):a.insertAdjacentHTML("afterbegin",Q),j=a.querySelector(
".image-analysis-box")}const K=j.querySelector(".image-analysis-text");K&&(K.textContent=A)}else if(H.
type==="python"){const A=H.content||{},j=A.id||`py_${Date.now()}`;if(!$[j]){const Q=`<div class="cod\
e-wrapper python-box collapsed" data-py-id="${j}" data-collapsed="true" data-code-key="${j}"><div cl\
ass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><di\
v class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B">\
<i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" t\
itle="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" dat\
a-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></b\
utton></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code<\
/div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-secti\
on"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output"><\
/code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",Q):a.insertAdjacentHTML("afterbegi\
n",Q),$[j]=a.querySelector(`[data-py-id="${j}"]`)}const K=$[j];if(K){if(A.code!==void 0){const Q=A.code==
null?"":String(A.code),ne=K.querySelector(".python-code");ne&&(ne.textContent=Q,ne.removeAttribute("\
data-highlighted"),queueHighlight(K,Q));const Te=K.querySelector('.copy-btn[data-copy="code"]');Te&&
Te.setAttribute("data-code",encodeURIComponent(Q).replace(/'/g,"%27"))}if(A.output!==void 0){const Q=A.
output==null?"":String(A.output),ne=K.querySelector(".python-output");ne&&(ne.textContent=Q);const Te=K.
querySelector('.copy-btn[data-copy="output"]');Te&&Te.setAttribute("data-code",encodeURIComponent(Q).
replace(/'/g,"%27"))}}}else if(H.type==="content"){const A=H.content===null||H.content===void 0?"":String(
H.content);y+=A,/[`~]/.test(A)&&activateDeferredCodingModeFromStream(y),_||(_=a.querySelector(".cont\
ent-area")||document.createElement("div"),_.className="prose prose-invert text-sm break-words",a.contains(
_)||a.appendChild(_)),Ce=!0}else if(H.type==="error"){L=!0,Y=!0,a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
H.content)),showToast(H.content||"Unknown error","error",!0);break}}catch{}if(ge&&k&&(k.textContent=
w,userAutoScroll&&(k.scrollTop=k.scrollHeight)),Ce&&_){const de=Date.now();if(de-F>100){const H=snapshotCodeCollapse(
_);renderAiMarkdownInto(_,y,{incrementalMath:!0}),applyCodeCollapse(_,H,!0),F=de}}scrollToBottom()}if(X&&
X(),_){const ce=snapshotCodeCollapse(_);renderAiMarkdownInto(_,y,{incrementalMath:!0}),applyCodeCollapse(
_,ce,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,y),a&&a.querySelectorAll(".thought-content").
forEach(Se=>Se.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(V){const ee=V.name==="AbortError"&&isManualStopAbortForThread(r);V.name==="Abo\
rtError"&&!ee&&await syncThreadAfterAbortedStream(r,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
ee||(Ae=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{R&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),X&&X(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Ae)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i},r)}o(resumePendingStream,"resumePendingStream");function updateThreadHighlighting(){
const e=get("thread-list");if(!e)return;e.querySelectorAll("[data-thread-id]").forEach(n=>{n.dataset.
threadId===String(currentThreadId)?n.classList.add("bg-gray-700/60","border-l-2","border-blue-500"):
n.classList.remove("bg-gray-700/60","border-l-2","border-blue-500")})}o(updateThreadHighlighting,"up\
dateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){snapshotSidebarHistory("\
loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory("loadThreads-s\
tart append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?t.value:"";
if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");return}const a=await(await apiFetch(
`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&page=${threadPage}`)).json(),r=get("th\
read-list");if(!r)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-r\
eplace-settings-open");return}const c=a&&Array.isArray(a.threads)?a.threads.length:-1,u=r.querySelectorAll(
"[data-thread-id]").length;if(c===0&&u>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const u=String(
c.id),f=document.createElement("div"),b=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",v=u===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${v}`,f.dataset.threadId=u,f.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${b} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${u}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(c.title||"No Title")}</spa\
n>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${u}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${u}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,f.onclick=k=>{k.target.closest("button")||k.target.closest("[dat\
a-thread-actions]")||loadMessages(u)},l?r.insertBefore(f,l):r.appendChild(f)}),hasMoreThreads=!!a.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+a.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error("Faile\
d to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,r=88,l=52,c=.5,u=8;let f=0,b=!1,y=0,w=null;
const v=o(()=>get(i),"indicatorEl"),k=o(()=>{const L=v();return L?L.querySelector(".ptr-pull-label"):
null},"labelEl"),_=o(L=>{const $=v();if(!$)return;$.style.height=Math.min(L,r)+"px",$.classList.toggle(
"active",L>2),$.classList.toggle("pull-ready",L>=a);const F=k();F&&(F.textContent=L>=a?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),C=o(()=>{const L=v();L&&(L.style.height=
"0px",L.classList.remove("active","pull-ready","refreshing"),L.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",L=>{if(w){b=!1;return}if(n.scrollTop>0){b=!1;return}const $=L.
touches[0];$&&(f=$.clientY,y=0,b=!0)},{passive:!0}),n.addEventListener("touchmove",L=>{if(!b||w)return;
if(n.scrollTop>0){b=!1;return}const $=L.touches[0];if(!$)return;const F=$.clientY-f;if(F<=0){y>0&&(y=
0,_(0)),b=!1;return}const Y=v();Y&&!Y.classList.contains("dragging")&&Y.classList.add("dragging"),y=
Math.min(F*c,r),_(y),F>=u&&L.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!b||
(b=!1,w))return;const L=v();L&&L.classList.remove("dragging");const $=y>=a;if(y=0,!$){C();return}let F;
try{F=t()}catch{F=null}const Y=v();if(Y){Y.classList.add("refreshing"),Y.style.height=l+"px";const X=Y.
querySelector(".ptr-pull-label");X&&(X.textContent="\u66F4\u65B0\u4E2D...")}F&&typeof F.then=="funct\
ion"?(w=F,F.catch(()=>{}).finally(()=>{w=null,C()})):(w=Promise.resolve(),setTimeout(()=>{w=null,C()},
400))}),n.addEventListener("touchcancel",()=>{b=!1,y=0,C()})}o(initPullToRefresh,"initPullToRefresh");
const initThreadPullToRefresh=o(()=>initPullToRefresh("thread-list",()=>loadThreads(!1)),"initThread\
PullToRefresh"),initGemPullToRefresh=o(()=>initPullToRefresh("gem-list",()=>loadGems()),"initGemPull\
ToRefresh"),initPullToRefreshAll=o(()=>{initThreadPullToRefresh(),initGemPullToRefresh()},"initPullT\
oRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;const mcpCardIdSelector=o(e=>"mcp_\
card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),mcpEscHtml=o(e=>String(e==null?
"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]),"mcpE\
scHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||"MCP")} / ${mcpEscHtml(e.tool_name||
e.internal_name||"")}`}o(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){if(!e)return null;
let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("div"),t.className=
"mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),e.appendChild(
t)),t}o(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){if(!e||!t||!t.
type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(e):null;if(n&&
!i)return;const a=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.querySelector(
'[data-mcp-card="'+a+'"]'))return;const r=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${a}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",r);return}if(t.type==="result"){let r=i.querySelector('[dat\
a-mcp-card="'+a+'"]');const l=t.summary||"";if(r)r.classList.remove("mcp-running"),r.classList.add("\
mcp-done"),r.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const c=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${a}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",c),r=i.querySelector('[data-mcp-card="'+a+'"]')}if(l){const c=document.
createElement("div");c.className="mcp-box-note",c.textContent=l.split(`
`)[0].slice(0,220),r&&r.appendChild(c)}return}if(t.type==="error"){let r=i.querySelector('[data-mcp-\
card="'+a+'"]');const l=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(r)r.classList.remove("mcp-running"),r.classList.add("mcp-error"),r.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const u=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",u),r=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
r&&r.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const r=get("mcp-decision-modal");if(r&&!r.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}o(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const u=JSON.parse(c);c=JSON.stringify(u,null,2)}catch{}a.textContent=c}const r=get(
"mcp-decision-allow"),l=get("mcp-decision-deny");r&&(r.onclick=()=>submitMcpDecision("allow")),l&&(l.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}o(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}o(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const a=e[0],r=a.
thread_id;n.textContent=e.length===1?`${a.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),r&&await loadMessages(r)});const l=get(
"batch-notification-close");l&&(l.onclick=()=>t.classList.add("hidden"))}o(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({}));(t.active||[]).some(a=>currentThreadId&&String(a.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const i=t.
completed||[];i.length&&(showGeminiBatchCompletionBanner(i),i.some(a=>String(a.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}o(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);let chatTransitionSequence=0,chatTransitionTimer=null;const CHAT_TRANSITION_DURATION_MS=460;function playChatTransition(e){
const t=get("chat-transition-veil");if(!t)return;const n=++chatTransitionSequence;if(chatTransitionTimer&&
(clearTimeout(chatTransitionTimer),chatTransitionTimer=null),window.matchMedia&&window.matchMedia("(\
prefers-reduced-motion: reduce)").matches){t.classList.remove("is-active"),t.removeAttribute("data-t\
ransition-kind");return}t.dataset.transitionKind=e||"history",t.classList.remove("is-active"),t.offsetWidth,
t.classList.add("is-active"),chatTransitionTimer=setTimeout(()=>{n===chatTransitionSequence&&(t.classList.
remove("is-active"),chatTransitionTimer=null)},CHAT_TRANSITION_DURATION_MS)}o(playChatTransition,"pl\
ayChatTransition");async function toggleBookmark(e,t){e&&e.stopPropagation(),await apiFetch(`/api/th\
reads/${t}/bookmark`,{method:"POST"}),loadThreads()}o(toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){
const n=++threadLoadSequence;window.closeHistoryModal&&window.closeHistoryModal();const i=!!t.preserveDraft,
a=!!t.silent;a||resumeChatAutoScroll({scroll:!1});const r=a?snapshotCodeCollapseByMessage(get("chat-\
container")):null;let l="",c="",u=[];if(i){const f=get("prompt-input");l=f?f.value:"",c=f?f.style.height:
"",u=currentImageUrls?currentImageUrls.slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();
a||playChatTransition("history"),currentThreadId=e!=null?String(e):e,t.skipHistory||history.pushState(
{},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),get("welcome-scree\
n").classList.add("hidden"),a||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());try{
const f=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);f.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const b=await apiFetch(f.toString());if(!b.
ok)throw new Error(`thread request failed (${b.status})`);const y=await b.json();if(!y||!Array.isArray(
y.messages))throw new Error("invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(
y&&y.title),allMessages=y.messages,threadHasOlderMessages=!!y.has_older_messages,oldestLoadedMessageId=
y.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const w=(allMessages||[]).filter(k=>k.
role==="user"&&k.content).map(k=>k.content);if(promptHistory=[...new Set(w.slice().reverse())],historyIndex=
-1,tempPrompt="",currentThreadPending=y.pending_job||null,setTemporaryChatUiState(!!(y&&y.is_temporary)),
applyTemporaryChatRuntimeMeta(y||{}),ensureTemporaryChatHeartbeat(!0),get("thread-custom-instruction")&&
(get("thread-custom-instruction").value=y.custom_instruction||""),get("enable-prompt-cache")&&(get("\
enable-prompt-cache").checked=!!y.enable_prompt_caching,updatePromptCacheUi()),y.last_model&&selectModelById(
y.last_model),y.last_gem_uuid&&loadedGems.length>0){const k=loadedGems.find(_=>_.uuid===y.last_gem_uuid);
k&&(threadGemMap[currentThreadId]=k,applyActiveGem(k))}const v=t.forceLatestLeaf?null:localStorage.getItem(
`fixed_branch_${currentThreadId}`);if(v&&allMessages.find(k=>String(k.id)===String(v))?currentLeafId=
v:allMessages.length>0?currentLeafId=allMessages[allMessages.length-1].id:currentLeafId=null,renderThreadTree(
a?{silent:a,keepScroll:a}:{silent:a,keepScroll:a,animate:!0}),a&&r?applyCodeCollapseByMessage(get("c\
hat-container"),r,!0):a||applyCodeCollapseByMessage(get("chat-container"),null,!0),currentThreadPending&&
!a&&!isPendingJobSuppressed(currentThreadPending.job_id)&&resumePendingStream(currentThreadPending),
i){const k=get("prompt-input");k&&(k.value=l||"",c?k.style.height=c:k.style.height="auto"),currentImageUrls=
u,currentImageUrls&&currentImageUrls.length?(get("file-preview").classList.remove("hidden"),get("fil\
e-name").innerText=`${currentImageUrls.length} files ready`):get("file-preview").classList.add("hidd\
en"),schedulePromptTokenEstimate(!0)}if(i||schedulePromptTokenEstimate(!0),window.innerWidth<768&&get(
"overlay").click(),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
return!0}catch(f){return n!==threadLoadSequence||(console.error("Failed to load chat thread:",f),a||
showChatLoadError(e),a||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)),!1}}o(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const r=await(await apiFetch(i.toString())).json(),l=Array.isArray(r.messages)?r.messages:[];
if(l.length){const c=new Set(allMessages.map(f=>f.id)),u=l.filter(f=>!c.has(f.id));u.length&&(allMessages=
u.concat(allMessages))}if(threadHasOlderMessages=!!r.has_older_messages,oldestLoadedMessageId=r.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;let r=null;if(i&&(r=a.scrollTop),a.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const l={};allMessages.
forEach(v=>{l[v.id]=v,v.childrenIds=[]}),allMessages.forEach(v=>{v.parent_id&&l[v.parent_id]&&l[v.parent_id].
childrenIds.push(v.id)}),(!currentLeafId||!l[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const c=[];let u=l[currentLeafId];for(;u;)c.unshift(u),u=l[u.parent_id];const f=buildTokenTotals(
c),b=buildTokenTotals(allMessages),y=document.createDocumentFragment();if(threadHasOlderMessages){const v=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
k=loadingOlderMessages?"disabled":"",_=document.createElement("div");_.className="mb-3 text-center",
_.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${k}><i class="fas fa-clock-rotate-left mr-1"></i>${v}</button>`,y.appendChild(_)}
c.forEach(v=>{const k=v.parent_id?l[v.parent_id]:null,_=k?k.childrenIds:allMessages.filter(L=>!L.parent_id).
map(L=>L.id),C=_.length>1?{current:_.indexOf(v.id)+1,total:_.length,siblings:_}:null;renderMessage(v.
id,v.role,v.content,v.image_url,v.thought_data,v.model,C,n,v.quote_text,v.tokens,v.tokens_in,v.tokens_out,
v.is_encrypted,v.tokens_content,v.tokens_thought,y,!1,v.parent_id,v.gem_name,v.batch_job)});const w=currentThreadPending;
if(w&&!isPendingJobSuppressed(w.job_id)){const v=w.message_id,k=new Set(c.map(L=>L.id)),_=c.length?c[c.
length-1]:null;if(v&&k.has(v)&&currentLeafId===v||!v&&_&&_.role==="user"){const L=w.job_id?`pending-${w.
job_id}`:null;renderPendingMessage(y,n,!1,L,w.model||null)}}if(a.appendChild(y),updateTotalTokenBar(
f.tokens_total,f,b),currentParentId=currentLeafId,i&&r!==null?restoreThreadTreeScroll(a,r):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(a),c.length){
const v=c[c.length-1]&&c[c.length-1].content;queueMathTypeset(a,v)}}o(renderThreadTree,"renderThread\
Tree");function restoreThreadTreeScroll(e,t){if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&
!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=
e.scrollTop,syncScrollToBottomButton()}o(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){
currentLeafId=e;const t={};allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(
i=>{i.parent_id&&t[i.parent_id]&&t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=
allMessages.length>0?allMessages[allMessages.length-1].id:null,renderThreadTree({animate:!0});return}
for(;t[n]&&t[n].childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree(
{animate:!0})}o(switchVersion,"switchVersion");async function loadGems(){try{const t=await(await apiFetch(
CHAT_CONFIG.urls.handleGems)).json();loadedGems=t;const n=get("gem-list");if(!n)return;n.innerHTML='\
<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-do\
wn ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-l\
abel"></span></div>',Array.isArray(t)&&t.forEach(i=>{const a=document.createElement("div");a.className=
"gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between it\
ems-center group",a.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa\
-gem text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><div class="flex it\
ems-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group\
-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,'${i.uuid}')"><i class="fas fa-p\
encil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:o\
pacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${i.uuid}')"><i class\
="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=r=>{r.target.closest("button")||activateGem(
i)},n.appendChild(a)})}catch(e){console.error("Failed to load gems:",e)}}o(loadGems,"loadGems");function setGemDefaultModelSelect(e){
const t=get("gem-default-model");if(!t)return;t.innerHTML="";const n=document.createElement("option");
n.value="",n.textContent="Use current model",t.appendChild(n),MODELS.forEach(a=>{const r=(a.items||[]).
filter(c=>!c.deprecated);if(!r.length)return;const l=document.createElement("optgroup");l.label=a.category,
r.forEach(c=>{const u=document.createElement("option");u.value=c.id,u.textContent=c.name,l.appendChild(
u)}),t.appendChild(l)});const i=e||"";if(i&&!Array.from(t.options).some(a=>a.value===i)){const a=document.
createElement("option");a.value=i,a.textContent=MODEL_NAME_BY_ID[i]||i,t.appendChild(a)}t.value=i}o(
setGemDefaultModelSelect,"setGemDefaultModelSelect");async function openEditGemModal(e,t){e.stopPropagation(),
editingGemUuid=t;try{const i=await(await apiFetch(`/api/gems/${t}`)).json();get("gem-name").value=i.
name,get("gem-desc").value=i.description||"",get("gem-inst").value=i.instruction,setGemDefaultModelSelect(
i.default_model),renderGemFixedPromptsForEdit(i.fixed_prompts),get("gem-modal-title").innerHTML='<i \
class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-btn").innerText="Save Changes",showModal(
"gem-modal"),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")}catch{showToast(
"Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}o(openEditGemModal,"o\
penEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.urls.handleGems,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,instruction:t})}),loadGems()}
o(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get("fixed-prompts-bar");
if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),get("active-gem-n\
ame").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),t){t.innerHTML=
"";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}n.length>0?(t.
classList.remove("hidden"),n.forEach((i,a)=>{const r=document.createElement("button");r.className="f\
ixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover:bg-gray-600 t\
ext-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex items-center",r.style.
animationDelay=`${a*40}ms`,r.textContent=String(i.name||""),r.onclick=()=>{const l=get("prompt-input");
l&&(l.value=i.content,l.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(r)})):t.classList.
add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML="",t.classList.
add("hidden"));get("sys-prompt-option").style.opacity="1"}o(applyActiveGem,"applyActiveGem");function syncActiveGemForThread(e){
const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(t)}o(syncActiveGemForThread,"syncActi\
veGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_gem_uuid:t,thread_id:e})})}catch{}}
o(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){currentThreadId?(threadGemMap[currentThreadId]=
e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadGemUuid(currentThreadId,e?e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(
e),allMessages&&allMessages.length>0&&startNewChat({preserveGem:!0}))}o(activateGem,"activateGem");function clearActiveGem(){
currentThreadId&&(delete threadGemMap[currentThreadId],saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=
null,applyActiveGem(null)}o(clearActiveGem,"clearActiveGem");function addGemFixedPromptRow(e="",t=""){
const n=get("gem-fixed-prompts-container");if(!n)return;const i=document.createElement("div");i.className=
"flex gap-2 items-start gem-fixed-prompt-row ui-enter",i.innerHTML=`
                <input type="text" class="gem-fp-name bg-gray-900 border border-gray-600 rounded p-1\
.5 text-white text-[10px] w-24" placeholder="\u540D\u524D" value="${escapeHtml(e)}" autocomplete="of\
f" spellcheck="false">
                <textarea class="gem-fp-content flex-1 bg-gray-900 border border-gray-600 rounded p-\
1.5 text-white text-[10px] h-9 resize-none" placeholder="\u30D7\u30ED\u30F3\u30D7\u30C8\u5185\u5BB9" spellcheck="false">${escapeHtml(
t)}</textarea>
                <button type="button" class="text-gray-500 hover:text-red-400 p-1.5" onclick="this.p\
arentElement.remove()"><i class="fas fa-times"></i></button>
            `,n.appendChild(i)}o(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),t=[];return e.forEach(n=>{const i=n.querySelector(
".gem-fp-name").value.trim(),a=n.querySelector(".gem-fp-content").value.trim();i&&a&&t.push({name:i,
content:a})}),t.length>0?JSON.stringify(t):null}o(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const t=get("gem-fixed-prompts-container");if(t){t.innerHTML="";try{e&&JSON.parse(e).forEach(i=>addGemFixedPromptRow(
i.name,i.content))}catch{}}}o(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}o(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}o(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=[
"sidebar-chat-title","mobile-chat-title"],a=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],r=["sidebar-chat-ttl","mobile-chat-ttl"];i.forEach(l=>{const c=get(l);c&&(c.textContent=e)}),
a.forEach(l=>{const c=get(l);c&&c.classList.toggle("hidden",!n)}),r.forEach(l=>{const c=get(l);c&&(n&&
t?(c.textContent=t,c.classList.remove("hidden")):(c.textContent="",c.classList.add("hidden")))})}o(updateCurrentChatHeaderUi,
"updateCurrentChatHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="str\
ing"?e:null,updateCurrentChatHeaderUi()}o(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){
tempChatExpiresAtMs=null,updateCurrentChatHeaderUi()}o(resetTemporaryChatExpiresAt,"resetTemporaryCh\
atExpiresAt");function applyTemporaryChatRuntimeMeta(e){if(!e||typeof e!="object")return;Object.prototype.
hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(e.timeout_seconds);let t=null;
const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.floor(n*1e3);else{const i=Number(
e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=
t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=null),updateCurrentChatHeaderUi()}
o(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
o(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,t=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let n=Number(e);return Number.isFinite(n)||(n=Number(t)),Number.isFinite(n)||(n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
n=Math.trunc(n),n<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(n=TEMP_CHAT_TIMEOUT_MIN_SECONDS),n>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(n=TEMP_CHAT_TIMEOUT_MAX_SECONDS),n}o(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
n=get("temporary-chat-welcome-desc");n&&(n.textContent=t);const i=get("temporary-chat-container");i&&
(i.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
o(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const t=get(
"set-temp-chat-timeout-seconds");t&&(t.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}o(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,t))}o(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const t=get("enable-temporary-c\
hat");t&&t.checked!==temporaryChatEnabled&&(t.checked=temporaryChatEnabled);const n=get("welcome-def\
ault-content");n&&n.classList.toggle("hidden",temporaryChatEnabled);const i=get("welcome-temporary-c\
ontent");i&&i.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}o(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}o(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}o(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const t=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
n=await t.json().catch(()=>({}));t.ok&&n&&applyTemporaryChatRuntimeMeta(n),t.ok&&n&&n.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}o(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const t=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==t)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=t,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}o(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const t=!!e;if(setTemporaryChatUiState(
t),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const n=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:t})}),i=await n.json().catch(()=>({}));if(!n.ok)throw new Error(i&&i.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(i&&i.is_temporary)),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}o(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
if(playChatTransition("new"),threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),
resetUploadState(),stopTemporaryChatHeartbeat(),setTemporaryChatUiState(!1),currentThreadTitle=null,
tempChatExpiresAtMs=null,currentThreadId=null,allMessages=[],promptHistory=[],historyIndex=-1,tempPrompt=
"",threadHasOlderMessages=!1,oldestLoadedMessageId=null,loadingOlderMessages=!1,currentLeafId=null,currentParentId=
null,currentThreadPending=null,updateTotalTokenBar(0),typeof window.__refreshAdminThreadEncState=="f\
unction")try{window.__refreshAdminThreadEncState()}catch{}e.skipHistory||history.pushState({},"","/"),
get("chat-container").innerHTML="",get("welcome-screen").classList.remove("hidden"),updateCurrentChatHeaderUi(),
get("thread-custom-instruction")&&(get("thread-custom-instruction").value=""),get("enable-prompt-cac\
he")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(
activeGem):applyActiveGem(null),loadThreads(),window.innerWidth<768&&get("overlay").click()}o(startNewChat,
"startNewChat");let threadModalLoadSeq=0;window.openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(
CHAT_CONFIG.urls.handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:temporaryChatEnabled})})).json();currentThreadId=i.id!==null&&i.id!==void 0?
String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.is_temporary)),setCurrentChatHeaderTitle(i&&i.title),
applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),history.pushState({},"","/c/"+
i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,t=String(currentThreadId);modalThreadId=t,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[n,i]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${t}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==t)return;if(n.ok){const a=await n.
json(),r=get("thread-app-global-sys-prompt-preview");r&&(r.value=a.global_system_prompt_effective||"");
const l=get("thread-app-global-sys-prompt-preview-status");l&&(a.global_system_prompt_enabled===!1?l.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":a.global_system_prompt_uses_time_fallback?
l.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
l.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=a.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=a.system_prompt_enabled!==
!1),get("thread-apply-global-sys-prompt")&&(get("thread-apply-global-sys-prompt").checked=a.apply_global_system_prompt!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=a.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",a.auto_system_prompt_notices_config||{})}if(i.ok){const a=await i.json();if(e!==threadModalLoadSeq||
modalThreadId!==t)return;const r=get("thread-custom-instruction");r&&(r.value=a.custom_instruction||
"");const l=get("thread-include-global-instruction");l&&(l.checked=a.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),a=i?i.value:"",r=get("thread-include-global-instruction"),l=r?r.checked:
!0,c=get("thread-global-sys-prompt"),u=get("thread-global-sys-prompt-enabled");let f=null;try{f=c||u?
{system_prompt:c?c.value:"",system_prompt_enabled:u?u.checked:!0,apply_global_system_prompt:get("thr\
ead-apply-global-sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(b){sendClientDebugLog("error","Payload c\
onstruction failed: "+b.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const b=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+b.status);let y=!0;if(f){sendClientDebugLog(
"info","Starting POST request for user settings");const w=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});y=w.ok,w.ok&&window.
applySavedUserSystemPromptSettings(f),sendClientDebugLog("info","POST request finished, status: "+w.
status)}b.ok&&y?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\u308C\u307E\u3057\u305F","s\
uccess")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}catch(b){sendClientDebugLog(
"error","Save failed with error: "+b.message),showToast("\u30A8\u30E9\u30FC: "+b.message,"error",!0)}finally{
t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=()=>{syncCompressionSettingsUi(),
showModal("compression-modal"),location.pathname!=="/compression"&&history.pushState({modal:"compres\
sion"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal("compression-modal"),!e&&
location.pathname==="/compression"&&history.back()},get("save-compression-settings-btn").onclick=()=>{
const e=get("compression-max-size").value,t=get("compression-max-dim").value,n=get("compression-outp\
ut-type").value,i=get("compression-format-only").checked;setCompressionSettings(e,t,n,i);const a=o((l,c)=>{
get(l)&&get(c)&&(get(c).value=get(l).value)},"syncBack");a("modal-gpt-image-size","gpt-image-size"),
a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-format","gpt-image-format"),a("m\
odal-gpt-image-compression","gpt-image-compression"),a("modal-gemini-image-aspect","gemini-image-asp\
ect"),a("modal-gemini-image-size","gemini-image-size"),a("modal-grok-image-aspect","grok-image-aspec\
t"),a("modal-grok-image-resolution","grok-image-resolution"),a("modal-grok-image-quality","grok-imag\
e-quality"),IDEOGRAM_IMAGE_FIELDS.forEach(l=>a(`modal-ideogram-image-${l}`,`ideogram-image-${l}`)),a(
"modal-ocr-table-format","ocr-table-format"),a("modal-ocr-pages","ocr-pages");const r=o((l,c)=>{get(
l)&&get(c)&&(get(c).checked=get(l).checked)},"syncBackChk");r("modal-ocr-extract-header","ocr-extrac\
t-header"),r("modal-ocr-extract-footer","ocr-extract-footer"),r("modal-ocr-include-blocks","ocr-incl\
ude-blocks"),r("modal-ocr-include-images","ocr-include-images"),window.closeCompressionModal(),showToast(
"\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","success")};async function deleteGem(e,t){e.
stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleGemItem.replace("0",t),
{method:"DELETE"}),loadGems())}o(deleteGem,"deleteGem");async function renameThread(e,t){e.stopPropagation();
const n=prompt("Title:");if(n){const i=await apiFetch(CHAT_CONFIG.urls.updateTitle.replace("0",t),{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:n})}),a=await i.json().catch(
()=>({}));i.ok&&currentThreadId===String(t)&&setCurrentChatHeaderTitle(a&&a.title||n),loadThreads()}}
o(renameThread,"renameThread");async function deleteThread(e,t){e.stopPropagation(),confirm("Delete?")&&
(await apiFetch(CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),currentThreadId===
t?startNewChat():loadThreads())}o(deleteThread,"deleteThread");async function deleteMessage(e,t){!t&&
!confirm("Delete this message and subsequent history?")||(await apiFetch(CHAT_CONFIG.urls.deleteMessage.
replace("0",e),{method:"DELETE"}),loadMessages(currentThreadId))}o(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=o(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=o(
e=>{if(!e)return"";try{const t=new Date(e);return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(t)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=o(e=>{if(!e)
return"";let t=String(e).trim();if(!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=o(e=>{const t=pdfNormalizeAttachmentPath(e);return t?
`${window.location.origin}/files/${encodeURI(t)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=o(
e=>{const t=pdfNormalizeAttachmentPath(e);return t?`${window.location.origin}/${PDF_IMAGE_EXTS.has((t.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(t)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=o(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const a=n&&n.filename?n.
filename:i.split("/").pop(),r=n&&n.source?String(n.source):"attachment",l=!!(n&&n.is_image),c=n&&n.url?
n.url:buildPdfAttachmentUrl(i),u=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:r,isImage:l,url:c,previewUrl:u}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(u=>maybeNeedsMathJax(u.content)||maybeNeedsMathJax(u.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",r=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(u=>{const f=u.
role==="user",b=u.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(u.quote_text)}\
</div>`:"",y=u.thought_text?`<div class="thought">${escapeHtml(u.thought_text)}</div>`:"",w=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(u.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(u.content||"")}</div>`,v=buildPdfMessageAttachments(u),k=v.length?`<div \
class="attachments">${v.map(L=>L.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(L.previewUrl)}\
" alt="${pdfEscapeAttr(L.filename)}"><div class="file-caption">${pdfEscapeAttr(L.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(L.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
L.filename)}</span><span class="file-source">${pdfEscapeAttr(L.source)}</span></span></a></div>`).join(
"")}</div>`:"",_=[];u.model&&!f&&_.push(u.model),u.tokens!==null&&u.tokens!==void 0&&_.push(`tokens:${u.
tokens}`),u.tokens_in!==null&&u.tokens_in!==void 0&&_.push(`in:${u.tokens_in}`),u.tokens_out!==null&&
u.tokens_out!==void 0&&_.push(`out:${u.tokens_out}`),u.tokens_thought!==null&&u.tokens_thought!==void 0&&
_.push(`thought:${u.tokens_thought}`),u.is_encrypted&&_.push("encrypted"),u.parent_id!==null&&u.parent_id!==
void 0&&_.push(`parent:#${u.parent_id}`);const C=_.length?`<div class="message-meta">${pdfEscapeAttr(
_.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(u.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${b}
                            ${w}
                            ${y}
                            ${k}
                            ${C}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(r)} - PDF Export</title>
        ${a}
        <style>
        :root { --ink:#0f172a; --muted:#475569; --line:#dbe3ee; --panel:#fff; --panel-soft:#f8fafc; \
--user:#0ea5e9; --ai:#10b981; --accent:#0f766e; }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body { font-family: "Noto Sans JP", system-ui, sans-serif; color: var(--ink); background: li\
near-gradient(180deg, #eef4fb 0%, #f8fbff 45%, #eef2f7 100%); }
        .page { max-width: 980px; margin: 0 auto; padding: 24px 18px 48px; }
        .cover { position: relative; overflow: hidden; border-radius: 26px; padding: 26px 24px; colo\
r: #eff6ff; background: linear-gradient(135deg, #0f172a 0%, #0b3b57 56%, #0f766e 100%); box-shadow: \
0 24px 48px rgba(15, 23, 42, 0.18); }
        .cover h1 { margin: 0 0 8px; font-size: 40px; line-height: 1.1; }
        .cover p { margin: 0; max-width: 72ch; color: rgba(226, 232, 240, 0.88); font-size: 14px; li\
ne-height: 1.8; }
        .meta-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; mar\
gin-top: 18px; }
        .meta-card { padding: 12px 14px; border-radius: 16px; background: rgba(15, 23, 42, 0.2); bor\
der: 1px solid rgba(255,255,255,0.15); }
        .meta-label { font-size: 11px; letter-spacing: 0.08em; color: rgba(226,232,240,0.68); margin\
-bottom: 6px; text-transform: uppercase; }
        .meta-value { font-size: 14px; font-weight: 700; word-break: break-word; }
        .message-list { margin-top: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message { border-radius: 22px; border: 1px solid var(--line); background: var(--panel); box\
-shadow: 0 14px 30px rgba(15, 23, 42, 0.06); overflow: hidden; break-inside: avoid; page-break-insid\
e: avoid; }
        .message.user { border-left: 6px solid var(--user); }
        .message.ai { border-left: 6px solid var(--ai); }
        .message-head { display:flex; gap:10px; justify-content:space-between; align-items:flex-star\
t; padding:14px 18px 0; }
        .message-role { display:inline-flex; align-items:center; gap:8px; font-weight:900; font-size\
:13px; }
        .message-role .dot { width:10px; height:10px; border-radius:50%; background: currentColor; }\

        .message-time { color: var(--muted); font-size: 11px; white-space: nowrap; }
        .message-body { padding: 12px 18px 18px; }
        .quote { margin:0 0 12px; padding:10px 12px; border-left:4px solid rgba(14,165,233,0.7); bac\
kground: var(--panel-soft); color: var(--muted); border-radius:12px; font-size:12px; line-height:1.7\
; }
        .thought { margin: 12px 0 0; padding: 12px 14px; border-radius: 14px; background: rgba(139, \
92, 246, 0.06); border: 1px solid rgba(139, 92, 246, 0.18); color: #4c1d95; font-size: 12px; line-he\
ight: 1.8; white-space: pre-wrap; }
        .content { font-size: 14px; line-height: 1.85; word-break: break-word; }
        .content pre { overflow:auto; padding:12px 14px; border-radius:14px; background:#0b1020; col\
or:#e2e8f0; border:1px solid rgba(15,23,42,0.18); }
        .content code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospac\
e; }
        .content blockquote { margin:12px 0; padding:8px 12px; border-left:4px solid rgba(14,165,233\
,0.65); background: rgba(14,165,233,0.06); border-radius:10px; color:#334155; }
        .content img { max-width:100%; height:auto; border-radius:14px; border:1px solid rgba(148,16\
3,184,0.28); margin:10px 0; }
        .attachments { margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minm\
ax(180px, 1fr)); gap: 12px; }
        .attachment { border-radius: 16px; border: 1px solid var(--line); background: #f8fafc; overf\
low: hidden; break-inside: avoid; }
        .attachment img { width: 100%; height: auto; display: block; }
        .attachment .file { display:flex; gap:10px; align-items:center; padding:12px 14px; color: va\
r(--ink); text-decoration:none; }
        .file-icon { font-size: 18px; color: var(--accent); }
        .file-name { display:block; font-weight:700; font-size:13px; word-break:break-word; }
        .file-source { display:block; color: var(--muted); font-size: 11px; margin-top: 3px; }
        .file-caption { padding:10px 12px; font-size:11px; color: var(--muted); }
        .message-meta { margin-top: 14px; text-align: right; color: var(--muted); font-size: 11px; l\
ine-height: 1.7; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        @media print { body { background:#fff; } .page { padding:0; max-width:none; } .cover, .messa\
ge { box-shadow:none; } .message, .attachment, .meta-card { break-inside: avoid; page-break-inside: \
avoid; } a { color: inherit; text-decoration: none; } }
        </style>
        </head>
        <body>
        <div class="page">
        <section class="cover">
        <h1>${pdfEscapeAttr(r)}</h1>
        <p>\u30B9\u30EC\u30C3\u30C9 ID: ${pdfEscapeAttr(t.public_id||"")}\u3002\u8868\u793A\u4E2D\u306E\u5C65\u6B74\u3092\u305D\u306E\u307E\u307E\u5370\u5237\u3067\u304D\u308B\u3088\u3046\u306B\u3001\u753B\u9762\u30AD\u30E3\u30D7\u30C1\
\u30E3\u3067\u306F\u306A\u304F\u5168\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u518D\u69CB\u6210\u3057\u3066\u51FA\u529B\u3057\u3066\u3044\u307E\u3059\u3002</p>
        <div class="meta-grid">
        ${l.map(u=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(u.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(u.value)}</div></div>`).join("")}
        </div>
        </section>
        <main id="pdf-message-list" class="message-list">${c||'<div class="meta-card" style="margin-\
top:20px;background:#fff;color:var(--muted);text-align:center;border:1px dashed rgba(148,163,184,0.5\
);padding:28px;border-radius:20px;">\u3053\u306E\u30B9\u30EC\u30C3\u30C9\u306B\u306F\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'}\
</main>
        </div>
        </body>
        </html>`},"buildPdfDocumentHtml");async function openThreadPdfPrintDialog(){if(!currentThreadId){
showToast("PDF\u5316\u3059\u308B\u30B9\u30EC\u30C3\u30C9\u3092\u958B\u3044\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}if(activePdfPrintFrame){showToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}const e={isLock:!0};activePdfPrintFrame=e;const t=showProgressToast("PDF\u51FA\u529B\u306E\u6E96\u5099\u4E2D\u3067\
\u3059","info");t.update(5);try{const n=new URL(PDF_PRINT_ROUTE.replace("0",currentThreadId),window.
location.origin);currentLeafId!=null&&String(currentLeafId).trim()&&n.searchParams.set("leaf_id",String(
currentLeafId));const i=await apiFetch(n.toString(),{headers:{Accept:"application/json"}});if(t.update(
20),!i.ok){activePdfPrintFrame=null,t&&t.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const a=await i.json().catch(()=>null);if(t.update(30),!a){activePdfPrintFrame=null,
t&&t.remove(),showToast("PDF\u30C7\u30FC\u30BF\u306E\u89E3\u6790\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const r=document.createElement("iframe");activePdfPrintFrame=r,r.setAttribute("ar\
ia-hidden","true"),r.style.position="fixed",r.style.right="0",r.style.bottom="0",r.style.width="1px",
r.style.height="1px",r.style.opacity="0",r.style.pointerEvents="none",r.style.border="0";let l=null;
const c=o(()=>{l&&(clearTimeout(l),l=null),t&&t.remove(),(activePdfPrintFrame===r||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{r.parentNode&&r.parentNode.removeChild(r)}catch{}},"cleanup");l=setTimeout(
()=>{activePdfPrintFrame===r&&(console.log("PDF print cleanup fallback triggered"),c())},6e4),r.onload=
async()=>{try{const b=r.contentDocument,y=r.contentWindow;if(!b||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(L=>maybeNeedsMathJax(L.content)||maybeNeedsMathJax(L.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),b.fonts&&b.fonts.ready)try{await b.fonts.ready}catch{}t.update(60);const k=Array.from(b.
images||[]),_=Promise.all(k.map(L=>L.complete?Promise.resolve():new Promise($=>{L.addEventListener("\
load",$,{once:!0}),L.addEventListener("error",$,{once:!0})})));if(await Promise.race([_,new Promise(
L=>setTimeout(L,5e3))]),t.update(80),b.getElementById("MathJax-script")){let L=0;for(;L<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise($=>setTimeout($,50)),L++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch($){console.error(
"PDF MathJax typeset failed",$)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const u=buildPdfDocumentHtml(a),f=new Blob([u],{type:"text/html"});r.src=URL.createObjectURL(
f),document.body.appendChild(r)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=o((u,f)=>i.
compare(u.filename||"",f.filename||""),"nameAsc"),r=o((u,f)=>i.compare(f.filename||"",u.filename||""),
"nameDesc"),l=o((u,f)=>(Number(f.ts)||0)-(Number(u.ts)||0),"tsDesc"),c=o((u,f)=>(Number(u.ts)||0)-(Number(
f.ts)||0),"tsAsc");return t==="name_asc"?n.sort((u,f)=>a(u,f)||l(u,f)):t==="name_desc"?n.sort((u,f)=>r(
u,f)||l(u,f)):t==="oldest"?n.sort((u,f)=>c(u,f)||a(u,f)):n.sort((u,f)=>l(u,f)||a(u,f)),n}o(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}o(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}o(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}o(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}o(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const f=t.querySelector(".lib-empty-state");f&&f.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const f=get("lib-total-count");f&&(f.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),r=i.filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(
a)),l=get("lib-total-count");if(l){const f=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?l.innerText=`${lib.files.length} / ${f} files`:l.innerText=`${f} files`}if(!r.length){
if(n)return;const f=lib.favoritesOnly&&!a?"fa-star":"fa-search",b=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${f}"></i></div>\
<p class="lib-empty-title">${b}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(a)):r).forEach(f=>{
try{const b=renderLibraryItem(f,c++);t.appendChild(b)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(u=>fileNameForSearch(u).includes(n)):t).filter(u=>u.type==="image"),r=lib.favoritesOnly?
a.filter(u=>u.is_favorite):a;if(!r.length)return;const l=r.map(u=>({url:u.url,filename:u.filename||u.
original_filename||u.url.split("/").pop(),element:null}));let c=l.findIndex(u=>u.url===e.url);c===-1&&
(c=0),openViewerWithItems(l,c)}o(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}o(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,a=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),r=e.type==="\
image"?`<img src="${escapeHtml(i)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(a)}"></i></div><span class="lib-file-badge">${escapeHtml(a?a.toUpperCase():"FI\
LE")}</span></div>`,l=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",u=e.is_favorite?"fas fa-star":"far f\
a-star",f=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
b=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${f}\
" aria-label="${f}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${u}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${b}${y}`,n.
onclick=()=>{lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-se\
lected")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(_=>{_.onclick=C=>{C.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const v=n.querySelector(".lib-del-btn");v&&(v.onclick=async _=>{_.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const k=n.querySelector(".lib-favorite-btn");return k&&
(k.onclick=async _=>{_.stopPropagation(),k.disabled=!0;try{const C=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
L=await C.json().catch(()=>({}));if(!C.ok||typeof L.is_favorite!="boolean")throw new Error(L.error||
"favorite update failed");e.is_favorite=L.is_favorite,renderLibraryGrid(),showToast(L.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),k.disabled=!1}}),n}o(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}o(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",a=FILE_BASE_URL+e,r=i==="image"?FILE_THUMB_BASE_URL+
e:null,l={filename:t,original_filename:t,filepath:e,url:a,thumbnail_url:r,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(l),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
o(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(r=>r.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const a=(i||"").trim();if(!a){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const r=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:a})}),l=await r.json().catch(
()=>({}));if(!r.ok){showToast(l.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=l.filename||a,setAttachmentNameForPath(e,t.filename));const c=get(
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(u=>{u.getAttribute("data-filename")===
e&&setRowAttachmentName(u,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}o(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,t){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),t&&t.parentNode&&t.remove(),lib.files&&(lib.files=
lib.files.filter(n=>n.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}o(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}o(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],t=(lib.files||[]).find(a=>a&&a.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const a=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);a.searchParams.set("filepath",e);const r=await apiFetch(
a.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),l=await r.json().catch(()=>({}));
if(!r.ok)throw new Error(l.error||`HTTP ${r.status}`);const c=Array.isArray(l.chats)?l.chats:[];if(!c.
length){i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(i.innerHTML="",c.forEach(u=>{const f=document.createElement("div");f.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const b=u.updated_at?new Date(u.
updated_at).toLocaleString():"";f.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(u.title||"")}">${escapeHtml(u.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(b)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=f.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(u.id))}),i.appendChild(f)}),l.has_more){const u=document.
createElement("p");u.className="text-[11px] text-gray-500 text-center pt-2",u.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(u)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const u=getLibSortOrder(),f=getLibSearchQuery(),b=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(b),sort:u,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),w=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!w.ok)throw new Error("HTTP "+w.status);r=await w.json(),l=!0}catch(u){i=u}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const u=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,b=new Set(c.map(w=>w&&w.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(w=>{if(c.length>=LIBRARY_PAGE_SIZE||!w||b.has(w))return;const v=getAttachmentNameForPath(
w)||w.split("/").pop()||w,k=(v.split(".").pop()||"").toLowerCase(),_=["png","jpg","jpeg","webp","gif"].
includes(k)?"image":"file",C=_==="image"?f+w:null;c.unshift({filename:v,original_filename:v,filepath:w,
url:u+w,thumbnail_url:C,type:_,ext:k,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),b.add(w)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const u=c.filter(b=>b&&b.filepath&&
b.url);let f=[];if(e){const b=new Set(lib.files.map(y=>y.filepath));f=u.filter(y=>!b.has(y.filepath)),
lib.files.push(...f)}else lib.files=u;lib.files.forEach(b=>{b&&b.filepath&&setAttachmentNameForPath(
b.filepath,b.filename||b.original_filename||"")}),lib.fileSet=new Set(lib.files.map(b=>b.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
f:null)}catch(u){i=i||u}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}o(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}o(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(a=>{const r=isAudioPath(a),
l=isVideoPath(a);if(r&&!e.audio||l&&!e.video){r&&(t+=1),l&&(n+=1);return}const c=normalizeAttachmentPath(
a);if(!c)return;const u=(lib.files||[]).find(f=>f&&f.filepath===a);u&&u.filename&&setAttachmentNameForPath(
c,u.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const a=[];t&&a.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&a.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${a.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}o(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(t=>{const n=(lib.
files||[]).find(i=>i&&i.filepath===t);if(n&&n.url){const i=document.createElement("a");i.href=n.url,
i.download=n.filename||n.original_filename||t.split("/").pop()||"file",document.body.appendChild(i),
i.click(),document.body.removeChild(i)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}o(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const t=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=t,showModal("legal-modal");const n=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!n.ok)return;const i=await n.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
i)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,t)=>{const n=decodeURIComponent(t),i=o(()=>{const a=e.getAttribute("data-copy")||"";e.innerHTML=a===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},a=>{console.error(a),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(i,2e3)})},window.copyMessage=(e,t)=>{const n=messageStore[e]||
"";copyToClipboard(n,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i \
class="fas fa-copy"></i>',2e3)},i=>{console.error(i),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(
()=>t.innerHTML='<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;
t.classList.contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}o(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}o(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let t=0,n=e;const i={};for((allMessages||[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n];t+=a.tokens||
Number(a.tokens_in||0)+Number(a.tokens_out||0),n=a.parent_id}return t}o(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n],r=a.model||"Unknown";t[r]||(t[r]={total:0,in:0,out:0,
thought:0});const l=a.tokens||Number(a.tokens_in||0)+Number(a.tokens_out||0);t[r].total+=l,t[r].in+=
Number(a.tokens_in||0),t[r].out+=Number(a.tokens_out||0),t[r].thought+=Number(a.tokens_thought||0),n=
a.parent_id}return t}o(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(a=>t[a.id]={...a,children:[]}),
allMessages.forEach(a=>{a.parent_id&&t[a.parent_id]?t[a.parent_id].children.push(t[a.id]):a.parent_id||
n.push(t[a.id])});function i(a){const r=document.createElement("div");r.className="flex flex-col ite\
ms-center mt-4";const l=document.createElement("div"),c=String(a.id)===String(currentLeafId),u=a.id===
threadFixedBranchId,f=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),b=getCumulativeTokensForNode(
a.id);if(l.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,l.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(f)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(a.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${b}</span>
                    </div>
                    ${u?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${c?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,l.onclick=y=>{y.stopPropagation(),selectedBranchNodeId=a.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},r.appendChild(l),a.children.length>0){const y=document.createElement("div");
y.className="w-px h-4 bg-gray-700",r.appendChild(y);const w=document.createElement("div");w.className=
"flex gap-4 items-start",a.children.forEach(v=>w.appendChild(i(v))),r.appendChild(w)}return r}o(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}o(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}o(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(u=>u.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const r=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);r.innerHTML="",Object.entries(l).sort((u,f)=>f[1].
total-u[1].total).forEach(([u,f])=>{const b=document.createElement("div");b.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",b.innerHTML=`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${u}</span>
                        <span class="text-blue-400 shrink-0">${f.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${f.in}</div>
                        <div title="Output tokens">Out: ${f.out}</div>
                        <div title="Thought/Reasoning tokens">${f.thought>0?`Th: ${f.thought}`:""}</\
div>
                    </div>
                `,r.appendChild(b)}),get("br-name-input").value=branchLabelNames[n.id]||"";const c=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(c.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
c.classList.replace("bg-amber-600","bg-gray-600")):(c.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
c.classList.replace("bg-gray-600","bg-amber-600"))}o(updateBranchDetailPane,"updateBranchDetailPane"),
get("branch-manage-btn")&&(get("branch-manage-btn").onclick=showBranchModal),get("br-save-name-btn").
onclick=()=>{if(!selectedBranchNodeId)return;const e=get("br-name-input").value.trim();e?branchLabelNames[selectedBranchNodeId]=
e:delete branchLabelNames[selectedBranchNodeId],saveBranchData(),renderBranchTreeVisualization(),showToast(
"\u540D\u524D\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F")},get("br-switch-btn").onclick=()=>{selectedBranchNodeId&&
(switchVersion(selectedBranchNodeId),window.closeBranchModal(),showToast("\u30D6\u30E9\u30F3\u30C1\u3092\u5207\u308A\u66FF\u3048\u307E\u3057\u305F"))},
get("br-fix-btn").onclick=()=>{selectedBranchNodeId&&(threadFixedBranchId===selectedBranchNodeId?(threadFixedBranchId=
null,showToast("\u56FA\u5B9A\u3092\u89E3\u9664\u3057\u307E\u3057\u305F")):(threadFixedBranchId=selectedBranchNodeId,
showToast("\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A\u3057\u307E\u3057\u305F")),saveBranchData(),
renderBranchTreeVisualization(),updateBranchDetailPane())},get("br-delete-btn").onclick=()=>{selectedBranchNodeId&&
confirm("\u3053\u306E\u30D6\u30E9\u30F3\u30C1\u3092\u524A\u9664\u3057\u3066\u3082\u3088\u308D\u3057\u3044\u3067\u3059\u304B\uFF1F\uFF08\u305D\u306E\u5F8C\u306E\u5168\u3066\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u3082\u524A\u9664\u3055\u308C\u307E\u3059\uFF09")&&
(deleteMessage(selectedBranchNodeId,!0),selectedBranchNodeId=null,setTimeout(()=>{renderBranchTreeVisualization(),
updateBranchDetailPane()},500))};let batchJobsCache=[],batchFilterMode="all",batchListTimer=null;function batchProviderLabel(e){
return{gemini:"Gemini",openai:"OpenAI",xai:"xAI",zai:"Z.AI"}[String(e||"").toLowerCase()]||e||"Batch"}
o(batchProviderLabel,"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\
\u9001\u4FE1\u5F85\u3061",JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",
JOB_STATE_RUNNING:"\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\
\u5B8C\u4E86",JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\
\u505C\u6B62",JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}
o(batchStateLabelShort,"batchStateLabelShort");function batchStateTone(e){const t=String(e||"").toUpperCase();
return t==="JOB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":t==="JOB\
_STATE_FAILED"?"border-red-500/40 bg-red-900/20 text-red-200":t==="JOB_STATE_CANCELLED"||t==="JOB_ST\
ATE_EXPIRED"?"border-gray-500/40 bg-gray-700/30 text-gray-300":t==="JOB_STATE_CANCELLING"?"border-am\
ber-500/40 bg-amber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}o(
batchStateTone,"batchStateTone");function batchFormatTime(e){if(!e)return"";let t=String(e);!/[zZ]$/.
test(t)&&!/[+-]\d\d:?\d\d$/.test(t)&&(t+="Z");const n=new Date(t);return isNaN(n.getTime())?String(e):
n.toLocaleString("ja-JP",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}o(batchFormatTime,
"batchFormatTime");function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove(
"batch-list-enter"),e.offsetWidth,e.classList.add("batch-list-enter"))}o(playBatchListAnimation,"pla\
yBatchListAnimation");function renderBatchJobs(e={}){const t=get("batch-list");if(!t)return;const n=batchJobsCache.
filter(a=>batchFilterMode==="active"?!!a.is_active:batchFilterMode==="done"?!a.is_active:!0),i=get("\
batch-count");if(i&&(i.textContent=`${n.length}\u4EF6`),t.innerHTML="",!n.length){t.innerHTML='<div \
class="batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}n.forEach(a=>{const r=document.createElement("div");r.className=
"batch-job-card";const l=escapeHtml(a.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),c=escapeHtml(
batchProviderLabel(a.provider)),u=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
b=escapeHtml(a.status_text||""),y=batchStateTone(a.state),w=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",v=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",k=a.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';r.innerHTML=`
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${l}">${l}\
</div>
                            <div class="batch-job-meta mt-1 flex flex-wrap items-center gap-2 text-[\
10px]">
                                <span class="inline-flex items-center gap-1"><i class="fas fa-layer-\
group"></i>${c}</span>
                                <span class="truncate max-w-[16rem]">${u}</span>
                                <span><i class="fas fa-history mr-1"></i>${f}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${y}">${escapeHtml(batchStateLabelShort(
a.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${b}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${w}${v}${k}
                    </div>`;const _=r.querySelector("[data-batch-open]");_&&(_.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const C=r.querySelector("[data-batch-cancel]");C&&(C.onclick=()=>cancelBatchJob(
a));const L=r.querySelector("[data-batch-delete]");L&&(L.onclick=()=>deleteBatchJob(a)),t.appendChild(
r)}),e.animate&&playBatchListAnimation()}o(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const t=await apiFetch("/api/batch/jobs");if(!t.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const n=await t.json().catch(()=>({}));batchJobsCache=Array.isArray(n.jobs)?n.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}o(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),n=await t.json().catch(()=>({}));if(!t.ok){showToast(n.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}o(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),n=await t.json().catch(()=>({}));
if(!t.ok){showToast(n.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}o(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(t=>{t.classList.toggle("is-active",t===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=o(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const a=i?get(i.inputId):null;get("api-key-modal-input").
value=a?a.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const r=get(
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),u=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=o(b=>{b.key==="Enter"&&(b.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),r.onclick=async()=>{
const b=get("api-key-modal-input").value.trim();if(!b){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=b);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:b})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=b)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",f),u(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",f),u(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",f),u(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const b=get("api-key-modal-input");b&&b.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const u=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,u)}catch{}finally{
a=!1}}o(r,"sendToServer"),console.log=function(...l){e.apply(console,l),r("log",l)},console.error=function(...l){
t.apply(console,l),r("error",l)},console.warn=function(...l){n.apply(console,l),r("warn",l)},console.
info=function(...l){i.apply(console,l),r("info",l)},window.addEventListener("error",function(l){r("e\
xception",[l.message,l.filename,l.lineno,l.colno,l.error])}),window.addEventListener("unhandledrejec\
tion",function(l){r("promise-rejection",[l.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();
