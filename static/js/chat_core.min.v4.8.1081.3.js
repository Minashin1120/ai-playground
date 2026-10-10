function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),r=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const M=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",M.forEach((P,j)=>{
const Z=P.getAttribute("data-local-url"),J=P.getAttribute("data-filename"),Ce=P.querySelector("img.u\
pload-preview")!==null;let T;if(Ce){let I=Z;if(!I&&J){const O=J.replace(/^\d+\//,"");I=buildAttachmentPreviewUrl(
O)}I&&(T=document.createElement("img"),T.src=I,T.className="thumb-item shadow-sm",T.dataset.viewerSrc=
I,T.dataset.viewerFilename=J||I.split("/").pop(),T.onclick=function(O){O.preventDefault(),openImageViewer(
this.dataset.viewerSrc,".thumb-item")},T.onerror=function(){this.parentElement.replaceChild(m("ERR"),
this)})}T||(T=m("FILE")),T.style.animationDelay=`${j*32}ms`,a.appendChild(T)}),M.length>0?a.classList.
remove("hidden"):a.classList.add("hidden")}function m(M){const P=document.createElement("div");return P.
className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-[9px] shadow-s\
m font-bold",P.innerText=M,P}o(m,"createFileThumb");const f=collectImageUrlsForSend(),b=uploadProgressState.
total,y=uploadProgressState.completed,v=uploadProgressState.active;b===0&&(e.classList.add("hidden"),
n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));const w=get(
"send-btn"),k=get("mic-btn"),S=get("mask-btn"),C=isStopMode;if(v>0?(w&&(w.disabled=!0),k&&(k.disabled=
!0),S&&(S.disabled=!0)):C||(w&&(w.disabled=!1),k&&(k.disabled=!1),S&&(S.disabled=!1)),v>0){const M=`\
Preparing... (${y}/${b})`;e.classList.remove("hidden"),t.innerText=M,r&&(r.innerText=`(${y}/${b})`);
let P=y*100,j=0;for(let Ce in uploadProgressState.perFilePct)P+=uploadProgressState.perFilePct[Ce],j++;
const Z=b>0?P/(b*100)*100:0,J=`${Math.min(100,Z)}%`;n&&i&&(n.classList.remove("hidden"),i.style.width=
J),l&&c&&(l.classList.remove("hidden"),c.style.width=J)}else r&&(r.innerText=""),l&&l.classList.add(
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
0);if(n<=1||i<=1||a<=1||r<=1)return;const l=(n-a)/2,c=(i-r)/2,m=a*markerView.scale,f=r*markerView.scale,
b=Math.min(n*.45,Math.max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=b-l-m,w=n-b-l,k=y-c-f,S=i-
y-c,C=o((M,P,j)=>Number.isFinite(M)?P>j?(P+j)/2:Math.min(j,Math.max(P,M)):0,"clampOffset");markerView.
offsetX=C(markerView.offsetX,v,w),markerView.offsetY=C(markerView.offsetY,k,S)}o(clampMarkerViewOffset,
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
height);const n=o((l,c,m=null,f=!1)=>{if(!l)return;const b=Math.max(0,l.x),y=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);m&&(t.fillStyle=m,t.fillRect(b,y,v,w)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(b+.5,y+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),m=Math.max(1,i.w),f=Math.max(1,i.h);t.clearRect(l,c,m,f),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((a,r,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const m=normalizeAttachmentSource(
r),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),b=t.get(c);if(b===void 0){const w=e.
length;t.set(c,w),e.push({path:c,source:m,name:f});return}const y=e[b];if(!y)return;const v=normalizeAttachmentSource(
y.source);(v==="unknown"&&m!=="unknown"||v==="library"&&m==="upload")&&(y.source=m),!normalizeAttachmentDisplayName(
y.name)&&f&&(y.name=f)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const r=a.getAttribute("data-filename");n(r,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const r=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const m=isAudioPath(
c),f=isVideoPath(c);if(m&&!t.audio||f&&!t.video){m&&(n+=1),f&&(i+=1);return}r.push(c)}),r.length!==currentImageUrls.
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
/edit-image"),a&&(a.onload=()=>{if(!get("marker-stage")||!r)return;const m=Math.max(1,Math.floor(a.clientWidth)),
f=Math.max(1,Math.floor(a.clientHeight));r.width=m,r.height=f,r.style.width=`${m}px`,r.style.height=
`${f}px`,r.style.left="0px",r.style.top="0px",l&&(l.width=m,l.height=f,l.style.width=`${m}px`,l.style.
height=`${f}px`,l.style.left="0px",l.style.top="0px"),markerState.naturalWidth=a.naturalWidth||m,markerState.
naturalHeight=a.naturalHeight||f;const b=r.getContext("2d");b&&b.clearRect(0,0,r.width,r.height),prepareMarkerBaseCanvas(
a,m,f),saveMarkerHistory(),markerState.mode==="crop"&&!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),
resetMarkerTransform()},a.src=t)}o(openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={
total:0,completed:0,active:0,perFilePct:{}};const uploadCancelTokens=new Set;function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
o(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){browserFastLocalFiles.
forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,m=c?c.getAttribute("data-local-url"):null;m&&URL.revokeObjectURL(
m)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=null,uploadProgressState={total:0,
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
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],m=c&&c.getSettings?c.getSettings():{},
f=String(m.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
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
const l=await new Promise((m,f)=>{t.toBlob(b=>{b?m(b):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
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
"gif"].includes(l),m=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):m,b=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=b,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=r,y.dataset.defaultDisplayName=r,y.dataset.sendNameCustomized="";const v=escapeHtml(
r),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",k=c?`<img src="${f}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${k}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${v}</div>
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
            `;const S=y.querySelector(".upload-preview");S&&(S.onclick=()=>openFileViewer(m,getRowAttachmentName(
y)||r));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const M=y.
querySelector(".upload-remove");M&&(M.onclick=()=>{uploadCancelTokens.add(b),browserFastLocalFiles.delete(
b),decrementUploadTotal(b);const j=y.getAttribute("data-filename");j&&(currentImageUrls=currentImageUrls.
filter(Z=>Z!==j)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const P=y.querySelector(".upload-marker");return P&&(P.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,r),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:b}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const r=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const m=l&&
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
                        ${m}
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
onclick=()=>{const k=i.getAttribute("data-filename"),S=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(S,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const S=i.
getAttribute("data-filename");S&&(currentImageUrls=currentImageUrls.filter(C=>C!==S)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const v=i.querySelector(".upload-marker");v&&(v.onclick=()=>openMarkerModalForRow(i));const w=i.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(!e)
return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);try{const i=await apiFetch(
"/upload/init",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},body:JSON.
stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.ok){const y=a&&a.error?a.error:"\u30A2\u30C3\
\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast(y,"error",!0),!1}const r=a.upload_id,l=a.chunk_size||10*1024*1024,c=Math.ceil(
e.size/l);for(let y=0;y<c;y++){const v=y*l,w=Math.min(e.size,v+l),k=e.slice(v,w);if(!await new Promise(
C=>{const M=new XMLHttpRequest;M.open("POST","/upload/chunk",!0),M.setRequestHeader("X-CSRF-Token",csrfToken),
M.upload.onprogress=j=>{if(j.lengthComputable&&t&&t.bar){const Z=v+j.loaded,J=Math.min(100,Math.floor(
Z/e.size*100));t.bar.style.width=`${J}%`,t.status&&(t.status.textContent=`${J}%`),t.uploadId&&updateGlobalUploadProgress(
t.uploadId,J)}window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity()},M.onload=()=>{M.status>=
200&&M.status<300?C(!0):C(!1)},M.onerror=()=>C(!1);const P=new FormData;P.append("upload_id",r),P.append(
"index",String(y)),P.append("total",String(c)),P.append("chunk",k,e.name),M.send(P)}))return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const m=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),f=await m.json();if(m.ok&&f&&f.filename){if(t&&t.row&&t.uploadId&&
uploadCancelTokens.has(t.uploadId))return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.
row.getAttribute("data-local-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");
const k=t.row.querySelector("img.upload-preview");if(k){const S=f.filename.replace(/^\d+\//,"");k.src=
buildAttachmentPreviewUrl(S)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(
y),t&&t.row&&(t.row.setAttribute("data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),
y)){const w=isRowAttachmentNameCustomized(t.row),k=defaultAttachmentDisplayName(y),S=w&&normalizeAttachmentDisplayName(
t.row.dataset.displayName)||k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,S)}return y&&
setAttachmentSourceForPath(y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),
(Array.isArray(f.filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(w=>addLibraryFileFromPath(
w)),!0}const b=f&&f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(b,"error",!0),!1}catch{return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}let i=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),
i=!0);const a=o(()=>{i&&window.ConnectionMonitor&&(window.ConnectionMonitor.operationEnded(),i=!1)},
"finishUploadOp"),r=new XMLHttpRequest;r.open("POST",CHAT_CONFIG.urls.upload,!0),r.setRequestHeader(
"X-CSRF-Token",csrfToken),r.upload.onprogress=c=>{if(c.lengthComputable&&t&&t.bar){const m=Math.min(
100,Math.floor(c.loaded/c.total*100));t.bar.style.width=`${m}%`,t.status&&(t.status.textContent=`${m}\
%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,m)}window.ConnectionMonitor&&window.ConnectionMonitor.
reportActivity()},r.onload=()=>{let c={};try{c=JSON.parse(r.responseText||"{}")}catch{}if(r.status>=
200&&r.status<300&&c&&c.filename){if(t&&t.row&&t.uploadId&&uploadCancelTokens.has(t.uploadId)){t.row&&
t.row.parentNode&&t.row.remove(),a(),n(!1);return}if(t&&t.row){const b=t.row.getAttribute("data-loca\
l-url");b&&URL.revokeObjectURL(b),t.row.removeAttribute("data-local-url");const y=t.row.querySelector(
"img.upload-preview");if(y){const v=c.filename.replace(/^\d+\//,"");y.src=buildAttachmentPreviewUrl(
v)}}const m=normalizeAttachmentPath(c.filename);if(m&&currentImageUrls.push(m),t&&t.row&&(t.row.setAttribute(
"data-filename",m||c.filename),setRowAttachmentSource(t.row,"upload"),m)){const b=isRowAttachmentNameCustomized(
t.row),y=defaultAttachmentDisplayName(m),v=b&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
y;t.row.dataset.defaultDisplayName=y,setRowAttachmentName(t.row,v)}m&&setAttachmentSourceForPath(m,"\
upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(c.filenames)&&
c.filenames.length?c.filenames:[c.filename]).forEach(b=>addLibraryFileFromPath(b)),a(),n(!0)}else{const m=c&&
c.error?c.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(m,"error",!0),a(),n(!1)}},r.onerror=()=>{t&&
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
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let m=44;for(let f=0;f<i.length;f++){const b=Math.
max(-1,Math.min(1,i[f]));l.setInt16(m,b<0?b*32768:b*32767,!0),m+=2}return new Blob([l],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const r=e.row.getAttribute("data-local-url");r&&URL.revokeObjectURL(
r);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),m=escapeHtml(a),f=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${m}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',b=e.row.querySelector(".upload-preview");b&&(b.outerHTML=f);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const w=e.row.getAttribute("data-filename"),k=w?buildFileUrl(w):
e.row.getAttribute("data-local-url");openFileViewer(k,getRowAttachmentName(e.row)||a||w||"")});const v=e.
row.querySelector(".upload-marker");v&&v.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
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
l=markerView.scale,c={x:0,y:0},m={x:0,y:0},f=[],b=16,y="",v=null,w=null,k=null,S=null,C=!1,M=null;const P=o(
E=>{const H=e.getBoundingClientRect(),Q=(E.clientX-H.left)*(e.width/H.width),te=(E.clientY-H.top)*(e.
height/H.height);return{x:Q,y:te}},"getPoint"),j=o((E,H)=>({x:(E.x+H.x)/2,y:(E.y+H.y)/2}),"getMid"),
Z=o((E,H)=>Math.hypot(E.x-H.x,E.y-H.y),"getDist");let J=!1;const Ce=o(()=>{v||(v=document.createElement(
"canvas"),w=v.getContext("2d")),k||(k=document.createElement("canvas"),S=k.getContext("2d")),(v.width!==
e.width||v.height!==e.height)&&(v.width=e.width,v.height=e.height),(k.width!==e.width||k.height!==e.
height)&&(k.width=e.width,k.height=e.height)},"ensureDrawBuffers"),T=o(()=>{if(!t||!v||!k)return;const E=Math.
max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(0,0,e.width,e.
height),t.drawImage(v,0,0),t.save(),t.globalAlpha=E,t.drawImage(k,0,0),t.restore()},"renderDrawPrevi\
ew"),I=o(()=>{S&&(S.strokeStyle=y,S.fillStyle=y,S.lineWidth=b,S.lineCap="round",S.lineJoin="round")},
"applyMarkerBrush"),O=o(E=>{if(!E)return!1;if(f.length===0)return f.push(E),!0;const H=f[f.length-1],
Q=E.x-H.x,te=E.y-H.y,ae=Math.hypot(Q,te),Ae=Math.max(.35,b*.04);if(ae<Ae)return!1;const se=Math.max(
1,b*.25),he=Math.max(1,Math.ceil(ae/se));for(let Y=1;Y<=he;Y++){const ve=Y/he;f.push({x:H.x+Q*ve,y:H.
y+te*ve})}return!0},"appendStrokePoint"),K=o(()=>{if(S&&(S.clearRect(0,0,k.width,k.height),f.length!==
0)){if(I(),f.length===1){const E=f[0];S.beginPath(),S.arc(E.x,E.y,b/2,0,Math.PI*2),S.fill();return}if(S.
beginPath(),S.moveTo(f[0].x,f[0].y),f.length===2)S.lineTo(f[1].x,f[1].y);else{for(let Q=1;Q<f.length-
2;Q++){const te=f[Q],ae=f[Q+1],Ae=j(te,ae);S.quadraticCurveTo(te.x,te.y,Ae.x,Ae.y)}const E=f[f.length-
2],H=f[f.length-1];S.quadraticCurveTo(E.x,E.y,H.x,H.y)}S.stroke()}},"renderStrokeLayer"),X=o((E,H)=>{
if(!E||!H)return null;const Q=Math.min(E.x,H.x),te=Math.min(E.y,H.y),ae=Math.abs(E.x-H.x),Ae=Math.abs(
E.y-H.y);return{x:Q,y:te,w:ae,h:Ae}},"normalizeMosaicRect"),ge=o(E=>{const H=n?Number(n.value||16):16,
Q=Math.max(6,Math.floor(H)),te=Math.floor(Q/2);return{x:E.x-te,y:E.y-te,w:Q,h:Q}},"buildMosaicRectFr\
omPoint"),_e=o(()=>{const E=document.createElement("canvas");E.width=e.width,E.height=e.height;const H=E.
getContext("2d");if(!H)return null;markerState.baseCanvas&&H.drawImage(markerState.baseCanvas,0,0),H.
drawImage(e,0,0);try{return H.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosaicSourc\
eImageData"),ce=o(E=>{if(!t||!E)return!1;const H=_e();if(!H)return!1;const Q=n?Number(n.value||16):16,
te=Math.max(4,Math.floor(Q/2)),ae=Math.max(0,Math.floor(E.x)),Ae=Math.max(0,Math.floor(E.y)),se=Math.
min(e.width,Math.ceil(E.x+E.w)),he=Math.min(e.height,Math.ceil(E.y+E.h));if(se<=ae||he<=Ae)return!1;
for(let Y=Ae;Y<he;Y+=te)for(let ve=ae;ve<se;ve+=te){const pt=Math.min(te,se-ve),ct=Math.min(te,he-Y),
mt=Math.min(e.width-1,Math.max(0,ve+Math.floor(pt/2))),Ve=(Math.min(e.height-1,Math.max(0,Y+Math.floor(
ct/2)))*e.width+mt)*4,bt=H.data[Ve],Mt=H.data[Ve+1],wt=H.data[Ve+2];t.fillStyle=`rgb(${bt},${Mt},${wt}\
)`,t.fillRect(ve,Y,pt,ct)}return!0},"applyMosaicRect"),me=o(E=>{if(!t)return;if(i.set(E.pointerId,{x:E.
clientX,y:E.clientY}),i.size>=2){const Q=Array.from(i.values()),te=Q[0],ae=Q[1];a=!0,J=!1,f=[],C=!1,
M=null,markerState.mosaicPreviewRect=null,r=Z(te,ae)||1,l=markerView.scale,c={x:markerView.offsetX,y:markerView.
offsetY},m=j(te,ae),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(E.pointerId),E.preventDefault();
return}if(a||markerState.mode==="crop")return;J=!0;const H=P(E);if(markerState.mode==="mosaic")C=!0,
M=H,markerState.mosaicPreviewRect=ge(H),renderCropOverlay();else{if(Ce(),!w||!S)return;w.clearRect(0,
0,v.width,v.height),w.drawImage(e,0,0),S.clearRect(0,0,k.width,k.height),b=n?Number(n.value||16):16,
y=normalizeMarkerHexColor(markerState.colorHex),f=[],O(H),K(),markerState.hasStroke=!0,T()}e.setPointerCapture&&
e.setPointerCapture(E.pointerId),E.preventDefault()},"start"),de=o(E=>{if(i.has(E.pointerId)&&i.set(
E.pointerId,{x:E.clientX,y:E.clientY}),a&&i.size>=2){const Q=Array.from(i.values()),te=Q[0],ae=Q[1],
Ae=j(te,ae),se=Z(te,ae)||1,he=l*(se/r);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.
minScale,he)),markerView.offsetX=c.x+(Ae.x-m.x),markerView.offsetY=c.y+(Ae.y-m.y),applyMarkerTransform(),
E.preventDefault();return}if(!J||!t)return;const H=P(E);if(markerState.mode==="mosaic"){if(!C||!M)return;
markerState.mosaicPreviewRect=X(M,H)||ge(H),renderCropOverlay()}else O(H)&&(K(),T());E.preventDefault()},
"move"),G=o(E=>{const H=J;if(i.delete(E.pointerId),i.size<2&&(a=!1),i.size===0){if(J=!1,H&&t&&markerState.
mode==="draw"&&f.length>0&&(K(),T()),H&&markerState.mode==="mosaic"&&M){const Q=P(E);let te=X(M,Q);(!te||
te.w<2||te.h<2)&&(te=ge(M)),ce(te)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(te))}f=[],
C=!1,M=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),H&&saveMarkerHistory()}e.releasePointerCapture&&
e.releasePointerCapture(E.pointerId),E.preventDefault()},"end");e.addEventListener("pointerdown",me),
e.addEventListener("pointermove",de),e.addEventListener("pointerup",G),e.addEventListener("pointerca\
ncel",G)}o(initMarkerCanvas,"initMarkerCanvas");function initCropCanvas(){const e=get("marker-crop-c\
anvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,m=0,f=markerView.
scale,b={x:0,y:0},y={x:0,y:0};const v=8,w=14,k=o((T,I,O)=>Math.min(O,Math.max(I,T)),"clamp"),S=o(T=>{
const I=e.getBoundingClientRect(),O=(T.clientX-I.left)*(e.width/I.width),K=(T.clientY-I.top)*(e.height/
I.height);return{x:O,y:K}},"getPoint"),C=o((T,I)=>({x:(T.x+I.x)/2,y:(T.y+I.y)/2}),"getMid"),M=o((T,I)=>Math.
hypot(T.x-I.x,T.y-I.y),"getDist"),P=o(()=>(markerState.cropRect||resetCropRectToFull(),markerState.cropRect),
"ensureCropRect"),j=o((T,I)=>{if(!I)return"move";const O=I.x,K=I.y,X=I.x+I.w,ge=I.y+I.h,_e=Math.abs(
T.x-O)<=w,ce=Math.abs(T.x-X)<=w,me=Math.abs(T.y-K)<=w,de=Math.abs(T.y-ge)<=w;if(_e&&me)return"nw";if(ce&&
me)return"ne";if(_e&&de)return"sw";if(ce&&de)return"se";if(me)return"n";if(de)return"s";if(_e)return"\
w";if(ce)return"e";if(T.x>O+w&&T.x<X-w&&T.y>K+w&&T.y<ge-w)return"move";const E=T.x<O?"left":T.x>X?"r\
ight":null,H=T.y<K?"top":T.y>ge?"bottom":null;if(E&&H){if(E==="left"&&H==="top")return"nw";if(E==="r\
ight"&&H==="top")return"ne";if(E==="left"&&H==="bottom")return"sw";if(E==="right"&&H==="bottom")return"\
se"}return E?E==="left"?"w":"e":H?H==="top"?"n":"s":"move"},"hitTest"),Z=o(T=>{if(markerState.mode!==
"crop")return;if(n.set(T.pointerId,{x:T.clientX,y:T.clientY}),n.size>=2){const K=Array.from(n.values()),
X=K[0],ge=K[1];c=!0,i=!1,m=M(X,ge)||1,f=markerView.scale,b={x:markerView.offsetX,y:markerView.offsetY},
y=C(X,ge),e.setPointerCapture&&e.setPointerCapture(T.pointerId),T.preventDefault();return}if(c)return;
i=!0;const I=S(T),O=P();r=j(I,O),a=I,l=O?{x:O.x,y:O.y,w:O.w,h:O.h}:null,renderCropOverlay(),e.setPointerCapture&&
e.setPointerCapture(T.pointerId),T.preventDefault()},"start"),J=o(T=>{if(markerState.mode!=="crop")return;
if(n.has(T.pointerId)&&n.set(T.pointerId,{x:T.clientX,y:T.clientY}),c&&n.size>=2){const E=Array.from(
n.values()),H=E[0],Q=E[1],te=C(H,Q),ae=M(H,Q)||1,Ae=f*(ae/m);markerView.scale=Math.min(markerView.maxScale,
Math.max(markerView.minScale,Ae)),markerView.offsetX=b.x+(te.x-y.x),markerView.offsetY=b.y+(te.y-y.y),
applyMarkerTransform(),renderCropOverlay(),T.preventDefault();return}if(!i||!a||!l)return;const I=S(
T),O=e.width,K=e.height,X={x:l.x,y:l.y,w:l.w,h:l.h},ge=l.x+l.w,_e=l.y+l.h,ce=o(()=>{const E=k(I.x,0,
ge-v);X.x=E,X.w=ge-E},"applyW"),me=o(()=>{X.w=k(I.x-l.x,v,O-l.x)},"applyE"),de=o(()=>{const E=k(I.y,
0,_e-v);X.y=E,X.h=_e-E},"applyN"),G=o(()=>{X.h=k(I.y-l.y,v,K-l.y)},"applyS");switch(r){case"move":{const E=I.
x-a.x,H=I.y-a.y;X.x=k(l.x+E,0,O-l.w),X.y=k(l.y+H,0,K-l.h);break}case"w":ce();break;case"e":me();break;case"\
n":de();break;case"s":G();break;case"nw":de(),ce();break;case"ne":de(),me();break;case"sw":G(),ce();
break;case"se":G(),me();break;default:break}X.x=k(X.x,0,O-X.w),X.y=k(X.y,0,K-X.h),markerState.cropRect=
X,renderCropOverlay(),T.preventDefault()},"move"),Ce=o(T=>{n.delete(T.pointerId),n.size<2&&(c=!1),n.
size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),e.releasePointerCapture&&e.releasePointerCapture(
T.pointerId),T.preventDefault()},"end");e.addEventListener("pointerdown",Z),e.addEventListener("poin\
termove",J),e.addEventListener("pointerup",Ce),e.addEventListener("pointercancel",Ce),e.addEventListener(
"pointerleave",Ce)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){const e=markerState.
row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get("marker-attach-ori\
ginal");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement("canvas");const r=markerState.
naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||n.height;a.width=
r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,0,r,l),c.drawImage(n,0,0,r,l),
markerState.cropRect){const C=r/n.width,M=l/n.height,P=Math.max(0,Math.floor(markerState.cropRect.x*
C)),j=Math.max(0,Math.floor(markerState.cropRect.y*M)),Z=Math.min(r,Math.max(1,Math.floor(markerState.
cropRect.w*C))),J=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.h*M))),Ce=document.createElement(
"canvas");Ce.width=Z,Ce.height=J;const T=Ce.getContext("2d");T&&(T.drawImage(a,P,j,Z,J,0,0,Z,J),a=Ce)}
const m=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!m){showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const b=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([
m],`${b}_marked.png`,{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(
".upload-status"),bar:e.querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\
\u53CD\u6620\u4E2D..."),updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),k=getRowAttachmentSource(
e);w&&!e.dataset.originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=k,setAttachmentSourceForPath(
w,k)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null}o(saveMarkerToRow,"sa\
veMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(e)||!HTMLMediaElement.prototype.
captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),new Promise(
n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,i.src=URL.createObjectURL(
e);let a=null,r=null,l=null,c=null,m=[],f=null;const b=o(()=>{f&&clearTimeout(f);try{URL.revokeObjectURL(
i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),l)try{l.disconnect()}catch{}
if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(()=>{b(),n(null)},"fail");
i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();if(!v||!v.length)return y();
r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),c=r.createMediaStreamSource(
new MediaStream(v)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=k=>{const S=k.inputBuffer.getChannelData(
0);m.push(new Float32Array(S))},c.connect(l),l.connect(r.destination);const w=isFinite(i.duration)?Math.
max(1,Math.ceil(i.duration*1e3)):0;w>0&&(f=setTimeout(()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,
""),S=encodeWav(m,r.sampleRate),C=new File([S],`${k}.audio.wav`,{type:"audio/wav"});b(),n(C)},w+250)),
await i.play(),i.onended=()=>{const k=(e.name||"video").replace(/\.[^/.]+$/,""),S=encodeWav(m,r.sampleRate),
C=new File([S],`${k}.audio.wav`,{type:"audio/wav"});b(),n(C)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,
"extractAudioFromVideo");async function handleFiles(e,t={}){if(!e||!e.length)return;const n=Array.from(
e).filter(Boolean);if(!n.length)return;const i=n;t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),
uploadProgressState.total+=i.length,uploadProgressState.active+=i.length,updateFilePreview();const a=!!(get(
"upload-audio-only")&&get("upload-audio-only").checked),r=getModelMediaSupport(get("model-select").value),
l=o(async b=>{let y=null;try{if(isAudioFile(b)&&!r.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(b)&&!r.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!b.type||!b.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const v=addUploadRow(b);updateFilePreview(),y=v.uploadId,uploadProgressState.perFilePct[y]=
0;let w=b;if(a&&isVideoFile(b)){const k=await extractAudioFromVideo(b,v);k?(w=k,updateUploadRowFile(
v,k),v&&v.status&&(v.status.textContent="\u97F3\u58F0\u306E\u307F")):(v&&v.status&&(v.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(get("enable-compression").checked&&b.type.startsWith("image/"))try{const k=getCompressionOutputType();
if(getCompressionFormatOnly())w=await convertImageFormatOnly(b,k);else{const C={maxSizeMB:getCompressionMaxSizeMB(),
maxWidthOrHeight:getCompressionMaxDim(),useWebWorker:!0};k&&k!=="original"&&(C.fileType=k),await ensureImageCompression();
const M=await window.imageCompression(b,C),P=new File([M],imageFilenameForMime(b.name,M.type||(k!=="\
original"?k:b.type)),{type:M.type||b.type,lastModified:b.lastModified||Date.now()});P.size>b.size?(showToast(
`\u5727\u7E2E\u5F8C\u306B\u30B5\u30A4\u30BA\u304C\u5897\u52A0\u3057\u307E\u3057\u305F: ${formatBytes(
b.size)} -> ${formatBytes(P.size)}\uFF08\u5143\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\uFF09`,"war\
ning",!0),w=b):w=P}w!==b&&updateUploadRowFile(v,w)}catch{}if(browserFastModeEnabled){const k=Array.from(
browserFastLocalFiles.values()).reduce((S,C)=>S+Number(C.file&&C.file.size||0),0);return browserFastLocalFiles.
size>=BROWSER_FAST_MAX_IMAGES||k+w.size>BROWSER_FAST_MAX_BYTES?(v&&v.status&&(v.status.textContent="\
\u4E0A\u9650\u8D85\u904E"),v&&v.row&&v.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059",
"error",!0),!1):(browserFastLocalFiles.set(v.uploadId,{file:w,rowObj:v}),v.status&&(v.status.textContent=
"\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),v.bar&&(v.bar.style.width="100\
%"),v.row&&(v.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(w,v)}finally{
y&&uploadProgressState.perFilePct.hasOwnProperty(y)&&(delete uploadProgressState.perFilePct[y],uploadProgressState.
completed++,uploadProgressState.active--),uploadProgressState.active<=0&&(uploadProgressState.total=
0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.perFilePct={}),updateFilePreview()}},
"processOne");let c=0;const m=Math.min(UPLOAD_CONCURRENCY,i.length),f=Array.from({length:m}).map(async()=>{
for(;;){const b=c++;if(b>=i.length)break;await l(i[b])}});await Promise.all(f)}o(handleFiles,"handle\
Files"),get("clear-file-btn").onclick=()=>{resetUploadState()},get("clear-mask-btn")&&(get("clear-ma\
sk-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-input")&&
(get("mask-btn").onclick=()=>{get("mask-input").click()},get("mask-input").addEventListener("change",
async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value="")}));
const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){const n=String(
e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||typeof window.DOMPurify.
sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=!0,console.
error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).replace(/\n/g,
"<br>");const i=protectMathSegments(n);markdownOpenFenceBody=/<\/svg/i.test(n)?findOpenMarkdownFenceBody(
n):null;let a="";try{a=window.marked.parse(i.text)}finally{markdownOpenFenceBody=null}const r=restoreMathSegments(
a,i.blocks,t);return window.DOMPurify.sanitize(r)}o(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");let markdownOpenFenceBody=null;
function findOpenMarkdownFenceBody(e){const t=String(e||"").split(/\r?\n/);let n=null,i=[];for(const a of t){
if(!n){const l=a.match(/^\s*(`{3,}|~{3,})(.*)$/);l&&!(l[1][0]==="`"&&l[2].includes("`"))&&(n=l[1],i=
[]);continue}const r=a.trim();if(r.length>=n.length&&r[0]===n[0]&&/^(`+|~+)$/.test(r)){n=null;continue}
i.push(a)}return n?i.join(`
`):null}o(findOpenMarkdownFenceBody,"findOpenMarkdownFenceBody");const SVG_CODE_RENDER_MAX_CHARS=3e5;
function getRenderableSvgCode(e,t){const n=String(e||"").trim().toLowerCase();if(n!=="svg"&&n!=="xml"&&
n!=="image/svg+xml")return"";const i=String(t||"").trim();if(!i||i.length>SVG_CODE_RENDER_MAX_CHARS)
return"";const a=i.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i,"");return!/^<svg[\s>]/i.
test(a)||!/<\/svg\s*>$/i.test(a)||markdownOpenFenceBody!==null&&markdownOpenFenceBody.replace(/\s+/g,
"")===i.replace(/\s+/g,"")?"":i}o(getRenderableSvgCode,"getRenderableSvgCode");function buildSvgCodeRenderHtml(e,t){
let n=!0;const i=String(e||"").replace(/<svg\b([^>]*)>/i,(r,l)=>{let c=l;/\sxmlns\s*=/i.test(c)||(c=
` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(e)&&!/\sxmlns:xlink\s*=/i.test(c)&&(c=` \
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const m=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return m&&Number(m[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
i)}`;return`<div class="svg-render-box svg-code-render" data-svg-key="${escapeHtml(String(t||""))}">\
<img src="${a}" alt="SVG"${n?' class="svg-code-fill"':""} decoding="async"></div>`}o(buildSvgCodeRenderHtml,
"buildSvgCodeRenderHtml");function getCanvasModeElements(){const e=get("canvas-panel");return e?{panel:e,
stage:get("conversation-stage"),title:get("canvas-panel-title"),status:get("canvas-panel-status"),blockCount:get(
"canvas-block-count"),blockList:get("canvas-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get(
"canvas-preview-lang"),sourceSelect:get("canvas-source-select"),frame:get("canvas-preview-frame"),empty:get(
"canvas-preview-empty"),sourceScroll:get("canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get(
"canvas-panel-copy-btn"),clearBtn:get("canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:
null}o(getCanvasModeElements,"getCanvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(
e||"").trim().toLowerCase();if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(
t||"");return/<!doctype\s+html/i.test(i)||/<html[\s>]/i.test(i)}o(isCanvasHtmlPreviewCandidate,"isCa\
nvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){const n=String(e&&e.lang?e.lang:"").trim(),
i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),a=!!(e&&e.open);return{...e,index:t,lang:n,code:i,
open:a,key:hashString(`${n||"TEXT"}
${i||""}`)}}o(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],a=[],r=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",m=[];for(const y of n){
if(!l){const k=y.match(r);if(k){l=k[2],c=String(k[3]||"").trim(),m=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const v=String(y||"").trim();if(v&&v.replace(/\s+/g,"")===l){const k=i[i.length-1];k&&(k.
code=m.join(`
`),k.open=!1),l=null,c="",m=[];continue}m.push(y);const w=i[i.length-1];w&&(w.code=m.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=m.join(`
`),y.open=!0)}const f=i.map((y,v)=>normalizeCanvasBlock(y,v)),b=selectCanvasPreviewBlock(f,t);return{
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
forEach(m=>{m&&m.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),r.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";r.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(r.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}o(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(m=>{const f=m.getAttribute("da\
ta-canvas-panel-view")===i;m.classList.toggle("active",f),m.setAttribute("aria-pressed",f?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}o(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const r=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",b=(String(i&&i.code?i.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${r}`,v=`${y}\u3001${b}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(v)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
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
null?i.code:""),m=r?isCanvasHtmlPreviewCandidate(l,c):!1,f=r?m?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",b=r?m?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=b),e.status&&(e.status.textContent=
f),e.previewLang&&(e.previewLang.textContent=r?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,v=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=v,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),r){canvasPreviewState.frameRenderToken+=1;const w=canvasPreviewState.frameRenderToken,
k=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),w);e.frame&&(e.frame.srcdoc=k,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{w!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:w,
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
(c,m)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const b=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},m);a.push(b),r===-1&&f.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
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
"`","~"])for(let c=3;c<=10;c++){const m=l.repeat(c);for(const f of r){const b=`${m}${f}
`,y=`
${m}`,v=b+a+y;i.includes(v)&&(i=i.split(v).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const r of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const m=l.repeat(c);
[`${r}${m}
${i}
${m}`,`${r}
${m}
${i}
${m}`,`
${r}${m}
${i}
${m}`,`
${r}
${m}
${i}
${m}`].forEach(b=>{n.includes(b)&&(n=n.split(b).join(`
`))})}return n}o(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}o(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}o(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[];if(!t)return{text:"",executions:n};const i=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let a=t.replace(i,(r,l,c)=>{const m=String(c||"").trim();try{const f=JSON.parse(m);n.push({code:f&&f.
code!=null?String(f.code):"",output:f&&f.output!=null?String(f.output):""})}catch{n.push({code:m,output:""})}
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
7"),m=encodeURIComponent(a).replace(/'/g,"%27"),f=hashString(`pyexec-detail
${i}
${a}
${t}`),b=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",y=`<button class="download-btn" data-\
code="${c}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,v=`<button class="coding-target-btn" data-code="${c}" data-code-key="${f}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${f}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(b)}</span><div class="code-actions">${v}${y}<button class="copy-btn" data-copy="code" \
data-code="${c}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${m}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
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
n.incrementalMath){const c=document.createElement("template");c.innerHTML=sanitizeMarkdownHtml(r.renderText,
{streamMathSegments:!0});const m=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(b=>{const y=b.getAttribute("data-stream-math-key");y&&m.set(y,b)});const f=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(b=>{const y=m.get(b.getAttribute(
"data-stream-math-key"));y?b.replaceWith(y):f.push(b)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(
e)),e.replaceChildren(c.content),wrapRenderedSvgBoxes(e),queueHighlight(e,r.renderText),queueIncrementalMathTypeset(
f);return}const l=collectSvgCodeRenders(e);e.innerHTML=sanitizeMarkdownHtml(r.renderText),restoreSvgCodeRenders(
e,l),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,r.renderText)}o(renderAiMarkdownInto,"renderA\
iMarkdownInto");function collectSvgCodeRenders(e){const t=new Map;return!e||typeof e.querySelectorAll!=
"function"||e.querySelectorAll(".svg-code-render[data-svg-key]").forEach(n=>{const i=n.getAttribute(
"data-svg-key");i&&!t.has(i)&&t.set(i,n)}),t}o(collectSvgCodeRenders,"collectSvgCodeRenders");function restoreSvgCodeRenders(e,t){
!e||!t||!t.size||typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-render[data-sv\
g-key]").forEach(n=>{const i=n.getAttribute("data-svg-key"),a=i?t.get(i):null;a&&(t.delete(i),n.replaceWith(
a))})}o(restoreSvgCodeRenders,"restoreSvgCodeRenders");function wrapRenderedSvgBoxes(e){!e||typeof e.
querySelectorAll!="function"||e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(
".svg-render-box")||t.closest("pre, code, .code-wrapper, .thought-container"))return;const n=document.
createElement("span");n.className="svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}
o(wrapRenderedSvgBoxes,"wrapRenderedSvgBoxes");function renderMessage(e,t,n,i,a,r,l=null,c=!0,m=null,f=null,b=null,y=null,v=null,w=null,k=null,S=null,C=!0,M=null,P=null,j=null){
const Z=t==="user",J=Z?"bg-blue-600":"bg-gray-700",Ce=Z?"justify-end":"justify-start";messageStore[e]=
n;const T=!Z&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},I=Z?n:T.text;let O=f;
if(O==null){const se=b!=null?Number(b):0,he=y!=null?Number(y):0;(b!=null||y!=null)&&(O=se+he)}messageMeta[e]=
{tokens_in:b,tokens_out:y,tokens_total:O,tokens_content:w,tokens_thought:k,is_encrypted:v,role:t,model:r,
parent_id:M,quote_text:m,image_url:i,gem_name:P,batch_job:j,python_executions:Z?[]:T.executions||[]};
let K="";m&&(K=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs text-gra\
y-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(m)}<\
/div>`);let X="";if(a&&!Z){let se="";try{se=JSON.parse(a).text||""}catch{se=a}se&&(X=`<div class="th\
ought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain t\
ext-purple-400"></i> Thinking Process</div><div class="thought-content collapsed">${escapeHtml(se)}<\
/div></div>`)}let ge="";if(i)try{const se=JSON.parse(i);if(se.length){const he=[];if(se.forEach(Y=>{
let ve=Y,pt="unknown";if(ve&&typeof ve=="object"&&(pt=normalizeAttachmentSource(ve.source),ve=ve.filepath||
ve.path||ve.url||ve.file||""),ve=normalizeAttachmentPath(ve)||ve,!ve)return;setAttachmentSourceForPath(
ve,pt);const ct=ve.replace(/^\d+\//,""),mt=buildFileUrl(ct),Ne=buildAttachmentPreviewUrl(ct),Ve=ve.split(
"/").pop(),bt=Ve.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(bt)?he.push(
buildChatImageHtml(Ne,{viewerSrc:mt,alt:Ve,title:Ve,filename:Ve})):he.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${mt}')" title="${Ve}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${Ve}</span></div>`)}),he.length>
0){let Y="grid-multi";he.length===1?Y="grid-1":he.length===2?Y="grid-2":he.length===3?Y="grid-3":he.
length===4&&(Y="grid-4"),ge=`<div class="image-grid ${Y}">${he.join("")}</div>`}}}catch{}const _e=Z?
"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}')"><i class="fas fa-rotate-right"></i>\
</button>`,ce=`<div class="msg-controls absolute -top-5 right-0 hidden group-hover:flex gap-1 z-10">\
<button class="ctrl-btn" onclick="window.copyMessage('${e}', this)"><i class="fas fa-copy"></i></but\
ton>${Z?`<button class="ctrl-btn edit-btn" data-id="${e}"><i class="fas fa-pen"></i></button>`:""}${_e}\
<button class="ctrl-btn" onclick="deleteMessage('${e}')"><i class="fas fa-trash"></i></button></div>`,
me=[];!Z&&r&&me.push(escapeHtml(r)),P&&(Z?me.push(`<span class="text-purple-300/90"><i class="fas fa\
-gem mr-0.5"></i>${escapeHtml(P)}</span>`):me.push(`<span class="text-purple-300/90"><i class="fas f\
a-gem mr-0.5"></i>${escapeHtml(P)}</span>`));const de=[];if(b!=null&&de.push(`In ${b}`),y!=null){let se=`\
Out ${y}`;k!=null&&Number(k)>0&&(se+=` (Thought ${k})`),de.push(se)}if(de.length||f!=null){const se=de.
length?de.join(" / "):`${f} tokens`;me.push(`<button class="underline decoration-dotted hover:text-w\
hite token-detail-btn" onclick="openTokenDetail('${e}')">${se}</button>`)}if(v!=null){const se=v?"fa\
-lock":"fa-lock-open",he=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":v?"Encrypted":
"Plain",Y=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-2\
00":"text-slate-300/80 hover:text-white";me.push(`<button class="${Y}" title="${he}" onclick="openEn\
cryptionSettings('${e}')"><i class="fas ${se}"></i></button>`)}if(!Z&&T.executions&&T.executions.length){
const se=T.executions.length,he=se>1?`Python \xD7${se}`:"Python";me.push(`<button type="button" clas\
s="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\
\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${he}</span></button>`)}const G=me.
length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${me.
join(" \u2022 ")}</div>`:"";let E;const H=!Z&&j?(()=>{const se=String(j.state||"").toUpperCase(),he=j.
status_text||(se==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
se==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${se==="JOB_STATE_FAILED"||
se==="JOB_STATE_CANCELLED"||se==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
se==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(he)}</div></div>`})():
"";Z?E=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(E=H+(I&&String(I).trim()?buildAiMarkdownHtml(I):j?'<div class="content-area prose prose\
-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(I)),E.includes("content-area")||(E=E.replace("prose ","content-area prose ")));let Q="";
if(l){const se=l.siblings[l.current-2],he=l.siblings[l.current];Q=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${se}\
)" ${se?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${he}\
)" ${he?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const te=c?"fade-in":"",ae=document.createElement("div");ae.className=`flex ${Ce} \
mb-4 ${te} relative message-group group`,ae.id=`msg-${e}`,ae.innerHTML=`<div class="message-bubble ${J}\
 text-white p-4 rounded-2xl shadow-md relative">${ce}${K}${X}${E}${ge}${Q}${G}</div>`;const Ae=S||get(
"chat-container");return Ae&&(Ae.appendChild(ae),C&&scrollToBottom(),Z||(queueMessageDecorations(ae,
I),syncCodingTargetButtons(ae),syncCodingModeUi(codingModeEnabled,{persist:!1}))),ae}o(renderMessage,
"renderMessage");function showTokenDetailModal(e=null){if(location.pathname!=="/token-details"){const t={
modal:"token-details"};e!==null&&(t.messageId=e),history.pushState(t,"","/token-details")}showModal(
"token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];
if(!t||!get("token-detail-modal"))return;const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",r=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",m=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detai\
l-in").innerText=a,get("token-detail-out").innerText=r,get("token-detail-content").innerText=l,get("\
token-detail-thought").innerText=c,get("token-detail-encrypted").innerText=m;const f=t.model?`${t.model}\
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
"click",y,!0)}o(c,"cleanup");function m(v){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(v)}o(m,"finalize");
function f(){m(!0)}o(f,"onOk");function b(){m(!1)}o(b,"onCancel");function y(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),b())}o(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",b),a&&a.addEventListener("click",b),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),m=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${r}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("\
chat-container");if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",m);else{
const b=document.createElement("div");b.innerHTML=m;const y=b.firstElementChild;y&&f.appendChild(y)}
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
var c,m;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((m=t.retryDelayMs)!=
null?m:180)||0),a=!!t.notifyOnFailure,r=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
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
 / none"}`,"info"),!1;const r=get(i.selectId),l=a.toLowerCase(),c=r?Array.from(r.options).find(m=>m.
value.toLowerCase()===l||m.textContent.trim().toLowerCase()===l):null;return!r||!c?(showToast(`${n.label}\
: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,"warning"),!1):
(r.value=c.value,r.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),showToast(
`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const r=a.toLowerCase(),
l={min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},c=parseSlashToggleArgument(
a),m=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(l,r)){m&&!m.checked&&!m.disabled&&(m.
checked=!0,m.dispatchEvent(new Event("change",{bubbles:!0})));const f=get("thinking-level");return f&&
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
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((v,w)=>{const k=document.createElement(
"div");k.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===
slashSelectedIndex?"bg-gray-700":""}`,k.innerHTML=`
                    <i class="fas ${v.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${v.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${v.description}</div>
                    </div>
                `;let S=!1;k.addEventListener("pointerdown",C=>{typeof C.button=="number"&&C.button!==
0||(C.preventDefault(),S=!0,selectSlashCommand(v.id))}),k.addEventListener("click",C=>{C.preventDefault(),
S||selectSlashCommand(v.id)}),k.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(k)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,m=r.top,f=260,
b=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&m>c){const v=Math.min(f,m-b);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(f,c-b);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${v}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}o(showSlashCommandSuggestions,
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
return;removeEphemeralMessageControls(c);const m=c.querySelector(".message-bubble");if(!m||!i.length)
return;const f=document.createElement("div");f.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,v])=>{const w=AI_SETTING_JUMP_TARGETS[y]||{label:y},k=document.createElement("button");k.
type="button",k.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const S=document.createElement("span");S.className="min-w-0 flex-1";const C=document.createElement(
"span");C.className="block text-xs font-bold text-blue-200",C.textContent=w.label;const M=document.createElement(
"span");M.className="block mt-0.5 text-[11px] text-gray-300 break-words",M.textContent=formatAiSettingValue(
v);const P=document.createElement("i");P.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",S.appendChild(C),S.appendChild(M),k.appendChild(S),k.appendChild(P),k.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(k)});const b=m.querySelector(".message-footer-\
meta");b?m.insertBefore(f,b):m.appendChild(f),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const r=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${r}" class="flex justify-start mb-4 ai-pending-row f\
ade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-\
tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const m=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),f=get(r);if(f&&f.remove(),m&&m.status==="ok"&&m.mode==="inspect"&&m.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(m.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
m.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(m.current,t,"inspect");
return}if(m&&m.status==="ok"&&m.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
m.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(m.applied).length}\u9805\u76EE\uFF09`,"success");try{const v=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(v),cacheUserSettings(v)}catch{}renderAiSettingsResultBubble(
m.applied,t);return}const b=m.message||m.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${b}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${b}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(b,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const m=get(r);m&&m.remove();const f=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(f),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
o(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}o(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const a=e.toLowerCase(),r=loadedGems.filter(w=>w.name.toLowerCase().includes(a)||w.description&&
w.description.toLowerCase().includes(a));if(r.length===0){hideGemSuggestions();return}gemSelectedIndex>=
r.length&&(gemSelectedIndex=0),n.innerHTML="",r.forEach((w,k)=>{const S=document.createElement("div");
S.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${k===gemSelectedIndex?
"bg-gray-700":""}`,S.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(w.name)}</div>
                        ${w.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
w.description)}</div>`:""}
                    </div>
                `,S.onclick=()=>selectGemSuggestion(w),S.onmouseenter=()=>{gemSelectedIndex=k,showGemSuggestions(
e)},n.appendChild(S)});const l=i.getBoundingClientRect(),c=window.innerHeight,m=c-l.bottom,f=l.top,b=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",m<180&&f>m){const w=Math.min(b,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(b,m-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}o(showGemSuggestions,
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
PythonBox");function collectBrowserFastPromptOptions(){const e=o((n,i)=>get(n)?!!get(n).checked:i,"c\
hecked"),t=o(n=>get(n)?get(n).value:null,"value");return{canvas_mode:!!canvasModeEnabled,enable_search:e(
"enable-search",!1),enable_url_context:e("enable-url-context",!1),enable_maps:e("enable-maps",!1),enable_python:e(
"enable-python",!1),enable_file_creation:e("enable-file-creation",!0),enable_mcp:typeof isMcpEnabledForSend==
"function"?!!isMcpEnabledForSend():!1,enable_thinking:e("enable-thinking",!1),thinking_level:t("thin\
king-level"),thinking_budget:t("thinking-budget"),reasoning_effort:t("reasoning-effort"),safety_setting:t(
"safety-setting"),enable_prompt_caching:e("enable-prompt-cache",!1)}}o(collectBrowserFastPromptOptions,
"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e){const t=String(get("mode\
l-select").value||"").trim(),n=await fetchBrowserFastBootstrap(!1);if(!browserFastApiKey||browserFastApiKeyModel!==
t)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
const i=Array.from(browserFastLocalFiles.values()),a=[];for(const T of i)a.push({inlineData:{mimeType:T.
file.type,data:await fileToBase64Payload(T.file)}});a.push({text:e});const r={},l=browserFastThinkingConfig(
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
abortController=new AbortController;let y="",v="";const w=[];let k=null,S=null,C=!1;const M={},P=[];
let j=null,Z="";const J=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;
let Ce=!1;try{const T=await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
t)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"\
application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(c),signal:abortController.
signal}));if(!T.ok){const de=await T.json().catch(()=>({}));throw new Error(de&&de.error&&de.error.message?
de.error.message:`Gemini API HTTP ${T.status}`)}window.ConnectionMonitor&&(Ce=!0,window.ConnectionMonitor.
operationStarted()),J&&J.setPhase("waiting"),get("prompt-input").value="",get("prompt-input").style.
height="auto";const I=T.body.getReader(),O=new TextDecoder;let K="";const X=o(de=>{const G=de.split(
/\r?\n/).filter(Q=>Q.startsWith("data:")).map(Q=>Q.slice(5).trim()).join("");if(!G||G==="[DONE]")return;
const E=JSON.parse(G);if(E.error)throw new Error(E.error.message||"Gemini API error");if((Array.isArray(
E.candidates)?E.candidates:[]).forEach(Q=>{(Q&&Q.content&&Array.isArray(Q.content.parts)?Q.content.parts:
[]).forEach(ae=>{if(ae&&typeof ae.thoughtSignature=="string"&&!w.includes(ae.thoughtSignature)&&w.push(
ae.thoughtSignature),ae&&ae.executableCode&&typeof ae.executableCode.code=="string"){const se=ae.executableCode.
code;y+=`
\`\`\`python
${se}
\`\`\`
`,j=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,Z=se,M[j]||(b.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(j)),M[j]=b.querySelector(`[data-py-id="${j}"]`)),updateBrowserFastPythonBox(
M[j],"code",se);return}if(ae&&ae.codeExecutionResult&&typeof ae.codeExecutionResult.output=="string"){
const se=ae.codeExecutionResult.output;y+=`
**Output:**
\`\`\`
${se}
\`\`\`
`;const he=j||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;P.push({code:Z||
"",output:se}),M[he]||(b.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(he)),M[he]=b.querySelector(
`[data-py-id="${he}"]`)),updateBrowserFastPythonBox(M[he],"output",se);return}const Ae=typeof ae.text==
"string"?ae.text:"";Ae&&(ae.thought===!0?v+=Ae:y+=Ae)})}),!C&&(y||v)){beginPendingToStreamTransition(
b);const Q=b.querySelector(".content-area");Q&&Q.remove(),C=!0}v&&(S||(b.insertAdjacentHTML("afterbe\
gin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></\
div>'),S=b.querySelector(".thought-content")),S.textContent=v),y&&(k||(k=document.createElement("div"),
k.className="content-area prose prose-invert text-sm break-words",b.appendChild(k)),renderAiMarkdownInto(
k,y,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:de,value:G}=await I.
read();if(de)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),J&&J.setPhase(
"receiving"),K+=O.decode(G,{stream:!0});const E=K.split(/\r?\n\r?\n/);K=E.pop()||"",E.forEach(X)}if(K+=
O.decode(),K.trim()&&X(K),!y.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
k&&renderAiMarkdownInto(k,y,{incrementalMath:!0}),S&&S.classList.add("collapsed"),P.length&&(y+=P.map(
de=>`
\`\`\`pyexec
${JSON.stringify(de)}
\`\`\`
`).join("")),i.length&&(J&&J.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),await uploadBrowserFastLocalFiles()),J&&J.setPhase("saving");const ge=collectImageUrlsForSend(),
_e=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:y,thought_content:v,model:t,image_urls:ge,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:n.parent_id||null,thought_signatures:w,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),b),ce=await _e.json().
catch(()=>({}));if(!_e.ok||!ce.thread_id)throw new Error(ce.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const me=!currentThreadId;currentThreadId=String(ce.thread_id),currentParentId=ce.assistant_message_id||
null,currentLeafId=ce.assistant_message_id||null,resetUploadState(),browserFastBootstrap=null,await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!me}),applyBrowserFastModeRestrictions(),loadThreads(
!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(T){if(T.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${T.message}`,
"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const I=T.message||"\u30A8\u30E9\u30FC";
b&&b.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(I));try{let O=y||"";P.length&&(O+=P.map(
ce=>`
\`\`\`pyexec
${JSON.stringify(ce)}
\`\`\`
`).join(""));const K=buildChatErrorMarkdown(I,O),X=i.length?[]:collectImageUrlsForSend(),ge=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:K,
thought_content:v||"",model:t,image_urls:X,prompt_options:collectBrowserFastPromptOptions(),temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:n&&n.parent_id?n.parent_id:null,thought_signatures:w,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),b),_e=await ge.
json().catch(()=>({}));if(ge.ok&&_e.thread_id){const ce=!currentThreadId;currentThreadId=String(_e.thread_id),
currentParentId=_e.assistant_message_id||null,currentLeafId=_e.assistant_message_id||null,resetUploadState(),
browserFastBootstrap=null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!ce}),
applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(O){sendClientDebugLog("error",`Browser fas\
t error persist failed: ${O&&O.message?O.message:O}`)}}}finally{Ce&&window.ConnectionMonitor&&window.
ConnectionMonitor.operationEnded(),J&&J(),setSendBtnToSendMode(),activeStreamingBubbleId===f&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
var Ft;if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const D=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(D);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let e=null;if(isBotDetectionActive()){if(e=await getTurnstileToken(),!e&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}e=await getTurnstileToken()}if(!e&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}e&&await verifyTurnstileOnServer(e)}const t=get("prompt-inp\
ut").value;if(pendingSlashCommand){const D=pendingSlashCommand,ue=t.trim(),Ie=get("model-select")?get(
"model-select").value:null;if(D==="settings"){if(!ue){showToast("\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini\
-2.5-flash\u306B\uFF09","info"),get("prompt-input").focus();return}if(!Ie){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
ue,Ie)}else executeMinimalSlashCommand(D,ue)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const n=t.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(n&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(D=>D.id===
n[1].toLowerCase())){executeMinimalSlashCommand(n[1].toLowerCase(),n[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const i=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(i&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(i)setBrowserFastModeEnabled(!1);else{const D=browserFastModeIneligibility(
t);if(!D){try{await sendBrowserFastMessage(t)}catch(ue){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${ue.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${D}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(ue){showToast(
ue.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}t.trim()&&(promptHistory.length===
0||promptHistory[0]!==t)&&(promptHistory.unshift(t),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const a=collectAttachmentItemsForSend(),r=a.map(D=>D.path),l=a.filter(D=>normalizeAttachmentSource(
D.source)==="upload").map(D=>D.path),c=getModelMediaSupport(get("model-select").value),m=r.some(D=>isAudioPath(
D)),f=r.some(D=>isVideoPath(D)),b=(get("model-select").value||"").toLowerCase(),y=get("enable-python"),
v=!!(y&&y.checked);if(m&&!c.audio||f&&!c.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!t.trim()&&r.length===0)return;if(isMistralOcrModel(
b)){const D=/https?:\/\/\S+/i.test(t);if(r.filter(Ie=>isAudioPath(Ie)||isVideoPath(Ie)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!r.length&&!D){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const w=t.trim();if(/^\/settings(?:\s|$)/i.
test(w)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(w)){const D=w.replace(/^\/settings\s*/i,"").trim();
if(!D){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ie=get("prompt-input");Ie.value="/settings ";const Re=extractSlashCommandToken(Ie.value);
lastSlashFilter=Re,showSlashCommandSuggestions(Re),Ie.focus();return}const ue=get("model-select")?get(
"model-select").value:null;if(!ue){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
D,ue);return}if(isGeminiLocalPythonMode(b,m,f,v)&&!await confirmGeminiLocalPythonSwitch())return;let k=null,
S=[];if(codingModeEnabled){const D=collectCodingCandidates(t),ue=D.filter(qe=>qe.prompt_source),Ie=D.
filter(qe=>!qe.prompt_source),Re=ue.reduce((qe,je)=>qe+String(je.code||"").length,0);if(Re>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let Ke=3e5-Re;const tt=[];for(let qe=Ie.length-1;qe>=0;qe--){const je=String(Ie[qe].
code||"").length;je>Ke||(tt.unshift(Ie[qe]),Ke-=je)}S=codingTargetSelection?tt.slice(-1):[...ue,...tt];
const rt=ue.length?ue[ue.length-1]:null;if(k=codingTargetSelection?S[0]:rt||S[S.length-1]||null,codingModeEffective=
!!(k&&String(k.code||"").trim()),codingModeEffective&&k.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const qe=String(((Ft=get(
"model-select"))==null?void 0:Ft.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(qe)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const C=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${t.length}\
 attachments=${r.length} search=${get("enable-search").checked}`);const M=t,P=hasMarkerHint()?MARKER_HINT_TEXT:
null;if(isGptImageModel()&&currentMaskImage&&r.length===0){showToast("Mask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=editingMessageId,Z=currentParentId,J=j!=null;j&&(editingMessageId=null,setEditUi(
!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const Ce=[],T=o(D=>{if(D==null)
return;let ue=document.getElementById(`msg-${D}`);for(;ue;)ue.classList&&ue.classList.contains("mess\
age-group")&&(Ce.push({node:ue,prevDisplay:ue.style.display}),ue.style.display="none"),ue=ue.nextElementSibling},
"hideRenderedBranchFrom"),I=o(()=>{Ce.forEach(({node:D,prevDisplay:ue})=>{D&&(D.style.display=ue||"")}),
Ce.length=0},"restoreHiddenBranch");j&&T(j);const O=Date.now(),K=renderMessage(O,"user",M,JSON.stringify(
r),null,null,null,!0,currentQuote,null,null,null,null,null,null,null,!0,Z,activeGem?activeGem.name:null);
let X=!1;const ge=/(https?:\/\/)?(x\.com|twitter\.com)\//i,_e=ge.test(M||"")||ge.test(currentQuote||
""),ce="grok-4-fast-reasoning",me=o(()=>{get("enable-search").checked=!0,get("model-select").value!==
ce&&selectModelById(ce)},"applyXLinkAuto");if(_e&&!isMistralOcrModel()&&!get("enable-search").checked)
if(autoSearchOnLinks)me();else{const D=get("auto-search-banner"),ue=get("auto-search-on-btn"),Ie=get(
"auto-search-off-btn"),Re=get("auto-search-remember");D&&ue&&Ie&&(Re&&(Re.checked=!1),await new Promise(
Ke=>{D.classList.remove("hidden");const tt=o(rt=>{D.classList.add("hidden"),ue.onclick=null,Ie.onclick=
null,Ke(rt)},"cleanup");ue.onclick=()=>tt("enable"),Ie.onclick=()=>tt("disable")}).then(async Ke=>{Ke===
"enable"?(me(),Re&&Re.checked&&(autoSearchOnLinks=!0,await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({auto_search_on_links:!0})}))):
X=!0}))}const de=String(get("reasoning-effort").value||"").toLowerCase(),G=String(get("model-select").
value||"").toLowerCase().includes("deepseek")&&de==="none",E={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:M,model:get("model-select").value,image_urls:r,image_items:a,uploaded_image_urls:l,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:G?!1:get("enable-thin\
king").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:P,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:Z,parent_id_explicit:J,disable_auto_search:X,
image_vision_model:currentVisionModel||null,coding_mode:C,coding_target:C?{id:k.candidate_id,code:k.
prompt_source?null:k.code,language:k.language||"text",key:k.key||null,message_id:k.message_id||null,
source:k.prompt_source?"prompt":"history",explicit:k.explicit===!0}:null,coding_candidates:C?S.map(D=>({
id:D.candidate_id,source:D.prompt_source?"prompt":"history",prompt_index:D.prompt_source?D.prompt_index:
null,code:D.prompt_source?null:D.code,language:D.language||"text",explicit:D.explicit===!0})):[],batch_mode:i,
canvas_mode:!!canvasModeEnabled};e&&(E.turnstile_token=e);const H=get("thread-custom-instruction");H&&
(E.thread_custom_instruction=H.value||""),activeGem?(E.system_prompt=activeGem.instruction,E.enable_system_prompt=
!0,E.gem_uuid=activeGem.uuid):E.gem_uuid=null,setSendBtnToStopMode();const Q="ai-"+Date.now(),te=String(
E.model||"").toLowerCase(),ae=!!E.enable_thinking||!!de&&de!=="none",Ae=te.includes("gemini")||te.includes(
"o1")||te.includes("o3")||te.includes("gpt-5")||te.includes("reasoning")&&!te.includes("non-reasonin\
g"),se=ae&&Ae;let he=buildPendingSkeletonHtml(E.model,"API\u306B\u9001\u4FE1\u4E2D...");get("chat-co\
ntainer").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending-row fade-in\
"><div id="${Q}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl roun\
ded-tl-none shadow-md relative">${he}</div></div>`),resumeChatAutoScroll();const Y=get(Q);activeStreamingBubbleId=
Q,canvasModeEnabled&&resetCanvasPreviewPanel();let ve=null;const pt=o(D=>!se||!Y?null:((!ve||!Y.contains(
ve))&&(ve=Y.querySelector(".thought-content")),ve||(Y.insertAdjacentHTML("afterbegin",'<div class="t\
hought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i cla\
ss="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" \
data-placeholder="1"></div></div>'),ve=Y.querySelector(".thought-content")),ve&&(ve.setAttribute("da\
ta-placeholder","1"),ve.textContent=D||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
ve),"ensureThoughtPlaceholder");se&&pt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const ct=currentThreadId,mt=nowPerfMs(),Ne=Date.now();let Ve=!1,
bt=!1,Mt=!1,wt=null,At=null,yt=null,Rt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Et=o((D,ue)=>{if(!ue||D==="status"&&Ve||D==="thought"&&bt||D==="content"&&Mt)return;const Ie=Math.
max(0,nowPerfMs()-mt);D==="status"?wt=Ie:D==="thought"?At=Ie:D==="content"&&(yt=Ie),reportFirstTokenLatency(
{latency_seconds:Ie/1e3,latency_ms:Ie,thread_id:Rt||currentThreadId,job_id:currentJobId,model:E.model,
first_event_type:D,client_sent_at_ms:Ne}),D==="status"?Ve=!0:D==="thought"?bt=!0:D==="content"&&(Mt=
!0)},"maybeReportFirstEventLatency"),ft=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let Bt=!1,Jt=!1,xt=null,vt=null,Xt=!1;try{E.thread_id&&activeGem&&(threadGemMap[E.thread_id]=
activeGem,pendingGemForNewThread=null);const D=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(E),signal:abortController.signal}),Y);if(sendClientDebugLog("info",`Prompt strea\
m response status: ${D.status}`),!D.ok){const Te=await D.json().catch(()=>({})),De=new Error(Te.error||
`HTTP ${D.status}`);throw De.serverCode=Te.code||null,De.serverModel=Te.model||E.model,De.acceptedJobId=
Te.job_id||null,De.acceptedThreadId=Te.thread_id||null,De}Bt=!0,window.ConnectionMonitor&&(Xt=!0,window.
ConnectionMonitor.operationStarted()),ft&&ft.setPhase("waiting"),get("prompt-input").value="",get("p\
rompt-input").style.height="auto",schedulePromptTokenEstimate(!0),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1}),resetUploadState(),clearQuote();const ue=o(()=>{if(!Y)return;const Te=Y.querySelector(
".content-area");if(Te&&Te.getAttribute("data-api-accepted")!=="1"&&(Te.setAttribute("data-api-accep\
ted","1"),!updatePendingSkeletonStatus(Y,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...",
"\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059"))){
Te.outerHTML=buildPendingSkeletonHtml(E.model,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...");
const De=Y.querySelector(".content-area");De&&De.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
Y,"\u63A5\u7D9A\u5B8C\u4E86\u3002\u30E2\u30C7\u30EB\u5FDC\u7B54\u3092\u5F85\u6A5F\u4E2D...","\u30AD\u30E5\u30FC\u5F85\u6A5F\u3084\u521D\
\u671F\u5316\u4E2D\u306E\u53EF\u80FD\u6027\u304C\u3042\u308A\u307E\u3059")}},"markApiAccepted");ue();
const Ie=D.body.getReader(),Re=new TextDecoder;let Ke="",tt="",rt="",qe=!0,je=null,Ze=null,We=null,gt=!1;
const jt={};let Yt=0,Qt=!1;for(;!Qt;){const{done:Te,value:De}=await Ie.read();if(Te)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ft&&ft.setPhase("receiving"),Ke+=Re.decode(De,{stream:!0});
let Je=Ke.split(`
`);Ke=Je.pop();let un=!1,pn=!1;for(let _t of Je)if(_t.trim())try{const be=JSON.parse(_t);if(be.type===
"thread_id"){ue();const Se=be.content!==null&&be.content!==void 0?String(be.content):be.content;Se&&
(Rt=Se,currentThreadId!==Se&&(currentThreadId=Se,history.pushState({},"","/c/"+Se)),activeGem&&(threadGemMap[Se]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(be.type==="job_\
id"){ue(),currentJobId=be.content,i&&showToast("Batch\u767B\u9332","info");continue}if(be.type==="se\
arch_status"){be.content==="searching"&&!We?(Y.insertAdjacentHTML("afterbegin",'<div class="search-b\
ox visible animate-pulse mb-2"><i class="fas fa-globe"></i> Searching web...</div>'),We=Y.querySelector(
".search-box")):be.content==="done"&&We&&(We.classList.remove("animate-pulse"),We.innerHTML='<i clas\
s="fas fa-check-circle text-green-400"></i> Search complete',setTimeout(()=>{We&&We.remove(),We=null},
2e3));continue}if(be.type==="mcp"){handleMcpStreamEvent(Y,be.content||{});continue}if(be.type==="mcp\
_decision_request"){openMcpDecisionModal(be.content||{});continue}if(be.type==="status"){ue();const Se=be.
content===null||be.content===void 0?"":String(be.content);if(Et("status",!!Se),qe&&Y){const Xe=Se||"\
\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(Y,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ot=Y.querySelector(".content-area");ot&&(ot.outerHTML=buildPendingSkeletonHtml(E.model,Xe),updatePendingSkeletonStatus(
Y,Xe,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
se&&pt(Se||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(qe){beginPendingToStreamTransition(
Y);const Se=Y.querySelector(".content-area");Se&&(Se.innerHTML=""),qe=!1}if(be.type==="coding_diff")
appendCodingLiveDiff(Y,be.content||{}),Et("content",!0);else if(be.type==="thought"){if(je||(je=Y.querySelector(
".thought-content")),rt+=be.content,Et("thought",!!be.content),!je){const Se='<div class="thought-co\
ntainer"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purp\
le-400"></i> Thinking Process</div><div class="thought-content"></div></div>';We?We.insertAdjacentHTML(
"afterend",Se):Y.insertAdjacentHTML("afterbegin",Se),je=Y.querySelector(".thought-content")}if(je&&je.
getAttribute("data-placeholder")==="1"){if(je.textContent="",je.removeAttribute("data-placeholder"),
je){const Se=je.parentElement.querySelector(".thought-header");Se&&Se.classList.remove("thinking-shi\
mmer")}rt=be.content}je.classList.remove("collapsed"),pn=!0}else if(be.type==="image_analysis"){const Se=be.
content===null||be.content===void 0?"":String(be.content);if(!Y)continue;let Xe=Y.querySelector(".im\
age-analysis-box");if(!Xe){const nt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border b\
order-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas fa-\
image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></div\
></div>';We?We.insertAdjacentHTML("afterend",nt):Y.insertAdjacentHTML("afterbegin",nt),Xe=Y.querySelector(
".image-analysis-box")}const ot=Xe.querySelector(".image-analysis-text");ot&&(ot.textContent=Se)}else if(be.
type==="python"){const Se=be.content||{},Xe=Se.id||`py_${Date.now()}`;if(!jt[Xe]){const nt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Xe}" data-collapsed="true" data-code-key="${Xe}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;We?We.insertAdjacentHTML("afterend",nt):Y.insertAdjacentHTML(
"afterbegin",nt),jt[Xe]=Y.querySelector(`[data-py-id="${Xe}"]`)}const ot=jt[Xe];if(ot){if(Se.code!==
void 0){const nt=Se.code==null?"":String(Se.code),ht=ot.querySelector(".python-code");ht&&(ht.textContent=
nt,ht.removeAttribute("data-highlighted"),queueHighlight(ot,nt));const St=ot.querySelector('.copy-bt\
n[data-copy="code"]');St&&St.setAttribute("data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}if(Se.
output!==void 0){const nt=Se.output==null?"":String(Se.output),ht=ot.querySelector(".python-output");
ht&&(ht.textContent=nt);const St=ot.querySelector('.copy-btn[data-copy="output"]');St&&St.setAttribute(
"data-code",encodeURIComponent(nt).replace(/'/g,"%27"))}}}else if(be.type==="content"){const Se=be.content===
null||be.content===void 0?"":String(be.content);tt+=Se,/[`~]/.test(Se)&&activateDeferredCodingModeFromStream(
tt),Ze||(Ze=Y.querySelector(".content-area")||document.createElement("div"),Ze.className="prose pros\
e-invert text-sm break-words",Y.contains(Ze)||Y.appendChild(Ze)),un=!0,Et("content",!!Se)}else if(be.
type==="error"){gt=!0,Qt=!0,Y.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(be.content)),showToast(
be.content||"Unknown error","error",!0);break}}catch{}if(pn&&je&&(je.textContent=rt,userAutoScroll&&
(je.scrollTop=je.scrollHeight)),un&&Ze){const _t=Date.now();if(_t-Yt>100){const be=snapshotCodeCollapse(
Ze);renderAiMarkdownInto(Ze,tt,{incrementalMath:!0}),applyCodeCollapse(Ze,be,!0),Yt=_t}}scrollToBottom()}
if(ft&&ft(),Ze){const Te=snapshotCodeCollapse(Ze);renderAiMarkdownInto(Ze,tt,{incrementalMath:!0}),applyCodeCollapse(
Ze,Te,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),Y)if(queueHighlight(Y,tt),enableLatencyMetrics){
const Te=nowPerfMs()-mt;reportFirstTokenLatency({is_total:!0,latency_seconds:Te/1e3,latency_ms:Te,thread_id:Rt||
currentThreadId,job_id:currentJobId,model:E.model,client_sent_at_ms:Ne,client_done_at_ms:Date.now()});
let De='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',Je=null;wt!==null&&(Je=wt),At!==null&&(Je===null||At<Je)&&(Je=
At),yt!==null&&(Je===null||yt<Je)&&(Je=yt),Je!==null&&(De+=`<div>Initial: ${(Je/1e3).toFixed(2)}s</d\
iv>`),yt!==null&&yt!==Je&&(De+=`<div>Content: ${(yt/1e3).toFixed(2)}s</div>`),De+=`<div class="font-\
bold text-gray-300">Total: ${(Te/1e3).toFixed(2)}s</div>`,currentJobId&&(De+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),De+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,De+="</div>",Y.insertAdjacentHTML("beforeend",De)}else Y.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),Y&&Y.querySelectorAll(".thought-co\
ntent").forEach(De=>De.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!j}),!gt&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(Te=>Te.json()).then(Te=>{Te.title&&
(document.title=Te.title+" - AI Chat",setCurrentChatHeaderTitle(Te.title),loadThreads())}):loadThreads(
!1)}catch(D){let ue=!1;const Ie=D.name==="AbortError"&&isManualStopAbortForThread(ct);if(D.name==="A\
bortError"&&!Ie&&(ue=await syncThreadAfterAbortedStream(ct,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${D.message}`),!Bt){K&&K.remove();const Re=Y&&Y.closest(
".fade-in");Re&&Re.remove(),delete messageStore[O],delete messageMeta[O]}if(D.serverCode==="request_\
already_accepted"&&D.acceptedJobId&&D.acceptedThreadId)Bt=!0,xt={job_id:D.acceptedJobId,thread_id:String(
D.acceptedThreadId),model:E.model},get("prompt-input").value="",get("prompt-input").style.height="au\
to",resetUploadState(),clearQuote();else if(Bt&&!Ie)vt={job_id:normalizeJobIdForUi(currentJobId),thread_id:currentThreadId!=
null?String(currentThreadId):null,model:E.model},window.ConnectionMonitor.setUnavailable("offline"),
showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(D.serverCode==="turnstile_required"){const Re=await getTurnstileToken();Re?(await verifyTurnstileOnServer(
Re,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(D.serverCode==="api_key_missing"){const Re=D.serverModel||E.model,Ke=await showApiKeyRequiredModalAsync(
Re);Ke==="set"?Jt=!0:Ke==="switch"?showModal("model-modal"):showToast(D.message||`${getModelNameById(
Re)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(D.
name!=="AbortError"){const Re="Connection Error: "+D.message;showToast(Re,"error",!0)}j&&!ue&&I()}finally{
Xt&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),ft&&ft(),setSendBtnToSendMode(),
updateFilePreview(),activeStreamingBubbleId===Q&&(activeStreamingBubbleId=null),abortController=null,
currentJobId=null,editingMessageId=null,setEditUi(!1)}if(xt){const D=currentThreadId!=null?String(currentThreadId):
null;return currentThreadId=xt.thread_id,(D!==currentThreadId||location.pathname!=="/c/"+currentThreadId)&&
history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(xt,currentThreadId)}
if(vt&&vt.thread_id)return reconnectPendingStreamUntilAvailable(vt,vt.thread_id);if(Jt)return sendMessage()}
o(sendMessage,"sendMessage");async function resumePendingStream(e){if(abortController||!e||!e.job_id||
!currentThreadId||isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?
String(e.model):"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)
return;if(activeStreamingBubbleId=n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-\
area.skeleton-pending")){const I=a.querySelector(".content-area");I?I.outerHTML=buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(
i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const r=currentThreadId,
l=i.toLowerCase(),c=l.includes("gemini")||l.includes("o1")||l.includes("o3")||l.includes("gpt-5")||l.
includes("reasoning")&&!l.includes("non-reasoning");let m=null;const f=o(I=>!c||!a?null:((!m||!a.contains(
m))&&(m=a.querySelector(".thought-content")),m||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),m=a.querySelector(".thought-content")),m&&(m.setAttribute("data-pla\
ceholder","1"),m.textContent=I||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),m),
"ensureThoughtPlaceholder");c&&f("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let b="",y="",v="",w=!0,k=null,S=null,C=null,M=!1;const P={};let j=0,Z=!1;const J=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let Ce=!1,T=!1;try{const I=await apiFetch("/chat\
_stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!I.ok)throw new Error(`Resume failed (${I.status})`);window.ConnectionMonitor&&
(T=!0,window.ConnectionMonitor.operationStarted()),J&&J.setPhase("waiting");const O=I.body.getReader(),
K=new TextDecoder;for(;!Z;){const{done:X,value:ge}=await O.read();if(X)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),J&&J.setPhase("receiving"),b+=K.decode(ge,{stream:!0});let _e=b.
split(`
`);b=_e.pop();let ce=!1,me=!1;for(let de of _e)if(de.trim())try{const G=JSON.parse(de);if(G.type==="\
job_id"){currentJobId=G.content||t;continue}if(G.type==="search_status"){G.content==="searching"&&!C?
(a.insertAdjacentHTML("afterbegin",'<div class="search-box visible animate-pulse mb-2"><i class="fas\
 fa-globe"></i> Searching web...</div>'),C=a.querySelector(".search-box")):G.content==="done"&&C&&(C.
classList.remove("animate-pulse"),C.innerHTML='<i class="fas fa-check-circle text-green-400"></i> Se\
arch complete',setTimeout(()=>{C&&C.remove(),C=null},2e3));continue}if(G.type==="mcp"){handleMcpStreamEvent(
a,G.content||{});continue}if(G.type==="mcp_decision_request"){openMcpDecisionModal(G.content||{});continue}
if(G.type==="status"){const E=G.content===null||G.content===void 0?"":String(G.content);if(w&&a){const H=E||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(a,H,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Q=a.querySelector(".content-area");Q&&(Q.outerHTML=buildPendingSkeletonHtml(i,H),updatePendingSkeletonStatus(
a,H,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
c&&f(E||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(w){beginPendingToStreamTransition(
a);const E=a.querySelector(".content-area");E&&(E.innerHTML=""),w=!1}if(G.type==="coding_diff")appendCodingLiveDiff(
a,G.content||{});else if(G.type==="thought"){if(k||(k=a.querySelector(".thought-content")),v+=G.content,
!k){const E='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';C?C.insertAdjacentHTML("afterend",E):a.insertAdjacentHTML("afterbegin",E),k=a.querySelector(
".thought-content")}if(k&&k.getAttribute("data-placeholder")==="1"){if(k.textContent="",k.removeAttribute(
"data-placeholder"),k){const E=k.parentElement.querySelector(".thought-header");E&&E.classList.remove(
"thinking-shimmer")}v=G.content}k.classList.remove("collapsed"),me=!0}else if(G.type==="image_analys\
is"){const E=G.content===null||G.content===void 0?"":String(G.content);if(!a)continue;let H=a.querySelector(
".image-analysis-box");if(!H){const te='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 borde\
r border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas \
fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></\
div></div>';C?C.insertAdjacentHTML("afterend",te):a.insertAdjacentHTML("afterbegin",te),H=a.querySelector(
".image-analysis-box")}const Q=H.querySelector(".image-analysis-text");Q&&(Q.textContent=E)}else if(G.
type==="python"){const E=G.content||{},H=E.id||`py_${Date.now()}`;if(!P[H]){const te=`<div class="co\
de-wrapper python-box collapsed" data-py-id="${H}" data-collapsed="true" data-code-key="${H}"><div c\
lass="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span><d\
iv class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"\
><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="" \
title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" da\
ta-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></\
button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Code\
</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-sect\
ion"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output">\
</code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",te):a.insertAdjacentHTML("afterbe\
gin",te),P[H]=a.querySelector(`[data-py-id="${H}"]`)}const Q=P[H];if(Q){if(E.code!==void 0){const te=E.
code==null?"":String(E.code),ae=Q.querySelector(".python-code");ae&&(ae.textContent=te,ae.removeAttribute(
"data-highlighted"),queueHighlight(Q,te));const Ae=Q.querySelector('.copy-btn[data-copy="code"]');Ae&&
Ae.setAttribute("data-code",encodeURIComponent(te).replace(/'/g,"%27"))}if(E.output!==void 0){const te=E.
output==null?"":String(E.output),ae=Q.querySelector(".python-output");ae&&(ae.textContent=te);const Ae=Q.
querySelector('.copy-btn[data-copy="output"]');Ae&&Ae.setAttribute("data-code",encodeURIComponent(te).
replace(/'/g,"%27"))}}}else if(G.type==="content"){const E=G.content===null||G.content===void 0?"":String(
G.content);y+=E,/[`~]/.test(E)&&activateDeferredCodingModeFromStream(y),S||(S=a.querySelector(".cont\
ent-area")||document.createElement("div"),S.className="prose prose-invert text-sm break-words",a.contains(
S)||a.appendChild(S)),ce=!0}else if(G.type==="error"){M=!0,Z=!0,a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(
G.content)),showToast(G.content||"Unknown error","error",!0);break}}catch{}if(me&&k&&(k.textContent=
v,userAutoScroll&&(k.scrollTop=k.scrollHeight)),ce&&S){const de=Date.now();if(de-j>100){const G=snapshotCodeCollapse(
S);renderAiMarkdownInto(S,y,{incrementalMath:!0}),applyCodeCollapse(S,G,!0),j=de}}scrollToBottom()}if(J&&
J(),S){const X=snapshotCodeCollapse(S);renderAiMarkdownInto(S,y,{incrementalMath:!0}),applyCodeCollapse(
S,X,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,y),a&&a.querySelectorAll(".thought-content").
forEach(ge=>ge.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}),
loadThreads(!1)}catch(I){const O=I.name==="AbortError"&&isManualStopAbortForThread(r);I.name==="Abor\
tError"&&!O&&await syncThreadAfterAbortedStream(r,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),O||
(Ce=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{T&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),J&&J(),
setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=null),
abortController=null,currentJobId=null,currentThreadPending=null}if(Ce)return reconnectPendingStreamUntilAvailable(
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
eplace-settings-open");return}const c=a&&Array.isArray(a.threads)?a.threads.length:-1,m=r.querySelectorAll(
"[data-thread-id]").length;if(c===0&&m>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-\
keep-existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-\
indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinne\
r fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel">\
</div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const m=String(
c.id),f=document.createElement("div"),b=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=m===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,f.dataset.threadId=m,f.innerHTML=`<div class="flex items-center gap-1 truncate flex-1\
"><button class="${b} hover:text-yellow-400 px-1" onclick="toggleBookmark(event, '${m}')"><i class="\
fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(c.title||"No Title")}</spa\
n>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 \
transition" data-thread-actions="1"><button class="text-gray-500 hover:text-white px-1 transition" o\
nclick="renameThread(event, '${m}')"><i class="fas fa-pen text-xs"></i></button><button class="text-\
gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${m}')"><i class="fas fa-\
trash text-xs"></i></button></div>`,f.onclick=k=>{k.target.closest("button")||k.target.closest("[dat\
a-thread-actions]")||loadMessages(m)},l?r.insertBefore(f,l):r.appendChild(f)}),hasMoreThreads=!!a.has_next,
hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+a.threads.length+"\
 append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error("Faile\
d to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,r=88,l=52,c=.5,m=8;let f=0,b=!1,y=0,v=null;
const w=o(()=>get(i),"indicatorEl"),k=o(()=>{const M=w();return M?M.querySelector(".ptr-pull-label"):
null},"labelEl"),S=o(M=>{const P=w();if(!P)return;P.style.height=Math.min(M,r)+"px",P.classList.toggle(
"active",M>2),P.classList.toggle("pull-ready",M>=a);const j=k();j&&(j.textContent=M>=a?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),C=o(()=>{const M=w();M&&(M.style.height=
"0px",M.classList.remove("active","pull-ready","refreshing"),M.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",M=>{if(v){b=!1;return}if(n.scrollTop>0){b=!1;return}const P=M.
touches[0];P&&(f=P.clientY,y=0,b=!0)},{passive:!0}),n.addEventListener("touchmove",M=>{if(!b||v)return;
if(n.scrollTop>0){b=!1;return}const P=M.touches[0];if(!P)return;const j=P.clientY-f;if(j<=0){y>0&&(y=
0,S(0)),b=!1;return}const Z=w();Z&&!Z.classList.contains("dragging")&&Z.classList.add("dragging"),y=
Math.min(j*c,r),S(y),j>=m&&M.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!b||
(b=!1,v))return;const M=w();M&&M.classList.remove("dragging");const P=y>=a;if(y=0,!P){C();return}let j;
try{j=t()}catch{j=null}const Z=w();if(Z){Z.classList.add("refreshing"),Z.style.height=l+"px";const J=Z.
querySelector(".ptr-pull-label");J&&(J.textContent="\u66F4\u65B0\u4E2D...")}j&&typeof j.then=="funct\
ion"?(v=j,j.catch(()=>{}).finally(()=>{v=null,C()})):(v=Promise.resolve(),setTimeout(()=>{v=null,C()},
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
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const m=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",m),r=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
r&&r.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const r=get("mcp-decision-modal");if(r&&!r.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}o(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const m=JSON.parse(c);c=JSON.stringify(m,null,2)}catch{}a.textContent=c}const r=get(
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
container")):null;let l="",c="",m=[];if(i){const f=get("prompt-input");l=f?f.value:"",c=f?f.style.height:
"",m=currentImageUrls?currentImageUrls.slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();
a||playChatTransition("history"),currentThreadId=e!=null?String(e):e,t.skipHistory||history.pushState(
{},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),get("welcome-scree\
n").classList.add("hidden"),a||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());try{
const f=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);f.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const b=await apiFetch(f.toString());if(!b.
ok)throw new Error(`thread request failed (${b.status})`);const y=await b.json();if(!y||!Array.isArray(
y.messages))throw new Error("invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(
y&&y.title),allMessages=y.messages,threadHasOlderMessages=!!y.has_older_messages,oldestLoadedMessageId=
y.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const v=(allMessages||[]).filter(k=>k.
role==="user"&&k.content).map(k=>k.content);if(promptHistory=[...new Set(v.slice().reverse())],historyIndex=
-1,tempPrompt="",currentThreadPending=y.pending_job||null,setTemporaryChatUiState(!!(y&&y.is_temporary)),
applyTemporaryChatRuntimeMeta(y||{}),ensureTemporaryChatHeartbeat(!0),get("thread-custom-instruction")&&
(get("thread-custom-instruction").value=y.custom_instruction||""),get("enable-prompt-cache")&&(get("\
enable-prompt-cache").checked=!!y.enable_prompt_caching,updatePromptCacheUi()),y.last_model&&selectModelById(
y.last_model),y.last_gem_uuid&&loadedGems.length>0){const k=loadedGems.find(S=>S.uuid===y.last_gem_uuid);
k&&(threadGemMap[currentThreadId]=k,applyActiveGem(k))}const w=t.forceLatestLeaf?null:localStorage.getItem(
`fixed_branch_${currentThreadId}`);if(w&&allMessages.find(k=>String(k.id)===String(w))?currentLeafId=
w:allMessages.length>0?currentLeafId=allMessages[allMessages.length-1].id:currentLeafId=null,renderThreadTree(
a?{silent:a,keepScroll:a}:{silent:a,keepScroll:a,animate:!0}),i||restoreLatestPromptOptions(),a&&r?applyCodeCollapseByMessage(
get("chat-container"),r,!0):a||applyCodeCollapseByMessage(get("chat-container"),null,!0),currentThreadPending&&
!a&&!isPendingJobSuppressed(currentThreadPending.job_id)&&resumePendingStream(currentThreadPending),
i){const k=get("prompt-input");k&&(k.value=l||"",c?k.style.height=c:k.style.height="auto"),currentImageUrls=
m,currentImageUrls&&currentImageUrls.length?(get("file-preview").classList.remove("hidden"),get("fil\
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
if(l.length){const c=new Set(allMessages.map(f=>f.id)),m=l.filter(f=>!c.has(f.id));m.length&&(allMessages=
m.concat(allMessages))}if(threadHasOlderMessages=!!r.has_older_messages,oldestLoadedMessageId=r.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;let r=null;if(i&&(r=a.scrollTop),a.innerHTML=
"",allMessages.length===0){currentParentId=null,updateTotalTokenBar(0);return}const l={};allMessages.
forEach(w=>{l[w.id]=w,w.childrenIds=[]}),allMessages.forEach(w=>{w.parent_id&&l[w.parent_id]&&l[w.parent_id].
childrenIds.push(w.id)}),(!currentLeafId||!l[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const c=[];let m=l[currentLeafId];for(;m;)c.unshift(m),m=l[m.parent_id];const f=buildTokenTotals(
c),b=buildTokenTotals(allMessages),y=document.createDocumentFragment();if(threadHasOlderMessages){const w=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
k=loadingOlderMessages?"disabled":"",S=document.createElement("div");S.className="mb-3 text-center",
S.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${k}><i class="fas fa-clock-rotate-left mr-1"></i>${w}</button>`,y.appendChild(S)}
c.forEach(w=>{const k=w.parent_id?l[w.parent_id]:null,S=k?k.childrenIds:allMessages.filter(M=>!M.parent_id).
map(M=>M.id),C=S.length>1?{current:S.indexOf(w.id)+1,total:S.length,siblings:S}:null;renderMessage(w.
id,w.role,w.content,w.image_url,w.thought_data,w.model,C,n,w.quote_text,w.tokens,w.tokens_in,w.tokens_out,
w.is_encrypted,w.tokens_content,w.tokens_thought,y,!1,w.parent_id,w.gem_name,w.batch_job)});const v=currentThreadPending;
if(v&&!isPendingJobSuppressed(v.job_id)){const w=v.message_id,k=new Set(c.map(M=>M.id)),S=c.length?c[c.
length-1]:null;if(w&&k.has(w)&&currentLeafId===w||!w&&S&&S.role==="user"){const M=v.job_id?`pending-${v.
job_id}`:null;renderPendingMessage(y,n,!1,M,v.model||null)}}if(a.appendChild(y),updateTotalTokenBar(
f.tokens_total,f,b),currentParentId=currentLeafId,i&&r!==null?restoreThreadTreeScroll(a,r):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(a),c.length){
const w=c[c.length-1]&&c[c.length-1].content;queueMathTypeset(a,w)}}o(renderThreadTree,"renderThread\
Tree");function restoreThreadTreeScroll(e,t){if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&
!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=
e.scrollTop,syncScrollToBottomButton()}o(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){
currentLeafId=e;const t={};allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(
i=>{i.parent_id&&t[i.parent_id]&&t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=
allMessages.length>0?allMessages[allMessages.length-1].id:null,renderThreadTree({animate:!0}),restoreLatestPromptOptions();
return}for(;t[n]&&t[n].childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=
n,renderThreadTree({animate:!0}),restoreLatestPromptOptions()}o(switchVersion,"switchVersion");async function loadGems(){
try{const t=await(await apiFetch(CHAT_CONFIG.urls.handleGems)).json();loadedGems=t;const n=get("gem-\
list");if(!n)return;n.innerHTML='<div id="gem-pull-indicator" class="ptr-pull-indicator" aria-hidden\
="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spi\
nner"></i><span class="ptr-pull-label"></span></div>',Array.isArray(t)&&t.forEach(i=>{const a=document.
createElement("div");a.className="gem-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text\
-gray-300 flex justify-between items-center group",a.innerHTML=`<div class="flex items-center gap-2 \
overflow-hidden"><i class="fas fa-gem text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}\
</span></div><div class="flex items-center gap-1"><button class="text-gray-400 hover:text-blue-400 o\
pacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="openEditGemModal(event,\
'${i.uuid}')"><i class="fas fa-pencil-alt text-[10px]"></i></button><button class="text-gray-400 hov\
er:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="delete\
Gem(event,'${i.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=r=>{r.target.
closest("button")||activateGem(i)},n.appendChild(a)})}catch(e){console.error("Failed to load gems:",
e)}}o(loadGems,"loadGems");function setGemDefaultModelSelect(e){const t=get("gem-default-model");if(!t)
return;t.innerHTML="";const n=document.createElement("option");n.value="",n.textContent="Use current\
 model",t.appendChild(n),MODELS.forEach(a=>{const r=(a.items||[]).filter(c=>!c.deprecated);if(!r.length)
return;const l=document.createElement("optgroup");l.label=a.category,r.forEach(c=>{const m=document.
createElement("option");m.value=c.id,m.textContent=c.name,l.appendChild(m)}),t.appendChild(l)});const i=e||
"";if(i&&!Array.from(t.options).some(a=>a.value===i)){const a=document.createElement("option");a.value=
i,a.textContent=MODEL_NAME_BY_ID[i]||i,t.appendChild(a)}t.value=i}o(setGemDefaultModelSelect,"setGem\
DefaultModelSelect");async function openEditGemModal(e,t){e.stopPropagation(),editingGemUuid=t;try{const i=await(await apiFetch(
`/api/gems/${t}`)).json();get("gem-name").value=i.name,get("gem-desc").value=i.description||"",get("\
gem-inst").value=i.instruction,setGemDefaultModelSelect(i.default_model),renderGemFixedPromptsForEdit(
i.fixed_prompts),get("gem-modal-title").innerHTML='<i class="fas fa-gem text-blue-500 mr-2"></i>Edit\
 Gem',get("save-gem-btn").innerText="Save Changes",showModal("gem-modal"),location.pathname!=="/gem"&&
history.pushState({modal:"gem"},"","/gem")}catch{showToast("Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}o(openEditGemModal,"openEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.
urls.handleGems,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,
instruction:t})}),loadGems()}o(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get(
"fixed-prompts-bar");if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),
get("active-gem-name").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),
t){t.innerHTML="";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}
n.length>0?(t.classList.remove("hidden"),n.forEach((i,a)=>{const r=document.createElement("button");
r.className="fixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover\
:bg-gray-600 text-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex item\
s-center",r.style.animationDelay=`${a*40}ms`,r.textContent=String(i.name||""),r.onclick=()=>{const l=get(
"prompt-input");l&&(l.value=i.content,l.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(
r)})):t.classList.add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML=
"",t.classList.add("hidden"));get("sys-prompt-option").style.opacity="1"}o(applyActiveGem,"applyActi\
veGem");function syncActiveGemForThread(e){const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(
t)}o(syncActiveGemForThread,"syncActiveGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({last_gem_uuid:t,thread_id:e})})}catch{}}o(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){
currentThreadId?(threadGemMap[currentThreadId]=e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\
\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,"success"),t||saveThreadGemUuid(currentThreadId,e?
e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(e),allMessages&&allMessages.length>0&&startNewChat(
{preserveGem:!0}))}o(activateGem,"activateGem");function clearActiveGem(){currentThreadId&&(delete threadGemMap[currentThreadId],
saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=null,applyActiveGem(null)}o(clearActiveGem,
"clearActiveGem");function addGemFixedPromptRow(e="",t=""){const n=get("gem-fixed-prompts-container");
if(!n)return;const i=document.createElement("div");i.className="flex gap-2 items-start gem-fixed-pro\
mpt-row ui-enter",i.innerHTML=`
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
restoreLatestPromptOptions(),resetUploadState(),stopTemporaryChatHeartbeat(),setTemporaryChatUiState(
!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,allMessages=[],promptHistory=
[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=null,loadingOlderMessages=
!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(0),typeof window.
__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}e.skipHistory||
history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").classList.remove(
"hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("thread-custom-instruct\
ion").value=""),get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),
e.preserveGem?activeGem&&applyActiveGem(activeGem):applyActiveGem(null),loadThreads(),window.innerWidth<
768&&get("overlay").click()}o(startNewChat,"startNewChat");let threadModalLoadSeq=0;window.openThreadModal=
async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=i.id!==null&&i.id!==void 0?String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.
is_temporary)),setCurrentChatHeaderTitle(i&&i.title),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
!0,c=get("thread-global-sys-prompt"),m=get("thread-global-sys-prompt-enabled");let f=null;try{f=c||m?
{system_prompt:c?c.value:"",system_prompt_enabled:m?m.checked:!0,apply_global_system_prompt:get("thr\
ead-apply-global-sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(b){sendClientDebugLog("error","Payload c\
onstruction failed: "+b.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const b=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+b.status);let y=!0;if(f){sendClientDebugLog(
"info","Starting POST request for user settings");const v=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});y=v.ok,v.ok&&window.
applySavedUserSystemPromptSettings(f),sendClientDebugLog("info","POST request finished, status: "+v.
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
n.url:buildPdfAttachmentUrl(i),m=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:r,isImage:l,url:c,previewUrl:m}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(m=>maybeNeedsMathJax(m.content)||maybeNeedsMathJax(m.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",r=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(m=>{const f=m.
role==="user",b=m.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(m.quote_text)}\
</div>`:"",y=m.thought_text?`<div class="thought">${escapeHtml(m.thought_text)}</div>`:"",v=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(m.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(m.content||"")}</div>`,w=buildPdfMessageAttachments(m),k=w.length?`<div \
class="attachments">${w.map(M=>M.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(M.previewUrl)}\
" alt="${pdfEscapeAttr(M.filename)}"><div class="file-caption">${pdfEscapeAttr(M.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(M.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
M.filename)}</span><span class="file-source">${pdfEscapeAttr(M.source)}</span></span></a></div>`).join(
"")}</div>`:"",S=[];m.model&&!f&&S.push(m.model),m.tokens!==null&&m.tokens!==void 0&&S.push(`tokens:${m.
tokens}`),m.tokens_in!==null&&m.tokens_in!==void 0&&S.push(`in:${m.tokens_in}`),m.tokens_out!==null&&
m.tokens_out!==void 0&&S.push(`out:${m.tokens_out}`),m.tokens_thought!==null&&m.tokens_thought!==void 0&&
S.push(`thought:${m.tokens_thought}`),m.is_encrypted&&S.push("encrypted"),m.parent_id!==null&&m.parent_id!==
void 0&&S.push(`parent:#${m.parent_id}`);const C=S.length?`<div class="message-meta">${pdfEscapeAttr(
S.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(m.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${b}
                            ${v}
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
        ${l.map(m=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(m.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(m.value)}</div></div>`).join("")}
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
some(M=>maybeNeedsMathJax(M.content)||maybeNeedsMathJax(M.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),b.fonts&&b.fonts.ready)try{await b.fonts.ready}catch{}t.update(60);const k=Array.from(b.
images||[]),S=Promise.all(k.map(M=>M.complete?Promise.resolve():new Promise(P=>{M.addEventListener("\
load",P,{once:!0}),M.addEventListener("error",P,{once:!0})})));if(await Promise.race([S,new Promise(
M=>setTimeout(M,5e3))]),t.update(80),b.getElementById("MathJax-script")){let M=0;for(;M<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(P=>setTimeout(P,50)),M++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(P){console.error(
"PDF MathJax typeset failed",P)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const m=buildPdfDocumentHtml(a),f=new Blob([m],{type:"text/html"});r.src=URL.createObjectURL(
f),document.body.appendChild(r)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=o((m,f)=>i.
compare(m.filename||"",f.filename||""),"nameAsc"),r=o((m,f)=>i.compare(f.filename||"",m.filename||""),
"nameDesc"),l=o((m,f)=>(Number(f.ts)||0)-(Number(m.ts)||0),"tsDesc"),c=o((m,f)=>(Number(m.ts)||0)-(Number(
f.ts)||0),"tsAsc");return t==="name_asc"?n.sort((m,f)=>a(m,f)||l(m,f)):t==="name_desc"?n.sort((m,f)=>r(
m,f)||l(m,f)):t==="oldest"?n.sort((m,f)=>c(m,f)||a(m,f)):n.sort((m,f)=>l(m,f)||a(m,f)),n}o(sortLibraryFiles,
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
a=(n?t.filter(m=>fileNameForSearch(m).includes(n)):t).filter(m=>m.type==="image"),r=lib.favoritesOnly?
a.filter(m=>m.is_favorite):a;if(!r.length)return;const l=r.map(m=>({url:m.url,filename:m.filename||m.
original_filename||m.url.split("/").pop(),element:null}));let c=l.findIndex(m=>m.url===e.url);c===-1&&
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
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",m=e.is_favorite?"fas fa-star":"far f\
a-star",f=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
b=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${f}\
" aria-label="${f}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${m}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${b}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",S=>{S.shiftKey&&S.preventDefault()}),n.onclick=
S=>{if(S&&S.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const C=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),M=C.findIndex(j=>j.dataset.filepath===lib.anchorPath),
P=C.indexOf(n);if(M!==-1&&P!==-1){const j=Math.min(M,P),Z=Math.max(M,P);for(let J=j;J<=Z;J++){const Ce=C[J].
dataset.filepath;Ce&&(lib.selected.add(Ce),C[J].classList.add("is-selected"))}try{const J=window.getSelection&&
window.getSelection();J&&J.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(S=>{S.onclick=C=>{C.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async S=>{S.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const k=n.querySelector(".lib-favorite-btn");return k&&
(k.onclick=async S=>{S.stopPropagation(),k.disabled=!0;try{const C=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
M=await C.json().catch(()=>({}));if(!C.ok||typeof M.is_favorite!="boolean")throw new Error(M.error||
"favorite update failed");e.is_favorite=M.is_favorite,renderLibraryGrid(),showToast(M.is_favorite?"\u304A\
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
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(m=>{m.getAttribute("data-filename")===
e&&setRowAttachmentName(m,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
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
return}if(i.innerHTML="",c.forEach(m=>{const f=document.createElement("div");f.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const b=m.updated_at?new Date(m.
updated_at).toLocaleString():"";f.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(m.title||"")}">${escapeHtml(m.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(b)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=f.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(m.id))}),i.appendChild(f)}),l.has_more){const m=document.
createElement("p");m.className="text-[11px] text-gray-500 text-center pt-2",m.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(m)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const m=getLibSortOrder(),f=getLibSearchQuery(),b=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(b),sort:m,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);r=await v.json(),l=!0}catch(m){i=m}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const m=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,b=new Set(c.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(c.length>=LIBRARY_PAGE_SIZE||!v||b.has(v))return;const w=getAttachmentNameForPath(
v)||v.split("/").pop()||v,k=(w.split(".").pop()||"").toLowerCase(),S=["png","jpg","jpeg","webp","gif"].
includes(k)?"image":"file",C=S==="image"?f+v:null;c.unshift({filename:w,original_filename:w,filepath:v,
url:m+v,thumbnail_url:C,type:S,ext:k,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),b.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const m=c.filter(b=>b&&b.filepath&&
b.url);let f=[];if(e){const b=new Set(lib.files.map(y=>y.filepath));f=m.filter(y=>!b.has(y.filepath)),
lib.files.push(...f)}else lib.files=m;lib.files.forEach(b=>{b&&b.filepath&&setAttachmentNameForPath(
b.filepath,b.filename||b.original_filename||"")}),lib.fileSet=new Set(lib.files.map(b=>b.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
f:null)}catch(m){i=i||m}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
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
a);if(!c)return;const m=(lib.files||[]).find(f=>f&&f.filepath===a);m&&m.filename&&setAttachmentNameForPath(
c,m.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
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
ms-center mt-4";const l=document.createElement("div"),c=String(a.id)===String(currentLeafId),m=a.id===
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
                    ${m?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${c?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,l.onclick=y=>{y.stopPropagation(),selectedBranchNodeId=a.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},r.appendChild(l),a.children.length>0){const y=document.createElement("div");
y.className="w-px h-4 bg-gray-700",r.appendChild(y);const v=document.createElement("div");v.className=
"flex gap-4 items-start",a.children.forEach(w=>v.appendChild(i(w))),r.appendChild(v)}return r}o(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}o(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}o(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(m=>m.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const r=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);r.innerHTML="",Object.entries(l).sort((m,f)=>f[1].
total-m[1].total).forEach(([m,f])=>{const b=document.createElement("div");b.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",b.innerHTML=`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${m}</span>
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
batchProviderLabel(a.provider)),m=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
b=escapeHtml(a.status_text||""),y=batchStateTone(a.state),v=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",w=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
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
                                <span class="truncate max-w-[16rem]">${m}</span>
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
                        ${v}${w}${k}
                    </div>`;const S=r.querySelector("[data-batch-open]");S&&(S.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const C=r.querySelector("[data-batch-cancel]");C&&(C.onclick=()=>cancelBatchJob(
a));const M=r.querySelector("[data-batch-delete]");M&&(M.onclick=()=>deleteBatchJob(a)),t.appendChild(
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
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),m=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=o(b=>{b.key==="Enter"&&(b.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),r.onclick=async()=>{
const b=get("api-key-modal-input").value.trim();if(!b){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=b);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:b})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=b)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",f),m(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",f),m(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",f),m(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const b=get("api-key-modal-input");b&&b.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const m=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,m)}catch{}finally{
a=!1}}o(r,"sendToServer"),console.log=function(...l){e.apply(console,l),r("log",l)},console.error=function(...l){
t.apply(console,l),r("error",l)},console.warn=function(...l){n.apply(console,l),r("warn",l)},console.
info=function(...l){i.apply(console,l),r("info",l)},window.addEventListener("error",function(l){r("e\
xception",[l.message,l.filename,l.lineno,l.colno,l.error])}),window.addEventListener("unhandledrejec\
tion",function(l){r("promise-rejection",[l.reason])}),setTimeout(()=>{console.log("Extended debug lo\
gging system active. Version: v4.8.506")},3e3)})();const BotAdminLog=(()=>{const t={telemetry:{label:"\
\u7591\u308F\u3057\u3044\u64CD\u4F5C",cls:"bg-yellow-600"},verify_ok:{label:"Turnstile \u78BA\u8A8D\u6210\u529F",
cls:"bg-green-600"},verify_fail:{label:"Turnstile \u78BA\u8A8D\u5931\u6557",cls:"bg-orange-600"},turnstile_blocked:{
label:"\u672A\u78BA\u8A8D\u306E\u305F\u3081\u62D2\u5426",cls:"bg-orange-600"},lock:{label:"\u4E00\u6642\u30ED\u30C3\u30AF",
cls:"bg-yellow-600"},lock_blocked:{label:"\u30ED\u30C3\u30AF\u4E2D\u306E\u305F\u3081\u62D2\u5426",cls:"\
bg-yellow-600"},ban:{label:"BAN",cls:"bg-red-600"},related_ban:{label:"\u95A2\u9023\u30A2\u30AB\u30A6\u30F3\u30C8\u304B\u3089\u306EBAN",
cls:"bg-red-600"},admin_action:{label:"\u7BA1\u7406\u8005\u306E\u64CD\u4F5C",cls:"bg-blue-600"},account_deleted:{
label:"\u30A2\u30AB\u30A6\u30F3\u30C8\u524A\u9664",cls:"bg-gray-600"}},n={web:"Web",android:"\u30A2\u30D7\u30EA",
admin:"\u7BA1\u7406\u8005",server:"\u30B5\u30FC\u30D0\u30FC"},i={endpoint:"\u901A\u4FE1\u5148",method:"\
\u65B9\u5F0F",path:"\u30D1\u30B9",client:"\u7AEF\u672B",lock_source:"\u30ED\u30C3\u30AF\u306E\u5BFE\u8C61",
lock_reason:"\u30ED\u30C3\u30AF\u306E\u7406\u7531",remaining_seconds:"\u30ED\u30C3\u30AF\u306E\u6B8B\u308A",
origin_username:"\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8",origin_user_id:"\
\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8ID",origin_client:"\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u7AEF\u672B",
lock_seconds:"\u30ED\u30C3\u30AF\u6642\u9593",applied_to:"\u30ED\u30C3\u30AF\u306E\u7BC4\u56F2",lock_count:"\
\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09",ban_at_count:"BAN\u306B\u306A\u308B\u56DE\u6570",
source_username:"BAN\u306E\u8D77\u70B9",source_user_id:"\u8D77\u70B9\u306E\u30A2\u30AB\u30A6\u30F3\u30C8ID",
action:"\u64CD\u4F5C",admin:"\u7BA1\u7406\u8005",enabled:"\u691C\u51FA",by:"\u524A\u9664\u3057\u305F\u4EBA"},
a={account:"\u30A2\u30AB\u30A6\u30F3\u30C8",ip:"IP\u30A2\u30C9\u30EC\u30B9",cookie:"\u7AEF\u672B\uFF08Cookie\uFF09"},
r={toggle_detection:"\u691C\u51FA\u306E\u5207\u308A\u66FF\u3048",ban:"BAN",unban:"BAN\u89E3\u9664\uFF08\u5358\u72EC\uFF09",
unban_linked:"BAN\u89E3\u9664\uFF08\u9023\u9396\uFF09",unlock:"\u30ED\u30C3\u30AF\u89E3\u9664"},l={userId:0,
username:"",exists:!1,type:"",items:[],hasMore:!1,selected:new Set,counts:{},total:0,data:null},c=o(
()=>get("bot-admin-detail"),"view"),m=o(()=>get("bot-admin-list-view"),"listView"),f=o(T=>{if(!T)return"\
-";const I=new Date(T);return isNaN(I.getTime())?String(T):I.toLocaleString("ja-JP")},"formatTime"),
b=o(T=>{const I=Math.max(0,Math.floor(Number(T)||0));return I>=60?`${Math.floor(I/60)}\u5206${I%60?I%
60+"\u79D2":""}`:`${I}\u79D2`},"formatDuration"),y=o((T,I)=>I==null||I===""?"-":T==="lock_source"?a[I]||
String(I):T==="client"||T==="origin_client"?n[I]||String(I):T==="action"?r[I]||String(I):T==="by"?I===
"self"?"\u672C\u4EBA":I==="admin"?"\u7BA1\u7406\u8005":String(I):T==="enabled"?I?"ON":"OFF":T==="rem\
aining_seconds"||T==="lock_seconds"?b(I):T==="applied_to"&&Array.isArray(I)?I.map(O=>a[O]||O).join("\
\u30FB"):typeof I=="object"?JSON.stringify(I,null,2):String(I),"formatDetailValue"),v=o(T=>{const I=String(
T||"").trim();if(!I)return"";let O=null;if(I.startsWith("{"))try{O=JSON.parse(I)}catch{O=null}if(O&&
typeof O=="object"&&!Array.isArray(O)){const K=Object.keys(O).map(X=>{const ge=y(X,O[X]),_e=i[X]||X;
return ge.length>120||ge.includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(_e)}</div><pre class="bot-log-pre">${escapeHtml(
ge)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(_e)}:</span> <span class="text-gray\
-300">${escapeHtml(ge)}</span></div>`}).join("");return I.length>600?`<details class="mt-1"><summary\
 class="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${K}</details>`:`<div \
class="mt-1 space-y-1">${K}</div>`}return I.length>300?`<details class="mt-1"><summary class="cursor\
-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(
I)}</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(I)}</div>`},"renderD\
etails"),w=o(T=>{if(!T)return'<div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs tex\
t-gray-400">\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u8A18\u9332\u3060\u3051\u304C\u6B8B\u3063\u3066\u3044\u307E\u3059\u3002</div>';
const I=o((ce,me)=>`<span class="${me} text-white px-2 py-0.5 rounded">${escapeHtml(ce)}</span>`,"ba\
dge"),O=[I(T.detection_enabled?"\u691C\u51FAON":"\u691C\u51FAOFF",T.detection_enabled?"bg-gray-600":
"bg-gray-700"),I(T.is_bot_banned?"BAN\u4E2D":"BAN\u306A\u3057",T.is_bot_banned?"bg-red-600":"bg-gray\
-600"),I(T.locks&&T.locks.length?"\u30ED\u30C3\u30AF\u4E2D":"\u30ED\u30C3\u30AF\u306A\u3057",T.locks&&
T.locks.length?"bg-yellow-600":"bg-gray-600")];T.is_admin&&O.push(I("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const K=(T.locks||[]).map(ce=>{const me=a[ce.source]||ce.source,de=ce.identifier?` ${escapeHtml(
ce.identifier)}${ce.source==="cookie"?"\u2026":""}`:"",G=ce.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
ce.origin.username||String(ce.origin.user_id||"-"))}\uFF08${escapeHtml(n[ce.origin.client]||ce.origin.
client||"-")}\uFF09`:"";return`<div class="text-gray-300">${escapeHtml(me)}${de}\u30FB\u6B8B\u308A${escapeHtml(
b(ce.remaining_seconds))}\u30FB${escapeHtml(ce.reason||"")}${G}</div>`}).join(""),X=T.is_bot_banned?
`<div class="text-gray-300">BAN\u306E\u7406\u7531: ${escapeHtml(T.bot_ban_reason||"-")}\uFF08${escapeHtml(
f(T.bot_banned_at))}\uFF09</div>`:"",ge=T.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${b(
T.turnstile_verified_seconds)}\uFF09`:"\u672A\u78BA\u8A8D",_e=T.locks&&T.locks.length?'<button class\
="bot-log-unlock bg-yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>':
"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${O.join("")}</div>
                        ${X}
                        ${K}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String(T.lock_count))} / ${escapeHtml(String(T.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(ge)}\
\u30FB\u5931\u6557 ${escapeHtml(String(T.turnstile_fail_count))} / ${escapeHtml(String(T.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round((T.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round((T.behavior_score||0)*10)/10))}\uFF09</div>
                        ${_e?`<div class="pt-1">${_e}</div>`:""}
                    </div>`},"renderState"),k=o(()=>{const T=o((O,K,X)=>`<button class="bot-log-filt\
er ${l.type===O?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(O)}">${escapeHtml(K)} (${X})</button>`,"chip"),I=[T("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((O,K)=>l.counts[K]-l.counts[O]).forEach(O=>{I.push(T(O,(t[O]||
{label:O}).label,l.counts[O]))}),I.join("")},"renderFilters"),S=o(T=>{const I=t[T.event_type]||{label:T.
event_type,cls:"bg-gray-600"},O=T.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[T.client]||T.client)}</span>`:"",K=l.selected.has(T.id)?"checked":"",X=T.score!==
null&&T.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String(T.score))}${T.
behavior_score!==null&&T.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String(T.behavior_score))}\
\uFF09`:""}</span>`:"",ge=T.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
T.reasons)}</div>`:"",_e=T.ip_address||T.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-\
log-wrap">${T.ip_address?"IP "+escapeHtml(T.ip_address):""}${T.ip_address&&T.user_agent?"\u30FB":""}${escapeHtml(
T.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${T.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${T.id}" ${K}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(f(T.created_at))}</span>
                            <span class="${I.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(I.label)}\
</span>
                            ${O}
                            ${X}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${T.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${ge}
                        ${v(T.details)}
                        ${_e}
                    </div>`},"renderItem"),C=o(()=>{const T=c();if(!T)return;const I=l.exists?"":'<s\
pan class="bg-gray-600 text-white px-2 py-0.5 rounded text-xs">\u524A\u9664\u6E08\u307F</span>',O=l.
items.length?l.items.map(S).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
T.innerHTML=`
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py\
-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>\u4E00\u89A7\u306B\u623B\u308B</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(l.username||
"ID "+l.userId)}</div>
                        ${I}
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-whi\
te px-2 py-1 rounded text-xs">\u66F4\u65B0</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${w(l.data&&l.data.state)}
                        <div class="text-[11px] text-gray-400">\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u306F\u3001\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u524A\u9664\u3057\u3066\u3082\u6B8B\u308A\u307E\u3059\u3002\u540C\u3058\u901A\u4FE1\u5148\u3078\u306E\u62D2\u5426\u306F1\
\u5206\u306B1\u4EF6\u307E\u3067\u8A18\u9332\u3057\u307E\u3059\u3002</div>
                        <div class="flex flex-wrap gap-1 text-xs">${k()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkb\
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${l.
items.length&&l.items.every(K=>l.selected.has(K.id))?"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-\
white px-2 py-1 rounded" ${l.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u8A18\u9332\u3092\u524A\u9664${l.
selected.size?`\uFF08${l.selected.size}\u4EF6\uFF09`:""}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white\
 px-2 py-1 rounded" ${l.total?"":"disabled"}>\u3059\u3079\u3066\u306E\u8A18\u9332\u3092\u524A\u9664</button>
                        </div>
                        <div class="space-y-2">${O}</div>
                        ${l.hasMore?'<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-60\
0 text-white px-2 py-1.5 rounded text-xs">\u3055\u3089\u306B\u8AAD\u307F\u8FBC\u3080</button>':""}
                    </div>`},"render"),M=o(async(T=!1)=>{const I=c();if(!I||!l.userId)return;T||(I.innerHTML=
'<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');
const O=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&O.set("type",l.type),
T&&l.items.length&&O.set("before_id",String(l.items[l.items.length-1].id));try{const K=await apiFetch(
`/api/bot/evidence?${O.toString()}`,{cache:"no-store"}),X=await K.json().catch(()=>({}));if(!K.ok)throw new Error(
X.error||String(K.status));l.counts=X.counts||{},l.total=X.total||0,l.hasMore=!!X.has_more,X.user&&(l.
username=X.user.username||l.username,l.exists=!!X.user.exists),T?l.items=l.items.concat(X.items||[]):
(l.items=X.items||[],l.data=X),C()}catch{T||(I.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),P=o(async(T,I)=>{if(confirm(I))try{const O=await apiFetch("/api/bot/evidence/d\
elete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.assign(
{user_id:l.userId},T))}),K=await O.json().catch(()=>({}));if(!O.ok)throw new Error(K.error||String(O.
status));showToast(`${K.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await M(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),j=o(async()=>{if(!(!l.username||!confirm(`${l.username} \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`)))
try{const T=await apiFetch("/api/bot/update",{method:"POST",headers:{"Content-Type":"application/jso\
n"},body:JSON.stringify({username:l.username,action:"unlock"})}),I=await T.json().catch(()=>({}));if(!T.
ok)throw new Error(I.error||String(T.status));showToast("\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F",
"success"),await M(!1)}catch{showToast("\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"unlock"),Z=o(()=>{const T=c();T&&(T.classList.add("hidden"),T.classList.remove("flex"),
T.innerHTML="");const I=m();I&&(I.classList.remove("hidden"),I.classList.add("flex")),l.userId=0},"s\
howList");return{open:o(async(T,I="")=>{const O=c();if(!O)return;Object.assign(l,{userId:Number(T)||
0,username:I,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,data:null}),l.selected.clear();
const K=m();K&&(K.classList.add("hidden"),K.classList.remove("flex")),O.classList.remove("hidden"),O.
classList.add("flex"),await M(!1)},"open"),showList:Z,bind:o(()=>{const T=c();!T||T.dataset.bound===
"1"||(T.dataset.bound="1",T.addEventListener("change",I=>{const O=I.target;if(!(!O||!O.classList))if(O.
classList.contains("bot-log-select")){const K=Number(O.getAttribute("data-log-id"));O.checked?l.selected.
add(K):l.selected.delete(K),C()}else O.classList.contains("bot-log-select-all")&&(l.items.forEach(K=>{
O.checked?l.selected.add(K.id):l.selected.delete(K.id)}),C())}),T.addEventListener("click",async I=>{
const O=I.target.closest("button");if(!(!O||O.disabled))if(O.classList.contains("bot-log-back"))Z(),
window.reloadBotAdminUsers&&await window.reloadBotAdminUsers();else if(O.classList.contains("bot-log\
-reload"))await M(!1);else if(O.classList.contains("bot-log-more"))await M(!0);else if(O.classList.contains(
"bot-log-filter"))l.type=O.getAttribute("data-type")||"",l.selected.clear(),await M(!1);else if(O.classList.
contains("bot-log-delete-one")){const K=Number(O.getAttribute("data-log-id"));K&&await P({ids:[K]},"\
\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
O.classList.contains("bot-log-delete-selected")?l.selected.size&&await P({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-delete-all")?await P({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-unlock")&&await j()}))},"bind")}})();window.BotAdminLog=BotAdminLog;
