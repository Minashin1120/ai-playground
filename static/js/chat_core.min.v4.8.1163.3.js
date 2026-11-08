function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),r=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const C=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",C.forEach((M,E)=>{
const D=M.getAttribute("data-local-url"),P=M.getAttribute("data-filename"),Z=M.querySelector("img.up\
load-preview")!==null;let j;if(Z){let W=D;if(!W&&P){const re=P.replace(/^\d+\//,"");W=buildAttachmentPreviewUrl(
re)}W&&(j=document.createElement("img"),j.src=W,j.className="thumb-item shadow-sm",j.dataset.viewerSrc=
W,j.dataset.viewerFilename=P||W.split("/").pop(),j.uploadRow=M,j.onclick=function(re){re.preventDefault(),
openImageViewer(this.dataset.viewerSrc,".thumb-item")},j.onerror=function(){this.parentElement.replaceChild(
d("ERR"),this)})}j||(j=d("FILE")),j.style.animationDelay=`${E*32}ms`,a.appendChild(j)}),C.length>0?a.
classList.remove("hidden"):a.classList.add("hidden")}function d(C){const M=document.createElement("d\
iv");return M.className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-\
[9px] shadow-sm font-bold",M.innerText=C,M}o(d,"createFileThumb");const f=collectImageUrlsForSend(),
h=uploadProgressState.total,y=uploadProgressState.completed,v=uploadProgressState.active;h===0&&(e.classList.
add("hidden"),n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));
const w=get("send-btn"),k=get("mic-btn"),x=get("mask-btn"),L=isStopMode;if(v>0?(w&&(w.disabled=!0),k&&
(k.disabled=!0),x&&(x.disabled=!0)):L||(w&&(w.disabled=!1),k&&(k.disabled=!1),x&&(x.disabled=!1)),v>
0){const C=`Preparing... (${y}/${h})`;e.classList.remove("hidden"),t.innerText=C,r&&(r.innerText=`(${y}\
/${h})`);let M=y*100,E=0;for(let Z in uploadProgressState.perFilePct)M+=uploadProgressState.perFilePct[Z],
E++;const D=h>0?M/(h*100)*100:0,P=`${Math.min(100,D)}%`;n&&i&&(n.classList.remove("hidden"),i.style.
width=P),l&&c&&(l.classList.remove("hidden"),c.style.width=P)}else r&&(r.innerText=""),l&&l.classList.
add("hidden"),f.length>0?(e.classList.remove("hidden"),t.innerText=`${f.length} files ready`,n&&n.classList.
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
rker-color]").forEach(f=>{const h=normalizeMarkerHexColor(f.getAttribute("data-marker-color"));f.classList.
toggle("active",h===e)})}o(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}o(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}o(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const a=get("marker-tool-hi\
nt");a&&(a.textContent=markerState.maskMode?e==="auto"?"\u753B\u50CF\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u8FD1\u4F3C\u8272\u306E\u3064\u306A\u304C\u3063\u305F\u7BC4\u56F2\u3092\u5857\u308A\u307E\u3059":
"\u767D\u304F\u5857\u3063\u305F\u90E8\u5206\u304C\u524A\u9664\u5BFE\u8C61\u306B\u306A\u308A\u307E\u3059 / \u4E8C\u672C\u6307\u3067\u62E1\u5927":
markerToolHints[e]||"");const r=get("marker-crop-reset");r&&r.classList.toggle("hidden",e!=="crop"),
document.querySelectorAll(".marker-edit-only").forEach(C=>C.classList.toggle("hidden",markerState.maskMode));
const l=get("marker-tool-mosaic"),c=get("marker-tool-crop");l&&l.classList.toggle("hidden",markerState.
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const d=get("marker-tool-draw"),f=get(
"marker-tool-auto");d&&(d.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
f&&(f.classList.toggle("hidden",!markerState.maskMode),f.classList.toggle("active",e==="auto")),d&&d.
classList.toggle("active",e==="draw");const h=get("marker-brush-control");h&&h.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const v=get("marker-modal-title"),w=get("marker-modal-de\
scription"),k=get("marker-save");v&&(v.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u4F5C\u6210":
"\u753B\u50CF\u7DE8\u96C6"),w&&(w.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\
\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":
"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
k&&(k.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const x=get("marker-canvas");x&&(x.style.pointerEvents=e==="crop"?"none":"auto");const L=get("marker\
-crop-canvas");L&&(L.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.cropRect||
markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}o(setMarkerMode,
"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}o(clearCropRect,
"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)return;const t=Math.
max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,y:0,w:t,h:n})}o(resetCropRectToFull,
"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.scale=Math.min(markerView.maxScale,
Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.scale<=markerView.minScale+1e-4){
markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marker-stage"),t=get("marker-viewport");
if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(1,e.clientHeight||0),a=Math.max(1,t.
offsetWidth||t.clientWidth||0),r=Math.max(1,t.offsetHeight||t.clientHeight||0);if(n<=1||i<=1||a<=1||
r<=1)return;const l=(n-a)/2,c=(i-r)/2,d=a*markerView.scale,f=r*markerView.scale,h=Math.min(n*.45,Math.
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=h-l-d,w=n-h-l,k=y-c-f,x=i-y-c,L=o((C,M,E)=>Number.
isFinite(C)?M>E?(M+E)/2:Math.min(E,Math.max(M,C)):0,"clampOffset");markerView.offsetX=L(markerView.offsetX,
v,w),markerView.offsetY=L(markerView.offsetY,k,x)}o(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
const e=get("marker-viewport");e&&(clampMarkerViewOffset(),e.style.transform=`translate(${markerView.
offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale})`)}o(applyMarkerTransform,"applyMarke\
rTransform");function resetMarkerTransform(){markerView.scale=1,markerView.offsetX=0,markerView.offsetY=
0,applyMarkerTransform()}o(resetMarkerTransform,"resetMarkerTransform");function getRowMarkerKey(e){
return e&&(e.dataset.uploadId||e.getAttribute("data-filename"))||null}o(getRowMarkerKey,"getRowMarke\
rKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(e);n&&(t?markerAppliedUploads.add(n):
markerAppliedUploads.delete(n));const i=e?e.querySelector(".upload-marker-tag"):null;i&&i.classList.
toggle("hidden",!t)}o(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){return markerAppliedUploads.
size>0}o(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){const t=String(e||"").
trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t==="uploaded"?"upload":
"unknown"}o(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
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
height);const n=o((l,c,d=null,f=!1)=>{if(!l)return;const h=Math.max(0,l.x),y=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);d&&(t.fillStyle=d,t.fillRect(h,y,v,w)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(h+.5,y+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),d=Math.max(1,i.w),f=Math.max(1,i.h);t.clearRect(l,c,d,f),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((a,r,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const d=normalizeAttachmentSource(
r),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),h=t.get(c);if(h===void 0){const w=e.
length;t.set(c,w),e.push({path:c,source:d,name:f});return}const y=e[h];if(!y)return;const v=normalizeAttachmentSource(
y.source);(v==="unknown"&&d!=="unknown"||v==="library"&&d==="upload")&&(y.source=d),!normalizeAttachmentDisplayName(
y.name)&&f&&(y.name=f)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const r=a.getAttribute("data-filename");n(r,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const r=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const d=isAudioPath(
c),f=isVideoPath(c);if(d&&!t.audio||f&&!t.video){d&&(n+=1),f&&(i+=1);return}r.push(c)}),r.length!==currentImageUrls.
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
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e,t=!1){
const n=getRowImageSource(e);if(!n){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e,markerState.maskMode=t;const i=e?e.querySelector(".truncate"):null;
markerState.filename=i?i.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=
[],markerState.naturalWidth=0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=
[],markerState.mosaicPreviewRect=null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode(
"draw");const a=get("marker-attach-original");a&&(a.checked=e.dataset.attachOriginal==="1");const r=get(
"marker-image"),l=get("marker-canvas"),c=get("marker-crop-canvas");if(l){const d=l.getContext("2d");
d&&d.clearRect(0,0,l.width,l.height)}if(c){const d=c.getContext("2d");d&&d.clearRect(0,0,c.width,c.height)}
resetMarkerTransform(),showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState(
{modal:"marker"},"","/edit-image"),r&&(r.onload=()=>{if(!get("marker-stage")||!l)return;const f=Math.
max(1,Math.floor(r.clientWidth)),h=Math.max(1,Math.floor(r.clientHeight));l.width=f,l.height=h,l.style.
width=`${f}px`,l.style.height=`${h}px`,l.style.left="0px",l.style.top="0px",c&&(c.width=f,c.height=h,
c.style.width=`${f}px`,c.style.height=`${h}px`,c.style.left="0px",c.style.top="0px"),markerState.naturalWidth=
r.naturalWidth||f,markerState.naturalHeight=r.naturalHeight||h;const y=l.getContext("2d");y&&y.clearRect(
0,0,l.width,l.height),prepareMarkerBaseCanvas(r,f,h),saveMarkerHistory(),markerState.mode==="crop"&&
!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),resetMarkerTransform()},r.src=n)}o(
openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={total:0,completed:0,active:0,
perFilePct:{}};const uploadCancelTokens=new Set;let uploadGeneration=0;function isUploadCancelled(e){
return!!(e&&(e.generation!==void 0&&e.generation!==uploadGeneration||e.uploadId&&uploadCancelTokens.
has(e.uploadId)||e.row&&!e.row.isConnected))}o(isUploadCancelled,"isUploadCancelled");function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
o(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){uploadGeneration++,
browserFastLocalFiles.forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,d=c?c.getAttribute("data-loc\
al-url"):null;d&&URL.revokeObjectURL(d)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=
null,uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.
clear();const e=get("file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");
t&&(t.innerHTML="",t.classList.add("hidden"));const n=get("upload-list");n&&(n.innerHTML='<div class\
="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'),
updateFilePreview(),updateMaskPreview();const i=get("file-input");i&&(i.value="");const a=get("photo\
-input");a&&(a.value="");const r=get("mask-input");r&&(r.value="")}o(resetUploadState,"resetUploadSt\
ate");async function uploadMaskFile(e){if(!e)return!1;const t=new FormData;t.append("file",e);try{const n=await fetch(
CHAT_CONFIG.urls.upload,{method:"POST",body:t}),i=await n.json();if(n.ok&&i.filename)return currentMaskImage=
i.filename,updateMaskPreview(),!0;showToast(i.error||"Mask upload failed","error",!0)}catch{showToast(
"Mask upload failed","error",!0)}return!1}o(uploadMaskFile,"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){
const n=get("camera-status");n&&(n.textContent=e||"",n.classList.toggle("text-red-300",!!t),n.classList.
toggle("text-gray-400",!t))}o(setCameraCaptureStatus,"setCameraCaptureStatus");function updateCameraCapturePendingUi(){
const e=cameraCapturePendingFiles.length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,
t.textContent=e?`\u6DFB\u4ED8 (${e})`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=
e===0||cameraCaptureBusy);const i=get("camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.
forEach((a,r)=>{const l=document.createElement("div");l.className="relative rounded overflow-hidden \
border border-gray-700 bg-black aspect-square",l.innerHTML=`
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
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],d=c&&c.getSettings?c.getSettings():{},
f=String(d.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
e;const h=get("camera-capture-btn");return h&&(h.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
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
const l=await new Promise((d,f)=>{t.toBlob(h=>{h?d(h):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
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
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{if(get("camera-split-enabled").checked)await openImageSplit(
r);else{const c=(await handleFiles(r,{openModal:!1,cameraCapture:!0})).filter(Boolean).length;c>0&&showToast(
`${c}\u679A\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success"),c<r.length&&(showToast(
"\u6DFB\u4ED8\u3067\u304D\u306A\u304B\u3063\u305F\u753B\u50CF\u306F\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u753B\u9762\u304B\u3089\u518D\u8A66\u884C\u3067\u304D\u307E\u3059\u3002",
"warning",!0),openUploadModal())}}catch(l){const c=l&&l.message?l.message:"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(c,"error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.
classList.contains("hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}
o(attachCameraCapturedFiles,"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay==
"function"&&window.hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.
pathname!=="/upload"&&history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-inf\
o");if(e){const n=(get("model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v\
4.1-flash"||n==="deepseek-v4-flash-vision-exp",a=["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4\
.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(n),r=n.includes("deepseek")&&!i||n.startsWith("glm\
-")&&!a;e.classList.toggle("hidden",!r)}_syncVisionModelDisplay()}o(openUploadModal,"openUploadModal");
function _syncVisionModelDisplay(){const e=get("vision-model-display");if(!e)return;const t=currentVisionModel;
if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(a=>{a.id===t&&(n=a.name)})),e.textContent=n}else
e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}o(_syncVisionModelDisplay,"_syncVisionModelDisp\
lay");function _openVisionModelSelector(){window._visionPickerActive=!0,openModelModal(),setTimeout(
()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},50)}o(_openVisionModelSelector,
"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.hideDropOverlay=="function"&&
window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname==="/upload"&&history.back()}
o(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){const e=get("upload-list");
if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(n=>{const i=n.getAttribute(
"data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(n,{source:getAttachmentSourceForPath(
n),displayName:getAttachmentNameForPath(n)})}),e.children.length===0&&(e.innerHTML='<div class="text\
-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
o(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}o(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const a=e.split("/").pop()||e,r=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||a,l=(a.split(".").pop()||"").toLowerCase(),c=["png","jpg","jpeg","webp",
"gif"].includes(l),d=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):d,h=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=h,y.setAttribute("data-filename",e),y.dataset.fileSource=
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
            `;const x=y.querySelector(".upload-preview");x&&(x.onclick=()=>openFileViewer(d,getRowAttachmentName(
y)||r));const L=y.querySelector(".upload-send-name");L&&(L.onclick=()=>promptRowAttachmentName(y));const C=y.
querySelector(".upload-remove");C&&(C.onclick=()=>{uploadCancelTokens.add(h),browserFastLocalFiles.delete(
h),decrementUploadTotal(h);const E=y.getAttribute("data-filename");E&&(currentImageUrls=currentImageUrls.
filter(D=>D!==E)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const M=y.querySelector(".upload-marker");return M&&(M.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,r),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:h}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const r=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const d=l&&
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
                        ${d}
                        <button class="upload-send-name text-[10px] text-gray-300 hover:text-white b\
order border-gray-700 rounded px-2 py-1">\u9001\u4FE1\u540D</button>
                        <button class="upload-remove text-[10px] text-gray-400 hover:text-red-400 bo\
rder border-gray-700 rounded px-2 py-1">\u524A\u9664</button>
                    </div>
                </div>
                <div class="upload-progress h-2 rounded mt-2 overflow-hidden">
                    <div style="width:0%"></div>
                </div>
            `,f&&i.setAttribute("data-local-url",f);const h=i.querySelector(".upload-preview");h&&(h.
onclick=()=>{const k=i.getAttribute("data-filename"),x=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),L=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(x,L)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const x=i.
getAttribute("data-filename");x&&(currentImageUrls=currentImageUrls.filter(L=>L!==x)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const v=i.querySelector(".upload-marker");v&&(v.onclick=()=>openMarkerModalForRow(i));const w=i.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(isUploadCancelled(
t)||!e)return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);
try{const i=await apiFetch("/upload/init",{method:"POST",headers:{"Content-Type":"application/json",
"X-CSRF-Token":csrfToken},body:JSON.stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.
ok){const y=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}const r=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let y=0;y<c;y++){const v=y*l,w=Math.min(e.size,
v+l),k=e.slice(v,w);if(!await new Promise(L=>{const C=new XMLHttpRequest;C.open("POST","/upload/chun\
k",!0),C.setRequestHeader("X-CSRF-Token",csrfToken),C.upload.onprogress=E=>{if(E.lengthComputable&&t&&
t.bar){const D=v+E.loaded,P=Math.min(100,Math.floor(D/e.size*100));t.bar.style.width=`${P}%`,t.status&&
(t.status.textContent=`${P}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,P)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},C.onload=()=>{C.status>=200&&C.status<300?L(!0):L(!1)},C.
onerror=()=>L(!1);const M=new FormData;M.append("upload_id",r),M.append("index",String(y)),M.append(
"total",String(c)),M.append("chunk",k,e.name),C.send(M)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),f=await d.json();if(d.ok&&f&&f.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const k=t.row.querySelector(
"img.upload-preview");if(k){const x=f.filename.replace(/^\d+\//,"");k.src=buildAttachmentPreviewUrl(
x)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),y)){const w=isRowAttachmentNameCustomized(
t.row),k=defaultAttachmentDisplayName(y),x=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,x)}return y&&setAttachmentSourceForPath(
y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(f.
filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(w=>addLibraryFileFromPath(w)),!0}const h=f&&
f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),!1}catch{return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t,n={}){return new Promise(i=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(i);return}if(isUploadCancelled(t)){i(!1);return}let a=!1;window.ConnectionMonitor&&
(window.ConnectionMonitor.operationStarted(),a=!0);const r=o(()=>{a&&window.ConnectionMonitor&&(window.
ConnectionMonitor.operationEnded(),a=!1)},"finishUploadOp"),l=new XMLHttpRequest;l.open("POST",CHAT_CONFIG.
urls.upload,!0),l.setRequestHeader("X-CSRF-Token",csrfToken),l.upload.onprogress=d=>{if(d.lengthComputable&&
t&&t.bar){const f=Math.min(100,Math.floor(d.loaded/d.total*100));t.bar.style.width=`${f}%`,t.status&&
(t.status.textContent=`${f}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,f)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},l.onload=()=>{let d={};try{d=JSON.parse(l.responseText||"\
{}")}catch{}if(l.status>=200&&l.status<300&&d&&d.filename){if(isUploadCancelled(t)){t.row&&t.row.parentNode&&
t.row.remove(),r(),i(!1);return}if(t&&t.row){const y=t.row.getAttribute("data-local-url");y&&URL.revokeObjectURL(
y),t.row.removeAttribute("data-local-url");const v=t.row.querySelector("img.upload-preview");if(v){const w=d.
filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(w)}}const f=normalizeAttachmentPath(d.
filename);if(f&&currentImageUrls.push(f),t&&t.row&&(t.row.setAttribute("data-filename",f||d.filename),
setRowAttachmentSource(t.row,"upload"),f)){const y=isRowAttachmentNameCustomized(t.row),v=defaultAttachmentDisplayName(
f),w=y&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||v;t.row.dataset.defaultDisplayName=
v,setRowAttachmentName(t.row,w)}f&&setAttachmentSourceForPath(f,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(d.filenames)&&d.filenames.length?d.filenames:[d.filename]).
forEach(y=>addLibraryFileFromPath(y)),r(),i(!0)}else{const f=[502,503,504].includes(l.status)||l.status>=
520&&l.status<=527;if(n.cameraCapture&&f&&!d.error&&!isUploadCancelled(t)){t&&t.status&&(t.status.textContent=
"\u5206\u5272\u3057\u3066\u518D\u9001\u4E2D..."),r(),uploadFileChunked(e,t).then(i);return}const h=d&&
d.error?d.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),r(),i(!1)}},l.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),r(),i(!1)};const c=new FormData;c.append("file",e),l.send(c)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(f=>{n+=f.length});const i=new Float32Array(n);
let a=0;e.forEach(f=>{i.set(f,a),a+=f.length});const r=new ArrayBuffer(44+i.length*2),l=new DataView(
r),c=o((f,h)=>{for(let y=0;y<h.length;y++)l.setUint8(f+y,h.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let d=44;for(let f=0;f<i.length;f++){const h=Math.
max(-1,Math.min(1,i[f]));l.setInt16(d,h<0?h*32768:h*32767,!0),d+=2}return new Blob([l],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const r=e.row.getAttribute("data-local-url");r&&URL.revokeObjectURL(
r);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),d=escapeHtml(a),f=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${d}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',h=e.row.querySelector(".upload-preview");h&&(h.outerHTML=f);const y=e.row.querySelector(
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
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},f=[],h=16,y="",v=null,w=null,k=null,x=null,L=null,C=!1,M=null;
const E=o(B=>{const V=e.getBoundingClientRect(),de=(B.clientX-V.left)*(e.width/V.width),fe=(B.clientY-
V.top)*(e.height/V.height);return{x:de,y:fe}},"getPoint"),D=o((B,V)=>({x:(B.x+V.x)/2,y:(B.y+V.y)/2}),
"getMid"),P=o((B,V)=>Math.hypot(B.x-V.x,B.y-V.y),"getDist");let Z=!1;const j=o(()=>{w||(w=document.createElement(
"canvas"),k=w.getContext("2d")),x||(x=document.createElement("canvas"),L=x.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(x.width!==e.width||x.height!==e.
height)&&(x.width=e.width,x.height=e.height)},"ensureDrawBuffers"),W=o(()=>{if(!t||!w||!x)return;const B=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=B,t.drawImage(x,0,0),t.restore()},"r\
enderDrawPreview"),re=o(()=>{L&&(L.strokeStyle=y,L.fillStyle=y,L.lineWidth=h,L.lineCap="round",L.lineJoin=
"round")},"applyMarkerBrush"),me=o(B=>{if(!B)return!1;if(f.length===0)return f.push(B),!0;const V=f[f.
length-1],de=B.x-V.x,fe=B.y-V.y,Q=Math.hypot(de,fe),le=Math.max(.35,h*.04);if(Q<le)return!1;const ee=Math.
max(1,h*.25),pe=Math.max(1,Math.ceil(Q/ee));for(let se=1;se<=pe;se++){const Fe=se/pe;f.push({x:V.x+de*
Fe,y:V.y+fe*Fe})}return!0},"appendStrokePoint"),ie=o(()=>{if(L&&(L.clearRect(0,0,x.width,x.height),f.
length!==0)){if(re(),f.length===1){const B=f[0];L.beginPath(),L.arc(B.x,B.y,h/2,0,Math.PI*2),L.fill();
return}if(L.beginPath(),L.moveTo(f[0].x,f[0].y),f.length===2)L.lineTo(f[1].x,f[1].y);else{for(let de=1;de<
f.length-2;de++){const fe=f[de],Q=f[de+1],le=D(fe,Q);L.quadraticCurveTo(fe.x,fe.y,le.x,le.y)}const B=f[f.
length-2],V=f[f.length-1];L.quadraticCurveTo(B.x,B.y,V.x,V.y)}L.stroke()}},"renderStrokeLayer"),ke=o(
(B,V)=>{if(!B||!V)return null;const de=Math.min(B.x,V.x),fe=Math.min(B.y,V.y),Q=Math.abs(B.x-V.x),le=Math.
abs(B.y-V.y);return{x:de,y:fe,w:Q,h:le}},"normalizeMosaicRect"),$=o(B=>{const V=n?Number(n.value||16):
16,de=Math.max(6,Math.floor(V)),fe=Math.floor(de/2);return{x:B.x-fe,y:B.y-fe,w:de,h:de}},"buildMosai\
cRectFromPoint"),N=o(()=>{const B=document.createElement("canvas");B.width=e.width,B.height=e.height;
const V=B.getContext("2d");if(!V)return null;markerState.baseCanvas&&V.drawImage(markerState.baseCanvas,
0,0),V.drawImage(e,0,0);try{return V.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),U=o(B=>{if(!t||!B)return!1;const V=N();if(!V)return!1;const de=n?Number(n.value||
16):16,fe=Math.max(4,Math.floor(de/2)),Q=Math.max(0,Math.floor(B.x)),le=Math.max(0,Math.floor(B.y)),
ee=Math.min(e.width,Math.ceil(B.x+B.w)),pe=Math.min(e.height,Math.ceil(B.y+B.h));if(ee<=Q||pe<=le)return!1;
for(let se=le;se<pe;se+=fe)for(let Fe=Q;Fe<ee;Fe+=fe){const ue=Math.min(fe,ee-Fe),Se=Math.min(fe,pe-
se),Me=Math.min(e.width-1,Math.max(0,Fe+Math.floor(ue/2))),De=(Math.min(e.height-1,Math.max(0,se+Math.
floor(Se/2)))*e.width+Me)*4,wt=V.data[De],he=V.data[De+1],pt=V.data[De+2];t.fillStyle=`rgb(${wt},${he}\
,${pt})`,t.fillRect(Fe,se,ue,Se)}return!0},"applyMosaicRect"),O=o(B=>{if(!t)return;if(i.set(B.pointerId,
{x:B.clientX,y:B.clientY}),i.size>=2){const de=Array.from(i.values()),fe=de[0],Q=de[1];a=!0,Z=!1,f=[],
C=!1,M=null,v=null,markerState.mosaicPreviewRect=null,r=P(fe,Q)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=D(fe,Q),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault();return}if(a||markerState.mode==="crop")return;Z=!0;const V=E(B);if(markerState.
mode==="auto"&&markerState.maskMode)v=V;else if(markerState.mode==="mosaic")C=!0,M=V,markerState.mosaicPreviewRect=
$(V),renderCropOverlay();else{if(j(),!k||!L)return;k.clearRect(0,0,w.width,w.height),k.drawImage(e,0,
0),L.clearRect(0,0,x.width,x.height),h=n?Number(n.value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),f=[],me(V),ie(),markerState.hasStroke=!0,W()}e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault()},"start"),q=o(B=>{if(i.has(B.pointerId)&&i.set(B.pointerId,{x:B.clientX,
y:B.clientY}),a&&i.size>=2){const de=Array.from(i.values()),fe=de[0],Q=de[1],le=D(fe,Q),ee=P(fe,Q)||
1,pe=l*(ee/r);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,pe)),markerView.
offsetX=c.x+(le.x-d.x),markerView.offsetY=c.y+(le.y-d.y),applyMarkerTransform(),B.preventDefault();return}
if(!Z||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){B.preventDefault();return}const V=E(
B);if(markerState.mode==="mosaic"){if(!C||!M)return;markerState.mosaicPreviewRect=ke(M,V)||$(V),renderCropOverlay()}else
me(V)&&(ie(),W());B.preventDefault()},"move"),te=o(B=>{const V=Z;if(i.delete(B.pointerId),i.size<2&&
(a=!1),i.size===0){Z=!1;const de=V&&B.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.
maskMode&&v?fillMaskSimilarRegion(v):!1;if(v=null,V&&t&&markerState.mode==="draw"&&f.length>0&&(ie(),
W()),V&&markerState.mode==="mosaic"&&M){const fe=E(B);let Q=ke(M,fe);(!Q||Q.w<2||Q.h<2)&&(Q=$(M)),U(
Q)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(Q))}f=[],C=!1,M=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),V&&(markerState.mode!=="auto"||de)&&saveMarkerHistory()}e.releasePointerCapture&&
e.releasePointerCapture(B.pointerId),B.preventDefault()},"end");e.addEventListener("pointerdown",O),
e.addEventListener("pointermove",q),e.addEventListener("pointerup",te),e.addEventListener("pointerca\
ncel",te)}o(initMarkerCanvas,"initMarkerCanvas");function fillMaskSimilarRegion(e){var Z;const t=get(
"marker-canvas"),n=markerState.baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,r=Math.floor(e.x),l=Math.floor(e.y);if(r<0||l<0||r>=i||l>=
a)return!1;const c=n.data,d=l*i+r,f=d*4;if(c[f+3]<16)return!1;const h=c[f],y=c[f+1],v=c[f+2],w=Number(
((Z=get("marker-auto-tolerance"))==null?void 0:Z.value)||32),k=new Uint8Array(i*a),x=[d],L=[],C=25e4,
M=o(j=>{const W=j*4;return c[W+3]>=16&&Math.max(Math.abs(c[W]-h),Math.abs(c[W+1]-y),Math.abs(c[W+2]-
v))<=w},"matches");for(k[d]=1;x.length;){const j=x.pop();if(k[j]===2||!M(j))continue;const W=Math.floor(
j/i);let re=j%i,me=re;for(;re>0&&k[W*i+re-1]!==2&&M(W*i+re-1);)re--;for(;me+1<i&&k[W*i+me+1]!==2&&M(
W*i+me+1);)me++;for(let ie=re;ie<=me;ie++)k[W*i+ie]=2;if(L.push(W,re,me+1),L.length/3>C)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const ie of[W-1,W+1]){if(ie<0||ie>=a)continue;let ke=re;for(;ke<=me;){const $=ie*
i+ke;if(k[$]===0&&M($))for(k[$]=1,x.push($);ke+1<=me&&k[ie*i+ke+1]===0&&M(ie*i+ke+1);)ke++,k[ie*i+ke]=
1;ke++}}}if(!L.length)return!1;const E=t.getContext("2d");if(!E)return!1;const D=E.getImageData(0,0,
i,a),P=D.data;for(let j=0;j<L.length;j+=3){const W=L[j]*i+L[j+1],re=L[j]*i+L[j+2];for(let me=W;me<re;me++){
const ie=me*4;P[ie]=P[ie+1]=P[ie+2]=P[ie+3]=255}}return E.putImageData(D,0,0),markerState.hasStroke=
!0,!0}o(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,
d=0,f=markerView.scale,h={x:0,y:0},y={x:0,y:0};const v=8,w=14,k=o((j,W,re)=>Math.min(re,Math.max(W,j)),
"clamp"),x=o(j=>{const W=e.getBoundingClientRect(),re=(j.clientX-W.left)*(e.width/W.width),me=(j.clientY-
W.top)*(e.height/W.height);return{x:re,y:me}},"getPoint"),L=o((j,W)=>({x:(j.x+W.x)/2,y:(j.y+W.y)/2}),
"getMid"),C=o((j,W)=>Math.hypot(j.x-W.x,j.y-W.y),"getDist"),M=o(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),E=o((j,W)=>{if(!W)return"move";const re=W.x,me=W.y,ie=W.x+W.
w,ke=W.y+W.h,$=Math.abs(j.x-re)<=w,N=Math.abs(j.x-ie)<=w,U=Math.abs(j.y-me)<=w,O=Math.abs(j.y-ke)<=w;
if($&&U)return"nw";if(N&&U)return"ne";if($&&O)return"sw";if(N&&O)return"se";if(U)return"n";if(O)return"\
s";if($)return"w";if(N)return"e";if(j.x>re+w&&j.x<ie-w&&j.y>me+w&&j.y<ke-w)return"move";const te=j.x<
re?"left":j.x>ie?"right":null,B=j.y<me?"top":j.y>ke?"bottom":null;if(te&&B){if(te==="left"&&B==="top")
return"nw";if(te==="right"&&B==="top")return"ne";if(te==="left"&&B==="bottom")return"sw";if(te==="ri\
ght"&&B==="bottom")return"se"}return te?te==="left"?"w":"e":B?B==="top"?"n":"s":"move"},"hitTest"),D=o(
j=>{if(markerState.mode!=="crop")return;if(n.set(j.pointerId,{x:j.clientX,y:j.clientY}),n.size>=2){const me=Array.
from(n.values()),ie=me[0],ke=me[1];c=!0,i=!1,d=C(ie,ke)||1,f=markerView.scale,h={x:markerView.offsetX,
y:markerView.offsetY},y=L(ie,ke),e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault();
return}if(c)return;i=!0;const W=x(j),re=M();r=E(W,re),a=W,l=re?{x:re.x,y:re.y,w:re.w,h:re.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault()},"start"),P=o(j=>{if(markerState.
mode!=="crop")return;if(n.has(j.pointerId)&&n.set(j.pointerId,{x:j.clientX,y:j.clientY}),c&&n.size>=
2){const te=Array.from(n.values()),B=te[0],V=te[1],de=L(B,V),fe=C(B,V)||1,Q=f*(fe/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Q)),markerView.offsetX=h.x+(de.x-y.x),markerView.
offsetY=h.y+(de.y-y.y),applyMarkerTransform(),renderCropOverlay(),j.preventDefault();return}if(!i||!a||
!l)return;const W=x(j),re=e.width,me=e.height,ie={x:l.x,y:l.y,w:l.w,h:l.h},ke=l.x+l.w,$=l.y+l.h,N=o(
()=>{const te=k(W.x,0,ke-v);ie.x=te,ie.w=ke-te},"applyW"),U=o(()=>{ie.w=k(W.x-l.x,v,re-l.x)},"applyE"),
O=o(()=>{const te=k(W.y,0,$-v);ie.y=te,ie.h=$-te},"applyN"),q=o(()=>{ie.h=k(W.y-l.y,v,me-l.y)},"appl\
yS");switch(r){case"move":{const te=W.x-a.x,B=W.y-a.y;ie.x=k(l.x+te,0,re-l.w),ie.y=k(l.y+B,0,me-l.h);
break}case"w":N();break;case"e":U();break;case"n":O();break;case"s":q();break;case"nw":O(),N();break;case"\
ne":O(),U();break;case"sw":q(),N();break;case"se":q(),U();break;default:break}ie.x=k(ie.x,0,re-ie.w),
ie.y=k(ie.y,0,me-ie.h),markerState.cropRect=ie,renderCropOverlay(),j.preventDefault()},"move"),Z=o(j=>{
n.delete(j.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),e.
releasePointerCapture&&e.releasePointerCapture(j.pointerId),j.preventDefault()},"end");e.addEventListener(
"pointerdown",D),e.addEventListener("pointermove",P),e.addEventListener("pointerup",Z),e.addEventListener(
"pointercancel",Z),e.addEventListener("pointerleave",Z)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const r=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const L=document.createElement("canvas");L.width=r,L.height=l;const C=L.getContext(
"2d");if(!C)return;C.fillStyle="#000000",C.fillRect(0,0,r,l),C.drawImage(n,0,0,r,l);const M=await new Promise(
P=>L.toBlob(P,"image/png"));if(!M){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const E=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([M],`${E}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,r,l),c.drawImage(n,0,0,r,l),markerState.cropRect){const L=r/n.width,C=l/n.height,M=Math.max(0,Math.
floor(markerState.cropRect.x*L)),E=Math.max(0,Math.floor(markerState.cropRect.y*C)),D=Math.min(r,Math.
max(1,Math.floor(markerState.cropRect.w*L))),P=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*C))),Z=document.createElement("canvas");Z.width=D,Z.height=P;const j=Z.getContext("2d");j&&(j.drawImage(
a,M,E,D,P,0,0,D,P),a=Z)}const d=await new Promise(L=>a.toBlob(L,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const h=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([d],`${h}_marked.png`,
{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),k=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=k,setAttachmentSourceForPath(
w,k)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(L=>L!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}o(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,r=null,l=null,c=null,d=[],f=null;const h=o(()=>{f&&clearTimeout(
f);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(
()=>{h(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();
if(!v||!v.length)return y();r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=r.createMediaStreamSource(new MediaStream(v)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=
k=>{const x=k.inputBuffer.getChannelData(0);d.push(new Float32Array(x))},c.connect(l),l.connect(r.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(f=setTimeout(()=>{const k=(e.
name||"video").replace(/\.[^/.]+$/,""),x=encodeWav(d,r.sampleRate),L=new File([x],`${k}.audio.wav`,{
type:"audio/wav"});h(),n(L)},w+250)),await i.play(),i.onended=()=>{const k=(e.name||"video").replace(
/\.[^/.]+$/,""),x=encodeWav(d,r.sampleRate),L=new File([x],`${k}.audio.wav`,{type:"audio/wav"});h(),
n(L)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
if(!e||!e.length)return;const n=Array.from(e).filter(Boolean);if(!n.length)return;const i=n,a=uploadGeneration;
t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=i.length,uploadProgressState.
active+=i.length,updateFilePreview();const r=!!(get("upload-audio-only")&&get("upload-audio-only").checked),
l=getModelMediaSupport(get("model-select").value),c=o(async v=>{let w=null;if(a!==uploadGeneration)return!1;
try{if(isAudioFile(v)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(v)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!v.type||!v.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const k=addUploadRow(v);k.generation=a,updateFilePreview(),w=k.uploadId,uploadProgressState.
perFilePct[w]=0;let x=v;if(r&&isVideoFile(v)){const C=await extractAudioFromVideo(v,k);C?(x=C,updateUploadRowFile(
k,C),k&&k.status&&(k.status.textContent="\u97F3\u58F0\u306E\u307F")):(k&&k.status&&(k.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{x=await applyImageCompression(v,
readComposerCompression()),x!==v&&updateUploadRowFile(k,x)}catch{}if(isUploadCancelled(k))return!1;if(browserFastModeEnabled){
const C=Array.from(browserFastLocalFiles.values()).reduce((M,E)=>M+Number(E.file&&E.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||C+x.size>BROWSER_FAST_MAX_BYTES?(k&&k.status&&
(k.status.textContent="\u4E0A\u9650\u8D85\u904E"),k&&k.row&&k.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(k.uploadId,{file:x,rowObj:k}),
k.status&&(k.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
k.bar&&(k.bar.style.width="100%"),k.row&&(k.row.dataset.browserFastLocal="1"),!0)}const L=await uploadFileWithProgress(
x,k,t);if(!L&&t.cameraCapture&&!isUploadCancelled(k)){const C=document.createElement("button");C.className=
"text-xs border border-gray-700 rounded px-2 py-1",C.textContent="\u518D\u8A66\u884C",C.onclick=()=>{
if(isUploadCancelled(k))return;C.disabled=!0;const M=k.row.getAttribute("data-local-url");M&&URL.revokeObjectURL(
M),k.row.remove(),handleFiles([x],{...t,openModal:!0,skipCompression:!0})},k.row.appendChild(C)}return L}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),d=new Array(i.length).fill(!1);let f=0;const h=Math.
min(UPLOAD_CONCURRENCY,i.length),y=Array.from({length:h}).map(async()=>{for(;;){const v=f++;if(v>=i.
length)break;d[v]=await c(i[v])}});return await Promise.all(y),d}o(handleFiles,"handleFiles"),get("c\
lear-file-btn").onclick=e=>{e.stopPropagation(),resetUploadState()},get("clear-mask-btn")&&(get("cle\
ar-mask-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),get("mask-btn")&&get("mask-in\
put")&&(get("mask-btn").onclick=()=>{if(isIdeogramObjectRemoval()){const e=Array.from(document.querySelectorAll(
"#upload-list .upload-row")).find(t=>getRowImageSource(t));if(!e){showToast("\u5148\u306B\u30DE\u30B9\u30AF\u5BFE\u8C61\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}openMarkerModalForRow(e,!0);return}get("mask-input").click()},get("mask-input").addEventListener(
"change",async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value=
"")}));const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){
const n=String(e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||
typeof window.DOMPurify.sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=
!0,console.error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).
replace(/\n/g,"<br>");const i=protectMathSegments(n);if(t.inline===!0&&typeof window.marked.parseInline==
"function")return window.DOMPurify.sanitize(restoreMathSegments(window.marked.parseInline(i.text),i.
blocks,t));markdownOpenFenceBody=/<\/svg/i.test(n)?findOpenMarkdownFenceBody(n):null;let r="";try{r=
window.marked.parse(i.text)}finally{markdownOpenFenceBody=null}const l=restoreMathSegments(r,i.blocks,
t);return window.DOMPurify.sanitize(l)}o(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");let markdownOpenFenceBody=null;
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
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const d=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return d&&Number(d[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
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
e||""),n=t.split(/\r?\n/),i=[],a=[],r=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",d=[];for(const y of n){
if(!l){const k=y.match(r);if(k){l=k[2],c=String(k[3]||"").trim(),d=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const v=String(y||"").trim();if(v&&v.replace(/\s+/g,"")===l){const k=i[i.length-1];k&&(k.
code=d.join(`
`),k.open=!1),l=null,c="",d=[];continue}d.push(y);const w=i[i.length-1];w&&(w.code=d.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=d.join(`
`),y.open=!0)}const f=i.map((y,v)=>normalizeCanvasBlock(y,v)),h=selectCanvasPreviewBlock(f,t);return{
renderText:a.join(`
`),blocks:f,primaryBlock:h?h.block:null,primaryIndex:h?h.index:-1,rawText:t}}o(parseCanvasMarkdown,"\
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
forEach(d=>{d&&d.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),r.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";r.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(r.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}o(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const f=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",f),d.setAttribute("aria-pressed",f?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}o(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const r=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",h=(String(i&&i.code?i.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${r}`,v=`${y}\u3001${h}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(v)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
r)}</span><span class="canvas-block-chip-preview">${escapeHtml(h)}</span></span><span class="canvas-\
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
null?i.code:""),d=r?isCanvasHtmlPreviewCandidate(l,c):!1,f=r?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",h=r?d?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=h),e.status&&(e.status.textContent=
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
(c,d)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const h=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},d);a.push(h),r===-1&&f.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
r===-1&&(r=0);const l=a[r]||a[0]||null;return{blocks:a,selectedIndex:r,selectedKey:l&&l.key?l.key:t.
codeKey||""}}o(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
var r;if((r=window.ActivityLog)==null||r.log("html_preview.start"),!e)return!1;const t=collectCanvasBlocksFromButton(
e);if(!t||!t.blocks||!t.blocks.length)return!1;const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.
blocks:[],i=n.findIndex(l=>l&&l.key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(
i);const a=t.blocks[t.selectedIndex]||t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.
rawText=a&&a.code!==void 0&&a.code!==null?String(a.code):"",canvasPreviewState.renderText=canvasPreviewState.
rawText,canvasPreviewState.selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.
selectedKey=t.selectedKey||a&&a.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),
canvasPreviewState.lastCanvasData={renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:a,
primaryIndex:canvasPreviewState.selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.
mobileView="preview",syncCanvasPanelViewUi("preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}o(previewCanvasCodeFromButton,
"previewCanvasCodeFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&
e.code!==null?e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(
n,t))return sanitizeHtmlForPreview(t);const a=n?`Canvas Preview: ${n}`:"Canvas Preview",r=escapeHtml(
t||"");return`<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" conte\
nt="width=device-width, initial-scale=1"><title>${escapeHtml(a)}</title><style>
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
"`","~"])for(let c=3;c<=10;c++){const d=l.repeat(c);for(const f of r){const h=`${d}${f}
`,y=`
${d}`,v=h+a+y;i.includes(v)&&(i=i.split(v).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const r of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const d=l.repeat(c);
[`${r}${d}
${i}
${d}`,`${r}
${d}
${i}
${d}`,`
${r}${d}
${i}
${d}`,`
${r}
${d}
${i}
${d}`].forEach(h=>{n.includes(h)&&(n=n.split(h).join(`
`))})}return n}o(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}o(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}o(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[],i=[];if(!t)return{text:"",executions:n};const a=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let r=t.replace(a,(v,w,k,x)=>{i.push(x);const L=String(k||"").trim();try{const C=JSON.parse(L);n.push(
{code:C&&C.code!=null?String(C.code):"",output:C&&C.output!=null?String(C.output):"",images:C&&Array.
isArray(C.images)?C.images.filter(M=>typeof M=="string"):[]})}catch{n.push({code:L,output:""})}return`\

`});const l=/!\[Agentic View\]\(([^)]+)\)/g;if(!n.length&&/!\[Agentic View\]\(/.test(t)&&(r=t.replace(
/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,(v,w,k,x)=>(i.push(
x),n.push({code:w,output:k||"",images:[]}),`
`))),n.length){const v=new Set(n.flatMap(w=>w.images||[]));for(const w of t.matchAll(l)){if(v.has(w[1]))
continue;let k=i.length-1;for(;k>0&&i[k]>w.index;)k--;const x=n[k];x.images=Array.from(new Set([...x.
images||[],w[1]]))}r=r.replace(l,"")}n.forEach(v=>{v.code&&(r=stripExactFencedBlock(r,"python",v.code),
r=stripExactFencedBlock(r,"py",v.code)),r=stripVisiblePythonOutputBlock(r,v.output)});const c=new Set(
n.flatMap(v=>v.images||[]));r=r.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
(v,w,k,x)=>!w&&c.has(x)&&/\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(x)?`![${k}](${x})`:v);const d=n.
slice().reverse().find(v=>v.images&&v.images.length),f=r.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,
""),h=new Set(Array.from(f.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),v=>v[1])),y=(d?d.images:[]).filter(
v=>/^\/files\/[A-Za-z0-9_./%-]+$/.test(v)&&!h.has(v));return y.length&&(r+=`

**Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF**

`+y.map(v=>`![Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF](${v})`).join(`

`)),r=r.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:r,executions:n}}o(extractPythonExecutionsFromContent,
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
r=escapeHtml(i)}catch{r=escapeHtml(i)}const l=(e.images||[]).filter(x=>/^\/files\/[A-Za-z0-9_./%-]+$/.
test(x)).map(x=>`![Python output](${x})`).join(`
`),c=l?`<div class="python-section"><div class="python-label">Images</div>${sanitizeMarkdownHtml(marked.
parse(l))}</div>`:"",d=escapeHtml(a),f=encodeURIComponent(i).replace(/'/g,"%27"),h=encodeURIComponent(
a).replace(/'/g,"%27"),y=hashString(`pyexec-detail
${i}
${a}
${t}`),v=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",w=`<button class="download-btn" data-\
code="${f}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,k=`<button class="coding-target-btn" data-code="${f}" data-code-key="${y}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${y}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(v)}</span><div class="code-actions">${k}${w}<button class="copy-btn" data-copy="code" \
data-code="${f}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${h}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${r}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${d}</code></pre></div>${c}</div></div>`}o(buildPythonExecDetailBoxHtml,"bu\
ildPythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/pyt\
hon-execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"",
"/python-execution")}showModal("python-exec-modal")}o(showPythonExecDetailModal,"showPythonExecDetai\
lModal");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("p\
ython-exec-modal-body"),a=get("python-exec-modal-title");if(!n||!i)return;const r=t&&Array.isArray(t.
python_executions)?t.python_executions:[];if(!r.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(a){const l=r.length>1?`\uFF08${r.length}\u4EF6\uFF09`:"";a.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${l}`}i.innerHTML=r.map((l,c)=>buildPythonExecDetailBoxHtml(l,c,r.length)).join(
""),i.querySelectorAll("img").forEach(l=>{l.addEventListener("click",()=>openImageViewer(l.dataset.viewerSrc||
l.src,"#python-exec-modal-body img"))}),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(
!0,{persist:!1})),showPythonExecDetailModal(e)}o(openPythonExecDetail,"openPythonExecDetail"),window.
openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){get("python-exec-moda\
l")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&history.back())}o(
closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;function buildAiMarkdownHtml(e){
const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(t.text,t.notes),i=canvasModeEnabled?
parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(
i),refreshCanvasPreviewPanel());const a=document.createElement("div");return a.className="prose pros\
e-invert text-sm break-words",a.innerHTML=sanitizeMarkdownHtml(i.renderText),wrapRenderedSvgBoxes(a),
lowBandwidthMode||(maybeNeedsHighlight(i.renderText,a)&&ensureHighlightLoaded().catch(()=>{}),maybeNeedsMathJax(
i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),a.outerHTML}o(buildAiMarkdownHtml,"buildAiMarkdo\
wnHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes),r=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,
blocks:[],primaryBlock:null,rawText:a};if(canvasModeEnabled&&(updateCanvasPreviewState(r),refreshCanvasPreviewPanel()),
n.incrementalMath){const c=document.createElement("template");c.innerHTML=sanitizeMarkdownHtml(r.renderText,
{streamMathSegments:!0});const d=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(h=>{const y=h.getAttribute("data-stream-math-key");y&&d.set(y,h)});const f=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(h=>{const y=d.get(h.getAttribute(
"data-stream-math-key"));y?h.replaceWith(y):f.push(h)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(
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
o(wrapRenderedSvgBoxes,"wrapRenderedSvgBoxes");const SEARCH_COMPLETE_HTML='<div class="search-box vi\
sible mb-2"><i class="fas fa-check-circle text-green-400"></i> Search complete</div>';function updateSearchBox(e,t,n){
return n!=="searching"&&n!=="done"||(t||(e.insertAdjacentHTML("afterbegin",SEARCH_COMPLETE_HTML),t=e.
querySelector(".search-box")),t.classList.toggle("animate-pulse",n==="searching"),t.innerHTML=n==="s\
earching"?'<i class="fas fa-globe"></i> Searching web...':'<i class="fas fa-check-circle text-green-\
400"></i> Search complete'),t}o(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,r,l=null,c=!0,d=null,f=null,h=null,y=null,v=null,w=null,k=null,x=null,L=!0,C=null,M=null,E=null,D=null){
const P=t==="user",Z=P?"bg-blue-600":"bg-gray-700",j=P?"justify-end":"justify-start";messageStore[e]=
n;const W=!P&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},re=P?n:W.text;let me=f;
if(me==null){const ue=h!=null?Number(h):0,Se=y!=null?Number(y):0;(h!=null||y!=null)&&(me=ue+Se)}const ie=costCacheUsage(
a);messageMeta[e]={tokens_in:h,tokens_out:y,tokens_total:me,tokens_content:w,tokens_thought:k,cache_read_tokens:ie?
ie.read:null,cache_write_tokens:ie?ie.write:null,cache_write_ttls:ie?ie.ttls:null,is_encrypted:v,role:t,
model:r,parent_id:C,quote_text:d,image_url:i,image_mask:D,skill_name:M,batch_job:E,python_executions:P?
[]:W.executions||[]};let ke="";d&&(ke=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-bl\
ue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50\
"></i>${escapeHtml(d)}</div>`);const $=M?`<div class="skill-notice visible"><i class="fas fa-wand-ma\
gic-sparkles"></i><span>Skill: <span class="font-bold">${escapeHtml(M)}</span></span></div>`:"";let N="";
if(a&&!P){let ue="";try{const Se=JSON.parse(a);ue=Se.text||"",Se.search_used===!0&&(N=SEARCH_COMPLETE_HTML)}catch{
ue=a}ue&&(N+=`<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content c\
ollapsed">${escapeHtml(ue)}</div></div>`)}let U="";const O=new Set;if(!P){const ue=String(re||"").replace(
/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,""),Se=Array.from(ue.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),
Me=>Me[1]);Se.push(...(W.executions||[]).flatMap(Me=>Me.images||[])),Se.forEach(Me=>{if(/^https?:\/\//i.
test(Me)&&!Me.startsWith(FILE_BASE_URL))try{if(new URL(Me).origin!==window.location.origin)return}catch{
return}const we=normalizeAttachmentPath(Me);we&&O.add(we)})}if(i)try{const ue=JSON.parse(i);if(ue.length){
const Se=[];if(ue.forEach(Me=>{let we=Me,De="unknown";if(we&&typeof we=="object"&&(De=normalizeAttachmentSource(
we.source),we=we.filepath||we.path||we.url||we.file||""),we=normalizeAttachmentPath(we)||we,!we||!P&&
O.has(we))return;setAttachmentSourceForPath(we,De);const wt=we.replace(/^\d+\//,""),he=buildFileUrl(
wt),pt=buildAttachmentPreviewUrl(wt),Pe=we.split("/").pop(),gt=Pe.split(".").pop().toLowerCase();["j\
pg","jpeg","png","webp","gif"].includes(gt)?Se.push(buildChatImageHtml(pt,{viewerSrc:he,alt:Pe,title:Pe,
filename:Pe})):Se.push(`<div class="file-thumb bg-gray-800 border border-gray-600 rounded flex flex-\
col items-center justify-center cursor-pointer hover:bg-gray-700" onclick="window.open('${he}')" tit\
le="${Pe}"><i class="fas fa-file text-2xl text-gray-400 mb-1"></i><span class="text-[9px] truncate w\
-20 text-center">${Pe}</span></div>`)}),Se.length>0){let Me="grid-multi";Se.length===1?Me="grid-1":Se.
length===2?Me="grid-2":Se.length===3?Me="grid-3":Se.length===4&&(Me="grid-4"),U=`<div class="image-g\
rid ${Me}">${Se.join("")}</div>`}}}catch{}let q="";if(P&&D)try{const ue=normalizeAttachmentPath(D)||
String(D),Se=ue.replace(/^\d+\//,""),Me=buildFileUrl(Se),we=buildAttachmentPreviewUrl(Se),De=ue.split(
"/").pop();q=`<div class="mt-2 max-w-xs rounded-lg border border-amber-300/40 bg-black/10 p-2"><div \
class="mb-1 text-[10px] font-semibold text-amber-100"><i class="fas fa-mask mr-1" aria-hidden="true"\
></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(we,
{viewerSrc:Me,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:De})}</div>`}catch{}const te=P?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,B=`<div class="msg-controls absolute -top-5 right-\
0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', t\
his)"><i class="fas fa-copy"></i></button>${P?`<button class="ctrl-btn edit-btn" data-id="${e}"><i c\
lass="fas fa-pen"></i></button>`:""}${te}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\
\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,V=[];!P&&r&&V.push(escapeHtml(r));const de=[];
if(h!=null&&de.push(`In ${h}`),y!=null){let ue=`Out ${y}`;k!=null&&Number(k)>0&&(ue+=` (Thought ${k}\
)`),de.push(ue)}if(de.length||f!=null){const ue=de.length?de.join(" / "):`${f} tokens`;V.push(`<butt\
on class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${e}\
')">${ue}</button>`)}if(v!=null){const ue=v?"fa-lock":"fa-lock-open",Se=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\
\uFF09":"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":v?"En\
crypted":"Plain",Me=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:t\
ext-cyan-200":"text-slate-300/80 hover:text-white";V.push(`<button class="${Me}" title="${Se}" oncli\
ck="openEncryptionSettings('${e}')"><i class="fas ${ue}"></i></button>`)}if(!P&&W.executions&&W.executions.
length){const ue=W.executions.length,Se=ue>1?`Python \xD7${ue}`:"Python";V.push(`<button type="butto\
n" class="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="\
Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${Se}</span><\
/button>`)}const fe=V.length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono me\
ssage-footer-meta">${V.join(" \u2022 ")}</div>`:"";let Q;const le=!P&&E?(()=>{const ue=String(E.state||
"").toUpperCase(),Se=E.status_text||(ue==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ue==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ue==="JOB_STATE_FAILED"||
ue==="JOB_STATE_CANCELLED"||ue==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ue==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(Se)}</div></div>`})():
"";P?Q=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(Q=le+(re&&String(re).trim()?buildAiMarkdownHtml(re):E?'<div class="content-area prose p\
rose-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(re)),Q.includes("content-area")||(Q=Q.replace("prose ","content-area prose ")));
let ee="";if(l){const ue=l.siblings[l.current-2],Se=l.siblings[l.current];ee=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ue}\
)" ${ue?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${Se}\
)" ${Se?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const pe=c?"fade-in":"",se=document.createElement("div");se.className=`flex ${j} m\
b-4 ${pe} relative message-group group`,se.id=`msg-${e}`,se.innerHTML=`<div class="message-bubble ${Z}\
 text-white p-4 rounded-2xl shadow-md relative">${B}${ke}${$}${N}${Q}${U}${q}${ee}${fe}</div>`;const Fe=x||
get("chat-container");return Fe&&(Fe.appendChild(se),L&&scrollToBottom(),P||(se.isConnected&&queueMessageDecorations(
se,re),syncCodingTargetButtons(se),syncCodingModeUi(codingModeEnabled,{persist:!1}))),se}o(renderMessage,
"renderMessage");function showTokenDetailModal(e=null,t=!1){if(location.pathname!=="/token-details"){
const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==null&&(n.messageId=e),history.pushState(
n,"","/token-details")}showModal("token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");
function openTokenDetail(e){var C;const t=messageMeta[e];if(!t||!get("token-detail-modal"))return;get(
"token-detail-usage").classList.remove("hidden"),get("token-detail-estimate").classList.add("hidden"),
cancelPromptTokenDetail();const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:"-",
a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",r=t.tokens_out!==null&&t.tokens_out!==void 0?
t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.tokens_thought!==
null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===null||t.is_encrypted===void 0?
"-":t.is_encrypted?"Encrypted":"Plain",f=t.cache_read_tokens!==null&&t.cache_read_tokens!==void 0?t.
cache_read_tokens:null,h=t.cache_write_tokens!==null&&t.cache_write_tokens!==void 0?t.cache_write_tokens:
null,y=f!==null||h!==null;get("token-detail-total").innerText=i,get("token-detail-in").innerText=a,get(
"token-detail-out").innerText=r,get("token-detail-content").innerText=l,get("token-detail-thought").
innerText=c,get("token-detail-encrypted").innerText=d;const v=get("token-detail-cache-read-row"),w=get(
"token-detail-cache-write-row"),k=get("token-detail-cache-write-ttl");if(v&&v.classList.toggle("hidd\
en",!y),w&&w.classList.toggle("hidden",!y),get("token-detail-cache-read")&&(get("token-detail-cache-\
read").innerText=f===null?"-":f),get("token-detail-cache-write")&&(get("token-detail-cache-write").innerText=
h===null?"-":h),k){const M=["5m","1h","30m","12h"].map(E=>{var P;const D=Number(((P=t.cache_write_ttls)==
null?void 0:P[E])||0);return D>0?`${E}: ${D}`:""}).filter(Boolean);k.textContent=M.length?`TTL ${M.join(
" \xB7 ")}`:"",k.classList.toggle("hidden",!y||!M.length)}const x=get("token-detail-cost");if(x&&(x.
replaceChildren(),x.classList.toggle("hidden",t.role!=="total"),t.role==="total")){for(const M of[costEstimateLabel(
t.cost_estimate),`\u96C6\u8A08\u3057\u305FAPI\u547C\u3073\u51FA\u3057: ${((C=t.cost_estimate)==null?
void 0:C.requests)||0}\u56DE`,COST_ESTIMATE_NOTE,`\u516C\u5F0F\u6599\u91D1\u78BA\u8A8D\u65E5: ${(costPricing==
null?void 0:costPricing.checked_at)||"\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"}`]){
const E=document.createElement("p");E.textContent=M,x.appendChild(E)}for(const[M,E]of Object.entries(
(costPricing==null?void 0:costPricing.sources)||{})){const D=document.createElement("a");D.href=E,D.
textContent=`${M} \u516C\u5F0F\u6599\u91D1 `,D.target="_blank",D.rel="noopener noreferrer",D.className=
"underline",x.appendChild(D)}}const L=t.model?`${t.model} (${t.role})`:`${t.role}`;get("token-detail\
-title").innerText=L,showTokenDetailModal(e)}o(openTokenDetail,"openTokenDetail");function closeTokenDetail(e=!1){
cancelPromptTokenDetail(),get("token-detail-modal")&&(hideModal("token-detail-modal"),!e&&location.pathname===
"/token-details"&&history.back())}o(closeTokenDetail,"closeTokenDetail");function openEncryptionSettings(e){
const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}o(openEncryptionSettings,"openEncrypti\
onSettings");function openEncryptionModal(e){if(!get("encryption-status-modal"))return;const n=get("\
encryption-status-title"),i=get("encryption-status-body"),a=get("encryption-status-admin-actions"),r=get(
"encryption-status-admin-toggle"),l=!!e;l?(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),
i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
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
ck",f),i&&i.removeEventListener("click",h),a&&a.removeEventListener("click",h),e.removeEventListener(
"click",y,!0)}o(c,"cleanup");function d(v){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(v)}o(d,"finalize");
function f(){d(!0)}o(f,"onOk");function h(){d(!1)}o(h,"onCancel");function y(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),h())}o(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",h),a&&a.addEventListener("click",h),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${r}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("\
chat-container");if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",d);else{
const h=document.createElement("div");h.innerHTML=d;const y=h.firstElementChild;y&&f.appendChild(y)}
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
var c,d;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((d=t.retryDelayMs)!=
null?d:180)||0),a=!!t.notifyOnFailure,r=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
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
\u4F7F\u3044\u65B9: ${n.label} default / none`,"info"),!1;const r=get(i.selectId),l=a.toLowerCase(),
c=r?Array.from(r.options).find(d=>d.value.toLowerCase()===l||d.textContent.trim().toLowerCase()===l):
null;return!r||!c?(showToast(`${n.label}: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1):(r.value=c.value,r.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),
showToast(`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const r=a.
toLowerCase(),l=activeThinkingSelect(),c=!!(l&&l.id==="reasoning-effort"),d=c?{off:"none",none:"none",
min:"low",minimal:"low",low:"low",mid:"medium",medium:"medium",high:"high",xhigh:"xhigh",max:"max"}:
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},f=parseSlashToggleArgument(
a),h=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(d,r)){const y=d[r],v=l?Array.from(l.options).
find(w=>w.value===y&&!w.disabled&&!w.classList.contains("hidden")):null;if(l&&!v)return showToast(`T\
hinking: \u300C${r}\u300D\u306F\u3053\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1;if(y==="none"){if(h&&h.disabled)return showToast("Thinking: \u3053\u306E\u30E2\u30C7\u30EB\u3067\u306FOFF\u306B\u3067\u304D\u307E\u305B\u3093",
"warning"),!1;h&&h.checked?(h.checked=!1,h.dispatchEvent(new Event("change",{bubbles:!0}))):l&&(l.value=
"none",l.dispatchEvent(new Event("change",{bubbles:!0})))}else h&&!h.checked&&!h.disabled&&(h.checked=
!0,h.dispatchEvent(new Event("change",{bubbles:!0}))),l&&(l.value=y,l.dispatchEvent(new Event("chang\
e",{bubbles:!0})));return refreshMinimalOptionItems(),showToast(`Thinking: ${r}`,"success"),!0}if(f===
void 0)return showToast(c?"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high / xhigh / max":
"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high","info"),!1}if(i.checkboxId&&a){const r=parseSlashToggleArgument(
a);if(r===void 0)return showToast(`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const l=get(i.
checkboxId);if(r!==null&&l&&l.checked===r)return showToast(`${i.label}: ${r?"ON":"OFF"}`,"info"),!0}
return handleMinimalOptionClick(i),!0}o(executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){
const t=String(e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||
"",i=n.match(/^[a-z][\w-]*/i);return i?i[0]:n}o(extractSlashCommandToken,"extractSlashCommandToken");
function hideSlashCommandSuggestions(){const e=get("slash-command-suggestions");e&&e.classList.add("\
hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=0}o(hideSlashCommandSuggestions,"hideSlashCom\
mandSuggestions");function showPendingSlashCommandIndicator(e){const t=get("slash-command-indicator"),
n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.find(r=>r.id===e);n.textContent=
i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");const a=get("prompt-input");a&&
i&&(a.dataset.originalPlaceholder=a.placeholder,a.placeholder=i.argumentHint||"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\
\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}o(showPendingSlashCommandIndicator,"showPe\
ndingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){const e=get("slash-command-\
indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get("prompt-input");t&&
t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.dataset.originalPlaceholder);
const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}o(
hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
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
                `;let x=!1;k.addEventListener("pointerdown",L=>{typeof L.button=="number"&&L.button!==
0||(L.preventDefault(),x=!0,selectSlashCommand(v.id))}),k.addEventListener("click",L=>{L.preventDefault(),
x||selectSlashCommand(v.id)}),k.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(k)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,d=r.top,f=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&d>c){const v=Math.min(f,d-h);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(f,c-h);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
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
return;removeEphemeralMessageControls(c);const d=c.querySelector(".message-bubble");if(!d||!i.length)
return;const f=document.createElement("div");f.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,v])=>{const w=AI_SETTING_JUMP_TARGETS[y]||{label:y},k=document.createElement("button");k.
type="button",k.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const x=document.createElement("span");x.className="min-w-0 flex-1";const L=document.createElement(
"span");L.className="block text-xs font-bold text-blue-200",L.textContent=w.label;const C=document.createElement(
"span");C.className="block mt-0.5 text-[11px] text-gray-300 break-words",C.textContent=formatAiSettingValue(
v);const M=document.createElement("i");M.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",x.appendChild(L),x.appendChild(C),k.appendChild(x),k.appendChild(M),k.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(k)});const h=d.querySelector(".message-footer-\
meta");h?d.insertBefore(f,h):d.appendChild(f),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const r=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${r}" class="flex justify-start mb-4 ai-pending-row f\
ade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-\
tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const d=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),f=get(r);if(f&&f.remove(),d&&d.status==="ok"&&d.mode==="inspect"&&d.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(d.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
d.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(d.current,t,"inspect");
return}if(d&&d.status==="ok"&&d.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
d.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(d.applied).length}\u9805\u76EE\uFF09`,"success");try{const v=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(v),cacheUserSettings(v)}catch{}renderAiSettingsResultBubble(
d.applied,t);return}const h=d.message||d.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${h}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${h}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(h,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const d=get(r);d&&d.remove();const f=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(f),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
o(runAiSettingsCommand,"runAiSettingsCommand");function hideSkillSuggestions(){const e=get("skill-su\
ggestions");e&&e.classList.add("hidden"),skillSuggestionsVisible=!1,skillSelectedIndex=0}o(hideSkillSuggestions,
"hideSkillSuggestions");function showSkillSuggestions(e=""){const t=get("skill-suggestions"),n=get("\
skill-suggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedSkills||loadedSkills.length===
0){hideSkillSuggestions();return}const a=e.toLowerCase(),r=loadedSkills.filter(w=>w.name.toLowerCase().
includes(a)||w.description&&w.description.toLowerCase().includes(a));if(r.length===0){hideSkillSuggestions();
return}skillSelectedIndex>=r.length&&(skillSelectedIndex=0),n.innerHTML="",r.forEach((w,k)=>{const x=document.
createElement("div");x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-\
gray-700 ${k===skillSelectedIndex?"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas fa-wand-magic-sparkles w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(w.name)}</div>
                        ${w.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
w.description)}</div>`:""}
                    </div>
                `,x.onclick=()=>selectSkillSuggestion(w),x.onmouseenter=()=>{skillSelectedIndex=k,showSkillSuggestions(
e)},n.appendChild(x)});const l=i.getBoundingClientRect(),c=window.innerHeight,d=c-l.bottom,f=l.top,h=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&f>d){const w=Math.min(h,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(h,d-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),skillSuggestionsVisible=!0}o(showSkillSuggestions,
"showSkillSuggestions");function selectSkillSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideSkillSuggestions(),
activateSkill(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}o(selectSkillSuggestion,
"selectSkillSuggestion");function browserFastModeIneligibility(e){const t=String(get("model-select")?
get("model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!t.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(t))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeSkill)return"Skills\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
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
"warning",!0),t}o(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(e=browserFastLocalFiles,t=!1){
const n=Array.from(e.entries()),i=[];for(const[a,r]of n){if(!r||!r.file||!r.rowObj)throw new Error("\
\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");if(r.
rowObj.status&&(r.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(r.file,r.rowObj))throw new Error(`${r.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);const c=r.rowObj.row.getAttribute("da\
ta-filename");c&&i.push(c),t?currentImageUrls=currentImageUrls.filter(d=>d!==c):browserFastLocalFiles.
delete(a)}return i}o(uploadBrowserFastLocalFiles,"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){
const t=get("enable-thinking");if(!t||!t.checked)return null;const n=String(get("thinking-level")?get(
"thinking-level").value:"high").toLowerCase();if(e.includes("2.5")){const a=Number(get("thinking-bud\
get")?get("thinking-budget").value:4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(a)?
Math.max(0,Math.min(32768,Math.trunc(a))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["M\
EDIUM","HIGH"].includes(i)&&(i="MEDIUM"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&
(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}o(browserFastThinkingConfig,"browserFastThinkingC\
onfig");function browserFastPythonBoxHtml(e){return`<div class="code-wrapper python-box collapsed" d\
ata-py-id="${e}" data-collapsed="true" data-code-key="${e}"><div class="code-header"><span class="co\
de-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button cl\
ass="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></\
i></button><button class="copy-btn" data-copy="code" data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\
\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" titl\
e="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></button></div></div><div class="cod\
e-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs langu\
age-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Out\
put</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`}o(
browserFastPythonBoxHtml,"browserFastPythonBoxHtml");function updateBrowserFastPythonBox(e,t,n){if(e){
if(t==="code"){const i=n==null?"":String(n),a=e.querySelector(".python-code");a&&(a.textContent=i,a.
removeAttribute("data-highlighted"),queueHighlight(e,i));const r=e.querySelector('.copy-btn[data-cop\
y="code"]');r&&r.setAttribute("data-code",encodeURIComponent(i).replace(/'/g,"%27"))}else if(t==="ou\
tput"){const i=n==null?"":String(n),a=e.querySelector(".python-output");a&&(a.textContent=i);const r=e.
querySelector('.copy-btn[data-copy="output"]');r&&r.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}}}o(updateBrowserFastPythonBox,"updateBrowserFastPythonBox");function collectBrowserFastPromptOptions(){
const e=o((i,a)=>get(i)?!!get(i).checked:a,"checked"),t=o(i=>get(i)?get(i).value:null,"value"),n=browserFastPreviousOptions&&
browserFastPreviousOptions.checks&&Object.prototype.hasOwnProperty.call(browserFastPreviousOptions.checks,
"enable-skills")?!!browserFastPreviousOptions.checks["enable-skills"]:!0;return{canvas_mode:!!canvasModeEnabled,
enable_search:e("enable-search",!1),enable_url_context:e("enable-url-context",!1),enable_maps:e("ena\
ble-maps",!1),enable_python:e("enable-python",!1),enable_file_creation:e("enable-file-creation",!0),
enable_mcp:typeof isMcpEnabledForSend=="function"?!!isMcpEnabledForSend():!1,enable_skills:n,enable_thinking:e(
"enable-thinking",!1),thinking_level:t("thinking-level"),thinking_budget:t("thinking-budget"),reasoning_effort:t(
"reasoning-effort"),safety_setting:t("safety-setting"),enable_prompt_caching:e("enable-prompt-cache",
!1)}}o(collectBrowserFastPromptOptions,"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e,t){
var q,te,B;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),r=[];for(const V of a)r.push({inlineData:{
mimeType:V.file.type,data:await fileToBase64Payload(V.file)}});r.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const d={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:r}],generationConfig:l};!!(get("enable-python")&&get("enable-python").
checked)&&(d.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden");const h=userAutoScroll&&isChatNearBottom();
h||pauseChatAutoScroll();const y=collectImageUrlsForSend(),v=Array.from(((q=get("file-preview-thumbs"))==
null?void 0:q.querySelectorAll("img"))||[]).map(V=>V.cloneNode(!0)),w=renderMessage(Date.now(),"user",
e,JSON.stringify(collectImageUrlsForSend()),null,null,null,!1,currentQuote,null,null,null,null,null,
null,null,h);if(v.length&&!y.length){const V=document.createElement("div");V.className="flex gap-2 f\
lex-wrap",v.forEach(de=>{de.removeAttribute("onclick"),V.appendChild(de)}),w.querySelector(".message\
-bubble").appendChild(V)}const k=takePromptForSend(),x=animatePromptToMessage(w,k,h),L=`browser-fast\
-${Date.now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start\
 mb-4 ai-pending-row fade-in"><div id="${L}" class="message-bubble ai-pending-bubble bg-gray-700 tex\
t-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(n,"Gemini\u3078\u76F4\u63A5\u9001\
\u4FE1\u4E2D...")}</div></div>`);const C=get(L),M=startPendingElapsed(C,t);activeStreamingBubbleId=L,
setSendBtnToStopMode(),h&&resumeChatAutoScroll(),abortController=new AbortController;let E="",D="",P=null;
const Z=new Set,j=[];let W=null,re=null,me=!1;const ie={},ke=[];let $=null,N="";const U=window.ProgressSpinner?
window.ProgressSpinner.startFlow("browserFast"):null;let O=!1;try{const V=await fetch(`https://gener\
ativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json","x-goog-api-key":browserFastApiKey},body:JSON.
stringify(d),signal:abortController.signal}));if(!V.ok){const Me=await V.json().catch(()=>({}));throw new Error(
Me&&Me.error&&Me.error.message?Me.error.message:`Gemini API HTTP ${V.status}`)}window.ConnectionMonitor&&
(O=!0,window.ConnectionMonitor.operationStarted()),U&&U.setPhase("waiting"),k.accept(!1),(te=window.
ActivityLog)==null||te.log("send.stream.start");let de=!1;const fe=V.body.getReader(),Q=new TextDecoder;
let le="";const ee=o(Me=>{var he,pt;const we=Me.split(/\r?\n/).filter(Pe=>Pe.startsWith("data:")).map(
Pe=>Pe.slice(5).trim()).join("");if(!we||we==="[DONE]")return;const De=JSON.parse(we);if(Number.isInteger(
(he=De.usageMetadata)==null?void 0:he.promptTokenCount)){P=P||{};for(const Pe of["promptTokenCount",
"candidatesTokenCount","thoughtsTokenCount","cachedContentTokenCount"])Number.isInteger(De.usageMetadata[Pe])&&
De.usageMetadata[Pe]>=0&&(P[Pe]=De.usageMetadata[Pe])}for(const Pe of De.candidates||[])for(const gt of((pt=
Pe.groundingMetadata)==null?void 0:pt.webSearchQueries)||[])typeof gt=="string"&&gt.trim()&&Z.add(gt);
if(Z.size&&(P=P||{},P.grounding_queries=Z.size),De.error)throw new Error(De.error.message||"Gemini A\
PI error");if((Array.isArray(De.candidates)?De.candidates:[]).forEach(Pe=>{(Pe&&Pe.content&&Array.isArray(
Pe.content.parts)?Pe.content.parts:[]).forEach(Je=>{if(Je&&typeof Je.thoughtSignature=="string"&&!j.
includes(Je.thoughtSignature)&&j.push(Je.thoughtSignature),Je&&Je.executableCode&&typeof Je.executableCode.
code=="string"){const mt=Je.executableCode.code;E+=`
\`\`\`python
${mt}
\`\`\`
`,$=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,N=mt,ie[$]||(C.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml($)),ie[$]=C.querySelector(`[data-py-id="${$}"]`)),updateBrowserFastPythonBox(
ie[$],"code",mt);return}if(Je&&Je.codeExecutionResult&&typeof Je.codeExecutionResult.output=="string"){
const mt=Je.codeExecutionResult.output;E+=`
**Output:**
\`\`\`
${mt}
\`\`\`
`;const Ct=$||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;ke.push({code:N||
"",output:mt}),ie[Ct]||(C.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(Ct)),ie[Ct]=C.querySelector(
`[data-py-id="${Ct}"]`)),updateBrowserFastPythonBox(ie[Ct],"output",mt);return}const At=typeof Je.text==
"string"?Je.text:"";At&&(Je.thought===!0?D+=At:E+=At)})}),!me&&(E||D)){beginPendingToStreamTransition(
C);const Pe=C.querySelector(".content-area");Pe&&Pe.remove(),me=!0}D&&(re||(C.insertAdjacentHTML("af\
terbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"\
><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></d\
iv></div>'),re=C.querySelector(".thought-content")),re.textContent=D),E&&(W||(W=document.createElement(
"div"),W.className="content-area prose prose-invert text-sm break-words",C.appendChild(W)),renderAiMarkdownInto(
W,E,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:Me,value:we}=await fe.
read();if(!Me&&!de&&(de=!0,(B=window.ActivityLog)==null||B.log("send.stream.first")),Me)break;window.
ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),U&&U.setPhase("receiving"),le+=Q.decode(
we,{stream:!0});const De=le.split(/\r?\n\r?\n/);le=De.pop()||"",De.forEach(ee)}if(le+=Q.decode(),le.
trim()&&ee(le),!E.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
W&&renderAiMarkdownInto(W,E,{incrementalMath:!0}),re&&re.classList.add("collapsed"),ke.length&&(E+=ke.
map(Me=>`
\`\`\`pyexec
${JSON.stringify(Me)}
\`\`\`
`).join(""));let pe=[];a.length&&(U&&U.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),pe=await uploadBrowserFastLocalFiles(k.files,!0)),U&&U.setPhase("saving");const se=[...y,
...pe],Fe=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:E,pricing_usage:P,thought_content:D,model:n,image_urls:se,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:j,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),C),ue=await Fe.json().
catch(()=>({}));if(!Fe.ok||!ue.thread_id)throw new Error(ue.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Se=!currentThreadId;currentThreadId=String(ue.thread_id),currentParentId=ue.assistant_message_id||
null,currentLeafId=ue.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!Se}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(V){if(x(),k.restore(),V.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${V.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const de=V.message||"\
\u30A8\u30E9\u30FC";C&&C.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(de));try{let fe=E||
"";ke.length&&(fe+=ke.map(se=>`
\`\`\`pyexec
${JSON.stringify(se)}
\`\`\`
`).join(""));const Q=buildChatErrorMarkdown(de,fe),le=a.length?[]:y,ee=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:Q,
pricing_usage:P,thought_content:D||"",model:n,image_urls:le,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:
null,thought_signatures:j,turnstile_token:botTurnstileTokenForRequest()}),signal:abortController&&!abortController.
signal.aborted?abortController.signal:void 0}),C),pe=await ee.json().catch(()=>({}));if(ee.ok&&pe.thread_id){
const se=!currentThreadId;currentThreadId=String(pe.thread_id),currentParentId=pe.assistant_message_id||
null,currentLeafId=pe.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!se}),applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(fe){
sendClientDebugLog("error",`Browser fast error persist failed: ${fe&&fe.message?fe.message:fe}`)}}}finally{
k.releaseFiles(),M&&M(),O&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),U&&U(),
setSendBtnToSendMode(),activeStreamingBubbleId===L&&(activeStreamingBubbleId=null),abortController=null,
updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
o(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var Gt,rn,Qt,St,ln;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const K=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(K);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const K=pendingSlashCommand,
_e=a.trim(),Ee=get("model-select")?get("model-select").value:null;if(K==="settings"){if(!_e){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!Ee){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
_e,Ee)}else executeMinimalSlashCommand(K,_e)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const r=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(r&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(K=>K.id===
r[1].toLowerCase())){executeMinimalSlashCommand(r[1].toLowerCase(),r[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const l=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(l&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(l)setBrowserFastModeEnabled(!1);else{const K=browserFastModeIneligibility(
a);if(!K){try{await sendBrowserFastMessage(a,t)}catch(_e){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${_e.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${K}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(_e){showToast(
_e.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),d=c.map(K=>K.path),f=c.filter(K=>normalizeAttachmentSource(
K.source)==="upload").map(K=>K.path),h=getModelMediaSupport(get("model-select").value),y=d.some(K=>isAudioPath(
K)),v=d.some(K=>isVideoPath(K)),w=(get("model-select").value||"").toLowerCase(),k=get("enable-python"),
x=!!(k&&k.checked);if(y&&!h.audio||v&&!h.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&d.length===0)return;if(isMistralOcrModel(
w)){const K=/https?:\/\/\S+/i.test(a);if(d.filter(Ee=>isAudioPath(Ee)||isVideoPath(Ee)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!d.length&&!K){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const L=a.trim();if(/^\/settings(?:\s|$)/i.
test(L)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(L)){const K=L.replace(/^\/settings\s*/i,"").trim();
if(!K){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ee=get("prompt-input");Ee.value="/settings ";const qe=extractSlashCommandToken(Ee.value);
lastSlashFilter=qe,showSlashCommandSuggestions(qe),Ee.focus();return}const _e=get("model-select")?get(
"model-select").value:null;if(!_e){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
K,_e);return}if(isGeminiLocalPythonMode(w,y,v,x)&&!await confirmGeminiLocalPythonSwitch())return;let C=null,
M=[];if(codingModeEnabled){const K=collectCodingCandidates(a),_e=K.filter(et=>et.prompt_source),Ee=K.
filter(et=>!et.prompt_source),qe=_e.reduce((et,Tt)=>et+String(Tt.code||"").length,0);if(qe>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let ot=3e5-qe;const ht=[];for(let et=Ee.length-1;et>=0;et--){const Tt=String(Ee[et].
code||"").length;Tt>ot||(ht.unshift(Ee[et]),ot-=Tt)}M=codingTargetSelection?ht.slice(-1):[..._e,...ht];
const Lt=_e.length?_e[_e.length-1]:null;if(C=codingTargetSelection?M[0]:Lt||M[M.length-1]||null,codingModeEffective=
!!(C&&String(C.code||"").trim()),codingModeEffective&&C.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const et=String(((Gt=get(
"model-select"))==null?void 0:Gt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(et)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const E=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${d.length} search=${get("enable-search").checked}`);const D=a,P=hasMarkerHint()?MARKER_HINT_TEXT:
null,Z=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&d.length===0){showToast("Ma\
sk \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if(Z&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=userAutoScroll&&isChatNearBottom();j||pauseChatAutoScroll();const W=editingMessageId,
re=promptSettingsBeforeEdit;promptSettingsBeforeEdit=null;const me=currentParentId,ie=W!=null;W&&(editingMessageId=
null,setEditUi(!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const ke=[],$=o(
K=>{if(K==null)return;let _e=document.getElementById(`msg-${K}`);for(;_e;)_e.classList&&_e.classList.
contains("message-group")&&(ke.push({node:_e,prevDisplay:_e.style.display}),_e.style.display="none"),
_e=_e.nextElementSibling},"hideRenderedBranchFrom"),N=o(()=>{ke.forEach(({node:K,prevDisplay:_e})=>{
K&&(K.style.display=_e||"")}),ke.length=0},"restoreHiddenBranch");W&&$(W);const U=Date.now(),O=isGptImageModel()||
Z?currentMaskImage:null,q=renderMessage(U,"user",D,JSON.stringify(d),null,null,null,!1,currentQuote,
null,null,null,null,null,null,null,j,me,activeSkill?activeSkill.name:null,null,O);let te=!1;const B=/(https?:\/\/)?(x\.com|twitter\.com)\//i,
V=!i&&(B.test(D||"")||B.test(currentQuote||"")),de="grok-4-fast-reasoning",fe=o(()=>{get("enable-sea\
rch").checked=!0,get("model-select").value!==de&&selectModelById(de)},"applyXLinkAuto");if(V&&!isMistralOcrModel()&&
!get("enable-search").checked)if(autoSearchOnLinks)fe();else{const K=get("auto-search-banner"),_e=get(
"auto-search-on-btn"),Ee=get("auto-search-off-btn"),qe=get("auto-search-remember");K&&_e&&Ee&&(qe&&(qe.
checked=!1),await new Promise(ot=>{K.classList.remove("hidden");const ht=o(Lt=>{K.classList.add("hid\
den"),_e.onclick=null,Ee.onclick=null,ot(Lt)},"cleanup");_e.onclick=()=>ht("enable"),Ee.onclick=()=>ht(
"disable")}).then(async ot=>{ot==="enable"?(fe(),qe&&qe.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):te=!0}))}const Q=String(get("reasoning-effort").value||"").
toLowerCase(),le=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&Q==="none",
ee={client_request_id:createClientRequestId(),thread_id:currentThreadId,message:D,model:get("model-s\
elect").value,image_urls:d,image_items:c,uploaded_image_urls:f,temporary_chat:temporaryChatEnabled,enable_search:get(
"enable-search").checked,enable_url_context:get("enable-url-context")?get("enable-url-context").checked:
!1,enable_maps:get("enable-maps")?get("enable-maps").checked:!1,enable_python:get("enable-python").checked,
enable_mcp:isMcpEnabledForSend(),enable_skills:get("enable-skills")?get("enable-skills").checked:!0,
enable_file_creation:get("enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:le?
!1:get("enable-thinking").checked,thinking_level:get("thinking-level").value,thinking_budget:get("th\
inking-budget")?get("thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:P,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&get("tts-voice-custom")?
get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-language")?get("tts-language").
value:null,tts_style:isTtsModel()&&get("tts-style")&&get("tts-style-wrap")&&!get("tts-style-wrap").classList.
contains("hidden")?get("tts-style").value:null,tts_speed:isTtsModel()&&get("tts-speed")?get("tts-spe\
ed").value:null,image_size:isGptImageModel()&&get("gpt-image-size")?get("gpt-image-size").value:null,
image_quality:isGptImageModel()&&get("gpt-image-quality")?get("gpt-image-quality").value:null,image_complexity:isGptImageModel()&&
get("gpt-image-complexity")?get("gpt-image-complexity").value:null,image_format:isGptImageModel()&&get(
"gpt-image-format")?get("gpt-image-format").value:null,image_compression:isGptImageModel()&&get("gpt\
-image-compression")?get("gpt-image-compression").value:null,image_frame_count:isGptImageModel()&&get(
"gpt-image-frame-count")?get("gpt-image-frame-count").value:null,image_frame_duration_ms:isGptImageModel()&&
get("gpt-image-frame-duration")?get("gpt-image-frame-duration").value:null,image_mask:O,gemini_image_aspect:isGeminiImageModel()&&
get("gemini-image-aspect")?get("gemini-image-aspect").value:null,gemini_image_size:isGeminiImageModel()&&
get("gemini-image-size")?get("gemini-image-size").value:null,grok_image_aspect:isGrokImageModel()&&get(
"grok-image-aspect")?get("grok-image-aspect").value:null,grok_image_resolution:isGrokImageModel()&&get(
"grok-image-resolution")?get("grok-image-resolution").value:null,grok_image_quality:isGrokImageModel()&&
get("grok-image-quality")?get("grok-image-quality").value:null,grok_image_format:isGrokImageModel()&&
get("grok-image-format")?get("grok-image-format").value:null,grok_image_count:isGrokImageModel()&&get(
"grok-image-count")?get("grok-image-count").value:null,ideogram_operation:isIdeogramModel()&&get("id\
eogram-image-operation")?get("ideogram-image-operation").value:null,ideogram_aspect:isIdeogramModel()&&
get("ideogram-image-aspect")?get("ideogram-image-aspect").value:null,ideogram_resolution:isIdeogramModel()&&
get("ideogram-image-resolution")?get("ideogram-image-resolution").value:null,ideogram_quality:isIdeogramModel()&&
get("ideogram-image-quality")?get("ideogram-image-quality").value:null,ideogram_speed:isIdeogramModel()&&
get("ideogram-image-speed")?get("ideogram-image-speed").value:null,ideogram_magic_prompt:isIdeogramModel()&&
get("ideogram-image-magic")?get("ideogram-image-magic").value:null,ideogram_style_type:isIdeogramModel()&&
get("ideogram-image-style")?get("ideogram-image-style").value:null,ideogram_negative_prompt:isIdeogramModel()&&
get("ideogram-image-negative")?get("ideogram-image-negative").value:null,ideogram_count:isIdeogramModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:me,parent_id_explicit:ie,disable_auto_search:te,
image_vision_model:currentVisionModel||null,coding_mode:E,coding_target:E?{id:C.candidate_id,code:C.
prompt_source?null:C.code,language:C.language||"text",key:C.key||null,message_id:C.message_id||null,
source:C.prompt_source?"prompt":"history",explicit:C.explicit===!0}:null,coding_candidates:E?M.map(K=>({
id:K.candidate_id,source:K.prompt_source?"prompt":"history",prompt_index:K.prompt_source?K.prompt_index:
null,code:K.prompt_source?null:K.code,language:K.language||"text",explicit:K.explicit===!0})):[],batch_mode:l,
canvas_mode:!!canvasModeEnabled};n&&(ee.turnstile_token=n);const pe=get("thread-custom-instruction");
pe&&(ee.thread_custom_instruction=pe.value||""),activeSkill?(ee.system_prompt=activeSkill.instruction,
ee.enable_system_prompt=!0,ee.skill_uuid=activeSkill.uuid):ee.skill_uuid=null;const se=takePromptForSend(
e.animate!==!1),Fe=animatePromptToMessage(q,se,j&&e.animate!==!1);setSendBtnToStopMode();const ue="a\
i-"+Date.now(),Se=String(ee.model||"").toLowerCase(),Me=!!ee.enable_thinking||!!Q&&Q!=="none",we=Se.
includes("gemini")||Se.includes("o1")||Se.includes("o3")||Se.includes("gpt-5")||Se.includes("reasoni\
ng")&&!Se.includes("non-reasoning"),De=Me&&we;let wt=buildPendingSkeletonHtml(ee.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending\
-row fade-in"><div id="${ue}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rou\
nded-2xl rounded-tl-none shadow-md relative">${wt}</div></div>`),j&&resumeChatAutoScroll();const he=get(
ue),pt=l?null:startPendingElapsed(he,t);activeStreamingBubbleId=ue,canvasModeEnabled&&resetCanvasPreviewPanel();
let Pe=null;const gt=o(K=>!De||!he?null:((!Pe||!he.contains(Pe))&&(Pe=he.querySelector(".thought-con\
tent")),Pe||(he.insertAdjacentHTML("afterbegin",'<div class="thought-container"><div class="thought-\
header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i>\
 Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>'),Pe=
he.querySelector(".thought-content")),Pe&&(Pe.setAttribute("data-placeholder","1"),Pe.textContent=K||
"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),Pe),"ensureThoughtPlaceholder");De&&
gt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),abortController=new AbortController;
const Je=currentThreadId,At=nowPerfMs(),mt=Date.now();let Ct=!1,on=!1,Ft=!1,It=null,Dt=null,kt=null,
Xt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;const Ht=o((K,_e)=>{if(!_e||
K==="status"&&Ct||K==="thought"&&on||K==="content"&&Ft)return;const Ee=Math.max(0,nowPerfMs()-At);K===
"status"?It=Ee:K==="thought"?Dt=Ee:K==="content"&&(kt=Ee),reportFirstTokenLatency({latency_seconds:Ee/
1e3,latency_ms:Ee,thread_id:Xt||currentThreadId,job_id:currentJobId,model:ee.model,first_event_type:K,
client_sent_at_ms:mt}),K==="status"?Ct=!0:K==="thought"?on=!0:K==="content"&&(Ft=!0)},"maybeReportFi\
rstEventLatency"),xt=window.ProgressSpinner?window.ProgressSpinner.startFlow("chat"):null;let Pt=!1,
Yt=!1,qt=null,_t=null,Ot=!1;try{ee.thread_id&&activeSkill&&(threadSkillMap[ee.thread_id]=activeSkill,
pendingSkillForNewThread=null);const K=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.urls.chatStream,
manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
ee),signal:abortController.signal}),he);if(sendClientDebugLog("info",`Prompt stream response status:\
 ${K.status}`),!K.ok){const Ge=await K.json().catch(()=>({})),Ve=new Error(Ge.error||`HTTP ${K.status}`);
throw Ve.serverCode=Ge.code||null,Ve.serverModel=Ge.model||ee.model,Ve.acceptedJobId=Ge.job_id||null,
Ve.acceptedThreadId=Ge.thread_id||null,Ve}Pt=!0,window.ConnectionMonitor&&(Ot=!0,window.ConnectionMonitor.
operationStarted()),xt&&xt.setPhase("waiting"),se.accept(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});
const _e=o(()=>{if(!he)return;const Ge=he.querySelector(".content-area");if(Ge&&Ge.getAttribute("dat\
a-api-accepted")!=="1"&&(Ge.setAttribute("data-api-accepted","1"),!updatePendingSkeletonStatus(he,"\u63A5\
\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...","\u5FDC\u7B54\u958B\u59CB\
\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){Ge.outerHTML=buildPendingSkeletonHtml(
ee.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const Ve=he.querySelector(".content-area");Ve&&Ve.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
he,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");_e(),(rn=window.ActivityLog)==null||rn.log("send.stream.start");let Ee=!1;const qe=K.
body.getReader(),ot=new TextDecoder;let ht="",Lt="",et="",Tt=!0,Ye=null,at=null,Mt=null,cn=!1;const Nt={};
let dn=0,Zt=!1;for(;!Zt;){const{done:Ge,value:Ve}=await qe.read();if(!Ge&&!Ee&&(Ee=!0,(Qt=window.ActivityLog)==
null||Qt.log("send.stream.first")),Ge&&((St=window.ActivityLog)==null||St.log("send.stream.done")),Ge)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),xt&&xt.setPhase("receiving"),
ht+=ot.decode(Ve,{stream:!0});let tt=ht.split(`
`);ht=tt.pop();let un=!1,pn=!1;for(let Ut of tt)if(Ut.trim())try{const Ae=JSON.parse(Ut);if(Ae.type===
"thread_id"){_e();const Oe=Ae.content!==null&&Ae.content!==void 0?String(Ae.content):Ae.content;Oe&&
(Xt=Oe,currentThreadId!==Oe&&(currentThreadId=Oe,history.pushState({},"","/c/"+Oe)),activeSkill&&(threadSkillMap[Oe]=
activeSkill,pendingSkillForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Ae.type==="\
job_id"){_e(),currentJobId=Ae.content,l&&showToast("Batch\u767B\u9332","info");continue}if(Ae.type===
"search_status"){Mt=updateSearchBox(he,Mt,Ae.content);continue}if(Ae.type==="mcp"){handleMcpStreamEvent(
he,Ae.content||{});continue}if(Ae.type==="mcp_decision_request"){openMcpDecisionModal(Ae.content||{});
continue}if(Ae.type==="status"){_e();const Oe=Ae.content===null||Ae.content===void 0?"":String(Ae.content);
if(Ht("status",!!Oe),Tt&&he){const Ke=Oe||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(
he,Ke,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const dt=he.querySelector(".content-area");dt&&(dt.outerHTML=buildPendingSkeletonHtml(ee.model,Ke),updatePendingSkeletonStatus(
he,Ke,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
De&&gt(Oe||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Tt){beginPendingToStreamTransition(
he);const Oe=he.querySelector(".content-area");Oe&&(Oe.innerHTML=""),Tt=!1}if(Ae.type==="coding_diff")
appendCodingLiveDiff(he,Ae.content||{}),Ht("content",!0);else if(Ae.type==="thought"){if(Ye||(Ye=he.
querySelector(".thought-content")),et+=Ae.content,Ht("thought",!!Ae.content),!Ye){const Oe='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';Mt?Mt.insertAdjacentHTML(
"afterend",Oe):he.insertAdjacentHTML("afterbegin",Oe),Ye=he.querySelector(".thought-content")}if(Ye&&
Ye.getAttribute("data-placeholder")==="1"){if(Ye.textContent="",Ye.removeAttribute("data-placeholder"),
Ye){const Oe=Ye.parentElement.querySelector(".thought-header");Oe&&Oe.classList.remove("thinking-shi\
mmer")}et=Ae.content}Ye.classList.remove("collapsed"),pn=!0}else if(Ae.type==="image_analysis"){const Oe=Ae.
content===null||Ae.content===void 0?"":String(Ae.content);if(!he)continue;let Ke=he.querySelector(".\
image-analysis-box");if(!Ke){const ut='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';Mt?Mt.insertAdjacentHTML("afterend",ut):he.insertAdjacentHTML("afterbegin",ut),Ke=he.querySelector(
".image-analysis-box")}const dt=Ke.querySelector(".image-analysis-text");dt&&(dt.textContent=Oe)}else if(Ae.
type==="python"){const Oe=Ae.content||{},Ke=Oe.id||`py_${Date.now()}`;if(!Nt[Ke]){const ut=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Ke}" data-collapsed="true" data-code-key="${Ke}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;Mt?Mt.insertAdjacentHTML("afterend",ut):he.insertAdjacentHTML(
"afterbegin",ut),Nt[Ke]=he.querySelector(`[data-py-id="${Ke}"]`)}const dt=Nt[Ke];if(dt){if(Oe.code!==
void 0){const ut=Oe.code==null?"":String(Oe.code),Et=dt.querySelector(".python-code");Et&&(Et.textContent=
ut,Et.removeAttribute("data-highlighted"),queueHighlight(dt,ut));const bt=dt.querySelector('.copy-bt\
n[data-copy="code"]');bt&&bt.setAttribute("data-code",encodeURIComponent(ut).replace(/'/g,"%27"))}if(Oe.
output!==void 0){const ut=Oe.output==null?"":String(Oe.output),Et=dt.querySelector(".python-output");
Et&&(Et.textContent=ut);const bt=dt.querySelector('.copy-btn[data-copy="output"]');bt&&bt.setAttribute(
"data-code",encodeURIComponent(ut).replace(/'/g,"%27"))}}}else if(Ae.type==="content"){const Oe=Ae.content===
null||Ae.content===void 0?"":String(Ae.content);Lt+=Oe,/[`~]/.test(Oe)&&activateDeferredCodingModeFromStream(
Lt),at||(at=he.querySelector(".content-area")||document.createElement("div"),at.className="prose pro\
se-invert text-sm break-words",he.contains(at)||he.appendChild(at)),un=!0,Ht("content",!!Oe)}else if(Ae.
type==="error"){cn=!0,Zt=!0,he.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Ae.content)),
showToast(Ae.content||"Unknown error","error",!0);break}}catch{}if(pn&&Ye&&(Ye.textContent=et,userAutoScroll&&
(Ye.scrollTop=Ye.scrollHeight)),un&&at){const Ut=Date.now();if(Ut-dn>100){const Ae=snapshotCodeCollapse(
at);renderAiMarkdownInto(at,Lt,{incrementalMath:!0}),applyCodeCollapse(at,Ae,!0),dn=Ut}}scrollToBottom()}
if(xt&&xt(),at){const Ge=snapshotCodeCollapse(at);renderAiMarkdownInto(at,Lt,{incrementalMath:!0}),applyCodeCollapse(
at,Ge,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),he)if(queueHighlight(he,Lt),enableLatencyMetrics){
const Ge=nowPerfMs()-At;reportFirstTokenLatency({is_total:!0,latency_seconds:Ge/1e3,latency_ms:Ge,thread_id:Xt||
currentThreadId,job_id:currentJobId,model:ee.model,client_sent_at_ms:mt,client_done_at_ms:Date.now()});
let Ve='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',tt=null;It!==null&&(tt=It),Dt!==null&&(tt===null||Dt<tt)&&(tt=
Dt),kt!==null&&(tt===null||kt<tt)&&(tt=kt),tt!==null&&(Ve+=`<div>Initial: ${(tt/1e3).toFixed(2)}s</d\
iv>`),kt!==null&&kt!==tt&&(Ve+=`<div>Content: ${(kt/1e3).toFixed(2)}s</div>`),Ve+=`<div class="font-\
bold text-gray-300">Total: ${(Ge/1e3).toFixed(2)}s</div>`,currentJobId&&(Ve+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),Ve+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,Ve+="</div>",he.insertAdjacentHTML("beforeend",Ve)}else he.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),he&&he.querySelectorAll(".thought-\
content").forEach(Ve=>Ve.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!W}),!cn&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(Ge=>Ge.json()).then(Ge=>{Ge.title&&
(document.title=Ge.title+" - AI Chat",setCurrentChatHeaderTitle(Ge.title),loadThreads())}):loadThreads(
!1)}catch(K){(ln=window.ActivityLog)==null||ln.log(K.name==="AbortError"?"send.cancelled":"send.erro\
r"),Fe();let _e=!1;const Ee=K.name==="AbortError"&&isManualStopAbortForThread(Je);if(K.name==="Abort\
Error"&&!Ee&&(_e=await syncThreadAfterAbortedStream(Je,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${K.message}`),!Pt){q&&q.remove();const qe=he&&he.closest(
".fade-in");qe&&qe.remove(),delete messageStore[U],delete messageMeta[U]}if(K.serverCode==="request_\
already_accepted"&&K.acceptedJobId&&K.acceptedThreadId)Pt=!0,qt={job_id:K.acceptedJobId,thread_id:String(
K.acceptedThreadId),model:ee.model,pendingStartedAt:l?null:t},se.accept();else if(Pt&&!Ee)_t={job_id:normalizeJobIdForUi(
currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):null,model:ee.model,pendingStartedAt:l?
null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\
\u307E\u3059\u3002","warning",!1);else if(K.serverCode==="turnstile_required"){const qe=await getTurnstileToken();
qe?(await verifyTurnstileOnServer(qe,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(K.serverCode==="api_key_missing"){const qe=K.serverModel||ee.model,ot=await showApiKeyRequiredModalAsync(
qe);ot==="set"?Yt=!0:ot==="switch"?showModal("model-modal"):showToast(K.message||`${getModelNameById(
qe)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(K.
name!=="AbortError"){const qe="Connection Error: "+K.message;showToast(qe,"error",!0)}if(!Pt){const qe=W&&
se.canRetry()&&!editingMessageId;se.restore(),qe&&(editingMessageId=W,currentParentId=me,promptSettingsBeforeEdit=
re,setEditUi(!0))}W&&!_e&&N()}finally{pt&&pt(),Ot&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),xt&&xt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===ue&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,setEditUi(!!editingMessageId)}if(qt){const K=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=qt.thread_id,(K!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
qt,currentThreadId)}if(_t&&_t.thread_id)return reconnectPendingStreamUntilAvailable(_t,_t.thread_id);
if(Yt&&se.canRetry())return sendMessageLogged({animate:!1})}o(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var re,me,ie,ke;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const $=a.
querySelector(".content-area");$?$.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const r=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const l=currentThreadId,
c=i.toLowerCase(),d=c.includes("gemini")||c.includes("o1")||c.includes("o3")||c.includes("gpt-5")||c.
includes("reasoning")&&!c.includes("non-reasoning");let f=null;const h=o($=>!d||!a?null:((!f||!a.contains(
f))&&(f=a.querySelector(".thought-content")),f||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),f=a.querySelector(".thought-content")),f&&(f.setAttribute("data-pla\
ceholder","1"),f.textContent=$||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),f),
"ensureThoughtPlaceholder");d&&h("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",v="",w="",k=!0,x=null,L=null,C=null,M=!1;const E={};let D=0,P=!1;const Z=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let j=!1,W=!1;try{const $=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!$.ok)throw new Error(`Resume failed (${$.status})`);window.ConnectionMonitor&&
(W=!0,window.ConnectionMonitor.operationStarted()),Z&&Z.setPhase("waiting"),(re=window.ActivityLog)==
null||re.log("send.stream.start");let N=!1;const U=$.body.getReader(),O=new TextDecoder;for(;!P;){const{
done:q,value:te}=await U.read();if(!q&&!N&&(N=!0,(me=window.ActivityLog)==null||me.log("send.stream.\
first")),q&&((ie=window.ActivityLog)==null||ie.log("send.stream.done")),q)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),Z&&Z.setPhase("receiving"),y+=O.decode(te,{stream:!0});let B=y.
split(`
`);y=B.pop();let V=!1,de=!1;for(let fe of B)if(fe.trim())try{const Q=JSON.parse(fe);if(Q.type==="job\
_id"){currentJobId=Q.content||t;continue}if(Q.type==="search_status"){C=updateSearchBox(a,C,Q.content);
continue}if(Q.type==="mcp"){handleMcpStreamEvent(a,Q.content||{});continue}if(Q.type==="mcp_decision\
_request"){openMcpDecisionModal(Q.content||{});continue}if(Q.type==="status"){const le=Q.content===null||
Q.content===void 0?"":String(Q.content);if(k&&a){const ee=le||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";
if(!updatePendingSkeletonStatus(a,ee,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const pe=a.querySelector(".content-area");pe&&(pe.outerHTML=buildPendingSkeletonHtml(i,ee),updatePendingSkeletonStatus(
a,ee,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
d&&h(le||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(k){beginPendingToStreamTransition(
a);const le=a.querySelector(".content-area");le&&(le.innerHTML=""),k=!1}if(Q.type==="coding_diff")appendCodingLiveDiff(
a,Q.content||{});else if(Q.type==="thought"){if(x||(x=a.querySelector(".thought-content")),w+=Q.content,
!x){const le='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content">\
</div></div>';C?C.insertAdjacentHTML("afterend",le):a.insertAdjacentHTML("afterbegin",le),x=a.querySelector(
".thought-content")}if(x&&x.getAttribute("data-placeholder")==="1"){if(x.textContent="",x.removeAttribute(
"data-placeholder"),x){const le=x.parentElement.querySelector(".thought-header");le&&le.classList.remove(
"thinking-shimmer")}w=Q.content}x.classList.remove("collapsed"),de=!0}else if(Q.type==="image_analys\
is"){const le=Q.content===null||Q.content===void 0?"":String(Q.content);if(!a)continue;let ee=a.querySelector(
".image-analysis-box");if(!ee){const se='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 bord\
er border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas\
 fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"><\
/div></div>';C?C.insertAdjacentHTML("afterend",se):a.insertAdjacentHTML("afterbegin",se),ee=a.querySelector(
".image-analysis-box")}const pe=ee.querySelector(".image-analysis-text");pe&&(pe.textContent=le)}else if(Q.
type==="python"){const le=Q.content||{},ee=le.id||`py_${Date.now()}`;if(!E[ee]){const se=`<div class\
="code-wrapper python-box collapsed" data-py-id="${ee}" data-collapsed="true" data-code-key="${ee}">\
<div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</s\
pan><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-labe\
l="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-co\
de="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-b\
tn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left">\
</i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label\
">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pytho\
n-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-ou\
tput"></code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",se):a.insertAdjacentHTML("a\
fterbegin",se),E[ee]=a.querySelector(`[data-py-id="${ee}"]`)}const pe=E[ee];if(pe){if(le.code!==void 0){
const se=le.code==null?"":String(le.code),Fe=pe.querySelector(".python-code");Fe&&(Fe.textContent=se,
Fe.removeAttribute("data-highlighted"),queueHighlight(pe,se));const ue=pe.querySelector('.copy-btn[d\
ata-copy="code"]');ue&&ue.setAttribute("data-code",encodeURIComponent(se).replace(/'/g,"%27"))}if(le.
output!==void 0){const se=le.output==null?"":String(le.output),Fe=pe.querySelector(".python-output");
Fe&&(Fe.textContent=se);const ue=pe.querySelector('.copy-btn[data-copy="output"]');ue&&ue.setAttribute(
"data-code",encodeURIComponent(se).replace(/'/g,"%27"))}}}else if(Q.type==="content"){const le=Q.content===
null||Q.content===void 0?"":String(Q.content);v+=le,/[`~]/.test(le)&&activateDeferredCodingModeFromStream(
v),L||(L=a.querySelector(".content-area")||document.createElement("div"),L.className="prose prose-in\
vert text-sm break-words",a.contains(L)||a.appendChild(L)),V=!0}else if(Q.type==="error"){M=!0,P=!0,
a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Q.content)),showToast(Q.content||"Unknown \
error","error",!0);break}}catch{}if(de&&x&&(x.textContent=w,userAutoScroll&&(x.scrollTop=x.scrollHeight)),
V&&L){const fe=Date.now();if(fe-D>100){const Q=snapshotCodeCollapse(L);renderAiMarkdownInto(L,v,{incrementalMath:!0}),
applyCodeCollapse(L,Q,!0),D=fe}}scrollToBottom()}if(Z&&Z(),L){const q=snapshotCodeCollapse(L);renderAiMarkdownInto(
L,v,{incrementalMath:!0}),applyCodeCollapse(L,q,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,
v),a&&a.querySelectorAll(".thought-content").forEach(te=>te.classList.add("collapsed")),await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0}),loadThreads(!1)}catch($){(ke=window.ActivityLog)==null||
ke.log($.name==="AbortError"?"send.cancelled":"send.stream.error");const N=$.name==="AbortError"&&isManualStopAbortForThread(
l);$.name==="AbortError"&&!N&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
N||(j=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{r&&r(),W&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
Z&&Z(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,currentThreadPending=null}if(j)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i,pendingStartedAt:e.pendingStartedAt},l)}o(resumePendingStream,"resumePendingStream");
function updateThreadHighlighting(){const e=get("thread-list");if(!e)return;e.querySelectorAll("[dat\
a-thread-id]").forEach(n=>{n.dataset.threadId===String(currentThreadId)?n.classList.add("bg-gray-700\
/60","border-l-2","border-blue-500"):n.classList.remove("bg-gray-700/60","border-l-2","border-blue-5\
00")})}o(updateThreadHighlighting,"updateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){
snapshotSidebarHistory("loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory(
"loadThreads-start append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?
t.value:"";if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");
return}const a=await(await apiFetch(`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&pa\
ge=${threadPage}`)).json(),r=get("thread-list");if(!r)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory(
"loadThreads-skip-replace-settings-open");return}const c=a&&Array.isArray(a.threads)?a.threads.length:
-1,d=r.querySelectorAll("[data-thread-id]").length;if(c===0&&d>0&&String(n||"").trim()){snapshotSidebarHistory(
"loadThreads-keep-existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" cla\
ss="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="\
fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scro\
ll-sentinel"></div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const d=String(
c.id),f=document.createElement("div"),h=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=d===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,f.dataset.threadId=d,f.dataset.threadTitle=c.title||"",f.innerHTML=`<div class="flex \
items-center gap-1 truncate flex-1"><button class="${h} hover:text-yellow-400 px-1" onclick="toggleB\
ookmark(event, '${d}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(
c.title||"No Title")}</span>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 m\
d:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:tex\
t-white px-1 transition" onclick="renameThread(event, '${d}')"><i class="fas fa-pen text-xs"></i></b\
utton><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, \
'${d}')"><i class="fas fa-trash text-xs"></i></button></div>`,f.onclick=k=>{k.target.closest("button")||
k.target.closest("[data-thread-actions]")||loadMessages(d)},l?r.insertBefore(f,l):r.appendChild(f)}),
hasMoreThreads=!!a.has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendere\
d count="+a.threads.length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){
console.error("Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=
!1,updateThreadHighlighting(),snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThre\
ads");function initPullToRefresh(e,t){const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,
r=88,l=52,c=.5,d=8;let f=0,h=!1,y=0,v=null;const w=o(()=>get(i),"indicatorEl"),k=o(()=>{const C=w();
return C?C.querySelector(".ptr-pull-label"):null},"labelEl"),x=o(C=>{const M=w();if(!M)return;M.style.
height=Math.min(C,r)+"px",M.classList.toggle("active",C>2),M.classList.toggle("pull-ready",C>=a);const E=k();
E&&(E.textContent=C>=a?"\u96E2\u3057\u3066\u66F4\u65B0":"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},
"applyPullUI"),L=o(()=>{const C=w();C&&(C.style.height="0px",C.classList.remove("active","pull-ready",
"refreshing"),C.classList.remove("dragging"))},"resetPullUI");n.addEventListener("touchstart",C=>{if(v){
h=!1;return}if(n.scrollTop>0){h=!1;return}const M=C.touches[0];M&&(f=M.clientY,y=0,h=!0)},{passive:!0}),
n.addEventListener("touchmove",C=>{if(!h||v)return;if(n.scrollTop>0){h=!1;return}const M=C.touches[0];
if(!M)return;const E=M.clientY-f;if(E<=0){y>0&&(y=0,x(0)),h=!1;return}const D=w();D&&!D.classList.contains(
"dragging")&&D.classList.add("dragging"),y=Math.min(E*c,r),x(y),E>=d&&C.preventDefault()},{passive:!1}),
n.addEventListener("touchend",()=>{if(!h||(h=!1,v))return;const C=w();C&&C.classList.remove("draggin\
g");const M=y>=a;if(y=0,!M){L();return}let E;try{E=t()}catch{E=null}const D=w();if(D){D.classList.add(
"refreshing"),D.style.height=l+"px";const P=D.querySelector(".ptr-pull-label");P&&(P.textContent="\u66F4\u65B0\
\u4E2D...")}E&&typeof E.then=="function"?(v=E,E.catch(()=>{}).finally(()=>{v=null,L()})):(v=Promise.
resolve(),setTimeout(()=>{v=null,L()},400))}),n.addEventListener("touchcancel",()=>{h=!1,y=0,L()})}o(
initPullToRefresh,"initPullToRefresh");const initThreadPullToRefresh=o(()=>initPullToRefresh("thread\
-list",()=>loadThreads(!1)),"initThreadPullToRefresh"),initSkillPullToRefresh=o(()=>initPullToRefresh(
"skill-list",()=>loadSkills()),"initSkillPullToRefresh"),initPullToRefreshAll=o(()=>{initThreadPullToRefresh(),
initSkillPullToRefresh()},"initPullToRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;
const mcpCardIdSelector=o(e=>"mcp_card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),
mcpEscHtml=o(e=>String(e==null?"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"\
&quot;","'":"&#39;"})[t]),"mcpEscHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||
"MCP")} / ${mcpEscHtml(e.tool_name||e.internal_name||"")}`}o(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){
if(!e)return null;let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("\
div"),t.className="mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),
e.appendChild(t)),t}o(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){
if(!e||!t||!t.type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(
e):null;if(n&&!i)return;const a=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.
querySelector('[data-mcp-card="'+a+'"]'))return;const r=`<div class="mcp-box mcp-running mb-2" data-\
mcp-card="${a}">
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
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const d=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",d),r=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
r&&r.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const r=get("mcp-decision-modal");if(r&&!r.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}o(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const d=JSON.parse(c);c=JSON.stringify(d,null,2)}catch{}a.textContent=c}const r=get(
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
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/skillini/batch/status");if(!e.ok)return;
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
const n=++threadLoadSequence,i=t.targetMessageId||new URLSearchParams(location.search).get("message");
window.closeHistoryModal&&window.closeHistoryModal();const a=!!t.preserveDraft,r=!!t.silent;r||resumeChatAutoScroll(
{scroll:!1});const l=r?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",d="",f=[];
if(a){const y=get("prompt-input");c=y?y.value:"",d=y?y.style.height:"",f=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,promptSettingsBeforeEdit=null,setEditUi(!1)}else cancelEdit();r||playChatTransition(
"history");const h=e!=null?String(e):e;h!==currentThreadId&&(modelChosenManually=!1),currentThreadId=
h,t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveSkillForThread(
currentThreadId),get("welcome-screen").classList.add("hidden"),r||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const y=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);y.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const v=i;v&&/^\d+$/.test(String(v))&&y.
searchParams.set("before_id",String(Number(v)+1));const w=await apiFetch(y.toString());if(!w.ok)throw new Error(
`thread request failed (${w.status})`);const k=await w.json();if(!k||!Array.isArray(k.messages))throw new Error(
"invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(k&&k.title),
allMessages=k.messages,renderBranchOrigin(k.branch_origin),threadHasOlderMessages=!!k.has_older_messages,
oldestLoadedMessageId=k.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const x=(allMessages||
[]).filter(M=>M.role==="user"&&M.content).map(M=>M.content);if(promptHistory=[...new Set(x.slice().reverse())],
historyIndex=-1,tempPrompt="",currentThreadPending=k.pending_job||null,setTemporaryChatUiState(!!(k&&
k.is_temporary)),applyTemporaryChatRuntimeMeta(k||{}),ensureTemporaryChatHeartbeat(!0),get("thread-c\
ustom-instruction")&&(get("thread-custom-instruction").value=k.custom_instruction||""),get("enable-p\
rompt-cache")&&(get("enable-prompt-cache").checked=!!k.enable_prompt_caching,updatePromptCacheUi()),
k.last_model&&!modelChosenManually&&selectModelById(k.last_model),k.last_skill_uuid&&loadedSkills.length>
0){const M=loadedSkills.find(E=>E.uuid===k.last_skill_uuid);M&&(threadSkillMap[currentThreadId]=M,applyActiveSkill(
M))}const L=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),C=i;if(C&&
allMessages.some(M=>String(M.id)===String(C))?currentLeafId=Number(C):L&&allMessages.find(M=>String(
M.id)===String(L))?currentLeafId=L:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(r?{silent:r,keepScroll:r}:{silent:r,keepScroll:r,animate:!0}),
a||restoreLatestPromptOptions(),r&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):r||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!r&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const M=get("prompt-input");M&&(M.value=c||"",
d?M.style.height=d:M.style.height="auto"),currentImageUrls=f,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i&&new URLSearchParams(
location.search).has("message")&&history.replaceState({},"",location.pathname),a||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(y){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",y),r||showChatLoadError(e),r||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}o(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const r=await(await apiFetch(i.toString())).json(),l=Array.isArray(r.messages)?r.messages:[];
if(l.length){const c=new Set(allMessages.map(f=>f.id)),d=l.filter(f=>!c.has(f.id));d.length&&(allMessages=
d.concat(allMessages))}if(threadHasOlderMessages=!!r.has_older_messages,oldestLoadedMessageId=r.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;const r=i?Array.from(a.children):[],l=r.map(
x=>x.getBoundingClientRect().height);let c=null;if(i&&(c=a.scrollTop),allMessages.length===0){a.replaceChildren(),
currentParentId=null,updateTotalTokenBar(0);return}const d={};allMessages.forEach(x=>{d[x.id]=x,x.childrenIds=
[]}),allMessages.forEach(x=>{x.parent_id&&d[x.parent_id]&&d[x.parent_id].childrenIds.push(x.id)}),(!currentLeafId||
!d[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null);const f=[];
let h=d[currentLeafId];for(;h;)f.unshift(h),h=d[h.parent_id];const y=buildTokenTotals(f),v=document.
createDocumentFragment();if(threadHasOlderMessages){const x=loadingOlderMessages?"\u8AAD\u307F\u8FBC\u307F\u4E2D...":
"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",L=loadingOlderMessages?"d\
isabled":"",C=document.createElement("div");C.className="mb-3 text-center",C.innerHTML=`<button id="\
load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-gray-600 text-gray-200 hov\
er:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="loadOlderMessages()" ${L}><\
i class="fas fa-clock-rotate-left mr-1"></i>${x}</button>`,v.appendChild(C)}f.forEach(x=>{const L=x.
parent_id?d[x.parent_id]:null,C=L?L.childrenIds:allMessages.filter(E=>!E.parent_id).map(E=>E.id),M=C.
length>1?{current:C.indexOf(x.id)+1,total:C.length,siblings:C}:null;renderMessage(x.id,x.role,x.content,
x.image_url,x.thought_data,x.model,M,n,x.quote_text,x.tokens,x.tokens_in,x.tokens_out,x.is_encrypted,
x.tokens_content,x.tokens_thought,v,!1,x.parent_id,x.skill_name,x.batch_job,x.prompt_options&&typeof x.
prompt_options=="object"?x.prompt_options.image_mask:null)});const w=currentThreadPending;if(w&&!isPendingJobSuppressed(
w.job_id)){const x=w.message_id,L=new Set(f.map(E=>E.id)),C=f.length?f[f.length-1]:null;if(x&&L.has(
x)&&currentLeafId===x||!x&&C&&C.role==="user"){const E=w.job_id?`pending-${w.job_id}`:null;renderPendingMessage(
v,n,!1,E,w.model||null)}}const k=[];if(i&&!lowBandwidthMode&&Array.from(v.children).forEach((x,L)=>{
const C=r.find(E=>E.id&&E.id===x.id)||(r.length===v.children.length?r[L]:null),M=C?l[r.indexOf(C)]:0;
M>0&&(x.style.minHeight=`${M}px`,k.push(x))}),a.replaceChildren(v),displayMissingMessageFiles(a,f),updateTotalTokenBar(
y.tokens_total,y),currentParentId=currentLeafId,i&&c!==null?restoreThreadTreeScroll(a,c):scrollToBottom(),
lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(a),f.length){
const x=queueMathTypeset(a,f.map(L=>L.content||"").join(`
`));Promise.resolve(x).finally(()=>{if(!k.length||!k[0].isConnected)return;const L=a.scrollTop;k.forEach(
C=>{C.style.minHeight=""}),restoreThreadTreeScroll(a,L)})}}o(renderThreadTree,"renderThreadTree");function restoreThreadTreeScroll(e,t){
if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&!chatManualPauseIntent?e.scrollTop=
e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=e.scrollTop,syncScrollToBottomButton()}
o(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){currentLeafId=e;const t={};
allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(i=>{i.parent_id&&t[i.parent_id]&&
t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null,renderThreadTree({animate:!0}),restoreLatestPromptOptions();return}for(;t[n]&&t[n].
childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree({animate:!0}),
restoreLatestPromptOptions()}o(switchVersion,"switchVersion");async function loadSkills(){try{const t=await(await apiFetch(
CHAT_CONFIG.urls.handleSkills)).json();loadedSkills=t;const n=get("skill-list");if(!n)return;n.innerHTML=
'<div id="skill-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow\
-down ptr-pull-icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pul\
l-label"></span></div>',Array.isArray(t)&&t.forEach(i=>{const a=document.createElement("div");a.className=
"skill-item p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between \
items-center group",a.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas \
fa-wand-magic-sparkles text-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><\
div class="flex items-center gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:\
opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="openEditSkillModal(event,'${i.uuid}')\
"><i class="fas fa-pencil-alt text-[10px]"></i></button><button class="text-gray-400 hover:text-red-\
400 opacity-100 md:opacity-0 md:group-hover:opacity-100 px-2 transition" onclick="deleteSkill(event,\
'${i.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=r=>{r.target.closest(
"button")||activateSkill(i)},n.appendChild(a)})}catch(e){console.error("Failed to load skills:",e)}}
o(loadSkills,"loadSkills");async function openEditSkillModal(e,t){e.stopPropagation(),editingSkillUuid=
t;try{const i=await(await apiFetch(`/api/skills/${t}`)).json();get("skill-name").value=i.name,get("s\
kill-desc").value=i.description||"",get("skill-inst").value=i.instruction,get("skill-modal-title").textContent=
"Edit Skill",get("save-skill-btn").innerText="Save Changes",showModal("skill-modal"),location.pathname!==
"/skill"&&history.pushState({modal:"skill"},"","/skill")}catch{showToast("Skill\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}o(openEditSkillModal,"openEditSkillModal");async function createSkill(e,t){await apiFetch(
CHAT_CONFIG.urls.handleSkills,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{name:e,instruction:t})}),loadSkills()}o(createSkill,"createSkill");function applyActiveSkill(e){activeSkill=
e||null,activeSkill?(get("active-skill-name").innerText=activeSkill.name,get("skill-active-indicator").
classList.remove("hidden")):get("skill-active-indicator").classList.add("hidden");const t=get("sys-p\
rompt-option");t&&(t.style.opacity="1")}o(applyActiveSkill,"applyActiveSkill");function syncActiveSkillForThread(e){
const t=e&&threadSkillMap[e]?threadSkillMap[e]:null;applyActiveSkill(t)}o(syncActiveSkillForThread,"\
syncActiveSkillForThread");async function saveThreadSkillUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_skill_uuid:t,
thread_id:e})})}catch{}}o(saveThreadSkillUuid,"saveThreadSkillUuid");function activateSkill(e,t){currentThreadId?
(threadSkillMap[currentThreadId]=e,applyActiveSkill(e),showToast(`Skill "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadSkillUuid(currentThreadId,e?e.uuid:null)):(pendingSkillForNewThread=e,applyActiveSkill(
e),allMessages&&allMessages.length>0&&startNewChat({preserveSkill:!0}))}o(activateSkill,"activateSki\
ll");function clearActiveSkill(){currentThreadId&&(delete threadSkillMap[currentThreadId],saveThreadSkillUuid(
currentThreadId,null)),pendingSkillForNewThread=null,applyActiveSkill(null)}o(clearActiveSkill,"clea\
rActiveSkill");function getCurrentChatHeaderTitleText(){return typeof currentThreadTitle=="string"&&
currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?"No Title":"AI Chat"}o(getCurrentChatHeaderTitleText,
"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){return temporaryChatEnabled?
`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:""}o(getTemporaryChatTimeoutLabel,
"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){const e=getCurrentChatHeaderTitleText(),
t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=["sidebar-chat-title","mobile-chat-title"],
a=["sidebar-chat-temporary-label","mobile-chat-temporary-label"],r=["sidebar-chat-ttl","mobile-chat-\
ttl"];i.forEach(l=>{const c=get(l);c&&(c.textContent=e)}),a.forEach(l=>{const c=get(l);c&&c.classList.
toggle("hidden",!n)}),r.forEach(l=>{const c=get(l);c&&(n&&t?(c.textContent=t,c.classList.remove("hid\
den")):(c.textContent="",c.classList.add("hidden")))})}o(updateCurrentChatHeaderUi,"updateCurrentCha\
tHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="string"?e:null,updateCurrentChatHeaderUi()}
o(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){tempChatExpiresAtMs=
null,updateCurrentChatHeaderUi()}o(resetTemporaryChatExpiresAt,"resetTemporaryChatExpiresAt");function applyTemporaryChatRuntimeMeta(e){
if(!e||typeof e!="object")return;Object.prototype.hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(
e.timeout_seconds);let t=null;const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.
floor(n*1e3);else{const i=Number(e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+
Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=
null),updateCurrentChatHeaderUi()}o(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
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
modelChosenManually=!1,restoreLatestPromptOptions(),resetUploadState(),stopTemporaryChatHeartbeat(),
setTemporaryChatUiState(!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,renderBranchOrigin(
null),allMessages=[],promptHistory=[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=
null,loadingOlderMessages=!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(
0),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
e.skipHistory||history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").
classList.remove("hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("threa\
d-custom-instruction").value=""),get("enable-skills")&&(get("enable-skills").checked=!0),get("enable\
-prompt-cache")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),e.preserveSkill?activeSkill&&
applyActiveSkill(activeSkill):applyActiveSkill(null),loadThreads();const t=threadLoadSequence;fetchSettingsSnapshot().
then(n=>{if(threadLoadSequence!==t||currentThreadId||modelChosenManually||e.preserveSkill&&activeSkill)
return;const i=n.use_last_chat_settings&&n.last_model||n.default_model;i&&MODELS.some(a=>a.items.some(
r=>r.id===i))&&selectModelById(i)}).catch(n=>console.warn("Failed to load chat defaults",n)),window.
innerWidth<768&&get("overlay").click()}o(startNewChat,"startNewChat");let threadModalLoadSeq=0;window.
openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
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
!0,c=get("thread-global-sys-prompt"),d=get("thread-global-sys-prompt-enabled");let f=null;try{f=c||d?
{system_prompt:c?c.value:"",system_prompt_enabled:d?d.checked:!0,apply_global_system_prompt:get("thr\
ead-apply-global-sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(h){sendClientDebugLog("error","Payload c\
onstruction failed: "+h.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const h=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+h.status);let y=!0;if(f){sendClientDebugLog(
"info","Starting POST request for user settings");const v=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});y=v.ok,v.ok&&window.
applySavedUserSystemPromptSettings(f),sendClientDebugLog("info","POST request finished, status: "+v.
status)}h.ok&&y?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\u308C\u307E\u3057\u305F","s\
uccess")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}catch(h){sendClientDebugLog(
"error","Save failed with error: "+h.message),showToast("\u30A8\u30E9\u30FC: "+h.message,"error",!0)}finally{
t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=()=>{syncCompressionSettingsUi(),
showModal("compression-modal"),location.pathname!=="/compression"&&history.pushState({modal:"compres\
sion"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal("compression-modal"),!e&&
location.pathname==="/compression"&&history.back()},get("save-compression-settings-btn").onclick=()=>{
const e=get("compression-max-size").value,t=get("compression-max-dim").value,n=get("compression-outp\
ut-type").value,i=get("compression-format-only").checked;setCompressionSettings(e,t,n,i);const a=o((l,c)=>{
get(l)&&get(c)&&(get(c).value=get(l).value)},"syncBack");a("modal-gpt-image-size","gpt-image-size"),
a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-complexity","gpt-image-complexit\
y"),a("modal-gpt-image-format","gpt-image-format"),a("modal-gpt-image-compression","gpt-image-compre\
ssion"),a("modal-gpt-image-frame-count","gpt-image-frame-count"),a("modal-gpt-image-frame-duration",
"gpt-image-frame-duration"),a("modal-gemini-image-aspect","gemini-image-aspect"),a("modal-gemini-ima\
ge-size","gemini-image-size"),a("modal-grok-image-aspect","grok-image-aspect"),a("modal-grok-image-r\
esolution","grok-image-resolution"),a("modal-grok-image-quality","grok-image-quality"),IDEOGRAM_IMAGE_FIELDS.
forEach(l=>a(`modal-ideogram-image-${l}`,`ideogram-image-${l}`)),a("modal-ocr-table-format","ocr-tab\
le-format"),a("modal-ocr-pages","ocr-pages");const r=o((l,c)=>{get(l)&&get(c)&&(get(c).checked=get(l).
checked)},"syncBackChk");r("modal-ocr-extract-header","ocr-extract-header"),r("modal-ocr-extract-foo\
ter","ocr-extract-footer"),r("modal-ocr-include-blocks","ocr-include-blocks"),r("modal-ocr-include-i\
mages","ocr-include-images"),window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success")};async function deleteSkill(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(
CHAT_CONFIG.urls.handleSkillItem.replace("0",t),{method:"DELETE"}),loadSkills())}o(deleteSkill,"dele\
teSkill");function renameThread(e,t){var y;e.stopPropagation();const n=get("thread-rename-modal"),i=get(
"thread-rename-form"),a=get("thread-rename-input"),r=get("thread-rename-save"),l=get("thread-rename-\
cancel"),c=get("thread-rename-error"),d=e.currentTarget;a.value=((y=d.closest("[data-thread-id]"))==
null?void 0:y.dataset.threadTitle)||"",c.textContent="",r.disabled=!1;let f=!1;const h=o(()=>{f||(hideModal(
"thread-rename-modal"),d.focus())},"close");l.onclick=h,n.onclick=v=>{v.target===n&&h()},n.onkeydown=
v=>{v.key==="Escape"&&(v.preventDefault(),v.stopPropagation(),h()),v.key==="Tab"&&(v.shiftKey&&document.
activeElement===a?(v.preventDefault(),l.focus()):!v.shiftKey&&document.activeElement===l&&(v.preventDefault(),
a.focus()))},i.onsubmit=async v=>{v.preventDefault();const w=a.value.trim();if(f||!w){w||a.focus();return}
f=!0,r.disabled=!0,l.disabled=!0;try{const k=await apiFetch(CHAT_CONFIG.urls.updateTitle.replace("0",
t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:w})}),x=await k.
json();if(!k.ok)throw new Error("rename failed");currentThreadId===String(t)&&setCurrentChatHeaderTitle(
x.title||w),f=!1,h(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
f=!1,r.disabled=!1,l.disabled=!1}},l.disabled=!1,showModal("thread-rename-modal"),a.focus()}o(renameThread,
"renameThread");async function deleteThread(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(
CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),currentThreadId===t?startNewChat():
loadThreads())}o(deleteThread,"deleteThread");async function deleteMessage(e,t){!t&&!confirm("Delete\
 this message and subsequent history?")||(await apiFetch(CHAT_CONFIG.urls.deleteMessage.replace("0",
e),{method:"DELETE"}),loadMessages(currentThreadId))}o(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
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
n.url:buildPdfAttachmentUrl(i),d=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:r,isImage:l,url:c,previewUrl:d}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(d=>maybeNeedsMathJax(d.content)||maybeNeedsMathJax(d.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",r=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(d=>{const f=d.
role==="user",h=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",y=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",v=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(d.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(d.content||"")}</div>`,w=buildPdfMessageAttachments(d),k=w.length?`<div \
class="attachments">${w.map(C=>C.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(C.previewUrl)}\
" alt="${pdfEscapeAttr(C.filename)}"><div class="file-caption">${pdfEscapeAttr(C.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(C.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
C.filename)}</span><span class="file-source">${pdfEscapeAttr(C.source)}</span></span></a></div>`).join(
"")}</div>`:"",x=[];d.model&&!f&&x.push(d.model),d.tokens!==null&&d.tokens!==void 0&&x.push(`tokens:${d.
tokens}`),d.tokens_in!==null&&d.tokens_in!==void 0&&x.push(`in:${d.tokens_in}`),d.tokens_out!==null&&
d.tokens_out!==void 0&&x.push(`out:${d.tokens_out}`),d.tokens_thought!==null&&d.tokens_thought!==void 0&&
x.push(`thought:${d.tokens_thought}`),d.is_encrypted&&x.push("encrypted"),d.parent_id!==null&&d.parent_id!==
void 0&&x.push(`parent:#${d.parent_id}`);const L=x.length?`<div class="message-meta">${pdfEscapeAttr(
x.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(d.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${h}
                            ${v}
                            ${y}
                            ${k}
                            ${L}
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
        ${l.map(d=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(d.label)}</div><d\
iv class="meta-value">${pdfEscapeAttr(d.value)}</div></div>`).join("")}
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
async()=>{try{const h=r.contentDocument,y=r.contentWindow;if(!h||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(C=>maybeNeedsMathJax(C.content)||maybeNeedsMathJax(C.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),h.fonts&&h.fonts.ready)try{await h.fonts.ready}catch{}t.update(60);const k=Array.from(h.
images||[]),x=Promise.all(k.map(C=>C.complete?Promise.resolve():new Promise(M=>{C.addEventListener("\
load",M,{once:!0}),C.addEventListener("error",M,{once:!0})})));if(await Promise.race([x,new Promise(
C=>setTimeout(C,5e3))]),t.update(80),h.getElementById("MathJax-script")){let C=0;for(;C<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(M=>setTimeout(M,50)),C++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(M){console.error(
"PDF MathJax typeset failed",M)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(a),f=new Blob([d],{type:"text/html"});r.src=URL.createObjectURL(
f),document.body.appendChild(r)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=o((d,f)=>i.
compare(d.filename||"",f.filename||""),"nameAsc"),r=o((d,f)=>i.compare(f.filename||"",d.filename||""),
"nameDesc"),l=o((d,f)=>(Number(f.ts)||0)-(Number(d.ts)||0),"tsDesc"),c=o((d,f)=>(Number(d.ts)||0)-(Number(
f.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,f)=>a(d,f)||l(d,f)):t==="name_desc"?n.sort((d,f)=>r(
d,f)||l(d,f)):t==="oldest"?n.sort((d,f)=>c(d,f)||a(d,f)):n.sort((d,f)=>l(d,f)||a(d,f)),n}o(sortLibraryFiles,
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
if(n)return;const f=lib.favoritesOnly&&!a?"fa-star":"fa-search",h=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${f}"></i></div>\
<p class="lib-empty-title">${h}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(a)):r).forEach(f=>{
try{const h=renderLibraryItem(f,c++);t.appendChild(h)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(d=>fileNameForSearch(d).includes(n)):t).filter(d=>d.type==="image"),r=lib.favoritesOnly?
a.filter(d=>d.is_favorite):a;if(!r.length)return;const l=r.map(d=>({url:d.url,filename:d.filename||d.
original_filename||d.url.split("/").pop(),element:null}));let c=l.findIndex(d=>d.url===e.url);c===-1&&
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
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",d=e.is_favorite?"fas fa-star":"far f\
a-star",f=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
h=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${f}\
" aria-label="${f}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${h}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",x=>{x.shiftKey&&x.preventDefault()}),n.onclick=
x=>{if(x&&x.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const L=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),C=L.findIndex(E=>E.dataset.filepath===lib.anchorPath),
M=L.indexOf(n);if(C!==-1&&M!==-1){const E=Math.min(C,M),D=Math.max(C,M);for(let P=E;P<=D;P++){const Z=L[P].
dataset.filepath;Z&&(lib.selected.add(Z),L[P].classList.add("is-selected"))}try{const P=window.getSelection&&
window.getSelection();P&&P.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(x=>{x.onclick=L=>{L.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async x=>{x.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const k=n.querySelector(".lib-favorite-btn");return k&&
(k.onclick=async x=>{x.stopPropagation(),k.disabled=!0;try{const L=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
C=await L.json().catch(()=>({}));if(!L.ok||typeof C.is_favorite!="boolean")throw new Error(C.error||
"favorite update failed");e.is_favorite=C.is_favorite,renderLibraryGrid(),showToast(C.is_favorite?"\u304A\
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
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(d=>{d.getAttribute("data-filename")===
e&&setRowAttachmentName(d,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
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
return}if(i.innerHTML="",c.forEach(d=>{const f=document.createElement("div");f.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const h=d.updated_at?new Date(d.
updated_at).toLocaleString():"";f.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(h)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=f.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(f)}),l.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const d=getLibSortOrder(),f=getLibSearchQuery(),h=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(h),sort:d,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);r=await v.json(),l=!0}catch(d){i=d}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const d=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,h=new Set(c.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(c.length>=LIBRARY_PAGE_SIZE||!v||h.has(v))return;const w=getAttachmentNameForPath(
v)||v.split("/").pop()||v,k=(w.split(".").pop()||"").toLowerCase(),x=["png","jpg","jpeg","webp","gif"].
includes(k)?"image":"file",L=x==="image"?f+v:null;c.unshift({filename:w,original_filename:w,filepath:v,
url:d+v,thumbnail_url:L,type:x,ext:k,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),h.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=c.filter(h=>h&&h.filepath&&
h.url);let f=[];if(e){const h=new Set(lib.files.map(y=>y.filepath));f=d.filter(y=>!h.has(y.filepath)),
lib.files.push(...f)}else lib.files=d;lib.files.forEach(h=>{h&&h.filepath&&setAttachmentNameForPath(
h.filepath,h.filename||h.original_filename||"")}),lib.fileSet=new Set(lib.files.map(h=>h.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
f:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
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
a);if(!c)return;const d=(lib.files||[]).find(f=>f&&f.filepath===a);d&&d.filename&&setAttachmentNameForPath(
c,d.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
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
ms-center mt-4";const l=document.createElement("div"),c=String(a.id)===String(currentLeafId),d=a.id===
threadFixedBranchId,f=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),h=getCumulativeTokensForNode(
a.id);if(l.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,l.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(f)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(a.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${h}</span>
                    </div>
                    ${d?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
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
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(d=>d.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const r=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);r.innerHTML="",Object.entries(l).sort((d,f)=>f[1].
total-d[1].total).forEach(([d,f])=>{const h=document.createElement("div");h.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",h.innerHTML=`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${d}</span>
                        <span class="text-blue-400 shrink-0">${f.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${f.in}</div>
                        <div title="Output tokens">Out: ${f.out}</div>
                        <div title="Thought/Reasoning tokens">${f.thought>0?`Th: ${f.thought}`:""}</\
div>
                    </div>
                `,r.appendChild(h)}),get("br-name-input").value=branchLabelNames[n.id]||"";const c=get(
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
batchProviderLabel(a.provider)),d=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
h=escapeHtml(a.status_text||""),y=batchStateTone(a.state),v=a.thread_exists?'<button type="button" d\
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
                                <span class="truncate max-w-[16rem]">${d}</span>
                                <span><i class="fas fa-history mr-1"></i>${f}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${y}">${escapeHtml(batchStateLabelShort(
a.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${h}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${v}${w}${k}
                    </div>`;const x=r.querySelector("[data-batch-open]");x&&(x.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const L=r.querySelector("[data-batch-cancel]");L&&(L.onclick=()=>cancelBatchJob(
a));const C=r.querySelector("[data-batch-delete]");C&&(C.onclick=()=>deleteBatchJob(a)),t.appendChild(
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
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),d=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=o(h=>{h.key==="Enter"&&(h.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),r.onclick=async()=>{
const h=get("api-key-modal-input").value.trim();if(!h){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=h);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:h})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=h)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",f),d(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",f),d(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",f),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const h=get("api-key-modal-input");h&&h.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const d=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,d)}catch{}finally{
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
cls:"bg-red-600"},unban:{label:"BAN\u89E3\u9664",cls:"bg-green-600"},admin_action:{label:"\u7BA1\u7406\u8005\u306E\u64CD\u4F5C",
cls:"bg-blue-600"},account_deleted:{label:"\u30A2\u30AB\u30A6\u30F3\u30C8\u524A\u9664",cls:"bg-gray-\
600"}},n={web:"Web",android:"\u30A2\u30D7\u30EA",admin:"\u7BA1\u7406\u8005",server:"\u30B5\u30FC\u30D0\u30FC"},
i={endpoint:"\u901A\u4FE1\u5148",method:"\u65B9\u5F0F",path:"\u30D1\u30B9",client:"\u7AEF\u672B",lock_source:"\
\u30ED\u30C3\u30AF\u306E\u5BFE\u8C61",lock_reason:"\u30ED\u30C3\u30AF\u306E\u7406\u7531",remaining_seconds:"\
\u30ED\u30C3\u30AF\u306E\u6B8B\u308A",origin_username:"\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8",
origin_user_id:"\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8ID",origin_client:"\
\u30ED\u30C3\u30AF\u3092\u304B\u3051\u305F\u7AEF\u672B",lock_seconds:"\u30ED\u30C3\u30AF\u6642\u9593",
applied_to:"\u30ED\u30C3\u30AF\u306E\u7BC4\u56F2",lock_count:"\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09",
ban_at_count:"BAN\u306B\u306A\u308B\u56DE\u6570",source_username:"BAN\u306E\u8D77\u70B9",source_user_id:"\
\u8D77\u70B9\u306E\u30A2\u30AB\u30A6\u30F3\u30C8ID",action:"\u64CD\u4F5C",admin:"\u7BA1\u7406\u8005",
enabled:"\u691C\u51FA",by:"\u5B9F\u884C\u3057\u305F\u4EBA",count:"\u4EF6\u6570",previous_reason:"\u89E3\u9664\u524D\
\u306EBAN\u306E\u7406\u7531",linked_from:"\u9023\u9396\u89E3\u9664\u306E\u8D77\u70B9"},a={account:"\u30A2\
\u30AB\u30A6\u30F3\u30C8",ip:"IP\u30A2\u30C9\u30EC\u30B9",cookie:"\u7AEF\u672B\uFF08Cookie\uFF09"},r={
toggle_detection:"\u691C\u51FA\u306E\u5207\u308A\u66FF\u3048",ban:"BAN",unban:"BAN\u89E3\u9664\uFF08\u5358\u72EC\uFF09",
unban_linked:"BAN\u89E3\u9664\uFF08\u9023\u9396\uFF09",unlock:"\u30ED\u30C3\u30AF\u89E3\u9664",unblock_identifiers:"\
IP\u30FB\u7AEF\u672B\u306EBAN\u89E3\u9664"},l={userId:0,username:"",exists:!1,type:"",items:[],hasMore:!1,
selected:new Set,counts:{},total:0,data:null},c={accounts:[],selected:new Set},d=o(()=>get("bot-log-\
detail"),"view"),f=o(()=>get("bot-log-list-view"),"listView"),h=o($=>{if(!$)return"-";const N=new Date(
$);return isNaN(N.getTime())?String($):N.toLocaleString("ja-JP")},"formatTime"),y=o($=>{const N=Math.
max(0,Math.floor(Number($)||0));return N>=60?`${Math.floor(N/60)}\u5206${N%60?N%60+"\u79D2":""}`:`${N}\
\u79D2`},"formatDuration"),v=o(($,N)=>N==null||N===""?"-":$==="lock_source"?a[N]||String(N):$==="cli\
ent"||$==="origin_client"?n[N]||String(N):$==="action"?r[N]||String(N):$==="by"?N==="self"?"\u672C\u4EBA":
N==="admin"?"\u7BA1\u7406\u8005":String(N):$==="enabled"?N?"ON":"OFF":$==="remaining_seconds"||$==="\
lock_seconds"?y(N):$==="applied_to"&&Array.isArray(N)?N.map(U=>a[U]||U).join("\u30FB"):typeof N=="ob\
ject"?JSON.stringify(N,null,2):String(N),"formatDetailValue"),w=o($=>{const N=String($||"").trim();if(!N)
return"";let U=null;if(N.startsWith("{"))try{U=JSON.parse(N)}catch{U=null}if(U&&typeof U=="object"&&
!Array.isArray(U)){const O=Object.keys(U).map(q=>{const te=v(q,U[q]),B=i[q]||q;return te.length>120||
te.includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(B)}</div><pre class="bot-log-pre">${escapeHtml(
te)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(B)}:</span> <span class="text-gray-\
300">${escapeHtml(te)}</span></div>`}).join("");return N.length>600?`<details class="mt-1"><summary \
class="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${O}</details>`:`<div c\
lass="mt-1 space-y-1">${O}</div>`}return N.length>300?`<details class="mt-1"><summary class="cursor-\
pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(
N)}</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(N)}</div>`},"renderD\
etails"),k=o(async($,N)=>{const U=await apiFetch($,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(N)}),O=await U.json().catch(()=>({}));if(!U.ok)throw new Error(O.error||
String(U.status));return O},"postJson"),x=o(($,N)=>`<span class="${N} text-white px-2 py-0.5 rounded\
">${escapeHtml($)}</span>`,"badge"),L=o(()=>{const $=get("bot-log-account-list");if(!$)return;const N=c.
accounts,U=N.length&&N.every(q=>c.selected.has(q.user_id)),O=get("bot-log-list-toolbar");if(O&&(O.innerHTML=
`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${U?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!N.length){$.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}$.innerHTML=N.map(q=>{const te=[];q.exists||te.push(x("\u524A\u9664\u6E08\u307F","bg-gray-600")),
q.is_bot_banned&&te.push(x("BAN\u4E2D","bg-red-600")),q.lock_remaining_seconds>0&&te.push(x(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
ceil(q.lock_remaining_seconds/60)}\u5206\uFF09`,"bg-yellow-600"));const B=q.evidence_count?`\u8A18\u9332 ${q.
evidence_count}\u4EF6\u30FB\u6700\u7D42 ${escapeHtml(h(q.last_event_at))}`:"\u8A18\u9332\u306A\u3057";
return`
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 round\
ed p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${q.
user_id}" ${c.selected.has(q.user_id)?"checked":""} aria-label="${escapeHtml(q.username||"ID "+q.user_id)}\
 \u3092\u9078\u629E">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${q.exists?
"text-gray-200":"text-gray-400"} font-bold bot-log-wrap">${escapeHtml(q.username||"ID "+q.user_id)}<\
/span>${te.join("")}</div>
                                <div class="text-[10px] text-gray-500">${B}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${q.user_id}" data-username="${escapeHtml(
q.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),C=o(async()=>{const $=get("bot-log-acc\
ount-list");if(!$)return;$.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const N=get("bot-log-search"),U=N?N.value.
trim():"";try{const O=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(U)}`,{cache:"\
no-store"}),q=await O.json().catch(()=>({}));if(!O.ok)throw new Error(q.error||String(O.status));c.accounts=
q.accounts||[];const te=new Set(c.accounts.map(B=>B.user_id));Array.from(c.selected).forEach(B=>{te.
has(B)||c.selected.delete(B)}),L()}catch{$.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\
\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),M=o($=>{if(!$)return"";const N=$.locks||[],U=$.banned_identifiers||[],
O=[];$.exists||O.push(x("\u524A\u9664\u6E08\u307F","bg-gray-600")),$.exists&&(O.push(x($.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",$.detection_enabled?"bg-gray-600":"bg-gray-700")),O.push(x($.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",$.is_bot_banned?"bg-red-600":"bg-gray-600"))),O.push(x(N.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",N.length?"bg-yellow-600":"bg-gray-600")),U.length&&O.push(x(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${U.length}\u4EF6`,"bg-red-600")),$.is_admin&&O.push(x("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const q=!!(l.counts.ban||l.counts.related_ban),te=$.is_bot_banned?`<div class="text-\
gray-300">BAN\u306E\u7406\u7531: ${escapeHtml($.bot_ban_reason||"-")}\uFF08${escapeHtml(h($.bot_banned_at))}\
\uFF09</div>${q?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",B=N.map(ee=>{const pe=a[ee.source]||ee.source,se=ee.identifier?` ${escapeHtml(ee.identifier)}${ee.
source==="cookie"?"\u2026":""}`:"",Fe=ee.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
ee.origin.username||String(ee.origin.user_id||"-"))}\uFF08${escapeHtml(n[ee.origin.client]||ee.origin.
client||"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(pe)}${se}\
\u30FB\u6B8B\u308A${escapeHtml(y(ee.remaining_seconds))}\u30FB${escapeHtml(ee.reason||"")}${Fe}</div\
>`}).join(""),V=U.map(ee=>`<div class="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[ee.
kind]||ee.kind)} ${escapeHtml(ee.identifier)}${ee.kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(ee.
reason||"-")}\uFF08${escapeHtml(h(ee.created_at))}\uFF09</div>`).join(""),de=U.length?'<div class="t\
ext-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002</div>':
"",fe=$.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${y($.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",Q=$.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String($.appeal_count))}\u4EF6`:"",le=[];return N.length&&le.push('<button class="bot-log-unlock bg-\
yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
U.length&&le.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 r\
ounded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${O.join("")}</div>
                        ${te}
                        ${B}
                        ${V}
                        ${de}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String($.lock_count))} / ${escapeHtml(String($.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(fe)}\
\u30FB\u5931\u6557 ${escapeHtml(String($.turnstile_fail_count))} / ${escapeHtml(String($.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round(($.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round(($.behavior_score||0)*10)/10))}\uFF09${Q}</div>
                        ${le.length?`<div class="flex flex-wrap gap-2 pt-1">${le.join("")}</div>`:""}\

                    </div>`},"renderState"),E=o(()=>{const $=o((U,O,q)=>`<button class="bot-log-filt\
er ${l.type===U?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(U)}">${escapeHtml(O)} (${q})</button>`,"chip"),N=[$("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((U,O)=>l.counts[O]-l.counts[U]).forEach(U=>{N.push($(U,(t[U]||
{label:U}).label,l.counts[U]))}),N.join("")},"renderFilters"),D=o($=>{const N=t[$.event_type]||{label:$.
event_type,cls:"bg-gray-600"},U=$.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[$.client]||$.client)}</span>`:"",O=l.selected.has($.id)?"checked":"",q=$.score!==
null&&$.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String($.score))}${$.
behavior_score!==null&&$.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String($.behavior_score))}\
\uFF09`:""}</span>`:"",te=$.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
$.reasons)}</div>`:"",B=$.ip_address||$.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${$.ip_address?"IP "+escapeHtml($.ip_address):""}${$.ip_address&&$.user_agent?"\u30FB":""}${escapeHtml(
$.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${$.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${$.id}" ${O}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(h($.created_at))}</span>
                            <span class="${N.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(N.label)}\
</span>
                            ${U}
                            ${q}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${$.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${te}
                        ${w($.details)}
                        ${B}
                    </div>`},"renderItem"),P=o(()=>{const $=d();if(!$)return;const N=l.items.length?
l.items.map(D).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
U=l.items.length&&l.items.every(O=>l.selected.has(O.id));$.innerHTML=`
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py\
-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>\u4E00\u89A7\u306B\u623B\u308B</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(l.username||
"ID "+l.userId)}</div>
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-whi\
te px-2 py-1 rounded text-xs">\u66F4\u65B0</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${M(l.data&&l.data.state)}
                        <div class="flex flex-wrap gap-1 text-xs">${E()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkb\
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${U?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-\
white px-2 py-1 rounded" ${l.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u8A18\u9332\u3092\u524A\u9664${l.
selected.size?`\uFF08${l.selected.size}\u4EF6\uFF09`:""}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white\
 px-2 py-1 rounded" ${l.total?"":"disabled"}>\u3059\u3079\u3066\u306E\u8A18\u9332\u3092\u524A\u9664</button>
                        </div>
                        <div class="space-y-2">${N}</div>
                        ${l.hasMore?'<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-60\
0 text-white px-2 py-1.5 rounded text-xs">\u3055\u3089\u306B\u8AAD\u307F\u8FBC\u3080</button>':""}
                    </div>`},"render"),Z=o(async($=!1)=>{const N=d();if(!N||!l.userId)return;$||(N.innerHTML=
'<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');
const U=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&U.set("type",l.type),
$&&l.items.length&&U.set("before_id",String(l.items[l.items.length-1].id));try{const O=await apiFetch(
`/api/bot/evidence?${U.toString()}`,{cache:"no-store"}),q=await O.json().catch(()=>({}));if(!O.ok)throw new Error(
q.error||String(O.status));l.counts=q.counts||{},l.total=q.total||0,l.hasMore=!!q.has_more,q.user&&(l.
username=q.user.username||l.username,l.exists=!!q.user.exists),$?l.items=l.items.concat(q.items||[]):
(l.items=q.items||[],l.data=q),P()}catch{$||(N.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),j=o(async($,N)=>{if(confirm(N))try{const U=await k("/api/bot/evidence/delete",
Object.assign({user_id:l.userId},$));showToast(`${U.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await Z(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),W=o(async($,N,U)=>{if(confirm(N))try{await k($,{user_id:l.userId}),showToast(
U,"success"),await Z(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","e\
rror",!0)}},"runAccountAction"),re=o(()=>{const $=d();$&&($.classList.add("hidden"),$.classList.remove(
"flex"),$.innerHTML="");const N=f();N&&(N.classList.remove("hidden"),N.classList.add("flex")),l.userId=
0},"showList"),me=o(async()=>{re(),await C()},"openList"),ie=o(async($,N="")=>{const U=d();if(!U)return;
Object.assign(l,{userId:Number($)||0,username:N,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),l.selected.clear();const O=f();O&&(O.classList.add("hidden"),O.classList.remove("flex")),
U.classList.remove("hidden"),U.classList.add("flex"),await Z(!1)},"open");return{open:ie,openList:me,
showList:re,bind:o(()=>{const $=get("bot-log-modal");if(!$||$.dataset.bound==="1")return;$.dataset.bound=
"1";const N=get("bot-log-search");N&&N.addEventListener("keydown",U=>{U.key==="Enter"&&C()}),$.addEventListener(
"change",U=>{const O=U.target;if(!(!O||!O.classList))if(O.classList.contains("bot-log-select")){const q=Number(
O.getAttribute("data-log-id"));O.checked?l.selected.add(q):l.selected.delete(q),P()}else if(O.classList.
contains("bot-log-select-all"))l.items.forEach(q=>{O.checked?l.selected.add(q.id):l.selected.delete(
q.id)}),P();else if(O.classList.contains("bot-log-account-select")){const q=Number(O.getAttribute("d\
ata-user-id"));O.checked?c.selected.add(q):c.selected.delete(q),L()}else O.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(q=>{O.checked?c.selected.add(q.user_id):c.selected.delete(
q.user_id)}),L())}),$.addEventListener("click",async U=>{const O=U.target.closest("button");if(!(!O||
O.disabled))if(O.id==="bot-log-search-btn"||O.id==="bot-log-refresh-btn")O.id==="bot-log-refresh-btn"&&
N&&(N.value=""),await C();else if(O.classList.contains("bot-log-open-account")){const q=Number(O.getAttribute(
"data-user-id"));q&&await ie(q,O.getAttribute("data-username")||"")}else if(O.classList.contains("bo\
t-log-delete-accounts")){const q=Array.from(c.selected);if(!q.length||!confirm(`\u9078\u629E\u3057\u305F${q.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const te=await k("/api/bot/evidence/delete",{user_ids:q});showToast(`${te.deleted||0}\u4EF6\u306E\u8A18\u9332\
\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await C()}catch{showToast(
"\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}}else if(O.
classList.contains("bot-log-back"))await me();else if(O.classList.contains("bot-log-reload"))await Z(
!1);else if(O.classList.contains("bot-log-more"))await Z(!0);else if(O.classList.contains("bot-log-f\
ilter"))l.type=O.getAttribute("data-type")||"",l.selected.clear(),await Z(!1);else if(O.classList.contains(
"bot-log-delete-one")){const q=Number(O.getAttribute("data-log-id"));q&&await j({ids:[q]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\
\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
O.classList.contains("bot-log-delete-selected")?l.selected.size&&await j({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-delete-all")?await j({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-unlock")?await W("/api/bot/account/clear-lock",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):O.classList.contains("bot-log-unbloc\
k")&&await W("/api/bot/account/clear-identifiers",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u304C\u8D77\u70B9\u306EIP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u3059\u308B\u3068\u3001\u305D\u306EIP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002`,
"IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F")})},"bind")}})();window.BotAdminLog=
BotAdminLog;let branchCreating=!1;const branchRequests=new Map,BRANCH_SAVE_WARNING="\u5206\u5C90\u5148\u306F\u901A\u5E38\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u3057\u3066\u4FDD\u5B58\
\u3055\u308C\u3001\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u30B3\u30D4\u30FC\u5BFE\u8C61\u306E\u6DFB\u4ED8\u30FB\u751F\u6210\u30D5\u30A1\u30A4\u30EB\u3082\u4FDD\u5B58\u5BFE\u8C61\u306B\u306A\u308A\u307E\u3059\u3002";
function confirmBranchSave(){return new Promise(e=>{const t=document.createElement("dialog");t.className=
"rounded-xl bg-gray-800 text-gray-100 p-6 max-w-lg border border-gray-600",t.setAttribute("aria-labe\
l","\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90");const n=document.createElement("p");
n.textContent=BRANCH_SAVE_WARNING,t.appendChild(n);const i=document.createElement("form");i.method="\
dialog",i.className="flex justify-end gap-3 mt-4";for(const[a,r]of[["cancel","\u30AD\u30E3\u30F3\u30BB\u30EB"],
["save","\u4FDD\u5B58\u3057\u3066\u5206\u5C90"]]){const l=document.createElement("button");l.value=a,
l.textContent=r,l.className="rounded px-3 py-2 border border-gray-600",i.appendChild(l)}t.appendChild(
i),t.addEventListener("close",()=>{const a=t.returnValue==="save";t.remove(),e(a)},{once:!0}),document.
body.appendChild(t),t.showModal()})}o(confirmBranchSave,"confirmBranchSave");async function branchChatMessage(...e){
return window.ActivityLog?window.ActivityLog.operation("branch",()=>branchChatMessageLogged.apply(this,
e)):branchChatMessageLogged.apply(this,e)}o(branchChatMessage,"branchChatMessage");async function branchChatMessageLogged(e,t){
var a;if(branchCreating||abortController||!currentThreadId||!/^\d+$/.test(String(e))||temporaryChatEnabled&&
!await confirmBranchSave())return;const n=String(currentThreadId),i=n+":"+e;branchRequests.has(i)||branchRequests.
set(i,crypto.randomUUID().replace(/-/g,"")),branchCreating=!0,t&&(t.disabled=!0);try{const r=await apiFetch(
`/api/threads/${encodeURIComponent(n)}/branch`,{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({message_id:Number(e),request_id:branchRequests.get(i),confirm_persist_temporary:!!temporaryChatEnabled})}),
l=await r.json();if(!r.ok)throw new Error(l.error||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
branchRequests.delete(i),await loadMessages(l.id,{targetMessageId:l.leaf_id}),await loadThreads(),showToast(
"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90\u3057\u307E\u3057\u305F")}catch(r){(a=
window.ActivityLog)==null||a.log("branch.error"),showToast(r.message||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{branchCreating=!1,t&&(t.disabled=!1)}}o(branchChatMessageLogged,"branchChatMessa\
geLogged");function renderBranchOrigin(e){let t=get("branch-origin-banner");if(!t){const n=get("chat\
-container");if(!n)return;t=document.createElement("div"),t.id="branch-origin-banner",t.className="p\
x-4 py-2 text-xs text-gray-400 border-b border-gray-700";const i=get("total-token-bar");i?i.before(t):
n.parentNode.insertBefore(t,n)}if(t.replaceChildren(),t.hidden=!e,!!e)if(e.available&&e.thread_id&&e.
message_id){const n=document.createElement("a");n.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F",
n.className="text-blue-400 hover:underline",n.href="/c/"+encodeURIComponent(e.thread_id)+"?message="+
Number(e.message_id),n.onclick=i=>{i.preventDefault(),loadMessages(e.thread_id,{targetMessageId:e.message_id})},
t.appendChild(n)}else t.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093"}
o(renderBranchOrigin,"renderBranchOrigin");function displayMissingMessageFiles(e,t){const n=o(i=>{try{
const a=new URL(i,location.origin);return a.origin===location.origin?decodeURIComponent(a.pathname).
replace(/^\/files\/(thumb\/)?/,""):""}catch{return String(i)}},"normalize");for(const i of t){if(!i.
missing_files||!i.missing_files.length)continue;const a=e.querySelector("#msg-"+Number(i.id));if(!a)
continue;const r=new Set(i.missing_files.flatMap(l=>{const c=String(l).replace(/^\/files\//,"");return[
c,c.replace(/^\d+\//,"")]}));a.querySelectorAll("img[src], a[href], audio[src], video[src], source[s\
rc], .file-thumb[onclick]").forEach(l=>{let c=n(l.getAttribute("src")||l.getAttribute("href")||"");if(l.
classList.contains("file-thumb")){const h=(l.getAttribute("onclick")||"").match(/window.open\('([^']+)'/);
h&&(c=n(h[1]))}if(!r.has(c))return;const d=document.createElement("span");d.className="inline-block \
rounded bg-gray-800 p-3 text-xs text-gray-400",d.textContent="\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",
d.title=c.split("/").pop();const f=l.closest(".chat-image-frame")||(l.tagName==="SOURCE"?l.parentNode:
l);f&&a.contains(f)&&f.replaceWith(d)})}}o(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}o(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},r=1/0;for(let l=1;l<=i;l++){if(i%l)continue;const c=i/l,d=Math.abs(
Math.log(Math.max(1,t)/l/(Math.max(1,n)/c)));(d<r-1e-9||Math.abs(d-r)<=1e-9&&(t>=n?l>a.columns:c>a.rows))&&
(a={columns:l,rows:c},r=d)}return a}o(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,r=Math.round(e/n.columns*a/2),l=Math.round(t/n.rows*a/2),c=[];for(let d=0;d<
n.rows;d++)for(let f=0;f<n.columns;f++){const h=Math.floor(f*e/n.columns),y=Math.floor((f+1)*e/n.columns),
v=Math.floor(d*t/n.rows),w=Math.floor((d+1)*t/n.rows);c.push({index:c.length,row:d,column:f,cellLeft:h,
cellTop:v,cellRight:y,cellBottom:w,left:Math.max(0,h-r),top:Math.max(0,v-l),right:Math.min(e,y+r),bottom:Math.
min(t,w+l)})}return c}o(splitTiles,"splitTiles");function splitFits(e,t,n){return Math.floor(t/e.columns)>=
16&&Math.floor(n/e.rows)>=16}o(splitFits,"splitFits");function splitTileCaption(e,t,n,i){return`Tile\
 ${e.index+1}/${t.columns*t.rows}  row ${e.row+1}/${t.rows}, col ${e.column+1}/${t.columns}  x ${e.left}\
-${e.right}, y ${e.top}-${e.bottom} of ${n}x${i}`}o(splitTileCaption,"splitTileCaption");function splitBaseName(e){
return String(e).split("/").pop().replace(/\.[^.]*$/,"").replace(/[\\/:*?"<>|\s]+/g,"_").replace(/^[_.]+|[_.]+$/g,
"").slice(0,80)||"image"}o(splitBaseName,"splitBaseName");function splitTileName(e,t,n){const i=n.columns*
n.rows;return`${e}_part${String(t.index+1).padStart(String(i).length,"0")}of${i}_r${t.row+1}c${t.column+
1}.png`}o(splitTileName,"splitTileName");function readSplitCompression(){let e;try{e=JSON.parse(localStorage.
getItem(SPLIT_COMPRESSION_KEY)||"{}")}catch{}return e=e&&typeof e=="object"?e:{},{enabled:e.enabled!==
!1,maxSizeMB:Math.max(.1,Math.min(10,Number(e.maxSizeMB)||1)),maxDimension:splitClamp(e.maxDimension,
256,8192,1920),outputType:["original","image/jpeg","image/png","image/webp"].includes(e.outputType)?
e.outputType:"original",formatOnly:e.formatOnly===!0}}o(readSplitCompression,"readSplitCompression");
function splitOptions(){return{pieces:splitClamp(get("image-split-pieces").value,2,30,4),overlap:splitClamp(
get("image-split-overlap").value,0,50,10),includeOriginal:get("image-split-original").checked,markOriginal:get(
"image-split-mark-original").checked,markPieces:get("image-split-mark-pieces").checked}}o(splitOptions,
"splitOptions");function splitCanvas(e,t){const n=document.createElement("canvas");if(n.width=e,n.height=
t,n.width!==e||n.height!==t||!n.getContext("2d"))throw new Error("\u753B\u50CF\u304C\u5927\u304D\u3059\u304E\u308B\u305F\u3081\u51E6\u7406\u3067\u304D\u307E\u305B\u3093\u3002");
return n}o(splitCanvas,"splitCanvas");async function loadSplitImage(e){const t=URL.createObjectURL(e),
n=new Image;try{return await new Promise((i,a)=>{n.onload=i,n.onerror=()=>a(new Error(`\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093: ${e.
name}`)),n.src=t}),{image:n,release:o(()=>{n.src="",URL.revokeObjectURL(t)},"release")}}catch(i){throw URL.
revokeObjectURL(t),i}}o(loadSplitImage,"loadSplitImage");function drawSplitOverview(e,t,n,i,a,r){const l=e.
getContext("2d"),c=e.width/n,d=e.height/i;l.drawImage(t,0,0,e.width,e.height);const f=r[0],h=f.right-
f.cellRight,y=f.bottom-f.cellBottom;l.fillStyle="rgba(250,204,21,0.18)";const v=[];for(let k=1;k<a.columns;k++){
const x=Math.floor(k*n/a.columns);l.fillRect((x-h)*c,0,2*h*c,e.height),v.push([x*c,0,x*c,e.height])}
for(let k=1;k<a.rows;k++){const x=Math.floor(k*i/a.rows);l.fillRect(0,(x-y)*d,e.width,2*y*d),v.push(
[0,x*d,e.width,x*d])}drawSplitLines(l,v,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const w=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));l.font=`bold ${w}px monospace`,
l.textAlign="center",l.textBaseline="middle";for(const k of r){const x=(k.cellLeft+k.cellRight)/2*c,
L=(k.cellTop+k.cellBottom)/2*d;l.lineWidth=Math.max(2,w/10),l.strokeStyle="rgba(0,0,0,0.7)",l.strokeText(
String(k.index+1),x,L),l.fillStyle="#facc15",l.fillText(String(k.index+1),x,L)}}o(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,r,l,c])=>{e.moveTo(a,r),e.lineTo(l,c)}),e.stroke(),e.restore()}o(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(r=>r?i(
r):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}o(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const r=a.
naturalWidth,l=a.naturalHeight,c=chooseSplitGrid(t.pieces,r,l);if(!splitFits(c,r,l))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const d=splitTiles(r,l,c,t.overlap),f=splitBaseName(e.name),h=[],y=o(async v=>{h.push(await applyImageCompression(
v,n))},"append");if(t.includeOriginal)if(t.markOriginal){const v=splitCanvas(r,l);drawSplitOverview(
v,a,r,l,c,d),await y(await splitCanvasFile(v,`${f}_split_grid.png`))}else await y(e);for(const v of d){
const w=v.right-v.left,k=v.bottom-v.top;let x=Math.max(12,Math.min(40,Math.max(w,k)/45)),L=0;const C=splitTileCaption(
v,c,r,l);if(t.markPieces){const D=splitCanvas(1,1),P=D.getContext("2d");P.font=`bold ${x}px monospac\
e`,x=Math.max(6,x*Math.min(1,w*.96/P.measureText(C).width)),L=Math.max(12,Math.floor(x*1.8)),D.width=
0}const M=splitCanvas(w,k+L),E=M.getContext("2d");if(L&&(E.fillStyle="#111827",E.fillRect(0,0,w,L)),
E.drawImage(a,v.left,v.top,w,k,0,L,w,k),t.markPieces){E.fillStyle="#fff",E.font=`bold ${x}px monospa\
ce`,E.textBaseline="middle",E.fillText(C,w*.02,L/2);const D=[];for(const P of[v.cellLeft,v.cellRight])
P>v.left&&P<v.right&&D.push([P-v.left,L,P-v.left,L+k]);for(const P of[v.cellTop,v.cellBottom])P>v.top&&
P<v.bottom&&D.push([0,L+P-v.top,w,L+P-v.top]);drawSplitLines(E,D,Math.max(1.5,Math.min(4,Math.max(w,
k)/500)),!0)}await y(await splitCanvasFile(M,splitTileName(f,v,c)))}return h}finally{i.release()}}o(
renderSplitFiles,"renderSplitFiles");function clearSplitDownloads(){splitState.urls.forEach(e=>URL.revokeObjectURL(
e)),splitState.urls=[],get("image-split-downloads").replaceChildren()}o(clearSplitDownloads,"clearSp\
litDownloads");function closeImageSplit(){!get("image-split-modal")||splitState.busy||(splitState.revision++,
splitState.release&&splitState.release(),splitState.release=null,splitState.image=null,splitState.files=
[],clearSplitDownloads(),get("image-split-preview").width=0,hideModal("image-split-compression-modal"),
hideModal("image-split-modal"))}o(closeImageSplit,"closeImageSplit");async function openImageSplit(e){
const t=Array.from(e).filter(Boolean);t.length&&(t.length>10&&showToast("\u753B\u50CF\u306E\u5206\u5272\u306F\u4E00\u5EA6\u306B10\u679A\u307E\u3067\u3067\u3059\u3002\u5148\u982D10\u679A\u306E\u307F\u4F7F\u3044\u307E\
\u3059\u3002","warning",!0),!splitState.busy&&(closeImageSplit(),splitState.files=t.slice(0,10),splitState.
index=0,showModal("image-split-modal"),await loadSplitPreview()))}o(openImageSplit,"openImageSplit");
async function loadSplitPreview(){const e=++splitState.revision;splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null,get("image-split-status").textContent="\u8AAD\u307F\u8FBC\u307F\u4E2D...";
try{const t=await loadSplitImage(splitState.files[splitState.index]);if(e!==splitState.revision){t.release();
return}splitState.image=t.image,splitState.release=t.release,get("image-split-status").textContent="",
updateSplitPreview()}catch(t){e===splitState.revision&&(get("image-split-status").textContent=t.message,
updateSplitPreview())}}o(loadSplitPreview,"loadSplitPreview");function updateSplitPreview(){const e=splitOptions(),
t=splitState.image,n=get("image-split-preview");get("image-split-mark-original").disabled=!e.includeOriginal||
splitState.busy,get("image-split-page").textContent=`${splitState.index+1} / ${splitState.files.length}\
\uFF08\u3059\u3079\u3066\u540C\u3058\u8A2D\u5B9A\u3067\u5206\u5272\u3057\u307E\u3059\uFF09`,get("ima\
ge-split-prev").disabled=splitState.busy||splitState.index===0,get("image-split-next").disabled=splitState.
busy||splitState.index>=splitState.files.length-1,get("image-split-count").textContent=`\u4F5C\u6210\u3055\u308C\u308B\u753B\u50CF: ${(e.
pieces+Number(e.includeOriginal))*splitState.files.length}\u679A`;let i=!1;if(t){const a=t.naturalWidth,
r=t.naturalHeight,l=chooseSplitGrid(e.pieces,a,r);i=splitFits(l,a,r);const c=Math.min(1,800/a,520/r);
n.width=Math.round(a*c),n.height=Math.round(r*c),i?drawSplitOverview(n,t,a,r,l,splitTiles(a,r,l,e.overlap)):
n.getContext("2d").drawImage(t,0,0,n.width,n.height),get("image-split-dimensions").textContent=i?`${l.
columns}\u5217 \xD7 ${l.rows}\u884C\u30FB1\u679A\u3042\u305F\u308A\u7D04 ${Math.floor(a/l.columns)}\xD7${Math.
floor(r/l.rows)} px\uFF08\u5143\u753B\u50CF ${a}\xD7${r} px\uFF09`:`\u3053\u306E\u753B\u50CF\u306F\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${e.
pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3002\u5206\u5272\u6570\u3092\u6E1B\u3089\u3057\u3066\u304F\u3060\u3055\u3044\u3002`}else
n.width=0,get("image-split-dimensions").textContent="";get("image-split-save").disabled=get("image-s\
plit-attach").disabled=splitState.busy||!i}o(updateSplitPreview,"updateSplitPreview");function setSplitBusy(e){
splitState.busy=e,get("image-split-modal").querySelectorAll("input,button").forEach(t=>{t.disabled=e}),
updateSplitPreview()}o(setSplitBusy,"setSplitBusy");async function runImageSplit(...e){return window.
ActivityLog?window.ActivityLog.operation("image_split",()=>runImageSplitLogged.apply(this,e)):runImageSplitLogged.
apply(this,e)}o(runImageSplit,"runImageSplit");async function runImageSplitLogged(e){var i;if(splitState.
busy)return;const t=splitOptions(),n=readSplitCompression();if(e&&browserFastModeEnabled&&browserFastLocalFiles.
size+splitState.files.length*(t.pieces+Number(t.includeOriginal))>BROWSER_FAST_MAX_IMAGES){get("imag\
e-split-status").textContent="\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059";
return}setSplitBusy(!0),clearSplitDownloads();try{const a=[];splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null;for(let r=0;r<splitState.files.length;r++)get("image-s\
plit-status").textContent=`\u51E6\u7406\u4E2D... ${r+1} / ${splitState.files.length}`,a.push(...await renderSplitFiles(
splitState.files[r],t,n));if(e){const r=Array.from(browserFastLocalFiles.values()).reduce((l,c)=>l+c.
file.size,0);if(browserFastModeEnabled&&r+a.reduce((l,c)=>l+c.size,0)>BROWSER_FAST_MAX_BYTES)throw new Error(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059");
await handleFiles(a,{skipCompression:!0}),setSplitBusy(!1),closeImageSplit()}else{for(const r of a){
const l=URL.createObjectURL(r);splitState.urls.push(l);const c=document.createElement("a");c.href=l,
c.download=r.name,c.textContent=r.name,c.className="block underline",get("image-split-downloads").append(
c),c.click()}get("image-split-status").textContent=`${a.length}\u679A\u306E\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3057\u307E\u3057\u305F\u3002\u4FDD\u5B58\u3055\u308C\u306A\u3044\u753B\u50CF\u306F\u4E0B\u306E\u30EA\u30F3\u30AF\u304B\u3089\u4FDD\u5B58\u3067\u304D\u307E\
\u3059\u3002`}}catch(a){(i=window.ActivityLog)==null||i.log("image_split.error"),get("image-split-st\
atus").textContent=a.message||"\u753B\u50CF\u3092\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002"}finally{
if(setSplitBusy(!1),splitState.files.length){const a=get("image-split-status").textContent;await loadSplitPreview(),
get("image-split-status").textContent=a}}}o(runImageSplitLogged,"runImageSplitLogged");function openSplitCompression(){
const e=readSplitCompression(),t="image-split-compression";get(`${t}-enabled`).checked=e.enabled,get(
`${t}-output-type`).value=e.outputType,get(`${t}-format-only`).checked=e.formatOnly,get(`${t}-max-si\
ze`).value=e.maxSizeMB,get(`${t}-max-dim`).value=e.maxDimension,syncSplitCompressionUi(),showModal(`${t}\
-modal`)}o(openSplitCompression,"openSplitCompression");function syncSplitCompressionUi(){const e="i\
mage-split-compression";for(const t of["size","dim"]){const n=get(`${e}-format-only`).checked;get(`${e}\
-max-${t}`).disabled=n,get(`${e}-${t}-wrap`).style.opacity=n?"0.4":"1"}}o(syncSplitCompressionUi,"sy\
ncSplitCompressionUi"),window.addEventListener("popstate",()=>{splitState.busy||closeImageSplit()}),
document.addEventListener("keydown",e=>{e.key!=="Escape"||!get("image-split-modal")||(get("image-spl\
it-compression-modal").classList.contains("hidden")?get("image-split-modal").classList.contains("hid\
den")||closeImageSplit():hideModal("image-split-compression-modal"))}),document.addEventListener("DO\
MContentLoaded",()=>{if(get("image-split-modal")){get("upload-split-btn").onclick=()=>get("image-spl\
it-input").click(),get("image-split-input").onchange=e=>{openImageSplit(e.target.files),e.target.value=
""},get("image-split-prev").onclick=()=>{splitState.index--,loadSplitPreview()},get("image-split-nex\
t").onclick=()=>{splitState.index++,loadSplitPreview()};for(const e of["pieces","overlap","original",
"mark-original","mark-pieces"])get(`image-split-${e}`).oninput=updateSplitPreview;get("image-split-c\
ancel").onclick=closeImageSplit,get("image-split-save").onclick=()=>runImageSplit(!1),get("image-spl\
it-attach").onclick=()=>runImageSplit(!0),get("image-split-settings").onclick=openSplitCompression,get(
"image-split-compression-enabled").onchange=get("image-split-compression-format-only").onchange=syncSplitCompressionUi,
get("image-split-compression-close").onclick=()=>hideModal("image-split-compression-modal"),get("ima\
ge-split-compression-save").onclick=()=>{const e="image-split-compression",t=get(`${e}-max-size`),n=get(
`${e}-max-dim`);if(!t.reportValidity()||!n.reportValidity())return;const i={enabled:get(`${e}-enable\
d`).checked,outputType:get(`${e}-output-type`).value,formatOnly:get(`${e}-format-only`).checked,maxSizeMB:Number(
t.value),maxDimension:Number(n.value)};localStorage.setItem(SPLIT_COMPRESSION_KEY,JSON.stringify(i)),
hideModal(`${e}-modal`),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","success")}}});
function takePromptForSend(e=!0){const t=get("prompt-input"),n=get("upload-list"),i=document.querySelector(
".composer-dock"),a=i==null?void 0:i.getBoundingClientRect().height,r=[t,get("file-preview"),get("qu\
ote-bar")].filter(E=>E&&E.getClientRects().length&&getComputedStyle(E).display!=="none").map(E=>E.getBoundingClientRect()),
l=Math.min(...r.map(E=>E.left)),c=Math.min(...r.map(E=>E.top)),d={left:l,top:c,width:Math.max(...r.map(
E=>E.right))-l,height:Math.max(...r.map(E=>E.bottom))-c},f=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,h={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},y=document.createElement("div");
y.hidden=!0,y.className="prompt-send-held-files",y.setAttribute("aria-hidden","true"),y.append(...h.
rows),document.body.appendChild(y);let v=!1,w=!1,k=!1;const x=o(()=>{v=!0},"onInput");t.addEventListener(
"input",x),t.value="",["file-input","photo-input","mask-input"].forEach(E=>{const D=get(E);D&&(D.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let L;i&&e&&!f&&
i.animate&&(L=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const C=o(()=>{!w||k||(h.files.forEach(E=>{var P,Z;const D=(Z=
(P=E==null?void 0:E.rowObj)==null?void 0:P.row)==null?void 0:Z.getAttribute("data-local-url");D&&URL.
revokeObjectURL(D)}),y.remove())},"releaseFiles"),M=o((E,D=!0)=>{w||(w=!0,E=E&&h.navigation===threadLoadSequence,
k=E,t.removeEventListener("input",x),E?(!v&&!t.value&&(t.value=h.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...h.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=h.mask),h.files.forEach((P,Z)=>browserFastLocalFiles.set(Z,P)),h.
markers.forEach(P=>markerAppliedUploads.add(P)),n&&n.prepend(...h.rows),y.remove(),!currentQuote&&h.
quote&&(currentQuote=h.quote,currentQuoteMessageId=h.quoteId,renderQuoteBarContent(h.quote),get("quo\
te-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=h.progress),updateFilePreview(),
updateMaskPreview(),schedulePromptTokenEstimate(!0)):D&&C())},"finish");return{source:d,canRetry:o(()=>!v&&
h.navigation===threadLoadSequence,"canRetry"),stopDock:o(()=>L==null?void 0:L.cancel(),"stopDock"),files:h.
files,releaseFiles:C,accept:o((E=!0)=>M(!1,E),"accept"),restore:o(()=>M(!0),"restore")}}o(takePromptForSend,
"takePromptForSend");function animatePromptToMessage(e,t,n){var y;let i,a=0,r=!1;const l=[],c=get("c\
hat-container"),d=get("prompt-input"),f=window.matchMedia("(prefers-reduced-motion: reduce)"),h=o(()=>{
var v,w;r||(r=!0,(v=window.ActivityLog)==null||v.log("ui.hide"),t.stopDock(),cancelAnimationFrame(a),
l.forEach(k=>k.cancel()),i==null||i.remove(),e==null||e.classList.remove("prompt-send-hidden"),window.
removeEventListener("resize",h),window.removeEventListener("popstate",h),window.removeEventListener(
"pagehide",h),document.removeEventListener("visibilitychange",h),c==null||c.removeEventListener("whe\
el",h),c==null||c.removeEventListener("touchstart",h),d==null||d.removeEventListener("input",h),(w=window.
visualViewport)==null||w.removeEventListener("resize",h),f.removeEventListener("change",h))},"stop");
if(!n||f.matches||!e||!Element.prototype.animate)return h;try{e.classList.remove("fade-in"),e.classList.
add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var v;try{if(!e.isConnected)return h();const w=e.
querySelector(".message-bubble"),k=w.getBoundingClientRect(),x=c.getBoundingClientRect();if(!k.width||
!k.height||k.bottom<=x.top||k.top>=x.bottom)return h();const L=t.source;i=w.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach(Z=>Z.removeAttribute("id")),i.querySelectorAll(".msg-contro\
ls, video, audio").forEach(Z=>Z.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute("ari\
a-hidden","true"),i.inert=!0,Object.assign(i.style,{width:k.width+"px",height:k.height+"px",left:k.left+
"px",top:k.top+"px"}),document.body.appendChild(i),(v=window.ActivityLog)==null||v.log("ui.show");const C=Math.
min(L.width/k.width,Math.max(L.height,42)/k.height,1),M=L.left+(L.width-k.width*C)/2-k.left,E=L.top+
(L.height-k.height*C)/2-k.top,D=i.animate([{transform:`translate(${M}px, ${E}px) scale(${C})`,borderRadius:"\
12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"translate(0, 0\
) scale(1)",borderRadius:getComputedStyle(w).borderRadius,backgroundColor:getComputedStyle(w).backgroundColor}],
{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});l.push(D),l.push(i.animate([{opacity:1,
offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),e.classList.remove("prompt\
-send-hidden"),l.push(w.animate([{opacity:0,offset:0},{opacity:0,offset:.8},{opacity:1}],{duration:350,
fill:"both"}));const P=o(()=>{if(!e.isConnected||!userAutoScroll||document.hidden)return h();const Z=w.
getBoundingClientRect();i.style.left=Z.left+"px",i.style.top=Z.top+"px",a=requestAnimationFrame(P)},
"watch");a=requestAnimationFrame(P),D.finished.then(h,h)}catch{console.warn("Prompt send animation f\
ailed"),h()}}),window.addEventListener("resize",h),window.addEventListener("popstate",h),window.addEventListener(
"pagehide",h),document.addEventListener("visibilitychange",h),c.addEventListener("wheel",h,{passive:!0}),
c.addEventListener("touchstart",h,{passive:!0}),d.addEventListener("input",h),(y=window.visualViewport)==
null||y.addEventListener("resize",h),f.addEventListener("change",h)}catch{console.warn("Prompt send \
animation failed"),h()}return h}o(animatePromptToMessage,"animatePromptToMessage");let costPricing=null,
costPricingFailed=!1,costBarPath=[],costBarReady=!1;const COST_ESTIMATE_NOTE="\u6709\u6599API\u306E\u901A\u5E38\u6599\u91D1\u306B\u3088\u308B\u6982\u7B97\uFF08USD\uFF09\u3002\u56DE\
\u7B54\u3054\u3068\u306E\u4FDD\u5B58\u6E08\u307F\u30E2\u30C7\u30EB\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5165\u529B\u4F7F\u7528\u91CF\u304C\u306A\u3044\u56DE\u7B54\u306F\u4F1A\u8A71\u5C65\u6B74\u304B\u3089\u63A8\u5B9A\u3057\u307E\u3059\u3002\u30E2\u30C7\u30EBAPI\u306E\u547C\u3073\u51FA\u3057\u3054\u3068\u306B\u5165\u529B\u30FB\u51FA\u529B\uFF08Thinking\u3092\u542B\u3080\uFF09\u3092\u5408\u7B97\u3057\u307E\u3059\u3002Ideogram\u306F\u30E2\u30C7\u30EB\u30FB\u54C1\u8CEA\u30FB\u901F\u5EA6\u30FB\u51E6\
\u7406\u5225\u306E\u5358\u4FA1\u3068\u753B\u50CF\u679A\u6570\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5358\u4FA1\u672A\u516C\u8868\u306E\u8A2D\u5B9A\u306F\u516C\u8868\u6700\u5B89\u5024\u3092\u4E0B\u9650\u3068\u3057\u3066\u300C\u4EE5\u4E0A\u30FB\u4E00\u90E8\u672A\u7B97\u5165\u300D\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u4F7F\u7528\u91CF\u672A\u53D6\u5F97\u306E\u547C\u3073\u51FA\u3057\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002API\u5831\u544A\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u30FB\u66F8\u8FBC\u3068\u691C\u7D22\u306A\u3069\u306E\u8FFD\u52A0\u6599\u91D1\
\u3092\u542B\u307F\u307E\u3059\u3002Batch API\u306F\u4E8B\u696D\u8005\u3054\u3068\u306E\u5272\u5F15\u7387\uFF08\u5272\u5F15\u306E\u306A\u3044\u4E8B\u696D\u8005\u3092\u542B\u3080\uFF09\u3067\u8A08\u7B97\u3057\u3001\u5272\u5F15\u5BFE\u8C61\u5916\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u306F\u901A\u5E38\u5358\u4FA1\u306E\u307E\u307E\u3067\u3059\u3002\u4FDD\u5B58\u6642\u9593\u3084\u5B9F\u884C\u6642\u9593\u306A\u3069\u53D6\u5F97\u3067\u304D\u306A\u3044\u8AB2\u91D1\u60C5\u5831\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002\u305D\u306E\u4ED6\
\u306E\u753B\u50CF\u30FB\u97F3\u58F0\u30FB\u52D5\u753B\u751F\u6210\u3001\u7A0E\u3001\u7121\u6599\u67A0\u30FB\u500B\u5225\u5951\u7D04\u306F\u542B\u307F\u307E\u305B\u3093\u3002\u6642\u9593\u5E2F\u3084\u5165\u529B\u7A2E\u5225\u3067\u5358\u4FA1\u304C\u7570\u306A\u308B\u5834\u5408\u306F\u7BC4\u56F2\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u904E\u53BB\u306E\u8ACB\u6C42\u984D\u3092\u518D\u73FE\u3059\u308B\u3082\u306E\u3067\u306F\u3042\u308A\u307E\u305B\u3093\u3002";
function costTokenNumber(e){if(e==null||e==="")return null;const t=Number(e);return Number.isFinite(
t)&&t>=0?t:null}o(costTokenNumber,"costTokenNumber");function costCacheUsage(e){let t=e;if(typeof e==
"string"){if(!e.includes("cache_"))return null;try{t=JSON.parse(e)}catch{return null}}if(!t||typeof t!=
"object"||!Array.isArray(t.pricing_calls))return null;let n=0,i=0,a=!1;const r={};for(const l of t.pricing_calls){
if(!l||typeof l!="object")continue;const c=costTokenNumber(l.cache_read_tokens),d=costTokenNumber(l.
cache_write_tokens);let f=0;for(const h of["5m","1h","30m","12h"]){const y=costTokenNumber(l[`cache_\
write_${h}_tokens`]);y!==null&&(r[h]=(r[h]||0)+y,f+=y,a=!0)}(c!==null||d!==null)&&(a=!0),n+=c||0,i+=
Math.max(d||0,f)}return a?{read:n,write:i,ttls:r}:null}o(costCacheUsage,"costCacheUsage");function costRate(e,t,n){
const i=e&&(e.aliases[t]||t),a=e&&e.models[i];if(!a)return null;const r={...a};for(const l of r.changes||
[])n>=l.from&&Object.assign(r,l);return r}o(costRate,"costRate");function addRequestCost(e,t,n,i,a=!1){
var h,y;if(e.requests++,String(t.model).startsWith("grok-")&&Number.isSafeInteger(t.cost_in_usd_ticks)&&
t.cost_in_usd_ticks>=0){e.low+=t.cost_in_usd_ticks/1e10,e.high+=t.cost_in_usd_ticks/1e10,e.priced++,
t.complete===!1&&e.unpriced++;return}const r=costRate(n,t.model,i);if(!r){e.unpriced++;return}if(r.unit===
"image"){const v=costTokenNumber(t.images),w=(h=r.image_rates)==null?void 0:h[t.operation],k=w==null?
void 0:w[r.image_selector==="speed"?t.speed:t.quality],x=(y=k==null?void 0:k.minimum)!=null?y:w==null?
void 0:w.minimum,L=typeof w=="number"?w:typeof k=="number"?k:x,C=Number.isFinite(L)&&typeof w!="numb\
er"&&typeof k!="number";if(v===null||!Number.isFinite(L)){e.unpriced++;return}e.low+=v*L,e.high+=v*L,
e.priced++,C&&(e.lowerBound=!0),(t.complete===!1||C)&&e.unpriced++;return}const l=costTokenNumber(t.
input_tokens),c=costTokenNumber(t.output_tokens);if(l===null&&c===null){e.unpriced++;return}const d=r.
threshold!=null&&(r.threshold_inclusive?(l||0)>=r.threshold:(l||0)>r.threshold);d&&e.long++;const f=requestTokenCost(
t,r,d,a);e.low+=f.low,e.high+=f.high,f.lowerBound&&(e.lowerBound=!0),e.priced++,(l===null||c===null||
t.complete===!1||f.incomplete)&&e.unpriced++}o(addRequestCost,"addRequestCost");function legacyImageRequest(e,t){
let n=e.image_url;try{typeof n=="string"&&(n=JSON.parse(n))}catch{n=n?[n]:[]}const i=Array.isArray(n)?
n.length:0,a=t.get(String(e.parent_id)),r=(a==null?void 0:a.prompt_options)||{};let l=a==null?void 0:
a.image_url,c=Array.isArray(l)?l.length>0:!!l&&l!=="[]"&&l!=="null",d=a;const f=new Set;for(;!c&&(d==
null?void 0:d.parent_id)!=null&&!f.has(String(d.id));)f.add(String(d.id)),d=t.get(String(d.parent_id)),
(d==null?void 0:d.role)==="assistant"&&d.image_url&&d.image_url!=="[]"&&d.image_url!=="null"&&(c=!0);
const h=["remove_object","remove_background"].includes(r.ideogram_operation)?r.ideogram_operation:c&&
e.model==="ideogram-4.5"?"edit":"generate";let y=r.ideogram_quality;(!["very_low","low","medium","hi\
gh"].includes(y)||y==="very_low"&&h==="generate")&&(y=h==="edit"?"medium":"high");const v=a?["turbo",
"default","quality"].includes(r.ideogram_speed)?r.ideogram_speed:"default":null;return{model:e.model,
images:i||null,operation:h,quality:y,speed:v,complete:!!a}}o(legacyImageRequest,"legacyImageRequest");
function estimateConversationCost(e,t=costPricing,n=e){var l,c,d,f,h,y;const i={low:0,high:0,priced:0,
unpriced:0,inferred:0,long:0,requests:0},a=new Map((n||[]).filter(Boolean).map(v=>[String(v.id),v])),
r=new Set;for(const v of e||[]){if(!v||v.role!=="assistant"||r.has(String(v.id)))continue;r.add(String(
v.id));let w=null;try{w=typeof v.thought_data=="string"?JSON.parse(v.thought_data):v.thought_data}catch{}
const k=((l=/^\d{4}-\d{2}-\d{2}/.exec(v.created_at||""))==null?void 0:l[0])||(t==null?void 0:t.checked_at),
x=((c=v.batch_job)==null?void 0:c.state)==="JOB_STATE_SUCCEEDED";if(Array.isArray(w==null?void 0:w.pricing_calls)){
w.pricing_calls.length||i.priced++;for(const E of w.pricing_calls){if(!E||typeof E!="object"){i.unpriced++;
continue}const D=((d=/^\d{4}-\d{2}-\d{2}/.exec(E.created_at||""))==null?void 0:d[0])||k,P=costRate(t,
E.model,D),Z=(P==null?void 0:P.image_selector)==="speed"&&!E.speed&&E.model===v.model?{...E,speed:legacyImageRequest(
v,a).speed}:E;addRequestCost(i,Z,t,D,x)}continue}if(((f=costRate(t,v.model,k))==null?void 0:f.unit)===
"image"){addRequestCost(i,legacyImageRequest(v,a),t,k);continue}const L=costTokenNumber(v.tokens_out);
if(L===null&&!v.content&&!v.tokens)continue;if(L===null){i.unpriced++;continue}let C=(h=costTokenNumber(
w==null?void 0:w.pricing_input_tokens))!=null?h:costTokenNumber(v.tokens_in),M=!1;if(C===null){i.inferred++,
C=0;let E=v.parent_id;const D=new Set([String(v.id)]);for(E==null&&(M=!0);E!=null;){const P=String(E);
if(D.has(P)||!a.has(P)){M=!0;break}D.add(P);const Z=a.get(P),j=Z.role==="user"?costTokenNumber(Z.tokens_in):
(y=costTokenNumber(Z.tokens_content))!=null?y:costTokenNumber(Z.tokens_out)===null?null:Math.max(0,Number(
Z.tokens_out)-(costTokenNumber(Z.tokens_thought)||0));j===null?M=!0:C+=j,E=Z.parent_id}}addRequestCost(
i,{model:v.model,input_tokens:C,output_tokens:L,complete:!M},t,k,x)}return i}o(estimateConversationCost,
"estimateConversationCost");function costEstimateLabel(e,t="\u8CBB\u7528\u4E88\u60F3"){if(!e||!e.priced)
return`${t}: \u7B97\u51FA\u4E0D\u53EF`;const n=o(a=>a>0&&a<1e-6?"<$0.000001":`$${a.toFixed(6)}`,"mon\
ey"),i=e.lowerBound?`${n(e.low)}\u4EE5\u4E0A`:e.high-e.low>1e-12?`${n(e.low)}\u301C${n(e.high)}`:n(e.
low);return`${t}: ${i}${e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":""}`}o(costEstimateLabel,
"costEstimateLabel");function refreshCostBar(e=costBarPath){costBarReady=!0,costBarPath=e;const t=get(
"total-cost-estimate");if(!t)return;const n=e.some(a=>a&&a.role==="assistant"&&(a.content||Number(a.
tokens_out)>0));t.classList.toggle("hidden",!n);const i=estimateConversationCost(e,costPricing,e);t.
textContent=costPricing?costEstimateLabel(i,"\u8CBB\u7528\u4E88\u60F3"):`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?
"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`,t.title=COST_ESTIMATE_NOTE,
messageMeta.__total__&&(messageMeta.__total__.cost_estimate=costPricing?i:null),t.onclick=()=>openTokenDetail(
"__total__")}o(refreshCostBar,"refreshCostBar"),fetch("/static/model_pricing.json?checked=2026-11-06",
{credentials:"same-origin"}).then(e=>{if(!e.ok)throw new Error("pricing_unavailable");return e.json()}).
then(e=>{costPricing=e,costBarReady&&refreshCostBar()}).catch(()=>{costPricingFailed=!0,costBarReady&&
refreshCostBar()});function requestTokenCost(e,t,n,i){var O,q,te,B,V,de,fe,Q,le,ee;const a=costTokenNumber(
e.input_tokens)||0,r=costTokenNumber(e.output_tokens)||0,l=costTokenNumber(e.cache_read_tokens)||0,c=costTokenNumber(
e.cache_write_tokens)||0,d=["5m","1h","30m","12h"],f=d.map(pe=>costTokenNumber(e[`cache_write_${pe}_\
tokens`])||0),h=Math.max(c,f.reduce((pe,se)=>pe+se,0)),y=Math.max(0,a-l-h),v=n?t.long_input:t.input,
w=n?t.long_output:t.output,k=n&&(O=t.long_cached_input)!=null?O:t.cached_input,x=t.cache_write_multipliers||
{},L=i&&(q=t.batch_multiplier)!=null?q:1,C=i?(te=t.batch_cached_multiplier)!=null?te:L:1,M=k!=null&&
L>0?C/L:1,E=Math.max(0,h-f.reduce((pe,se)=>pe+se,0)),D=Object.values(x);let P=!!e.pricing_incomplete||
l+h>a||!!e.explicit_cache;l&&k==null&&(P=!0);let Z=y*v+l*(k!=null?k:v)*M,j=y*(n?v:(B=t.max_input)!=null?
B:v)+l*(n?k!=null?k:v:(de=(V=t.max_cached_input)!=null?V:k)!=null?de:v)*M;for(let pe=0;pe<d.length;pe++){
const se=x[d[pe]];f[pe]&&se==null&&(P=!0),Z+=f[pe]*v*(se!=null?se:1),j+=f[pe]*(n?v:(fe=t.max_input)!=
null?fe:v)*(se!=null?se:1)}E&&(P=!0,Z+=E*v*(D.length?Math.min(...D):1),j+=E*(n?v:(Q=t.max_input)!=null?
Q:v)*(D.length?Math.max(...D):1));let W=(Z+r*w)*L/1e6,re=(j+r*((le=t.max_output)!=null?le:w))*L/1e6,
me=!1;const ie=t.tool_rates||{},ke=o((pe,se)=>{if(pe){if(ie[se]==null){P=!0;return}W+=pe*ie[se],re+=
pe*ie[se]}},"add"),$=costTokenNumber(e.web_search_calls)||0;ke($,e.search_preview?"web_search_previe\
w":"web_search"),$&&!e.search_preview&&t.search_content_tokens&&(W+=$*t.search_content_tokens*v/1e6,
re+=$*t.search_content_tokens*v/1e6),ke(costTokenNumber(e.file_search_calls),"file_search");const N=costTokenNumber(
e.grounding_queries)||0;ke(ie.grounding_prompt!=null?Math.min(1,N):N,ie.grounding_prompt!=null?"grou\
nding_prompt":"grounding_query");const U=costTokenNumber(e.code_execution_calls)||0;return U&&!e.code_execution_free&&
(t.provider==="openai"?(ke(((ee=costTokenNumber(e.container_sessions))!=null?ee:1)*(costTokenNumber(
e.container_memory_gb)||1),"container_session"),P=!0):t.provider==="anthropic"?(ke(1,"code_execution\
_minimum"),P=!0,me=!0):ke(U,"code_execution")),{low:W,high:re,incomplete:P,lowerBound:me}}o(requestTokenCost,
"requestTokenCost");
