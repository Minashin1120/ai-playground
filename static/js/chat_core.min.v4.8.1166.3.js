function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),r=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const L=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",L.forEach((M,E)=>{
const D=M.getAttribute("data-local-url"),N=M.getAttribute("data-filename"),G=M.querySelector("img.up\
load-preview")!==null;let j;if(G){let J=D;if(!J&&N){const oe=N.replace(/^\d+\//,"");J=buildAttachmentPreviewUrl(
oe)}J&&(j=document.createElement("img"),j.src=J,j.className="thumb-item shadow-sm",j.dataset.viewerSrc=
J,j.dataset.viewerFilename=N||J.split("/").pop(),j.uploadRow=M,j.onclick=function(oe){oe.preventDefault(),
openImageViewer(this.dataset.viewerSrc,".thumb-item")},j.onerror=function(){this.parentElement.replaceChild(
d("ERR"),this)})}j||(j=d("FILE")),j.style.animationDelay=`${E*32}ms`,a.appendChild(j)}),L.length>0?a.
classList.remove("hidden"):a.classList.add("hidden")}function d(L){const M=document.createElement("d\
iv");return M.className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-\
[9px] shadow-sm font-bold",M.innerText=L,M}o(d,"createFileThumb");const m=collectImageUrlsForSend(),
h=uploadProgressState.total,y=uploadProgressState.completed,v=uploadProgressState.active;h===0&&(e.classList.
add("hidden"),n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));
const w=get("send-btn"),k=get("mic-btn"),x=get("mask-btn"),C=isStopMode;if(v>0?(w&&(w.disabled=!0),k&&
(k.disabled=!0),x&&(x.disabled=!0)):C||(w&&(w.disabled=!1),k&&(k.disabled=!1),x&&(x.disabled=!1)),v>
0){const L=`Preparing... (${y}/${h})`;e.classList.remove("hidden"),t.innerText=L,r&&(r.innerText=`(${y}\
/${h})`);let M=y*100,E=0;for(let G in uploadProgressState.perFilePct)M+=uploadProgressState.perFilePct[G],
E++;const D=h>0?M/(h*100)*100:0,N=`${Math.min(100,D)}%`;n&&i&&(n.classList.remove("hidden"),i.style.
width=N),l&&c&&(l.classList.remove("hidden"),c.style.width=N)}else r&&(r.innerText=""),l&&l.classList.
add("hidden"),m.length>0?(e.classList.remove("hidden"),t.innerText=`${m.length} files ready`,n&&n.classList.
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
rker-color]").forEach(m=>{const h=normalizeMarkerHexColor(m.getAttribute("data-marker-color"));m.classList.
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
document.querySelectorAll(".marker-edit-only").forEach(L=>L.classList.toggle("hidden",markerState.maskMode));
const l=get("marker-tool-mosaic"),c=get("marker-tool-crop");l&&l.classList.toggle("hidden",markerState.
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const d=get("marker-tool-draw"),m=get(
"marker-tool-auto");d&&(d.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
m&&(m.classList.toggle("hidden",!markerState.maskMode),m.classList.toggle("active",e==="auto")),d&&d.
classList.toggle("active",e==="draw");const h=get("marker-brush-control");h&&h.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const v=get("marker-modal-title"),w=get("marker-modal-de\
scription"),k=get("marker-save");v&&(v.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u4F5C\u6210":
"\u753B\u50CF\u7DE8\u96C6"),w&&(w.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\
\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":
"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
k&&(k.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const x=get("marker-canvas");x&&(x.style.pointerEvents=e==="crop"?"none":"auto");const C=get("marker\
-crop-canvas");C&&(C.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.cropRect||
markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}o(setMarkerMode,
"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}o(clearCropRect,
"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)return;const t=Math.
max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,y:0,w:t,h:n})}o(resetCropRectToFull,
"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.scale=Math.min(markerView.maxScale,
Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.scale<=markerView.minScale+1e-4){
markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marker-stage"),t=get("marker-viewport");
if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(1,e.clientHeight||0),a=Math.max(1,t.
offsetWidth||t.clientWidth||0),r=Math.max(1,t.offsetHeight||t.clientHeight||0);if(n<=1||i<=1||a<=1||
r<=1)return;const l=(n-a)/2,c=(i-r)/2,d=a*markerView.scale,m=r*markerView.scale,h=Math.min(n*.45,Math.
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=h-l-d,w=n-h-l,k=y-c-m,x=i-y-c,C=o((L,M,E)=>Number.
isFinite(L)?M>E?(M+E)/2:Math.min(E,Math.max(M,L)):0,"clampOffset");markerView.offsetX=C(markerView.offsetX,
v,w),markerView.offsetY=C(markerView.offsetY,k,x)}o(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
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
height);const n=o((l,c,d=null,m=!1)=>{if(!l)return;const h=Math.max(0,l.x),y=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);d&&(t.fillStyle=d,t.fillRect(h,y,v,w)),t.save(),m&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(h+.5,y+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),d=Math.max(1,i.w),m=Math.max(1,i.h);t.clearRect(l,c,d,m),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}o(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
o(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=o((a,r,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const d=normalizeAttachmentSource(
r),m=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),h=t.get(c);if(h===void 0){const w=e.
length;t.set(c,w),e.push({path:c,source:d,name:m});return}const y=e[h];if(!y)return;const v=normalizeAttachmentSource(
y.source);(v==="unknown"&&d!=="unknown"||v==="library"&&d==="upload")&&(y.source=d),!normalizeAttachmentDisplayName(
y.name)&&m&&(y.name=m)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const r=a.getAttribute("data-filename");n(r,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}o(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}o(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const r=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const d=isAudioPath(
c),m=isVideoPath(c);if(d&&!t.audio||m&&!t.video){d&&(n+=1),m&&(i+=1);return}r.push(c)}),r.length!==currentImageUrls.
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
{modal:"marker"},"","/edit-image"),r&&(r.onload=()=>{if(!get("marker-stage")||!l)return;const m=Math.
max(1,Math.floor(r.clientWidth)),h=Math.max(1,Math.floor(r.clientHeight));l.width=m,l.height=h,l.style.
width=`${m}px`,l.style.height=`${h}px`,l.style.left="0px",l.style.top="0px",c&&(c.width=m,c.height=h,
c.style.width=`${m}px`,c.style.height=`${h}px`,c.style.left="0px",c.style.top="0px"),markerState.naturalWidth=
r.naturalWidth||m,markerState.naturalHeight=r.naturalHeight||h;const y=l.getContext("2d");y&&y.clearRect(
0,0,l.width,l.height),prepareMarkerBaseCanvas(r,m,h),saveMarkerHistory(),markerState.mode==="crop"&&
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
m=String(d.facingMode||"").toLowerCase();m==="user"||m==="environment"?cameraCaptureFacingMode=m:cameraCaptureFacingMode=
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
const l=await new Promise((d,m)=>{t.toBlob(h=>{h?d(h):m(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
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
"gif"].includes(l),d=buildFileUrl(e),m=c?buildAttachmentPreviewUrl(e):d,h=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=h,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=r,y.dataset.defaultDisplayName=r,y.dataset.sendNameCustomized="";const v=escapeHtml(
r),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",k=c?`<img src="${m}" loading="lazy" decoding="async" class="up\
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
y)||r));const C=y.querySelector(".upload-send-name");C&&(C.onclick=()=>promptRowAttachmentName(y));const L=y.
querySelector(".upload-remove");L&&(L.onclick=()=>{uploadCancelTokens.add(h),browserFastLocalFiles.delete(
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
tton>':"";let m="";l?(m=URL.createObjectURL(e),c=`<img src="${m}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${r}">`):(m=URL.createObjectURL(e),c=
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
            `,m&&i.setAttribute("data-local-url",m);const h=i.querySelector(".upload-preview");h&&(h.
onclick=()=>{const k=i.getAttribute("data-filename"),x=k?buildFileUrl(k):i.getAttribute("data-local-\
url"),C=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||k||"";openFileViewer(x,C)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const k=i.getAttribute("data-local-url");k&&URL.revokeObjectURL(k);const x=i.
getAttribute("data-filename");x&&(currentImageUrls=currentImageUrls.filter(C=>C!==x)),setRowMarkerState(
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
v+l),k=e.slice(v,w);if(!await new Promise(C=>{const L=new XMLHttpRequest;L.open("POST","/upload/chun\
k",!0),L.setRequestHeader("X-CSRF-Token",csrfToken),L.upload.onprogress=E=>{if(E.lengthComputable&&t&&
t.bar){const D=v+E.loaded,N=Math.min(100,Math.floor(D/e.size*100));t.bar.style.width=`${N}%`,t.status&&
(t.status.textContent=`${N}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,N)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},L.onload=()=>{L.status>=200&&L.status<300?C(!0):C(!1)},L.
onerror=()=>C(!1);const M=new FormData;M.append("upload_id",r),M.append("index",String(y)),M.append(
"total",String(c)),M.append("chunk",k,e.name),L.send(M)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const d=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),m=await d.json();if(d.ok&&m&&m.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const k=t.row.querySelector(
"img.upload-preview");if(k){const x=m.filename.replace(/^\d+\//,"");k.src=buildAttachmentPreviewUrl(
x)}}const y=normalizeAttachmentPath(m.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||m.filename),setRowAttachmentSource(t.row,"upload"),y)){const w=isRowAttachmentNameCustomized(
t.row),k=defaultAttachmentDisplayName(y),x=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
k;t.row.dataset.defaultDisplayName=k,setRowAttachmentName(t.row,x)}return y&&setAttachmentSourceForPath(
y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(m.
filenames)&&m.filenames.length?m.filenames:[m.filename]).forEach(w=>addLibraryFileFromPath(w)),!0}const h=m&&
m.error?m.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),!1}catch{return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t,n={}){return new Promise(i=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(i);return}if(isUploadCancelled(t)){i(!1);return}let a=!1;window.ConnectionMonitor&&
(window.ConnectionMonitor.operationStarted(),a=!0);const r=o(()=>{a&&window.ConnectionMonitor&&(window.
ConnectionMonitor.operationEnded(),a=!1)},"finishUploadOp"),l=new XMLHttpRequest;l.open("POST",CHAT_CONFIG.
urls.upload,!0),l.setRequestHeader("X-CSRF-Token",csrfToken),l.upload.onprogress=d=>{if(d.lengthComputable&&
t&&t.bar){const m=Math.min(100,Math.floor(d.loaded/d.total*100));t.bar.style.width=`${m}%`,t.status&&
(t.status.textContent=`${m}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,m)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},l.onload=()=>{let d={};try{d=JSON.parse(l.responseText||"\
{}")}catch{}if(l.status>=200&&l.status<300&&d&&d.filename){if(isUploadCancelled(t)){t.row&&t.row.parentNode&&
t.row.remove(),r(),i(!1);return}if(t&&t.row){const y=t.row.getAttribute("data-local-url");y&&URL.revokeObjectURL(
y),t.row.removeAttribute("data-local-url");const v=t.row.querySelector("img.upload-preview");if(v){const w=d.
filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(w)}}const m=normalizeAttachmentPath(d.
filename);if(m&&currentImageUrls.push(m),t&&t.row&&(t.row.setAttribute("data-filename",m||d.filename),
setRowAttachmentSource(t.row,"upload"),m)){const y=isRowAttachmentNameCustomized(t.row),v=defaultAttachmentDisplayName(
m),w=y&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||v;t.row.dataset.defaultDisplayName=
v,setRowAttachmentName(t.row,w)}m&&setAttachmentSourceForPath(m,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(d.filenames)&&d.filenames.length?d.filenames:[d.filename]).
forEach(y=>addLibraryFileFromPath(y)),r(),i(!0)}else{const m=[502,503,504].includes(l.status)||l.status>=
520&&l.status<=527;if(n.cameraCapture&&m&&!d.error&&!isUploadCancelled(t)){t&&t.status&&(t.status.textContent=
"\u5206\u5272\u3057\u3066\u518D\u9001\u4E2D..."),r(),uploadFileChunked(e,t).then(i);return}const h=d&&
d.error?d.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),r(),i(!1)}},l.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),r(),i(!1)};const c=new FormData;c.append("file",e),l.send(c)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(m=>{n+=m.length});const i=new Float32Array(n);
let a=0;e.forEach(m=>{i.set(m,a),a+=m.length});const r=new ArrayBuffer(44+i.length*2),l=new DataView(
r),c=o((m,h)=>{for(let y=0;y<h.length;y++)l.setUint8(m+y,h.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let d=44;for(let m=0;m<i.length;m++){const h=Math.
max(-1,Math.min(1,i[m]));l.setInt16(d,h<0?h*32768:h*32767,!0),d+=2}return new Blob([l],{type:"audio/\
wav"})}o(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}o(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const r=e.row.getAttribute("data-local-url");r&&URL.revokeObjectURL(
r);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),d=escapeHtml(a),m=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${d}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',h=e.row.querySelector(".upload-preview");h&&(h.outerHTML=m);const y=e.row.querySelector(
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
l=markerView.scale,c={x:0,y:0},d={x:0,y:0},m=[],h=16,y="",v=null,w=null,k=null,x=null,C=null,L=!1,M=null;
const E=o(B=>{const K=e.getBoundingClientRect(),ce=(B.clientX-K.left)*(e.width/K.width),de=(B.clientY-
K.top)*(e.height/K.height);return{x:ce,y:de}},"getPoint"),D=o((B,K)=>({x:(B.x+K.x)/2,y:(B.y+K.y)/2}),
"getMid"),N=o((B,K)=>Math.hypot(B.x-K.x,B.y-K.y),"getDist");let G=!1;const j=o(()=>{w||(w=document.createElement(
"canvas"),k=w.getContext("2d")),x||(x=document.createElement("canvas"),C=x.getContext("2d")),(w.width!==
e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(x.width!==e.width||x.height!==e.
height)&&(x.width=e.width,x.height=e.height)},"ensureDrawBuffers"),J=o(()=>{if(!t||!w||!x)return;const B=markerState.
maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));t.clearRect(
0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=B,t.drawImage(x,0,0),t.restore()},"r\
enderDrawPreview"),oe=o(()=>{C&&(C.strokeStyle=y,C.fillStyle=y,C.lineWidth=h,C.lineCap="round",C.lineJoin=
"round")},"applyMarkerBrush"),me=o(B=>{if(!B)return!1;if(m.length===0)return m.push(B),!0;const K=m[m.
length-1],ce=B.x-K.x,de=B.y-K.y,te=Math.hypot(ce,de),re=Math.max(.35,h*.04);if(te<re)return!1;const Q=Math.
max(1,h*.25),fe=Math.max(1,Math.ceil(te/Q));for(let ue=1;ue<=fe;ue++){const Ge=ue/fe;m.push({x:K.x+ce*
Ge,y:K.y+de*Ge})}return!0},"appendStrokePoint"),ie=o(()=>{if(C&&(C.clearRect(0,0,x.width,x.height),m.
length!==0)){if(oe(),m.length===1){const B=m[0];C.beginPath(),C.arc(B.x,B.y,h/2,0,Math.PI*2),C.fill();
return}if(C.beginPath(),C.moveTo(m[0].x,m[0].y),m.length===2)C.lineTo(m[1].x,m[1].y);else{for(let ce=1;ce<
m.length-2;ce++){const de=m[ce],te=m[ce+1],re=D(de,te);C.quadraticCurveTo(de.x,de.y,re.x,re.y)}const B=m[m.
length-2],K=m[m.length-1];C.quadraticCurveTo(B.x,B.y,K.x,K.y)}C.stroke()}},"renderStrokeLayer"),ye=o(
(B,K)=>{if(!B||!K)return null;const ce=Math.min(B.x,K.x),de=Math.min(B.y,K.y),te=Math.abs(B.x-K.x),re=Math.
abs(B.y-K.y);return{x:ce,y:de,w:te,h:re}},"normalizeMosaicRect"),$=o(B=>{const K=n?Number(n.value||16):
16,ce=Math.max(6,Math.floor(K)),de=Math.floor(ce/2);return{x:B.x-de,y:B.y-de,w:ce,h:ce}},"buildMosai\
cRectFromPoint"),R=o(()=>{const B=document.createElement("canvas");B.width=e.width,B.height=e.height;
const K=B.getContext("2d");if(!K)return null;markerState.baseCanvas&&K.drawImage(markerState.baseCanvas,
0,0),K.drawImage(e,0,0);try{return K.getImageData(0,0,e.width,e.height)}catch{return null}},"getMosa\
icSourceImageData"),U=o(B=>{if(!t||!B)return!1;const K=R();if(!K)return!1;const ce=n?Number(n.value||
16):16,de=Math.max(4,Math.floor(ce/2)),te=Math.max(0,Math.floor(B.x)),re=Math.max(0,Math.floor(B.y)),
Q=Math.min(e.width,Math.ceil(B.x+B.w)),fe=Math.min(e.height,Math.ceil(B.y+B.h));if(Q<=te||fe<=re)return!1;
for(let ue=re;ue<fe;ue+=de)for(let Ge=te;Ge<Q;Ge+=de){const Oe=Math.min(de,Q-Ge),et=Math.min(de,fe-ue),
ge=Math.min(e.width-1,Math.max(0,Ge+Math.floor(Oe/2))),Se=(Math.min(e.height-1,Math.max(0,ue+Math.floor(
et/2)))*e.width+ge)*4,De=K.data[Se],he=K.data[Se+1],dt=K.data[Se+2];t.fillStyle=`rgb(${De},${he},${dt}\
)`,t.fillRect(Ge,ue,Oe,et)}return!0},"applyMosaicRect"),P=o(B=>{if(!t)return;if(i.set(B.pointerId,{x:B.
clientX,y:B.clientY}),i.size>=2){const ce=Array.from(i.values()),de=ce[0],te=ce[1];a=!0,G=!1,m=[],L=
!1,M=null,v=null,markerState.mosaicPreviewRect=null,r=N(de,te)||1,l=markerView.scale,c={x:markerView.
offsetX,y:markerView.offsetY},d=D(de,te),renderCropOverlay(),e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault();return}if(a||markerState.mode==="crop")return;G=!0;const K=E(B);if(markerState.
mode==="auto"&&markerState.maskMode)v=K;else if(markerState.mode==="mosaic")L=!0,M=K,markerState.mosaicPreviewRect=
$(K),renderCropOverlay();else{if(j(),!k||!C)return;k.clearRect(0,0,w.width,w.height),k.drawImage(e,0,
0),C.clearRect(0,0,x.width,x.height),h=n?Number(n.value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(
markerState.colorHex),m=[],me(K),ie(),markerState.hasStroke=!0,J()}e.setPointerCapture&&e.setPointerCapture(
B.pointerId),B.preventDefault()},"start"),q=o(B=>{if(i.has(B.pointerId)&&i.set(B.pointerId,{x:B.clientX,
y:B.clientY}),a&&i.size>=2){const ce=Array.from(i.values()),de=ce[0],te=ce[1],re=D(de,te),Q=N(de,te)||
1,fe=l*(Q/r);markerView.scale=Math.min(markerView.maxScale,Math.max(markerView.minScale,fe)),markerView.
offsetX=c.x+(re.x-d.x),markerView.offsetY=c.y+(re.y-d.y),applyMarkerTransform(),B.preventDefault();return}
if(!G||!t)return;if(markerState.mode==="auto"&&markerState.maskMode){B.preventDefault();return}const K=E(
B);if(markerState.mode==="mosaic"){if(!L||!M)return;markerState.mosaicPreviewRect=ye(M,K)||$(K),renderCropOverlay()}else
me(K)&&(ie(),J());B.preventDefault()},"move"),ee=o(B=>{const K=G;if(i.delete(B.pointerId),i.size<2&&
(a=!1),i.size===0){G=!1;const ce=K&&B.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.
maskMode&&v?fillMaskSimilarRegion(v):!1;if(v=null,K&&t&&markerState.mode==="draw"&&m.length>0&&(ie(),
J()),K&&markerState.mode==="mosaic"&&M){const de=E(B);let te=ye(M,de);(!te||te.w<2||te.h<2)&&(te=$(M)),
U(te)&&(markerState.hasStroke=!0,markerState.mosaicRects.push(te))}m=[],L=!1,M=null,markerState.mosaicPreviewRect=
null,renderCropOverlay(),K&&(markerState.mode!=="auto"||ce)&&saveMarkerHistory()}e.releasePointerCapture&&
e.releasePointerCapture(B.pointerId),B.preventDefault()},"end");e.addEventListener("pointerdown",P),
e.addEventListener("pointermove",q),e.addEventListener("pointerup",ee),e.addEventListener("pointerca\
ncel",ee)}o(initMarkerCanvas,"initMarkerCanvas");function fillMaskSimilarRegion(e){var G;const t=get(
"marker-canvas"),n=markerState.baseImageData;if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,r=Math.floor(e.x),l=Math.floor(e.y);if(r<0||l<0||r>=i||l>=
a)return!1;const c=n.data,d=l*i+r,m=d*4;if(c[m+3]<16)return!1;const h=c[m],y=c[m+1],v=c[m+2],w=Number(
((G=get("marker-auto-tolerance"))==null?void 0:G.value)||32),k=new Uint8Array(i*a),x=[d],C=[],L=25e4,
M=o(j=>{const J=j*4;return c[J+3]>=16&&Math.max(Math.abs(c[J]-h),Math.abs(c[J+1]-y),Math.abs(c[J+2]-
v))<=w},"matches");for(k[d]=1;x.length;){const j=x.pop();if(k[j]===2||!M(j))continue;const J=Math.floor(
j/i);let oe=j%i,me=oe;for(;oe>0&&k[J*i+oe-1]!==2&&M(J*i+oe-1);)oe--;for(;me+1<i&&k[J*i+me+1]!==2&&M(
J*i+me+1);)me++;for(let ie=oe;ie<=me;ie++)k[J*i+ie]=2;if(C.push(J,oe,me+1),C.length/3>L)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const ie of[J-1,J+1]){if(ie<0||ie>=a)continue;let ye=oe;for(;ye<=me;){const $=ie*
i+ye;if(k[$]===0&&M($))for(k[$]=1,x.push($);ye+1<=me&&k[ie*i+ye+1]===0&&M(ie*i+ye+1);)ye++,k[ie*i+ye]=
1;ye++}}}if(!C.length)return!1;const E=t.getContext("2d");if(!E)return!1;const D=E.getImageData(0,0,
i,a),N=D.data;for(let j=0;j<C.length;j+=3){const J=C[j]*i+C[j+1],oe=C[j]*i+C[j+2];for(let me=J;me<oe;me++){
const ie=me*4;N[ie]=N[ie+1]=N[ie+2]=N[ie+3]=255}}return E.putImageData(D,0,0),markerState.hasStroke=
!0,!0}o(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,
d=0,m=markerView.scale,h={x:0,y:0},y={x:0,y:0};const v=8,w=14,k=o((j,J,oe)=>Math.min(oe,Math.max(J,j)),
"clamp"),x=o(j=>{const J=e.getBoundingClientRect(),oe=(j.clientX-J.left)*(e.width/J.width),me=(j.clientY-
J.top)*(e.height/J.height);return{x:oe,y:me}},"getPoint"),C=o((j,J)=>({x:(j.x+J.x)/2,y:(j.y+J.y)/2}),
"getMid"),L=o((j,J)=>Math.hypot(j.x-J.x,j.y-J.y),"getDist"),M=o(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),E=o((j,J)=>{if(!J)return"move";const oe=J.x,me=J.y,ie=J.x+J.
w,ye=J.y+J.h,$=Math.abs(j.x-oe)<=w,R=Math.abs(j.x-ie)<=w,U=Math.abs(j.y-me)<=w,P=Math.abs(j.y-ye)<=w;
if($&&U)return"nw";if(R&&U)return"ne";if($&&P)return"sw";if(R&&P)return"se";if(U)return"n";if(P)return"\
s";if($)return"w";if(R)return"e";if(j.x>oe+w&&j.x<ie-w&&j.y>me+w&&j.y<ye-w)return"move";const ee=j.x<
oe?"left":j.x>ie?"right":null,B=j.y<me?"top":j.y>ye?"bottom":null;if(ee&&B){if(ee==="left"&&B==="top")
return"nw";if(ee==="right"&&B==="top")return"ne";if(ee==="left"&&B==="bottom")return"sw";if(ee==="ri\
ght"&&B==="bottom")return"se"}return ee?ee==="left"?"w":"e":B?B==="top"?"n":"s":"move"},"hitTest"),D=o(
j=>{if(markerState.mode!=="crop")return;if(n.set(j.pointerId,{x:j.clientX,y:j.clientY}),n.size>=2){const me=Array.
from(n.values()),ie=me[0],ye=me[1];c=!0,i=!1,d=L(ie,ye)||1,m=markerView.scale,h={x:markerView.offsetX,
y:markerView.offsetY},y=C(ie,ye),e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault();
return}if(c)return;i=!0;const J=x(j),oe=M();r=E(J,oe),a=J,l=oe?{x:oe.x,y:oe.y,w:oe.w,h:oe.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault()},"start"),N=o(j=>{if(markerState.
mode!=="crop")return;if(n.has(j.pointerId)&&n.set(j.pointerId,{x:j.clientX,y:j.clientY}),c&&n.size>=
2){const ee=Array.from(n.values()),B=ee[0],K=ee[1],ce=C(B,K),de=L(B,K)||1,te=m*(de/d);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,te)),markerView.offsetX=h.x+(ce.x-y.x),markerView.
offsetY=h.y+(ce.y-y.y),applyMarkerTransform(),renderCropOverlay(),j.preventDefault();return}if(!i||!a||
!l)return;const J=x(j),oe=e.width,me=e.height,ie={x:l.x,y:l.y,w:l.w,h:l.h},ye=l.x+l.w,$=l.y+l.h,R=o(
()=>{const ee=k(J.x,0,ye-v);ie.x=ee,ie.w=ye-ee},"applyW"),U=o(()=>{ie.w=k(J.x-l.x,v,oe-l.x)},"applyE"),
P=o(()=>{const ee=k(J.y,0,$-v);ie.y=ee,ie.h=$-ee},"applyN"),q=o(()=>{ie.h=k(J.y-l.y,v,me-l.y)},"appl\
yS");switch(r){case"move":{const ee=J.x-a.x,B=J.y-a.y;ie.x=k(l.x+ee,0,oe-l.w),ie.y=k(l.y+B,0,me-l.h);
break}case"w":R();break;case"e":U();break;case"n":P();break;case"s":q();break;case"nw":P(),R();break;case"\
ne":P(),U();break;case"sw":q(),R();break;case"se":q(),U();break;default:break}ie.x=k(ie.x,0,oe-ie.w),
ie.y=k(ie.y,0,me-ie.h),markerState.cropRect=ie,renderCropOverlay(),j.preventDefault()},"move"),G=o(j=>{
n.delete(j.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),e.
releasePointerCapture&&e.releasePointerCapture(j.pointerId),j.preventDefault()},"end");e.addEventListener(
"pointerdown",D),e.addEventListener("pointermove",N),e.addEventListener("pointerup",G),e.addEventListener(
"pointercancel",G),e.addEventListener("pointerleave",G)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const r=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const C=document.createElement("canvas");C.width=r,C.height=l;const L=C.getContext(
"2d");if(!L)return;L.fillStyle="#000000",L.fillRect(0,0,r,l),L.drawImage(n,0,0,r,l);const M=await new Promise(
N=>C.toBlob(N,"image/png"));if(!M){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const E=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([M],`${E}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,r,l),c.drawImage(n,0,0,r,l),markerState.cropRect){const C=r/n.width,L=l/n.height,M=Math.max(0,Math.
floor(markerState.cropRect.x*C)),E=Math.max(0,Math.floor(markerState.cropRect.y*L)),D=Math.min(r,Math.
max(1,Math.floor(markerState.cropRect.w*C))),N=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*L))),G=document.createElement("canvas");G.width=D,G.height=N;const j=G.getContext("2d");j&&(j.drawImage(
a,M,E,D,N,0,0,D,N),a=G)}const d=await new Promise(C=>a.toBlob(C,"image/png",.92));if(!d){showToast("\
\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const h=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([d],`${h}_marked.png`,
{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".upload-status"),bar:e.
querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\u4E2D..."),
updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),k=getRowAttachmentSource(e);w&&!e.dataset.
originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=k,setAttachmentSourceForPath(
w,k)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(C=>C!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}o(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,r=null,l=null,c=null,d=[],m=null;const h=o(()=>{m&&clearTimeout(
m);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(
()=>{h(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();
if(!v||!v.length)return y();r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=r.createMediaStreamSource(new MediaStream(v)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=
k=>{const x=k.inputBuffer.getChannelData(0);d.push(new Float32Array(x))},c.connect(l),l.connect(r.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(m=setTimeout(()=>{const k=(e.
name||"video").replace(/\.[^/.]+$/,""),x=encodeWav(d,r.sampleRate),C=new File([x],`${k}.audio.wav`,{
type:"audio/wav"});h(),n(C)},w+250)),await i.play(),i.onended=()=>{const k=(e.name||"video").replace(
/\.[^/.]+$/,""),x=encodeWav(d,r.sampleRate),C=new File([x],`${k}.audio.wav`,{type:"audio/wav"});h(),
n(C)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
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
perFilePct[w]=0;let x=v;if(r&&isVideoFile(v)){const L=await extractAudioFromVideo(v,k);L?(x=L,updateUploadRowFile(
k,L),k&&k.status&&(k.status.textContent="\u97F3\u58F0\u306E\u307F")):(k&&k.status&&(k.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{x=await applyImageCompression(v,
readComposerCompression()),x!==v&&updateUploadRowFile(k,x)}catch{}if(isUploadCancelled(k))return!1;if(browserFastModeEnabled){
const L=Array.from(browserFastLocalFiles.values()).reduce((M,E)=>M+Number(E.file&&E.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||L+x.size>BROWSER_FAST_MAX_BYTES?(k&&k.status&&
(k.status.textContent="\u4E0A\u9650\u8D85\u904E"),k&&k.row&&k.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(k.uploadId,{file:x,rowObj:k}),
k.status&&(k.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
k.bar&&(k.bar.style.width="100%"),k.row&&(k.row.dataset.browserFastLocal="1"),!0)}const C=await uploadFileWithProgress(
x,k,t);if(!C&&t.cameraCapture&&!isUploadCancelled(k)){const L=document.createElement("button");L.className=
"text-xs border border-gray-700 rounded px-2 py-1",L.textContent="\u518D\u8A66\u884C",L.onclick=()=>{
if(isUploadCancelled(k))return;L.disabled=!0;const M=k.row.getAttribute("data-local-url");M&&URL.revokeObjectURL(
M),k.row.remove(),handleFiles([x],{...t,openModal:!0,skipCompression:!0})},k.row.appendChild(L)}return C}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),d=new Array(i.length).fill(!1);let m=0;const h=Math.
min(UPLOAD_CONCURRENCY,i.length),y=Array.from({length:h}).map(async()=>{for(;;){const v=m++;if(v>=i.
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
`),y.open=!0)}const m=i.map((y,v)=>normalizeCanvasBlock(y,v)),h=selectCanvasPreviewBlock(m,t);return{
renderText:a.join(`
`),blocks:m,primaryBlock:h?h.block:null,primaryIndex:h?h.index:-1,rawText:t}}o(parseCanvasMarkdown,"\
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
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const m=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",m),d.setAttribute("aria-pressed",m?"true":"fa\
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
null?i.code:""),d=r?isCanvasHtmlPreviewCandidate(l,c):!1,m=r?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",h=r?d?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=h),e.status&&(e.status.textContent=
m),e.previewLang&&(e.previewLang.textContent=r?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
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
(c,d)=>{const m=decodeCanvasPreviewButtonCode(c);if(!m)return;const h=normalizeCanvasBlock({lang:m.lang,
code:m.code,open:!1},d);a.push(h),r===-1&&m.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
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
"`","~"])for(let c=3;c<=10;c++){const d=l.repeat(c);for(const m of r){const h=`${d}${m}
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
let r=t.replace(a,(v,w,k,x)=>{i.push(x);const C=String(k||"").trim();try{const L=JSON.parse(C);n.push(
{code:L&&L.code!=null?String(L.code):"",output:L&&L.output!=null?String(L.output):"",images:L&&Array.
isArray(L.images)?L.images.filter(M=>typeof M=="string"):[]})}catch{n.push({code:C,output:""})}return`\

`});const l=/!\[Agentic View\]\(([^)]+)\)/g;if(!n.length&&/!\[Agentic View\]\(/.test(t)&&(r=t.replace(
/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,(v,w,k,x)=>(i.push(
x),n.push({code:w,output:k||"",images:[]}),`
`))),n.length){const v=new Set(n.flatMap(w=>w.images||[]));for(const w of t.matchAll(l)){if(v.has(w[1]))
continue;let k=i.length-1;for(;k>0&&i[k]>w.index;)k--;const x=n[k];x.images=Array.from(new Set([...x.
images||[],w[1]]))}r=r.replace(l,"")}n.forEach(v=>{v.code&&(r=stripExactFencedBlock(r,"python",v.code),
r=stripExactFencedBlock(r,"py",v.code)),r=stripVisiblePythonOutputBlock(r,v.output)});const c=new Set(
n.flatMap(v=>v.images||[]));r=r.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
(v,w,k,x)=>!w&&c.has(x)&&/\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(x)?`![${k}](${x})`:v);const d=n.
slice().reverse().find(v=>v.images&&v.images.length),m=r.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,
""),h=new Set(Array.from(m.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),v=>v[1])),y=(d?d.images:[]).filter(
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
parse(l))}</div>`:"",d=escapeHtml(a),m=encodeURIComponent(i).replace(/'/g,"%27"),h=encodeURIComponent(
a).replace(/'/g,"%27"),y=hashString(`pyexec-detail
${i}
${a}
${t}`),v=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",w=`<button class="download-btn" data-\
code="${m}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,k=`<button class="coding-target-btn" data-code="${m}" data-code-key="${y}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${y}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(v)}</span><div class="code-actions">${k}${w}<button class="copy-btn" data-copy="code" \
data-code="${m}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
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
ey]").forEach(h=>{const y=h.getAttribute("data-stream-math-key");y&&d.set(y,h)});const m=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(h=>{const y=d.get(h.getAttribute(
"data-stream-math-key"));y?h.replaceWith(y):m.push(h)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(
e)),e.replaceChildren(c.content),wrapRenderedSvgBoxes(e),queueHighlight(e,r.renderText),queueIncrementalMathTypeset(
m);return}const l=collectSvgCodeRenders(e);e.innerHTML=sanitizeMarkdownHtml(r.renderText),restoreSvgCodeRenders(
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
400"></i> Search complete'),t}o(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,r,l=null,c=!0,d=null,m=null,h=null,y=null,v=null,w=null,k=null,x=null,C=!0,L=null,M=null,E=null,D=null,N=null){
const G=t==="user",j=G?"bg-blue-600":"bg-gray-700",J=G?"justify-end":"justify-start";messageStore[e]=
n;const oe=!G&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},me=G?n:oe.text;let ie=m;
if(ie==null){const ge=h!=null?Number(h):0,pe=y!=null?Number(y):0;(h!=null||y!=null)&&(ie=ge+pe)}const ye=costCacheUsage(
a);messageMeta[e]={tokens_in:h,tokens_out:y,tokens_total:ie,tokens_content:w,tokens_thought:k,cache_read_tokens:ye?
ye.read:null,cache_write_tokens:ye?ye.write:null,cache_write_ttls:ye?ye.ttls:null,is_encrypted:v,role:t,
model:r,parent_id:L,quote_text:d,image_url:i,image_mask:D,skill_name:M,batch_job:E,processing_time:N,
python_executions:G?[]:oe.executions||[]};let $="";d&&($=`<div class="mb-2 p-2 bg-black/20 rounded b\
order-l-4 border-blue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-l\
eft mr-1 opacity-50"></i>${escapeHtml(d)}</div>`);const R=M?`<div class="skill-notice visible"><i cl\
ass="fas fa-wand-magic-sparkles"></i><span>Skill: <span class="font-bold">${escapeHtml(M)}</span></s\
pan></div>`:"";let U="";if(a&&!G){let ge="";try{const pe=JSON.parse(a);ge=pe.text||"",pe.search_used===
!0&&(U=SEARCH_COMPLETE_HTML)}catch{ge=a}ge&&(U+=`<div class="thought-container"><div class="thought-\
header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process\
</div><div class="thought-content collapsed">${escapeHtml(ge)}</div></div>`)}let P="";const q=new Set;
if(!G){const ge=String(me||"").replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,""),pe=Array.
from(ge.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),Se=>Se[1]);pe.push(...(oe.executions||[]).flatMap(Se=>Se.
images||[])),pe.forEach(Se=>{if(/^https?:\/\//i.test(Se)&&!Se.startsWith(FILE_BASE_URL))try{if(new URL(
Se).origin!==window.location.origin)return}catch{return}const De=normalizeAttachmentPath(Se);De&&q.add(
De)})}if(i)try{const ge=JSON.parse(i);if(ge.length){const pe=[];if(ge.forEach(Se=>{let De=Se,he="unk\
nown";if(De&&typeof De=="object"&&(he=normalizeAttachmentSource(De.source),De=De.filepath||De.path||
De.url||De.file||""),De=normalizeAttachmentPath(De)||De,!De||!G&&q.has(De))return;setAttachmentSourceForPath(
De,he);const dt=De.replace(/^\d+\//,""),Ne=buildFileUrl(dt),gt=buildAttachmentPreviewUrl(dt),ze=De.split(
"/").pop(),wt=ze.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(wt)?pe.push(
buildChatImageHtml(gt,{viewerSrc:Ne,alt:ze,title:ze,filename:ze})):pe.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${Ne}')" title="${ze}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${ze}</span></div>`)}),pe.length>
0){let Se="grid-multi";pe.length===1?Se="grid-1":pe.length===2?Se="grid-2":pe.length===3?Se="grid-3":
pe.length===4&&(Se="grid-4"),P=`<div class="image-grid ${Se}">${pe.join("")}</div>`}}}catch{}let ee="";
if(G&&D)try{const ge=normalizeAttachmentPath(D)||String(D),pe=ge.replace(/^\d+\//,""),Se=buildFileUrl(
pe),De=buildAttachmentPreviewUrl(pe),he=ge.split("/").pop();ee=`<div class="mt-2 max-w-xs rounded-lg\
 border border-amber-300/40 bg-black/10 p-2"><div class="mb-1 text-[10px] font-semibold text-amber-1\
00"><i class="fas fa-mask mr-1" aria-hidden="true"></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(
De,{viewerSrc:Se,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:he})}</div>`}catch{}const B=G?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,K=`<div class="msg-controls absolute -top-5 right-\
0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', t\
his)"><i class="fas fa-copy"></i></button>${G?`<button class="ctrl-btn edit-btn" data-id="${e}"><i c\
lass="fas fa-pen"></i></button>`:""}${B}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\u65B0\
\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,ce=[];!G&&r&&ce.push(escapeHtml(r));const de=[];
if(h!=null&&de.push(`In ${h}`),y!=null){let ge=`Out ${y}`;k!=null&&Number(k)>0&&(ge+=` (Thought ${k}\
)`),de.push(ge)}if(de.length||m!=null){const ge=de.length?de.join(" / "):`${m} tokens`;ce.push(`<but\
ton class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${e}\
')">${ge}</button>`)}if(v!=null){const ge=v?"fa-lock":"fa-lock-open",pe=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\
\uFF09":"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":v?"En\
crypted":"Plain",Se=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:t\
ext-cyan-200":"text-slate-300/80 hover:text-white";ce.push(`<button class="${Se}" title="${pe}" oncl\
ick="openEncryptionSettings('${e}')"><i class="fas ${ge}"></i></button>`)}if(!G&&oe.executions&&oe.executions.
length){const ge=oe.executions.length,pe=ge>1?`Python \xD7${ge}`:"Python";ce.push(`<button type="but\
ton" class="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label\
="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${pe}</span\
></button>`)}const te=ce.length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono\
 message-footer-meta">${ce.join(" \u2022 ")}</div>`:"",re=G?"":buildProcessingTimeHtml(e,N);let Q;const fe=!G&&
E?(()=>{const ge=String(E.state||"").toUpperCase(),pe=E.status_text||(ge==="JOB_STATE_SUCCEEDED"?"Ba\
tch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":ge==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":
"Batch API\u3067\u51E6\u7406\u4E2D\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-l\
g border ${ge==="JOB_STATE_FAILED"||ge==="JOB_STATE_CANCELLED"||ge==="JOB_STATE_EXPIRED"?"border-red\
-400/40 bg-red-950/30 text-red-100":ge==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950\
/30 text-emerald-100":"border-violet-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><d\
iv class="font-semibold"><i class="fas fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-\
90">${escapeHtml(pe)}</div></div>`})():"";G?Q=`<div class="content-area whitespace-pre-wrap font-san\
s text-sm break-words">${escapeHtml(n||"")}</div>`:(Q=fe+(me&&String(me).trim()?buildAiMarkdownHtml(
me):E?'<div class="content-area prose prose-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</d\
iv>':buildAiMarkdownHtml(me)),Q.includes("content-area")||(Q=Q.replace("prose ","content-area prose ")));
let ue="";if(l){const ge=l.siblings[l.current-2],pe=l.siblings[l.current];ue=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ge}\
)" ${ge?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${pe}\
)" ${pe?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const Ge=c?"fade-in":"",Oe=document.createElement("div");Oe.className=`flex ${J} m\
b-4 ${Ge} relative message-group group`,Oe.id=`msg-${e}`,Oe.innerHTML=`<div class="message-bubble ${j}\
 text-white p-4 rounded-2xl shadow-md relative">${K}${$}${R}${U}${Q}${P}${ee}${ue}${te}${re}</div>`;
const et=x||get("chat-container");return et&&(et.appendChild(Oe),C&&scrollToBottom(),G||(Oe.isConnected&&
queueMessageDecorations(Oe,me),syncCodingTargetButtons(Oe),syncCodingModeUi(codingModeEnabled,{persist:!1}))),
Oe}o(renderMessage,"renderMessage");const PROCESSING_PHASE_LABELS={send:"\u9001\u4FE1\u301C\u53D7\u4ED8",
queue:"\u5F85\u3061\u884C\u5217",provider:"\u521D\u56DE\u5FDC\u7B54\u307E\u3067",generation:"\u751F\u6210",
receive:"\u53D7\u4FE1\u30FB\u8868\u793A"};function formatProcessingDuration(e){const t=Number(e);if(!Number.
isFinite(t)||t<0)return"-";if(t<6e4)return`${(t/1e3).toFixed(2)}\u79D2`;const n=t/1e3,i=Math.floor(n/
60);return`${i}\u5206${(n-i*60).toFixed(1)}\u79D2`}o(formatProcessingDuration,"formatProcessingDurat\
ion");function buildProcessingTimeHtml(e,t){if(!t||t.total_ms===null||t.total_ms===void 0)return"";const n=formatProcessingDuration(
t.total_ms);return`<div class="mt-1 text-right font-mono text-[10px] text-slate-400/80 processing-ti\
me-row"><button type="button" class="underline decoration-dotted hover:text-white processing-time-bt\
n" onclick="openProcessingTimeDetail('${e}')" title="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33" aria-label="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33">\u51E6\u7406\u6642\u9593 ${escapeHtml(
n)}</button></div>`}o(buildProcessingTimeHtml,"buildProcessingTimeHtml");function openProcessingTimeDetail(e){
const t=messageMeta[e],n=t&&t.processing_time;if(!n||n.total_ms===null||n.total_ms===void 0||!get("p\
rocessing-time-modal"))return;get("processing-time-total").textContent=formatProcessingDuration(n.total_ms);
const a=get("processing-time-first-row");if(a){const d=n.first_response_ms!==null&&n.first_response_ms!==
void 0;a.classList.toggle("hidden",!d),d&&get("processing-time-first")&&(get("processing-time-first").
textContent=formatProcessingDuration(n.first_response_ms))}const r=get("processing-time-phases");if(r){
r.replaceChildren();const d=Array.isArray(n.phases)?n.phases:[];for(const m of d){if(!m||m.ms===null||
m.ms===void 0)continue;const h=PROCESSING_PHASE_LABELS[m.key]||m.key,y=document.createElement("div");
y.className="flex justify-between gap-4";const v=document.createElement("span");v.textContent=h;const w=document.
createElement("span");w.className="font-mono",w.textContent=formatProcessingDuration(m.ms),y.appendChild(
v),y.appendChild(w),r.appendChild(y)}r.classList.toggle("hidden",r.childElementCount===0)}const l=get(
"processing-time-note");l&&(n.source==="client"?l.textContent="\u56DE\u7B54\u5B8C\u4E86\u6642\u306B\u7AEF\u672B\u3067\u8A18\u9332\u3057\u305F\u6642\u9593\u3067\u3059\u3002\u8A73\u7D30\u306A\u5185\u8A33\u306F\u5C65\u6B74\u306E\u518D\u8AAD\u307F\u8FBC\u307F\u5F8C\u306B\u8868\u793A\u3055\
\u308C\u307E\u3059\u3002":n.phases&&n.phases.length?l.textContent="\u8A18\u9332\u3067\u304D\u305F\u51E6\u7406\u533A\u9593\u306E\u5185\u8A33\u3067\u3059\u3002":
l.textContent="\u3053\u306E\u56DE\u7B54\u3067\u306F\u5408\u8A08\u6642\u9593\u3060\u3051\u3092\u8A18\u9332\u3057\u3066\u3044\u307E\u3059\u3002");
const c=get("processing-time-title");c&&(c.textContent=t&&t.model?`${t.model} (${t.role||"assistant"}\
)`:"\u56DE\u7B54\u51E6\u7406\u6642\u9593"),showModal("processing-time-modal")}o(openProcessingTimeDetail,
"openProcessingTimeDetail");function closeProcessingTimeDetail(){hideModal("processing-time-modal")}
o(closeProcessingTimeDetail,"closeProcessingTimeDetail");function showTokenDetailModal(e=null,t=!1){
if(location.pathname!=="/token-details"){const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==
null&&(n.messageId=e),history.pushState(n,"","/token-details")}showModal("token-detail-modal")}o(showTokenDetailModal,
"showTokenDetailModal");function openTokenDetail(e){var L;const t=messageMeta[e];if(!t||!get("token-\
detail-modal"))return;get("token-detail-usage").classList.remove("hidden"),get("token-detail-estimat\
e").classList.add("hidden"),cancelPromptTokenDetail();const i=t.tokens_total!==null&&t.tokens_total!==
void 0?t.tokens_total:"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",r=t.tokens_out!==
null&&t.tokens_out!==void 0?t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.
tokens_content:"-",c=t.tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===
null||t.is_encrypted===void 0?"-":t.is_encrypted?"Encrypted":"Plain",m=t.cache_read_tokens!==null&&t.
cache_read_tokens!==void 0?t.cache_read_tokens:null,h=t.cache_write_tokens!==null&&t.cache_write_tokens!==
void 0?t.cache_write_tokens:null,y=m!==null||h!==null;get("token-detail-total").innerText=i,get("tok\
en-detail-in").innerText=a,get("token-detail-out").innerText=r,get("token-detail-content").innerText=
l,get("token-detail-thought").innerText=c,get("token-detail-encrypted").innerText=d;const v=get("tok\
en-detail-cache-read-row"),w=get("token-detail-cache-write-row"),k=get("token-detail-cache-write-ttl");
if(v&&v.classList.toggle("hidden",!y),w&&w.classList.toggle("hidden",!y),get("token-detail-cache-rea\
d")&&(get("token-detail-cache-read").innerText=m===null?"-":m),get("token-detail-cache-write")&&(get(
"token-detail-cache-write").innerText=h===null?"-":h),k){const M=["5m","1h","30m","12h"].map(E=>{var N;
const D=Number(((N=t.cache_write_ttls)==null?void 0:N[E])||0);return D>0?`${E}: ${D}`:""}).filter(Boolean);
k.textContent=M.length?`TTL ${M.join(" \xB7 ")}`:"",k.classList.toggle("hidden",!y||!M.length)}const x=get(
"token-detail-cost");if(x&&(x.replaceChildren(),x.classList.toggle("hidden",t.role!=="total"),t.role===
"total")){for(const M of[costEstimateLabel(t.cost_estimate),`\u96C6\u8A08\u3057\u305FAPI\u547C\u3073\u51FA\u3057: ${((L=
t.cost_estimate)==null?void 0:L.requests)||0}\u56DE`,COST_ESTIMATE_NOTE,`\u516C\u5F0F\u6599\u91D1\u78BA\u8A8D\u65E5: ${(costPricing==
null?void 0:costPricing.checked_at)||"\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"}`]){
const E=document.createElement("p");E.textContent=M,x.appendChild(E)}for(const[M,E]of Object.entries(
(costPricing==null?void 0:costPricing.sources)||{})){const D=document.createElement("a");D.href=E,D.
textContent=`${M} \u516C\u5F0F\u6599\u91D1 `,D.target="_blank",D.rel="noopener noreferrer",D.className=
"underline",x.appendChild(D)}}const C=t.model?`${t.model} (${t.role})`:`${t.role}`;get("token-detail\
-title").innerText=C,showTokenDetailModal(e)}o(openTokenDetail,"openTokenDetail");function closeTokenDetail(e=!1){
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
ck",m),i&&i.removeEventListener("click",h),a&&a.removeEventListener("click",h),e.removeEventListener(
"click",y,!0)}o(c,"cleanup");function d(v){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(v)}o(d,"finalize");
function m(){d(!0)}o(m,"onOk");function h(){d(!1)}o(h,"onCancel");function y(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),h())}o(y,"onOverlay"),n&&n.addEventListener("click",m),i&&i.addEventListener(
"click",h),a&&a.addEventListener("click",h),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${r}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,m=e||get("\
chat-container");if(m){if(typeof m.insertAdjacentHTML=="function")m.insertAdjacentHTML("beforeend",d);else{
const h=document.createElement("div");h.innerHTML=d;const y=h.firstElementChild;y&&m.appendChild(y)}
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
""?String(currentThreadId):null;if(!l||r&&l!==r)return!1;for(let m=0;m<=n;m++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==l?!1:(await loadMessages(l,{preserveDraft:!0,silent:!0}),
!0)}catch{m<n&&i>0&&await new Promise(y=>setTimeout(y,i))}return a&&(currentThreadId!=null&&currentThreadId!==
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
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},m=parseSlashToggleArgument(
a),h=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(d,r)){const y=d[r],v=l?Array.from(l.options).
find(w=>w.value===y&&!w.disabled&&!w.classList.contains("hidden")):null;if(l&&!v)return showToast(`T\
hinking: \u300C${r}\u300D\u306F\u3053\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1;if(y==="none"){if(h&&h.disabled)return showToast("Thinking: \u3053\u306E\u30E2\u30C7\u30EB\u3067\u306FOFF\u306B\u3067\u304D\u307E\u305B\u3093",
"warning"),!1;h&&h.checked?(h.checked=!1,h.dispatchEvent(new Event("change",{bubbles:!0}))):l&&(l.value=
"none",l.dispatchEvent(new Event("change",{bubbles:!0})))}else h&&!h.checked&&!h.disabled&&(h.checked=
!0,h.dispatchEvent(new Event("change",{bubbles:!0}))),l&&(l.value=y,l.dispatchEvent(new Event("chang\
e",{bubbles:!0})));return refreshMinimalOptionItems(),showToast(`Thinking: ${r}`,"success"),!0}if(m===
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
                `;let x=!1;k.addEventListener("pointerdown",C=>{typeof C.button=="number"&&C.button!==
0||(C.preventDefault(),x=!0,selectSlashCommand(v.id))}),k.addEventListener("click",C=>{C.preventDefault(),
x||selectSlashCommand(v.id)}),k.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(k)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,d=r.top,m=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&d>c){const v=Math.min(m,d-h);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(m,c-h);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
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
return;const m=document.createElement("div");m.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,v])=>{const w=AI_SETTING_JUMP_TARGETS[y]||{label:y},k=document.createElement("button");k.
type="button",k.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const x=document.createElement("span");x.className="min-w-0 flex-1";const C=document.createElement(
"span");C.className="block text-xs font-bold text-blue-200",C.textContent=w.label;const L=document.createElement(
"span");L.className="block mt-0.5 text-[11px] text-gray-300 break-words",L.textContent=formatAiSettingValue(
v);const M=document.createElement("i");M.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",x.appendChild(C),x.appendChild(L),k.appendChild(x),k.appendChild(M),k.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),m.appendChild(k)});const h=d.querySelector(".message-footer-\
meta");h?d.insertBefore(m,h):d.appendChild(m),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
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
json().catch(()=>({})),m=get(r);if(m&&m.remove(),d&&d.status==="ok"&&d.mode==="inspect"&&d.current){
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
const d=get(r);d&&d.remove();const m=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(m),showToast(
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
e)},n.appendChild(x)});const l=i.getBoundingClientRect(),c=window.innerHeight,d=c-l.bottom,m=l.top,h=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&m>d){const w=Math.min(h,m-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
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
var q,ee,B;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),r=[];for(const K of a)r.push({inlineData:{
mimeType:K.file.type,data:await fileToBase64Payload(K.file)}});r.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const d={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:r}],generationConfig:l};!!(get("enable-python")&&get("enable-python").
checked)&&(d.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden");const h=userAutoScroll&&isChatNearBottom();
h||pauseChatAutoScroll();const y=collectImageUrlsForSend(),v=Array.from(((q=get("file-preview-thumbs"))==
null?void 0:q.querySelectorAll("img"))||[]).map(K=>K.cloneNode(!0)),w=renderMessage(Date.now(),"user",
e,JSON.stringify(collectImageUrlsForSend()),null,null,null,!1,currentQuote,null,null,null,null,null,
null,null,h);if(v.length&&!y.length){const K=document.createElement("div");K.className="flex gap-2 f\
lex-wrap",v.forEach(ce=>{ce.removeAttribute("onclick"),K.appendChild(ce)}),w.querySelector(".message\
-bubble").appendChild(K)}const k=takePromptForSend(),x=animatePromptToMessage(w,k,h),C=`browser-fast\
-${Date.now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start\
 mb-4 ai-pending-row fade-in"><div id="${C}" class="message-bubble ai-pending-bubble bg-gray-700 tex\
t-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(n,"Gemini\u3078\u76F4\u63A5\u9001\
\u4FE1\u4E2D...")}</div></div>`);const L=get(C),M=startPendingElapsed(L,t);activeStreamingBubbleId=C,
setSendBtnToStopMode(),h&&resumeChatAutoScroll(),abortController=new AbortController;let E="",D="",N=null;
const G=new Set,j=[];let J=null,oe=null,me=!1;const ie={},ye=[];let $=null,R="";const U=window.ProgressSpinner?
window.ProgressSpinner.startFlow("browserFast"):null;let P=!1;try{const K=await fetch(`https://gener\
ativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json","x-goog-api-key":browserFastApiKey},body:JSON.
stringify(d),signal:abortController.signal}));if(!K.ok){const ge=await K.json().catch(()=>({}));throw new Error(
ge&&ge.error&&ge.error.message?ge.error.message:`Gemini API HTTP ${K.status}`)}window.ConnectionMonitor&&
(P=!0,window.ConnectionMonitor.operationStarted()),U&&U.setPhase("waiting"),k.accept(!1),(ee=window.
ActivityLog)==null||ee.log("send.stream.start");let ce=!1;const de=K.body.getReader(),te=new TextDecoder;
let re="";const Q=o(ge=>{var he,dt;const pe=ge.split(/\r?\n/).filter(Ne=>Ne.startsWith("data:")).map(
Ne=>Ne.slice(5).trim()).join("");if(!pe||pe==="[DONE]")return;const Se=JSON.parse(pe);if(Number.isInteger(
(he=Se.usageMetadata)==null?void 0:he.promptTokenCount)){N=N||{};for(const Ne of["promptTokenCount",
"candidatesTokenCount","thoughtsTokenCount","cachedContentTokenCount"])Number.isInteger(Se.usageMetadata[Ne])&&
Se.usageMetadata[Ne]>=0&&(N[Ne]=Se.usageMetadata[Ne])}for(const Ne of Se.candidates||[])for(const gt of((dt=
Ne.groundingMetadata)==null?void 0:dt.webSearchQueries)||[])typeof gt=="string"&&gt.trim()&&G.add(gt);
if(G.size&&(N=N||{},N.grounding_queries=G.size),Se.error)throw new Error(Se.error.message||"Gemini A\
PI error");if((Array.isArray(Se.candidates)?Se.candidates:[]).forEach(Ne=>{(Ne&&Ne.content&&Array.isArray(
Ne.content.parts)?Ne.content.parts:[]).forEach(ze=>{if(ze&&typeof ze.thoughtSignature=="string"&&!j.
includes(ze.thoughtSignature)&&j.push(ze.thoughtSignature),ze&&ze.executableCode&&typeof ze.executableCode.
code=="string"){const pt=ze.executableCode.code;E+=`
\`\`\`python
${pt}
\`\`\`
`,$=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,R=pt,ie[$]||(L.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml($)),ie[$]=L.querySelector(`[data-py-id="${$}"]`)),updateBrowserFastPythonBox(
ie[$],"code",pt);return}if(ze&&ze.codeExecutionResult&&typeof ze.codeExecutionResult.output=="string"){
const pt=ze.codeExecutionResult.output;E+=`
**Output:**
\`\`\`
${pt}
\`\`\`
`;const Ct=$||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;ye.push({code:R||
"",output:pt}),ie[Ct]||(L.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(Ct)),ie[Ct]=L.querySelector(
`[data-py-id="${Ct}"]`)),updateBrowserFastPythonBox(ie[Ct],"output",pt);return}const wt=typeof ze.text==
"string"?ze.text:"";wt&&(ze.thought===!0?D+=wt:E+=wt)})}),!me&&(E||D)){beginPendingToStreamTransition(
L);const Ne=L.querySelector(".content-area");Ne&&Ne.remove(),me=!0}D&&(oe||(L.insertAdjacentHTML("af\
terbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"\
><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></d\
iv></div>'),oe=L.querySelector(".thought-content")),oe.textContent=D),E&&(J||(J=document.createElement(
"div"),J.className="content-area prose prose-invert text-sm break-words",L.appendChild(J)),renderAiMarkdownInto(
J,E,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:ge,value:pe}=await de.
read();if(!ge&&!ce&&(ce=!0,(B=window.ActivityLog)==null||B.log("send.stream.first")),ge)break;window.
ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),U&&U.setPhase("receiving"),re+=te.decode(
pe,{stream:!0});const Se=re.split(/\r?\n\r?\n/);re=Se.pop()||"",Se.forEach(Q)}if(re+=te.decode(),re.
trim()&&Q(re),!E.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
J&&renderAiMarkdownInto(J,E,{incrementalMath:!0}),oe&&oe.classList.add("collapsed"),ye.length&&(E+=ye.
map(ge=>`
\`\`\`pyexec
${JSON.stringify(ge)}
\`\`\`
`).join(""));let fe=[];a.length&&(U&&U.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),fe=await uploadBrowserFastLocalFiles(k.files,!0)),U&&U.setPhase("saving");const ue=[...y,
...fe],Ge=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:E,pricing_usage:N,thought_content:D,model:n,image_urls:ue,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:j,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),L),Oe=await Ge.json().
catch(()=>({}));if(!Ge.ok||!Oe.thread_id)throw new Error(Oe.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const et=!currentThreadId;currentThreadId=String(Oe.thread_id),currentParentId=Oe.assistant_message_id||
null,currentLeafId=Oe.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!et}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(K){if(x(),k.restore(),K.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${K.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const ce=K.message||"\
\u30A8\u30E9\u30FC";L&&L.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(ce));try{let de=E||
"";ye.length&&(de+=ye.map(ue=>`
\`\`\`pyexec
${JSON.stringify(ue)}
\`\`\`
`).join(""));const te=buildChatErrorMarkdown(ce,de),re=a.length?[]:y,Q=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:te,
pricing_usage:N,thought_content:D||"",model:n,image_urls:re,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:
null,thought_signatures:j,turnstile_token:botTurnstileTokenForRequest()}),signal:abortController&&!abortController.
signal.aborted?abortController.signal:void 0}),L),fe=await Q.json().catch(()=>({}));if(Q.ok&&fe.thread_id){
const ue=!currentThreadId;currentThreadId=String(fe.thread_id),currentParentId=fe.assistant_message_id||
null,currentLeafId=fe.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!ue}),applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(de){
sendClientDebugLog("error",`Browser fast error persist failed: ${de&&de.message?de.message:de}`)}}}finally{
k.releaseFiles(),M&&M(),P&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),U&&U(),
setSendBtnToSendMode(),activeStreamingBubbleId===C&&(activeStreamingBubbleId=null),abortController=null,
updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
o(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var Ut,ln,Zt,_t,cn;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const X=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(X);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const X=pendingSlashCommand,
_e=a.trim(),Ae=get("model-select")?get("model-select").value:null;if(X==="settings"){if(!_e){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!Ae){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
_e,Ae)}else executeMinimalSlashCommand(X,_e)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const r=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(r&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(X=>X.id===
r[1].toLowerCase())){executeMinimalSlashCommand(r[1].toLowerCase(),r[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const l=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(l&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(l)setBrowserFastModeEnabled(!1);else{const X=browserFastModeIneligibility(
a);if(!X){try{await sendBrowserFastMessage(a,t)}catch(_e){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${_e.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${X}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(_e){showToast(
_e.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),d=c.map(X=>X.path),m=c.filter(X=>normalizeAttachmentSource(
X.source)==="upload").map(X=>X.path),h=getModelMediaSupport(get("model-select").value),y=d.some(X=>isAudioPath(
X)),v=d.some(X=>isVideoPath(X)),w=(get("model-select").value||"").toLowerCase(),k=get("enable-python"),
x=!!(k&&k.checked);if(y&&!h.audio||v&&!h.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&d.length===0)return;if(isMistralOcrModel(
w)){const X=/https?:\/\/\S+/i.test(a);if(d.filter(Ae=>isAudioPath(Ae)||isVideoPath(Ae)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!d.length&&!X){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const C=a.trim();if(/^\/settings(?:\s|$)/i.
test(C)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(C)){const X=C.replace(/^\/settings\s*/i,"").trim();
if(!X){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ae=get("prompt-input");Ae.value="/settings ";const Ue=extractSlashCommandToken(Ae.value);
lastSlashFilter=Ue,showSlashCommandSuggestions(Ue),Ae.focus();return}const _e=get("model-select")?get(
"model-select").value:null;if(!_e){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
X,_e);return}if(isGeminiLocalPythonMode(w,y,v,x)&&!await confirmGeminiLocalPythonSwitch())return;let L=null,
M=[];if(codingModeEnabled){const X=collectCodingCandidates(a),_e=X.filter(Ze=>Ze.prompt_source),Ae=X.
filter(Ze=>!Ze.prompt_source),Ue=_e.reduce((Ze,Lt)=>Ze+String(Lt.code||"").length,0);if(Ue>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let ot=3e5-Ue;const ht=[];for(let Ze=Ae.length-1;Ze>=0;Ze--){const Lt=String(Ae[Ze].
code||"").length;Lt>ot||(ht.unshift(Ae[Ze]),ot-=Lt)}M=codingTargetSelection?ht.slice(-1):[..._e,...ht];
const St=_e.length?_e[_e.length-1]:null;if(L=codingTargetSelection?M[0]:St||M[M.length-1]||null,codingModeEffective=
!!(L&&String(L.code||"").trim()),codingModeEffective&&L.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const Ze=String(((Ut=get(
"model-select"))==null?void 0:Ut.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(Ze)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const E=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${d.length} search=${get("enable-search").checked}`);const D=a,N=hasMarkerHint()?MARKER_HINT_TEXT:
null,G=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&d.length===0){showToast("Ma\
sk \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if(G&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=userAutoScroll&&isChatNearBottom();j||pauseChatAutoScroll();const J=editingMessageId,
oe=promptSettingsBeforeEdit;promptSettingsBeforeEdit=null;const me=currentParentId,ie=J!=null;J&&(editingMessageId=
null,setEditUi(!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const ye=[],$=o(
X=>{if(X==null)return;let _e=document.getElementById(`msg-${X}`);for(;_e;)_e.classList&&_e.classList.
contains("message-group")&&(ye.push({node:_e,prevDisplay:_e.style.display}),_e.style.display="none"),
_e=_e.nextElementSibling},"hideRenderedBranchFrom"),R=o(()=>{ye.forEach(({node:X,prevDisplay:_e})=>{
X&&(X.style.display=_e||"")}),ye.length=0},"restoreHiddenBranch");J&&$(J);const U=Date.now(),P=isGptImageModel()||
G?currentMaskImage:null,q=renderMessage(U,"user",D,JSON.stringify(d),null,null,null,!1,currentQuote,
null,null,null,null,null,null,null,j,me,activeSkill?activeSkill.name:null,null,P);let ee=!1;const B=/(https?:\/\/)?(x\.com|twitter\.com)\//i,
K=!i&&(B.test(D||"")||B.test(currentQuote||"")),ce="grok-4-fast-reasoning",de=o(()=>{get("enable-sea\
rch").checked=!0,get("model-select").value!==ce&&selectModelById(ce)},"applyXLinkAuto");if(K&&!isMistralOcrModel()&&
!get("enable-search").checked)if(autoSearchOnLinks)de();else{const X=get("auto-search-banner"),_e=get(
"auto-search-on-btn"),Ae=get("auto-search-off-btn"),Ue=get("auto-search-remember");X&&_e&&Ae&&(Ue&&(Ue.
checked=!1),await new Promise(ot=>{X.classList.remove("hidden");const ht=o(St=>{X.classList.add("hid\
den"),_e.onclick=null,Ae.onclick=null,ot(St)},"cleanup");_e.onclick=()=>ht("enable"),Ae.onclick=()=>ht(
"disable")}).then(async ot=>{ot==="enable"?(de(),Ue&&Ue.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):ee=!0}))}const te=String(get("reasoning-effort").value||"").
toLowerCase(),re=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&te==="non\
e",Q={client_request_id:createClientRequestId(),thread_id:currentThreadId,message:D,model:get("model\
-select").value,image_urls:d,image_items:c,uploaded_image_urls:m,temporary_chat:temporaryChatEnabled,
enable_search:get("enable-search").checked,enable_url_context:get("enable-url-context")?get("enable-\
url-context").checked:!1,enable_maps:get("enable-maps")?get("enable-maps").checked:!1,enable_python:get(
"enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_skills:get("enable-skills")?get("en\
able-skills").checked:!0,enable_file_creation:get("enable-file-creation")?get("enable-file-creation").
checked:!0,enable_thinking:re?!1:get("enable-thinking").checked,thinking_level:get("thinking-level").
value,thinking_budget:get("thinking-budget")?get("thinking-budget").value:null,reasoning_effort:get(
"reasoning-effort").value,enable_system_prompt:get("enable-sys-prompt").checked,enable_prompt_caching:get(
"enable-prompt-cache")?get("enable-prompt-cache").checked:!1,marker_system_prompt:N,safety_setting:get(
"safety-setting").value,tts_voice:isTtsModel()&&get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&
get("tts-voice-custom")?get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-langu\
age")?get("tts-language").value:null,tts_style:isTtsModel()&&get("tts-style")&&get("tts-style-wrap")&&
!get("tts-style-wrap").classList.contains("hidden")?get("tts-style").value:null,tts_speed:isTtsModel()&&
get("tts-speed")?get("tts-speed").value:null,image_size:isGptImageModel()&&get("gpt-image-size")?get(
"gpt-image-size").value:null,image_quality:isGptImageModel()&&get("gpt-image-quality")?get("gpt-imag\
e-quality").value:null,image_complexity:isGptImageModel()&&get("gpt-image-complexity")?get("gpt-imag\
e-complexity").value:null,image_format:isGptImageModel()&&get("gpt-image-format")?get("gpt-image-for\
mat").value:null,image_compression:isGptImageModel()&&get("gpt-image-compression")?get("gpt-image-co\
mpression").value:null,image_frame_count:isGptImageModel()&&get("gpt-image-frame-count")?get("gpt-im\
age-frame-count").value:null,image_frame_duration_ms:isGptImageModel()&&get("gpt-image-frame-duratio\
n")?get("gpt-image-frame-duration").value:null,image_mask:P,gemini_image_aspect:isGeminiImageModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:me,parent_id_explicit:ie,disable_auto_search:ee,
image_vision_model:currentVisionModel||null,coding_mode:E,coding_target:E?{id:L.candidate_id,code:L.
prompt_source?null:L.code,language:L.language||"text",key:L.key||null,message_id:L.message_id||null,
source:L.prompt_source?"prompt":"history",explicit:L.explicit===!0}:null,coding_candidates:E?M.map(X=>({
id:X.candidate_id,source:X.prompt_source?"prompt":"history",prompt_index:X.prompt_source?X.prompt_index:
null,code:X.prompt_source?null:X.code,language:X.language||"text",explicit:X.explicit===!0})):[],batch_mode:l,
canvas_mode:!!canvasModeEnabled};n&&(Q.turnstile_token=n);const fe=get("thread-custom-instruction");
fe&&(Q.thread_custom_instruction=fe.value||""),activeSkill?(Q.system_prompt=activeSkill.instruction,
Q.enable_system_prompt=!0,Q.skill_uuid=activeSkill.uuid):Q.skill_uuid=null;const ue=takePromptForSend(
e.animate!==!1),Ge=animatePromptToMessage(q,ue,j&&e.animate!==!1);setSendBtnToStopMode();const Oe="a\
i-"+Date.now(),et=String(Q.model||"").toLowerCase(),ge=!!Q.enable_thinking||!!te&&te!=="none",pe=et.
includes("gemini")||et.includes("o1")||et.includes("o3")||et.includes("gpt-5")||et.includes("reasoni\
ng")&&!et.includes("non-reasoning"),Se=ge&&pe;let De=buildPendingSkeletonHtml(Q.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending\
-row fade-in"><div id="${Oe}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rou\
nded-2xl rounded-tl-none shadow-md relative">${De}</div></div>`),j&&resumeChatAutoScroll();const he=get(
Oe),dt=l?null:startPendingElapsed(he,t);activeStreamingBubbleId=Oe,canvasModeEnabled&&resetCanvasPreviewPanel();
let Ne=null;const gt=o(X=>!Se||!he?null:((!Ne||!he.contains(Ne))&&(Ne=he.querySelector(".thought-con\
tent")),Ne||(he.insertAdjacentHTML("afterbegin",'<div class="thought-container"><div class="thought-\
header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i>\
 Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>'),Ne=
he.querySelector(".thought-content")),Ne&&(Ne.setAttribute("data-placeholder","1"),Ne.textContent=X||
"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),Ne),"ensureThoughtPlaceholder");Se&&
gt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),abortController=new AbortController;
const ze=currentThreadId,wt=nowPerfMs(),pt=Date.now();let Ct=!1,rn=!1,Dt=!1,$t=null,Ht=null,It=null,
Yt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;const qt=o((X,_e)=>{if(!_e||
X==="status"&&Ct||X==="thought"&&rn||X==="content"&&Dt)return;const Ae=Math.max(0,nowPerfMs()-wt);X===
"status"?$t=Ae:X==="thought"?Ht=Ae:X==="content"&&(It=Ae),reportFirstTokenLatency({latency_seconds:Ae/
1e3,latency_ms:Ae,thread_id:Yt||currentThreadId,job_id:currentJobId,model:Q.model,first_event_type:X,
client_sent_at_ms:pt}),X==="status"?Ct=!0:X==="thought"?rn=!0:X==="content"&&(Dt=!0)},"maybeReportFi\
rstEventLatency"),kt=window.ProgressSpinner?window.ProgressSpinner.startFlow("chat"):null;let Pt=!1,
Qt=!1,Gt=null,xt=null,Ot=!1;try{Q.thread_id&&activeSkill&&(threadSkillMap[Q.thread_id]=activeSkill,pendingSkillForNewThread=
null);const X=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.urls.chatStream,manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Q),signal:abortController.
signal}),he);if(sendClientDebugLog("info",`Prompt stream response status: ${X.status}`),!X.ok){const He=await X.
json().catch(()=>({})),We=new Error(He.error||`HTTP ${X.status}`);throw We.serverCode=He.code||null,
We.serverModel=He.model||Q.model,We.acceptedJobId=He.job_id||null,We.acceptedThreadId=He.thread_id||
null,We}Pt=!0,window.ConnectionMonitor&&(Ot=!0,window.ConnectionMonitor.operationStarted()),kt&&kt.setPhase(
"waiting"),ue.accept(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const _e=o(()=>{if(!he)return;
const He=he.querySelector(".content-area");if(He&&He.getAttribute("data-api-accepted")!=="1"&&(He.setAttribute(
"data-api-accepted","1"),!updatePendingSkeletonStatus(he,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){
He.outerHTML=buildPendingSkeletonHtml(Q.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const We=he.querySelector(".content-area");We&&We.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
he,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");_e(),(ln=window.ActivityLog)==null||ln.log("send.stream.start");let Ae=!1;const Ue=X.
body.getReader(),ot=new TextDecoder;let ht="",St="",Ze="",Lt=!0,Ye=null,st=null,Tt=null,dn=!1;const Nt={};
let un=0,en=!1;for(;!en;){const{done:He,value:We}=await Ue.read();if(!He&&!Ae&&(Ae=!0,(Zt=window.ActivityLog)==
null||Zt.log("send.stream.first")),He&&((_t=window.ActivityLog)==null||_t.log("send.stream.done")),He)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),kt&&kt.setPhase("receiving"),
ht+=ot.decode(We,{stream:!0});let Mt=ht.split(`
`);ht=Mt.pop();let Rt=!1,mn=!1;for(let zt of Mt)if(zt.trim())try{const Me=JSON.parse(zt);if(Me.type===
"thread_id"){_e();const Ie=Me.content!==null&&Me.content!==void 0?String(Me.content):Me.content;Ie&&
(Yt=Ie,currentThreadId!==Ie&&(currentThreadId=Ie,history.pushState({},"","/c/"+Ie)),activeSkill&&(threadSkillMap[Ie]=
activeSkill,pendingSkillForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Me.type==="\
job_id"){_e(),currentJobId=Me.content,l&&showToast("Batch\u767B\u9332","info");continue}if(Me.type===
"search_status"){Tt=updateSearchBox(he,Tt,Me.content);continue}if(Me.type==="mcp"){handleMcpStreamEvent(
he,Me.content||{});continue}if(Me.type==="mcp_decision_request"){openMcpDecisionModal(Me.content||{});
continue}if(Me.type==="status"){_e();const Ie=Me.content===null||Me.content===void 0?"":String(Me.content);
if(qt("status",!!Ie),Lt&&he){const Ke=Ie||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(
he,Ke,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ut=he.querySelector(".content-area");ut&&(ut.outerHTML=buildPendingSkeletonHtml(Q.model,Ke),updatePendingSkeletonStatus(
he,Ke,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
Se&&gt(Ie||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Lt){beginPendingToStreamTransition(
he);const Ie=he.querySelector(".content-area");Ie&&(Ie.innerHTML=""),Lt=!1}if(Me.type==="coding_diff")
appendCodingLiveDiff(he,Me.content||{}),qt("content",!0);else if(Me.type==="thought"){if(Ye||(Ye=he.
querySelector(".thought-content")),Ze+=Me.content,qt("thought",!!Me.content),!Ye){const Ie='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';Tt?Tt.insertAdjacentHTML(
"afterend",Ie):he.insertAdjacentHTML("afterbegin",Ie),Ye=he.querySelector(".thought-content")}if(Ye&&
Ye.getAttribute("data-placeholder")==="1"){if(Ye.textContent="",Ye.removeAttribute("data-placeholder"),
Ye){const Ie=Ye.parentElement.querySelector(".thought-header");Ie&&Ie.classList.remove("thinking-shi\
mmer")}Ze=Me.content}Ye.classList.remove("collapsed"),mn=!0}else if(Me.type==="image_analysis"){const Ie=Me.
content===null||Me.content===void 0?"":String(Me.content);if(!he)continue;let Ke=he.querySelector(".\
image-analysis-box");if(!Ke){const mt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';Tt?Tt.insertAdjacentHTML("afterend",mt):he.insertAdjacentHTML("afterbegin",mt),Ke=he.querySelector(
".image-analysis-box")}const ut=Ke.querySelector(".image-analysis-text");ut&&(ut.textContent=Ie)}else if(Me.
type==="python"){const Ie=Me.content||{},Ke=Ie.id||`py_${Date.now()}`;if(!Nt[Ke]){const mt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${Ke}" data-collapsed="true" data-code-key="${Ke}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;Tt?Tt.insertAdjacentHTML("afterend",mt):he.insertAdjacentHTML(
"afterbegin",mt),Nt[Ke]=he.querySelector(`[data-py-id="${Ke}"]`)}const ut=Nt[Ke];if(ut){if(Ie.code!==
void 0){const mt=Ie.code==null?"":String(Ie.code),At=ut.querySelector(".python-code");At&&(At.textContent=
mt,At.removeAttribute("data-highlighted"),queueHighlight(ut,mt));const bt=ut.querySelector('.copy-bt\
n[data-copy="code"]');bt&&bt.setAttribute("data-code",encodeURIComponent(mt).replace(/'/g,"%27"))}if(Ie.
output!==void 0){const mt=Ie.output==null?"":String(Ie.output),At=ut.querySelector(".python-output");
At&&(At.textContent=mt);const bt=ut.querySelector('.copy-btn[data-copy="output"]');bt&&bt.setAttribute(
"data-code",encodeURIComponent(mt).replace(/'/g,"%27"))}}}else if(Me.type==="content"){const Ie=Me.content===
null||Me.content===void 0?"":String(Me.content);St+=Ie,/[`~]/.test(Ie)&&activateDeferredCodingModeFromStream(
St),st||(st=he.querySelector(".content-area")||document.createElement("div"),st.className="prose pro\
se-invert text-sm break-words",he.contains(st)||he.appendChild(st)),Rt=!0,qt("content",!!Ie)}else if(Me.
type==="error"){dn=!0,en=!0,he.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Me.content)),
showToast(Me.content||"Unknown error","error",!0);break}}catch{}if(mn&&Ye&&(Ye.textContent=Ze,userAutoScroll&&
(Ye.scrollTop=Ye.scrollHeight)),Rt&&st){const zt=Date.now();if(zt-un>100){const Me=snapshotCodeCollapse(
st);renderAiMarkdownInto(st,St,{incrementalMath:!0}),applyCodeCollapse(st,Me,!0),un=zt}}scrollToBottom()}
if(kt&&kt(),st){const He=snapshotCodeCollapse(st);renderAiMarkdownInto(st,St,{incrementalMath:!0}),applyCodeCollapse(
st,He,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),he){queueHighlight(he,St);const He=nowPerfMs()-
wt;enableLatencyMetrics&&reportFirstTokenLatency({is_total:!0,latency_seconds:He/1e3,latency_ms:He,thread_id:Yt||
currentThreadId,job_id:currentJobId,model:Q.model,client_sent_at_ms:pt,client_done_at_ms:Date.now()});
let We=null;$t!==null&&(We=$t),Ht!==null&&(We===null||Ht<We)&&(We=Ht),It!==null&&(We===null||It<We)&&
(We=It);const Mt={total_ms:He,first_response_ms:We,phases:We!==null?[{key:"generation",ms:Math.max(0,
He-We)}]:[],source:"client"};messageMeta[Oe]=Object.assign(messageMeta[Oe]||{},{role:"assistant",model:Q.
model,processing_time:Mt});const Rt=he.querySelector(".processing-time-row");Rt&&Rt.remove(),he.insertAdjacentHTML(
"beforeend",buildProcessingTimeHtml(Oe,Mt))}editingMessageId=null,setEditUi(!1),he&&he.querySelectorAll(
".thought-content").forEach(We=>We.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!J}),!dn&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(He=>He.json()).then(He=>{He.title&&
(document.title=He.title+" - AI Chat",setCurrentChatHeaderTitle(He.title),loadThreads())}):loadThreads(
!1)}catch(X){(cn=window.ActivityLog)==null||cn.log(X.name==="AbortError"?"send.cancelled":"send.erro\
r"),Ge();let _e=!1;const Ae=X.name==="AbortError"&&isManualStopAbortForThread(ze);if(X.name==="Abort\
Error"&&!Ae&&(_e=await syncThreadAfterAbortedStream(ze,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${X.message}`),!Pt){q&&q.remove();const Ue=he&&he.closest(
".fade-in");Ue&&Ue.remove(),delete messageStore[U],delete messageMeta[U]}if(X.serverCode==="request_\
already_accepted"&&X.acceptedJobId&&X.acceptedThreadId)Pt=!0,Gt={job_id:X.acceptedJobId,thread_id:String(
X.acceptedThreadId),model:Q.model,pendingStartedAt:l?null:t},ue.accept();else if(Pt&&!Ae)xt={job_id:normalizeJobIdForUi(
currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):null,model:Q.model,pendingStartedAt:l?
null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\
\u307E\u3059\u3002","warning",!1);else if(X.serverCode==="turnstile_required"){const Ue=await getTurnstileToken();
Ue?(await verifyTurnstileOnServer(Ue,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(X.serverCode==="api_key_missing"){const Ue=X.serverModel||Q.model,ot=await showApiKeyRequiredModalAsync(
Ue);ot==="set"?Qt=!0:ot==="switch"?showModal("model-modal"):showToast(X.message||`${getModelNameById(
Ue)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(X.
name!=="AbortError"){const Ue="Connection Error: "+X.message;showToast(Ue,"error",!0)}if(!Pt){const Ue=J&&
ue.canRetry()&&!editingMessageId;ue.restore(),Ue&&(editingMessageId=J,currentParentId=me,promptSettingsBeforeEdit=
oe,setEditUi(!0))}J&&!_e&&R()}finally{dt&&dt(),Ot&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),kt&&kt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===Oe&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,setEditUi(!!editingMessageId)}if(Gt){const X=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=Gt.thread_id,(X!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
Gt,currentThreadId)}if(xt&&xt.thread_id)return reconnectPendingStreamUntilAvailable(xt,xt.thread_id);
if(Qt&&ue.canRetry())return sendMessageLogged({animate:!1})}o(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var oe,me,ie,ye;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const $=a.
querySelector(".content-area");$?$.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const r=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const l=currentThreadId,
c=i.toLowerCase(),d=c.includes("gemini")||c.includes("o1")||c.includes("o3")||c.includes("gpt-5")||c.
includes("reasoning")&&!c.includes("non-reasoning");let m=null;const h=o($=>!d||!a?null:((!m||!a.contains(
m))&&(m=a.querySelector(".thought-content")),m||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),m=a.querySelector(".thought-content")),m&&(m.setAttribute("data-pla\
ceholder","1"),m.textContent=$||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),m),
"ensureThoughtPlaceholder");d&&h("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",v="",w="",k=!0,x=null,C=null,L=null,M=!1;const E={};let D=0,N=!1;const G=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let j=!1,J=!1;try{const $=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!$.ok)throw new Error(`Resume failed (${$.status})`);window.ConnectionMonitor&&
(J=!0,window.ConnectionMonitor.operationStarted()),G&&G.setPhase("waiting"),(oe=window.ActivityLog)==
null||oe.log("send.stream.start");let R=!1;const U=$.body.getReader(),P=new TextDecoder;for(;!N;){const{
done:q,value:ee}=await U.read();if(!q&&!R&&(R=!0,(me=window.ActivityLog)==null||me.log("send.stream.\
first")),q&&((ie=window.ActivityLog)==null||ie.log("send.stream.done")),q)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),G&&G.setPhase("receiving"),y+=P.decode(ee,{stream:!0});let B=y.
split(`
`);y=B.pop();let K=!1,ce=!1;for(let de of B)if(de.trim())try{const te=JSON.parse(de);if(te.type==="j\
ob_id"){currentJobId=te.content||t;continue}if(te.type==="search_status"){L=updateSearchBox(a,L,te.content);
continue}if(te.type==="mcp"){handleMcpStreamEvent(a,te.content||{});continue}if(te.type==="mcp_decis\
ion_request"){openMcpDecisionModal(te.content||{});continue}if(te.type==="status"){const re=te.content===
null||te.content===void 0?"":String(te.content);if(k&&a){const Q=re||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";
if(!updatePendingSkeletonStatus(a,Q,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const fe=a.querySelector(".content-area");fe&&(fe.outerHTML=buildPendingSkeletonHtml(i,Q),updatePendingSkeletonStatus(
a,Q,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
d&&h(re||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(k){beginPendingToStreamTransition(
a);const re=a.querySelector(".content-area");re&&(re.innerHTML=""),k=!1}if(te.type==="coding_diff")appendCodingLiveDiff(
a,te.content||{});else if(te.type==="thought"){if(x||(x=a.querySelector(".thought-content")),w+=te.content,
!x){const re='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content">\
</div></div>';L?L.insertAdjacentHTML("afterend",re):a.insertAdjacentHTML("afterbegin",re),x=a.querySelector(
".thought-content")}if(x&&x.getAttribute("data-placeholder")==="1"){if(x.textContent="",x.removeAttribute(
"data-placeholder"),x){const re=x.parentElement.querySelector(".thought-header");re&&re.classList.remove(
"thinking-shimmer")}w=te.content}x.classList.remove("collapsed"),ce=!0}else if(te.type==="image_anal\
ysis"){const re=te.content===null||te.content===void 0?"":String(te.content);if(!a)continue;let Q=a.
querySelector(".image-analysis-box");if(!Q){const ue='<div class="image-analysis-box mb-2 p-2 bg-blu\
e-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1">\
<i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] tex\
t-gray-300"></div></div>';L?L.insertAdjacentHTML("afterend",ue):a.insertAdjacentHTML("afterbegin",ue),
Q=a.querySelector(".image-analysis-box")}const fe=Q.querySelector(".image-analysis-text");fe&&(fe.textContent=
re)}else if(te.type==="python"){const re=te.content||{},Q=re.id||`py_${Date.now()}`;if(!E[Q]){const ue=`\
<div class="code-wrapper python-box collapsed" data-py-id="${Q}" data-collapsed="true" data-code-key\
="${Q}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Exec\
ution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" a\
ria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code"\
 data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class\
="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-alig\
n-left"></i></button></div></div><div class="code-body"><div class="python-section"><div class="pyth\
on-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div clas\
s="python-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext p\
ython-output"></code></pre></div></div></div>`;L?L.insertAdjacentHTML("afterend",ue):a.insertAdjacentHTML(
"afterbegin",ue),E[Q]=a.querySelector(`[data-py-id="${Q}"]`)}const fe=E[Q];if(fe){if(re.code!==void 0){
const ue=re.code==null?"":String(re.code),Ge=fe.querySelector(".python-code");Ge&&(Ge.textContent=ue,
Ge.removeAttribute("data-highlighted"),queueHighlight(fe,ue));const Oe=fe.querySelector('.copy-btn[d\
ata-copy="code"]');Oe&&Oe.setAttribute("data-code",encodeURIComponent(ue).replace(/'/g,"%27"))}if(re.
output!==void 0){const ue=re.output==null?"":String(re.output),Ge=fe.querySelector(".python-output");
Ge&&(Ge.textContent=ue);const Oe=fe.querySelector('.copy-btn[data-copy="output"]');Oe&&Oe.setAttribute(
"data-code",encodeURIComponent(ue).replace(/'/g,"%27"))}}}else if(te.type==="content"){const re=te.content===
null||te.content===void 0?"":String(te.content);v+=re,/[`~]/.test(re)&&activateDeferredCodingModeFromStream(
v),C||(C=a.querySelector(".content-area")||document.createElement("div"),C.className="prose prose-in\
vert text-sm break-words",a.contains(C)||a.appendChild(C)),K=!0}else if(te.type==="error"){M=!0,N=!0,
a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(te.content)),showToast(te.content||"Unknow\
n error","error",!0);break}}catch{}if(ce&&x&&(x.textContent=w,userAutoScroll&&(x.scrollTop=x.scrollHeight)),
K&&C){const de=Date.now();if(de-D>100){const te=snapshotCodeCollapse(C);renderAiMarkdownInto(C,v,{incrementalMath:!0}),
applyCodeCollapse(C,te,!0),D=de}}scrollToBottom()}if(G&&G(),C){const q=snapshotCodeCollapse(C);renderAiMarkdownInto(
C,v,{incrementalMath:!0}),applyCodeCollapse(C,q,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,
v),a&&a.querySelectorAll(".thought-content").forEach(ee=>ee.classList.add("collapsed")),await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0}),loadThreads(!1)}catch($){(ye=window.ActivityLog)==null||
ye.log($.name==="AbortError"?"send.cancelled":"send.stream.error");const R=$.name==="AbortError"&&isManualStopAbortForThread(
l);$.name==="AbortError"&&!R&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
R||(j=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{r&&r(),J&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
G&&G(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
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
ll-sentinel"></div>',threadObserver){threadObserver.disconnect();const m=get("scroll-sentinel");m&&threadObserver.
observe(m)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const d=String(
c.id),m=document.createElement("div"),h=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=d===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";m.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,m.dataset.threadId=d,m.dataset.threadTitle=c.title||"",m.innerHTML=`<div class="flex \
items-center gap-1 truncate flex-1"><button class="${h} hover:text-yellow-400 px-1" onclick="toggleB\
ookmark(event, '${d}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(
c.title||"No Title")}</span>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 m\
d:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:tex\
t-white px-1 transition" onclick="renameThread(event, '${d}')"><i class="fas fa-pen text-xs"></i></b\
utton><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, \
'${d}')"><i class="fas fa-trash text-xs"></i></button></div>`,m.onclick=k=>{k.target.closest("button")||
k.target.closest("[data-thread-actions]")||loadMessages(d)},l?r.insertBefore(m,l):r.appendChild(m)}),
hasMoreThreads=!!a.has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendere\
d count="+a.threads.length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){
console.error("Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=
!1,updateThreadHighlighting(),snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThre\
ads");function initPullToRefresh(e,t){const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,
r=88,l=52,c=.5,d=8;let m=0,h=!1,y=0,v=null;const w=o(()=>get(i),"indicatorEl"),k=o(()=>{const L=w();
return L?L.querySelector(".ptr-pull-label"):null},"labelEl"),x=o(L=>{const M=w();if(!M)return;M.style.
height=Math.min(L,r)+"px",M.classList.toggle("active",L>2),M.classList.toggle("pull-ready",L>=a);const E=k();
E&&(E.textContent=L>=a?"\u96E2\u3057\u3066\u66F4\u65B0":"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},
"applyPullUI"),C=o(()=>{const L=w();L&&(L.style.height="0px",L.classList.remove("active","pull-ready",
"refreshing"),L.classList.remove("dragging"))},"resetPullUI");n.addEventListener("touchstart",L=>{if(v){
h=!1;return}if(n.scrollTop>0){h=!1;return}const M=L.touches[0];M&&(m=M.clientY,y=0,h=!0)},{passive:!0}),
n.addEventListener("touchmove",L=>{if(!h||v)return;if(n.scrollTop>0){h=!1;return}const M=L.touches[0];
if(!M)return;const E=M.clientY-m;if(E<=0){y>0&&(y=0,x(0)),h=!1;return}const D=w();D&&!D.classList.contains(
"dragging")&&D.classList.add("dragging"),y=Math.min(E*c,r),x(y),E>=d&&L.preventDefault()},{passive:!1}),
n.addEventListener("touchend",()=>{if(!h||(h=!1,v))return;const L=w();L&&L.classList.remove("draggin\
g");const M=y>=a;if(y=0,!M){C();return}let E;try{E=t()}catch{E=null}const D=w();if(D){D.classList.add(
"refreshing"),D.style.height=l+"px";const N=D.querySelector(".ptr-pull-label");N&&(N.textContent="\u66F4\u65B0\
\u4E2D...")}E&&typeof E.then=="function"?(v=E,E.catch(()=>{}).finally(()=>{v=null,C()})):(v=Promise.
resolve(),setTimeout(()=>{v=null,C()},400))}),n.addEventListener("touchcancel",()=>{h=!1,y=0,C()})}o(
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
{scroll:!1});const l=r?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",d="",m=[];
if(a){const y=get("prompt-input");c=y?y.value:"",d=y?y.style.height:"",m=currentImageUrls?currentImageUrls.
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
M))}const C=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),L=i;if(L&&
allMessages.some(M=>String(M.id)===String(L))?currentLeafId=Number(L):C&&allMessages.find(M=>String(
M.id)===String(C))?currentLeafId=C:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(r?{silent:r,keepScroll:r}:{silent:r,keepScroll:r,animate:!0}),
a||restoreLatestPromptOptions(),r&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):r||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!r&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const M=get("prompt-input");M&&(M.value=c||"",
d?M.style.height=d:M.style.height="auto"),currentImageUrls=m,currentImageUrls&&currentImageUrls.length?
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
if(l.length){const c=new Set(allMessages.map(m=>m.id)),d=l.filter(m=>!c.has(m.id));d.length&&(allMessages=
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
!d[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null);const m=[];
let h=d[currentLeafId];for(;h;)m.unshift(h),h=d[h.parent_id];const y=buildTokenTotals(m),v=document.
createDocumentFragment();if(threadHasOlderMessages){const x=loadingOlderMessages?"\u8AAD\u307F\u8FBC\u307F\u4E2D...":
"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",C=loadingOlderMessages?"d\
isabled":"",L=document.createElement("div");L.className="mb-3 text-center",L.innerHTML=`<button id="\
load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-gray-600 text-gray-200 hov\
er:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="loadOlderMessages()" ${C}><\
i class="fas fa-clock-rotate-left mr-1"></i>${x}</button>`,v.appendChild(L)}m.forEach(x=>{const C=x.
parent_id?d[x.parent_id]:null,L=C?C.childrenIds:allMessages.filter(E=>!E.parent_id).map(E=>E.id),M=L.
length>1?{current:L.indexOf(x.id)+1,total:L.length,siblings:L}:null;renderMessage(x.id,x.role,x.content,
x.image_url,x.thought_data,x.model,M,n,x.quote_text,x.tokens,x.tokens_in,x.tokens_out,x.is_encrypted,
x.tokens_content,x.tokens_thought,v,!1,x.parent_id,x.skill_name,x.batch_job,x.prompt_options&&typeof x.
prompt_options=="object"?x.prompt_options.image_mask:null,x.processing_time)});const w=currentThreadPending;
if(w&&!isPendingJobSuppressed(w.job_id)){const x=w.message_id,C=new Set(m.map(E=>E.id)),L=m.length?m[m.
length-1]:null;if(x&&C.has(x)&&currentLeafId===x||!x&&L&&L.role==="user"){const E=w.job_id?`pending-${w.
job_id}`:null;renderPendingMessage(v,n,!1,E,w.model||null)}}const k=[];if(i&&!lowBandwidthMode&&Array.
from(v.children).forEach((x,C)=>{const L=r.find(E=>E.id&&E.id===x.id)||(r.length===v.children.length?
r[C]:null),M=L?l[r.indexOf(L)]:0;M>0&&(x.style.minHeight=`${M}px`,k.push(x))}),a.replaceChildren(v),
displayMissingMessageFiles(a,m),updateTotalTokenBar(y.tokens_total,y),currentParentId=currentLeafId,
i&&c!==null?restoreThreadTreeScroll(a,c):scrollToBottom(),lowBandwidthMode)queueMessageDecorations(a,
a&&a.textContent||"");else if(queueHighlight(a),m.length){const x=queueMathTypeset(a,m.map(C=>C.content||
"").join(`
`));Promise.resolve(x).finally(()=>{if(!k.length||!k[0].isConnected)return;const C=a.scrollTop;k.forEach(
L=>{L.style.minHeight=""}),restoreThreadTreeScroll(a,C)})}}o(renderThreadTree,"renderThreadTree");function restoreThreadTreeScroll(e,t){
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
!0,c=get("thread-global-sys-prompt"),d=get("thread-global-sys-prompt-enabled");let m=null;try{m=c||d?
{system_prompt:c?c.value:"",system_prompt_enabled:d?d.checked:!0,apply_global_system_prompt:get("thr\
ead-apply-global-sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(h){sendClientDebugLog("error","Payload c\
onstruction failed: "+h.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const h=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+h.status);let y=!0;if(m){sendClientDebugLog(
"info","Starting POST request for user settings");const v=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)});y=v.ok,v.ok&&window.
applySavedUserSystemPromptSettings(m),sendClientDebugLog("info","POST request finished, status: "+v.
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
null?void 0:y.dataset.threadTitle)||"",c.textContent="",r.disabled=!1;let m=!1;const h=o(()=>{m||(hideModal(
"thread-rename-modal"),d.focus())},"close");l.onclick=h,n.onclick=v=>{v.target===n&&h()},n.onkeydown=
v=>{v.key==="Escape"&&(v.preventDefault(),v.stopPropagation(),h()),v.key==="Tab"&&(v.shiftKey&&document.
activeElement===a?(v.preventDefault(),l.focus()):!v.shiftKey&&document.activeElement===l&&(v.preventDefault(),
a.focus()))},i.onsubmit=async v=>{v.preventDefault();const w=a.value.trim();if(m||!w){w||a.focus();return}
m=!0,r.disabled=!0,l.disabled=!0;try{const k=await apiFetch(CHAT_CONFIG.urls.updateTitle.replace("0",
t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:w})}),x=await k.
json();if(!k.ok)throw new Error("rename failed");currentThreadId===String(t)&&setCurrentChatHeaderTitle(
x.title||w),m=!1,h(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
m=!1,r.disabled=!1,l.disabled=!1}},l.disabled=!1,showModal("thread-rename-modal"),a.focus()}o(renameThread,
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
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(d=>{const m=d.
role==="user",h=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",y=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",v=m?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(d.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(d.content||"")}</div>`,w=buildPdfMessageAttachments(d),k=w.length?`<div \
class="attachments">${w.map(L=>L.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(L.previewUrl)}\
" alt="${pdfEscapeAttr(L.filename)}"><div class="file-caption">${pdfEscapeAttr(L.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(L.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
L.filename)}</span><span class="file-source">${pdfEscapeAttr(L.source)}</span></span></a></div>`).join(
"")}</div>`:"",x=[];d.model&&!m&&x.push(d.model),d.tokens!==null&&d.tokens!==void 0&&x.push(`tokens:${d.
tokens}`),d.tokens_in!==null&&d.tokens_in!==void 0&&x.push(`in:${d.tokens_in}`),d.tokens_out!==null&&
d.tokens_out!==void 0&&x.push(`out:${d.tokens_out}`),d.tokens_thought!==null&&d.tokens_thought!==void 0&&
x.push(`thought:${d.tokens_thought}`),d.is_encrypted&&x.push("encrypted"),d.parent_id!==null&&d.parent_id!==
void 0&&x.push(`parent:#${d.parent_id}`);const C=x.length?`<div class="message-meta">${pdfEscapeAttr(
x.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${m?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${m?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${m?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(d.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${h}
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
some(L=>maybeNeedsMathJax(L.content)||maybeNeedsMathJax(L.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),h.fonts&&h.fonts.ready)try{await h.fonts.ready}catch{}t.update(60);const k=Array.from(h.
images||[]),x=Promise.all(k.map(L=>L.complete?Promise.resolve():new Promise(M=>{L.addEventListener("\
load",M,{once:!0}),L.addEventListener("error",M,{once:!0})})));if(await Promise.race([x,new Promise(
L=>setTimeout(L,5e3))]),t.update(80),h.getElementById("MathJax-script")){let L=0;for(;L<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(M=>setTimeout(M,50)),L++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(M){console.error(
"PDF MathJax typeset failed",M)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(a),m=new Blob([d],{type:"text/html"});r.src=URL.createObjectURL(
m),document.body.appendChild(r)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}o(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}o(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}o(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=o((d,m)=>i.
compare(d.filename||"",m.filename||""),"nameAsc"),r=o((d,m)=>i.compare(m.filename||"",d.filename||""),
"nameDesc"),l=o((d,m)=>(Number(m.ts)||0)-(Number(d.ts)||0),"tsDesc"),c=o((d,m)=>(Number(d.ts)||0)-(Number(
m.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,m)=>a(d,m)||l(d,m)):t==="name_desc"?n.sort((d,m)=>r(
d,m)||l(d,m)):t==="oldest"?n.sort((d,m)=>c(d,m)||a(d,m)):n.sort((d,m)=>l(d,m)||a(d,m)),n}o(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}o(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}o(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}o(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}o(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const m=t.querySelector(".lib-empty-state");m&&m.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const m=get("lib-total-count");m&&(m.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),r=i.filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(
a)),l=get("lib-total-count");if(l){const m=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?l.innerText=`${lib.files.length} / ${m} files`:l.innerText=`${m} files`}if(!r.length){
if(n)return;const m=lib.favoritesOnly&&!a?"fa-star":"fa-search",h=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${m}"></i></div>\
<p class="lib-empty-title">${h}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(a)):r).forEach(m=>{
try{const h=renderLibraryItem(m,c++);t.appendChild(h)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
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
a-star",m=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
h=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${m}\
" aria-label="${m}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${h}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",x=>{x.shiftKey&&x.preventDefault()}),n.onclick=
x=>{if(x&&x.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const C=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),L=C.findIndex(E=>E.dataset.filepath===lib.anchorPath),
M=C.indexOf(n);if(L!==-1&&M!==-1){const E=Math.min(L,M),D=Math.max(L,M);for(let N=E;N<=D;N++){const G=C[N].
dataset.filepath;G&&(lib.selected.add(G),C[N].classList.add("is-selected"))}try{const N=window.getSelection&&
window.getSelection();N&&N.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(x=>{x.onclick=C=>{C.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async x=>{x.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const k=n.querySelector(".lib-favorite-btn");return k&&
(k.onclick=async x=>{x.stopPropagation(),k.disabled=!0;try{const C=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
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
return}if(i.innerHTML="",c.forEach(d=>{const m=document.createElement("div");m.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const h=d.updated_at?new Date(d.
updated_at).toLocaleString():"";m.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(h)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=m.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(m)}),l.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const d=getLibSortOrder(),m=getLibSearchQuery(),h=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(h),sort:d,q:m,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);r=await v.json(),l=!0}catch(d){i=d}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const d=FILE_BASE_URL,
m=FILE_THUMB_BASE_URL,h=new Set(c.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(c.length>=LIBRARY_PAGE_SIZE||!v||h.has(v))return;const w=getAttachmentNameForPath(
v)||v.split("/").pop()||v,k=(w.split(".").pop()||"").toLowerCase(),x=["png","jpg","jpeg","webp","gif"].
includes(k)?"image":"file",C=x==="image"?m+v:null;c.unshift({filename:w,original_filename:w,filepath:v,
url:d+v,thumbnail_url:C,type:x,ext:k,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),h.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=c.filter(h=>h&&h.filepath&&
h.url);let m=[];if(e){const h=new Set(lib.files.map(y=>y.filepath));m=d.filter(y=>!h.has(y.filepath)),
lib.files.push(...m)}else lib.files=d;lib.files.forEach(h=>{h&&h.filepath&&setAttachmentNameForPath(
h.filepath,h.filename||h.original_filename||"")}),lib.fileSet=new Set(lib.files.map(h=>h.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
m:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
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
a);if(!c)return;const d=(lib.files||[]).find(m=>m&&m.filepath===a);d&&d.filename&&setAttachmentNameForPath(
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
threadFixedBranchId,m=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),h=getCumulativeTokensForNode(
a.id);if(l.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,l.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(m)}</div>
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
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);r.innerHTML="",Object.entries(l).sort((d,m)=>m[1].
total-d[1].total).forEach(([d,m])=>{const h=document.createElement("div");h.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",h.innerHTML=`
                    <div class="flex justify-between font-bold text-gray-300 mb-1">
                        <span class="truncate pr-2">${d}</span>
                        <span class="text-blue-400 shrink-0">${m.total}</span>
                    </div>
                    <div class="grid grid-cols-3 gap-1 text-[9px] text-gray-500 font-mono">
                        <div title="Input tokens">In: ${m.in}</div>
                        <div title="Output tokens">Out: ${m.out}</div>
                        <div title="Thought/Reasoning tokens">${m.thought>0?`Th: ${m.thought}`:""}</\
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
batchProviderLabel(a.provider)),d=escapeHtml(a.model||""),m=escapeHtml(batchFormatTime(a.created_at)),
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
                                <span><i class="fas fa-history mr-1"></i>${m}</span>
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
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),d=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),m=o(h=>{h.key==="Enter"&&(h.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",m),r.onclick=async()=>{
const h=get("api-key-modal-input").value.trim();if(!h){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=h);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:h})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=h)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",m),d(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",m),d(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",m),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const h=get("api-key-modal-input");h&&h.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const d=c.map(m=>{try{return m instanceof Error?m.stack||m.message:typeof m=="object"?JSON.
stringify(m):String(m)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,d)}catch{}finally{
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
detail"),"view"),m=o(()=>get("bot-log-list-view"),"listView"),h=o($=>{if(!$)return"-";const R=new Date(
$);return isNaN(R.getTime())?String($):R.toLocaleString("ja-JP")},"formatTime"),y=o($=>{const R=Math.
max(0,Math.floor(Number($)||0));return R>=60?`${Math.floor(R/60)}\u5206${R%60?R%60+"\u79D2":""}`:`${R}\
\u79D2`},"formatDuration"),v=o(($,R)=>R==null||R===""?"-":$==="lock_source"?a[R]||String(R):$==="cli\
ent"||$==="origin_client"?n[R]||String(R):$==="action"?r[R]||String(R):$==="by"?R==="self"?"\u672C\u4EBA":
R==="admin"?"\u7BA1\u7406\u8005":String(R):$==="enabled"?R?"ON":"OFF":$==="remaining_seconds"||$==="\
lock_seconds"?y(R):$==="applied_to"&&Array.isArray(R)?R.map(U=>a[U]||U).join("\u30FB"):typeof R=="ob\
ject"?JSON.stringify(R,null,2):String(R),"formatDetailValue"),w=o($=>{const R=String($||"").trim();if(!R)
return"";let U=null;if(R.startsWith("{"))try{U=JSON.parse(R)}catch{U=null}if(U&&typeof U=="object"&&
!Array.isArray(U)){const P=Object.keys(U).map(q=>{const ee=v(q,U[q]),B=i[q]||q;return ee.length>120||
ee.includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(B)}</div><pre class="bot-log-pre">${escapeHtml(
ee)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(B)}:</span> <span class="text-gray-\
300">${escapeHtml(ee)}</span></div>`}).join("");return R.length>600?`<details class="mt-1"><summary \
class="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${P}</details>`:`<div c\
lass="mt-1 space-y-1">${P}</div>`}return R.length>300?`<details class="mt-1"><summary class="cursor-\
pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(
R)}</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(R)}</div>`},"renderD\
etails"),k=o(async($,R)=>{const U=await apiFetch($,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(R)}),P=await U.json().catch(()=>({}));if(!U.ok)throw new Error(P.error||
String(U.status));return P},"postJson"),x=o(($,R)=>`<span class="${R} text-white px-2 py-0.5 rounded\
">${escapeHtml($)}</span>`,"badge"),C=o(()=>{const $=get("bot-log-account-list");if(!$)return;const R=c.
accounts,U=R.length&&R.every(q=>c.selected.has(q.user_id)),P=get("bot-log-list-toolbar");if(P&&(P.innerHTML=
`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${U?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!R.length){$.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}$.innerHTML=R.map(q=>{const ee=[];q.exists||ee.push(x("\u524A\u9664\u6E08\u307F","bg-gray-600")),
q.is_bot_banned&&ee.push(x("BAN\u4E2D","bg-red-600")),q.lock_remaining_seconds>0&&ee.push(x(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
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
/span>${ee.join("")}</div>
                                <div class="text-[10px] text-gray-500">${B}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${q.user_id}" data-username="${escapeHtml(
q.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),L=o(async()=>{const $=get("bot-log-acc\
ount-list");if(!$)return;$.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const R=get("bot-log-search"),U=R?R.value.
trim():"";try{const P=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(U)}`,{cache:"\
no-store"}),q=await P.json().catch(()=>({}));if(!P.ok)throw new Error(q.error||String(P.status));c.accounts=
q.accounts||[];const ee=new Set(c.accounts.map(B=>B.user_id));Array.from(c.selected).forEach(B=>{ee.
has(B)||c.selected.delete(B)}),C()}catch{$.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\
\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),M=o($=>{if(!$)return"";const R=$.locks||[],U=$.banned_identifiers||[],
P=[];$.exists||P.push(x("\u524A\u9664\u6E08\u307F","bg-gray-600")),$.exists&&(P.push(x($.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",$.detection_enabled?"bg-gray-600":"bg-gray-700")),P.push(x($.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",$.is_bot_banned?"bg-red-600":"bg-gray-600"))),P.push(x(R.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",R.length?"bg-yellow-600":"bg-gray-600")),U.length&&P.push(x(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${U.length}\u4EF6`,"bg-red-600")),$.is_admin&&P.push(x("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const q=!!(l.counts.ban||l.counts.related_ban),ee=$.is_bot_banned?`<div class="text-\
gray-300">BAN\u306E\u7406\u7531: ${escapeHtml($.bot_ban_reason||"-")}\uFF08${escapeHtml(h($.bot_banned_at))}\
\uFF09</div>${q?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",B=R.map(Q=>{const fe=a[Q.source]||Q.source,ue=Q.identifier?` ${escapeHtml(Q.identifier)}${Q.source===
"cookie"?"\u2026":""}`:"",Ge=Q.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
Q.origin.username||String(Q.origin.user_id||"-"))}\uFF08${escapeHtml(n[Q.origin.client]||Q.origin.client||
"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(fe)}${ue}\u30FB\u6B8B\u308A${escapeHtml(
y(Q.remaining_seconds))}\u30FB${escapeHtml(Q.reason||"")}${Ge}</div>`}).join(""),K=U.map(Q=>`<div cl\
ass="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[Q.kind]||Q.kind)} ${escapeHtml(Q.identifier)}${Q.
kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(Q.reason||"-")}\uFF08${escapeHtml(h(Q.created_at))}\uFF09<\
/div>`).join(""),ce=U.length?'<div class="text-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002\
</div>':"",de=$.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${y($.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",te=$.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String($.appeal_count))}\u4EF6`:"",re=[];return R.length&&re.push('<button class="bot-log-unlock bg-\
yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
U.length&&re.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 r\
ounded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${P.join("")}</div>
                        ${ee}
                        ${B}
                        ${K}
                        ${ce}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String($.lock_count))} / ${escapeHtml(String($.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(de)}\
\u30FB\u5931\u6557 ${escapeHtml(String($.turnstile_fail_count))} / ${escapeHtml(String($.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round(($.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round(($.behavior_score||0)*10)/10))}\uFF09${te}</div>
                        ${re.length?`<div class="flex flex-wrap gap-2 pt-1">${re.join("")}</div>`:""}\

                    </div>`},"renderState"),E=o(()=>{const $=o((U,P,q)=>`<button class="bot-log-filt\
er ${l.type===U?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(U)}">${escapeHtml(P)} (${q})</button>`,"chip"),R=[$("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((U,P)=>l.counts[P]-l.counts[U]).forEach(U=>{R.push($(U,(t[U]||
{label:U}).label,l.counts[U]))}),R.join("")},"renderFilters"),D=o($=>{const R=t[$.event_type]||{label:$.
event_type,cls:"bg-gray-600"},U=$.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[$.client]||$.client)}</span>`:"",P=l.selected.has($.id)?"checked":"",q=$.score!==
null&&$.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String($.score))}${$.
behavior_score!==null&&$.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String($.behavior_score))}\
\uFF09`:""}</span>`:"",ee=$.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
$.reasons)}</div>`:"",B=$.ip_address||$.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${$.ip_address?"IP "+escapeHtml($.ip_address):""}${$.ip_address&&$.user_agent?"\u30FB":""}${escapeHtml(
$.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${$.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${$.id}" ${P}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(h($.created_at))}</span>
                            <span class="${R.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(R.label)}\
</span>
                            ${U}
                            ${q}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${$.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${ee}
                        ${w($.details)}
                        ${B}
                    </div>`},"renderItem"),N=o(()=>{const $=d();if(!$)return;const R=l.items.length?
l.items.map(D).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
U=l.items.length&&l.items.every(P=>l.selected.has(P.id));$.innerHTML=`
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
                        <div class="space-y-2">${R}</div>
                        ${l.hasMore?'<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-60\
0 text-white px-2 py-1.5 rounded text-xs">\u3055\u3089\u306B\u8AAD\u307F\u8FBC\u3080</button>':""}
                    </div>`},"render"),G=o(async($=!1)=>{const R=d();if(!R||!l.userId)return;$||(R.innerHTML=
'<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');
const U=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&U.set("type",l.type),
$&&l.items.length&&U.set("before_id",String(l.items[l.items.length-1].id));try{const P=await apiFetch(
`/api/bot/evidence?${U.toString()}`,{cache:"no-store"}),q=await P.json().catch(()=>({}));if(!P.ok)throw new Error(
q.error||String(P.status));l.counts=q.counts||{},l.total=q.total||0,l.hasMore=!!q.has_more,q.user&&(l.
username=q.user.username||l.username,l.exists=!!q.user.exists),$?l.items=l.items.concat(q.items||[]):
(l.items=q.items||[],l.data=q),N()}catch{$||(R.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),j=o(async($,R)=>{if(confirm(R))try{const U=await k("/api/bot/evidence/delete",
Object.assign({user_id:l.userId},$));showToast(`${U.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await G(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),J=o(async($,R,U)=>{if(confirm(R))try{await k($,{user_id:l.userId}),showToast(
U,"success"),await G(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","e\
rror",!0)}},"runAccountAction"),oe=o(()=>{const $=d();$&&($.classList.add("hidden"),$.classList.remove(
"flex"),$.innerHTML="");const R=m();R&&(R.classList.remove("hidden"),R.classList.add("flex")),l.userId=
0},"showList"),me=o(async()=>{oe(),await L()},"openList"),ie=o(async($,R="")=>{const U=d();if(!U)return;
Object.assign(l,{userId:Number($)||0,username:R,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),l.selected.clear();const P=m();P&&(P.classList.add("hidden"),P.classList.remove("flex")),
U.classList.remove("hidden"),U.classList.add("flex"),await G(!1)},"open");return{open:ie,openList:me,
showList:oe,bind:o(()=>{const $=get("bot-log-modal");if(!$||$.dataset.bound==="1")return;$.dataset.bound=
"1";const R=get("bot-log-search");R&&R.addEventListener("keydown",U=>{U.key==="Enter"&&L()}),$.addEventListener(
"change",U=>{const P=U.target;if(!(!P||!P.classList))if(P.classList.contains("bot-log-select")){const q=Number(
P.getAttribute("data-log-id"));P.checked?l.selected.add(q):l.selected.delete(q),N()}else if(P.classList.
contains("bot-log-select-all"))l.items.forEach(q=>{P.checked?l.selected.add(q.id):l.selected.delete(
q.id)}),N();else if(P.classList.contains("bot-log-account-select")){const q=Number(P.getAttribute("d\
ata-user-id"));P.checked?c.selected.add(q):c.selected.delete(q),C()}else P.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(q=>{P.checked?c.selected.add(q.user_id):c.selected.delete(
q.user_id)}),C())}),$.addEventListener("click",async U=>{const P=U.target.closest("button");if(!(!P||
P.disabled))if(P.id==="bot-log-search-btn"||P.id==="bot-log-refresh-btn")P.id==="bot-log-refresh-btn"&&
R&&(R.value=""),await L();else if(P.classList.contains("bot-log-open-account")){const q=Number(P.getAttribute(
"data-user-id"));q&&await ie(q,P.getAttribute("data-username")||"")}else if(P.classList.contains("bo\
t-log-delete-accounts")){const q=Array.from(c.selected);if(!q.length||!confirm(`\u9078\u629E\u3057\u305F${q.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const ee=await k("/api/bot/evidence/delete",{user_ids:q});showToast(`${ee.deleted||0}\u4EF6\u306E\u8A18\u9332\
\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await L()}catch{showToast(
"\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}}else if(P.
classList.contains("bot-log-back"))await me();else if(P.classList.contains("bot-log-reload"))await G(
!1);else if(P.classList.contains("bot-log-more"))await G(!0);else if(P.classList.contains("bot-log-f\
ilter"))l.type=P.getAttribute("data-type")||"",l.selected.clear(),await G(!1);else if(P.classList.contains(
"bot-log-delete-one")){const q=Number(P.getAttribute("data-log-id"));q&&await j({ids:[q]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\
\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
P.classList.contains("bot-log-delete-selected")?l.selected.size&&await j({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
P.classList.contains("bot-log-delete-all")?await j({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
P.classList.contains("bot-log-unlock")?await J("/api/bot/account/clear-lock",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):P.classList.contains("bot-log-unbloc\
k")&&await J("/api/bot/account/clear-identifiers",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
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
d.title=c.split("/").pop();const m=l.closest(".chat-image-frame")||(l.tagName==="SOURCE"?l.parentNode:
l);m&&a.contains(m)&&m.replaceWith(d)})}}o(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}o(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},r=1/0;for(let l=1;l<=i;l++){if(i%l)continue;const c=i/l,d=Math.abs(
Math.log(Math.max(1,t)/l/(Math.max(1,n)/c)));(d<r-1e-9||Math.abs(d-r)<=1e-9&&(t>=n?l>a.columns:c>a.rows))&&
(a={columns:l,rows:c},r=d)}return a}o(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,r=Math.round(e/n.columns*a/2),l=Math.round(t/n.rows*a/2),c=[];for(let d=0;d<
n.rows;d++)for(let m=0;m<n.columns;m++){const h=Math.floor(m*e/n.columns),y=Math.floor((m+1)*e/n.columns),
v=Math.floor(d*t/n.rows),w=Math.floor((d+1)*t/n.rows);c.push({index:c.length,row:d,column:m,cellLeft:h,
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
getContext("2d"),c=e.width/n,d=e.height/i;l.drawImage(t,0,0,e.width,e.height);const m=r[0],h=m.right-
m.cellRight,y=m.bottom-m.cellBottom;l.fillStyle="rgba(250,204,21,0.18)";const v=[];for(let k=1;k<a.columns;k++){
const x=Math.floor(k*n/a.columns);l.fillRect((x-h)*c,0,2*h*c,e.height),v.push([x*c,0,x*c,e.height])}
for(let k=1;k<a.rows;k++){const x=Math.floor(k*i/a.rows);l.fillRect(0,(x-y)*d,e.width,2*y*d),v.push(
[0,x*d,e.width,x*d])}drawSplitLines(l,v,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const w=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));l.font=`bold ${w}px monospace`,
l.textAlign="center",l.textBaseline="middle";for(const k of r){const x=(k.cellLeft+k.cellRight)/2*c,
C=(k.cellTop+k.cellBottom)/2*d;l.lineWidth=Math.max(2,w/10),l.strokeStyle="rgba(0,0,0,0.7)",l.strokeText(
String(k.index+1),x,C),l.fillStyle="#facc15",l.fillText(String(k.index+1),x,C)}}o(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,r,l,c])=>{e.moveTo(a,r),e.lineTo(l,c)}),e.stroke(),e.restore()}o(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(r=>r?i(
r):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}o(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const r=a.
naturalWidth,l=a.naturalHeight,c=chooseSplitGrid(t.pieces,r,l);if(!splitFits(c,r,l))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const d=splitTiles(r,l,c,t.overlap),m=splitBaseName(e.name),h=[],y=o(async v=>{h.push(await applyImageCompression(
v,n))},"append");if(t.includeOriginal)if(t.markOriginal){const v=splitCanvas(r,l);drawSplitOverview(
v,a,r,l,c,d),await y(await splitCanvasFile(v,`${m}_split_grid.png`))}else await y(e);for(const v of d){
const w=v.right-v.left,k=v.bottom-v.top;let x=Math.max(12,Math.min(40,Math.max(w,k)/45)),C=0;const L=splitTileCaption(
v,c,r,l);if(t.markPieces){const D=splitCanvas(1,1),N=D.getContext("2d");N.font=`bold ${x}px monospac\
e`,x=Math.max(6,x*Math.min(1,w*.96/N.measureText(L).width)),C=Math.max(12,Math.floor(x*1.8)),D.width=
0}const M=splitCanvas(w,k+C),E=M.getContext("2d");if(C&&(E.fillStyle="#111827",E.fillRect(0,0,w,C)),
E.drawImage(a,v.left,v.top,w,k,0,C,w,k),t.markPieces){E.fillStyle="#fff",E.font=`bold ${x}px monospa\
ce`,E.textBaseline="middle",E.fillText(L,w*.02,C/2);const D=[];for(const N of[v.cellLeft,v.cellRight])
N>v.left&&N<v.right&&D.push([N-v.left,C,N-v.left,C+k]);for(const N of[v.cellTop,v.cellBottom])N>v.top&&
N<v.bottom&&D.push([0,C+N-v.top,w,C+N-v.top]);drawSplitLines(E,D,Math.max(1.5,Math.min(4,Math.max(w,
k)/500)),!0)}await y(await splitCanvasFile(M,splitTileName(m,v,c)))}return h}finally{i.release()}}o(
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
E=>E.right))-l,height:Math.max(...r.map(E=>E.bottom))-c},m=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,h={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},y=document.createElement("div");
y.hidden=!0,y.className="prompt-send-held-files",y.setAttribute("aria-hidden","true"),y.append(...h.
rows),document.body.appendChild(y);let v=!1,w=!1,k=!1;const x=o(()=>{v=!0},"onInput");t.addEventListener(
"input",x),t.value="",["file-input","photo-input","mask-input"].forEach(E=>{const D=get(E);D&&(D.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let C;i&&e&&!m&&
i.animate&&(C=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const L=o(()=>{!w||k||(h.files.forEach(E=>{var N,G;const D=(G=
(N=E==null?void 0:E.rowObj)==null?void 0:N.row)==null?void 0:G.getAttribute("data-local-url");D&&URL.
revokeObjectURL(D)}),y.remove())},"releaseFiles"),M=o((E,D=!0)=>{w||(w=!0,E=E&&h.navigation===threadLoadSequence,
k=E,t.removeEventListener("input",x),E?(!v&&!t.value&&(t.value=h.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...h.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=h.mask),h.files.forEach((N,G)=>browserFastLocalFiles.set(G,N)),h.
markers.forEach(N=>markerAppliedUploads.add(N)),n&&n.prepend(...h.rows),y.remove(),!currentQuote&&h.
quote&&(currentQuote=h.quote,currentQuoteMessageId=h.quoteId,renderQuoteBarContent(h.quote),get("quo\
te-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=h.progress),updateFilePreview(),
updateMaskPreview(),schedulePromptTokenEstimate(!0)):D&&L())},"finish");return{source:d,canRetry:o(()=>!v&&
h.navigation===threadLoadSequence,"canRetry"),stopDock:o(()=>C==null?void 0:C.cancel(),"stopDock"),files:h.
files,releaseFiles:L,accept:o((E=!0)=>M(!1,E),"accept"),restore:o(()=>M(!0),"restore")}}o(takePromptForSend,
"takePromptForSend");function animatePromptToMessage(e,t,n){var y;let i,a=0,r=!1;const l=[],c=get("c\
hat-container"),d=get("prompt-input"),m=window.matchMedia("(prefers-reduced-motion: reduce)"),h=o(()=>{
var v,w;r||(r=!0,(v=window.ActivityLog)==null||v.log("ui.hide"),t.stopDock(),cancelAnimationFrame(a),
l.forEach(k=>k.cancel()),i==null||i.remove(),e==null||e.classList.remove("prompt-send-hidden"),window.
removeEventListener("resize",h),window.removeEventListener("popstate",h),window.removeEventListener(
"pagehide",h),document.removeEventListener("visibilitychange",h),c==null||c.removeEventListener("whe\
el",h),c==null||c.removeEventListener("touchstart",h),d==null||d.removeEventListener("input",h),(w=window.
visualViewport)==null||w.removeEventListener("resize",h),m.removeEventListener("change",h))},"stop");
if(!n||m.matches||!e||!Element.prototype.animate)return h;try{e.classList.remove("fade-in"),e.classList.
add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var v;try{if(!e.isConnected)return h();const w=e.
querySelector(".message-bubble"),k=w.getBoundingClientRect(),x=c.getBoundingClientRect();if(!k.width||
!k.height||k.bottom<=x.top||k.top>=x.bottom)return h();const C=t.source;i=w.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach(G=>G.removeAttribute("id")),i.querySelectorAll(".msg-contro\
ls, video, audio").forEach(G=>G.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute("ari\
a-hidden","true"),i.inert=!0,Object.assign(i.style,{width:k.width+"px",height:k.height+"px",left:k.left+
"px",top:k.top+"px"}),document.body.appendChild(i),(v=window.ActivityLog)==null||v.log("ui.show");const L=Math.
min(C.width/k.width,Math.max(C.height,42)/k.height,1),M=C.left+(C.width-k.width*L)/2-k.left,E=C.top+
(C.height-k.height*L)/2-k.top,D=i.animate([{transform:`translate(${M}px, ${E}px) scale(${L})`,borderRadius:"\
12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"translate(0, 0\
) scale(1)",borderRadius:getComputedStyle(w).borderRadius,backgroundColor:getComputedStyle(w).backgroundColor}],
{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});l.push(D),l.push(i.animate([{opacity:1,
offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),e.classList.remove("prompt\
-send-hidden"),l.push(w.animate([{opacity:0,offset:0},{opacity:0,offset:.8},{opacity:1}],{duration:350,
fill:"both"}));const N=o(()=>{if(!e.isConnected||!userAutoScroll||document.hidden)return h();const G=w.
getBoundingClientRect();i.style.left=G.left+"px",i.style.top=G.top+"px",a=requestAnimationFrame(N)},
"watch");a=requestAnimationFrame(N),D.finished.then(h,h)}catch{console.warn("Prompt send animation f\
ailed"),h()}}),window.addEventListener("resize",h),window.addEventListener("popstate",h),window.addEventListener(
"pagehide",h),document.addEventListener("visibilitychange",h),c.addEventListener("wheel",h,{passive:!0}),
c.addEventListener("touchstart",h,{passive:!0}),d.addEventListener("input",h),(y=window.visualViewport)==
null||y.addEventListener("resize",h),m.addEventListener("change",h)}catch{console.warn("Prompt send \
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
cache_write_tokens);let m=0;for(const h of["5m","1h","30m","12h"]){const y=costTokenNumber(l[`cache_\
write_${h}_tokens`]);y!==null&&(r[h]=(r[h]||0)+y,m+=y,a=!0)}(c!==null||d!==null)&&(a=!0),n+=c||0,i+=
Math.max(d||0,m)}return a?{read:n,write:i,ttls:r}:null}o(costCacheUsage,"costCacheUsage");function costRate(e,t,n){
const i=e&&(e.aliases[t]||t),a=e&&e.models[i];if(!a)return null;const r={...a};for(const l of r.changes||
[])n>=l.from&&Object.assign(r,l);return r}o(costRate,"costRate");function addRequestCost(e,t,n,i,a=!1){
var h,y;if(e.requests++,String(t.model).startsWith("grok-")&&Number.isSafeInteger(t.cost_in_usd_ticks)&&
t.cost_in_usd_ticks>=0){e.low+=t.cost_in_usd_ticks/1e10,e.high+=t.cost_in_usd_ticks/1e10,e.priced++,
t.complete===!1&&e.unpriced++;return}const r=costRate(n,t.model,i);if(!r){e.unpriced++;return}if(r.unit===
"image"){const v=costTokenNumber(t.images),w=(h=r.image_rates)==null?void 0:h[t.operation],k=w==null?
void 0:w[r.image_selector==="speed"?t.speed:t.quality],x=(y=k==null?void 0:k.minimum)!=null?y:w==null?
void 0:w.minimum,C=typeof w=="number"?w:typeof k=="number"?k:x,L=Number.isFinite(C)&&typeof w!="numb\
er"&&typeof k!="number";if(v===null||!Number.isFinite(C)){e.unpriced++;return}e.low+=v*C,e.high+=v*C,
e.priced++,L&&(e.lowerBound=!0),(t.complete===!1||L)&&e.unpriced++;return}const l=costTokenNumber(t.
input_tokens),c=costTokenNumber(t.output_tokens);if(l===null&&c===null){e.unpriced++;return}const d=r.
threshold!=null&&(r.threshold_inclusive?(l||0)>=r.threshold:(l||0)>r.threshold);d&&e.long++;const m=requestTokenCost(
t,r,d,a);e.low+=m.low,e.high+=m.high,m.lowerBound&&(e.lowerBound=!0),e.priced++,(l===null||c===null||
t.complete===!1||m.incomplete)&&e.unpriced++}o(addRequestCost,"addRequestCost");function legacyImageRequest(e,t){
let n=e.image_url;try{typeof n=="string"&&(n=JSON.parse(n))}catch{n=n?[n]:[]}const i=Array.isArray(n)?
n.length:0,a=t.get(String(e.parent_id)),r=(a==null?void 0:a.prompt_options)||{};let l=a==null?void 0:
a.image_url,c=Array.isArray(l)?l.length>0:!!l&&l!=="[]"&&l!=="null",d=a;const m=new Set;for(;!c&&(d==
null?void 0:d.parent_id)!=null&&!m.has(String(d.id));)m.add(String(d.id)),d=t.get(String(d.parent_id)),
(d==null?void 0:d.role)==="assistant"&&d.image_url&&d.image_url!=="[]"&&d.image_url!=="null"&&(c=!0);
const h=["remove_object","remove_background"].includes(r.ideogram_operation)?r.ideogram_operation:c&&
e.model==="ideogram-4.5"?"edit":"generate";let y=r.ideogram_quality;(!["very_low","low","medium","hi\
gh"].includes(y)||y==="very_low"&&h==="generate")&&(y=h==="edit"?"medium":"high");const v=a?["turbo",
"default","quality"].includes(r.ideogram_speed)?r.ideogram_speed:"default":null;return{model:e.model,
images:i||null,operation:h,quality:y,speed:v,complete:!!a}}o(legacyImageRequest,"legacyImageRequest");
function estimateConversationCost(e,t=costPricing,n=e){var l,c,d,m,h,y;const i={low:0,high:0,priced:0,
unpriced:0,inferred:0,long:0,requests:0},a=new Map((n||[]).filter(Boolean).map(v=>[String(v.id),v])),
r=new Set;for(const v of e||[]){if(!v||v.role!=="assistant"||r.has(String(v.id)))continue;r.add(String(
v.id));let w=null;try{w=typeof v.thought_data=="string"?JSON.parse(v.thought_data):v.thought_data}catch{}
const k=((l=/^\d{4}-\d{2}-\d{2}/.exec(v.created_at||""))==null?void 0:l[0])||(t==null?void 0:t.checked_at),
x=((c=v.batch_job)==null?void 0:c.state)==="JOB_STATE_SUCCEEDED";if(Array.isArray(w==null?void 0:w.pricing_calls)){
w.pricing_calls.length||i.priced++;for(const E of w.pricing_calls){if(!E||typeof E!="object"){i.unpriced++;
continue}const D=((d=/^\d{4}-\d{2}-\d{2}/.exec(E.created_at||""))==null?void 0:d[0])||k,N=costRate(t,
E.model,D),G=(N==null?void 0:N.image_selector)==="speed"&&!E.speed&&E.model===v.model?{...E,speed:legacyImageRequest(
v,a).speed}:E;addRequestCost(i,G,t,D,x)}continue}if(((m=costRate(t,v.model,k))==null?void 0:m.unit)===
"image"){addRequestCost(i,legacyImageRequest(v,a),t,k);continue}const C=costTokenNumber(v.tokens_out);
if(C===null&&!v.content&&!v.tokens)continue;if(C===null){i.unpriced++;continue}let L=(h=costTokenNumber(
w==null?void 0:w.pricing_input_tokens))!=null?h:costTokenNumber(v.tokens_in),M=!1;if(L===null){i.inferred++,
L=0;let E=v.parent_id;const D=new Set([String(v.id)]);for(E==null&&(M=!0);E!=null;){const N=String(E);
if(D.has(N)||!a.has(N)){M=!0;break}D.add(N);const G=a.get(N),j=G.role==="user"?costTokenNumber(G.tokens_in):
(y=costTokenNumber(G.tokens_content))!=null?y:costTokenNumber(G.tokens_out)===null?null:Math.max(0,Number(
G.tokens_out)-(costTokenNumber(G.tokens_thought)||0));j===null?M=!0:L+=j,E=G.parent_id}}addRequestCost(
i,{model:v.model,input_tokens:L,output_tokens:C,complete:!M},t,k,x)}return i}o(estimateConversationCost,
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
refreshCostBar()});function requestTokenCost(e,t,n,i){var P,q,ee,B,K,ce,de,te,re,Q;const a=costTokenNumber(
e.input_tokens)||0,r=costTokenNumber(e.output_tokens)||0,l=costTokenNumber(e.cache_read_tokens)||0,c=costTokenNumber(
e.cache_write_tokens)||0,d=["5m","1h","30m","12h"],m=d.map(fe=>costTokenNumber(e[`cache_write_${fe}_\
tokens`])||0),h=Math.max(c,m.reduce((fe,ue)=>fe+ue,0)),y=Math.max(0,a-l-h),v=n?t.long_input:t.input,
w=n?t.long_output:t.output,k=n&&(P=t.long_cached_input)!=null?P:t.cached_input,x=t.cache_write_multipliers||
{},C=i&&(q=t.batch_multiplier)!=null?q:1,L=i?(ee=t.batch_cached_multiplier)!=null?ee:C:1,M=k!=null&&
C>0?L/C:1,E=Math.max(0,h-m.reduce((fe,ue)=>fe+ue,0)),D=Object.values(x);let N=!!e.pricing_incomplete||
l+h>a||!!e.explicit_cache;l&&k==null&&(N=!0);let G=y*v+l*(k!=null?k:v)*M,j=y*(n?v:(B=t.max_input)!=null?
B:v)+l*(n?k!=null?k:v:(ce=(K=t.max_cached_input)!=null?K:k)!=null?ce:v)*M;for(let fe=0;fe<d.length;fe++){
const ue=x[d[fe]];m[fe]&&ue==null&&(N=!0),G+=m[fe]*v*(ue!=null?ue:1),j+=m[fe]*(n?v:(de=t.max_input)!=
null?de:v)*(ue!=null?ue:1)}E&&(N=!0,G+=E*v*(D.length?Math.min(...D):1),j+=E*(n?v:(te=t.max_input)!=null?
te:v)*(D.length?Math.max(...D):1));let J=(G+r*w)*C/1e6,oe=(j+r*((re=t.max_output)!=null?re:w))*C/1e6,
me=!1;const ie=t.tool_rates||{},ye=o((fe,ue)=>{if(fe){if(ie[ue]==null){N=!0;return}J+=fe*ie[ue],oe+=
fe*ie[ue]}},"add"),$=costTokenNumber(e.web_search_calls)||0;ye($,e.search_preview?"web_search_previe\
w":"web_search"),$&&!e.search_preview&&t.search_content_tokens&&(J+=$*t.search_content_tokens*v/1e6,
oe+=$*t.search_content_tokens*v/1e6),ye(costTokenNumber(e.file_search_calls),"file_search");const R=costTokenNumber(
e.grounding_queries)||0;ye(ie.grounding_prompt!=null?Math.min(1,R):R,ie.grounding_prompt!=null?"grou\
nding_prompt":"grounding_query");const U=costTokenNumber(e.code_execution_calls)||0;return U&&!e.code_execution_free&&
(t.provider==="openai"?(ye(((Q=costTokenNumber(e.container_sessions))!=null?Q:1)*(costTokenNumber(e.
container_memory_gb)||1),"container_session"),N=!0):t.provider==="anthropic"?(ye(1,"code_execution_m\
inimum"),N=!0,me=!0):ye(U,"code_execution")),{low:J,high:oe,incomplete:N,lowerBound:me}}o(requestTokenCost,
"requestTokenCost");
