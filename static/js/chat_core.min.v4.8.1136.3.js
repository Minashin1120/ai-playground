function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),r=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const C=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",C.forEach((A,I)=>{
const J=A.getAttribute("data-local-url"),D=A.getAttribute("data-filename"),ie=A.querySelector("img.u\
pload-preview")!==null;let j;if(ie){let Q=J;if(!Q&&D){const ce=D.replace(/^\d+\//,"");Q=buildAttachmentPreviewUrl(
ce)}Q&&(j=document.createElement("img"),j.src=Q,j.className="thumb-item shadow-sm",j.dataset.viewerSrc=
Q,j.dataset.viewerFilename=D||Q.split("/").pop(),j.uploadRow=A,j.onclick=function(ce){ce.preventDefault(),
openImageViewer(this.dataset.viewerSrc,".thumb-item")},j.onerror=function(){this.parentElement.replaceChild(
u("ERR"),this)})}j||(j=u("FILE")),j.style.animationDelay=`${I*32}ms`,a.appendChild(j)}),C.length>0?a.
classList.remove("hidden"):a.classList.add("hidden")}function u(C){const A=document.createElement("d\
iv");return A.className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-\
[9px] shadow-sm font-bold",A.innerText=C,A}o(u,"createFileThumb");const f=collectImageUrlsForSend(),
h=uploadProgressState.total,y=uploadProgressState.completed,w=uploadProgressState.active;h===0&&(e.classList.
add("hidden"),n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));
const v=get("send-btn"),x=get("mic-btn"),S=get("mask-btn"),L=isStopMode;if(w>0?(v&&(v.disabled=!0),x&&
(x.disabled=!0),S&&(S.disabled=!0)):L||(v&&(v.disabled=!1),x&&(x.disabled=!1),S&&(S.disabled=!1)),w>
0){const C=`Preparing... (${y}/${h})`;e.classList.remove("hidden"),t.innerText=C,r&&(r.innerText=`(${y}\
/${h})`);let A=y*100,I=0;for(let ie in uploadProgressState.perFilePct)A+=uploadProgressState.perFilePct[ie],
I++;const J=h>0?A/(h*100)*100:0,D=`${Math.min(100,J)}%`;n&&i&&(n.classList.remove("hidden"),i.style.
width=D),l&&c&&(l.classList.remove("hidden"),c.style.width=D)}else r&&(r.innerText=""),l&&l.classList.
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
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const u=get("marker-tool-draw"),f=get(
"marker-tool-auto");u&&(u.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
f&&(f.classList.toggle("hidden",!markerState.maskMode),f.classList.toggle("active",e==="auto")),u&&u.
classList.toggle("active",e==="draw");const h=get("marker-brush-control");h&&h.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const w=get("marker-modal-title"),v=get("marker-modal-de\
scription"),x=get("marker-save");w&&(w.innerHTML=markerState.maskMode?'<i class="fas fa-mask text-ye\
llow-300"></i> \u30DE\u30B9\u30AF\u4F5C\u6210':'<i class="fas fa-highlighter text-yellow-300"></i> \u753B\
\u50CF\u7DE8\u96C6'),v&&(v.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\
\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\
\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
x&&(x.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const S=get("marker-canvas");S&&(S.style.pointerEvents=e==="crop"?"none":"auto");const L=get("marker\
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
r<=1)return;const l=(n-a)/2,c=(i-r)/2,u=a*markerView.scale,f=r*markerView.scale,h=Math.min(n*.45,Math.
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),w=h-l-u,v=n-h-l,x=y-c-f,S=i-y-c,L=o((C,A,I)=>Number.
isFinite(C)?A>I?(A+I)/2:Math.min(I,Math.max(A,C)):0,"clampOffset");markerView.offsetX=L(markerView.offsetX,
w,v),markerView.offsetY=L(markerView.offsetY,x,S)}o(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
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
height);const n=o((l,c,u=null,f=!1)=>{if(!l)return;const h=Math.max(0,l.x),y=Math.max(0,l.y),w=Math.
max(1,l.w),v=Math.max(1,l.h);u&&(t.fillStyle=u,t.fillRect(h,y,w,v)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(h+.5,y+.5,Math.max(1,w-1),Math.max(1,v-1)),t.restore()},"\
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
r),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),h=t.get(c);if(h===void 0){const v=e.
length;t.set(c,v),e.push({path:c,source:u,name:f});return}const y=e[h];if(!y)return;const w=normalizeAttachmentSource(
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
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e,t=!1){
const n=getRowImageSource(e);if(!n){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e,markerState.maskMode=t;const i=e?e.querySelector(".truncate"):null;
markerState.filename=i?i.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=
[],markerState.naturalWidth=0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=
[],markerState.mosaicPreviewRect=null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode(
"draw");const a=get("marker-attach-original");a&&(a.checked=e.dataset.attachOriginal==="1");const r=get(
"marker-image"),l=get("marker-canvas"),c=get("marker-crop-canvas");if(l){const u=l.getContext("2d");
u&&u.clearRect(0,0,l.width,l.height)}if(c){const u=c.getContext("2d");u&&u.clearRect(0,0,c.width,c.height)}
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
browserFastLocalFiles.forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,u=c?c.getAttribute("data-loc\
al-url"):null;u&&URL.revokeObjectURL(u)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=
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
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],u=c&&c.getSettings?c.getSettings():{},
f=String(u.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
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
const l=await new Promise((u,f)=>{t.toBlob(h=>{h?u(h):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
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
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{get("camera-split-enabled").checked?await openImageSplit(r):
await handleFiles(r,{openModal:!1}),get("camera-split-enabled").checked||showToast(`${r.length}\u679A\u306E\u753B\u50CF\u3092\
\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success")}catch(l){const c=l&&l.message?l.message:"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F";showToast(c,"error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending(
{keepStatus:!0}),e&&!e.classList.contains("hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}
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
"gif"].includes(l),u=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):u,h=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=h,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=r,y.dataset.defaultDisplayName=r,y.dataset.sendNameCustomized="";const w=escapeHtml(
r),v=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",x=c?`<img src="${f}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${w}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${x}
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
            `;const S=y.querySelector(".upload-preview");S&&(S.onclick=()=>openFileViewer(u,getRowAttachmentName(
y)||r));const L=y.querySelector(".upload-send-name");L&&(L.onclick=()=>promptRowAttachmentName(y));const C=y.
querySelector(".upload-remove");C&&(C.onclick=()=>{uploadCancelTokens.add(h),browserFastLocalFiles.delete(
h),decrementUploadTotal(h);const I=y.getAttribute("data-filename");I&&(currentImageUrls=currentImageUrls.
filter(J=>J!==I)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const A=y.querySelector(".upload-marker");return A&&(A.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,r),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:h}}o(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
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
            `,f&&i.setAttribute("data-local-url",f);const h=i.querySelector(".upload-preview");h&&(h.
onclick=()=>{const x=i.getAttribute("data-filename"),S=x?buildFileUrl(x):i.getAttribute("data-local-\
url"),L=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||x||"";openFileViewer(S,L)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const x=i.getAttribute("data-local-url");x&&URL.revokeObjectURL(x);const S=i.
getAttribute("data-filename");S&&(currentImageUrls=currentImageUrls.filter(L=>L!==S)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const w=i.querySelector(".upload-marker");w&&(w.onclick=()=>openMarkerModalForRow(i));const v=i.querySelector(
".upload-send-name");return v&&(v.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}o(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(isUploadCancelled(
t)||!e)return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);
try{const i=await apiFetch("/upload/init",{method:"POST",headers:{"Content-Type":"application/json",
"X-CSRF-Token":csrfToken},body:JSON.stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.
ok){const y=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}const r=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let y=0;y<c;y++){const w=y*l,v=Math.min(e.size,
w+l),x=e.slice(w,v);if(!await new Promise(L=>{const C=new XMLHttpRequest;C.open("POST","/upload/chun\
k",!0),C.setRequestHeader("X-CSRF-Token",csrfToken),C.upload.onprogress=I=>{if(I.lengthComputable&&t&&
t.bar){const J=w+I.loaded,D=Math.min(100,Math.floor(J/e.size*100));t.bar.style.width=`${D}%`,t.status&&
(t.status.textContent=`${D}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,D)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},C.onload=()=>{C.status>=200&&C.status<300?L(!0):L(!1)},C.
onerror=()=>L(!1);const A=new FormData;A.append("upload_id",r),A.append("index",String(y)),A.append(
"total",String(c)),A.append("chunk",x,e.name),C.send(A)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const u=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:r})}),f=await u.json();if(u.ok&&f&&f.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const v=t.row.getAttribute("data-loca\
l-url");v&&URL.revokeObjectURL(v),t.row.removeAttribute("data-local-url");const x=t.row.querySelector(
"img.upload-preview");if(x){const S=f.filename.replace(/^\d+\//,"");x.src=buildAttachmentPreviewUrl(
S)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),y)){const v=isRowAttachmentNameCustomized(
t.row),x=defaultAttachmentDisplayName(y),S=v&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
x;t.row.dataset.defaultDisplayName=x,setRowAttachmentName(t.row,S)}return y&&setAttachmentSourceForPath(
y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(f.
filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(v=>addLibraryFileFromPath(v)),!0}const h=f&&
f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),!1}catch{return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}o(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t){return new Promise(n=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(n);return}if(isUploadCancelled(t)){n(!1);return}let i=!1;window.ConnectionMonitor&&
(window.ConnectionMonitor.operationStarted(),i=!0);const a=o(()=>{i&&window.ConnectionMonitor&&(window.
ConnectionMonitor.operationEnded(),i=!1)},"finishUploadOp"),r=new XMLHttpRequest;r.open("POST",CHAT_CONFIG.
urls.upload,!0),r.setRequestHeader("X-CSRF-Token",csrfToken),r.upload.onprogress=c=>{if(c.lengthComputable&&
t&&t.bar){const u=Math.min(100,Math.floor(c.loaded/c.total*100));t.bar.style.width=`${u}%`,t.status&&
(t.status.textContent=`${u}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,u)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},r.onload=()=>{let c={};try{c=JSON.parse(r.responseText||"\
{}")}catch{}if(r.status>=200&&r.status<300&&c&&c.filename){if(isUploadCancelled(t)){t.row&&t.row.parentNode&&
t.row.remove(),a(),n(!1);return}if(t&&t.row){const h=t.row.getAttribute("data-local-url");h&&URL.revokeObjectURL(
h),t.row.removeAttribute("data-local-url");const y=t.row.querySelector("img.upload-preview");if(y){const w=c.
filename.replace(/^\d+\//,"");y.src=buildAttachmentPreviewUrl(w)}}const u=normalizeAttachmentPath(c.
filename);if(u&&currentImageUrls.push(u),t&&t.row&&(t.row.setAttribute("data-filename",u||c.filename),
setRowAttachmentSource(t.row,"upload"),u)){const h=isRowAttachmentNameCustomized(t.row),y=defaultAttachmentDisplayName(
u),w=h&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||y;t.row.dataset.defaultDisplayName=
y,setRowAttachmentName(t.row,w)}u&&setAttachmentSourceForPath(u,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(c.filenames)&&c.filenames.length?c.filenames:[c.filename]).
forEach(h=>addLibraryFileFromPath(h)),a(),n(!0)}else{const u=c&&c.error?c.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(u,"error",!0),a(),n(!1)}},r.onerror=()=>{
t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),a(),n(!1)};const l=new FormData;l.append("file",e),r.send(l)})}o(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}o(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}o(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(f=>{n+=f.length});const i=new Float32Array(n);
let a=0;e.forEach(f=>{i.set(f,a),a+=f.length});const r=new ArrayBuffer(44+i.length*2),l=new DataView(
r),c=o((f,h)=>{for(let y=0;y<h.length;y++)l.setUint8(f+y,h.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let u=44;for(let f=0;f<i.length;f++){const h=Math.
max(-1,Math.min(1,i[f]));l.setInt16(u,h<0?h*32768:h*32767,!0),u+=2}return new Blob([l],{type:"audio/\
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
er">FILE</div>',h=e.row.querySelector(".upload-preview");h&&(h.outerHTML=f);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const v=e.row.getAttribute("data-filename"),x=v?buildFileUrl(v):
e.row.getAttribute("data-local-url");openFileViewer(x,getRowAttachmentName(e.row)||a||v||"")});const w=e.
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
l=markerView.scale,c={x:0,y:0},u={x:0,y:0},f=[],h=16,y="",w=null,v=null,x=null,S=null,L=null,C=!1,A=null;
const I=o(P=>{const K=e.getBoundingClientRect(),be=(P.clientX-K.left)*(e.width/K.width),pe=(P.clientY-
K.top)*(e.height/K.height);return{x:be,y:pe}},"getPoint"),J=o((P,K)=>({x:(P.x+K.x)/2,y:(P.y+K.y)/2}),
"getMid"),D=o((P,K)=>Math.hypot(P.x-K.x,P.y-K.y),"getDist");let ie=!1;const j=o(()=>{v||(v=document.
createElement("canvas"),x=v.getContext("2d")),S||(S=document.createElement("canvas"),L=S.getContext(
"2d")),(v.width!==e.width||v.height!==e.height)&&(v.width=e.width,v.height=e.height),(S.width!==e.width||
S.height!==e.height)&&(S.width=e.width,S.height=e.height)},"ensureDrawBuffers"),Q=o(()=>{if(!t||!v||
!S)return;const P=markerState.maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.
opacity)||.6));t.clearRect(0,0,e.width,e.height),t.drawImage(v,0,0),t.save(),t.globalAlpha=P,t.drawImage(
S,0,0),t.restore()},"renderDrawPreview"),ce=o(()=>{L&&(L.strokeStyle=y,L.fillStyle=y,L.lineWidth=h,L.
lineCap="round",L.lineJoin="round")},"applyMarkerBrush"),me=o(P=>{if(!P)return!1;if(f.length===0)return f.
push(P),!0;const K=f[f.length-1],be=P.x-K.x,pe=P.y-K.y,te=Math.hypot(be,pe),X=Math.max(.35,h*.04);if(te<
X)return!1;const de=Math.max(1,h*.25),ne=Math.max(1,Math.ceil(te/de));for(let re=1;re<=ne;re++){const Se=re/
ne;f.push({x:K.x+be*Se,y:K.y+pe*Se})}return!0},"appendStrokePoint"),ae=o(()=>{if(L&&(L.clearRect(0,0,
S.width,S.height),f.length!==0)){if(ce(),f.length===1){const P=f[0];L.beginPath(),L.arc(P.x,P.y,h/2,
0,Math.PI*2),L.fill();return}if(L.beginPath(),L.moveTo(f[0].x,f[0].y),f.length===2)L.lineTo(f[1].x,f[1].
y);else{for(let be=1;be<f.length-2;be++){const pe=f[be],te=f[be+1],X=J(pe,te);L.quadraticCurveTo(pe.
x,pe.y,X.x,X.y)}const P=f[f.length-2],K=f[f.length-1];L.quadraticCurveTo(P.x,P.y,K.x,K.y)}L.stroke()}},
"renderStrokeLayer"),Te=o((P,K)=>{if(!P||!K)return null;const be=Math.min(P.x,K.x),pe=Math.min(P.y,K.
y),te=Math.abs(P.x-K.x),X=Math.abs(P.y-K.y);return{x:be,y:pe,w:te,h:X}},"normalizeMosaicRect"),E=o(P=>{
const K=n?Number(n.value||16):16,be=Math.max(6,Math.floor(K)),pe=Math.floor(be/2);return{x:P.x-pe,y:P.
y-pe,w:be,h:be}},"buildMosaicRectFromPoint"),R=o(()=>{const P=document.createElement("canvas");P.width=
e.width,P.height=e.height;const K=P.getContext("2d");if(!K)return null;markerState.baseCanvas&&K.drawImage(
markerState.baseCanvas,0,0),K.drawImage(e,0,0);try{return K.getImageData(0,0,e.width,e.height)}catch{
return null}},"getMosaicSourceImageData"),Z=o(P=>{if(!t||!P)return!1;const K=R();if(!K)return!1;const be=n?
Number(n.value||16):16,pe=Math.max(4,Math.floor(be/2)),te=Math.max(0,Math.floor(P.x)),X=Math.max(0,Math.
floor(P.y)),de=Math.min(e.width,Math.ceil(P.x+P.w)),ne=Math.min(e.height,Math.ceil(P.y+P.h));if(de<=
te||ne<=X)return!1;for(let re=X;re<ne;re+=pe)for(let Se=te;Se<de;Se+=pe){const we=Math.min(pe,de-Se),
Ke=Math.min(pe,ne-re),Qe=Math.min(e.width-1,Math.max(0,Se+Math.floor(we/2))),Ue=(Math.min(e.height-1,
Math.max(0,re+Math.floor(Ke/2)))*e.width+Qe)*4,ue=K.data[Ue],qe=K.data[Ue+1],Xe=K.data[Ue+2];t.fillStyle=
`rgb(${ue},${qe},${Xe})`,t.fillRect(Se,re,we,Ke)}return!0},"applyMosaicRect"),O=o(P=>{if(!t)return;if(i.
set(P.pointerId,{x:P.clientX,y:P.clientY}),i.size>=2){const be=Array.from(i.values()),pe=be[0],te=be[1];
a=!0,ie=!1,f=[],C=!1,A=null,w=null,markerState.mosaicPreviewRect=null,r=D(pe,te)||1,l=markerView.scale,
c={x:markerView.offsetX,y:markerView.offsetY},u=J(pe,te),renderCropOverlay(),e.setPointerCapture&&e.
setPointerCapture(P.pointerId),P.preventDefault();return}if(a||markerState.mode==="crop")return;ie=!0;
const K=I(P);if(markerState.mode==="auto"&&markerState.maskMode)w=K;else if(markerState.mode==="mosa\
ic")C=!0,A=K,markerState.mosaicPreviewRect=E(K),renderCropOverlay();else{if(j(),!x||!L)return;x.clearRect(
0,0,v.width,v.height),x.drawImage(e,0,0),L.clearRect(0,0,S.width,S.height),h=n?Number(n.value||16):16,
y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(markerState.colorHex),f=[],me(K),ae(),markerState.
hasStroke=!0,Q()}e.setPointerCapture&&e.setPointerCapture(P.pointerId),P.preventDefault()},"start"),
H=o(P=>{if(i.has(P.pointerId)&&i.set(P.pointerId,{x:P.clientX,y:P.clientY}),a&&i.size>=2){const be=Array.
from(i.values()),pe=be[0],te=be[1],X=J(pe,te),de=D(pe,te)||1,ne=l*(de/r);markerView.scale=Math.min(markerView.
maxScale,Math.max(markerView.minScale,ne)),markerView.offsetX=c.x+(X.x-u.x),markerView.offsetY=c.y+(X.
y-u.y),applyMarkerTransform(),P.preventDefault();return}if(!ie||!t)return;if(markerState.mode==="aut\
o"&&markerState.maskMode){P.preventDefault();return}const K=I(P);if(markerState.mode==="mosaic"){if(!C||
!A)return;markerState.mosaicPreviewRect=Te(A,K)||E(K),renderCropOverlay()}else me(K)&&(ae(),Q());P.preventDefault()},
"move"),G=o(P=>{const K=ie;if(i.delete(P.pointerId),i.size<2&&(a=!1),i.size===0){ie=!1;const be=K&&P.
type!=="pointercancel"&&markerState.mode==="auto"&&markerState.maskMode&&w?fillMaskSimilarRegion(w):
!1;if(w=null,K&&t&&markerState.mode==="draw"&&f.length>0&&(ae(),Q()),K&&markerState.mode==="mosaic"&&
A){const pe=I(P);let te=Te(A,pe);(!te||te.w<2||te.h<2)&&(te=E(A)),Z(te)&&(markerState.hasStroke=!0,markerState.
mosaicRects.push(te))}f=[],C=!1,A=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),K&&(markerState.
mode!=="auto"||be)&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(P.pointerId),
P.preventDefault()},"end");e.addEventListener("pointerdown",O),e.addEventListener("pointermove",H),e.
addEventListener("pointerup",G),e.addEventListener("pointercancel",G)}o(initMarkerCanvas,"initMarker\
Canvas");function fillMaskSimilarRegion(e){var ie;const t=get("marker-canvas"),n=markerState.baseImageData;
if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,r=Math.floor(e.x),l=Math.floor(e.y);if(r<0||l<0||r>=i||l>=
a)return!1;const c=n.data,u=l*i+r,f=u*4;if(c[f+3]<16)return!1;const h=c[f],y=c[f+1],w=c[f+2],v=Number(
((ie=get("marker-auto-tolerance"))==null?void 0:ie.value)||32),x=new Uint8Array(i*a),S=[u],L=[],C=25e4,
A=o(j=>{const Q=j*4;return c[Q+3]>=16&&Math.max(Math.abs(c[Q]-h),Math.abs(c[Q+1]-y),Math.abs(c[Q+2]-
w))<=v},"matches");for(x[u]=1;S.length;){const j=S.pop();if(x[j]===2||!A(j))continue;const Q=Math.floor(
j/i);let ce=j%i,me=ce;for(;ce>0&&x[Q*i+ce-1]!==2&&A(Q*i+ce-1);)ce--;for(;me+1<i&&x[Q*i+me+1]!==2&&A(
Q*i+me+1);)me++;for(let ae=ce;ae<=me;ae++)x[Q*i+ae]=2;if(L.push(Q,ce,me+1),L.length/3>C)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const ae of[Q-1,Q+1]){if(ae<0||ae>=a)continue;let Te=ce;for(;Te<=me;){const E=ae*
i+Te;if(x[E]===0&&A(E))for(x[E]=1,S.push(E);Te+1<=me&&x[ae*i+Te+1]===0&&A(ae*i+Te+1);)Te++,x[ae*i+Te]=
1;Te++}}}if(!L.length)return!1;const I=t.getContext("2d");if(!I)return!1;const J=I.getImageData(0,0,
i,a),D=J.data;for(let j=0;j<L.length;j+=3){const Q=L[j]*i+L[j+1],ce=L[j]*i+L[j+2];for(let me=Q;me<ce;me++){
const ae=me*4;D[ae]=D[ae+1]=D[ae+2]=D[ae+3]=255}}return I.putImageData(J,0,0),markerState.hasStroke=
!0,!0}o(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,r=null,l=null,c=!1,
u=0,f=markerView.scale,h={x:0,y:0},y={x:0,y:0};const w=8,v=14,x=o((j,Q,ce)=>Math.min(ce,Math.max(Q,j)),
"clamp"),S=o(j=>{const Q=e.getBoundingClientRect(),ce=(j.clientX-Q.left)*(e.width/Q.width),me=(j.clientY-
Q.top)*(e.height/Q.height);return{x:ce,y:me}},"getPoint"),L=o((j,Q)=>({x:(j.x+Q.x)/2,y:(j.y+Q.y)/2}),
"getMid"),C=o((j,Q)=>Math.hypot(j.x-Q.x,j.y-Q.y),"getDist"),A=o(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),I=o((j,Q)=>{if(!Q)return"move";const ce=Q.x,me=Q.y,ae=Q.x+Q.
w,Te=Q.y+Q.h,E=Math.abs(j.x-ce)<=v,R=Math.abs(j.x-ae)<=v,Z=Math.abs(j.y-me)<=v,O=Math.abs(j.y-Te)<=v;
if(E&&Z)return"nw";if(R&&Z)return"ne";if(E&&O)return"sw";if(R&&O)return"se";if(Z)return"n";if(O)return"\
s";if(E)return"w";if(R)return"e";if(j.x>ce+v&&j.x<ae-v&&j.y>me+v&&j.y<Te-v)return"move";const G=j.x<
ce?"left":j.x>ae?"right":null,P=j.y<me?"top":j.y>Te?"bottom":null;if(G&&P){if(G==="left"&&P==="top")
return"nw";if(G==="right"&&P==="top")return"ne";if(G==="left"&&P==="bottom")return"sw";if(G==="right"&&
P==="bottom")return"se"}return G?G==="left"?"w":"e":P?P==="top"?"n":"s":"move"},"hitTest"),J=o(j=>{if(markerState.
mode!=="crop")return;if(n.set(j.pointerId,{x:j.clientX,y:j.clientY}),n.size>=2){const me=Array.from(
n.values()),ae=me[0],Te=me[1];c=!0,i=!1,u=C(ae,Te)||1,f=markerView.scale,h={x:markerView.offsetX,y:markerView.
offsetY},y=L(ae,Te),e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault();return}
if(c)return;i=!0;const Q=S(j),ce=A();r=I(Q,ce),a=Q,l=ce?{x:ce.x,y:ce.y,w:ce.w,h:ce.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault()},"start"),D=o(j=>{if(markerState.
mode!=="crop")return;if(n.has(j.pointerId)&&n.set(j.pointerId,{x:j.clientX,y:j.clientY}),c&&n.size>=
2){const G=Array.from(n.values()),P=G[0],K=G[1],be=L(P,K),pe=C(P,K)||1,te=f*(pe/u);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,te)),markerView.offsetX=h.x+(be.x-y.x),markerView.
offsetY=h.y+(be.y-y.y),applyMarkerTransform(),renderCropOverlay(),j.preventDefault();return}if(!i||!a||
!l)return;const Q=S(j),ce=e.width,me=e.height,ae={x:l.x,y:l.y,w:l.w,h:l.h},Te=l.x+l.w,E=l.y+l.h,R=o(
()=>{const G=x(Q.x,0,Te-w);ae.x=G,ae.w=Te-G},"applyW"),Z=o(()=>{ae.w=x(Q.x-l.x,w,ce-l.x)},"applyE"),
O=o(()=>{const G=x(Q.y,0,E-w);ae.y=G,ae.h=E-G},"applyN"),H=o(()=>{ae.h=x(Q.y-l.y,w,me-l.y)},"applyS");
switch(r){case"move":{const G=Q.x-a.x,P=Q.y-a.y;ae.x=x(l.x+G,0,ce-l.w),ae.y=x(l.y+P,0,me-l.h);break}case"\
w":R();break;case"e":Z();break;case"n":O();break;case"s":H();break;case"nw":O(),R();break;case"ne":O(),
Z();break;case"sw":H(),R();break;case"se":H(),Z();break;default:break}ae.x=x(ae.x,0,ce-ae.w),ae.y=x(
ae.y,0,me-ae.h),markerState.cropRect=ae,renderCropOverlay(),j.preventDefault()},"move"),ie=o(j=>{n.delete(
j.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,r=null,l=null),e.releasePointerCapture&&
e.releasePointerCapture(j.pointerId),j.preventDefault()},"end");e.addEventListener("pointerdown",J),
e.addEventListener("pointermove",D),e.addEventListener("pointerup",ie),e.addEventListener("pointerca\
ncel",ie),e.addEventListener("pointerleave",ie)}o(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const r=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const L=document.createElement("canvas");L.width=r,L.height=l;const C=L.getContext(
"2d");if(!C)return;C.fillStyle="#000000",C.fillRect(0,0,r,l),C.drawImage(n,0,0,r,l);const A=await new Promise(
D=>L.toBlob(D,"image/png"));if(!A){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const I=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([A],`${I}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=r,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,r,l),c.drawImage(n,0,0,r,l),markerState.cropRect){const L=r/n.width,C=l/n.height,A=Math.max(0,Math.
floor(markerState.cropRect.x*L)),I=Math.max(0,Math.floor(markerState.cropRect.y*C)),J=Math.min(r,Math.
max(1,Math.floor(markerState.cropRect.w*L))),D=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*C))),ie=document.createElement("canvas");ie.width=J,ie.height=D;const j=ie.getContext("2d");j&&(j.
drawImage(a,A,I,J,D,0,0,J,D),a=ie)}const u=await new Promise(L=>a.toBlob(L,"image/png",.92));if(!u){
showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0);return}const h=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([u],
`${h}_marked.png`,{type:"image/png"}),w={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".\
upload-status"),bar:e.querySelector(".upload-progress > div")};w.status&&(w.status.textContent="\u7DE8\u96C6\u53CD\u6620\
\u4E2D..."),updateUploadRowFile(w,y);const v=e.getAttribute("data-filename"),x=getRowAttachmentSource(
e);v&&!e.dataset.originalFilename&&(e.dataset.originalFilename=v,e.dataset.originalSource=x,setAttachmentSourceForPath(
v,x)),await uploadFileWithProgress(y,w)?(v&&(currentImageUrls=currentImageUrls.filter(L=>L!==v)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}o(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,r=null,l=null,c=null,u=[],f=null;const h=o(()=>{f&&clearTimeout(
f);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(w=>w.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(r)try{r.close()}catch{}},"cleanup"),y=o(
()=>{h(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const w=a.getAudioTracks();
if(!w||!w.length)return y();r=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=r.createMediaStreamSource(new MediaStream(w)),l=r.createScriptProcessor(4096,1,1),l.onaudioprocess=
x=>{const S=x.inputBuffer.getChannelData(0);u.push(new Float32Array(S))},c.connect(l),l.connect(r.destination);
const v=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;v>0&&(f=setTimeout(()=>{const x=(e.
name||"video").replace(/\.[^/.]+$/,""),S=encodeWav(u,r.sampleRate),L=new File([S],`${x}.audio.wav`,{
type:"audio/wav"});h(),n(L)},v+250)),await i.play(),i.onended=()=>{const x=(e.name||"video").replace(
/\.[^/.]+$/,""),S=encodeWav(u,r.sampleRate),L=new File([S],`${x}.audio.wav`,{type:"audio/wav"});h(),
n(L)}}catch{y()}},i.onerror=()=>y()}))}o(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
if(!e||!e.length)return;const n=Array.from(e).filter(Boolean);if(!n.length)return;const i=n,a=uploadGeneration;
t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=i.length,uploadProgressState.
active+=i.length,updateFilePreview();const r=!!(get("upload-audio-only")&&get("upload-audio-only").checked),
l=getModelMediaSupport(get("model-select").value),c=o(async y=>{let w=null;if(a!==uploadGeneration)return!1;
try{if(isAudioFile(y)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(y)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!y.type||!y.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const v=addUploadRow(y);v.generation=a,updateFilePreview(),w=v.uploadId,uploadProgressState.
perFilePct[w]=0;let x=y;if(r&&isVideoFile(y)){const S=await extractAudioFromVideo(y,v);S?(x=S,updateUploadRowFile(
v,S),v&&v.status&&(v.status.textContent="\u97F3\u58F0\u306E\u307F")):(v&&v.status&&(v.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&y.type.startsWith("image/"))try{x=await applyImageCompression(y,
readComposerCompression()),x!==y&&updateUploadRowFile(v,x)}catch{}if(isUploadCancelled(v))return!1;if(browserFastModeEnabled){
const S=Array.from(browserFastLocalFiles.values()).reduce((L,C)=>L+Number(C.file&&C.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||S+x.size>BROWSER_FAST_MAX_BYTES?(v&&v.status&&
(v.status.textContent="\u4E0A\u9650\u8D85\u904E"),v&&v.row&&v.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(v.uploadId,{file:x,rowObj:v}),
v.status&&(v.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
v.bar&&(v.bar.style.width="100%"),v.row&&(v.row.dataset.browserFastLocal="1"),!0)}return await uploadFileWithProgress(
x,v)}finally{a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne");let u=0;const f=Math.min(UPLOAD_CONCURRENCY,i.length),
h=Array.from({length:f}).map(async()=>{for(;;){const y=u++;if(y>=i.length)break;await c(i[y])}});await Promise.
all(h)}o(handleFiles,"handleFiles"),get("clear-file-btn").onclick=e=>{e.stopPropagation(),resetUploadState()},
get("clear-mask-btn")&&(get("clear-mask-btn").onclick=()=>{currentMaskImage=null,updateMaskPreview()}),
get("mask-btn")&&get("mask-input")&&(get("mask-btn").onclick=()=>{if(isIdeogramObjectRemoval()){const e=Array.
from(document.querySelectorAll("#upload-list .upload-row")).find(t=>getRowImageSource(t));if(!e){showToast(
"\u5148\u306B\u30DE\u30B9\u30AF\u5BFE\u8C61\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}openMarkerModalForRow(e,!0);return}get("mask-input").click()},get("mask-input").addEventListener(
"change",async e=>{const t=e.target.files&&e.target.files[0];t&&(await uploadMaskFile(t),e.target.value=
"")}));const messageMeta={};let markdownLibraryFallbackReported=!1;function sanitizeMarkdownHtml(e,t={}){
const n=String(e||"");if(!window.marked||typeof window.marked.parse!="function"||!window.DOMPurify||
typeof window.DOMPurify.sanitize!="function")return markdownLibraryFallbackReported||(markdownLibraryFallbackReported=
!0,console.error("Markdown sanitizer is unavailable; rendering escaped plain text.")),escapeHtml(n).
replace(/\n/g,"<br>");const i=protectMathSegments(n);markdownOpenFenceBody=/<\/svg/i.test(n)?findOpenMarkdownFenceBody(
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
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const u=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return u&&Number(u[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
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
e||""),n=t.split(/\r?\n/),i=[],a=[],r=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",u=[];for(const y of n){
if(!l){const x=y.match(r);if(x){l=x[2],c=String(x[3]||"").trim(),u=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const w=String(y||"").trim();if(w&&w.replace(/\s+/g,"")===l){const x=i[i.length-1];x&&(x.
code=u.join(`
`),x.open=!1),l=null,c="",u=[];continue}u.push(y);const v=i[i.length-1];v&&(v.code=u.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=u.join(`
`),y.open=!0)}const f=i.map((y,w)=>normalizeCanvasBlock(y,w)),h=selectCanvasPreviewBlock(f,t);return{
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
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",h=(String(i&&i.code?i.code:"").split(/\r?\n/).find(v=>v.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${r}`,w=`${y}\u3001${h}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(w)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
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
null?i.code:""),u=r?isCanvasHtmlPreviewCandidate(l,c):!1,f=r?u?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",h=r?u?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=h),e.status&&(e.status.textContent=
f),e.previewLang&&(e.previewLang.textContent=r?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,w=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=w,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),r){canvasPreviewState.frameRenderToken+=1;const v=canvasPreviewState.frameRenderToken,
x=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),v);e.frame&&(e.frame.srcdoc=x,e.frame.
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
(c,u)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const h=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},u);a.push(h),r===-1&&f.codeKey===t.codeKey&&(r=a.length-1)}),!a.length)return null;
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
"`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);for(const f of r){const h=`${u}${f}
`,y=`
${u}`,w=h+a+y;i.includes(w)&&(i=i.split(w).join(""))}}return i}o(stripExactFencedBlock,"stripExactFe\
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
${u}`].forEach(h=>{n.includes(h)&&(n=n.split(h).join(`
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
${t}`),h=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",y=`<button class="download-btn" data-\
code="${c}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,w=`<button class="coding-target-btn" data-code="${c}" data-code-key="${f}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${f}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(h)}</span><div class="code-actions">${w}${y}<button class="copy-btn" data-copy="code" \
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
n.incrementalMath){const c=document.createElement("template");c.innerHTML=sanitizeMarkdownHtml(r.renderText,
{streamMathSegments:!0});const u=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(h=>{const y=h.getAttribute("data-stream-math-key");y&&u.set(y,h)});const f=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(h=>{const y=u.get(h.getAttribute(
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
400"></i> Search complete'),t}o(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,r,l=null,c=!0,u=null,f=null,h=null,y=null,w=null,v=null,x=null,S=null,L=!0,C=null,A=null,I=null,J=null){
const D=t==="user",ie=D?"bg-blue-600":"bg-gray-700",j=D?"justify-end":"justify-start";messageStore[e]=
n;const Q=!D&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},ce=D?n:Q.text;let me=f;
if(me==null){const ne=h!=null?Number(h):0,re=y!=null?Number(y):0;(h!=null||y!=null)&&(me=ne+re)}messageMeta[e]=
{tokens_in:h,tokens_out:y,tokens_total:me,tokens_content:v,tokens_thought:x,is_encrypted:w,role:t,model:r,
parent_id:C,quote_text:u,image_url:i,image_mask:J,gem_name:A,batch_job:I,python_executions:D?[]:Q.executions||
[]};let ae="";u&&(ae=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs te\
xt-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(
u)}</div>`);let Te="";if(a&&!D){let ne="";try{const re=JSON.parse(a);ne=re.text||"",re.search_used===
!0&&(Te=SEARCH_COMPLETE_HTML)}catch{ne=a}ne&&(Te+=`<div class="thought-container"><div class="though\
t-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Proce\
ss</div><div class="thought-content collapsed">${escapeHtml(ne)}</div></div>`)}let E="";if(i)try{const ne=JSON.
parse(i);if(ne.length){const re=[];if(ne.forEach(Se=>{let we=Se,Ke="unknown";if(we&&typeof we=="obje\
ct"&&(Ke=normalizeAttachmentSource(we.source),we=we.filepath||we.path||we.url||we.file||""),we=normalizeAttachmentPath(
we)||we,!we)return;setAttachmentSourceForPath(we,Ke);const Qe=we.replace(/^\d+\//,""),je=buildFileUrl(
Qe),Ue=buildAttachmentPreviewUrl(Qe),ue=we.split("/").pop(),qe=ue.split(".").pop().toLowerCase();["j\
pg","jpeg","png","webp","gif"].includes(qe)?re.push(buildChatImageHtml(Ue,{viewerSrc:je,alt:ue,title:ue,
filename:ue})):re.push(`<div class="file-thumb bg-gray-800 border border-gray-600 rounded flex flex-\
col items-center justify-center cursor-pointer hover:bg-gray-700" onclick="window.open('${je}')" tit\
le="${ue}"><i class="fas fa-file text-2xl text-gray-400 mb-1"></i><span class="text-[9px] truncate w\
-20 text-center">${ue}</span></div>`)}),re.length>0){let Se="grid-multi";re.length===1?Se="grid-1":re.
length===2?Se="grid-2":re.length===3?Se="grid-3":re.length===4&&(Se="grid-4"),E=`<div class="image-g\
rid ${Se}">${re.join("")}</div>`}}}catch{}let R="";if(D&&J)try{const ne=normalizeAttachmentPath(J)||
String(J),re=ne.replace(/^\d+\//,""),Se=buildFileUrl(re),we=buildAttachmentPreviewUrl(re),Ke=ne.split(
"/").pop();R=`<div class="mt-2 max-w-xs rounded-lg border border-amber-300/40 bg-black/10 p-2"><div \
class="mb-1 text-[10px] font-semibold text-amber-100"><i class="fas fa-mask mr-1" aria-hidden="true"\
></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(we,
{viewerSrc:Se,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:Ke})}</div>`}catch{}const Z=D?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,O=`<div class="msg-controls absolute -top-5 right-\
0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', t\
his)"><i class="fas fa-copy"></i></button>${D?`<button class="ctrl-btn edit-btn" data-id="${e}"><i c\
lass="fas fa-pen"></i></button>`:""}${Z}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\u65B0\
\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,H=[];!D&&r&&H.push(escapeHtml(r)),A&&(D?
H.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(A)}</span>`):
H.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(A)}</span>`));
const G=[];if(h!=null&&G.push(`In ${h}`),y!=null){let ne=`Out ${y}`;x!=null&&Number(x)>0&&(ne+=` (Th\
ought ${x})`),G.push(ne)}if(G.length||f!=null){const ne=G.length?G.join(" / "):`${f} tokens`;H.push(
`<button class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDet\
ail('${e}')">${ne}</button>`)}if(w!=null){const ne=w?"fa-lock":"fa-lock-open",re=isAdminUser?w?"\u6697\u53F7\u5316\u72B6\
\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":
w?"Encrypted":"Plain",Se=isAdminUser?w?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 ho\
ver:text-cyan-200":"text-slate-300/80 hover:text-white";H.push(`<button class="${Se}" title="${re}" \
onclick="openEncryptionSettings('${e}')"><i class="fas ${ne}"></i></button>`)}if(!D&&Q.executions&&Q.
executions.length){const ne=Q.executions.length,re=ne>1?`Python \xD7${ne}`:"Python";H.push(`<button \
type="button" class="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" a\
ria-label="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${re}\
</span></button>`)}const P=H.length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-\
mono message-footer-meta">${H.join(" \u2022 ")}</div>`:"";let K;const be=!D&&I?(()=>{const ne=String(
I.state||"").toUpperCase(),re=I.status_text||(ne==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ne==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ne==="JOB_STATE_FAILED"||
ne==="JOB_STATE_CANCELLED"||ne==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ne==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(re)}</div></div>`})():
"";D?K=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(K=be+(ce&&String(ce).trim()?buildAiMarkdownHtml(ce):I?'<div class="content-area prose p\
rose-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(ce)),K.includes("content-area")||(K=K.replace("prose ","content-area prose ")));
let pe="";if(l){const ne=l.siblings[l.current-2],re=l.siblings[l.current];pe=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ne}\
)" ${ne?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${re}\
)" ${re?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const te=c?"fade-in":"",X=document.createElement("div");X.className=`flex ${j} mb-\
4 ${te} relative message-group group`,X.id=`msg-${e}`,X.innerHTML=`<div class="message-bubble ${ie} \
text-white p-4 rounded-2xl shadow-md relative">${O}${ae}${Te}${K}${E}${R}${pe}${P}</div>`;const de=S||
get("chat-container");return de&&(de.appendChild(X),L&&scrollToBottom(),D||(queueMessageDecorations(
X,ce),syncCodingTargetButtons(X),syncCodingModeUi(codingModeEnabled,{persist:!1}))),X}o(renderMessage,
"renderMessage");function showTokenDetailModal(e=null,t=!1){if(location.pathname!=="/token-details"){
const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==null&&(n.messageId=e),history.pushState(
n,"","/token-details")}showModal("token-detail-modal")}o(showTokenDetailModal,"showTokenDetailModal");
function openTokenDetail(e){const t=messageMeta[e];if(!t||!get("token-detail-modal"))return;get("tok\
en-detail-usage").classList.remove("hidden"),get("token-detail-estimate").classList.add("hidden"),cancelPromptTokenDetail();
const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:"-",a=t.tokens_in!==null&&t.tokens_in!==
void 0?t.tokens_in:"-",r=t.tokens_out!==null&&t.tokens_out!==void 0?t.tokens_out:"-",l=t.tokens_content!==
null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.tokens_thought!==null&&t.tokens_thought!==void 0?
t.tokens_thought:"-",u=t.is_encrypted===null||t.is_encrypted===void 0?"-":t.is_encrypted?"Encrypted":
"Plain";get("token-detail-total").innerText=i,get("token-detail-in").innerText=a,get("token-detail-o\
ut").innerText=r,get("token-detail-content").innerText=l,get("token-detail-thought").innerText=c,get(
"token-detail-encrypted").innerText=u;const f=t.model?`${t.model} (${t.role})`:`${t.role}`;get("toke\
n-detail-title").innerText=f,showTokenDetailModal(e)}o(openTokenDetail,"openTokenDetail");function closeTokenDetail(e=!1){
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
"click",y,!0)}o(c,"cleanup");function u(w){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),r(w)}o(u,"finalize");
function f(){u(!0)}o(f,"onOk");function h(){u(!1)}o(h,"onCancel");function y(w){w.target===e&&(w.preventDefault(),
w.stopImmediatePropagation(),h())}o(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",h),a&&a.addEventListener("click",h),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const r=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),u=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${r}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("\
chat-container");if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",u);else{
const h=document.createElement("div");h.innerHTML=u;const y=h.firstElementChild;y&&f.appendChild(y)}
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
\u4F7F\u3044\u65B9: ${n.label} default / none`,"info"),!1;const r=get(i.selectId),l=a.toLowerCase(),
c=r?Array.from(r.options).find(u=>u.value.toLowerCase()===l||u.textContent.trim().toLowerCase()===l):
null;return!r||!c?(showToast(`${n.label}: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1):(r.value=c.value,r.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),
showToast(`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const r=a.
toLowerCase(),l=activeThinkingSelect(),c=!!(l&&l.id==="reasoning-effort"),u=c?{off:"none",none:"none",
min:"low",minimal:"low",low:"low",mid:"medium",medium:"medium",high:"high",xhigh:"xhigh",max:"max"}:
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},f=parseSlashToggleArgument(
a),h=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(u,r)){const y=u[r],w=l?Array.from(l.options).
find(v=>v.value===y&&!v.disabled&&!v.classList.contains("hidden")):null;if(l&&!w)return showToast(`T\
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
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((w,v)=>{const x=document.createElement(
"div");x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${v===
slashSelectedIndex?"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas ${w.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${w.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${w.description}</div>
                    </div>
                `;let S=!1;x.addEventListener("pointerdown",L=>{typeof L.button=="number"&&L.button!==
0||(L.preventDefault(),S=!0,selectSlashCommand(w.id))}),x.addEventListener("click",L=>{L.preventDefault(),
S||selectSlashCommand(w.id)}),x.onmouseenter=()=>{slashSelectedIndex=v,showSlashCommandSuggestions(e)},
n.appendChild(x)});const r=i.getBoundingClientRect(),l=window.innerHeight,c=l-r.bottom,u=r.top,f=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&u>c){const w=Math.min(f,u-h);t.style.top="auto",t.style.bottom=`${l-r.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(f,c-h);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
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
forEach(([y,w])=>{const v=AI_SETTING_JUMP_TARGETS[y]||{label:y},x=document.createElement("button");x.
type="button",x.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const S=document.createElement("span");S.className="min-w-0 flex-1";const L=document.createElement(
"span");L.className="block text-xs font-bold text-blue-200",L.textContent=v.label;const C=document.createElement(
"span");C.className="block mt-0.5 text-[11px] text-gray-300 break-words",C.textContent=formatAiSettingValue(
w);const A=document.createElement("i");A.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",S.appendChild(L),S.appendChild(C),x.appendChild(S),x.appendChild(A),x.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(x)});const h=u.querySelector(".message-footer-\
meta");h?u.insertBefore(f,h):u.appendChild(f),scrollToBottom()}o(renderAiSettingsResultBubble,"rende\
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
u.applied,t);return}const h=u.message||u.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${h}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${h}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(h,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
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
r.length&&(gemSelectedIndex=0),n.innerHTML="",r.forEach((v,x)=>{const S=document.createElement("div");
S.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${x===gemSelectedIndex?
"bg-gray-700":""}`,S.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(v.name)}</div>
                        ${v.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
v.description)}</div>`:""}
                    </div>
                `,S.onclick=()=>selectGemSuggestion(v),S.onmouseenter=()=>{gemSelectedIndex=x,showGemSuggestions(
e)},n.appendChild(S)});const l=i.getBoundingClientRect(),c=window.innerHeight,u=c-l.bottom,f=l.top,h=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",u<180&&f>u){const v=Math.min(h,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(h,u-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
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
"warning",!0),t}o(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(e=browserFastLocalFiles,t=!1){
const n=Array.from(e.entries()),i=[];for(const[a,r]of n){if(!r||!r.file||!r.rowObj)throw new Error("\
\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");if(r.
rowObj.status&&(r.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(r.file,r.rowObj))throw new Error(`${r.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);const c=r.rowObj.row.getAttribute("da\
ta-filename");c&&i.push(c),t?currentImageUrls=currentImageUrls.filter(u=>u!==c):browserFastLocalFiles.
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
const e=o((n,i)=>get(n)?!!get(n).checked:i,"checked"),t=o(n=>get(n)?get(n).value:null,"value");return{
canvas_mode:!!canvasModeEnabled,enable_search:e("enable-search",!1),enable_url_context:e("enable-url\
-context",!1),enable_maps:e("enable-maps",!1),enable_python:e("enable-python",!1),enable_file_creation:e(
"enable-file-creation",!0),enable_mcp:typeof isMcpEnabledForSend=="function"?!!isMcpEnabledForSend():
!1,enable_thinking:e("enable-thinking",!1),thinking_level:t("thinking-level"),thinking_budget:t("thi\
nking-budget"),reasoning_effort:t("reasoning-effort"),safety_setting:t("safety-setting"),enable_prompt_caching:e(
"enable-prompt-cache",!1)}}o(collectBrowserFastPromptOptions,"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e,t){
var Z,O,H;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),r=[];for(const G of a)r.push({inlineData:{
mimeType:G.file.type,data:await fileToBase64Payload(G.file)}});r.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const u={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:r}],generationConfig:l};!!(get("enable-python")&&get("enable-python").
checked)&&(u.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden");const h=userAutoScroll&&isChatNearBottom();
h||pauseChatAutoScroll();const y=collectImageUrlsForSend(),w=Array.from(((Z=get("file-preview-thumbs"))==
null?void 0:Z.querySelectorAll("img"))||[]).map(G=>G.cloneNode(!0)),v=renderMessage(Date.now(),"user",
e,JSON.stringify(collectImageUrlsForSend()),null,null,null,!1,currentQuote,null,null,null,null,null,
null,null,h);if(w.length&&!y.length){const G=document.createElement("div");G.className="flex gap-2 f\
lex-wrap",w.forEach(P=>{P.removeAttribute("onclick"),G.appendChild(P)}),v.querySelector(".message-bu\
bble").appendChild(G)}const x=takePromptForSend(),S=animatePromptToMessage(v,x,h),L=`browser-fast-${Date.
now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai\
-pending-row fade-in"><div id="${L}" class="message-bubble ai-pending-bubble bg-gray-700 text-white \
p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(n,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}\
</div></div>`);const C=get(L),A=startPendingElapsed(C,t);activeStreamingBubbleId=L,setSendBtnToStopMode(),
h&&resumeChatAutoScroll(),abortController=new AbortController;let I="",J="";const D=[];let ie=null,j=null,
Q=!1;const ce={},me=[];let ae=null,Te="";const E=window.ProgressSpinner?window.ProgressSpinner.startFlow(
"browserFast"):null;let R=!1;try{const G=await fetch(`https://generativelanguage.googleapis.com/v1be\
ta/models/${encodeURIComponent(n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(
u),signal:abortController.signal}));if(!G.ok){const we=await G.json().catch(()=>({}));throw new Error(
we&&we.error&&we.error.message?we.error.message:`Gemini API HTTP ${G.status}`)}window.ConnectionMonitor&&
(R=!0,window.ConnectionMonitor.operationStarted()),E&&E.setPhase("waiting"),x.accept(!1),(O=window.ActivityLog)==
null||O.log("send.stream.start");let P=!1;const K=G.body.getReader(),be=new TextDecoder;let pe="";const te=o(
we=>{const Ke=we.split(/\r?\n/).filter(Ue=>Ue.startsWith("data:")).map(Ue=>Ue.slice(5).trim()).join(
"");if(!Ke||Ke==="[DONE]")return;const Qe=JSON.parse(Ke);if(Qe.error)throw new Error(Qe.error.message||
"Gemini API error");if((Array.isArray(Qe.candidates)?Qe.candidates:[]).forEach(Ue=>{(Ue&&Ue.content&&
Array.isArray(Ue.content.parts)?Ue.content.parts:[]).forEach(qe=>{if(qe&&typeof qe.thoughtSignature==
"string"&&!D.includes(qe.thoughtSignature)&&D.push(qe.thoughtSignature),qe&&qe.executableCode&&typeof qe.
executableCode.code=="string"){const pt=qe.executableCode.code;I+=`
\`\`\`python
${pt}
\`\`\`
`,ae=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,Te=pt,ce[ae]||(C.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(ae)),ce[ae]=C.querySelector(`[data-py-id="${ae}"]`)),updateBrowserFastPythonBox(
ce[ae],"code",pt);return}if(qe&&qe.codeExecutionResult&&typeof qe.codeExecutionResult.output=="strin\
g"){const pt=qe.codeExecutionResult.output;I+=`
**Output:**
\`\`\`
${pt}
\`\`\`
`;const bt=ae||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;me.push({code:Te||
"",output:pt}),ce[bt]||(C.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(bt)),ce[bt]=C.querySelector(
`[data-py-id="${bt}"]`)),updateBrowserFastPythonBox(ce[bt],"output",pt);return}const Xe=typeof qe.text==
"string"?qe.text:"";Xe&&(qe.thought===!0?J+=Xe:I+=Xe)})}),!Q&&(I||J)){beginPendingToStreamTransition(
C);const Ue=C.querySelector(".content-area");Ue&&Ue.remove(),Q=!0}J&&(j||(C.insertAdjacentHTML("afte\
rbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><\
i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div\
></div>'),j=C.querySelector(".thought-content")),j.textContent=J),I&&(ie||(ie=document.createElement(
"div"),ie.className="content-area prose prose-invert text-sm break-words",C.appendChild(ie)),renderAiMarkdownInto(
ie,I,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:we,value:Ke}=await K.
read();if(!we&&!P&&(P=!0,(H=window.ActivityLog)==null||H.log("send.stream.first")),we)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),E&&E.setPhase("receiving"),pe+=be.decode(Ke,{stream:!0});const Qe=pe.
split(/\r?\n\r?\n/);pe=Qe.pop()||"",Qe.forEach(te)}if(pe+=be.decode(),pe.trim()&&te(pe),!I.trim())throw new Error(
"Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
ie&&renderAiMarkdownInto(ie,I,{incrementalMath:!0}),j&&j.classList.add("collapsed"),me.length&&(I+=me.
map(we=>`
\`\`\`pyexec
${JSON.stringify(we)}
\`\`\`
`).join(""));let X=[];a.length&&(E&&E.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),X=await uploadBrowserFastLocalFiles(x.files,!0)),E&&E.setPhase("saving");const de=[...y,...X],
ne=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:I,thought_content:J,model:n,image_urls:de,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:D,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),C),re=await ne.json().
catch(()=>({}));if(!ne.ok||!re.thread_id)throw new Error(re.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Se=!currentThreadId;currentThreadId=String(re.thread_id),currentParentId=re.assistant_message_id||
null,currentLeafId=re.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!Se}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(G){if(S(),x.restore(),G.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${G.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const P=G.message||"\u30A8\
\u30E9\u30FC";C&&C.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(P));try{let K=I||"";me.length&&
(K+=me.map(de=>`
\`\`\`pyexec
${JSON.stringify(de)}
\`\`\`
`).join(""));const be=buildChatErrorMarkdown(P,K),pe=a.length?[]:y,te=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:be,
thought_content:J||"",model:n,image_urls:pe,prompt_options:collectBrowserFastPromptOptions(),temporary_chat:temporaryChatEnabled,
thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:null,thought_signatures:D,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController&&!abortController.signal.aborted?abortController.signal:void 0}),C),X=await te.
json().catch(()=>({}));if(te.ok&&X.thread_id){const de=!currentThreadId;currentThreadId=String(X.thread_id),
currentParentId=X.assistant_message_id||null,currentLeafId=X.assistant_message_id||null,browserFastBootstrap=
null,await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!de}),applyBrowserFastModeRestrictions(),
loadThreads(!1)}}catch(K){sendClientDebugLog("error",`Browser fast error persist failed: ${K&&K.message?
K.message:K}`)}}}finally{x.releaseFiles(),A&&A(),R&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),E&&E(),setSendBtnToSendMode(),activeStreamingBubbleId===L&&(activeStreamingBubbleId=
null),abortController=null,updateFilePreview()}}o(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
o(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var $t,jt,ln,Vt,vt;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const U=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(U);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const U=pendingSlashCommand,
xe=a.trim(),Ne=get("model-select")?get("model-select").value:null;if(U==="settings"){if(!xe){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!Ne){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
xe,Ne)}else executeMinimalSlashCommand(U,xe)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const r=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(r&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(U=>U.id===
r[1].toLowerCase())){executeMinimalSlashCommand(r[1].toLowerCase(),r[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const l=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(l&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(l)setBrowserFastModeEnabled(!1);else{const U=browserFastModeIneligibility(
a);if(!U){try{await sendBrowserFastMessage(a,t)}catch(xe){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${xe.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${U}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(xe){showToast(
xe.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),u=c.map(U=>U.path),f=c.filter(U=>normalizeAttachmentSource(
U.source)==="upload").map(U=>U.path),h=getModelMediaSupport(get("model-select").value),y=u.some(U=>isAudioPath(
U)),w=u.some(U=>isVideoPath(U)),v=(get("model-select").value||"").toLowerCase(),x=get("enable-python"),
S=!!(x&&x.checked);if(y&&!h.audio||w&&!h.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&u.length===0)return;if(isMistralOcrModel(
v)){const U=/https?:\/\/\S+/i.test(a);if(u.filter(Ne=>isAudioPath(Ne)||isVideoPath(Ne)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!u.length&&!U){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const L=a.trim();if(/^\/settings(?:\s|$)/i.
test(L)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(L)){const U=L.replace(/^\/settings\s*/i,"").trim();
if(!U){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ne=get("prompt-input");Ne.value="/settings ";const Ie=extractSlashCommandToken(Ne.value);
lastSlashFilter=Ie,showSlashCommandSuggestions(Ie),Ne.focus();return}const xe=get("model-select")?get(
"model-select").value:null;if(!xe){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
U,xe);return}if(isGeminiLocalPythonMode(v,y,w,S)&&!await confirmGeminiLocalPythonSwitch())return;let C=null,
A=[];if(codingModeEnabled){const U=collectCodingCandidates(a),xe=U.filter(Ze=>Ze.prompt_source),Ne=U.
filter(Ze=>!Ze.prompt_source),Ie=xe.reduce((Ze,xt)=>Ze+String(xt.code||"").length,0);if(Ie>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let rt=3e5-Ie;const ut=[];for(let Ze=Ne.length-1;Ze>=0;Ze--){const xt=String(Ne[Ze].
code||"").length;xt>rt||(ut.unshift(Ne[Ze]),rt-=xt)}A=codingTargetSelection?ut.slice(-1):[...xe,...ut];
const wt=xe.length?xe[xe.length-1]:null;if(C=codingTargetSelection?A[0]:wt||A[A.length-1]||null,codingModeEffective=
!!(C&&String(C.code||"").trim()),codingModeEffective&&C.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const Ze=String((($t=get(
"model-select"))==null?void 0:$t.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(Ze)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const I=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${u.length} search=${get("enable-search").checked}`);const J=a,D=hasMarkerHint()?MARKER_HINT_TEXT:
null,ie=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&u.length===0){showToast("M\
ask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if(ie&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=userAutoScroll&&isChatNearBottom();j||pauseChatAutoScroll();const Q=editingMessageId,
ce=currentParentId,me=Q!=null;Q&&(editingMessageId=null,setEditUi(!1)),playSendAnimation(),get("welc\
ome-screen").classList.add("hidden");const ae=[],Te=o(U=>{if(U==null)return;let xe=document.getElementById(
`msg-${U}`);for(;xe;)xe.classList&&xe.classList.contains("message-group")&&(ae.push({node:xe,prevDisplay:xe.
style.display}),xe.style.display="none"),xe=xe.nextElementSibling},"hideRenderedBranchFrom"),E=o(()=>{
ae.forEach(({node:U,prevDisplay:xe})=>{U&&(U.style.display=xe||"")}),ae.length=0},"restoreHiddenBran\
ch");Q&&Te(Q);const R=Date.now(),Z=isGptImageModel()||ie?currentMaskImage:null,O=renderMessage(R,"us\
er",J,JSON.stringify(u),null,null,null,!1,currentQuote,null,null,null,null,null,null,null,j,ce,activeGem?
activeGem.name:null,null,Z);let H=!1;const G=/(https?:\/\/)?(x\.com|twitter\.com)\//i,P=!i&&(G.test(
J||"")||G.test(currentQuote||"")),K="grok-4-fast-reasoning",be=o(()=>{get("enable-search").checked=!0,
get("model-select").value!==K&&selectModelById(K)},"applyXLinkAuto");if(P&&!isMistralOcrModel()&&!get(
"enable-search").checked)if(autoSearchOnLinks)be();else{const U=get("auto-search-banner"),xe=get("au\
to-search-on-btn"),Ne=get("auto-search-off-btn"),Ie=get("auto-search-remember");U&&xe&&Ne&&(Ie&&(Ie.
checked=!1),await new Promise(rt=>{U.classList.remove("hidden");const ut=o(wt=>{U.classList.add("hid\
den"),xe.onclick=null,Ne.onclick=null,rt(wt)},"cleanup");xe.onclick=()=>ut("enable"),Ne.onclick=()=>ut(
"disable")}).then(async rt=>{rt==="enable"?(be(),Ie&&Ie.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):H=!0}))}const pe=String(get("reasoning-effort").value||"").
toLowerCase(),te=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&pe==="non\
e",X={client_request_id:createClientRequestId(),thread_id:currentThreadId,message:J,model:get("model\
-select").value,image_urls:u,image_items:c,uploaded_image_urls:f,temporary_chat:temporaryChatEnabled,
enable_search:get("enable-search").checked,enable_url_context:get("enable-url-context")?get("enable-\
url-context").checked:!1,enable_maps:get("enable-maps")?get("enable-maps").checked:!1,enable_python:get(
"enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get("enable-file-crea\
tion")?get("enable-file-creation").checked:!0,enable_thinking:te?!1:get("enable-thinking").checked,thinking_level:get(
"thinking-level").value,thinking_budget:get("thinking-budget")?get("thinking-budget").value:null,reasoning_effort:get(
"reasoning-effort").value,enable_system_prompt:get("enable-sys-prompt").checked,enable_prompt_caching:get(
"enable-prompt-cache")?get("enable-prompt-cache").checked:!1,marker_system_prompt:D,safety_setting:get(
"safety-setting").value,tts_voice:isTtsModel()&&get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&
get("tts-voice-custom")?get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-langu\
age")?get("tts-language").value:null,tts_style:isTtsModel()&&get("tts-style")&&get("tts-style-wrap")&&
!get("tts-style-wrap").classList.contains("hidden")?get("tts-style").value:null,tts_speed:isTtsModel()&&
get("tts-speed")?get("tts-speed").value:null,image_size:isGptImageModel()&&get("gpt-image-size")?get(
"gpt-image-size").value:null,image_quality:isGptImageModel()&&get("gpt-image-quality")?get("gpt-imag\
e-quality").value:null,image_format:isGptImageModel()&&get("gpt-image-format")?get("gpt-image-format").
value:null,image_compression:isGptImageModel()&&get("gpt-image-compression")?get("gpt-image-compress\
ion").value:null,image_mask:Z,gemini_image_aspect:isGeminiImageModel()&&get("gemini-image-aspect")?get(
"gemini-image-aspect").value:null,gemini_image_size:isGeminiImageModel()&&get("gemini-image-size")?get(
"gemini-image-size").value:null,grok_image_aspect:isGrokImageModel()&&get("grok-image-aspect")?get("\
grok-image-aspect").value:null,grok_image_resolution:isGrokImageModel()&&get("grok-image-resolution")?
get("grok-image-resolution").value:null,grok_image_quality:isGrokImageModel()&&get("grok-image-quali\
ty")?get("grok-image-quality").value:null,grok_image_format:isGrokImageModel()&&get("grok-image-form\
at")?get("grok-image-format").value:null,grok_image_count:isGrokImageModel()&&get("grok-image-count")?
get("grok-image-count").value:null,ideogram_operation:isIdeogramModel()&&get("ideogram-image-operati\
on")?get("ideogram-image-operation").value:null,ideogram_aspect:isIdeogramModel()&&get("ideogram-ima\
ge-aspect")?get("ideogram-image-aspect").value:null,ideogram_resolution:isIdeogramModel()&&get("ideo\
gram-image-resolution")?get("ideogram-image-resolution").value:null,ideogram_quality:isIdeogramModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:ce,parent_id_explicit:me,disable_auto_search:H,
image_vision_model:currentVisionModel||null,coding_mode:I,coding_target:I?{id:C.candidate_id,code:C.
prompt_source?null:C.code,language:C.language||"text",key:C.key||null,message_id:C.message_id||null,
source:C.prompt_source?"prompt":"history",explicit:C.explicit===!0}:null,coding_candidates:I?A.map(U=>({
id:U.candidate_id,source:U.prompt_source?"prompt":"history",prompt_index:U.prompt_source?U.prompt_index:
null,code:U.prompt_source?null:U.code,language:U.language||"text",explicit:U.explicit===!0})):[],batch_mode:l,
canvas_mode:!!canvasModeEnabled};n&&(X.turnstile_token=n);const de=get("thread-custom-instruction");
de&&(X.thread_custom_instruction=de.value||""),activeGem?(X.system_prompt=activeGem.instruction,X.enable_system_prompt=
!0,X.gem_uuid=activeGem.uuid):X.gem_uuid=null;const ne=takePromptForSend(e.animate!==!1),re=animatePromptToMessage(
O,ne,j&&e.animate!==!1);setSendBtnToStopMode();const Se="ai-"+Date.now(),we=String(X.model||"").toLowerCase(),
Ke=!!X.enable_thinking||!!pe&&pe!=="none",Qe=we.includes("gemini")||we.includes("o1")||we.includes("\
o3")||we.includes("gpt-5")||we.includes("reasoning")&&!we.includes("non-reasoning"),je=Ke&&Qe;let Ue=buildPendingSkeletonHtml(
X.model,"API\u306B\u9001\u4FE1\u4E2D...");get("chat-container").insertAdjacentHTML("beforeend",`<div\
 class="flex justify-start mb-4 ai-pending-row fade-in"><div id="${Se}" class="message-bubble ai-pen\
ding-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${Ue}</div></\
div>`),j&&resumeChatAutoScroll();const ue=get(Se),qe=l?null:startPendingElapsed(ue,t);activeStreamingBubbleId=
Se,canvasModeEnabled&&resetCanvasPreviewPanel();let Xe=null;const pt=o(U=>!je||!ue?null:((!Xe||!ue.contains(
Xe))&&(Xe=ue.querySelector(".thought-content")),Xe||(ue.insertAdjacentHTML("afterbegin",'<div class=\
"thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed\
" data-placeholder="1"></div></div>'),Xe=ue.querySelector(".thought-content")),Xe&&(Xe.setAttribute(
"data-placeholder","1"),Xe.textContent=U||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
Xe),"ensureThoughtPlaceholder");je&&pt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const bt=currentThreadId,nn=nowPerfMs(),sn=Date.now();let an=!1,
Sn=!1,on=!1,Ct=null,St=null,_t=null,Bt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const Ft=o((U,xe)=>{if(!xe||U==="status"&&an||U==="thought"&&Sn||U==="content"&&on)return;const Ne=Math.
max(0,nowPerfMs()-nn);U==="status"?Ct=Ne:U==="thought"?St=Ne:U==="content"&&(_t=Ne),reportFirstTokenLatency(
{latency_seconds:Ne/1e3,latency_ms:Ne,thread_id:Bt||currentThreadId,job_id:currentJobId,model:X.model,
first_event_type:U,client_sent_at_ms:sn}),U==="status"?an=!0:U==="thought"?Sn=!0:U==="content"&&(on=
!0)},"maybeReportFirstEventLatency"),yt=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let Tt=!1,rn=!1,Mt=null,At=null,Et=!1;try{X.thread_id&&activeGem&&(threadGemMap[X.thread_id]=
activeGem,pendingGemForNewThread=null);const U=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(X),signal:abortController.signal}),ue);if(sendClientDebugLog("info",`Prompt stre\
am response status: ${U.status}`),!U.ok){const Re=await U.json().catch(()=>({})),ze=new Error(Re.error||
`HTTP ${U.status}`);throw ze.serverCode=Re.code||null,ze.serverModel=Re.model||X.model,ze.acceptedJobId=
Re.job_id||null,ze.acceptedThreadId=Re.thread_id||null,ze}Tt=!0,window.ConnectionMonitor&&(Et=!0,window.
ConnectionMonitor.operationStarted()),yt&&yt.setPhase("waiting"),ne.accept(),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1});const xe=o(()=>{if(!ue)return;const Re=ue.querySelector(".content-area");if(Re&&Re.
getAttribute("data-api-accepted")!=="1"&&(Re.setAttribute("data-api-accepted","1"),!updatePendingSkeletonStatus(
ue,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){
Re.outerHTML=buildPendingSkeletonHtml(X.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const ze=ue.querySelector(".content-area");ze&&ze.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
ue,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");xe(),(jt=window.ActivityLog)==null||jt.log("send.stream.start");let Ne=!1;const Ie=U.
body.getReader(),rt=new TextDecoder;let ut="",wt="",Ze="",xt=!0,tt=null,et=null,kt=null,Ln=!1;const Jt={};
let Dt=0,cn=!1;for(;!cn;){const{done:Re,value:ze}=await Ie.read();if(!Re&&!Ne&&(Ne=!0,(ln=window.ActivityLog)==
null||ln.log("send.stream.first")),Re&&((Vt=window.ActivityLog)==null||Vt.log("send.stream.done")),Re)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),yt&&yt.setPhase("receiving"),
ut+=rt.decode(ze,{stream:!0});let at=ut.split(`
`);ut=at.pop();let Ht=!1,dn=!1;for(let It of at)if(It.trim())try{const Ce=JSON.parse(It);if(Ce.type===
"thread_id"){xe();const $e=Ce.content!==null&&Ce.content!==void 0?String(Ce.content):Ce.content;$e&&
(Bt=$e,currentThreadId!==$e&&(currentThreadId=$e,history.pushState({},"","/c/"+$e)),activeGem&&(threadGemMap[$e]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Ce.type==="job_\
id"){xe(),currentJobId=Ce.content,l&&showToast("Batch\u767B\u9332","info");continue}if(Ce.type==="se\
arch_status"){kt=updateSearchBox(ue,kt,Ce.content);continue}if(Ce.type==="mcp"){handleMcpStreamEvent(
ue,Ce.content||{});continue}if(Ce.type==="mcp_decision_request"){openMcpDecisionModal(Ce.content||{});
continue}if(Ce.type==="status"){xe();const $e=Ce.content===null||Ce.content===void 0?"":String(Ce.content);
if(Ft("status",!!$e),xt&&ue){const nt=$e||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(
ue,nt,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Ye=ue.querySelector(".content-area");Ye&&(Ye.outerHTML=buildPendingSkeletonHtml(X.model,nt),updatePendingSkeletonStatus(
ue,nt,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
je&&pt($e||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(xt){beginPendingToStreamTransition(
ue);const $e=ue.querySelector(".content-area");$e&&($e.innerHTML=""),xt=!1}if(Ce.type==="coding_diff")
appendCodingLiveDiff(ue,Ce.content||{}),Ft("content",!0);else if(Ce.type==="thought"){if(tt||(tt=ue.
querySelector(".thought-content")),Ze+=Ce.content,Ft("thought",!!Ce.content),!tt){const $e='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';kt?kt.insertAdjacentHTML(
"afterend",$e):ue.insertAdjacentHTML("afterbegin",$e),tt=ue.querySelector(".thought-content")}if(tt&&
tt.getAttribute("data-placeholder")==="1"){if(tt.textContent="",tt.removeAttribute("data-placeholder"),
tt){const $e=tt.parentElement.querySelector(".thought-header");$e&&$e.classList.remove("thinking-shi\
mmer")}Ze=Ce.content}tt.classList.remove("collapsed"),dn=!0}else if(Ce.type==="image_analysis"){const $e=Ce.
content===null||Ce.content===void 0?"":String(Ce.content);if(!ue)continue;let nt=ue.querySelector(".\
image-analysis-box");if(!nt){const mt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';kt?kt.insertAdjacentHTML("afterend",mt):ue.insertAdjacentHTML("afterbegin",mt),nt=ue.querySelector(
".image-analysis-box")}const Ye=nt.querySelector(".image-analysis-text");Ye&&(Ye.textContent=$e)}else if(Ce.
type==="python"){const $e=Ce.content||{},nt=$e.id||`py_${Date.now()}`;if(!Jt[nt]){const mt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${nt}" data-collapsed="true" data-code-key="${nt}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;kt?kt.insertAdjacentHTML("afterend",mt):ue.insertAdjacentHTML(
"afterbegin",mt),Jt[nt]=ue.querySelector(`[data-py-id="${nt}"]`)}const Ye=Jt[nt];if(Ye){if($e.code!==
void 0){const mt=$e.code==null?"":String($e.code),Pt=Ye.querySelector(".python-code");Pt&&(Pt.textContent=
mt,Pt.removeAttribute("data-highlighted"),queueHighlight(Ye,mt));const Ot=Ye.querySelector('.copy-bt\
n[data-copy="code"]');Ot&&Ot.setAttribute("data-code",encodeURIComponent(mt).replace(/'/g,"%27"))}if($e.
output!==void 0){const mt=$e.output==null?"":String($e.output),Pt=Ye.querySelector(".python-output");
Pt&&(Pt.textContent=mt);const Ot=Ye.querySelector('.copy-btn[data-copy="output"]');Ot&&Ot.setAttribute(
"data-code",encodeURIComponent(mt).replace(/'/g,"%27"))}}}else if(Ce.type==="content"){const $e=Ce.content===
null||Ce.content===void 0?"":String(Ce.content);wt+=$e,/[`~]/.test($e)&&activateDeferredCodingModeFromStream(
wt),et||(et=ue.querySelector(".content-area")||document.createElement("div"),et.className="prose pro\
se-invert text-sm break-words",ue.contains(et)||ue.appendChild(et)),Ht=!0,Ft("content",!!$e)}else if(Ce.
type==="error"){Ln=!0,cn=!0,ue.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Ce.content)),
showToast(Ce.content||"Unknown error","error",!0);break}}catch{}if(dn&&tt&&(tt.textContent=Ze,userAutoScroll&&
(tt.scrollTop=tt.scrollHeight)),Ht&&et){const It=Date.now();if(It-Dt>100){const Ce=snapshotCodeCollapse(
et);renderAiMarkdownInto(et,wt,{incrementalMath:!0}),applyCodeCollapse(et,Ce,!0),Dt=It}}scrollToBottom()}
if(yt&&yt(),et){const Re=snapshotCodeCollapse(et);renderAiMarkdownInto(et,wt,{incrementalMath:!0}),applyCodeCollapse(
et,Re,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),ue)if(queueHighlight(ue,wt),enableLatencyMetrics){
const Re=nowPerfMs()-nn;reportFirstTokenLatency({is_total:!0,latency_seconds:Re/1e3,latency_ms:Re,thread_id:Bt||
currentThreadId,job_id:currentJobId,model:X.model,client_sent_at_ms:sn,client_done_at_ms:Date.now()});
let ze='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',at=null;Ct!==null&&(at=Ct),St!==null&&(at===null||St<at)&&(at=
St),_t!==null&&(at===null||_t<at)&&(at=_t),at!==null&&(ze+=`<div>Initial: ${(at/1e3).toFixed(2)}s</d\
iv>`),_t!==null&&_t!==at&&(ze+=`<div>Content: ${(_t/1e3).toFixed(2)}s</div>`),ze+=`<div class="font-\
bold text-gray-300">Total: ${(Re/1e3).toFixed(2)}s</div>`,currentJobId&&(ze+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),ze+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,ze+="</div>",ue.insertAdjacentHTML("beforeend",ze)}else ue.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),ue&&ue.querySelectorAll(".thought-\
content").forEach(ze=>ze.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!Q}),!Ln&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(Re=>Re.json()).then(Re=>{Re.title&&
(document.title=Re.title+" - AI Chat",setCurrentChatHeaderTitle(Re.title),loadThreads())}):loadThreads(
!1)}catch(U){(vt=window.ActivityLog)==null||vt.log(U.name==="AbortError"?"send.cancelled":"send.erro\
r"),re();let xe=!1;const Ne=U.name==="AbortError"&&isManualStopAbortForThread(bt);if(U.name==="Abort\
Error"&&!Ne&&(xe=await syncThreadAfterAbortedStream(bt,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${U.message}`),!Tt){O&&O.remove();const Ie=ue&&ue.closest(
".fade-in");Ie&&Ie.remove(),delete messageStore[R],delete messageMeta[R]}if(U.serverCode==="request_\
already_accepted"&&U.acceptedJobId&&U.acceptedThreadId)Tt=!0,Mt={job_id:U.acceptedJobId,thread_id:String(
U.acceptedThreadId),model:X.model,pendingStartedAt:l?null:t},ne.accept();else if(Tt&&!Ne)At={job_id:normalizeJobIdForUi(
currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):null,model:X.model,pendingStartedAt:l?
null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\
\u307E\u3059\u3002","warning",!1);else if(U.serverCode==="turnstile_required"){const Ie=await getTurnstileToken();
Ie?(await verifyTurnstileOnServer(Ie,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(U.serverCode==="api_key_missing"){const Ie=U.serverModel||X.model,rt=await showApiKeyRequiredModalAsync(
Ie);rt==="set"?rn=!0:rt==="switch"?showModal("model-modal"):showToast(U.message||`${getModelNameById(
Ie)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(U.
name!=="AbortError"){const Ie="Connection Error: "+U.message;showToast(Ie,"error",!0)}Tt||ne.restore(),
Q&&!xe&&E()}finally{qe&&qe(),Et&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
yt&&yt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===Se&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,editingMessageId=null,setEditUi(!1)}if(Mt){const U=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=Mt.thread_id,(U!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
Mt,currentThreadId)}if(At&&At.thread_id)return reconnectPendingStreamUntilAvailable(At,At.thread_id);
if(rn&&ne.canRetry())return sendMessageLogged({animate:!1})}o(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var ce,me,ae,Te;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const E=a.
querySelector(".content-area");E?E.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const r=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const l=currentThreadId,
c=i.toLowerCase(),u=c.includes("gemini")||c.includes("o1")||c.includes("o3")||c.includes("gpt-5")||c.
includes("reasoning")&&!c.includes("non-reasoning");let f=null;const h=o(E=>!u||!a?null:((!f||!a.contains(
f))&&(f=a.querySelector(".thought-content")),f||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),f=a.querySelector(".thought-content")),f&&(f.setAttribute("data-pla\
ceholder","1"),f.textContent=E||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),f),
"ensureThoughtPlaceholder");u&&h("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",w="",v="",x=!0,S=null,L=null,C=null,A=!1;const I={};let J=0,D=!1;const ie=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let j=!1,Q=!1;try{const E=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!E.ok)throw new Error(`Resume failed (${E.status})`);window.ConnectionMonitor&&
(Q=!0,window.ConnectionMonitor.operationStarted()),ie&&ie.setPhase("waiting"),(ce=window.ActivityLog)==
null||ce.log("send.stream.start");let R=!1;const Z=E.body.getReader(),O=new TextDecoder;for(;!D;){const{
done:H,value:G}=await Z.read();if(!H&&!R&&(R=!0,(me=window.ActivityLog)==null||me.log("send.stream.f\
irst")),H&&((ae=window.ActivityLog)==null||ae.log("send.stream.done")),H)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),ie&&ie.setPhase("receiving"),y+=O.decode(G,{stream:!0});let P=y.
split(`
`);y=P.pop();let K=!1,be=!1;for(let pe of P)if(pe.trim())try{const te=JSON.parse(pe);if(te.type==="j\
ob_id"){currentJobId=te.content||t;continue}if(te.type==="search_status"){C=updateSearchBox(a,C,te.content);
continue}if(te.type==="mcp"){handleMcpStreamEvent(a,te.content||{});continue}if(te.type==="mcp_decis\
ion_request"){openMcpDecisionModal(te.content||{});continue}if(te.type==="status"){const X=te.content===
null||te.content===void 0?"":String(te.content);if(x&&a){const de=X||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";
if(!updatePendingSkeletonStatus(a,de,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ne=a.querySelector(".content-area");ne&&(ne.outerHTML=buildPendingSkeletonHtml(i,de),updatePendingSkeletonStatus(
a,de,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
u&&h(X||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(x){beginPendingToStreamTransition(
a);const X=a.querySelector(".content-area");X&&(X.innerHTML=""),x=!1}if(te.type==="coding_diff")appendCodingLiveDiff(
a,te.content||{});else if(te.type==="thought"){if(S||(S=a.querySelector(".thought-content")),v+=te.content,
!S){const X='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this\
)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"><\
/div></div>';C?C.insertAdjacentHTML("afterend",X):a.insertAdjacentHTML("afterbegin",X),S=a.querySelector(
".thought-content")}if(S&&S.getAttribute("data-placeholder")==="1"){if(S.textContent="",S.removeAttribute(
"data-placeholder"),S){const X=S.parentElement.querySelector(".thought-header");X&&X.classList.remove(
"thinking-shimmer")}v=te.content}S.classList.remove("collapsed"),be=!0}else if(te.type==="image_anal\
ysis"){const X=te.content===null||te.content===void 0?"":String(te.content);if(!a)continue;let de=a.
querySelector(".image-analysis-box");if(!de){const re='<div class="image-analysis-box mb-2 p-2 bg-bl\
ue-900/20 border border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"\
><i class="fas fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] te\
xt-gray-300"></div></div>';C?C.insertAdjacentHTML("afterend",re):a.insertAdjacentHTML("afterbegin",re),
de=a.querySelector(".image-analysis-box")}const ne=de.querySelector(".image-analysis-text");ne&&(ne.
textContent=X)}else if(te.type==="python"){const X=te.content||{},de=X.id||`py_${Date.now()}`;if(!I[de]){
const re=`<div class="code-wrapper python-box collapsed" data-py-id="${de}" data-collapsed="true" da\
ta-code-key="${de}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 Python Execution</span><div class="code-actions"><button class="code-toggle" aria-expanded="false" \
title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data\
-copy="code" data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><\
button class="copy-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class=\
"fas fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div\
 class="python-label">Code</div><pre><code class="hljs language-python python-code"></code></pre></d\
iv><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language\
-plaintext python-output"></code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",re):a.insertAdjacentHTML(
"afterbegin",re),I[de]=a.querySelector(`[data-py-id="${de}"]`)}const ne=I[de];if(ne){if(X.code!==void 0){
const re=X.code==null?"":String(X.code),Se=ne.querySelector(".python-code");Se&&(Se.textContent=re,Se.
removeAttribute("data-highlighted"),queueHighlight(ne,re));const we=ne.querySelector('.copy-btn[data\
-copy="code"]');we&&we.setAttribute("data-code",encodeURIComponent(re).replace(/'/g,"%27"))}if(X.output!==
void 0){const re=X.output==null?"":String(X.output),Se=ne.querySelector(".python-output");Se&&(Se.textContent=
re);const we=ne.querySelector('.copy-btn[data-copy="output"]');we&&we.setAttribute("data-code",encodeURIComponent(
re).replace(/'/g,"%27"))}}}else if(te.type==="content"){const X=te.content===null||te.content===void 0?
"":String(te.content);w+=X,/[`~]/.test(X)&&activateDeferredCodingModeFromStream(w),L||(L=a.querySelector(
".content-area")||document.createElement("div"),L.className="prose prose-invert text-sm break-words",
a.contains(L)||a.appendChild(L)),K=!0}else if(te.type==="error"){A=!0,D=!0,a.insertAdjacentHTML("bef\
oreend",buildChatErrorBubbleHtml(te.content)),showToast(te.content||"Unknown error","error",!0);break}}catch{}
if(be&&S&&(S.textContent=v,userAutoScroll&&(S.scrollTop=S.scrollHeight)),K&&L){const pe=Date.now();if(pe-
J>100){const te=snapshotCodeCollapse(L);renderAiMarkdownInto(L,w,{incrementalMath:!0}),applyCodeCollapse(
L,te,!0),J=pe}}scrollToBottom()}if(ie&&ie(),L){const H=snapshotCodeCollapse(L);renderAiMarkdownInto(
L,w,{incrementalMath:!0}),applyCodeCollapse(L,H,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,
w),a&&a.querySelectorAll(".thought-content").forEach(G=>G.classList.add("collapsed")),await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0}),loadThreads(!1)}catch(E){(Te=window.ActivityLog)==null||
Te.log(E.name==="AbortError"?"send.cancelled":"send.stream.error");const R=E.name==="AbortError"&&isManualStopAbortForThread(
l);E.name==="AbortError"&&!R&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
R||(j=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{r&&r(),Q&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
ie&&ie(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
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
-1,u=r.querySelectorAll("[data-thread-id]").length;if(c===0&&u>0&&String(n||"").trim()){snapshotSidebarHistory(
"loadThreads-keep-existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" cla\
ss="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="\
fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scro\
ll-sentinel"></div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const u=String(
c.id),f=document.createElement("div"),h=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",v=u===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${v}`,f.dataset.threadId=u,f.dataset.threadTitle=c.title||"",f.innerHTML=`<div class="flex \
items-center gap-1 truncate flex-1"><button class="${h} hover:text-yellow-400 px-1" onclick="toggleB\
ookmark(event, '${u}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(
c.title||"No Title")}</span>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 m\
d:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:tex\
t-white px-1 transition" onclick="renameThread(event, '${u}')"><i class="fas fa-pen text-xs"></i></b\
utton><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, \
'${u}')"><i class="fas fa-trash text-xs"></i></button></div>`,f.onclick=x=>{x.target.closest("button")||
x.target.closest("[data-thread-actions]")||loadMessages(u)},l?r.insertBefore(f,l):r.appendChild(f)}),
hasMoreThreads=!!a.has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendere\
d count="+a.threads.length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){
console.error("Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=
!1,updateThreadHighlighting(),snapshotSidebarHistory("loadThreads-finally")}}o(loadThreads,"loadThre\
ads");function initPullToRefresh(e,t){const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,
r=88,l=52,c=.5,u=8;let f=0,h=!1,y=0,w=null;const v=o(()=>get(i),"indicatorEl"),x=o(()=>{const C=v();
return C?C.querySelector(".ptr-pull-label"):null},"labelEl"),S=o(C=>{const A=v();if(!A)return;A.style.
height=Math.min(C,r)+"px",A.classList.toggle("active",C>2),A.classList.toggle("pull-ready",C>=a);const I=x();
I&&(I.textContent=C>=a?"\u96E2\u3057\u3066\u66F4\u65B0":"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},
"applyPullUI"),L=o(()=>{const C=v();C&&(C.style.height="0px",C.classList.remove("active","pull-ready",
"refreshing"),C.classList.remove("dragging"))},"resetPullUI");n.addEventListener("touchstart",C=>{if(w){
h=!1;return}if(n.scrollTop>0){h=!1;return}const A=C.touches[0];A&&(f=A.clientY,y=0,h=!0)},{passive:!0}),
n.addEventListener("touchmove",C=>{if(!h||w)return;if(n.scrollTop>0){h=!1;return}const A=C.touches[0];
if(!A)return;const I=A.clientY-f;if(I<=0){y>0&&(y=0,S(0)),h=!1;return}const J=v();J&&!J.classList.contains(
"dragging")&&J.classList.add("dragging"),y=Math.min(I*c,r),S(y),I>=u&&C.preventDefault()},{passive:!1}),
n.addEventListener("touchend",()=>{if(!h||(h=!1,w))return;const C=v();C&&C.classList.remove("draggin\
g");const A=y>=a;if(y=0,!A){L();return}let I;try{I=t()}catch{I=null}const J=v();if(J){J.classList.add(
"refreshing"),J.style.height=l+"px";const D=J.querySelector(".ptr-pull-label");D&&(D.textContent="\u66F4\u65B0\
\u4E2D...")}I&&typeof I.then=="function"?(w=I,I.catch(()=>{}).finally(()=>{w=null,L()})):(w=Promise.
resolve(),setTimeout(()=>{w=null,L()},400))}),n.addEventListener("touchcancel",()=>{h=!1,y=0,L()})}o(
initPullToRefresh,"initPullToRefresh");const initThreadPullToRefresh=o(()=>initPullToRefresh("thread\
-list",()=>loadThreads(!1)),"initThreadPullToRefresh"),initGemPullToRefresh=o(()=>initPullToRefresh(
"gem-list",()=>loadGems()),"initGemPullToRefresh"),initPullToRefreshAll=o(()=>{initThreadPullToRefresh(),
initGemPullToRefresh()},"initPullToRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;
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
const n=++threadLoadSequence,i=t.targetMessageId||new URLSearchParams(location.search).get("message");
window.closeHistoryModal&&window.closeHistoryModal();const a=!!t.preserveDraft,r=!!t.silent;r||resumeChatAutoScroll(
{scroll:!1});const l=r?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",u="",f=[];
if(a){const y=get("prompt-input");c=y?y.value:"",u=y?y.style.height:"",f=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,setEditUi(!1)}else cancelEdit();r||playChatTransition("history");const h=e!=
null?String(e):e;h!==currentThreadId&&(modelChosenManually=!1),currentThreadId=h,t.skipHistory||history.
pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(currentThreadId),get("wel\
come-screen").classList.add("hidden"),r||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const y=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);y.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const w=i;w&&/^\d+$/.test(String(w))&&y.
searchParams.set("before_id",String(Number(w)+1));const v=await apiFetch(y.toString());if(!v.ok)throw new Error(
`thread request failed (${v.status})`);const x=await v.json();if(!x||!Array.isArray(x.messages))throw new Error(
"invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(x&&x.title),
allMessages=x.messages,renderBranchOrigin(x.branch_origin),threadHasOlderMessages=!!x.has_older_messages,
oldestLoadedMessageId=x.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const S=(allMessages||
[]).filter(A=>A.role==="user"&&A.content).map(A=>A.content);if(promptHistory=[...new Set(S.slice().reverse())],
historyIndex=-1,tempPrompt="",currentThreadPending=x.pending_job||null,setTemporaryChatUiState(!!(x&&
x.is_temporary)),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(!0),get("thread-c\
ustom-instruction")&&(get("thread-custom-instruction").value=x.custom_instruction||""),get("enable-p\
rompt-cache")&&(get("enable-prompt-cache").checked=!!x.enable_prompt_caching,updatePromptCacheUi()),
x.last_model&&!modelChosenManually&&selectModelById(x.last_model),x.last_gem_uuid&&loadedGems.length>
0){const A=loadedGems.find(I=>I.uuid===x.last_gem_uuid);A&&(threadGemMap[currentThreadId]=A,applyActiveGem(
A))}const L=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),C=i;if(C&&
allMessages.some(A=>String(A.id)===String(C))?currentLeafId=Number(C):L&&allMessages.find(A=>String(
A.id)===String(L))?currentLeafId=L:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(r?{silent:r,keepScroll:r}:{silent:r,keepScroll:r,animate:!0}),
a||restoreLatestPromptOptions(),r&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):r||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!r&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const A=get("prompt-input");A&&(A.value=c||"",
u?A.style.height=u:A.style.height="auto"),currentImageUrls=f,currentImageUrls&&currentImageUrls.length?
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
if(l.length){const c=new Set(allMessages.map(f=>f.id)),u=l.filter(f=>!c.has(f.id));u.length&&(allMessages=
u.concat(allMessages))}if(threadHasOlderMessages=!!r.has_older_messages,oldestLoadedMessageId=r.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
o(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;let r=null;if(i&&(r=a.scrollTop),allMessages.
length===0){a.replaceChildren(),currentParentId=null,updateTotalTokenBar(0);return}const l={};allMessages.
forEach(v=>{l[v.id]=v,v.childrenIds=[]}),allMessages.forEach(v=>{v.parent_id&&l[v.parent_id]&&l[v.parent_id].
childrenIds.push(v.id)}),(!currentLeafId||!l[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null);const c=[];let u=l[currentLeafId];for(;u;)c.unshift(u),u=l[u.parent_id];const f=buildTokenTotals(
c),h=buildTokenTotals(allMessages),y=document.createDocumentFragment();if(threadHasOlderMessages){const v=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
x=loadingOlderMessages?"disabled":"",S=document.createElement("div");S.className="mb-3 text-center",
S.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${x}><i class="fas fa-clock-rotate-left mr-1"></i>${v}</button>`,y.appendChild(S)}
c.forEach(v=>{const x=v.parent_id?l[v.parent_id]:null,S=x?x.childrenIds:allMessages.filter(C=>!C.parent_id).
map(C=>C.id),L=S.length>1?{current:S.indexOf(v.id)+1,total:S.length,siblings:S}:null;renderMessage(v.
id,v.role,v.content,v.image_url,v.thought_data,v.model,L,n,v.quote_text,v.tokens,v.tokens_in,v.tokens_out,
v.is_encrypted,v.tokens_content,v.tokens_thought,y,!1,v.parent_id,v.gem_name,v.batch_job,v.prompt_options&&
typeof v.prompt_options=="object"?v.prompt_options.image_mask:null)});const w=currentThreadPending;if(w&&
!isPendingJobSuppressed(w.job_id)){const v=w.message_id,x=new Set(c.map(C=>C.id)),S=c.length?c[c.length-
1]:null;if(v&&x.has(v)&&currentLeafId===v||!v&&S&&S.role==="user"){const C=w.job_id?`pending-${w.job_id}`:
null;renderPendingMessage(y,n,!1,C,w.model||null)}}if(a.replaceChildren(y),displayMissingMessageFiles(
a,c),updateTotalTokenBar(f.tokens_total,f,h),currentParentId=currentLeafId,i&&r!==null?restoreThreadTreeScroll(
a,r):scrollToBottom(),lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(
a),c.length){const v=c[c.length-1]&&c[c.length-1].content;queueMathTypeset(a,v)}}o(renderThreadTree,
"renderThreadTree");function restoreThreadTreeScroll(e,t){if(!e)return;const n=e.scrollHeight-e.clientHeight;
userAutoScroll&&!chatManualPauseIntent?e.scrollTop=e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,
n)),chatLastScrollTop=e.scrollTop,syncScrollToBottomButton()}o(restoreThreadTreeScroll,"restoreThrea\
dTreeScroll");function switchVersion(e){currentLeafId=e;const t={};allMessages.forEach(i=>{t[i.id]=i,
i.childrenIds=[]}),allMessages.forEach(i=>{i.parent_id&&t[i.parent_id]&&t[i.parent_id].childrenIds.push(
i.id)});let n=e;if(!t[n]){currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null,
renderThreadTree({animate:!0}),restoreLatestPromptOptions();return}for(;t[n]&&t[n].childrenIds.length>
0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree({animate:!0}),restoreLatestPromptOptions()}
o(switchVersion,"switchVersion");async function loadGems(){try{const t=await(await apiFetch(CHAT_CONFIG.
urls.handleGems)).json();loadedGems=t;const n=get("gem-list");if(!n)return;n.innerHTML='<div id="gem\
-pull-indicator" class="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-\
icon"></i><i class="fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span\
></div>',Array.isArray(t)&&t.forEach(i=>{const a=document.createElement("div");a.className="gem-item\
 p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 flex justify-between items-cente\
r group",a.innerHTML=`<div class="flex items-center gap-2 overflow-hidden"><i class="fas fa-gem text\
-blue-500"></i><span class="truncate">${escapeHtml(i.name)}</span></div><div class="flex items-cente\
r gap-1"><button class="text-gray-400 hover:text-blue-400 opacity-100 md:opacity-0 md:group-hover:op\
acity-100 px-2 transition" onclick="openEditGemModal(event,'${i.uuid}')"><i class="fas fa-pencil-alt\
 text-[10px]"></i></button><button class="text-gray-400 hover:text-red-400 opacity-100 md:opacity-0 \
md:group-hover:opacity-100 px-2 transition" onclick="deleteGem(event,'${i.uuid}')"><i class="fas fa-\
trash text-[10px]"></i></button></div>`,a.onclick=r=>{r.target.closest("button")||activateGem(i)},n.
appendChild(a)})}catch(e){console.error("Failed to load gems:",e)}}o(loadGems,"loadGems");function setGemDefaultModelSelect(e){
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
if(playChatTransition("new"),threadLoadSequence++,abortController&&abortController.abort(),modelChosenManually=
!1,cancelEdit(),restoreLatestPromptOptions(),resetUploadState(),stopTemporaryChatHeartbeat(),setTemporaryChatUiState(
!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,renderBranchOrigin(null),allMessages=
[],promptHistory=[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=null,
loadingOlderMessages=!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(
0),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
e.skipHistory||history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").
classList.remove("hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("threa\
d-custom-instruction").value=""),get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!1,
updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(activeGem):applyActiveGem(null),loadThreads();
const t=threadLoadSequence;fetchSettingsSnapshot().then(n=>{if(threadLoadSequence!==t||currentThreadId||
modelChosenManually||e.preserveGem&&activeGem)return;const i=n.use_last_chat_settings&&n.last_model||
n.default_model;i&&MODELS.some(a=>a.items.some(r=>r.id===i))&&selectModelById(i)}).catch(n=>console.
warn("Failed to load chat defaults",n)),window.innerWidth<768&&get("overlay").click()}o(startNewChat,
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
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(h){sendClientDebugLog("error","Payload c\
onstruction failed: "+h.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const h=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+h.status);let y=!0;if(f){sendClientDebugLog(
"info","Starting POST request for user settings");const w=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(f)});y=w.ok,w.ok&&window.
applySavedUserSystemPromptSettings(f),sendClientDebugLog("info","POST request finished, status: "+w.
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
{method:"DELETE"}),loadGems())}o(deleteGem,"deleteGem");function renameThread(e,t){var y;e.stopPropagation();
const n=get("thread-rename-modal"),i=get("thread-rename-form"),a=get("thread-rename-input"),r=get("t\
hread-rename-save"),l=get("thread-rename-cancel"),c=get("thread-rename-error"),u=e.currentTarget;a.value=
((y=u.closest("[data-thread-id]"))==null?void 0:y.dataset.threadTitle)||"",c.textContent="",r.disabled=
!1;let f=!1;const h=o(()=>{f||(hideModal("thread-rename-modal"),u.focus())},"close");l.onclick=h,n.onclick=
w=>{w.target===n&&h()},n.onkeydown=w=>{w.key==="Escape"&&(w.preventDefault(),w.stopPropagation(),h()),
w.key==="Tab"&&(w.shiftKey&&document.activeElement===a?(w.preventDefault(),l.focus()):!w.shiftKey&&document.
activeElement===l&&(w.preventDefault(),a.focus()))},i.onsubmit=async w=>{w.preventDefault();const v=a.
value.trim();if(f||!v){v||a.focus();return}f=!0,r.disabled=!0,l.disabled=!0;try{const x=await apiFetch(
CHAT_CONFIG.urls.updateTitle.replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},
body:JSON.stringify({title:v})}),S=await x.json();if(!x.ok)throw new Error("rename failed");currentThreadId===
String(t)&&setCurrentChatHeaderTitle(S.title||v),f=!1,h(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
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
n.url:buildPdfAttachmentUrl(i),u=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:r,isImage:l,url:c,previewUrl:u}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=o(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(u=>maybeNeedsMathJax(u.content)||maybeNeedsMathJax(u.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",r=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(u=>{const f=u.
role==="user",h=u.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(u.quote_text)}\
</div>`:"",y=u.thought_text?`<div class="thought">${escapeHtml(u.thought_text)}</div>`:"",w=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(u.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(u.content||"")}</div>`,v=buildPdfMessageAttachments(u),x=v.length?`<div \
class="attachments">${v.map(C=>C.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(C.previewUrl)}\
" alt="${pdfEscapeAttr(C.filename)}"><div class="file-caption">${pdfEscapeAttr(C.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(C.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
C.filename)}</span><span class="file-source">${pdfEscapeAttr(C.source)}</span></span></a></div>`).join(
"")}</div>`:"",S=[];u.model&&!f&&S.push(u.model),u.tokens!==null&&u.tokens!==void 0&&S.push(`tokens:${u.
tokens}`),u.tokens_in!==null&&u.tokens_in!==void 0&&S.push(`in:${u.tokens_in}`),u.tokens_out!==null&&
u.tokens_out!==void 0&&S.push(`out:${u.tokens_out}`),u.tokens_thought!==null&&u.tokens_thought!==void 0&&
S.push(`thought:${u.tokens_thought}`),u.is_encrypted&&S.push("encrypted"),u.parent_id!==null&&u.parent_id!==
void 0&&S.push(`parent:#${u.parent_id}`);const L=S.length?`<div class="message-meta">${pdfEscapeAttr(
S.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(u.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${h}
                            ${w}
                            ${y}
                            ${x}
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
async()=>{try{const h=r.contentDocument,y=r.contentWindow;if(!h||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(C=>maybeNeedsMathJax(C.content)||maybeNeedsMathJax(C.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),h.fonts&&h.fonts.ready)try{await h.fonts.ready}catch{}t.update(60);const x=Array.from(h.
images||[]),S=Promise.all(x.map(C=>C.complete?Promise.resolve():new Promise(A=>{C.addEventListener("\
load",A,{once:!0}),C.addEventListener("error",A,{once:!0})})));if(await Promise.race([S,new Promise(
C=>setTimeout(C,5e3))]),t.update(80),h.getElementById("MathJax-script")){let C=0;for(;C<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(A=>setTimeout(A,50)),C++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(A){console.error(
"PDF MathJax typeset failed",A)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
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
if(n)return;const f=lib.favoritesOnly&&!a?"fa-star":"fa-search",h=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${f}"></i></div>\
<p class="lib-empty-title">${h}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(a)):r).forEach(f=>{
try{const h=renderLibraryItem(f,c++);t.appendChild(h)}catch{}})}o(renderLibraryGrid,"renderLibraryGr\
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
h=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${f}\
" aria-label="${f}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${u}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${r}</div>${l}${h}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",S=>{S.shiftKey&&S.preventDefault()}),n.onclick=
S=>{if(S&&S.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const L=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),C=L.findIndex(I=>I.dataset.filepath===lib.anchorPath),
A=L.indexOf(n);if(C!==-1&&A!==-1){const I=Math.min(C,A),J=Math.max(C,A);for(let D=I;D<=J;D++){const ie=L[D].
dataset.filepath;ie&&(lib.selected.add(ie),L[D].classList.add("is-selected"))}try{const D=window.getSelection&&
window.getSelection();D&&D.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(S=>{S.onclick=L=>{L.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const v=n.querySelector(".lib-del-btn");v&&(v.onclick=async S=>{S.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const x=n.querySelector(".lib-favorite-btn");return x&&
(x.onclick=async S=>{S.stopPropagation(),x.disabled=!0;try{const L=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
C=await L.json().catch(()=>({}));if(!L.ok||typeof C.is_favorite!="boolean")throw new Error(C.error||
"favorite update failed");e.is_favorite=C.is_favorite,renderLibraryGrid(),showToast(C.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),x.disabled=!1}}),n}o(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
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
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const h=u.updated_at?new Date(u.
updated_at).toLocaleString():"";f.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(u.title||"")}">${escapeHtml(u.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(h)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=f.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(u.id))}),i.appendChild(f)}),l.has_more){const u=document.
createElement("p");u.className="text-[11px] text-gray-500 text-center pt-2",u.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(u)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}o(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let r=null,l=!1;try{const u=getLibSortOrder(),f=getLibSearchQuery(),h=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(h),sort:u,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),w=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!w.ok)throw new Error("HTTP "+w.status);r=await w.json(),l=!0}catch(u){i=u}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(r)?r:
r&&Array.isArray(r.files)?r.files:[];r&&!Array.isArray(r)&&(lib.totalCount=Number(r.total)||0,lib.hasMore=
!!r.has_more,lib.nextOffset=(Number(r.offset)||0)+(Number(r.limit)||c.length));try{const u=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,h=new Set(c.map(w=>w&&w.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(w=>{if(c.length>=LIBRARY_PAGE_SIZE||!w||h.has(w))return;const v=getAttachmentNameForPath(
w)||w.split("/").pop()||w,x=(v.split(".").pop()||"").toLowerCase(),S=["png","jpg","jpeg","webp","gif"].
includes(x)?"image":"file",L=S==="image"?f+w:null;c.unshift({filename:v,original_filename:v,filepath:w,
url:u+w,thumbnail_url:L,type:S,ext:x,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),h.add(w)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const u=c.filter(h=>h&&h.filepath&&
h.url);let f=[];if(e){const h=new Set(lib.files.map(y=>y.filepath));f=u.filter(y=>!h.has(y.filepath)),
lib.files.push(...f)}else lib.files=u;lib.files.forEach(h=>{h&&h.filepath&&setAttachmentNameForPath(
h.filepath,h.filename||h.original_filename||"")}),lib.fileSet=new Set(lib.files.map(h=>h.filepath)),
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
total-u[1].total).forEach(([u,f])=>{const h=document.createElement("div");h.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",h.innerHTML=`
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
batchProviderLabel(a.provider)),u=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
h=escapeHtml(a.status_text||""),y=batchStateTone(a.state),w=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",v=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",x=a.is_active?"":'<button type="butto\
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
                    <div class="batch-job-status mt-2 text-[11px] break-words">${h}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${w}${v}${x}
                    </div>`;const S=r.querySelector("[data-batch-open]");S&&(S.onclick=()=>{window.closeBatchModal(),
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
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),u=o(
()=>{r.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=o(h=>{h.key==="Enter"&&(h.preventDefault(),
r.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),r.onclick=async()=>{
const h=get("api-key-modal-input").value.trim();if(!h){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=h);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:h})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=h)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",f),u(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",f),u(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",f),u(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const h=get("api-key-modal-input");h&&h.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function r(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const u=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,u)}catch{}finally{
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
selected:new Set,counts:{},total:0,data:null},c={accounts:[],selected:new Set},u=o(()=>get("bot-log-\
detail"),"view"),f=o(()=>get("bot-log-list-view"),"listView"),h=o(E=>{if(!E)return"-";const R=new Date(
E);return isNaN(R.getTime())?String(E):R.toLocaleString("ja-JP")},"formatTime"),y=o(E=>{const R=Math.
max(0,Math.floor(Number(E)||0));return R>=60?`${Math.floor(R/60)}\u5206${R%60?R%60+"\u79D2":""}`:`${R}\
\u79D2`},"formatDuration"),w=o((E,R)=>R==null||R===""?"-":E==="lock_source"?a[R]||String(R):E==="cli\
ent"||E==="origin_client"?n[R]||String(R):E==="action"?r[R]||String(R):E==="by"?R==="self"?"\u672C\u4EBA":
R==="admin"?"\u7BA1\u7406\u8005":String(R):E==="enabled"?R?"ON":"OFF":E==="remaining_seconds"||E==="\
lock_seconds"?y(R):E==="applied_to"&&Array.isArray(R)?R.map(Z=>a[Z]||Z).join("\u30FB"):typeof R=="ob\
ject"?JSON.stringify(R,null,2):String(R),"formatDetailValue"),v=o(E=>{const R=String(E||"").trim();if(!R)
return"";let Z=null;if(R.startsWith("{"))try{Z=JSON.parse(R)}catch{Z=null}if(Z&&typeof Z=="object"&&
!Array.isArray(Z)){const O=Object.keys(Z).map(H=>{const G=w(H,Z[H]),P=i[H]||H;return G.length>120||G.
includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(P)}</div><pre class="bot-log-pre">${escapeHtml(
G)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(P)}:</span> <span class="text-gray-3\
00">${escapeHtml(G)}</span></div>`}).join("");return R.length>600?`<details class="mt-1"><summary cl\
ass="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${O}</details>`:`<div cla\
ss="mt-1 space-y-1">${O}</div>`}return R.length>300?`<details class="mt-1"><summary class="cursor-po\
inter text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(R)}\
</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(R)}</div>`},"renderDeta\
ils"),x=o(async(E,R)=>{const Z=await apiFetch(E,{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify(R)}),O=await Z.json().catch(()=>({}));if(!Z.ok)throw new Error(O.error||String(
Z.status));return O},"postJson"),S=o((E,R)=>`<span class="${R} text-white px-2 py-0.5 rounded">${escapeHtml(
E)}</span>`,"badge"),L=o(()=>{const E=get("bot-log-account-list");if(!E)return;const R=c.accounts,Z=R.
length&&R.every(H=>c.selected.has(H.user_id)),O=get("bot-log-list-toolbar");if(O&&(O.innerHTML=`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${Z?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!R.length){E.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}E.innerHTML=R.map(H=>{const G=[];H.exists||G.push(S("\u524A\u9664\u6E08\u307F","bg-gray-600")),
H.is_bot_banned&&G.push(S("BAN\u4E2D","bg-red-600")),H.lock_remaining_seconds>0&&G.push(S(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
ceil(H.lock_remaining_seconds/60)}\u5206\uFF09`,"bg-yellow-600"));const P=H.evidence_count?`\u8A18\u9332 ${H.
evidence_count}\u4EF6\u30FB\u6700\u7D42 ${escapeHtml(h(H.last_event_at))}`:"\u8A18\u9332\u306A\u3057";
return`
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 round\
ed p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${H.
user_id}" ${c.selected.has(H.user_id)?"checked":""} aria-label="${escapeHtml(H.username||"ID "+H.user_id)}\
 \u3092\u9078\u629E">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${H.exists?
"text-gray-200":"text-gray-400"} font-bold bot-log-wrap">${escapeHtml(H.username||"ID "+H.user_id)}<\
/span>${G.join("")}</div>
                                <div class="text-[10px] text-gray-500">${P}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${H.user_id}" data-username="${escapeHtml(
H.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),C=o(async()=>{const E=get("bot-log-acc\
ount-list");if(!E)return;E.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const R=get("bot-log-search"),Z=R?R.value.
trim():"";try{const O=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(Z)}`,{cache:"\
no-store"}),H=await O.json().catch(()=>({}));if(!O.ok)throw new Error(H.error||String(O.status));c.accounts=
H.accounts||[];const G=new Set(c.accounts.map(P=>P.user_id));Array.from(c.selected).forEach(P=>{G.has(
P)||c.selected.delete(P)}),L()}catch{E.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),A=o(E=>{if(!E)return"";const R=E.locks||[],Z=E.banned_identifiers||[],
O=[];E.exists||O.push(S("\u524A\u9664\u6E08\u307F","bg-gray-600")),E.exists&&(O.push(S(E.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",E.detection_enabled?"bg-gray-600":"bg-gray-700")),O.push(S(E.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",E.is_bot_banned?"bg-red-600":"bg-gray-600"))),O.push(S(R.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",R.length?"bg-yellow-600":"bg-gray-600")),Z.length&&O.push(S(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${Z.length}\u4EF6`,"bg-red-600")),E.is_admin&&O.push(S("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const H=!!(l.counts.ban||l.counts.related_ban),G=E.is_bot_banned?`<div class="text-g\
ray-300">BAN\u306E\u7406\u7531: ${escapeHtml(E.bot_ban_reason||"-")}\uFF08${escapeHtml(h(E.bot_banned_at))}\
\uFF09</div>${H?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",P=R.map(de=>{const ne=a[de.source]||de.source,re=de.identifier?` ${escapeHtml(de.identifier)}${de.
source==="cookie"?"\u2026":""}`:"",Se=de.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
de.origin.username||String(de.origin.user_id||"-"))}\uFF08${escapeHtml(n[de.origin.client]||de.origin.
client||"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(ne)}${re}\
\u30FB\u6B8B\u308A${escapeHtml(y(de.remaining_seconds))}\u30FB${escapeHtml(de.reason||"")}${Se}</div\
>`}).join(""),K=Z.map(de=>`<div class="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[de.
kind]||de.kind)} ${escapeHtml(de.identifier)}${de.kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(de.
reason||"-")}\uFF08${escapeHtml(h(de.created_at))}\uFF09</div>`).join(""),be=Z.length?'<div class="t\
ext-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002</div>':
"",pe=E.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${y(E.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",te=E.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String(E.appeal_count))}\u4EF6`:"",X=[];return R.length&&X.push('<button class="bot-log-unlock bg-ye\
llow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
Z.length&&X.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 ro\
unded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${O.join("")}</div>
                        ${G}
                        ${P}
                        ${K}
                        ${be}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String(E.lock_count))} / ${escapeHtml(String(E.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(pe)}\
\u30FB\u5931\u6557 ${escapeHtml(String(E.turnstile_fail_count))} / ${escapeHtml(String(E.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round((E.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round((E.behavior_score||0)*10)/10))}\uFF09${te}</div>
                        ${X.length?`<div class="flex flex-wrap gap-2 pt-1">${X.join("")}</div>`:""}
                    </div>`},"renderState"),I=o(()=>{const E=o((Z,O,H)=>`<button class="bot-log-filt\
er ${l.type===Z?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(Z)}">${escapeHtml(O)} (${H})</button>`,"chip"),R=[E("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((Z,O)=>l.counts[O]-l.counts[Z]).forEach(Z=>{R.push(E(Z,(t[Z]||
{label:Z}).label,l.counts[Z]))}),R.join("")},"renderFilters"),J=o(E=>{const R=t[E.event_type]||{label:E.
event_type,cls:"bg-gray-600"},Z=E.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[E.client]||E.client)}</span>`:"",O=l.selected.has(E.id)?"checked":"",H=E.score!==
null&&E.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String(E.score))}${E.
behavior_score!==null&&E.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String(E.behavior_score))}\
\uFF09`:""}</span>`:"",G=E.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
E.reasons)}</div>`:"",P=E.ip_address||E.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${E.ip_address?"IP "+escapeHtml(E.ip_address):""}${E.ip_address&&E.user_agent?"\u30FB":""}${escapeHtml(
E.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${E.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${E.id}" ${O}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(h(E.created_at))}</span>
                            <span class="${R.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(R.label)}\
</span>
                            ${Z}
                            ${H}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${E.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${G}
                        ${v(E.details)}
                        ${P}
                    </div>`},"renderItem"),D=o(()=>{const E=u();if(!E)return;const R=l.items.length?
l.items.map(J).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
Z=l.items.length&&l.items.every(O=>l.selected.has(O.id));E.innerHTML=`
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py\
-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>\u4E00\u89A7\u306B\u623B\u308B</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(l.username||
"ID "+l.userId)}</div>
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-whi\
te px-2 py-1 rounded text-xs">\u66F4\u65B0</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${A(l.data&&l.data.state)}
                        <div class="flex flex-wrap gap-1 text-xs">${I()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkb\
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${Z?
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
                    </div>`},"render"),ie=o(async(E=!1)=>{const R=u();if(!R||!l.userId)return;E||(R.
innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D.\
..</div>');const Z=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&Z.set("t\
ype",l.type),E&&l.items.length&&Z.set("before_id",String(l.items[l.items.length-1].id));try{const O=await apiFetch(
`/api/bot/evidence?${Z.toString()}`,{cache:"no-store"}),H=await O.json().catch(()=>({}));if(!O.ok)throw new Error(
H.error||String(O.status));l.counts=H.counts||{},l.total=H.total||0,l.hasMore=!!H.has_more,H.user&&(l.
username=H.user.username||l.username,l.exists=!!H.user.exists),E?l.items=l.items.concat(H.items||[]):
(l.items=H.items||[],l.data=H),D()}catch{E||(R.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),j=o(async(E,R)=>{if(confirm(R))try{const Z=await x("/api/bot/evidence/delete",
Object.assign({user_id:l.userId},E));showToast(`${Z.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await ie(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),Q=o(async(E,R,Z)=>{if(confirm(R))try{await x(E,{user_id:l.userId}),showToast(
Z,"success"),await ie(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","\
error",!0)}},"runAccountAction"),ce=o(()=>{const E=u();E&&(E.classList.add("hidden"),E.classList.remove(
"flex"),E.innerHTML="");const R=f();R&&(R.classList.remove("hidden"),R.classList.add("flex")),l.userId=
0},"showList"),me=o(async()=>{ce(),await C()},"openList"),ae=o(async(E,R="")=>{const Z=u();if(!Z)return;
Object.assign(l,{userId:Number(E)||0,username:R,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),l.selected.clear();const O=f();O&&(O.classList.add("hidden"),O.classList.remove("flex")),
Z.classList.remove("hidden"),Z.classList.add("flex"),await ie(!1)},"open");return{open:ae,openList:me,
showList:ce,bind:o(()=>{const E=get("bot-log-modal");if(!E||E.dataset.bound==="1")return;E.dataset.bound=
"1";const R=get("bot-log-search");R&&R.addEventListener("keydown",Z=>{Z.key==="Enter"&&C()}),E.addEventListener(
"change",Z=>{const O=Z.target;if(!(!O||!O.classList))if(O.classList.contains("bot-log-select")){const H=Number(
O.getAttribute("data-log-id"));O.checked?l.selected.add(H):l.selected.delete(H),D()}else if(O.classList.
contains("bot-log-select-all"))l.items.forEach(H=>{O.checked?l.selected.add(H.id):l.selected.delete(
H.id)}),D();else if(O.classList.contains("bot-log-account-select")){const H=Number(O.getAttribute("d\
ata-user-id"));O.checked?c.selected.add(H):c.selected.delete(H),L()}else O.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(H=>{O.checked?c.selected.add(H.user_id):c.selected.delete(
H.user_id)}),L())}),E.addEventListener("click",async Z=>{const O=Z.target.closest("button");if(!(!O||
O.disabled))if(O.id==="bot-log-search-btn"||O.id==="bot-log-refresh-btn")O.id==="bot-log-refresh-btn"&&
R&&(R.value=""),await C();else if(O.classList.contains("bot-log-open-account")){const H=Number(O.getAttribute(
"data-user-id"));H&&await ae(H,O.getAttribute("data-username")||"")}else if(O.classList.contains("bo\
t-log-delete-accounts")){const H=Array.from(c.selected);if(!H.length||!confirm(`\u9078\u629E\u3057\u305F${H.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const G=await x("/api/bot/evidence/delete",{user_ids:H});showToast(`${G.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\
\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await C()}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F","error",!0)}}else if(O.classList.contains("bot-log-back"))await me();else if(O.classList.
contains("bot-log-reload"))await ie(!1);else if(O.classList.contains("bot-log-more"))await ie(!0);else if(O.
classList.contains("bot-log-filter"))l.type=O.getAttribute("data-type")||"",l.selected.clear(),await ie(
!1);else if(O.classList.contains("bot-log-delete-one")){const H=Number(O.getAttribute("data-log-id"));
H&&await j({ids:[H]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
O.classList.contains("bot-log-delete-selected")?l.selected.size&&await j({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-delete-all")?await j({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
O.classList.contains("bot-log-unlock")?await Q("/api/bot/account/clear-lock",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):O.classList.contains("bot-log-unbloc\
k")&&await Q("/api/bot/account/clear-identifiers",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
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
h&&(c=n(h[1]))}if(!r.has(c))return;const u=document.createElement("span");u.className="inline-block \
rounded bg-gray-800 p-3 text-xs text-gray-400",u.textContent="\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",
u.title=c.split("/").pop();const f=l.closest(".chat-image-frame")||(l.tagName==="SOURCE"?l.parentNode:
l);f&&a.contains(f)&&f.replaceWith(u)})}}o(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}o(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},r=1/0;for(let l=1;l<=i;l++){if(i%l)continue;const c=i/l,u=Math.abs(
Math.log(Math.max(1,t)/l/(Math.max(1,n)/c)));(u<r-1e-9||Math.abs(u-r)<=1e-9&&(t>=n?l>a.columns:c>a.rows))&&
(a={columns:l,rows:c},r=u)}return a}o(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,r=Math.round(e/n.columns*a/2),l=Math.round(t/n.rows*a/2),c=[];for(let u=0;u<
n.rows;u++)for(let f=0;f<n.columns;f++){const h=Math.floor(f*e/n.columns),y=Math.floor((f+1)*e/n.columns),
w=Math.floor(u*t/n.rows),v=Math.floor((u+1)*t/n.rows);c.push({index:c.length,row:u,column:f,cellLeft:h,
cellTop:w,cellRight:y,cellBottom:v,left:Math.max(0,h-r),top:Math.max(0,w-l),right:Math.min(e,y+r),bottom:Math.
min(t,v+l)})}return c}o(splitTiles,"splitTiles");function splitFits(e,t,n){return Math.floor(t/e.columns)>=
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
getContext("2d"),c=e.width/n,u=e.height/i;l.drawImage(t,0,0,e.width,e.height);const f=r[0],h=f.right-
f.cellRight,y=f.bottom-f.cellBottom;l.fillStyle="rgba(250,204,21,0.18)";const w=[];for(let x=1;x<a.columns;x++){
const S=Math.floor(x*n/a.columns);l.fillRect((S-h)*c,0,2*h*c,e.height),w.push([S*c,0,S*c,e.height])}
for(let x=1;x<a.rows;x++){const S=Math.floor(x*i/a.rows);l.fillRect(0,(S-y)*u,e.width,2*y*u),w.push(
[0,S*u,e.width,S*u])}drawSplitLines(l,w,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const v=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));l.font=`bold ${v}px monospace`,
l.textAlign="center",l.textBaseline="middle";for(const x of r){const S=(x.cellLeft+x.cellRight)/2*c,
L=(x.cellTop+x.cellBottom)/2*u;l.lineWidth=Math.max(2,v/10),l.strokeStyle="rgba(0,0,0,0.7)",l.strokeText(
String(x.index+1),S,L),l.fillStyle="#facc15",l.fillText(String(x.index+1),S,L)}}o(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,r,l,c])=>{e.moveTo(a,r),e.lineTo(l,c)}),e.stroke(),e.restore()}o(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(r=>r?i(
r):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}o(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const r=a.
naturalWidth,l=a.naturalHeight,c=chooseSplitGrid(t.pieces,r,l);if(!splitFits(c,r,l))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const u=splitTiles(r,l,c,t.overlap),f=splitBaseName(e.name),h=[],y=o(async w=>{h.push(await applyImageCompression(
w,n))},"append");if(t.includeOriginal)if(t.markOriginal){const w=splitCanvas(r,l);drawSplitOverview(
w,a,r,l,c,u),await y(await splitCanvasFile(w,`${f}_split_grid.png`))}else await y(e);for(const w of u){
const v=w.right-w.left,x=w.bottom-w.top;let S=Math.max(12,Math.min(40,Math.max(v,x)/45)),L=0;const C=splitTileCaption(
w,c,r,l);if(t.markPieces){const J=splitCanvas(1,1),D=J.getContext("2d");D.font=`bold ${S}px monospac\
e`,S=Math.max(6,S*Math.min(1,v*.96/D.measureText(C).width)),L=Math.max(12,Math.floor(S*1.8)),J.width=
0}const A=splitCanvas(v,x+L),I=A.getContext("2d");if(L&&(I.fillStyle="#111827",I.fillRect(0,0,v,L)),
I.drawImage(a,w.left,w.top,v,x,0,L,v,x),t.markPieces){I.fillStyle="#fff",I.font=`bold ${S}px monospa\
ce`,I.textBaseline="middle",I.fillText(C,v*.02,L/2);const J=[];for(const D of[w.cellLeft,w.cellRight])
D>w.left&&D<w.right&&J.push([D-w.left,L,D-w.left,L+x]);for(const D of[w.cellTop,w.cellBottom])D>w.top&&
D<w.bottom&&J.push([0,L+D-w.top,v,L+D-w.top]);drawSplitLines(I,J,Math.max(1.5,Math.min(4,Math.max(v,
x)/500)),!0)}await y(await splitCanvasFile(A,splitTileName(f,w,c)))}return h}finally{i.release()}}o(
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
ote-bar")].filter(I=>I&&I.getClientRects().length&&getComputedStyle(I).display!=="none").map(I=>I.getBoundingClientRect()),
l=Math.min(...r.map(I=>I.left)),c=Math.min(...r.map(I=>I.top)),u={left:l,top:c,width:Math.max(...r.map(
I=>I.right))-l,height:Math.max(...r.map(I=>I.bottom))-c},f=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,h={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},y=document.createElement("div");
y.hidden=!0,y.className="prompt-send-held-files",y.setAttribute("aria-hidden","true"),y.append(...h.
rows),document.body.appendChild(y);let w=!1,v=!1,x=!1;const S=o(()=>{w=!0},"onInput");t.addEventListener(
"input",S),t.value="",["file-input","photo-input","mask-input"].forEach(I=>{const J=get(I);J&&(J.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let L;i&&e&&!f&&
i.animate&&(L=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const C=o(()=>{!v||x||(h.files.forEach(I=>{var D,ie;const J=(ie=
(D=I==null?void 0:I.rowObj)==null?void 0:D.row)==null?void 0:ie.getAttribute("data-local-url");J&&URL.
revokeObjectURL(J)}),y.remove())},"releaseFiles"),A=o((I,J=!0)=>{v||(v=!0,I=I&&h.navigation===threadLoadSequence,
x=I,t.removeEventListener("input",S),I?(!w&&!t.value&&(t.value=h.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...h.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=h.mask),h.files.forEach((D,ie)=>browserFastLocalFiles.set(ie,D)),
h.markers.forEach(D=>markerAppliedUploads.add(D)),n&&n.prepend(...h.rows),y.remove(),!currentQuote&&
h.quote&&(currentQuote=h.quote,currentQuoteMessageId=h.quoteId,get("quote-text-display").innerText=h.
quote,get("quote-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=h.progress),
updateFilePreview(),updateMaskPreview(),schedulePromptTokenEstimate(!0)):J&&C())},"finish");return{source:u,
canRetry:o(()=>!w&&h.navigation===threadLoadSequence,"canRetry"),stopDock:o(()=>L==null?void 0:L.cancel(),
"stopDock"),files:h.files,releaseFiles:C,accept:o((I=!0)=>A(!1,I),"accept"),restore:o(()=>A(!0),"res\
tore")}}o(takePromptForSend,"takePromptForSend");function animatePromptToMessage(e,t,n){var y;let i,
a=0,r=!1;const l=[],c=get("chat-container"),u=get("prompt-input"),f=window.matchMedia("(prefers-redu\
ced-motion: reduce)"),h=o(()=>{var w,v;r||(r=!0,(w=window.ActivityLog)==null||w.log("ui.hide"),t.stopDock(),
cancelAnimationFrame(a),l.forEach(x=>x.cancel()),i==null||i.remove(),e==null||e.classList.remove("pr\
ompt-send-hidden"),window.removeEventListener("resize",h),window.removeEventListener("popstate",h),window.
removeEventListener("pagehide",h),document.removeEventListener("visibilitychange",h),c==null||c.removeEventListener(
"wheel",h),c==null||c.removeEventListener("touchstart",h),u==null||u.removeEventListener("input",h),
(v=window.visualViewport)==null||v.removeEventListener("resize",h),f.removeEventListener("change",h))},
"stop");if(!n||f.matches||!e||!Element.prototype.animate)return h;try{e.classList.remove("fade-in"),
e.classList.add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var w;try{if(!e.isConnected)return h();
const v=e.querySelector(".message-bubble"),x=v.getBoundingClientRect(),S=c.getBoundingClientRect();if(!x.
width||!x.height||x.bottom<=S.top||x.top>=S.bottom)return h();const L=t.source;i=v.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach(ie=>ie.removeAttribute("id")),i.querySelectorAll(".msg-cont\
rols, video, audio").forEach(ie=>ie.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute(
"aria-hidden","true"),i.inert=!0,Object.assign(i.style,{width:x.width+"px",height:x.height+"px",left:x.
left+"px",top:x.top+"px"}),document.body.appendChild(i),(w=window.ActivityLog)==null||w.log("ui.show");
const C=Math.min(L.width/x.width,Math.max(L.height,42)/x.height,1),A=L.left+(L.width-x.width*C)/2-x.
left,I=L.top+(L.height-x.height*C)/2-x.top,J=i.animate([{transform:`translate(${A}px, ${I}px) scale(${C}\
)`,borderRadius:"12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"\
translate(0, 0) scale(1)",borderRadius:getComputedStyle(v).borderRadius,backgroundColor:getComputedStyle(
v).backgroundColor}],{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});l.push(J),l.
push(i.animate([{opacity:1,offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),
e.classList.remove("prompt-send-hidden"),l.push(v.animate([{opacity:0,offset:0},{opacity:0,offset:.8},
{opacity:1}],{duration:350,fill:"both"}));const D=o(()=>{if(!e.isConnected||!userAutoScroll||document.
hidden)return h();const ie=v.getBoundingClientRect();i.style.left=ie.left+"px",i.style.top=ie.top+"p\
x",a=requestAnimationFrame(D)},"watch");a=requestAnimationFrame(D),J.finished.then(h,h)}catch{console.
warn("Prompt send animation failed"),h()}}),window.addEventListener("resize",h),window.addEventListener(
"popstate",h),window.addEventListener("pagehide",h),document.addEventListener("visibilitychange",h),
c.addEventListener("wheel",h,{passive:!0}),c.addEventListener("touchstart",h,{passive:!0}),u.addEventListener(
"input",h),(y=window.visualViewport)==null||y.addEventListener("resize",h),f.addEventListener("chang\
e",h)}catch{console.warn("Prompt send animation failed"),h()}return h}o(animatePromptToMessage,"anim\
atePromptToMessage");
