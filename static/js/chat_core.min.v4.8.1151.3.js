function updateFilePreview(){const e=get("file-preview"),t=get("file-name"),n=get("upload-total-prog\
ress"),i=get("upload-total-progress-bar"),a=get("file-preview-thumbs"),o=get("upload-modal-status-te\
xt"),l=get("upload-modal-total-progress"),c=get("upload-modal-total-progress-bar");if(!e||!t)return;
if(a){const C=document.querySelectorAll("#upload-list .upload-row");a.innerHTML="",C.forEach((M,E)=>{
const H=M.getAttribute("data-local-url"),B=M.getAttribute("data-filename"),te=M.querySelector("img.u\
pload-preview")!==null;let j;if(te){let V=H;if(!V&&B){const ue=B.replace(/^\d+\//,"");V=buildAttachmentPreviewUrl(
ue)}V&&(j=document.createElement("img"),j.src=V,j.className="thumb-item shadow-sm",j.dataset.viewerSrc=
V,j.dataset.viewerFilename=B||V.split("/").pop(),j.uploadRow=M,j.onclick=function(ue){ue.preventDefault(),
openImageViewer(this.dataset.viewerSrc,".thumb-item")},j.onerror=function(){this.parentElement.replaceChild(
u("ERR"),this)})}j||(j=u("FILE")),j.style.animationDelay=`${E*32}ms`,a.appendChild(j)}),C.length>0?a.
classList.remove("hidden"):a.classList.add("hidden")}function u(C){const M=document.createElement("d\
iv");return M.className="thumb-item bg-gray-800 flex items-center justify-center text-gray-500 text-\
[9px] shadow-sm font-bold",M.innerText=C,M}r(u,"createFileThumb");const f=collectImageUrlsForSend(),
h=uploadProgressState.total,y=uploadProgressState.completed,v=uploadProgressState.active;h===0&&(e.classList.
add("hidden"),n&&n.classList.add("hidden"),l&&l.classList.add("hidden"),a&&a.classList.add("hidden"));
const w=get("send-btn"),x=get("mic-btn"),_=get("mask-btn"),S=isStopMode;if(v>0?(w&&(w.disabled=!0),x&&
(x.disabled=!0),_&&(_.disabled=!0)):S||(w&&(w.disabled=!1),x&&(x.disabled=!1),_&&(_.disabled=!1)),v>
0){const C=`Preparing... (${y}/${h})`;e.classList.remove("hidden"),t.innerText=C,o&&(o.innerText=`(${y}\
/${h})`);let M=y*100,E=0;for(let te in uploadProgressState.perFilePct)M+=uploadProgressState.perFilePct[te],
E++;const H=h>0?M/(h*100)*100:0,B=`${Math.min(100,H)}%`;n&&i&&(n.classList.remove("hidden"),i.style.
width=B),l&&c&&(l.classList.remove("hidden"),c.style.width=B)}else o&&(o.innerText=""),l&&l.classList.
add("hidden"),f.length>0?(e.classList.remove("hidden"),t.innerText=`${f.length} files ready`,n&&n.classList.
add("hidden")):(e.classList.add("hidden"),t.innerText="",n&&n.classList.add("hidden"));schedulePromptTokenEstimate()}
r(updateFilePreview,"updateFilePreview");function updateMaskPreview(){const e=get("mask-preview"),t=get(
"mask-name");!e||!t||(currentMaskImage?(e.classList.remove("hidden"),t.innerText=`Mask: ${currentMaskImage.
split("/").pop()}`):(e.classList.add("hidden"),t.innerText=""))}r(updateMaskPreview,"updateMaskPrevi\
ew");const markerToolHints={draw:"\u30DE\u30FC\u30AB\u30FC\uFF08\u8272\u30FB\u900F\u660E\u5EA6\u5909\u66F4\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
mosaic:"\u30C9\u30E9\u30C3\u30B0\u3067\u7BC4\u56F2\u30E2\u30B6\u30A4\u30AF\uFF08\u8907\u6570\u8FFD\u52A0\u53EF\uFF09 / \u4E8C\u672C\u6307\u3067\u62E1\u5927",
crop:"\u5916\u5074\u3092\u30C9\u30E9\u30C3\u30B0\u3057\u3066\u5207\u308A\u53D6\u308A / \u4E8C\u672C\u6307\u3067\u62E1\u5927"};
function normalizeMarkerHexColor(e){const t=String(e||"").trim().toLowerCase();if(/^#[0-9a-f]{6}$/.test(
t))return t;if(/^#[0-9a-f]{3}$/.test(t)){const n=t[1],i=t[2],a=t[3];return`#${n}${n}${i}${i}${a}${a}`}
return"#facc15"}r(normalizeMarkerHexColor,"normalizeMarkerHexColor");function markerHexToRgb(e){const t=normalizeMarkerHexColor(
e);return{r:parseInt(t.slice(1,3),16),g:parseInt(t.slice(3,5),16),b:parseInt(t.slice(5,7),16)}}r(markerHexToRgb,
"markerHexToRgb");function clampMarkerOpacityPct(e,t=60){const n=Number(e),i=Number.isFinite(n)?n:t;
return Math.max(MARKER_OPACITY_MIN_PCT,Math.min(MARKER_OPACITY_MAX_PCT,i))}r(clampMarkerOpacityPct,"\
clampMarkerOpacityPct");function formatMarkerOpacityPct(e){const t=Math.round(clampMarkerOpacityPct(
e)*10)/10;return Number.isInteger(t)?String(t):String(t).replace(/\.0$/,"")}r(formatMarkerOpacityPct,
"formatMarkerOpacityPct");function getMarkerStrokeStyle(){const e=markerHexToRgb(markerState.colorHex),
t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.opacity)||.6));return`rgba(${e.r},${e.
g},${e.b},${t})`}r(getMarkerStrokeStyle,"getMarkerStrokeStyle");function syncMarkerColorControls(){const e=normalizeMarkerHexColor(
markerState.colorHex);markerState.colorHex=e;const t=Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(
markerState.opacity)||.6));markerState.opacity=t;const n=t*100,i=formatMarkerOpacityPct(n),a=get("ma\
rker-color-picker");a&&a.value!==e&&(a.value=e);const o=get("marker-opacity");o&&o.value!==i&&(o.value=
i);const l=get("marker-opacity-number");l&&l.value!==i&&(l.value=i);const c=get("marker-opacity-valu\
e");c&&(c.textContent=`${i}%`),document.querySelectorAll("#marker-toolbar .marker-color-chip[data-ma\
rker-color]").forEach(f=>{const h=normalizeMarkerHexColor(f.getAttribute("data-marker-color"));f.classList.
toggle("active",h===e)})}r(syncMarkerColorControls,"syncMarkerColorControls");function setMarkerColor(e){
markerState.colorHex=normalizeMarkerHexColor(e),syncMarkerColorControls()}r(setMarkerColor,"setMarke\
rColor");function setMarkerOpacity(e){const t=clampMarkerOpacityPct(e,60);markerState.opacity=t/100,
syncMarkerColorControls()}r(setMarkerOpacity,"setMarkerOpacity");function setMarkerMode(e){markerState.
mode=e,e!=="mosaic"&&(markerState.mosaicPreviewRect=null);const t=get("marker-tool-draw"),n=get("mar\
ker-tool-mosaic"),i=get("marker-tool-crop");t&&t.classList.toggle("active",e==="draw"),n&&n.classList.
toggle("active",e==="mosaic"),i&&i.classList.toggle("active",e==="crop");const a=get("marker-tool-hi\
nt");a&&(a.textContent=markerState.maskMode?e==="auto"?"\u753B\u50CF\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u8FD1\u4F3C\u8272\u306E\u3064\u306A\u304C\u3063\u305F\u7BC4\u56F2\u3092\u5857\u308A\u307E\u3059":
"\u767D\u304F\u5857\u3063\u305F\u90E8\u5206\u304C\u524A\u9664\u5BFE\u8C61\u306B\u306A\u308A\u307E\u3059 / \u4E8C\u672C\u6307\u3067\u62E1\u5927":
markerToolHints[e]||"");const o=get("marker-crop-reset");o&&o.classList.toggle("hidden",e!=="crop"),
document.querySelectorAll(".marker-edit-only").forEach(C=>C.classList.toggle("hidden",markerState.maskMode));
const l=get("marker-tool-mosaic"),c=get("marker-tool-crop");l&&l.classList.toggle("hidden",markerState.
maskMode),c&&c.classList.toggle("hidden",markerState.maskMode);const u=get("marker-tool-draw"),f=get(
"marker-tool-auto");u&&(u.textContent=markerState.maskMode?"\u30D6\u30E9\u30B7":"\u30DE\u30FC\u30AB\u30FC"),
f&&(f.classList.toggle("hidden",!markerState.maskMode),f.classList.toggle("active",e==="auto")),u&&u.
classList.toggle("active",e==="draw");const h=get("marker-brush-control");h&&h.classList.toggle("hid\
den",markerState.maskMode&&e==="auto");const y=get("marker-auto-tolerance-control");y&&y.classList.toggle(
"hidden",!markerState.maskMode||e!=="auto");const v=get("marker-modal-title"),w=get("marker-modal-de\
scription"),x=get("marker-save");v&&(v.innerHTML=markerState.maskMode?'<i class="fas fa-mask text-ye\
llow-300"></i> \u30DE\u30B9\u30AF\u4F5C\u6210':'<i class="fas fa-highlighter text-yellow-300"></i> \u753B\
\u50CF\u7DE8\u96C6'),w&&(w.textContent=markerState.maskMode?"\u767D\u304F\u5857\u3063\u305F\u9818\u57DF\u304C\u524A\u9664\u5BFE\u8C61\u3067\u3059\u3002\u30D6\u30E9\u30B7\u307E\u305F\u306F\u81EA\u52D5\u5857\u308A\u3092\u4F7F\u3048\u307E\u3059\u3002\u30DE\u30B9\u30AF\u306F\u5BFE\u8C61\u753B\u50CF\
\u3068\u540C\u3058\u30B5\u30A4\u30BA\u3067\u4F5C\u6210\u3055\u308C\u307E\u3059\u3002":"\u30DE\u30FC\u30AB\u30FC\u30FB\u30E2\u30B6\u30A4\u30AF\u30FB\u30C8\u30EA\u30DF\
\u30F3\u30B0\u304C\u53EF\u80FD\u3067\u3059\u3002\u9001\u4FE1\u6642\u306B\u7DE8\u96C6\u6E08\u307F\u753B\u50CF\u306E\u78BA\u8A8D\u3092\u4FC3\u3057\u307E\u3059\u3002"),
x&&(x.textContent=markerState.maskMode?"\u30DE\u30B9\u30AF\u3092\u4F5C\u6210":"\u4FDD\u5B58\u3057\u3066\u53CD\u6620");
const _=get("marker-canvas");_&&(_.style.pointerEvents=e==="crop"?"none":"auto");const S=get("marker\
-crop-canvas");S&&(S.style.pointerEvents=e==="crop"?"auto":"none"),e==="crop"&&(!markerState.cropRect||
markerState.cropRect.w<=1||markerState.cropRect.h<=1)&&resetCropRectToFull(),renderCropOverlay()}r(setMarkerMode,
"setMarkerMode");function clearCropRect(){resetCropRectToFull(),renderCropOverlay()}r(clearCropRect,
"clearCropRect");function resetCropRectToFull(){const e=get("marker-crop-canvas");if(!e)return;const t=Math.
max(1,e.width||0),n=Math.max(1,e.height||0);t<=1||n<=1||(markerState.cropRect={x:0,y:0,w:t,h:n})}r(resetCropRectToFull,
"resetCropRectToFull");function clampMarkerViewOffset(){if(markerView.scale=Math.min(markerView.maxScale,
Math.max(markerView.minScale,Number(markerView.scale)||1)),markerView.scale<=markerView.minScale+1e-4){
markerView.offsetX=0,markerView.offsetY=0;return}const e=get("marker-stage"),t=get("marker-viewport");
if(!e||!t)return;const n=Math.max(1,e.clientWidth||0),i=Math.max(1,e.clientHeight||0),a=Math.max(1,t.
offsetWidth||t.clientWidth||0),o=Math.max(1,t.offsetHeight||t.clientHeight||0);if(n<=1||i<=1||a<=1||
o<=1)return;const l=(n-a)/2,c=(i-o)/2,u=a*markerView.scale,f=o*markerView.scale,h=Math.min(n*.45,Math.
max(24,n*.12)),y=Math.min(i*.45,Math.max(24,i*.12)),v=h-l-u,w=n-h-l,x=y-c-f,_=i-y-c,S=r((C,M,E)=>Number.
isFinite(C)?M>E?(M+E)/2:Math.min(E,Math.max(M,C)):0,"clampOffset");markerView.offsetX=S(markerView.offsetX,
v,w),markerView.offsetY=S(markerView.offsetY,x,_)}r(clampMarkerViewOffset,"clampMarkerViewOffset");function applyMarkerTransform(){
const e=get("marker-viewport");e&&(clampMarkerViewOffset(),e.style.transform=`translate(${markerView.
offsetX}px, ${markerView.offsetY}px) scale(${markerView.scale})`)}r(applyMarkerTransform,"applyMarke\
rTransform");function resetMarkerTransform(){markerView.scale=1,markerView.offsetX=0,markerView.offsetY=
0,applyMarkerTransform()}r(resetMarkerTransform,"resetMarkerTransform");function getRowMarkerKey(e){
return e&&(e.dataset.uploadId||e.getAttribute("data-filename"))||null}r(getRowMarkerKey,"getRowMarke\
rKey");function setRowMarkerState(e,t){const n=getRowMarkerKey(e);n&&(t?markerAppliedUploads.add(n):
markerAppliedUploads.delete(n));const i=e?e.querySelector(".upload-marker-tag"):null;i&&i.classList.
toggle("hidden",!t)}r(setRowMarkerState,"setRowMarkerState");function hasMarkerHint(){return markerAppliedUploads.
size>0}r(hasMarkerHint,"hasMarkerHint");function normalizeAttachmentSource(e){const t=String(e||"").
trim().toLowerCase();return t==="library"||t==="lib"?"library":t==="upload"||t==="uploaded"?"upload":
"unknown"}r(normalizeAttachmentSource,"normalizeAttachmentSource");function normalizeAttachmentDisplayName(e){
if(e==null)return"";let t=String(e).replace(/\u0000/g,"");return t=t.replace(/\r/g," ").replace(/\n/g,
" ").replace(/\t/g," "),t=t.trim(),!t||(t=t.split("/").pop().split("\\").pop().trim(),t=t.replace(/\s{2,}/g,
" "),t=t.replace(/[<>:"/\\|?*]+/g,"_"),!t||t==="."||t==="..")?"":(t.length>180&&(t=t.slice(0,180).trim()),
t)}r(normalizeAttachmentDisplayName,"normalizeAttachmentDisplayName");function defaultAttachmentDisplayName(e){
const t=normalizeAttachmentPath(e);return t?t.split("/").pop()||t:""}r(defaultAttachmentDisplayName,
"defaultAttachmentDisplayName");function setAttachmentNameForPath(e,t){const n=normalizeAttachmentPath(
e);if(!n)return;const i=normalizeAttachmentDisplayName(t)||defaultAttachmentDisplayName(n);i&&attachmentNameByPath.
set(n,i)}r(setAttachmentNameForPath,"setAttachmentNameForPath");function getAttachmentNameForPath(e){
const t=normalizeAttachmentPath(e);if(!t)return"";const n=normalizeAttachmentDisplayName(attachmentNameByPath.
get(t));return n||defaultAttachmentDisplayName(t)}r(getAttachmentNameForPath,"getAttachmentNameForPa\
th");function setRowAttachmentName(e,t){if(!e)return;const n=normalizeAttachmentDisplayName(t)||getAttachmentNameForPath(
e.getAttribute("data-filename"))||"file";e.dataset.displayName=n;const i=e.querySelector(".truncate");
i&&(i.textContent=n);const a=e.getAttribute("data-filename");a&&setAttachmentNameForPath(a,n)}r(setRowAttachmentName,
"setRowAttachmentName");function isRowAttachmentNameCustomized(e){return!!(e&&e.dataset.sendNameCustomized===
"1")}r(isRowAttachmentNameCustomized,"isRowAttachmentNameCustomized");function setRowAttachmentNameCustomized(e,t){
e&&(e.dataset.sendNameCustomized=t?"1":"")}r(setRowAttachmentNameCustomized,"setRowAttachmentNameCus\
tomized");function getRowDefaultAttachmentName(e){if(!e)return"file";const t=e.getAttribute("data-fi\
lename");if(t)return defaultAttachmentDisplayName(t)||"file";const n=normalizeAttachmentDisplayName(
e.dataset.defaultDisplayName);return n||normalizeAttachmentDisplayName(e.dataset.displayName)||"file"}
r(getRowDefaultAttachmentName,"getRowDefaultAttachmentName");function promptRowAttachmentName(e){if(!e)
return;const t=getRowAttachmentName(e)||getRowDefaultAttachmentName(e)||"file",n=prompt("\u9001\u4FE1\u6642\u306E\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\
\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u7A7A\u6B04\u3067\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3059\uFF09",
t);if(n===null)return;const i=normalizeAttachmentDisplayName(n);if(!i){const a=getRowDefaultAttachmentName(
e);setRowAttachmentName(e,a),setRowAttachmentNameCustomized(e,!1),showToast("\u9001\u4FE1\u540D\u3092\u30C7\u30D5\u30A9\u30EB\u30C8\u306B\u623B\u3057\u307E\u3057\u305F",
"success");return}setRowAttachmentName(e,i),setRowAttachmentNameCustomized(e,!0),showToast("\u9001\u4FE1\u540D\u3092\u66F4\u65B0\u3057\u307E\
\u3057\u305F","success")}r(promptRowAttachmentName,"promptRowAttachmentName");function getRowAttachmentName(e){
if(!e)return"";const t=e.getAttribute("data-filename"),n=getAttachmentNameForPath(t);if(n)return n;const i=normalizeAttachmentDisplayName(
e.dataset.displayName);if(i)return i;const a=e.querySelector(".truncate"),o=normalizeAttachmentDisplayName(
a?a.textContent:"");return o||getAttachmentNameForPath(t)}r(getRowAttachmentName,"getRowAttachmentNa\
me");function setAttachmentSourceForPath(e,t){const n=normalizeAttachmentPath(e);if(!n)return;const i=normalizeAttachmentSource(
t);i!=="unknown"&&attachmentSourceByPath.set(n,i)}r(setAttachmentSourceForPath,"setAttachmentSourceF\
orPath");function getAttachmentSourceForPath(e){const t=normalizeAttachmentPath(e);return t?normalizeAttachmentSource(
attachmentSourceByPath.get(t)):"unknown"}r(getAttachmentSourceForPath,"getAttachmentSourceForPath");
function setRowAttachmentSource(e,t){if(!e)return;const n=normalizeAttachmentSource(t);e.dataset.fileSource=
n;const i=e.getAttribute("data-filename");i&&setAttachmentSourceForPath(i,n)}r(setRowAttachmentSource,
"setRowAttachmentSource");function getRowAttachmentSource(e){if(!e)return"unknown";const t=normalizeAttachmentSource(
e.dataset.fileSource);if(t!=="unknown")return t;const n=e.getAttribute("data-filename");return getAttachmentSourceForPath(
n)}r(getRowAttachmentSource,"getRowAttachmentSource");function getRowOriginalAttachmentSource(e){if(!e)
return"unknown";const t=normalizeAttachmentSource(e.dataset.originalSource);if(t!=="unknown")return t;
const n=e.getAttribute("data-original-filename");return getAttachmentSourceForPath(n)}r(getRowOriginalAttachmentSource,
"getRowOriginalAttachmentSource");function prepareMarkerBaseCanvas(e,t,n){const i=document.createElement(
"canvas");i.width=t,i.height=n;const a=i.getContext("2d");a?(a.drawImage(e,0,0,t,n),markerState.baseImageData=
a.getImageData(0,0,t,n),markerState.baseCanvas=i):(markerState.baseImageData=null,markerState.baseCanvas=
null)}r(prepareMarkerBaseCanvas,"prepareMarkerBaseCanvas");function renderCropOverlay(){const e=get(
"marker-crop-canvas");if(!e)return;const t=e.getContext("2d");if(!t)return;t.clearRect(0,0,e.width,e.
height);const n=r((l,c,u=null,f=!1)=>{if(!l)return;const h=Math.max(0,l.x),y=Math.max(0,l.y),v=Math.
max(1,l.w),w=Math.max(1,l.h);u&&(t.fillStyle=u,t.fillRect(h,y,v,w)),t.save(),f&&t.setLineDash([6,4]),
t.strokeStyle=c,t.lineWidth=2,t.strokeRect(h+.5,y+.5,Math.max(1,v-1),Math.max(1,w-1)),t.restore()},"\
drawRect"),i=markerState.cropRect,a=i&&i.x===0&&i.y===0&&Math.abs(i.w-e.width)<1&&Math.abs(i.h-e.height)<
1;if(i&&(markerState.mode==="crop"||!a)){t.fillStyle="rgba(0,0,0,0.35)",t.fillRect(0,0,e.width,e.height);
const l=Math.max(0,i.x),c=Math.max(0,i.y),u=Math.max(1,i.w),f=Math.max(1,i.h);t.clearRect(l,c,u,f),markerState.
mode==="crop"?n(i,"rgba(250,204,21,0.9)"):n(i,"rgba(250,204,21,0.4)")}if(markerState.mode==="crop"||
markerState.mode!=="mosaic")return;(Array.isArray(markerState.mosaicRects)?markerState.mosaicRects:[]).
forEach(l=>n(l,"rgba(250,204,21,0.9)","rgba(250,204,21,0.10)")),markerState.mosaicPreviewRect&&n(markerState.
mosaicPreviewRect,"rgba(56,189,248,0.95)","rgba(56,189,248,0.14)",!0)}r(renderCropOverlay,"renderCro\
pOverlay");function collectImageUrlsForSend(){return collectAttachmentItemsForSend().map(e=>e.path)}
r(collectImageUrlsForSend,"collectImageUrlsForSend");function collectAttachmentItemsForSend(){const e=[],
t=new Map,n=r((a,o,l)=>{const c=normalizeAttachmentPath(a);if(!c)return;const u=normalizeAttachmentSource(
o),f=normalizeAttachmentDisplayName(l)||getAttachmentNameForPath(c),h=t.get(c);if(h===void 0){const w=e.
length;t.set(c,w),e.push({path:c,source:u,name:f});return}const y=e[h];if(!y)return;const v=normalizeAttachmentSource(
y.source);(v==="unknown"&&u!=="unknown"||v==="library"&&u==="upload")&&(y.source=u),!normalizeAttachmentDisplayName(
y.name)&&f&&(y.name=f)},"pushItem"),i=get("upload-list");return i&&i.querySelectorAll("[data-filenam\
e]").forEach(a=>{const o=a.getAttribute("data-filename");n(o,getRowAttachmentSource(a),getRowAttachmentName(
a));const l=a.getAttribute("data-original-filename");a.dataset.attachOriginal==="1"&&n(l,getRowOriginalAttachmentSource(
a),getAttachmentNameForPath(l))}),currentImageUrls&&currentImageUrls.length&&currentImageUrls.forEach(
a=>{n(a,getAttachmentSourceForPath(a),getAttachmentNameForPath(a))}),e}r(collectAttachmentItemsForSend,
"collectAttachmentItemsForSend");function collectUploadedImageUrlsForSend(){return collectAttachmentItemsForSend().
filter(e=>normalizeAttachmentSource(e.source)==="upload").map(e=>e.path)}r(collectUploadedImageUrlsForSend,
"collectUploadedImageUrlsForSend");function purgeUnsupportedAttachments(e=!0){const t=getModelMediaSupport(
get("model-select").value);let n=0,i=0;if(Array.isArray(currentImageUrls)&&currentImageUrls.length){
const o=[];currentImageUrls.forEach(l=>{const c=normalizeAttachmentPath(l);if(!c)return;const u=isAudioPath(
c),f=isVideoPath(c);if(u&&!t.audio||f&&!t.video){u&&(n+=1),f&&(i+=1);return}o.push(c)}),o.length!==currentImageUrls.
length&&(currentImageUrls=o)}const a=get("upload-list");if(a&&(a.querySelectorAll("[data-filename]").
forEach(o=>{const l=o.getAttribute("data-filename");l&&!currentImageUrls.includes(l)&&(isAudioPath(l)||
isVideoPath(l))&&(setRowMarkerState(o,!1),o.remove())}),a.children.length===0&&(a.innerHTML='<div cl\
ass="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')),
updateFilePreview(),e&&(n||i)){const o=[];n&&o.push(`${n}\u4EF6\u306E\u97F3\u58F0`),i&&o.push(`${i}\u4EF6\
\u306E\u52D5\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${o.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u524A\u9664\u3057\u307E\
\u3057\u305F`,"error",!0)}}r(purgeUnsupportedAttachments,"purgeUnsupportedAttachments");function getRowImageSource(e){
if(!e)return"";const t=e.getAttribute("data-local-url");if(t)return t;const n=e.getAttribute("data-f\
ilename");return n?buildFileUrl(n):""}r(getRowImageSource,"getRowImageSource");function buildFileUrl(e){
const t=normalizeAttachmentPath(e);return t?FILE_BASE_URL+t:""}r(buildFileUrl,"buildFileUrl");function buildAttachmentPreviewUrl(e){
const t=normalizeAttachmentPath(e);return t?isImagePath(t)?FILE_THUMB_BASE_URL+t:FILE_BASE_URL+t:""}
r(buildAttachmentPreviewUrl,"buildAttachmentPreviewUrl"),window.closeMarkerModal=(e=!1)=>{hideModal(
"marker-modal"),!e&&location.pathname==="/edit-image"&&history.back()};function openMarkerModalForRow(e,t=!1){
const n=getRowImageSource(e);if(!n){showToast("\u753B\u50CF\u304C\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}markerState.row=e,markerState.maskMode=t;const i=e?e.querySelector(".truncate"):null;
markerState.filename=i?i.textContent.trim():"image.png",markerState.hasStroke=!1,markerState.history=
[],markerState.naturalWidth=0,markerState.naturalHeight=0,markerState.cropRect=null,markerState.mosaicRects=
[],markerState.mosaicPreviewRect=null,markerState.baseCanvas=null,markerState.baseImageData=null,setMarkerMode(
"draw");const a=get("marker-attach-original");a&&(a.checked=e.dataset.attachOriginal==="1");const o=get(
"marker-image"),l=get("marker-canvas"),c=get("marker-crop-canvas");if(l){const u=l.getContext("2d");
u&&u.clearRect(0,0,l.width,l.height)}if(c){const u=c.getContext("2d");u&&u.clearRect(0,0,c.width,c.height)}
resetMarkerTransform(),showModal("marker-modal"),location.pathname!=="/edit-image"&&history.pushState(
{modal:"marker"},"","/edit-image"),o&&(o.onload=()=>{if(!get("marker-stage")||!l)return;const f=Math.
max(1,Math.floor(o.clientWidth)),h=Math.max(1,Math.floor(o.clientHeight));l.width=f,l.height=h,l.style.
width=`${f}px`,l.style.height=`${h}px`,l.style.left="0px",l.style.top="0px",c&&(c.width=f,c.height=h,
c.style.width=`${f}px`,c.style.height=`${h}px`,c.style.left="0px",c.style.top="0px"),markerState.naturalWidth=
o.naturalWidth||f,markerState.naturalHeight=o.naturalHeight||h;const y=l.getContext("2d");y&&y.clearRect(
0,0,l.width,l.height),prepareMarkerBaseCanvas(o,f,h),saveMarkerHistory(),markerState.mode==="crop"&&
!markerState.cropRect&&resetCropRectToFull(),renderCropOverlay(),resetMarkerTransform()},o.src=n)}r(
openMarkerModalForRow,"openMarkerModalForRow");let uploadProgressState={total:0,completed:0,active:0,
perFilePct:{}};const uploadCancelTokens=new Set;let uploadGeneration=0;function isUploadCancelled(e){
return!!(e&&(e.generation!==void 0&&e.generation!==uploadGeneration||e.uploadId&&uploadCancelTokens.
has(e.uploadId)||e.row&&!e.row.isConnected))}r(isUploadCancelled,"isUploadCancelled");function updateGlobalUploadProgress(e,t){
uploadProgressState.perFilePct.hasOwnProperty(e)&&(uploadProgressState.perFilePct[e]=t,updateFilePreview())}
r(updateGlobalUploadProgress,"updateGlobalUploadProgress");function resetUploadState(){uploadGeneration++,
browserFastLocalFiles.forEach(l=>{const c=l&&l.rowObj?l.rowObj.row:null,u=c?c.getAttribute("data-loc\
al-url"):null;u&&URL.revokeObjectURL(u)}),browserFastLocalFiles.clear(),currentImageUrls=[],currentMaskImage=
null,uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},uploadCancelTokens.clear(),markerAppliedUploads.
clear();const e=get("file-preview");e&&e.classList.add("hidden");const t=get("file-preview-thumbs");
t&&(t.innerHTML="",t.classList.add("hidden"));const n=get("upload-list");n&&(n.innerHTML='<div class\
="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>'),
updateFilePreview(),updateMaskPreview();const i=get("file-input");i&&(i.value="");const a=get("photo\
-input");a&&(a.value="");const o=get("mask-input");o&&(o.value="")}r(resetUploadState,"resetUploadSt\
ate");async function uploadMaskFile(e){if(!e)return!1;const t=new FormData;t.append("file",e);try{const n=await fetch(
CHAT_CONFIG.urls.upload,{method:"POST",body:t}),i=await n.json();if(n.ok&&i.filename)return currentMaskImage=
i.filename,updateMaskPreview(),!0;showToast(i.error||"Mask upload failed","error",!0)}catch{showToast(
"Mask upload failed","error",!0)}return!1}r(uploadMaskFile,"uploadMaskFile");function setCameraCaptureStatus(e,t=!1){
const n=get("camera-status");n&&(n.textContent=e||"",n.classList.toggle("text-red-300",!!t),n.classList.
toggle("text-gray-400",!t))}r(setCameraCaptureStatus,"setCameraCaptureStatus");function updateCameraCapturePendingUi(){
const e=cameraCapturePendingFiles.length,t=get("camera-attach-btn");t&&(t.disabled=e===0||cameraCaptureBusy,
t.textContent=e?`\u6DFB\u4ED8 (${e})`:"\u6DFB\u4ED8 (0)");const n=get("camera-clear-btn");n&&(n.disabled=
e===0||cameraCaptureBusy);const i=get("camera-capture-preview-list");i&&(i.innerHTML="",cameraCapturePendingPreviewUrls.
forEach((a,o)=>{const l=document.createElement("div");l.className="relative rounded overflow-hidden \
border border-gray-700 bg-black aspect-square",l.innerHTML=`
                        <img src="${a}" alt="capture ${o+1}" class="w-full h-full object-cover block\
">
                        <div class="absolute bottom-0 right-0 text-[10px] px-1 py-0.5 bg-black/70 te\
xt-white">${o+1}</div>
                    `,i.appendChild(l)}),i.classList.toggle("hidden",e===0))}r(updateCameraCapturePendingUi,
"updateCameraCapturePendingUi");function resetCameraCapturePending(e={}){for(;cameraCapturePendingPreviewUrls.
length;){const t=cameraCapturePendingPreviewUrls.pop();try{URL.revokeObjectURL(t)}catch{}}cameraCapturePendingFiles.
length=0,updateCameraCapturePendingUi(),e.keepStatus||setCameraCaptureStatus(cameraCaptureStream?"\u64AE\u5F71\
\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002":
"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...")}r(resetCameraCapturePending,"resetCameraCapturePend\
ing");function stopCameraCaptureStream(){const e=get("camera-video");if(e&&e.srcObject){try{e.pause()}catch{}
e.srcObject=null}if(cameraCaptureStream)try{cameraCaptureStream.getTracks().forEach(i=>{try{i.stop()}catch{}})}catch{}
cameraCaptureStream=null,cameraCaptureBusy=!1;const t=get("camera-capture-btn");t&&(t.disabled=!0);const n=get(
"camera-switch-btn");n&&(n.disabled=!0)}r(stopCameraCaptureStream,"stopCameraCaptureStream");async function startCameraCaptureStream(e="\
environment"){const t=get("camera-video");if(!t)throw new Error("camera video element not found");if(!navigator.
mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error("\u3053\u306E\u30D6\u30E9\u30A6\u30B6\u306F\u30AB\u30E1\u30E9API\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093");
stopCameraCaptureStream(),setCameraCaptureStatus("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u4E2D...");const n=get(
"camera-switch-btn");n&&(n.disabled=!0);const i=[{video:{facingMode:{ideal:e},width:{ideal:1920},height:{
ideal:1080}},audio:!1},{video:{facingMode:e},audio:!1},{video:!0,audio:!1}];let a=null;for(const o of i)
try{const l=await navigator.mediaDevices.getUserMedia(o);cameraCaptureStream=l,t.srcObject=l;try{await t.
play()}catch{}const c=l.getVideoTracks&&l.getVideoTracks()[0],u=c&&c.getSettings?c.getSettings():{},
f=String(u.facingMode||"").toLowerCase();f==="user"||f==="environment"?cameraCaptureFacingMode=f:cameraCaptureFacingMode=
e;const h=get("camera-capture-btn");return h&&(h.disabled=!1),n&&(n.disabled=!1),setCameraCaptureStatus(
cameraCapturePendingFiles.length>0?`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`:
"\u64AE\u5F71\u3057\u3066\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002\u6700\u5F8C\u306B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002"),
updateCameraCapturePendingUi(),l}catch(l){a=l}throw a||new Error("\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F")}
r(startCameraCaptureStream,"startCameraCaptureStream");async function openCameraCaptureModal(){if(!window.
isSecureContext&&location.hostname!=="localhost"&&location.hostname!=="127.0.0.1"){showToast("\u30AB\u30E1\u30E9\u8D77\u52D5\u306F\
 HTTPS / localhost \u74B0\u5883\u3067\u5229\u7528\u3067\u304D\u307E\u3059\u3002\u5199\u771F\u9078\u629E\u306B\u5207\u308A\u66FF\u3048\u307E\u3059\u3002",
"warning",!0);const e=get("photo-input");e&&e.click();return}resetCameraCapturePending({keepStatus:!0}),
updateCameraCapturePendingUi(),showModal("camera-capture-modal"),location.pathname!=="/camera"&&history.
pushState({modal:"camera"},"","/camera");try{await startCameraCaptureStream(cameraCaptureFacingMode||
"environment")}catch(e){const t=e&&e.message?e.message:"\u30AB\u30E1\u30E9\u3092\u8D77\u52D5\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F";
setCameraCaptureStatus(t,!0),showToast(t,"error",!0);const n=get("camera-capture-btn");n&&(n.disabled=
!0);const i=get("camera-attach-btn");i&&(i.disabled=!0)}}r(openCameraCaptureModal,"openCameraCapture\
Modal");function closeCameraCaptureModal(e={}){const t=e.skipHistory||!1;hideModal("camera-capture-m\
odal",e),!t&&location.pathname==="/camera"&&history.back()}r(closeCameraCaptureModal,"closeCameraCap\
tureModal");async function toggleCameraCaptureFacing(){if(cameraCaptureBusy)return;const e=get("came\
ra-switch-btn");e&&(e.disabled=!0);const t=String(cameraCaptureFacingMode||"").toLowerCase()==="user"?
"environment":"user";cameraCaptureFacingMode=t;try{await startCameraCaptureStream(t)}catch(n){const i=n&&
n.message?n.message:"\u30AB\u30E1\u30E9\u5207\u66FF\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
i,!0),showToast(i,"error",!0)}finally{e&&get("camera-capture-modal")&&!get("camera-capture-modal").classList.
contains("hidden")&&(e.disabled=!1)}}r(toggleCameraCaptureFacing,"toggleCameraCaptureFacing");function buildCameraCaptureFilename(){
const e=new Date,t=r(a=>String(a).padStart(2,"0"),"pad"),n=String(e.getMilliseconds()).padStart(3,"0");
cameraCaptureSequence=(cameraCaptureSequence+1)%1e3;const i=String(cameraCaptureSequence).padStart(3,
"0");return`camera_${e.getFullYear()}${t(e.getMonth()+1)}${t(e.getDate())}_${t(e.getHours())}${t(e.getMinutes())}${t(
e.getSeconds())}_${n}_${i}.jpg`}r(buildCameraCaptureFilename,"buildCameraCaptureFilename");async function captureCameraShot(){
if(cameraCaptureBusy)return;const e=get("camera-video"),t=get("camera-canvas"),n=get("camera-capture\
-modal");if(!e||!t||!n)return;if(!e.videoWidth||!e.videoHeight){showToast("\u30AB\u30E1\u30E9\u6620\u50CF\u306E\u6E96\u5099\u4E2D\u3067\u3059\u3002\u5C11\u3057\u5F85\u3063\u3066\u304B\u3089\u518D\u5EA6\u304A\u8A66\u3057\u304F\
\u3060\u3055\u3044\u3002","warning",!0);return}cameraCaptureBusy=!0;const i=get("camera-capture-btn");
i&&(i.disabled=!0);const a=get("camera-attach-btn");a&&(a.disabled=!0),setCameraCaptureStatus("\u64AE\u5F71\u4E2D..\
.");try{t.width=e.videoWidth,t.height=e.videoHeight;const o=t.getContext("2d");if(!o)throw new Error(
"\u64AE\u5F71\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F");o.drawImage(e,0,0,t.width,t.height);
const l=await new Promise((u,f)=>{t.toBlob(h=>{h?u(h):f(new Error("\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F"))},
"image/jpeg",.92)}),c=new File([l],buildCameraCaptureFilename(),{type:"image/jpeg",lastModified:Date.
now()});cameraCapturePendingFiles.push(c),cameraCapturePendingPreviewUrls.push(URL.createObjectURL(l)),
updateCameraCapturePendingUi(),setCameraCaptureStatus(`${cameraCapturePendingFiles.length}\u679A\u64AE\u5F71\u6E08\u307F\u3002\u7D9A\u3051\u3066\u64AE\
\u5F71\u3059\u308B\u304B\u300C\u6DFB\u4ED8\u300D\u3092\u62BC\u3057\u3066\u304F\u3060\u3055\u3044\u3002`)}catch(o){
const l=o&&o.message?o.message:"\u64AE\u5F71\u306B\u5931\u6557\u3057\u307E\u3057\u305F";setCameraCaptureStatus(
l,!0),showToast(l,"error",!0)}finally{cameraCaptureBusy=!1,i&&n&&!n.classList.contains("hidden")&&(i.
disabled=!1),updateCameraCapturePendingUi()}}r(captureCameraShot,"captureCameraShot");async function attachCameraCapturedFiles(){
if(cameraCaptureBusy)return;if(!cameraCapturePendingFiles.length){showToast("\u5148\u306B\u64AE\u5F71\u3057\u3066\u304F\u3060\u3055\u3044",
"warning",!0);return}const e=get("camera-capture-modal");cameraCaptureBusy=!0;const t=get("camera-ca\
pture-btn"),n=get("camera-switch-btn"),i=get("camera-attach-btn"),a=get("camera-clear-btn");t&&(t.disabled=
!0),n&&(n.disabled=!0),i&&(i.disabled=!0),a&&(a.disabled=!0);const o=Array.from(cameraCapturePendingFiles).
reverse();closeCameraCaptureModal({skipReset:!0}),cameraCaptureBusy=!0,setCameraCaptureStatus(`${o.length}\
\u679A\u3092\u6DFB\u4ED8\u4E2D...`);try{if(get("camera-split-enabled").checked)await openImageSplit(
o);else{const c=(await handleFiles(o,{openModal:!1,cameraCapture:!0})).filter(Boolean).length;c>0&&showToast(
`${c}\u679A\u306E\u753B\u50CF\u3092\u6DFB\u4ED8\u3057\u307E\u3057\u305F`,"success"),c<o.length&&(showToast(
"\u6DFB\u4ED8\u3067\u304D\u306A\u304B\u3063\u305F\u753B\u50CF\u306F\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u753B\u9762\u304B\u3089\u518D\u8A66\u884C\u3067\u304D\u307E\u3059\u3002",
"warning",!0),openUploadModal())}}catch(l){const c=l&&l.message?l.message:"\u64AE\u5F71\u753B\u50CF\u306E\u6DFB\u4ED8\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
showToast(c,"error",!0)}finally{cameraCaptureBusy=!1,resetCameraCapturePending({keepStatus:!0}),e&&!e.
classList.contains("hidden")&&(t&&(t.disabled=!1),n&&(n.disabled=!1),updateCameraCapturePendingUi())}}
r(attachCameraCapturedFiles,"attachCameraCapturedFiles");function openUploadModal(){typeof window.hideDropOverlay==
"function"&&window.hideDropOverlay(),syncUploadRowsFromCurrent(),showModal("upload-modal"),location.
pathname!=="/upload"&&history.pushState({modal:"upload"},"","/upload");const e=get("vision-model-inf\
o");if(e){const n=(get("model-select")?get("model-select").value:"").toLowerCase(),i=n==="deepseek-v\
4.1-flash"||n==="deepseek-v4-flash-vision-exp",a=["glm-5.3-flash","glm-5.3-flashx","glm-4.6v","glm-4\
.6v-flashx","glm-4.6v-flash","glm-4.5v"].includes(n),o=n.includes("deepseek")&&!i||n.startsWith("glm\
-")&&!a;e.classList.toggle("hidden",!o)}_syncVisionModelDisplay()}r(openUploadModal,"openUploadModal");
function _syncVisionModelDisplay(){const e=get("vision-model-display");if(!e)return;const t=currentVisionModel;
if(t){let n=t;MODELS.forEach(i=>(i.items||[]).forEach(a=>{a.id===t&&(n=a.name)})),e.textContent=n}else
e.textContent="\u8A2D\u5B9A\u304B\u3089\u9078\u629E"}r(_syncVisionModelDisplay,"_syncVisionModelDisp\
lay");function _openVisionModelSelector(){window._visionPickerActive=!0,openModelModal(),setTimeout(
()=>{const e=get("model-search");e&&(e.value=""),renderModelList("")},50)}r(_openVisionModelSelector,
"_openVisionModelSelector");function closeUploadModal(e=!1){typeof window.hideDropOverlay=="function"&&
window.hideDropOverlay(),hideModal("upload-modal"),!e&&location.pathname==="/upload"&&history.back()}
r(closeUploadModal,"closeUploadModal");function syncUploadRowsFromCurrent(){const e=get("upload-list");
if(!e)return;const t=new Set;e.querySelectorAll("[data-filename]").forEach(n=>{const i=n.getAttribute(
"data-filename");i&&t.add(i)}),currentImageUrls.forEach(n=>{t.has(n)||addStoredUploadRow(n,{source:getAttachmentSourceForPath(
n),displayName:getAttachmentNameForPath(n)})}),e.children.length===0&&(e.innerHTML='<div class="text\
-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')}
r(syncUploadRowsFromCurrent,"syncUploadRowsFromCurrent");function decrementUploadTotal(e){uploadProgressState.
total>0&&uploadProgressState.total--,uploadProgressState.perFilePct.hasOwnProperty(e)&&(delete uploadProgressState.
perFilePct[e],uploadProgressState.active>0&&uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview()}r(decrementUploadTotal,"decrementUploadTotal");function addStoredUploadRow(e,t={}){
if(!e||(e=normalizeAttachmentPath(e),!e))return null;const n=normalizeAttachmentSource(t.source),i=get(
"upload-list");if(!i)return null;i.children.length===1&&i.children[0].classList.contains("text-gray-\
500")&&(i.innerHTML="");const a=e.split("/").pop()||e,o=normalizeAttachmentDisplayName(t.displayName)||
getAttachmentNameForPath(e)||a,l=(a.split(".").pop()||"").toLowerCase(),c=["png","jpg","jpeg","webp",
"gif"].includes(l),u=buildFileUrl(e),f=c?buildAttachmentPreviewUrl(e):u,h=`lib_${Date.now()}_${Math.
random().toString(36).slice(2,8)}`,y=document.createElement("div");y.className="upload-row ui-enter \
bg-gray-900/60 rounded p-2",y.dataset.uploadId=h,y.setAttribute("data-filename",e),y.dataset.fileSource=
n,y.dataset.displayName=o,y.dataset.defaultDisplayName=o,y.dataset.sendNameCustomized="";const v=escapeHtml(
o),w=c&&!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\
\u753B\u50CF\u7DE8\u96C6</button>':"",x=c?`<img src="${f}" loading="lazy" decoding="async" class="up\
load-preview w-12 h-12 object-cover rounded border border-gray-700 cursor-pointer" alt="${v}">`:'<di\
v class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center justi\
fy-center text-gray-400 text-sm cursor-pointer">FILE</div>';y.innerHTML=`
                <div class="flex items-center gap-3">
                    ${x}
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
            `;const _=y.querySelector(".upload-preview");_&&(_.onclick=()=>openFileViewer(u,getRowAttachmentName(
y)||o));const S=y.querySelector(".upload-send-name");S&&(S.onclick=()=>promptRowAttachmentName(y));const C=y.
querySelector(".upload-remove");C&&(C.onclick=()=>{uploadCancelTokens.add(h),browserFastLocalFiles.delete(
h),decrementUploadTotal(h);const E=y.getAttribute("data-filename");E&&(currentImageUrls=currentImageUrls.
filter(H=>H!==E)),setRowMarkerState(y,!1),y.remove(),updateFilePreview(),i.children.length===0&&(i.innerHTML=
'<div class="text-xs text-gray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const M=y.querySelector(".upload-marker");return M&&(M.onclick=()=>openMarkerModalForRow(y)),setAttachmentSourceForPath(
e,n),setAttachmentNameForPath(e,o),i.prepend(y),{row:y,bar:y.querySelector(".upload-progress > div"),
status:y.querySelector(".upload-status"),uploadId:h}}r(addStoredUploadRow,"addStoredUploadRow");function addUploadRow(e){
const t=get("upload-list");if(!t)return null;t.children.length===1&&t.children[0].classList.contains(
"text-gray-500")&&(t.innerHTML="");const n=`up_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,
i=document.createElement("div");i.className="upload-row ui-enter bg-gray-900/60 rounded p-2",i.dataset.
uploadId=n,i.dataset.fileSource="upload";const a=normalizeAttachmentDisplayName(e.name||"file")||"fi\
le";i.dataset.displayName=a,i.dataset.defaultDisplayName=a,i.dataset.sendNameCustomized="";const o=escapeHtml(
a),l=e&&e.type&&e.type.startsWith("image/");let c='<div class="upload-preview w-12 h-12 bg-gray-800 \
rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm">FILE</div>';const u=l&&
!browserFastModeEnabled?'<button class="upload-marker text-[10px] border rounded px-2 py-1">\u753B\u50CF\u7DE8\u96C6</bu\
tton>':"";let f="";l?(f=URL.createObjectURL(e),c=`<img src="${f}" class="upload-preview w-12 h-12 ob\
ject-cover rounded border border-gray-700 cursor-pointer" alt="${o}">`):(f=URL.createObjectURL(e),c=
'<div class="upload-preview w-12 h-12 bg-gray-800 rounded border border-gray-700 flex items-center j\
ustify-center text-gray-400 text-sm cursor-pointer">FILE</div>'),i.innerHTML=`
                <div class="flex items-center gap-3">
                    ${c}
                    <div class="flex-1 min-w-0">
                        <div class="truncate text-xs text-gray-200">${o}</div>
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
onclick=()=>{const x=i.getAttribute("data-filename"),_=x?buildFileUrl(x):i.getAttribute("data-local-\
url"),S=normalizeAttachmentDisplayName(i.dataset.displayName)||e.name||x||"";openFileViewer(_,S)});const y=i.
querySelector(".upload-remove");y&&(y.onclick=()=>{uploadCancelTokens.add(n),browserFastLocalFiles.delete(
n),decrementUploadTotal(n);const x=i.getAttribute("data-local-url");x&&URL.revokeObjectURL(x);const _=i.
getAttribute("data-filename");_&&(currentImageUrls=currentImageUrls.filter(S=>S!==_)),setRowMarkerState(
i,!1),i.remove(),updateFilePreview(),t.children.length===0&&(t.innerHTML='<div class="text-xs text-g\
ray-500">\u307E\u3060\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u304C\u3042\u308A\u307E\u305B\u3093\u3002</div>')});
const v=i.querySelector(".upload-marker");v&&(v.onclick=()=>openMarkerModalForRow(i));const w=i.querySelector(
".upload-send-name");return w&&(w.onclick=()=>promptRowAttachmentName(i)),t.prepend(i),{uploadId:n,row:i,
status:i.querySelector(".upload-status"),bar:i.querySelector(".upload-progress > div")}}r(addUploadRow,
"addUploadRow");const CHUNK_THRESHOLD_BYTES=20*1024*1024;async function uploadFileChunked(e,t){if(isUploadCancelled(
t)||!e)return!1;let n=!1;window.ConnectionMonitor&&(window.ConnectionMonitor.operationStarted(),n=!0);
try{const i=await apiFetch("/upload/init",{method:"POST",headers:{"Content-Type":"application/json",
"X-CSRF-Token":csrfToken},body:JSON.stringify({filename:e.name,size:e.size})}),a=await i.json();if(!i.
ok){const y=a&&a.error?a.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
return t&&t.status&&(t.status.textContent="\u5931\u6557"),showToast(y,"error",!0),!1}const o=a.upload_id,
l=a.chunk_size||10*1024*1024,c=Math.ceil(e.size/l);for(let y=0;y<c;y++){const v=y*l,w=Math.min(e.size,
v+l),x=e.slice(v,w);if(!await new Promise(S=>{const C=new XMLHttpRequest;C.open("POST","/upload/chun\
k",!0),C.setRequestHeader("X-CSRF-Token",csrfToken),C.upload.onprogress=E=>{if(E.lengthComputable&&t&&
t.bar){const H=v+E.loaded,B=Math.min(100,Math.floor(H/e.size*100));t.bar.style.width=`${B}%`,t.status&&
(t.status.textContent=`${B}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,B)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},C.onload=()=>{C.status>=200&&C.status<300?S(!0):S(!1)},C.
onerror=()=>S(!1);const M=new FormData;M.append("upload_id",o),M.append("index",String(y)),M.append(
"total",String(c)),M.append("chunk",x,e.name),C.send(M)}))return t&&t.status&&(t.status.textContent=
"\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}t&&t.status&&(t.status.textContent="\u51E6\u7406\u4E2D...");const u=await apiFetch("/\
upload/complete",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrfToken},
body:JSON.stringify({upload_id:o})}),f=await u.json();if(u.ok&&f&&f.filename){if(isUploadCancelled(t))
return t.row&&t.row.parentNode&&t.row.remove(),!1;if(t&&t.row){const w=t.row.getAttribute("data-loca\
l-url");w&&URL.revokeObjectURL(w),t.row.removeAttribute("data-local-url");const x=t.row.querySelector(
"img.upload-preview");if(x){const _=f.filename.replace(/^\d+\//,"");x.src=buildAttachmentPreviewUrl(
_)}}const y=normalizeAttachmentPath(f.filename);if(y&&currentImageUrls.push(y),t&&t.row&&(t.row.setAttribute(
"data-filename",y||f.filename),setRowAttachmentSource(t.row,"upload"),y)){const w=isRowAttachmentNameCustomized(
t.row),x=defaultAttachmentDisplayName(y),_=w&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||
x;t.row.dataset.defaultDisplayName=x,setRowAttachmentName(t.row,_)}return y&&setAttachmentSourceForPath(
y,"upload"),t&&t.status&&(t.status.textContent="\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(f.
filenames)&&f.filenames.length?f.filenames:[f.filename]).forEach(w=>addLibraryFileFromPath(w)),!0}const h=f&&
f.error?f.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";return t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),!1}catch{return t&&t.status&&
(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),!1}finally{n&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded()}}r(uploadFileChunked,
"uploadFileChunked");function uploadFileWithProgress(e,t,n={}){return new Promise(i=>{if(e&&e.size>CHUNK_THRESHOLD_BYTES){
uploadFileChunked(e,t).then(i);return}if(isUploadCancelled(t)){i(!1);return}let a=!1;window.ConnectionMonitor&&
(window.ConnectionMonitor.operationStarted(),a=!0);const o=r(()=>{a&&window.ConnectionMonitor&&(window.
ConnectionMonitor.operationEnded(),a=!1)},"finishUploadOp"),l=new XMLHttpRequest;l.open("POST",CHAT_CONFIG.
urls.upload,!0),l.setRequestHeader("X-CSRF-Token",csrfToken),l.upload.onprogress=u=>{if(u.lengthComputable&&
t&&t.bar){const f=Math.min(100,Math.floor(u.loaded/u.total*100));t.bar.style.width=`${f}%`,t.status&&
(t.status.textContent=`${f}%`),t.uploadId&&updateGlobalUploadProgress(t.uploadId,f)}window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity()},l.onload=()=>{let u={};try{u=JSON.parse(l.responseText||"\
{}")}catch{}if(l.status>=200&&l.status<300&&u&&u.filename){if(isUploadCancelled(t)){t.row&&t.row.parentNode&&
t.row.remove(),o(),i(!1);return}if(t&&t.row){const y=t.row.getAttribute("data-local-url");y&&URL.revokeObjectURL(
y),t.row.removeAttribute("data-local-url");const v=t.row.querySelector("img.upload-preview");if(v){const w=u.
filename.replace(/^\d+\//,"");v.src=buildAttachmentPreviewUrl(w)}}const f=normalizeAttachmentPath(u.
filename);if(f&&currentImageUrls.push(f),t&&t.row&&(t.row.setAttribute("data-filename",f||u.filename),
setRowAttachmentSource(t.row,"upload"),f)){const y=isRowAttachmentNameCustomized(t.row),v=defaultAttachmentDisplayName(
f),w=y&&normalizeAttachmentDisplayName(t.row.dataset.displayName)||v;t.row.dataset.defaultDisplayName=
v,setRowAttachmentName(t.row,w)}f&&setAttachmentSourceForPath(f,"upload"),t&&t.status&&(t.status.textContent=
"\u5B8C\u4E86"),updateFilePreview(),(Array.isArray(u.filenames)&&u.filenames.length?u.filenames:[u.filename]).
forEach(y=>addLibraryFileFromPath(y)),o(),i(!0)}else{const f=[502,503,504].includes(l.status)||l.status>=
520&&l.status<=527;if(n.cameraCapture&&f&&!u.error&&!isUploadCancelled(t)){t&&t.status&&(t.status.textContent=
"\u5206\u5272\u3057\u3066\u518D\u9001\u4E2D..."),o(),uploadFileChunked(e,t).then(i);return}const h=u&&
u.error?u.error:"\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F";t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast(h,"error",!0),o(),i(!1)}},l.onerror=()=>{t&&
t.status&&(t.status.textContent="\u5931\u6557"),showToast("\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\u751F\u3057\u307E\u3057\u305F",
"error",!0),o(),i(!1)};const c=new FormData;c.append("file",e),l.send(c)})}r(uploadFileWithProgress,
"uploadFileWithProgress");function isVideoFile(e){return e?e.type&&e.type.startsWith("video/")?!0:VIDEO_EXTS.
includes(getFileExt(e.name||"")):!1}r(isVideoFile,"isVideoFile");function isAudioFile(e){return e?e.
type&&e.type.startsWith("audio/")?!0:AUDIO_EXTS.includes(getFileExt(e.name||"")):!1}r(isAudioFile,"i\
sAudioFile");function encodeWav(e,t){let n=0;e.forEach(f=>{n+=f.length});const i=new Float32Array(n);
let a=0;e.forEach(f=>{i.set(f,a),a+=f.length});const o=new ArrayBuffer(44+i.length*2),l=new DataView(
o),c=r((f,h)=>{for(let y=0;y<h.length;y++)l.setUint8(f+y,h.charCodeAt(y))},"writeString");c(0,"RIFF"),
l.setUint32(4,36+i.length*2,!0),c(8,"WAVE"),c(12,"fmt "),l.setUint32(16,16,!0),l.setUint16(20,1,!0),
l.setUint16(22,1,!0),l.setUint32(24,t,!0),l.setUint32(28,t*2,!0),l.setUint16(32,2,!0),l.setUint16(34,
16,!0),c(36,"data"),l.setUint32(40,i.length*2,!0);let u=44;for(let f=0;f<i.length;f++){const h=Math.
max(-1,Math.min(1,i[f]));l.setInt16(u,h<0?h*32768:h*32767,!0),u+=2}return new Blob([l],{type:"audio/\
wav"})}r(encodeWav,"encodeWav");function pickAudioRecorderType(){if(typeof MediaRecorder=="undefined")
return"";const e=["audio/webm;codecs=opus","audio/webm","audio/ogg;codecs=opus","audio/ogg"];for(const t of e)
if(MediaRecorder.isTypeSupported(t))return t;return""}r(pickAudioRecorderType,"pickAudioRecorderType");
function updateUploadRowFile(e,t){if(!e||!e.row||!t)return;const n=e.row.querySelector(".truncate"),
i=isRowAttachmentNameCustomized(e.row),a=i?normalizeAttachmentDisplayName(e.row.dataset.displayName)||
"file":normalizeAttachmentDisplayName(t.name||"file")||"file";n&&(n.textContent=a),e.row.dataset.displayName=
a,i||(e.row.dataset.defaultDisplayName=a);const o=e.row.getAttribute("data-local-url");o&&URL.revokeObjectURL(
o);const l=URL.createObjectURL(t);e.row.setAttribute("data-local-url",l);const c=t.type&&t.type.startsWith(
"image/"),u=escapeHtml(a),f=c?`<img src="${l}" class="upload-preview w-12 h-12 object-cover rounded \
border border-gray-700 cursor-pointer" alt="${u}">`:'<div class="upload-preview w-12 h-12 bg-gray-80\
0 rounded border border-gray-700 flex items-center justify-center text-gray-400 text-sm cursor-point\
er">FILE</div>',h=e.row.querySelector(".upload-preview");h&&(h.outerHTML=f);const y=e.row.querySelector(
".upload-preview");y&&(y.onclick=()=>{const w=e.row.getAttribute("data-filename"),x=w?buildFileUrl(w):
e.row.getAttribute("data-local-url");openFileViewer(x,getRowAttachmentName(e.row)||a||w||"")});const v=e.
row.querySelector(".upload-marker");v&&v.classList.toggle("hidden",!c),c||(setRowMarkerState(e.row,!1),
e.row.dataset.originalFilename="",e.row.dataset.originalSource="",e.row.dataset.attachOriginal="")}r(
updateUploadRowFile,"updateUploadRowFile");function saveMarkerHistory(){const e=get("marker-canvas");
if(!e)return;const t=e.getContext("2d");if(!t)return;const n=Array.isArray(markerState.mosaicRects)?
markerState.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[];markerState.history.push({imageData:t.
getImageData(0,0,e.width,e.height),mosaicRects:n}),markerState.history.length>40&&markerState.history.
shift()}r(saveMarkerHistory,"saveMarkerHistory");function undoMarkerCanvas(){if(markerState.history.
length<=1)return;markerState.history.pop();const e=get("marker-canvas");if(!e)return;const t=e.getContext(
"2d");if(!t)return;const n=markerState.history[markerState.history.length-1];t.clearRect(0,0,e.width,
e.height),n&&n.imageData?(t.putImageData(n.imageData,0,0),markerState.mosaicRects=Array.isArray(n.mosaicRects)?
n.mosaicRects.map(i=>({x:i.x,y:i.y,w:i.w,h:i.h})):[]):n?(t.putImageData(n,0,0),markerState.mosaicRects=
[]):markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,markerState.hasStroke=markerState.
history.length>1,renderCropOverlay()}r(undoMarkerCanvas,"undoMarkerCanvas");function clearMarkerCanvas(){
const e=get("marker-canvas");if(!e)return;const t=e.getContext("2d");t&&t.clearRect(0,0,e.width,e.height),
markerState.hasStroke=!1,markerState.mosaicRects=[],markerState.mosaicPreviewRect=null,renderCropOverlay(),
saveMarkerHistory()}r(clearMarkerCanvas,"clearMarkerCanvas");function initMarkerCanvas(){const e=get(
"marker-canvas");if(!e)return;const t=e.getContext("2d"),n=get("marker-size"),i=new Map;let a=!1,o=0,
l=markerView.scale,c={x:0,y:0},u={x:0,y:0},f=[],h=16,y="",v=null,w=null,x=null,_=null,S=null,C=!1,M=null;
const E=r(P=>{const ee=e.getBoundingClientRect(),de=(P.clientX-ee.left)*(e.width/ee.width),ye=(P.clientY-
ee.top)*(e.height/ee.height);return{x:de,y:ye}},"getPoint"),H=r((P,ee)=>({x:(P.x+ee.x)/2,y:(P.y+ee.y)/
2}),"getMid"),B=r((P,ee)=>Math.hypot(P.x-ee.x,P.y-ee.y),"getDist");let te=!1;const j=r(()=>{w||(w=document.
createElement("canvas"),x=w.getContext("2d")),_||(_=document.createElement("canvas"),S=_.getContext(
"2d")),(w.width!==e.width||w.height!==e.height)&&(w.width=e.width,w.height=e.height),(_.width!==e.width||
_.height!==e.height)&&(_.width=e.width,_.height=e.height)},"ensureDrawBuffers"),V=r(()=>{if(!t||!w||
!_)return;const P=markerState.maskMode?1:Math.max(MARKER_OPACITY_MIN_ALPHA,Math.min(1,Number(markerState.
opacity)||.6));t.clearRect(0,0,e.width,e.height),t.drawImage(w,0,0),t.save(),t.globalAlpha=P,t.drawImage(
_,0,0),t.restore()},"renderDrawPreview"),ue=r(()=>{S&&(S.strokeStyle=y,S.fillStyle=y,S.lineWidth=h,S.
lineCap="round",S.lineJoin="round")},"applyMarkerBrush"),me=r(P=>{if(!P)return!1;if(f.length===0)return f.
push(P),!0;const ee=f[f.length-1],de=P.x-ee.x,ye=P.y-ee.y,Z=Math.hypot(de,ye),pe=Math.max(.35,h*.04);
if(Z<pe)return!1;const X=Math.max(1,h*.25),Te=Math.max(1,Math.ceil(Z/X));for(let ne=1;ne<=Te;ne++){const ce=ne/
Te;f.push({x:ee.x+de*ce,y:ee.y+ye*ce})}return!0},"appendStrokePoint"),re=r(()=>{if(S&&(S.clearRect(0,
0,_.width,_.height),f.length!==0)){if(ue(),f.length===1){const P=f[0];S.beginPath(),S.arc(P.x,P.y,h/
2,0,Math.PI*2),S.fill();return}if(S.beginPath(),S.moveTo(f[0].x,f[0].y),f.length===2)S.lineTo(f[1].x,
f[1].y);else{for(let de=1;de<f.length-2;de++){const ye=f[de],Z=f[de+1],pe=H(ye,Z);S.quadraticCurveTo(
ye.x,ye.y,pe.x,pe.y)}const P=f[f.length-2],ee=f[f.length-1];S.quadraticCurveTo(P.x,P.y,ee.x,ee.y)}S.
stroke()}},"renderStrokeLayer"),ke=r((P,ee)=>{if(!P||!ee)return null;const de=Math.min(P.x,ee.x),ye=Math.
min(P.y,ee.y),Z=Math.abs(P.x-ee.x),pe=Math.abs(P.y-ee.y);return{x:de,y:ye,w:Z,h:pe}},"normalizeMosai\
cRect"),I=r(P=>{const ee=n?Number(n.value||16):16,de=Math.max(6,Math.floor(ee)),ye=Math.floor(de/2);
return{x:P.x-ye,y:P.y-ye,w:de,h:de}},"buildMosaicRectFromPoint"),O=r(()=>{const P=document.createElement(
"canvas");P.width=e.width,P.height=e.height;const ee=P.getContext("2d");if(!ee)return null;markerState.
baseCanvas&&ee.drawImage(markerState.baseCanvas,0,0),ee.drawImage(e,0,0);try{return ee.getImageData(
0,0,e.width,e.height)}catch{return null}},"getMosaicSourceImageData"),K=r(P=>{if(!t||!P)return!1;const ee=O();
if(!ee)return!1;const de=n?Number(n.value||16):16,ye=Math.max(4,Math.floor(de/2)),Z=Math.max(0,Math.
floor(P.x)),pe=Math.max(0,Math.floor(P.y)),X=Math.min(e.width,Math.ceil(P.x+P.w)),Te=Math.min(e.height,
Math.ceil(P.y+P.h));if(X<=Z||Te<=pe)return!1;for(let ne=pe;ne<Te;ne+=ye)for(let ce=Z;ce<X;ce+=ye){const Me=Math.
min(ye,X-ce),we=Math.min(ye,Te-ne),Ye=Math.min(e.width-1,Math.max(0,ce+Math.floor(Me/2))),st=(Math.min(
e.height-1,Math.max(0,ne+Math.floor(we/2)))*e.width+Ye)*4,pt=ee.data[st],se=ee.data[st+1],yt=ee.data[st+
2];t.fillStyle=`rgb(${pt},${se},${yt})`,t.fillRect(ce,ne,Me,we)}return!0},"applyMosaicRect"),R=r(P=>{
if(!t)return;if(i.set(P.pointerId,{x:P.clientX,y:P.clientY}),i.size>=2){const de=Array.from(i.values()),
ye=de[0],Z=de[1];a=!0,te=!1,f=[],C=!1,M=null,v=null,markerState.mosaicPreviewRect=null,o=B(ye,Z)||1,
l=markerView.scale,c={x:markerView.offsetX,y:markerView.offsetY},u=H(ye,Z),renderCropOverlay(),e.setPointerCapture&&
e.setPointerCapture(P.pointerId),P.preventDefault();return}if(a||markerState.mode==="crop")return;te=
!0;const ee=E(P);if(markerState.mode==="auto"&&markerState.maskMode)v=ee;else if(markerState.mode===
"mosaic")C=!0,M=ee,markerState.mosaicPreviewRect=I(ee),renderCropOverlay();else{if(j(),!x||!S)return;
x.clearRect(0,0,w.width,w.height),x.drawImage(e,0,0),S.clearRect(0,0,_.width,_.height),h=n?Number(n.
value||16):16,y=markerState.maskMode?"#ffffff":normalizeMarkerHexColor(markerState.colorHex),f=[],me(
ee),re(),markerState.hasStroke=!0,V()}e.setPointerCapture&&e.setPointerCapture(P.pointerId),P.preventDefault()},
"start"),G=r(P=>{if(i.has(P.pointerId)&&i.set(P.pointerId,{x:P.clientX,y:P.clientY}),a&&i.size>=2){const de=Array.
from(i.values()),ye=de[0],Z=de[1],pe=H(ye,Z),X=B(ye,Z)||1,Te=l*(X/o);markerView.scale=Math.min(markerView.
maxScale,Math.max(markerView.minScale,Te)),markerView.offsetX=c.x+(pe.x-u.x),markerView.offsetY=c.y+
(pe.y-u.y),applyMarkerTransform(),P.preventDefault();return}if(!te||!t)return;if(markerState.mode===
"auto"&&markerState.maskMode){P.preventDefault();return}const ee=E(P);if(markerState.mode==="mosaic"){
if(!C||!M)return;markerState.mosaicPreviewRect=ke(M,ee)||I(ee),renderCropOverlay()}else me(ee)&&(re(),
V());P.preventDefault()},"move"),Q=r(P=>{const ee=te;if(i.delete(P.pointerId),i.size<2&&(a=!1),i.size===
0){te=!1;const de=ee&&P.type!=="pointercancel"&&markerState.mode==="auto"&&markerState.maskMode&&v?fillMaskSimilarRegion(
v):!1;if(v=null,ee&&t&&markerState.mode==="draw"&&f.length>0&&(re(),V()),ee&&markerState.mode==="mos\
aic"&&M){const ye=E(P);let Z=ke(M,ye);(!Z||Z.w<2||Z.h<2)&&(Z=I(M)),K(Z)&&(markerState.hasStroke=!0,markerState.
mosaicRects.push(Z))}f=[],C=!1,M=null,markerState.mosaicPreviewRect=null,renderCropOverlay(),ee&&(markerState.
mode!=="auto"||de)&&saveMarkerHistory()}e.releasePointerCapture&&e.releasePointerCapture(P.pointerId),
P.preventDefault()},"end");e.addEventListener("pointerdown",R),e.addEventListener("pointermove",G),e.
addEventListener("pointerup",Q),e.addEventListener("pointercancel",Q)}r(initMarkerCanvas,"initMarker\
Canvas");function fillMaskSimilarRegion(e){var te;const t=get("marker-canvas"),n=markerState.baseImageData;
if(!t||!n)return showToast("\u753B\u50CF\u306E\u8272\u3092\u8AAD\u307F\u53D6\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),!1;const i=n.width,a=n.height,o=Math.floor(e.x),l=Math.floor(e.y);if(o<0||l<0||o>=i||l>=
a)return!1;const c=n.data,u=l*i+o,f=u*4;if(c[f+3]<16)return!1;const h=c[f],y=c[f+1],v=c[f+2],w=Number(
((te=get("marker-auto-tolerance"))==null?void 0:te.value)||32),x=new Uint8Array(i*a),_=[u],S=[],C=25e4,
M=r(j=>{const V=j*4;return c[V+3]>=16&&Math.max(Math.abs(c[V]-h),Math.abs(c[V+1]-y),Math.abs(c[V+2]-
v))<=w},"matches");for(x[u]=1;_.length;){const j=_.pop();if(x[j]===2||!M(j))continue;const V=Math.floor(
j/i);let ue=j%i,me=ue;for(;ue>0&&x[V*i+ue-1]!==2&&M(V*i+ue-1);)ue--;for(;me+1<i&&x[V*i+me+1]!==2&&M(
V*i+me+1);)me++;for(let re=ue;re<=me;re++)x[V*i+re]=2;if(S.push(V,ue,me+1),S.length/3>C)return showToast(
"\u7BC4\u56F2\u304C\u8907\u96D1\u3067\u3059\u3002\u8A31\u5BB9\u5DEE\u3092\u4E0B\u3052\u3066\u304F\u3060\u3055\u3044",
"error",!0),!1;for(const re of[V-1,V+1]){if(re<0||re>=a)continue;let ke=ue;for(;ke<=me;){const I=re*
i+ke;if(x[I]===0&&M(I))for(x[I]=1,_.push(I);ke+1<=me&&x[re*i+ke+1]===0&&M(re*i+ke+1);)ke++,x[re*i+ke]=
1;ke++}}}if(!S.length)return!1;const E=t.getContext("2d");if(!E)return!1;const H=E.getImageData(0,0,
i,a),B=H.data;for(let j=0;j<S.length;j+=3){const V=S[j]*i+S[j+1],ue=S[j]*i+S[j+2];for(let me=V;me<ue;me++){
const re=me*4;B[re]=B[re+1]=B[re+2]=B[re+3]=255}}return E.putImageData(H,0,0),markerState.hasStroke=
!0,!0}r(fillMaskSimilarRegion,"fillMaskSimilarRegion");function initCropCanvas(){const e=get("marker\
-crop-canvas");if(!e)return;const t=e.getContext("2d"),n=new Map;let i=!1,a=null,o=null,l=null,c=!1,
u=0,f=markerView.scale,h={x:0,y:0},y={x:0,y:0};const v=8,w=14,x=r((j,V,ue)=>Math.min(ue,Math.max(V,j)),
"clamp"),_=r(j=>{const V=e.getBoundingClientRect(),ue=(j.clientX-V.left)*(e.width/V.width),me=(j.clientY-
V.top)*(e.height/V.height);return{x:ue,y:me}},"getPoint"),S=r((j,V)=>({x:(j.x+V.x)/2,y:(j.y+V.y)/2}),
"getMid"),C=r((j,V)=>Math.hypot(j.x-V.x,j.y-V.y),"getDist"),M=r(()=>(markerState.cropRect||resetCropRectToFull(),
markerState.cropRect),"ensureCropRect"),E=r((j,V)=>{if(!V)return"move";const ue=V.x,me=V.y,re=V.x+V.
w,ke=V.y+V.h,I=Math.abs(j.x-ue)<=w,O=Math.abs(j.x-re)<=w,K=Math.abs(j.y-me)<=w,R=Math.abs(j.y-ke)<=w;
if(I&&K)return"nw";if(O&&K)return"ne";if(I&&R)return"sw";if(O&&R)return"se";if(K)return"n";if(R)return"\
s";if(I)return"w";if(O)return"e";if(j.x>ue+w&&j.x<re-w&&j.y>me+w&&j.y<ke-w)return"move";const Q=j.x<
ue?"left":j.x>re?"right":null,P=j.y<me?"top":j.y>ke?"bottom":null;if(Q&&P){if(Q==="left"&&P==="top")
return"nw";if(Q==="right"&&P==="top")return"ne";if(Q==="left"&&P==="bottom")return"sw";if(Q==="right"&&
P==="bottom")return"se"}return Q?Q==="left"?"w":"e":P?P==="top"?"n":"s":"move"},"hitTest"),H=r(j=>{if(markerState.
mode!=="crop")return;if(n.set(j.pointerId,{x:j.clientX,y:j.clientY}),n.size>=2){const me=Array.from(
n.values()),re=me[0],ke=me[1];c=!0,i=!1,u=C(re,ke)||1,f=markerView.scale,h={x:markerView.offsetX,y:markerView.
offsetY},y=S(re,ke),e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault();return}
if(c)return;i=!0;const V=_(j),ue=M();o=E(V,ue),a=V,l=ue?{x:ue.x,y:ue.y,w:ue.w,h:ue.h}:null,renderCropOverlay(),
e.setPointerCapture&&e.setPointerCapture(j.pointerId),j.preventDefault()},"start"),B=r(j=>{if(markerState.
mode!=="crop")return;if(n.has(j.pointerId)&&n.set(j.pointerId,{x:j.clientX,y:j.clientY}),c&&n.size>=
2){const Q=Array.from(n.values()),P=Q[0],ee=Q[1],de=S(P,ee),ye=C(P,ee)||1,Z=f*(ye/u);markerView.scale=
Math.min(markerView.maxScale,Math.max(markerView.minScale,Z)),markerView.offsetX=h.x+(de.x-y.x),markerView.
offsetY=h.y+(de.y-y.y),applyMarkerTransform(),renderCropOverlay(),j.preventDefault();return}if(!i||!a||
!l)return;const V=_(j),ue=e.width,me=e.height,re={x:l.x,y:l.y,w:l.w,h:l.h},ke=l.x+l.w,I=l.y+l.h,O=r(
()=>{const Q=x(V.x,0,ke-v);re.x=Q,re.w=ke-Q},"applyW"),K=r(()=>{re.w=x(V.x-l.x,v,ue-l.x)},"applyE"),
R=r(()=>{const Q=x(V.y,0,I-v);re.y=Q,re.h=I-Q},"applyN"),G=r(()=>{re.h=x(V.y-l.y,v,me-l.y)},"applyS");
switch(o){case"move":{const Q=V.x-a.x,P=V.y-a.y;re.x=x(l.x+Q,0,ue-l.w),re.y=x(l.y+P,0,me-l.h);break}case"\
w":O();break;case"e":K();break;case"n":R();break;case"s":G();break;case"nw":R(),O();break;case"ne":R(),
K();break;case"sw":G(),O();break;case"se":G(),K();break;default:break}re.x=x(re.x,0,ue-re.w),re.y=x(
re.y,0,me-re.h),markerState.cropRect=re,renderCropOverlay(),j.preventDefault()},"move"),te=r(j=>{n.delete(
j.pointerId),n.size<2&&(c=!1),n.size===0&&(renderCropOverlay(),i=!1,a=null,o=null,l=null),e.releasePointerCapture&&
e.releasePointerCapture(j.pointerId),j.preventDefault()},"end");e.addEventListener("pointerdown",H),
e.addEventListener("pointermove",B),e.addEventListener("pointerup",te),e.addEventListener("pointerca\
ncel",te),e.addEventListener("pointerleave",te)}r(initCropCanvas,"initCropCanvas");async function saveMarkerToRow(){
const e=markerState.row,t=get("marker-image"),n=get("marker-canvas");if(!e||!t||!n)return;const i=get(
"marker-attach-original");i&&(e.dataset.attachOriginal=i.checked?"1":"");let a=document.createElement(
"canvas");const o=markerState.naturalWidth||t.naturalWidth||n.width,l=markerState.naturalHeight||t.naturalHeight||
n.height;if(markerState.maskMode){if(!markerState.hasStroke){showToast("\u524A\u9664\u3059\u308B\u7BC4\u56F2\u3092\u767D\u304F\u5857\u3063\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const S=document.createElement("canvas");S.width=o,S.height=l;const C=S.getContext(
"2d");if(!C)return;C.fillStyle="#000000",C.fillRect(0,0,o,l),C.drawImage(n,0,0,o,l);const M=await new Promise(
B=>S.toBlob(B,"image/png"));if(!M){showToast("\u30DE\u30B9\u30AF\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const E=(markerState.filename||"image").replace(/\.[^/.]+$/,"");await uploadMaskFile(
new File([M],`${E}_mask.png`,{type:"image/png"}))&&(window.closeMarkerModal(),markerState.row=null,markerState.
maskMode=!1);return}a.width=o,a.height=l;const c=a.getContext("2d");if(!c)return;if(c.drawImage(t,0,
0,o,l),c.drawImage(n,0,0,o,l),markerState.cropRect){const S=o/n.width,C=l/n.height,M=Math.max(0,Math.
floor(markerState.cropRect.x*S)),E=Math.max(0,Math.floor(markerState.cropRect.y*C)),H=Math.min(o,Math.
max(1,Math.floor(markerState.cropRect.w*S))),B=Math.min(l,Math.max(1,Math.floor(markerState.cropRect.
h*C))),te=document.createElement("canvas");te.width=H,te.height=B;const j=te.getContext("2d");j&&(j.
drawImage(a,M,E,H,B,0,0,H,B),a=te)}const u=await new Promise(S=>a.toBlob(S,"image/png",.92));if(!u){
showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u751F\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0);return}const h=(markerState.filename||"marked.png").replace(/\.[^/.]+$/,""),y=new File([u],
`${h}_marked.png`,{type:"image/png"}),v={row:e,uploadId:e.dataset.uploadId,status:e.querySelector(".\
upload-status"),bar:e.querySelector(".upload-progress > div")};v.status&&(v.status.textContent="\u7DE8\u96C6\u53CD\u6620\
\u4E2D..."),updateUploadRowFile(v,y);const w=e.getAttribute("data-filename"),x=getRowAttachmentSource(
e);w&&!e.dataset.originalFilename&&(e.dataset.originalFilename=w,e.dataset.originalSource=x,setAttachmentSourceForPath(
w,x)),await uploadFileWithProgress(y,v)?(w&&(currentImageUrls=currentImageUrls.filter(S=>S!==w)),setRowAttachmentSource(
e,"upload"),setRowMarkerState(e,!0)):showToast("\u7DE8\u96C6\u753B\u50CF\u306E\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),updateFilePreview(),window.closeMarkerModal(),markerState.row=null,markerState.maskMode=
!1}r(saveMarkerToRow,"saveMarkerToRow");async function extractAudioFromVideo(e,t){return!isVideoFile(
e)||!HTMLMediaElement.prototype.captureStream?null:(t&&t.status&&(t.status.textContent="\u97F3\u58F0\u62BD\u51FA\u4E2D..."),
new Promise(n=>{const i=document.createElement("video");i.preload="auto",i.muted=!0,i.playsInline=!0,
i.src=URL.createObjectURL(e);let a=null,o=null,l=null,c=null,u=[],f=null;const h=r(()=>{f&&clearTimeout(
f);try{URL.revokeObjectURL(i.src)}catch{}try{i.remove()}catch{}if(a&&a.getTracks().forEach(v=>v.stop()),
l)try{l.disconnect()}catch{}if(c)try{c.disconnect()}catch{}if(o)try{o.close()}catch{}},"cleanup"),y=r(
()=>{h(),n(null)},"fail");i.onloadedmetadata=async()=>{try{a=i.captureStream();const v=a.getAudioTracks();
if(!v||!v.length)return y();o=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16e3}),
c=o.createMediaStreamSource(new MediaStream(v)),l=o.createScriptProcessor(4096,1,1),l.onaudioprocess=
x=>{const _=x.inputBuffer.getChannelData(0);u.push(new Float32Array(_))},c.connect(l),l.connect(o.destination);
const w=isFinite(i.duration)?Math.max(1,Math.ceil(i.duration*1e3)):0;w>0&&(f=setTimeout(()=>{const x=(e.
name||"video").replace(/\.[^/.]+$/,""),_=encodeWav(u,o.sampleRate),S=new File([_],`${x}.audio.wav`,{
type:"audio/wav"});h(),n(S)},w+250)),await i.play(),i.onended=()=>{const x=(e.name||"video").replace(
/\.[^/.]+$/,""),_=encodeWav(u,o.sampleRate),S=new File([_],`${x}.audio.wav`,{type:"audio/wav"});h(),
n(S)}}catch{y()}},i.onerror=()=>y()}))}r(extractAudioFromVideo,"extractAudioFromVideo");async function handleFiles(e,t={}){
if(!e||!e.length)return;const n=Array.from(e).filter(Boolean);if(!n.length)return;const i=n,a=uploadGeneration;
t.openModal!==!1?openUploadModal():syncUploadRowsFromCurrent(),uploadProgressState.total+=i.length,uploadProgressState.
active+=i.length,updateFilePreview();const o=!!(get("upload-audio-only")&&get("upload-audio-only").checked),
l=getModelMediaSupport(get("model-select").value),c=r(async v=>{let w=null;if(a!==uploadGeneration)return!1;
try{if(isAudioFile(v)&&!l.audio)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(isVideoFile(v)&&!l.video)return showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;if(browserFastModeEnabled&&(!v.type||!v.type.startsWith("image/")))return showToast("\u9AD8\u901F\u30E2\
\u30FC\u30C9\u3067\u306F\u753B\u50CF\u30D5\u30A1\u30A4\u30EB\u3060\u3051\u3092\u6DFB\u4ED8\u3067\u304D\u307E\u3059",
"error",!0),uploadProgressState.total>0&&uploadProgressState.total--,uploadProgressState.active>0&&uploadProgressState.
active--,!1;const x=addUploadRow(v);x.generation=a,updateFilePreview(),w=x.uploadId,uploadProgressState.
perFilePct[w]=0;let _=v;if(o&&isVideoFile(v)){const C=await extractAudioFromVideo(v,x);C?(_=C,updateUploadRowFile(
x,C),x&&x.status&&(x.status.textContent="\u97F3\u58F0\u306E\u307F")):(x&&x.status&&(x.status.textContent=
"\u62BD\u51FA\u5931\u6557: \u52D5\u753B\u9001\u4FE1"),showToast("\u97F3\u58F0\u62BD\u51FA\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u52D5\u753B\u306E\u307E\u307E\u9001\u4FE1\u3057\u307E\u3059\u3002",
"error",!0))}if(!t.skipCompression&&v.type.startsWith("image/"))try{_=await applyImageCompression(v,
readComposerCompression()),_!==v&&updateUploadRowFile(x,_)}catch{}if(isUploadCancelled(x))return!1;if(browserFastModeEnabled){
const C=Array.from(browserFastLocalFiles.values()).reduce((M,E)=>M+Number(E.file&&E.file.size||0),0);
return browserFastLocalFiles.size>=BROWSER_FAST_MAX_IMAGES||C+_.size>BROWSER_FAST_MAX_BYTES?(x&&x.status&&
(x.status.textContent="\u4E0A\u9650\u8D85\u904E"),x&&x.row&&x.row.remove(),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A08\
12MB\u307E\u3067\u3067\u3059","error",!0),!1):(browserFastLocalFiles.set(x.uploadId,{file:_,rowObj:x}),
x.status&&(x.status.textContent="\u30ED\u30FC\u30AB\u30EB\u4FDD\u6301\uFF08\u672A\u4FDD\u5B58\uFF09"),
x.bar&&(x.bar.style.width="100%"),x.row&&(x.row.dataset.browserFastLocal="1"),!0)}const S=await uploadFileWithProgress(
_,x,t);if(!S&&t.cameraCapture&&!isUploadCancelled(x)){const C=document.createElement("button");C.className=
"text-xs border border-gray-700 rounded px-2 py-1",C.textContent="\u518D\u8A66\u884C",C.onclick=()=>{
if(isUploadCancelled(x))return;C.disabled=!0;const M=x.row.getAttribute("data-local-url");M&&URL.revokeObjectURL(
M),x.row.remove(),handleFiles([_],{...t,openModal:!0,skipCompression:!0})},x.row.appendChild(C)}return S}finally{
a===uploadGeneration&&(w&&uploadProgressState.perFilePct.hasOwnProperty(w)&&(delete uploadProgressState.
perFilePct[w],uploadProgressState.completed++,uploadProgressState.active--),uploadProgressState.active<=
0&&(uploadProgressState.total=0,uploadProgressState.completed=0,uploadProgressState.active=0,uploadProgressState.
perFilePct={}),updateFilePreview())}},"processOne"),u=new Array(i.length).fill(!1);let f=0;const h=Math.
min(UPLOAD_CONCURRENCY,i.length),y=Array.from({length:h}).map(async()=>{for(;;){const v=f++;if(v>=i.
length)break;u[v]=await c(i[v])}});return await Promise.all(y),u}r(handleFiles,"handleFiles"),get("c\
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
replace(/\n/g,"<br>");const i=protectMathSegments(n);markdownOpenFenceBody=/<\/svg/i.test(n)?findOpenMarkdownFenceBody(
n):null;let a="";try{a=window.marked.parse(i.text)}finally{markdownOpenFenceBody=null}const o=restoreMathSegments(
a,i.blocks,t);return window.DOMPurify.sanitize(o)}r(sanitizeMarkdownHtml,"sanitizeMarkdownHtml");let markdownOpenFenceBody=null;
function findOpenMarkdownFenceBody(e){const t=String(e||"").split(/\r?\n/);let n=null,i=[];for(const a of t){
if(!n){const l=a.match(/^\s*(`{3,}|~{3,})(.*)$/);l&&!(l[1][0]==="`"&&l[2].includes("`"))&&(n=l[1],i=
[]);continue}const o=a.trim();if(o.length>=n.length&&o[0]===n[0]&&/^(`+|~+)$/.test(o)){n=null;continue}
i.push(a)}return n?i.join(`
`):null}r(findOpenMarkdownFenceBody,"findOpenMarkdownFenceBody");const SVG_CODE_RENDER_MAX_CHARS=3e5;
function getRenderableSvgCode(e,t){const n=String(e||"").trim().toLowerCase();if(n!=="svg"&&n!=="xml"&&
n!=="image/svg+xml")return"";const i=String(t||"").trim();if(!i||i.length>SVG_CODE_RENDER_MAX_CHARS)
return"";const a=i.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i,"");return!/^<svg[\s>]/i.
test(a)||!/<\/svg\s*>$/i.test(a)||markdownOpenFenceBody!==null&&markdownOpenFenceBody.replace(/\s+/g,
"")===i.replace(/\s+/g,"")?"":i}r(getRenderableSvgCode,"getRenderableSvgCode");function buildSvgCodeRenderHtml(e,t){
let n=!0;const i=String(e||"").replace(/<svg\b([^>]*)>/i,(o,l)=>{let c=l;/\sxmlns\s*=/i.test(c)||(c=
` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(e)&&!/\sxmlns:xlink\s*=/i.test(c)&&(c=` \
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const u=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return u&&Number(u[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
i)}`;return`<div class="svg-render-box svg-code-render" data-svg-key="${escapeHtml(String(t||""))}">\
<img src="${a}" alt="SVG"${n?' class="svg-code-fill"':""} decoding="async"></div>`}r(buildSvgCodeRenderHtml,
"buildSvgCodeRenderHtml");function getCanvasModeElements(){const e=get("canvas-panel");return e?{panel:e,
stage:get("conversation-stage"),title:get("canvas-panel-title"),status:get("canvas-panel-status"),blockCount:get(
"canvas-block-count"),blockList:get("canvas-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get(
"canvas-preview-lang"),sourceSelect:get("canvas-source-select"),frame:get("canvas-preview-frame"),empty:get(
"canvas-preview-empty"),sourceScroll:get("canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get(
"canvas-panel-copy-btn"),clearBtn:get("canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:
null}r(getCanvasModeElements,"getCanvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(
e||"").trim().toLowerCase();if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(
t||"");return/<!doctype\s+html/i.test(i)||/<html[\s>]/i.test(i)}r(isCanvasHtmlPreviewCandidate,"isCa\
nvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){const n=String(e&&e.lang?e.lang:"").trim(),
i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),a=!!(e&&e.open);return{...e,index:t,lang:n,code:i,
open:a,key:hashString(`${n||"TEXT"}
${i||""}`)}}r(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],a=[],o=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",u=[];for(const y of n){
if(!l){const x=y.match(o);if(x){l=x[2],c=String(x[3]||"").trim(),u=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const v=String(y||"").trim();if(v&&v.replace(/\s+/g,"")===l){const x=i[i.length-1];x&&(x.
code=u.join(`
`),x.open=!1),l=null,c="",u=[];continue}u.push(y);const w=i[i.length-1];w&&(w.code=u.join(`
`))}if(l&&i.length){const y=i[i.length-1];y&&(y.code=u.join(`
`),y.open=!0)}const f=i.map((y,v)=>normalizeCanvasBlock(y,v)),h=selectCanvasPreviewBlock(f,t);return{
renderText:a.join(`
`),blocks:f,primaryBlock:h?h.block:null,primaryIndex:h?h.index:-1,rawText:t}}r(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const o=i[n];return{block:o,index:n,previewType:isCanvasHtmlPreviewCandidate(
o.lang,o.code)?"html":"code"}}if(i.length>0){const o=i.length-1,l=i[o];return{block:l,index:o,previewType:isCanvasHtmlPreviewCandidate(
l.lang,l.code)?"html":"code"}}const a=String(t||"");return isCanvasHtmlPreviewCandidate("",a)?{block:normalizeCanvasBlock(
{lang:"html",code:a,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}r(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}r(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),a=!!t&&t===i;n.classList.toggle("can\
vas-active",a),n.setAttribute("aria-pressed",a?"true":"false"),n.setAttribute("data-canvas-active",a?
"1":"0"),n.innerHTML=a?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
r(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}r(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
a={preview:0,blocks:1,source:2},o=i[n];if(!o||!(t in a)||!(n in a))return;canvasPreviewState.viewAnimationToken+=
1;const l=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(u=>{u&&u.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),o.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";o.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(o.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}r(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(u=>{const f=u.getAttribute("da\
ta-canvas-panel-view")===i;u.classList.toggle("active",f),u.setAttribute("aria-pressed",f?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}r(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const o=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",h=(String(i&&i.code?i.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${o}`,v=`${y}\u3001${h}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(v)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
o)}</span><span class="canvas-block-chip-preview">${escapeHtml(h)}</span></span><span class="canvas-\
block-chip-state">${l?"\u8868\u793A\u4E2D":c}</span></button>`}).join("")}r(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((o,l)=>{const c=String(
o&&o.lang?o.lang:"text").trim()||"text";return`#${l+1} ${c}`}),a=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==a&&(e.sourceSelect.innerHTML=i.map((o,l)=>`<option value="${l}">${escapeHtml(
o)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=a),e.sourceSelect.value=String(
n)}r(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}r(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),a=String(e||""),o=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const l=new DOMParser().parseFromString(a,"text/html"),c=l.createElement("scr\
ipt");return c.setAttribute("data-canvas-scroll-bridge","true"),c.textContent=o,(l.body||l.documentElement).
appendChild(c),`<!DOCTYPE html>
`+l.documentElement.outerHTML}catch{return`${a}<script data-canvas-scroll-bridge>${o}<\/script>`}}r(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const t=e&&e.data?e.data:null;if(!t||t.type!=="canvas-preview-scroll")return;const n=getCanvasModeElements();
!n||!n.frame||e.source!==n.frame.contentWindow||t.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(t.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(t.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const t=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{t===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}r(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const t=getCanvasModeElements();if(t){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){t.panel.classList.add("hidden"),t.panel.classList.remove(
"canvas-panel-open","canvas-closing"),t.stage&&t.stage.classList.remove("canvas-enabled");return}t.panel.
classList.remove("canvas-panel-open"),t.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{t.panel.classList.add("hidden"),t.panel.classList.remove("canvas-closing"),t.
stage&&t.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}r(hideCanvasPreviewPanel,
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
en")),t.empty&&t.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}r(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const t=e||canvasPreviewState.lastCanvasData;
if(!t)return null;canvasPreviewState.lastCanvasData=t,canvasPreviewState.blocks=Array.isArray(t.blocks)?
t.blocks.slice():[],canvasPreviewState.rawText=String(t.rawText||""),canvasPreviewState.renderText=String(
t.renderText||"");const n=canvasPreviewState.blocks,i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!n.length){const l=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return l&&l.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=l.block.
key||"",l.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let a=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?a=i:canvasPreviewState.selectionMode="auto";const o=n[a]||null;return canvasPreviewState.
selectedIndex=o?a:-1,canvasPreviewState.selectedKey=o&&o.key?o.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),o}r(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,a=n&&Number.isInteger(
n.index)?n.index:-1,o=!!i,l=String(i&&i.lang?i.lang:"").trim(),c=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),u=o?isCanvasHtmlPreviewCandidate(l,c):!1,f=o?u?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",h=o?u?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=h),e.status&&(e.status.textContent=
f),e.previewLang&&(e.previewLang.textContent=o?l||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,v=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=v,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),o){canvasPreviewState.frameRenderToken+=1;const w=canvasPreviewState.frameRenderToken,
x=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),w);e.frame&&(e.frame.srcdoc=x,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{w!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:w,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}r(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const a=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
a&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}r(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(a=>a&&a.key===n);return i===-1?!1:applyCanvasSelection(i)}
r(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
a=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:a}}r(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const c=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[c],selectedIndex:0,selectedKey:c.key||t.codeKey||""}}const a=[];let o=-1;if(i.forEach(
(c,u)=>{const f=decodeCanvasPreviewButtonCode(c);if(!f)return;const h=normalizeCanvasBlock({lang:f.lang,
code:f.code,open:!1},u);a.push(h),o===-1&&f.codeKey===t.codeKey&&(o=a.length-1)}),!a.length)return null;
o===-1&&(o=0);const l=a[o]||a[0]||null;return{blocks:a,selectedIndex:o,selectedKey:l&&l.key?l.key:t.
codeKey||""}}r(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
var o;if((o=window.ActivityLog)==null||o.log("html_preview.start"),!e)return!1;const t=collectCanvasBlocksFromButton(
e);if(!t||!t.blocks||!t.blocks.length)return!1;const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.
blocks:[],i=n.findIndex(l=>l&&l.key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(
i);const a=t.blocks[t.selectedIndex]||t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.
rawText=a&&a.code!==void 0&&a.code!==null?String(a.code):"",canvasPreviewState.renderText=canvasPreviewState.
rawText,canvasPreviewState.selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.
selectedKey=t.selectedKey||a&&a.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),
canvasPreviewState.lastCanvasData={renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:a,
primaryIndex:canvasPreviewState.selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.
mobileView="preview",syncCanvasPanelViewUi("preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}r(previewCanvasCodeFromButton,
"previewCanvasCodeFromButton");function buildCanvasPreviewDocument(e){const t=String(e&&e.code!==void 0&&
e.code!==null?e.code:""),n=String(e&&e.lang?e.lang:"").trim().toLowerCase();if(isCanvasHtmlPreviewCandidate(
n,t))return sanitizeHtmlForPreview(t);const a=n?`Canvas Preview: ${n}`:"Canvas Preview",o=escapeHtml(
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
            </style></head><body><div class="frame"><div class="label">${escapeHtml(a)}</div><pre>${o||
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}r(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,t={}){const n=t.persist!==
!1;if(canvasModeEnabled=!!e,n)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const i=get("enable-canvas-mode");if(i&&i.checked!==canvasModeEnabled&&(i.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(t.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!t.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}r(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}r(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,t,n){let i=normalizeMarkdownNewlines(
e);const a=normalizeMarkdownNewlines(n);if(!a&&a!=="")return i;const o=t?[String(t),""]:[""];for(const l of[
"`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);for(const f of o){const h=`${u}${f}
`,y=`
${u}`,v=h+a+y;i.includes(v)&&(i=i.split(v).join(""))}}return i}r(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const o of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const u=l.repeat(c);
[`${o}${u}
${i}
${u}`,`${o}
${u}
${i}
${u}`,`
${o}${u}
${i}
${u}`,`
${o}
${u}
${i}
${u}`].forEach(h=>{n.includes(h)&&(n=n.split(h).join(`
`))})}return n}r(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}r(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}r(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[],i=[];if(!t)return{text:"",executions:n};const a=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let o=t.replace(a,(v,w,x,_)=>{i.push(_);const S=String(x||"").trim();try{const C=JSON.parse(S);n.push(
{code:C&&C.code!=null?String(C.code):"",output:C&&C.output!=null?String(C.output):"",images:C&&Array.
isArray(C.images)?C.images.filter(M=>typeof M=="string"):[]})}catch{n.push({code:S,output:""})}return`\

`});const l=/!\[Agentic View\]\(([^)]+)\)/g;if(!n.length&&/!\[Agentic View\]\(/.test(t)&&(o=t.replace(
/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,(v,w,x,_)=>(i.push(
_),n.push({code:w,output:x||"",images:[]}),`
`))),n.length){const v=new Set(n.flatMap(w=>w.images||[]));for(const w of t.matchAll(l)){if(v.has(w[1]))
continue;let x=i.length-1;for(;x>0&&i[x]>w.index;)x--;const _=n[x];_.images=Array.from(new Set([..._.
images||[],w[1]]))}o=o.replace(l,"")}n.forEach(v=>{v.code&&(o=stripExactFencedBlock(o,"python",v.code),
o=stripExactFencedBlock(o,"py",v.code)),o=stripVisiblePythonOutputBlock(o,v.output)});const c=new Set(
n.flatMap(v=>v.images||[]));o=o.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
(v,w,x,_)=>!w&&c.has(_)&&/\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(_)?`![${x}](${_})`:v);const u=n.
slice().reverse().find(v=>v.images&&v.images.length),f=o.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,
""),h=new Set(Array.from(f.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),v=>v[1])),y=(u?u.images:[]).filter(
v=>/^\/files\/[A-Za-z0-9_./%-]+$/.test(v)&&!h.has(v));return y.length&&(o+=`

**Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF**

`+y.map(v=>`![Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF](${v})`).join(`

`)),o=o.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:o,executions:n}}r(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(o=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(o)?n.push(o.trim()):
i.push(o)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}r(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}r(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",a=e&&e.output!=null?String(e.output):"";let o="";try{window.
hljs&&typeof window.hljs.highlight=="function"?o=window.hljs.highlight(i,{language:"python"}).value:
o=escapeHtml(i)}catch{o=escapeHtml(i)}const l=(e.images||[]).filter(_=>/^\/files\/[A-Za-z0-9_./%-]+$/.
test(_)).map(_=>`![Python output](${_})`).join(`
`),c=l?`<div class="python-section"><div class="python-label">Images</div>${sanitizeMarkdownHtml(marked.
parse(l))}</div>`:"",u=escapeHtml(a),f=encodeURIComponent(i).replace(/'/g,"%27"),h=encodeURIComponent(
a).replace(/'/g,"%27"),y=hashString(`pyexec-detail
${i}
${a}
${t}`),v=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",w=`<button class="download-btn" data-\
code="${f}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,x=`<button class="coding-target-btn" data-code="${f}" data-code-key="${y}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${y}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(v)}</span><div class="code-actions">${x}${w}<button class="copy-btn" data-copy="code" \
data-code="${f}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${h}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${o}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${u}</code></pre></div>${c}</div></div>`}r(buildPythonExecDetailBoxHtml,"bu\
ildPythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/pyt\
hon-execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"",
"/python-execution")}showModal("python-exec-modal")}r(showPythonExecDetailModal,"showPythonExecDetai\
lModal");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("p\
ython-exec-modal-body"),a=get("python-exec-modal-title");if(!n||!i)return;const o=t&&Array.isArray(t.
python_executions)?t.python_executions:[];if(!o.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(a){const l=o.length>1?`\uFF08${o.length}\u4EF6\uFF09`:"";a.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${l}`}i.innerHTML=o.map((l,c)=>buildPythonExecDetailBoxHtml(l,c,o.length)).join(
""),i.querySelectorAll("img").forEach(l=>{l.addEventListener("click",()=>openImageViewer(l.dataset.viewerSrc||
l.src,"#python-exec-modal-body img"))}),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(
!0,{persist:!1})),showPythonExecDetailModal(e)}r(openPythonExecDetail,"openPythonExecDetail"),window.
openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){get("python-exec-moda\
l")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&history.back())}r(
closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;function buildAiMarkdownHtml(e){
const t=extractMcpExecutionNotesFromContent(e),n=appendMcpExecutionNotes(t.text,t.notes),i=canvasModeEnabled?
parseCanvasMarkdown(n):{renderText:n,blocks:[],primaryBlock:null,rawText:n};canvasModeEnabled&&(updateCanvasPreviewState(
i),refreshCanvasPreviewPanel());const a=document.createElement("div");return a.className="prose pros\
e-invert text-sm break-words",a.innerHTML=sanitizeMarkdownHtml(i.renderText),wrapRenderedSvgBoxes(a),
lowBandwidthMode||(maybeNeedsHighlight(i.renderText,a)&&ensureHighlightLoaded().catch(()=>{}),maybeNeedsMathJax(
i.renderText)&&ensureMathJaxLoaded().catch(()=>{})),a.outerHTML}r(buildAiMarkdownHtml,"buildAiMarkdo\
wnHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes),o=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,
blocks:[],primaryBlock:null,rawText:a};if(canvasModeEnabled&&(updateCanvasPreviewState(o),refreshCanvasPreviewPanel()),
n.incrementalMath){const c=document.createElement("template");c.innerHTML=sanitizeMarkdownHtml(o.renderText,
{streamMathSegments:!0});const u=new Map;e.querySelectorAll(".stream-math-segment[data-stream-math-k\
ey]").forEach(h=>{const y=h.getAttribute("data-stream-math-key");y&&u.set(y,h)});const f=[];c.content.
querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(h=>{const y=u.get(h.getAttribute(
"data-stream-math-key"));y?h.replaceWith(y):f.push(h)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(
e)),e.replaceChildren(c.content),wrapRenderedSvgBoxes(e),queueHighlight(e,o.renderText),queueIncrementalMathTypeset(
f);return}const l=collectSvgCodeRenders(e);e.innerHTML=sanitizeMarkdownHtml(o.renderText),restoreSvgCodeRenders(
e,l),wrapRenderedSvgBoxes(e),queueMessageDecorations(e,o.renderText)}r(renderAiMarkdownInto,"renderA\
iMarkdownInto");function collectSvgCodeRenders(e){const t=new Map;return!e||typeof e.querySelectorAll!=
"function"||e.querySelectorAll(".svg-code-render[data-svg-key]").forEach(n=>{const i=n.getAttribute(
"data-svg-key");i&&!t.has(i)&&t.set(i,n)}),t}r(collectSvgCodeRenders,"collectSvgCodeRenders");function restoreSvgCodeRenders(e,t){
!e||!t||!t.size||typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-render[data-sv\
g-key]").forEach(n=>{const i=n.getAttribute("data-svg-key"),a=i?t.get(i):null;a&&(t.delete(i),n.replaceWith(
a))})}r(restoreSvgCodeRenders,"restoreSvgCodeRenders");function wrapRenderedSvgBoxes(e){!e||typeof e.
querySelectorAll!="function"||e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(
".svg-render-box")||t.closest("pre, code, .code-wrapper, .thought-container"))return;const n=document.
createElement("span");n.className="svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}
r(wrapRenderedSvgBoxes,"wrapRenderedSvgBoxes");const SEARCH_COMPLETE_HTML='<div class="search-box vi\
sible mb-2"><i class="fas fa-check-circle text-green-400"></i> Search complete</div>';function updateSearchBox(e,t,n){
return n!=="searching"&&n!=="done"||(t||(e.insertAdjacentHTML("afterbegin",SEARCH_COMPLETE_HTML),t=e.
querySelector(".search-box")),t.classList.toggle("animate-pulse",n==="searching"),t.innerHTML=n==="s\
earching"?'<i class="fas fa-globe"></i> Searching web...':'<i class="fas fa-check-circle text-green-\
400"></i> Search complete'),t}r(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,o,l=null,c=!0,u=null,f=null,h=null,y=null,v=null,w=null,x=null,_=null,S=!0,C=null,M=null,E=null,H=null){
const B=t==="user",te=B?"bg-blue-600":"bg-gray-700",j=B?"justify-end":"justify-start";messageStore[e]=
n;const V=!B&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},ue=B?n:V.text;let me=f;
if(me==null){const ne=h!=null?Number(h):0,ce=y!=null?Number(y):0;(h!=null||y!=null)&&(me=ne+ce)}messageMeta[e]=
{tokens_in:h,tokens_out:y,tokens_total:me,tokens_content:w,tokens_thought:x,is_encrypted:v,role:t,model:o,
parent_id:C,quote_text:u,image_url:i,image_mask:H,gem_name:M,batch_job:E,python_executions:B?[]:V.executions||
[]};let re="";u&&(re=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blue-400 text-xs te\
xt-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"></i>${escapeHtml(
u)}</div>`);let ke="";if(a&&!B){let ne="";try{const ce=JSON.parse(a);ne=ce.text||"",ce.search_used===
!0&&(ke=SEARCH_COMPLETE_HTML)}catch{ne=a}ne&&(ke+=`<div class="thought-container"><div class="though\
t-header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Proce\
ss</div><div class="thought-content collapsed">${escapeHtml(ne)}</div></div>`)}let I="";const O=new Set;
if(!B){const ne=String(ue||"").replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,""),ce=Array.
from(ne.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),Me=>Me[1]);ce.push(...(V.executions||[]).flatMap(Me=>Me.
images||[])),ce.forEach(Me=>{if(/^https?:\/\//i.test(Me)&&!Me.startsWith(FILE_BASE_URL))try{if(new URL(
Me).origin!==window.location.origin)return}catch{return}const we=normalizeAttachmentPath(Me);we&&O.add(
we)})}if(i)try{const ne=JSON.parse(i);if(ne.length){const ce=[];if(ne.forEach(Me=>{let we=Me,Ye="unk\
nown";if(we&&typeof we=="object"&&(Ye=normalizeAttachmentSource(we.source),we=we.filepath||we.path||
we.url||we.file||""),we=normalizeAttachmentPath(we)||we,!we||!B&&O.has(we))return;setAttachmentSourceForPath(
we,Ye);const Ie=we.replace(/^\d+\//,""),st=buildFileUrl(Ie),pt=buildAttachmentPreviewUrl(Ie),se=we.split(
"/").pop(),yt=se.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(yt)?ce.push(
buildChatImageHtml(pt,{viewerSrc:st,alt:se,title:se,filename:se})):ce.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${st}')" title="${se}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${se}</span></div>`)}),ce.length>
0){let Me="grid-multi";ce.length===1?Me="grid-1":ce.length===2?Me="grid-2":ce.length===3?Me="grid-3":
ce.length===4&&(Me="grid-4"),I=`<div class="image-grid ${Me}">${ce.join("")}</div>`}}}catch{}let K="";
if(B&&H)try{const ne=normalizeAttachmentPath(H)||String(H),ce=ne.replace(/^\d+\//,""),Me=buildFileUrl(
ce),we=buildAttachmentPreviewUrl(ce),Ye=ne.split("/").pop();K=`<div class="mt-2 max-w-xs rounded-lg \
border border-amber-300/40 bg-black/10 p-2"><div class="mb-1 text-[10px] font-semibold text-amber-10\
0"><i class="fas fa-mask mr-1" aria-hidden="true"></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(
we,{viewerSrc:Me,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:Ye})}</div>`}catch{}const R=B?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,G=`<div class="msg-controls absolute -top-5 right-\
0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', t\
his)"><i class="fas fa-copy"></i></button>${B?`<button class="ctrl-btn edit-btn" data-id="${e}"><i c\
lass="fas fa-pen"></i></button>`:""}${R}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\u65B0\
\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,Q=[];!B&&o&&Q.push(escapeHtml(o)),M&&(B?
Q.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(M)}</span>`):
Q.push(`<span class="text-purple-300/90"><i class="fas fa-gem mr-0.5"></i>${escapeHtml(M)}</span>`));
const P=[];if(h!=null&&P.push(`In ${h}`),y!=null){let ne=`Out ${y}`;x!=null&&Number(x)>0&&(ne+=` (Th\
ought ${x})`),P.push(ne)}if(P.length||f!=null){const ne=P.length?P.join(" / "):`${f} tokens`;Q.push(
`<button class="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDet\
ail('${e}')">${ne}</button>`)}if(v!=null){const ne=v?"fa-lock":"fa-lock-open",ce=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\
\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":
v?"Encrypted":"Plain",Me=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 ho\
ver:text-cyan-200":"text-slate-300/80 hover:text-white";Q.push(`<button class="${Me}" title="${ce}" \
onclick="openEncryptionSettings('${e}')"><i class="fas ${ne}"></i></button>`)}if(!B&&V.executions&&V.
executions.length){const ne=V.executions.length,ce=ne>1?`Python \xD7${ne}`:"Python";Q.push(`<button \
type="button" class="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" a\
ria-label="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${ce}\
</span></button>`)}const ee=Q.length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font\
-mono message-footer-meta">${Q.join(" \u2022 ")}</div>`:"";let de;const ye=!B&&E?(()=>{const ne=String(
E.state||"").toUpperCase(),ce=E.status_text||(ne==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
ne==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${ne==="JOB_STATE_FAILED"||
ne==="JOB_STATE_CANCELLED"||ne==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
ne==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(ce)}</div></div>`})():
"";B?de=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(de=ye+(ue&&String(ue).trim()?buildAiMarkdownHtml(ue):E?'<div class="content-area prose \
prose-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(ue)),de.includes("content-area")||(de=de.replace("prose ","content-area prose ")));
let Z="";if(l){const ne=l.siblings[l.current-2],ce=l.siblings[l.current];Z=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ne}\
)" ${ne?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ce}\
)" ${ce?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const pe=c?"fade-in":"",X=document.createElement("div");X.className=`flex ${j} mb-\
4 ${pe} relative message-group group`,X.id=`msg-${e}`,X.innerHTML=`<div class="message-bubble ${te} \
text-white p-4 rounded-2xl shadow-md relative">${G}${re}${ke}${de}${I}${K}${Z}${ee}</div>`;const Te=_||
get("chat-container");return Te&&(Te.appendChild(X),S&&scrollToBottom(),B||(X.isConnected&&queueMessageDecorations(
X,ue),syncCodingTargetButtons(X),syncCodingModeUi(codingModeEnabled,{persist:!1}))),X}r(renderMessage,
"renderMessage");function showTokenDetailModal(e=null,t=!1){if(location.pathname!=="/token-details"){
const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==null&&(n.messageId=e),history.pushState(
n,"","/token-details")}showModal("token-detail-modal")}r(showTokenDetailModal,"showTokenDetailModal");
function openTokenDetail(e){var y;const t=messageMeta[e];if(!t||!get("token-detail-modal"))return;get(
"token-detail-usage").classList.remove("hidden"),get("token-detail-estimate").classList.add("hidden"),
cancelPromptTokenDetail();const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:"-",
a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",o=t.tokens_out!==null&&t.tokens_out!==void 0?
t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.tokens_thought!==
null&&t.tokens_thought!==void 0?t.tokens_thought:"-",u=t.is_encrypted===null||t.is_encrypted===void 0?
"-":t.is_encrypted?"Encrypted":"Plain";get("token-detail-total").innerText=i,get("token-detail-in").
innerText=a,get("token-detail-out").innerText=o,get("token-detail-content").innerText=l,get("token-d\
etail-thought").innerText=c,get("token-detail-encrypted").innerText=u;const f=get("token-detail-cost");
if(f&&(f.replaceChildren(),f.classList.toggle("hidden",t.role!=="total"),t.role==="total")){for(const v of[
costEstimateLabel(t.cost_estimate),`\u96C6\u8A08\u3057\u305FAPI\u547C\u3073\u51FA\u3057: ${((y=t.cost_estimate)==
null?void 0:y.requests)||0}\u56DE`,COST_ESTIMATE_NOTE,`\u516C\u5F0F\u6599\u91D1\u78BA\u8A8D\u65E5: ${(costPricing==
null?void 0:costPricing.checked_at)||"\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"}`]){
const w=document.createElement("p");w.textContent=v,f.appendChild(w)}for(const[v,w]of Object.entries(
(costPricing==null?void 0:costPricing.sources)||{})){const x=document.createElement("a");x.href=w,x.
textContent=`${v} \u516C\u5F0F\u6599\u91D1 `,x.target="_blank",x.rel="noopener noreferrer",x.className=
"underline",f.appendChild(x)}}const h=t.model?`${t.model} (${t.role})`:`${t.role}`;get("token-detail\
-title").innerText=h,showTokenDetailModal(e)}r(openTokenDetail,"openTokenDetail");function closeTokenDetail(e=!1){
cancelPromptTokenDetail(),get("token-detail-modal")&&(hideModal("token-detail-modal"),!e&&location.pathname===
"/token-details"&&history.back())}r(closeTokenDetail,"closeTokenDetail");function openEncryptionSettings(e){
const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}r(openEncryptionSettings,"openEncrypti\
onSettings");function openEncryptionModal(e){if(!get("encryption-status-modal"))return;const n=get("\
encryption-status-title"),i=get("encryption-status-body"),a=get("encryption-status-admin-actions"),o=get(
"encryption-status-admin-toggle"),l=!!e;l?(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),
i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
a&&o&&(!!(isAdminUser&&currentThreadId)?(a.classList.remove("hidden"),o.dataset.enable=l?"0":"1",o.disabled=
!1,o.textContent=l?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
o.className=l?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):a.classList.add("hidden")),showEncryptionStatusModal()}r(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}r(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const t=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${t?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const i=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,t,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=i}}r(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}r(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}r(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}r(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=r((e,t,n,i)=>{
const a=(e||"").toLowerCase();return!a.includes("gemini")||a.includes("image")||a.includes("nano")||
a.includes("tts")||a.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=r(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),a=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(o=>{let l=!1;function c(){n&&n.removeEventListener("cli\
ck",f),i&&i.removeEventListener("click",h),a&&a.removeEventListener("click",h),e.removeEventListener(
"click",y,!0)}r(c,"cleanup");function u(v){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),o(v)}r(u,"finalize");
function f(){u(!0)}r(f,"onOk");function h(){u(!1)}r(h,"onCancel");function y(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),h())}r(y,"onOverlay"),n&&n.addEventListener("click",f),i&&i.addEventListener(
"click",h),a&&a.addEventListener("click",h),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const o=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),u=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${o}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,f=e||get("\
chat-container");if(f){if(typeof f.insertAdjacentHTML=="function")f.insertAdjacentHTML("beforeend",u);else{
const h=document.createElement("div");h.innerHTML=u;const y=h.firstElementChild;y&&f.appendChild(y)}
n&&scrollToBottom()}}r(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){
if(!e||e.getAttribute("data-stream-transition")==="1")return;const t=e.querySelector(".content-area");
t&&(t.classList.remove("pending-shimmer","skeleton-pending"),t.removeAttribute("data-skeleton-kind")),
e.setAttribute("data-stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add(
"ai-stream-transition"),t&&(t.classList.add("ai-stream-content-transition"),setTimeout(()=>{t&&t.classList.
remove("ai-stream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-trans\
ition")},320)}r(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){
return e==null||e===""?null:String(e)}r(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}r(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const t=Array.from(e.querySelectorAll(".prose")).some(c=>String(c.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(c=>!!String(c.textContent||
"").trim()&&c.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const a=e.parentElement;
if(!a)return null;const o=a.cloneNode(!0);o.setAttribute("data-local-stopped-partial","1"),o.classList.
remove("fade-in");const l=o.querySelector(".message-bubble");if(l&&(l.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),l.removeAttribute("data-stream-transition"),l.removeAttribute("id"),!o.
querySelector('[data-stopped-partial-note="1"]'))){const c=document.createElement("div");c.setAttribute(
"data-stopped-partial-note","1"),c.className="text-[10px] text-amber-200/90 mt-2 text-right",c.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",l.appendChild(c)}return{html:o.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}r(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const a=get("chat-\
container");return a?(a.querySelectorAll('[data-local-stopped-partial="1"]').forEach(o=>o.remove()),
a.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}r(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}r(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}r(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}r(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var c,u;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((u=t.retryDelayMs)!=
null?u:180)||0),a=!!t.notifyOnFailure,o=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!l||o&&l!==o)return!1;for(let f=0;f<=n;f++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==l?!1:(await loadMessages(l,{preserveDraft:!0,silent:!0}),
!0)}catch{f<n&&i>0&&await new Promise(y=>setTimeout(y,i))}return a&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===l&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}r(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(t){console.warn("Vi\
bration failed:",t)}}r(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const t=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(n=>n.kind==="minimal"&&!minimalPromptMode?!1:n.label.
toLowerCase().includes(t)||n.description.toLowerCase().includes(t))}r(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,t){if(String(e||"").toLowerCase()!=="thinking")
return e;const i=String(t||"").trimStart().match(/^\/thinking(\s+.*)$/i);return i?`thinking${i[1]}`.
toLowerCase():e}r(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const t=String(e||"").trim().toLowerCase();if(!t||t==="toggle"||t==="\u5207\u66FF"||t==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(t))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(t))return!1}r(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(o=>o.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(o=>o.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const a=String(n.presetArgument||t||"").trim();if(i.selectId){if(!a)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} default / none`,"info"),!1;const o=get(i.selectId),l=a.toLowerCase(),
c=o?Array.from(o.options).find(u=>u.value.toLowerCase()===l||u.textContent.trim().toLowerCase()===l):
null;return!o||!c?(showToast(`${n.label}: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1):(o.value=c.value,o.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),
showToast(`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const o=a.
toLowerCase(),l=activeThinkingSelect(),c=!!(l&&l.id==="reasoning-effort"),u=c?{off:"none",none:"none",
min:"low",minimal:"low",low:"low",mid:"medium",medium:"medium",high:"high",xhigh:"xhigh",max:"max"}:
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},f=parseSlashToggleArgument(
a),h=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(u,o)){const y=u[o],v=l?Array.from(l.options).
find(w=>w.value===y&&!w.disabled&&!w.classList.contains("hidden")):null;if(l&&!v)return showToast(`T\
hinking: \u300C${o}\u300D\u306F\u3053\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1;if(y==="none"){if(h&&h.disabled)return showToast("Thinking: \u3053\u306E\u30E2\u30C7\u30EB\u3067\u306FOFF\u306B\u3067\u304D\u307E\u305B\u3093",
"warning"),!1;h&&h.checked?(h.checked=!1,h.dispatchEvent(new Event("change",{bubbles:!0}))):l&&(l.value=
"none",l.dispatchEvent(new Event("change",{bubbles:!0})))}else h&&!h.checked&&!h.disabled&&(h.checked=
!0,h.dispatchEvent(new Event("change",{bubbles:!0}))),l&&(l.value=y,l.dispatchEvent(new Event("chang\
e",{bubbles:!0})));return refreshMinimalOptionItems(),showToast(`Thinking: ${o}`,"success"),!0}if(f===
void 0)return showToast(c?"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high / xhigh / max":
"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high","info"),!1}if(i.checkboxId&&a){const o=parseSlashToggleArgument(
a);if(o===void 0)return showToast(`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const l=get(i.
checkboxId);if(o!==null&&l&&l.checked===o)return showToast(`${i.label}: ${o?"ON":"OFF"}`,"info"),!0}
return handleMinimalOptionClick(i),!0}r(executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){
const t=String(e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||
"",i=n.match(/^[a-z][\w-]*/i);return i?i[0]:n}r(extractSlashCommandToken,"extractSlashCommandToken");
function hideSlashCommandSuggestions(){const e=get("slash-command-suggestions");e&&e.classList.add("\
hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=0}r(hideSlashCommandSuggestions,"hideSlashCom\
mandSuggestions");function showPendingSlashCommandIndicator(e){const t=get("slash-command-indicator"),
n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.find(o=>o.id===e);n.textContent=
i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");const a=get("prompt-input");a&&
i&&(a.dataset.originalPlaceholder=a.placeholder,a.placeholder=i.argumentHint||"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\
\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}r(showPendingSlashCommandIndicator,"showPe\
ndingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){const e=get("slash-command-\
indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get("prompt-input");t&&
t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.dataset.originalPlaceholder);
const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}r(
hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const a=visibleSlashCommands(e);if(a.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((v,w)=>{const x=document.createElement(
"div");x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===
slashSelectedIndex?"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas ${v.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${v.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${v.description}</div>
                    </div>
                `;let _=!1;x.addEventListener("pointerdown",S=>{typeof S.button=="number"&&S.button!==
0||(S.preventDefault(),_=!0,selectSlashCommand(v.id))}),x.addEventListener("click",S=>{S.preventDefault(),
_||selectSlashCommand(v.id)}),x.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(x)});const o=i.getBoundingClientRect(),l=window.innerHeight,c=l-o.bottom,u=o.top,f=260,
h=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,o.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&u>c){const v=Math.min(f,u-h);t.style.top="auto",t.style.bottom=`${l-o.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(f,c-h);t.style.top=`${o.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${v}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}r(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const l=String(n||"").trimStart();t.value=
l.substring(1+i.length).trimStart()}else{const l=n.lastIndexOf("/");l!==-1?t.value=n.substring(0,l).
trimEnd():t.value=""}hideSlashCommandSuggestions();const a=SLASH_COMMANDS.find(l=>l.id===e),o=t.value.
trim();if(a&&a.autocompleteArgument&&!o){t.value=`${a.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(a&&a.kind==="minimal"&&(!a.
requiresArgument||o)){t.value="",executeMinimalSlashCommand(e,o),t.dispatchEvent(new Event("input",{
bubbles:!0})),t.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),t.focus(),t.
dispatchEvent(new Event("input",{bubbles:!0}))}r(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
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
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}r(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,t){
const n=get(`tab-${e}`);let i=get(t);if(!n||!i)return null;for(;i.parentElement&&i.parentElement!==n;)
i=i.parentElement;return i.parentElement===n?i:get(t)}r(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const t=AI_SETTING_JUMP_TARGETS[e];if(!t){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(t.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const n=get(t.control);n&&(n.scrollIntoView({behavior:"smooth",block:"center"}),n.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const n=findSettingsJumpElement(t.tab,t.control);n?jumpToSetting(t.tab,n):switchTab(t.tab||"general")},
320)}r(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const t=e.querySelector(".msg-controls");t&&t.remove()}r(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,t,n="update"){const i=Object.
entries(e||{}),a=`settings-result-${Date.now()}`,o=n==="inspect",l=i.length?o?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
o?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
c=renderMessage(a,"assistant",l,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!c)
return;removeEphemeralMessageControls(c);const u=c.querySelector(".message-bubble");if(!u||!i.length)
return;const f=document.createElement("div");f.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,v])=>{const w=AI_SETTING_JUMP_TARGETS[y]||{label:y},x=document.createElement("button");x.
type="button",x.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const _=document.createElement("span");_.className="min-w-0 flex-1";const S=document.createElement(
"span");S.className="block text-xs font-bold text-blue-200",S.textContent=w.label;const C=document.createElement(
"span");C.className="block mt-0.5 text-[11px] text-gray-300 break-words",C.textContent=formatAiSettingValue(
v);const M=document.createElement("i");M.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",_.appendChild(S),_.appendChild(C),x.appendChild(_),x.appendChild(M),x.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),f.appendChild(x)});const h=u.querySelector(".message-footer-\
meta");h?u.insertBefore(f,h):u.appendChild(f),scrollToBottom()}r(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const o=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${o}" class="flex justify-start mb-4 ai-pending-row f\
ade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-\
tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const u=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),f=get(o);if(f&&f.remove(),u&&u.status==="ok"&&u.mode==="inspect"&&u.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(u.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
u.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(u.current,t,"inspect");
return}if(u&&u.status==="ok"&&u.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
u.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(u.applied).length}\u9805\u76EE\uFF09`,"success");try{const v=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(v),cacheUserSettings(v)}catch{}renderAiSettingsResultBubble(
u.applied,t);return}const h=u.message||u.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${h}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${h}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(h,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const u=get(o);u&&u.remove();const f=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(f),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
r(runAiSettingsCommand,"runAiSettingsCommand");function hideGemSuggestions(){const e=get("gem-sugges\
tions");e&&e.classList.add("hidden"),gemSuggestionsVisible=!1,gemSelectedIndex=0}r(hideGemSuggestions,
"hideGemSuggestions");function showGemSuggestions(e=""){const t=get("gem-suggestions"),n=get("gem-su\
ggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedGems||loadedGems.length===0){hideGemSuggestions();
return}const a=e.toLowerCase(),o=loadedGems.filter(w=>w.name.toLowerCase().includes(a)||w.description&&
w.description.toLowerCase().includes(a));if(o.length===0){hideGemSuggestions();return}gemSelectedIndex>=
o.length&&(gemSelectedIndex=0),n.innerHTML="",o.forEach((w,x)=>{const _=document.createElement("div");
_.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${x===gemSelectedIndex?
"bg-gray-700":""}`,_.innerHTML=`
                    <i class="fas fa-gem w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(w.name)}</div>
                        ${w.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
w.description)}</div>`:""}
                    </div>
                `,_.onclick=()=>selectGemSuggestion(w),_.onmouseenter=()=>{gemSelectedIndex=x,showGemSuggestions(
e)},n.appendChild(_)});const l=i.getBoundingClientRect(),c=window.innerHeight,u=c-l.bottom,f=l.top,h=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",u<180&&f>u){const w=Math.min(h,f-y);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(h,u-y);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),gemSuggestionsVisible=!0}r(showGemSuggestions,
"showGemSuggestions");function selectGemSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideGemSuggestions(),
activateGem(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}r(selectGemSuggestion,"sel\
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
""}r(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const a=String(i.result||""),o=a.indexOf(",");if(o<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
a.slice(o+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}r(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const a=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(o=>{o&&
a.push({thoughtSignature:String(o)})}),i.text&&a.push({text:String(i.text)});for(const o of Array.isArray(
i.images)?i.images:[])try{const l=await fetch(buildFileUrl(o.path),{credentials:"same-origin",cache:"\
no-store"});if(!l.ok)throw new Error(`HTTP ${l.status}`);const c=await l.blob();a.push({inlineData:{
mimeType:o.mime_type||c.type||"application/octet-stream",data:await fileToBase64Payload(c)}})}catch{
n++}a.length&&t.push({role:i.role,parts:a})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}r(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(e=browserFastLocalFiles,t=!1){
const n=Array.from(e.entries()),i=[];for(const[a,o]of n){if(!o||!o.file||!o.rowObj)throw new Error("\
\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");if(o.
rowObj.status&&(o.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(o.file,o.rowObj))throw new Error(`${o.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);const c=o.rowObj.row.getAttribute("da\
ta-filename");c&&i.push(c),t?currentImageUrls=currentImageUrls.filter(u=>u!==c):browserFastLocalFiles.
delete(a)}return i}r(uploadBrowserFastLocalFiles,"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){
const t=get("enable-thinking");if(!t||!t.checked)return null;const n=String(get("thinking-level")?get(
"thinking-level").value:"high").toLowerCase();if(e.includes("2.5")){const a=Number(get("thinking-bud\
get")?get("thinking-budget").value:4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(a)?
Math.max(0,Math.min(32768,Math.trunc(a))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["M\
EDIUM","HIGH"].includes(i)&&(i="MEDIUM"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&
(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}r(browserFastThinkingConfig,"browserFastThinkingC\
onfig");function browserFastPythonBoxHtml(e){return`<div class="code-wrapper python-box collapsed" d\
ata-py-id="${e}" data-collapsed="true" data-code-key="${e}"><div class="code-header"><span class="co\
de-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button cl\
ass="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></\
i></button><button class="copy-btn" data-copy="code" data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\
\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" titl\
e="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></button></div></div><div class="cod\
e-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs langu\
age-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Out\
put</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`}r(
browserFastPythonBoxHtml,"browserFastPythonBoxHtml");function updateBrowserFastPythonBox(e,t,n){if(e){
if(t==="code"){const i=n==null?"":String(n),a=e.querySelector(".python-code");a&&(a.textContent=i,a.
removeAttribute("data-highlighted"),queueHighlight(e,i));const o=e.querySelector('.copy-btn[data-cop\
y="code"]');o&&o.setAttribute("data-code",encodeURIComponent(i).replace(/'/g,"%27"))}else if(t==="ou\
tput"){const i=n==null?"":String(n),a=e.querySelector(".python-output");a&&(a.textContent=i);const o=e.
querySelector('.copy-btn[data-copy="output"]');o&&o.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}}}r(updateBrowserFastPythonBox,"updateBrowserFastPythonBox");function collectBrowserFastPromptOptions(){
const e=r((n,i)=>get(n)?!!get(n).checked:i,"checked"),t=r(n=>get(n)?get(n).value:null,"value");return{
canvas_mode:!!canvasModeEnabled,enable_search:e("enable-search",!1),enable_url_context:e("enable-url\
-context",!1),enable_maps:e("enable-maps",!1),enable_python:e("enable-python",!1),enable_file_creation:e(
"enable-file-creation",!0),enable_mcp:typeof isMcpEnabledForSend=="function"?!!isMcpEnabledForSend():
!1,enable_thinking:e("enable-thinking",!1),thinking_level:t("thinking-level"),thinking_budget:t("thi\
nking-budget"),reasoning_effort:t("reasoning-effort"),safety_setting:t("safety-setting"),enable_prompt_caching:e(
"enable-prompt-cache",!1)}}r(collectBrowserFastPromptOptions,"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e,t){
var R,G,Q;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),o=[];for(const P of a)o.push({inlineData:{
mimeType:P.file.type,data:await fileToBase64Payload(P.file)}});o.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const u={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:o}],generationConfig:l};!!(get("enable-python")&&get("enable-python").
checked)&&(u.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&
(promptHistory.unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt=
"",playSendAnimation(),get("welcome-screen").classList.add("hidden");const h=userAutoScroll&&isChatNearBottom();
h||pauseChatAutoScroll();const y=collectImageUrlsForSend(),v=Array.from(((R=get("file-preview-thumbs"))==
null?void 0:R.querySelectorAll("img"))||[]).map(P=>P.cloneNode(!0)),w=renderMessage(Date.now(),"user",
e,JSON.stringify(collectImageUrlsForSend()),null,null,null,!1,currentQuote,null,null,null,null,null,
null,null,h);if(v.length&&!y.length){const P=document.createElement("div");P.className="flex gap-2 f\
lex-wrap",v.forEach(ee=>{ee.removeAttribute("onclick"),P.appendChild(ee)}),w.querySelector(".message\
-bubble").appendChild(P)}const x=takePromptForSend(),_=animatePromptToMessage(w,x,h),S=`browser-fast\
-${Date.now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start\
 mb-4 ai-pending-row fade-in"><div id="${S}" class="message-bubble ai-pending-bubble bg-gray-700 tex\
t-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(n,"Gemini\u3078\u76F4\u63A5\u9001\
\u4FE1\u4E2D...")}</div></div>`);const C=get(S),M=startPendingElapsed(C,t);activeStreamingBubbleId=S,
setSendBtnToStopMode(),h&&resumeChatAutoScroll(),abortController=new AbortController;let E="",H="",B=null;
const te=[];let j=null,V=null,ue=!1;const me={},re=[];let ke=null,I="";const O=window.ProgressSpinner?
window.ProgressSpinner.startFlow("browserFast"):null;let K=!1;try{const P=await fetch(`https://gener\
ativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json","x-goog-api-key":browserFastApiKey},body:JSON.
stringify(u),signal:abortController.signal}));if(!P.ok){const we=await P.json().catch(()=>({}));throw new Error(
we&&we.error&&we.error.message?we.error.message:`Gemini API HTTP ${P.status}`)}window.ConnectionMonitor&&
(K=!0,window.ConnectionMonitor.operationStarted()),O&&O.setPhase("waiting"),x.accept(!1),(G=window.ActivityLog)==
null||G.log("send.stream.start");let ee=!1;const de=P.body.getReader(),ye=new TextDecoder;let Z="";const pe=r(
we=>{var pt;const Ye=we.split(/\r?\n/).filter(se=>se.startsWith("data:")).map(se=>se.slice(5).trim()).
join("");if(!Ye||Ye==="[DONE]")return;const Ie=JSON.parse(Ye);if(Number.isInteger((pt=Ie.usageMetadata)==
null?void 0:pt.promptTokenCount)){B={};for(const se of["promptTokenCount","candidatesTokenCount","th\
oughtsTokenCount"])Number.isInteger(Ie.usageMetadata[se])&&Ie.usageMetadata[se]>=0&&(B[se]=Ie.usageMetadata[se])}
if(Ie.error)throw new Error(Ie.error.message||"Gemini API error");if((Array.isArray(Ie.candidates)?Ie.
candidates:[]).forEach(se=>{(se&&se.content&&Array.isArray(se.content.parts)?se.content.parts:[]).forEach(
He=>{if(He&&typeof He.thoughtSignature=="string"&&!te.includes(He.thoughtSignature)&&te.push(He.thoughtSignature),
He&&He.executableCode&&typeof He.executableCode.code=="string"){const ut=He.executableCode.code;E+=`\

\`\`\`python
${ut}
\`\`\`
`,ke=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,I=ut,me[ke]||(C.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(ke)),me[ke]=C.querySelector(`[data-py-id="${ke}"]`)),updateBrowserFastPythonBox(
me[ke],"code",ut);return}if(He&&He.codeExecutionResult&&typeof He.codeExecutionResult.output=="strin\
g"){const ut=He.codeExecutionResult.output;E+=`
**Output:**
\`\`\`
${ut}
\`\`\`
`;const vt=ke||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;re.push({code:I||
"",output:ut}),me[vt]||(C.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(vt)),me[vt]=C.querySelector(
`[data-py-id="${vt}"]`)),updateBrowserFastPythonBox(me[vt],"output",ut);return}const Tt=typeof He.text==
"string"?He.text:"";Tt&&(He.thought===!0?H+=Tt:E+=Tt)})}),!ue&&(E||H)){beginPendingToStreamTransition(
C);const se=C.querySelector(".content-area");se&&se.remove(),ue=!0}H&&(V||(C.insertAdjacentHTML("aft\
erbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)">\
<i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></di\
v></div>'),V=C.querySelector(".thought-content")),V.textContent=H),E&&(j||(j=document.createElement(
"div"),j.className="content-area prose prose-invert text-sm break-words",C.appendChild(j)),renderAiMarkdownInto(
j,E,{incrementalMath:!0})),scrollToBottom()},"consumeEvent");for(;;){const{done:we,value:Ye}=await de.
read();if(!we&&!ee&&(ee=!0,(Q=window.ActivityLog)==null||Q.log("send.stream.first")),we)break;window.
ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),O&&O.setPhase("receiving"),Z+=ye.decode(
Ye,{stream:!0});const Ie=Z.split(/\r?\n\r?\n/);Z=Ie.pop()||"",Ie.forEach(pe)}if(Z+=ye.decode(),Z.trim()&&
pe(Z),!E.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
j&&renderAiMarkdownInto(j,E,{incrementalMath:!0}),V&&V.classList.add("collapsed"),re.length&&(E+=re.
map(we=>`
\`\`\`pyexec
${JSON.stringify(we)}
\`\`\`
`).join(""));let X=[];a.length&&(O&&O.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),X=await uploadBrowserFastLocalFiles(x.files,!0)),O&&O.setPhase("saving");const Te=[...y,...X],
ne=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:E,pricing_usage:B,thought_content:H,model:n,image_urls:Te,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:te,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),C),ce=await ne.json().
catch(()=>({}));if(!ne.ok||!ce.thread_id)throw new Error(ce.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
const Me=!currentThreadId;currentThreadId=String(ce.thread_id),currentParentId=ce.assistant_message_id||
null,currentLeafId=ce.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!Me}),applyBrowserFastModeRestrictions(),loadThreads(!1),showToast(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(P){if(_(),x.restore(),P.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${P.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const ee=P.message||"\
\u30A8\u30E9\u30FC";C&&C.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(ee));try{let de=E||
"";re.length&&(de+=re.map(Te=>`
\`\`\`pyexec
${JSON.stringify(Te)}
\`\`\`
`).join(""));const ye=buildChatErrorMarkdown(ee,de),Z=a.length?[]:y,pe=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:ye,
pricing_usage:B,thought_content:H||"",model:n,image_urls:Z,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:
null,thought_signatures:te,turnstile_token:botTurnstileTokenForRequest()}),signal:abortController&&!abortController.
signal.aborted?abortController.signal:void 0}),C),X=await pe.json().catch(()=>({}));if(pe.ok&&X.thread_id){
const Te=!currentThreadId;currentThreadId=String(X.thread_id),currentParentId=X.assistant_message_id||
null,currentLeafId=X.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!Te}),applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(de){
sendClientDebugLog("error",`Browser fast error persist failed: ${de&&de.message?de.message:de}`)}}}finally{
x.releaseFiles(),M&&M(),K&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),O&&O(),
setSendBtnToSendMode(),activeStreamingBubbleId===S&&(activeStreamingBubbleId=null),abortController=null,
updateFilePreview()}}r(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
r(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var Dt,on,Kt,_t,rn;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const W=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(W);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const W=pendingSlashCommand,
xe=a.trim(),Ae=get("model-select")?get("model-select").value:null;if(W==="settings"){if(!xe){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!Ae){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
xe,Ae)}else executeMinimalSlashCommand(W,xe)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const o=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(o&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(W=>W.id===
o[1].toLowerCase())){executeMinimalSlashCommand(o[1].toLowerCase(),o[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const l=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(l&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(l)setBrowserFastModeEnabled(!1);else{const W=browserFastModeIneligibility(
a);if(!W){try{await sendBrowserFastMessage(a,t)}catch(xe){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${xe.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${W}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(xe){showToast(
xe.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),u=c.map(W=>W.path),f=c.filter(W=>normalizeAttachmentSource(
W.source)==="upload").map(W=>W.path),h=getModelMediaSupport(get("model-select").value),y=u.some(W=>isAudioPath(
W)),v=u.some(W=>isVideoPath(W)),w=(get("model-select").value||"").toLowerCase(),x=get("enable-python"),
_=!!(x&&x.checked);if(y&&!h.audio||v&&!h.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&u.length===0)return;if(isMistralOcrModel(
w)){const W=/https?:\/\/\S+/i.test(a);if(u.filter(Ae=>isAudioPath(Ae)||isVideoPath(Ae)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!u.length&&!W){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const S=a.trim();if(/^\/settings(?:\s|$)/i.
test(S)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(S)){const W=S.replace(/^\/settings\s*/i,"").trim();
if(!W){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Ae=get("prompt-input");Ae.value="/settings ";const Fe=extractSlashCommandToken(Ae.value);
lastSlashFilter=Fe,showSlashCommandSuggestions(Fe),Ae.focus();return}const xe=get("model-select")?get(
"model-select").value:null;if(!xe){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
W,xe);return}if(isGeminiLocalPythonMode(w,y,v,_)&&!await confirmGeminiLocalPythonSwitch())return;let C=null,
M=[];if(codingModeEnabled){const W=collectCodingCandidates(a),xe=W.filter(Qe=>Qe.prompt_source),Ae=W.
filter(Qe=>!Qe.prompt_source),Fe=xe.reduce((Qe,Lt)=>Qe+String(Lt.code||"").length,0);if(Fe>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let at=3e5-Fe;const ft=[];for(let Qe=Ae.length-1;Qe>=0;Qe--){const Lt=String(Ae[Qe].
code||"").length;Lt>at||(ft.unshift(Ae[Qe]),at-=Lt)}M=codingTargetSelection?ft.slice(-1):[...xe,...ft];
const St=xe.length?xe[xe.length-1]:null;if(C=codingTargetSelection?M[0]:St||M[M.length-1]||null,codingModeEffective=
!!(C&&String(C.code||"").trim()),codingModeEffective&&C.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const Qe=String(((Dt=get(
"model-select"))==null?void 0:Dt.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(Qe)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const E=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${u.length} search=${get("enable-search").checked}`);const H=a,B=hasMarkerHint()?MARKER_HINT_TEXT:
null,te=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&u.length===0){showToast("M\
ask \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if(te&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const j=userAutoScroll&&isChatNearBottom();j||pauseChatAutoScroll();const V=editingMessageId,
ue=promptSettingsBeforeEdit;promptSettingsBeforeEdit=null;const me=currentParentId,re=V!=null;V&&(editingMessageId=
null,setEditUi(!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const ke=[],I=r(
W=>{if(W==null)return;let xe=document.getElementById(`msg-${W}`);for(;xe;)xe.classList&&xe.classList.
contains("message-group")&&(ke.push({node:xe,prevDisplay:xe.style.display}),xe.style.display="none"),
xe=xe.nextElementSibling},"hideRenderedBranchFrom"),O=r(()=>{ke.forEach(({node:W,prevDisplay:xe})=>{
W&&(W.style.display=xe||"")}),ke.length=0},"restoreHiddenBranch");V&&I(V);const K=Date.now(),R=isGptImageModel()||
te?currentMaskImage:null,G=renderMessage(K,"user",H,JSON.stringify(u),null,null,null,!1,currentQuote,
null,null,null,null,null,null,null,j,me,activeGem?activeGem.name:null,null,R);let Q=!1;const P=/(https?:\/\/)?(x\.com|twitter\.com)\//i,
ee=!i&&(P.test(H||"")||P.test(currentQuote||"")),de="grok-4-fast-reasoning",ye=r(()=>{get("enable-se\
arch").checked=!0,get("model-select").value!==de&&selectModelById(de)},"applyXLinkAuto");if(ee&&!isMistralOcrModel()&&
!get("enable-search").checked)if(autoSearchOnLinks)ye();else{const W=get("auto-search-banner"),xe=get(
"auto-search-on-btn"),Ae=get("auto-search-off-btn"),Fe=get("auto-search-remember");W&&xe&&Ae&&(Fe&&(Fe.
checked=!1),await new Promise(at=>{W.classList.remove("hidden");const ft=r(St=>{W.classList.add("hid\
den"),xe.onclick=null,Ae.onclick=null,at(St)},"cleanup");xe.onclick=()=>ft("enable"),Ae.onclick=()=>ft(
"disable")}).then(async at=>{at==="enable"?(ye(),Fe&&Fe.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):Q=!0}))}const Z=String(get("reasoning-effort").value||"").toLowerCase(),
pe=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&Z==="none",X={client_request_id:createClientRequestId(),
thread_id:currentThreadId,message:H,model:get("model-select").value,image_urls:u,image_items:c,uploaded_image_urls:f,
temporary_chat:temporaryChatEnabled,enable_search:get("enable-search").checked,enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_file_creation:get(
"enable-file-creation")?get("enable-file-creation").checked:!0,enable_thinking:pe?!1:get("enable-thi\
nking").checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get(
"thinking-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get(
"enable-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cac\
he").checked:!1,marker_system_prompt:B,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
get("tts-voice")?get("tts-voice").value:null,tts_voice_custom:isTtsModel()&&get("tts-voice-custom")?
get("tts-voice-custom").value:null,tts_language:isTtsModel()&&get("tts-language")?get("tts-language").
value:null,tts_style:isTtsModel()&&get("tts-style")&&get("tts-style-wrap")&&!get("tts-style-wrap").classList.
contains("hidden")?get("tts-style").value:null,tts_speed:isTtsModel()&&get("tts-speed")?get("tts-spe\
ed").value:null,image_size:isGptImageModel()&&get("gpt-image-size")?get("gpt-image-size").value:null,
image_quality:isGptImageModel()&&get("gpt-image-quality")?get("gpt-image-quality").value:null,image_format:isGptImageModel()&&
get("gpt-image-format")?get("gpt-image-format").value:null,image_compression:isGptImageModel()&&get(
"gpt-image-compression")?get("gpt-image-compression").value:null,image_mask:R,gemini_image_aspect:isGeminiImageModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:me,parent_id_explicit:re,disable_auto_search:Q,
image_vision_model:currentVisionModel||null,coding_mode:E,coding_target:E?{id:C.candidate_id,code:C.
prompt_source?null:C.code,language:C.language||"text",key:C.key||null,message_id:C.message_id||null,
source:C.prompt_source?"prompt":"history",explicit:C.explicit===!0}:null,coding_candidates:E?M.map(W=>({
id:W.candidate_id,source:W.prompt_source?"prompt":"history",prompt_index:W.prompt_source?W.prompt_index:
null,code:W.prompt_source?null:W.code,language:W.language||"text",explicit:W.explicit===!0})):[],batch_mode:l,
canvas_mode:!!canvasModeEnabled};n&&(X.turnstile_token=n);const Te=get("thread-custom-instruction");
Te&&(X.thread_custom_instruction=Te.value||""),activeGem?(X.system_prompt=activeGem.instruction,X.enable_system_prompt=
!0,X.gem_uuid=activeGem.uuid):X.gem_uuid=null;const ne=takePromptForSend(e.animate!==!1),ce=animatePromptToMessage(
G,ne,j&&e.animate!==!1);setSendBtnToStopMode();const Me="ai-"+Date.now(),we=String(X.model||"").toLowerCase(),
Ye=!!X.enable_thinking||!!Z&&Z!=="none",Ie=we.includes("gemini")||we.includes("o1")||we.includes("o3")||
we.includes("gpt-5")||we.includes("reasoning")&&!we.includes("non-reasoning"),st=Ye&&Ie;let pt=buildPendingSkeletonHtml(
X.model,"API\u306B\u9001\u4FE1\u4E2D...");get("chat-container").insertAdjacentHTML("beforeend",`<div\
 class="flex justify-start mb-4 ai-pending-row fade-in"><div id="${Me}" class="message-bubble ai-pen\
ding-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${pt}</div></\
div>`),j&&resumeChatAutoScroll();const se=get(Me),yt=l?null:startPendingElapsed(se,t);activeStreamingBubbleId=
Me,canvasModeEnabled&&resetCanvasPreviewPanel();let He=null;const Tt=r(W=>!st||!se?null:((!He||!se.contains(
He))&&(He=se.querySelector(".thought-content")),He||(se.insertAdjacentHTML("afterbegin",'<div class=\
"thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed\
" data-placeholder="1"></div></div>'),He=se.querySelector(".thought-content")),He&&(He.setAttribute(
"data-placeholder","1"),He.textContent=W||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
He),"ensureThoughtPlaceholder");st&&Tt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const ut=currentThreadId,vt=nowPerfMs(),sn=Date.now();let Ln=!1,
an=!1,Rt=!1,Et=null,Bt=null,wt=null,Vt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const jt=r((W,xe)=>{if(!xe||W==="status"&&Ln||W==="thought"&&an||W==="content"&&Rt)return;const Ae=Math.
max(0,nowPerfMs()-vt);W==="status"?Et=Ae:W==="thought"?Bt=Ae:W==="content"&&(wt=Ae),reportFirstTokenLatency(
{latency_seconds:Ae/1e3,latency_ms:Ae,thread_id:Vt||currentThreadId,job_id:currentJobId,model:X.model,
first_event_type:W,client_sent_at_ms:sn}),W==="status"?Ln=!0:W==="thought"?an=!0:W==="content"&&(Rt=
!0)},"maybeReportFirstEventLatency"),xt=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let $t=!1,Jt=!1,Ft=null,kt=null,It=!1;try{X.thread_id&&activeGem&&(threadGemMap[X.thread_id]=
activeGem,pendingGemForNewThread=null);const W=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(X),signal:abortController.signal}),se);if(sendClientDebugLog("info",`Prompt stre\
am response status: ${W.status}`),!W.ok){const De=await W.json().catch(()=>({})),ze=new Error(De.error||
`HTTP ${W.status}`);throw ze.serverCode=De.code||null,ze.serverModel=De.model||X.model,ze.acceptedJobId=
De.job_id||null,ze.acceptedThreadId=De.thread_id||null,ze}$t=!0,window.ConnectionMonitor&&(It=!0,window.
ConnectionMonitor.operationStarted()),xt&&xt.setPhase("waiting"),ne.accept(),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1});const xe=r(()=>{if(!se)return;const De=se.querySelector(".content-area");if(De&&De.
getAttribute("data-api-accepted")!=="1"&&(De.setAttribute("data-api-accepted","1"),!updatePendingSkeletonStatus(
se,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){
De.outerHTML=buildPendingSkeletonHtml(X.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const ze=se.querySelector(".content-area");ze&&ze.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
se,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");xe(),(on=window.ActivityLog)==null||on.log("send.stream.start");let Ae=!1;const Fe=W.
body.getReader(),at=new TextDecoder;let ft="",St="",Qe="",Lt=!0,Je=null,nt=null,Ct=null,ln=!1;const Pt={};
let cn=0,Xt=!1;for(;!Xt;){const{done:De,value:ze}=await Fe.read();if(!De&&!Ae&&(Ae=!0,(Kt=window.ActivityLog)==
null||Kt.log("send.stream.first")),De&&((_t=window.ActivityLog)==null||_t.log("send.stream.done")),De)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),xt&&xt.setPhase("receiving"),
ft+=at.decode(ze,{stream:!0});let Ze=ft.split(`
`);ft=Ze.pop();let dn=!1,un=!1;for(let Ht of Ze)if(Ht.trim())try{const Ce=JSON.parse(Ht);if(Ce.type===
"thread_id"){xe();const Pe=Ce.content!==null&&Ce.content!==void 0?String(Ce.content):Ce.content;Pe&&
(Vt=Pe,currentThreadId!==Pe&&(currentThreadId=Pe,history.pushState({},"","/c/"+Pe)),activeGem&&(threadGemMap[Pe]=
activeGem,pendingGemForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Ce.type==="job_\
id"){xe(),currentJobId=Ce.content,l&&showToast("Batch\u767B\u9332","info");continue}if(Ce.type==="se\
arch_status"){Ct=updateSearchBox(se,Ct,Ce.content);continue}if(Ce.type==="mcp"){handleMcpStreamEvent(
se,Ce.content||{});continue}if(Ce.type==="mcp_decision_request"){openMcpDecisionModal(Ce.content||{});
continue}if(Ce.type==="status"){xe();const Pe=Ce.content===null||Ce.content===void 0?"":String(Ce.content);
if(jt("status",!!Pe),Lt&&se){const We=Pe||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(
se,We,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ct=se.querySelector(".content-area");ct&&(ct.outerHTML=buildPendingSkeletonHtml(X.model,We),updatePendingSkeletonStatus(
se,We,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
st&&Tt(Pe||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Lt){beginPendingToStreamTransition(
se);const Pe=se.querySelector(".content-area");Pe&&(Pe.innerHTML=""),Lt=!1}if(Ce.type==="coding_diff")
appendCodingLiveDiff(se,Ce.content||{}),jt("content",!0);else if(Ce.type==="thought"){if(Je||(Je=se.
querySelector(".thought-content")),Qe+=Ce.content,jt("thought",!!Ce.content),!Je){const Pe='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';Ct?Ct.insertAdjacentHTML(
"afterend",Pe):se.insertAdjacentHTML("afterbegin",Pe),Je=se.querySelector(".thought-content")}if(Je&&
Je.getAttribute("data-placeholder")==="1"){if(Je.textContent="",Je.removeAttribute("data-placeholder"),
Je){const Pe=Je.parentElement.querySelector(".thought-header");Pe&&Pe.classList.remove("thinking-shi\
mmer")}Qe=Ce.content}Je.classList.remove("collapsed"),un=!0}else if(Ce.type==="image_analysis"){const Pe=Ce.
content===null||Ce.content===void 0?"":String(Ce.content);if(!se)continue;let We=se.querySelector(".\
image-analysis-box");if(!We){const dt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';Ct?Ct.insertAdjacentHTML("afterend",dt):se.insertAdjacentHTML("afterbegin",dt),We=se.querySelector(
".image-analysis-box")}const ct=We.querySelector(".image-analysis-text");ct&&(ct.textContent=Pe)}else if(Ce.
type==="python"){const Pe=Ce.content||{},We=Pe.id||`py_${Date.now()}`;if(!Pt[We]){const dt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${We}" data-collapsed="true" data-code-key="${We}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;Ct?Ct.insertAdjacentHTML("afterend",dt):se.insertAdjacentHTML(
"afterbegin",dt),Pt[We]=se.querySelector(`[data-py-id="${We}"]`)}const ct=Pt[We];if(ct){if(Pe.code!==
void 0){const dt=Pe.code==null?"":String(Pe.code),Mt=ct.querySelector(".python-code");Mt&&(Mt.textContent=
dt,Mt.removeAttribute("data-highlighted"),queueHighlight(ct,dt));const gt=ct.querySelector('.copy-bt\
n[data-copy="code"]');gt&&gt.setAttribute("data-code",encodeURIComponent(dt).replace(/'/g,"%27"))}if(Pe.
output!==void 0){const dt=Pe.output==null?"":String(Pe.output),Mt=ct.querySelector(".python-output");
Mt&&(Mt.textContent=dt);const gt=ct.querySelector('.copy-btn[data-copy="output"]');gt&&gt.setAttribute(
"data-code",encodeURIComponent(dt).replace(/'/g,"%27"))}}}else if(Ce.type==="content"){const Pe=Ce.content===
null||Ce.content===void 0?"":String(Ce.content);St+=Pe,/[`~]/.test(Pe)&&activateDeferredCodingModeFromStream(
St),nt||(nt=se.querySelector(".content-area")||document.createElement("div"),nt.className="prose pro\
se-invert text-sm break-words",se.contains(nt)||se.appendChild(nt)),dn=!0,jt("content",!!Pe)}else if(Ce.
type==="error"){ln=!0,Xt=!0,se.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Ce.content)),
showToast(Ce.content||"Unknown error","error",!0);break}}catch{}if(un&&Je&&(Je.textContent=Qe,userAutoScroll&&
(Je.scrollTop=Je.scrollHeight)),dn&&nt){const Ht=Date.now();if(Ht-cn>100){const Ce=snapshotCodeCollapse(
nt);renderAiMarkdownInto(nt,St,{incrementalMath:!0}),applyCodeCollapse(nt,Ce,!0),cn=Ht}}scrollToBottom()}
if(xt&&xt(),nt){const De=snapshotCodeCollapse(nt);renderAiMarkdownInto(nt,St,{incrementalMath:!0}),applyCodeCollapse(
nt,De,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),se)if(queueHighlight(se,St),enableLatencyMetrics){
const De=nowPerfMs()-vt;reportFirstTokenLatency({is_total:!0,latency_seconds:De/1e3,latency_ms:De,thread_id:Vt||
currentThreadId,job_id:currentJobId,model:X.model,client_sent_at_ms:sn,client_done_at_ms:Date.now()});
let ze='<div class="mt-2 pt-2 border-t border-gray-700/30 flex flex-col gap-1 items-end opacity-70 t\
ext-[10px] font-mono text-gray-400">',Ze=null;Et!==null&&(Ze=Et),Bt!==null&&(Ze===null||Bt<Ze)&&(Ze=
Bt),wt!==null&&(Ze===null||wt<Ze)&&(Ze=wt),Ze!==null&&(ze+=`<div>Initial: ${(Ze/1e3).toFixed(2)}s</d\
iv>`),wt!==null&&wt!==Ze&&(ze+=`<div>Content: ${(wt/1e3).toFixed(2)}s</div>`),ze+=`<div class="font-\
bold text-gray-300">Total: ${(De/1e3).toFixed(2)}s</div>`,currentJobId&&(ze+=`<div class="text-[9px]\
 opacity-50">Job ID: ${escapeHtml(currentJobId)}</div>`),ze+=`<div class="text-[10px] mt-1">${escapeHtml(
get("model-select").value)}</div>`,ze+="</div>",se.insertAdjacentHTML("beforeend",ze)}else se.insertAdjacentHTML(
"beforeend",`<div class="text-[10px] text-gray-500/50 mt-2 text-right font-mono">${escapeHtml(get("m\
odel-select").value)}</div>`);editingMessageId=null,setEditUi(!1),se&&se.querySelectorAll(".thought-\
content").forEach(ze=>ze.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!V}),!ln&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(De=>De.json()).then(De=>{De.title&&
(document.title=De.title+" - AI Chat",setCurrentChatHeaderTitle(De.title),loadThreads())}):loadThreads(
!1)}catch(W){(rn=window.ActivityLog)==null||rn.log(W.name==="AbortError"?"send.cancelled":"send.erro\
r"),ce();let xe=!1;const Ae=W.name==="AbortError"&&isManualStopAbortForThread(ut);if(W.name==="Abort\
Error"&&!Ae&&(xe=await syncThreadAfterAbortedStream(ut,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${W.message}`),!$t){G&&G.remove();const Fe=se&&se.closest(
".fade-in");Fe&&Fe.remove(),delete messageStore[K],delete messageMeta[K]}if(W.serverCode==="request_\
already_accepted"&&W.acceptedJobId&&W.acceptedThreadId)$t=!0,Ft={job_id:W.acceptedJobId,thread_id:String(
W.acceptedThreadId),model:X.model,pendingStartedAt:l?null:t},ne.accept();else if($t&&!Ae)kt={job_id:normalizeJobIdForUi(
currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):null,model:X.model,pendingStartedAt:l?
null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\
\u307E\u3059\u3002","warning",!1);else if(W.serverCode==="turnstile_required"){const Fe=await getTurnstileToken();
Fe?(await verifyTurnstileOnServer(Fe,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(W.serverCode==="api_key_missing"){const Fe=W.serverModel||X.model,at=await showApiKeyRequiredModalAsync(
Fe);at==="set"?Jt=!0:at==="switch"?showModal("model-modal"):showToast(W.message||`${getModelNameById(
Fe)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(W.
name!=="AbortError"){const Fe="Connection Error: "+W.message;showToast(Fe,"error",!0)}if(!$t){const Fe=V&&
ne.canRetry()&&!editingMessageId;ne.restore(),Fe&&(editingMessageId=V,currentParentId=me,promptSettingsBeforeEdit=
ue,setEditUi(!0))}V&&!xe&&O()}finally{yt&&yt(),It&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),xt&&xt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===Me&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,setEditUi(!!editingMessageId)}if(Ft){const W=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=Ft.thread_id,(W!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
Ft,currentThreadId)}if(kt&&kt.thread_id)return reconnectPendingStreamUntilAvailable(kt,kt.thread_id);
if(Jt&&ne.canRetry())return sendMessageLogged({animate:!1})}r(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var ue,me,re,ke;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const I=a.
querySelector(".content-area");I?I.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const o=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,setSendBtnToStopMode(),resumeChatAutoScroll(),
canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=new AbortController;const l=currentThreadId,
c=i.toLowerCase(),u=c.includes("gemini")||c.includes("o1")||c.includes("o3")||c.includes("gpt-5")||c.
includes("reasoning")&&!c.includes("non-reasoning");let f=null;const h=r(I=>!u||!a?null:((!f||!a.contains(
f))&&(f=a.querySelector(".thought-content")),f||(a.insertAdjacentHTML("afterbegin",'<div class="thou\
ght-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i class=\
"fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed" dat\
a-placeholder="1"></div></div>'),f=a.querySelector(".thought-content")),f&&(f.setAttribute("data-pla\
ceholder","1"),f.textContent=I||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),f),
"ensureThoughtPlaceholder");u&&h("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",v="",w="",x=!0,_=null,S=null,C=null,M=!1;const E={};let H=0,B=!1;const te=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let j=!1,V=!1;try{const I=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!I.ok)throw new Error(`Resume failed (${I.status})`);window.ConnectionMonitor&&
(V=!0,window.ConnectionMonitor.operationStarted()),te&&te.setPhase("waiting"),(ue=window.ActivityLog)==
null||ue.log("send.stream.start");let O=!1;const K=I.body.getReader(),R=new TextDecoder;for(;!B;){const{
done:G,value:Q}=await K.read();if(!G&&!O&&(O=!0,(me=window.ActivityLog)==null||me.log("send.stream.f\
irst")),G&&((re=window.ActivityLog)==null||re.log("send.stream.done")),G)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),te&&te.setPhase("receiving"),y+=R.decode(Q,{stream:!0});let P=y.
split(`
`);y=P.pop();let ee=!1,de=!1;for(let ye of P)if(ye.trim())try{const Z=JSON.parse(ye);if(Z.type==="jo\
b_id"){currentJobId=Z.content||t;continue}if(Z.type==="search_status"){C=updateSearchBox(a,C,Z.content);
continue}if(Z.type==="mcp"){handleMcpStreamEvent(a,Z.content||{});continue}if(Z.type==="mcp_decision\
_request"){openMcpDecisionModal(Z.content||{});continue}if(Z.type==="status"){const pe=Z.content===null||
Z.content===void 0?"":String(Z.content);if(x&&a){const X=pe||"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";
if(!updatePendingSkeletonStatus(a,X,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const Te=a.querySelector(".content-area");Te&&(Te.outerHTML=buildPendingSkeletonHtml(i,X),updatePendingSkeletonStatus(
a,X,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
u&&h(pe||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(x){beginPendingToStreamTransition(
a);const pe=a.querySelector(".content-area");pe&&(pe.innerHTML=""),x=!1}if(Z.type==="coding_diff")appendCodingLiveDiff(
a,Z.content||{});else if(Z.type==="thought"){if(_||(_=a.querySelector(".thought-content")),w+=Z.content,
!_){const pe='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content">\
</div></div>';C?C.insertAdjacentHTML("afterend",pe):a.insertAdjacentHTML("afterbegin",pe),_=a.querySelector(
".thought-content")}if(_&&_.getAttribute("data-placeholder")==="1"){if(_.textContent="",_.removeAttribute(
"data-placeholder"),_){const pe=_.parentElement.querySelector(".thought-header");pe&&pe.classList.remove(
"thinking-shimmer")}w=Z.content}_.classList.remove("collapsed"),de=!0}else if(Z.type==="image_analys\
is"){const pe=Z.content===null||Z.content===void 0?"":String(Z.content);if(!a)continue;let X=a.querySelector(
".image-analysis-box");if(!X){const ne='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 borde\
r border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas \
fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></\
div></div>';C?C.insertAdjacentHTML("afterend",ne):a.insertAdjacentHTML("afterbegin",ne),X=a.querySelector(
".image-analysis-box")}const Te=X.querySelector(".image-analysis-text");Te&&(Te.textContent=pe)}else if(Z.
type==="python"){const pe=Z.content||{},X=pe.id||`py_${Date.now()}`;if(!E[X]){const ne=`<div class="\
code-wrapper python-box collapsed" data-py-id="${X}" data-collapsed="true" data-code-key="${X}"><div\
 class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span>\
<div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\
\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="\
" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" \
data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i>\
</button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Co\
de</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-se\
ction"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output\
"></code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",ne):a.insertAdjacentHTML("after\
begin",ne),E[X]=a.querySelector(`[data-py-id="${X}"]`)}const Te=E[X];if(Te){if(pe.code!==void 0){const ne=pe.
code==null?"":String(pe.code),ce=Te.querySelector(".python-code");ce&&(ce.textContent=ne,ce.removeAttribute(
"data-highlighted"),queueHighlight(Te,ne));const Me=Te.querySelector('.copy-btn[data-copy="code"]');
Me&&Me.setAttribute("data-code",encodeURIComponent(ne).replace(/'/g,"%27"))}if(pe.output!==void 0){const ne=pe.
output==null?"":String(pe.output),ce=Te.querySelector(".python-output");ce&&(ce.textContent=ne);const Me=Te.
querySelector('.copy-btn[data-copy="output"]');Me&&Me.setAttribute("data-code",encodeURIComponent(ne).
replace(/'/g,"%27"))}}}else if(Z.type==="content"){const pe=Z.content===null||Z.content===void 0?"":
String(Z.content);v+=pe,/[`~]/.test(pe)&&activateDeferredCodingModeFromStream(v),S||(S=a.querySelector(
".content-area")||document.createElement("div"),S.className="prose prose-invert text-sm break-words",
a.contains(S)||a.appendChild(S)),ee=!0}else if(Z.type==="error"){M=!0,B=!0,a.insertAdjacentHTML("bef\
oreend",buildChatErrorBubbleHtml(Z.content)),showToast(Z.content||"Unknown error","error",!0);break}}catch{}
if(de&&_&&(_.textContent=w,userAutoScroll&&(_.scrollTop=_.scrollHeight)),ee&&S){const ye=Date.now();
if(ye-H>100){const Z=snapshotCodeCollapse(S);renderAiMarkdownInto(S,v,{incrementalMath:!0}),applyCodeCollapse(
S,Z,!0),H=ye}}scrollToBottom()}if(te&&te(),S){const G=snapshotCodeCollapse(S);renderAiMarkdownInto(S,
v,{incrementalMath:!0}),applyCodeCollapse(S,G,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,v),
a&&a.querySelectorAll(".thought-content").forEach(Q=>Q.classList.add("collapsed")),await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0}),loadThreads(!1)}catch(I){(ke=window.ActivityLog)==null||
ke.log(I.name==="AbortError"?"send.cancelled":"send.stream.error");const O=I.name==="AbortError"&&isManualStopAbortForThread(
l);I.name==="AbortError"&&!O&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
O||(j=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{o&&o(),V&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
te&&te(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,currentThreadPending=null}if(j)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i,pendingStartedAt:e.pendingStartedAt},l)}r(resumePendingStream,"resumePendingStream");
function updateThreadHighlighting(){const e=get("thread-list");if(!e)return;e.querySelectorAll("[dat\
a-thread-id]").forEach(n=>{n.dataset.threadId===String(currentThreadId)?n.classList.add("bg-gray-700\
/60","border-l-2","border-blue-500"):n.classList.remove("bg-gray-700/60","border-l-2","border-blue-5\
00")})}r(updateThreadHighlighting,"updateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){
snapshotSidebarHistory("loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory(
"loadThreads-start append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?
t.value:"";if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");
return}const a=await(await apiFetch(`${CHAT_CONFIG.urls.handleThreads}?q=${encodeURIComponent(n)}&pa\
ge=${threadPage}`)).json(),o=get("thread-list");if(!o)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory(
"loadThreads-skip-replace-settings-open");return}const c=a&&Array.isArray(a.threads)?a.threads.length:
-1,u=o.querySelectorAll("[data-thread-id]").length;if(c===0&&u>0&&String(n||"").trim()){snapshotSidebarHistory(
"loadThreads-keep-existing-empty-search");return}if(o.innerHTML='<div id="thread-pull-indicator" cla\
ss="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="\
fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scro\
ll-sentinel"></div>',threadObserver){threadObserver.disconnect();const f=get("scroll-sentinel");f&&threadObserver.
observe(f)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const u=String(
c.id),f=document.createElement("div"),h=c.is_bookmarked?"text-yellow-400":"text-gray-500",y=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=u===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";f.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,f.dataset.threadId=u,f.dataset.threadTitle=c.title||"",f.innerHTML=`<div class="flex \
items-center gap-1 truncate flex-1"><button class="${h} hover:text-yellow-400 px-1" onclick="toggleB\
ookmark(event, '${u}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(
c.title||"No Title")}</span>${y}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 m\
d:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:tex\
t-white px-1 transition" onclick="renameThread(event, '${u}')"><i class="fas fa-pen text-xs"></i></b\
utton><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, \
'${u}')"><i class="fas fa-trash text-xs"></i></button></div>`,f.onclick=x=>{x.target.closest("button")||
x.target.closest("[data-thread-actions]")||loadMessages(u)},l?o.insertBefore(f,l):o.appendChild(f)}),
hasMoreThreads=!!a.has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendere\
d count="+a.threads.length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){
console.error("Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=
!1,updateThreadHighlighting(),snapshotSidebarHistory("loadThreads-finally")}}r(loadThreads,"loadThre\
ads");function initPullToRefresh(e,t){const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,
o=88,l=52,c=.5,u=8;let f=0,h=!1,y=0,v=null;const w=r(()=>get(i),"indicatorEl"),x=r(()=>{const C=w();
return C?C.querySelector(".ptr-pull-label"):null},"labelEl"),_=r(C=>{const M=w();if(!M)return;M.style.
height=Math.min(C,o)+"px",M.classList.toggle("active",C>2),M.classList.toggle("pull-ready",C>=a);const E=x();
E&&(E.textContent=C>=a?"\u96E2\u3057\u3066\u66F4\u65B0":"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},
"applyPullUI"),S=r(()=>{const C=w();C&&(C.style.height="0px",C.classList.remove("active","pull-ready",
"refreshing"),C.classList.remove("dragging"))},"resetPullUI");n.addEventListener("touchstart",C=>{if(v){
h=!1;return}if(n.scrollTop>0){h=!1;return}const M=C.touches[0];M&&(f=M.clientY,y=0,h=!0)},{passive:!0}),
n.addEventListener("touchmove",C=>{if(!h||v)return;if(n.scrollTop>0){h=!1;return}const M=C.touches[0];
if(!M)return;const E=M.clientY-f;if(E<=0){y>0&&(y=0,_(0)),h=!1;return}const H=w();H&&!H.classList.contains(
"dragging")&&H.classList.add("dragging"),y=Math.min(E*c,o),_(y),E>=u&&C.preventDefault()},{passive:!1}),
n.addEventListener("touchend",()=>{if(!h||(h=!1,v))return;const C=w();C&&C.classList.remove("draggin\
g");const M=y>=a;if(y=0,!M){S();return}let E;try{E=t()}catch{E=null}const H=w();if(H){H.classList.add(
"refreshing"),H.style.height=l+"px";const B=H.querySelector(".ptr-pull-label");B&&(B.textContent="\u66F4\u65B0\
\u4E2D...")}E&&typeof E.then=="function"?(v=E,E.catch(()=>{}).finally(()=>{v=null,S()})):(v=Promise.
resolve(),setTimeout(()=>{v=null,S()},400))}),n.addEventListener("touchcancel",()=>{h=!1,y=0,S()})}r(
initPullToRefresh,"initPullToRefresh");const initThreadPullToRefresh=r(()=>initPullToRefresh("thread\
-list",()=>loadThreads(!1)),"initThreadPullToRefresh"),initGemPullToRefresh=r(()=>initPullToRefresh(
"gem-list",()=>loadGems()),"initGemPullToRefresh"),initPullToRefreshAll=r(()=>{initThreadPullToRefresh(),
initGemPullToRefresh()},"initPullToRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;
const mcpCardIdSelector=r(e=>"mcp_card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),
mcpEscHtml=r(e=>String(e==null?"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"\
&quot;","'":"&#39;"})[t]),"mcpEscHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||
"MCP")} / ${mcpEscHtml(e.tool_name||e.internal_name||"")}`}r(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){
if(!e)return null;let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("\
div"),t.className="mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),
e.appendChild(t)),t}r(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){
if(!e||!t||!t.type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(
e):null;if(n&&!i)return;const a=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.
querySelector('[data-mcp-card="'+a+'"]'))return;const o=`<div class="mcp-box mcp-running mb-2" data-\
mcp-card="${a}">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",o);return}if(t.type==="result"){let o=i.querySelector('[dat\
a-mcp-card="'+a+'"]');const l=t.summary||"";if(o)o.classList.remove("mcp-running"),o.classList.add("\
mcp-done"),o.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const c=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${a}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",c),o=i.querySelector('[data-mcp-card="'+a+'"]')}if(l){const c=document.
createElement("div");c.className="mcp-box-note",c.textContent=l.split(`
`)[0].slice(0,220),o&&o.appendChild(c)}return}if(t.type==="error"){let o=i.querySelector('[data-mcp-\
card="'+a+'"]');const l=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(o)o.classList.remove("mcp-running"),o.classList.add("mcp-error"),o.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const u=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",u),o=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
o&&o.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const o=get("mcp-decision-modal");if(o&&!o.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}r(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const u=JSON.parse(c);c=JSON.stringify(u,null,2)}catch{}a.textContent=c}const o=get(
"mcp-decision-allow"),l=get("mcp-decision-deny");o&&(o.onclick=()=>submitMcpDecision("allow")),l&&(l.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}r(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}r(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const a=e[0],o=a.
thread_id;n.textContent=e.length===1?`${a.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),o&&await loadMessages(o)});const l=get(
"batch-notification-close");l&&(l.onclick=()=>t.classList.add("hidden"))}r(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({}));(t.active||[]).some(a=>currentThreadId&&String(a.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const i=t.
completed||[];i.length&&(showGeminiBatchCompletionBanner(i),i.some(a=>String(a.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}r(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);let chatTransitionSequence=0,chatTransitionTimer=null;const CHAT_TRANSITION_DURATION_MS=460;function playChatTransition(e){
const t=get("chat-transition-veil");if(!t)return;const n=++chatTransitionSequence;if(chatTransitionTimer&&
(clearTimeout(chatTransitionTimer),chatTransitionTimer=null),window.matchMedia&&window.matchMedia("(\
prefers-reduced-motion: reduce)").matches){t.classList.remove("is-active"),t.removeAttribute("data-t\
ransition-kind");return}t.dataset.transitionKind=e||"history",t.classList.remove("is-active"),t.offsetWidth,
t.classList.add("is-active"),chatTransitionTimer=setTimeout(()=>{n===chatTransitionSequence&&(t.classList.
remove("is-active"),chatTransitionTimer=null)},CHAT_TRANSITION_DURATION_MS)}r(playChatTransition,"pl\
ayChatTransition");async function toggleBookmark(e,t){e&&e.stopPropagation(),await apiFetch(`/api/th\
reads/${t}/bookmark`,{method:"POST"}),loadThreads()}r(toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){
const n=++threadLoadSequence,i=t.targetMessageId||new URLSearchParams(location.search).get("message");
window.closeHistoryModal&&window.closeHistoryModal();const a=!!t.preserveDraft,o=!!t.silent;o||resumeChatAutoScroll(
{scroll:!1});const l=o?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",u="",f=[];
if(a){const y=get("prompt-input");c=y?y.value:"",u=y?y.style.height:"",f=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,promptSettingsBeforeEdit=null,setEditUi(!1)}else cancelEdit();o||playChatTransition(
"history");const h=e!=null?String(e):e;h!==currentThreadId&&(modelChosenManually=!1),currentThreadId=
h,t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveGemForThread(
currentThreadId),get("welcome-screen").classList.add("hidden"),o||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const y=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);y.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const v=i;v&&/^\d+$/.test(String(v))&&y.
searchParams.set("before_id",String(Number(v)+1));const w=await apiFetch(y.toString());if(!w.ok)throw new Error(
`thread request failed (${w.status})`);const x=await w.json();if(!x||!Array.isArray(x.messages))throw new Error(
"invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(x&&x.title),
allMessages=x.messages,renderBranchOrigin(x.branch_origin),threadHasOlderMessages=!!x.has_older_messages,
oldestLoadedMessageId=x.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const _=(allMessages||
[]).filter(M=>M.role==="user"&&M.content).map(M=>M.content);if(promptHistory=[...new Set(_.slice().reverse())],
historyIndex=-1,tempPrompt="",currentThreadPending=x.pending_job||null,setTemporaryChatUiState(!!(x&&
x.is_temporary)),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(!0),get("thread-c\
ustom-instruction")&&(get("thread-custom-instruction").value=x.custom_instruction||""),get("enable-p\
rompt-cache")&&(get("enable-prompt-cache").checked=!!x.enable_prompt_caching,updatePromptCacheUi()),
x.last_model&&!modelChosenManually&&selectModelById(x.last_model),x.last_gem_uuid&&loadedGems.length>
0){const M=loadedGems.find(E=>E.uuid===x.last_gem_uuid);M&&(threadGemMap[currentThreadId]=M,applyActiveGem(
M))}const S=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),C=i;if(C&&
allMessages.some(M=>String(M.id)===String(C))?currentLeafId=Number(C):S&&allMessages.find(M=>String(
M.id)===String(S))?currentLeafId=S:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(o?{silent:o,keepScroll:o}:{silent:o,keepScroll:o,animate:!0}),
a||restoreLatestPromptOptions(),o&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):o||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!o&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const M=get("prompt-input");M&&(M.value=c||"",
u?M.style.height=u:M.style.height="auto"),currentImageUrls=f,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i&&new URLSearchParams(
location.search).has("message")&&history.replaceState({},"",location.pathname),a||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(y){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",y),o||showChatLoadError(e),o||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}r(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const o=await(await apiFetch(i.toString())).json(),l=Array.isArray(o.messages)?o.messages:[];
if(l.length){const c=new Set(allMessages.map(f=>f.id)),u=l.filter(f=>!c.has(f.id));u.length&&(allMessages=
u.concat(allMessages))}if(threadHasOlderMessages=!!o.has_older_messages,oldestLoadedMessageId=o.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
r(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;const o=i?Array.from(a.children):[],l=o.map(
S=>S.getBoundingClientRect().height);let c=null;if(i&&(c=a.scrollTop),allMessages.length===0){a.replaceChildren(),
currentParentId=null,updateTotalTokenBar(0);return}const u={};allMessages.forEach(S=>{u[S.id]=S,S.childrenIds=
[]}),allMessages.forEach(S=>{S.parent_id&&u[S.parent_id]&&u[S.parent_id].childrenIds.push(S.id)}),(!currentLeafId||
!u[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null);const f=[];
let h=u[currentLeafId];for(;h;)f.unshift(h),h=u[h.parent_id];const y=buildTokenTotals(f),v=buildTokenTotals(
allMessages),w=document.createDocumentFragment();if(threadHasOlderMessages){const S=loadingOlderMessages?
"\u8AAD\u307F\u8FBC\u307F\u4E2D...":"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",
C=loadingOlderMessages?"disabled":"",M=document.createElement("div");M.className="mb-3 text-center",
M.innerHTML=`<button id="load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-g\
ray-600 text-gray-200 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="lo\
adOlderMessages()" ${C}><i class="fas fa-clock-rotate-left mr-1"></i>${S}</button>`,w.appendChild(M)}
f.forEach(S=>{const C=S.parent_id?u[S.parent_id]:null,M=C?C.childrenIds:allMessages.filter(H=>!H.parent_id).
map(H=>H.id),E=M.length>1?{current:M.indexOf(S.id)+1,total:M.length,siblings:M}:null;renderMessage(S.
id,S.role,S.content,S.image_url,S.thought_data,S.model,E,n,S.quote_text,S.tokens,S.tokens_in,S.tokens_out,
S.is_encrypted,S.tokens_content,S.tokens_thought,w,!1,S.parent_id,S.gem_name,S.batch_job,S.prompt_options&&
typeof S.prompt_options=="object"?S.prompt_options.image_mask:null)});const x=currentThreadPending;if(x&&
!isPendingJobSuppressed(x.job_id)){const S=x.message_id,C=new Set(f.map(H=>H.id)),M=f.length?f[f.length-
1]:null;if(S&&C.has(S)&&currentLeafId===S||!S&&M&&M.role==="user"){const H=x.job_id?`pending-${x.job_id}`:
null;renderPendingMessage(w,n,!1,H,x.model||null)}}const _=[];if(i&&!lowBandwidthMode&&Array.from(w.
children).forEach((S,C)=>{const M=o.find(H=>H.id&&H.id===S.id)||(o.length===w.children.length?o[C]:null),
E=M?l[o.indexOf(M)]:0;E>0&&(S.style.minHeight=`${E}px`,_.push(S))}),a.replaceChildren(w),displayMissingMessageFiles(
a,f),updateTotalTokenBar(y.tokens_total,y,v),currentParentId=currentLeafId,i&&c!==null?restoreThreadTreeScroll(
a,c):scrollToBottom(),lowBandwidthMode)queueMessageDecorations(a,a&&a.textContent||"");else if(queueHighlight(
a),f.length){const S=queueMathTypeset(a,f.map(C=>C.content||"").join(`
`));Promise.resolve(S).finally(()=>{if(!_.length||!_[0].isConnected)return;const C=a.scrollTop;_.forEach(
M=>{M.style.minHeight=""}),restoreThreadTreeScroll(a,C)})}}r(renderThreadTree,"renderThreadTree");function restoreThreadTreeScroll(e,t){
if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&!chatManualPauseIntent?e.scrollTop=
e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=e.scrollTop,syncScrollToBottomButton()}
r(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){currentLeafId=e;const t={};
allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(i=>{i.parent_id&&t[i.parent_id]&&
t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null,renderThreadTree({animate:!0}),restoreLatestPromptOptions();return}for(;t[n]&&t[n].
childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree({animate:!0}),
restoreLatestPromptOptions()}r(switchVersion,"switchVersion");async function loadGems(){try{const t=await(await apiFetch(
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
="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=o=>{o.target.closest("button")||activateGem(
i)},n.appendChild(a)})}catch(e){console.error("Failed to load gems:",e)}}r(loadGems,"loadGems");function setGemDefaultModelSelect(e){
const t=get("gem-default-model");if(!t)return;t.innerHTML="";const n=document.createElement("option");
n.value="",n.textContent="Use current model",t.appendChild(n),MODELS.forEach(a=>{const o=(a.items||[]).
filter(c=>!c.deprecated);if(!o.length)return;const l=document.createElement("optgroup");l.label=a.category,
o.forEach(c=>{const u=document.createElement("option");u.value=c.id,u.textContent=c.name,l.appendChild(
u)}),t.appendChild(l)});const i=e||"";if(i&&!Array.from(t.options).some(a=>a.value===i)){const a=document.
createElement("option");a.value=i,a.textContent=MODEL_NAME_BY_ID[i]||i,t.appendChild(a)}t.value=i}r(
setGemDefaultModelSelect,"setGemDefaultModelSelect");async function openEditGemModal(e,t){e.stopPropagation(),
editingGemUuid=t;try{const i=await(await apiFetch(`/api/gems/${t}`)).json();get("gem-name").value=i.
name,get("gem-desc").value=i.description||"",get("gem-inst").value=i.instruction,setGemDefaultModelSelect(
i.default_model),renderGemFixedPromptsForEdit(i.fixed_prompts),get("gem-modal-title").innerHTML='<i \
class="fas fa-gem text-blue-500 mr-2"></i>Edit Gem',get("save-gem-btn").innerText="Save Changes",showModal(
"gem-modal"),location.pathname!=="/gem"&&history.pushState({modal:"gem"},"","/gem")}catch{showToast(
"Gem\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}r(openEditGemModal,"o\
penEditGemModal");async function createGem(e,t){await apiFetch(CHAT_CONFIG.urls.handleGems,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:e,instruction:t})}),loadGems()}
r(createGem,"createGem");function applyActiveGem(e){activeGem=e||null;const t=get("fixed-prompts-bar");
if(activeGem){if(activeGem.default_model&&selectModelById(activeGem.default_model),get("active-gem-n\
ame").innerText=activeGem.name,get("gem-active-indicator").classList.remove("hidden"),t){t.innerHTML=
"";let n=[];try{activeGem.fixed_prompts&&(n=JSON.parse(activeGem.fixed_prompts))}catch{}n.length>0?(t.
classList.remove("hidden"),n.forEach((i,a)=>{const o=document.createElement("button");o.className="f\
ixed-prompt-chip whitespace-nowrap px-4 py-1.5 text-[11px] font-bold bg-gray-700 hover:bg-gray-600 t\
ext-gray-100 rounded-full transition-all shadow-md border border-gray-600/50 flex items-center",o.style.
animationDelay=`${a*40}ms`,o.textContent=String(i.name||""),o.onclick=()=>{const l=get("prompt-input");
l&&(l.value=i.content,l.dispatchEvent(new Event("input")),sendMessage())},t.appendChild(o)})):t.classList.
add("hidden")}}else get("gem-active-indicator").classList.add("hidden"),t&&(t.innerHTML="",t.classList.
add("hidden"));get("sys-prompt-option").style.opacity="1"}r(applyActiveGem,"applyActiveGem");function syncActiveGemForThread(e){
const t=e&&threadGemMap[e]?threadGemMap[e]:null;applyActiveGem(t)}r(syncActiveGemForThread,"syncActi\
veGemForThread");async function saveThreadGemUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_gem_uuid:t,thread_id:e})})}catch{}}
r(saveThreadGemUuid,"saveThreadGemUuid");function activateGem(e,t){currentThreadId?(threadGemMap[currentThreadId]=
e,applyActiveGem(e),showToast(`Gem "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadGemUuid(currentThreadId,e?e.uuid:null)):(pendingGemForNewThread=e,applyActiveGem(
e),allMessages&&allMessages.length>0&&startNewChat({preserveGem:!0}))}r(activateGem,"activateGem");function clearActiveGem(){
currentThreadId&&(delete threadGemMap[currentThreadId],saveThreadGemUuid(currentThreadId,null)),pendingGemForNewThread=
null,applyActiveGem(null)}r(clearActiveGem,"clearActiveGem");function addGemFixedPromptRow(e="",t=""){
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
            `,n.appendChild(i)}r(addGemFixedPromptRow,"addGemFixedPromptRow");function collectGemFixedPrompts(){
const e=document.querySelectorAll(".gem-fixed-prompt-row"),t=[];return e.forEach(n=>{const i=n.querySelector(
".gem-fp-name").value.trim(),a=n.querySelector(".gem-fp-content").value.trim();i&&a&&t.push({name:i,
content:a})}),t.length>0?JSON.stringify(t):null}r(collectGemFixedPrompts,"collectGemFixedPrompts");function renderGemFixedPromptsForEdit(e){
const t=get("gem-fixed-prompts-container");if(t){t.innerHTML="";try{e&&JSON.parse(e).forEach(i=>addGemFixedPromptRow(
i.name,i.content))}catch{}}}r(renderGemFixedPromptsForEdit,"renderGemFixedPromptsForEdit");function getCurrentChatHeaderTitleText(){
return typeof currentThreadTitle=="string"&&currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?
"No Title":"AI Chat"}r(getCurrentChatHeaderTitleText,"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){
return temporaryChatEnabled?`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:
""}r(getTemporaryChatTimeoutLabel,"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){
const e=getCurrentChatHeaderTitleText(),t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=[
"sidebar-chat-title","mobile-chat-title"],a=["sidebar-chat-temporary-label","mobile-chat-temporary-l\
abel"],o=["sidebar-chat-ttl","mobile-chat-ttl"];i.forEach(l=>{const c=get(l);c&&(c.textContent=e)}),
a.forEach(l=>{const c=get(l);c&&c.classList.toggle("hidden",!n)}),o.forEach(l=>{const c=get(l);c&&(n&&
t?(c.textContent=t,c.classList.remove("hidden")):(c.textContent="",c.classList.add("hidden")))})}r(updateCurrentChatHeaderUi,
"updateCurrentChatHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="str\
ing"?e:null,updateCurrentChatHeaderUi()}r(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){
tempChatExpiresAtMs=null,updateCurrentChatHeaderUi()}r(resetTemporaryChatExpiresAt,"resetTemporaryCh\
atExpiresAt");function applyTemporaryChatRuntimeMeta(e){if(!e||typeof e!="object")return;Object.prototype.
hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(e.timeout_seconds);let t=null;
const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.floor(n*1e3);else{const i=Number(
e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=
t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=null),updateCurrentChatHeaderUi()}
r(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
r(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,t=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let n=Number(e);return Number.isFinite(n)||(n=Number(t)),Number.isFinite(n)||(n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
n=Math.trunc(n),n<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(n=TEMP_CHAT_TIMEOUT_MIN_SECONDS),n>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(n=TEMP_CHAT_TIMEOUT_MAX_SECONDS),n}r(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
n=get("temporary-chat-welcome-desc");n&&(n.textContent=t);const i=get("temporary-chat-container");i&&
(i.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
r(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const t=get(
"set-temp-chat-timeout-seconds");t&&(t.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}r(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,t))}r(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const t=get("enable-temporary-c\
hat");t&&t.checked!==temporaryChatEnabled&&(t.checked=temporaryChatEnabled);const n=get("welcome-def\
ault-content");n&&n.classList.toggle("hidden",temporaryChatEnabled);const i=get("welcome-temporary-c\
ontent");i&&i.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}r(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}r(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}r(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const t=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
n=await t.json().catch(()=>({}));t.ok&&n&&applyTemporaryChatRuntimeMeta(n),t.ok&&n&&n.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}r(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const t=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==t)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=t,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}r(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const t=!!e;if(setTemporaryChatUiState(
t),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const n=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:t})}),i=await n.json().catch(()=>({}));if(!n.ok)throw new Error(i&&i.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(i&&i.is_temporary)),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}r(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
if(playChatTransition("new"),threadLoadSequence++,abortController&&abortController.abort(),cancelEdit(),
modelChosenManually=!1,restoreLatestPromptOptions(),resetUploadState(),stopTemporaryChatHeartbeat(),
setTemporaryChatUiState(!1),currentThreadTitle=null,tempChatExpiresAtMs=null,currentThreadId=null,renderBranchOrigin(
null),allMessages=[],promptHistory=[],historyIndex=-1,tempPrompt="",threadHasOlderMessages=!1,oldestLoadedMessageId=
null,loadingOlderMessages=!1,currentLeafId=null,currentParentId=null,currentThreadPending=null,updateTotalTokenBar(
0),typeof window.__refreshAdminThreadEncState=="function")try{window.__refreshAdminThreadEncState()}catch{}
e.skipHistory||history.pushState({},"","/"),get("chat-container").innerHTML="",get("welcome-screen").
classList.remove("hidden"),updateCurrentChatHeaderUi(),get("thread-custom-instruction")&&(get("threa\
d-custom-instruction").value=""),get("enable-prompt-cache")&&(get("enable-prompt-cache").checked=!1,
updatePromptCacheUi()),e.preserveGem?activeGem&&applyActiveGem(activeGem):applyActiveGem(null),loadThreads();
const t=threadLoadSequence;fetchSettingsSnapshot().then(n=>{if(threadLoadSequence!==t||currentThreadId||
modelChosenManually||e.preserveGem&&activeGem)return;const i=n.use_last_chat_settings&&n.last_model||
n.default_model;i&&MODELS.some(a=>a.items.some(o=>o.id===i))&&selectModelById(i)}).catch(n=>console.
warn("Failed to load chat defaults",n)),window.innerWidth<768&&get("overlay").click()}r(startNewChat,
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
json(),o=get("thread-app-global-sys-prompt-preview");o&&(o.value=a.global_system_prompt_effective||"");
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
modalThreadId!==t)return;const o=get("thread-custom-instruction");o&&(o.value=a.custom_instruction||
"");const l=get("thread-include-global-instruction");l&&(l.checked=a.include_global_instruction!==!1)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),a=i?i.value:"",o=get("thread-include-global-instruction"),l=o?o.checked:
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
ut-type").value,i=get("compression-format-only").checked;setCompressionSettings(e,t,n,i);const a=r((l,c)=>{
get(l)&&get(c)&&(get(c).value=get(l).value)},"syncBack");a("modal-gpt-image-size","gpt-image-size"),
a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-format","gpt-image-format"),a("m\
odal-gpt-image-compression","gpt-image-compression"),a("modal-gemini-image-aspect","gemini-image-asp\
ect"),a("modal-gemini-image-size","gemini-image-size"),a("modal-grok-image-aspect","grok-image-aspec\
t"),a("modal-grok-image-resolution","grok-image-resolution"),a("modal-grok-image-quality","grok-imag\
e-quality"),IDEOGRAM_IMAGE_FIELDS.forEach(l=>a(`modal-ideogram-image-${l}`,`ideogram-image-${l}`)),a(
"modal-ocr-table-format","ocr-table-format"),a("modal-ocr-pages","ocr-pages");const o=r((l,c)=>{get(
l)&&get(c)&&(get(c).checked=get(l).checked)},"syncBackChk");o("modal-ocr-extract-header","ocr-extrac\
t-header"),o("modal-ocr-extract-footer","ocr-extract-footer"),o("modal-ocr-include-blocks","ocr-incl\
ude-blocks"),o("modal-ocr-include-images","ocr-include-images"),window.closeCompressionModal(),showToast(
"\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","success")};async function deleteGem(e,t){e.
stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleGemItem.replace("0",t),
{method:"DELETE"}),loadGems())}r(deleteGem,"deleteGem");function renameThread(e,t){var y;e.stopPropagation();
const n=get("thread-rename-modal"),i=get("thread-rename-form"),a=get("thread-rename-input"),o=get("t\
hread-rename-save"),l=get("thread-rename-cancel"),c=get("thread-rename-error"),u=e.currentTarget;a.value=
((y=u.closest("[data-thread-id]"))==null?void 0:y.dataset.threadTitle)||"",c.textContent="",o.disabled=
!1;let f=!1;const h=r(()=>{f||(hideModal("thread-rename-modal"),u.focus())},"close");l.onclick=h,n.onclick=
v=>{v.target===n&&h()},n.onkeydown=v=>{v.key==="Escape"&&(v.preventDefault(),v.stopPropagation(),h()),
v.key==="Tab"&&(v.shiftKey&&document.activeElement===a?(v.preventDefault(),l.focus()):!v.shiftKey&&document.
activeElement===l&&(v.preventDefault(),a.focus()))},i.onsubmit=async v=>{v.preventDefault();const w=a.
value.trim();if(f||!w){w||a.focus();return}f=!0,o.disabled=!0,l.disabled=!0;try{const x=await apiFetch(
CHAT_CONFIG.urls.updateTitle.replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},
body:JSON.stringify({title:w})}),_=await x.json();if(!x.ok)throw new Error("rename failed");currentThreadId===
String(t)&&setCurrentChatHeaderTitle(_.title||w),f=!1,h(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
f=!1,o.disabled=!1,l.disabled=!1}},l.disabled=!1,showModal("thread-rename-modal"),a.focus()}r(renameThread,
"renameThread");async function deleteThread(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(
CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),currentThreadId===t?startNewChat():
loadThreads())}r(deleteThread,"deleteThread");async function deleteMessage(e,t){!t&&!confirm("Delete\
 this message and subsequent history?")||(await apiFetch(CHAT_CONFIG.urls.deleteMessage.replace("0",
e),{method:"DELETE"}),loadMessages(currentThreadId))}r(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=r(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=r(
e=>{if(!e)return"";try{const t=new Date(e);return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(t)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=r(e=>{if(!e)
return"";let t=String(e).trim();if(!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=r(e=>{const t=pdfNormalizeAttachmentPath(e);return t?
`${window.location.origin}/files/${encodeURI(t)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=r(
e=>{const t=pdfNormalizeAttachmentPath(e);return t?`${window.location.origin}/${PDF_IMAGE_EXTS.has((t.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(t)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=r(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const a=n&&n.filename?n.
filename:i.split("/").pop(),o=n&&n.source?String(n.source):"attachment",l=!!(n&&n.is_image),c=n&&n.url?
n.url:buildPdfAttachmentUrl(i),u=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:o,isImage:l,url:c,previewUrl:u}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=r(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(u=>maybeNeedsMathJax(u.content)||maybeNeedsMathJax(u.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",o=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(u=>{const f=u.
role==="user",h=u.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(u.quote_text)}\
</div>`:"",y=u.thought_text?`<div class="thought">${escapeHtml(u.thought_text)}</div>`:"",v=f?`<div \
class="content" style="white-space: pre-wrap;">${escapeHtml(u.content||"")}</div>`:`<div class="cont\
ent">${sanitizeMarkdownHtml(u.content||"")}</div>`,w=buildPdfMessageAttachments(u),x=w.length?`<div \
class="attachments">${w.map(C=>C.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(C.previewUrl)}\
" alt="${pdfEscapeAttr(C.filename)}"><div class="file-caption">${pdfEscapeAttr(C.filename)}</div></d\
iv>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(C.url)}" target="_blank" rel="no\
referrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
C.filename)}</span><span class="file-source">${pdfEscapeAttr(C.source)}</span></span></a></div>`).join(
"")}</div>`:"",_=[];u.model&&!f&&_.push(u.model),u.tokens!==null&&u.tokens!==void 0&&_.push(`tokens:${u.
tokens}`),u.tokens_in!==null&&u.tokens_in!==void 0&&_.push(`in:${u.tokens_in}`),u.tokens_out!==null&&
u.tokens_out!==void 0&&_.push(`out:${u.tokens_out}`),u.tokens_thought!==null&&u.tokens_thought!==void 0&&
_.push(`thought:${u.tokens_thought}`),u.is_encrypted&&_.push("encrypted"),u.parent_id!==null&&u.parent_id!==
void 0&&_.push(`parent:#${u.parent_id}`);const S=_.length?`<div class="message-meta">${pdfEscapeAttr(
_.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${f?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${f?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${f?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(u.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${h}
                            ${v}
                            ${y}
                            ${x}
                            ${S}
                        </div>
                    </article>
                `}).join("");return`
        <!DOCTYPE html>
        <html lang="ja">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${pdfEscapeAttr(o)} - PDF Export</title>
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
        <h1>${pdfEscapeAttr(o)}</h1>
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
"error",!0);return}const o=document.createElement("iframe");activePdfPrintFrame=o,o.setAttribute("ar\
ia-hidden","true"),o.style.position="fixed",o.style.right="0",o.style.bottom="0",o.style.width="1px",
o.style.height="1px",o.style.opacity="0",o.style.pointerEvents="none",o.style.border="0";let l=null;
const c=r(()=>{l&&(clearTimeout(l),l=null),t&&t.remove(),(activePdfPrintFrame===o||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{o.parentNode&&o.parentNode.removeChild(o)}catch{}},"cleanup");l=setTimeout(
()=>{activePdfPrintFrame===o&&(console.log("PDF print cleanup fallback triggered"),c())},6e4),o.onload=
async()=>{try{const h=o.contentDocument,y=o.contentWindow;if(!h||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(C=>maybeNeedsMathJax(C.content)||maybeNeedsMathJax(C.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),h.fonts&&h.fonts.ready)try{await h.fonts.ready}catch{}t.update(60);const x=Array.from(h.
images||[]),_=Promise.all(x.map(C=>C.complete?Promise.resolve():new Promise(M=>{C.addEventListener("\
load",M,{once:!0}),C.addEventListener("error",M,{once:!0})})));if(await Promise.race([_,new Promise(
C=>setTimeout(C,5e3))]),t.update(80),h.getElementById("MathJax-script")){let C=0;for(;C<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(M=>setTimeout(M,50)),C++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(M){console.error(
"PDF MathJax typeset failed",M)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const u=buildPdfDocumentHtml(a),f=new Blob([u],{type:"text/html"});o.src=URL.createObjectURL(
f),document.body.appendChild(o)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}r(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}r(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}r(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=r((u,f)=>i.
compare(u.filename||"",f.filename||""),"nameAsc"),o=r((u,f)=>i.compare(f.filename||"",u.filename||""),
"nameDesc"),l=r((u,f)=>(Number(f.ts)||0)-(Number(u.ts)||0),"tsDesc"),c=r((u,f)=>(Number(u.ts)||0)-(Number(
f.ts)||0),"tsAsc");return t==="name_asc"?n.sort((u,f)=>a(u,f)||l(u,f)):t==="name_desc"?n.sort((u,f)=>o(
u,f)||l(u,f)):t==="oldest"?n.sort((u,f)=>c(u,f)||a(u,f)):n.sort((u,f)=>l(u,f)||a(u,f)),n}r(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}r(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}r(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}r(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}r(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const f=t.querySelector(".lib-empty-state");f&&f.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const f=get("lib-total-count");f&&(f.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),o=i.filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(
a)),l=get("lib-total-count");if(l){const f=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?l.innerText=`${lib.files.length} / ${f} files`:l.innerText=`${f} files`}if(!o.length){
if(n)return;const f=lib.favoritesOnly&&!a?"fa-star":"fa-search",h=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${f}"></i></div>\
<p class="lib-empty-title">${h}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(f=>lib.favoritesOnly&&!f.is_favorite?!1:!a||fileNameForSearch(f).includes(a)):o).forEach(f=>{
try{const h=renderLibraryItem(f,c++);t.appendChild(h)}catch{}})}r(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(u=>fileNameForSearch(u).includes(n)):t).filter(u=>u.type==="image"),o=lib.favoritesOnly?
a.filter(u=>u.is_favorite):a;if(!o.length)return;const l=o.map(u=>({url:u.url,filename:u.filename||u.
original_filename||u.url.split("/").pop(),element:null}));let c=l.findIndex(u=>u.url===e.url);c===-1&&
(c=0),openViewerWithItems(l,c)}r(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}r(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,a=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),o=e.type==="\
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
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${o}</div>${l}${h}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",_=>{_.shiftKey&&_.preventDefault()}),n.onclick=
_=>{if(_&&_.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const S=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),C=S.findIndex(E=>E.dataset.filepath===lib.anchorPath),
M=S.indexOf(n);if(C!==-1&&M!==-1){const E=Math.min(C,M),H=Math.max(C,M);for(let B=E;B<=H;B++){const te=S[B].
dataset.filepath;te&&(lib.selected.add(te),S[B].classList.add("is-selected"))}try{const B=window.getSelection&&
window.getSelection();B&&B.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(_=>{_.onclick=S=>{S.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async _=>{_.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const x=n.querySelector(".lib-favorite-btn");return x&&
(x.onclick=async _=>{_.stopPropagation(),x.disabled=!0;try{const S=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
C=await S.json().catch(()=>({}));if(!S.ok||typeof C.is_favorite!="boolean")throw new Error(C.error||
"favorite update failed");e.is_favorite=C.is_favorite,renderLibraryGrid(),showToast(C.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),x.disabled=!1}}),n}r(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}r(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",a=FILE_BASE_URL+e,o=i==="image"?FILE_THUMB_BASE_URL+
e:null,l={filename:t,original_filename:t,filepath:e,url:a,thumbnail_url:o,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(l),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
r(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(o=>o.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const a=(i||"").trim();if(!a){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const o=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:a})}),l=await o.json().catch(
()=>({}));if(!o.ok){showToast(l.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=l.filename||a,setAttachmentNameForPath(e,t.filename));const c=get(
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(u=>{u.getAttribute("data-filename")===
e&&setRowAttachmentName(u,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}r(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,t){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),t&&t.parentNode&&t.remove(),lib.files&&(lib.files=
lib.files.filter(n=>n.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}r(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}r(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],t=(lib.files||[]).find(a=>a&&a.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const a=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);a.searchParams.set("filepath",e);const o=await apiFetch(
a.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),l=await o.json().catch(()=>({}));
if(!o.ok)throw new Error(l.error||`HTTP ${o.status}`);const c=Array.isArray(l.chats)?l.chats:[];if(!c.
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
</div>'}}}r(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let o=null,l=!1;try{const u=getLibSortOrder(),f=getLibSearchQuery(),h=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(h),sort:u,q:f,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);o=await v.json(),l=!0}catch(u){i=u}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(o)?o:
o&&Array.isArray(o.files)?o.files:[];o&&!Array.isArray(o)&&(lib.totalCount=Number(o.total)||0,lib.hasMore=
!!o.has_more,lib.nextOffset=(Number(o.offset)||0)+(Number(o.limit)||c.length));try{const u=FILE_BASE_URL,
f=FILE_THUMB_BASE_URL,h=new Set(c.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(c.length>=LIBRARY_PAGE_SIZE||!v||h.has(v))return;const w=getAttachmentNameForPath(
v)||v.split("/").pop()||v,x=(w.split(".").pop()||"").toLowerCase(),_=["png","jpg","jpeg","webp","gif"].
includes(x)?"image":"file",S=_==="image"?f+v:null;c.unshift({filename:w,original_filename:w,filepath:v,
url:u+v,thumbnail_url:S,type:_,ext:x,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),h.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const u=c.filter(h=>h&&h.filepath&&
h.url);let f=[];if(e){const h=new Set(lib.files.map(y=>y.filepath));f=u.filter(y=>!h.has(y.filepath)),
lib.files.push(...f)}else lib.files=u;lib.files.forEach(h=>{h&&h.filepath&&setAttachmentNameForPath(
h.filepath,h.filename||h.original_filename||"")}),lib.fileSet=new Set(lib.files.map(h=>h.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
f:null)}catch(u){i=i||u}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}r(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}r(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(a=>{const o=isAudioPath(a),
l=isVideoPath(a);if(o&&!e.audio||l&&!e.video){o&&(t+=1),l&&(n+=1);return}const c=normalizeAttachmentPath(
a);if(!c)return;const u=(lib.files||[]).find(f=>f&&f.filepath===a);u&&u.filename&&setAttachmentNameForPath(
c,u.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const a=[];t&&a.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&a.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${a.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}r(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(t=>{const n=(lib.
files||[]).find(i=>i&&i.filepath===t);if(n&&n.url){const i=document.createElement("a");i.href=n.url,
i.download=n.filename||n.original_filename||t.split("/").pop()||"file",document.body.appendChild(i),
i.click(),document.body.removeChild(i)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}r(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const t=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=t,showModal("legal-modal");const n=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!n.ok)return;const i=await n.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
i)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,t)=>{const n=decodeURIComponent(t),i=r(()=>{const a=e.getAttribute("data-copy")||"";e.innerHTML=a===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},a=>{console.error(a),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(i,2e3)})},window.copyMessage=(e,t)=>{const n=messageStore[e]||
"";copyToClipboard(n,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i \
class="fas fa-copy"></i>',2e3)},i=>{console.error(i),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(
()=>t.innerHTML='<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;
t.classList.contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}r(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}r(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let t=0,n=e;const i={};for((allMessages||[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n];t+=a.tokens||
Number(a.tokens_in||0)+Number(a.tokens_out||0),n=a.parent_id}return t}r(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n],o=a.model||"Unknown";t[o]||(t[o]={total:0,in:0,out:0,
thought:0});const l=a.tokens||Number(a.tokens_in||0)+Number(a.tokens_out||0);t[o].total+=l,t[o].in+=
Number(a.tokens_in||0),t[o].out+=Number(a.tokens_out||0),t[o].thought+=Number(a.tokens_thought||0),n=
a.parent_id}return t}r(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(a=>t[a.id]={...a,children:[]}),
allMessages.forEach(a=>{a.parent_id&&t[a.parent_id]?t[a.parent_id].children.push(t[a.id]):a.parent_id||
n.push(t[a.id])});function i(a){const o=document.createElement("div");o.className="flex flex-col ite\
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
updateBranchDetailPane()},o.appendChild(l),a.children.length>0){const y=document.createElement("div");
y.className="w-px h-4 bg-gray-700",o.appendChild(y);const v=document.createElement("div");v.className=
"flex gap-4 items-start",a.children.forEach(w=>v.appendChild(i(w))),o.appendChild(v)}return o}r(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}r(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}r(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(u=>u.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const o=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);o.innerHTML="",Object.entries(l).sort((u,f)=>f[1].
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
                `,o.appendChild(h)}),get("br-name-input").value=branchLabelNames[n.id]||"";const c=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(c.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
c.classList.replace("bg-amber-600","bg-gray-600")):(c.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
c.classList.replace("bg-gray-600","bg-amber-600"))}r(updateBranchDetailPane,"updateBranchDetailPane"),
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
r(batchProviderLabel,"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\
\u9001\u4FE1\u5F85\u3061",JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",
JOB_STATE_RUNNING:"\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\
\u5B8C\u4E86",JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\
\u505C\u6B62",JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}
r(batchStateLabelShort,"batchStateLabelShort");function batchStateTone(e){const t=String(e||"").toUpperCase();
return t==="JOB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":t==="JOB\
_STATE_FAILED"?"border-red-500/40 bg-red-900/20 text-red-200":t==="JOB_STATE_CANCELLED"||t==="JOB_ST\
ATE_EXPIRED"?"border-gray-500/40 bg-gray-700/30 text-gray-300":t==="JOB_STATE_CANCELLING"?"border-am\
ber-500/40 bg-amber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}r(
batchStateTone,"batchStateTone");function batchFormatTime(e){if(!e)return"";let t=String(e);!/[zZ]$/.
test(t)&&!/[+-]\d\d:?\d\d$/.test(t)&&(t+="Z");const n=new Date(t);return isNaN(n.getTime())?String(e):
n.toLocaleString("ja-JP",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}r(batchFormatTime,
"batchFormatTime");function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove(
"batch-list-enter"),e.offsetWidth,e.classList.add("batch-list-enter"))}r(playBatchListAnimation,"pla\
yBatchListAnimation");function renderBatchJobs(e={}){const t=get("batch-list");if(!t)return;const n=batchJobsCache.
filter(a=>batchFilterMode==="active"?!!a.is_active:batchFilterMode==="done"?!a.is_active:!0),i=get("\
batch-count");if(i&&(i.textContent=`${n.length}\u4EF6`),t.innerHTML="",!n.length){t.innerHTML='<div \
class="batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}n.forEach(a=>{const o=document.createElement("div");o.className=
"batch-job-card";const l=escapeHtml(a.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),c=escapeHtml(
batchProviderLabel(a.provider)),u=escapeHtml(a.model||""),f=escapeHtml(batchFormatTime(a.created_at)),
h=escapeHtml(a.status_text||""),y=batchStateTone(a.state),v=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",w=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",x=a.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';o.innerHTML=`
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
                        ${v}${w}${x}
                    </div>`;const _=o.querySelector("[data-batch-open]");_&&(_.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const S=o.querySelector("[data-batch-cancel]");S&&(S.onclick=()=>cancelBatchJob(
a));const C=o.querySelector("[data-batch-delete]");C&&(C.onclick=()=>deleteBatchJob(a)),t.appendChild(
o)}),e.animate&&playBatchListAnimation()}r(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const t=await apiFetch("/api/batch/jobs");if(!t.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const n=await t.json().catch(()=>({}));batchJobsCache=Array.isArray(n.jobs)?n.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}r(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),n=await t.json().catch(()=>({}));if(!t.ok){showToast(n.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}r(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),n=await t.json().catch(()=>({}));
if(!t.ok){showToast(n.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}r(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(t=>{t.classList.toggle("is-active",t===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=r(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const a=i?get(i.inputId):null;get("api-key-modal-input").
value=a?a.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const o=get(
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),u=r(
()=>{o.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),f=r(h=>{h.key==="Enter"&&(h.preventDefault(),
o.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",f),o.onclick=async()=>{
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
let a=!1;async function o(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const u=c.map(f=>{try{return f instanceof Error?f.stack||f.message:typeof f=="object"?JSON.
stringify(f):String(f)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,u)}catch{}finally{
a=!1}}r(o,"sendToServer"),console.log=function(...l){e.apply(console,l),o("log",l)},console.error=function(...l){
t.apply(console,l),o("error",l)},console.warn=function(...l){n.apply(console,l),o("warn",l)},console.
info=function(...l){i.apply(console,l),o("info",l)},window.addEventListener("error",function(l){o("e\
xception",[l.message,l.filename,l.lineno,l.colno,l.error])}),window.addEventListener("unhandledrejec\
tion",function(l){o("promise-rejection",[l.reason])}),setTimeout(()=>{console.log("Extended debug lo\
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
\u30AB\u30A6\u30F3\u30C8",ip:"IP\u30A2\u30C9\u30EC\u30B9",cookie:"\u7AEF\u672B\uFF08Cookie\uFF09"},o={
toggle_detection:"\u691C\u51FA\u306E\u5207\u308A\u66FF\u3048",ban:"BAN",unban:"BAN\u89E3\u9664\uFF08\u5358\u72EC\uFF09",
unban_linked:"BAN\u89E3\u9664\uFF08\u9023\u9396\uFF09",unlock:"\u30ED\u30C3\u30AF\u89E3\u9664",unblock_identifiers:"\
IP\u30FB\u7AEF\u672B\u306EBAN\u89E3\u9664"},l={userId:0,username:"",exists:!1,type:"",items:[],hasMore:!1,
selected:new Set,counts:{},total:0,data:null},c={accounts:[],selected:new Set},u=r(()=>get("bot-log-\
detail"),"view"),f=r(()=>get("bot-log-list-view"),"listView"),h=r(I=>{if(!I)return"-";const O=new Date(
I);return isNaN(O.getTime())?String(I):O.toLocaleString("ja-JP")},"formatTime"),y=r(I=>{const O=Math.
max(0,Math.floor(Number(I)||0));return O>=60?`${Math.floor(O/60)}\u5206${O%60?O%60+"\u79D2":""}`:`${O}\
\u79D2`},"formatDuration"),v=r((I,O)=>O==null||O===""?"-":I==="lock_source"?a[O]||String(O):I==="cli\
ent"||I==="origin_client"?n[O]||String(O):I==="action"?o[O]||String(O):I==="by"?O==="self"?"\u672C\u4EBA":
O==="admin"?"\u7BA1\u7406\u8005":String(O):I==="enabled"?O?"ON":"OFF":I==="remaining_seconds"||I==="\
lock_seconds"?y(O):I==="applied_to"&&Array.isArray(O)?O.map(K=>a[K]||K).join("\u30FB"):typeof O=="ob\
ject"?JSON.stringify(O,null,2):String(O),"formatDetailValue"),w=r(I=>{const O=String(I||"").trim();if(!O)
return"";let K=null;if(O.startsWith("{"))try{K=JSON.parse(O)}catch{K=null}if(K&&typeof K=="object"&&
!Array.isArray(K)){const R=Object.keys(K).map(G=>{const Q=v(G,K[G]),P=i[G]||G;return Q.length>120||Q.
includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(P)}</div><pre class="bot-log-pre">${escapeHtml(
Q)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(P)}:</span> <span class="text-gray-3\
00">${escapeHtml(Q)}</span></div>`}).join("");return O.length>600?`<details class="mt-1"><summary cl\
ass="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${R}</details>`:`<div cla\
ss="mt-1 space-y-1">${R}</div>`}return O.length>300?`<details class="mt-1"><summary class="cursor-po\
inter text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(O)}\
</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(O)}</div>`},"renderDeta\
ils"),x=r(async(I,O)=>{const K=await apiFetch(I,{method:"POST",headers:{"Content-Type":"application/\
json"},body:JSON.stringify(O)}),R=await K.json().catch(()=>({}));if(!K.ok)throw new Error(R.error||String(
K.status));return R},"postJson"),_=r((I,O)=>`<span class="${O} text-white px-2 py-0.5 rounded">${escapeHtml(
I)}</span>`,"badge"),S=r(()=>{const I=get("bot-log-account-list");if(!I)return;const O=c.accounts,K=O.
length&&O.every(G=>c.selected.has(G.user_id)),R=get("bot-log-list-toolbar");if(R&&(R.innerHTML=`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${K?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!O.length){I.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}I.innerHTML=O.map(G=>{const Q=[];G.exists||Q.push(_("\u524A\u9664\u6E08\u307F","bg-gray-600")),
G.is_bot_banned&&Q.push(_("BAN\u4E2D","bg-red-600")),G.lock_remaining_seconds>0&&Q.push(_(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
ceil(G.lock_remaining_seconds/60)}\u5206\uFF09`,"bg-yellow-600"));const P=G.evidence_count?`\u8A18\u9332 ${G.
evidence_count}\u4EF6\u30FB\u6700\u7D42 ${escapeHtml(h(G.last_event_at))}`:"\u8A18\u9332\u306A\u3057";
return`
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 round\
ed p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${G.
user_id}" ${c.selected.has(G.user_id)?"checked":""} aria-label="${escapeHtml(G.username||"ID "+G.user_id)}\
 \u3092\u9078\u629E">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${G.exists?
"text-gray-200":"text-gray-400"} font-bold bot-log-wrap">${escapeHtml(G.username||"ID "+G.user_id)}<\
/span>${Q.join("")}</div>
                                <div class="text-[10px] text-gray-500">${P}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${G.user_id}" data-username="${escapeHtml(
G.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),C=r(async()=>{const I=get("bot-log-acc\
ount-list");if(!I)return;I.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const O=get("bot-log-search"),K=O?O.value.
trim():"";try{const R=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(K)}`,{cache:"\
no-store"}),G=await R.json().catch(()=>({}));if(!R.ok)throw new Error(G.error||String(R.status));c.accounts=
G.accounts||[];const Q=new Set(c.accounts.map(P=>P.user_id));Array.from(c.selected).forEach(P=>{Q.has(
P)||c.selected.delete(P)}),S()}catch{I.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),M=r(I=>{if(!I)return"";const O=I.locks||[],K=I.banned_identifiers||[],
R=[];I.exists||R.push(_("\u524A\u9664\u6E08\u307F","bg-gray-600")),I.exists&&(R.push(_(I.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",I.detection_enabled?"bg-gray-600":"bg-gray-700")),R.push(_(I.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",I.is_bot_banned?"bg-red-600":"bg-gray-600"))),R.push(_(O.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",O.length?"bg-yellow-600":"bg-gray-600")),K.length&&R.push(_(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${K.length}\u4EF6`,"bg-red-600")),I.is_admin&&R.push(_("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const G=!!(l.counts.ban||l.counts.related_ban),Q=I.is_bot_banned?`<div class="text-g\
ray-300">BAN\u306E\u7406\u7531: ${escapeHtml(I.bot_ban_reason||"-")}\uFF08${escapeHtml(h(I.bot_banned_at))}\
\uFF09</div>${G?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",P=O.map(X=>{const Te=a[X.source]||X.source,ne=X.identifier?` ${escapeHtml(X.identifier)}${X.source===
"cookie"?"\u2026":""}`:"",ce=X.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
X.origin.username||String(X.origin.user_id||"-"))}\uFF08${escapeHtml(n[X.origin.client]||X.origin.client||
"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(Te)}${ne}\u30FB\u6B8B\u308A${escapeHtml(
y(X.remaining_seconds))}\u30FB${escapeHtml(X.reason||"")}${ce}</div>`}).join(""),ee=K.map(X=>`<div c\
lass="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[X.kind]||X.kind)} ${escapeHtml(X.
identifier)}${X.kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(X.reason||"-")}\uFF08${escapeHtml(h(X.
created_at))}\uFF09</div>`).join(""),de=K.length?'<div class="text-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\
\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002</div>':
"",ye=I.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${y(I.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",Z=I.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String(I.appeal_count))}\u4EF6`:"",pe=[];return O.length&&pe.push('<button class="bot-log-unlock bg-\
yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
K.length&&pe.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 r\
ounded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${R.join("")}</div>
                        ${Q}
                        ${P}
                        ${ee}
                        ${de}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String(I.lock_count))} / ${escapeHtml(String(I.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(ye)}\
\u30FB\u5931\u6557 ${escapeHtml(String(I.turnstile_fail_count))} / ${escapeHtml(String(I.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round((I.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round((I.behavior_score||0)*10)/10))}\uFF09${Z}</div>
                        ${pe.length?`<div class="flex flex-wrap gap-2 pt-1">${pe.join("")}</div>`:""}\

                    </div>`},"renderState"),E=r(()=>{const I=r((K,R,G)=>`<button class="bot-log-filt\
er ${l.type===K?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(K)}">${escapeHtml(R)} (${G})</button>`,"chip"),O=[I("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((K,R)=>l.counts[R]-l.counts[K]).forEach(K=>{O.push(I(K,(t[K]||
{label:K}).label,l.counts[K]))}),O.join("")},"renderFilters"),H=r(I=>{const O=t[I.event_type]||{label:I.
event_type,cls:"bg-gray-600"},K=I.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[I.client]||I.client)}</span>`:"",R=l.selected.has(I.id)?"checked":"",G=I.score!==
null&&I.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String(I.score))}${I.
behavior_score!==null&&I.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String(I.behavior_score))}\
\uFF09`:""}</span>`:"",Q=I.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
I.reasons)}</div>`:"",P=I.ip_address||I.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${I.ip_address?"IP "+escapeHtml(I.ip_address):""}${I.ip_address&&I.user_agent?"\u30FB":""}${escapeHtml(
I.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${I.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${I.id}" ${R}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(h(I.created_at))}</span>
                            <span class="${O.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(O.label)}\
</span>
                            ${K}
                            ${G}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${I.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${Q}
                        ${w(I.details)}
                        ${P}
                    </div>`},"renderItem"),B=r(()=>{const I=u();if(!I)return;const O=l.items.length?
l.items.map(H).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
K=l.items.length&&l.items.every(R=>l.selected.has(R.id));I.innerHTML=`
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
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${K?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-\
white px-2 py-1 rounded" ${l.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u8A18\u9332\u3092\u524A\u9664${l.
selected.size?`\uFF08${l.selected.size}\u4EF6\uFF09`:""}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white\
 px-2 py-1 rounded" ${l.total?"":"disabled"}>\u3059\u3079\u3066\u306E\u8A18\u9332\u3092\u524A\u9664</button>
                        </div>
                        <div class="space-y-2">${O}</div>
                        ${l.hasMore?'<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-60\
0 text-white px-2 py-1.5 rounded text-xs">\u3055\u3089\u306B\u8AAD\u307F\u8FBC\u3080</button>':""}
                    </div>`},"render"),te=r(async(I=!1)=>{const O=u();if(!O||!l.userId)return;I||(O.
innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D.\
..</div>');const K=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&K.set("t\
ype",l.type),I&&l.items.length&&K.set("before_id",String(l.items[l.items.length-1].id));try{const R=await apiFetch(
`/api/bot/evidence?${K.toString()}`,{cache:"no-store"}),G=await R.json().catch(()=>({}));if(!R.ok)throw new Error(
G.error||String(R.status));l.counts=G.counts||{},l.total=G.total||0,l.hasMore=!!G.has_more,G.user&&(l.
username=G.user.username||l.username,l.exists=!!G.user.exists),I?l.items=l.items.concat(G.items||[]):
(l.items=G.items||[],l.data=G),B()}catch{I||(O.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),j=r(async(I,O)=>{if(confirm(O))try{const K=await x("/api/bot/evidence/delete",
Object.assign({user_id:l.userId},I));showToast(`${K.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await te(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),V=r(async(I,O,K)=>{if(confirm(O))try{await x(I,{user_id:l.userId}),showToast(
K,"success"),await te(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","\
error",!0)}},"runAccountAction"),ue=r(()=>{const I=u();I&&(I.classList.add("hidden"),I.classList.remove(
"flex"),I.innerHTML="");const O=f();O&&(O.classList.remove("hidden"),O.classList.add("flex")),l.userId=
0},"showList"),me=r(async()=>{ue(),await C()},"openList"),re=r(async(I,O="")=>{const K=u();if(!K)return;
Object.assign(l,{userId:Number(I)||0,username:O,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),l.selected.clear();const R=f();R&&(R.classList.add("hidden"),R.classList.remove("flex")),
K.classList.remove("hidden"),K.classList.add("flex"),await te(!1)},"open");return{open:re,openList:me,
showList:ue,bind:r(()=>{const I=get("bot-log-modal");if(!I||I.dataset.bound==="1")return;I.dataset.bound=
"1";const O=get("bot-log-search");O&&O.addEventListener("keydown",K=>{K.key==="Enter"&&C()}),I.addEventListener(
"change",K=>{const R=K.target;if(!(!R||!R.classList))if(R.classList.contains("bot-log-select")){const G=Number(
R.getAttribute("data-log-id"));R.checked?l.selected.add(G):l.selected.delete(G),B()}else if(R.classList.
contains("bot-log-select-all"))l.items.forEach(G=>{R.checked?l.selected.add(G.id):l.selected.delete(
G.id)}),B();else if(R.classList.contains("bot-log-account-select")){const G=Number(R.getAttribute("d\
ata-user-id"));R.checked?c.selected.add(G):c.selected.delete(G),S()}else R.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(G=>{R.checked?c.selected.add(G.user_id):c.selected.delete(
G.user_id)}),S())}),I.addEventListener("click",async K=>{const R=K.target.closest("button");if(!(!R||
R.disabled))if(R.id==="bot-log-search-btn"||R.id==="bot-log-refresh-btn")R.id==="bot-log-refresh-btn"&&
O&&(O.value=""),await C();else if(R.classList.contains("bot-log-open-account")){const G=Number(R.getAttribute(
"data-user-id"));G&&await re(G,R.getAttribute("data-username")||"")}else if(R.classList.contains("bo\
t-log-delete-accounts")){const G=Array.from(c.selected);if(!G.length||!confirm(`\u9078\u629E\u3057\u305F${G.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const Q=await x("/api/bot/evidence/delete",{user_ids:G});showToast(`${Q.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\
\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await C()}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\
\u3057\u305F","error",!0)}}else if(R.classList.contains("bot-log-back"))await me();else if(R.classList.
contains("bot-log-reload"))await te(!1);else if(R.classList.contains("bot-log-more"))await te(!0);else if(R.
classList.contains("bot-log-filter"))l.type=R.getAttribute("data-type")||"",l.selected.clear(),await te(
!1);else if(R.classList.contains("bot-log-delete-one")){const G=Number(R.getAttribute("data-log-id"));
G&&await j({ids:[G]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
R.classList.contains("bot-log-delete-selected")?l.selected.size&&await j({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
R.classList.contains("bot-log-delete-all")?await j({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
R.classList.contains("bot-log-unlock")?await V("/api/bot/account/clear-lock",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):R.classList.contains("bot-log-unbloc\
k")&&await V("/api/bot/account/clear-identifiers",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u304C\u8D77\u70B9\u306EIP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u89E3\u9664\u3059\u308B\u3068\u3001\u305D\u306EIP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u308B\u3088\u3046\u306B\u306A\u308A\u307E\u3059\u3002`,
"IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664\u3057\u307E\u3057\u305F")})},"bind")}})();window.BotAdminLog=
BotAdminLog;let branchCreating=!1;const branchRequests=new Map,BRANCH_SAVE_WARNING="\u5206\u5C90\u5148\u306F\u901A\u5E38\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u3057\u3066\u4FDD\u5B58\
\u3055\u308C\u3001\u81EA\u52D5\u524A\u9664\u3055\u308C\u307E\u305B\u3093\u3002\u30B3\u30D4\u30FC\u5BFE\u8C61\u306E\u6DFB\u4ED8\u30FB\u751F\u6210\u30D5\u30A1\u30A4\u30EB\u3082\u4FDD\u5B58\u5BFE\u8C61\u306B\u306A\u308A\u307E\u3059\u3002";
function confirmBranchSave(){return new Promise(e=>{const t=document.createElement("dialog");t.className=
"rounded-xl bg-gray-800 text-gray-100 p-6 max-w-lg border border-gray-600",t.setAttribute("aria-labe\
l","\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90");const n=document.createElement("p");
n.textContent=BRANCH_SAVE_WARNING,t.appendChild(n);const i=document.createElement("form");i.method="\
dialog",i.className="flex justify-end gap-3 mt-4";for(const[a,o]of[["cancel","\u30AD\u30E3\u30F3\u30BB\u30EB"],
["save","\u4FDD\u5B58\u3057\u3066\u5206\u5C90"]]){const l=document.createElement("button");l.value=a,
l.textContent=o,l.className="rounded px-3 py-2 border border-gray-600",i.appendChild(l)}t.appendChild(
i),t.addEventListener("close",()=>{const a=t.returnValue==="save";t.remove(),e(a)},{once:!0}),document.
body.appendChild(t),t.showModal()})}r(confirmBranchSave,"confirmBranchSave");async function branchChatMessage(...e){
return window.ActivityLog?window.ActivityLog.operation("branch",()=>branchChatMessageLogged.apply(this,
e)):branchChatMessageLogged.apply(this,e)}r(branchChatMessage,"branchChatMessage");async function branchChatMessageLogged(e,t){
var a;if(branchCreating||abortController||!currentThreadId||!/^\d+$/.test(String(e))||temporaryChatEnabled&&
!await confirmBranchSave())return;const n=String(currentThreadId),i=n+":"+e;branchRequests.has(i)||branchRequests.
set(i,crypto.randomUUID().replace(/-/g,"")),branchCreating=!0,t&&(t.disabled=!0);try{const o=await apiFetch(
`/api/threads/${encodeURIComponent(n)}/branch`,{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({message_id:Number(e),request_id:branchRequests.get(i),confirm_persist_temporary:!!temporaryChatEnabled})}),
l=await o.json();if(!o.ok)throw new Error(l.error||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
branchRequests.delete(i),await loadMessages(l.id,{targetMessageId:l.leaf_id}),await loadThreads(),showToast(
"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90\u3057\u307E\u3057\u305F")}catch(o){(a=
window.ActivityLog)==null||a.log("branch.error"),showToast(o.message||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{branchCreating=!1,t&&(t.disabled=!1)}}r(branchChatMessageLogged,"branchChatMessa\
geLogged");function renderBranchOrigin(e){let t=get("branch-origin-banner");if(!t){const n=get("chat\
-container");if(!n)return;t=document.createElement("div"),t.id="branch-origin-banner",t.className="p\
x-4 py-2 text-xs text-gray-400 border-b border-gray-700";const i=get("total-token-bar");i?i.before(t):
n.parentNode.insertBefore(t,n)}if(t.replaceChildren(),t.hidden=!e,!!e)if(e.available&&e.thread_id&&e.
message_id){const n=document.createElement("a");n.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F",
n.className="text-blue-400 hover:underline",n.href="/c/"+encodeURIComponent(e.thread_id)+"?message="+
Number(e.message_id),n.onclick=i=>{i.preventDefault(),loadMessages(e.thread_id,{targetMessageId:e.message_id})},
t.appendChild(n)}else t.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093"}
r(renderBranchOrigin,"renderBranchOrigin");function displayMissingMessageFiles(e,t){const n=r(i=>{try{
const a=new URL(i,location.origin);return a.origin===location.origin?decodeURIComponent(a.pathname).
replace(/^\/files\/(thumb\/)?/,""):""}catch{return String(i)}},"normalize");for(const i of t){if(!i.
missing_files||!i.missing_files.length)continue;const a=e.querySelector("#msg-"+Number(i.id));if(!a)
continue;const o=new Set(i.missing_files.flatMap(l=>{const c=String(l).replace(/^\/files\//,"");return[
c,c.replace(/^\d+\//,"")]}));a.querySelectorAll("img[src], a[href], audio[src], video[src], source[s\
rc], .file-thumb[onclick]").forEach(l=>{let c=n(l.getAttribute("src")||l.getAttribute("href")||"");if(l.
classList.contains("file-thumb")){const h=(l.getAttribute("onclick")||"").match(/window.open\('([^']+)'/);
h&&(c=n(h[1]))}if(!o.has(c))return;const u=document.createElement("span");u.className="inline-block \
rounded bg-gray-800 p-3 text-xs text-gray-400",u.textContent="\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",
u.title=c.split("/").pop();const f=l.closest(".chat-image-frame")||(l.tagName==="SOURCE"?l.parentNode:
l);f&&a.contains(f)&&f.replaceWith(u)})}}r(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}r(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},o=1/0;for(let l=1;l<=i;l++){if(i%l)continue;const c=i/l,u=Math.abs(
Math.log(Math.max(1,t)/l/(Math.max(1,n)/c)));(u<o-1e-9||Math.abs(u-o)<=1e-9&&(t>=n?l>a.columns:c>a.rows))&&
(a={columns:l,rows:c},o=u)}return a}r(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,o=Math.round(e/n.columns*a/2),l=Math.round(t/n.rows*a/2),c=[];for(let u=0;u<
n.rows;u++)for(let f=0;f<n.columns;f++){const h=Math.floor(f*e/n.columns),y=Math.floor((f+1)*e/n.columns),
v=Math.floor(u*t/n.rows),w=Math.floor((u+1)*t/n.rows);c.push({index:c.length,row:u,column:f,cellLeft:h,
cellTop:v,cellRight:y,cellBottom:w,left:Math.max(0,h-o),top:Math.max(0,v-l),right:Math.min(e,y+o),bottom:Math.
min(t,w+l)})}return c}r(splitTiles,"splitTiles");function splitFits(e,t,n){return Math.floor(t/e.columns)>=
16&&Math.floor(n/e.rows)>=16}r(splitFits,"splitFits");function splitTileCaption(e,t,n,i){return`Tile\
 ${e.index+1}/${t.columns*t.rows}  row ${e.row+1}/${t.rows}, col ${e.column+1}/${t.columns}  x ${e.left}\
-${e.right}, y ${e.top}-${e.bottom} of ${n}x${i}`}r(splitTileCaption,"splitTileCaption");function splitBaseName(e){
return String(e).split("/").pop().replace(/\.[^.]*$/,"").replace(/[\\/:*?"<>|\s]+/g,"_").replace(/^[_.]+|[_.]+$/g,
"").slice(0,80)||"image"}r(splitBaseName,"splitBaseName");function splitTileName(e,t,n){const i=n.columns*
n.rows;return`${e}_part${String(t.index+1).padStart(String(i).length,"0")}of${i}_r${t.row+1}c${t.column+
1}.png`}r(splitTileName,"splitTileName");function readSplitCompression(){let e;try{e=JSON.parse(localStorage.
getItem(SPLIT_COMPRESSION_KEY)||"{}")}catch{}return e=e&&typeof e=="object"?e:{},{enabled:e.enabled!==
!1,maxSizeMB:Math.max(.1,Math.min(10,Number(e.maxSizeMB)||1)),maxDimension:splitClamp(e.maxDimension,
256,8192,1920),outputType:["original","image/jpeg","image/png","image/webp"].includes(e.outputType)?
e.outputType:"original",formatOnly:e.formatOnly===!0}}r(readSplitCompression,"readSplitCompression");
function splitOptions(){return{pieces:splitClamp(get("image-split-pieces").value,2,30,4),overlap:splitClamp(
get("image-split-overlap").value,0,50,10),includeOriginal:get("image-split-original").checked,markOriginal:get(
"image-split-mark-original").checked,markPieces:get("image-split-mark-pieces").checked}}r(splitOptions,
"splitOptions");function splitCanvas(e,t){const n=document.createElement("canvas");if(n.width=e,n.height=
t,n.width!==e||n.height!==t||!n.getContext("2d"))throw new Error("\u753B\u50CF\u304C\u5927\u304D\u3059\u304E\u308B\u305F\u3081\u51E6\u7406\u3067\u304D\u307E\u305B\u3093\u3002");
return n}r(splitCanvas,"splitCanvas");async function loadSplitImage(e){const t=URL.createObjectURL(e),
n=new Image;try{return await new Promise((i,a)=>{n.onload=i,n.onerror=()=>a(new Error(`\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093: ${e.
name}`)),n.src=t}),{image:n,release:r(()=>{n.src="",URL.revokeObjectURL(t)},"release")}}catch(i){throw URL.
revokeObjectURL(t),i}}r(loadSplitImage,"loadSplitImage");function drawSplitOverview(e,t,n,i,a,o){const l=e.
getContext("2d"),c=e.width/n,u=e.height/i;l.drawImage(t,0,0,e.width,e.height);const f=o[0],h=f.right-
f.cellRight,y=f.bottom-f.cellBottom;l.fillStyle="rgba(250,204,21,0.18)";const v=[];for(let x=1;x<a.columns;x++){
const _=Math.floor(x*n/a.columns);l.fillRect((_-h)*c,0,2*h*c,e.height),v.push([_*c,0,_*c,e.height])}
for(let x=1;x<a.rows;x++){const _=Math.floor(x*i/a.rows);l.fillRect(0,(_-y)*u,e.width,2*y*u),v.push(
[0,_*u,e.width,_*u])}drawSplitLines(l,v,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const w=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));l.font=`bold ${w}px monospace`,
l.textAlign="center",l.textBaseline="middle";for(const x of o){const _=(x.cellLeft+x.cellRight)/2*c,
S=(x.cellTop+x.cellBottom)/2*u;l.lineWidth=Math.max(2,w/10),l.strokeStyle="rgba(0,0,0,0.7)",l.strokeText(
String(x.index+1),_,S),l.fillStyle="#facc15",l.fillText(String(x.index+1),_,S)}}r(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,o,l,c])=>{e.moveTo(a,o),e.lineTo(l,c)}),e.stroke(),e.restore()}r(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(o=>o?i(
o):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}r(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const o=a.
naturalWidth,l=a.naturalHeight,c=chooseSplitGrid(t.pieces,o,l);if(!splitFits(c,o,l))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const u=splitTiles(o,l,c,t.overlap),f=splitBaseName(e.name),h=[],y=r(async v=>{h.push(await applyImageCompression(
v,n))},"append");if(t.includeOriginal)if(t.markOriginal){const v=splitCanvas(o,l);drawSplitOverview(
v,a,o,l,c,u),await y(await splitCanvasFile(v,`${f}_split_grid.png`))}else await y(e);for(const v of u){
const w=v.right-v.left,x=v.bottom-v.top;let _=Math.max(12,Math.min(40,Math.max(w,x)/45)),S=0;const C=splitTileCaption(
v,c,o,l);if(t.markPieces){const H=splitCanvas(1,1),B=H.getContext("2d");B.font=`bold ${_}px monospac\
e`,_=Math.max(6,_*Math.min(1,w*.96/B.measureText(C).width)),S=Math.max(12,Math.floor(_*1.8)),H.width=
0}const M=splitCanvas(w,x+S),E=M.getContext("2d");if(S&&(E.fillStyle="#111827",E.fillRect(0,0,w,S)),
E.drawImage(a,v.left,v.top,w,x,0,S,w,x),t.markPieces){E.fillStyle="#fff",E.font=`bold ${_}px monospa\
ce`,E.textBaseline="middle",E.fillText(C,w*.02,S/2);const H=[];for(const B of[v.cellLeft,v.cellRight])
B>v.left&&B<v.right&&H.push([B-v.left,S,B-v.left,S+x]);for(const B of[v.cellTop,v.cellBottom])B>v.top&&
B<v.bottom&&H.push([0,S+B-v.top,w,S+B-v.top]);drawSplitLines(E,H,Math.max(1.5,Math.min(4,Math.max(w,
x)/500)),!0)}await y(await splitCanvasFile(M,splitTileName(f,v,c)))}return h}finally{i.release()}}r(
renderSplitFiles,"renderSplitFiles");function clearSplitDownloads(){splitState.urls.forEach(e=>URL.revokeObjectURL(
e)),splitState.urls=[],get("image-split-downloads").replaceChildren()}r(clearSplitDownloads,"clearSp\
litDownloads");function closeImageSplit(){!get("image-split-modal")||splitState.busy||(splitState.revision++,
splitState.release&&splitState.release(),splitState.release=null,splitState.image=null,splitState.files=
[],clearSplitDownloads(),get("image-split-preview").width=0,hideModal("image-split-compression-modal"),
hideModal("image-split-modal"))}r(closeImageSplit,"closeImageSplit");async function openImageSplit(e){
const t=Array.from(e).filter(Boolean);t.length&&(t.length>10&&showToast("\u753B\u50CF\u306E\u5206\u5272\u306F\u4E00\u5EA6\u306B10\u679A\u307E\u3067\u3067\u3059\u3002\u5148\u982D10\u679A\u306E\u307F\u4F7F\u3044\u307E\
\u3059\u3002","warning",!0),!splitState.busy&&(closeImageSplit(),splitState.files=t.slice(0,10),splitState.
index=0,showModal("image-split-modal"),await loadSplitPreview()))}r(openImageSplit,"openImageSplit");
async function loadSplitPreview(){const e=++splitState.revision;splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null,get("image-split-status").textContent="\u8AAD\u307F\u8FBC\u307F\u4E2D...";
try{const t=await loadSplitImage(splitState.files[splitState.index]);if(e!==splitState.revision){t.release();
return}splitState.image=t.image,splitState.release=t.release,get("image-split-status").textContent="",
updateSplitPreview()}catch(t){e===splitState.revision&&(get("image-split-status").textContent=t.message,
updateSplitPreview())}}r(loadSplitPreview,"loadSplitPreview");function updateSplitPreview(){const e=splitOptions(),
t=splitState.image,n=get("image-split-preview");get("image-split-mark-original").disabled=!e.includeOriginal||
splitState.busy,get("image-split-page").textContent=`${splitState.index+1} / ${splitState.files.length}\
\uFF08\u3059\u3079\u3066\u540C\u3058\u8A2D\u5B9A\u3067\u5206\u5272\u3057\u307E\u3059\uFF09`,get("ima\
ge-split-prev").disabled=splitState.busy||splitState.index===0,get("image-split-next").disabled=splitState.
busy||splitState.index>=splitState.files.length-1,get("image-split-count").textContent=`\u4F5C\u6210\u3055\u308C\u308B\u753B\u50CF: ${(e.
pieces+Number(e.includeOriginal))*splitState.files.length}\u679A`;let i=!1;if(t){const a=t.naturalWidth,
o=t.naturalHeight,l=chooseSplitGrid(e.pieces,a,o);i=splitFits(l,a,o);const c=Math.min(1,800/a,520/o);
n.width=Math.round(a*c),n.height=Math.round(o*c),i?drawSplitOverview(n,t,a,o,l,splitTiles(a,o,l,e.overlap)):
n.getContext("2d").drawImage(t,0,0,n.width,n.height),get("image-split-dimensions").textContent=i?`${l.
columns}\u5217 \xD7 ${l.rows}\u884C\u30FB1\u679A\u3042\u305F\u308A\u7D04 ${Math.floor(a/l.columns)}\xD7${Math.
floor(o/l.rows)} px\uFF08\u5143\u753B\u50CF ${a}\xD7${o} px\uFF09`:`\u3053\u306E\u753B\u50CF\u306F\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${e.
pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3002\u5206\u5272\u6570\u3092\u6E1B\u3089\u3057\u3066\u304F\u3060\u3055\u3044\u3002`}else
n.width=0,get("image-split-dimensions").textContent="";get("image-split-save").disabled=get("image-s\
plit-attach").disabled=splitState.busy||!i}r(updateSplitPreview,"updateSplitPreview");function setSplitBusy(e){
splitState.busy=e,get("image-split-modal").querySelectorAll("input,button").forEach(t=>{t.disabled=e}),
updateSplitPreview()}r(setSplitBusy,"setSplitBusy");async function runImageSplit(...e){return window.
ActivityLog?window.ActivityLog.operation("image_split",()=>runImageSplitLogged.apply(this,e)):runImageSplitLogged.
apply(this,e)}r(runImageSplit,"runImageSplit");async function runImageSplitLogged(e){var i;if(splitState.
busy)return;const t=splitOptions(),n=readSplitCompression();if(e&&browserFastModeEnabled&&browserFastLocalFiles.
size+splitState.files.length*(t.pieces+Number(t.includeOriginal))>BROWSER_FAST_MAX_IMAGES){get("imag\
e-split-status").textContent="\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059";
return}setSplitBusy(!0),clearSplitDownloads();try{const a=[];splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null;for(let o=0;o<splitState.files.length;o++)get("image-s\
plit-status").textContent=`\u51E6\u7406\u4E2D... ${o+1} / ${splitState.files.length}`,a.push(...await renderSplitFiles(
splitState.files[o],t,n));if(e){const o=Array.from(browserFastLocalFiles.values()).reduce((l,c)=>l+c.
file.size,0);if(browserFastModeEnabled&&o+a.reduce((l,c)=>l+c.size,0)>BROWSER_FAST_MAX_BYTES)throw new Error(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059");
await handleFiles(a,{skipCompression:!0}),setSplitBusy(!1),closeImageSplit()}else{for(const o of a){
const l=URL.createObjectURL(o);splitState.urls.push(l);const c=document.createElement("a");c.href=l,
c.download=o.name,c.textContent=o.name,c.className="block underline",get("image-split-downloads").append(
c),c.click()}get("image-split-status").textContent=`${a.length}\u679A\u306E\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3057\u307E\u3057\u305F\u3002\u4FDD\u5B58\u3055\u308C\u306A\u3044\u753B\u50CF\u306F\u4E0B\u306E\u30EA\u30F3\u30AF\u304B\u3089\u4FDD\u5B58\u3067\u304D\u307E\
\u3059\u3002`}}catch(a){(i=window.ActivityLog)==null||i.log("image_split.error"),get("image-split-st\
atus").textContent=a.message||"\u753B\u50CF\u3092\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002"}finally{
if(setSplitBusy(!1),splitState.files.length){const a=get("image-split-status").textContent;await loadSplitPreview(),
get("image-split-status").textContent=a}}}r(runImageSplitLogged,"runImageSplitLogged");function openSplitCompression(){
const e=readSplitCompression(),t="image-split-compression";get(`${t}-enabled`).checked=e.enabled,get(
`${t}-output-type`).value=e.outputType,get(`${t}-format-only`).checked=e.formatOnly,get(`${t}-max-si\
ze`).value=e.maxSizeMB,get(`${t}-max-dim`).value=e.maxDimension,syncSplitCompressionUi(),showModal(`${t}\
-modal`)}r(openSplitCompression,"openSplitCompression");function syncSplitCompressionUi(){const e="i\
mage-split-compression";for(const t of["size","dim"]){const n=get(`${e}-format-only`).checked;get(`${e}\
-max-${t}`).disabled=n,get(`${e}-${t}-wrap`).style.opacity=n?"0.4":"1"}}r(syncSplitCompressionUi,"sy\
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
".composer-dock"),a=i==null?void 0:i.getBoundingClientRect().height,o=[t,get("file-preview"),get("qu\
ote-bar")].filter(E=>E&&E.getClientRects().length&&getComputedStyle(E).display!=="none").map(E=>E.getBoundingClientRect()),
l=Math.min(...o.map(E=>E.left)),c=Math.min(...o.map(E=>E.top)),u={left:l,top:c,width:Math.max(...o.map(
E=>E.right))-l,height:Math.max(...o.map(E=>E.bottom))-c},f=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,h={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},y=document.createElement("div");
y.hidden=!0,y.className="prompt-send-held-files",y.setAttribute("aria-hidden","true"),y.append(...h.
rows),document.body.appendChild(y);let v=!1,w=!1,x=!1;const _=r(()=>{v=!0},"onInput");t.addEventListener(
"input",_),t.value="",["file-input","photo-input","mask-input"].forEach(E=>{const H=get(E);H&&(H.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let S;i&&e&&!f&&
i.animate&&(S=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const C=r(()=>{!w||x||(h.files.forEach(E=>{var B,te;const H=(te=
(B=E==null?void 0:E.rowObj)==null?void 0:B.row)==null?void 0:te.getAttribute("data-local-url");H&&URL.
revokeObjectURL(H)}),y.remove())},"releaseFiles"),M=r((E,H=!0)=>{w||(w=!0,E=E&&h.navigation===threadLoadSequence,
x=E,t.removeEventListener("input",_),E?(!v&&!t.value&&(t.value=h.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...h.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=h.mask),h.files.forEach((B,te)=>browserFastLocalFiles.set(te,B)),
h.markers.forEach(B=>markerAppliedUploads.add(B)),n&&n.prepend(...h.rows),y.remove(),!currentQuote&&
h.quote&&(currentQuote=h.quote,currentQuoteMessageId=h.quoteId,get("quote-text-display").innerText=h.
quote,get("quote-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=h.progress),
updateFilePreview(),updateMaskPreview(),schedulePromptTokenEstimate(!0)):H&&C())},"finish");return{source:u,
canRetry:r(()=>!v&&h.navigation===threadLoadSequence,"canRetry"),stopDock:r(()=>S==null?void 0:S.cancel(),
"stopDock"),files:h.files,releaseFiles:C,accept:r((E=!0)=>M(!1,E),"accept"),restore:r(()=>M(!0),"res\
tore")}}r(takePromptForSend,"takePromptForSend");function animatePromptToMessage(e,t,n){var y;let i,
a=0,o=!1;const l=[],c=get("chat-container"),u=get("prompt-input"),f=window.matchMedia("(prefers-redu\
ced-motion: reduce)"),h=r(()=>{var v,w;o||(o=!0,(v=window.ActivityLog)==null||v.log("ui.hide"),t.stopDock(),
cancelAnimationFrame(a),l.forEach(x=>x.cancel()),i==null||i.remove(),e==null||e.classList.remove("pr\
ompt-send-hidden"),window.removeEventListener("resize",h),window.removeEventListener("popstate",h),window.
removeEventListener("pagehide",h),document.removeEventListener("visibilitychange",h),c==null||c.removeEventListener(
"wheel",h),c==null||c.removeEventListener("touchstart",h),u==null||u.removeEventListener("input",h),
(w=window.visualViewport)==null||w.removeEventListener("resize",h),f.removeEventListener("change",h))},
"stop");if(!n||f.matches||!e||!Element.prototype.animate)return h;try{e.classList.remove("fade-in"),
e.classList.add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var v;try{if(!e.isConnected)return h();
const w=e.querySelector(".message-bubble"),x=w.getBoundingClientRect(),_=c.getBoundingClientRect();if(!x.
width||!x.height||x.bottom<=_.top||x.top>=_.bottom)return h();const S=t.source;i=w.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach(te=>te.removeAttribute("id")),i.querySelectorAll(".msg-cont\
rols, video, audio").forEach(te=>te.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute(
"aria-hidden","true"),i.inert=!0,Object.assign(i.style,{width:x.width+"px",height:x.height+"px",left:x.
left+"px",top:x.top+"px"}),document.body.appendChild(i),(v=window.ActivityLog)==null||v.log("ui.show");
const C=Math.min(S.width/x.width,Math.max(S.height,42)/x.height,1),M=S.left+(S.width-x.width*C)/2-x.
left,E=S.top+(S.height-x.height*C)/2-x.top,H=i.animate([{transform:`translate(${M}px, ${E}px) scale(${C}\
)`,borderRadius:"12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"\
translate(0, 0) scale(1)",borderRadius:getComputedStyle(w).borderRadius,backgroundColor:getComputedStyle(
w).backgroundColor}],{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});l.push(H),l.
push(i.animate([{opacity:1,offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),
e.classList.remove("prompt-send-hidden"),l.push(w.animate([{opacity:0,offset:0},{opacity:0,offset:.8},
{opacity:1}],{duration:350,fill:"both"}));const B=r(()=>{if(!e.isConnected||!userAutoScroll||document.
hidden)return h();const te=w.getBoundingClientRect();i.style.left=te.left+"px",i.style.top=te.top+"p\
x",a=requestAnimationFrame(B)},"watch");a=requestAnimationFrame(B),H.finished.then(h,h)}catch{console.
warn("Prompt send animation failed"),h()}}),window.addEventListener("resize",h),window.addEventListener(
"popstate",h),window.addEventListener("pagehide",h),document.addEventListener("visibilitychange",h),
c.addEventListener("wheel",h,{passive:!0}),c.addEventListener("touchstart",h,{passive:!0}),u.addEventListener(
"input",h),(y=window.visualViewport)==null||y.addEventListener("resize",h),f.addEventListener("chang\
e",h)}catch{console.warn("Prompt send animation failed"),h()}return h}r(animatePromptToMessage,"anim\
atePromptToMessage");let costPricing=null,costPricingFailed=!1,costBarPath=[],costBarAll=[],costBarReady=!1;
const COST_ESTIMATE_NOTE="\u6709\u6599API\u306E\u901A\u5E38\u6599\u91D1\u306B\u3088\u308B\u6982\u7B97\uFF08USD\uFF09\u3002\u56DE\u7B54\u3054\u3068\u306E\u4FDD\u5B58\u6E08\u307F\u30E2\u30C7\u30EB\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5165\u529B\u4F7F\u7528\u91CF\u304C\u306A\u3044\u56DE\u7B54\u306F\u4F1A\u8A71\u5C65\u6B74\u304B\u3089\u63A8\u5B9A\u3057\u307E\u3059\u3002\u30E2\u30C7\u30EBAPI\u306E\u547C\u3073\u51FA\u3057\
\u3054\u3068\u306B\u5165\u529B\u30FB\u51FA\u529B\uFF08Thinking\u3092\u542B\u3080\uFF09\u3092\u5408\u7B97\u3057\u307E\u3059\u3002Ideogram 4.5\u306F\u751F\u6210\u30FB\u7DE8\u96C6\u306E\u753B\u50CF\u679A\u6570\u3068\u54C1\u8CEA\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u4F7F\u7528\u91CF\u672A\u53D6\u5F97\u306E\u547C\u3073\u51FA\u3057\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002\u30AD\u30E3\u30C3\u30B7\u30E5\u5272\u5F15\u30FB\u4FDD\u5B58\u6599\u3001\u691C\u7D22\u30FB\u30C4\u30FC\u30EB\u306E\
\u8FFD\u52A0\u6599\u91D1\u3001\u305D\u306E\u4ED6\u306E\u753B\u50CF\u30FB\u97F3\u58F0\u30FB\u52D5\u753B\u751F\u6210\u3001\u7A0E\u3001\u7121\u6599\u67A0\u30FB\u500B\u5225\u5951\u7D04\u306F\u542B\u307F\u307E\u305B\u3093\u3002\u6642\u9593\u5E2F\u3084\u5165\u529B\u7A2E\u5225\u3067\u5358\u4FA1\u304C\u7570\u306A\u308B\u5834\u5408\u306F\u7BC4\u56F2\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u904E\u53BB\u306E\u8ACB\u6C42\u984D\u3092\u518D\u73FE\u3059\u308B\u3082\u306E\u3067\u306F\u3042\u308A\u307E\u305B\u3093\u3002";
function costTokenNumber(e){if(e==null||e==="")return null;const t=Number(e);return Number.isFinite(
t)&&t>=0?t:null}r(costTokenNumber,"costTokenNumber");function costRate(e,t,n){const i=e&&(e.aliases[t]||
t),a=e&&e.models[i];if(!a)return null;const o={...a};for(const l of o.changes||[])n>=l.from&&Object.
assign(o,l);return o}r(costRate,"costRate");function addRequestCost(e,t,n,i,a=!1){var v,w,x,_,S;e.requests++;
const o=costRate(n,t.model,i);if(!o){e.unpriced++;return}if(o.unit==="image"){const C=costTokenNumber(
t.images),M=(w=(v=o.image_rates)==null?void 0:v[t.operation])==null?void 0:w[t.quality];if(C===null||
!Number.isFinite(M)){e.unpriced++;return}e.low+=C*M,e.high+=C*M,e.priced++,t.complete===!1&&e.unpriced++;
return}const l=costTokenNumber(t.input_tokens),c=costTokenNumber(t.output_tokens);if(l===null&&c===null){
e.unpriced++;return}const u=o.threshold!=null&&(o.threshold_inclusive?(l||0)>=o.threshold:(l||0)>o.threshold);
u&&e.long++;const f=u?o.long_input:o.input,h=u?o.long_output:o.output,y=a&&(x=o.batch_multiplier)!=null?
x:1;e.low+=((l||0)*f+(c||0)*h)*y/1e6,e.high+=((l||0)*(u?f:(_=o.max_input)!=null?_:f)+(c||0)*((S=o.max_output)!=
null?S:h))*y/1e6,e.priced++,(l===null||c===null||t.complete===!1)&&e.unpriced++}r(addRequestCost,"ad\
dRequestCost");function legacyImageRequest(e,t){let n=e.image_url;try{typeof n=="string"&&(n=JSON.parse(
n))}catch{n=n?[n]:[]}const i=Array.isArray(n)?n.length:0,a=t.get(String(e.parent_id)),o=(a==null?void 0:
a.prompt_options)||{};let l=a==null?void 0:a.image_url,c=Array.isArray(l)?l.length>0:!!l&&l!=="[]"&&
l!=="null",u=a;const f=new Set;for(;!c&&(u==null?void 0:u.parent_id)!=null&&!f.has(String(u.id));)f.
add(String(u.id)),u=t.get(String(u.parent_id)),(u==null?void 0:u.role)==="assistant"&&u.image_url&&u.
image_url!=="[]"&&u.image_url!=="null"&&(c=!0);const h=["remove_object","remove_background"].includes(
o.ideogram_operation)?o.ideogram_operation:c?"edit":"generate";let y=o.ideogram_quality;return(!["ve\
ry_low","low","medium","high"].includes(y)||y==="very_low"&&h==="generate")&&(y=h==="edit"?"medium":
"high"),{model:e.model,images:i||null,operation:h,quality:y,complete:!!a}}r(legacyImageRequest,"lega\
cyImageRequest");function estimateConversationCost(e,t=costPricing,n=e){var l,c,u,f,h,y;const i={low:0,
high:0,priced:0,unpriced:0,inferred:0,long:0,requests:0},a=new Map((n||[]).filter(Boolean).map(v=>[String(
v.id),v])),o=new Set;for(const v of e||[]){if(!v||v.role!=="assistant"||o.has(String(v.id)))continue;
o.add(String(v.id));let w=null;try{w=typeof v.thought_data=="string"?JSON.parse(v.thought_data):v.thought_data}catch{}
const x=((l=/^\d{4}-\d{2}-\d{2}/.exec(v.created_at||""))==null?void 0:l[0])||(t==null?void 0:t.checked_at),
_=((c=v.batch_job)==null?void 0:c.state)==="JOB_STATE_SUCCEEDED";if(Array.isArray(w==null?void 0:w.pricing_calls)){
w.pricing_calls.length||i.priced++;for(const E of w.pricing_calls){if(!E||typeof E!="object"){i.unpriced++;
continue}const H=((u=/^\d{4}-\d{2}-\d{2}/.exec(E.created_at||""))==null?void 0:u[0])||x;addRequestCost(
i,E,t,H,_)}continue}if(((f=costRate(t,v.model,x))==null?void 0:f.unit)==="image"){addRequestCost(i,legacyImageRequest(
v,a),t,x);continue}const S=costTokenNumber(v.tokens_out);if(S===null&&!v.content&&!v.tokens)continue;
if(S===null){i.unpriced++;continue}let C=(h=costTokenNumber(w==null?void 0:w.pricing_input_tokens))!=
null?h:costTokenNumber(v.tokens_in),M=!1;if(C===null){i.inferred++,C=0;let E=v.parent_id;const H=new Set(
[String(v.id)]);for(E==null&&(M=!0);E!=null;){const B=String(E);if(H.has(B)||!a.has(B)){M=!0;break}H.
add(B);const te=a.get(B),j=te.role==="user"?costTokenNumber(te.tokens_in):(y=costTokenNumber(te.tokens_content))!=
null?y:costTokenNumber(te.tokens_out)===null?null:Math.max(0,Number(te.tokens_out)-(costTokenNumber(
te.tokens_thought)||0));j===null?M=!0:C+=j,E=te.parent_id}}addRequestCost(i,{model:v.model,input_tokens:C,
output_tokens:S,complete:!M},t,x,_)}return i}r(estimateConversationCost,"estimateConversationCost");
function costEstimateLabel(e,t="\u8CBB\u7528\u4E88\u60F3"){if(!e||!e.priced)return`${t}: \u7B97\u51FA\u4E0D\u53EF`;
const n=r(a=>a>0&&a<1e-6?"<$0.000001":`$${a.toFixed(6)}`,"money"),i=e.high-e.low>1e-12?`${n(e.low)}\u301C${n(
e.high)}`:n(e.low);return`${t}: ${i}${e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":""}`}r(
costEstimateLabel,"costEstimateLabel");function refreshCostBar(e=costBarPath,t=costBarAll){costBarReady=
!0,costBarPath=e,costBarAll=t;for(const[n,i,a,o]of[["total-cost-estimate",e,"__total__","\u8CBB\u7528\u4E88\u60F3"],
["total-cost-estimate-all",t,"__total_all_branches__","\u5168\u5206\u5C90\u306E\u8CBB\u7528\u4E88\u60F3"]]){
const l=get(n);if(!l)continue;const c=i.some(f=>f&&f.role==="assistant"&&(f.content||Number(f.tokens_out)>
0));l.classList.toggle("hidden",!c);const u=estimateConversationCost(i,costPricing,t.length?t:e);l.textContent=
costPricing?costEstimateLabel(u,o):`${o}: ${costPricingFailed?"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`,
l.title=COST_ESTIMATE_NOTE,messageMeta[a]&&(messageMeta[a].cost_estimate=costPricing?u:null),l.onclick=
()=>openTokenDetail(a)}}r(refreshCostBar,"refreshCostBar"),fetch("/static/model_pricing.json?checked\
=2026-11-06",{credentials:"same-origin"}).then(e=>{if(!e.ok)throw new Error("pricing_unavailable");return e.
json()}).then(e=>{costPricing=e,costBarReady&&refreshCostBar()}).catch(()=>{costPricingFailed=!0,costBarReady&&
refreshCostBar()});
