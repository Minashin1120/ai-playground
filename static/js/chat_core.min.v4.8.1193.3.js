function buildSvgCodeRenderHtml(e,t){let n=!0;const i=String(e||"").replace(/<svg\b([^>]*)>/i,(o,r)=>{
let c=r;/\sxmlns\s*=/i.test(c)||(c=` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(e)&&!/\sxmlns:xlink\s*=/i.
test(c)&&(c=` xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const d=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return d&&Number(d[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
i)}`;return`<div class="svg-render-box svg-code-render" data-svg-key="${escapeHtml(String(t||""))}">\
<img src="${a}" alt="SVG"${n?' class="svg-code-fill"':""} decoding="async"></div>`}l(buildSvgCodeRenderHtml,
"buildSvgCodeRenderHtml");function getCanvasModeElements(){const e=get("canvas-panel");return e?{panel:e,
stage:get("conversation-stage"),title:get("canvas-panel-title"),status:get("canvas-panel-status"),blockCount:get(
"canvas-block-count"),blockList:get("canvas-block-list"),panelTabs:get("canvas-panel-tabs"),previewLang:get(
"canvas-preview-lang"),sourceSelect:get("canvas-source-select"),frame:get("canvas-preview-frame"),empty:get(
"canvas-preview-empty"),sourceScroll:get("canvas-source-scroll"),code:get("canvas-code-text"),copyBtn:get(
"canvas-panel-copy-btn"),clearBtn:get("canvas-panel-clear-btn"),closeBtn:get("canvas-panel-close-btn")}:
null}l(getCanvasModeElements,"getCanvasModeElements");function isCanvasHtmlPreviewCandidate(e,t){const n=String(
e||"").trim().toLowerCase();if(n==="html"||n==="htm"||n==="xhtml")return!0;if(n)return!1;const i=String(
t||"");return/<!doctype\s+html/i.test(i)||/<html[\s>]/i.test(i)}l(isCanvasHtmlPreviewCandidate,"isCa\
nvasHtmlPreviewCandidate");function normalizeCanvasBlock(e,t){const n=String(e&&e.lang?e.lang:"").trim(),
i=String(e&&e.code!==void 0&&e.code!==null?e.code:""),a=!!(e&&e.open);return{...e,index:t,lang:n,code:i,
open:a,key:hashString(`${n||"TEXT"}
${i||""}`)}}l(normalizeCanvasBlock,"normalizeCanvasBlock");function parseCanvasMarkdown(e){const t=String(
e||""),n=t.split(/\r?\n/),i=[],a=[],o=/^(\s*)(`{3,}|~{3,})(.*)$/;let r=null,c="",d=[];for(const y of n){
if(!r){const x=y.match(o);if(x){r=x[2],c=String(x[3]||"").trim(),d=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
y);continue}const b=String(y||"").trim();if(b&&b.replace(/\s+/g,"")===r){const x=i[i.length-1];x&&(x.
code=d.join(`
`),x.open=!1),r=null,c="",d=[];continue}d.push(y);const w=i[i.length-1];w&&(w.code=d.join(`
`))}if(r&&i.length){const y=i[i.length-1];y&&(y.code=d.join(`
`),y.open=!0)}const m=i.map((y,b)=>normalizeCanvasBlock(y,b)),g=selectCanvasPreviewBlock(m,t);return{
renderText:a.join(`
`),blocks:m,primaryBlock:g?g.block:null,primaryIndex:g?g.index:-1,rawText:t}}l(parseCanvasMarkdown,"\
parseCanvasMarkdown");function selectCanvasPreviewBlock(e,t="",n=-1){const i=Array.isArray(e)?e:[];if(Number.
isInteger(n)&&n>=0&&n<i.length){const o=i[n];return{block:o,index:n,previewType:isCanvasHtmlPreviewCandidate(
o.lang,o.code)?"html":"code"}}if(i.length>0){const o=i.length-1,r=i[o];return{block:r,index:o,previewType:isCanvasHtmlPreviewCandidate(
r.lang,r.code)?"html":"code"}}const a=String(t||"");return isCanvasHtmlPreviewCandidate("",a)?{block:normalizeCanvasBlock(
{lang:"html",code:a,open:!0,fallback:!0},0),index:-1,previewType:"html"}:null}l(selectCanvasPreviewBlock,
"selectCanvasPreviewBlock");function getCanvasSelectedBlock(){const e=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!e.length){const i=String(canvasPreviewState.rawText||"");return isCanvasHtmlPreviewCandidate(
"",i)?{block:normalizeCanvasBlock({lang:"html",code:i,open:!0,fallback:!0},0),index:-1}:null}const t=Number.
isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1,n=selectCanvasPreviewBlock(
e,canvasPreviewState.rawText,t);return!n||!n.block?null:n}l(getCanvasSelectedBlock,"getCanvasSelecte\
dBlock");function syncCanvasPreviewButtons(e=document){if(!e||typeof e.querySelectorAll!="function")
return;const t=String(canvasPreviewState.selectedKey||"");e.querySelectorAll(".canvas-preview-btn").
forEach(n=>{const i=String(n.getAttribute("data-code-key")||""),a=!!t&&t===i;n.classList.toggle("can\
vas-active",a),n.setAttribute("aria-pressed",a?"true":"false"),n.setAttribute("data-canvas-active",a?
"1":"0"),n.innerHTML=a?'<i class="fas fa-layer-group"></i>':'<i class="fas fa-window-restore"></i>',
n.title=a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B",
n.setAttribute("aria-label",a?"Canvas\u3067\u8868\u793A\u4E2D":"Canvas\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3059\u308B")})}
l(syncCanvasPreviewButtons,"syncCanvasPreviewButtons");function isCanvasMobileLayout(){try{return window.
matchMedia("(max-width: 1023px)").matches}catch{return!1}}l(isCanvasMobileLayout,"isCanvasMobileLayo\
ut");function animateCanvasMobileViewEntry(e,t,n){if(!e||!isCanvasMobileLayout()||t===n)return;const i={
preview:get("canvas-preview-shell"),blocks:get("canvas-block-shell"),source:get("canvas-source-shell")},
a={preview:0,blocks:1,source:2},o=i[n];if(!o||!(t in a)||!(n in a))return;canvasPreviewState.viewAnimationToken+=
1;const r=canvasPreviewState.viewAnimationToken;canvasPreviewState.viewAnimationTimer&&(clearTimeout(
canvasPreviewState.viewAnimationTimer),canvasPreviewState.viewAnimationTimer=null),Object.values(i).
forEach(d=>{d&&d.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),o.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";o.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{r===canvasPreviewState.viewAnimationToken&&
(o.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}l(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var r,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const m=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",m),d.setAttribute("aria-pressed",m?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(r=n.blockList).
focus)==null||c.call(r,{preventScroll:!0})))}l(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const o=String(i&&i.lang?i.lang:"text").trim()||"text",r=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",g=(String(i&&i.code?i.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),y=`${r?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${o}`,b=`${y}\u3001${g}`;return`<butto\
n type="button" class="canvas-block-chip${r?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
y)}" aria-label="${escapeHtml(b)}" aria-pressed="${r?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
o)}</span><span class="canvas-block-chip-preview">${escapeHtml(g)}</span></span><span class="canvas-\
block-chip-state">${r?"\u8868\u793A\u4E2D":c}</span></button>`}).join("")}l(renderCanvasBlockChips,"\
renderCanvasBlockChips");function renderCanvasSourceOptions(){const e=getCanvasModeElements();if(!e||
!e.sourceSelect)return;const t=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];
if(!t.length){e.sourceSelect.innerHTML='<option value="">-</option>',e.sourceSelect.disabled=!0,e.sourceSelect.
dataset.canvasOptionsSignature="";return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:t.length-1;e.sourceSelect.disabled=!1;const i=t.map((o,r)=>{const c=String(
o&&o.lang?o.lang:"text").trim()||"text";return`#${r+1} ${c}`}),a=JSON.stringify(i);e.sourceSelect.dataset.
canvasOptionsSignature!==a&&(e.sourceSelect.innerHTML=i.map((o,r)=>`<option value="${r}">${escapeHtml(
o)}</option>`).join(""),e.sourceSelect.dataset.canvasOptionsSignature=a),e.sourceSelect.value=String(
n)}l(renderCanvasSourceOptions,"renderCanvasSourceOptions");function resetCanvasScrollState(){canvasPreviewState.
sourceScrollTop=0,canvasPreviewState.sourceScrollLeft=0,canvasPreviewState.frameScrollX=0,canvasPreviewState.
frameScrollY=0;const e=getCanvasModeElements();e&&e.sourceScroll&&(e.sourceScroll.scrollTop=0,e.sourceScroll.
scrollLeft=0)}l(resetCanvasScrollState,"resetCanvasScrollState");function instrumentCanvasPreviewDocument(e,t){
const n=Math.max(0,Number(canvasPreviewState.frameScrollX)||0),i=Math.max(0,Number(canvasPreviewState.
frameScrollY)||0),a=String(e||""),o=`(function(){const token=${JSON.stringify(t)};let timer=0;functi\
on report(){parent.postMessage({type:'canvas-preview-scroll',token:token,x:window.scrollX||0,y:windo\
w.scrollY||0},'*')}addEventListener('scroll',function(){clearTimeout(timer);timer=setTimeout(report,\
40)},{passive:true});addEventListener('message',function(event){const data=event.data||{};if(data.ty\
pe==='canvas-preview-restore-scroll'&&data.token===token){requestAnimationFrame(function(){scrollTo(\
Number(data.x)||0,Number(data.y)||0);report()})}});requestAnimationFrame(function(){scrollTo(${n},${i}\
);report()})})();`;try{const r=new DOMParser().parseFromString(a,"text/html"),c=r.createElement("scr\
ipt");return c.setAttribute("data-canvas-scroll-bridge","true"),c.textContent=o,(r.body||r.documentElement).
appendChild(c),`<!DOCTYPE html>
`+r.documentElement.outerHTML}catch{return`${a}<script data-canvas-scroll-bridge>${o}<\/script>`}}l(
instrumentCanvasPreviewDocument,"instrumentCanvasPreviewDocument"),window.addEventListener("message",
e=>{const t=e&&e.data?e.data:null;if(!t||t.type!=="canvas-preview-scroll")return;const n=getCanvasModeElements();
!n||!n.frame||e.source!==n.frame.contentWindow||t.token===canvasPreviewState.frameRenderToken&&(canvasPreviewState.
frameScrollX=Math.max(0,Number(t.x)||0),canvasPreviewState.frameScrollY=Math.max(0,Number(t.y)||0))});
function showCanvasPreviewPanel(){const e=getCanvasModeElements();if(!e)return;canvasPreviewState.panelAnimationToken+=
1;const t=canvasPreviewState.panelAnimationToken;canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.
panelHideTimer),canvasPreviewState.panelHideTimer=null),e.panel.classList.remove("hidden","canvas-cl\
osing"),e.stage&&e.stage.classList.add("canvas-enabled"),requestAnimationFrame(()=>{t===canvasPreviewState.
panelAnimationToken&&e.panel.classList.add("canvas-panel-open")})}l(showCanvasPreviewPanel,"showCanv\
asPreviewPanel");function hideCanvasPreviewPanel(e=!0){const t=getCanvasModeElements();if(t){if(canvasPreviewState.
panelAnimationToken+=1,canvasPreviewState.panelHideTimer&&(clearTimeout(canvasPreviewState.panelHideTimer),
canvasPreviewState.panelHideTimer=null),!e){t.panel.classList.add("hidden"),t.panel.classList.remove(
"canvas-panel-open","canvas-closing"),t.stage&&t.stage.classList.remove("canvas-enabled");return}t.panel.
classList.remove("canvas-panel-open"),t.panel.classList.add("canvas-closing"),canvasPreviewState.panelHideTimer=
window.setTimeout(()=>{t.panel.classList.add("hidden"),t.panel.classList.remove("canvas-closing"),t.
stage&&t.stage.classList.remove("canvas-enabled"),canvasPreviewState.panelHideTimer=null},220)}}l(hideCanvasPreviewPanel,
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
en")),t.empty&&t.empty.classList.remove("hidden"),syncCanvasPreviewButtons())}l(resetCanvasPreviewPanel,
"resetCanvasPreviewPanel");function updateCanvasPreviewState(e=null){const t=e||canvasPreviewState.lastCanvasData;
if(!t)return null;canvasPreviewState.lastCanvasData=t,canvasPreviewState.blocks=Array.isArray(t.blocks)?
t.blocks.slice():[],canvasPreviewState.rawText=String(t.rawText||""),canvasPreviewState.renderText=String(
t.renderText||"");const n=canvasPreviewState.blocks,i=Number.isInteger(canvasPreviewState.selectedIndex)?
canvasPreviewState.selectedIndex:-1;if(!n.length){const r=selectCanvasPreviewBlock([],canvasPreviewState.
rawText);return r&&r.block?(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey=r.block.
key||"",r.block):(canvasPreviewState.selectedIndex=-1,canvasPreviewState.selectedKey="",canvasPreviewState.
selectionMode="auto",i!==-1&&resetCanvasScrollState(),null)}let a=n.length-1;canvasPreviewState.selectionMode===
"manual"&&i>=0&&i<n.length?a=i:canvasPreviewState.selectionMode="auto";const o=n[a]||null;return canvasPreviewState.
selectedIndex=o?a:-1,canvasPreviewState.selectedKey=o&&o.key?o.key:"",i!==canvasPreviewState.selectedIndex&&
resetCanvasScrollState(),o}l(updateCanvasPreviewState,"updateCanvasPreviewState");function refreshCanvasPreviewPanel(){
const e=getCanvasModeElements();if(!e||!canvasModeEnabled)return;showCanvasPreviewPanel(),syncCanvasPanelViewUi(
canvasPreviewState.mobileView||"preview",{focus:!1});const t=Array.isArray(canvasPreviewState.blocks)?
canvasPreviewState.blocks:[],n=getCanvasSelectedBlock(),i=n&&n.block?n.block:null,a=n&&Number.isInteger(
n.index)?n.index:-1,o=!!i,r=String(i&&i.lang?i.lang:"").trim(),c=String(i&&i.code!==void 0&&i.code!==
null?i.code:""),d=o?isCanvasHtmlPreviewCandidate(r,c):!1,m=o?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",g=o?d?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${r||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=g),e.status&&(e.status.textContent=
m),e.previewLang&&(e.previewLang.textContent=o?r||"text":"idle");const y=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,b=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=y,e.sourceScroll.
scrollLeft=b,canvasPreviewState.sourceScrollTop=e.sourceScroll.scrollTop,canvasPreviewState.sourceScrollLeft=
e.sourceScroll.scrollLeft),e.blockCount&&(e.blockCount.textContent=String(t.length)),renderCanvasBlockChips(),
renderCanvasSourceOptions(),o){canvasPreviewState.frameRenderToken+=1;const w=canvasPreviewState.frameRenderToken,
x=instrumentCanvasPreviewDocument(buildCanvasPreviewDocument(i),w);e.frame&&(e.frame.srcdoc=x,e.frame.
classList.remove("hidden"),e.frame.addEventListener("load",()=>{w!==canvasPreviewState.frameRenderToken||
!e.frame.contentWindow||e.frame.contentWindow.postMessage({type:"canvas-preview-restore-scroll",token:w,
x:canvasPreviewState.frameScrollX,y:canvasPreviewState.frameScrollY},"*")},{once:!0})),e.empty&&e.empty.
classList.add("hidden")}else e.frame&&(e.frame.removeAttribute("srcdoc"),e.frame.classList.add("hidd\
en")),e.empty&&e.empty.classList.remove("hidden");syncCanvasPreviewButtons()}l(refreshCanvasPreviewPanel,
"refreshCanvasPreviewPanel");function applyCanvasSelection(e,t={}){const n=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(!n.length)return!1;const i=Number(e);if(!Number.isInteger(i)||
i<0||i>=n.length)return!1;const a=canvasPreviewState.selectedIndex!==i;return canvasPreviewState.selectedIndex=
i,canvasPreviewState.selectedKey=n[i]&&n[i].key?n[i].key:"",canvasPreviewState.selectionMode="manual",
a&&resetCanvasScrollState(),syncCanvasPanelViewUi(t.view||"preview",{focus:!1,animate:t.animateView===
!0,fromView:t.transitionFrom}),renderCanvasBlockChips(),syncCanvasPreviewButtons(),refreshCanvasPreviewPanel(),
!0}l(applyCanvasSelection,"applyCanvasSelection");function applyCanvasSelectionByKey(e){const t=Array.
isArray(canvasPreviewState.blocks)?canvasPreviewState.blocks:[];if(!t.length)return!1;const n=String(
e||"");if(!n)return!1;const i=t.findIndex(a=>a&&a.key===n);return i===-1?!1:applyCanvasSelection(i)}
l(applyCanvasSelectionByKey,"applyCanvasSelectionByKey");function decodeCanvasPreviewButtonCode(e){if(!e)
return null;const t=e.getAttribute("data-code")||"";if(!t)return null;let n="";try{n=decodeURIComponent(
t)}catch{n=t}const i=String(e.getAttribute("data-canvas-lang")||e.getAttribute("data-lang")||"").trim(),
a=String(e.getAttribute("data-code-key")||hashString(`${i||"TEXT"}
${n||""}`));return{code:n,lang:i,codeKey:a}}l(decodeCanvasPreviewButtonCode,"decodeCanvasPreviewButt\
onCode");function collectCanvasBlocksFromButton(e){const t=decodeCanvasPreviewButtonCode(e);if(!t)return null;
const n=e&&typeof e.closest=="function"?e.closest(".message-group"):null,i=n?Array.from(n.querySelectorAll(
".canvas-preview-btn")):[];if(!i.length){const c=normalizeCanvasBlock({lang:t.lang,code:t.code,open:!1},
0);return{blocks:[c],selectedIndex:0,selectedKey:c.key||t.codeKey||""}}const a=[];let o=-1;if(i.forEach(
(c,d)=>{const m=decodeCanvasPreviewButtonCode(c);if(!m)return;const g=normalizeCanvasBlock({lang:m.lang,
code:m.code,open:!1},d);a.push(g),o===-1&&m.codeKey===t.codeKey&&(o=a.length-1)}),!a.length)return null;
o===-1&&(o=0);const r=a[o]||a[0]||null;return{blocks:a,selectedIndex:o,selectedKey:r&&r.key?r.key:t.
codeKey||""}}l(collectCanvasBlocksFromButton,"collectCanvasBlocksFromButton");function previewCanvasCodeFromButton(e){
var o;if((o=window.ActivityLog)==null||o.log("html_preview.start"),!e)return!1;const t=collectCanvasBlocksFromButton(
e);if(!t||!t.blocks||!t.blocks.length)return!1;const n=Array.isArray(canvasPreviewState.blocks)?canvasPreviewState.
blocks:[],i=n.findIndex(r=>r&&r.key===t.selectedKey);if(i!==-1&&n.length>1)return applyCanvasSelection(
i);const a=t.blocks[t.selectedIndex]||t.blocks[0]||null;return canvasPreviewState.blocks=t.blocks,canvasPreviewState.
rawText=a&&a.code!==void 0&&a.code!==null?String(a.code):"",canvasPreviewState.renderText=canvasPreviewState.
rawText,canvasPreviewState.selectedIndex=Number.isInteger(t.selectedIndex)?t.selectedIndex:0,canvasPreviewState.
selectedKey=t.selectedKey||a&&a.key||"",canvasPreviewState.selectionMode="manual",resetCanvasScrollState(),
canvasPreviewState.lastCanvasData={renderText:canvasPreviewState.renderText,blocks:t.blocks,primaryBlock:a,
primaryIndex:canvasPreviewState.selectedIndex,rawText:canvasPreviewState.rawText},canvasPreviewState.
mobileView="preview",syncCanvasPanelViewUi("preview",{focus:!1}),refreshCanvasPreviewPanel(),!0}l(previewCanvasCodeFromButton,
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
'<span class="muted">Canvas\u3067\u8868\u793A\u4E2D</span>'}</pre></div></body></html>`}l(buildCanvasPreviewDocument,
"buildCanvasPreviewDocument");function syncCanvasModeUi(e=canvasModeEnabled,t={}){const n=t.persist!==
!1;if(canvasModeEnabled=!!e,n)try{localStorage.setItem(CANVAS_MODE_STORAGE_KEY,canvasModeEnabled?"tr\
ue":"false")}catch{}const i=get("enable-canvas-mode");if(i&&i.checked!==canvasModeEnabled&&(i.checked=
canvasModeEnabled),!canvasModeEnabled){if(hideCanvasPreviewPanel(t.animate!==!1),!activeStreamingBubbleId&&
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}return}if(showCanvasPreviewPanel(),
isCanvasMobileLayout()&&syncCanvasPanelViewUi("preview",{focus:!1}),syncCanvasPanelViewUi(canvasPreviewState.
mobileView||"preview",{focus:!1}),!t.skipReset){if(activeStreamingBubbleId)refreshCanvasPreviewPanel();else if(resetCanvasPreviewPanel(),
currentThreadId)try{renderThreadTree({silent:!0,keepScroll:!0})}catch{}}}l(syncCanvasModeUi,"syncCan\
vasModeUi");function normalizeMarkdownNewlines(e){return String(e||"").replace(/\r\n/g,`
`).replace(/\r/g,`
`)}l(normalizeMarkdownNewlines,"normalizeMarkdownNewlines");function stripExactFencedBlock(e,t,n){let i=normalizeMarkdownNewlines(
e);const a=normalizeMarkdownNewlines(n);if(!a&&a!=="")return i;const o=t?[String(t),""]:[""];for(const r of[
"`","~"])for(let c=3;c<=10;c++){const d=r.repeat(c);for(const m of o){const g=`${d}${m}
`,y=`
${d}`,b=g+a+y;i.includes(b)&&(i=i.split(b).join(""))}}return i}l(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const o of a)for(const r of["`","~"])for(let c=3;c<=10;c++){const d=r.repeat(c);
[`${o}${d}
${i}
${d}`,`${o}
${d}
${i}
${d}`,`
${o}${d}
${i}
${d}`,`
${o}
${d}
${i}
${d}`].forEach(g=>{n.includes(g)&&(n=n.split(g).join(`
`))})}return n}l(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}l(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}l(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[],i=[];if(!t)return{text:"",executions:n};const a=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let o=t.replace(a,(b,w,x,k)=>{i.push(k);const C=String(x||"").trim();try{const T=JSON.parse(C);n.push(
{code:T&&T.code!=null?String(T.code):"",output:T&&T.output!=null?String(T.output):"",images:T&&Array.
isArray(T.images)?T.images.filter(A=>typeof A=="string"):[]})}catch{n.push({code:C,output:""})}return`\

`});const r=/!\[Agentic View\]\(([^)]+)\)/g;if(!n.length&&/!\[Agentic View\]\(/.test(t)&&(o=t.replace(
/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,(b,w,x,k)=>(i.push(
k),n.push({code:w,output:x||"",images:[]}),`
`))),n.length){const b=new Set(n.flatMap(w=>w.images||[]));for(const w of t.matchAll(r)){if(b.has(w[1]))
continue;let x=i.length-1;for(;x>0&&i[x]>w.index;)x--;const k=n[x];k.images=Array.from(new Set([...k.
images||[],w[1]]))}o=o.replace(r,"")}n.forEach(b=>{b.code&&(o=stripExactFencedBlock(o,"python",b.code),
o=stripExactFencedBlock(o,"py",b.code)),o=stripVisiblePythonOutputBlock(o,b.output)});const c=new Set(
n.flatMap(b=>b.images||[]));o=o.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
(b,w,x,k)=>!w&&c.has(k)&&/\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(k)?`![${x}](${k})`:b);const d=n.
slice().reverse().find(b=>b.images&&b.images.length),m=o.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,
""),g=new Set(Array.from(m.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),b=>b[1])),y=(d?d.images:[]).filter(
b=>/^\/files\/[A-Za-z0-9_./%-]+$/.test(b)&&!g.has(b));return y.length&&(o+=`

**Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF**

`+y.map(b=>`![Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF](${b})`).join(`

`)),o=o.replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),{text:o,executions:n}}l(extractPythonExecutionsFromContent,
"extractPythonExecutionsFromContent");function extractMcpExecutionNotesFromContent(e){const t=normalizeMarkdownNewlines(
e),n=[];if(!t)return{text:"",notes:n};const i=[];return t.split(`
`).forEach(o=>{/^>\s*(?:🔧|🚫)\s*\*\*MCPツール実行(?:[:：]|は|（)/.test(o)?n.push(o.trim()):
i.push(o)}),{text:i.join(`
`).replace(/[ \t]+\n/g,`
`).replace(/\n{3,}/g,`

`).replace(/^\n+/,"").replace(/\n+$/,""),notes:n}}l(extractMcpExecutionNotesFromContent,"extractMcpE\
xecutionNotesFromContent");function appendMcpExecutionNotes(e,t){const n=String(e||"").trim(),i=Array.
isArray(t)?t.filter(Boolean):[];return i.length?n?`${n}

${i.join(`
`)}`:i.join(`
`):n}l(appendMcpExecutionNotes,"appendMcpExecutionNotes");function buildPythonExecDetailBoxHtml(e,t,n){
const i=e&&e.code!=null?String(e.code):"",a=e&&e.output!=null?String(e.output):"";let o="";try{window.
hljs&&typeof window.hljs.highlight=="function"?o=window.hljs.highlight(i,{language:"python"}).value:
o=escapeHtml(i)}catch{o=escapeHtml(i)}const r=(e.images||[]).filter(k=>/^\/files\/[A-Za-z0-9_./%-]+$/.
test(k)).map(k=>`![Python output](${k})`).join(`
`),c=r?`<div class="python-section"><div class="python-label">Images</div>${sanitizeMarkdownHtml(marked.
parse(r))}</div>`:"",d=escapeHtml(a),m=encodeURIComponent(i).replace(/'/g,"%27"),g=encodeURIComponent(
a).replace(/'/g,"%27"),y=hashString(`pyexec-detail
${i}
${a}
${t}`),b=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",w=`<button class="download-btn" data-\
code="${m}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,x=`<button class="coding-target-btn" data-code="${m}" data-code-key="${y}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${y}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(b)}</span><div class="code-actions">${x}${w}<button class="copy-btn" data-copy="code" \
data-code="${m}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${g}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${o}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${d}</code></pre></div>${c}</div></div>`}l(buildPythonExecDetailBoxHtml,"bu\
ildPythonExecDetailBoxHtml");function showPythonExecDetailModal(e=null){if(location.pathname!=="/pyt\
hon-execution"){const t={modal:"python-execution"};e!==null&&(t.messageId=e),history.pushState(t,"",
"/python-execution")}showModal("python-exec-modal")}l(showPythonExecDetailModal,"showPythonExecDetai\
lModal");function openPythonExecDetail(e){const t=messageMeta[e],n=get("python-exec-modal"),i=get("p\
ython-exec-modal-body"),a=get("python-exec-modal-title");if(!n||!i)return;const o=t&&Array.isArray(t.
python_executions)?t.python_executions:[];if(!o.length){showToast("Python\u5B9F\u884C\u7D50\u679C\u304C\u3042\u308A\u307E\u305B\u3093",
"info",!1);return}if(a){const r=o.length>1?`\uFF08${o.length}\u4EF6\uFF09`:"";a.textContent=`Python \
\u5B9F\u884C\u7D50\u679C${r}`}i.innerHTML=o.map((r,c)=>buildPythonExecDetailBoxHtml(r,c,o.length)).join(
""),i.querySelectorAll("img").forEach(r=>{r.addEventListener("click",()=>openImageViewer(r.dataset.viewerSrc||
r.src,"#python-exec-modal-body img",r))}),codingModeEnabled&&(syncCodingTargetButtons(i),syncCodingModeUi(
!0,{persist:!1})),showPythonExecDetailModal(e)}l(openPythonExecDetail,"openPythonExecDetail"),window.
openPythonExecDetail=openPythonExecDetail;function closePythonExecDetail(e=!1){get("python-exec-moda\
l")&&(hideModal("python-exec-modal"),!e&&location.pathname==="/python-execution"&&history.back())}l(
closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;function buildAiMarkdownHtml(e,t={}){
const n=extractMcpExecutionNotesFromContent(e),i=appendMcpExecutionNotes(n.text,n.notes);if(typeof window.
buildMathStepsBubbleHtml=="function"){const r=window.buildMathStepsBubbleHtml(i,t);if(r)return!lowBandwidthMode&&
maybeNeedsMathJax(i)&&ensureMathJaxLoaded().catch(()=>{}),r}const a=canvasModeEnabled?parseCanvasMarkdown(
i):{renderText:i,blocks:[],primaryBlock:null,rawText:i};canvasModeEnabled&&(updateCanvasPreviewState(
a),refreshCanvasPreviewPanel());const o=document.createElement("div");return o.className="prose pros\
e-invert text-sm break-words",o.innerHTML=sanitizeMarkdownHtml(a.renderText),wrapRenderedSvgBoxes(o),
lowBandwidthMode||(maybeNeedsHighlight(a.renderText,o)&&ensureHighlightLoaded().catch(()=>{}),maybeNeedsMathJax(
a.renderText)&&ensureMathJaxLoaded().catch(()=>{})),o.outerHTML}l(buildAiMarkdownHtml,"buildAiMarkdo\
wnHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes);if(typeof window.renderMathStepsInto=="function"&&window.
renderMathStepsInto(e,a,n)){lowBandwidthMode||maybeNeedsMathJax(a)&&ensureMathJaxLoaded().catch(()=>{});
return}const o=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,blocks:[],primaryBlock:null,rawText:a};
if(canvasModeEnabled&&(updateCanvasPreviewState(o),refreshCanvasPreviewPanel()),n.incrementalMath){const c=document.
createElement("template");c.innerHTML=sanitizeMarkdownHtml(o.renderText,{streamMathSegments:!0});const d=new Map;
e.querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(g=>{const y=g.getAttribute(
"data-stream-math-key");y&&d.set(y,g)});const m=[];c.content.querySelectorAll(".stream-math-segment[\
data-stream-math-key]").forEach(g=>{const y=d.get(g.getAttribute("data-stream-math-key"));y?g.replaceWith(
y):m.push(g)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(e)),e.replaceChildren(c.content),
wrapRenderedSvgBoxes(e),queueHighlight(e,o.renderText),queueIncrementalMathTypeset(m);return}const r=collectSvgCodeRenders(
e);e.innerHTML=sanitizeMarkdownHtml(o.renderText),restoreSvgCodeRenders(e,r),wrapRenderedSvgBoxes(e),
queueMessageDecorations(e,o.renderText)}l(renderAiMarkdownInto,"renderAiMarkdownInto");function collectSvgCodeRenders(e){
const t=new Map;return!e||typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-rende\
r[data-svg-key]").forEach(n=>{const i=n.getAttribute("data-svg-key");i&&!t.has(i)&&t.set(i,n)}),t}l(
collectSvgCodeRenders,"collectSvgCodeRenders");function restoreSvgCodeRenders(e,t){!e||!t||!t.size||
typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-render[data-svg-key]").forEach(
n=>{const i=n.getAttribute("data-svg-key"),a=i?t.get(i):null;a&&(t.delete(i),n.replaceWith(a))})}l(restoreSvgCodeRenders,
"restoreSvgCodeRenders");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}l(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");const SEARCH_COMPLETE_HTML='<div class="search-box visible mb-2"><i class="fas fa-chec\
k-circle text-green-400"></i> Search complete</div>';function updateSearchBox(e,t,n){return n!=="sea\
rching"&&n!=="done"||(t||(e.insertAdjacentHTML("afterbegin",SEARCH_COMPLETE_HTML),t=e.querySelector(
".search-box")),t.classList.toggle("animate-pulse",n==="searching"),t.innerHTML=n==="searching"?'<i \
class="fas fa-globe"></i> Searching web...':'<i class="fas fa-check-circle text-green-400"></i> Sear\
ch complete'),t}l(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,o,r=null,c=!0,d=null,m=null,g=null,y=null,b=null,w=null,x=null,k=null,C=!0,T=null,A=null,N=null,G=null,L=null,$=null){
const M=t==="user",P=M?"bg-blue-600":"bg-gray-700",R=M?"justify-end":"justify-start";messageStore[e]=
n;const D=!M&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},H=M?n:D.text;let ee=m;
if(ee==null){const fe=g!=null?Number(g):0,ye=y!=null?Number(y):0;(g!=null||y!=null)&&(ee=fe+ye)}const O=costCacheUsage(
a);messageMeta[e]={tokens_in:g,tokens_out:y,tokens_total:ee,tokens_content:w,tokens_thought:x,cache_read_tokens:O?
O.read:null,cache_write_tokens:O?O.write:null,cache_write_ttls:O?O.ttls:null,is_encrypted:b,role:t,model:o,
parent_id:T,quote_text:d,image_url:i,image_mask:G,skill_name:A,batch_job:N,thought_data:a,processing_time:L,
python_executions:M?[]:D.executions||[]};let B="";d&&(B=`<div class="mb-2 p-2 bg-black/20 rounded bo\
rder-l-4 border-blue-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-le\
ft mr-1 opacity-50"></i>${escapeHtml(d)}</div>`);const J=A?`<div class="skill-notice visible"><i cla\
ss="fas fa-wand-magic-sparkles"></i><span>Skill: <span class="font-bold">${escapeHtml(A)}</span></sp\
an></div>`:"";let F="";if(a&&!M){let fe="";try{const ye=JSON.parse(a);fe=ye.text||"",ye.search_used===
!0&&(F=SEARCH_COMPLETE_HTML)}catch{fe=a}fe&&(F+=`<div class="thought-container"><div class="thought-\
header" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process\
</div><div class="thought-content collapsed">${escapeHtml(fe)}</div></div>`)}let q="";const te=new Set;
if(!M){const fe=String(H||"").replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,""),ye=Array.
from(fe.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),ge=>ge[1]);ye.push(...(D.executions||[]).flatMap(ge=>ge.
images||[])),ye.forEach(ge=>{if(/^https?:\/\//i.test(ge)&&!ge.startsWith(FILE_BASE_URL))try{if(new URL(
ge).origin!==window.location.origin)return}catch{return}const ie=normalizeAttachmentPath(ge);ie&&te.
add(ie)})}if(i)try{const fe=JSON.parse(i);if(fe.length){const ye=[];if(fe.forEach(ge=>{let ie=ge,it="\
unknown";if(ie&&typeof ie=="object"&&(it=normalizeAttachmentSource(ie.source),ie=ie.filepath||ie.path||
ie.url||ie.file||""),ie=normalizeAttachmentPath(ie)||ie,!ie||!M&&te.has(ie))return;setAttachmentSourceForPath(
ie,it);const st=ie.replace(/^\d+\//,""),vt=buildFileUrl(st),Mt=buildAttachmentPreviewUrl(st),He=ie.split(
"/").pop(),at=He.split(".").pop().toLowerCase();["jpg","jpeg","png","webp","gif"].includes(at)?ye.push(
buildChatImageHtml(Mt,{viewerSrc:vt,alt:He,title:He,filename:He})):ye.push(`<div class="file-thumb b\
g-gray-800 border border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer h\
over:bg-gray-700" onclick="window.open('${vt}')" title="${He}"><i class="fas fa-file text-2xl text-g\
ray-400 mb-1"></i><span class="text-[9px] truncate w-20 text-center">${He}</span></div>`)}),ye.length>
0){let ge="grid-multi";ye.length===1?ge="grid-1":ye.length===2?ge="grid-2":ye.length===3?ge="grid-3":
ye.length===4&&(ge="grid-4"),q=`<div class="image-grid ${ge}">${ye.join("")}</div>`}}}catch{}let V="";
if(M&&G)try{const fe=normalizeAttachmentPath(G)||String(G),ye=fe.replace(/^\d+\//,""),ge=buildFileUrl(
ye),ie=buildAttachmentPreviewUrl(ye),it=fe.split("/").pop();V=`<div class="mt-2 max-w-xs rounded-lg \
border border-amber-300/40 bg-black/10 p-2"><div class="mb-1 text-[10px] font-semibold text-amber-10\
0"><i class="fas fa-mask mr-1" aria-hidden="true"></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(
ie,{viewerSrc:ge,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:it})}</div>`}catch{}const X=M?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,re=`<div class="msg-controls absolute -top-5 right\
-0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', \
this)"><i class="fas fa-copy"></i></button>${M?`<button class="ctrl-btn edit-btn" data-id="${e}"><i \
class="fas fa-pen"></i></button>`:""}${X}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\
\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,ce=[];!M&&o&&ce.push(escapeHtml(o));const Q=[];
if(g!=null&&Q.push(`In ${g}`),y!=null){let fe=`Out ${y}`;x!=null&&Number(x)>0&&(fe+=` (Thought ${x})`),
Q.push(fe)}if(Q.length||m!=null){const fe=Q.length?Q.join(" / "):`${m} tokens`;ce.push(`<button clas\
s="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${e}')">${fe}\
</button>`)}if(!M&&(n||y!=null&&Number(y)>0||a)){const ge=typeof costPricing!="undefined"&&costPricing?
"\u8CBB\u7528\u4E88\u60F3: \u8A08\u7B97\u4E2D\u2026":`\u8CBB\u7528\u4E88\u60F3: ${typeof costPricingFailed!=
"undefined"&&costPricingFailed?"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`;
ce.push(`<button class="underline decoration-dotted hover:text-white msg-cost-estimate-btn" data-msg\
-id="${escapeHtml(String(e))}" onclick="openTokenDetail('${e}')">${ge}</button>`)}if(b!=null){const fe=b?
"fa-lock":"fa-lock-open",ye=isAdminUser?b?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":b?"Encrypted":
"Plain",ge=isAdminUser?b?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-\
200":"text-slate-300/80 hover:text-white";ce.push(`<button class="${ge}" title="${ye}" onclick="open\
EncryptionSettings('${e}')"><i class="fas ${fe}"></i></button>`)}if(!M&&D.executions&&D.executions.length){
const fe=D.executions.length,ye=fe>1?`Python \xD7${fe}`:"Python";ce.push(`<button type="button" clas\
s="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label="Python\u5B9F\
\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${ye}</span></button>`)}const ue=ce.
length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono message-footer-meta">${ce.
join(" \u2022 ")}</div>`:"",ne=M?"":buildProcessingTimeHtml(e,L);let xe;const ve=!M&&N?(()=>{const fe=String(
N.state||"").toUpperCase(),ye=N.status_text||(fe==="JOB_STATE_SUCCEEDED"?"Batch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":
fe==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":"Batch API\u3067\u51E6\u7406\u4E2D\
\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-lg border ${fe==="JOB_STATE_FAILED"||
fe==="JOB_STATE_CANCELLED"||fe==="JOB_STATE_EXPIRED"?"border-red-400/40 bg-red-950/30 text-red-100":
fe==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950/30 text-emerald-100":"border-violet\
-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><div class="font-semibold"><i class="f\
as fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-90">${escapeHtml(ye)}</div></div>`})():
"";M?xe=`<div class="content-area whitespace-pre-wrap font-sans text-sm break-words">${escapeHtml(n||
"")}</div>`:(xe=ve+(H&&String(H).trim()?buildAiMarkdownHtml(H,{mathSteps:{messageId:e,state:$}}):N?'\
<div class="content-area prose prose-invert text-sm break-words text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':
buildAiMarkdownHtml(H,{mathSteps:{messageId:e,state:$}})),xe.includes("content-area")||(xe=xe.replace(
"prose ","content-area prose ")));let we="";if(r){const fe=r.siblings[r.current-2],ye=r.siblings[r.current];
we=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${fe}\
)" ${fe?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${r.current} / ${r.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${ye}\
)" ${ye?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const Oe=c?"fade-in":"",Ce=document.createElement("div");Ce.className=`flex ${R} m\
b-4 ${Oe} relative message-group group`,Ce.id=`msg-${e}`,Ce.innerHTML=`<div class="message-bubble ${P}\
 text-white p-4 rounded-2xl shadow-md relative">${re}${B}${J}${F}${xe}${q}${V}${we}${ue}${ne}</div>`;
const lt=k||get("chat-container");if(lt&&(lt.appendChild(Ce),C&&scrollToBottom(),!M)){const fe=Ce.querySelector(
".message-bubble");if(fe&&(fe.setAttribute("data-message-id",String(e)),$!=null))try{fe.setAttribute(
"data-math-steps-state",typeof $=="string"?$:JSON.stringify($))}catch{}Ce.isConnected&&queueMessageDecorations(
Ce,H),syncCodingTargetButtons(Ce),syncCodingModeUi(codingModeEnabled,{persist:!1})}return Ce}l(renderMessage,
"renderMessage");const PROCESSING_PHASE_LABELS={send:"\u9001\u4FE1\u301C\u53D7\u4ED8",queue:"\u5F85\u3061\u884C\u5217",
provider:"\u521D\u56DE\u5FDC\u7B54\u307E\u3067",generation:"\u751F\u6210",receive:"\u53D7\u4FE1\u30FB\u8868\u793A"};
function formatProcessingDuration(e){const t=Number(e);if(!Number.isFinite(t)||t<0)return"-";if(t<6e4)
return`${(t/1e3).toFixed(2)}\u79D2`;const n=t/1e3,i=Math.floor(n/60);return`${i}\u5206${(n-i*60).toFixed(
1)}\u79D2`}l(formatProcessingDuration,"formatProcessingDuration");function buildProcessingTimeHtml(e,t){
if(!t||t.total_ms===null||t.total_ms===void 0)return"";const n=formatProcessingDuration(t.total_ms);
return`<div class="mt-1 text-right font-mono text-[10px] text-slate-400/80 processing-time-row"><but\
ton type="button" class="underline decoration-dotted hover:text-white processing-time-btn" onclick="\
openProcessingTimeDetail('${e}')" title="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33" aria-label="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33">\u51E6\u7406\u6642\u9593 ${escapeHtml(
n)}</button></div>`}l(buildProcessingTimeHtml,"buildProcessingTimeHtml");function openProcessingTimeDetail(e){
const t=messageMeta[e],n=t&&t.processing_time;if(!n||n.total_ms===null||n.total_ms===void 0||!get("p\
rocessing-time-modal"))return;get("processing-time-total").textContent=formatProcessingDuration(n.total_ms);
const a=get("processing-time-first-row");if(a){const d=n.first_response_ms!==null&&n.first_response_ms!==
void 0;a.classList.toggle("hidden",!d),d&&get("processing-time-first")&&(get("processing-time-first").
textContent=formatProcessingDuration(n.first_response_ms))}const o=get("processing-time-phases");if(o){
o.replaceChildren();const d=Array.isArray(n.phases)?n.phases:[];for(const m of d){if(!m||m.ms===null||
m.ms===void 0)continue;const g=PROCESSING_PHASE_LABELS[m.key]||m.key,y=document.createElement("div");
y.className="flex justify-between gap-4";const b=document.createElement("span");b.textContent=g;const w=document.
createElement("span");w.className="font-mono",w.textContent=formatProcessingDuration(m.ms),y.appendChild(
b),y.appendChild(w),o.appendChild(y)}o.classList.toggle("hidden",o.childElementCount===0)}const r=get(
"processing-time-note");r&&(n.source==="client"?r.textContent="\u56DE\u7B54\u5B8C\u4E86\u6642\u306B\u7AEF\u672B\u3067\u8A18\u9332\u3057\u305F\u6642\u9593\u3067\u3059\u3002\u8A73\u7D30\u306A\u5185\u8A33\u306F\u5C65\u6B74\u306E\u518D\u8AAD\u307F\u8FBC\u307F\u5F8C\u306B\u8868\u793A\u3055\
\u308C\u307E\u3059\u3002":n.phases&&n.phases.length?r.textContent="\u8A18\u9332\u3067\u304D\u305F\u51E6\u7406\u533A\u9593\u306E\u5185\u8A33\u3067\u3059\u3002":
r.textContent="\u3053\u306E\u56DE\u7B54\u3067\u306F\u5408\u8A08\u6642\u9593\u3060\u3051\u3092\u8A18\u9332\u3057\u3066\u3044\u307E\u3059\u3002");
const c=get("processing-time-title");c&&(c.textContent=t&&t.model?`${t.model} (${t.role||"assistant"}\
)`:"\u56DE\u7B54\u51E6\u7406\u6642\u9593"),showModal("processing-time-modal")}l(openProcessingTimeDetail,
"openProcessingTimeDetail");function closeProcessingTimeDetail(){hideModal("processing-time-modal")}
l(closeProcessingTimeDetail,"closeProcessingTimeDetail");function showTokenDetailModal(e=null,t=!1){
if(location.pathname!=="/token-details"){const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==
null&&(n.messageId=e),history.pushState(n,"","/token-details")}showModal("token-detail-modal")}l(showTokenDetailModal,
"showTokenDetailModal");function openTokenDetail(e){const t=messageMeta[e];if(!t||!get("token-detail\
-modal"))return;get("token-detail-usage").classList.remove("hidden"),get("token-detail-estimate").classList.
add("hidden"),cancelPromptTokenDetail();const i=t.tokens_total!==null&&t.tokens_total!==void 0?t.tokens_total:
"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",o=t.tokens_out!==null&&t.tokens_out!==
void 0?t.tokens_out:"-",r=t.tokens_content!==null&&t.tokens_content!==void 0?t.tokens_content:"-",c=t.
tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===null||t.is_encrypted===
void 0?"-":t.is_encrypted?"Encrypted":"Plain",m=t.cache_read_tokens!==null&&t.cache_read_tokens!==void 0?
t.cache_read_tokens:null,g=t.cache_write_tokens!==null&&t.cache_write_tokens!==void 0?t.cache_write_tokens:
null,y=m!==null||g!==null;get("token-detail-total").innerText=i,get("token-detail-in").innerText=a,get(
"token-detail-out").innerText=o,get("token-detail-content").innerText=r,get("token-detail-thought").
innerText=c,get("token-detail-encrypted").innerText=d;const b=get("token-detail-cache-read-row"),w=get(
"token-detail-cache-write-row"),x=get("token-detail-cache-write-ttl");if(b&&b.classList.toggle("hidd\
en",!y),w&&w.classList.toggle("hidden",!y),get("token-detail-cache-read")&&(get("token-detail-cache-\
read").innerText=m===null?"-":m),get("token-detail-cache-write")&&(get("token-detail-cache-write").innerText=
g===null?"-":g),x){const T=["5m","1h","30m","12h"].map(A=>{var G;const N=Number(((G=t.cache_write_ttls)==
null?void 0:G[A])||0);return N>0?`${A}: ${N}`:""}).filter(Boolean);x.textContent=T.length?`TTL ${T.join(
" \xB7 ")}`:"",x.classList.toggle("hidden",!y||!T.length)}const k=get("token-detail-cost");if(k){k.replaceChildren();
const T=t.cost_estimate||(t.role==="assistant"&&costPricing&&typeof estimateConversationCost=="funct\
ion"?estimateConversationCost([{id:e,role:t.role,model:t.model,thought_data:t.thought_data,tokens_in:t.
tokens_in,tokens_out:t.tokens_out,tokens_thought:t.tokens_thought,tokens_content:t.tokens_content,parent_id:t.
parent_id,image_url:t.image_url,batch_job:t.batch_job}],costPricing,costBarPath):null),A=t.role==="t\
otal"&&T||t.role==="assistant"&&T;if(k.classList.toggle("hidden",!A),A){const N=document.createElement(
"p");N.textContent=costEstimateLabel(T);const G=document.createElement("p");G.id="token-detail-cost-\
jpy",k.append(N,G),fillCostEstimateJpy(G,T);const L=Array.isArray(T==null?void 0:T.items)?T.items:[];
if(L.length){const $=document.createElement("details");$.className="pt-2 border-t border-gray-700";const M=document.
createElement("summary");M.className="cursor-pointer",M.textContent=`\u30EA\u30AF\u30A8\u30B9\u30C8\u5185\u8A33\uFF08${L.
length}\u4EF6\uFF09`,$.appendChild(M),appendCostItemList($,L),k.appendChild($)}for(const $ of[`\u96C6\u8A08\u3057\u305FA\
PI\u547C\u3073\u51FA\u3057: ${(T==null?void 0:T.requests)||0}\u56DE`,COST_ESTIMATE_NOTE,`\u516C\u5F0F\u6599\u91D1\u78BA\u8A8D\u65E5: ${(costPricing==
null?void 0:costPricing.checked_at)||"\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"}`]){
const M=document.createElement("p");M.textContent=$,k.appendChild(M)}for(const[$,M]of Object.entries(
(costPricing==null?void 0:costPricing.sources)||{})){const P=document.createElement("a");P.href=M,P.
textContent=`${$} \u516C\u5F0F\u6599\u91D1 `,P.target="_blank",P.rel="noopener noreferrer",P.className=
"underline",k.appendChild(P)}}}const C=t.model?`${t.model} (${t.role})`:`${t.role}`;get("token-detai\
l-title").innerText=C,showTokenDetailModal(e)}l(openTokenDetail,"openTokenDetail");function closeTokenDetail(e=!1){
cancelPromptTokenDetail(),get("token-detail-modal")&&(hideModal("token-detail-modal"),!e&&location.pathname===
"/token-details"&&history.back())}l(closeTokenDetail,"closeTokenDetail");function openEncryptionSettings(e){
const t=messageMeta[e];t&&openEncryptionModal(t.is_encrypted)}l(openEncryptionSettings,"openEncrypti\
onSettings");function openEncryptionModal(e){if(!get("encryption-status-modal"))return;const n=get("\
encryption-status-title"),i=get("encryption-status-body"),a=get("encryption-status-admin-actions"),o=get(
"encryption-status-admin-toggle"),r=!!e;r?(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),
i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002")):
(n&&(n.innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093"),i&&(i.innerText=isAdminUser?
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u518D\u6697\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
"\u3053\u306E\u30E1\u30C3\u30BB\u30FC\u30B8\u306F\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u305B\u3093\u3002")),
a&&o&&(!!(isAdminUser&&currentThreadId)?(a.classList.remove("hidden"),o.dataset.enable=r?"0":"1",o.disabled=
!1,o.textContent=r?"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u5FA9\u53F7\u5316":"\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u518D\u6697\u53F7\u5316",
o.className=r?"w-full px-3 py-2 text-xs font-bold rounded text-white bg-amber-600 hover:bg-amber-500\
 btn-hover":"w-full px-3 py-2 text-xs font-bold rounded text-white bg-cyan-700 hover:bg-cyan-600 btn\
-hover"):a.classList.add("hidden")),showEncryptionStatusModal()}l(openEncryptionModal,"openEncryptio\
nModal");function showEncryptionStatusModal(){location.pathname!=="/encryption-status"&&history.pushState(
{modal:"encryption-status"},"","/encryption-status"),showModal("encryption-status-modal")}l(showEncryptionStatusModal,
"showEncryptionStatusModal");async function toggleThreadEncryptionFromModal(){const e=get("encryptio\
n-status-admin-toggle");if(!e||!isAdminUser||!currentThreadId||e.disabled)return;const t=e.getAttribute(
"data-enable")==="1";if(!confirm(`\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3092${t?"\u518D\u6697\u53F7\u5316":
"\u5FA9\u53F7\u5316"}\u3057\u307E\u3059\u304B\uFF1F`))return;e.disabled=!0;const i=e.textContent;e.textContent=
"\u51E6\u7406\u4E2D...";try{if(typeof window.__setAdminThreadEncryption!="function"){showToast("\u6697\u53F7\u5316\u64CD\
\u4F5C\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093","error",!0);return}await window.__setAdminThreadEncryption(
currentThreadId,t,{confirmPrompt:!1,reloadCurrent:!0})&&closeEncryptionModal()}finally{e.disabled=!1,
e.textContent=i}}l(toggleThreadEncryptionFromModal,"toggleThreadEncryptionFromModal");function closeEncryptionModal(e=!1){
hideModal("encryption-status-modal"),!e&&location.pathname==="/encryption-status"&&history.back()}l(
closeEncryptionModal,"closeEncryptionModal");function goToEncryptionSettings(){hideModal("encryption\
-status-modal"),location.pathname==="/encryption-status"&&history.replaceState({modal:"settings",from:"\
/encryption-status"},"","/settings"),typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab(
"security"),setTimeout(()=>{const e=isAdminUser&&get("admin-enc-card")||get("e2ee-card");e&&e.scrollIntoView(
{behavior:"smooth",block:"center"})},150))}l(goToEncryptionSettings,"goToEncryptionSettings");function openTemporaryChatSettings(){
typeof openSettingsModal=="function"&&(openSettingsModal(),switchTab("general"),setTimeout(()=>{const e=get(
"temp-chat-settings-card");e&&(e.scrollIntoView({behavior:"smooth",block:"center"}),e.classList.add(
"ring-1","ring-amber-400/70"),setTimeout(()=>e.classList.remove("ring-1","ring-amber-400/70"),1400))},
150))}l(openTemporaryChatSettings,"openTemporaryChatSettings");const isGeminiLocalPythonMode=l((e,t,n,i)=>{
const a=(e||"").toLowerCase();return!a.includes("gemini")||a.includes("image")||a.includes("nano")||
a.includes("tts")||a.includes("native-audio")?!1:!!i&&(t||n)},"isGeminiLocalPythonMode"),confirmGeminiLocalPythonSwitch=l(
async()=>{if(!isGeminiLocalPyDialogEnabled())return!0;const e=get("gemini-local-python-modal");if(!e)
return!0;const t=get("gemini-local-python-dont-show"),n=get("gemini-local-python-continue"),i=get("g\
emini-local-python-cancel"),a=get("gemini-local-python-close");return t&&(t.checked=!1),showModal("g\
emini-local-python-modal"),await new Promise(o=>{let r=!1;function c(){n&&n.removeEventListener("cli\
ck",m),i&&i.removeEventListener("click",g),a&&a.removeEventListener("click",g),e.removeEventListener(
"click",y,!0)}l(c,"cleanup");function d(b){if(r)return;r=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),o(b)}l(d,"finalize");
function m(){d(!0)}l(m,"onOk");function g(){d(!1)}l(g,"onCancel");function y(b){b.target===e&&(b.preventDefault(),
b.stopImmediatePropagation(),g())}l(y,"onOverlay"),n&&n.addEventListener("click",m),i&&i.addEventListener(
"click",g),a&&a.addEventListener("click",g),e.addEventListener("click",y,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const o=t?"fade-in":"",r=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${o}"><div${r} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,m=e||get("\
chat-container");if(m){if(typeof m.insertAdjacentHTML=="function")m.insertAdjacentHTML("beforeend",d);else{
const g=document.createElement("div");g.innerHTML=d;const y=g.firstElementChild;y&&m.appendChild(y)}
n&&scrollToBottom()}}l(renderPendingMessage,"renderPendingMessage");function beginPendingToStreamTransition(e){
if(!e||e.getAttribute("data-stream-transition")==="1")return;const t=e.querySelector(".content-area");
t&&(t.classList.remove("pending-shimmer","skeleton-pending"),t.removeAttribute("data-skeleton-kind")),
e.setAttribute("data-stream-transition","1"),e.classList.remove("ai-pending-bubble"),e.classList.add(
"ai-stream-transition"),t&&(t.classList.add("ai-stream-content-transition"),setTimeout(()=>{t&&t.classList.
remove("ai-stream-content-transition")},300)),setTimeout(()=>{e&&e.classList.remove("ai-stream-trans\
ition")},320)}l(beginPendingToStreamTransition,"beginPendingToStreamTransition");function normalizeJobIdForUi(e){
return e==null||e===""?null:String(e)}l(normalizeJobIdForUi,"normalizeJobIdForUi");function getActiveStreamingBubbleElement(){
return activeStreamingBubbleId?get(activeStreamingBubbleId):null}l(getActiveStreamingBubbleElement,"\
getActiveStreamingBubbleElement");function captureStoppedPartialBubbleSnapshot(e){if(!e)return null;
const t=Array.from(e.querySelectorAll(".prose")).some(c=>String(c.textContent||"").trim()),n=!!e.querySelector(
".python-box"),i=Array.from(e.querySelectorAll(".thought-content")).some(c=>!!String(c.textContent||
"").trim()&&c.getAttribute("data-placeholder")!=="1");if(!t&&!n&&!i)return null;const a=e.parentElement;
if(!a)return null;const o=a.cloneNode(!0);o.setAttribute("data-local-stopped-partial","1"),o.classList.
remove("fade-in");const r=o.querySelector(".message-bubble");if(r&&(r.classList.remove("ai-pending-b\
ubble","ai-stream-transition"),r.removeAttribute("data-stream-transition"),r.removeAttribute("id"),!o.
querySelector('[data-stopped-partial-note="1"]'))){const c=document.createElement("div");c.setAttribute(
"data-stopped-partial-note","1"),c.className="text-[10px] text-amber-200/90 mt-2 text-right",c.textContent=
"\u505C\u6B62\u6E08\u307F\uFF08\u9014\u4E2D\u307E\u3067\uFF09",r.appendChild(c)}return{html:o.outerHTML,
threadId:currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null}}l(captureStoppedPartialBubbleSnapshot,
"captureStoppedPartialBubbleSnapshot");function appendStoppedPartialBubbleSnapshot(e,t=null){if(!e||
!e.html)return!1;const n=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null,i=t!=
null&&t!==""?String(t):e.threadId?String(e.threadId):null;if(i&&n&&i!==n)return!1;const a=get("chat-\
container");return a?(a.querySelectorAll('[data-local-stopped-partial="1"]').forEach(o=>o.remove()),
a.insertAdjacentHTML("beforeend",e.html),scrollToBottom(),!0):!1}l(appendStoppedPartialBubbleSnapshot,
"appendStoppedPartialBubbleSnapshot");function suppressPendingJob(e){const t=normalizeJobIdForUi(e);
t&&suppressedPendingJobIds.add(t)}l(suppressPendingJob,"suppressPendingJob");function isPendingJobSuppressed(e){
const t=normalizeJobIdForUi(e);return!!(t&&suppressedPendingJobIds.has(t))}l(isPendingJobSuppressed,
"isPendingJobSuppressed");function isManualStopAbortForThread(e=null){if(!manualStopContext)return!1;
const t=manualStopContext.threadId?String(manualStopContext.threadId):null,n=e!=null&&e!==""?String(
e):null,i=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;return!(t&&n&&t!==
n||t&&i&&t!==i)}l(isManualStopAbortForThread,"isManualStopAbortForThread");async function syncThreadAfterAbortedStream(e=null,t={}){
var c,d;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((d=t.retryDelayMs)!=
null?d:180)||0),a=!!t.notifyOnFailure,o=e!=null&&e!==""?String(e):null,r=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!r||o&&r!==o)return!1;for(let m=0;m<=n;m++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==r?!1:(await loadMessages(r,{preserveDraft:!0,silent:!0}),
!0)}catch{m<n&&i>0&&await new Promise(y=>setTimeout(y,i))}return a&&(currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null)===r&&showToast("\u505C\u6B62\u5F8C\u306E\u5C65\u6B74\u540C\u671F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u753B\u9762\u3092\u518D\u8AAD\u307F\u8FBC\u307F\u3059\u308B\u3068\u78BA\u5B9F\u3067\u3059\u3002",
"warning",!0),!1}l(syncThreadAfterAbortedStream,"syncThreadAfterAbortedStream");function vibrateHelper(e){
try{typeof navigator!="undefined"&&navigator.vibrate&&navigator.vibrate(e)}catch(t){console.warn("Vi\
bration failed:",t)}}l(vibrateHelper,"vibrateHelper");function visibleSlashCommands(e=""){const t=String(
e||"").toLowerCase();return SLASH_COMMANDS.filter(n=>n.kind==="minimal"&&!minimalPromptMode?!1:n.label.
toLowerCase().includes(t)||n.description.toLowerCase().includes(t))}l(visibleSlashCommands,"visibleS\
lashCommands");function slashCommandSuggestionFilter(e,t){if(String(e||"").toLowerCase()!=="thinking")
return e;const i=String(t||"").trimStart().match(/^\/thinking(\s+.*)$/i);return i?`thinking${i[1]}`.
toLowerCase():e}l(slashCommandSuggestionFilter,"slashCommandSuggestionFilter");function parseSlashToggleArgument(e){
const t=String(e||"").trim().toLowerCase();if(!t||t==="toggle"||t==="\u5207\u66FF"||t==="\u5207\u308A\u66FF\u3048")
return null;if(["on","true","1","\u30AA\u30F3","\u6709\u52B9"].includes(t))return!0;if(["off","false",
"0","\u30AA\u30D5","\u7121\u52B9"].includes(t))return!1}l(parseSlashToggleArgument,"parseSlashToggle\
Argument");function executeMinimalSlashCommand(e,t=""){const n=MINIMAL_SLASH_COMMANDS.find(o=>o.id===
e);if(!n||!minimalPromptMode)return!1;if(n.action==="options")return openMinimalOptions(),!0;const i=MINIMAL_POPUP_ITEMS.
find(o=>o.key===n.itemKey);if(!i||!minimalOptionVisible(i))return showToast(`/${e} \u306F\u73FE\u5728\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;if(minimalOptionDisabled(i)&&i.special!=="thinking")return showToast(`/${e} \u306F\u73FE\u5728\u5909\u66F4\u3067\u304D\u307E\u305B\u3093`,
"warning"),!0;const a=String(n.presetArgument||t||"").trim();if(i.selectId){if(!a)return showToast(`\
\u4F7F\u3044\u65B9: ${n.label} default / none`,"info"),!1;const o=get(i.selectId),r=a.toLowerCase(),
c=o?Array.from(o.options).find(d=>d.value.toLowerCase()===r||d.textContent.trim().toLowerCase()===r):
null;return!o||!c?(showToast(`${n.label}: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1):(o.value=c.value,o.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),
showToast(`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const o=a.
toLowerCase(),r=activeThinkingSelect(),c=!!(r&&r.id==="reasoning-effort"),d=c?{off:"none",none:"none",
min:"low",minimal:"low",low:"low",mid:"medium",medium:"medium",high:"high",xhigh:"xhigh",max:"max"}:
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},m=parseSlashToggleArgument(
a),g=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(d,o)){const y=d[o],b=r?Array.from(r.options).
find(w=>w.value===y&&!w.disabled&&!w.classList.contains("hidden")):null;if(r&&!b)return showToast(`T\
hinking: \u300C${o}\u300D\u306F\u3053\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1;if(y==="none"){if(g&&g.disabled)return showToast("Thinking: \u3053\u306E\u30E2\u30C7\u30EB\u3067\u306FOFF\u306B\u3067\u304D\u307E\u305B\u3093",
"warning"),!1;g&&g.checked?(g.checked=!1,g.dispatchEvent(new Event("change",{bubbles:!0}))):r&&(r.value=
"none",r.dispatchEvent(new Event("change",{bubbles:!0})))}else g&&!g.checked&&!g.disabled&&(g.checked=
!0,g.dispatchEvent(new Event("change",{bubbles:!0}))),r&&(r.value=y,r.dispatchEvent(new Event("chang\
e",{bubbles:!0})));return refreshMinimalOptionItems(),showToast(`Thinking: ${o}`,"success"),!0}if(m===
void 0)return showToast(c?"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high / xhigh / max":
"\u4F7F\u3044\u65B9: /thinking on / off / min / low / mid / high","info"),!1}if(i.checkboxId&&a){const o=parseSlashToggleArgument(
a);if(o===void 0)return showToast(`\u4F7F\u3044\u65B9: ${n.label} on / off`,"info"),!1;const r=get(i.
checkboxId);if(o!==null&&r&&r.checked===o)return showToast(`${i.label}: ${o?"ON":"OFF"}`,"info"),!0}
return handleMinimalOptionClick(i),!0}l(executeMinimalSlashCommand,"executeMinimalSlashCommand");function extractSlashCommandToken(e){
const t=String(e||"").trimStart();if(!t.startsWith("/"))return null;const n=t.substring(1).split(/\s+/)[0]||
"",i=n.match(/^[a-z][\w-]*/i);return i?i[0]:n}l(extractSlashCommandToken,"extractSlashCommandToken");
function hideSlashCommandSuggestions(){const e=get("slash-command-suggestions");e&&e.classList.add("\
hidden"),slashSuggestionsVisible=!1,slashSelectedIndex=0}l(hideSlashCommandSuggestions,"hideSlashCom\
mandSuggestions");function showPendingSlashCommandIndicator(e){const t=get("slash-command-indicator"),
n=get("slash-command-name");if(!t||!n)return;const i=SLASH_COMMANDS.find(o=>o.id===e);n.textContent=
i?i.label:`/${e}`,t.classList.remove("hidden"),t.classList.add("flex");const a=get("prompt-input");a&&
i&&(a.dataset.originalPlaceholder=a.placeholder,a.placeholder=i.argumentHint||"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\
\u30EB\u3092gemini-2.5-flash\u306B\u5909\u66F4\uFF09...")}l(showPendingSlashCommandIndicator,"showPe\
ndingSlashCommandIndicator");function hidePendingSlashCommandIndicator(){const e=get("slash-command-\
indicator");e&&(e.classList.remove("flex"),e.classList.add("hidden"));const t=get("prompt-input");t&&
t.dataset.originalPlaceholder&&(t.placeholder=t.dataset.originalPlaceholder,delete t.dataset.originalPlaceholder);
const n=pendingSlashCommand==="settings";pendingSlashCommand=null,n&&clearAiSettingsConversation()}l(
hidePendingSlashCommandIndicator,"hidePendingSlashCommandIndicator");function showSlashCommandSuggestions(e=""){
const t=get("slash-command-suggestions"),n=get("slash-command-list"),i=get("input-row");if(!t||!n||!i)
return;const a=visibleSlashCommands(e);if(a.length===0){hideSlashCommandSuggestions();return}slashSelectedIndex=
Math.min(slashSelectedIndex,a.length-1),n.innerHTML="",a.forEach((b,w)=>{const x=document.createElement(
"div");x.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-gray-700 ${w===
slashSelectedIndex?"bg-gray-700":""}`,x.innerHTML=`
                    <i class="fas ${b.icon||"fa-terminal"} w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="font-mono text-blue-300">${b.label}</div>
                        <div class="text-[11px] text-gray-400 truncate">${b.description}</div>
                    </div>
                `;let k=!1;x.addEventListener("pointerdown",C=>{typeof C.button=="number"&&C.button!==
0||(C.preventDefault(),k=!0,selectSlashCommand(b.id))}),x.addEventListener("click",C=>{C.preventDefault(),
k||selectSlashCommand(b.id)}),x.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(x)});const o=i.getBoundingClientRect(),r=window.innerHeight,c=r-o.bottom,d=o.top,m=260,
g=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,o.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&d>c){const b=Math.min(m,d-g);t.style.top="auto",t.style.bottom=`${r-o.top+4}px`,n.style.
maxHeight=`${b}px`}else{const b=Math.min(m,c-g);t.style.top=`${o.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${b}px`}t.classList.remove("hidden"),slashSuggestionsVisible=!0}l(showSlashCommandSuggestions,
"showSlashCommandSuggestions");function selectSlashCommand(e){const t=get("prompt-input");if(!t)return;
const n=t.value,i=extractSlashCommandToken(n);if(i!==null){const r=String(n||"").trimStart();t.value=
r.substring(1+i.length).trimStart()}else{const r=n.lastIndexOf("/");r!==-1?t.value=n.substring(0,r).
trimEnd():t.value=""}hideSlashCommandSuggestions();const a=SLASH_COMMANDS.find(r=>r.id===e),o=t.value.
trim();if(a&&a.autocompleteArgument&&!o){t.value=`${a.label} `,slashSelectedIndex=0,lastSlashFilter=
null,t.dispatchEvent(new Event("input",{bubbles:!0})),t.focus();return}if(a&&a.kind==="minimal"&&(!a.
requiresArgument||o)){t.value="",executeMinimalSlashCommand(e,o),t.dispatchEvent(new Event("input",{
bubbles:!0})),t.focus();return}pendingSlashCommand=e,showPendingSlashCommandIndicator(e),t.focus(),t.
dispatchEvent(new Event("input",{bubbles:!0}))}l(selectSlashCommand,"selectSlashCommand");const AI_SETTING_JUMP_TARGETS={
default_model:{label:"\u65E2\u5B9A\u306E\u30E2\u30C7\u30EB",tab:"general",control:"set-default-model"},
default_vision_model:{label:"Vision Model",tab:"general",control:"set-default-vision-model"},use_last_chat_settings:{
label:"\u524D\u56DE\u306E\u8A2D\u5B9A\u3092\u7D99\u7D9A",tab:"general",control:"set-use-last-setting\
s"},default_enable_search:{label:"\u65E2\u5B9A\u306ESearch",tab:"general",control:"set-default-searc\
h"},default_gemini_free_tier:{label:"\u65E2\u5B9A\u306E\u7121\u6599\u67A0",tab:"general",control:"se\
t-default-gemini-free-tier"},default_gemini_flex:{label:"\u65E2\u5B9A\u306EFlex",tab:"general",control:"\
set-default-gemini-flex"},default_enable_url_context:{label:"\u65E2\u5B9A\u306EURLs",tab:"general",control:"\
set-default-url-context"},default_enable_maps:{label:"\u65E2\u5B9A\u306EMaps",tab:"general",control:"\
set-default-maps"},default_enable_python:{label:"\u65E2\u5B9A\u306EPython",tab:"general",control:"se\
t-default-python"},default_enable_file_creation:{label:"\u65E2\u5B9A\u306EFile",tab:"general",control:"\
set-default-file-creation"},default_enable_thinking:{label:"\u65E2\u5B9A\u306EThinking",tab:"general",
control:"set-default-thinking"},default_thinking_level:{label:"Thinking Level",tab:"general",control:"\
set-default-thinking-level"},default_thinking_budget:{label:"Thinking Budget",tab:"general",control:"\
set-default-thinking-budget"},default_reasoning_effort:{label:"Reasoning Effort",tab:"general",control:"\
set-default-reasoning-effort"},default_enable_system_prompt:{label:"\u65E2\u5B9A\u306ESysPrompt",tab:"\
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
return"\u66F4\u65B0\u6E08\u307F"}return String(e)}l(formatAiSettingValue,"formatAiSettingValue");function findSettingsJumpElement(e,t){
const n=get(`tab-${e}`);let i=get(t);if(!n||!i)return null;for(;i.parentElement&&i.parentElement!==n;)
i=i.parentElement;return i.parentElement===n?i:get(t)}l(findSettingsJumpElement,"findSettingsJumpEle\
ment");function openAiSettingJumpTarget(e){const t=AI_SETTING_JUMP_TARGETS[e];if(!t){typeof window.openSettingsModal==
"function"&&window.openSettingsModal();return}if(t.modal==="rich-paste"){openRichPasteModal(),setTimeout(
()=>{const n=get(t.control);n&&(n.scrollIntoView({behavior:"smooth",block:"center"}),n.focus({preventScroll:!0}))},
260);return}typeof window.openSettingsModal=="function"&&window.openSettingsModal(),setTimeout(()=>{
const n=findSettingsJumpElement(t.tab,t.control);n?jumpToSetting(t.tab,n):switchTab(t.tab||"general")},
320)}l(openAiSettingJumpTarget,"openAiSettingJumpTarget");function removeEphemeralMessageControls(e){
if(!e)return;const t=e.querySelector(".msg-controls");t&&t.remove()}l(removeEphemeralMessageControls,
"removeEphemeralMessageControls");function renderAiSettingsResultBubble(e,t,n="update"){const i=Object.
entries(e||{}),a=`settings-result-${Date.now()}`,o=n==="inspect",r=i.length?o?`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\u3002

\u78BA\u8A8D\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\u3002

\u5909\u66F4\u3057\u305F\u9805\u76EE\u3092\u30BF\u30C3\u30D7\u3059\u308B\u3068\u3001\u8A2D\u5B9A\u753B\u9762\u306E\u8A72\u5F53\u7B87\u6240\u3078\u79FB\u52D5\u3067\u304D\u307E\u3059\u3002`:
o?"\u78BA\u8A8D\u3067\u304D\u308B\u8A2D\u5B9A\u9805\u76EE\u304C\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002":
"\u5909\u66F4\u3055\u308C\u305F\u8A2D\u5B9A\u9805\u76EE\u306F\u3042\u308A\u307E\u305B\u3093\u3067\u3057\u305F\u3002",
c=renderMessage(a,"assistant",r,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);if(!c)
return;removeEphemeralMessageControls(c);const d=c.querySelector(".message-bubble");if(!d||!i.length)
return;const m=document.createElement("div");m.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([y,b])=>{const w=AI_SETTING_JUMP_TARGETS[y]||{label:y},x=document.createElement("button");x.
type="button",x.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const k=document.createElement("span");k.className="min-w-0 flex-1";const C=document.createElement(
"span");C.className="block text-xs font-bold text-blue-200",C.textContent=w.label;const T=document.createElement(
"span");T.className="block mt-0.5 text-[11px] text-gray-300 break-words",T.textContent=formatAiSettingValue(
b);const A=document.createElement("i");A.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",k.appendChild(C),k.appendChild(T),x.appendChild(k),x.appendChild(A),x.addEventListener(
"click",()=>openAiSettingJumpTarget(y)),m.appendChild(x)});const g=d.querySelector(".message-footer-\
meta");g?d.insertBefore(m,g):d.appendChild(m),scrollToBottom()}l(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const o=`settings-pending-${n}`,r=get("chat-container");
r&&(r.insertAdjacentHTML("beforeend",`<div id="${o}" class="flex justify-start mb-4 ai-pending-row f\
ade-in"><div class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-\
tl-none shadow-md relative">${buildPendingSkeletonHtml(t,"\u8A2D\u5B9A\u30EA\u30AF\u30A8\u30B9\u30C8\u3092\u78BA\u8A8D\u3057\u3066\u3044\u307E\u3059...")}\
</div></div>`),scrollToBottom());try{const d=await(await apiFetch("/api/settings/apply-ai-prompt",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:e,model:t,conversation:aiSettingsConversation})})).
json().catch(()=>({})),m=get(o);if(m&&m.remove(),d&&d.status==="ok"&&d.mode==="inspect"&&d.current){
appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(d.current,"inspect")),
showToast(`\u73FE\u5728\u306E\u8A2D\u5B9A\u3092\u78BA\u8A8D\u3057\u307E\u3057\u305F\uFF08${Object.keys(
d.current).length}\u9805\u76EE\uFF09`,"success"),renderAiSettingsResultBubble(d.current,t,"inspect");
return}if(d&&d.status==="ok"&&d.applied){appendAiSettingsConversation("assistant",summarizeAiSettingsConversationValues(
d.applied,"update")),showToast(`\u8A2D\u5B9A\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F\uFF08${Object.
keys(d.applied).length}\u9805\u76EE\uFF09`,"success");try{const b=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(b),cacheUserSettings(b)}catch{}renderAiSettingsResultBubble(
d.applied,t);return}const g=d.message||d.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${g}`);
const y=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${g}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
y),showToast(g,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const d=get(o);d&&d.remove();const m=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(m),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
l(runAiSettingsCommand,"runAiSettingsCommand");function hideSkillSuggestions(){const e=get("skill-su\
ggestions");e&&e.classList.add("hidden"),skillSuggestionsVisible=!1,skillSelectedIndex=0}l(hideSkillSuggestions,
"hideSkillSuggestions");function showSkillSuggestions(e=""){const t=get("skill-suggestions"),n=get("\
skill-suggestions-list"),i=get("input-row");if(!t||!n||!i)return;if(!loadedSkills||loadedSkills.length===
0){hideSkillSuggestions();return}const a=e.toLowerCase(),o=loadedSkills.filter(w=>w.name.toLowerCase().
includes(a)||w.description&&w.description.toLowerCase().includes(a));if(o.length===0){hideSkillSuggestions();
return}skillSelectedIndex>=o.length&&(skillSelectedIndex=0),n.innerHTML="",o.forEach((w,x)=>{const k=document.
createElement("div");k.className=`px-3 py-2 flex items-center gap-3 cursor-pointer text-sm hover:bg-\
gray-700 ${x===skillSelectedIndex?"bg-gray-700":""}`,k.innerHTML=`
                    <i class="fas fa-wand-magic-sparkles w-4 text-blue-400"></i>
                    <div class="flex-1 min-w-0">
                        <div class="text-blue-300 truncate font-medium">${escapeHtml(w.name)}</div>
                        ${w.description?`<div class="text-[11px] text-gray-400 truncate">${escapeHtml(
w.description)}</div>`:""}
                    </div>
                `,k.onclick=()=>selectSkillSuggestion(w),k.onmouseenter=()=>{skillSelectedIndex=x,showSkillSuggestions(
e)},n.appendChild(k)});const r=i.getBoundingClientRect(),c=window.innerHeight,d=c-r.bottom,m=r.top,g=260,
y=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,r.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&m>d){const w=Math.min(g,m-y);t.style.top="auto",t.style.bottom=`${c-r.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(g,d-y);t.style.top=`${r.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),skillSuggestionsVisible=!0}l(showSkillSuggestions,
"showSkillSuggestions");function selectSkillSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideSkillSuggestions(),
activateSkill(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}l(selectSkillSuggestion,
"selectSkillSuggestion");function browserFastModeIneligibility(e){const t=String(get("model-select")?
get("model-select").value:"").toLowerCase();if(!e||!e.trim())return"\u30D7\u30ED\u30F3\u30D7\u30C8\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044";
if(!t.startsWith("gemini-")||/(image|native-audio|tts|live)/.test(t))return"Gemini\u30C6\u30AD\u30B9\u30C8\u30E2\u30C7\u30EB\u5C02\u7528\u3067\u3059";
if(currentImageUrls.length)return"\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u6E08\u307F\u6DFB\u4ED8\u304C\u3042\u308B\u305F\u3081\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(activeSkill)return"Skills\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(currentQuote||editingMessageId)return"\u5F15\u7528\u30FB\u7DE8\u96C6\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(codingModeEnabled)return"Coding Mode\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
if(["enable-search","enable-url-context","enable-maps","enable-sys-prompt","enable-prompt-cache","en\
able-mcp"].some(r=>{const c=get(r);return!!(c&&c.checked)}))return"\u691C\u7D22\u30FBURL\u53C2\u7167\u30FB\u30B7\u30B9\u30C6\u30E0\u6A5F\u80FD\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\u8981\u3067\u3059";
const i=get("thread-custom-instruction");if(i&&String(i.value||"").trim())return"\u30C1\u30E3\u30C3\u30C8\u56FA\u6709\u6307\u793A\u5229\u7528\u6642\u306F\u901A\u5E38\u30E2\u30FC\u30C9\u304C\u5FC5\
\u8981\u3067\u3059";const a=Array.from(browserFastLocalFiles.values());return a.length>BROWSER_FAST_MAX_IMAGES?
"\u753B\u50CF\u306F4\u679A\u307E\u3067\u3067\u3059":a.reduce((r,c)=>r+Number(c.file&&c.file.size||0),
0)>BROWSER_FAST_MAX_BYTES?"\u753B\u50CF\u5408\u8A08\u306F12MB\u307E\u3067\u3067\u3059":a.some(r=>!r.
file||!String(r.file.type||"").startsWith("image/"))?"\u753B\u50CF\u4EE5\u5916\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093":
""}l(browserFastModeIneligibility,"browserFastModeIneligibility");function fileToBase64Payload(e){return new Promise(
(t,n)=>{const i=new FileReader;i.onload=()=>{const a=String(i.result||""),o=a.indexOf(",");if(o<0)return n(
new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F"));t(
a.slice(o+1))},i.onerror=()=>n(i.error||new Error("\u753B\u50CF\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F")),
i.readAsDataURL(e)})}l(fileToBase64Payload,"fileToBase64Payload");async function buildBrowserFastHistoryContents(e){
const t=[];let n=0;for(const i of Array.isArray(e)?e:[]){if(!i||!["user","model"].includes(i.role))continue;
const a=[];i.role==="model"&&Array.isArray(i.thought_signatures)&&i.thought_signatures.forEach(o=>{o&&
a.push({thoughtSignature:String(o)})}),i.text&&a.push({text:String(i.text)});for(const o of Array.isArray(
i.images)?i.images:[])try{const r=await fetch(buildFileUrl(o.path),{credentials:"same-origin",cache:"\
no-store"});if(!r.ok)throw new Error(`HTTP ${r.status}`);const c=await r.blob();a.push({inlineData:{
mimeType:o.mime_type||c.type||"application/octet-stream",data:await fileToBase64Payload(c)}})}catch{
n++}a.length&&t.push({role:i.role,parts:a})}return n&&showToast(`\u5C65\u6B74\u753B\u50CF${n}\u4EF6\u3092\u518D\u53D6\u5F97\u3067\u304D\
\u306A\u304B\u3063\u305F\u305F\u3081\u3001\u30C6\u30AD\u30B9\u30C8\u5C65\u6B74\u3060\u3051\u3067\u7D9A\u884C\u3057\u307E\u3059`,
"warning",!0),t}l(buildBrowserFastHistoryContents,"buildBrowserFastHistoryContents");async function uploadBrowserFastLocalFiles(e=browserFastLocalFiles,t=!1){
const n=Array.from(e.entries()),i=[];for(const[a,o]of n){if(!o||!o.file||!o.rowObj)throw new Error("\
\u30ED\u30FC\u30AB\u30EB\u753B\u50CF\u306E\u72B6\u614B\u304C\u5931\u308F\u308C\u307E\u3057\u305F");if(o.
rowObj.status&&(o.rowObj.status.textContent="\u56DE\u7B54\u5B8C\u4E86\u30FB\u30B5\u30FC\u30D0\u30FC\u4FDD\u5B58\u4E2D..."),
!await uploadFileWithProgress(o.file,o.rowObj))throw new Error(`${o.file.name||"\u753B\u50CF"}\u3092\u30B5\u30FC\u30D0\u30FC\u3078\
\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F`);const c=o.rowObj.row.getAttribute("da\
ta-filename");c&&i.push(c),t?currentImageUrls=currentImageUrls.filter(d=>d!==c):browserFastLocalFiles.
delete(a)}return i}l(uploadBrowserFastLocalFiles,"uploadBrowserFastLocalFiles");function browserFastThinkingConfig(e){
const t=get("enable-thinking");if(!t||!t.checked)return null;const n=String(get("thinking-level")?get(
"thinking-level").value:"high").toLowerCase();if(e.includes("2.5")){const a=Number(get("thinking-bud\
get")?get("thinking-budget").value:4096);return{includeThoughts:!0,thinkingBudget:Number.isFinite(a)?
Math.max(0,Math.min(32768,Math.trunc(a))):4096}}let i=n.toUpperCase();return e.includes("3.6")&&!["M\
EDIUM","HIGH"].includes(i)&&(i="MEDIUM"),e.includes("3.5")&&!["MINIMAL","MEDIUM","HIGH"].includes(i)&&
(i="MINIMAL"),{includeThoughts:!0,thinkingLevel:i}}l(browserFastThinkingConfig,"browserFastThinkingC\
onfig");function browserFastPythonBoxHtml(e){return`<div class="code-wrapper python-box collapsed" d\
ata-py-id="${e}" data-collapsed="true" data-code-key="${e}"><div class="code-header"><span class="co\
de-lang"><i class="fas fa-terminal"></i> Python Execution</span><div class="code-actions"><button cl\
ass="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\u958B"><i class="fas fa-chevron-down"></\
i></button><button class="copy-btn" data-copy="code" data-code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\
\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" data-copy="output" data-code="" titl\
e="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i></button></div></div><div class="cod\
e-body"><div class="python-section"><div class="python-label">Code</div><pre><code class="hljs langu\
age-python python-code"></code></pre></div><div class="python-section"><div class="python-label">Out\
put</div><pre><code class="hljs language-plaintext python-output"></code></pre></div></div></div>`}l(
browserFastPythonBoxHtml,"browserFastPythonBoxHtml");function updateBrowserFastPythonBox(e,t,n){if(e){
if(t==="code"){const i=n==null?"":String(n),a=e.querySelector(".python-code");a&&(a.textContent=i,a.
removeAttribute("data-highlighted"),queueHighlight(e,i));const o=e.querySelector('.copy-btn[data-cop\
y="code"]');o&&o.setAttribute("data-code",encodeURIComponent(i).replace(/'/g,"%27"))}else if(t==="ou\
tput"){const i=n==null?"":String(n),a=e.querySelector(".python-output");a&&(a.textContent=i);const o=e.
querySelector('.copy-btn[data-copy="output"]');o&&o.setAttribute("data-code",encodeURIComponent(i).replace(
/'/g,"%27"))}}}l(updateBrowserFastPythonBox,"updateBrowserFastPythonBox");function collectBrowserFastPromptOptions(){
const e=l((i,a)=>get(i)?!!get(i).checked:a,"checked"),t=l(i=>get(i)?get(i).value:null,"value"),n=browserFastPreviousOptions&&
browserFastPreviousOptions.checks&&Object.prototype.hasOwnProperty.call(browserFastPreviousOptions.checks,
"enable-skills")?!!browserFastPreviousOptions.checks["enable-skills"]:!0;return{canvas_mode:!!canvasModeEnabled,
enable_search:e("enable-search",!1),gemini_free_tier:e("enable-gemini-free-tier",!1),enable_url_context:e(
"enable-url-context",!1),enable_maps:e("enable-maps",!1),enable_python:e("enable-python",!1),enable_file_creation:e(
"enable-file-creation",!0),enable_mcp:typeof isMcpEnabledForSend=="function"?!!isMcpEnabledForSend():
!1,enable_skills:n,enable_thinking:e("enable-thinking",!1),thinking_level:t("thinking-level"),thinking_budget:t(
"thinking-budget"),reasoning_effort:t("reasoning-effort"),safety_setting:t("safety-setting"),enable_prompt_caching:e(
"enable-prompt-cache",!1),math_steps:typeof window.collectMathStepsSendOption=="function"?window.collectMathStepsSendOption():
!1}}l(collectBrowserFastPromptOptions,"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e,t){
var X,re,ce;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),o=[];for(const Q of a)o.push({inlineData:{
mimeType:Q.file.type,data:await fileToBase64Payload(Q.file)}});o.push({text:e});const r={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(r.thinkingConfig=c);const d={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:o}],generationConfig:r},m=[];i.project_files_notice&&m.push(String(i.project_files_notice)),
i.project_instruction&&m.push(`[Project Instructions]:
${String(i.project_instruction)}`),i.math_steps_system_prompt&&m.push(String(i.math_steps_system_prompt)),
m.length&&(d.systemInstruction={parts:[{text:m.join(`

`)}]});const g=!!(get("enable-python")&&get("enable-python").checked),y=[];g&&y.push({codeExecution:{}});
const b=i.project_uuid&&i.project_file_tool_declaration?{name:i.project_file_tool_declaration.name,description:i.
project_file_tool_declaration.description,parameters:{type:"object",properties:i.project_file_tool_declaration.
parameters&&i.project_file_tool_declaration.parameters.properties||{},required:i.project_file_tool_declaration.
parameters&&i.project_file_tool_declaration.parameters.required||["name"]}}:null;b&&y.push({functionDeclarations:[
b]}),y.length&&(d.tools=y),typeof window.hideProjectViewForChat=="function"&&window.hideProjectViewForChat(),
e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&(promptHistory.unshift(e),promptHistory.
length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt="",playSendAnimation(),get("welcome-scre\
en").classList.add("hidden");const w=userAutoScroll&&isChatNearBottom();w||pauseChatAutoScroll();const x=collectImageUrlsForSend(),
k=Array.from(((X=get("file-preview-thumbs"))==null?void 0:X.querySelectorAll("img"))||[]).map(Q=>Q.cloneNode(
!0)),C=renderMessage(Date.now(),"user",e,JSON.stringify(collectImageUrlsForSend()),null,null,null,!1,
currentQuote,null,null,null,null,null,null,null,w);if(k.length&&!x.length){const Q=document.createElement(
"div");Q.className="flex gap-2 flex-wrap",k.forEach(ue=>{ue.removeAttribute("onclick"),Q.appendChild(
ue)}),C.querySelector(".message-bubble").appendChild(Q)}const T=takePromptForSend(),A=animatePromptToMessage(
C,T,w),N=`browser-fast-${Date.now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div cla\
ss="flex justify-start mb-4 ai-pending-row fade-in"><div id="${N}" class="message-bubble ai-pending-\
bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(
n,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}</div></div>`);const G=get(N),L=startPendingElapsed(
G,t);activeStreamingBubbleId=N,setSendBtnToStopMode(),w&&resumeChatAutoScroll(),abortController=new AbortController;
let $="",M="",P=null;const R=new Set,D=[];let H=null,ee=null,O=!1;const B={},J=[];let F=null,q="";const te=window.
ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;let V=!1;try{let Q=!1;const ue=b?
8:1;for(let Ce=0;Ce<ue;Ce++){const lt=await fetch(`https://generativelanguage.googleapis.com/v1beta/\
models/${encodeURIComponent(n)}:streamGenerateContent?alt=sse`,manualSpinnerRequestOptions({method:"\
POST",headers:{"Content-Type":"application/json","x-goog-api-key":browserFastApiKey},body:JSON.stringify(
d),signal:abortController.signal}));if(!lt.ok){const He=await lt.json().catch(()=>({}));throw new Error(
He&&He.error&&He.error.message?He.error.message:`Gemini API HTTP ${lt.status}`)}window.ConnectionMonitor&&
!V&&(V=!0,window.ConnectionMonitor.operationStarted()),te&&te.setPhase("waiting"),Ce===0&&(T.accept(
!1),(re=window.ActivityLog)==null||re.log("send.stream.start"));const fe=lt.body.getReader(),ye=new TextDecoder;
let ge="";const ie=[],it=[],st=l(He=>{var Rt,At;const at=He.split(/\r?\n/).filter(qe=>qe.startsWith(
"data:")).map(qe=>qe.slice(5).trim()).join("");if(!at||at==="[DONE]")return;const Ke=JSON.parse(at);
if(Number.isInteger((Rt=Ke.usageMetadata)==null?void 0:Rt.promptTokenCount)){P=P||{};for(const qe of[
"promptTokenCount","candidatesTokenCount","thoughtsTokenCount","cachedContentTokenCount"])Number.isInteger(
Ke.usageMetadata[qe])&&Ke.usageMetadata[qe]>=0&&(P[qe]=Ke.usageMetadata[qe])}for(const qe of Ke.candidates||
[])for(const ut of((At=qe.groundingMetadata)==null?void 0:At.webSearchQueries)||[])typeof ut=="strin\
g"&&ut.trim()&&R.add(ut);if(R.size&&(P=P||{},P.grounding_queries=R.size),Ke.error)throw new Error(Ke.
error.message||"Gemini API error");if((Array.isArray(Ke.candidates)?Ke.candidates:[]).forEach(qe=>{(qe&&
qe.content&&Array.isArray(qe.content.parts)?qe.content.parts:[]).forEach(Ve=>{if(Ve&&typeof Ve.thoughtSignature==
"string"&&!D.includes(Ve.thoughtSignature)&&D.push(Ve.thoughtSignature),Ve&&Ve.functionCall&&Ve.functionCall.
name==="read_project_file"){ie.push(Ve),it.push(Ve.functionCall);return}if(Ve&&Ve.executableCode&&typeof Ve.
executableCode.code=="string"){const tt=Ve.executableCode.code;$+=`
\`\`\`python
${tt}
\`\`\`
`,F=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,q=tt,B[F]||(G.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml(F)),B[F]=G.querySelector(`[data-py-id="${F}"]`)),updateBrowserFastPythonBox(
B[F],"code",tt);return}if(Ve&&Ve.codeExecutionResult&&typeof Ve.codeExecutionResult.output=="string"){
const tt=Ve.codeExecutionResult.output;$+=`
**Output:**
\`\`\`
${tt}
\`\`\`
`;const pt=F||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;J.push({code:q||
"",output:tt}),B[pt]||(G.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(pt)),B[pt]=G.querySelector(
`[data-py-id="${pt}"]`)),updateBrowserFastPythonBox(B[pt],"output",tt);return}const wt=typeof Ve.text==
"string"?Ve.text:"";wt&&(Ve.thought===!0?M+=wt:$+=wt)})}),!O&&($||M)){beginPendingToStreamTransition(
G);const qe=G.querySelector(".content-area");qe&&qe.remove(),O=!0}M&&(ee||(G.insertAdjacentHTML("aft\
erbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)">\
<i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></di\
v></div>'),ee=G.querySelector(".thought-content")),ee.textContent=M),$&&(H||(H=document.createElement(
"div"),H.className="content-area prose prose-invert text-sm break-words",G.appendChild(H)),renderAiMarkdownInto(
H,$,{incrementalMath:!0,streaming:!0,mathSteps:{jobId:null,streaming:!0}})),scrollToBottom()},"consu\
meEvent");for(;;){const{done:He,value:at}=await fe.read();if(!He&&!Q&&(Q=!0,(ce=window.ActivityLog)==
null||ce.log("send.stream.first")),He)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),
te&&te.setPhase("receiving"),ge+=ye.decode(at,{stream:!0});const Ke=ge.split(/\r?\n\r?\n/);ge=Ke.pop()||
"",Ke.forEach(st)}if(ge+=ye.decode(),ge.trim()&&st(ge),!it.length||!i.project_uuid)break;d.contents.
push({role:"model",parts:ie});const vt=[],Mt=String(window.CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.
CHAT_CONFIG.urls.readProjectFileApi||"/api/projects/0/files/read").replace("/projects/0/","/projects\
/"+encodeURIComponent(String(i.project_uuid))+"/");for(const He of it){let at="Error reading project\
 file.";try{const Lt=await(await apiFetch(Mt,{method:"POST",headers:{"Content-Type":"application/jso\
n"},body:JSON.stringify({name:He.args&&He.args.name||"",offset:He.args&&He.args.offset||0,limit:He.args&&
He.args.limit||2e4}),signal:abortController.signal})).json().catch(()=>({}));at=Lt.content||Lt.error||
at}catch(Ke){at=`Error reading project file: ${Ke&&Ke.message?Ke.message:Ke}`}vt.push({functionResponse:{
name:"read_project_file",response:{response:at}}})}d.contents.push({role:"user",parts:vt})}if(!$.trim())
throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
H&&renderAiMarkdownInto(H,$,{incrementalMath:!0,streaming:!1,mathSteps:{jobId:null,streaming:!1}}),ee&&
ee.classList.add("collapsed"),J.length&&($+=J.map(Ce=>`
\`\`\`pyexec
${JSON.stringify(Ce)}
\`\`\`
`).join(""));let ne=[];a.length&&(te&&te.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),ne=await uploadBrowserFastLocalFiles(T.files,!0)),te&&te.setPhase("saving");const xe=[...x,
...ne],ve=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:$,pricing_usage:P,thought_content:M,model:n,image_urls:xe,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,project_id:!currentThreadId&&typeof window.
getActiveProjectId=="function"?window.getActiveProjectId():null,parent_id:i.parent_id||null,thought_signatures:D,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),G),we=await ve.json().
catch(()=>({}));if(!ve.ok||!we.thread_id)throw new Error(we.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(we.assistant_message_id!==void 0&&we.assistant_message_id!==null&&typeof window.bindMathStepsMessageId==
"function"){const Ce=G?G.querySelector(".math-steps"):null;Ce&&await window.bindMathStepsMessageId(Ce,
we.assistant_message_id)}const Oe=!currentThreadId;currentThreadId=String(we.thread_id),currentParentId=
we.assistant_message_id||null,currentLeafId=we.assistant_message_id||null,browserFastBootstrap=null,
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!Oe}),applyBrowserFastModeRestrictions(),
loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(Q){if(A(),T.restore(),Q.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${Q.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const ue=Q.message||"\
\u30A8\u30E9\u30FC";G&&G.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(ue));try{let ne=$||
"";J.length&&(ne+=J.map(Ce=>`
\`\`\`pyexec
${JSON.stringify(Ce)}
\`\`\`
`).join(""));const xe=buildChatErrorMarkdown(ue,ne),ve=a.length?[]:x,we=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:xe,
pricing_usage:P,thought_content:M||"",model:n,image_urls:ve,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:
null,thought_signatures:D,turnstile_token:botTurnstileTokenForRequest()}),signal:abortController&&!abortController.
signal.aborted?abortController.signal:void 0}),G),Oe=await we.json().catch(()=>({}));if(we.ok&&Oe.thread_id){
const Ce=!currentThreadId;currentThreadId=String(Oe.thread_id),currentParentId=Oe.assistant_message_id||
null,currentLeafId=Oe.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!Ce}),applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(ne){
sendClientDebugLog("error",`Browser fast error persist failed: ${ne&&ne.message?ne.message:ne}`)}}}finally{
T.releaseFiles(),L&&L(),V&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),te&&te(),
setSendBtnToSendMode(),activeStreamingBubbleId===N&&(activeStreamingBubbleId=null),abortController=null,
updateFilePreview()}}l(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
l(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var Ot,jt,Ft,zt,Dt;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const Y=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(Y);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const Y=pendingSlashCommand,
ke=a.trim(),Re=get("model-select")?get("model-select").value:null;if(Y==="settings"){if(!ke){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!Re){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
ke,Re)}else executeMinimalSlashCommand(Y,ke)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const o=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(o&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(Y=>Y.id===
o[1].toLowerCase())){executeMinimalSlashCommand(o[1].toLowerCase(),o[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const r=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(r&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(r)setBrowserFastModeEnabled(!1);else{const Y=browserFastModeIneligibility(
a);if(!Y){try{await sendBrowserFastMessage(a,t)}catch(ke){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${ke.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${Y}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(ke){showToast(
ke.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),d=c.map(Y=>Y.path),m=c.filter(Y=>normalizeAttachmentSource(
Y.source)==="upload").map(Y=>Y.path),g=getModelMediaSupport(get("model-select").value),y=d.some(Y=>isAudioPath(
Y)),b=d.some(Y=>isVideoPath(Y)),w=(get("model-select").value||"").toLowerCase(),x=get("enable-python"),
k=!!(x&&x.checked);if(y&&!g.audio||b&&!g.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&d.length===0)return;if(isMistralOcrModel(
w)){const Y=/https?:\/\/\S+/i.test(a);if(d.filter(Re=>isAudioPath(Re)||isVideoPath(Re)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!d.length&&!Y){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const C=a.trim();if(/^\/settings(?:\s|$)/i.
test(C)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(C)){const Y=C.replace(/^\/settings\s*/i,"").trim();
if(!Y){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const Re=get("prompt-input");Re.value="/settings ";const Ge=extractSlashCommandToken(Re.value);
lastSlashFilter=Ge,showSlashCommandSuggestions(Ge),Re.focus();return}const ke=get("model-select")?get(
"model-select").value:null;if(!ke){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
Y,ke);return}if(isGeminiLocalPythonMode(w,y,b,k)&&!await confirmGeminiLocalPythonSwitch())return;let T=null,
A=[];if(codingModeEnabled){const Y=collectCodingCandidates(a),ke=Y.filter(ot=>ot.prompt_source),Re=Y.
filter(ot=>!ot.prompt_source),Ge=ke.reduce((ot,Le)=>ot+String(Le.code||"").length,0);if(Ge>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let et=3e5-Ge;const mt=[];for(let ot=Re.length-1;ot>=0;ot--){const Le=String(Re[ot].
code||"").length;Le>et||(mt.unshift(Re[ot]),et-=Le)}A=codingTargetSelection?mt.slice(-1):[...ke,...mt];
const ft=ke.length?ke[ke.length-1]:null;if(T=codingTargetSelection?A[0]:ft||A[A.length-1]||null,codingModeEffective=
!!(T&&String(T.code||"").trim()),codingModeEffective&&T.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const ot=String(((Ot=get(
"model-select"))==null?void 0:Ot.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(ot)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const N=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${d.length} search=${get("enable-search").checked}`);const G=a,L=hasMarkerHint()?MARKER_HINT_TEXT:
null,$=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&d.length===0){showToast("Ma\
sk \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if($&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const M=userAutoScroll&&isChatNearBottom();M||pauseChatAutoScroll();const P=editingMessageId,
R=promptSettingsBeforeEdit;promptSettingsBeforeEdit=null;const D=currentParentId,H=P!=null;P&&(editingMessageId=
null,setEditUi(!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const ee=[],O=l(
Y=>{if(Y==null)return;let ke=document.getElementById(`msg-${Y}`);for(;ke;)ke.classList&&ke.classList.
contains("message-group")&&(ee.push({node:ke,prevDisplay:ke.style.display}),ke.style.display="none"),
ke=ke.nextElementSibling},"hideRenderedBranchFrom"),B=l(()=>{ee.forEach(({node:Y,prevDisplay:ke})=>{
Y&&(Y.style.display=ke||"")}),ee.length=0},"restoreHiddenBranch");P&&O(P);const J=Date.now(),F=isGptImageModel()||
$?currentMaskImage:null,q=renderMessage(J,"user",G,JSON.stringify(d),null,null,null,!1,currentQuote,
null,null,null,null,null,null,null,M,D,activeSkill?activeSkill.name:null,null,F);let te=!1;const V=/(https?:\/\/)?(x\.com|twitter\.com)\//i,
X=!i&&(V.test(G||"")||V.test(currentQuote||"")),re="grok-4-fast-reasoning",ce=l(()=>{get("enable-sea\
rch").checked=!0,get("model-select").value!==re&&selectModelById(re)},"applyXLinkAuto");if(X&&!isMistralOcrModel()&&
!get("enable-search").checked)if(autoSearchOnLinks)ce();else{const Y=get("auto-search-banner"),ke=get(
"auto-search-on-btn"),Re=get("auto-search-off-btn"),Ge=get("auto-search-remember");Y&&ke&&Re&&(Ge&&(Ge.
checked=!1),await new Promise(et=>{Y.classList.remove("hidden");const mt=l(ft=>{Y.classList.add("hid\
den"),ke.onclick=null,Re.onclick=null,et(ft)},"cleanup");ke.onclick=()=>mt("enable"),Re.onclick=()=>mt(
"disable")}).then(async et=>{et==="enable"?(ce(),Ge&&Ge.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):te=!0}))}const Q=String(get("reasoning-effort").value||"").
toLowerCase(),ue=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&Q==="none",
ne={client_request_id:createClientRequestId(),client_platform:"Web",deepseek_language:(document.documentElement.
lang||navigator.language||"").toLowerCase().startsWith("ja")?"\u65E5\u672C\u8A9E":(document.documentElement.
lang||navigator.language||"").toLowerCase().startsWith("zh")?"\u4E2D\u6587":"English",thread_id:currentThreadId,
message:G,model:get("model-select").value,image_urls:d,image_items:c,uploaded_image_urls:m,temporary_chat:temporaryChatEnabled,
enable_search:get("enable-search").checked,gemini_free_tier:!!(get("enable-gemini-free-tier")&&get("\
enable-gemini-free-tier").checked),gemini_flex:!!(get("enable-gemini-flex")&&get("enable-gemini-flex").
checked),enable_url_context:get("enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get(
"enable-maps")?get("enable-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),
enable_skills:get("enable-skills")?get("enable-skills").checked:!0,enable_file_creation:get("enable-\
file-creation")?get("enable-file-creation").checked:!0,enable_thinking:ue?!1:get("enable-thinking").
checked,thinking_level:get("thinking-level").value,thinking_budget:get("thinking-budget")?get("think\
ing-budget").value:null,reasoning_effort:get("reasoning-effort").value,enable_system_prompt:get("ena\
ble-sys-prompt").checked,enable_prompt_caching:get("enable-prompt-cache")?get("enable-prompt-cache").
checked:!1,marker_system_prompt:L,safety_setting:get("safety-setting").value,tts_voice:isTtsModel()&&
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
get("gpt-image-frame-duration")?get("gpt-image-frame-duration").value:null,image_mask:F,gemini_image_aspect:isGeminiImageModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:D,parent_id_explicit:H,disable_auto_search:te,
image_vision_model:currentVisionModel||null,coding_mode:N,coding_target:N?{id:T.candidate_id,code:T.
prompt_source?null:T.code,language:T.language||"text",key:T.key||null,message_id:T.message_id||null,
source:T.prompt_source?"prompt":"history",explicit:T.explicit===!0}:null,coding_candidates:N?A.map(Y=>({
id:Y.candidate_id,source:Y.prompt_source?"prompt":"history",prompt_index:Y.prompt_source?Y.prompt_index:
null,code:Y.prompt_source?null:Y.code,language:Y.language||"text",explicit:Y.explicit===!0})):[],batch_mode:r,
canvas_mode:!!canvasModeEnabled,math_steps:typeof window.collectMathStepsSendOption=="function"?window.
collectMathStepsSendOption():!1,project_id:!currentThreadId&&typeof window.getActiveProjectId=="func\
tion"?window.getActiveProjectId():null};typeof window.hideProjectViewForChat=="function"&&window.hideProjectViewForChat(),
n&&(ne.turnstile_token=n);const xe=get("thread-custom-instruction");xe&&(ne.thread_custom_instruction=
xe.value||""),activeSkill?(ne.system_prompt=activeSkill.instruction,ne.enable_system_prompt=!0,ne.skill_uuid=
activeSkill.uuid):ne.skill_uuid=null;const ve=takePromptForSend(e.animate!==!1),we=animatePromptToMessage(
q,ve,M&&e.animate!==!1);setSendBtnToStopMode();const Oe="ai-"+Date.now(),Ce=String(ne.model||"").toLowerCase(),
lt=!!ne.enable_thinking||!!Q&&Q!=="none",fe=Ce.includes("gemini")||Ce.includes("o1")||Ce.includes("o\
3")||Ce.includes("gpt-5")||Ce.includes("reasoning")&&!Ce.includes("non-reasoning"),ye=lt&&fe;let ge=buildPendingSkeletonHtml(
ne.model,"API\u306B\u9001\u4FE1\u4E2D...");get("chat-container").insertAdjacentHTML("beforeend",`<di\
v class="flex justify-start mb-4 ai-pending-row fade-in"><div id="${Oe}" class="message-bubble ai-pe\
nding-bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${ge}</div><\
/div>`),M&&resumeChatAutoScroll();const ie=get(Oe),it=r?null:startPendingElapsed(ie,t);activeStreamingBubbleId=
Oe,canvasModeEnabled&&resetCanvasPreviewPanel();let st=null;const vt=l(Y=>!ye||!ie?null:((!st||!ie.contains(
st))&&(st=ie.querySelector(".thought-content")),st||(ie.insertAdjacentHTML("afterbegin",'<div class=\
"thought-container"><div class="thought-header thinking-shimmer" onclick="toggleThinking(this)"><i c\
lass="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content collapsed\
" data-placeholder="1"></div></div>'),st=ie.querySelector(".thought-content")),st&&(st.setAttribute(
"data-placeholder","1"),st.textContent=Y||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
st),"ensureThoughtPlaceholder");ye&&vt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
abortController=new AbortController;const Mt=currentThreadId,He=nowPerfMs(),at=Date.now();let Ke=!1,
Lt=!1,Rt=!1,At=null,qe=null,ut=null,Ve=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):
null;const wt=l((Y,ke)=>{if(!ke||Y==="status"&&Ke||Y==="thought"&&Lt||Y==="content"&&Rt)return;const Re=Math.
max(0,nowPerfMs()-He);Y==="status"?At=Re:Y==="thought"?qe=Re:Y==="content"&&(ut=Re),reportFirstTokenLatency(
{latency_seconds:Re/1e3,latency_ms:Re,thread_id:Ve||currentThreadId,job_id:currentJobId,model:ne.model,
first_event_type:Y,client_sent_at_ms:at}),Y==="status"?Ke=!0:Y==="thought"?Lt=!0:Y==="content"&&(Rt=
!0)},"maybeReportFirstEventLatency"),tt=window.ProgressSpinner?window.ProgressSpinner.startFlow("cha\
t"):null;let pt=!1,Gt=!1,Bt=null,Et=null,Ut=!1;try{ne.thread_id&&activeSkill&&(threadSkillMap[ne.thread_id]=
activeSkill,pendingSkillForNewThread=null);const Y=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.
urls.chatStream,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify(ne),signal:abortController.signal}),ie);if(sendClientDebugLog("info",`Prompt str\
eam response status: ${Y.status}`),!Y.ok){const Ue=await Y.json().catch(()=>({})),Xe=new Error(Ue.error||
`HTTP ${Y.status}`);throw Xe.serverCode=Ue.code||null,Xe.serverModel=Ue.model||ne.model,Xe.acceptedJobId=
Ue.job_id||null,Xe.acceptedThreadId=Ue.thread_id||null,Xe}pt=!0,window.ConnectionMonitor&&(Ut=!0,window.
ConnectionMonitor.operationStarted()),tt&&tt.setPhase("waiting"),ve.accept(),codingModeEnabled&&syncCodingModeUi(
!0,{persist:!1});const ke=l(()=>{if(!ie)return;const Ue=ie.querySelector(".content-area");if(Ue&&Ue.
getAttribute("data-api-accepted")!=="1"&&(Ue.setAttribute("data-api-accepted","1"),!updatePendingSkeletonStatus(
ie,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){
Ue.outerHTML=buildPendingSkeletonHtml(ne.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const Xe=ie.querySelector(".content-area");Xe&&Xe.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
ie,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");ke(),(jt=window.ActivityLog)==null||jt.log("send.stream.start");let Re=!1;const Ge=Y.
body.getReader(),et=new TextDecoder;let mt="",ft="",ot="",Le=!0,$e=null,We=null,nt=null,Ct=!1;const $t={};
let Nt=0,xt=!1;for(;!xt;){const{done:Ue,value:Xe}=await Ge.read();if(!Ue&&!Re&&(Re=!0,(Ft=window.ActivityLog)==
null||Ft.log("send.stream.first")),Ue&&((zt=window.ActivityLog)==null||zt.log("send.stream.done")),Ue)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),tt&&tt.setPhase("receiving"),
mt+=et.decode(Xe,{stream:!0});let Qt=mt.split(`
`);mt=Qt.pop();let Ht=!1,Zt=!1;for(let Vt of Qt)if(Vt.trim())try{const Se=JSON.parse(Vt);if(Se.type===
"thread_id"){ke();const Be=Se.content!==null&&Se.content!==void 0?String(Se.content):Se.content;Be&&
(Ve=Be,currentThreadId!==Be&&(currentThreadId=Be,history.pushState({},"","/c/"+Be)),activeSkill&&(threadSkillMap[Be]=
activeSkill,pendingSkillForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Se.type==="\
job_id"){ke(),currentJobId=Se.content,ie&&ie.setAttribute("data-job-id",String(currentJobId||"")),r&&
showToast("Batch\u767B\u9332","info");continue}if(Se.type==="search_status"){nt=updateSearchBox(ie,nt,
Se.content);continue}if(Se.type==="mcp"){handleMcpStreamEvent(ie,Se.content||{});continue}if(Se.type===
"mcp_decision_request"){openMcpDecisionModal(Se.content||{});continue}if(Se.type==="status"){ke();const Be=Se.
content===null||Se.content===void 0?"":String(Se.content);if(wt("status",!!Be),Le&&ie){const bt=Be||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(ie,bt,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const kt=ie.querySelector(".content-area");kt&&(kt.outerHTML=buildPendingSkeletonHtml(ne.model,bt),updatePendingSkeletonStatus(
ie,bt,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
ye&&vt(Be||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(Le){beginPendingToStreamTransition(
ie);const Be=ie.querySelector(".content-area");Be&&(Be.innerHTML=""),Le=!1}if(Se.type==="coding_diff")
appendCodingLiveDiff(ie,Se.content||{}),wt("content",!0);else if(Se.type==="thought"){if($e||($e=ie.
querySelector(".thought-content")),ot+=Se.content,wt("thought",!!Se.content),!$e){const Be='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';nt?nt.insertAdjacentHTML(
"afterend",Be):ie.insertAdjacentHTML("afterbegin",Be),$e=ie.querySelector(".thought-content")}if($e&&
$e.getAttribute("data-placeholder")==="1"){if($e.textContent="",$e.removeAttribute("data-placeholder"),
$e){const Be=$e.parentElement.querySelector(".thought-header");Be&&Be.classList.remove("thinking-shi\
mmer")}ot=Se.content}$e.classList.remove("collapsed"),Zt=!0}else if(Se.type==="image_analysis"){const Be=Se.
content===null||Se.content===void 0?"":String(Se.content);if(!ie)continue;let bt=ie.querySelector(".\
image-analysis-box");if(!bt){const dt='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';nt?nt.insertAdjacentHTML("afterend",dt):ie.insertAdjacentHTML("afterbegin",dt),bt=ie.querySelector(
".image-analysis-box")}const kt=bt.querySelector(".image-analysis-text");kt&&(kt.textContent=Be)}else if(Se.
type==="python"){const Be=Se.content||{},bt=Be.id||`py_${Date.now()}`;if(!$t[bt]){const dt=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${bt}" data-collapsed="true" data-code-key="${bt}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;nt?nt.insertAdjacentHTML("afterend",dt):ie.insertAdjacentHTML(
"afterbegin",dt),$t[bt]=ie.querySelector(`[data-py-id="${bt}"]`)}const kt=$t[bt];if(kt){if(Be.code!==
void 0){const dt=Be.code==null?"":String(Be.code),Wt=kt.querySelector(".python-code");Wt&&(Wt.textContent=
dt,Wt.removeAttribute("data-highlighted"),queueHighlight(kt,dt));const en=kt.querySelector('.copy-bt\
n[data-copy="code"]');en&&en.setAttribute("data-code",encodeURIComponent(dt).replace(/'/g,"%27"))}if(Be.
output!==void 0){const dt=Be.output==null?"":String(Be.output),Wt=kt.querySelector(".python-output");
Wt&&(Wt.textContent=dt);const en=kt.querySelector('.copy-btn[data-copy="output"]');en&&en.setAttribute(
"data-code",encodeURIComponent(dt).replace(/'/g,"%27"))}}}else if(Se.type==="content_replace")ft=Se.
content===null||Se.content===void 0?"":String(Se.content),We||(We=ie.querySelector(".content-area")||
document.createElement("div"),We.className="prose prose-invert text-sm break-words",ie.contains(We)||
ie.appendChild(We)),Ht=!0;else if(Se.type==="content"){const Be=Se.content===null||Se.content===void 0?
"":String(Se.content);ft+=Be,/[`~]/.test(Be)&&activateDeferredCodingModeFromStream(ft),We||(We=ie.querySelector(
".content-area")||document.createElement("div"),We.className="prose prose-invert text-sm break-words",
ie.contains(We)||ie.appendChild(We)),Ht=!0,wt("content",!!Be)}else if(Se.type==="error"){Ct=!0,xt=!0,
ie.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Se.content)),showToast(Se.content||"Unkno\
wn error","error",!0);break}}catch{}if(Zt&&$e&&($e.textContent=ot,userAutoScroll&&($e.scrollTop=$e.scrollHeight)),
Ht&&We){const Vt=Date.now();if(Vt-Nt>100){const Se=snapshotCodeCollapse(We);renderAiMarkdownInto(We,
ft,{incrementalMath:!0,streaming:!0,mathSteps:{jobId:currentJobId,streaming:!0}}),applyCodeCollapse(
We,Se,!0),Nt=Vt}}scrollToBottom()}if(tt&&tt(),We){const Ue=snapshotCodeCollapse(We);renderAiMarkdownInto(
We,ft,{incrementalMath:!0,streaming:!1,mathSteps:{jobId:currentJobId,streaming:!1}}),applyCodeCollapse(
We,Ue,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),ie){queueHighlight(ie,ft);const Ue=nowPerfMs()-
He;enableLatencyMetrics&&reportFirstTokenLatency({is_total:!0,latency_seconds:Ue/1e3,latency_ms:Ue,thread_id:Ve||
currentThreadId,job_id:currentJobId,model:ne.model,client_sent_at_ms:at,client_done_at_ms:Date.now()});
let Xe=null;At!==null&&(Xe=At),qe!==null&&(Xe===null||qe<Xe)&&(Xe=qe),ut!==null&&(Xe===null||ut<Xe)&&
(Xe=ut);const Qt={total_ms:Ue,first_response_ms:Xe,phases:Xe!==null?[{key:"generation",ms:Math.max(0,
Ue-Xe)}]:[],source:"client"};messageMeta[Oe]=Object.assign(messageMeta[Oe]||{},{role:"assistant",model:ne.
model,processing_time:Qt});const Ht=ie.querySelector(".processing-time-row");Ht&&Ht.remove(),ie.insertAdjacentHTML(
"beforeend",buildProcessingTimeHtml(Oe,Qt))}editingMessageId=null,setEditUi(!1),ie&&ie.querySelectorAll(
".thought-content").forEach(Xe=>Xe.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0,forceLatestLeaf:!!P}),!Ct&&codingModeEnabled&&(codingTargetSelection=null,syncCodingModeUi(
!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.querySelectorAll(".message-group").length<=
2||!currentThreadTitle||currentThreadTitle==="New Chat"||currentThreadTitle==="No Title"?apiFetch("/\
api/generate_title",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
thread_id:currentThreadId,model_id:get("model-select").value})}).then(Ue=>Ue.json()).then(Ue=>{Ue.title&&
(document.title=Ue.title+" - AI Chat",setCurrentChatHeaderTitle(Ue.title),loadThreads())}):loadThreads(
!1)}catch(Y){(Dt=window.ActivityLog)==null||Dt.log(Y.name==="AbortError"?"send.cancelled":"send.erro\
r"),we();let ke=!1;const Re=Y.name==="AbortError"&&isManualStopAbortForThread(Mt);if(Y.name==="Abort\
Error"&&!Re&&(ke=await syncThreadAfterAbortedStream(Mt,{retries:2,retryDelayMs:180,notifyOnFailure:!0})),
sendClientDebugLog("error",`Prompt send error: ${Y.message}`),!pt){q&&q.remove();const Ge=ie&&ie.closest(
".fade-in");Ge&&Ge.remove(),delete messageStore[J],delete messageMeta[J]}if(Y.serverCode==="request_\
already_accepted"&&Y.acceptedJobId&&Y.acceptedThreadId)pt=!0,Bt={job_id:Y.acceptedJobId,thread_id:String(
Y.acceptedThreadId),model:ne.model,pendingStartedAt:r?null:t},ve.accept();else if(pt&&!Re)Et={job_id:normalizeJobIdForUi(
currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):null,model:ne.model,pendingStartedAt:r?
null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\
\u307E\u3059\u3002","warning",!1);else if(Y.serverCode==="turnstile_required"){const Ge=await getTurnstileToken();
Ge?(await verifyTurnstileOnServer(Ge,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(Y.serverCode==="api_key_missing"){const Ge=Y.serverModel||ne.model,et=await showApiKeyRequiredModalAsync(
Ge);et==="set"?Gt=!0:et==="switch"?showModal("model-modal"):showToast(Y.message||`${getModelNameById(
Ge)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(Y.
name!=="AbortError"){const Ge="Connection Error: "+Y.message;showToast(Ge,"error",!0)}if(!pt){const Ge=P&&
ve.canRetry()&&!editingMessageId;ve.restore(),Ge&&(editingMessageId=P,currentParentId=D,promptSettingsBeforeEdit=
R,setEditUi(!0))}P&&!ke&&B()}finally{it&&it(),Ut&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),tt&&tt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===Oe&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,setEditUi(!!editingMessageId)}if(Bt){const Y=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=Bt.thread_id,(Y!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
Bt,currentThreadId)}if(Et&&Et.thread_id)return reconnectPendingStreamUntilAvailable(Et,Et.thread_id);
if(Gt&&ve.canRetry())return sendMessageLogged({animate:!1})}l(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var R,D,H,ee;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const O=a.
querySelector(".content-area");O?O.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const o=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,a&&a.setAttribute("data-job-id",String(
t||"")),setSendBtnToStopMode(),resumeChatAutoScroll(),canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=
new AbortController;const r=currentThreadId,c=i.toLowerCase(),d=c.includes("gemini")||c.includes("o1")||
c.includes("o3")||c.includes("gpt-5")||c.includes("reasoning")&&!c.includes("non-reasoning");let m=null;
const g=l(O=>!d||!a?null:((!m||!a.contains(m))&&(m=a.querySelector(".thought-content")),m||(a.insertAdjacentHTML(
"afterbegin",'<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="t\
oggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="\
thought-content collapsed" data-placeholder="1"></div></div>'),m=a.querySelector(".thought-content")),
m&&(m.setAttribute("data-placeholder","1"),m.textContent=O||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
m),"ensureThoughtPlaceholder");d&&g("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let y="",b="",w="",x=!0,k=null,C=null,T=null,A=!1;const N={};let G=0,L=!1;const $=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let M=!1,P=!1;try{const O=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!O.ok)throw new Error(`Resume failed (${O.status})`);window.ConnectionMonitor&&
(P=!0,window.ConnectionMonitor.operationStarted()),$&&$.setPhase("waiting"),(R=window.ActivityLog)==
null||R.log("send.stream.start");let B=!1;const J=O.body.getReader(),F=new TextDecoder;for(;!L;){const{
done:q,value:te}=await J.read();if(!q&&!B&&(B=!0,(D=window.ActivityLog)==null||D.log("send.stream.fi\
rst")),q&&((H=window.ActivityLog)==null||H.log("send.stream.done")),q)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),$&&$.setPhase("receiving"),y+=F.decode(te,{stream:!0});let V=y.
split(`
`);y=V.pop();let X=!1,re=!1;for(let ce of V)if(ce.trim())try{const Q=JSON.parse(ce);if(Q.type==="job\
_id"){currentJobId=Q.content||t,a&&a.setAttribute("data-job-id",String(currentJobId||""));continue}if(Q.
type==="search_status"){T=updateSearchBox(a,T,Q.content);continue}if(Q.type==="mcp"){handleMcpStreamEvent(
a,Q.content||{});continue}if(Q.type==="mcp_decision_request"){openMcpDecisionModal(Q.content||{});continue}
if(Q.type==="status"){const ue=Q.content===null||Q.content===void 0?"":String(Q.content);if(x&&a){const ne=ue||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(a,ne,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const xe=a.querySelector(".content-area");xe&&(xe.outerHTML=buildPendingSkeletonHtml(i,ne),updatePendingSkeletonStatus(
a,ne,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
d&&g(ue||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(x){beginPendingToStreamTransition(
a);const ue=a.querySelector(".content-area");ue&&(ue.innerHTML=""),x=!1}if(Q.type==="coding_diff")appendCodingLiveDiff(
a,Q.content||{});else if(Q.type==="thought"){if(k||(k=a.querySelector(".thought-content")),w+=Q.content,
!k){const ue='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content">\
</div></div>';T?T.insertAdjacentHTML("afterend",ue):a.insertAdjacentHTML("afterbegin",ue),k=a.querySelector(
".thought-content")}if(k&&k.getAttribute("data-placeholder")==="1"){if(k.textContent="",k.removeAttribute(
"data-placeholder"),k){const ue=k.parentElement.querySelector(".thought-header");ue&&ue.classList.remove(
"thinking-shimmer")}w=Q.content}k.classList.remove("collapsed"),re=!0}else if(Q.type==="image_analys\
is"){const ue=Q.content===null||Q.content===void 0?"":String(Q.content);if(!a)continue;let ne=a.querySelector(
".image-analysis-box");if(!ne){const ve='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 bord\
er border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas\
 fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"><\
/div></div>';T?T.insertAdjacentHTML("afterend",ve):a.insertAdjacentHTML("afterbegin",ve),ne=a.querySelector(
".image-analysis-box")}const xe=ne.querySelector(".image-analysis-text");xe&&(xe.textContent=ue)}else if(Q.
type==="python"){const ue=Q.content||{},ne=ue.id||`py_${Date.now()}`;if(!N[ne]){const ve=`<div class\
="code-wrapper python-box collapsed" data-py-id="${ne}" data-collapsed="true" data-code-key="${ne}">\
<div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</s\
pan><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-labe\
l="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-co\
de="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-b\
tn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left">\
</i></button></div></div><div class="code-body"><div class="python-section"><div class="python-label\
">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pytho\
n-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-ou\
tput"></code></pre></div></div></div>`;T?T.insertAdjacentHTML("afterend",ve):a.insertAdjacentHTML("a\
fterbegin",ve),N[ne]=a.querySelector(`[data-py-id="${ne}"]`)}const xe=N[ne];if(xe){if(ue.code!==void 0){
const ve=ue.code==null?"":String(ue.code),we=xe.querySelector(".python-code");we&&(we.textContent=ve,
we.removeAttribute("data-highlighted"),queueHighlight(xe,ve));const Oe=xe.querySelector('.copy-btn[d\
ata-copy="code"]');Oe&&Oe.setAttribute("data-code",encodeURIComponent(ve).replace(/'/g,"%27"))}if(ue.
output!==void 0){const ve=ue.output==null?"":String(ue.output),we=xe.querySelector(".python-output");
we&&(we.textContent=ve);const Oe=xe.querySelector('.copy-btn[data-copy="output"]');Oe&&Oe.setAttribute(
"data-code",encodeURIComponent(ve).replace(/'/g,"%27"))}}}else if(Q.type==="content"){const ue=Q.content===
null||Q.content===void 0?"":String(Q.content);b+=ue,/[`~]/.test(ue)&&activateDeferredCodingModeFromStream(
b),C||(C=a.querySelector(".content-area")||document.createElement("div"),C.className="prose prose-in\
vert text-sm break-words",a.contains(C)||a.appendChild(C)),X=!0}else if(Q.type==="error"){A=!0,L=!0,
a.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Q.content)),showToast(Q.content||"Unknown \
error","error",!0);break}}catch{}if(re&&k&&(k.textContent=w,userAutoScroll&&(k.scrollTop=k.scrollHeight)),
X&&C){const ce=Date.now();if(ce-G>100){const Q=snapshotCodeCollapse(C);renderAiMarkdownInto(C,b,{incrementalMath:!0,
streaming:!0,mathSteps:{jobId:t,streaming:!0}}),applyCodeCollapse(C,Q,!0),G=ce}}scrollToBottom()}if($&&
$(),C){const q=snapshotCodeCollapse(C);renderAiMarkdownInto(C,b,{incrementalMath:!0,streaming:!1,mathSteps:{
jobId:currentJobId||t,streaming:!1}}),applyCodeCollapse(C,q,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(
a,b),a&&a.querySelectorAll(".thought-content").forEach(te=>te.classList.add("collapsed")),await loadMessages(
currentThreadId,{preserveDraft:!0,silent:!0}),loadThreads(!1)}catch(O){(ee=window.ActivityLog)==null||
ee.log(O.name==="AbortError"?"send.cancelled":"send.stream.error");const B=O.name==="AbortError"&&isManualStopAbortForThread(
r);O.name==="AbortError"&&!B&&await syncThreadAfterAbortedStream(r,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
B||(M=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{o&&o(),P&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
$&&$(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,currentThreadPending=null}if(M)return reconnectPendingStreamUntilAvailable(
{job_id:t,model:i,pendingStartedAt:e.pendingStartedAt},r)}l(resumePendingStream,"resumePendingStream");
function updateThreadHighlighting(){const e=get("thread-list");if(!e)return;e.querySelectorAll("[dat\
a-thread-id]").forEach(n=>{n.dataset.threadId===String(currentThreadId)?n.classList.add("bg-gray-700\
/60","border-l-2","border-blue-500"):n.classList.remove("bg-gray-700/60","border-l-2","border-blue-5\
00")})}l(updateThreadHighlighting,"updateThreadHighlighting");async function loadThreads(e=!1){if(threadLoading){
snapshotSidebarHistory("loadThreads-skipped-busy append="+!!e);return}threadLoading=!0,snapshotSidebarHistory(
"loadThreads-start append="+!!e);try{e||(threadPage=1,hasMoreThreads=!0);const t=get("search-box"),n=t?
t.value:"";if(!e&&isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skipped-settings-open");
return}const i=String(n||"").trim()?"":"&project_scope=unassigned",o=await(await apiFetch(`${CHAT_CONFIG.
urls.handleThreads}?q=${encodeURIComponent(n)}&page=${threadPage}${i}`)).json(),r=get("thread-list");
if(!r)return;if(!e){if(isSettingsModalOpen()){snapshotSidebarHistory("loadThreads-skip-replace-setti\
ngs-open");return}const d=o&&Array.isArray(o.threads)?o.threads.length:-1,m=r.querySelectorAll("[dat\
a-thread-id]").length;if(d===0&&m>0&&String(n||"").trim()){snapshotSidebarHistory("loadThreads-keep-\
existing-empty-search");return}if(r.innerHTML='<div id="thread-pull-indicator" class="ptr-pull-indic\
ator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="fas fa-spinner fa-\
spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scroll-sentinel"></div\
>',threadObserver){threadObserver.disconnect();const g=get("scroll-sentinel");g&&threadObserver.observe(
g)}}const c=get("scroll-sentinel");o&&Array.isArray(o.threads)?(o.threads.forEach(d=>{const m=String(
d.id),g=document.createElement("div"),y=d.is_bookmarked?"text-yellow-400":"text-gray-500",b=d.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=d.project_name?`<span class="text-[9px] text-amber-200 bg-amber-900/30 border border-amber-500/\
40 rounded px-1 py-0 shrink-0 max-w-[76px] truncate" title="${escapeHtml(d.project_name)}">${escapeHtml(
d.project_name)}</span>`:"",k=m===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":
"";g.className=`p-2 rounded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex jus\
tify-between items-center group ${k}`,g.dataset.threadId=m,g.dataset.threadTitle=d.title||"",g.innerHTML=
`<div class="flex items-center gap-1 truncate flex-1"><button class="${y} hover:text-yellow-400 px-1\
" onclick="toggleBookmark(event, '${m}')"><i class="fas fa-star text-[10px]"></i></button><span clas\
s="truncate">${escapeHtml(d.title||"No Title")}</span>${w}${b}</div><div class="flex items-center ga\
p-1 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition" data-thread-actions="1"><button \
class="text-gray-500 hover:text-amber-300 px-1 transition" title="\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u3078\u79FB\u52D5" data-move-project-btn="\
1"><i class="fas fa-folder-open text-xs"></i></button><button class="text-gray-500 hover:text-white \
px-1 transition" onclick="renameThread(event, '${m}')"><i class="fas fa-pen text-xs"></i></button><b\
utton class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, '${m}')"\
><i class="fas fa-trash text-xs"></i></button></div>`;const C=g.querySelector("[data-move-project-bt\
n]");C&&(C.onclick=T=>{typeof window.openMoveThreadProjectModal=="function"&&window.openMoveThreadProjectModal(
T,m,d.title||"No Title",d.project_id||null)}),g.onclick=T=>{T.target.closest("button")||T.target.closest(
"[data-thread-actions]")||loadMessages(m)},c?r.insertBefore(g,c):r.appendChild(g)}),hasMoreThreads=!!o.
has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendered count="+o.threads.
length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){console.error(
"Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=!1,updateThreadHighlighting(),
snapshotSidebarHistory("loadThreads-finally")}}l(loadThreads,"loadThreads");function initPullToRefresh(e,t){
const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,o=88,r=52,c=.5,d=8;let m=0,g=!1,y=0,b=null;
const w=l(()=>get(i),"indicatorEl"),x=l(()=>{const T=w();return T?T.querySelector(".ptr-pull-label"):
null},"labelEl"),k=l(T=>{const A=w();if(!A)return;A.style.height=Math.min(T,o)+"px",A.classList.toggle(
"active",T>2),A.classList.toggle("pull-ready",T>=a);const N=x();N&&(N.textContent=T>=a?"\u96E2\u3057\u3066\u66F4\u65B0":
"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},"applyPullUI"),C=l(()=>{const T=w();T&&(T.style.height=
"0px",T.classList.remove("active","pull-ready","refreshing"),T.classList.remove("dragging"))},"reset\
PullUI");n.addEventListener("touchstart",T=>{if(b){g=!1;return}if(n.scrollTop>0){g=!1;return}const A=T.
touches[0];A&&(m=A.clientY,y=0,g=!0)},{passive:!0}),n.addEventListener("touchmove",T=>{if(!g||b)return;
if(n.scrollTop>0){g=!1;return}const A=T.touches[0];if(!A)return;const N=A.clientY-m;if(N<=0){y>0&&(y=
0,k(0)),g=!1;return}const G=w();G&&!G.classList.contains("dragging")&&G.classList.add("dragging"),y=
Math.min(N*c,o),k(y),N>=d&&T.preventDefault()},{passive:!1}),n.addEventListener("touchend",()=>{if(!g||
(g=!1,b))return;const T=w();T&&T.classList.remove("dragging");const A=y>=a;if(y=0,!A){C();return}let N;
try{N=t()}catch{N=null}const G=w();if(G){G.classList.add("refreshing"),G.style.height=r+"px";const L=G.
querySelector(".ptr-pull-label");L&&(L.textContent="\u66F4\u65B0\u4E2D...")}N&&typeof N.then=="funct\
ion"?(b=N,N.catch(()=>{}).finally(()=>{b=null,C()})):(b=Promise.resolve(),setTimeout(()=>{b=null,C()},
400))}),n.addEventListener("touchcancel",()=>{g=!1,y=0,C()})}l(initPullToRefresh,"initPullToRefresh");
const initThreadPullToRefresh=l(()=>initPullToRefresh("thread-list",()=>loadThreads(!1)),"initThread\
PullToRefresh"),initSkillPullToRefresh=l(()=>initPullToRefresh("skill-list",()=>loadSkills()),"initS\
killPullToRefresh"),initPullToRefreshAll=l(()=>{initThreadPullToRefresh(),initSkillPullToRefresh()},
"initPullToRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;const mcpCardIdSelector=l(
e=>"mcp_card_"+String(e).replace(/[^A-Za-z0-9_-]/g,"_"),"mcpCardIdSelector"),mcpEscHtml=l(e=>String(
e==null?"":e).replace(/[&<>"']/g,t=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[t]),
"mcpEscHtml");function mcpCardTitle(e){return`${mcpEscHtml(e.server_name||"MCP")} / ${mcpEscHtml(e.tool_name||
e.internal_name||"")}`}l(mcpCardTitle,"mcpCardTitle");function getMcpExecutionList(e){if(!e)return null;
let t=e.querySelector(".mcp-execution-list");return t||(t=document.createElement("div"),t.className=
"mcp-execution-list mt-3",t.setAttribute("aria-label","MCP\u30C4\u30FC\u30EB\u5B9F\u884C"),e.appendChild(
t)),t}l(getMcpExecutionList,"getMcpExecutionList");function handleMcpStreamEvent(e,t){if(!e||!t||!t.
type)return;const n=["start","result","error"].includes(t.type),i=n?getMcpExecutionList(e):null;if(n&&
!i)return;const a=mcpCardIdSelector(t.id||"mcp_"+Date.now());if(t.type==="start"){if(i.querySelector(
'[data-mcp-card="'+a+'"]'))return;const o=`<div class="mcp-box mcp-running mb-2" data-mcp-card="${a}\
">
    <span class="mcp-spinner"></span>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u4E2D...</span>
</div>`;i.insertAdjacentHTML("beforeend",o);return}if(t.type==="result"){let o=i.querySelector('[dat\
a-mcp-card="'+a+'"]');const r=t.summary||"";if(o)o.classList.remove("mcp-running"),o.classList.add("\
mcp-done"),o.innerHTML=`<i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>`;else{const c=`<div class=\
"mcp-box mcp-done mb-2" data-mcp-card="${a}">
    <i class="fas fa-check-circle mcp-box-ok"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5B9F\u884C\u3057\u307E\u3057\u305F</span>
</div>`;i.insertAdjacentHTML("beforeend",c),o=i.querySelector('[data-mcp-card="'+a+'"]')}if(r){const c=document.
createElement("div");c.className="mcp-box-note",c.textContent=r.split(`
`)[0].slice(0,220),o&&o.appendChild(c)}return}if(t.type==="error"){let o=i.querySelector('[data-mcp-\
card="'+a+'"]');const r=t.message||"MCP\u30C4\u30FC\u30EB\u306E\u5B9F\u884C\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
if(o)o.classList.remove("mcp-running"),o.classList.add("mcp-error"),o.innerHTML=`<i class="fas fa-ti\
mes-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const d=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",d),o=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(r).slice(0,300),
o&&o.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const o=get("mcp-decision-modal");if(o&&!o.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}l(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const d=JSON.parse(c);c=JSON.stringify(d,null,2)}catch{}a.textContent=c}const o=get(
"mcp-decision-allow"),r=get("mcp-decision-deny");o&&(o.onclick=()=>submitMcpDecision("allow")),r&&(r.
onclick=()=>submitMcpDecision("deny"));try{showModal("mcp-decision-modal")}catch{}}l(openMcpDecisionModal,
"openMcpDecisionModal");async function submitMcpDecision(e){const t=get("mcp-decision-modal");try{t&&
hideModal("mcp-decision-modal")}catch{}const n=activeMcpDecision?activeMcpDecision.jobId:null,i=activeMcpDecision?
activeMcpDecision.id:null;if(activeMcpDecision=null,!!n)try{await apiFetch("/api/mcp/chat/"+encodeURIComponent(
n)+"/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision:e,
id:i})})}catch{}}l(submitMcpDecision,"submitMcpDecision"),document.readyState==="loading"?document.addEventListener(
"DOMContentLoaded",initPullToRefreshAll,{once:!0}):initPullToRefreshAll();let geminiBatchStatusPollBusy=!1;
function showGeminiBatchCompletionBanner(e){const t=get("batch-notification-banner"),n=get("batch-no\
tification-text"),i=get("batch-notification-open");if(!t||!n||!e||!e.length)return;const a=e[0],o=a.
thread_id;n.textContent=e.length===1?`${a.model} \u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`:
`${e.length}\u4EF6\u306EBatch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002`,t.classList.
remove("hidden"),i&&(i.onclick=async()=>{t.classList.add("hidden"),o&&await loadMessages(o)});const r=get(
"batch-notification-close");r&&(r.onclick=()=>t.classList.add("hidden"))}l(showGeminiBatchCompletionBanner,
"showGeminiBatchCompletionBanner");async function refreshGeminiBatchStatus(){if(!geminiBatchStatusPollBusy){
geminiBatchStatusPollBusy=!0;try{const e=await apiFetch("/api/gemini/batch/status");if(!e.ok)return;
const t=await e.json().catch(()=>({}));(t.active||[]).some(a=>currentThreadId&&String(a.thread_id)===
String(currentThreadId))&&await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0});const i=t.
completed||[];i.length&&(showGeminiBatchCompletionBanner(i),i.some(a=>String(a.thread_id)===String(currentThreadId))&&
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0}))}catch{}finally{geminiBatchStatusPollBusy=
!1}}}l(refreshGeminiBatchStatus,"refreshGeminiBatchStatus"),refreshGeminiBatchStatus(),setInterval(refreshGeminiBatchStatus,
2e3);let chatTransitionSequence=0,chatTransitionTimer=null;const CHAT_TRANSITION_DURATION_MS=460;function playChatTransition(e){
const t=get("chat-transition-veil");if(!t)return;const n=++chatTransitionSequence;if(chatTransitionTimer&&
(clearTimeout(chatTransitionTimer),chatTransitionTimer=null),window.matchMedia&&window.matchMedia("(\
prefers-reduced-motion: reduce)").matches){t.classList.remove("is-active"),t.removeAttribute("data-t\
ransition-kind");return}t.dataset.transitionKind=e||"history",t.classList.remove("is-active"),t.offsetWidth,
t.classList.add("is-active"),chatTransitionTimer=setTimeout(()=>{n===chatTransitionSequence&&(t.classList.
remove("is-active"),chatTransitionTimer=null)},CHAT_TRANSITION_DURATION_MS)}l(playChatTransition,"pl\
ayChatTransition");async function toggleBookmark(e,t){e&&e.stopPropagation(),await apiFetch(`/api/th\
reads/${t}/bookmark`,{method:"POST"}),loadThreads()}l(toggleBookmark,"toggleBookmark");async function loadMessages(e,t={}){
const n=++threadLoadSequence,i=t.targetMessageId||new URLSearchParams(location.search).get("message");
window.closeHistoryModal&&window.closeHistoryModal();const a=!!t.preserveDraft,o=!!t.silent;o||resumeChatAutoScroll(
{scroll:!1});const r=o?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",d="",m=[];
if(a){const y=get("prompt-input");c=y?y.value:"",d=y?y.style.height:"",m=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,promptSettingsBeforeEdit=null,setEditUi(!1)}else cancelEdit();o||playChatTransition(
"history");const g=e!=null?String(e):e;g!==currentThreadId&&(modelChosenManually=!1),currentThreadId=
g,t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveSkillForThread(
currentThreadId),get("welcome-screen").classList.add("hidden"),o||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const y=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);y.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const b=i;b&&/^\d+$/.test(String(b))&&y.
searchParams.set("before_id",String(Number(b)+1));const w=await apiFetch(y.toString());if(!w.ok)throw new Error(
`thread request failed (${w.status})`);const x=await w.json();if(!x||!Array.isArray(x.messages))throw new Error(
"invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(x&&x.title),
typeof window.hideProjectViewForChat=="function"&&window.hideProjectViewForChat(),typeof window.setActiveProjectContext==
"function"&&window.setActiveProjectContext(x&&x.project_id?x.project_id:null,x&&x.project_name?x.project_name:
""),allMessages=x.messages,renderBranchOrigin(x.branch_origin),threadHasOlderMessages=!!x.has_older_messages,
oldestLoadedMessageId=x.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const k=(allMessages||
[]).filter(A=>A.role==="user"&&A.content).map(A=>A.content);if(promptHistory=[...new Set(k.slice().reverse())],
historyIndex=-1,tempPrompt="",currentThreadPending=x.pending_job||null,setTemporaryChatUiState(!!(x&&
x.is_temporary)),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(!0),get("thread-c\
ustom-instruction")&&(get("thread-custom-instruction").value=x.custom_instruction||""),get("enable-p\
rompt-cache")&&(get("enable-prompt-cache").checked=!!x.enable_prompt_caching,updatePromptCacheUi()),
x.last_model&&!modelChosenManually&&selectModelById(x.last_model),x.last_skill_uuid&&loadedSkills.length>
0){const A=loadedSkills.find(N=>N.uuid===x.last_skill_uuid);A&&(threadSkillMap[currentThreadId]=A,applyActiveSkill(
A))}const C=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),T=i;if(T&&
allMessages.some(A=>String(A.id)===String(T))?currentLeafId=Number(T):C&&allMessages.find(A=>String(
A.id)===String(C))?currentLeafId=C:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(o?{silent:o,keepScroll:o}:{silent:o,keepScroll:o,animate:!0}),
a||restoreLatestPromptOptions(),o&&r?applyCodeCollapseByMessage(get("chat-container"),r,!0):o||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!o&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const A=get("prompt-input");A&&(A.value=c||"",
d?A.style.height=d:A.style.height="auto"),currentImageUrls=m,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i&&new URLSearchParams(
location.search).has("message")&&history.replaceState({},"",location.pathname),a||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(y){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",y),o||showChatLoadError(e),o||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}l(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const o=await(await apiFetch(i.toString())).json(),r=Array.isArray(o.messages)?o.messages:[];
if(r.length){const c=new Set(allMessages.map(m=>m.id)),d=r.filter(m=>!c.has(m.id));d.length&&(allMessages=
d.concat(allMessages))}if(threadHasOlderMessages=!!o.has_older_messages,oldestLoadedMessageId=o.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
l(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;const o=i?Array.from(a.children):[],r=o.map(
k=>k.getBoundingClientRect().height);let c=null;if(i&&(c=a.scrollTop),allMessages.length===0){a.replaceChildren(),
currentParentId=null,updateTotalTokenBar(0);return}const d={};allMessages.forEach(k=>{d[k.id]=k,k.childrenIds=
[]}),allMessages.forEach(k=>{k.parent_id&&d[k.parent_id]&&d[k.parent_id].childrenIds.push(k.id)}),(!currentLeafId||
!d[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null);const m=[];
let g=d[currentLeafId];for(;g;)m.unshift(g),g=d[g.parent_id];const y=buildTokenTotals(m),b=document.
createDocumentFragment();if(threadHasOlderMessages){const k=loadingOlderMessages?"\u8AAD\u307F\u8FBC\u307F\u4E2D...":
"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",C=loadingOlderMessages?"d\
isabled":"",T=document.createElement("div");T.className="mb-3 text-center",T.innerHTML=`<button id="\
load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-gray-600 text-gray-200 hov\
er:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="loadOlderMessages()" ${C}><\
i class="fas fa-clock-rotate-left mr-1"></i>${k}</button>`,b.appendChild(T)}m.forEach(k=>{const C=k.
parent_id?d[k.parent_id]:null,T=C?C.childrenIds:allMessages.filter(N=>!N.parent_id).map(N=>N.id),A=T.
length>1?{current:T.indexOf(k.id)+1,total:T.length,siblings:T}:null;renderMessage(k.id,k.role,k.content,
k.image_url,k.thought_data,k.model,A,n,k.quote_text,k.tokens,k.tokens_in,k.tokens_out,k.is_encrypted,
k.tokens_content,k.tokens_thought,b,!1,k.parent_id,k.skill_name,k.batch_job,k.prompt_options&&typeof k.
prompt_options=="object"?k.prompt_options.image_mask:null,k.processing_time,k.math_steps_state)});const w=currentThreadPending;
if(w&&!isPendingJobSuppressed(w.job_id)){const k=w.message_id,C=new Set(m.map(N=>N.id)),T=m.length?m[m.
length-1]:null;if(k&&C.has(k)&&currentLeafId===k||!k&&T&&T.role==="user"){const N=w.job_id?`pending-${w.
job_id}`:null;renderPendingMessage(b,n,!1,N,w.model||null)}}const x=[];if(i&&!lowBandwidthMode&&Array.
from(b.children).forEach((k,C)=>{const T=o.find(N=>N.id&&N.id===k.id)||(o.length===b.children.length?
o[C]:null),A=T?r[o.indexOf(T)]:0;A>0&&(k.style.minHeight=`${A}px`,x.push(k))}),a.replaceChildren(b),
displayMissingMessageFiles(a,m),updateTotalTokenBar(y.tokens_total,y),currentParentId=currentLeafId,
i&&c!==null?restoreThreadTreeScroll(a,c):scrollToBottom(),lowBandwidthMode)queueMessageDecorations(a,
a&&a.textContent||"");else if(queueHighlight(a),m.length){const k=queueMathTypeset(a,m.map(C=>C.content||
"").join(`
`));Promise.resolve(k).finally(()=>{if(!x.length||!x[0].isConnected)return;const C=a.scrollTop;x.forEach(
T=>{T.style.minHeight=""}),restoreThreadTreeScroll(a,C)})}}l(renderThreadTree,"renderThreadTree");function restoreThreadTreeScroll(e,t){
if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&!chatManualPauseIntent?e.scrollTop=
e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=e.scrollTop,syncScrollToBottomButton()}
l(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){currentLeafId=e;const t={};
allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(i=>{i.parent_id&&t[i.parent_id]&&
t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null,renderThreadTree({animate:!0}),restoreLatestPromptOptions();return}for(;t[n]&&t[n].
childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree({animate:!0}),
restoreLatestPromptOptions()}l(switchVersion,"switchVersion");async function loadSkills(){try{const t=await(await apiFetch(
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
'${i.uuid}')"><i class="fas fa-trash text-[10px]"></i></button></div>`,a.onclick=o=>{o.target.closest(
"button")||activateSkill(i)},n.appendChild(a)})}catch(e){console.error("Failed to load skills:",e)}}
l(loadSkills,"loadSkills");async function openEditSkillModal(e,t){e.stopPropagation(),editingSkillUuid=
t;try{const i=await(await apiFetch(`/api/skills/${t}`)).json();get("skill-name").value=i.name,get("s\
kill-desc").value=i.description||"",get("skill-inst").value=i.instruction,get("skill-modal-title").textContent=
"Edit Skill",get("save-skill-btn").innerText="Save Changes",showModal("skill-modal"),location.pathname!==
"/skill"&&history.pushState({modal:"skill"},"","/skill")}catch{showToast("Skill\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}l(openEditSkillModal,"openEditSkillModal");async function createSkill(e,t){await apiFetch(
CHAT_CONFIG.urls.handleSkills,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{name:e,instruction:t})}),loadSkills()}l(createSkill,"createSkill");function applyActiveSkill(e){activeSkill=
e||null,activeSkill?(get("active-skill-name").innerText=activeSkill.name,get("skill-active-indicator").
classList.remove("hidden")):get("skill-active-indicator").classList.add("hidden");const t=get("sys-p\
rompt-option");t&&(t.style.opacity="1")}l(applyActiveSkill,"applyActiveSkill");function syncActiveSkillForThread(e){
const t=e&&threadSkillMap[e]?threadSkillMap[e]:null;applyActiveSkill(t)}l(syncActiveSkillForThread,"\
syncActiveSkillForThread");async function saveThreadSkillUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_skill_uuid:t,
thread_id:e})})}catch{}}l(saveThreadSkillUuid,"saveThreadSkillUuid");function activateSkill(e,t){currentThreadId?
(threadSkillMap[currentThreadId]=e,applyActiveSkill(e),showToast(`Skill "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadSkillUuid(currentThreadId,e?e.uuid:null)):(pendingSkillForNewThread=e,applyActiveSkill(
e),allMessages&&allMessages.length>0&&startNewChat({preserveSkill:!0}))}l(activateSkill,"activateSki\
ll");function clearActiveSkill(){currentThreadId&&(delete threadSkillMap[currentThreadId],saveThreadSkillUuid(
currentThreadId,null)),pendingSkillForNewThread=null,applyActiveSkill(null)}l(clearActiveSkill,"clea\
rActiveSkill");function getCurrentChatHeaderTitleText(){return typeof currentThreadTitle=="string"&&
currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?"No Title":"AI Chat"}l(getCurrentChatHeaderTitleText,
"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){return temporaryChatEnabled?
`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:""}l(getTemporaryChatTimeoutLabel,
"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){const e=getCurrentChatHeaderTitleText(),
t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=["sidebar-chat-title","mobile-chat-title"],
a=["sidebar-chat-temporary-label","mobile-chat-temporary-label"],o=["sidebar-chat-ttl","mobile-chat-\
ttl"];i.forEach(r=>{const c=get(r);c&&(c.textContent=e)}),a.forEach(r=>{const c=get(r);c&&c.classList.
toggle("hidden",!n)}),o.forEach(r=>{const c=get(r);c&&(n&&t?(c.textContent=t,c.classList.remove("hid\
den")):(c.textContent="",c.classList.add("hidden")))})}l(updateCurrentChatHeaderUi,"updateCurrentCha\
tHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="string"?e:null,updateCurrentChatHeaderUi()}
l(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){tempChatExpiresAtMs=
null,updateCurrentChatHeaderUi()}l(resetTemporaryChatExpiresAt,"resetTemporaryChatExpiresAt");function applyTemporaryChatRuntimeMeta(e){
if(!e||typeof e!="object")return;Object.prototype.hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(
e.timeout_seconds);let t=null;const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.
floor(n*1e3);else{const i=Number(e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+
Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=
null),updateCurrentChatHeaderUi()}l(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
l(ensureCurrentChatHeaderTicker,"ensureCurrentChatHeaderTicker");function normalizeTemporaryChatTimeoutSeconds(e,t=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS){
let n=Number(e);return Number.isFinite(n)||(n=Number(t)),Number.isFinite(n)||(n=TEMP_CHAT_DEFAULT_TIMEOUT_SECONDS),
n=Math.trunc(n),n<TEMP_CHAT_TIMEOUT_MIN_SECONDS&&(n=TEMP_CHAT_TIMEOUT_MIN_SECONDS),n>TEMP_CHAT_TIMEOUT_MAX_SECONDS&&
(n=TEMP_CHAT_TIMEOUT_MAX_SECONDS),n}l(normalizeTemporaryChatTimeoutSeconds,"normalizeTemporaryChatTi\
meoutSeconds");function updateTemporaryChatDescriptionText(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=`\u3053\u306E\u30DA\u30FC\u30B8\u304C\u975E\u8868\u793A/\u5207\u65AD\u306E\u72B6\u614B\u3067 ${e}\
 \u79D2\u7D4C\u904E\u3059\u308B\u3068\u3001\u3053\u306E\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u3068\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3067\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664\u3057\u307E\u3059\uFF08\u30E9\u30A4\u30D6\u30E9\u30EA\u6DFB\u4ED8\u306F\u9664\u5916\uFF09\u3002`,
n=get("temporary-chat-welcome-desc");n&&(n.textContent=t);const i=get("temporary-chat-container");i&&
(i.title=`\u5207\u65AD\u5F8C ${e} \u79D2\u3067\u3001\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u3068\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u6DFB\u4ED8\u3092\u81EA\u52D5\u524A\u9664`)}
l(updateTemporaryChatDescriptionText,"updateTemporaryChatDescriptionText");function applyTemporaryChatTimeoutSeconds(e){
temporaryChatTimeoutSeconds=normalizeTemporaryChatTimeoutSeconds(e,temporaryChatTimeoutSeconds);const t=get(
"set-temp-chat-timeout-seconds");t&&(t.value=String(temporaryChatTimeoutSeconds)),updateTemporaryChatDescriptionText(),
updateCurrentChatHeaderUi(),temporaryChatEnabled&&ensureTemporaryChatHeartbeat(!1)}l(applyTemporaryChatTimeoutSeconds,
"applyTemporaryChatTimeoutSeconds");function getTemporaryChatHeartbeatIntervalMs(){const e=normalizeTemporaryChatTimeoutSeconds(
temporaryChatTimeoutSeconds),t=Math.floor(e*1e3/3);return Math.max(TEMP_CHAT_HEARTBEAT_MIN_MS,Math.min(
TEMP_CHAT_HEARTBEAT_MAX_MS,t))}l(getTemporaryChatHeartbeatIntervalMs,"getTemporaryChatHeartbeatInter\
valMs");function setTemporaryChatUiState(e){temporaryChatEnabled=!!e;const t=get("enable-temporary-c\
hat");t&&t.checked!==temporaryChatEnabled&&(t.checked=temporaryChatEnabled);const n=get("welcome-def\
ault-content");n&&n.classList.toggle("hidden",temporaryChatEnabled);const i=get("welcome-temporary-c\
ontent");i&&i.classList.toggle("hidden",!temporaryChatEnabled),temporaryChatEnabled||(tempChatExpiresAtMs=
null),updateTemporaryChatDescriptionText(),updateCurrentChatHeaderUi()}l(setTemporaryChatUiState,"se\
tTemporaryChatUiState");function stopTemporaryChatHeartbeat(){tempChatHeartbeatTimer&&(clearInterval(
tempChatHeartbeatTimer),tempChatHeartbeatTimer=null),tempChatHeartbeatIntervalMs=0,tempChatHeartbeatInFlight=
!1}l(stopTemporaryChatHeartbeat,"stopTemporaryChatHeartbeat");function canHeartbeatTemporaryChat(){return!!(temporaryChatEnabled&&
currentThreadId&&document.visibilityState==="visible")}l(canHeartbeatTemporaryChat,"canHeartbeatTemp\
oraryChat");async function sendTemporaryChatHeartbeat(e=!1){if(canHeartbeatTemporaryChat()&&!(tempChatHeartbeatInFlight&&
!e)){tempChatHeartbeatInFlight=!0;try{const t=await apiFetch("/api/temporary_chat/heartbeat",{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({thread_id:currentThreadId,active:!0})}),
n=await t.json().catch(()=>({}));t.ok&&n&&applyTemporaryChatRuntimeMeta(n),t.ok&&n&&n.is_temporary===
!1&&(setTemporaryChatUiState(!1),stopTemporaryChatHeartbeat())}catch{}finally{tempChatHeartbeatInFlight=
!1}}}l(sendTemporaryChatHeartbeat,"sendTemporaryChatHeartbeat");function ensureTemporaryChatHeartbeat(e=!1){
if(!temporaryChatEnabled||!currentThreadId){stopTemporaryChatHeartbeat();return}const t=getTemporaryChatHeartbeatIntervalMs();
(!tempChatHeartbeatTimer||tempChatHeartbeatIntervalMs!==t)&&(tempChatHeartbeatTimer&&clearInterval(tempChatHeartbeatTimer),
tempChatHeartbeatIntervalMs=t,tempChatHeartbeatTimer=setInterval(()=>{sendTemporaryChatHeartbeat(!1)},
tempChatHeartbeatIntervalMs)),e&&sendTemporaryChatHeartbeat(!0)}l(ensureTemporaryChatHeartbeat,"ensu\
reTemporaryChatHeartbeat");async function applyTemporaryChatSetting(e){const t=!!e;if(setTemporaryChatUiState(
t),!currentThreadId)return ensureTemporaryChatHeartbeat(!0),!0;try{const n=await apiFetch(`/api/thre\
ads/${currentThreadId}/settings`,{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.
stringify({is_temporary:t})}),i=await n.json().catch(()=>({}));if(!n.ok)throw new Error(i&&i.error||
"\u8A2D\u5B9A\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F");return setTemporaryChatUiState(
!!(i&&i.is_temporary)),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(!0),!0}catch{
return showToast("\u4E00\u6642\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),!1}}l(applyTemporaryChatSetting,"applyTemporaryChatSetting");function startNewChat(e={}){
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
applyActiveSkill(activeSkill):applyActiveSkill(null),!e.preserveProject&&typeof window.closeProjectView==
"function"?window.closeProjectView({skipHistory:!0,keepWelcome:!0}):e.preserveProject&&typeof window.
setActiveProjectContext=="function"&&window.setActiveProjectContext(e.projectId||(typeof window.getActiveProjectId==
"function"?window.getActiveProjectId():null),e.projectName||""),loadThreads();const t=threadLoadSequence;
fetchSettingsSnapshot().then(n=>{if(threadLoadSequence!==t||currentThreadId||modelChosenManually||e.
preserveSkill&&activeSkill)return;const i=n.use_last_chat_settings&&n.last_model||n.default_model;i&&
MODELS.some(a=>a.items.some(o=>o.id===i))&&selectModelById(i)}).catch(n=>console.warn("Failed to loa\
d chat defaults",n)),window.innerWidth<768&&get("overlay").click()}l(startNewChat,"startNewChat");let threadModalLoadSeq=0;
window.openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.
handleThreads,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled,
project_id:typeof window.getActiveProjectId=="function"?window.getActiveProjectId():null})})).json();
currentThreadId=i.id!==null&&i.id!==void 0?String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.is_temporary)),
setCurrentChatHeaderTitle(i&&i.title),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}const e=++threadModalLoadSeq,t=String(currentThreadId);modalThreadId=t,showModal(
"thread-modal"),location.pathname!=="/chat-settings"&&history.pushState({modal:"thread"},"","/chat-s\
ettings");try{const[n,i]=await Promise.all([apiFetch(CHAT_CONFIG.urls.handleSettingsQuery),apiFetch(
`/api/threads/${t}/settings`)]);if(e!==threadModalLoadSeq||modalThreadId!==t)return;if(n.ok){const a=await n.
json(),o=get("thread-app-global-sys-prompt-preview");o&&(o.value=a.global_system_prompt_effective||"");
const r=get("thread-app-global-sys-prompt-preview-status");r&&(a.global_system_prompt_enabled===!1?r.
textContent="\u73FE\u5728\u306F\u7121\u52B9\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002":a.global_system_prompt_uses_time_fallback?
r.textContent="\u7BA1\u7406\u8005\u8A2D\u5B9A\u304C\u7A7A\u6B04\u306E\u305F\u3081\u3001\u6642\u523B\u306E\u65E2\u5B9A\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002":
r.textContent="\u7BA1\u7406\u8005\u304C\u8A2D\u5B9A\u3057\u305F\u5168\u4F53\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8\u304C\u9069\u7528\u3055\u308C\u3066\u3044\u307E\u3059\u3002"),
get("thread-global-sys-prompt")&&(get("thread-global-sys-prompt").value=a.system_prompt||""),get("th\
read-global-sys-prompt-enabled")&&(get("thread-global-sys-prompt-enabled").checked=a.system_prompt_enabled!==
!1),get("thread-apply-global-sys-prompt")&&(get("thread-apply-global-sys-prompt").checked=a.apply_global_system_prompt!==
!1),window.ensureThreadAutoSystemPromptCard(),get("thread-apply-auto-sys-prompt-notices")&&(get("thr\
ead-apply-auto-sys-prompt-notices").checked=a.apply_auto_system_prompt_notices!==!1),window.applyAutoSystemPromptConfigToForm(
"thread",a.auto_system_prompt_notices_config||{})}if(i.ok){const a=await i.json();if(e!==threadModalLoadSeq||
modalThreadId!==t)return;const o=get("thread-custom-instruction");o&&(o.value=a.custom_instruction||
"");const r=get("thread-include-global-instruction");r&&(r.checked=a.include_global_instruction!==!1),
typeof window.syncThreadProjectSettingsModal=="function"&&window.syncThreadProjectSettingsModal(a)}}catch{
showToast("\u30C1\u30E3\u30C3\u30C8\u8A2D\u5B9A\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}},window.closeThreadModal=(e=!1)=>{hideModal("thread-modal"),!e&&location.pathname==="/c\
hat-settings"&&history.back()},get("save-thread-settings-btn").onclick=async()=>{const e=modalThreadId;
if(sendClientDebugLog("info","Save clicked for thread: "+e),!e)return;const t=get("save-thread-setti\
ngs-btn"),n=t?t.textContent:"";t&&(t.disabled=!0,t.textContent="\u4FDD\u5B58\u4E2D...");const i=get(
"thread-custom-instruction"),a=i?i.value:"",o=get("thread-include-global-instruction"),r=o?o.checked:
!0,c=get("thread-include-project-instruction"),d=c?c.checked:!0,m=get("thread-project-select"),g={custom_instruction:a,
include_global_instruction:r,include_project_instruction:d};m&&(g.project_id=m.value||null);const y=get(
"thread-global-sys-prompt"),b=get("thread-global-sys-prompt-enabled");let w=null;try{w=y||b?{system_prompt:y?
y.value:"",system_prompt_enabled:b?b.checked:!0,apply_global_system_prompt:get("thread-apply-global-\
sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get("\
thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(x){sendClientDebugLog("error","Payload c\
onstruction failed: "+x.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const x=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify(g)});let k=null;if(x.ok)try{k=await x.
clone().json()}catch{}sendClientDebugLog("info","PUT request finished, status: "+x.status);let C=!0;
if(w){sendClientDebugLog("info","Starting POST request for user settings");const T=await apiFetch(CHAT_CONFIG.
urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(w)});
C=T.ok,T.ok&&window.applySavedUserSystemPromptSettings(w),sendClientDebugLog("info","POST request fi\
nished, status: "+T.status)}x.ok&&C?(k&&String(currentThreadId||"")===String(e)&&typeof window.setActiveProjectContext==
"function"&&window.setActiveProjectContext(k.project_id||null,k.project_name||""),typeof window.loadProjects==
"function"&&window.loadProjects(),loadThreads(),window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\u308C\u307E\u3057\u305F",
"success")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}catch(x){
sendClientDebugLog("error","Save failed with error: "+x.message),showToast("\u30A8\u30E9\u30FC: "+x.
message,"error",!0)}finally{t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=
()=>{syncCompressionSettingsUi(),showModal("compression-modal"),location.pathname!=="/compression"&&
history.pushState({modal:"compression"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal(
"compression-modal"),!e&&location.pathname==="/compression"&&history.back()},get("save-compression-s\
ettings-btn").onclick=()=>{const e=get("compression-max-size").value,t=get("compression-max-dim").value,
n=get("compression-output-type").value,i=get("compression-format-only").checked;setCompressionSettings(
e,t,n,i);const a=l((r,c)=>{get(r)&&get(c)&&(get(c).value=get(r).value)},"syncBack");a("modal-gpt-ima\
ge-size","gpt-image-size"),a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-compl\
exity","gpt-image-complexity"),a("modal-gpt-image-format","gpt-image-format"),a("modal-gpt-image-com\
pression","gpt-image-compression"),a("modal-gpt-image-frame-count","gpt-image-frame-count"),a("modal\
-gpt-image-frame-duration","gpt-image-frame-duration"),a("modal-gemini-image-aspect","gemini-image-a\
spect"),a("modal-gemini-image-size","gemini-image-size"),a("modal-grok-image-aspect","grok-image-asp\
ect"),a("modal-grok-image-resolution","grok-image-resolution"),a("modal-grok-image-quality","grok-im\
age-quality"),IDEOGRAM_IMAGE_FIELDS.forEach(r=>a(`modal-ideogram-image-${r}`,`ideogram-image-${r}`)),
a("modal-ocr-table-format","ocr-table-format"),a("modal-ocr-pages","ocr-pages");const o=l((r,c)=>{get(
r)&&get(c)&&(get(c).checked=get(r).checked)},"syncBackChk");o("modal-ocr-extract-header","ocr-extrac\
t-header"),o("modal-ocr-extract-footer","ocr-extract-footer"),o("modal-ocr-include-blocks","ocr-incl\
ude-blocks"),o("modal-ocr-include-images","ocr-include-images"),window.closeCompressionModal(),showToast(
"\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F","success")};async function deleteSkill(e,t){
e.stopPropagation(),confirm("Delete?")&&(await apiFetch(CHAT_CONFIG.urls.handleSkillItem.replace("0",
t),{method:"DELETE"}),loadSkills())}l(deleteSkill,"deleteSkill");function renameThread(e,t){var y;e.
stopPropagation();const n=get("thread-rename-modal"),i=get("thread-rename-form"),a=get("thread-renam\
e-input"),o=get("thread-rename-save"),r=get("thread-rename-cancel"),c=get("thread-rename-error"),d=e.
currentTarget;a.value=((y=d.closest("[data-thread-id]"))==null?void 0:y.dataset.threadTitle)||"",c.textContent=
"",o.disabled=!1;let m=!1;const g=l(()=>{m||(hideModal("thread-rename-modal"),d.focus())},"close");r.
onclick=g,n.onclick=b=>{b.target===n&&g()},n.onkeydown=b=>{b.key==="Escape"&&(b.preventDefault(),b.stopPropagation(),
g()),b.key==="Tab"&&(b.shiftKey&&document.activeElement===a?(b.preventDefault(),r.focus()):!b.shiftKey&&
document.activeElement===r&&(b.preventDefault(),a.focus()))},i.onsubmit=async b=>{b.preventDefault();
const w=a.value.trim();if(m||!w){w||a.focus();return}m=!0,o.disabled=!0,r.disabled=!0;try{const x=await apiFetch(
CHAT_CONFIG.urls.updateTitle.replace("0",t),{method:"PUT",headers:{"Content-Type":"application/json"},
body:JSON.stringify({title:w})}),k=await x.json();if(!x.ok)throw new Error("rename failed");currentThreadId===
String(t)&&setCurrentChatHeaderTitle(k.title||w),m=!1,g(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\
\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
m=!1,o.disabled=!1,r.disabled=!1}},r.disabled=!1,showModal("thread-rename-modal"),a.focus()}l(renameThread,
"renameThread");async function deleteThread(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(
CHAT_CONFIG.urls.handleThreadItem.replace("0",t),{method:"DELETE"}),currentThreadId===t?startNewChat():
loadThreads())}l(deleteThread,"deleteThread");async function deleteMessage(e,t){!t&&!confirm("Delete\
 this message and subsequent history?")||(await apiFetch(CHAT_CONFIG.urls.deleteMessage.replace("0",
e),{method:"DELETE"}),loadMessages(currentThreadId))}l(deleteMessage,"deleteMessage");let activePdfPrintFrame=null;
const PDF_IMAGE_EXTS=new Set(["jpg","jpeg","png","webp","gif","bmp","avif","svg"]),PDF_PRINT_ROUTE=CHAT_CONFIG.
urls.exportThreadPdf,pdfEscapeAttr=l(e=>escapeHtml(e==null?"":String(e)),"pdfEscapeAttr"),pdfFormatTimestamp=l(
e=>{if(!e)return"";try{const t=new Date(e);return Number.isNaN(t.getTime())?String(e):new Intl.DateTimeFormat(
"ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digi\
t"}).format(t)}catch{return String(e)}},"pdfFormatTimestamp"),pdfNormalizeAttachmentPath=l(e=>{if(!e)
return"";let t=String(e).trim();if(!t)return"";try{t.includes("://")&&(t=new URL(t,window.location.origin).
pathname||"")}catch{}t.includes("?")&&(t=t.split("?",1)[0]),t.includes("#")&&(t=t.split("#",1)[0]),t=
t.replace(/^\/+/,""),t.startsWith("files/")&&(t=t.slice(6));try{t=decodeURIComponent(t)}catch{}return t},
"pdfNormalizeAttachmentPath"),buildPdfAttachmentUrl=l(e=>{const t=pdfNormalizeAttachmentPath(e);return t?
`${window.location.origin}/files/${encodeURI(t)}`:""},"buildPdfAttachmentUrl"),buildPdfAttachmentPreviewUrl=l(
e=>{const t=pdfNormalizeAttachmentPath(e);return t?`${window.location.origin}/${PDF_IMAGE_EXTS.has((t.
split(".").pop()||"").toLowerCase())?"files/thumb/":"files/"}${encodeURI(t)}`:""},"buildPdfAttachmen\
tPreviewUrl"),buildPdfMessageAttachments=l(e=>(Array.isArray(e&&e.attachments)?e.attachments:[]).map(
n=>{const i=pdfNormalizeAttachmentPath(n&&n.path?n.path:n);if(!i)return null;const a=n&&n.filename?n.
filename:i.split("/").pop(),o=n&&n.source?String(n.source):"attachment",r=!!(n&&n.is_image),c=n&&n.url?
n.url:buildPdfAttachmentUrl(i),d=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:o,isImage:r,url:c,previewUrl:d}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=l(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(d=>maybeNeedsMathJax(d.content)||maybeNeedsMathJax(d.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",o=t.title||"AI Chat",r=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(d=>{const m=d.
role==="user",g=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",y=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",b=!m&&typeof window.
buildMathStepsPdfHtml=="function"?window.buildMathStepsPdfHtml(d.content||"",d.math_steps_state):"",
w=m?`<div class="content" style="white-space: pre-wrap;">${escapeHtml(d.content||"")}</div>`:`<div c\
lass="content">${b||sanitizeMarkdownHtml(d.content||"")}</div>`,x=buildPdfMessageAttachments(d),k=x.
length?`<div class="attachments">${x.map(A=>A.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(
A.previewUrl)}" alt="${pdfEscapeAttr(A.filename)}"><div class="file-caption">${pdfEscapeAttr(A.filename)}\
</div></div>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(A.url)}" target="_blank\
" rel="noreferrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
A.filename)}</span><span class="file-source">${pdfEscapeAttr(A.source)}</span></span></a></div>`).join(
"")}</div>`:"",C=[];d.model&&!m&&C.push(d.model),d.tokens!==null&&d.tokens!==void 0&&C.push(`tokens:${d.
tokens}`),d.tokens_in!==null&&d.tokens_in!==void 0&&C.push(`in:${d.tokens_in}`),d.tokens_out!==null&&
d.tokens_out!==void 0&&C.push(`out:${d.tokens_out}`),d.tokens_thought!==null&&d.tokens_thought!==void 0&&
C.push(`thought:${d.tokens_thought}`),d.is_encrypted&&C.push("encrypted"),d.parent_id!==null&&d.parent_id!==
void 0&&C.push(`parent:#${d.parent_id}`);const T=C.length?`<div class="message-meta">${pdfEscapeAttr(
C.join(" \u2022 "))}</div>`:"";return`
                    <article class="message ${m?"user":"ai"}">
                        <div class="message-head">
                            <div class="message-role" style="color:${m?"var(--user)":"var(--ai)"}"><\
span class="dot"></span><span>${m?"User":"Assistant"}</span></div>
                            <div class="message-time">${pdfEscapeAttr(pdfFormatTimestamp(d.timestamp))}\
</div>
                        </div>
                        <div class="message-body">
                            ${g}
                            ${w}
                            ${y}
                            ${k}
                            ${T}
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
        .math-steps-pdf-step { margin: 0 0 18px; }
        .math-steps-pdf-heading { margin: 0 0 8px; font-size: 15px; font-weight: 800; color: var(--a\
ccent); }
        .math-steps-pdf-note { color: var(--muted); font-size: 11px; margin-top: 4px; }
        .mathsvg-render { display:block; text-align:center; margin: 10px 0; }
        .mathsvg-render img { max-width:100%; height:auto; border-radius:12px; border:1px solid rgba\
(148,163,184,0.28); }
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
        ${r.map(d=>`<div class="meta-card"><div class="meta-label">${pdfEscapeAttr(d.label)}</div><d\
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
"error",!0);return}const o=document.createElement("iframe");activePdfPrintFrame=o,o.setAttribute("ar\
ia-hidden","true"),o.style.position="fixed",o.style.right="0",o.style.bottom="0",o.style.width="1px",
o.style.height="1px",o.style.opacity="0",o.style.pointerEvents="none",o.style.border="0";let r=null;
const c=l(()=>{r&&(clearTimeout(r),r=null),t&&t.remove(),(activePdfPrintFrame===o||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{o.parentNode&&o.parentNode.removeChild(o)}catch{}},"cleanup");r=setTimeout(
()=>{activePdfPrintFrame===o&&(console.log("PDF print cleanup fallback triggered"),c())},6e4),o.onload=
async()=>{try{const g=o.contentDocument,y=o.contentWindow;if(!g||!y){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(T=>maybeNeedsMathJax(T.content)||maybeNeedsMathJax(T.thought_text))&&(y.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),g.fonts&&g.fonts.ready)try{await g.fonts.ready}catch{}t.update(60);const x=Array.from(g.
images||[]),k=Promise.all(x.map(T=>T.complete?Promise.resolve():new Promise(A=>{T.addEventListener("\
load",A,{once:!0}),T.addEventListener("error",A,{once:!0})})));if(await Promise.race([k,new Promise(
T=>setTimeout(T,5e3))]),t.update(80),g.getElementById("MathJax-script")){let T=0;for(;T<100&&(!y.MathJax||
typeof y.MathJax.typesetPromise!="function");)await new Promise(A=>setTimeout(A,50)),T++;if(y.MathJax&&
typeof y.MathJax.typesetPromise=="function")try{await y.MathJax.typesetPromise()}catch(A){console.error(
"PDF MathJax typeset failed",A)}}t.update(95),setTimeout(()=>{try{y.focus(),y.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),y.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(a),m=new Blob([d],{type:"text/html"});o.src=URL.createObjectURL(
m),document.body.appendChild(o)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}l(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}l(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}l(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=l((d,m)=>i.
compare(d.filename||"",m.filename||""),"nameAsc"),o=l((d,m)=>i.compare(m.filename||"",d.filename||""),
"nameDesc"),r=l((d,m)=>(Number(m.ts)||0)-(Number(d.ts)||0),"tsDesc"),c=l((d,m)=>(Number(d.ts)||0)-(Number(
m.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,m)=>a(d,m)||r(d,m)):t==="name_desc"?n.sort((d,m)=>o(
d,m)||r(d,m)):t==="oldest"?n.sort((d,m)=>c(d,m)||a(d,m)):n.sort((d,m)=>r(d,m)||a(d,m)),n}l(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}l(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}l(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}l(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}l(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const m=t.querySelector(".lib-empty-state");m&&m.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const m=get("lib-total-count");m&&(m.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),o=i.filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(
a)),r=get("lib-total-count");if(r){const m=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?r.innerText=`${lib.files.length} / ${m} files`:r.innerText=`${m} files`}if(!o.length){
if(n)return;const m=lib.favoritesOnly&&!a?"fa-star":"fa-search",g=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",y=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${m}"></i></div>\
<p class="lib-empty-title">${g}</p><p class="lib-empty-sub">${y}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(a)):o).forEach(m=>{
try{const g=renderLibraryItem(m,c++);t.appendChild(g)}catch{}})}l(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(g=>fileNameForSearch(g).includes(n)):t).filter(g=>g.type==="image"),o=lib.favoritesOnly?
a.filter(g=>g.is_favorite):a;if(!o.length)return;const r=new Map,c=get("lib-grid");c&&c.querySelectorAll(
".library-thumb-card").forEach(g=>{const y=g.dataset&&g.dataset.filepath,b=g.querySelector(".library\
-thumb-media");y&&b&&r.set(y,b)});const d=o.map(g=>({url:g.url,filename:g.filename||g.original_filename||
g.url.split("/").pop(),element:r.get(g.filepath)||null}));let m=d.findIndex(g=>g.url===e.url);m===-1&&
(m=0),openViewerWithItems(d,m)}l(openLibraryImage,"openLibraryImage");function libraryFileIcon(e){const t={
pdf:"fa-file-pdf",image:"fa-image",file:"fa-file"},n=String(e||"").toLowerCase();return n==="pdf"?t.
pdf:["png","jpg","jpeg","gif","webp","bmp","svg","heic"].includes(n)?t.image:t.file}l(libraryFileIcon,
"libraryFileIcon");function renderLibraryItem(e,t=0){const n=document.createElement("div");n.className=
"library-thumb-card",t!=null&&(n.style.animationDelay=`${Math.min(t*.035,.45)}s`);const i=e.thumbnail_url||
e.thumb_url||e.url,a=String(e.ext||(e.filename||"").split(".").pop()||"").toLowerCase(),o=e.type==="\
image"?`<img src="${escapeHtml(i)}" alt="${escapeHtml(e.filename)}" loading="lazy" decoding="async" \
class="library-thumb-media">`:`<div class="library-thumb-file"><div class="lib-file-icon"><i class="\
fas ${libraryFileIcon(a)}"></i></div><span class="lib-file-badge">${escapeHtml(a?a.toUpperCase():"FI\
LE")}</span></div>`,r=`<div class="lib-overlay"><a href="${escapeHtml(e.url)}" download="${escapeHtml(
e.filename)}" class="lib-overlay-btn" onclick="event.stopPropagation()" title="\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas\
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",d=e.is_favorite?"fas fa-star":"far f\
a-star",m=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
g=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${m}\
" aria-label="${m}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
y=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${o}</div>${r}${g}${y}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",k=>{k.shiftKey&&k.preventDefault()}),n.onclick=
k=>{if(k&&k.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const C=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),T=C.findIndex(N=>N.dataset.filepath===lib.anchorPath),
A=C.indexOf(n);if(T!==-1&&A!==-1){const N=Math.min(T,A),G=Math.max(T,A);for(let L=N;L<=G;L++){const $=C[L].
dataset.filepath;$&&(lib.selected.add($),C[L].classList.add("is-selected"))}try{const L=window.getSelection&&
window.getSelection();L&&L.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(k=>{k.onclick=C=>{C.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async k=>{k.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const x=n.querySelector(".lib-favorite-btn");return x&&
(x.onclick=async k=>{k.stopPropagation(),x.disabled=!0;try{const C=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepath:e.filepath})}),
T=await C.json().catch(()=>({}));if(!C.ok||typeof T.is_favorite!="boolean")throw new Error(T.error||
"favorite update failed");e.is_favorite=T.is_favorite,renderLibraryGrid(),showToast(T.is_favorite?"\u304A\
\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3057\u307E\u3057\u305F":"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3057\u307E\u3057\u305F",
"success")}catch{showToast("\u304A\u6C17\u306B\u5165\u308A\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),x.disabled=!1}}),n}l(renderLibraryItem,"renderLibraryItem");function renderLibrarySkeleton(e){
if(e){e.innerHTML="";for(let t=0;t<12;t++){const n=document.createElement("div");n.className="lib-sk\
eleton-card",n.style.animationDelay=`${Math.min(t*.04,.5)}s`,n.innerHTML='<div class="lib-skeleton-t\
humb"></div><div class="lib-skeleton-bar"><span class="lib-skeleton-line" style="width:78%"></span><\
span class="lib-skeleton-line" style="width:45%"></span></div>',e.appendChild(n)}}}l(renderLibrarySkeleton,
"renderLibrarySkeleton");function addLibraryFileFromPath(e){if(!e||(lib.fileSet||(lib.fileSet=new Set),
lib.fileSet.has(e)))return;const t=e.split("/").pop()||e,n=(t.split(".").pop()||"").toLowerCase(),i=[
"png","jpg","jpeg","webp","gif"].includes(n)?"image":"file",a=FILE_BASE_URL+e,o=i==="image"?FILE_THUMB_BASE_URL+
e:null,r={filename:t,original_filename:t,filepath:e,url:a,thumbnail_url:o,type:i,ext:n,ts:Math.floor(
Date.now()/1e3)};setAttachmentNameForPath(e,t),lib.fileSet.add(e),lib.files||(lib.files=[]),lib.files.
unshift(r),get("lib-grid")&&lib.modal&&lib.modal.classList.contains("modal-open")&&renderLibraryGrid()}
l(addLibraryFileFromPath,"addLibraryFileFromPath");async function renameSelectedLibraryFile(){if(!lib.
selected||lib.selected.size!==1)return;const e=Array.from(lib.selected)[0],t=(lib.files||[]).find(o=>o.
filepath===e),n=t&&t.filename||e.split("/").pop()||e,i=prompt("\u65B0\u3057\u3044\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
n);if(i===null)return;const a=(i||"").trim();if(!a){showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}try{const o=await apiFetch(CHAT_CONFIG.urls.renameLibraryFile,{method:"POST",headers:{
"Content-Type":"application/json"},body:JSON.stringify({filepath:e,filename:a})}),r=await o.json().catch(
()=>({}));if(!o.ok){showToast(r.error||"\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}t&&(t.filename=r.filename||a,setAttachmentNameForPath(e,t.filename));const c=get(
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(d=>{d.getAttribute("data-filename")===
e&&setRowAttachmentName(d,t?t.filename:r.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
showToast("\u30D5\u30A1\u30A4\u30EB\u540D\u3092\u5909\u66F4\u3057\u307E\u3057\u305F","success")}catch{
showToast("\u540D\u524D\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}l(renameSelectedLibraryFile,
"renameSelectedLibraryFile");async function deleteSingleLibraryFile(e,t){if(e&&confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))
try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,{method:"POST",headers:{"Content-Type":"applica\
tion/json"},body:JSON.stringify({filenames:[e]})}),t&&t.parentNode&&t.remove(),lib.files&&(lib.files=
lib.files.filter(n=>n.filepath!==e)),lib.fileSet&&lib.fileSet.delete(e),lib.selected.delete(e),renderLibraryGrid(),
window.updateLibSelectionUi()}catch{showToast("\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}l(deleteSingleLibraryFile,"deleteSingleLibraryFile");function closeFileUsageModal(){hideModal(
"lib-usage-modal")}l(closeFileUsageModal,"closeFileUsageModal"),window.closeFileUsageModal=closeFileUsageModal;
async function showSelectedFileUsage(){if(!lib.selected||lib.selected.size!==1)return;const e=Array.
from(lib.selected)[0],t=(lib.files||[]).find(a=>a&&a.filepath===e),n=get("lib-usage-title"),i=get("l\
ib-usage-list");if(i){n&&(n.textContent=t&&(t.filename||t.original_filename)||e.split("/").pop()||e),
i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-spinner fa-spin mr\
-2"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026</div>',showModal("lib-usage-modal");try{const a=new URL(
CHAT_CONFIG.urls.getFileUsageChats,window.location.origin);a.searchParams.set("filepath",e);const o=await apiFetch(
a.toString(),{cache:"no-store",headers:{Accept:"application/json"}}),r=await o.json().catch(()=>({}));
if(!o.ok)throw new Error(r.error||`HTTP ${o.status}`);const c=Array.isArray(r.chats)?r.chats:[];if(!c.
length){i.innerHTML='<div class="text-sm text-gray-400 text-center py-8"><i class="fas fa-comment-do\
ts text-xl mb-2 block"></i>\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u4F7F\u7528\u3057\u3066\u3044\u308B\u30C1\u30E3\u30C3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}if(i.innerHTML="",c.forEach(d=>{const m=document.createElement("div");m.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const g=d.updated_at?new Date(d.
updated_at).toLocaleString():"";m.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(g)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const y=m.querySelector("button");y&&(y.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(m)}),r.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}l(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let o=null,r=!1;try{const d=getLibSortOrder(),m=getLibSearchQuery(),g=e?lib.nextOffset:
0,y=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(g),sort:d,q:m,favorites_only:lib.
favoritesOnly?"1":"0"}),b=await apiFetch(a+"?"+y.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!b.ok)throw new Error("HTTP "+b.status);o=await b.json(),r=!0}catch(d){i=d}if(!r){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(o)?o:
o&&Array.isArray(o.files)?o.files:[];o&&!Array.isArray(o)&&(lib.totalCount=Number(o.total)||0,lib.hasMore=
!!o.has_more,lib.nextOffset=(Number(o.offset)||0)+(Number(o.limit)||c.length));try{const d=FILE_BASE_URL,
m=FILE_THUMB_BASE_URL,g=new Set(c.map(b=>b&&b.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(b=>{if(c.length>=LIBRARY_PAGE_SIZE||!b||g.has(b))return;const w=getAttachmentNameForPath(
b)||b.split("/").pop()||b,x=(w.split(".").pop()||"").toLowerCase(),k=["png","jpg","jpeg","webp","gif"].
includes(x)?"image":"file",C=k==="image"?m+b:null;c.unshift({filename:w,original_filename:w,filepath:b,
url:d+b,thumbnail_url:C,type:k,ext:x,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),g.add(b)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=c.filter(g=>g&&g.filepath&&
g.url);let m=[];if(e){const g=new Set(lib.files.map(y=>y.filepath));m=d.filter(y=>!g.has(y.filepath)),
lib.files.push(...m)}else lib.files=d;lib.files.forEach(g=>{g&&g.filepath&&setAttachmentNameForPath(
g.filepath,g.filename||g.original_filename||"")}),lib.fileSet=new Set(lib.files.map(g=>g.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
m:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002","error",!0):t.innerHTML='<div class="l\
ib-empty-state"><div class="lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p clas\
s="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div\
>'),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore)}l(loadLibraryFiles,"loadLibraryFiles");async function deleteSelectedFiles(){
if(confirm("\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))try{await apiFetch(CHAT_CONFIG.urls.deleteFilesBatch,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filenames:Array.from(
lib.selected)})}),loadLibraryFiles()}catch{alert("\u524A\u9664\u30A8\u30E9\u30FC")}}l(deleteSelectedFiles,
"deleteSelectedFiles");function attachSelectedLibraryFiles(){if(!lib.selected.size)return;const e=getModelMediaSupport(
get("model-select").value);let t=0,n=0;if(Array.from(lib.selected).forEach(a=>{const o=isAudioPath(a),
r=isVideoPath(a);if(o&&!e.audio||r&&!e.video){o&&(t+=1),r&&(n+=1);return}const c=normalizeAttachmentPath(
a);if(!c)return;const d=(lib.files||[]).find(m=>m&&m.filepath===a);d&&d.filename&&setAttachmentNameForPath(
c,d.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
ibrary")}),syncUploadRowsFromCurrent(),updateFilePreview(),lib.selected.clear(),window.updateLibSelectionUi(),
window.closeLibModal(),t||n){const a=[];t&&a.push(`${t}\u4EF6\u306E\u97F3\u58F0`),n&&a.push(`${n}\u4EF6\u306E\u52D5\
\u753B`),showToast(`\u3053\u306E\u30E2\u30C7\u30EB\u306F${a.join("\u30FB")}\u5165\u529B\u306B\u975E\u5BFE\u5FDC\u306E\u305F\u3081\u9664\u5916\u3057\u307E\u3057\u305F`,
"error",!0)}else showToast("\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u6DFB\u4ED8\u3057\u307E\u3057\u305F",
"success")}l(attachSelectedLibraryFiles,"attachSelectedLibraryFiles");function downloadSelectedLibraryFiles(){
if(!lib.selected||!lib.selected.size)return;const e=Array.from(lib.selected);e.forEach(t=>{const n=(lib.
files||[]).find(i=>i&&i.filepath===t);if(n&&n.url){const i=document.createElement("a");i.href=n.url,
i.download=n.filename||n.original_filename||t.split("/").pop()||"file",document.body.appendChild(i),
i.click(),document.body.removeChild(i)}}),showToast(`${e.length}\u4EF6\u306E\u30D5\u30A1\u30A4\u30EB\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3057\u307E\u3057\u305F`,
"success")}l(downloadSelectedLibraryFiles,"downloadSelectedLibraryFiles"),window.showLegal=async e=>{
const t=e==="terms"?"\u5229\u7528\u898F\u7D04":"\u30D7\u30E9\u30A4\u30D0\u30B7\u30FC\u30DD\u30EA\u30B7\u30FC";
get("legal-title").innerText=t,showModal("legal-modal");const n=await apiFetch("/static/legal/"+e+".\
md?t="+Date.now());if(!n.ok)return;const i=await n.text();get("legal-content").innerHTML=sanitizeMarkdownHtml(
i)},window.showAlphaInfo=()=>{if(typeof showModal=="function"){showModal("alpha-info-modal");return}
const e=get("alpha-info-modal");e&&(e.classList.remove("hidden"),e.style.display="flex")},window.copyCode=
(e,t)=>{const n=decodeURIComponent(t),i=l(()=>{const a=e.getAttribute("data-copy")||"";e.innerHTML=a===
"output"?'<i class="fas fa-align-left"></i>':'<i class="fas fa-copy"></i>'},"restoreIcon");copyToClipboard(
n,()=>{e.innerHTML='<i class="fas fa-check"></i>',setTimeout(i,2e3)},a=>{console.error(a),e.innerHTML=
'<i class="fas fa-times"></i>',setTimeout(i,2e3)})},window.copyMessage=(e,t)=>{const n=messageStore[e]||
"",i=typeof window.prepareMathStepsCopyText=="function"?window.prepareMathStepsCopyText(n):n;copyToClipboard(
i,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i class="fas fa-copy"\
></i>',2e3)},a=>{console.error(a),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(()=>t.innerHTML=
'<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;t.classList.
contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
branchLabelNames={},threadFixedBranchId=null;function loadBranchData(){if(!currentThreadId)return;const e=localStorage.
getItem(`branch_names_${currentThreadId}`);branchLabelNames=e?JSON.parse(e):{},threadFixedBranchId=localStorage.
getItem(`fixed_branch_${currentThreadId}`)}l(loadBranchData,"loadBranchData");function saveBranchData(){
currentThreadId&&(localStorage.setItem(`branch_names_${currentThreadId}`,JSON.stringify(branchLabelNames)),
threadFixedBranchId?localStorage.setItem(`fixed_branch_${currentThreadId}`,threadFixedBranchId):localStorage.
removeItem(`fixed_branch_${currentThreadId}`))}l(saveBranchData,"saveBranchData");function getCumulativeTokensForNode(e){
let t=0,n=e;const i={};for((allMessages||[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n];t+=a.tokens||
Number(a.tokens_in||0)+Number(a.tokens_out||0),n=a.parent_id}return t}l(getCumulativeTokensForNode,"\
getCumulativeTokensForNode");function getPerModelTokensForPath(e){const t={};let n=e;const i={};for((allMessages||
[]).forEach(a=>i[a.id]=a);n&&i[n];){const a=i[n],o=a.model||"Unknown";t[o]||(t[o]={total:0,in:0,out:0,
thought:0});const r=a.tokens||Number(a.tokens_in||0)+Number(a.tokens_out||0);t[o].total+=r,t[o].in+=
Number(a.tokens_in||0),t[o].out+=Number(a.tokens_out||0),t[o].thought+=Number(a.tokens_thought||0),n=
a.parent_id}return t}l(getPerModelTokensForPath,"getPerModelTokensForPath"),window.showBranchModal=()=>{
if(!currentThreadId){showToast("\u30C1\u30E3\u30C3\u30C8\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}loadBranchData(),selectedBranchNodeId=null,renderBranchTreeVisualization(),updateBranchDetailPane(),
showModal("branch-modal"),location.pathname!=="/branch"&&history.pushState({modal:"branch"},"","/bra\
nch");const e=buildTokenTotals(allMessages);get("branch-total-tokens").innerText=e.tokens_total||0},
window.closeBranchModal=(e=!1)=>{hideModal("branch-modal"),!e&&location.pathname==="/branch"&&history.
back()};function renderBranchTreeVisualization(){const e=get("branch-tree-canvas");if(e.innerHTML="",
!allMessages||allMessages.length===0)return;const t={},n=[];allMessages.forEach(a=>t[a.id]={...a,children:[]}),
allMessages.forEach(a=>{a.parent_id&&t[a.parent_id]?t[a.parent_id].children.push(t[a.id]):a.parent_id||
n.push(t[a.id])});function i(a){const o=document.createElement("div");o.className="flex flex-col ite\
ms-center mt-4";const r=document.createElement("div"),c=String(a.id)===String(currentLeafId),d=a.id===
threadFixedBranchId,m=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),g=getCumulativeTokensForNode(
a.id);if(r.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,r.innerHTML=`
                    <div class="font-bold truncate">${escapeHtml(m)}</div>
                    <div class="text-[9px] text-gray-500 flex justify-between mt-1 gap-2">
                        <span class="truncate">${escapeHtml(a.model||"-")}</span>
                        <span class="text-blue-400 font-mono font-bold" title="Cumulative tokens for\
 this path">${g}</span>
                    </div>
                    ${d?'<div class="absolute -top-1 -right-1 w-3 h-3 bg-amber-500 rounded-full bord\
er border-gray-900 shadow-sm" title="Fixed Branch"></div>':""}
                    ${c?'<div class="absolute -top-1 -left-1 w-3 h-3 bg-blue-500 rounded-full border\
 border-gray-900 shadow-sm" title="Current Branch"></div>':""}
                `,r.onclick=y=>{y.stopPropagation(),selectedBranchNodeId=a.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},o.appendChild(r),a.children.length>0){const y=document.createElement("div");
y.className="w-px h-4 bg-gray-700",o.appendChild(y);const b=document.createElement("div");b.className=
"flex gap-4 items-start",a.children.forEach(w=>b.appendChild(i(w))),o.appendChild(b)}return o}l(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}l(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}l(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(d=>d.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const o=get(
"branch-model-breakdown"),r=getPerModelTokensForPath(n.id);o.innerHTML="",Object.entries(r).sort((d,m)=>m[1].
total-d[1].total).forEach(([d,m])=>{const g=document.createElement("div");g.className="bg-gray-800/5\
0 p-2 rounded border border-gray-700/50",g.innerHTML=`
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
                `,o.appendChild(g)}),get("br-name-input").value=branchLabelNames[n.id]||"";const c=get(
"br-fix-btn");selectedBranchNodeId===threadFixedBranchId?(c.innerText="\u56FA\u5B9A\u3092\u89E3\u9664",
c.classList.replace("bg-amber-600","bg-gray-600")):(c.innerText="\u30E1\u30A4\u30F3\u30EB\u30FC\u30C8\u306B\u56FA\u5B9A",
c.classList.replace("bg-gray-600","bg-amber-600"))}l(updateBranchDetailPane,"updateBranchDetailPane"),
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
l(batchProviderLabel,"batchProviderLabel");function batchStateLabelShort(e){return{JOB_STATE_QUEUED:"\
\u9001\u4FE1\u5F85\u3061",JOB_STATE_VALIDATING:"\u691C\u8A3C\u4E2D",JOB_STATE_PENDING:"\u5F85\u6A5F\u4E2D",
JOB_STATE_RUNNING:"\u5B9F\u884C\u4E2D",JOB_STATE_FINALIZING:"\u7D50\u679C\u53D6\u5F97\u4E2D",JOB_STATE_SUCCEEDED:"\
\u5B8C\u4E86",JOB_STATE_FAILED:"\u5931\u6557",JOB_STATE_CANCELLING:"\u505C\u6B62\u4E2D",JOB_STATE_CANCELLED:"\
\u505C\u6B62",JOB_STATE_EXPIRED:"\u671F\u9650\u5207\u308C"}[String(e||"").toUpperCase()]||"\u78BA\u8A8D\u4E2D"}
l(batchStateLabelShort,"batchStateLabelShort");function batchStateTone(e){const t=String(e||"").toUpperCase();
return t==="JOB_STATE_SUCCEEDED"?"border-emerald-500/40 bg-emerald-900/20 text-emerald-200":t==="JOB\
_STATE_FAILED"?"border-red-500/40 bg-red-900/20 text-red-200":t==="JOB_STATE_CANCELLED"||t==="JOB_ST\
ATE_EXPIRED"?"border-gray-500/40 bg-gray-700/30 text-gray-300":t==="JOB_STATE_CANCELLING"?"border-am\
ber-500/40 bg-amber-900/20 text-amber-200":"border-violet-500/40 bg-violet-900/20 text-violet-200"}l(
batchStateTone,"batchStateTone");function batchFormatTime(e){if(!e)return"";let t=String(e);!/[zZ]$/.
test(t)&&!/[+-]\d\d:?\d\d$/.test(t)&&(t+="Z");const n=new Date(t);return isNaN(n.getTime())?String(e):
n.toLocaleString("ja-JP",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})}l(batchFormatTime,
"batchFormatTime");function playBatchListAnimation(){const e=get("batch-list");e&&(e.classList.remove(
"batch-list-enter"),e.offsetWidth,e.classList.add("batch-list-enter"))}l(playBatchListAnimation,"pla\
yBatchListAnimation");function renderBatchJobs(e={}){const t=get("batch-list");if(!t)return;const n=batchJobsCache.
filter(a=>batchFilterMode==="active"?!!a.is_active:batchFilterMode==="done"?!a.is_active:!0),i=get("\
batch-count");if(i&&(i.textContent=`${n.length}\u4EF6`),t.innerHTML="",!n.length){t.innerHTML='<div \
class="batch-empty"><i class="fas fa-layer-group"></i><span>Batch\u51E6\u7406\u306E\u5C65\u6B74\u306F\u3042\u308A\u307E\u305B\u3093</span></div>',
e.animate&&playBatchListAnimation();return}n.forEach(a=>{const o=document.createElement("div");o.className=
"batch-job-card";const r=escapeHtml(a.thread_title||"\u7121\u984C\u306E\u30C1\u30E3\u30C3\u30C8"),c=escapeHtml(
batchProviderLabel(a.provider)),d=escapeHtml(a.model||""),m=escapeHtml(batchFormatTime(a.created_at)),
g=escapeHtml(a.status_text||""),y=batchStateTone(a.state),b=a.thread_exists?'<button type="button" d\
ata-batch-open class="batch-action-btn batch-action-open"><i class="fas fa-comment-dots"></i>\u958B\u304F</but\
ton>':"",w=a.can_cancel?'<button type="button" data-batch-cancel class="batch-action-btn batch-actio\
n-cancel"><i class="fas fa-stop"></i>\u505C\u6B62</button>':"",x=a.is_active?"":'<button type="butto\
n" data-batch-delete class="batch-action-btn batch-action-danger"><i class="fas fa-trash"></i>\u5C65\u6B74\u304B\u3089\u524A\u9664\
</button>';o.innerHTML=`
                    <div class="flex items-start justify-between gap-3">
                        <div class="min-w-0">
                            <div class="batch-job-title text-sm font-bold truncate" title="${r}">${r}\
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
                    <div class="batch-job-status mt-2 text-[11px] break-words">${g}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${b}${w}${x}
                    </div>`;const k=o.querySelector("[data-batch-open]");k&&(k.onclick=()=>{window.closeBatchModal(),
loadMessages(a.thread_id)});const C=o.querySelector("[data-batch-cancel]");C&&(C.onclick=()=>cancelBatchJob(
a));const T=o.querySelector("[data-batch-delete]");T&&(T.onclick=()=>deleteBatchJob(a)),t.appendChild(
o)}),e.animate&&playBatchListAnimation()}l(renderBatchJobs,"renderBatchJobs");async function loadBatchJobs(e={}){
try{const t=await apiFetch("/api/batch/jobs");if(!t.ok){e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error");return}const n=await t.json().catch(()=>({}));batchJobsCache=Array.isArray(n.jobs)?n.jobs:[],
renderBatchJobs()}catch{e.silent||showToast("Batch\u51E6\u7406\u306E\u5C65\u6B74\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error")}}l(loadBatchJobs,"loadBatchJobs");async function cancelBatchJob(e){if(!confirm("\u3053\u306EBatch\u51E6\u7406\u3092\u505C\
\u6B62\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(`/api/batch/jobs/${encodeURIComponent(
e.job_id)}/cancel`,{method:"POST"}),n=await t.json().catch(()=>({}));if(!t.ok){showToast(n.error||"B\
atch\u51E6\u7406\u3092\u505C\u6B62\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0);return}
showToast("Batch\u51E6\u7406\u3092\u505C\u6B62\u3057\u307E\u3057\u305F","success"),await loadBatchJobs(
{silent:!0}),String(e.thread_id)===String(currentThreadId)&&await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0})}l(cancelBatchJob,"cancelBatchJob");async function deleteBatchJob(e){if(!confirm("\u3053\u306EBatch\
\u51E6\u7406\u306E\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F"))return;const t=await apiFetch(
`/api/batch/jobs/${encodeURIComponent(e.job_id)}`,{method:"DELETE"}),n=await t.json().catch(()=>({}));
if(!t.ok){showToast(n.error||"Batch\u5C65\u6B74\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0);return}showToast("Batch\u5C65\u6B74\u3092\u524A\u9664\u3057\u307E\u3057\u305F","success"),
await loadBatchJobs({silent:!0})}l(deleteBatchJob,"deleteBatchJob"),window.showBatchModal=()=>{showModal(
"batch-modal"),location.pathname!=="/batch"&&history.pushState({modal:"batch"},"","/batch"),loadBatchJobs(),
batchListTimer&&clearInterval(batchListTimer),batchListTimer=setInterval(()=>{const e=get("batch-mod\
al");!e||e.classList.contains("hidden")||loadBatchJobs({silent:!0})},5e3)},window.closeBatchModal=(e=!1)=>{
hideModal("batch-modal"),batchListTimer&&(clearInterval(batchListTimer),batchListTimer=null),!e&&location.
pathname==="/batch"&&history.back()},get("batch-manage-btn")&&(get("batch-manage-btn").onclick=()=>window.
showBatchModal()),get("batch-refresh-btn")&&(get("batch-refresh-btn").onclick=()=>loadBatchJobs()),document.
querySelectorAll(".batch-filter-tab").forEach(e=>{e.onclick=()=>{batchFilterMode=e.dataset.batchFilter||
"all",document.querySelectorAll(".batch-filter-tab").forEach(t=>{t.classList.toggle("is-active",t===
e)}),renderBatchJobs({animate:!0})}});const showApiKeyRequiredModalAsync=l(e=>new Promise(t=>{const n=getModelNameById(
e),i=getModelProviderInfo(e);get("api-key-modal-model-name").textContent=`${n}\uFF08${e}\uFF09`,get(
"api-key-modal-desc").textContent=`\u3053\u306E\u30E2\u30C7\u30EB\u3092\u4F7F\u7528\u3059\u308B\u306B\u306F${i?
i.label:"API\u30AD\u30FC"}\u306E\u8A2D\u5B9A\u304C\u5FC5\u8981\u3067\u3059\u3002`,get("api-key-modal\
-key-label").textContent=i?i.label:"API Key";const a=i?get(i.inputId):null;get("api-key-modal-input").
value=a?a.value:"",get("api-key-modal-input").placeholder="API\u30AD\u30FC\u3092\u5165\u529B";const o=get(
"api-key-modal-save-btn"),r=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),d=l(
()=>{o.onclick=null,r.onclick=null,c.onclick=null},"cleanup"),m=l(g=>{g.key==="Enter"&&(g.preventDefault(),
o.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",m),o.onclick=async()=>{
const g=get("api-key-modal-input").value.trim();if(!g){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const y=get(i.inputId);y&&(y.value=g);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:g})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=g)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",m),d(),t("set")},r.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",m),d(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",m),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const g=get("api-key-modal-input");g&&g.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function o(r,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const d=c.map(m=>{try{return m instanceof Error?m.stack||m.message:typeof m=="object"?JSON.
stringify(m):String(m)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(r,d)}catch{}finally{
a=!1}}l(o,"sendToServer"),console.log=function(...r){e.apply(console,r),o("log",r)},console.error=function(...r){
t.apply(console,r),o("error",r)},console.warn=function(...r){n.apply(console,r),o("warn",r)},console.
info=function(...r){i.apply(console,r),o("info",r)},window.addEventListener("error",function(r){o("e\
xception",[r.message,r.filename,r.lineno,r.colno,r.error])}),window.addEventListener("unhandledrejec\
tion",function(r){o("promise-rejection",[r.reason])}),setTimeout(()=>{console.log("Extended debug lo\
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
IP\u30FB\u7AEF\u672B\u306EBAN\u89E3\u9664"},r={userId:0,username:"",exists:!1,type:"",items:[],hasMore:!1,
selected:new Set,counts:{},total:0,data:null},c={accounts:[],selected:new Set},d=l(()=>get("bot-log-\
detail"),"view"),m=l(()=>get("bot-log-list-view"),"listView"),g=l(O=>{if(!O)return"-";const B=new Date(
O);return isNaN(B.getTime())?String(O):B.toLocaleString("ja-JP")},"formatTime"),y=l(O=>{const B=Math.
max(0,Math.floor(Number(O)||0));return B>=60?`${Math.floor(B/60)}\u5206${B%60?B%60+"\u79D2":""}`:`${B}\
\u79D2`},"formatDuration"),b=l((O,B)=>B==null||B===""?"-":O==="lock_source"?a[B]||String(B):O==="cli\
ent"||O==="origin_client"?n[B]||String(B):O==="action"?o[B]||String(B):O==="by"?B==="self"?"\u672C\u4EBA":
B==="admin"?"\u7BA1\u7406\u8005":String(B):O==="enabled"?B?"ON":"OFF":O==="remaining_seconds"||O==="\
lock_seconds"?y(B):O==="applied_to"&&Array.isArray(B)?B.map(J=>a[J]||J).join("\u30FB"):typeof B=="ob\
ject"?JSON.stringify(B,null,2):String(B),"formatDetailValue"),w=l(O=>{const B=String(O||"").trim();if(!B)
return"";let J=null;if(B.startsWith("{"))try{J=JSON.parse(B)}catch{J=null}if(J&&typeof J=="object"&&
!Array.isArray(J)){const F=Object.keys(J).map(q=>{const te=b(q,J[q]),V=i[q]||q;return te.length>120||
te.includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(V)}</div><pre class="bot-log-pre">${escapeHtml(
te)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(V)}:</span> <span class="text-gray-\
300">${escapeHtml(te)}</span></div>`}).join("");return B.length>600?`<details class="mt-1"><summary \
class="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${F}</details>`:`<div c\
lass="mt-1 space-y-1">${F}</div>`}return B.length>300?`<details class="mt-1"><summary class="cursor-\
pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(
B)}</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(B)}</div>`},"renderD\
etails"),x=l(async(O,B)=>{const J=await apiFetch(O,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(B)}),F=await J.json().catch(()=>({}));if(!J.ok)throw new Error(F.error||
String(J.status));return F},"postJson"),k=l((O,B)=>`<span class="${B} text-white px-2 py-0.5 rounded\
">${escapeHtml(O)}</span>`,"badge"),C=l(()=>{const O=get("bot-log-account-list");if(!O)return;const B=c.
accounts,J=B.length&&B.every(q=>c.selected.has(q.user_id)),F=get("bot-log-list-toolbar");if(F&&(F.innerHTML=
`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${J?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!B.length){O.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}O.innerHTML=B.map(q=>{const te=[];q.exists||te.push(k("\u524A\u9664\u6E08\u307F","bg-gray-600")),
q.is_bot_banned&&te.push(k("BAN\u4E2D","bg-red-600")),q.lock_remaining_seconds>0&&te.push(k(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
ceil(q.lock_remaining_seconds/60)}\u5206\uFF09`,"bg-yellow-600"));const V=q.evidence_count?`\u8A18\u9332 ${q.
evidence_count}\u4EF6\u30FB\u6700\u7D42 ${escapeHtml(g(q.last_event_at))}`:"\u8A18\u9332\u306A\u3057";
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
                                <div class="text-[10px] text-gray-500">${V}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${q.user_id}" data-username="${escapeHtml(
q.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),T=l(async()=>{const O=get("bot-log-acc\
ount-list");if(!O)return;O.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const B=get("bot-log-search"),J=B?B.value.
trim():"";try{const F=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(J)}`,{cache:"\
no-store"}),q=await F.json().catch(()=>({}));if(!F.ok)throw new Error(q.error||String(F.status));c.accounts=
q.accounts||[];const te=new Set(c.accounts.map(V=>V.user_id));Array.from(c.selected).forEach(V=>{te.
has(V)||c.selected.delete(V)}),C()}catch{O.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\
\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),A=l(O=>{if(!O)return"";const B=O.locks||[],J=O.banned_identifiers||[],
F=[];O.exists||F.push(k("\u524A\u9664\u6E08\u307F","bg-gray-600")),O.exists&&(F.push(k(O.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",O.detection_enabled?"bg-gray-600":"bg-gray-700")),F.push(k(O.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",O.is_bot_banned?"bg-red-600":"bg-gray-600"))),F.push(k(B.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",B.length?"bg-yellow-600":"bg-gray-600")),J.length&&F.push(k(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${J.length}\u4EF6`,"bg-red-600")),O.is_admin&&F.push(k("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const q=!!(r.counts.ban||r.counts.related_ban),te=O.is_bot_banned?`<div class="text-\
gray-300">BAN\u306E\u7406\u7531: ${escapeHtml(O.bot_ban_reason||"-")}\uFF08${escapeHtml(g(O.bot_banned_at))}\
\uFF09</div>${q?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",V=B.map(ne=>{const xe=a[ne.source]||ne.source,ve=ne.identifier?` ${escapeHtml(ne.identifier)}${ne.
source==="cookie"?"\u2026":""}`:"",we=ne.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
ne.origin.username||String(ne.origin.user_id||"-"))}\uFF08${escapeHtml(n[ne.origin.client]||ne.origin.
client||"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(xe)}${ve}\
\u30FB\u6B8B\u308A${escapeHtml(y(ne.remaining_seconds))}\u30FB${escapeHtml(ne.reason||"")}${we}</div\
>`}).join(""),X=J.map(ne=>`<div class="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[ne.
kind]||ne.kind)} ${escapeHtml(ne.identifier)}${ne.kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(ne.
reason||"-")}\uFF08${escapeHtml(g(ne.created_at))}\uFF09</div>`).join(""),re=J.length?'<div class="t\
ext-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002</div>':
"",ce=O.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${y(O.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",Q=O.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String(O.appeal_count))}\u4EF6`:"",ue=[];return B.length&&ue.push('<button class="bot-log-unlock bg-\
yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
J.length&&ue.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 r\
ounded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${F.join("")}</div>
                        ${te}
                        ${V}
                        ${X}
                        ${re}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String(O.lock_count))} / ${escapeHtml(String(O.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(ce)}\
\u30FB\u5931\u6557 ${escapeHtml(String(O.turnstile_fail_count))} / ${escapeHtml(String(O.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round((O.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round((O.behavior_score||0)*10)/10))}\uFF09${Q}</div>
                        ${ue.length?`<div class="flex flex-wrap gap-2 pt-1">${ue.join("")}</div>`:""}\

                    </div>`},"renderState"),N=l(()=>{const O=l((J,F,q)=>`<button class="bot-log-filt\
er ${r.type===J?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(J)}">${escapeHtml(F)} (${q})</button>`,"chip"),B=[O("","\u3059\u3079\u3066",
r.total)];return Object.keys(r.counts).sort((J,F)=>r.counts[F]-r.counts[J]).forEach(J=>{B.push(O(J,(t[J]||
{label:J}).label,r.counts[J]))}),B.join("")},"renderFilters"),G=l(O=>{const B=t[O.event_type]||{label:O.
event_type,cls:"bg-gray-600"},J=O.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[O.client]||O.client)}</span>`:"",F=r.selected.has(O.id)?"checked":"",q=O.score!==
null&&O.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String(O.score))}${O.
behavior_score!==null&&O.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String(O.behavior_score))}\
\uFF09`:""}</span>`:"",te=O.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
O.reasons)}</div>`:"",V=O.ip_address||O.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${O.ip_address?"IP "+escapeHtml(O.ip_address):""}${O.ip_address&&O.user_agent?"\u30FB":""}${escapeHtml(
O.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${O.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${O.id}" ${F}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(g(O.created_at))}</span>
                            <span class="${B.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(B.label)}\
</span>
                            ${J}
                            ${q}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${O.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${te}
                        ${w(O.details)}
                        ${V}
                    </div>`},"renderItem"),L=l(()=>{const O=d();if(!O)return;const B=r.items.length?
r.items.map(G).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
J=r.items.length&&r.items.every(F=>r.selected.has(F.id));O.innerHTML=`
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <button class="bot-log-back bg-gray-700 hover:bg-gray-600 text-white px-2 py\
-1 rounded text-xs"><i class="fas fa-arrow-left mr-1"></i>\u4E00\u89A7\u306B\u623B\u308B</button>
                        <div class="text-sm font-bold text-white bot-log-wrap">${escapeHtml(r.username||
"ID "+r.userId)}</div>
                        <button class="bot-log-reload ml-auto bg-gray-700 hover:bg-gray-600 text-whi\
te px-2 py-1 rounded text-xs">\u66F4\u65B0</button>
                    </div>
                    <div class="flex-1 overflow-y-auto space-y-2">
                        ${A(r.data&&r.data.state)}
                        <div class="flex flex-wrap gap-1 text-xs">${N()}</div>
                        <div class="flex flex-wrap items-center gap-2 text-xs">
                            <label class="flex items-center gap-1 text-gray-300"><input type="checkb\
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${J?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                            <button class="bot-log-delete-selected bg-red-600 hover:bg-red-500 text-\
white px-2 py-1 rounded" ${r.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u8A18\u9332\u3092\u524A\u9664${r.
selected.size?`\uFF08${r.selected.size}\u4EF6\uFF09`:""}</button>
                            <button class="bot-log-delete-all bg-red-800 hover:bg-red-700 text-white\
 px-2 py-1 rounded" ${r.total?"":"disabled"}>\u3059\u3079\u3066\u306E\u8A18\u9332\u3092\u524A\u9664</button>
                        </div>
                        <div class="space-y-2">${B}</div>
                        ${r.hasMore?'<button class="bot-log-more w-full bg-gray-700 hover:bg-gray-60\
0 text-white px-2 py-1.5 rounded text-xs">\u3055\u3089\u306B\u8AAD\u307F\u8FBC\u3080</button>':""}
                    </div>`},"render"),$=l(async(O=!1)=>{const B=d();if(!B||!r.userId)return;O||(B.innerHTML=
'<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');
const J=new URLSearchParams({user_id:String(r.userId),limit:String(50)});r.type&&J.set("type",r.type),
O&&r.items.length&&J.set("before_id",String(r.items[r.items.length-1].id));try{const F=await apiFetch(
`/api/bot/evidence?${J.toString()}`,{cache:"no-store"}),q=await F.json().catch(()=>({}));if(!F.ok)throw new Error(
q.error||String(F.status));r.counts=q.counts||{},r.total=q.total||0,r.hasMore=!!q.has_more,q.user&&(r.
username=q.user.username||r.username,r.exists=!!q.user.exists),O?r.items=r.items.concat(q.items||[]):
(r.items=q.items||[],r.data=q),L()}catch{O||(B.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),M=l(async(O,B)=>{if(confirm(B))try{const J=await x("/api/bot/evidence/delete",
Object.assign({user_id:r.userId},O));showToast(`${J.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),r.selected.clear(),await $(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),P=l(async(O,B,J)=>{if(confirm(B))try{await x(O,{user_id:r.userId}),showToast(
J,"success"),await $(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","e\
rror",!0)}},"runAccountAction"),R=l(()=>{const O=d();O&&(O.classList.add("hidden"),O.classList.remove(
"flex"),O.innerHTML="");const B=m();B&&(B.classList.remove("hidden"),B.classList.add("flex")),r.userId=
0},"showList"),D=l(async()=>{R(),await T()},"openList"),H=l(async(O,B="")=>{const J=d();if(!J)return;
Object.assign(r,{userId:Number(O)||0,username:B,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),r.selected.clear();const F=m();F&&(F.classList.add("hidden"),F.classList.remove("flex")),
J.classList.remove("hidden"),J.classList.add("flex"),await $(!1)},"open");return{open:H,openList:D,showList:R,
bind:l(()=>{const O=get("bot-log-modal");if(!O||O.dataset.bound==="1")return;O.dataset.bound="1";const B=get(
"bot-log-search");B&&B.addEventListener("keydown",J=>{J.key==="Enter"&&T()}),O.addEventListener("cha\
nge",J=>{const F=J.target;if(!(!F||!F.classList))if(F.classList.contains("bot-log-select")){const q=Number(
F.getAttribute("data-log-id"));F.checked?r.selected.add(q):r.selected.delete(q),L()}else if(F.classList.
contains("bot-log-select-all"))r.items.forEach(q=>{F.checked?r.selected.add(q.id):r.selected.delete(
q.id)}),L();else if(F.classList.contains("bot-log-account-select")){const q=Number(F.getAttribute("d\
ata-user-id"));F.checked?c.selected.add(q):c.selected.delete(q),C()}else F.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(q=>{F.checked?c.selected.add(q.user_id):c.selected.delete(
q.user_id)}),C())}),O.addEventListener("click",async J=>{const F=J.target.closest("button");if(!(!F||
F.disabled))if(F.id==="bot-log-search-btn"||F.id==="bot-log-refresh-btn")F.id==="bot-log-refresh-btn"&&
B&&(B.value=""),await T();else if(F.classList.contains("bot-log-open-account")){const q=Number(F.getAttribute(
"data-user-id"));q&&await H(q,F.getAttribute("data-username")||"")}else if(F.classList.contains("bot\
-log-delete-accounts")){const q=Array.from(c.selected);if(!q.length||!confirm(`\u9078\u629E\u3057\u305F${q.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const te=await x("/api/bot/evidence/delete",{user_ids:q});showToast(`${te.deleted||0}\u4EF6\u306E\u8A18\u9332\
\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await T()}catch{showToast(
"\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}}else if(F.
classList.contains("bot-log-back"))await D();else if(F.classList.contains("bot-log-reload"))await $(
!1);else if(F.classList.contains("bot-log-more"))await $(!0);else if(F.classList.contains("bot-log-f\
ilter"))r.type=F.getAttribute("data-type")||"",r.selected.clear(),await $(!1);else if(F.classList.contains(
"bot-log-delete-one")){const q=Number(F.getAttribute("data-log-id"));q&&await M({ids:[q]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\
\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
F.classList.contains("bot-log-delete-selected")?r.selected.size&&await M({ids:Array.from(r.selected)},
`\u9078\u629E\u3057\u305F${r.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
F.classList.contains("bot-log-delete-all")?await M({all:!0},`${r.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${r.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
F.classList.contains("bot-log-unlock")?await P("/api/bot/account/clear-lock",`${r.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):F.classList.contains("bot-log-unbloc\
k")&&await P("/api/bot/account/clear-identifiers",`${r.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
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
["save","\u4FDD\u5B58\u3057\u3066\u5206\u5C90"]]){const r=document.createElement("button");r.value=a,
r.textContent=o,r.className="rounded px-3 py-2 border border-gray-600",i.appendChild(r)}t.appendChild(
i),t.addEventListener("close",()=>{const a=t.returnValue==="save";t.remove(),e(a)},{once:!0}),document.
body.appendChild(t),t.showModal()})}l(confirmBranchSave,"confirmBranchSave");async function branchChatMessage(...e){
return window.ActivityLog?window.ActivityLog.operation("branch",()=>branchChatMessageLogged.apply(this,
e)):branchChatMessageLogged.apply(this,e)}l(branchChatMessage,"branchChatMessage");async function branchChatMessageLogged(e,t){
var a;if(branchCreating||abortController||!currentThreadId||!/^\d+$/.test(String(e))||temporaryChatEnabled&&
!await confirmBranchSave())return;const n=String(currentThreadId),i=n+":"+e;branchRequests.has(i)||branchRequests.
set(i,crypto.randomUUID().replace(/-/g,"")),branchCreating=!0,t&&(t.disabled=!0);try{const o=await apiFetch(
`/api/threads/${encodeURIComponent(n)}/branch`,{method:"POST",headers:{"Content-Type":"application/j\
son"},body:JSON.stringify({message_id:Number(e),request_id:branchRequests.get(i),confirm_persist_temporary:!!temporaryChatEnabled})}),
r=await o.json();if(!o.ok)throw new Error(r.error||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
branchRequests.delete(i),await loadMessages(r.id,{targetMessageId:r.leaf_id}),await loadThreads(),showToast(
"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90\u3057\u307E\u3057\u305F")}catch(o){(a=
window.ActivityLog)==null||a.log("branch.error"),showToast(o.message||"\u5206\u5C90\u3092\u4F5C\u6210\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}finally{branchCreating=!1,t&&(t.disabled=!1)}}l(branchChatMessageLogged,"branchChatMessa\
geLogged");function renderBranchOrigin(e){let t=get("branch-origin-banner");if(!t){const n=get("chat\
-container");if(!n)return;t=document.createElement("div"),t.id="branch-origin-banner",t.className="p\
x-4 py-2 text-xs text-gray-400 border-b border-gray-700";const i=get("total-token-bar");i?i.before(t):
n.parentNode.insertBefore(t,n)}if(t.replaceChildren(),t.hidden=!e,!!e)if(e.available&&e.thread_id&&e.
message_id){const n=document.createElement("a");n.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u3092\u958B\u304F",
n.className="text-blue-400 hover:underline",n.href="/c/"+encodeURIComponent(e.thread_id)+"?message="+
Number(e.message_id),n.onclick=i=>{i.preventDefault(),loadMessages(e.thread_id,{targetMessageId:e.message_id})},
t.appendChild(n)}else t.textContent="\u5206\u5C90\u5143\u306E\u30C1\u30E3\u30C3\u30C8\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093"}
l(renderBranchOrigin,"renderBranchOrigin");function displayMissingMessageFiles(e,t){const n=l(i=>{try{
const a=new URL(i,location.origin);return a.origin===location.origin?decodeURIComponent(a.pathname).
replace(/^\/files\/(thumb\/)?/,""):""}catch{return String(i)}},"normalize");for(const i of t){if(!i.
missing_files||!i.missing_files.length)continue;const a=e.querySelector("#msg-"+Number(i.id));if(!a)
continue;const o=new Set(i.missing_files.flatMap(r=>{const c=String(r).replace(/^\/files\//,"");return[
c,c.replace(/^\d+\//,"")]}));a.querySelectorAll("img[src], a[href], audio[src], video[src], source[s\
rc], .file-thumb[onclick]").forEach(r=>{let c=n(r.getAttribute("src")||r.getAttribute("href")||"");if(r.
classList.contains("file-thumb")){const g=(r.getAttribute("onclick")||"").match(/window.open\('([^']+)'/);
g&&(c=n(g[1]))}if(!o.has(c))return;const d=document.createElement("span");d.className="inline-block \
rounded bg-gray-800 p-3 text-xs text-gray-400",d.textContent="\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",
d.title=c.split("/").pop();const m=r.closest(".chat-image-frame")||(r.tagName==="SOURCE"?r.parentNode:
r);m&&a.contains(m)&&m.replaceWith(d)})}}l(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}l(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},o=1/0;for(let r=1;r<=i;r++){if(i%r)continue;const c=i/r,d=Math.abs(
Math.log(Math.max(1,t)/r/(Math.max(1,n)/c)));(d<o-1e-9||Math.abs(d-o)<=1e-9&&(t>=n?r>a.columns:c>a.rows))&&
(a={columns:r,rows:c},o=d)}return a}l(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,o=Math.round(e/n.columns*a/2),r=Math.round(t/n.rows*a/2),c=[];for(let d=0;d<
n.rows;d++)for(let m=0;m<n.columns;m++){const g=Math.floor(m*e/n.columns),y=Math.floor((m+1)*e/n.columns),
b=Math.floor(d*t/n.rows),w=Math.floor((d+1)*t/n.rows);c.push({index:c.length,row:d,column:m,cellLeft:g,
cellTop:b,cellRight:y,cellBottom:w,left:Math.max(0,g-o),top:Math.max(0,b-r),right:Math.min(e,y+o),bottom:Math.
min(t,w+r)})}return c}l(splitTiles,"splitTiles");function splitFits(e,t,n){return Math.floor(t/e.columns)>=
16&&Math.floor(n/e.rows)>=16}l(splitFits,"splitFits");function splitTileCaption(e,t,n,i){return`Tile\
 ${e.index+1}/${t.columns*t.rows}  row ${e.row+1}/${t.rows}, col ${e.column+1}/${t.columns}  x ${e.left}\
-${e.right}, y ${e.top}-${e.bottom} of ${n}x${i}`}l(splitTileCaption,"splitTileCaption");function splitBaseName(e){
return String(e).split("/").pop().replace(/\.[^.]*$/,"").replace(/[\\/:*?"<>|\s]+/g,"_").replace(/^[_.]+|[_.]+$/g,
"").slice(0,80)||"image"}l(splitBaseName,"splitBaseName");function splitTileName(e,t,n){const i=n.columns*
n.rows;return`${e}_part${String(t.index+1).padStart(String(i).length,"0")}of${i}_r${t.row+1}c${t.column+
1}.png`}l(splitTileName,"splitTileName");function readSplitCompression(){let e;try{e=JSON.parse(localStorage.
getItem(SPLIT_COMPRESSION_KEY)||"{}")}catch{}return e=e&&typeof e=="object"?e:{},{enabled:e.enabled!==
!1,maxSizeMB:Math.max(.1,Math.min(10,Number(e.maxSizeMB)||1)),maxDimension:splitClamp(e.maxDimension,
256,8192,1920),outputType:["original","image/jpeg","image/png","image/webp"].includes(e.outputType)?
e.outputType:"original",formatOnly:e.formatOnly===!0}}l(readSplitCompression,"readSplitCompression");
function splitOptions(){return{pieces:splitClamp(get("image-split-pieces").value,2,30,4),overlap:splitClamp(
get("image-split-overlap").value,0,50,10),includeOriginal:get("image-split-original").checked,markOriginal:get(
"image-split-mark-original").checked,markPieces:get("image-split-mark-pieces").checked}}l(splitOptions,
"splitOptions");function splitCanvas(e,t){const n=document.createElement("canvas");if(n.width=e,n.height=
t,n.width!==e||n.height!==t||!n.getContext("2d"))throw new Error("\u753B\u50CF\u304C\u5927\u304D\u3059\u304E\u308B\u305F\u3081\u51E6\u7406\u3067\u304D\u307E\u305B\u3093\u3002");
return n}l(splitCanvas,"splitCanvas");async function loadSplitImage(e){const t=URL.createObjectURL(e),
n=new Image;try{return await new Promise((i,a)=>{n.onload=i,n.onerror=()=>a(new Error(`\u753B\u50CF\u3092\u8AAD\u307F\u8FBC\u3081\u307E\u305B\u3093: ${e.
name}`)),n.src=t}),{image:n,release:l(()=>{n.src="",URL.revokeObjectURL(t)},"release")}}catch(i){throw URL.
revokeObjectURL(t),i}}l(loadSplitImage,"loadSplitImage");function drawSplitOverview(e,t,n,i,a,o){const r=e.
getContext("2d"),c=e.width/n,d=e.height/i;r.drawImage(t,0,0,e.width,e.height);const m=o[0],g=m.right-
m.cellRight,y=m.bottom-m.cellBottom;r.fillStyle="rgba(250,204,21,0.18)";const b=[];for(let x=1;x<a.columns;x++){
const k=Math.floor(x*n/a.columns);r.fillRect((k-g)*c,0,2*g*c,e.height),b.push([k*c,0,k*c,e.height])}
for(let x=1;x<a.rows;x++){const k=Math.floor(x*i/a.rows);r.fillRect(0,(k-y)*d,e.width,2*y*d),b.push(
[0,k*d,e.width,k*d])}drawSplitLines(r,b,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const w=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));r.font=`bold ${w}px monospace`,
r.textAlign="center",r.textBaseline="middle";for(const x of o){const k=(x.cellLeft+x.cellRight)/2*c,
C=(x.cellTop+x.cellBottom)/2*d;r.lineWidth=Math.max(2,w/10),r.strokeStyle="rgba(0,0,0,0.7)",r.strokeText(
String(x.index+1),k,C),r.fillStyle="#facc15",r.fillText(String(x.index+1),k,C)}}l(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,o,r,c])=>{e.moveTo(a,o),e.lineTo(r,c)}),e.stroke(),e.restore()}l(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(o=>o?i(
o):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}l(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const o=a.
naturalWidth,r=a.naturalHeight,c=chooseSplitGrid(t.pieces,o,r);if(!splitFits(c,o,r))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const d=splitTiles(o,r,c,t.overlap),m=splitBaseName(e.name),g=[],y=l(async b=>{g.push(await applyImageCompression(
b,n))},"append");if(t.includeOriginal)if(t.markOriginal){const b=splitCanvas(o,r);drawSplitOverview(
b,a,o,r,c,d),await y(await splitCanvasFile(b,`${m}_split_grid.png`))}else await y(e);for(const b of d){
const w=b.right-b.left,x=b.bottom-b.top;let k=Math.max(12,Math.min(40,Math.max(w,x)/45)),C=0;const T=splitTileCaption(
b,c,o,r);if(t.markPieces){const G=splitCanvas(1,1),L=G.getContext("2d");L.font=`bold ${k}px monospac\
e`,k=Math.max(6,k*Math.min(1,w*.96/L.measureText(T).width)),C=Math.max(12,Math.floor(k*1.8)),G.width=
0}const A=splitCanvas(w,x+C),N=A.getContext("2d");if(C&&(N.fillStyle="#111827",N.fillRect(0,0,w,C)),
N.drawImage(a,b.left,b.top,w,x,0,C,w,x),t.markPieces){N.fillStyle="#fff",N.font=`bold ${k}px monospa\
ce`,N.textBaseline="middle",N.fillText(T,w*.02,C/2);const G=[];for(const L of[b.cellLeft,b.cellRight])
L>b.left&&L<b.right&&G.push([L-b.left,C,L-b.left,C+x]);for(const L of[b.cellTop,b.cellBottom])L>b.top&&
L<b.bottom&&G.push([0,C+L-b.top,w,C+L-b.top]);drawSplitLines(N,G,Math.max(1.5,Math.min(4,Math.max(w,
x)/500)),!0)}await y(await splitCanvasFile(A,splitTileName(m,b,c)))}return g}finally{i.release()}}l(
renderSplitFiles,"renderSplitFiles");function clearSplitDownloads(){splitState.urls.forEach(e=>URL.revokeObjectURL(
e)),splitState.urls=[],get("image-split-downloads").replaceChildren()}l(clearSplitDownloads,"clearSp\
litDownloads");function closeImageSplit(){!get("image-split-modal")||splitState.busy||(splitState.revision++,
splitState.release&&splitState.release(),splitState.release=null,splitState.image=null,splitState.files=
[],clearSplitDownloads(),get("image-split-preview").width=0,hideModal("image-split-compression-modal"),
hideModal("image-split-modal"))}l(closeImageSplit,"closeImageSplit");async function openImageSplit(e){
const t=Array.from(e).filter(Boolean);t.length&&(t.length>10&&showToast("\u753B\u50CF\u306E\u5206\u5272\u306F\u4E00\u5EA6\u306B10\u679A\u307E\u3067\u3067\u3059\u3002\u5148\u982D10\u679A\u306E\u307F\u4F7F\u3044\u307E\
\u3059\u3002","warning",!0),!splitState.busy&&(closeImageSplit(),splitState.files=t.slice(0,10),splitState.
index=0,showModal("image-split-modal"),await loadSplitPreview()))}l(openImageSplit,"openImageSplit");
async function loadSplitPreview(){const e=++splitState.revision;splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null,get("image-split-status").textContent="\u8AAD\u307F\u8FBC\u307F\u4E2D...";
try{const t=await loadSplitImage(splitState.files[splitState.index]);if(e!==splitState.revision){t.release();
return}splitState.image=t.image,splitState.release=t.release,get("image-split-status").textContent="",
updateSplitPreview()}catch(t){e===splitState.revision&&(get("image-split-status").textContent=t.message,
updateSplitPreview())}}l(loadSplitPreview,"loadSplitPreview");function updateSplitPreview(){const e=splitOptions(),
t=splitState.image,n=get("image-split-preview");get("image-split-mark-original").disabled=!e.includeOriginal||
splitState.busy,get("image-split-page").textContent=`${splitState.index+1} / ${splitState.files.length}\
\uFF08\u3059\u3079\u3066\u540C\u3058\u8A2D\u5B9A\u3067\u5206\u5272\u3057\u307E\u3059\uFF09`,get("ima\
ge-split-prev").disabled=splitState.busy||splitState.index===0,get("image-split-next").disabled=splitState.
busy||splitState.index>=splitState.files.length-1,get("image-split-count").textContent=`\u4F5C\u6210\u3055\u308C\u308B\u753B\u50CF: ${(e.
pieces+Number(e.includeOriginal))*splitState.files.length}\u679A`;let i=!1;if(t){const a=t.naturalWidth,
o=t.naturalHeight,r=chooseSplitGrid(e.pieces,a,o);i=splitFits(r,a,o);const c=Math.min(1,800/a,520/o);
n.width=Math.round(a*c),n.height=Math.round(o*c),i?drawSplitOverview(n,t,a,o,r,splitTiles(a,o,r,e.overlap)):
n.getContext("2d").drawImage(t,0,0,n.width,n.height),get("image-split-dimensions").textContent=i?`${r.
columns}\u5217 \xD7 ${r.rows}\u884C\u30FB1\u679A\u3042\u305F\u308A\u7D04 ${Math.floor(a/r.columns)}\xD7${Math.
floor(o/r.rows)} px\uFF08\u5143\u753B\u50CF ${a}\xD7${o} px\uFF09`:`\u3053\u306E\u753B\u50CF\u306F\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${e.
pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3002\u5206\u5272\u6570\u3092\u6E1B\u3089\u3057\u3066\u304F\u3060\u3055\u3044\u3002`}else
n.width=0,get("image-split-dimensions").textContent="";get("image-split-save").disabled=get("image-s\
plit-attach").disabled=splitState.busy||!i}l(updateSplitPreview,"updateSplitPreview");function setSplitBusy(e){
splitState.busy=e,get("image-split-modal").querySelectorAll("input,button").forEach(t=>{t.disabled=e}),
updateSplitPreview()}l(setSplitBusy,"setSplitBusy");async function runImageSplit(...e){return window.
ActivityLog?window.ActivityLog.operation("image_split",()=>runImageSplitLogged.apply(this,e)):runImageSplitLogged.
apply(this,e)}l(runImageSplit,"runImageSplit");async function runImageSplitLogged(e){var i;if(splitState.
busy)return;const t=splitOptions(),n=readSplitCompression();if(e&&browserFastModeEnabled&&browserFastLocalFiles.
size+splitState.files.length*(t.pieces+Number(t.includeOriginal))>BROWSER_FAST_MAX_IMAGES){get("imag\
e-split-status").textContent="\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059";
return}setSplitBusy(!0),clearSplitDownloads();try{const a=[];splitState.release&&splitState.release(),
splitState.release=null,splitState.image=null;for(let o=0;o<splitState.files.length;o++)get("image-s\
plit-status").textContent=`\u51E6\u7406\u4E2D... ${o+1} / ${splitState.files.length}`,a.push(...await renderSplitFiles(
splitState.files[o],t,n));if(e){const o=Array.from(browserFastLocalFiles.values()).reduce((r,c)=>r+c.
file.size,0);if(browserFastModeEnabled&&o+a.reduce((r,c)=>r+c.size,0)>BROWSER_FAST_MAX_BYTES)throw new Error(
"\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u753B\u50CF\u306F4\u679A\u30FB\u5408\u8A0812MB\u307E\u3067\u3067\u3059");
await handleFiles(a,{skipCompression:!0}),setSplitBusy(!1),closeImageSplit()}else{for(const o of a){
const r=URL.createObjectURL(o);splitState.urls.push(r);const c=document.createElement("a");c.href=r,
c.download=o.name,c.textContent=o.name,c.className="block underline",get("image-split-downloads").append(
c),c.click()}get("image-split-status").textContent=`${a.length}\u679A\u306E\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9\u3092\u958B\u59CB\u3057\u307E\u3057\u305F\u3002\u4FDD\u5B58\u3055\u308C\u306A\u3044\u753B\u50CF\u306F\u4E0B\u306E\u30EA\u30F3\u30AF\u304B\u3089\u4FDD\u5B58\u3067\u304D\u307E\
\u3059\u3002`}}catch(a){(i=window.ActivityLog)==null||i.log("image_split.error"),get("image-split-st\
atus").textContent=a.message||"\u753B\u50CF\u3092\u5206\u5272\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002"}finally{
if(setSplitBusy(!1),splitState.files.length){const a=get("image-split-status").textContent;await loadSplitPreview(),
get("image-split-status").textContent=a}}}l(runImageSplitLogged,"runImageSplitLogged");function openSplitCompression(){
const e=readSplitCompression(),t="image-split-compression";get(`${t}-enabled`).checked=e.enabled,get(
`${t}-output-type`).value=e.outputType,get(`${t}-format-only`).checked=e.formatOnly,get(`${t}-max-si\
ze`).value=e.maxSizeMB,get(`${t}-max-dim`).value=e.maxDimension,syncSplitCompressionUi(),showModal(`${t}\
-modal`)}l(openSplitCompression,"openSplitCompression");function syncSplitCompressionUi(){const e="i\
mage-split-compression";for(const t of["size","dim"]){const n=get(`${e}-format-only`).checked;get(`${e}\
-max-${t}`).disabled=n,get(`${e}-${t}-wrap`).style.opacity=n?"0.4":"1"}}l(syncSplitCompressionUi,"sy\
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
ote-bar")].filter(N=>N&&N.getClientRects().length&&getComputedStyle(N).display!=="none").map(N=>N.getBoundingClientRect()),
r=Math.min(...o.map(N=>N.left)),c=Math.min(...o.map(N=>N.top)),d={left:r,top:c,width:Math.max(...o.map(
N=>N.right))-r,height:Math.max(...o.map(N=>N.bottom))-c},m=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,g={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},y=document.createElement("div");
y.hidden=!0,y.className="prompt-send-held-files",y.setAttribute("aria-hidden","true"),y.append(...g.
rows),document.body.appendChild(y);let b=!1,w=!1,x=!1;const k=l(()=>{b=!0},"onInput");t.addEventListener(
"input",k),t.value="",["file-input","photo-input","mask-input"].forEach(N=>{const G=get(N);G&&(G.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let C;i&&e&&!m&&
i.animate&&(C=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const T=l(()=>{!w||x||(g.files.forEach(N=>{var L,$;const G=($=
(L=N==null?void 0:N.rowObj)==null?void 0:L.row)==null?void 0:$.getAttribute("data-local-url");G&&URL.
revokeObjectURL(G)}),y.remove())},"releaseFiles"),A=l((N,G=!0)=>{w||(w=!0,N=N&&g.navigation===threadLoadSequence,
x=N,t.removeEventListener("input",k),N?(!b&&!t.value&&(t.value=g.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...g.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=g.mask),g.files.forEach((L,$)=>browserFastLocalFiles.set($,L)),g.
markers.forEach(L=>markerAppliedUploads.add(L)),n&&n.prepend(...g.rows),y.remove(),!currentQuote&&g.
quote&&(currentQuote=g.quote,currentQuoteMessageId=g.quoteId,renderQuoteBarContent(g.quote),get("quo\
te-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=g.progress),updateFilePreview(),
updateMaskPreview(),schedulePromptTokenEstimate(!0)):G&&T())},"finish");return{source:d,canRetry:l(()=>!b&&
g.navigation===threadLoadSequence,"canRetry"),stopDock:l(()=>C==null?void 0:C.cancel(),"stopDock"),files:g.
files,releaseFiles:T,accept:l((N=!0)=>A(!1,N),"accept"),restore:l(()=>A(!0),"restore")}}l(takePromptForSend,
"takePromptForSend");function animatePromptToMessage(e,t,n){var y;let i,a=0,o=!1;const r=[],c=get("c\
hat-container"),d=get("prompt-input"),m=window.matchMedia("(prefers-reduced-motion: reduce)"),g=l(()=>{
var b,w;o||(o=!0,(b=window.ActivityLog)==null||b.log("ui.hide"),t.stopDock(),cancelAnimationFrame(a),
r.forEach(x=>x.cancel()),i==null||i.remove(),e==null||e.classList.remove("prompt-send-hidden"),window.
removeEventListener("resize",g),window.removeEventListener("popstate",g),window.removeEventListener(
"pagehide",g),document.removeEventListener("visibilitychange",g),c==null||c.removeEventListener("whe\
el",g),c==null||c.removeEventListener("touchstart",g),d==null||d.removeEventListener("input",g),(w=window.
visualViewport)==null||w.removeEventListener("resize",g),m.removeEventListener("change",g))},"stop");
if(!n||m.matches||!e||!Element.prototype.animate)return g;try{e.classList.remove("fade-in"),e.classList.
add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var b;try{if(!e.isConnected)return g();const w=e.
querySelector(".message-bubble"),x=w.getBoundingClientRect(),k=c.getBoundingClientRect();if(!x.width||
!x.height||x.bottom<=k.top||x.top>=k.bottom)return g();const C=t.source;i=w.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach($=>$.removeAttribute("id")),i.querySelectorAll(".msg-contro\
ls, video, audio").forEach($=>$.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute("ari\
a-hidden","true"),i.inert=!0,Object.assign(i.style,{width:x.width+"px",height:x.height+"px",left:x.left+
"px",top:x.top+"px"}),document.body.appendChild(i),(b=window.ActivityLog)==null||b.log("ui.show");const T=Math.
min(C.width/x.width,Math.max(C.height,42)/x.height,1),A=C.left+(C.width-x.width*T)/2-x.left,N=C.top+
(C.height-x.height*T)/2-x.top,G=i.animate([{transform:`translate(${A}px, ${N}px) scale(${T})`,borderRadius:"\
12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"translate(0, 0\
) scale(1)",borderRadius:getComputedStyle(w).borderRadius,backgroundColor:getComputedStyle(w).backgroundColor}],
{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});r.push(G),r.push(i.animate([{opacity:1,
offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),e.classList.remove("prompt\
-send-hidden"),r.push(w.animate([{opacity:0,offset:0},{opacity:0,offset:.8},{opacity:1}],{duration:350,
fill:"both"}));const L=l(()=>{if(!e.isConnected||!userAutoScroll||document.hidden)return g();const $=w.
getBoundingClientRect();i.style.left=$.left+"px",i.style.top=$.top+"px",a=requestAnimationFrame(L)},
"watch");a=requestAnimationFrame(L),G.finished.then(g,g)}catch{console.warn("Prompt send animation f\
ailed"),g()}}),window.addEventListener("resize",g),window.addEventListener("popstate",g),window.addEventListener(
"pagehide",g),document.addEventListener("visibilitychange",g),c.addEventListener("wheel",g,{passive:!0}),
c.addEventListener("touchstart",g,{passive:!0}),d.addEventListener("input",g),(y=window.visualViewport)==
null||y.addEventListener("resize",g),m.addEventListener("change",g)}catch{console.warn("Prompt send \
animation failed"),g()}return g}l(animatePromptToMessage,"animatePromptToMessage");let costPricing=null,
costPricingFailed=!1,costBarPath=[],costBarReady=!1;const COST_ESTIMATE_NOTE="\u6709\u6599API\u306E\u901A\u5E38\u6599\u91D1\u306B\u3088\u308B\u6982\u7B97\uFF08USD\uFF09\u3002\u56DE\
\u7B54\u3054\u3068\u306E\u4FDD\u5B58\u6E08\u307F\u30E2\u30C7\u30EB\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5165\u529B\u4F7F\u7528\u91CF\u304C\u306A\u3044\u56DE\u7B54\u306F\u4F1A\u8A71\u5C65\u6B74\u304B\u3089\u63A8\u5B9A\u3057\u307E\u3059\u3002\u30E2\u30C7\u30EBAPI\u306E\u547C\u3073\u51FA\u3057\u3054\u3068\u306B\u5165\u529B\u30FB\u51FA\u529B\uFF08Thinking\u3092\u542B\u3080\uFF09\u3092\u5408\u7B97\u3057\u307E\u3059\u3002Ideogram\u306F\u30E2\u30C7\u30EB\u30FB\u54C1\u8CEA\u30FB\u901F\u5EA6\u30FB\u51E6\
\u7406\u5225\u306E\u5358\u4FA1\u3068\u753B\u50CF\u679A\u6570\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5358\u4FA1\u672A\u516C\u8868\u306E\u8A2D\u5B9A\u306F\u516C\u8868\u6700\u5B89\u5024\u3092\u4E0B\u9650\u3068\u3057\u3066\u300C\u4EE5\u4E0A\u30FB\u4E00\u90E8\u672A\u7B97\u5165\u300D\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u4F7F\u7528\u91CF\u672A\u53D6\u5F97\u306E\u547C\u3073\u51FA\u3057\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002API\u5831\u544A\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u30FB\u66F8\u8FBC\u3068\u691C\u7D22\u306A\u3069\u306E\u8FFD\u52A0\u6599\u91D1\
\u3092\u542B\u307F\u307E\u3059\u3002Batch API\u306F\u4E8B\u696D\u8005\u3054\u3068\u306E\u5272\u5F15\u7387\uFF08\u5272\u5F15\u306E\u306A\u3044\u4E8B\u696D\u8005\u3092\u542B\u3080\uFF09\u3001Gemini Flex\u306F\u6A19\u6E96\u6599\u91D1\u306E50%\u3067\u8A08\u7B97\u3057\u3001\u5272\u5F15\u5BFE\u8C61\u5916\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u306F\u901A\u5E38\u5358\u4FA1\u306E\u307E\u307E\u3067\u3059\u3002\u4FDD\u5B58\u6642\u9593\u3084\u5B9F\u884C\u6642\u9593\u306A\u3069\u53D6\
\u5F97\u3067\u304D\u306A\u3044\u8AB2\u91D1\u60C5\u5831\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002\u305D\u306E\u4ED6\u306E\u753B\u50CF\u30FB\u97F3\u58F0\u30FB\u52D5\u753B\u751F\u6210\u3001\u7A0E\u3001\u7121\u6599\u67A0\u30FB\u500B\u5225\u5951\u7D04\u306F\u542B\u307F\u307E\u305B\u3093\u3002\u6642\u9593\u5E2F\u3084\u5165\u529B\u7A2E\u5225\u3067\u5358\u4FA1\u304C\u7570\u306A\u308B\u5834\u5408\u306F\u7BC4\u56F2\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u904E\u53BB\u306E\u8ACB\u6C42\u984D\u3092\u518D\u73FE\u3059\u308B\u3082\u306E\u3067\u306F\u3042\u308A\u307E\u305B\u3093\u3002";
function costTokenNumber(e){if(e==null||e==="")return null;const t=Number(e);return Number.isFinite(
t)&&t>=0?t:null}l(costTokenNumber,"costTokenNumber");function costCacheUsage(e){let t=e;if(typeof e==
"string"){if(!e.includes("cache_"))return null;try{t=JSON.parse(e)}catch{return null}}if(!t||typeof t!=
"object"||!Array.isArray(t.pricing_calls))return null;let n=0,i=0,a=!1;const o={};for(const r of t.pricing_calls){
if(!r||typeof r!="object")continue;const c=costTokenNumber(r.cache_read_tokens),d=costTokenNumber(r.
cache_write_tokens);let m=0;for(const g of["5m","1h","30m","12h"]){const y=costTokenNumber(r[`cache_\
write_${g}_tokens`]);y!==null&&(o[g]=(o[g]||0)+y,m+=y,a=!0)}(c!==null||d!==null)&&(a=!0),n+=c||0,i+=
Math.max(d||0,m)}return a?{read:n,write:i,ttls:o}:null}l(costCacheUsage,"costCacheUsage");function costRate(e,t,n){
const i=e&&(e.aliases[t]||t),a=e&&e.models[i];if(!a)return null;const o={...a};for(const r of o.changes||
[])n>=r.from&&Object.assign(o,r);return o}l(costRate,"costRate");function addRequestCost(e,t,n,i,a=!1,o=!1){
if(e.requests++,String(t.model).startsWith("grok-")&&Number.isSafeInteger(t.cost_in_usd_ticks)&&t.cost_in_usd_ticks>=
0){e.low+=t.cost_in_usd_ticks/1e10,e.high+=t.cost_in_usd_ticks/1e10,e.priced++,t.complete===!1&&e.unpriced++;
return}const r=costRate(n,t.model,i);if(!r){e.unpriced++;return}if(r.unit&&r.unit!=="token"){requestUnitCost(
e,t,r,a,o);return}const c=costTokenNumber(t.input_tokens),d=costTokenNumber(t.output_tokens),m=r.image_per_image&&
t.images!=null||r.audio_tokens_per_second&&(t.audio_in_ms!=null||t.audio_out_ms!=null||t.duration_ms!=
null);if(c===null&&d===null&&!m){e.unpriced++;return}const g=r.threshold!=null&&(r.threshold_inclusive?
(c||0)>=r.threshold:(c||0)>r.threshold);g&&e.long++;const y=requestTokenCost(t,r,g,a,o);e.low+=y.low,
e.high+=y.high,y.lowerBound&&(e.lowerBound=!0),e.priced++,((c===null||d===null)&&!y.derived||t.complete===
!1||y.incomplete)&&e.unpriced++}l(addRequestCost,"addRequestCost");function legacyImageRequest(e,t){
let n=e.image_url;try{typeof n=="string"&&(n=JSON.parse(n))}catch{n=n?[n]:[]}const i=Array.isArray(n)?
n.length:0,a=t.get(String(e.parent_id)),o=(a==null?void 0:a.prompt_options)||{};let r=a==null?void 0:
a.image_url,c=Array.isArray(r)?r.length>0:!!r&&r!=="[]"&&r!=="null",d=a;const m=new Set;for(;!c&&(d==
null?void 0:d.parent_id)!=null&&!m.has(String(d.id));)m.add(String(d.id)),d=t.get(String(d.parent_id)),
(d==null?void 0:d.role)==="assistant"&&d.image_url&&d.image_url!=="[]"&&d.image_url!=="null"&&(c=!0);
const g=String(e.model||"");if(g.startsWith("gpt-image-"))return{model:e.model,images:i||1,operation:c?
"edit":"generate",quality:o.gpt_image_quality||"auto",size:o.gpt_image_size||"auto",complete:!!a};if(g.
startsWith("grok-imagine-image"))return{model:e.model,images:i||1,operation:c?"edit":"generate",quality:o.
grok_image_quality||"medium",resolution:o.grok_image_resolution||"1k",input_images:c?1:0,complete:!!a};
const y=e.model==="ideogram-4.6",b=["remove_object","remove_background"].includes(o.ideogram_operation)?
o.ideogram_operation:c&&(y||e.model==="ideogram-4.5")?"edit":"generate";let w=o.ideogram_quality;(![
"very_low","low","medium","high"].includes(w)||w==="very_low"&&b==="generate"&&!y)&&(w=b==="edit"||y?
"medium":"high");let x=null;if(y){const C=String(o.ideogram_resolution||"").toLowerCase();b==="gener\
ate"?x=C==="4k"&&w!=="very_low"?"4k":C==="2k"||C==="4k"?"2k":"1k":w==="very_low"&&(x="2k")}const k=a?
["turbo","default","quality"].includes(o.ideogram_speed)?o.ideogram_speed:"default":null;return{model:e.
model,images:i||null,operation:b,quality:w,speed:k,...x?{resolution:x}:{},complete:!!a}}l(legacyImageRequest,
"legacyImageRequest");function estimateConversationCost(e,t=costPricing,n=e){var c,d,m,g,y;const i={
low:0,high:0,priced:0,unpriced:0,inferred:0,long:0,requests:0,items:[]},a=l((b,w,x,k,C)=>{const T={low:i.
low,high:i.high,priced:i.priced,unpriced:i.unpriced,long:i.long};addRequestCost(i,b,t,w,k,C),i.items.
push(describeRequestCost(i,T,b,w,t,!!k,!!C,!!x))},"capture"),o=new Map((n||[]).filter(Boolean).map(b=>[
String(b.id),b])),r=new Set;for(const b of e||[]){if(!b||b.role!=="assistant"||r.has(String(b.id)))continue;
r.add(String(b.id));let w=null;try{w=typeof b.thought_data=="string"?JSON.parse(b.thought_data):b.thought_data}catch{}
!w&&Array.isArray(b.pricing_calls)&&(w={pricing_calls:b.pricing_calls});const x=((c=/^\d{4}-\d{2}-\d{2}/.
exec(b.created_at||""))==null?void 0:c[0])||(t==null?void 0:t.checked_at),k=((d=b.batch_job)==null?void 0:
d.state)==="JOB_STATE_SUCCEEDED"||b.batch===!0,C=(o.get(String(b.parent_id))||b).prompt_options||b.prompt_options||
{},T=C&&C.gemini_flex===!0||b.flex===!0;if(Array.isArray(w==null?void 0:w.pricing_calls)){w.pricing_calls.
length||i.priced++;for(const P of w.pricing_calls){if(!P||typeof P!="object"){i.unpriced++;continue}
const R=((m=/^\d{4}-\d{2}-\d{2}/.exec(P.created_at||""))==null?void 0:m[0])||x,D=costRate(t,P.model,
R),H=(D==null?void 0:D.image_selector)==="speed"&&!P.speed&&P.model===b.model?{...P,speed:legacyImageRequest(
b,o).speed}:P;a(H,R,!1,k,T)}continue}const A=costRate(t,b.model,x);if((A==null?void 0:A.unit)==="ima\
ge"){a(legacyImageRequest(b,o),x,!1,!1,T);continue}if(A!=null&&A.unit&&A.unit!=="token"){a(legacyUnitRequest(
b,o,A),x,!0,!1,T);continue}const N=costTokenNumber(b.tokens_out);if(N===null&&!b.content&&!b.tokens)
continue;if(N===null){i.unpriced++;continue}let G=(g=costTokenNumber(w==null?void 0:w.pricing_input_tokens))!=
null?g:costTokenNumber(b.tokens_in),L=!1,$=!1;if(G===null){i.inferred++,$=!0,G=0;let P=b.parent_id;const R=new Set(
[String(b.id)]);for(P==null&&(L=!0);P!=null;){const D=String(P);if(R.has(D)||!o.has(D)){L=!0;break}R.
add(D);const H=o.get(D),ee=H.role==="user"?costTokenNumber(H.tokens_in):(y=costTokenNumber(H.tokens_content))!=
null?y:costTokenNumber(H.tokens_out)===null?null:Math.max(0,Number(H.tokens_out)-(costTokenNumber(H.
tokens_thought)||0));ee===null?L=!0:G+=ee,P=H.parent_id}}const M={model:b.model,input_tokens:G,output_tokens:N,
complete:!L};if(A!=null&&A.image_per_image){let P=b.image_url;try{typeof P=="string"&&(P=JSON.parse(
P))}catch{P=P?[P]:[]}const R=Array.isArray(P)&&P.length?P.length:/!\[Image\]\(/i.test(String(b.content||
""))?1:0;R&&(M.images=R)}a(M,x,$,k,T)}return i}l(estimateConversationCost,"estimateConversationCost");
function costMoneyText(e){const t=Number(e||0);return t>0&&t<1e-6?"<$0.000001":`$${t.toFixed(6)}`}l(
costMoneyText,"costMoneyText");function costAmountText(e){return!e||!e.priced?"\u7B97\u51FA\u4E0D\u53EF":
e.lowerBound?`${costMoneyText(e.low)}\u4EE5\u4E0A`:e.high-e.low>1e-12?`${costMoneyText(e.low)}\u301C${costMoneyText(
e.high)}`:costMoneyText(e.low)}l(costAmountText,"costAmountText");function describeRequestCost(e,t,n,i,a,o,r,c){
const d=l(x=>costTokenNumber(x),"count");let m=d(n.cache_write_tokens)||0;for(const x of["5m","1h","\
30m","12h"])m=Math.max(m,d(n[`cache_write_${x}_tokens`])||0);const g=["5m","1h","30m","12h"].reduce(
(x,k)=>x+(d(n[`cache_write_${k}_tokens`])||0),0),y=["web_search_calls","file_search_calls","groundin\
g_queries","code_execution_calls"].reduce((x,k)=>x+(d(n[k])||0),0),b=e.priced>t.priced,w=b?e.unpriced>
t.unpriced?"\u4E00\u90E8\u672A\u7B97\u5165":"":costRate(a,n.model,i)?"\u4F7F\u7528\u91CF\u672A\u53D6\u5F97":
"\u5358\u4FA1\u672A\u767B\u9332";return{model:String(n.model||""),date:i,input:d(n.input_tokens),output:d(
n.output_tokens),cache_read:d(n.cache_read_tokens),cache_write:Math.max(m,g)||null,images:d(n.images),
pages:d(n.pages),characters:d(n.characters),songs:d(n.songs),duration_ms:d(n.duration_ms),tools:y,low:e.
low-t.low,high:e.high-t.high,priced:b,reason:w,inferred:!!c,long:e.long>t.long,batch:!!o,flex:!!r,api_key_id:String(
n.api_key_id||""),api_key_label:String(n.api_key_label||""),api_key_inferred:!!n.api_key_inferred}}l(
describeRequestCost,"describeRequestCost");function costItemTokenText(e){var n,i;const t=[];return e.
images!==null&&e.images!==void 0&&t.push(`${e.images}\u679A`),e.songs!==null&&e.songs!==void 0&&t.push(
`${e.songs}\u66F2`),e.pages!==null&&e.pages!==void 0&&t.push(`${e.pages}\u30DA\u30FC\u30B8`),e.characters!==
null&&e.characters!==void 0&&t.push(`${e.characters}\u6587\u5B57`),e.duration_ms!==null&&e.duration_ms!==
void 0&&t.push(`${(e.duration_ms/1e3).toFixed(1)}\u79D2`),(e.input!==null||e.output!==null)&&t.push(
`\u5165 ${(n=e.input)!=null?n:"-"} / \u51FA ${(i=e.output)!=null?i:"-"}`),(e.cache_read||e.cache_write)&&
t.push(`\u30AD\u30E3\u30C3\u30B7\u30E5 \u8AAD\u8FBC ${e.cache_read||0} / \u66F8\u8FBC ${e.cache_write||
0}`),e.tools&&t.push(`\u30C4\u30FC\u30EB ${e.tools}\u56DE`),t.join(" \xB7 ")}l(costItemTokenText,"co\
stItemTokenText");function costItemJpyText(e,t){if(!t||!e.priced)return"";const n=jpyMoneyText(e.low*
t.rate);return e.high-e.low>1e-12?`\u7D04${n}\u301C${jpyMoneyText(e.high*t.rate)}`:`\u7D04${n}`}l(costItemJpyText,
"costItemJpyText");function appendCostItemList(e,t){const n=document.createElement("div");n.className=
"text-[11px]",n.style.cssText="max-height:16rem;overflow-y:auto;margin-top:0.5rem;",t.forEach((i,a)=>{
const o=document.createElement("div");o.style.cssText="border-top:1px solid rgba(107,114,128,0.4);pa\
dding:0.25rem 0;";const r=document.createElement("div");r.className="flex justify-between";const c=document.
createElement("span");c.className="font-mono",c.textContent=`${a+1}. ${i.model||"\u30E2\u30C7\u30EB\u4E0D\u660E"}`;
const d=document.createElement("span");d.className="font-mono",d.textContent=costAmountText(i),r.append(
c,d),o.appendChild(r);const m=document.createElement("div");m.className="text-gray-400";const g=[];i.
inferred&&g.push("\u5C65\u6B74\u304B\u3089\u63A8\u5B9A"),i.long&&g.push("\u9577\u6587\u6599\u91D1"),
i.batch&&g.push("Batch"),i.flex&&g.push("Flex"),i.reason&&g.push(i.reason);const y=costItemJpyText(i,
costJpyRate),b=[costItemTokenText(i),i.date,g.join(" / "),y].filter(Boolean);b.length&&(m.textContent=
b.join(" \xB7 ")),b.length&&o.appendChild(m),n.appendChild(o)}),e.appendChild(n)}l(appendCostItemList,
"appendCostItemList");function costEstimateLabel(e,t="\u8CBB\u7528\u4E88\u60F3"){return!e||!e.priced?
`${t}: \u7B97\u51FA\u4E0D\u53EF`:`${t}: ${costAmountText(e)}${e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":
""}`}l(costEstimateLabel,"costEstimateLabel");const COST_JPY_RATE_STORAGE_KEY="cost_jpy_rate_v1",COST_JPY_RATE_TTL_MS=720*
60*1e3,COST_JPY_RATE_NOTE="\u7121\u6599\u306E\u516C\u958B\u70BA\u66FF\u30EC\u30FC\u30C8\uFF08\u6B27\u5DDE\u4E2D\u592E\u9280\u884C\u306E\u53C2\u8003\u30EC\u30FC\u30C8\u306A\u3069\uFF09\u3067\u30C9\u30EB\u304B\u3089\u5186\u3078\u63DB\u7B97\u3057\u305F\u6982\u7B97\u3067\u3059\u3002\u5B9F\u969B\u306E\u8ACB\u6C42\u6642\u306E\u30EC\u30FC\u30C8\u3068\u306F\u7570\u306A\u308A\u307E\u3059\u3002",
COST_JPY_RATE_SOURCES=[{name:"ECB",url:"https://api.frankfurter.dev/v2/rates?base=USD&quotes=JPY",read:l(
e=>{const t=Array.isArray(e)?e.find(n=>n&&n.quote==="JPY"):null;return{rate:Number(t==null?void 0:t.
rate),date:t==null?void 0:t.date}},"read")},{name:"currency-api",url:"https://cdn.jsdelivr.net/npm/@\
fawazahmed0/currency-api@latest/v1/currencies/usd.json",read:l(e=>{var t;return{rate:Number((t=e==null?
void 0:e.usd)==null?void 0:t.jpy),date:e==null?void 0:e.date}},"read")}];let costJpyRate=null,costJpyRateFailed=!1,
costJpyRateFailedAt=0,costJpyRatePromise=null;function normalizeJpyRate(e){const t=Number(e==null?void 0:
e.rate);return!Number.isFinite(t)||t<50||t>500?null:{rate:t,date:typeof e.date=="string"?e.date.slice(
0,10):"",source:e.source||""}}l(normalizeJpyRate,"normalizeJpyRate");function readCachedJpyRate(){try{
const e=JSON.parse(localStorage.getItem(COST_JPY_RATE_STORAGE_KEY)||"null");return!e||!(Date.now()-Number(
e.saved)<COST_JPY_RATE_TTL_MS)?null:normalizeJpyRate(e)}catch{return null}}l(readCachedJpyRate,"read\
CachedJpyRate");function loadJpyRate(){if(costJpyRate)return Promise.resolve(costJpyRate);if(costJpyRateFailed&&
Date.now()-costJpyRateFailedAt<6e4)return Promise.resolve(null);if(costJpyRatePromise)return costJpyRatePromise;
costJpyRateFailed=!1;const e=readCachedJpyRate();return e?(costJpyRate=e,Promise.resolve(e)):(costJpyRatePromise=
(async()=>{for(const t of COST_JPY_RATE_SOURCES){let n=null;try{const i=new AbortController;n=setTimeout(
()=>i.abort(),6e3);const a=await fetch(t.url,{signal:i.signal,credentials:"omit"});if(!a.ok)continue;
const o=normalizeJpyRate({...t.read(await a.json()),source:t.name});if(!o)continue;costJpyRate=o;try{
localStorage.setItem(COST_JPY_RATE_STORAGE_KEY,JSON.stringify({...o,saved:Date.now()}))}catch{}return o}catch{}finally{
clearTimeout(n)}}return costJpyRateFailed=!0,costJpyRateFailedAt=Date.now(),null})().finally(()=>{costJpyRatePromise=
null}),costJpyRatePromise)}l(loadJpyRate,"loadJpyRate");function jpyMoneyText(e){return e>0?e<.01?"0\
.01\u5186\u672A\u6E80":e<1e3?`${e.toFixed(2)}\u5186`:`${Math.round(e).toLocaleString("ja-JP")}\u5186`:
"0\u5186"}l(jpyMoneyText,"jpyMoneyText");function costEstimateJpyLabel(e,t){if(!e||!e.priced||!t)return"\
\u65E5\u672C\u5186\u63DB\u7B97: \u7B97\u51FA\u4E0D\u53EF";const n=jpyMoneyText(e.low*t.rate),i=e.lowerBound?
n.endsWith("\u672A\u6E80")?n:`${n}\u4EE5\u4E0A`:e.high-e.low>1e-12?`${n}\u301C${jpyMoneyText(e.high*
t.rate)}`:n,a=e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":"",o=t.date?`\u30FB${t.date}`:
"";return`\u65E5\u672C\u5186\u63DB\u7B97: \u7D04${i}${a}\uFF08\u53C2\u8003\u30EC\u30FC\u30C8 1\u30C9\u30EB=${t.
rate.toFixed(2)}\u5186${o}\uFF09`}l(costEstimateJpyLabel,"costEstimateJpyLabel");function fillCostEstimateJpy(e,t){
if(e){if(!t||!t.priced){e.textContent=`\u65E5\u672C\u5186\u63DB\u7B97: ${!costPricing&&!costPricingFailed?
"\u63DB\u7B97\u4E2D\u2026":"\u7B97\u51FA\u4E0D\u53EF"}`;return}if(e.title=COST_JPY_RATE_NOTE,e.textContent=
"\u65E5\u672C\u5186\u63DB\u7B97: \u63DB\u7B97\u4E2D\u2026",costJpyRate){e.textContent=costEstimateJpyLabel(
t,costJpyRate);return}loadJpyRate().then(n=>{e.textContent=n?costEstimateJpyLabel(t,n):"\u65E5\u672C\u5186\u63DB\u7B97: \u70BA\u66FF\u30EC\u30FC\u30C8\
\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"})}}l(fillCostEstimateJpy,"fillCo\
stEstimateJpy");function refreshCostBar(e=costBarPath){costBarReady=!0,costBarPath=e;const t=get("to\
tal-cost-estimate");if(!t)return;const n=e.some(a=>a&&a.role==="assistant"&&(a.content||Number(a.tokens_out)>
0));t.classList.toggle("hidden",!n);const i=estimateConversationCost(e,costPricing,e);t.textContent=
costPricing?costEstimateLabel(i,"\u8CBB\u7528\u4E88\u60F3"):`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?
"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`,t.title=COST_ESTIMATE_NOTE,
messageMeta.__total__&&(messageMeta.__total__.cost_estimate=costPricing?i:null),t.onclick=()=>openTokenDetail(
"__total__"),typeof document!="undefined"&&typeof document.querySelectorAll=="function"&&document.querySelectorAll(
".msg-cost-estimate-btn[data-msg-id]").forEach(a=>{const o=a.dataset.msgId,r=e.find(d=>d&&String(d.id)===
String(o));if(!r)return;const c=costPricing?estimateConversationCost([r],costPricing,e):null;messageMeta[o]&&
(messageMeta[o].cost_estimate=c),a.textContent=costPricing?costEstimateLabel(c,"\u8CBB\u7528\u4E88\u60F3"):
`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`,
a.title=COST_ESTIMATE_NOTE})}l(refreshCostBar,"refreshCostBar"),fetch("/static/model_pricing.json?ch\
ecked=2026-11-28",{credentials:"same-origin"}).then(e=>{if(!e.ok)throw new Error("pricing_unavailabl\
e");return e.json()}).then(e=>{costPricing=e,costBarReady&&refreshCostBar(),usageCostPendingData&&renderUsageCostSummary(
usageCostPendingData)}).catch(()=>{costPricingFailed=!0,costBarReady&&refreshCostBar(),usageCostPendingData&&
setUsageCostStatus("\u8CBB\u7528\u4E88\u60F3: \u7B97\u51FA\u4E0D\u53EF")});function legacyUnitRequest(e,t,n){
const i=t.get(String(e.parent_id));if(n.unit==="song")return{model:e.model,songs:1,complete:!0};if(n.
unit==="page"){const a=/合計処理ページ数:\s*(\d+)/.exec(String(e.content||""));return{model:e.
model,pages:a?Number(a[1]):1,complete:!!a}}if(n.unit==="character"){const a=i!=null&&i.content?String(
i.content).length:null;return{model:e.model,characters:a,complete:a!==null}}if(n.unit==="video"){const a=(i==
null?void 0:i.prompt_options)||{},o=String(e.model||"").startsWith("grok-")?a.grok_video_duration:a.
gemini_video_duration,r=costTokenNumber(o),c=String((String(e.model||"").startsWith("grok-")?a.grok_video_resolution:
a.gemini_video_resolution)||"").toLowerCase();return{model:e.model,duration_ms:r?r*1e3:null,resolution:c||
null,complete:!1}}return{model:e.model,complete:!1}}l(legacyUnitRequest,"legacyUnitRequest");function requestImageUnitCost(e,t,n,i,a){
var k,C,T,A,N,G,L;const o=a?.5:1,r=i&&(k=n.batch_multiplier)!=null?k:1,c=costTokenNumber(t.input_tokens),
d=costTokenNumber(t.output_tokens);if((c!==null||d!==null)&&(n.image_output!=null||n.output!=null)){
const $=requestTokenCost(t,n,!1,i,a);e.low+=$.low,e.high+=$.high,$.lowerBound&&(e.lowerBound=!0),e.priced++,
(t.complete===!1||$.incomplete)&&e.unpriced++;return}const m=costTokenNumber(t.images);if(m===null){
e.unpriced++;return}let g=null,y=null,b=!1,w=!1;if(n.per_image_rates){const $=n.per_image_rates,M=t.
quality&&t.quality!=="auto"?$[t.quality]:null,P=t.size&&t.size!=="auto"?t.size:null,R=[];M?P&&M[P]!=
null?R.push(M[P]):Object.values(M).forEach(D=>{typeof D=="number"&&R.push(D)}):Object.values($).forEach(
D=>Object.entries(D).forEach(([H,ee])=>{(!P||H===P)&&typeof ee=="number"&&R.push(ee)})),R.length&&(g=
m*Math.min(...R)*r*o,y=m*Math.max(...R)*r*o,R.length>1&&(w=!0))}if(g===null&&n.resolution_rates&&t.resolution&&
n.resolution_rates[t.resolution]!=null){const $=n.resolution_rates[t.resolution],M=typeof $=="number"?
$:(T=(C=$[t.quality])!=null?C:$.low)!=null?T:Object.values($)[0];Number.isFinite(M)&&(g=m*M,y=m*M)}if(g===
null){const $=(G=(A=n.image_rates)==null?void 0:A[t.operation])!=null?G:(N=n.image_rates)==null?void 0:
N.generate,M=$==null?void 0:$[n.image_selector==="speed"?t.speed:t.quality],P=String(t.resolution||"").
toLowerCase(),R=M&&typeof M=="object"&&P&&typeof M[P]=="number"?M[P]:null;if(typeof $=="number"||typeof M==
"number"||typeof R=="number"){const D=typeof $=="number"?$:typeof M=="number"?M:R;g=m*D*o,y=m*D*o}else{
const D=M&&typeof M=="object"?Object.entries(M).filter(([H,ee])=>H!=="minimum"&&typeof ee=="number").
map(([,H])=>H):[];if(D.length)g=m*Math.min(...D)*o,y=m*Math.max(...D)*o,Math.min(...D)!==Math.max(...D)&&
(w=!0);else{const H=(L=M==null?void 0:M.minimum)!=null?L:$==null?void 0:$.minimum;if(b=Number.isFinite(
H),!Number.isFinite(H)){e.unpriced++;return}g=m*H*o,y=m*H*o}}}const x=(costTokenNumber(t.input_images)||
0)*(n.input_image_rate||0);e.low+=g+x,e.high+=y+x,e.priced++,b&&(e.lowerBound=!0),(t.complete===!1||
b||w)&&e.unpriced++}l(requestImageUnitCost,"requestImageUnitCost");function requestUnitCost(e,t,n,i=!1,a=!1){
var y,b,w,x;if(n.unit==="image"){requestImageUnitCost(e,t,n,i,a);return}const o=i&&(y=n.batch_multiplier)!=
null?y:1,r=a?.5:1;let c=0,d=0,m=t.complete===!1,g=!1;if(n.unit==="song")c=d=((b=costTokenNumber(t.songs))!=
null?b:1)*(n.song_rate||0);else if(n.unit==="page"){const k=costTokenNumber(t.pages);if(k===null){e.
unpriced++;return}c=d=k*(n.page_rate||0)*o}else if(n.unit==="character"){const k=costTokenNumber(t.characters);
if(k===null){e.unpriced++;return}c=d=k*(n.character_rate||0)/1e6}else if(n.unit==="duration"){const k=costTokenNumber(
t.duration_ms),C=(costTokenNumber(t.audio_in_ms)||0)+(costTokenNumber(t.audio_out_ms)||0),T=k!==null?
k/6e4:t.audio_in_ms!=null||t.audio_out_ms!=null?C/6e4:null;if(T===null&&n.duration_rate_per_minute>0)
c=n.duration_rate_per_minute/60,d=n.duration_rate_per_minute,m=!0;else{const A=(T||0)*(n.duration_rate_per_minute||
0),N=(costTokenNumber(t.standby_ms)||0)/6e4*(n.standby_rate_per_minute||0),G=(costTokenNumber(t.text_turns)||
0)*(n.text_turn_rate||0);c=d=A+N+G}}else if(n.unit==="video"){const k=costTokenNumber(t.input_tokens),
C=costTokenNumber(t.video_output_tokens);if(n.video_output!=null&&k!==null&&C!==null){const T=costTokenNumber(
t.output_tokens)||C,A=Math.max(0,T-C);c=d=(k*(n.input||0)+A*(n.output||0)+C*n.video_output)/1e6}else{
const T=t.duration_ms!=null?costTokenNumber(t.duration_ms)/1e3:costTokenNumber(t.duration_seconds),A=t.
without_audio&&n.video_only_rates?n.video_only_rates:n.video_rates||{},N=String(t.resolution||"").toLowerCase(),
G=N&&A[N]!=null?A[N]:null,L=Object.values(A).filter($=>typeof $=="number");if(T!==null){const $=(w=G!=
null?G:A.default)!=null?w:L.length?Math.min(...L):0,M=(x=G!=null?G:A.default)!=null?x:L.length?Math.
max(...L):0,P=(costTokenNumber(t.input_images)||0)*(n.input_image_rate||0);c=T*$+P,d=T*M+P,G==null&&
L.length>1&&(m=!0)}else if(L.length){const $=String(t.model||"").startsWith("grok-")?5:8;c=$*Math.min(
...L),d=$*Math.max(...L),m=!0}else{e.unpriced++;return}}}else{e.unpriced++;return}e.low+=c*r,e.high+=
d*r,e.priced++,g&&(e.lowerBound=!0),m&&e.unpriced++}l(requestUnitCost,"requestUnitCost");function requestTokenCost(e,t,n,i,a=!1){
var fe,ye,ge,ie,it,st,vt,Mt,He,at,Ke,Lt,Rt,At,qe,ut,Ve,wt,tt,pt,Gt,Bt,Et,Ut,Ot,jt,Ft,zt,Dt,Y,ke,Re,Ge,
et,mt,ft,ot;const o=costTokenNumber(e.input_tokens),r=costTokenNumber(e.output_tokens);let c=o||0,d=r||
0,m=!1;const g=costTokenNumber(e.audio_input_tokens),y=costTokenNumber(e.text_input_tokens),b=costTokenNumber(
e.image_input_tokens),w=costTokenNumber(e.audio_output_tokens),x=costTokenNumber(e.image_output_tokens);
let k=null,C=null;t.audio_tokens_per_second&&(o===null&&(e.audio_in_ms!=null||e.duration_ms!=null)&&
(k=(costTokenNumber((fe=e.audio_in_ms)!=null?fe:e.duration_ms)||0)/1e3*t.audio_tokens_per_second,m=!0),
r===null&&e.audio_out_ms!=null&&t.audio_output!=null&&(C=(costTokenNumber(e.audio_out_ms)||0)/1e3*t.
audio_tokens_per_second,m=!0));const T=costTokenNumber(e.cache_read_tokens)||0,A=costTokenNumber(e.cache_write_tokens)||
0,N=["5m","1h","30m","12h"],G=N.map(Le=>costTokenNumber(e[`cache_write_${Le}_tokens`])||0),L=Math.max(
A,G.reduce((Le,$e)=>Le+$e,0)),$=Math.max(0,c-T-L),M=n?t.long_input:(ye=t.text_input)!=null?ye:t.input,
P=n?t.long_output:t.output,R=n&&(ge=t.long_cached_input)!=null?ge:t.cached_input,D=t.cache_write_multipliers||
{},H=i&&(ie=t.batch_multiplier)!=null?ie:1,ee=i?(it=t.batch_cached_multiplier)!=null?it:H:1,O=R!=null&&
H>0?ee/H:1,B=Math.max(0,L-G.reduce((Le,$e)=>Le+$e,0)),J=Object.values(D);let F=!!e.pricing_incomplete||
!!t.pricing_unlisted||T+L>c||!!e.explicit_cache;T&&R==null&&(F=!0);let q=0,te=0;if(g!==null||y!==null||
b!==null){const Le=costTokenNumber(e.cached_text_tokens),$e=costTokenNumber(e.cached_image_tokens),We=costTokenNumber(
e.cached_audio_tokens),nt=Le!==null||$e!==null||We!==null,Ct=Math.max(0,(y||0)-(Le||0)),$t=Math.max(
0,(b||0)-($e||0)),Nt=Math.max(0,(g||0)-(We||0)),xt=Math.max(0,$-Ct-$t-Nt);q=Ct*M+$t*((st=t.image_input)!=
null?st:M)+Nt*((vt=t.audio_input)!=null?vt:M)+xt*M,te=Ct*((Mt=t.max_input)!=null?Mt:M)+$t*((He=t.image_input)!=
null?He:M)+Nt*((at=t.audio_input)!=null?at:M)+xt*((Ke=t.max_input)!=null?Ke:M),nt?(q+=((Le||0)*(R!=null?
R:M)+($e||0)*((At=(Rt=(Lt=t.cached_image_input)!=null?Lt:R)!=null?Rt:t.image_input)!=null?At:M)+(We||
0)*((Ve=(ut=(qe=t.cached_audio_input)!=null?qe:R)!=null?ut:t.audio_input)!=null?Ve:M))*O,te+=((Le||0)*
(R!=null?R:M)+($e||0)*((pt=(tt=(wt=t.cached_image_input)!=null?wt:R)!=null?tt:t.image_input)!=null?pt:
M)+(We||0)*((Et=(Bt=(Gt=t.cached_audio_input)!=null?Gt:R)!=null?Bt:t.audio_input)!=null?Et:M))*O):(q+=
T*(R!=null?R:M)*O,te+=T*((Ot=(Ut=t.max_cached_input)!=null?Ut:R)!=null?Ot:M)*O)}else if(k!==null)q=k*
((jt=t.audio_input)!=null?jt:M),te=q;else{const Le=Math.max((Ft=t.max_input)!=null?Ft:M,...t.unit===
"image"&&t.image_input!=null?[t.image_input]:[]);q=$*M+T*(R!=null?R:M)*O,te=$*(n?M:Le)+T*(n?R!=null?
R:M:(Dt=(zt=t.max_cached_input)!=null?zt:R)!=null?Dt:M)*O}for(let Le=0;Le<N.length;Le++){const $e=D[N[Le]];
G[Le]&&$e==null&&(F=!0),q+=G[Le]*M*($e!=null?$e:1),te+=G[Le]*(n?M:(Y=t.max_input)!=null?Y:M)*($e!=null?
$e:1)}B&&(F=!0,q+=B*M*(J.length?Math.min(...J):1),te+=B*(n?M:(ke=t.max_input)!=null?ke:M)*(J.length?
Math.max(...J):1));let X=0,re=0,ce=0;if(x!==null&&t.image_output!=null){const Le=Math.max(0,d-x);X=Le*
P+x*t.image_output,re=Le*((Re=t.max_output)!=null?Re:P)+x*t.image_output}else if(w!==null&&t.audio_output!=
null){const Le=Math.max(0,d-w);X=Le*P+w*t.audio_output,re=Le*((Ge=t.max_output)!=null?Ge:P)+w*t.audio_output}else if(C!==
null&&t.audio_output!=null)X=C*t.audio_output,re=X;else if(t.image_per_image&&e.images!=null){const Le=costTokenNumber(
e.images)||0,$e=String(e.resolution||"").toLowerCase(),We=t.image_per_image,nt=(et=We[$e])!=null?et:
We.default,Ct=Object.values(We).filter(xt=>typeof xt=="number"),$t=nt!=null?nt:Ct.length?Math.min(...Ct):
0,Nt=nt!=null?nt:Ct.length?Math.max(...Ct):0;if(Le>0){m=!0,ce=Le*$t;const xt=x!==null?Math.max(0,d-x):
r!==null&&t.image_output!=null?0:d;X=xt*P,re=xt*((mt=t.max_output)!=null?mt:P)+Le*Math.max(0,Nt-$t)*
1e6}}else t.image_output!=null&&d>0?(X=d*P,re=d*t.image_output,F=!0):(X=d*P,re=d*((ft=t.max_output)!=
null?ft:P));const Q=H*(a?.5:1);let ue=((q+X)/1e6+ce)*Q,ne=((te+re)/1e6+ce)*Q,xe=!1;const ve=t.tool_rates||
{},we=l((Le,$e)=>{if(Le){if(ve[$e]==null){F=!0;return}ue+=Le*ve[$e],ne+=Le*ve[$e]}},"add"),Oe=costTokenNumber(
e.web_search_calls)||0;we(Oe,e.search_preview?"web_search_preview":"web_search"),Oe&&!e.search_preview&&
t.search_content_tokens&&(ue+=Oe*t.search_content_tokens*M/1e6,ne+=Oe*t.search_content_tokens*M/1e6),
we(costTokenNumber(e.file_search_calls),"file_search");const Ce=costTokenNumber(e.grounding_queries)||
0;we(ve.grounding_prompt!=null?Math.min(1,Ce):Ce,ve.grounding_prompt!=null?"grounding_prompt":"groun\
ding_query");const lt=costTokenNumber(e.code_execution_calls)||0;return lt&&!e.code_execution_free&&
(t.provider==="openai"?(we(((ot=costTokenNumber(e.container_sessions))!=null?ot:1)*(costTokenNumber(
e.container_memory_gb)||1),"container_session"),F=!0):t.provider==="anthropic"?(we(1,"code_execution\
_minimum"),F=!0,xe=!0):we(lt,"code_execution")),{low:ue,high:ne,incomplete:F,lowerBound:xe,derived:m}}
l(requestTokenCost,"requestTokenCost");const USAGE_COST_PERIODS=[{id:"1d",label:"1\u65E5\u4EE5\u5185"},
{id:"7d",label:"7\u65E5\u4EE5\u5185"},{id:"14d",label:"2\u9031\u9593\u4EE5\u5185"},{id:"1m",label:"1\
\u304B\u6708\u4EE5\u5185"},{id:"3m",label:"3\u304B\u6708\u4EE5\u5185"},{id:"1y",label:"1\u5E74\u9593"}];
let usageCostPeriod="1m",usageCostLoadedPeriod=null,usageCostRequest=0,usageCostPendingData=null;function usageCostElements(){
return{panel:get("tab-usage"),status:get("usage-cost-status"),total:get("usage-cost-total"),jpy:get(
"usage-cost-jpy"),meta:get("usage-cost-meta"),breakdown:get("usage-cost-breakdown"),note:get("usage-\
cost-note")}}l(usageCostElements,"usageCostElements");function markUsageCostPeriod(e){document.querySelectorAll(
"#usage-period-buttons [data-period]").forEach(t=>{const n=t.dataset.period===e;t.classList.toggle("\
is-active",n),t.setAttribute("aria-pressed",n?"true":"false")})}l(markUsageCostPeriod,"markUsageCost\
Period");function setUsageCostStatus(e){const t=usageCostElements().status;t&&(t.textContent=e)}l(setUsageCostStatus,
"setUsageCostStatus");function usageCostModelRows(e){const t=new Map;for(const n of e||[]){if(!n)continue;
const i=n.model||"\u30E2\u30C7\u30EB\u4E0D\u660E";let a=t.get(i);a||(a={model:i,low:0,high:0,priced:0,
unpriced:0,requests:0},t.set(i,a)),a.low+=Number(n.low)||0,a.high+=Number(n.high)||0,a.requests++,n.
priced?a.priced++:a.unpriced++}return[...t.values()].sort((n,i)=>i.high-n.high||i.low-n.low)}l(usageCostModelRows,
"usageCostModelRows");function usageCostKeyRows(e){const t=new Map;for(const n of e||[]){if(!n)continue;
const i=n.api_key_id||"unknown",a=n.api_key_label||"\u672A\u8A18\u9332\uFF08\u30AD\u30FC\u672A\u8A2D\u5B9A\uFF09";
let o=t.get(i);o||(o={key:i,label:a,low:0,high:0,priced:0,unpriced:0,requests:0,inferred:0},t.set(i,
o)),o.low+=Number(n.low)||0,o.high+=Number(n.high)||0,o.requests++,n.priced?o.priced++:o.unpriced++,
n.api_key_inferred&&o.inferred++}return[...t.values()].sort((n,i)=>i.high-n.high||i.low-n.low)}l(usageCostKeyRows,
"usageCostKeyRows");function appendUsageCostBreakdownSection(e,t,n,i,a){if(!n.length)return;const o=document.
createElement("div");o.className="text-[11px] font-bold text-gray-400 mt-3",o.textContent=t,e.appendChild(
o),n.forEach(r=>{const c=document.createElement("div");c.className="border-t border-gray-700/60 py-1\
.5";const d=document.createElement("div");d.className="flex justify-between gap-3";const m=document.
createElement("span");m.className="font-mono text-[11px] break-all",m.textContent=i(r);const g=document.
createElement("span");g.className="font-mono text-[11px] shrink-0",g.textContent=costAmountText(r)+(r.
unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":""),d.append(m,g),c.appendChild(d);const y=document.
createElement("div");y.className="text-[10px] text-gray-400",y.textContent=a(r),c.appendChild(y),e.appendChild(
c)})}l(appendUsageCostBreakdownSection,"appendUsageCostBreakdownSection");function renderUsageCostBreakdown(e,t=[]){
const n=usageCostElements().breakdown;n&&(n.replaceChildren(),appendUsageCostBreakdownSection(n,"API\
\u30AD\u30FC\u5225",t,i=>i.label,i=>`API ${i.requests}\u56DE \xB7 \u7B97\u5B9A ${i.priced}\u56DE`+(i.
unpriced?` \xB7 \u672A\u7B97\u5165 ${i.unpriced}\u56DE`:"")+(i.inferred?" \xB7 \u4E00\u90E8\u73FE\u5728\u306E\u8A2D\u5B9A\u304B\u3089\u63A8\u5B9A":
"")),appendUsageCostBreakdownSection(n,"\u30E2\u30C7\u30EB\u5225",e,i=>i.model,i=>`API ${i.requests}\
\u56DE \xB7 \u7B97\u5B9A ${i.priced}\u56DE`+(i.unpriced?` \xB7 \u672A\u7B97\u5165 ${i.unpriced}\u56DE`:
"")))}l(renderUsageCostBreakdown,"renderUsageCostBreakdown");function renderUsageCostSummary(e){const t=usageCostElements();
if(!t.panel||!e)return;if(!costPricing){usageCostPendingData=costPricingFailed?null:e,setUsageCostStatus(
`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`);
return}usageCostPendingData=null;const n=Array.isArray(e.messages)?e.messages:[],i=estimateConversationCost(
n,costPricing,n);if(t.total&&(t.textContent=costAmountText(i)+(i.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":
""),t.title=COST_ESTIMATE_NOTE),t.jpy&&fillCostEstimateJpy(t.jpy,i),t.meta){const a=[`\u56DE\u7B54 ${e.
answers}\u4EF6`,`API\u547C\u3073\u51FA\u3057 ${i.requests}\u56DE`];i.inferred&&a.push(`\u5C65\u6B74\u304B\u3089\u63A8\u5B9A ${i.
inferred}\u4EF6`),i.unpriced&&a.push(`\u4E00\u90E8\u672A\u7B97\u5165 ${i.unpriced}\u4EF6`),e.unreadable&&
a.push(`\u4F7F\u7528\u91CF\u672A\u53D6\u5F97 ${e.unreadable}\u4EF6`),e.truncated&&a.push("\u4EF6\u6570\u304C\u591A\u3044\u305F\u3081\u76F4\u8FD1\u5206\
\u306E\u307F"),t.meta.textContent=a.join(" \xB7 ")}renderUsageCostBreakdown(usageCostModelRows(i.items),
usageCostKeyRows(i.items)),setUsageCostStatus(""),t.note&&t.note.classList.remove("hidden")}l(renderUsageCostSummary,
"renderUsageCostSummary");function loadUsageCostSummary(e=usageCostPeriod){const t=usageCostElements();
if(!t.panel||(usageCostPeriod=USAGE_COST_PERIODS.some(i=>i.id===e)?e:"1m",markUsageCostPeriod(usageCostPeriod),
usageCostLoadedPeriod===usageCostPeriod&&t.total&&t.total.textContent&&!costPricingFailed))return;const n=++usageCostRequest;
setUsageCostStatus("\u96C6\u8A08\u4E2D\u2026"),fetch(`/api/usage/costs?period=${encodeURIComponent(usageCostPeriod)}`,
{credentials:"same-origin",headers:{Accept:"application/json"}}).then(i=>{if(!i.ok)throw new Error("\
usage_unavailable");return i.json()}).then(i=>{n===usageCostRequest&&(usageCostLoadedPeriod=usageCostPeriod,
renderUsageCostSummary(i))}).catch(()=>{n===usageCostRequest&&setUsageCostStatus("\u96C6\u8A08\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002")})}l(loadUsageCostSummary,"loadUsageCostSumm\
ary");function refreshUsageCostSummary(){usageCostLoadedPeriod=null,loadUsageCostSummary(usageCostPeriod)}
l(refreshUsageCostSummary,"refreshUsageCostSummary");function selectUsageCostPeriod(e){if(usageCostPeriod===
e&&usageCostLoadedPeriod===e){markUsageCostPeriod(e);return}usageCostLoadedPeriod=null,loadUsageCostSummary(
e)}l(selectUsageCostPeriod,"selectUsageCostPeriod");const MATH_STEPS_HEADING_RE=/^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*(\d+)\s*(.*)$/i,
MATH_STEPS_DEFAULT_CONFIG=Object.freeze({basics:!1,level:"easy",step_size:"normal",diagrams:"auto",animation:!0}),
MATH_STEPS_PROGRESS_ROUTE="/api/math_steps/progress",MATH_STEPS_SIMPLIFY_ROUTE="/api/math_steps/simp\
lify",MATH_STEPS_SVG_MAX_CHARS=3e5,MATH_STEPS_MAX_EXPANDED_PER_STEP=8,MATH_STEPS_MAX_DISPLAY_STEPS=50,
mathStepsUiState={master:!1,enabled:!1,config:Object.assign({},MATH_STEPS_DEFAULT_CONFIG),loaded:!1,
modalBusy:!1},mathStepsRootState=new WeakMap,mathStepsTextByRoot=new WeakMap,mathStepsProgressTimers=new WeakMap,
mathStepsProgressInFlight=new WeakSet;function clampMathStepsIndex(e,t){const n=Math.floor(Number(e));
if(!Number.isFinite(n)||n<0)return 0;const i=Math.max(0,Number(t)||0);return Math.min(n,i)}l(clampMathStepsIndex,
"clampMathStepsIndex");function normalizeMathStepsConfig(e){let t=e;if(typeof t=="string")try{t=JSON.
parse(t)}catch{t=null}const n=t&&typeof t=="object"?t:{},i=["easy","normal","detailed"].includes(String(
n.level||""))?String(n.level):MATH_STEPS_DEFAULT_CONFIG.level,a=["small","normal","large"].includes(
String(n.step_size||""))?String(n.step_size):MATH_STEPS_DEFAULT_CONFIG.step_size,o=["auto","off"].includes(
String(n.diagrams||""))?String(n.diagrams):MATH_STEPS_DEFAULT_CONFIG.diagrams;return{basics:n.basics===
!0,level:i,step_size:a,diagrams:o,animation:n.animation!==!1}}l(normalizeMathStepsConfig,"normalizeM\
athStepsConfig");function normalizeMathStepsExpanded(e){const t={};return!e||typeof e!="object"||Array.
isArray(e)||Object.keys(e).forEach(n=>{if(!/^\d{1,4}$/.test(String(n)))return;const i=e[n];if(!Array.
isArray(i))return;const a=[];i.slice(0,MATH_STEPS_MAX_EXPANDED_PER_STEP).forEach(o=>{if(!o||typeof o!=
"object")return;const r=typeof o.body=="string"?o.body:"";if(!r.trim())return;const c=typeof o.title==
"string"?o.title:"";a.push({title:c.trim(),body:r})}),a.length&&(t[String(n)]=a)}),t}l(normalizeMathStepsExpanded,
"normalizeMathStepsExpanded");function normalizeMathStepsMessageState(e){let t=e;if(typeof t=="strin\
g")try{t=JSON.parse(t)}catch{t=null}const n=t&&typeof t=="object"&&!Array.isArray(t)?t:{},i={};return n.
simplified&&typeof n.simplified=="object"&&!Array.isArray(n.simplified)&&Object.keys(n.simplified).forEach(
a=>{const o=n.simplified[a];typeof o=="string"&&o.trim()&&(i[String(a)]=o)}),{current:clampMathStepsIndex(
n.current,50),max_reached:clampMathStepsIndex(n.max_reached,50),simplified:i,expanded:normalizeMathStepsExpanded(
n.expanded)}}l(normalizeMathStepsMessageState,"normalizeMathStepsMessageState");function buildMathStepsDisplaySteps(e,t){
const n=[],i=Array.isArray(e)?e:[],a=t&&typeof t=="object"?t:{};return i.forEach(o=>{const r=a[String(
o.index)];if(Array.isArray(r)&&r.length){r.forEach((c,d)=>{n.push({index:n.length,sourceIndex:o.index,
partIndex:d,expanded:!0,title:String(c.title||o.title||""),body:String(c.body||"")})});return}n.push(
{index:n.length,sourceIndex:o.index,partIndex:null,expanded:!1,title:o.title||"",body:o.body||""})}),
n.length>MATH_STEPS_MAX_DISPLAY_STEPS&&(n.length=MATH_STEPS_MAX_DISPLAY_STEPS),n}l(buildMathStepsDisplaySteps,
"buildMathStepsDisplaySteps");function parseMathSteps(e){const t=String(e||"").replace(/\r\n?/g,`
`),n=[],i=[];let a=null,o=null;const r=t.split(`
`);for(const m of r){if(o){const g=m.trim();g.length>=o.length&&g[0]===o[0]&&/^(`+|~+)$/.test(g)&&(o=
null)}else{const g=m.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);if(g&&!(g[1][0]==="`"&&g[2].includes("`"))){
o=g[1],(a?a.body:i).push(m);continue}const y=m.match(MATH_STEPS_HEADING_RE);if(y){a&&n.push(a),a={index:n.
length,number:Number(y[1])||n.length+1,title:String(y[2]||"").trim(),body:[]};continue}}(a?a.body:i).
push(m)}a&&n.push(a);const c=n.map((m,g)=>({index:g,number:m.number,title:m.title,body:m.body.join(`\

`).replace(/^\s*\n+/,"").replace(/\n+\s*$/,"")})),d=i.join(`
`).trim();return c.length&&d&&(c[0].body=`${d}

${c[0].body}`.trim()),{steps:c,preamble:d,hasSteps:c.length>0}}l(parseMathSteps,"parseMathSteps"),window.
parseMathSteps=parseMathSteps;function mathStepsReducedMotion(){try{return!!(window.matchMedia&&window.
matchMedia("(prefers-reduced-motion: reduce)").matches)}catch{return!1}}l(mathStepsReducedMotion,"ma\
thStepsReducedMotion");function renderMathStepsBodyHtml(e){const t=String(e||"").trim();return t?sanitizeMarkdownHtml(
t):'<div class="math-steps-empty">\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u306E\u672C\u6587\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>'}
l(renderMathStepsBodyHtml,"renderMathStepsBodyHtml");function sanitizeMathSvgMarkup(e){const t=String(
e||"").trim();if(!t||t.length>MATH_STEPS_SVG_MAX_CHARS||!/^<svg[\s>]/i.test(t.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i,
"")))return"";let n=null;try{n=new DOMParser().parseFromString(t,"image/svg+xml")}catch{return""}if(!n||
!n.documentElement||n.querySelector("parsererror"))return"";const i=n.documentElement;if(String(i.nodeName||
"").toLowerCase()!=="svg")return"";const a=mathStepsReducedMotion()||mathStepsUiState.config.animation===
!1,o=new Set(["script","foreignobject","iframe","object","embed","image","video","audio"]),r=new Set(
["animate","animatetransform","animatemotion","set","discard"]),c=l(m=>{!m||!m.childNodes||Array.from(
m.childNodes).forEach(g=>{if(g.nodeType===1){const y=String(g.nodeName||"").toLowerCase();if(o.has(y)||
a&&r.has(y)){g.remove();return}if(y==="style"){const b=g.textContent||"";if(/@import|url\(\s*["']?\s*(?:https?:)?\/\//i.
test(b)){g.remove();return}if(a&&/@keyframes|animation/i.test(b)){g.remove();return}}Array.from(g.attributes||
[]).forEach(b=>{const w=String(b.name||"").toLowerCase(),x=String(b.value||"");if(w.startsWith("on")){
g.removeAttribute(b.name);return}(w==="href"||w==="xlink:href")&&!/^#/.test(x.trim())&&g.removeAttribute(
b.name),a&&/animation(-name|-duration|-iteration-count)?$/.test(w)&&g.removeAttribute(b.name),w==="s\
tyle"&&/url\(\s*["']?\s*(?:https?:)?\/\//i.test(x)&&g.removeAttribute(b.name)}),c(g)}})},"walk");c(i);
let d="";try{d=new XMLSerializer().serializeToString(i)}catch{return""}return/^<svg[\s>]/i.test(d)?d:
""}l(sanitizeMathSvgMarkup,"sanitizeMathSvgMarkup");function buildMathSvgRenderHtml(e){const t=sanitizeMathSvgMarkup(
e);if(!t)return"";let n=t.replace(/<svg\b([^>]*)>/i,(o,r)=>{let c=r;return/\sxmlns\s*=/i.test(c)||(c=
` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(t)&&!/\sxmlns:xlink\s*=/i.test(c)&&(c=` \
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`),`<svg${c}>`});const i=`data:image/svg+xml;charset=u\
tf-8,${encodeURIComponent(n)}`,a=hashString(t);return`<span class="mathsvg-render" data-svg-key="${escapeHtml(
a)}"><img src="${i}" alt="\u56F3" loading="lazy" decoding="async"></span>`}l(buildMathSvgRenderHtml,
"buildMathSvgRenderHtml"),window.buildMathSvgRenderHtml=buildMathSvgRenderHtml;function mathStepsStateAttr(e){
try{return escapeHtml(JSON.stringify(e||{}))}catch{return""}}l(mathStepsStateAttr,"mathStepsStateAtt\
r");function mathStepsStepMarkdown(e,t){const n=e.steps[t];if(!n)return"";if(n.expanded)return n.body;
const i=e.simplified?e.simplified[String(n.sourceIndex)]:"",a=e.showOriginal&&e.showOriginal.has(t);
return i&&!a?i:n.body}l(mathStepsStepMarkdown,"mathStepsStepMarkdown");function mathStepsCanRestore(e,t){
const n=e.steps[t];return n?n.expanded?!0:!!((e.simplified?e.simplified[String(n.sourceIndex)]:"")&&
!e.showOriginal.has(t)):!1}l(mathStepsCanRestore,"mathStepsCanRestore");function mathStepsTotalLabel(e){
const t=e.steps.length;return`${e.current+1} / ${t}${e.streaming?"+":""}`}l(mathStepsTotalLabel,"mat\
hStepsTotalLabel");function mathStepsPendingNext(e){return!!e.streaming&&e.current>=e.steps.length-1}
l(mathStepsPendingNext,"mathStepsPendingNext");function buildMathStepsHtml(e,t={}){const n=parseMathSteps(
e);if(!n.steps.length)return"";const i=normalizeMathStepsMessageState(t.state||{}),a=n.steps,o=buildMathStepsDisplaySteps(
a,i.expanded),r={originalSteps:a,expanded:Object.assign({},i.expanded),steps:o,current:clampMathStepsIndex(
t.current!==void 0?t.current:i.current,o.length-1),maxReached:Math.max(i.max_reached,i.current),simplified:Object.
assign({},i.simplified),showOriginal:new Set,streaming:!!t.streaming},c=r.steps[r.current],d=mathStepsStepMarkdown(
r,r.current),m=mathStepsCanRestore(r,r.current),g=mathStepsPendingNext(r),y=r.current>=r.steps.length-
1,b=y&&!g?"\u5B8C\u4E86":g?"\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u6E96\u5099\u4E2D\u2026":"\u308F\u304B\u3063\
\u305F\u3001\u6B21\u3078",w=g||y&&!g,x=t.messageId!==void 0&&t.messageId!==null?String(t.messageId):
"",k=t.jobId!==void 0&&t.jobId!==null?String(t.jobId):"";return`<div class="math-steps content-area"\
 role="group" aria-label="\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u89E3\u8AAC" ${['data-math-steps-root\
="1"',`data-current="${r.current}"`,`data-max-reached="${r.maxReached}"`,`data-total="${r.steps.length}\
"`,`data-streaming="${r.streaming?"1":"0"}"`,x?`data-message-id="${escapeHtml(x)}"`:"",k?`data-job-i\
d="${escapeHtml(k)}"`:"",`data-math-steps-state="${mathStepsStateAttr({current:i.current,max_reached:i.
max_reached,simplified:i.simplified,expanded:i.expanded})}"`].filter(Boolean).join(" ")}><div class=\
"math-steps-header"><span class="math-steps-counter">\u30B9\u30C6\u30C3\u30D7 ${mathStepsTotalLabel(
r)}</span><span class="math-steps-title">${escapeHtml(c.title||"")}</span></div><div class="math-ste\
ps-body" data-step-index="${r.current}" aria-live="polite"><div class="prose prose-invert text-sm br\
eak-words math-steps-prose">${renderMathStepsBodyHtml(d)}</div>`+(m?'<button type="button" class="ma\
th-steps-restore" data-math-steps-action="restore">\u5143\u306E\u8AAC\u660E\u306B\u623B\u3059</button>':
"")+`</div><div class="math-steps-actions"><button type="button" class="math-steps-next" data-math-s\
teps-action="next"${w?' disabled aria-disabled="true"':""}>${escapeHtml(b)}</button><div class="math\
-steps-sub-actions"><button type="button" class="math-steps-prev" data-math-steps-action="prev"${r.current<=
0?' disabled aria-disabled="true"':""}>\u524D\u3078</button><button type="button" class="math-steps-restart" d\
ata-math-steps-action="restart"${r.current<=0?' disabled aria-disabled="true"':""}>\u6700\u521D\u304B\u3089</button><but\
ton type="button" class="math-steps-simplify" data-math-steps-action="simplify"${r.streaming?' disab\
led aria-disabled="true"':""} title="\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u3001\u3084\u3055\u3057\u3044\u8907\u6570\u306E\u30B9\u30C6\u30C3\u30D7\u306B\u5206\u3051\u307E\u3059"><i class="fas fa-wand-magic-sparkles"\
 aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F</button></div></div><div class="math-steps-status" role="status" ari\
a-live="polite"></div></div>`}l(buildMathStepsHtml,"buildMathStepsHtml"),window.buildMathStepsHtml=buildMathStepsHtml;
function buildMathStepsBubbleHtml(e,t={}){if(!parseMathSteps(e).steps.length)return"";const i=t.mathSteps||
{};return buildMathStepsHtml(e,{state:i.state,messageId:i.messageId,jobId:i.jobId,streaming:!!i.streaming})}
l(buildMathStepsBubbleHtml,"buildMathStepsBubbleHtml"),window.buildMathStepsBubbleHtml=buildMathStepsBubbleHtml;
function mathStepsRootCounterEl(e){return e?e.querySelector(".math-steps-counter"):null}l(mathStepsRootCounterEl,
"mathStepsRootCounterEl");function mathStepsStatus(e,t){const n=e?e.querySelector(".math-steps-statu\
s"):null;n&&(n.textContent=String(t||""))}l(mathStepsStatus,"mathStepsStatus");function getMathStepsRootState(e){
if(!e)return null;let t=mathStepsRootState.get(e);if(t)return t;const n=e.getAttribute("data-message\
-id")||"";let i=mathStepsTextByRoot.get(e)||"";!i&&n&&typeof messageStore=="object"&&messageStore&&(i=
messageStore[n]||"");const a=parseMathSteps(i);if(!a.steps.length)return null;const o=normalizeMathStepsMessageState(
e.getAttribute("data-math-steps-state")||""),r=a.steps,c=buildMathStepsDisplaySteps(r,o.expanded);return t=
{originalSteps:r,expanded:Object.assign({},o.expanded),steps:c,current:clampMathStepsIndex(e.getAttribute(
"data-current"),c.length-1),maxReached:Math.max(o.max_reached,clampMathStepsIndex(e.getAttribute("da\
ta-max-reached"),c.length-1)),simplified:Object.assign({},o.simplified),showOriginal:new Set,streaming:e.
getAttribute("data-streaming")==="1"},mathStepsRootState.set(e,t),t}l(getMathStepsRootState,"getMath\
StepsRootState");function paintMathStepsRoot(e){const t=getMathStepsRootState(e);if(!e||!t||!t.steps.
length)return;t.current=clampMathStepsIndex(t.current,t.steps.length-1);const n=t.steps[t.current],i=mathStepsRootCounterEl(
e);i&&(i.textContent=`\u30B9\u30C6\u30C3\u30D7 ${mathStepsTotalLabel(t)}`);const a=e.querySelector("\
.math-steps-title");a&&(a.textContent=n.title||"");const o=e.querySelector(".math-steps-body"),r=mathStepsStepMarkdown(
t,t.current),c=mathStepsCanRestore(t,t.current);if(o){const x=new Map;o.querySelectorAll(".mathsvg-r\
ender[data-svg-key]").forEach(k=>{const C=k.getAttribute("data-svg-key");C&&!x.has(C)&&x.set(C,k)}),
o.setAttribute("data-step-index",String(t.current)),o.innerHTML=`<div class="prose prose-invert text\
-sm break-words math-steps-prose">${renderMathStepsBodyHtml(r)}</div>`+(c?'<button type="button" cla\
ss="math-steps-restore" data-math-steps-action="restore">\u5143\u306E\u8AAC\u660E\u306B\u623B\u3059</button>':
""),o.querySelectorAll(".mathsvg-render[data-svg-key]").forEach(k=>{const C=x.get(k.getAttribute("da\
ta-svg-key"));C&&k.replaceWith(C)});try{queueMessageDecorations(o,r)}catch{}}const d=mathStepsPendingNext(
t),m=t.current>=t.steps.length-1,g=e.querySelector(".math-steps-next");if(g){g.textContent=m&&!d?"\u5B8C\u4E86":
d?"\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u6E96\u5099\u4E2D\u2026":"\u308F\u304B\u3063\u305F\u3001\u6B21\u3078";
const x=d||m&&!d;g.disabled=x,g.setAttribute("aria-disabled",x?"true":"false")}const y=e.querySelector(
".math-steps-prev");if(y){const x=t.current<=0;y.disabled=x,y.setAttribute("aria-disabled",x?"true":
"false")}const b=e.querySelector(".math-steps-restart");if(b){const x=t.current<=0;b.disabled=x,b.setAttribute(
"aria-disabled",x?"true":"false")}const w=e.querySelector(".math-steps-simplify");w&&!w.dataset.busy&&
(w.disabled=!!t.streaming,w.setAttribute("aria-disabled",t.streaming?"true":"false")),e.setAttribute(
"data-current",String(t.current)),e.setAttribute("data-max-reached",String(t.maxReached)),e.setAttribute(
"data-total",String(t.steps.length)),e.setAttribute("data-streaming",t.streaming?"1":"0"),t.renderedStep=
t.current,t.renderedMarkdown=r}l(paintMathStepsRoot,"paintMathStepsRoot");function renderMathStepsInto(e,t,n={}){
if(!e)return!1;const i=parseMathSteps(t);if(!i.steps.length)return!1;const a=e.querySelector(".math-\
steps"),o=a?getMathStepsRootState(a):null,r=normalizeMathStepsMessageState(n.state||{}),c=o||{originalSteps:i.
steps,expanded:Object.assign({},r.expanded),steps:buildMathStepsDisplaySteps(i.steps,r.expanded),current:r.
current,maxReached:r.max_reached,simplified:Object.assign({},r.simplified),showOriginal:new Set,streaming:!!n.
streaming};c.originalSteps=i.steps,c.expanded=c.expanded||{},c.steps=buildMathStepsDisplaySteps(c.originalSteps,
c.expanded),c.streaming=n.streaming===!0,c.current=clampMathStepsIndex(c.current,c.steps.length-1),c.
maxReached=Math.max(c.maxReached,c.current);const d=a&&o&&o.renderedStep===c.current?o.renderedMarkdown:
null,m=mathStepsStepMarkdown(c,c.current),g=a?a.querySelector(".math-steps-body"):null,y=!!g&&d!==null&&
d===m,b=buildMathStepsHtml(t,{state:{current:c.current,max_reached:c.maxReached,simplified:c.simplified,
expanded:c.expanded},messageId:n.mathSteps&&n.mathSteps.messageId!==void 0?n.mathSteps.messageId:void 0,
jobId:n.mathSteps&&n.mathSteps.jobId!==void 0?n.mathSteps.jobId:void 0,streaming:c.streaming});if(!b)
return!1;e.innerHTML=b;const w=e.querySelector(".math-steps");if(!w)return!1;mathStepsRootState.set(
w,c),mathStepsTextByRoot.set(w,String(t||"")),c.renderedStep=c.current,c.renderedMarkdown=m;const x=w.
querySelector(".math-steps-body");if(y&&g&&x){const k=new Map;g.querySelectorAll(".mathsvg-render[da\
ta-svg-key]").forEach(C=>{const T=C.getAttribute("data-svg-key");T&&!k.has(T)&&k.set(T,C)}),x.querySelectorAll(
".mathsvg-render[data-svg-key]").forEach(C=>{const T=k.get(C.getAttribute("data-svg-key"));T&&C.replaceWith(
T)}),x.replaceWith(g)}else if(x)try{queueMessageDecorations(x,m)}catch{}return!0}l(renderMathStepsInto,
"renderMathStepsInto"),window.renderMathStepsInto=renderMathStepsInto;function mathStepsRootTargets(e){
if(!e)return null;const t=e.getAttribute("data-message-id"),n=e.getAttribute("data-job-id");return t&&
/^\d+$/.test(String(t))?{message_id:Number(t)}:n?{job_id:String(n)}:null}l(mathStepsRootTargets,"mat\
hStepsRootTargets");function postMathStepsProgress(e,t=0){const n=getMathStepsRootState(e),i=mathStepsRootTargets(
e);if(!n||!i||mathStepsProgressInFlight.has(e))return;const a=Object.assign({},i,{current:n.current,
max_reached:n.maxReached});return mathStepsProgressInFlight.add(e),apiFetch(MATH_STEPS_PROGRESS_ROUTE,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(a)}).then(o=>{if(!(o&&
o.ok))throw new Error(`progress ${o?o.status:"failed"}`)}).catch(()=>{t<4&&window.setTimeout(()=>postMathStepsProgress(
e,t+1),1200*(t+1))}).finally(()=>{mathStepsProgressInFlight.delete(e)})}l(postMathStepsProgress,"pos\
tMathStepsProgress");function bindMathStepsMessageId(e,t){if(!(!e||t===void 0||t===null))return e.setAttribute(
"data-message-id",String(t)),postMathStepsProgress(e)}l(bindMathStepsMessageId,"bindMathStepsMessage\
Id"),window.bindMathStepsMessageId=bindMathStepsMessageId;function scheduleMathStepsProgress(e){if(!e)
return;const t=mathStepsProgressTimers.get(e);t&&window.clearTimeout(t),mathStepsProgressTimers.set(
e,window.setTimeout(()=>{mathStepsProgressTimers.delete(e),postMathStepsProgress(e)},450))}l(scheduleMathStepsProgress,
"scheduleMathStepsProgress");function forgetMathStepsVariants(e,t){const n=mathStepsRootTargets(e),i=getMathStepsRootState(
e);!n||!i||apiFetch(MATH_STEPS_PROGRESS_ROUTE,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify(Object.assign({},n,{current:i.current,max_reached:i.maxReached,remove_simplified:[
t],remove_expanded:[t]}))}).catch(()=>{})}l(forgetMathStepsVariants,"forgetMathStepsVariants");function requestMathStepsSimplify(e,t){
const n=getMathStepsRootState(e),i=mathStepsRootTargets(e);if(!n||!i)return;if(!i.message_id){showToast(
"\u56DE\u7B54\u304C\u4FDD\u5B58\u3055\u308C\u305F\u5F8C\u306B\u300C\u3082\u3063\u3068\u3084\u3055\u3057\u304F\u300D\u3092\u4F7F\u3048\u307E\u3059",
"info",!1);return}if(n.streaming){showToast("\u56DE\u7B54\u306E\u751F\u6210\u304C\u7D42\u308F\u3063\u3066\u304B\u3089\u4F7F\u3048\u307E\u3059",
"info",!1);return}const a=n.steps[n.current];if(!a)return;const o={message_id:i.message_id,step_index:a.
sourceIndex};a.partIndex!==null&&a.partIndex!==void 0&&(o.part_index=a.partIndex),t&&(t.dataset.busy=
"1",t.disabled=!0,t.innerHTML='<i class="fas fa-circle-notch fa-spin" aria-hidden="true"></i>\u3084\u3055\u3057\u304F\u3057\u3066\u3044\
\u307E\u3059\u2026'),mathStepsStatus(e,"\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u3001\u3084\u3055\u3057\u3044\u8907\u6570\u306E\u30B9\u30C6\u30C3\u30D7\u306B\u5206\u3051\u3066\u3044\u307E\u3059\u2026"),
apiFetch(MATH_STEPS_SIMPLIFY_ROUTE,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(o)}).then(async r=>{const c=await r.json().catch(()=>({}));let d=c&&Array.isArray(c.steps)?
c.steps:[];if(!d.length&&c&&typeof c.text=="string"&&c.text.trim()&&(d=[{title:a.title,body:c.text}]),
!r.ok||!c||c.status!=="ok"||!d.length){const y=r.status===409?"\u3053\u308C\u4EE5\u4E0A\u30B9\u30C6\u30C3\u30D7\u3092\u5897\u3084\u305B\u307E\u305B\u3093":
c&&c.error?c.error:"\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F";
throw new Error(y)}const m=normalizeMathStepsExpanded({[String(a.sourceIndex)]:d})[String(a.sourceIndex)];
if(!m||!m.length)throw new Error("\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
const g=normalizeMathStepsMessageState(c.math_steps_state||{});n.simplified=Object.assign({},g.simplified),
n.expanded=Object.assign({},n.expanded,g.expanded,{[String(a.sourceIndex)]:m}),n.steps=buildMathStepsDisplaySteps(
n.originalSteps,n.expanded),n.current=clampMathStepsIndex(n.current,n.steps.length-1),n.showOriginal.
delete(n.current),t&&(delete t.dataset.busy,t.disabled=!1,t.innerHTML='<i class="fas fa-wand-magic-s\
parkles" aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F'),mathStepsStatus(e,`\u3084\u3055\u3057\u3044\
\u8AAC\u660E\u306B\u5206\u3051\u307E\u3057\u305F\uFF08${m.length}\u30B9\u30C6\u30C3\u30D7\uFF09`),paintMathStepsRoot(
e)}).catch(r=>{mathStepsStatus(e,""),showToast(r&&r.message?r.message:"\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),t&&(delete t.dataset.busy,t.disabled=!1,t.innerHTML='<i class="fas fa-wand-magic-sparkle\
s" aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F')})}l(requestMathStepsSimplify,
"requestMathStepsSimplify");function handleMathStepsAction(e){const t=e.target&&e.target.closest?e.target.
closest("[data-math-steps-action]"):null;if(!t)return;const n=t.closest(".math-steps");if(!n)return;
const i=getMathStepsRootState(n);if(!i)return;const a=t.getAttribute("data-math-steps-action");if(a===
"simplify"){e.preventDefault(),requestMathStepsSimplify(n,t);return}if(e.preventDefault(),a==="resto\
re"){const o=i.steps[i.current];if(!o)return;if(o.expanded){const r=i.steps.findIndex(c=>c.sourceIndex===
o.sourceIndex);delete i.expanded[String(o.sourceIndex)],delete i.simplified[String(o.sourceIndex)],i.
steps=buildMathStepsDisplaySteps(i.originalSteps,i.expanded),i.current=clampMathStepsIndex(r>=0?r:0,
i.steps.length-1),i.maxReached=clampMathStepsIndex(i.maxReached,i.steps.length-1),i.showOriginal.delete(
i.current)}else i.showOriginal.add(i.current),delete i.simplified[String(o.sourceIndex)];forgetMathStepsVariants(
n,o.sourceIndex),mathStepsStatus(n,"\u5143\u306E\u8AAC\u660E\u306B\u623B\u3057\u307E\u3057\u305F"),scheduleMathStepsProgress(
n)}else if(a==="next"){if(i.current>=i.steps.length-1)return;i.current=clampMathStepsIndex(i.current+
1,i.steps.length-1),i.maxReached=Math.max(i.maxReached,i.current),i.showOriginal.delete(i.current),mathStepsStatus(
n,`\u30B9\u30C6\u30C3\u30D7 ${i.current+1} \u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059`),scheduleMathStepsProgress(
n)}else if(a==="prev"){if(i.current<=0)return;i.current=clampMathStepsIndex(i.current-1,i.steps.length-
1),i.showOriginal.delete(i.current),mathStepsStatus(n,`\u30B9\u30C6\u30C3\u30D7 ${i.current+1} \u306B\u623B\u308A\u307E\u3057\
\u305F`),scheduleMathStepsProgress(n)}else if(a==="restart")i.current=0,i.showOriginal.clear(),mathStepsStatus(
n,"\u6700\u521D\u306E\u30B9\u30C6\u30C3\u30D7\u306B\u623B\u308A\u307E\u3057\u305F"),scheduleMathStepsProgress(
n);else return;paintMathStepsRoot(n)}l(handleMathStepsAction,"handleMathStepsAction");function mathStepsChip(){
return get("enable-math-steps")}l(mathStepsChip,"mathStepsChip");function setMathStepsEnabled(e,t={}){
const n=!!e&&mathStepsUiState.master;mathStepsUiState.enabled=n;const i=mathStepsChip();return i&&i.
checked!==n&&(i.checked=n),n}l(setMathStepsEnabled,"setMathStepsEnabled"),window.setMathStepsEnabled=
setMathStepsEnabled;function applyMathStepsMaster(e){mathStepsUiState.master=!!e;const t=get("math-s\
teps-container");t&&t.classList.toggle("hidden",!mathStepsUiState.master),mathStepsUiState.master||setMathStepsEnabled(
!1,{persist:!1})}l(applyMathStepsMaster,"applyMathStepsMaster");function applyMathStepsSettings(e){!e||
typeof e!="object"||(mathStepsUiState.config=normalizeMathStepsConfig(Object.prototype.hasOwnProperty.
call(e,"math_steps_config")?e.math_steps_config:mathStepsUiState.config),Object.prototype.hasOwnProperty.
call(e,"math_steps_enabled")&&applyMathStepsMaster(e.math_steps_enabled===!0),mathStepsUiState.loaded=
!0)}l(applyMathStepsSettings,"applyMathStepsSettings"),window.applyMathStepsSettings=applyMathStepsSettings;
function applyMathStepsChatDefaults(e,t){if(!e||typeof e!="object")return;if(applyMathStepsSettings(
e),!mathStepsUiState.master){setMathStepsEnabled(!1,{persist:!1});return}const n=Object.prototype.hasOwnProperty.
call(e,"last_math_steps")?e.last_math_steps:void 0,i=Object.prototype.hasOwnProperty.call(e,"default\
_math_steps")?e.default_math_steps:!1;if(t&&n!==void 0&&n!==null){setMathStepsEnabled(n===!0,{persist:!1});
return}setMathStepsEnabled(i===!0,{persist:!1})}l(applyMathStepsChatDefaults,"applyMathStepsChatDefa\
ults"),window.applyMathStepsChatDefaults=applyMathStepsChatDefaults;function collectMathStepsSendOption(){
return!!(mathStepsUiState.master&&mathStepsUiState.enabled)}l(collectMathStepsSendOption,"collectMat\
hStepsSendOption"),window.collectMathStepsSendOption=collectMathStepsSendOption;function openMathStepsModal(){
const e=get("math-steps-config-modal");if(!e)return;const t=normalizeMathStepsConfig(mathStepsUiState.
config);get("math-steps-cfg-basics")&&(get("math-steps-cfg-basics").checked=t.basics),get("math-step\
s-cfg-level")&&(get("math-steps-cfg-level").value=t.level),get("math-steps-cfg-step-size")&&(get("ma\
th-steps-cfg-step-size").value=t.step_size),get("math-steps-cfg-diagrams")&&(get("math-steps-cfg-dia\
grams").value=t.diagrams),get("math-steps-cfg-animation")&&(get("math-steps-cfg-animation").checked=
t.animation);const n=get("math-steps-config-status");if(n&&(n.textContent=""),typeof showModal=="fun\
ction"?showModal("math-steps-config-modal"):e.classList.remove("hidden"),location.pathname!=="/math-\
steps")try{history.pushState({modal:"math-steps"},"","/math-steps")}catch{}}l(openMathStepsModal,"op\
enMathStepsModal"),window.openMathStepsModal=openMathStepsModal;function closeMathStepsModal(e=!1){const t=get(
"math-steps-config-modal");if(t&&(typeof hideModal=="function"?hideModal("math-steps-config-modal"):
t.classList.add("hidden")),!e&&location.pathname==="/math-steps")try{history.back()}catch{}}l(closeMathStepsModal,
"closeMathStepsModal"),window.closeMathStepsModal=closeMathStepsModal;async function saveMathStepsConfig(){
if(mathStepsUiState.modalBusy)return;const e=normalizeMathStepsConfig({basics:!!(get("math-steps-cfg\
-basics")&&get("math-steps-cfg-basics").checked),level:get("math-steps-cfg-level")?get("math-steps-c\
fg-level").value:void 0,step_size:get("math-steps-cfg-step-size")?get("math-steps-cfg-step-size").value:
void 0,diagrams:get("math-steps-cfg-diagrams")?get("math-steps-cfg-diagrams").value:void 0,animation:!!(get(
"math-steps-cfg-animation")&&get("math-steps-cfg-animation").checked)});mathStepsUiState.modalBusy=!0;
const t=get("math-steps-config-save");t&&(t.disabled=!0);const n=get("math-steps-config-status");n&&
(n.textContent="\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u2026");try{if(!(await apiFetch(CHAT_CONFIG.
urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
math_steps_config:e})})).ok)throw new Error("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F");
mathStepsUiState.config=e,typeof userSettingsSnapshot=="object"&&userSettingsSnapshot&&(userSettingsSnapshot.
math_steps_config=e);try{browserFastBootstrap=null}catch{}n&&(n.textContent="\u4FDD\u5B58\u3057\u307E\u3057\u305F"),
showToast("\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u306E\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1),closeMathStepsModal()}catch(i){n&&(n.textContent=""),showToast(i&&i.message?i.message:
"\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}finally{
mathStepsUiState.modalBusy=!1,t&&(t.disabled=!1)}}l(saveMathStepsConfig,"saveMathStepsConfig"),window.
saveMathStepsConfig=saveMathStepsConfig;function buildMathStepsPdfHtml(e,t){const n=parseMathSteps(e);
if(!n.steps.length)return"";const i=normalizeMathStepsMessageState(t);return buildMathStepsDisplaySteps(
n.steps,i.expanded).map(o=>{const r=o.expanded?"":i.simplified[String(o.sourceIndex)],c=o.expanded?o.
body:r&&r.trim()?r:o.body;return`<div class="math-steps-pdf-step"><h3 class="math-steps-pdf-heading"\
>\u30B9\u30C6\u30C3\u30D7 ${o.index+1}${o.title?` ${pdfEscapeAttr(o.title)}`:""}</h3><div class="mat\
h-steps-pdf-body">${sanitizeMarkdownHtml(c)}</div>`+(o.expanded||r&&r.trim()?'<div class="math-steps\
-pdf-note">\uFF08\u3084\u3055\u3057\u3044\u8AAC\u660E\uFF09</div>':"")+"</div>"}).join("")}l(buildMathStepsPdfHtml,
"buildMathStepsPdfHtml"),window.buildMathStepsPdfHtml=buildMathStepsPdfHtml;function prepareMathStepsCopyText(e){
const t=String(e||"");if(!t)return t;const n=l(o=>String(o||"").replace(/(`{3,}|~{3,})mathsvg[^\n]*\n[\s\S]*?\n\1[ \t]*(?=\n|$)/gi,
`\uFF3B\u56F3\uFF3D
`),"withoutMathSvg"),i=parseMathSteps(t);if(!i.steps.length)return n(t);const a=[];return i.steps.forEach(
o=>{const r=o.title?` ${o.title}`:"";a.push(`## \u30B9\u30C6\u30C3\u30D7 ${o.index+1}${r}`),a.push(""),
a.push(n(o.body).trim()),a.push("")}),a.join(`
`).replace(/\n{3,}/g,`

`).trim()}l(prepareMathStepsCopyText,"prepareMathStepsCopyText"),window.prepareMathStepsCopyText=prepareMathStepsCopyText;
function initMathSteps(){const e=mathStepsChip();e&&e.addEventListener("change",()=>{const t=!!e.checked;
if(t){const n=[];if(typeof canvasModeEnabled!="undefined"&&canvasModeEnabled){try{syncCanvasModeUi(!1)}catch{}
n.push("Canvas")}if(typeof codingModeEnabled!="undefined"&&codingModeEnabled){try{syncCodingModeUi(!1)}catch{}
n.push("Coding")}n.length&&showToast(`${n.join("\u3068")}\u3092\u30AA\u30D5\u306B\u3057\u307E\u3057\u305F`,
"info",!1)}setMathStepsEnabled(t,{persist:!0})}),["enable-canvas-mode","enable-coding-mode"].forEach(
t=>{const n=get(t);n&&n.addEventListener("change",()=>{!n.checked||!mathStepsUiState.enabled||(setMathStepsEnabled(
!1,{persist:!1}),showToast("\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u3092\u30AA\u30D5\u306B\u3057\u307E\u3057\u305F",
"info",!1))})}),document.addEventListener("click",t=>{try{handleMathStepsAction(t)}catch(n){console.
error(n)}}),ensureUserSettingsSnapshot().then(t=>{if(!t||(applyMathStepsSettings(t),!mathStepsUiState.
master))return;if(!!t.use_last_chat_settings){const i=Object.prototype.hasOwnProperty.call(t,"last_m\
ath_steps")?t.last_math_steps:null;if(i!=null){setMathStepsEnabled(i===!0,{persist:!1});return}}setMathStepsEnabled(
t.default_math_steps===!0,{persist:!1})}).catch(()=>{}),document.addEventListener("keydown",t=>{var n;
t.key==="Escape"&&((n=get("math-steps-config-modal"))!=null&&n.classList.contains("modal-open"))&&closeMathStepsModal()})}
l(initMathSteps,"initMathSteps"),window.initMathSteps=initMathSteps,document.addEventListener("DOMCo\
ntentLoaded",()=>{try{initMathSteps()}catch(e){console.error("math steps init failed",e)}},{once:!0}),
l((function(){const t={projects:[],activeProjectUuid:null,activeProjectName:"",viewingProjectUuid:null,
viewingProjectDetail:null,editingProjectUuid:null,movingThreadId:null,projectThreadsPage:1,projectThreadsHasNext:!1,
projectThreadsQuery:"",searchTimer:null},n=l(L=>document.getElementById(L),"byId");function i(L,$){return String(
L||"").replace("/0","/"+encodeURIComponent(String($||"")))}l(i,"projectUrl");function a(L,$){return(window.
CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.handleProjectFileItem||"/api/projects/\
0/files/0").replace("/projects/0/","/projects/"+encodeURIComponent(String(L||""))+"/").replace(/\/files\/0$/,
"/files/"+encodeURIComponent(String($||"")))}l(a,"projectFileItemUrl");function o(L){const $=Number(
L||0);return $<1024?`${$} B`:$<1024*1024?`${($/1024).toFixed(1)} KB`:`${($/(1024*1024)).toFixed(1)} \
MB`}l(o,"formatBytes");function r(L){return typeof escapeHtml=="function"?escapeHtml(L==null?"":String(
L)):String(L==null?"":L).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,
"&quot;").replace(/'/g,"&#39;")}l(r,"esc");function c(){const L=[];try{typeof MODELS!="undefined"&&Array.
isArray(MODELS)&&MODELS.forEach($=>{($.items||[]).forEach(M=>{const P=String(M.id||""),R=P.toLowerCase();
P&&(/(image|video|tts|audio|live|embedding|ocr|transcribe|lyria|veo|sora|ideogram|dall-e|imagen|search)/.
test(R)||(R.startsWith("gemini-")||R.startsWith("gpt-")||R.startsWith("o1-")||R.startsWith("o3-")||R.
startsWith("claude-")||R.startsWith("grok-")||R.startsWith("glm-"))&&L.push({id:P,name:M.name||P,group:$.
group||""}))})})}catch{}return L}l(c,"listVisionCapableModels");function d(L,$,M){if(!L)return;const P=String(
$||"").trim(),R=M?`\u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u65E2\u5B9A\u5024\u3092\u4F7F\u3046 (${M}\
)`:"\u30E6\u30FC\u30B6\u30FC\u8A2D\u5B9A\u306E\u65E2\u5B9A\u5024\u3092\u4F7F\u3046",D=c();let H=`<op\
tion value="">${r(R)}</option>`,ee=!P;D.forEach(O=>{O.id===P&&(ee=!0),H+=`<option value="${r(O.id)}"\
>${r(O.name)}</option>`}),P&&!ee&&(H+=`<option value="${r(P)}">${r(P)}</option>`),L.innerHTML=H,L.value=
P}l(d,"populateVisionModelSelect");function m(){try{const L=n("set-default-vision-model");if(L&&L.value)
return L.value;if(typeof currentVisionModel=="string"&&currentVisionModel)return currentVisionModel}catch{}
return"gemini-3-flash-preview"}l(m,"getDefaultVisionModelFromSettings");function g(){const L=n("proj\
ect-chat-banner"),$=n("project-chat-banner-name"),M=n("project-chat-banner-meta");if(!L)return;const P=n(
"project-view"),R=!!(P&&!P.classList.contains("hidden"));if(!t.activeProjectUuid||R){L.classList.add(
"hidden");return}const D=t.projects.find(ee=>ee.uuid===t.activeProjectUuid),H=t.activeProjectName||D&&
D.name||"Project";if($&&($.textContent=H),M){const ee=D?Number(D.file_count||0):0;M.textContent=ee>0?
`\u5171\u6709\u30D5\u30A1\u30A4\u30EB ${ee}\u4EF6`:""}L.classList.remove("hidden")}l(g,"updateProjec\
tChatBanner");function y(){const L=n("project-list");if(L){if(!t.projects.length){L.innerHTML='<div \
class="px-2 py-1 text-[11px] text-gray-500">\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306A\u3057</div>';
return}L.innerHTML="",t.projects.forEach($=>{const M=t.viewingProjectUuid===$.uuid||t.activeProjectUuid===
$.uuid,P=document.createElement("div");P.className=`px-2 py-1.5 rounded hover:bg-gray-700 cursor-poi\
nter text-xs text-gray-300 flex items-center justify-between gap-1 group ${M?"bg-gray-700/60 border-\
l-2 border-amber-400":""}`,P.dataset.projectUuid=$.uuid;const R=Number($.thread_count||0);P.innerHTML=
`
                        <div class="flex items-center gap-1.5 min-w-0 flex-1">
                            <i class="fas fa-folder text-amber-300/90 text-[11px] shrink-0"></i>
                            <span class="truncate">${r($.name)}</span>
                        </div>
                        <div class="flex items-center gap-1 shrink-0">
                            <span class="text-[10px] text-gray-500">${R}</span>
                            <button type="button" class="text-gray-500 hover:text-white px-1 opacity\
-100 md:opacity-0 md:group-hover:opacity-100 transition" data-project-edit="${r($.uuid)}" title="\u30D7\u30ED\u30B8\
\u30A7\u30AF\u30C8\u3092\u7DE8\u96C6">
                                <i class="fas fa-pen text-[10px]"></i>
                            </button>
                        </div>
                    `,P.onclick=D=>{if(D.target&&D.target.closest?D.target.closest("[data-project-ed\
it]"):null){D.stopPropagation(),window.openProjectModal($.uuid);return}window.openProjectView($.uuid)},
L.appendChild(P)})}}l(y,"renderSidebarProjects");async function b(){try{const L=window.CHAT_CONFIG&&
window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.handleProjects||"/api/projects",$=await apiFetch(L);
if(!$.ok)return t.projects;const M=await $.json();Array.isArray(M)&&(t.projects=M,y(),g())}catch(L){
console.error("Failed to load projects:",L)}return t.projects}l(b,"loadProjects");function w(L){const $=n(
"project-view-file-list"),M=n("project-view-file-count");if(!$)return;const P=L&&Array.isArray(L.files)?
L.files:[];if(M&&(M.textContent=`(${P.length})`),!P.length){$.innerHTML='<div class="text-xs text-gr\
ay-500 py-3 text-center">\u5171\u6709\u30D5\u30A1\u30A4\u30EB\u306F\u307E\u3060\u3042\u308A\u307E\u305B\u3093\u3002\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u307E\u305F\u306F\u30E9\u30A4\u30D6\u30E9\u30EA\u304B\u3089\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002</div>';
return}$.innerHTML="",P.forEach(R=>{const D=document.createElement("div");D.className="flex items-ce\
nter justify-between gap-2 rounded-lg border border-gray-700/70 bg-gray-900/70 px-2.5 py-2 text-xs";
const H=R.is_image?"fa-image text-purple-300":"fa-file-lines text-blue-300",ee=R.note?`<div class="t\
ext-[10px] text-gray-400 truncate">${r(R.note)}</div>`:"";D.innerHTML=`
                        <div class="flex items-center gap-2 min-w-0 flex-1">
                            <i class="fas ${H} shrink-0"></i>
                            <div class="min-w-0 flex-1">
                                <div class="text-gray-200 truncate" title="${r(R.filename)}">${r(R.filename)}\
</div>
                                ${ee}
                                <div class="text-[10px] text-gray-500">${r(o(R.size))}${R.is_image?"\
 \xB7 \u753B\u50CF":""}</div>
                            </div>
                        </div>
                        <div class="flex items-center gap-1 shrink-0">
                            <button type="button" class="text-gray-400 hover:text-blue-300 p-1" data\
-pf-note="${R.id}" title="\u30E1\u30E2\u3092\u7DE8\u96C6"><i class="fas fa-note-sticky text-[11px]"></i></button>
                            <button type="button" class="text-gray-400 hover:text-red-400 p-1" data-\
pf-remove="${R.id}" title="\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u304B\u3089\u5916\u3059"><i class="fas fa-times text-[11px]"></i></button>
                        </div>
                    `,D.onclick=async O=>{if(O.target&&O.target.closest?O.target.closest("[data-pf-n\
ote]"):null){O.stopPropagation();const F=prompt("\u3053\u306E\u30D5\u30A1\u30A4\u30EB\u306E\u77ED\u3044\u8AAC\u660E\u30FB\u30E1\u30E2\uFF08\u30E2\u30C7\u30EB\u304C\u53C2\u7167\u5224\u65AD\u306B\u4F7F\u3044\u307E\u3059\uFF09:",
R.note||"");if(F===null)return;try{(await apiFetch(a(L.uuid,R.id),{method:"PUT",headers:{"Content-Ty\
pe":"application/json"},body:JSON.stringify({note:F})})).ok?(await C(),showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u30E1\u30E2\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F",
"success")):showToast("\u30E1\u30E2\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}catch{showToast("\u30E1\u30E2\u306E\u66F4\u65B0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}return}if(O.target&&O.target.closest?O.target.closest("[data-pf-remove]"):null){if(O.stopPropagation(),
!confirm(`\u300C${R.filename}\u300D\u3092\u3053\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u304B\u3089\u5916\u3057\u307E\u3059\u304B\uFF1F
\uFF08\u30D5\u30A1\u30A4\u30EB\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u5B9F\u4F53\u306F\u6B8B\u308A\u307E\u3059\uFF09`))
return;try{(await apiFetch(a(L.uuid,R.id),{method:"DELETE"})).ok?(await C(),await b(),showToast("\u30D7\u30ED\u30B8\
\u30A7\u30AF\u30C8\u304B\u3089\u30D5\u30A1\u30A4\u30EB\u3092\u5916\u3057\u307E\u3057\u305F","success")):
showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F","er\
ror",!0)}catch{showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u89E3\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}},$.appendChild(D)})}l(w,"renderProjectFiles");function x(L){const $=n("project-view-vi\
sion-toggle"),M=n("project-view-vision-model-wrap"),P=n("project-view-vision-model-select"),R=n("pro\
ject-view-vision-hint");if(!$)return;const D=!!(L&&L.use_vision_for_images);$.checked=D,M&&M.classList.
toggle("hidden",!D),P&&d(P,L?L.vision_model:"",m()),R&&(R.textContent=D?`ON: \u5171\u6709\u753B\u50CF\u304C\u53C2\u7167\u3055\u308C\u305F\u3068\u304D ${L.
effective_vision_model||m()} \u3067\u30C6\u30AD\u30B9\u30C8\u89E3\u6790\u3057\u307E\u3059\u3002`:"OF\
F\u306E\u3068\u304D\u306F\u5171\u6709\u753B\u50CF\u306E\u4E2D\u8EAB\u3092\u8AAD\u307E\u305A\u3001\u30E1\u30BF\u60C5\u5831\u3068\u6848\u5185\u306E\u307F\u8FD4\u3057\u307E\u3059\u3002")}
l(x,"renderProjectVisionBar");async function k(L,$=!1){var H,ee;const M=n("project-view-thread-list"),
P=n("project-view-thread-more-btn");if(!M||!L)return;$||(t.projectThreadsPage=1,t.projectThreadsHasNext=
!1);const D=`${i(((ee=(H=window.CHAT_CONFIG)==null?void 0:H.urls)==null?void 0:ee.handleProjectThreads)||
"/api/projects/0/threads",L)}?q=${encodeURIComponent(t.projectThreadsQuery||"")}&page=${t.projectThreadsPage}`;
try{const O=await apiFetch(D);if(!O.ok)return;const B=await O.json(),J=Array.isArray(B.threads)?B.threads:
[];$||(M.innerHTML=""),!$&&!J.length?M.innerHTML='<div class="text-xs text-gray-500 py-4 text-center\
">\u3053\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306B\u306F\u307E\u3060\u30C1\u30E3\u30C3\u30C8\u304C\u3042\u308A\u307E\u305B\u3093\u3002\u300C\u65B0\u898F\u30C1\u30E3\u30C3\u30C8\u300D\u304B\u3089\u59CB\u3081\u3089\u308C\u307E\u3059\u3002</div>':
J.forEach(F=>{const q=String(F.id),te=document.createElement("div"),V=F.is_bookmarked?"text-yellow-4\
00":"text-gray-500",X=F.is_temporary?'<span class="text-[9px] text-amber-300 border border-amber-500\
/50 rounded px-1">\u4E00\u6642</span>':"";te.className="p-2 rounded-lg hover:bg-gray-700/70 cursor-p\
ointer text-xs text-gray-200 flex items-center justify-between gap-2 group border border-transparent\
 hover:border-gray-700",te.innerHTML=`
                                <div class="flex items-center gap-1.5 min-w-0 flex-1">
                                    <button type="button" class="${V} hover:text-yellow-400 px-1" da\
ta-pt-star="${r(q)}"><i class="fas fa-star text-[10px]"></i></button>
                                    <span class="truncate">${r(F.title||"New Chat")}</span>
                                    ${X}
                                </div>
                                <div class="flex items-center gap-1 shrink-0 opacity-100 md:opacity-\
0 md:group-hover:opacity-100 transition">
                                    <button type="button" class="text-gray-400 hover:text-amber-300 \
px-1" data-pt-move="${r(q)}" title="\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u79FB\u52D5\u30FB\u89E3\u9664"><i class="fas fa-folder-open text-[11px]"></i></but\
ton>
                                    <button type="button" class="text-gray-400 hover:text-white px-1\
" data-pt-rename="${r(q)}" title="\u540D\u524D\u5909\u66F4"><i class="fas fa-pen text-[11px]"></i></button>
                                    <button type="button" class="text-gray-400 hover:text-red-400 px\
-1" data-pt-delete="${r(q)}" title="\u524A\u9664"><i class="fas fa-trash text-[11px]"></i></button>
                                </div>
                            `,te.onclick=async re=>{var xe,ve,we,Oe,Ce,lt,fe,ye;if((ve=(xe=re.target)==
null?void 0:xe.closest)==null?void 0:ve.call(xe,"[data-pt-star]")){re.stopPropagation(),typeof toggleBookmark==
"function"&&await toggleBookmark(re,q),await k(L,!1);return}if((Oe=(we=re.target)==null?void 0:we.closest)==
null?void 0:Oe.call(we,"[data-pt-move]")){re.stopPropagation(),window.openMoveThreadProjectModal(re,
q,F.title||"New Chat",L);return}if((lt=(Ce=re.target)==null?void 0:Ce.closest)==null?void 0:lt.call(
Ce,"[data-pt-rename]")){re.stopPropagation(),typeof renameThread=="function"&&await renameThread(re,
q),setTimeout(()=>k(L,!1),300);return}if((ye=(fe=re.target)==null?void 0:fe.closest)==null?void 0:ye.
call(fe,"[data-pt-delete]")){re.stopPropagation(),typeof deleteThread=="function"&&await deleteThread(
re,q),setTimeout(()=>{k(L,!1),b()},300);return}window.hideProjectViewForChat(),typeof loadMessages==
"function"&&loadMessages(q)},M.appendChild(te)}),t.projectThreadsHasNext=!!B.has_next,t.projectThreadsHasNext&&
t.projectThreadsPage++,P&&P.classList.toggle("hidden",!t.projectThreadsHasNext)}catch(O){console.error(
"Failed to load project threads:",O)}}l(k,"loadProjectThreads");async function C(){var D,H;if(!t.viewingProjectUuid)
return null;const L=i(((H=(D=window.CHAT_CONFIG)==null?void 0:D.urls)==null?void 0:H.handleProjectItem)||
"/api/projects/0",t.viewingProjectUuid),$=await apiFetch(L);if(!$.ok)return null;const M=await $.json();
t.viewingProjectDetail=M,t.activeProjectUuid=M.uuid,t.activeProjectName=M.name,n("project-view-name")&&
(n("project-view-name").textContent=M.name||"");const P=n("project-view-desc");P&&(P.textContent=M.description||
"",P.classList.toggle("hidden",!M.description));const R=n("project-view-instruction-preview");return R&&
(R.textContent=M.instruction&&String(M.instruction).trim()?M.instruction:"\uFF08\u672A\u8A2D\u5B9A\uFF09"),
x(M),w(M),M}l(C,"refreshViewingProject"),window.openProjectView=async(L,$={})=>{if(!L)return;t.viewingProjectUuid=
String(L),t.activeProjectUuid=String(L);const M=t.projects.find(R=>R.uuid===t.viewingProjectUuid);M&&
(t.activeProjectName=M.name);const P=n("project-view");if(P&&P.classList.remove("hidden"),n("welcome\
-screen")&&n("welcome-screen").classList.add("hidden"),g(),y(),!$.skipHistory){const R="/p/"+encodeURIComponent(
t.viewingProjectUuid);location.pathname!==R&&history.pushState({projectUuid:t.viewingProjectUuid},"",
R)}if(window.innerWidth<768&&n("overlay"))try{n("overlay").click()}catch{}t.projectThreadsQuery="",n(
"project-view-thread-search")&&(n("project-view-thread-search").value=""),await C(),await k(t.viewingProjectUuid,
!1)},window.hideProjectViewForChat=()=>{const L=n("project-view");L&&L.classList.add("hidden"),t.viewingProjectUuid=
null,g(),y()},window.closeProjectView=(L={})=>{const $=n("project-view");$&&$.classList.add("hidden"),
t.viewingProjectUuid=null,t.viewingProjectDetail=null,L.keepContext||(t.activeProjectUuid=null,t.activeProjectName=
""),g(),y(),!L.skipHistory&&location.pathname.startsWith("/p/")&&history.pushState({},"","/"),!L.keepWelcome&&
!currentThreadId&&n("welcome-screen")&&n("welcome-screen").classList.remove("hidden")},window.setActiveProjectContext=
(L,$)=>{t.activeProjectUuid=L?String(L):null,t.activeProjectName=$?String($):"",g(),y()},window.getActiveProjectId=
()=>t.activeProjectUuid||null,window.openProjectModal=async(L=null)=>{var J,F;t.editingProjectUuid=L?
String(L):null;const $=n("project-modal-title"),M=n("project-name-input"),P=n("project-desc-input"),
R=n("project-inst-input"),D=n("project-use-vision-toggle"),H=n("project-vision-model-wrap"),ee=n("pr\
oject-vision-model-select");let O=null;if(t.editingProjectUuid&&(O=t.viewingProjectDetail&&t.viewingProjectDetail.
uuid===t.editingProjectUuid?t.viewingProjectDetail:t.projects.find(q=>q.uuid===t.editingProjectUuid),
!O))try{const q=await apiFetch(i(((F=(J=window.CHAT_CONFIG)==null?void 0:J.urls)==null?void 0:F.handleProjectItem)||
"/api/projects/0",t.editingProjectUuid));q.ok&&(O=await q.json())}catch{}$&&($.textContent=O?"Edit P\
roject":"Create New Project"),M&&(M.value=O&&O.name||""),P&&(P.value=O&&O.description||""),R&&(R.value=
O&&O.instruction||"");const B=!!(O&&O.use_vision_for_images);D&&(D.checked=B),H&&H.classList.toggle(
"hidden",!B),ee&&d(ee,O?O.vision_model:"",m()),showModal("project-modal"),M&&setTimeout(()=>M.focus(),
50)},window.closeProjectModal=(L=!1)=>{hideModal("project-modal"),t.editingProjectUuid=null,!L&&location.
pathname==="/project"&&history.back()},window.openMoveThreadProjectModal=async(L,$,M,P=null)=>{L&&typeof L.
stopPropagation=="function"&&L.stopPropagation(),t.movingThreadId=String($||""),t.projects.length||await b();
const R=n("move-thread-project-chat-name");R&&(R.textContent=M||"\u30C1\u30E3\u30C3\u30C8");const D=n(
"move-thread-project-select");if(D){let H='<option value="">\uFF08\u672A\u6240\u5C5E / \u901A\u5E38\u306E\u5C65\u6B74\u3078\u623B\u3059\uFF09</option>';
t.projects.forEach(ee=>{H+=`<option value="${r(ee.uuid)}">${r(ee.name)}</option>`}),D.innerHTML=H,D.
value=P?String(P):""}showModal("move-thread-project-modal")},window.closeMoveThreadProjectModal=()=>{
hideModal("move-thread-project-modal"),t.movingThreadId=null},window.syncThreadProjectSettingsModal=
L=>{const $=n("thread-project-select"),M=n("thread-project-instruction-controls"),P=n("thread-includ\
e-project-instruction"),R=n("thread-project-instruction-preview"),D=n("thread-open-project-btn");if(!$)
return;let H='<option value="">\uFF08\u672A\u6240\u5C5E\uFF09</option>';t.projects.forEach(B=>{H+=`<\
option value="${r(B.uuid)}">${r(B.name)}</option>`});const ee=L&&L.project_id?String(L.project_id):"";
ee&&!t.projects.some(B=>B.uuid===ee)&&(H+=`<option value="${r(ee)}">${r(L.project_name||ee)}</option\
>`),$.innerHTML=H,$.value=ee,P&&(P.checked=L?L.include_project_instruction!==!1:!0);const O=l(()=>{const B=$.
value||"";if(M&&M.classList.toggle("hidden",!B),R)if(B&&L&&B===String(L.project_id||""))R.value=L.project_instruction||
"";else if(B){const J=t.projects.find(F=>F.uuid===B);R.value=J&&J.instruction||""}else R.value="";D&&
(D.onclick=()=>{$.value&&(typeof window.closeThreadModal=="function"&&window.closeThreadModal(!0),window.
openProjectView($.value))})},"updateControls");$.onchange=O,O()};async function T(){var R,D;if(!t.viewingProjectUuid)
return;const L=n("project-view-vision-toggle"),$=n("project-view-vision-model-select"),M=!!(L&&L.checked),
P=$&&$.value||"";try{const H=i(((D=(R=window.CHAT_CONFIG)==null?void 0:R.urls)==null?void 0:D.handleProjectItem)||
"/api/projects/0",t.viewingProjectUuid),ee=await apiFetch(H,{method:"PUT",headers:{"Content-Type":"a\
pplication/json"},body:JSON.stringify({use_vision_for_images:M,vision_model:P})});if(ee.ok){const O=await ee.
json();t.viewingProjectDetail=O,x(O),await b(),showToast("\u753B\u50CF\u306EVision\u30E2\u30C7\u30EB\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success")}else showToast("Vision\u30E2\u30C7\u30EB\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}catch{showToast("Vision\u30E2\u30C7\u30EB\u8A2D\u5B9A\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}l(T,"saveProjectVisionInline");async function A(L){var M,P;if(!t.viewingProjectUuid||!Array.
isArray(L)||!L.length)return;const $=i(((P=(M=window.CHAT_CONFIG)==null?void 0:M.urls)==null?void 0:
P.handleProjectFiles)||"/api/projects/0/files",t.viewingProjectUuid);try{const R=await apiFetch($,{method:"\
POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filepaths:L})}),D=await R.json().
catch(()=>({}));if(!R.ok){showToast(D.error==="project_file_limit_exceeded"?"\u5171\u6709\u30D5\u30A1\u30A4\u30EB\u306E\u4E0A\u9650\uFF0850\u4EF6\uFF09\u306B\u9054\u3057\u3066\u3044\u307E\u3059":
"\u30D5\u30A1\u30A4\u30EB\u306E\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const H=Array.isArray(D.added)?D.added.length:0;await C(),await b(),H>0?showToast(`\u5171\u6709\u30D5\u30A1\u30A4\u30EB\u3092 ${H}\
 \u4EF6\u8FFD\u52A0\u3057\u307E\u3057\u305F`,"success"):showToast("\u9078\u629E\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u306F\u65E2\u306B\u8FFD\u52A0\u6E08\u307F\u3067\u3059",
"info")}catch{showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u8FFD\u52A0\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}l(A,"attachFilepathsToViewingProject");async function N(L){if(!t.viewingProjectUuid||!L||
!L.length)return;const $=window.CHAT_CONFIG&&window.CHAT_CONFIG.urls&&window.CHAT_CONFIG.urls.upload||
"/upload",M=[];for(const P of Array.from(L)){const R=new FormData;R.append("file",P);try{const D=await apiFetch(
$,{method:"POST",body:R}),H=await D.json().catch(()=>({})),ee=H.filepath||H.path||H.url;D.ok&&ee?M.push(
ee):showToast(`\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u5931\u6557: ${P.name}`,"error",!0)}catch{showToast(
`\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u5931\u6557: ${P.name}`,"error",!0)}}M.length&&await A(M)}l(N,
"uploadFilesForViewingProject");function G(){n("add-project-btn")&&(n("add-project-btn").onclick=()=>window.
openProjectModal(null)),n("project-use-vision-toggle")&&(n("project-use-vision-toggle").onchange=()=>{
const L=n("project-vision-model-wrap");L&&L.classList.toggle("hidden",!n("project-use-vision-toggle").
checked)}),n("save-project-btn")&&(n("save-project-btn").onclick=async()=>{var ee,O,B,J,F,q,te,V,X;const L=(((ee=
n("project-name-input"))==null?void 0:ee.value)||"").trim(),$=(((O=n("project-desc-input"))==null?void 0:
O.value)||"").trim(),M=((B=n("project-inst-input"))==null?void 0:B.value)||"",P=!!((J=n("project-use\
-vision-toggle"))!=null&&J.checked),R=((F=n("project-vision-model-select"))==null?void 0:F.value)||"";
if(!L){showToast("\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u540D\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}const D=!!t.editingProjectUuid,H=D?i(((te=(q=window.CHAT_CONFIG)==null?void 0:q.urls)==
null?void 0:te.handleProjectItem)||"/api/projects/0",t.editingProjectUuid):((X=(V=window.CHAT_CONFIG)==
null?void 0:V.urls)==null?void 0:X.handleProjects)||"/api/projects";try{const re=await apiFetch(H,{method:D?
"PUT":"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:L,description:$,instruction:M,
use_vision_for_images:P,vision_model:R})}),ce=await re.json().catch(()=>({}));if(!re.ok){showToast(ce.
error==="name_taken"?"\u540C\u3058\u540D\u524D\u306E\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u304C\u65E2\u306B\u3042\u308A\u307E\u3059":
"\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","\
error",!0);return}window.closeProjectModal(!0),await b(),D&&t.viewingProjectUuid===ce.uuid?await C():
!D&&ce.uuid&&await window.openProjectView(ce.uuid),showToast(D?"\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F":
"\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u3092\u4F5C\u6210\u3057\u307E\u3057\u305F","success")}catch{showToast(
"\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","\
error",!0)}}),n("save-move-thread-project-btn")&&(n("save-move-thread-project-btn").onclick=async()=>{
var $;if(!t.movingThreadId)return;const L=(($=n("move-thread-project-select"))==null?void 0:$.value)||
null;try{const M=await apiFetch(`/api/threads/${encodeURIComponent(t.movingThreadId)}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({project_id:L})});if(!M.ok){showToast(
"\u30C1\u30E3\u30C3\u30C8\u306E\u79FB\u52D5\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0);return}
const P=await M.json().catch(()=>({}));String(currentThreadId||"")===t.movingThreadId&&window.setActiveProjectContext(
P.project_id||null,P.project_name||""),window.closeMoveThreadProjectModal(),await b(),typeof loadThreads==
"function"&&loadThreads(),t.viewingProjectUuid&&await k(t.viewingProjectUuid,!1),showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u6240\u5C5E\u30D7\
\u30ED\u30B8\u30A7\u30AF\u30C8\u3092\u66F4\u65B0\u3057\u307E\u3057\u305F","success")}catch{showToast(
"\u30C1\u30E3\u30C3\u30C8\u306E\u79FB\u52D5\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}),
n("project-chat-banner-open")&&(n("project-chat-banner-open").onclick=()=>{t.activeProjectUuid&&window.
openProjectView(t.activeProjectUuid)}),n("project-view-back-btn")&&(n("project-view-back-btn").onclick=
()=>{window.closeProjectView(),typeof startNewChat=="function"&&startNewChat()}),n("project-view-new\
-chat-btn")&&(n("project-view-new-chat-btn").onclick=()=>{const L=t.viewingProjectUuid,$=t.viewingProjectDetail&&
t.viewingProjectDetail.name||t.activeProjectName;window.hideProjectViewForChat(),typeof startNewChat==
"function"&&startNewChat({preserveProject:!0,projectId:L,projectName:$}),window.setActiveProjectContext(
L,$);const M=n("prompt-input");M&&M.focus()}),n("project-view-edit-btn")&&(n("project-view-edit-btn").
onclick=()=>{t.viewingProjectUuid&&window.openProjectModal(t.viewingProjectUuid)}),n("project-view-e\
dit-inst-btn")&&(n("project-view-edit-inst-btn").onclick=()=>{t.viewingProjectUuid&&window.openProjectModal(
t.viewingProjectUuid)}),n("project-view-delete-btn")&&(n("project-view-delete-btn").onclick=async()=>{
var $,M;const L=t.viewingProjectDetail;if(L&&confirm(`\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u300C${L.
name}\u300D\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F
\u6240\u5C5E\u3057\u3066\u3044\u305F\u30C1\u30E3\u30C3\u30C8\u306F\u524A\u9664\u3055\u308C\u305A\u3001\u901A\u5E38\u306E\u5C65\u6B74\uFF08\u672A\u6240\u5C5E\uFF09\u3078\u623B\u308A\u307E\u3059\u3002`))
try{const P=i(((M=($=window.CHAT_CONFIG)==null?void 0:$.urls)==null?void 0:M.handleProjectItem)||"/a\
pi/projects/0",L.uuid);if(!(await apiFetch(P,{method:"DELETE"})).ok){showToast("\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306E\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}window.closeProjectView(),await b(),typeof loadThreads=="function"&&loadThreads(),
showToast("\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u3092\u524A\u9664\u3057\u307E\u3057\u305F\uFF08\u6240\u5C5E\u30C1\u30E3\u30C3\u30C8\u306F\u672A\u6240\u5C5E\u3078\u79FB\u52D5\u3057\u307E\u3057\u305F\uFF09",
"success")}catch{showToast("\u30D7\u30ED\u30B8\u30A7\u30AF\u30C8\u306E\u524A\u9664\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}),n("project-view-vision-toggle")&&(n("project-view-vision-toggle").onchange=()=>{const L=n(
"project-view-vision-model-wrap");L&&L.classList.toggle("hidden",!n("project-view-vision-toggle").checked),
T()}),n("project-view-vision-model-select")&&(n("project-view-vision-model-select").onchange=()=>T()),
n("project-view-upload-file-btn")&&n("project-view-file-input")&&(n("project-view-upload-file-btn").
onclick=()=>{const L=n("project-view-file-input");L&&(L.value="",L.click())},n("project-view-file-in\
put").onchange=L=>{const $=L.target&&L.target.files;$&&$.length&&N($)}),n("project-view-lib-file-btn")&&
(n("project-view-lib-file-btn").onclick=()=>{t.viewingProjectUuid&&(window.__libProjectAttachCallback=
async L=>{window.__libProjectAttachCallback=null,typeof window.closeLibModal=="function"&&window.closeLibModal(
!0),await A(L)},typeof openLibModal=="function"&&openLibModal(!0))}),n("project-view-thread-search")&&
(n("project-view-thread-search").oninput=L=>{t.projectThreadsQuery=(L.target&&L.target.value||"").trim(),
t.searchTimer&&clearTimeout(t.searchTimer),t.searchTimer=setTimeout(()=>{t.viewingProjectUuid&&k(t.viewingProjectUuid,
!1)},250)}),n("project-view-thread-more-btn")&&(n("project-view-thread-more-btn").onclick=()=>{t.viewingProjectUuid&&
t.projectThreadsHasNext&&k(t.viewingProjectUuid,!0)})}l(G,"bindProjectsUi"),window.loadProjects=b,window.
refreshViewingProjectIfOpen=()=>{t.viewingProjectUuid&&(C(),k(t.viewingProjectUuid,!1))},document.addEventListener(
"DOMContentLoaded",()=>{try{G(),b().then(()=>{const L=window.CHAT_CONFIG&&window.CHAT_CONFIG.initialProjectId;
L&&window.openProjectView(L,{skipHistory:!0})})}catch(L){console.error("projects init failed",L)}},{
once:!0})}),"initProjectsModule")();
