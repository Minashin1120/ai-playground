function buildSvgCodeRenderHtml(e,t){let n=!0;const i=String(e||"").replace(/<svg\b([^>]*)>/i,(o,l)=>{
let c=l;/\sxmlns\s*=/i.test(c)||(c=` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(e)&&!/\sxmlns:xlink\s*=/i.
test(c)&&(c=` xmlns:xlink="http://www.w3.org/1999/xlink"${c}`);const d=c.match(/\swidth\s*=\s*["']?\s*([\d.]+)\s*(px)?\s*(?:["'\s/]|$)/i);
return d&&Number(d[1])>0&&(n=!1),`<svg${c}>`}),a=`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
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
e||""),n=t.split(/\r?\n/),i=[],a=[],o=/^(\s*)(`{3,}|~{3,})(.*)$/;let l=null,c="",d=[];for(const b of n){
if(!l){const x=b.match(o);if(x){l=x[2],c=String(x[3]||"").trim(),d=[],i.push({lang:c,code:"",open:!0}),
a.push('<div class="canvas-code-placeholder">Canvas\u3067\u8868\u793A\u4E2D</div>');continue}a.push(
b);continue}const v=String(b||"").trim();if(v&&v.replace(/\s+/g,"")===l){const x=i[i.length-1];x&&(x.
code=d.join(`
`),x.open=!1),l=null,c="",d=[];continue}d.push(b);const w=i[i.length-1];w&&(w.code=d.join(`
`))}if(l&&i.length){const b=i[i.length-1];b&&(b.code=d.join(`
`),b.open=!0)}const m=i.map((b,v)=>normalizeCanvasBlock(b,v)),g=selectCanvasPreviewBlock(m,t);return{
renderText:a.join(`
`),blocks:m,primaryBlock:g?g.block:null,primaryIndex:g?g.index:-1,rawText:t}}r(parseCanvasMarkdown,"\
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
forEach(d=>{d&&d.classList.remove("canvas-view-enter-from-left","canvas-view-enter-from-right")}),o.
offsetWidth;const c=a[n]<a[t]?"canvas-view-enter-from-left":"canvas-view-enter-from-right";o.classList.
add(c),canvasPreviewState.viewAnimationTimer=setTimeout(()=>{l===canvasPreviewState.viewAnimationToken&&
(o.classList.remove(c),canvasPreviewState.viewAnimationTimer=null)},340)}r(animateCanvasMobileViewEntry,
"animateCanvasMobileViewEntry");function syncCanvasPanelViewUi(e=canvasPreviewState.mobileView,t={}){
var l,c;const n=getCanvasModeElements();if(!n||!n.panel)return;const i=["preview","blocks","source"].
includes(e)?e:"preview",a=["preview","blocks","source"].includes(t.fromView)?t.fromView:canvasPreviewState.
mobileView;canvasPreviewState.mobileView=i,n.panel.dataset.canvasMobileView=i,(n.panelTabs?Array.from(
n.panelTabs.querySelectorAll("[data-canvas-panel-view]")):[]).forEach(d=>{const m=d.getAttribute("da\
ta-canvas-panel-view")===i;d.classList.toggle("active",m),d.setAttribute("aria-pressed",m?"true":"fa\
lse")}),t.animate===!0&&animateCanvasMobileViewEntry(n,a,i),t.focus!==!1&&isCanvasMobileLayout()&&(i===
"preview"&&n.frame&&!n.frame.classList.contains("hidden")?n.frame.focus({preventScroll:!0}):i==="sou\
rce"&&n.sourceScroll?n.sourceScroll.focus({preventScroll:!0}):i==="blocks"&&n.blockList&&((c=(l=n.blockList).
focus)==null||c.call(l,{preventScroll:!0})))}r(syncCanvasPanelViewUi,"syncCanvasPanelViewUi");function renderCanvasBlockChips(){
const e=getCanvasModeElements();if(!e||!e.blockList)return;const t=Array.isArray(canvasPreviewState.
blocks)?canvasPreviewState.blocks:[];if(e.blockCount&&(e.blockCount.textContent=String(t.length)),!t.
length){e.blockList.innerHTML='<div class="px-2 py-3 text-xs text-gray-500">\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D</div>';
return}const n=Number.isInteger(canvasPreviewState.selectedIndex)?canvasPreviewState.selectedIndex:-1;
e.blockList.innerHTML=t.map((i,a)=>{const o=String(i&&i.lang?i.lang:"text").trim()||"text",l=a===n,c=i&&
i.open?"\u751F\u6210\u4E2D":"\u8868\u793A",g=(String(i&&i.code?i.code:"").split(/\r?\n/).find(w=>w.trim())||
"\u7A7A\u306E\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF").trim().replace(/\s+/g," ").slice(0,120),b=`${l?
"\u73FE\u5728\u8868\u793A\u4E2D":"\u5207\u308A\u66FF\u3048"}: ${o}`,v=`${b}\u3001${g}`;return`<butto\
n type="button" class="canvas-block-chip${l?" active":""}" data-canvas-block-index="${a}" title="${escapeHtml(
b)}" aria-label="${escapeHtml(v)}" aria-pressed="${l?"true":"false"}"><span class="canvas-block-chip\
-index">#${a+1}</span><span class="canvas-block-chip-main"><span class="canvas-block-chip-lang">${escapeHtml(
o)}</span><span class="canvas-block-chip-preview">${escapeHtml(g)}</span></span><span class="canvas-\
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
null?i.code:""),d=o?isCanvasHtmlPreviewCandidate(l,c):!1,m=o?d?"HTML \u3092\u30EA\u30A2\u30EB\u30BF\u30A4\u30E0\u3067\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
i&&i.open?"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u751F\u6210\u4E2D":"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u30D7\u30EC\u30D3\u30E5\u30FC\u3057\u3066\u3044\u307E\u3059":
"\u30B3\u30FC\u30C9\u30D6\u30ED\u30C3\u30AF\u3092\u5F85\u6A5F\u4E2D",g=o?d?`HTML Canvas Preview${t.length>
1&&a>=0?` #${a+1}/${t.length}`:""}`:`Canvas Preview: ${l||"text"}${t.length>1&&a>=0?` #${a+1}/${t.length}`:
""}`:"Canvas\u3067\u8868\u793A\u4E2D";e.title&&(e.title.textContent=g),e.status&&(e.status.textContent=
m),e.previewLang&&(e.previewLang.textContent=o?l||"text":"idle");const b=e.sourceScroll?e.sourceScroll.
scrollTop:canvasPreviewState.sourceScrollTop,v=e.sourceScroll?e.sourceScroll.scrollLeft:canvasPreviewState.
sourceScrollLeft;if(e.code&&(e.code.textContent=c),e.sourceScroll&&(e.sourceScroll.scrollTop=b,e.sourceScroll.
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
(c,d)=>{const m=decodeCanvasPreviewButtonCode(c);if(!m)return;const g=normalizeCanvasBlock({lang:m.lang,
code:m.code,open:!1},d);a.push(g),o===-1&&m.codeKey===t.codeKey&&(o=a.length-1)}),!a.length)return null;
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
"`","~"])for(let c=3;c<=10;c++){const d=l.repeat(c);for(const m of o){const g=`${d}${m}
`,b=`
${d}`,v=g+a+b;i.includes(v)&&(i=i.split(v).join(""))}}return i}r(stripExactFencedBlock,"stripExactFe\
ncedBlock");function stripVisiblePythonOutputBlock(e,t){let n=normalizeMarkdownNewlines(e);const i=normalizeMarkdownNewlines(
t==null?"":String(t)),a=[`**Output:**
`,`**Output:** 
`,"**Output:**"];for(const o of a)for(const l of["`","~"])for(let c=3;c<=10;c++){const d=l.repeat(c);
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
`))})}return n}r(stripVisiblePythonOutputBlock,"stripVisiblePythonOutputBlock");function buildChatErrorBubbleHtml(e){
const t=String(e==null?"":e).trim()||"Unknown error";return`<div class="text-red-400 text-xs mt-2 bo\
rder border-red-500 p-2 rounded chat-error-box" role="alert"><i class="fas fa-triangle-exclamation m\
r-1"></i>Error: ${escapeHtml(t)}</div>`}r(buildChatErrorBubbleHtml,"buildChatErrorBubbleHtml");function buildChatErrorMarkdown(e,t=""){
let n=String(e==null?"":e).trim()||"Unknown error";n=n.replace(/```/g,"'''"),n.length>5e4&&(n=n.slice(
0,5e4)+"\u2026");const i="```chat_error\n"+n+"\n```",a=String(t==null?"":t).replace(/\s+$/,"");return a?
a+`

`+i:i}r(buildChatErrorMarkdown,"buildChatErrorMarkdown");function extractPythonExecutionsFromContent(e){
const t=normalizeMarkdownNewlines(e),n=[],i=[];if(!t)return{text:"",executions:n};const a=/(?:^|\n)(`{3,}|~{3,})pyexec[ \t]*\n([\s\S]*?)\n\1[ \t]*(?=\n|$)/g;
let o=t.replace(a,(v,w,x,k)=>{i.push(k);const S=String(x||"").trim();try{const C=JSON.parse(S);n.push(
{code:C&&C.code!=null?String(C.code):"",output:C&&C.output!=null?String(C.output):"",images:C&&Array.
isArray(C.images)?C.images.filter(M=>typeof M=="string"):[]})}catch{n.push({code:S,output:""})}return`\

`});const l=/!\[Agentic View\]\(([^)]+)\)/g;if(!n.length&&/!\[Agentic View\]\(/.test(t)&&(o=t.replace(
/(?:^|\n)```python\n([\s\S]*?)\n```(?:\n+\*\*Output:\*\*\n```\n([\s\S]*?)\n```)?/g,(v,w,x,k)=>(i.push(
k),n.push({code:w,output:x||"",images:[]}),`
`))),n.length){const v=new Set(n.flatMap(w=>w.images||[]));for(const w of t.matchAll(l)){if(v.has(w[1]))
continue;let x=i.length-1;for(;x>0&&i[x]>w.index;)x--;const k=n[x];k.images=Array.from(new Set([...k.
images||[],w[1]]))}o=o.replace(l,"")}n.forEach(v=>{v.code&&(o=stripExactFencedBlock(o,"python",v.code),
o=stripExactFencedBlock(o,"py",v.code)),o=stripVisiblePythonOutputBlock(o,v.output)});const c=new Set(
n.flatMap(v=>v.images||[]));o=o.replace(/(`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`)|(?<!!)\[([^\]\n]*)\]\(([^)\s]+)\)/g,
(v,w,x,k)=>!w&&c.has(k)&&/\.(?:png|jpe?g|webp|gif|bmp)(?:[?#].*)?$/i.test(k)?`![${x}](${k})`:v);const d=n.
slice().reverse().find(v=>v.images&&v.images.length),m=o.replace(/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,
""),g=new Set(Array.from(m.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),v=>v[1])),b=(d?d.images:[]).filter(
v=>/^\/files\/[A-Za-z0-9_./%-]+$/.test(v)&&!g.has(v));return b.length&&(o+=`

**Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF**

`+b.map(v=>`![Python\u306E\u6700\u5F8C\u306E\u5B9F\u884C\u753B\u50CF](${v})`).join(`

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
o=escapeHtml(i)}catch{o=escapeHtml(i)}const l=(e.images||[]).filter(k=>/^\/files\/[A-Za-z0-9_./%-]+$/.
test(k)).map(k=>`![Python output](${k})`).join(`
`),c=l?`<div class="python-section"><div class="python-label">Images</div>${sanitizeMarkdownHtml(marked.
parse(l))}</div>`:"",d=escapeHtml(a),m=encodeURIComponent(i).replace(/'/g,"%27"),g=encodeURIComponent(
a).replace(/'/g,"%27"),b=hashString(`pyexec-detail
${i}
${a}
${t}`),v=n>1?`Python Execution ${t+1}/${n}`:"Python Execution",w=`<button class="download-btn" data-\
code="${m}" data-lang="python" title="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9" aria-label="\u30B3\u30FC\u30C9\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9"><i class="fas fa-download"\
></i></button>`,x=`<button class="coding-target-btn" data-code="${m}" data-code-key="${b}" data-codi\
ng-lang="python" aria-pressed="false" title="Coding Mode\u306E\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A" aria-label="\u7DE8\u96C6\u5BFE\u8C61\u306B\u6307\u5B9A"><i class="fas\
 fa-quote-right"></i></button>`;return`<div class="code-wrapper python-box" data-collapsed="false" d\
ata-code-key="${b}"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i>\
 ${escapeHtml(v)}</span><div class="code-actions">${x}${w}<button class="copy-btn" data-copy="code" \
data-code="${m}" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button cl\
ass="copy-btn" data-copy="output" data-code="${g}" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas \
fa-align-left"></i></button></div></div><div class="code-body"><div class="python-section"><div clas\
s="python-label">Code</div><pre><code class="hljs language-python python-code">${o}</code></pre></di\
v><div class="python-section"><div class="python-label">Output</div><pre><code class="hljs language-\
plaintext python-output">${d}</code></pre></div>${c}</div></div>`}r(buildPythonExecDetailBoxHtml,"bu\
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
closePythonExecDetail,"closePythonExecDetail"),window.closePythonExecDetail=closePythonExecDetail;function buildAiMarkdownHtml(e,t={}){
const n=extractMcpExecutionNotesFromContent(e),i=appendMcpExecutionNotes(n.text,n.notes);if(typeof window.
buildMathStepsBubbleHtml=="function"){const l=window.buildMathStepsBubbleHtml(i,t);if(l)return!lowBandwidthMode&&
maybeNeedsMathJax(i)&&ensureMathJaxLoaded().catch(()=>{}),l}const a=canvasModeEnabled?parseCanvasMarkdown(
i):{renderText:i,blocks:[],primaryBlock:null,rawText:i};canvasModeEnabled&&(updateCanvasPreviewState(
a),refreshCanvasPreviewPanel());const o=document.createElement("div");return o.className="prose pros\
e-invert text-sm break-words",o.innerHTML=sanitizeMarkdownHtml(a.renderText),wrapRenderedSvgBoxes(o),
lowBandwidthMode||(maybeNeedsHighlight(a.renderText,o)&&ensureHighlightLoaded().catch(()=>{}),maybeNeedsMathJax(
a.renderText)&&ensureMathJaxLoaded().catch(()=>{})),o.outerHTML}r(buildAiMarkdownHtml,"buildAiMarkdo\
wnHtml");function renderAiMarkdownInto(e,t,n={}){if(!e)return;const i=extractMcpExecutionNotesFromContent(
t),a=appendMcpExecutionNotes(i.text,i.notes);if(typeof window.renderMathStepsInto=="function"&&window.
renderMathStepsInto(e,a,n)){lowBandwidthMode||maybeNeedsMathJax(a)&&ensureMathJaxLoaded().catch(()=>{});
return}const o=canvasModeEnabled?parseCanvasMarkdown(a):{renderText:a,blocks:[],primaryBlock:null,rawText:a};
if(canvasModeEnabled&&(updateCanvasPreviewState(o),refreshCanvasPreviewPanel()),n.incrementalMath){const c=document.
createElement("template");c.innerHTML=sanitizeMarkdownHtml(o.renderText,{streamMathSegments:!0});const d=new Map;
e.querySelectorAll(".stream-math-segment[data-stream-math-key]").forEach(g=>{const b=g.getAttribute(
"data-stream-math-key");b&&d.set(b,g)});const m=[];c.content.querySelectorAll(".stream-math-segment[\
data-stream-math-key]").forEach(g=>{const b=d.get(g.getAttribute("data-stream-math-key"));b?g.replaceWith(
b):m.push(g)}),restoreSvgCodeRenders(c.content,collectSvgCodeRenders(e)),e.replaceChildren(c.content),
wrapRenderedSvgBoxes(e),queueHighlight(e,o.renderText),queueIncrementalMathTypeset(m);return}const l=collectSvgCodeRenders(
e);e.innerHTML=sanitizeMarkdownHtml(o.renderText),restoreSvgCodeRenders(e,l),wrapRenderedSvgBoxes(e),
queueMessageDecorations(e,o.renderText)}r(renderAiMarkdownInto,"renderAiMarkdownInto");function collectSvgCodeRenders(e){
const t=new Map;return!e||typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-rende\
r[data-svg-key]").forEach(n=>{const i=n.getAttribute("data-svg-key");i&&!t.has(i)&&t.set(i,n)}),t}r(
collectSvgCodeRenders,"collectSvgCodeRenders");function restoreSvgCodeRenders(e,t){!e||!t||!t.size||
typeof e.querySelectorAll!="function"||e.querySelectorAll(".svg-code-render[data-svg-key]").forEach(
n=>{const i=n.getAttribute("data-svg-key"),a=i?t.get(i):null;a&&(t.delete(i),n.replaceWith(a))})}r(restoreSvgCodeRenders,
"restoreSvgCodeRenders");function wrapRenderedSvgBoxes(e){!e||typeof e.querySelectorAll!="function"||
e.querySelectorAll("svg").forEach(t=>{if(!t||!t.parentNode||t.closest(".svg-render-box")||t.closest(
"pre, code, .code-wrapper, .thought-container"))return;const n=document.createElement("span");n.className=
"svg-render-box",t.parentNode.insertBefore(n,t),n.appendChild(t)})}r(wrapRenderedSvgBoxes,"wrapRende\
redSvgBoxes");const SEARCH_COMPLETE_HTML='<div class="search-box visible mb-2"><i class="fas fa-chec\
k-circle text-green-400"></i> Search complete</div>';function updateSearchBox(e,t,n){return n!=="sea\
rching"&&n!=="done"||(t||(e.insertAdjacentHTML("afterbegin",SEARCH_COMPLETE_HTML),t=e.querySelector(
".search-box")),t.classList.toggle("animate-pulse",n==="searching"),t.innerHTML=n==="searching"?'<i \
class="fas fa-globe"></i> Searching web...':'<i class="fas fa-check-circle text-green-400"></i> Sear\
ch complete'),t}r(updateSearchBox,"updateSearchBox");function renderMessage(e,t,n,i,a,o,l=null,c=!0,d=null,m=null,g=null,b=null,v=null,w=null,x=null,k=null,S=!0,C=null,M=null,E=null,q=null,N=null,W=null){
const I=t==="user",j=I?"bg-blue-600":"bg-gray-700",ne=I?"justify-end":"justify-start";messageStore[e]=
n;const oe=!I&&n?extractPythonExecutionsFromContent(n):{text:n||"",executions:[]},te=I?n:oe.text;let ge=m;
if(ge==null){const he=g!=null?Number(g):0,fe=b!=null?Number(b):0;(g!=null||b!=null)&&(ge=he+fe)}const $=costCacheUsage(
a);messageMeta[e]={tokens_in:g,tokens_out:b,tokens_total:ge,tokens_content:w,tokens_thought:x,cache_read_tokens:$?
$.read:null,cache_write_tokens:$?$.write:null,cache_write_ttls:$?$.ttls:null,is_encrypted:v,role:t,model:o,
parent_id:C,quote_text:d,image_url:i,image_mask:q,skill_name:M,batch_job:E,processing_time:N,python_executions:I?
[]:oe.executions||[]};let O="";d&&(O=`<div class="mb-2 p-2 bg-black/20 rounded border-l-4 border-blu\
e-400 text-xs text-gray-300 italic truncate max-w-full"><i class="fas fa-quote-left mr-1 opacity-50"\
></i>${escapeHtml(d)}</div>`);const V=M?`<div class="skill-notice visible"><i class="fas fa-wand-mag\
ic-sparkles"></i><span>Skill: <span class="font-bold">${escapeHtml(M)}</span></span></div>`:"";let B="";
if(a&&!I){let he="";try{const fe=JSON.parse(a);he=fe.text||"",fe.search_used===!0&&(B=SEARCH_COMPLETE_HTML)}catch{
he=a}he&&(B+=`<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content c\
ollapsed">${escapeHtml(he)}</div></div>`)}let G="";const ae=new Set;if(!I){const he=String(te||"").replace(
/`{3,}[\s\S]*?`{3,}|~{3,}[\s\S]*?~{3,}|`[^`\n]*`/g,""),fe=Array.from(he.matchAll(/!\[[^\]\n]*\]\(([^)\s]+)\)/g),
ve=>ve[1]);fe.push(...(oe.executions||[]).flatMap(ve=>ve.images||[])),fe.forEach(ve=>{if(/^https?:\/\//i.
test(ve)&&!ve.startsWith(FILE_BASE_URL))try{if(new URL(ve).origin!==window.location.origin)return}catch{
return}const ee=normalizeAttachmentPath(ve);ee&&ae.add(ee)})}if(i)try{const he=JSON.parse(i);if(he.length){
const fe=[];if(he.forEach(ve=>{let ee=ve,nt="unknown";if(ee&&typeof ee=="object"&&(nt=normalizeAttachmentSource(
ee.source),ee=ee.filepath||ee.path||ee.url||ee.file||""),ee=normalizeAttachmentPath(ee)||ee,!ee||!I&&
ae.has(ee))return;setAttachmentSourceForPath(ee,nt);const Pe=ee.replace(/^\d+\//,""),mt=buildFileUrl(
Pe),Ke=buildAttachmentPreviewUrl(Pe),at=ee.split("/").pop(),lt=at.split(".").pop().toLowerCase();["j\
pg","jpeg","png","webp","gif"].includes(lt)?fe.push(buildChatImageHtml(Ke,{viewerSrc:mt,alt:at,title:at,
filename:at})):fe.push(`<div class="file-thumb bg-gray-800 border border-gray-600 rounded flex flex-\
col items-center justify-center cursor-pointer hover:bg-gray-700" onclick="window.open('${mt}')" tit\
le="${at}"><i class="fas fa-file text-2xl text-gray-400 mb-1"></i><span class="text-[9px] truncate w\
-20 text-center">${at}</span></div>`)}),fe.length>0){let ve="grid-multi";fe.length===1?ve="grid-1":fe.
length===2?ve="grid-2":fe.length===3?ve="grid-3":fe.length===4&&(ve="grid-4"),G=`<div class="image-g\
rid ${ve}">${fe.join("")}</div>`}}}catch{}let R="";if(I&&q)try{const he=normalizeAttachmentPath(q)||
String(q),fe=he.replace(/^\d+\//,""),ve=buildFileUrl(fe),ee=buildAttachmentPreviewUrl(fe),nt=he.split(
"/").pop();R=`<div class="mt-2 max-w-xs rounded-lg border border-amber-300/40 bg-black/10 p-2"><div \
class="mb-1 text-[10px] font-semibold text-amber-100"><i class="fas fa-mask mr-1" aria-hidden="true"\
></i>\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09</div>${buildChatImageHtml(ee,
{viewerSrc:ve,alt:"Object Remover mask",title:"\u30DE\u30B9\u30AF\u753B\u50CF\uFF08\u524A\u9664\u7BC4\u56F2\uFF09",
filename:nt})}</div>`}catch{}const F=I?"":`<button class="ctrl-btn" onclick="regenerateMessage('${e}\
')"><i class="fas fa-rotate-right"></i></button>`,me=`<div class="msg-controls absolute -top-5 right\
-0 hidden group-hover:flex gap-1 z-10"><button class="ctrl-btn" onclick="window.copyMessage('${e}', \
this)"><i class="fas fa-copy"></i></button>${I?`<button class="ctrl-btn edit-btn" data-id="${e}"><i \
class="fas fa-pen"></i></button>`:""}${F}${/^\d+$/.test(String(e))?`<button class="ctrl-btn" title="\
\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" aria-label="\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8\u306B\u5206\u5C90" onclick="branchChatMessage('${e}\
', this)"><i class="fas fa-code-branch"></i></button>`:""}<button class="ctrl-btn" onclick="deleteMe\
ssage('${e}')"><i class="fas fa-trash"></i></button></div>`,re=[];!I&&o&&re.push(escapeHtml(o));const X=[];
if(g!=null&&X.push(`In ${g}`),b!=null){let he=`Out ${b}`;x!=null&&Number(x)>0&&(he+=` (Thought ${x})`),
X.push(he)}if(X.length||m!=null){const he=X.length?X.join(" / "):`${m} tokens`;re.push(`<button clas\
s="underline decoration-dotted hover:text-white token-detail-btn" onclick="openTokenDetail('${e}')">${he}\
</button>`)}if(v!=null){const he=v?"fa-lock":"fa-lock-open",fe=isAdminUser?v?"\u6697\u53F7\u5316\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u5FA9\u53F7\u5316\uFF09":
"\u5E73\u6587\u72B6\u614B\uFF08\u30BF\u30C3\u30D7\u3067\u518D\u6697\u53F7\u5316\uFF09":v?"Encrypted":
"Plain",ve=isAdminUser?v?"text-amber-300/90 hover:text-amber-200":"text-cyan-300/90 hover:text-cyan-\
200":"text-slate-300/80 hover:text-white";re.push(`<button class="${ve}" title="${fe}" onclick="open\
EncryptionSettings('${e}')"><i class="fas ${he}"></i></button>`)}if(!I&&oe.executions&&oe.executions.
length){const he=oe.executions.length,fe=he>1?`Python \xD7${he}`:"Python";re.push(`<button type="but\
ton" class="python-exec-btn" onclick="openPythonExecDetail('${e}')" title="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A" aria-label\
="Python\u5B9F\u884C\u7D50\u679C\u3092\u8868\u793A"><i class="fas fa-terminal"></i><span>${fe}</span\
></button>`)}const ce=re.length?`<div class="text-[10px] text-slate-300/90 mt-2 text-right font-mono\
 message-footer-meta">${re.join(" \u2022 ")}</div>`:"",Q=I?"":buildProcessingTimeHtml(e,N);let ye;const ue=!I&&
E?(()=>{const he=String(E.state||"").toUpperCase(),fe=E.status_text||(he==="JOB_STATE_SUCCEEDED"?"Ba\
tch\u51E6\u7406\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F":he==="JOB_STATE_FAILED"?"Batch\u51E6\u7406\u306B\u5931\u6557\u3057\u307E\u3057\u305F":
"Batch API\u3067\u51E6\u7406\u4E2D\u3067\u3059");return`<div class="batch-status-card mb-3 rounded-l\
g border ${he==="JOB_STATE_FAILED"||he==="JOB_STATE_CANCELLED"||he==="JOB_STATE_EXPIRED"?"border-red\
-400/40 bg-red-950/30 text-red-100":he==="JOB_STATE_SUCCEEDED"?"border-emerald-400/40 bg-emerald-950\
/30 text-emerald-100":"border-violet-400/40 bg-violet-950/30 text-violet-100"} px-3 py-2 text-xs"><d\
iv class="font-semibold"><i class="fas fa-layer-group mr-1"></i>Batch</div><div class="mt-1 opacity-\
90">${escapeHtml(fe)}</div></div>`})():"";I?ye=`<div class="content-area whitespace-pre-wrap font-sa\
ns text-sm break-words">${escapeHtml(n||"")}</div>`:(ye=ue+(te&&String(te).trim()?buildAiMarkdownHtml(
te,{mathSteps:{messageId:e,state:W}}):E?'<div class="content-area prose prose-invert text-sm break-w\
ords text-gray-300">\u56DE\u7B54\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>':buildAiMarkdownHtml(
te,{mathSteps:{messageId:e,state:W}})),ye.includes("content-area")||(ye=ye.replace("prose ","content\
-area prose ")));let ke="";if(l){const he=l.siblings[l.current-2],fe=l.siblings[l.current];ke=`
                    <div class="flex items-center gap-2 text-[10px] text-gray-400 mt-1 select-none">\

                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${he}\
)" ${he?"":"disabled"}><i class="fas fa-chevron-left"></i></button>
                        <span>${l.current} / ${l.total}</span>
                        <button class="hover:text-white disabled:opacity-30" onclick="switchVersion(${fe}\
)" ${fe?"":"disabled"}><i class="fas fa-chevron-right"></i></button>
                    </div>
                `}const Be=c?"fade-in":"",Ue=document.createElement("div");Ue.className=`flex ${ne} \
mb-4 ${Be} relative message-group group`,Ue.id=`msg-${e}`,Ue.innerHTML=`<div class="message-bubble ${j}\
 text-white p-4 rounded-2xl shadow-md relative">${me}${O}${V}${B}${ye}${G}${R}${ke}${ce}${Q}</div>`;
const Je=k||get("chat-container");if(Je&&(Je.appendChild(Ue),S&&scrollToBottom(),!I)){const he=Ue.querySelector(
".message-bubble");if(he&&(he.setAttribute("data-message-id",String(e)),W!=null))try{he.setAttribute(
"data-math-steps-state",typeof W=="string"?W:JSON.stringify(W))}catch{}Ue.isConnected&&queueMessageDecorations(
Ue,te),syncCodingTargetButtons(Ue),syncCodingModeUi(codingModeEnabled,{persist:!1})}return Ue}r(renderMessage,
"renderMessage");const PROCESSING_PHASE_LABELS={send:"\u9001\u4FE1\u301C\u53D7\u4ED8",queue:"\u5F85\u3061\u884C\u5217",
provider:"\u521D\u56DE\u5FDC\u7B54\u307E\u3067",generation:"\u751F\u6210",receive:"\u53D7\u4FE1\u30FB\u8868\u793A"};
function formatProcessingDuration(e){const t=Number(e);if(!Number.isFinite(t)||t<0)return"-";if(t<6e4)
return`${(t/1e3).toFixed(2)}\u79D2`;const n=t/1e3,i=Math.floor(n/60);return`${i}\u5206${(n-i*60).toFixed(
1)}\u79D2`}r(formatProcessingDuration,"formatProcessingDuration");function buildProcessingTimeHtml(e,t){
if(!t||t.total_ms===null||t.total_ms===void 0)return"";const n=formatProcessingDuration(t.total_ms);
return`<div class="mt-1 text-right font-mono text-[10px] text-slate-400/80 processing-time-row"><but\
ton type="button" class="underline decoration-dotted hover:text-white processing-time-btn" onclick="\
openProcessingTimeDetail('${e}')" title="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33" aria-label="\u56DE\u7B54\u51E6\u7406\u6642\u9593\u306E\u5185\u8A33">\u51E6\u7406\u6642\u9593 ${escapeHtml(
n)}</button></div>`}r(buildProcessingTimeHtml,"buildProcessingTimeHtml");function openProcessingTimeDetail(e){
const t=messageMeta[e],n=t&&t.processing_time;if(!n||n.total_ms===null||n.total_ms===void 0||!get("p\
rocessing-time-modal"))return;get("processing-time-total").textContent=formatProcessingDuration(n.total_ms);
const a=get("processing-time-first-row");if(a){const d=n.first_response_ms!==null&&n.first_response_ms!==
void 0;a.classList.toggle("hidden",!d),d&&get("processing-time-first")&&(get("processing-time-first").
textContent=formatProcessingDuration(n.first_response_ms))}const o=get("processing-time-phases");if(o){
o.replaceChildren();const d=Array.isArray(n.phases)?n.phases:[];for(const m of d){if(!m||m.ms===null||
m.ms===void 0)continue;const g=PROCESSING_PHASE_LABELS[m.key]||m.key,b=document.createElement("div");
b.className="flex justify-between gap-4";const v=document.createElement("span");v.textContent=g;const w=document.
createElement("span");w.className="font-mono",w.textContent=formatProcessingDuration(m.ms),b.appendChild(
v),b.appendChild(w),o.appendChild(b)}o.classList.toggle("hidden",o.childElementCount===0)}const l=get(
"processing-time-note");l&&(n.source==="client"?l.textContent="\u56DE\u7B54\u5B8C\u4E86\u6642\u306B\u7AEF\u672B\u3067\u8A18\u9332\u3057\u305F\u6642\u9593\u3067\u3059\u3002\u8A73\u7D30\u306A\u5185\u8A33\u306F\u5C65\u6B74\u306E\u518D\u8AAD\u307F\u8FBC\u307F\u5F8C\u306B\u8868\u793A\u3055\
\u308C\u307E\u3059\u3002":n.phases&&n.phases.length?l.textContent="\u8A18\u9332\u3067\u304D\u305F\u51E6\u7406\u533A\u9593\u306E\u5185\u8A33\u3067\u3059\u3002":
l.textContent="\u3053\u306E\u56DE\u7B54\u3067\u306F\u5408\u8A08\u6642\u9593\u3060\u3051\u3092\u8A18\u9332\u3057\u3066\u3044\u307E\u3059\u3002");
const c=get("processing-time-title");c&&(c.textContent=t&&t.model?`${t.model} (${t.role||"assistant"}\
)`:"\u56DE\u7B54\u51E6\u7406\u6642\u9593"),showModal("processing-time-modal")}r(openProcessingTimeDetail,
"openProcessingTimeDetail");function closeProcessingTimeDetail(){hideModal("processing-time-modal")}
r(closeProcessingTimeDetail,"closeProcessingTimeDetail");function showTokenDetailModal(e=null,t=!1){
if(location.pathname!=="/token-details"){const n={modal:"token-details"};t&&(n.promptEstimate=!0),e!==
null&&(n.messageId=e),history.pushState(n,"","/token-details")}showModal("token-detail-modal")}r(showTokenDetailModal,
"showTokenDetailModal");function openTokenDetail(e){var C,M;const t=messageMeta[e];if(!t||!get("toke\
n-detail-modal"))return;get("token-detail-usage").classList.remove("hidden"),get("token-detail-estim\
ate").classList.add("hidden"),cancelPromptTokenDetail();const i=t.tokens_total!==null&&t.tokens_total!==
void 0?t.tokens_total:"-",a=t.tokens_in!==null&&t.tokens_in!==void 0?t.tokens_in:"-",o=t.tokens_out!==
null&&t.tokens_out!==void 0?t.tokens_out:"-",l=t.tokens_content!==null&&t.tokens_content!==void 0?t.
tokens_content:"-",c=t.tokens_thought!==null&&t.tokens_thought!==void 0?t.tokens_thought:"-",d=t.is_encrypted===
null||t.is_encrypted===void 0?"-":t.is_encrypted?"Encrypted":"Plain",m=t.cache_read_tokens!==null&&t.
cache_read_tokens!==void 0?t.cache_read_tokens:null,g=t.cache_write_tokens!==null&&t.cache_write_tokens!==
void 0?t.cache_write_tokens:null,b=m!==null||g!==null;get("token-detail-total").innerText=i,get("tok\
en-detail-in").innerText=a,get("token-detail-out").innerText=o,get("token-detail-content").innerText=
l,get("token-detail-thought").innerText=c,get("token-detail-encrypted").innerText=d;const v=get("tok\
en-detail-cache-read-row"),w=get("token-detail-cache-write-row"),x=get("token-detail-cache-write-ttl");
if(v&&v.classList.toggle("hidden",!b),w&&w.classList.toggle("hidden",!b),get("token-detail-cache-rea\
d")&&(get("token-detail-cache-read").innerText=m===null?"-":m),get("token-detail-cache-write")&&(get(
"token-detail-cache-write").innerText=g===null?"-":g),x){const E=["5m","1h","30m","12h"].map(q=>{var W;
const N=Number(((W=t.cache_write_ttls)==null?void 0:W[q])||0);return N>0?`${q}: ${N}`:""}).filter(Boolean);
x.textContent=E.length?`TTL ${E.join(" \xB7 ")}`:"",x.classList.toggle("hidden",!b||!E.length)}const k=get(
"token-detail-cost");if(k&&(k.replaceChildren(),k.classList.toggle("hidden",t.role!=="total"),t.role===
"total")){const E=document.createElement("p");E.textContent=costEstimateLabel(t.cost_estimate);const q=document.
createElement("p");q.id="token-detail-cost-jpy",k.append(E,q),fillCostEstimateJpy(q,t.cost_estimate);
const N=Array.isArray((C=t.cost_estimate)==null?void 0:C.items)?t.cost_estimate.items:[];if(N.length){
const W=document.createElement("details");W.className="pt-2 border-t border-gray-700";const I=document.
createElement("summary");I.className="cursor-pointer",I.textContent=`\u30EA\u30AF\u30A8\u30B9\u30C8\u5185\u8A33\uFF08${N.
length}\u4EF6\uFF09`,W.appendChild(I),appendCostItemList(W,N),k.appendChild(W)}for(const W of[`\u96C6\u8A08\u3057\u305FA\
PI\u547C\u3073\u51FA\u3057: ${((M=t.cost_estimate)==null?void 0:M.requests)||0}\u56DE`,COST_ESTIMATE_NOTE,
`\u516C\u5F0F\u6599\u91D1\u78BA\u8A8D\u65E5: ${(costPricing==null?void 0:costPricing.checked_at)||"\u53D6\
\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"}`]){const I=document.createElement("p");I.textContent=
W,k.appendChild(I)}for(const[W,I]of Object.entries((costPricing==null?void 0:costPricing.sources)||{})){
const j=document.createElement("a");j.href=I,j.textContent=`${W} \u516C\u5F0F\u6599\u91D1 `,j.target=
"_blank",j.rel="noopener noreferrer",j.className="underline",k.appendChild(j)}}const S=t.model?`${t.
model} (${t.role})`:`${t.role}`;get("token-detail-title").innerText=S,showTokenDetailModal(e)}r(openTokenDetail,
"openTokenDetail");function closeTokenDetail(e=!1){cancelPromptTokenDetail(),get("token-detail-modal")&&
(hideModal("token-detail-modal"),!e&&location.pathname==="/token-details"&&history.back())}r(closeTokenDetail,
"closeTokenDetail");function openEncryptionSettings(e){const t=messageMeta[e];t&&openEncryptionModal(
t.is_encrypted)}r(openEncryptionSettings,"openEncryptionSettings");function openEncryptionModal(e){if(!get(
"encryption-status-modal"))return;const n=get("encryption-status-title"),i=get("encryption-status-bo\
dy"),a=get("encryption-status-admin-actions"),o=get("encryption-status-admin-toggle"),l=!!e;l?(n&&(n.
innerText="\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059"),i&&(i.innerText=isAdminUser?"\u3053\u306E\u30E1\
\u30C3\u30BB\u30FC\u30B8\u306FE2EE\u3067\u6697\u53F7\u5316\u3055\u308C\u3066\u3044\u307E\u3059\u3002\u7BA1\u7406\u8005\u306F\u4E0B\u306E\u30DC\u30BF\u30F3\u3067\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u5168\u4F53\u3092\u5FA9\u53F7\u5316\u3067\u304D\u307E\u3059\u3002":
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
ck",m),i&&i.removeEventListener("click",g),a&&a.removeEventListener("click",g),e.removeEventListener(
"click",b,!0)}r(c,"cleanup");function d(v){if(l)return;l=!0,t&&t.checked&&(setGeminiLocalPyDialogEnabled(
!1),syncGeminiLocalPyDialogSetting()),c(),hideModal("gemini-local-python-modal"),o(v)}r(d,"finalize");
function m(){d(!0)}r(m,"onOk");function g(){d(!1)}r(g,"onCancel");function b(v){v.target===e&&(v.preventDefault(),
v.stopImmediatePropagation(),g())}r(b,"onOverlay"),n&&n.addEventListener("click",m),i&&i.addEventListener(
"click",g),a&&a.addEventListener("click",g),e.addEventListener("click",b,!0)})},"confirmGeminiLocalP\
ythonSwitch");function renderPendingMessage(e=null,t=!0,n=!0,i=null,a=null){const o=t?"fade-in":"",l=i?
` id="${i}"`:"",c=buildPendingSkeletonHtml(a,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."),d=`<div clas\
s="flex justify-start mb-4 ai-pending-row ${o}"><div${l} class="message-bubble ai-pending-bubble bg-\
gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${c}</div></div>`,m=e||get("\
chat-container");if(m){if(typeof m.insertAdjacentHTML=="function")m.insertAdjacentHTML("beforeend",d);else{
const g=document.createElement("div");g.innerHTML=d;const b=g.firstElementChild;b&&m.appendChild(b)}
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
var c,d;const n=Math.max(0,Number((c=t.retries)!=null?c:1)||0),i=Math.max(0,Number((d=t.retryDelayMs)!=
null?d:180)||0),a=!!t.notifyOnFailure,o=e!=null&&e!==""?String(e):null,l=currentThreadId!=null&&currentThreadId!==
""?String(currentThreadId):null;if(!l||o&&l!==o)return!1;for(let m=0;m<=n;m++)try{return currentThreadId!=
null&&currentThreadId!==""&&String(currentThreadId)!==l?!1:(await loadMessages(l,{preserveDraft:!0,silent:!0}),
!0)}catch{m<n&&i>0&&await new Promise(b=>setTimeout(b,i))}return a&&(currentThreadId!=null&&currentThreadId!==
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
c=o?Array.from(o.options).find(d=>d.value.toLowerCase()===l||d.textContent.trim().toLowerCase()===l):
null;return!o||!c?(showToast(`${n.label}: \u6307\u5B9A\u5024\u300C${a}\u300D\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1):(o.value=c.value,o.dispatchEvent(new Event("change",{bubbles:!0})),refreshMinimalOptionItems(),
showToast(`${i.label}: ${c.textContent.trim()}`,"success"),!0)}if(i.special==="thinking"&&a){const o=a.
toLowerCase(),l=activeThinkingSelect(),c=!!(l&&l.id==="reasoning-effort"),d=c?{off:"none",none:"none",
min:"low",minimal:"low",low:"low",mid:"medium",medium:"medium",high:"high",xhigh:"xhigh",max:"max"}:
{min:"minimal",minimal:"minimal",low:"low",mid:"medium",medium:"medium",high:"high"},m=parseSlashToggleArgument(
a),g=get(i.checkboxId);if(Object.prototype.hasOwnProperty.call(d,o)){const b=d[o],v=l?Array.from(l.options).
find(w=>w.value===b&&!w.disabled&&!w.classList.contains("hidden")):null;if(l&&!v)return showToast(`T\
hinking: \u300C${o}\u300D\u306F\u3053\u306E\u30E2\u30C7\u30EB\u3067\u306F\u5229\u7528\u3067\u304D\u307E\u305B\u3093`,
"warning"),!1;if(b==="none"){if(g&&g.disabled)return showToast("Thinking: \u3053\u306E\u30E2\u30C7\u30EB\u3067\u306FOFF\u306B\u3067\u304D\u307E\u305B\u3093",
"warning"),!1;g&&g.checked?(g.checked=!1,g.dispatchEvent(new Event("change",{bubbles:!0}))):l&&(l.value=
"none",l.dispatchEvent(new Event("change",{bubbles:!0})))}else g&&!g.checked&&!g.disabled&&(g.checked=
!0,g.dispatchEvent(new Event("change",{bubbles:!0}))),l&&(l.value=b,l.dispatchEvent(new Event("chang\
e",{bubbles:!0})));return refreshMinimalOptionItems(),showToast(`Thinking: ${o}`,"success"),!0}if(m===
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
                `;let k=!1;x.addEventListener("pointerdown",S=>{typeof S.button=="number"&&S.button!==
0||(S.preventDefault(),k=!0,selectSlashCommand(v.id))}),x.addEventListener("click",S=>{S.preventDefault(),
k||selectSlashCommand(v.id)}),x.onmouseenter=()=>{slashSelectedIndex=w,showSlashCommandSuggestions(e)},
n.appendChild(x)});const o=i.getBoundingClientRect(),l=window.innerHeight,c=l-o.bottom,d=o.top,m=260,
g=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,o.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",c<180&&d>c){const v=Math.min(m,d-g);t.style.top="auto",t.style.bottom=`${l-o.top+4}px`,n.style.
maxHeight=`${v}px`}else{const v=Math.min(m,c-g);t.style.top=`${o.bottom+4}px`,t.style.bottom="auto",
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
return;removeEphemeralMessageControls(c);const d=c.querySelector(".message-bubble");if(!d||!i.length)
return;const m=document.createElement("div");m.className="mt-3 space-y-2 ai-settings-result-list",i.
forEach(([b,v])=>{const w=AI_SETTING_JUMP_TARGETS[b]||{label:b},x=document.createElement("button");x.
type="button",x.className="w-full flex items-center gap-3 rounded-xl border border-white/10 bg-black\
/20 px-3 py-2.5 text-left hover:bg-black/30 hover:border-blue-400/40 transition ai-settings-result-i\
tem";const k=document.createElement("span");k.className="min-w-0 flex-1";const S=document.createElement(
"span");S.className="block text-xs font-bold text-blue-200",S.textContent=w.label;const C=document.createElement(
"span");C.className="block mt-0.5 text-[11px] text-gray-300 break-words",C.textContent=formatAiSettingValue(
v);const M=document.createElement("i");M.className="fas fa-arrow-up-right-from-square text-[10px] te\
xt-blue-300 shrink-0",k.appendChild(S),k.appendChild(C),x.appendChild(k),x.appendChild(M),x.addEventListener(
"click",()=>openAiSettingJumpTarget(b)),m.appendChild(x)});const g=d.querySelector(".message-footer-\
meta");g?d.insertBefore(m,g):d.appendChild(m),scrollToBottom()}r(renderAiSettingsResultBubble,"rende\
rAiSettingsResultBubble");async function runAiSettingsCommand(e,t){pendingSlashCommand!=="settings"&&
(pendingSlashCommand="settings",showPendingSlashCommandIndicator("settings")),appendAiSettingsConversation(
"user",e);const n=Date.now(),i=renderMessage(`settings-user-${n}`,"user",`/settings ${e}`,null,null,
null,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(i);const a=get(
"welcome-screen");a&&a.classList.add("hidden");const o=`settings-pending-${n}`,l=get("chat-container");
l&&(l.insertAdjacentHTML("beforeend",`<div id="${o}" class="flex justify-start mb-4 ai-pending-row f\
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
keys(d.applied).length}\u9805\u76EE\uFF09`,"success");try{const v=await apiFetch(CHAT_CONFIG.urls.handleSettingsQuery).
then(w=>w.json());populateAiSafeFormFields(v),cacheUserSettings(v)}catch{}renderAiSettingsResultBubble(
d.applied,t);return}const g=d.message||d.error||"\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F";
appendAiSettingsConversation("assistant",`\u8A2D\u5B9A\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F: ${g}`);
const b=renderMessage(`settings-error-${Date.now()}`,"assistant",`\u8A2D\u5B9A\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002

${g}`,null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(
b),showToast(g,"error",!0)}catch{appendAiSettingsConversation("assistant","\u8A2D\u5B9A\u64CD\u4F5C\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002");
const d=get(o);d&&d.remove();const m=renderMessage(`settings-error-${Date.now()}`,"assistant","\u8A2D\u5B9A\u5909\u66F4\u306E\
\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
null,null,t,null,!0,null,null,null,null,null,null,null,null,!0);removeEphemeralMessageControls(m),showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u901A\u4FE1\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}}
r(runAiSettingsCommand,"runAiSettingsCommand");function hideSkillSuggestions(){const e=get("skill-su\
ggestions");e&&e.classList.add("hidden"),skillSuggestionsVisible=!1,skillSelectedIndex=0}r(hideSkillSuggestions,
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
e)},n.appendChild(k)});const l=i.getBoundingClientRect(),c=window.innerHeight,d=c-l.bottom,m=l.top,g=260,
b=8;if(t.style.position="fixed",t.style.left=`${Math.max(8,l.left)}px`,t.style.zIndex="80",t.style.maxHeight=
"none",d<180&&m>d){const w=Math.min(g,m-b);t.style.top="auto",t.style.bottom=`${c-l.top+4}px`,n.style.
maxHeight=`${w}px`}else{const w=Math.min(g,d-b);t.style.top=`${l.bottom+4}px`,t.style.bottom="auto",
n.style.maxHeight=`${w}px`}t.classList.remove("hidden"),skillSuggestionsVisible=!0}r(showSkillSuggestions,
"showSkillSuggestions");function selectSkillSuggestion(e){const t=get("prompt-input");if(!t)return;const n=t.
value,i=n.lastIndexOf("@");i!==-1?t.value=n.substring(0,i).trimEnd():t.value="",hideSkillSuggestions(),
activateSkill(e),t.focus(),t.dispatchEvent(new Event("input",{bubbles:!0}))}r(selectSkillSuggestion,
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
ta-filename");c&&i.push(c),t?currentImageUrls=currentImageUrls.filter(d=>d!==c):browserFastLocalFiles.
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
const e=r((i,a)=>get(i)?!!get(i).checked:a,"checked"),t=r(i=>get(i)?get(i).value:null,"value"),n=browserFastPreviousOptions&&
browserFastPreviousOptions.checks&&Object.prototype.hasOwnProperty.call(browserFastPreviousOptions.checks,
"enable-skills")?!!browserFastPreviousOptions.checks["enable-skills"]:!0;return{canvas_mode:!!canvasModeEnabled,
enable_search:e("enable-search",!1),gemini_free_tier:e("enable-gemini-free-tier",!1),enable_url_context:e(
"enable-url-context",!1),enable_maps:e("enable-maps",!1),enable_python:e("enable-python",!1),enable_file_creation:e(
"enable-file-creation",!0),enable_mcp:typeof isMcpEnabledForSend=="function"?!!isMcpEnabledForSend():
!1,enable_skills:n,enable_thinking:e("enable-thinking",!1),thinking_level:t("thinking-level"),thinking_budget:t(
"thinking-budget"),reasoning_effort:t("reasoning-effort"),safety_setting:t("safety-setting"),enable_prompt_caching:e(
"enable-prompt-cache",!1),math_steps:typeof window.collectMathStepsSendOption=="function"?window.collectMathStepsSendOption():
!1}}r(collectBrowserFastPromptOptions,"collectBrowserFastPromptOptions");async function sendBrowserFastMessage(e,t){
var G,ae,R;const n=String(get("model-select").value||"").trim(),i=await fetchBrowserFastBootstrap(!1);
if(!browserFastApiKey||browserFastApiKeyModel!==n)throw new Error("\u9078\u629E\u4E2D\u30E2\u30C7\u30EB\u306E\u4FDD\u5B58\u6E08\u307FGemini API\u30AD\u30FC\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\
\u305F");const a=Array.from(browserFastLocalFiles.values()),o=[];for(const F of a)o.push({inlineData:{
mimeType:F.file.type,data:await fileToBase64Payload(F.file)}});o.push({text:e});const l={},c=browserFastThinkingConfig(
n.toLowerCase());c&&(l.thinkingConfig=c);const d={contents:[...await buildBrowserFastHistoryContents(
i.history),{role:"user",parts:o}],generationConfig:l};i.math_steps_system_prompt&&(d.systemInstruction=
{parts:[{text:String(i.math_steps_system_prompt)}]}),!!(get("enable-python")&&get("enable-python").checked)&&
(d.tools=[{codeExecution:{}}]),e.trim()&&(promptHistory.length===0||promptHistory[0]!==e)&&(promptHistory.
unshift(e),promptHistory.length>100&&promptHistory.pop()),historyIndex=-1,tempPrompt="",playSendAnimation(),
get("welcome-screen").classList.add("hidden");const g=userAutoScroll&&isChatNearBottom();g||pauseChatAutoScroll();
const b=collectImageUrlsForSend(),v=Array.from(((G=get("file-preview-thumbs"))==null?void 0:G.querySelectorAll(
"img"))||[]).map(F=>F.cloneNode(!0)),w=renderMessage(Date.now(),"user",e,JSON.stringify(collectImageUrlsForSend()),
null,null,null,!1,currentQuote,null,null,null,null,null,null,null,g);if(v.length&&!b.length){const F=document.
createElement("div");F.className="flex gap-2 flex-wrap",v.forEach(me=>{me.removeAttribute("onclick"),
F.appendChild(me)}),w.querySelector(".message-bubble").appendChild(F)}const x=takePromptForSend(),k=animatePromptToMessage(
w,x,g),S=`browser-fast-${Date.now()}`;get("chat-container").insertAdjacentHTML("beforeend",`<div cla\
ss="flex justify-start mb-4 ai-pending-row fade-in"><div id="${S}" class="message-bubble ai-pending-\
bubble bg-gray-700 text-white p-4 rounded-2xl rounded-tl-none shadow-md relative">${buildPendingSkeletonHtml(
n,"Gemini\u3078\u76F4\u63A5\u9001\u4FE1\u4E2D...")}</div></div>`);const C=get(S),M=startPendingElapsed(
C,t);activeStreamingBubbleId=S,setSendBtnToStopMode(),g&&resumeChatAutoScroll(),abortController=new AbortController;
let E="",q="",N=null;const W=new Set,I=[];let j=null,ne=null,oe=!1;const te={},ge=[];let $=null,O="";
const V=window.ProgressSpinner?window.ProgressSpinner.startFlow("browserFast"):null;let B=!1;try{const F=await fetch(
`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(n)}:streamGenerateCont\
ent?alt=sse`,manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json","\
x-goog-api-key":browserFastApiKey},body:JSON.stringify(d),signal:abortController.signal}));if(!F.ok){
const Je=await F.json().catch(()=>({}));throw new Error(Je&&Je.error&&Je.error.message?Je.error.message:
`Gemini API HTTP ${F.status}`)}window.ConnectionMonitor&&(B=!0,window.ConnectionMonitor.operationStarted()),
V&&V.setPhase("waiting"),x.accept(!1),(ae=window.ActivityLog)==null||ae.log("send.stream.start");let me=!1;
const re=F.body.getReader(),X=new TextDecoder;let ce="";const Q=r(Je=>{var ee,nt;const he=Je.split(/\r?\n/).
filter(Pe=>Pe.startsWith("data:")).map(Pe=>Pe.slice(5).trim()).join("");if(!he||he==="[DONE]")return;
const fe=JSON.parse(he);if(Number.isInteger((ee=fe.usageMetadata)==null?void 0:ee.promptTokenCount)){
N=N||{};for(const Pe of["promptTokenCount","candidatesTokenCount","thoughtsTokenCount","cachedConten\
tTokenCount"])Number.isInteger(fe.usageMetadata[Pe])&&fe.usageMetadata[Pe]>=0&&(N[Pe]=fe.usageMetadata[Pe])}
for(const Pe of fe.candidates||[])for(const mt of((nt=Pe.groundingMetadata)==null?void 0:nt.webSearchQueries)||
[])typeof mt=="string"&&mt.trim()&&W.add(mt);if(W.size&&(N=N||{},N.grounding_queries=W.size),fe.error)
throw new Error(fe.error.message||"Gemini API error");if((Array.isArray(fe.candidates)?fe.candidates:
[]).forEach(Pe=>{(Pe&&Pe.content&&Array.isArray(Pe.content.parts)?Pe.content.parts:[]).forEach(Ke=>{
if(Ke&&typeof Ke.thoughtSignature=="string"&&!I.includes(Ke.thoughtSignature)&&I.push(Ke.thoughtSignature),
Ke&&Ke.executableCode&&typeof Ke.executableCode.code=="string"){const lt=Ke.executableCode.code;E+=`\

\`\`\`python
${lt}
\`\`\`
`,$=`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`,O=lt,te[$]||(C.insertAdjacentHTML(
"afterbegin",browserFastPythonBoxHtml($)),te[$]=C.querySelector(`[data-py-id="${$}"]`)),updateBrowserFastPythonBox(
te[$],"code",lt);return}if(Ke&&Ke.codeExecutionResult&&typeof Ke.codeExecutionResult.output=="string"){
const lt=Ke.codeExecutionResult.output;E+=`
**Output:**
\`\`\`
${lt}
\`\`\`
`;const wt=$||`browserFastPy_${Date.now()}_${Math.random().toString(36).slice(2,8)}`;ge.push({code:O||
"",output:lt}),te[wt]||(C.insertAdjacentHTML("afterbegin",browserFastPythonBoxHtml(wt)),te[wt]=C.querySelector(
`[data-py-id="${wt}"]`)),updateBrowserFastPythonBox(te[wt],"output",lt);return}const at=typeof Ke.text==
"string"?Ke.text:"";at&&(Ke.thought===!0?q+=at:E+=at)})}),!oe&&(E||q)){beginPendingToStreamTransition(
C);const Pe=C.querySelector(".content-area");Pe&&Pe.remove(),oe=!0}q&&(ne||(C.insertAdjacentHTML("af\
terbegin",'<div class="thought-container"><div class="thought-header" onclick="toggleThinking(this)"\
><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content"></d\
iv></div>'),ne=C.querySelector(".thought-content")),ne.textContent=q),E&&(j||(j=document.createElement(
"div"),j.className="content-area prose prose-invert text-sm break-words",C.appendChild(j)),renderAiMarkdownInto(
j,E,{incrementalMath:!0,streaming:!0,mathSteps:{jobId:null,streaming:!0}})),scrollToBottom()},"consu\
meEvent");for(;;){const{done:Je,value:he}=await re.read();if(!Je&&!me&&(me=!0,(R=window.ActivityLog)==
null||R.log("send.stream.first")),Je)break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),
V&&V.setPhase("receiving"),ce+=X.decode(he,{stream:!0});const fe=ce.split(/\r?\n\r?\n/);ce=fe.pop()||
"",fe.forEach(Q)}if(ce+=X.decode(),ce.trim()&&Q(ce),!E.trim())throw new Error("Gemini\u304B\u3089\u56DE\u7B54\u672C\u6587\u304C\u8FD4\u3055\u308C\u307E\u305B\u3093\u3067\u3057\
\u305F");j&&renderAiMarkdownInto(j,E,{incrementalMath:!0,streaming:!1,mathSteps:{jobId:null,streaming:!1}}),
ne&&ne.classList.add("collapsed"),ge.length&&(E+=ge.map(Je=>`
\`\`\`pyexec
${JSON.stringify(Je)}
\`\`\`
`).join(""));let ye=[];a.length&&(V&&V.setPhase("saving"),showToast("\u56DE\u7B54\u304C\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u753B\u50CF\u3068\u5C65\u6B74\u3092\u30B5\u30FC\u30D0\u30FC\u3078\u4FDD\u5B58\u3057\u3066\u3044\u307E\u3059\u3002",
"info",!1),ye=await uploadBrowserFastLocalFiles(x.files,!0)),V&&V.setPhase("saving");const ue=[...b,
...ye],ke=await fetchChatStreamWithUnavailableRetry("/api/browser_fast_mode/save",manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({client_request_id:createClientRequestId(),
message:e,assistant_content:E,pricing_usage:N,thought_content:q,model:n,image_urls:ue,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i.parent_id||null,thought_signatures:I,
turnstile_token:botTurnstileTokenForRequest()}),signal:abortController.signal}),C),Be=await ke.json().
catch(()=>({}));if(!ke.ok||!Be.thread_id)throw new Error(Be.error||"DB\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F");
if(Be.assistant_message_id!==void 0&&Be.assistant_message_id!==null&&typeof window.bindMathStepsMessageId==
"function"){const Je=C?C.querySelector(".math-steps"):null;Je&&await window.bindMathStepsMessageId(Je,
Be.assistant_message_id)}const Ue=!currentThreadId;currentThreadId=String(Be.thread_id),currentParentId=
Be.assistant_message_id||null,currentLeafId=Be.assistant_message_id||null,browserFastBootstrap=null,
await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,skipHistory:!Ue}),applyBrowserFastModeRestrictions(),
loadThreads(!1),showToast("\u9AD8\u901F\u30E2\u30FC\u30C9\u306E\u56DE\u7B54\u3092\u5C65\u6B74\u3078\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success",!1)}catch(F){if(k(),x.restore(),F.name!=="AbortError"){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${F.
message}`,"error",!0),get("prompt-input").value||(get("prompt-input").value=e);const me=F.message||"\
\u30A8\u30E9\u30FC";C&&C.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(me));try{let re=E||
"";ge.length&&(re+=ge.map(ue=>`
\`\`\`pyexec
${JSON.stringify(ue)}
\`\`\`
`).join(""));const X=buildChatErrorMarkdown(me,re),ce=a.length?[]:b,Q=await fetchChatStreamWithUnavailableRetry(
"/api/browser_fast_mode/save",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"ap\
plication/json"},body:JSON.stringify({client_request_id:createClientRequestId(),message:e,assistant_content:X,
pricing_usage:N,thought_content:q||"",model:n,image_urls:ce,prompt_options:collectBrowserFastPromptOptions(),
temporary_chat:temporaryChatEnabled,thread_id:currentThreadId||null,parent_id:i&&i.parent_id?i.parent_id:
null,thought_signatures:I,turnstile_token:botTurnstileTokenForRequest()}),signal:abortController&&!abortController.
signal.aborted?abortController.signal:void 0}),C),ye=await Q.json().catch(()=>({}));if(Q.ok&&ye.thread_id){
const ue=!currentThreadId;currentThreadId=String(ye.thread_id),currentParentId=ye.assistant_message_id||
null,currentLeafId=ye.assistant_message_id||null,browserFastBootstrap=null,await loadMessages(currentThreadId,
{preserveDraft:!0,silent:!0,skipHistory:!ue}),applyBrowserFastModeRestrictions(),loadThreads(!1)}}catch(re){
sendClientDebugLog("error",`Browser fast error persist failed: ${re&&re.message?re.message:re}`)}}}finally{
x.releaseFiles(),M&&M(),B&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),V&&V(),
setSendBtnToSendMode(),activeStreamingBubbleId===S&&(activeStreamingBubbleId=null),abortController=null,
updateFilePreview()}}r(sendBrowserFastMessage,"sendBrowserFastMessage");async function sendMessage(){
return window.ActivityLog?window.ActivityLog.operation("send",()=>sendMessageLogged()):sendMessageLogged()}
r(sendMessage,"sendMessage");async function sendMessageLogged(e={}){var At,Et,Ft,ln,Yt;const t=performance.
now();if(vibrateHelper(50),abortController){showToast("\u56DE\u7B54\u751F\u6210\u4E2D\u3067\u3059\u3002\u5B8C\u4E86\u307E\u3067\u304A\u5F85\u3061\u3044\u305F\u3060\u304F\u304B\u3001\u505C\u6B62\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(uploadProgressState.active>0){showToast("\u30D5\u30A1\u30A4\u30EB\u306E\u9001\u4FE1\u30FB\u51E6\u7406\u4E2D\u3067\u3059\u3002\u3057\u3070\u3089\u304F\u304A\u5F85\u3061\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(isLyriaRealtimeModel()){const J=get("prompt-input").value;get("prompt-input").
value="",get("prompt-input").style.height="auto",window.openLyriaStudio&&window.openLyriaStudio(J);return}
if(isBotDetectionActive()&&registerSendButtonSpam()>=8&&!await runSendSpamVerification()){showToast(
"\u9001\u4FE1\u64CD\u4F5C\u304C\u901F\u3059\u304E\u308B\u305F\u3081\u3001\u78BA\u8A8D\u5F8C\u306B\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}let n=null;if(isBotDetectionActive()){if(n=await getTurnstileToken(),!n&&!botDetectionVerified){
try{await runBotDetectionGate()}catch{}n=await getTurnstileToken()}if(!n&&!botDetectionVerified){showToast(
"\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0),botTelemetry.send(!0);return}n&&await verifyTurnstileOnServer(n)}const i=isIdeogramBackgroundRemoval(),
a=ideogramPromptForSend(get("prompt-input").value);if(pendingSlashCommand&&!i){const J=pendingSlashCommand,
we=a.trim(),je=get("model-select")?get("model-select").value:null;if(J==="settings"){if(!we){showToast(
"\u8A2D\u5B9A\u5909\u66F4\u306E\u6307\u793A\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044\uFF08\u4F8B: \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092gemini-2.5-flash\u306B\uFF09",
"info"),get("prompt-input").focus();return}if(!je){showToast("\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
we,je)}else executeMinimalSlashCommand(J,we)?(get("prompt-input").value="",get("prompt-input").style.
height="auto",hidePendingSlashCommandIndicator()):get("prompt-input").focus();return}const o=a.trim().
match(/^\/([a-z][\w-]*)(?:\s+(.*))?$/i);if(o&&minimalPromptMode&&MINIMAL_SLASH_COMMANDS.some(J=>J.id===
o[1].toLowerCase())){executeMinimalSlashCommand(o[1].toLowerCase(),o[2]||"")&&(hideSlashCommandSuggestions(),
get("prompt-input").value="",get("prompt-input").style.height="auto");return}const l=!!(get("enable-\
batch-mode")&&get("enable-batch-mode").checked);if(l&&codingModeEnabled){showToast("Batch API\u3067\u306FCodin\
g Mode\u3092\u5229\u7528\u3067\u304D\u307E\u305B\u3093\u3002Batch\u3092\u89E3\u9664\u3059\u308B\u304BCoding\u3092\u89E3\u9664\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!0);return}if(browserFastModeEnabled)if(l)setBrowserFastModeEnabled(!1);else{const J=browserFastModeIneligibility(
a);if(!J){try{await sendBrowserFastMessage(a,t)}catch(we){showToast(`\u9AD8\u901F\u30E2\u30FC\u30C9: ${we.
message||"\u958B\u59CB\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F"}`,"error",!0)}return}if(showToast(
`\u9AD8\u901F\u30E2\u30FC\u30C9\u6761\u4EF6\u5916: ${J}\u3002\u901A\u5E38\u30E2\u30FC\u30C9\u3078\u5207\u308A\u66FF\u3048\u307E\u3059\u3002`,
"warning",!0),browserFastLocalFiles.size)try{await uploadBrowserFastLocalFiles()}catch(we){showToast(
we.message||"\u901A\u5E38\u30E2\u30FC\u30C9\u7528\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0);return}return setBrowserFastModeEnabled(!1),sendMessage()}a.trim()&&(promptHistory.length===
0||promptHistory[0]!==a)&&(promptHistory.unshift(a),promptHistory.length>100&&promptHistory.pop()),historyIndex=
-1,tempPrompt="";const c=collectAttachmentItemsForSend(),d=c.map(J=>J.path),m=c.filter(J=>normalizeAttachmentSource(
J.source)==="upload").map(J=>J.path),g=getModelMediaSupport(get("model-select").value),b=d.some(J=>isAudioPath(
J)),v=d.some(J=>isVideoPath(J)),w=(get("model-select").value||"").toLowerCase(),x=get("enable-python"),
k=!!(x&&x.checked);if(b&&!g.audio||v&&!g.video){showToast("\u3053\u306E\u30E2\u30C7\u30EB\u306F\u97F3\u58F0/\u52D5\u753B\u5165\u529B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093",
"error",!0),purgeUnsupportedAttachments(!0);return}if(!a.trim()&&d.length===0)return;if(isMistralOcrModel(
w)){const J=/https?:\/\/\S+/i.test(a);if(d.filter(je=>isAudioPath(je)||isVideoPath(je)).length){showToast(
"Mistral OCR \u306F\u97F3\u58F0\u30FB\u52D5\u753B\u306B\u5BFE\u5FDC\u3057\u3066\u3044\u307E\u305B\u3093\u3002PDF / \u753B\u50CF / DOCX / PPTX \u3092\u6DFB\u4ED8\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(!d.length&&!J){showToast("Mistral OCR \u306F\u6587\u66F8\u5C02\u7528\u3067\u3059\u3002PDF\u30FB\u753B\u50CF\u30FBDOCX\u30FBPPTX \u3092\u6DFB\u4ED8\u3059\u308B\u304B\u3001\u516C\u958BURL\u3092\u5165\u529B\
\u3057\u3066\u304F\u3060\u3055\u3044\u3002","error",!0);return}}const S=a.trim();if(/^\/settings(?:\s|$)/i.
test(S)&&isMistralOcrModel()){showToast("Mistral OCR \u306F\u8A2D\u5B9A\u5909\u66F4\u30B3\u30DE\u30F3\u30C9\u306B\u4F7F\u3048\u307E\u305B\u3093\u3002\u30C1\u30E3\u30C3\u30C8\u30E2\u30C7\u30EB\u3092\u9078\u3093\u3067\u304F\u3060\u3055\u3044\u3002",
"error",!0);return}if(/^\/settings(?:\s|$)/i.test(S)){const J=S.replace(/^\/settings\s*/i,"").trim();
if(!J){showToast("\u4F7F\u3044\u65B9: /settings \u30C7\u30D5\u30A9\u30EB\u30C8\u30E2\u30C7\u30EB\u3092 gemini-2.5-flash \u306B\u5909\u66F4\u3057\u3066 thinking \u3092\u30AA\u30F3\u306B",
"info");const je=get("prompt-input");je.value="/settings ";const qe=extractSlashCommandToken(je.value);
lastSlashFilter=qe,showSlashCommandSuggestions(qe),je.focus();return}const we=get("model-select")?get(
"model-select").value:null;if(!we){showToast("\u30E2\u30C7\u30EB\u304C\u9078\u629E\u3055\u308C\u3066\u3044\u307E\u305B\u3093",
"error",!0);return}get("prompt-input").value="",get("prompt-input").style.height="auto",await runAiSettingsCommand(
J,we);return}if(isGeminiLocalPythonMode(w,b,v,k)&&!await confirmGeminiLocalPythonSwitch())return;let C=null,
M=[];if(codingModeEnabled){const J=collectCodingCandidates(a),we=J.filter(et=>et.prompt_source),je=J.
filter(et=>!et.prompt_source),qe=we.reduce((et,kt)=>et+String(kt.code||"").length,0);if(qe>3e5){showToast(
"\u5165\u529B\u5185\u306E\u7DE8\u96C6\u5019\u88DC\u30B3\u30FC\u30C9\u5408\u8A08\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\u9650300,000\u6587\u5B57\uFF09",
"error",!0);return}let Xe=3e5-qe;const ft=[];for(let et=je.length-1;et>=0;et--){const kt=String(je[et].
code||"").length;kt>Xe||(ft.unshift(je[et]),Xe-=kt)}M=codingTargetSelection?ft.slice(-1):[...we,...ft];
const gt=we.length?we[we.length-1]:null;if(C=codingTargetSelection?M[0]:gt||M[M.length-1]||null,codingModeEffective=
!!(C&&String(C.code||"").trim()),codingModeEffective&&C.code.length>3e5){showToast("\u7DE8\u96C6\u5BFE\u8C61\u30B3\u30FC\u30C9\u304C\u5927\u304D\u3059\u304E\u307E\u3059\uFF08\u4E0A\
\u9650300,000\u6587\u5B57\uFF09","error",!0);return}if(codingModeEffective){const et=String(((At=get(
"model-select"))==null?void 0:At.value)||"").toLowerCase();if(/(image|video|tts|audio|native-audio)/.
test(et)){showToast("Coding Mode\u3067\u306F\u30C6\u30AD\u30B9\u30C8\u751F\u6210\u30E2\u30C7\u30EB\u3092\u9078\u629E\u3057\u3066\u304F\u3060\u3055\u3044",
"error",!0);return}}}const E=codingModeEnabled&&codingModeEffective;sendClientDebugLog("info",`Promp\
t send start: model=${get("model-select").value} thread=${currentThreadId||"-"} text_len=${a.length}\
 attachments=${d.length} search=${get("enable-search").checked}`);const q=a,N=hasMarkerHint()?MARKER_HINT_TEXT:
null,W=isIdeogramObjectRemoval();if(isGptImageModel()&&currentMaskImage&&d.length===0){showToast("Ma\
sk \u306F\u753B\u50CF\u5165\u529B\u304C\u5FC5\u8981\u3067\u3059","error",!0);return}if(W&&!currentMaskImage){
showToast("Object Remover \u306B\u306F\u30DE\u30B9\u30AF\u753B\u50CF\u304C\u5FC5\u8981\u3067\u3059",
"error",!0);return}const I=userAutoScroll&&isChatNearBottom();I||pauseChatAutoScroll();const j=editingMessageId,
ne=promptSettingsBeforeEdit;promptSettingsBeforeEdit=null;const oe=currentParentId,te=j!=null;j&&(editingMessageId=
null,setEditUi(!1)),playSendAnimation(),get("welcome-screen").classList.add("hidden");const ge=[],$=r(
J=>{if(J==null)return;let we=document.getElementById(`msg-${J}`);for(;we;)we.classList&&we.classList.
contains("message-group")&&(ge.push({node:we,prevDisplay:we.style.display}),we.style.display="none"),
we=we.nextElementSibling},"hideRenderedBranchFrom"),O=r(()=>{ge.forEach(({node:J,prevDisplay:we})=>{
J&&(J.style.display=we||"")}),ge.length=0},"restoreHiddenBranch");j&&$(j);const V=Date.now(),B=isGptImageModel()||
W?currentMaskImage:null,G=renderMessage(V,"user",q,JSON.stringify(d),null,null,null,!1,currentQuote,
null,null,null,null,null,null,null,I,oe,activeSkill?activeSkill.name:null,null,B);let ae=!1;const R=/(https?:\/\/)?(x\.com|twitter\.com)\//i,
F=!i&&(R.test(q||"")||R.test(currentQuote||"")),me="grok-4-fast-reasoning",re=r(()=>{get("enable-sea\
rch").checked=!0,get("model-select").value!==me&&selectModelById(me)},"applyXLinkAuto");if(F&&!isMistralOcrModel()&&
!get("enable-search").checked)if(autoSearchOnLinks)re();else{const J=get("auto-search-banner"),we=get(
"auto-search-on-btn"),je=get("auto-search-off-btn"),qe=get("auto-search-remember");J&&we&&je&&(qe&&(qe.
checked=!1),await new Promise(Xe=>{J.classList.remove("hidden");const ft=r(gt=>{J.classList.add("hid\
den"),we.onclick=null,je.onclick=null,Xe(gt)},"cleanup");we.onclick=()=>ft("enable"),je.onclick=()=>ft(
"disable")}).then(async Xe=>{Xe==="enable"?(re(),qe&&qe.checked&&(autoSearchOnLinks=!0,await apiFetch(
CHAT_CONFIG.urls.handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify({auto_search_on_links:!0})}))):ae=!0}))}const X=String(get("reasoning-effort").value||"").
toLowerCase(),ce=String(get("model-select").value||"").toLowerCase().includes("deepseek")&&X==="none",
Q={client_request_id:createClientRequestId(),thread_id:currentThreadId,message:q,model:get("model-se\
lect").value,image_urls:d,image_items:c,uploaded_image_urls:m,temporary_chat:temporaryChatEnabled,enable_search:get(
"enable-search").checked,gemini_free_tier:!!(get("enable-gemini-free-tier")&&get("enable-gemini-free\
-tier").checked),gemini_flex:!!(get("enable-gemini-flex")&&get("enable-gemini-flex").checked),enable_url_context:get(
"enable-url-context")?get("enable-url-context").checked:!1,enable_maps:get("enable-maps")?get("enabl\
e-maps").checked:!1,enable_python:get("enable-python").checked,enable_mcp:isMcpEnabledForSend(),enable_skills:get(
"enable-skills")?get("enable-skills").checked:!0,enable_file_creation:get("enable-file-creation")?get(
"enable-file-creation").checked:!0,enable_thinking:ce?!1:get("enable-thinking").checked,thinking_level:get(
"thinking-level").value,thinking_budget:get("thinking-budget")?get("thinking-budget").value:null,reasoning_effort:get(
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
n")?get("gpt-image-frame-duration").value:null,image_mask:B,gemini_image_aspect:isGeminiImageModel()&&
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
quote_message_id:currentQuote?currentQuoteMessageId:null,parent_id:oe,parent_id_explicit:te,disable_auto_search:ae,
image_vision_model:currentVisionModel||null,coding_mode:E,coding_target:E?{id:C.candidate_id,code:C.
prompt_source?null:C.code,language:C.language||"text",key:C.key||null,message_id:C.message_id||null,
source:C.prompt_source?"prompt":"history",explicit:C.explicit===!0}:null,coding_candidates:E?M.map(J=>({
id:J.candidate_id,source:J.prompt_source?"prompt":"history",prompt_index:J.prompt_source?J.prompt_index:
null,code:J.prompt_source?null:J.code,language:J.language||"text",explicit:J.explicit===!0})):[],batch_mode:l,
canvas_mode:!!canvasModeEnabled,math_steps:typeof window.collectMathStepsSendOption=="function"?window.
collectMathStepsSendOption():!1};n&&(Q.turnstile_token=n);const ye=get("thread-custom-instruction");
ye&&(Q.thread_custom_instruction=ye.value||""),activeSkill?(Q.system_prompt=activeSkill.instruction,
Q.enable_system_prompt=!0,Q.skill_uuid=activeSkill.uuid):Q.skill_uuid=null;const ue=takePromptForSend(
e.animate!==!1),ke=animatePromptToMessage(G,ue,I&&e.animate!==!1);setSendBtnToStopMode();const Be="a\
i-"+Date.now(),Ue=String(Q.model||"").toLowerCase(),Je=!!Q.enable_thinking||!!X&&X!=="none",he=Ue.includes(
"gemini")||Ue.includes("o1")||Ue.includes("o3")||Ue.includes("gpt-5")||Ue.includes("reasoning")&&!Ue.
includes("non-reasoning"),fe=Je&&he;let ve=buildPendingSkeletonHtml(Q.model,"API\u306B\u9001\u4FE1\u4E2D...");
get("chat-container").insertAdjacentHTML("beforeend",`<div class="flex justify-start mb-4 ai-pending\
-row fade-in"><div id="${Be}" class="message-bubble ai-pending-bubble bg-gray-700 text-white p-4 rou\
nded-2xl rounded-tl-none shadow-md relative">${ve}</div></div>`),I&&resumeChatAutoScroll();const ee=get(
Be),nt=l?null:startPendingElapsed(ee,t);activeStreamingBubbleId=Be,canvasModeEnabled&&resetCanvasPreviewPanel();
let Pe=null;const mt=r(J=>!fe||!ee?null:((!Pe||!ee.contains(Pe))&&(Pe=ee.querySelector(".thought-con\
tent")),Pe||(ee.insertAdjacentHTML("afterbegin",'<div class="thought-container"><div class="thought-\
header thinking-shimmer" onclick="toggleThinking(this)"><i class="fas fa-brain text-purple-400"></i>\
 Thinking Process</div><div class="thought-content collapsed" data-placeholder="1"></div></div>'),Pe=
ee.querySelector(".thought-content")),Pe&&(Pe.setAttribute("data-placeholder","1"),Pe.textContent=J||
"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),Pe),"ensureThoughtPlaceholder");fe&&
mt("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),abortController=new AbortController;
const Ke=currentThreadId,at=nowPerfMs(),lt=Date.now();let wt=!1,an=!1,Sn=!1,Kt=null,_t=null,St=null,
Xt=currentThreadId!=null&&currentThreadId!==""?String(currentThreadId):null;const Mt=r((J,we)=>{if(!we||
J==="status"&&wt||J==="thought"&&an||J==="content"&&Sn)return;const je=Math.max(0,nowPerfMs()-at);J===
"status"?Kt=je:J==="thought"?_t=je:J==="content"&&(St=je),reportFirstTokenLatency({latency_seconds:je/
1e3,latency_ms:je,thread_id:Xt||currentThreadId,job_id:currentJobId,model:Q.model,first_event_type:J,
client_sent_at_ms:lt}),J==="status"?wt=!0:J==="thought"?an=!0:J==="content"&&(Sn=!0)},"maybeReportFi\
rstEventLatency"),xt=window.ProgressSpinner?window.ProgressSpinner.startFlow("chat"):null;let Tt=!1,
on=!1,jt=null,Ct=null,rn=!1;try{Q.thread_id&&activeSkill&&(threadSkillMap[Q.thread_id]=activeSkill,pendingSkillForNewThread=
null);const J=await fetchChatStreamWithUnavailableRetry(CHAT_CONFIG.urls.chatStream,manualSpinnerRequestOptions(
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Q),signal:abortController.
signal}),ee);if(sendClientDebugLog("info",`Prompt stream response status: ${J.status}`),!J.ok){const Fe=await J.
json().catch(()=>({})),Ge=new Error(Fe.error||`HTTP ${J.status}`);throw Ge.serverCode=Fe.code||null,
Ge.serverModel=Fe.model||Q.model,Ge.acceptedJobId=Fe.job_id||null,Ge.acceptedThreadId=Fe.thread_id||
null,Ge}Tt=!0,window.ConnectionMonitor&&(rn=!0,window.ConnectionMonitor.operationStarted()),xt&&xt.setPhase(
"waiting"),ue.accept(),codingModeEnabled&&syncCodingModeUi(!0,{persist:!1});const we=r(()=>{if(!ee)return;
const Fe=ee.querySelector(".content-area");if(Fe&&Fe.getAttribute("data-api-accepted")!=="1"&&(Fe.setAttribute(
"data-api-accepted","1"),!updatePendingSkeletonStatus(ee,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))){
Fe.outerHTML=buildPendingSkeletonHtml(Q.model,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...");
const Ge=ee.querySelector(".content-area");Ge&&Ge.setAttribute("data-api-accepted","1"),updatePendingSkeletonStatus(
ee,"\u63A5\u7D9A\u3057\u307E\u3057\u305F\u3002\u5FDC\u7B54\u3092\u5F85\u3063\u3066\u3044\u307E\u3059...",
"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u72B6\u6CC1\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")}},
"markApiAccepted");we(),(Et=window.ActivityLog)==null||Et.log("send.stream.start");let je=!1;const qe=J.
body.getReader(),Xe=new TextDecoder;let ft="",gt="",et="",kt=!0,tt=null,ct=null,ht=null,cn=!1;const dn={};
let un=0,Dt=!1;for(;!Dt;){const{done:Fe,value:Ge}=await qe.read();if(!Fe&&!je&&(je=!0,(Ft=window.ActivityLog)==
null||Ft.log("send.stream.first")),Fe&&((ln=window.ActivityLog)==null||ln.log("send.stream.done")),Fe)
break;window.ConnectionMonitor&&window.ConnectionMonitor.reportActivity(),xt&&xt.setPhase("receiving"),
ft+=Xe.decode(Ge,{stream:!0});let Ht=ft.split(`
`);ft=Ht.pop();let $t=!1,qt=!1;for(let It of Ht)if(It.trim())try{const Me=JSON.parse(It);if(Me.type===
"thread_id"){we();const Ie=Me.content!==null&&Me.content!==void 0?String(Me.content):Me.content;Ie&&
(Xt=Ie,currentThreadId!==Ie&&(currentThreadId=Ie,history.pushState({},"","/c/"+Ie)),activeSkill&&(threadSkillMap[Ie]=
activeSkill,pendingSkillForNewThread=null),ensureTemporaryChatHeartbeat(!0));continue}if(Me.type==="\
job_id"){we(),currentJobId=Me.content,ee&&ee.setAttribute("data-job-id",String(currentJobId||"")),l&&
showToast("Batch\u767B\u9332","info");continue}if(Me.type==="search_status"){ht=updateSearchBox(ee,ht,
Me.content);continue}if(Me.type==="mcp"){handleMcpStreamEvent(ee,Me.content||{});continue}if(Me.type===
"mcp_decision_request"){openMcpDecisionModal(Me.content||{});continue}if(Me.type==="status"){we();const Ie=Me.
content===null||Me.content===void 0?"":String(Me.content);if(Mt("status",!!Ie),kt&&ee){const ot=Ie||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(ee,ot,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const dt=ee.querySelector(".content-area");dt&&(dt.outerHTML=buildPendingSkeletonHtml(Q.model,ot),updatePendingSkeletonStatus(
ee,ot,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
fe&&mt(Ie||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(kt){beginPendingToStreamTransition(
ee);const Ie=ee.querySelector(".content-area");Ie&&(Ie.innerHTML=""),kt=!1}if(Me.type==="coding_diff")
appendCodingLiveDiff(ee,Me.content||{}),Mt("content",!0);else if(Me.type==="thought"){if(tt||(tt=ee.
querySelector(".thought-content")),et+=Me.content,Mt("thought",!!Me.content),!tt){const Ie='<div cla\
ss="thought-container"><div class="thought-header" onclick="toggleThinking(this)"><i class="fas fa-b\
rain text-purple-400"></i> Thinking Process</div><div class="thought-content"></div></div>';ht?ht.insertAdjacentHTML(
"afterend",Ie):ee.insertAdjacentHTML("afterbegin",Ie),tt=ee.querySelector(".thought-content")}if(tt&&
tt.getAttribute("data-placeholder")==="1"){if(tt.textContent="",tt.removeAttribute("data-placeholder"),
tt){const Ie=tt.parentElement.querySelector(".thought-header");Ie&&Ie.classList.remove("thinking-shi\
mmer")}et=Me.content}tt.classList.remove("collapsed"),qt=!0}else if(Me.type==="image_analysis"){const Ie=Me.
content===null||Me.content===void 0?"":String(Me.content);if(!ee)continue;let ot=ee.querySelector(".\
image-analysis-box");if(!ot){const Ze='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 border\
 border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas f\
a-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></d\
iv></div>';ht?ht.insertAdjacentHTML("afterend",Ze):ee.insertAdjacentHTML("afterbegin",Ze),ot=ee.querySelector(
".image-analysis-box")}const dt=ot.querySelector(".image-analysis-text");dt&&(dt.textContent=Ie)}else if(Me.
type==="python"){const Ie=Me.content||{},ot=Ie.id||`py_${Date.now()}`;if(!dn[ot]){const Ze=`<div cla\
ss="code-wrapper python-box collapsed" data-py-id="${ot}" data-collapsed="true" data-code-key="${ot}\
"><div class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution<\
/span><div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-la\
bel="\u5C55\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-\
code="" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy\
-btn" data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left\
"></i></button></div></div><div class="code-body"><div class="python-section"><div class="python-lab\
el">Code</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="pyt\
hon-section"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-\
output"></code></pre></div></div></div>`;ht?ht.insertAdjacentHTML("afterend",Ze):ee.insertAdjacentHTML(
"afterbegin",Ze),dn[ot]=ee.querySelector(`[data-py-id="${ot}"]`)}const dt=dn[ot];if(dt){if(Ie.code!==
void 0){const Ze=Ie.code==null?"":String(Ie.code),Pt=dt.querySelector(".python-code");Pt&&(Pt.textContent=
Ze,Pt.removeAttribute("data-highlighted"),queueHighlight(dt,Ze));const Gt=dt.querySelector('.copy-bt\
n[data-copy="code"]');Gt&&Gt.setAttribute("data-code",encodeURIComponent(Ze).replace(/'/g,"%27"))}if(Ie.
output!==void 0){const Ze=Ie.output==null?"":String(Ie.output),Pt=dt.querySelector(".python-output");
Pt&&(Pt.textContent=Ze);const Gt=dt.querySelector('.copy-btn[data-copy="output"]');Gt&&Gt.setAttribute(
"data-code",encodeURIComponent(Ze).replace(/'/g,"%27"))}}}else if(Me.type==="content"){const Ie=Me.content===
null||Me.content===void 0?"":String(Me.content);gt+=Ie,/[`~]/.test(Ie)&&activateDeferredCodingModeFromStream(
gt),ct||(ct=ee.querySelector(".content-area")||document.createElement("div"),ct.className="prose pro\
se-invert text-sm break-words",ee.contains(ct)||ee.appendChild(ct)),$t=!0,Mt("content",!!Ie)}else if(Me.
type==="error"){cn=!0,Dt=!0,ee.insertAdjacentHTML("beforeend",buildChatErrorBubbleHtml(Me.content)),
showToast(Me.content||"Unknown error","error",!0);break}}catch{}if(qt&&tt&&(tt.textContent=et,userAutoScroll&&
(tt.scrollTop=tt.scrollHeight)),$t&&ct){const It=Date.now();if(It-un>100){const Me=snapshotCodeCollapse(
ct);renderAiMarkdownInto(ct,gt,{incrementalMath:!0,streaming:!0,mathSteps:{jobId:currentJobId,streaming:!0}}),
applyCodeCollapse(ct,Me,!0),un=It}}scrollToBottom()}if(xt&&xt(),ct){const Fe=snapshotCodeCollapse(ct);
renderAiMarkdownInto(ct,gt,{incrementalMath:!0,streaming:!1,mathSteps:{jobId:currentJobId,streaming:!1}}),
applyCodeCollapse(ct,Fe,!0)}if(scrollToBottom(),vibrateHelper([100,50,100]),ee){queueHighlight(ee,gt);
const Fe=nowPerfMs()-at;enableLatencyMetrics&&reportFirstTokenLatency({is_total:!0,latency_seconds:Fe/
1e3,latency_ms:Fe,thread_id:Xt||currentThreadId,job_id:currentJobId,model:Q.model,client_sent_at_ms:lt,
client_done_at_ms:Date.now()});let Ge=null;Kt!==null&&(Ge=Kt),_t!==null&&(Ge===null||_t<Ge)&&(Ge=_t),
St!==null&&(Ge===null||St<Ge)&&(Ge=St);const Ht={total_ms:Fe,first_response_ms:Ge,phases:Ge!==null?[
{key:"generation",ms:Math.max(0,Fe-Ge)}]:[],source:"client"};messageMeta[Be]=Object.assign(messageMeta[Be]||
{},{role:"assistant",model:Q.model,processing_time:Ht});const $t=ee.querySelector(".processing-time-\
row");$t&&$t.remove(),ee.insertAdjacentHTML("beforeend",buildProcessingTimeHtml(Be,Ht))}editingMessageId=
null,setEditUi(!1),ee&&ee.querySelectorAll(".thought-content").forEach(Ge=>Ge.classList.add("collaps\
ed")),await loadMessages(currentThreadId,{preserveDraft:!0,silent:!0,forceLatestLeaf:!!j}),!cn&&codingModeEnabled&&
(codingTargetSelection=null,syncCodingModeUi(!0,{persist:!1})),userAutoScroll&&scrollToBottom(),document.
querySelectorAll(".message-group").length<=2||!currentThreadTitle||currentThreadTitle==="New Chat"||
currentThreadTitle==="No Title"?apiFetch("/api/generate_title",{method:"POST",headers:{"Content-Type":"\
application/json"},body:JSON.stringify({thread_id:currentThreadId,model_id:get("model-select").value})}).
then(Fe=>Fe.json()).then(Fe=>{Fe.title&&(document.title=Fe.title+" - AI Chat",setCurrentChatHeaderTitle(
Fe.title),loadThreads())}):loadThreads(!1)}catch(J){(Yt=window.ActivityLog)==null||Yt.log(J.name==="\
AbortError"?"send.cancelled":"send.error"),ke();let we=!1;const je=J.name==="AbortError"&&isManualStopAbortForThread(
Ke);if(J.name==="AbortError"&&!je&&(we=await syncThreadAfterAbortedStream(Ke,{retries:2,retryDelayMs:180,
notifyOnFailure:!0})),sendClientDebugLog("error",`Prompt send error: ${J.message}`),!Tt){G&&G.remove();
const qe=ee&&ee.closest(".fade-in");qe&&qe.remove(),delete messageStore[V],delete messageMeta[V]}if(J.
serverCode==="request_already_accepted"&&J.acceptedJobId&&J.acceptedThreadId)Tt=!0,jt={job_id:J.acceptedJobId,
thread_id:String(J.acceptedThreadId),model:Q.model,pendingStartedAt:l?null:t},ue.accept();else if(Tt&&
!je)Ct={job_id:normalizeJobIdForUi(currentJobId),thread_id:currentThreadId!=null?String(currentThreadId):
null,model:Q.model,pendingStartedAt:l?null:t},window.ConnectionMonitor.setUnavailable("offline"),showToast(
"\u56DE\u7B54\u3078\u306E\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u30D0\u30C3\u30AF\u30B0\u30E9\u30A6\u30F3\u30C9\u51E6\u7406\u3078\u81EA\u52D5\u518D\u63A5\u7D9A\u3057\u307E\u3059\u3002",
"warning",!1);else if(J.serverCode==="turnstile_required"){const qe=await getTurnstileToken();qe?(await verifyTurnstileOnServer(
qe,!0),showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"warning",!1)):showToast("\u5B89\u5168\u6027\u306E\u78BA\u8A8D\u3092\u5B8C\u4E86\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u3057\u3070\u3089\u304F\u5F85\u3063\u3066\u304B\u3089\u518D\u9001\u4FE1\u3057\u3066\u304F\u3060\u3055\u3044\u3002",
"error",!0)}else if(J.serverCode==="api_key_missing"){const qe=J.serverModel||Q.model,Xe=await showApiKeyRequiredModalAsync(
qe);Xe==="set"?on=!0:Xe==="switch"?showModal("model-modal"):showToast(J.message||`${getModelNameById(
qe)} \u306EAPI\u30AD\u30FC\u304C\u8A2D\u5B9A\u3055\u308C\u3066\u3044\u307E\u305B\u3093`,"error",!0)}else if(J.
name!=="AbortError"){const qe="Connection Error: "+J.message;showToast(qe,"error",!0)}if(!Tt){const qe=j&&
ue.canRetry()&&!editingMessageId;ue.restore(),qe&&(editingMessageId=j,currentParentId=oe,promptSettingsBeforeEdit=
ne,setEditUi(!0))}j&&!we&&O()}finally{nt&&nt(),rn&&window.ConnectionMonitor&&window.ConnectionMonitor.
operationEnded(),xt&&xt(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===Be&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,setEditUi(!!editingMessageId)}if(jt){const J=currentThreadId!=
null?String(currentThreadId):null;return currentThreadId=jt.thread_id,(J!==currentThreadId||location.
pathname!=="/c/"+currentThreadId)&&history.pushState({},"","/c/"+currentThreadId),reconnectPendingStreamUntilAvailable(
jt,currentThreadId)}if(Ct&&Ct.thread_id)return reconnectPendingStreamUntilAvailable(Ct,Ct.thread_id);
if(on&&ue.canRetry())return sendMessageLogged({animate:!1})}r(sendMessageLogged,"sendMessageLogged");
async function resumePendingStream(e){var ne,oe,te,ge;if(abortController||!e||!e.job_id||!currentThreadId||
isPendingJobSuppressed(e.job_id))return;const t=e.job_id,n=`pending-${t}`,i=e&&e.model?String(e.model):
"";get(n)||renderPendingMessage(get("chat-container"),!0,!0,n,i);const a=get(n);if(!a)return;if(activeStreamingBubbleId=
n,a.classList.add("ai-pending-bubble"),!a.querySelector(".content-area.skeleton-pending")){const $=a.
querySelector(".content-area");$?$.outerHTML=buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."):
a.insertAdjacentHTML("afterbegin",buildPendingSkeletonHtml(i,"\u56DE\u7B54\u3092\u751F\u6210\u4E2D..."))}
const o=startPendingElapsed(a,e.pendingStartedAt);currentJobId=t,a&&a.setAttribute("data-job-id",String(
t||"")),setSendBtnToStopMode(),resumeChatAutoScroll(),canvasModeEnabled&&resetCanvasPreviewPanel(),abortController=
new AbortController;const l=currentThreadId,c=i.toLowerCase(),d=c.includes("gemini")||c.includes("o1")||
c.includes("o3")||c.includes("gpt-5")||c.includes("reasoning")&&!c.includes("non-reasoning");let m=null;
const g=r($=>!d||!a?null:((!m||!a.contains(m))&&(m=a.querySelector(".thought-content")),m||(a.insertAdjacentHTML(
"afterbegin",'<div class="thought-container"><div class="thought-header thinking-shimmer" onclick="t\
oggleThinking(this)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="\
thought-content collapsed" data-placeholder="1"></div></div>'),m=a.querySelector(".thought-content")),
m&&(m.setAttribute("data-placeholder","1"),m.textContent=$||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D..."),
m),"ensureThoughtPlaceholder");d&&g("\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");
let b="",v="",w="",x=!0,k=null,S=null,C=null,M=!1;const E={};let q=0,N=!1;const W=window.ProgressSpinner?
window.ProgressSpinner.startFlow("chatResume"):null;let I=!1,j=!1;try{const $=await apiFetch("/chat_\
stream_resume",manualSpinnerRequestOptions({method:"POST",headers:{"Content-Type":"application/json"},
body:JSON.stringify({thread_id:currentThreadId,job_id:t,turnstile_token:botTurnstileTokenForRequest()}),
signal:abortController.signal}));if(!$.ok)throw new Error(`Resume failed (${$.status})`);window.ConnectionMonitor&&
(j=!0,window.ConnectionMonitor.operationStarted()),W&&W.setPhase("waiting"),(ne=window.ActivityLog)==
null||ne.log("send.stream.start");let O=!1;const V=$.body.getReader(),B=new TextDecoder;for(;!N;){const{
done:G,value:ae}=await V.read();if(!G&&!O&&(O=!0,(oe=window.ActivityLog)==null||oe.log("send.stream.\
first")),G&&((te=window.ActivityLog)==null||te.log("send.stream.done")),G)break;window.ConnectionMonitor&&
window.ConnectionMonitor.reportActivity(),W&&W.setPhase("receiving"),b+=B.decode(ae,{stream:!0});let R=b.
split(`
`);b=R.pop();let F=!1,me=!1;for(let re of R)if(re.trim())try{const X=JSON.parse(re);if(X.type==="job\
_id"){currentJobId=X.content||t,a&&a.setAttribute("data-job-id",String(currentJobId||""));continue}if(X.
type==="search_status"){C=updateSearchBox(a,C,X.content);continue}if(X.type==="mcp"){handleMcpStreamEvent(
a,X.content||{});continue}if(X.type==="mcp_decision_request"){openMcpDecisionModal(X.content||{});continue}
if(X.type==="status"){const ce=X.content===null||X.content===void 0?"":String(X.content);if(x&&a){const Q=ce||
"\u30E2\u30C7\u30EB\u51E6\u7406\u4E2D...";if(!updatePendingSkeletonStatus(a,Q,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059")){
const ye=a.querySelector(".content-area");ye&&(ye.outerHTML=buildPendingSkeletonHtml(i,Q),updatePendingSkeletonStatus(
a,Q,"\u5FDC\u7B54\u958B\u59CB\u307E\u3067\u306E\u9032\u6357\u3092\u8868\u793A\u3057\u3066\u3044\u307E\u3059"))}}
d&&g(ce||"\u63A8\u8AD6\u30D7\u30ED\u30BB\u30B9\u3092\u6E96\u5099\u4E2D...");continue}if(x){beginPendingToStreamTransition(
a);const ce=a.querySelector(".content-area");ce&&(ce.innerHTML=""),x=!1}if(X.type==="coding_diff")appendCodingLiveDiff(
a,X.content||{});else if(X.type==="thought"){if(k||(k=a.querySelector(".thought-content")),w+=X.content,
!k){const ce='<div class="thought-container"><div class="thought-header" onclick="toggleThinking(thi\
s)"><i class="fas fa-brain text-purple-400"></i> Thinking Process</div><div class="thought-content">\
</div></div>';C?C.insertAdjacentHTML("afterend",ce):a.insertAdjacentHTML("afterbegin",ce),k=a.querySelector(
".thought-content")}if(k&&k.getAttribute("data-placeholder")==="1"){if(k.textContent="",k.removeAttribute(
"data-placeholder"),k){const ce=k.parentElement.querySelector(".thought-header");ce&&ce.classList.remove(
"thinking-shimmer")}w=X.content}k.classList.remove("collapsed"),me=!0}else if(X.type==="image_analys\
is"){const ce=X.content===null||X.content===void 0?"":String(X.content);if(!a)continue;let Q=a.querySelector(
".image-analysis-box");if(!Q){const ue='<div class="image-analysis-box mb-2 p-2 bg-blue-900/20 borde\
r border-blue-500/30 rounded"><div class="text-[10px] text-blue-300 font-medium mb-1"><i class="fas \
fa-image mr-1"></i>Image Analysis</div><div class="image-analysis-text text-[11px] text-gray-300"></\
div></div>';C?C.insertAdjacentHTML("afterend",ue):a.insertAdjacentHTML("afterbegin",ue),Q=a.querySelector(
".image-analysis-box")}const ye=Q.querySelector(".image-analysis-text");ye&&(ye.textContent=ce)}else if(X.
type==="python"){const ce=X.content||{},Q=ce.id||`py_${Date.now()}`;if(!E[Q]){const ue=`<div class="\
code-wrapper python-box collapsed" data-py-id="${Q}" data-collapsed="true" data-code-key="${Q}"><div\
 class="code-header"><span class="code-lang"><i class="fas fa-terminal"></i> Python Execution</span>\
<div class="code-actions"><button class="code-toggle" aria-expanded="false" title="\u5C55\u958B" aria-label="\u5C55\
\u958B"><i class="fas fa-chevron-down"></i></button><button class="copy-btn" data-copy="code" data-code="\
" title="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC" aria-label="\u30B3\u30FC\u30C9\u3092\u30B3\u30D4\u30FC"><i class="fas fa-copy"></i></button><button class="copy-btn" \
data-copy="output" data-code="" title="\u51FA\u529B\u3092\u30B3\u30D4\u30FC" aria-label="\u51FA\u529B\u3092\u30B3\u30D4\u30FC"><i class="fas fa-align-left"></i>\
</button></div></div><div class="code-body"><div class="python-section"><div class="python-label">Co\
de</div><pre><code class="hljs language-python python-code"></code></pre></div><div class="python-se\
ction"><div class="python-label">Output</div><pre><code class="hljs language-plaintext python-output\
"></code></pre></div></div></div>`;C?C.insertAdjacentHTML("afterend",ue):a.insertAdjacentHTML("after\
begin",ue),E[Q]=a.querySelector(`[data-py-id="${Q}"]`)}const ye=E[Q];if(ye){if(ce.code!==void 0){const ue=ce.
code==null?"":String(ce.code),ke=ye.querySelector(".python-code");ke&&(ke.textContent=ue,ke.removeAttribute(
"data-highlighted"),queueHighlight(ye,ue));const Be=ye.querySelector('.copy-btn[data-copy="code"]');
Be&&Be.setAttribute("data-code",encodeURIComponent(ue).replace(/'/g,"%27"))}if(ce.output!==void 0){const ue=ce.
output==null?"":String(ce.output),ke=ye.querySelector(".python-output");ke&&(ke.textContent=ue);const Be=ye.
querySelector('.copy-btn[data-copy="output"]');Be&&Be.setAttribute("data-code",encodeURIComponent(ue).
replace(/'/g,"%27"))}}}else if(X.type==="content"){const ce=X.content===null||X.content===void 0?"":
String(X.content);v+=ce,/[`~]/.test(ce)&&activateDeferredCodingModeFromStream(v),S||(S=a.querySelector(
".content-area")||document.createElement("div"),S.className="prose prose-invert text-sm break-words",
a.contains(S)||a.appendChild(S)),F=!0}else if(X.type==="error"){M=!0,N=!0,a.insertAdjacentHTML("befo\
reend",buildChatErrorBubbleHtml(X.content)),showToast(X.content||"Unknown error","error",!0);break}}catch{}
if(me&&k&&(k.textContent=w,userAutoScroll&&(k.scrollTop=k.scrollHeight)),F&&S){const re=Date.now();if(re-
q>100){const X=snapshotCodeCollapse(S);renderAiMarkdownInto(S,v,{incrementalMath:!0,streaming:!0,mathSteps:{
jobId:t,streaming:!0}}),applyCodeCollapse(S,X,!0),q=re}}scrollToBottom()}if(W&&W(),S){const G=snapshotCodeCollapse(
S);renderAiMarkdownInto(S,v,{incrementalMath:!0,streaming:!1,mathSteps:{jobId:currentJobId||t,streaming:!1}}),
applyCodeCollapse(S,G,!0)}vibrateHelper([100,50,100]),a&&queueHighlight(a,v),a&&a.querySelectorAll("\
.thought-content").forEach(ae=>ae.classList.add("collapsed")),await loadMessages(currentThreadId,{preserveDraft:!0,
silent:!0}),loadThreads(!1)}catch($){(ge=window.ActivityLog)==null||ge.log($.name==="AbortError"?"se\
nd.cancelled":"send.stream.error");const O=$.name==="AbortError"&&isManualStopAbortForThread(l);$.name===
"AbortError"&&!O&&await syncThreadAfterAbortedStream(l,{retries:2,retryDelayMs:180,notifyOnFailure:!0}),
O||(I=!0,window.ConnectionMonitor.setUnavailable("offline"),showToast("\u56DE\u7B54\u3078\u306E\u518D\u63A5\u7D9A\u304C\u5207\u308C\u307E\u3057\u305F\u3002\u81EA\u52D5\u7684\u306B\u518D\u8A66\u884C\u3057\u307E\u3059\u3002",
"warning",!1))}finally{o&&o(),j&&window.ConnectionMonitor&&window.ConnectionMonitor.operationEnded(),
W&&W(),setSendBtnToSendMode(),updateFilePreview(),activeStreamingBubbleId===n&&(activeStreamingBubbleId=
null),abortController=null,currentJobId=null,currentThreadPending=null}if(I)return reconnectPendingStreamUntilAvailable(
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
-1,d=o.querySelectorAll("[data-thread-id]").length;if(c===0&&d>0&&String(n||"").trim()){snapshotSidebarHistory(
"loadThreads-keep-existing-empty-search");return}if(o.innerHTML='<div id="thread-pull-indicator" cla\
ss="ptr-pull-indicator" aria-hidden="true"><i class="fas fa-arrow-down ptr-pull-icon"></i><i class="\
fas fa-spinner fa-spin ptr-pull-spinner"></i><span class="ptr-pull-label"></span></div><div id="scro\
ll-sentinel"></div>',threadObserver){threadObserver.disconnect();const m=get("scroll-sentinel");m&&threadObserver.
observe(m)}}const l=get("scroll-sentinel");a&&Array.isArray(a.threads)?(a.threads.forEach(c=>{const d=String(
c.id),m=document.createElement("div"),g=c.is_bookmarked?"text-yellow-400":"text-gray-500",b=c.is_temporary?
'<span class="text-[9px] text-amber-300 border border-amber-500/50 rounded px-1 py-0">\u4E00\u6642</span>':
"",w=d===String(currentThreadId)?"bg-gray-700/60 border-l-2 border-blue-500":"";m.className=`p-2 rou\
nded hover:bg-gray-700 cursor-pointer text-sm text-gray-300 truncate flex justify-between items-cent\
er group ${w}`,m.dataset.threadId=d,m.dataset.threadTitle=c.title||"",m.innerHTML=`<div class="flex \
items-center gap-1 truncate flex-1"><button class="${g} hover:text-yellow-400 px-1" onclick="toggleB\
ookmark(event, '${d}')"><i class="fas fa-star text-[10px]"></i></button><span class="truncate">${escapeHtml(
c.title||"No Title")}</span>${b}</div><div class="flex items-center gap-1 opacity-100 md:opacity-0 m\
d:group-hover:opacity-100 transition" data-thread-actions="1"><button class="text-gray-500 hover:tex\
t-white px-1 transition" onclick="renameThread(event, '${d}')"><i class="fas fa-pen text-xs"></i></b\
utton><button class="text-gray-500 hover:text-red-400 px-1 transition" onclick="deleteThread(event, \
'${d}')"><i class="fas fa-trash text-xs"></i></button></div>`,m.onclick=x=>{x.target.closest("button")||
x.target.closest("[data-thread-actions]")||loadMessages(d)},l?o.insertBefore(m,l):o.appendChild(m)}),
hasMoreThreads=!!a.has_next,hasMoreThreads&&threadPage++,snapshotSidebarHistory("loadThreads-rendere\
d count="+a.threads.length+" append="+!!e)):snapshotSidebarHistory("loadThreads-empty-or-invalid")}catch(t){
console.error("Failed to load threads:",t),snapshotSidebarHistory("loadThreads-error")}finally{threadLoading=
!1,updateThreadHighlighting(),snapshotSidebarHistory("loadThreads-finally")}}r(loadThreads,"loadThre\
ads");function initPullToRefresh(e,t){const n=get(e);if(!n)return;const i=`${e}-pull-indicator`,a=60,
o=88,l=52,c=.5,d=8;let m=0,g=!1,b=0,v=null;const w=r(()=>get(i),"indicatorEl"),x=r(()=>{const C=w();
return C?C.querySelector(".ptr-pull-label"):null},"labelEl"),k=r(C=>{const M=w();if(!M)return;M.style.
height=Math.min(C,o)+"px",M.classList.toggle("active",C>2),M.classList.toggle("pull-ready",C>=a);const E=x();
E&&(E.textContent=C>=a?"\u96E2\u3057\u3066\u66F4\u65B0":"\u5F15\u3063\u5F35\u3063\u3066\u66F4\u65B0")},
"applyPullUI"),S=r(()=>{const C=w();C&&(C.style.height="0px",C.classList.remove("active","pull-ready",
"refreshing"),C.classList.remove("dragging"))},"resetPullUI");n.addEventListener("touchstart",C=>{if(v){
g=!1;return}if(n.scrollTop>0){g=!1;return}const M=C.touches[0];M&&(m=M.clientY,b=0,g=!0)},{passive:!0}),
n.addEventListener("touchmove",C=>{if(!g||v)return;if(n.scrollTop>0){g=!1;return}const M=C.touches[0];
if(!M)return;const E=M.clientY-m;if(E<=0){b>0&&(b=0,k(0)),g=!1;return}const q=w();q&&!q.classList.contains(
"dragging")&&q.classList.add("dragging"),b=Math.min(E*c,o),k(b),E>=d&&C.preventDefault()},{passive:!1}),
n.addEventListener("touchend",()=>{if(!g||(g=!1,v))return;const C=w();C&&C.classList.remove("draggin\
g");const M=b>=a;if(b=0,!M){S();return}let E;try{E=t()}catch{E=null}const q=w();if(q){q.classList.add(
"refreshing"),q.style.height=l+"px";const N=q.querySelector(".ptr-pull-label");N&&(N.textContent="\u66F4\u65B0\
\u4E2D...")}E&&typeof E.then=="function"?(v=E,E.catch(()=>{}).finally(()=>{v=null,S()})):(v=Promise.
resolve(),setTimeout(()=>{v=null,S()},400))}),n.addEventListener("touchcancel",()=>{g=!1,b=0,S()})}r(
initPullToRefresh,"initPullToRefresh");const initThreadPullToRefresh=r(()=>initPullToRefresh("thread\
-list",()=>loadThreads(!1)),"initThreadPullToRefresh"),initSkillPullToRefresh=r(()=>initPullToRefresh(
"skill-list",()=>loadSkills()),"initSkillPullToRefresh"),initPullToRefreshAll=r(()=>{initThreadPullToRefresh(),
initSkillPullToRefresh()},"initPullToRefreshAll");let activeMcpDecision=null,mcpDecisionModalBound=!1;
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
    <span class="mcp-box-sub">\u5931\u6557</span>`;else{const d=`<div class="mcp-box mcp-error mb-2"\
 data-mcp-card="${a}">
    <i class="fas fa-times-circle mcp-box-err"></i>
    <span class="mcp-box-title">${mcpCardTitle(t)}</span>
    <span class="mcp-box-sub">\u5931\u6557</span>
</div>`;i.insertAdjacentHTML("beforeend",d),o=i.querySelector('[data-mcp-card="'+a+'"]')}const c=document.
createElement("div");c.className="mcp-box-note mcp-box-note-err",c.textContent=String(l).slice(0,300),
o&&o.appendChild(c);return}if(t.type==="decision_resolved"){if(activeMcpDecision&&activeMcpDecision.
id&&t.id&&activeMcpDecision.id===t.id){const o=get("mcp-decision-modal");if(o&&!o.classList.contains(
"hidden"))try{hideModal("mcp-decision-modal")}catch{}activeMcpDecision=null}return}}r(handleMcpStreamEvent,
"handleMcpStreamEvent");function openMcpDecisionModal(e){if(!get("mcp-decision-modal")||!e||activeMcpDecision&&
activeMcpDecision.id===e.id)return;activeMcpDecision={id:e.id||null,jobId:currentJobId||null};const n=get(
"mcp-decision-server"),i=get("mcp-decision-tool"),a=get("mcp-decision-args");if(n&&(n.textContent=e.
server_name||"\u4E0D\u660E\u306A\u30B5\u30FC\u30D0\u30FC"),i&&(i.textContent=e.tool_name||""),a){let c=e.
args_preview||"";try{const d=JSON.parse(c);c=JSON.stringify(d,null,2)}catch{}a.textContent=c}const o=get(
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
{scroll:!1});const l=o?snapshotCodeCollapseByMessage(get("chat-container")):null;let c="",d="",m=[];
if(a){const b=get("prompt-input");c=b?b.value:"",d=b?b.style.height:"",m=currentImageUrls?currentImageUrls.
slice():[],editingMessageId=null,promptSettingsBeforeEdit=null,setEditUi(!1)}else cancelEdit();o||playChatTransition(
"history");const g=e!=null?String(e):e;g!==currentThreadId&&(modelChosenManually=!1),currentThreadId=
g,t.skipHistory||history.pushState({},"","/c/"+e),updateThreadHighlighting(),syncActiveSkillForThread(
currentThreadId),get("welcome-screen").classList.add("hidden"),o||(get("chat-container").innerHTML=buildChatLoadingSkeletonHtml());
try{const b=new URL(CHAT_CONFIG.urls.handleThreadItem.replace("0",e),window.location.origin);b.searchParams.
set("limit",String(getEffectiveThreadInitialMessageLimit()));const v=i;v&&/^\d+$/.test(String(v))&&b.
searchParams.set("before_id",String(Number(v)+1));const w=await apiFetch(b.toString());if(!w.ok)throw new Error(
`thread request failed (${w.status})`);const x=await w.json();if(!x||!Array.isArray(x.messages))throw new Error(
"invalid thread response");if(n!==threadLoadSequence)return!1;setCurrentChatHeaderTitle(x&&x.title),
allMessages=x.messages,renderBranchOrigin(x.branch_origin),threadHasOlderMessages=!!x.has_older_messages,
oldestLoadedMessageId=x.oldest_loaded_id||(allMessages.length?allMessages[0].id:null);const k=(allMessages||
[]).filter(M=>M.role==="user"&&M.content).map(M=>M.content);if(promptHistory=[...new Set(k.slice().reverse())],
historyIndex=-1,tempPrompt="",currentThreadPending=x.pending_job||null,setTemporaryChatUiState(!!(x&&
x.is_temporary)),applyTemporaryChatRuntimeMeta(x||{}),ensureTemporaryChatHeartbeat(!0),get("thread-c\
ustom-instruction")&&(get("thread-custom-instruction").value=x.custom_instruction||""),get("enable-p\
rompt-cache")&&(get("enable-prompt-cache").checked=!!x.enable_prompt_caching,updatePromptCacheUi()),
x.last_model&&!modelChosenManually&&selectModelById(x.last_model),x.last_skill_uuid&&loadedSkills.length>
0){const M=loadedSkills.find(E=>E.uuid===x.last_skill_uuid);M&&(threadSkillMap[currentThreadId]=M,applyActiveSkill(
M))}const S=t.forceLatestLeaf?null:localStorage.getItem(`fixed_branch_${currentThreadId}`),C=i;if(C&&
allMessages.some(M=>String(M.id)===String(C))?currentLeafId=Number(C):S&&allMessages.find(M=>String(
M.id)===String(S))?currentLeafId=S:allMessages.length>0?currentLeafId=allMessages[allMessages.length-
1].id:currentLeafId=null,renderThreadTree(o?{silent:o,keepScroll:o}:{silent:o,keepScroll:o,animate:!0}),
a||restoreLatestPromptOptions(),o&&l?applyCodeCollapseByMessage(get("chat-container"),l,!0):o||applyCodeCollapseByMessage(
get("chat-container"),null,!0),currentThreadPending&&!o&&!isPendingJobSuppressed(currentThreadPending.
job_id)&&resumePendingStream(currentThreadPending),a){const M=get("prompt-input");M&&(M.value=c||"",
d?M.style.height=d:M.style.height="auto"),currentImageUrls=m,currentImageUrls&&currentImageUrls.length?
(get("file-preview").classList.remove("hidden"),get("file-name").innerText=`${currentImageUrls.length}\
 files ready`):get("file-preview").classList.add("hidden"),schedulePromptTokenEstimate(!0)}if(i&&new URLSearchParams(
location.search).has("message")&&history.replaceState({},"",location.pathname),a||schedulePromptTokenEstimate(
!0),window.innerWidth<768&&get("overlay").click(),typeof window.__refreshAdminThreadEncState=="funct\
ion")try{window.__refreshAdminThreadEncState()}catch{}return!0}catch(b){return n!==threadLoadSequence||
(console.error("Failed to load chat thread:",b),o||showChatLoadError(e),o||showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0)),!1}}r(loadMessages,"loadMessages");async function loadOlderMessages(){if(loadingOlderMessages||
!currentThreadId||!threadHasOlderMessages||!oldestLoadedMessageId)return;loadingOlderMessages=!0;const e=get(
"chat-container"),t=e?e.scrollHeight:0,n=e?e.scrollTop:0;try{const i=new URL(CHAT_CONFIG.urls.handleThreadItem.
replace("0",currentThreadId),window.location.origin);i.searchParams.set("before_id",String(oldestLoadedMessageId)),
i.searchParams.set("limit",String(getEffectiveThreadOlderPageSize())),i.searchParams.set("include_me\
ta","0");const o=await(await apiFetch(i.toString())).json(),l=Array.isArray(o.messages)?o.messages:[];
if(l.length){const c=new Set(allMessages.map(m=>m.id)),d=l.filter(m=>!c.has(m.id));d.length&&(allMessages=
d.concat(allMessages))}if(threadHasOlderMessages=!!o.has_older_messages,oldestLoadedMessageId=o.oldest_loaded_id||
(allMessages.length?allMessages[0].id:null),renderThreadTree({silent:!0,keepScroll:!0}),e){const c=e.
scrollHeight;e.scrollTop=Math.max(0,n+(c-t))}}catch{showToast("\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u306E\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}finally{loadingOlderMessages=!1;const i=get("load-older-messages-btn");i&&threadHasOlderMessages&&
(i.disabled=!1,i.innerHTML='<i class="fas fa-clock-rotate-left mr-1"></i>\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080')}}
r(loadOlderMessages,"loadOlderMessages");function renderThreadTree(e={}){const t=!!e.silent,n=!!e.animate&&
!t,i=!!e.keepScroll,a=get("chat-container");if(!a)return;const o=i?Array.from(a.children):[],l=o.map(
k=>k.getBoundingClientRect().height);let c=null;if(i&&(c=a.scrollTop),allMessages.length===0){a.replaceChildren(),
currentParentId=null,updateTotalTokenBar(0);return}const d={};allMessages.forEach(k=>{d[k.id]=k,k.childrenIds=
[]}),allMessages.forEach(k=>{k.parent_id&&d[k.parent_id]&&d[k.parent_id].childrenIds.push(k.id)}),(!currentLeafId||
!d[currentLeafId])&&(currentLeafId=allMessages.length>0?allMessages[allMessages.length-1].id:null);const m=[];
let g=d[currentLeafId];for(;g;)m.unshift(g),g=d[g.parent_id];const b=buildTokenTotals(m),v=document.
createDocumentFragment();if(threadHasOlderMessages){const k=loadingOlderMessages?"\u8AAD\u307F\u8FBC\u307F\u4E2D...":
"\u904E\u53BB\u30E1\u30C3\u30BB\u30FC\u30B8\u3092\u8AAD\u307F\u8FBC\u3080",S=loadingOlderMessages?"d\
isabled":"",C=document.createElement("div");C.className="mb-3 text-center",C.innerHTML=`<button id="\
load-older-messages-btn" class="px-3 py-1.5 text-xs rounded border border-gray-600 text-gray-200 hov\
er:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed" onclick="loadOlderMessages()" ${S}><\
i class="fas fa-clock-rotate-left mr-1"></i>${k}</button>`,v.appendChild(C)}m.forEach(k=>{const S=k.
parent_id?d[k.parent_id]:null,C=S?S.childrenIds:allMessages.filter(E=>!E.parent_id).map(E=>E.id),M=C.
length>1?{current:C.indexOf(k.id)+1,total:C.length,siblings:C}:null;renderMessage(k.id,k.role,k.content,
k.image_url,k.thought_data,k.model,M,n,k.quote_text,k.tokens,k.tokens_in,k.tokens_out,k.is_encrypted,
k.tokens_content,k.tokens_thought,v,!1,k.parent_id,k.skill_name,k.batch_job,k.prompt_options&&typeof k.
prompt_options=="object"?k.prompt_options.image_mask:null,k.processing_time,k.math_steps_state)});const w=currentThreadPending;
if(w&&!isPendingJobSuppressed(w.job_id)){const k=w.message_id,S=new Set(m.map(E=>E.id)),C=m.length?m[m.
length-1]:null;if(k&&S.has(k)&&currentLeafId===k||!k&&C&&C.role==="user"){const E=w.job_id?`pending-${w.
job_id}`:null;renderPendingMessage(v,n,!1,E,w.model||null)}}const x=[];if(i&&!lowBandwidthMode&&Array.
from(v.children).forEach((k,S)=>{const C=o.find(E=>E.id&&E.id===k.id)||(o.length===v.children.length?
o[S]:null),M=C?l[o.indexOf(C)]:0;M>0&&(k.style.minHeight=`${M}px`,x.push(k))}),a.replaceChildren(v),
displayMissingMessageFiles(a,m),updateTotalTokenBar(b.tokens_total,b),currentParentId=currentLeafId,
i&&c!==null?restoreThreadTreeScroll(a,c):scrollToBottom(),lowBandwidthMode)queueMessageDecorations(a,
a&&a.textContent||"");else if(queueHighlight(a),m.length){const k=queueMathTypeset(a,m.map(S=>S.content||
"").join(`
`));Promise.resolve(k).finally(()=>{if(!x.length||!x[0].isConnected)return;const S=a.scrollTop;x.forEach(
C=>{C.style.minHeight=""}),restoreThreadTreeScroll(a,S)})}}r(renderThreadTree,"renderThreadTree");function restoreThreadTreeScroll(e,t){
if(!e)return;const n=e.scrollHeight-e.clientHeight;userAutoScroll&&!chatManualPauseIntent?e.scrollTop=
e.scrollHeight:e.scrollTop=Math.max(0,Math.min(t,n)),chatLastScrollTop=e.scrollTop,syncScrollToBottomButton()}
r(restoreThreadTreeScroll,"restoreThreadTreeScroll");function switchVersion(e){currentLeafId=e;const t={};
allMessages.forEach(i=>{t[i.id]=i,i.childrenIds=[]}),allMessages.forEach(i=>{i.parent_id&&t[i.parent_id]&&
t[i.parent_id].childrenIds.push(i.id)});let n=e;if(!t[n]){currentLeafId=allMessages.length>0?allMessages[allMessages.
length-1].id:null,renderThreadTree({animate:!0}),restoreLatestPromptOptions();return}for(;t[n]&&t[n].
childrenIds.length>0;){const i=t[n].childrenIds;n=Math.max(...i)}currentLeafId=n,renderThreadTree({animate:!0}),
restoreLatestPromptOptions()}r(switchVersion,"switchVersion");async function loadSkills(){try{const t=await(await apiFetch(
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
r(loadSkills,"loadSkills");async function openEditSkillModal(e,t){e.stopPropagation(),editingSkillUuid=
t;try{const i=await(await apiFetch(`/api/skills/${t}`)).json();get("skill-name").value=i.name,get("s\
kill-desc").value=i.description||"",get("skill-inst").value=i.instruction,get("skill-modal-title").textContent=
"Edit Skill",get("save-skill-btn").innerText="Save Changes",showModal("skill-modal"),location.pathname!==
"/skill"&&history.pushState({modal:"skill"},"","/skill")}catch{showToast("Skill\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}}r(openEditSkillModal,"openEditSkillModal");async function createSkill(e,t){await apiFetch(
CHAT_CONFIG.urls.handleSkills,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(
{name:e,instruction:t})}),loadSkills()}r(createSkill,"createSkill");function applyActiveSkill(e){activeSkill=
e||null,activeSkill?(get("active-skill-name").innerText=activeSkill.name,get("skill-active-indicator").
classList.remove("hidden")):get("skill-active-indicator").classList.add("hidden");const t=get("sys-p\
rompt-option");t&&(t.style.opacity="1")}r(applyActiveSkill,"applyActiveSkill");function syncActiveSkillForThread(e){
const t=e&&threadSkillMap[e]?threadSkillMap[e]:null;applyActiveSkill(t)}r(syncActiveSkillForThread,"\
syncActiveSkillForThread");async function saveThreadSkillUuid(e,t){try{await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({last_skill_uuid:t,
thread_id:e})})}catch{}}r(saveThreadSkillUuid,"saveThreadSkillUuid");function activateSkill(e,t){currentThreadId?
(threadSkillMap[currentThreadId]=e,applyActiveSkill(e),showToast(`Skill "${e.name}" \u3092\u3053\u306E\u30C1\u30E3\u30C3\u30C8\u306B\u9069\u7528\u3057\u307E\u3057\u305F`,
"success"),t||saveThreadSkillUuid(currentThreadId,e?e.uuid:null)):(pendingSkillForNewThread=e,applyActiveSkill(
e),allMessages&&allMessages.length>0&&startNewChat({preserveSkill:!0}))}r(activateSkill,"activateSki\
ll");function clearActiveSkill(){currentThreadId&&(delete threadSkillMap[currentThreadId],saveThreadSkillUuid(
currentThreadId,null)),pendingSkillForNewThread=null,applyActiveSkill(null)}r(clearActiveSkill,"clea\
rActiveSkill");function getCurrentChatHeaderTitleText(){return typeof currentThreadTitle=="string"&&
currentThreadTitle.trim()?currentThreadTitle.trim():currentThreadId?"No Title":"AI Chat"}r(getCurrentChatHeaderTitleText,
"getCurrentChatHeaderTitleText");function getTemporaryChatTimeoutLabel(){return temporaryChatEnabled?
`${normalizeTemporaryChatTimeoutSeconds(temporaryChatTimeoutSeconds)}\u79D2`:""}r(getTemporaryChatTimeoutLabel,
"getTemporaryChatTimeoutLabel");function updateCurrentChatHeaderUi(){const e=getCurrentChatHeaderTitleText(),
t=getTemporaryChatTimeoutLabel(),n=!!temporaryChatEnabled,i=["sidebar-chat-title","mobile-chat-title"],
a=["sidebar-chat-temporary-label","mobile-chat-temporary-label"],o=["sidebar-chat-ttl","mobile-chat-\
ttl"];i.forEach(l=>{const c=get(l);c&&(c.textContent=e)}),a.forEach(l=>{const c=get(l);c&&c.classList.
toggle("hidden",!n)}),o.forEach(l=>{const c=get(l);c&&(n&&t?(c.textContent=t,c.classList.remove("hid\
den")):(c.textContent="",c.classList.add("hidden")))})}r(updateCurrentChatHeaderUi,"updateCurrentCha\
tHeaderUi");function setCurrentChatHeaderTitle(e){currentThreadTitle=typeof e=="string"?e:null,updateCurrentChatHeaderUi()}
r(setCurrentChatHeaderTitle,"setCurrentChatHeaderTitle");function resetTemporaryChatExpiresAt(){tempChatExpiresAtMs=
null,updateCurrentChatHeaderUi()}r(resetTemporaryChatExpiresAt,"resetTemporaryChatExpiresAt");function applyTemporaryChatRuntimeMeta(e){
if(!e||typeof e!="object")return;Object.prototype.hasOwnProperty.call(e,"timeout_seconds")&&applyTemporaryChatTimeoutSeconds(
e.timeout_seconds);let t=null;const n=Number(e.temp_chat_expires_at);if(Number.isFinite(n)&&n>0)t=Math.
floor(n*1e3);else{const i=Number(e.temp_chat_remaining_seconds);Number.isFinite(i)&&i>=0&&(t=Date.now()+
Math.floor(i*1e3))}t!==null?tempChatExpiresAtMs=t:(e.is_temporary===!1||!temporaryChatEnabled)&&(tempChatExpiresAtMs=
null),updateCurrentChatHeaderUi()}r(applyTemporaryChatRuntimeMeta,"applyTemporaryChatRuntimeMeta");function ensureCurrentChatHeaderTicker(){}
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
d-custom-instruction").value=""),get("enable-skills")&&(get("enable-skills").checked=!0),get("enable\
-prompt-cache")&&(get("enable-prompt-cache").checked=!1,updatePromptCacheUi()),e.preserveSkill?activeSkill&&
applyActiveSkill(activeSkill):applyActiveSkill(null),loadThreads();const t=threadLoadSequence;fetchSettingsSnapshot().
then(n=>{if(threadLoadSequence!==t||currentThreadId||modelChosenManually||e.preserveSkill&&activeSkill)
return;const i=n.use_last_chat_settings&&n.last_model||n.default_model;i&&MODELS.some(a=>a.items.some(
o=>o.id===i))&&selectModelById(i)}).catch(n=>console.warn("Failed to load chat defaults",n)),window.
innerWidth<768&&get("overlay").click()}r(startNewChat,"startNewChat");let threadModalLoadSeq=0;window.
openThreadModal=async()=>{if(!currentThreadId)try{const i=await(await apiFetch(CHAT_CONFIG.urls.handleThreads,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({is_temporary:temporaryChatEnabled})})).
json();currentThreadId=i.id!==null&&i.id!==void 0?String(i.id):i.id,setTemporaryChatUiState(!!(i&&i.
is_temporary)),setCurrentChatHeaderTitle(i&&i.title),applyTemporaryChatRuntimeMeta(i||{}),ensureTemporaryChatHeartbeat(
!0),history.pushState({},"","/c/"+i.id),loadThreads()}catch{showToast("\u30C1\u30E3\u30C3\u30C8\u306E\u4F5C\u6210\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
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
!0,c=get("thread-global-sys-prompt"),d=get("thread-global-sys-prompt-enabled");let m=null;try{m=c||d?
{system_prompt:c?c.value:"",system_prompt_enabled:d?d.checked:!0,apply_global_system_prompt:get("thr\
ead-apply-global-sys-prompt")?get("thread-apply-global-sys-prompt").checked:!0,apply_auto_system_prompt_notices:get(
"thread-apply-auto-sys-prompt-notices")?get("thread-apply-auto-sys-prompt-notices").checked:!0,auto_system_prompt_notices_config:window.
collectAutoSystemPromptConfigFromForm("thread")}:null}catch(g){sendClientDebugLog("error","Payload c\
onstruction failed: "+g.message),showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0),t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58");return}try{sendClientDebugLog("info",
"Starting PUT request for thread: "+e);const g=await apiFetch(`/api/threads/${e}/settings`,{method:"\
PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({custom_instruction:a,include_global_instruction:l})});
sendClientDebugLog("info","PUT request finished, status: "+g.status);let b=!0;if(m){sendClientDebugLog(
"info","Starting POST request for user settings");const v=await apiFetch(CHAT_CONFIG.urls.handleSettings,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(m)});b=v.ok,v.ok&&window.
applySavedUserSystemPromptSettings(m),sendClientDebugLog("info","POST request finished, status: "+v.
status)}g.ok&&b?(window.closeThreadModal(),showToast("\u4FDD\u5B58\u3055\u308C\u307E\u3057\u305F","s\
uccess")):showToast("\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",!0)}catch(g){sendClientDebugLog(
"error","Save failed with error: "+g.message),showToast("\u30A8\u30E9\u30FC: "+g.message,"error",!0)}finally{
t&&(t.disabled=!1,t.textContent=n||"\u4FDD\u5B58")}},window.openCompressionModal=()=>{syncCompressionSettingsUi(),
showModal("compression-modal"),location.pathname!=="/compression"&&history.pushState({modal:"compres\
sion"},"","/compression")},window.closeCompressionModal=(e=!1)=>{hideModal("compression-modal"),!e&&
location.pathname==="/compression"&&history.back()},get("save-compression-settings-btn").onclick=()=>{
const e=get("compression-max-size").value,t=get("compression-max-dim").value,n=get("compression-outp\
ut-type").value,i=get("compression-format-only").checked;setCompressionSettings(e,t,n,i);const a=r((l,c)=>{
get(l)&&get(c)&&(get(c).value=get(l).value)},"syncBack");a("modal-gpt-image-size","gpt-image-size"),
a("modal-gpt-image-quality","gpt-image-quality"),a("modal-gpt-image-complexity","gpt-image-complexit\
y"),a("modal-gpt-image-format","gpt-image-format"),a("modal-gpt-image-compression","gpt-image-compre\
ssion"),a("modal-gpt-image-frame-count","gpt-image-frame-count"),a("modal-gpt-image-frame-duration",
"gpt-image-frame-duration"),a("modal-gemini-image-aspect","gemini-image-aspect"),a("modal-gemini-ima\
ge-size","gemini-image-size"),a("modal-grok-image-aspect","grok-image-aspect"),a("modal-grok-image-r\
esolution","grok-image-resolution"),a("modal-grok-image-quality","grok-image-quality"),IDEOGRAM_IMAGE_FIELDS.
forEach(l=>a(`modal-ideogram-image-${l}`,`ideogram-image-${l}`)),a("modal-ocr-table-format","ocr-tab\
le-format"),a("modal-ocr-pages","ocr-pages");const o=r((l,c)=>{get(l)&&get(c)&&(get(c).checked=get(l).
checked)},"syncBackChk");o("modal-ocr-extract-header","ocr-extract-header"),o("modal-ocr-extract-foo\
ter","ocr-extract-footer"),o("modal-ocr-include-blocks","ocr-include-blocks"),o("modal-ocr-include-i\
mages","ocr-include-images"),window.closeCompressionModal(),showToast("\u8A2D\u5B9A\u3092\u4FDD\u5B58\u3057\u307E\u3057\u305F",
"success")};async function deleteSkill(e,t){e.stopPropagation(),confirm("Delete?")&&(await apiFetch(
CHAT_CONFIG.urls.handleSkillItem.replace("0",t),{method:"DELETE"}),loadSkills())}r(deleteSkill,"dele\
teSkill");function renameThread(e,t){var b;e.stopPropagation();const n=get("thread-rename-modal"),i=get(
"thread-rename-form"),a=get("thread-rename-input"),o=get("thread-rename-save"),l=get("thread-rename-\
cancel"),c=get("thread-rename-error"),d=e.currentTarget;a.value=((b=d.closest("[data-thread-id]"))==
null?void 0:b.dataset.threadTitle)||"",c.textContent="",o.disabled=!1;let m=!1;const g=r(()=>{m||(hideModal(
"thread-rename-modal"),d.focus())},"close");l.onclick=g,n.onclick=v=>{v.target===n&&g()},n.onkeydown=
v=>{v.key==="Escape"&&(v.preventDefault(),v.stopPropagation(),g()),v.key==="Tab"&&(v.shiftKey&&document.
activeElement===a?(v.preventDefault(),l.focus()):!v.shiftKey&&document.activeElement===l&&(v.preventDefault(),
a.focus()))},i.onsubmit=async v=>{v.preventDefault();const w=a.value.trim();if(m||!w){w||a.focus();return}
m=!0,o.disabled=!0,l.disabled=!0;try{const x=await apiFetch(CHAT_CONFIG.urls.updateTitle.replace("0",
t),{method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:w})}),k=await x.
json();if(!x.ok)throw new Error("rename failed");currentThreadId===String(t)&&setCurrentChatHeaderTitle(
k.title||w),m=!1,g(),loadThreads()}catch{c.textContent="\u540D\u524D\u306E\u5909\u66F4\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002"}finally{
m=!1,o.disabled=!1,l.disabled=!1}},l.disabled=!1,showModal("thread-rename-modal"),a.focus()}r(renameThread,
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
n.url:buildPdfAttachmentUrl(i),d=n&&n.preview_url?n.preview_url:buildPdfAttachmentPreviewUrl(i);return{
path:i,filename:a,source:o,isImage:l,url:c,previewUrl:d}}).filter(Boolean),"buildPdfMessageAttachmen\
ts"),buildPdfDocumentHtml=r(e=>{const t=e&&e.thread?e.thread:{},n=Array.isArray(e&&e.messages)?e.messages:
[],a=n.some(d=>maybeNeedsMathJax(d.content)||maybeNeedsMathJax(d.thought_text))?`
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" as\
ync data-cfasync="false"><\/script>`:"",o=t.title||"AI Chat",l=[{label:"Exported At",value:pdfFormatTimestamp(
e&&e.generated_at)},{label:"Leaf Message",value:e&&e.leaf_id?`#${e.leaf_id}`:"none"},{label:"Message\
s",value:String(n.length)},{label:"Version",value:`AI Playground ${appVersion}`}],c=n.map(d=>{const m=d.
role==="user",g=d.quote_text?`<div class="quote"><strong>Quote</strong><br>${escapeHtml(d.quote_text)}\
</div>`:"",b=d.thought_text?`<div class="thought">${escapeHtml(d.thought_text)}</div>`:"",v=!m&&typeof window.
buildMathStepsPdfHtml=="function"?window.buildMathStepsPdfHtml(d.content||"",d.math_steps_state):"",
w=m?`<div class="content" style="white-space: pre-wrap;">${escapeHtml(d.content||"")}</div>`:`<div c\
lass="content">${v||sanitizeMarkdownHtml(d.content||"")}</div>`,x=buildPdfMessageAttachments(d),k=x.
length?`<div class="attachments">${x.map(M=>M.isImage?`<div class="attachment"><img src="${pdfEscapeAttr(
M.previewUrl)}" alt="${pdfEscapeAttr(M.filename)}"><div class="file-caption">${pdfEscapeAttr(M.filename)}\
</div></div>`:`<div class="attachment"><a class="file" href="${pdfEscapeAttr(M.url)}" target="_blank\
" rel="noreferrer noopener"><span class="file-icon">\u{1F4C4}</span><span><span class="file-name">${pdfEscapeAttr(
M.filename)}</span><span class="file-source">${pdfEscapeAttr(M.source)}</span></span></a></div>`).join(
"")}</div>`:"",S=[];d.model&&!m&&S.push(d.model),d.tokens!==null&&d.tokens!==void 0&&S.push(`tokens:${d.
tokens}`),d.tokens_in!==null&&d.tokens_in!==void 0&&S.push(`in:${d.tokens_in}`),d.tokens_out!==null&&
d.tokens_out!==void 0&&S.push(`out:${d.tokens_out}`),d.tokens_thought!==null&&d.tokens_thought!==void 0&&
S.push(`thought:${d.tokens_thought}`),d.is_encrypted&&S.push("encrypted"),d.parent_id!==null&&d.parent_id!==
void 0&&S.push(`parent:#${d.parent_id}`);const C=S.length?`<div class="message-meta">${pdfEscapeAttr(
S.join(" \u2022 "))}</div>`:"";return`
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
                            ${b}
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
"error",!0);return}const o=document.createElement("iframe");activePdfPrintFrame=o,o.setAttribute("ar\
ia-hidden","true"),o.style.position="fixed",o.style.right="0",o.style.bottom="0",o.style.width="1px",
o.style.height="1px",o.style.opacity="0",o.style.pointerEvents="none",o.style.border="0";let l=null;
const c=r(()=>{l&&(clearTimeout(l),l=null),t&&t.remove(),(activePdfPrintFrame===o||activePdfPrintFrame===
e)&&(activePdfPrintFrame=null);try{o.parentNode&&o.parentNode.removeChild(o)}catch{}},"cleanup");l=setTimeout(
()=>{activePdfPrintFrame===o&&(console.log("PDF print cleanup fallback triggered"),c())},6e4),o.onload=
async()=>{try{const g=o.contentDocument,b=o.contentWindow;if(!g||!b){c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\
\u307E\u3057\u305F","error",!0);return}if(t.update(40),(Array.isArray(a&&a.messages)?a.messages:[]).
some(C=>maybeNeedsMathJax(C.content)||maybeNeedsMathJax(C.thought_text))&&(b.MathJax={tex:{inlineMath:[
["\\(","\\)"],["$","$"]],displayMath:[["$$","$$"],["\\[","\\]"]],processEscapes:!0},options:{ignoreHtmlClass:"\
tex2jax_ignore|mathjax_ignore",processHtmlClass:"tex2jax_process|mathjax_process"},startup:{typeset:!1}}),
t.update(50),g.fonts&&g.fonts.ready)try{await g.fonts.ready}catch{}t.update(60);const x=Array.from(g.
images||[]),k=Promise.all(x.map(C=>C.complete?Promise.resolve():new Promise(M=>{C.addEventListener("\
load",M,{once:!0}),C.addEventListener("error",M,{once:!0})})));if(await Promise.race([k,new Promise(
C=>setTimeout(C,5e3))]),t.update(80),g.getElementById("MathJax-script")){let C=0;for(;C<100&&(!b.MathJax||
typeof b.MathJax.typesetPromise!="function");)await new Promise(M=>setTimeout(M,50)),C++;if(b.MathJax&&
typeof b.MathJax.typesetPromise=="function")try{await b.MathJax.typesetPromise()}catch(M){console.error(
"PDF MathJax typeset failed",M)}}t.update(95),setTimeout(()=>{try{b.focus(),b.addEventListener("afte\
rprint",()=>{c()},{once:!0}),t.update(100),setTimeout(()=>{t&&t.remove()},1e3),b.print()}catch{c(),showToast(
"PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u3092\u958B\u3051\u307E\u305B\u3093\u3067\u3057\u305F","err\
or",!0)}},100)}catch{c(),showToast("PDF\u5370\u5237\u30E2\u30FC\u30C0\u30EB\u306E\u6E96\u5099\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)}};const d=buildPdfDocumentHtml(a),m=new Blob([d],{type:"text/html"});o.src=URL.createObjectURL(
m),document.body.appendChild(o)}catch{t&&t.remove(),activePdfPrintFrame=null,showToast("PDF\u51FA\u529B\u4E2D\u306B\u30A8\u30E9\u30FC\u304C\u767A\
\u751F\u3057\u307E\u3057\u305F","error",!0)}}r(openThreadPdfPrintDialog,"openThreadPdfPrintDialog");
function exportCurrentThreadPdf(){openThreadPdfPrintDialog().catch(()=>{showToast("PDF\u51FA\u529B\u306B\u5931\u6557\u3057\u307E\u3057\u305F",
"error",!0)})}r(exportCurrentThreadPdf,"exportCurrentThreadPdf"),window.regenerateMessage=e=>{const t=allMessages.
find(n=>n.id==e);if(!t||!t.parent_id){showToast("\u518D\u751F\u6210\u3067\u304D\u308B\u30E1\u30C3\u30BB\u30FC\u30B8\u304C\u898B\u3064\u304B\u308A\u307E\u305B\u3093",
"error",!0);return}beginEditMessage(t.parent_id,!0)};function getLibSortOrder(){const e=get("lib-sor\
t");let t=e?e.value:"";return t||(t=localStorage.getItem(LIB_SORT_KEY)||"newest"),e&&e.value!==t&&(e.
value=t),t||"newest"}r(getLibSortOrder,"getLibSortOrder");function sortLibraryFiles(e){const t=getLibSortOrder(),
n=Array.isArray(e)?e.slice():[],i=new Intl.Collator("ja",{numeric:!0,sensitivity:"base"}),a=r((d,m)=>i.
compare(d.filename||"",m.filename||""),"nameAsc"),o=r((d,m)=>i.compare(m.filename||"",d.filename||""),
"nameDesc"),l=r((d,m)=>(Number(m.ts)||0)-(Number(d.ts)||0),"tsDesc"),c=r((d,m)=>(Number(d.ts)||0)-(Number(
m.ts)||0),"tsAsc");return t==="name_asc"?n.sort((d,m)=>a(d,m)||l(d,m)):t==="name_desc"?n.sort((d,m)=>o(
d,m)||l(d,m)):t==="oldest"?n.sort((d,m)=>c(d,m)||a(d,m)):n.sort((d,m)=>l(d,m)||a(d,m)),n}r(sortLibraryFiles,
"sortLibraryFiles");function getLibSearchQuery(){const e=lib.searchQuery||(get("lib-search")?get("li\
b-search").value:"")||"";return String(e).trim().toLocaleLowerCase()}r(getLibSearchQuery,"getLibSear\
chQuery");function updateLibraryLoadMoreUi(){const e=get("lib-load-more-btn");e&&(e.hidden=!lib.hasMore||
!!lib.loading,e.disabled=!!lib.loading)}r(updateLibraryLoadMoreUi,"updateLibraryLoadMoreUi");function updateLibFavoriteFilterUi(){
const e=get("lib-favorite-filter-btn");if(!e)return;const t=!!lib.favoritesOnly;e.classList.toggle("\
is-active",t),e.setAttribute("aria-pressed",t?"true":"false");const n=e.querySelector("i");n&&(n.className=
t?"fas fa-star":"far fa-star")}r(updateLibFavoriteFilterUi,"updateLibFavoriteFilterUi");function fileNameForSearch(e){
return String(e&&e.filename||"").toLocaleLowerCase()}r(fileNameForSearch,"fileNameForSearch");function renderLibraryGrid(e=null){
const t=get("lib-grid");if(!t)return;updateLibFavoriteFilterUi(),updateLibraryLoadMoreUi();const n=Array.
isArray(e);if(n){const m=t.querySelector(".lib-empty-state");m&&m.remove()}else t.innerHTML="";if(!lib.
files||!lib.files.length){if(n)return;t.innerHTML='<div class="lib-empty-state"><div class="lib-empt\
y-icon"><i class="fas fa-folder"></i></div><p class="lib-empty-title">\u30D5\u30A1\u30A4\u30EB\u304C\u307E\u3060\u3042\u308A\u307E\u305B\u3093</p><p class="lib-\
empty-sub">\u30A2\u30C3\u30D7\u30ED\u30FC\u30C9\u3057\u305F\u30D5\u30A1\u30A4\u30EB\u304C\u3053\u3053\u306B\u8868\u793A\u3055\u308C\u307E\u3059\u3002</p></div>';
const m=get("lib-total-count");m&&(m.innerText="0 files");return}const i=sortLibraryFiles(lib.files),
a=getLibSearchQuery(),o=i.filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(
a)),l=get("lib-total-count");if(l){const m=Number(lib.totalCount)||lib.files.length;lib.hasMore||a||
lib.favoritesOnly?l.innerText=`${lib.files.length} / ${m} files`:l.innerText=`${m} files`}if(!o.length){
if(n)return;const m=lib.favoritesOnly&&!a?"fa-star":"fa-search",g=lib.favoritesOnly&&!a?"\u304A\u6C17\u306B\u5165\u308A\u304C\u3042\u308A\u307E\u305B\u3093":
"\u4E00\u81F4\u3059\u308B\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",b=lib.favoritesOnly&&
!a?"\u30D5\u30A1\u30A4\u30EB\u306E\u661F\u30DC\u30BF\u30F3\u304B\u3089\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0\u3067\u304D\u307E\u3059\u3002":
"\u691C\u7D22\u6761\u4EF6\u3084\u4E26\u3073\u9806\u3092\u5909\u66F4\u3057\u3066\u304F\u3060\u3055\u3044\u3002";
t.innerHTML=`<div class="lib-empty-state"><div class="lib-empty-icon"><i class="fas ${m}"></i></div>\
<p class="lib-empty-title">${g}</p><p class="lib-empty-sub">${b}</p></div>`;return}let c=0;(n?sortLibraryFiles(
e).filter(m=>lib.favoritesOnly&&!m.is_favorite?!1:!a||fileNameForSearch(m).includes(a)):o).forEach(m=>{
try{const g=renderLibraryItem(m,c++);t.appendChild(g)}catch{}})}r(renderLibraryGrid,"renderLibraryGr\
id");function openLibraryImage(e){if(!lib.files)return;const t=sortLibraryFiles(lib.files),n=getLibSearchQuery(),
a=(n?t.filter(d=>fileNameForSearch(d).includes(n)):t).filter(d=>d.type==="image"),o=lib.favoritesOnly?
a.filter(d=>d.is_favorite):a;if(!o.length)return;const l=o.map(d=>({url:d.url,filename:d.filename||d.
original_filename||d.url.split("/").pop(),element:null}));let c=l.findIndex(d=>d.url===e.url);c===-1&&
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
 fa-download"></i></a></div>`,c=e.is_favorite?" is-favorite":"",d=e.is_favorite?"fas fa-star":"far f\
a-star",m=e.is_favorite?"\u304A\u6C17\u306B\u5165\u308A\u304B\u3089\u5916\u3059":"\u304A\u6C17\u306B\u5165\u308A\u306B\u8FFD\u52A0",
g=`<div class="lib-thumb-actions"><button class="lib-favorite-btn lib-action-circle${c}" title="${m}\
" aria-label="${m}" aria-pressed="${e.is_favorite?"true":"false"}"><i class="${d}"></i></button><but\
ton class="lib-open-btn lib-action-circle" title="\u958B\u304F"><i class="fas fa-eye"></i></button><button cla\
ss="lib-del-btn lib-action-circle lib-del" title="\u524A\u9664"><i class="fas fa-trash"></i></button></div>`,
b=`<div class="lib-thumb-bar"><span class="lib-thumb-name" title="${escapeHtml(e.filename)}">${escapeHtml(
e.filename)}</span></div>`;n.innerHTML=`<div class="lib-thumb-media-wrap">${o}</div>${l}${g}${b}`,n.
dataset.filepath=e.filepath,n.addEventListener("mousedown",k=>{k.shiftKey&&k.preventDefault()}),n.onclick=
k=>{if(k&&k.shiftKey&&lib.anchorPath&&lib.anchorPath!==e.filepath){const S=Array.from(n.parentNode?n.
parentNode.querySelectorAll(".library-thumb-card"):[]),C=S.findIndex(E=>E.dataset.filepath===lib.anchorPath),
M=S.indexOf(n);if(C!==-1&&M!==-1){const E=Math.min(C,M),q=Math.max(C,M);for(let N=E;N<=q;N++){const W=S[N].
dataset.filepath;W&&(lib.selected.add(W),S[N].classList.add("is-selected"))}try{const N=window.getSelection&&
window.getSelection();N&&N.removeAllRanges()}catch{}window.updateLibSelectionUi();return}}lib.anchorPath=
e.filepath,lib.selected.has(e.filepath)?(lib.selected.delete(e.filepath),n.classList.remove("is-sele\
cted")):(lib.selected.add(e.filepath),n.classList.add("is-selected")),window.updateLibSelectionUi()},
lib.selected&&lib.selected.has(e.filepath)&&n.classList.add("is-selected"),n.querySelectorAll(".lib-\
open-btn").forEach(k=>{k.onclick=S=>{S.stopPropagation(),e.type==="image"?openLibraryImage(e):openFileViewer(
e.url,e.filename)}});const w=n.querySelector(".lib-del-btn");w&&(w.onclick=async k=>{k.stopPropagation(),
await deleteSingleLibraryFile(e.filepath,n)});const x=n.querySelector(".lib-favorite-btn");return x&&
(x.onclick=async k=>{k.stopPropagation(),x.disabled=!0;try{const S=await apiFetch(CHAT_CONFIG.urls.toggleFileFavorite,
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
"upload-list");c&&c.querySelectorAll("[data-filename]").forEach(d=>{d.getAttribute("data-filename")===
e&&setRowAttachmentName(d,t?t.filename:l.filename||a)}),renderLibraryGrid(),window.updateLibSelectionUi(),
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
return}if(i.innerHTML="",c.forEach(d=>{const m=document.createElement("div");m.className="flex items\
-center gap-3 rounded-lg border border-gray-700 bg-gray-800/70 p-3";const g=d.updated_at?new Date(d.
updated_at).toLocaleString():"";m.innerHTML=`<div class="min-w-0 flex-1"><div class="text-sm text-gr\
ay-200 truncate" title="${escapeHtml(d.title||"")}">${escapeHtml(d.title||"\u65B0\u3057\u3044\u30C1\u30E3\u30C3\u30C8")}\
</div><div class="text-[11px] text-gray-500 mt-1">${escapeHtml(g)}</div></div><button type="button" \
class="lib-action-btn lib-btn-accent shrink-0"><i class="fas fa-folder"></i><span>\u958B\u304F</span></button>`;
const b=m.querySelector("button");b&&(b.onclick=async()=>{closeFileUsageModal(),window.closeLibModal&&
window.closeLibModal(!0),await loadMessages(String(d.id))}),i.appendChild(m)}),l.has_more){const d=document.
createElement("p");d.className="text-[11px] text-gray-500 text-center pt-2",d.textContent="\u8868\u793A\u3067\u304D\u308B\u30C1\u30E3\u30C3\u30C8\
\u306F\u6700\u5927100\u4EF6\u3067\u3059\u3002",i.appendChild(d)}}catch{i.innerHTML='<div class="text\
-sm text-red-300 text-center py-8"><i class="fas fa-exclamation-triangle mr-2"></i>\u4F7F\u7528\u30C1\u30E3\u30C3\u30C8\u306E\u53D6\u5F97\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\
</div>'}}}r(showSelectedFileUsage,"showSelectedFileUsage");async function loadLibraryFiles(e=!1){const t=get(
"lib-grid"),n=get("lib-load-more-btn");if(lib.loading||e&&!lib.hasMore)return;lib.loading=!0,e||(lib.
nextOffset=0,lib.totalCount=0,lib.hasMore=!1),e||renderLibrarySkeleton(t);let i=null;const a=CHAT_CONFIG.
urls.getFilesLib;let o=null,l=!1;try{const d=getLibSortOrder(),m=getLibSearchQuery(),g=e?lib.nextOffset:
0,b=new URLSearchParams({limit:String(LIBRARY_PAGE_SIZE),offset:String(g),sort:d,q:m,favorites_only:lib.
favoritesOnly?"1":"0"}),v=await apiFetch(a+"?"+b.toString(),{cache:"no-store",headers:{Accept:"appli\
cation/json"}});if(!v.ok)throw new Error("HTTP "+v.status);o=await v.json(),l=!0}catch(d){i=d}if(!l){
console.error("Library load failed:",i),!e&&t?t.innerHTML='<div class="lib-empty-state"><div class="\
lib-empty-icon"><i class="fas fa-exclamation-triangle"></i></div><p class="lib-empty-title">\u30E9\u30A4\u30D6\u30E9\u30EA\u306E\u8AAD\u307F\
\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F</p><p class="lib-empty-sub">\u901A\u4FE1\u72B6\u6CC1\u3092\u78BA\u8A8D\u3057\u3066\u6642\u9593\u3092\u304A\u3044\u3066\u518D\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002</p></div>':
e&&showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\u4E00\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002",
"error",!0),lib.loading=!1,n&&(n.disabled=!1,n.hidden=!lib.hasMore);return}let c=Array.isArray(o)?o:
o&&Array.isArray(o.files)?o.files:[];o&&!Array.isArray(o)&&(lib.totalCount=Number(o.total)||0,lib.hasMore=
!!o.has_more,lib.nextOffset=(Number(o.offset)||0)+(Number(o.limit)||c.length));try{const d=FILE_BASE_URL,
m=FILE_THUMB_BASE_URL,g=new Set(c.map(v=>v&&v.filepath).filter(Boolean));(!e&&Array.isArray(currentImageUrls)?
currentImageUrls:[]).forEach(v=>{if(c.length>=LIBRARY_PAGE_SIZE||!v||g.has(v))return;const w=getAttachmentNameForPath(
v)||v.split("/").pop()||v,x=(w.split(".").pop()||"").toLowerCase(),k=["png","jpg","jpeg","webp","gif"].
includes(x)?"image":"file",S=k==="image"?m+v:null;c.unshift({filename:w,original_filename:w,filepath:v,
url:d+v,thumbnail_url:S,type:k,ext:x,is_favorite:!1,ts:Math.floor(Date.now()/1e3)}),g.add(v)})}catch{}
try{lib.selected||(lib.selected=new Set),e||lib.selected.clear();const d=c.filter(g=>g&&g.filepath&&
g.url);let m=[];if(e){const g=new Set(lib.files.map(b=>b.filepath));m=d.filter(b=>!g.has(b.filepath)),
lib.files.push(...m)}else lib.files=d;lib.files.forEach(g=>{g&&g.filepath&&setAttachmentNameForPath(
g.filepath,g.filename||g.original_filename||"")}),lib.fileSet=new Set(lib.files.map(g=>g.filepath)),
lib.totalCount||(lib.totalCount=lib.files.length),window.updateLibSelectionUi(),renderLibraryGrid(e?
m:null)}catch(d){i=i||d}i&&t&&(console.error("Library load failed:",i),e?showToast("\u8FFD\u52A0\u8AAD\u307F\u8FBC\u307F\u306B\u5931\u6557\u3057\u307E\u3057\u305F\u3002\u3082\u3046\
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
a);if(!c)return;const d=(lib.files||[]).find(m=>m&&m.filepath===a);d&&d.filename&&setAttachmentNameForPath(
c,d.filename),currentImageUrls.includes(c)||currentImageUrls.push(c),setAttachmentSourceForPath(c,"l\
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
"",i=typeof window.prepareMathStepsCopyText=="function"?window.prepareMathStepsCopyText(n):n;copyToClipboard(
i,()=>{t.innerHTML='<i class="fas fa-check"></i>',setTimeout(()=>t.innerHTML='<i class="fas fa-copy"\
></i>',2e3)},a=>{console.error(a),t.innerHTML='<i class="fas fa-times"></i>',setTimeout(()=>t.innerHTML=
'<i class="fas fa-copy"></i>',2e3)})},window.toggleThinking=e=>{const t=e.nextElementSibling;t.classList.
contains("collapsed")?t.classList.remove("collapsed"):t.classList.add("collapsed")};let selectedBranchNodeId=null,
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
ms-center mt-4";const l=document.createElement("div"),c=String(a.id)===String(currentLeafId),d=a.id===
threadFixedBranchId,m=branchLabelNames[a.id]||(a.role==="user"?"User":"AI"),g=getCumulativeTokensForNode(
a.id);if(l.className=`ui-enter-scale px-3 py-2 rounded-lg border cursor-pointer transition-all text-\
[10px] min-w-[120px] max-w-[180px] text-center relative ${selectedBranchNodeId===a.id?"ring-2 ring-p\
urple-500 border-purple-400":"border-gray-700 hover:border-gray-500"} ${c?"bg-blue-900/40 border-blu\
e-500/50":"bg-gray-800"}`,l.innerHTML=`
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
                `,l.onclick=b=>{b.stopPropagation(),selectedBranchNodeId=a.id,renderBranchTreeVisualization(),
updateBranchDetailPane()},o.appendChild(l),a.children.length>0){const b=document.createElement("div");
b.className="w-px h-4 bg-gray-700",o.appendChild(b);const v=document.createElement("div");v.className=
"flex gap-4 items-start",a.children.forEach(w=>v.appendChild(i(w))),o.appendChild(v)}return o}r(i,"r\
enderNodeRecursive"),n.forEach(a=>e.appendChild(i(a)))}r(renderBranchTreeVisualization,"renderBranch\
TreeVisualization");function formatBranchCreatedAt(e){if(!e)return"-";const t=new Date(e);return isNaN(
t.getTime())?String(e):t.toLocaleString("ja-JP",{year:"numeric",month:"2-digit",day:"2-digit",hour:"\
2-digit",minute:"2-digit"})}r(formatBranchCreatedAt,"formatBranchCreatedAt");function updateBranchDetailPane(){
const e=get("branch-detail-panel"),t=get("branch-empty-panel");if(!selectedBranchNodeId||!allMessages){
e.classList.add("hidden"),t.classList.remove("hidden");return}const n=allMessages.find(d=>d.id===selectedBranchNodeId);
if(!n)return;e.classList.remove("hidden"),t.classList.add("hidden"),get("br-id").innerText=n.id,get(
"br-date").innerText=formatBranchCreatedAt(n.created_at),get("br-model").innerText=n.model||"-";const i=n.
tokens||Number(n.tokens_in||0)+Number(n.tokens_out||0),a=getCumulativeTokensForNode(n.id);get("br-to\
kens").innerHTML=`<span title="Current message tokens">${i}</span> <span class="text-gray-500">/</sp\
an> <span class="text-purple-400 font-bold" title="Path total tokens">${a} total</span>`;const o=get(
"branch-model-breakdown"),l=getPerModelTokensForPath(n.id);o.innerHTML="",Object.entries(l).sort((d,m)=>m[1].
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
batchProviderLabel(a.provider)),d=escapeHtml(a.model||""),m=escapeHtml(batchFormatTime(a.created_at)),
g=escapeHtml(a.status_text||""),b=batchStateTone(a.state),v=a.thread_exists?'<button type="button" d\
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
                                <span class="truncate max-w-[16rem]">${d}</span>
                                <span><i class="fas fa-history mr-1"></i>${m}</span>
                            </div>
                        </div>
                        <span class="batch-state-badge shrink-0 ${b}">${escapeHtml(batchStateLabelShort(
a.state))}</span>
                    </div>
                    <div class="batch-job-status mt-2 text-[11px] break-words">${g}</div>
                    ${a.error?`<div class="batch-job-error mt-1 text-[10px] break-words">${escapeHtml(
a.error)}</div>`:""}
                    <div class="mt-3 flex flex-wrap gap-2">
                        ${v}${w}${x}
                    </div>`;const k=o.querySelector("[data-batch-open]");k&&(k.onclick=()=>{window.closeBatchModal(),
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
"api-key-modal-save-btn"),l=get("api-key-modal-fallback-btn"),c=get("api-key-modal-cancel-btn"),d=r(
()=>{o.onclick=null,l.onclick=null,c.onclick=null},"cleanup"),m=r(g=>{g.key==="Enter"&&(g.preventDefault(),
o.click())},"onKeydown");get("api-key-modal-input").addEventListener("keydown",m),o.onclick=async()=>{
const g=get("api-key-modal-input").value.trim();if(!g){showToast("API\u30AD\u30FC\u3092\u5165\u529B\u3057\u3066\u304F\u3060\u3055\u3044",
"error");return}if(i){const b=get(i.inputId);b&&(b.value=g);try{if(!(await apiFetch(CHAT_CONFIG.urls.
handleSettings,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[i.keyField]:g})})).
ok){showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\u3057\u305F","error",
!0);return}userSettingsSnapshot&&(userSettingsSnapshot[i.keyField]=g)}catch{showToast("API\u30AD\u30FC\u306E\u4FDD\u5B58\u306B\u5931\u6557\u3057\u307E\
\u3057\u305F","error",!0);return}}hideModal("api-key-required-modal"),get("api-key-modal-input").removeEventListener(
"keydown",m),d(),t("set")},l.onclick=()=>{hideModal("api-key-required-modal"),get("api-key-modal-inp\
ut").removeEventListener("keydown",m),d(),t("switch")},c.onclick=()=>{hideModal("api-key-required-mo\
dal"),get("api-key-modal-input").removeEventListener("keydown",m),d(),t("cancel")},showModal("api-ke\
y-required-modal"),setTimeout(()=>{const g=get("api-key-modal-input");g&&g.focus()},350)}),"showApiK\
eyRequiredModalAsync");(function(){const e=console.log,t=console.error,n=console.warn,i=console.info;
let a=!1;async function o(l,c){if(a||!isClientDebugLogEnabled()||c&&c[0]===ADMIN_SIDEBAR_DEBUG_PREFIX)
return;a=!0;const d=c.map(m=>{try{return m instanceof Error?m.stack||m.message:typeof m=="object"?JSON.
stringify(m):String(m)}catch{return"[Unserializable Object]"}}).join(" ");try{sendClientDebugLog(l,d)}catch{}finally{
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
selected:new Set,counts:{},total:0,data:null},c={accounts:[],selected:new Set},d=r(()=>get("bot-log-\
detail"),"view"),m=r(()=>get("bot-log-list-view"),"listView"),g=r($=>{if(!$)return"-";const O=new Date(
$);return isNaN(O.getTime())?String($):O.toLocaleString("ja-JP")},"formatTime"),b=r($=>{const O=Math.
max(0,Math.floor(Number($)||0));return O>=60?`${Math.floor(O/60)}\u5206${O%60?O%60+"\u79D2":""}`:`${O}\
\u79D2`},"formatDuration"),v=r(($,O)=>O==null||O===""?"-":$==="lock_source"?a[O]||String(O):$==="cli\
ent"||$==="origin_client"?n[O]||String(O):$==="action"?o[O]||String(O):$==="by"?O==="self"?"\u672C\u4EBA":
O==="admin"?"\u7BA1\u7406\u8005":String(O):$==="enabled"?O?"ON":"OFF":$==="remaining_seconds"||$==="\
lock_seconds"?b(O):$==="applied_to"&&Array.isArray(O)?O.map(V=>a[V]||V).join("\u30FB"):typeof O=="ob\
ject"?JSON.stringify(O,null,2):String(O),"formatDetailValue"),w=r($=>{const O=String($||"").trim();if(!O)
return"";let V=null;if(O.startsWith("{"))try{V=JSON.parse(O)}catch{V=null}if(V&&typeof V=="object"&&
!Array.isArray(V)){const B=Object.keys(V).map(G=>{const ae=v(G,V[G]),R=i[G]||G;return ae.length>120||
ae.includes(`
`)?`<div class="mt-1"><div class="text-gray-500">${escapeHtml(R)}</div><pre class="bot-log-pre">${escapeHtml(
ae)}</pre></div>`:`<div><span class="text-gray-500">${escapeHtml(R)}:</span> <span class="text-gray-\
300">${escapeHtml(ae)}</span></div>`}).join("");return O.length>600?`<details class="mt-1"><summary \
class="cursor-pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary>${B}</details>`:`<div c\
lass="mt-1 space-y-1">${B}</div>`}return O.length>300?`<details class="mt-1"><summary class="cursor-\
pointer text-gray-400">\u8A73\u7D30\u3092\u8868\u793A</summary><pre class="bot-log-pre">${escapeHtml(
O)}</pre></details>`:`<div class="mt-1 text-gray-400 bot-log-wrap">${escapeHtml(O)}</div>`},"renderD\
etails"),x=r(async($,O)=>{const V=await apiFetch($,{method:"POST",headers:{"Content-Type":"applicati\
on/json"},body:JSON.stringify(O)}),B=await V.json().catch(()=>({}));if(!V.ok)throw new Error(B.error||
String(V.status));return B},"postJson"),k=r(($,O)=>`<span class="${O} text-white px-2 py-0.5 rounded\
">${escapeHtml($)}</span>`,"badge"),S=r(()=>{const $=get("bot-log-account-list");if(!$)return;const O=c.
accounts,V=O.length&&O.every(G=>c.selected.has(G.user_id)),B=get("bot-log-list-toolbar");if(B&&(B.innerHTML=
`
                        <label class="flex items-center gap-1 text-gray-300"><input type="checkbox" \
class="bot-log-account-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u3059\u3079\u3066\u9078\u629E" ${V?
"checked":""}>\u8868\u793A\u4E2D\u3092\u3059\u3079\u3066\u9078\u629E</label>
                        <button class="bot-log-delete-accounts bg-red-600 hover:bg-red-500 text-whit\
e px-2 py-1 rounded" ${c.selected.size?"":"disabled"}>\u9078\u629E\u3057\u305F\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u4E00\u62EC\u524A\u9664${c.
selected.size?`\uFF08${c.selected.size}\u4EF6\uFF09`:""}</button>`),!O.length){$.innerHTML='<div cla\
ss="text-xs text-gray-400">\u8A18\u9332\u306E\u3042\u308B\u30A2\u30AB\u30A6\u30F3\u30C8\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>';
return}$.innerHTML=O.map(G=>{const ae=[];G.exists||ae.push(k("\u524A\u9664\u6E08\u307F","bg-gray-600")),
G.is_bot_banned&&ae.push(k("BAN\u4E2D","bg-red-600")),G.lock_remaining_seconds>0&&ae.push(k(`\u30ED\u30C3\u30AF\u4E2D\uFF08\u6B8B\u308A${Math.
ceil(G.lock_remaining_seconds/60)}\u5206\uFF09`,"bg-yellow-600"));const R=G.evidence_count?`\u8A18\u9332 ${G.
evidence_count}\u4EF6\u30FB\u6700\u7D42 ${escapeHtml(g(G.last_event_at))}`:"\u8A18\u9332\u306A\u3057";
return`
                        <div class="flex items-center gap-2 bg-gray-900 border border-gray-700 round\
ed p-2 text-xs">
                            <input type="checkbox" class="bot-log-account-select" data-user-id="${G.
user_id}" ${c.selected.has(G.user_id)?"checked":""} aria-label="${escapeHtml(G.username||"ID "+G.user_id)}\
 \u3092\u9078\u629E">
                            <div class="flex-1 min-w-0">
                                <div class="flex flex-wrap items-center gap-1"><span class="${G.exists?
"text-gray-200":"text-gray-400"} font-bold bot-log-wrap">${escapeHtml(G.username||"ID "+G.user_id)}<\
/span>${ae.join("")}</div>
                                <div class="text-[10px] text-gray-500">${R}</div>
                            </div>
                            <button class="bot-log-open-account shrink-0 whitespace-nowrap bg-gray-7\
00 hover:bg-gray-600 text-white px-2 py-1 rounded" data-user-id="${G.user_id}" data-username="${escapeHtml(
G.username||"")}">\u8A18\u9332\u3092\u898B\u308B</button>
                        </div>`}).join("")},"renderAccounts"),C=r(async()=>{const $=get("bot-log-acc\
ount-list");if(!$)return;$.innerHTML='<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinn\
er fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>';const O=get("bot-log-search"),V=O?O.value.
trim():"";try{const B=await apiFetch(`/api/bot/evidence/accounts?q=${encodeURIComponent(V)}`,{cache:"\
no-store"}),G=await B.json().catch(()=>({}));if(!B.ok)throw new Error(G.error||String(B.status));c.accounts=
G.accounts||[];const ae=new Set(c.accounts.map(R=>R.user_id));Array.from(c.selected).forEach(R=>{ae.
has(R)||c.selected.delete(R)}),S()}catch{$.innerHTML='<div class="text-xs text-red-400">\u30A2\u30AB\u30A6\u30F3\u30C8\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\
\u307E\u305B\u3093\u3067\u3057\u305F\u3002</div>',showToast("Bot\u691C\u51FA\u30ED\u30B0\u306E\u4E00\u89A7\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"loadAccounts"),M=r($=>{if(!$)return"";const O=$.locks||[],V=$.banned_identifiers||[],
B=[];$.exists||B.push(k("\u524A\u9664\u6E08\u307F","bg-gray-600")),$.exists&&(B.push(k($.detection_enabled?
"\u691C\u51FAON":"\u691C\u51FAOFF",$.detection_enabled?"bg-gray-600":"bg-gray-700")),B.push(k($.is_bot_banned?
"BAN\u4E2D":"BAN\u306A\u3057",$.is_bot_banned?"bg-red-600":"bg-gray-600"))),B.push(k(O.length?"\u30ED\u30C3\u30AF\u4E2D":
"\u30ED\u30C3\u30AF\u306A\u3057",O.length?"bg-yellow-600":"bg-gray-600")),V.length&&B.push(k(`IP\u30FB\u7AEF\u672B\u306E\
BAN ${V.length}\u4EF6`,"bg-red-600")),$.is_admin&&B.push(k("\u7BA1\u7406\u8005\uFF08\u76E3\u8996\u306E\u5BFE\u8C61\u5916\uFF09",
"bg-blue-600"));const G=!!(l.counts.ban||l.counts.related_ban),ae=$.is_bot_banned?`<div class="text-\
gray-300">BAN\u306E\u7406\u7531: ${escapeHtml($.bot_ban_reason||"-")}\uFF08${escapeHtml(g($.bot_banned_at))}\
\uFF09</div>${G?"":'<div class="text-gray-400">\u3053\u306EBAN\u306E\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\uFF08\u8A18\u9332\u3092\u59CB\u3081\u308B\u524D\u306EBAN\u304B\u3001\u8A18\u9332\u304C\u524A\u9664\u3055\u308C\u3066\u3044\u307E\u3059\uFF09\u3002</div>'}`:
"",R=O.map(Q=>{const ye=a[Q.source]||Q.source,ue=Q.identifier?` ${escapeHtml(Q.identifier)}${Q.source===
"cookie"?"\u2026":""}`:"",ke=Q.origin?`\u30FB\u304B\u3051\u305F\u30A2\u30AB\u30A6\u30F3\u30C8: ${escapeHtml(
Q.origin.username||String(Q.origin.user_id||"-"))}\uFF08${escapeHtml(n[Q.origin.client]||Q.origin.client||
"-")}\uFF09`:"";return`<div class="text-gray-300">\u30ED\u30C3\u30AF: ${escapeHtml(ye)}${ue}\u30FB\u6B8B\u308A${escapeHtml(
b(Q.remaining_seconds))}\u30FB${escapeHtml(Q.reason||"")}${ke}</div>`}).join(""),F=V.map(Q=>`<div cl\
ass="text-gray-300">IP\u30FB\u7AEF\u672B\u306EBAN: ${escapeHtml(a[Q.kind]||Q.kind)} ${escapeHtml(Q.identifier)}${Q.
kind==="cookie"?"\u2026":""}\u30FB${escapeHtml(Q.reason||"-")}\uFF08${escapeHtml(g(Q.created_at))}\uFF09<\
/div>`).join(""),me=V.length?'<div class="text-gray-400">IP\u30FB\u7AEF\u672B\u306EBAN\u304C\u3042\u308B\u3068\u3001\u540C\u3058IP\u30A2\u30C9\u30EC\u30B9\u3084\u7AEF\u672B\u304B\u3089\u306F\u65B0\u3057\u3044\u30A2\u30AB\u30A6\u30F3\u30C8\u3092\u4F5C\u308C\u307E\u305B\u3093\u3002\
</div>':"",re=$.turnstile_verified_seconds>0?`\u78BA\u8A8D\u6E08\u307F\uFF08\u6B8B\u308A${b($.turnstile_verified_seconds)}\
\uFF09`:"\u672A\u78BA\u8A8D",X=$.appeal_count?`\u30FB\u7570\u8B70\u7533\u3057\u7ACB\u3066 ${escapeHtml(
String($.appeal_count))}\u4EF6`:"",ce=[];return O.length&&ce.push('<button class="bot-log-unlock bg-\
yellow-600 hover:bg-yellow-500 text-white px-2 py-1 rounded">\u30ED\u30C3\u30AF\u3092\u89E3\u9664</button>'),
V.length&&ce.push('<button class="bot-log-unblock bg-red-600 hover:bg-red-500 text-white px-2 py-1 r\
ounded">IP\u30FB\u7AEF\u672B\u306EBAN\u3092\u89E3\u9664</button>'),`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs space-y-1">
                        <div class="flex flex-wrap items-center gap-1">${B.join("")}</div>
                        ${ae}
                        ${R}
                        ${F}
                        ${me}
                        <div class="text-gray-400">\u30ED\u30C3\u30AF\u56DE\u6570\uFF081\u6642\u9593\uFF09: ${escapeHtml(
String($.lock_count))} / ${escapeHtml(String($.lock_count_limit))}\u30FBTurnstile: ${escapeHtml(re)}\
\u30FB\u5931\u6557 ${escapeHtml(String($.turnstile_fail_count))} / ${escapeHtml(String($.turnstile_fail_limit))}\
\u30FB\u5224\u5B9A\u30B9\u30B3\u30A2 ${escapeHtml(String(Math.round(($.score||0)*10)/10))}\uFF08\u64CD\u4F5C ${escapeHtml(
String(Math.round(($.behavior_score||0)*10)/10))}\uFF09${X}</div>
                        ${ce.length?`<div class="flex flex-wrap gap-2 pt-1">${ce.join("")}</div>`:""}\

                    </div>`},"renderState"),E=r(()=>{const $=r((V,B,G)=>`<button class="bot-log-filt\
er ${l.type===V?"bg-blue-600 hover:bg-blue-500":"bg-gray-700 hover:bg-gray-600"} text-white px-2 py-\
1 rounded" data-type="${escapeHtml(V)}">${escapeHtml(B)} (${G})</button>`,"chip"),O=[$("","\u3059\u3079\u3066",
l.total)];return Object.keys(l.counts).sort((V,B)=>l.counts[B]-l.counts[V]).forEach(V=>{O.push($(V,(t[V]||
{label:V}).label,l.counts[V]))}),O.join("")},"renderFilters"),q=r($=>{const O=t[$.event_type]||{label:$.
event_type,cls:"bg-gray-600"},V=$.client?`<span class="bg-gray-700 text-gray-200 px-1.5 py-0.5 round\
ed">${escapeHtml(n[$.client]||$.client)}</span>`:"",B=l.selected.has($.id)?"checked":"",G=$.score!==
null&&$.score!==void 0?`<span class="text-gray-400">\u30B9\u30B3\u30A2 ${escapeHtml(String($.score))}${$.
behavior_score!==null&&$.behavior_score!==void 0?`\uFF08\u64CD\u4F5C ${escapeHtml(String($.behavior_score))}\
\uFF09`:""}</span>`:"",ae=$.reasons?`<div class="mt-1 text-gray-300 bot-log-wrap">\u7406\u7531: ${escapeHtml(
$.reasons)}</div>`:"",R=$.ip_address||$.user_agent?`<div class="mt-1 text-[10px] text-gray-500 bot-l\
og-wrap">${$.ip_address?"IP "+escapeHtml($.ip_address):""}${$.ip_address&&$.user_agent?"\u30FB":""}${escapeHtml(
$.user_agent||"")}</div>`:"";return`
                    <div class="bg-gray-900 border border-gray-700 rounded p-2 text-xs" data-log-id=\
"${$.id}">
                        <div class="flex flex-wrap items-center gap-1">
                            <input type="checkbox" class="bot-log-select" data-log-id="${$.id}" ${B}\
 aria-label="\u3053\u306E\u8A18\u9332\u3092\u9078\u629E">
                            <span class="text-gray-400">${escapeHtml(g($.created_at))}</span>
                            <span class="${O.cls} text-white px-1.5 py-0.5 rounded">${escapeHtml(O.label)}\
</span>
                            ${V}
                            ${G}
                            <button class="bot-log-delete-one ml-auto text-gray-400 hover:text-white\
 px-1" data-log-id="${$.id}" title="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664" aria-label="\u3053\u306E\u8A18\u9332\u3092\u524A\u9664"><i class="fas fa-trash"></i></butt\
on>
                        </div>
                        ${ae}
                        ${w($.details)}
                        ${R}
                    </div>`},"renderItem"),N=r(()=>{const $=d();if(!$)return;const O=l.items.length?
l.items.map(q).join(""):'<div class="text-xs text-gray-400 py-2">\u8A18\u9332\u306F\u3042\u308A\u307E\u305B\u3093\u3002</div>',
V=l.items.length&&l.items.every(B=>l.selected.has(B.id));$.innerHTML=`
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
ox" class="bot-log-select-all" aria-label="\u8868\u793A\u4E2D\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u9078\u629E" ${V?
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
                    </div>`},"render"),W=r(async($=!1)=>{const O=d();if(!O||!l.userId)return;$||(O.innerHTML=
'<div class="text-xs text-gray-400 py-2"><i class="fas fa-spinner fa-spin mr-1"></i>\u8AAD\u307F\u8FBC\u307F\u4E2D...</div>');
const V=new URLSearchParams({user_id:String(l.userId),limit:String(50)});l.type&&V.set("type",l.type),
$&&l.items.length&&V.set("before_id",String(l.items[l.items.length-1].id));try{const B=await apiFetch(
`/api/bot/evidence?${V.toString()}`,{cache:"no-store"}),G=await B.json().catch(()=>({}));if(!B.ok)throw new Error(
G.error||String(B.status));l.counts=G.counts||{},l.total=G.total||0,l.hasMore=!!G.has_more,G.user&&(l.
username=G.user.username||l.username,l.exists=!!G.user.exists),$?l.items=l.items.concat(G.items||[]):
(l.items=G.items||[],l.data=G),N()}catch{$||(O.innerHTML='<div class="text-xs text-red-400">\u8A18\u9332\u306E\u53D6\u5F97\u306B\u5931\u6557\
\u3057\u307E\u3057\u305F\u3002</div>'),showToast("\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"load"),I=r(async($,O)=>{if(confirm(O))try{const V=await x("/api/bot/evidence/delete",
Object.assign({user_id:l.userId},$));showToast(`${V.deleted||0}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,
"success"),l.selected.clear(),await W(!1)}catch{showToast("\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0)}},"deleteLogs"),j=r(async($,O,V)=>{if(confirm(O))try{await x($,{user_id:l.userId}),showToast(
V,"success"),await W(!1)}catch{showToast("\u64CD\u4F5C\u306B\u5931\u6557\u3057\u307E\u3057\u305F","e\
rror",!0)}},"runAccountAction"),ne=r(()=>{const $=d();$&&($.classList.add("hidden"),$.classList.remove(
"flex"),$.innerHTML="");const O=m();O&&(O.classList.remove("hidden"),O.classList.add("flex")),l.userId=
0},"showList"),oe=r(async()=>{ne(),await C()},"openList"),te=r(async($,O="")=>{const V=d();if(!V)return;
Object.assign(l,{userId:Number($)||0,username:O,exists:!1,type:"",items:[],hasMore:!1,counts:{},total:0,
data:null}),l.selected.clear();const B=m();B&&(B.classList.add("hidden"),B.classList.remove("flex")),
V.classList.remove("hidden"),V.classList.add("flex"),await W(!1)},"open");return{open:te,openList:oe,
showList:ne,bind:r(()=>{const $=get("bot-log-modal");if(!$||$.dataset.bound==="1")return;$.dataset.bound=
"1";const O=get("bot-log-search");O&&O.addEventListener("keydown",V=>{V.key==="Enter"&&C()}),$.addEventListener(
"change",V=>{const B=V.target;if(!(!B||!B.classList))if(B.classList.contains("bot-log-select")){const G=Number(
B.getAttribute("data-log-id"));B.checked?l.selected.add(G):l.selected.delete(G),N()}else if(B.classList.
contains("bot-log-select-all"))l.items.forEach(G=>{B.checked?l.selected.add(G.id):l.selected.delete(
G.id)}),N();else if(B.classList.contains("bot-log-account-select")){const G=Number(B.getAttribute("d\
ata-user-id"));B.checked?c.selected.add(G):c.selected.delete(G),S()}else B.classList.contains("bot-l\
og-account-select-all")&&(c.accounts.forEach(G=>{B.checked?c.selected.add(G.user_id):c.selected.delete(
G.user_id)}),S())}),$.addEventListener("click",async V=>{const B=V.target.closest("button");if(!(!B||
B.disabled))if(B.id==="bot-log-search-btn"||B.id==="bot-log-refresh-btn")B.id==="bot-log-refresh-btn"&&
O&&(O.value=""),await C();else if(B.classList.contains("bot-log-open-account")){const G=Number(B.getAttribute(
"data-user-id"));G&&await te(G,B.getAttribute("data-username")||"")}else if(B.classList.contains("bo\
t-log-delete-accounts")){const G=Array.from(c.selected);if(!G.length||!confirm(`\u9078\u629E\u3057\u305F${G.
length}\u4EF6\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`))
return;try{const ae=await x("/api/bot/evidence/delete",{user_ids:G});showToast(`${ae.deleted||0}\u4EF6\u306E\u8A18\u9332\
\u3092\u524A\u9664\u3057\u307E\u3057\u305F`,"success"),c.selected.clear(),await C()}catch{showToast(
"\u8A18\u9332\u3092\u524A\u9664\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F","error",!0)}}else if(B.
classList.contains("bot-log-back"))await oe();else if(B.classList.contains("bot-log-reload"))await W(
!1);else if(B.classList.contains("bot-log-more"))await W(!0);else if(B.classList.contains("bot-log-f\
ilter"))l.type=B.getAttribute("data-type")||"",l.selected.clear(),await W(!1);else if(B.classList.contains(
"bot-log-delete-one")){const G=Number(B.getAttribute("data-log-id"));G&&await I({ids:[G]},"\u3053\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\
\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002")}else
B.classList.contains("bot-log-delete-selected")?l.selected.size&&await I({ids:Array.from(l.selected)},
`\u9078\u629E\u3057\u305F${l.selected.size}\u4EF6\u306E\u8A18\u9332\u3092\u524A\u9664\u3057\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
B.classList.contains("bot-log-delete-all")?await I({all:!0},`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30DC\u30C3\u30C8\u691C\u51FA\u306E\u8A18\u9332\u3092\u3059\u3079\u3066\uFF08${l.total}\u4EF6\uFF09\u524A\u9664\u3057\
\u307E\u3059\u304B\uFF1F\u3053\u306E\u64CD\u4F5C\u306F\u53D6\u308A\u6D88\u305B\u307E\u305B\u3093\u3002`):
B.classList.contains("bot-log-unlock")?await j("/api/bot/account/clear-lock",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
 \u306E\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3059\u304B\uFF1F
\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8\u306EIP\u30A2\u30C9\u30EC\u30B9\u3068\u7AEF\u672B\u306B\u304B\u304B\u3063\u3066\u3044\u308B\u30ED\u30C3\u30AF\u3082\u89E3\u9664\u3057\u307E\u3059\u3002`,
"\u30ED\u30C3\u30AF\u3092\u89E3\u9664\u3057\u307E\u3057\u305F"):B.classList.contains("bot-log-unbloc\
k")&&await j("/api/bot/account/clear-identifiers",`${l.username||"\u3053\u306E\u30A2\u30AB\u30A6\u30F3\u30C8"}\
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
classList.contains("file-thumb")){const g=(l.getAttribute("onclick")||"").match(/window.open\('([^']+)'/);
g&&(c=n(g[1]))}if(!o.has(c))return;const d=document.createElement("span");d.className="inline-block \
rounded bg-gray-800 p-3 text-xs text-gray-400",d.textContent="\u30D5\u30A1\u30A4\u30EB\u304C\u3042\u308A\u307E\u305B\u3093",
d.title=c.split("/").pop();const m=l.closest(".chat-image-frame")||(l.tagName==="SOURCE"?l.parentNode:
l);m&&a.contains(m)&&m.replaceWith(d)})}}r(displayMissingMessageFiles,"displayMissingMessageFiles");
const SPLIT_COMPRESSION_KEY="image_split_compression",splitState={files:[],index:0,image:null,release:null,
revision:0,busy:!1,urls:[]};function splitClamp(e,t,n,i){const a=Number.parseInt(e,10);return Number.
isFinite(a)?Math.max(t,Math.min(n,a)):i}r(splitClamp,"splitClamp");function chooseSplitGrid(e,t,n){const i=splitClamp(
e,2,30,4);let a={columns:i,rows:1},o=1/0;for(let l=1;l<=i;l++){if(i%l)continue;const c=i/l,d=Math.abs(
Math.log(Math.max(1,t)/l/(Math.max(1,n)/c)));(d<o-1e-9||Math.abs(d-o)<=1e-9&&(t>=n?l>a.columns:c>a.rows))&&
(a={columns:l,rows:c},o=d)}return a}r(chooseSplitGrid,"chooseSplitGrid");function splitTiles(e,t,n,i){
const a=splitClamp(i,0,50,10)/100,o=Math.round(e/n.columns*a/2),l=Math.round(t/n.rows*a/2),c=[];for(let d=0;d<
n.rows;d++)for(let m=0;m<n.columns;m++){const g=Math.floor(m*e/n.columns),b=Math.floor((m+1)*e/n.columns),
v=Math.floor(d*t/n.rows),w=Math.floor((d+1)*t/n.rows);c.push({index:c.length,row:d,column:m,cellLeft:g,
cellTop:v,cellRight:b,cellBottom:w,left:Math.max(0,g-o),top:Math.max(0,v-l),right:Math.min(e,b+o),bottom:Math.
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
getContext("2d"),c=e.width/n,d=e.height/i;l.drawImage(t,0,0,e.width,e.height);const m=o[0],g=m.right-
m.cellRight,b=m.bottom-m.cellBottom;l.fillStyle="rgba(250,204,21,0.18)";const v=[];for(let x=1;x<a.columns;x++){
const k=Math.floor(x*n/a.columns);l.fillRect((k-g)*c,0,2*g*c,e.height),v.push([k*c,0,k*c,e.height])}
for(let x=1;x<a.rows;x++){const k=Math.floor(x*i/a.rows);l.fillRect(0,(k-b)*d,e.width,2*b*d),v.push(
[0,k*d,e.width,k*d])}drawSplitLines(l,v,Math.max(2,Math.min(5,Math.max(e.width,e.height)/400)),!1);const w=Math.
max(14,Math.min(64,Math.min(e.width/a.columns,e.height/a.rows)*.18));l.font=`bold ${w}px monospace`,
l.textAlign="center",l.textBaseline="middle";for(const x of o){const k=(x.cellLeft+x.cellRight)/2*c,
S=(x.cellTop+x.cellBottom)/2*d;l.lineWidth=Math.max(2,w/10),l.strokeStyle="rgba(0,0,0,0.7)",l.strokeText(
String(x.index+1),k,S),l.fillStyle="#facc15",l.fillText(String(x.index+1),k,S)}}r(drawSplitOverview,
"drawSplitOverview");function drawSplitLines(e,t,n,i){e.save(),e.lineWidth=n,e.strokeStyle="rgba(250\
,204,21,0.6)",e.shadowColor="rgba(0,0,0,0.7)",e.shadowBlur=2,i&&e.setLineDash([n*4,n*3]),e.beginPath(),
t.forEach(([a,o,l,c])=>{e.moveTo(a,o),e.lineTo(l,c)}),e.stroke(),e.restore()}r(drawSplitLines,"drawS\
plitLines");async function splitCanvasFile(e,t){try{const n=await new Promise((i,a)=>e.toBlob(o=>o?i(
o):a(new Error("\u753B\u50CF\u3092\u66F8\u304D\u51FA\u305B\u307E\u305B\u3093\u3002")),"image/png"));
return new File([n],t,{type:"image/png"})}finally{e.width=0,e.height=0}}r(splitCanvasFile,"splitCanv\
asFile");async function renderSplitFiles(e,t,n){const i=await loadSplitImage(e),a=i.image;try{const o=a.
naturalWidth,l=a.naturalHeight,c=chooseSplitGrid(t.pieces,o,l);if(!splitFits(c,o,l))throw new Error(
`\u753B\u50CF\u304C\u5C0F\u3055\u3059\u304E\u308B\u305F\u3081${t.pieces}\u679A\u306B\u5206\u5272\u3067\u304D\u307E\u305B\u3093: ${e.
name}`);const d=splitTiles(o,l,c,t.overlap),m=splitBaseName(e.name),g=[],b=r(async v=>{g.push(await applyImageCompression(
v,n))},"append");if(t.includeOriginal)if(t.markOriginal){const v=splitCanvas(o,l);drawSplitOverview(
v,a,o,l,c,d),await b(await splitCanvasFile(v,`${m}_split_grid.png`))}else await b(e);for(const v of d){
const w=v.right-v.left,x=v.bottom-v.top;let k=Math.max(12,Math.min(40,Math.max(w,x)/45)),S=0;const C=splitTileCaption(
v,c,o,l);if(t.markPieces){const q=splitCanvas(1,1),N=q.getContext("2d");N.font=`bold ${k}px monospac\
e`,k=Math.max(6,k*Math.min(1,w*.96/N.measureText(C).width)),S=Math.max(12,Math.floor(k*1.8)),q.width=
0}const M=splitCanvas(w,x+S),E=M.getContext("2d");if(S&&(E.fillStyle="#111827",E.fillRect(0,0,w,S)),
E.drawImage(a,v.left,v.top,w,x,0,S,w,x),t.markPieces){E.fillStyle="#fff",E.font=`bold ${k}px monospa\
ce`,E.textBaseline="middle",E.fillText(C,w*.02,S/2);const q=[];for(const N of[v.cellLeft,v.cellRight])
N>v.left&&N<v.right&&q.push([N-v.left,S,N-v.left,S+x]);for(const N of[v.cellTop,v.cellBottom])N>v.top&&
N<v.bottom&&q.push([0,S+N-v.top,w,S+N-v.top]);drawSplitLines(E,q,Math.max(1.5,Math.min(4,Math.max(w,
x)/500)),!0)}await b(await splitCanvasFile(M,splitTileName(m,v,c)))}return g}finally{i.release()}}r(
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
l=Math.min(...o.map(E=>E.left)),c=Math.min(...o.map(E=>E.top)),d={left:l,top:c,width:Math.max(...o.map(
E=>E.right))-l,height:Math.max(...o.map(E=>E.bottom))-c},m=window.matchMedia("(prefers-reduced-motio\
n: reduce)").matches,g={navigation:threadLoadSequence,text:t.value,images:currentImageUrls,mask:currentMaskImage,
quote:currentQuote,quoteId:currentQuoteMessageId,rows:n?Array.from(n.childNodes):[],files:new Map(browserFastLocalFiles),
markers:new Set(markerAppliedUploads),progress:uploadProgressState},b=document.createElement("div");
b.hidden=!0,b.className="prompt-send-held-files",b.setAttribute("aria-hidden","true"),b.append(...g.
rows),document.body.appendChild(b);let v=!1,w=!1,x=!1;const k=r(()=>{v=!0},"onInput");t.addEventListener(
"input",k),t.value="",["file-input","photo-input","mask-input"].forEach(E=>{const q=get(E);q&&(q.value=
"")}),t.style.height="auto",currentImageUrls=[],currentMaskImage=null,browserFastLocalFiles.clear(),
markerAppliedUploads.clear(),uploadProgressState={total:0,completed:0,active:0,perFilePct:{}},n&&n.replaceChildren(),
clearQuote(),updateFilePreview(),schedulePromptTokenEstimate(!0),updateMaskPreview();let S;i&&e&&!m&&
i.animate&&(S=i.animate([{height:a+"px"},{height:i.getBoundingClientRect().height+"px"}],{duration:350,
easing:"cubic-bezier(0.22, 1, 0.36, 1)"}));const C=r(()=>{!w||x||(g.files.forEach(E=>{var N,W;const q=(W=
(N=E==null?void 0:E.rowObj)==null?void 0:N.row)==null?void 0:W.getAttribute("data-local-url");q&&URL.
revokeObjectURL(q)}),b.remove())},"releaseFiles"),M=r((E,q=!0)=>{w||(w=!0,E=E&&g.navigation===threadLoadSequence,
x=E,t.removeEventListener("input",k),E?(!v&&!t.value&&(t.value=g.text,t.style.height="auto",t.style.
height=Math.min(t.scrollHeight,150)+"px"),currentImageUrls=Array.from(new Set([...g.images,...currentImageUrls])),
currentMaskImage||(currentMaskImage=g.mask),g.files.forEach((N,W)=>browserFastLocalFiles.set(W,N)),g.
markers.forEach(N=>markerAppliedUploads.add(N)),n&&n.prepend(...g.rows),b.remove(),!currentQuote&&g.
quote&&(currentQuote=g.quote,currentQuoteMessageId=g.quoteId,renderQuoteBarContent(g.quote),get("quo\
te-bar").classList.add("visible")),uploadProgressState.total||(uploadProgressState=g.progress),updateFilePreview(),
updateMaskPreview(),schedulePromptTokenEstimate(!0)):q&&C())},"finish");return{source:d,canRetry:r(()=>!v&&
g.navigation===threadLoadSequence,"canRetry"),stopDock:r(()=>S==null?void 0:S.cancel(),"stopDock"),files:g.
files,releaseFiles:C,accept:r((E=!0)=>M(!1,E),"accept"),restore:r(()=>M(!0),"restore")}}r(takePromptForSend,
"takePromptForSend");function animatePromptToMessage(e,t,n){var b;let i,a=0,o=!1;const l=[],c=get("c\
hat-container"),d=get("prompt-input"),m=window.matchMedia("(prefers-reduced-motion: reduce)"),g=r(()=>{
var v,w;o||(o=!0,(v=window.ActivityLog)==null||v.log("ui.hide"),t.stopDock(),cancelAnimationFrame(a),
l.forEach(x=>x.cancel()),i==null||i.remove(),e==null||e.classList.remove("prompt-send-hidden"),window.
removeEventListener("resize",g),window.removeEventListener("popstate",g),window.removeEventListener(
"pagehide",g),document.removeEventListener("visibilitychange",g),c==null||c.removeEventListener("whe\
el",g),c==null||c.removeEventListener("touchstart",g),d==null||d.removeEventListener("input",g),(w=window.
visualViewport)==null||w.removeEventListener("resize",g),m.removeEventListener("change",g))},"stop");
if(!n||m.matches||!e||!Element.prototype.animate)return g;try{e.classList.remove("fade-in"),e.classList.
add("prompt-send-hidden"),a=requestAnimationFrame(()=>{var v;try{if(!e.isConnected)return g();const w=e.
querySelector(".message-bubble"),x=w.getBoundingClientRect(),k=c.getBoundingClientRect();if(!x.width||
!x.height||x.bottom<=k.top||x.top>=k.bottom)return g();const S=t.source;i=w.cloneNode(!0),i.removeAttribute(
"id"),i.querySelectorAll("[id]").forEach(W=>W.removeAttribute("id")),i.querySelectorAll(".msg-contro\
ls, video, audio").forEach(W=>W.remove()),i.classList.add("prompt-send-overlay"),i.setAttribute("ari\
a-hidden","true"),i.inert=!0,Object.assign(i.style,{width:x.width+"px",height:x.height+"px",left:x.left+
"px",top:x.top+"px"}),document.body.appendChild(i),(v=window.ActivityLog)==null||v.log("ui.show");const C=Math.
min(S.width/x.width,Math.max(S.height,42)/x.height,1),M=S.left+(S.width-x.width*C)/2-x.left,E=S.top+
(S.height-x.height*C)/2-x.top,q=i.animate([{transform:`translate(${M}px, ${E}px) scale(${C})`,borderRadius:"\
12px",backgroundColor:getComputedStyle(get("input-row")).backgroundColor},{transform:"translate(0, 0\
) scale(1)",borderRadius:getComputedStyle(w).borderRadius,backgroundColor:getComputedStyle(w).backgroundColor}],
{duration:350,easing:"cubic-bezier(0.22, 1, 0.36, 1)",fill:"both"});l.push(q),l.push(i.animate([{opacity:1,
offset:0},{opacity:1,offset:.8},{opacity:0}],{duration:350,fill:"both"})),e.classList.remove("prompt\
-send-hidden"),l.push(w.animate([{opacity:0,offset:0},{opacity:0,offset:.8},{opacity:1}],{duration:350,
fill:"both"}));const N=r(()=>{if(!e.isConnected||!userAutoScroll||document.hidden)return g();const W=w.
getBoundingClientRect();i.style.left=W.left+"px",i.style.top=W.top+"px",a=requestAnimationFrame(N)},
"watch");a=requestAnimationFrame(N),q.finished.then(g,g)}catch{console.warn("Prompt send animation f\
ailed"),g()}}),window.addEventListener("resize",g),window.addEventListener("popstate",g),window.addEventListener(
"pagehide",g),document.addEventListener("visibilitychange",g),c.addEventListener("wheel",g,{passive:!0}),
c.addEventListener("touchstart",g,{passive:!0}),d.addEventListener("input",g),(b=window.visualViewport)==
null||b.addEventListener("resize",g),m.addEventListener("change",g)}catch{console.warn("Prompt send \
animation failed"),g()}return g}r(animatePromptToMessage,"animatePromptToMessage");let costPricing=null,
costPricingFailed=!1,costBarPath=[],costBarReady=!1;const COST_ESTIMATE_NOTE="\u6709\u6599API\u306E\u901A\u5E38\u6599\u91D1\u306B\u3088\u308B\u6982\u7B97\uFF08USD\uFF09\u3002\u56DE\
\u7B54\u3054\u3068\u306E\u4FDD\u5B58\u6E08\u307F\u30E2\u30C7\u30EB\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5165\u529B\u4F7F\u7528\u91CF\u304C\u306A\u3044\u56DE\u7B54\u306F\u4F1A\u8A71\u5C65\u6B74\u304B\u3089\u63A8\u5B9A\u3057\u307E\u3059\u3002\u30E2\u30C7\u30EBAPI\u306E\u547C\u3073\u51FA\u3057\u3054\u3068\u306B\u5165\u529B\u30FB\u51FA\u529B\uFF08Thinking\u3092\u542B\u3080\uFF09\u3092\u5408\u7B97\u3057\u307E\u3059\u3002Ideogram\u306F\u30E2\u30C7\u30EB\u30FB\u54C1\u8CEA\u30FB\u901F\u5EA6\u30FB\u51E6\
\u7406\u5225\u306E\u5358\u4FA1\u3068\u753B\u50CF\u679A\u6570\u3067\u8A08\u7B97\u3057\u307E\u3059\u3002\u5358\u4FA1\u672A\u516C\u8868\u306E\u8A2D\u5B9A\u306F\u516C\u8868\u6700\u5B89\u5024\u3092\u4E0B\u9650\u3068\u3057\u3066\u300C\u4EE5\u4E0A\u30FB\u4E00\u90E8\u672A\u7B97\u5165\u300D\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u4F7F\u7528\u91CF\u672A\u53D6\u5F97\u306E\u547C\u3073\u51FA\u3057\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002API\u5831\u544A\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u30FB\u66F8\u8FBC\u3068\u691C\u7D22\u306A\u3069\u306E\u8FFD\u52A0\u6599\u91D1\
\u3092\u542B\u307F\u307E\u3059\u3002Batch API\u306F\u4E8B\u696D\u8005\u3054\u3068\u306E\u5272\u5F15\u7387\uFF08\u5272\u5F15\u306E\u306A\u3044\u4E8B\u696D\u8005\u3092\u542B\u3080\uFF09\u3001Gemini Flex\u306F\u6A19\u6E96\u6599\u91D1\u306E50%\u3067\u8A08\u7B97\u3057\u3001\u5272\u5F15\u5BFE\u8C61\u5916\u306E\u30AD\u30E3\u30C3\u30B7\u30E5\u8AAD\u8FBC\u306F\u901A\u5E38\u5358\u4FA1\u306E\u307E\u307E\u3067\u3059\u3002\u4FDD\u5B58\u6642\u9593\u3084\u5B9F\u884C\u6642\u9593\u306A\u3069\u53D6\
\u5F97\u3067\u304D\u306A\u3044\u8AB2\u91D1\u60C5\u5831\u306F\u4E00\u90E8\u672A\u7B97\u5165\u3067\u3059\u3002\u305D\u306E\u4ED6\u306E\u753B\u50CF\u30FB\u97F3\u58F0\u30FB\u52D5\u753B\u751F\u6210\u3001\u7A0E\u3001\u7121\u6599\u67A0\u30FB\u500B\u5225\u5951\u7D04\u306F\u542B\u307F\u307E\u305B\u3093\u3002\u6642\u9593\u5E2F\u3084\u5165\u529B\u7A2E\u5225\u3067\u5358\u4FA1\u304C\u7570\u306A\u308B\u5834\u5408\u306F\u7BC4\u56F2\u3067\u8868\u793A\u3057\u307E\u3059\u3002\u904E\u53BB\u306E\u8ACB\u6C42\u984D\u3092\u518D\u73FE\u3059\u308B\u3082\u306E\u3067\u306F\u3042\u308A\u307E\u305B\u3093\u3002";
function costTokenNumber(e){if(e==null||e==="")return null;const t=Number(e);return Number.isFinite(
t)&&t>=0?t:null}r(costTokenNumber,"costTokenNumber");function costCacheUsage(e){let t=e;if(typeof e==
"string"){if(!e.includes("cache_"))return null;try{t=JSON.parse(e)}catch{return null}}if(!t||typeof t!=
"object"||!Array.isArray(t.pricing_calls))return null;let n=0,i=0,a=!1;const o={};for(const l of t.pricing_calls){
if(!l||typeof l!="object")continue;const c=costTokenNumber(l.cache_read_tokens),d=costTokenNumber(l.
cache_write_tokens);let m=0;for(const g of["5m","1h","30m","12h"]){const b=costTokenNumber(l[`cache_\
write_${g}_tokens`]);b!==null&&(o[g]=(o[g]||0)+b,m+=b,a=!0)}(c!==null||d!==null)&&(a=!0),n+=c||0,i+=
Math.max(d||0,m)}return a?{read:n,write:i,ttls:o}:null}r(costCacheUsage,"costCacheUsage");function costRate(e,t,n){
const i=e&&(e.aliases[t]||t),a=e&&e.models[i];if(!a)return null;const o={...a};for(const l of o.changes||
[])n>=l.from&&Object.assign(o,l);return o}r(costRate,"costRate");function addRequestCost(e,t,n,i,a=!1,o=!1){
var b,v;if(e.requests++,String(t.model).startsWith("grok-")&&Number.isSafeInteger(t.cost_in_usd_ticks)&&
t.cost_in_usd_ticks>=0){e.low+=t.cost_in_usd_ticks/1e10,e.high+=t.cost_in_usd_ticks/1e10,e.priced++,
t.complete===!1&&e.unpriced++;return}const l=costRate(n,t.model,i);if(!l){e.unpriced++;return}if(l.unit===
"image"){const w=costTokenNumber(t.images),x=(b=l.image_rates)==null?void 0:b[t.operation],k=x==null?
void 0:x[l.image_selector==="speed"?t.speed:t.quality],S=(v=k==null?void 0:k.minimum)!=null?v:x==null?
void 0:x.minimum,C=typeof x=="number"?x:typeof k=="number"?k:S,M=Number.isFinite(C)&&typeof x!="numb\
er"&&typeof k!="number";if(w===null||!Number.isFinite(C)){e.unpriced++;return}e.low+=w*C*(o?.5:1),e.
high+=w*C*(o?.5:1),e.priced++,M&&(e.lowerBound=!0),(t.complete===!1||M)&&e.unpriced++;return}const c=costTokenNumber(
t.input_tokens),d=costTokenNumber(t.output_tokens);if(c===null&&d===null){e.unpriced++;return}const m=l.
threshold!=null&&(l.threshold_inclusive?(c||0)>=l.threshold:(c||0)>l.threshold);m&&e.long++;const g=requestTokenCost(
t,l,m,a,o);e.low+=g.low,e.high+=g.high,g.lowerBound&&(e.lowerBound=!0),e.priced++,(c===null||d===null||
t.complete===!1||g.incomplete)&&e.unpriced++}r(addRequestCost,"addRequestCost");function legacyImageRequest(e,t){
let n=e.image_url;try{typeof n=="string"&&(n=JSON.parse(n))}catch{n=n?[n]:[]}const i=Array.isArray(n)?
n.length:0,a=t.get(String(e.parent_id)),o=(a==null?void 0:a.prompt_options)||{};let l=a==null?void 0:
a.image_url,c=Array.isArray(l)?l.length>0:!!l&&l!=="[]"&&l!=="null",d=a;const m=new Set;for(;!c&&(d==
null?void 0:d.parent_id)!=null&&!m.has(String(d.id));)m.add(String(d.id)),d=t.get(String(d.parent_id)),
(d==null?void 0:d.role)==="assistant"&&d.image_url&&d.image_url!=="[]"&&d.image_url!=="null"&&(c=!0);
const g=["remove_object","remove_background"].includes(o.ideogram_operation)?o.ideogram_operation:c&&
e.model==="ideogram-4.5"?"edit":"generate";let b=o.ideogram_quality;(!["very_low","low","medium","hi\
gh"].includes(b)||b==="very_low"&&g==="generate")&&(b=g==="edit"?"medium":"high");const v=a?["turbo",
"default","quality"].includes(o.ideogram_speed)?o.ideogram_speed:"default":null;return{model:e.model,
images:i||null,operation:g,quality:b,speed:v,complete:!!a}}r(legacyImageRequest,"legacyImageRequest");
function estimateConversationCost(e,t=costPricing,n=e){var c,d,m,g,b,v;const i={low:0,high:0,priced:0,
unpriced:0,inferred:0,long:0,requests:0,items:[]},a=r((w,x,k,S,C)=>{const M={low:i.low,high:i.high,priced:i.
priced,unpriced:i.unpriced,long:i.long};addRequestCost(i,w,t,x,S,C),i.items.push(describeRequestCost(
i,M,w,x,t,!!S,!!C,!!k))},"capture"),o=new Map((n||[]).filter(Boolean).map(w=>[String(w.id),w])),l=new Set;
for(const w of e||[]){if(!w||w.role!=="assistant"||l.has(String(w.id)))continue;l.add(String(w.id));
let x=null;try{x=typeof w.thought_data=="string"?JSON.parse(w.thought_data):w.thought_data}catch{}!x&&
Array.isArray(w.pricing_calls)&&(x={pricing_calls:w.pricing_calls});const k=((c=/^\d{4}-\d{2}-\d{2}/.
exec(w.created_at||""))==null?void 0:c[0])||(t==null?void 0:t.checked_at),S=((d=w.batch_job)==null?void 0:
d.state)==="JOB_STATE_SUCCEEDED"||w.batch===!0,C=(o.get(String(w.parent_id))||w).prompt_options||w.prompt_options||
{},M=C&&C.gemini_flex===!0||w.flex===!0;if(Array.isArray(x==null?void 0:x.pricing_calls)){x.pricing_calls.
length||i.priced++;for(const I of x.pricing_calls){if(!I||typeof I!="object"){i.unpriced++;continue}
const j=((m=/^\d{4}-\d{2}-\d{2}/.exec(I.created_at||""))==null?void 0:m[0])||k,ne=costRate(t,I.model,
j),oe=(ne==null?void 0:ne.image_selector)==="speed"&&!I.speed&&I.model===w.model?{...I,speed:legacyImageRequest(
w,o).speed}:I;a(oe,j,!1,S,M)}continue}if(((g=costRate(t,w.model,k))==null?void 0:g.unit)==="image"){
a(legacyImageRequest(w,o),k,!1,!1,M);continue}const E=costTokenNumber(w.tokens_out);if(E===null&&!w.
content&&!w.tokens)continue;if(E===null){i.unpriced++;continue}let q=(b=costTokenNumber(x==null?void 0:
x.pricing_input_tokens))!=null?b:costTokenNumber(w.tokens_in),N=!1,W=!1;if(q===null){i.inferred++,W=
!0,q=0;let I=w.parent_id;const j=new Set([String(w.id)]);for(I==null&&(N=!0);I!=null;){const ne=String(
I);if(j.has(ne)||!o.has(ne)){N=!0;break}j.add(ne);const oe=o.get(ne),te=oe.role==="user"?costTokenNumber(
oe.tokens_in):(v=costTokenNumber(oe.tokens_content))!=null?v:costTokenNumber(oe.tokens_out)===null?null:
Math.max(0,Number(oe.tokens_out)-(costTokenNumber(oe.tokens_thought)||0));te===null?N=!0:q+=te,I=oe.
parent_id}}a({model:w.model,input_tokens:q,output_tokens:E,complete:!N},k,W,S,M)}return i}r(estimateConversationCost,
"estimateConversationCost");function costMoneyText(e){const t=Number(e||0);return t>0&&t<1e-6?"<$0.0\
00001":`$${t.toFixed(6)}`}r(costMoneyText,"costMoneyText");function costAmountText(e){return!e||!e.priced?
"\u7B97\u51FA\u4E0D\u53EF":e.lowerBound?`${costMoneyText(e.low)}\u4EE5\u4E0A`:e.high-e.low>1e-12?`${costMoneyText(
e.low)}\u301C${costMoneyText(e.high)}`:costMoneyText(e.low)}r(costAmountText,"costAmountText");function describeRequestCost(e,t,n,i,a,o,l,c){
const d=r(x=>costTokenNumber(x),"count");let m=d(n.cache_write_tokens)||0;for(const x of["5m","1h","\
30m","12h"])m=Math.max(m,d(n[`cache_write_${x}_tokens`])||0);const g=["5m","1h","30m","12h"].reduce(
(x,k)=>x+(d(n[`cache_write_${k}_tokens`])||0),0),b=["web_search_calls","file_search_calls","groundin\
g_queries","code_execution_calls"].reduce((x,k)=>x+(d(n[k])||0),0),v=e.priced>t.priced,w=v?e.unpriced>
t.unpriced?"\u4E00\u90E8\u672A\u7B97\u5165":"":costRate(a,n.model,i)?"\u4F7F\u7528\u91CF\u672A\u53D6\u5F97":
"\u5358\u4FA1\u672A\u767B\u9332";return{model:String(n.model||""),date:i,input:d(n.input_tokens),output:d(
n.output_tokens),cache_read:d(n.cache_read_tokens),cache_write:Math.max(m,g)||null,images:d(n.images),
tools:b,low:e.low-t.low,high:e.high-t.high,priced:v,reason:w,inferred:!!c,long:e.long>t.long,batch:!!o,
flex:!!l}}r(describeRequestCost,"describeRequestCost");function costItemTokenText(e){var n,i;const t=[];
return e.images!==null&&e.images!==void 0&&t.push(`${e.images}\u679A`),(e.input!==null||e.output!==null)&&
t.push(`\u5165 ${(n=e.input)!=null?n:"-"} / \u51FA ${(i=e.output)!=null?i:"-"}`),(e.cache_read||e.cache_write)&&
t.push(`\u30AD\u30E3\u30C3\u30B7\u30E5 \u8AAD\u8FBC ${e.cache_read||0} / \u66F8\u8FBC ${e.cache_write||
0}`),e.tools&&t.push(`\u30C4\u30FC\u30EB ${e.tools}\u56DE`),t.join(" \xB7 ")}r(costItemTokenText,"co\
stItemTokenText");function costItemJpyText(e,t){if(!t||!e.priced)return"";const n=jpyMoneyText(e.low*
t.rate);return e.high-e.low>1e-12?`\u7D04${n}\u301C${jpyMoneyText(e.high*t.rate)}`:`\u7D04${n}`}r(costItemJpyText,
"costItemJpyText");function appendCostItemList(e,t){const n=document.createElement("div");n.className=
"text-[11px]",n.style.cssText="max-height:16rem;overflow-y:auto;margin-top:0.5rem;",t.forEach((i,a)=>{
const o=document.createElement("div");o.style.cssText="border-top:1px solid rgba(107,114,128,0.4);pa\
dding:0.25rem 0;";const l=document.createElement("div");l.className="flex justify-between";const c=document.
createElement("span");c.className="font-mono",c.textContent=`${a+1}. ${i.model||"\u30E2\u30C7\u30EB\u4E0D\u660E"}`;
const d=document.createElement("span");d.className="font-mono",d.textContent=costAmountText(i),l.append(
c,d),o.appendChild(l);const m=document.createElement("div");m.className="text-gray-400";const g=[];i.
inferred&&g.push("\u5C65\u6B74\u304B\u3089\u63A8\u5B9A"),i.long&&g.push("\u9577\u6587\u6599\u91D1"),
i.batch&&g.push("Batch"),i.flex&&g.push("Flex"),i.reason&&g.push(i.reason);const b=costItemJpyText(i,
costJpyRate),v=[costItemTokenText(i),i.date,g.join(" / "),b].filter(Boolean);v.length&&(m.textContent=
v.join(" \xB7 ")),v.length&&o.appendChild(m),n.appendChild(o)}),e.appendChild(n)}r(appendCostItemList,
"appendCostItemList");function costEstimateLabel(e,t="\u8CBB\u7528\u4E88\u60F3"){return!e||!e.priced?
`${t}: \u7B97\u51FA\u4E0D\u53EF`:`${t}: ${costAmountText(e)}${e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":
""}`}r(costEstimateLabel,"costEstimateLabel");const COST_JPY_RATE_STORAGE_KEY="cost_jpy_rate_v1",COST_JPY_RATE_TTL_MS=720*
60*1e3,COST_JPY_RATE_NOTE="\u7121\u6599\u306E\u516C\u958B\u70BA\u66FF\u30EC\u30FC\u30C8\uFF08\u6B27\u5DDE\u4E2D\u592E\u9280\u884C\u306E\u53C2\u8003\u30EC\u30FC\u30C8\u306A\u3069\uFF09\u3067\u30C9\u30EB\u304B\u3089\u5186\u3078\u63DB\u7B97\u3057\u305F\u6982\u7B97\u3067\u3059\u3002\u5B9F\u969B\u306E\u8ACB\u6C42\u6642\u306E\u30EC\u30FC\u30C8\u3068\u306F\u7570\u306A\u308A\u307E\u3059\u3002",
COST_JPY_RATE_SOURCES=[{name:"ECB",url:"https://api.frankfurter.dev/v2/rates?base=USD&quotes=JPY",read:r(
e=>{const t=Array.isArray(e)?e.find(n=>n&&n.quote==="JPY"):null;return{rate:Number(t==null?void 0:t.
rate),date:t==null?void 0:t.date}},"read")},{name:"currency-api",url:"https://cdn.jsdelivr.net/npm/@\
fawazahmed0/currency-api@latest/v1/currencies/usd.json",read:r(e=>{var t;return{rate:Number((t=e==null?
void 0:e.usd)==null?void 0:t.jpy),date:e==null?void 0:e.date}},"read")}];let costJpyRate=null,costJpyRateFailed=!1,
costJpyRateFailedAt=0,costJpyRatePromise=null;function normalizeJpyRate(e){const t=Number(e==null?void 0:
e.rate);return!Number.isFinite(t)||t<50||t>500?null:{rate:t,date:typeof e.date=="string"?e.date.slice(
0,10):"",source:e.source||""}}r(normalizeJpyRate,"normalizeJpyRate");function readCachedJpyRate(){try{
const e=JSON.parse(localStorage.getItem(COST_JPY_RATE_STORAGE_KEY)||"null");return!e||!(Date.now()-Number(
e.saved)<COST_JPY_RATE_TTL_MS)?null:normalizeJpyRate(e)}catch{return null}}r(readCachedJpyRate,"read\
CachedJpyRate");function loadJpyRate(){if(costJpyRate)return Promise.resolve(costJpyRate);if(costJpyRateFailed&&
Date.now()-costJpyRateFailedAt<6e4)return Promise.resolve(null);if(costJpyRatePromise)return costJpyRatePromise;
costJpyRateFailed=!1;const e=readCachedJpyRate();return e?(costJpyRate=e,Promise.resolve(e)):(costJpyRatePromise=
(async()=>{for(const t of COST_JPY_RATE_SOURCES){let n=null;try{const i=new AbortController;n=setTimeout(
()=>i.abort(),6e3);const a=await fetch(t.url,{signal:i.signal,credentials:"omit"});if(!a.ok)continue;
const o=normalizeJpyRate({...t.read(await a.json()),source:t.name});if(!o)continue;costJpyRate=o;try{
localStorage.setItem(COST_JPY_RATE_STORAGE_KEY,JSON.stringify({...o,saved:Date.now()}))}catch{}return o}catch{}finally{
clearTimeout(n)}}return costJpyRateFailed=!0,costJpyRateFailedAt=Date.now(),null})().finally(()=>{costJpyRatePromise=
null}),costJpyRatePromise)}r(loadJpyRate,"loadJpyRate");function jpyMoneyText(e){return e>0?e<.01?"0\
.01\u5186\u672A\u6E80":e<1e3?`${e.toFixed(2)}\u5186`:`${Math.round(e).toLocaleString("ja-JP")}\u5186`:
"0\u5186"}r(jpyMoneyText,"jpyMoneyText");function costEstimateJpyLabel(e,t){if(!e||!e.priced||!t)return"\
\u65E5\u672C\u5186\u63DB\u7B97: \u7B97\u51FA\u4E0D\u53EF";const n=jpyMoneyText(e.low*t.rate),i=e.lowerBound?
n.endsWith("\u672A\u6E80")?n:`${n}\u4EE5\u4E0A`:e.high-e.low>1e-12?`${n}\u301C${jpyMoneyText(e.high*
t.rate)}`:n,a=e.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":"",o=t.date?`\u30FB${t.date}`:
"";return`\u65E5\u672C\u5186\u63DB\u7B97: \u7D04${i}${a}\uFF08\u53C2\u8003\u30EC\u30FC\u30C8 1\u30C9\u30EB=${t.
rate.toFixed(2)}\u5186${o}\uFF09`}r(costEstimateJpyLabel,"costEstimateJpyLabel");function fillCostEstimateJpy(e,t){
if(e){if(!t||!t.priced){e.textContent=`\u65E5\u672C\u5186\u63DB\u7B97: ${!costPricing&&!costPricingFailed?
"\u63DB\u7B97\u4E2D\u2026":"\u7B97\u51FA\u4E0D\u53EF"}`;return}if(e.title=COST_JPY_RATE_NOTE,e.textContent=
"\u65E5\u672C\u5186\u63DB\u7B97: \u63DB\u7B97\u4E2D\u2026",costJpyRate){e.textContent=costEstimateJpyLabel(
t,costJpyRate);return}loadJpyRate().then(n=>{e.textContent=n?costEstimateJpyLabel(t,n):"\u65E5\u672C\u5186\u63DB\u7B97: \u70BA\u66FF\u30EC\u30FC\u30C8\
\u3092\u53D6\u5F97\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F"})}}r(fillCostEstimateJpy,"fillCo\
stEstimateJpy");function refreshCostBar(e=costBarPath){costBarReady=!0,costBarPath=e;const t=get("to\
tal-cost-estimate");if(!t)return;const n=e.some(a=>a&&a.role==="assistant"&&(a.content||Number(a.tokens_out)>
0));t.classList.toggle("hidden",!n);const i=estimateConversationCost(e,costPricing,e);t.textContent=
costPricing?costEstimateLabel(i,"\u8CBB\u7528\u4E88\u60F3"):`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?
"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`,t.title=COST_ESTIMATE_NOTE,
messageMeta.__total__&&(messageMeta.__total__.cost_estimate=costPricing?i:null),t.onclick=()=>openTokenDetail(
"__total__")}r(refreshCostBar,"refreshCostBar"),fetch("/static/model_pricing.json?checked=2026-11-06",
{credentials:"same-origin"}).then(e=>{if(!e.ok)throw new Error("pricing_unavailable");return e.json()}).
then(e=>{costPricing=e,costBarReady&&refreshCostBar(),usageCostPendingData&&renderUsageCostSummary(usageCostPendingData)}).
catch(()=>{costPricingFailed=!0,costBarReady&&refreshCostBar(),usageCostPendingData&&setUsageCostStatus(
"\u8CBB\u7528\u4E88\u60F3: \u7B97\u51FA\u4E0D\u53EF")});function requestTokenCost(e,t,n,i,a=!1){var G,
ae,R,F,me,re,X,ce,Q,ye;const o=costTokenNumber(e.input_tokens)||0,l=costTokenNumber(e.output_tokens)||
0,c=costTokenNumber(e.cache_read_tokens)||0,d=costTokenNumber(e.cache_write_tokens)||0,m=["5m","1h",
"30m","12h"],g=m.map(ue=>costTokenNumber(e[`cache_write_${ue}_tokens`])||0),b=Math.max(d,g.reduce((ue,ke)=>ue+
ke,0)),v=Math.max(0,o-c-b),w=n?t.long_input:t.input,x=n?t.long_output:t.output,k=n&&(G=t.long_cached_input)!=
null?G:t.cached_input,S=t.cache_write_multipliers||{},C=i&&(ae=t.batch_multiplier)!=null?ae:1,M=i?(R=
t.batch_cached_multiplier)!=null?R:C:1,E=k!=null&&C>0?M/C:1,q=Math.max(0,b-g.reduce((ue,ke)=>ue+ke,0)),
N=Object.values(S);let W=!!e.pricing_incomplete||c+b>o||!!e.explicit_cache;c&&k==null&&(W=!0);let I=v*
w+c*(k!=null?k:w)*E,j=v*(n?w:(F=t.max_input)!=null?F:w)+c*(n?k!=null?k:w:(re=(me=t.max_cached_input)!=
null?me:k)!=null?re:w)*E;for(let ue=0;ue<m.length;ue++){const ke=S[m[ue]];g[ue]&&ke==null&&(W=!0),I+=
g[ue]*w*(ke!=null?ke:1),j+=g[ue]*(n?w:(X=t.max_input)!=null?X:w)*(ke!=null?ke:1)}q&&(W=!0,I+=q*w*(N.
length?Math.min(...N):1),j+=q*(n?w:(ce=t.max_input)!=null?ce:w)*(N.length?Math.max(...N):1));let ne=(I+
l*x)*C*(a?.5:1)/1e6,oe=(j+l*((Q=t.max_output)!=null?Q:x))*C*(a?.5:1)/1e6,te=!1;const ge=t.tool_rates||
{},$=r((ue,ke)=>{if(ue){if(ge[ke]==null){W=!0;return}ne+=ue*ge[ke],oe+=ue*ge[ke]}},"add"),O=costTokenNumber(
e.web_search_calls)||0;$(O,e.search_preview?"web_search_preview":"web_search"),O&&!e.search_preview&&
t.search_content_tokens&&(ne+=O*t.search_content_tokens*w/1e6,oe+=O*t.search_content_tokens*w/1e6),$(
costTokenNumber(e.file_search_calls),"file_search");const V=costTokenNumber(e.grounding_queries)||0;
$(ge.grounding_prompt!=null?Math.min(1,V):V,ge.grounding_prompt!=null?"grounding_prompt":"grounding_\
query");const B=costTokenNumber(e.code_execution_calls)||0;return B&&!e.code_execution_free&&(t.provider===
"openai"?($(((ye=costTokenNumber(e.container_sessions))!=null?ye:1)*(costTokenNumber(e.container_memory_gb)||
1),"container_session"),W=!0):t.provider==="anthropic"?($(1,"code_execution_minimum"),W=!0,te=!0):$(
B,"code_execution")),{low:ne,high:oe,incomplete:W,lowerBound:te}}r(requestTokenCost,"requestTokenCos\
t");const USAGE_COST_PERIODS=[{id:"1d",label:"1\u65E5\u4EE5\u5185"},{id:"7d",label:"7\u65E5\u4EE5\u5185"},
{id:"14d",label:"2\u9031\u9593\u4EE5\u5185"},{id:"1m",label:"1\u304B\u6708\u4EE5\u5185"},{id:"3m",label:"\
3\u304B\u6708\u4EE5\u5185"},{id:"1y",label:"1\u5E74\u9593"}];let usageCostPeriod="1m",usageCostLoadedPeriod=null,
usageCostRequest=0,usageCostPendingData=null;function usageCostElements(){return{panel:get("tab-usag\
e"),status:get("usage-cost-status"),total:get("usage-cost-total"),jpy:get("usage-cost-jpy"),meta:get(
"usage-cost-meta"),breakdown:get("usage-cost-breakdown"),note:get("usage-cost-note")}}r(usageCostElements,
"usageCostElements");function markUsageCostPeriod(e){document.querySelectorAll("#usage-period-button\
s [data-period]").forEach(t=>{const n=t.dataset.period===e;t.classList.toggle("is-active",n),t.setAttribute(
"aria-pressed",n?"true":"false")})}r(markUsageCostPeriod,"markUsageCostPeriod");function setUsageCostStatus(e){
const t=usageCostElements().status;t&&(t.textContent=e)}r(setUsageCostStatus,"setUsageCostStatus");function usageCostModelRows(e){
const t=new Map;for(const n of e||[]){if(!n)continue;const i=n.model||"\u30E2\u30C7\u30EB\u4E0D\u660E";
let a=t.get(i);a||(a={model:i,low:0,high:0,priced:0,unpriced:0,requests:0},t.set(i,a)),a.low+=Number(
n.low)||0,a.high+=Number(n.high)||0,a.requests++,n.priced?a.priced++:a.unpriced++}return[...t.values()].
sort((n,i)=>i.high-n.high||i.low-n.low)}r(usageCostModelRows,"usageCostModelRows");function renderUsageCostBreakdown(e){
const t=usageCostElements().breakdown;if(!t||(t.replaceChildren(),!e.length))return;const n=document.
createElement("div");n.className="text-[11px] font-bold text-gray-400 mt-3",n.textContent="\u30E2\u30C7\u30EB\u5225",
t.appendChild(n),e.forEach(i=>{const a=document.createElement("div");a.className="border-t border-gr\
ay-700/60 py-1.5";const o=document.createElement("div");o.className="flex justify-between gap-3";const l=document.
createElement("span");l.className="font-mono text-[11px] break-all",l.textContent=i.model;const c=document.
createElement("span");c.className="font-mono text-[11px] shrink-0",c.textContent=costAmountText(i)+(i.
unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":""),o.append(l,c),a.appendChild(o);const d=document.
createElement("div");d.className="text-[10px] text-gray-400",d.textContent=`API ${i.requests}\u56DE \xB7 \u7B97\u5B9A ${i.
priced}\u56DE`+(i.unpriced?` \xB7 \u672A\u7B97\u5165 ${i.unpriced}\u56DE`:""),a.appendChild(d),t.appendChild(
a)})}r(renderUsageCostBreakdown,"renderUsageCostBreakdown");function renderUsageCostSummary(e){const t=usageCostElements();
if(!t.panel||!e)return;if(!costPricing){usageCostPendingData=costPricingFailed?null:e,setUsageCostStatus(
`\u8CBB\u7528\u4E88\u60F3: ${costPricingFailed?"\u7B97\u51FA\u4E0D\u53EF":"\u6599\u91D1\u3092\u8AAD\u307F\u8FBC\u307F\u4E2D\u2026"}`);
return}usageCostPendingData=null;const n=Array.isArray(e.messages)?e.messages:[],i=estimateConversationCost(
n,costPricing,n);if(t.total&&(t.textContent=costAmountText(i)+(i.unpriced?"\uFF08\u4E00\u90E8\u672A\u7B97\u5165\uFF09":
""),t.title=COST_ESTIMATE_NOTE),t.jpy&&fillCostEstimateJpy(t.jpy,i),t.meta){const a=[`\u56DE\u7B54 ${e.
answers}\u4EF6`,`API\u547C\u3073\u51FA\u3057 ${i.requests}\u56DE`];i.inferred&&a.push(`\u5C65\u6B74\u304B\u3089\u63A8\u5B9A ${i.
inferred}\u4EF6`),i.unpriced&&a.push(`\u4E00\u90E8\u672A\u7B97\u5165 ${i.unpriced}\u4EF6`),e.unreadable&&
a.push(`\u4F7F\u7528\u91CF\u672A\u53D6\u5F97 ${e.unreadable}\u4EF6`),e.truncated&&a.push("\u4EF6\u6570\u304C\u591A\u3044\u305F\u3081\u76F4\u8FD1\u5206\
\u306E\u307F"),t.meta.textContent=a.join(" \xB7 ")}renderUsageCostBreakdown(usageCostModelRows(i.items)),
setUsageCostStatus(""),t.note&&t.note.classList.remove("hidden")}r(renderUsageCostSummary,"renderUsa\
geCostSummary");function loadUsageCostSummary(e=usageCostPeriod){const t=usageCostElements();if(!t.panel||
(usageCostPeriod=USAGE_COST_PERIODS.some(i=>i.id===e)?e:"1m",markUsageCostPeriod(usageCostPeriod),usageCostLoadedPeriod===
usageCostPeriod&&t.total&&t.total.textContent&&!costPricingFailed))return;const n=++usageCostRequest;
setUsageCostStatus("\u96C6\u8A08\u4E2D\u2026"),fetch(`/api/usage/costs?period=${encodeURIComponent(usageCostPeriod)}`,
{credentials:"same-origin",headers:{Accept:"application/json"}}).then(i=>{if(!i.ok)throw new Error("\
usage_unavailable");return i.json()}).then(i=>{n===usageCostRequest&&(usageCostLoadedPeriod=usageCostPeriod,
renderUsageCostSummary(i))}).catch(()=>{n===usageCostRequest&&setUsageCostStatus("\u96C6\u8A08\u3067\u304D\u307E\u305B\u3093\u3067\u3057\u305F\u3002\u6642\u9593\u3092\u304A\u3044\u3066\u518D\
\u5EA6\u304A\u8A66\u3057\u304F\u3060\u3055\u3044\u3002")})}r(loadUsageCostSummary,"loadUsageCostSumm\
ary");function refreshUsageCostSummary(){usageCostLoadedPeriod=null,loadUsageCostSummary(usageCostPeriod)}
r(refreshUsageCostSummary,"refreshUsageCostSummary");function selectUsageCostPeriod(e){if(usageCostPeriod===
e&&usageCostLoadedPeriod===e){markUsageCostPeriod(e);return}usageCostLoadedPeriod=null,loadUsageCostSummary(
e)}r(selectUsageCostPeriod,"selectUsageCostPeriod");const MATH_STEPS_HEADING_RE=/^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*(\d+)\s*(.*)$/i,
MATH_STEPS_DEFAULT_CONFIG=Object.freeze({basics:!1,level:"easy",step_size:"normal",diagrams:"auto",animation:!0}),
MATH_STEPS_PROGRESS_ROUTE="/api/math_steps/progress",MATH_STEPS_SIMPLIFY_ROUTE="/api/math_steps/simp\
lify",MATH_STEPS_SVG_MAX_CHARS=3e5,MATH_STEPS_MAX_EXPANDED_PER_STEP=8,MATH_STEPS_MAX_DISPLAY_STEPS=50,
mathStepsUiState={master:!1,enabled:!1,config:Object.assign({},MATH_STEPS_DEFAULT_CONFIG),loaded:!1,
modalBusy:!1},mathStepsRootState=new WeakMap,mathStepsTextByRoot=new WeakMap,mathStepsProgressTimers=new WeakMap,
mathStepsProgressInFlight=new WeakSet;function clampMathStepsIndex(e,t){const n=Math.floor(Number(e));
if(!Number.isFinite(n)||n<0)return 0;const i=Math.max(0,Number(t)||0);return Math.min(n,i)}r(clampMathStepsIndex,
"clampMathStepsIndex");function normalizeMathStepsConfig(e){let t=e;if(typeof t=="string")try{t=JSON.
parse(t)}catch{t=null}const n=t&&typeof t=="object"?t:{},i=["easy","normal","detailed"].includes(String(
n.level||""))?String(n.level):MATH_STEPS_DEFAULT_CONFIG.level,a=["small","normal","large"].includes(
String(n.step_size||""))?String(n.step_size):MATH_STEPS_DEFAULT_CONFIG.step_size,o=["auto","off"].includes(
String(n.diagrams||""))?String(n.diagrams):MATH_STEPS_DEFAULT_CONFIG.diagrams;return{basics:n.basics===
!0,level:i,step_size:a,diagrams:o,animation:n.animation!==!1}}r(normalizeMathStepsConfig,"normalizeM\
athStepsConfig");function normalizeMathStepsExpanded(e){const t={};return!e||typeof e!="object"||Array.
isArray(e)||Object.keys(e).forEach(n=>{if(!/^\d{1,4}$/.test(String(n)))return;const i=e[n];if(!Array.
isArray(i))return;const a=[];i.slice(0,MATH_STEPS_MAX_EXPANDED_PER_STEP).forEach(o=>{if(!o||typeof o!=
"object")return;const l=typeof o.body=="string"?o.body:"";if(!l.trim())return;const c=typeof o.title==
"string"?o.title:"";a.push({title:c.trim(),body:l})}),a.length&&(t[String(n)]=a)}),t}r(normalizeMathStepsExpanded,
"normalizeMathStepsExpanded");function normalizeMathStepsMessageState(e){let t=e;if(typeof t=="strin\
g")try{t=JSON.parse(t)}catch{t=null}const n=t&&typeof t=="object"&&!Array.isArray(t)?t:{},i={};return n.
simplified&&typeof n.simplified=="object"&&!Array.isArray(n.simplified)&&Object.keys(n.simplified).forEach(
a=>{const o=n.simplified[a];typeof o=="string"&&o.trim()&&(i[String(a)]=o)}),{current:clampMathStepsIndex(
n.current,50),max_reached:clampMathStepsIndex(n.max_reached,50),simplified:i,expanded:normalizeMathStepsExpanded(
n.expanded)}}r(normalizeMathStepsMessageState,"normalizeMathStepsMessageState");function buildMathStepsDisplaySteps(e,t){
const n=[],i=Array.isArray(e)?e:[],a=t&&typeof t=="object"?t:{};return i.forEach(o=>{const l=a[String(
o.index)];if(Array.isArray(l)&&l.length){l.forEach((c,d)=>{n.push({index:n.length,sourceIndex:o.index,
partIndex:d,expanded:!0,title:String(c.title||o.title||""),body:String(c.body||"")})});return}n.push(
{index:n.length,sourceIndex:o.index,partIndex:null,expanded:!1,title:o.title||"",body:o.body||""})}),
n.length>MATH_STEPS_MAX_DISPLAY_STEPS&&(n.length=MATH_STEPS_MAX_DISPLAY_STEPS),n}r(buildMathStepsDisplaySteps,
"buildMathStepsDisplaySteps");function parseMathSteps(e){const t=String(e||"").replace(/\r\n?/g,`
`),n=[],i=[];let a=null,o=null;const l=t.split(`
`);for(const m of l){if(o){const g=m.trim();g.length>=o.length&&g[0]===o[0]&&/^(`+|~+)$/.test(g)&&(o=
null)}else{const g=m.match(/^\s{0,3}(`{3,}|~{3,})(.*)$/);if(g&&!(g[1][0]==="`"&&g[2].includes("`"))){
o=g[1],(a?a.body:i).push(m);continue}const b=m.match(MATH_STEPS_HEADING_RE);if(b){a&&n.push(a),a={index:n.
length,number:Number(b[1])||n.length+1,title:String(b[2]||"").trim(),body:[]};continue}}(a?a.body:i).
push(m)}a&&n.push(a);const c=n.map((m,g)=>({index:g,number:m.number,title:m.title,body:m.body.join(`\

`).replace(/^\s*\n+/,"").replace(/\n+\s*$/,"")})),d=i.join(`
`).trim();return c.length&&d&&(c[0].body=`${d}

${c[0].body}`.trim()),{steps:c,preamble:d,hasSteps:c.length>0}}r(parseMathSteps,"parseMathSteps"),window.
parseMathSteps=parseMathSteps;function mathStepsReducedMotion(){try{return!!(window.matchMedia&&window.
matchMedia("(prefers-reduced-motion: reduce)").matches)}catch{return!1}}r(mathStepsReducedMotion,"ma\
thStepsReducedMotion");function renderMathStepsBodyHtml(e){const t=String(e||"").trim();return t?sanitizeMarkdownHtml(
t):'<div class="math-steps-empty">\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u306E\u672C\u6587\u3092\u6E96\u5099\u3057\u3066\u3044\u307E\u3059\u2026</div>'}
r(renderMathStepsBodyHtml,"renderMathStepsBodyHtml");function sanitizeMathSvgMarkup(e){const t=String(
e||"").trim();if(!t||t.length>MATH_STEPS_SVG_MAX_CHARS||!/^<svg[\s>]/i.test(t.replace(/^(?:<\?xml[\s\S]*?\?>\s*|<!--[\s\S]*?-->\s*|<!DOCTYPE[^>]*>\s*)*/i,
"")))return"";let n=null;try{n=new DOMParser().parseFromString(t,"image/svg+xml")}catch{return""}if(!n||
!n.documentElement||n.querySelector("parsererror"))return"";const i=n.documentElement;if(String(i.nodeName||
"").toLowerCase()!=="svg")return"";const a=mathStepsReducedMotion()||mathStepsUiState.config.animation===
!1,o=new Set(["script","foreignobject","iframe","object","embed","image","video","audio"]),l=new Set(
["animate","animatetransform","animatemotion","set","discard"]),c=r(m=>{!m||!m.childNodes||Array.from(
m.childNodes).forEach(g=>{if(g.nodeType===1){const b=String(g.nodeName||"").toLowerCase();if(o.has(b)||
a&&l.has(b)){g.remove();return}if(b==="style"){const v=g.textContent||"";if(/@import|url\(\s*["']?\s*(?:https?:)?\/\//i.
test(v)){g.remove();return}if(a&&/@keyframes|animation/i.test(v)){g.remove();return}}Array.from(g.attributes||
[]).forEach(v=>{const w=String(v.name||"").toLowerCase(),x=String(v.value||"");if(w.startsWith("on")){
g.removeAttribute(v.name);return}(w==="href"||w==="xlink:href")&&!/^#/.test(x.trim())&&g.removeAttribute(
v.name),a&&/animation(-name|-duration|-iteration-count)?$/.test(w)&&g.removeAttribute(v.name),w==="s\
tyle"&&/url\(\s*["']?\s*(?:https?:)?\/\//i.test(x)&&g.removeAttribute(v.name)}),c(g)}})},"walk");c(i);
let d="";try{d=new XMLSerializer().serializeToString(i)}catch{return""}return/^<svg[\s>]/i.test(d)?d:
""}r(sanitizeMathSvgMarkup,"sanitizeMathSvgMarkup");function buildMathSvgRenderHtml(e){const t=sanitizeMathSvgMarkup(
e);if(!t)return"";let n=t.replace(/<svg\b([^>]*)>/i,(o,l)=>{let c=l;return/\sxmlns\s*=/i.test(c)||(c=
` xmlns="http://www.w3.org/2000/svg"${c}`),/\bxlink:/i.test(t)&&!/\sxmlns:xlink\s*=/i.test(c)&&(c=` \
xmlns:xlink="http://www.w3.org/1999/xlink"${c}`),`<svg${c}>`});const i=`data:image/svg+xml;charset=u\
tf-8,${encodeURIComponent(n)}`,a=hashString(t);return`<span class="mathsvg-render" data-svg-key="${escapeHtml(
a)}"><img src="${i}" alt="\u56F3" loading="lazy" decoding="async"></span>`}r(buildMathSvgRenderHtml,
"buildMathSvgRenderHtml"),window.buildMathSvgRenderHtml=buildMathSvgRenderHtml;function mathStepsStateAttr(e){
try{return escapeHtml(JSON.stringify(e||{}))}catch{return""}}r(mathStepsStateAttr,"mathStepsStateAtt\
r");function mathStepsStepMarkdown(e,t){const n=e.steps[t];if(!n)return"";if(n.expanded)return n.body;
const i=e.simplified?e.simplified[String(n.sourceIndex)]:"",a=e.showOriginal&&e.showOriginal.has(t);
return i&&!a?i:n.body}r(mathStepsStepMarkdown,"mathStepsStepMarkdown");function mathStepsCanRestore(e,t){
const n=e.steps[t];return n?n.expanded?!0:!!((e.simplified?e.simplified[String(n.sourceIndex)]:"")&&
!e.showOriginal.has(t)):!1}r(mathStepsCanRestore,"mathStepsCanRestore");function mathStepsTotalLabel(e){
const t=e.steps.length;return`${e.current+1} / ${t}${e.streaming?"+":""}`}r(mathStepsTotalLabel,"mat\
hStepsTotalLabel");function mathStepsPendingNext(e){return!!e.streaming&&e.current>=e.steps.length-1}
r(mathStepsPendingNext,"mathStepsPendingNext");function buildMathStepsHtml(e,t={}){const n=parseMathSteps(
e);if(!n.steps.length)return"";const i=normalizeMathStepsMessageState(t.state||{}),a=n.steps,o=buildMathStepsDisplaySteps(
a,i.expanded),l={originalSteps:a,expanded:Object.assign({},i.expanded),steps:o,current:clampMathStepsIndex(
t.current!==void 0?t.current:i.current,o.length-1),maxReached:Math.max(i.max_reached,i.current),simplified:Object.
assign({},i.simplified),showOriginal:new Set,streaming:!!t.streaming},c=l.steps[l.current],d=mathStepsStepMarkdown(
l,l.current),m=mathStepsCanRestore(l,l.current),g=mathStepsPendingNext(l),b=l.current>=l.steps.length-
1,v=b&&!g?"\u5B8C\u4E86":g?"\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u6E96\u5099\u4E2D\u2026":"\u308F\u304B\u3063\
\u305F\u3001\u6B21\u3078",w=g||b&&!g,x=t.messageId!==void 0&&t.messageId!==null?String(t.messageId):
"",k=t.jobId!==void 0&&t.jobId!==null?String(t.jobId):"";return`<div class="math-steps content-area"\
 role="group" aria-label="\u6570\u5B66\u30B9\u30C6\u30C3\u30D7\u89E3\u8AAC" ${['data-math-steps-root\
="1"',`data-current="${l.current}"`,`data-max-reached="${l.maxReached}"`,`data-total="${l.steps.length}\
"`,`data-streaming="${l.streaming?"1":"0"}"`,x?`data-message-id="${escapeHtml(x)}"`:"",k?`data-job-i\
d="${escapeHtml(k)}"`:"",`data-math-steps-state="${mathStepsStateAttr({current:i.current,max_reached:i.
max_reached,simplified:i.simplified,expanded:i.expanded})}"`].filter(Boolean).join(" ")}><div class=\
"math-steps-header"><span class="math-steps-counter">\u30B9\u30C6\u30C3\u30D7 ${mathStepsTotalLabel(
l)}</span><span class="math-steps-title">${escapeHtml(c.title||"")}</span></div><div class="math-ste\
ps-body" data-step-index="${l.current}" aria-live="polite"><div class="prose prose-invert text-sm br\
eak-words math-steps-prose">${renderMathStepsBodyHtml(d)}</div>`+(m?'<button type="button" class="ma\
th-steps-restore" data-math-steps-action="restore">\u5143\u306E\u8AAC\u660E\u306B\u623B\u3059</button>':
"")+`</div><div class="math-steps-actions"><button type="button" class="math-steps-next" data-math-s\
teps-action="next"${w?' disabled aria-disabled="true"':""}>${escapeHtml(v)}</button><div class="math\
-steps-sub-actions"><button type="button" class="math-steps-prev" data-math-steps-action="prev"${l.current<=
0?' disabled aria-disabled="true"':""}>\u524D\u3078</button><button type="button" class="math-steps-restart" d\
ata-math-steps-action="restart"${l.current<=0?' disabled aria-disabled="true"':""}>\u6700\u521D\u304B\u3089</button><but\
ton type="button" class="math-steps-simplify" data-math-steps-action="simplify"${l.streaming?' disab\
led aria-disabled="true"':""} title="\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u3001\u3084\u3055\u3057\u3044\u8907\u6570\u306E\u30B9\u30C6\u30C3\u30D7\u306B\u5206\u3051\u307E\u3059"><i class="fas fa-wand-magic-sparkles"\
 aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F</button></div></div><div class="math-steps-status" role="status" ari\
a-live="polite"></div></div>`}r(buildMathStepsHtml,"buildMathStepsHtml"),window.buildMathStepsHtml=buildMathStepsHtml;
function buildMathStepsBubbleHtml(e,t={}){if(!parseMathSteps(e).steps.length)return"";const i=t.mathSteps||
{};return buildMathStepsHtml(e,{state:i.state,messageId:i.messageId,jobId:i.jobId,streaming:!!i.streaming})}
r(buildMathStepsBubbleHtml,"buildMathStepsBubbleHtml"),window.buildMathStepsBubbleHtml=buildMathStepsBubbleHtml;
function mathStepsRootCounterEl(e){return e?e.querySelector(".math-steps-counter"):null}r(mathStepsRootCounterEl,
"mathStepsRootCounterEl");function mathStepsStatus(e,t){const n=e?e.querySelector(".math-steps-statu\
s"):null;n&&(n.textContent=String(t||""))}r(mathStepsStatus,"mathStepsStatus");function getMathStepsRootState(e){
if(!e)return null;let t=mathStepsRootState.get(e);if(t)return t;const n=e.getAttribute("data-message\
-id")||"";let i=mathStepsTextByRoot.get(e)||"";!i&&n&&typeof messageStore=="object"&&messageStore&&(i=
messageStore[n]||"");const a=parseMathSteps(i);if(!a.steps.length)return null;const o=normalizeMathStepsMessageState(
e.getAttribute("data-math-steps-state")||""),l=a.steps,c=buildMathStepsDisplaySteps(l,o.expanded);return t=
{originalSteps:l,expanded:Object.assign({},o.expanded),steps:c,current:clampMathStepsIndex(e.getAttribute(
"data-current"),c.length-1),maxReached:Math.max(o.max_reached,clampMathStepsIndex(e.getAttribute("da\
ta-max-reached"),c.length-1)),simplified:Object.assign({},o.simplified),showOriginal:new Set,streaming:e.
getAttribute("data-streaming")==="1"},mathStepsRootState.set(e,t),t}r(getMathStepsRootState,"getMath\
StepsRootState");function paintMathStepsRoot(e){const t=getMathStepsRootState(e);if(!e||!t||!t.steps.
length)return;t.current=clampMathStepsIndex(t.current,t.steps.length-1);const n=t.steps[t.current],i=mathStepsRootCounterEl(
e);i&&(i.textContent=`\u30B9\u30C6\u30C3\u30D7 ${mathStepsTotalLabel(t)}`);const a=e.querySelector("\
.math-steps-title");a&&(a.textContent=n.title||"");const o=e.querySelector(".math-steps-body"),l=mathStepsStepMarkdown(
t,t.current),c=mathStepsCanRestore(t,t.current);if(o){const x=new Map;o.querySelectorAll(".mathsvg-r\
ender[data-svg-key]").forEach(k=>{const S=k.getAttribute("data-svg-key");S&&!x.has(S)&&x.set(S,k)}),
o.setAttribute("data-step-index",String(t.current)),o.innerHTML=`<div class="prose prose-invert text\
-sm break-words math-steps-prose">${renderMathStepsBodyHtml(l)}</div>`+(c?'<button type="button" cla\
ss="math-steps-restore" data-math-steps-action="restore">\u5143\u306E\u8AAC\u660E\u306B\u623B\u3059</button>':
""),o.querySelectorAll(".mathsvg-render[data-svg-key]").forEach(k=>{const S=x.get(k.getAttribute("da\
ta-svg-key"));S&&k.replaceWith(S)});try{queueMessageDecorations(o,l)}catch{}}const d=mathStepsPendingNext(
t),m=t.current>=t.steps.length-1,g=e.querySelector(".math-steps-next");if(g){g.textContent=m&&!d?"\u5B8C\u4E86":
d?"\u6B21\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u6E96\u5099\u4E2D\u2026":"\u308F\u304B\u3063\u305F\u3001\u6B21\u3078";
const x=d||m&&!d;g.disabled=x,g.setAttribute("aria-disabled",x?"true":"false")}const b=e.querySelector(
".math-steps-prev");if(b){const x=t.current<=0;b.disabled=x,b.setAttribute("aria-disabled",x?"true":
"false")}const v=e.querySelector(".math-steps-restart");if(v){const x=t.current<=0;v.disabled=x,v.setAttribute(
"aria-disabled",x?"true":"false")}const w=e.querySelector(".math-steps-simplify");w&&!w.dataset.busy&&
(w.disabled=!!t.streaming,w.setAttribute("aria-disabled",t.streaming?"true":"false")),e.setAttribute(
"data-current",String(t.current)),e.setAttribute("data-max-reached",String(t.maxReached)),e.setAttribute(
"data-total",String(t.steps.length)),e.setAttribute("data-streaming",t.streaming?"1":"0"),t.renderedStep=
t.current,t.renderedMarkdown=l}r(paintMathStepsRoot,"paintMathStepsRoot");function renderMathStepsInto(e,t,n={}){
if(!e)return!1;const i=parseMathSteps(t);if(!i.steps.length)return!1;const a=e.querySelector(".math-\
steps"),o=a?getMathStepsRootState(a):null,l=normalizeMathStepsMessageState(n.state||{}),c=o||{originalSteps:i.
steps,expanded:Object.assign({},l.expanded),steps:buildMathStepsDisplaySteps(i.steps,l.expanded),current:l.
current,maxReached:l.max_reached,simplified:Object.assign({},l.simplified),showOriginal:new Set,streaming:!!n.
streaming};c.originalSteps=i.steps,c.expanded=c.expanded||{},c.steps=buildMathStepsDisplaySteps(c.originalSteps,
c.expanded),c.streaming=n.streaming===!0,c.current=clampMathStepsIndex(c.current,c.steps.length-1),c.
maxReached=Math.max(c.maxReached,c.current);const d=a&&o&&o.renderedStep===c.current?o.renderedMarkdown:
null,m=mathStepsStepMarkdown(c,c.current),g=a?a.querySelector(".math-steps-body"):null,b=!!g&&d!==null&&
d===m,v=buildMathStepsHtml(t,{state:{current:c.current,max_reached:c.maxReached,simplified:c.simplified,
expanded:c.expanded},messageId:n.mathSteps&&n.mathSteps.messageId!==void 0?n.mathSteps.messageId:void 0,
jobId:n.mathSteps&&n.mathSteps.jobId!==void 0?n.mathSteps.jobId:void 0,streaming:c.streaming});if(!v)
return!1;e.innerHTML=v;const w=e.querySelector(".math-steps");if(!w)return!1;mathStepsRootState.set(
w,c),mathStepsTextByRoot.set(w,String(t||"")),c.renderedStep=c.current,c.renderedMarkdown=m;const x=w.
querySelector(".math-steps-body");if(b&&g&&x){const k=new Map;g.querySelectorAll(".mathsvg-render[da\
ta-svg-key]").forEach(S=>{const C=S.getAttribute("data-svg-key");C&&!k.has(C)&&k.set(C,S)}),x.querySelectorAll(
".mathsvg-render[data-svg-key]").forEach(S=>{const C=k.get(S.getAttribute("data-svg-key"));C&&S.replaceWith(
C)}),x.replaceWith(g)}else if(x)try{queueMessageDecorations(x,m)}catch{}return!0}r(renderMathStepsInto,
"renderMathStepsInto"),window.renderMathStepsInto=renderMathStepsInto;function mathStepsRootTargets(e){
if(!e)return null;const t=e.getAttribute("data-message-id"),n=e.getAttribute("data-job-id");return t&&
/^\d+$/.test(String(t))?{message_id:Number(t)}:n?{job_id:String(n)}:null}r(mathStepsRootTargets,"mat\
hStepsRootTargets");function postMathStepsProgress(e,t=0){const n=getMathStepsRootState(e),i=mathStepsRootTargets(
e);if(!n||!i||mathStepsProgressInFlight.has(e))return;const a=Object.assign({},i,{current:n.current,
max_reached:n.maxReached});return mathStepsProgressInFlight.add(e),apiFetch(MATH_STEPS_PROGRESS_ROUTE,
{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(a)}).then(o=>{if(!(o&&
o.ok))throw new Error(`progress ${o?o.status:"failed"}`)}).catch(()=>{t<4&&window.setTimeout(()=>postMathStepsProgress(
e,t+1),1200*(t+1))}).finally(()=>{mathStepsProgressInFlight.delete(e)})}r(postMathStepsProgress,"pos\
tMathStepsProgress");function bindMathStepsMessageId(e,t){if(!(!e||t===void 0||t===null))return e.setAttribute(
"data-message-id",String(t)),postMathStepsProgress(e)}r(bindMathStepsMessageId,"bindMathStepsMessage\
Id"),window.bindMathStepsMessageId=bindMathStepsMessageId;function scheduleMathStepsProgress(e){if(!e)
return;const t=mathStepsProgressTimers.get(e);t&&window.clearTimeout(t),mathStepsProgressTimers.set(
e,window.setTimeout(()=>{mathStepsProgressTimers.delete(e),postMathStepsProgress(e)},450))}r(scheduleMathStepsProgress,
"scheduleMathStepsProgress");function forgetMathStepsVariants(e,t){const n=mathStepsRootTargets(e),i=getMathStepsRootState(
e);!n||!i||apiFetch(MATH_STEPS_PROGRESS_ROUTE,{method:"POST",headers:{"Content-Type":"application/js\
on"},body:JSON.stringify(Object.assign({},n,{current:i.current,max_reached:i.maxReached,remove_simplified:[
t],remove_expanded:[t]}))}).catch(()=>{})}r(forgetMathStepsVariants,"forgetMathStepsVariants");function requestMathStepsSimplify(e,t){
const n=getMathStepsRootState(e),i=mathStepsRootTargets(e);if(!n||!i)return;if(!i.message_id){showToast(
"\u56DE\u7B54\u304C\u4FDD\u5B58\u3055\u308C\u305F\u5F8C\u306B\u300C\u3082\u3063\u3068\u3084\u3055\u3057\u304F\u300D\u3092\u4F7F\u3048\u307E\u3059",
"info",!1);return}if(n.streaming){showToast("\u56DE\u7B54\u306E\u751F\u6210\u304C\u7D42\u308F\u3063\u3066\u304B\u3089\u4F7F\u3048\u307E\u3059",
"info",!1);return}const a=n.steps[n.current];if(!a)return;const o={message_id:i.message_id,step_index:a.
sourceIndex};a.partIndex!==null&&a.partIndex!==void 0&&(o.part_index=a.partIndex),t&&(t.dataset.busy=
"1",t.disabled=!0,t.innerHTML='<i class="fas fa-circle-notch fa-spin" aria-hidden="true"></i>\u3084\u3055\u3057\u304F\u3057\u3066\u3044\
\u307E\u3059\u2026'),mathStepsStatus(e,"\u3053\u306E\u30B9\u30C6\u30C3\u30D7\u3092\u3001\u3084\u3055\u3057\u3044\u8907\u6570\u306E\u30B9\u30C6\u30C3\u30D7\u306B\u5206\u3051\u3066\u3044\u307E\u3059\u2026"),
apiFetch(MATH_STEPS_SIMPLIFY_ROUTE,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.
stringify(o)}).then(async l=>{const c=await l.json().catch(()=>({}));let d=c&&Array.isArray(c.steps)?
c.steps:[];if(!d.length&&c&&typeof c.text=="string"&&c.text.trim()&&(d=[{title:a.title,body:c.text}]),
!l.ok||!c||c.status!=="ok"||!d.length){const b=l.status===409?"\u3053\u308C\u4EE5\u4E0A\u30B9\u30C6\u30C3\u30D7\u3092\u5897\u3084\u305B\u307E\u305B\u3093":
c&&c.error?c.error:"\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F";
throw new Error(b)}const m=normalizeMathStepsExpanded({[String(a.sourceIndex)]:d})[String(a.sourceIndex)];
if(!m||!m.length)throw new Error("\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F");
const g=normalizeMathStepsMessageState(c.math_steps_state||{});n.simplified=Object.assign({},g.simplified),
n.expanded=Object.assign({},n.expanded,g.expanded,{[String(a.sourceIndex)]:m}),n.steps=buildMathStepsDisplaySteps(
n.originalSteps,n.expanded),n.current=clampMathStepsIndex(n.current,n.steps.length-1),n.showOriginal.
delete(n.current),t&&(delete t.dataset.busy,t.disabled=!1,t.innerHTML='<i class="fas fa-wand-magic-s\
parkles" aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F'),mathStepsStatus(e,`\u3084\u3055\u3057\u3044\
\u8AAC\u660E\u306B\u5206\u3051\u307E\u3057\u305F\uFF08${m.length}\u30B9\u30C6\u30C3\u30D7\uFF09`),paintMathStepsRoot(
e)}).catch(l=>{mathStepsStatus(e,""),showToast(l&&l.message?l.message:"\u3084\u3055\u3057\u3044\u8AAC\u660E\u3092\u4F5C\u308C\u307E\u305B\u3093\u3067\u3057\u305F",
"error",!0),t&&(delete t.dataset.busy,t.disabled=!1,t.innerHTML='<i class="fas fa-wand-magic-sparkle\
s" aria-hidden="true"></i>\u3082\u3063\u3068\u3084\u3055\u3057\u304F')})}r(requestMathStepsSimplify,
"requestMathStepsSimplify");function handleMathStepsAction(e){const t=e.target&&e.target.closest?e.target.
closest("[data-math-steps-action]"):null;if(!t)return;const n=t.closest(".math-steps");if(!n)return;
const i=getMathStepsRootState(n);if(!i)return;const a=t.getAttribute("data-math-steps-action");if(a===
"simplify"){e.preventDefault(),requestMathStepsSimplify(n,t);return}if(e.preventDefault(),a==="resto\
re"){const o=i.steps[i.current];if(!o)return;if(o.expanded){const l=i.steps.findIndex(c=>c.sourceIndex===
o.sourceIndex);delete i.expanded[String(o.sourceIndex)],delete i.simplified[String(o.sourceIndex)],i.
steps=buildMathStepsDisplaySteps(i.originalSteps,i.expanded),i.current=clampMathStepsIndex(l>=0?l:0,
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
n);else return;paintMathStepsRoot(n)}r(handleMathStepsAction,"handleMathStepsAction");function mathStepsChip(){
return get("enable-math-steps")}r(mathStepsChip,"mathStepsChip");function setMathStepsEnabled(e,t={}){
const n=!!e&&mathStepsUiState.master;mathStepsUiState.enabled=n;const i=mathStepsChip();return i&&i.
checked!==n&&(i.checked=n),n}r(setMathStepsEnabled,"setMathStepsEnabled"),window.setMathStepsEnabled=
setMathStepsEnabled;function applyMathStepsMaster(e){mathStepsUiState.master=!!e;const t=get("math-s\
teps-container");t&&t.classList.toggle("hidden",!mathStepsUiState.master),mathStepsUiState.master||setMathStepsEnabled(
!1,{persist:!1})}r(applyMathStepsMaster,"applyMathStepsMaster");function applyMathStepsSettings(e){!e||
typeof e!="object"||(mathStepsUiState.config=normalizeMathStepsConfig(Object.prototype.hasOwnProperty.
call(e,"math_steps_config")?e.math_steps_config:mathStepsUiState.config),Object.prototype.hasOwnProperty.
call(e,"math_steps_enabled")&&applyMathStepsMaster(e.math_steps_enabled===!0),mathStepsUiState.loaded=
!0)}r(applyMathStepsSettings,"applyMathStepsSettings"),window.applyMathStepsSettings=applyMathStepsSettings;
function applyMathStepsChatDefaults(e,t){if(!e||typeof e!="object")return;if(applyMathStepsSettings(
e),!mathStepsUiState.master){setMathStepsEnabled(!1,{persist:!1});return}const n=Object.prototype.hasOwnProperty.
call(e,"last_math_steps")?e.last_math_steps:void 0,i=Object.prototype.hasOwnProperty.call(e,"default\
_math_steps")?e.default_math_steps:!1;if(t&&n!==void 0&&n!==null){setMathStepsEnabled(n===!0,{persist:!1});
return}setMathStepsEnabled(i===!0,{persist:!1})}r(applyMathStepsChatDefaults,"applyMathStepsChatDefa\
ults"),window.applyMathStepsChatDefaults=applyMathStepsChatDefaults;function collectMathStepsSendOption(){
return!!(mathStepsUiState.master&&mathStepsUiState.enabled)}r(collectMathStepsSendOption,"collectMat\
hStepsSendOption"),window.collectMathStepsSendOption=collectMathStepsSendOption;function openMathStepsModal(){
const e=get("math-steps-config-modal");if(!e)return;const t=normalizeMathStepsConfig(mathStepsUiState.
config);get("math-steps-cfg-basics")&&(get("math-steps-cfg-basics").checked=t.basics),get("math-step\
s-cfg-level")&&(get("math-steps-cfg-level").value=t.level),get("math-steps-cfg-step-size")&&(get("ma\
th-steps-cfg-step-size").value=t.step_size),get("math-steps-cfg-diagrams")&&(get("math-steps-cfg-dia\
grams").value=t.diagrams),get("math-steps-cfg-animation")&&(get("math-steps-cfg-animation").checked=
t.animation);const n=get("math-steps-config-status");if(n&&(n.textContent=""),typeof showModal=="fun\
ction"?showModal("math-steps-config-modal"):e.classList.remove("hidden"),location.pathname!=="/math-\
steps")try{history.pushState({modal:"math-steps"},"","/math-steps")}catch{}}r(openMathStepsModal,"op\
enMathStepsModal"),window.openMathStepsModal=openMathStepsModal;function closeMathStepsModal(e=!1){const t=get(
"math-steps-config-modal");if(t&&(typeof hideModal=="function"?hideModal("math-steps-config-modal"):
t.classList.add("hidden")),!e&&location.pathname==="/math-steps")try{history.back()}catch{}}r(closeMathStepsModal,
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
mathStepsUiState.modalBusy=!1,t&&(t.disabled=!1)}}r(saveMathStepsConfig,"saveMathStepsConfig"),window.
saveMathStepsConfig=saveMathStepsConfig;function buildMathStepsPdfHtml(e,t){const n=parseMathSteps(e);
if(!n.steps.length)return"";const i=normalizeMathStepsMessageState(t);return buildMathStepsDisplaySteps(
n.steps,i.expanded).map(o=>{const l=o.expanded?"":i.simplified[String(o.sourceIndex)],c=o.expanded?o.
body:l&&l.trim()?l:o.body;return`<div class="math-steps-pdf-step"><h3 class="math-steps-pdf-heading"\
>\u30B9\u30C6\u30C3\u30D7 ${o.index+1}${o.title?` ${pdfEscapeAttr(o.title)}`:""}</h3><div class="mat\
h-steps-pdf-body">${sanitizeMarkdownHtml(c)}</div>`+(o.expanded||l&&l.trim()?'<div class="math-steps\
-pdf-note">\uFF08\u3084\u3055\u3057\u3044\u8AAC\u660E\uFF09</div>':"")+"</div>"}).join("")}r(buildMathStepsPdfHtml,
"buildMathStepsPdfHtml"),window.buildMathStepsPdfHtml=buildMathStepsPdfHtml;function prepareMathStepsCopyText(e){
const t=String(e||"");if(!t)return t;const n=r(o=>String(o||"").replace(/(`{3,}|~{3,})mathsvg[^\n]*\n[\s\S]*?\n\1[ \t]*(?=\n|$)/gi,
`\uFF3B\u56F3\uFF3D
`),"withoutMathSvg"),i=parseMathSteps(t);if(!i.steps.length)return n(t);const a=[];return i.steps.forEach(
o=>{const l=o.title?` ${o.title}`:"";a.push(`## \u30B9\u30C6\u30C3\u30D7 ${o.index+1}${l}`),a.push(""),
a.push(n(o.body).trim()),a.push("")}),a.join(`
`).replace(/\n{3,}/g,`

`).trim()}r(prepareMathStepsCopyText,"prepareMathStepsCopyText"),window.prepareMathStepsCopyText=prepareMathStepsCopyText;
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
r(initMathSteps,"initMathSteps"),window.initMathSteps=initMathSteps,document.addEventListener("DOMCo\
ntentLoaded",()=>{try{initMathSteps()}catch(e){console.error("math steps init failed",e)}},{once:!0});
