"""Exercise the real split renderer in Chromium, without uploading user data."""
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / 'static/js/chat_core_parts'


class ImageSplitWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest('playwright is not installed')
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch(executable_path=shutil.which('chromium'))
        except Exception as e:
            cls.pw.stop()
            raise unittest.SkipTest(f'Chromium is unavailable: {e}')
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(ROOT / 'templates'), autoescape=True)
        cls.markup = env.get_template('chat/overlay_image_split.html').render()
        source = (PARTS / 'chat_core.part01_bootstrap_utils.js').read_text()
        # imageFilenameForMime is part of the shared pipeline, not duplicated in the fixture.
        cls.compression = source[source.index('        const IMAGE_EXTENSION_BY_MIME ='):source.index('        const setCompressionSettings =')]
        cls.script = (PARTS / 'chat_core.part18_image_split.js').read_text()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.context = self.browser.new_context(accept_downloads=True)
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.page.route('https://split.test/**', lambda route: route.fulfill(body='<html><body></body></html>', content_type='text/html'))
        self.page.goto('https://split.test/')
        self.page.set_content('<style>.hidden{display:none}</style>' + self.markup + '''<button id="upload-split-btn"></button><input id="image-split-input" type="file" multiple>
            <input id="enable-compression" type="checkbox" checked>''')
        self.page.add_script_tag(content='''
            const get = id => document.getElementById(id);
            const showModal = id => get(id).classList.remove('hidden');
            const hideModal = id => get(id).classList.add('hidden');
            const showToast = () => {};
            const formatBytes = String;
            const getCompressionMaxSizeMB = () => 1;
            const getCompressionMaxDim = () => 1920;
            const getCompressionOutputType = () => 'original';
            const getCompressionFormatOnly = () => false;
            const ensureImageCompression = async () => {};
            let compressionCalls = [];
            window.imageCompression = async (file, options) => {
                compressionCalls.push(options); return file;
            };
            let browserFastModeEnabled = false;
            const browserFastLocalFiles = new Map();
            const BROWSER_FAST_MAX_IMAGES = 4, BROWSER_FAST_MAX_BYTES = 12 * 1024 * 1024;
            let attached = null, attachOptions = null;
            const handleFiles = async (files, options) => { attached = files; attachOptions = options; };
            async function makeSource(width=101, height=67) {
                const c = document.createElement('canvas'); c.width=width; c.height=height;
                const ctx=c.getContext('2d'); ctx.fillStyle='#12ab34'; ctx.fillRect(0,0,width,height);
                ctx.clearRect(0,0,1,1);
                const blob=await new Promise(resolve=>c.toBlob(resolve,'image/png'));
                return new File([blob], 'sample.png', {type:'image/png'});
            }
            async function dimensions(file) {
                const b=await createImageBitmap(file); const size=[b.width,b.height]; b.close(); return size;
            }
        ''' + self.compression + self.script)
        self.page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")

    def test_grid_overlap_bounds_and_reading_order(self):
        result = self.page.evaluate('''() => {
            const g=chooseSplitGrid(4,101,67), t=splitTiles(101,67,g,10);
            return {g,t, square:chooseSplitGrid(2,100,100), portrait:chooseSplitGrid(5,100,500),
                fits:splitFits(chooseSplitGrid(30,32,32),32,32),
                name:splitTileName('photo', {index:5,row:1,column:1}, {columns:4,rows:3})};
        }''')
        self.assertEqual(result['g'], {'columns': 2, 'rows': 2})
        self.assertEqual(result['square'], {'columns': 2, 'rows': 1})
        self.assertEqual(result['portrait'], {'columns': 1, 'rows': 5})
        self.assertFalse(result['fits'])
        self.assertEqual(result['name'], 'photo_part06of12_r2c2.png')
        self.assertEqual([(t['row'], t['column']) for t in result['t']], [(0,0),(0,1),(1,0),(1,1)])
        self.assertEqual((result['t'][0]['right'], result['t'][1]['left']), (53,47))
        self.assertEqual((result['t'][-1]['right'],result['t'][-1]['bottom']), (101,67))

    def test_lossless_original_dimensions_pixels_and_transparency(self):
        result = self.page.evaluate('''async () => {
            const source=await makeSource();
            const files=await renderSplitFiles(source,{pieces:4,overlap:0,includeOriginal:true,markOriginal:false,markPieces:false},{enabled:false});
            const b=await createImageBitmap(files[1]), c=splitCanvas(b.width,b.height), ctx=c.getContext('2d'); ctx.drawImage(b,0,0);
            const pixels=[...ctx.getImageData(0,0,2,1).data]; b.close();
            return {sizes:await Promise.all(files.map(dimensions)), pixels, original:files[0]===source,
                types:files.map(f=>f.type), compressionCalls:compressionCalls.length};
        }''')
        self.assertEqual(result['sizes'], [[101,67],[50,33],[51,33],[50,34],[51,34]])
        self.assertEqual(result['pixels'], [0,0,0,0,18,171,52,255])
        self.assertTrue(result['original'])
        self.assertEqual(result['compressionCalls'],0)
        self.assertEqual(set(result['types']), {'image/png'})

    def test_marked_outputs_caption_and_overview(self):
        result = self.page.evaluate('''async () => {
            const files=await renderSplitFiles(await makeSource(400,300),{pieces:4,overlap:10,includeOriginal:true,markOriginal:true,markPieces:true},{enabled:false});
            return {sizes:await Promise.all(files.map(dimensions)), names:files.map(f=>f.name)};
        }''')
        self.assertEqual(result['sizes'][0],[400,300])
        self.assertEqual(result['sizes'][1],[210,170])  # 158px crop plus the 12px caption strip
        self.assertEqual(result['names'][0], 'sample_split_grid.png')

    def test_settings_independent_and_compression_applied_only_once(self):
        self.page.evaluate("localStorage.setItem('compression_max_dim','4096')")
        self.page.evaluate('async () => await openImageSplit([await makeSource()])')
        self.page.click('#image-split-settings')
        self.assertTrue(self.page.is_checked('#image-split-compression-enabled'))
        self.assertEqual(self.page.input_value('#image-split-compression-max-dim'),'1920')
        self.page.fill('#image-split-compression-max-dim','512')
        self.page.click('#image-split-compression-save')
        self.assertEqual(self.page.evaluate("localStorage.getItem('compression_max_dim')"), '4096')
        self.page.evaluate('''async () => { await openImageSplit([await makeSource()]); await runImageSplit(true); }''')
        self.assertEqual(self.page.evaluate('attached.length'),5)
        self.assertEqual(self.page.evaluate('compressionCalls.length'),5)
        self.assertEqual(self.page.evaluate('compressionCalls.map(x=>x.maxWidthOrHeight)'),[512]*5)
        self.assertEqual(self.page.evaluate('attachOptions'), {'skipCompression':True})
        self.page.evaluate('openSplitCompression()')
        self.assertEqual(self.page.input_value('#image-split-compression-max-dim'),'512')

    def test_format_only_disabled_fields_and_save_without_upload(self):
        self.page.evaluate("localStorage.setItem('image_split_compression', JSON.stringify({enabled:false}))")
        self.page.evaluate('''async () => await openImageSplit([await makeSource(),await makeSource()])''')
        self.page.evaluate('openSplitCompression()')
        self.page.check('#image-split-compression-enabled')
        self.page.check('#image-split-compression-format-only')
        self.assertTrue(self.page.is_disabled('#image-split-compression-max-dim'))
        self.page.click('#image-split-compression-close')
        self.page.evaluate('runImageSplit(false)')
        self.assertEqual(self.page.locator('#image-split-downloads a').count(),10)
        self.assertIsNone(self.page.evaluate('attached'))
        self.page.evaluate('closeImageSplit()')
        self.assertEqual(self.page.evaluate('splitState.urls.length'),0)

    def test_bad_image_tiny_image_and_fast_mode_fail_without_upload(self):
        self.page.evaluate('''async () => await openImageSplit([new File(['bad'],'bad.png',{type:'image/png'})])''')
        self.assertIn('画像を読み込めません',self.page.text_content('#image-split-status'))
        self.assertTrue(self.page.is_disabled('#image-split-attach'))
        self.page.evaluate('''async () => await openImageSplit([await makeSource(16,16)])''')
        self.assertIn('小さすぎる',self.page.text_content('#image-split-dimensions'))
        self.page.evaluate('''async () => { await openImageSplit([await makeSource()]); browserFastModeEnabled=true; await runImageSplit(true); }''')
        self.assertIn('4枚',self.page.text_content('#image-split-status'))
        self.assertIsNone(self.page.evaluate('attached'))

    def test_jpeg_orientation_is_applied_before_splitting(self):
        sizes = self.page.evaluate("""async () => {
            const c=splitCanvas(80,40); c.getContext('2d').fillRect(0,0,80,40);
            const blob=await new Promise(r=>c.toBlob(r,'image/jpeg'));
            const bytes=new Uint8Array(await blob.arrayBuffer());
            // APP1 / EXIF, big endian TIFF, one Orientation=6 (90 degrees) entry.
            const exif=new Uint8Array([255,225,0,34,69,120,105,102,0,0,77,77,0,42,0,0,0,8,0,1,1,18,0,3,0,0,0,1,0,6,0,0,0,0,0,0]);
            const source=new File([bytes.slice(0,2),exif,bytes.slice(2)],'rotated.jpg',{type:'image/jpeg'});
            const out=await renderSplitFiles(source,{pieces:4,overlap:0,includeOriginal:false,markPieces:false},{enabled:false});
            return await Promise.all(out.map(dimensions));
        }""")
        self.assertEqual(sizes, [[40,20]]*4)  # Portrait ties prefer four rows.

    def test_format_conversion_only_does_not_resize(self):
        result = self.page.evaluate("""async () => {
            window.imageCompression.drawFileInCanvas = async file => {
                const image=await createImageBitmap(file), canvas=splitCanvas(image.width,image.height);
                canvas.getContext('2d').drawImage(image,0,0); return [image,canvas];
            };
            window.imageCompression.cleanupCanvasMemory = canvas => {canvas.width=0;canvas.height=0;};
            const files=await renderSplitFiles(await makeSource(1000,680),
                {pieces:4,overlap:0,includeOriginal:false,markPieces:false},
                {enabled:true,formatOnly:true,outputType:'image/jpeg',maxDimension:256,maxSizeMB:0.1});
            return {sizes:await Promise.all(files.map(dimensions)),types:files.map(f=>f.type),names:files.map(f=>f.name),calls:compressionCalls.length};
        }""")
        self.assertEqual(result['sizes'], [[500,340]]*4)
        self.assertEqual(result['types'], ['image/jpeg']*4)
        self.assertTrue(all(name.endswith('.jpg') for name in result['names']))
        self.assertEqual(result['calls'], 0)
