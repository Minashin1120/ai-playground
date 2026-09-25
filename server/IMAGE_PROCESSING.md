# 画像処理の設計

サムネイル生成、画像形式の変換、生成画像の検証、SVG描画、2段階認証のQRコード生成の方針と制約をまとめます。実装は `image_tools.py` にあります。画像を扱う処理を変更する前に読んでください。

## 方針

- **常駐プロセスではPillowを読み込みません。** 常駐プロセスとは、gunicornのワーカーとRQのワーカーです。`app.py` の先頭にある `sys.modules["PIL"] = None` で読み込みを禁止しています。
  - 理由の1つ目は、メモリと起動時間です。V4.8.1028からV4.8.1029への変更で、`import app` 直後のRSSは1プロセスあたり約10MB（約277MB→約267MB）減り、読み込みは約1秒速くなりました。
  - 理由の2つ目は、以前Pillowの読み込みで障害が起きた疑いがあることです。ネイティブコードによる画像の解析を、アプリ本体から切り離すためです。
- google-genai、openpyxl、pypdf、qrcodeは、Pillowを任意で読み込むだけなので、Pillowなしでも動きます。
- **画素を扱う処理は、使い捨ての子プロセスに任せます。** 使うのは `cwebp`、`dwebp`、`ffmpeg`、CairoSVGです。Python側で行うのは、ヘッダーの解析、構造の検証、PNGの書き出し、子プロセスの呼び出しだけです。
- **Pillowは `requirements.txt` に残しています。** WeasyPrint（PDF生成）とCairoSVG（SVG→PNG）が必須の依存としているためです。どちらも子プロセスで動くので、Pillowは子プロセスの中でだけ読み込まれます。
- リッチペーストPDFのReportLab代替経路は廃止しました。ReportLabはPillowが必須のためです。WeasyPrintが失敗した場合は `pdf_generation_failed`（500）を返します。

## 処理ごとの経路

| 処理 | 呼び出し元 | 実装 | 失敗したとき |
|---|---|---|---|
| サムネイル（WebP） | `routes_files.py` の `serve_file_thumb` | `_make_thumbnail_webp`。PNG・JPEG・静止画WebPは `cwebp`、GIF・BMP・AVIFと、`cwebp` が失敗した場合は `ffmpeg`（`libwebp`） | 元ファイルへリダイレクトする |
| PNGへの変換（Grok、OpenAIの画像編集） | `background.py` | `_convert_image_to_png`。静止画WebPは `dwebp`、それ以外は `ffmpeg`（先頭フレームだけ） | 元データのまま送る |
| マスク（OpenAIの画像編集） | `background.py` | 8ビットのRGBA PNGで構造が正しければ、そのまま使う。それ以外は `_convert_image_to_png(..., rgba=True)` | `Failed to process mask file.` |
| 縦横比（Grokの画像→動画） | `background.py` | `_probe_image` | 縦横比を指定しない |
| 生成画像の検証 | `agentic_media.py` | `_probe_image` と `_validate_image_structure` | `not a supported image` |
| SVG→PNG | `agentic_media.py` | サニタイズの後、`_rasterize_svg_png`（`python -m cairosvg` を子プロセスで実行） | `Rasterized generated image is invalid` |
| TOTPのQRコード | `routes_settings.py` | `_qr_png_bytes`（`qrcode` の `get_matrix()` と `_encode_png` による1ビットPNG） | — |

## 形式ごとの対応

| 形式 | ヘッダー解析 | サムネイル | PNGへの変換 |
|---|---|---|---|
| PNG | IHDR | `cwebp` | `ffmpeg`（`png_pipe`） |
| JPEG | SOFまで走査 | `cwebp` | 変換対象外（そのまま送れる形式） |
| WebP（静止画） | VP8 / VP8L / VP8X | `cwebp` | `dwebp` |
| WebP（アニメーション） | VP8Xのフラグで判別 | 非対応（元画像で代用） | 非対応（元データのまま送る） |
| GIF | 論理画面サイズ | `ffmpeg`（`gif`、先頭フレーム） | `ffmpeg` |
| BMP | DIBヘッダー | `ffmpeg`（`bmp_pipe`） | `ffmpeg` |
| AVIF | ftypのブランドと `ispe` | `ffmpeg`（`mov`） | `ffmpeg` |
| HEIC | ftypのブランドと `ispe` | 非対応（元画像で代用） | 非対応 |

## 安全上の制約（変更しても外さないこと）

- **画素数の上限。** `_IMAGE_MAX_PIXELS`（4,000万画素。旧 `Image.MAX_IMAGE_PIXELS` と同じ値）を、ヘッダーから求めた寸法で確認してからコマンドを起動します。寸法が分からない画像は処理しません。
- **ffmpegに入力形式を自動判別させない。** 必ず `-protocol_whitelist pipe` を付け、`_probe_image` で判別した形式に対応する入力形式を `-f` で固定します（`_FFMPEG_DEMUXERS`）。自動判別させると、プレイリスト系の形式を装った入力によって、ローカルファイルやURLを読みに行かせることができてしまいます。
- **入出力は標準入出力だけ。** アップロードは暗号化して保存しているため、復号した画像を一時ファイルとしてディスクに書きません。
- **コマンドの実行方法。** シェルは使わず、引数はリストで渡します。`nice -n 10` で優先度を下げます。タイムアウトはサムネイル10秒、変換とSVGは30秒です。出力の上限はサムネイル2MB、PNG 50MBです。
- **出力の検証。** 出力はシグネチャと寸法で確認し、合わないものは使いません。
- **サムネイルの同時生成数。** サムネイルの生成は、`storage.py` の生成枠（プロセス内で1件、Redisで全体を調整）で直列化しています。

## 既知の制限と、以前との違い

- HEICのサムネイルと変換には対応していません（Pillowのときも非対応でした）。
- アニメーションWebPは、`dwebp` もffmpeg 5.1も読めません。
- `cwebp` の縮小は、libwebpの縮小処理です（以前はPillowのLanczos）。EXIFの向きは以前と同じく反映しません。
- 透過のあるパレット画像は、サムネイルでも透過を保ちます（以前はRGBに変換していました）。
- `ffmpeg` によるPNG変換では、ピクセル形式をffmpegが自動で選びます（RGB、RGBA、パレットなど）。`rgba=True` のときだけRGBAに固定します。

## 依存

- **OSパッケージ**: `webp`（`cwebp`／`dwebp`）と `ffmpeg`（AVIFにはlibdav1dが必要）です。確認した版は、Debian 12のlibwebp 1.2.4とffmpeg 5.1です。
  - コマンドのパスは読み込み時に `shutil.which` で解決します。見つからない場合、アプリは起動し、各処理が上の表の「失敗したとき」の動作になります。
- **Pythonパッケージ**: `pillow`（子プロセス専用）、`CairoSVG`、`qrcode` です。`reportlab` は依存から外しました。

## 変更するときのルール

- `app.py` と `server/*.py` で `PIL`、`cairosvg`、`reportlab` をimportしないでください。`tests/test_image_tools_regressions.py` の静的チェックと読み込みチェックで検出されます。
- テストでもPillowは使えません（`app` を読み込んだ後はimportできません）。テスト用の画像は、`_encode_png`、ffmpegの `lavfi`、base64の定数で作ってください。
- 新しい形式に対応する手順:
  1. `_probe_image` にヘッダー解析を加える。
  2. `_FFMPEG_DEMUXERS` に入力形式を加える（`ffmpeg -demuxers` にあり、ファイル参照を伴わないもの）。
  3. `cwebp` が読める形式なら `_CWEBP_INPUT_FORMATS` にも加える。
  4. `storage.py` の `_IMAGE_THUMB_EXTS` を確認し、テストにフィクスチャを加える。
- 本番で確認するときは、`grep -l _imaging /proc/<ワーカーのPID>/maps` が何も返さないことを見ます。これでPillowのネイティブモジュールが読み込まれていないことを確認できます。
