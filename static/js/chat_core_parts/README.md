# chat_core_parts — チャットコアの部品ファイル

`static/js/chat_core.v4.8.*.js`（約2.2万行）は、編集しやすくするためにこのディレクトリの**順序付き部品**（`chat_core.partNN_名前.js`）に分割されています。`scripts/build_frontend.sh` が部品を番号順に連結して結合ソース `chat_core.v4.8.*.js` を再生成し、それを圧縮して `chat_core.min.v4.8.*.js` を作ります。

- 部品は**連結順に番号が振られており、順番を変えてはいけません**（変えると動作が壊れます）。
- 結合ソースは部品の連結と**バイト単位で一致**することが検証で保証されています。
- **編集対象は部品ファイル**です。結合ソースと圧縮ファイルは手で編集せず、編集後は必ず `scripts/build_frontend.sh` を実行してください。

## 部品の探し方

1. 変更したい要素（`get('some-id')` の ID、関数名、`window.xxx` など）を `grep` で探す。
2. ヒットした部品ファイルだけを読んで編集する。必要なら隣接する部品（前後の番号）も確認する。
3. 部品を編集したら `scripts/build_frontend.sh` → 必要に応じて `scripts/prepare_version.sh` / `scripts/publish_version.sh`。

## 各部品の概要

| 部品 | 行数 | 主な内容 |
|---|---|---|
| `part01_bootstrap_utils.js` | ~1,580 | 基盤ユーティリティ。`get()` ヘルパー、設定保存ボタン制御、画像読み込み失敗フォールバック、テーマ（`applyThemeColor` / `THEME_STORAGE_KEY`）、Liquid Glass、圧縮設定、アダプティブぼかし（低負荷自動化）、モーダルの開閉（`showModal` / `hideModal`） |
| `part02_rich_paste.js` | ~1,580 | リッチペースト（HTML整形・サニタイズ・印刷）、MathJax読み込み、Google連携解除、各種定数（添付・低帯域・一時チャット・Coding/Canvas・マーカー） |
| `part03_security_token_promptbar.js` | ~1,550 | セキュリティ（polyfill.io等のブロック）、トークン見積もり、プロンプトバー表示、Thinkingレベル、コード折りたたみ、低帯域モード適用 |
| `part04_minimal_options_thinking.js` | ~1,580 | ミニマル表示のプラスボタンポップアップ、Thinkingスライダー、設定タブ（`TAB_LABELS`）、Botロックオーバーレイ、SWキャッシュモード（`applyCacheMode`）、Turnstile初期化 |
| `part05_settings_modal.js` | ~1,610 | 設定モーダル本体。モデル定義 `MODELS` 配列、`MODEL_NAME_BY_ID` / `MODEL_TAGS`、音声（TTS/STS）一覧、画像・音声・動画拡張子、スラッシュコマンド定義、ウェルカム画面クイックアクセス |
| `part06_model_media_prompt_cache.js` | ~490 | モデル一覧描画・Visionモデル選択、PromptCache 制約、ブラウザ高速モードの選択肢制御 |
| `part07_domcontent_initial.js` | ~1,700 | DOMContentLoaded 初期処理（前半）。テーマ初期化、高速モードトグル、アカウントエクスポート/インポート、一時チャットトグル、セッション管理UI |
| `part08_domcontent_account_transfer.js` | ~1,520 | アカウント移行（エクスポート/インポート/重複修復）続き、管理者スレッド暗号化、自動システムプロンプト設定、テーマ設定バインド、`MODAL_CONFIG`（URLルート↔モーダル対応）と `closeModalById` |
| `part09_domcontent_popstate_modals.js` | ~1,290 | `popstate` ハンドラ（戻る/進むでモーダル開閉・スレッド復元）、2FA（TOTP/WebAuthn）、リアルタイム音声セッション（Gemini Live・`RealtimeVoiceSession`・音声再生）、フィードバック管理 |
| `part10_domcontent_final.js` | ~1,410 | Lyria RealTime スタジオ、音声スタジオUI（`VoiceStudio`）、ファイルライブラリ／Gemモーダルの閉じる・選択更新 |
| `part11_file_preview_upload.js` | ~1,390 | ファイルプレビュー・画像ビューア、マーカー編集モーダル、カメラキャプチャ、アップロード行の描画 |
| `part12_upload_camera_canvas.js` | ~1,610 | アップロード進捗、カメラストリーム制御、マーカー編集（ブラシ・モザイク・切り抜き）、Canvasモバイル表示、音声キャプチャ |
| `part13_canvas_coding_stream.js` | ~1,490 | Canvasプレビュー生成、Codingモード、ストリーム描画（ペンディング表示）、スラッシュコマンドパレット、Gem候補（`@`）、Python実行モーダル |
| `part14_send_message_browser_fast.js` | ~1,660 | 送信処理 `sendMessage`、ブラウザ直接送信・高速モード、プル更新、Xリンク自動検索、`/settings`（AI設定アシスタント）、引用・返信 |
| `part15_slash_tempchat_threads.js` | ~1,350 | 一時チャット（ハートビート・タイムアウト）、スレッド設定モーダル、圧縮設定モーダル、PDFエクスポート、Gem適用、ブランチ集計 |
| `part16_gems_branch_debug.js` | ~670 | ファイルライブラリ操作、ブランチ管理UI、メッセージコピー、規約・Alpha案内モーダル、Thinking表示切替、クライアントデバッグログ |

> 注: `part07`〜`part10` は `document.addEventListener('DOMContentLoaded', () => { ... })` という1つの大きな初期化コールバックを4分割したものです。各部品は文の切れ目で区切られていますが、単体では不完全なJSになります（連結すると完全一致します）。境界の前後を調べるときは `part07`〜`part10` をまとめて参照してください。
