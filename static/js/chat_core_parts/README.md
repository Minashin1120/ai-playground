# chat_core_parts — チャットコアの部品ファイル

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `chat_core.part01_bootstrap_utils.js` | 基盤ユーティリティ。`get()` ヘルパー、設定保存ボタン制御、画像読み込み失敗フォールバック、テーマ（`applyThemeColor` / `THEME_STORAGE_KEY`）、Liquid Glass、圧縮設定、アダプティブぼかし（低負荷自動化）、モーダルの開閉（`showModal` / `hideModal`） | 基盤・テーマ・モーダルを変更するとき |
| `chat_core.part02_rich_paste.js` | リッチペースト（HTML整形・サニタイズ・印刷）、MathJax読み込み、Google／Minashin連携解除・連携UI（`updateGoogleLinkUI` / `updateMinashinLinkUI`）、各種定数（添付・低帯域・一時チャット・Coding/Canvas・マーカー） | リッチペースト・共通定数を変更するとき |
| `chat_core.part03_security_token_promptbar.js` | セキュリティ（polyfill.io等のブロック）、トークン見積もり、プロンプトバー表示、Thinkingレベル、コード折りたたみ、低帯域モード適用 | セキュリティ・トークン・Coding対象を変更するとき |
| `chat_core.part04_minimal_options_thinking.js` | ミニマル表示のプラスボタンポップアップ、Thinkingスライダー、設定タブ（`TAB_LABELS`）、Botロックオーバーレイ、SWキャッシュモード（`applyCacheMode`）、Turnstile初期化 | ミニマル表示・Thinking・キャッシュを変更するとき |
| `chat_core.part05_settings_modal.js` | 設定モーダル本体。モデル定義 `MODELS` 配列、`MODEL_NAME_BY_ID` / `MODEL_TAGS`、音声（TTS/STS）一覧、画像・音声・動画拡張子、スラッシュコマンド定義、ウェルカム画面クイックアクセス | 設定画面・モデル定義を変更するとき |
| `chat_core.part06_model_media_prompt_cache.js` | モデル一覧描画・Visionモデル選択、PromptCache 制約、ブラウザ高速モードとBatchの選択肢制御、MCP（外部モデル連携）設定UI（`loadMcpServers` / `renderMcpServers` / `bindMcpSettingsUi`） | モデル表示・PromptCache・MCP設定を変更するとき |
| `chat_core.part07_domcontent_initial.js` | DOMContentLoaded 初期処理（前半）。テーマ初期化、高速モード・Batchトグル、アカウントエクスポート/インポート、一時チャットトグル、セッション管理UI | 初期化・アカウント設定を変更するとき |
| `chat_core.part08_domcontent_account_transfer.js` | アカウント移行（エクスポート/インポート/重複修復）続き、管理者スレッド暗号化、自動システムプロンプト設定、テーマ設定バインド、`MODAL_CONFIG`（URLルート↔モーダル対応）と `closeModalById` | アカウント移行・モーダルのURLを変更するとき |
| `chat_core.part09_domcontent_popstate_modals.js` | `popstate` ハンドラ（戻る/進むでモーダル開閉・スレッド復元）、2FA（TOTP/WebAuthn）、リアルタイム音声セッション（Gemini Live・`RealtimeVoiceSession`・音声再生）、フィードバック管理 | 戻る操作・認証・音声・フィードバックを変更するとき |
| `chat_core.part10_domcontent_final.js` | Lyria RealTime スタジオ、音声スタジオUI（`VoiceStudio`）、ファイルライブラリ／Gemモーダルの閉じる・選択更新 | 音声スタジオ・ライブラリの初期化を変更するとき |
| `chat_core.part11_file_preview_upload.js` | ファイルプレビュー・画像ビューア、マーカー編集モーダル、カメラキャプチャ、アップロード行の描画 | ファイル表示・アップロードを変更するとき |
| `chat_core.part12_upload_camera_canvas.js` | アップロード進捗、カメラストリーム制御、マーカー編集（ブラシ・モザイク・切り抜き）、Canvasモバイル表示、音声キャプチャ | 画像編集・Canvas・音声入力を変更するとき |
| `chat_core.part13_canvas_coding_stream.js` | Canvasプレビュー生成、Codingモード、ストリーム描画（ペンディング・Batch状態表示）、スラッシュコマンドパレット、Gem候補（`@`）、Python実行モーダル | 吹き出し・Canvas・ストリーム表示を変更するとき |
| `chat_core.part14_send_message_browser_fast.js` | 送信処理 `sendMessage`、ブラウザ直接送信・高速モード・各プロバイダのBatch送信、プル更新、Xリンク自動検索、`/settings`（AI設定アシスタント）、引用・返信 | 送信・直接送信・履歴一覧を変更するとき |
| `chat_core.part15_slash_tempchat_threads.js` | 一時チャット（ハートビート・タイムアウト）、各プロバイダのBatch状態ポーリング・完了通知、スレッド設定モーダル、圧縮設定モーダル、PDFエクスポート、Gem適用、ブランチ集計 | 履歴読み込み・一時チャット・スレッド設定を変更するとき |
| `chat_core.part16_gems_branch_debug.js` | ファイルライブラリ操作、ブランチ管理UI、Batch処理一覧モーダル（取得・停止・履歴削除・チャットを開く）、メッセージコピー、規約・Alpha案内モーダル、Thinking表示切替、クライアントデバッグログ、管理者のBot検出ログ画面（`BotAdminLog`：記録のあるアカウントの一覧・一括削除、アカウントごとの状態と記録。開閉とURLは part08） | ブランチ管理・ライブラリ・Batch一覧を変更するとき |
| `chat_core.part17_thread_branch.js` | 別チャットへの全分岐コピー、分岐元リンク、一時チャット保存警告、欠損ファイル表示 | 別チャットへの分岐・分岐元・欠損ファイル表示を変更するとき |
| `../../../scripts/build_frontend.sh` | 部品を番号順に連結し、トップレベルの `//! @chat-core-bundle-split` で配信用JSを分割・圧縮する | JS部品の編集後にビルドするとき。結合ソース・圧縮ファイルは手で編集しない |
