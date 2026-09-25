# server — Flask 本体の分割ソース

| ファイル | 主な内容 | いつ開くか |
|---|---|---|
| `../app.py` | 起動・版番号と `_SERVER_PARTS`。部品は定義順に `exec()` され `app` の名前空間を共有するため、通常の `import server.*` にはしない | 版番号と部品の読み込み順だけを確認するとき |
| `mobile_auth.py` | Android専用Bearer認証、API許可範囲、HTTPS・Cookie分離、レスポンス保護。全フックより先に読み込む | ネイティブ認証、端末トークン、API権限 |
| `routes_mobile.py` | Android端末連携、確認コード承認、トークン発行・失効、接続仕様API | Android連携フロー、接続情報 |
| `routes_mobile_auth.py` | Androidのネイティブ新規登録・パスワード／パスキーログイン・TOTP／WebAuthn 2FA・セキュリティ管理・初回セットアップとZIP取り込み・旧ブラウザー連携 | アプリ内認証、2FA・パスキー管理、初回設定 |
| `request_hooks.py` | `@app.before_request`。古いログイン flash の除去、設定保存 flash の掃除、ユーザー別アップロード上限 | リクエスト前処理、flash 漏れ、アップロードサイズ |
| `storage.py` | 容量制限、アップロードパス、添付の正規化、PDF/DOCX テキスト抽出、チャンクアップロード、サムネイルとメディアのメモリキャッシュ | ファイル保存、容量、添付、チャンク |
| `image_tools.py` | 画像ヘッダー解析・構造検証、cwebp／dwebp／ffmpeg／CairoSVG を子プロセスで使うサムネイル・PNG変換・SVG描画、PNG書き出しとTOTP用QR。Pillowは読み込まない | サムネイル、画像形式の変換、画像検証、QRコード |
| `crypto.py` | 暗号化鍵リング、`encrypt_val` / `decrypt_val`、バイト暗号化、TTS 音声選択、`secure_delete` | 暗号化、鍵、削除 |
| `providers.py` | モデル種別判定、Mistral OCR、Gemini 文字起こし、PCM/WAV 変換、生成バイトの保存 | プロバイダ分岐、OCR、STT、音声形式 |
| `create_file.py` | チャットの `create_file` ツール（txt/md/pdf/docx/xlsx をライブラリへ保存） | ファイル作成ツール |
| `edit_file.py` | チャットの `edit_file` ツール（xlsx/docx/pdf の編集） | ファイル編集ツール |
| `agentic_media.py` | エージェント画像の SVG サニタイズ、sandbox 画像 URL の書き換え、生成音声の保存 | 画像エージェント、SVG、sandbox 画像 |
| `settings_ai.py` | 設定モーダルの AI アシスタント、文字起こし設定、Vision 解析、Realtime 音声の補助 | 設定の AI 更新、文字起こしプロンプト |
| `lyria.py` | Lyria RealTime のサーバー側セッション（Google へ WebSocket、ブラウザへ SSE） | リアルタイム音楽 |
| `realtime.py` | OpenAI Realtime / Grok Voice / Gemini native-audio のサーバー側 STS セッション | リアルタイム音声会話 |
| `models.py` | SQLAlchemy モデル（`User`, `Thread`, `Message`, `GeminiBatchJob`、Gemini／OpenAI Batch状態、`Gem`、セッション、BAN 等）。`mcp_service` のモデルもここで `db` に載せる | カラム追加、ユーザー設定、チャット保存形式、Batch状態 |
| `account_transfer.py` | アカウント輸出入の形式・ジョブ、設定・秘密情報・スレッド・ファイルの移行処理 | エクスポート、インポート、移行アーカイブ |
| `request_identity.py` | ユーザー読込、CSRFトークン、接続元情報、チャット遅延トレース | ユーザー識別、CSRF取得、レイテンシ計測 |
| `account_security.py` | クライアントトークン、関連アカウントBAN・解除、アカウント即時削除 | BAN連鎖、識別子、アカウント削除 |
| `chat_state.py` | スレッド公開ID、送信の冪等制御、ユーザーセッション作成・失効 | 二重送信防止、スレッド解決、セッション管理 |
| `app_settings_schema.py` | CSRFコンテキスト、アプリ設定、DBカラム・インデックスの互換更新 | 起動時スキーマ補完、AppSetting、DBインデックス |
| `temp_chat.py` | 添付参照の正規化・削除、一時チャットの在席・期限切れ監視 | 一時チャット、アップロード追跡、自動削除 |
| `request_security.py` | 自動システム通知、レスポンスキャッシュ・gzip、メンテナンス、Bot検知・Turnstile | リクエストフック、Bot対策、キャッシュヘッダー |
| `token_utils.py` | レート制限、モデル別トークン計測、Thinking集計、チャットエラー整形 | トークン数、レート制限、Botスコア評価 |
| `background.py` | RQ のチャット生成本体、Gemini／OpenAI Batch送信、E2EE 移行、Coding Mode、ストリームの Redis 蓄積 | 生成ジョブ、Batch送信、ストリーム、コーディングモード |
| `routes_pages.py` | `/`, `/c/<id>`, help, changelog, `/api/version`, `sw.js` などページと入口 | 画面ルート、版 API、PWA |
| `routes_auth.py` | ログイン、Google / Minashin / Passkey、2FA、signup、setup、logout | 認証、SSO、新規登録 |
| `routes_chat.py` | `/chat_stream`、Gemini／OpenAI／xAI Batch状態取得・一覧・停止・履歴削除、トークン見積、停止、一時チャット heartbeat、タイトル生成 | 通常のテキストチャット送信、Batch状態更新、Batch履歴管理 |
| `routes_realtime.py` | Lyria / Realtime の HTTP+SSE、Gemini STS 保存、`robots.txt` | リアルタイム API の HTTP 面 |
| `routes_files.py` | ファイル・サムネイル配信、Batch状態を含むスレッド一覧・作成・取得・削除 | ファイル URL、スレッド基本操作 |
| `rich_paste_pdf.py` | スレッドPDFとリッチペーストPDFの組版・サニタイズ・テーマ処理 | PDF出力、リッチペースト印刷 |
| `routes_threads_library.py` | 暗号化スキャン、管理者スレッド、スレッド設定、ファイルライブラリ操作 | 暗号化、スレッド設定、ライブラリCRUD |
| `routes_account.py` | アカウント削除、輸出入ジョブ、重複修復、フィードバック、簡易ログイン | アカウント移行 API |
| `routes_admin.py` | BAN、ボット検知、Turnstile、速度テスト、管理者のユーザー操作 | 管理、BAN、Turnstile |
| `routes_settings.py` | `/api/settings`、AI 設定プロンプト、セッション、2FA 設定、Gem、メンテナンス | 設定保存、Gem、セッション |
| `routes_media.py` | TTS / STT / STS、アップロード、容量 API、レイテンシ、クライアントログ | 音声合成、アップロード API |
| `../mcp_service/` | 外部MCPの接続・OAuth・ツール実行。チャット側は `background.py` と設定ルートで連携 | MCP接続・認証・実行 |
