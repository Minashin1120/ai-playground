# server — Flask 本体の分割ソース

| ファイル | 主な内容 | いつ開くか |
|---|---|---|
| `../app.py` | 起動・版番号と `_SERVER_PARTS`。部品は定義順に `exec()` され `app` の名前空間を共有するため、通常の `import server.*` にはしない | 版番号と部品の読み込み順だけを確認するとき |
| `mobile_auth.py` | Android専用Bearer認証、API許可範囲、HTTPS・Cookie分離、レスポンス保護。全フックより先に読み込む | ネイティブ認証、端末トークン、API権限 |
| `routes_mobile.py` | Android接続仕様API、アカウント情報・失効、廃止済み端末連携の終了応答 | Android接続情報・旧連携リンクの扱い |
| `routes_mobile_account.py` | Androidの設定用API：再認証、ユーザー名・パスワード変更、アカウント削除、簡易ログイン、ログインセッション、2FAの全体無効化、E2EE切替 | Android設定のアカウント・セキュリティ操作 |
| `routes_mobile_sync.py` | Androidのサーバー不使用モード：端末チャットの同期（変更一覧・アップロード・削除）、本人確認後のAPIキー取り出し、管理者の診断ログ受信（`diagnostics/android-diagnostics.log`） | 同期API、端末へのキー取り込み、Android診断ログ |
| `routes_mobile_integrity.py` | AndroidのPlay Integrity検証、Turnstileフォールバック、`/android/integrity/turnstile*` とチャット用Turnstileチケット（`/api/mobile/v1/security/turnstile*`） | 不正アプリ対策、Turnstile画面、チャット送信前の確認 |
| `routes_mobile_auth.py` | Androidのネイティブ新規登録・パスワード／Google／TOTPログイン・トークン交換・初回セットアップとZIP取り込み・旧ブラウザー連携 | アプリ内認証、初回設定 |
| `routes_mobile_passkey.py` | Androidのパスキー（WebAuthn）ログイン・2FA・パスキー／TOTP／2FAのセキュリティ管理（`/api/mobile/v1/security*`） | アプリ内パスキー、2FA設定 |
| `request_hooks.py` | `@app.before_request`。古いログイン flash の除去、設定保存 flash の掃除、ユーザー別アップロード上限 | リクエスト前処理、flash 漏れ、アップロードサイズ |
| `diagnostic_logging.py` | アカウントごとのオプトイン調査ログ、Redisの設定時間TTL、リクエスト・ワーカー・処理段階の関連付け | サーバー調査ログの同意・収集・保持・添付を変更するとき |
| `logging_policy.py` / `logging_schema.json` | Web・Android・サーバーで共有する許可項目と機密除外、既存運用ログの安全な出力 | 記録項目・型・固定値、例外やSDKのログ出力を変更するとき |
| `storage.py` | 容量制限、アップロードパス、添付の正規化、PDF/DOCX テキスト抽出、チャンクアップロード、サムネイルとメディアのメモリキャッシュ | ファイル保存、容量、添付、チャンク |
| `image_tools.py` | 画像ヘッダー解析・構造検証、cwebp／dwebp／ffmpeg／CairoSVG を子プロセスで使うサムネイル・PNG変換・SVG描画、PNG書き出しとTOTP用QR。Pillowは読み込まない | サムネイル、画像形式の変換、画像検証、QRコード |
| `IMAGE_PROCESSING.md` | 画像処理の方針（常駐プロセスでPillowを読み込まない）、処理と形式ごとの経路、安全上の制約、既知の制限、変更時のルール | `image_tools.py` や画像を扱う呼び出し元を変更する前 |
| `crypto.py` | 暗号化鍵リング、`encrypt_val` / `decrypt_val`、バイト暗号化、TTS 音声選択、`secure_delete` | 暗号化、鍵、削除 |
| `providers.py` | モデル種別判定、Mistral OCR、Gemini 文字起こし、PCM/WAV 変換、生成バイトの保存 | プロバイダ分岐、OCR、STT、音声形式 |
| `ideogram.py` | Ideogram 画像生成・Precise Edit・背景削除のリクエスト組み立て（モデル別のエンドポイント・サイズ・品質）、HTTP呼び出し、エラー文言 | Ideogramモデルの追加・パラメータ変更、生成エラー |
| `create_file.py` | チャットの `create_file` ツール（txt/md/pdf/docx/xlsx をライブラリへ保存） | ファイル作成ツール |
| `edit_file.py` | チャットの `edit_file` ツール（xlsx/docx/pdf の編集） | ファイル編集ツール |
| `agentic_media.py` | エージェント画像の SVG サニタイズ、sandbox 画像 URL の書き換え、生成音声の保存 | 画像エージェント、SVG、sandbox 画像 |
| `settings_ai.py` | 設定モーダルの AI アシスタント、文字起こし設定、Vision 解析、Realtime 音声の補助 | 設定の AI 更新、文字起こしプロンプト |
| `lyria.py` | Lyria RealTime のサーバー側セッション（Google へ WebSocket、ブラウザへ SSE）、ワーカー間の Redis 中継、保存 | リアルタイム音楽 |
| `realtime.py` | サーバー側 STS セッションの状態（`RtSession`）・キュー・ワーカー間の Redis 中継・ワーカーの振り分け・会話の保存。プロバイダ別のループは下の3部品 | リアルタイム音声会話の共通部分、保存、ブリッジ |
| `realtime_openai.py` | OpenAI Realtime（会話・翻訳）／GPT-Live／xAI Voice のセッション設定と送受信ループ | OpenAI・xAI の Realtime プロトコル |
| `realtime_gemini.py` | Gemini Live のセッション設定・送受信ループ・ターン終了処理 | Gemini Live のプロトコル |
| `realtime_xai_stt.py` | xAI のストリーミング文字起こし（`wss://api.x.ai/v1/stt`）の送受信ループとチャンク結合 | Grok Voice 文字起こし |
| `models.py` | SQLAlchemy モデル（`User`, `Thread`, `Message`, `GeminiBatchJob`、各プロバイダのBatch状態、`Skill`、セッション、BAN 等）。旧 `gem` テーブル／カラムの移行は `app_settings_schema.py` の `ensure_skill_schema()`。`mcp_service` のモデルもここで `db` に載せる | カラム追加、ユーザー設定、チャット保存形式、Batch状態 |
| `account_transfer.py` | アカウント輸出入の形式・ジョブ、設定・秘密情報・スレッド・ファイルの移行処理 | エクスポート、インポート、移行アーカイブ |
| `request_identity.py` | ユーザー読込、CSRFトークン、接続元情報、チャット遅延トレース | ユーザー識別、CSRF取得、レイテンシ計測 |
| `account_security.py` | クライアントトークン、関連アカウントBAN・解除、アカウント即時削除 | BAN連鎖、識別子、アカウント削除 |
| `chat_state.py` | スレッド公開ID、送信の冪等制御、ユーザーセッション作成・失効 | 二重送信防止、スレッド解決、セッション管理 |
| `app_settings_schema.py` | CSRFコンテキスト、アプリ設定、DBカラム・インデックスの互換更新 | 起動時スキーマ補完、AppSetting、DBインデックス |
| `temp_chat.py` | 添付参照の正規化・削除、一時チャットの在席・期限切れ監視 | 一時チャット、アップロード追跡、自動削除 |
| `request_security_notices.py` | 自動注入システムプロンプト（ユーザー単位）の案内文の既定値・一覧・ユーザー設定の読み書き・プレビュー、現在時刻の案内、送信時の補助情報（添付名・引用元・保存済みプロンプト）の再利用 | 案内文の追加・変更、既定値、設定画面のプレビュー |
| `request_security.py` | リクエストフック（クライアントトークン・レスポンスキャッシュ・gzip・メンテナンス・BAN・セッション）と、Bot検知・Turnstile・連投ロック | リクエストフック、Bot対策、キャッシュヘッダー |
| `skills.py` | Skillカタログ（名前・短い説明・指示全文の同梱）、`use_skill` の各プロバイダ向けツール定義と実行、経路ごとのツール可否 | Skillのプロンプト注入・ツール呼び出しを変更するとき |
| `token_estimation.py` | 入力欄の本文・引用・添付を公式APIで計測、短時間キャッシュ、概算への切替、詳細の増分内訳 | 入力見積の精度・計測APIを変更するとき |
| `token_utils.py` | レート制限、アカウント単位の認証失敗制限（Web・Android共通）とTOTPコードの再利用防止、モデル別トークン計測、Thinking集計、チャットエラー整形 | トークン数、レート制限、ログイン・2FAの総当たり対策、Botスコア評価 |
| `model_usage.py` | モデルAPI呼び出しごとの数値使用量、累積usageの統合、Thinking・同期メタデータの正規化 | 費用予想の呼び出し記録とトークン集計を変更するとき |
| `model_usage_pricing.py` | キャッシュの読込・書込、検索・ホスト実行ツール、xAI請求額の数値使用量 | 追加料金の観測と課金内訳を変更するとき |
| `model_usage_http.py` | HTTPXのリクエスト・応答をストリームのまま観測し、SDK自動ツール呼び出し・再試行も記録 | HTTP通信と使用量の対応付けを変更するとき |
| `background.py` | RQ のチャット生成本体、Gemini／OpenAI／xAI／Z.AI Batch送信、ストリームの Redis 蓄積 | 生成ジョブ、Batch送信、ストリーム |
| `background_migration.py` | E2EE 有効化・無効化に伴うメッセージ・添付ファイルの一括変換 | E2EE 移行ジョブ |
| `background_python.py` | Python 実行ツール（bubblewrap サンドボックス、出力の秘匿化、実行画像の受け渡し） | Python 実行の制限・出力処理 |
| `background_deepseek.py` | DeepSeek のストリーム tool call 統合、検索リクエストと結果整形 | DeepSeek の検索・ツール呼び出し |
| `background_stream_claude.py` | `background_chat_task` の Claude 分岐（メッセージ変換・PDF/画像入力・prompt caching・thinking・MCP/スキル tool round）。`locals()` を受け取り、更新後の回答本文と思考を返す | Claude の生成分岐 |
| `background_stream_deepseek.py` | `background_chat_task` の DeepSeek 分岐（ストリーム tool round・画像/vision・検索引用）。呼び出し元の `locals()` を受け取り、共有の後処理に必要な値を返す | DeepSeek の生成分岐 |
| `background_stream_gemini.py` | `background_chat_task` の Gemini 分岐の振り分け（文字起こし・TTS・画像生成・テキスト生成）と各専用部品の呼び出し。`locals()` を受け取り、共有の後処理に必要な値を返す | Gemini の分岐選択 |
| `background_stream_gemini_chat.py` | `background_chat_task` の Gemini テキスト生成・function calling 分岐（ツール構成・リクエスト組み立て・ストリーム受信）。`locals()` を受け取り、更新後の回答本文と思考を返す | Gemini のテキスト生成 |
| `background_stream_gemini_citations.py` | Gemini の検索・URL Context 引用の差し込み位置（UTF-8 バイトの出典位置から本文位置を推定）、引用リンク生成、URL 取得状態の表示名、思考テキストの取り出し。純粋関数のみ | 引用・出典表示の整形 |
| `background_stream_gemini_embedding.py` | `background_chat_task` の Gemini Embedding 分岐（埋め込みベクトルの生成と先頭次元の表示）。`locals()` を受け取り、更新後の回答本文を返す | Gemini Embedding |
| `background_stream_gemini_files.py` | Gemini Files API のファイル部品・アップロード（再試行）・ファイルキャッシュの照合と更新。`user_id` / Vertex AI 判定 / genai クライアントは引数で受け取る | Gemini の添付アップロード・キャッシュ |
| `background_stream_gemini_image.py` | `background_chat_task` の Gemini 画像生成分岐（画像モデルの呼び出し・Batch・生成画像の保存）。`locals()` を受け取り、更新後の回答本文を返す | Gemini の画像生成 |
| `background_stream_gemini_music.py` | `background_chat_task` の Gemini 音楽生成分岐（Lyria 3 の歌詞・音声の保存、Lyria RealTime の案内）。`locals()` を受け取り、更新後の回答本文を返す | Gemini の音楽生成 |
| `background_stream_gemini_omni.py` | `background_chat_task` の Gemini Omni 1.1 Flash 分岐（Interactions API による動画生成・URI 取得・保存）。`locals()` を受け取り、更新後の回答本文を返す | Gemini Omni の動画生成 |
| `background_stream_gemini_tail.py` | Gemini テキスト生成の後処理（sandbox 画像の置換、検索・URL Context の引用、Python 実行結果の追記）。生成側から値を受け取り、更新後の回答本文を返す | Gemini の回答後処理 |
| `background_stream_gemini_transcribe.py` | `background_chat_task` の Gemini 3.5 Transcribe 分岐（音声ファイルのアップロードと文字起こし）。`locals()` を受け取り、更新後の回答本文を返す | Gemini の文字起こし |
| `background_stream_gemini_tts.py` | `background_chat_task` の Gemini TTS 分岐（音声合成・WAV 保存）。`locals()` を受け取り、更新後の回答本文を返す | Gemini の音声合成 |
| `background_stream_gemini_video.py` | `background_chat_task` の Gemini 動画生成分岐（Veo 3.1 / Omni Flash の投入・ポーリング・保存）。`locals()` を受け取り、更新後の回答本文を返す | Gemini の動画生成 |
| `background_stream_gpt_image.py` | `background_chat_task` の GPT Image 分岐（Images API／Responses API の画像生成・編集・マスク処理・GIF 出力・保存）。`locals()` を受け取り、更新後の回答本文を返す | GPT Image の画像生成 |
| `background_stream_grok_imagine.py` | `background_chat_task` の Grok Imagine 画像分岐（生成・最大3枚の参照画像による編集・保存）。`locals()` を受け取り、更新後の回答本文を返す | Grok Imagine の画像生成 |
| `background_stream_grok_imagine_video.py` | `background_chat_task` の Grok Imagine 動画分岐（生成リクエスト・進捗ポーリング・保存）。`locals()` を受け取り、更新後の回答本文を返す | Grok Imagine の動画生成 |
| `background_stream_grok_native.py` | `background_chat_task` の Grok Native SDK 分岐（検索ツール・reasoning・サンプリング設定・履歴画像・引用の組み立て）。`locals()` を受け取り、更新後の回答本文と思考を返す | Grok の生成分岐 |
| `background_stream_ideogram.py` | `background_chat_task` の Ideogram 分岐（画像生成・Precise Edit・背景除去・オブジェクト除去、入力画像とマスクの準備、生成画像の保存）。`locals()` を受け取り、更新後の回答本文を返す | Ideogram の画像生成 |
| `background_stream_kimi_zai.py` | `background_chat_task` の Kimi／Z.AI 分岐（vision 解析・Z.AI 画像入力・tool round・Z.AI Batch送信）。`locals()` を受け取り、共有の後処理に必要な値を返す | Kimi／Z.AI の生成分岐 |
| `background_stream_mistral_ocr.py` | `background_chat_task` の Mistral OCR 分岐（PDF・画像・DOCX・PPTX のアップロード／OCR、Markdown 組み立て、生成画像の保存）。`locals()` を受け取り、更新後の回答本文を返す | Mistral OCR |
| `background_stream_openai.py` | `background_chat_task` の OpenAI Responses API／Grok フォールバック分岐（入力構築・ツール構成・MCP・Batch送信・ストリーム消費）。`locals()` を受け取り、共有の後処理に必要な値を返す | OpenAI／xAI の生成分岐 |
| `background_stream_openai_images.py` | Responses 分岐のコードインタプリタ出力（コンテナ画像・ファイルの取得と保存、生成画像の追記、`pyexec` の記録）。`OpenAICodeInterpreterOutputs` に `append_content` / `append_text` / `get_full_res` を渡す | コードインタプリタの画像・出力 |
| `background_stream_openai_search.py` | `background_chat_task` の OpenAI Search API 分岐（Chat Completions の web search・履歴画像・ファイル添付・引用の組み立て）。`locals()` を受け取り、更新後の回答本文を返す | OpenAI Search の生成分岐 |
| `background_stream_tail.py` | `background_chat_task` の全プロバイダ共通の後処理（回答本文の整形・Coding Mode の差分適用・使用量と思考の保存・アシスタントメッセージの永続化） | 生成結果の保存・完了通知 |
| `background_stream_tts.py` | `background_chat_task` の TTS 分岐（Google Cloud TTS・xAI TTS・OpenAI TTS、音声の保存と `<audio>` の追記）。`locals()` を受け取り、更新後の回答本文を返す | 音声合成 |
| `background_coding.py` | Coding Mode の差分生成・編集適用・修復プロンプト | コーディングモード |
| `background_math_steps.py` | 数学ステップ解説の固定出力形式（`## ステップ N` と `mathsvg`）、設定の正規化、ステップ抽出、「もっとやさしく」のプロンプト | 数学ステップのプロンプト・形式・設定 |
| `background_attachments.py` | `background_chat_task` の添付読込（パス・サイズ検証、PDF/DOCX/PPTX/XLSX/テキストの抽出、ファイルキャッシュ更新）。`img_list` / `options` / `user_id` / モデル種別を受け取り、`loaded_files` と `file_errors` を返す | 添付ファイルの読み込み |
| `background_history.py` | `background_chat_task` の履歴復元（祖先のたどり・トークン数の保存）と、過去メッセージの画像・文書テキストの読み込み。履歴リストは引数で受け取る | 会話履歴・過去添付の復元 |
| `background_mcp.py` | `background_chat_task` の MCP 実行環境の遅延構築（`tools/list`・案内文のシステムプロンプト注入）と実行結果の Markdown | MCP の準備・結果表示 |
| `routes_pages.py` | `/`, `/c/<id>`, help, changelog, `/api/version`, `sw.js` などページと入口 | 画面ルート、版 API、PWA |
| `routes_auth.py` | ログイン、Google / Minashin / Passkey、2FA、signup、setup、logout | 認証、SSO、新規登録 |
| `routes_batch.py` | Gemini／OpenAI／xAI／Z.AI Batchの状態取得・一覧・停止・履歴削除、Batchモデルの判定と状態ラベル | Batch状態更新、Batch履歴管理 |
| `routes_browser_fast.py` | Browser fast mode（ブラウザから Gemini へ直接送信）の bootstrap・保存、プロンプト設定の記録 | Browser fast mode、`Message.prompt_options` の記録 |
| `routes_stream.py` | `/chat_stream`（NDJSON ストリーム、SSE 購読・ハートビート・終了ジョブの再生、引用元の解決） | 通常のテキストチャット送信 |
| `routes_math_steps.py` | 数学ステップの進捗保存（`/api/math_steps/progress`）と「もっとやさしく」の再生成（`/api/math_steps/simplify`）。所有者確認・上限・使用量の記録 | 数学ステップの進捗・簡略化API |
| `routes_chat.py` | トークン見積、再開ストリームの受信ループ、停止、一時チャット heartbeat、タイトル生成、PromptCache のキー | 送信の補助 API、タイトル生成 |
| `routes_realtime.py` | Lyria / Realtime の HTTP+SSE、Gemini STS 保存、`robots.txt` | リアルタイム API の HTTP 面 |
| `thread_branches.py` | 指定時点までの全分岐のコピー、分岐元情報、共有ファイルの参照確認・削除保護 | 別チャットへの分岐、共有ファイルの寿命 |
| `routes_files.py` | ファイル・サムネイル配信、Batch状態を含むスレッド一覧・作成・取得・削除 | ファイル URL、スレッド基本操作 |
| `rich_paste_pdf.py` | スレッドPDFとリッチペーストPDFの組版・サニタイズ・テーマ処理 | PDF出力、リッチペースト印刷 |
| `routes_threads_library.py` | 暗号化スキャン、管理者スレッド、スレッド設定、ファイルライブラリ操作 | 暗号化、スレッド設定、ライブラリCRUD |
| `routes_account.py` | アカウント削除、移行（輸出入）ジョブの状態・中止・完了版の取得・ダウンロード | アカウント削除・エクスポート API |
| `routes_account_import.py` | アカウント取り込みのアップロード（開始・チャンク・完了・中止）と古い一時ファイルの掃除、取り込み本体（`/api/account/import`） | アカウント取り込み API |
| `routes_account_dedupe.py` | 取り込み済みレコードの重複判定（スレッド・Skill・ファイル・フィードバック・遅延計測・トレースの署名）、参照ファイルの走査、重複のプレビューと実行 | 重複修復 API |
| `routes_admin_feedback.py` | フィードバックの送信・一覧・返信・削除（送信者本人と管理者。ディレクトリごと削除）、返信の履歴（`FeedbackReply`。送信者本人と管理者が返信し、チェックしたときだけ操作ログ `logs/reply-<返信ID>.jsonl` を一緒に保存）と、フィードバックごとのディレクトリ（`feedback/<ID>/`）への情報（`feedback.json`）と操作ログ（`logs/activity.jsonl`）の保存、フィードバック確認用の一時ログイン | フィードバック、操作ログ（ログの収集を強化） |
| `routes_admin_bot.py` | BAN 申し立ての一覧・既読・更新、ボット検知（テレメトリ、Turnstile 検証、ロック／解除）、主管理者だけの速度テスト | 管理、BAN、Turnstile |
| `routes_admin_users.py` | 管理者のユーザー検索、ボットの証跡（一覧・アカウント・削除）、アカウントのロック・識別子の解除、管理者によるユーザー更新、テーマ色と WebAuthn の補助 | 管理、ユーザー操作 |
| `feedback_chat_copy.py` | フィードバックに添付する「開いているチャットのコピー」：チャットIDから辿れる記録（復号したメッセージ、Batch・遅延計測・同期・添付キャッシュ・Skill・Redis・一時チャットの状態、ログの該当行）と復号した添付ファイルの保存（`feedback/<ID>/chat.jsonl` と `chat.files/`）、Android端末だけの添付の受信（`/api/feedback/<id>/chat_files`）、保存数・容量の上限、アカウント削除時のフィードバックのディレクトリの削除 | チャットのコピーの内容・保存先・上限 |
| `feedback_images.py` | フィードバックと返信に添付する画像：送信後の受信（`/api/feedback/<id>/images`、返信は `/api/feedback/<id>/replies/<返信ID>/images`）、ファイル先頭による形式判定、枚数・容量の上限、`feedback/<ID>/images/` への保存 | フィードバックの画像添付の上限・保存先 |
| `routes_settings.py` | `/api/settings` の保存・取得、クライアントデバッグログの受信 | 設定保存 |
| `routes_settings_ai.py` | AI 設定プロンプト（`/api/settings/apply-ai-prompt`）、設定アシスタントのツール定義とLLM呼び出し | 自然言語での設定変更 |
| `routes_settings_security.py` | セッション一覧・失効（`/api/sessions*`）、2FA（TOTP）設定、WebAuthn（パスキー）の登録・削除 | セッション・2FA・パスキー |
| `routes_settings_skills.py` | Skill（`/api/skills` と互換 `/api/gems`）の一覧・作成・更新・削除、メンテナンス切替 | Skill、メンテナンス |
| `routes_media.py` | TTS / STT / STS（音声合成・文字起こし・音声間変換） | 音声合成・文字起こし |
| `routes_media_upload.py` | `/upload`（通常・分割の開始／チャンク／完了）とアップロードサイズ超過（413）の処理 | 添付アップロード API |
| `routes_media_storage.py` | 容量 API（`/api/storage`）、最初のトークンまでの遅延計測（`/api/metrics/first_token`）、クライアントログ（`/api/client_log`） | 容量表示・遅延計測・端末ログ受信 |
| `../mcp_service/` | 外部MCPの接続・OAuth・ツール実行。チャット側は `background.py` と設定ルートで連携 | MCP接続・認証・実行 |
