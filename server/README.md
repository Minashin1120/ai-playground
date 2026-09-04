# server — Flask 本体の分割ソース

`app.py` は起動・設定・版番号の入口です。機能の実装は、このディレクトリのファイルに分けてあります。

**重要:** これらの `.py` は通常の `import` では使いません。`app.py` が起動時に `exec()` で読み込み、名前はこれまでどおり `app` モジュールに載ります。そのため `from app import User` や gunicorn の `app:app`、既存テストはそのまま動きます。ここを `from server.models import User` の形に書き換えないでください。

## 探し方

1. 下の表で対象ファイルを決める。
2. そのファイルだけを `grep` し、ヒットした前後を読む。
3. `app.py` 全体や、関係ない `server/*.py` を通読しない。
4. 関数名が分からないときは、まずこの README と `grep`（関数名・ルート URL・モデル名）で絞り込む。

版番号（`SYSTEM_VERSION` / `APP_VERSION`）の変更は **`app.py` だけ** で行います。公開スクリプトがこの2つを `app.py` から読みます。

## 読み込み順

`app.py` は次の順で `exec()` します。後ろのファイルは前のファイルで定義した名前（`app`、`db`、`User` など）を使います。順番を変えると起動に失敗します。

## ファイル一覧

| ファイル | 主な内容 | いつ開くか |
|---|---|---|
| `request_hooks.py` | `@app.before_request`。古いログイン flash の除去、設定保存 flash の掃除、ユーザー別アップロード上限 | リクエスト前処理、flash 漏れ、アップロードサイズ |
| `storage.py` | 容量制限、アップロードパス、添付の正規化、PDF/DOCX テキスト抽出、チャンクアップロード、サムネイルとメディアのメモリキャッシュ | ファイル保存、容量、添付、チャンク |
| `crypto.py` | 暗号化鍵リング、`encrypt_val` / `decrypt_val`、バイト暗号化、TTS 音声選択、`secure_delete` | 暗号化、鍵、削除 |
| `providers.py` | モデル種別判定、Mistral OCR、Gemini 文字起こし、PCM/WAV 変換、生成バイトの保存 | プロバイダ分岐、OCR、STT、音声形式 |
| `create_file.py` | チャットの `create_file` ツール（txt/md/pdf/docx/xlsx をライブラリへ保存） | ファイル作成ツール |
| `edit_file.py` | チャットの `edit_file` ツール（xlsx/docx/pdf の編集） | ファイル編集ツール |
| `agentic_media.py` | エージェント画像の SVG サニタイズ、sandbox 画像 URL の書き換え、生成音声の保存 | 画像エージェント、SVG、sandbox 画像 |
| `settings_ai.py` | 設定モーダルの AI アシスタント、文字起こし設定、Vision 解析、Realtime 音声の補助 | 設定の AI 更新、文字起こしプロンプト |
| `lyria.py` | Lyria RealTime のサーバー側セッション（Google へ WebSocket、ブラウザへ SSE） | リアルタイム音楽 |
| `realtime.py` | OpenAI Realtime / Grok Voice / Gemini native-audio のサーバー側 STS セッション | リアルタイム音声会話 |
| `models.py` | SQLAlchemy モデル（`User`, `Thread`, `Message`, `Gem`, セッション、BAN 等）。`mcp_service` のモデルもここで `db` に載せる | カラム追加、ユーザー設定、チャット保存形式 |
| `account_transfer.py` | アカウント輸出入の形式・ジョブ、設定・秘密情報・スレッド・ファイルの移行処理 | エクスポート、インポート、移行アーカイブ |
| `request_identity.py` | ユーザー読込、CSRFトークン、接続元情報、チャット遅延トレース | ユーザー識別、CSRF取得、レイテンシ計測 |
| `account_security.py` | クライアントトークン、関連アカウントBAN・解除、アカウント即時削除 | BAN連鎖、識別子、アカウント削除 |
| `chat_state.py` | スレッド公開ID、送信の冪等制御、ユーザーセッション作成・失効 | 二重送信防止、スレッド解決、セッション管理 |
| `app_settings_schema.py` | CSRFコンテキスト、アプリ設定、DBカラム・インデックスの互換更新 | 起動時スキーマ補完、AppSetting、DBインデックス |
| `temp_chat.py` | 添付参照の正規化・削除、一時チャットの在席・期限切れ監視 | 一時チャット、アップロード追跡、自動削除 |
| `request_security.py` | 自動システム通知、レスポンスキャッシュ・gzip、メンテナンス、Bot検知・Turnstile | リクエストフック、Bot対策、キャッシュヘッダー |
| `token_utils.py` | レート制限、モデル別トークン計測、Thinking集計、チャットエラー整形 | トークン数、レート制限、Botスコア評価 |
| `background.py` | RQ のチャット生成本体、E2EE 移行、Coding Mode、ストリームの Redis 蓄積 | 生成ジョブ、ストリーム、コーディングモード |
| `routes_pages.py` | `/`, `/c/<id>`, help, changelog, `/api/version`, `sw.js` などページと入口 | 画面ルート、版 API、PWA |
| `routes_auth.py` | ログイン、Google / Minashin / Passkey、2FA、signup、setup、logout | 認証、SSO、新規登録 |
| `routes_chat.py` | `/chat_stream`、トークン見積、停止、一時チャット heartbeat、タイトル生成 | 通常のテキストチャット送信 |
| `routes_realtime.py` | Lyria / Realtime の HTTP+SSE、Gemini STS 保存、`robots.txt` | リアルタイム API の HTTP 面 |
| `routes_files.py` | ファイル・サムネイル配信、スレッド一覧・作成・取得・削除 | ファイル URL、スレッド基本操作 |
| `rich_paste_pdf.py` | スレッドPDFとリッチペーストPDFの組版・サニタイズ・テーマ処理 | PDF出力、リッチペースト印刷 |
| `routes_threads_library.py` | 暗号化スキャン、管理者スレッド、スレッド設定、ファイルライブラリ操作 | 暗号化、スレッド設定、ライブラリCRUD |
| `routes_account.py` | アカウント削除、輸出入ジョブ、重複修復、フィードバック、簡易ログイン | アカウント移行 API |
| `routes_admin.py` | BAN、ボット検知、Turnstile、速度テスト、管理者のユーザー操作 | 管理、BAN、Turnstile |
| `routes_settings.py` | `/api/settings`、AI 設定プロンプト、セッション、2FA 設定、Gem、メンテナンス | 設定保存、Gem、セッション |
| `routes_media.py` | TTS / STT / STS、アップロード、容量 API、レイテンシ、クライアントログ | 音声合成、アップロード API |

外部 MCP の接続・OAuth・ツール実行は `mcp_service/` です。チャット画面から MCP を呼ぶ経路だけが `background.py` や設定ルートと繋がります。

## 編集時の注意

- 新しいトップレベル関数や `@app.route` を足すときは、内容に合う既存ファイルへ入れてください。入口の処理（Flask 生成、`SYSTEM_VERSION`、`db`、`login_manager`）だけが `app.py` に残っています。
- ファイルを増やす・順を変えるときは、`app.py` の `_SERVER_PARTS` も同じ順で更新します。
- 直接 `import server.models` しないでください。テストとワーカーは `import app` です。
