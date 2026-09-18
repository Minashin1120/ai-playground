# Androidクライアント開発・接続ガイド

この文書は、AI Chat Playgroundの公式Androidクライアントの通信仕様、実装構成、APKの構築・配布、サーバー設定をまとめたものです。Androidプロジェクトは同じリポジトリの `android/` にあり、モジュールは `android/app/` です。GitHub Actionsで署名付きdebug/release APKを構築し、GitHub Releaseへ配布します。Google Playへの登録とAAB配布は対象外です。

接続先は `https://ai.minashin1120.com`。認証プロトコルは `device_pairing_v1`、接続仕様の版は `api_version: 1` です。Webのリリース番号とAndroidのversionCodeは別々に管理します。

## 目次

1. [構成・対応範囲](#1-構成対応範囲)
2. [認証と端末連携](#2-認証と端末連携)
3. [認証APIの詳細](#3-認証apiの詳細)
4. [チャット・ファイルAPI](#4-チャットファイルapi)
5. [ストリーミング・切断復帰](#5-ストリーミング切断復帰)
6. [Androidプロジェクトの作成](#6-androidプロジェクトの作成)
7. [Kotlin通信コード](#7-kotlin通信コード)
8. [端末保存・画面・ライフサイクル](#8-端末保存画面ライフサイクル)
9. [サーバー設定と公開](#9-サーバー設定と公開)
10. [試験・受け入れ条件](#10-試験受け入れ条件)
11. [障害対応・制限・今後の拡張](#11-障害対応制限今後の拡張)
12. [実装ファイル・公式資料](#12-実装ファイル公式資料)

## 1. 構成・対応範囲

```text
Androidのネイティブ画面 ── HTTPS + Bearer ── Apache ── Gunicorn/Flask
                                                       ├─ MariaDB（ユーザー・履歴・端末セッション）
Androidから開くブラウザー ── HTTPS + Cookie ──────────────┤
                                                       └─ Redis ── RQワーカー ── AI事業者
```

AndroidからMariaDB、Redis、RQ、AI事業者の秘密鍵へ直接アクセスしません。Webと同じアカウント、履歴、保存済みのモデルAPIキーを利用します。APIキーの初期登録・変更はWebの設定画面で行います。

| 機能 | このAPIでの扱い |
|---|---|
| ユーザー認証 | 既存ブラウザーログインと端末連携。パスワード、Google、Minashin SSO、Passkey、2FAの既存経路を利用 |
| 通常のチャット | スレッド一覧・検索・作成・取得・削除、ブックマーク、タイトル・スレッド指示、送信、ストリーム再接続、停止、メッセージの編集・再生成・分岐切替、ネイティブPDF出力 |
| 添付 | 通常アップロード、大容量チャンクアップロード、進捗表示・キャンセル、Photo Picker・カメラ、MIME別プレビュー、画像圧縮設定、本人のファイル・サムネイル取得、容量表示、アプリ内プレビュー（画像・テキスト・PDF・音声・動画。未対応形式は外部アプリ） |
| 一時チャット | 作成・heartbeat。自動削除タイマーをクライアント側でも考慮する |
| ファイルライブラリ | 一覧・検索・お気に入り・名前変更・削除・チャットへの再利用、アプリ内プレビュー。所有者ディレクトリのみ対象 |
| Gems | 一覧・作成・編集・削除・チャットへの適用・`@`候補、既定モデル適用、固定プロンプトの編集・送信 |
| 一般設定・セッション | 既定モデル／Vision、Thinking／Web検索／URL／Maps／Python／File／SysPrompt／MCP、Thinking level／budget、Reasoning effort、Safety、テーマ色、Enter送信、ライト／Liquid Glass、プロンプトバー、STT既定、システムプロンプト、前回設定継続、リンク自動検索、一時チャット期限、フィードバック、MCP有効切替、この端末のセッション表示と失効。APIキー・パスワード・2FA登録はWeb導線 |
| モデル一覧 | `/me` の `models`。提供元、用途、対応能力、廃止状態、選択可否を返す。Android内のWeb由来カタログで正式表示名・説明・価格・タグ・追加順を補完。互換用の `model_ids` も維持 |
| APIキー・プロフィール・アカウント削除・2FA設定 | Webで操作。Androidトークンでは設定・管理APIにアクセスできない |
| WebのE2EE設定 | 有効なまま連携・履歴取得・添付を利用可能。現在の実装はサーバー管理鍵による保存時暗号化であり、端末だけが復号できるE2EEではない |
| Batch管理 | ネイティブで送信、一覧、状態更新、停止、履歴削除、完了通知に対応 |
| 画像・動画・OCR・TTS・文字起こし | `/chat_stream` と同じ保存・停止・再接続境界でネイティブ対応。GPT／Gemini／Grok画像、GPT-Imageマスク、Gemini動画（長さ・比率・解像度）、Grok動画、OCR、TTS、Thinking量、xAI詳細を表示・送信 |
| Python・MCP・Coding結果 | Python・MCP・Coding差分のストリームイベントを構造化カードで表示。MCPの秘密設定とCoding対象の高度な編集はWeb導線 |
| リアルタイム音声・Lyria | OpenAI／Grok RealtimeとLyriaはBearer認証のネイティブセッション、マイク入力、SSE再生、保存に対応。Gemini LiveとSTSは認証済みWebへの明示的な導線 |
| アプリ更新検出 | 起動時にGitHub Releasesの `android-vX.Y.Z` だけを確認し、現在のAndroid版より新しい安定版があればReleaseページを案内。APKの取得・インストールはユーザー操作 |
| ブラウザー高速モード | 対象外。APIキーをAndroidへ返すbootstrap APIは許可しない |
| アプリ配布・真正性検証 | APK署名・Play配布は別途。client_idや端末名はアプリ署名の証明ではない |

Android版は通常チャット、メタデータ付きモデル選択、Thinking・Web検索・Prompt Cache、履歴、スレッド設定、一時チャット、添付、ファイルライブラリ、Gems、Web相当の設定タブ、この端末のセッション、編集・再生成・分岐、ネイティブPDF、画像・動画・OCR・TTS・文字起こし、Batchの送信・管理・完了通知、生成停止、切断後の再接続、端末連携・失効、スラッシュコマンド、OpenAI／Grok／Gemini Native Audio／Gemini Live Realtime、Lyriaに対応します。本文はMarkdown、コード、表、数式近似、添付プレビューに加え、検索・Python・MCP・Coding差分を構造化カードで表示します。ファイルライブラリとチャットの添付はアプリ内で画像・テキスト・PDF・音声・動画を確認でき、未対応形式は外部アプリで開けます。APIキー・パスワード・2FA登録・MCP OAuth秘密、管理機能は認証済みWebへの明示的な導線を使います。

暗号化の境界は「HTTPSで通信」「サーバーが保存データを暗号化・復号」「端末トークンをAndroid Keystoreの鍵で暗号化保存」です。既存Webの `enable_e2ee` はサーバー側の `encrypt_val` / `decrypt_val` とファイル暗号化に使われます。そのため、この設定を理由にAndroidを拒否する必要はありません。真の端末間E2EEに変更する場合は、端末鍵の生成・共有・回復とAI処理時の平文の扱いを別途設計する必要があります。

## 2. 認証と端末連携

### 2.1 連携の流れ

1. Androidが `POST /api/mobile/v1/device` を送信する。
2. サーバーがアプリ用の `device_code` と、人が入力する `user_code` を返す。有効期限は10分。
3. アプリ画面に確認コードを表示し、Custom Tabsで `/android/connect?code=<user_code>` を開く。`user_code` は12文字の確認コードだけを含み、`device_code` やアクセストークンはURLに含めない。
4. ブラウザーで未ログインなら通常のログインを完了する。SSOや2FAの終了後、連携画面へ戻る。
5. URLの確認コードが有効なら連携画面に申請内容が自動表示される。利用者はコード入力なしで端末名・権限を確認して「この端末を許可」を押す。「拒否」も可能。手動入力はリンクが使えない場合のフォールバック。
6. アプリは5秒以上の間隔で `POST /api/mobile/v1/token` を送る。ブラウザーからアプリへ戻ったときに再開してよい。
7. 承認後、一度だけ `access_token` が返る。アプリは保存後に `GET /api/mobile/v1/me` を呼んでアカウントを確認する。
8. 以後は `Authorization: Bearer <access_token>` で許可されたAPIを呼ぶ。

確認コードは人がアプリとブラウザーを照合するためのものです。`device_code` やアクセストークンをURL、ディープリンク、クリップボード、ログに載せないでください。

この方式は端末認可の考え方を参考にした独自プロトコルです。[RFC 8628](https://www.rfc-editor.org/rfc/rfc8628.html)準拠のOAuthサーバーではありません。JSON形式、固定client_id、エラーとポーリングの仕様は本書に従い、一般的なOAuth SDKへそのまま設定しないでください。

### 2.2 認証境界

| 通信 | 認証 | CSRF | Cookie |
|---|---|---|---|
| ブラウザーのログイン・連携承認 | 既存のWebセッション | POSTに既存CSRFトークンが必須 | ブラウザーが管理 |
| device/token | 短期device_codeによる連携プロトコル | CookieとOriginを拒否しJSON POSTだけを受理 | 送信禁止 |
| Androidのチャット等 | 有効なAndroid専用Bearer | 専用Bearerを検証したリクエストだけ免除 | 送信禁止 |

通常のWebリクエストのCSRF対策を解除していません。APIというURLだけでCSRF免除になることもありません。Androidは独立したCookieなしHTTPクライアントを使います。`Origin`、`Referer`、`X-CSRF-Token` を追加する必要はありません。

ネイティブHTTP通信はブラウザーのCORS制限を受けません。`Access-Control-Allow-Origin: *` やCookieの `SameSite=None` を設定する必要はありません。

### 2.3 トークンの性質と失効

- トークンは `aip_android_` で始まる不透明な文字列。JWTとして解析しない。
- 有効期限は承認から30日。アクセスのたびに延長しない。refresh_tokenはない。
- DBには `UserSession.session_id = android:<SHA-256>` を保存し、生トークンを保存しない。
- 承認から引き換えまでの短時間だけ、既存の暗号化処理で暗号化したトークンをRedisに保持する。Redisの期限は申請開始から最大10分。
- 承認と引き換えはRedisのLuaで原子的に判定し、同時リクエストで二重承認・二重引き換えを防ぐ。
- 引き換え成功応答を通信障害で受け取れなかった場合、同じtoken APIでは回収できない。新しい端末連携を開始する。
- Webの設定 → セッション管理に `Official Android: <端末名>` が表示される。個別失効・他のセッション失効・全失効がAndroidにも適用される。
- アプリからのログアウトは `POST /api/mobile/v1/revoke` 後に端末保存を消す。オフライン時のローカル削除だけではサーバー側失効は成立しない。
- ブラウザーの通常ログアウトはそのブラウザーのセッションを失効する操作。別のAndroidセッションを終了したい場合はセッション管理を使う。
- 失効は次のHTTPリクエストから適用。すでに始まった生成や開いているストリームを途中で自動停止する機能ではない。

## 3. 認証APIの詳細

### 3.1 共通仕様

HTTPS必須。JSONはUTF-8。APIレスポンスと認証ページは `Cache-Control: private, no-store`。アプリはリダイレクトの自動追跡を無効にし、HTTPステータスとContent-Typeを確認します。native APIが返すCookieは保存・再送しません。

以下の相対パスは固定の接続先に対して解決してください。サーバーからのパスを使う場合も、scheme・host・portが元の接続先と一致することを検証します。

### 3.2 接続仕様

`GET /api/mobile/v1/config` は認証不要。主なフィールドは次のとおりです。

```json
{
  "api_version": 1,
  "client_id": "official-android",
  "auth_flow": "device_pairing_v1",
  "device_endpoint": "/api/mobile/v1/device",
  "token_endpoint": "/api/mobile/v1/token",
  "verification_uri": "/android/connect",
  "token_expires_in": 2592000,
  "grant_expires_in": 600,
  "poll_interval": 5,
  "stream_format": "application/x-ndjson",
  "e2ee_supported": true,
  "encryption_mode": "server_managed_at_rest"
}
```

実際の応答には `system_version`、`me_endpoint`、`revoke_endpoint`、`allowed_endpoints` も含みます。`allowed_endpoints` の `<thread_id>`、`<path:filename>` はFlaskのパス変数表記です。

### 3.3 端末連携の開始

`POST /api/mobile/v1/device`

```json
{"client_id":"official-android","device_name":"Pixel — 個人用"}
```

`device_name` は任意、既定 `Android`、前後空白を除いて1〜80文字。制御文字は禁止。IMEI、Android ID、シリアル番号などを端末名として集める必要はありません。

成功は200。応答例のコードは説明用です。

```json
{
  "device_code":"<43文字の秘密コード>",
  "user_code":"8A63D240C17F",
  "verification_uri":"/android/connect",
  "expires_in":600,
  "interval":5
}
```

IPごとに10分で10申請まで。共有ネットワークでは複数利用者が合算されます。device/tokenのJSON本文は最大4KiBです。

### 3.4 承認の確認・トークン取得

`POST /api/mobile/v1/token`

```json
{"client_id":"official-android","device_code":"<deviceの応答をそのまま使用>"}
```

成功200：

```json
{
  "access_token":"aip_android_<不透明な文字列>",
  "token_type":"Bearer",
  "expires_in":2591988,
  "scope":"chat files",
  "api_version":1
}
```

ポーリングは同一device_codeで最短5秒、IP単位では1分120回まで。`interval` と `Retry-After` を尊重します。10分を超えたら再申請します。ネットワーク障害では待機時間を延ばし、同時に複数のポーリング処理を作らないでください。

| HTTP | error | クライアントの動作 |
|---|---|---|
| 400 | `authorization_pending` | 5秒以上待って再確認 |
| 429 | `slow_down` / `rate_limited` | Retry-After以上待ち、バックオフする |
| 400 | `access_denied` | 連携を終了し拒否を表示 |
| 400 | `expired_token` | 期限切れまたは引き換え済み。新しく連携を開始 |
| 400 | `invalid_client` / `invalid_grant` / `invalid_request` | 送信内容を修正 |
| 400 | `cookies_or_origin_not_allowed` | ネイティブ側のCookie・Originを除去 |
| 400 | `authorization_not_allowed` | device/tokenにAuthorizationを付けない |
| 400 | `https_required` | HTTPS接続とプロキシの転送情報を確認 |
| 415 | `json_post_required` | JSON POSTへ修正 |
| 413 | フレームワークのサイズエラー | 本文を4KiB以下にする。HTMLの場合もJSON解析を強行しない |
| 503 | `temporarily_unavailable` | Redis等の障害。時間を置いて再試行 |
| 503 | `mobile_api_disabled` | サーバー管理者がAndroid接続を停止中 |

### 3.5 アカウント確認・ログアウト

`GET /api/mobile/v1/me`：Bearer必須。`id`、`username`、UTC ISO 8601の `expires_at`、`e2ee_enabled`、`encryption_mode`、`default_model`、`model_catalog_version`、`models` を返します。各モデルには `id`、`name`、`provider`、`provider_label`、`mode`、`capabilities`、`deprecated`、`selectable` が含まれます。互換用の `model_ids` も維持します。秘密鍵・APIキー・パスワードは返しません。`e2ee_supported` は既存Web設定との互換性を表す値であり、端末間暗号化の保証ではありません。

`POST /api/mobile/v1/revoke`：Bearer必須、本文 `{}`。成功は `{"status":"revoked"}`。同じトークンでの次の呼び出しは401になります。

Webのログアウトと同様、端末の失効操作はBot確認待ち・ロック・BAN・メンテナンス中でも可能です。Android API自体を無効化している間はこの操作も停止するため、必要に応じてWebのセッション管理を使います。

通常APIの401はローカルトークンを破棄して再連携します。403は再ログインだけで解消するとは限りません。権限外、BAN、Turnstile、ロック等を区別します。

## 4. チャット・ファイルAPI

以下はBearer認証の許可リストです。一覧にないエンドポイントやHTTPメソッドは403 `insufficient_scope`。許可されていても、各ルートの所有者検査・容量制限・モデル認証・Bot判定を通る必要があります。

| メソッド・パス | 入力 | 主な応答 |
|---|---|---|
| GET `/api/threads` | `page=1&q=検索語`。ページは1始まり | `threads`, `has_next`, `next_page`。1ページ20件 |
| POST `/api/threads` | `{}` または `{"is_temporary":true}` | `id`, `title`, 一時チャットのメタデータ |
| GET `/api/threads/<id>` | `limit=50&before_id=<最古ID>&include_meta=1` | `messages`, `has_older_messages`, `oldest_loaded_id`, `pending_job` 等 |
| DELETE `/api/threads/<id>` | 本文不要 | `{"status":"deleted"}`。履歴・紐付く添付を削除するため画面で確認する |
| GET `/api/threads/<id>/settings` | なし | スレッド指示、共通指示の有無、一時チャットの期限情報 |
| PUT `/api/threads/<id>/settings` | `custom_instruction`, `include_global_instruction`, `is_temporary` | 更新後の一時チャット状態。指示は100,000文字以下 |
| PUT `/api/threads/<id>/title` | `title` | 正規化後のタイトル。最大200文字 |
| POST `/api/threads/<id>/bookmark` | `{}` | 切り替え後の `is_bookmarked` |
| POST `/chat_stream` | 下の送信JSON | NDJSONストリーム、またはエラーJSON |
| POST `/chat_stream_resume` | `{"thread_id":"...","job_id":"..."}` | 蓄積内容と継続ストリーム |
| POST `/api/stop_chat` | `thread_id` と、分かれば `job_id` | `status`, `job_id`, `source`。停止信号の受付であり即時停止完了ではない |
| POST `/api/realtime/start`、GET `/api/realtime/stream`、POST `/api/realtime/audio` | セッション開始のモデル／voice、SSEの `session_id`、音声PCMチャンク | OpenAI／Grok Realtimeのサーバーセッション。音声イベントはBase64 PCM、APIキーは返さない |
| POST `/api/realtime/commit`、`/api/realtime/cancel`、`/api/realtime/save` | `session_id`、保存時は任意の `thread_id` | 発話確定、破棄、履歴保存 |
| POST `/api/gemini/music/start`、GET `/api/gemini/music/stream` | `weighted_prompts`、SSEの `session_id` | LyriaセッションとBase64音声スナップショット |
| POST `/api/gemini/music/command`、`/cancel`、`/save` | `session_id`、`control`／`action`、保存時は任意の `thread_id` | 一時停止／再開、破棄、履歴保存 |
| POST `/api/token_estimate` | `model`, `message`, 必要なら `image_urls` | 既存のトークン見積応答。請求額の確定値ではない |
| POST `/api/temporary_chat/heartbeat` | `thread_id`, `active` | 一時チャットの状態・期限情報 |
| GET `/c/<thread_id>/pdf` | 任意で `leaf_id` | スレッドのPDF用JSON。AndroidはネイティブA4へ描画して共有する |
| POST `/upload` | multipart/form-data、同名 `file` フィールドを必要数 | `filename`（先頭）, `filenames`（全件） |
| POST `/upload/init` | `{"filename":"...","size":<bytes>}` | `upload_id`, `chunk_size`。チャンクは既定10MiB単位 |
| POST `/upload/chunk` | multipart/form-data。`upload_id`, `index`, `total`, `chunk` | `received`, `total`, `index`, `chunks`。順序どおりのみ受理 |
| POST `/upload/complete` | `{"upload_id":"..."}` | 確定後の `filename`, `filenames` |
| GET/HEAD `/files/<filename>` | uploadの応答を使用 | 本人のファイル。Range対応はファイルの保存状態による |
| GET/HEAD `/files/thumb/<filename>` | 本人の画像 | WebPサムネイル。失敗時は元ファイルを明示的に取得してよい |
| GET `/api/storage` | なし | 現行の容量使用量・制限情報 |
| GET `/api/files` | `limit`, `offset`, `sort`, `q`, `favorites_only` | 本人のファイルライブラリ（`files`, `total`, `has_more`） |
| POST `/api/files/favorite` | `filepath` | お気に入りの切り替え |
| POST `/api/files/rename` | `filepath`, `filename` | 表示名の変更 |
| POST `/api/files/delete` | `filenames`（配列） | 本人のファイルを削除 |
| GET `/api/gems` | なし | 本人のGem配列（トップレベルJSON配列） |
| POST `/api/gems` | `name`, `description`, `instruction`, `default_model` | 作成したGem |
| GET/PUT/DELETE `/api/gems/<uuid>` | PUTは作成と同じ項目 | 取得・更新・削除。所有者のみ |
| GET `/api/mobile/v1/preferences` | なし | 既定モデル／Vision、ツール既定値、Enter送信、ライト／Liquid Glass、テーマ、プロンプトバー、STT、システムプロンプト、前回設定、2FA状態、この端末のセッション情報。APIキーは含まない |
| PUT `/api/mobile/v1/preferences` | GETと同じ非秘密フィールド。`prompt_bar_mode`（normal／compact／minimal）、`system_prompt`、`stt_model`、`mic_transcribe_mode` など | 更新後の設定。APIキー・パスワード・E2EE切替等の未知キーは無視し、値を変更しない |
| GET `/api/feedback`、POST `/api/feedback` | POSTは `title`, `message` | 本人のフィードバック一覧と送信。管理者用の全件取得や返信更新は対象外 |
| GET `/api/mcp/servers` | なし | 登録済みMCPサーバーの名前・有効状態・接続状態。秘密は含まない |
| PUT `/api/mcp/servers/<id>` | `{ "enabled": true/false }` のみ | 本人のMCPサーバーの有効切替。Bearer／OAuth秘密の更新は403 |
| GET `/api/gemini/batch/status` | なし | Provider状態を更新し、完了・実行中Batchを返す |
| GET `/api/batch/jobs` | なし | 本人のBatch履歴（最大500件） |
| POST `/api/batch/jobs/<job_id>/cancel` | `{}` | 本人の実行中Batchを停止 |
| DELETE `/api/batch/jobs/<job_id>` | なし | 本人の終了済みBatch履歴を削除 |
| POST `/api/mcp/chat/<job_id>/decision` | `decision`（`allow` / `deny`） | 本人の実行中チャットで要求された変更系MCPツールを今回だけ許可または拒否 |

IDは文字列として扱います。古い履歴の数値IDを受け取る場合もあります。`limit` は最大200。`before_id` を使って古いメッセージをページングし、初期表示で全履歴を要求しないでください。

### 4.1 送信JSONの最小例

```json
{
  "thread_id":"<作成済みスレッドID>",
  "model":"<通常チャット対応の有効なモデルID>",
  "message":"こんにちは",
  "client_request_id":"<送信操作ごとに生成したUUID>",
  "image_urls":[]
}
```

`model` は省略できません。`/me` の `models` から `selectable: true` のモデルだけを選び、`mode` に応じた入力と `capabilities` に含まれる設定だけを表示します。ネイティブ対象は `chat`、`image`、`video`、`ocr`、`tts`、`transcription`、`agent` に加えてOpenAI／Grokの `realtime_audio` とLyriaの `music` です。Gemini LiveとSTSはWeb導線を使います。既存のモデル用APIキーがない場合は400と `code: api_key_missing` 等が返るため、Webの設定を案内します。

`gem_uuid` を付けると、そのGemの指示が回答生成に適用されます。`/api/gems` が返す本人の `uuid` だけを使い、他ユーザーのGemを指定しないでください。現在のスレッドの最後に使ったGemは、スレッド取得応答の `last_gem_uuid` で確認できます。

対応モデルでは `batch_mode: true` を付けるとProviderのBatch APIへ投入します。Androidは30秒間隔で実行中ジョブを確認し、通知権限がある端末では終了時にシステム通知を表示します。`enable_python` と `enable_mcp` は通常チャットの構造化ツールイベントを有効にしますが、秘密設定は返しません。

`thread_id` を省略すると送信時にスレッドを作成できます。ただし再送と画面状態の管理を簡単にするため、初期実装では先にスレッドを作ってIDを保存する方法が扱いやすいです。

`client_request_id` は8〜64文字の英数字・`_`・`-`。UUID文字列を使用できます。利用者の同じ送信操作の再試行では同じIDを維持します。409 `code: request_already_accepted` が返れば応答中の `thread_id` と `job_id` で履歴確認・再接続します。425 `submission_in_progress` なら短く待ちます。冪等情報は永久保存ではないため、長時間後の自動再送はせず履歴を照合してください。

通常の `message` は最大500,000文字。空のメッセージは有効な添付がある場合のみ認められます。通常API全体の本文上限等も別途適用されます。返信は `parent_id` の木構造で分岐します。通常送信は現在の表示経路の末端IDを `parent_id` に渡し、編集・再生成は編集対象メッセージの親IDを `parent_id` に、`parent_id_explicit: true` を付けて送ります（最初のメッセージを編集するときは `parent_id` を `null` にします）。他のスレッドのメッセージIDを送らないでください。

### 4.2 添付

1. AndroidのStorage Access Framework／Photo Picker／カメラで選択・撮影したURIを開く。
2. URIのバイト列を `/upload` の `file` パートとして送る。端末内のパス文字列を送る方式ではない。選択中は端末側で総バイト数から進捗率を計算して表示し、キャンセル操作で転送を中断できる。画像はアップロード前に、一般設定の圧縮設定（有効・最大サイズMB・最大辺px・出力形式・形式のみ変換）に従って端末内で縮小・再エンコードする。GIFはアニメーション維持のため対象外とする。
3. 8MiBを超える添付は `/upload/init` → `/upload/chunk`（応答の `chunk_size` 単位）→ `/upload/complete` の順で送る。チャンクは順序どおりに送り、`409` のときは履歴を照合してやり直す。中断したチャンクセッションはサーバー側で期限切れ回収される。
4. 応答 `filenames` の相対参照をチャットの `image_urls` に入れる。これは画像以外の対応ファイルにも使う既存のフィールド名。
5. 表示時は同一サーバーの `/files/<参照>` にBearerを付ける。パスの各要素をURLエンコードする。
6. 画像は `/files/thumb/<参照>` のWebPサムネイルでメッセージ内にプレビューする。画像以外はMIME・拡張子に応じた種別ラベルを付け、外部アプリで開く。

既定の最大添付数は30、単発アップロードは1ファイル64MiBまで。上限はサーバー設定が優先です。

AIが返した外部画像URL、リダイレクト先、任意リンクにはAndroidトークンを付けません。認証付きHTTPクライアントを全画像ロードのグローバル既定にしないでください。

## 5. ストリーミング・切断復帰

`/chat_stream` と `/chat_stream_resume` の形式は `application/x-ndjson`。UTF-8のJSONオブジェクトを改行で区切ります。SSEの `data:`、`event:`、`[DONE]` を前提にした実装は使用できません。

```text
{"type":"thread_id","content":"abc123"}
{"type":"job_id","content":"job_..."}
{"type":"status","content":"..."}
{"type":"content","content":"こんにちは"}
{"type":"content","content":"。"}
{"type":"done","content":"..."}
```

これは通常テキストの説明例です。`content` はイベントによって文字列・オブジェクト等が変わる可能性があります。

| type | 処理 |
|---|---|
| `thread_id` / `job_id` | 受信直後に現在の送信操作へ紐付けて保存 |
| `status` | 進行状況表示を更新 |
| `thought` | 思考表示領域に追記。アシスタント本文と分ける |
| `content` | 通常テキストモデルでは本文を追記 |
| `done` | ストリーム完了。その後GETで保存済み履歴を取得し表示を確定 |
| `error` | 生成エラーとして終了。保存済みのエラーメッセージも履歴から確認 |
| その他 | 未知のtypeだけを理由に落とさない。画像・検索・ツール等は対応UIを追加するまで安全に扱う |

ネットワークの読み取り1回がJSONの1件とは限りません。複数行がまとめて届くことも、1行や日本語のUTF-8が分割されることもあります。バイトからテキストへの復号と改行バッファを使い、1行が完成してからJSONを解析します。

通常APIは接続15秒・読み取り30秒、生成ストリームは無通信の読み取り待ちを660秒とする例を後述します。ストリームはUIからキャンセル可能にし、`done/error` のないEOFを成功と扱わないでください。現行サーバーに一定周期のheartbeat送信保証はありません。CDNの無通信タイムアウトはApacheの設定だけでは延長できません。

### 5.1 切断した場合

1. 同じpromptを新しいrequest IDで即再送しない。
2. `GET /api/threads/<id>?limit=50` を実行し、保存済みメッセージと `pending_job` を確認する。
3. pendingのjobがあれば `/chat_stream_resume` を呼ぶ。
4. resumeは先に蓄積済み本文・思考を返す。そのため、その生成についての仮表示バッファをクリアしてから再接続内容を取り込む。既存表示にそのまま追記すると重複する。
5. resumeが404 `no pending job` / `job mismatch` なら履歴を再取得して終了状態を確認する。

ストリームには完全なイベント連番・exactly-once保証がありません。競合による重複等があり得るため、完了後のDB履歴を表示の正本にします。アプリ停止中もRQでの生成が続く場合があります。

## 6. Androidプロジェクトの作成

### 6.1 開発環境

既存の `android/` をAndroid Studioで開きます。新しくEmpty Activityを作り直す必要はありません。固定applicationIdは `com.minashin1120.aiplayground` です。上書きインストールを維持するため変更しません。リポジトリのルートはFlask用、Gradleのルートは `android/` で、Actionsの定義だけはリポジトリ直下の `.github/workflows/` に置きます。

Android 8.0以上（minSdk 26）、compileSdk / targetSdk 36、JDK 17、AGP 8.13.0、Gradle 8.13、Kotlin 2.2.10、Compose BOM 2025.08.00を使用します。版の正本は `android/app/build.gradle.kts` と `android/build.gradle.kts`、Wrapperのpropertiesです。Gradle配布ZIPとWrapper JARはSHA-256を検証します。

主な通信・認可画面用の依存（すでに設定済み）：

```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:5.3.0")
    implementation("androidx.browser:browser:1.9.0")
}
```

更新時は[OkHttp公式リポジトリ](https://github.com/lysine-dev/okhttp#releases)と[AndroidX Browserの公式リリース情報](https://developer.android.com/jetpack/androidx/releases/browser)で互換性を確認し、Actionsのテスト・lint・両APK構築を通します。動的バージョン指定は使いません。

### 6.2 Manifestとネットワーク

`AndroidManifest.xml` の既存manifest/applicationへ次の属性を統合します。下の断片で既存Activity定義を置き換えないでください。

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <uses-permission android:name="android.permission.INTERNET" />
    <application
        android:usesCleartextTraffic="false"
        android:allowBackup="false"
        android:networkSecurityConfig="@xml/network_security_config">
        <!-- Android Studioが作成したActivity等をここに維持 -->
    </application>
</manifest>
```

`res/xml/network_security_config.xml`：

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

証明書やホスト名検証を無効化しません。開発機の独自CAが必要ならdebug用設定に限定し、releaseから除外します。ネットワーク設定の詳細は[Android公式資料](https://developer.android.com/privacy-and-security/security-config)を参照してください。

この連携方式にはコールバック用の独自URIスキームや `assetlinks.json` は不要です。将来App Linksで自動復帰させる場合は、正式applicationIdと配布署名のSHA-256を確定してから別途設定します。

### 6.3 実装構成

```text
MainActivity.kt                  ブラウザー・添付の起動、画面ライフサイクル
ChatViewModel.kt                 端末連携、履歴、送信、再接続、添付、失効
data/PlaygroundApi.kt            固定接続先、CookieなしHTTP、JSON/NDJSON、キャンセル
data/TokenStore.kt               Keystoreによるトークン暗号化保存
data/Models.kt                   データ変換、エラー表示、応答サイズ制限
ui/PlaygroundScreen.kt           連携、履歴、モデル選択、チャット画面
ui/ChatComponents.kt             入力、メッセージ、添付、コードブロック
```

これらは `android/app/src/main/java/com/minashin1120/aiplayground/` 配下です。ファイルの地図は [Android README](../android/README.md) にあります。

### 6.4 ビルドと配布

通常はGitHubのActions → **Android CI** → **Run workflow**（main）を使います。`android/**` またはAndroid workflowの変更をmainへ反映したときも自動実行されます。サーバーだけの更新はAndroidビルドを起動しません。

1. 初回だけ、Actionsが `android/ci/debug.keystore` を生成し、同じファイルと `signing-fingerprint.txt` をmainへ記録・送信します。保存に失敗した場合はAPKを作りません。
2. 固定鍵を含む確定コミットをcheckoutし、JDK 17とAndroid SDK 36を準備します。
3. JVM通信テスト、lint、debug APK構築、APK署名検証を行い、`app-debug` artifactを30日保存します。
4. 同じ確定コミットからrelease APKを構築・検証し、`app-release-signed` artifactを90日保存します。
5. `android/version.properties` のVERSION_NAMEから `android-vX.Y.Z` タグを作り、APK・APK SHA-256・署名証明書SHA-256をGitHub Releaseに掲載します。Webの `v4.8.xxx` タグとは分離しています。

PRでは書き込み・鍵生成・Release作成をしません。固定鍵がまだmainに無い場合は、先にmainのAndroid CIを完了します。Actionsの標準 `GITHUB_TOKEN` だけを使用し、署名用のカスタムSecretは不要です。リポジトリのActions書き込み設定やブランチ保護によってmainへの記録が拒否される場合は、管理者がその権限を整備してください。スクリプトは保護を迂回しません。

### 6.5 固定署名鍵の運用

debug/releaseとも `android/ci/debug.keystore` だけを使います。storePassword / keyPasswordは `android`、keyAliasは `androiddebugkey`。鍵が存在する場合は生成しません。空ファイルや履歴上で削除済みの鍵を検出した場合は停止し、正しい既存ファイルを復元します。ローカル鍵へのフォールバック、base64化、署名用Secretsは使いません。

`android/ci/signing-fingerprint.txt` は初回の証明書SHA-256です。CIは `keytool -list -v` のSHA-1/SHA-256をログに出し、`apksigner verify --print-certs` のAPK署名証明書と固定fingerprintを照合します。APK自体の `.sha256` はファイル改ざん検知用であり、証明書fingerprintとは別です。どちらの鍵関連ファイルも更新の都度作り直さないでください。

初回に保存された署名証明書（2026-09-15）：

```text
SHA-1:   0B:69:41:70:FF:D5:23:86:16:82:3D:D6:82:71:FA:DC:15:F9:10:54
SHA-256: 6FA6AD4EAA885046A12123F72D8699300AB6554E22E9AC2D740FC21553B0ECA0
```

**この署名鍵は公開共有鍵です。誰でも同じ署名のAPKを作成できるため、署名一致だけでは公式配布元を証明できません。** APKはこのリポジトリのReleaseから取得し、第三者の再配布APKを信用しないでください。これは指定の再ビルド・上書き互換性を優先した運用で、秘密鍵を非公開にする一般的なストア配布設計とは異なります。Android Keystore内の端末トークン暗号化鍵やサーバーのデータ暗号化鍵を公開するものではありません。

### 6.6 更新・再実行・ローカル構築

配布する変更では `VERSION_CODE` を必ず増やし、`VERSION_NAME` も未使用の値に変更します。同一タグが別コミットを指す場合はRelease作成を失敗させ、タグや既存APKを上書きしません。同じコミットの再実行では既存Releaseを保持します。手動 **Android Release** は `source_sha` にmain上の確定コミットを指定でき、任意の `tag_name` を入れる場合もVERSION_NAMEと一致する必要があります。鍵が無い状態でReleaseだけを実行しても生成は行いません。

固定鍵の初回CI保存後、任意でローカル構築する場合：

```bash
cd android
./gradlew testDebugUnitTest lintDebug assembleDebug
./gradlew lintRelease assembleRelease
```

出力先はリポジトリから見て `android/app/build/outputs/apk/debug/app-debug.apk` と `android/app/build/outputs/apk/release/app-release.apk` です。ローカルに鍵が無ければ最新mainを取得してください。外部PCで鍵を作る必要はありません。

APKを端末へダウンロードし、そのダウンロード元アプリの「不明なアプリのインストール」を許可して開きます。既存アプリのapplicationIdと証明書が同じで、versionCodeが下がらなければ通常は上書きできます。過去の異なる鍵で入れたAPKでは `INSTALL_FAILED_UPDATE_INCOMPATIBLE` になります。その場合だけアンインストールして入れ直します（端末連携・端末内データは消えます）。[公式ビルド手順](https://developer.android.com/build/building-cmdline)と[アプリ署名](https://developer.android.com/studio/publish/app-signing)も参照してください。

## 7. Kotlin通信コード

以下はプロトコル理解用の短い通信例です。実際にビルドされる実装は `android/app/src/main/` が正本です。そこでは画面・永続保存・キャンセル・サイズ制限を含めて実装しています。ActionsのJVMテストとlintは実機試験の代わりではないため、端末確認項目も実施してください。

### 7.1 Cookieを持たない専用クライアント

```kotlin
import okhttp3.CookieJar
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class PlaygroundApi {
    private val origin = "https://ai.minashin1120.com/".toHttpUrl()
    private val jsonType = "application/json; charset=utf-8".toMediaType()
    private val http = OkHttpClient.Builder()
        .cookieJar(CookieJar.NO_COOKIES)
        .followRedirects(false)
        .followSslRedirects(false)
        .retryOnConnectionFailure(false)
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .callTimeout(90, TimeUnit.SECONDS)
        .build()
    private val streaming = http.newBuilder()
        .readTimeout(660, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.SECONDS)
        .build()

    data class JsonReply(val status: Int, val body: JSONObject, val retryAfter: Long?)

    private fun request(path: String, token: String?): Request.Builder {
        require(path.startsWith("/") && !path.startsWith("//"))
        val url = requireNotNull(origin.resolve(path))
        require(url.scheme == origin.scheme && url.host == origin.host && url.port == origin.port)
        return Request.Builder().url(url)
            .header("User-Agent", "AIPlayground-Android/1.0")
            .apply { if (token != null) header("Authorization", "Bearer $token") }
    }

    // Blocking: call from Dispatchers.IO, never the UI thread.
    fun json(path: String, payload: JSONObject? = null, token: String? = null): JsonReply {
        val builder = request(path, token).header("Accept", "application/json")
        if (payload != null) builder.post(payload.toString().toRequestBody(jsonType))
        http.newCall(builder.build()).execute().use { response ->
            val body = response.body ?: throw IOException("Empty response")
            if (body.contentType()?.subtype != "json") throw IOException("Non-JSON HTTP ${response.code}")
            return JsonReply(response.code, JSONObject(body.string()),
                response.header("Retry-After")?.toLongOrNull())
        }
    }

    // onEvent runs on the calling I/O thread. Dispatch UI updates separately.
    fun stream(path: String, payload: JSONObject, token: String, onEvent: (JSONObject) -> Unit) {
        val req = request(path, token).header("Accept", "application/x-ndjson")
            .post(payload.toString().toRequestBody(jsonType)).build()
        streaming.newCall(req).execute().use { response ->
            val body = response.body ?: throw IOException("Empty response")
            if (!response.isSuccessful || body.contentType()?.subtype != "x-ndjson") {
                // Production: parse bounded JSON errors into status/code/error;
                // retain request_already_accepted's thread_id and job_id.
                throw IOException("Stream HTTP ${response.code}")
            }
            val source = body.source()
            while (!source.exhausted()) {
                // Bound each line. Increase deliberately for supported media events.
                val line = source.readUtf8LineStrict(8L * 1024 * 1024)
                if (line.isBlank()) continue
                val event = JSONObject(line)
                onEvent(event)
                if (event.optString("type") in setOf("done", "error")) return
            }
            throw IOException("Stream interrupted before terminal event")
        }
    }
}
```

本番の実装ではCallを保持し、コルーチンのキャンセル時に `Call.cancel()` を呼んでください。`withContext(Dispatchers.IO)` だけではブロッキング通信が即座に停止する保証はありません。ストリーム用クライアントと通常用クライアントを分けることで、停止・履歴取得を長い読み取り待ちから独立させます。

エラー本文をログへ丸ごと書かず、HTTP status、安定したerror/code、アプリ版など必要最小限の診断情報だけを扱います。実装時には通常JSON本文にもサイズ上限を設け、大きいファイルはメモリーへ一括読み込みしません。

### 7.2 ブラウザーを開く

コードはアプリ画面に表示したまま、メインスレッドから開きます。

```kotlin
import android.net.Uri
import androidx.browser.customtabs.CustomTabsIntent

CustomTabsIntent.Builder().build().launchUrl(
    activity,
    Uri.parse("https://ai.minashin1120.com/android/connect")
)
```

Custom Tabsは既存ブラウザーのログインを利用できます。対応ブラウザーがない場合は通常のブラウザーIntentへフォールバックし、それも使えなければ利用者へ説明します。埋め込みWebViewでパスワードやWeb Cookieを回収する実装は不要です。[Custom Tabsの公式説明](https://developer.android.com/develop/ui/views/layout/webapps/overview-of-android-custom-tabs)を参照してください。

### 7.3 ポーリングの状態管理

```text
未連携 → device発行 → コード表示／ブラウザー承認待ち
                     ├─ authorization_pending → 待機 → token確認
                     ├─ 429 → Retry-After＋バックオフ → token確認
                     ├─ 承認 → トークン保存 → me検証 → ログイン済み
                     └─ 拒否・期限切れ → 未連携
```

タイマーは単調増加時計（Androidの `SystemClock.elapsedRealtime()`）を使い、端末の時計変更で無限に待たないようにします。アプリがバックグラウンドの間にポーリングを続ける必然性はありません。ブラウザーから戻った `onStart` 相当で、残り期限と前回問い合わせ時刻を確認して再開します。

アプリプロセスが終了した場合の初期実装は、新規device発行からやり直して構いません。device_codeを永続化するならアクセストークンと同等に保護し、期限切れ時に必ず削除します。

## 8. 端末保存・画面・ライフサイクル

### 8.1 トークン保存

Android KeystoreでAES-GCM用の鍵を生成し、アクセストークンはその鍵で暗号化してアプリ内部ストレージへ保存します。Keystoreにトークンそのものを保存するのではなく、暗号化鍵を保存する構成です。

- 保存形式に版番号、ランダムIV、暗号文＋認証タグ、expires_atを含める。
- 暗号化のたびに新しいIVを生成する。同じ鍵でIVを使い回さない。
- 保存にはバックアップ対象外の `context.noBackupFilesDir` を用いる。クラウド・端末移行のバックアップ設定も確認する。
- メモリーキャッシュはログアウト時に消去する。トークンをSavedStateHandle、Composeの保存状態、クラッシュ報告、分析SDKへ入れない。
- 鍵が利用できない・復号できない場合は保存を消して再連携し、暗号化を迂回しない。
- 401や期限切れ時は再連携。認証を要しない画面までネットワーク再試行を無限に続けない。

鍵の生成・保護の詳細は[Android Keystore公式資料](https://developer.android.com/privacy-and-security/keystore)を参照してください。

### 8.2 画面と状態

ViewModelが送信状態を保持し、Composeはその状態を表示する構成にします。一時チャットのheartbeatは対象スレッドを表示してアプリがフォアグラウンドにある間だけ送信し、離席中は自動削除タイマーを妨げません。回転や再Compositionをきっかけに送信・ポーリングを二重起動しないでください。送信ごとのUUID、thread_id、job_id、仮表示中の本文・思考を区別します。

通信表示は「送信中」「応答待ち」「受信中」など実際に観測した状態から決めます。`finally` で待機表示を解除します。Web側の共通スピナーは `static/js/progress_spinner.js` が管理しており、Androidは自身のViewModelで同じ考え方を実装します。

MarkdownはHTML/WebViewへ変換せず、許可したブロックとインライン記法だけをComposeで描画します。リンクはhostを持つHTTP(S)だけを外部ブラウザーで開き、認証付きHTTPクライアントへ渡しません。コードは選択、コピー、長い場合の折り畳みに対応します。数式はLaTeXをネイティブの読みやすい近似へ変換し、未対応コマンドは元の文字列を表示します。表、画像プレビュー、Web検索とPython実行の状態カードもComposeで描画します。768dp以上の幅では履歴と会話を2ペインで同時表示し、それ未満では履歴Drawerを使います。スマートフォンの設定・ライブラリ等は全画面に近いIME対応モーダルとし、Androidの戻る操作は開いているモーダル、Drawer、入力状態の順に閉じます。音声権限、カメラ権限、通知権限は、その機能を実装して利用者が使用する段階で追加します。

## 9. サーバー設定と公開

### 9.1 アプリ設定

```dotenv
MOBILE_API_ENABLED=1
```

未設定時も有効。停止したい場合は `0` を設定してWebサービスを再起動します。環境変数名とサンプル値は [.env.example](../.env.example) にあります。実際の `.env` を共有・Git追跡しないでください。

今回、DBへのカラム・テーブル追加はありません。既存の `UserSession` を使うため、Android対応を理由に `RUN_SCHEMA_MIGRATIONS=1` に変更する必要はありません。Redisには `mobile:` 名前空間の短期データを追加します。新しいポートや別の認証サービスも不要です。

GunicornとRQは既存の暗号化鍵設定・DB・Redisを共有します。Redisが停止した場合、device/tokenは503を返して発行を止めます。既発行トークンの検証自体はDBを参照しますが、チャット生成等のRedis依存機能は影響を受けます。

### 9.2 Apache

HTTPS VirtualHost内から [apache/mobile-api.inc.conf](apache/mobile-api.inc.conf) をIncludeします。配布サンプルは [apache/ai-playground.conf](apache/ai-playground.conf)。配置先を環境に合わせて指定してください。

```apache
ProxyPreserveHost On
ProxyPass / http://127.0.0.1:3111/ connectiontimeout=5 timeout=660 retry=0
ProxyPassReverse / http://127.0.0.1:3111/
ProxyTimeout 660
Include /opt/ai-playground/deploy/apache/mobile-api.inc.conf
```

includeはHTTPSのVirtualHost専用です。HTTP側に入れてHTTP通信をHTTPSと偽装する設定にしないでください。元のホスト名、TLS証明書、静的資産のAlias、既存のProxyPass除外を維持します。[Apache公式のmod_proxy資料](https://httpd.apache.org/docs/2.4/mod/mod_proxy.html)も参照してください。

今回の本番対象は `/etc/apache2/sites-enabled/ai.minashin1120.com.conf`。そのHTTPS VirtualHostのProxyTimeoutを300秒から660秒にし、アプリ配置先の `deploy/apache/mobile-api.inc.conf` を読み込む構成に更新します。上の `/opt/ai-playground` は配布例です。実際に設定された絶対パスは対象VirtualHostのInclude行で確認してください。

共有includeは次を設定します。

- Apacheが `X-Forwarded-Proto: https` と `X-Forwarded-Port: 443` を上書きする。
- API、チャット、アップロード、ファイル、連携画面で `ProxyErrorOverride Off` にし、FlaskのJSONエラーを維持する。
- 端末認証とチャットストリームは共有キャッシュ禁止、参照元送信禁止、Apacheによるgzip圧縮対象外にする。

Gunicornの3111番はloopbackにバインドし、外部へ公開しません。インターネットからGunicornへ直接到達できると、転送ヘッダーを信頼する前提が崩れます。`TRUSTED_HOSTS` は公開ホストに限定します。

反映時はバックアップを保持し、構文確認の成功後にreloadします。

```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

includeを利用する構成では、アプリの配置先を移動するときもIncludeパスを更新してください。Android対応前のコードへ戻す際にincludeファイルだけ消すとApacheが再読込できなくなるため、先にvhostのIncludeを外すか、ファイルを保持します。

### 9.3 CDN・WAF・ログ

- `Authorization` ヘッダーをGunicornまで転送する。削除・書き換えしない。
- `/api/mobile/*`、`/android/connect`、`/chat_stream*`、認証付き `/api/*` と `/files/*` を共有キャッシュへ保存しない。
- キャッシュルールで「全ページを強制キャッシュ」してアプリの `no-store` を上書きしない。
- APIのPOSTにJavaScriptチャレンジを要求するとネイティブクライアントがHTMLを受け取る。必要なら対象ホスト・対象APIの該当ルールだけを調整する。
- User-Agentやclient_idだけを根拠にBot対策・WAF全体を無効化しない。
- CDN固有の無通信時間制限はプランや設定にも依存する。長時間生成・再接続を実環境で試験する。
- `Authorization`、Cookie、device_code、token応答、チャット本文をApacheやアプリのログ書式に追加しない。

CDNの管理画面の設定変更はこのリポジトリの設定例だけでは反映されません。まず公開URLで実際のステータス、Content-Type、Cache-Controlを確認し、問題が出たルールを特定して変更します。

### 9.4 リリース手順

この環境では編集完了後、公開向け更新履歴を用意して次を実行します。長時間コマンドはバックグラウンドで1回起動して、そのプロセスの完了を確認します。

```bash
scripts/prepare_version.sh --notes "Androidクライアント用の端末連携と通信APIを追加しました。"
scripts/publish_version.sh --message "Add Android pairing and scoped native API access"
# 計画のファイル一覧と現在のSYSTEM_VERSIONを確認してから指定する
scripts/publish_version.sh --message "Add Android pairing and scoped native API access" --confirm V4.8.xxx
```

`V4.8.xxx` は実際の現在版に置き換えます。prepareは同じリリースに複数回実行しません。publishは指定されたパスのみcommit/tag/pushし、サービスを再起動して公開URLを確認します。Apache設定のreloadは別途必要です。

## 10. 試験・受け入れ条件

### 10.1 サーバーの自動試験

`tests/test_mobile_api.py` は分離SQLiteとUnixソケットの一時Redisを使い、実際のLuaを含めて検証します。redis-serverがない環境ではこのテスト群はskipされるため、公開確認ではskipを成功と取り違えないでください。

対象は、承認・拒否・期限切れ・二重引き換え防止・並列競合・ポーリング制限・Redis障害・HTTPS・停止スイッチ・Cookie/Origin混在拒否・WebのCSRF維持・失効・BAN・E2EE・Turnstile・スレッド所有者・タイトル・スレッド指示・ブックマーク・一時チャット・添付所有者・チャンクアップロードの確定と順序検証・ファイルライブラリの所有者境界・Gemの所有者境界とCRUD・一般設定の許可項目と秘密鍵の拒否・PDF出力の所有者境界・モデルメタデータと選択可否です。

既存の全体回帰テストはprepareの内部で実行します。実際のAI事業者への有料生成、Android UI、署名APKのインストールはサーバー単体テストでは検証しません。

### 10.2 秘密情報を使わない公開URL確認

```bash
curl --fail-with-body --silent --show-error \
  https://ai.minashin1120.com/api/mobile/v1/config

curl --silent --show-error --include \
  https://ai.minashin1120.com/api/mobile/v1/me
```

configは200 JSON、未認証meは401 JSONが期待値です。テスト出力を共有するときにSet-Cookie等が混ざる場合は除去します。トークン付きcurlコマンドをシェル履歴やプロセス引数へ残さず、認証付き試験にはテスト専用クライアントを使います。

### 10.3 APKの実機確認

| 試験 | 期待結果 |
|---|---|
| パスワード／SSO／Passkey／2FAで連携 | 元の認証を省略せず、承認画面に戻れる |
| 拒否・放置・別端末コード | 拒否・期限切れを表示し、誤連携しない |
| 回転・バックグラウンド・プロセス終了 | 二重送信しない。期限とログイン状態を整合させる |
| Wi-Fiからモバイル回線へ切替 | 切断を検知し、job_idから復帰できる |
| 409 / 425 / 429 / 503 | 同じ投稿を重複保存せず、適切に待機する |
| 長い生成とdone前のEOF | 途中終了を成功表示せず、保存済み履歴で照合 |
| 複数添付・大きすぎる添付 | 上限エラーを表示。再送で二重投稿しない |
| 外部画像・リンク | 外部ホストへBearerが送られない |
| モデル選択と設定 | 非チャット・廃止モデルを選べず、対応能力に含まれない設定を送らない |
| Markdown・コード・リンク | 文字選択、コードコピー・折り畳みが動作し、非HTTP(S)リンクを開かない |
| Webから端末失効 | 次のAndroid API呼び出しが401になり再連携を促す |
| WebのE2EE設定 | 有効なまま連携・履歴閲覧・添付ができ、設定が変わらない |
| BAN・Turnstile | 機能制限を回避せず説明する |
| 署名付きrelease APK | インストール・更新・端末保存・バックアップ除外を確認 |

## 11. 障害対応・制限・今後の拡張

| 現象 | 調べる項目 |
|---|---|
| 200だがHTMLが返る | CDNチャレンジ、ログインページ、リダイレクトの自動追跡 |
| 403/503がHTMLに変わる | ApacheのProxyErrorOverride、CDNのカスタムエラー |
| HTTPS接続なのにhttps_required | HTTPS vhostのInclude、転送ヘッダーの上書き、実際の接続先 |
| tokenがずっとpending | アプリとブラウザーのコード照合、承認操作、Redisを全Webワーカーで共有しているか |
| token取得後401 | 期限、DBセッションの失効、トークン保存破損、誤ったホスト |
| cookies_or_origin_not_allowed | OkHttpのCookieJar、WebView/CookieManagerとの共有、共通Interceptor |
| insufficient_scope | 許可されていないURL・メソッドを呼んでいないか |
| turnstile_required | 同じアカウントでWebの安全性確認を完了して再試行。サーバー側の認証マーカーは既存仕様で15分 |
| 古いサーバーの e2ee_not_supported | サーバーを互換対応版へ更新する。Webの暗号化設定を無効化する必要はない |
| 生成中に切断 | CDN/Gunicorn/Apache/端末の待機時間、回線変更、アプリのライフサイクル |
| 再接続後に本文が重複 | 再接続前の仮表示バッファをクリアしているか、最後にDB履歴へ置き換えているか |

Turnstileの確認にはアプリ内の「Webで安全性を確認」導線を用意します。今回の専用APIは既存の確認マーカーを使い、Androidという理由で確認を免除していません。このため、Bot対策対象アカウントでは継続利用中に再確認が必要になる場合があります。

今後の拡張候補は、ブラウザー認可コード＋PKCEと検証済みApp Links、自動復帰、refresh tokenのローテーション、端末間E2EEの新しい設計、Webと説明・価格まで共有するモデルカタログ、リアルタイム音声、イベント連番による再開です。これらは実装済み機能としてクライアントへ表示しないでください。

数式はWebのMathJaxではなくネイティブ近似で描画するため、複雑な組版では表現が異なる場合があります。メッセージの編集・再生成・分岐切替とネイティブPDF出力は実装済みです。画像の圧縮設定（最大サイズ・最大辺・形式・形式のみ変換）も実装済みで、GIFは対象外です。PDFは本文をA4へ再構成する簡易出力で、Webの印刷レイアウトとは体裁が異なります。

`official-android` という公開client_idを知っていれば、他のクライアントも連携申請自体は作れます。利用者のブラウザー認証と明示承認が権限の根拠です。「公式APKからの通信だけを許可する」要件がある場合は、正式な署名・配布基盤とアプリ検証を別途設計します。

## 12. 実装ファイル・公式資料

| ファイル | 担当 |
|---|---|
| [server/mobile_auth.py](../server/mobile_auth.py) | Bearer検証、許可リスト、HTTPS・Cookie境界、レスポンス制御 |
| [server/routes_mobile.py](../server/routes_mobile.py) | 接続仕様、device、承認、token、me、revoke |
| [templates/android_connect.html](../templates/android_connect.html) | コード入力・権限確認・承認／拒否 |
| [server/request_security.py](../server/request_security.py) | 既存セッション・BAN・Bot・メンテナンスとの統合 |
| [server/app_settings_schema.py](../server/app_settings_schema.py) | 条件を満たすnative通信だけのCSRF免除 |
| [tests/test_mobile_api.py](../tests/test_mobile_api.py) | 分離環境での認証・API・競合回帰テスト |
| [apache/mobile-api.inc.conf](apache/mobile-api.inc.conf) | ApacheのHTTPS転送、JSONエラー維持、キャッシュ設定 |

APIの具体的な契約は本書と現行サーバーコードが正本です。Android・HTTPライブラリ・Apacheの一般仕様は各節の公式資料を参照し、依存更新・配布時には情報を再確認してください。
