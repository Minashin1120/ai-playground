# AI Chat Playground

[![Version](https://img.shields.io/badge/version-V4.8.867-2563eb)](static/changelogs/20260828_v4.8.867.md)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-f59e0b)](LICENSE)

AI Chat Playground は、複数の生成AIを1つの画面から利用するためのセルフホスト型Webアプリケーションです。OpenAI、Google Gemini／Vertex AI、Anthropic、DeepSeek、Kimi（Moonshot）、xAI、Mistral OCRに対応し、BYOK、チャット履歴、ファイル、画像・音声・動画、Passkey／TOTP、PWA、アカウントデータ移行などを提供します。

> [!WARNING]
> V4.8.615以前には既知のセキュリティ上の問題があります。新規導入・本番運用には使用しないでください。

> [!IMPORTANT]
> このアプリは `pip install` だけでは動作しません。MariaDB、Redis、Gunicorn、4つのRQワーカー、Apacheまたは同等のリバースプロキシ、HTTPS、永続ストレージ、秘密情報の設定が必要です。本番導入は本書の「本番環境への導入」を最後まで実施してください。

## 目次

- [主な機能](#主な機能)
- [構成](#構成)
- [必要条件](#必要条件)
- [本番環境への導入](#本番環境への導入)
- [初回セットアップ](#初回セットアップ)
- [更新](#更新)
- [バックアップと復元](#バックアップと復元)
- [開発とテスト](#開発とテスト)
- [トラブルシューティング](#トラブルシューティング)
- [ドキュメント案内](#ドキュメント案内)
- [ライセンス](#ライセンス)

## 主な機能

- 複数プロバイダーのテキスト、画像、音声、Realtime、動画モデル
- ユーザー自身のAPIキーを暗号化して保存するBYOK
- Gemini APIとVertex AIの切り替え、モデル別APIキー
- ストリーミング応答、生成ジョブのRedis/RQ実行、切断後の再接続
- 画像・PDF等の添付、分割アップロード、ファイルライブラリ
- チャットとファイルのE2EE、TOTP、WebAuthn Passkey、セッション管理
- Cloudflare Turnstile、レート制限、管理者向けBAN・診断機能
- PWA、オフライン画面、テーマ、低帯域・パフォーマンス設定

対応モデルは [MODELS.md](MODELS.md) を参照してください。モデル名、提供状況、価格は変わるため、実際の利用前に各プロバイダーの公式情報も確認してください。

## 構成

本番時の通信経路は次のとおりです。

```text
ブラウザー
  └─ HTTPS :443
      └─ Apache / 任意のリバースプロキシ
          └─ Gunicorn :3111（Flaskアプリ）
              ├─ MariaDB（ユーザー、設定、チャットメタデータ）
              ├─ Redis DB 10（キュー、進捗、レート制限等）
              ├─ RQ worker × 4（生成・移行などのバックグラウンド処理）
              └─ instance/uploads（ユーザーのアップロード）
```

Gunicornだけ、またはRQワーカーだけを起動しても、すべての機能は動作しません。

## 必要条件

基準構成はDebian 12とPython 3.11、MariaDB 10.6以降、Redis 6以降です。Ubuntu 22.04／24.04 LTSでも運用できますが、標準Pythonの版が異なるため、Python 3.11を別途用意するか、使用するPython版で依存関係と全テストを確認してください。少なくとも次のソフトウェアが必要です。

- Python 3.11、`venv`、開発用ヘッダー、ビルドツール
- MariaDBまたはMySQL互換サーバー
- Redis
- Apache 2.4（本書の例）またはSSEと大容量アップロードに対応した同等のプロキシ
- ffmpeg（音声・動画処理）
- bubblewrapとutil-linuxの`prlimit`（Pythonコード実行サンドボックス）
- Cairo、Pango等（CairoSVG／WeasyPrintによるPDF・画像生成）
- TLS証明書（Certbot等）

目安として2 vCPU、4 GB RAM、十分なディスク容量を用意してください。複数の重い生成を並行する場合は増強が必要です。ネイティブWindowsはbubblewrapを利用できないため、本番環境には推奨しません。Windowsでの開発はWSL2を利用してください。

## 本番環境への導入

以下では、ドメインを `chat.example.com`、実行ユーザーを `ai-playground`、配置先を `/opt/ai-playground` とします。実際の値へ置き換えてください。

### 1. OSパッケージを導入する

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv python3.11-dev \
  build-essential pkg-config libssl-dev libffi-dev \
  mariadb-server redis-server apache2 certbot python3-certbot-apache \
  ffmpeg bubblewrap util-linux libcairo2 libpango-1.0-0 \
  libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 shared-mime-info
```

上のコマンドはDebian 12向けです。ディストリビューション標準リポジトリにPython 3.11がない場合は、OSが提供する安全な方法で3.11を追加してください。別のPython版を利用する場合は、依存ライブラリと全テストの互換性を確認する必要があります。

サービスを有効化します。

```bash
sudo systemctl enable --now mariadb redis-server apache2
```

### 2. 専用ユーザーとアプリを用意する

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin ai-playground
sudo git clone https://github.com/Minashin1120/ai-playground.git /opt/ai-playground
sudo chown -R ai-playground:ai-playground /opt/ai-playground
sudo -u ai-playground python3.11 -m venv /opt/ai-playground/venv
sudo -u ai-playground /opt/ai-playground/venv/bin/pip install --upgrade pip
sudo -u ai-playground /opt/ai-playground/venv/bin/pip install -r /opt/ai-playground/requirements.txt
```

`venv/`、`.env`、`secret.key`、`instance/` はGitへコミットしないでください。

### 3. MariaDBを設定する

最初にMariaDB自体を保護します。

```bash
sudo mariadb-secure-installation
sudo mariadb
```

MariaDBコンソールで、データベースとローカル接続専用ユーザーを作成します。

```sql
CREATE DATABASE ai_chat_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'ai_chat_user'@'localhost' IDENTIFIED BY '十分に長いランダムなパスワード';
GRANT ALL PRIVILEGES ON ai_chat_db.* TO 'ai_chat_user'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

MariaDBを外部公開しないでください。`DATABASE_URL` 内のパスワードに `@`、`:`、`/`、`#`、`%` などがある場合はURLエンコードが必要です。

### 4. 環境変数を設定する

サンプルをコピーし、所有者と権限を制限します。

```bash
sudo -u ai-playground cp /opt/ai-playground/.env.example /opt/ai-playground/.env
sudo chmod 600 /opt/ai-playground/.env
sudo -u ai-playground /opt/ai-playground/venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(64))'
```

最後のコマンドの出力を `FLASK_SECRET_KEY` に設定してください。最低限、次の値が必要です。

```ini
FLASK_SECRET_KEY=ここへ生成した値
DATABASE_URL=mysql+pymysql://ai_chat_user:URLエンコード済みパスワード@localhost/ai_chat_db?charset=utf8mb4
REDIS_URL=redis://127.0.0.1:6379/10
TRUSTED_HOSTS=chat.example.com
PRIMARY_ADMIN_USERNAME=
RUN_SCHEMA_MIGRATIONS=0
```

`PRIMARY_ADMIN_USERNAME` は初回登録が終わるまで空欄にします。値と同じユーザー名は新規登録できないため、先に登録してから値を設定し、全サービスを再起動して管理者化します。

APIキーはユーザーが画面から登録できます。管理者用フォールバックを使う場合だけ、該当する `OPENAI_API_KEY`、`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`、`DEEPSEEK_API_KEY`、`MOONSHOT_API_KEY`、`XAI_API_KEY` 等を `.env` に設定します。GoogleログインにはGoogle OAuthが必要です。新規登録フォームはTurnstile検証を必須としているため、初回アカウント作成前に `TURNSTILE_SITE_KEY` と `TURNSTILE_SECRET_KEY` も設定してください。主要項目と既定値は [.env.example](.env.example) を参照してください。

> [!CAUTION]
> `FLASK_SECRET_KEY` と起動時に生成される `secret.key` は別物です。前者を変えるとログインセッションが無効になり、後者を失うと保存済みAPIキー等を復号できません。両方を秘密バックアップに含めてください。

### 5. 初回のDB作成を確認する

アプリのインポート時にテーブルが作成され、必要な必須列が検査されます。サービス化する前に実行ユーザーで一度確認します。

```bash
cd /opt/ai-playground
sudo -u ai-playground /opt/ai-playground/venv/bin/python -c 'from app import app; print(app.config["SYSTEM_VERSION"])'
sudo test -f /opt/ai-playground/secret.key
sudo chown ai-playground:ai-playground /opt/ai-playground/secret.key
sudo chmod 600 /opt/ai-playground/secret.key
```

既存環境の更新で追加スキーマ移行が必要なリリースだけ、バックアップ取得後に一時的に `RUN_SCHEMA_MIGRATIONS=1` として起動し、完了後は `0` に戻してください。複数プロセスで同時に移行を走らせないでください。

### 6. systemdを設定する

同梱のサンプルは `/opt/ai-playground` と `ai-playground` ユーザーを前提にしています。

```bash
sudo cp /opt/ai-playground/deploy/systemd/ai-chat.service /etc/systemd/system/
sudo cp /opt/ai-playground/deploy/systemd/ai-chat-worker@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-chat.service
sudo systemctl enable --now ai-chat-worker@{1..4}.service
```

確認します。

```bash
systemctl --no-pager --full status ai-chat.service 'ai-chat-worker@1.service'
curl -H 'Host: chat.example.com' http://127.0.0.1:3111/api/version
```

別のユーザー名・配置先を使う場合は、コピー前に [deploy/systemd/README.md](deploy/systemd/README.md) の説明に従って `User`、`Group`、`WorkingDirectory`、`EnvironmentFile`、`ExecStart`、ログ出力先を変更します。

### 7. ApacheとHTTPSを設定する

先にDNSのA／AAAAレコードをサーバーへ向け、80／443番をファイアウォールで許可します。必要なモジュールを有効にし、Apacheの既定Webルートを使って証明書を取得します。

```bash
sudo a2enmod proxy proxy_http headers rewrite ssl reqtimeout env alias deflate
sudo certbot certonly --webroot -w /var/www/html -d 実際のドメイン
sudo cp /opt/ai-playground/deploy/apache/ai-playground.conf /etc/apache2/sites-available/chat.example.com.conf
sudo sed -i 's/chat\.example\.com/実際のドメイン/g' /etc/apache2/sites-available/chat.example.com.conf
sudo a2ensite chat.example.com.conf
sudo apache2ctl configtest
sudo systemctl reload apache2
sudo certbot renew --dry-run
```

サンプルには次の重要設定が含まれます。

- `ProxyPreserveHost On`：Host検証、OAuth、WebAuthnのRP IDを正しく保つ
- `RequestHeader set X-Forwarded-Proto "https"`：アプリへ元のスキームを通知
- `ProxyTimeout 660`：長時間の生成とワーカー上限に合わせる
- `SetEnv proxy-sendchunked 1`：SSEを逐次転送
- `RequestReadTimeout body=0` と `LimitRequestBody`：分割・大容量アップロード
- HSTS、`X-Frame-Options`、`X-Content-Type-Options`、`Referrer-Policy`

Cloudflare等のCDNを前段に置く場合も、オリジン証明書検証を有効にし、SSEをバッファしないでください。アプリの接続先3111番ポートはループバックだけで待ち受け、インターネットへ公開しないでください。

### 8. OAuthとTurnstileを設定する

Google OAuthを利用する場合、Google Cloud ConsoleのOAuth 2.0クライアントへ次を登録します。

- 承認済みJavaScript生成元：`https://chat.example.com`
- 承認済みリダイレクトURI：`https://chat.example.com/login/google/callback`

`.env` に `GOOGLE_CLIENT_ID` と `GOOGLE_CLIENT_SECRET` を設定します。Cloudflare Turnstileには実ドメインを登録し、`TURNSTILE_SITE_KEY` と `TURNSTILE_SECRET_KEY` を設定します。Turnstileは新規登録に必須です。Google OAuthは任意であり、ローカルのユーザー名／パスワード認証だけで運用する場合は省略できます。設定変更後はWebと全ワーカーを再起動してください。

### 9. 最終確認

```bash
curl -fsS https://chat.example.com/api/version
systemctl is-active ai-chat.service
systemctl is-active 'ai-chat-worker@1.service' 'ai-chat-worker@2.service' \
  'ai-chat-worker@3.service' 'ai-chat-worker@4.service'
redis-cli -n 10 ping
sudo mariadb -e 'SELECT 1'
sudo journalctl -u ai-chat.service -u 'ai-chat-worker@1.service' -n 100 --no-pager
```

ブラウザーでは、登録、ログイン、初期セットアップ、APIキー保存、通常チャット、ストリーム表示、ファイル添付、再読み込み後の履歴、ログアウトを確認してください。

## 初回セットアップ

1. Turnstileを設定し、`.env` の `PRIMARY_ADMIN_USERNAME` が空欄であることを確認します。
2. 公開範囲を運用者のIPへ一時制限した状態で、`https://chat.example.com/signup` から管理者にするユーザーを登録します。
3. 登録直後のセットアップ画面で既定モデルとAPIキーを設定します。
4. `.env` の `PRIMARY_ADMIN_USERNAME` に登録済みユーザー名を正確に設定し、Webと4ワーカーを再起動します。
5. 再ログインして管理者画面へアクセスできることを確認し、強固なパスワード、2FAまたはPasskey、管理者APIキーフォールバック方針を設定します。
6. 必要に応じてE2EEを有効にし、復旧に必要な情報を安全に保管してから公開制限を解除します。

`PRIMARY_ADMIN_USERNAME` と同じ名前は予約済みとして新規登録を拒否します。必ず「登録してから環境変数へ設定」の順で進めてください。

## 更新

更新前にDB、`instance/`、`.env`、`secret.key`をバックアップし、リリースの変更履歴を確認します。

```bash
cd /opt/ai-playground
sudo -u ai-playground git pull --ff-only
sudo -u ai-playground /opt/ai-playground/venv/bin/pip install -r requirements.txt
sudo systemctl restart ai-chat.service 'ai-chat-worker@1.service' \
  'ai-chat-worker@2.service' 'ai-chat-worker@3.service' 'ai-chat-worker@4.service'
curl -fsS https://chat.example.com/api/version
```

リポジトリ同梱の `scripts/restart_services.sh` はサービス名が上記と同じ環境で利用できます。Cloudflareを利用する場合、必要に応じて `scripts/purge_cloudflare_cache.sh` でキャッシュを消去できます。Cloudflare用の認証情報は `.env` に保存せず、権限を絞った専用ファイルまたは実行環境から渡すことも検討してください。

## バックアップと復元

最低限、次の対象を同じ時点のスナップショットとして暗号化バックアップします。

- MariaDBの `ai_chat_db`
- `/opt/ai-playground/instance/`
- `/opt/ai-playground/.env`
- `/opt/ai-playground/secret.key`

```bash
sudo mariadb-dump --single-transaction ai_chat_db | gzip > ai_chat_db.sql.gz
sudo tar --create --gzip --file ai-playground-files.tar.gz \
  /opt/ai-playground/instance /opt/ai-playground/.env /opt/ai-playground/secret.key
```

バックアップにはAPIキー、チャット、アップロード等の機密情報が含まれます。アクセス制御と暗号化を必須とし、復元テストを定期的に行ってください。`secret.key` のないDBバックアップから暗号化済み認証情報を復旧することはできません。

## 開発とテスト

ローカルでの最小起動は次のとおりです。Secure Cookieを利用するため、認証を含む完全な確認にはローカルHTTPSまたは本番相当のプロキシを推奨します。

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

別の端末でワーカーを起動します。

```bash
source venv/bin/activate
python worker.py
```

テストはプロジェクトルートで実行します。

```bash
venv/bin/python -m pytest -q
node --check static/js/progress_spinner.js
node --check static/js/chat_core.v*.js
```

画面のJavaScriptとCSSは、読みやすいソースと圧縮済みファイルの両方をリポジトリに含めています。通常の導入や更新では追加のビルドは不要です。ソースを編集したあとに圧縮ファイルを作り直す場合だけ、Node.js 20以降を用意して次を実行します。

```bash
./scripts/build_frontend.sh
```

アイコンサブセットを作り直す場合は `venv/bin/python scripts/build_icon_subset.py` を使います。

テスト構成は [tests/README.md](tests/README.md)、各ディレクトリの役割はそれぞれのREADMEを参照してください。

## トラブルシューティング

### 起動直後に500になる

`journalctl -u ai-chat.service` を確認し、`FLASK_SECRET_KEY`、`DATABASE_URL`、MariaDB接続、必須DB列を確認します。既存DBを更新した直後は、対象リリースのスキーマ移行手順を確認してください。

### ログイン状態が維持されない

HTTPSでアクセスしているか、プロキシがHostと `X-Forwarded-Proto: https` を渡しているか、`TRUSTED_HOSTS` に実ドメインがあるかを確認します。

### 送信後に応答が始まらない

Redisと4ワーカーの状態、ワーカーの購読キューを確認します。

```bash
redis-cli -n 10 ping
systemctl --no-pager status 'ai-chat-worker@1.service'
journalctl -u 'ai-chat-worker@1.service' -n 100 --no-pager
```

### ストリームがまとめて表示される／途中で切れる

リバースプロキシやCDNのレスポンスバッファリングを無効にし、タイムアウトをGunicorn／RQの上限以上へ設定します。

### Pythonコード実行が失敗する

`bwrap` と `prlimit` が実行ユーザーのPATHにあり、カーネルのuser namespaceが利用可能か確認します。サンドボックスを無効化して本番運用するのではなく、OS側の制限を解決してください。

### ファイルが大きいと413／408になる

`.env` の `UPLOAD_MAX_MB` とApacheの `LimitRequestBody`、`RequestReadTimeout`、CDNのアップロード上限をすべて確認します。最も小さい上限が適用されます。

## ドキュメント案内

- [MODELS.md](MODELS.md)：対応モデルと価格表示の扱い
- [CONTRIBUTING.md](CONTRIBUTING.md)：変更、テスト、Pull Request
- [SECURITY.md](SECURITY.md)：脆弱性報告と安全な運用
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：第三者ソフトウェアとライセンス
- [deploy/README.md](deploy/README.md)：配布用サーバー設定
- [templates/README.md](templates/README.md)：Jinjaテンプレート
- [static/README.md](static/README.md)：静的資産
- [tests/README.md](tests/README.md)：回帰テスト

## ライセンス

このリポジトリ固有のコードとドキュメントは、個別に別の表示があるものを除き、[MIT License](LICENSE) で提供します。著作権表示とライセンス本文を、複製または重要な部分とともに保持してください。

同梱・実行時取得する第三者コンポーネントには、それぞれ別のライセンスが適用されます。再配布やCDN資産の自己ホストを行う場合は、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) と各配布元のライセンスを確認し、必要な著作権表示・ライセンス本文・NOTICEを保持してください。AIプロバイダー、Google OAuth、Cloudflare等の外部サービス利用規約も別途適用されます。

Copyright (c) 2026 Minashin1120
