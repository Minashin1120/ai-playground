# Androidネイティブ認証・アカウント作成・初期セットアップ実装計画

> 2026-09-21 の Codex セッション（`01a0c4ae-656d-7760-9c63-ad4ee621a65e`）で承認された実装計画の正本。
> 実装の進捗はこの文書のチェックリストで管理する。

## 概要

Androidアプリに、アカウント登録・ログイン・2FA・Passkey管理・初期セットアップを追加する。

既存の確認コード＋ブラウザー連携は削除せず、「従来のブラウザー連携」として非推奨の別導線に残す。

## 実装内容

- Androidに認証画面を追加
  - ユーザー名・パスワード登録
  - ユーザー名・パスワードログイン
- Google（Credential Manager）/ Minashinログイン
  - Passkeyログイン
  - TOTP / WebAuthn 2FA
  - Passkey登録・削除
  - TOTP登録・無効化
  - 既定2FA方式、Passkey専用ログイン、Googleログイン時の2FA省略設定
- OAuthは埋め込みWebViewを使わずCustom Tabsを使用する。
- OAuth後はHTTPS App LinksでAndroidへ復帰し、URLにはアクセストークンを含めず、一回限りの認可コードを交換する。
- PasskeyはAndroid Credential Managerを使用し、サーバー側でAndroidアプリのRP ID・署名証明書Originだけを許可する。
- Play Store外配布を前提に、Play Integrityは補助的なリスク信号として使う。Playライセンスは必須にせず、判定不能・高リスク・試行集中時だけTurnstileへフォールバックする。([Play Integrity公式](https://developer.android.com/google/play/integrity/setup)、[Play Store外配布の公式仕様](https://developer.android.com/google/play/integrity/classic))
- 既存のブラウザー連携フローと旧Android版との互換性を維持する。

## サーバーAPI・認証境界

新しいAndroid認証用ルート部品を追加し、既存の共有認証ヘルパーを再利用する。

追加するAPI群:

- `/api/mobile/v1/auth/signup`
- `/api/mobile/v1/auth/login`
- `/api/mobile/v1/auth/oauth/start`
- `/api/mobile/v1/auth/oauth/exchange`
- `/api/mobile/v1/auth/passkey/options`
- `/api/mobile/v1/auth/passkey/verify`
- `/api/mobile/v1/auth/2fa/totp`
- `/api/mobile/v1/auth/2fa/webauthn/options`
- `/api/mobile/v1/auth/2fa/webauthn/verify`
- `/api/mobile/v1/security/totp/*`
- `/api/mobile/v1/security/passkeys/*`
- `/api/mobile/v1/setup`
- `/api/mobile/v1/setup/import/*`

認証成功時は既存の `UserSession` とAndroid用Bearerトークンを発行する。`is_setup_completed` が false の間は、セットアップAPIだけを許可し、通常のチャットAPIは `setup_required` で拒否する。

認証トランザクション、OAuth state、PKCE、WebAuthn challenge、Turnstile proofはRedisに短時間だけ保存する。DBスキーマは原則変更しない。

## 初期セットアップ

登録・初回ログイン後に、Android内で以下のウィザードを表示する。

1. 歓迎・スタートアップ画面
2. 任意のアカウントZIPインポート
3. 既定モデル、APIキー、Vertex設定、E2EE設定

対象APIキーはOpenAI、Gemini、Anthropic、DeepSeek、Kimi、Mistral、xAI、Google系とする。ZIPインポートでは、進捗表示、分割アップロード、再試行、キャンセル、設定確認、容量超過時のファイル選択に対応する。

Web版の初期セットアップ保存処理も共通化し、現在保存漏れのあるAnthropic APIキーを正しく保存する。

## Android側の変更

- `AuthScreen`、`SetupWizard`、認証状態管理を追加
- `PlaygroundApi` に認証・セットアップ・インポートAPIを追加
- `ChatViewModel` の未認証、2FA待機、OAuth復帰、セットアップ待機状態を拡張
- `TokenStore` にOAuth state・PKCE verifierなどの一時情報を安全に保存
- `Credential Manager`依存を固定バージョンで追加
- `AndroidManifest.xml` にHTTPS App Linksを追加
- 既存の `PairingScreen` は非推奨の「従来のブラウザー連携」として残す

## テスト・受け入れ条件

- 新規登録 → 歓迎画面 → ZIPインポート → APIキー・モデル・E2EE設定 → チャット開始
- パスワードログイン、Google、Minashin、Passkeyの各ログイン
- TOTP / WebAuthn 2FA
- Passkey・TOTPの登録、削除、無効化、既定方式変更
- OAuth state、PKCE、認可コード再利用、異常なApp Linkの拒否
- Turnstileフォールバック、レート制限、Play Integrity判定不能時の救済
- APIキー・パスワード・BearerトークンがログやURLに残らないこと
- セットアップ途中のアプリ終了・再起動からの復帰
- ZIPインポートの再試行、キャンセル、容量超過、設定確認
- 既存Webログインとブラウザー連携の回帰
- 旧Android版が従来どおり連携できること

## 文書・公開

`app/server/README.md`、`app/deploy/ANDROID_CLIENT.md`、`app/android/README.md`、`app/android/WEB_PARITY.md`を更新する。

Google / MinashinのOAuthリダイレクトURI、HTTPS App Linksの`assetlinks.json`、Play IntegrityのCloud Project設定を本番環境へ追加する。

WebとAndroidの両方を変更するため、Webのバージョン準備・公開処理で1つのcommitにまとめ、Androidのビルド・テスト・lint・署名確認はGitHub Actionsのみで実行する。AndroidのローカルGradle実行は行わない。

## 前提

- メールアドレス確認は追加せず、現行Web版と同じくユーザー名・パスワードで登録する。
- OAuth認証操作のみシステムブラウザーを使用する。
- 既存ブラウザー連携は非推奨として残す。
- 認証トークンは既存同様30日有効のBearer方式とし、refresh tokenは追加しない。
- 機密データ用ディレクトリにはエージェントからアクセスしない。

## 実装進捗

| 項目 | 状態 |
|---|---|
| サーバー: signup / login / TOTP / exchange / Google IDトークン・Google／Minashin OAuth / assetlinks | 実装済み |
| サーバー: パスキー options・verify、WebAuthn 2FA options・verify | 実装済み |
| サーバー: セキュリティ管理（TOTP setup/enable/disable、パスキー登録/削除、既定2FA方式、パスキーのみログイン、Googleログイン時の2FA省略） | 実装済み |
| サーバー: アカウントZIPのチャンク取り込み（既存WebルートをBearer許可） | 実装済み |
| Android: ユーザー名／パスワード登録・ログイン、Google Credential Manager、TOTP | 実装済み |
| Android: パスキーログイン、WebAuthn 2FA、パスキー登録/削除、TOTP登録/無効化、2FA設定 | 実装済み |
| Android: 初回セットアップ（モデル／APIキー／Vertex AI／E2EE） | 実装済み |
| Android: アカウントZIP取り込み（進捗・キャンセル・設定変更確認） | 実装済み |
| Play Integrity Standard API と Turnstile 自動フォールバック | 実装済み（サーバー回帰テスト追加。GitHub Actions・Play services搭載端末での確認待ち） |
| 実機での Credential Manager 動作確認 | 未確認（GitHub Actionsでは検証不可） |
| Play Integrity 実機動作とTurnstileからの認証再開 | 未確認（GitHub Actionsでは検証不可） |

計画からの差分:

- Googleの通常ログインは Credential Manager でIDトークンを取得し `/api/mobile/v1/auth/google` へ送信する。ブラウザー方式の `/android/auth/google/start`・`/android/auth/minashin/start` は互換用に残している。
- 取り込みは `/api/mobile/v1/setup/import/*` を新設せず、Webの `/api/account/import/*` をBearer許可リストへ追加して再利用している。
- パスキーは `rp_id` を接続先ホスト、Android originを `android:apk-key-hash:`（`ANDROID_APP_LINK_SHA256` から導出）として検証する。
