# AI Playground Android

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `../deploy/ANDROID_CLIENT.md` | 通信仕様・アプリ操作・構築・署名・公開手順 | 開発・導入・障害対応 |
| `WEB_PARITY.md` | Web版との機能・デザイン差分、完成条件、実装順、進捗の正本 | Web版相当の機能・デザインを実装するとき |
| `NATIVE_AUTH_PLAN.md` | ネイティブ認証・初期セットアップの承認済み実装計画と進捗 | 認証・2FA・パスキー・取り込みの作業を再開するとき |
| `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties` | Android Gradleプロジェクトのルート設定 | SDK・Kotlin・Gradleを更新するとき |
| `version.properties` | AndroidのversionCodeとversionName | 新しいAndroid版を配布するとき |
| `app/build.gradle.kts` | 固定applicationId・共有署名・依存関係 | アプリ構成・署名を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/MainActivity.kt` | Custom Tabs・HTTPS App Links・添付共有・APKインストール・Activityライフサイクル | ブラウザー認証からの復帰、ファイル表示、アプリ更新 |
| `app/src/main/java/com/minashin1120/aiplayground/data/GoogleAuthClient.kt` | Credential ManagerによるGoogle IDトークン取得 | Googleのネイティブログインを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppUpdateViewModel.kt` | 更新検出・ダウンロード状態・キャンセル・再試行 | アプリ更新フローを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppChangelogViewModel.kt` | Android版更新履歴Markdownの取得状態・再試行 | アプリ版更新履歴の取得を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/AppUpdateChecker.kt`, `AppUpdateDownloader.kt` | GitHub Release資産の確認、APK取得、SHA-256検証 | 更新元・資産名・ダウンロードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/OfflineCacheStore.kt` | アカウント別の暗号化オフライン履歴・ファイル保存、同期設定、カテゴリ削除 | オフライン閲覧・端末保存・キャッシュ管理を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppUpdateDialog.kt` | APK更新の確認、進捗、設定案内、再試行ダイアログ | 更新UI・文言・導線 |
| `app/src/main/java/com/minashin1120/aiplayground/ChatViewModel.kt` | ネイティブ認証・パスキー／TOTP 2FA・セキュリティ管理・初回セットアップ／ZIP取り込み・旧端末連携・履歴・送信・再接続・アップロード状態 | アプリ操作・通信フロー |
| `app/src/main/java/com/minashin1120/aiplayground/data/ConnectionStatus.kt` | Web相当のハートビート接続状態・HTTP障害分類・監視間隔 | 接続状態・復帰表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/BatchNotifications.kt` | Batch完了通知チャンネルと安全な通知表示 | Batch通知を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/BubbleNotifications.kt` | ユーザー操作で作成するチャットバブル通知と会話ショートカット | Androidバブル・通知・ショートカットを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/` | CookieなしHTTP、NDJSON、APIモデル、Keystore保存、Credential Managerのパスキー手順（`PasskeyClient.kt`） | 認証・通信・保存 |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundScreen.kt`, `ChatComponents.kt` | Compose画面、履歴ドロワー／タブレット2ペイン、メッセージ操作、状態カード、入力欄 | 画面構造と操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/StartupSplash.kt`, `app/src/main/res/drawable/ic_playground_mark.xml` | 起動時のロゴズーム・画面リビールとアクセシビリティ対応 | 起動アニメーション・ロゴ素材を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsDialog.kt` | Web設定タブ相当のネイティブ設定（検索、一般／プロンプト／表示／データ／フィードバック／MCP／圧縮／セッション、秘密項目のWeb導線） | 設定画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/SlashCommands.kt` | 入力欄の `/` コマンド定義と解析 | スラッシュコマンドを追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AdvancedTools.kt` | Batch管理と高度な生成機能・安全なWeb導線 | 高度な機能メニューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/CanvasCodingPanels.kt` | Canvasプレビュー・編集とCoding履歴対象選択 | Canvas／Codingの編集操作を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImageMaskEditor.kt` | GPT-Image用の端末マスク描画とPNG化 | 画像マスクの描画・送信を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RealtimeStudio.kt` | Realtime音声（Gemini 3.8 Live / Extended Thinkingを含む）・Lyria音楽のネイティブスタジオ | 音声／音楽セッションUIを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RichPasteDialog.kt` | クリップボードHTML／テキストの安全な取り込み | リッチ貼り付けの入力変換を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ModelPicker.kt` | WebのSelect Model構成に合わせた検索・提供元／用途／対応能力フィルター・選択状態 | モデル選択画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundDialog.kt` | 共通のネイティブモーダル枠、ヘッダー、フッター、キーボード余白 | 設定・Gems・ライブラリ・Batchの画面枠を調整するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppChangelogDialog.kt` | Android版更新履歴のネイティブMarkdown表示、読み込み・再試行・戻る操作 | 更新履歴画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/GenerationOptionsPanel.kt` | モデル別生成設定のネイティブ入力パネル | 詳細設定の表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/GenerationOptions.kt` | モデル別の既存APIオプション、値検証、送信対象の限定 | 画像・動画・OCR・TTS・Thinking設定を追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/RealtimeModels.kt` | Realtime／Lyriaのセッション状態 | 音声・音楽ストリームの状態表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/WebModelCatalog.kt`, `app/src/main/assets/web-model-catalog.json` | Webの表示名・説明・価格・タグ・追加順をネイティブAPIのモデル一覧に合成 | モデル情報の表示差を確認するとき |
| `ci/sync-web-catalog.mjs` | Webモデル定義からAndroid表示用JSONを生成・`--check`で一致検証 | Webのモデル定義変更をAndroidへ同期するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundTheme.kt` | Web版トークンに対応する配色・タイポグラフィ・形状・レイアウト寸法 | 色・テーマ・ブランドを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownText.kt` | 安全なMarkdown、リンク、引用、数式、表、画像、折り畳み・コピー対応コード表示 | メッセージ本文表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LatexText.kt` | LaTeXをWebViewなしで読めるネイティブ表示へ変換 | 数式表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RemoteImage.kt` | 同一originの添付画像をBearer付きで取得し、キャッシュしてプレビュー | 添付・画像プレビューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/FileViewer.kt` | ファイルライブラリとチャット添付のアプリ内プレビュー（画像・テキスト・PDF・音声・動画） | ファイルの開き方・プレビュー種別を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ThreadPdf.kt` | スレッドのネイティブA4 PDF出力（WebView不使用） | PDF出力を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageCompression.kt` | Web相当の画像圧縮設定と、アップロード前の縮小・再エンコード | 画像圧縮を編集するとき |
| `app/src/main/res/` | アイコン、HTTPS設定、バックアップ除外、添付共有範囲 | Androidリソース・保護設定 |
| `ci/changelogs/` | Android版のバージョンごとの公開更新履歴Markdown | アプリ版の更新履歴を追加・修正するとき |
| `app/src/test/` | HTTP認証境界・ストリーム・データ解析・Markdown・数式のテスト | 通信・表示実装を変更するとき |
| `ci/bootstrap-keystore.sh` | Actions限定の初回鍵生成と保存 | 初回署名準備 |
| `ci/verify-keystore.sh`, `ci/verify-apk.sh` | 固定証明書とAPK署名の一致確認 | 署名・上書き更新を検証するとき |
| `ci/debug.keystore`, `ci/signing-fingerprint.txt` | Actionsが初回だけ保存する共有固定鍵と証明書のSHA-256 | 再利用のみ。削除・置換・再生成しない |
| `../.github/workflows/android.yml`, `../.github/workflows/release.yml` | CIと署名APKのリリース | GitHub Actionsの運用 |
