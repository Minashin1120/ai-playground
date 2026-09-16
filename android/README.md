# AI Playground Android

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `../deploy/ANDROID_CLIENT.md` | 通信仕様・アプリ操作・構築・署名・公開手順 | 開発・導入・障害対応 |
| `WEB_PARITY.md` | Web版との機能・デザイン差分、完成条件、実装順、進捗の正本 | Web版相当の機能・デザインを実装するとき |
| `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties` | Android Gradleプロジェクトのルート設定 | SDK・Kotlin・Gradleを更新するとき |
| `version.properties` | AndroidのversionCodeとversionName | 新しいAndroid版を配布するとき |
| `app/build.gradle.kts` | 固定applicationId・共有署名・依存関係 | アプリ構成・署名を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/MainActivity.kt` | ブラウザー起動・添付共有・Activityライフサイクル | 端末連携やファイル表示 |
| `app/src/main/java/com/minashin1120/aiplayground/ChatViewModel.kt` | 認証・履歴・送信・再接続・アップロード状態 | アプリ操作・通信フロー |
| `app/src/main/java/com/minashin1120/aiplayground/BatchNotifications.kt` | Batch完了通知チャンネルと安全な通知表示 | Batch通知を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/` | CookieなしHTTP、NDJSON、APIモデル、Keystore保存 | 認証・通信・保存 |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundScreen.kt`, `ChatComponents.kt` | Compose画面、履歴ドロワー／タブレット2ペイン、メッセージ操作、状態カード、入力欄 | 画面構造と操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AdvancedTools.kt` | Batch管理と高度な生成機能・安全なWeb導線 | 高度な機能メニューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundTheme.kt` | Web版トークンに対応する配色・タイポグラフィ・形状・レイアウト寸法 | 色・テーマ・ブランドを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownText.kt` | 安全なMarkdown、リンク、引用、数式、表、画像、折り畳み・コピー対応コード表示 | メッセージ本文表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LatexText.kt` | LaTeXをWebViewなしで読めるネイティブ表示へ変換 | 数式表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RemoteImage.kt` | 同一originの添付画像をBearer付きで取得し、キャッシュしてプレビュー | 添付・画像プレビューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ThreadPdf.kt` | スレッドのネイティブA4 PDF出力（WebView不使用） | PDF出力を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageCompression.kt` | Web相当の画像圧縮設定と、アップロード前の縮小・再エンコード | 画像圧縮を編集するとき |
| `app/src/main/res/` | アイコン、HTTPS設定、バックアップ除外、添付共有範囲 | Androidリソース・保護設定 |
| `app/src/test/` | HTTP認証境界・ストリーム・データ解析・Markdown・数式のテスト | 通信・表示実装を変更するとき |
| `ci/bootstrap-keystore.sh` | Actions限定の初回鍵生成と保存 | 初回署名準備 |
| `ci/verify-keystore.sh`, `ci/verify-apk.sh` | 固定証明書とAPK署名の一致確認 | 署名・上書き更新を検証するとき |
| `ci/debug.keystore`, `ci/signing-fingerprint.txt` | Actionsが初回だけ保存する共有固定鍵と証明書のSHA-256 | 再利用のみ。削除・置換・再生成しない |
| `../.github/workflows/android.yml`, `../.github/workflows/release.yml` | CIと署名APKのリリース | GitHub Actionsの運用 |
