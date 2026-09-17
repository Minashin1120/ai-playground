# AI Playground Android

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `../deploy/ANDROID_CLIENT.md` | 通信仕様・アプリ操作・構築・署名・公開手順 | 開発・導入・障害対応 |
| `WEB_PARITY.md` | Web版との機能・デザイン差分、完成条件、実装順、進捗の正本 | Web版相当の機能・デザインを実装するとき |
| `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties` | Android Gradleプロジェクトのルート設定 | SDK・Kotlin・Gradleを更新するとき |
| `version.properties` | AndroidのversionCodeとversionName | 新しいAndroid版を配布するとき |
| `app/build.gradle.kts` | 固定applicationId・共有署名・依存関係 | アプリ構成・署名を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/MainActivity.kt` | ブラウザー起動・添付共有・Activityライフサイクル | 端末連携やファイル表示 |
| `app/src/main/java/com/minashin1120/aiplayground/data/AppUpdateChecker.kt` | GitHubのAndroid Release確認とバージョン比較 | 起動時のアプリ更新検出 |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppUpdateDialog.kt` | 新しいAndroid版の更新案内ダイアログ | 更新通知の文言・導線 |
| `app/src/main/java/com/minashin1120/aiplayground/ChatViewModel.kt` | 認証・履歴・送信・再接続・アップロード状態 | アプリ操作・通信フロー |
| `app/src/main/java/com/minashin1120/aiplayground/BatchNotifications.kt` | Batch完了通知チャンネルと安全な通知表示 | Batch通知を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/` | CookieなしHTTP、NDJSON、APIモデル、Keystore保存 | 認証・通信・保存 |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundScreen.kt`, `ChatComponents.kt` | Compose画面、履歴ドロワー／タブレット2ペイン、メッセージ操作、状態カード、入力欄 | 画面構造と操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/StartupSplash.kt`, `app/src/main/res/drawable/ic_playground_mark.xml` | 起動時のロゴズーム・画面リビールとアクセシビリティ対応 | 起動アニメーション・ロゴ素材を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsDialog.kt` | Web設定タブ相当のネイティブ設定（検索、一般／プロンプト／表示／データ／フィードバック／MCP／圧縮／セッション、秘密項目のWeb導線） | 設定画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/SlashCommands.kt` | 入力欄の `/` コマンド定義と解析 | スラッシュコマンドを追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AdvancedTools.kt` | Batch管理と高度な生成機能・安全なWeb導線 | 高度な機能メニューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/CanvasCodingPanels.kt` | Canvasプレビュー・編集とCoding履歴対象選択 | Canvas／Codingの編集操作を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImageMaskEditor.kt` | GPT-Image用の端末マスク描画とPNG化 | 画像マスクの描画・送信を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RealtimeStudio.kt` | Realtime音声・Lyria音楽のネイティブスタジオ | 音声／音楽セッションUIを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RichPasteDialog.kt` | クリップボードHTML／テキストの安全な取り込み | リッチ貼り付けの入力変換を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ModelPicker.kt` | WebのSelect Model構成に合わせた検索・提供元／用途／対応能力フィルター・選択状態 | モデル選択画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundDialog.kt` | 共通のネイティブモーダル枠、ヘッダー、フッター、キーボード余白 | 設定・Gems・ライブラリ・Batchの画面枠を調整するとき |
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
| `app/src/test/` | HTTP認証境界・ストリーム・データ解析・Markdown・数式のテスト | 通信・表示実装を変更するとき |
| `ci/bootstrap-keystore.sh` | Actions限定の初回鍵生成と保存 | 初回署名準備 |
| `ci/verify-keystore.sh`, `ci/verify-apk.sh` | 固定証明書とAPK署名の一致確認 | 署名・上書き更新を検証するとき |
| `ci/debug.keystore`, `ci/signing-fingerprint.txt` | Actionsが初回だけ保存する共有固定鍵と証明書のSHA-256 | 再利用のみ。削除・置換・再生成しない |
| `../.github/workflows/android.yml`, `../.github/workflows/release.yml` | CIと署名APKのリリース | GitHub Actionsの運用 |
