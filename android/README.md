# AI Playground Android

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `../deploy/ANDROID_CLIENT.md` | 通信仕様・アプリ操作・構築・署名・公開手順 | 開発・導入・障害対応 |
| `WEB_PARITY.md` | Web版との機能・デザイン差分、完成条件、実装順、進捗の正本 | Web版相当の機能・デザインを実装するとき |
| `ANDROID_ONLY.md` | Web版と意図的に異なる点だけを記録する正本（ここにない差分は追従漏れ） | Web版とAndroid版の差分を判断・変更するとき |
| `NATIVE_AUTH_PLAN.md` | ネイティブ認証・初期セットアップの承認済み実装計画と進捗 | 認証・2FA・パスキー・取り込みの作業を再開するとき |
| `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties` | Android Gradleプロジェクトのルート設定 | SDK・Kotlin・Gradleを更新するとき |
| `version.properties` | AndroidのversionCodeとversionName | 新しいAndroid版を配布するとき |
| `app/build.gradle.kts` | 固定applicationId・共有署名・依存関係 | アプリ構成・署名を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/MainActivity.kt` | Custom Tabs・HTTPS App Links・添付共有（送信・他アプリの共有シートからの受信）・APKインストール・Activityライフサイクル | ブラウザー認証からの復帰、ファイル表示、共有シート添付、アプリ更新 |
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
| `app/src/main/java/com/minashin1120/aiplayground/data/PlayIntegrityClient.kt` | Play Integrity Standard APIの事前準備・要求内容に結び付いた認証token取得 | Android認証の端末リスク信号 |
| `app/src/main/java/com/minashin1120/aiplayground/data/BrowserLoginPkce.kt` | ブラウザー経由ログインのPKCE（S256）verifier・challenge生成 | Google・MinashinのApp Link復帰を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundScreen.kt`, `ChatComponents.kt` | Compose画面の組み立て（ドロワー／タブレット2ペイン、各モーダルの開閉）、メッセージ操作、Coding差分（Live Code Changes）とBatch完了バナー、生成中の回答 | 画面構造と操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LiveAnswer.kt` | 生成待ちのスケルトン、Searching web／Search complete、Image Analysis、Python Execution、右下の通信スピナー | 生成中の表示を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ProgressText.kt` | Web `progress_spinner.js` の文言表と通信中の処理一覧（`PlaygroundApi.progress`） | 通信スピナーの文言や対象を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/Composer.kt` | Webのプロンプトバー（引用・Coding対象・編集バー、モデルボタン、Canvas／Coding／Batch、詳細チップ、添付プレビュー、固定プロンプト、Gem表示、`/`・`@` 候補、入力シェル、送信・停止、トークン見積もり） | 入力欄の見た目や操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/QuoteSelection.kt` | 会話の範囲選択に「Quote」を加える選択ツールバー | 引用の操作を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ComposerRules.kt` | Webの `toggleOptions()` 相当（モデル別のチップ表示・強制値・Thinking level／Effortの候補）、PromptCacheの提供元、トークン見積もりの文言 | モデル追加でチップの表示条件を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/CodingTargets.kt` | Coding Modeの対象候補の抽出、対象バーの文言、送信時の候補と上限 | Coding Modeの送信内容を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/Sidebar.kt` | Webの `#sidebar` 相当（ツールバー、検索、Gems、スレッド一覧、引っ張って更新、フッター） | サイドバーを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ChatHeader.kt` | スマートフォン用の上部バー（Webの `header.main-chrome-header`） | 上部バーを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebDialogs.kt` | Webの `confirm()`／`prompt()` 相当、チャット履歴・アルファ版・利用規約のモーダル、共通オーバーレイ | 確認ダイアログや小さなモーダルを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/LowBandwidth.kt` | Web相当の低速回線モードの判定・設定の切り替え・表示文言 | 低速回線モードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/StartupSplash.kt`, `app/src/main/res/drawable/ic_playground_mark.xml` | 起動時のロゴズーム・画面リビールとアクセシビリティ対応 | 起動アニメーション・ロゴ素材を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsDialog.kt` | Web設定モーダルの枠（見出し、検索と結果一覧、10タブ、キャンセル／保存） | 設定画面の枠や検索を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsTabs.kt` | 設定のタブ定義と一般・APIキー・プロンプト・表示タブのカード | これらのタブの項目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsTabsOther.kt` | データ・アカウント・セキュリティ・2要素認証・フィードバック・MCPタブのカードとAndroidカード | これらのタブの項目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsForm.kt` | 設定の編集中の値と保存時の送信内容 | 設定項目を追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsFields.kt` | 設定カード内の入力部品（チェックボックス、ラジオ、select、入力欄、小ボタン、スイッチ行） | 設定の部品の見た目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ColorPickerDialog.kt` | テーマ色の選択（Webの `input[type=color]` 相当） | 色選択を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/CompressionDialog.kt` | 「画像・圧縮詳細設定」モーダル（圧縮と画像・OCRの生成設定） | 圧縮設定を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsAccount.kt` | アカウント・セキュリティ・2要素認証タブのカードとデータタブの「アカウントデータ」カード | アカウント操作やデータ移行の画面を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ReauthDialog.kt` | 重要な操作の前の本人確認（ANDROID_ONLY.md）と、設定の操作を実行する `AccountOps` | 本人確認の流れを変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImportDialogs.kt` | インポート時の設定変更の確認と、容量超過時のファイル選択 | インポートの確認画面を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/AccountApi.kt` | 設定のアカウント・セッション・2FA・MCP・データ移行のAPI呼び出し | これらの通信を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/AccountTransfer.kt` | エクスポート・インポート・重複修復の進行（設定を閉じても続く） | データ移行の手順を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/SlashCommands.kt` | Webの `/` コマンド定義、候補の絞り込み、コマンド名の取り出し、on／off引数の解釈 | スラッシュコマンドを追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/AiSettings.kt` | `/settings` のAI設定変更（移動先と値の表示）、APIキー未設定時の提供元、Xリンクの判定 | `/settings` やAPIキー画面を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/BatchDialog.kt` | Webの `#batch-modal`（Batch処理：フィルタ、状態バッジ、開く・停止・履歴から削除、5秒ごとの更新） | Batch処理の画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ConfirmDialogs.kt` | Webの確認モーダル（外部ツール操作の確認、ローカルPython実行に切替） | 送信前・生成中の確認ダイアログを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/CanvasPanel.kt` | Webの `#canvas-panel`（スマートフォンは全画面、1024dp以上は右側。Blocks／Source、Copy、Clear、×） | Canvasの表示や操作を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/CanvasBlocks.kt` | Canvasモードのコードブロック抽出（Web `parseCanvasMarkdown`）、選択、見出し・状態の文言 | Canvasに出すブロックや文言を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImageMarkerEditor.kt` | Webの「画像編集」（マーカー、モザイク、トリミング、二本指の拡大、保存して反映） | 画像編集の画面や保存を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageMarker.kt` | 画像編集の計算（線の補間、モザイク範囲、トリミングの掴み位置、拡大時の位置制限、保存名） | 画像編集の操作の決まりを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MinimalOptions.kt` | ミニマル表示の上部モデルバー、＋の「オプション」ポップアップ、Thinkingのスライドバー | ミニマル表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/MessageExtras.kt` | 回答からのPython実行結果とMCP実行メモの取り出し、Batch状態カードの文言 | 回答本文の前処理やBatchカードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/UploadSheet.kt` | Webの「ファイルアップロード」画面（ボタン、Vision Model、進行状況、ファイルごとの画像編集・送信名・削除） | 添付の追加画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RealtimeStudio.kt` | Webの音声ドック（`#sts-panel`）と音声スタジオ、提供元別の音声設定、録音して `/sts` へ送る文字起こし系モデルの振り分け | 音声セッションUIを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LyriaStudio.kt` | Webの「Lyria RealTime Studio」（重み付きプロンプト、音楽設定、再生操作、チャットへ保存） | Lyriaスタジオを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RichPasteDialog.kt` | Webの「リッチ貼り付け」画面の枠（クリップボードHTMLのテキスト化、モデルへの指示、既定値の保存） | リッチ貼り付けを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ModelPicker.kt` | Webの「Select Model」（検索、固定タグバー、カテゴリ見出し、モデルカード、PromptCacheのロックバナー） | モデル選択画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/GemDialog.kt` | Webの `#gem-modal`（Gemの作成・編集、固定プロンプトの行）とTailwind風の入力欄 `TwInput` | Gemの編集画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LibraryDialog.kt` | Webの `#lib-modal`（検索、表示順、お気に入りのみ、複数選択の添付・ダウンロード・名前変更・使用チャット・削除、グリッド、使用チャットのモーダル） | ファイルライブラリを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/BranchManager.kt` | Webの `#branch-modal`（ツリー、ブランチ詳細、モデル別内訳、名前・固定、切替、削除、凡例と合計） | ブランチ管理を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/Library.kt` | ライブラリの並び順、音声・動画の判定、モデルの音声・動画入力対応（Web `getModelMediaSupport`） | ライブラリの添付規則を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/BranchTree.kt` | ブランチツリーの配置計算とパス累計トークン | ブランチ管理の表示計算を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ChatInstructionsDialog.kt` | Webの `#thread-modal`（Chat Instructions：チャット専用の指示、全体プロンプトの参照、ユーザーシステムプロンプト、自動注入の簡易版） | SysPromptの⚙から開く画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundDialog.kt` | Webモーダル（オーバーレイ、ぼかし、パネル、見出し、下部）の共通枠 | 設定・Gems・ライブラリ・Batchの画面枠を調整するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundMotion.kt` | 共通の時間・イージング、アニメーション削除設定への追従、モーダルの開閉・予測型「戻る」、一覧・登場・押下の動きの部品 | 画面の動き・遷移を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppChangelogDialog.kt` | Android版更新履歴のネイティブMarkdown表示、読み込み・再試行・戻る操作 | 更新履歴画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/GenerationOptionsPanel.kt` | Webの生成パネル（`composer_gen_image.html`／`composer_gen_media.html`）の描画、Lyria RealTimeの案内バー、入力制限の注記 | 詳細設定の表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/GenerationOptions.kt` | Webの生成パネルの項目とモデル別の表示規則（`p07` の `update*Ui` 相当）、値検証、送信対象の限定。File／SysPrompt／URLs／Thinking／Effort／SafetyはComposerのチップが持つ | 画像・動画・OCR・TTSの設定を追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/RealtimeModels.kt` | Realtime／Lyriaのセッション状態 | 音声・音楽ストリームの状態表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/WebModelCatalog.kt`, `app/src/main/assets/web-model-catalog.json` | Webの表示名・説明・価格・タグ・追加順をネイティブAPIのモデル一覧に合成 | モデル情報の表示差を確認するとき |
| `ci/sync-web-catalog.mjs` | Webモデル定義からAndroid表示用JSONを生成・`--check`で一致検証 | Webのモデル定義変更をAndroidへ同期するとき |
| `ci/audit-web-strings.py` | Androidの日本語文字列のうち、Webのテンプレート・JS・サーバー文言にないものを一覧表示 | 文言をWebと突き合わせるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundTheme.kt` | Webトークンから作るMaterial配色、Tailwindの文字サイズ、形状、レイアウト寸法、ライト／ダークの判定 | 色・テーマ・ブランドを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebTokens.kt` | Webの `:root` トークン（ダーク／ライト）、テーマ色の段階計算、Tailwind v3の色、ライトテーマでの置き換え | Webの色をそのまま使うとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebFonts.kt`, `app/src/main/res/values/font_certs.xml` | Google Fonts経由のNoto Sans JP／JetBrains Mono | フォントを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebComponents.kt` | Web再現の共通部品（Font Awesomeアイコン、トグル、チェックボックス、select、チップ、設定カード、ボタン、モーダルの見出し・下部） | Webと同じ見た目の部品を使うとき |
| `app/src/main/res/drawable/fa_*.xml`, `ci/sync-web-icons.py` | WebのFont Awesomeサブセットから生成したアイコン（手で編集しない。`--check` で一致確認） | Webで使うアイコンが増減したとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownText.kt` | Webの `marked`（GFM・`breaks`）と `pre-wrap` 表示を再現するMarkdown（リスト、引用、表、コード、`chat_error`、色分け、インラインコード）、回答中の生HTML（`<details>`・表を含む）とSVG | メッセージ本文表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MessageBubble.kt` | Webの `.message-bubble`（吹き出し、タップで出る操作ボタン、引用、Thinking、添付グリッド、分岐切替、フッター）、合計トークン帯、トークン詳細・暗号化モーダル、Welcome、一番下へ | メッセージの見た目を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LatexText.kt` | LaTeXをWebViewなしで読めるネイティブ表示へ変換 | 数式表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RemoteImage.kt` | 同一originの添付画像をBearer付きで取得し、キャッシュしてプレビュー | 添付・画像プレビューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/FileViewer.kt` | Webの画像ビューアー（件数・前後移動・Download／Copy URL／Reuse／Close）とファイル表示パネル（テキスト・PDF・音声・動画） | ファイルや画像の表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ThreadPdf.kt` | スレッドのネイティブA4 PDF出力（WebView不使用） | PDF出力を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageCompression.kt` | Web相当の画像圧縮設定と、アップロード前の縮小・再エンコード | 画像圧縮を編集するとき |
| `app/src/main/res/` | アイコン、HTTPS設定、バックアップ除外、添付共有範囲 | Androidリソース・保護設定 |
| `ci/changelogs/` | Android版のバージョンごとの公開更新履歴Markdown | アプリ版の更新履歴を追加・修正するとき |
| `app/src/test/` | HTTP認証境界・ストリーム・データ解析・Markdown・数式のテスト | 通信・表示実装を変更するとき |
| `app/src/test/java/com/minashin1120/aiplayground/ui/*ScreenshotTest.kt` | Roborazziによる画面のスクリーンショット（Actionsの `android-reports` に保存） | Webとの見た目を比べるとき |
| `THIRD_PARTY_NOTICES.md` | Android版に含める第三者素材の表示 | アイコン・フォントなどを追加するとき |
| `ci/bootstrap-keystore.sh` | Actions限定の初回鍵生成と保存 | 初回署名準備 |
| `ci/verify-keystore.sh`, `ci/verify-apk.sh` | 固定証明書とAPK署名の一致確認 | 署名・上書き更新を検証するとき |
| `ci/debug.keystore`, `ci/signing-fingerprint.txt` | Actionsが初回だけ保存する共有固定鍵と証明書のSHA-256 | 再利用のみ。削除・置換・再生成しない |
| `../.github/workflows/android.yml`, `../.github/workflows/release.yml` | CIと署名APKのリリース | GitHub Actionsの運用 |
