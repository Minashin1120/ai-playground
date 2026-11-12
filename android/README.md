# AI Playground Android

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `docs/CONNECTION_MODES.md` | 利用者向けの接続モードの説明（接続先の指定、サーバー不使用モード、アカウントなしで開始、同期、セルフホスト側の設定） | 公開向けの説明を確認・更新するとき |
| `docs/DIAGNOSTICS.md` | 利用者向けの診断ログの説明（対象、記録する内容・しない内容、端末とサーバーでの保存、止め方） | 診断ログの記録項目・送信・保存を確認・変更するとき |
| `docs/NETWORK_TRANSPORT.md` | アプリ・APK・事業者・サーバーの通信経路、待ち行列と配信の調査結果・修正・観測の限界 | 複数の通信が遅延・失敗する問題を調査するとき |
| `docs/VIDEO_ATTACHMENT_INCIDENT.md` | 2026年10月の、動画を添付した送信で回答が止まる3つの問題の症状・条件・原因・修正 | 同じ症状の調査、または説明している仕組み（Files API、端末の暗号化方式、アプリを離れたときの受信）を変えるとき |
| `../deploy/ANDROID_CLIENT.md` | 通信仕様・アプリ操作・構築・署名・公開手順 | 開発・導入・障害対応 |
| `WEB_PARITY.md` | Web版との機能・デザイン差分、完成条件、実装順、進捗の正本 | Web版相当の機能・デザインを実装するとき |
| `ANDROID_ONLY.md` | Web版と意図的に異なる点だけを記録する正本（ここにない差分は追従漏れ） | Web版とAndroid版の差分を判断・変更するとき |
| `NATIVE_AUTH_PLAN.md` | ネイティブ認証・初期セットアップの承認済み実装計画と進捗 | 認証・2FA・パスキー・取り込みの作業を再開するとき |
| `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties` | Android Gradleプロジェクトのルート設定 | SDK・Kotlin・Gradleを更新するとき |
| `version.properties` | AndroidのversionCodeとversionName | 新しいAndroid版を配布するとき |
| `app/build.gradle.kts` | 固定applicationId・共有署名・依存関係 | アプリ構成・署名を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/MainActivity.kt` | Custom Tabs・HTTPS App Links・添付共有（送信・他アプリの共有シートからの受信（ファイルは添付、テキストは入力欄へ追加）、「画像を分割」と「メッセージとしてプレビュー」の共有先）・APKインストール・Activityライフサイクル | ブラウザー認証からの復帰、ファイル表示、共有シート添付、アプリ更新 |
| `app/src/main/java/com/minashin1120/aiplayground/ProcessTextActivity.kt` | 他アプリのテキスト選択メニュー「プロンプトとして添付」（`ACTION_PROCESS_TEXT`）。選択した文字を共有シートと同じ経路で入力欄へ追加し、選択元は置き換えない | テキスト選択メニューからの入力を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/TranslateOverlayActivity.kt`, `ui/SelectionTranslationDialog.kt`, `data/SelectionTranslator.kt`, `TranslationSetting.kt` | 他アプリのテキスト選択メニューと共有シートの「AIで翻訳」（`ACTION_PROCESS_TEXT`、テキストと画像の `ACTION_SEND`、画像の `ACTION_SEND_MULTIPLE`）。画像は写っている文字を翻訳し、プロンプトバーと同じ圧縮を翻訳用の別の値で行う。選択元・共有元のアプリを背後に残す透過画面で、設定の「Android」カードで選んだモデル・言語・Thinking・Effortの翻訳を表示する。サーバー経由は一時チャットを作って回答後に削除し、サーバー不使用モードとアカウントなしは端末から直接送る | テキスト選択からの翻訳を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/GoogleAuthClient.kt` | Credential ManagerによるGoogle IDトークン取得 | Googleのネイティブログインを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppUpdateViewModel.kt` | 更新検出と画面への状態中継・インストール段階 | アプリ更新フローを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppUpdateDownloadManager.kt` | プロセス全体で1つの更新状態・APKダウンロード・キャンセル・ダイアログの非表示／再表示 | 更新ダウンロードの状態遷移を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppUpdateDownloadService.kt` | ダウンロード中のフォアグラウンドサービス、進捗・完了通知、更新通知チャンネル | アプリ外でのダウンロード継続・通知を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/GenerationService.kt`, `app/src/main/java/com/minashin1120/aiplayground/data/local/GenerationTracker.kt` | 回答の生成・受信中のフォアグラウンドサービス（通知とCPUの保持）と通知チャンネル、実行中の生成の数え上げ（サーバー不使用モードで回答の保存後まで保護する画像・動画のmodeは `needsKeepAlive` で判定） | 生成中のアプリ保護・アプリを離れたときの継続・通知を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AppChangelogViewModel.kt` | Android版更新履歴Markdownの取得状態・再試行 | アプリ版更新履歴の取得を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/NetworkTrace.kt` | サーバー・直接通信・更新・DNS補助の通信段階、接続再利用、失敗分類、同時通信数の機密情報を含まないログ | 通信停止箇所の診断を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/NetworkDns.kt` | 端末DNS失敗時のCloudflare／Google DNS over HTTPS補助、対象ドメイン制限、前面画面への通知配信と間隔、DNSエラー文言 | 名前解決の復旧・補助通知を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/AppUpdateChecker.kt`, `AppUpdateDownloader.kt` | GitHub Release資産の確認、APK取得、SHA-256検証 | 更新元・資産名・ダウンロードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/PendingFeedbackStore.kt`, `PendingFeedbackSender.kt` | 接続できないときのフィードバックと選択した添付の暗号化保存、送信済み段階の保存、自動再送 | フィードバックの送信保留・再送を変更するとき |
| `docs/PENDING_FEEDBACK.md` | フィードバックの端末保存、再送のタイミング、添付とログの扱い | フィードバック送信保留の動作を確認するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/OfflineCacheStore.kt` | アカウント別の暗号化オフライン履歴・ファイル保存、同期設定、カテゴリ削除 | オフライン閲覧・端末保存・キャッシュ管理を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppUpdateDialog.kt` | APK更新の確認、進捗、設定案内、再試行ダイアログと、非表示中（ダウンロード中・インストール待ち）の画面上部バー | 更新UI・文言・導線 |
| `app/src/main/java/com/minashin1120/aiplayground/ChatViewModel.kt` | 画面の入口と状態：アカウント・履歴・スレッド・アップロード状態、各協作者への薄い公開入口（協作者の組み立ては `ChatGraph.kt`） | アプリ操作・通信フロー |
| `app/src/main/java/com/minashin1120/aiplayground/ChatGraph.kt` | 協作者の組み立てと配線：状態・保存先・各協作者（接続・認証・送信・添付・音声・設定）を生成し、終了時に解放する | 協作者を追加・並び替えるとき、起動時の配線を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAttachmentFiles.kt` | 添付の読み込み・プレビュー・保存：端末内ファイル、サーバー（同じ接続先）、外部https画像、オフラインキャッシュ | 添付の表示・ダウンロードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAttachmentUpload.kt` | 添付アップロードのキュー・転送（全体／分割）・進捗・キャンセル・リセット、マスク画像・編集済み添付のアップロード | 添付の送信やマスク・編集画像の反映を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatImageSplit.kt` | 画像分割ダイアログの状態と、分割結果の添付・端末保存の流れ | 画像分割を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatMicRecorder.kt` | チャット入力の音声録音と文字起こし（レベル表示・`/transcribe` 送信） | マイク入力や文字起こしを変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatRealtimeSession.kt` | Web `RealtimeVoiceSession`：リアルタイム音声セッションの開始・停止、PCM音声の送受信、文字起こしと応答の反映、`/sts` の一回録音（文字起こし系モデル） | リアルタイム音声・`/sts` を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatLyriaStudio.kt` | Web `LyriaRealtimeStudio`：重み付きプロンプト、音楽設定、再生・一時停止・停止・コンテキストリセット、音声のチャット保存 | Lyria RealTime スタジオを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAudioOutput.kt` | リアルタイム音声と Lyria が共有するPCM再生（`AudioTrack`）の作成 | 音声出力の形式・バッファを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSyncRunner.kt` | 端末モードの未送信分の同期：回答後・復帰時・接続回復時の予約、失敗時の再試行（30秒から10分）、「今すぐ同期」、進捗と結果の表示 | 同期のタイミング・再試行・表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatState.kt` | チャット画面の状態（`ChatState`・画面遷移種別・取り込み設定の差分） | 状態の項目や初期値、画面遷移の種別を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatDeviceAnswer.kt` | 端末で生成する回答の判定・停止・送信可否（オフライン時を含む） | 端末回答の停止や送信の端末モード判定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatLibrary.kt` | Web `#files-modal`：ファイル一覧・検索・並び順・お気に入り・名前変更・削除・添付（再アップロードなし）・保存・使用チャットの表示 | ライブラリの一覧・操作・ダウンロードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatFeedback.kt` | Web `#feedback-modal`・`#fb-admin-list`：フィードバックの一覧・状態変更・返信・削除、端末保存しての送信と再送、チャットのコピーと添付 | フィードバックの送信・返信・再送を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatMessageLoader.kt` | Web `loadMessages`：履歴の取得（サーバー／オフラインキャッシュ）・過去分の追加・送信中を保つ再読込・停止したジョブの再結合抑止 | 履歴の読み込みや再読込の条件を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatComposer.kt` | 送信準備：入力・引用、モデル選択と各モデルのオプション規則、PromptCache、Coding対象、添付名 | プロンプトバーの状態やモデル切替・オプションを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatPreferences.kt` | 全般設定：設定の読込・保存、リッチ貼り付けプロンプト、ストレージ使用量、MCPサーバー、オフライン履歴キャッシュ（方式・モバイル通信・全件同期・削除） | 設定の保存読込・MCP・オフラインキャッシュを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatDiagnostics.kt` | 管理者向け診断の送信（10秒ごと・停滞時のスレッドダンプ）と、端末の活動ログ（`ChatState` の変化の記録・所有者の切替） | 診断の送信・活動ログの記録項目を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatBanState.kt` | BAN境界：BAN時の作業停止、理由・日時の表示（`/banned`）、異議申し立ての送信と解除後の復帰 | BAN画面や異議申し立てを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAuth.kt` | ログインとアカウントセキュリティ：ID/Google/ブラウザ（PKCE）ログイン、Play Integrity・Turnstile、TOTP・パスキー（ログイン・2段階認証・登録）、セキュリティ設定の保存 | ログイン方式や2段階認証・パスキーを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAccountImport.kt` | 初回設定のアカウントZIP取り込み：分割アップロード（再開・中止）、適用、設定の上書き確認 | アカウントのインポート手順や確認画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAccountTransfer.kt` | 設定のデータタブ：アカウントのエクスポート・取り込み・重複修復の本体（設定モーダルを閉じても継続）と、取り込み後の変更カテゴリだけの再読込 | データの移行・取り込み・重複修復を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAnswerState.kt` | 回答の状態更新：`updateAnswer` / `updateLive` / `markApiAccepted`（受け付け済みの骨組み表示）を送信・受信・端末回答で共有 | 回答の状態更新や受け付け表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAccountSetup.kt` | アカウント読込と初回設定：`/me`・`/setup` の取得と保存、表示用モデル一覧の合成、最後のアカウントのオフライン復元 | アカウント情報の読込・初回設定・起動時の復元を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAiSettings.kt` | AI設定コマンド（`/settings <指示>`）：会話（直近10件）、モデル経由の設定の確認・変更、結果の一時バブル | `/settings` の会話・結果表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSlashCommands.kt` | 最小モードのスラッシュコマンド：ON/OFF・Thinkingの段階・Safety・Compress・一時チャットの切替と案内文 | `/canvas` などのコマンド動作・案内文を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatPdfExport.kt` | スレッドのPDF出力：分岐（`leaf_id`）のペイロード取得と、共有キャッシュへの端末生成A4 PDF | PDF出力の対象・保存先・失敗時の案内を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatProfiles.kt` | 端末プロフィールと接続先：アカウントなし／サーバー不使用モードの切替、端末チャットの保存先、オフライン時の可否判定、接続先の確認・保存 | 端末モード・オフライン判定・接続先を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatEditing.kt` | プロンプトオプションの復元と編集：編集開始・再生成・編集の取消・分岐の切替（‑ Web `restorePromptOptionsForEdit` / `beginEditMessage`） | 編集・再生成・分岐の切替を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSendEntry.kt` | 送信の入口の確認：Xリンクの自動検索（`applyXLinkAuto` / `#auto-search-banner`）、APIキーの保存と再送、モデル表示名 | 送信前の確認・APIキー要求を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatAttachments.kt` | 添付の入口：アップロード・中止・一覧のクリア・圧縮設定・ダウンロード・共有テキストのプレビュー・画像分割 | 添付の受け渡し・中止・ダウンロードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatForeground.kt` | 前面／背面の切替：復帰時の接続・回答の再結合判定、背面時の音声・Lyria・ポーリング停止、一時チャットの期限更新（ハートビート） | 前面復帰・バックグラウンド・一時チャットの期限を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatConnectionMonitor.kt` | サーバー到達性：ネットワークコールバック、前面時のハートビート、低速3回で不安定、復帰バナーの自動非表示、接続の活動ログ、待機ソケットの破棄とView Model終了時の解放 | 接続判定・ハートビート・接続バナーを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatThreadList.kt` | スレッド一覧と検索：300msデバウンス、ページ送り、オフラインキャッシュ、サーバー不使用時の端末チャット | 履歴一覧・検索・ページ送りを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatNavigator.kt` | 画面のチャット移動：新規チャット、スレッドを開く（読み込み中の骨組み・失敗時の再試行）、戻る、過去分の追加、前面復帰時の再読込 | チャットの切替・戻る・読み込みを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatThreadActions.kt` | スレッドとメッセージの操作：削除・名前変更・ブックマーク・分岐・メッセージ削除・暗号化の変更・チャット設定の保存 | 履歴やメッセージの操作・分岐を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSessionLifecycle.kt` | セッションの終了：ログアウト、401 の後始末（履歴・進行中ジョブ・音声セッション・端末バブルの解放）、失敗の通知（通信状態と案内文） | ログアウト・401・エラー表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatStartup.kt` | 起動時の復元（プロフィール・保存済みアカウント・サーバー設定・モデル一覧）と、接続バナーの「再接続」 | 起動時の読み込み順・復元・再接続を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatBatch.kt` | Batch処理：履歴一覧、完了バナー、2秒ごとの状態確認、停止と削除 | Batchの一覧・通知・ポーリングを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSkills.kt` | Skill：一覧の取得、保存・削除、このチャットへの適用（`last_skill_uuid`）、`@` 候補からの適用 | Skillの一覧・適用を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatStreamReceiver.kt` | 回答ストリームの受信：イベントの適用（本文・推論・ツールカード・MCP確認・エラー）と、サーバーで継続中の回答への再接続 | 受信イベントの扱い・再接続を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSendStream.kt` | 回答の送信：受け付け待ちの再試行（メンテナンス・サーバー停止・オフライン）、受け付け後のサーバー継続回答への再接続 | 送信の再試行・切断時の再接続を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSendRequest.kt` | 送信内容の組み立て：`/chat_stream` へ送る本文・添付・生成設定・Coding Mode・Skill・分岐（親メッセージ） | 送信する内容や項目を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSendCompletion.kt` | 送信後の後処理：回答終了時の履歴の再読込・タイトル生成・Batch通知と、受け付けられなかった送信の入力の復元・案内 | 送信完了後の画面・エラー表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatSubmissionRunner.kt` | 送信・再開・停止：送信前の確認と本文の組み立て、回答ジョブの開始（新規スレッド作成）、サーバーで継続中の回答への再結合、停止要求（端末生成の回答は取り消し） | 送信の手順・再開・停止を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatLowBandwidth.kt` | 低速回線モード：回線とデータ節約からの自動判定、サイドバーの auto → on → off 切替と保存 | 低速回線モードの判定・切替を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatBotGuard.kt` | 連投ロック（`/api/bot/lock`）、タップのテレメトリ（`/api/bot-telemetry`）、送信時の振動 | 連投の判定・BOTテレメトリ・振動を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/PromptEditSettings.kt` | 編集開始直前のプロンプト設定の退避とキャンセル時の復元 | プロンプト編集をキャンセルしたときの設定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatDefaults.kt` | 起動時・新規チャット共通の既定モデルとプロンプト設定の適用 | 起動時と新規チャットの設定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ChatGenerations.kt` | チャットごとの回答生成ジョブ・受信状態・同期前後のIDと画面への反映、生成中のサービス保持 | チャット切り替え後の生成継続・並行生成・停止対象を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ChatNavigationHistory.kt` | チャットを開いた順序と戻る先（新規チャットを含む） | Androidの戻る操作でチャットを切り替えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ServerOrigin.kt` | 現在の接続先（公式ホスト／セルフホスト）、接続先入力の検証、`/api/mobile/v1/config` からの対応ログイン方法の判定、アプリ固有スキームへの戻り指定 | 接続先の指定・同一オリジン判定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/backend/ChatBackend.kt` | 画面が呼ぶJSONエンドポイントの窓口（サーバー送信とローカル応答の切り替え） | 送信先・ローカルモードの経路を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/secure/EncryptedFileStore.kt` | Android KeystoreのAES-GCMによる暗号化ファイル（書き込み途中で壊れない保存） | 端末内の暗号化保存を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/SseReader.kt` | Server-Sent Eventsの読み取り（サーバーのRealtime／Lyriaと事業者APIのストリーム） | SSEの解析を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/` | サーバー不使用モードの直接送信：事業者APIへのHTTP（`DirectHttp.kt`、許可ホストのみ）、モデル別の振り分け（`DirectRouter.kt`）、Gemini（`GeminiDirect.kt`、Vertex AIは `VertexAuth.kt`）／OpenAI Responses（xAI含む、`OpenAiResponsesDirect.kt`）／Anthropic（`AnthropicDirect.kt`）／Chat Completions（`ChatCompletionsDirect.kt`）／画像生成（OpenAI／xAI／Ideogram）とTTS（`MediaDirect.kt`）の各エンジン（共通の枠は `DirectEngine.kt`）、システムプロンプトの組み立て（`SystemPromptBuilder.kt`）、添付のテキスト化（`AttachmentExtractor.kt`） | 端末からの送信・対応モデル・プロンプト組み立てを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/ProviderTokenCounter.kt` | 入力欄の本文・引用・添付の公式API計測、短時間キャッシュ | 端末からの入力見積の精度・計測APIを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/SkillTool.kt` | サーバー不使用モードのSkillツール：保存済みSkillの名前と短い説明をツール定義に載せ、`use_skill` で指示全文（長さの上限あり）を読み込む | 端末からのSkill適用・ツール定義を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/local/TokenEstimateBreakdown.kt` | 詳細を開いたときの本文・引用・添付の増分計測と結合時の差の算出 | 端末内の入力トークン見積の内訳を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PromptTokenEstimateDialog.kt` | 入力見積の詳細（開いた時点の入力、計測方法、本文・引用・添付ごとの内訳、再試行） | 入力見積をタップしたときの詳細画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/local/AttachmentTokenEstimator.kt` | 入力見積の添付トークン計算（Geminiの画像・PDF・音声・動画、テキスト抽出） | 端末内の添付トークン計算を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/local/` | 端末内プロファイル：チャット・添付・プロンプト設定とマスク参照の暗号化保存、サーバー不使用モードの送信待ち（`LocalChatStore.kt`）、設定・Skill・APIキー（`LocalSettingsStore.kt`）、プロファイルの切り替え（`LocalProfiles.kt`）、サーバーと同じ形で応答する窓口（`LocalChatBackend.kt`）、回答の生成に使うサーバーの履歴・添付の読み込み（`ServerHistory.kt`） | アカウントなし利用・サーバー不使用モードの保存や応答を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/sync/SyncEngine.kt` | サーバー不使用モードの送信待ちの保存（添付の先行アップロード、メッセージの送信、新しいチャットのタイトル、旧版の削除の反映） | 回答をアカウントへ送る手順・失敗時の扱いを変更するとき |
| `app/src/main/assets/serverless-defaults.json`, `ci/sync-serverless-defaults.py` | サーバーから生成したモデル一覧・自動注入プロンプト・Coding Mode指示（`--check` で差分検査） | サーバーのモデル一覧や自動注入プロンプトを変えたとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ServerlessSettings.kt` | 設定の「接続」カード（サーバー不使用モード、アカウントなしの説明とサーバーへのログイン） | 接続モードの設定画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ConnectionStatus.kt` | Web相当のハートビート接続状態・HTTP障害分類・監視間隔 | 接続状態・復帰表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/BatchNotifications.kt` | Batch完了通知チャンネルと安全な通知表示 | Batch通知を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/AnswerNotifications.kt` | アプリを離れている間に回答が終わったときの完了通知と通知チャンネル、戻ったときの消去 | 回答完了通知を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/BubbleNotifications.kt` | ユーザー操作で作成するチャットバブル通知と会話ショートカット、Android 10〜16のバブル専用 `ChatBubbleActivity` | Androidバブル・通知・ショートカットを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/VibrationSetting.kt` | 送信時・回答完了時の振動のオン/オフ（端末に保存。「Android」カードのスイッチ、`ChatViewModel.vibrate` が参照） | 振動の設定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ToolbarNotification.kt` | 通知パネルに常駐するツールバー通知（有効・無効の保存、ボタン一覧 `TOOLBAR_ACTIONS`、再起動後の再表示） | 通知ツールバーのボタンや表示条件を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ImageSplitTileService.kt`, `ImageSplitOverlayActivity.kt` | クイック設定の「画像を分割」タイルと、背後のアプリを残す透過画面での画像選択・分割・保存・共有 | 通知ツールバーからの画像分割を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/` | CookieなしHTTPとNDJSON（`PlaygroundApi.kt`）、APIモデル（`Models.kt`、`BatchModels.kt`）、Keystore保存（`TokenStore.kt`）、Credential Managerのパスキー手順（`PasskeyClient.kt`）。通知チャンネルの作成・更新状態の保持・前面判定は `../PlaygroundApplication.kt` | 認証・通信・保存 |
| `app/src/main/java/com/minashin1120/aiplayground/data/PlayIntegrityClient.kt` | Play Integrity Standard APIの事前準備・要求内容に結び付いた認証token取得 | Android認証の端末リスク信号 |
| `app/src/main/java/com/minashin1120/aiplayground/data/BotTelemetry.kt` | Web版のBot判定に合わせたタップの集計（`/api/bot-telemetry`）と、送信の連打の回数（`/api/bot/lock`） | Bot判定の基準・送信条件を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/BrowserLoginPkce.kt` | ブラウザー経由ログインのPKCE（S256）verifier・challenge生成 | Google・MinashinのApp Link復帰を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundScreen.kt`, `PlaygroundDialogs.kt`, `PlaygroundOverlays.kt`, `PlaygroundAuthScreens.kt`, `PlaygroundConversation.kt`, `ChatComponents.kt` | Compose画面の組み立て（ドロワー／タブレット2ペイン）、ダイアログ・シート・確認の開閉条件とその状態（`PlaygroundDialogs.kt` / `PlaygroundOverlays.kt`）、認証・初期設定・BAN・接続バナー画面、メッセージ操作、Coding差分（Live Code Changes）とBatch完了バナー、生成中の回答 | 画面構造と操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InAppCameraDialog.kt` | CameraXを使うアプリ内カメラのプレビュー、権限、ズーム、無音撮影、端末の向きに応じた撮影、画質優先の撮影設定、連続撮影と撮影分の一覧・添付・クリア、撮影分を画像分割へ渡す切り替え | カメラ撮影を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InAppCameraSession.kt` | CameraXのプレビュー・撮影の接続、ライフサイクルに応じた向きの検知、撮影の向き、ズーム・フォーカス | カメラの端末連携や撮影設定を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/CapturedPhotoOrientation.kt` | 撮影したJPEGのEXIFの向きを画素へ反映 | 撮影後の画像の向きを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ChatAutoScroll.kt` | Web `part05` の会話の自動追従（上へのスクロールで停止、下端付近で再開、「一番下へ」の表示条件） | 生成中のスクロールの動きを変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InlineVideoPlayer.kt` | 回答バブル内の動画取得・再試行とネイティブ動画プレイヤー | 回答内の動画再生を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/DirectStreamTrace.kt` | 直接通信の段階・受信バイト・空行とSSEイベント件数を、内容を渡さず集計する診断 | 送信後の空白時間やThinking受信の停滞を調べるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/direct/StreamingJsonBody.kt` | 大きな事業者APIリクエストのJSONを分割して書き出す | 送信時のメモリ使用量やJSON符号化を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LiveAnswer.kt` | 生成待ち・チャット読み込み中のスケルトン、チャット読み込み失敗のカード、Searching web／Search complete、Image Analysis、Python Execution、右下の通信スピナー | 生成中の表示を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/LogPolicy.kt` | 共通スキーマによる機密除外・ルート正規化・処理段階の記録 | 許可イベント・項目、処理の記録方法を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ActivityLog.kt` | 「ログの収集を強化」の操作ログ（通信・画面・状態の変化・タップ・エラーと診断ログの各段階。入力した本文・回答・ファイルの中身は含めない）をアプリのデータ領域に直近1時間ぶん記録し、フィードバック送信時に渡す | 操作ログの記録項目・保存・送信を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/Diagnostics.kt` | 管理者アカウントだけの診断ログ（処理の段階・所要時間・サイズ・固定エラー分類・停滞時のアプリコード位置。入力した本文・ファイルの中身は含めない）の端末への記録 | サーバー不使用モードの不具合調査で記録する項目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ProgressText.kt` | Web `progress_spinner.js` の文言表と通信中の処理一覧（`PlaygroundApi.progress`） | 通信スピナーの文言や対象を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AnswerBubbleMotion.kt` | プロンプト到着後の回答バブルの横スライド、回答待ちから保存済み回答への登場状態の引き継ぎ | 回答バブルの登場順序・アニメーションを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PromptSendMotion.kt` | 送信内容の移動・縮小・吹き出しへの切替、描画レイヤーと座標の一時管理、動きの削減・中断時の後始末 | プロンプト送信アニメーションを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/Composer.kt` | Webのプロンプトバー（引用・Coding対象・編集バー、モデルボタン、Canvas／Coding／Batch、Skillsスイッチ、添付プレビュー、Skill表示、`/`・`@` 候補、入力シェル、送信・停止、トークン見積もり） | 入力欄の見た目や操作を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ComposerChips.kt` | Webのプロンプトバーのオプションチップ（詳細チップの表示切替、Thinking／SysPrompt／Safety／PromptCache／Compress／一時チャット、チップ内のセレクトとバジェット入力） | 詳細チップの中身や見た目を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InteractionDiagnostics.kt` | 選択・数式操作の受動的なログ観測（ランダムID、フォーカス、メニュー、400ms後の状態、横スクロール・描画状態） | 選択や数式操作のログを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/QuoteSelection.kt` | 会話の範囲選択に「Quote」を加える選択ツールバー | 引用の操作を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/QuoteNavigation.kt` | 引用バーの「引用元へ移動」の位置決め：引用の上余白、引用元の行の解決（安定した行キーを優先し数値IDは予備、生成中の回答は進行中の行）、必要なスクロール量 | 引用元へ移動するときの位置を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ComposerRules.kt` | Webの `toggleOptions()` 相当（モデル別のチップ表示・強制値・Thinking levelの候補。EffortのモデルはThinkingボタン内にEffortを表示）、PromptCacheの提供元、トークン見積もりの文言 | モデル追加でチップの表示条件を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/CodingTargets.kt` | Coding Modeの対象候補の抽出、対象バーの文言、送信時の候補と上限 | Coding Modeの送信内容を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/Sidebar.kt` | Webの `#sidebar` 相当（ツールバー、検索、Skills、スレッド一覧、引っ張って更新、フッター） | サイドバーを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ChatHeader.kt` | スマートフォン用の上部バー（Webの `header.main-chrome-header`） | 上部バーを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebDialogs.kt` | チャット名の変更、Webの `confirm()`／`prompt()` 相当、チャット履歴・アルファ版・利用規約のモーダル、共通オーバーレイ | 確認ダイアログや小さなモーダルを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/LowBandwidth.kt` | Web相当の低速回線モードの判定・設定の切り替え・表示文言 | 低速回線モードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/StartupSplash.kt`, `app/src/main/java/com/minashin1120/aiplayground/ui/StartupSplashPolicy.kt`, `app/src/main/res/drawable/ic_playground_mark.xml` | 起動時のぼかし・光・ロゴと「AI Playground」の表示、起動完了後のロゴのズームによる画面リビールとアクセシビリティ対応（ロゴはベクターのパスで描画。図形は `ic_playground_mark.xml` と同じ）。`StartupSplashPolicy.kt` は、画面回転などの同一プロセス内の再生成では再生せず、プロセス終了後のタスク復元では再生する判定 | 起動アニメーション・ロゴ素材を編集するとき |
| `app/src/main/res/mipmap-anydpi-v26/ic_launcher*.xml`, `app/src/main/res/drawable/ic_launcher_{background,foreground,monochrome}.xml` | ランチャーアイコン（ベクターのアダプティブアイコン。テーマアイコン用の単色版を含む）。通知の小アイコンは `ic_playground.xml` | ランチャーアイコンを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsDialog.kt` | Web設定モーダルの枠（見出し、検索と結果一覧、10タブ、キャンセル／保存） | 設定画面の枠や検索を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsTabs.kt` | 設定のタブ定義と一般・APIキー・プロンプト・表示タブのカード | これらのタブの項目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsTabsOther.kt` | データ・アカウント・セキュリティ・2要素認証・フィードバック・MCPタブのカードとAndroidカード | これらのタブの項目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsForm.kt` | 設定の編集中の値と保存時の送信内容 | 設定項目を追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SettingsFields.kt` | 設定カード内の入力部品（チェックボックス、ラジオ、select、入力欄、小ボタン、スイッチ行） | 設定の部品の見た目を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ColorPickerDialog.kt` | テーマ色の選択（Webの `input[type=color]` 相当） | 色選択を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/CompressionDialog.kt` | 「画像・圧縮詳細設定」モーダル（圧縮と画像・OCRの生成設定）。圧縮の部分（`CompressionModal`）は「AIで翻訳」と画像分割の専用圧縮設定でも使う | 圧縮設定を変えるとき |
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
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImageMarkerEditor.kt` | 「画像編集」（マーカー、モザイク、トリミング、左右90度の回転、二本指の拡大、保存して反映）とObject Remover用マスク作成（近似色の自動塗り） | 画像編集・マスク作成の画面や保存を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageMarker.kt` | 画像編集の計算（線の補間、モザイク範囲、トリミングの掴み位置、拡大時の位置制限、回転の座標変換、保存名） | 画像編集の操作の決まりを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ImageSplitDialog.kt` | Webと共通の「画像を分割」ダイアログ（プレビュー、分割数、重なり、元画像の扱い、添付／共有／保存） | 画像分割の画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SharedTextPreviewDialog.kt`、`data/SharedTextMath.kt` | Android版だけの「メッセージとしてプレビュー」ダイアログ（共有されたテキストをAIの回答として `MessageBubble` で全画面に表示）と、数式の区切りが落ちたテキストへの区切りの補完 | 共有テキストのプレビューを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageSplit.kt` | 画像分割の計算（行列の決め方、重なり付きの範囲、EXIFの向きの逆変換、ファイル名、座標の帯の文言） | 画像分割の決まりを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageSplitCompression.kt` | 画像分割専用の圧縮設定の保存と、添付・保存・共有で共通の圧縮処理 | 分割画像の圧縮設定・適用を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageSplitRenderer.kt` | 画像分割の描画と保存（メモリ上限付きの元画像共有／部分デコード、番号・分割線・重なり、直接圧縮、添付用URI、Pictures／選択フォルダーへの保存） | 分割画像の見た目や保存先を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MinimalOptions.kt` | ミニマル表示の上部モデルバー、＋の「オプション」ポップアップ（対応モデルのマスク作成を含む）、Thinkingのスライドバー | ミニマル表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/MessageExtras.kt` | 回答からのPython実行結果とMCP実行メモの取り出し、Batch状態カードの文言 | 回答本文の前処理やBatchカードを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/UploadSheet.kt` | Webの「ファイルアップロード」画面（ボタン、Vision Model、進行状況、ファイルごとの画像編集・送信名・削除）と「画像を分割」ボタン | 添付の追加画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RealtimeStudio.kt` | Webの音声ドック（`#sts-panel`）と音声スタジオ、提供元別の音声設定、録音して `/sts` へ送る文字起こし系モデルの振り分け | 音声セッションUIを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LyriaStudio.kt` | Webの「Lyria RealTime Studio」（重み付きプロンプト、音楽設定、再生操作、チャットへ保存） | Lyriaスタジオを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RichPasteDialog.kt` | Webの「リッチ貼り付け」画面の枠（クリップボードHTMLのテキスト化、モデルへの指示、既定値の保存） | リッチ貼り付けを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ModelPicker.kt` | Webの「Select Model」（検索、固定タグバー、プロバイダー見出し、LLMを先頭にしたモデルカード、PromptCacheのロックバナー） | モデル選択画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SkillDialog.kt` | Webの `#skill-modal`（Skillの作成・編集）とTailwind風の入力欄 `TwInput` | Skillの編集画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LibraryDialog.kt` | Webの `#lib-modal`（検索、表示順、お気に入りのみ、複数選択の添付・ダウンロード・名前変更・使用チャット・削除、グリッド、使用チャットのモーダル） | ファイルライブラリを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/BranchManager.kt` | Webの `#branch-modal`（ツリー、ブランチ詳細、モデル別内訳、名前・固定、切替、削除、凡例と合計） | ブランチ管理を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/Library.kt` | ライブラリの並び順、音声・動画の判定、モデルの音声・動画入力対応（Web `getModelMediaSupport`） | ライブラリの添付規則を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/BranchSnapshot.kt` | 指定時点までの全分岐抽出、ファイル参照抽出、欠損リンク表示 | 別チャットへの分岐・共有ファイルを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/BranchTree.kt` | ブランチツリーの配置計算とパス累計トークン | ブランチ管理の表示計算を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/ChatInstructionsDialog.kt` | Webの `#thread-modal`（Chat Instructions：チャット専用の指示、全体プロンプトの参照、ユーザーシステムプロンプト、自動注入の簡易版） | SysPromptの⚙から開く画面を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundDialog.kt` | Webモーダル（オーバーレイ、ぼかし、パネル、見出し、下部）の共通枠 | 設定・Skills・ライブラリ・Batchの画面枠を調整するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InAppNoticeState.kt` | アプリ内通知の新着順・個別期限・消去状態・画面回転時の復元と表示先ウィンドウの管理 | 通知の保持・表示期限・モーダル間の引き継ぎを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InAppNoticeHost.kt` | アプリ内通知の共通Provider、通常画面・モーダルの表示ホスト、DNS補助通知の前面受信、通知を前面に出す確認ダイアログ | 通知の重なり・安全領域・表示するウィンドウを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InAppNoticeBanner.kt` | 画面上部に積む通知の描画、上下スワイプ、登場・退出・位置変更の動きと読み上げ | 通知の見た目・閉じる操作・アニメーションを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundMotion.kt` | 共通の時間・イージング、アニメーション削除設定への追従、モーダルの開閉・予測型「戻る」、一覧・登場・押下の動きの部品 | 画面の動き・遷移を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/AppChangelogDialog.kt` | Android版更新履歴のネイティブMarkdown表示、読み込み・再試行・戻る操作 | 更新履歴画面を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/GenerationOptionsPanel.kt` | Webの生成パネル（`composer_gen_image.html`／`composer_gen_media.html`）の描画、Lyria RealTimeの案内バー、入力制限の注記 | 詳細設定の表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/GenerationOptions.kt` | Webの生成パネルの項目とモデル別の表示規則（`p07` の `update*Ui` 相当）、値検証、送信対象の限定。File／SysPrompt／URLs／Thinking／Effort／SafetyはComposerのチップが持つ | 画像・動画・OCR・TTSの設定を追加するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/RealtimeModels.kt` | Realtime／Lyriaのセッション状態 | 音声・音楽ストリームの状態表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ModelUsageLedger.kt`, `app/src/main/java/com/minashin1120/aiplayground/data/ModelUsagePricing.kt`, `app/src/main/java/com/minashin1120/aiplayground/data/direct/UsageContext.kt` | 通信ごとの使用量・Thinkingの累積、生成コルーチン単位の分離、保存・同期用の数値記録（テスト: `ModelUsageLedgerTest.kt` / `local/ModelUsageStorageTest.kt` / `direct/DirectUsageTest.kt`） | 直接通信の費用予想・使用量保存を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ModelCostEstimator.kt`, `app/src/main/java/com/minashin1120/aiplayground/data/ModelCostPricing.kt`, `app/src/main/assets/model-pricing.json` | Webと共通の公式料金表、回答単位の費用予想・長文料金、キャッシュとツール追加料金、Ideogram全モデル、会話と全分岐の合算（テスト: `CacheToolCostTest.kt` / `IdeogramCostTest.kt`） | トークンバーの費用予想を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/WebModelCatalog.kt`, `app/src/main/assets/web-model-catalog.json` | Webの表示名・説明・価格・タグ・追加順をネイティブAPIのモデル一覧に合成 | モデル情報の表示差を確認するとき |
| `ci/sync-web-catalog.mjs` | Webモデル定義からAndroid表示用JSONを生成・`--check`で一致検証 | Webのモデル定義変更をAndroidへ同期するとき |
| `ci/audit-web-strings.py` | Androidの日本語文字列のうち、Webのテンプレート・JS・サーバー文言にないものを一覧表示 | 文言をWebと突き合わせるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundTheme.kt` | Webトークンから作るMaterial配色、Tailwindの文字サイズ、形状、レイアウト寸法、ライト／ダークの判定 | 色・テーマ・ブランドを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebTokens.kt` | Webの `:root` トークン（ダーク／ライト）、テーマ色の段階計算、Tailwind v3の色、ライトテーマでの置き換え | Webの色をそのまま使うとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebFonts.kt`, `app/src/main/res/values/font_certs.xml` | Google Fonts経由のNoto Sans JP／JetBrains Mono | フォントを変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/WebComponents.kt` | Web再現の共通部品（Font Awesomeアイコン、トグル、チェックボックス、select、チップ、設定カード、ボタン、モーダルの見出し・下部） | Webと同じ見た目の部品を使うとき |
| `app/src/main/res/drawable/fa_*.xml`, `ci/sync-web-icons.py` | WebのFont Awesomeサブセットから生成したアイコン（手で編集しない。`--check` で一致確認） | Webで使うアイコンが増減したとき |
| `ci/test-html-preview.py` | AndroidのHTMLプレビュー用HTMLをChromiumで描画し、日本語・CSS・JavaScript・隔離・危険スクリプト除去を検証 | HTMLプレビューの描画を確認するとき（AndroidのビルドやWebViewの実機検証は行わない） |
| `app/src/main/java/com/minashin1120/aiplayground/ui/HtmlCodePreviewDialog.kt`, `app/src/main/assets/html-preview-security.js` | HTMLコードブロックのプレビュー（専用URLから読み込む隔離したWebView、Webと共通の危険スクリプト除去） | HTMLプレビューの表示・安全対策を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownText.kt` | Webの `marked`（GFM・`breaks`）と `pre-wrap` 表示を再現するMarkdownの描画（リスト、引用、表、コード、`chat_error`、インラインコード）、回答中の生HTML（`<details>`・表を含む）とSVG | メッセージ本文表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownBlocks.kt` | Webの `marked` が作るブロックの解析（見出し・引用・リスト・表・フェンス・生HTML／SVG・画像・数式と、リンク先の検証） | Markdownのブロック解釈を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MarkdownColors.kt` | 回答本文の色（本文・引用・リンク・インラインコード・コードブロック・表・区切り線・highlight.js のテーマ色） | 回答本文の配色を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/SvgAnimation.kt` | アニメーション付きSVG（SMIL・CSS `@keyframes`）の検出と、スクリプト・外部参照を止めたWebViewでの再生（Webと同じ `<img>` のデータURI） | SVGのアニメーション表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/InlineAudioPlayer.kt` | 回答内の `<audio controls src>`（TTS・音楽・Lyriaの出力）を再生するインラインプレーヤー（初回タップで取得、再生・シーク・再生速度の切り替え。Web版の専用プレーヤーと同じ見た目）と、画面が渡す添付ダウンローダー `LocalAttachmentDownloader` | 回答内の音声の再生を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MessageBubble.kt` | Webの `.message-bubble`（吹き出し、タップで出る操作ボタン、引用、Thinking、添付グリッドとObject Removerマスク画像、分岐切替、フッター、回答処理時間と内訳）、合計・入力見積トークン帯、トークン詳細・暗号化モーダル、Welcome、一番下へ | メッセージの見た目を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/MathJaxRenderer.kt` | 同梱MathJaxを画面に出さないWebViewで動かしてTeXをSVGへ組版、キャッシュ、文中・独立行の数式の配置と描画 | 数式の組版・配置を編集するとき |
| `app/src/main/assets/mathjax/` | MathJax 3.2.2（`tex-svg.js`、TeX拡張、`LICENSE`）と実行用の `host.html` | MathJaxの版や設定を変えるとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/LatexText.kt` | LaTeXの読みやすい近似（MathJaxの読み込み中・組版できない式・コピー時の文字） | 近似表示を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/RemoteImage.kt` | 同一originの添付画像をBearer付きで取得し、キャッシュしてプレビュー | 添付・画像プレビューを編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/ui/FileViewer.kt` | Webの画像ビューアー（件数・前後移動・Download／Copy URL／Reuse／Close）とファイル表示パネル（テキスト・PDF・音声・動画） | ファイルや画像の表示を変更するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ThreadPdf.kt` | スレッドのネイティブA4 PDF出力（WebView不使用） | PDF出力を編集するとき |
| `app/src/main/java/com/minashin1120/aiplayground/data/ImageCompression.kt` | Web相当の画像圧縮設定（保存・読み込み）と、アップロード前の縮小・再エンコード、描画済みBitmapの直接圧縮。プロンプトバー・「AIで翻訳」・画像分割が、それぞれの保存先で同じ処理を使う | 画像圧縮を編集するとき |
| `app/src/main/res/` | アイコン、HTTPS設定、バックアップ除外、添付共有範囲 | Androidリソース・保護設定 |
| `ci/changelogs/` | Android版のバージョンごとの公開更新履歴Markdown | アプリ版の更新履歴を追加・修正するとき |
| `app/src/test/` | HTTP認証境界・ストリーム・データ解析・Markdown・数式のテスト | 通信・表示実装を変更するとき |
| `app/src/test/java/com/minashin1120/aiplayground/ui/*ScreenshotTest.kt` | Roborazziによる画面のスクリーンショット（Actionsの `android-reports` に保存） | Webとの見た目を比べるとき |
| `THIRD_PARTY_NOTICES.md` | Android版に含める第三者素材の表示 | アイコン・フォントなどを追加するとき |
| `ci/bootstrap-keystore.sh` | Actions限定の初回鍵生成と保存 | 初回署名準備 |
| `ci/run-gradle.sh` | Gradle実行の共通ラッパー。依存関係の一時的なダウンロード失敗だけ1回再試行 | CIのビルド手順を変えるとき |
| `ci/verify-keystore.sh`, `ci/verify-apk.sh` | 固定証明書とAPK署名の一致確認 | 署名・上書き更新を検証するとき |
| `ci/debug.keystore`, `ci/signing-fingerprint.txt` | Actionsが初回だけ保存する共有固定鍵と証明書のSHA-256 | 再利用のみ。削除・置換・再生成しない |
| `../.github/workflows/android.yml`, `../.github/workflows/release.yml` | CIと署名APKのリリース | GitHub Actionsの運用 |
| `app/src/androidTest/java/com/minashin1120/aiplayground/ui/HtmlPreviewDeviceTest.kt`, `ci/test-html-preview-device.sh` | GitHub ActionsのAndroidエミュレーターでHTMLプレビューのWebViewと実際の画面描画を検証 | HTMLプレビューのAndroid上の表示を検証するとき |
