# chat_core_parts — チャットコアの部品ファイル

`static/js/chat_core.v4.8.*.js`（約2.8万行）は、編集しやすくするためにこのディレクトリの**順序付き部品**（`chat_core.partNN_名前.js`）に分割されています。`scripts/build_frontend.sh` が部品を番号順に連結して結合ソース `chat_core.v4.8.*.js` を再生成し、それを圧縮してブラウザー用のファイルを作ります。

ブラウザーへは、結合ソースを1回で圧縮し、トップレベルの `//! @chat-core-bundle-split` 行（現在は2か所）で切った複数のファイル（`chat_core.min.v4.8.*.1.js`、`.2.js`、… をこの順に読み込む）で配信します。この行を移すときは、トップレベルの文の切れ目で、前のファイルが読み込み中に後ろのファイルの関数を呼ばない位置に置いてください（`tests/test_chat_core_bundles.py` がブラウザーで確認します）。1ファイル40万バイト・合計105万バイトの上限があり、近づいたら分割位置の行を追加・移動します。

- 部品は**連結順に番号が振られており、順番を変えてはいけません**（変えると動作が壊れます）。
- 結合ソースは部品の連結と**バイト単位で一致**することが検証で保証されています。
- **編集対象は部品ファイル**です。結合ソースと圧縮ファイルは手で編集せず、編集後は必ず `scripts/build_frontend.sh` を実行してください。
- 部品は約1,000行（上限1,500行）を目安に分かれています。境界は原則として文の切れ目ですが、長いクラスや関数の内部で分かれることがあります。
- コードを探すときは部品名と下の表を使い、必要なら `scripts/show_regions.sh static/js/chat_core_parts` で `// region:` の範囲を確認してください。

| ファイル名 | 主な内容 | いつ開くか |
|---|---|---|
| `chat_core.part01_nativeconsolelog.js` | 基盤ユーティリティ。`get()` ヘルパー、コンソール、設定保存ボタン制御、画像読み込み失敗フォールバック、テーマ（`applyThemeColor` / `THEME_STORAGE_KEY`）、Liquid Glass、モーダルの開閉（`showModal` / `hideModal`） | 基盤・テーマ・モーダルを変更するとき |
| `chat_core.part02_syncthemeinputs.js` | テーマ色入力の同期、圧縮設定と共通の画像圧縮処理、アダプティブぼかし（低負荷自動化）、ユーザーシステムプロンプト保存時の同期、モーダル・トーストの後始末 | テーマ入力・圧縮・ぼかしを変更するとき |
| `chat_core.part03_insertclipboardimageblob.js` | リッチペースト（HTML整形・サニタイズ・印刷）、貼り付け画像、MathJax読み込み | リッチペースト・貼り付け画像を変更するとき |
| `chat_core.part04_refreshcsrftoken.js` | Google／Minashin連携解除・連携UI（`updateGoogleLinkUI` / `updateMinashinLinkUI`）、各種定数（添付・低帯域・一時チャット・Coding/Canvas・マーカー）、CSRFトークン更新、セキュリティ（polyfill.io等のブロック） | 連携UI・共通定数・セキュリティを変更するとき |
| `chat_core.part05_updatependingskeletonstatus.js` | トークン見積もり・クリックで開く詳細内訳、プロンプトバー表示・ミニマル表示の操作項目、Thinkingレベル、コード折りたたみ、低帯域モード適用、送信待ちスケルトン | トークン表示・プロンプトバー・Thinkingを変更するとき |
| `chat_core.part06_buildminimaloptionitem.js` | ミニマル表示のプラスボタンポップアップ、Thinkingスライダー、Coding対象の表示、`applyChatDefaults`、編集時のスナップショット復元、`restoreLatestPromptOptions` | ミニマル表示・Thinking・プロンプト設定の復元を変更するとき |
| `chat_core.part07_reconnectpendingstreamuntilavailable.js` | 送信待ちストリームの再接続、設定タブ（`TAB_LABELS`）、Botロックオーバーレイ、SWキャッシュモード（`applyCacheMode`）、Turnstile初期化 | 再接続・設定タブ・キャッシュを変更するとき |
| `chat_core.part08_getsettingstabsmaxscroll.js` | 設定タブの横スクロール、チャットの自動スクロールと最下部移動、画像ビューア、コピーのフォールバック、引用返信 | スクロール・画像ビューア・コピー・引用を変更するとき |
| `chat_core.part09_models.js` | モデル定義 `MODELS`、モデルタグ・スラッシュコマンド・AI設定アシスタント、Skill候補（`@`）、添付とモデルの補助、PromptCache、メディア対応、音声一覧とモデル種別判定、STS/TTS UI | モデル定義・音声・PromptCacheを変更するとき |
| `chat_core.part10_updatestsoptions.js` | 音声（TTS/STS）オプションの更新、モデル名・タグの補助、設定モーダルのモデル／音声一覧 | 音声オプション・設定のモデル一覧を変更するとき |
| `chat_core.part11_getmodelcapabilitysearchterms.js` | モデル一覧の描画・検索・Visionモデル選択、PromptCacheの制約、ブラウザ高速モードとBatchの選択肢制御、MCP（外部モデル連携）設定UI | モデル表示・PromptCache・MCP設定を変更するとき |
| `chat_core.part12_toggleoptions.js` | SysPromptなどの切替（`toggleOptions`）、モデルに応じた選択の復元、高速モード・Batchトグル、モデル別の項目制御 | オプション切替・モデル別の表示制御を変更するとき |
| `chat_core.part13_importfileselection.js` | DOMContentLoaded 初期処理（前半）。テーマ初期化、アカウントエクスポート/インポートのファイル選択、一時チャット、セッション管理UI | 初期化・アカウントファイル選択を変更するとき |
| `chat_core.part14_collectautosystempromptconfigfromform.js` | アカウント移行（エクスポート/インポート/重複修復）続き、管理者スレッド暗号化、自動システムプロンプト設定、テーマ設定バインド | アカウント移行・自動プロンプト・テーマ設定を変更するとき |
| `chat_core.part15_openbotadminmodal.js` | `MODAL_CONFIG`（URLルート↔モーダル対応）と `closeModalById`、`popstate` ハンドラ（戻る/進むでモーダル開閉・スレッド復元）、Bot管理画面の入口 | モーダルのURL・戻る操作を変更するとき |
| `chat_core.part16_stopsilencemonitor.js` | 2FA（TOTP/WebAuthn）、リアルタイム音声セッション（Gemini Live・`RealtimeVoiceSession`・録音・無音監視・音声再生）、フィードバック管理 | 認証・音声・フィードバックを変更するとき |
| `chat_core.part17_handlestreammessage.js` | 音声セッションの応答処理、Lyria RealTime スタジオ、音声スタジオUI（`VoiceStudio`）、ファイルライブラリ／Skillモーダルの閉じる・選択更新 | 音声セッション・音声スタジオ・ライブラリ初期化を変更するとき |
| `chat_core.part18_setlibbtnlabel.js` | ファイルライブラリの選択UI・ラベル、ファイルプレビュー・画像ビューア、画像編集・Object Remover用マスク作成モーダル、カメラキャプチャ、アップロード行の描画 | ファイル表示・ライブラリ選択・マスク作成を変更するとき |
| `chat_core.part19_setcameracapturestatus.js` | アップロード進捗、カメラストリーム制御、画像編集（ブラシ・モザイク・切り抜き）とマスク描画、Canvasモバイル表示、音声キャプチャ | 画像編集・カメラ・Canvas・音声入力を変更するとき |
| `chat_core.part20_initmarkercanvas.js` | マーカーCanvasの初期化、画像マーカーの描画・変換・保存、類似色の自動塗り | 画像マーカー・マスク描画を変更するとき |
| `chat_core.part21_buildsvgcoderenderhtml.js` | メッセージ吹き出し（本文・添付画像・マスク画像・フッター・回答処理時間と内訳）、SVG/Canvasプレビュー生成、Codingモード | 吹き出し・添付・Canvas・Coding表示を変更するとき |
| `chat_core.part22_updatesearchbox.js` | ストリーム描画（検索中・ペンディング・Batch状態表示）、スラッシュコマンドパレット、Skill候補、Python実行モーダル | ストリーム表示・スラッシュ・Skill・Python実行を変更するとき |
| `chat_core.part23_selectslashcommand.js` | スラッシュコマンドの選択・実行、送信処理 `sendMessage` の入口、ブラウザ直接送信・高速モード | 送信・スラッシュコマンド・直接送信を変更するとき |
| `chat_core.part24_proceed.js` | 各プロバイダのBatch送信、送信後の即時描画、プル更新、Xリンク自動検索、`/settings`（AI設定アシスタント）、引用・返信 | 送信・Batch・履歴一覧を変更するとき |
| `chat_core.part25_updatethreadhighlighting.js` | スレッド一覧の読み込み・追記、一時チャット（ハートビート・タイムアウト）、各プロバイダのBatch状態ポーリング・完了通知 | 履歴読み込み・一時チャット・Batch状態を変更するとき |
| `chat_core.part26_ensurecurrentchatheaderticker.js` | 一時チャットの説明文、スレッド設定モーダル、圧縮設定モーダル、PDFエクスポート、Skill適用、ブランチ集計、新規チャット | 一時チャット・スレッド設定・PDFを変更するとき |
| `chat_core.part27_renderlibraryskeleton.js` | ファイルライブラリの骨組み・一覧操作、ブランチ管理UI、Batch処理一覧モーダル（取得・停止・履歴削除・チャットを開く） | ライブラリ一覧・ブランチ・Batch一覧を変更するとき |
| `chat_core.part28_renderaccounts.js` | メッセージコピー、規約・Alpha案内モーダル、Thinking表示切替、クライアントデバッグログ、管理者のBot検出ログ画面、別チャットへの全分岐コピー、画像分割、送信アニメーション | コピー・デバッグ・Botログ・分岐・画像分割・送信アニメーションを変更するとき |
| `chat_core.part29_estimateconversationcost.js` | API呼び出しごとの費用予想・長文料金、Ideogram全モデルの画像単価と削除料金、会話・全分岐の合算、トークン詳細向けキャッシュ読込・TTL別書込の集計、公式料金データの読込 | 費用予想・キャッシュ内訳の計算を変更するとき |
| `chat_core.part30_mathsteps.js` | 数学ステップ解説（`parseMathSteps` / `buildMathStepsHtml` / `renderMathStepsInto` / ステップ操作・進捗保存・「もっとやさしく」）、`mathsvg` の画像描画、設定（マスター・チャット既定値・歯車モーダル）、PDF・コピー用の変換 | 数学ステップ・mathsvg・その設定を変更するとき |
| `../../../scripts/build_frontend.sh` | 部品を番号順に連結し、トップレベルの `//! @chat-core-bundle-split` で配信用JSを分割・圧縮する | JS部品の編集後にビルドするとき。結合ソース・圧縮ファイルは手で編集しない |

## 再分割が必要になったら

部品が `CHAT_CORE_PART_MAX_LINES`（1,500行）を超えると `scripts/verify_changes.sh` が警告します。その場合は `scripts/rebuild_chat_core_parts.sh` を実行して再分割し、このREADMEの表を更新してください。1,000行を超えた部品は警告の対象ですが、公開はブロックされません。
