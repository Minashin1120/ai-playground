# 数学ステップ解説（Math Steps）の機能と実装

AI Chat Playground の「数学ステップ解説（Math Steps）」機能について、機能概要、利用方法、処理アーキテクチャ、API仕様、Web/Android両クライアントでの実装、プロンプト設計、テスト・セキュリティ仕様を解説します。対象は 2026年11月13日リリースの V4.8.1168（Web）および V1.70.112（Android）以降のリポジトリ実装です。

---

## 1. 機能概要

「数学ステップ解説」は、数学や数式の解法を「1ステップずつ」段階的に提示し、学習者が自分のペースで理解を深められる対話支援機能です。

### 主な特徴

1. **1ステップごとの段階的提示**
   - AIの回答全体を一度に見せるのではなく、解法の各段階（「## ステップ N 見出し」）ごとに区切り、1ステップずつ集中して読むことができます。
   - 「次へ」「前へ」「最初から解き直す」などのナビゲーションにより、前後の論理展開を往復しながら学習できます。

2. **進捗の永続化とデバイス間同期**
   - 閲覧中のステップ位置（`current`）や到達した最大ステップ（`max_reached`）はサーバーに保存されます。
   - ページのリロードや別端末（Web・Android）からのアクセスでも、学習の続きから再開できます。
   - エンドツーエンド暗号化（E2EE）が有効な会話では、進捗情報も自動的に暗号化されて保存されます。

3. **「もっとやさしく」再生成（ステップ簡略化）**
   - 理解が難しい特定のステップだけを、元の問題や答えを変えずに「より丁寧・平易な表現」へ書き換えて表示できます。
   - 簡略化したステップは個別に保存され、いつでも「元の説明に戻す」ことが可能です。
   - 再生成に伴うトークン利用量や料金（Usage）も記録されます。

4. **安全なアニメーション図解（`mathsvg`）**
   - ステップの理解を助ける図解を、安全にサニタイズされたインラインSVG（`mathsvg`）として表示します。
   - SMILやCSSによる動きのある図解に対応しつつ、OSの「視差効果を減らす（prefers-reduced-motion）」設定時には静止画に自動抑制されます。

5. **柔軟な学習レベル設定**
   - 「説明の詳しさ（やさしい／ふつう／くわしい）」
   - 「1ステップの大きさ（小さめ／ふつう／大きめ）」
   - 「図の使用（必要なら使う／使わない）」
   - 「図のアニメーション（使う／使わない）」
   - 「基礎から説明する（前提の用語・公式の復習を冒頭に入れる）」
   上記の設定をプロンプトバーの歯車アイコン、またはアカウント設定から細かくカスタマイズできます。

6. **エクスポート・共有対応**
   - スレッドのPDFエクスポート時やクリップボードコピー時には、折りたたまれたステップが自動的にすべて展開された状態で出力されます。

---

## 2. 利用方法

### 基本的な使い方（Web / Android共通）

1. **機能を有効化する**
   - 初回は「設定」モーダルまたはミニマルオプションで「数学ステップ解説」をONにします。
   - 入力欄（プロンプトバー）付近に「数学ステップ」チップが表示されます。
   - チップのチェックボックスをONにすると、その送信でステップ解説プロンプトが有効になります。

2. **数学の問題を送信する**
   - 例：「$2x + 5 = 15$ を解いて」「三角関数の加法定理を使って $\sin 75^\circ$ を求めて」などの問題を送信します。
   - 回答が生成されると、ステップ解説専用のカードUIが表示されます。

3. **ステップを進める**
   - 画面下部の「ステップ 2 へ」ボタンを押して次へ進みます。
   - 途中が難しければ「もっとやさしく」ボタンを押すと、そのステップだけが分かりやすい解説に再生成されます。
   - 「前へ」や「最初から解き直す」で前のステップに戻ることも可能です。

4. **詳細設定を調整する**
   - チップ右端の歯車アイコンをクリック（Androidはタップ）すると、「数学ステップの設定」モーダルが開き、難易度や図の有無を調整できます。

---

## 3. 使用ファイルと該当行番号一覧

数学ステップ機能は、サーバーバックエンド、Webフロントエンド、Androidクライアント、テストコードにわたり密に連携しています。主要な実装ファイルと行番号は下表の通りです。

### サーバー側（Python / Flask / SQLAlchemy / Redis / RQ）

| ファイルパス | 行番号 | 主な役割・定義内容 |
|---|---|---|
| `app/server/background_math_steps.py` | L1-L227 | **数学ステップ解説のコア仕様・プロンプト・抽出ロジック**<br>- L7-L20: 設定既定値・制限定数（`MATH_STEPS_DEFAULTS`, `MATH_STEPS_MAX_CURRENT=50`, `MATH_STEPS_MAX_SIMPLIFIED_PER_STEP=20_000`, `MATH_STEPS_MAX_SIMPLIFIED_TOTAL=100_000`）<br>- L22-L25: 見出し抽出・コードブロック判定正規表現（`MATH_STEPS_HEADING_RE`, `_FENCE_RE`）<br>- L27-L64: フォーマット強制プロンプト定義（`MATH_STEPS_FORMAT_PROMPT`）<br>- L86-L111: 設定の正規化とJSON変換（`normalize_math_steps_config`, `dump_math_steps_config`）<br>- L114-L142: 各種指示文生成ヘルパー<br>- L145-L156: システムプロンプト動的構築（`build_math_steps_system_prompt`）<br>- L159-L175: ステップ簡略化プロンプト構築（`build_math_steps_simplify_prompt`）<br>- L178-L227: 回答テキストからのステップ抽出（`extract_math_steps`） |
| `app/server/routes_math_steps.py` | L1-L344 | **数学ステップAPIエンドポイント・状態管理・モデル実行**<br>- L15-L56: 進捗状態のパースと正規化（`_load_math_steps_state`, `_dump_math_steps_state`）<br>- L59-L76: 暗号化（E2EE）対応の状態読み書き（`_read_math_steps_state`, `_write_math_steps_state`）<br>- L78-L96: 所有権検証付きメッセージ解決（`_resolve_math_steps_message`）<br>- L113-L142: トークン利用量記録（`_append_math_steps_usage`）<br>- L144-L205: プロバイダー別モデル呼び出し（Gemini/Anthropic/DeepSeek/xAI/Zai/Kimi/OpenAI）<br>- L220-L257: 進捗保存API（`POST /api/math_steps/progress`、レート制限120回/分）<br>- L260-L344: ステップ簡略化API（`POST /api/math_steps/simplify`、レート制限20回/分） |
| `app/server/models.py` | L48-L50<br>L188 | **データベースモデル定義**<br>- `User` モデル: `math_steps_enabled`, `math_steps_config`, `default_math_steps`<br>- `Message` モデル: `math_steps_state` |
| `app/server/app_settings_schema.py` | L848-L883 | **データベースマイグレーションDDL**<br>- `ensure_user_math_steps_columns()`: userテーブルのカラム追加<br>- `ensure_message_math_steps_state_column()`: messageテーブルのカラム追加 |
| `app/server/background.py` | L735-L741 | バックグラウンドワーカーでのシステムプロンプトへの数学ステップ指示結合 |
| `app/server/routes_stream.py` | L213-L217<br>L376<br>L556-L557 | 通常チャットストリーミング送信時のオプション受領、`prompt_options` 保存、RQタスクへの引き渡し |
| `app/server/routes_browser_fast.py` | L155-L159<br>L167-L174<br>L359-L381 | ブラウザ高速モードでの `math_steps_system_prompt` 提供および保存処理 |
| `app/server/routes_settings.py`<br>`app/server/settings_ai.py` | L80-L82, L221-L224<br>L21, L77-L79, L180-L188 | Web設定API（`GET/POST /api/settings`）およびAI設定での数学ステップ設定読み書き |
| `app/server/routes_mobile.py` | L201, L277-L279, L417-L418 | モバイルアプリ向け設定同期API（`GET/POST /api/mobile/preferences`） |
| `app/server/mobile_auth.py` | L37 | モバイルアプリからの許可APIリストへの `math_steps_progress`, `math_steps_simplify` 登録 |
| `app/server/account_transfer.py`<br>`app/server/routes_account_import.py` | L19, L417-L438, L898-L903<br>L587-L596 | アカウントデータエクスポート・インポート時の暗号化・復号・データ検証 |
| `app/server/logging_schema.json` | L70-L74<br>L2296-L2297 | ロギングイベント定義（`math_steps.progress_*`, `math_steps.simplify_*`）およびAPIパス登録 |

---

### Webフロントエンド側（JavaScript / HTML / CSS）

| ファイルパス | 行番号 | 主な役割・定義内容 |
|---|---|---|
| `app/static/js/chat_core_parts/chat_core.part30_mathsteps.js` | L1-L874 | **Web版数学ステップ機能の全フロントエンドロジック**<br>- L7-L30: 定数・正規表現・共有状態オブジェクト（`mathStepsUiState`, `mathStepsRootState`）<br>- L32-L75: 設定・進捗状態の正規化ヘルパー<br>- L78-L130: クライアント側ステップ見出し解析（`parseMathSteps`）<br>- L140-L147: KaTeX/Markdownレンダリング（`renderMathStepsBodyHtml`）<br>- L149-L231: `mathsvg` サニタイズ・描画生成（`sanitizeMathSvgMarkup`, `buildMathSvgRenderHtml`）<br>- L258-L415: ステップUIのHTML生成・DOM差分更新（`buildMathStepsHtml`, `paintMathStepsRoot`）<br>- L417-L475: メッセージへの描画適用（`renderMathStepsInto`）<br>- L477-L534: 進捗保存とデバウンス制御（`postMathStepsProgress`, `scheduleMathStepsProgress`）<br>- L536-L602: 「もっとやさしく」リクエストと復元（`requestMathStepsSimplify`, `forgetMathStepsSimplified`）<br>- L604-L647: クリックイベント制御（`handleMathStepsAction`）<br>- L649-L703: プロンプトバーのチップ・設定連動（`setMathStepsEnabled`, `applyMathStepsSettings`）<br>- L705-L777: 設定モーダルの表示・保存処理（`openMathStepsModal`, `saveMathStepsConfig`）<br>- L779-L817: PDFエクスポート・クリップボード用展開ヘルパー<br>- L819-L874: 初期化処理（`initMathSteps`） |
| `app/templates/chat/composer_controls.html` | L18-L26 | プロンプトバーの「数学ステップ」トグルチップおよび設定歯車ボタンのHTML |
| `app/templates/chat/overlay_dialogs.html` | L146-L198 | 「数学ステップの設定」モーダルダイアログのHTML（セレクトボックス・チェックボックス） |
| `app/templates/chat/overlay_settings.html` | （該当箇所） | 全体設定モーダル内の既定値スイッチ（`#set-default-math-steps`） |
| `app/templates/setup.html` | L675-L676 | 初期セットアップ画面での設定項目マッピング |
| `app/static/css/chat.custom.v4.8.1168.css` | （末尾付近） | 数学ステップ用CSSクラス（`.math-steps-root`, `.math-steps-next`, `.mathsvg-render` 等）、ライトテーマ対応、視差効果軽減（`prefers-reduced-motion`） |

---

### Androidクライアント側（Kotlin / Jetpack Compose）

| ファイルパス | 行番号 | 主な役割・定義内容 |
|---|---|---|
| `app/android/app/src/main/java/com/minashin1120/aiplayground/data/MathSteps.kt` | L1-L218 | **Android側データモデル・パーサー・展開ユーティリティ**<br>- L9-L65: `MathStepsConfig`（設定モデル、JSON変換、プロンプト用展開 `promptValues()`）<br>- L75-L130: `MathStepsState`（進捗状態モデル、JSON変換、上限クランプ）<br>- L133: `MathStep`（1ステップの表現）<br>- L135-L177: ステップ見出し抽出パーサー（`extractMathSteps`）<br>- L180-L217: PDF・クリップボード用展開テキスト生成（`mathStepsPdfMarkdown`, `mathStepsCopyText`） |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ui/MathSteps.kt` | L1-L332 | **Android版Jetpack Compose UIコンポーネント**<br>- L63-L108: `MathStepsConfigFields`（設定編集フォーム）<br>- L110-L135: `MathStepsSettingsDialog`（設定ダイアログ）<br>- L137-L311: `MathStepsContent`（ステップ表示、プログレスバー、次へ/前へ/最初からボタン、「もっとやさしく」ボタン、アニメーションSVG表示）<br>- L313-L326: `MathStepsGear`（プロンプトバーの歯車アイコン） |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ChatMathSteps.kt` | L1-L128 | **Android非同期通信・状態更新コントローラー**<br>- L33-L56: 楽観的進捗保存（`saveProgress` → `/api/math_steps/progress`）<br>- L59-L81: 簡略化の復元（`clearSimplified`）<br>- L83-L114: ステップ簡略化実行（`simplify` → `/api/math_steps/simplify`）<br>- L116-L127: メッセージコレクションの不変更新（`updateState`） |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ChatComposer.kt` | L21, L188-L189 | プロンプトバーへのチップ追加と状態連動 |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ChatPreferences.kt` | L23, L58-L62 | ユーザー環境設定の送受信 |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ui/MessageBubble.kt` | L132-L149 | 回答メッセージバブルでの `MathStepsContent` 組み込み |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/ui/SvgAnimation.kt` | L1-L180 | ComposeでのアニメーションSVGレンダリング |
| `app/android/app/src/main/java/com/minashin1120/aiplayground/data/direct/SystemPromptBuilder.kt` | L20付近 | サーバーレス直接通信時の数学ステッププロンプト注入 |

---

### テストスイート

| ファイルパス | 行番号 | 主なテスト内容 |
|---|---|---|
| `app/tests/test_math_steps_server.py` | L1-L306 | **サーバー側ユニット・回帰テスト**<br>- 設定既定値とプロンプト生成テスト<br>- バックグラウンド/高速モードでのプロンプト注入テスト<br>- コードブロック内の見出し誤認防止テスト<br>- 暗号化（E2EE）進捗データの読み書きテスト<br>- 設定CRUDテスト<br>- 進捗API（クランプ、所有権検証）テスト<br>- 簡略化API（バリデーション、モデル呼び出し、Usage記録）テスト<br>- モバイル設定同期テスト |
| `app/tests/test_math_steps_ui.py` | L1-L292 | **WebフロントエンドUI回帰テスト**<br>- JSバンドル結合テスト<br>- `mathsvg` サニタイズ検証<br>- プロンプトバー・設定モーダルの存在検証<br>- ステップの1つずつ表示・ナビゲーション（次へ・前へ・最初から）テスト<br>- 進捗API送信・簡略化置き換え・復元テスト<br>- PDF・クリップボード展開テスト<br>- 通常Markdownフォールバックテスト |
| `app/android/app/src/test/java/com/minashin1120/aiplayground/data/MathStepsTest.kt` | L1-L108 | AndroidデータモデルのJSONシリアライズ・デシリアライズ、ステップパーサー、PDF/コピー展開テスト |
| `app/android/app/src/test/java/com/minashin1120/aiplayground/data/direct/MathStepsPromptTest.kt` | L1-L29 | Androidローカルプロンプトビルダーの展開テスト |

---

## 4. ドキュメント構成（詳細別紙）

機能の各詳細については、以下の各ドキュメントを参照してください。

1. [全体アーキテクチャと処理フロー](architecture.md)
   - システム全体のデータフローと状態遷移
   - 通常ストリーミング／ブラウザ高速モードでのシーケンス図
   - 進捗保存と「もっとやさしく」の非同期シーケンス図
   - データベース構造とエンドツーエンド暗号化（E2EE）
2. [サーバーAPIとバックエンド仕様](server_api.md)
   - `/api/math_steps/progress` および `/api/math_steps/simplify` のAPI仕様
   - 認可・レート制限・データクランプ仕様
   - プロバイダー別モデル呼び出しディスパッチャ（Gemini, Anthropic, DeepSeek, xAI 等）
   - トークン使用量（Usage）とコスト記録
3. [プロンプトエンジニアリングとフォーマット規約](prompt_engineering.md)
   - 数学家庭教師ペルソナのプロンプト設計思想
   - 「## ステップ N 見出し」規約とコードフェンス保護
   - 設定に応じた動的プロンプトパラメータ展開
   - 「もっとやさしく（Simplify）」専用プロンプトの設計
4. [Webフロントエンド実装（chat_core.part30）](web_frontend.md)
   - UI構造とDOM差分更新ライフサイクル
   - KaTeX数式レンダリングと `mathsvg` の安全な描画（サニタイズ処理）
   - 楽観的進捗更新とデバウンス送信
   - PDFエクスポート・クリップボードコピー時の全ステップ展開
   - レスポンシブデザイン、ダーク/ライトテーマ、視差効果軽減
5. [Androidクライアント実装（Jetpack Compose）](android_client.md)
   - Jetpack ComposeによるUIコンポーネント設計
   - `ChatMathSteps` による非同期通信と不変状態更新
   - Web版との見た目・操作性の完全同期
   - サーバーレス（直接API通信）モードでの動作
6. [テスト仕様・セキュリティ・保守運用](testing_and_security.md)
   - テストスイートの構成と回帰防止策
   - 機密情報保護（本文・プロンプト・回答のログ非出力ポリシー）
   - アカウントデータ移行（エクスポート/インポート）での取り扱い
   - トラブルシューティング
