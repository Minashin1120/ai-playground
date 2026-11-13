# 数学ステップ解説：アーキテクチャと処理フロー

本ドキュメントでは、「数学ステップ解説」機能の全体アーキテクチャ、処理の流れ、シーケンス図、およびデータ永続化と暗号化（E2EE）の仕組みを解説します。

---

## 1. 全体アーキテクチャ概要

数学ステップ解説は、プロンプトによる回答形式の制御、クライアント側での段階的レンダリング、サーバー側での進捗永続化、およびオンデマンドのステップ簡略化（再生成）から構成されます。

```mermaid
flowchart TD
    subgraph Client [クライアント層 (Web / Android)]
        UI[ステップ解説UI<br>KaTeX数式 / mathsvg描画]
        Parser[ステップパーサー<br>## ステップ N の解析]
        State[進捗状態管理<br>current / max_reached / simplified]
    end

    subgraph Server [サーバー層 (Flask / RQ)]
        API[routes_math_steps.py<br>/api/math_steps/*]
        Stream[routes_stream.py / background.py<br>システムプロンプト動的注入]
        Core[background_math_steps.py<br>プロンプト生成・抽出・設定正規化]
    end

    subgraph Storage [データ・キュー層]
        DB[(MySQL / SQLite<br>User / Message)]
        Redis[(Redis / RQ)]
    end

    subgraph Provider [AIプロバイダー]
        LLM[Gemini / OpenAI / Anthropic / xAI 等]
    end

    UI --> Parser
    Parser --> State
    State -->|進捗保存 POST| API
    State -->|もっとやさしく POST| API
    API --> Core
    API --> DB
    API -->|ステップ再生成| LLM
    Stream --> Core
    Stream --> Redis
    Redis --> LLM
```

---

## 2. 処理フローとシーケンス

### 2.1 チャット送信とストリーミング生成フロー

ユーザーが「数学ステップ」を有効にしてメッセージを送信した際、システムプロンプトに数学家庭教師ペルソナと出力形式規約が結合されます。

```mermaid
sequenceDiagram
    autonumber
    participant U as ユーザー
    participant C as クライアント (Web/Android)
    participant S as Flask サーバー
    participant W as RQ ワーカー
    participant P as AI プロバイダー

    U->>C: 問題を入力し「送信」（数学ステップ ON）
    C->>S: POST /chat_stream { math_steps: true, message: "..." }
    S->>S: normalize_math_steps_config(user.math_steps_config)
    S->>S: user_msg.prompt_options に math_steps: true を保存
    S->>W: 生成タスクを Redis キューへ投入
    S-->>C: SSE ストリーミング開始 (assistant メッセージ枠)
    W->>W: build_math_steps_system_prompt() でシステムプロンプト結合
    W->>P: LLM へ生成リクエスト送信
    loop ストリーミング受信
        P-->>W: トークンチャンク
        W-->>S: キュー経由配信
        S-->>C: SSE chunk イベント転送
        C->>C: 受信テキストを parseMathSteps() で段階解析
    end
    W->>S: 生成完了通知
    S-->>C: SSE done イベント
    C->>C: ステップ1のみを表示し、次へボタンを有効化
```

#### ブラウザ高速モード（Browser Fast Mode）の場合
ブラウザ高速モード（Gemini APIキーを端末に持ち、クライアントから直接Geminiへ通信するモード）では、以下のフローとなります。
1. クライアントが `POST /api/browser_fast_mode/bootstrap` を呼び出す際、`math_steps: true` を渡します。
2. サーバー（`server/routes_browser_fast.py`）はレスポンスに `math_steps_system_prompt`（ユーザー設定を反映したシステムプロンプト）を含めて返します。
3. クライアントはGeminiへの直接リクエスト時にこのプロンプトをシステム指示として渡し、回答を受信します。
4. 会話保存時（`POST /api/browser_fast_mode/save`）に、`prompt_options` へ `math_steps: true` が記録されます。

---

### 2.2 ステップ進行と進捗保存フロー

ユーザーが「ステップ 2 へ」などのボタンを押した際、クライアントは即座にUIを更新（楽観的UI更新）し、バックグラウンドで進捗をサーバーへ保存します。

```mermaid
sequenceDiagram
    autonumber
    participant U as ユーザー
    participant C as クライアント (Web/Android)
    participant S as Flask サーバー
    participant D as データベース (Message)

    U->>C: 「ステップ 2 へ」をクリック
    C->>C: current: 1, max_reached: max(1, max_reached) に更新
    C->>C: ステップ2の本文と図を描画（KaTeX / mathsvg）
    Note over C: デバウンスタイマー待機 (Web: 280ms / Android: 即時Coroutine)
    C->>S: POST /api/math_steps/progress<br>{ message_id: 123, current: 1, max_reached: 1 }
    S->>S: レート制限確認 (120回/分)
    S->>S: メッセージ所有権確認 (current_user.id == thread.user_id)
    S->>S: 値をクランプ (0 <= current <= 50)
    S->>D: math_steps_state を JSON 保存 (E2EE の場合は暗号化)
    S-->>C: HTTP 200 OK { status: 'ok', math_steps_state: {...} }
```

---

### 2.3 「もっとやさしく」再生成（ステップ簡略化）フロー

理解が難しいステップで「もっとやさしく」ボタンを押すと、そのステップだけを対象とした平易な解説が生成されます。

```mermaid
sequenceDiagram
    autonumber
    participant U as ユーザー
    participant C as クライアント (Web/Android)
    participant S as Flask サーバー
    participant P as AI プロバイダー
    participant D as データベース (Message)

    U->>C: 「もっとやさしく」をクリック (例: ステップ 2)
    C->>C: ボタンをローディング状態（ビジー表示）に変更
    C->>S: POST /api/math_steps/simplify<br>{ message_id: 123, step_index: 1 }
    S->>S: レート制限確認 (20回/分)
    S->>S: メッセージ所有権確認 & テキストモデル確認
    S->>S: extract_math_steps() で対象ステップの本文を取得
    S->>S: build_math_steps_simplify_prompt() を前置
    S->>P: 回答時と同じモデルへワンショット生成リクエスト
    P-->>S: 簡略化されたステップ本文
    S->>S: 見出しタグの除去サニタイズ
    S->>S: 文字数上限チェック (1ステップ20,000文字 / 全体100,000文字)
    S->>D: math_steps_state.simplified["1"] を保存 (E2EE暗号化)
    S->>D: トークン利用量（Usage）を thought_data に追記
    S-->>C: HTTP 200 OK { status: 'ok', text: "...", math_steps_state: {...} }
    C->>C: ステップ2の表示を簡略化テキストへ差し替え
    C->>C: 「元の説明に戻す」ボタンを表示
```

---

## 3. データモデルと永続化

### 3.1 ユーザー設定（`User` モデル）

ユーザーのアカウント設定として、以下のカラムが追加されています（`server/models.py` L48-L50）。

| カラム名 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `math_steps_enabled` | Boolean | `False` | 数学ステップ機能の全体スイッチ（マスター設定）。ONにするとプロンプトバーにチップが表示されます。 |
| `math_steps_config` | Text | `None` | 詳細設定を保持するJSON文字列。 |
| `default_math_steps` | Boolean | `False` | 新規チャット作成時にプロンプトバーの数学ステップを既定でONにするかどうか。 |

#### `math_steps_config` のJSON構造
```json
{
  "basics": false,
  "level": "easy",
  "step_size": "normal",
  "diagrams": "auto",
  "animation": true
}
```
- `basics` (bool): 基礎から説明するかどうか（前提公式の復習の有無）
- `level` (str): `"easy"`（やさしい）, `"normal"`（ふつう）, `"detailed"`（くわしい）
- `step_size` (str): `"small"`（小さめ）, `"normal"`（ふつう）, `"large"`（大きめ）
- `diagrams` (str): `"auto"`（必要なら使う）, `"off"`（使わない）
- `animation` (bool): 図のアニメーションを許可するかどうか

---

### 3.2 メッセージ進捗状態（`Message` モデル）

AI回答メッセージ（`role == "assistant"`）の進捗状態は、`math_steps_state` カラム（Text、`server/models.py` L188）に保持されます。

#### `math_steps_state` のJSON構造
```json
{
  "current": 1,
  "max_reached": 3,
  "simplified": {
    "1": "このステップのよりやさしい解説文..."
  }
}
```
- `current` (int): 現在表示しているステップのインデックス（0始まり、最大50）。
- `max_reached` (int): ユーザーが到達した最大のステップインデックス。一度到達した値より小さくなることはありません。
- `simplified` (dict[str, str]): 「もっとやさしく」で生成されたステップごとの代替テキスト。キーはステップインデックス文字列。

---

## 4. エンドツーエンド暗号化（E2EE）への対応

AI Chat Playground では、ユーザーが暗号化会話を有効にしている場合、メッセージ本文および付随するメタデータがクライアントとサーバー間で暗号化されます。数学ステップ機能でも、プライバシー保護の観点からE2EEとの完全な統合が行われています。

1. **読み出し時の自動復号（`server/routes_math_steps.py` L59-L68）**
   - メッセージの `is_encrypted` フラグが真である場合、保存された `math_steps_state` を `decrypt_val()` で復号してからパースします。
   - 回答本文（`_math_steps_message_content`）も `decrypt_val()` を経由して復号されます。
2. **書き込み時の自動暗号化（`server/routes_math_steps.py` L70-L76）**
   - 更新された状態JSON文字列を `encrypt_val()` で暗号化してから `message.math_steps_state` にセットしてコミットします。
3. **Usage記録の保護（`server/routes_math_steps.py` L122-L142）**
   - ステップ簡略化で消費されたトークン使用量は `message.thought_data` に記録されますが、こちらも暗号化メッセージの場合は暗号化して保存されます。
4. **アカウントデータ移行（エクスポート/インポート）**
   - エクスポート時（`server/account_transfer.py` L417-L420）は平文に復号してエクスポートアーカイブに格納。
   - インポート時（`server/routes_account_import.py` L587-L596）はアカウントの暗号化設定に応じて再暗号化して保存されます。
