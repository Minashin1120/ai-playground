# 数学ステップ解説：サーバーAPIとバックエンド仕様

本ドキュメントでは、「数学ステップ解説」機能のサーバー側APIエンドポイント、バリデーション規約、レートリミット、モデル呼び出しディスパッチャ、および利用量記録の仕様を解説します。

---

## 1. APIエンドポイント一覧

数学ステップ機能専用のエンドポイントは、`server/routes_math_steps.py` に定義されています。

| メソッド | パス | 認証 | レートリミット | 概要 |
|---|---|---|---|---|
| `POST` | `/api/math_steps/progress` | 要ログイン | 120回 / 60秒 / ユーザー | 現在のステップ位置および最大到達ステップの保存、または分割・簡略化ステップの破棄 |
| `POST` | `/api/math_steps/simplify` | 要ログイン | 20回 / 60秒 / ユーザー | 指定した1ステップを「もっとやさしい解説」に再生成 |

モバイルクライアント（Android）からのアクセスも許可されており、`server/mobile_auth.py`（L37）にエンドポイント名が登録されています。

---

## 2. API詳細仕様

### 2.1 進捗保存：`POST /api/math_steps/progress`

ユーザーのステップ移動（次へ／前へ／最初から）に合わせて進捗を保存します。また、「もっとやさしく」で作った分割ステップや以前に簡略化したステップを元に戻す操作（`remove_expanded` / `remove_simplified`）も本エンドポイントで受け付けます。

#### リクエストヘッダー
- `Content-Type: application/json`

#### リクエストボディ
```json
{
  "message_id": 12345,
  "current": 2,
  "max_reached": 3,
  "remove_expanded": [1],
  "remove_simplified": [2]
}
```
または `job_id` による指定も可能です：
```json
{
  "job_id": "job_abcdef123456",
  "current": 2,
  "max_reached": 3
}
```

| フィールド名 | 型 | 必須 | 説明 |
|---|---|---|---|
| `message_id` | 整数 / 文字列 | 条件付き | 対象のアシスタントメッセージID（`job_id` とのいずれか一方必須） |
| `job_id` | 文字列 | 条件付き | 生成ジョブID（最大128文字） |
| `current` | 整数 | 任意 | 現在表示しているステップインデックス（0〜50にクランプ、分割後の表示順） |
| `max_reached` | 整数 | 任意 | 到達した最大ステップインデックス（0〜50にクランプ。現在の `max_reached` や `current` より小さくはならない） |
| `remove_expanded` | 配列[整数] | 任意 | 「もっとやさしく」で分割したステップを元の1ステップに戻す（元ステップのインデックスを指定） |
| `remove_simplified` | 配列[整数] | 任意 | 旧形式の簡略化テキストを破棄し、元の説明に戻すステップインデックスの配列 |

#### レスポンス（成功時：HTTP 200 OK）
```json
{
  "status": "ok",
  "math_steps_state": {
    "current": 2,
    "max_reached": 3,
    "simplified": {},
    "expanded": {}
  }
}
```

#### エラーレスポンス
- `429 Too Many Requests`: `{"error": "rate_limit"}`（1分間に120回超過）
- `400 Bad Request`: `{"error": "invalid_request"}` または `{"error": "invalid_message"}` / `{"error": "invalid_job"}`
- `404 Not Found`: `{"error": "message_not_found"}`（メッセージが存在しない、または自アカウント所有でない場合）

---

### 2.2 ステップ分割：`POST /api/math_steps/simplify`

指定されたステップを、「もっとやさしい」複数の小さなステップに分けてLLMに生成させます。分割後の各ステップは元の回答へ差し込まれ、読み進めながらステップ数を増やせます。分割済みの小ステップをさらに分割することもできます。

#### リクエストボディ
```json
{
  "message_id": 12345,
  "step_index": 1,
  "part_index": 0
}
```

| フィールド名 | 型 | 必須 | 説明 |
|---|---|---|---|
| `message_id` | 整数 / 文字列 | 必須 | 対象のアシスタントメッセージID |
| `step_index` | 整数 | 必須 | 対象の元ステップのインデックス（0始まり、`expanded` のキー） |
| `part_index` | 整数 | 任意 | 分割済みステップのうち、さらに分割したい小ステップの位置。省略すると元ステップ全体を分割 |

#### 処理の流れ
1. **検証**: ユーザーのメッセージ所有権、メッセージロール（`assistant`）、本文の有無、ステップ／`part_index` の範囲を検証。
2. **モデル対応判定**: 元のメッセージ生成に使用されたモデル（`message.model`）がテキストモデルであるか判定（画像専用・音声専用モデルは除外）。
3. **対象抽出**: 元テキストから対象ステップを取得。`part_index` 指定時は保存済みの分割ステップを対象にする（最大100,000文字以内）。
4. **プロンプト構築**: ユーザーの `math_steps_config` に基づく `build_math_steps_simplify_prompt()` と対象テキストを結合。2〜4個の小さなステップへの分割を指示。
5. **LLM呼び出し**: 元メッセージと同じプロバイダー・APIキー設定でモデルを実行（`_call_math_steps_simplify_model`）。
6. **解析・上限チェック**:
   - 返答を `extract_math_steps()` でステップ見出しごとに解析。見出しがない場合は全文を1ステップの本文として扱う。
   - 1元ステップあたり最大8分割、表示ステップ合計は最大50。
   - 1分割あたり最大20,000文字、1メッセージ内の全分割・簡略化合計で最大100,000文字の制限を適用。
7. **保存とUsage記録**: `math_steps_state.expanded[step_index]` を更新（E2EE暗号化考慮）し、モデル使用量を `thought_data` に記録。

#### レスポンス（成功時：HTTP 200 OK）
```json
{
  "status": "ok",
  "text": "やさしい本文1\n\nやさしい本文2",
  "steps": [
    { "title": "やさしく整理", "body": "やさしい本文1" },
    { "title": "やさしく計算", "body": "やさしい本文2" }
  ],
  "step_index": 1,
  "math_steps_state": {
    "current": 1,
    "max_reached": 3,
    "simplified": {},
    "expanded": {
      "1": [
        { "title": "やさしく整理", "body": "やさしい本文1" },
        { "title": "やさしく計算", "body": "やさしい本文2" }
      ]
    }
  }
}
```
`steps` は対象 `step_index` の分割後リスト全体です。旧クライアント向けに `text`（本文の連結）も返します。

#### エラーレスポンス
- `429 Too Many Requests`: `{"error": "rate_limit"}`（1分間に20回超過）
- `400 Bad Request`: `{"error": "invalid_step_index"}` / `{"error": "invalid_part_index"}` / `{"error": "empty_step"}` / `{"error": "unsupported_model"}`
- `404 Not Found`: `{"error": "message_not_found"}`
- `409 Conflict`: `{"error": "message_has_no_content"}` / `{"error": "simplified_limit_reached"}` / `{"error": "too_many_steps"}`
- `413 Payload Too Large`: `{"error": "step_too_large"}`
- `502 Bad Gateway`: `{"error": "simplify_failed"}` / `{"error": "empty_result"}`

---

## 3. モデル呼び出しディスパッチャ（`_call_math_steps_simplify_model`）

ステップ簡略化では、会話で使用されたプロバイダーとモデルに合わせて、`server/routes_math_steps.py`（L144-L205）が適切なSDK・クライアントへ振り分けます。

| プロバイダー | SDK / 接続方式 | 呼び出しメソッド | 認証情報 |
|---|---|---|---|
| **Gemini** | `google-genai` SDK | `client.models.generate_content` | ユーザーのGemini APIキー（無料枠スイッチ `gemini_free_tier` 対応）または Vertex AI 認証 |
| **Anthropic** | `anthropic` SDK | `client.messages.create` | ユーザーの Anthropic API キー |
| **DeepSeek** | `openai` SDK 互換 | `client.chat.completions.create` (baseURL: `api.deepseek.com`) | ユーザーの DeepSeek API キー |
| **xAI** | `openai` SDK 互換 | `client.chat.completions.create` (baseURL: xAI API Host) | ユーザーの xAI API キー |
| **Z.AI** | `openai` SDK 互換 | `client.chat.completions.create` (baseURL: `api.z.ai/api/paas/v4`) | ユーザーの Z.AI API キー |
| **Kimi** | `openai` SDK 互換 | `client.chat.completions.create` (baseURL: `api.moonshot.ai/v1`) | ユーザーの Moonshot API キー |
| **OpenAI** | `openai` SDK | `client.responses.create` または completions | ユーザーの OpenAI API キー |

### 対象モデルの判定規約（`_is_math_steps_text_model`）
モデル名に以下のキーワードを含む特殊モデルは、ステップ解説の再生成対象から除外されます（L207-L217）：
`image`, `video`, `tts`, `audio`, `native-audio`, `embedding`, `transcribe`, `realtime`, `ocr`, `music`, `voice`

---

## 4. トークン使用量（Usage）とコスト追跡

「もっとやさしく」の再生成は追加のLLM推論を伴うため、コストの不透明化を防ぐ目的で使用量がメッセージに追跡記録されます（L122-L142）。

1. `ModelUsageCall` インスタンスを作成し、プロバイダーから返された `usage_metadata` や `usage` オブジェクトを監視。
2. 処理完了後、`call.snapshot()` を取得。
3. アシスタントメッセージの `thought_data`（JSONメタデータ）内にある `pricing_calls` リストへ、再生成ごとのスナップショットを追加保存。
4. これにより、チャット画面の「回答処理時間・トークン内訳」モーダルにおいて、初回生成分とステップ再生成分の両方のコスト・トークン数が合算および内訳表示されます。
