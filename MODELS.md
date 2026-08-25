# 対応モデル (V4.8.856)

全対応モデルの一覧です。モデル定義の一次ソースは `static/js/chat_core.v4.8.856.js` の `MODELS` 配列です。
モデル選択モーダルには各モデルの公式 API 価格（`price` フィールド）が表示されます。

**価格最終確認:** 2026-08-15
**出典:** OpenAI / Google Gemini / Anthropic / DeepSeek / xAI / Mistral 公式 pricing ページ

---

## 凡例

- **ID**: APIリクエスト時に使用されるモデル識別子
- **Provider**: OpenAI / Google Gemini / xAI Grok / Anthropic / DeepSeek / Mistral
- **Status**: Active（現行） / Deprecated（非推奨・UI非表示だが過去スレッド互換性維持）
- **料金**: モーダル表示用の要約（1M = 100万トークン）。キャッシュ価格は省略する場合あり
- **Agentic View**: 対応バッジがあるモデルは、コード実行による画像クロップを再観察しながら推論を継続可能

---

## 価格サマリ（アクティブモデル主要項目）

| 系統 | 代表モデル | 料金（要約） |
|------|-----------|-------------|
| Gemini 3.7 | gemini-3.7-flash | In $0.75/1M, Out $3.75/1M（導入価格） |
| Gemini 3.6 | gemini-3.6-flash | In $1.50/1M, Out $7.50/1M |
| Gemini 3.5 | gemini-3.5-flash | In $1.50/1M, Out $9.00/1M |
| Gemini 3.5 | gemini-3.5-flash-lite | In $0.30/1M, Out $2.50/1M |
| Gemini 3.1 | gemini-3.1-flash-lite | In $0.25/1M, Out $1.50/1M |
| Gemini 3.x | gemini-3.1-pro-preview | In $2.00/1M, Out $12.00/1M (≤200k) |
| Gemini 2.5 | gemini-2.5-flash-lite | In $0.10/1M, Out $0.40/1M |
| Gemini 2.5 | gemini-2.5-flash | In $0.30/1M, Out $2.50/1M |
| Gemini Image | gemini-3.1-flash-lite-image | In $0.25/1M, Text/Thinking Out $1.50/1M, Image Out $30/1M（$0.0336/1K image） |
| OpenAI GPT | gpt-5.6-sol | In $5.00/1M, Cached $0.50/1M, Out $30.00/1M（>272K: In $10.00/1M, Out $45.00/1M） |
| OpenAI GPT | gpt-5.6-terra | In $2.00/1M, Cached $0.20/1M, Out $12.00/1M（>272K: In $4.00/1M, Out $18.00/1M） |
| OpenAI GPT | gpt-5.6-luna | In $0.20/1M, Cached $0.02/1M, Out $1.20/1M（>272K: In $0.40/1M, Out $1.80/1M） |
| OpenAI GPT | gpt-5.5 | In $5.00/1M, Out $30.00/1M |
| OpenAI GPT | gpt-5.5-pro | In $30.00/1M, Out $180.00/1M |
| OpenAI GPT | gpt-5.4 | In $2.50/1M, Out $15.00/1M |
| OpenAI GPT | gpt-5.4-mini | In $0.75/1M, Out $4.50/1M |
| OpenAI GPT | gpt-5.4-nano | In $0.20/1M, Out $1.25/1M |
| OpenAI GPT | gpt-5.2 | In $1.75/1M, Out $14.00/1M |
| OpenAI GPT | gpt-5.1 | In $1.25/1M, Out $10.00/1M |
| OpenAI GPT | gpt-5-mini | In $0.25/1M, Out $2.00/1M |
| OpenAI GPT | gpt-4o | In $2.50/1M, Out $10.00/1M |
| OpenAI GPT | gpt-4o-mini | In $0.15/1M, Out $0.60/1M |
| OpenAI Transcription | gpt-transcribe | $0.0045 / min |
| OpenAI Transcription | gpt-live-transcribe | $0.017 / min |
| Claude | claude-opus-4-6 | In $5.00/1M, Out $25.00/1M |
| Claude | claude-sonnet-4-6 | In $3.00/1M, Out $15.00/1M |
| DeepSeek | deepseek-v4-flash | In $0.14/1M (miss), Out $0.28/1M |
| DeepSeek | deepseek-v4-pro | In $0.435/1M (miss), Out $0.87/1M |
| xAI Grok | grok-4.6 / grok-4.5 | In $2.00/1M, Out $6.00/1M（200k超: $4.00/$12.00） |
| xAI Grok | grok-4.3 / grok-4.20 系 | In $1.25/1M, Out $2.50/1M |
| xAI Grok | grok-build-0.1 | In $1.00/1M, Out $2.00/1M |
| xAI Voice | grok-voice-* | $0.05 / min |
| xAI TTS | grok-tts | $15.00 / 1M chars |
| xAI Imagine | grok-imagine-image-2.0 | from $0.04 / image |
| xAI Imagine | grok-imagine-image | $0.02 / image |
| xAI Imagine | grok-imagine-image-quality | $0.05 / image |
| xAI Imagine | grok-imagine-video | $0.05 / second |
| Mistral OCR | mistral-ocr-4-0 | $4 / 1,000 pages（注釈付き $5 / 1,000 pages） |

※ 画像・音声・Realtime モデルの詳細は `MODELS` 配列の `price` を参照。

---

*最終更新: 2026-08-26 (V4.8.856)*
*ソース: `static/js/chat_core.v4.8.856.js` MODELS配列*
