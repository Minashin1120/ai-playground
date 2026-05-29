# Supported Models (V4.8.558)

全対応モデルの一覧です。モデル定義の一次ソースは `static/js/chat_core.v4.8.558.js` の `MODELS` 配列です。

---

## 凡例

- **ID**: APIリクエスト時に使用されるモデル識別子
- **Provider**: OpenAI / Google Gemini / xAI Grok / Anthropic / DeepSeek
- **Status**: Active（現行） / Deprecated（非推奨・UI非表示だが過去スレッド互換性維持）

---

## OpenAI GPT

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gpt-5.5` | GPT-5.5 | 最上位実験的モデル（アカウント依存） | 未公開 | Active |
| `gpt-5.5-pro` | GPT-5.5 Pro | GPT-5.5 高性能版 | 未公開 | Active |
| `gpt-5.5-mini` | GPT-5.5 mini | GPT-5.5 低コスト版 | 未公開 | Active |
| `gpt-5.5-nano` | GPT-5.5 nano | GPT-5.5 最速版 | 未公開 | Active |
| `gpt-5.4` | GPT-5.4 | 実験的モデル（アカウント依存） | 未公開 | Active |
| `gpt-5.4-pro` | GPT-5.4 Pro | GPT-5.4 高性能版 | 未公開 | Active |
| `gpt-5.4-mini` | GPT-5.4 mini | GPT-5.4 低コスト版 | 未公開 | Active |
| `gpt-5.4-nano` | GPT-5.4 nano | GPT-5.4 最速版 | 未公開 | Active |
| `gpt-5.2` | GPT-5.2 (Responses API) | 高推論能力モデル | In $1.75/1M, Out $14/1M | Active |
| `gpt-5.1` | GPT-5.1 | 高インテリジェンス | In $1.25/1M, Out $10/1M | Active |
| `gpt-5-mini` | GPT-5 mini | 小型効率モデル | In $0.25/1M, Out $2/1M | Active |
| `gpt-5-search-api` | GPT-5 Search (API) | 検索最適化モデル | — | Active |
| `gpt-4o` | GPT-4o | マルチモーダル主力モデル | In $2.50/1M, Out $10.00/1M | Active |
| `gpt-4o-mini` | GPT-4o mini | 高速低コストモデル | In $0.15/1M, Out $0.60/1M | Active |

---

## OpenAI Image Gen

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gpt-image-2` | GPT Image 2 | 最新画像生成・編集（最先端） | Text In $5/1M; Image In $8/1M; Image Out $30/1M | Active |
| `gpt-image-1.5` | GPT Image 1.5 | 前世代主力画像モデル | Text In $5/1M, Text Out $10/1M; Image Out $32/1M | Active |
| `gpt-image-1` | GPT Image 1 | 標準画質 | Text In $5/1M; Image Out $40/1M | Active |
| `gpt-image-1-mini` | GPT Image 1 Mini | 高速低解像度 | Text In $2/1M; Image Out $8/1M | Active |

---

## OpenAI Audio (TTS/STT)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gpt-4o-mini-tts` | GPT-4o Mini TTS | OpenAI テキスト読み上げ | Text In $0.60/1M, Audio Out $12/1M | Active |
| `gpt-4o-mini-transcribe` | GPT-4o Mini Transcribe | OpenAI 音声認識（標準） | — | Active |
| `gpt-4o-transcribe` | GPT-4o Transcribe | OpenAI 高精度音声認識 | — | Active |
| `gpt-4o-transcribe-diarize` | GPT-4o Transcribe (Diarize) | 話者分離付き音声認識 | — | Active |
| `whisper-1` | Whisper-1 | OpenAI Whisper 音声認識 | — | Active |

---

## OpenAI Realtime Audio (STS)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gpt-realtime-2` | OpenAI Realtime 2 | 最上位音声対話モデル | Audio In $32/1M, Audio Out $64/1M | Active |
| `gpt-realtime-1.5` | OpenAI Realtime 1.5 | 最新音声対話モデル | — | Active |
| `gpt-realtime` | OpenAI Realtime | 音声対話モデル | — | Active |
| `gpt-realtime-mini` | OpenAI Realtime Mini | 低遅延小型音声対話 | — | Active |
| `gpt-realtime-translate` | OpenAI Realtime Translate | ストリーミング音声翻訳 | $0.034/分 | Active |
| `gpt-realtime-whisper` | OpenAI Realtime Whisper | ストリーミング音声認識 | $0.017/分 | Active |

---

## Google Gemini (Chat)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro | 次世代ネイティブマルチモーダル | In $2.00/1M, Out $12.00/1M (<=200k) | Active |
| `gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash-Lite | 最速・低コスト | In $0.25/1M, Out $1.50/1M | Active |
| `gemini-3-flash-preview` | Gemini 3.0 Flash | 高速・低コスト | In $0.50/1M, Out $3.00/1M | Active |
| `gemini-3-pro-preview` | Gemini 3.0 Pro | 複雑タスク向け高性能 | In $2.00/1M, Out $12.00/1M (<=200k) | Active |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash-Lite | 最速・低コスト | — | Active |
| `gemini-2.5-flash` | Gemini 2.5 Flash | バランス性能 | In $0.30/1M, Out $2.50/1M | Active |

---

## Google Gemini Image (Nano Banana)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gemini-2.5-flash-image` | Nano Banana | 高速画像生成 | In $0.30/1M, Out $0.039/画像 | Active |
| `gemini-3.1-flash-image-preview` | Nano Banana 2 | Gemini 3.1 Flash 画像生成 | Preview（価格変動あり） | Active |
| `gemini-3-pro-image-preview` | Nano Banana Pro | 高品質画像生成 | In $2.00/1M, Out $0.134(1K/2K) or $0.24(4K) | Active |

---

## Google Gemini Audio (TTS/Live)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `gemini-3.1-flash-tts-preview` | Gemini 3.1 Flash TTS | Google TTS（Preview） | Text In $0.25/1M, Audio Out $5/1M | Active |
| `gemini-2.5-flash-preview-tts` | Gemini 2.5 Flash TTS | Google TTS（Preview） | Text In $0.50/1M, Audio Out $10/1M | Active |
| `gemini-2.5-pro-preview-tts` | Gemini 2.5 Pro TTS | Google TTS Pro（Preview） | Text In $1.00/1M, Audio Out $20/1M | Active |
| `google-tts-studio` | Google TTS (Studio) | 高品質スタジオ音声 | $160/100万文字 | Active |
| `google-tts-neural` | Google TTS (Neural2) | 標準ニューラル音声 | $16/100万文字 | Active |
| `gemini-2.5-flash-native-audio-preview-12-2025` | Gemini 2.5 Flash Native Audio (Live) | Google Live 音声対話 | — | Active |
| `gemini-3.1-flash-live-preview` | Gemini 3.1 Flash Live | Google Live 音声対話 | — | Active |

---

## xAI Grok (Chat)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `grok-4.3` | Grok 4.3 | 最速・高インテリジェンス主力モデル | In $1.25/1M, Out $2.50/1M | Active |
| `grok-build-0.1` | Grok Build 0.1 (Coding) | 高速エージェントコーディングモデル（視覚・推論対応） | In $1.00/1M, Out $2.00/1M | Active |
| `grok-4.20-reasoning` | Grok 4.20 (Reasoning) | 主力推論モデル | In $2.00/1M, Out $6.00/1M | Active |
| `grok-4.20-non-reasoning` | Grok 4.20 (Non-Reasoning) | 主力標準モデル | In $2.00/1M, Out $6.00/1M | Active |
| `grok-4.20-multi-agent` | Grok 4.20 Multi-Agent | エージェント型主力モデル | In $2.00/1M, Out $6.00/1M | Active |
| `grok-4-1-fast-reasoning` | Grok 4.1 Fast (Reasoning) | 高速推論モデル | In $0.20/1M, Out $0.50/1M | Deprecated |
| `grok-4-1-fast-non-reasoning` | Grok 4.1 Fast (Non-Reasoning) | 高速標準モデル | In $0.20/1M, Out $0.50/1M | Deprecated |
| `grok-4-fast-reasoning` | Grok 4 Fast (Reasoning) | 旧世代推論モデル | In $0.20/1M, Out $0.50/1M | Deprecated |
| `grok-4-fast-non-reasoning` | Grok 4 Fast (Non-Reasoning) | 旧世代標準モデル | In $0.20/1M, Out $0.50/1M | Deprecated |

---

## xAI Grok Voice (Realtime STS)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `grok-voice-latest` | Grok Voice Latest | 常に最新フラッグシップを指す推奨エイリアス | $0.05/分 | Active |
| `grok-voice-think-fast-1.0` | Grok Voice Think Fast 1.0 | 現行主力音声モデル（高度推論） | $0.05/分 | Active |
| `grok-voice-fast-1.0` | Grok Voice Fast 1.0 | 旧世代音声モデル | $0.05/分 | Deprecated |
| `grok-voice-agent` | Grok Voice Agent | 音声エージェントAPI | — | Deprecated |

---

## xAI Grok TTS

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `grok-tts` | Grok TTS | xAI テキスト読み上げ（voice/speed/language 対応） | $15/100万文字 | Active |

---

## xAI Grok Imagine (Image/Video)

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `grok-imagine-image-quality` | Grok Imagine Image Quality | 次世代画像生成（1K/2K対応） | $0.05(1K) / $0.07(2K) | Active |
| `grok-imagine-image` | Grok Imagine Image | 標準画像生成 | $0.02/画像 | Active |
| `grok-imagine-image-pro` | Grok Imagine Image Pro | 高品質画像生成 | $0.07/画像 | Active |
| `grok-imagine-video` | Grok Imagine Video | 動画生成 | $0.05/秒 | Active |

---

## Anthropic Claude

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `claude-opus-4-6` | Claude Opus 4.6 | 深い推論・複雑タスク向け最上位モデル | — | Active |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | 速度と知能のバランス（Extended Thinking対応） | — | Active |

---

## DeepSeek V4

| ID | 名称 | 説明 | 料金 | Status |
|----|------|------|------|--------|
| `deepseek-v4-flash` | DeepSeek V4 Flash | 高速モデル（思考/非思考モード切替可） | — | Active |
| `deepseek-v4-pro` | DeepSeek V4 Pro | 高性能モデル（思考/非思考モード切替可） | — | Active |

---

## 集計

| カテゴリ | アクティブ | 非推奨 | 合計 |
|----------|-----------|--------|------|
| OpenAI GPT | 14 | 0 | 14 |
| OpenAI Image Gen | 4 | 0 | 4 |
| OpenAI Audio (TTS/STT) | 5 | 0 | 5 |
| OpenAI Realtime (STS) | 6 | 0 | 6 |
| Google Gemini (Chat) | 6 | 0 | 6 |
| Google Gemini Image | 3 | 0 | 3 |
| Google Gemini Audio | 7 | 0 | 7 |
| xAI Grok (Chat) | 5 | 4 | 9 |
| xAI Grok Voice | 2 | 2 | 4 |
| xAI Grok TTS | 1 | 0 | 1 |
| xAI Grok Imagine | 4 | 0 | 4 |
| Anthropic Claude | 2 | 0 | 2 |
| DeepSeek V4 | 2 | 0 | 2 |
| **合計** | **61** | **6** | **67** |

---

*最終更新: 2026-05-27 (V4.8.558)*
*ソース: `static/js/chat_core.v4.8.558.js` MODELS配列 / `app.py` STTモデル定義*
