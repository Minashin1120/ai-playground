# モデル・料金・プロバイダー

モデルの追加・切替、入力計測、費用予想、Batch、PromptCacheを変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_deepseek_v4_flash_0731.py](../test_deepseek_v4_flash_0731.py) | deepseek / v4 / flash / 0731 |
| [test_deepseek_v4_flash_vision_exp.py](../test_deepseek_v4_flash_vision_exp.py) | deepseek / v4 / flash / vision / exp |
| [test_default_model_stability_regressions.py](../test_default_model_stability_regressions.py) | 既定モデルの安定性 |
| [test_gemini_batch_regressions.py](../test_gemini_batch_regressions.py) | gemini / batch |
| [test_gemini_latest_models.py](../test_gemini_latest_models.py) | gemini / latest / models |
| [test_gemini_media_token_estimate_regressions.py](../test_gemini_media_token_estimate_regressions.py) | gemini / media / token / estimate |
| [test_gemini_new_models.py](../test_gemini_new_models.py) | gemini / new / models |
| [test_gpt_5_6_family.py](../test_gpt_5_6_family.py) | gpt / 5 / 6 / family |
| [test_gpt_6_family.py](../test_gpt_6_family.py) | gpt / 6 / family |
| [test_grok_45_46_family.py](../test_grok_45_46_family.py) | grok / 45 / 46 / family |
| [test_model_cost_estimates.py](../test_model_cost_estimates.py) | 回答ごとの料金・長文の境界・分岐・公式入力使用量 |
| [test_no_hardcoded_model_prompts_regressions.py](../test_no_hardcoded_model_prompts_regressions.py) | モデルへの固定指示の直書き防止 |
| [test_openai_batch_regressions.py](../test_openai_batch_regressions.py) | openai / batch |
| [test_prompt_cache_clear.py](../test_prompt_cache_clear.py) | prompt / cache / clear |
| [test_provider_token_estimation.py](../test_provider_token_estimation.py) | 事業者の入力計測APIと概算への切替 |
| [test_user_system_prompt_sync_regressions.py](../test_user_system_prompt_sync_regressions.py) | user / system / prompt / sync |
| [test_xai_batch_regressions.py](../test_xai_batch_regressions.py) | xai / batch |
| [test_xai_model_coverage.py](../test_xai_model_coverage.py) | xai / model / coverage |

## 入力計測・費用予想の契約

入力見積と確定した使用量を混ぜません。API計測が失敗したときは概算として扱い、欠損をゼロ使用量にしません。費用予想は回答ごとの保存済みモデルを使い、現在のモデル選択では再計算しません。長文料金は各リクエストの入力で判定し、境界の「超」と「以上」を区別します。Thinkingを出力に二重加算しません。

料金表の正本は `static/model_pricing.json`、Android用の同一データは `android/app/src/main/assets/model-pricing.json` です。公式資料の確認日・参照先・料金を更新し、フロントエンドビルドで同期します。料金や使用量が不明な回答を無料として合算しません。

モデルへの新しい指示文は自動注入システムプロンプトの設定項目にします。固定文言をコードへ追加したり、直書き禁止テストの許可リストを拡大して回避したりしません。

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。
