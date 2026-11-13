# モデル・料金・プロバイダー

モデルの追加・切替、入力計測、費用予想、Batch、PromptCacheを変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_deepseek_v4_flash_0731.py](../test_deepseek_v4_flash_0731.py) | deepseek / v4 / flash / 0731 |
| [test_deepseek_v4_flash_vision_exp.py](../test_deepseek_v4_flash_vision_exp.py) | deepseek / v4 / flash / vision / exp |
| [test_deepseek_search.py](../test_deepseek_search.py) | DeepSeek Search API、Search有効時の検索、検索料金 |
| [test_default_model_stability_regressions.py](../test_default_model_stability_regressions.py) | 既定モデルの安定性 |
| [test_gemini_batch_regressions.py](../test_gemini_batch_regressions.py) | gemini / batch |
| [test_gemini_latest_models.py](../test_gemini_latest_models.py) | gemini / latest / models |
| [test_gemini_media_token_estimate_regressions.py](../test_gemini_media_token_estimate_regressions.py) | gemini / media / token / estimate |
| [test_gemini_new_models.py](../test_gemini_new_models.py) | gemini / new / models |
| [test_gemini_free_tier_regressions.py](../test_gemini_free_tier_regressions.py) | Gemini無料枠スイッチと有料・無料キーの分離 |
| [test_gpt_5_6_family.py](../test_gpt_5_6_family.py) | gpt / 5 / 6 / family |
| [test_gpt_6_family.py](../test_gpt_6_family.py) | gpt / 6 / family |
| [test_grok_45_46_family.py](../test_grok_45_46_family.py) | grok / 45 / 46 / family |
| [test_model_usage_ledger.py](../test_model_usage_ledger.py) | HTTP通信単位の使用量、SDK自動ツール呼び出し・再試行、累積イベント・再取得の重複防止 |
| [test_cache_tool_costs.py](../test_cache_tool_costs.py) | キャッシュ読込・TTL別書込、トークン詳細向けキャッシュ集計、検索追加料金、xAI実請求、数値のみの保存・同期 |
| [test_ideogram_costs.py](../test_ideogram_costs.py) | Ideogram全5モデル・速度・削除API・未公表料金の下限・旧回答の速度復元 |
| [test_cost_request_accounting.py](../test_cost_request_accounting.py) | API呼び出し別の長文料金・モデル・Thinking・Ideogram 4.5画像単価 |
| [test_model_cost_estimates.py](../test_model_cost_estimates.py) | 回答ごとの料金・長文の境界・分岐・公式入力使用量 |
| [test_no_hardcoded_model_prompts_regressions.py](../test_no_hardcoded_model_prompts_regressions.py) | モデルへの固定指示の直書き防止 |
| [test_openai_batch_regressions.py](../test_openai_batch_regressions.py) | openai / batch |
| [test_prompt_cache_clear.py](../test_prompt_cache_clear.py) | prompt / cache / clear |
| [test_provider_token_estimation.py](../test_provider_token_estimation.py) | 事業者の入力計測APIと概算への切替 |
| [test_skills_prompt_tool_regressions.py](../test_skills_prompt_tool_regressions.py) | Skillカタログ注入、プロンプトバーのSkillsスイッチ（`enable_skills`）、`use_skill` のプロバイダ別ツール |
| [test_user_system_prompt_sync_regressions.py](../test_user_system_prompt_sync_regressions.py) | user / system / prompt / sync |
| [test_xai_batch_regressions.py](../test_xai_batch_regressions.py) | xai / batch |
| [test_xai_model_coverage.py](../test_xai_model_coverage.py) | xai / model / coverage |

## 入力計測・費用予想の契約

入力見積と確定した使用量を混ぜません。API計測が失敗したときは概算として扱い、欠損をゼロ使用量にしません。費用予想は回答ごとの保存済みモデルを使い、現在のモデル選択では再計算しません。長文料金は各リクエストの入力で判定し、境界の「超」と「以上」を区別します。Thinkingを出力に二重加算しません。

料金表の正本は `static/model_pricing.json`、Android用の同一データは `android/app/src/main/assets/model-pricing.json` です。公式資料の確認日・参照先・料金を更新し、フロントエンドビルドで同期します。料金や使用量が不明な回答を無料として合算しません。

モデルへの新しい指示文は自動注入システムプロンプトの設定項目にします。固定文言をコードへ追加したり、直書き禁止テストの許可リストを拡大して回避したりしません。

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。

モデルAPIの使用量は回答内の `pricing_calls` に通信ごとに保存します。同じストリームの累積usageは更新し、別リクエストは加算します。長文料金は各リクエストで判定します。再取得のGETは生成回数に加えず、使用量未取得の通信は一部未算入にします。Thinkingは出力に含めたまま内訳へ記録します。Ideogram 4.5は生成と編集の品質別単価×実際の画像枚数で検証します。

キャッシュ読込・書込は通常入力から差し引いて別単価で計算します。OpenAIは30分／12時間、Claudeは5分／1時間の書込を区別し、不明なTTLは範囲と一部未算入にします。トークン詳細では、保存済みの `pricing_calls` からキャッシュ読込とTTL別の書込を集計して表示し、キャッシュ情報のない回答では行を出しません。検索の有効化だけで課金せず、実際の課金対象呼び出しを数え、ストリームイベントの重複を加算しません。OpenAIのページ閲覧・ページ内検索をWeb検索料金に重ねません。Batch割引をツール料金へ適用しません。Batchのキャッシュ読込は事業者ごとの公表値を使い、Geminiの一部モデルのように通常単価のままの場合は割引しません。xAIの `cost_in_usd_ticks` はキャッシュ・ツールを含むため、報告額を優先し他料金を二重加算しません。取得できない保存時間・実行時間は一部未算入とし、検索文・URL・ツールIDを使用量メタデータへ保存しません。

Ideogramは選択可能な5モデルを対象とし、速度を保存・同期します。背景削除は専用API単価、オブジェクト削除と2a／2.0の単価未公表設定は公表最安値を下限にして「以上・一部未算入」と表示します。古い呼び出し記録の速度は保存済みの親プロンプトからだけ復元します。
