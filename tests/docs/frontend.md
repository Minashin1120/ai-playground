# Web画面・JS部品

画面、テンプレート、部品の追加・改名、操作・表示を変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_ai_settings_inspection_regressions.py](../test_ai_settings_inspection_regressions.py) | ai / settings / inspection |
| [test_chat_auto_scroll_regressions.py](../test_chat_auto_scroll_regressions.py) | chat / auto / scroll |
| [test_chat_core_bundles.py](../test_chat_core_bundles.py) | 結合・分割・圧縮JSのChromium読み込み |
| [test_chat_core_parts_regressions.py](../test_chat_core_parts_regressions.py) | 部品名の連番・連結結果・大きさ |
| [test_chat_loading_regressions.py](../test_chat_loading_regressions.py) | chat / loading |
| [test_edge_to_edge_regressions.py](../test_edge_to_edge_regressions.py) | edge / to / edge |
| [test_edit_cancel_settings_regressions.py](../test_edit_cancel_settings_regressions.py) | edit / cancel / settings |
| [test_edit_keeps_manual_model_regressions.py](../test_edit_keeps_manual_model_regressions.py) | edit / keeps / manual / model |
| [test_feedback_ui_requests.py](../test_feedback_ui_requests.py) | feedback / ui / requests |
| [test_minimal_prompt_mode_regressions.py](../test_minimal_prompt_mode_regressions.py) | minimal / prompt / mode |
| [test_modal_design_system_regressions.py](../test_modal_design_system_regressions.py) | モーダルの共通構造・共通CSS・ライトテーマ上書き |
| [test_modal_performance_regressions.py](../test_modal_performance_regressions.py) | modal / performance |
| [test_processing_time_regressions.py](../test_processing_time_regressions.py) | 回答処理時間の表示・内訳モーダル |
| [test_progress_spinner_regressions.py](../test_progress_spinner_regressions.py) | progress / spinner |
| [test_prompt_options_record_regressions.py](../test_prompt_options_record_regressions.py) | prompt / options / record |
| [test_prompt_send_motion.py](../test_prompt_send_motion.py) | 送信時の入力欄と吹き出しのアニメーション |
| [test_pwa_regressions.py](../test_pwa_regressions.py) | pwa |
| [test_python_exec_ui_regressions.py](../test_python_exec_ui_regressions.py) | python / exec / ui |
| [test_quote_source_regressions.py](../test_quote_source_regressions.py) | quote / source |
| [test_send_button_scope_regressions.py](../test_send_button_scope_regressions.py) | send / button / scope |
| [test_settings_flash_leak_regressions.py](../test_settings_flash_leak_regressions.py) | settings / flash / leak |
| [test_settings_model_picker_regressions.py](../test_settings_model_picker_regressions.py) | モデル選択UI・モデル情報 |
| [test_skills_ui_regressions.py](../test_skills_ui_regressions.py) | Skillのサイドバー・モーダル・`@`候補・プロンプトバーのSkillsスイッチ・CSS・旧Gem URL互換 |
| [test_settings_redesign_regressions.py](../test_settings_redesign_regressions.py) | settings / redesign |
| [test_settings_tabs_regressions.py](../test_settings_tabs_regressions.py) | settings / tabs |
| [test_slash_settings_mode_regressions.py](../test_slash_settings_mode_regressions.py) | slash / settings / mode |
| [test_thread_pull_to_refresh_regressions.py](../test_thread_pull_to_refresh_regressions.py) | thread / pull / to / refresh |
| [test_ui_design_refresh_regressions.py](../test_ui_design_refresh_regressions.py) | ui / design / refresh |
| [test_welcome_quick_start_regressions.py](../test_welcome_quick_start_regressions.py) | welcome / quick / start |

## JS部品を追加・改名するときの契約

- `chat_core.part01_*.js` から連番にします。`part03a` のような枝番号は使いません。番号順の連結が配信ソースと完全一致することを検査します。
- `static/js/chat_core_parts/README.md` と実際の担当範囲・読み込み順を一致させます。部品を編集して `scripts/build_frontend.sh` で生成し、結合ソースと圧縮ファイルは手で編集しません。
- 部品は同じグローバル名前空間を共有します。名前の重複と、先に読み込む配信ファイルが後の配信ファイルの関数を初期化中に呼ぶことを避けます。分割後の各配信ファイルも単独で構文が通る必要があります。
- 行数は検証スクリプトが1,000行超で警告し、部品回帰テストと `CHAT_CORE_PART_MAX_LINES` が1,500行超で失敗します。再分割の正本手順に従い、READMEを同時更新します。行数確認だけを目的に全ファイルを読んだり計測したりしません。

部品構成を変更した場合は `test_chat_core_parts_regressions.py` と `test_chat_core_bundles.py`、機能本体の関連テストを選びます。ChromiumやPlaywrightがない環境ではブラウザー検証がスキップされるため、成功と区別して報告します。

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。

費用予想の内訳計算は連番の末尾（現在は `chat_core.part29_estimateconversationcost.js`）に分離します。Grokの添付のみ送信のテストは `chat_session.stream()` を抽出境界にし、直前へ使用量の記録などが追加されても空本文の検証を維持します。
