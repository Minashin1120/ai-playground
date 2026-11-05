# 会話・通信・性能

生成、保存、ストリーム、再接続、履歴、性能を変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_chat_error_persist_regressions.py](../test_chat_error_persist_regressions.py) | chat / error / persist |
| [test_connection_status_regressions.py](../test_connection_status_regressions.py) | connection / status |
| [test_generation_restart_resilience.py](../test_generation_restart_resilience.py) | generation / restart / resilience |
| [test_history_document_text_regressions.py](../test_history_document_text_regressions.py) | history / document / text |
| [test_network_read_deadline.py](../test_network_read_deadline.py) | network / read / deadline |
| [test_performance_regressions.py](../test_performance_regressions.py) | performance |
| [test_prompt_prefix_stability.py](../test_prompt_prefix_stability.py) | prompt / prefix / stability |
| [test_request_timing_regressions.py](../test_request_timing_regressions.py) | request / timing |
| [test_stream_late_subscriber_regressions.py](../test_stream_late_subscriber_regressions.py) | stream / late / subscriber |
| [test_stream_transport_regressions.py](../test_stream_transport_regressions.py) | stream / transport |

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。
