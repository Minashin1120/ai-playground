# 検証・公開・運用

版更新、検証、公開、記録スクリプトと実行環境を変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_documentation_regressions.py](../test_documentation_regressions.py) | documentation |
| [test_extract_before_extend.py](../test_extract_before_extend.py) | 予算超過ファイルを増やさない判定（`check-extract`、`~/docs/file_layout_rules.md`） |
| [test_file_size_budget.py](../test_file_size_budget.py) | ファイルサイズ予算・例外リスト（`~/docs/context_cost_reduction_plan.md`） |
| [test_key_rotation_regressions.py](../test_key_rotation_regressions.py) | key / rotation |
| [test_prepare_preflight.py](../test_prepare_preflight.py) | 版更新前の検証順序・失敗時の無変更・スキップの範囲・文書索引 |
| [test_readme_change_units.py](../test_readme_change_units.py) | 分割READMEの変更単位表（`主な内容` / `いつ開くか`）と `server/` `android/` の被覆（`~/docs/file_layout_rules.md`） |
| [test_production_schema_guard_regressions.py](../test_production_schema_guard_regressions.py) | production / schema / guard |
| [test_release_scripts.py](../test_release_scripts.py) | 版・資産・changelog・記録範囲・公開スクリプト契約 |
| [test_server_memory_regressions.py](../test_server_memory_regressions.py) | server / memory |
| [test_system_logs_removed.py](../test_system_logs_removed.py) | system / logs / removed |

## 版更新・記録の契約

`test_prepare_preflight.py` は実際の `prepare_version.sh` と補助シェルを一時ディレクトリへコピーし、ビルド・検証・版更新だけをスタブにして順序と失敗時の状態を確認します。実リポジトリの版上げ、公開、Git操作、サービス再起動は実行しません。

Webの版更新前に全回帰テスト、更新後に資産・構文の確認を行います。スキップが明示されても構文・資産確認は残します。公開・記録は許可したパスだけを扱い、運用変更の経路にWeb・Androidの配布物を混ぜません。

スクリプトの入口と補助ファイルは [scripts/README.md](../../scripts/README.md)、実行順序は [verification.md](../../scripts/docs/verification.md) から特定します。

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。
