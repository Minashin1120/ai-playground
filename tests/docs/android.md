# Android・モバイルAPI

モバイルAPI、同期、Androidの文言・モデル・配布を変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_android_changelog_regressions.py](../test_android_changelog_regressions.py) | Android版別changelogと配布経路 |
| [test_mobile_api.py](../test_mobile_api.py) | Android専用API |
| [test_mobile_sync_api.py](../test_mobile_sync_api.py) | 端末とサーバーの同期 |

## Android側のテスト

ネイティブ単体テストは `android/app/src/test/` の実装と同じパッケージに置きます。対象クラス名でファイルを特定し、無関係なテストは読みません。追加・改名・分割したテストは、担当機能を記載するAndroidのREADMEの対象行から到達できる状態にします。

Gradleによる単体テスト・lint・コンパイル・署名はActionsだけです。pushしたSHAとActionsの `head_sha`、Releaseのコミットを照合し、debug/release APK、レポート、署名照合、Release添付物まで確認します。Webと両方の変更は同じcommitで公開し、Android専用の重複pushをしません。

Androidの文書・配布経路は [android/README.md](../../android/README.md) と [Android検証・配布](../../scripts/docs/android_checks.md) を参照してください。

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。

`test_mobile_sync_api.py` は `pricing_calls` と `tokens_thought` の保存・再取得、任意の本文などが数値メタデータへ混ざらないことも検証します。ネイティブ側の記録・保存・HTTP経路は `ModelUsageLedgerTest.kt`、`local/ModelUsageStorageTest.kt`、`direct/DirectUsageTest.kt` をActionsで実行します。

キャッシュ・ツール料金とIdeogram全モデルの単価・速度・下限表示は `data/CacheToolCostTest.kt` と `data/IdeogramCostTest.kt` でWebと同じ契約を検証します。Batchのキャッシュ読込単価は `data/ModelCostEstimatorTest.kt` で事業者別に検証します。キャッシュのTTL別書込、32bitを超えるxAI請求ticks、検索の重複イベントを数値メタデータとして保存・同期し、本文・検索文・ツールIDは保存しません。
