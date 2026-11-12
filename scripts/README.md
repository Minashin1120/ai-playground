# 保守・検証スクリプトの入口

| ファイル・文書 | 主な内容 | いつ開くか |
|---|---|---|
| `docs/verification.md` | 版更新前のテストと、版更新後の構文・資産確認 | 検証順序・失敗時・スキップを確認するとき |
| `docs/release_workflow.md` | 変更対象ごとの記録・公開経路 | 実行する公開経路を選ぶとき |
| `docs/android_checks.md` | Android版番号・履歴・省略指定・Actionsの完了条件 | Androidの記録・配布条件を確認するとき |
| `verify_changes.sh` | 現時点の構文・Android生成物の同期・版付き資産・文書・回帰テスト、任意の公開URL確認。成功時は1行、失敗時はログ末尾20行だけを表示（`--verbose` で各チェックを表示） | ファイルやサービスを変更せず検証するとき |
| `show_versions.sh` | `SYSTEM_VERSION` / `APP_VERSION` と `_SERVER_PARTS` の読み込み順だけを表示 | 版番号やサーバー部品の順序を確認するとき |
| `show_regions.sh` | `# region:` / `// region:` マーカーの領域名と行範囲だけを表示 | 巨大ファイルの全体を読まずに必要な領域を選ぶとき |
| `sync_android_assets.sh` | WebソースからAndroidの生成物（serverless defaults・モデルカタログ・アイコン）を再生成 | Web/Androidの生成物を同期するとき。`prepare_version.sh` と `record_changes.sh --target android` が自動実行する |
| `prepare_version.sh` | 引数・notes確認、版更新前検証、版番号・changelog更新、生成資産の確認 | Web公開用の版を準備するとき |
| `_prepare_checks.sh` | 現在の版のビルドと検証、新版資産のビルドと静的検証 | prepareの各検証段階を変更するとき |
| `publish_version.sh` | 計画、検証、再起動、キャッシュ削除、公開URL確認、許可ファイルのcommit・tag・push | Web版を公開するとき |
| `record_changes.sh` | Android・運用変更を対象限定でcommit・push。Androidでは先に生成物を自動同期 | Web版上げ・再起動を伴わず記録するとき |
| `deploy_server.sh` | サーバーだけの変更の検証・再起動・公開URL確認・記録 | Web UI・Android・運用スクリプトを含まないサーバー変更を反映するとき |
| `restart_services.sh` | Webと2ワーカーの再起動、PID変更と応答確認 | 参照デプロイ構成の再起動を変更するとき |
| `wait_for_restart_headroom.sh` | 空きメモリ・memory/I/O PSI・swap-inの監視 | 再起動前のリソース確認を変更するとき |
| `purge_cloudflare_cache.sh` | 設定したホストのCDNキャッシュ削除 | 再起動成功後のキャッシュ処理を変更するとき |
| `build_frontend.sh` | JS部品の連結・配信用分割・圧縮、CSS圧縮、共通料金表のAndroid同期 | JS部品やフロントエンド資産の編集後にビルドするとき |
| `rebuild_chat_core_parts.sh / _rebuild_chat_core_parts.js` | 肥大化したチャットコアの自動再分割とREADMEの表の生成 | 既存部品のサイズ警告を受けて再分割するとき |
| `rotate_encryption_key.py` | secret.keyのローテーション | deploy/KEY_ROTATION.mdの手順を実行するとき |
| `send_agent_feedback.py` | 別件の問題を管理者フィードバックへ登録、重複・機密らしき本文の拒否 | 今回直さない問題を見つけたとき |
| `update_feedback_status.py` | 管理APIで状態・返信・対応者を更新 | 修正済みの既知フィードバックの状態を変えるとき |
| `build_icon_subset.py` | 使用中のFont Awesomeアイコンの部分集合生成 | 同梱アイコンを更新するとき |
| `test_landing_demo_dom.js` | 軽量DOM shimによるランディング画面検証 | ランディングのDOMを変更するとき |
| `measure_landing_cdp.py / verify_landing_geometry.js` | ブラウザー上の描画・座標検証 | ランディングの配置・描画を確認するとき |
| `_release_lib.sh` | 共通のパス・短い出力・Python呼出・Android省略trailer | シェルの共通処理を変更するとき |
| `_release_common.py` | 版・資産・公開文書・notes・Git対象と記録分類の共通処理 | 該当する関数をrgで特定して変更するとき |
| `_dom_shim.js` | ランディング画面テスト用のDOM代替 | DOMテストの共通機能を変更するとき |
