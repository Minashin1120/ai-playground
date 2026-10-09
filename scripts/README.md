# 保守・検証スクリプト

このディレクトリにはサービス再起動、キャッシュ削除、ランディング画面のDOM／描画検証、版確認用のスクリプトがあります。

- `verify_changes.sh`：構文、必須の版付き資産、公開文書の版番号、回帰テストをまとめて確認する
- `prepare_version.sh`：版番号と版付きチャットJS/CSSを次の版へ進め、公開更新履歴を書き、圧縮ファイルを作り直す
- `publish_version.sh`：確認後に稼働中サービスへ反映し、キャッシュを消し、版をリポジトリへ記録する
- `record_changes.sh`：Android単独または運用ファイル単独の変更を、Web版上げ・サービス再起動なしで対象限定して記録する
- `deploy_server.sh`：`server/`・`tests/`・`worker.py`・`deploy/*.md` だけの変更を、Web版上げ・公開更新履歴・CDN削除・タグなしで、検証 → 再起動 → 公開URL確認 → 記録する
- `restart_services.sh`：再起動前のリソース余力を確認してから `ai-chat.service` と `ai-chat-worker@1..2.service` を再起動し、PID変更とWeb応答を確認（RQワーカーはメモリ負荷対策で2つのみ有効）
- `wait_for_restart_headroom.sh`：空きメモリ、memory/I/O PSI、swap-inを短時間監視し、圧迫中のサービス再起動を防ぐ内部ヘルパ
- `purge_cloudflare_cache.sh`：指定ゾーンのホストキャッシュを削除
- `build_frontend.sh`：バージョン付きチャットJS/CSSと補助スクリプトを圧縮する
- `rebuild_chat_core_parts.sh`（`_rebuild_chat_core_parts.js`）：`verify_changes.sh` がチャットコア部品の行数超過を警告したときに、部品を再分割して `static/js/chat_core_parts/README.md` の表を作り直す（概要は手で確認して直す）
- `rotate_encryption_key.py`：`secret.key` の安全なローテーション（手順は [../deploy/KEY_ROTATION.md](../deploy/KEY_ROTATION.md)）
- `build_icon_subset.py`：使用中のFont Awesomeアイコンだけを同梱する
- `test_landing_demo_dom.js`：軽量DOM shimを用いたランディング画面テスト
- `measure_landing_cdp.py`／`verify_landing_geometry.js`：ブラウザー上の描画・座標検証
- `_release_lib.sh`／`_release_common.py`／`_dom_shim.js`：リリース系スクリプトとランディングテストが共有する内部ヘルパ（単独では実行しない）

Webの画面（`templates/`・`static/`）や `app.py` を変更した場合は `prepare_version.sh` と `publish_version.sh` を使います。サーバーのPythonだけの変更は `deploy_server.sh` を使います。Android単独変更は `record_changes.sh --target android`、公開サービスへ影響しない運用スクリプト・Android workflow単独変更は `record_changes.sh --target operations` を使います。どの記録スクリプトも確認なしでは計画だけ表示して終了し、対象外ファイルが混ざっていれば停止します。Android Actionsは配布物に関係するパスが変わった場合だけ起動します。`android/version.properties` の `VERSION_NAME` を進めたのに、その版の `android/ci/changelogs/vX.Y.Z.md`（箇条書き1行以上）が同じ変更に無い場合、`record_changes.sh --target android` と `publish_version.sh` は記録前に停止します。`android/app/src/main/` を変えたのに `VERSION_NAME` を進めていない場合も停止します。テストやビルド設定だけの変更は版を上げずに記録でき、Android CIは検証だけを行い、GitHub Releaseを作りません（`record_changes.sh` の計画に `Android Release: none` と表示されます）。

一部のスクリプトは、このリポジトリの参照デプロイ構成に合わせたサービス名やヘルスチェック先を使用しています。セルフホスト環境で利用する前に値を変更してください。これらはアプリの起動に必須ではありません。
