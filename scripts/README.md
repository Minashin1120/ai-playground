# 保守・検証スクリプト

このディレクトリにはサービス再起動、キャッシュ削除、ランディング画面のDOM／描画検証、版確認用のスクリプトがあります。

- `verify_changes.sh`：構文、必須の版付き資産、公開文書の版番号、回帰テストをまとめて確認する
- `prepare_version.sh`：版番号と版付きチャットJS/CSSを次の版へ進め、公開更新履歴を書き、圧縮ファイルを作り直す
- `publish_version.sh`：確認後に稼働中サービスへ反映し、キャッシュを消し、版をリポジトリへ記録する
- `restart_services.sh`：`ai-chat.service` と `ai-chat-worker@1..2.service` を再起動し、PID変更とWeb応答を確認（RQワーカーはメモリ負荷対策で2つのみ有効）
- `purge_cloudflare_cache.sh`：指定ゾーンのホストキャッシュを削除
- `build_frontend.sh`：バージョン付きチャットJS/CSSと補助スクリプトを圧縮する
- `build_icon_subset.py`：使用中のFont Awesomeアイコンだけを同梱する
- `test_landing_demo_dom.js`：軽量DOM shimを用いたランディング画面テスト
- `measure_landing_cdp.py`／`verify_landing_geometry.js`：ブラウザー上の描画・座標検証

`verify_changes.sh` は何度でも実行できます。`prepare_version.sh` は直し終わったあとだけ使います。`publish_version.sh` は実行前に `--message` と、現在の `SYSTEM_VERSION` を付けた `--confirm` が必要です。確認なしでは計画だけ表示して終了します。再起動に失敗した場合はそこで止まり、キャッシュ削除と版の記録は行いません。

一部のスクリプトは、このリポジトリの参照デプロイ構成に合わせたサービス名やヘルスチェック先を使用しています。セルフホスト環境で利用する前に値を変更してください。これらはアプリの起動に必須ではありません。
