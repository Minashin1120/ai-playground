# 保守・検証スクリプト

このディレクトリにはサービス再起動、Cloudflareキャッシュ削除、ランディング画面のDOM／描画検証用スクリプトがあります。

- `restart_services.sh`：`ai-chat.service` と `ai-chat-worker@1..4.service` を再起動し、PID変更とWeb応答を確認
- `purge_cloudflare_cache.sh`：指定ゾーンのホストキャッシュを削除
- `build_frontend.sh`：バージョン付きチャットJS/CSSと補助スクリプトを圧縮する
- `build_icon_subset.py`：使用中のFont Awesomeアイコンだけを同梱する
- `test_landing_demo_dom.js`：軽量DOM shimを用いたランディング画面テスト
- `measure_landing_cdp.py`／`verify_landing_geometry.js`：ブラウザー上の描画・座標検証

一部のスクリプトは、このリポジトリの参照デプロイ構成に合わせたサービス名やヘルスチェック先を使用しています。セルフホスト環境で利用する前に値を変更してください。これらはアプリの起動に必須ではありません。
