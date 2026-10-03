# Jinjaテンプレート

Flaskが描画するHTMLテンプレートです。公開ランディング（`landing.html`）、認証（`login.html`、`signup.html`、`verify_2fa.html`）、初期セットアップ（`setup.html`）、チャット、ヘルプ、更新履歴、エラー（`403.html`、`404.html`、`banned.html`）、メンテナンス、Android連携の確認ページ（`android_connect.html`）、外部MCPのOAuth完了ページ（`mcp_oauth_callback.html`）、PWAメタ情報・Webフォント・アイコンCSSの共有部品を含みます。チャット画面の本体は `chat.html` を入口にし、部品は `chat/` にあります。部品の説明は `chat/README.md` を先に読んでください。

フォームにはCSRFトークンを含め、ユーザー入力を安易に `safe` 扱いしないでください。外部スクリプト・CSS・フォントを追加する場合は、CSP、Subresource Integrity、プライバシー、可用性、ライセンスを確認し、`THIRD_PARTY_NOTICES.md` を更新します。アイコンは `vendor/icons/` のサブセット、Webフォントは `web_fonts.html` の非同期読み込みを使います。

画面によってDOM構造が異なるため、共通JavaScriptから要素を参照するときは存在確認が必要です。
