# 静的資産

Flaskの `/static/` から配信するCSS、JavaScript、PWA資産、公開変更履歴、法的文書、第三者ライブラリを格納します。

- `css/`：画面スタイルとバージョン付きチャットCSS
- `js/`：クライアントロジックとバージョン付きチャットコア
- `pwa/`：PWAアイコン
- `docs/`：機能解説の文書（Batch処理）
- `changelogs/`：アプリ内で公開する更新履歴
- `legal/`：利用規約とプライバシーポリシー
- `vendor/`：リポジトリへ同梱する第三者JavaScriptとアイコンサブセット

直下には、PWAの `manifest.webmanifest`、Service Workerの `sw.js`、オフライン画面の `offline.html`、README用画像の `github_promo.png` があります。

チャット用のJavaScriptとCSSには、ブラウザーキャッシュを安全に更新するためのバージョン番号がファイル名に含まれます。テンプレートからの参照はアプリの表示バージョンに応じて生成されます。
