# JavaScript

`chat_core.v*.js` がチャット画面の中心で、モデル定義、送信、ストリーム、履歴、設定、各モーダルを扱います。その他のファイルはPWA、ランディング画面、ローディングスピナー等の独立機能です。

主な構成:

- `chat_core.v*.js`：チャット、モデル、設定、履歴、モーダル
- `progress_spinner.js`：通信中の共通進捗表示
- `pwa_install.js`：PWAの導入と表示モード連携
- `landing.js`／`landing_demo.js`：公開ランディング画面

チャットコアのファイル名には、ブラウザーキャッシュ更新用のバージョン番号が含まれます。`chat_core.v*.js` は編集・テスト用のソース、`chat_core.min.v*.js` はブラウザーへ配信する圧縮ファイルです。ソースを変えたあとは `scripts/build_frontend.sh` で圧縮ファイルを更新します。
