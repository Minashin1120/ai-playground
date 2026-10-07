# JavaScript

`chat_core.v*.js` がチャット画面の中心で、モデル定義、送信、ストリーム、履歴、設定、各モーダルを扱います。その他のファイルはPWA、ランディング画面、ローディングスピナー等の独立機能です。

主な構成:

- `chat_core.v*.js`：チャット、モデル、設定、履歴、モーダル（下記の部品から自動生成される結合ソース）
- `chat_core_parts/`：`chat_core` の編集用ソースを順序付き部品に分割したもの
- `activity_log.js`：「ログの収集を強化」の操作ログ（ブラウザーのlocalStorageに直近1時間を記録し、フィードバック送信時に添付）
- `progress_spinner.js`：通信中の共通進捗表示
- `connection_monitor.js`：サーバー接続状態（オフライン・不安定・メンテナンス・復帰）の監視と表示
- `pwa_install.js`：PWAの導入と表示モード連携
- `landing_demo.js`：公開ランディング画面のチャットデモ

`activity_log.js`、`progress_spinner.js`、`pwa_install.js`、`connection_monitor.js`、`landing_demo.js` には、配信用の `*.min.js` が隣にあります（`scripts/build_frontend.sh` が生成します）。

チャットコアのファイル名には、ブラウザーキャッシュ更新用のバージョン番号が含まれます。`chat_core.v*.js` は編集・テスト用のソース、`chat_core.min.v*.js` はブラウザーへ配信する圧縮ファイルです。

## chat_core の分割について

`chat_core.v*.js` は1ファイルで約2.5万行あるため、編集は `static/js/chat_core_parts/` 配下の順序付き部品（`chat_core.partNN_名前.js`）に対して行います。`scripts/build_frontend.sh` が部品を順に連結して `chat_core.v*.js`（結合ソース）を再生成し、それを圧縮して `chat_core.min.v*.js` を作ります。

- **編集対象は部品ファイル**。部品を編集したら必ず `scripts/build_frontend.sh` を実行して結合ソースと圧縮ファイルを更新してください（回帰テストと検証は結合ソースを検査します）。
- 部品は連結順に番号が付いており、**結合ソースは部品の連結とバイト単位で一致**します（検証で確認されます）。
- 部品ファイルはバージョン番号を持ちません（バージョン管理されるのは結合ソースと圧縮ファイルのみ）。
