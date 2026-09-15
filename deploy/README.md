# サーバー設定サンプル

このディレクトリには、本番導入時にアプリ外へ配置する設定例があります。

- `systemd/`：Gunicornと2つのRQワーカー
- `apache/`：HTTPSリバースプロキシ、SSE、大容量アップロード
- [ANDROID_CLIENT.md](ANDROID_CLIENT.md)：Android連携API、認証、アプリ構築、サーバー設定、運用・試験

サンプルは実行ユーザー `ai-playground`、配置先 `/opt/ai-playground`、ドメイン `chat.example.com` を前提にしています。コピー前に環境へ合わせて変更し、秘密情報を設定ファイルへ直接書かないでください。導入順序と検証は [../README.md](../README.md#本番環境への導入) を参照してください。
