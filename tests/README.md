# 回帰テストの入口

| 文書・ファイル | 主な内容 | いつ開くか |
|---|---|---|
| [docs/workflow.md](docs/workflow.md) | 検証順序、対象限定実行、失敗時、スキップ、文書の更新 | 検証手順を確認するとき |
| [docs/frontend.md](docs/frontend.md) | Web画面・JS部品のテスト一覧と実装上の契約 | 画面、テンプレート、部品の追加・改名、操作・表示を変更するとき |
| [docs/providers.md](docs/providers.md) | モデル・料金・プロバイダーのテスト一覧と実装上の契約 | モデルの追加・切替、入力計測、費用予想、Batch、PromptCacheを変更するとき |
| [docs/media.md](docs/media.md) | 画像・音声・動画・ツールのテスト一覧と実装上の契約 | メディア、生成ファイル、OCR、Python・MCPの実行を変更するとき |
| [docs/chat_server.md](docs/chat_server.md) | 会話・通信・性能のテスト一覧と実装上の契約 | 生成、保存、ストリーム、再接続、履歴、性能を変更するとき |
| [docs/security_accounts.md](docs/security_accounts.md) | 認証・アカウント・安全性のテスト一覧と実装上の契約 | ログイン、2FA、SSO、Bot対策、アカウント・暗号化を変更するとき |
| [docs/storage_feedback.md](docs/storage_feedback.md) | 添付・ライブラリ・フィードバック・ログのテスト一覧と実装上の契約 | ファイル保存、添付・画像UI、フィードバック、ログを変更するとき |
| [docs/android.md](docs/android.md) | Android・モバイルAPIのテスト一覧と実装上の契約 | モバイルAPI、同期、Androidの文言・モデル・配布を変更するとき |
| [docs/operations.md](docs/operations.md) | 検証・公開・運用のテスト一覧と実装上の契約 | 版更新、検証、公開、記録スクリプトと実行環境を変更するとき |
| `conftest.py` | pytestの隔離DB設定 | テスト用DBの隔離そのものを変更するとき |
| `app_source.py` / `chat_template.py` | 分割ソース・テンプレートを読むテスト用ヘルパ | テストのソース取得方法を変更するとき |
