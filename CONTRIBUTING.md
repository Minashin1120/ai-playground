# コントリビューションガイド

Issue、ドキュメント修正、バグ修正、テスト追加を歓迎します。変更前に最新の `main` を取得し、関連Issueがあればリンクしてください。

## 開発環境

Python 3.11、MariaDB、Redis、Node.js（JavaScript構文確認用）、ffmpeg、bubblewrapを用意します。セットアップは [README.md](README.md#開発とテスト) を参照してください。

## 変更時の原則

- 秘密情報、実ユーザーデータ、ログ、`secret.key`、`.env`、`instance/`、バックアップをコミットしない。
- DBスキーマ変更には前方移行、ロールバック時の影響、既存データ互換性を説明する。
- 既存データ、公開API、保存済みチャットとの後方互換性を保つ。
- JavaScript変更では、すべての画面に同じDOM要素が存在するとは限らないことを考慮する。
- 共通機能を変更するときは既存の責務分担を保ち、同じ処理を複数箇所へ重複実装しない。
- 依存物や画像を追加する場合は、出典、ライセンス、再配布条件を確認し、必要に応じて `THIRD_PARTY_NOTICES.md` を更新する。
- 利用者に影響する変更では、READMEや公開変更履歴も更新する。

## テスト

```bash
venv/bin/python -m pytest -q
node --check static/js/progress_spinner.js
node --check static/js/chat_core.v*.js
```

`static/js/chat_core.v*.js` や `static/css/chat.custom.v*.css` を編集した場合は、配信用の圧縮ファイルを更新します。

```bash
./scripts/build_frontend.sh
```

アイコンを追加・削除した場合は `venv/bin/python scripts/build_icon_subset.py` も実行します。

変更範囲に対応する回帰テストを追加してください。テスト構成は [tests/README.md](tests/README.md) を参照してください。

## Pull Request

Pull Requestには目的、変更内容、検証結果、DB・設定・互換性への影響、画面変更時のスクリーンショットを記載してください。1つのPRへ無関係な変更を混在させないでください。

投稿されたコントリビューションは、リポジトリの [MIT License](LICENSE) の条件で提供できるものとします。第三者コードを投稿する場合は、その権利とライセンス互換性を投稿者が確認してください。
