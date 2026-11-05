# 変更対象ごとの記録・公開

| 変更対象 | 経路 |
|---|---|
| Web UI・公開資産・`app.py`、Webとサーバーの両方 | `prepare_version.sh` → `publish_version.sh` の計画 → 同じ版を `--confirm` に指定 |
| WebとAndroidの両方 | Webと同じ経路で1つのcommitに含め、対象SHAのAndroid ActionsとReleaseまで確認 |
| サーバーだけ（`server/`・`worker.py`・`deploy/*.md` と対応するテスト） | `deploy_server.sh` の計画 → `--confirm SERVER` |
| Androidだけ | `record_changes.sh --target android` の計画 → 表示されたAndroid版を `--confirm` に指定 → Actions確認 |
| 運用スクリプト・対象限定テスト・テスト案内・Android workflow | 対象限定の検証 → `record_changes.sh --target operations` の計画 → 表示された確認文字列を指定 |
| 実行・配信されない文書だけ、またはテストだけ | 指示された対象ファイルだけを検証・記録。Web版上げ・再起動を行わない |

計画は未追跡のディレクトリもファイル単位へ展開して表示します。`tests/docs/` ではMarkdown文書だけを運用対象とし、同じディレクトリのPythonなどは対象外です。計画に意図しないファイルや対象外のファイルが含まれていれば停止します。記録は許可したファイルだけを扱い、全体を一括stageしません。Web公開では再起動成功後にキャッシュを削除し、公開URLを確認してからcommit・tag・pushします。再起動失敗時に先へ進みません。

利用者が更新後作業の保留を明示した場合は、その指示を優先します。テスト省略と公開全体の保留を混同しません。版更新前検証の詳細は [verification.md](verification.md)、Androidの条件は [android_checks.md](android_checks.md) を参照してください。

このリポジトリの参照デプロイ構成を前提としたサービス名・URLがあります。セルフホストでは利用前に値を確認してください。運用スクリプトはアプリ起動に必須ではありません。
