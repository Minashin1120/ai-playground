# Androidの版とActions

アプリ本体（`android/app/src/main/`）を変えた場合は `VERSION_CODE` と未使用の `VERSION_NAME` を進め、同じ変更に `android/ci/changelogs/vX.Y.Z.md` を新設します。日本語の箇条書きで利用者から見える変更を書き、既存の版の履歴を上書きしません。ファイルがない・本文がない・禁止表現がある変更は記録計画で停止します。

テストやビルド設定だけの変更は、配布物を更新しない経路では版を上げません。新しいAndroid版の配布時だけReleaseを作ります。文書だけの変更は記録できますが、配布物に影響するパスがなければActionsは起動しません。正確な対象は `.github/workflows/android.yml` と `_release_common.py` で一致させます。

単体テスト省略の明示がある場合、記録・公開スクリプトに `--skip-android-tests` を指定します。commitに `Android-Skip-Unit-Tests: true` が追加されます。lint省略も明示された場合だけ `--skip-android-lint`（`Android-Skip-Lint: true`）を指定します。APK生成・固定署名の照合・Release確認は省きません。Actionsが起動しない変更への省略指定は計画で拒否され、手動実行では全チェックが実行されます。

コンパイル・単体テスト・lint・APK/AAB生成・署名・インストールをローカルのGradleやAndroid SDKで実行してはいけません。Actionsで対象SHAの成功、`app-debug`・`android-reports`・`app-release-signed`、ReleaseのAPK・SHA-256・署名fingerprintとコミットの一致まで確認します。

プロジェクトの入口は [android/README.md](../../android/README.md)、通信仕様は [ANDROID_CLIENT.md](../../deploy/ANDROID_CLIENT.md) を参照してください。
