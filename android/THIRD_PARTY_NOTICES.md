# Android版の第三者素材

| 素材 | 使い方 | ライセンス |
|---|---|---|
| Font Awesome Free 6.5.2 のアイコン | Web版と同じアイコンを `app/src/main/res/drawable/fa_*.xml` に変換して同梱（`ci/sync-web-icons.py`） | アイコン：CC BY 4.0（https://fontawesome.com/license/free） |
| Noto Sans JP、JetBrains Mono | Google Fontsの配信（Google Play開発者サービス経由）で実行時に取得。APKには含めない | SIL Open Font License 1.1 |
| `app/src/main/res/values/font_certs.xml` | Google Fontsの配信元を検証する証明書（Android Open Source Projectのサンプルから取得） | Apache License 2.0 |
