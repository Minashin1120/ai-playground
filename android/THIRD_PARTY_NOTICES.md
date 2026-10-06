# Android版の第三者素材

| 素材 | 使い方 | ライセンス |
|---|---|---|
| Font Awesome Free 6.5.2 のアイコン | Web版と同じアイコンを `app/src/main/res/drawable/fa_*.xml` に変換して同梱（`ci/sync-web-icons.py`） | アイコン：CC BY 4.0（https://fontawesome.com/license/free） |
| Noto Sans JP、JetBrains Mono | Google Fontsの配信（Google Play開発者サービス経由）で実行時に取得。APKには含めない | SIL Open Font License 1.1 |
| MathJax 3.2.2（`app/src/main/assets/mathjax/`） | 数式の組版。npmの `mathjax@3.2.2` から `es5/tex-svg.js` とTeX拡張（`es5/input/tex/extensions/`、`all-packages.js` を除く）を改変せずに同梱し、画面に出さないWebViewで実行 | Apache License 2.0（同じ場所の `LICENSE`） |
| `app/src/main/res/values/font_certs.xml` | Google Fontsの配信元を検証する証明書（Android Open Source Projectのサンプルから取得） | Apache License 2.0 |
