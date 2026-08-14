# 第三者ソフトウェアとライセンス

AI Chat Playground本体のMIT Licenseは、第三者のコード、フォント、アイコン、API、商標へ適用範囲を広げるものではありません。各項目には各権利者のライセンスと利用規約が適用されます。

## リポジトリに同梱するブラウザーライブラリ

| コンポーネント | バージョン | 配置先 | ライセンス | 配布元 |
|---|---:|---|---|---|
| DOMPurify | 3.4.11 | `static/vendor/dompurify-3.4.11.min.js` | Apache-2.0（本配布で選択） | <https://github.com/cure53/DOMPurify> |
| html2canvas-pro | 2.3.2 | `static/vendor/html2canvas-pro-2.3.2.min.js` | MIT | <https://github.com/yorickshan/html2canvas-pro> |
| jsPDF | 2.5.1 | `static/vendor/jspdf-2.5.1.umd.min.js` | MIT | <https://github.com/parallax/jsPDF> |
| Marked | 4.3.0 | `static/vendor/marked-4.3.0.min.js` | MIT | <https://github.com/markedjs/marked> |
| Font Awesome Free（使用アイコンのサブセット） | 6.5.2 | `static/vendor/icons/` | Icons: CC-BY-4.0、Fonts: OFL-1.1、Code: MIT | <https://fontawesome.com/license/free> |

各ファイル先頭の著作権・ライセンスコメントを削除しないでください。ライセンス本文は `LICENSES/` に収録しています。jsPDFの配布ファイル内に含まれる依存コンポーネントのライセンスコメントも保持してください。

生成済みの `static/css/chat.tailwind.v*.css` にはTailwind CSS由来のコードが含まれ、Tailwind CSSはMIT Licenseです。配布元は <https://github.com/tailwindlabs/tailwindcss> です。

## 実行時にCDNから取得する資産

| コンポーネント | 指定バージョン | ライセンス | 配布元 |
|---|---:|---|---|
| browser-image-compression | 2.0.2 | MIT | <https://github.com/Donaldcwl/browser-image-compression> |
| @github/webauthn-json | 2.1.1 | MIT | <https://github.com/github/webauthn-json> |
| MathJax | 3.x | Apache-2.0 | <https://github.com/mathjax/MathJax> |
| highlight.js | 11.9.0 | BSD-3-Clause | <https://github.com/highlightjs/highlight.js> |
| Noto Sans JP | Google Fonts配信版（400／700、非同期読み込み） | OFL-1.1 | <https://fonts.google.com/noto> |
| JetBrains Mono | Google Fonts配信版（400、非同期読み込み） | OFL-1.1 | <https://github.com/JetBrains/JetBrainsMono> |

CDNから取得するだけでリポジトリへバイナリを同梱していない資産も、ブラウザー上では第三者コードとして実行されます。自己ホストへ切り替える場合は、対象バージョンのライセンス本文、著作権表示、NOTICE、フォント名の予約条件等を確認し、配布物へ同梱してください。

## Python依存パッケージ

Python依存関係は [requirements.txt](requirements.txt) に固定しています。これらは本リポジトリへソースやwheelとして同梱せず、導入時にPyPI等から取得します。各パッケージにはMIT、BSD、Apache-2.0、MPL、LGPL等の個別ライセンスが適用されます。

アプリをコンテナ、VMイメージ、アプライアンス等として再配布する場合、インストール済みパッケージも配布物に含まれます。その時点の実ファイルとメタデータからライセンス一覧を生成し、各ライセンスが要求する本文、著作権表示、NOTICE、ソース提供条件を満たしてください。`requirements.txt` の記載だけで再配布条件を満たすとは限りません。

確認例:

```bash
python -m pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-license-file
```

`pip-licenses` 自体は監査補助であり、法的判断の代替ではありません。

## 外部サービス

OpenAI、Google／Vertex AI、Anthropic、DeepSeek、Moonshot、xAI、Cloudflare、Google OAuth等は本プロジェクトのライセンス対象ではありません。各サービスの規約、API利用条件、商標ガイドライン、データ処理条件、料金が別途適用されます。画面に表示されるサービス名・モデル名は各権利者の商標または名称であり、本プロジェクトとの提携や承認を意味しません。

## 素材を追加・更新するとき

1. 配布元と正確なバージョンを記録する。
2. 商用利用、改変、再配布、表示、NOTICE、ソース公開の条件を確認する。
3. ファイル内のライセンスヘッダーを保持する。
4. この文書を更新し、必要なライセンス本文を同梱する。
5. 互換性や権利関係が不明な素材は取り込まない。

本書はライセンス情報の整理を目的とし、法的助言ではありません。
