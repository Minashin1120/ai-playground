# CSS

- `app-design.css`：全画面共通のデザイントークンと基本コンポーネント
- `chat.custom.v*.css`：チャット固有の手書きスタイル
- `chat.tailwind.v*.css`：生成済みTailwind CSS
- `theme-light.css`：OSがライトテーマのときに全画面へ適用するホワイトモードの上書き（各ページの最後に読み込む）
- `theme-light-manual.css`：ライトテーマを手動で選んだときだけチャット画面が読み込む補助上書き（`chat.html` の `initial_light_mode_enabled` と設定切替）
- `landing.css`：公開ランディング画面

チャット用CSSのファイル名にはキャッシュ更新用のバージョン番号が含まれます。`chat.custom.v*.css` は編集用、`chat.custom.min.v*.css` は配信用です。画面スタイルはデスクトップとモバイル、ライトとダークの両テーマに対応しています。
