# chat — チャット画面テンプレートの部品

チャット本体の HTML は `templates/chat.html` を入口にし、画面の塊ごとにこのディレクトリへ分けています。Flask の `{% include 'chat/....html' %}` で結合します。描画時の変数（`csrf_token`、`system_version`、`current_user` など）は親と同じです。

**DOM の `id` とクラス名は JS（`static/js/chat_core_parts/`）が直接参照します。** 部品へ移すときも id を変えないでください。

## 探し方

1. 下の表で対象ファイルを決める。
2. その HTML だけを開く。ボタン色なら CSS（`static/css/chat.custom.v*.css`）が先で、HTML は id の確認程度にする。
3. `chat.html` 全体や、結合済み JS（`chat_core.v*.js` / `.min.js`）を通読しない。
4. 要素 id が分からないときは、この README のあと `grep` で `templates/chat/` を探す。

## ファイル一覧

| ファイル | 主な内容 | いつ開くか |
|---|---|---|
| `../chat.html` | doctype、head（CSS/JS 読み込み、テーマ、Turnstile）、body の include 一覧 | 新規アセットの読み込み、viewport、最初の描画 |
| `chrome.html` | flash、Turnstile 箱、オフライン帯、画像ビューア、トースト、Alpha バー、引用ポップオーバー | 通知、画像ビューア、引用 UI |
| `sidebar.html` | 左サイドバー（新規チャット、検索、スレッド一覧、フッター） | サイドバー、スレッド一覧、モバイルメニュー |
| `main_stage.html` | モバイルヘッダー、会話コンテナ、Welcome、Canvas パネル | チャット表示、Welcome、Canvas |
| `composer.html` | 束ね役。下記 `composer_*.html` を include するだけ | 分割先を決めるとき（下の行へ） |
| `composer_context_bars.html` | 引用バー、Coding対象バー、編集バー | 引用、編集、Coding対象のバー表示 |
| `composer_controls.html` | モデル選択行と標準オプション行（File / MCP 等のチェック） | 送信まわりのオプション、MCPスイッチ |
| `composer_attachments.html` | 添付ファイル／マスクのプレビュー | 添付、マスク表示 |
| `composer_gen_image.html` | GPT / Gemini / Grok の画像生成パネル | 画像生成の詳細設定 |
| `composer_gen_media.html` | xAI 詳細パネル、動画/音楽/OCR/TTS パネル | メディア生成の詳細設定 |
| `composer_panels.html` | STS パネル、Voice スタジオ帯、自動検索バナー | STS、Voice、自動検索 |
| `composer_input.html` | 入力欄、送信、スラッシュ/`@` 候補、アップロード状態 | 送信ボタン、入力欄、スラッシュコマンド |
| `composer_popups.html` | ミニマルモードのポップアップ、思考量スライダー | ミニマルモード、思考量 |
| `overlay_upload.html` | ドロップオーバーレイ、アップロードモーダル、カメラ | アップロード、カメラ |
| `overlay_voice.html` | Lyria スタジオ、Voice スタジオ、マーカー編集 | 音声スタジオ、画像マーカー |
| `overlay_thread.html` | 履歴モーダル、スレッド設定、圧縮設定 | 履歴、スレッド設定、画像圧縮 |
| `overlay_model_gem.html` | モデル選択モーダル、Gem 作成/編集 | モデル一覧、Gem |
| `overlay_settings.html` | 設定モーダル（タブ、2FA、MCP 設定を含む） | 設定画面、MCP タブ、2FA タブ |
| `overlay_dialogs.html` | MCP 変更確認、Gemini ローカル Python、インポート確認 | 確認ダイアログ |
| `overlay_library.html` | ファイルライブラリ、規約、Alpha 案内、ブランチ管理、管理者の Bot 画面 | ライブラリ、ブランチ |
| `overlay_version.html` | 新バージョン通知モーダル | 更新通知 |
| `scripts.html` | `CHAT_CONFIG`、コア JS の読み込み、トークン/Python/暗号化モーダル、サイドバー用インライン JS | 初期 JS 設定、インラインスクリプト |

`pwa_meta.html`、`web_fonts.html`、`icon_css.html` はチャット専用ではなく、他画面と共有する部品です。`templates/` 直下にあります。

## 編集時の注意

- 新しいモーダルを足すときは、近い `overlay_*.html` に入れるか、ファイルを増やして `chat.html` の include に1行足します。
- 閉じタグの対応は部品内で完結させてください。`{% if %}` を部品をまたいで分割しないでください。
- 画面の見た目は CSS、動きは `chat_core_parts` です。HTML は構造と id の置き場所です。
