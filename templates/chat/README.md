# chat — チャット画面テンプレートの部品

| ファイル | 主な内容 | いつ開くか |
|---|---|---|
| `../chat.html` | doctype、head（CSS/JS 読み込み、テーマ、Turnstile）、body の include 一覧 | 新規アセットの読み込み、viewport、最初の描画 |
| `chrome.html` | flash、Turnstile 箱、オフライン帯、画像ビューア、トースト、Alpha バー、引用ポップオーバー | 通知、画像ビューア、引用 UI |
| `sidebar.html` | 左サイドバー（新規チャット、検索、スレッド一覧、フッター） | サイドバー、スレッド一覧、モバイルメニュー |
| `main_stage.html` | モバイルヘッダー、Batch完了バナー、会話コンテナ、Welcome、Canvas パネル | ヘッダー、Batch完了通知、チャット表示、Welcome、Canvas |
| `composer.html` | 束ね役。下記 `composer_*.html` を include するだけ | 分割先を決めるとき（下の行へ） |
| `composer_context_bars.html` | 引用バー、Coding対象バー、編集バー | 引用、編集、Coding対象のバー表示 |
| `composer_controls.html` | モデル選択行と標準オプション行（Gemini／OpenAI／xAI／Z.AI Batch / File / MCP 等のチェック） | 送信まわりのオプション、MCPスイッチ |
| `composer_attachments.html` | 添付ファイル／マスクのプレビュー | 添付、マスク表示 |
| `composer_gen_image.html` | GPT / Gemini / Grok / Ideogram の画像生成パネル | 画像生成の詳細設定 |
| `composer_gen_media.html` | xAI 詳細パネル、動画/音楽/OCR/TTS パネル | メディア生成の詳細設定 |
| `composer_panels.html` | STS パネル（音声ドック：マイク・文字起こし・折りたたみ設定）、自動検索バナー | STS、音声ドック、自動検索 |
| `composer_input.html` | 入力欄、送信、スラッシュ/`@` 候補、アップロード状態 | 送信ボタン、入力欄、スラッシュコマンド |
| `composer_popups.html` | ミニマルモードのポップアップ、思考量スライダー | ミニマルモード、思考量 |
| `overlay_upload.html` | ドロップオーバーレイ、アップロードモーダル、カメラ | アップロード、カメラ |
| `overlay_image_split.html` | 画像分割のプレビュー・設定・添付・個別保存と専用圧縮モーダル | 画像分割の画面を変更するとき |
| `compression_fields.html` | プロンプトバーと画像分割で共用する圧縮設定の入力欄 | 圧縮の入力項目を変更するとき |
| `overlay_voice.html` | Lyria スタジオ、Voice スタジオ（音声ドックの拡大表示）、マーカー編集 | 音声スタジオ、画像マーカー |
| `overlay_thread.html` | 履歴モーダル、スレッド設定、圧縮設定、Batch処理一覧モーダル | 履歴、スレッド設定、画像圧縮、Batch処理の確認・停止・削除 |
| `overlay_model_gem.html` | モデル選択モーダル、Gem 作成/編集 | モデル一覧、Gem |
| `overlay_settings.html` | 設定モーダル（タブ、2FA、MCP 設定を含む） | 設定画面、MCP タブ、2FA タブ |
| `overlay_dialogs.html` | チャット名の変更、MCP 変更確認、Gemini ローカル Python、インポート確認 | 確認ダイアログ |
| `overlay_library.html` | ファイルライブラリ、規約、Alpha 案内、ブランチ管理、管理者の Bot 画面 | ライブラリ、ブランチ |
| `overlay_version.html` | 新バージョン通知モーダル | 更新通知 |
| `scripts.html` | `CHAT_CONFIG`、コア JS（`chat_core_bundles` の複数ファイル）の読み込み、トークン/Python/暗号化モーダル、サイドバー用インライン JS | 初期 JS 設定、インラインスクリプト |
