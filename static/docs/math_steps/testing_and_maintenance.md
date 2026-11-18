# 数学ステップ解説：テスト仕様・セキュリティ・保守運用

本ドキュメントでは、「数学ステップ解説」機能のテストスイート構成、セキュリティ・プライバシー保護規約、アカウント移行での扱い、およびトラブルシューティングを解説します。

---

## 1. テストスイートの構成

本機能には、サーバー側・フロントエンドUI・Androidクライアントのそれぞれに手厚い回帰テストが配置されています。

### 1.1 サーバー回帰テスト（`tests/test_math_steps_server.py`）

| テスト関数名 | 検証内容 |
|---|---|
| `test_config_defaults_and_prompt_builder` | デフォルト設定の正規化、プレースホルダー置換、指示文ビルダーの整合性 |
| `test_background_and_browser_fast_inject_the_math_prompt` | 通常RQタスクおよびブラウザ高速モード時のシステムプロンプト結合 |
| `test_extract_math_steps_ignores_fenced_headings` | コードフェンス内（Pythonコード内の `# ステップ 1` 等）を見出しとして誤認しないことの検証 |
| `test_encrypted_math_steps_state_round_trip` | E2EE暗号化されたメッセージにおける `math_steps_state` の暗号化・復号・永続化のラウンドトリップ |
| `test_settings_get_and_post_math_steps` | `/api/settings` を経由した設定値（`math_steps_enabled`, `math_steps_config`, `default_math_steps`）の取得・更新 |
| `test_progress_api_ownership_and_clamping` | `/api/math_steps/progress` の所有者検証（他ユーザーのメッセージ拒否）、および 0〜50 の数値クランプ |
| `test_simplify_validation_and_success` | `/api/math_steps/simplify` のバリデーション、モデル呼び出し、Usage記録、および見出しストリップ |
| `test_prompt_options_and_message_payload` | `prompt_options` への `math_steps` フラグの記録とAPIレスポンスの検証 |
| `test_mobile_preferences_math_steps` | モバイル用設定API（`/api/mobile/preferences`）での同期 |
| `test_browser_fast_bootstrap_returns_math_prompt` | 高速モードのブートストラップレスポンスに `math_steps_system_prompt` が含まれることの確認 |

---

### 1.2 WebフロントエンドUIテスト（`tests/test_math_steps_ui.py`）

| テスト関数名 | 検証内容 |
|---|---|
| `test_new_part_is_last_and_wired` | `chat_core.part30_mathsteps.js` が部品順序の末尾に正しく配置され、スクリプトバンドルに結合されていること |
| `test_mathsvg_stays_image_only_and_svg_is_unchanged` | `mathsvg` フェンスが画像としてレンダリングされ、コードUIを出さないこと |
| `test_settings_and_composer_controls_exist` | プロンプトバーのチップ、歯車アイコン、設定モーダルがHTMLに正しく存在すること |
| `test_steps_render_one_at_a_time_with_image_only_mathsvg` | 1ステップずつ段階的に表示され、不要なステップが隠れていること |
| `test_next_back_restart_and_finish` | 次へ／前へ／最初から／完了ボタンの動作とDOMの更新 |
| `test_progress_is_posted_after_step_change` | ステップ移動後に `/api/math_steps/progress` への非同期送信がスケジュールされること |
| `test_simplify_splits_the_current_step_and_can_be_restored` | 「もっとやさしく」で現在のステップが複数の小ステップへ分割され、増えたステップを移動でき、「元の説明に戻す」で元の1ステップに戻ること |
| `test_without_step_headings_normal_markdown_is_unchanged` | ステップ見出しを含まない通常の回答は、一切崩れずに通常のMarkdownとして表示されること（フォールバック） |
| `test_pdf_and_copy_helpers_expand_steps_and_diagrams` | スレッドPDFエクスポートおよびクリップボードコピー時に全ステップが展開されること |
| `test_master_off_hides_chip_and_gear_modal_saves_partial_config` | マスター設定OFF時にチップが非表示になること、およびモーダルでの部分設定保存 |

---

### 1.3 Androidユニットテスト

- `android/.../data/MathStepsTest.kt`: `MathStepsConfig` と `MathStepsState` のJSON変換、バリデーション、`extractMathSteps()` パーサー、PDF/コピー展開ロジックの単体テスト。
- `android/.../data/direct/MathStepsPromptTest.kt`: 直接通信用プロンプトビルダーの単体テスト。

---

## 2. セキュリティとプライバシー保護

### 2.1 機密情報非出力ポリシー（No-Log Policy）
`server/routes_math_steps.py` の先頭（L4-5）に明記されている通り、数学ステップ機能では**「本文・プロンプト・回答はログに出さない」**ことが徹底されています。
- ログ（`logger.info` / `logger.warning`）には `user_id`, `message_id`, `step_index`, `chars`（文字数）などの数値・メタデータのみが出力され、数学の問題文や計算過程、生成文は一切記録されません。

### 2.2 厳格な認可とメッセージ所有権検証
`/api/math_steps/progress` および `/api/math_steps/simplify` では、`current_user.id == thread.user_id` のDBジョインクエリ（`_resolve_math_steps_message`）により、ログインユーザー自身が所有するスレッドのメッセージ以外へのアクセスを完全に遮断します。

### 2.3 `mathsvg` サニタイズによる多層防御
AIが生成したSVGに含まれるスクリプト実行や外部データ持ち出しを防ぐため、フロントエンドで徹底的なタグ除去（`<script>`, `<foreignObject>` 等）および属性チェックが行われます。

### 2.4 リソース枯渇・悪用防止の制約
- **レートリミット**:
  - 進捗保存: 120回 / 60秒
  - ステップ分割: 20回 / 60秒
- **文字数・分割数上限**:
  - 1分割の本文: 最大 20,000 文字
  - 1元ステップの分割数: 最大 8
  - 1メッセージ内の分割・簡略化合計: 最大 100,000 文字
  - SVGマークアップ: 最大 300,000 文字
  - 表示ステップ数: 最大 50 ステップ

---

## 3. アカウントデータ移行（エクスポート・インポート）

ユーザーのアカウントデータ移行（バックアップ・リストア）においても、数学ステップの設定と進捗状態が完全に保持されます。

- **エクスポート（`server/account_transfer.py`）**:
  - ユーザー設定: `math_steps_enabled`, `math_steps_config`, `default_math_steps`
  - メッセージ進捗: 暗号化されているメッセージであっても一時的に安全に復号され、エクスポートアーカイブに格納されます。
- **インポート（`server/routes_account_import.py`）**:
  - インポート時に文字数制限（`_safe_account_import_text`）と状態構造の妥当性検証が行われ、復元先アカウントの暗号化設定に従って再暗号化されます。

---

## 4. トラブルシューティング

| 現象 | 主な原因 | 対処方法 |
|---|---|---|
| ステップ解説UIにならず、通常の文章として表示される | モデルの回答に「`## ステップ 1 見出し`」という見出しが含まれていない。 | プロンプトバーで「数学ステップ」チップがONになっているか確認してください。モデルが指示に従わなかった場合は「もう一度生成」を試してください。 |
| 「もっとやさしく」を押してもエラーになる | 1分間のレートリミット（20回）を超過した、または画像専用・音声専用モデルを使用している。 | 少し待ってから再試行するか、対応しているテキストモデル（Gemini, GPT-4o, Claude 3.7 Sonnet 等）で再生成してください。 |
| 図（SVG）が表示されない | 設定で「図の使用」が「使わない」になっている、またはモデルが図を不要と判断した。 | チップの歯車アイコンから設定を開き、「図の使用」を「必要なら使う（おすすめ）」に変更してください。 |
| 端末が重い、または図が激しく動いて見づらい | SVGアニメーションが動作している。 | 設定で「図のアニメーションを使う」をOFFにするか、OSのアクセシビリティ設定で「視差効果を減らす」をONにしてください。 |
