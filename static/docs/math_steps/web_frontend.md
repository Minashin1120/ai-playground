# 数学ステップ解説：Webフロントエンド実装

本ドキュメントでは、Web版クライアントにおける「数学ステップ解説」のフロントエンド実装（`static/js/chat_core_parts/chat_core.part30_mathsteps.js`）、UI構造、KaTeX数式および `mathsvg` の描画、サニタイズ処理、進捗同期、およびエクスポート仕様を解説します。

---

## 1. フロントエンド構成とモジュール設計

Web版の実装は、JavaScript分割構成ルールに基づき `static/js/chat_core_parts/chat_core.part30_mathsteps.js`（874行）に集約されています。

### 主な内部状態と管理構造
```javascript
const mathStepsUiState = {
    master: false,     // アカウント設定の全体スイッチ（math_steps_enabled）
    enabled: false,    // プロンプトバーのトグルチップのON/OFF状態
    config: { ... },   // ユーザーの詳細設定（basics, level, step_size, diagrams, animation）
    loaded: false,     // 設定ロード完了フラグ
    modalBusy: false   // 設定モーダルの保存通信中フラグ
};

// DOM要素とパース結果・タイマーを結び付ける WeakMap / WeakSet
const mathStepsRootState = new WeakMap();       // element -> MathStepsState
const mathStepsTextByRoot = new WeakMap();      // element -> パース済みステップ配列
const mathStepsProgressTimers = new WeakMap();  // element -> debounce timer ID
const mathStepsProgressInFlight = new WeakSet(); // element -> 通信中フラグ
```

---

## 2. UI構造とDOMツリー

各アシスタント回答の内部に `.math-steps-root` が生成され、1ステップずつフォーカスされた画面を構成します。

```text
.math-steps-root (data-current, data-max-reached, data-total-steps)
├── .math-steps-header
│   ├── .math-steps-counter  ("ステップ 2 / 4" などの進行状況)
│   └── .math-steps-title    ("使う公式を確認する" などの見出し)
├── .math-steps-body.prose
│   └── (KaTeX数式を含むMarkdown本文、および .mathsvg-render)
├── .math-steps-actions
│   ├── button.math-steps-next ("ステップ 3 へ" / "解き終わった！" などの大型ボタン)
│   └── .math-steps-sub-actions
│       ├── button.math-steps-prev ("前へ")
│       ├── button.math-steps-restart ("最初から")
│       └── button.math-steps-simplify ("もっとやさしく")
│   └── button.math-steps-restore ("元の説明に戻す" - 簡略化適用時のみ)
└── .math-steps-status (スクリーンリーダー向け状態通知 aria-live="polite")
```

---

## 3. 数式（KaTeX）と図解（`mathsvg`）の安全な描画

### 3.1 数式レンダリング（`renderMathStepsBodyHtml`）
本文（Markdown）に含まれるインライン数式（`$...$`）およびブロック数式（`$$...$$`）は、アプリ標準のKaTeXレンダラーを経由して高速かつ美しく表示されます（L140-L147）。

### 3.2 `mathsvg` サニタイズ処理（`sanitizeMathSvgMarkup`）
AIが生成したSVGマークアップは、XSS（クロスサイトスクリプティング）や不正な外部通信を防ぐため、厳格なサニタイズ（L149-L216）が行われます。

1. **悪意あるタグの完全除去**
   - `<script>`, `<style>`, `<foreignObject>`, `<iframe>`, `<frame>`, `<object>`, `<embed>`, `<link>`, `<meta>` などのタグを正規表現で完全に削除します。
2. **インラインイベント属性の除去**
   - `onload=`, `onclick=`, `onerror=` など `on[a-z]+=` にマッチするすべてのイベントハンドラ属性を削除します。
3. **外部URL・プロトコルの遮断**
   - `href` や `xlink:href` 属性に含まれる `http:`, `https:`, `//`, `javascript:`, `data:` を禁止し、SVG内のID参照（`#myGradient` 等）のみを許可します。
4. **SVGの完全性検証**
   - `<svg` で始まり `</svg>` で終わる完全なマークアップであること、および `viewBox` 属性が存在することを確認します。
   - 最大文字数は 300,000 文字（`MATH_STEPS_SVG_MAX_CHARS`）に制限されます。

### 3.3 アニメーションと視差効果軽減（`mathStepsReducedMotion`）
OSのアクセシビリティ設定（`prefers-reduced-motion: reduce`）が有効な場合、CSSおよびSVGアニメーションが自動的に抑制されます。
- `mathStepsReducedMotion()` が真の場合、`<animate>`, `<animateTransform>`, `<set>` などのSMILアニメーションタグを静止状態へ無効化します。
- CSSトランジションも `transition: none` となります。

---

## 4. 状態管理と進捗の同期

### 4.1 楽観的UI更新（Optimistic Update）
ユーザーが「次へ」ボタンをクリックすると：
1. `mathStepsRootState` の `current` がインクリメントされ、`max_reached` が更新されます。
2. 即座に `paintMathStepsRoot()` が呼び出され、待機時間なしで次のステップの本文が描画されます。
3. ボタンのテキストが「ステップ N+1 へ」または「解き終わった！」へ更新されます。

### 4.2 デバウンスによるサーバー送信（`scheduleMathStepsProgress`）
ボタンの連打によるサーバー負荷やレートリミット（120回/分）への抵触を防ぐため、進捗保存は **280ms** のデバウンスタイマーを介して非同期送信されます（L486-L534）。
- 通信エラー（429 Too Many Requests やネットワーク切断）が発生した場合、バックオフ付きで最大3回のリトライを行います。

### 4.3 「もっとやさしく」のインプレース切り替えと復元
1. 「もっとやさしく」ボタンを押すと、ボタンがビジー状態（スピナー表示・文言「生成中...」）になります。
2. `POST /api/math_steps/simplify` が完了すると、現在のステップ本文が簡略化された平易なテキストへ置き換わります。
3. 下部に破線枠の「元の説明に戻す」ボタンが現れます。
4. 「元の説明に戻す」を押すと、`remove_simplified: [stepIndex]` が送信され、瞬時に元の解説文へ戻ります。

---

## 5. エクスポートと共有の対応

### 5.1 スレッドPDFエクスポート（`buildMathStepsPdfHtml`）
チャット画面のPDFエクスポート機能では、画面上で折りたたまれている全ステップをフラットに展開し、印刷用レイアウトとして整形して出力します（L779-L793）。
- 各ステップが `<h3>ステップ N 見出し</h3>` と本文の形式で縦に並び、`mathsvg` も埋め込まれた状態で印刷・PDF化されます。
- 簡略化が適用されているステップは、簡略化されたテキストが出力されます。

### 5.2 クリップボードコピー（`prepareMathStepsCopyText`）
メッセージのコピー操作時には、全ステップが「`## ステップ N 見出し`」で整形されたプレーンテキスト（Markdown形式）としてクリップボードへコピーされます（L795-L817）。

---

## 6. CSSとスタイリング設計

`static/css/chat.custom.v4.8.1168.css` に定義されているスタイルは、モーダルデザインガイドおよびカラーパレットに準拠しています。

- **ダークモード / ライトモード対応**: `@media (prefers-color-scheme: light)` によるコントラスト調整。
- **モバイル対応**: 画面幅 640px 以下の環境では、ボタンの最小高さが `3.25rem` に拡張され、親指でタップしやすいサイズになります。
- **アクセシビリティ**: 適切な `aria-label`, `aria-live="polite"`, `role="region"` を設定。
