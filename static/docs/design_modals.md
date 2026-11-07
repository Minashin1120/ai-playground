# モーダル デザインガイド

この文書は AI Playground のモーダル（ダイアログ）のデザイン言語を定義する。見た目の基準は **設定モーダル** と **モデル選択モーダル** で、今後追加するモーダルはすべてこの言語に従う。

- 実装の正本（チャット）: `static/css/chat.custom.v*.css` の「Shared modal design」
- 実装の正本（公開ページ）: `static/css/app-design.css` の「Shared modal design」
- ライトテーマ: `static/css/theme-light.css` / `static/css/theme-light-manual.css`
- Android: `android/app/src/main/java/com/minashin1120/aiplayground/ui/PlaygroundDialog.kt`

> チャット画面は `app-design.css` を読み込まないため、同じブロックを2つのCSSに置いている。値を変えるときは必ず両方を同じ変更で更新する。

## 1. 原則

1. **1つのモーダル言語**: パネル・ヘッダー・本文・フッターの4層構造を共通化する。モーダルごとに独自の背景・角丸・ヘッダーを作らない。
2. **設定モーダルが基準**: 背景グラデーション、22pxの角丸、テーマ色のヘッダー、44pxのアイコン、38pxの閉じるボタンを踏襲する。
3. **内容より枠を優先**: 本文の中身（設定カード、一覧、フォーム）は機能に合わせてよい。枠・余白・見出し・ボタンのルールは変えない。
4. **ライト・ダーク両対応**: 新しい色を直書きするときは、必ずライトテーマの上書きも同じ変更で追加する。
5. **セマンティックな色はアイコンだけ**: 警告・危険・情報などの意味は `app-modal-title-icon--*` の色で表し、パネル全体の色は変えない。

## 2. 構造（Anatomy）

```html
<div id="example-modal"
     class="fixed inset-0 bg-gray-900/95 z-50 hidden items-center justify-center backdrop-blur-sm modal-overlay app-modal-overlay">
  <div class="app-modal-panel app-modal-panel--md modal-panel" role="dialog" aria-modal="true" aria-labelledby="example-modal-title">
    <div class="app-modal-header">
      <div class="app-modal-title-wrap">
        <div class="app-modal-title-icon"><i class="fas fa-circle-info"></i></div>
        <div class="app-modal-title-text">
          <h2 id="example-modal-title" class="app-modal-title">タイトル</h2>
          <p class="app-modal-subtitle">何をするモーダルかを1行で</p>
        </div>
      </div>
      <button type="button" class="app-modal-close" aria-label="閉じる">
        <i class="fas fa-times"></i>
      </button>
    </div>

    <div class="app-modal-body">
      <!-- 本文 -->
    </div>

    <div class="app-modal-footer">
      <button type="button" class="app-modal-btn app-modal-btn--ghost">キャンセル</button>
      <button type="button" class="app-modal-btn app-modal-btn--primary">保存</button>
    </div>
  </div>
</div>
```

### 2.1 オーバーレイ

| 項目 | 値 |
|---|---|
| 背景 | `rgba(3, 7, 16, 0.72)`（ライト: `rgba(15, 23, 42, 0.35)`） |
| ぼかし | `blur(10px) saturate(140%)` |
| 余白 | 16px（モバイルは 8px） |
| 開閉 | `.modal-overlay` の `modal-open` / `modal-close` に追従 |

- 必ず `modal-overlay` と `app-modal-overlay` の両方を付ける。`modal-overlay` は開閉アニメーション、`app-modal-overlay` は見た目を担当する。
- 背景色を Tailwind の `bg-gray-900/95` などで上書きしない（`app-modal-overlay` が優先される）。

### 2.2 パネル

| 項目 | 値 |
|---|---|
| 角丸 | 22px（モバイル 18px） |
| 背景 | `linear-gradient(180deg, rgba(10,14,28,.98), rgba(6,9,18,.99))` |
| 境界線 | `1px solid rgba(255,255,255,.08)` |
| 影 | `inset 0 1px 0 rgba(255,255,255,.05), var(--shadow)` |
| 最大高さ | `min(90dvh, 880px)` |
| レイアウト | `flex` 縦並び、`overflow: hidden` |

幅は次の修飾子で決める。本文の量に合わせて選び、`max-w-*` を直接付けない。

| クラス | 幅 | 用途 |
|---|---|---|
| `app-modal-panel--sm` | 440px | 短い確認、トークン詳細 |
| `app-modal-panel--md` | 560px | フォーム、警告、カメラ |
| `app-modal-panel--lg` | 760px | 設定詳細、履歴、画像編集 |
| `app-modal-panel--xl` | 980px | 大きな一覧・ブランチ管理 |
| `app-modal-panel--full` | 1100px・高さ固定 | インポート選択など全画面に近い操作 |

高さを固定して一覧をスクロールさせたいときは `app-modal-panel--tall` を併用する。`--full` / `--tall` はモバイルで画面いっぱいになる。

### 2.3 ヘッダー

- 高さは内容に応じて可変。上下 16px / 左右 20px（モバイル 14px / 8〜14px）。
- 背景はテーマ色の淡いグラデーション: `linear-gradient(180deg, rgba(var(--theme-rgb), .12), transparent)`。
- タイトルアイコンは 44×44px、角丸14px、テーマ色のグラデーション＋グロー。
- タイトルは 18px / 700。サブタイトルは 12px / `var(--muted)`。
- 閉じるボタンは 38×38px、角丸12px。ホバーで90度回転する。
- ヘッダー内に操作ボタンを置く場合は `.app-modal-header-actions` でまとめる（例: Lyria の保存ボタン）。

アイコンの意味色は次の修飾子で表す。既定はテーマ色。

| クラス | 色 | 例 |
|---|---|---|
| なし | テーマ色 | 設定、モデル、履歴 |
| `app-modal-title-icon--info` | 青 | 利用規約、トークン詳細 |
| `app-modal-title-icon--success` | 緑 | 保存・添付・完了 |
| `app-modal-title-icon--warn` | アンバー | 警告、確認、画像編集 |
| `app-modal-title-icon--danger` | 赤 | 削除、APIキー不足、BAN |
| `app-modal-title-icon--purple` | 紫 | 音楽・ブランチ |
| `app-modal-title-icon--cyan` | シアン | 音声・ファイル使用状況 |

### 2.4 ツールバー／検索

- ヘッダー直下の検索やフィルターは `.app-modal-toolbar` に入れる。
- 検索入力は `.app-modal-search` で包み、左に虫眼鏡アイコンを置く。
- 設定・モデルモーダルの既存ツールバーはこの見た目に合わせてある。

### 2.5 本文

- `.app-modal-body` がスクロール領域。`flex: 1 1 auto`、縦スクロール、`overscroll-behavior: contain`。
- 余白は 18px / 20px（モバイル 14px）。
- 背景は右上にテーマ色のごく淡い放射グラデーション。
- 全面グリッドや独自スクロールを持つ本文は `app-modal-body--flush` を使い、余白を内側の要素に任せる。
- 本文内のカードは設定モーダルの `.settings-card` に近い面（`rgba(255,255,255,.03)` 前後、角丸12〜16px、薄い境界線）を使う。

### 2.6 フッター

- `.app-modal-footer` は右寄せ・折り返しあり。上に境界線、下方向の暗いグラデーション。
- ボタンは `.app-modal-btn` を基本とし、次の修飾子を使う。
  - `app-modal-btn--ghost`: キャンセル・閉じる
  - `app-modal-btn--primary`: 保存・送信・決定（テーマ色）
  - `app-modal-btn--danger`: 削除・危険な操作
- 左側に補足情報を置く場合は `justify-between` を追加する（インポート確認など）。
- 原則として「キャンセル（ghost）→ 主操作（primary）」の順に並べる。

### 2.7 アクセシビリティ

- パネルに `role="dialog"` `aria-modal="true"` `aria-labelledby="<タイトルID>"` を付ける。
- 閉じるボタンには `aria-label="閉じる"` を付ける。
- タイトルは `h2`、本文の見出しは `h3` 以下にする。
- アイコンだけのボタンには必ず `aria-label` か `title` を付ける。
- フォーカス可能な要素を `display:none` で隠す場合、モーダルを閉じたあとに元の要素へフォーカスを戻す（既存の `showModal` / `hideModal` に従う）。

## 3. 動作

- 開閉は既存の `showModal(id)` / `hideModal(id)` を使う。`hidden` の付け外しと `modal-open` / `modal-close` の付与は共通処理に任せる。
- 閉じる操作は、フッターのボタンと同じID・関数に接続する。ヘッダーの閉じるボタンから別処理を新設しない。
- 背景クリックで閉じるかどうかは既存の挙動を維持する。設定・モデルのように背景クリックで閉じないモーダルもある。
- 二重スクロールを作らない。パネルは `overflow: hidden`、スクロールは `.app-modal-body` または一覧要素だけに持たせる。

## 4. ライトテーマ

ライトテーマでは次の値を上書きする（`theme-light.css` / `theme-light-manual.css`）。

- パネル: 白〜`#f7f9fc` のグラデーション、境界線 `rgba(15,23,42,.10)`
- ヘッダー: テーマ色 10% のグラデーション
- 閉じるボタン: `rgba(15,23,42,.04)`、ホバー `.08`
- フッター: 白系グラデーション
- 入力欄: 白背景、境界線 `rgba(15,23,42,.14)`

新しいモーダルを追加したときは、ライトテーマで次を確認する。

- パネルと本文の境目が見えるか
- 入力欄の文字が読めるか
- アイコンとサブタイトルが薄すぎないか
- フッターのボタンが背景に埋もれていないか

## 5. モバイル

- 768px以下では余白8px、パネルは幅100%、角丸18px。
- `--full` / `--tall` は `calc(100dvh - 16px)` の高さになる。
- ヘッダーは 14px / 8px、タイトル16px、サブタイトル11px、本文14px。
- フッターのボタンは折り返してよい。横に収まらない場合は `app-modal-btn` の padding を維持したまま折り返す。

## 6. やってはいけないこと

- モーダルごとに `bg-gray-800 rounded-lg p-6` の独自パネルを作る。
- ヘッダーを `h2` 1行だけで済ませる（アイコン・サブタイトル・閉じるボタンがない）。
- `max-w-*` や `p-*` で共通パネルの幅・余白を上書きする。
- 閉じるボタンを絶対配置の `✕` や `text-gray-400` だけで済ませる。
- ダーク専用の色を直書きして、ライトテーマの上書きを忘れる。
- モーダル全体を警告色・危険色にする（意味色はアイコンだけ）。

## 7. 新しいモーダルを追加するチェックリスト

1. 既存のモーダルから近いものを選び、`app-modal-overlay` / `app-modal-panel` / `app-modal-header` / `app-modal-body` / `app-modal-footer` を使う。
2. 幅は `--sm` 〜 `--full` から選ぶ。高さ固定が必要なら `--tall` を足す。
3. タイトル・サブタイトル・アイコン・閉じるボタンを揃える。
4. フッターのボタンを `app-modal-btn--ghost` / `--primary` / `--danger` に置き換える。
5. ライトテーマで表示を確認する。
6. モバイル幅（375px前後）で、本文がスクロールでき、フッターが画面外に出ないことを確認する。
7. Android版に同じ機能がある場合は、`PlaygroundDialog` の同じ構造で表示されることを確認する。

## 8. 現在の適用状況

| 画面 | モーダル | アイコン色 |
|---|---|---|
| 設定 | 設定 | テーマ |
| モデル | モデル選択 | テーマ |
| 履歴 | チャット履歴 | info |
| チャット設定 | Chat Instructions | success |
| 画像 | 画像・圧縮詳細設定 | info |
| 画像 | 画像を分割 / 分割の圧縮 | info |
| アップロード | ファイルアップロード | info |
| アップロード | リッチ貼り付け | warn |
| アップロード | カメラ撮影 | success |
| ライブラリ | ファイルライブラリ | テーマ |
| ライブラリ | 使用されているチャット | cyan |
| ライブラリ | 利用規約 | info |
| ライブラリ | アルファ版の注意 | warn |
| ライブラリ | ブランチ管理 | purple |
| 管理者 | アカウント管理 / Bot検出ログ | danger |
| 音声 | Lyria RealTime Studio | purple |
| 音声 | 音声スタジオ | cyan |
| 画像編集 | 画像編集 / マスク作成 | warn |
| 画像編集 | 画像分割 | info |
| 確認 | チャット名変更 | テーマ |
| 確認 | 外部ツール操作 | warn |
| 確認 | ローカルPython実行 | warn |
| 確認 | 設定インポート | info |
| 情報 | トークン詳細 | info |
| 情報 | Python実行結果 | warn |
| 情報 | 暗号化状態 | info |
| 情報 | 新バージョン通知 | info |
| 警告 | ブラウザ直接送信・高速モード | warn |
| 警告 | APIキーが必要 | danger |
| 公開ページ | 利用規約（landing / login / help） | info |
| セットアップ | インポートファイル / 設定確認 | success / info |
| Batch | Batch処理 | テーマ |

設定・モデル・ライブラリ・Batch は独自クラス（`settings-*` / `model-*` / `lib-*` / `batch-*`）を持つが、見た目はこのガイドの値に合わせている。新しい共通クラスへ移行する場合も、この表の見た目を変えない。
