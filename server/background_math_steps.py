# --- Math steps (step-by-step math explanation) ---
# The answer format is parsed by Web/Android, so this text is format-critical and
# intentionally fixed. It is registered in tests/test_no_hardcoded_model_prompts_regressions.py.
import json
import re

MATH_STEPS_DEFAULTS = {
    "basics": False,
    "level": "easy",
    "step_size": "normal",
    "diagrams": "auto",
    "animation": True,
}
MATH_STEPS_LEVELS = ("easy", "normal", "detailed")
MATH_STEPS_STEP_SIZES = ("small", "normal", "large")
MATH_STEPS_DIAGRAM_MODES = ("auto", "off")
MATH_STEPS_MAX_CURRENT = 50
MATH_STEPS_MAX_SIMPLIFIED_PER_STEP = 20_000
MATH_STEPS_MAX_SIMPLIFIED_TOTAL = 100_000
MATH_STEPS_MAX_STEPS = 50
# 「もっとやさしく」 splits one step into smaller steps. The stored list replaces
# the original step, so the expanded text is bounded like a simplified variant.
MATH_STEPS_MAX_EXPANDED_PER_STEP = 8
MATH_STEPS_MAX_EXPANDED_TITLE = 120

MATH_STEPS_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:ステップ|Step)\s*(\d+)\s*(.*)$", re.IGNORECASE
)
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")

MATH_STEPS_FORMAT_PROMPT = """[Math Steps]
あなたは数学に苦手意識を持つ学習者を支える、辛抱強い数学の家庭教師です。相手は計算を途中で諦めやすいので、結論を急がず、1つずつ確実に進めてください。

話し方:
- やさしい日本語で書く。専門用語は使う前に短く定義する。
- 記号や文字が何を表すかを最初に説明する（例: x は求めたい個数）。
- 計算は途中の行を省略しない。式変形は1行ずつ示す。
- 「なぜそうするのか」を短く添える。長い講義にしない。
- 答えを出したら、元の問題に当てはめて確認する。
- 選択式・クイズ・採点・「考えてみましょう」だけで終わる書き方はしない。

手順の分け方:
- 1ステップにつき1つの考え方だけを扱う。
- 見出しは「## ステップ N 見出し」の形式を厳守する（N は1から連番）。見出しは15文字以内の日本語。
- 最終ステップの見出しは「答え」または「答えの確認」にする。
- 後のステップや最終答えを先に明かさない。前のステップの結果だけを使う。
- ステップ数は {{math_step_size}}。全部で最大{{math_max_steps}}ステップまで。
- 難易度: {{math_level}}。
- 基礎から説明: {{math_basics}}。
- 回答は、ステップ1から最終ステップまでを1回で全部書く（クライアントが1ステップずつ表示する）。続きから説明する場合も、途中のステップだけで止めない。

ノートの続きから説明するとき（途中まで解いた画像・書きかけの計算）:
- 画像やPDFに、印刷された問題だけでなく手書き・入力済みの途中式がある場合、それは学習者のノートです。問題を最初から解き直さない。
- ノートの最後に書かれている式を読み取り、その次の行から説明を始める。最初の見出しを「問題を整理する」にしない。「続きを確認する」「まちがいを直す」など、続きから始まる見出しにする。
- 合っている部分を最初から復習し直さない。どこまで合っているかを1〜2文で示す（例:「3行目の 2x=6 までは合っています。ここから続けます。」）。
- 途中に間違いがあるときは、指摘だけで終わらせない。次の3つを必ず同じ回答に入れる:
  1. 消す場所を、行番号と式の両方で具体的に示し、消しゴムで消すよう伝える（例:「3行目の『2x+3=9』を消しゴムで消してください」。符号や数字の一部だけが違うときは「『+3』だけを消してください」。タブレットなら「その行を取り消してください」）。行番号が分かりにくい画像では、式そのものを引用して「この式」と示す。
  2. 消したあとに書き直す正しい式を、そのまま書く（例:「かわりに『2x=6』と書きます」）。
  3. 書き直した行から、途中式を省略せずに答えまで続ける。
- 間違いが広い範囲に及ぶときは「4行目から最後までを消してください」と範囲で伝え、その行から正しい方針でやり直す。問題の最初から説明し直さない。
- 「どこが違うか考えてみましょう」「この続きを計算してみましょう」だけで終わらない。ユーザーが「合ってる？」と聞いただけでも、修正後の計算と答えまで必ず書く。
- 間違いがないときは、そのことを短く伝え、答えの確認（検算）まで行う。問題を最初から解き直さない。
- ノートの式は学習者の書き方（文字・順序）を尊重して読み取る。読み取れない箇所だけ「その行だけ書いてください」と1文で頼み、推測で別の式に置き換えない。
- 問題文が写っていないときは、ノートの式から何を求めるかを推測し、最初に1文だけ確認する（例:「x を求める問題として続けます」）。
- 「基礎から説明」がONでも、続きの場合は前提の復習から始めない。必要な前提は、修正や次の計算に必要な分だけ短く添える。
- 手書き・入力済みの途中式がなく、問題文だけの画像なら、これまでどおりステップ1から解く。
- 残りの計算に必要なステップ数だけにする。ステップ数を水増ししない。

出力形式（厳守・HTMLやJSONは禁止。以下は最初から解く場合の例）:
## ステップ 1 問題を整理する
本文（Markdown。数式は LaTeX 可）

```mathsvg
<svg viewBox="0 0 480 240" role="img" aria-label="図の説明">...</svg>
```

## ステップ 2 使う公式を確認する
本文...

ノートの続きから説明する場合の例:
## ステップ 1 まちがいを直す
3行目の「2x + 3 = 9」を消しゴムで消して、「2x = 6」と書き直してください。
...
## ステップ 2 続きを計算する
...
## ステップ 3 答えの確認
...

図について:
- 図が本当に理解を助けるときだけ mathsvg を使う。1つの回答で最大2個。
- mathsvg の中には完全な <svg> を1つだけ入れる。viewBox を必ず付ける。
- <script>、<foreignObject>、外部URL、<image href="http..."> は禁止。
- 動きが必要な場合だけ SMIL（animate / animateTransform / set）または CSS @keyframes を使う。アニメーション: {{math_animation}}。
- 図の設定: {{math_diagrams}}。不要なら mathsvg を省く。
"""


def _as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "on", "yes"):
            return True
        if lowered in ("0", "false", "off", "no", ""):
            return False
    return bool(default)


def _as_choice(value, allowed, default):
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def normalize_math_steps_config(value):
    """Return the clamped per-user math steps config (always a full dict)."""
    raw = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except (TypeError, ValueError):
            raw = None
    if not isinstance(raw, dict):
        raw = {}
    return {
        "basics": _as_bool(raw.get("basics"), MATH_STEPS_DEFAULTS["basics"]),
        "level": _as_choice(raw.get("level"), MATH_STEPS_LEVELS, MATH_STEPS_DEFAULTS["level"]),
        "step_size": _as_choice(
            raw.get("step_size"), MATH_STEPS_STEP_SIZES, MATH_STEPS_DEFAULTS["step_size"]
        ),
        "diagrams": _as_choice(
            raw.get("diagrams"), MATH_STEPS_DIAGRAM_MODES, MATH_STEPS_DEFAULTS["diagrams"]
        ),
        "animation": _as_bool(raw.get("animation"), MATH_STEPS_DEFAULTS["animation"]),
    }


def dump_math_steps_config(value):
    """Serialize a normalized config for storage in the TEXT column."""
    return json.dumps(normalize_math_steps_config(value), ensure_ascii=False, separators=(",", ":"))


def _level_instruction(level):
    return {
        "easy": "小学高学年でも追えるやさしさ。言い換えを多めに。",
        "normal": "高校生くらいの標準的な説明。",
        "detailed": "式の意味と根拠を丁寧に。ただし1ステップ1話題は守る。",
    }[level]


def _step_size_instruction(step_size):
    return {
        "small": "小さめ（1ステップは短く、式変形を細かく分ける）",
        "normal": "標準（1ステップ1つの考え方）",
        "large": "大きめ（関連する計算をまとめるが、途中式は省略しない）",
    }[step_size]


def _diagram_instruction(diagrams):
    if diagrams == "off":
        return "図は使わない（mathsvg を出力しない）"
    return "必要なときだけ使う（最大2個）"


def _basics_instruction(basics):
    if basics:
        return (
            "ON。最初のステップで、この問題に必要な前提の定義・公式を、"
            "小さな具体例つきで確認する。その後に問題を解き始める"
            "（ノートの続きから説明するときは、前提の復習を挟まずノートの確認から始める）。"
        )
    return (
        "OFF。前提の復習は最小限にし、問題の整理から始める"
        "（ノートの続きから説明するときは、ノートの確認から始める）。"
    )


def build_math_steps_system_prompt(config):
    """Fill the fixed format template with the user's normalized settings."""
    normalized = normalize_math_steps_config(config)
    return (
        MATH_STEPS_FORMAT_PROMPT
        .replace("{{math_basics}}", _basics_instruction(normalized["basics"]))
        .replace("{{math_level}}", _level_instruction(normalized["level"]))
        .replace("{{math_step_size}}", _step_size_instruction(normalized["step_size"]))
        .replace("{{math_diagrams}}", _diagram_instruction(normalized["diagrams"]))
        .replace("{{math_animation}}", "使ってよい" if normalized["animation"] else "使わない（静止画のみ）")
        .replace("{{math_max_steps}}", str(MATH_STEPS_MAX_STEPS))
    )


def build_math_steps_simplify_prompt(config):
    """Return the split-and-rewrite instruction (the caller appends the step body).

    The answer keeps the normal step-heading format because the Web/Android clients
    splice the returned steps into the answer and let the reader continue there.
    """
    normalized = normalize_math_steps_config(config)
    tone = _level_instruction(normalized["level"])
    return (
        "[Math Steps / もっとやさしく]\n"
        "あなたは数学に苦手意識を持つ学習者の隣にいる家庭教師です。"
        "以下の1ステップを、意味や計算を変えずに、もっとやさしい日本語の"
        "2〜4個の小さなステップに分けて書き直してください。\n"
        "- 1ステップにつき1つの考え方だけを扱い、前のステップの結果だけを使う。\n"
        "- 見出しは「## ステップ N 見出し」の形式を厳守する（N は1から連番、見出しは15文字以内）。\n"
        "- 計算は途中の行を省略しない。式変形は1行ずつ示す。\n"
        "- 記号や文字が何を表すかを言葉で添える。\n"
        "- 元の意味・計算・答えを変えない。\n"
        "- 元のステップに mathsvg がある場合だけ、いずれか1つのステップに同じ内容の mathsvg を残してよい。\n"
        f"- やさしさの目安: {tone}\n"
        "- HTML・JSON・前置き・後書きは禁止。見出しと本文以外を書かない。\n\n"
        "分けるステップ:\n"
    )


def extract_math_steps(text):
    """Split an assistant answer into [{index, title, body}] using the step headings.

    Headings inside fenced code blocks (including mathsvg) are ignored. Returns an
    empty list when the answer has no step heading, so callers fall back to normal
    Markdown rendering.
    """
    raw = str(text or "")
    if not raw:
        return []
    steps = []
    preamble = []
    current = None
    fence = None
    for line in raw.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            (current["lines"] if current is not None else preamble).append(line)
            continue
        if fence is None:
            heading = MATH_STEPS_HEADING_RE.match(line)
            if heading:
                title = heading.group(2).strip()
                current = {
                    "index": len(steps),
                    "number": int(heading.group(1)),
                    "title": title,
                    "lines": [],
                }
                steps.append(current)
                continue
        (current["lines"] if current is not None else preamble).append(line)
    result = []
    for step in steps:
        body = "\n".join(step["lines"]).strip()
        if step["index"] == 0 and preamble:
            prefix = "\n".join(preamble).strip()
            body = f"{prefix}\n\n{body}".strip() if body else prefix
        result.append({
            "index": step["index"],
            "number": step["number"],
            "title": step["title"],
            "body": body,
        })
    return result
