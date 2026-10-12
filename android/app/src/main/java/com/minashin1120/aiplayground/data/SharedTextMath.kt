package com.minashin1120.aiplayground.data

/**
 * メッセージとしてプレビュー (ANDROID_ONLY.md): some apps share an answer as plain text that keeps the TeX of
 * its formulas but drops the `$` / `\(` / `\[` delimiters. When the shared text has no math delimiter at all,
 * each run that contains a known math command is wrapped in `$…$`, or in `$$…$$` when it is the whole line,
 * so the preview draws it as math. Code blocks, code spans and `\begin{…}…\end{…}` (already drawn) stay as they are.
 */
object SharedTextMath {
    fun addMissingDelimiters(text: String): String {
        if (text.contains('$') || text.contains("\\(") || text.contains("\\[")) return text
        if (COMMAND.findAll(text).none { it.groupValues[1] in MATH_COMMANDS }) return text
        var fence: String? = null
        var environment: String? = null
        return text.lines().joinToString("\n") { line ->
            val trimmed = line.trimStart()
            when {
                fence != null -> { if (trimmed.startsWith(fence!!)) fence = null; line }
                trimmed.startsWith("```") || trimmed.startsWith("~~~") -> { fence = trimmed.take(3); line }
                environment != null -> { if (line.contains("\\end{$environment}")) environment = null; line }
                else -> {
                    BEGIN.findAll(line).lastOrNull()?.let { begin ->
                        if (!line.contains("\\end{${begin.groupValues[1]}}", ignoreCase = false)) environment = begin.groupValues[1]
                    }
                    wrapLine(line)
                }
            }
        }
    }

    private fun wrapLine(line: String): String {
        val prefix = PREFIX.find(line)?.value.orEmpty()
        val body = line.substring(prefix.length)
        val out = StringBuilder(prefix)
        var i = 0
        while (i < body.length) {
            when {
                body[i] == '`' -> {
                    val end = body.indexOf('`', i + 1)
                    val stop = if (end < 0) body.length else end + 1
                    out.append(body, i, stop); i = stop
                }
                body.startsWith("\\begin{", i) -> {
                    val name = BEGIN.find(body, i)?.groupValues?.get(1)
                    val close = name?.let { body.indexOf("\\end{$it}", i) } ?: -1
                    val stop = if (close < 0) body.length else close + "\\end{$name}".length
                    out.append(body, i, stop); i = stop
                }
                else -> {
                    val end = mathRunEnd(body, i)
                    if (end == i) {
                        // A long word is copied whole, so no run starts inside it.
                        var stop = i + 1
                        if (body[i].isAsciiLetter()) while (stop < body.length && body[stop].isAsciiLetter()) stop++
                        out.append(body, i, stop); i = stop; continue
                    }
                    val (start, stop) = trimRun(body, i, end)
                    val run = body.substring(start, stop)
                    if (start < stop && COMMAND.findAll(run).any { it.groupValues[1] in MATH_COMMANDS }) {
                        out.append(body, i, start)
                        val wholeLine = prefix.isEmpty() && body.substring(0, start).isBlank() && body.substring(stop).isBlank()
                        out.append(if (wholeLine) "$$$run$$" else "$$run$").append(body, stop, end)
                    } else {
                        out.append(body, i, end)
                    }
                    i = end
                }
            }
        }
        return out.toString()
    }

    /** End of the run of TeX-like tokens starting at [from]; a word of three or more letters or any other character ends it. */
    private fun mathRunEnd(s: String, from: Int): Int {
        var i = from
        while (i < s.length) {
            val c = s[i]
            when {
                c == '\\' && s.startsWith("\\begin{", i) -> return i
                c == '\\' && i + 1 < s.length && s[i + 1].isAsciiLetter() -> {
                    i++
                    while (i < s.length && s[i].isAsciiLetter()) i++
                }
                c == '\\' && i + 1 < s.length && s[i + 1] in ESCAPED -> i += 2
                c == '{' -> {
                    val close = matchingBrace(s, i) ?: return i
                    i = close + 1
                }
                c.isAsciiLetter() -> {
                    var j = i
                    while (j < s.length && s[j].isAsciiLetter()) j++
                    if (j - i >= 3) return i
                    i = j
                }
                c.isDigit() || c == ' ' || c in OPERATORS -> i++
                else -> return i
            }
        }
        return i
    }

    private fun matchingBrace(s: String, open: Int): Int? {
        var depth = 0
        var i = open
        while (i < s.length) {
            when (s[i]) {
                '\\' -> i++
                '{' -> depth++
                '}' -> if (--depth == 0) return i
                '`' -> return null
            }
            i++
        }
        return null
    }

    /** Drops spaces, sentence punctuation, unpaired brackets and short English words from both ends of a run. */
    private fun trimRun(s: String, from: Int, to: Int): Pair<Int, Int> {
        var start = from
        var end = to
        var changed = true
        while (changed && start < end) {
            changed = false
            while (start < end && s[start] in LEADING_TRIM) { start++; changed = true }
            while (end > start && s[end - 1] in TRAILING_TRIM && !(end - 2 >= start && s[end - 2] == '\\')) { end--; changed = true }
            STOP_WORD_START.find(s.substring(start, end))?.let { start += it.value.length; changed = true }
            STOP_WORD_END.find(s.substring(start, end))?.let { end -= it.value.length; changed = true }
        }
        return start to end
    }

    private fun Char.isAsciiLetter() = this in 'a'..'z' || this in 'A'..'Z'

    private val COMMAND = Regex("""\\([A-Za-z]+)""")
    private val BEGIN = Regex("""\\begin\{([A-Za-z*]+)}""")
    /** A Markdown list, quote or heading marker kept outside the math. */
    private val PREFIX = Regex("""^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s?|#{1,6}\s+)""")
    private const val OPERATORS = "+-=<>/*^_()[]|,.!':;~&"
    private const val ESCAPED = "\\{}|,;:!# %&_"
    private const val LEADING_TRIM = " ,.;:)]"
    private const val TRAILING_TRIM = " ,.;:([&"
    private const val STOP_WORDS = "is|of|to|in|at|be|as|by|on|or|an|if|it|we|so"
    private val STOP_WORD_START = Regex("""^(?i:$STOP_WORDS)(?=\s)""")
    private val STOP_WORD_END = Regex("""(?<=\s)(?i:$STOP_WORDS)$""")

    /** Commands that only appear in formulas, so a Windows path or an escaped `\n` is not taken for math. */
    private val MATH_COMMANDS = setOf(
        "frac", "dfrac", "tfrac", "cfrac", "sqrt", "sum", "prod", "coprod", "int", "iint", "iiint", "oint", "lim", "limsup", "liminf",
        "infty", "partial", "nabla", "cdot", "cdots", "ldots", "dots", "vdots", "ddots", "times", "div", "pm", "mp", "ast", "star",
        "leq", "geq", "le", "ge", "neq", "ne", "approx", "equiv", "sim", "simeq", "cong", "propto", "ll", "gg",
        "to", "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "leftrightarrow", "Leftrightarrow", "longrightarrow",
        "Longrightarrow", "mapsto", "implies", "iff", "in", "notin", "ni", "subset", "subseteq", "supset", "supseteq",
        "cup", "cap", "bigcup", "bigcap", "setminus", "emptyset", "varnothing", "forall", "exists", "neg", "land", "lor", "oplus", "otimes",
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota", "kappa", "lambda",
        "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
        "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "log", "ln", "lg", "exp",
        "max", "min", "sup", "inf", "arg", "det", "dim", "ker", "gcd", "deg", "Pr",
        "mathbb", "mathrm", "mathbf", "mathcal", "mathit", "mathsf", "mathfrak", "boldsymbol", "operatorname", "text", "textrm", "textbf",
        "left", "right", "bigl", "bigr", "Bigl", "Bigr", "big", "Big", "hat", "widehat", "bar", "vec", "dot", "ddot", "tilde", "widetilde",
        "overline", "underline", "overbrace", "underbrace", "overrightarrow", "binom", "dbinom", "pmod", "bmod", "mod",
        "quad", "qquad", "angle", "circ", "perp", "parallel", "mid", "ell", "hbar", "Re", "Im", "aleph",
        "lfloor", "rfloor", "lceil", "rceil", "langle", "rangle", "vert", "Vert", "because", "therefore", "displaystyle", "limits",
    )
}
