package com.minashin1120.aiplayground.ui

/**
 * A bounded, native LaTeX-to-text renderer. The Web build typesets with MathJax;
 * the Android client intentionally renders a readable approximation without a WebView
 * or bundled JavaScript. Unsupported commands fall back to their literal form instead
 * of failing, so no markup is ever executed.
 */
internal fun latexToDisplay(tex: String): String {
    var value = tex.trim()
    if (value.isEmpty()) return ""

    // Escaped literals must be restored before command handling.
    val escaped = mapOf(
        "\\%" to "%", "\\$" to "$", "\\#" to "#",
        "\\_" to "_", "\\{" to "{", "\\}" to "}", "\\&" to "&",
    )
    escaped.forEach { (from, to) -> value = value.replace(from, to) }

    // Spacing and sizing commands carry no meaning in a text approximation.
    listOf(
        "\\left", "\\right", "\\displaystyle", "\\limits", "\\nolimits",
        "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\big", "\\Big",
    ).forEach { value = value.replace(it, "") }
    listOf("\\,", "\\!", "\\;", "\\:", "\\ ").forEach { value = value.replace(it, " ") }
    value = value.replace("\\quad", "  ").replace("\\qquad", "    ")
    value = value.replace(Regex("""\\begin\{[A-Za-z*]+}"""), "").replace(Regex("""\\end\{[A-Za-z*]+}"""), "")

    // Layout commands are resolved innermost-first so nested braces stay safe.
    var pass = 0
    while (pass < 6) {
        val before = value
        value = Regex("""\\(?:d|t)?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}""")
            .replace(value) { "(${it.groupValues[1]})/(${it.groupValues[2]})" }
        value = Regex("""\\sqrt\s*\[([^{}]*)\]\s*\{([^{}]*)\}""")
            .replace(value) { "${it.groupValues[1]}\u221A(${it.groupValues[2]})" }
        value = Regex("""\\sqrt\s*\{([^{}]*)\}""")
            .replace(value) { "\u221A(${it.groupValues[1]})" }
        value = Regex("""\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}""")
            .replace(value) { it.groupValues[1] }
        value = Regex("""\\overline\s*\{([^{}]*)\}""").replace(value) { it.groupValues[1] + "\u0305" }
        value = Regex("""\\underline\s*\{([^{}]*)\}""").replace(value) { it.groupValues[1] + "\u0332" }
        value = Regex("""\\hat\s*\{([^{}]*)\}""").replace(value) { it.groupValues[1] + "\u0302" }
        value = Regex("""\\vec\s*\{([^{}]*)\}""").replace(value) { it.groupValues[1] + "\u20D7" }
        pass++
        if (value == before) break
    }

    // Command names are mapped as whole tokens so \\int and \\infty stay distinct.
    val symbols = mapOf(
        "infty" to "\u221E", "sum" to "\u2211", "prod" to "\u220F", "int" to "\u222B",
        "oint" to "\u222E", "partial" to "\u2202", "nabla" to "\u2207",
        "pm" to "\u00B1", "mp" to "\u2213", "times" to "\u00D7", "cdot" to "\u00B7",
        "div" to "\u00F7", "ast" to "\u2217",
        "leq" to "\u2264", "le" to "\u2264", "geq" to "\u2265", "ge" to "\u2265",
        "neq" to "\u2260", "ne" to "\u2260", "approx" to "\u2248", "equiv" to "\u2261",
        "cong" to "\u2245", "propto" to "\u221D",
        "rightarrow" to "\u2192", "to" to "\u2192", "leftarrow" to "\u2190",
        "Rightarrow" to "\u21D2", "Leftarrow" to "\u21D0", "leftrightarrow" to "\u2194",
        "mapsto" to "\u21A6", "in" to "\u2208", "notin" to "\u2209", "subset" to "\u2282",
        "supset" to "\u2283", "subseteq" to "\u2286", "cup" to "\u222A", "cap" to "\u2229",
        "emptyset" to "\u2205", "forall" to "\u2200", "exists" to "\u2203", "neg" to "\u00AC",
        "land" to "\u2227", "lor" to "\u2228", "angle" to "\u2220", "degree" to "\u00B0",
        "alpha" to "\u03B1", "beta" to "\u03B2", "gamma" to "\u03B3", "delta" to "\u03B4",
        "epsilon" to "\u03B5", "varepsilon" to "\u03B5", "zeta" to "\u03B6", "eta" to "\u03B7",
        "theta" to "\u03B8", "vartheta" to "\u03D1", "iota" to "\u03B9", "kappa" to "\u03BA",
        "lambda" to "\u03BB", "mu" to "\u03BC", "nu" to "\u03BD", "xi" to "\u03BE",
        "pi" to "\u03C0", "rho" to "\u03C1", "sigma" to "\u03C3", "tau" to "\u03C4",
        "upsilon" to "\u03C5", "phi" to "\u03C6", "varphi" to "\u03D5", "chi" to "\u03C7",
        "psi" to "\u03C8", "omega" to "\u03C9", "Gamma" to "\u0393", "Delta" to "\u0394",
        "Theta" to "\u0398", "Lambda" to "\u039B", "Xi" to "\u039E", "Pi" to "\u03A0",
        "Sigma" to "\u03A3", "Phi" to "\u03A6", "Psi" to "\u03A8", "Omega" to "\u03A9",
        "cdots" to "\u22EF", "ldots" to "\u2026", "dots" to "\u2026", "vdots" to "\u22EE",
        "ddots" to "\u22F1",
    )
    value = Regex("""\\([A-Za-z]+)""").replace(value) { match ->
        symbols[match.groupValues[1]] ?: match.value
    }

    // Scripts are resolved before braces are dropped so x^{n+1} stays intact.
    value = Regex("""\^\{([^{}]*)\}""").replace(value) { script(it.groupValues[1], SUPERSCRIPTS, "^") }
    value = Regex("""\^(\S)""").replace(value) { script(it.groupValues[1], SUPERSCRIPTS, "^") }
    value = Regex("""_\{([^{}]*)\}""").replace(value) { script(it.groupValues[1], SUBSCRIPTS, "_") }
    value = Regex("""_(\S)""").replace(value) { script(it.groupValues[1], SUBSCRIPTS, "_") }

    value = value.replace("\\\\", "\n").replace("&", "  ")
    value = value.replace("{", "").replace("}", "")
    return value.trim()
}

private fun script(body: String, table: Map<Char, Char>, prefix: String): String {
    if (body.isEmpty()) return ""
    val builder = StringBuilder()
    for (character in body) {
        val mapped = table[character] ?: return "$prefix($body)"
        builder.append(mapped)
    }
    return builder.toString()
}

private val SUPERSCRIPTS = mapOf(
    '0' to '\u2070', '1' to '\u00B9', '2' to '\u00B2', '3' to '\u00B3', '4' to '\u2074',
    '5' to '\u2075', '6' to '\u2076', '7' to '\u2077', '8' to '\u2078', '9' to '\u2079',
    '+' to '\u207A', '-' to '\u207B', '=' to '\u207C', '(' to '\u207D', ')' to '\u207E',
    'n' to '\u207F', 'i' to '\u2071', 'a' to '\u1D43', 'b' to '\u1D47', 'c' to '\u1D9C',
    'd' to '\u1D48', 'e' to '\u1D49', 'f' to '\u1DA0', 'g' to '\u1D4D', 'h' to '\u02B0',
    'j' to '\u02B2', 'k' to '\u1D4F', 'l' to '\u02E1', 'm' to '\u1D50', 'o' to '\u1D52',
    'p' to '\u1D56', 'r' to '\u02B3', 's' to '\u02E2', 't' to '\u1D57', 'u' to '\u1D58',
    'v' to '\u1D5B', 'w' to '\u02B7', 'x' to '\u02E3', 'y' to '\u02B8', 'z' to '\u1DBB',
)

private val SUBSCRIPTS = mapOf(
    '0' to '\u2080', '1' to '\u2081', '2' to '\u2082', '3' to '\u2083', '4' to '\u2084',
    '5' to '\u2085', '6' to '\u2086', '7' to '\u2087', '8' to '\u2088', '9' to '\u2089',
    '+' to '\u208A', '-' to '\u208B', '=' to '\u208C', '(' to '\u208D', ')' to '\u208E',
    'a' to '\u2090', 'e' to '\u2091', 'h' to '\u2095', 'i' to '\u1D62', 'j' to '\u2C7C',
    'k' to '\u2096', 'l' to '\u2097', 'm' to '\u2098', 'n' to '\u2099', 'o' to '\u2092',
    'p' to '\u209A', 'r' to '\u1D63', 's' to '\u209B', 't' to '\u209C', 'u' to '\u1D64',
    'v' to '\u1D65', 'x' to '\u2093',
)
