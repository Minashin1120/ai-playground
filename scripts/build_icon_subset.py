#!/usr/bin/env python3
"""Build a self-hosted Font Awesome subset from used icon classes.

Scans templates and JavaScript, extracts Font Awesome icon names, subsets the
official webfonts, and writes a small CSS file. The full Font Awesome CDN
stylesheet is not required at runtime.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = (ROOT / "templates", ROOT / "static" / "js", ROOT / "static" / "css")
EXTRA_FILES = (ROOT / "static" / "offline.html",)
OUTPUT_DIR = ROOT / "static" / "vendor" / "icons"

FA_VERSION = "6.5.2"
FA_CSS_URL = f"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/{FA_VERSION}/css/all.min.css"
FA_FONT_URLS = {
    "solid": f"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/{FA_VERSION}/webfonts/fa-solid-900.woff2",
    "brands": f"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/{FA_VERSION}/webfonts/fa-brands-400.woff2",
    "regular": f"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/{FA_VERSION}/webfonts/fa-regular-400.woff2",
}
FA_LICENSE_URL = f"https://raw.githubusercontent.com/FortAwesome/Font-Awesome/{FA_VERSION}/LICENSE.txt"

UTILITY_NAMES = {
    "spin", "pulse", "fw", "lg", "xs", "sm", "2x", "3x", "4x", "5x",
    "stack", "inverse", "ul", "li", "border", "pull-left", "pull-right",
    "subset", "classic", "solid", "regular", "brands", "sharp",
}
STYLE_PREFIXES = {"fas", "far", "fab", "fal", "fad", "fass", "fa"}
BRAND_HINTS = {"github", "google"}
ICON_CLASS_RE = re.compile(r"""(?:class\s*=\s*["'][^"']*\b|(?:classList\.(?:add|toggle)\(|className\s*=\s*["'`])[^"'`]*\b)?(?:fa(?:s|r|b|l|d|ss)?\s+)fa-([a-z0-9-]+)""")
ICON_TOKEN_RE = re.compile(r"""\bfa-([a-z0-9-]+)\b""")
PREFIX_RE = re.compile(r"""\b(fas|far|fab|fa-solid|fa-regular|fa-brands)\b""")
CONTENT_RULE_RE = re.compile(r"""([^{}]+)\{content:"(\\[0-9a-fA-F]+)"\}""")
SELECTOR_ICON_RE = re.compile(r"""\.fa-([a-z0-9-]+):before""")


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCAN_DIRS:
        if not directory.exists():
            continue
        files.extend(
            path for path in directory.rglob("*")
            if path.suffix.lower() in {".html", ".js", ".css"}
            and "vendor/icons" not in path.as_posix()
        )
    for extra in EXTRA_FILES:
        if extra.is_file():
            files.append(extra)
    return files


def collect_used_icons() -> tuple[set[str], set[str]]:
    icons: set[str] = set()
    prefixes: set[str] = set()
    for path in iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        prefixes.update(PREFIX_RE.findall(text))
        for match in ICON_TOKEN_RE.finditer(text):
            name = match.group(1)
            if name in UTILITY_NAMES or name in STYLE_PREFIXES:
                continue
            icons.add(name)
    return icons, prefixes


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ai-playground-icon-subset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def parse_icon_codepoints(css_text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for selectors, code in CONTENT_RULE_RE.findall(css_text):
        for selector in selectors.split(","):
            match = SELECTOR_ICON_RE.search(selector)
            if match:
                mapping[match.group(1)] = code
    return mapping


def subset_font(pyftsubset: str, source: Path, dest: Path, unicodes: list[str]) -> None:
    if not unicodes:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        pyftsubset,
        str(source),
        f"--unicodes={','.join(unicodes)}",
        "--flavor=woff2",
        f"--output-file={dest}",
        "--canonical-order",
        "--name-IDs=*",
        "--name-legacy",
        "--layout-features=*",
    ]
    subprocess.run(command, check=True)


def write_subset_css(dest: Path, used_map: dict[str, str], families: dict[str, Path]) -> None:
    lines = [
        "/* Font Awesome Free 6.5.2 subset for AI Chat Playground.",
        " * Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT.",
        " * https://fontawesome.com/license/free",
        " */",
        ':root{--fa-style-family:"Font Awesome 6 Free";--fa-style:900}',
        '.fa,.fas,.far,.fab,.fa-solid,.fa-regular,.fa-brands{-moz-osx-font-smoothing:grayscale;-webkit-font-smoothing:antialiased;display:inline-block;font-style:normal;font-variant:normal;line-height:1;text-rendering:auto}',
        '.fa,.fas,.fa-solid{font-family:"Font Awesome 6 Free";font-weight:900}',
        '.far,.fa-regular{font-family:"Font Awesome 6 Free";font-weight:400}',
        '.fab,.fa-brands{font-family:"Font Awesome 6 Brands";font-weight:400}',
        "@keyframes fa-spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}",
        ".fa-spin{animation:fa-spin 2s linear infinite}",
        ".fa-fw{text-align:center;width:1.25em}",
        ".fa-xs{font-size:.75em}",
        ".fa-sm{font-size:.875em}",
        ".fa-lg{font-size:1.25em;line-height:.05em;vertical-align:-.075em}",
    ]
    if "solid" in families:
        lines.append(
            '@font-face{font-family:"Font Awesome 6 Free";font-style:normal;font-weight:900;'
            "font-display:swap;src:url(fa-solid-subset.woff2) format(\"woff2\")}"
        )
    if "regular" in families:
        lines.append(
            '@font-face{font-family:"Font Awesome 6 Free";font-style:normal;font-weight:400;'
            "font-display:swap;src:url(fa-regular-subset.woff2) format(\"woff2\")}"
        )
    if "brands" in families:
        lines.append(
            '@font-face{font-family:"Font Awesome 6 Brands";font-style:normal;font-weight:400;'
            "font-display:swap;src:url(fa-brands-subset.woff2) format(\"woff2\")}"
        )
    for name, code in sorted(used_map.items()):
        lines.append(f'.fa-{name}:before{{content:"{code}"}}')
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_pyftsubset() -> str:
    candidates = [
        ROOT / "venv" / "bin" / "pyftsubset",
        Path(shutil.which("pyftsubset") or ""),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("pyftsubset is required (fonttools). Install project requirements first.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/fa-src"))
    args = parser.parse_args()
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    used_icons, prefixes = collect_used_icons()
    if not used_icons:
        raise SystemExit("No Font Awesome icons found in templates or JavaScript.")

    css_path = cache_dir / "all.min.css"
    if not css_path.is_file():
        download(FA_CSS_URL, css_path)
    mapping = parse_icon_codepoints(css_path.read_text(encoding="utf-8"))
    missing = sorted(name for name in used_icons if name not in mapping)
    if missing:
        raise SystemExit(f"Unknown Font Awesome icon names: {', '.join(missing)}")

    used_map = {name: mapping[name] for name in sorted(used_icons)}
    unicodes = sorted({f"U+{code[2:].upper()}" for code in used_map.values()})

    uses_brands = bool({"fab", "fa-brands"} & prefixes) or bool(used_icons & BRAND_HINTS)
    uses_regular = bool({"far", "fa-regular"} & prefixes)
    families = {"solid": cache_dir / "fa-solid-900.woff2"}
    if uses_brands:
        families["brands"] = cache_dir / "fa-brands-400.woff2"
    if uses_regular:
        families["regular"] = cache_dir / "fa-regular-400.woff2"

    for family, dest in families.items():
        if not dest.is_file():
            download(FA_FONT_URLS[family], dest)

    pyftsubset = resolve_pyftsubset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fa-subset-") as tmp:
        tmp_dir = Path(tmp)
        built: dict[str, Path] = {}
        if "solid" in families:
            built["solid"] = OUTPUT_DIR / "fa-solid-subset.woff2"
            subset_font(pyftsubset, families["solid"], tmp_dir / "solid.woff2", unicodes)
            shutil.copyfile(tmp_dir / "solid.woff2", built["solid"])
        if "brands" in families:
            built["brands"] = OUTPUT_DIR / "fa-brands-subset.woff2"
            subset_font(pyftsubset, families["brands"], tmp_dir / "brands.woff2", unicodes)
            shutil.copyfile(tmp_dir / "brands.woff2", built["brands"])
        if "regular" in families:
            built["regular"] = OUTPUT_DIR / "fa-regular-subset.woff2"
            subset_font(pyftsubset, families["regular"], tmp_dir / "regular.woff2", unicodes)
            shutil.copyfile(tmp_dir / "regular.woff2", built["regular"])

    write_subset_css(OUTPUT_DIR / "fa-subset.css", used_map, built)
    license_path = cache_dir / "LICENSE.txt"
    if not license_path.is_file():
        download(FA_LICENSE_URL, license_path)
    shutil.copyfile(license_path, OUTPUT_DIR / "LICENSE.txt")

    css_size = (OUTPUT_DIR / "fa-subset.css").stat().st_size
    font_size = sum(path.stat().st_size for path in built.values())
    print(
        f"Wrote {len(used_map)} icons to {OUTPUT_DIR} "
        f"(css={css_size}B fonts={font_size}B families={','.join(built)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
