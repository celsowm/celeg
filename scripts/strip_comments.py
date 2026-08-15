#!/usr/bin/env python3
"""Strip prose C++ comments from the celeg source tree.

Removes // and /* */ comments via a character-level tokenizer that
understands string literals, char literals, raw string literals
(R"delim(...)delim") and the backslash-newline continuation gotcha for
// comments. Single-line block comments that only name a call-site
argument (e.g. /*add_bos=*/false, /*req*/) are preserved. Lines that
become comment-only are deleted entirely rather than left blank.

Usage:
    python scripts/strip_comments.py                 # dry-run report
    python scripts/strip_comments.py --apply          # write changes
    python scripts/strip_comments.py --selftest       # run tokenizer tests
    python scripts/strip_comments.py --paths a b c    # custom scope
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCOPE = ["src", "include", "tests", "benchmarks", "validation"]
EXTENSIONS = {".cpp", ".hpp", ".h", ".cu", ".cuh"}

PARAM_COMMENT_RE = re.compile(r"^/\*\s*[A-Za-z_][A-Za-z0-9_]*=?\s*\*/$")

RAW_STRING_START = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\n]{0,16})\(')
STRING_START = re.compile(r'(?:u8|u|U|L)?"')
CHAR_START = re.compile(r"(?:u8|u|U|L)?'")


def is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def scan_quoted(text: str, i: int, quote: str) -> int:
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return n


def scan_line_comment(text: str, i: int) -> int:
    n = len(text)
    i += 2
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] in "\r\n":
            i += 1
            if text[i] == "\r":
                i += 1
            if i < n and text[i] == "\n":
                i += 1
            continue
        if c in "\r\n":
            return i
        i += 1
    return n


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split text into ('code' | 'line_comment' | 'block_comment', text) runs."""
    tokens: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    def flush() -> None:
        if buf:
            tokens.append(("code", "".join(buf)))
            buf.clear()

    while i < n:
        c = text[i]
        boundary = i == 0 or not is_ident_char(text[i - 1])

        if boundary:
            m = RAW_STRING_START.match(text, i)
            if m:
                delim = m.group(1)
                close_seq = ")" + delim + '"'
                close = text.find(close_seq, m.end())
                end = close + len(close_seq) if close != -1 else n
                buf.append(text[i:end])
                i = end
                continue
            m = STRING_START.match(text, i)
            if m:
                end = scan_quoted(text, m.end(), '"')
                buf.append(text[i:end])
                i = end
                continue
            m = CHAR_START.match(text, i)
            if m:
                end = scan_quoted(text, m.end(), "'")
                buf.append(text[i:end])
                i = end
                continue

        if text[i : i + 2] == "//":
            flush()
            end = scan_line_comment(text, i)
            tokens.append(("line_comment", text[i:end]))
            i = end
            continue

        if text[i : i + 2] == "/*":
            flush()
            close = text.find("*/", i + 2)
            end = close + 2 if close != -1 else n
            tokens.append(("block_comment", text[i:end]))
            i = end
            continue

        buf.append(c)
        i += 1

    flush()
    return tokens


def render(tokens: list[tuple[str, str]]) -> tuple[str, int, int]:
    """Reassemble tokens into stripped text.

    Returns (text, comment_lines_removed, param_comments_preserved).
    """
    lines: list[str] = []
    cur: list[str] = []
    has_real_code = False
    had_removed = False
    removed_count = 0
    preserved_count = 0

    def end_line() -> None:
        nonlocal cur, has_real_code, had_removed, removed_count
        text = "".join(cur)
        if has_real_code:
            lines.append(text.rstrip() if had_removed else text)
        elif had_removed:
            removed_count += 1
        else:
            lines.append(text)
        cur = []
        has_real_code = False
        had_removed = False

    for kind, txt in tokens:
        if kind == "code":
            parts = txt.split("\n")
            for idx, part in enumerate(parts):
                cur.append(part)
                if part.strip():
                    has_real_code = True
                if idx < len(parts) - 1:
                    end_line()
        elif kind == "line_comment":
            had_removed = True
        elif kind == "block_comment":
            if "\n" not in txt and PARAM_COMMENT_RE.match(txt):
                cur.append(txt)
                has_real_code = True
                preserved_count += 1
            elif "\n" in txt:
                had_removed = True
                end_line()
                had_removed = True
            else:
                had_removed = True

    end_line()
    return "\n".join(lines), removed_count, preserved_count


def strip_comments(text: str) -> tuple[str, int, int]:
    return render(tokenize(text))


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

SELFTEST_CASES = [
    (
        "string literal containing //",
        'if (two == "==" || two == "//") return two;\n',
        'if (two == "==" || two == "//") return two;\n',
    ),
    (
        "string literal containing http://",
        'std::cout << "celeg-serve listening on http://" << args.host;\n',
        'std::cout << "celeg-serve listening on http://" << args.host;\n',
    ),
    (
        "raw string literal with embedded /* // \" {",
        'auto s = R"json({"a": "//not a comment", "b": "/* also not */"})json";\n',
        'auto s = R"json({"a": "//not a comment", "b": "/* also not */"})json";\n',
    ),
    (
        "preprocessor line with trailing comment",
        "#else // !_WIN32\n",
        "#else\n",
    ),
    (
        "backslash-continued line comment spans next line",
        "foo();  // comment part 1 \\\nstill comment\nbar();\n",
        "foo();\nbar();\n",
    ),
    (
        "char literals containing / and *",
        "char a = '/';\nchar b = '*';\n",
        "char a = '/';\nchar b = '*';\n",
    ),
    (
        "param-name block comments preserved",
        "foo(/*add_bos=*/false, /*req*/ nullptr);\n",
        "foo(/*add_bos=*/false, /*req*/ nullptr);\n",
    ),
    (
        "multi-line // prose deleted entirely",
        "// first line of explanation\n// second line of explanation\nint x = 1;\n",
        "int x = 1;\n",
    ),
    (
        "multi-line block comment prose deleted entirely, no blank remnant",
        "int a = 1;\n/* This is a\n   long prose\n   explanation */\nint b = 2;\n",
        "int a = 1;\nint b = 2;\n",
    ),
    (
        "multi-line block comment with code on both boundary lines",
        "foo();  /* explanation\n   of what foo does\n   in detail */  bar();\n",
        "foo();\n  bar();\n",
    ),
    (
        "indented single-line block comment alone on its line",
        "    /* todo: fix this later */\nint y = 2;\n",
        "int y = 2;\n",
    ),
    (
        "trailing single-line block comment",
        "int z = 3; /* explain z */\n",
        "int z = 3;\n",
    ),
    (
        "digit separator single-quote is not a char literal",
        "auto n = 1'000'000;\n",
        "auto n = 1'000'000;\n",
    ),
]


def run_selftest() -> bool:
    ok = True
    for name, input_text, expected in SELFTEST_CASES:
        actual, _, _ = strip_comments(input_text)
        if actual == expected:
            print(f"PASS  {name}")
        else:
            ok = False
            print(f"FAIL  {name}")
            print(f"  expected: {expected!r}")
            print(f"  actual:   {actual!r}")
    return ok


# --------------------------------------------------------------------------
# File processing
# --------------------------------------------------------------------------

BOM = b"\xef\xbb\xbf"


def process_file(path: Path, apply: bool) -> tuple[int, int, bool]:
    raw = path.read_bytes()
    has_bom = raw.startswith(BOM)
    text = raw[len(BOM) :].decode("utf-8") if has_bom else raw.decode("utf-8")

    new_text, removed, preserved = strip_comments(text)
    changed = new_text != text

    if apply and changed:
        out = (BOM if has_bom else b"") + new_text.encode("utf-8")
        path.write_bytes(out)

    return removed, preserved, changed


def iter_source_files(paths: list[Path]) -> list[Path]:
    files = []
    for base in paths:
        if base.is_file():
            if base.suffix in EXTENSIONS:
                files.append(base)
            continue
        for ext in EXTENSIONS:
            files.extend(base.rglob(f"*{ext}"))
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes to disk (default is dry-run)")
    parser.add_argument("--selftest", action="store_true", help="run tokenizer self-tests and exit")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_SCOPE,
        help=f"directories/files to process (default: {' '.join(DEFAULT_SCOPE)})",
    )
    args = parser.parse_args()

    if args.selftest:
        return 0 if run_selftest() else 1

    if not run_selftest():
        print("\nSelf-test failed; aborting without touching any files.", file=sys.stderr)
        return 1

    scope_paths = [(p if Path(p).is_absolute() else REPO_ROOT / p) for p in args.paths]
    files = iter_source_files(scope_paths)
    if not files:
        print("No source files found in scope.", file=sys.stderr)
        return 1

    total_removed = 0
    total_preserved = 0
    total_changed_files = 0

    for path in files:
        removed, preserved, changed = process_file(path, apply=args.apply)
        total_removed += removed
        total_preserved += preserved
        if changed:
            total_changed_files += 1
            rel = path.relative_to(REPO_ROOT)
            print(f"{'apply' if args.apply else 'dry-run'}  {rel}  -{removed} lines, {preserved} preserved")

    mode = "Applied" if args.apply else "Would change (dry-run)"
    print(
        f"\n{mode}: {total_changed_files}/{len(files)} files, "
        f"{total_removed} comment lines removed, {total_preserved} param comments preserved."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
