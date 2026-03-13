import argparse
import re
import sys
from pathlib import Path

SINGLE_BACKTICK_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
ROLE_SUFFIX_RE = re.compile(r":[A-Za-z0-9_.:-]+:$")
CODE_BLOCK_DIRECTIVE_RE = re.compile(
    r"^\.\.\s+(code-block|code|parsed-literal)::"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="check invalid single-backtick usage in rst files"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="rst files to check",
    )
    return parser.parse_args()


def should_ignore_match(line, start, end):
    if end < len(line) and line[end] == "_":
        return True

    prefix = line[:start].rstrip()
    if ROLE_SUFFIX_RE.search(prefix):
        return True

    return False


def find_invalid_single_backticks(line):
    matches = []
    for match in SINGLE_BACKTICK_RE.finditer(line):
        start, end = match.span()
        if should_ignore_match(line, start, end):
            continue
        matches.append(match)
    return matches


def get_indent_width(line):
    return len(line) - len(line.lstrip(" "))


def starts_literal_block(line):
    stripped = line.strip()
    if not stripped:
        return False
    if CODE_BLOCK_DIRECTIVE_RE.match(stripped):
        return True
    if stripped == "::":
        return True
    if stripped.endswith("::") and not stripped.startswith(".. "):
        return True
    return False


def check_file(path):
    has_error = False
    lines = path.read_text(encoding="utf-8").splitlines()
    pending_literal_block = False
    pending_indent = 0
    in_literal_block = False
    block_indent = 0

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = get_indent_width(line)

        if in_literal_block:
            if not stripped:
                continue
            if indent >= block_indent:
                continue
            in_literal_block = False

        if pending_literal_block:
            if not stripped:
                continue
            if indent > pending_indent and stripped.startswith(":"):
                continue
            if indent > pending_indent:
                in_literal_block = True
                # Use the first content line indent as block baseline.
                block_indent = indent
                continue
            pending_literal_block = False

        if starts_literal_block(line):
            pending_literal_block = True
            pending_indent = indent
            continue

        matches = find_invalid_single_backticks(line)
        for match in matches:
            has_error = True
            column = match.start() + 1
            snippet = match.group(0)
            print(
                f"{path}:{lineno}:{column}: invalid single backticks {snippet}; use ``...`` for inline literals or an rst role/link when appropriate."
            )
    return has_error


def main():
    args = parse_args()
    has_error = False

    for file_name in args.files:
        path = Path(file_name)
        if path.suffix != ".rst" or not path.exists():
            continue
        has_error |= check_file(path)

    if has_error:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
