from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PATTERNS = ["FAL_KEY", "Authorization", "Bearer", "token=", "key="]
DEFAULT_REGEXES = [
    r"fal-[A-Za-z0-9_\-]{12,}",
    r"hf_[A-Za-z0-9_]{12,}",
]
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    pattern: str


def _is_text_candidate(path: Path, max_bytes: int) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def scan_path(root: Path, patterns: list[str], regexes: list[str], max_bytes: int) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    compiled = [re.compile(pattern) for pattern in regexes]
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_text_candidate(path, max_bytes):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern in line:
                    findings.append(Finding(rel, line_number, "literal", pattern))
            for regex in compiled:
                if regex.search(line):
                    findings.append(Finding(rel, line_number, "regex", regex.pattern))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan generated outputs for secret-like strings without printing values.")
    parser.add_argument("--path", required=True, help="Output folder to scan.")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_PATTERNS, help="Literal strings to search for.")
    parser.add_argument("--regex", nargs="*", default=DEFAULT_REGEXES, help="Regex patterns to search for.")
    parser.add_argument("--max-bytes", type=int, default=2_000_000, help="Skip files larger than this size.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.path)
    if not root.exists():
        payload = {"status": "failed", "error": f"path not found: {root}", "findings": []}
        print(json.dumps(payload, indent=2))
        return 2
    findings = scan_path(root, args.patterns, args.regex, args.max_bytes)
    payload = {
        "status": "failed" if findings else "passed",
        "path": str(root),
        "findings": [finding.__dict__ for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif findings:
        print(f"Secret-like strings found under {root}:")
        for finding in findings:
            print(f"- {finding.path}:{finding.line} {finding.kind}:{finding.pattern}")
    else:
        print(f"No secret-like strings found under {root}.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
