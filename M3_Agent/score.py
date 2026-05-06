"""Summarize GPT evaluation results stored in local JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count how many entries were marked correct/incorrect by gpt_eval.\n"
            "By default it looks for robot.jsonl in the same directory as this script."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=["data/results/robot.jsonl"],
        help="One or more JSONL files to summarize (relative paths are resolved against this script).",
    )
    return parser.parse_args()


def resolve_path(raw_path: str, script_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = script_dir / path
    if candidate.exists():
        return candidate
    if path.exists():
        return path
    return path


def iter_entries(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}#{line_no}: invalid JSON - {exc}") from exc


def normalize_gpt_eval(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "1", "yes"}:
            return True
        if lowered in {"false", "incorrect", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    raise ValueError(f"Unsupported gpt_eval value: {value!r}")


def summarize_file(path: Path) -> Tuple[Counter, int]:
    counts = Counter({True: 0, False: 0})
    total = 0
    for line_no, entry in iter_entries(path):
        if "gpt_eval" not in entry:
            raise KeyError(f"{path}#{line_no}: missing 'gpt_eval' field")
        verdict = normalize_gpt_eval(entry["gpt_eval"])
        counts[verdict] += 1
        total += 1
    return counts, total


def format_summary_line(label: str, counts: Counter, total: int) -> str:
    if total == 0:
        return f"  {label}: no entries."
    return f"  {label}: {counts[True]/total:.3f}"


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    exit_code = 0

    for raw_path in args.files:
        path = resolve_path(raw_path, script_dir)
        try:
            counts, total = summarize_file(path)
        except FileNotFoundError:
            print(f"{path}: file not found", file=sys.stderr)
            exit_code = 1
            continue
        except Exception as exc: 
            print(exc, file=sys.stderr)
            exit_code = 1
            continue
        print(f"{path.name}:")
        print(format_summary_line("Accuracy", counts, total))
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
