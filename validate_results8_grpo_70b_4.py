#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

# Ensure we can import from the local src package when running from repo root
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SRC_DIR))

# Try to import validator and handle missing dependency gracefully
try:
    from src.utils.json_check.verify import is_valid_json_feedback  # type: ignore
except Exception as e:
    print("[FATAL] Failed to import validator from src.utils.json_check.verify.\n"
          "Make sure you're running this from the repo root and dependencies are installed.\n"
          f"Import error: {e}")
    sys.exit(1)

# jsonschema may not be listed in requirements.txt; detect ImportError earlier
try:
    import jsonschema  # noqa: F401
except Exception:
    print("[FATAL] Missing dependency 'jsonschema'. Install with: pip install jsonschema")
    sys.exit(1)


def load_json(path: Path) -> Tuple[bool, Dict, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return True, data, ""
    except json.JSONDecodeError as e:
        return False, {}, f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}"
    except Exception as e:
        return False, {}, f"ReadError: {e}"


def find_candidate_files(base_dir: Path):
    # Heuristic: model outputs live in leaf folders as '0.json'. Avoid 'prompt.json' etc.
    for path in base_dir.rglob("0.json"):
        # Only include files under generations subtree if base_dir is results folder
        yield path


def main():
    parser = argparse.ArgumentParser(description="Validate generated JSONs and report schema failures.")
    parser.add_argument(
        "--base",
        type=str,
        default=str(ROOT / "results8_GRPO_70B_4"),
        help="Base directory to scan (defaults to results8_GRPO_70B_4 in repo root)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print passing files as well."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of files to check (0 = no limit)."
    )

    args = parser.parse_args()

    base_dir = Path(args.base).resolve()
    if not base_dir.exists():
        print(f"[FATAL] Base directory does not exist: {base_dir}")
        sys.exit(1)

    # If user points at the results folder, prefer the 'generations' subtree
    generations_dir = base_dir / "generations"
    scan_dir = generations_dir if generations_dir.exists() else base_dir

    print(f"Scanning: {scan_dir}")

    total = 0
    passed = 0
    failed = 0

    failure_reasons = Counter()
    failures_by_file = []

    for i, json_path in enumerate(find_candidate_files(scan_dir), start=1):
        if args.limit and total >= args.limit:
            break
        total += 1

        ok, data, load_err = load_json(json_path)
        if not ok:
            failed += 1
            reason = load_err
            failure_reasons[reason] += 1
            failures_by_file.append((str(json_path), reason))
            print(f"[FAIL] {json_path}: {reason}")
            continue

        is_valid, error_text = is_valid_json_feedback(data)
        if is_valid:
            passed += 1
            if args.verbose:
                print(f"[OK]   {json_path}")
        else:
            failed += 1
            reason = error_text.strip() or "SchemaValidationError (no details)"
            failure_reasons[reason] += 1
            failures_by_file.append((str(json_path), reason))
            print(f"[FAIL] {json_path}: {reason}")

    print("\nSummary:")
    print(f"  Total checked: {total}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")

    if failed:
        print("\nTop failure reasons:")
        for reason, count in failure_reasons.most_common(10):
            print(f"  {count:4d}  {reason}")

        # Optionally, print first few failing files for each reason
        grouped = defaultdict(list)
        for path_str, reason in failures_by_file:
            grouped[reason].append(path_str)

        print("\nExamples:")
        for reason, paths in list(grouped.items())[:5]:
            print(f"- {reason}")
            for p in paths[:3]:
                print(f"    {p}")

    # Exit with non-zero code if any failures were found
    sys.exit(0 if failed == 0 else 2)


if __name__ == "__main__":
    main() 