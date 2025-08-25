#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "results8_GRPO_70B"
OUT_DIR = Path(__file__).resolve().parent
LIST_TXT = OUT_DIR / "list.txt"
MAP_JSON = OUT_DIR / "folder_to_rplan.json"


def collect(results_dir: Path) -> tuple[list[str], dict[str, str]]:
    rplan_ids: list[str] = []
    folder_to_rplan: dict[str, str] = {}

    if not results_dir.exists():
        return rplan_ids, folder_to_rplan

    for entry in sorted(results_dir.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if not entry.is_dir():
            continue
        sample_path = entry / "analysis" / "sample.json"
        if not sample_path.exists():
            continue
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rplan_id = data.get("rplan_id")
            if rplan_id:
                rid = str(rplan_id)
                rplan_ids.append(rid)
                folder_to_rplan[entry.name] = rid
        except Exception:
            continue

    return rplan_ids, folder_to_rplan


def main() -> None:
    rplan_ids, folder_to_rplan = collect(RESULTS_DIR)

    # Write list.txt
    lines = [f"{rid}.json" for rid in rplan_ids]
    with open(LIST_TXT, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + ("\n" if lines else ""))

    # Write mapping JSON
    with open(MAP_JSON, "w", encoding="utf-8") as out:
        json.dump(folder_to_rplan, out, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Wrote {len(lines)} ids to {LIST_TXT}")
    print(f"Wrote {len(folder_to_rplan)} mappings to {MAP_JSON}")


if __name__ == "__main__":
    main() 