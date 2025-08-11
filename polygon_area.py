#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Union

Number = Union[int, float]
Point = Tuple[Number, Number]


def compute_polygon_area(points: List[Point]) -> float:
    """Compute polygon area using the shoelace formula.

    The polygon does not need to be explicitly closed; the function
    will treat the sequence as a ring by wrapping the last to the first.
    """
    if len(points) < 3:
        raise ValueError("A polygon needs at least 3 points")

    area2 = 0.0  # Twice the area
    num_points = len(points)
    for i in range(num_points):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % num_points]
        area2 += (x1 * y2) - (x2 * y1)
    return abs(area2) * 0.5


def parse_points(obj: object) -> Union[List[Point], None]:
    """Try to extract a polygon list of (x, y) from various JSON shapes.

    Supported inputs:
    - A list of objects: [{"x": num, "y": num}, ...]
    - An object with key "floor_polygon": same as above
    - An object with key "spaces": will return None here; handled separately
    """
    # Top-level list of points
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        if all("x" in p and "y" in p for p in obj):
            return [(float(p["x"]), float(p["y"])) for p in obj]

    # Object containing a floor_polygon
    if isinstance(obj, dict) and "floor_polygon" in obj:
        fp = obj["floor_polygon"]
        if isinstance(fp, list) and fp and isinstance(fp[0], dict):
            if all("x" in p and "y" in p for p in fp):
                return [(float(p["x"]), float(p["y"])) for p in fp]

    # Not directly a single polygon
    return None


def find_space_polygons(obj: object) -> List[Tuple[str, List[Point]]]:
    """If the input is a dataset with spaces[*].floor_polygon, collect them.

    Returns list of (label, points), where label is the space id or index.
    """
    results: List[Tuple[str, List[Point]]] = []
    if not isinstance(obj, dict):
        return results
    spaces = obj.get("spaces")
    if not isinstance(spaces, list):
        return results
    for idx, space in enumerate(spaces):
        if not isinstance(space, dict):
            continue
        pts = parse_points(space)
        if pts:
            label = str(space.get("id", f"space[{idx}]"))
            results.append((label, pts))
    return results


def load_json_from_source(path: str) -> object:
    if path == "-":
        data = sys.stdin.read()
        return json.loads(data)
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the area of a polygon provided as JSON.\n"
        "Input can be: an array of {x,y}, an object with floor_polygon, or a dataset with spaces[]."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="-",
        help="JSON source file path, or '-' to read from stdin (default).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places to show in the printed area.",
    )
    args = parser.parse_args()

    try:
        obj = load_json_from_source(args.source)
    except json.JSONDecodeError as e:
        print(f"[FATAL] Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Failed to load input: {e}")
        sys.exit(1)

    # Case 1: single polygon
    pts = parse_points(obj)
    if pts:
        area = compute_polygon_area(pts)
        print(f"area: {area:.{args.precision}f}")
        return

    # Case 2: dataset with spaces[*].floor_polygon
    space_polys = find_space_polygons(obj)
    if space_polys:
        for label, pts in space_polys:
            area = compute_polygon_area(pts)
            print(f"{label}: {area:.{args.precision}f}")
        return

    print("[ERROR] Could not find a polygon. Provide a list of {x,y} or an object with floor_polygon.")
    sys.exit(2)


if __name__ == "__main__":
    main() 