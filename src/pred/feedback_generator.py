import os
from dotenv import load_dotenv
from shapely.geometry import Polygon
from src.utils.json_check.verify import is_valid_json_feedback

class FeedbackGenerator:
    @staticmethod
    def analyze(output_floor_plan, input_prompt, tol=1e-6, area_tol=30):
        rooms = output_floor_plan.get("rooms", [])
        polygons = {}

        for idx, room in enumerate(rooms):
            try:
                room_id = room.get("id")
                poly_points = room.get("floor_polygon", [])
                if not poly_points or not room_id:
                    continue

                points = [(float(pt["x"]), float(pt["z"])) for pt in poly_points]
                poly = Polygon(points)

                if not poly.is_valid:
                    poly = poly.buffer(0)

                if poly.is_valid and poly.area > tol:
                    if room_id not in polygons:
                        polygons[room_id] = poly
                    else:
                        room_id = f"{room_id}_{idx}"
                        polygons[room_id] = poly

            except Exception:
                continue

        overlap_locations = []
        total_overlap_area = 0.0

        room_ids = list(polygons.keys())
        for i in range(len(room_ids)):
            for j in range(i + 1, len(room_ids)):
                id1 = room_ids[i]
                id2 = room_ids[j]
                poly1 = polygons[id1]
                poly2 = polygons[id2]

                if poly1.intersects(poly2):
                    intersection = poly1.intersection(poly2)
                    if not intersection.is_empty and intersection.area > tol:
                        area = intersection.area
                        total_overlap_area += area
                        overlap_locations.append({
                            "room1": id1,
                            "room2": id2,
                            "overlap_area": round(area, 2)
                        })

        is_overlapping = total_overlap_area > tol
        total_floor_area = sum(poly.area for poly in polygons.values() if poly.is_valid)
        overlap_percentage = (
            (total_overlap_area / total_floor_area * 100)
            if (total_floor_area > tol) else 0
        )

        expected_room_count = input_prompt.get("room_count")
        expected_room_types = input_prompt.get("room_types")
        expected_total_area = input_prompt.get("total_area")

        actual_room_count = len(polygons)
        actual_room_types = []
        for room in output_floor_plan.get("rooms", []):
            if not isinstance(room, dict):
                continue
            rt = room.get("room_type")
            if rt is not None:
                actual_room_types.append(rt)
        actual_total_area = total_floor_area

        room_count_match = (expected_room_count == actual_room_count) if expected_room_count is not None else None
        room_types_match = (set(expected_room_types) == set(actual_room_types)) if expected_room_types is not None else None
        total_area_match = (abs(expected_total_area - actual_total_area) <= area_tol) if (expected_total_area is not None and isinstance(expected_total_area, (int, float))) else None

        is_valid, feedback = is_valid_json_feedback(output_floor_plan)
        return {
            "is_overlapping": is_overlapping,
            "total_overlap_area": round(total_overlap_area, 2),
            "overlap_percentage": round(overlap_percentage, 2),
            "overlap_locations": overlap_locations,
            "is_valid_json": is_valid,
            "is_valid_json_feedback": feedback,
            "room_count": {"expected": expected_room_count, "actual": actual_room_count, "match": room_count_match},
            "room_types": {"expected": expected_total_area, "actual": actual_room_types, "match": room_types_match},
            "total_area": {"expected": expected_total_area, "actual": round(actual_total_area, 2), "tolerance": area_tol, "match": total_area_match},
        }

    @staticmethod
    def create_feedback(overlap_metrics):
        feedback = ""

        if not overlap_metrics.get("is_valid_json"):
            feedback += "Invalid JSON. "
            feedback += overlap_metrics.get("is_valid_json_feedback")
        else:
            if overlap_metrics["room_count"]["match"] is False:
                feedback += f"Expected room count {overlap_metrics['room_count']['expected']}, but got {overlap_metrics['room_count']['actual']}. "
            if overlap_metrics["room_types"]["match"] is False:
                feedback += f"Expected room types {overlap_metrics['room_types']['expected']}, but got {overlap_metrics['room_types']['actual']}. "
            if overlap_metrics["total_area"]["match"] is False:
                feedback += f"Expected total area {overlap_metrics['total_area']['expected']}, but got {overlap_metrics['total_area']['actual']:.2f}. "

        if overlap_metrics.get("is_overlapping", False):
            feedback += (
                "The generated floor plan contains overlapping regions. "
                f"Total overlapping area is {overlap_metrics['total_overlap_area']:.2f} square units, "
                f"which represents {overlap_metrics['overlap_percentage']:.2f}% of the total floor area. "
            )

            unique_pairs = {}
            for loc in overlap_metrics.get("overlap_locations", []):
                pair = tuple(sorted([loc['room1'], loc['room2']]))
                unique_pairs[pair] = unique_pairs.get(pair, 0.0) + loc['overlap_area']

            if unique_pairs:
                feedback += "\nThe following overlaps have been detected:\n"
                for pair, area in unique_pairs.items():
                    feedback += (
                        f"  - Rooms {pair[0]} and {pair[1]} overlap by "
                        f"{area:.2f} square units.\n"
                    )
                feedback += "Please revise the floor plan to remove these overlaps. \n"

        return feedback