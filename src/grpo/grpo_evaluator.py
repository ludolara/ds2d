from shapely.geometry import Polygon
from src.pred.extract_output_json import extract_output_json
from src.dataset_convert.rplan_graph import RPLANGraph
# from src.utils.json_check.verify import is_valid_json, is_valid_json_feedback
from src.utils.json_check.verify import is_valid_json
import json

class GRPOEvaluator:
    @staticmethod
    def evaluate(output_floor_plan, input_prompt, round_digits: int = 4):
        try:
            output_floor_plan = extract_output_json(output_floor_plan)
            input_graph_json = json.loads(input_prompt.get("input_graph", "{}"))
            
            polygons_overlap = {}
            polygons_area = {} # (excluding doors)
            
            for idx, room in enumerate(output_floor_plan.get("spaces", [])):
                pts = room.get("floor_polygon", [])
                room_type = room.get("room_type", "unknown")
                try:
                    coords = [(float(p["x"]), float(p["y"])) for p in pts]
                    poly = Polygon(coords)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and poly.area:
                        room_id = str(room.get("id", idx))
                        
                        key_overlap = room_id if room_id not in polygons_overlap else f"{room_id}_{idx}"
                        polygons_overlap[key_overlap] = poly
                        
                        # if room_type not in ["interior_door", "front_door"]:
                        if room_type not in ["interior_door"]:
                            key_area = room_id if room_id not in polygons_area else f"{room_id}_{idx}"
                            polygons_area[key_area] = poly
                except Exception:
                    continue

            total_overlap = 0.0
            ids_overlap = list(polygons_overlap)
            for i in range(len(ids_overlap)):
                for j in range(i + 1, len(ids_overlap)):
                    inter = polygons_overlap[ids_overlap[i]].intersection(polygons_overlap[ids_overlap[j]])
                    total_overlap += inter.area

            total_area = sum(poly.area for poly in polygons_area.values() if poly.is_valid)
            overlap_ratio = total_overlap / total_area if total_area > 0 else 0
            is_overlap = (round(overlap_ratio, round_digits-1) != 0)

            # is_valid, feedback = is_valid_json_feedback(output_floor_plan)
            is_valid = is_valid_json(output_floor_plan)

            # expected_room_count = input_prompt.get("room_count", 0)
            expected_total_area = input_prompt.get("total_area", 0)

            # if expected_room_count > 0:
            #     actual_room_count = len(polygons_area)
            #     room_count_match = (actual_room_count == expected_room_count)
            # else:
            #     room_count_match = False
            
            if expected_total_area > 0:
                total_area_ratio = total_area / expected_total_area
            else:
                total_area_ratio = 0

            input_graph = RPLANGraph.from_labeled_adjacency(input_graph_json)
            output_graph = RPLANGraph.from_ds2d(output_floor_plan)
            compatibility_score = output_graph.compatibility_score_scaled(input_graph)

#             print(f"""
# {'='*60}
# GRPO EVALUATION DEBUG
# JSON Feedback: {feedback}

# Input Prompt:
# {input_prompt}

# Output Floor Plan:
# {output_floor_plan}

# Metrics:
# - JSON validity: {is_valid}
# - Total area: {total_area_ratio}
# - Is overlap: {is_overlap}
# - Compatibility score: {compatibility_score}
# {'='*60}
# """)

            return {
                "is_valid_json": is_valid,
                # "room_count": room_count_match,
                "total_area": round(total_area_ratio, round_digits),
                "is_overlap": is_overlap,
                "compatibility": round(compatibility_score, round_digits)
            }
        except Exception as e:
            # print(f"Error in evaluate: {e}")
            # print(traceback.format_exc())
            return { "is_valid_json": False }
