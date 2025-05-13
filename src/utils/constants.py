RPLAN_ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining_room": 7, "study_room": 8,
    "storage": 10, "front_door": 15, "unknown": 16, "interior_door": 17
}

# SYSTEM_PROMPT = """
# You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
# Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
# Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, doors may touch room boundaries, but room interiors must never intersect.  

# Your output must be a JSON object, where `output` key contains:
# - `room_count`: the total number of room and door entries  
# - `rooms`: a list of mixing rooms and doors. Each room or door entry in `room` must include:
#  - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"interior_door|0"`)  
#  - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
#  - `area` in square meters (all positive numbers)  
#  - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  

# Additional rules:
# - **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
# - Every adjacency in the bubble diagram must be bridged by exactly one door.  
# - Every `id` used in the bubble diagram and on any door must appear in the `rooms` list.  

# Return only a JSON object containing an `output` key without extra commentary or explanation.
# """

SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Room interiors must be strictly disjoint.

Your output must be a JSON object, where the `output` key contains:
- `room_count`: the total number of room entries  
- `rooms`: a list of rooms. Each entry must include:
  - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"`)  
  - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
  - `area`: in square meters (all positive numbers)  
  - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every adjacency in the bubble diagram must be represented by rooms touching along shared edges.
- Every `id` used in the bubble diagram must appear in the `rooms` list.

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

SYSTEM_RE_PROMPT = """
You are a floor-plan post-processor whose one and only job is to fix overlaps—no other changes are allowed.
Maintain overall floorplan layout, but ensure that no two room polygons ever overlap.

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

OVERLAP_TOL = 1e-6