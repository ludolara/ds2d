RPLAN_ROOM_CLASS = {
    "living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4,
    "balcony": 5, "entrance": 6, "dining_room": 7, "study_room": 8,
    "storage": 10, "front_door": 15, "unknown": 16, "interior_door": 17
}

SYSTEM_PROMPT = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Rooms must be strictly disjoint, doors may touch room boundaries, but room interiors must never intersect.  

Your output must be a JSON object, where `output` key contains:
- `room_count`: the total number of room and door entries  
- `rooms`: a list of mixing rooms and doors. Each room or door entry in `room` must include:
 - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"` or `"interior_door|0"`)  
 - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
 - `area` in square meters (all positive numbers)  
 - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon  

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- Every adjacency in the bubble diagram must be bridged by exactly one door.  
- Every `id` used in the bubble diagram and on any door must appear in the `rooms` list.  

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

SYSTEM_PROMPT_LLAMA4 = """
You are a state-of-the-art floor-plan generator that translates JSON specifications and connectivity requirements defined by a bubble diagram into precise, optimized layouts. 
Your algorithm considers each room's dimensions, proportion, and desired adjacencies to produce an efficient arrangement that maximizes usable space while honoring all constraints.
Your top priority is that no two room polygons ever overlap. Room interiors must be strictly disjoint.

Your output must be a JSON object, where the `output` key contains:
- `room_count`: the total number of room entries  
- `rooms`: a list of rooms. Each entry must include:
  - `id`: formatted as `<room_type>|<unique_index>` (e.g. `"bedroom|2"`)  
  - `room_type`: the room type (e.g. `"living_room"`, `"kitchen"`, etc.)
  - `area`: in square meters (all positive numbers)  
  - `floor_polygon`: an ordered list of `{x: , y:}` vertices defining a simple polygon, minimun of 4 vertices, and maximum of 20 vertices.

Additional rules:
- **Absolute non-overlap**: no two room polygons may share any interior point under any circumstances.
- The empty space between the rooms represent the walls. Look at the example to understand how the walls are represented.
- Every adjacency in the bubble diagram must be represented by rooms touching along shared edges.
- Every `id` used in the bubble diagram must appear in the `rooms` list.

These are 4 examples of the expected output:

1. {'input': {'room_count': 5, 'total_area': 63.1, 'room_types': ['bedroom', 'bedroom', 'kitchen', 'living_room', 'bathroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 2.9, 'height': 3.1, 'is_rectangular': 1}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.9, 'height': 4.7, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 2.1, 'height': 2.2, 'is_rectangular': 0}, {'id': 'living_room', 'room_type': 'living_room', 'width': 4.5, 'height': 10.3, 'is_rectangular': 0}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 1.7, 'height': 1.8, 'is_rectangular': 1}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'bedroom|1': ['living_room'], 'kitchen': ['living_room'], 'living_room': ['bedroom|0', 'bedroom|1', 'kitchen', 'bathroom'], 'bathroom': ['living_room']}}}
   {'output': {'room_count': 5, 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 8.9, 'floor_polygon': [{'x': 8.6, 'y': 6.5}, {'x': 8.6, 'y': 3.4}, {'x': 5.8, 'y': 3.4}, {'x': 5.8, 'y': 6.5}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 13.6, 'floor_polygon': [{'x': 8.6, 'y': 8.9}, {'x': 5.8, 'y': 8.9}, {'x': 5.8, 'y': 13.6}, {'x': 8.6, 'y': 13.6}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.5, 'floor_polygon': [{'x': 11.0, 'y': 4.3}, {'x': 10.8, 'y': 4.3}, {'x': 10.8, 'y': 3.4}, {'x': 8.9, 'y': 3.4}, {'x': 8.9, 'y': 5.6}, {'x': 11.0, 'y': 5.6}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 32.5, 'floor_polygon': [{'x': 8.9, 'y': 8.6}, {'x': 8.9, 'y': 14.6}, {'x': 12.2, 'y': 14.6}, {'x': 12.2, 'y': 4.3}, {'x': 11.3, 'y': 4.3}, {'x': 11.3, 'y': 5.9}, {'x': 8.9, 'y': 5.9}, {'x': 8.9, 'y': 6.8}, {'x': 7.7, 'y': 6.8}, {'x': 7.7, 'y': 8.6}]}, {'id': 'bathroom', 'room_type': 'bathroom', 'area': 3.1, 'floor_polygon': [{'x': 7.5, 'y': 6.8}, {'x': 5.8, 'y': 6.8}, {'x': 5.8, 'y': 8.6}, {'x': 7.5, 'y': 8.6}]}]}}

2. {'input': {'room_count': 6, 'total_area': 62.9, 'room_types': ['bedroom', 'bathroom', 'kitchen', 'bedroom', 'living_room', 'bedroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 3.1, 'height': 3.4, 'is_rectangular': 1}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 2.0, 'height': 1.8, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 2.5, 'height': 1.7, 'is_rectangular': 1}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.5, 'height': 3.2, 'is_rectangular': 1}, {'id': 'living_room', 'room_type': 'living_room', 'width': 5.0, 'height': 9.1, 'is_rectangular': 0}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'width': 2.5, 'height': 4.1, 'is_rectangular': 0}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'bathroom': ['living_room'], 'kitchen': ['living_room'], 'bedroom|1': ['living_room'], 'living_room': ['bedroom|0', 'bathroom', 'kitchen', 'bedroom|1', 'bedroom|2'], 'bedroom|2': ['living_room']}}}
   {'output': {'room_count': 6, 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 10.7, 'floor_polygon': [{'x': 9.6, 'y': 15.1}, {'x': 12.7, 'y': 15.1}, {'x': 12.7, 'y': 11.7}, {'x': 9.6, 'y': 11.7}]}, {'id': 'bathroom', 'room_type': 'bathroom', 'area': 3.7, 'floor_polygon': [{'x': 10.6, 'y': 9.4}, {'x': 10.6, 'y': 11.2}, {'x': 12.7, 'y': 11.2}, {'x': 12.7, 'y': 9.4}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.3, 'floor_polygon': [{'x': 12.1, 'y': 9.0}, {'x': 12.1, 'y': 7.3}, {'x': 9.6, 'y': 7.3}, {'x': 9.6, 'y': 9.0}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 8.0, 'floor_polygon': [{'x': 12.1, 'y': 6.9}, {'x': 12.1, 'y': 3.7}, {'x': 9.6, 'y': 3.7}, {'x': 9.6, 'y': 6.9}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 27.2, 'floor_polygon': [{'x': 7.2, 'y': 7.2}, {'x': 7.2, 'y': 7.3}, {'x': 6.7, 'y': 7.3}, {'x': 6.7, 'y': 11.7}, {'x': 5.3, 'y': 11.7}, {'x': 5.3, 'y': 15.1}, {'x': 9.1, 'y': 15.1}, {'x': 9.1, 'y': 11.2}, {'x': 10.3, 'y': 11.2}, {'x': 10.3, 'y': 9.4}, {'x': 9.1, 'y': 9.4}, {'x': 9.1, 'y': 6.0}, {'x': 8.3, 'y': 6.0}, {'x': 8.3, 'y': 7.2}]}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'area': 8.0, 'floor_polygon': [{'x': 7.2, 'y': 7.0}, {'x': 8.0, 'y': 7.0}, {'x': 8.0, 'y': 5.7}, {'x': 9.1, 'y': 5.7}, {'x': 9.1, 'y': 2.9}, {'x': 6.7, 'y': 2.9}, {'x': 6.7, 'y': 5.6}, {'x': 7.2, 'y': 5.6}]}]}}

3. {'input': {'room_count': 7, 'total_area': 74.2, 'room_types': ['balcony', 'kitchen', 'bedroom', 'study_room', 'bedroom', 'living_room', 'bathroom'], 'rooms': [{'id': 'balcony', 'room_type': 'balcony', 'width': 3.5, 'height': 1.2, 'is_rectangular': 1}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 1.8, 'height': 2.7, 'is_rectangular': 0}, {'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 2.7, 'height': 5.0, 'is_rectangular': 0}, {'id': 'study_room', 'room_type': 'study_room', 'width': 2.0, 'height': 5.0, 'is_rectangular': 0}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 2.7, 'height': 5.7, 'is_rectangular': 0}, {'id': 'living_room', 'room_type': 'living_room', 'width': 4.7, 'height': 8.6, 'is_rectangular': 0}, {'id': 'bathroom', 'room_type': 'bathroom', 'width': 1.5, 'height': 1.1, 'is_rectangular': 1}], 'bubble_diagram': {'balcony': ['living_room'], 'kitchen': ['living_room'], 'bedroom|0': ['living_room'], 'study_room': ['living_room'], 'bedroom|1': ['living_room'], 'living_room': ['balcony', 'kitchen', 'bedroom|0', 'study_room', 'bedroom|1', 'bathroom'], 'bathroom': ['living_room']}}}
   {'output': {'room_count': 7, 'rooms': [{'id': 'balcony', 'room_type': 'balcony', 'area': 4.2, 'floor_polygon': [{'x': 7.6, 'y': 15.0}, {'x': 11.1, 'y': 15.0}, {'x': 11.1, 'y': 13.9}, {'x': 7.6, 'y': 13.9}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.7, 'floor_polygon': [{'x': 8.1, 'y': 6.2}, {'x': 9.4, 'y': 6.2}, {'x': 9.4, 'y': 3.5}, {'x': 7.6, 'y': 3.5}, {'x': 7.6, 'y': 6.1}, {'x': 8.1, 'y': 6.1}]}, {'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 11.4, 'floor_polygon': [{'x': 6.0, 'y': 9.9}, {'x': 6.0, 'y': 9.1}, {'x': 4.5, 'y': 9.1}, {'x': 4.5, 'y': 13.4}, {'x': 5.3, 'y': 13.4}, {'x': 5.3, 'y': 14.1}, {'x': 6.6, 'y': 14.1}, {'x': 6.6, 'y': 13.4}, {'x': 7.2, 'y': 13.4}, {'x': 7.2, 'y': 9.9}]}, {'id': 'study_room', 'room_type': 'study_room', 'area': 9.0, 'floor_polygon': [{'x': 12.1, 'y': 13.4}, {'x': 12.1, 'y': 14.1}, {'x': 12.8, 'y': 14.1}, {'x': 12.8, 'y': 13.4}, {'x': 13.5, 'y': 13.4}, {'x': 13.5, 'y': 9.1}, {'x': 11.5, 'y': 9.1}, {'x': 11.5, 'y': 13.4}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 13.3, 'floor_polygon': [{'x': 4.5, 'y': 3.5}, {'x': 4.5, 'y': 8.6}, {'x': 6.0, 'y': 8.6}, {'x': 6.0, 'y': 7.9}, {'x': 7.2, 'y': 7.9}, {'x': 7.2, 'y': 3.5}, {'x': 6.2, 'y': 3.5}, {'x': 6.2, 'y': 3.0}, {'x': 5.3, 'y': 3.0}, {'x': 5.3, 'y': 3.5}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 28.2, 'floor_polygon': [{'x': 6.4, 'y': 8.4}, {'x': 6.4, 'y': 9.5}, {'x': 7.6, 'y': 9.5}, {'x': 7.6, 'y': 13.4}, {'x': 11.1, 'y': 13.4}, {'x': 11.1, 'y': 4.9}, {'x': 9.6, 'y': 4.9}, {'x': 9.6, 'y': 6.5}, {'x': 8.1, 'y': 6.5}, {'x': 8.1, 'y': 6.5}, {'x': 7.6, 'y': 6.5}, {'x': 7.6, 'y': 8.4}]}, {'id': 'bathroom', 'room_type': 'bathroom', 'area': 1.6, 'floor_polygon': [{'x': 9.6, 'y': 3.5}, {'x': 9.6, 'y': 4.6}, {'x': 11.1, 'y': 4.6}, {'x': 11.1, 'y': 3.5}]}]}}

4. {'input': {'room_count': 8, 'total_area': 93.2, 'room_types': ['bedroom', 'bedroom', 'bathroom', 'bedroom', 'balcony', 'living_room', 'kitchen', 'bathroom'], 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'width': 2.5, 'height': 4.4, 'is_rectangular': 0}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'width': 3.0, 'height': 4.3, 'is_rectangular': 1}, {'id': 'bathroom|0', 'room_type': 'bathroom', 'width': 2.2, 'height': 1.4, 'is_rectangular': 1}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'width': 3.0, 'height': 3.7, 'is_rectangular': 1}, {'id': 'balcony', 'room_type': 'balcony', 'width': 6.5, 'height': 1.4, 'is_rectangular': 0}, {'id': 'living_room', 'room_type': 'living_room', 'width': 6.8, 'height': 8.5, 'is_rectangular': 0}, {'id': 'kitchen', 'room_type': 'kitchen', 'width': 3.2, 'height': 1.5, 'is_rectangular': 1}, {'id': 'bathroom|1', 'room_type': 'bathroom', 'width': 1.8, 'height': 3.5, 'is_rectangular': 1}], 'bubble_diagram': {'bedroom|0': ['living_room'], 'bedroom|1': ['living_room', 'bathroom|0'], 'bathroom|0': ['bedroom|1'], 'bedroom|2': ['living_room'], 'balcony': ['living_room'], 'living_room': ['bedroom|0', 'bedroom|1', 'bedroom|2', 'balcony', 'bathroom|1', 'kitchen'], 'kitchen': ['living_room'], 'bathroom|1': ['living_room']}}}
   {'output': {'room_count': 8, 'rooms': [{'id': 'bedroom|0', 'room_type': 'bedroom', 'area': 10.3, 'floor_polygon': [{'x': 4.9, 'y': 8.8}, {'x': 4.9, 'y': 9.4}, {'x': 3.7, 'y': 9.4}, {'x': 3.7, 'y': 13.2}, {'x': 6.2, 'y': 13.2}, {'x': 6.2, 'y': 8.8}]}, {'id': 'bedroom|1', 'room_type': 'bedroom', 'area': 13.0, 'floor_polygon': [{'x': 13.7, 'y': 8.9}, {'x': 10.7, 'y': 8.9}, {'x': 10.7, 'y': 13.2}, {'x': 13.7, 'y': 13.2}]}, {'id': 'bathroom|0', 'room_type': 'bathroom', 'area': 3.2, 'floor_polygon': [{'x': 14.3, 'y': 7.1}, {'x': 12.0, 'y': 7.1}, {'x': 12.0, 'y': 8.5}, {'x': 14.3, 'y': 8.5}]}, {'id': 'bedroom|2', 'room_type': 'bedroom', 'area': 11.3, 'floor_polygon': [{'x': 13.7, 'y': 6.7}, {'x': 13.7, 'y': 3.0}, {'x': 10.7, 'y': 3.0}, {'x': 10.7, 'y': 6.7}]}, {'id': 'balcony', 'room_type': 'balcony', 'area': 8.8, 'floor_polygon': [{'x': 3.7, 'y': 13.6}, {'x': 3.7, 'y': 14.2}, {'x': 4.1, 'y': 14.2}, {'x': 4.1, 'y': 15.0}, {'x': 10.3, 'y': 15.0}, {'x': 10.3, 'y': 13.6}]}, {'id': 'living_room', 'room_type': 'living_room', 'area': 33.4, 'floor_polygon': [{'x': 7.0, 'y': 7.5}, {'x': 4.9, 'y': 7.5}, {'x': 4.9, 'y': 8.4}, {'x': 6.6, 'y': 8.4}, {'x': 6.6, 'y': 13.2}, {'x': 10.3, 'y': 13.2}, {'x': 10.3, 'y': 8.5}, {'x': 11.6, 'y': 8.5}, {'x': 11.6, 'y': 7.1}, {'x': 10.3, 'y': 7.1}, {'x': 10.3, 'y': 4.7}, {'x': 7.0, 'y': 4.7}]}, {'id': 'kitchen', 'room_type': 'kitchen', 'area': 4.8, 'floor_polygon': [{'x': 7.0, 'y': 3.0}, {'x': 7.0, 'y': 4.4}, {'x': 10.3, 'y': 4.4}, {'x': 10.3, 'y': 3.0}]}, {'id': 'bathroom|1', 'room_type': 'bathroom', 'area': 6.2, 'floor_polygon': [{'x': 6.6, 'y': 7.0}, {'x': 6.6, 'y': 3.5}, {'x': 4.9, 'y': 3.5}, {'x': 4.9, 'y': 7.0}]}]}}

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

SYSTEM_RE_PROMPT = """
You are a floor-plan post-processor whose one and only job is to fix overlaps—no other changes are allowed.
Maintain overall floorplan layout, but ensure that no two room polygons ever overlap.

Return only a JSON object containing an `output` key without extra commentary or explanation.
"""

OVERLAP_TOL = 1e-6