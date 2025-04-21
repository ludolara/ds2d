def create_input(sample, is_str=True):
    inp = {
        "room_count": sample.get("room_count"),
        "total_area": sample.get("total_area"),
        "room_types": sample.get("room_types"),
        "rooms": [
            {
                "room_type": room.get("room_type"),
                "width": room.get("width"),
                "height": room.get("height"),
                "is_regular": room.get("is_regular"),
            }
            for room in sample.get("rooms", [])
        ],
        # "edges": sample.get("edges", []),
    }
    if is_str:
        return str({"input": inp})
    else:
        return inp
    
def create_output(sample):
    if isinstance(sample, dict) and "edges" in sample:
        sample.pop("edges")
    output = {"output": sample}
    return str(output)
    