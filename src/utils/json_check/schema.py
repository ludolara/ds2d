schema = {
  "type": "object",
  # "additionalProperties": False,
  "properties": {
    "room_count": {
      "type": "integer"
    },
    # "total_area": {
    #   "type": "number"
    # },
    # "room_types": {
    #   "type": "array",
    #   "items": {
    #     "type": "string"
    #   }
    # },
    "rooms": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "id": {
            # "type": ["string", "integer"]
            "type": "string",
          },
          "room_type": {
            "type": "string"
          },
          "area": {
            "type": "number"
          },
          # "width": {
          #   "type": "number"
          # },
          # "height": {
          #   "type": "number"
          # },
          # "is_regular": {
          #   "type": "integer"
          # },
          "is_rectangular": {
            "type": "integer"
          },
          "floor_polygon": {
            "type": "array",
            "minItems": 3,
            "maxItems": 15,
            "items": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "x": {
                  "type": "number"
                },
                "y": {
                  "type": "number"
                }
              }
              # "anyOf": [
              #   {"required": ["x", "y"]}
              # ]
            }
          }
        },
        "required": ["id", "room_type", "area", "floor_polygon"]
      }
    },
  },
  # "required": ["room_count", "total_area", "room_types", "rooms"]
  "required": ["room_count", "rooms"]
}
