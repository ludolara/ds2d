schema = {
  "type": "object",
  # "additionalProperties": False,
  "properties": {
    "room_count": {
      "type": "integer"
    },
    "total_area": {
      "type": "number"
    },
    "room_types": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "rooms": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "id": {
            "type": "string"
          },
          "room_type": {
            "type": "string"
          },
          "area": {
            "type": "number"
          },
          "width": {
            "type": "number"
          },
          "height": {
            "type": "number"
          },
          "is_regular": {
            "type": "integer"
          },
          "floor_polygon": {
            "type": "array",
            "minItems": 3,
            "maxItems": 30,
            "items": {
              "type": "object",
              "additionalProperties": False,
              "properties": {
                "x": {
                  "type": "number"
                },
                "y": {
                  "type": "number"
                },
                "z": {
                  "type": "number"
                }
              },
              "anyOf": [
                {"required": ["x", "z"]},
                {"required": ["x", "y"]}
              ]
            }
          }
        },
        "required": ["id", "room_type", "area", "width", "height", "floor_polygon"]
      }
    },
  },
  "required": ["room_count", "total_area", "room_types", "rooms"]
}
