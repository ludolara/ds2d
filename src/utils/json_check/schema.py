schema = {
  "type": "object",
  "properties": {
    "room_count": {
      "type": "integer"
    },
    "spaces": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "id": {
            "type": "string",
          },
          "room_type": {
            "type": "string"
          },
          "area": {
            "type": "number"
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
            }
          }
        },
        "required": ["id", "room_type", "area", "floor_polygon"]
      }
    },
  },
  "required": ["room_count", "spaces"]
}
