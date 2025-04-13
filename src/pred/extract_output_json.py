import json
from src.utils import repair_json

# def extract_output_json(input_str: str):
#     try:
#         output_index = input_str.find("Output:")
#         if output_index == -1:
#             output_index = input_str.find("assistant") + len("assistant")
#         else:
#            output_index + len("Output:\n")
#         if output_index == -1:
#             return None

#         output_str = input_str[output_index:]
#         output_dict = json.loads(output_str)

#         return output_dict
#     except json.JSONDecodeError:
#         try:
#             json_repaired = repair_json(output_str, return_objects=True)
#             if json_repaired != "":
#                 return json_repaired
#             else:
#                 return {}
#         except Exception:
#             return {}

import json

def extract_output_json(input_str: str):
    """
    Extracts and returns a JSON object from the input string by locating the assistant marker.
    The function searches for the string "assistant" and extracts all text following it.
    It then attempts to parse the extracted substring as JSON. If parsing fails, it calls repair_json.
    If the resulting JSON is wrapped in an "output" key, it returns that value.
    
    :param input_str: The string containing the assistant output.
    :return: The parsed JSON object (or the value of the "output" key if present), 
             or an empty dict if parsing fails.
    """
    assistant_marker = "assistant"
    marker_index = input_str.find(assistant_marker)
    
    if marker_index != -1:
        output_str = input_str[marker_index + len(assistant_marker):].lstrip()
    else:
        output_str = input_str

    try:
        parsed_json = json.loads(output_str)
    except json.JSONDecodeError:
        try:
            json_repaired = repair_json(output_str, return_objects=True)
            parsed_json = json_repaired if json_repaired != "" else {}
        except Exception:
            parsed_json = {}
    
    if isinstance(parsed_json, dict) and "output" in parsed_json:
        return parsed_json["output"]
    
    if isinstance(parsed_json, dict) and "floor_plan" in parsed_json:
        return parsed_json["floor_plan"]
    
    return parsed_json
