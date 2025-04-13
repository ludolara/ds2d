import re
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from .schema import schema

def clean_validation_expection(error_text: str) -> str:
    pattern = r"\n\nFailed validating.*?(?=\n\s*On instance)"
    cleaned_text = re.sub(pattern, "", error_text, flags=re.DOTALL | re.MULTILINE)
    return cleaned_text

def is_valid_json(json_data, strict=False):
    # _schema = schema if strict else strict_schema
    try:
        validate(json_data, schema)
        return True
    except ValidationError as e:
        return False

def is_valid_json_feedback(json_data, strict=False):
    # _schema = strict_schema if strict else feedback_schema
    try:
        validate(json_data, schema)
        return True, ""
    except ValidationError as e:
        return False, clean_validation_expection(str(e))
