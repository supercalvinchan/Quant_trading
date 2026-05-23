from __future__ import annotations

from datetime import date
import re
from typing import Any


def validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    source_name: str,
    allow_unknown_fields: bool = False,
) -> Any:
    if not isinstance(schema, dict):
        raise ValueError(f"{source_name} schema must be an object")
    return _validate_node(
        value=value,
        schema=schema,
        source_name=source_name,
        allow_unknown_fields=allow_unknown_fields,
    )


def _validate_node(
    *,
    value: Any,
    schema: dict[str, Any],
    source_name: str,
    allow_unknown_fields: bool,
) -> Any:
    allowed_types = _normalize_types(schema.get("type"))
    if allowed_types and not any(_matches_type(value, item) for item in allowed_types):
        raise ValueError(f"{source_name} must be {_format_expected_types(allowed_types)}, got {_type_name(value)}")

    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list):
            raise ValueError(f"{source_name} schema enum must be an array")
        if value not in enum_values:
            raise ValueError(f"{source_name} must be one of {enum_values}")

    normalized = _normalize_scalar(value, allowed_types)
    if isinstance(normalized, str):
        _validate_string_constraints(source_name, normalized, schema)
    if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
        _validate_number_constraints(source_name, normalized, schema)
    if isinstance(normalized, list):
        return _validate_array(
            value=normalized,
            schema=schema,
            source_name=source_name,
            allow_unknown_fields=allow_unknown_fields,
        )
    if isinstance(normalized, dict):
        return _validate_object(
            value=normalized,
            schema=schema,
            source_name=source_name,
            allow_unknown_fields=allow_unknown_fields,
        )
    return normalized


def _validate_object(
    *,
    value: dict[str, Any],
    schema: dict[str, Any],
    source_name: str,
    allow_unknown_fields: bool,
) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        raise ValueError(f"{source_name} schema.properties must be an object")
    required = schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, list):
        raise ValueError(f"{source_name} schema.required must be an array")

    for field in required:
        if field not in value:
            raise ValueError(f"{source_name}.{field} is required")

    reject_unknown = bool(properties) and not allow_unknown_fields and schema.get("additionalProperties", None) is not True
    if schema.get("additionalProperties", None) is False:
        reject_unknown = True
    if reject_unknown:
        unknown_fields = [key for key in value if key not in properties]
        if unknown_fields:
            raise ValueError(f"{source_name} contains unknown fields: {unknown_fields}")

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key in properties:
            normalized[key] = _validate_node(
                value=raw,
                schema=properties[key],
                source_name=f"{source_name}.{key}",
                allow_unknown_fields=allow_unknown_fields,
            )
        else:
            normalized[key] = raw

    for constraint in schema.get("x_constraints", []) or []:
        _validate_constraint(normalized, constraint, source_name)
    return normalized


def _validate_array(
    *,
    value: list[Any],
    schema: dict[str, Any],
    source_name: str,
    allow_unknown_fields: bool,
) -> list[Any]:
    item_schema = schema.get("items")
    if item_schema is None:
        return list(value)
    if not isinstance(item_schema, dict):
        raise ValueError(f"{source_name} schema.items must be an object")
    return [
        _validate_node(
            value=item,
            schema=item_schema,
            source_name=f"{source_name}[{idx}]",
            allow_unknown_fields=allow_unknown_fields,
        )
        for idx, item in enumerate(value)
    ]


def _validate_constraint(value: dict[str, Any], constraint: dict[str, Any], source_name: str) -> None:
    if not isinstance(constraint, dict):
        raise ValueError(f"{source_name} x_constraints items must be objects")
    ctype = str(constraint.get("type", "")).strip()
    if ctype != "compare":
        raise ValueError(f"{source_name} unsupported x_constraint type: {ctype}")
    left_path = str(constraint.get("left", "")).strip()
    op = str(constraint.get("op", "")).strip()
    right_path = str(constraint.get("right", "")).strip()
    message = str(constraint.get("message", "")).strip() or f"{source_name} compare constraint failed"
    left = _get_path_value(value, left_path)
    right = _get_path_value(value, right_path)
    if left is None or right is None:
        return
    ok = False
    if op == "<=":
        ok = left <= right
    elif op == "<":
        ok = left < right
    elif op == ">=":
        ok = left >= right
    elif op == ">":
        ok = left > right
    elif op == "==":
        ok = left == right
    else:
        raise ValueError(f"{source_name} unsupported compare op: {op}")
    if not ok:
        raise ValueError(message)


def _get_path_value(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in [item for item in path.split(".") if item]:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _validate_string_constraints(source_name: str, value: str, schema: dict[str, Any]) -> None:
    min_length = schema.get("minLength")
    if min_length is not None and len(value) < int(min_length):
        raise ValueError(f"{source_name} length must be >= {int(min_length)}")
    max_length = schema.get("maxLength")
    if max_length is not None and len(value) > int(max_length):
        raise ValueError(f"{source_name} length must be <= {int(max_length)}")
    pattern = schema.get("pattern")
    if pattern is not None and re.fullmatch(str(pattern), value) is None:
        raise ValueError(f"{source_name} must match pattern {pattern}")
    if schema.get("format") == "date":
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{source_name} must be ISO date YYYY-MM-DD") from exc


def _validate_number_constraints(source_name: str, value: int | float, schema: dict[str, Any]) -> None:
    minimum = schema.get("minimum")
    if minimum is not None and float(value) < float(minimum):
        raise ValueError(f"{source_name} must be >= {minimum}")
    maximum = schema.get("maximum")
    if maximum is not None and float(value) > float(maximum):
        raise ValueError(f"{source_name} must be <= {maximum}")


def _normalize_scalar(value: Any, allowed_types: list[str]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and "number" in allowed_types and "integer" not in allowed_types:
        return float(value)
    return value


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return False


def _normalize_types(schema_type: Any) -> list[str]:
    if schema_type is None:
        return []
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list) and all(isinstance(item, str) for item in schema_type):
        return list(schema_type)
    raise ValueError(f"unsupported schema type: {schema_type!r}")


def _format_expected_types(items: list[str]) -> str:
    return " | ".join(items)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__
