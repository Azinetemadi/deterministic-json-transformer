#!/usr/bin/env python3
"""Deterministic JSON transformer with basic schema validation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any, Dict, List, Tuple


JsonType = Any


def load_json(path: str) -> JsonType:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_pointer(pointer: str) -> List[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"Invalid JSON pointer: {pointer}")
    parts = pointer.split("/")[1:]
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def get_by_pointer(doc: JsonType, pointer: str) -> Tuple[bool, JsonType]:
    if pointer == "":
        return True, doc
    current = doc
    for part in decode_pointer(pointer):
        if isinstance(current, list):
            if not part.isdigit():
                return False, None
            idx = int(part)
            if idx >= len(current):
                return False, None
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        else:
            return False, None
    return True, current


def ensure_list_length(target: List[JsonType], idx: int) -> None:
    while len(target) <= idx:
        target.append(None)


def set_by_pointer(doc: JsonType, pointer: str, value: JsonType) -> JsonType:
    if pointer == "":
        return value
    current = doc
    parts = decode_pointer(pointer)
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        next_part = parts[i + 1] if not is_last else None

        if isinstance(current, list):
            if not part.isdigit():
                raise ValueError(f"Array index expected at '{part}'")
            idx = int(part)
            ensure_list_length(current, idx)
            if is_last:
                current[idx] = value
                return doc
            if current[idx] is None:
                current[idx] = [] if (next_part and next_part.isdigit()) else {}
            current = current[idx]
            continue

        if isinstance(current, dict):
            if is_last:
                current[part] = value
                return doc
            if part not in current or current[part] is None:
                current[part] = [] if (next_part and next_part.isdigit()) else {}
            current = current[part]
            continue

        raise ValueError(f"Cannot set pointer through non-container at '{part}'")
    return doc


def delete_by_pointer(doc: JsonType, pointer: str) -> None:
    if pointer == "":
        raise ValueError("Refusing to delete root document")
    current = doc
    parts = decode_pointer(pointer)
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        if isinstance(current, list):
            if not part.isdigit():
                return
            idx = int(part)
            if idx >= len(current):
                return
            if is_last:
                current[idx] = None
                return
            current = current[idx]
            continue
        if isinstance(current, dict):
            if part not in current:
                return
            if is_last:
                current.pop(part, None)
                return
            current = current.get(part)
            continue
        return


def build_from_includes(doc: JsonType, includes: List[str]) -> JsonType:
    output: JsonType = {} if isinstance(doc, dict) else [] if isinstance(doc, list) else None
    for pointer in includes:
        found, value = get_by_pointer(doc, pointer)
        if not found:
            continue
        output = set_by_pointer(output, pointer, value)
    return output


def apply_filters(doc: JsonType, filters: Dict[str, Any]) -> JsonType:
    includes = filters.get("include", []) or []
    excludes = filters.get("exclude", []) or []

    if includes:
        output = build_from_includes(doc, includes)
    else:
        output = copy.deepcopy(doc)

    for pointer in excludes:
        delete_by_pointer(output, pointer)
    return output


def type_matches(value: JsonType, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def validate(instance: JsonType, schema: Dict[str, Any], path: str = "") -> List[str]:
    errors: List[str] = []
    expected_type = schema.get("type")
    if expected_type:
        if isinstance(expected_type, list):
            if not any(type_matches(instance, t) for t in expected_type):
                errors.append(f"{path or '/'}: expected {expected_type}")
                return errors
        else:
            if not type_matches(instance, expected_type):
                errors.append(f"{path or '/'}: expected {expected_type}")
                return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path or '/'}: value not in enum")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                errors.append(f"{path or '/'}: missing required '{key}'")

        properties = schema.get("properties", {})
        for key, value in instance.items():
            next_path = f"{path}/{key}" if path else f"/{key}"
            if key in properties:
                errors.extend(validate(value, properties[key], next_path))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{next_path}: additional properties not allowed")

    if isinstance(instance, list) and "items" in schema:
        for idx, item in enumerate(instance):
            next_path = f"{path}/{idx}" if path else f"/{idx}"
            errors.extend(validate(item, schema["items"], next_path))

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic JSON transformer")
    parser.add_argument("--input", default="-", help="Input JSON file or '-' for stdin")
    parser.add_argument("--schema", default="schema.json", help="Schema file")
    parser.add_argument("--filter", dest="filter_path", default="filter.json", help="Filter file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_json(args.input)

    try:
        schema = load_json(args.schema)
    except FileNotFoundError:
        schema = None

    if schema is not None:
        errors = validate(payload, schema)
        if errors:
            for err in errors:
                print(f"Validation error: {err}", file=sys.stderr)
            return 1

    try:
        filters = load_json(args.filter_path)
    except FileNotFoundError:
        filters = {}

    output = apply_filters(payload, filters)
    json.dump(output, sys.stdout, sort_keys=True, indent=2 if args.pretty else None, ensure_ascii=True)
    if args.pretty:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
